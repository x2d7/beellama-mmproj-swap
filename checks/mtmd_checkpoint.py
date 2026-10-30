#!/usr/bin/env python3
"""mtmd + on-disk checkpoints: after a cold restore, an image conversation must
continue from the saved prefix without re-encoding images and without a full
prompt reprocess.

Flow:
  1. server A: fresh (cache_prompt=false) conversation text->image->text->image,
     remember messages and the assistant reply; save slot; stop A
  2. server B (cold, model reloaded): restore the slot; send the same conversation
     plus the assistant reply and a new user question
  3. assert: cached_tokens ~ first prompt, no mmproj swap, small TTFT
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from smoke import (BIN, MODEL, MMPROJ, TEMPLATE, SLOTDIR, PORT, img, txt,
                   server_cmd, wait_ready, post)


def swaps(path=None):
    """Count mmproj swap markers in this script's own server log."""
    path = path or LOG_B
    if not os.path.exists(path):
        return (0, 0)
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    return (sum(1 for l in lines if "swapping mmproj to GPU" in l),
            sum(1 for l in lines if "swapping mmproj back to CPU" in l))

LOG_A = "/tmp/llama_server.log"
LOG_B = "/tmp/llama_server.log"
NAME = "mtmd-ckpt.bin"


EXTRA = ["--ctx-size", "24000", "--rs-rollback", "8"]


def start(log_path):
    log = open(log_path, "wb")
    proc = subprocess.Popen(server_cmd(EXTRA), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    wait_ready(proc)
    return proc, log


def stop(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    log.close()


def chat(messages, max_tokens=48, cache_prompt=True):
    t0 = time.time()
    resp = post("/v1/chat/completions", {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "seed": 12345,
        "cache_prompt": cache_prompt,
        "id_slot": 0,
    })
    wall = time.time() - t0
    msg = resp["choices"][0]["message"]
    usage = resp.get("usage", {})
    timings = resp.get("timings", {})
    return {
        "wall_s": round(wall, 2),
        "reply": (msg.get("content") or "").strip(),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "prompt_ms": round(timings.get("prompt_ms") or 0, 1),
    }


def main():
    os.makedirs(SLOTDIR, exist_ok=True)
    messages = [{
        "role": "user",
        "content": [
            txt("Первое изображение:"), img("1"),
            txt("Второе изображение:"), img("2"),
            txt("Запомни, что на них нарисовано. Ответь одним словом."),
        ],
    }]
    out = {}

    proc, log = start(LOG_A)
    try:
        s0 = swaps()
        first = chat(messages, max_tokens=32, cache_prompt=False)
        time.sleep(0.5)
        s1 = swaps()
        first["swaps_in"] = s1[0] - s0[0]
        out["first"] = first
        # realistic multi-turn: add one more user turn so the server creates a
        # checkpoint at that structured message boundary
        turn2 = messages + [
            {"role": "assistant", "content": first["reply"] or "1 и 2"},
            {"role": "user", "content": "Коротко подтверди: сколько было картинок?"},
        ]
        second = chat(turn2, max_tokens=24, cache_prompt=True)
        out["second_turn"] = second
        messages = turn2 + [{"role": "assistant", "content": second["reply"] or "две"}]
        out["save"] = post("/slots/0?action=save", {"filename": NAME})
        out["size_bytes"] = os.path.getsize(os.path.join(SLOTDIR, NAME))
    finally:
        stop(proc, log)

    # cold restore
    proc, log = start(LOG_B)
    try:
        t0 = time.time()
        out["restore"] = post("/slots/0?action=restore", {"filename": NAME})
        out["restore_s"] = round(time.time() - t0, 2)
        s0 = swaps()
        follow = chat(messages + [
            {"role": "user", "content": "Какие цифры были на картинках? Ответь числом и цифрами."},
        ], max_tokens=48, cache_prompt=True)
        time.sleep(0.5)
        s1 = swaps()
        follow["swaps_in"] = s1[0] - s0[0]
        follow["swaps_out"] = s1[1] - s0[1]
        out["follow"] = follow
        out["cached_ratio"] = round(min(1.0, (follow["cached_tokens"] or 0) /
                                    max(1, first["prompt_tokens"] or 1)), 3)
    finally:
        stop(proc, log)

    with open(LOG_B, "r", errors="replace") as f:
        logtext = f.read()
    out["log"] = {
        "slot_lines": [l for l in logtext.splitlines()
                       if "checkpoint" in l.lower() or "restored" in l.lower()][-20:],
        "errors": [l for l in logtext.splitlines()
                   if "error" in l.lower() or "failed" in l.lower()][:10],
    }
    out["verdict"] = {
        "restored_without_full_reprocess": (out["cached_ratio"] >= 0.95),
        "no_image_reencode_after_restore": follow["swaps_in"] == 0,
        "ttft_ms": follow["prompt_ms"],
    }
    with open("/tmp/opencode/mtmd-ckpt.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
