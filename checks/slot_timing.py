#!/usr/bin/env python3
"""Slot save/restore timing on the real disk path (/mnt/hdd/slot_saves, NTFS/FUSE HDD).

Purpose: verify that after the v0.4.9 upgrade a saved slot still restores without
full prompt reprocessing, and measure how long the disk round-trip actually takes.

Per build:
  1. start server with the profile settings (cache-ram 0, slot-save-path on /mnt/hdd)
  2. prefill a long prompt (~N tokens), record prefill time
  3. save slot -> wall time, file size(s)
  4. restart the server (cold), restore the slot -> wall time
  5. re-send the same prompt -> TTFT + cached tokens (must be ~prompt length)

Usage: slot_timing.py <bin_dir> <label> [--tokens N]
"""
import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.request

MODEL = "/home/cyrsia/adisk/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q3_K_XL-nomtp.gguf"
TEMPLATE = "/home/cyrsia/adisk/models/chat_template.jinja"
SLOTDIR = "/mnt/hdd/slot_saves"
PORT = 10000
CORPUS = "/tmp/opencode/corpus.txt"


def make_prompt(approx_tokens):
    # ~4 chars per token for English text; cap by available corpus
    with open(CORPUS, "r", errors="replace") as f:
        text = f.read()
    chars = approx_tokens * 4
    text = text[:chars]
    return "Прочитай текст ниже и ответь одним словом: 'готово'.\n\n" + text


def server_cmd(bin_dir, log):
    return [
        "stdbuf", "-oL", f"{bin_dir}/llama-server",
        "-m", MODEL,
        "-ngl", "99", "--flash-attn", "on", "--port", str(PORT), "-np", "1",
        "--cache-ram", "0", "--no-cache-idle-slots",
        "--slot-save-path", SLOTDIR,
        "--jinja", "--chat-template-file", TEMPLATE,
        "--temp", "0.2", "--reasoning", "off",
        "--ctx-size", "32000", "-ctk", "kvarn6", "-ctv", "kvarn6",
        "-b", "48", "-ub", "48",
    ]


def start(bin_dir, log_path):
    log = open(log_path, "wb")
    proc = subprocess.Popen(server_cmd(bin_dir, log_path), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}, see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return proc, log
        except Exception:
            time.sleep(2)
    raise RuntimeError("server not ready")


def stop(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    log.close()


def post(path, payload=None, timeout=3600):
    data = json.dumps(payload).encode() if payload is not None else b"{}"
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(prompt, max_tokens=8):
    t0 = time.time()
    resp = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "cache_prompt": True,
        "id_slot": 0,
    })
    wall = time.time() - t0
    timings = resp.get("timings", {})
    usage = resp.get("usage", {})
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "prompt_ms": round(timings.get("prompt_ms") or 0, 1),
        "prompt_per_second": round(timings.get("prompt_per_second") or 0, 1),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "predicted_per_second": timings.get("predicted_per_second"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin_dir")
    ap.add_argument("label")
    ap.add_argument("--tokens", type=int, default=12000)
    args = ap.parse_args()

    os.makedirs(SLOTDIR, exist_ok=True)
    prompt = make_prompt(args.tokens)
    name = f"upgrade-{args.label}.bin"
    path = os.path.join(SLOTDIR, name)
    log1 = f"/tmp/opencode/slot-{args.label}-1.log"
    log2 = f"/tmp/opencode/slot-{args.label}-2.log"
    out = {"label": args.label, "bin": args.bin_dir, "target_tokens": args.tokens}
    dbg = {}

    proc, log = start(args.bin_dir, log1)
    try:
        t0 = time.time()
        warm = chat("Скажи 'ок'.", max_tokens=4)
        out["warm"] = warm
        prefill = chat(prompt)
        out["prefill"] = prefill
        out["prefill"]["wall_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        save = post("/slots/0?action=save", {"filename": name})
        out["save_s"] = round(time.time() - t0, 2)
        out["save_resp"] = save
        files = {}
        for suffix in ("", ".ckpt"):
            p = path + suffix
            if os.path.exists(p):
                files[os.path.basename(p)] = os.path.getsize(p)
        out["files"] = files
        out["file_mb"] = {k: round(v / 1024 / 1024, 1) for k, v in files.items()}
    finally:
        stop(proc, log)

    # cold restore: new process, model loaded, then restore from disk
    proc, log = start(args.bin_dir, log2)
    try:
        t0 = time.time()
        restore = post("/slots/0?action=restore", {"filename": name})
        out["restore_s"] = round(time.time() - t0, 2)
        out["restore_resp"] = restore
        # strictest case: identical prompt must be fully served from the restored cache
        out["post_restore_same"] = chat(prompt)
        # normal case: conversation continues by a few tokens, prefix must be reused
        out["post_restore_extend"] = chat(prompt + "\n\nОтветь одним словом: продолжение.")
    finally:
        stop(proc, log)

    with open(log2, "r", errors="replace") as f:
        logtext = f.read()
    out["log2"] = {
        "slot_lines": [l for l in logtext.splitlines()
                       if re.search(r"slot|checkpoint|cache", l, re.I)][-25:],
        "errors": [l for l in logtext.splitlines()
                   if re.search(r"error|failed|invalid|incompatible", l, re.I)][:15],
    }

    # verdict: restored cache must avoid a full prefill
    pr = out["prefill"]
    same = out["post_restore_same"]
    ext = out["post_restore_extend"]
    n = pr.get("prompt_tokens") or 0
    def reused(rec, expected):
        c = rec.get("cached_tokens") or 0
        return c >= max(1, int(expected * 0.95))
    out["verdict"] = {
        "prefill_prompt_tokens": n,
        "prefill_ms": pr.get("prompt_ms"),
        "restore_same_cached": same.get("cached_tokens"),
        "restore_same_ms": same.get("prompt_ms"),
        "restore_extend_cached": ext.get("cached_tokens"),
        "restore_extend_ms": ext.get("prompt_ms"),
        "no_full_reprocess_same": reused(same, n),
        "no_full_reprocess_extend": reused(ext, n),
    }

    with open(f"/tmp/opencode/slot-{args.label}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
