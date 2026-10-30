#!/usr/bin/env python3
"""Smoke checks for the v0.4.9 upgrade.

Cases:
  1. text completion
  2. single image with a big digit -> model must name the digit
  3. interleaved text/image/text/image/text/image, cache_prompt=false
     -> exactly ONE mmproj swap cycle, all images encoded in it, correct count/order
  4. same interleaved layout with a different text prefix (fresh prompt again)
  5. slot save/restore in-process + prompt cache reuse

Each request records the mmproj swap counters from the server log delta.
"""
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = "/home/cyrsia/adisk/llama/building_mmproj/beellama.cpp.3"
BIN = f"{ROOT}/build-v049/bin/llama-server"
MODEL = "/home/cyrsia/adisk/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q3_K_XL-nomtp.gguf"
MMPROJ = "/home/cyrsia/adisk/models/Qwen3.8-27B/mmproj-BF16.gguf"
TEMPLATE = "/home/cyrsia/adisk/models/chat_template.jinja"
SLOTDIR = "/tmp/opencode/slots"
LOG = "/tmp/llama_server.log"
PORT = 10000
IMG = {d: f"/tmp/opencode/testimgs/d{d}.png" for d in ("1", "2", "3")}


def server_cmd(extra):
    return [
        "stdbuf", "-oL", BIN,
        "-m", MODEL,
        "-ngl", "99", "--flash-attn", "on", "--port", str(PORT), "-np", "1",
        "--cache-ram", "0", "--no-cache-idle-slots",
        "--slot-save-path", SLOTDIR,
        "--jinja", "--chat-template-file", TEMPLATE,
        "--temp", "0.2", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
        "--presence-penalty", "0.0", "--repeat-penalty", "1.0",
        "--reasoning", "off",
        "--mmproj", MMPROJ,
        "--mmproj-gpu-swap", "--mmproj-gpu-swap-reserve-mb", "2048",
        "--ctx-size", "98000", "-ctk", "kvarn6", "-ctv", "kvarn6",
        "-b", "48", "-ub", "48",
    ] + extra


def wait_ready(proc, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError("server not ready")


def post(path, payload, timeout=3600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img(d):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(IMG[d])}"}}


def txt(t):
    return {"type": "text", "text": t}


def chat(content, max_tokens=64, cache_prompt=True):
    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 12345,
        "cache_prompt": cache_prompt,
        "id_slot": 0,
    }
    t0 = time.time()
    resp = post("/v1/chat/completions", payload)
    wall = time.time() - t0
    msg = resp["choices"][0]["message"]
    usage = resp.get("usage", {})
    timings = resp.get("timings", {})
    return {
        "wall_s": round(wall, 2),
        "content": (msg.get("content") or "").strip()[:500],
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "prompt_ms": round(timings.get("prompt_ms") or 0, 1),
        "predicted_per_second": round(timings.get("predicted_per_second") or 0, 1),
    }


def log_lines():
    with open(LOG, "r", errors="replace") as f:
        return f.readlines()


def swaps():
    lines = log_lines()
    return (sum(1 for l in lines if "swapping mmproj to GPU" in l),
            sum(1 for l in lines if "swapping mmproj back to CPU" in l))


def measure(name, content, max_tokens=64, cache_prompt=True, expect=None):
    s0 = swaps()
    res = chat(content, max_tokens=max_tokens, cache_prompt=cache_prompt)
    time.sleep(0.5)
    s1 = swaps()
    res["swaps_in"] = s1[0] - s0[0]
    res["swaps_out"] = s1[1] - s0[1]
    if expect:
        res["expect"] = expect
        res["match"] = all(str(e).lower() in res["content"].lower() for e in expect)
    return res


def vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True)
        used, total = out.strip().split(",")
        return {"used_mib": int(used), "total_mib": int(total)}
    except Exception:
        return {}


def main():
    extra = [a for a in sys.argv[1:] if not a.endswith(".json")]
    out_path = next((a for a in sys.argv[1:] if a.endswith(".json")), "/tmp/opencode/smoke.json")
    os.makedirs(SLOTDIR, exist_ok=True)
    log_f = open(LOG, "wb")
    proc = subprocess.Popen(server_cmd(extra), stdout=log_f, stderr=subprocess.STDOUT,
                            start_new_session=True)
    results = {"args_extra": extra, "checks": {}}
    try:
        t0 = time.time()
        wait_ready(proc)
        results["load_s"] = round(time.time() - t0, 1)
        results["vram_after_load"] = vram()

        results["checks"]["text"] = measure(
            "text", "Одним предложением: что такое KV-кэш?", max_tokens=80)

        results["checks"]["image_digit"] = measure(
            "image_digit", [img("2"), txt("Какая цифра на картинке? Ответь только цифрой.")],
            max_tokens=8, expect=["2"])

        # text -> image -> text -> image -> text -> image, fresh (not cached) prompt
        interleaved = [
            txt("Первая картинка:"), img("1"),
            txt("Вторая картинка:"), img("2"),
            txt("Третья картинка:"), img("3"),
            txt("Сколько всего картинок в сообщении выше? Перечисли цифры на них по порядку, "
                "кратко: число, затем цифры."),
        ]
        results["checks"]["interleaved"] = measure(
            "interleaved", interleaved, max_tokens=64, cache_prompt=False, expect=["3", "1"])

        # same layout, different text so the prompt is definitely a fresh one
        interleaved2 = [
            txt("Набор A:"), img("3"),
            txt("Набор B:"), img("1"),
            txt("Набор C:"), img("2"),
            txt("Сколько изображений? Перечисли их цифры по порядку: число, затем цифры."),
        ]
        results["checks"]["interleaved2"] = measure(
            "interleaved2", interleaved2, max_tokens=64, cache_prompt=False, expect=["3"])

        # slot save / restore in-process
        prompt = "Запомни слово: КВАРЦ. Ответь 'ок'."
        first = measure("slot_first", prompt, max_tokens=8)
        save = post("/slots/0?action=save", {"filename": "smoke-slot.bin"})
        second = measure("slot_second", prompt, max_tokens=8)
        restore = post("/slots/0?action=restore", {"filename": "smoke-slot.bin"})
        third = measure("slot_third", prompt, max_tokens=8)
        results["checks"]["slot"] = {
            "first": first, "second": second, "third": third,
            "save": save, "restore": restore,
            "state_bytes": os.path.getsize(os.path.join(SLOTDIR, "smoke-slot.bin")),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), 9)
        log_f.close()

    with open(LOG, "r", errors="replace") as f:
        log = f.read()
    results["log"] = {
        "swap_in": log.count("swapping mmproj to GPU"),
        "swap_out": log.count("swapping mmproj back to CPU"),
        "errors": [l for l in log.splitlines()
                   if re.search(r"error|failed|assert|out of memory|OOM", l, re.I)][:25],
        "tail_policy": [l for l in log.splitlines() if "tail" in l.lower()][:10],
        "encode_decode": [l for l in log.splitlines() if "image encoding" in l or "image decoding" in l][:20],
        "gpu_swap_stats": [l for l in log.splitlines() if "gpu_swap:" in l][-6:],
    }
    results["verdict"] = {
        "interleaved_one_swap": (
            results["checks"]["interleaved"]["swaps_in"] == 1
            and results["checks"]["interleaved"]["swaps_out"] == 1),
        "interleaved2_one_swap": (
            results["checks"]["interleaved2"]["swaps_in"] == 1
            and results["checks"]["interleaved2"]["swaps_out"] == 1),
        "digit_match": results["checks"]["image_digit"].get("match"),
        "interleaved_count_match": results["checks"]["interleaved"].get("match"),
        "slot_restore_swaps_zero": results["checks"]["slot"]["third"]["swaps_in"] == 0,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
