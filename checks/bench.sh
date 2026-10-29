#!/bin/bash
# llama-bench comparison for the v0.4.9 upgrade.
# usage: bench.sh <build_dir_with_bin> <out_prefix>
set -euo pipefail
BIN_DIR="$1"
OUT="$2"
MODEL=/home/cyrsia/adisk/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q3_K_XL-nomtp.gguf
mkdir -p "$(dirname "$OUT")"

echo "### $(basename "$(dirname "$BIN_DIR")") / $(basename "$BIN_DIR")"
"$BIN_DIR/llama-bench" --version 2>&1 | head -2 || true

for UB in 48 128; do
    echo "--- ub=$UB"
    "$BIN_DIR/llama-bench" -m "$MODEL" \
        -p 512,2048 -n 64 -b 128 -ub "$UB" \
        -ctk kvarn6 -ctv kvarn6 -fa 1 -r 2 -o json \
        > "${OUT}-ub${UB}.json" 2> "${OUT}-ub${UB}.log"
    python3 - "${OUT}-ub${UB}.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
for r in rows:
    print(f"  {r['n_prompt']:>5}p {r['n_gen']:>4}g  {r['avg_ts']:9.2f} t/s  +- {r['stddev_ts']:.2f}")
PY
done
