# Dynamic batch and context growth

Both features target the same goal on a fixed VRAM budget: process a long prompt
as fast as the memory allows, while keeping the whole context available for
generation. They are independent -- either can be used alone.

## Memory model (Qwen3.8-27B, KVarN6, RTX 5060 Ti 16 GB, `-np 1`)

| item | size | scales with |
|---|---|---|
| model weights (IQ4_XS / Q3_K_XL) | 12.2 / 11.4 GiB | fixed |
| KVarN records (K+V) | 25.5 KiB/token (kvarn6) | `n_ctx` (preallocated) |
| GDN recurrent state | 150 MiB | fixed |
| KVarN stage + exact tail (1024) | 92 MiB | fixed |
| compute buffers | ~0.31 + 2·n_ctx/2^20 MiB per ubatch token | `n_ubatch`, `n_ctx` |
| KVarN prefill window | `min(n_kv, --kvarn-window-chunk) · 4 KiB` | current KV fill |
| driver + CUDA VMM pool | ~200 MiB | fixed |

Rules that came out of the measurements:

* freezing the ubatch frees at most ~250 MiB, which is the same order as the
  prefill window at depth -- so batching alone is not enough at 90k+;
* ~380 MiB of true free VRAM must remain after the compute buffer, otherwise the
  tail FlashAttention kernel aborts (`ggml_cuda_pool_vmm::alloc`). 98k/512,
  94k/768 and 92k/768 fail this way; 98k/384, 94k/512 and 88k/768 pass;
* the KV cache is allocated for the whole capacity: there is no "gradual" KV
  allocation, only reallocation steps.

## Tier 0 -- logical batch (always on, free)

With `-np 1` the physical batch is `min(-b, n_ubatch)`. Lowering the logical
batch per iteration costs nothing and needs no scheduler work. This is what the
controller uses for fine-grained adjustments.

## Tier 1 -- ubatch cap (re-reserves the scheduler)

`llama_set_n_ubatch(ctx, n)` clamps `n` to the value captured at context
creation, updates `cparams.n_ubatch` and marks the scheduler for re-reserve on
the next `llama_decode()`. The KV cache is untouched; only the compute buffers
change size. Changes are applied between decodes, never inside one.

Inside a prompt the cap only steps **down**; it returns to `-ub` for the next
task. `--dyn-batch-step` (default 16) is the deadband, `--dyn-batch-max-drop`
limits how far one step may move (0 = jump straight to the target).

## Choosing the ubatch

Two modes for `--dyn-batch-curve`:

* `auto` -- size the ubatch from free VRAM at the task start:
  `ub = (free + compute(cur) - guard) / compute_per_ub`, capped by `-ub`;
* explicit `kv:ub,kv:ub,...` -- a planned (depth, ubatch) curve, interpolated
  between points. Use this when the plain "as much as fits" choice is not the
  fastest one (a large ubatch at deep context can be slower than a medium one,
  and a large start context forces a small ubatch everywhere).

`--dyn-batch-guard-mb` (default 300, profiles use 32) is the VRAM kept free;
`--dyn-batch-ub-min` is the floor.

## Context growth and shrink

`--ctx-grow N` starts the context with `N` cells instead of `--ctx-size` and
lets the server grow it when a prompt no longer fits:

1. captures the live sequence state (attention + recurrent) into host memory;
2. frees the old cache, allocates the larger one;
3. restores the state and re-reserves the scheduler;
4. re-sizes the ubatch from the new free VRAM, or from the curve when one is set.

Growth is monotonic for the life of the server and preserves the cached
sequences, so later turns do not pay for it again. The target is the prompt
length plus ~3% (never `--ctx-size` straight away): a cache sized for the prompt
keeps the ubatch large, while a jump to the maximum starves it.

`POST /slots/<id>?action=restore` grows the context first when the saved slot
holds more tokens than the current capacity, instead of failing with
"no available space in KV cache".

**Shrinking** is implemented (`llama_resize_ctx(..., allow_shrink)`) and runs at
prompt start when the request cannot reuse a cached prefix (common prefix <
1024 tokens): the sequence memory is wiped and the capacity is rebuilt without a
state copy. Measured 95488 -> 10752 cells in 29-45 ms with free VRAM recovering
from 39 to 2337 MiB. It is behind `LLAMA_CTX_RESIZE=1` because two problems
remain:

* after a shrink, the ubatch planner keeps the small cap in that path (the 8k
  prompt right after a 90k one ran at ~422 t/s instead of ~800);
* raising the cap immediately after a resize hits an assert in
  `llama_kv_cache::set_input_kq_mask_tail_mapped` (stale KVarN tail mapping),
  and raising it inside a running prompt hits the same assert because the tail
  arena is sized from the graph ubatch;
* the CUDA VMM pool cannot be trimmed mid-run either: cached CUDA graphs and
  the scheduler still hold pointers into the mapped regions, so unmapping the
  unused tail causes "illegal memory access" in `launch_mul_mat_q`. The pool is
  released when the device context is destroyed; the ubatch planner therefore
  sees less free VRAM after a large allocation than is physically unused.

Without the per-request shrink the default configuration is stable: the KV
capacity is sized once from `--ctx-grow` and grows on demand.

## Tuning process

1. Pick the cache width from the KVarN quality ladder (`docs/beellama-features.md`).
2. Measure the prefill rate surface: `llama-bench -p 512 -d 0,16384,65536,98304
   -ub 48,128,256,384,512,768`.
3. Pick the largest `--ctx-grow` that still loads **and** survives a deep
   prefill with the intended `-ub` (verify with a real request, not just load).
4. Fit `--dyn-batch-curve` so the ubatch steps down before the memory edge:
   keep `ub` maximal where it is cheap, drop it where the measurements show a
   loss, and leave `ub * compute_per_ub + 380 MiB` inside the free VRAM.
5. Validate: same greedy output at fixed vs dynamic ubatch, total prefill time
   and peak VRAM on 8k/32k/64k/92k prompts, and one multi-turn run with a
   checkpoint restore.

Measured on the machine above (IQ4_XS, start 88064, curve 768 down to 448):
prefill 8k 9.2 s, 48k 61.8 s, 64k 94.7 s, 84k 138.3 s, 92k 162.5 s -- 2-5%
faster than the static 98k/384 layout for prompts below 84k and level above it.

## Safety

* never grow the cap inside a running prompt, only shrink;
* keep `ub >= n_rs_seq + 1` with recurrent rollback, and `ub >= image_max_tokens`
  with a vision projector;
* emergency drop by one curve step when free VRAM falls below the guard;
* growth fails closed: if the larger cache cannot be allocated, the old state is
  rebuilt and the request continues at the previous capacity.
