#include "server-gpu-swap.h"

#include "llama.h"
#include "ggml.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstring>
#include <chrono>

//
// mmproj GPU-swap manager (KV-cache variant)
//
// The model weights always stay on the GPU. During image encoding we instead
// offload the KV cache (context) buffers to host RAM -- freeing VRAM so the
// vision projector (mmproj) fits on the device -- then restore the KV cache
// once encoding is done. The KV cache is a separate set of backend buffers
// (llama_kv_cache::ctxs_bufs), so it can be freed/relallocated independently
// of the model weights.
//

void gpu_swap_log_vram(const char * tag, struct llama_context * lctx) {
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_cuda_get_device_memory(0, &free_vram, &total_vram);
    size_t used_vram = total_vram - free_vram;

    // scheduler compute buffers per backend
    char sched_detail[256] = {0};
    size_t sched_total = 0;
    int n_bufts = 0;
    if (lctx) {
        ggml_backend_sched_t sched = llama_context_get_sched(lctx);
        if (sched) {
            int n_backends = ggml_backend_sched_get_n_backends(sched);
            n_bufts = n_backends;
            int pos = 0;
            for (int i = 0; i < n_backends && pos < (int)sizeof(sched_detail) - 1; i++) {
                ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
                const char * name = ggml_backend_name(backend);
                size_t buf_size = ggml_backend_sched_get_buffer_size(sched, backend);
                sched_total += buf_size;
                int n = snprintf(sched_detail + pos, sizeof(sched_detail) - pos,
                        "%s%s=%zu", i > 0 ? " " : "", name, buf_size / (1024 * 1024));
                if (n > 0) pos += n;
            }
        }
    }

    fprintf(stderr, "[gpu_swap] %s: VRAM used=%zu MiB free=%zu MiB sched=%s sum=%zu MiB backends=%d\n",
            tag,
            used_vram / (1024 * 1024),
            free_vram / (1024 * 1024),
            sched_detail,
            sched_total / (1024 * 1024),
            n_bufts);
}

void gpu_swap_manager::reset() {
    std::lock_guard<std::mutex> lock(mtx);
    state = gpu_swap_state::MODEL_ON_GPU;
    enabled = false;
    swap_stats = {};
}

gpu_swap_state gpu_swap_manager::get_state() const {
    return state;
}

void gpu_swap_manager::print_stats() const {
    fprintf(stderr,
        "gpu_swap: kv_to_cpu=%d mmproj_to_gpu=%d mmproj_to_cpu=%d kv_to_gpu=%d "
        "(kv_to_cpu %.1f ms, mmproj_to_gpu %.1f ms, mmproj_to_cpu %.1f ms, kv_to_gpu %.1f ms)\n",
        swap_stats.n_swap_kv_to_cpu, swap_stats.n_swap_mmproj_to_gpu,
        swap_stats.n_swap_mmproj_to_cpu, swap_stats.n_swap_kv_to_gpu,
        (double)swap_stats.t_swap_kv_to_cpu_us / 1000.0,
        (double)swap_stats.t_swap_mmproj_to_gpu_us / 1000.0,
        (double)swap_stats.t_swap_mmproj_to_cpu_us / 1000.0,
        (double)swap_stats.t_swap_kv_to_gpu_us / 1000.0);
}

bool gpu_swap_manager::swap_to_mmproj_gpu(struct mtmd_context * /*mctx*/,
                                          struct llama_model * model,
                                          struct llama_context * lctx) {
    std::lock_guard<std::mutex> lock(mtx);

    if (state == gpu_swap_state::ERROR) {
        fprintf(stderr, "%s: swap manager is in ERROR state, refusing operation\n", __func__);
        return false;
    }
    if (state != gpu_swap_state::MODEL_ON_GPU) {
        fprintf(stderr, "%s: wrong state (expected MODEL_ON_GPU)\n", __func__);
        return false;
    }

    gpu_swap_log_vram("before_offload", lctx);

    state = gpu_swap_state::TRANSITIONING;

    const int64_t t_start = ggml_time_us();

    bool ok = llama_kv_cache_offload_to_cpu(lctx);
    if (!ok) {
        fprintf(stderr, "%s: failed to offload KV cache to CPU, entering ERROR state\n", __func__);
        state = gpu_swap_state::ERROR;
        return false;
    }

    // evict the swappable weight slice (if configured) to free additional VRAM
    if (model) {
        const size_t swap_size = llama_model_swap_weights_size(model);
        if (swap_size > 0) {
            const int64_t t_swap = ggml_time_us();
            if (!llama_model_swap_weights_offload(model)) {
                fprintf(stderr, "%s: failed to offload swappable weights to CPU, entering ERROR state\n", __func__);
                state = gpu_swap_state::ERROR;
                return false;
            }
            swap_stats.n_swap_mmproj_to_gpu++;
            swap_stats.t_swap_mmproj_to_gpu_us += (ggml_time_us() - t_swap);
            fprintf(stderr, "gpu_swap: evicted %.2f MiB of swappable weights (%.0f ms)\n",
                    (double) swap_size / 1024.0 / 1024.0, (double) (ggml_time_us() - t_swap) / 1000.0);
        }
    }

    // The KV tensors moved to a CPU buffer; reset the scheduler so it rebuilds
    // its graphs against the new tensor backends on the next forward pass.
    if (lctx) {
        ggml_backend_sched_reset(llama_context_get_sched(lctx));
        // Free the main-model forward compute buffer (VRAM) during image
        // encoding: it is not needed for the CLIP forward, and
        // mtmd_helper_decode_image_chunk() -> llama_decode() will re-reserve it
        // (sized only for the small image-token ubatch) once encoding writes
        // the embeddings. This frees several hundred MiB so the vision compute
        // buffer fits alongside the projector weights.
        ggml_backend_sched_free_buffers(llama_context_get_sched(lctx));
    }

    gpu_swap_log_vram("after_offload", lctx);

    state = gpu_swap_state::MMPROJ_ON_GPU;

    const int64_t t_end = ggml_time_us();
    swap_stats.n_swap_kv_to_cpu++;
    swap_stats.n_swap_mmproj_to_gpu++;
    swap_stats.t_swap_kv_to_cpu_us   += (t_end - t_start);
    swap_stats.t_swap_mmproj_to_gpu_us += (t_end - t_start);

    return true;
}

bool gpu_swap_manager::swap_to_model_gpu(struct mtmd_context * /*mctx*/,
                                         struct llama_model * model,
                                         struct llama_context * lctx) {
    std::lock_guard<std::mutex> lock(mtx);

    if (state == gpu_swap_state::ERROR) {
        fprintf(stderr, "%s: swap manager is in ERROR state, refusing operation\n", __func__);
        return false;
    }
    if (state != gpu_swap_state::MMPROJ_ON_GPU) {
        fprintf(stderr, "%s: wrong state (expected MMPROJ_ON_GPU)\n", __func__);
        return false;
    }

    gpu_swap_log_vram("before_restore", lctx);

    state = gpu_swap_state::TRANSITIONING;

    const int64_t t_start = ggml_time_us();

    bool ok = llama_kv_cache_restore_to_gpu(lctx);
    if (!ok) {
        fprintf(stderr, "%s: failed to restore KV cache to GPU, entering ERROR state\n", __func__);
        state = gpu_swap_state::ERROR;
        return false;
    }

    // reload the swappable weight slice before any decode touches the model
    if (model) {
        const size_t swap_size = llama_model_swap_weights_size(model);
        if (swap_size > 0) {
            const int64_t t_swap = ggml_time_us();
            if (!llama_model_swap_weights_restore(model)) {
                fprintf(stderr, "%s: failed to restore swappable weights, entering ERROR state\n", __func__);
                state = gpu_swap_state::ERROR;
                return false;
            }
            swap_stats.n_swap_mmproj_to_cpu++;
            swap_stats.t_swap_mmproj_to_cpu_us += (ggml_time_us() - t_swap);
            fprintf(stderr, "gpu_swap: restored %.2f MiB of swappable weights (%.0f ms)\n",
                    (double) swap_size / 1024.0 / 1024.0, (double) (ggml_time_us() - t_swap) / 1000.0);
        }
    }

    if (lctx) {
        // Force the scheduler to re-reserve with the expected buffer sizes.
        // Without this each swap cycle grows the compute buffer (confirmed:
        // 35 MiB expected vs 154 MiB actual after a session with images).
        llama_context_sched_update(lctx);
    }

    gpu_swap_log_vram("after_restore", lctx);

    state = gpu_swap_state::MODEL_ON_GPU;

    const int64_t t_end = ggml_time_us();
    swap_stats.n_swap_mmproj_to_cpu++;
    swap_stats.n_swap_kv_to_gpu++;
    swap_stats.t_swap_mmproj_to_cpu_us += (t_end - t_start);
    swap_stats.t_swap_kv_to_gpu_us    += (t_end - t_start);

    return true;
}
