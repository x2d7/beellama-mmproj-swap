#pragma once

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

#include <mutex>
#include <vector>
#include <string>
#include <cstdint>

enum class gpu_swap_state {
    MODEL_ON_GPU,    // KV cache on GPU (text inference ready)
    MMPROJ_ON_GPU,   // KV cache offloaded to RAM, mmproj on GPU (image inference ready)
    TRANSITIONING,   // Switching state
    ERROR,           // Error state: swap failed and cannot recover safely
};

struct gpu_swap_manager {
    gpu_swap_state state = gpu_swap_state::MODEL_ON_GPU;
    bool enabled = false;

    // How much GPU VRAM we try to free when moving mmproj to GPU (bytes).
    // Used only for logging / sanity; the KV cache size dictates what is freed.
    size_t target_free_bytes = 0;

    std::mutex mtx;

    struct stats {
        int n_swap_kv_to_cpu = 0;
        int n_swap_mmproj_to_gpu = 0;
        int n_swap_mmproj_to_cpu = 0;
        int n_swap_kv_to_gpu = 0;
        int64_t t_swap_kv_to_cpu_us = 0;
        int64_t t_swap_mmproj_to_gpu_us = 0;
        int64_t t_swap_mmproj_to_cpu_us = 0;
        int64_t t_swap_kv_to_gpu_us = 0;
    };
    stats swap_stats;

    // Offload the KV cache to host RAM (freeing VRAM) so the caller can load
    // mmproj to GPU for image encoding.
    bool swap_to_mmproj_gpu(struct mtmd_context * mctx,
                            struct llama_model * model,
                            struct llama_context * lctx);

    // Restore the KV cache to GPU after mmproj is back on CPU.
    bool swap_to_model_gpu(struct mtmd_context * mctx,
                           struct llama_model * model,
                           struct llama_context * lctx);

    void reset();

    gpu_swap_state get_state() const;
    void print_stats() const;
};
