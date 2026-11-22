#include "models.h"

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    ml.get_key(LLM_KV_LOOP_START,     core_start);
    ml.get_key(LLM_KV_LOOP_END,       core_end);
    ml.get_key(LLM_KV_LOOP_COUNT,     n_passes);
    ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
    ml.get_key(LLM_KV_LOOP_VARYING,   varying, false);

    n_phys = hparams.n_layer();
    const uint32_t block = hparams.attn_res_block_size;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[p];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[p];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[p];
        hparams.is_swa_impl[il]       = hparams.is_swa_impl[p];
        hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    loop_layers.resize(core_end - core_start);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = create_tensor(tn(down[c],   "weight", il), {2*n_embd, rank}, 0);
            l.ctrl_gain[c]   = create_tensor(tn(gain[c],   "weight", il), {rank, n_embd}, 0);
            l.ctrl_forget[c] = create_tensor(tn(forget[c], "weight", il), {rank, n_embd}, 0);
        }
    }

    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

// the loop slots copy the physical layers at the end of load_arch_tensors, before the generic pass
// attaches the optional ".scale" tensors; copy them again so the later passes run scaled experts
void llama_model_inkling2_loop::reload_arch_tensor_layout(llama_model_loader &) {
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

// Stack ctrl_gain and ctrl_forget of every loop controller into one [rank, 2*n_embd]
// tensor: both read the same gate vector, so one gemv replaces two. Byte copies of the
// F32 weights, no conversion.
void llama_model_inkling2_loop::post_load_data() {
    llama_model_inkling2::post_load_data();

    int n_gf = 0;
    for (const auto & l : loop_layers) {
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            if (l.ctrl_gain[c] != nullptr && l.ctrl_forget[c] != nullptr &&
                    l.ctrl_gain[c]->type == GGML_TYPE_F32 && l.ctrl_forget[c]->type == GGML_TYPE_F32) {
                n_gf++;
            }
        }
    }
    if (n_gf == 0) {
        return;
    }

    struct ggml_init_params ip = {
        /* .mem_size   = */ (size_t) ggml_tensor_overhead()*(size_t) n_gf + 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctrl_ctx = ggml_init(ip);
    GGML_ASSERT(ctrl_ctx != nullptr);

    for (auto & l : loop_layers) {
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            ggml_tensor * g = l.ctrl_gain[c];
            ggml_tensor * f = l.ctrl_forget[c];
            if (g == nullptr || f == nullptr || g->type != GGML_TYPE_F32 || f->type != GGML_TYPE_F32) {
                continue;
            }
            GGML_ASSERT(f->ne[0] == g->ne[0] && f->ne[1] == g->ne[1]);
            l.ctrl_gain_forget[c] = ggml_new_tensor_2d(ctrl_ctx, GGML_TYPE_F32, g->ne[0], 2*g->ne[1]);
        }
    }

    ggml_backend_dev_t dev = dev_layer(0);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend != nullptr);
    ctrl_buf = ggml_backend_alloc_ctx_tensors(ctrl_ctx, backend);
    GGML_ASSERT(ctrl_buf != nullptr);

    for (auto & l : loop_layers) {
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            ggml_tensor * dst = l.ctrl_gain_forget[c];
            if (dst == nullptr) {
                continue;
            }
            const size_t nbytes = ggml_nbytes(l.ctrl_gain[c]);
            std::vector<char> tmp(nbytes);
            ggml_backend_tensor_get(l.ctrl_gain[c], tmp.data(), 0, nbytes);
            ggml_backend_tensor_set(dst, tmp.data(), 0, nbytes);
            ggml_backend_tensor_get(l.ctrl_forget[c], tmp.data(), 0, nbytes);
            ggml_backend_tensor_set(dst, tmp.data(), nbytes, nbytes);
        }
    }
    fprintf(stderr, "%s: stacked %d controller gain/forget pairs\n", __func__, n_gf);
    ggml_backend_free(backend);
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_inkling2_loop::graph::graph(const llama_model & model, const llm_graph_params & params) : llama_model_inkling2::graph(model, params, no_build{}) {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    const int start = m.core_start;
    const int end   = m.core_end;

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < start; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < m.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            cur = build_layer(il, ip, cur, blocks, mix(bank, bank_out, loop.msg_query), &loop);
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }
    build_output(cur, blocks);
}
