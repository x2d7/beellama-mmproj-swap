#include <unistd.h>

extern bool g_inkling_abl_norm; // llama-graph.cpp
#include "models.h"

// fused gated residual of the loop controller (ggml.c), not part of the public ggml.h API
extern "C" struct ggml_tensor * ggml_gated_residual(
        struct ggml_context * ctx,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * message,
        struct ggml_tensor  * gain,
        struct ggml_tensor  * forget);


// fused relative-position bias (ggml.c), not part of the public ggml.h API
extern "C" struct ggml_tensor * ggml_rel_bias_fused(
        struct ggml_context * ctx,
        struct ggml_tensor  * rcur,
        struct ggml_tensor  * sink,
        struct ggml_tensor  * bias,
        struct ggml_tensor  * rel_pos);


// fused mixture over the residual bank (ggml.c), not part of the public ggml.h API
extern "C" struct ggml_tensor * ggml_mix_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * const * blocks,
        int                   n_blocks,
        struct ggml_tensor  * query,
        float                 eps);

// fused short convolution + recurrent state store (ggml.c); not part of the public ggml.h API
extern "C" struct ggml_tensor * ggml_ssm_conv_store(
        struct ggml_context * ctx,
        struct ggml_tensor  * history,
        struct ggml_tensor  * weight,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * state,
        int64_t               n_conv,
        int64_t               ch_off,
        int64_t               st_head,
        float                 scale,
        struct ggml_tensor  * add2);


#include "../llama-memory-hybrid-iswa.h"

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    qkv_packs.assign(hparams.n_layer_all, qkv_pack{});

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // candidate for a packed projection: SWA layers read the same input for q/k/v/gate,
        // so pack the maximal subset of those weights that share a quantized type
        // NOTE: the general "maximal same-type subset" was implemented and measured neutral
        // (15.79/15.79 vs 15.79-15.81 ms, text identical, +70 MB VRAM), so the subset is kept
        // to the one that wins: k/v/gate must all agree, q joins only if it agrees too.
        if (hparams.is_swa(i) && layer.wq && layer.wk && layer.wv && layer.wqkv_gate &&
                layer.wk->type == layer.wv->type && layer.wk->type == layer.wqkv_gate->type) {
            ggml_tensor * t[4] = { layer.wq, layer.wk, layer.wv, layer.wqkv_gate };
            int best_n = 3;
            enum ggml_type best_type = layer.wk->type;
            if (best_n >= 2) {
                auto & p = qkv_packs[i];
                p.wq = layer.wq;
                p.wk = layer.wk;
                p.wv = layer.wv;
                p.wg = layer.wqkv_gate;
                p.inc_q = p.wq->type == best_type;
                p.inc_k = p.wk->type == best_type;
                p.inc_v = p.wv->type == best_type;
                p.inc_g = p.wg->type == best_type;
                p.n_inc = (int) p.inc_q + (int) p.inc_k + (int) p.inc_v + (int) p.inc_g;
                if (p.n_inc >= 2) {
                    p.packed = layer.wk; // marker: non-null means "wants packing"
                }
            }
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

void llama_model_inkling2::post_load_data() {
    const int n_phys = n_layer_phys();

    // what to build: packed projections and F16 relative-position bias tables
    int n_pack = 0;
    for (int il = 0; il < n_phys; ++il) {
        auto & p = qkv_packs[il];
        if (p.packed == nullptr) {
            continue;
        }
        if (hdmd.rot.count(p.wk) || hdmd.rot.count(p.wv) || hdmd.rot.count(p.wg) ||
                (p.inc_q && hdmd.rot.count(p.wq))) {
            // folded weights read a rotated input; the packed gemv would skip that transform
            p.packed = nullptr;
            continue;
        }
        n_pack++;
    }
    int n_bias = 0;
    for (int il = 0; il < n_phys; ++il) {
        if (layers[il].attn_rel_bias != nullptr) {
            n_bias++;
        }
    }
    fprintf(stderr, "%s: %d packed projections, %d folded bias tables\n", __func__, n_pack, n_bias);
    for (int il = 0; il < n_phys; ++il) {
        if (qkv_packs[il].packed != nullptr) {
            continue;
        }
        const auto & L = layers[il];
        if (L.wk == nullptr) {
            continue;
        }
        fprintf(stderr, "  layer %2d NOT packed (swa=%d): q=%s k=%s v=%s gate=%s\n", il, (int) hparams.is_swa(il),
                L.wq ? ggml_type_name(L.wq->type) : "-", ggml_type_name(L.wk->type),
                L.wv ? ggml_type_name(L.wv->type) : "-", L.wqkv_gate ? ggml_type_name(L.wqkv_gate->type) : "-");
    }
    if (n_pack == 0 && n_bias == 0) {
        return;
    }

    struct ggml_init_params ip = {
        /* .mem_size   = */ (size_t) ggml_tensor_overhead()*(size_t) (n_pack + n_bias) + 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    pack_ctx = ggml_init(ip);
    GGML_ASSERT(pack_ctx != nullptr);

    rel_bias16.assign(n_phys, nullptr);
    for (int il = 0; il < n_phys; ++il) {
        ggml_tensor * src = layers[il].attn_rel_bias;
        if (src == nullptr) {
            continue;
        }
        GGML_ASSERT(src->type == GGML_TYPE_F32);
        rel_bias16[il] = ggml_new_tensor_2d(pack_ctx, GGML_TYPE_F16, src->ne[0], src->ne[1] + 1);
    }

    for (int il = 0; il < n_phys; ++il) {
        auto & p = qkv_packs[il];
        if (p.packed == nullptr) {
            continue;
        }
        const int64_t rows = (p.inc_q ? p.wq->ne[1] : 0) + (p.inc_k ? p.wk->ne[1] : 0) +
                             (p.inc_v ? p.wv->ne[1] : 0) + (p.inc_g ? p.wg->ne[1] : 0);
        p.packed = ggml_new_tensor_2d(pack_ctx, p.wk->type, p.wk->ne[0], rows);
        p.packed_ready = true;
    }

    ggml_backend_dev_t dev = dev_layer(0);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend != nullptr);
    pack_buf = ggml_backend_alloc_ctx_tensors(pack_ctx, backend);
    GGML_ASSERT(pack_buf != nullptr);

    // F16 bias tables, padded by one zero column (the extra output head of the projection)
    for (int il = 0; il < n_phys; ++il) {
        ggml_tensor * dst = rel_bias16[il];
        if (dst == nullptr) {
            continue;
        }
        ggml_tensor * src = layers[il].attn_rel_bias;
        const int64_t n_rel = src->ne[0];
        const int64_t n_ext = src->ne[1];

        std::vector<float> h((size_t) n_rel*n_ext);
        ggml_backend_tensor_get(src, h.data(), 0, h.size()*sizeof(float));

        std::vector<ggml_fp16_t> tmp((size_t) n_rel*(n_ext + 1));
        for (int64_t e = 0; e < n_ext + 1; ++e) {
            for (int64_t r = 0; r < n_rel; ++r) {
                tmp[(size_t) r + (size_t) e*n_rel] = e < n_ext
                    ? ggml_fp32_to_fp16(h[(size_t) r + (size_t) e*n_rel])
                    : ggml_fp32_to_fp16(0.0f);
            }
        }
        ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size()*sizeof(ggml_fp16_t));
    }

    for (int il = 0; il < n_phys; ++il) {
        auto & p = qkv_packs[il];
        if (p.packed == nullptr) {
            continue;
        }

        size_t off = 0;
        for (ggml_tensor * src : { p.inc_q ? p.wq : nullptr, p.inc_k ? p.wk : nullptr,
                                   p.inc_v ? p.wv : nullptr, p.inc_g ? p.wg : nullptr }) {
            if (src == nullptr) {
                continue;
            }
            GGML_ASSERT(src->type == p.packed->type);
            const size_t nbytes = ggml_nbytes(src);
            std::vector<char> tmp(nbytes);
            ggml_backend_tensor_get(src, tmp.data(), 0, nbytes);
            ggml_backend_tensor_set(p.packed, tmp.data(), off, nbytes);
            off += nbytes;
        }
    }

    ggml_backend_free(backend);
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale
// one fused op: y = x*scale * min(1, threshold / (sqrt(sum((x*scale)^2)) + 1e-6))
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    // one fused CUDA op: y = x*scale * min(1, threshold / (sqrt(sum(x^2)) + 1e-6))
    if (clip_fused) {
        return ggml_clip_l2(ctx0, cur, scale, threshold);
    }
    // reference chain, kept for the in-process A/B
    ggml_tensor * nrm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    ggml_tensor * den = ggml_scale_bias(ctx0, nrm, 1.0f, 1e-6f);
    ggml_tensor * f   = ggml_clamp(ctx0, ggml_div(ctx0, ggml_repeat(ctx0, ggml_fill(ctx0, den, threshold), den), den), -INFINITY, 1.0f);
    ggml_tensor * res = ggml_mul(ctx0, cur, f);
    if (scale) {
        res = ggml_mul(ctx0, res, scale);
    }
    return res;
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    // fused version: one kernel, blocks over (embedding tile, query, token). The first
    // attempt used one block per token and was 2.9 % slower; this layout gives 9*n_q
    // blocks at decode.
    if (abl_mix && query->ne[1] == 1) {
        return ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]); // ablation: drop the mix
    }
    if (mix_fused && blocks.size() <= 8 && ggml_is_contiguous(cur) && ggml_is_contiguous(query)) {
        std::vector<ggml_tensor *> blk;
        blk.reserve(blocks.size());
        for (ggml_tensor * b : blocks) {
            if (!ggml_is_contiguous(b) || b->ne[0] != cur->ne[0] || b->ne[1] != cur->ne[1]) {
                blk.clear();
                break;
            }
            blk.push_back(b);
        }
        if (blk.size() == blocks.size()) {
            return ggml_mix_attn(ctx0, cur, blk.data(), (int) blk.size(), query, 1e-6f);
        }
    }
    ggml_tensor * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    for (ggml_tensor * block : blocks) {
        states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
    }
    states = ggml_rms_norm(ctx0, states, 1e-6f);

    ggml_tensor * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset,
        ggml_tensor * add2) {
    if (abl_conv) {
        return cur; // ablation: drop the short convolution and its epilogue
    }
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    if (!dynamic && cparams.n_rs_seq == 0 && conv_fused) {
        // fused: convolution, history read, state store and the residual+scale epilogue
        // in one kernel: out = (cur + conv)/sqrt(n_conv + 1)
        ggml_tensor * out = ggml_ssm_conv_store(ctx0, history, weight,
                ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs),
                state, n_conv, offset, inp->mctx->get_head(),
                1.0f/sqrtf((float) n_conv + 1), add2);
        return ggml_reshape_2d(ctx0, out, n_channels, n_tokens);
    }

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);
    if (add2 != nullptr) {
        return ggml_add(ctx0, ggml_add(ctx0, cur, out), add2); // chain path keeps the old math
    }

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl,
        ggml_tensor * rms_cur) {
    if (abl_ctrl) {
        return cur; // ablation: drop the loop controllers
    }
    // rms_cur is shared by the q, k and v controllers: their inputs are the same tensor
    ggml_tensor * gate = ggml_concat(ctx0, rms_cur ? rms_cur : ggml_rms_norm(ctx0, cur, 1e-6f),
            ggml_rms_norm(ctx0, message, 1e-6f), 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = build_lora_mm(loop.ctrl_gain[ctrl],   gate);
    ggml_tensor * forget = build_lora_mm(loop.ctrl_forget[ctrl], gate);

    if (epi_fused && loras->empty() && n_tokens == 1) {
        // one kernel: out = cur + tanh(gain)*message - tanh(forget)*cur
        return ggml_gated_residual(ctx0, cur, message, gain, forget);
    }

    gain   = ggml_tanh(ctx0, gain);
    forget = ggml_tanh(ctx0, forget);

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();

    ggml_tensor * inp_pos = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    zero_sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);

    if (abl_aproj) {
        // ablation: constant stand-ins for the attention projections
        const int64_t hd = hparams.n_embd_head_k(0);
        zero_q  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hd*n_head,    n_tokens), 0.0f);
        zero_kv = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hd*n_head_kv, n_tokens), 0.0f);
        zero_o  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd,       n_tokens), 0.0f);
    }

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);

    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the message of controller ctrl, [n_embd, n_tokens]
    auto message = [&](int ctrl) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, messages, n_embd, n_tokens, messages->nb[2], ctrl*messages->nb[1]));
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    // one zero sink vector for every layer: the learned sink logits are folded into kq_b
    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (layer.attn_res_query) {
        ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], (is_swa ? 0 : j)*mixed->nb[1]));
        }
    }

    if (messages) {
        // the messages make the Q, K and V inputs of a SWA layer differ
        ggml_tensor * rms_in = ggml_rms_norm(ctx0, inputs[0], 1e-6f);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], message(LOOP_CTRL_Q + j), *loop, LOOP_CTRL_Q + j, rms_in);
        }
    }

    if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    const bool block_start = (ip % hparams.attn_res_block_size == 0);
    ggml_tensor * gate_proj = nullptr;
    {
        ggml_tensor * Qcur = nullptr;
        ggml_tensor * Kcur = nullptr;
        ggml_tensor * Vcur = nullptr;

        const auto & pack = static_cast<const llama_model_inkling2 &>(model).qkv_packs[ip];
        if (abl_aproj) {
            Qcur = zero_q;
            Kcur = zero_kv;
            Vcur = zero_kv;
            gate_proj = zero_q;
        } else if (pack.packed_ready && pack_fused && pack5_fused && loras->empty() && is_swa && !messages && n_tokens == 1 &&
                inputs[0] == inputs[1]) {
            // one gemv over the packed weights; with a single token each projection is a
            // contiguous slice of the output rows
            const int64_t n_q = n_embd_head*n_head;
            ggml_tensor * out = ggml_mul_mat(ctx0, pack.packed, inputs[0]);
            cb(out, "qkvg", il);
            GGML_ASSERT(out->ne[1] == 1);
            const size_t esz = sizeof(float);
            size_t off = 0;
            if (pack.inc_q) { Qcur = ggml_view_2d(ctx0, out, n_q, 1, out->nb[1], off); off += (size_t) n_q*esz; }
            else            { Qcur = build_lora_mm(layer.wq, inputs[0]); }
            if (pack.inc_k) { Kcur = ggml_view_2d(ctx0, out, n_embd_k_gqa, 1, out->nb[1], off); off += (size_t) n_embd_k_gqa*esz; }
            else            { Kcur = build_lora_mm(layer.wk, inputs[1]); }
            if (pack.inc_v) { Vcur = ggml_view_2d(ctx0, out, n_embd_k_gqa, 1, out->nb[1], off); off += (size_t) n_embd_k_gqa*esz; }
            else            { Vcur = build_lora_mm(layer.wv, inputs[2]); }
            if (pack.inc_g) { gate_proj = ggml_view_2d(ctx0, out, n_q, 1, out->nb[1], off); off += (size_t) n_q*esz; }
            GGML_ASSERT(off == (size_t) out->ne[0]*esz);
        } else {
            Qcur = build_lora_mm(layer.wq, inputs[0]);
            Kcur = build_lora_mm(layer.wk, inputs[1]);
            Vcur = build_lora_mm(layer.wv, inputs[2]);
        }

        Kcur = convolve(Kcur, layer.attn_k_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
        Vcur = convolve(Vcur, layer.attn_v_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
        ggml_tensor * Rcur = (abl_aproj || abl_rel)
                ? ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams.n_relative, n_head, n_tokens), 0.0f)
                : build_lora_mm(layer.attn_rel, inputs[0]);

        Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, hparams.f_qk_clip);
        Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           hparams.f_qk_clip);
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
        if (!abl_rel) {
            Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip);
        }

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
            Rcur = ggml_mul(ctx0, Rcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

        const int64_t n_stream = (is_swa ? attn->self_kq_mask_swa : attn->self_kq_mask)->ne[3];
        const int64_t n_kv     = rel_pos->ne[0];

        ggml_tensor * sinks = (abl_aproj || abl_rel)
                ? ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_head, n_tokens), 0.0f)
                : build_lora_mm(layer.attn_sink_proj, inputs[0]);

        ggml_tensor * kq_b = nullptr;
        if (abl_bias) {
            // ablation: keep flash attention and a bias of the same shape/type, but zero it,
            // so the whole relative-position chain (Rcur gemv, table, sink, permute, gather)
            // is what disappears. ggml_fill is F32 only, hence the cast.
            ggml_tensor * z = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
                    n_kv, n_tokens/n_stream, n_head, n_stream), 0.0f);
            kq_b = ggml_cast(ctx0, z, GGML_TYPE_F16);
            sinks = zero_sinks;
        } else {
            // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
            // the table runs in F16: the gather and the mask fold are per token work on n_kv rows
            // the memory-fit probe builds a graph from a model whose derived tensors are not
            // uploaded yet, so the folded table is optional
            const auto & bias_tables = static_cast<const llama_model_inkling2 &>(model).rel_bias16;
            ggml_tensor * rel_bias_w = (relb_fused && (size_t) ip < bias_tables.size()) ? bias_tables[ip] : nullptr;
            if (rel_bias_w == nullptr) {
                rel_bias_w = ggml_cast(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), GGML_TYPE_F16);
            }
            {
            if (rbf_fused && Rcur->ne[1] == n_head && Rcur->ne[2] == n_tokens &&
                    rel_bias_w->ne[0] == Rcur->ne[0] && ggml_is_contiguous(Rcur) &&
                    rel_pos->ne[0] == n_kv) {
                // one kernel for the table, the sink subtract and the gather by distance
                kq_b = ggml_cast(ctx0, ggml_rel_bias_fused(ctx0, Rcur, sinks, rel_bias_w, rel_pos), GGML_TYPE_F16);
            } else {
            ggml_tensor * table = ggml_mul_mat(ctx0, rel_bias_w, ggml_cast(ctx0, Rcur, GGML_TYPE_F16));
            // mul_mat returns F32, the gather and the mask fold are cheaper in F16
            table = ggml_cast(ctx0, table, GGML_TYPE_F16);

            // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
            table = ggml_sub(ctx0, table, ggml_cast(ctx0, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens), GGML_TYPE_F16));
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));

            // F16 output: ggml_get_rows always returns F32, which costs n_kv*n_head*n_tokens of VRAM
            kq_b = ggml_get_rows_as(ctx0, table, rel_pos, GGML_TYPE_F16);
            }
            }
            kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, n_tokens/n_stream, n_stream);
            // [n_kv, n_tokens/n_stream, n_head, n_stream], the layout of the mask and of the flash attention bias
            // note: kept as a permuted view, a contiguous copy of it costs n_kv*n_head*n_tokens per layer
            kq_b = ggml_permute(ctx0, kq_b, 2, 0, 1, 3);
            cb(kq_b, "kq_b", il);

            sinks = zero_sinks;
        }

        if (abl_attn) {
            // ablation: skip flash attention only. Every projection above still runs (and
            // every weight stays referenced), so this is the wall cost of FA + its KV reads.
            cur = ggml_scale(ctx0, ggml_reshape_2d(ctx0, Qcur, n_embd_head*n_head, n_tokens), 0.0f);
        } else {
            cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il);
        }

        if (!gate_proj) {
            gate_proj = build_lora_mm(layer.wqkv_gate, inputs[0]);
        }
        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate_proj));
        // the residual join of this layer is folded into the conv epilogue when the layer
        // does not start a new residual stream
        cur = convolve(abl_aproj ? zero_o : build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa,
                (add2_fused && !block_start) ? inpSA : nullptr);
        cb(cur, "attn_out", il);
    }

    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (block_start) {
        blocks.push_back(inpSA);
    } else if (!add2_fused) {
        cur = ggml_add(ctx0, inpSA, cur);
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    if (abl_moe) {
        // ablation: drop the whole FFN (routed experts and shared expert)
        cur = ggml_scale(ctx0, cur, 0.0f);
    } else {
    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    }

    // the residual add is folded into the convolution epilogue: (moe + conv)*scale + ffn_inp
    cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd,
            abl_conv ? nullptr : (add2_fused ? ffn_inp : nullptr));
    cb(cur, "ffn_out", il);

    if (!add2_fused) {
        cur = ggml_add(ctx0, ffn_inp, cur);
    }
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

// one stat() per graph build: arm toggles for in-process A/B, so a comparison has no
// server restart or request-ordering bias. The looped graph uses the no_build ctor.


void llama_model_inkling2::graph::init_arm_flags() {
    conv_fused = access("/tmp/opencode/conv_off", F_OK) != 0;
    clip_fused = access("/tmp/opencode/clip_off", F_OK) != 0;
    pack_fused = access("/tmp/opencode/pack_off", F_OK) != 0;
    mix_fused  = access("/tmp/opencode/mix_off",  F_OK) != 0;
    relb_fused = access("/tmp/opencode/relb_off", F_OK) != 0;
    epi_fused  = access("/tmp/opencode/epi_off",  F_OK) != 0;
    abl_rel    = access("/tmp/opencode/abl_rel",  F_OK) == 0;
    abl_moe    = access("/tmp/opencode/abl_moe",  F_OK) == 0;
    abl_ctrl   = access("/tmp/opencode/abl_ctrl", F_OK) == 0;
    abl_conv   = access("/tmp/opencode/abl_conv", F_OK) == 0;
    abl_mix    = access("/tmp/opencode/abl_mix",  F_OK) == 0;
    abl_aproj  = access("/tmp/opencode/abl_aproj", F_OK) == 0;
    pack5_fused = access("/tmp/opencode/pack5_off", F_OK) != 0;
    add2_fused = access("/tmp/opencode/add2_off", F_OK) != 0;
    g_inkling_abl_norm = access("/tmp/opencode/abl_norm", F_OK) == 0;
    abl_attn   = access("/tmp/opencode/abl_attn", F_OK) == 0;
    abl_bias   = access("/tmp/opencode/abl_bias", F_OK) == 0;
    rbf_fused  = access("/tmp/opencode/rbf_off",  F_OK) != 0;
    static bool once = [] { return true; }();
    if (once) {
        fprintf(stderr, "arm flags: conv=%d clip=%d pack=%d mix=%d relb=%d epi=%d abl_rel=%d abl_moe=%d abl_ctrl=%d abl_attn=%d abl_bias=%d rbf=%d abl_conv=%d abl_norm=%d abl_mix=%d add2=%d abl_aproj=%d pack5=%d\n", (int) conv_fused, (int) clip_fused, (int) pack_fused, (int) mix_fused, (int) relb_fused, (int) epi_fused, (int) abl_rel, (int) abl_moe, (int) abl_ctrl, (int) abl_attn, (int) abl_bias, (int) rbf_fused, (int) abl_conv, (int) g_inkling_abl_norm, (int) abl_mix, (int) add2_fused, (int) abl_aproj, (int) pack5_fused);
        once = false;
    }
}

llama_model_inkling2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params), model(model) {
    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);

}
