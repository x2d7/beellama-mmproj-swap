#include "common.cuh"

// Attention of n_q queries over the states [last block, ..., first block, cur].
// dst[i, j, t] = sum_s state_s[i,t]/rms_s * softmax_s(scale * q[i,j] . state_s[:,t]/rms_s)
//
// One block per (embedding tile, query, token): every block recomputes the n_st
// reductions (n_st*ne0 reads, L2 resident) and then writes its own 256 outputs.
// The previous version used one block per token, i.e. a single SM at decode, and was
// measured 2.9 % slower than the unfused chain.

static __global__ void mix_attn_f32(
        const float * __restrict__ q,        // [ne0, n_q] broadcast over tokens
        const float * __restrict__ cur,      // [ne0, n_tok]
        const float * __restrict__ b0, const float * __restrict__ b1,
        const float * __restrict__ b2, const float * __restrict__ b3,
        const float * __restrict__ b4, const float * __restrict__ b5,
        const float * __restrict__ b6, const float * __restrict__ b7,
        const int    n_blocks,
        float       * __restrict__ y,        // [ne0, n_q, n_tok]
        const int64_t ne0, const int64_t n_q, const int64_t n_tok,
        const float   eps,
        const float   scale) {
    const int64_t t = blockIdx.z;
    const int64_t j = blockIdx.y;
    if (t >= n_tok || j >= n_q) {
        return;
    }

    const float * blk_all[8] = { b0, b1, b2, b3, b4, b5, b6, b7 };
    const float * st[9];
    const int n_st = n_blocks + 1;
    for (int s = 0; s < n_blocks; ++s) {
        st[s] = blk_all[n_blocks - 1 - s];
    }
    st[n_blocks] = cur;

    __shared__ float2 red[256];
    __shared__ float wgt[9]; // softmax weight over the states
    __shared__ float rms_s[9];
    const int tid = threadIdx.x;

    // every block scores all states; the states are a few tens of KB and stay in L2
    for (int s = 0; s < n_st; ++s) {
        const float * xs = st[s] + t*ne0;
        const float * qs = q + j*ne0;
        float ss = 0.0f;
        float sd = 0.0f;
        for (int64_t i = tid; i < ne0; i += blockDim.x) {
            const float v = xs[i];
            ss += v*v;
            sd += v*qs[i];
        }
        // one reduction for both the rms and the dot
        red[tid] = make_float2(ss, sd);
        __syncthreads();
        for (int w = blockDim.x/2; w > 0; w >>= 1) {
            if (tid < w) {
                red[tid].x += red[tid + w].x;
                red[tid].y += red[tid + w].y;
            }
            __syncthreads();
        }
        if (tid == 0) {
            rms_s[s] = sqrtf(red[0].x/ne0 + eps);
            wgt[s]   = scale*red[0].y/rms_s[s];
        }
        __syncthreads();
    }

    // softmax over the states, divided by their rms
    if (tid == 0) {
        float mx = -INFINITY;
        for (int s = 0; s < n_st; ++s) {
            mx = fmaxf(mx, wgt[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < n_st; ++s) {
            wgt[s] = expf(wgt[s] - mx);
            sum += wgt[s];
        }
        for (int s = 0; s < n_st; ++s) {
            // out[i] = sum_s state_s[i]/rms_s * p_s
            wgt[s] = wgt[s]/sum/rms_s[s];
        }
    }
    __syncthreads();

    float * ys = y + (t*n_q + j)*ne0; // dst is [ne0, n_q, n_tok]
    for (int64_t i = blockIdx.x*blockDim.x + tid; i < ne0; i += blockDim.x*gridDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_st; ++s) {
            acc += st[s][i + t*ne0]*wgt[s];
        }
        ys[i] = acc;
    }
}


// Gated residual of the inkling2 loop controller:
//   out[i] = cur[i] + tanh(gain[i])*msg[i] - tanh(forget[i])*cur[i]
// Four elementwise kernels (two tanh, two mul, sub, add) become one. Marked in
// ggml_cuda_op_mix_attn by n_blocks == -1.
static __global__ void gated_residual_f32(
        const float * __restrict__ cur,
        const float * __restrict__ msg,
        const float * __restrict__ gain,
        const float * __restrict__ forget,
        float       * __restrict__ y,
        const int64_t ne0) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne0) {
        return;
    }
    const float c = cur[i];
    y[i] = c + tanhf(gain[i])*msg[i] - tanhf(forget[i])*c;
}



// Relative-position bias of one attention layer, in one kernel:
//   out[head, kv, t] = sum_r bias16[r, head, rel_pos[kv, t]]*rcur[r, head, t] - sink[head, t]
// rel_pos is [n_kv, n_tokens]: the distance is per (kv, token), not per kv.
// Replaces cast + mul_mat + cast + cast + sub + cont + get_rows. Marked by n_blocks == -2.
static __global__ void rel_bias_fused_f32(
        const float       * __restrict__ rcur,     // [n_rel, n_head, nt]
        const float       * __restrict__ sink,     // [n_head, nt]
        const ggml_fp16_t * __restrict__ bias,     // [n_rel, n_ext+1]
        const int32_t     * __restrict__ rel_pos,  // [n_kv, nt]
        float             * __restrict__ out,      // [n_head, n_kv, nt]
        const int n_rel, const int n_head, const int n_ext, const int n_kv, const int nt) {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n_head*n_kv*nt) {
        return;
    }
    const int h  = idx % n_head;
    const int kv = (idx / n_head) % n_kv;
    const int t  = idx / (n_head*n_kv);

    const int e = rel_pos[kv + (size_t) t*n_kv];
    const ggml_fp16_t * br = bias + (size_t) e*n_rel;
    const float * rr = rcur + (size_t) h*n_rel + (size_t) t*n_rel*n_head;

    float acc = 0.0f;
    for (int r = 0; r < n_rel; ++r) {
        acc += __half2float(*(const __half *) &br[r])*rr[r];
    }
    // same rounding order as the chain: the table is cast to F16, then the sink (also F16)
    // is subtracted, and only then is the result rounded again
    const float t16 = __half2float(__float2half(acc));
    const float s16 = __half2float(__float2half(sink[h + (size_t) t*n_head]));
    out[idx] = __half2float(__float2half(t16 - s16));
}

void ggml_cuda_op_mix_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * cur = dst->src[1];
    const int n_blocks = ggml_get_op_params_i32(dst, 0);

    if (n_blocks == -2) {
        // fused relative-position bias: src0 = rcur, src1 = sink, src2 = bias f16, src3 = rel_pos
        const ggml_tensor * rcur    = dst->src[0];
        const ggml_tensor * sink    = dst->src[1];
        const ggml_tensor * bias    = dst->src[2];
        const ggml_tensor * rel_pos = dst->src[3];

        const int n_rel  = (int) rcur->ne[0];
        const int n_head = (int) rcur->ne[1];
        const int nt     = (int) rcur->ne[2];
        const int n_ext  = (int) bias->ne[1] - 1;
        const int n_kv   = (int) dst->ne[1];

        const int64_t n = (int64_t) n_head*n_kv*nt;
        const dim3 block_dims(256);
        const dim3 block_nums((unsigned) ((n + 255)/256));
        cudaStream_t stream = ctx.stream();
        rel_bias_fused_f32<<<block_nums, block_dims, 0, stream>>>(
                (const float *) rcur->data, (const float *) sink->data,
                (const ggml_fp16_t *) bias->data, (const int32_t *) rel_pos->data,
                (float *) dst->data, n_rel, n_head, n_ext, n_kv, nt);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (n_blocks == -1) {
        // gated residual: src[0] = cur, src[1] = message, src[2] = gain, src[3] = forget
        const int64_t ne0 = dst->ne[0]*dst->ne[1];
        const dim3 block_dims(256);
        const dim3 block_nums((unsigned) ((ne0 + 255)/256));
        cudaStream_t stream = ctx.stream();
        gated_residual_f32<<<block_nums, block_dims, 0, stream>>>(
                (const float *) dst->src[0]->data, (const float *) dst->src[1]->data,
                (const float *) dst->src[2]->data, (const float *) dst->src[3]->data,
                (float *) dst->data, ne0);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    float eps;
    memcpy(&eps, (const char *) dst->op_params + sizeof(int32_t), sizeof(float));

    const float * qd = (const float *) q->data;
    const float * cd = (const float *) cur->data;
    float       * yd = (float       *) dst->data;

    const float * bp[8] = { nullptr };
    for (int i = 0; i < n_blocks; ++i) {
        bp[i] = (const float *) dst->src[2 + i]->data;
    }

    const int64_t ne0 = q->ne[0];
    const dim3 block_dims(256);
    const dim3 block_nums((unsigned) ((ne0 + 255)/256), (unsigned) q->ne[1], (unsigned) cur->ne[1]);

    cudaStream_t stream = ctx.stream();
    mix_attn_f32<<<block_nums, block_dims, 0, stream>>>(
            qd, cd, bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6], bp[7],
            n_blocks, yd, ne0, q->ne[1], cur->ne[1], eps, 1.0f/sqrtf((float) ne0));
    CUDA_CHECK(cudaGetLastError());
}
