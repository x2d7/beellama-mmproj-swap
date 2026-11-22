#include "common.cuh"

// y = x * scale * min(1, threshold / sqrt(sum(x^2) + eps)) over dim 0, contiguous F32.
// Same math as ggml_sqrt(ggml_sum_rows(ggml_sqr(x))) + fill/div/clamp/mul, in one kernel.

static __global__ void clip_l2_f32(
        const float * __restrict__ x,
        const float * __restrict__ scale, // optional, [ne0, ne1] or [ne0]
        float       * __restrict__ y,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float   threshold,
        const bool    scale_per_head) {
    const int64_t row = blockIdx.x;
    if (row >= ne1*ne2*ne3) {
        return;
    }
    const int64_t i1 = row % ne1;
    const int64_t t  = row / ne1;
    const int64_t i2 = t % ne2;
    const int64_t i3 = t / ne2;
    const int64_t off = ((i3*ne2 + i2)*ne1 + i1)*ne0;

    const float * xs = x + off;
    float       * ys = y + off;

    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < ne0; i += blockDim.x) {
        const float v = xs[i];
        sum += v*v;
    }

    __shared__ float red[256];
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }

    float f = threshold / (sqrtf(red[0]) + 1e-6f);
    f = fminf(1.0f, f);

    for (int64_t i = threadIdx.x; i < ne0; i += blockDim.x) {
        float v = xs[i]*f;
        if (scale != nullptr) {
            v *= scale_per_head ? scale[i1*ne0 + i] : scale[i];
        }
        ys[i] = v;
    }
}

void ggml_cuda_op_clip_l2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * x = (const float *) src0->data;
    float       * y = (float       *) dst->data;
    const float * sc = src1 ? (const float *) src1->data : nullptr;
    const bool scale_per_head = src1 && src1->ne[1] > 1;

    float threshold;
    memcpy(&threshold, dst->op_params, sizeof(float));

    const int64_t nrows = src0->ne[1]*src0->ne[2]*src0->ne[3];
    const dim3 block_dims(256);
    const dim3 block_nums((unsigned) nrows);

    cudaStream_t stream = ctx.stream();
    clip_l2_f32<<<block_nums, block_dims, 0, stream>>>(
            x, sc, y, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], threshold, scale_per_head);
    CUDA_CHECK(cudaGetLastError());
}
