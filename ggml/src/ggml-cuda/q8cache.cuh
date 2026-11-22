#pragma once

#include <cstddef>
#include <cstdint>
#include "ggml.h"

// One q8_1 copy of a mul_mat src1 vector, reused by every gemv that reads the same
// tensor within one graph compute.  The copy is dropped as soon as any node writes to
// the source range, so the cache can never return stale data.
void   ggml_cuda_q8_cache_set_evict(void (*fn)(void * ud, void * ptr, size_t size), void * ud);
void   ggml_cuda_q8_cache_reset(bool enabled);
void * ggml_cuda_q8_cache_get(const void * src, int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                              enum ggml_type type_src0, size_t * size);
void   ggml_cuda_q8_cache_put(const void * src, size_t src_bytes,
                              int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                              enum ggml_type type_src0, void * buf, size_t size);
void   ggml_cuda_q8_cache_invalidate(const void * ptr, size_t bytes);
