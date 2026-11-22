#include "q8cache.cuh"

#include <vector>
#include <cstdio>
#include <cstdlib>

namespace {

struct entry {
    const void * src   = nullptr;
    size_t       bytes = 0;   // span of the source tensor
    int64_t      ne10 = 0, ne11 = 0, ne12 = 0, ne13 = 0;
    enum ggml_type type_src0 = GGML_TYPE_COUNT;
    void       * buf   = nullptr;
    size_t       size  = 0;
};

constexpr size_t MAX_ENTRIES = 8;

std::vector<entry> g_cache;
bool               g_enabled = true;
void (*g_evict)(void *, void *, size_t) = nullptr;
void * g_evict_ud = nullptr;

// every buffer handed to the cache, in allocation order: the pool is a bump
// allocator with a LIFO free, so the whole list is returned in reverse at the end of
// the compute. Dropping a cache entry must NOT free, it may sit below live ones.
// The evict callback belongs to the backend context (and therefore the pool) that
// allocated the buffer: the cache is global, but the compute pools are per backend
// context, so the clip backend must not free a llama-pool buffer (and vice versa).
struct alloc_rec {
    void * ptr;
    size_t size;
    void (*evict)(void *, void *, size_t);
    void * ud;
};

std::vector<alloc_rec> g_allocs;

long g_hits = 0, g_misses = 0, g_resets = 0;

void evict_all() {
    for (size_t i = g_allocs.size(); i-- > 0; ) {
        const auto & a = g_allocs[i];
        if (a.evict != nullptr && a.ptr != nullptr) {
            a.evict(a.ud, a.ptr, a.size);
        }
    }
    g_allocs.clear();
    g_cache.clear();
}

} // namespace

void ggml_cuda_q8_cache_set_evict(void (*fn)(void * ud, void * ptr, size_t size), void * ud) {
    g_evict    = fn;
    g_evict_ud = ud;
}

void ggml_cuda_q8_cache_reset(bool enabled) {
    evict_all();
    g_enabled = enabled;
    if (getenv("Q8C_STATS") != nullptr) {
        fprintf(stderr, "q8cache: hits %ld misses %ld (%.0f%% hit)\n",
                g_hits, g_misses, 100.0*g_hits/(g_hits+g_misses+1e-9));
    }
}

void * ggml_cuda_q8_cache_get(const void * src, int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                              enum ggml_type type_src0, size_t * size) {
    if (!g_enabled) {
        return nullptr;
    }
    g_misses++;
    for (const entry & e : g_cache) {
        // the q8_1 block layout of this path does not depend on src0->type
        // (quantize_row_q8_1_cuda ignores it), so a copy is shared across weight types
        if (e.src == src && e.ne10 == ne10 && e.ne11 == ne11 && e.ne12 == ne12 && e.ne13 == ne13) {
            g_hits++;
            *size = e.size;
            return e.buf;
        }
    }
    return nullptr;
}

void ggml_cuda_q8_cache_put(const void * src, size_t src_bytes,
                            int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                            enum ggml_type type_src0, void * buf, size_t size) {
    if (!g_enabled) {
        return;
    }
    if (g_cache.size() >= MAX_ENTRIES) {
        // drop the oldest reference only; the buffer itself is freed at reset
        g_cache.erase(g_cache.begin());
    }
    g_cache.push_back({ src, src_bytes, ne10, ne11, ne12, ne13, type_src0, buf, size });
    g_allocs.push_back({ buf, size, g_evict, g_evict_ud });
}

void ggml_cuda_q8_cache_invalidate(const void * ptr, size_t bytes) {
    if (g_cache.empty() || ptr == nullptr || bytes == 0) {
        return;
    }
    const char * p0 = (const char *) ptr;
    const char * p1 = p0 + bytes;
    for (size_t i = 0; i < g_cache.size(); ) {
        const char * s0 = (const char *) g_cache[i].src;
        const char * s1 = s0 + g_cache[i].bytes;
        if (p0 < s1 && s0 < p1) {
            // a node writes into the cached source: drop the reference, the buffer
            // is returned to the pool at reset
            g_cache.erase(g_cache.begin() + i);
        } else {
            ++i;
        }
    }
}
