#pragma once

// Dynamic prefill batching: choose the physical batch (ubatch) from the current
// KV occupancy and the free VRAM, so a long prompt starts with a large batch and
// smoothly steps down as the KVarN prefill window and the KV cache grow.
//
// See docs/dyn-batch.md for the memory model and the measured curves.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct server_dyn_batch_point {
    uint32_t kv = 0; // KV occupancy in tokens
    uint32_t ub = 0; // ubatch target at that occupancy
};

struct server_dyn_batch {
    std::vector<server_dyn_batch_point> curve;
    uint32_t ub_min  = 48;
    uint32_t ub_max  = 0;   // captured from the context at startup
    int32_t  guard_mb = 300;
    uint32_t step    = 16;  // deadband between cap changes
    uint32_t max_drop = 0;  // 0 = jump straight to the target (sharp), >0 = smooth
    bool     verbose = false;
    bool     auto_curve = false; // derive the curve from free VRAM at runtime

    bool enabled() const { return auto_curve || !curve.empty(); }

    // spec: "off" or "kv:ub,kv:ub,..." e.g. "0:256,16384:192,65536:64,98304:48"
    bool configure(const std::string & spec, uint32_t ub_ctor) {
        curve.clear();
        auto_curve = false;
        ub_max = ub_ctor;
        if (spec.empty() || spec == "off" || spec == "none") {
            return false;
        }
        if (spec == "auto") {
            auto_curve = true;
            return true;
        }
        size_t pos = 0;
        while (pos < spec.size()) {
            const size_t comma = spec.find(',', pos);
            const std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            pos = comma == std::string::npos ? spec.size() : comma + 1;
            if (tok.empty()) {
                continue;
            }
            const size_t colon = tok.find(':');
            if (colon == std::string::npos) {
                std::fprintf(stderr, "dyn-batch: bad curve point '%s' (expected kv:ub)\n", tok.c_str());
                curve.clear();
                return false;
            }
            const long kv = std::strtol(tok.substr(0, colon).c_str(), nullptr, 10);
            const long ub = std::strtol(tok.substr(colon + 1).c_str(), nullptr, 10);
            if (kv < 0 || ub <= 0) {
                std::fprintf(stderr, "dyn-batch: bad curve point '%s'\n", tok.c_str());
                curve.clear();
                return false;
            }
            curve.push_back({(uint32_t) kv, (uint32_t) ub});
        }
        std::sort(curve.begin(), curve.end(), [](const auto & a, const auto & b) { return a.kv < b.kv; });
        if (curve.size() < 2) {
            std::fprintf(stderr, "dyn-batch: curve needs at least two points\n");
            curve.clear();
            return false;
        }
        return true;
    }

    uint32_t target(uint64_t n_kv) const {
        if (curve.empty()) {
            return ub_max;
        }
        if (n_kv <= curve.front().kv) {
            return curve.front().ub;
        }
        if (n_kv >= curve.back().kv) {
            return curve.back().ub;
        }
        for (size_t i = 1; i < curve.size(); i++) {
            if (n_kv <= curve[i].kv) {
                const auto & a = curve[i - 1];
                const auto & b = curve[i];
                const double w = double(n_kv - a.kv) / double(b.kv - a.kv);
                const double v = double(a.ub) + w * (double(b.ub) - double(a.ub));
                return (uint32_t) std::max(1.0, v);
            }
        }
        return curve.back().ub;
    }

    // Auto curve: largest ubatch whose compute buffer plus the transient KVarN
    // prefill window fits in the free VRAM (minus the guard). `compute_mb_per_ub`
    // is the measured scheduler compute buffer divided by the current ubatch.
    uint32_t plan_auto(uint64_t n_kv, int64_t free_mb, double compute_mb_per_ub, uint32_t cur_ub,
                       int64_t extra_reserve_mb = 0) const {
        if (compute_mb_per_ub <= 0.0) {
            return ub_max;
        }
        // free VRAM already accounts for the window allocated so far; the guard
        // is the reserve for the remaining window growth (max 256 MiB by 64k).
        // Using the current free keeps the ubatch maximal for as long as possible
        // and only drops it when the guard is actually reached.
        const double free = free_mb >= 0 ? double(free_mb) : 0.0;
        const double avail = free + compute_mb_per_ub * double(cur_ub);
        const double target = (avail - double(guard_mb) - double(extra_reserve_mb)) / compute_mb_per_ub;
        uint32_t t = (uint32_t) std::max(1.0, target);
        t = (t / 8) * 8;
        t = std::min(t, ub_max);
        t = std::max(t, ub_min);
        return t;
    }

    // Returns the ubatch cap to use at this occupancy.
    uint32_t plan(uint64_t n_kv, int64_t free_mb) const {
        uint32_t t = target(n_kv);
        t = std::min(t, ub_max);
        t = std::max(t, ub_min);
        // emergency: keep the guard budget even if the curve says otherwise
        if (free_mb >= 0 && free_mb < guard_mb) {
            t = std::max(ub_min, t > step ? t - step : ub_min);
        }
        return t;
    }

    // Should the cap be changed from `cur` to `t` now? One scheduler
    // re-reserve per step, so only react when the curve moved by a deadband.
    bool should_apply(uint32_t cur, uint32_t t) const {
        return t + step <= cur || t > cur + step;
    }
};
