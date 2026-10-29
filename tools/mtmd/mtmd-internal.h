#pragma once

#include "mtmd.h"

#include <string>
#include <vector>

// !!! Internal header, to be used by mtmd and its unit tests only !!!

#define MTMD_INTERNAL_HEADER

// bitmap is null for text parts
struct mtmd_internal_part {
    std::string text;
    const mtmd_bitmap * bitmap;
    // only used for text parts
    bool parse_special = false;
    // Non-zero only for bitmap parts produced by expanding a lazy (video) source.
    // Frames emitted by the same lazy callback share a group id; parts with
    // different non-zero groups (e.g. frames of different videos) are never
    // temporally merged. Zero means "no group info", fall back to can_merge_with.
    int merge_group = 0;
};

// [QWEN_VIDEO] merged parts are erased from `parts`, so one group always maps to one part
std::vector<std::vector<const mtmd_bitmap *>> mtmd_group_mergeable_bitmaps(std::vector<mtmd_internal_part> & parts, int n_merge);
