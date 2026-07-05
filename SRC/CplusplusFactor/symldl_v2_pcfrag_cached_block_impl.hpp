#pragma once

#include <vector>

#include "xlupanels.hpp"

struct SymLDLV2CachedPartnerBlock
{
    int_t gid;
    size_t cols_begin;
    int_t len;
    std::vector<int_t> cols;
};

static inline int_t symldl_v2_cached_block_len(
    const std::vector<int_t> &payload, size_t cols_begin, int_t len,
    const std::vector<int_t> &cols)
{
    (void) payload;
    (void) cols_begin;
    return cols.empty() ? len : static_cast<int_t>(cols.size());
}

static inline int_t symldl_v2_cached_col(
    const std::vector<int_t> &payload,
    const SymLDLV2CachedPartnerBlock &block, size_t pos)
{
    return block.cols.empty() ? payload[block.cols_begin + pos]
                              : block.cols[pos];
}

static inline int_t symldl_v2_cached_len(
    const SymLDLV2CachedPartnerBlock &block)
{
    return block.cols.empty() ? block.len
                              : static_cast<int_t>(block.cols.size());
}
