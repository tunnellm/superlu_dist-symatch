#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

struct SymLDLV2RowDownDirectRange
{
    int_t start;
    int_t len;
    int chunk_pc;
};

template <typename Ftype>
static void symldl_v2_append_row_down_demand_record(
    xLUstruct_t<Ftype> *lu, std::vector<int_t> &payload, int_t panel,
    int dest_pc, int chunk_pc, const std::vector<int_t> &blocks)
{
    if (blocks.empty())
        return;
    if (blocks.size() >
        static_cast<size_t>(std::numeric_limits<int_t>::max()))
        ABORT("SymFact V2 row-down demand record is too large.");

    payload.push_back(panel);
    payload.push_back(dest_pc);
    payload.push_back(chunk_pc);

    if (superlu_sym_v2_row_l_compressed_plan())
    {
        std::vector<int_t> ranges;
        ranges.reserve(blocks.size());
        int_t start = blocks[0];
        int_t prev = blocks[0];
        if (start < 0)
            ABORT("SymFact V2 row-down compressed demand has invalid block id.");
        for (size_t bi = 1; bi < blocks.size(); ++bi)
        {
            int_t gid = blocks[bi];
            if (gid <= prev)
                ABORT("SymFact V2 row-down compressed demand is not sorted unique.");
            if (prev < std::numeric_limits<int_t>::max() &&
                gid == prev + 1)
            {
                prev = gid;
                continue;
            }
            ranges.push_back(start);
            ranges.push_back(prev - start + 1);
            start = gid;
            prev = gid;
        }
        ranges.push_back(start);
        ranges.push_back(prev - start + 1);

        if (ranges.size() < blocks.size())
        {
            size_t nranges = ranges.size() / 2;
            if (nranges >
                static_cast<size_t>(std::numeric_limits<int_t>::max()))
                ABORT("SymFact V2 row-down compressed demand has too many ranges.");
            payload.push_back(-static_cast<int_t>(nranges));
            payload.insert(payload.end(), ranges.begin(), ranges.end());
            return;
        }
    }

    payload.push_back(static_cast<int_t>(blocks.size()));
    payload.insert(payload.end(), blocks.begin(), blocks.end());
    (void) lu;
}

template <typename Ftype>
static void symldl_v2_build_row_down_slot_segments(
    xLUstruct_t<Ftype> *lu, size_t slot,
    std::vector<SymLDLV2RowDownDirectRange> &ranges,
    std::vector<SymV2RowDownSendSegmentGPU> &out,
    int &value_count_out)
{
    value_count_out = 0;
    if (ranges.empty())
        return;

    int_t lk = static_cast<int_t>(slot / static_cast<size_t>(lu->Pc));
    int_t k0 = lu->symV2PanelGid(lk);
    if (k0 < 0 || k0 >= lu->nsupers)
        ABORT("SymFact V2 row-down send panel is invalid.");
    int_t ksupc = lu->supersize(k0);
    if (ksupc <= 0)
        ABORT("SymFact V2 row-down send panel width is invalid.");

    std::sort(ranges.begin(), ranges.end(),
              [](const SymLDLV2RowDownDirectRange &a,
                 const SymLDLV2RowDownDirectRange &b)
              {
                  if (a.start != b.start)
                      return a.start < b.start;
                  return a.chunk_pc < b.chunk_pc;
              });

    std::vector<SymLDLV2RowDownDirectRange> merged_ranges;
    merged_ranges.reserve(ranges.size());
    size_t requested_blocks = 0;
    for (size_t ri = 0; ri < ranges.size(); ++ri)
    {
        SymLDLV2RowDownDirectRange cur = ranges[ri];
        if (cur.chunk_pc < 0 || cur.chunk_pc >= lu->Pc ||
            cur.start < 0 || cur.len <= 0 ||
            cur.start > std::numeric_limits<int_t>::max() - cur.len)
            ABORT("SymFact V2 row-down request is invalid.");
        int_t cur_end = cur.start + cur.len;
        if (!merged_ranges.empty())
        {
            SymLDLV2RowDownDirectRange &prev = merged_ranges.back();
            int_t prev_end = prev.start + prev.len;
            if (cur.start < prev_end && cur.chunk_pc != prev.chunk_pc)
                ABORT("SymFact V2 row-down request overlaps across chunks.");
            if (cur.chunk_pc == prev.chunk_pc && cur.start <= prev_end)
            {
                if (cur_end > prev_end)
                    prev.len = cur_end - prev.start;
                continue;
            }
        }
        merged_ranges.push_back(cur);
    }
    ranges.swap(merged_ranges);

    for (size_t ri = 0; ri < ranges.size(); ++ri)
    {
        if (static_cast<size_t>(ranges[ri].len) >
            std::numeric_limits<size_t>::max() - requested_blocks)
            ABORT("SymFact V2 row-down request count overflows.");
        requested_blocks += static_cast<size_t>(ranges[ri].len);
    }
    if (requested_blocks == 0)
        return;

    std::vector<SymLDLV2RowDownDirectRange> scan_ranges = ranges;
    std::sort(scan_ranges.begin(), scan_ranges.end(),
              [](const SymLDLV2RowDownDirectRange &a,
                 const SymLDLV2RowDownDirectRange &b)
              {
                  if (a.chunk_pc != b.chunk_pc)
                      return a.chunk_pc < b.chunk_pc;
                  return a.start < b.start;
              });

    struct SourceSegment
    {
        int_t gid;
        int_t nrows;
        size_t map_offset;
    };
    std::vector<SourceSegment> segments;
    segments.reserve(requested_blocks);

    size_t group_begin = 0;
    while (group_begin < scan_ranges.size())
    {
        int chunk_pc = scan_ranges[group_begin].chunk_pc;
        size_t group_end = group_begin + 1;
        while (group_end < scan_ranges.size() &&
               scan_ranges[group_end].chunk_pc == chunk_pc)
            ++group_end;

        size_t flat = static_cast<size_t>(lk) *
                          static_cast<size_t>(lu->Pc) +
                      static_cast<size_t>(chunk_pc);
        if (flat >= lu->symL2LSendMeta.size() ||
            flat >= lu->symV2PartnerLSendSizes.size() ||
            flat >= lu->symV2PartnerLMapOffsets.size())
            ABORT("SymFact V2 row-down source map is invalid.");
        if (lu->symV2PartnerLSendSizes[flat] < 0)
            ABORT("SymFact V2 row-down source map size is invalid.");
        const std::vector<int_t> &meta = lu->symL2LSendMeta[flat];
        size_t map_pos = lu->symV2PartnerLMapOffsets[flat];
        size_t map_end =
            map_pos + static_cast<size_t>(lu->symV2PartnerLSendSizes[flat]);
        if (map_end > lu->symV2PartnerLPackedMaps.size() ||
            map_end < map_pos)
            ABORT("SymFact V2 row-down source map bounds are invalid.");

        auto range_group_contains_gid = [&](int_t gid) -> bool
        {
            size_t lo = group_begin;
            size_t hi = group_end;
            while (lo < hi)
            {
                size_t mid = lo + (hi - lo) / 2;
                if (scan_ranges[mid].start <= gid)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo == group_begin)
                return false;
            const SymLDLV2RowDownDirectRange &r = scan_ranges[lo - 1];
            return gid >= r.start && gid < r.start + r.len;
        };

        size_t meta_pos = 0;
        while (meta_pos < meta.size())
        {
            if (meta_pos + 2 > meta.size())
                ABORT("SymFact V2 row-down metadata is truncated.");
            int_t block_gid = meta[meta_pos++];
            int_t len = meta[meta_pos++];
            if (len < 0 || meta_pos + static_cast<size_t>(len) > meta.size())
                ABORT("SymFact V2 row-down metadata block is invalid.");
            size_t value_count = symldl_v2_checked_product(
                static_cast<size_t>(len), static_cast<size_t>(ksupc),
                "SymFact V2 row-down source map segment overflows.");
            if (map_pos + value_count > map_end || map_pos + value_count < map_pos)
                ABORT("SymFact V2 row-down source map segment is invalid.");
            if (range_group_contains_gid(block_gid))
            {
                SourceSegment seg;
                seg.gid = block_gid;
                seg.nrows = len;
                seg.map_offset = map_pos;
                segments.push_back(seg);
            }
            map_pos += value_count;
            meta_pos += static_cast<size_t>(len);
        }
        if (map_pos != map_end)
            ABORT("SymFact V2 row-down source map size mismatch.");
        group_begin = group_end;
    }

    if (segments.size() != requested_blocks)
        ABORT("SymFact V2 row-down did not find all requested blocks.");
    std::sort(segments.begin(), segments.end(),
              [](const SourceSegment &a, const SourceSegment &b)
              {
                  return a.gid < b.gid;
              });
    for (size_t si = 1; si < segments.size(); ++si)
        if (segments[si - 1].gid == segments[si].gid)
            ABORT("SymFact V2 row-down source block is duplicated.");

    size_t row_count = 0;
    out.reserve(out.size() + segments.size());
    for (size_t si = 0; si < segments.size(); ++si)
    {
        if (segments[si].nrows <= 0 ||
            static_cast<size_t>(segments[si].nrows) >
                std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(ksupc))
            ABORT("SymFact V2 row-down segment has invalid width.");
        size_t block_values =
            static_cast<size_t>(segments[si].nrows) *
            static_cast<size_t>(ksupc);
        if (segments[si].map_offset + block_values >
                lu->symV2PartnerLPackedMaps.size() ||
            segments[si].map_offset + block_values < segments[si].map_offset)
            ABORT("SymFact V2 row-down segment map is invalid.");
        if (row_count >
            std::numeric_limits<size_t>::max() -
                static_cast<size_t>(segments[si].nrows))
            ABORT("SymFact V2 row-down row count overflows.");

        SymV2RowDownSendSegmentGPU gpu_seg;
        gpu_seg.map_offset = segments[si].map_offset;
        gpu_seg.nrows = segments[si].nrows;
        if (row_count >
            static_cast<size_t>(std::numeric_limits<int_t>::max()))
            ABORT("SymFact V2 row-down destination offset is too large.");
        gpu_seg.dst_row_offset = static_cast<int_t>(row_count);
        out.push_back(gpu_seg);
        row_count += static_cast<size_t>(segments[si].nrows);
    }

    size_t value_count = symldl_v2_checked_product(
        row_count, static_cast<size_t>(ksupc),
        "SymFact V2 row-down value count overflows.");
    if (value_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 row-down value count exceeds MPI limit.");
    value_count_out = static_cast<int>(value_count);
}

#endif
