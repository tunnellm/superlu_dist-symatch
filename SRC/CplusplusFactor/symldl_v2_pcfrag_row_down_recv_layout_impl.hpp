#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_pcfrag_cached_block_impl.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
static void symldl_v2_build_row_down_recv_layout(
    xLUstruct_t<Ftype> *lu, size_t row_chunk_count, size_t l2l_slots,
    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        &row_down_blocks_by_panel,
    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        &row_down_recv_blocks)
{
    lu->symV2RowFragRecvSizes.assign(row_chunk_count, 0);
    lu->symV2RowFragRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                     std::vector<int_t>());
    lu->symV2RowFragRecvMap.assign(row_chunk_count, std::vector<int_t>());
    lu->symV2RowFragRecvMapOffsets.assign(row_chunk_count, 0);
    lu->symV2RowFragRecvMapsGPU.assign(row_chunk_count, NULL);
    lu->symV2RowDownRecvSizes.assign(row_chunk_count, 0);
    lu->symV2RowDownSegs.clear();
    lu->symV2RowDownSegOffsets.assign(l2l_slots + 1, 0);

    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        std::vector<SymLDLV2CachedPartnerBlock> &blocks =
            row_down_blocks_by_panel[static_cast<size_t>(k0)];
        if (blocks.empty())
            continue;
        std::sort(blocks.begin(), blocks.end(),
                  [](const SymLDLV2CachedPartnerBlock &a,
                     const SymLDLV2CachedPartnerBlock &b)
                  {
                      return a.gid < b.gid;
                  });
        std::vector<SymLDLV2CachedPartnerBlock> unique_blocks;
        unique_blocks.reserve(blocks.size());
        for (size_t bi = 0; bi < blocks.size(); ++bi)
        {
            if (!unique_blocks.empty() &&
                unique_blocks.back().gid == blocks[bi].gid)
                continue;
            unique_blocks.push_back(blocks[bi]);
        }
        blocks.swap(unique_blocks);

        int_t row_nblocks = static_cast<int_t>(blocks.size());
        int_t row_nrows = 0;
        for (size_t bi = 0; bi < blocks.size(); ++bi)
            row_nrows += static_cast<int_t>(blocks[bi].cols.size());
        int_t row_index_size =
            LPANEL_HEADER_SIZE + 2 * row_nblocks + 1 + row_nrows;
        if (row_index_size > lu->maxSymV2RowFragIdxRecvCount)
            ABORT("SymFact V2 row-down index exceeds receive buffer.");
        if (static_cast<int64_t>(row_nrows) *
                static_cast<int64_t>(lu->supersize(k0)) >
            static_cast<int64_t>(lu->maxSymV2RowFragValRecvCount))
            ABORT("SymFact V2 row-down values exceed receive buffer.");

        std::vector<int_t> &row_index =
            lu->symV2RowFragRecvIndex[static_cast<size_t>(k0)];
        row_index.assign(static_cast<size_t>(row_index_size), 0);
        row_index[0] = row_nblocks;
        row_index[1] = row_nrows;
        row_index[2] = 0;
        row_index[3] = lu->supersize(k0);
        int_t row_gid_ptr = LPANEL_HEADER_SIZE;
        int_t row_px_ptr = LPANEL_HEADER_SIZE + row_nblocks;
        int_t row_data_ptr = LPANEL_HEADER_SIZE + 2 * row_nblocks + 1;
        row_index[row_px_ptr] = 0;
        for (int_t bi = 0; bi < row_nblocks; ++bi)
        {
            row_index[row_gid_ptr + bi] = blocks[static_cast<size_t>(bi)].gid;
            row_index[row_px_ptr + bi + 1] =
                row_index[row_px_ptr + bi] +
                static_cast<int_t>(blocks[static_cast<size_t>(bi)].cols.size());
            for (size_t rc = 0;
                 rc < blocks[static_cast<size_t>(bi)].cols.size(); ++rc)
                row_index[row_data_ptr++] =
                    blocks[static_cast<size_t>(bi)].cols[rc];
        }

        for (int chunk_pc = 0; chunk_pc < lu->Pc; ++chunk_pc)
        {
            size_t recv_pos = static_cast<size_t>(k0) *
                                  static_cast<size_t>(lu->Pc) +
                              static_cast<size_t>(chunk_pc);
            std::vector<SymLDLV2CachedPartnerBlock> &recv_blocks =
                row_down_recv_blocks[recv_pos];
            std::vector<int_t> &recv_map =
                lu->symV2RowFragRecvMap[recv_pos];
            long long expected_values = 0;
            int_t src_offset = 0;
            for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
            {
                std::vector<SymLDLV2CachedPartnerBlock>::const_iterator it =
                    std::lower_bound(
                        blocks.begin(), blocks.end(), recv_blocks[rb].gid,
                        [](const SymLDLV2CachedPartnerBlock &block,
                           int_t gid)
                        {
                            return block.gid < gid;
                        });
                if (it == blocks.end() || it->gid != recv_blocks[rb].gid)
                    ABORT("SymFact V2 row-down receive map cannot find a block.");
                int_t ib = static_cast<int_t>(it - blocks.begin());
                int_t nrows =
                    static_cast<int_t>(recv_blocks[rb].cols.size());
                recv_map.push_back(row_index[row_px_ptr + ib]);
                recv_map.push_back(nrows);
                recv_map.push_back(src_offset);
                SymV2RowDownSeg seg;
                seg.gid = recv_blocks[rb].gid;
                seg.chunk_pc = chunk_pc;
                seg.nrows = nrows;
                seg.dst_row_offset = row_index[row_px_ptr + ib];
                seg.value_count = nrows * lu->supersize(k0);
                seg.map_offset = static_cast<size_t>(src_offset);
                lu->symV2RowDownSegs.push_back(seg);
                src_offset += nrows * lu->supersize(k0);
                expected_values += static_cast<long long>(nrows) *
                                   static_cast<long long>(lu->supersize(k0));
            }
            if (expected_values >
                static_cast<long long>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 row-down receive size exceeds MPI limit.");
            lu->symV2RowFragRecvSizes[recv_pos] =
                static_cast<int>(expected_values);
            lu->symV2RowDownRecvSizes[recv_pos] =
                static_cast<int>(expected_values);
        }
    }
}

#endif
