#pragma once

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_pcfrag_cached_block_impl.hpp"
#include "symldl_v2_pcfrag_partner_metadata_impl.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
static void symldl_v2_build_partner_l_recv_maps(
    xLUstruct_t<Ftype> *lu,
    const SymLDLV2PartnerMetaPayload &meta)
{
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload || lu->Pr <= 1)
        return;

    const bool derive_recv_sizes =
        superlu_sym_v2_scoped_fragment_metadata();
    const bool verify_recv_sizes =
        derive_recv_sizes &&
        superlu_sym_v2_scoped_fragment_metadata_verify();
    std::vector<int> global_recv_sizes;
    if (!derive_recv_sizes || verify_recv_sizes)
    {
        size_t table_count = symldl_v2_checked_product(
            symldl_v2_checked_product(
                static_cast<size_t>(lu->nsupers),
                static_cast<size_t>(lu->Pc),
                "SymFact V2 partner receive table overflows."),
            static_cast<size_t>(lu->Pr),
            "SymFact V2 partner receive table overflows.");
        if (table_count >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 partner receive table exceeds MPI limit.");

        std::vector<int> local_recv_sizes(table_count, 0);
        global_recv_sizes.assign(table_count, 0);
        for (int_t lk = 0; lk < lu->symV2PanelCount(); ++lk)
        {
            int_t k0 = lu->symV2PanelGid(lk);
            if (k0 < 0 || k0 >= lu->nsupers)
                continue;
            for (int pc = 0; pc < lu->Pc; ++pc)
            {
                size_t flat = static_cast<size_t>(lk) *
                                  static_cast<size_t>(lu->Pc) +
                              static_cast<size_t>(pc);
                int size = flat < lu->symV2PartnerLSendSizes.size()
                               ? lu->symV2PartnerLSendSizes[flat]
                               : 0;
                size_t pos = (static_cast<size_t>(k0) *
                                  static_cast<size_t>(lu->Pc) +
                              static_cast<size_t>(pc)) *
                                 static_cast<size_t>(lu->Pr) +
                             static_cast<size_t>(lu->myrow);
                local_recv_sizes[pos] = size;
            }
        }
        MPI_Allreduce(local_recv_sizes.data(), global_recv_sizes.data(),
                      static_cast<int>(table_count), MPI_INT, MPI_SUM,
                      lu->grid->comm);
    }

    const std::vector<int_t> &all_meta_payload = meta.payload;
    const std::vector<size_t> &meta_counts = meta.counts;
    const std::vector<size_t> &meta_displs = meta.displs;
    int comm_size = meta.comm_size;

    size_t compact_count = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 partner compact table overflows.");
    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        cached_partner_blocks(static_cast<size_t>(lu->nsupers));
    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        cached_partner_recv_blocks(compact_count);
    const bool indexed_recv = superlu_sym_v2_recv_map_index();

    for (int r = 0; r < comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(meta_displs[r]);
        size_t rank_end = meta_pos + static_cast<size_t>(meta_counts[r]);
        int source_pr =
            symldl_v2_partner_metadata_source_row(meta, r, lu);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 partner metadata payload is truncated.");
            int_t target_pc = all_meta_payload[meta_pos++];
            int_t k0 = all_meta_payload[meta_pos++];
            int_t meta_len = all_meta_payload[meta_pos++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k0 < 0 || k0 >= lu->nsupers || meta_len < 0 ||
                meta_pos + static_cast<size_t>(meta_len) > rank_end)
                ABORT("SymFact V2 partner metadata payload is invalid.");

            size_t block_pos = meta_pos;
            size_t block_end = meta_pos + static_cast<size_t>(meta_len);
            if (target_pc == lu->mycol)
            {
                while (block_pos < block_end)
                {
                    if (block_pos + 2 > block_end)
                        ABORT("SymFact V2 partner metadata block is truncated.");
                    SymLDLV2CachedPartnerBlock block;
                    block.gid = all_meta_payload[block_pos++];
                    block.len = all_meta_payload[block_pos++];
                    if (block.len < 0 ||
                        block_pos + static_cast<size_t>(block.len) >
                            block_end)
                        ABORT("SymFact V2 partner metadata block is invalid.");
                    block.cols_begin = block_pos;
                    if (!indexed_recv)
                        block.cols.assign(
                            all_meta_payload.begin() + block_pos,
                            all_meta_payload.begin() + block_pos + block.len);
                    block_pos += static_cast<size_t>(block.len);
                    cached_partner_blocks[static_cast<size_t>(k0)]
                        .push_back(block);
                    size_t recv_pos = static_cast<size_t>(k0) *
                                          static_cast<size_t>(lu->Pr) +
                                      static_cast<size_t>(source_pr);
                    cached_partner_recv_blocks[recv_pos].push_back(block);
                }
            }
            meta_pos = block_end;
        }
    }

    lu->symV2PartnerLRecvSizes.assign(compact_count, 0);
    lu->symV2PartnerLRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                      std::vector<int_t>());
    lu->symV2PartnerLRecvIndexBySrc.assign(compact_count,
                                           std::vector<int_t>());
    lu->symV2PartnerLRecvMap.assign(compact_count, std::vector<int_t>());
    lu->symV2PartnerLRecvMapOffsets.assign(compact_count, 0);
    lu->symV2PartnerLRecvMapsGPU.assign(compact_count, NULL);

    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        if (!derive_recv_sizes)
        {
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t src_pos = (static_cast<size_t>(k0) *
                                      static_cast<size_t>(lu->Pc) +
                                  static_cast<size_t>(lu->mycol)) *
                                     static_cast<size_t>(lu->Pr) +
                                 static_cast<size_t>(pr);
                size_t dst_pos = static_cast<size_t>(k0) *
                                     static_cast<size_t>(lu->Pr) +
                                 static_cast<size_t>(pr);
                lu->symV2PartnerLRecvSizes[dst_pos] =
                    global_recv_sizes[src_pos];
            }
        }

        std::vector<SymLDLV2CachedPartnerBlock> &blocks =
            cached_partner_blocks[static_cast<size_t>(k0)];
        if (blocks.empty())
            continue;
        std::sort(blocks.begin(), blocks.end(),
                  [](const SymLDLV2CachedPartnerBlock &a,
                     const SymLDLV2CachedPartnerBlock &b)
                  {
                      return a.gid < b.gid;
                  });

        int_t nblocks = static_cast<int_t>(blocks.size());
        int_t nrows = 0;
        for (size_t ib = 0; ib < blocks.size(); ++ib)
            nrows += symldl_v2_cached_len(blocks[ib]);
        int_t index_size = LPANEL_HEADER_SIZE + 2 * nblocks + 1 + nrows;
        if (index_size > lu->maxSymPartnerLidxCount)
            ABORT("SymFact V2 partner fragment index exceeds staging buffer.");
        if (static_cast<int64_t>(nrows) *
                static_cast<int64_t>(lu->supersize(k0)) >
            static_cast<int64_t>(lu->maxSymPartnerLvalCount))
            ABORT("SymFact V2 partner fragment values exceed staging buffer.");

        std::vector<int_t> &index =
            lu->symV2PartnerLRecvIndex[static_cast<size_t>(k0)];
        index.assign(static_cast<size_t>(index_size), 0);
        index[0] = nblocks;
        index[1] = nrows;
        index[2] = 0;
        index[3] = lu->supersize(k0);
        int_t gid_ptr = LPANEL_HEADER_SIZE;
        int_t px_ptr = LPANEL_HEADER_SIZE + nblocks;
        int_t row_ptr = LPANEL_HEADER_SIZE + 2 * nblocks + 1;
        index[px_ptr] = 0;
        for (int_t ib = 0; ib < nblocks; ++ib)
        {
            index[gid_ptr + ib] = blocks[ib].gid;
            index[px_ptr + ib + 1] =
                index[px_ptr + ib] + symldl_v2_cached_len(blocks[ib]);
            for (int_t j = 0; j < symldl_v2_cached_len(blocks[ib]); ++j)
                index[row_ptr++] = symldl_v2_cached_col(
                    all_meta_payload, blocks[ib], static_cast<size_t>(j));
        }

        std::vector<std::pair<int_t, int_t> > lookup;
        lookup.reserve(static_cast<size_t>(nblocks));
        for (int_t ib = 0; ib < nblocks; ++ib)
        {
            if (!lookup.empty() && lookup.back().first == blocks[ib].gid)
                continue;
            lookup.push_back(std::make_pair(blocks[ib].gid,
                                            index[px_ptr + ib]));
        }
        auto block_offset = [&](int_t gid) -> int_t
        {
            std::vector<std::pair<int_t, int_t> >::const_iterator it =
                std::lower_bound(
                    lookup.begin(), lookup.end(),
                    std::make_pair(gid, static_cast<int_t>(0)),
                    [](const std::pair<int_t, int_t> &a,
                       const std::pair<int_t, int_t> &b)
                    {
                        return a.first < b.first;
                    });
            return (it != lookup.end() && it->first == gid)
                       ? it->second
                       : GLOBAL_BLOCK_NOT_FOUND;
        };

        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            size_t recv_pos = static_cast<size_t>(k0) *
                                  static_cast<size_t>(lu->Pr) +
                              static_cast<size_t>(pr);
            std::vector<SymLDLV2CachedPartnerBlock> &recv_blocks =
                cached_partner_recv_blocks[recv_pos];
            std::vector<int_t> &recv_map =
                lu->symV2PartnerLRecvMap[recv_pos];
            std::vector<int_t> &src_index =
                lu->symV2PartnerLRecvIndexBySrc[recv_pos];
            if (!recv_blocks.empty())
            {
                int_t src_nblocks =
                    static_cast<int_t>(recv_blocks.size());
                int_t src_nrows = 0;
                for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
                    src_nrows += symldl_v2_cached_len(recv_blocks[rb]);
                int_t src_index_size =
                    LPANEL_HEADER_SIZE + 2 * src_nblocks + 1 + src_nrows;
                if (src_index_size > lu->maxSymPartnerLidxCount)
                    ABORT("SymFact V2 partner source index exceeds staging buffer.");
                src_index.assign(static_cast<size_t>(src_index_size), 0);
                src_index[0] = src_nblocks;
                src_index[1] = src_nrows;
                src_index[2] = 0;
                src_index[3] = lu->supersize(k0);
                int_t src_gid_ptr = LPANEL_HEADER_SIZE;
                int_t src_px_ptr = LPANEL_HEADER_SIZE + src_nblocks;
                int_t src_row_ptr =
                    LPANEL_HEADER_SIZE + 2 * src_nblocks + 1;
                src_index[src_px_ptr] = 0;
                for (int_t rb = 0; rb < src_nblocks; ++rb)
                {
                    src_index[src_gid_ptr + rb] =
                        recv_blocks[static_cast<size_t>(rb)].gid;
                    src_index[src_px_ptr + rb + 1] =
                        src_index[src_px_ptr + rb] +
                        symldl_v2_cached_len(
                            recv_blocks[static_cast<size_t>(rb)]);
                    for (int_t rc = 0;
                         rc < symldl_v2_cached_len(
                                  recv_blocks[static_cast<size_t>(rb)]);
                         ++rc)
                        src_index[src_row_ptr++] =
                            symldl_v2_cached_col(
                                all_meta_payload,
                                recv_blocks[static_cast<size_t>(rb)],
                                static_cast<size_t>(rc));
                }
            }

            long long expected_values = 0;
            int_t src_offset = 0;
            for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
            {
                int_t recv_offset = block_offset(recv_blocks[rb].gid);
                if (recv_offset == GLOBAL_BLOCK_NOT_FOUND)
                    ABORT("SymFact V2 partner receive map cannot find a block.");
                int_t rows = symldl_v2_cached_len(recv_blocks[rb]);
                recv_map.push_back(recv_offset);
                recv_map.push_back(rows);
                recv_map.push_back(src_offset);
                src_offset += rows * lu->supersize(k0);
                expected_values += static_cast<long long>(rows) *
                                   static_cast<long long>(lu->supersize(k0));
            }
            if (derive_recv_sizes)
            {
                if (expected_values < 0 ||
                    expected_values > std::numeric_limits<int>::max())
                    ABORT("SymFact V2 partner receive size exceeds MPI limit.");
                lu->symV2PartnerLRecvSizes[recv_pos] =
                    static_cast<int>(expected_values);
                if (verify_recv_sizes)
                {
                    size_t src_pos = (static_cast<size_t>(k0) *
                                          static_cast<size_t>(lu->Pc) +
                                      static_cast<size_t>(lu->mycol)) *
                                         static_cast<size_t>(lu->Pr) +
                                     static_cast<size_t>(pr);
                    if (global_recv_sizes[src_pos] != expected_values)
                        ABORT("SymFact V2 scoped partner receive size mismatch.");
                }
            }
            if (expected_values !=
                static_cast<long long>(lu->symV2PartnerLRecvSizes[recv_pos]))
                ABORT("SymFact V2 partner receive map size mismatch.");
        }
    }

    size_t total_recv_map = 0;
    for (size_t pos = 0; pos < lu->symV2PartnerLRecvMap.size(); ++pos)
    {
        lu->symV2PartnerLRecvMapOffsets[pos] = total_recv_map;
        size_t map_size = lu->symV2PartnerLRecvMap[pos].size();
        if (total_recv_map >
            std::numeric_limits<size_t>::max() - map_size)
            ABORT("SymFact V2 partner receive map size overflows.");
        total_recv_map += map_size;
    }
    lu->symV2PartnerLRecvMapPoolCount = total_recv_map;
}

#endif
