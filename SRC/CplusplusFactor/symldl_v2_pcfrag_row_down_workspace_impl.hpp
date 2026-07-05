#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

#include "symldl_v2_pcfrag_row_down_segments_impl.hpp"

template <typename Ftype>
static void symldl_v2_build_row_down_maps(xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload ||
        lu->Pr <= 1 || lu->Pc <= 1 ||
        !superlu_sym_v2_pc_fragment_schur())
        return;

    if (!superlu_sym_v2_pc_fragment_ldl_native() ||
        !superlu_sym_v2_row_l_plan_v2() ||
        !superlu_sym_v2_row_l_plan_v2_exchange() ||
        !superlu_sym_v2_row_l_direct_recv() ||
        !superlu_sym_v2_row_l_compressed_plan() ||
        !superlu_sym_v2_row_l_lazy_sendmap() ||
        !superlu_sym_v2_row_l_pack_all_dest() ||
        !superlu_sym_v2_row_l_separate_send_staging())
        ABORT("SymFact V2 Pc-fragment row-down exchange requires the LDL-native lazy row-down path.");
    if (superlu_sym_v2_row_l_plan_v2_dryrun())
        ABORT("SymFact V2 row-down exchange requires the active row-down plan.");

    int row_comm_size = 0;
    int row_comm_rank = -1;
    MPI_Comm_size(lu->grid3d->rscp.comm, &row_comm_size);
    MPI_Comm_rank(lu->grid3d->rscp.comm, &row_comm_rank);
    if (row_comm_size != lu->Pc || row_comm_rank != lu->mycol)
        ABORT("SymFact V2 row-down planner found an unexpected row communicator.");

    size_t row_chunk_count = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pc),
        "SymFact V2 row-down table size overflows.");
    size_t l2l_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->symV2PanelCount()),
        static_cast<size_t>(lu->Pc),
        "SymFact V2 row-down send table size overflows.");

    SymLDLV2PartnerMetaPayload meta =
        symldl_v2_collect_partner_l_metadata(lu);
    const std::vector<int_t> &all_meta_payload = meta.payload;

    std::vector<std::vector<int_t> >
        row_blocks_by_panel(static_cast<size_t>(lu->nsupers));
    for (int r = 0; r < meta.comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(meta.displs[r]);
        size_t rank_end = meta_pos + static_cast<size_t>(meta.counts[r]);
        int source_pr = MYROW(r, lu->grid);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 row-down metadata payload is truncated.");
            int_t target_pc = all_meta_payload[meta_pos++];
            int_t k0 = all_meta_payload[meta_pos++];
            int_t meta_len = all_meta_payload[meta_pos++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k0 < 0 || k0 >= lu->nsupers || meta_len < 0 ||
                meta_pos + static_cast<size_t>(meta_len) > rank_end)
                ABORT("SymFact V2 row-down metadata payload is invalid.");
            size_t block_pos = meta_pos;
            size_t block_end = meta_pos + static_cast<size_t>(meta_len);
            if (source_pr == lu->myrow)
            {
                std::vector<int_t> &row_blocks =
                    row_blocks_by_panel[static_cast<size_t>(k0)];
                while (block_pos < block_end)
                {
                    if (block_pos + 2 > block_end)
                        ABORT("SymFact V2 row-down metadata block is truncated.");
                    int_t gid = all_meta_payload[block_pos++];
                    int_t len = all_meta_payload[block_pos++];
                    if (len < 0 ||
                        block_pos + static_cast<size_t>(len) > block_end)
                        ABORT("SymFact V2 row-down metadata block has invalid length.");
                    row_blocks.push_back(gid);
                    block_pos += static_cast<size_t>(len);
                }
            }
            meta_pos += static_cast<size_t>(meta_len);
        }
    }

    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        std::vector<int_t> &row_blocks =
            row_blocks_by_panel[static_cast<size_t>(k0)];
        if (row_blocks.empty())
            continue;
        std::sort(row_blocks.begin(), row_blocks.end());
        row_blocks.erase(std::unique(row_blocks.begin(), row_blocks.end()),
                         row_blocks.end());
    }

    std::vector<std::vector<int_t> >
        needed_row_blocks_by_panel(static_cast<size_t>(lu->nsupers));
    for (int r = 0; r < meta.comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(meta.displs[r]);
        size_t rank_end = meta_pos + static_cast<size_t>(meta.counts[r]);
        int source_pr = MYROW(r, lu->grid);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 row-down metadata payload is truncated.");
            int_t target_pc = all_meta_payload[meta_pos++];
            int_t k0 = all_meta_payload[meta_pos++];
            int_t meta_len = all_meta_payload[meta_pos++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k0 < 0 || k0 >= lu->nsupers || meta_len < 0 ||
                meta_pos + static_cast<size_t>(meta_len) > rank_end)
                ABORT("SymFact V2 row-down metadata payload is invalid.");

            size_t block_pos = meta_pos;
            size_t block_end = meta_pos + static_cast<size_t>(meta_len);
            if (target_pc == lu->mycol)
            {
                const std::vector<int_t> &row_blocks =
                    row_blocks_by_panel[static_cast<size_t>(k0)];
                while (block_pos < block_end)
                {
                    if (block_pos + 2 > block_end)
                        ABORT("SymFact V2 row-down metadata block is truncated.");
                    int_t gj = all_meta_payload[block_pos++];
                    int_t len = all_meta_payload[block_pos++];
                    if (len < 0 ||
                        block_pos + static_cast<size_t>(len) > block_end)
                        ABORT("SymFact V2 row-down metadata block has invalid length.");
                    if (!row_blocks.empty() &&
                        lu->symV2PanelRoot(gj) == lu->mycol)
                    {
                        int_t lj = lu->symV2PanelIndex(gj);
                        if (lj >= 0)
                        {
                            xlpanel_t<Ftype> &dst_panel = lu->lPanelVec[lj];
                            if (!dst_panel.isEmpty())
                            {
                                std::vector<int_t>::const_iterator it =
                                    std::lower_bound(row_blocks.begin(),
                                                     row_blocks.end(), gj);
                                for (; it != row_blocks.end(); ++it)
                                {
                                    if (dst_panel.find(*it) !=
                                        GLOBAL_BLOCK_NOT_FOUND)
                                        needed_row_blocks_by_panel[
                                            static_cast<size_t>(k0)]
                                            .push_back(*it);
                                }
                            }
                        }
                    }
                    block_pos += static_cast<size_t>(len);
                }
            }
            meta_pos = block_end;
        }
    }

    std::vector<std::vector<int_t> > requested_by_chunk(row_chunk_count);
    std::vector<std::vector<int_t> >
        row_down_payloads(static_cast<size_t>(lu->Pc));
    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        std::vector<int_t> &needed =
            needed_row_blocks_by_panel[static_cast<size_t>(k0)];
        if (needed.empty())
            continue;
        std::sort(needed.begin(), needed.end());
        needed.erase(std::unique(needed.begin(), needed.end()),
                     needed.end());
        int source_pc = lu->symV2PanelRoot(k0);
        if (source_pc < 0 || source_pc >= lu->Pc)
            ABORT("SymFact V2 row-down source process column is invalid.");
        for (size_t bi = 0; bi < needed.size(); ++bi)
        {
            int_t gid = needed[bi];
            int chunk_pc = lu->symV2PanelRoot(gid);
            if (chunk_pc < 0 || chunk_pc >= lu->Pc)
                ABORT("SymFact V2 row-down chunk process column is invalid.");
            requested_by_chunk[static_cast<size_t>(k0) *
                                   static_cast<size_t>(lu->Pc) +
                               static_cast<size_t>(chunk_pc)]
                .push_back(gid);
        }
        for (int chunk_pc = 0; chunk_pc < lu->Pc; ++chunk_pc)
        {
            std::vector<int_t> &blocks =
                requested_by_chunk[static_cast<size_t>(k0) *
                                       static_cast<size_t>(lu->Pc) +
                                   static_cast<size_t>(chunk_pc)];
            if (blocks.empty())
                continue;
            symldl_v2_append_row_down_demand_record(
                lu, row_down_payloads[static_cast<size_t>(source_pc)],
                k0, lu->mycol, chunk_pc, blocks);
        }
    }

    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        row_down_blocks_by_panel(static_cast<size_t>(lu->nsupers));
    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        row_down_recv_blocks(row_chunk_count);
    for (int r = 0; r < meta.comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(meta.displs[r]);
        size_t rank_end = meta_pos + static_cast<size_t>(meta.counts[r]);
        int source_pr = MYROW(r, lu->grid);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 row-down metadata payload is truncated.");
            int_t target_pc = all_meta_payload[meta_pos++];
            int_t k0 = all_meta_payload[meta_pos++];
            int_t meta_len = all_meta_payload[meta_pos++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k0 < 0 || k0 >= lu->nsupers || meta_len < 0 ||
                meta_pos + static_cast<size_t>(meta_len) > rank_end)
                ABORT("SymFact V2 row-down metadata payload is invalid.");
            size_t block_pos = meta_pos;
            size_t block_end = meta_pos + static_cast<size_t>(meta_len);
            if (source_pr == lu->myrow)
            {
                size_t req_pos = static_cast<size_t>(k0) *
                                     static_cast<size_t>(lu->Pc) +
                                 static_cast<size_t>(target_pc);
                const std::vector<int_t> &requested =
                    requested_by_chunk[req_pos];
                if (!requested.empty())
                {
                    while (block_pos < block_end)
                    {
                        if (block_pos + 2 > block_end)
                            ABORT("SymFact V2 row-down metadata block is truncated.");
                        SymLDLV2CachedPartnerBlock block;
                        block.gid = all_meta_payload[block_pos++];
                        block.len = all_meta_payload[block_pos++];
                        if (block.len < 0 ||
                            block_pos + static_cast<size_t>(block.len) >
                                block_end)
                            ABORT("SymFact V2 row-down metadata block has invalid length.");
                        if (std::binary_search(requested.begin(),
                                               requested.end(),
                                               block.gid))
                        {
                            block.cols_begin = block_pos;
                            block.cols.assign(
                                all_meta_payload.begin() + block_pos,
                                all_meta_payload.begin() + block_pos +
                                    block.len);
                            row_down_blocks_by_panel[static_cast<size_t>(k0)]
                                .push_back(block);
                            row_down_recv_blocks[req_pos].push_back(block);
                        }
                        block_pos += static_cast<size_t>(block.len);
                    }
                }
            }
            meta_pos = block_end;
        }
    }

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

    std::vector<int> send_counts(static_cast<size_t>(lu->Pc), 0);
    std::vector<int> recv_counts(static_cast<size_t>(lu->Pc), 0);
    std::vector<int> send_displs(static_cast<size_t>(lu->Pc), 0);
    std::vector<int> recv_displs(static_cast<size_t>(lu->Pc), 0);
    int total_send_count = 0;
    for (int pc = 0; pc < lu->Pc; ++pc)
    {
        if (row_down_payloads[static_cast<size_t>(pc)].size() >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 row-down demand payload exceeds MPI limit.");
        send_displs[static_cast<size_t>(pc)] = total_send_count;
        send_counts[static_cast<size_t>(pc)] =
            static_cast<int>(row_down_payloads[static_cast<size_t>(pc)].size());
        total_send_count += send_counts[static_cast<size_t>(pc)];
    }
    std::vector<int_t> send_payload(static_cast<size_t>(total_send_count));
    for (int pc = 0; pc < lu->Pc; ++pc)
        std::copy(row_down_payloads[static_cast<size_t>(pc)].begin(),
                  row_down_payloads[static_cast<size_t>(pc)].end(),
                  send_payload.begin() + send_displs[static_cast<size_t>(pc)]);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, lu->grid3d->rscp.comm);
    int total_recv_count = 0;
    for (int pc = 0; pc < lu->Pc; ++pc)
    {
        if (recv_counts[static_cast<size_t>(pc)] < 0)
            ABORT("SymFact V2 row-down demand receive count is invalid.");
        recv_displs[static_cast<size_t>(pc)] = total_recv_count;
        total_recv_count += recv_counts[static_cast<size_t>(pc)];
    }
    std::vector<int_t> recv_payload(static_cast<size_t>(total_recv_count));
    MPI_Alltoallv(send_payload.empty() ? NULL : send_payload.data(),
                  send_counts.data(), send_displs.data(), mpi_int_t,
                  recv_payload.empty() ? NULL : recv_payload.data(),
                  recv_counts.data(), recv_displs.data(), mpi_int_t,
                  lu->grid3d->rscp.comm);

    std::vector<std::vector<SymLDLV2RowDownDirectRange> >
        slot_direct_ranges(l2l_slots);
    for (int sender_pc = 0; sender_pc < lu->Pc; ++sender_pc)
    {
        size_t pos = static_cast<size_t>(recv_displs[static_cast<size_t>(sender_pc)]);
        size_t end = pos + static_cast<size_t>(recv_counts[static_cast<size_t>(sender_pc)]);
        while (pos < end)
        {
            if (pos + 4 > end)
                ABORT("SymFact V2 row-down demand record is truncated.");
            int_t k0 = recv_payload[pos++];
            int dest_pc = static_cast<int>(recv_payload[pos++]);
            int chunk_pc = static_cast<int>(recv_payload[pos++]);
            int_t encoded_count = recv_payload[pos++];
            if (encoded_count == 0 ||
                encoded_count == std::numeric_limits<int_t>::min())
                ABORT("SymFact V2 row-down demand record is invalid.");
            bool range_encoded = encoded_count < 0;
            int_t encoded_items =
                range_encoded ? -encoded_count : encoded_count;
            if (encoded_items <= 0)
                ABORT("SymFact V2 row-down demand record is invalid.");
            size_t payload_items = static_cast<size_t>(encoded_items);
            if (range_encoded)
            {
                if (payload_items > std::numeric_limits<size_t>::max() / 2)
                    ABORT("SymFact V2 row-down compressed demand overflows.");
                payload_items *= 2;
            }
            if (k0 < 0 || k0 >= lu->nsupers ||
                dest_pc < 0 || dest_pc >= lu->Pc ||
                chunk_pc < 0 || chunk_pc >= lu->Pc ||
                payload_items > end - pos)
                ABORT("SymFact V2 row-down demand record is invalid.");
            if (dest_pc != sender_pc || lu->symV2PanelRoot(k0) != lu->mycol)
                ABORT("SymFact V2 row-down demand arrived at the wrong source column.");

            size_t request_pos = pos;
            size_t request_end = pos + payload_items;
            pos = request_end;
            int_t lk = lu->symV2PanelIndex(k0);
            if (lk < 0)
                continue;
            size_t slot = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(dest_pc);
            if (slot >= slot_direct_ranges.size())
                ABORT("SymFact V2 row-down demand slot is invalid.");

            std::vector<SymLDLV2RowDownDirectRange> &ranges =
                slot_direct_ranges[slot];
            auto append_range = [&](int_t start, int_t len)
            {
                if (start < 0 || len <= 0 ||
                    start > std::numeric_limits<int_t>::max() - len)
                    ABORT("SymFact V2 row-down demand range is invalid.");
                SymLDLV2RowDownDirectRange range;
                range.start = start;
                range.len = len;
                range.chunk_pc = chunk_pc;
                ranges.push_back(range);
            };

            if (range_encoded)
            {
                int_t previous_end = 0;
                bool have_previous = false;
                for (size_t rp = request_pos; rp < request_end; rp += 2)
                {
                    int_t start = recv_payload[rp];
                    int_t len = recv_payload[rp + 1];
                    if (have_previous && start < previous_end)
                        ABORT("SymFact V2 row-down compressed demand ranges overlap or are unsorted.");
                    append_range(start, len);
                    previous_end = start + len;
                    have_previous = true;
                }
            }
            else
            {
                int_t run_start = recv_payload[request_pos];
                int_t prev = run_start;
                if (run_start < 0)
                    ABORT("SymFact V2 row-down explicit demand has invalid block id.");
                for (size_t ri = request_pos + 1; ri < request_end; ++ri)
                {
                    int_t gid = recv_payload[ri];
                    if (gid <= prev)
                        ABORT("SymFact V2 row-down explicit demand is not sorted unique.");
                    if (prev < std::numeric_limits<int_t>::max() &&
                        gid == prev + 1)
                    {
                        prev = gid;
                        continue;
                    }
                    append_range(run_start, prev - run_start + 1);
                    run_start = gid;
                    prev = gid;
                }
                append_range(run_start, prev - run_start + 1);
            }
        }
    }

    std::vector<std::vector<SymV2RowDownSendSegmentGPU> >
        slot_send_segments(l2l_slots);
    std::vector<int> slot_send_segment_values(l2l_slots, 0);
    for (size_t slot = 0; slot < slot_direct_ranges.size(); ++slot)
        symldl_v2_build_row_down_slot_segments(
            lu, slot, slot_direct_ranges[slot],
            slot_send_segments[slot], slot_send_segment_values[slot]);

    lu->symV2RowDownSendSegsHost.clear();
    std::vector<std::vector<int_t> >
        size_reply_payloads(static_cast<size_t>(lu->Pc));
    for (size_t slot = 0; slot < slot_send_segments.size(); ++slot)
    {
        if (slot >= lu->symV2RowDownSendSizes.size() ||
            slot >= lu->symV2RowDownSendSegOffsets.size() ||
            slot >= lu->symV2RowDownSendSegCounts.size())
            ABORT("SymFact V2 row-down send table is invalid.");
        lu->symV2RowDownSendSegOffsets[slot] =
            lu->symV2RowDownSendSegsHost.size();
        if (slot_send_segments[slot].size() >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 row-down has too many send segments.");
        lu->symV2RowDownSendSegCounts[slot] =
            static_cast<int>(slot_send_segments[slot].size());
        int send_size = slot_send_segment_values[slot];
        if (send_size < 0)
            ABORT("SymFact V2 row-down send size is invalid.");
        lu->symV2RowDownSendSizes[slot] = send_size;
        lu->symV2RowDownSendSegsHost.insert(
            lu->symV2RowDownSendSegsHost.end(),
            slot_send_segments[slot].begin(),
            slot_send_segments[slot].end());
        if (send_size > 0)
        {
            int_t lk = static_cast<int_t>(slot / static_cast<size_t>(lu->Pc));
            int dest_pc = static_cast<int>(slot % static_cast<size_t>(lu->Pc));
            int_t k0 = lu->symV2PanelGid(lk);
            size_reply_payloads[static_cast<size_t>(dest_pc)].push_back(k0);
            size_reply_payloads[static_cast<size_t>(dest_pc)].push_back(
                static_cast<int_t>(send_size));
        }
    }
    lu->symV2RowDownSendSegPoolCount =
        lu->symV2RowDownSendSegsHost.size();

    std::vector<int> size_send_counts(static_cast<size_t>(lu->Pc), 0);
    std::vector<int> size_recv_counts(static_cast<size_t>(lu->Pc), 0);
    std::vector<int> size_send_displs(static_cast<size_t>(lu->Pc), 0);
    std::vector<int> size_recv_displs(static_cast<size_t>(lu->Pc), 0);
    int total_size_send = 0;
    for (int pc = 0; pc < lu->Pc; ++pc)
    {
        if (size_reply_payloads[static_cast<size_t>(pc)].size() >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 row-down size payload exceeds MPI limit.");
        size_send_displs[static_cast<size_t>(pc)] = total_size_send;
        size_send_counts[static_cast<size_t>(pc)] =
            static_cast<int>(size_reply_payloads[static_cast<size_t>(pc)].size());
        total_size_send += size_send_counts[static_cast<size_t>(pc)];
    }
    std::vector<int_t> size_send_payload(static_cast<size_t>(total_size_send));
    for (int pc = 0; pc < lu->Pc; ++pc)
        std::copy(size_reply_payloads[static_cast<size_t>(pc)].begin(),
                  size_reply_payloads[static_cast<size_t>(pc)].end(),
                  size_send_payload.begin() +
                      size_send_displs[static_cast<size_t>(pc)]);
    MPI_Alltoall(size_send_counts.data(), 1, MPI_INT,
                 size_recv_counts.data(), 1, MPI_INT,
                 lu->grid3d->rscp.comm);
    int total_size_recv = 0;
    for (int pc = 0; pc < lu->Pc; ++pc)
    {
        if (size_recv_counts[static_cast<size_t>(pc)] < 0)
            ABORT("SymFact V2 row-down size receive count is invalid.");
        size_recv_displs[static_cast<size_t>(pc)] = total_size_recv;
        total_size_recv += size_recv_counts[static_cast<size_t>(pc)];
    }
    std::vector<int_t> size_recv_payload(static_cast<size_t>(total_size_recv));
    MPI_Alltoallv(size_send_payload.empty() ? NULL : size_send_payload.data(),
                  size_send_counts.data(), size_send_displs.data(), mpi_int_t,
                  size_recv_payload.empty() ? NULL : size_recv_payload.data(),
                  size_recv_counts.data(), size_recv_displs.data(), mpi_int_t,
                  lu->grid3d->rscp.comm);
    for (int source_pc = 0; source_pc < lu->Pc; ++source_pc)
    {
        size_t pos = static_cast<size_t>(
            size_recv_displs[static_cast<size_t>(source_pc)]);
        size_t end = pos + static_cast<size_t>(
            size_recv_counts[static_cast<size_t>(source_pc)]);
        while (pos < end)
        {
            if (pos + 2 > end)
                ABORT("SymFact V2 row-down size reply is truncated.");
            int_t k0 = size_recv_payload[pos++];
            int_t send_size = size_recv_payload[pos++];
            if (k0 < 0 || k0 >= lu->nsupers || send_size < 0)
                ABORT("SymFact V2 row-down size reply is invalid.");
            long long expected = 0;
            for (int chunk_pc = 0; chunk_pc < lu->Pc; ++chunk_pc)
                expected += lu->symV2RowFragRecvSizes[
                    static_cast<size_t>(k0) * static_cast<size_t>(lu->Pc) +
                    static_cast<size_t>(chunk_pc)];
            if (source_pc == lu->symV2PanelRoot(k0) &&
                expected != static_cast<long long>(send_size))
                ABORT("SymFact V2 row-down send/receive size mismatch.");
        }
    }

    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        int_t needed_count = static_cast<int_t>(
            needed_row_blocks_by_panel[static_cast<size_t>(k0)].size());
        if (needed_count > 0)
            lu->symV2RowDownPlanReady[static_cast<size_t>(k0)] = 1;
        symldl_v2_trace_pcfrag_plan(lu, "plan", k0, needed_count);
    }

    size_t total_recv_map = 0;
    for (size_t pos = 0; pos < lu->symV2RowFragRecvMap.size(); ++pos)
    {
        lu->symV2RowFragRecvMapOffsets[pos] = total_recv_map;
        size_t map_size = lu->symV2RowFragRecvMap[pos].size();
        if (total_recv_map >
            std::numeric_limits<size_t>::max() - map_size)
            ABORT("SymFact V2 row-fragment receive map size overflows.");
        total_recv_map += map_size;
    }
    lu->symV2RowFragRecvMapPoolCount = total_recv_map;
}


#endif
