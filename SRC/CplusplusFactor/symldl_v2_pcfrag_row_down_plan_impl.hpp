#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_pcfrag_partner_metadata_impl.hpp"

#ifdef HAVE_CUDA

#include "symldl_v2_pcfrag_row_down_exchange_impl.hpp"
#include "symldl_v2_pcfrag_row_down_recv_layout_impl.hpp"

template <typename Ftype>
static void symldl_v2_build_row_down_maps(
    xLUstruct_t<Ftype> *lu,
    const SymLDLV2PartnerMetaPayload &source_row_meta,
    const SymLDLV2PartnerMetaPayload &target_column_meta)
{
    const bool gpu_pc_fragment =
        lu->symV2UsesGpuFactor() &&
        symldl_v2_use_gpu_pc_fragment_schur(lu->grid3d);
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload ||
        !gpu_pc_fragment)
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

    const std::vector<int_t> &source_row_payload = source_row_meta.payload;
    const std::vector<int_t> &target_column_payload =
        target_column_meta.payload;

    std::vector<std::vector<int_t> >
        row_blocks_by_panel(static_cast<size_t>(lu->nsupers));
    for (int r = 0; r < source_row_meta.comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(source_row_meta.displs[r]);
        size_t rank_end = meta_pos +
                          static_cast<size_t>(source_row_meta.counts[r]);
        int source_pr =
            symldl_v2_partner_metadata_source_row(source_row_meta, r, lu);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 row-down metadata payload is truncated.");
            int_t target_pc = source_row_payload[meta_pos++];
            int_t k0 = source_row_payload[meta_pos++];
            int_t meta_len = source_row_payload[meta_pos++];
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
                    int_t gid = source_row_payload[block_pos++];
                    int_t len = source_row_payload[block_pos++];
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
    for (int r = 0; r < target_column_meta.comm_size; ++r)
    {
        size_t meta_pos =
            static_cast<size_t>(target_column_meta.displs[r]);
        size_t rank_end = meta_pos +
                          static_cast<size_t>(
                              target_column_meta.counts[r]);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 row-down metadata payload is truncated.");
            int_t target_pc = target_column_payload[meta_pos++];
            int_t k0 = target_column_payload[meta_pos++];
            int_t meta_len = target_column_payload[meta_pos++];
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
                    int_t gj = target_column_payload[block_pos++];
                    int_t len = target_column_payload[block_pos++];
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
    for (int r = 0; r < source_row_meta.comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(source_row_meta.displs[r]);
        size_t rank_end = meta_pos +
                          static_cast<size_t>(source_row_meta.counts[r]);
        int source_pr =
            symldl_v2_partner_metadata_source_row(source_row_meta, r, lu);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 row-down metadata payload is truncated.");
            int_t target_pc = source_row_payload[meta_pos++];
            int_t k0 = source_row_payload[meta_pos++];
            int_t meta_len = source_row_payload[meta_pos++];
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
                        block.gid = source_row_payload[block_pos++];
                        block.len = source_row_payload[block_pos++];
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
                                source_row_payload.begin() + block_pos,
                                source_row_payload.begin() + block_pos +
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

    symldl_v2_build_row_down_recv_layout(
        lu, row_chunk_count, l2l_slots, row_down_blocks_by_panel,
        row_down_recv_blocks);

    std::vector<std::vector<SymLDLV2RowDownDirectRange> >
        slot_direct_ranges =
            symldl_v2_exchange_row_down_demands(lu, row_down_payloads,
                                                l2l_slots);

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

    symldl_v2_exchange_row_down_size_replies(lu, size_reply_payloads);

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
