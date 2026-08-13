#pragma once

#include <algorithm>
#include <cstring>
#include <limits>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_blas.hpp"
#include "symldl_v2_cpu_exchange_impl.hpp"
#include "symldl_v2_cpu_hybrid_update_impl.hpp"

template <typename Ftype>
static void symldl_v2_cpu_hybrid_reconstruct_collapsed_raw(
    xLUstruct_t<Ftype> *lu, int_t k, int slot)
{
    if (lu->mycol != lu->symV2PanelRoot(k))
        return;
    int_t local_panel = lu->symV2PanelIndex(k);
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
        ABORT("SymFact V2 CPU hybrid collapsed panel is invalid.");
    xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
    if (panel.isEmpty())
        return;
    int_t width = panel.ncols();
    if (width <= 0 || static_cast<size_t>(k) >=
                          lu->symV2DiagBlocks.size() ||
        lu->symV2DiagBlocks[static_cast<size_t>(k)] == NULL)
        ABORT("SymFact V2 CPU hybrid diagonal data is missing.");
    size_t count = symldl_v2_checked_product(
        static_cast<size_t>(panel.nzrows()), static_cast<size_t>(width),
        "SymFact V2 CPU hybrid raw panel overflows.");
    if (count > lu->symV2CpuRawPanelCapacity)
        ABORT("SymFact V2 CPU hybrid raw panel exceeds workspace.");
    symldl_v2_cpu_gemm<Ftype>(
        "N", "N", panel.nzrows(), width, width, one<Ftype>(),
        panel.val, panel.LDA(),
        lu->symV2DiagBlocks[static_cast<size_t>(k)], width,
        zeroT<Ftype>(), lu->symV2CpuRawPanelBufs[slot], panel.LDA());
}

template <typename Ftype>
static void symldl_v2_cpu_hybrid_pack_partner_raw(
    xLUstruct_t<Ftype> *lu, int_t k, int_t local_panel,
    int destination_pc, int slot)
{
    size_t plan = static_cast<size_t>(local_panel) * lu->Pc +
                  static_cast<size_t>(destination_pc);
    if (plan >= lu->symV2CpuPartnerSendSizes.size() ||
        plan >= lu->symV2CpuPartnerSendOffsets.size() ||
        plan + 1 >= lu->symV2CpuPartnerSegOffsets.size())
        ABORT("SymFact V2 CPU hybrid partner plan is invalid.");
    size_t count = lu->symV2CpuPartnerSendSizes[plan];
    if (count == 0)
        return;
    xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
    int_t width = panel.ncols();
    if (panel.isEmpty() || width <= 0 || count % width != 0 ||
        static_cast<size_t>(k) >= lu->symV2DiagBlocks.size() ||
        lu->symV2DiagBlocks[static_cast<size_t>(k)] == NULL)
        ABORT("SymFact V2 CPU hybrid partner source is invalid.");
    int_t packed_rows = static_cast<int_t>(count / width);
    if (count > lu->symV2CpuRawPanelCapacity)
        ABORT("SymFact V2 CPU hybrid partner scratch exceeds workspace.");
    Ftype *gathered = lu->symV2CpuRawPanelBufs[slot];
    for (size_t segment_id = lu->symV2CpuPartnerSegOffsets[plan];
         segment_id < lu->symV2CpuPartnerSegOffsets[plan + 1];
         ++segment_id)
    {
        const SymLDLV2CpuPackSegment &segment =
            lu->symV2CpuPartnerSegments[segment_id];
        int_t source_row = panel.stRow(segment.source_block);
        for (int_t column = 0; column < width; ++column)
            for (int_t row = 0; row < segment.row_count; ++row)
            {
                int_t source_offset = row;
                if (segment.row_permutation_offset !=
                    SYM_LDL_V2_CPU_CONTIGUOUS_ROWS)
                {
                    size_t permutation = segment.row_permutation_offset +
                                         static_cast<size_t>(row);
                    if (permutation >=
                        lu->symV2CpuPartnerRowPermutations.size())
                        ABORT("SymFact V2 CPU hybrid partner row plan is invalid.");
                    source_offset =
                        lu->symV2CpuPartnerRowPermutations[permutation];
                }
                gathered[segment.packed_row_offset + row +
                         column * packed_rows] =
                    panel.val[source_row + source_offset +
                              column * panel.LDA()];
            }
    }
    Ftype *destination = lu->symV2CpuPartnerSendBufs[slot] +
                         lu->symV2CpuPartnerSendOffsets[plan];
    symldl_v2_cpu_gemm<Ftype>(
        "N", "N", packed_rows, width, width, one<Ftype>(),
        gathered, packed_rows,
        lu->symV2DiagBlocks[static_cast<size_t>(k)], width,
        zeroT<Ftype>(), destination, packed_rows);
}

template <typename Ftype>
static void symldl_v2_cpu_hybrid_assemble_peer(
    xLUstruct_t<Ftype> *lu, int_t k, int slot, int pr,
    int_t local_panel, int source_pc)
{
    size_t source_plan = static_cast<size_t>(k) * lu->Pr + pr;
    size_t peer_slot = static_cast<size_t>(slot) * lu->Pr + pr;
    size_t count = lu->symV2CpuPartnerRecvSizes[source_plan];
    if (count == 0)
        return;
    const Ftype *source_values = NULL;
    if (PNUM(pr, source_pc, lu->grid) == lu->iam)
    {
        size_t send_plan = static_cast<size_t>(local_panel) * lu->Pc +
                           static_cast<size_t>(lu->mycol);
        source_values = lu->symV2CpuPartnerSendBufs[slot] +
                        lu->symV2CpuPartnerSendOffsets[send_plan];
    }
    else
    {
        size_t offset = lu->symV2CpuPartnerRecvOffsets[peer_slot];
        if (offset == std::numeric_limits<size_t>::max())
            ABORT("SymFact V2 CPU hybrid partner offset is invalid.");
        source_values = lu->symV2CpuPartnerRecvBufs[slot] + offset;
    }
    const std::vector<int_t> &source_index =
        lu->symV2PartnerLRecvIndexBySrc[source_plan];
    const std::vector<int_t> &map =
        lu->symV2CpuPartnerAssembleMaps[source_plan];
    const std::vector<int_t> &aggregate_index =
        lu->symV2CpuPartnerAssembledIndex[static_cast<size_t>(k)];
    if (source_index.empty() || aggregate_index.empty() ||
        map.size() % 3 != 0)
        ABORT("SymFact V2 CPU hybrid partner assembly map is invalid.");
    xlpanel_t<Ftype> source_panel(
        const_cast<int_t *>(source_index.data()),
        const_cast<Ftype *>(source_values));
    int_t aggregate_rows = aggregate_index[1];
    int_t width = lu->supersize(k);
    Ftype *aggregate = lu->symV2CpuRawPanelBufs[slot];
    for (size_t piece = 0; piece < map.size(); piece += 3)
    {
        int_t destination_row = map[piece];
        int_t rows = map[piece + 1];
        int_t source_row = map[piece + 2];
        for (int_t column = 0; column < width; ++column)
            std::memcpy(
                aggregate + destination_row + column * aggregate_rows,
                source_values + source_row + column * source_panel.LDA(),
                static_cast<size_t>(rows) * sizeof(Ftype));
    }
}

template <typename Ftype>
static void symldl_v2_cpu_hybrid_reserve_peer(
    xLUstruct_t<Ftype> *lu, int_t k, int slot, int pr, int delta)
{
    size_t source_plan = static_cast<size_t>(k) * lu->Pr + pr;
    if (source_plan >= lu->symV2PartnerLRecvIndexBySrc.size())
        ABORT("SymFact V2 CPU hybrid output plan is invalid.");
    const std::vector<int_t> &index =
        lu->symV2PartnerLRecvIndexBySrc[source_plan];
    if (index.empty())
        return;
    xlpanel_t<Ftype> panel(const_cast<int_t *>(index.data()), NULL);
    for (int_t block = 0; block < panel.nblocks(); ++block)
        symldl_v2_cpu_note_task_pending(
            lu, panel.gid(block), slot, delta);
}

template <typename Ftype>
static void symldl_v2_cpu_issue_hybrid_exchange(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot,
    SymLDLV2CpuExchangeRoute route)
{
    if (route != SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY &&
        route != SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT)
        ABORT("SymFact V2 CPU hybrid exchange route is invalid.");
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuPartnerSendBufs.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuPartnerRecvBufs.size() ||
        lu->symV2CpuSlotRequestCounts[slot] != 0 ||
        lu->symV2CpuExchangeStates[slot].active)
        ABORT("SymFact V2 CPU hybrid slot is not reusable.");
    if (route == SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT &&
        (static_cast<size_t>(slot) >= lu->symV2CpuRowSendBufs.size() ||
         static_cast<size_t>(slot) >= lu->symV2CpuRowRecvBufs.size()))
        ABORT("SymFact V2 CPU hybrid row workspace is missing.");

    double exchange_start = SuperLU_timer_();
    int tag_ub = lu->symFactTagUb;
    if (tag_ub <= 0)
        ABORT("SymFact V2 CPU hybrid MPI tag bound is invalid.");
    int_t source_pc = lu->symV2PanelRoot(k);
    int_t local_panel = lu->symV2PanelIndex(k);
    size_t recv_base = static_cast<size_t>(k) * lu->Pr;
    size_t peer_base = static_cast<size_t>(slot) * lu->Pr;
    if (recv_base + static_cast<size_t>(lu->Pr) >
            lu->symV2CpuPartnerRecvSizes.size() ||
        peer_base + static_cast<size_t>(lu->Pr) >
            lu->symV2CpuPartnerRecvChunksRemaining.size() ||
        peer_base + static_cast<size_t>(lu->Pr) >
            lu->symV2CpuPartnerUpdateSubmitted.size() ||
        peer_base + static_cast<size_t>(lu->Pr) >
            lu->symV2CpuPartnerPeerAssembled.size() ||
        peer_base + static_cast<size_t>(lu->Pr) >
            lu->symV2CpuPartnerRecvOffsets.size())
        ABORT("SymFact V2 CPU hybrid exchange state is undersized.");
    std::fill(lu->symV2CpuPartnerRecvChunksRemaining.begin() + peer_base,
              lu->symV2CpuPartnerRecvChunksRemaining.begin() +
                  peer_base + lu->Pr, 0);
    std::fill(lu->symV2CpuPartnerUpdateSubmitted.begin() + peer_base,
              lu->symV2CpuPartnerUpdateSubmitted.begin() +
                  peer_base + lu->Pr, 0);
    std::fill(lu->symV2CpuPartnerPeerAssembled.begin() + peer_base,
              lu->symV2CpuPartnerPeerAssembled.begin() +
                  peer_base + lu->Pr, 0);
    std::fill(lu->symV2CpuPartnerRecvOffsets.begin() + peer_base,
              lu->symV2CpuPartnerRecvOffsets.begin() + peer_base + lu->Pr,
              std::numeric_limits<size_t>::max());

    size_t request_count = 0;
    size_t partner_recv_total = 0;
    double post_start = SuperLU_timer_();
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        size_t count = lu->symV2CpuPartnerRecvSizes[recv_base + pr];
        int source = PNUM(pr, source_pc, lu->grid);
        if (count == 0 || source == lu->iam)
            continue;
        if (partner_recv_total > lu->symV2CpuPartnerRecvCapacity ||
            count > lu->symV2CpuPartnerRecvCapacity - partner_recv_total)
            ABORT("SymFact V2 CPU hybrid partner receive exceeds workspace.");
        lu->symV2CpuPartnerRecvOffsets[peer_base + pr] =
            partner_recv_total;
        size_t chunks = symldl_v2_cpu_post_receive_chunks(
            lu, slot, request_count,
            lu->symV2CpuPartnerRecvBufs[slot] + partner_recv_total,
            count, source, SLU_MPI_TAG(5, k), lu->grid->comm, pr);
        if (chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 CPU hybrid partner receive has too many chunks.");
        lu->symV2CpuPartnerRecvChunksRemaining[peer_base + pr] =
            static_cast<int>(chunks);
        partner_recv_total += count;
    }

    const std::vector<int_t> &row_index =
        lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
    size_t row_values = 0;
    int row_chunks = 0;
    if (route == SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT && !row_index.empty())
    {
        row_values = symldl_v2_checked_product(
            static_cast<size_t>(row_index[1]),
            static_cast<size_t>(lu->supersize(k)),
            "SymFact V2 CPU hybrid row receive overflows.");
        if (row_values > lu->symV2CpuRowRecvCapacity)
            ABORT("SymFact V2 CPU hybrid row receive exceeds workspace.");
        if (row_values > 0 && lu->mycol != source_pc)
        {
            size_t chunks = symldl_v2_cpu_post_receive_chunks(
                lu, slot, request_count, lu->symV2CpuRowRecvBufs[slot],
                row_values, static_cast<int>(source_pc), SLU_MPI_TAG(5, k),
                lu->grid3d->rscp.comm, -2);
            if (chunks >
                static_cast<size_t>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 CPU hybrid row receive has too many chunks.");
            row_chunks = static_cast<int>(chunks);
        }
    }
    lu->symV2CpuRecvPostTime += SuperLU_timer_() - post_start;
    size_t receive_request_count = request_count;

    if (lu->mycol == source_pc)
    {
        if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
            ABORT("SymFact V2 CPU hybrid source panel is invalid.");
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t send_plan = static_cast<size_t>(local_panel) * lu->Pc + pc;
            size_t count = lu->symV2CpuPartnerSendSizes[send_plan];
            if (count == 0)
                continue;
            bool active = route == SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY;
            for (int pr = 0; pr < lu->Pr && !active; ++pr)
                active = lu->symV2CpuPartnerSendRowActive[
                    send_plan * lu->Pr + pr] != 0;
            if (!active)
                continue;
            double pack_start = SuperLU_timer_();
            symldl_v2_cpu_hybrid_pack_partner_raw(
                lu, k, local_panel, pc, slot);
            lu->symV2CpuPartnerPackTime += SuperLU_timer_() - pack_start;
        }
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t send_plan = static_cast<size_t>(local_panel) * lu->Pc + pc;
            size_t count = lu->symV2CpuPartnerSendSizes[send_plan];
            if (count == 0)
                continue;
            Ftype *buffer = lu->symV2CpuPartnerSendBufs[slot] +
                            lu->symV2CpuPartnerSendOffsets[send_plan];
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                if (route == SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT &&
                    !lu->symV2CpuPartnerSendRowActive[
                        send_plan * lu->Pr + pr])
                    continue;
                int destination = PNUM(pr, pc, lu->grid);
                if (destination == lu->iam)
                    continue;
                double send_start = SuperLU_timer_();
                size_t chunks = symldl_v2_cpu_post_send_chunks(
                    lu, slot, request_count, buffer, count, destination,
                    SLU_MPI_TAG(5, k), lu->grid->comm);
                lu->symV2CpuSendPostTime += SuperLU_timer_() - send_start;
                lu->symV2RouteProfileNotePartnerSend(
                    count, chunks, SUPERLU_MIN(
                        count, static_cast<size_t>(INT_MAX)));
                lu->symV2CpuPartnerBytes += count * sizeof(Ftype);
            }
        }

        if (route == SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT)
            for (int pc = 0; pc < lu->Pc; ++pc)
            {
                size_t send_plan = static_cast<size_t>(local_panel) * lu->Pc + pc;
                size_t count = lu->symV2CpuRowSendSizes[send_plan];
                if (count == 0)
                    continue;
                double pack_start = SuperLU_timer_();
                symldl_v2_cpu_pack_row_destination(
                    lu, local_panel, pc, panel,
                    lu->symV2CpuRowSendBufs[slot]);
                lu->symV2CpuRowPackTime += SuperLU_timer_() - pack_start;
                if (pc == lu->mycol)
                {
                    if (count != row_values)
                        ABORT("SymFact V2 CPU hybrid self row size mismatch.");
                    continue;
                }
                double send_start = SuperLU_timer_();
                size_t chunks = symldl_v2_cpu_post_send_chunks(
                    lu, slot, request_count,
                    lu->symV2CpuRowSendBufs[slot] +
                        lu->symV2CpuRowSendOffsets[send_plan],
                    count, pc, SLU_MPI_TAG(5, k), lu->grid3d->rscp.comm);
                lu->symV2CpuSendPostTime += SuperLU_timer_() - send_start;
                lu->symV2RouteProfileNoteRowSend(
                    count, chunks, SUPERLU_MIN(
                        count, static_cast<size_t>(INT_MAX)));
                lu->symV2CpuRowBytes += count * sizeof(Ftype);
            }
    }

    const std::vector<int_t> &aggregate_index =
        lu->symV2CpuPartnerAssembledIndex[static_cast<size_t>(k)];
    if (!aggregate_index.empty())
    {
        size_t aggregate_values = symldl_v2_checked_product(
            static_cast<size_t>(aggregate_index[1]),
            static_cast<size_t>(lu->supersize(k)),
            "SymFact V2 CPU hybrid aggregate overflows.");
        std::fill(lu->symV2CpuRawPanelBufs[slot],
                  lu->symV2CpuRawPanelBufs[slot] + aggregate_values,
                  zeroT<Ftype>());
    }
    lu->symV2CpuSlotSendBegins[slot] = receive_request_count;
    lu->symV2CpuSlotRequestCounts[slot] = request_count;
    for (int pr = 0; pr < lu->Pr; ++pr)
        symldl_v2_cpu_hybrid_reserve_peer(lu, k, slot, pr, 1);
    symldl_v2_cpu_note_slot_pending(lu, slot, 1);

    SymLDLV2CpuExchangeState &state = lu->symV2CpuExchangeStates[slot];
    state.k = k;
    state.parent = parent;
    state.local_panel = local_panel;
    state.source_pc = static_cast<int>(source_pc);
    state.row_chunks_remaining = row_chunks;
    state.receive_request_count = receive_request_count;
    state.pending_receive_chunks = receive_request_count;
    state.route = route;
    state.active = 1;
    ++lu->symV2CpuExchangeIssues;
    uint64_t active_exchanges = 0;
    for (size_t active_slot = 0;
         active_slot < lu->symV2CpuExchangeStates.size(); ++active_slot)
        active_exchanges +=
            lu->symV2CpuExchangeStates[active_slot].active != 0;
    lu->symV2CpuActiveExchangeHighWater = SUPERLU_MAX(
        lu->symV2CpuActiveExchangeHighWater, active_exchanges);
    lu->symV2CpuFragmentExchangeTime += SuperLU_timer_() - exchange_start;
}

template <typename Ftype>
static bool symldl_v2_cpu_progress_hybrid_exchange(
    xLUstruct_t<Ftype> *lu, int slot, bool blocking)
{
    SymLDLV2CpuExchangeState &state = lu->symV2CpuExchangeStates[slot];
    if (!state.active)
        return false;
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    size_t peer_base = static_cast<size_t>(slot) * lu->Pr;
    bool progressed = false;
    if (state.pending_receive_chunks > 0)
    {
        int completed = 0;
        double mpi_start = SuperLU_timer_();
        if (MPI_Testsome(
                static_cast<int>(state.receive_request_count),
                lu->symV2CpuRequests.data() + request_base, &completed,
                lu->symV2CpuWaitIndices.data(),
                lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
            ABORT("SymFact V2 CPU hybrid receive progress failed.");
        lu->symV2CpuRecvProgressTime += SuperLU_timer_() - mpi_start;
        ++lu->symV2CpuMpiTestsomeCalls;
        if (completed == 0 && blocking)
        {
            mpi_start = SuperLU_timer_();
            if (MPI_Waitsome(
                    static_cast<int>(state.receive_request_count),
                    lu->symV2CpuRequests.data() + request_base, &completed,
                    lu->symV2CpuWaitIndices.data(),
                    lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
                ABORT("SymFact V2 CPU hybrid receive wait failed.");
            lu->symV2CpuRecvWaitTime += SuperLU_timer_() - mpi_start;
            ++lu->symV2CpuMpiWaitsomeCalls;
            ++lu->symV2CpuBlockingProgressCalls;
        }
        if (completed == MPI_UNDEFINED || completed < 0 ||
            static_cast<size_t>(completed) > state.pending_receive_chunks)
            ABORT("SymFact V2 CPU hybrid receive progress is invalid.");
        for (int item = 0; item < completed; ++item)
        {
            int request = lu->symV2CpuWaitIndices[item];
            if (request < 0 || static_cast<size_t>(request) >=
                                   state.receive_request_count)
                ABORT("SymFact V2 CPU hybrid receive completion is invalid.");
            int peer = lu->symV2CpuRequestPeers[request_base + request];
            if (peer >= 0)
            {
                int &remaining =
                    lu->symV2CpuPartnerRecvChunksRemaining[peer_base + peer];
                if (remaining <= 0)
                    ABORT("SymFact V2 CPU hybrid partner completed twice.");
                --remaining;
            }
            else if (peer == -2)
            {
                if (state.row_chunks_remaining <= 0)
                    ABORT("SymFact V2 CPU hybrid row completed twice.");
                --state.row_chunks_remaining;
            }
            else
                ABORT("SymFact V2 CPU hybrid receive peer is invalid.");
        }
        if (completed > 0)
        {
            progressed = true;
            state.pending_receive_chunks -= static_cast<size_t>(completed);
            lu->symV2CpuMpiCompletions += completed;
        }
    }

    const std::vector<int_t> &aggregate_index =
        lu->symV2CpuPartnerAssembledIndex[static_cast<size_t>(state.k)];
    xlpanel_t<Ftype> column_panel;
    if (!aggregate_index.empty())
        column_panel = xlpanel_t<Ftype>(
            const_cast<int_t *>(aggregate_index.data()),
            lu->symV2CpuRawPanelBufs[slot]);
    xlpanel_t<Ftype> row_panel;
    if (state.route == SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY)
        row_panel = lu->lPanelVec[state.local_panel];
    else
    {
        const std::vector<int_t> &row_index =
            lu->symV2RowFragRecvIndex[static_cast<size_t>(state.k)];
        Ftype *row_values = lu->symV2CpuRowRecvBufs[slot];
        if (lu->mycol == state.source_pc && !row_index.empty())
        {
            size_t self_plan = static_cast<size_t>(state.local_panel) *
                                   lu->Pc + lu->mycol;
            row_values = lu->symV2CpuRowSendBufs[slot] +
                         lu->symV2CpuRowSendOffsets[self_plan];
        }
        if (!row_index.empty())
            row_panel = xlpanel_t<Ftype>(
                const_cast<int_t *>(row_index.data()), row_values);
    }

    size_t recv_base = static_cast<size_t>(state.k) * lu->Pr;
    bool single_thread = lu->symV2CpuWorkerCount <= 1;
    bool released_work = false;
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        size_t peer_slot = peer_base + pr;
        size_t source_plan = recv_base + pr;
        size_t count = lu->symV2CpuPartnerRecvSizes[source_plan];
        if (count == 0 ||
            lu->symV2CpuPartnerRecvChunksRemaining[peer_slot] != 0)
            continue;
        if (!lu->symV2CpuPartnerPeerAssembled[peer_slot])
        {
            double assembly_start = SuperLU_timer_();
            symldl_v2_cpu_hybrid_assemble_peer(
                lu, state.k, slot, pr, state.local_panel,
                state.source_pc);
            double assembly_time = SuperLU_timer_() - assembly_start;
            lu->symV2CpuFragmentAssemblyTime += assembly_time;
            lu->symV2CpuHybridAssemblyTime += assembly_time;
            lu->symV2CpuPartnerPeerAssembled[peer_slot] = 1;
            progressed = true;
        }
        if (state.row_chunks_remaining != 0 ||
            lu->symV2CpuPartnerUpdateSubmitted[peer_slot])
            continue;
        double release_start = SuperLU_timer_();
        if (!row_panel.isEmpty() && !column_panel.isEmpty())
        {
            symldl_v2_cpu_hybrid_submit_peer_ranges(
                lu, state.k, state.parent, slot, pr,
                row_panel, column_panel);
            ++lu->symV2CpuReleaseEventsPartner;
            released_work = true;
        }
        symldl_v2_cpu_hybrid_reserve_peer(
            lu, state.k, slot, pr, -1);
        lu->symV2CpuPartnerUpdateSubmitted[peer_slot] = 1;
        ++lu->symV2CpuHybridPeerReleases;
        lu->symV2CpuHybridReleaseTime +=
            SuperLU_timer_() - release_start;
        progressed = true;
        if (single_thread && released_work)
            break;
    }

    if (state.pending_receive_chunks == 0)
    {
        bool all_submitted = true;
        for (int pr = 0; pr < lu->Pr; ++pr)
            if (lu->symV2CpuPartnerRecvSizes[recv_base + pr] != 0 &&
                !lu->symV2CpuPartnerUpdateSubmitted[peer_base + pr])
                all_submitted = false;
        if (all_submitted)
        {
            state = SymLDLV2CpuExchangeState();
            symldl_v2_cpu_note_slot_pending(lu, slot, -1);
            ++lu->symV2CpuExchangeCompletions;
            ++lu->symV2CpuPanelsCompleted;
            progressed = true;
            if (lu->symV2CpuSlotRequestCounts[slot] ==
                lu->symV2CpuSlotSendBegins[slot])
                symldl_v2_cpu_drain_slot_sends(lu, slot);
        }
    }
    return progressed;
}
