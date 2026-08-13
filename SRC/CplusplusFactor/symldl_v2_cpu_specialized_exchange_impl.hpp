#pragma once

#include <algorithm>
#include <cstring>
#include <limits>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_blas.hpp"
#include "symldl_v2_cpu_exchange_impl.hpp"

// Topology specialization changes only how each panel's Schur operands become
// ready.  All numerical work is still released through the common CPU task
// submission policy in symldl_v2_cpu_update_impl.hpp.

template <typename Ftype>
static void symldl_v2_cpu_reserve_column_outputs(
    xLUstruct_t<Ftype> *lu, int_t k, int slot, int pr, int delta)
{
    size_t receive_slot = static_cast<size_t>(k) * lu->Pr + pr;
    if (receive_slot >= lu->symV2PartnerLRecvIndexBySrc.size())
        ABORT("SymFact V2 CPU specialized output plan is invalid.");
    const std::vector<int_t> &index =
        lu->symV2PartnerLRecvIndexBySrc[receive_slot];
    if (index.empty())
        return;
    xlpanel_t<Ftype> panel(const_cast<int_t *>(index.data()), NULL);
    for (int_t block = 0; block < panel.nblocks(); ++block)
        symldl_v2_cpu_note_task_pending(
            lu, panel.gid(block), slot, delta);
}

template <typename Ftype>
static void symldl_v2_cpu_issue_pc1_exchange(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot)
{
    if (lu->Pr <= 1 || lu->Pc != 1)
        ABORT("SymFact V2 CPU Pc=1 route has an invalid process grid.");
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuPartnerSendBufs.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuPartnerRecvBufs.size())
        ABORT("SymFact V2 CPU Pc=1 exchange slot is invalid.");
    if (lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] != 0 ||
        lu->symV2CpuExchangeStates[static_cast<size_t>(slot)].active)
        ABORT("SymFact V2 CPU Pc=1 slot still owns an exchange.");

    double start = SuperLU_timer_();
    int tag_ub = lu->symFactTagUb;
    int_t local_panel = lu->symV2PanelIndex(k);
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
        ABORT("SymFact V2 CPU Pc=1 source panel is invalid.");
    xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];

    size_t recv_base = static_cast<size_t>(k) * lu->Pr;
    size_t peer_base = static_cast<size_t>(slot) * lu->Pr;
    std::fill(lu->symV2CpuPartnerRecvChunksRemaining.begin() + peer_base,
              lu->symV2CpuPartnerRecvChunksRemaining.begin() +
                  peer_base + lu->Pr, 0);
    std::fill(lu->symV2CpuPartnerUpdateSubmitted.begin() + peer_base,
              lu->symV2CpuPartnerUpdateSubmitted.begin() +
                  peer_base + lu->Pr, 0);
    std::fill(lu->symV2CpuPartnerRecvOffsets.begin() + peer_base,
              lu->symV2CpuPartnerRecvOffsets.begin() + peer_base + lu->Pr,
              std::numeric_limits<size_t>::max());

    Ftype *recv_buffer = lu->symV2CpuPartnerRecvBufs[slot];
    Ftype *send_buffer = lu->symV2CpuPartnerSendBufs[slot];
    size_t request_count = 0;
    size_t recv_total = 0;
    double mpi_start = SuperLU_timer_();
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        size_t count = lu->symV2CpuPartnerRecvSizes[recv_base + pr];
        int source = PNUM(pr, 0, lu->grid);
        if (count == 0 || source == lu->iam)
            continue;
        if (recv_total > lu->symV2CpuPartnerRecvCapacity ||
            count > lu->symV2CpuPartnerRecvCapacity - recv_total)
            ABORT("SymFact V2 CPU Pc=1 receive exceeds workspace.");
        lu->symV2CpuPartnerRecvOffsets[peer_base + pr] = recv_total;
        size_t chunks = symldl_v2_cpu_post_receive_chunks(
            lu, slot, request_count, recv_buffer + recv_total, count,
            source, SLU_MPI_TAG(5, k), lu->grid->comm, pr);
        if (chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 CPU Pc=1 receive has too many chunks.");
        lu->symV2CpuPartnerRecvChunksRemaining[peer_base + pr] =
            static_cast<int>(chunks);
        recv_total += count;
    }
    lu->symV2CpuRecvPostTime += SuperLU_timer_() - mpi_start;
    size_t receive_requests = request_count;

    size_t send_slot = static_cast<size_t>(local_panel);
    if (send_slot >= lu->symV2CpuPartnerSendSizes.size())
        ABORT("SymFact V2 CPU Pc=1 send plan is missing.");
    size_t send_count = lu->symV2CpuPartnerSendSizes[send_slot];
    if (send_count > 0)
    {
        double pack_start = SuperLU_timer_();
        symldl_v2_cpu_pack_partner_destination(
            lu, local_panel, 0, lu->symV2CpuRawPanelBufs[slot],
            send_buffer);
        lu->symV2CpuPartnerPackTime += SuperLU_timer_() - pack_start;
        Ftype *payload = send_buffer +
                         lu->symV2CpuPartnerSendOffsets[send_slot];
        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            int destination = PNUM(pr, 0, lu->grid);
            if (destination == lu->iam)
                continue;
            double send_start = SuperLU_timer_();
            size_t chunks = symldl_v2_cpu_post_send_chunks(
                lu, slot, request_count, payload, send_count, destination,
                SLU_MPI_TAG(5, k), lu->grid->comm);
            lu->symV2CpuSendPostTime += SuperLU_timer_() - send_start;
            lu->symV2RouteProfileNotePartnerSend(
                send_count, chunks,
                SUPERLU_MIN(send_count, static_cast<size_t>(INT_MAX)));
            lu->symV2CpuPartnerBytes +=
                static_cast<uint64_t>(send_count * sizeof(Ftype));
        }
    }

    lu->symV2CpuSlotSendBegins[slot] = receive_requests;
    lu->symV2CpuSlotRequestCounts[slot] = request_count;
    if (!panel.isEmpty())
        for (int pr = 0; pr < lu->Pr; ++pr)
            symldl_v2_cpu_reserve_column_outputs(lu, k, slot, pr, 1);
    symldl_v2_cpu_note_slot_pending(lu, slot, 1);

    SymLDLV2CpuExchangeState &state = lu->symV2CpuExchangeStates[slot];
    state.k = k;
    state.parent = parent;
    state.local_panel = local_panel;
    state.source_pc = 0;
    state.receive_request_count = receive_requests;
    state.pending_receive_chunks = receive_requests;
    state.route = SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY;
    state.active = 1;
    ++lu->symV2CpuExchangeIssues;
    lu->symV2CpuFragmentExchangeTime += SuperLU_timer_() - start;
}

template <typename Ftype>
static bool symldl_v2_cpu_progress_pc1_exchange(
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
        if (MPI_Testsome(static_cast<int>(state.receive_request_count),
                         lu->symV2CpuRequests.data() + request_base,
                         &completed, lu->symV2CpuWaitIndices.data(),
                         lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
            ABORT("SymFact V2 CPU Pc=1 receive progress failed.");
        lu->symV2CpuRecvProgressTime += SuperLU_timer_() - mpi_start;
        ++lu->symV2CpuMpiTestsomeCalls;
        if (completed == 0 && blocking)
        {
            mpi_start = SuperLU_timer_();
            if (MPI_Waitsome(static_cast<int>(state.receive_request_count),
                             lu->symV2CpuRequests.data() + request_base,
                             &completed, lu->symV2CpuWaitIndices.data(),
                             lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
                ABORT("SymFact V2 CPU Pc=1 receive wait failed.");
            lu->symV2CpuRecvWaitTime += SuperLU_timer_() - mpi_start;
            ++lu->symV2CpuMpiWaitsomeCalls;
            ++lu->symV2CpuBlockingProgressCalls;
        }
        if (completed == MPI_UNDEFINED || completed < 0 ||
            static_cast<size_t>(completed) > state.pending_receive_chunks)
            ABORT("SymFact V2 CPU Pc=1 receive progress is invalid.");
        for (int item = 0; item < completed; ++item)
        {
            int request = lu->symV2CpuWaitIndices[item];
            if (request < 0 ||
                static_cast<size_t>(request) >=
                    state.receive_request_count)
                ABORT("SymFact V2 CPU Pc=1 receive completion is invalid.");
            int pr = lu->symV2CpuRequestPeers[request_base + request];
            if (pr < 0 || pr >= lu->Pr)
                ABORT("SymFact V2 CPU Pc=1 receive peer is invalid.");
            int &remaining =
                lu->symV2CpuPartnerRecvChunksRemaining[peer_base + pr];
            if (remaining <= 0)
                ABORT("SymFact V2 CPU Pc=1 receive completed twice.");
            --remaining;
        }
        if (completed > 0)
        {
            progressed = true;
            state.pending_receive_chunks -= static_cast<size_t>(completed);
            lu->symV2CpuMpiCompletions += completed;
        }
    }

    xlpanel_t<Ftype> &row_panel = lu->lPanelVec[state.local_panel];
    size_t recv_base = static_cast<size_t>(state.k) * lu->Pr;
    bool single_thread = lu->symV2CpuWorkerCount <= 1;
    bool released_numerical_work = false;
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        size_t peer_slot = peer_base + pr;
        size_t receive_slot = recv_base + pr;
        size_t count = lu->symV2CpuPartnerRecvSizes[receive_slot];
        if (count == 0 || lu->symV2CpuPartnerUpdateSubmitted[peer_slot] ||
            lu->symV2CpuPartnerRecvChunksRemaining[peer_slot] != 0)
            continue;
        const std::vector<int_t> &column_index =
            lu->symV2PartnerLRecvIndexBySrc[receive_slot];
        if (!row_panel.isEmpty() && !column_index.empty())
        {
            Ftype *values = NULL;
            if (PNUM(pr, 0, lu->grid) == lu->iam)
            {
                size_t send_slot = static_cast<size_t>(state.local_panel);
                values = lu->symV2CpuPartnerSendBufs[slot] +
                         lu->symV2CpuPartnerSendOffsets[send_slot];
            }
            else
            {
                size_t offset = lu->symV2CpuPartnerRecvOffsets[peer_slot];
                if (offset == std::numeric_limits<size_t>::max())
                    ABORT("SymFact V2 CPU Pc=1 receive offset is invalid.");
                values = lu->symV2CpuPartnerRecvBufs[slot] + offset;
            }
            xlpanel_t<Ftype> column_panel(
                const_cast<int_t *>(column_index.data()), values);
            symldl_v2_cpu_submit_schur_task_set(
                lu, state.k, state.parent, slot, row_panel.index,
                row_panel.val, row_panel.haveDiag() ? 1 : 0,
                column_panel.index, column_panel.val, 0,
                column_panel.nblocks());
            ++lu->symV2CpuReleaseEventsPartner;
            released_numerical_work = true;
        }
        if (!row_panel.isEmpty())
            symldl_v2_cpu_reserve_column_outputs(
                lu, state.k, slot, pr, -1);
        lu->symV2CpuPartnerUpdateSubmitted[peer_slot] = 1;
        progressed = true;
        if (single_thread && released_numerical_work)
            break;
    }

    if (state.pending_receive_chunks == 0)
    {
        bool all_updates_submitted = true;
        for (int pr = 0; pr < lu->Pr; ++pr)
            if (lu->symV2CpuPartnerRecvSizes[recv_base + pr] != 0 &&
                !lu->symV2CpuPartnerUpdateSubmitted[peer_base + pr])
                all_updates_submitted = false;
        if (!all_updates_submitted && !single_thread)
            ABORT("SymFact V2 CPU Pc=1 update was not submitted.");
        if (all_updates_submitted)
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


template <typename Ftype>
static bool symldl_v2_cpu_progress_exchange(
    xLUstruct_t<Ftype> *lu, int slot, bool blocking)
{
    const SymLDLV2CpuExchangeState &state =
        lu->symV2CpuExchangeStates[slot];
    if (!state.active)
        return false;
    if (symldl_v2_cpu_scheduler_kind() ==
        SYM_LDL_V2_CPU_SCHEDULER_HYBRID)
        return symldl_v2_cpu_progress_hybrid_exchange(
            lu, slot, blocking);
    if (state.route == SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY)
        return symldl_v2_cpu_progress_pc1_exchange(lu, slot, blocking);
    return symldl_v2_cpu_progress_fragment_exchange(lu, slot, blocking);
}

template <typename Ftype>
static bool symldl_v2_cpu_progress_all_exchanges(
    xLUstruct_t<Ftype> *lu, bool blocking)
{
    bool progressed = false;
    int blocking_slot = -1;
    int_t blocking_k = std::numeric_limits<int_t>::max();
    for (size_t slot = 0; slot < lu->symV2CpuExchangeStates.size(); ++slot)
    {
        const SymLDLV2CpuExchangeState &state =
            lu->symV2CpuExchangeStates[slot];
        if (!state.active)
            continue;
        if (state.k < blocking_k)
        {
            blocking_k = state.k;
            blocking_slot = static_cast<int>(slot);
        }
        progressed = symldl_v2_cpu_progress_exchange(
                         lu, static_cast<int>(slot), false) || progressed;
    }
    if (!progressed && blocking && blocking_slot >= 0)
        progressed = symldl_v2_cpu_progress_exchange(
            lu, blocking_slot, true);
    return progressed;
}

template <typename Ftype>
static void symldl_v2_cpu_complete_exchange(
    xLUstruct_t<Ftype> *lu, int slot)
{
    while (lu->symV2CpuExchangeStates[slot].active)
        symldl_v2_cpu_progress_exchange(lu, slot, true);
}
