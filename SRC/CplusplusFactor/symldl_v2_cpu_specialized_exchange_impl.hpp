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

template <typename T, typename Ftype>
static size_t symldl_v2_cpu_post_ibcast_chunks(
    xLUstruct_t<Ftype> *lu, int slot, size_t &request_count, T *buffer,
    size_t count, MPI_Datatype datatype, int root, MPI_Comm comm,
    SymLDLV2CpuRequestKind kind)
{
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    size_t chunks = symldl_v2_cpu_mpi_chunk_count(count);
    if (chunks > 1)
        lu->symV2CpuOversizedMpiChunks += chunks;
    size_t offset = 0;
    for (size_t chunk = 0; chunk < chunks; ++chunk)
    {
        if (request_count >= lu->symV2CpuRequestsPerSlot)
            ABORT("SymFact V2 CPU broadcast request workspace is undersized.");
        int chunk_count = symldl_v2_cpu_mpi_chunk_size(count, offset);
        size_t request = request_base + request_count++;
        lu->symV2CpuRequestKinds[request] = kind;
        lu->symV2CpuRequestPeers[request] = -1;
        if (MPI_Ibcast(buffer + offset, chunk_count, datatype, root, comm,
                       &lu->symV2CpuRequests[request]) != MPI_SUCCESS)
            ABORT("SymFact V2 CPU nonblocking panel broadcast failed.");
        offset += static_cast<size_t>(chunk_count);
    }
    return chunks;
}

template <typename Ftype>
static int_t symldl_v2_cpu_pr1_select_columns(
    xLUstruct_t<Ftype> *lu, int_t k, int slot, xlpanel_t<Ftype> &panel)
{
    if (static_cast<size_t>(slot) >= lu->symV2CpuPr1ColumnBlockBufs.size())
        ABORT("SymFact V2 CPU Pr=1 column workspace is missing.");
    const std::vector<int_t> &planned =
        lu->symV2PartnerLRecvIndexBySrc[static_cast<size_t>(k)];
    int_t *selected = lu->symV2CpuPr1ColumnBlockBufs[slot];
    int_t count = 0;
    int_t source = panel.haveDiag() ? 1 : 0;
    if (!planned.empty())
    {
        xlpanel_t<Ftype> planned_panel(
            const_cast<int_t *>(planned.data()), NULL);
        for (int_t target = 0; target < planned_panel.nblocks(); ++target)
        {
            int_t gid = planned_panel.gid(target);
            while (source < panel.nblocks() && panel.gid(source) < gid)
                ++source;
            if (source >= panel.nblocks() || panel.gid(source) != gid)
                ABORT("SymFact V2 CPU Pr=1 output block is missing from panel.");
            if (static_cast<size_t>(count) >=
                lu->symV2CpuPr1ColumnBlockCapacity)
                ABORT("SymFact V2 CPU Pr=1 column workspace is undersized.");
            selected[count++] = source;
        }
    }
    if (symldl_v2_cpu_ownership_check_enabled())
    {
        int_t selected_item = 0;
        int_t first = panel.haveDiag() ? 1 : 0;
        for (int_t block = first; block < panel.nblocks(); ++block)
        {
            bool selected_here =
                selected_item < count && selected[selected_item] == block;
            bool output_is_local =
                lu->symV2PanelIndex(panel.gid(block)) >= 0;
            if (selected_here != output_is_local)
                ABORT("SymFact V2 CPU Pr=1 selected-column plan is incomplete.");
            if (selected_here)
                ++selected_item;
        }
        if (selected_item != count)
            ABORT("SymFact V2 CPU Pr=1 selected-column plan is invalid.");
    }
    return count;
}

template <typename Ftype>
static void symldl_v2_cpu_pr1_reconstruct_raw(
    xLUstruct_t<Ftype> *lu, int slot, xlpanel_t<Ftype> &panel,
    Ftype *diag, Ftype *raw)
{
    int_t first = panel.haveDiag() ? 1 : 0;
    int_t first_row = first ? panel.stRow(1) : 0;
    int_t rows = panel.nzrows() - first_row;
    int_t width = panel.ncols();
    if (rows <= 0 || width <= 0)
        return;
    double start = lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
    symldl_v2_cpu_gemm<Ftype>(
        "N", "N", rows, width, width, one<Ftype>(),
        panel.val + first_row, panel.LDA(), diag, width, zeroT<Ftype>(),
        raw + first_row, panel.LDA());
    if (lu->symV2CpuProfileEnabled)
    {
#ifdef _OPENMP
#pragma omp atomic update
#endif
        lu->symV2CpuPr1ReconstructTime += SuperLU_timer_() - start;
    }
    (void) slot;
}

template <typename Ftype>
static void symldl_v2_cpu_pr1_reconstruct_raw_block(
    xLUstruct_t<Ftype> *lu, xlpanel_t<Ftype> &panel, int_t source_block,
    Ftype *diag, Ftype *raw)
{
    int_t first = panel.haveDiag() ? 1 : 0;
    if (source_block < first || source_block >= panel.nblocks())
        ABORT("SymFact V2 CPU Pr=1 reconstruction block is invalid.");
    int_t rows = panel.nbrow(source_block);
    int_t width = panel.ncols();
    if (rows <= 0 || width <= 0)
        return;
    int_t row_begin = panel.stRow(source_block);
    double start = lu->symV2CpuProfileEnabled ? SuperLU_timer_() : 0.0;
    symldl_v2_cpu_gemm<Ftype>(
        "N", "N", rows, width, width, one<Ftype>(),
        panel.val + row_begin, panel.LDA(), diag, width, zeroT<Ftype>(),
        raw + row_begin, panel.LDA());
    if (lu->symV2CpuProfileEnabled)
        lu->symV2CpuPr1ReconstructTime += SuperLU_timer_() - start;
    ++lu->symV2CpuPr1BlockReconstructs;
}

template <typename Ftype>
static int_t symldl_v2_cpu_pr1_parent_item(
    xlpanel_t<Ftype> &panel, const int_t *selected, int_t selected_count,
    int_t parent)
{
    for (int_t item = 0; item < selected_count; ++item)
        if (panel.gid(selected[item]) == parent)
            return item;
    return -1;
}

static int_t symldl_v2_cpu_pr1_serial_item(
    int_t cursor, int_t selected_count, int_t parent_item)
{
    if (cursor < 0 || cursor >= selected_count)
        ABORT("SymFact V2 CPU Pr=1 serial cursor is invalid.");
    if (parent_item < 0)
        return cursor;
    if (parent_item >= selected_count)
        ABORT("SymFact V2 CPU Pr=1 parent item is invalid.");
    if (cursor == 0)
        return parent_item;
    return cursor <= parent_item ? cursor - 1 : cursor;
}

template <typename Ftype>
static void symldl_v2_cpu_issue_pr1_exchange(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot)
{
    if (lu->Pr != 1 || lu->Pc <= 1)
        ABORT("SymFact V2 CPU Pr=1 route has an invalid process grid.");
    if (slot < 0 || static_cast<size_t>(slot) >= lu->LidxRecvBufs.size() ||
        static_cast<size_t>(slot) >= lu->LvalRecvBufs.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuPr1DiagBufs.size())
        ABORT("SymFact V2 CPU Pr=1 exchange slot is invalid.");
    if (lu->symV2CpuSlotRequestCounts[slot] != 0 ||
        lu->symV2CpuExchangeStates[slot].active)
        ABORT("SymFact V2 CPU Pr=1 slot still owns an exchange.");

    int root = static_cast<int>(lu->symV2PanelRoot(k));
    int_t index_count = lu->LidxSendCounts[static_cast<size_t>(k)];
    int_t value_count = lu->LvalSendCounts[static_cast<size_t>(k)];
    int_t width = lu->supersize(k);
    if (index_count < 0 || value_count < 0 || width <= 0)
        ABORT("SymFact V2 CPU Pr=1 panel counts are invalid.");
    if (index_count > lu->maxLidxCount || value_count > lu->maxLvalCount)
        ABORT("SymFact V2 CPU Pr=1 panel exceeds receive workspace.");
    size_t diag_count = symldl_v2_checked_product(
        static_cast<size_t>(width), static_cast<size_t>(width),
        "SymFact V2 CPU Pr=1 diagonal broadcast overflows.");
    if (diag_count > lu->symV2CpuPr1DiagCapacity)
        ABORT("SymFact V2 CPU Pr=1 diagonal exceeds workspace.");
    int_t local_panel = lu->symV2PanelIndex(k);
    int_t *index = lu->LidxRecvBufs[slot];
    Ftype *values = lu->LvalRecvBufs[slot];
    Ftype *diag = lu->symV2CpuPr1DiagBufs[slot];
    if (lu->mycol == root)
    {
        if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
            ABORT("SymFact V2 CPU Pr=1 source panel is invalid.");
        xlpanel_t<Ftype> &source = lu->lPanelVec[local_panel];
        if (source.isEmpty() || source.nzvalSize() != value_count)
            ABORT("SymFact V2 CPU Pr=1 source panel has invalid values.");
        index = source.index;
        values = source.val;
        if (lu->symV2DiagBlocks[static_cast<size_t>(k)] == NULL)
            ABORT("SymFact V2 CPU Pr=1 diagonal block is missing.");
        diag = lu->symV2DiagBlocks[static_cast<size_t>(k)];
    }

    size_t request_count = 0;
    double start = SuperLU_timer_();
    size_t index_chunks = symldl_v2_cpu_post_ibcast_chunks(
        lu, slot, request_count, index, static_cast<size_t>(index_count),
        mpi_int_t, root, lu->grid3d->rscp.comm,
        SYM_LDL_V2_CPU_REQUEST_PANEL_INDEX);
    size_t value_chunks = symldl_v2_cpu_post_ibcast_chunks(
        lu, slot, request_count, values, static_cast<size_t>(value_count),
        get_mpi_type<Ftype>(), root, lu->grid3d->rscp.comm,
        SYM_LDL_V2_CPU_REQUEST_PANEL_VALUES);
    size_t diag_chunks = symldl_v2_cpu_post_ibcast_chunks(
        lu, slot, request_count, diag, diag_count, get_mpi_type<Ftype>(),
        root, lu->grid3d->rscp.comm, SYM_LDL_V2_CPU_REQUEST_DIAG);
    if (index_chunks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        value_chunks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        diag_chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 CPU Pr=1 broadcast has too many chunks.");
    lu->symV2CpuPr1BroadcastTime += SuperLU_timer_() - start;
    if (lu->mycol == root)
    {
        uint64_t recipients = static_cast<uint64_t>(lu->Pc - 1);
        lu->symV2CpuPr1IndexBytes += static_cast<uint64_t>(index_count) *
                                    sizeof(int_t) * recipients;
        lu->symV2CpuPr1ValueBytes += static_cast<uint64_t>(value_count) *
                                    sizeof(Ftype) * recipients;
        lu->symV2CpuPr1DiagBytes += static_cast<uint64_t>(diag_count) *
                                   sizeof(Ftype) * recipients;
    }

    lu->symV2CpuSlotRequestCounts[slot] = request_count;
    lu->symV2CpuSlotSendBegins[slot] = request_count;
    symldl_v2_cpu_reserve_column_outputs(lu, k, slot, 0, 1);
    symldl_v2_cpu_note_slot_pending(lu, slot, 1);
    SymLDLV2CpuExchangeState &state = lu->symV2CpuExchangeStates[slot];
    state.k = k;
    state.parent = parent;
    state.local_panel = local_panel;
    state.source_pc = root;
    state.receive_request_count = request_count;
    state.pending_receive_chunks = request_count;
    state.panel_index_chunks_remaining = static_cast<int>(index_chunks);
    state.panel_value_chunks_remaining = static_cast<int>(value_chunks);
    state.diag_chunks_remaining = static_cast<int>(diag_chunks);
    state.route = SYM_LDL_V2_CPU_ROUTE_PR1_FULL_PANEL;
    state.active = 1;
    ++lu->symV2CpuExchangeIssues;

    if (lu->mycol == root)
    {
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        state.selected_column_count =
            symldl_v2_cpu_pr1_select_columns(lu, k, slot, panel);
        state.serial_parent_item = symldl_v2_cpu_pr1_parent_item(
            panel, lu->symV2CpuPr1ColumnBlockBufs[slot],
            state.selected_column_count, parent);
        state.reconstruction_complete = 1;
    }
}

template <typename Ftype>
static bool symldl_v2_cpu_progress_pr1_exchange(
    xLUstruct_t<Ftype> *lu, int slot, bool blocking)
{
    SymLDLV2CpuExchangeState &state = lu->symV2CpuExchangeStates[slot];
    if (!state.active)
        return false;
    bool progressed = false;
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    if (state.pending_receive_chunks > 0)
    {
        int completed = 0;
        double mpi_start = SuperLU_timer_();
        if (MPI_Testsome(static_cast<int>(state.receive_request_count),
                         lu->symV2CpuRequests.data() + request_base,
                         &completed, lu->symV2CpuWaitIndices.data(),
                         lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
            ABORT("SymFact V2 CPU Pr=1 broadcast progress failed.");
        lu->symV2CpuRecvProgressTime += SuperLU_timer_() - mpi_start;
        ++lu->symV2CpuMpiTestsomeCalls;
        if (completed == 0 && blocking)
        {
            mpi_start = SuperLU_timer_();
            if (MPI_Waitsome(static_cast<int>(state.receive_request_count),
                             lu->symV2CpuRequests.data() + request_base,
                             &completed, lu->symV2CpuWaitIndices.data(),
                             lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
                ABORT("SymFact V2 CPU Pr=1 broadcast wait failed.");
            lu->symV2CpuRecvWaitTime += SuperLU_timer_() - mpi_start;
            ++lu->symV2CpuMpiWaitsomeCalls;
            ++lu->symV2CpuBlockingProgressCalls;
        }
        if (completed == MPI_UNDEFINED || completed < 0 ||
            static_cast<size_t>(completed) > state.pending_receive_chunks)
            ABORT("SymFact V2 CPU Pr=1 broadcast progress is invalid.");
        for (int item = 0; item < completed; ++item)
        {
            int request = lu->symV2CpuWaitIndices[item];
            if (request < 0 ||
                static_cast<size_t>(request) >=
                    state.receive_request_count)
                ABORT("SymFact V2 CPU Pr=1 broadcast completion is invalid.");
            unsigned char kind =
                lu->symV2CpuRequestKinds[request_base + request];
            int *remaining = NULL;
            if (kind == SYM_LDL_V2_CPU_REQUEST_PANEL_INDEX)
                remaining = &state.panel_index_chunks_remaining;
            else if (kind == SYM_LDL_V2_CPU_REQUEST_PANEL_VALUES)
                remaining = &state.panel_value_chunks_remaining;
            else if (kind == SYM_LDL_V2_CPU_REQUEST_DIAG)
                remaining = &state.diag_chunks_remaining;
            else
                ABORT("SymFact V2 CPU Pr=1 request kind is invalid.");
            if (*remaining <= 0)
                ABORT("SymFact V2 CPU Pr=1 request completed twice.");
            --*remaining;
        }
        if (completed > 0)
        {
            state.pending_receive_chunks -= static_cast<size_t>(completed);
            lu->symV2CpuMpiCompletions += completed;
            progressed = true;
        }
    }

    if (state.pending_receive_chunks == 0 &&
        !state.reconstruction_started && !state.reconstruction_complete)
    {
        int_t *index = lu->LidxRecvBufs[slot];
        Ftype *values = lu->LvalRecvBufs[slot];
        xlpanel_t<Ftype> panel(index, values);
        state.selected_column_count =
            symldl_v2_cpu_pr1_select_columns(lu, state.k, slot, panel);
        state.serial_parent_item = symldl_v2_cpu_pr1_parent_item(
            panel, lu->symV2CpuPr1ColumnBlockBufs[slot],
            state.selected_column_count, state.parent);
        state.reconstruction_started = 1;
        if (state.selected_column_count == 0)
            state.reconstruction_complete = 1;
        else
        {
#ifdef _OPENMP
            if (omp_in_parallel() && omp_get_num_threads() > 1)
            {
#pragma omp atomic update
                ++lu->symV2CpuDeferredTasksActive;
                int_t source_k = state.k;
                uint64_t generation =
                    symldl_v2_cpu_slot_generation(lu, slot, source_k);
#pragma omp task firstprivate(slot, source_k, generation) shared(lu)
                {
                    symldl_v2_cpu_assert_slot_generation(
                        lu, slot, source_k, generation);
                    xlpanel_t<Ftype> task_panel(
                        lu->LidxRecvBufs[slot], lu->LvalRecvBufs[slot]);
                    symldl_v2_cpu_pr1_reconstruct_raw(
                        lu, slot, task_panel, lu->symV2CpuPr1DiagBufs[slot],
                        lu->symV2CpuRawPanelBufs[slot]);
#pragma omp atomic write
                    lu->symV2CpuExchangeStates[slot].reconstruction_complete = 1;
#pragma omp atomic update
                    --lu->symV2CpuDeferredTasksActive;
                }
            }
            else
#endif
            {
                if (lu->symV2CpuWorkerCount <= 1)
                    state.serial_block_reconstruction = 1;
                else
                    symldl_v2_cpu_pr1_reconstruct_raw(
                        lu, slot, panel, lu->symV2CpuPr1DiagBufs[slot],
                        lu->symV2CpuRawPanelBufs[slot]);
                state.reconstruction_complete = 1;
            }
            if (!state.serial_block_reconstruction)
                ++lu->symV2CpuPr1ReconstructTasks;
        }
        progressed = true;
    }

    unsigned char reconstruction_complete = 0;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    reconstruction_complete = state.reconstruction_complete;
    if (reconstruction_complete && !state.updates_submitted)
    {
        xlpanel_t<Ftype> panel;
        if (lu->mycol == state.source_pc)
            panel = lu->lPanelVec[state.local_panel];
        else
            panel = xlpanel_t<Ftype>(
                lu->LidxRecvBufs[slot], lu->LvalRecvBufs[slot]);
        int_t *selected = lu->symV2CpuPr1ColumnBlockBufs[slot];
        if (lu->symV2CpuWorkerCount <= 1 &&
            state.selected_column_count > 0)
        {
            int_t item = symldl_v2_cpu_pr1_serial_item(
                state.serial_update_cursor, state.selected_column_count,
                state.serial_parent_item);
            int_t source_block = selected[item];
            if (state.serial_block_reconstruction)
                symldl_v2_cpu_pr1_reconstruct_raw_block(
                    lu, panel, source_block,
                    lu->symV2CpuPr1DiagBufs[slot],
                    lu->symV2CpuRawPanelBufs[slot]);
            bool lookahead = panel.gid(source_block) == state.parent;
            symldl_v2_cpu_submit_selected_task_range(
                lu, state.k, state.parent, slot, panel.index, panel.val,
                panel.haveDiag() ? 1 : 0, panel.index,
                lu->symV2CpuRawPanelBufs[slot], selected, item, item + 1,
                lookahead, 0);
            ++state.serial_update_cursor;
        }
        else
        {
            symldl_v2_cpu_submit_selected_schur_tasks(
                lu, state.k, state.parent, slot, panel,
                lu->symV2CpuRawPanelBufs[slot], selected,
                state.selected_column_count);
            state.serial_update_cursor = state.selected_column_count;
        }
        if (state.serial_update_cursor >= state.selected_column_count)
        {
            symldl_v2_cpu_reserve_column_outputs(
                lu, state.k, slot, 0, -1);
            state.updates_submitted = 1;
            ++lu->symV2CpuReleaseEventsFullPanel;
        }
        progressed = true;
    }

    if (state.pending_receive_chunks == 0 && state.updates_submitted)
    {
        state = SymLDLV2CpuExchangeState();
        symldl_v2_cpu_note_slot_pending(lu, slot, -1);
        ++lu->symV2CpuExchangeCompletions;
        ++lu->symV2CpuPanelsCompleted;
        symldl_v2_cpu_drain_slot_sends(lu, slot);
        progressed = true;
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
    if (state.route == SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY)
        return symldl_v2_cpu_progress_pc1_exchange(lu, slot, blocking);
    if (state.route == SYM_LDL_V2_CPU_ROUTE_PR1_FULL_PANEL)
        return symldl_v2_cpu_progress_pr1_exchange(lu, slot, blocking);
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
