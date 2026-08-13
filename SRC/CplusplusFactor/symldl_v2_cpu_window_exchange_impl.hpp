#pragma once

#include <algorithm>
#include <cstring>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_exchange_impl.hpp"
#include "symldl_v2_cpu_panel_impl.hpp"

template <typename Ftype>
static void symldl_v2_cpu_window_reconstruct_collapsed_raw_panel(
    xLUstruct_t<Ftype> *lu, int_t k, int slot)
{
    if (lu->mycol != lu->symV2PanelRoot(k))
        return;
    int_t local_panel = lu->symV2PanelIndex(k);
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
        ABORT("SymFact V2 CPU window source panel is invalid.");
    xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
    if (panel.isEmpty())
        return;
    int_t width = panel.ncols();
    if (width <= 0 || panel.nzrows() <= 0 ||
        static_cast<size_t>(k) >= lu->symV2DiagBlocks.size() ||
        lu->symV2DiagBlocks[static_cast<size_t>(k)] == NULL)
        ABORT("SymFact V2 CPU window diagonal data is missing.");
    size_t values = static_cast<size_t>(panel.nzrows()) * width;
    if (values > lu->symV2CpuRawPanelCapacity)
        ABORT("SymFact V2 CPU window raw panel exceeds workspace.");
    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    symldl_v2_cpu_gemm<Ftype>(
        "N", "N", panel.nzrows(), width, width, alpha,
        panel.val, panel.LDA(),
        lu->symV2DiagBlocks[static_cast<size_t>(k)], width, beta,
        lu->symV2CpuRawPanelBufs[static_cast<size_t>(slot)], panel.LDA());
}

template <typename Ftype>
static void symldl_v2_cpu_window_pack_partner_raw(
    xLUstruct_t<Ftype> *lu, int_t k, int_t local_panel,
    int destination_pc, Ftype *send_buffer)
{
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount() ||
        static_cast<size_t>(k) >= lu->symV2DiagBlocks.size() ||
        lu->symV2DiagBlocks[static_cast<size_t>(k)] == NULL)
        ABORT("SymFact V2 CPU window partner source is invalid.");
    size_t plan = static_cast<size_t>(local_panel) * lu->Pc +
                  static_cast<size_t>(destination_pc);
    if (plan >= lu->symV2CpuPartnerSendSizes.size() ||
        plan >= lu->symV2CpuPartnerSendOffsets.size() ||
        plan + 1 >= lu->symV2CpuPartnerSegOffsets.size())
        ABORT("SymFact V2 CPU window partner-send plan is invalid.");
    size_t value_count = lu->symV2CpuPartnerSendSizes[plan];
    if (value_count == 0)
        return;

    xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
    int_t width = panel.ncols();
    if (panel.isEmpty() || width <= 0 || value_count % width != 0)
        ABORT("SymFact V2 CPU window partner-send width is invalid.");
    int_t packed_rows = static_cast<int_t>(value_count / width);
    Ftype *destination = send_buffer +
                         lu->symV2CpuPartnerSendOffsets[plan];
    const Ftype *diagonal = lu->symV2DiagBlocks[static_cast<size_t>(k)];

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
                        ABORT("SymFact V2 CPU window partner row plan is invalid.");
                    source_offset =
                        lu->symV2CpuPartnerRowPermutations[permutation];
                }
                Ftype sum = zeroT<Ftype>();
                for (int_t inner = 0; inner < width; ++inner)
                    sum += panel.val[source_row + source_offset +
                                     inner * panel.LDA()] *
                           diagonal[inner + column * width];
                destination[segment.packed_row_offset + row +
                            column * packed_rows] = sum;
            }
    }
}

template <typename Ftype>
static void symldl_v2_cpu_window_wait_requests(
    xLUstruct_t<Ftype> *lu, int slot, size_t begin, size_t count,
    const char *failure)
{
    if (count == 0)
        return;
    if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        begin > lu->symV2CpuRequestsPerSlot ||
        count > lu->symV2CpuRequestsPerSlot - begin)
        ABORT("SymFact V2 CPU window request count exceeds MPI limits.");
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    if (MPI_Waitall(static_cast<int>(count),
                    lu->symV2CpuRequests.data() + request_base + begin,
                    MPI_STATUSES_IGNORE) != MPI_SUCCESS)
        ABORT(failure);
}

template <typename Ftype>
static void symldl_v2_cpu_window_reset_requests(
    xLUstruct_t<Ftype> *lu, int slot, size_t request_count)
{
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    for (size_t request = 0; request < request_count; ++request)
    {
        lu->symV2CpuRequests[request_base + request] = MPI_REQUEST_NULL;
        lu->symV2CpuRequestPeers[request_base + request] = -1;
        lu->symV2CpuRequestKinds[request_base + request] =
            SYM_LDL_V2_CPU_REQUEST_NONE;
    }
    lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] = 0;
    lu->symV2CpuSlotSendBegins[static_cast<size_t>(slot)] = 0;
}

template <typename Ftype>
static void symldl_v2_cpu_window_exchange(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot)
{
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuWindowStates.size() ||
        lu->symV2CpuWindowStates[static_cast<size_t>(slot)].ready ||
        lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] != 0)
        ABORT("SymFact V2 CPU window slot is not reusable.");
    double exchange_start = SuperLU_timer_();
    int tag_ub = lu->symFactTagUb;
    if (tag_ub <= 0)
        ABORT("SymFact V2 CPU window MPI tag bound is invalid.");
    SymLDLV2CpuExchangeRoute route = symldl_v2_cpu_exchange_route(lu);
    int_t source_pc = lu->symV2PanelRoot(k);
    int_t local_panel = lu->symV2PanelIndex(k);

    if (route == SYM_LDL_V2_CPU_ROUTE_COLLAPSED)
    {
        symldl_v2_cpu_window_reconstruct_collapsed_raw_panel(lu, k, slot);
        Ftype *raw = NULL;
        (void) symldl_v2_cpu_exchange_panel(lu, k, slot, &raw);
        SymLDLV2CpuWindowState &state =
            lu->symV2CpuWindowStates[static_cast<size_t>(slot)];
        state.k = k;
        state.parent = parent;
        state.local_panel = local_panel;
        state.source_pc = static_cast<int>(source_pc);
        state.route = route;
        state.ready = 1;
        ++lu->symV2CpuWindowPanels;
        ++lu->symV2CpuPanelsIssued;
        ++lu->symV2CpuPanelsCompleted;
        ++lu->symV2CpuRoutePanels[static_cast<int>(route)];
        lu->symV2CpuWindowExchangeTime +=
            SuperLU_timer_() - exchange_start;
        return;
    }

    if (static_cast<size_t>(k) >=
            lu->symV2CpuPartnerAssembledIndex.size() ||
        static_cast<size_t>(k) >= lu->symV2RowFragRecvIndex.size())
        ABORT("SymFact V2 CPU window fragment metadata is missing.");
    Ftype *partner_send = lu->symV2CpuPartnerSendBufs[slot];
    Ftype *partner_recv = lu->symV2CpuPartnerRecvBufs[slot];
    size_t request_count = 0;
    size_t receive_request_count = 0;
    size_t partner_receive_total = 0;
    size_t recv_base = static_cast<size_t>(k) * lu->Pr;
    size_t slot_peer_base = static_cast<size_t>(slot) * lu->Pr;

    /* Mirror the GPU partner helper: receive, demand-pack, send, wait, assemble. */
    double post_start = SuperLU_timer_();
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        size_t count = lu->symV2CpuPartnerRecvSizes[recv_base + pr];
        int source = PNUM(pr, source_pc, lu->grid);
        if (count == 0 || source == lu->iam)
            continue;
        if (partner_receive_total > lu->symV2CpuPartnerRecvCapacity ||
            count > lu->symV2CpuPartnerRecvCapacity - partner_receive_total)
            ABORT("SymFact V2 CPU window partner receive exceeds workspace.");
        lu->symV2CpuPartnerRecvOffsets[slot_peer_base + pr] =
            partner_receive_total;
        symldl_v2_cpu_post_receive_chunks(
            lu, slot, request_count, partner_recv + partner_receive_total,
            count, source, SLU_MPI_TAG(5, k), lu->grid->comm, pr);
        partner_receive_total += count;
    }
    size_t partner_receive_count = request_count;
    receive_request_count += partner_receive_count;
    lu->symV2CpuRecvPostTime += SuperLU_timer_() - post_start;

    if (lu->mycol == source_pc)
    {
        if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
            ABORT("SymFact V2 CPU window source panel is invalid.");
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t send_plan = static_cast<size_t>(local_panel) * lu->Pc + pc;
            size_t count = lu->symV2CpuPartnerSendSizes[send_plan];
            if (count == 0)
                continue;
            bool active = false;
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos = send_plan * lu->Pr + pr;
                if (active_pos >= lu->symV2CpuPartnerSendRowActive.size())
                    ABORT("SymFact V2 CPU window partner send mask is missing.");
                active = active ||
                         lu->symV2CpuPartnerSendRowActive[active_pos];
            }
            if (!active)
                continue;
            double pack_start = SuperLU_timer_();
            symldl_v2_cpu_window_pack_partner_raw(
                lu, k, local_panel, pc, partner_send);
            lu->symV2CpuPartnerPackTime += SuperLU_timer_() - pack_start;
        }
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t send_plan = static_cast<size_t>(local_panel) * lu->Pc + pc;
            size_t count = lu->symV2CpuPartnerSendSizes[send_plan];
            if (count == 0)
                continue;
            Ftype *buffer = partner_send +
                            lu->symV2CpuPartnerSendOffsets[send_plan];
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos = send_plan * lu->Pr + pr;
                if (!lu->symV2CpuPartnerSendRowActive[active_pos])
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
                        count, static_cast<size_t>(
                                   std::numeric_limits<int>::max())));
                lu->symV2CpuPartnerBytes += count * sizeof(Ftype);
            }
        }
    }

    double wait_start = SuperLU_timer_();
    symldl_v2_cpu_window_wait_requests(
        lu, slot, 0, partner_receive_count,
        "SymFact V2 CPU window partner receives did not complete.");
    lu->symV2CpuRecvWaitTime += SuperLU_timer_() - wait_start;

    double assembly_start = SuperLU_timer_();
    const std::vector<int_t> &partner_index =
        lu->symV2CpuPartnerAssembledIndex[static_cast<size_t>(k)];
    int_t partner_rows = partner_index.empty() ? 0 : partner_index[1];
    int_t width = lu->supersize(k);
    Ftype *assembled = lu->symV2CpuPartnerAssembledBufs[slot];
    if (partner_rows > 0)
        std::fill(assembled,
                  assembled + static_cast<size_t>(partner_rows) * width,
                  zeroT<Ftype>());
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        size_t source_plan = recv_base + pr;
        size_t count = lu->symV2CpuPartnerRecvSizes[source_plan];
        if (count == 0)
            continue;
        int source = PNUM(pr, source_pc, lu->grid);
        const Ftype *source_values = NULL;
        if (source == lu->iam)
        {
            size_t send_plan = static_cast<size_t>(local_panel) * lu->Pc +
                               static_cast<size_t>(lu->mycol);
            source_values = partner_send +
                            lu->symV2CpuPartnerSendOffsets[send_plan];
        }
        else
            source_values = partner_recv +
                lu->symV2CpuPartnerRecvOffsets[slot_peer_base + pr];
        const std::vector<int_t> &source_index =
            lu->symV2PartnerLRecvIndexBySrc[source_plan];
        const std::vector<int_t> &map =
            lu->symV2CpuPartnerAssembleMaps[source_plan];
        if (map.size() % 3 != 0 || source_index.empty())
            ABORT("SymFact V2 CPU window partner assembly map is invalid.");
        xlpanel_t<Ftype> source_panel(
            const_cast<int_t *>(source_index.data()),
            const_cast<Ftype *>(source_values));
        for (size_t piece = 0; piece < map.size(); piece += 3)
        {
            int_t destination_row = map[piece];
            int_t rows = map[piece + 1];
            int_t source_row = map[piece + 2];
            for (int_t column = 0; column < width; ++column)
                std::memcpy(
                    assembled + destination_row + column * partner_rows,
                    source_values + source_row + column * source_panel.LDA(),
                    static_cast<size_t>(rows) * sizeof(Ftype));
        }
    }
    lu->symV2CpuFragmentAssemblyTime +=
        SuperLU_timer_() - assembly_start;
    lu->symV2CpuWindowAssemblyTime +=
        SuperLU_timer_() - assembly_start;

    /* Mirror the GPU row helper only after partner assembly is enqueued. */
    if (route == SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT)
    {
        if (static_cast<size_t>(slot) >= lu->symV2CpuRowSendBufs.size() ||
            static_cast<size_t>(slot) >= lu->symV2CpuRowRecvBufs.size())
            ABORT("SymFact V2 CPU window row workspace is missing.");
        Ftype *row_send = lu->symV2CpuRowSendBufs[slot];
        Ftype *row_recv = lu->symV2CpuRowRecvBufs[slot];
        const std::vector<int_t> &row_index =
            lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
        size_t row_receive_total = 0;
        size_t row_recv_base = static_cast<size_t>(k) * lu->Pc;
        if (row_recv_base + static_cast<size_t>(lu->Pc) >
            lu->symV2RowFragRecvSizes.size())
            ABORT("SymFact V2 CPU window row receive sizes are missing.");
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            int count = lu->symV2RowFragRecvSizes[row_recv_base + pc];
            if (count < 0 ||
                static_cast<size_t>(count) >
                    std::numeric_limits<size_t>::max() - row_receive_total)
                ABORT("SymFact V2 CPU window row receive size is invalid.");
            row_receive_total += static_cast<size_t>(count);
        }
        if (row_receive_total > lu->symV2CpuRowRecvCapacity)
            ABORT("SymFact V2 CPU window row receive exceeds workspace.");

        size_t row_receive_begin = request_count;
        post_start = SuperLU_timer_();
        if (row_receive_total > 0 && lu->mycol != source_pc)
            symldl_v2_cpu_post_receive_chunks(
                lu, slot, request_count, row_recv, row_receive_total,
                static_cast<int>(source_pc), SLU_MPI_TAG(5, k),
                lu->grid3d->rscp.comm, -2);
        size_t row_receive_count = request_count - row_receive_begin;
        receive_request_count += row_receive_count;
        lu->symV2CpuRecvPostTime += SuperLU_timer_() - post_start;

        if (lu->mycol == source_pc)
        {
            xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
            for (int pc = 0; pc < lu->Pc; ++pc)
            {
                size_t send_plan = static_cast<size_t>(local_panel) * lu->Pc + pc;
                size_t count = lu->symV2CpuRowSendSizes[send_plan];
                if (pc == lu->mycol || count == 0)
                    continue;
                double pack_start = SuperLU_timer_();
                symldl_v2_cpu_pack_row_destination(
                    lu, local_panel, pc, panel, row_send);
                lu->symV2CpuRowPackTime += SuperLU_timer_() - pack_start;
            }
            for (int pc = 0; pc < lu->Pc; ++pc)
            {
                size_t send_plan = static_cast<size_t>(local_panel) * lu->Pc + pc;
                size_t count = lu->symV2CpuRowSendSizes[send_plan];
                if (pc == lu->mycol || count == 0)
                    continue;
                size_t offset = lu->symV2CpuRowSendOffsets[send_plan];
                double send_start = SuperLU_timer_();
                size_t chunks = symldl_v2_cpu_post_send_chunks(
                    lu, slot, request_count, row_send + offset, count, pc,
                    SLU_MPI_TAG(5, k), lu->grid3d->rscp.comm);
                lu->symV2CpuSendPostTime += SuperLU_timer_() - send_start;
                lu->symV2RouteProfileNoteRowSend(
                    count, chunks, SUPERLU_MIN(
                        count, static_cast<size_t>(
                                   std::numeric_limits<int>::max())));
                lu->symV2CpuRowBytes += count * sizeof(Ftype);
            }
        }

        wait_start = SuperLU_timer_();
        symldl_v2_cpu_window_wait_requests(
            lu, slot, row_receive_begin, row_receive_count,
            "SymFact V2 CPU window row receives did not complete.");
        lu->symV2CpuRecvWaitTime += SuperLU_timer_() - wait_start;
        if (row_receive_total > 0 && lu->mycol == source_pc)
        {
            size_t self_plan = static_cast<size_t>(local_panel) * lu->Pc +
                               static_cast<size_t>(lu->mycol);
            if (lu->symV2CpuRowSendSizes[self_plan] != row_receive_total)
                ABORT("SymFact V2 CPU window row self-pack size mismatch.");
            double pack_start = SuperLU_timer_();
            symldl_v2_cpu_pack_row_destination(
                lu, local_panel, lu->mycol, lu->lPanelVec[local_panel],
                row_send);
            lu->symV2CpuRowPackTime += SuperLU_timer_() - pack_start;
        }
        if (!row_index.empty())
        {
            size_t expected = symldl_v2_checked_product(
                static_cast<size_t>(row_index[1]),
                static_cast<size_t>(width),
                "SymFact V2 CPU window row layout overflows.");
            if (row_receive_total != expected)
                ABORT("SymFact V2 CPU window row receive does not match its layout.");
        }
    }

    lu->symV2CpuSlotSendBegins[static_cast<size_t>(slot)] = 0;
    lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] = request_count;
    if (request_count > receive_request_count)
    {
        double send_wait_start = SuperLU_timer_();
        symldl_v2_cpu_window_wait_requests(
            lu, slot, 0, request_count,
            "SymFact V2 CPU window sends did not complete.");
        lu->symV2CpuSendDrainTime +=
            SuperLU_timer_() - send_wait_start;
        ++lu->symV2CpuSendDrainCalls;
    }
    symldl_v2_cpu_window_reset_requests(lu, slot, request_count);

    SymLDLV2CpuWindowState &state =
        lu->symV2CpuWindowStates[static_cast<size_t>(slot)];
    state.k = k;
    state.parent = parent;
    state.local_panel = local_panel;
    state.source_pc = static_cast<int>(source_pc);
    state.receive_request_count = receive_request_count;
    state.request_count = request_count;
    state.route = route;
    state.ready = 1;
    ++lu->symV2CpuWindowPanels;
    ++lu->symV2CpuPanelsIssued;
    ++lu->symV2CpuPanelsCompleted;
    ++lu->symV2CpuRoutePanels[static_cast<int>(route)];
    lu->symV2CpuWindowExchangeTime += SuperLU_timer_() - exchange_start;
}

template <typename Ftype>
static void symldl_v2_cpu_window_panels(
    xLUstruct_t<Ftype> *lu, int slot,
    xlpanel_t<Ftype> *row_panel, xlpanel_t<Ftype> *column_panel)
{
    if (row_panel == NULL || column_panel == NULL || slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuWindowStates.size())
        ABORT("SymFact V2 CPU window panel request is invalid.");
    SymLDLV2CpuWindowState &state =
        lu->symV2CpuWindowStates[static_cast<size_t>(slot)];
    if (!state.ready)
        ABORT("SymFact V2 CPU window panel is not ready.");
    int_t k = state.k;
    if (state.route == SYM_LDL_V2_CPU_ROUTE_COLLAPSED)
    {
        int_t *index = lu->LidxRecvBufs[slot];
        Ftype *values = lu->LvalRecvBufs[slot];
        if (lu->mycol == state.source_pc)
        {
            xlpanel_t<Ftype> &local = lu->lPanelVec[state.local_panel];
            index = local.index;
            values = local.val;
        }
        *row_panel = xlpanel_t<Ftype>(index, values);
        *column_panel = xlpanel_t<Ftype>(
            index, lu->symV2CpuRawPanelBufs[slot]);
        return;
    }

    const std::vector<int_t> &column_index =
        lu->symV2CpuPartnerAssembledIndex[static_cast<size_t>(k)];
    if (!column_index.empty())
        *column_panel = xlpanel_t<Ftype>(
            const_cast<int_t *>(column_index.data()),
            lu->symV2CpuPartnerAssembledBufs[slot]);
    if (state.route == SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY)
    {
        if (state.local_panel < 0 || state.local_panel >= lu->symV2PanelCount())
            ABORT("SymFact V2 CPU window local row panel is invalid.");
        *row_panel = lu->lPanelVec[state.local_panel];
        return;
    }
    const std::vector<int_t> &row_index =
        lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
    Ftype *row_values = lu->symV2CpuRowRecvBufs[slot];
    if (lu->mycol == state.source_pc && !row_index.empty())
    {
        size_t send_plan = static_cast<size_t>(state.local_panel) * lu->Pc +
                           static_cast<size_t>(lu->mycol);
        row_values = lu->symV2CpuRowSendBufs[slot] +
                     lu->symV2CpuRowSendOffsets[send_plan];
    }
    if (!row_index.empty())
        *row_panel = xlpanel_t<Ftype>(
            const_cast<int_t *>(row_index.data()), row_values);
}

template <typename Ftype>
static void symldl_v2_cpu_window_release(
    xLUstruct_t<Ftype> *lu, int slot)
{
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuWindowStates.size())
        ABORT("SymFact V2 CPU window release slot is invalid.");
    lu->symV2CpuWindowStates[static_cast<size_t>(slot)] =
        SymLDLV2CpuWindowState();
}
