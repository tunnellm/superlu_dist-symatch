#pragma once

#include <algorithm>
#include <cstring>
#include <limits>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_update_impl.hpp"

static inline int symldl_v2_cpu_mpi_chunk_size(
    size_t total, size_t offset)
{
    if (offset > total)
        ABORT("SymFact V2 CPU MPI chunk offset is invalid.");
    size_t remaining = total - offset;
    return static_cast<int>(SUPERLU_MIN(
        remaining,
        static_cast<size_t>(std::numeric_limits<int>::max())));
}

template <typename Ftype>
static size_t symldl_v2_cpu_post_receive_chunks(
    xLUstruct_t<Ftype> *lu, int slot, size_t &request_count,
    Ftype *buffer, size_t count, int source, int tag, MPI_Comm comm,
    int peer)
{
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    size_t chunks = symldl_v2_cpu_mpi_chunk_count(count);
    if (chunks > 1)
        lu->symV2CpuOversizedMpiChunks += chunks;
    size_t offset = 0;
    for (size_t chunk = 0; chunk < chunks; ++chunk)
    {
        if (request_count >= lu->symV2CpuRequestsPerSlot ||
            request_base + request_count >= lu->symV2CpuRequests.size())
            ABORT("SymFact V2 CPU request workspace is undersized.");
        int chunk_count = symldl_v2_cpu_mpi_chunk_size(count, offset);
        size_t request = request_base + request_count++;
        lu->symV2CpuRequestPeers[request] = peer;
        if (MPI_Irecv(buffer + offset, chunk_count, get_mpi_type<Ftype>(),
                      source, tag, comm,
                      &lu->symV2CpuRequests[request]) != MPI_SUCCESS)
            ABORT("SymFact V2 CPU fragment receive could not be posted.");
        offset += static_cast<size_t>(chunk_count);
    }
    return chunks;
}

template <typename Ftype>
static size_t symldl_v2_cpu_post_send_chunks(
    xLUstruct_t<Ftype> *lu, int slot, size_t &request_count,
    Ftype *buffer, size_t count, int destination, int tag, MPI_Comm comm)
{
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    size_t chunks = symldl_v2_cpu_mpi_chunk_count(count);
    if (chunks > 1)
        lu->symV2CpuOversizedMpiChunks += chunks;
    size_t offset = 0;
    for (size_t chunk = 0; chunk < chunks; ++chunk)
    {
        if (request_count >= lu->symV2CpuRequestsPerSlot ||
            request_base + request_count >= lu->symV2CpuRequests.size())
            ABORT("SymFact V2 CPU request workspace is undersized.");
        int chunk_count = symldl_v2_cpu_mpi_chunk_size(count, offset);
        size_t request = request_base + request_count++;
        lu->symV2CpuRequestPeers[request] = -3;
        if (MPI_Isend(buffer + offset, chunk_count, get_mpi_type<Ftype>(),
                      destination, tag, comm,
                      &lu->symV2CpuRequests[request]) != MPI_SUCCESS)
            ABORT("SymFact V2 CPU fragment send could not be posted.");
        offset += static_cast<size_t>(chunk_count);
    }
    return chunks;
}

template <typename Ftype>
static void symldl_v2_cpu_drain_slot_sends(
    xLUstruct_t<Ftype> *lu, int slot)
{
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuSlotRequestCounts.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuSlotSendBegins.size())
        ABORT("SymFact V2 CPU send-drain slot is invalid.");
    if (static_cast<size_t>(slot) < lu->symV2CpuExchangeStates.size() &&
        lu->symV2CpuExchangeStates[static_cast<size_t>(slot)].active)
        ABORT("SymFact V2 CPU send-drain slot still owns an exchange.");
    size_t request_count =
        lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)];
    size_t send_begin =
        lu->symV2CpuSlotSendBegins[static_cast<size_t>(slot)];
    if (send_begin > request_count ||
        request_count > lu->symV2CpuRequestsPerSlot)
        ABORT("SymFact V2 CPU send-drain state is invalid.");
    size_t send_count = request_count - send_begin;
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    if (send_count > 0)
    {
        if (send_count >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 CPU send-drain count exceeds MPI limits.");
        double drain_start = SuperLU_timer_();
        if (MPI_Waitall(static_cast<int>(send_count),
                        lu->symV2CpuRequests.data() + request_base + send_begin,
                        MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            ABORT("SymFact V2 CPU fragment sends did not complete.");
        lu->symV2CpuSendDrainTime += SuperLU_timer_() - drain_start;
        ++lu->symV2CpuSendDrainCalls;
    }
    for (size_t request = 0; request < request_count; ++request)
    {
        lu->symV2CpuRequests[request_base + request] = MPI_REQUEST_NULL;
        lu->symV2CpuRequestPeers[request_base + request] = -1;
    }
    lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] = 0;
    lu->symV2CpuSlotSendBegins[static_cast<size_t>(slot)] = 0;
}

template <typename Ftype>
static void symldl_v2_cpu_pack_partner_destination(
    xLUstruct_t<Ftype> *lu, int_t local_panel, int destination_pc,
    const Ftype *raw_values, Ftype *send_buffer)
{
    size_t slot = static_cast<size_t>(local_panel) *
                      static_cast<size_t>(lu->Pc) +
                  static_cast<size_t>(destination_pc);
    if (slot >= lu->symV2CpuPartnerSendSizes.size() ||
        slot >= lu->symV2CpuPartnerSendOffsets.size() ||
        slot + 1 >= lu->symV2CpuPartnerSegOffsets.size())
        ABORT("SymFact V2 CPU partner-send plan is invalid.");
    size_t value_count = lu->symV2CpuPartnerSendSizes[slot];
    if (value_count <= 0)
        return;

    xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
    int_t width = panel.ncols();
    if (width <= 0 || value_count % width != 0)
        ABORT("SymFact V2 CPU partner-send width is invalid.");
    int_t packed_rows = static_cast<int_t>(
        value_count / static_cast<size_t>(width));
    Ftype *destination = send_buffer +
                         lu->symV2CpuPartnerSendOffsets[slot];
    for (size_t segment_id = lu->symV2CpuPartnerSegOffsets[slot];
         segment_id < lu->symV2CpuPartnerSegOffsets[slot + 1];
         ++segment_id)
    {
        const SymLDLV2CpuPackSegment &segment =
            lu->symV2CpuPartnerSegments[segment_id];
        int_t source_row = panel.stRow(segment.source_block);
        if (segment.row_permutation_offset ==
            SYM_LDL_V2_CPU_CONTIGUOUS_ROWS)
        {
            for (int_t column = 0; column < width; ++column)
                std::memcpy(
                    destination + segment.packed_row_offset +
                        column * packed_rows,
                    raw_values + source_row + column * panel.LDA(),
                    static_cast<size_t>(segment.row_count) * sizeof(Ftype));
            continue;
        }
        for (int_t column = 0; column < width; ++column)
            for (int_t row = 0; row < segment.row_count; ++row)
            {
                size_t permutation = segment.row_permutation_offset +
                                     static_cast<size_t>(row);
                if (permutation >=
                    lu->symV2CpuPartnerRowPermutations.size())
                    ABORT("SymFact V2 CPU partner row plan is invalid.");
                int_t source_offset =
                    lu->symV2CpuPartnerRowPermutations[permutation];
                destination[segment.packed_row_offset + row +
                            column * packed_rows] =
                    raw_values[source_row + source_offset +
                               column * panel.LDA()];
            }
    }
}

template <typename Ftype>
static void symldl_v2_cpu_pack_row_destination(
    xLUstruct_t<Ftype> *lu, int_t local_panel, int destination_pc,
    xlpanel_t<Ftype> &panel, Ftype *send_buffer)
{
    size_t slot = static_cast<size_t>(local_panel) *
                      static_cast<size_t>(lu->Pc) +
                  static_cast<size_t>(destination_pc);
    if (slot >= lu->symV2CpuRowSendSizes.size() ||
        slot >= lu->symV2CpuRowSendOffsets.size() ||
        slot + 1 >= lu->symV2CpuRowSegOffsets.size())
        ABORT("SymFact V2 CPU row-send plan is invalid.");
    size_t value_count = lu->symV2CpuRowSendSizes[slot];
    if (value_count <= 0)
        return;
    int_t width = panel.ncols();
    if (width <= 0 || value_count % width != 0)
        ABORT("SymFact V2 CPU row-send width is invalid.");
    int_t packed_rows = static_cast<int_t>(
        value_count / static_cast<size_t>(width));
    Ftype *destination = send_buffer + lu->symV2CpuRowSendOffsets[slot];
    for (size_t segment_id = lu->symV2CpuRowSegOffsets[slot];
         segment_id < lu->symV2CpuRowSegOffsets[slot + 1]; ++segment_id)
    {
        const SymLDLV2CpuPackSegment &segment =
            lu->symV2CpuRowSegments[segment_id];
        int_t source_row = panel.stRow(segment.source_block);
        if (segment.row_permutation_offset ==
            SYM_LDL_V2_CPU_CONTIGUOUS_ROWS)
        {
            for (int_t column = 0; column < width; ++column)
                std::memcpy(
                    destination + segment.packed_row_offset +
                        column * packed_rows,
                    panel.val + source_row + column * panel.LDA(),
                    static_cast<size_t>(segment.row_count) * sizeof(Ftype));
            continue;
        }
        for (int_t column = 0; column < width; ++column)
            for (int_t row = 0; row < segment.row_count; ++row)
            {
                size_t permutation = segment.row_permutation_offset +
                                     static_cast<size_t>(row);
                if (permutation >= lu->symV2CpuRowPermutations.size())
                    ABORT("SymFact V2 CPU row permutation is invalid.");
                int_t source_offset =
                    lu->symV2CpuRowPermutations[permutation];
                destination[segment.packed_row_offset + row +
                            column * packed_rows] =
                    panel.val[source_row + source_offset +
                              column * panel.LDA()];
            }
    }
}

template <typename Ftype>
static void symldl_v2_cpu_note_slot_pending(
    xLUstruct_t<Ftype> *lu, int slot, int delta)
{
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuRawPanelBufs.size())
        ABORT("SymFact V2 CPU exchange slot is invalid.");
#ifdef _OPENMP
#pragma omp atomic update
#endif
    lu->symV2CpuSlotPending[slot] += delta;
}

template <typename Ftype>
static void symldl_v2_cpu_reserve_partner_outputs(
    xLUstruct_t<Ftype> *lu, int_t k, int slot, int pr, int delta)
{
    size_t receive_slot = static_cast<size_t>(k) * lu->Pr + pr;
    if (receive_slot >= lu->symV2CpuPartnerRecvSizes.size() ||
        receive_slot >= lu->symV2PartnerLRecvIndexBySrc.size() ||
        static_cast<size_t>(k) >= lu->symV2RowFragRecvIndex.size())
        ABORT("SymFact V2 CPU exchange reservation is invalid.");
    const std::vector<int_t> &row_index =
        lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
    const std::vector<int_t> &column_index =
        lu->symV2PartnerLRecvIndexBySrc[receive_slot];
    if (row_index.empty() || column_index.empty() ||
        lu->symV2CpuPartnerRecvSizes[receive_slot] == 0)
        return;
    xlpanel_t<Ftype> column_panel(
        const_cast<int_t *>(column_index.data()), NULL);
    for (int_t source_j = 0; source_j < column_panel.nblocks(); ++source_j)
        symldl_v2_cpu_note_task_pending(
            lu, column_panel.gid(source_j), slot, delta);
}

template <typename Ftype>
static void symldl_v2_cpu_issue_fragment_exchange(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot,
    bool submit_tasks)
{
    if (!submit_tasks)
        ABORT("SymFact V2 CPU asynchronous exchange requires task submission.");
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuPartnerSendBufs.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuPartnerRecvBufs.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuRowSendBufs.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuRowRecvBufs.size())
        ABORT("SymFact V2 CPU fragment slot is invalid.");

    if (lu->symV2CpuRequestsPerSlot == 0 ||
        lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] != 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuExchangeStates.size() ||
        lu->symV2CpuExchangeStates[static_cast<size_t>(slot)].active)
        ABORT("SymFact V2 CPU fragment slot still owns MPI requests.");

    double start = SuperLU_timer_();
    int tag_ub = lu->symFactTagUb;
    int_t source_pc = lu->symV2PanelRoot(k);
    int_t local_panel = lu->symV2PanelIndex(k);
    Ftype *partner_send = lu->symV2CpuPartnerSendBufs[slot];
    Ftype *partner_recv = lu->symV2CpuPartnerRecvBufs[slot];
    Ftype *row_send = lu->symV2CpuRowSendBufs[slot];
    Ftype *row_recv = lu->symV2CpuRowRecvBufs[slot];
    size_t recv_base = static_cast<size_t>(k) * lu->Pr;
    if (recv_base + static_cast<size_t>(lu->Pr) >
            lu->symV2CpuPartnerRecvSizes.size() ||
        static_cast<size_t>(k) >= lu->symV2RowFragRecvIndex.size())
        ABORT("SymFact V2 CPU fragment metadata is missing.");

    size_t slot_peer_base = static_cast<size_t>(slot) * lu->Pr;
    if (slot_peer_base + static_cast<size_t>(lu->Pr) >
            lu->symV2CpuPartnerRecvChunksRemaining.size() ||
        slot_peer_base + static_cast<size_t>(lu->Pr) >
            lu->symV2CpuPartnerUpdateSubmitted.size() ||
        slot_peer_base + static_cast<size_t>(lu->Pr) >
            lu->symV2CpuPartnerRecvOffsets.size())
        ABORT("SymFact V2 CPU per-slot exchange state is undersized.");
    std::fill(lu->symV2CpuPartnerRecvChunksRemaining.begin() +
                  slot_peer_base,
              lu->symV2CpuPartnerRecvChunksRemaining.begin() +
                  slot_peer_base + lu->Pr, 0);
    std::fill(lu->symV2CpuPartnerUpdateSubmitted.begin() + slot_peer_base,
              lu->symV2CpuPartnerUpdateSubmitted.begin() +
                  slot_peer_base + lu->Pr, 0);
    std::fill(lu->symV2CpuPartnerRecvOffsets.begin() + slot_peer_base,
              lu->symV2CpuPartnerRecvOffsets.begin() +
                  slot_peer_base + lu->Pr,
              std::numeric_limits<size_t>::max());

    size_t request_count = 0;
    size_t partner_receive_total = 0;
    double receive_post_start = SuperLU_timer_();
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        size_t count = lu->symV2CpuPartnerRecvSizes[recv_base + pr];
        int source = PNUM(pr, source_pc, lu->grid);
        if (count == 0 || source == lu->iam)
            continue;
        if (count > lu->symV2CpuPartnerRecvCapacity -
                        SUPERLU_MIN(partner_receive_total,
                                    lu->symV2CpuPartnerRecvCapacity))
            ABORT("SymFact V2 CPU partner receive exceeds workspace.");
        lu->symV2CpuPartnerRecvOffsets[
            slot_peer_base + static_cast<size_t>(pr)] =
            partner_receive_total;
        size_t chunks = symldl_v2_cpu_post_receive_chunks(
            lu, slot, request_count, partner_recv + partner_receive_total,
            count, source, SLU_MPI_TAG(5, k), lu->grid->comm, pr);
        if (chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 CPU partner receive has too many chunks.");
        lu->symV2CpuPartnerRecvChunksRemaining[
            slot_peer_base + static_cast<size_t>(pr)] =
            static_cast<int>(chunks);
        partner_receive_total += count;
    }
    lu->symV2CpuRecvPostTime += SuperLU_timer_() - receive_post_start;

    const std::vector<int_t> &row_index =
        lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
    if (!row_index.empty() && row_index.size() < 2)
        ABORT("SymFact V2 CPU row receive index is invalid.");
    size_t row_values = row_index.empty() ? 0 :
        symldl_v2_checked_product(
            static_cast<size_t>(row_index[1]),
            static_cast<size_t>(lu->supersize(k)),
            "SymFact V2 CPU row receive size overflows.");
    if (row_values > lu->symV2CpuRowRecvCapacity)
        ABORT("SymFact V2 CPU row receive exceeds workspace.");
    int row_chunks_remaining = 0;
    if (row_values > 0 && lu->mycol != source_pc)
    {
        receive_post_start = SuperLU_timer_();
        size_t chunks = symldl_v2_cpu_post_receive_chunks(
            lu, slot, request_count, row_recv, row_values,
            static_cast<int>(source_pc), SLU_MPI_TAG(5, k),
            lu->grid3d->rscp.comm, -2);
        if (chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 CPU row receive has too many chunks.");
        row_chunks_remaining = static_cast<int>(chunks);
        lu->symV2CpuRecvPostTime += SuperLU_timer_() - receive_post_start;
    }
    size_t receive_request_count = request_count;

    if (lu->mycol == source_pc)
    {
        if (local_panel < 0 || local_panel >= lu->symV2PanelCount())
            ABORT("SymFact V2 CPU fragment source panel is invalid.");
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        const Ftype *raw_values = lu->symV2CpuRawPanelBufs[slot];
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t send_slot = static_cast<size_t>(local_panel) *
                                   static_cast<size_t>(lu->Pc) +
                               static_cast<size_t>(pc);
            size_t count = lu->symV2CpuPartnerSendSizes[send_slot];
            if (count == 0)
                continue;
            if (send_slot >= lu->symV2PartnerLSendAnyActive.size())
                ABORT("SymFact V2 CPU partner send summary is invalid.");
            if (!lu->symV2PartnerLSendAnyActive[send_slot])
                continue;
            double pack_start = SuperLU_timer_();
            symldl_v2_cpu_pack_partner_destination(
                lu, local_panel, pc, raw_values, partner_send);
            lu->symV2CpuPartnerPackTime += SuperLU_timer_() - pack_start;
            Ftype *buffer = partner_send +
                            lu->symV2CpuPartnerSendOffsets[send_slot];
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos =
                    send_slot * static_cast<size_t>(lu->Pr) +
                    static_cast<size_t>(pr);
                if (active_pos >=
                    lu->symV2PartnerLSendRowActive.size())
                    ABORT("SymFact V2 CPU partner send mask is invalid.");
                if (!lu->symV2PartnerLSendRowActive[active_pos])
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
                    count, chunks,
                    SUPERLU_MIN(
                        count,
                        static_cast<size_t>(std::numeric_limits<int>::max())));
                lu->symV2CpuPartnerBytes += static_cast<uint64_t>(
                    count * sizeof(Ftype));
            }
        }

        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t send_slot = static_cast<size_t>(local_panel) *
                                   static_cast<size_t>(lu->Pc) +
                               static_cast<size_t>(pc);
            if (send_slot >= lu->symV2CpuRowSendSizes.size() ||
                send_slot >= lu->symV2CpuRowSendOffsets.size())
                ABORT("SymFact V2 CPU row-send plan is missing.");
            size_t count = lu->symV2CpuRowSendSizes[send_slot];
            if (count == 0)
                continue;
            size_t offset = lu->symV2CpuRowSendOffsets[send_slot];
            if (offset > lu->symV2CpuRowSendCapacity ||
                count > lu->symV2CpuRowSendCapacity - offset)
                ABORT("SymFact V2 CPU row send exceeds workspace.");
            double pack_start = SuperLU_timer_();
            symldl_v2_cpu_pack_row_destination(
                lu, local_panel, pc, panel, row_send);
            lu->symV2CpuRowPackTime += SuperLU_timer_() - pack_start;
            if (pc == lu->mycol)
            {
                if (count != row_values)
                    ABORT("SymFact V2 CPU self row-fragment size mismatch.");
                continue;
            }
            double send_start = SuperLU_timer_();
            size_t chunks = symldl_v2_cpu_post_send_chunks(
                lu, slot, request_count, row_send + offset, count, pc,
                SLU_MPI_TAG(5, k), lu->grid3d->rscp.comm);
            lu->symV2CpuSendPostTime += SuperLU_timer_() - send_start;
            lu->symV2RouteProfileNoteRowSend(
                count, chunks,
                SUPERLU_MIN(
                    count,
                    static_cast<size_t>(std::numeric_limits<int>::max())));
            lu->symV2CpuRowBytes += static_cast<uint64_t>(
                count * sizeof(Ftype));
        }
    }

    lu->symV2CpuSlotSendBegins[static_cast<size_t>(slot)] =
        receive_request_count;
    lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] =
        request_count;
    for (int pr = 0; pr < lu->Pr; ++pr)
        symldl_v2_cpu_reserve_partner_outputs(lu, k, slot, pr, 1);
    symldl_v2_cpu_note_slot_pending(lu, slot, 1);

    SymLDLV2CpuExchangeState &state =
        lu->symV2CpuExchangeStates[static_cast<size_t>(slot)];
    state.k = k;
    state.parent = parent;
    state.local_panel = local_panel;
    state.source_pc = static_cast<int>(source_pc);
    state.row_chunks_remaining = row_chunks_remaining;
    state.receive_request_count = receive_request_count;
    state.pending_receive_chunks = receive_request_count;
    state.active = 1;
    ++lu->symV2CpuExchangeIssues;
    uint64_t active_exchanges = 0;
    for (size_t active_slot = 0;
         active_slot < lu->symV2CpuExchangeStates.size(); ++active_slot)
        active_exchanges +=
            lu->symV2CpuExchangeStates[active_slot].active != 0;
    lu->symV2CpuActiveExchangeHighWater = SUPERLU_MAX(
        lu->symV2CpuActiveExchangeHighWater, active_exchanges);
    lu->symV2CpuFragmentExchangeTime += SuperLU_timer_() - start;
}

template <typename Ftype>
static bool symldl_v2_cpu_progress_fragment_exchange(
    xLUstruct_t<Ftype> *lu, int slot, bool blocking)
{
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuExchangeStates.size())
        ABORT("SymFact V2 CPU exchange-progress slot is invalid.");
    SymLDLV2CpuExchangeState &state =
        lu->symV2CpuExchangeStates[static_cast<size_t>(slot)];
    if (!state.active)
        return false;

    double progress_wall_start = SuperLU_timer_();
    bool made_progress = false;
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    size_t slot_peer_base = static_cast<size_t>(slot) * lu->Pr;
    if (state.receive_request_count >
        static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 CPU receive request count exceeds MPI limits.");

    if (state.pending_receive_chunks > 0)
    {
        int completed = 0;
        double mpi_start = SuperLU_timer_();
        if (MPI_Testsome(static_cast<int>(state.receive_request_count),
                         lu->symV2CpuRequests.data() + request_base,
                         &completed, lu->symV2CpuWaitIndices.data(),
                         lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
            ABORT("SymFact V2 CPU receive progress failed.");
        lu->symV2CpuRecvProgressTime += SuperLU_timer_() - mpi_start;
        ++lu->symV2CpuMpiTestsomeCalls;
        if (completed == MPI_UNDEFINED)
            ABORT("SymFact V2 CPU receive progress lost active requests.");
        if (completed == 0 && blocking)
        {
            mpi_start = SuperLU_timer_();
            if (MPI_Waitsome(static_cast<int>(state.receive_request_count),
                             lu->symV2CpuRequests.data() + request_base,
                             &completed, lu->symV2CpuWaitIndices.data(),
                             lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
                ABORT("SymFact V2 CPU receive wait failed.");
            lu->symV2CpuRecvWaitTime += SuperLU_timer_() - mpi_start;
            ++lu->symV2CpuMpiWaitsomeCalls;
            ++lu->symV2CpuBlockingProgressCalls;
        }
        if (completed == MPI_UNDEFINED || completed < 0 ||
            static_cast<size_t>(completed) > state.pending_receive_chunks)
            ABORT("SymFact V2 CPU fragment receive made invalid progress.");
        if (completed > 0)
        {
            made_progress = true;
            lu->symV2CpuMpiCompletions +=
                static_cast<uint64_t>(completed);
            for (int item = 0; item < completed; ++item)
            {
                int request = lu->symV2CpuWaitIndices[
                    static_cast<size_t>(item)];
                if (request < 0 ||
                    static_cast<size_t>(request) >=
                        state.receive_request_count)
                    ABORT("SymFact V2 CPU receive completion is invalid.");
                int peer = lu->symV2CpuRequestPeers[
                    request_base + static_cast<size_t>(request)];
                if (peer >= 0)
                {
                    size_t peer_slot = slot_peer_base +
                                       static_cast<size_t>(peer);
                    int &remaining =
                        lu->symV2CpuPartnerRecvChunksRemaining[peer_slot];
                    if (remaining <= 0)
                        ABORT("SymFact V2 CPU partner receive completed twice.");
                    --remaining;
                }
                else if (peer == -2)
                {
                    if (state.row_chunks_remaining <= 0)
                        ABORT("SymFact V2 CPU row receive completed twice.");
                    --state.row_chunks_remaining;
                }
                else
                    ABORT("SymFact V2 CPU receive has an invalid peer.");
            }
            state.pending_receive_chunks -=
                static_cast<size_t>(completed);
        }
    }

    if (state.row_chunks_remaining == 0)
    {
        const std::vector<int_t> &row_index =
            lu->symV2RowFragRecvIndex[static_cast<size_t>(state.k)];
        if (!row_index.empty() && row_index.size() < 2)
            ABORT("SymFact V2 CPU row receive index is invalid.");
        Ftype *row_values = lu->symV2CpuRowRecvBufs[slot];
        if (lu->mycol == state.source_pc && !row_index.empty())
        {
            if (state.local_panel < 0 ||
                state.local_panel >= lu->symV2PanelCount())
                ABORT("SymFact V2 CPU self row panel is invalid.");
            size_t send_slot = static_cast<size_t>(state.local_panel) *
                                   static_cast<size_t>(lu->Pc) +
                               static_cast<size_t>(lu->mycol);
            if (send_slot >= lu->symV2CpuRowSendSizes.size())
                ABORT("SymFact V2 CPU self row plan is invalid.");
            row_values = lu->symV2CpuRowSendBufs[slot] +
                         lu->symV2CpuRowSendOffsets[send_slot];
        }
        xlpanel_t<Ftype> row_panel;
        if (!row_index.empty())
            row_panel = xlpanel_t<Ftype>(
                const_cast<int_t *>(row_index.data()), row_values);

        size_t recv_base = static_cast<size_t>(state.k) * lu->Pr;
        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            size_t peer_slot = slot_peer_base + static_cast<size_t>(pr);
            size_t receive_slot = recv_base + static_cast<size_t>(pr);
            size_t count = lu->symV2CpuPartnerRecvSizes[receive_slot];
            if (count == 0 ||
                lu->symV2CpuPartnerUpdateSubmitted[peer_slot] != 0 ||
                lu->symV2CpuPartnerRecvChunksRemaining[peer_slot] != 0)
                continue;

            const std::vector<int_t> &column_index =
                lu->symV2PartnerLRecvIndexBySrc[receive_slot];
            if (!row_index.empty() && !column_index.empty())
            {
                int source = PNUM(pr, state.source_pc, lu->grid);
                Ftype *column_values = NULL;
                if (source == lu->iam)
                {
                    size_t send_slot =
                        static_cast<size_t>(state.local_panel) *
                            static_cast<size_t>(lu->Pc) +
                        static_cast<size_t>(lu->mycol);
                    if (send_slot >= lu->symV2CpuPartnerSendSizes.size() ||
                        lu->symV2CpuPartnerSendSizes[send_slot] != count)
                        ABORT("SymFact V2 CPU self partner-fragment size mismatch.");
                    column_values = lu->symV2CpuPartnerSendBufs[slot] +
                        lu->symV2CpuPartnerSendOffsets[send_slot];
                }
                else
                {
                    size_t offset =
                        lu->symV2CpuPartnerRecvOffsets[peer_slot];
                    if (offset == std::numeric_limits<size_t>::max())
                        ABORT("SymFact V2 CPU partner receive offset is invalid.");
                    column_values =
                        lu->symV2CpuPartnerRecvBufs[slot] + offset;
                }
                xlpanel_t<Ftype> column_panel(
                    const_cast<int_t *>(column_index.data()),
                    column_values);
                symldl_v2_cpu_submit_dual_schur_tasks(
                    lu, state.k, state.parent, slot, row_panel,
                    column_panel);
            }
            symldl_v2_cpu_reserve_partner_outputs(
                lu, state.k, slot, pr, -1);
            lu->symV2CpuPartnerUpdateSubmitted[peer_slot] = 1;
            made_progress = true;
        }
    }

    if (state.pending_receive_chunks == 0)
    {
        if (state.row_chunks_remaining != 0)
            ABORT("SymFact V2 CPU row fragment did not complete.");
        size_t recv_base = static_cast<size_t>(state.k) * lu->Pr;
        for (int pr = 0; pr < lu->Pr; ++pr)
            if (lu->symV2CpuPartnerRecvSizes[
                    recv_base + static_cast<size_t>(pr)] != 0 &&
                lu->symV2CpuPartnerUpdateSubmitted[
                    slot_peer_base + static_cast<size_t>(pr)] == 0)
                ABORT("SymFact V2 CPU partner update was not submitted.");

        state = SymLDLV2CpuExchangeState();
        symldl_v2_cpu_note_slot_pending(lu, slot, -1);
        ++lu->symV2CpuExchangeCompletions;
        ++lu->symV2CpuPanelsCompleted;
        made_progress = true;
        if (lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] ==
            lu->symV2CpuSlotSendBegins[static_cast<size_t>(slot)])
            symldl_v2_cpu_drain_slot_sends(lu, slot);
    }

    lu->symV2CpuFragmentExchangeTime +=
        SuperLU_timer_() - progress_wall_start;
    return made_progress;
}

template <typename Ftype>
static bool symldl_v2_cpu_progress_all_fragment_exchanges(
    xLUstruct_t<Ftype> *lu, bool blocking)
{
    bool made_progress = false;
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
        made_progress = symldl_v2_cpu_progress_fragment_exchange(
                            lu, static_cast<int>(slot), false) ||
                        made_progress;
    }
    if (blocking && !made_progress && blocking_slot >= 0 &&
        lu->symV2CpuExchangeStates[
            static_cast<size_t>(blocking_slot)].active)
        made_progress = symldl_v2_cpu_progress_fragment_exchange(
            lu, blocking_slot, true);
    return made_progress;
}

template <typename Ftype>
static void symldl_v2_cpu_exchange_fragments_and_update(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot,
    bool submit_tasks)
{
    symldl_v2_cpu_issue_fragment_exchange(
        lu, k, parent, slot, submit_tasks);
    while (lu->symV2CpuExchangeStates[static_cast<size_t>(slot)].active)
        symldl_v2_cpu_progress_fragment_exchange(lu, slot, true);
}
