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
static void symldl_v2_cpu_post_send_chunks(
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
}

template <typename Ftype>
static void symldl_v2_cpu_drain_slot_sends(
    xLUstruct_t<Ftype> *lu, int slot)
{
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuSlotRequestCounts.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuSlotSendBegins.size())
        ABORT("SymFact V2 CPU send-drain slot is invalid.");
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
static void symldl_v2_cpu_exchange_fragments_and_update(
    xLUstruct_t<Ftype> *lu, int_t k, int_t parent, int slot,
    bool submit_tasks)
{
    if (slot < 0 ||
        static_cast<size_t>(slot) >= lu->symV2CpuPartnerSendBufs.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuPartnerRecvBufs.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuRowSendBufs.size() ||
        static_cast<size_t>(slot) >= lu->symV2CpuRowRecvBufs.size())
        ABORT("SymFact V2 CPU fragment slot is invalid.");

    if (lu->symV2CpuRequestsPerSlot == 0 ||
        lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] != 0)
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

    std::fill(lu->symV2CpuPartnerRecvChunksRemaining.begin(),
              lu->symV2CpuPartnerRecvChunksRemaining.end(), 0);
    std::fill(lu->symV2CpuPartnerUpdateSubmitted.begin(),
              lu->symV2CpuPartnerUpdateSubmitted.end(), 0);
    std::fill(lu->symV2CpuPartnerRecvOffsets.begin(),
              lu->symV2CpuPartnerRecvOffsets.end(),
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
        lu->symV2CpuPartnerRecvOffsets[static_cast<size_t>(pr)] =
            partner_receive_total;
        size_t chunks = symldl_v2_cpu_post_receive_chunks(
            lu, slot, request_count, partner_recv + partner_receive_total,
            count, source, SLU_MPI_TAG(5, k), lu->grid->comm, pr);
        if (chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 CPU partner receive has too many chunks.");
        lu->symV2CpuPartnerRecvChunksRemaining[
            static_cast<size_t>(pr)] = static_cast<int>(chunks);
        partner_receive_total += count;
    }
    lu->symV2CpuRecvPostTime += SuperLU_timer_() - receive_post_start;

    const std::vector<int_t> &row_index =
        lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
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
            double pack_start = SuperLU_timer_();
            symldl_v2_cpu_pack_partner_destination(
                lu, local_panel, pc, raw_values, partner_send);
            lu->symV2CpuPartnerPackTime += SuperLU_timer_() - pack_start;
            Ftype *buffer = partner_send +
                            lu->symV2CpuPartnerSendOffsets[send_slot];
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                int destination = PNUM(pr, pc, lu->grid);
                if (destination == lu->iam)
                    continue;
                double send_start = SuperLU_timer_();
                symldl_v2_cpu_post_send_chunks(
                    lu, slot, request_count, buffer, count, destination,
                    SLU_MPI_TAG(5, k), lu->grid->comm);
                lu->symV2CpuSendPostTime += SuperLU_timer_() - send_start;
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
            symldl_v2_cpu_post_send_chunks(
                lu, slot, request_count, row_send + offset, count, pc,
                SLU_MPI_TAG(5, k), lu->grid3d->rscp.comm);
            lu->symV2CpuSendPostTime += SuperLU_timer_() - send_start;
            lu->symV2CpuRowBytes += static_cast<uint64_t>(
                count * sizeof(Ftype));
        }
    }

    Ftype *row_fragment_values = row_recv;
    if (lu->mycol == source_pc && row_values > 0)
    {
        size_t send_slot = static_cast<size_t>(local_panel) *
                               static_cast<size_t>(lu->Pc) +
                           static_cast<size_t>(lu->mycol);
        if (send_slot >= lu->symV2CpuRowSendSizes.size() ||
            lu->symV2CpuRowSendSizes[send_slot] != row_values)
            ABORT("SymFact V2 CPU self row-fragment size mismatch.");
        row_fragment_values = row_send +
            lu->symV2CpuRowSendOffsets[send_slot];
    }
    xlpanel_t<Ftype> row_panel;
    if (!row_index.empty())
        row_panel = xlpanel_t<Ftype>(
            const_cast<int_t *>(row_index.data()), row_fragment_values);

    auto update_partner_fragment = [&](int pr)
    {
        size_t receive_slot = recv_base + static_cast<size_t>(pr);
        size_t count = lu->symV2CpuPartnerRecvSizes[receive_slot];
        const std::vector<int_t> &column_index =
            lu->symV2PartnerLRecvIndexBySrc[receive_slot];
        if (count == 0 || column_index.empty() || row_index.empty())
            return;
        int source = PNUM(pr, source_pc, lu->grid);
        Ftype *column_values = NULL;
        if (source == lu->iam)
        {
            size_t send_slot = static_cast<size_t>(local_panel) *
                                   static_cast<size_t>(lu->Pc) +
                               static_cast<size_t>(lu->mycol);
            if (send_slot >= lu->symV2CpuPartnerSendSizes.size() ||
                lu->symV2CpuPartnerSendSizes[send_slot] != count)
                ABORT("SymFact V2 CPU self partner-fragment size mismatch.");
            column_values = partner_send +
                lu->symV2CpuPartnerSendOffsets[send_slot];
        }
        else
        {
            size_t offset = lu->symV2CpuPartnerRecvOffsets[
                static_cast<size_t>(pr)];
            if (offset == std::numeric_limits<size_t>::max())
                ABORT("SymFact V2 CPU partner receive offset is invalid.");
            column_values = partner_recv + offset;
        }
        xlpanel_t<Ftype> column_panel(
            const_cast<int_t *>(column_index.data()), column_values);
        if (submit_tasks)
            symldl_v2_cpu_submit_dual_schur_tasks(
                lu, k, parent, slot, row_panel, column_panel);
        else
            symldl_v2_cpu_dual_schur_update(
                lu, k, row_panel, column_panel);
    };

    bool row_ready = row_chunks_remaining == 0;
    auto submit_ready_partners = [&]()
    {
        if (!row_ready)
            return;
        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            size_t receive_slot = recv_base + static_cast<size_t>(pr);
            if (lu->symV2CpuPartnerRecvSizes[receive_slot] == 0 ||
                lu->symV2CpuPartnerUpdateSubmitted[
                    static_cast<size_t>(pr)] != 0 ||
                lu->symV2CpuPartnerRecvChunksRemaining[
                    static_cast<size_t>(pr)] != 0)
                continue;
            update_partner_fragment(pr);
            lu->symV2CpuPartnerUpdateSubmitted[
                static_cast<size_t>(pr)] = 1;
        }
    };
    submit_ready_partners();

    size_t pending_receive_chunks = receive_request_count;
    size_t request_base = static_cast<size_t>(slot) *
                          lu->symV2CpuRequestsPerSlot;
    while (pending_receive_chunks > 0)
    {
        if (receive_request_count >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 CPU receive request count exceeds MPI limits.");
        int completed = 0;
        double progress_start = SuperLU_timer_();
        if (MPI_Testsome(static_cast<int>(receive_request_count),
                         lu->symV2CpuRequests.data() + request_base,
                         &completed, lu->symV2CpuWaitIndices.data(),
                         lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
            ABORT("SymFact V2 CPU receive progress failed.");
        lu->symV2CpuRecvProgressTime +=
            SuperLU_timer_() - progress_start;
        ++lu->symV2CpuMpiTestsomeCalls;
        if (completed == MPI_UNDEFINED)
            ABORT("SymFact V2 CPU receive progress lost active requests.");
        if (completed == 0)
        {
            double wait_start = SuperLU_timer_();
            if (MPI_Waitsome(static_cast<int>(receive_request_count),
                             lu->symV2CpuRequests.data() + request_base,
                             &completed, lu->symV2CpuWaitIndices.data(),
                             lu->symV2CpuWaitStatuses.data()) != MPI_SUCCESS)
                ABORT("SymFact V2 CPU receive wait failed.");
            lu->symV2CpuRecvWaitTime += SuperLU_timer_() - wait_start;
            ++lu->symV2CpuMpiWaitsomeCalls;
        }
        if (completed == MPI_UNDEFINED || completed <= 0 ||
            static_cast<size_t>(completed) > pending_receive_chunks)
            ABORT("SymFact V2 CPU fragment receive made no progress.");
        lu->symV2CpuMpiCompletions += static_cast<uint64_t>(completed);
        for (int item = 0; item < completed; ++item)
        {
            int request = lu->symV2CpuWaitIndices[
                static_cast<size_t>(item)];
            if (request < 0 ||
                static_cast<size_t>(request) >= receive_request_count)
                ABORT("SymFact V2 CPU receive completion is invalid.");
            int peer = lu->symV2CpuRequestPeers[
                request_base + static_cast<size_t>(request)];
            if (peer >= 0)
            {
                int &remaining = lu->symV2CpuPartnerRecvChunksRemaining[
                    static_cast<size_t>(peer)];
                if (remaining <= 0)
                    ABORT("SymFact V2 CPU partner receive completed twice.");
                --remaining;
            }
            else if (peer == -2)
            {
                if (row_chunks_remaining <= 0)
                    ABORT("SymFact V2 CPU row receive completed twice.");
                --row_chunks_remaining;
                row_ready = row_chunks_remaining == 0;
            }
            else
                ABORT("SymFact V2 CPU receive has an invalid peer.");
        }
        pending_receive_chunks -= static_cast<size_t>(completed);
        submit_ready_partners();
    }
    if (!row_ready)
        ABORT("SymFact V2 CPU row fragment did not complete.");
    submit_ready_partners();

    lu->symV2CpuSlotSendBegins[static_cast<size_t>(slot)] =
        receive_request_count;
    lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] =
        request_count;
    if (request_count == receive_request_count)
        symldl_v2_cpu_drain_slot_sends(lu, slot);
    lu->symV2CpuFragmentExchangeTime += SuperLU_timer_() - start;
}
