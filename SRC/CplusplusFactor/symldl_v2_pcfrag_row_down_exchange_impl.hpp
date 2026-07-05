#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "symldl_v2_pcfrag_row_down_segments_impl.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
static std::vector<std::vector<SymLDLV2RowDownDirectRange> >
symldl_v2_exchange_row_down_demands(
    xLUstruct_t<Ftype> *lu,
    const std::vector<std::vector<int_t> > &row_down_payloads,
    size_t l2l_slots)
{
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
        size_t pos = static_cast<size_t>(
            recv_displs[static_cast<size_t>(sender_pc)]);
        size_t end =
            pos + static_cast<size_t>(recv_counts[static_cast<size_t>(sender_pc)]);
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
    return slot_direct_ranges;
}

template <typename Ftype>
static void symldl_v2_exchange_row_down_size_replies(
    xLUstruct_t<Ftype> *lu,
    const std::vector<std::vector<int_t> > &size_reply_payloads)
{
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
}

#endif
