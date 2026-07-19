#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_pcfrag_partner_metadata_impl.hpp"

struct SymLDLV2PartnerExclusion
{
    int_t source_rank;
    int_t destination_rank;
    int_t panel;
};

static bool symldl_v2_partner_exclusion_less(
    const SymLDLV2PartnerExclusion &left,
    const SymLDLV2PartnerExclusion &right)
{
    if (left.source_rank != right.source_rank)
        return left.source_rank < right.source_rank;
    if (left.destination_rank != right.destination_rank)
        return left.destination_rank < right.destination_rank;
    return left.panel < right.panel;
}

static bool symldl_v2_partner_exclusion_equal(
    const SymLDLV2PartnerExclusion &left,
    const SymLDLV2PartnerExclusion &right)
{
    return left.source_rank == right.source_rank &&
           left.destination_rank == right.destination_rank &&
           left.panel == right.panel;
}

template <typename Ftype>
static size_t symldl_v2_partner_send_value_count(
    const xLUstruct_t<Ftype> *lu, size_t flat)
{
    if (lu->symV2UsesCpuFactor())
    {
        if (flat >= lu->symV2CpuPartnerSendSizes.size())
            ABORT("SymFact V2 CPU partner send size is missing.");
        return lu->symV2CpuPartnerSendSizes[flat];
    }
    if (flat >= lu->symV2PartnerLSendSizes.size())
        ABORT("SymFact V2 GPU partner send size is missing.");
    int values = lu->symV2PartnerLSendSizes[flat];
    if (values < 0)
        ABORT("SymFact V2 partner send size is invalid.");
    return static_cast<size_t>(values);
}

template <typename Ftype>
static size_t symldl_v2_partner_recv_value_count(
    const xLUstruct_t<Ftype> *lu, size_t slot)
{
    if (lu->symV2UsesCpuFactor())
    {
        if (slot >= lu->symV2CpuPartnerRecvSizes.size())
            ABORT("SymFact V2 CPU partner receive size is missing.");
        return lu->symV2CpuPartnerRecvSizes[slot];
    }
    if (slot >= lu->symV2PartnerLRecvSizes.size())
        ABORT("SymFact V2 GPU partner receive size is missing.");
    int values = lu->symV2PartnerLRecvSizes[slot];
    if (values < 0)
        ABORT("SymFact V2 partner receive size is invalid.");
    return static_cast<size_t>(values);
}

static void symldl_v2_partner_profile_add(
    uint64_t *total, size_t value, const char *message)
{
    if (value > static_cast<size_t>(std::numeric_limits<uint64_t>::max()) ||
        *total > std::numeric_limits<uint64_t>::max() -
                     static_cast<uint64_t>(value))
        ABORT(message);
    *total += static_cast<uint64_t>(value);
}

template <typename Ftype>
static bool symldl_v2_partner_chunk_is_needed(
    xLUstruct_t<Ftype> *lu, int_t column_gid,
    const std::vector<int_t> &row_blocks)
{
    if (row_blocks.empty() ||
        lu->symV2PanelRoot(column_gid) != lu->mycol)
        return false;
    int_t local_panel = lu->symV2PanelIndex(column_gid);
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount() ||
        lu->symV2PanelGid(local_panel) != column_gid)
        return false;
    xlpanel_t<Ftype> &destination = lu->lPanelVec[local_panel];
    if (destination.isEmpty())
        return false;
    typename std::vector<int_t>::const_iterator row =
        std::lower_bound(row_blocks.begin(), row_blocks.end(), column_gid);
    for (; row != row_blocks.end(); ++row)
        if (destination.find(*row) != GLOBAL_BLOCK_NOT_FOUND)
            return true;
    return false;
}

static std::vector<int_t> symldl_v2_route_partner_exclusions(
    const std::vector<std::vector<SymLDLV2PartnerExclusion> > &outbound,
    MPI_Comm comm)
{
    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);
    if (outbound.size() != static_cast<size_t>(comm_size))
        ABORT("SymFact V2 partner exclusion route has wrong size.");

    std::vector<unsigned long long> send_counts(
        static_cast<size_t>(comm_size), 0);
    std::vector<unsigned long long> recv_counts(
        static_cast<size_t>(comm_size), 0);
    std::vector<size_t> send_displs(static_cast<size_t>(comm_size), 0);
    size_t send_total = 0;
    bool local_fits = true;
    for (int destination = 0; destination < comm_size; ++destination)
    {
        const std::vector<SymLDLV2PartnerExclusion> &records =
            outbound[static_cast<size_t>(destination)];
        if (records.size() >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(3))
            ABORT("SymFact V2 partner exclusion payload size overflows.");
        size_t count = records.size() * static_cast<size_t>(3);
        if (count > static_cast<size_t>(
                        std::numeric_limits<unsigned long long>::max()) ||
            send_total > std::numeric_limits<size_t>::max() - count)
            ABORT("SymFact V2 partner exclusion payload size overflows.");
        send_counts[static_cast<size_t>(destination)] =
            static_cast<unsigned long long>(count);
        send_displs[static_cast<size_t>(destination)] = send_total;
        send_total += count;
        local_fits = local_fits &&
            count <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            send_displs[static_cast<size_t>(destination)] <=
                static_cast<size_t>(std::numeric_limits<int>::max());
    }

    if (MPI_Alltoall(send_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                     recv_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                     comm) != MPI_SUCCESS)
        ABORT("SymFact V2 partner exclusion size exchange failed.");

    std::vector<size_t> recv_displs(static_cast<size_t>(comm_size), 0);
    size_t recv_total = 0;
    for (int source = 0; source < comm_size; ++source)
    {
        unsigned long long count = recv_counts[static_cast<size_t>(source)];
        if (count > static_cast<unsigned long long>(
                        std::numeric_limits<size_t>::max()) ||
            recv_total > std::numeric_limits<size_t>::max() -
                             static_cast<size_t>(count))
            ABORT("SymFact V2 partner exclusion receive size overflows.");
        recv_displs[static_cast<size_t>(source)] = recv_total;
        recv_total += static_cast<size_t>(count);
        local_fits = local_fits &&
            count <= static_cast<unsigned long long>(
                         std::numeric_limits<int>::max()) &&
            recv_displs[static_cast<size_t>(source)] <=
                static_cast<size_t>(std::numeric_limits<int>::max());
    }
    local_fits = local_fits &&
        send_total <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
        recv_total <= static_cast<size_t>(std::numeric_limits<int>::max());
    int local_fits_int = local_fits ? 1 : 0;
    int all_fit = 0;
    if (MPI_Allreduce(&local_fits_int, &all_fit, 1, MPI_INT, MPI_MIN,
                      comm) != MPI_SUCCESS)
        ABORT("SymFact V2 partner exclusion route-size reduction failed.");

    std::vector<int_t> send_payload(send_total);
    for (int destination = 0; destination < comm_size; ++destination)
    {
        size_t position = send_displs[static_cast<size_t>(destination)];
        const std::vector<SymLDLV2PartnerExclusion> &records =
            outbound[static_cast<size_t>(destination)];
        for (size_t record = 0; record < records.size(); ++record)
        {
            send_payload[position++] = records[record].source_rank;
            send_payload[position++] = records[record].destination_rank;
            send_payload[position++] = records[record].panel;
        }
    }

    if (!all_fit)
    {
        SymLDLV2PartnerMetaPayload gathered =
            symldl_v2_allgather_metadata(send_payload, comm, NULL);
        return gathered.payload;
    }

    std::vector<int> send_counts_int(static_cast<size_t>(comm_size), 0);
    std::vector<int> recv_counts_int(static_cast<size_t>(comm_size), 0);
    std::vector<int> send_displs_int(static_cast<size_t>(comm_size), 0);
    std::vector<int> recv_displs_int(static_cast<size_t>(comm_size), 0);
    for (int peer = 0; peer < comm_size; ++peer)
    {
        send_counts_int[static_cast<size_t>(peer)] =
            static_cast<int>(send_counts[static_cast<size_t>(peer)]);
        recv_counts_int[static_cast<size_t>(peer)] =
            static_cast<int>(recv_counts[static_cast<size_t>(peer)]);
        send_displs_int[static_cast<size_t>(peer)] =
            static_cast<int>(send_displs[static_cast<size_t>(peer)]);
        recv_displs_int[static_cast<size_t>(peer)] =
            static_cast<int>(recv_displs[static_cast<size_t>(peer)]);
    }
    std::vector<int_t> received(recv_total);
    if (MPI_Alltoallv(
            send_payload.empty() ? NULL : send_payload.data(),
            send_counts_int.data(), send_displs_int.data(), mpi_int_t,
            received.empty() ? NULL : received.data(),
            recv_counts_int.data(), recv_displs_int.data(), mpi_int_t,
            comm) != MPI_SUCCESS)
        ABORT("SymFact V2 partner exclusion exchange failed.");
    return received;
}

template <typename Ftype>
static void symldl_v2_compute_partner_filter_profile(
    xLUstruct_t<Ftype> *lu)
{
    lu->symV2PartnerCandidateRemoteRecipients = 0;
    lu->symV2PartnerActiveRemoteRecipients = 0;
    lu->symV2PartnerCandidateRemoteValues = 0;
    lu->symV2PartnerActiveRemoteValues = 0;
    lu->symV2PartnerActiveSelfRecipients = 0;
    size_t send_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->symV2PanelCount()),
        static_cast<size_t>(lu->Pc),
        "SymFact V2 partner profile table overflows.");
    for (size_t flat = 0; flat < send_slots; ++flat)
    {
        size_t values = symldl_v2_partner_send_value_count(lu, flat);
        if (values == 0)
            continue;
        int pc = static_cast<int>(flat % static_cast<size_t>(lu->Pc));
        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            size_t active_pos = flat * static_cast<size_t>(lu->Pr) +
                                static_cast<size_t>(pr);
            if (active_pos >= lu->symV2PartnerLSendRowActive.size())
                ABORT("SymFact V2 partner profile mask is invalid.");
            int destination = PNUM(pr, pc, lu->grid);
            bool active = lu->symV2PartnerLSendRowActive[active_pos] != 0;
            if (destination == lu->iam)
            {
                if (active)
                    ++lu->symV2PartnerActiveSelfRecipients;
                continue;
            }
            ++lu->symV2PartnerCandidateRemoteRecipients;
            symldl_v2_partner_profile_add(
                &lu->symV2PartnerCandidateRemoteValues, values,
                "SymFact V2 candidate partner values overflow.");
            if (!active)
                continue;
            ++lu->symV2PartnerActiveRemoteRecipients;
            symldl_v2_partner_profile_add(
                &lu->symV2PartnerActiveRemoteValues, values,
                "SymFact V2 active partner values overflow.");
        }
    }
}

template <typename Ftype>
static void symldl_v2_build_partner_filter_plan(
    xLUstruct_t<Ftype> *lu,
    const SymLDLV2PartnerMetaPayload &metadata)
{
    double plan_start = SuperLU_timer_();
    size_t flat_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->symV2PanelCount()),
        static_cast<size_t>(lu->Pc),
        "SymFact V2 partner filter send table overflows.");
    size_t send_slots = symldl_v2_checked_product(
        flat_slots, static_cast<size_t>(lu->Pr),
        "SymFact V2 partner filter send table overflows.");
    size_t recv_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 partner filter receive table overflows.");
    if (lu->symV2PartnerLSendRowActive.size() != send_slots ||
        lu->symV2PartnerLSendAnyActive.size() != flat_slots ||
        lu->symV2PartnerLRecvActive.size() != recv_slots)
        ABORT("SymFact V2 partner filter tables are not initialized.");

    // A real source chunk is useful by default. Destinations report only the
    // exceptions, so planning traffic scales with inactive recipients.
    std::fill(lu->symV2PartnerLSendRowActive.begin(),
              lu->symV2PartnerLSendRowActive.end(), 0);
    std::fill(lu->symV2PartnerLSendAnyActive.begin(),
              lu->symV2PartnerLSendAnyActive.end(), 0);
    std::fill(lu->symV2PartnerLRecvActive.begin(),
              lu->symV2PartnerLRecvActive.end(), 0);
    for (size_t flat = 0; flat < flat_slots; ++flat)
    {
        if (symldl_v2_partner_send_value_count(lu, flat) == 0)
            continue;
        lu->symV2PartnerLSendAnyActive[flat] = 1;
        for (int pr = 0; pr < lu->Pr; ++pr)
            lu->symV2PartnerLSendRowActive[
                flat * static_cast<size_t>(lu->Pr) +
                static_cast<size_t>(pr)] = 1;
    }
    lu->symV2PartnerExclusionRecords = 0;
    lu->symV2PartnerExclusionPayloadBytes = 0;
    lu->symV2PartnerExclusionExchangeTime = 0.0;

    int comm_rank = -1;
    int comm_size = 0;
    MPI_Comm_rank(lu->grid->comm, &comm_rank);
    MPI_Comm_size(lu->grid->comm, &comm_size);
    if (metadata.comm_size != comm_size || comm_rank < 0 ||
        metadata.counts.size() != static_cast<size_t>(comm_size) ||
        metadata.displs.size() != static_cast<size_t>(comm_size))
        ABORT("SymFact V2 partner filter metadata communicator is invalid.");

    std::vector<std::vector<int_t> > row_blocks_by_panel(
        static_cast<size_t>(lu->nsupers));
    for (int rank = 0; rank < metadata.comm_size; ++rank)
    {
        size_t position = metadata.displs[static_cast<size_t>(rank)];
        size_t end = position + metadata.counts[static_cast<size_t>(rank)];
        int source_pr = MYROW(rank, lu->grid);
        while (position < end)
        {
            if (position + 3 > end)
                ABORT("SymFact V2 partner filter metadata is truncated.");
            int_t target_pc = metadata.payload[position++];
            int_t panel = metadata.payload[position++];
            int_t length = metadata.payload[position++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                panel < 0 || panel >= lu->nsupers || length < 0 ||
                position + static_cast<size_t>(length) > end)
                ABORT("SymFact V2 partner filter metadata is invalid.");
            size_t record_end = position + static_cast<size_t>(length);
            if (source_pr == lu->myrow)
            {
                std::vector<int_t> &rows =
                    row_blocks_by_panel[static_cast<size_t>(panel)];
                while (position < record_end)
                {
                    if (position + 2 > record_end)
                        ABORT("SymFact V2 partner filter block is truncated.");
                    int_t gid = metadata.payload[position++];
                    int_t count = metadata.payload[position++];
                    if (count < 0 ||
                        position + static_cast<size_t>(count) > record_end)
                        ABORT("SymFact V2 partner filter block is invalid.");
                    rows.push_back(gid);
                    position += static_cast<size_t>(count);
                }
            }
            position = record_end;
        }
    }
    for (int_t panel = 0; panel < lu->nsupers; ++panel)
    {
        std::vector<int_t> &rows =
            row_blocks_by_panel[static_cast<size_t>(panel)];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }

    // Each destination applies the same local-output predicate as the former
    // active-demand plan, but routes only negative decisions to the source.
    std::vector<std::vector<SymLDLV2PartnerExclusion> > outbound(
        static_cast<size_t>(comm_size));
    std::vector<unsigned char> candidate_seen(recv_slots, 0);
    for (int rank = 0; rank < metadata.comm_size; ++rank)
    {
        size_t position = metadata.displs[static_cast<size_t>(rank)];
        size_t end = position + metadata.counts[static_cast<size_t>(rank)];
        int source_pr = MYROW(rank, lu->grid);
        while (position < end)
        {
            if (position + 3 > end)
                ABORT("SymFact V2 partner filter metadata is truncated.");
            int_t target_pc = metadata.payload[position++];
            int_t panel = metadata.payload[position++];
            int_t length = metadata.payload[position++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                panel < 0 || panel >= lu->nsupers || length < 0 ||
                position + static_cast<size_t>(length) > end)
                ABORT("SymFact V2 partner filter metadata is invalid.");
            size_t record_end = position + static_cast<size_t>(length);
            bool needed = false;
            while (position < record_end)
            {
                if (position + 2 > record_end)
                    ABORT("SymFact V2 partner filter block is truncated.");
                int_t column_gid = metadata.payload[position++];
                int_t count = metadata.payload[position++];
                if (count < 0 ||
                    position + static_cast<size_t>(count) > record_end)
                    ABORT("SymFact V2 partner filter block is invalid.");
                if (!needed && target_pc == lu->mycol)
                    needed = symldl_v2_partner_chunk_is_needed(
                        lu, column_gid,
                        row_blocks_by_panel[static_cast<size_t>(panel)]);
                position += static_cast<size_t>(count);
            }
            if (target_pc == lu->mycol)
            {
                size_t receive_slot =
                    static_cast<size_t>(panel) *
                        static_cast<size_t>(lu->Pr) +
                    static_cast<size_t>(source_pr);
                if (receive_slot >= candidate_seen.size() ||
                    candidate_seen[receive_slot])
                    ABORT("SymFact V2 partner filter metadata is duplicated.");
                candidate_seen[receive_slot] = 1;
                lu->symV2PartnerLRecvActive[receive_slot] = needed ? 1 : 0;
                if (!needed)
                {
                    SymLDLV2PartnerExclusion exclusion;
                    exclusion.source_rank = rank;
                    exclusion.destination_rank = lu->iam;
                    exclusion.panel = panel;
                    outbound[static_cast<size_t>(rank)].push_back(exclusion);
                }
            }
            position = record_end;
        }
    }

    size_t local_records = 0;
    for (int source = 0; source < comm_size; ++source)
    {
        std::vector<SymLDLV2PartnerExclusion> &records =
            outbound[static_cast<size_t>(source)];
        std::sort(records.begin(), records.end(),
                  symldl_v2_partner_exclusion_less);
        records.erase(std::unique(records.begin(), records.end(),
                                  symldl_v2_partner_exclusion_equal),
                      records.end());
        if (local_records >
            std::numeric_limits<size_t>::max() - records.size())
            ABORT("SymFact V2 partner exclusion count overflows.");
        local_records += records.size();
    }
    symldl_v2_partner_profile_add(
        &lu->symV2PartnerExclusionRecords, local_records,
        "SymFact V2 partner exclusion count overflows.");
    size_t payload_entries = symldl_v2_checked_product(
        local_records, static_cast<size_t>(3),
        "SymFact V2 partner exclusion payload overflows.");
    size_t payload_bytes = symldl_v2_checked_product(
        payload_entries, sizeof(int_t),
        "SymFact V2 partner exclusion payload overflows.");
    symldl_v2_partner_profile_add(
        &lu->symV2PartnerExclusionPayloadBytes, payload_bytes,
        "SymFact V2 partner exclusion payload overflows.");

    double exchange_start = SuperLU_timer_();
    std::vector<int_t> exclusions = symldl_v2_route_partner_exclusions(
        outbound, lu->grid->comm);
    lu->symV2PartnerExclusionExchangeTime =
        SuperLU_timer_() - exchange_start;
    if (exclusions.size() % 3 != 0)
        ABORT("SymFact V2 partner exclusion payload is truncated.");
    for (size_t position = 0; position < exclusions.size(); position += 3)
    {
        int_t source_rank = exclusions[position];
        int_t destination_rank = exclusions[position + 1];
        int_t panel = exclusions[position + 2];
        if (source_rank < 0 || source_rank >= comm_size ||
            destination_rank < 0 || destination_rank >= comm_size ||
            panel < 0 || panel >= lu->nsupers)
            ABORT("SymFact V2 partner exclusion record is invalid.");
        if (source_rank != lu->iam)
            continue;
        if (MYROW(source_rank, lu->grid) != lu->myrow ||
            lu->symV2PanelRoot(panel) != lu->mycol)
            ABORT("SymFact V2 partner exclusion reached the wrong source.");
        int_t local_panel = lu->symV2PanelIndex(panel);
        if (local_panel < 0 || local_panel >= lu->symV2PanelCount() ||
            lu->symV2PanelGid(local_panel) != panel)
            ABORT("SymFact V2 partner exclusion source panel is missing.");
        int destination_pc = MYCOL(destination_rank, lu->grid);
        int destination_pr = MYROW(destination_rank, lu->grid);
        size_t flat = static_cast<size_t>(local_panel) *
                          static_cast<size_t>(lu->Pc) +
                      static_cast<size_t>(destination_pc);
        size_t active_pos = flat * static_cast<size_t>(lu->Pr) +
                            static_cast<size_t>(destination_pr);
        if (active_pos >= lu->symV2PartnerLSendRowActive.size() ||
            !lu->symV2PartnerLSendRowActive[active_pos])
            ABORT("SymFact V2 partner exclusion send index is invalid.");
        lu->symV2PartnerLSendRowActive[active_pos] = 0;
    }

    for (size_t flat = 0; flat < flat_slots; ++flat)
    {
        unsigned char active = 0;
        for (int pr = 0; pr < lu->Pr; ++pr)
            active = active || lu->symV2PartnerLSendRowActive[
                flat * static_cast<size_t>(lu->Pr) +
                static_cast<size_t>(pr)];
        lu->symV2PartnerLSendAnyActive[flat] = active;
    }

    uint64_t local_receive_records = 0;
    uint64_t local_source_records = 0;
    for (size_t slot = 0; slot < lu->symV2PartnerLRecvActive.size(); ++slot)
        local_receive_records += lu->symV2PartnerLRecvActive[slot] ? 1 : 0;
    for (size_t position = 0;
         position < lu->symV2PartnerLSendRowActive.size(); ++position)
        local_source_records +=
            lu->symV2PartnerLSendRowActive[position] ? 1 : 0;
    uint64_t global_receive_records = 0;
    uint64_t global_source_records = 0;
    if (MPI_Allreduce(&local_receive_records, &global_receive_records, 1,
                      MPI_UINT64_T, MPI_SUM, lu->grid->comm) != MPI_SUCCESS ||
        MPI_Allreduce(&local_source_records, &global_source_records, 1,
                      MPI_UINT64_T, MPI_SUM,
                      lu->grid->comm) != MPI_SUCCESS)
        ABORT("SymFact V2 partner filter validation reduction failed.");
    if (global_receive_records != global_source_records)
        ABORT("SymFact V2 partner filter send/receive counts do not match.");

    symldl_v2_compute_partner_filter_profile(lu);
    lu->symV2PartnerFilterPlanTime = SuperLU_timer_() - plan_start;
}

template <typename Ftype>
static void symldl_v2_validate_partner_filter_plan(
    xLUstruct_t<Ftype> *lu)
{
    uint64_t local[4] = {0, 0, 0, 0};
    for (int_t local_panel = 0; local_panel < lu->symV2PanelCount();
         ++local_panel)
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(local_panel) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc);
            size_t values = symldl_v2_partner_send_value_count(lu, flat);
            bool any_active = false;
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos = flat * static_cast<size_t>(lu->Pr) +
                                    static_cast<size_t>(pr);
                if (active_pos >= lu->symV2PartnerLSendRowActive.size() ||
                    !lu->symV2PartnerLSendRowActive[active_pos])
                    continue;
                any_active = true;
                if (values == 0)
                    ABORT("SymFact V2 active partner send is empty.");
                int destination = PNUM(pr, pc, lu->grid);
                if (destination == lu->iam)
                    continue;
                ++local[0];
                symldl_v2_partner_profile_add(
                    &local[1], values,
                    "SymFact V2 partner send validation overflows.");
            }
            if (flat >= lu->symV2PartnerLSendAnyActive.size() ||
                (lu->symV2PartnerLSendAnyActive[flat] != 0) != any_active)
                ABORT("SymFact V2 partner send summary does not match.");
        }

    for (int_t panel = 0; panel < lu->nsupers; ++panel)
        for (int source_pr = 0; source_pr < lu->Pr; ++source_pr)
        {
            size_t slot = static_cast<size_t>(panel) *
                              static_cast<size_t>(lu->Pr) +
                          static_cast<size_t>(source_pr);
            bool active = lu->symV2PartnerLRecvActive[slot] != 0;
            size_t values = symldl_v2_partner_recv_value_count(lu, slot);
            if (!active)
            {
                if (values != 0 ||
                    !lu->symV2PartnerLRecvIndexBySrc[slot].empty())
                    ABORT("SymFact V2 inactive partner receive is materialized.");
                continue;
            }
            if (values == 0 ||
                lu->symV2PartnerLRecvIndexBySrc[slot].empty())
                ABORT("SymFact V2 active partner receive is missing.");
            int source_pc = static_cast<int>(lu->symV2PanelRoot(panel));
            int source = PNUM(source_pr, source_pc, lu->grid);
            if (source == lu->iam)
                continue;
            ++local[2];
            symldl_v2_partner_profile_add(
                &local[3], values,
                "SymFact V2 partner receive validation overflows.");
        }

    uint64_t global[4] = {0, 0, 0, 0};
    if (MPI_Allreduce(local, global, 4, MPI_UINT64_T, MPI_SUM,
                      lu->grid->comm) != MPI_SUCCESS)
        ABORT("SymFact V2 partner plan validation reduction failed.");
    if (global[0] != global[2] || global[1] != global[3])
        ABORT("SymFact V2 partner send/receive plans do not match.");
}
