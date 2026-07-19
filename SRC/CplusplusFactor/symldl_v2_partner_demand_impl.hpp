#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_pcfrag_partner_metadata_impl.hpp"

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
    xLUstruct_t<Ftype> *lu, int_t gj,
    const std::vector<int_t> &row_blocks)
{
    if (row_blocks.empty() || lu->symV2PanelRoot(gj) != lu->mycol)
        return false;
    int_t local_panel = lu->symV2PanelIndex(gj);
    if (local_panel < 0 || local_panel >= lu->symV2PanelCount() ||
        lu->symV2PanelGid(local_panel) != gj)
        return false;
    xlpanel_t<Ftype> &destination = lu->lPanelVec[local_panel];
    if (destination.isEmpty())
        return false;
    typename std::vector<int_t>::const_iterator row =
        std::lower_bound(row_blocks.begin(), row_blocks.end(), gj);
    for (; row != row_blocks.end(); ++row)
        if (destination.find(*row) != GLOBAL_BLOCK_NOT_FOUND)
            return true;
    return false;
}

static std::vector<int_t> symldl_v2_route_partner_demands(
    const std::vector<std::vector<int_t> > &outbound, MPI_Comm comm)
{
    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);
    if (outbound.size() != static_cast<size_t>(comm_size))
        ABORT("SymFact V2 partner demand route has wrong size.");

    std::vector<unsigned long long> send_counts(
        static_cast<size_t>(comm_size), 0);
    std::vector<unsigned long long> recv_counts(
        static_cast<size_t>(comm_size), 0);
    std::vector<size_t> send_displs(static_cast<size_t>(comm_size), 0);
    size_t send_total = 0;
    bool local_fits = true;
    for (int destination = 0; destination < comm_size; ++destination)
    {
        const std::vector<int_t> &payload =
            outbound[static_cast<size_t>(destination)];
        if (payload.size() >
                static_cast<size_t>(
                    std::numeric_limits<unsigned long long>::max()) ||
            send_total > std::numeric_limits<size_t>::max() - payload.size())
            ABORT("SymFact V2 partner demand payload size overflows.");
        send_counts[static_cast<size_t>(destination)] =
            static_cast<unsigned long long>(payload.size());
        send_displs[static_cast<size_t>(destination)] = send_total;
        send_total += payload.size();
        local_fits = local_fits &&
            payload.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            send_displs[static_cast<size_t>(destination)] <=
                static_cast<size_t>(std::numeric_limits<int>::max());
    }

    if (MPI_Alltoall(send_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                     recv_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                     comm) != MPI_SUCCESS)
        ABORT("SymFact V2 partner demand size exchange failed.");

    std::vector<size_t> recv_displs(static_cast<size_t>(comm_size), 0);
    size_t recv_total = 0;
    for (int source = 0; source < comm_size; ++source)
    {
        unsigned long long count = recv_counts[static_cast<size_t>(source)];
        if (count > static_cast<unsigned long long>(
                        std::numeric_limits<size_t>::max()) ||
            recv_total > std::numeric_limits<size_t>::max() -
                             static_cast<size_t>(count))
            ABORT("SymFact V2 partner demand receive size overflows.");
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
        ABORT("SymFact V2 partner demand route-size reduction failed.");

    std::vector<int_t> send_payload(send_total);
    for (int destination = 0; destination < comm_size; ++destination)
        std::copy(outbound[static_cast<size_t>(destination)].begin(),
                  outbound[static_cast<size_t>(destination)].end(),
                  send_payload.begin() +
                      send_displs[static_cast<size_t>(destination)]);

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
        ABORT("SymFact V2 partner demand exchange failed.");
    return received;
}

template <typename Ftype>
static void symldl_v2_compute_partner_demand_profile(
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
static void symldl_v2_build_partner_demand_plan(
    xLUstruct_t<Ftype> *lu,
    const SymLDLV2PartnerMetaPayload &metadata)
{
    double plan_start = SuperLU_timer_();
    size_t send_slots = symldl_v2_checked_product(
        symldl_v2_checked_product(
            static_cast<size_t>(lu->symV2PanelCount()),
            static_cast<size_t>(lu->Pc),
            "SymFact V2 partner demand send table overflows."),
        static_cast<size_t>(lu->Pr),
        "SymFact V2 partner demand send table overflows.");
    size_t recv_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 partner demand receive table overflows.");
    if (lu->symV2PartnerLSendRowActive.size() != send_slots ||
        lu->symV2PartnerLRecvActive.size() != recv_slots)
        ABORT("SymFact V2 partner demand tables are not initialized.");
    std::fill(lu->symV2PartnerLSendRowActive.begin(),
              lu->symV2PartnerLSendRowActive.end(), 0);
    std::fill(lu->symV2PartnerLRecvActive.begin(),
              lu->symV2PartnerLRecvActive.end(), 0);
    lu->symV2PartnerDemandRecords = 0;
    lu->symV2PartnerDemandPayloadBytes = 0;
    lu->symV2PartnerMetadataPayloadBytes = 0;
    lu->symV2PartnerDemandExchangeTime = 0.0;

    int comm_rank = -1;
    int comm_size = 0;
    MPI_Comm_rank(lu->grid->comm, &comm_rank);
    MPI_Comm_size(lu->grid->comm, &comm_size);
    if (metadata.comm_size != comm_size || comm_rank < 0 ||
        static_cast<size_t>(comm_rank) >= metadata.counts.size())
        ABORT("SymFact V2 partner demand metadata communicator is invalid.");
    size_t metadata_bytes = symldl_v2_checked_product(
        metadata.counts[static_cast<size_t>(comm_rank)], sizeof(int_t),
        "SymFact V2 partner metadata profile overflows.");
    symldl_v2_partner_profile_add(
        &lu->symV2PartnerMetadataPayloadBytes,
        metadata_bytes,
        "SymFact V2 partner metadata profile overflows.");

    // Gather the row-side block gids this rank can pair with each panel.
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
                ABORT("SymFact V2 partner demand metadata is truncated.");
            int_t target_pc = metadata.payload[position++];
            int_t k = metadata.payload[position++];
            int_t length = metadata.payload[position++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k < 0 || k >= lu->nsupers || length < 0 ||
                position + static_cast<size_t>(length) > end)
                ABORT("SymFact V2 partner demand metadata is invalid.");
            size_t record_end = position + static_cast<size_t>(length);
            if (source_pr == lu->myrow)
            {
                std::vector<int_t> &rows =
                    row_blocks_by_panel[static_cast<size_t>(k)];
                while (position < record_end)
                {
                    if (position + 2 > record_end)
                        ABORT("SymFact V2 partner demand block is truncated.");
                    int_t gid = metadata.payload[position++];
                    int_t count = metadata.payload[position++];
                    if (count < 0 ||
                        position + static_cast<size_t>(count) > record_end)
                        ABORT("SymFact V2 partner demand block is invalid.");
                    rows.push_back(gid);
                    position += static_cast<size_t>(count);
                }
            }
            position = record_end;
        }
    }
    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        std::vector<int_t> &rows =
            row_blocks_by_panel[static_cast<size_t>(k)];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }

    // Demand a whole source chunk only when it can update a local block.
    for (int rank = 0; rank < metadata.comm_size; ++rank)
    {
        size_t position = metadata.displs[static_cast<size_t>(rank)];
        size_t end = position + metadata.counts[static_cast<size_t>(rank)];
        int source_pr = MYROW(rank, lu->grid);
        while (position < end)
        {
            if (position + 3 > end)
                ABORT("SymFact V2 partner demand metadata is truncated.");
            int_t target_pc = metadata.payload[position++];
            int_t k = metadata.payload[position++];
            int_t length = metadata.payload[position++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k < 0 || k >= lu->nsupers || length < 0 ||
                position + static_cast<size_t>(length) > end)
                ABORT("SymFact V2 partner demand metadata is invalid.");
            size_t record_end = position + static_cast<size_t>(length);
            bool needed = false;
            while (position < record_end)
            {
                if (position + 2 > record_end)
                    ABORT("SymFact V2 partner demand block is truncated.");
                int_t gj = metadata.payload[position++];
                int_t count = metadata.payload[position++];
                if (count < 0 ||
                    position + static_cast<size_t>(count) > record_end)
                    ABORT("SymFact V2 partner demand block is invalid.");
                if (!needed && target_pc == lu->mycol)
                    needed = symldl_v2_partner_chunk_is_needed(
                        lu, gj,
                        row_blocks_by_panel[static_cast<size_t>(k)]);
                position += static_cast<size_t>(count);
            }
            if (needed)
            {
                size_t receive_slot = static_cast<size_t>(k) * lu->Pr +
                                      static_cast<size_t>(source_pr);
                lu->symV2PartnerLRecvActive[receive_slot] = 1;
            }
        }
    }

    // Route destination demand back to the rank that owns each source chunk.
    std::vector<std::vector<int_t> > outbound(
        static_cast<size_t>(comm_size));
    for (int_t k = 0; k < lu->nsupers; ++k)
        for (int source_pr = 0; source_pr < lu->Pr; ++source_pr)
        {
            size_t receive_slot = static_cast<size_t>(k) * lu->Pr +
                                  static_cast<size_t>(source_pr);
            if (!lu->symV2PartnerLRecvActive[receive_slot])
                continue;
            int source_pc = static_cast<int>(lu->symV2PanelRoot(k));
            int source_rank = PNUM(source_pr, source_pc, lu->grid);
            if (source_rank < 0 || source_rank >= comm_size)
                ABORT("SymFact V2 partner demand source is invalid.");
            std::vector<int_t> &payload =
                outbound[static_cast<size_t>(source_rank)];
            payload.push_back(source_rank);
            payload.push_back(lu->mycol);
            payload.push_back(lu->myrow);
            payload.push_back(k);
            payload.push_back(source_pr);
            if (lu->symV2PartnerDemandRecords ==
                std::numeric_limits<uint64_t>::max())
                ABORT("SymFact V2 partner demand record count overflows.");
            ++lu->symV2PartnerDemandRecords;
        }
    size_t local_payload_entries = 0;
    for (int destination = 0; destination < comm_size; ++destination)
    {
        const std::vector<int_t> &payload =
            outbound[static_cast<size_t>(destination)];
        if (local_payload_entries >
            std::numeric_limits<size_t>::max() - payload.size())
            ABORT("SymFact V2 partner demand payload size overflows.");
        local_payload_entries += payload.size();
    }
    size_t payload_bytes = symldl_v2_checked_product(
        local_payload_entries, sizeof(int_t),
        "SymFact V2 partner demand profile overflows.");
    symldl_v2_partner_profile_add(
        &lu->symV2PartnerDemandPayloadBytes,
        payload_bytes,
        "SymFact V2 partner demand profile overflows.");

    double exchange_start = SuperLU_timer_();
    std::vector<int_t> demands = symldl_v2_route_partner_demands(
        outbound, lu->grid->comm);
    lu->symV2PartnerDemandExchangeTime =
        SuperLU_timer_() - exchange_start;
    if (demands.size() % 5 != 0)
        ABORT("SymFact V2 partner demand payload is truncated.");
    for (size_t position = 0; position < demands.size(); position += 5)
    {
        int_t source_rank = demands[position];
        int_t target_pc = demands[position + 1];
        int_t destination_pr = demands[position + 2];
        int_t k = demands[position + 3];
        int_t source_pr = demands[position + 4];
        if (source_rank < 0 || source_rank >= comm_size ||
            target_pc < 0 || target_pc >= lu->Pc ||
            destination_pr < 0 || destination_pr >= lu->Pr ||
            k < 0 || k >= lu->nsupers ||
            source_pr < 0 || source_pr >= lu->Pr)
            ABORT("SymFact V2 partner demand record is invalid.");
        if (source_rank != lu->iam)
            continue;
        if (source_pr != lu->myrow ||
            lu->symV2PanelRoot(k) != lu->mycol)
            ABORT("SymFact V2 partner demand reached the wrong source.");
        int_t local_panel = lu->symV2PanelIndex(k);
        if (local_panel < 0 || local_panel >= lu->symV2PanelCount() ||
            lu->symV2PanelGid(local_panel) != k)
            ABORT("SymFact V2 partner demand source panel is missing.");
        size_t flat = static_cast<size_t>(local_panel) * lu->Pc +
                      static_cast<size_t>(target_pc);
        size_t active_pos = flat * static_cast<size_t>(lu->Pr) +
                            static_cast<size_t>(destination_pr);
        if (active_pos >= lu->symV2PartnerLSendRowActive.size())
            ABORT("SymFact V2 partner demand send index is invalid.");
        lu->symV2PartnerLSendRowActive[active_pos] = 1;
    }

    uint64_t local_receive_records = 0;
    uint64_t local_source_records = 0;
    for (size_t slot = 0; slot < lu->symV2PartnerLRecvActive.size(); ++slot)
        local_receive_records += lu->symV2PartnerLRecvActive[slot] ? 1 : 0;
    for (size_t pos = 0; pos < lu->symV2PartnerLSendRowActive.size(); ++pos)
        local_source_records += lu->symV2PartnerLSendRowActive[pos] ? 1 : 0;
    uint64_t global_receive_records = 0;
    uint64_t global_source_records = 0;
    if (MPI_Allreduce(&local_receive_records, &global_receive_records, 1,
                      MPI_UINT64_T, MPI_SUM, lu->grid->comm) != MPI_SUCCESS ||
        MPI_Allreduce(&local_source_records, &global_source_records, 1,
                      MPI_UINT64_T, MPI_SUM,
                      lu->grid->comm) != MPI_SUCCESS)
        ABORT("SymFact V2 partner demand validation reduction failed.");
    if (global_receive_records != global_source_records)
        ABORT("SymFact V2 partner demand send/receive counts do not match.");

    symldl_v2_compute_partner_demand_profile(lu);
    lu->symV2PartnerDemandPlanTime = SuperLU_timer_() - plan_start;
}

template <typename Ftype>
static void symldl_v2_validate_partner_demand_plan(
    xLUstruct_t<Ftype> *lu)
{
    uint64_t local[4] = {0, 0, 0, 0};
    for (int_t local_panel = 0; local_panel < lu->symV2PanelCount();
         ++local_panel)
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(local_panel) * lu->Pc +
                          static_cast<size_t>(pc);
            size_t values = symldl_v2_partner_send_value_count(lu, flat);
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos = flat * static_cast<size_t>(lu->Pr) +
                                    static_cast<size_t>(pr);
                if (!lu->symV2PartnerLSendRowActive[active_pos])
                    continue;
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
        }

    for (int_t k = 0; k < lu->nsupers; ++k)
        for (int source_pr = 0; source_pr < lu->Pr; ++source_pr)
        {
            size_t slot = static_cast<size_t>(k) * lu->Pr +
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
            int source_pc = static_cast<int>(lu->symV2PanelRoot(k));
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
