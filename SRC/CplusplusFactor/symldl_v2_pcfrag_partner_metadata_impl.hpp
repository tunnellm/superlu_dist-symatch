#pragma once

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "xlupanels.hpp"

struct SymLDLV2PartnerMetaPayload
{
    int comm_size;
    std::vector<size_t> counts;
    std::vector<size_t> displs;
    std::vector<int> source_rows;
    std::vector<int_t> payload;
};

struct SymLDLV2ScopedPartnerMetaPayload
{
    // Source lookup is row-local; receive demand is column-local after filtering.
    SymLDLV2PartnerMetaPayload source_row;
    SymLDLV2PartnerMetaPayload target_column;
};

template <typename Ftype>
static int symldl_v2_partner_metadata_source_row(
    const SymLDLV2PartnerMetaPayload &metadata, int segment,
    const xLUstruct_t<Ftype> *lu)
{
    if (segment < 0 || segment >= metadata.comm_size)
        ABORT("SymFact V2 metadata segment is invalid.");
    if (!metadata.source_rows.empty() &&
        metadata.source_rows.size() !=
            static_cast<size_t>(metadata.comm_size))
        ABORT("SymFact V2 metadata source-row table is invalid.");
    int source_pr = metadata.source_rows.empty()
                        ? MYROW(segment, lu->grid)
                        : metadata.source_rows[static_cast<size_t>(segment)];
    if (source_pr < 0 || source_pr >= lu->Pr)
        ABORT("SymFact V2 metadata source row is invalid.");
    return source_pr;
}

static SymLDLV2PartnerMetaPayload symldl_v2_allgather_metadata(
    const std::vector<int_t> &local_payload, MPI_Comm comm,
    uint64_t *oversized_chunks)
{
    SymLDLV2PartnerMetaPayload result;
    result.comm_size = 0;
    MPI_Comm_size(comm, &result.comm_size);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (local_payload.size() >
        static_cast<size_t>(std::numeric_limits<unsigned long long>::max()))
        ABORT("SymFact V2 metadata payload size overflows.");
    unsigned long long local_count =
        static_cast<unsigned long long>(local_payload.size());
    std::vector<unsigned long long> gathered_counts(
        static_cast<size_t>(result.comm_size), 0);
    if (MPI_Allgather(&local_count, 1, MPI_UNSIGNED_LONG_LONG,
                      gathered_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                      comm) != MPI_SUCCESS)
        ABORT("SymFact V2 metadata size exchange failed.");

    result.counts.assign(static_cast<size_t>(result.comm_size), 0);
    result.displs.assign(static_cast<size_t>(result.comm_size), 0);
    size_t total = 0;
    bool fits_allgatherv = true;
    for (int source = 0; source < result.comm_size; ++source)
    {
        if (gathered_counts[source] >
            static_cast<unsigned long long>(
                std::numeric_limits<size_t>::max()))
            ABORT("SymFact V2 metadata payload size overflows.");
        size_t count = static_cast<size_t>(gathered_counts[source]);
        if (count > std::numeric_limits<size_t>::max() - total)
            ABORT("SymFact V2 metadata payload size overflows.");
        result.counts[source] = count;
        result.displs[source] = total;
        total += count;
        fits_allgatherv = fits_allgatherv &&
            count <= static_cast<size_t>(INT_MAX);
    }
    fits_allgatherv = fits_allgatherv &&
        total <= static_cast<size_t>(INT_MAX);
    result.payload.assign(total, 0);

    if (fits_allgatherv)
    {
        std::vector<int> counts(static_cast<size_t>(result.comm_size), 0);
        std::vector<int> displs(static_cast<size_t>(result.comm_size), 0);
        for (int source = 0; source < result.comm_size; ++source)
        {
            counts[source] = static_cast<int>(result.counts[source]);
            displs[source] = static_cast<int>(result.displs[source]);
        }
        if (MPI_Allgatherv(
                local_payload.empty() ? NULL : local_payload.data(),
                static_cast<int>(local_payload.size()), mpi_int_t,
                result.payload.empty() ? NULL : result.payload.data(),
                counts.data(), displs.data(), mpi_int_t, comm) != MPI_SUCCESS)
            ABORT("SymFact V2 metadata exchange failed.");
        return result;
    }

    for (int source = 0; source < result.comm_size; ++source)
    {
        int_t *destination = result.counts[source] == 0 ? NULL :
            result.payload.data() + result.displs[source];
        if (rank == source && result.counts[source] != 0)
            std::memcpy(destination, local_payload.data(),
                        result.counts[source] * sizeof(int_t));
        size_t offset = 0;
        while (offset < result.counts[source])
        {
            int chunk = static_cast<int>(std::min(
                static_cast<size_t>(INT_MAX),
                result.counts[source] - offset));
            if (MPI_Bcast(destination + offset, chunk, mpi_int_t,
                          source, comm) != MPI_SUCCESS)
                ABORT("SymFact V2 chunked metadata exchange failed.");
            offset += static_cast<size_t>(chunk);
            if (oversized_chunks != NULL)
                ++*oversized_chunks;
        }
    }
    return result;
}

static size_t symldl_v2_metadata_bytes(size_t entries)
{
    if (entries > std::numeric_limits<size_t>::max() / sizeof(int_t))
        ABORT("SymFact V2 metadata byte count overflows.");
    return entries * sizeof(int_t);
}

template <typename Ftype>
static std::vector<int_t> symldl_v2_build_local_partner_l_metadata(
    xLUstruct_t<Ftype> *lu)
{
    std::vector<int_t> local_meta_payload;
    for (int_t lk = 0; lk < lu->symV2PanelCount(); ++lk)
    {
        int_t k0 = lu->symV2PanelGid(lk);
        if (k0 < 0 || k0 >= lu->nsupers)
            continue;
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc);
            if (flat >= lu->symL2LSendMeta.size() ||
                lu->symL2LSendMeta[flat].empty())
                continue;
            if (lu->symL2LSendMeta[flat].size() >
                static_cast<size_t>(std::numeric_limits<int_t>::max()))
                ABORT("SymFact V2 partner metadata is too large.");
            local_meta_payload.push_back(pc);
            local_meta_payload.push_back(k0);
            local_meta_payload.push_back(
                static_cast<int_t>(lu->symL2LSendMeta[flat].size()));
            local_meta_payload.insert(local_meta_payload.end(),
                                      lu->symL2LSendMeta[flat].begin(),
                                      lu->symL2LSendMeta[flat].end());
        }
    }
    return local_meta_payload;
}

template <typename Ftype>
static void symldl_v2_profile_metadata_collective(
    xLUstruct_t<Ftype> *lu, const std::vector<int_t> &local_payload,
    int comm_size)
{
    if (!lu->SCT->factorCommProfileEnabled || comm_size <= 1)
        return;

    const unsigned long long limit =
        std::numeric_limits<unsigned long long>::max();
    const unsigned long long recipients =
        static_cast<unsigned long long>(comm_size - 1);
    const unsigned long long payload_bytes =
        static_cast<unsigned long long>(
            symldl_v2_metadata_bytes(local_payload.size()));
    const size_t payload_chunks = local_payload.empty()
                                      ? 0
                                      : (local_payload.size() - 1) /
                                                static_cast<size_t>(INT_MAX) +
                                            1;
    if (payload_chunks > static_cast<size_t>(limit - 1))
        ABORT("SymFact V2 metadata profile message count overflows.");
    const unsigned long long messages_per_recipient =
        1 + static_cast<unsigned long long>(payload_chunks);
    if (messages_per_recipient > limit / recipients)
        ABORT("SymFact V2 metadata profile message count overflows.");
    const unsigned long long messages =
        messages_per_recipient * recipients;
    if (payload_bytes > limit - sizeof(unsigned long long) ||
        payload_bytes + sizeof(unsigned long long) > limit / recipients)
        ABORT("SymFact V2 metadata profile byte count overflows.");
    const unsigned long long bytes =
        (payload_bytes + sizeof(unsigned long long)) * recipients;
    const unsigned long long max_payload_bytes = SUPERLU_MAX(
        static_cast<unsigned long long>(sizeof(unsigned long long)),
        static_cast<unsigned long long>(SUPERLU_MIN(
            local_payload.size(), static_cast<size_t>(INT_MAX))) *
            sizeof(int_t));
    if (lu->SCT->factorCommMetadataMessages > limit - messages ||
        lu->SCT->factorCommMetadataBytes > limit - bytes)
        ABORT("SymFact V2 metadata profile counter overflows.");
    lu->SCT->factorCommMetadataMessages += messages;
    lu->SCT->factorCommMetadataBytes += bytes;
    lu->SCT->factorCommMaxMetadataBytes = SUPERLU_MAX(
        lu->SCT->factorCommMaxMetadataBytes, max_payload_bytes);
}

static void symldl_v2_append_metadata_for_target(
    const SymLDLV2PartnerMetaPayload &metadata, int segment, int target_pc,
    int target_count, int_t panel_count, std::vector<int_t> *filtered)
{
    if (segment < 0 || segment >= metadata.comm_size || target_pc < 0 ||
        target_pc >= target_count || panel_count < 0 || filtered == NULL ||
        metadata.counts.size() != static_cast<size_t>(metadata.comm_size) ||
        metadata.displs.size() != static_cast<size_t>(metadata.comm_size))
        ABORT("SymFact V2 metadata filter segment is invalid.");
    size_t position = metadata.displs[static_cast<size_t>(segment)];
    size_t count = metadata.counts[static_cast<size_t>(segment)];
    if (position > metadata.payload.size() ||
        count > metadata.payload.size() - position)
        ABORT("SymFact V2 metadata filter bounds are invalid.");
    size_t end = position + count;
    while (position < end)
    {
        size_t record_begin = position;
        if (position + 3 > end)
            ABORT("SymFact V2 metadata filter payload is truncated.");
        int_t record_target = metadata.payload[position++];
        int_t k = metadata.payload[position++];
        int_t length = metadata.payload[position++];
        if (record_target < 0 || record_target >= target_count || k < 0 ||
            k >= panel_count || length < 0 ||
            static_cast<size_t>(length) > end - position)
            ABORT("SymFact V2 metadata filter payload is invalid.");
        size_t record_end = position + static_cast<size_t>(length);
        if (record_target == target_pc)
            filtered->insert(filtered->end(),
                             metadata.payload.begin() + record_begin,
                             metadata.payload.begin() + record_end);
        position = record_end;
    }
}

template <typename Ftype>
static void symldl_v2_verify_scoped_partner_metadata(
    xLUstruct_t<Ftype> *lu, const std::vector<int_t> &local_payload,
    const SymLDLV2ScopedPartnerMetaPayload &scoped)
{
    SymLDLV2PartnerMetaPayload global = symldl_v2_allgather_metadata(
        local_payload, lu->grid->comm, NULL);

    std::vector<int_t> expected_row;
    for (int pc = 0; pc < lu->Pc; ++pc)
    {
        int rank = PNUM(lu->myrow, pc, lu->grid);
        if (rank < 0 || rank >= global.comm_size)
            ABORT("SymFact V2 metadata verification rank is invalid.");
        size_t begin = global.displs[static_cast<size_t>(rank)];
        size_t count = global.counts[static_cast<size_t>(rank)];
        if (scoped.source_row.counts[static_cast<size_t>(pc)] != count)
            ABORT("SymFact V2 scoped row metadata count mismatch.");
        expected_row.insert(expected_row.end(),
                            global.payload.begin() + begin,
                            global.payload.begin() + begin + count);
    }
    if (expected_row != scoped.source_row.payload)
        ABORT("SymFact V2 scoped row metadata mismatch.");

    std::vector<int_t> expected_column;
    for (int pr = 0; pr < lu->Pr; ++pr)
    {
        size_t row_begin = expected_column.size();
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            int rank = PNUM(pr, pc, lu->grid);
            if (rank < 0 || rank >= global.comm_size)
                ABORT("SymFact V2 metadata verification rank is invalid.");
            symldl_v2_append_metadata_for_target(
                global, rank, lu->mycol, lu->Pc, lu->nsupers,
                &expected_column);
        }
        if (scoped.target_column.counts[static_cast<size_t>(pr)] !=
            expected_column.size() - row_begin)
            ABORT("SymFact V2 scoped column metadata count mismatch.");
    }
    if (expected_column != scoped.target_column.payload)
        ABORT("SymFact V2 scoped column metadata mismatch.");

    if (lu->grid3d->iam == 0)
    {
        std::printf(
            "SymFact V2 scoped fragment metadata verification: passed\n");
        std::fflush(stdout);
    }
}

template <typename Ftype>
static SymLDLV2PartnerMetaPayload
symldl_v2_collect_partner_l_metadata(xLUstruct_t<Ftype> *lu)
{
    const bool profile = superlu_sym_v2_route_profile() ||
                         superlu_sym_v2_factor_comm_profile();
    const double gather_start = profile ? SuperLU_timer_() : 0.0;
    std::vector<int_t> local_meta_payload =
        symldl_v2_build_local_partner_l_metadata(lu);
    SymLDLV2PartnerMetaPayload result = symldl_v2_allgather_metadata(
        local_meta_payload, lu->grid->comm,
        &lu->symV2CpuOversizedMpiChunks);
    result.source_rows.resize(static_cast<size_t>(result.comm_size));
    for (int rank = 0; rank < result.comm_size; ++rank)
        result.source_rows[static_cast<size_t>(rank)] =
            MYROW(rank, lu->grid);
    if (profile)
    {
        if (local_meta_payload.size() >
                std::numeric_limits<unsigned long long>::max() /
                    sizeof(int_t) ||
            result.payload.size() >
                std::numeric_limits<unsigned long long>::max() /
                    sizeof(int_t))
            ABORT("SymFact V2 metadata profile byte count overflows.");
        if (lu->symV2PartnerMetadataGatherCalls ==
                std::numeric_limits<unsigned long long>::max() ||
            lu->symV2PartnerMetadataLocalBytes >
                std::numeric_limits<unsigned long long>::max() -
                    local_meta_payload.size() * sizeof(int_t) ||
            lu->symV2PartnerMetadataReceivedBytes >
                std::numeric_limits<unsigned long long>::max() -
                    result.payload.size() * sizeof(int_t))
            ABORT("SymFact V2 metadata profile counter overflows.");
        ++lu->symV2PartnerMetadataGatherCalls;
        lu->symV2PartnerMetadataLocalBytes +=
            local_meta_payload.size() * sizeof(int_t);
        lu->symV2PartnerMetadataReceivedBytes +=
            result.payload.size() * sizeof(int_t);
        lu->symV2PartnerMetadataGatherTime +=
            SuperLU_timer_() - gather_start;
    }
    symldl_v2_profile_metadata_collective(
        lu, local_meta_payload, result.comm_size);
    return result;
}

template <typename Ftype>
static SymLDLV2ScopedPartnerMetaPayload
symldl_v2_collect_scoped_partner_l_metadata(xLUstruct_t<Ftype> *lu)
{
    int row_comm_size = 0;
    int row_comm_rank = -1;
    int column_comm_size = 0;
    int column_comm_rank = -1;
    MPI_Comm_size(lu->grid3d->rscp.comm, &row_comm_size);
    MPI_Comm_rank(lu->grid3d->rscp.comm, &row_comm_rank);
    MPI_Comm_size(lu->grid3d->cscp.comm, &column_comm_size);
    MPI_Comm_rank(lu->grid3d->cscp.comm, &column_comm_rank);
    if (row_comm_size != lu->Pc || row_comm_rank != lu->mycol ||
        column_comm_size != lu->Pr || column_comm_rank != lu->myrow)
        ABORT("SymFact V2 scoped metadata communicator is invalid.");

    const bool profile = superlu_sym_v2_route_profile() ||
                         superlu_sym_v2_factor_comm_profile();
    const double gather_start = profile ? SuperLU_timer_() : 0.0;
    std::vector<int_t> local_meta_payload =
        symldl_v2_build_local_partner_l_metadata(lu);

    SymLDLV2ScopedPartnerMetaPayload result;
    const double row_start = profile ? SuperLU_timer_() : 0.0;
    result.source_row = symldl_v2_allgather_metadata(
        local_meta_payload, lu->grid3d->rscp.comm,
        &lu->symV2CpuOversizedMpiChunks);
    const double row_time = profile ? SuperLU_timer_() - row_start : 0.0;
    result.source_row.source_rows.assign(
        static_cast<size_t>(result.source_row.comm_size), lu->myrow);

    std::vector<int_t> target_payload;
    for (int segment = 0; segment < result.source_row.comm_size; ++segment)
        symldl_v2_append_metadata_for_target(
            result.source_row, segment, lu->mycol, lu->Pc, lu->nsupers,
            &target_payload);

    const double column_start = profile ? SuperLU_timer_() : 0.0;
    result.target_column = symldl_v2_allgather_metadata(
        target_payload, lu->grid3d->cscp.comm,
        &lu->symV2CpuOversizedMpiChunks);
    const double column_time =
        profile ? SuperLU_timer_() - column_start : 0.0;
    result.target_column.source_rows.resize(
        static_cast<size_t>(result.target_column.comm_size));
    for (int pr = 0; pr < result.target_column.comm_size; ++pr)
        result.target_column.source_rows[static_cast<size_t>(pr)] = pr;

    if (superlu_sym_v2_scoped_fragment_metadata_verify())
        symldl_v2_verify_scoped_partner_metadata(
            lu, local_meta_payload, result);

    if (profile)
    {
        const size_t local_bytes =
            symldl_v2_metadata_bytes(local_meta_payload.size());
        const size_t row_bytes =
            symldl_v2_metadata_bytes(result.source_row.payload.size());
        const size_t target_bytes =
            symldl_v2_metadata_bytes(target_payload.size());
        const size_t column_bytes =
            symldl_v2_metadata_bytes(result.target_column.payload.size());
        if (row_bytes > std::numeric_limits<size_t>::max() - column_bytes ||
            local_bytes > std::numeric_limits<size_t>::max() - row_bytes ||
            local_bytes + row_bytes >
                std::numeric_limits<size_t>::max() - target_bytes ||
            local_bytes + row_bytes + target_bytes >
                std::numeric_limits<size_t>::max() - column_bytes)
            ABORT("SymFact V2 scoped metadata peak size overflows.");
        const size_t peak_bytes =
            local_bytes + row_bytes + target_bytes + column_bytes;
        const unsigned long long limit =
            std::numeric_limits<unsigned long long>::max();
        if (lu->symV2PartnerMetadataGatherCalls > limit - 2 ||
            lu->symV2PartnerMetadataLocalBytes > limit - local_bytes ||
            row_bytes > limit - column_bytes ||
            lu->symV2PartnerMetadataReceivedBytes >
                limit - row_bytes - column_bytes ||
            lu->symV2PartnerMetadataRowReceivedBytes > limit - row_bytes ||
            lu->symV2PartnerMetadataColumnReceivedBytes >
                limit - column_bytes)
            ABORT("SymFact V2 scoped metadata profile counter overflows.");
        lu->symV2PartnerMetadataGatherCalls += 2;
        lu->symV2PartnerMetadataLocalBytes += local_bytes;
        lu->symV2PartnerMetadataReceivedBytes += row_bytes + column_bytes;
        lu->symV2PartnerMetadataRowReceivedBytes += row_bytes;
        lu->symV2PartnerMetadataColumnReceivedBytes += column_bytes;
        lu->symV2PartnerMetadataScopedPeakBytes = SUPERLU_MAX(
            lu->symV2PartnerMetadataScopedPeakBytes,
            static_cast<unsigned long long>(peak_bytes));
        lu->symV2PartnerMetadataGatherTime +=
            SuperLU_timer_() - gather_start;
        lu->symV2PartnerMetadataRowGatherTime += row_time;
        lu->symV2PartnerMetadataColumnGatherTime += column_time;
    }
    symldl_v2_profile_metadata_collective(
        lu, local_meta_payload, row_comm_size);
    symldl_v2_profile_metadata_collective(
        lu, target_payload, column_comm_size);
    return result;
}
