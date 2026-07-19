#pragma once

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <vector>

#include "xlupanels.hpp"

struct SymLDLV2PartnerMetaPayload
{
    int comm_size;
    std::vector<size_t> counts;
    std::vector<size_t> displs;
    std::vector<int_t> payload;
};

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

template <typename Ftype>
static SymLDLV2PartnerMetaPayload
symldl_v2_collect_partner_l_metadata(xLUstruct_t<Ftype> *lu)
{
    const bool profile = superlu_sym_v2_route_profile();
    const double gather_start = profile ? SuperLU_timer_() : 0.0;
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
    SymLDLV2PartnerMetaPayload result = symldl_v2_allgather_metadata(
        local_meta_payload, lu->grid->comm,
        &lu->symV2CpuOversizedMpiChunks);
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
    return result;
}
