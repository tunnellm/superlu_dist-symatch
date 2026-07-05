#pragma once

#include <limits>
#include <vector>

#include "xlupanels.hpp"

#ifdef HAVE_CUDA

struct SymLDLV2PartnerMetaPayload
{
    int comm_size;
    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<int_t> payload;
};

template <typename Ftype>
static SymLDLV2PartnerMetaPayload
symldl_v2_collect_partner_l_metadata(xLUstruct_t<Ftype> *lu)
{
    SymLDLV2PartnerMetaPayload result;
    result.comm_size = 0;
    MPI_Comm_size(lu->grid->comm, &result.comm_size);

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
    if (local_meta_payload.size() >
        static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 partner metadata payload exceeds MPI limit.");

    int local_meta_count = static_cast<int>(local_meta_payload.size());
    result.counts.assign(static_cast<size_t>(result.comm_size), 0);
    MPI_Allgather(&local_meta_count, 1, MPI_INT,
                  result.counts.data(), 1, MPI_INT, lu->grid->comm);

    result.displs.assign(static_cast<size_t>(result.comm_size), 0);
    long long total_meta_count = 0;
    for (int r = 0; r < result.comm_size; ++r)
    {
        if (result.counts[r] < 0)
            ABORT("SymFact V2 partner metadata count is invalid.");
        if (total_meta_count >
            static_cast<long long>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 partner metadata payload exceeds MPI limit.");
        result.displs[r] = static_cast<int>(total_meta_count);
        total_meta_count += result.counts[r];
    }
    if (total_meta_count >
        static_cast<long long>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 partner metadata payload exceeds MPI limit.");

    result.payload.assign(static_cast<size_t>(total_meta_count), 0);
    MPI_Allgatherv(local_meta_payload.empty() ? NULL
                                             : local_meta_payload.data(),
                   local_meta_count, mpi_int_t,
                   result.payload.empty() ? NULL : result.payload.data(),
                   result.counts.data(), result.displs.data(), mpi_int_t,
                   lu->grid->comm);
    return result;
}

#endif
