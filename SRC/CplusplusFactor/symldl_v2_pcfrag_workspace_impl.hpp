#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

template <typename Ftype>
static size_t symldl_v2_alloc_bytes(int_t count, size_t elem_size,
                                    const char *what)
{
    if (count < 0)
        ABORT(what);
    size_t n = static_cast<size_t>(count);
    if (elem_size != 0 && n > std::numeric_limits<size_t>::max() / elem_size)
        ABORT(what);
    return n * elem_size;
}

static inline void symldl_v2_cuda_malloc_or_abort(void **ptr, size_t bytes,
                                                 const char *what)
{
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err == cudaSuccess)
        return;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (mem_err == cudaSuccess)
        fprintf(stderr,
                "%s: requested %zu bytes, free %zu bytes, total %zu bytes.\n",
                what, bytes, free_bytes, total_bytes);
    else
        fprintf(stderr, "%s: requested %zu bytes.\n", what, bytes);
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    ABORT("SymFact V2 GPU allocation failed.");
}

template <typename Ftype>
static void symldl_v2_trace_pcfrag_plan(xLUstruct_t<Ftype> *lu,
                                        const char *where, int_t k,
                                        int_t needed_count)
{
    if (!superlu_sym_v2_trace_pcfrag())
        return;

    static int printed = 0;
    if (printed >= 64 || lu == NULL || lu->grid3d == NULL ||
        k < 0 || k >= lu->nsupers)
        return;

    const std::vector<int_t> &partner =
        lu->symV2PartnerLRecvIndex[static_cast<size_t>(k)];
    const std::vector<int_t> &row =
        lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
    int_t partner_blocks = partner.empty() ? 0 : partner[0];
    int_t partner_rows = partner.empty() ? 0 : partner[1];
    int_t row_blocks = row.empty() ? 0 : row[0];
    int_t row_rows = row.empty() ? 0 : row[1];

    long long partner_recv = 0;
    size_t partner_base =
        static_cast<size_t>(k) * static_cast<size_t>(lu->Pr);
    if (partner_base + static_cast<size_t>(lu->Pr) <=
        lu->symV2PartnerLRecvSizes.size())
    {
        for (int pr = 0; pr < lu->Pr; ++pr)
            partner_recv += lu->symV2PartnerLRecvSizes[
                partner_base + static_cast<size_t>(pr)];
    }

    long long row_recv = 0;
    size_t row_base = static_cast<size_t>(k) * static_cast<size_t>(lu->Pc);
    if (row_base + static_cast<size_t>(lu->Pc) <=
        lu->symV2RowFragRecvSizes.size())
    {
        for (int pc = 0; pc < lu->Pc; ++pc)
            row_recv += lu->symV2RowFragRecvSizes[
                row_base + static_cast<size_t>(pc)];
    }

    long long row_send = 0;
    int_t lk = lu->symV2PanelIndex(k);
    if (lk >= 0)
    {
        size_t send_base =
            static_cast<size_t>(lk) * static_cast<size_t>(lu->Pc);
        if (send_base + static_cast<size_t>(lu->Pc) <=
            lu->symV2RowDownSendSizes.size())
        {
            for (int pc = 0; pc < lu->Pc; ++pc)
                row_send += lu->symV2RowDownSendSizes[
                    send_base + static_cast<size_t>(pc)];
        }
    }

    if (needed_count == 0 && partner_blocks == 0 && row_blocks == 0 &&
        partner_recv == 0 && row_recv == 0 && row_send == 0)
        return;

    std::fprintf(stderr,
                 "[symv2-pcfrag] rank %d %s k %d panel_root %d diag_root %d needed %lld partner_blocks %lld partner_rows %lld partner_recv %lld row_blocks %lld row_rows %lld row_recv %lld row_send %lld\n",
                 lu->grid3d->iam, where, static_cast<int>(k),
                 static_cast<int>(lu->symV2PanelRoot(k)),
                 static_cast<int>(lu->symV2DiagRoot(k)),
                 static_cast<long long>(needed_count),
                 static_cast<long long>(partner_blocks),
                 static_cast<long long>(partner_rows), partner_recv,
                 static_cast<long long>(row_blocks),
                 static_cast<long long>(row_rows), row_recv, row_send);
    std::fflush(stderr);
    ++printed;
}

template <typename Ftype>
static void symldl_v2_compute_pcfrag_scratch(
    xLUstruct_t<Ftype> *lu,
    LUStruct_type<Ftype> *)
{
    if (!lu->useSymV2Solve())
    {
        lu->maxSymPartnerLvalCount = lu->maxLvalCount;
        lu->maxSymPartnerLidxCount = lu->maxLidxCount;
        lu->maxSymPartnerLSendStageCount = 0;
        return;
    }

    lu->maxSymPartnerLvalCount = 0;
    lu->maxSymPartnerLidxCount = 0;
    lu->maxSymPartnerLSendStageCount = 0;
    lu->maxSymV2RowFragStageCount = 0;
    lu->maxSymV2RowFragValRecvCount = 0;
    lu->maxSymV2RowFragIdxRecvCount = 0;
    lu->maxSymV2RowFragValSendCount = 0;

    if (lu->Pr <= 1)
    {
        lu->maxSymPartnerLvalCount = lu->maxLvalCount;
        lu->maxSymPartnerLidxCount = lu->maxLidxCount;
        return;
    }

    size_t partner_count_size = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pc),
        "SymFact V2 partner-L count table overflows.");
    size_t row_source_count_size = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 row-fragment count table overflows.");
    if (partner_count_size > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        row_source_count_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 fragment count table exceeds MPI limit.");

    std::vector<long long> local_partner_val(partner_count_size, 0);
    std::vector<long long> global_partner_val(partner_count_size, 0);
    std::vector<long long> local_partner_meta(partner_count_size, 0);
    std::vector<long long> global_partner_meta(partner_count_size, 0);
    std::vector<long long> local_row_val(row_source_count_size, 0);
    std::vector<long long> global_row_val(row_source_count_size, 0);
    std::vector<long long> local_row_meta(row_source_count_size, 0);
    std::vector<long long> global_row_meta(row_source_count_size, 0);
    long long local_row_send_val = 0;
    long long max_row_send_val = 0;

    for (int_t li = 0; li < lu->symV2PanelCount(); ++li)
    {
        int_t k0 = lu->symV2PanelGid(li);
        if (k0 < 0 || k0 >= lu->nsupers ||
            lu->isNodeInMyGrid == NULL || lu->isNodeInMyGrid[k0] != 1)
            continue;

        xlpanel_t<Ftype> &lpanel = lu->lPanelVec[li];
        if (lpanel.isEmpty())
            continue;

        int_t knsupc = lu->supersize(k0);
        int_t first_block = lpanel.haveDiag() ? 1 : 0;
        for (int_t lb = first_block; lb < lpanel.nblocks(); ++lb)
        {
            int_t ik = lpanel.gid(lb);
            int ikcol = static_cast<int>(lu->symV2PanelRoot(ik));
            if (ikcol < 0 || ikcol >= lu->Pc)
                ABORT("SymFact V2 partner-L target process column is invalid.");
            int_t len = lpanel.nbrow(lb);
            if (len <= 0)
                continue;

            long long block_values =
                static_cast<long long>(len) *
                static_cast<long long>(knsupc);
            size_t col_pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(lu->Pc) +
                             static_cast<size_t>(ikcol);
            size_t row_pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(lu->Pr) +
                             static_cast<size_t>(lu->myrow);
            local_partner_val[col_pos] += block_values;
            local_row_val[row_pos] += block_values;
            local_partner_meta[col_pos] += static_cast<long long>(len) + 2;
            local_row_meta[row_pos] += static_cast<long long>(len) + 2;
            local_row_send_val =
                SUPERLU_MAX(local_row_send_val, block_values);
        }
    }

    MPI_Allreduce(local_partner_val.data(), global_partner_val.data(),
                  static_cast<int>(partner_count_size), MPI_LONG_LONG,
                  MPI_SUM, lu->grid->comm);
    MPI_Allreduce(local_partner_meta.data(), global_partner_meta.data(),
                  static_cast<int>(partner_count_size), MPI_LONG_LONG,
                  MPI_SUM, lu->grid->comm);
    MPI_Allreduce(local_row_val.data(), global_row_val.data(),
                  static_cast<int>(row_source_count_size), MPI_LONG_LONG,
                  MPI_SUM, lu->grid->comm);
    MPI_Allreduce(local_row_meta.data(), global_row_meta.data(),
                  static_cast<int>(row_source_count_size), MPI_LONG_LONG,
                  MPI_SUM, lu->grid->comm);
    MPI_Allreduce(&local_row_send_val, &max_row_send_val, 1,
                  MPI_LONG_LONG, MPI_MAX, lu->grid->comm);

    long long max_partner_val = 0;
    long long max_partner_meta = 0;
    long long max_row_val = 0;
    long long max_row_meta = 0;
    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t pos = static_cast<size_t>(k0) *
                             static_cast<size_t>(lu->Pc) +
                         static_cast<size_t>(pc);
            max_partner_val =
                SUPERLU_MAX(max_partner_val, global_partner_val[pos]);
            max_partner_meta =
                SUPERLU_MAX(max_partner_meta, global_partner_meta[pos]);
        }
        if (superlu_sym_v2_pc_fragment_schur() &&
            superlu_sym_v2_pc_fragment_ldl_native() && lu->Pc > 1)
        {
            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(lu->Pr) +
                             static_cast<size_t>(pr);
                max_row_val =
                    SUPERLU_MAX(max_row_val, global_row_val[pos]);
                max_row_meta =
                    SUPERLU_MAX(max_row_meta, global_row_meta[pos]);
            }
        }
    }

    long long max_partner_idx =
        (max_partner_meta > 0) ? max_partner_meta + LPANEL_HEADER_SIZE + 1 : 0;
    long long max_row_idx =
        (max_row_meta > 0) ? max_row_meta + LPANEL_HEADER_SIZE + 1 : 0;
    if (superlu_sym_v2_pc_fragment_schur() &&
        superlu_sym_v2_pc_fragment_ldl_native() && lu->Pc > 1)
    {
        long long row_send_multiplier = static_cast<long long>(lu->Pc - 1);
        if (max_row_val > 0 &&
            row_send_multiplier >
                std::numeric_limits<long long>::max() / max_row_val)
            ABORT("SymFact V2 row-fragment send size overflows.");
        max_row_send_val =
            SUPERLU_MAX(max_row_send_val, max_row_val * row_send_multiplier);
    }

    if (max_partner_val > static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_partner_idx > static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_row_val > static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_row_idx > static_cast<long long>(std::numeric_limits<int_t>::max()) ||
        max_row_send_val > static_cast<long long>(std::numeric_limits<int_t>::max()))
        ABORT("SymFact V2 fragment staging size exceeds int_t range.");

    lu->maxSymPartnerLvalCount = static_cast<int_t>(max_partner_val);
    lu->maxSymPartnerLidxCount = static_cast<int_t>(max_partner_idx);
    lu->maxSymPartnerLSendStageCount = lu->maxLvalCount;
    lu->maxSymV2RowFragStageCount = static_cast<int_t>(max_row_val);
    lu->maxSymV2RowFragValRecvCount = static_cast<int_t>(max_row_val);
    lu->maxSymV2RowFragIdxRecvCount = static_cast<int_t>(max_row_idx);
    lu->maxSymV2RowFragValSendCount = static_cast<int_t>(max_row_send_val);
}

template <typename Ftype>
static void symldl_v2_allocate_fragment_host_buffers(xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve())
        return;

    int nstreams = lu->options != NULL ? lu->options->num_lookaheads : 0;
    if (nstreams <= 0)
        return;

    lu->symPartnerLvalRecvBufs.assign(static_cast<size_t>(nstreams), NULL);
    lu->symPartnerLidxRecvBufs.assign(static_cast<size_t>(nstreams), NULL);

#ifdef HAVE_CUDA
    bool pc_fragment = symldl_v2_use_pc_fragment_schur(lu->grid3d);
    lu->symV2RowFragHostRecvBufs.assign(static_cast<size_t>(nstreams), NULL);
    lu->symV2RowFragHostSendBufs.assign(static_cast<size_t>(nstreams), NULL);

    bool pooled_pinned =
        lu->superlu_acc_offload && !superlu_cuda_aware_mpi() &&
        superlu_sym_v2_pinned_staging() &&
        superlu_sym_v2_pinned_staging_pool();
    if (pooled_pinned && lu->maxSymPartnerLvalCount > 0)
    {
        gpuErrchk(cudaMallocHost(
            (void **) &lu->symV2PartnerLHostRecvPoolPinned,
            symldl_v2_alloc_bytes<Ftype>(
                lu->maxSymPartnerLvalCount, sizeof(Ftype),
                "SymFact V2 pooled partner receive staging overflows.")));
        lu->symV2PartnerLHostRecvPoolPinnedCount =
            static_cast<size_t>(lu->maxSymPartnerLvalCount);
        lu->symV2PartnerLHostRecvPinned = 1;
    }
    if (pooled_pinned && pc_fragment &&
        lu->maxSymV2RowFragStageCount > 0)
    {
        gpuErrchk(cudaMallocHost(
            (void **) &lu->symV2RowFragHostRecvPoolPinned,
            symldl_v2_alloc_bytes<Ftype>(
                lu->maxSymV2RowFragStageCount, sizeof(Ftype),
                "SymFact V2 pooled row receive staging overflows.")));
        lu->symV2RowFragHostRecvPoolPinnedCount =
            static_cast<size_t>(lu->maxSymV2RowFragStageCount);
        lu->symV2RowFragHostRecvPinned = 1;
    }
    if (pooled_pinned && pc_fragment &&
        superlu_sym_v2_row_l_separate_send_staging())
    {
        int_t row_send_count = SUPERLU_MAX(lu->maxSymV2RowFragStageCount,
                                           lu->maxSymV2RowFragValSendCount);
        if (row_send_count > 0)
        {
            gpuErrchk(cudaMallocHost(
                (void **) &lu->symV2RowFragHostSendPoolPinned,
                symldl_v2_alloc_bytes<Ftype>(
                    row_send_count, sizeof(Ftype),
                    "SymFact V2 pooled row send staging overflows.")));
            lu->symV2RowFragHostSendPoolPinnedCount =
                static_cast<size_t>(row_send_count);
            lu->symV2RowFragHostSendPinned = 1;
        }
    }
#endif

    for (int s = 0; s < nstreams; ++s)
    {
        size_t partner_val_bytes = symldl_v2_alloc_bytes<Ftype>(
            lu->maxSymPartnerLvalCount, sizeof(Ftype),
            "SymFact V2 partner receive staging overflows.");
        size_t partner_idx_bytes = symldl_v2_alloc_bytes<Ftype>(
            lu->maxSymPartnerLidxCount, sizeof(int_t),
            "SymFact V2 partner index staging overflows.");

#ifdef HAVE_CUDA
        if (lu->symV2PartnerLHostRecvPoolPinned != NULL)
            lu->symPartnerLvalRecvBufs[s] =
                lu->symV2PartnerLHostRecvPoolPinned;
        else if (partner_val_bytes != 0 && lu->superlu_acc_offload &&
                 superlu_sym_v2_pinned_staging())
        {
            gpuErrchk(cudaMallocHost(
                (void **) &lu->symPartnerLvalRecvBufs[s],
                partner_val_bytes));
            lu->symV2PartnerLHostRecvPinned = 1;
        }
        else
#endif
        if (partner_val_bytes != 0)
            lu->symPartnerLvalRecvBufs[s] =
                (Ftype *) SUPERLU_MALLOC(partner_val_bytes);
        if (partner_idx_bytes != 0)
            lu->symPartnerLidxRecvBufs[s] =
                (int_t *) SUPERLU_MALLOC(partner_idx_bytes);

#ifdef HAVE_CUDA
        if (pc_fragment)
        {
            size_t row_recv_bytes = symldl_v2_alloc_bytes<Ftype>(
                lu->maxSymV2RowFragStageCount, sizeof(Ftype),
                "SymFact V2 row receive staging overflows.");
            int_t row_send_count = SUPERLU_MAX(lu->maxSymV2RowFragStageCount,
                                               lu->maxSymV2RowFragValSendCount);
            size_t row_send_bytes = symldl_v2_alloc_bytes<Ftype>(
                row_send_count, sizeof(Ftype),
                "SymFact V2 row send staging overflows.");
            if (lu->symV2RowFragHostRecvPoolPinned != NULL)
                lu->symV2RowFragHostRecvBufs[s] =
                    lu->symV2RowFragHostRecvPoolPinned;
            else if (row_recv_bytes != 0 && lu->superlu_acc_offload &&
                     superlu_sym_v2_pinned_staging())
            {
                gpuErrchk(cudaMallocHost(
                    (void **) &lu->symV2RowFragHostRecvBufs[s],
                    row_recv_bytes));
                lu->symV2RowFragHostRecvPinned = 1;
            }
            else if (row_recv_bytes != 0)
                lu->symV2RowFragHostRecvBufs[s] =
                    (Ftype *) SUPERLU_MALLOC(row_recv_bytes);

            if (lu->symV2RowFragHostSendPoolPinned != NULL)
                lu->symV2RowFragHostSendBufs[s] =
                    lu->symV2RowFragHostSendPoolPinned;
            else if (row_send_bytes != 0 &&
                     superlu_sym_v2_row_l_separate_send_staging() &&
                     lu->superlu_acc_offload &&
                     superlu_sym_v2_pinned_staging())
            {
                gpuErrchk(cudaMallocHost(
                    (void **) &lu->symV2RowFragHostSendBufs[s],
                    row_send_bytes));
                lu->symV2RowFragHostSendPinned = 1;
            }
            else if (row_send_bytes != 0 &&
                     superlu_sym_v2_row_l_separate_send_staging())
                lu->symV2RowFragHostSendBufs[s] =
                    (Ftype *) SUPERLU_MALLOC(row_send_bytes);
        }
#endif

        if ((partner_val_bytes != 0 && lu->symPartnerLvalRecvBufs[s] == NULL) ||
            (partner_idx_bytes != 0 && lu->symPartnerLidxRecvBufs[s] == NULL))
            ABORT("Malloc fails for SymFact V2 partner receive staging.");
    }
}

template <typename Ftype>
static void symldl_v2_initialize_pcfrag_tables(xLUstruct_t<Ftype> *lu)
{
#ifdef HAVE_CUDA
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload)
        return;

    const bool pc_fragment = symldl_v2_use_pc_fragment_schur(lu->grid3d);
    int_t local_cols = lu->symV2PanelCount();
    size_t l2l_slots = symldl_v2_checked_product(
        static_cast<size_t>(local_cols), static_cast<size_t>(lu->Pc),
        "SymFact V2 partner-L table size overflows.");
    lu->symV2PartnerLSendBufsGPU.assign(l2l_slots, NULL);
    lu->symL2LSendMapsGPU.assign(l2l_slots, NULL);
    lu->symL2LSendMeta.assign(l2l_slots, std::vector<int_t>());
    lu->symV2PartnerLHostSendBufs.assign(l2l_slots, std::vector<Ftype>());
    lu->symV2PartnerLHostSendBufsPinned.assign(l2l_slots, NULL);
    lu->symV2PartnerLHostSendScratchOffsets.assign(l2l_slots, 0);
    lu->symV2PartnerLMapOffsets.assign(l2l_slots, 0);
    lu->symV2PartnerLSendSizes.assign(l2l_slots, 0);
    lu->symV2PartnerLPrepacked.assign(static_cast<size_t>(local_cols), 0);
    lu->symPanelReadyEventIds.assign(static_cast<size_t>(lu->nsupers), -1);
    lu->symV2UsePcFragmentSchur.assign(
        static_cast<size_t>(lu->nsupers),
        pc_fragment ? 1 : 0);

    size_t partner_active = symldl_v2_checked_product(
        l2l_slots, static_cast<size_t>(lu->Pr),
        "SymFact V2 partner-L active table overflows.");
    lu->symV2PartnerLSendRowActive.assign(partner_active, 0);
    if (pc_fragment)
    {
        size_t row_active = symldl_v2_checked_product(
            l2l_slots, static_cast<size_t>(lu->Pc),
            "SymFact V2 row-fragment active table overflows.");
        lu->symV2RowFragSendActive.assign(row_active, 0);
    }
    else
    {
        lu->symV2RowFragSendActive.clear();
    }

    size_t partner_recv_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 partner receive table overflows.");
    lu->symV2PartnerLRecvSizes.assign(partner_recv_slots, 0);
    lu->symV2PartnerLRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                      std::vector<int_t>());
    lu->symV2PartnerLRecvIndexBySrc.assign(partner_recv_slots,
                                           std::vector<int_t>());
    lu->symV2PartnerLRecvMap.assign(partner_recv_slots,
                                    std::vector<int_t>());
    lu->symV2PartnerLRecvMapOffsets.assign(partner_recv_slots, 0);
    lu->symV2PartnerLRecvMapsGPU.assign(partner_recv_slots, NULL);
    if (pc_fragment)
    {
        size_t row_recv_slots = symldl_v2_checked_product(
            static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pc),
            "SymFact V2 row receive table overflows.");
        lu->symV2RowFragRecvSizes.assign(row_recv_slots, 0);
        lu->symV2RowFragRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                         std::vector<int_t>());
        lu->symV2RowFragRecvMap.assign(row_recv_slots, std::vector<int_t>());
        lu->symV2RowFragRecvMapOffsets.assign(row_recv_slots, 0);
        lu->symV2RowFragRecvMapsGPU.assign(row_recv_slots, NULL);

        lu->symV2RowDownSendSizes.assign(l2l_slots, 0);
        lu->symV2RowDownSendSegOffsets.assign(l2l_slots, 0);
        lu->symV2RowDownSendSegCounts.assign(l2l_slots, 0);
        lu->symV2RowDownSendSegsGPU.assign(l2l_slots, NULL);
        lu->symV2RowDownSegOffsets.assign(l2l_slots + 1, 0);
        lu->symV2RowDownRecvSizes.assign(row_recv_slots, 0);
        lu->symV2RowDownPlanReady.assign(static_cast<size_t>(lu->nsupers), 0);
    }
    else
    {
        lu->symV2RowFragRecvSizes.clear();
        lu->symV2RowFragRecvIndex.clear();
        lu->symV2RowFragRecvMap.clear();
        lu->symV2RowFragRecvMapOffsets.clear();
        lu->symV2RowFragRecvMapsGPU.clear();

        lu->symV2RowDownSendSizes.clear();
        lu->symV2RowDownSendSegOffsets.clear();
        lu->symV2RowDownSendSegCounts.clear();
        lu->symV2RowDownSendSegsGPU.clear();
        lu->symV2RowDownSegOffsets.clear();
        lu->symV2RowDownRecvSizes.clear();
        lu->symV2RowDownPlanReady.clear();
    }
    lu->symV2RowDownSegs.clear();

    lu->symV2ExchangeSendSizesScratch.assign(static_cast<size_t>(lu->Pc), 0);
    lu->symV2ExchangeRecvSizesScratch.assign(static_cast<size_t>(lu->Pr), 0);
    lu->symV2ExchangeRecvOffsetsScratch.assign(static_cast<size_t>(lu->Pr), -1);
    lu->symV2ExchangeRecvReqsScratch.clear();
    lu->symV2ExchangeRecvReqsScratch.reserve(static_cast<size_t>(lu->Pr));
    lu->symV2ExchangeSendReqsScratch.clear();
    lu->symV2ExchangeSendReqsScratch.reserve(
        symldl_v2_checked_product(static_cast<size_t>(lu->Pr),
                                  static_cast<size_t>(lu->Pc),
                                  "SymFact V2 exchange request table overflows."));
    lu->symV2ExchangeRecvPeersScratch.clear();
    lu->symV2ExchangeRecvPeersScratch.reserve(static_cast<size_t>(lu->Pr));
    lu->symV2ExchangeWaitIndicesScratch.assign(static_cast<size_t>(lu->Pr), 0);
    lu->symV2ExchangeWaitStatusesScratch.resize(static_cast<size_t>(lu->Pr));
    if (pc_fragment)
    {
        lu->symV2RowFragSendCountsScratch.assign(static_cast<size_t>(lu->Pc), 0);
        lu->symV2RowFragSendOffsetsScratch.assign(static_cast<size_t>(lu->Pc), 0);
    }
    else
    {
        lu->symV2RowFragSendCountsScratch.clear();
        lu->symV2RowFragSendOffsetsScratch.clear();
    }
    lu->symV2RowFragSendReqsScratch.clear();
#else
    (void) lu;
#endif
}

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

template <typename Ftype>
static void symldl_v2_build_partner_l_send_maps(xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload || lu->Pr <= 1)
        return;

    int_t local_cols = lu->symV2PanelCount();
    size_t l2l_slots = symldl_v2_checked_product(
        static_cast<size_t>(local_cols), static_cast<size_t>(lu->Pc),
        "SymFact V2 partner-L map table overflows.");
    if (lu->symL2LSendMeta.size() != l2l_slots ||
        lu->symV2PartnerLMapOffsets.size() != l2l_slots ||
        lu->symV2PartnerLSendSizes.size() != l2l_slots)
        ABORT("SymFact V2 partner-L send tables are not initialized.");

    std::vector<size_t> map_counts(l2l_slots, 0);
    std::vector<size_t> meta_counts(l2l_slots, 0);
    for (int_t lk = 0; lk < local_cols; ++lk)
    {
        xlpanel_t<Ftype> &lpanel = lu->lPanelVec[lk];
        if (lpanel.isEmpty())
            continue;
        int_t jb = lu->symV2PanelGid(lk);
        int_t knsupc = lu->supersize(jb);
        int_t first_block = lpanel.haveDiag() ? 1 : 0;
            for (int_t lb = first_block; lb < lpanel.nblocks(); ++lb)
        {
            int_t ik = lpanel.gid(lb);
            int ikcol = static_cast<int>(lu->symV2PanelRoot(ik));
            if (ikcol < 0 || ikcol >= lu->Pc)
                ABORT("SymFact V2 partner-L target process column is invalid.");
            int_t len = lpanel.nbrow(lb);
            if (len <= 0)
                continue;
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(ikcol);
            size_t map_add = symldl_v2_checked_product(
                static_cast<size_t>(len), static_cast<size_t>(knsupc),
                "SymFact V2 partner-L send map overflows.");
            if (map_counts[flat] >
                std::numeric_limits<size_t>::max() - map_add)
                ABORT("SymFact V2 partner-L send map overflows.");
            map_counts[flat] += map_add;
            size_t meta_add = static_cast<size_t>(len) + 2;
            if (meta_counts[flat] >
                std::numeric_limits<size_t>::max() - meta_add)
                ABORT("SymFact V2 partner-L metadata overflows.");
            meta_counts[flat] += meta_add;
        }
    }

    size_t total_partner_send = 0;
    size_t max_panel_scratch = 0;
    for (int_t lk = 0; lk < local_cols; ++lk)
    {
        size_t panel_scratch = 0;
        for (int pc = 0; pc < lu->Pc; ++pc)
        {
            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc);
            lu->symV2PartnerLHostSendScratchOffsets[flat] = panel_scratch;
            if (panel_scratch >
                std::numeric_limits<size_t>::max() - map_counts[flat])
                ABORT("SymFact V2 partner-L send staging overflows.");
            panel_scratch += map_counts[flat];
        }
        max_panel_scratch = SUPERLU_MAX(max_panel_scratch, panel_scratch);
    }
    if (max_panel_scratch >
        static_cast<size_t>(std::numeric_limits<int_t>::max()))
        ABORT("SymFact V2 partner-L send staging exceeds int_t range.");
    lu->maxSymPartnerLSendStageCount =
        static_cast<int_t>(max_panel_scratch);

    for (size_t flat = 0; flat < l2l_slots; ++flat)
    {
        if (map_counts[flat] >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 partner-L send map exceeds MPI limit.");
        if (meta_counts[flat] >
            static_cast<size_t>(std::numeric_limits<int_t>::max()))
            ABORT("SymFact V2 partner-L metadata is too large.");
        lu->symV2PartnerLMapOffsets[flat] = total_partner_send;
        total_partner_send += map_counts[flat];
        if (total_partner_send < lu->symV2PartnerLMapOffsets[flat])
            ABORT("SymFact V2 partner-L send map overflows.");
        lu->symV2PartnerLSendSizes[flat] =
            static_cast<int>(map_counts[flat]);
        if (meta_counts[flat] > 0)
            lu->symL2LSendMeta[flat].resize(meta_counts[flat]);
        if (map_counts[flat] > 0 &&
            !superlu_sym_v2_pinned_staging())
            lu->symV2PartnerLHostSendBufs[flat].resize(map_counts[flat]);
    }
    lu->symV2PartnerLPackedMaps.assign(total_partner_send, 0);
    lu->symL2LSendMapPoolCount = total_partner_send;
    lu->symV2PartnerLSendBufPoolCount =
        superlu_sym_v2_pc_fragment_ldl_native() ? 0 : total_partner_send;
    if (superlu_sym_v2_pinned_staging() &&
        superlu_sym_v2_pinned_staging_pool() &&
        !superlu_cuda_aware_mpi() && max_panel_scratch > 0)
        lu->symV2PartnerLHostSendPoolPinnedCount = max_panel_scratch;

    std::vector<size_t> map_write_offsets = lu->symV2PartnerLMapOffsets;
    std::vector<size_t> meta_write_offsets(l2l_slots, 0);
    std::vector<std::pair<int_t, int_t> > row_order;
    for (int_t lk = 0; lk < local_cols; ++lk)
    {
        xlpanel_t<Ftype> &lpanel = lu->lPanelVec[lk];
        if (lpanel.isEmpty())
            continue;
        int_t jb = lu->symV2PanelGid(lk);
        int_t knsupc = lu->supersize(jb);
        int_t nsupr = lpanel.LDA();
        int_t first_block = lpanel.haveDiag() ? 1 : 0;
        dLocalLU_t *Llu = lu->host_Llu;
        int_t *raw_lsub = (Llu != NULL && Llu->Lrowind_bc_ptr != NULL)
                              ? Llu->Lrowind_bc_ptr[lk]
                              : NULL;
        int_t *raw_lloc = (Llu != NULL && Llu->Lindval_loc_bc_ptr != NULL)
                              ? Llu->Lindval_loc_bc_ptr[lk]
                              : NULL;
        if (raw_lsub == NULL || raw_lloc == NULL)
            ABORT("SymFact V2 partner-L send maps require local L metadata.");
        int_t raw_nb = 0;
        int_t raw_idx_i = 0;
        int_t raw_idx_v = 0;
        if (lu->myrow == lu->symV2DiagRoot(jb))
        {
            raw_nb = raw_lsub[0] - 1;
            raw_idx_i = raw_nb + 2;
            raw_idx_v = 2 * raw_nb + 3;
        }
        else
        {
            raw_nb = raw_lsub[0];
            raw_idx_i = raw_nb;
            raw_idx_v = 2 * raw_nb;
        }
        if (raw_nb < 0 || raw_lsub[1] != nsupr)
            ABORT("SymFact V2 partner-L local L metadata is inconsistent.");
        for (int_t lb = first_block; lb < lpanel.nblocks(); ++lb)
        {
            int_t ik = lpanel.gid(lb);
            int ikcol = static_cast<int>(lu->symV2PanelRoot(ik));
            int_t len = lpanel.nbrow(lb);
            if (len <= 0)
                continue;
            int_t *row_ids = lpanel.rowList(lb);
            bool rows_sorted = true;
            for (int_t i = 1; i < len; ++i)
            {
                if (row_ids[i - 1] > row_ids[i])
                {
                    rows_sorted = false;
                    break;
                }
            }
            if (!rows_sorted)
            {
                row_order.clear();
                row_order.reserve(static_cast<size_t>(len));
                for (int_t i = 0; i < len; ++i)
                    row_order.push_back(std::make_pair(row_ids[i], i));
                std::sort(row_order.begin(), row_order.end());
            }

            size_t flat = static_cast<size_t>(lk) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(ikcol);
            std::vector<int_t> &meta = lu->symL2LSendMeta[flat];
            size_t meta_pos = meta_write_offsets[flat];
            if (meta_pos + static_cast<size_t>(len) + 2 > meta.size())
                ABORT("SymFact V2 partner-L metadata overrun.");
            meta[meta_pos++] = ik;
            meta[meta_pos++] = len;
            for (int_t i = 0; i < len; ++i)
                meta[meta_pos++] =
                    rows_sorted ? row_ids[i] : row_order[i].first;
            meta_write_offsets[flat] = meta_pos;

            size_t map_pos = map_write_offsets[flat];
            size_t map_end = lu->symV2PartnerLMapOffsets[flat] +
                             map_counts[flat];
            int_t raw_lb = lb - first_block;
            if (raw_lb < 0 || raw_lb >= raw_nb)
                ABORT("SymFact V2 partner-L local L block is invalid.");
            int_t raw_lptr = raw_lloc[raw_lb + raw_idx_i];
            int_t raw_vptr = raw_lloc[raw_lb + raw_idx_v];
            if (raw_lsub[raw_lptr] != ik ||
                raw_lsub[raw_lptr + 1] != len)
                ABORT("SymFact V2 partner-L local L block metadata mismatch.");
            for (int_t col = 0; col < knsupc; ++col)
            {
                for (int_t i = 0; i < len; ++i)
                {
                    if (map_pos >= map_end)
                        ABORT("SymFact V2 partner-L send map overrun.");
                    int_t src_row = rows_sorted ? i : row_order[i].second;
                    lu->symV2PartnerLPackedMaps[map_pos++] =
                        raw_vptr + src_row + col * nsupr;
                }
            }
            map_write_offsets[flat] = map_pos;

            for (int pr = 0; pr < lu->Pr; ++pr)
            {
                size_t active_pos =
                    flat * static_cast<size_t>(lu->Pr) +
                    static_cast<size_t>(pr);
                if (active_pos >= lu->symV2PartnerLSendRowActive.size())
                    ABORT("SymFact V2 partner-L send mask is invalid.");
                lu->symV2PartnerLSendRowActive[active_pos] = 1;
            }
        }
    }

    for (size_t flat = 0; flat < l2l_slots; ++flat)
    {
        if (map_write_offsets[flat] !=
            lu->symV2PartnerLMapOffsets[flat] + map_counts[flat])
            ABORT("SymFact V2 partner-L send map size mismatch.");
        if (meta_write_offsets[flat] != meta_counts[flat])
            ABORT("SymFact V2 partner-L metadata size mismatch.");
    }
}

template <typename Ftype>
static int_t symldl_v2_cached_block_len(
    const std::vector<int_t> &payload, size_t cols_begin, int_t len,
    const std::vector<int_t> &cols)
{
    return cols.empty() ? len : static_cast<int_t>(cols.size());
}

struct SymLDLV2CachedPartnerBlock
{
    int_t gid;
    size_t cols_begin;
    int_t len;
    std::vector<int_t> cols;
};

static inline int_t symldl_v2_cached_col(
    const std::vector<int_t> &payload,
    const SymLDLV2CachedPartnerBlock &block, size_t pos)
{
    return block.cols.empty() ? payload[block.cols_begin + pos]
                              : block.cols[pos];
}

static inline int_t symldl_v2_cached_len(
    const SymLDLV2CachedPartnerBlock &block)
{
    return block.cols.empty() ? block.len
                              : static_cast<int_t>(block.cols.size());
}

template <typename Ftype>
static void symldl_v2_build_partner_l_recv_maps(xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload || lu->Pr <= 1)
        return;

    size_t table_count = symldl_v2_checked_product(
        symldl_v2_checked_product(static_cast<size_t>(lu->nsupers),
                                  static_cast<size_t>(lu->Pc),
                                  "SymFact V2 partner receive table overflows."),
        static_cast<size_t>(lu->Pr),
        "SymFact V2 partner receive table overflows.");
    if (table_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 partner receive table exceeds MPI limit.");

    std::vector<int> local_recv_sizes(table_count, 0);
    std::vector<int> global_recv_sizes(table_count, 0);
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
            int size = flat < lu->symV2PartnerLSendSizes.size()
                           ? lu->symV2PartnerLSendSizes[flat]
                           : 0;
            size_t pos = (static_cast<size_t>(k0) *
                              static_cast<size_t>(lu->Pc) +
                          static_cast<size_t>(pc)) *
                             static_cast<size_t>(lu->Pr) +
                         static_cast<size_t>(lu->myrow);
            local_recv_sizes[pos] = size;
        }
    }
    MPI_Allreduce(local_recv_sizes.data(), global_recv_sizes.data(),
                  static_cast<int>(table_count), MPI_INT, MPI_SUM,
                  lu->grid->comm);

    SymLDLV2PartnerMetaPayload meta =
        symldl_v2_collect_partner_l_metadata(lu);
    const std::vector<int_t> &all_meta_payload = meta.payload;
    const std::vector<int> &meta_counts = meta.counts;
    const std::vector<int> &meta_displs = meta.displs;
    int comm_size = meta.comm_size;

    size_t compact_count = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 partner compact table overflows.");
    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        cached_partner_blocks(static_cast<size_t>(lu->nsupers));
    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        cached_partner_recv_blocks(compact_count);
    const bool indexed_recv = superlu_sym_v2_recv_map_index();

    for (int r = 0; r < comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(meta_displs[r]);
        size_t rank_end = meta_pos + static_cast<size_t>(meta_counts[r]);
        int source_pr = MYROW(r, lu->grid);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 partner metadata payload is truncated.");
            int_t target_pc = all_meta_payload[meta_pos++];
            int_t k0 = all_meta_payload[meta_pos++];
            int_t meta_len = all_meta_payload[meta_pos++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k0 < 0 || k0 >= lu->nsupers || meta_len < 0 ||
                meta_pos + static_cast<size_t>(meta_len) > rank_end)
                ABORT("SymFact V2 partner metadata payload is invalid.");

            size_t block_pos = meta_pos;
            size_t block_end = meta_pos + static_cast<size_t>(meta_len);
            if (target_pc == lu->mycol)
            {
                while (block_pos < block_end)
                {
                    if (block_pos + 2 > block_end)
                        ABORT("SymFact V2 partner metadata block is truncated.");
                    SymLDLV2CachedPartnerBlock block;
                    block.gid = all_meta_payload[block_pos++];
                    block.len = all_meta_payload[block_pos++];
                    if (block.len < 0 ||
                        block_pos + static_cast<size_t>(block.len) >
                            block_end)
                        ABORT("SymFact V2 partner metadata block is invalid.");
                    block.cols_begin = block_pos;
                    if (!indexed_recv)
                        block.cols.assign(
                            all_meta_payload.begin() + block_pos,
                            all_meta_payload.begin() + block_pos + block.len);
                    block_pos += static_cast<size_t>(block.len);
                    cached_partner_blocks[static_cast<size_t>(k0)]
                        .push_back(block);
                    size_t recv_pos = static_cast<size_t>(k0) *
                                          static_cast<size_t>(lu->Pr) +
                                      static_cast<size_t>(source_pr);
                    cached_partner_recv_blocks[recv_pos].push_back(block);
                }
            }
            meta_pos = block_end;
        }
    }

    lu->symV2PartnerLRecvSizes.assign(compact_count, 0);
    lu->symV2PartnerLRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                      std::vector<int_t>());
    lu->symV2PartnerLRecvIndexBySrc.assign(compact_count,
                                           std::vector<int_t>());
    lu->symV2PartnerLRecvMap.assign(compact_count, std::vector<int_t>());
    lu->symV2PartnerLRecvMapOffsets.assign(compact_count, 0);
    lu->symV2PartnerLRecvMapsGPU.assign(compact_count, NULL);

    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            size_t src_pos = (static_cast<size_t>(k0) *
                                  static_cast<size_t>(lu->Pc) +
                              static_cast<size_t>(lu->mycol)) *
                                 static_cast<size_t>(lu->Pr) +
                             static_cast<size_t>(pr);
            size_t dst_pos = static_cast<size_t>(k0) *
                                 static_cast<size_t>(lu->Pr) +
                             static_cast<size_t>(pr);
            lu->symV2PartnerLRecvSizes[dst_pos] =
                global_recv_sizes[src_pos];
        }

        std::vector<SymLDLV2CachedPartnerBlock> &blocks =
            cached_partner_blocks[static_cast<size_t>(k0)];
        if (blocks.empty())
            continue;
        std::sort(blocks.begin(), blocks.end(),
                  [](const SymLDLV2CachedPartnerBlock &a,
                     const SymLDLV2CachedPartnerBlock &b)
                  {
                      return a.gid < b.gid;
                  });

        int_t nblocks = static_cast<int_t>(blocks.size());
        int_t nrows = 0;
        for (size_t ib = 0; ib < blocks.size(); ++ib)
            nrows += symldl_v2_cached_len(blocks[ib]);
        int_t index_size = LPANEL_HEADER_SIZE + 2 * nblocks + 1 + nrows;
        if (index_size > lu->maxSymPartnerLidxCount)
            ABORT("SymFact V2 partner fragment index exceeds staging buffer.");
        if (static_cast<int64_t>(nrows) *
                static_cast<int64_t>(lu->supersize(k0)) >
            static_cast<int64_t>(lu->maxSymPartnerLvalCount))
            ABORT("SymFact V2 partner fragment values exceed staging buffer.");

        std::vector<int_t> &index =
            lu->symV2PartnerLRecvIndex[static_cast<size_t>(k0)];
        index.assign(static_cast<size_t>(index_size), 0);
        index[0] = nblocks;
        index[1] = nrows;
        index[2] = 0;
        index[3] = lu->supersize(k0);
        int_t gid_ptr = LPANEL_HEADER_SIZE;
        int_t px_ptr = LPANEL_HEADER_SIZE + nblocks;
        int_t row_ptr = LPANEL_HEADER_SIZE + 2 * nblocks + 1;
        index[px_ptr] = 0;
        for (int_t ib = 0; ib < nblocks; ++ib)
        {
            index[gid_ptr + ib] = blocks[ib].gid;
            index[px_ptr + ib + 1] =
                index[px_ptr + ib] + symldl_v2_cached_len(blocks[ib]);
            for (int_t j = 0; j < symldl_v2_cached_len(blocks[ib]); ++j)
                index[row_ptr++] = symldl_v2_cached_col(
                    all_meta_payload, blocks[ib], static_cast<size_t>(j));
        }

        std::vector<std::pair<int_t, int_t> > lookup;
        lookup.reserve(static_cast<size_t>(nblocks));
        for (int_t ib = 0; ib < nblocks; ++ib)
        {
            if (!lookup.empty() && lookup.back().first == blocks[ib].gid)
                continue;
            lookup.push_back(std::make_pair(blocks[ib].gid,
                                            index[px_ptr + ib]));
        }
        auto block_offset = [&](int_t gid) -> int_t
        {
            std::vector<std::pair<int_t, int_t> >::const_iterator it =
                std::lower_bound(
                    lookup.begin(), lookup.end(),
                    std::make_pair(gid, static_cast<int_t>(0)),
                    [](const std::pair<int_t, int_t> &a,
                       const std::pair<int_t, int_t> &b)
                    {
                        return a.first < b.first;
                    });
            return (it != lookup.end() && it->first == gid)
                       ? it->second
                       : GLOBAL_BLOCK_NOT_FOUND;
        };

        for (int pr = 0; pr < lu->Pr; ++pr)
        {
            size_t recv_pos = static_cast<size_t>(k0) *
                                  static_cast<size_t>(lu->Pr) +
                              static_cast<size_t>(pr);
            std::vector<SymLDLV2CachedPartnerBlock> &recv_blocks =
                cached_partner_recv_blocks[recv_pos];
            std::vector<int_t> &recv_map =
                lu->symV2PartnerLRecvMap[recv_pos];
            std::vector<int_t> &src_index =
                lu->symV2PartnerLRecvIndexBySrc[recv_pos];
            if (!recv_blocks.empty())
            {
                int_t src_nblocks =
                    static_cast<int_t>(recv_blocks.size());
                int_t src_nrows = 0;
                for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
                    src_nrows += symldl_v2_cached_len(recv_blocks[rb]);
                int_t src_index_size =
                    LPANEL_HEADER_SIZE + 2 * src_nblocks + 1 + src_nrows;
                if (src_index_size > lu->maxSymPartnerLidxCount)
                    ABORT("SymFact V2 partner source index exceeds staging buffer.");
                src_index.assign(static_cast<size_t>(src_index_size), 0);
                src_index[0] = src_nblocks;
                src_index[1] = src_nrows;
                src_index[2] = 0;
                src_index[3] = lu->supersize(k0);
                int_t src_gid_ptr = LPANEL_HEADER_SIZE;
                int_t src_px_ptr = LPANEL_HEADER_SIZE + src_nblocks;
                int_t src_row_ptr =
                    LPANEL_HEADER_SIZE + 2 * src_nblocks + 1;
                src_index[src_px_ptr] = 0;
                for (int_t rb = 0; rb < src_nblocks; ++rb)
                {
                    src_index[src_gid_ptr + rb] =
                        recv_blocks[static_cast<size_t>(rb)].gid;
                    src_index[src_px_ptr + rb + 1] =
                        src_index[src_px_ptr + rb] +
                        symldl_v2_cached_len(
                            recv_blocks[static_cast<size_t>(rb)]);
                    for (int_t rc = 0;
                         rc < symldl_v2_cached_len(
                                  recv_blocks[static_cast<size_t>(rb)]);
                         ++rc)
                        src_index[src_row_ptr++] =
                            symldl_v2_cached_col(
                                all_meta_payload,
                                recv_blocks[static_cast<size_t>(rb)],
                                static_cast<size_t>(rc));
                }
            }

            long long expected_values = 0;
            int_t src_offset = 0;
            for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
            {
                int_t recv_offset = block_offset(recv_blocks[rb].gid);
                if (recv_offset == GLOBAL_BLOCK_NOT_FOUND)
                    ABORT("SymFact V2 partner receive map cannot find a block.");
                int_t rows = symldl_v2_cached_len(recv_blocks[rb]);
                recv_map.push_back(recv_offset);
                recv_map.push_back(rows);
                recv_map.push_back(src_offset);
                src_offset += rows * lu->supersize(k0);
                expected_values += static_cast<long long>(rows) *
                                   static_cast<long long>(lu->supersize(k0));
            }
            if (expected_values !=
                static_cast<long long>(lu->symV2PartnerLRecvSizes[recv_pos]))
                ABORT("SymFact V2 partner receive map size mismatch.");
        }
    }

    size_t total_recv_map = 0;
    for (size_t pos = 0; pos < lu->symV2PartnerLRecvMap.size(); ++pos)
    {
        lu->symV2PartnerLRecvMapOffsets[pos] = total_recv_map;
        size_t map_size = lu->symV2PartnerLRecvMap[pos].size();
        if (total_recv_map >
            std::numeric_limits<size_t>::max() - map_size)
            ABORT("SymFact V2 partner receive map size overflows.");
        total_recv_map += map_size;
    }
    lu->symV2PartnerLRecvMapPoolCount = total_recv_map;
}

struct SymLDLV2RowDownDirectRange
{
    int_t start;
    int_t len;
    int chunk_pc;
};

template <typename Ftype>
static void symldl_v2_append_row_down_demand_record(
    xLUstruct_t<Ftype> *lu, std::vector<int_t> &payload, int_t panel,
    int dest_pc, int chunk_pc, const std::vector<int_t> &blocks)
{
    if (blocks.empty())
        return;
    if (blocks.size() >
        static_cast<size_t>(std::numeric_limits<int_t>::max()))
        ABORT("SymFact V2 row-down demand record is too large.");

    payload.push_back(panel);
    payload.push_back(dest_pc);
    payload.push_back(chunk_pc);

    if (superlu_sym_v2_row_l_compressed_plan())
    {
        std::vector<int_t> ranges;
        ranges.reserve(blocks.size());
        int_t start = blocks[0];
        int_t prev = blocks[0];
        if (start < 0)
            ABORT("SymFact V2 row-down compressed demand has invalid block id.");
        for (size_t bi = 1; bi < blocks.size(); ++bi)
        {
            int_t gid = blocks[bi];
            if (gid <= prev)
                ABORT("SymFact V2 row-down compressed demand is not sorted unique.");
            if (prev < std::numeric_limits<int_t>::max() &&
                gid == prev + 1)
            {
                prev = gid;
                continue;
            }
            ranges.push_back(start);
            ranges.push_back(prev - start + 1);
            start = gid;
            prev = gid;
        }
        ranges.push_back(start);
        ranges.push_back(prev - start + 1);

        if (ranges.size() < blocks.size())
        {
            size_t nranges = ranges.size() / 2;
            if (nranges >
                static_cast<size_t>(std::numeric_limits<int_t>::max()))
                ABORT("SymFact V2 row-down compressed demand has too many ranges.");
            payload.push_back(-static_cast<int_t>(nranges));
            payload.insert(payload.end(), ranges.begin(), ranges.end());
            return;
        }
    }

    payload.push_back(static_cast<int_t>(blocks.size()));
    payload.insert(payload.end(), blocks.begin(), blocks.end());
    (void) lu;
}

template <typename Ftype>
static void symldl_v2_build_row_down_slot_segments(
    xLUstruct_t<Ftype> *lu, size_t slot,
    std::vector<SymLDLV2RowDownDirectRange> &ranges,
    std::vector<SymV2RowDownSendSegmentGPU> &out,
    int &value_count_out)
{
    value_count_out = 0;
    if (ranges.empty())
        return;

    int_t lk = static_cast<int_t>(slot / static_cast<size_t>(lu->Pc));
    int_t k0 = lu->symV2PanelGid(lk);
    if (k0 < 0 || k0 >= lu->nsupers)
        ABORT("SymFact V2 row-down send panel is invalid.");
    int_t ksupc = lu->supersize(k0);
    if (ksupc <= 0)
        ABORT("SymFact V2 row-down send panel width is invalid.");

    std::sort(ranges.begin(), ranges.end(),
              [](const SymLDLV2RowDownDirectRange &a,
                 const SymLDLV2RowDownDirectRange &b)
              {
                  if (a.start != b.start)
                      return a.start < b.start;
                  return a.chunk_pc < b.chunk_pc;
              });

    std::vector<SymLDLV2RowDownDirectRange> merged_ranges;
    merged_ranges.reserve(ranges.size());
    size_t requested_blocks = 0;
    for (size_t ri = 0; ri < ranges.size(); ++ri)
    {
        SymLDLV2RowDownDirectRange cur = ranges[ri];
        if (cur.chunk_pc < 0 || cur.chunk_pc >= lu->Pc ||
            cur.start < 0 || cur.len <= 0 ||
            cur.start > std::numeric_limits<int_t>::max() - cur.len)
            ABORT("SymFact V2 row-down request is invalid.");
        int_t cur_end = cur.start + cur.len;
        if (!merged_ranges.empty())
        {
            SymLDLV2RowDownDirectRange &prev = merged_ranges.back();
            int_t prev_end = prev.start + prev.len;
            if (cur.start < prev_end && cur.chunk_pc != prev.chunk_pc)
                ABORT("SymFact V2 row-down request overlaps across chunks.");
            if (cur.chunk_pc == prev.chunk_pc && cur.start <= prev_end)
            {
                if (cur_end > prev_end)
                    prev.len = cur_end - prev.start;
                continue;
            }
        }
        merged_ranges.push_back(cur);
    }
    ranges.swap(merged_ranges);

    for (size_t ri = 0; ri < ranges.size(); ++ri)
    {
        if (static_cast<size_t>(ranges[ri].len) >
            std::numeric_limits<size_t>::max() - requested_blocks)
            ABORT("SymFact V2 row-down request count overflows.");
        requested_blocks += static_cast<size_t>(ranges[ri].len);
    }
    if (requested_blocks == 0)
        return;

    std::vector<SymLDLV2RowDownDirectRange> scan_ranges = ranges;
    std::sort(scan_ranges.begin(), scan_ranges.end(),
              [](const SymLDLV2RowDownDirectRange &a,
                 const SymLDLV2RowDownDirectRange &b)
              {
                  if (a.chunk_pc != b.chunk_pc)
                      return a.chunk_pc < b.chunk_pc;
                  return a.start < b.start;
              });

    struct SourceSegment
    {
        int_t gid;
        int_t nrows;
        size_t map_offset;
    };
    std::vector<SourceSegment> segments;
    segments.reserve(requested_blocks);

    size_t group_begin = 0;
    while (group_begin < scan_ranges.size())
    {
        int chunk_pc = scan_ranges[group_begin].chunk_pc;
        size_t group_end = group_begin + 1;
        while (group_end < scan_ranges.size() &&
               scan_ranges[group_end].chunk_pc == chunk_pc)
            ++group_end;

        size_t flat = static_cast<size_t>(lk) *
                          static_cast<size_t>(lu->Pc) +
                      static_cast<size_t>(chunk_pc);
        if (flat >= lu->symL2LSendMeta.size() ||
            flat >= lu->symV2PartnerLSendSizes.size() ||
            flat >= lu->symV2PartnerLMapOffsets.size())
            ABORT("SymFact V2 row-down source map is invalid.");
        if (lu->symV2PartnerLSendSizes[flat] < 0)
            ABORT("SymFact V2 row-down source map size is invalid.");
        const std::vector<int_t> &meta = lu->symL2LSendMeta[flat];
        size_t map_pos = lu->symV2PartnerLMapOffsets[flat];
        size_t map_end =
            map_pos + static_cast<size_t>(lu->symV2PartnerLSendSizes[flat]);
        if (map_end > lu->symV2PartnerLPackedMaps.size() ||
            map_end < map_pos)
            ABORT("SymFact V2 row-down source map bounds are invalid.");

        auto range_group_contains_gid = [&](int_t gid) -> bool
        {
            size_t lo = group_begin;
            size_t hi = group_end;
            while (lo < hi)
            {
                size_t mid = lo + (hi - lo) / 2;
                if (scan_ranges[mid].start <= gid)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo == group_begin)
                return false;
            const SymLDLV2RowDownDirectRange &r = scan_ranges[lo - 1];
            return gid >= r.start && gid < r.start + r.len;
        };

        size_t meta_pos = 0;
        while (meta_pos < meta.size())
        {
            if (meta_pos + 2 > meta.size())
                ABORT("SymFact V2 row-down metadata is truncated.");
            int_t block_gid = meta[meta_pos++];
            int_t len = meta[meta_pos++];
            if (len < 0 || meta_pos + static_cast<size_t>(len) > meta.size())
                ABORT("SymFact V2 row-down metadata block is invalid.");
            size_t value_count = symldl_v2_checked_product(
                static_cast<size_t>(len), static_cast<size_t>(ksupc),
                "SymFact V2 row-down source map segment overflows.");
            if (map_pos + value_count > map_end || map_pos + value_count < map_pos)
                ABORT("SymFact V2 row-down source map segment is invalid.");
            if (range_group_contains_gid(block_gid))
            {
                SourceSegment seg;
                seg.gid = block_gid;
                seg.nrows = len;
                seg.map_offset = map_pos;
                segments.push_back(seg);
            }
            map_pos += value_count;
            meta_pos += static_cast<size_t>(len);
        }
        if (map_pos != map_end)
            ABORT("SymFact V2 row-down source map size mismatch.");
        group_begin = group_end;
    }

    if (segments.size() != requested_blocks)
        ABORT("SymFact V2 row-down did not find all requested blocks.");
    std::sort(segments.begin(), segments.end(),
              [](const SourceSegment &a, const SourceSegment &b)
              {
                  return a.gid < b.gid;
              });
    for (size_t si = 1; si < segments.size(); ++si)
        if (segments[si - 1].gid == segments[si].gid)
            ABORT("SymFact V2 row-down source block is duplicated.");

    size_t row_count = 0;
    out.reserve(out.size() + segments.size());
    for (size_t si = 0; si < segments.size(); ++si)
    {
        if (segments[si].nrows <= 0 ||
            static_cast<size_t>(segments[si].nrows) >
                std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(ksupc))
            ABORT("SymFact V2 row-down segment has invalid width.");
        size_t block_values =
            static_cast<size_t>(segments[si].nrows) *
            static_cast<size_t>(ksupc);
        if (segments[si].map_offset + block_values >
                lu->symV2PartnerLPackedMaps.size() ||
            segments[si].map_offset + block_values < segments[si].map_offset)
            ABORT("SymFact V2 row-down segment map is invalid.");
        if (row_count >
            std::numeric_limits<size_t>::max() -
                static_cast<size_t>(segments[si].nrows))
            ABORT("SymFact V2 row-down row count overflows.");

        SymV2RowDownSendSegmentGPU gpu_seg;
        gpu_seg.map_offset = segments[si].map_offset;
        gpu_seg.nrows = segments[si].nrows;
        if (row_count >
            static_cast<size_t>(std::numeric_limits<int_t>::max()))
            ABORT("SymFact V2 row-down destination offset is too large.");
        gpu_seg.dst_row_offset = static_cast<int_t>(row_count);
        out.push_back(gpu_seg);
        row_count += static_cast<size_t>(segments[si].nrows);
    }

    size_t value_count = symldl_v2_checked_product(
        row_count, static_cast<size_t>(ksupc),
        "SymFact V2 row-down value count overflows.");
    if (value_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 row-down value count exceeds MPI limit.");
    value_count_out = static_cast<int>(value_count);
}

template <typename Ftype>
static void symldl_v2_build_row_down_maps(xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload ||
        lu->Pr <= 1 || lu->Pc <= 1 ||
        !superlu_sym_v2_pc_fragment_schur())
        return;

    if (!superlu_sym_v2_pc_fragment_ldl_native() ||
        !superlu_sym_v2_row_l_plan_v2() ||
        !superlu_sym_v2_row_l_plan_v2_exchange() ||
        !superlu_sym_v2_row_l_direct_recv() ||
        !superlu_sym_v2_row_l_compressed_plan() ||
        !superlu_sym_v2_row_l_lazy_sendmap() ||
        !superlu_sym_v2_row_l_pack_all_dest() ||
        !superlu_sym_v2_row_l_separate_send_staging())
        ABORT("SymFact V2 Pc-fragment row-down exchange requires the LDL-native lazy row-down path.");
    if (superlu_sym_v2_row_l_plan_v2_dryrun())
        ABORT("SymFact V2 row-down exchange requires the active row-down plan.");

    int row_comm_size = 0;
    int row_comm_rank = -1;
    MPI_Comm_size(lu->grid3d->rscp.comm, &row_comm_size);
    MPI_Comm_rank(lu->grid3d->rscp.comm, &row_comm_rank);
    if (row_comm_size != lu->Pc || row_comm_rank != lu->mycol)
        ABORT("SymFact V2 row-down planner found an unexpected row communicator.");

    size_t row_chunk_count = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pc),
        "SymFact V2 row-down table size overflows.");
    size_t l2l_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->symV2PanelCount()),
        static_cast<size_t>(lu->Pc),
        "SymFact V2 row-down send table size overflows.");

    SymLDLV2PartnerMetaPayload meta =
        symldl_v2_collect_partner_l_metadata(lu);
    const std::vector<int_t> &all_meta_payload = meta.payload;

    std::vector<std::vector<int_t> >
        row_blocks_by_panel(static_cast<size_t>(lu->nsupers));
    for (int r = 0; r < meta.comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(meta.displs[r]);
        size_t rank_end = meta_pos + static_cast<size_t>(meta.counts[r]);
        int source_pr = MYROW(r, lu->grid);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 row-down metadata payload is truncated.");
            int_t target_pc = all_meta_payload[meta_pos++];
            int_t k0 = all_meta_payload[meta_pos++];
            int_t meta_len = all_meta_payload[meta_pos++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k0 < 0 || k0 >= lu->nsupers || meta_len < 0 ||
                meta_pos + static_cast<size_t>(meta_len) > rank_end)
                ABORT("SymFact V2 row-down metadata payload is invalid.");
            size_t block_pos = meta_pos;
            size_t block_end = meta_pos + static_cast<size_t>(meta_len);
            if (source_pr == lu->myrow)
            {
                std::vector<int_t> &row_blocks =
                    row_blocks_by_panel[static_cast<size_t>(k0)];
                while (block_pos < block_end)
                {
                    if (block_pos + 2 > block_end)
                        ABORT("SymFact V2 row-down metadata block is truncated.");
                    int_t gid = all_meta_payload[block_pos++];
                    int_t len = all_meta_payload[block_pos++];
                    if (len < 0 ||
                        block_pos + static_cast<size_t>(len) > block_end)
                        ABORT("SymFact V2 row-down metadata block has invalid length.");
                    row_blocks.push_back(gid);
                    block_pos += static_cast<size_t>(len);
                }
            }
            meta_pos += static_cast<size_t>(meta_len);
        }
    }

    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        std::vector<int_t> &row_blocks =
            row_blocks_by_panel[static_cast<size_t>(k0)];
        if (row_blocks.empty())
            continue;
        std::sort(row_blocks.begin(), row_blocks.end());
        row_blocks.erase(std::unique(row_blocks.begin(), row_blocks.end()),
                         row_blocks.end());
    }

    std::vector<std::vector<int_t> >
        needed_row_blocks_by_panel(static_cast<size_t>(lu->nsupers));
    for (int r = 0; r < meta.comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(meta.displs[r]);
        size_t rank_end = meta_pos + static_cast<size_t>(meta.counts[r]);
        int source_pr = MYROW(r, lu->grid);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 row-down metadata payload is truncated.");
            int_t target_pc = all_meta_payload[meta_pos++];
            int_t k0 = all_meta_payload[meta_pos++];
            int_t meta_len = all_meta_payload[meta_pos++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k0 < 0 || k0 >= lu->nsupers || meta_len < 0 ||
                meta_pos + static_cast<size_t>(meta_len) > rank_end)
                ABORT("SymFact V2 row-down metadata payload is invalid.");

            size_t block_pos = meta_pos;
            size_t block_end = meta_pos + static_cast<size_t>(meta_len);
            if (target_pc == lu->mycol)
            {
                const std::vector<int_t> &row_blocks =
                    row_blocks_by_panel[static_cast<size_t>(k0)];
                while (block_pos < block_end)
                {
                    if (block_pos + 2 > block_end)
                        ABORT("SymFact V2 row-down metadata block is truncated.");
                    int_t gj = all_meta_payload[block_pos++];
                    int_t len = all_meta_payload[block_pos++];
                    if (len < 0 ||
                        block_pos + static_cast<size_t>(len) > block_end)
                        ABORT("SymFact V2 row-down metadata block has invalid length.");
                    if (!row_blocks.empty() &&
                        lu->symV2PanelRoot(gj) == lu->mycol)
                    {
                        int_t lj = lu->symV2PanelIndex(gj);
                        if (lj >= 0)
                        {
                            xlpanel_t<Ftype> &dst_panel = lu->lPanelVec[lj];
                            if (!dst_panel.isEmpty())
                            {
                                std::vector<int_t>::const_iterator it =
                                    std::lower_bound(row_blocks.begin(),
                                                     row_blocks.end(), gj);
                                for (; it != row_blocks.end(); ++it)
                                {
                                    if (dst_panel.find(*it) !=
                                        GLOBAL_BLOCK_NOT_FOUND)
                                        needed_row_blocks_by_panel[
                                            static_cast<size_t>(k0)]
                                            .push_back(*it);
                                }
                            }
                        }
                    }
                    block_pos += static_cast<size_t>(len);
                }
            }
            meta_pos = block_end;
        }
    }

    std::vector<std::vector<int_t> > requested_by_chunk(row_chunk_count);
    std::vector<std::vector<int_t> >
        row_down_payloads(static_cast<size_t>(lu->Pc));
    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        std::vector<int_t> &needed =
            needed_row_blocks_by_panel[static_cast<size_t>(k0)];
        if (needed.empty())
            continue;
        std::sort(needed.begin(), needed.end());
        needed.erase(std::unique(needed.begin(), needed.end()),
                     needed.end());
        int source_pc = lu->symV2PanelRoot(k0);
        if (source_pc < 0 || source_pc >= lu->Pc)
            ABORT("SymFact V2 row-down source process column is invalid.");
        for (size_t bi = 0; bi < needed.size(); ++bi)
        {
            int_t gid = needed[bi];
            int chunk_pc = lu->symV2PanelRoot(gid);
            if (chunk_pc < 0 || chunk_pc >= lu->Pc)
                ABORT("SymFact V2 row-down chunk process column is invalid.");
            requested_by_chunk[static_cast<size_t>(k0) *
                                   static_cast<size_t>(lu->Pc) +
                               static_cast<size_t>(chunk_pc)]
                .push_back(gid);
        }
        for (int chunk_pc = 0; chunk_pc < lu->Pc; ++chunk_pc)
        {
            std::vector<int_t> &blocks =
                requested_by_chunk[static_cast<size_t>(k0) *
                                       static_cast<size_t>(lu->Pc) +
                                   static_cast<size_t>(chunk_pc)];
            if (blocks.empty())
                continue;
            symldl_v2_append_row_down_demand_record(
                lu, row_down_payloads[static_cast<size_t>(source_pc)],
                k0, lu->mycol, chunk_pc, blocks);
        }
    }

    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        row_down_blocks_by_panel(static_cast<size_t>(lu->nsupers));
    std::vector<std::vector<SymLDLV2CachedPartnerBlock> >
        row_down_recv_blocks(row_chunk_count);
    for (int r = 0; r < meta.comm_size; ++r)
    {
        size_t meta_pos = static_cast<size_t>(meta.displs[r]);
        size_t rank_end = meta_pos + static_cast<size_t>(meta.counts[r]);
        int source_pr = MYROW(r, lu->grid);
        while (meta_pos < rank_end)
        {
            if (meta_pos + 3 > rank_end)
                ABORT("SymFact V2 row-down metadata payload is truncated.");
            int_t target_pc = all_meta_payload[meta_pos++];
            int_t k0 = all_meta_payload[meta_pos++];
            int_t meta_len = all_meta_payload[meta_pos++];
            if (target_pc < 0 || target_pc >= lu->Pc ||
                k0 < 0 || k0 >= lu->nsupers || meta_len < 0 ||
                meta_pos + static_cast<size_t>(meta_len) > rank_end)
                ABORT("SymFact V2 row-down metadata payload is invalid.");
            size_t block_pos = meta_pos;
            size_t block_end = meta_pos + static_cast<size_t>(meta_len);
            if (source_pr == lu->myrow)
            {
                size_t req_pos = static_cast<size_t>(k0) *
                                     static_cast<size_t>(lu->Pc) +
                                 static_cast<size_t>(target_pc);
                const std::vector<int_t> &requested =
                    requested_by_chunk[req_pos];
                if (!requested.empty())
                {
                    while (block_pos < block_end)
                    {
                        if (block_pos + 2 > block_end)
                            ABORT("SymFact V2 row-down metadata block is truncated.");
                        SymLDLV2CachedPartnerBlock block;
                        block.gid = all_meta_payload[block_pos++];
                        block.len = all_meta_payload[block_pos++];
                        if (block.len < 0 ||
                            block_pos + static_cast<size_t>(block.len) >
                                block_end)
                            ABORT("SymFact V2 row-down metadata block has invalid length.");
                        if (std::binary_search(requested.begin(),
                                               requested.end(),
                                               block.gid))
                        {
                            block.cols_begin = block_pos;
                            block.cols.assign(
                                all_meta_payload.begin() + block_pos,
                                all_meta_payload.begin() + block_pos +
                                    block.len);
                            row_down_blocks_by_panel[static_cast<size_t>(k0)]
                                .push_back(block);
                            row_down_recv_blocks[req_pos].push_back(block);
                        }
                        block_pos += static_cast<size_t>(block.len);
                    }
                }
            }
            meta_pos = block_end;
        }
    }

    lu->symV2RowFragRecvSizes.assign(row_chunk_count, 0);
    lu->symV2RowFragRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                     std::vector<int_t>());
    lu->symV2RowFragRecvMap.assign(row_chunk_count, std::vector<int_t>());
    lu->symV2RowFragRecvMapOffsets.assign(row_chunk_count, 0);
    lu->symV2RowFragRecvMapsGPU.assign(row_chunk_count, NULL);
    lu->symV2RowDownRecvSizes.assign(row_chunk_count, 0);
    lu->symV2RowDownSegs.clear();
    lu->symV2RowDownSegOffsets.assign(l2l_slots + 1, 0);

    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        std::vector<SymLDLV2CachedPartnerBlock> &blocks =
            row_down_blocks_by_panel[static_cast<size_t>(k0)];
        if (blocks.empty())
            continue;
        std::sort(blocks.begin(), blocks.end(),
                  [](const SymLDLV2CachedPartnerBlock &a,
                     const SymLDLV2CachedPartnerBlock &b)
                  {
                      return a.gid < b.gid;
                  });
        std::vector<SymLDLV2CachedPartnerBlock> unique_blocks;
        unique_blocks.reserve(blocks.size());
        for (size_t bi = 0; bi < blocks.size(); ++bi)
        {
            if (!unique_blocks.empty() &&
                unique_blocks.back().gid == blocks[bi].gid)
                continue;
            unique_blocks.push_back(blocks[bi]);
        }
        blocks.swap(unique_blocks);

        int_t row_nblocks = static_cast<int_t>(blocks.size());
        int_t row_nrows = 0;
        for (size_t bi = 0; bi < blocks.size(); ++bi)
            row_nrows += static_cast<int_t>(blocks[bi].cols.size());
        int_t row_index_size =
            LPANEL_HEADER_SIZE + 2 * row_nblocks + 1 + row_nrows;
        if (row_index_size > lu->maxSymV2RowFragIdxRecvCount)
            ABORT("SymFact V2 row-down index exceeds receive buffer.");
        if (static_cast<int64_t>(row_nrows) *
                static_cast<int64_t>(lu->supersize(k0)) >
            static_cast<int64_t>(lu->maxSymV2RowFragValRecvCount))
            ABORT("SymFact V2 row-down values exceed receive buffer.");

        std::vector<int_t> &row_index =
            lu->symV2RowFragRecvIndex[static_cast<size_t>(k0)];
        row_index.assign(static_cast<size_t>(row_index_size), 0);
        row_index[0] = row_nblocks;
        row_index[1] = row_nrows;
        row_index[2] = 0;
        row_index[3] = lu->supersize(k0);
        int_t row_gid_ptr = LPANEL_HEADER_SIZE;
        int_t row_px_ptr = LPANEL_HEADER_SIZE + row_nblocks;
        int_t row_data_ptr = LPANEL_HEADER_SIZE + 2 * row_nblocks + 1;
        row_index[row_px_ptr] = 0;
        for (int_t bi = 0; bi < row_nblocks; ++bi)
        {
            row_index[row_gid_ptr + bi] = blocks[static_cast<size_t>(bi)].gid;
            row_index[row_px_ptr + bi + 1] =
                row_index[row_px_ptr + bi] +
                static_cast<int_t>(blocks[static_cast<size_t>(bi)].cols.size());
            for (size_t rc = 0;
                 rc < blocks[static_cast<size_t>(bi)].cols.size(); ++rc)
                row_index[row_data_ptr++] =
                    blocks[static_cast<size_t>(bi)].cols[rc];
        }

        for (int chunk_pc = 0; chunk_pc < lu->Pc; ++chunk_pc)
        {
            size_t recv_pos = static_cast<size_t>(k0) *
                                  static_cast<size_t>(lu->Pc) +
                              static_cast<size_t>(chunk_pc);
            std::vector<SymLDLV2CachedPartnerBlock> &recv_blocks =
                row_down_recv_blocks[recv_pos];
            std::vector<int_t> &recv_map =
                lu->symV2RowFragRecvMap[recv_pos];
            long long expected_values = 0;
            int_t src_offset = 0;
            for (size_t rb = 0; rb < recv_blocks.size(); ++rb)
            {
                std::vector<SymLDLV2CachedPartnerBlock>::const_iterator it =
                    std::lower_bound(
                        blocks.begin(), blocks.end(), recv_blocks[rb].gid,
                        [](const SymLDLV2CachedPartnerBlock &block,
                           int_t gid)
                        {
                            return block.gid < gid;
                        });
                if (it == blocks.end() || it->gid != recv_blocks[rb].gid)
                    ABORT("SymFact V2 row-down receive map cannot find a block.");
                int_t ib = static_cast<int_t>(it - blocks.begin());
                int_t nrows =
                    static_cast<int_t>(recv_blocks[rb].cols.size());
                recv_map.push_back(row_index[row_px_ptr + ib]);
                recv_map.push_back(nrows);
                recv_map.push_back(src_offset);
                SymV2RowDownSeg seg;
                seg.gid = recv_blocks[rb].gid;
                seg.chunk_pc = chunk_pc;
                seg.nrows = nrows;
                seg.dst_row_offset = row_index[row_px_ptr + ib];
                seg.value_count = nrows * lu->supersize(k0);
                seg.map_offset = static_cast<size_t>(src_offset);
                lu->symV2RowDownSegs.push_back(seg);
                src_offset += nrows * lu->supersize(k0);
                expected_values += static_cast<long long>(nrows) *
                                   static_cast<long long>(lu->supersize(k0));
            }
            if (expected_values >
                static_cast<long long>(std::numeric_limits<int>::max()))
                ABORT("SymFact V2 row-down receive size exceeds MPI limit.");
            lu->symV2RowFragRecvSizes[recv_pos] =
                static_cast<int>(expected_values);
            lu->symV2RowDownRecvSizes[recv_pos] =
                static_cast<int>(expected_values);
        }
    }

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
        size_t pos = static_cast<size_t>(recv_displs[static_cast<size_t>(sender_pc)]);
        size_t end = pos + static_cast<size_t>(recv_counts[static_cast<size_t>(sender_pc)]);
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

    std::vector<std::vector<SymV2RowDownSendSegmentGPU> >
        slot_send_segments(l2l_slots);
    std::vector<int> slot_send_segment_values(l2l_slots, 0);
    for (size_t slot = 0; slot < slot_direct_ranges.size(); ++slot)
        symldl_v2_build_row_down_slot_segments(
            lu, slot, slot_direct_ranges[slot],
            slot_send_segments[slot], slot_send_segment_values[slot]);

    lu->symV2RowDownSendSegsHost.clear();
    std::vector<std::vector<int_t> >
        size_reply_payloads(static_cast<size_t>(lu->Pc));
    for (size_t slot = 0; slot < slot_send_segments.size(); ++slot)
    {
        if (slot >= lu->symV2RowDownSendSizes.size() ||
            slot >= lu->symV2RowDownSendSegOffsets.size() ||
            slot >= lu->symV2RowDownSendSegCounts.size())
            ABORT("SymFact V2 row-down send table is invalid.");
        lu->symV2RowDownSendSegOffsets[slot] =
            lu->symV2RowDownSendSegsHost.size();
        if (slot_send_segments[slot].size() >
            static_cast<size_t>(std::numeric_limits<int>::max()))
            ABORT("SymFact V2 row-down has too many send segments.");
        lu->symV2RowDownSendSegCounts[slot] =
            static_cast<int>(slot_send_segments[slot].size());
        int send_size = slot_send_segment_values[slot];
        if (send_size < 0)
            ABORT("SymFact V2 row-down send size is invalid.");
        lu->symV2RowDownSendSizes[slot] = send_size;
        lu->symV2RowDownSendSegsHost.insert(
            lu->symV2RowDownSendSegsHost.end(),
            slot_send_segments[slot].begin(),
            slot_send_segments[slot].end());
        if (send_size > 0)
        {
            int_t lk = static_cast<int_t>(slot / static_cast<size_t>(lu->Pc));
            int dest_pc = static_cast<int>(slot % static_cast<size_t>(lu->Pc));
            int_t k0 = lu->symV2PanelGid(lk);
            size_reply_payloads[static_cast<size_t>(dest_pc)].push_back(k0);
            size_reply_payloads[static_cast<size_t>(dest_pc)].push_back(
                static_cast<int_t>(send_size));
        }
    }
    lu->symV2RowDownSendSegPoolCount =
        lu->symV2RowDownSendSegsHost.size();

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

    for (int_t k0 = 0; k0 < lu->nsupers; ++k0)
    {
        int_t needed_count = static_cast<int_t>(
            needed_row_blocks_by_panel[static_cast<size_t>(k0)].size());
        if (needed_count > 0)
            lu->symV2RowDownPlanReady[static_cast<size_t>(k0)] = 1;
        symldl_v2_trace_pcfrag_plan(lu, "plan", k0, needed_count);
    }

    size_t total_recv_map = 0;
    for (size_t pos = 0; pos < lu->symV2RowFragRecvMap.size(); ++pos)
    {
        lu->symV2RowFragRecvMapOffsets[pos] = total_recv_map;
        size_t map_size = lu->symV2RowFragRecvMap[pos].size();
        if (total_recv_map >
            std::numeric_limits<size_t>::max() - map_size)
            ABORT("SymFact V2 row-fragment receive map size overflows.");
        total_recv_map += map_size;
    }
    lu->symV2RowFragRecvMapPoolCount = total_recv_map;
}

template <typename Ftype>
static void symldl_v2_materialize_pcfrag_metadata(xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload)
        return;

    if (lu->symV2PartnerLSendBufPoolCount > 0)
    {
        if (lu->symV2PartnerLSendBufPoolGPU != NULL)
            ABORT("SymFact V2 partner send value pool already exists.");
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symV2PartnerLSendBufPoolGPU,
            sizeof(Ftype) * lu->symV2PartnerLSendBufPoolCount,
            "SymFact V2 partner send value pool allocation");
        for (size_t flat = 0; flat < lu->symV2PartnerLSendSizes.size();
             ++flat)
        {
            int size = lu->symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            size_t offset = lu->symV2PartnerLMapOffsets[flat];
            if (offset + static_cast<size_t>(size) >
                    lu->symV2PartnerLSendBufPoolCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 partner send value offset is invalid.");
            lu->symV2PartnerLSendBufsGPU[flat] =
                lu->symV2PartnerLSendBufPoolGPU + offset;
        }
    }

    const bool stream_l2l_send_maps =
        superlu_sym_v2_pc_fragment_ldl_native();
    if (lu->symL2LSendMapPoolCount > 0 && !stream_l2l_send_maps)
    {
        if (lu->symV2PartnerLPackedMaps.size() !=
            lu->symL2LSendMapPoolCount)
            ABORT("SymFact V2 partner-L send map pool size mismatch.");
        if (lu->symL2LSendMapPoolGPU != NULL)
            ABORT("SymFact V2 partner-L send map pool already exists.");
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symL2LSendMapPoolGPU,
            sizeof(int_t) * lu->symL2LSendMapPoolCount,
            "SymFact V2 partner-L send map pool allocation");
        gpuErrchk(cudaMemcpy(lu->symL2LSendMapPoolGPU,
                             lu->symV2PartnerLPackedMaps.data(),
                             sizeof(int_t) * lu->symL2LSendMapPoolCount,
                             cudaMemcpyHostToDevice));
        for (size_t flat = 0; flat < lu->symV2PartnerLSendSizes.size();
             ++flat)
        {
            int size = lu->symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            size_t offset = lu->symV2PartnerLMapOffsets[flat];
            if (offset + static_cast<size_t>(size) >
                    lu->symL2LSendMapPoolCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 partner-L send map offset is invalid.");
            lu->symL2LSendMapsGPU[flat] =
                lu->symL2LSendMapPoolGPU + offset;
        }
    }

    if (lu->symV2PartnerLHostSendPoolPinnedCount > 0)
    {
        if (lu->symV2PartnerLHostSendPoolPinned != NULL)
            ABORT("SymFact V2 partner send staging pool already exists.");
        gpuErrchk(cudaMallocHost(
            (void **) &lu->symV2PartnerLHostSendPoolPinned,
            sizeof(Ftype) * lu->symV2PartnerLHostSendPoolPinnedCount));
        for (size_t flat = 0; flat < lu->symV2PartnerLSendSizes.size();
             ++flat)
        {
            int size = lu->symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            size_t offset = lu->symV2PartnerLHostSendScratchOffsets[flat];
            if (offset + static_cast<size_t>(size) >
                    lu->symV2PartnerLHostSendPoolPinnedCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 partner send staging offset is invalid.");
            lu->symV2PartnerLHostSendBufsPinned[flat] =
                lu->symV2PartnerLHostSendPoolPinned + offset;
        }
    }
    else if (superlu_sym_v2_pinned_staging() && !superlu_cuda_aware_mpi())
    {
        for (size_t flat = 0; flat < lu->symV2PartnerLSendSizes.size();
             ++flat)
        {
            int size = lu->symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            if (lu->symV2PartnerLHostSendBufsPinned[flat] != NULL)
                ABORT("SymFact V2 partner send staging already exists.");
            gpuErrchk(cudaMallocHost(
                (void **) &lu->symV2PartnerLHostSendBufsPinned[flat],
                sizeof(Ftype) * static_cast<size_t>(size)));
        }
    }

    if (lu->symV2PartnerLRecvMapPoolCount > 0)
    {
        if (lu->symV2PartnerLRecvMapPoolGPU != NULL)
            ABORT("SymFact V2 partner receive map pool already exists.");
        std::vector<int_t> packed(lu->symV2PartnerLRecvMapPoolCount, 0);
        if (lu->symV2PartnerLRecvMap.size() !=
                lu->symV2PartnerLRecvMapOffsets.size() ||
            lu->symV2PartnerLRecvMap.size() !=
                lu->symV2PartnerLRecvMapsGPU.size())
            ABORT("SymFact V2 partner receive map table size mismatch.");
        for (size_t pos = 0; pos < lu->symV2PartnerLRecvMap.size(); ++pos)
        {
            if (lu->symV2PartnerLRecvMap[pos].empty())
                continue;
            size_t offset = lu->symV2PartnerLRecvMapOffsets[pos];
            if (offset + lu->symV2PartnerLRecvMap[pos].size() >
                    lu->symV2PartnerLRecvMapPoolCount ||
                offset + lu->symV2PartnerLRecvMap[pos].size() < offset)
                ABORT("SymFact V2 partner receive map offset is invalid.");
            std::copy(lu->symV2PartnerLRecvMap[pos].begin(),
                      lu->symV2PartnerLRecvMap[pos].end(),
                      packed.begin() + offset);
        }
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symV2PartnerLRecvMapPoolGPU,
            sizeof(int_t) * lu->symV2PartnerLRecvMapPoolCount,
            "SymFact V2 partner receive map pool allocation");
        gpuErrchk(cudaMemcpy(lu->symV2PartnerLRecvMapPoolGPU,
                             packed.data(),
                             sizeof(int_t) *
                                 lu->symV2PartnerLRecvMapPoolCount,
                             cudaMemcpyHostToDevice));
        for (size_t pos = 0; pos < lu->symV2PartnerLRecvMap.size(); ++pos)
        {
            if (!lu->symV2PartnerLRecvMap[pos].empty())
                lu->symV2PartnerLRecvMapsGPU[pos] =
                    lu->symV2PartnerLRecvMapPoolGPU +
                    lu->symV2PartnerLRecvMapOffsets[pos];
        }
    }

    if (lu->symV2RowFragRecvMapPoolCount > 0)
    {
        if (lu->symV2RowFragRecvMapPoolGPU != NULL)
            ABORT("SymFact V2 row-fragment receive map pool already exists.");
        std::vector<int_t> packed(lu->symV2RowFragRecvMapPoolCount, 0);
        if (lu->symV2RowFragRecvMap.size() !=
                lu->symV2RowFragRecvMapOffsets.size() ||
            lu->symV2RowFragRecvMap.size() !=
                lu->symV2RowFragRecvMapsGPU.size())
            ABORT("SymFact V2 row-fragment receive map table size mismatch.");
        for (size_t pos = 0; pos < lu->symV2RowFragRecvMap.size(); ++pos)
        {
            if (lu->symV2RowFragRecvMap[pos].empty())
                continue;
            size_t offset = lu->symV2RowFragRecvMapOffsets[pos];
            if (offset + lu->symV2RowFragRecvMap[pos].size() >
                    lu->symV2RowFragRecvMapPoolCount ||
                offset + lu->symV2RowFragRecvMap[pos].size() < offset)
                ABORT("SymFact V2 row-fragment receive map offset is invalid.");
            std::copy(lu->symV2RowFragRecvMap[pos].begin(),
                      lu->symV2RowFragRecvMap[pos].end(),
                      packed.begin() + offset);
        }
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symV2RowFragRecvMapPoolGPU,
            sizeof(int_t) * lu->symV2RowFragRecvMapPoolCount,
            "SymFact V2 row-fragment receive map pool allocation");
        gpuErrchk(cudaMemcpy(lu->symV2RowFragRecvMapPoolGPU,
                             packed.data(),
                             sizeof(int_t) *
                                 lu->symV2RowFragRecvMapPoolCount,
                             cudaMemcpyHostToDevice));
        for (size_t pos = 0; pos < lu->symV2RowFragRecvMap.size(); ++pos)
        {
            if (!lu->symV2RowFragRecvMap[pos].empty())
                lu->symV2RowFragRecvMapsGPU[pos] =
                    lu->symV2RowFragRecvMapPoolGPU +
                    lu->symV2RowFragRecvMapOffsets[pos];
        }
    }

    if (lu->symV2RowDownSendSegPoolCount > 0)
    {
        if (lu->symV2RowDownSendSegPoolGPU != NULL)
            ABORT("SymFact V2 row-down send segment pool already exists.");
        if (lu->symV2RowDownSendSegsHost.size() !=
            lu->symV2RowDownSendSegPoolCount)
            ABORT("SymFact V2 row-down send segment pool size mismatch.");
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symV2RowDownSendSegPoolGPU,
            sizeof(SymV2RowDownSendSegmentGPU) *
                lu->symV2RowDownSendSegPoolCount,
            "SymFact V2 row-down send segment pool allocation");
        gpuErrchk(cudaMemcpy(lu->symV2RowDownSendSegPoolGPU,
                             lu->symV2RowDownSendSegsHost.data(),
                             sizeof(SymV2RowDownSendSegmentGPU) *
                                 lu->symV2RowDownSendSegPoolCount,
                             cudaMemcpyHostToDevice));
        if (lu->symV2RowDownSendSegsGPU.size() !=
                lu->symV2RowDownSendSizes.size() ||
            lu->symV2RowDownSendSegOffsets.size() !=
                lu->symV2RowDownSendSizes.size() ||
            lu->symV2RowDownSendSegCounts.size() !=
                lu->symV2RowDownSendSizes.size())
            ABORT("SymFact V2 row-down send segment table size mismatch.");
        for (size_t slot = 0; slot < lu->symV2RowDownSendSizes.size();
             ++slot)
        {
            if (lu->symV2RowDownSendSizes[slot] <= 0)
                continue;
            int count = lu->symV2RowDownSendSegCounts[slot];
            if (count <= 0)
                ABORT("SymFact V2 row-down send segment slot is empty.");
            size_t offset = lu->symV2RowDownSendSegOffsets[slot];
            if (offset + static_cast<size_t>(count) >
                    lu->symV2RowDownSendSegPoolCount ||
                offset + static_cast<size_t>(count) < offset)
                ABORT("SymFact V2 row-down send segment offset is invalid.");
            lu->symV2RowDownSendSegsGPU[slot] =
                lu->symV2RowDownSendSegPoolGPU + offset;
        }
    }
}
#endif
