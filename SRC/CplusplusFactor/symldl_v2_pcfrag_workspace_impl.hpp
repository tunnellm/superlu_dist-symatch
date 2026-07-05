#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "xlupanels.hpp"

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
        if (superlu_sym_v2_pc_fragment_schur() && lu->Pc > 1)
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
    if (superlu_sym_v2_pc_fragment_schur() && lu->Pc > 1)
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
    bool pc_fragment =
        superlu_sym_v2_pc_fragment_schur() && lu->Pr > 1 && lu->Pc > 1;
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
        (lu->Pr > 1 && lu->Pc > 1 &&
         superlu_sym_v2_pc_fragment_schur()) ? 1 : 0);

    size_t partner_active = symldl_v2_checked_product(
        l2l_slots, static_cast<size_t>(lu->Pr),
        "SymFact V2 partner-L active table overflows.");
    size_t row_active = symldl_v2_checked_product(
        l2l_slots, static_cast<size_t>(lu->Pc),
        "SymFact V2 row-fragment active table overflows.");
    lu->symV2PartnerLSendRowActive.assign(partner_active, 0);
    lu->symV2RowFragSendActive.assign(row_active, 0);

    size_t partner_recv_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pr),
        "SymFact V2 partner receive table overflows.");
    size_t row_recv_slots = symldl_v2_checked_product(
        static_cast<size_t>(lu->nsupers), static_cast<size_t>(lu->Pc),
        "SymFact V2 row receive table overflows.");
    lu->symV2PartnerLRecvSizes.assign(partner_recv_slots, 0);
    lu->symV2PartnerLRecvIndex.assign(static_cast<size_t>(lu->nsupers),
                                      std::vector<int_t>());
    lu->symV2PartnerLRecvIndexBySrc.assign(partner_recv_slots,
                                           std::vector<int_t>());
    lu->symV2PartnerLRecvMap.assign(partner_recv_slots,
                                    std::vector<int_t>());
    lu->symV2PartnerLRecvMapOffsets.assign(partner_recv_slots, 0);
    lu->symV2PartnerLRecvMapsGPU.assign(partner_recv_slots, NULL);
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
    lu->symV2RowDownSegs.clear();
    lu->symV2RowDownRecvSizes.assign(row_recv_slots, 0);
    lu->symV2RowDownPlanReady.assign(static_cast<size_t>(lu->nsupers), 0);

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
    lu->symV2RowFragSendCountsScratch.assign(static_cast<size_t>(lu->Pc), 0);
    lu->symV2RowFragSendOffsetsScratch.assign(static_cast<size_t>(lu->Pc), 0);
    lu->symV2RowFragSendReqsScratch.clear();
#else
    (void) lu;
#endif
}

#ifdef HAVE_CUDA
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
            for (int_t col = 0; col < knsupc; ++col)
            {
                for (int_t i = 0; i < len; ++i)
                {
                    if (map_pos >= map_end)
                        ABORT("SymFact V2 partner-L send map overrun.");
                    int_t src_row = rows_sorted ? i : row_order[i].second;
                    lu->symV2PartnerLPackedMaps[map_pos++] =
                        lpanel.blkPtrOffset(lb) + src_row + col * nsupr;
                }
            }
            map_write_offsets[flat] = map_pos;
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
static void symldl_v2_materialize_pcfrag_metadata(xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload)
        return;

    if (lu->symL2LSendMapPoolCount > 0)
    {
        if (lu->symV2PartnerLPackedMaps.size() !=
            lu->symL2LSendMapPoolCount)
            ABORT("SymFact V2 partner-L send map pool size mismatch.");
        if (lu->symL2LSendMapPoolGPU != NULL)
            ABORT("SymFact V2 partner-L send map pool already exists.");
        gpuErrchk(cudaMalloc(
            (void **) &lu->symL2LSendMapPoolGPU,
            sizeof(int_t) * lu->symL2LSendMapPoolCount));
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
}
#endif
