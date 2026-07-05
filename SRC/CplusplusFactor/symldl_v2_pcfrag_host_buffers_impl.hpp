#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_pcfrag_workspace_utils.hpp"

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
