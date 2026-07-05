#pragma once

#include <algorithm>
#include <vector>

#include "xlupanels.hpp"

template <typename Ftype>
static int *symldl_v2_make_node_mask(xLUstruct_t<Ftype> *lu)
{
    int *mask = int32Calloc_dist((int) lu->nsupers);
    if (mask == NULL)
        ABORT("Calloc fails for SymFact V2 local node mask.");

    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        if (lu->trf3Dpartition != NULL &&
            lu->trf3Dpartition->superGridMap != NULL &&
            lu->trf3Dpartition->superGridMap[k] != NOT_IN_GRID)
            mask[k] = 1;
    }
    return mask;
}

template <typename Ftype>
static void symldl_v2_initialize_diag_state(xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve())
        return;

    if (lu->options != NULL && lu->options->batchCount > 0)
        ABORT("SymFact GPU3DVERSION=2 does not support batchCount>0 until LDL-native batch sizing is implemented.");
    if (superlu_sym_v2_pc_fragment_ldl_native() &&
        !superlu_sym_v2_pc_fragment_schur())
        ABORT("GPU3DV2_PC_FRAGMENT_LDL_NATIVE requires GPU3DV2_PC_FRAGMENT_SCHUR=1.");

    symldl_v2_validate_async_pcfrag_config();
    lu->symV2DiagBlocks.assign((size_t) lu->nsupers, (Ftype *) NULL);
#ifdef HAVE_CUDA
    lu->symV2DiagBlocksGPU.assign((size_t) lu->nsupers, (Ftype *) NULL);
#endif
}

template <typename Ftype>
static void symldl_v2_build_l_panels(
    xLUstruct_t<Ftype> *lu,
    LUStruct_type<Ftype> *LUstruct,
    std::vector<int_t> &localLvalSendCounts,
    std::vector<int_t> &localLidxSendCounts)
{
    int_t **Lrowind_bc_ptr = LUstruct->Llu->Lrowind_bc_ptr;
    Ftype **Lnzval_bc_ptr = LUstruct->Llu->Lnzval_bc_ptr;

    for (int_t i = 0; i < lu->symV2PanelCount(); ++i)
    {
        int_t k0 = lu->symV2PanelGid(i);
        int_t *lsub = Lrowind_bc_ptr ? Lrowind_bc_ptr[i] : NULL;
        Ftype *lval = Lnzval_bc_ptr ? Lnzval_bc_ptr[i] : NULL;
        if (lsub != NULL && lu->isNodeInMyGrid[k0] == 1)
        {
            int_t isDiagIncluded = (lu->myrow == lu->symV2DiagRoot(k0)) ? 1 : 0;
            xlpanel_t<Ftype> lpanel(k0, lsub, lval, lu->xsup, isDiagIncluded);
            lu->lPanelVec[i] = lpanel;
            lu->maxLvalCount = std::max(lu->lPanelVec[i].nzvalSize(),
                                        lu->maxLvalCount);
            lu->maxLidxCount = std::max(lu->lPanelVec[i].indexSize(),
                                        lu->maxLidxCount);
            localLvalSendCounts[i] = lu->lPanelVec[i].nzvalSize();
            localLidxSendCounts[i] = lu->lPanelVec[i].indexSize();
        }
    }
}

template <typename Ftype>
static void symldl_v2_exchange_l_panel_counts(
    xLUstruct_t<Ftype> *lu,
    const std::vector<int_t> &localLvalSendCounts,
    const std::vector<int_t> &localLidxSendCounts)
{
    std::vector<int_t> localLvalBySuper((size_t) lu->nsupers, 0);
    std::vector<int_t> localLidxBySuper((size_t) lu->nsupers, 0);
    for (int_t i = 0; i < lu->symV2PanelCount(); ++i)
    {
        int_t k0 = lu->symV2PanelGid(i);
        localLvalBySuper[k0] = localLvalSendCounts[i];
        localLidxBySuper[k0] = localLidxSendCounts[i];
    }

    MPI_Allreduce(localLvalBySuper.data(), lu->LvalSendCounts.data(),
                  (int) lu->nsupers, mpi_int_t, MPI_SUM, lu->grid3d->rscp.comm);
    MPI_Allreduce(localLidxBySuper.data(), lu->LidxSendCounts.data(),
                  (int) lu->nsupers, mpi_int_t, MPI_SUM, lu->grid3d->rscp.comm);
}
