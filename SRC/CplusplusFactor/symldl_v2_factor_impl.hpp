#pragma once

#include "xlupanels.hpp"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::pdgstrf3dSymV2()
{
    if (!useSymV2Solve())
        ABORT("GPU3DVERSION=2 requires SymFact=YES.");
    if (!superlu_acc_offload)
        ABORT("SymFact GPU3DVERSION=2 requires GPU_ACC.");
    if (trf3Dpartition == NULL ||
        !trf3Dpartition->symV2ScheduleEnabled ||
        trf3Dpartition->symV2FactorNodes == NULL ||
        trf3Dpartition->symV2FactorLevelPtr == NULL ||
        trf3Dpartition->symV2NodeIperm == NULL ||
        trf3Dpartition->sForests == NULL ||
        trf3Dpartition->myTreeIdxs == NULL ||
        trf3Dpartition->myZeroTrIdxs == NULL)
        ABORT("SymFact GPU3DVERSION=2 requires an LDL forest.");

    symGPU3DVersion = 2;

    int tag_ub = set_tag_ub();
    gEtreeInfo_t gEtreeInfo = trf3Dpartition->gEtreeInfo;
    int_t *myNodeCount = trf3Dpartition->myNodeCount;
    int_t *myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t *myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    int_t **treePerm = trf3Dpartition->treePerm;
    sForest_t **sForests = trf3Dpartition->sForests;

    SCT->pdgstrfTimer = SuperLU_timer_();
    for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        if (!myZeroTrIdxs[ilvl])
        {
            sForest_t *sforest = sForests[myTreeIdxs[ilvl]];
            if (sforest != NULL)
            {
                double t_factor = SuperLU_timer_();
#ifdef HAVE_CUDA
                dsparseTreeFactorGPU(sforest, dFBufs, &gEtreeInfo, tag_ub);
#else
                ABORT("SymFact GPU3DVERSION=2 requires GPU_ACC.");
#endif
                SCT->tFactor3D[ilvl] = SuperLU_timer_() - t_factor;
                sforest->cost = SCT->tFactor3D[ilvl];
            }

            if (ilvl < maxLvl - 1)
            {
#ifdef HAVE_CUDA
                ancestorReduction3dGPU(ilvl, myNodeCount, treePerm);
#else
                ABORT("SymFact GPU3DVERSION=2 requires GPU_ACC.");
#endif
            }
        }

        SCT->tSchCompUdt3d[ilvl] =
            ilvl == 0 ? SCT->NetSchurUpTimer
                      : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];
    }

    MPI_Barrier(grid3d->comm);
    SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer;

    return 0;
}
