#pragma once

#include <cinttypes>
#include <cstdio>

#include "xlupanels.hpp"
#include "symldl_v2_cpu_scheduler_impl.hpp"
#include "symldl_v2_cpu_reduction_impl.hpp"
#include "symldl_v2_cpu_profile_impl.hpp"
#include "symldl_v2_factor_gpu_bridge.hpp"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::pdgstrf3dSymV2()
{
    if (!useSymV2Solve())
        ABORT("GPU3DVERSION=2 requires SymFact=YES.");
    if (trf3Dpartition == NULL ||
        !trf3Dpartition->symV2ScheduleEnabled ||
        trf3Dpartition->symV2FactorNodes == NULL ||
        trf3Dpartition->symV2FactorLevelPtr == NULL ||
        trf3Dpartition->symV2NodeIperm == NULL ||
        trf3Dpartition->sForests == NULL ||
        trf3Dpartition->myTreeIdxs == NULL ||
        trf3Dpartition->myZeroTrIdxs == NULL)
        ABORT("SymFact V2 requires an LDL forest.");
    symGPU3DVersion = 2;
    symV2RouteProfileReset();
    symV2CpuProfileEnabled = superlu_sym_v2_route_profile();
    SymLDLV2CpuCapacitySnapshot cpu_capacity_before = {0, 0};
    if (symV2UsesCpuFactor())
    {
        cpu_capacity_before = symldl_v2_cpu_capacity_snapshot(this);
        symV2CpuFactorLoopActive = true;
    }

    int tag_ub = set_tag_ub();
    gEtreeInfo_t gEtreeInfo = trf3Dpartition->gEtreeInfo;
    int_t *myNodeCount = trf3Dpartition->myNodeCount;
    int_t *myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t *myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    int_t **treePerm = trf3Dpartition->treePerm;
    sForest_t **sForests = trf3Dpartition->sForests;

    SCT->pdgstrfTimer = SuperLU_timer_();
    const bool profile_cpu_window =
        symV2UsesCpuFactor() && symV2CpuProfileEnabled &&
        symldl_v2_cpu_scheduler_kind() ==
            SYM_LDL_V2_CPU_SCHEDULER_WINDOW;
    if (profile_cpu_window && grid3d->iam == 0)
        printf("SymFact V2 CPU window numeric factor begin: levels=%d slots=%zu\n",
               maxLvl, symV2CpuWindowStates.size());
    for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        if (!myZeroTrIdxs[ilvl])
        {
            sForest_t *sforest = sForests[myTreeIdxs[ilvl]];
            if (sforest != NULL)
            {
                if (profile_cpu_window && grid3d->iam == 0)
                    printf("SymFact V2 CPU window forest begin: level=%d nodes=%lld\n",
                           ilvl, static_cast<long long>(sforest->nNodes));
                double t_factor = SuperLU_timer_();
                if (symV2UsesCpuFactor())
                {
                    symldl_v2_cpu_factor_forest(
                        this, sforest, dFBufs, &gEtreeInfo);
                }
                else
                {
#ifdef HAVE_CUDA
                    pdgstrf3d_symv2_factor_forest_cuda_bridge(
                        static_cast<void *>(this), sforest, dFBufs,
                        &gEtreeInfo, tag_ub);
#else
                    ABORT("SymFact V2 GPU backend is unavailable.");
#endif
                }
                SCT->tFactor3D[ilvl] = SuperLU_timer_() - t_factor;
                sforest->cost = SCT->tFactor3D[ilvl];
                if (profile_cpu_window && grid3d->iam == 0)
                    printf("SymFact V2 CPU window forest end: level=%d time=%.6f\n",
                           ilvl, SCT->tFactor3D[ilvl]);
            }

            if (ilvl < maxLvl - 1)
            {
                if (symV2UsesCpuFactor())
                {
                    if (symldl_v2_cpu_scheduler_kind() ==
                        SYM_LDL_V2_CPU_SCHEDULER_WINDOW)
                        symldl_v2_cpu_window_ancestor_reduction(
                            this, ilvl, myNodeCount, treePerm);
                    else
                        symldl_v2_cpu_ancestor_reduction(
                            this, ilvl, myNodeCount, treePerm);
                }
                else
                {
#ifdef HAVE_CUDA
                    pdgstrf3d_symv2_ancestor_cuda_bridge(
                        static_cast<void *>(this), ilvl, myNodeCount,
                        treePerm);
#else
                    ABORT("SymFact V2 GPU backend is unavailable.");
#endif
                }
            }
        }

        SCT->tSchCompUdt3d[ilvl] =
            ilvl == 0 ? SCT->NetSchurUpTimer
                      : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];
    }

    if (symV2UsesCpuFactor())
    {
        symV2CpuFactorLoopActive = false;
        SymLDLV2CpuCapacitySnapshot cpu_capacity_after =
            symldl_v2_cpu_capacity_snapshot(this);
        if (cpu_capacity_before.hash != cpu_capacity_after.hash ||
            cpu_capacity_before.total != cpu_capacity_after.total)
        {
            ++symV2CpuRuntimeVectorGrowths;
            ABORT("SymFact V2 CPU factor loop grew persistent vectors.");
        }
        symV2CpuOutstandingRequests =
            symldl_v2_cpu_count_outstanding_requests(this);
        if (symV2CpuOutstandingRequests != 0)
            ABORT("SymFact V2 CPU factorization left MPI requests active.");
    }

    MPI_Barrier(grid3d->comm);
    SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer;
    symV2RouteProfilePrint("factor");
    if (symV2UsesCpuFactor() && symV2CpuProfileEnabled)
        symldl_v2_cpu_profile_print(this);

    return 0;
}
