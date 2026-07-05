#include "dsymldl_v2_driver.h"

int dSymV2SolveEnabled(superlu_dist_options_t *options, int gpu3dVersion)
{
    return options != NULL && options->SymFact == YES &&
           gpu3dVersion == 2;
}

static int_t
dSymV2QuerySpace_dist(int_t n, dLUstruct_t *LUstruct,
                      dtrf3Dpartition_t *trf3Dpartition,
                      SuperLUStat_t *stat,
                      superlu_dist_mem_usage_t *mem_usage)
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t *xsup = Glu_persist->xsup;
    int_t iword = sizeof(int_t);
    int_t dword = sizeof(double);
    int_t nsupers = Glu_persist->supno[n - 1] + 1;

    mem_usage->for_lu = 0.;
    mem_usage->total = 0.;
    if (trf3Dpartition == NULL ||
        trf3Dpartition->symV2LocalPanelGids == NULL ||
        Llu == NULL || Llu->Lrowind_bc_ptr == NULL) {
        mem_usage->total = stat->peak_buffer;
        return 0;
    }

    for (int_t lk = 0; lk < trf3Dpartition->symV2LocalPanelCount; ++lk) {
        int_t gb = trf3Dpartition->symV2LocalPanelGids[lk];
        if (gb < 0 || gb >= nsupers) continue;

        int_t *index = Llu->Lrowind_bc_ptr[lk];
        if (index != NULL) {
            mem_usage->for_lu += (float)
                ((BC_HEADER + index[0] * LB_DESCRIPTOR + index[1]) * iword);
            mem_usage->for_lu += (float)(index[1] * SuperSize(gb) * dword);
        }
    }

    mem_usage->total = mem_usage->for_lu + stat->peak_buffer;
    return 0;
}

int_t *dSymV2CreateIdentityIpermSupno(int_t nsupers)
{
    int_t *iperm_c_supno = intMalloc_dist(nsupers);
    if (!iperm_c_supno)
        ABORT("Malloc fails for SymV2 iperm_c_supno[].");

    for (int_t k = 0; k < nsupers; ++k)
        iperm_c_supno[k] = k;

    return iperm_c_supno;
}

void dSymV2LluBufInit(dLUValSubBuf_t *LUvsb)
{
    LUvsb->Lsub_buf = intMalloc_dist(1);
    LUvsb->Lval_buf = doubleMalloc_dist(1);
    LUvsb->Usub_buf = intMalloc_dist(1);
    LUvsb->Uval_buf = doubleMalloc_dist(1);

    if (!LUvsb->Lsub_buf || !LUvsb->Lval_buf ||
        !LUvsb->Usub_buf || !LUvsb->Uval_buf)
        ABORT("Malloc fails for SymV2 LU staging sentinels.");
}

void dSymV2PrintFactorStats(int_t n, dLUstruct_t *LUstruct,
                            dtrf3Dpartition_t *trf3Dpartition,
                            gridinfo3d_t *grid3d,
                            superlu_dist_options_t *options,
                            SuperLUStat_t *stat, int *info,
                            int parSymbFact, float flinfo,
                            float dist_mem_use, float GA_mem_use,
                            superlu_dist_mem_usage_t symb_mem_usage,
                            superlu_dist_mem_usage_t *num_mem_usage)
{
    int_t tiny_pivots_local = stat->TinyPivots;
    int_t sytrf_2x2_local = stat->sytrf_2x2;
    int_t inertia_local[3] = {
        stat->inertia[0], stat->inertia[1], stat->inertia[2]
    };
    int_t inertia_sum[3];
    float for_lu = 0.0, total = 0.0, avg = 0.0, loc_max = 0.0;
    float mem_stage[3] = {0.0, 0.0, 0.0};
    struct { float val; int rank; } local_struct, global_struct;
    int nprocs3d = grid3d->nprow * grid3d->npcol * grid3d->npdep;

    MPI_Allreduce(&tiny_pivots_local, &stat->TinyPivots, 1,
                  mpi_int_t, MPI_SUM, grid3d->comm);
    MPI_Allreduce(&sytrf_2x2_local, &stat->sytrf_2x2, 1,
                  mpi_int_t, MPI_SUM, grid3d->comm);
    MPI_Allreduce(inertia_local, inertia_sum, 3, mpi_int_t,
                  MPI_SUM, grid3d->comm);
    stat->inertia[0] = inertia_sum[0];
    stat->inertia[1] = inertia_sum[1];
    stat->inertia[2] = inertia_sum[2];

    dSymV2QuerySpace_dist(n, LUstruct, trf3Dpartition, stat,
                          num_mem_usage);

    if (parSymbFact == TRUE)
    {
        mem_stage[0] = (-flinfo);
        mem_stage[1] = (-dist_mem_use);
        loc_max = SUPERLU_MAX(mem_stage[0], mem_stage[1]);
        if (options->RowPerm != NO)
            loc_max = SUPERLU_MAX(loc_max, GA_mem_use);
    }
    else
    {
        mem_stage[0] = symb_mem_usage.total + GA_mem_use;
        mem_stage[1] = symb_mem_usage.for_lu + dist_mem_use +
                       num_mem_usage->for_lu;
        loc_max = SUPERLU_MAX(mem_stage[0], mem_stage[1]);
    }

    mem_stage[2] = num_mem_usage->total;
    loc_max = SUPERLU_MAX(loc_max, mem_stage[2]);

    local_struct.val = loc_max;
    local_struct.rank = grid3d->iam;
    MPI_Reduce(&local_struct, &global_struct, 1, MPI_FLOAT_INT,
               MPI_MAXLOC, 0, grid3d->comm);
    int all_highmark_rank = global_struct.rank;
    float all_highmark_mem = global_struct.val * 1e-6;

    MPI_Reduce(&loc_max, &avg, 1, MPI_FLOAT, MPI_SUM, 0,
               grid3d->comm);
    MPI_Reduce(&num_mem_usage->for_lu, &for_lu, 1, MPI_FLOAT,
               MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(&num_mem_usage->total, &total, 1, MPI_FLOAT,
               MPI_SUM, 0, grid3d->comm);

    local_struct.val = num_mem_usage->for_lu;
    MPI_Reduce(&local_struct, &global_struct, 1, MPI_FLOAT_INT,
               MPI_MAXLOC, 0, grid3d->comm);
    int lu_max_rank = global_struct.rank;
    float lu_max_mem = global_struct.val * 1e-6;

    local_struct.val = stat->peak_buffer;
    MPI_Reduce(&local_struct, &global_struct, 1, MPI_FLOAT_INT,
               MPI_MAXLOC, 0, grid3d->comm);
    int buffer_peak_rank = global_struct.rank;
    float buffer_peak = global_struct.val * 1e-6;

    if (grid3d->iam == 0)
    {
        printf("\n** Memory Usage **********************************\n");
        printf("** Total highmark (MB):\n"
               "    Sum-of-all : %8.2f | Avg : %8.2f  | Max : %8.2f\n",
               avg * 1e-6,
               avg / nprocs3d * 1e-6,
               all_highmark_mem);
        printf("    Max at rank %d, different stages (MB):\n"
               "\t. symbfact        %8.2f\n"
               "\t. distribution    %8.2f\n"
               "\t. numfact         %8.2f\n",
               all_highmark_rank, mem_stage[0] * 1e-6,
               mem_stage[1] * 1e-6, mem_stage[2] * 1e-6);
        printf("** NUMfact space (MB): (sum-of-all-processes)\n"
               "    L/D :        %8.2f |  Total : %8.2f\n",
               for_lu * 1e-6, total * 1e-6);
        printf("\t. max at rank %d, max L/D memory (MB): %8.2f\n"
               "\t. max at rank %d, peak buffer (MB):    %8.2f\n",
               lu_max_rank, lu_max_mem,
               buffer_peak_rank, buffer_peak);
        printf("**************************************************\n\n");
        printf("** number of Tiny Pivots: %8d\n\n", stat->TinyPivots);
        printf("** number of 2x2 Pivots by sytrf: %8d\n\n", stat->sytrf_2x2);
        printf("** Inertia (pos,neg,zero): %10d %10d %10d\n\n",
               stat->inertia[0], stat->inertia[1], stat->inertia[2]);
        printf("info %10d\n", *info);
        fflush(stdout);
    }
}
