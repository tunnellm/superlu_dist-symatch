#pragma once 
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cassert>
#include "superlu_defs.h"
#include "luAuxStructTemplated.hpp"
#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#include "xlupanels_GPU.cuh"
#include "cublas_cusolver_wrappers.hpp"
#include "gpu_mpi_utils.hpp"
#endif
#include "lupanels.hpp"  //unneeded??
#include "xlupanels.hpp"
#include "superlu_blas.hpp"

#ifdef HAVE_CUDA
template <>
int_t xLUstruct_t<double>::dSymStartL2UGPU(int_t k, int_t stream_offset);
#endif

#ifdef SLU_SYM_GPU3D_DEBUG_TRACE
static inline void xlu_sym_gpu3d_trace(gridinfo3d_t *grid3d, const char *msg)
{
    std::printf("[sym-gpu3d-trace] rank %d: %s\n",
                (grid3d != NULL) ? grid3d->iam : -1, msg);
    std::fflush(stdout);
}
#else
static inline void xlu_sym_gpu3d_trace(gridinfo3d_t *grid3d, const char *msg)
{
    (void)grid3d;
    (void)msg;
}
#endif

static inline size_t xlu_checked_product(size_t a, size_t b, const char *what)
{
    (void) what;
    if (a != 0 && b > static_cast<size_t>(-1) / a)
        ABORT("Workspace size overflows allocation size.");
    return a * b;
}

static inline size_t xlu_checked_alloc_bytes(int_t count, size_t elem_size,
                                             const char *what)
{
    if (count < 0)
        ABORT("Negative allocation size.");
    size_t n = static_cast<size_t>(count);
    if (static_cast<int_t>(n) != count)
        ABORT("Allocation size overflows int_t.");
    return xlu_checked_product(n, elem_size, what);
}

static inline size_t xlu_checked_square_alloc_bytes(int_t dim, size_t elem_size,
                                                    const char *what)
{
    if (dim < 0)
        ABORT("Negative allocation size.");
    size_t n = static_cast<size_t>(dim);
    if (static_cast<int_t>(n) != dim)
        ABORT("Allocation size overflows int_t.");
    size_t count = xlu_checked_product(n, n, what);
    return xlu_checked_product(count, elem_size, what);
}

static inline size_t xlu_checked_bigv_alloc_bytes(int_t ldt, int_t num_threads,
                                                  size_t elem_size,
                                                  const char *what)
{
    if (ldt < 0 || num_threads < 0)
        ABORT("Negative allocation size.");
    size_t n = static_cast<size_t>(ldt);
    size_t nt = static_cast<size_t>(num_threads);
    if (static_cast<int_t>(n) != ldt || static_cast<int_t>(nt) != num_threads)
        ABORT("Allocation size overflows int_t.");
    size_t count = xlu_checked_product(8, n, what);
    count = xlu_checked_product(count, n, what);
    count = xlu_checked_product(count, nt, what);
    return xlu_checked_product(count, elem_size, what);
}

static inline int xlu_gpu3d_contract()
{
    const char *env = std::getenv("GPU3DCONTRACT");
    if (env == NULL || env[0] == '\0')
        return 0;

    char *end = NULL;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0 || value > 3)
        ABORT("GPU3DCONTRACT must be one of 0, 1, 2, or 3.");
    return (int)value;
}

static inline double xlu_env_double(const char *name, double fallback)
{
    const char *env = std::getenv(name);
    if (env == NULL || env[0] == '\0')
        return fallback;

    char *end = NULL;
    double value = std::strtod(env, &end);
    if (end == env || *end != '\0' || !std::isfinite(value) || value <= 0.0)
        ABORT("Invalid positive floating-point environment value.");
    return value;
}

static inline int xlu_sytrf_count_2x2(const int *ipiv, int n)
{
    int n2x2 = 0;
    for (int i = 0; i < n;)
    {
        if (ipiv[i] > 0)
        {
            ++i;
        }
        else
        {
            ++n2x2;
            i += 2;
        }
    }
    return n2x2;
}

static inline double xlu_sym_inverse_scaled_residual(const double *a, int lda,
                                                     const double *ainv, int n)
{
    double a_norm = 0.0;
    double inv_norm = 0.0;
    double err_norm = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double row_sum = 0.0;
        double inv_row_sum = 0.0;
        double err_row_sum = 0.0;

        for (int j = 0; j < n; ++j)
        {
            const double aij = (i >= j) ? a[i + (size_t)j * lda]
                                        : a[j + (size_t)i * lda];
            row_sum += std::fabs(aij);
            inv_row_sum += std::fabs(ainv[i + (size_t)j * n]);

            double prod = 0.0;
            for (int kk = 0; kk < n; ++kk)
            {
                const double aik = (i >= kk) ? a[i + (size_t)kk * lda]
                                             : a[kk + (size_t)i * lda];
                prod += aik * ainv[kk + (size_t)j * n];
            }
            if (!std::isfinite(prod))
                return DBL_MAX;
            const double target = (i == j) ? 1.0 : 0.0;
            err_row_sum += std::fabs(prod - target);
        }

        a_norm = SUPERLU_MAX(a_norm, row_sum);
        inv_norm = SUPERLU_MAX(inv_norm, inv_row_sum);
        err_norm = SUPERLU_MAX(err_norm, err_row_sum);
    }

    if (!std::isfinite(a_norm) || !std::isfinite(inv_norm) ||
        !std::isfinite(err_norm) || inv_norm == 0.0)
        return DBL_MAX;

    double denom = a_norm * inv_norm * DBL_EPSILON * (double)SUPERLU_MAX(n, 1);
    if (denom < DBL_EPSILON)
        denom = DBL_EPSILON;
    return err_norm / denom;
}

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
template <typename Ftype>
void xLUstruct_t<Ftype>::printSymGPU3DTiming()
{
    static const char *labels[SYM_GPU3D_T_COUNT] = {
        "l2u_start",
        "diag_d2h",
        "diag_d2h_copy",
        "diag_d2h_wait",
        "diag_prefetch_issue",
        "diag_prefetch_wait",
        "cpu_sytrf",
        "gpu_sytri",
        "gpu_sytri_validate",
        "cpu_sytri",
        "diag_pack",
        "diag_bcast",
        "inv_h2d",
        "ldiag_d2d",
        "lpanel_transform",
        "l2u_finish",
        "panel_bcast",
        "panel_bcast_mpi",
        "panel_index_d2h",
        "panel_bcast_singleton",
        "schur_update",
        "schur_sync",
        "lookahead_update",
        "lookahead_sync",
        "exclude_update",
        "sched_lookahead_dispatch",
        "sched_prefetch_ready",
        "sched_factor_dispatch",
        "sched_bcast_advance",
        "sched_final_sync",
        "initial_factor_dispatch",
        "initial_panel_bcast",
        "factor_tree_wall"
    };
    static const char *stat_labels[SYM_GPU3D_S_COUNT] = {
        "factor_trees",
        "factor_nodes",
        "initial_factor_nodes",
        "parent_factor_nodes",
        "lookahead_updates",
        "exclude_updates",
        "panel_bcasts",
        "panel_bcast_bytes",
        "panel_bcast_mpi_bytes",
        "panel_index_d2h_bytes",
        "l2u_local_bytes",
        "l2u_send_bytes",
        "l2u_recv_bytes",
        "l2u_host_staging_bytes",
        "l2u_cuda_aware_send_bytes",
        "diag_d2h_bytes",
        "diag_prefetch_hits",
        "diag_prefetch_misses",
        "diag_prefetch_issues",
        "sched_windows",
        "sched_window_nodes",
        "sched_ready_bcasts",
        "sched_max_window",
        "sched_max_num_la"
    };

    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (!mpi_initialized || mpi_finalized || grid3d == NULL)
        return;

    double sum_time[SYM_GPU3D_T_COUNT] = {};
    double max_time[SYM_GPU3D_T_COUNT] = {};
    long long sum_count[SYM_GPU3D_T_COUNT] = {};
    long long sum_stat[SYM_GPU3D_S_COUNT] = {};
    long long min_stat[SYM_GPU3D_S_COUNT] = {};
    long long max_stat[SYM_GPU3D_S_COUNT] = {};
    int nranks = 1;

    MPI_Comm_size(grid3d->comm, &nranks);
    MPI_Reduce(symGPU3DTime, sum_time, SYM_GPU3D_T_COUNT, MPI_DOUBLE,
               MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(symGPU3DTime, max_time, SYM_GPU3D_T_COUNT, MPI_DOUBLE,
               MPI_MAX, 0, grid3d->comm);
    MPI_Reduce(symGPU3DCount, sum_count, SYM_GPU3D_T_COUNT,
               MPI_LONG_LONG_INT, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(symGPU3DStat, sum_stat, SYM_GPU3D_S_COUNT,
               MPI_LONG_LONG_INT, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(symGPU3DStat, min_stat, SYM_GPU3D_S_COUNT,
               MPI_LONG_LONG_INT, MPI_MIN, 0, grid3d->comm);
    MPI_Reduce(symGPU3DStat, max_stat, SYM_GPU3D_S_COUNT,
               MPI_LONG_LONG_INT, MPI_MAX, 0, grid3d->comm);

    if (grid3d->iam != 0)
        return;

    printf("** SymFact GPU3D timing debug (SLU_ENABLE_SYM_GPU3D_TIMING) **\n");
    printf("   %-22s %12s %12s %12s\n", "phase", "sum(s)", "max_rank(s)", "calls");
    for (int i = 0; i < SYM_GPU3D_T_COUNT; ++i)
    {
        if (sum_count[i] == 0 && sum_time[i] == 0.0 && max_time[i] == 0.0)
            continue;
        printf("   %-22s %12.6f %12.6f %12lld\n",
               labels[i], sum_time[i], max_time[i], sum_count[i]);
    }
    printf("** SymFact GPU3D rank stats (SLU_ENABLE_SYM_GPU3D_TIMING) **\n");
    printf("   %-28s %16s %16s %16s %16s\n",
           "stat", "sum", "avg_rank", "min_rank", "max_rank");
    for (int i = 0; i < SYM_GPU3D_S_COUNT; ++i)
    {
        if (sum_stat[i] == 0 && min_stat[i] == 0 && max_stat[i] == 0)
            continue;
        double avg = (nranks > 0) ? ((double)sum_stat[i] / (double)nranks) : 0.0;
        printf("   %-28s %16lld %16.2f %16lld %16lld\n",
               stat_labels[i], sum_stat[i], avg, min_stat[i], max_stat[i]);
    }
    fflush(stdout);
}
#endif

template <typename Ftype>
diagFactBufs_type<Ftype> **xLUstruct_t<Ftype>::initDiagFactBufsArr(int_t num_bufs, int_t ldt)
{

    // diagFactBufs_type<Ftype> **dFBufs = new diagFactBufs_type<Ftype> *[num_bufs]; // use SuperLU_MALLOC instead
    size_t ptr_bytes = xlu_checked_alloc_bytes(num_bufs,
                                               sizeof(diagFactBufs_type<Ftype> *),
                                               "diagonal factor buffer array");
    size_t block_bytes = xlu_checked_square_alloc_bytes(ldt, sizeof(Ftype),
                                                        "diagonal factor block");
    if (num_bufs == 0)
        return NULL;
    diagFactBufs_type<Ftype> **dFBufs = (diagFactBufs_type<Ftype> **)SUPERLU_MALLOC(ptr_bytes);
    if (dFBufs == NULL)
        ABORT("Malloc fails for diagonal factor buffer array.");
    for (int_t i = 0; i < num_bufs; i++)
    {
        // dFBufs[i] = new diagFactBufs_type<Ftype>; // use SuperLU_MALLOC instead
        dFBufs[i] = (diagFactBufs_type<Ftype> *)SUPERLU_MALLOC(sizeof(diagFactBufs_type<Ftype>));
        if (dFBufs[i] == NULL)
            ABORT("Malloc fails for diagonal factor buffers.");
        dFBufs[i]->BlockUFactor = (Ftype *)SUPERLU_MALLOC(block_bytes);
        dFBufs[i]->BlockLFactor = (Ftype *)SUPERLU_MALLOC(block_bytes);
        if (dFBufs[i]->BlockUFactor == NULL || dFBufs[i]->BlockLFactor == NULL)
            ABORT("Malloc fails for diagonal factor buffers.");
    }
    return dFBufs;
}

template <typename Ftype>
int xLUstruct_t<Ftype>::freeDiagFactBufsArr(int_t num_bufs, diagFactBufs_type<Ftype> ** dFBufs)
{
    for (int i = 0; i < num_bufs; i++)
    {
        SUPERLU_FREE(dFBufs[i]->BlockUFactor);
        SUPERLU_FREE(dFBufs[i]->BlockLFactor);
        SUPERLU_FREE(dFBufs[i]);
    }
    /* Sherry fix:
     * mxLeafNode can be 0 for the replicated layers of the processes ?? */
    if ( num_bufs ) SUPERLU_FREE(dFBufs);

    return 0;
}


#ifdef HAVE_CUDA
template <typename Ftype>
xupanel_t<Ftype> xLUstruct_t<Ftype>::getKUpanel(int_t k, int_t offset)
{
    return (
        myrow == krow(k) ? 
        uPanelVec[g2lRow(k)] : 
        xupanel_t<Ftype>(UidxRecvBufs[offset], UvalRecvBufs[offset],
            A_gpu.UidxRecvBufs[offset], A_gpu.UvalRecvBufs[offset])
    );
}

template <typename Ftype>
xlpanel_t<Ftype> xLUstruct_t<Ftype>::getKLpanel(int_t k, int_t offset)
{ 
    return (
        mycol == kcol(k) ? 
        lPanelVec[g2lCol(k)] : 
        xlpanel_t<Ftype>(LidxRecvBufs[offset], LvalRecvBufs[offset],
            A_gpu.LidxRecvBufs[offset], A_gpu.LvalRecvBufs[offset])
    );
}
#endif

template <typename Ftype>
Ftype* getBigV(int_t ldt, int_t num_threads)
{
    Ftype *bigV;
    size_t bigv_bytes = xlu_checked_bigv_alloc_bytes(ldt, num_threads,
                                                     sizeof(Ftype),
                                                     "dgemm buffV");
    if (!(bigV = (Ftype*) SUPERLU_MALLOC (bigv_bytes)))
        ABORT ("Malloc failed for dgemm buffV");
    return bigV;
}

/* Constructor */
template <typename Ftype>
xLUstruct_t<Ftype>::xLUstruct_t(int_t nsupers_, int_t ldt_,
                             trf3dpartitionType<Ftype> *trf3Dpartition_, 
                             LUStruct_type<Ftype> *LUstruct,
                             gridinfo3d_t *grid3d_in,
                             SCT_t *SCT_, superlu_dist_options_t *options_,
                             SuperLUStat_t *stat_, 
                             threshPivValType<Ftype> thresh_, int *info_) :
                             nsupers(nsupers_), trf3Dpartition(trf3Dpartition_),
                             ldt(ldt_), /* maximum supernode size */
				     grid3d(grid3d_in), SCT(SCT_),
				     options(options_), stat(stat_),
	                             LUstructPtr(LUstruct), symL2UOrders(NULL),
	                             symFactWork(NULL), symFactIPIV(NULL),
	                             symFactWorkSize(0), symFactTagUb(0),
	                             thresh(thresh_), info(info_), anc25d(grid3d_in)
{
    xlu_sym_gpu3d_trace(grid3d, "enter xLUstruct_t constructor");
    maxLvl = log2i(grid3d->zscp.Np) + 1;
    isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl, trf3Dpartition->myNodeCount, trf3Dpartition->treePerm);
    superlu_acc_offload = sp_ienv_dist(10, options); // get_acc_offload();
    xlu_sym_gpu3d_trace(grid3d, "constructor after isNodeInMyGrid");

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d_in->iam, "Enter xLUstruct_t constructor");
#endif
    grid = &(grid3d->grid2d);
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW(iam, grid);
    mycol = MYCOL(iam, grid);
    if (options->SymFact == YES)
    {
        if (options->CommL != YES)
            ABORT("LUv1 SymFact requires CommL=YES to reconstruct U panels.");
        symFactTagUb = set_tag_ub();
        if (symFactTagUb <= 0)
            ABORT("Invalid MPI tag upper bound for LUv1 SymFact L2U communication.");
    }
    xsup = LUstruct->Glu_persist->xsup;
    int_t **Lrowind_bc_ptr = LUstruct->Llu->Lrowind_bc_ptr;
    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    Ftype **Lnzval_bc_ptr = LUstruct->Llu->Lnzval_bc_ptr;
    Ftype **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;

    lPanelVec = new xlpanel_t<Ftype>[CEILING(nsupers, Pc)];
    uPanelVec = new xupanel_t<Ftype>[CEILING(nsupers, Pr)];
    xlu_sym_gpu3d_trace(grid3d, "constructor after panel vector allocation");
    // create the lvectors
    maxLvalCount = 0;
    maxLidxCount = 0;
    maxUvalCount = 0;
    maxUidxCount = 0;

    std::vector<int_t> localLvalSendCounts(CEILING(nsupers, Pc), 0);
    std::vector<int_t> localUvalSendCounts(CEILING(nsupers, Pr), 0);
    std::vector<int_t> localLidxSendCounts(CEILING(nsupers, Pc), 0);
    std::vector<int_t> localUidxSendCounts(CEILING(nsupers, Pr), 0);

    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        int_t k0 = i * Pc + mycol;
        if (Lrowind_bc_ptr[i] != NULL && isNodeInMyGrid[k0] == 1)
        {
            int_t isDiagIncluded = 0;

            if (myrow == krow(k0))
                isDiagIncluded = 1;
            xlpanel_t<Ftype> lpanel(k0, Lrowind_bc_ptr[i], Lnzval_bc_ptr[i], xsup, isDiagIncluded);
            lPanelVec[i] = lpanel;
            maxLvalCount = std::max(lPanelVec[i].nzvalSize(), maxLvalCount);
            maxLidxCount = std::max(lPanelVec[i].indexSize(), maxLidxCount);
            localLvalSendCounts[i] = lPanelVec[i].nzvalSize();
            localLidxSendCounts[i] = lPanelVec[i].indexSize();
        }
    }

    // create the vectors
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            int_t globalId = i * Pr + myrow;
            xupanel_t<Ftype> upanel(globalId, Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
            uPanelVec[i] = upanel;
            maxUvalCount = std::max(uPanelVec[i].nzvalSize(), maxUvalCount);
            maxUidxCount = std::max(uPanelVec[i].indexSize(), maxUidxCount);
            localUvalSendCounts[i] = uPanelVec[i].nzvalSize();
            localUidxSendCounts[i] = uPanelVec[i].indexSize();
        }
    }

    // compute the send sizes
    // send and recv count for 2d comm
    LvalSendCounts.resize(nsupers);
    UvalSendCounts.resize(nsupers);
    LidxSendCounts.resize(nsupers);
    UidxSendCounts.resize(nsupers);

    std::vector<int_t> recvBuf(std::max(CEILING(nsupers, Pr), CEILING(nsupers, Pc)), 0);

    for (int pr = 0; pr < Pr; pr++)
    {
        int npr = CEILING(nsupers, Pr);
        std::copy(localUvalSendCounts.begin(), localUvalSendCounts.end(), recvBuf.begin());
        // Send the value counts ;
        MPI_Bcast((void *)recvBuf.data(), npr, mpi_int_t, pr, grid3d->cscp.comm);
        for (int i = 0; i * Pr + pr < nsupers; i++)
        {
            UvalSendCounts[i * Pr + pr] = recvBuf[i];
        }

        std::copy(localUidxSendCounts.begin(), localUidxSendCounts.end(), recvBuf.begin());
        // send the index count
        MPI_Bcast((void *)recvBuf.data(), npr, mpi_int_t, pr, grid3d->cscp.comm);
        for (int i = 0; i * Pr + pr < nsupers; i++)
        {
            UidxSendCounts[i * Pr + pr] = recvBuf[i];
        }
    }

    for (int pc = 0; pc < Pc; pc++)
    {
        int npc = CEILING(nsupers, Pc);
        std::copy(localLvalSendCounts.begin(), localLvalSendCounts.end(), recvBuf.begin());
        // Send the value counts ;
        MPI_Bcast((void *)recvBuf.data(), npc, mpi_int_t, pc, grid3d->rscp.comm);
        for (int i = 0; i * Pc + pc < nsupers; i++)
        {
            LvalSendCounts[i * Pc + pc] = recvBuf[i];
        }

        std::copy(localLidxSendCounts.begin(), localLidxSendCounts.end(), recvBuf.begin());
        // send the index count
        MPI_Bcast((void *)recvBuf.data(), npc, mpi_int_t, pc, grid3d->rscp.comm);
        for (int i = 0; i * Pc + pc < nsupers; i++)
        {
            LidxSendCounts[i * Pc + pc] = recvBuf[i];
        }
    }

    maxUvalCount = *std::max_element(UvalSendCounts.begin(), UvalSendCounts.end());
    maxUidxCount = *std::max_element(UidxSendCounts.begin(), UidxSendCounts.end());
    maxLvalCount = *std::max_element(LvalSendCounts.begin(), LvalSendCounts.end());
    maxLidxCount = *std::max_element(LidxSendCounts.begin(), LidxSendCounts.end());
#ifdef SLU_SYM_GPU3D_DEBUG_TRACE
    std::printf("[sym-gpu3d-trace] rank %d: constructor counts nsupers=%lld Pr=%lld Pc=%lld numLA=%d maxLval=%lld maxUval=%lld maxLidx=%lld maxUidx=%lld\n",
                (grid3d != NULL) ? grid3d->iam : -1,
                (long long)nsupers, (long long)Pr, (long long)Pc,
                options->num_lookaheads,
                (long long)maxLvalCount, (long long)maxUvalCount,
                (long long)maxLidxCount, (long long)maxUidxCount);
    std::fflush(stdout);
#endif

    // Allocate bigV, indirect
    nThreads = getNumThreads(iam);
    // bigV = dgetBigV(ldt, nThreads);
    bigV = getBigV<Ftype>(ldt, nThreads);
    if (nThreads < 0 || ldt < 0)
        ABORT("Negative allocation size.");
    size_t indirect_count = xlu_checked_product(static_cast<size_t>(nThreads),
                                                static_cast<size_t>(ldt),
                                                "panel indirect workspace");
    size_t indirect_bytes = xlu_checked_product(indirect_count, sizeof(int_t),
                                                "panel indirect workspace");
    indirect = (int_t *)SUPERLU_MALLOC(indirect_bytes);
    indirectRow = (int_t *)SUPERLU_MALLOC(indirect_bytes);
    indirectCol = (int_t *)SUPERLU_MALLOC(indirect_bytes);
    if (indirect == NULL || indirectRow == NULL || indirectCol == NULL)
        ABORT("Malloc fails for panel indirect workspace.");
    xlu_sym_gpu3d_trace(grid3d, "constructor after indirect workspace allocation");

    // allocating communication buffers
    LvalRecvBufs.resize(options->num_lookaheads);
    UvalRecvBufs.resize(options->num_lookaheads);
    LidxRecvBufs.resize(options->num_lookaheads);
    UidxRecvBufs.resize(options->num_lookaheads);
    // bcastLval.resize(options->num_lookaheads);
    // bcastUval.resize(options->num_lookaheads);
    // bcastLidx.resize(options->num_lookaheads);
    // bcastUidx.resize(options->num_lookaheads);

    for (int i = 0; i < options->num_lookaheads; i++)
    {
        size_t lval_bytes = xlu_checked_alloc_bytes(maxLvalCount, sizeof(Ftype),
                                                    "L value receive buffer");
        size_t uval_bytes = xlu_checked_alloc_bytes(maxUvalCount, sizeof(Ftype),
                                                    "U value receive buffer");
        size_t lidx_bytes = xlu_checked_alloc_bytes(maxLidxCount, sizeof(int_t),
                                                    "L index receive buffer");
        size_t uidx_bytes = xlu_checked_alloc_bytes(maxUidxCount, sizeof(int_t),
                                                    "U index receive buffer");
        LvalRecvBufs[i] = (Ftype *)SUPERLU_MALLOC(lval_bytes);
        UvalRecvBufs[i] = (Ftype *)SUPERLU_MALLOC(uval_bytes);
        LidxRecvBufs[i] = (int_t *)SUPERLU_MALLOC(lidx_bytes);
        UidxRecvBufs[i] = (int_t *)SUPERLU_MALLOC(uidx_bytes);
        if ((lval_bytes != 0 && LvalRecvBufs[i] == NULL) ||
            (uval_bytes != 0 && UvalRecvBufs[i] == NULL) ||
            (lidx_bytes != 0 && LidxRecvBufs[i] == NULL) ||
            (uidx_bytes != 0 && UidxRecvBufs[i] == NULL))
            ABORT("Malloc fails for panel receive buffers.");

        //TODO: check if setup correctly
        #pragma warning disabling bcaststruct 
        #if 0
        bcastStruct bcLval(grid3d->rscp.comm, MPI_DOUBLE, SYNC);
        bcastLval[i] = bcLval;
        bcastStruct bcUval(grid3d->cscp.comm, MPI_DOUBLE, SYNC);
        bcastUval[i] = bcUval;
        bcastStruct bcLidx(grid3d->rscp.comm, mpi_int_t, SYNC);
        bcastLidx[i] = bcLidx;
        bcastStruct bcUidx(grid3d->cscp.comm, mpi_int_t, SYNC);
        bcastUidx[i] = bcUidx;
        #endif
    }
    xlu_sym_gpu3d_trace(grid3d, "constructor after panel receive buffer allocation");

    numDiagBufs = 2*options->num_lookaheads;
    diagFactBufs.resize(numDiagBufs);  /* Sherry?? numDiagBufs == 32 hard-coded */
    // bcastDiagRow.resize(numDiagBufs);
    // bcastDiagCol.resize(numDiagBufs);

    for (int i = 0; i < numDiagBufs; i++) /* Sherry?? these strcutures not used */
    {
        diagFactBufs[i] = (Ftype *)SUPERLU_MALLOC(
            xlu_checked_square_alloc_bytes(ldt, sizeof(Ftype), "diagonal factor buffer"));
        if (diagFactBufs[i] == NULL)
            ABORT("Malloc fails for diagonal factor buffer.");
        // bcastStruct bcDiagRow(grid3d->rscp.comm, MPI_DOUBLE, SYNC);
        // bcastDiagRow[i] = bcDiagRow;
        // bcastStruct bcDiagCol(grid3d->cscp.comm, MPI_DOUBLE, SYNC);
        // bcastDiagCol[i] = bcDiagCol;
    }
    xlu_sym_gpu3d_trace(grid3d, "constructor after diagonal buffer allocation");

    int mxLeafNode = 0;
    int_t *myTreeIdxs = trf3Dpartition->myTreeIdxs;
    // int_t *myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t **sForests = trf3Dpartition->sForests;
    for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode)
            mxLeafNode = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
    }
    //Yang: how is dFBufs being used in the c++ factorization code? Shall we call dinitDiagFactBufsArrMod instead to save memory? 
    dFBufs = initDiagFactBufsArr(numDiagBufs, ldt);
    maxLeafNodes = mxLeafNode;
    xlu_sym_gpu3d_trace(grid3d, "constructor after dFBufs allocation");

    xlu_sym_gpu3d_trace(grid3d, "constructor before initSymFactWorkspace");
    initSymFactWorkspace();
    xlu_sym_gpu3d_trace(grid3d, "constructor after initSymFactWorkspace");
    
    double tGPU = SuperLU_timer_();
    if(superlu_acc_offload)
    {
    #ifdef HAVE_CUDA
        xlu_sym_gpu3d_trace(grid3d, "constructor before setLUstruct_GPU");
        setLUstruct_GPU();  /* Set up LU structure and buffers on GPU */
        xlu_sym_gpu3d_trace(grid3d, "constructor after setLUstruct_GPU");

        // TODO: remove it, checking is very slow
        if(0)
            checkGPU();     
    #endif
    }
        
    tGPU = SuperLU_timer_() -tGPU;
#if ( PRNTlevel >= 1 )    
    printf("Time to intialize GPU DS= %g\n",tGPU );
#endif

    // if (superluAccOffload)

    // for(int pc=0;pc<Pc; pc++)
    // {
    //     MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    //     ...
    // }

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d_in->iam, "Exit xLUstruct_t constructor");
#endif
    
} /* constructor xLUstruct_t */

template <typename Ftype>
int xLUstruct_t<Ftype>::initSymFactWorkspace()
{
    return 0;
}

template <typename Ftype>
int xLUstruct_t<Ftype>::freeSymFactWorkspace()
{
    return 0;
}

template <typename Ftype>
int xLUstruct_t<Ftype>::ensureSymFactWorkSize(int64_t minSize)
{
    return 0;
}

template <>
inline int xLUstruct_t<double>::initSymFactWorkspace()
{
    if (options->SymFact != YES)
        return 0;

    xlu_sym_gpu3d_trace(grid3d, "enter initSymFactWorkspace");
    symGPU3DContract = xlu_gpu3d_contract();
    symContractValidateTol = (symGPU3DContract == 1)
        ? xlu_env_double("GPU3DCONTRACT_VALIDATE_TOL", 1.0e8)
        : 1.0e8;
    symContract1Accepted = 0;
    symContract1Fallbacks = 0;
    symContract1MaxResid = 0.0;

    if (ldt < 0 || maxLvalCount < 0)
        ABORT("Negative SymFact workspace size.");
    size_t diag_work_count = xlu_checked_product(static_cast<size_t>(ldt),
                                                 static_cast<size_t>(ldt),
                                                 "SymFact work");
    int64_t diag_work_size = (int64_t)diag_work_count;
    if (diag_work_size < 0 || (size_t)diag_work_size != diag_work_count)
        ABORT("SymFact workspace size overflows int64_t.");
    int64_t panel_work_size = (int64_t)maxLvalCount;
    int64_t workspace_size = SUPERLU_MAX(diag_work_size, panel_work_size);
    if (workspace_size <= 0)
        workspace_size = 1;

    symFactWorkSize = workspace_size;
    size_t work_count = (size_t)workspace_size;
    if ((int64_t)work_count != workspace_size)
        ABORT("SymFact workspace size overflows allocation size.");
    symFactWork = (double *)SUPERLU_MALLOC(
        xlu_checked_product(work_count, sizeof(double), "SymFact work"));
    if (symFactWork == NULL)
        ABORT("Malloc fails for SymFact work[].");

    size_t ipiv_count = (size_t)ldt;
    if ((int_t)ipiv_count != ldt)
        ABORT("SymFact IPIV size overflows allocation size.");
    symFactIPIV = (int *)SUPERLU_MALLOC(
        xlu_checked_product(ipiv_count, sizeof(int), "SymFact IPIV"));
    if (symFactIPIV == NULL)
        ABORT("Malloc fails for SymFact IPIV[].");

    size_t order_count = xlu_checked_product(2, ipiv_count, "SymFact L2U order");
    symL2UOrders = (int *)SUPERLU_MALLOC(
        xlu_checked_product(order_count, sizeof(int), "SymFact L2U order"));
    if (symL2UOrders == NULL)
        ABORT("Malloc fails for SymFact L2U order workspace.");
    xlu_sym_gpu3d_trace(grid3d, "initSymFactWorkspace after CPU workspace allocation");

#ifdef HAVE_CUDA
    if (superlu_acc_offload)
    {
#ifdef SLU_SYM_GPU3D_DEBUG_TRACE
        std::printf("[sym-gpu3d-trace] rank %d: initSymFactWorkspace GPU setup local_cols=%lld local_rows=%lld Pc=%lld Pr=%lld contract=%d\n",
                    (grid3d != NULL) ? grid3d->iam : -1,
                    (long long)CEILING(nsupers, Pc),
                    (long long)CEILING(nsupers, Pr),
                    (long long)Pc, (long long)Pr, symGPU3DContract);
        std::fflush(stdout);
#endif
        int_t local_cols = CEILING(nsupers, Pc);
        size_t l2u_slots = xlu_checked_product(static_cast<size_t>(local_cols),
                                               static_cast<size_t>(Pc),
                                               "SymFact GPU L2U buffers");
        symL2USendBufsGPU.assign(l2u_slots, NULL);
        symL2USendMapsGPU.assign(l2u_slots, NULL);
        symL2ULocalMapsGPU.assign(CEILING(nsupers, Pr), NULL);
        symPanelReadyEventIds.assign(nsupers, -1);
        symDiagPrefetchEventIds.assign(nsupers, -1);

        dLocalLU_t *Llu = LUstructPtr->Llu;
        if (Pr == 1 && Pc == 1)
        {
            int_t local_rows = CEILING(nsupers, Pr);
            for (int_t lk = 0; lk < local_rows; ++lk)
            {
                int_t k = myrow + lk * Pr;
                if (k >= nsupers || isNodeInMyGrid[k] != 1)
                    continue;

                xupanel_t<double> &upanel = uPanelVec[g2lRow(k)];
                xlpanel_t<double> &lpanel = lPanelVec[g2lCol(k)];
                int_t *usub = Llu->Ufstnz_br_ptr[lk];
                if (upanel.isEmpty() || lpanel.isEmpty() || usub == NULL)
                    continue;

                std::vector<int_t> local_map(upanel.nzvalSize(), -1);
                int_t ksupc = SuperSize(k);
                int_t klst = FstBlockC(k + 1);
                int_t usub_ptr = BR_HEADER;
                int_t dst_col = 0;
                int_t nub = usub[0];

                for (int_t ub = 0; ub < nub; ++ub)
                {
                    int_t jb = usub[usub_ptr];
                    int_t gsupc = SuperSize(jb);
                    int_t lblock = lpanel.find(jb);
                    if (lblock == GLOBAL_BLOCK_NOT_FOUND)
                        ABORT("SymFact local GPU L2U map cannot find an L block.");

                    int_t *lrows = lpanel.rowList(lblock);
                    int_t n_lrows = lpanel.nbrow(lblock);
                    for (int_t col = 0; col < gsupc; ++col)
                    {
                        int_t segsize = klst - usub[usub_ptr + UB_DESCRIPTOR + col];
                        if (segsize <= 0)
                            continue;

                        int_t src_row = GLOBAL_BLOCK_NOT_FOUND;
                        for (int_t rr = 0; rr < n_lrows; ++rr)
                        {
                            if (lrows[rr] == col)
                            {
                                src_row = rr;
                                break;
                            }
                        }
                        if (src_row == GLOBAL_BLOCK_NOT_FOUND)
                            ABORT("SymFact local GPU L2U map cannot find an L row.");

                        for (int_t row = 0; row < ksupc; ++row)
                        {
                            int_t dst = dst_col * upanel.LDA() + row;
                            if (row >= ksupc - segsize)
                                local_map[dst] = lpanel.blkPtrOffset(lblock) +
                                                 src_row + row * lpanel.LDA();
                        }
                        ++dst_col;
                    }
                    usub_ptr += UB_DESCRIPTOR + gsupc;
                }

                if (dst_col != upanel.nzcols())
                    ABORT("SymFact local GPU L2U map has an invalid U column count.");

                gpuErrchk(cudaMalloc((void **)&symL2ULocalMapsGPU[lk],
                                     xlu_checked_product(static_cast<size_t>(upanel.nzvalSize()),
                                                         sizeof(int_t),
                                                         "SymFact local GPU L2U map")));
                gpuErrchk(cudaMemcpy(symL2ULocalMapsGPU[lk], local_map.data(),
                                     sizeof(int_t) * static_cast<size_t>(upanel.nzvalSize()),
                                     cudaMemcpyHostToDevice));
            }
            xlu_sym_gpu3d_trace(grid3d, "initSymFactWorkspace after local GPU L2U map setup");
        }

        for (int_t lk = 0; lk < local_cols; ++lk)
        {
            if (Llu->Send_CommL == NULL || Llu->Send_CommL[lk].ComQuant == NULL)
                continue;

            int_t *lsub = Llu->Lrowind_bc_ptr[lk];
            int_t *lloc = Llu->Lindval_loc_bc_ptr[lk];
            if (lsub == NULL || lloc == NULL || lsub[0] <= 0)
                continue;

            int_t jb = mycol + lk * Pc;
            if (jb >= nsupers)
                continue;

            int_t nb;
            int_t idx_i;
            int_t idx_v;
            if (myrow == krow(jb))
            {
                nb = lsub[0] - 1;
                idx_i = nb + 2;
                idx_v = 2 * nb + 3;
            }
            else
            {
                nb = lsub[0];
                idx_i = nb;
                idx_v = 2 * nb;
            }
            if (nb <= 0)
                continue;

            int_t knsupc = SuperSize(jb);
            int_t nsupr = lsub[1];
            std::vector<std::vector<int_t> > host_maps(Pc);
            for (int pc = 0; pc < Pc; ++pc)
            {
                int size = Llu->Send_CommL[lk].ComQuant[pc].size;
                if (size > 0)
                    host_maps[pc].reserve(size);
            }

            for (int_t lb = 0; lb < nb; ++lb)
            {
                int_t luptr_tmp = lloc[lb + idx_v];
                int_t lptr_tmp = lloc[lb + idx_i];
                int_t ik = lsub[lptr_tmp];
                int ikcol = PCOL(ik, grid);
                int_t len = lsub[lptr_tmp + 1];
                int_t fsupc = FstBlockC(ik);

                std::vector<std::pair<int_t, int_t> > row_order;
                row_order.reserve(len);
                for (int_t i = 0; i < len; ++i)
                    row_order.push_back(std::make_pair(lsub[lptr_tmp + 2 + i] - fsupc, i));
                std::sort(row_order.begin(), row_order.end());

                std::vector<int_t> &map = host_maps[ikcol];
                map.push_back(-(ik + 1));
                for (int_t i = 0; i < len; ++i)
                {
                    int_t src_row = row_order[i].second;
                    for (int_t j = 0; j < knsupc; ++j)
                        map.push_back(luptr_tmp + src_row + j * nsupr);
                }
            }

            for (int pc = 0; pc < Pc; ++pc)
            {
                int size = Llu->Send_CommL[lk].ComQuant[pc].size;
                if (size <= 0)
                    continue;
                if (host_maps[pc].size() != static_cast<size_t>(size))
                    ABORT("SymFact GPU L2U send map size mismatch.");

                size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                              static_cast<size_t>(pc);
                gpuErrchk(cudaMalloc((void **)&symL2USendBufsGPU[flat],
                                     xlu_checked_product(static_cast<size_t>(size),
                                                         sizeof(double),
                                                         "SymFact GPU L2U send buffer")));
                gpuErrchk(cudaMalloc((void **)&symL2USendMapsGPU[flat],
                                     xlu_checked_product(static_cast<size_t>(size),
                                                         sizeof(int_t),
                                                         "SymFact GPU L2U send map")));
                gpuErrchk(cudaMemcpy(symL2USendMapsGPU[flat], host_maps[pc].data(),
                                     sizeof(int_t) * static_cast<size_t>(size),
                                     cudaMemcpyHostToDevice));
            }
        }
        xlu_sym_gpu3d_trace(grid3d, "initSymFactWorkspace after send GPU L2U map setup");
    }
#endif

    xlu_sym_gpu3d_trace(grid3d, "exit initSymFactWorkspace");
    return 0;
}

template <>
inline int xLUstruct_t<double>::freeSymFactWorkspace()
{
    if (symL2UOrders != NULL)
    {
        SUPERLU_FREE(symL2UOrders);
        symL2UOrders = NULL;
    }

    if (symFactWork != NULL)
    {
        SUPERLU_FREE(symFactWork);
        symFactWork = NULL;
    }

    if (symFactIPIV != NULL)
    {
        SUPERLU_FREE(symFactIPIV);
        symFactIPIV = NULL;
    }

#ifdef HAVE_CUDA
    for (size_t i = 0; i < symL2USendBufsGPU.size(); ++i)
        if (symL2USendBufsGPU[i] != NULL)
            gpuErrchk(cudaFree(symL2USendBufsGPU[i]));
    for (size_t i = 0; i < symL2USendMapsGPU.size(); ++i)
        if (symL2USendMapsGPU[i] != NULL)
            gpuErrchk(cudaFree(symL2USendMapsGPU[i]));
    for (size_t i = 0; i < symL2ULocalMapsGPU.size(); ++i)
        if (symL2ULocalMapsGPU[i] != NULL)
            gpuErrchk(cudaFree(symL2ULocalMapsGPU[i]));
    symL2USendBufsGPU.clear();
    symL2USendMapsGPU.clear();
    symL2ULocalMapsGPU.clear();
    symPanelReadyEventIds.clear();
    for (size_t i = 0; i < symDiagPrefetchBufs.size(); ++i)
        if (symDiagPrefetchBufs[i] != NULL)
            gpuErrchk(cudaFreeHost(symDiagPrefetchBufs[i]));
    for (size_t i = 0; i < symDiagPrefetchDoneEvents.size(); ++i)
        gpuErrchk(cudaEventDestroy(symDiagPrefetchDoneEvents[i]));
    symDiagPrefetchBufs.clear();
    symDiagPrefetchDoneEvents.clear();
    symDiagPrefetchEventIds.clear();
    symDiagPrefetchNodes.clear();
#endif

    symFactWorkSize = 0;
    return 0;
}

template <>
inline int xLUstruct_t<double>::ensureSymFactWorkSize(int64_t minSize)
{
    if (minSize <= symFactWorkSize)
        return 0;

    if (symFactWork != NULL)
        SUPERLU_FREE(symFactWork);

    symFactWorkSize = minSize;
    size_t work_count = (size_t)symFactWorkSize;
    if ((int64_t)work_count != symFactWorkSize)
        ABORT("SymFact workspace size overflows allocation size.");
    symFactWork = (double *)SUPERLU_MALLOC(
        xlu_checked_product(work_count, sizeof(double), "SymFact work"));
    if (symFactWork == NULL)
        ABORT("Malloc fails for SymFact work[].");

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurComplementUpdate(
    int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

#pragma omp parallel for
    for (size_t ij = 0; ij < (nlb - st_lb) * nub; ij++)
    {
        int_t ii = ij / nub + st_lb;
        int_t jj = ij % nub;
        blockUpdate(k, ii, jj, lpanel, upanel);
    }

    return 0;
}

// should be called from an openMP region
template <typename Ftype>
int_t *xLUstruct_t<Ftype>::computeIndirectMap(indirectMapType direction, int_t srcLen, int_t *srcVec,
                                         int_t dstLen, int_t *dstVec)
{
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        return srcVec;
    }
    int_t thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif
    int_t *dstIdx = indirect + thread_id * ldt;
    for (int_t i = 0; i < dstLen; i++)
    {
        // if(thread_id < dstLen)
        dstIdx[dstVec[i]] = i;
    }

    int_t *RCmap = (direction == ROW_MAP) ? indirectRow : indirectCol;
    RCmap += thread_id * ldt;

    for (int_t i = 0; i < srcLen; i++)
    {
        RCmap[i] = dstIdx[srcVec[i]];
    }

    return RCmap;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dScatter(int_t m, int_t n,
                              int_t gi, int_t gj,
                              Ftype *Src, int_t ldsrc,
                              int_t *srcRowList, int_t *srcColList)
{

    Ftype *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    if (gj > gi) // its in upanel
    {
        int li = g2lRow(gi);
        int lj = uPanelVec[li].find(gj);
        Dst = uPanelVec[li].blkPtr(lj);
        lddst = supersize(gi);
        dstRowLen = supersize(gi);
        dstRowList = NULL;
        dstColLen = uPanelVec[li].nbcol(lj);
        dstColList = uPanelVec[li].colList(lj);
        // std::cout<<li<<" "<<lj<<" Dst[0] is"<<Dst[0] << "\n";
    }
    else
    {
        int lj = g2lCol(gj);
        int li = lPanelVec[lj].find(gi);
        Dst = lPanelVec[lj].blkPtr(li);
        lddst = lPanelVec[lj].LDA();
        dstRowLen = lPanelVec[lj].nbrow(li);
        dstRowList = lPanelVec[lj].rowList(li);
        dstColLen = supersize(gj);
        dstColList = NULL;
    }

    // compute source row to dest row mapping
    int_t *rowS2D = computeIndirectMap(ROW_MAP, m, srcRowList,
                                       dstRowLen, dstRowList);

    // compute source col to dest col mapping
    int_t *colS2D = computeIndirectMap(COL_MAP, n, srcColList,
                                       dstColLen, dstColList);

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::packedU2skyline(LUStruct_type<Ftype> *LUstruct)
{

    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    Ftype **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            int_t globalId = i * Pr + myrow;
            uPanelVec[i].packed2skyline(globalId, Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
        }
    }

    return 0;
}

int numProcsPerNode(MPI_Comm baseCommunicator);
// int numProcsPerNode(MPI_Comm baseCommunicator)
// {
//     MPI_Comm sharedComm;
//     MPI_Comm_split_type(baseCommunicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedComm);
//     int count = 0;
//     MPI_Comm_size(sharedComm, &count);
//     return count;
// }


template <typename Ftype>
int_t xLUstruct_t<Ftype>::lookAheadUpdate(
    int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t laILoc = lpanel.find(laIdx);
    int_t nub = upanel.nblocks();
    int_t laJLoc = upanel.find(laIdx);

#pragma omp parallel
    {
        /*Next lpanelUpdate*/
#pragma omp for nowait
        for (size_t ii = st_lb; ii < nlb; ii++)
        {
            int_t jj = laJLoc;
            if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
                blockUpdate(k, ii, jj, lpanel, upanel);
        }

        /*Next upanelUpdate*/
#pragma omp for nowait
        for (size_t jj = 0; jj < nub; jj++)
        {
            int_t ii = laILoc;
            if (laILoc != GLOBAL_BLOCK_NOT_FOUND && jj != laJLoc)
                blockUpdate(k, ii, jj, lpanel, upanel);
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::blockUpdate(int_t k,
                                 int_t ii, int_t jj, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    int thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif

    Ftype *V = bigV + thread_id * ldt * ldt;

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    superlu_gemm<Ftype>("N", "N",
                  lpanel.nbrow(ii), upanel.nbcol(jj), supersize(k), alpha,
                  lpanel.blkPtr(ii), lpanel.LDA(),
                  upanel.blkPtr(jj), upanel.LDA(), beta,
                  V, lpanel.nbrow(ii));

    // now do the scatter
    int_t ib = lpanel.gid(ii);
    int_t jb = upanel.gid(jj);

    dScatter(lpanel.nbrow(ii), upanel.nbcol(jj),
             ib, jb, V, lpanel.nbrow(ii),
             lpanel.rowList(ii), upanel.colList(jj));
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurCompUpdateExcludeOne(
    int_t k, int_t ex, // suypernodes to be excluded
    xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = upanel.find(ex);

#pragma omp parallel for
    for (size_t ij = 0; ij < (nlb - st_lb) * nub; ij++)
    {
        int_t ii = ij / nub + st_lb;
        int_t jj = ij % nub;

        if (ii != exILoc && jj != exJLoc)
            blockUpdate(k, ii, jj, lpanel, upanel);
    }
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dDiagFactorPanelSolve(int_t k, int_t offset, diagFactBufs_type<Ftype>**dFBufs)
{
    if (options->SymFact == YES)
        return dSymDiagFactorPanelSolve(k, offset, offset, dFBufs);


    int_t ksupc = SuperSize(k);
    /*=======   Diagonal Factorization      ======*/
    if (iam == procIJ(k, k))
    {
        lPanelVec[g2lCol(k)].diagFactor(k, dFBufs[offset]->BlockUFactor, ksupc,
                                        thresh, xsup, options, stat, info);
        lPanelVec[g2lCol(k)].packDiagBlock(dFBufs[offset]->BlockLFactor, ksupc);
    }

    /*=======   Diagonal Broadcast          ======*/
    if (myrow == krow(k))
        MPI_Bcast((void *)dFBufs[offset]->BlockLFactor, ksupc * ksupc,
                  MPI_DOUBLE, kcol(k), (grid->rscp).comm);
    if (mycol == kcol(k))
        MPI_Bcast((void *)dFBufs[offset]->BlockUFactor, ksupc * ksupc,
                  MPI_DOUBLE, krow(k), (grid->cscp).comm);

    /*=======   Panel Update                ======*/
    if (myrow == krow(k))
        uPanelVec[g2lRow(k)].panelSolve(ksupc, dFBufs[offset]->BlockLFactor, ksupc);

    if (mycol == kcol(k))
        lPanelVec[g2lCol(k)].panelSolve(ksupc, dFBufs[offset]->BlockUFactor, ksupc);

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymDiagFactorPanelSolve(int_t k, int_t handle_offset,
                                                   int_t buffer_offset,
                                                   diagFactBufs_type<Ftype> **dFBufs)
{
    ABORT("LUv1 SymFact is implemented for double precision only.");
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymStartL2U(int_t k, int_t stream_offset)
{
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymFinishL2U(int_t k)
{
    return 0;
}

#ifdef HAVE_CUDA
template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymStartDiagPrefetch(int_t k, int_t stream_offset)
{
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymStartDiagPrefetch(int_t k,
                                                        int_t stream_offset)
{
    if (options->SymFact != YES || !superlu_acc_offload)
        return 0;
    if (!(Pr == 1 && Pc == 1 &&
          grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1))
        return 0;
    if (k < 0 || k >= nsupers || iam != procIJ(k, k))
        return 0;
    if (stream_offset < 0 ||
        stream_offset >= static_cast<int_t>(symDiagPrefetchBufs.size()) ||
        stream_offset >= static_cast<int_t>(symDiagPrefetchDoneEvents.size()) ||
        stream_offset >= static_cast<int_t>(symDiagPrefetchNodes.size()))
        return 0;
    if (symDiagPrefetchBufs[stream_offset] == NULL)
        return 0;
    if (static_cast<size_t>(k) >= symDiagPrefetchEventIds.size())
        return 0;
    if (symDiagPrefetchEventIds[k] >= 0)
        return 0;
    if (symDiagPrefetchNodes[stream_offset] != -1 &&
        symDiagPrefetchNodes[stream_offset] != k)
        return 0;

    xlpanel_t<double> &lpanel = lPanelVec[g2lCol(k)];
    if (lpanel.isEmpty() || !lpanel.haveDiag())
        return 0;

    int_t ksupc = SuperSize(k);
    cudaStream_t stream = A_gpu.lookAheadLStream[stream_offset];
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double sym_prefetch_t = SuperLU_timer_();
#endif
    gpuErrchk(cudaMemcpy2DAsync(symDiagPrefetchBufs[stream_offset],
                                ldt * sizeof(double),
                                lpanel.blkPtrGPU(0),
                                lpanel.LDA() * sizeof(double),
                                ksupc * sizeof(double), ksupc,
                                cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaEventRecord(symDiagPrefetchDoneEvents[stream_offset],
                              stream));
    symDiagPrefetchNodes[stream_offset] = k;
    symDiagPrefetchEventIds[k] = stream_offset;
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symStatAdd(SYM_GPU3D_S_DIAG_PREFETCH_ISSUES);
    symStatAdd(SYM_GPU3D_S_DIAG_D2H_BYTES,
               static_cast<long long>(ksupc) * static_cast<long long>(ksupc) *
               static_cast<long long>(sizeof(double)));
    symTimingAdd(SYM_GPU3D_T_DIAG_PREFETCH_ISSUE,
                 SuperLU_timer_() - sym_prefetch_t);
#endif
    return 0;
}
#endif

template <>
inline int_t xLUstruct_t<double>::dSymStartL2U(int_t k, int_t stream_offset)
{
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    SymTimingScope sym_timer(this, SYM_GPU3D_T_L2U_START);
#endif
    if (options->SymFact != YES || options->CommL != YES)
    {
        if (options->SymFact == YES)
            ABORT("LUv1 SymFact requires CommL=YES to reconstruct U panels.");
        return 0;
    }

    if (symL2UOrders == NULL)
        ABORT("SymFact L2U order workspace is not allocated.");
    if (symFactTagUb <= 0)
        ABORT("Invalid MPI tag upper bound for LUv1 SymFact L2U communication.");

#ifdef HAVE_CUDA
    if (superlu_acc_offload)
        return dSymStartL2UGPU(k, stream_offset);
#else
    (void)stream_offset;
#endif

    dStartL2U_comm(k, grid, options, LUstructPtr, stat, info, SCT, symFactTagUb,
                   symL2UOrders, ldt);
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymFinishL2U(int_t k)
{
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    SymTimingScope sym_timer(this, SYM_GPU3D_T_L2U_FINISH);
#endif
    if (options->SymFact != YES || options->CommL != YES)
    {
        if (options->SymFact == YES)
            ABORT("LUv1 SymFact requires CommL=YES to reconstruct U panels.");
        return 0;
    }

#ifdef HAVE_CUDA
    if (superlu_acc_offload && Pr == 1 && Pc == 1)
        return 0;
#endif

    dLocalLU_t *Llu = LUstructPtr->Llu;
    dWaitL2U_recv(k, grid, options, LUstructPtr, stat, SCT);

    if (myrow == krow(k))
    {
        int_t lk = LBi(k, grid);
        if (Llu->Ufstnz_br_ptr[lk] != NULL && Llu->Unzval_br_ptr[lk] != NULL)
        {
            uPanelVec[g2lRow(k)].loadFromSkyline(k, Llu->Ufstnz_br_ptr[lk],
                                                 Llu->Unzval_br_ptr[lk], xsup);
#ifdef HAVE_CUDA
            if (superlu_acc_offload)
                uPanelVec[g2lRow(k)].copyBackToGPU();
#endif
        }
    }

    dWaitL2U_send(k, grid, options, LUstructPtr, stat, info, SCT, symFactTagUb);
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymDiagFactorPanelSolve(int_t k, int_t handle_offset,
                                                           int_t buffer_offset,
                                                           ddiagFactBufs_t **dFBufs)
{
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symStatAdd(SYM_GPU3D_S_FACTOR_NODES);
#endif
    dSymStartL2U(k, handle_offset);

    int_t ksupc = SuperSize(k);
    double *invDiag = dFBufs[buffer_offset]->BlockUFactor;
    double *origDiag = dFBufs[buffer_offset]->BlockLFactor;
    int contract = symGPU3DContract;
    bool invDiagOnDevice = false;
    if (contract == 3)
        ABORT("GPU3DCONTRACT=3 is reserved for later SymFact diagonal experiments.");

#ifndef SLU_HAVE_LAPACK
    ABORT("LUv1 SymFact requires LAPACK dsytrf/dsytri support.");
#else
    if (iam == procIJ(k, k))
    {
        xlpanel_t<double> &lpanel = lPanelVec[g2lCol(k)];
        double *diag = lpanel.blkPtr(0);
        int_t panelLdd = lpanel.LDA();
        int_t ldd = panelLdd;
#ifdef HAVE_CUDA
        if (superlu_acc_offload)
        {
            bool prefetched = false;
            if (Pr == 1 && Pc == 1 &&
                grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1 &&
                static_cast<size_t>(k) < symDiagPrefetchEventIds.size())
            {
                int event_id = symDiagPrefetchEventIds[k];
                if (event_id >= 0 &&
                    event_id < static_cast<int>(symDiagPrefetchBufs.size()) &&
                    event_id < static_cast<int>(symDiagPrefetchDoneEvents.size()) &&
                    event_id < static_cast<int>(symDiagPrefetchNodes.size()) &&
                    symDiagPrefetchNodes[event_id] == k)
                {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double sym_prefetch_wait_t = SuperLU_timer_();
#endif
                    gpuErrchk(cudaEventSynchronize(symDiagPrefetchDoneEvents[event_id]));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAdd(SYM_GPU3D_T_DIAG_PREFETCH_WAIT,
                                 SuperLU_timer_() - sym_prefetch_wait_t);
                    symStatAdd(SYM_GPU3D_S_DIAG_PREFETCH_HITS);
#endif
                    diag = symDiagPrefetchBufs[event_id];
                    ldd = ldt;
                    symDiagPrefetchEventIds[k] = -1;
                    symDiagPrefetchNodes[event_id] = -1;
                    prefetched = true;
                }
            }
            if (!prefetched)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symStatAdd(SYM_GPU3D_S_DIAG_PREFETCH_MISSES);
                symStatAdd(SYM_GPU3D_S_DIAG_D2H_BYTES,
                           static_cast<long long>(ksupc) * static_cast<long long>(ksupc) *
                           static_cast<long long>(sizeof(double)));
                double sym_t = SuperLU_timer_();
                cudaStream_t cuStream = A_gpu.cuStreams[handle_offset];
                cudaEvent_t start_event = A_gpu.diagD2HStartEvents[handle_offset];
                cudaEvent_t end_event = A_gpu.diagD2HEndEvents[handle_offset];
                gpuErrchk(cudaEventRecord(start_event, cuStream));
                gpuErrchk(cudaMemcpy2DAsync(diag, ldd * sizeof(double),
                                            lpanel.blkPtrGPU(0),
                                            panelLdd * sizeof(double),
                                            ksupc * sizeof(double), ksupc,
                                            cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaEventRecord(end_event, cuStream));
                gpuErrchk(cudaEventSynchronize(end_event));
                float copy_ms = 0.0f;
                gpuErrchk(cudaEventElapsedTime(&copy_ms, start_event, end_event));
                double total = SuperLU_timer_() - sym_t;
                double copy = (double)copy_ms * 1.0e-3;
                symTimingAdd(SYM_GPU3D_T_DIAG_D2H, total);
                symTimingAdd(SYM_GPU3D_T_DIAG_D2H_COPY, copy);
                symTimingAdd(SYM_GPU3D_T_DIAG_D2H_WAIT, SUPERLU_MAX(0.0, total - copy));
#else
                gpuErrchk(cudaMemcpy2D(diag, ldd * sizeof(double),
                                       lpanel.blkPtrGPU(0),
                                       panelLdd * sizeof(double),
                                       ksupc * sizeof(double), ksupc,
                                       cudaMemcpyDeviceToHost));
#endif
            }
        }
#endif
        int nsupc_i = (int)ksupc;
        int ldd_i = (int)ldd;
        int panel_ldd_i = (int)panelLdd;
        int_t jfst = FstBlockC(k);
        int lwork = -1;
        int lapack_info = 0;
        char uplo = 'L';
        double *work = symFactWork;
        int *ipiv = symFactIPIV;
        bool diagonal_done = false;

        if (work == NULL || ipiv == NULL)
            ABORT("LUv1 SymFact workspace is not allocated.");

        if (contract == 2)
            for (int_t j = 0; j < ksupc; ++j)
                memcpy(&origDiag[j * ksupc], &diag[j * ldd],
                       ksupc * sizeof(double));

        double thresh1 = thresh / 10.0;

#ifdef HAVE_CUDA
        if (contract == 1)
        {
            if (!superlu_acc_offload)
                ABORT("GPU3DCONTRACT=1 requires GPU offload.");

            cusolverDnHandle_t cusolverH = A_gpu.cuSolveHandles[handle_offset];
            cudaStream_t cuStream = A_gpu.cuStreams[handle_offset];
            double *dDiag = lpanel.blkPtrGPU(0);
            double *dWork = A_gpu.diagFactWork[handle_offset];
            int *dIpiv = A_gpu.diagFactIPIV[handle_offset];
            int *dInfo = A_gpu.diagFactInfo[handle_offset];
            int sytrf_lwork = 0;
            int sytri_lwork = 0;
            int gpu_info = 0;

            gpuCusolverErrchk(cusolverDnSetStream(cusolverH, cuStream));
            gpuCusolverErrchk(cusolverDnDsytrf_bufferSize(cusolverH, nsupc_i,
                                                          dDiag, panel_ldd_i,
                                                          &sytrf_lwork));
            gpuCusolverErrchk(cusolverDnDsytrf(cusolverH, CUBLAS_FILL_MODE_LOWER,
                                               nsupc_i, dDiag, panel_ldd_i, dIpiv,
                                               dWork, sytrf_lwork, dInfo));
            gpuErrchk(cudaMemcpyAsync(&gpu_info, dInfo, sizeof(int),
                                      cudaMemcpyDeviceToHost, cuStream));
            gpuErrchk(cudaStreamSynchronize(cuStream));

            if (gpu_info == 0)
            {
                gpuErrchk(cudaMemcpy2DAsync(work, ksupc * sizeof(double),
                                            dDiag, panelLdd * sizeof(double),
                                            ksupc * sizeof(double), ksupc,
                                            cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaMemcpyAsync(ipiv, dIpiv,
                                          ksupc * sizeof(int),
                                          cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaStreamSynchronize(cuStream));

                const double tol_inertia = 1e-30;
                int inertia[3];
                inertia_from_dsytrf(uplo, nsupc_i, work, nsupc_i, ipiv,
                                    tol_inertia, inertia);
                int n2x2 = xlu_sytrf_count_2x2(ipiv, nsupc_i);

                gpuCusolverErrchk(cusolverDnDsytri_bufferSize(cusolverH,
                                                              CUBLAS_FILL_MODE_LOWER,
                                                              nsupc_i, dDiag,
                                                              panel_ldd_i, dIpiv,
                                                              &sytri_lwork));
                gpuCusolverErrchk(cusolverDnDsytri(cusolverH, CUBLAS_FILL_MODE_LOWER,
                                                   nsupc_i, dDiag, panel_ldd_i, dIpiv,
                                                   dWork, sytri_lwork, dInfo));
                gpuErrchk(cudaMemcpyAsync(&gpu_info, dInfo, sizeof(int),
                                          cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaStreamSynchronize(cuStream));

                if (gpu_info == 0)
                {
                    gpuErrchk(cudaMemcpy2DAsync(work, ksupc * sizeof(double),
                                                dDiag, panelLdd * sizeof(double),
                                                ksupc * sizeof(double), ksupc,
                                                cudaMemcpyDeviceToHost, cuStream));
                    gpuErrchk(cudaStreamSynchronize(cuStream));

                    for (int_t j = 0; j < ksupc; ++j)
                        for (int_t i = j + 1; i < ksupc; ++i)
                            work[j + i * ksupc] = work[i + j * ksupc];

                    double scaled_resid =
                        xlu_sym_inverse_scaled_residual(diag, ldd, work, nsupc_i);
                    if (scaled_resid <= symContractValidateTol)
                    {
                        for (int_t j = 0; j < ksupc; ++j)
                            memcpy(&invDiag[j * ksupc], &work[j * ksupc],
                                   ksupc * sizeof(double));
                        for (int_t j = 0; j < ksupc; ++j)
                            memcpy(&diag[j * ldd], &work[j * ksupc],
                                   ksupc * sizeof(double));

                        stat->sytrf_2x2 += n2x2;
                        stat->inertia[0] += inertia[0];
                        stat->inertia[1] += inertia[1];
                        stat->inertia[2] += inertia[2];
                        stat->ops[FACT] += (flops_t)ksupc * ksupc * ksupc;
                        *info = 0;
                        diagonal_done = true;
                        ++symContract1Accepted;
                        symContract1MaxResid =
                            SUPERLU_MAX(symContract1MaxResid, scaled_resid);
                    }
                }
            }

            if (!diagonal_done)
                ++symContract1Fallbacks;
        }
#else
        if (contract == 1)
            ABORT("GPU3DCONTRACT=1 requires CUDA support.");
#endif

        if (!diagonal_done)
        {
            int ntiny = 0;
            int n2x2 = 0;
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double sym_sytrf_t = SuperLU_timer_();
#endif
            if (options->ReplaceTinyPivot == YES)
            {
                dsytrf_mod_(&uplo, &nsupc_i, diag, &ldd_i, &thresh1, ipiv,
                             work, &lwork, info, &ntiny, &n2x2);
                int64_t requested_work_size = (int64_t)work[0];
                ensureSymFactWorkSize(requested_work_size);
                work = symFactWork;
                lwork = (int)requested_work_size;
                dsytrf_mod_(&uplo, &nsupc_i, diag, &ldd_i, &thresh1, ipiv,
                             work, &lwork, info, &ntiny, &n2x2);
                stat->TinyPivots += ntiny;
                stat->sytrf_2x2 += n2x2;
            }
            else
            {
                dsytrf_(&uplo, &nsupc_i, diag, &ldd_i, ipiv, work, &lwork, info);
                int64_t requested_work_size = (int64_t)work[0];
                ensureSymFactWorkSize(requested_work_size);
                work = symFactWork;
                lwork = (int)requested_work_size;
                dsytrf_(&uplo, &nsupc_i, diag, &ldd_i, ipiv, work, &lwork, info);
                n2x2 = xlu_sytrf_count_2x2(ipiv, nsupc_i);
            }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAdd(SYM_GPU3D_T_CPU_SYTRF, SuperLU_timer_() - sym_sytrf_t);
#endif

            if (*info > 0)
                *info += jfst;
            else if (*info < 0)
                *info -= jfst;

            const double tol_inertia = 1e-30;
            int inertia[3];
            inertia_from_dsytrf(uplo, nsupc_i, diag, ldd_i, ipiv,
                                tol_inertia, inertia);
            stat->inertia[0] += inertia[0];
            stat->inertia[1] += inertia[1];
            stat->inertia[2] += inertia[2];

            bool inverse_done = false;
#ifdef HAVE_CUDA
            if (contract == 2 && ntiny == 0 && n2x2 == 0)
            {
                if (!superlu_acc_offload)
                    ABORT("GPU3DCONTRACT=2 requires GPU offload.");
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double sym_gpu_sytri_t = SuperLU_timer_();
#endif

                cusolverDnHandle_t cusolverH = A_gpu.cuSolveHandles[handle_offset];
                cudaStream_t cuStream = A_gpu.cuStreams[handle_offset];
                double *dDiag = lpanel.blkPtrGPU(0);
                double *dWork = A_gpu.diagFactWork[handle_offset];
                int *dIpiv = A_gpu.diagFactIPIV[handle_offset];
                int *dInfo = A_gpu.diagFactInfo[handle_offset];
                int sytri_lwork = 0;
                int gpu_info = 0;

                gpuCusolverErrchk(cusolverDnSetStream(cusolverH, cuStream));
                gpuErrchk(cudaMemcpy2DAsync(dDiag, panelLdd * sizeof(double),
                                            diag, ldd * sizeof(double),
                                            ksupc * sizeof(double), ksupc,
                                            cudaMemcpyHostToDevice, cuStream));
                gpuErrchk(cudaMemcpyAsync(dIpiv, ipiv,
                                          ksupc * sizeof(int),
                                          cudaMemcpyHostToDevice, cuStream));
                gpuCusolverErrchk(cusolverDnDsytri_bufferSize(cusolverH,
                                                              CUBLAS_FILL_MODE_LOWER,
                                                              nsupc_i, dDiag,
                                                              panel_ldd_i, dIpiv,
                                                              &sytri_lwork));
                gpuCusolverErrchk(cusolverDnDsytri(cusolverH, CUBLAS_FILL_MODE_LOWER,
                                                   nsupc_i, dDiag, panel_ldd_i, dIpiv,
                                                   dWork, sytri_lwork, dInfo));
                gpuErrchk(cudaMemcpyAsync(&gpu_info, dInfo, sizeof(int),
                                          cudaMemcpyDeviceToHost, cuStream));
                gpuErrchk(cudaStreamSynchronize(cuStream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_GPU_SYTRI, SuperLU_timer_() - sym_gpu_sytri_t);
#endif

                if (gpu_info == 0)
                {
                    gpuErrchk(cudaMemcpy2DAsync(work, ksupc * sizeof(double),
                                                dDiag, panelLdd * sizeof(double),
                                                ksupc * sizeof(double), ksupc,
                                                cudaMemcpyDeviceToHost, cuStream));
                    gpuErrchk(cudaStreamSynchronize(cuStream));

                    for (int_t j = 0; j < ksupc; ++j)
                        for (int_t i = j + 1; i < ksupc; ++i)
                            work[j + i * ksupc] = work[i + j * ksupc];

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    double sym_validate_t = SuperLU_timer_();
#endif
                    double scaled_resid =
                        xlu_sym_inverse_scaled_residual(origDiag, nsupc_i,
                                                        work, nsupc_i);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                    symTimingAdd(SYM_GPU3D_T_GPU_SYTRI_VALIDATE,
                                 SuperLU_timer_() - sym_validate_t);
#endif
                    if (scaled_resid <= symContractValidateTol)
                    {
                        for (int_t j = 0; j < ksupc; ++j)
                            memcpy(&diag[j * ldd], &work[j * ksupc],
                                   ksupc * sizeof(double));
                        inverse_done = true;
                    }
                }
            }
#else
            if (contract == 2)
                ABORT("GPU3DCONTRACT=2 requires CUDA support.");
#endif

            if (!inverse_done)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double sym_cpu_sytri_t = SuperLU_timer_();
#endif
                dsytri_(&uplo, &nsupc_i, diag, &ldd_i, ipiv, work, &lapack_info);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_CPU_SYTRI, SuperLU_timer_() - sym_cpu_sytri_t);
#endif
            }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double sym_pack_t = SuperLU_timer_();
#endif
            for (int_t j = 0; j < ksupc; ++j)
                for (int_t i = j + 1; i < ksupc; ++i)
                    diag[j + i * ldd] = diag[i + j * ldd];

            for (int_t j = 0; j < ksupc; ++j)
                memcpy(&invDiag[j * ksupc], &diag[j * ldd], ksupc * sizeof(double));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAdd(SYM_GPU3D_T_DIAG_PACK, SuperLU_timer_() - sym_pack_t);
#endif

            stat->ops[FACT] += (flops_t)ksupc * ksupc * ksupc;
        }
    }

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double sym_bcast_t = SuperLU_timer_();
#endif
#ifdef HAVE_CUDA
    if (superlu_acc_offload && mycol == kcol(k) && superlu_cuda_aware_mpi())
    {
        int stream_id = handle_offset;
        if (stream_id < 0 || stream_id >= A_gpu.numCudaStreams)
            stream_id = 0;
        cudaStream_t cuStream = A_gpu.cuStreams[stream_id];
        double *dInvDiag = A_gpu.dFBufs[buffer_offset];

        if (iam == procIJ(k, k))
        {
            gpuErrchk(cudaMemcpyAsync(dInvDiag, invDiag,
                                      ksupc * ksupc * sizeof(double),
                                      cudaMemcpyHostToDevice, cuStream));
            gpuErrchk(cudaStreamSynchronize(cuStream));
        }
        superlu_gpu_mpi_bcast(dInvDiag, invDiag, sizeof(double),
                              static_cast<int>(ksupc * ksupc), MPI_DOUBLE,
                              krow(k), (grid->cscp).comm);
        invDiagOnDevice = true;
    }
    else
#endif
    {
        if (mycol == kcol(k))
            MPI_Bcast((void *)invDiag, ksupc * ksupc, MPI_DOUBLE,
                      krow(k), (grid->cscp).comm);
    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    symTimingAdd(SYM_GPU3D_T_DIAG_BCAST, SuperLU_timer_() - sym_bcast_t);
#endif

    if (mycol == kcol(k))
    {
        xlpanel_t<double> &lpanel = lPanelVec[g2lCol(k)];
        if (lpanel.isEmpty())
            return 0;
#ifdef HAVE_CUDA
        if (superlu_acc_offload)
        {
            int64_t gpu_work_size = (int64_t)lpanel.nzrows() * (int64_t)ksupc;
            if (gpu_work_size > (int64_t)maxLvalCount)
                ABORT("SymFact GPU L-panel workspace is too small.");

            cublasHandle_t cubHandle = A_gpu.cuHandles[handle_offset];
            cudaStream_t cuStream = A_gpu.cuStreams[handle_offset];
            double *dInvDiag = A_gpu.dFBufs[buffer_offset];

            if (!invDiagOnDevice)
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double sym_inv_h2d_t = SuperLU_timer_();
#endif
                gpuErrchk(cudaMemcpyAsync(dInvDiag, invDiag,
                                          ksupc * ksupc * sizeof(double),
                                          cudaMemcpyHostToDevice, cuStream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_INV_H2D,
                             SuperLU_timer_() - sym_inv_h2d_t);
#endif
            }
            if (lpanel.haveDiag())
            {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                double sym_ldiag_t = SuperLU_timer_();
#endif
                gpuErrchk(cudaMemcpy2DAsync(lpanel.blkPtrGPU(0),
                                            lpanel.LDA() * sizeof(double),
                                            dInvDiag, ksupc * sizeof(double),
                                            ksupc * sizeof(double), ksupc,
                                            cudaMemcpyDeviceToDevice, cuStream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                symTimingAdd(SYM_GPU3D_T_LDIAG_D2D,
                             SuperLU_timer_() - sym_ldiag_t);
#endif
            }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            double sym_lpanel_t = SuperLU_timer_();
#endif
            lpanel.panelSolveSymmetricGPU(cubHandle, cuStream,
                                          ksupc, dInvDiag, ksupc,
                                          A_gpu.lookAheadLGemmBuffer[handle_offset],
                                          lpanel.nzrows());
            gpuErrchk(cudaEventRecord(A_gpu.panelReadyEvents[handle_offset],
                                      cuStream));
            if (Pr == 1 && Pc == 1 &&
                k >= 0 && static_cast<size_t>(k) < symPanelReadyEventIds.size())
                symPanelReadyEventIds[k] = handle_offset;
            bool local_singleton_panel =
                (Pr == 1 && Pc == 1 &&
                 grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1);
            if (!local_singleton_panel)
                gpuErrchk(cudaStreamSynchronize(cuStream));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
            symTimingAdd(SYM_GPU3D_T_LPANEL_TRANSFORM,
                         SuperLU_timer_() - sym_lpanel_t);
#endif
        }
        else
#endif
        {
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        double sym_lpanel_t = SuperLU_timer_();
#endif
        ensureSymFactWorkSize((int64_t)lpanel.nzrows() * (int64_t)ksupc);
        lpanel.panelSolveSymmetric(ksupc, invDiag, ksupc, symFactWork,
                                   lpanel.nzrows());
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        symTimingAdd(SYM_GPU3D_T_LPANEL_TRANSFORM,
                     SuperLU_timer_() - sym_lpanel_t);
#endif
        }
    }
#endif

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dPanelBcast(int_t k, int_t offset)
{
    if (options->SymFact == YES)
        dSymFinishL2U(k);

    /*=======   Panel Broadcast             ======*/
    xupanel_t<Ftype> k_upanel(UidxRecvBufs[offset], UvalRecvBufs[offset]);
    xlpanel_t<Ftype> k_lpanel(LidxRecvBufs[offset], LvalRecvBufs[offset]);
    if (myrow == krow(k))
        k_upanel = uPanelVec[g2lRow(k)];

    if (mycol == kcol(k))
        k_lpanel = lPanelVec[g2lCol(k)];

    if (UidxSendCounts[k] > 0)
    {
        MPI_Bcast(k_upanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
        MPI_Bcast(k_upanel.val, UvalSendCounts[k], MPI_DOUBLE, krow(k), grid3d->cscp.comm);
    }

    if (LidxSendCounts[k] > 0)
    {
        MPI_Bcast(k_lpanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
        MPI_Bcast(k_lpanel.val, LvalSendCounts[k], MPI_DOUBLE, kcol(k), grid3d->rscp.comm);
    }
    return 0;
}
