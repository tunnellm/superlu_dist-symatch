/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/



/*! @file
 * \brief Solves a system of linear equations A*X=B using 3D process grid.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * Oak Ridge National Lab
 * May 12, 2021
 * October 5, 2021
 * Last update: November 8, 2021  v7.2.0
 */

/*
 Solve-only setup
  turn off: equil, rowperm, colperm, 
  options->ilu_level = 0;
   1. DOFACT -> distribution
   2. FACTORED -> solve
*/

#include "superlu_ddefs.h"
//#include "TRF3dV100/superlu_summit.h"
#include "superlu_upacked.h"
#include "dsymldl_v2_driver.h"
// #include "pddistribute3d.h"

// #include "dssvx3dAux.c"

// int_t dgatherAllFactoredLU3d( dtrf3Dpartition_t*  trf3Dpartition,
// 			   dLUstruct_t* LUstruct, gridinfo3d_t* grid3d, SCT_t* SCT );
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
// #define DBG_MATCHING

static void dPrintFactorCommProfile(SCT_t *SCT, gridinfo3d_t *grid3d,
                                    const char *backend)
{
    if (!SCT->factorCommProfileEnabled)
        return;

    unsigned long long local[12] = {
        SCT->factorCommDataMessages +
            SCT->factorCommMetadataMessages +
            SCT->factorCommPackedMessages + SCT->factorCommAuxMessages +
            SCT->factorCommReductionMessages,
        SCT->factorCommDataBytes + SCT->factorCommMetadataBytes +
            SCT->factorCommPackedBytes + SCT->factorCommAuxBytes +
            SCT->factorCommReductionBytes,
        SCT->factorCommDataMessages,
        SCT->factorCommDataBytes,
        SCT->factorCommMetadataMessages,
        SCT->factorCommMetadataBytes,
        SCT->factorCommPackedMessages,
        SCT->factorCommPackedBytes,
        SCT->factorCommAuxMessages,
        SCT->factorCommAuxBytes,
        SCT->factorCommReductionMessages,
        SCT->factorCommReductionBytes
    };
    unsigned long long sum[12] = {0};
    unsigned long long rank_max[12] = {0};
    unsigned long long local_message_max[5] = {
        SCT->factorCommMaxDataBytes,
        SCT->factorCommMaxMetadataBytes,
        SCT->factorCommMaxPackedBytes,
        SCT->factorCommMaxAuxBytes,
        SCT->factorCommMaxReductionBytes
    };
    unsigned long long message_max[5] = {0};

    MPI_Reduce(local, sum, 12, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               grid3d->comm);
    MPI_Reduce(local, rank_max, 12, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
               grid3d->comm);
    MPI_Reduce(local_message_max, message_max, 5, MPI_UNSIGNED_LONG_LONG,
               MPI_MAX, 0, grid3d->comm);

    if (grid3d->iam != 0)
        return;

    printf(
        "Factor communication profile (logical payloads, sum/max-rank): backend=%s total_messages=%llu/%llu total_bytes=%llu/%llu data_messages=%llu/%llu data_bytes=%llu/%llu metadata_messages=%llu/%llu metadata_bytes=%llu/%llu packed_messages=%llu/%llu packed_bytes=%llu/%llu auxiliary_messages=%llu/%llu auxiliary_bytes=%llu/%llu reduction_messages=%llu/%llu reduction_bytes=%llu/%llu max_data_message_bytes=%llu max_metadata_message_bytes=%llu max_packed_message_bytes=%llu max_auxiliary_message_bytes=%llu max_reduction_message_bytes=%llu\n",
        backend, sum[0], rank_max[0], sum[1], rank_max[1], sum[2],
        rank_max[2], sum[3], rank_max[3], sum[4], rank_max[4], sum[5],
        rank_max[5], sum[6], rank_max[6], sum[7], rank_max[7], sum[8],
        rank_max[8], sum[9], rank_max[9], sum[10], rank_max[10], sum[11],
        rank_max[11], message_max[0], message_max[1], message_max[2],
        message_max[3], message_max[4]);
    fflush(stdout);
}

#ifdef GPU_ACC
enum {
    GPU_MEMORY_BASELINE_USED = 0,
    GPU_MEMORY_FACTOR_END_USED,
    GPU_MEMORY_OBSERVED_PEAK_USED,
    GPU_MEMORY_FACTOR_END_DELTA,
    GPU_MEMORY_OBSERVED_PEAK_DELTA,
    GPU_MEMORY_METRIC_COUNT
};

static int dGpuMemoryProfileSetting(void)
{
    const char *value = getenv("SUPERLU_GPU_MEMORY_PROFILE");
    if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0)
        return 0;
    if (strcmp(value, "1") == 0)
        return 1;
    return -1;
}

static int dGpuMemoryProfileCollectiveEnabled(
    superlu_dist_options_t *options, gridinfo3d_t *grid3d)
{
    enum {
        GPU_MEMORY_SETTING_INVALID = 1,
        GPU_MEMORY_SETTING_ENABLED = 2,
        GPU_MEMORY_SETTING_DISABLED = 4
    };
    int setting = dGpuMemoryProfileSetting();
    int requested = setting == 1 && sp_ienv_dist(10, options);
    int local_state = requested ? GPU_MEMORY_SETTING_ENABLED
                                : GPU_MEMORY_SETTING_DISABLED;
    int global_state = 0;

    if (setting < 0)
        local_state |= GPU_MEMORY_SETTING_INVALID;
    MPI_Allreduce(&local_state, &global_state, 1, MPI_INT, MPI_BOR,
                  grid3d->comm);

    if (global_state & GPU_MEMORY_SETTING_INVALID)
        ABORT("SUPERLU_GPU_MEMORY_PROFILE must be 0 or 1 on every rank.");
    if ((global_state & GPU_MEMORY_SETTING_ENABLED) &&
        (global_state & GPU_MEMORY_SETTING_DISABLED))
        ABORT("GPU memory profiling must be enabled consistently on every rank.");
    return (global_state & GPU_MEMORY_SETTING_ENABLED) != 0;
}

static double dGpuMemoryNonnegativeDelta(uint64_t value,
                                         uint64_t baseline)
{
    return value >= baseline ? (double) (value - baseline) : 0.0;
}

static void dPrintGpuFactorMemoryProfile(
    const superlu_gpu_memory_stats_t *stats, gridinfo3d_t *grid3d,
    const char *backend)
{
    struct { double value; int rank; } local_max[GPU_MEMORY_METRIC_COUNT];
    struct { double value; int rank; } global_max[GPU_MEMORY_METRIC_COUNT];
    double local[GPU_MEMORY_METRIC_COUNT] = {0.0};
    int valid = stats->valid && stats->factor_end_recorded;
    int valid_ranks = 0;
    int query_failures = stats->query_failures;
    int total_query_failures = 0;
    int sample_min = valid ? stats->samples : INT_MAX;
    int sample_max = valid ? stats->samples : 0;
    int global_sample_min = 0;
    int global_sample_max = 0;
    int nranks = 0;

    MPI_Comm_size(grid3d->comm, &nranks);
    if (valid) {
        local[GPU_MEMORY_BASELINE_USED] =
            (double) stats->baseline_used_bytes;
        local[GPU_MEMORY_FACTOR_END_USED] =
            (double) stats->factor_end_used_bytes;
        local[GPU_MEMORY_OBSERVED_PEAK_USED] =
            (double) stats->peak_used_bytes;
        local[GPU_MEMORY_FACTOR_END_DELTA] =
            dGpuMemoryNonnegativeDelta(stats->factor_end_used_bytes,
                                       stats->baseline_used_bytes);
        local[GPU_MEMORY_OBSERVED_PEAK_DELTA] =
            dGpuMemoryNonnegativeDelta(stats->peak_used_bytes,
                                       stats->baseline_used_bytes);
    }

    for (int i = 0; i < GPU_MEMORY_METRIC_COUNT; ++i) {
        local_max[i].value = valid ? local[i] : -1.0;
        local_max[i].rank = grid3d->iam;
    }

    MPI_Reduce(&valid, &valid_ranks, 1, MPI_INT, MPI_SUM, 0,
               grid3d->comm);
    MPI_Reduce(&query_failures, &total_query_failures, 1, MPI_INT,
               MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(&sample_min, &global_sample_min, 1, MPI_INT, MPI_MIN, 0,
               grid3d->comm);
    MPI_Reduce(&sample_max, &global_sample_max, 1, MPI_INT, MPI_MAX, 0,
               grid3d->comm);
    MPI_Reduce(local_max, global_max, GPU_MEMORY_METRIC_COUNT,
               MPI_DOUBLE_INT, MPI_MAXLOC, 0, grid3d->comm);

    if (grid3d->iam != 0)
        return;

    if (valid_ranks == 0) {
        printf("GPU_FACTOR_MEMORY backend=%s status=unavailable "
               "query_failures=%d\n",
               backend, total_query_failures);
        fflush(stdout);
        return;
    }

    const double bytes_to_mb = 1.0e-6;
    const char *status =
        valid_ranks == nranks && total_query_failures == 0
            ? "complete" : "partial";
    printf(
        "GPU_FACTOR_MEMORY backend=%s status=%s source=gpuMemGetInfo "
        "scope=device-wide unit=MB valid_ranks=%d/%d samples_min=%d "
        "samples_max=%d query_failures=%d "
        "baseline_used_max=%.3f baseline_used_max_rank=%d "
        "factor_end_used_max=%.3f factor_end_used_max_rank=%d "
        "observed_peak_used_max=%.3f observed_peak_used_max_rank=%d "
        "factor_end_delta_max=%.3f factor_end_delta_max_rank=%d "
        "observed_peak_delta_max=%.3f observed_peak_delta_max_rank=%d\n",
        backend, status, valid_ranks, nranks, global_sample_min,
        global_sample_max, total_query_failures,
        global_max[GPU_MEMORY_BASELINE_USED].value * bytes_to_mb,
        global_max[GPU_MEMORY_BASELINE_USED].rank,
        global_max[GPU_MEMORY_FACTOR_END_USED].value * bytes_to_mb,
        global_max[GPU_MEMORY_FACTOR_END_USED].rank,
        global_max[GPU_MEMORY_OBSERVED_PEAK_USED].value * bytes_to_mb,
        global_max[GPU_MEMORY_OBSERVED_PEAK_USED].rank,
        global_max[GPU_MEMORY_FACTOR_END_DELTA].value * bytes_to_mb,
        global_max[GPU_MEMORY_FACTOR_END_DELTA].rank,
        global_max[GPU_MEMORY_OBSERVED_PEAK_DELTA].value * bytes_to_mb,
        global_max[GPU_MEMORY_OBSERVED_PEAK_DELTA].rank);
    printf("GPU_FACTOR_MEMORY note: deltas include all resident allocations "
           "above the pre-factor baseline, and the peak is checkpoint-observed; "
           "values are device-wide, ranks sharing a GPU overlap, and pinned "
           "host memory is excluded.\n");
    fflush(stdout);
}
#endif

/* Build a block-row distributed structural copy of a replicated NC matrix.
 * This is used only to pass the matched quotient graph to ParMETIS. */
static void dDistributeGlobalNCByRows(const SuperMatrix *G, gridinfo_t *grid,
                                     SuperMatrix *G_loc)
{
    const NCformat *Gstore = (const NCformat *) G->Store;
    const int_t n = G->nrow;
    const int nprocs = grid->nprow * grid->npcol;
    const int iam = grid->iam;
    const int_t base = n / nprocs;
    const int_t remainder = n % nprocs;
    const int_t m_loc = base + (iam < remainder);
    const int_t fst_row = iam * base + SUPERLU_MIN((int_t) iam, remainder);
    int_t *rowptr = intCalloc_dist(m_loc + 1);
    int_t *colind;
    int_t *next;
    double *nzval;
    int_t col, pos, row, nnz_loc;

    if (G->nrow != G->ncol || G->Stype != SLU_NC || Gstore == NULL)
        ABORT("The symmetric ParMETIS quotient graph must be square NC storage.");
    if (rowptr == NULL)
        ABORT("Malloc fails for quotient-graph row pointers.");

    for (col = 0; col < n; ++col) {
        for (pos = Gstore->colptr[col]; pos < Gstore->colptr[col + 1]; ++pos) {
            row = Gstore->rowind[pos];
            if (fst_row <= row && row < fst_row + m_loc)
                ++rowptr[row - fst_row + 1];
        }
    }
    for (row = 0; row < m_loc; ++row)
        rowptr[row + 1] += rowptr[row];
    nnz_loc = rowptr[m_loc];

    colind = nnz_loc ? intMalloc_dist(nnz_loc) : NULL;
    nzval = nnz_loc ? doubleMalloc_dist(nnz_loc) : NULL;
    next = m_loc ? intMalloc_dist(m_loc) : NULL;
    if ((nnz_loc && (colind == NULL || nzval == NULL)) ||
        (m_loc && next == NULL))
        ABORT("Malloc fails for the distributed quotient graph.");
    if (m_loc)
        memcpy(next, rowptr, m_loc * sizeof(int_t));

    for (col = 0; col < n; ++col) {
        for (pos = Gstore->colptr[col]; pos < Gstore->colptr[col + 1]; ++pos) {
            row = Gstore->rowind[pos];
            if (fst_row <= row && row < fst_row + m_loc) {
                int_t dst = next[row - fst_row]++;
                colind[dst] = col;
                nzval[dst] = 1.0;
            }
        }
    }
    if (next != NULL)
        SUPERLU_FREE(next);

    dCreate_CompRowLoc_Matrix_dist(G_loc, n, n, nnz_loc, m_loc, fst_row,
                                   nzval, colind, rowptr, SLU_NR_loc,
                                   SLU_D, SLU_GE);
}

/* Expand a ParMETIS ordering of matched 1x1/2x2 quotient vertices to the
 * fine-column ordering consumed by the symmetric LU factorization. */
static void dExpandMatchedParmetisOrder(
    int_t n, const crs_info_t *crs_info, const int_t *crs_perm_c,
    int noDomains, int_t **p_sizes, int_t **p_fstVtxSep,
    int_t *perm_c, superlu_dist_options_t *options)
{
    const int_t n_crs = crs_info->n_crs;
    const int nseps = 2 * noDomains - 1;
    int_t *crs_sizes = *p_sizes;
    int_t *crs_fst = *p_fstVtxSep;
    int_t *old_prefix = intMalloc_dist(n_crs + 1);
    int_t *new_prefix = intMalloc_dist(n_crs + 1);
    int_t *rev = intMalloc_dist(n_crs);
    int_t *seen = intCalloc_dist(n_crs);
    int_t *sep_owner = intMalloc_dist(n_crs);
    int_t *sizes = intMalloc_dist(2 * noDomains);
    int_t *fstVtxSep = intMalloc_dist(2 * noDomains);
    int_t c, q, t, sep;

    if (n_crs <= 0 || crs_info->crs_vrts == NULL || crs_perm_c == NULL ||
        crs_sizes == NULL || crs_fst == NULL || old_prefix == NULL ||
        new_prefix == NULL || rev == NULL || seen == NULL ||
        sep_owner == NULL || sizes == NULL || fstVtxSep == NULL)
        ABORT("Invalid or incomplete matched quotient ordering.");

    old_prefix[0] = 0;
    for (c = 0; c < n_crs; ++c) {
        int_t width = crs_info->crs_vrts[c];
        if (width != 1 && width != 2)
            ABORT("Matched quotient vertices must have width one or two.");
        old_prefix[c + 1] = old_prefix[c] + width;
        q = crs_perm_c[c];
        if (q < 0 || q >= n_crs || seen[q] != 0)
            ABORT("ParMETIS returned an invalid quotient permutation.");
        seen[q] = 1;
        rev[q] = c;
    }
    if (old_prefix[n_crs] != n)
        ABORT("Matched quotient widths do not cover all fine columns.");

    new_prefix[0] = 0;
    for (q = 0; q < n_crs; ++q)
        new_prefix[q + 1] = new_prefix[q] + crs_info->crs_vrts[rev[q]];

    if (options->indicator_2x2 != NULL)
        SUPERLU_FREE(options->indicator_2x2);
    options->indicator_2x2 = int32Malloc_dist(n);
    if (options->indicator_2x2 == NULL)
        ABORT("Malloc fails for indicator_2x2[].");

    for (c = 0; c < n_crs; ++c) {
        q = crs_perm_c[c];
        for (t = 0; t < crs_info->crs_vrts[c]; ++t)
            perm_c[old_prefix[c] + t] = new_prefix[q] + t;
    }
    for (q = 0; q < n_crs; ++q) {
        int_t first = new_prefix[q];
        if (crs_info->crs_vrts[rev[q]] == 1) {
            options->indicator_2x2[first] = 1;
        } else {
            options->indicator_2x2[first] = 2;
            options->indicator_2x2[first + 1] = 0;
        }
    }

    for (q = 0; q < n_crs; ++q)
        sep_owner[q] = SLU_EMPTY;
    for (sep = 0; sep < nseps; ++sep) {
        int_t first = crs_fst[sep];
        int_t last = first + crs_sizes[sep];
        if (first < 0 || last < first || last > n_crs)
            ABORT("ParMETIS returned an invalid quotient separator interval.");
        fstVtxSep[sep] = new_prefix[first];
        sizes[sep] = new_prefix[last] - new_prefix[first];
        for (q = first; q < last; ++q) {
            if (sep_owner[q] != SLU_EMPTY)
                ABORT("ParMETIS quotient separator intervals overlap.");
            sep_owner[q] = sep;
        }
    }
    for (q = 0; q < n_crs; ++q)
        if (sep_owner[q] == SLU_EMPTY)
            ABORT("ParMETIS quotient separators do not cover the graph.");
    sizes[nseps] = 0;
    fstVtxSep[nseps] = n;

    SUPERLU_FREE(crs_sizes);
    SUPERLU_FREE(crs_fst);
    SUPERLU_FREE(old_prefix);
    SUPERLU_FREE(new_prefix);
    SUPERLU_FREE(rev);
    SUPERLU_FREE(seen);
    SUPERLU_FREE(sep_owner);
    *p_sizes = sizes;
    *p_fstVtxSep = fstVtxSep;
}

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * PDGSSVX3D solves a system of linear equations A*X=B,
 * by using Gaussian elimination with "static pivoting" to
 * compute the LU factorization of A.
 *
 * Static pivoting is a technique that combines the numerical stability
 * of partial pivoting with the scalability of Cholesky (no pivoting),
 * to run accurately and efficiently on large numbers of processors.
 * See our paper at http://www.nersc.gov/~xiaoye/SuperLU/ for a detailed
 * description of the parallel algorithms.
 *
 * The input matrices A and B are distributed by block rows.
 * Here is a graphical illustration (0-based indexing):
 *
 *                        A                B
 *               0 ---------------       ------
 *                   |           |        |  |
 *                   |           |   P0   |  |
 *                   |           |        |  |
 *                 ---------------       ------
 *        - fst_row->|           |        |  |
 *        |          |           |        |  |
 *       m_loc       |           |   P1   |  |
 *        |          |           |        |  |
 *        -          |           |        |  |
 *                 ---------------       ------
 *                   |    .      |        |. |
 *                   |    .      |        |. |
 *                   |    .      |        |. |
 *                 ---------------       ------
 *
 * where, fst_row is the row number of the first row,
 *        m_loc is the number of rows local to this processor
 * These are defined in the 'SuperMatrix' structure, see supermatrix.h.
 *
 *
 * Here are the options for using this code:
 *
 *   1. Independent of all the other options specified below, the
 *      user must supply
 *
 *      -  B, the matrix of right-hand sides, distributed by block rows,
 *            and its dimensions ldb (local) and nrhs (global)
 *      -  grid, a structure describing the 2D processor mesh
 *      -  options->IterRefine, which determines whether or not to
 *            improve the accuracy of the computed solution using
 *            iterative refinement
 *
 *      On output, B is overwritten with the solution X.
 *
 *   2. Depending on options->Fact, the user has four options
 *      for solving A*X=B. The standard option is for factoring
 *      A "from scratch". (The other options, described below,
 *      are used when A is sufficiently similar to a previously
 *      solved problem to save time by reusing part or all of
 *      the previous factorization.)
 *
 *      -  options->Fact = DOFACT: A is factored "from scratch"
 *
 *      In this case the user must also supply
 *
 *        o  A, the input matrix
 *
 *        as well as the following options to determine what matrix to
 *        factorize.
 *
 *        o  options->Equil,   to specify how to scale the rows and columns
 *                             of A to "equilibrate" it (to try to reduce its
 *                             condition number and so improve the
 *                             accuracy of the computed solution)
 *
 *        o  options->RowPerm, to specify how to permute the rows of A
 *                             (typically to control numerical stability)
 *
 *        o  options->ColPerm, to specify how to permute the columns of A
 *                             (typically to control fill-in and enhance
 *                             parallelism during factorization)
 *
 *        o  options->ReplaceTinyPivot, to specify how to deal with tiny
 *                             pivots encountered during factorization
 *                             (to control numerical stability)
 *
 *      The outputs returned include
 *
 *        o  ScalePermstruct,  modified to describe how the input matrix A
 *                             was equilibrated and permuted:
 *          .  ScalePermstruct->DiagScale, indicates whether the rows and/or
 *                                         columns of A were scaled
 *          .  ScalePermstruct->R, array of row scale factors
 *          .  ScalePermstruct->C, array of column scale factors
 *          .  ScalePermstruct->perm_r, row permutation vector
 *          .  ScalePermstruct->perm_c, column permutation vector
 *
 *          (part of ScalePermstruct may also need to be supplied on input,
 *           depending on options->RowPerm and options->ColPerm as described
 *           later).
 *
 *        o  A, the input matrix A overwritten by the scaled and permuted
 *              matrix diag(R)*A*diag(C)*Pc^T, where
 *              Pc is the row permutation matrix determined by
 *                  ScalePermstruct->perm_c
 *              diag(R) and diag(C) are diagonal scaling matrices determined
 *                  by ScalePermstruct->DiagScale, ScalePermstruct->R and
 *                  ScalePermstruct->C
 *
 *        o  LUstruct, which contains the L and U factorization of A1 where
 *
 *                A1 = Pc*Pr*diag(R)*A*diag(C)*Pc^T = L*U
 *
 *               (Note that A1 = Pc*Pr*Aout, where Aout is the matrix stored
 *                in A on output.)
 *
 *   3. The second value of options->Fact assumes that a matrix with the same
 *      sparsity pattern as A has already been factored:
 *
 *      -  options->Fact = SamePattern: A is factored, assuming that it has
 *            the same nonzero pattern as a previously factored matrix. In
 *            this case the algorithm saves time by reusing the previously
 *            computed column permutation vector stored in
 *            ScalePermstruct->perm_c and the "elimination tree" of A
 *            stored in LUstruct->etree
 *
 *      In this case the user must still specify the following options
 *      as before:
 *
 *        o  options->Equil
 *        o  options->RowPerm
 *        o  options->ReplaceTinyPivot
 *
 *      but not options->ColPerm, whose value is ignored. This is because the
 *      previous column permutation from ScalePermstruct->perm_c is used as
 *      input. The user must also supply
 *
 *        o  A, the input matrix
 *        o  ScalePermstruct->perm_c, the column permutation
 *        o  LUstruct->etree, the elimination tree
 *
 *      The outputs returned include
 *
 *        o  A, the input matrix A overwritten by the scaled and permuted
 *              matrix as described above
 *        o  ScalePermstruct, modified to describe how the input matrix A was
 *                            equilibrated and row permuted
 *        o  LUstruct, modified to contain the new L and U factors
 *
 *   4. The third value of options->Fact assumes that a matrix B with the same
 *      sparsity pattern as A has already been factored, and where the
 *      row permutation of B can be reused for A. This is useful when A and B
 *      have similar numerical values, so that the same row permutation
 *      will make both factorizations numerically stable. This lets us reuse
 *      all of the previously computed structure of L and U.
 *
 *      -  options->Fact = SamePattern_SameRowPerm: A is factored,
 *            assuming not only the same nonzero pattern as the previously
 *            factored matrix B, but reusing B's row permutation.
 *
 *      In this case the user must still specify the following options
 *      as before:
 *
 *        o  options->Equil
 *        o  options->ReplaceTinyPivot
 *
 *      but not options->RowPerm or options->ColPerm, whose values are
 *      ignored. This is because the permutations from ScalePermstruct->perm_r
 *      and ScalePermstruct->perm_c are used as input.
 *
 *      The user must also supply
 *
 *        o  A, the input matrix
 *        o  ScalePermstruct->DiagScale, how the previous matrix was row
 *                                       and/or column scaled
 *        o  ScalePermstruct->R, the row scalings of the previous matrix,
 *                               if any
 *        o  ScalePermstruct->C, the columns scalings of the previous matrix,
 *                               if any
 *        o  ScalePermstruct->perm_r, the row permutation of the previous
 *                                    matrix
 *        o  ScalePermstruct->perm_c, the column permutation of the previous
 *                                    matrix
 *        o  all of LUstruct, the previously computed information about
 *                            L and U (the actual numerical values of L and U
 *                            stored in LUstruct->Llu are ignored)
 *
 *      The outputs returned include
 *
 *        o  A, the input matrix A overwritten by the scaled and permuted
 *              matrix as described above
 *        o  ScalePermstruct,  modified to describe how the input matrix A was
 *                             equilibrated (thus ScalePermstruct->DiagScale,
 *                             R and C may be modified)
 *        o  LUstruct, modified to contain the new L and U factors
 *
 *   5. The fourth and last value of options->Fact assumes that A is
 *      identical to a matrix that has already been factored on a previous
 *      call, and reuses its entire LU factorization
 *
 *      -  options->Fact = Factored: A is identical to a previously
 *            factorized matrix, so the entire previous factorization
 *            can be reused.
 *
 *      In this case all the other options mentioned above are ignored
 *      (options->Equil, options->RowPerm, options->ColPerm,
 *       options->ReplaceTinyPivot)
 *
 *      The user must also supply
 *
 *        o  A, the unfactored matrix, only in the case that iterative
 *              refinment is to be done (specifically A must be the output
 *              A from the previous call, so that it has been scaled and permuted)
 *        o  all of ScalePermstruct
 *        o  all of LUstruct, including the actual numerical values of
 *           L and U
 *
 *      all of which are unmodified on output.
 *
 * Arguments
 * =========
 *
 * options (input) superlu_dist_options_t* (global)
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *         The following fields should be defined for this structure:
 *
 *         o Fact (fact_t)
 *           Specifies whether or not the factored form of the matrix
 *           A is supplied on entry, and if not, how the matrix A should
 *           be factorized based on the previous history.
 *
 *           = DOFACT: The matrix A will be factorized from scratch.
 *                 Inputs:  A
 *                          options->Equil, RowPerm, ColPerm, ReplaceTinyPivot
 *                 Outputs: modified A
 *                             (possibly row and/or column scaled and/or
 *                              permuted)
 *                          all of ScalePermstruct
 *                          all of LUstruct
 *
 *           = SamePattern: the matrix A will be factorized assuming
 *             that a factorization of a matrix with the same sparsity
 *             pattern was performed prior to this one. Therefore, this
 *             factorization will reuse column permutation vector
 *             ScalePermstruct->perm_c and the elimination tree
 *             LUstruct->etree
 *                 Inputs:  A
 *                          options->Equil, RowPerm, ReplaceTinyPivot
 *                          ScalePermstruct->perm_c
 *                          LUstruct->etree
 *                 Outputs: modified A
 *                             (possibly row and/or column scaled and/or
 *                              permuted)
 *                          rest of ScalePermstruct (DiagScale, R, C, perm_r)
 *                          rest of LUstruct (GLU_persist, Llu)
 *
 *           = SamePattern_SameRowPerm: the matrix A will be factorized
 *             assuming that a factorization of a matrix with the same
 *             sparsity	pattern and similar numerical values was performed
 *             prior to this one. Therefore, this factorization will reuse
 *             both row and column scaling factors R and C, and the
 *             both row and column permutation vectors perm_r and perm_c,
 *             distributed data structure set up from the previous symbolic
 *             factorization.
 *                 Inputs:  A
 *                          options->Equil, ReplaceTinyPivot
 *                          all of ScalePermstruct
 *                          all of LUstruct
 *                 Outputs: modified A
 *                             (possibly row and/or column scaled and/or
 *                              permuted)
 *                          modified LUstruct->Llu
 *           = FACTORED: the matrix A is already factored.
 *                 Inputs:  all of ScalePermstruct
 *                          all of LUstruct
 *
 *         o Equil (yes_no_t)
 *           Specifies whether to equilibrate the system.
 *           = NO:  no equilibration.
 *           = YES: scaling factors are computed to equilibrate the system:
 *                      diag(R)*A*diag(C)*inv(diag(C))*X = diag(R)*B.
 *                  Whether or not the system will be equilibrated depends
 *                  on the scaling of the matrix A, but if equilibration is
 *                  used, A is overwritten by diag(R)*A*diag(C) and B by
 *                  diag(R)*B.
 *
 *         o RowPerm (rowperm_t)
 *           Specifies how to permute rows of the matrix A.
 *           = NATURAL:   use the natural ordering.
 *           = LargeDiag_MC64: use the Duff/Koster algorithm to permute rows of
 *                        the original matrix to make the diagonal large
 *                        relative to the off-diagonal.
 *           = LargeDiag_HPWM: use the parallel approximate-weight perfect
 *                        matching to permute rows of the original matrix
 *                        to make the diagonal large relative to the
 *                        off-diagonal.
 *           = MY_PERMR:  use the ordering given in ScalePermstruct->perm_r
 *                        input by the user.
 *
 *         o ColPerm (colperm_t)
 *           Specifies what type of column permutation to use to reduce fill.
 *           = NATURAL:       natural ordering.
 *           = MMD_AT_PLUS_A: minimum degree ordering on structure of A'+A.
 *           = MMD_ATA:       minimum degree ordering on structure of A'*A.
 *           = MY_PERMC:      the ordering given in ScalePermstruct->perm_c.
 *
 *         o ReplaceTinyPivot (yes_no_t)
 *           = NO:  do not modify pivots
 *           = YES: replace tiny pivots by sqrt(epsilon)*norm(A) during
 *                  LU factorization.
 *
 *         o IterRefine (IterRefine_t)
 *           Specifies how to perform iterative refinement.
 *           = NO:     no iterative refinement.
 *           = SLU_DOUBLE: accumulate residual in double precision.
 *           = SLU_EXTRA:  accumulate residual in extra precision.
 *
 *         NOTE: all options must be indentical on all processes when
 *               calling this routine.
 *
 * A (input) SuperMatrix* (local); A resides on all 3D processes.
 *         On entry, matrix A in A*X=B, of dimension (A->nrow, A->ncol).
 *           The number of linear equations is A->nrow. The type of A must be:
 *           Stype = SLU_NR_loc; Dtype = SLU_D; Mtype = SLU_GE.
 *           That is, A is stored in distributed compressed row format.
 *           See supermatrix.h for the definition of 'SuperMatrix'.
 *           This routine only handles square A, however, the LU factorization
 *           routine PDGSTRF can factorize rectangular matrices.
 *
 *	   Internally, A is gathered on 2D processs grid-0, call it A2d.
 *         On exit, A2d may be overwtirren by diag(R)*A*diag(C)*Pc^T,
 *           depending on ScalePermstruct->DiagScale and options->ColPerm:
 *             if ScalePermstruct->DiagScale != NOEQUIL, A2d is overwritten by
 *                diag(R)*A*diag(C).
 *             if options->ColPerm != NATURAL, A2d is further overwritten by
 *                diag(R)*A*diag(C)*Pc^T.
 *           If all the above condition are true, the LU decomposition is
 *           performed on the matrix Pc*Pr*diag(R)*A*diag(C)*Pc^T.
 *
 * ScalePermstruct (input/output) dScalePermstruct_t* (global)
 *         The data structure to store the scaling and permutation vectors
 *         describing the transformations performed to the matrix A.
 *         It contains the following fields:
 *
 *         o DiagScale (DiagScale_t)
 *           Specifies the form of equilibration that was done.
 *           = NOEQUIL: no equilibration.
 *           = ROW:     row equilibration, i.e., A was premultiplied by
 *                      diag(R).
 *           = COL:     Column equilibration, i.e., A was postmultiplied
 *                      by diag(C).
 *           = BOTH:    both row and column equilibration, i.e., A was
 *                      replaced by diag(R)*A*diag(C).
 *           If options->Fact = FACTORED or SamePattern_SameRowPerm,
 *           DiagScale is an input argument; otherwise it is an output
 *           argument.
 *
 *         o perm_r (int*)
 *           Row permutation vector, which defines the permutation matrix Pr;
 *           perm_r[i] = j means row i of A is in position j in Pr*A.
 *           If options->RowPerm = MY_PERMR, or
 *           options->Fact = SamePattern_SameRowPerm, perm_r is an
 *           input argument; otherwise it is an output argument.
 *
 *         o perm_c (int*)
 *           Column permutation vector, which defines the
 *           permutation matrix Pc; perm_c[i] = j means column i of A is
 *           in position j in A*Pc.
 *           If options->ColPerm = MY_PERMC or options->Fact = SamePattern
 *           or options->Fact = SamePattern_SameRowPerm, perm_c is an
 *           input argument; otherwise, it is an output argument.
 *           On exit, perm_c may be overwritten by the product of the input
 *           perm_c and a permutation that postorders the elimination tree
 *           of Pc*A'*A*Pc'; perm_c is not changed if the elimination tree
 *           is already in postorder.
 *
 *         o R (double *) dimension (A->nrow)
 *           The row scale factors for A.
 *           If DiagScale = ROW or BOTH, A is multiplied on the left by
 *                          diag(R).
 *           If DiagScale = NOEQUIL or COL, R is not defined.
 *           If options->Fact = FACTORED or SamePattern_SameRowPerm, R is
 *           an input argument; otherwise, R is an output argument.
 *
 *         o C (double *) dimension (A->ncol)
 *           The column scale factors for A.
 *           If DiagScale = COL or BOTH, A is multiplied on the right by
 *                          diag(C).
 *           If DiagScale = NOEQUIL or ROW, C is not defined.
 *           If options->Fact = FACTORED or SamePattern_SameRowPerm, C is
 *           an input argument; otherwise, C is an output argument.
 *
 * B       (input/output) double* (local)
 *         On entry, the right-hand side matrix of dimension (m_loc, nrhs),
 *           where, m_loc is the number of rows stored locally on my
 *           process and is defined in the data structure of matrix A.
 *         On exit, the solution matrix if info = 0;
 *
 * ldb     (input) int (local)
 *         The leading dimension of matrix B.
 *
 * nrhs    (input) int (global)
 *         The number of right-hand sides.
 *         If nrhs = 0, only LU decomposition is performed, the forward
 *         and back substitutions are skipped.
 *
 * grid    (input) gridinfo_t* (global)
 *         The 2D process mesh. It contains the MPI communicator, the number
 *         of process rows (NPROW), the number of process columns (NPCOL),
 *         and my process rank. It is an input argument to all the
 *         parallel routines.
 *         Grid can be initialized by subroutine SUPERLU_GRIDINIT.
 *         See superlu_ddefs.h for the definition of 'gridinfo_t'.
 *
 * LUstruct (input/output) dLUstruct_t*
 *         The data structures to store the distributed L and U factors.
 *         It contains the following fields:
 *
 *         o etree (int*) dimension (A->ncol) (global)
 *           Elimination tree of Pc*(A'+A)*Pc' or Pc*A'*A*Pc'.
 *           It is computed in sp_colorder() during the first factorization,
 *           and is reused in the subsequent factorizations of the matrices
 *           with the same nonzero pattern.
 *           On exit of sp_colorder(), the columns of A are permuted so that
 *           the etree is in a certain postorder. This postorder is reflected
 *           in ScalePermstruct->perm_c.
 *           NOTE:
 *           Etree is a vector of parent pointers for a forest whose vertices
 *           are the integers 0 to A->ncol-1; etree[root]==A->ncol.
 *
 *         o Glu_persist (Glu_persist_t*) (global)
 *           Global data structure (xsup, supno) replicated on all processes,
 *           describing the supernode partition in the factored matrices
 *           L and U:
 *	       xsup[s] is the leading column of the s-th supernode,
 *             supno[i] is the supernode number to which column i belongs.
 *
 *         o Llu (dLocalLU_t*) (local)
 *           The distributed data structures to store L and U factors.
 *           See superlu_ddefs.h for the definition of 'dLocalLU_t'.
 *
 * SOLVEstruct (input/output) dSOLVEstruct_t*
 *         The data structure to hold the communication pattern used
 *         in the phases of triangular solution and iterative refinement.
 *         This pattern should be intialized only once for repeated solutions.
 *         If options->SolveInitialized = YES, it is an input argument.
 *         If options->SolveInitialized = NO and nrhs != 0, it is an output
 *         argument. See superlu_ddefs.h for the definition of 'dSOLVEstruct_t'.
 *
 * berr    (output) double*, dimension (nrhs) (global)
 *         The componentwise relative backward error of each solution
 *         vector X(j) (i.e., the smallest relative change in
 *         any element of A or B that makes X(j) an exact solution).
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics on runtime and floating-point operation count.
 *        See util_dist.h for the definition of 'SuperLUStat_t'.
 *
 * info    (output) int*
 *         = 0: successful exit
 *         < 0: if info = -i, the i-th argument had an illegal value
 *         > 0: if info = i, and i is
 *             <= A->ncol: U(i,i) is exactly zero. The factorization has
 *                been completed, but the factor U is exactly singular,
 *                so the solution could not be computed.
 *             > A->ncol: number of bytes allocated when memory allocation
 *                failure occurred, plus A->ncol.
 *
 * See superlu_ddefs.h for the definitions of varioous data types.
 * </pre>
 */

void pdgssvx3d(superlu_dist_options_t *options, SuperMatrix *A,
			   dScalePermstruct_t *ScalePermstruct,
			   double B[], int ldb, int nrhs, gridinfo3d_t *grid3d,
			   dLUstruct_t *LUstruct, dSOLVEstruct_t *SOLVEstruct,
			   double *berr, SuperLUStat_t *stat, int *info)
{
    NRformat_loc *Astore = A->Store;
    SuperMatrix GA = {0}; /* Global A in NC format, when one is required. */
    NCformat *GAstore;
    double *a_GA;

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    Glu_freeable_t *Glu_freeable = NULL;
	/* The nonzero structures of L and U factors, which are
	   replicated on all processrs.
	   (lsub, xlsub) contains the compressed subscript of
	   supernodes in L.
	   (usub, xusub) contains the compressed subscript of
	   nonzero segments in U.
	   If options->Fact != SamePattern_SameRowPerm, they are
	   computed by SYMBFACT routine, and then used by PDDISTRIBUTE
	   routine. They will be freed after PDDISTRIBUTE routine.
	   If options->Fact == SamePattern_SameRowPerm, these
	   structures are not used.                                  */
    yes_no_t parSymbFact = options->ParSymbFact;
    fact_t Fact;
    double *a;
    int_t *colptr, *rowind;
    int_t *perm_r;			/* row permutations from partial pivoting */
    int_t *perm_c;			/* column permutation vector */
    int_t *etree;			/* elimination tree */
    int_t *rowptr, *colind; /* Local A in NR */
    int colequ, Equil, factored, job, notran, rowequ, need_value;
    int_t i, j, k, irow, m, n, nnz;
    int_t nnz_loc, m_loc, fst_row, icol;
    int iam, iinfo, permc_spec;
    int ldx; /* LDA for matrix X (local). */
    char equed[1], norm[1];
    double *C, *R, *C1, *R1, amax, anorm, colcnd, rowcnd;
    double *X, *b_col, *b_work, *x_col;
    double   t, t1, t2, t3;
    float GA_mem_use = 0.0;	/* memory usage by global A */
    float dist_mem_use = 0.0; /* memory usage during distribution */
    superlu_dist_mem_usage_t num_mem_usage = {0}, symb_mem_usage = {0};
    float flinfo = 0.0; /* track memory usage of parallel symbolic factorization */
    bool Solve3D = true;
    int_t nsupers;
#if (PRNTlevel >= 1)
    double dmin, dsum, dprod;
#endif

    crs_info_t crs_info = {0};


    dtrf3Dpartition_t *trf3Dpartition=LUstruct->trf3Dpart;
    int gpu3dVersion = 1; // default is to use C++ code in CplusplusFactor/ directory
    if (getenv("GPU3DVERSION")) {
       gpu3dVersion = atoi(getenv("GPU3DVERSION"));
    }

#ifdef GPU_ACC
    LUgpu_Handle LUgpu;
    superlu_gpu_memory_stats_t gpu_memory_stats = {0};
    int gpu_memory_profile_requested = 0;
#endif

    LUstruct->dt = 'd';

    // get the 2d grid
    gridinfo_t *grid = &(grid3d->grid2d);
    iam = grid->iam;
    int use_sym_v2_solve = dSymV2SolveEnabled(options, gpu3dVersion);

    /* Test the options choices. */
    *info = 0;

    if ( options->SolveOnly == YES ) {
	options->Fact = DOFACT;       // this is set to enable distribution 
	options->Equil = NO;
	options->RowPerm = NOROWPERM;
	options->ColPerm = NATURAL;
	options->ILU_level = 0;
    }

    Fact = options->Fact;

    validateInput_pdgssvx3d(options, A, ldb, nrhs, grid3d, info);

    /* The symmetric parallel-symbolic front end is shared by symmetric LU
       and V2 LDLT.  Its V2 symbolic-to-numeric adapter currently implements
       only the block-cyclic panel/diagonal mapping. */
    if (Fact != FACTORED && parSymbFact == YES && options->SymFact == YES) {
        const char *mapping = getenv("GPU3DV2_MAPPING");
        const int cyclic_mapping =
            mapping == NULL || mapping[0] == '\0' ||
            strcmp(mapping, "CYCLIC") == 0;

        if (options->ColPerm != PARMETIS ||
            !SLU_IS_SYMATCH_ROWPERM(options->RowPerm))
            ABORT("SymFact=YES with ParSymbFact=YES requires "
                  "ColPerm=PARMETIS and symmetric matching.");
        if (gpu3dVersion != 0 && gpu3dVersion != 2)
            ABORT("SymFact=YES with ParSymbFact=YES supports only "
                  "GPU3DVERSION=0 (symmetric LU) or GPU3DVERSION=2 (LDLT).");
        if (gpu3dVersion == 2 && !cyclic_mapping)
            ABORT("ParMETIS parallel-symbolic LDLT supports only the "
                  "block-cyclic GPU3DV2_MAPPING=CYCLIC mapping; "
                  "ID, ID2D, and GREEDY are not supported.");
    }
    if (parSymbFact == YES && Fact != DOFACT && Fact != FACTORED) {
        ABORT("The 3D ParSymbFact path does not yet support symbolic-pattern reuse.");
    }
    if (parSymbFact == YES && Fact == DOFACT &&
        options->ColPerm != NATURAL && options->ColPerm != MY_PERMC &&
        options->ColPerm != PARMETIS) {
        ABORT("The 3D ParSymbFact path requires ColPerm=NATURAL, MY_PERMC, or PARMETIS.");
    }
#ifndef HAVE_PARMETIS
    if (parSymbFact == YES && Fact == DOFACT &&
        options->ColPerm == PARMETIS) {
        ABORT("ColPerm=PARMETIS requires a build configured with ParMETIS support.");
    }
#endif

    /* Initialization. */

    options->Algo3d = YES;

    /* definition of factored seen by each process layer */
    factored = (Fact == FACTORED);

    /* Save the inputs: ldb -> ldb3d, and B -> B3d, Astore -> Astore3d,
       so that the names {ldb, B, and Astore} can be used internally.
       B3d and Astore3d will be assigned back to B and Astore on return.*/
    int ldb3d = ldb;
    NRformat_loc *Astore3d = (NRformat_loc *)A->Store;
    NRformat_loc3d *A3d = SOLVEstruct->A3d;

    /* B3d is aliased to B;
       B2d is allocated;
       B is then aliased to B2d for the following 2D solve;
    */
    dGatherNRformat_loc3d_allgrid(options, Fact, (NRformat_loc *)A->Store,
				     B, ldb, nrhs, grid3d, &A3d);

    B = (double *)A3d->B2d; /* B is now pointing to B2d,
			   allocated in dGatherNRformat_loc3d.  */
    // PrintDouble5("after gather B=B2d", ldb, B);

    SOLVEstruct->A3d = A3d; /* This structure need to be persistent across
				   multiple calls of pdgssvx3d()   */

    NRformat_loc *Astore0 = A3d->A_nfmt; // on all grids
    NRformat_loc *A_orig = A->Store;
//////

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Enter pdgssvx3d()");
#endif

    /* Perform preprocessing steps on process layer zero, including:
       gather 3D matrices {A, B} onto 2D grid-0, preprocessing steps:
	   - equilibration,
	   - ordering,
	   - symbolic factorization,
	   - distribution of L & U                                      */

    m = A->nrow;
    n = A->ncol;
    // checkNRFMT(Astore0, (NRformat_loc *) A->Store);

    // On input, A->Store is on 3D, now A->Store is re-assigned to 2D store
    A->Store = Astore0; // on all grids
    ldb = Astore0->m_loc;

    /* The following code now works on all grids */
    Astore = (NRformat_loc *)A->Store;
    nnz_loc = Astore->nnz_loc;
    m_loc = Astore->m_loc;
    fst_row = Astore->fst_row;
    a = (double *)Astore->nzval;
    rowptr = Astore->rowptr;
    colind = Astore->colind;

    /* Structures needed for parallel symbolic factorization */
    int_t *sizes, *fstVtxSep;
    int noDomains, nprocs_num;
    MPI_Comm symb_comm; /* communicator for symbolic factorization */
    int col, key;		/* parameters for creating a new communicator */
    Pslu_freeable_t Pslu_freeable;
    int_t *dist_xlsub = NULL, *dist_lsub = NULL;
    int_t *dist_xusub = NULL, *dist_usub = NULL;
    int_t *dist_setree = NULL;
    float dist_symb_mem = 0.0;

    sizes = NULL;
    fstVtxSep = NULL;
    symb_comm = MPI_COMM_NULL;

    Equil = (!factored && options->Equil == YES);
    notran = (options->Trans == NOTRANS);
    iam = grid->iam;

    if (grid3d->zscp.Iam == 0) { /* on 2D grid-0 */
	/* The following code now works on 2D grid-0 */

	job = 5;
	/* Extract equilibration status from a previous factorization */
	if (factored || (Fact == SamePattern_SameRowPerm && Equil)) {
	   rowequ = (ScalePermstruct->DiagScale == ROW) ||
			 (ScalePermstruct->DiagScale == BOTH);
	   colequ = (ScalePermstruct->DiagScale == COL) ||
			 (ScalePermstruct->DiagScale == BOTH);
	} else {
	   rowequ = colequ = FALSE;
	}

	/* Not factored & ask for equilibration, then alloc R & C */
	if (Equil && Fact != SamePattern_SameRowPerm)
	     dallocScalePermstruct_RC(ScalePermstruct, m, n);

	/* The following arrays are replicated on all processes. */
	perm_r = ScalePermstruct->perm_r;
	perm_c = ScalePermstruct->perm_c;
	etree = LUstruct->etree;
	R = ScalePermstruct->R;
	C = ScalePermstruct->C;

	/* ------------------------------------------------------------
	   Diagonal scaling to equilibrate the matrix.
	   ------------------------------------------------------------ */
	if (Equil) {
	    dscaleMatrixDiagonally(options->SymFact, Fact, ScalePermstruct,
				  A, stat, grid, &rowequ, &colequ, &iinfo);
	    if (iinfo < 0) {
    		*info = -20 - iinfo;
		return;
	    }

	} /* end if Equil ... LAPACK style, not involving MC64 */

	if (!factored) { /* Skip this if already factored. */
	    /*
	     * Gather A from the distributed compressed row format to
	     * global A in compressed column format.
	     * Numerical values are gathered only when a row permutation
	     * for large diagonal is sought after.
	     */

		    if (Fact != SamePattern_SameRowPerm &&
			(parSymbFact == NO || options->RowPerm != NOROWPERM))
	    {
		/* @OGUZ-EDIT keep GA for natural order stats */
		int need_value = (options->RowPerm == LargeDiag_MC64 ||
						  SLU_IS_SYMATCH_ROWPERM(options->RowPerm) ||
							  options->RowPerm == NOROWPERM
						  );
		pdCompRow_loc_to_CompCol_global(need_value, A, grid, &GA);
		GAstore = (NCformat *)GA.Store;
		nnz = GAstore->nnz;
		GA_mem_use = (nnz + n + 1) * sizeof(int_t) + need_value * nnz * sizeof(double);
		if (!need_value)
		    assert(GAstore->nzval == NULL);
	    }

	    /* ------------------------------------------------------------
	       Find the row permutation for A.
	       ------------------------------------------------------------ */
	    dperform_row_permutation(options, Fact, ScalePermstruct, LUstruct,
				m, n, grid, A, &GA, stat, job, Equil,
				&rowequ, &colequ, &crs_info, &iinfo);
	    if (parSymbFact == YES && options->ColPerm == PARMETIS &&
		options->SymFact == YES &&
		(iinfo != 0 || crs_info.n_crs <= 0 ||
		 crs_info.crs_vrts == NULL || crs_info.ftoc == NULL ||
		 GA.Store == NULL))
		ABORT("Symmetric ParMETIS ordering requires successful matching "
		      "and complete contraction metadata.");

	} /* end if (!factored) */

	/* Compute norm(A), which will be used to adjust small diagonal. */
	if (!factored || options->IterRefine)
	    anorm = dcomputeA_Norm(notran, A, grid);

	/* ------------------------------------------------------------
	   Perform ordering and symbolic factorization
	   ------------------------------------------------------------ */
	if (!factored) {
	    t = SuperLU_timer_();
		t2 = SuperLU_timer_();
	    /*
	     * Get column permutation vector perm_c[], according to permc_spec:
	     *   permc_spec = NATURAL:  natural ordering
	     *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
	     *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
	     *   permc_spec = METIS_AT_PLUS_A: METIS on structure of A'+A
	     *   permc_spec = PARMETIS: parallel METIS on structure of A'+A
	     *   permc_spec = MY_PERMC: the ordering already supplied in perm_c[]
	     */
	    permc_spec = options->ColPerm;

	    if (parSymbFact == YES || permc_spec == PARMETIS) {
		nprocs_num = grid->nprow * grid->npcol;
		if (parSymbFact == YES && permc_spec == PARMETIS &&
		    options->SymFact == YES) {
		    int_t nd_vertices = crs_info.n_crs;
		    int max_domains = (int) SUPERLU_MIN((int_t) nprocs_num,
						      nd_vertices);
		    if (max_domains < 1)
			ABORT("The ParMETIS graph has no vertices.");
		    noDomains = 1;
		    while (noDomains <= max_domains / 2)
			noDomains *= 2;
		} else {
		    /* Preserve the established domain count for every existing
		       nonsymmetric and non-ParMETIS path. */
		    noDomains = (int)(pow(2, ((int)LOG2(nprocs_num))));
		}

		/* create a new communicator for the first noDomains
		   processes in grid->comm */
		key = iam;
		if (iam < noDomains)
			col = 0;
		else
			col = MPI_UNDEFINED;
		MPI_Comm_split(grid->comm, col, key, &symb_comm);

		if (permc_spec == NATURAL || permc_spec == MY_PERMC) {
		    if (permc_spec == NATURAL) {
			for (j = 0; j < n; ++j)	perm_c[j] = j;
		    }
		    if (!(sizes = intMalloc_dist(2 * noDomains)))
			ABORT("SUPERLU_MALLOC fails for sizes.");
		    if (!(fstVtxSep = intMalloc_dist(2 * noDomains)))
			ABORT("SUPERLU_MALLOC fails for fstVtxSep.");
		    for (i = 0; i < 2 * noDomains - 2; ++i) {
			sizes[i] = 0;
			fstVtxSep[i] = 0;
		    }
		    sizes[2 * noDomains - 2] = m;
		    fstVtxSep[2 * noDomains - 2] = 0;
		} else if (permc_spec != PARMETIS) {
		    /* same as before */
		    printf("{%4d,%4d}: pdgssvx3d: invalid ColPerm option when ParSymbfact is used\n",
				  (int)MYROW(grid->iam, grid), (int)MYCOL(grid->iam, grid));
		}
	    } /* end ... use parmetis */

	    if (permc_spec != MY_PERMC && Fact == DOFACT) {
		t3 = SuperLU_timer_();
		if (parSymbFact == YES && permc_spec == PARMETIS &&
		    options->SymFact == YES) {
		    SuperMatrix GA_c = {0};
		    SuperMatrix GA_c_loc = {0};
		    int_t *crs_perm_c;
		    int_t *identity;
		    int_t n_c;

		    if (!SLU_IS_SYMATCH_ROWPERM(options->RowPerm) ||
			GA.Store == NULL)
			ABORT("Symmetric ParMETIS ordering requires a matched global graph.");
		    coarsen_graph_v3(&GA, &GA_c, &crs_info);
		    n_c = GA_c.nrow;
		    crs_perm_c = intMalloc_dist(n_c);
		    identity = intMalloc_dist(n_c);
		    if (crs_perm_c == NULL || identity == NULL)
			ABORT("Malloc fails for the quotient-graph permutation.");
		    for (i = 0; i < n_c; ++i)
			identity[i] = i;

		    dDistributeGlobalNCByRows(&GA_c, grid, &GA_c_loc);
		    flinfo = get_perm_c_parmetis(&GA_c_loc, identity, crs_perm_c,
					 nprocs_num, noDomains, &sizes,
					 &fstVtxSep, grid, &symb_comm);
		    if (flinfo > 0)
			ABORT("ERROR in matched quotient-graph ParMETIS ordering.");
		    dExpandMatchedParmetisOrder(n, &crs_info, crs_perm_c,
					 noDomains, &sizes, &fstVtxSep,
					 perm_c, options);
		    check_perm_dist("matched_parmetis_perm_c", n, perm_c);

		    Destroy_CompRowLoc_Matrix_dist(&GA_c_loc);
		    Destroy_CompCol_Matrix_dist(&GA_c);
		    SUPERLU_FREE(crs_perm_c);
		    SUPERLU_FREE(identity);
		    SUPERLU_FREE(crs_info.crs_vrts);
		    SUPERLU_FREE(crs_info.ftoc);
		    crs_info.crs_vrts = NULL;
		    crs_info.ftoc = NULL;
		} else if (permc_spec == PARMETIS) {
		/* Get column permutation vector in perm_c.                   *
		 * This routine takes as input the distributed input matrix A *
		 * and does not modify it.  It also allocates memory for      *
		 * sizes[] and fstVtxSep[] arrays, that contain information   *
		 * on the separator tree computed by ParMETIS.                */
		    flinfo = get_perm_c_parmetis(A, perm_r, perm_c, nprocs_num,
						 noDomains, &sizes, &fstVtxSep,
						 grid, &symb_comm);
		    if (flinfo > 0)
			ABORT("ERROR in get perm_c parmetis.");
		
	  } else if (!(parSymbFact == YES && permc_spec == NATURAL)) {
		  t3 = SuperLU_timer_();
		  
		  /* generate uncoarsened versions of GA and perm_c but also
			 maintain the contracted versions */
		  /* @EDIT-SYMATCH Add branch here */
		  if (SLU_IS_SYMATCH_ROWPERM(options->RowPerm))
		  {
			  /* @EDIT-SYMATCH 2. Bc = coarsen(B) */
		      SuperMatrix GA_c;

			  t = SuperLU_timer_();
			  
		      // coarsen_graph_v2(&GA, &GA_c, crs_info.n_crs, crs_info.crs_vrts);
			  coarsen_graph_v3(&GA, &GA_c, &crs_info);
			  
			  t = SuperLU_timer_()-t;

			#if ( PRNTlevel>=1 )
			printf("coarsen_graph_v2 (Bc): %f \n",t);
			#endif
			
			  NCformat  *cGstore  = (NCformat *) GA_c.Store;
		      int		 n_c	  = GA_c.nrow;	// coarse dimension
		      int_t		 nnz_c	  = cGstore->nnz;
		      int_t	    *c_colptr = cGstore->colptr;
		      int_t	    *c_rowind = cGstore->rowind;
		      // double    *c_nzval  = cGstore->nzval;

			  #if ( DEBUGlevel>=1 )
			  is_symmetric(n_c, nnz_c, c_colptr, c_rowind, c_nzval);
			  #endif
			  // exit(33);

			  t = SuperLU_timer_();

			  
			  
			  /* @EDIT-SYMATCH 3. Pc = fill(Bc) */
		      int_t *crs_perm_c =       // Sherry: why not intMalloc_dist?
			  (int_t *)malloc(sizeof(*crs_perm_c) * GA_c.nrow);
		      get_perm_c_dist(iam, permc_spec, &GA_c, crs_perm_c);

			  t = SuperLU_timer_()-t;
			  printf("get_perm_c_dist (Pc = fill(Bc)): %f \n",t);

			  #if ( PRNTlevel>=1 )
			  printf("get_perm_c_dist (Pc = fill(Bc)): %f \n",t);
			  #endif

		      /* Compute coarse etree of Pc*A*Pc' */
			  

		      /* @EDIT-SYMATCH 4. C = Pc Bc PcT */
			  t = SuperLU_timer_();
		      /* colptr/rowind/nzval are both input and output */
		      apply_perm_sym_pattern(n_c, nnz_c, c_colptr, c_rowind,
									 crs_perm_c);

			  t = SuperLU_timer_()-t;
			  printf("apply_perm_sym (C = PcBcPc^T): %f \n",t);

			  #if ( PRNTlevel>=1 )
			  printf("apply_perm_sym (C = PcBcPc^T): %f \n",t);
			  #endif
			  
			  #if ( DEBUGlevel>=1 )
			  is_symmetric(n_c, nnz_c, c_colptr, c_rowind, c_nzval);
			  #endif
			  
			  // exit(44);

			  
			  /* etree */
		      int_t *c_etree = intMalloc_dist(n_c);
		      int_t *c_colend = (int_t*) intMalloc_dist(n_c);

		      for (i = 0; i < n_c; ++i)
				  c_colend[i] = c_colptr[i+1];

			  t=SuperLU_timer_();
			  
			  sp_symetree_dist(c_colptr, c_colend, c_rowind, n_c, c_etree);
			  
			  t = SuperLU_timer_()-t;
			  printf("sp_symetree_dist: %f \n",t);

              #if ( PRNTlevel>=1 )
			  printf("sp_symetree_dist: %f \n",t);
              #endif
			  
			  #ifdef DBG_MATCHING
			  FILE *outfile = fopen("debug-output", "a");
			  fprintf(outfile, "=== c_etree (C) ===\n");
			  for (i = 0; i < n_c; ++i)
				  fprintf(outfile, "%d %d\n", i, c_etree[i]);
			  fprintf(outfile, "\n\n\n");
			  fclose(outfile);
			  #endif


			  
		      /* Postorder the etree -> crs_perm_c[] is modified */
			  t=SuperLU_timer_();
		      int_t *post = (int_t *) TreePostorder_dist(n_c, c_etree);

			  t = SuperLU_timer_()-t;
			  printf("TreePostorder: %f \n",t);

			  // exit(44);
			  
			  
			  /* @EDIT-SYMATCH 5. D = Pe C PeT */
		      /* Permute GA_c again by post[] */
			  t=SuperLU_timer_();
		      apply_perm_sym_pattern(n_c, nnz_c, c_colptr, c_rowind, post);
			  
			  #if ( DEBUGlevel>=1 )
		      is_symmetric(n_c, nnz_c, c_colptr, c_rowind, c_nzval);
			  #endif
			  
			  t = SuperLU_timer_()-t;
			  printf("apply_perm_sym (D = PeCPe^T): %f \n",t);

			  #if ( PRNTlevel>=1 )
			  printf("apply_perm_sym (D = PeCPe^T): %f \n",t);
			  #endif
			  
			  // exit(55);

			  t=SuperLU_timer_();

		      int *iwork = int32Malloc_dist(n_c);
		      for (i = 0; i < n_c; ++i)
				  iwork[i] = post[crs_perm_c[i]]; // product of crs_perm_c and post
		      for (i = 0; i < n_c; ++i)
				  crs_perm_c[i] = iwork[i];

			  #ifdef DBG_MATCHING
			  outfile = fopen("debug-output", "a");
			  fprintf(outfile, "=== crs_perm_c (after post) ===\n");
			  for (i = 0; i < n_c; ++i)
				  fprintf(outfile, "%d %d\n", i, crs_perm_c[i]);
			  fprintf(outfile, "\n\n\n");
			  fclose(outfile);
			  #endif

		      /* Renumber coarse etree in postorder */
		      for (i = 0; i < n_c; ++i)
				  iwork[post[i]] = post[c_etree[i]];
		      for (i = 0; i < n_c; ++i)
				  c_etree[i] = iwork[i];
		      //PrintInt10("postordered coarse etree", n_c, c_etree);
			  
		      #if ( DEBUGlevel>=1 )
			  is_postorder(n_c, c_etree);
			  #endif
			  
			  #ifdef DBG_MATCHING
			  outfile = fopen("debug-output", "a");
			  fprintf(outfile, "=== c_etree (D) ===\n");
			  for (i = 0; i < n_c; ++i)
				  fprintf(outfile, "%d %d\n", i, c_etree[i]);
			  fprintf(outfile, "\n\n\n");
			  /* indicator on B */
			  int *indicator_2x2_tmp = (int*) int32Malloc_dist(n);
			  for (i = 0, j = 0; i < crs_info.n_crs; ++i)
			  {
				  if (crs_info.crs_vrts[i] == 1)
					  indicator_2x2_tmp[j++] = 1;
				  else if (crs_info.crs_vrts[i] == 2)
				  {
					  indicator_2x2_tmp[j++] = 2;
					  indicator_2x2_tmp[j++] = 0;
				  }
			  }
			  fclose(outfile);
			  #endif

			  // exit(66);

		      SUPERLU_FREE(c_colend);
		      SUPERLU_FREE(post);
		      SUPERLU_FREE(iwork);

		      // fprintf(stdout, "Projecting back the coarse column permutation ...\n");
		      if ( !(options->indicator_2x2 = (int*) int32Malloc_dist(n)) )
			  ABORT("Malloc fails for indicator_2x2[].");
		      int *indicator_2x2 = options->indicator_2x2;

		      /* cumulative crs_vrts. */
		      int_t *crs_vrts_cum = (int_t *) intMalloc_dist(crs_info.n_crs+1);
		      crs_vrts_cum[0] = 0;
			  
			  #if 1   // Oguz's code: before apply crs_perm_c[]
		      for (i = 0; i < crs_info.n_crs; ++i)
				  crs_vrts_cum[i+1] = crs_vrts_cum[i] + crs_info.crs_vrts[i];
			  #else   // Sherry mod
		      for (i = 0; i < crs_info.n_crs; ++i) {
			  crs_vrts_cum[i+1] = crs_vrts_cum[i] + crs_info.crs_vrts[i];
			  if (crs_info.crs_vrts[i] == 1) { /* 1x1 pivot */
			      options->indicator_2x2[crs_vrts_cum[i]] = 1;
			  } else if (crs_info.crs_vrts[i] == 2) { /* 2x2 pivot */
			      options->indicator_2x2[crs_vrts_cum[i]] = 2;
			      options->indicator_2x2[crs_vrts_cum[i]+1] = 0;
			  } else {
			      ABORT("Invalid value of crs_vrts[i]");
			  }
		      }
			  #endif
			  
		      /* reverse crs_perm_c. */
		      int_t *rev_crs_perm_c = (int_t *) intMalloc_dist(crs_info.n_crs);
		      for (i = 0; i < crs_info.n_crs; ++i)
				  rev_crs_perm_c[crs_perm_c[i]] = i;

		      int cur = 0, rev_i;

			  #if 0  // Oguz's code
		      /* Expand crs_perm_c into perm_c[] */
		      for (i = 0; i < crs_info.n_crs; ++i) // new label in coarse G
			  {
			      rev_i = rev_crs_perm_c[i]; // old label in coarse G
			      for (j = crs_vrts_cum[rev_i]; j < crs_vrts_cum[rev_i+1]; ++j)
				  perm_c[j] = cur++;
			  }
			  #else  // Sherry mod
		      /* Expand crs_perm_c into perm_c[] */
		      for (i = 0; i < crs_info.n_crs; ++i)	// new label in coarse G
			  {
				  rev_i = rev_crs_perm_c[i]; // old label in coarse G
				  for (j = crs_vrts_cum[rev_i]; j < crs_vrts_cum[rev_i+1]; ++j)
					  perm_c[j] = cur++;

				  /* Set up indicator_2x2[] for the vertices permuted by
					 perm_c */
				  j = crs_vrts_cum[rev_i];
				  k = perm_c[j];
				  if (crs_info.crs_vrts[rev_i] == 1) { /* 1x1 pivot */
					  indicator_2x2[k] = 1;
				  } else if (crs_info.crs_vrts[rev_i] == 2) { /* 2x2 pivot */
					  indicator_2x2[ k ] = 2;
					  indicator_2x2[ k+1 ] = 0;
				  } else {
					  ABORT("Invalid value of crs_vrts[i]");
				  }
		      }

			  #if ( DEBUGlevel>=1 )
		      PrintInt32("indicator_2x2", n, options->indicator_2x2);
			  #endif

			  #ifdef DBG_MATCHING
			  outfile = fopen("debug-output", "a");
			  int indicator_pass = 1;
			  for (i = 0; i < n; ++i)
			  {
				  if (indicator_2x2_tmp[i] != indicator_2x2[perm_c[i]])
				  {
					  fprintf(outfile, "indicator mismatch at idx %d %d\n",
							  i, perm_c[i]);
					  indicator_pass = 0;
				  }
			  }
			  if (!indicator_pass)
				  fprintf(outfile, "indicator computation FAIL\n");
			  else
				  fprintf(outfile, "indicator computation PASS\n");
			  fprintf(outfile, "\n\n\n");

			  fprintf(outfile, "=== coarse to fine info (D -> E) ===\n");
			  int *crs_vrts_cum2 = (int*) int32Malloc_dist(crs_info.n_crs+1);
			  crs_vrts_cum2[0] = 0;
			  j = 0;
			  for (i = 0; i < crs_info.n_crs; ++i)
			  {
				  rev_i = rev_crs_perm_c[i];
				  crs_vrts_cum2[i+1] = crs_vrts_cum2[i] + crs_info.crs_vrts[rev_i];
				  if (crs_info.crs_vrts[rev_i] == 1)
				  {
					  fprintf(outfile, "%d %d\n", i, j);
					  ++j;
				  }
				  else if (crs_info.crs_vrts[rev_i] == 2)
				  {
					  fprintf(outfile, "%d %d %d\n", i, j, j+1);
					  j += 2;
				  }
			  }
			  fprintf(outfile, "\n\n\n");
			  fclose(outfile);
			  #endif

			  // exit(77);


		      /* Expand coarse etree to fine etree      */
		      int parent, rev_p, fine_p;
		      for (i = 0; i < crs_info.n_crs; ++i)  // new label in coarse G
			  {
				  parent = c_etree[i]; // Sherry: what if parent is ROOT (= n_c)?
				  if (parent == n_c ) {
					  fine_p = n; // root
				  } else {
					  rev_p = rev_crs_perm_c[parent]; // old label
					  fine_p = crs_vrts_cum[rev_p]; // expanded old label
					  fine_p = perm_c[fine_p]; // new parent
				  }
				  rev_i = rev_crs_perm_c[i]; // old label in coarse G
				  j = crs_vrts_cum[rev_i];
				  k = perm_c[j];

				  if (crs_info.crs_vrts[rev_i] == 1) { /* 1x1 pivot */
					  etree[k] = fine_p; //perm_c[fine_p];
				  } else if (crs_info.crs_vrts[rev_i] == 2) { /* 2x2 pivot */
					  etree[k] = k+1;
					  etree[k+1] = fine_p; //perm_c[fine_p];
				  }
		      }
		      // Set root ??? - ROOT MAY HAVE multiple children
		      //etree[n-1] = n;

		      //PrintInt10("etree", n, etree);
			  #endif

			  // is_postorder(n, etree);

			  #ifdef DBG_MATCHING
			  outfile = fopen("debug-output", "a");
			  fprintf(outfile, "=== etree (E) ===\n");
			  for (i = 0; i < n; ++i)
				  fprintf(outfile, "%d %d\n", i, etree[i]);
			  fprintf(outfile, "\n\n\n");

			  fprintf(outfile, "=== coarse e-tree to fine e-tree  ===\n");
			  int tmp[4];
			  for (i = 0; i < crs_info.n_crs; ++i)
			  {
				  j = 0;
				  fprintf(outfile, "%d ", i);
				  int is_crs = (crs_vrts_cum2[i+1] - crs_vrts_cum2[i] == 2);
				  if (is_crs)
				  {
					  fprintf(outfile, "(C %d %d): ",
							  crs_vrts_cum2[i], crs_vrts_cum2[i]+1);
					  tmp[j++] = crs_vrts_cum2[i];
					  tmp[j++] = crs_vrts_cum2[i] + 1;
				  }
				  else
				  {
					  fprintf(outfile, "(N %d): ", crs_vrts_cum2[i]);
					  tmp[j++] = crs_vrts_cum2[i];
				  }

				  int parent = c_etree[i];
				  if (parent == n_c)
					  fprintf(outfile, "ROOT ");
				  else
				  {
					  is_crs = (crs_vrts_cum2[parent+1] - crs_vrts_cum2[parent] == 2);
					  fprintf(outfile, "%d ", parent);
					  if (is_crs)
					  {
						  fprintf(outfile, "(C %d %d): ",
								  crs_vrts_cum2[parent], crs_vrts_cum2[parent]+1);
						  tmp[j++] = crs_vrts_cum2[parent];
					  	  tmp[j++] = crs_vrts_cum2[parent] + 1;
					  }
					  else
					  {
						  fprintf(outfile, "(N %d): ", crs_vrts_cum2[parent]);
						  tmp[j++] = crs_vrts_cum2[parent];
					  }
				  }

				  fprintf(outfile, " | ");
				  for (k = 0; k < j; ++k)
					  fprintf(outfile, "%d: %d - ", tmp[k], etree[tmp[k]]);

				  fprintf(outfile, "\n");
			  }

			  fclose(outfile);
			  #endif

			  //exit(88);

		      SUPERLU_FREE(c_etree);
		      SUPERLU_FREE(crs_vrts_cum);
		      if (rev_crs_perm_c)
				  SUPERLU_FREE(rev_crs_perm_c);

		      check_perm_dist("uncoarsen_perm_c", GA.nrow, perm_c);

		      // fprintf(stdout, "DONE.\n"); fflush(stdout);
			  fflush(stdout);
			}  /* end if (SLU_IS_SYMATCH_ROWPERM(options->RowPerm)) */		
			else {
				get_perm_c_dist(iam, permc_spec, &GA, perm_c);
			}
		}

		t3 = SuperLU_timer_() - t3;
		#if ( PRNTlevel>=1 )
		printf("!factored: %f \n", t3);
		#endif

	    }

	    stat->utime[COLPERM] = SuperLU_timer_() - t2;

		printf("COLPERM: %f \n", stat->utime[COLPERM]);

		// exit(79);

	    /* Compute the elimination tree of Pc*(A'+A)*Pc' or Pc*A'*A*Pc'
	       (a.k.a. column etree), depending on the choice of ColPerm.
	       Adjust perm_c[] to be consistent with a postorder of etree.
	       Permute columns of A to form A*Pc'. */
	    if (Fact != SamePattern_SameRowPerm) {
		if (parSymbFact == NO)
		{
		    /* Allocating Glu_freeable used by symbfact */
		    if (!(Glu_freeable = (Glu_freeable_t *)
			  SUPERLU_MALLOC(sizeof(Glu_freeable_t))))
				ABORT("Malloc fails for Glu_freeable.");
		    /* compute symbolic LU or ILU */
		    permCol_SymbolicFact3d(options, n, &GA, perm_c, etree,
					   Glu_persist, Glu_freeable, stat,
					   &symb_mem_usage,
					   grid3d);
		} /* end serial symbolic factorization */
		else { /* parallel symbolic factorization */
		    t = SuperLU_timer_();
		    flinfo = symbfact_dist(options, nprocs_num, noDomains,
					  A, perm_c, perm_r,
					  sizes, fstVtxSep, &Pslu_freeable,
					  &(grid->comm), &symb_comm,
					  &symb_mem_usage);
		    stat->utime[SYMBFAC] = SuperLU_timer_() - t;
		    if (flinfo > 0)
		        ABORT("Insufficient memory for parallel symbolic factorization.");

		    /* Convert the process-local parallel-symbolic output once on
		       layer zero.  The compact 2D structures are subsequently
		       replicated down each z-line and assembled independently on
		       every process layer. */
		    dist_symb_mem = ddist_symbLU(options, n, &Pslu_freeable,
					      Glu_persist, &dist_xlsub,
					      &dist_lsub, &dist_xusub,
					      &dist_usub, grid);
		    if (dist_symb_mem > 0)
			ABORT("Insufficient memory while redistributing parallel symbolic data.");

		    nsupers = getNsupers(n, Glu_persist);
		    dist_setree = ddist_build_supno_tree(
			n, nsupers, sizes, fstVtxSep, noDomains,
			dist_xlsub, dist_lsub, dist_xusub, dist_usub,
			Glu_persist, grid);

		    /* Install a scalar etree consistent with the supernodal tree.
		       The direct 3D partition initializer consumes dist_setree, but
		       later common code still carries LUstruct->etree. */
		    for (j = 0; j < nsupers; ++j) {
			int_t col_first = Glu_persist->xsup[j];
			int_t col_last = Glu_persist->xsup[j + 1] - 1;
			for (i = col_first; i < col_last; ++i)
			    etree[i] = i + 1;
			etree[col_last] = (dist_setree[j] == nsupers)
			    ? n : Glu_persist->xsup[dist_setree[j]];
		    }
		}

		/* Destroy GA */
		/* @OGUZ-EDIT keep GA for natural order stats */
		if (parSymbFact == NO || options->RowPerm != NOROWPERM)
		    Destroy_CompCol_Matrix_dist(&GA);

	    } /* end if Fact not SamePattern_SameRowPerm */
	} /* end if not Factored */
    } /* end 2D process layer 0 */

	// exit(43);

    MPI_Bcast(&rowequ, 1, MPI_INT, 0, grid3d->zscp.comm);
    MPI_Bcast(&colequ, 1, MPI_INT, 0, grid3d->zscp.comm);
    /* Now all processes in 3D grid participate */
    
    /* Broadcast Permuted A and symbolic factorization data from 2d to 3d grid */
    /* Sherry Q: original input A, not permuted yet */
    
    if (Fact != SamePattern_SameRowPerm && !factored) // place the exact conditions later //all the grid must execute this
    {
	if (parSymbFact == NO) {
		if (Glu_freeable == NULL)
		{
		    if (!(Glu_freeable = (Glu_freeable_t *)
				SUPERLU_MALLOC(sizeof(Glu_freeable_t))))
			ABORT("Malloc fails for Glu_freeable.");
		}
		dbcastPermutedSparseA(A, ScalePermstruct, Glu_freeable,
				     LUstruct, grid3d);
		} else {
		    dbcastDistSymbLU(A, ScalePermstruct, LUstruct,
				      &dist_xlsub, &dist_lsub,
				      &dist_xusub, &dist_usub,
				      &dist_symb_mem, &dist_setree,
				      grid3d);
		}
	}

	perm_r = ScalePermstruct->perm_r;
	perm_c = ScalePermstruct->perm_c;
	etree = LUstruct->etree;
	R = ScalePermstruct->R;
	C = ScalePermstruct->C;
	nsupers = getNsupers(n, LUstruct->Glu_persist);
	Astore = (NRformat_loc *)A->Store;
	a = (double *)Astore->nzval;
	rowptr = Astore->rowptr;
	colind = Astore->colind;
	Glu_persist = LUstruct->Glu_persist;

	// perform the  3D distribution
	if (!factored)
	{
	    /* CASE OF SERIAL SYMBOLIC */
  	    /* Apply column permutation to the original distributed A */
	    if (SLU_IS_SYMATCH_ROWPERM(options->RowPerm)) {  /* Sherry mod: need appy perm_r[]
						    to columns */
		for (j = 0; j < nnz_loc; ++j) colind[j] = perm_c[perm_r[colind[j]]];
	    } else {
		for (j = 0; j < nnz_loc; ++j) colind[j] = perm_c[colind[j]];
	    }
		// free quauntities used in Parmetis
		if (sizes)
			SUPERLU_FREE(sizes);
		if (fstVtxSep)
			SUPERLU_FREE(fstVtxSep);
		if (symb_comm != MPI_COMM_NULL)
			MPI_Comm_free(&symb_comm);
		if ( Fact != SamePattern_SameRowPerm){
			if (LUstruct->trf3Dpart != NULL) {
				dDestroy_trf3Dpartition(LUstruct->trf3Dpart);
				LUstruct->trf3Dpart = NULL;
			}
			LUstruct->trf3Dpart = (dtrf3Dpartition_t *)SUPERLU_MALLOC(sizeof(dtrf3Dpartition_t));
			if (LUstruct->trf3Dpart == NULL)
				ABORT("Malloc fails for the 3D factorization partition.");
			// computes the new partition for 3D factorization here
			trf3Dpartition=LUstruct->trf3Dpart;
			if (parSymbFact == YES) {
				if (options->SymFact == YES && gpu3dVersion == 2)
					dSymV2TrfPartitionInitFromDistSymb(
						nsupers, dist_setree, dist_xlsub,
						dist_lsub, LUstruct, grid3d, options);
				else
					dnewTrfPartitionInitFromSetree(nsupers,
							 dist_setree,
							 LUstruct, grid3d);
			}
			else if (options->SymFact == YES && gpu3dVersion == 2)
				dSymV2TrfPartitionInit(nsupers, LUstruct, Glu_freeable,
						       grid3d, options);
			else
				dnewTrfPartitionInit(nsupers, LUstruct, grid3d);
			if (dist_setree != NULL) {
				SUPERLU_FREE(dist_setree);
				dist_setree = NULL;
			}
		}
	}
	// perform the  3D distribution
	if (!factored)
	{ /* Skip this if already factored. */

	//if (parSymbFact == NO || ???? Fact == SamePattern_SameRowPerm) {
	if ( parSymbFact == NO ) {

			/* Distribute Pc*Pr*diag(R)*A*diag(C)*Pc' into L and U storage.
				NOTE: the row permutation Pc*Pr is applied internally in the
				distribution routine. */
			t = SuperLU_timer_();

			if (options->SymFact == YES && gpu3dVersion == 2)
				dist_mem_use = dSymV2Distribute3d(options, n, A,
								 ScalePermstruct, Glu_freeable,
								 LUstruct, grid3d);
			else
				dist_mem_use = pddistribute3d_Yang(options, n, A,
								  ScalePermstruct, Glu_freeable,
								  LUstruct, grid3d);
			stat->utime[DIST] = SuperLU_timer_() - t;

			/* Deallocate storage used in symbolic factorization. */
			if (Fact != SamePattern_SameRowPerm)
			{
				iinfo = symbfact_SubFree(Glu_freeable);
				SUPERLU_FREE(Glu_freeable);
			}
		}
		else
		{
			/* Distribute Pc*Pr*diag(R)*A*diag(C)*Pc' into L and U storage.
				NOTE: the row permutation Pc*Pr is applied internally in the
				distribution routine. */

			t = SuperLU_timer_();
			if (options->SymFact == YES && gpu3dVersion == 2)
				dist_mem_use = dSymV2Distribute3dFromSymb(
					options, n, A, ScalePermstruct,
					dist_xlsub, dist_lsub,
					dist_xusub, dist_usub,
					dist_symb_mem, LUstruct, grid3d);
			else
				dist_mem_use = ddist_psymbtonum3d(
					options, n, A, ScalePermstruct,
					dist_xlsub, dist_lsub,
					dist_xusub, dist_usub,
					dist_symb_mem, LUstruct, grid3d);
			if (dist_mem_use > 0)
				ABORT("Not enough memory available for dist_psymbtonum\n");
			dist_xlsub = dist_lsub = NULL;
			dist_xusub = dist_usub = NULL;

			stat->utime[DIST] = SuperLU_timer_() - t;

		}

		/* Flatten L metadata into one buffer. */
		if ( Fact != SamePattern_SameRowPerm &&
		     !dSymV2SolveEnabled(options, gpu3dVersion) ) {
			pdflatten_LDATA(options, n, LUstruct, grid, stat);
		}

		if(Fact != SamePattern_SameRowPerm){
			// checkDist3DLUStruct(LUstruct, grid3d);
			// zeros out the Supernodes that are not owned by the grid
			if (!dSymV2SolveEnabled(options, gpu3dVersion))
				dinit3DLUstructForest(trf3Dpartition->myTreeIdxs,
						      trf3Dpartition->myZeroTrIdxs,
						      trf3Dpartition->sForests,
						      LUstruct, grid3d);

			dLUValSubBuf_t *LUvsb = SUPERLU_MALLOC(sizeof(dLUValSubBuf_t));
			if (dSymV2SolveEnabled(options, gpu3dVersion))
				dSymV2LluBufInit(LUvsb);
			else
				dLluBufInit(LUvsb, LUstruct);
			trf3Dpartition->LUvsb = LUvsb;
			if (dSymV2SolveEnabled(options, gpu3dVersion))
				trf3Dpartition->iperm_c_supno =
					dSymV2CreateIdentityIpermSupno(nsupers);
			else
				trf3Dpartition->iperm_c_supno =
					create_iperm_c_supno(nsupers, options,
							     LUstruct->Glu_persist,
							     LUstruct->etree,
							     LUstruct->Llu->Lrowind_bc_ptr,
							     LUstruct->Llu->Ufstnz_br_ptr,
							     grid3d);
		}


		/*if (!iam) printf ("\tDISTRIBUTE time  %8.2f\n", stat->utime[DIST]); */

		MPI_Bcast(&anorm, 1, MPI_DOUBLE, 0, grid3d->zscp.comm);

		/* Perform numerical factorization in parallel on all process layers.*/

		/* nvshmem related. The nvshmem_malloc has to be called before dtrs_compute_communication_structure, otherwise solve is much slower*/
		#ifdef HAVE_NVSHMEM
			int nc = CEILING( nsupers, grid->npcol);
			int nr = CEILING( nsupers, grid->nprow);
			int flag_bc_size = RDMA_FLAG_SIZE * (nc+1);
			int flag_rd_size = RDMA_FLAG_SIZE * nr * 2;
			int my_flag_bc_size = RDMA_FLAG_SIZE * (nc+1);
			int my_flag_rd_size = RDMA_FLAG_SIZE * nr * 2;
			int maxrecvsz = sp_ienv_dist(3, options)* nrhs + SUPERLU_MAX( XK_H, LSUM_H );
			int ready_x_size = maxrecvsz*nc;
			int ready_lsum_size = 2*maxrecvsz*nr;
			if (!use_sym_v2_solve && get_acc_solve()){
			nv_init_wrapper(grid->comm);
		    dprepare_multiGPU_buffers(flag_bc_size,flag_rd_size,ready_x_size,ready_lsum_size,my_flag_bc_size,my_flag_rd_size);
			}
		#endif


		SCT_t *SCT = (SCT_t *)SUPERLU_MALLOC(sizeof(SCT_t));
		slu_SCT_init(SCT);

#if (PRNTlevel >= 1)
		if (grid3d->iam == 0)
		{
			printf("after 3D initialization.\n");
			fflush(stdout);
		}
#endif

        if ( options->SolveOnly != YES ) { // Now we need factorization

#ifdef GPU_ACC
		gpu_memory_profile_requested =
			dGpuMemoryProfileCollectiveEnabled(options, grid3d);
		if (gpu_memory_profile_requested) {
			MPI_Barrier(grid3d->comm);
			superlu_gpu_memory_tracker_start();
			MPI_Barrier(grid3d->comm);
		}
#endif
		t = SuperLU_timer_();

		/*factorize in grid 1*/
		// if(grid3d->zscp.Iam)
		// get environment variable TRF3DVERSION
#ifdef GPU_ACC
		if (gpu3dVersion == 2 && !dSymV2SolveEnabled(options, gpu3dVersion))
			ABORT("GPU3DVERSION=2 requires SymFact=YES.");

		if (gpu3dVersion == 1 || dSymV2SolveEnabled(options, gpu3dVersion))
		{ /* this is the new C++ code in CplusplusFactor/ directory */
#if (PRNTlevel>=1)
			if (!grid3d->iam)
				printf("Using pdgstrf3d+gpu version %d\n", gpu3dVersion);
#endif

			int ldt = sp_ienv_dist(3, options); /* Size of maximum supernode */
			double s_eps = smach_dist("Epsilon");
			double thresh = s_eps * anorm;
			
			if(options->batchCount == 0)
			{
#define TEMPLATED_VERSION
#ifdef TEMPLATED_VERSION
dLUgpu_Handle dLUgpu = dCreateLUgpuHandle(nsupers, ldt, trf3Dpartition, LUstruct, grid3d,
						  SCT, options, stat, thresh, info);
			if (gpu_memory_profile_requested)
				superlu_gpu_memory_tracker_sample();

			/* call pdgstrf3d() in C++ code */
			if (use_sym_v2_solve)
				pdgstrf3d_LUv2(dLUgpu);
			else
				pdgstrf3d_LUv1(dLUgpu);
			if (gpu_memory_profile_requested) {
				superlu_gpu_memory_tracker_mark_factor_end();
				MPI_Barrier(grid3d->comm);
			}

			if (use_sym_v2_solve && nrhs > 0 && *info == 0) {
				SOLVEstruct->symldl_v2_factor_handle = (void *) dLUgpu;
				dLUgpu = NULL;
			} else {
				dCopyLUGPU2Host(dLUgpu, LUstruct);
			}
			if (dLUgpu != NULL)
				dDestroyLUgpuHandle(dLUgpu);
		    //TODO: dCreateLUgpuHandle,pdgstrf3d_LUpackedInterface,dCopyLUGPU2Host,dDestroyLUgpuHandle haven't been created
#else // non-templated version (not used anymore)
			/* call constructor in C++ code */
			LUgpu = dCreateLUgpuHandle(nsupers, ldt, trf3Dpartition, LUstruct, grid3d,
						  SCT, options, stat, thresh, info);

			/* call pdgstrf3d() in C++ code */
			pdgstrf3d_LUpackedInterface(LUgpu);

			copyLUGPU2Host(LUgpu, LUstruct);
			destroyLUgpuHandle(LUgpu);
#endif /* end if TEMPLATED_VERSION */

       	      	 } else { /* batched version */
		 
#ifdef HAVE_MAGMA
			double tic = SuperLU_timer_();
			dBatchFactorize_Handle batch_ws = dgetBatchFactorizeWorkspace(
			    nsupers, ldt, trf3Dpartition, LUstruct, grid3d, options, stat, info);
			if (gpu_memory_profile_requested)
				superlu_gpu_memory_tracker_sample();

			double setup_time = SuperLU_timer_() - tic;

			int maxLvl = log2i(grid3d->zscp.Np) + 1;

			tic = SuperLU_timer_();
			for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {
			    if (!trf3Dpartition->myZeroTrIdxs[ilvl]) {
				sForest_t *sforest = trf3Dpartition->sForests[trf3Dpartition->myTreeIdxs[ilvl]];
				if (sforest)
					dsparseTreeFactorBatchGPU(batch_ws, sforest);
			     }
			    if (gpu_memory_profile_requested)
				superlu_gpu_memory_tracker_sample();
			}
			double factor_time = SuperLU_timer_() - tic;
			if (gpu_memory_profile_requested) {
				superlu_gpu_memory_tracker_mark_factor_end();
				MPI_Barrier(grid3d->comm);
			}

			tic = SuperLU_timer_();
			dcopyGPULUDataToHost(batch_ws, LUstruct, grid3d, SCT, options, stat);
			dfreeBatchFactorizeWorkspace(batch_ws);
			double transfer_time = SuperLU_timer_() - tic;
			double total_time = transfer_time + factor_time + setup_time;
#if ( PRNTlevel >= 1 )
			printf("Batch Setup time = %.4f (%.2f %% of total)\n", setup_time, 100 * setup_time / total_time);
			printf("Batch Factorization time = %.4f (%.2f %% of total)\n", factor_time, 100 * factor_time / total_time);
			printf("Transfer time = %.4f (%.2f %% of total)\n", transfer_time, 100 * transfer_time / total_time);
			printf("Total time = %.4f\n", total_time);
#endif
#else // no MAGMA
			// TODO: How should we handle this?
			ABORT("Fatal error: Batched mode requires magma support!\n");
#endif 
	      	   } /* end if batchCount == 0 */
		 
		    // print other stuff
		    // if (!grid3d->zscp.Iam)
		    // 	SCT_printSummary(grid, SCT);
		    reduceStat(FACT, stat, grid3d);
		}
		else /* gpu3dVersion==0, this is the old C code, with less GPU offload */
#endif /* matching ifdef GPU_ACC */
		{
#ifndef GPU_ACC
			if (gpu3dVersion == 2 && !use_sym_v2_solve)
				ABORT("GPU3DVERSION=2 requires SymFact=YES.");
			if (use_sym_v2_solve) {
				if (options->batchCount != 0)
					ABORT("SymFact V2 CPU backend does not support batchCount>0.");
				int ldt = sp_ienv_dist(3, options);
				double s_eps = smach_dist("Epsilon");
				double thresh = s_eps * anorm;
				dLUgpu_Handle dLUcpu = dCreateLUgpuHandle(
					nsupers, ldt, trf3Dpartition, LUstruct, grid3d,
					SCT, options, stat, thresh, info);
				pdgstrf3d_LUv2(dLUcpu);
				if (nrhs > 0 && *info == 0) {
					SOLVEstruct->symldl_v2_factor_handle = (void *) dLUcpu;
					dLUcpu = NULL;
				} else {
					dCopyLUGPU2Host(dLUcpu, LUstruct);
				}
				if (dLUcpu != NULL)
					dDestroyLUgpuHandle(dLUcpu);
				reduceStat(FACT, stat, grid3d);
			} else
#endif
			{
				pdgstrf3d(options, m, n, anorm, trf3Dpartition, SCT,
					  LUstruct, grid3d, stat, info);
			}

			// dDumpLblocks3D(nsupers, grid3d, LUstruct->Glu_persist, LUstruct->Llu);
		}
			double numeric_factor_local = SuperLU_timer_() - t;
#ifdef GPU_ACC
			if (gpu_memory_profile_requested)
				superlu_gpu_memory_tracker_stop(&gpu_memory_stats);
#endif
			double numeric_factor_max = 0.0;
			MPI_Reduce(&numeric_factor_local, &numeric_factor_max, 1,
				   MPI_DOUBLE, MPI_MAX, 0, grid3d->comm);
			if (grid3d->iam == 0) {
				printf("NUMERIC_FACTOR time %12.6f\n", numeric_factor_max);
				fflush(stdout);
			}
			dPrintFactorCommProfile(
				SCT, grid3d,
				use_sym_v2_solve
					? "symldl-v2"
					: (options->SymFact == YES ? "symmetric-u"
						                           : "unsymmetric-lu"));
#ifdef GPU_ACC
			if (gpu_memory_profile_requested) {
				const char *gpu_memory_backend =
					use_sym_v2_solve
						? "symldl-v2"
						: (options->SymFact == YES ? "symmetric-lu"
							                           : "unsymmetric-lu");
				dPrintGpuFactorMemoryProfile(
					&gpu_memory_stats, grid3d, gpu_memory_backend);
			}
#endif
		} // matching if not SolveOnly ... end Factorization

	/* Now proceed with the Solve setup */
		if (get_new3dsolve() && !use_sym_v2_solve){
			dbroadcastAncestor3d(trf3Dpartition, LUstruct, grid3d, SCT);
		}

		if ( options->Fact != SamePattern_SameRowPerm && !use_sym_v2_solve) {
			if (get_new3dsolve() && Solve3D==true){
				dtrs_compute_communication_structure(options, n, LUstruct,
							ScalePermstruct, trf3Dpartition->supernodeMask, grid, stat);
			}else{
				int* supernodeMask = int32Malloc_dist(nsupers);
				for(int ii=0; ii<nsupers; ii++)
					supernodeMask[ii]=1;
				dtrs_compute_communication_structure(options, n, LUstruct,
							ScalePermstruct, supernodeMask, grid, stat);
				SUPERLU_FREE(supernodeMask);
			}
		}


		stat->utime[FACT] = SuperLU_timer_() - t;

		/*factorize in grid 1*/
		// if(grid3d->zscp.Iam)
		double tgather = SuperLU_timer_();
		if(Solve3D==false){
		dgatherAllFactoredLU(trf3Dpartition, LUstruct, grid3d, SCT);
		}
		SCT->gatherLUtimer += SuperLU_timer_() - tgather;
		/*print stats for bottom grid*/

		// Write LU to file
		int writeLU = 0;
		if (getenv("WRITELU"))
		{
			writeLU = atoi(getenv("WRITELU"));
		}

		if (writeLU)
		{
			if (!grid3d->zscp.Iam)
				dwriteLUtoDisk(nsupers, LUstruct->Glu_persist->xsup, LUstruct);
		}

		int checkLU = 0;
		if (getenv("CHECKLU"))
		{
			checkLU = atoi(getenv("CHECKLU"));
		}

		if (checkLU)
		{
			if (!grid3d->zscp.Iam)
				dcheckLUFromDisk(nsupers, LUstruct->Glu_persist->xsup, LUstruct);
		}

#if (PRNTlevel >= 1)
		if (!grid3d->zscp.Iam)
		{
			slu_SCT_print(grid, SCT);
			slu_SCT_print3D(grid3d, SCT);
		}
		slu_SCT_printComm3D(grid3d, SCT);

		/*print memory usage*/
		d3D_printMemUse(trf3Dpartition, LUstruct, grid3d);

		SCT->gatherLUtimer += SuperLU_timer_() - tgather;
		/*print stats for bottom grid*/
		/*print forest weight and costs*/
		printForestWeightCost(trf3Dpartition->sForests, SCT, grid3d);
		/*reduces stat from all the layers*/
#endif

		slu_SCT_free(SCT);

	} /* end if not Factored ... factor on all process layers */

	if (use_sym_v2_solve && !factored && options->PrintStat) {
		dSymV2PrintFactorStats(n, LUstruct, trf3Dpartition, grid3d,
				       options, stat, info, parSymbFact,
				       flinfo, dist_mem_use, GA_mem_use,
				       symb_mem_usage, &num_mem_usage);
	}

	if (!use_sym_v2_solve) { /* Print factor statistics over the full 3D grid. */
		if (!factored)
		{
			if (options->PrintStat)
			{
				int_t TinyPivots,sytrf_2x2,inertia[3];
				float for_lu, total, avg, loc_max;
				float mem_stage[3];
				struct { float val; int rank; } local_struct, global_struct;
				int nprocs3d = grid3d->nprow * grid3d->npcol * grid3d->npdep;

				MPI_Reduce( &stat->TinyPivots, &TinyPivots, 1, mpi_int_t,
						   MPI_SUM, 0, grid3d->comm );

				MPI_Reduce( &stat->sytrf_2x2, &sytrf_2x2, 1, mpi_int_t,
						   MPI_SUM, 0, grid3d->comm );


				MPI_Reduce( &stat->inertia, &inertia, 3, mpi_int_t,
						   MPI_SUM, 0, grid3d->comm );

				/*-- Compute high watermark of all stages --*/
				if (parSymbFact == TRUE)
				{
					/* The memory used in the redistribution routine
				   includes the memory used for storing the symbolic
				   structure and the memory allocated for numerical
				   factorization */
					mem_stage[0] = (-flinfo); /* symbfact step */
					mem_stage[1] = (-dist_mem_use);      /* distribution step */
					loc_max = SUPERLU_MAX( mem_stage[0], mem_stage[1]);
					if (options->RowPerm != NOROWPERM)
						loc_max = SUPERLU_MAX(loc_max, GA_mem_use);
				}
				else
				{
					mem_stage[0] = symb_mem_usage.total + GA_mem_use; /* symbfact step */
					mem_stage[1] = symb_mem_usage.for_lu + dist_mem_use + num_mem_usage.for_lu;            /* distribution step */
					loc_max = SUPERLU_MAX(mem_stage[0], mem_stage[1] );
				}

				dQuerySpace_dist(n, LUstruct, grid, stat, &num_mem_usage);
				mem_stage[2] = num_mem_usage.total;  /* numerical factorization step */

				loc_max = SUPERLU_MAX(loc_max, mem_stage[2] ); /* local max of 3 stages */

				local_struct.val = loc_max;
				local_struct.rank = grid3d->iam;
				MPI_Reduce( &local_struct, &global_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, grid3d->comm );
				int all_highmark_rank = 0;
				float all_highmark_mem = 0.;
				if (grid3d->iam == 0) {
					all_highmark_rank = global_struct.rank;
					all_highmark_mem = global_struct.val * 1e-6;
				}

				MPI_Reduce( &loc_max, &avg,
						   1, MPI_FLOAT, MPI_SUM, 0, grid3d->comm );
				MPI_Reduce( &num_mem_usage.for_lu, &for_lu,
						   1, MPI_FLOAT, MPI_SUM, 0, grid3d->comm );
				MPI_Reduce( &num_mem_usage.total, &total,
						   1, MPI_FLOAT, MPI_SUM, 0, grid3d->comm );

				/*-- Compute memory usage of numerical factorization --*/
				local_struct.val = num_mem_usage.for_lu;
				MPI_Reduce(&local_struct, &global_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, grid3d->comm);
				int lu_max_rank = 0;
				float lu_max_mem = 0.;
				if (grid3d->iam == 0) {
					lu_max_rank = global_struct.rank;
					lu_max_mem = global_struct.val * 1e-6;
				}

				local_struct.val = stat->peak_buffer;
				MPI_Reduce( &local_struct, &global_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, grid3d->comm );
				int buffer_peak_rank = 0;
				float buffer_peak = 0.;
				if (grid3d->iam == 0) {
					buffer_peak_rank = global_struct.rank;
					buffer_peak = global_struct.val * 1e-6;
				}
				if (grid3d->iam == 0)
				{
					stat->TinyPivots = TinyPivots;
					stat->sytrf_2x2 = sytrf_2x2;
					stat->inertia[0] = inertia[0];
					stat->inertia[1] = inertia[1];
					stat->inertia[2] = inertia[2];
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
						   all_highmark_rank, mem_stage[0] * 1e-6, mem_stage[1] * 1e-6, mem_stage[2] * 1e-6);
					printf("** NUMfact space (MB): (sum-of-all-processes)\n"
						   "    L\\U :        %8.2f |  Total : %8.2f\n",
						   for_lu * 1e-6, total * 1e-6);
					printf("\t. max at rank %d, max L+U memory (MB): %8.2f\n"
						   "\t. max at rank %d, peak buffer (MB):    %8.2f\n",
						   lu_max_rank, lu_max_mem,
						   buffer_peak_rank, buffer_peak);
					printf("**************************************************\n\n");
					printf("** number of Tiny Pivots: %8d\n\n", stat->TinyPivots);
					printf("** number of 2x2 Pivots by sytrf: %8d\n\n", stat->sytrf_2x2);
					printf("** Inertia (pos,neg,zero): %10d %10d %10d\n\n", stat->inertia[0],stat->inertia[1],stat->inertia[2]);
					printf("info %10d\n",*info);
					fflush(stdout);
				}
			} /* end printing stats */

		} /* end if !factored */
	} /* end if non-V2 factor statistics */

		if(Solve3D){

			if ( options->Fact == DOFACT || options->Fact == SamePattern ) {
			/* Need to reset the solve's communication pattern,
			because perm_r[] and/or perm_c[] is changed.    */
			if ( options->SolveInitialized == YES ) { /* Initialized before */
				void *sym_v2_factor_handle =
					use_sym_v2_solve ? SOLVEstruct->symldl_v2_factor_handle : NULL;
				if (use_sym_v2_solve)
					SOLVEstruct->symldl_v2_factor_handle = NULL;
				dSolveFinalize(options, SOLVEstruct); /* Clean up structure */
				if (use_sym_v2_solve)
					SOLVEstruct->symldl_v2_factor_handle = sym_v2_factor_handle;
				pdgstrs_delete_device_lsum_x(SOLVEstruct);
				options->SolveInitialized = NO;   /* Reset the solve state */
			}
			}

			if (get_new3dsolve() && !use_sym_v2_solve){


			if (options->DiagInv == YES && (Fact != FACTORED))
			{
				pdCompute_Diag_Inv(options, n, LUstruct, grid, stat, info);

				// The following #ifdef GPU_ACC block frees and reallocates GPU data for trisolve. The data seems to be overwritten by pdgstrf3d.
				int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
#if defined(GPU_ACC)

				pdconvertU(options, grid, LUstruct, stat, n);

				// checkGPU(gpuFree(LUstruct->Llu->d_xsup));
				// checkGPU(gpuFree(LUstruct->Llu->d_bcols_masked));
				// checkGPU(gpuFree(LUstruct->Llu->d_LRtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_LBtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_URtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_UBtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lrowind_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lindval_loc_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lrowind_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lindval_loc_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lnzval_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Linv_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Uinv_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_ilsum));
				// checkGPU(gpuFree(LUstruct->Llu->d_grid));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lnzval_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Linv_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Uinv_bc_dat));

				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_xsup, (n + 1) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_xsup, LUstruct->Glu_persist->xsup, (n + 1) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc( (void**)&LUstruct->Llu->d_bcols_masked, LUstruct->Llu->nbcol_masked * sizeof(int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_bcols_masked, LUstruct->Llu->bcols_masked, LUstruct->Llu->nbcol_masked * sizeof(int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_LRtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_LBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_URtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_UBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_LRtree_ptr, LUstruct->Llu->LRtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_LBtree_ptr, LUstruct->Llu->LBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_URtree_ptr, LUstruct->Llu->URtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_UBtree_ptr, LUstruct->Llu->UBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lrowind_bc_dat, (LUstruct->Llu->Lrowind_bc_cnt) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lrowind_bc_dat, LUstruct->Llu->Lrowind_bc_dat, (LUstruct->Llu->Lrowind_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lindval_loc_bc_dat, (LUstruct->Llu->Lindval_loc_bc_cnt) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lindval_loc_bc_dat, LUstruct->Llu->Lindval_loc_bc_dat, (LUstruct->Llu->Lindval_loc_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lrowind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lrowind_bc_offset, LUstruct->Llu->Lrowind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lindval_loc_bc_offset, LUstruct->Llu->Lindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lnzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lnzval_bc_offset, LUstruct->Llu->Lnzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Linv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Linv_bc_offset, LUstruct->Llu->Linv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Uinv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Uinv_bc_offset, LUstruct->Llu->Uinv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_ilsum, (CEILING(nsupers, grid->nprow) + 1) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_ilsum, LUstruct->Llu->ilsum, (CEILING(nsupers, grid->nprow) + 1) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lnzval_bc_dat, (LUstruct->Llu->Lnzval_bc_cnt) * sizeof(double)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Linv_bc_dat, (LUstruct->Llu->Linv_bc_cnt) * sizeof(double)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Uinv_bc_dat, (LUstruct->Llu->Uinv_bc_cnt) * sizeof(double)));
				// checkGPU(gpuMalloc( (void**)&LUstruct->Llu->d_grid, sizeof(gridinfo_t)));
    			// checkGPU(gpuMemcpy(LUstruct->Llu->d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));
#endif
if (!use_sym_v2_solve && get_acc_solve()){
#ifdef GPU_ACC
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Linv_bc_dat, LUstruct->Llu->Linv_bc_dat,
								   (LUstruct->Llu->Linv_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Uinv_bc_dat, LUstruct->Llu->Uinv_bc_dat,
								   (LUstruct->Llu->Uinv_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Lnzval_bc_dat, LUstruct->Llu->Lnzval_bc_dat,
								   (LUstruct->Llu->Lnzval_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));
#endif
}
			}
			}
		} else { /* else if(Solve3D) */

			if (grid3d->zscp.Iam == 0){  /* on 2D grid-0 */

			if ( options->Fact == DOFACT || options->Fact == SamePattern ) {
			/* Need to reset the solve's communication pattern,
			because perm_r[] and/or perm_c[] is changed.    */
			if ( options->SolveInitialized == YES ) { /* Initialized before */
				void *sym_v2_factor_handle =
					use_sym_v2_solve ? SOLVEstruct->symldl_v2_factor_handle : NULL;
				if (use_sym_v2_solve)
					SOLVEstruct->symldl_v2_factor_handle = NULL;
				dSolveFinalize(options, SOLVEstruct); /* Clean up structure */
				if (use_sym_v2_solve)
					SOLVEstruct->symldl_v2_factor_handle = sym_v2_factor_handle;
				pdgstrs_delete_device_lsum_x(SOLVEstruct);
				options->SolveInitialized = NO;   /* Reset the solve state */
			}
			}

#if (defined(GPU_ACC))
			if (options->DiagInv == NO && get_acc_solve())
			{
				if (iam == 0)
				{
					printf("!!WARNING: GPU trisolve requires setting options->DiagInv==YES\n");
					printf("           otherwise, use CPU trisolve\n");
					fflush(stdout);
				}
				// exit(0);  // Sherry: need to return an error flag
			}
#endif

			if (options->DiagInv == YES && (Fact != FACTORED))
			{
				pdCompute_Diag_Inv(options, n, LUstruct, grid, stat, info);

				// The following #ifdef GPU_ACC block frees and reallocates GPU data for trisolve. The data seems to be overwritten by pdgstrf3d.
				int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
#ifdef GPU_ACC

				pdconvertU(options, grid, LUstruct, stat, n);

				// checkGPU(gpuFree(LUstruct->Llu->d_xsup));
				// checkGPU(gpuFree(LUstruct->Llu->d_bcols_masked));
				// checkGPU(gpuFree(LUstruct->Llu->d_LRtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_LBtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_URtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_UBtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lrowind_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lindval_loc_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lrowind_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lindval_loc_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lnzval_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Linv_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Uinv_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_ilsum));
				// checkGPU(gpuFree(LUstruct->Llu->d_grid));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lnzval_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Linv_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Uinv_bc_dat));

				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_xsup, (n + 1) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_xsup, LUstruct->Glu_persist->xsup, (n + 1) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc( (void**)&LUstruct->Llu->d_bcols_masked, LUstruct->Llu->nbcol_masked * sizeof(int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_bcols_masked, LUstruct->Llu->bcols_masked, LUstruct->Llu->nbcol_masked * sizeof(int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_LRtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_LBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_URtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_UBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_LRtree_ptr, LUstruct->Llu->LRtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_LBtree_ptr, LUstruct->Llu->LBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_URtree_ptr, LUstruct->Llu->URtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_UBtree_ptr, LUstruct->Llu->UBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lrowind_bc_dat, (LUstruct->Llu->Lrowind_bc_cnt) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lrowind_bc_dat, LUstruct->Llu->Lrowind_bc_dat, (LUstruct->Llu->Lrowind_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lindval_loc_bc_dat, (LUstruct->Llu->Lindval_loc_bc_cnt) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lindval_loc_bc_dat, LUstruct->Llu->Lindval_loc_bc_dat, (LUstruct->Llu->Lindval_loc_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lrowind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lrowind_bc_offset, LUstruct->Llu->Lrowind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lindval_loc_bc_offset, LUstruct->Llu->Lindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lnzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lnzval_bc_offset, LUstruct->Llu->Lnzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Linv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Linv_bc_offset, LUstruct->Llu->Linv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Uinv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Uinv_bc_offset, LUstruct->Llu->Uinv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_ilsum, (CEILING(nsupers, grid->nprow) + 1) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_ilsum, LUstruct->Llu->ilsum, (CEILING(nsupers, grid->nprow) + 1) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lnzval_bc_dat, (LUstruct->Llu->Lnzval_bc_cnt) * sizeof(double)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Linv_bc_dat, (LUstruct->Llu->Linv_bc_cnt) * sizeof(double)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Uinv_bc_dat, (LUstruct->Llu->Uinv_bc_cnt) * sizeof(double)));
				// checkGPU(gpuMalloc( (void**)&LUstruct->Llu->d_grid, sizeof(gridinfo_t)));
    			// checkGPU(gpuMemcpy(LUstruct->Llu->d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));
#endif

if (!use_sym_v2_solve && get_acc_solve()){
#ifdef GPU_ACC

				checkGPU(gpuMemcpy(LUstruct->Llu->d_Linv_bc_dat, LUstruct->Llu->Linv_bc_dat,
								   (LUstruct->Llu->Linv_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Uinv_bc_dat, LUstruct->Llu->Uinv_bc_dat,
								   (LUstruct->Llu->Uinv_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Lnzval_bc_dat, LUstruct->Llu->Lnzval_bc_dat,
								   (LUstruct->Llu->Lnzval_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));
#endif
}
			}
			}
		}


		/* ------------------------------------------------------------
		   Compute the solution matrix X.
		   ------------------------------------------------------------ */
		if ((nrhs > 0) && (*info == 0))
		{
		if (options->SolveInitialized == NO){
			if (!use_sym_v2_solve && get_acc_solve()){
			if (get_new3dsolve() && Solve3D==true){
				pdgstrs_init_device_lsum_x(options, n, m_loc, nrhs, grid,LUstruct, SOLVEstruct,trf3Dpartition->supernodeMask);
			}else{
				int* supernodeMask = int32Malloc_dist(nsupers);
				for(int ii=0; ii<nsupers; ii++)
					supernodeMask[ii]=1;
				pdgstrs_init_device_lsum_x(options, n, m_loc, nrhs, grid,LUstruct, SOLVEstruct,supernodeMask);
				SUPERLU_FREE(supernodeMask);
			}
			}
		}

		stat->utime[SOLVE] = 0.0;
		if(Solve3D){

			// if (!(b_work = doubleMalloc_dist(n)))
			// 	ABORT("Malloc fails for b_work[]");
			/* ------------------------------------------------------
			   Scale the right-hand side if equilibration was performed
			   ------------------------------------------------------*/
			if (notran)
			{
				if (rowequ)
				{
					b_col = B;
					for (j = 0; j < nrhs; ++j)
					{
						irow = fst_row;
						for (i = 0; i < m_loc; ++i)
						{
							b_col[i] *= R[irow];
							++irow;
						}
						b_col += ldb;
					}
				}
			}
			else if (colequ)
			{
				b_col = B;
				for (j = 0; j < nrhs; ++j)
				{
					irow = fst_row;
					for (i = 0; i < m_loc; ++i)
					{
						b_col[i] *= C[irow];
						++irow;
					}
					b_col += ldb;
				}
			}

			/* Save a copy of the right-hand side. */
			ldx = ldb;
			if (!(X = doubleMalloc_dist(((size_t)ldx) * nrhs)))
				ABORT("Malloc fails for X[]");
			x_col = X;
			b_col = B;
			for (j = 0; j < nrhs; ++j)
			{
				for (i = 0; i < m_loc; ++i)
					x_col[i] = b_col[i];
				x_col += ldx;
				b_col += ldb;
			}

			/* ------------------------------------------------------
			   Solve the linear system.
			   ------------------------------------------------------*/

			if (options->SolveInitialized == NO)
			/* First time */
			/* Inside this routine, SolveInitialized is set to YES.
			For repeated call to pdgssvx3d(), no need to re-initialilze
			the Solve data & communication structures, unless a new
			factorization with Fact == DOFACT or SamePattern is asked for. */
			{
				if (use_sym_v2_solve)
					dSymV2SolveInit(options, A, perm_r, perm_c, nrhs,
							LUstruct, trf3Dpartition, grid3d, SOLVEstruct);
				else
					dSolveInit(options, A, perm_r, perm_c, nrhs, LUstruct,
							grid, SOLVEstruct);
			}
			if (use_sym_v2_solve) {
				pdgstrs3d_symldl (options, n, LUstruct,ScalePermstruct, trf3Dpartition, grid3d, X,
				m_loc, fst_row, ldb, nrhs,SOLVEstruct, stat, info);
			} else if (get_new3dsolve()){
				pdgstrs3d_newsolve (options, n, LUstruct,ScalePermstruct, trf3Dpartition, grid3d, X,
				m_loc, fst_row, ldb, nrhs,SOLVEstruct, stat, info);
			}else{
				pdgstrs3d (options, n, LUstruct,ScalePermstruct, trf3Dpartition, grid3d, X,
				m_loc, fst_row, ldb, nrhs,SOLVEstruct, stat, info);
			}
			if (options->IterRefine)
				{
				/* Improve the solution by iterative refinement. */
				int_t *it, *colind_gsmv = SOLVEstruct->A_colind_gsmv;
				dSOLVEstruct_t *SOLVEstruct1; /* Used by refinement */

				t = SuperLU_timer_ ();
				if (options->RefineInitialized == NO || Fact == DOFACT) {
					/* All these cases need to re-initialize gsmv structure */
					if (options->RefineInitialized)
					pdgsmv_finalize (SOLVEstruct->gsmv_comm);
					pdgsmv_init (A, SOLVEstruct->row_to_proc, grid,
						SOLVEstruct->gsmv_comm);

					/* Save a copy of the transformed local col indices
					in colind_gsmv[]. */
					if (colind_gsmv) SUPERLU_FREE (colind_gsmv);
					if (!(it = intMalloc_dist (nnz_loc)))
					ABORT ("Malloc fails for colind_gsmv[]");
					colind_gsmv = SOLVEstruct->A_colind_gsmv = it;
					for (i = 0; i < nnz_loc; ++i) colind_gsmv[i] = colind[i];
					options->RefineInitialized = YES;
				}
				else if (Fact == SamePattern || Fact == SamePattern_SameRowPerm) {
					double at;
					int_t k, jcol, p;
					/* Swap to beginning the part of A corresponding to the
					local part of X, as was done in pdgsmv_init() */
					for (i = 0; i < m_loc; ++i) { /* Loop through each row */
					k = rowptr[i];
					for (j = rowptr[i]; j < rowptr[i + 1]; ++j)
						{
						jcol = colind[j];
						p = SOLVEstruct->row_to_proc[jcol];
						if (p == iam)
							{	/* Local */
							at = a[k];
							a[k] = a[j];
							a[j] = at;
							++k;
							}
						}
					}

					/* Re-use the local col indices of A obtained from the
					previous call to pdgsmv_init() */
					for (i = 0; i < nnz_loc; ++i)
					colind[i] = colind_gsmv[i];
				}

				if (nrhs == 1)
					{	/* Use the existing solve structure */
					SOLVEstruct1 = SOLVEstruct;
					}
				else {
				/* For nrhs > 1, since refinement is performed for RHS
			one at a time, the communication structure for pdgstrs
			is different than the solve with nrhs RHS.
			So we use SOLVEstruct1 for the refinement step.
			*/
					if (!(SOLVEstruct1 = (dSOLVEstruct_t *)
						SUPERLU_MALLOC(sizeof(dSOLVEstruct_t))))
						ABORT ("Malloc fails for SOLVEstruct1");
					/* Copy the same stuff */
					SOLVEstruct1->row_to_proc = SOLVEstruct->row_to_proc;
					SOLVEstruct1->inv_perm_c = SOLVEstruct->inv_perm_c;
					SOLVEstruct1->num_diag_procs = SOLVEstruct->num_diag_procs;
					SOLVEstruct1->diag_procs = SOLVEstruct->diag_procs;
					SOLVEstruct1->diag_len = SOLVEstruct->diag_len;
					SOLVEstruct1->gsmv_comm = SOLVEstruct->gsmv_comm;
					SOLVEstruct1->A_colind_gsmv = SOLVEstruct->A_colind_gsmv;
					SOLVEstruct1->symldl_v2_solve_meta = NULL;
					SOLVEstruct1->symldl_v2_factor_handle = NULL;

					/* Initialize the *gstrs_comm for 1 RHS. */
					if (!(SOLVEstruct1->gstrs_comm = (pxgstrs_comm_t *)
						SUPERLU_MALLOC (sizeof (pxgstrs_comm_t))))
						ABORT ("Malloc fails for gstrs_comm[]");
					pdgstrs_init (n, m_loc, 1, fst_row, perm_r, perm_c, grid,
							LUstruct->Glu_persist, SOLVEstruct1);
					if (!use_sym_v2_solve && get_acc_solve()){
					int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
					pdgstrs_init_device_lsum_x(options, n, m_loc, 1, grid,LUstruct, SOLVEstruct1,trf3Dpartition->supernodeMask);
					}
					}

				pdgsrfs3d (options, n, A, anorm, LUstruct, ScalePermstruct, grid3d, trf3Dpartition,
					B, ldb, X, ldx, nrhs, SOLVEstruct1, berr, stat, info);

				/* Deallocate the storage associated with SOLVEstruct1 */
				if (nrhs > 1)
					{
					pdgstrs_delete_device_lsum_x(SOLVEstruct1);
					pxgstrs_finalize (SOLVEstruct1->gstrs_comm);
					SUPERLU_FREE (SOLVEstruct1);
					}

				stat->utime[REFINE] = SuperLU_timer_ () - t;
				} /* end IterRefine */
		}else{

			if (grid3d->zscp.Iam == 0){  /* on 2D grid-0 */

			/* ------------------------------------------------------
			   Scale the right-hand side if equilibration was performed
			   ------------------------------------------------------*/
			if (notran)
			{
				if (rowequ)
				{
					b_col = B;
					for (j = 0; j < nrhs; ++j)
					{
						irow = fst_row;
						for (i = 0; i < m_loc; ++i)
						{
							b_col[i] *= R[irow];
							++irow;
						}
						b_col += ldb;
					}
				}
			}
			else if (colequ)
			{
				b_col = B;
				for (j = 0; j < nrhs; ++j)
				{
					irow = fst_row;
					for (i = 0; i < m_loc; ++i)
					{
						b_col[i] *= C[irow];
						++irow;
					}
					b_col += ldb;
				}
			}

			/* Save a copy of the right-hand side. */
			ldx = ldb;
			if (!(X = doubleMalloc_dist(((size_t)ldx) * nrhs)))
				ABORT("Malloc fails for X[]");
			x_col = X;
			b_col = B;
			for (j = 0; j < nrhs; ++j)
			{
				for (i = 0; i < m_loc; ++i)
					x_col[i] = b_col[i];
				x_col += ldx;
				b_col += ldb;
			}

			/* ------------------------------------------------------
			   Solve the linear system.
			   ------------------------------------------------------*/
			if (options->SolveInitialized == NO)
			/* First time */
			/* Inside this routine, SolveInitialized is set to YES.
			For repeated call to pdgssvx3d(), no need to re-initialilze
			the Solve data & communication structures, unless a new
			factorization with Fact == DOFACT or SamePattern is asked for. */
			{
				dSolveInit(options, A, perm_r, perm_c, nrhs, LUstruct,
							grid, SOLVEstruct);
			}
			pdgstrs(options, n, LUstruct, ScalePermstruct, grid, X, m_loc,
				fst_row, ldb, nrhs, SOLVEstruct, stat, info);

			/* ------------------------------------------------------------
			Use iterative refinement to improve the computed solution and
			compute error bounds and backward error estimates for it.
			------------------------------------------------------------ */
			if (options->IterRefine)
				{
				/* Improve the solution by iterative refinement. */
				int_t *it, *colind_gsmv = SOLVEstruct->A_colind_gsmv;
				dSOLVEstruct_t *SOLVEstruct1; /* Used by refinement */

				t = SuperLU_timer_ ();
				if (options->RefineInitialized == NO || Fact == DOFACT) {
					/* All these cases need to re-initialize gsmv structure */
					if (options->RefineInitialized)
					pdgsmv_finalize (SOLVEstruct->gsmv_comm);
					pdgsmv_init (A, SOLVEstruct->row_to_proc, grid,
						SOLVEstruct->gsmv_comm);

					/* Save a copy of the transformed local col indices
					in colind_gsmv[]. */
					if (colind_gsmv) SUPERLU_FREE (colind_gsmv);
					if (!(it = intMalloc_dist (nnz_loc)))
					ABORT ("Malloc fails for colind_gsmv[]");
					colind_gsmv = SOLVEstruct->A_colind_gsmv = it;
					for (i = 0; i < nnz_loc; ++i) colind_gsmv[i] = colind[i];
					options->RefineInitialized = YES;
				}
				else if (Fact == SamePattern || Fact == SamePattern_SameRowPerm) {
					double at;
					int_t k, jcol, p;
					/* Swap to beginning the part of A corresponding to the
					local part of X, as was done in pdgsmv_init() */
					for (i = 0; i < m_loc; ++i) { /* Loop through each row */
					k = rowptr[i];
					for (j = rowptr[i]; j < rowptr[i + 1]; ++j)
						{
						jcol = colind[j];
						p = SOLVEstruct->row_to_proc[jcol];
						if (p == iam)
							{	/* Local */
							at = a[k];
							a[k] = a[j];
							a[j] = at;
							++k;
							}
						}
					}

					/* Re-use the local col indices of A obtained from the
					previous call to pdgsmv_init() */
					for (i = 0; i < nnz_loc; ++i)
					colind[i] = colind_gsmv[i];
				}

				if (nrhs == 1)
					{	/* Use the existing solve structure */
					SOLVEstruct1 = SOLVEstruct;
					}
				else {
				/* For nrhs > 1, since refinement is performed for RHS
			one at a time, the communication structure for pdgstrs
			is different than the solve with nrhs RHS.
			So we use SOLVEstruct1 for the refinement step.
			*/
					if (!(SOLVEstruct1 = (dSOLVEstruct_t *)
						SUPERLU_MALLOC(sizeof(dSOLVEstruct_t))))
						ABORT ("Malloc fails for SOLVEstruct1");
					/* Copy the same stuff */
					SOLVEstruct1->row_to_proc = SOLVEstruct->row_to_proc;
					SOLVEstruct1->inv_perm_c = SOLVEstruct->inv_perm_c;
					SOLVEstruct1->num_diag_procs = SOLVEstruct->num_diag_procs;
					SOLVEstruct1->diag_procs = SOLVEstruct->diag_procs;
					SOLVEstruct1->diag_len = SOLVEstruct->diag_len;
					SOLVEstruct1->gsmv_comm = SOLVEstruct->gsmv_comm;
					SOLVEstruct1->A_colind_gsmv = SOLVEstruct->A_colind_gsmv;
					SOLVEstruct1->symldl_v2_solve_meta = NULL;
					SOLVEstruct1->symldl_v2_factor_handle = NULL;

					/* Initialize the *gstrs_comm for 1 RHS. */
					if (!(SOLVEstruct1->gstrs_comm = (pxgstrs_comm_t *)
						SUPERLU_MALLOC (sizeof (pxgstrs_comm_t))))
						ABORT ("Malloc fails for gstrs_comm[]");
					pdgstrs_init (n, m_loc, 1, fst_row, perm_r, perm_c, grid,
							LUstruct->Glu_persist, SOLVEstruct1);
					if (!use_sym_v2_solve && get_acc_solve()){
					int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
					int* supernodeMask = int32Malloc_dist(nsupers);
					for(int ii=0; ii<nsupers; ii++)
						supernodeMask[ii]=1;
					pdgstrs_init_device_lsum_x(options, n, m_loc, 1, grid,LUstruct, SOLVEstruct1,supernodeMask);
					SUPERLU_FREE(supernodeMask);
					}
					}

				pdgsrfs (options, n, A, anorm, LUstruct, ScalePermstruct, grid,
					B, ldb, X, ldx, nrhs, SOLVEstruct1, berr, stat, info);

				/* Deallocate the storage associated with SOLVEstruct1 */
				if (nrhs > 1)
					{
					pdgstrs_delete_device_lsum_x(SOLVEstruct1);
					pxgstrs_finalize (SOLVEstruct1->gstrs_comm);
					SUPERLU_FREE (SOLVEstruct1);
					}

				stat->utime[REFINE] = SuperLU_timer_ () - t;
				} /* end IterRefine */
			}
		}

if (grid3d->zscp.Iam == 0)  /* on 2D grid-0 */
	{
		/* Permute the solution matrix B <= Pc'*X. */
		pdPermute_Dense_Matrix (fst_row, m_loc, SOLVEstruct->row_to_proc,
					SOLVEstruct->inv_perm_c,
					X, ldx, B, ldb, nrhs, grid);
#if ( DEBUGlevel>=2 )
		printf ("\n (%d) .. After pdPermute_Dense_Matrix(): b =\n", iam);
		for (i = 0; i < m_loc; ++i)
		    printf ("\t(%d)\t%4d\t%.10f\n", iam, i + fst_row, B[i]);
#endif
			/* Transform the solution matrix X to a solution of the original
			   system before the equilibration. */
			if (notran)
			{
				if (colequ)
				{
					b_col = B;
					for (j = 0; j < nrhs; ++j)
					{
						irow = fst_row;
						for (i = 0; i < m_loc; ++i)
						{
							b_col[i] *= C[irow];
							++irow;
						}
						b_col += ldb;
				    }
			    }
		    }
			else if (rowequ)
		    {
			b_col = B;
			for (j = 0; j < nrhs; ++j)
			    {
				irow = fst_row;
				for (i = 0; i < m_loc; ++i)
				    {
					b_col[i] *= R[irow];
					++irow;
				    }
				b_col += ldb;
			    }
		    }

		// SUPERLU_FREE (b_work);
	}
	if (grid3d->zscp.Iam == 0 || Solve3D)
		SUPERLU_FREE (X);

	} /* end if nrhs > 0 and factor successful */

#if ( PRNTlevel>=1 )
	if (!grid3d->iam) {
	    printf (".. DiagScale = %d\n", ScalePermstruct->DiagScale);
        }
#endif


	if ( grid3d->zscp.Iam == 0 ) { // only process layer 0
	/* Deallocate R and/or C if it was not used. */
	if (Equil && Fact != SamePattern_SameRowPerm)
	    {
		switch (ScalePermstruct->DiagScale) {
		    case NOEQUIL:
			SUPERLU_FREE (R);
			SUPERLU_FREE (C);
			break;
		    case ROW:
			SUPERLU_FREE (C);
			break;
		    case COL:
			SUPERLU_FREE (R);
			break;
	            default: break;
		}
	}
    if (SLU_IS_SYMATCH_ROWPERM(options->RowPerm)) {
	SUPERLU_FREE(options->indicator_2x2);
	options->indicator_2x2 = NULL;
    }
#if 0
	if (!factored && Fact != SamePattern_SameRowPerm && !parSymbFact)
	    Destroy_CompCol_Permuted_dist (&GAC);
#endif

	} /* process layer 0 done solve */

	/* Scatter the solution from 2D grid-0 to 3D grid */
	if (nrhs > 0)
		dScatter_B3d(A3d, grid3d);

	B = A3d->B3d;		 // B is now assigned back to B3d on return
	A->Store = Astore3d; // restore Astore to 3D

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(iam, "Exit pdgssvx3d()");
#endif
}
