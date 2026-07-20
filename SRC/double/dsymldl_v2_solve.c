/*! @file
 * \brief LDL-native solve for the SymFact GPU3D v2 factor path.
 */
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "superlu_ddefs.h"
#include "superlu_upacked.h"
#include "dsymldl_v2_cpu_solve.h"
#include "dsymldl_v2_solve_graph.h"
#include "dsymldl_v2_nvshmem_solve.h"

static size_t
pdgstrs3d_checked_product(size_t a, size_t b, const char *what)
{
    (void) what;
    if (a != 0 && b > ((size_t)-1) / a)
        ABORT("Workspace size overflows allocation size.");
    return a * b;
}

static int_t
pdgstrs3d_checked_workspace_count(int_t a, int_t b, int_t c, int_t d,
                                  const char *what)
{
    if (a < 0 || b < 0 || c < 0 || d < 0)
        ABORT("Negative workspace size.");

    size_t first = pdgstrs3d_checked_product((size_t) a, (size_t) b, what);
    size_t second = pdgstrs3d_checked_product((size_t) c, (size_t) d, what);
    if (first > ((size_t)-1) - second)
        ABORT("Workspace size overflows allocation size.");

    size_t total = first + second;
    int_t out = (int_t) total;
    if (out < 0 || (size_t) out != total)
        ABORT("Workspace size overflows int_t.");
    return out;
}

static int_t
pdgstrs3d_checked_size_to_int_t(size_t count, const char *what)
{
    (void) what;
    int_t out = (int_t) count;
    if (out < 0 || (size_t) out != count)
        ABORT("Workspace size overflows int_t.");
    return out;
}

static size_t
pdgstrs3d_checked_alloc_bytes(int_t count, size_t elem_size,
                              const char *what)
{
    if (count < 0)
        ABORT("Negative allocation size.");
    size_t n = (size_t) count;
    if ((int_t) n != count)
        ABORT("Allocation size overflows int_t.");
    return pdgstrs3d_checked_product(n, elem_size, what);
}

static int
pdgstrs3d_symv2_owner(dtrf3Dpartition_t *trf3Dpartition, int_t k)
{
    if (trf3Dpartition == NULL ||
        trf3Dpartition->symV2DiagOwner == NULL)
        ABORT("SymFact V2 solve redistribution requires owner tables.");
    return trf3Dpartition->symV2DiagOwner[k];
}

static int_t
pdgstrs3d_symv2_row_index(dtrf3Dpartition_t *trf3Dpartition, int_t k)
{
    int_t lk;
    if (trf3Dpartition == NULL ||
        trf3Dpartition->symV2RowLocalIndex == NULL)
        ABORT("SymFact V2 solve redistribution requires local row indexes.");
    lk = trf3Dpartition->symV2RowLocalIndex[k];
    if (lk < 0)
        ABORT("SymFact V2 solve redistribution missing local row index.");
    return lk;
}

typedef struct {
    int initialized;
    int procs;
    int nrhs;
    int b_send_count;
    int b_recv_count;
    int x_send_count;
    int x_recv_count;
    int_t *b_source_rows;
    int_t *b_recv_rows;
    int_t *x_source_offsets;
    int_t *x_source_widths;
    int_t *x_recv_rows;
    double *b_send_values;
    double *b_recv_values;
    double *x_send_values;
    double *x_recv_values;
} pdgstrs3d_symldl_redistribution_t;

static void
pdgstrs3d_symldl_redistribution_free(
    pdgstrs3d_symldl_redistribution_t *plan)
{
    if (plan == NULL)
        return;
    if (plan->x_recv_values) SUPERLU_FREE(plan->x_recv_values);
    if (plan->x_send_values) SUPERLU_FREE(plan->x_send_values);
    if (plan->b_recv_values) SUPERLU_FREE(plan->b_recv_values);
    if (plan->b_send_values) SUPERLU_FREE(plan->b_send_values);
    if (plan->x_recv_rows) SUPERLU_FREE(plan->x_recv_rows);
    if (plan->x_source_widths) SUPERLU_FREE(plan->x_source_widths);
    if (plan->x_source_offsets) SUPERLU_FREE(plan->x_source_offsets);
    if (plan->b_recv_rows) SUPERLU_FREE(plan->b_recv_rows);
    if (plan->b_source_rows) SUPERLU_FREE(plan->b_source_rows);
    memset(plan, 0, sizeof(*plan));
}

static int
pdgstrs3d_symldl_total_rows(const int *counts, const int *displs, int procs)
{
    if (procs <= 0)
        return 0;
    if (counts[procs - 1] < 0 || displs[procs - 1] < 0 ||
        displs[procs - 1] > INT_MAX - counts[procs - 1])
        ABORT("SymLDL redistribution row count overflows int.");
    return displs[procs - 1] + counts[procs - 1];
}

static void
pdgstrs3d_symldl_redistribution_create(
    pdgstrs3d_symldl_redistribution_t *plan,
    int_t n, int_t m_loc, int nrhs, int_t fst_row,
    dScalePermstruct_t *ScalePermstruct, Glu_persist_t *Glu_persist,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d,
    dSOLVEstruct_t *SOLVEstruct, int_t *ilsum)
{
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
    int procs;
    int *cursor;
    int_t *send_indices;
    int_t *perm_r = ScalePermstruct->perm_r;
    int_t *perm_c = ScalePermstruct->perm_c;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    int_t nsupers = supno[n - 1] + 1;

    if (plan == NULL || gstrs_comm == NULL || nrhs <= 0)
        ABORT("SymLDL redistribution setup is invalid.");
    if (plan->initialized) {
        if (plan->nrhs != nrhs)
            ABORT("SymLDL redistribution RHS capacity changed unexpectedly.");
        return;
    }
    MPI_Comm_size(grid3d->comm, &procs);
    plan->procs = procs;
    plan->nrhs = nrhs;
    cursor = (int *) SUPERLU_MALLOC((size_t) SUPERLU_MAX(1, procs) *
                                    sizeof(int));
    if (cursor == NULL)
        ABORT("Malloc fails for SymLDL redistribution cursors.");

    {
        int *send_counts = gstrs_comm->B_to_X_SendCnt;
        int *recv_counts = send_counts + 2 * procs;
        int *send_displs = send_counts + 4 * procs;
        int *recv_displs = send_counts + 6 * procs;
        plan->b_send_count = pdgstrs3d_symldl_total_rows(
            send_counts, send_displs, procs);
        plan->b_recv_count = pdgstrs3d_symldl_total_rows(
            recv_counts, recv_displs, procs);
        plan->b_source_rows = intMalloc_dist(
            SUPERLU_MAX(1, plan->b_send_count));
        plan->b_recv_rows = intMalloc_dist(
            SUPERLU_MAX(1, plan->b_recv_count));
        send_indices = intMalloc_dist(SUPERLU_MAX(1, plan->b_send_count));
        if (plan->b_source_rows == NULL || plan->b_recv_rows == NULL ||
            send_indices == NULL)
            ABORT("Malloc fails for SymLDL B-to-X redistribution plan.");
        memcpy(cursor, send_displs, (size_t) procs * sizeof(int));
        if (grid3d->zscp.Iam == 0) {
            for (int_t i = 0, grow = fst_row; i < m_loc; ++i, ++grow) {
                int_t irow = perm_c[perm_r[grow]];
                int_t gid = BlockNum(irow);
                int peer = pdgstrs3d_symv2_owner(trf3Dpartition, gid);
                int pos = cursor[peer]++;
                if (pos < 0 || pos >= plan->b_send_count)
                    ABORT("SymLDL B-to-X redistribution plan overflows.");
                plan->b_source_rows[pos] = i;
                send_indices[pos] = irow;
            }
        }
        for (int peer = 0; peer < procs; ++peer)
            if (cursor[peer] != send_displs[peer] + send_counts[peer])
                ABORT("SymLDL B-to-X redistribution counts do not match the plan.");
        MPI_Alltoallv(send_indices, send_counts, send_displs, mpi_int_t,
                      plan->b_recv_rows, recv_counts, recv_displs, mpi_int_t,
                      grid3d->comm);
        SUPERLU_FREE(send_indices);
        plan->b_send_values = doubleMalloc_dist(SUPERLU_MAX(
            (int_t) 1, pdgstrs3d_checked_workspace_count(
                plan->b_send_count, nrhs, 0, 0,
                "SymLDL B-to-X send values")));
        plan->b_recv_values = doubleMalloc_dist(SUPERLU_MAX(
            (int_t) 1, pdgstrs3d_checked_workspace_count(
                plan->b_recv_count, nrhs, 0, 0,
                "SymLDL B-to-X receive values")));
        if (plan->b_send_values == NULL || plan->b_recv_values == NULL)
            ABORT("Malloc fails for SymLDL B-to-X value buffers.");
    }

    {
        int *send_counts = gstrs_comm->X_to_B_SendCnt;
        int *recv_counts = send_counts + 2 * procs;
        int *send_displs = send_counts + 4 * procs;
        int *recv_displs = send_counts + 6 * procs;
        int iam = grid3d->iam;
        int_t *row_to_proc = SOLVEstruct->row_to_proc;
        plan->x_send_count = pdgstrs3d_symldl_total_rows(
            send_counts, send_displs, procs);
        plan->x_recv_count = pdgstrs3d_symldl_total_rows(
            recv_counts, recv_displs, procs);
        plan->x_source_offsets = intMalloc_dist(
            SUPERLU_MAX(1, plan->x_send_count));
        plan->x_source_widths = intMalloc_dist(
            SUPERLU_MAX(1, plan->x_send_count));
        plan->x_recv_rows = intMalloc_dist(
            SUPERLU_MAX(1, plan->x_recv_count));
        send_indices = intMalloc_dist(SUPERLU_MAX(1, plan->x_send_count));
        if (plan->x_source_offsets == NULL ||
            plan->x_source_widths == NULL || plan->x_recv_rows == NULL ||
            send_indices == NULL)
            ABORT("Malloc fails for SymLDL X-to-B redistribution plan.");
        memcpy(cursor, send_displs, (size_t) procs * sizeof(int));
        for (int_t k = 0; k < nsupers; ++k) {
            int_t width;
            int_t row_slot;
            int_t x_offset;
            int_t irow;
            if (iam != pdgstrs3d_symv2_owner(trf3Dpartition, k))
                continue;
            width = SuperSize(k);
            row_slot = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
            x_offset = X_BLK(row_slot);
            irow = FstBlockC(k);
            for (int_t i = 0; i < width; ++i, ++irow) {
                int peer = (int) row_to_proc[irow];
                int pos = cursor[peer]++;
                if (pos < 0 || pos >= plan->x_send_count)
                    ABORT("SymLDL X-to-B redistribution plan overflows.");
                send_indices[pos] = irow;
                plan->x_source_offsets[pos] = x_offset + i;
                plan->x_source_widths[pos] = width;
            }
        }
        for (int peer = 0; peer < procs; ++peer)
            if (cursor[peer] != send_displs[peer] + send_counts[peer])
                ABORT("SymLDL X-to-B redistribution counts do not match the plan.");
        MPI_Alltoallv(send_indices, send_counts, send_displs, mpi_int_t,
                      plan->x_recv_rows, recv_counts, recv_displs, mpi_int_t,
                      grid3d->comm);
        SUPERLU_FREE(send_indices);
        plan->x_send_values = doubleMalloc_dist(SUPERLU_MAX(
            (int_t) 1, pdgstrs3d_checked_workspace_count(
                plan->x_send_count, nrhs, 0, 0,
                "SymLDL X-to-B send values")));
        plan->x_recv_values = doubleMalloc_dist(SUPERLU_MAX(
            (int_t) 1, pdgstrs3d_checked_workspace_count(
                plan->x_recv_count, nrhs, 0, 0,
                "SymLDL X-to-B receive values")));
        if (plan->x_send_values == NULL || plan->x_recv_values == NULL)
            ABORT("Malloc fails for SymLDL X-to-B value buffers.");
    }
    SUPERLU_FREE(cursor);
    plan->initialized = 1;
}

static int_t
pdReDistribute3d_B_to_X_symv2(double *B, int_t m_loc, int nrhs, int_t ldb,
                         int_t fst_row, int_t * ilsum, double *x,
                         dScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist,
                         dtrf3Dpartition_t *trf3Dpartition,
                         gridinfo3d_t * grid3d, dSOLVEstruct_t * SOLVEstruct,
                         pdgstrs3d_symldl_redistribution_t *plan)
{
    int *SendCnt_nrhs, *RecvCnt_nrhs;
    int *sdispls_nrhs, *rdispls_nrhs;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    int procs = plan->procs;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

    (void) m_loc; (void) fst_row; (void) ScalePermstruct;
    (void) ilsum;
    if (plan == NULL || !plan->initialized || plan->nrhs != nrhs)
        ABORT("SymLDL B-to-X redistribution plan is unavailable.");
    SendCnt_nrhs = gstrs_comm->B_to_X_SendCnt + procs;
    RecvCnt_nrhs = gstrs_comm->B_to_X_SendCnt + 3 * procs;
    sdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 5 * procs;
    rdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 7 * procs;
    for (int row = 0; row < plan->b_send_count; ++row)
        for (int rhs = 0; rhs < nrhs; ++rhs)
            plan->b_send_values[row * nrhs + rhs] =
                B[plan->b_source_rows[row] + (int_t) rhs * ldb];
    MPI_Alltoallv(plan->b_send_values, SendCnt_nrhs, sdispls_nrhs, MPI_DOUBLE,
                  plan->b_recv_values, RecvCnt_nrhs, rdispls_nrhs, MPI_DOUBLE,
                  grid3d->comm);
    for (int row = 0; row < plan->b_recv_count; ++row) {
        int_t irow = plan->b_recv_rows[row];
        int_t k = BlockNum(irow);
        int_t width = SuperSize(k);
        int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
        int_t offset = X_BLK(lk);
        x[offset - XK_H] = k;
        irow -= FstBlockC(k);
        for (int rhs = 0; rhs < nrhs; ++rhs)
            x[offset + irow + (int_t) rhs * width] =
                plan->b_recv_values[row * nrhs + rhs];
    }
    return 0;
}

static int_t
pdReDistribute3d_X_to_B_symv2(int_t n, double *B, int_t m_loc, int_t ldb,
                         int_t fst_row, int nrhs, double *x, int_t * ilsum,
                         dScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist,
                         dtrf3Dpartition_t *trf3Dpartition,
                         gridinfo3d_t * grid3d, dSOLVEstruct_t * SOLVEstruct,
                         pdgstrs3d_symldl_redistribution_t *plan)
{
    int *SendCnt_nrhs, *RecvCnt_nrhs;
    int *sdispls_nrhs, *rdispls_nrhs;
    int procs = plan->procs;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

    (void) n; (void) ilsum; (void) ScalePermstruct;
    (void) Glu_persist; (void) trf3Dpartition;
    if (plan == NULL || !plan->initialized || plan->nrhs != nrhs)
        ABORT("SymLDL X-to-B redistribution plan is unavailable.");
    SendCnt_nrhs = gstrs_comm->X_to_B_SendCnt + procs;
    RecvCnt_nrhs = gstrs_comm->X_to_B_SendCnt + 3 * procs;
    sdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 5 * procs;
    rdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 7 * procs;
    for (int row = 0; row < plan->x_send_count; ++row)
        for (int rhs = 0; rhs < nrhs; ++rhs)
            plan->x_send_values[row * nrhs + rhs] =
                x[plan->x_source_offsets[row] +
                  (int_t) rhs * plan->x_source_widths[row]];
    MPI_Alltoallv(plan->x_send_values, SendCnt_nrhs, sdispls_nrhs, MPI_DOUBLE,
                  plan->x_recv_values, RecvCnt_nrhs, rdispls_nrhs, MPI_DOUBLE,
                  grid3d->comm);
    for (int row = 0; row < plan->x_recv_count; ++row) {
        int_t local_row = plan->x_recv_rows[row] - fst_row;
        if (local_row < 0 || local_row >= m_loc)
            ABORT("SymLDL X-to-B redistribution row is not local.");
        for (int rhs = 0; rhs < nrhs; ++rhs)
            B[local_row + (int_t) rhs * ldb] =
                plan->x_recv_values[row * nrhs + rhs];
    }
    return 0;
}

static int
pdgstrs3d_symldl_env_enabled(const char *name)
{
    const char *value = getenv(name);
    return value != NULL && atoi(value) != 0;
}

static int
pdgstrs3d_symldl_count_to_int(int_t count, const char *what)
{
    int out = (int) count;
    (void) what;
    if (count < 0 || (int_t) out != count)
        ABORT("SymLDL solve MPI count overflows int.");
    return out;
}

static void
pdgstrs3d_symldl_grow_double_buffer(double **buffer, int_t *capacity,
                                    int_t need, const char *what)
{
    if (need <= *capacity)
        return;
    if (*buffer)
        SUPERLU_FREE(*buffer);
    if (!(*buffer = doubleMalloc_dist(need)))
        ABORT(what);
    *capacity = need;
}

static int *
pdgstrs3d_symldl_diag_owners(int_t nsupers,
                             dtrf3Dpartition_t *trf3Dpartition,
                             gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    MPI_Comm solve_comm = grid3d->comm;
    int rank;
    int *local_owner;
    int *owner;
    int_t k;
    size_t owner_bytes = pdgstrs3d_checked_product((size_t) nsupers,
                                                   sizeof(int),
                                                   "SymLDL owner table");

    if (trf3Dpartition != NULL && trf3Dpartition->symV2DiagOwner != NULL) {
        if (!(owner = (int *) SUPERLU_MALLOC(owner_bytes)))
            ABORT("Malloc fails for SymLDL owner table.");
        memcpy(owner, trf3Dpartition->symV2DiagOwner, owner_bytes);
        return owner;
    }

    MPI_Comm_rank(solve_comm, &rank);
    if (!(local_owner = (int *) SUPERLU_MALLOC(owner_bytes)))
        ABORT("Malloc fails for SymLDL owner workspace.");
    if (!(owner = (int *) SUPERLU_MALLOC(owner_bytes)))
        ABORT("Malloc fails for SymLDL owner table.");

    for (k = 0; k < nsupers; ++k) {
        int diag_2d_rank = PNUM(PROW(k, grid), PCOL(k, grid), grid);
        local_owner[k] = (grid3d->zscp.Iam == 0 && grid->iam == diag_2d_rank)
                             ? rank
                             : INT_MAX;
    }

    MPI_Allreduce(local_owner, owner, pdgstrs3d_symldl_count_to_int(nsupers,
                  "SymLDL owner table"), MPI_INT, MPI_MIN, solve_comm);
    for (k = 0; k < nsupers; ++k)
        if (owner[k] == INT_MAX)
            ABORT("SymLDL solve could not identify a diagonal owner.");

    SUPERLU_FREE(local_owner);
    return owner;
}

typedef struct {
    int has_panel;
    int has_diag;
    int_t nsupr;
    int_t diag_luptr;
    int_t nblocks;
    int_t row_count;
    int_t lusup_count;
    double *lusup;
    int_t *block_luptr;
    int_t *block_nbrow;
    int_t *block_row_start;
    int_t *rows;
    int *row_dest_global;
} pdgstrs3d_symldl_panel_meta_t;

typedef struct {
    double *x;
    int_t x_cap;
} pdgstrs3d_symldl_workspace_t;

typedef struct {
    int_t n;
    int_t nsupers;
    int nrhs;
    int global_nprocs;
    int nprow;
    int npcol;
    int znp;
    int superlu_acc_offload;
    int superlu_acc_solve;
    int superlu_n_gemm;
    Glu_persist_t *Glu_persist;
    dLocalLU_t *Llu;
    dtrf3Dpartition_t *trf3Dpartition;
    int *supernodeMask;
    int *diag_owner;
    pdgstrs3d_symldl_panel_meta_t *panel_meta;
    dSymLDLSolveGraph *solve_graph;
    pdgstrs3d_symldl_workspace_t work;
    pdgstrs3d_symldl_redistribution_t redistribution;
    void *cpu_solve_state;
    void *nvshmem_state;
    void *factor_gpu_handle;
    double host_panel_copy_time;
    int factor_gpu_synchronized;
    int reused;
} pdgstrs3d_symldl_solve_meta_t;

typedef struct {
    double metadata;
    double workspace;
    double nvshmem_setup;
    double b_to_x;
    double solve_core;
    double forward_wall;
    double forward_compute;
    double forward_values;
    double forward_apply;
    double diag_wall;
    double diag_compute;
    double backward_wall;
    double backward_compute;
    double x_to_b;
    double gpu_h2d;
    double gpu_compute;
    double gpu_d2h;
    double cpu_event_setup;
    double cpu_event_progress;
    double cpu_event_numeric;
} pdgstrs3d_symldl_timer_t;

static void
pdgstrs3d_symldl_reset_solve_stats(pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL)
        return;
    meta->host_panel_copy_time = 0.0;
    meta->factor_gpu_synchronized = 0;
}

static double
pdgstrs3d_symldl_panel_solve_ops(pdgstrs3d_symldl_panel_meta_t *kmeta,
                                 int_t ksupc, int nrhs);

static double
pdgstrs3d_symldl_panel_solve_ops(pdgstrs3d_symldl_panel_meta_t *kmeta,
                                 int_t ksupc, int nrhs)
{
    double ops = 0.0;
    if (kmeta == NULL)
        return 0.0;
    for (int_t block = 0; block < kmeta->nblocks; ++block)
        ops += 2.0 * (double) kmeta->block_nbrow[block] *
               (double) nrhs * (double) ksupc;
    return ops;
}

static int
pdgstrs3d_symldl_local_panel_active(dtrf3Dpartition_t *trf3Dpartition,
                                    pdgstrs3d_symldl_panel_meta_t *kmeta,
                                    int_t k)
{
    if (kmeta == NULL || !kmeta->has_panel)
        return 0;
    if (trf3Dpartition == NULL || trf3Dpartition->superGridMap == NULL)
        ABORT("SymLDL solve requires LDL-native supernode grid metadata.");
    return trf3Dpartition->superGridMap[k] == IN_GRID_AIJ;
}

static pdgstrs3d_symldl_panel_meta_t *
pdgstrs3d_symldl_panel_meta_create(int_t nsupers, dLocalLU_t *Llu,
                                   Glu_persist_t *Glu_persist, gridinfo_t *grid,
                                   dtrf3Dpartition_t *trf3Dpartition,
                                   int *supernodeMask, int *diag_owner)
{
    pdgstrs3d_symldl_panel_meta_t *meta;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;

    if (!(meta = (pdgstrs3d_symldl_panel_meta_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) nsupers,
                              sizeof(pdgstrs3d_symldl_panel_meta_t),
                              "SymLDL panel metadata"))))
        ABORT("Malloc fails for SymLDL panel metadata.");

    for (int_t k = 0; k < nsupers; ++k) {
        meta[k].has_panel = 0;
        meta[k].has_diag = 0;
        meta[k].nsupr = 0;
        meta[k].diag_luptr = 0;
        meta[k].nblocks = 0;
        meta[k].row_count = 0;
        meta[k].lusup_count = 0;
        meta[k].lusup = NULL;
        meta[k].block_luptr = NULL;
        meta[k].block_nbrow = NULL;
        meta[k].block_row_start = NULL;
        meta[k].rows = NULL;
        meta[k].row_dest_global = NULL;

        int use_symv2_owner =
            trf3Dpartition != NULL &&
            trf3Dpartition->symV2PanelRoot != NULL &&
            trf3Dpartition->symV2PanelLocalIndex != NULL;
        int panel_root = use_symv2_owner
                             ? trf3Dpartition->symV2PanelRoot[k]
                             : PCOL(k, grid);

        if ((supernodeMask != NULL && !supernodeMask[k]) ||
            MYCOL(grid->iam, grid) != panel_root)
            continue;

        int_t lk_col = use_symv2_owner
                           ? trf3Dpartition->symV2PanelLocalIndex[k]
                           : LBj(k, grid);
        if (lk_col < 0)
            ABORT("SymLDL solve missing local L panel index.");
        int_t *lsub = Llu->Lrowind_bc_ptr[lk_col];
        double *lusup = Llu->Lnzval_bc_ptr[lk_col];
        if (lsub == NULL || lusup == NULL)
            continue;

        int_t lptr = BC_HEADER;
        int_t luptr = 0;
        int_t nblocks = 0;
        int_t row_count = 0;

        for (int_t lb = 0; lb < lsub[0]; ++lb) {
            int_t ik = lsub[lptr];
            int_t nbrow = lsub[lptr + 1];
            if (ik == k) {
                if (nbrow != SuperSize(k))
                    ABORT("SymLDL solve diagonal block has an unexpected size.");
                meta[k].has_diag = 1;
                meta[k].diag_luptr = luptr;
            } else {
                ++nblocks;
                row_count += nbrow;
            }
            lptr += LB_DESCRIPTOR + nbrow;
            luptr += nbrow;
        }

        meta[k].has_panel = 1;
        meta[k].nsupr = lsub[1];
        meta[k].lusup = lusup;
        meta[k].nblocks = nblocks;
        meta[k].row_count = row_count;
        meta[k].lusup_count = pdgstrs3d_checked_workspace_count(
            lsub[1], SuperSize(k), 0, 0, "SymLDL L panel values");

        if (nblocks > 0) {
            if (!(meta[k].block_luptr = intMalloc_dist(nblocks)) ||
                !(meta[k].block_nbrow = intMalloc_dist(nblocks)) ||
                !(meta[k].block_row_start = intMalloc_dist(nblocks)))
                ABORT("Malloc fails for SymLDL block metadata.");
        }
        if (row_count > 0) {
            if (!(meta[k].rows = intMalloc_dist(row_count)) ||
                !(meta[k].row_dest_global = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product((size_t) row_count,
                                                sizeof(int),
                                                "SymLDL row destination metadata"))))
                ABORT("Malloc fails for SymLDL row metadata.");
        }

        lptr = BC_HEADER;
        luptr = 0;
        int_t block = 0;
        int_t row = 0;
        for (int_t lb = 0; lb < lsub[0]; ++lb) {
            int_t ik = lsub[lptr];
            int_t nbrow = lsub[lptr + 1];
            int_t rows = lptr + LB_DESCRIPTOR;
            if (ik != k) {
                meta[k].block_luptr[block] = luptr;
                meta[k].block_nbrow[block] = nbrow;
                meta[k].block_row_start[block] = row;
                for (int_t r = 0; r < nbrow; ++r) {
                    int_t grow = lsub[rows + r];
                    meta[k].rows[row] = grow;
                    meta[k].row_dest_global[row] = diag_owner[BlockNum(grow)];
                    ++row;
                }
                ++block;
            }
            lptr += LB_DESCRIPTOR + nbrow;
            luptr += nbrow;
        }
    }

    return meta;
}

static void
pdgstrs3d_symldl_panel_meta_free(pdgstrs3d_symldl_panel_meta_t *meta,
                                 int_t nsupers)
{
    if (meta == NULL)
        return;

    for (int_t k = 0; k < nsupers; ++k) {
        if (meta[k].row_dest_global) SUPERLU_FREE(meta[k].row_dest_global);
        if (meta[k].rows) SUPERLU_FREE(meta[k].rows);
        if (meta[k].block_row_start) SUPERLU_FREE(meta[k].block_row_start);
        if (meta[k].block_nbrow) SUPERLU_FREE(meta[k].block_nbrow);
        if (meta[k].block_luptr) SUPERLU_FREE(meta[k].block_luptr);
    }
    SUPERLU_FREE(meta);
}

static void
pdgstrs3d_symldl_workspace_free(pdgstrs3d_symldl_workspace_t *work)
{
    if (work == NULL)
        return;
    if (work->x) SUPERLU_FREE(work->x);
    memset(work, 0, sizeof(*work));
}

static void
pdgstrs3d_symldl_zero_double_buffer(double *buffer, int_t count,
                                    const char *what)
{
    if (buffer == NULL || count <= 0)
        return;
    memset(buffer, 0, pdgstrs3d_checked_alloc_bytes(count, sizeof(double),
                                                    what));
}

static void
pdgstrs3d_symldl_workspace_prepare_x(pdgstrs3d_symldl_solve_meta_t *meta,
                                     int_t x_count)
{
    pdgstrs3d_symldl_workspace_t *work = &meta->work;

    if (x_count <= 0)
        x_count = 1;
    pdgstrs3d_symldl_grow_double_buffer(&work->x, &work->x_cap, x_count,
                                        "Malloc fails for x[].");
    pdgstrs3d_symldl_zero_double_buffer(work->x, x_count,
                                        "3D SymLDL solve x workspace");
}

static void
pdgstrs3d_symldl_sync_factor_gpu(pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL || meta->factor_gpu_handle == NULL ||
        meta->factor_gpu_synchronized)
        return;
    if (!meta->superlu_acc_offload) {
        meta->factor_gpu_synchronized = 1;
        return;
    }
#if defined(GPU_ACC)
    dSymLDLFactorGPUSynchronize((dLUgpu_Handle) meta->factor_gpu_handle);
#else
    ABORT("SymLDL V2 solve received GPU factor state in a non-CUDA build.");
#endif
    meta->factor_gpu_synchronized = 1;
}

static int
pdgstrs3d_symldl_factor_offload(superlu_dist_options_t *options)
{
#if defined(GPU_ACC)
    return sp_ienv_dist(10, options);
#else
    (void) options;
    return 0;
#endif
}

static dSymLDLSolveGraph *
pdgstrs3d_symldl_solve_graph_create(
    pdgstrs3d_symldl_solve_meta_t *meta, gridinfo3d_t *grid3d)
{
    int_t panel_count = meta->trf3Dpartition->symV2LocalPanelCount;
    int_t *supno = meta->Glu_persist->supno;
    int_t block_count = 0;
    int_t row_count = 0;
    dSymLDLPanelDesc *panels;
    dSymLDLBlockDesc *blocks;
    int_t *rows;
    dSymLDLSolveGraph *graph;

    for (int_t slot = 0; slot < panel_count; ++slot) {
        int_t gid = meta->trf3Dpartition->symV2LocalPanelGids[slot];
        pdgstrs3d_symldl_panel_meta_t *panel = &meta->panel_meta[gid];
        block_count = pdgstrs3d_checked_size_to_int_t(
            (size_t) block_count + (size_t) panel->nblocks,
            "SymLDL solve graph blocks");
        row_count = pdgstrs3d_checked_size_to_int_t(
            (size_t) row_count + (size_t) panel->row_count,
            "SymLDL solve graph rows");
    }

    panels = panel_count > 0
                 ? (dSymLDLPanelDesc *) SUPERLU_MALLOC(
                       pdgstrs3d_checked_alloc_bytes(
                           panel_count, sizeof(*panels),
                           "SymLDL solve graph panels"))
                 : NULL;
    blocks = block_count > 0
                 ? (dSymLDLBlockDesc *) SUPERLU_MALLOC(
                       pdgstrs3d_checked_alloc_bytes(
                           block_count, sizeof(*blocks),
                           "SymLDL solve graph blocks"))
                 : NULL;
    rows = row_count > 0 ? intMalloc_dist(row_count) : NULL;
    if ((panel_count > 0 && panels == NULL) ||
        (block_count > 0 && blocks == NULL) ||
        (row_count > 0 && rows == NULL))
        ABORT("Malloc fails for SymLDL solve graph descriptors.");

    int_t block_pos = 0;
    int_t row_pos = 0;
    for (int_t slot = 0; slot < panel_count; ++slot) {
        int_t gid = meta->trf3Dpartition->symV2LocalPanelGids[slot];
        pdgstrs3d_symldl_panel_meta_t *source = &meta->panel_meta[gid];
        dSymLDLPanelDesc *panel = &panels[slot];
        panel->gid = gid;
        panel->width = meta->Glu_persist->xsup[gid + 1] -
                       meta->Glu_persist->xsup[gid];
        panel->nsupr = source->nsupr;
        panel->diag_luptr = source->diag_luptr;
        panel->block_begin = block_pos;
        panel->block_count = source->nblocks;
        panel->value_count = source->lusup_count;
        panel->has_diag = source->has_diag;
        panel->active = source->has_panel;
        panel->values = source->lusup;

        for (int_t local = 0; local < source->nblocks; ++local) {
            int_t source_row = source->block_row_start[local];
            int_t nbrow = source->block_nbrow[local];
            int_t target = BlockNum(source->rows[source_row]);
            dSymLDLBlockDesc *block = &blocks[block_pos++];
            block->panel_id = slot;
            block->target_gid = target;
            block->luptr = source->block_luptr[local];
            block->nbrow = nbrow;
            block->row_begin = row_pos;
            for (int_t row = 0; row < nbrow; ++row) {
                int_t grow = source->rows[source_row + row];
                if (BlockNum(grow) != target)
                    ABORT("SymLDL solve graph block spans supernodes.");
                rows[row_pos++] = grow;
            }
        }
    }
    if (block_pos != block_count || row_pos != row_count)
        ABORT("SymLDL solve graph descriptor count is inconsistent.");

    graph = dSymLDLSolveGraphCreate(
        meta->n, meta->nsupers, panel_count, panels, block_count, blocks,
        row_count, rows, meta->Glu_persist->xsup, meta->Llu->ilsum,
        meta->trf3Dpartition, grid3d);
    if (rows != NULL) SUPERLU_FREE(rows);
    if (blocks != NULL) SUPERLU_FREE(blocks);
    if (panels != NULL) SUPERLU_FREE(panels);
    if (graph == NULL)
        ABORT("SymLDL solve graph setup failed.");
    return graph;
}

static void
pdgstrs3d_symldl_prepare_host_factor_panels(
    pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL || meta->panel_meta == NULL ||
        meta->factor_gpu_handle == NULL)
        return;
    if (!meta->superlu_acc_offload)
        return;

    pdgstrs3d_symldl_sync_factor_gpu(meta);
    double t = SuperLU_timer_();
    for (int_t k = 0; k < meta->nsupers; ++k) {
        pdgstrs3d_symldl_panel_meta_t *kmeta = &meta->panel_meta[k];
        if (!kmeta->has_panel)
            continue;
#if defined(GPU_ACC)
        if (dSymLDLFactorGPUCopyPanelToHost(
                (dLUgpu_Handle) meta->factor_gpu_handle, k) != 0)
            ABORT("Failed to copy a SymLDL factor panel to host.");
#else
        ABORT("SymLDL V2 solve cannot copy factor GPU panels in a non-CUDA build.");
#endif
    }
    if (meta->solve_graph != NULL) {
        for (int_t slot = 0; slot < meta->solve_graph->panel_count; ++slot) {
            int_t gid = meta->solve_graph->panel_gids[slot];
            meta->solve_graph->panels[slot].values =
                meta->panel_meta[gid].lusup;
        }
    }
    meta->host_panel_copy_time += SuperLU_timer_() - t;
}

static int
pdgstrs3d_symldl_use_nvshmem(pdgstrs3d_symldl_solve_meta_t *meta)
{
#if defined(GPU_ACC)
    return meta != NULL && meta->superlu_acc_offload &&
           meta->superlu_acc_solve &&
           meta->factor_gpu_handle != NULL &&
           dSymLDLNVSHMEMSolveAvailable();
#else
    (void) meta;
    return 0;
#endif
}

static void
pdgstrs3d_symldl_validate_nvshmem_panel_layout(
    pdgstrs3d_symldl_solve_meta_t *meta, int_t local_panel, int_t k,
    pdgstrs3d_symldl_panel_meta_t *panel, gridinfo3d_t *grid3d)
{
    int_t *lsub;
    int_t *lloc;
    int_t update_count;
    int_t index_offset;
    int_t value_offset;
    int_t width;
    int_t *xsup;
    gridinfo_t *grid;
    int myrow;
    int root;

    if (meta == NULL || panel == NULL || grid3d == NULL ||
        local_panel < 0 ||
        local_panel >= meta->trf3Dpartition->symV2LocalPanelCount ||
        k < 0 || k >= meta->nsupers)
        ABORT("SymLDL NVSHMEM panel validation is invalid.");

    xsup = meta->Glu_persist->xsup;
    grid = &grid3d->grid2d;
    lsub = meta->Llu->Lrowind_bc_ptr[local_panel];
    lloc = meta->Llu->Lindval_loc_bc_ptr[local_panel];
    width = SuperSize(k);
    if (!panel->has_panel) {
        if (lsub != NULL || lloc != NULL ||
            meta->Llu->Lnzval_bc_ptr[local_panel] != NULL)
            ABORT("SymLDL NVSHMEM inactive panel has factor storage.");
        return;
    }
    if (lsub == NULL || lloc == NULL ||
        meta->Llu->Lnzval_bc_ptr[local_panel] == NULL || lsub[0] <= 0 ||
        lsub[1] <= 0 || panel->nsupr != lsub[1] ||
        panel->lusup != meta->Llu->Lnzval_bc_ptr[local_panel] ||
        panel->lusup_count != lsub[1] * width)
        ABORT("SymLDL NVSHMEM panel storage differs from the solve metadata.");

    myrow = MYROW(grid->iam, grid);
    root = meta->trf3Dpartition->symV2DiagRoot[k];
    if (myrow == root) {
        if (lsub[BC_HEADER] != k || !panel->has_diag ||
            panel->diag_luptr != 0)
            ABORT("SymLDL NVSHMEM diagonal panel layout is inconsistent.");
        update_count = lsub[0] - 1;
        index_offset = update_count + 2;
        value_offset = 2 * update_count + 3;
    } else {
        if (panel->has_diag)
            ABORT("SymLDL NVSHMEM off-diagonal panel contains a diagonal block.");
        update_count = lsub[0];
        index_offset = update_count;
        value_offset = 2 * update_count;
    }
    if (update_count < 0 || panel->nblocks != update_count)
        ABORT("SymLDL NVSHMEM update count differs from the original panel map.");

    for (int_t block = 0; block < update_count; ++block) {
        int_t lptr = lloc[index_offset + block];
        int_t luptr = lloc[value_offset + block];
        int_t target = lsub[lptr];
        int_t nbrow = lsub[lptr + 1];
        int_t row_start = panel->block_row_start[block];
        if (lptr < BC_HEADER || luptr < 0 || target == k || nbrow <= 0 ||
            panel->block_luptr[block] != luptr ||
            panel->block_nbrow[block] != nbrow || row_start < 0 ||
            row_start + nbrow > panel->row_count)
            ABORT("SymLDL NVSHMEM block descriptor differs from the original panel map.");
        for (int_t row = 0; row < nbrow; ++row)
            if (panel->rows[row_start + row] !=
                lsub[lptr + LB_DESCRIPTOR + row])
                ABORT("SymLDL NVSHMEM row descriptor differs from the original panel map.");
    }
}

static void
pdgstrs3d_symldl_nvshmem_prepare(
    pdgstrs3d_symldl_solve_meta_t *meta, int_t n, int_t nlb,
    int_t ldalsum, int nrhs, gridinfo3d_t *grid3d)
{
    if (meta == NULL)
        return;
    if (!pdgstrs3d_symldl_use_nvshmem(meta))
        ABORT("SymLDL NVSHMEM solve requires GPU offload and NVSHMEM.");
    if (meta->nvshmem_state != NULL)
        return;
#if defined(GPU_ACC)
    pdgstrs3d_symldl_sync_factor_gpu(meta);
    dSymLDLSolveGraph *graph = meta->solve_graph;
    if (graph == NULL)
        ABORT("SymLDL NVSHMEM solve graph is unavailable.");
    int_t *xsup = graph->xsup;
    int_t panel_count = graph->panel_count;
    int_t block_count = graph->block_count;
    int_t row_count = graph->factor_row_count;
    int_t retained_panel_count = 0;
    int_t active_panel_count = 0;
    int_t active_block_count = 0;
    int_t active_row_count = 0;
    double retained_transpose_work = 0.0;
    double active_transpose_work = 0.0;
    for (int_t lp = 0; lp < panel_count; ++lp) {
        int_t k = meta->trf3Dpartition->symV2LocalPanelGids[lp];
        pdgstrs3d_symldl_panel_meta_t *panel = &meta->panel_meta[k];
        if (!panel->has_panel)
            continue;
        int panel_active = pdgstrs3d_symldl_local_panel_active(
            meta->trf3Dpartition, panel, k);
        ++retained_panel_count;
        if (panel_active) {
            ++active_panel_count;
            active_block_count = pdgstrs3d_checked_size_to_int_t(
                (size_t) active_block_count + (size_t) panel->nblocks,
                "SymLDL NVSHMEM 3D active block descriptors");
            active_row_count = pdgstrs3d_checked_size_to_int_t(
                (size_t) active_row_count + (size_t) panel->row_count,
                "SymLDL NVSHMEM 3D active row descriptors");
        }
        for (int_t block = 0; block < panel->nblocks; ++block) {
            double work = (double) panel->block_nbrow[block] *
                          (double) SuperSize(k) * (double) nrhs;
            retained_transpose_work += work;
            if (panel_active)
                active_transpose_work += work;
        }
    }
    if (pdgstrs3d_symldl_env_enabled("GPU3DV2_SYM_SOLVE_TIMING")) {
        fprintf(stderr,
                "SymLDL NVSHMEM 3D factor-volume profile: rank=%d z=%d "
                "retained_panels=%lld active_panels=%lld "
                "retained_blocks=%lld active_blocks=%lld "
                "retained_rows=%lld active_rows=%lld "
                "retained_transpose_work=%.0f active_transpose_work=%.0f\n",
                grid3d->iam, grid3d->zscp.Iam,
                (long long) retained_panel_count,
                (long long) active_panel_count,
                (long long) block_count,
                (long long) active_block_count,
                (long long) row_count,
                (long long) active_row_count,
                retained_transpose_work, active_transpose_work);
        fflush(stderr);
    }
    dSymLDLNVPanelDesc *device_panels = panel_count > 0
        ? (dSymLDLNVPanelDesc *) SUPERLU_MALLOC(
              pdgstrs3d_checked_alloc_bytes(
                  panel_count, sizeof(*device_panels),
                  "SymLDL NVSHMEM solve panel descriptors"))
        : NULL;
    if (panel_count > 0 && device_panels == NULL)
        ABORT("Malloc fails for SymLDL NVSHMEM panel bindings.");

    for (int_t lp = 0; lp < panel_count; ++lp) {
        int_t k = graph->panel_gids[lp];
        pdgstrs3d_symldl_panel_meta_t *source = &meta->panel_meta[k];
        pdgstrs3d_symldl_validate_nvshmem_panel_layout(
            meta, lp, k, source, grid3d);
        device_panels[lp] = graph->panels[lp];
        device_panels[lp].values = NULL;
        if (!source->has_panel)
            continue;
        int_t device_count = 0;
        if (dSymLDLFactorGPUGetPanel(
                (dLUgpu_Handle) meta->factor_gpu_handle, k,
                &device_panels[lp].values, &device_count) != 0 ||
            device_panels[lp].values == NULL ||
            device_count < device_panels[lp].value_count)
            ABORT("SymLDL NVSHMEM solve could not access a retained L panel.");
    }

    int_t x_count = pdgstrs3d_checked_workspace_count(
        ldalsum, nrhs, nlb, XK_H,
        "SymLDL NVSHMEM solve X workspace");
    int_t lsum_count = pdgstrs3d_checked_workspace_count(
        ldalsum, nrhs, nlb, LSUM_H,
        "SymLDL NVSHMEM solve sum workspace");
    meta->nvshmem_state = dSymLDLNVSHMEMSolveCreate(
        nrhs, x_count, lsum_count, graph, device_panels,
        meta->trf3Dpartition, grid3d);
    if (meta->nvshmem_state == NULL)
        ABORT("SymLDL NVSHMEM solve setup failed.");
    if (device_panels != NULL)
        SUPERLU_FREE(device_panels);
#else
    (void) n;
    (void) nlb;
    (void) ldalsum;
    (void) nrhs;
    (void) grid3d;
    ABORT("SymLDL NVSHMEM solve requires a CUDA build.");
#endif
}

static void
pdgstrs3d_symldl_nvshmem_take_timers(
    pdgstrs3d_symldl_solve_meta_t *meta,
    pdgstrs3d_symldl_timer_t *timer)
{
    double setup = 0.0;
    double h2d = 0.0;
    double forward = 0.0;
    double sparse_reduce = 0.0;
    double sparse_broadcast = 0.0;
    double diagonal = 0.0;
    double backward = 0.0;
    double d2h = 0.0;
    if (meta == NULL || meta->nvshmem_state == NULL || timer == NULL)
        return;
    dSymLDLNVSHMEMSolveTakeTimers(
        meta->nvshmem_state, &setup, &forward, &sparse_reduce,
        &sparse_broadcast, &diagonal, &backward, &h2d, &d2h);
    timer->nvshmem_setup += setup;
    timer->forward_compute += forward;
    timer->forward_values += sparse_reduce;
    timer->forward_apply += sparse_broadcast;
    timer->diag_compute += diagonal;
    timer->backward_compute += backward;
    timer->gpu_h2d += h2d;
    timer->gpu_compute += forward + diagonal + backward;
    timer->gpu_d2h += d2h;
}

static void
pdgstrs3d_symldl_cpu_take_timers(
    pdgstrs3d_symldl_solve_meta_t *meta,
    pdgstrs3d_symldl_timer_t *timer,
    dSymLDLCPUSolveCommStats *comm_stats)
{
    double setup = 0.0;
    double forward = 0.0;
    double diagonal = 0.0;
    double backward = 0.0;
    double progress = 0.0;
    double numeric = 0.0;
    if (meta == NULL || meta->cpu_solve_state == NULL || timer == NULL)
        return;
    dSymLDLCPUSolveTakeTimers(
        (dSymLDLCPUSolveHandle *) meta->cpu_solve_state,
        &setup, &forward, &diagonal, &backward, &progress, &numeric);
    if (comm_stats != NULL)
        dSymLDLCPUSolveTakeCommStats(
            (dSymLDLCPUSolveHandle *) meta->cpu_solve_state, comm_stats);
    timer->cpu_event_setup += setup;
    timer->forward_compute += forward;
    timer->diag_compute += diagonal;
    timer->backward_compute += backward;
    timer->cpu_event_progress += progress;
    timer->cpu_event_numeric += numeric;
}

static void
pdgstrs3d_symldl_solve_meta_destroy(pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL)
        return;
    if (meta->nvshmem_state)
        dSymLDLNVSHMEMSolveDestroy(meta->nvshmem_state);
    if (meta->cpu_solve_state)
        dSymLDLCPUSolveDestroy(
            (dSymLDLCPUSolveHandle *) meta->cpu_solve_state);
    if (meta->factor_gpu_handle)
        dDestroyLUgpuHandle((dLUgpu_Handle) meta->factor_gpu_handle);
    pdgstrs3d_symldl_redistribution_free(&meta->redistribution);
    pdgstrs3d_symldl_workspace_free(&meta->work);
    dSymLDLSolveGraphDestroy(meta->solve_graph);
    pdgstrs3d_symldl_panel_meta_free(meta->panel_meta, meta->nsupers);
    if (meta->diag_owner) SUPERLU_FREE(meta->diag_owner);
    SUPERLU_FREE(meta);
}

void
pdgstrs3d_symldl_finalize(dSOLVEstruct_t *SOLVEstruct)
{
    if (SOLVEstruct == NULL || SOLVEstruct->symldl_v2_solve_meta == NULL)
    {
        if (SOLVEstruct != NULL && SOLVEstruct->symldl_v2_factor_handle != NULL) {
            dDestroyLUgpuHandle(
                (dLUgpu_Handle) SOLVEstruct->symldl_v2_factor_handle);
            SOLVEstruct->symldl_v2_factor_handle = NULL;
        }
        return;
    }
    pdgstrs3d_symldl_solve_meta_destroy(
        (pdgstrs3d_symldl_solve_meta_t *) SOLVEstruct->symldl_v2_solve_meta);
    SOLVEstruct->symldl_v2_solve_meta = NULL;
    SOLVEstruct->symldl_v2_factor_handle = NULL;
}

static int
pdgstrs3d_symldl_solve_meta_valid(pdgstrs3d_symldl_solve_meta_t *meta,
                                  int_t n, int_t nsupers, int nrhs,
                                  Glu_persist_t *Glu_persist,
                                  dLocalLU_t *Llu,
                                  dtrf3Dpartition_t *trf3Dpartition,
                                  int *supernodeMask, gridinfo3d_t *grid3d,
                                  int global_nprocs, int superlu_acc_offload,
                                  int superlu_acc_solve, int superlu_n_gemm)
{
    gridinfo_t *grid = &(grid3d->grid2d);

    return meta != NULL &&
           meta->n == n &&
           meta->nsupers == nsupers &&
           meta->nrhs == nrhs &&
           meta->global_nprocs == global_nprocs &&
           meta->nprow == grid->nprow &&
           meta->npcol == grid->npcol &&
           meta->znp == grid3d->zscp.Np &&
           meta->superlu_acc_offload == superlu_acc_offload &&
           meta->superlu_acc_solve == superlu_acc_solve &&
           meta->superlu_n_gemm == superlu_n_gemm &&
           meta->Glu_persist == Glu_persist &&
           meta->Llu == Llu &&
           meta->trf3Dpartition == trf3Dpartition &&
           meta->supernodeMask == supernodeMask;
}

static pdgstrs3d_symldl_solve_meta_t *
pdgstrs3d_symldl_solve_meta_get(dSOLVEstruct_t *SOLVEstruct, int_t n,
                                int_t nsupers, int nrhs,
                                dLUstruct_t *LUstruct,
                                dtrf3Dpartition_t *trf3Dpartition,
                                int *supernodeMask, gridinfo_t *grid,
                                gridinfo3d_t *grid3d, int global_nprocs,
                                int superlu_acc_offload,
                                int superlu_acc_solve, int superlu_n_gemm)
{
    pdgstrs3d_symldl_solve_meta_t *meta =
        (pdgstrs3d_symldl_solve_meta_t *) SOLVEstruct->symldl_v2_solve_meta;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;

    if (pdgstrs3d_symldl_solve_meta_valid(meta, n, nsupers, nrhs,
                                          Glu_persist, Llu, trf3Dpartition,
                                          supernodeMask, grid3d,
                                          global_nprocs,
                                          superlu_acc_offload,
                                          superlu_acc_solve,
                                          superlu_n_gemm)) {
        if (SOLVEstruct->symldl_v2_factor_handle != NULL) {
            if (meta->factor_gpu_handle != NULL)
                ABORT("SymLDL solve metadata already owns a factor GPU handle.");
            meta->factor_gpu_handle = SOLVEstruct->symldl_v2_factor_handle;
            SOLVEstruct->symldl_v2_factor_handle = NULL;
        }
        meta->reused = 1;
        return meta;
    }

    void *pending_factor_gpu_handle = SOLVEstruct->symldl_v2_factor_handle;
    SOLVEstruct->symldl_v2_factor_handle = NULL;
    pdgstrs3d_symldl_finalize(SOLVEstruct);
    SOLVEstruct->symldl_v2_factor_handle = pending_factor_gpu_handle;
    if (!(meta = (pdgstrs3d_symldl_solve_meta_t *)
              SUPERLU_MALLOC(sizeof(pdgstrs3d_symldl_solve_meta_t))))
        ABORT("Malloc fails for SymLDL solve metadata.");
    memset(meta, 0, sizeof(*meta));
    meta->n = n;
    meta->nsupers = nsupers;
    meta->nrhs = nrhs;
    meta->global_nprocs = global_nprocs;
    meta->nprow = grid->nprow;
    meta->npcol = grid->npcol;
    meta->znp = grid3d->zscp.Np;
    meta->superlu_acc_offload = superlu_acc_offload;
    meta->superlu_acc_solve = superlu_acc_solve;
    meta->superlu_n_gemm = superlu_n_gemm;
    if (getenv("GPU3DV2_TRACE")) {
        fprintf(stderr,
                "[sym-v2-trace] rank %d: metadata taking factor handle %p\n",
                grid3d->iam, SOLVEstruct->symldl_v2_factor_handle);
        fflush(stderr);
    }
    meta->factor_gpu_handle = SOLVEstruct->symldl_v2_factor_handle;
    SOLVEstruct->symldl_v2_factor_handle = NULL;
    meta->Glu_persist = Glu_persist;
    meta->Llu = Llu;
    meta->trf3Dpartition = trf3Dpartition;
    meta->supernodeMask = supernodeMask;
    meta->diag_owner =
        pdgstrs3d_symldl_diag_owners(nsupers, trf3Dpartition, grid3d);
    meta->panel_meta = pdgstrs3d_symldl_panel_meta_create(nsupers, Llu,
                                                          Glu_persist, grid,
                                                          trf3Dpartition,
                                                          supernodeMask,
                                                          meta->diag_owner);
    meta->solve_graph = pdgstrs3d_symldl_solve_graph_create(meta, grid3d);
    meta->reused = 0;
    SOLVEstruct->symldl_v2_solve_meta = meta;
    return meta;
}

void
pdgstrs3d_symldl_init_meta(superlu_dist_options_t *options, int_t n, int nrhs,
                           dLUstruct_t *LUstruct,
                           dtrf3Dpartition_t *trf3Dpartition,
                           gridinfo3d_t *grid3d,
                           dSOLVEstruct_t *SOLVEstruct)
{
    if (options == NULL || LUstruct == NULL || LUstruct->Glu_persist == NULL ||
        LUstruct->Llu == NULL || trf3Dpartition == NULL ||
        grid3d == NULL || SOLVEstruct == NULL)
        ABORT("SymLDL solve metadata initialization received invalid arguments.");

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int *supernodeMask = trf3Dpartition->supernodeMask;
    int global_nprocs;

    MPI_Comm_size(grid3d->comm, &global_nprocs);
    (void) pdgstrs3d_symldl_solve_meta_get(
        SOLVEstruct, n, nsupers, nrhs, LUstruct, trf3Dpartition,
        supernodeMask, &grid3d->grid2d, grid3d, global_nprocs,
        pdgstrs3d_symldl_factor_offload(options), sp_ienv_dist(11, options),
        sp_ienv_dist(7, options));
}

static int
pdgstrs3d_symldl_global_rank(gridinfo3d_t *grid3d, int rank2d, int z)
{
    return (grid3d->rankorder == 1)
               ? rank2d * grid3d->npdep + z
               : z * (grid3d->nprow * grid3d->npcol) + rank2d;
}

static int_t
pdgstrs3d_symldl_init_comm(int_t n, int_t m_loc, int_t nrhs,
                           int_t fst_row, int_t perm_r[], int_t perm_c[],
                           gridinfo3d_t *grid3d,
                           Glu_persist_t *Glu_persist,
                           dtrf3Dpartition_t *trf3Dpartition,
                           dSOLVEstruct_t *SOLVEstruct)
{
    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *sdispls_nrhs, *rdispls, *rdispls_nrhs;
    int *itemp, *ptr_to_ibuf;
    int_t *row_to_proc;
    int_t i, gbi, k, l, knsupc, nsupers, *xsup, *supno;
    int_t irow;
    int iam, p, q, procs;
    pxgstrs_comm_t *gstrs_comm;

    if (trf3Dpartition == NULL ||
        trf3Dpartition->symV2DiagOwner == NULL)
        ABORT("SymFact V2 solve initialization requires LDL-native owner tables.");

    procs = grid3d->nprow * grid3d->npcol * grid3d->npdep;
    iam = grid3d->iam;
    gstrs_comm = SOLVEstruct->gstrs_comm;
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = Glu_persist->supno[n - 1] + 1;
    row_to_proc = SOLVEstruct->row_to_proc;

    if (!(itemp = SUPERLU_MALLOC(8 * procs * sizeof(int))))
        ABORT("Malloc fails for SymV2 B_to_X_itemp[].");
    SendCnt = itemp;
    SendCnt_nrhs = itemp + procs;
    RecvCnt = itemp + 2 * procs;
    RecvCnt_nrhs = itemp + 3 * procs;
    sdispls = itemp + 4 * procs;
    sdispls_nrhs = itemp + 5 * procs;
    rdispls = itemp + 6 * procs;
    rdispls_nrhs = itemp + 7 * procs;

    for (p = 0; p < procs; ++p) SendCnt[p] = 0;
    if (grid3d->zscp.Iam == 0) {
        for (i = 0, l = fst_row; i < m_loc; ++i, ++l) {
            irow = perm_c[perm_r[l]];
            gbi = BlockNum(irow);
            p = trf3Dpartition->symV2DiagOwner[gbi];
            ++SendCnt[p];
        }
    }

    MPI_Alltoall(SendCnt, 1, MPI_INT, RecvCnt, 1, MPI_INT,
                 grid3d->comm);
    sdispls[0] = rdispls[0] = 0;
    for (p = 1; p < procs; ++p) {
        sdispls[p] = sdispls[p - 1] + SendCnt[p - 1];
        rdispls[p] = rdispls[p - 1] + RecvCnt[p - 1];
    }
    for (p = 0; p < procs; ++p) {
        SendCnt_nrhs[p] = SendCnt[p] * nrhs;
        sdispls_nrhs[p] = sdispls[p] * nrhs;
        RecvCnt_nrhs[p] = RecvCnt[p] * nrhs;
        rdispls_nrhs[p] = rdispls[p] * nrhs;
    }
    gstrs_comm->B_to_X_SendCnt = SendCnt;

    if (!(itemp = SUPERLU_MALLOC(8 * procs * sizeof(int))))
        ABORT("Malloc fails for SymV2 X_to_B_itemp[].");
    SendCnt = itemp;
    SendCnt_nrhs = itemp + procs;
    RecvCnt = itemp + 2 * procs;
    RecvCnt_nrhs = itemp + 3 * procs;
    sdispls = itemp + 4 * procs;
    sdispls_nrhs = itemp + 5 * procs;
    rdispls = itemp + 6 * procs;
    rdispls_nrhs = itemp + 7 * procs;

    for (p = 0; p < procs; ++p) SendCnt[p] = 0;
    for (k = 0; k < nsupers; ++k) {
        p = trf3Dpartition->symV2DiagOwner[k];
        if (iam != p)
            continue;
        knsupc = SuperSize(k);
        irow = FstBlockC(k);
        for (i = 0; i < knsupc; ++i) {
            q = row_to_proc[irow++];
            ++SendCnt[q];
        }
    }

    MPI_Alltoall(SendCnt, 1, MPI_INT, RecvCnt, 1, MPI_INT,
                 grid3d->comm);
    sdispls[0] = rdispls[0] = 0;
    sdispls_nrhs[0] = rdispls_nrhs[0] = 0;
    SendCnt_nrhs[0] = SendCnt[0] * nrhs;
    RecvCnt_nrhs[0] = RecvCnt[0] * nrhs;
    for (p = 1; p < procs; ++p) {
        sdispls[p] = sdispls[p - 1] + SendCnt[p - 1];
        rdispls[p] = rdispls[p - 1] + RecvCnt[p - 1];
        sdispls_nrhs[p] = sdispls[p] * nrhs;
        rdispls_nrhs[p] = rdispls[p] * nrhs;
        SendCnt_nrhs[p] = SendCnt[p] * nrhs;
        RecvCnt_nrhs[p] = RecvCnt[p] * nrhs;
    }
    gstrs_comm->X_to_B_SendCnt = SendCnt;

    if (!(ptr_to_ibuf = SUPERLU_MALLOC(2 * procs * sizeof(int))))
        ABORT("Malloc fails for SymV2 ptr_to_ibuf[].");
    gstrs_comm->ptr_to_ibuf = ptr_to_ibuf;
    gstrs_comm->ptr_to_dbuf = ptr_to_ibuf + procs;

    return 0;
}

int
dSymV2SolveInit(superlu_dist_options_t *options, SuperMatrix *A,
                int_t perm_r[], int_t perm_c[], int_t nrhs,
                dLUstruct_t *LUstruct, dtrf3Dpartition_t *trf3Dpartition,
                gridinfo3d_t *grid3d, dSOLVEstruct_t *SOLVEstruct)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int_t *row_to_proc, *inv_perm_c, *itemp;
    NRformat_loc *Astore;
    int_t i, fst_row, m_loc, p;
    int procs2d;

    Astore = (NRformat_loc *) A->Store;
    fst_row = Astore->fst_row;
    m_loc = Astore->m_loc;
    procs2d = grid->nprow * grid->npcol;
    SOLVEstruct->symldl_v2_solve_meta = NULL;

    if (!(row_to_proc = intMalloc_dist(A->nrow)))
        ABORT("Malloc fails for row_to_proc[].");
    SOLVEstruct->row_to_proc = row_to_proc;
    if (!(inv_perm_c = intMalloc_dist(A->ncol)))
        ABORT("Malloc fails for inv_perm_c[].");
    if (SLU_IS_SYMATCH_ROWPERM(options->RowPerm) && options->Algo3d == YES) {
        for (i = 0; i < A->ncol; ++i) inv_perm_c[perm_c[perm_r[i]]] = i;
    } else {
        for (i = 0; i < A->ncol; ++i) inv_perm_c[perm_c[i]] = i;
    }
    SOLVEstruct->inv_perm_c = inv_perm_c;

    if (!(itemp = intMalloc_dist(procs2d + 1)))
        ABORT("Malloc fails for itemp[].");
    MPI_Allgather(&fst_row, 1, mpi_int_t, itemp, 1, mpi_int_t,
                  grid->comm);
    itemp[procs2d] = A->nrow;
    for (p = 0; p < procs2d; ++p) {
        int global_layer0_rank =
            pdgstrs3d_symldl_global_rank(grid3d, (int) p, 0);
        for (i = itemp[p]; i < itemp[p + 1]; ++i)
            row_to_proc[i] = global_layer0_rank;
    }
    SUPERLU_FREE(itemp);

    if (trf3Dpartition == NULL ||
        trf3Dpartition->symV2DiagOwner == NULL)
        ABORT("SymFact V2 solve initialization requires LDL-native owner tables.");

    SOLVEstruct->num_diag_procs = 0;
    SOLVEstruct->diag_procs = NULL;
    SOLVEstruct->diag_len = NULL;

    if (!(SOLVEstruct->gstrs_comm = (pxgstrs_comm_t *)
              SUPERLU_MALLOC(sizeof(pxgstrs_comm_t))))
        ABORT("Malloc fails for gstrs_comm[].");
    pdgstrs3d_symldl_init_comm(A->ncol, m_loc, nrhs, fst_row, perm_r,
                               perm_c, grid3d, LUstruct->Glu_persist,
                               trf3Dpartition, SOLVEstruct);

    if (!(SOLVEstruct->gsmv_comm = (pdgsmv_comm_t *)
              SUPERLU_MALLOC(sizeof(pdgsmv_comm_t))))
        ABORT("Malloc fails for gsmv_comm[].");
    SOLVEstruct->A_colind_gsmv = NULL;
    pdgstrs3d_symldl_init_meta(options, A->ncol, nrhs, LUstruct,
                               trf3Dpartition, grid3d, SOLVEstruct);
#if (defined(GPU_ACC))
    SOLVEstruct->d_lsum = NULL;
    SOLVEstruct->d_lsum_save = NULL;
    SOLVEstruct->d_x = NULL;
    SOLVEstruct->d_fmod = NULL;
    SOLVEstruct->d_fmod_save = NULL;
    SOLVEstruct->d_bmod = NULL;
    SOLVEstruct->d_bmod_save = NULL;
#endif

    options->SolveInitialized = YES;
    return 0;
}

static void
pdgstrs3d_symldl_timer_print(pdgstrs3d_symldl_timer_t *timer,
                             pdgstrs3d_symldl_solve_meta_t *meta,
                             gridinfo3d_t *grid3d)
{
    enum { SYMLDL_TIMER_COUNT = 21 };
    double local[SYMLDL_TIMER_COUNT];
    double maxv[SYMLDL_TIMER_COUNT];
    double sumv[SYMLDL_TIMER_COUNT];
    struct { double val; int rank; } local_max[SYMLDL_TIMER_COUNT];
    struct { double val; int rank; } global_max[SYMLDL_TIMER_COUNT];
    int rank, nprocs;

    local[0] = timer->metadata;
    local[1] = timer->workspace;
    local[2] = timer->b_to_x;
    local[3] = timer->solve_core;
    local[4] = timer->forward_wall;
    local[5] = timer->forward_compute;
    local[6] = timer->forward_values;
    local[7] = timer->forward_apply;
    local[8] = timer->diag_wall;
    local[9] = timer->diag_compute;
    local[10] = timer->backward_wall;
    local[11] = timer->backward_compute;
    local[12] = timer->x_to_b;
    local[13] = timer->gpu_h2d;
    local[14] = timer->gpu_compute;
    local[15] = timer->gpu_d2h;
    local[16] = meta != NULL ? meta->host_panel_copy_time : 0.0;
    local[17] = timer->nvshmem_setup;
    local[18] = timer->cpu_event_setup;
    local[19] = timer->cpu_event_progress;
    local[20] = timer->cpu_event_numeric;

    MPI_Comm_rank(grid3d->comm, &rank);
    MPI_Comm_size(grid3d->comm, &nprocs);
    for (int i = 0; i < SYMLDL_TIMER_COUNT; ++i) {
        local_max[i].val = local[i];
        local_max[i].rank = rank;
    }
    MPI_Reduce(local, maxv, SYMLDL_TIMER_COUNT, MPI_DOUBLE, MPI_MAX, 0,
               grid3d->comm);
    MPI_Reduce(local, sumv, SYMLDL_TIMER_COUNT, MPI_DOUBLE, MPI_SUM, 0,
               grid3d->comm);
    MPI_Reduce(local_max, global_max, SYMLDL_TIMER_COUNT, MPI_DOUBLE_INT,
               MPI_MAXLOC, 0, grid3d->comm);

    if (rank == 0) {
        static const char *names[SYMLDL_TIMER_COUNT] = {
            "metadata", "workspace", "B_to_X", "solve_core",
            "L_wall", "forward_compute", "forward_z_reduce",
            "forward_z_bcast", "D_wall", "diag_compute", "LT_wall",
            "backward_compute", "X_to_B", "gpu_h2d", "gpu_compute",
            "gpu_d2h", "host_panel_copy", "nvshmem_setup",
            "cpu_event_setup", "cpu_event_progress", "cpu_event_numeric"
        };
        printf("SymLDL V2 solve timing (max_rank / avg_rank / max_rank_id):\n");
        for (int i = 0; i < SYMLDL_TIMER_COUNT; ++i)
            printf("  %-17s %.6f / %.6f / %d\n", names[i], maxv[i],
                   sumv[i] / (double) nprocs, global_max[i].rank);
    }
}

static void
pdgstrs3d_symldl_cpu_comm_print(
    const dSymLDLCPUSolveCommStats *stats, gridinfo3d_t *grid3d)
{
    enum { COMM_CLASS_COUNT = 5 };
    unsigned long long local_messages[COMM_CLASS_COUNT];
    unsigned long long local_bytes[COMM_CLASS_COUNT];
    unsigned long long sum_messages[COMM_CLASS_COUNT];
    unsigned long long sum_bytes[COMM_CLASS_COUNT];
    unsigned long long max_messages[COMM_CLASS_COUNT];
    unsigned long long max_bytes[COMM_CLASS_COUNT];
    int rank;

    local_messages[0] = stats->forward_x_messages;
    local_messages[1] = stats->forward_partial_messages;
    local_messages[2] = stats->backward_x_messages;
    local_messages[3] = stats->backward_partial_messages;
    local_messages[4] = local_messages[0] + local_messages[1] +
                        local_messages[2] + local_messages[3];
    local_bytes[0] = stats->forward_x_bytes;
    local_bytes[1] = stats->forward_partial_bytes;
    local_bytes[2] = stats->backward_x_bytes;
    local_bytes[3] = stats->backward_partial_bytes;
    local_bytes[4] = local_bytes[0] + local_bytes[1] +
                     local_bytes[2] + local_bytes[3];

    MPI_Comm_rank(grid3d->comm, &rank);
    MPI_Reduce(local_messages, sum_messages, COMM_CLASS_COUNT,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(local_bytes, sum_bytes, COMM_CLASS_COUNT,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, grid3d->comm);
    MPI_Reduce(local_messages, max_messages, COMM_CLASS_COUNT,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, grid3d->comm);
    MPI_Reduce(local_bytes, max_bytes, COMM_CLASS_COUNT,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, grid3d->comm);

    if (rank == 0) {
        static const char *names[COMM_CLASS_COUNT] = {
            "L_x_bcast", "L_partial_reduce", "LT_x_bcast",
            "LT_partial_reduce", "total"
        };
        printf("SymLDL V2 CPU solve communication "
               "(sum_messages / sum_bytes / max_rank_messages / "
               "max_rank_bytes):\n");
        for (int i = 0; i < COMM_CLASS_COUNT; ++i)
            printf("  %-17s %llu / %llu / %llu / %llu\n", names[i],
                   sum_messages[i], sum_bytes[i], max_messages[i],
                   max_bytes[i]);
    }
}

static void
pdgstrs3d_symldl_distributed(superlu_dist_options_t *options, int_t n,
           dLUstruct_t *LUstruct, dScalePermstruct_t *ScalePermstruct,
           dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d, double *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           dSOLVEstruct_t *SOLVEstruct, SuperLUStat_t *stat, int *info)
{
    gridinfo_t *grid = &grid3d->grid2d;
    MPI_Comm global_comm = grid3d->comm;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    int_t *ilsum = Llu->ilsum;
    int *supernodeMask = trf3Dpartition->supernodeMask;
    int_t nsupers = supno[n - 1] + 1;
    int_t nlb = trf3Dpartition->symV2RowLocalIndex != NULL
                    ? SUPERLU_MAX((int_t) 1,
                                  trf3Dpartition->symV2LocalRowCount)
                    : CEILING(nsupers, grid->nprow);
    int_t ldalsum = Llu->ldalsum;
    int_t x_count = pdgstrs3d_checked_workspace_count(
        ldalsum, nrhs, nlb, XK_H, "3D SymLDL solve x workspace");
    int global_rank;
    int global_nprocs;
    int superlu_acc_offload = pdgstrs3d_symldl_factor_offload(options);
    int superlu_acc_solve = sp_ienv_dist(11, options);
    int superlu_n_gemm = sp_ienv_dist(7, options);
    pdgstrs3d_symldl_solve_meta_t *solve_meta;
    pdgstrs3d_symldl_workspace_t *workspace;
    pdgstrs3d_symldl_panel_meta_t *panel_meta;
    double *x;
    double tx;
    double tx_st;
    double ttmp;
    pdgstrs3d_symldl_timer_t symldl_timer;
    dSymLDLCPUSolveCommStats cpu_comm_stats;
    xtrsTimer_t xtrsTimer;

    (void) info;
    MPI_Comm_rank(global_comm, &global_rank);
    MPI_Comm_size(global_comm, &global_nprocs);
    memset(&symldl_timer, 0, sizeof(symldl_timer));
    memset(&cpu_comm_stats, 0, sizeof(cpu_comm_stats));

    ttmp = SuperLU_timer_();
    solve_meta = (pdgstrs3d_symldl_solve_meta_t *)
        SOLVEstruct->symldl_v2_solve_meta;
    if (!pdgstrs3d_symldl_solve_meta_valid(
            solve_meta, n, nsupers, nrhs, Glu_persist, Llu,
            trf3Dpartition, supernodeMask, grid3d, global_nprocs,
            superlu_acc_offload, superlu_acc_solve, superlu_n_gemm))
        ABORT("SymLDL solve requires prebuilt V2 solve metadata.");
    solve_meta->reused = 1;
    pdgstrs3d_symldl_reset_solve_stats(solve_meta);
    panel_meta = solve_meta->panel_meta;
    symldl_timer.metadata = SuperLU_timer_() - ttmp;

    ttmp = SuperLU_timer_();
    pdgstrs3d_symldl_workspace_prepare_x(solve_meta, x_count);
    if (pdgstrs3d_symldl_use_nvshmem(solve_meta)) {
        pdgstrs3d_symldl_nvshmem_prepare(
            solve_meta, n, nlb, ldalsum, nrhs, grid3d);
    } else {
        pdgstrs3d_symldl_prepare_host_factor_panels(solve_meta);
        if (solve_meta->cpu_solve_state == NULL) {
            solve_meta->cpu_solve_state = dSymLDLCPUSolveCreate(
                solve_meta->solve_graph, nrhs, trf3Dpartition, grid3d);
            if (solve_meta->cpu_solve_state == NULL)
                ABORT("SymLDL CPU solve setup failed.");
        }
    }
    pdgstrs3d_symldl_redistribution_create(
        &solve_meta->redistribution, n, m_loc, nrhs, fst_row,
        ScalePermstruct, Glu_persist, trf3Dpartition, grid3d,
        SOLVEstruct, ilsum);
    workspace = &solve_meta->work;
    x = workspace->x;
    symldl_timer.workspace = SuperLU_timer_() - ttmp;

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;
    initTRStimer(&xtrsTimer, grid);

    tx = SuperLU_timer_();
    pdReDistribute3d_B_to_X_symv2(
        B, m_loc, nrhs, ldb, fst_row, ilsum, x, ScalePermstruct,
        Glu_persist, trf3Dpartition, grid3d, SOLVEstruct,
        &solve_meta->redistribution);
    xtrsTimer.t_pxReDistribute_B_to_X = SuperLU_timer_() - tx;
    symldl_timer.b_to_x = xtrsTimer.t_pxReDistribute_B_to_X;

    MPI_Barrier(global_comm);
    tx_st = SuperLU_timer_();

    tx = SuperLU_timer_();
    if (solve_meta->nvshmem_state != NULL) {
        if (dSymLDLNVSHMEMForward(solve_meta->nvshmem_state,
                                  x, x_count) != 0)
            ABORT("SymLDL NVSHMEM forward solve failed.");
    } else {
        if (solve_meta->cpu_solve_state == NULL ||
            dSymLDLCPUForward(
                (dSymLDLCPUSolveHandle *) solve_meta->cpu_solve_state,
                x, x_count, nrhs) != 0)
            ABORT("SymLDL CPU forward solve failed.");
        for (int_t edge = 0; edge < solve_meta->solve_graph->block_count;
             ++edge) {
            const dSymLDLBlockDesc *block =
                &solve_meta->solve_graph->blocks[edge];
            const dSymLDLPanelDesc *panel =
                &solve_meta->solve_graph->panels[block->panel_id];
            stat->ops[SOLVE] += (flops_t) (
                2.0 * (double) block->nbrow * (double) panel->width *
                (double) nrhs);
        }
    }
    xtrsTimer.t_forwardSolve = SuperLU_timer_() - tx;
    symldl_timer.forward_wall = xtrsTimer.t_forwardSolve;

    tx = SuperLU_timer_();
    if (solve_meta->nvshmem_state != NULL) {
        if (dSymLDLNVSHMEMDiagonal(solve_meta->nvshmem_state,
                                   x, x_count) != 0)
            ABORT("SymLDL NVSHMEM diagonal solve failed.");
        for (int_t k = 0; k < nsupers; ++k)
            if (global_rank == solve_meta->diag_owner[k]) {
                int_t width = SuperSize(k);
                stat->ops[SOLVE] += (flops_t) (
                    2.0 * (double) width * (double) width * (double) nrhs);
            }
    } else {
        if (dSymLDLCPUDiagonal(
                (dSymLDLCPUSolveHandle *) solve_meta->cpu_solve_state,
                x, x_count, nrhs) != 0)
            ABORT("SymLDL CPU diagonal solve failed.");
        int myrow = MYROW(grid->iam, grid);
        int mycol = MYCOL(grid->iam, grid);
        for (int_t slot = 0; slot < solve_meta->solve_graph->panel_count;
             ++slot) {
            const dSymLDLPanelDesc *panel =
                &solve_meta->solve_graph->panels[slot];
            if (myrow == solve_meta->solve_graph->diag_roots[panel->gid] &&
                mycol == solve_meta->solve_graph->panel_roots[panel->gid])
                stat->ops[SOLVE] += (flops_t) (
                    2.0 * (double) panel->width * (double) panel->width *
                    (double) nrhs);
        }
    }
    symldl_timer.diag_wall = SuperLU_timer_() - tx;

    tx = SuperLU_timer_();
    if (solve_meta->nvshmem_state != NULL) {
        if (dSymLDLNVSHMEMBackward(solve_meta->nvshmem_state,
                                   x, x_count) != 0)
            ABORT("SymLDL NVSHMEM backward solve failed.");
        for (int_t k = 0; k < nsupers; ++k) {
            pdgstrs3d_symldl_panel_meta_t *panel = &panel_meta[k];
            if (!panel->has_panel ||
                !pdgstrs3d_symldl_local_panel_active(
                    trf3Dpartition, panel, k))
                continue;
            stat->ops[SOLVE] += (flops_t)
                pdgstrs3d_symldl_panel_solve_ops(
                    panel, SuperSize(k), nrhs);
        }
    } else {
        if (dSymLDLCPUBackward(
                (dSymLDLCPUSolveHandle *) solve_meta->cpu_solve_state,
                x, x_count, nrhs) != 0)
            ABORT("SymLDL CPU backward solve failed.");
        for (int_t edge = 0; edge < solve_meta->solve_graph->block_count;
             ++edge) {
            const dSymLDLBlockDesc *block =
                &solve_meta->solve_graph->blocks[edge];
            const dSymLDLPanelDesc *panel =
                &solve_meta->solve_graph->panels[block->panel_id];
            stat->ops[SOLVE] += (flops_t) (
                2.0 * (double) block->nbrow * (double) panel->width *
                (double) nrhs);
        }
    }
    xtrsTimer.t_backwardSolve = SuperLU_timer_() - tx;
    symldl_timer.backward_wall = xtrsTimer.t_backwardSolve;

    MPI_Barrier(global_comm);
    stat->utime[SOLVE] = SuperLU_timer_() - tx_st;
    symldl_timer.solve_core = stat->utime[SOLVE];

    tx = SuperLU_timer_();
    pdReDistribute3d_X_to_B_symv2(
        n, B, m_loc, ldb, fst_row, nrhs, x, ilsum, ScalePermstruct,
        Glu_persist, trf3Dpartition, grid3d, SOLVEstruct,
        &solve_meta->redistribution);
    xtrsTimer.t_pxReDistribute_X_to_B = SuperLU_timer_() - tx;
    symldl_timer.x_to_b = xtrsTimer.t_pxReDistribute_X_to_B;

    reduceStat(SOLVE, stat, grid3d);
    pdgstrs3d_symldl_nvshmem_take_timers(solve_meta, &symldl_timer);
    pdgstrs3d_symldl_cpu_take_timers(
        solve_meta, &symldl_timer, &cpu_comm_stats);
    if (pdgstrs3d_symldl_env_enabled("GPU3DV2_SYM_SOLVE_TIMING") ||
        pdgstrs3d_symldl_env_enabled("SUPERLU_SOLVE_PHASE_TIMING")) {
        pdgstrs3d_symldl_timer_print(&symldl_timer, solve_meta, grid3d);
        if (solve_meta->cpu_solve_state != NULL)
            pdgstrs3d_symldl_cpu_comm_print(&cpu_comm_stats, grid3d);
    }

#if (PRNTlevel >= 1)
    printTRStimer(&xtrsTimer, grid3d);
#endif
}
/*! \brief
 *
 *   LDL-native solve for the SymFact GPU3D v2 factor path. This uses the
 *   tree-scheduled distributed solve for all process grids, including the
 *   1x1x1 local case, while keeping the existing B<->X redistribution
 *   contract.
 */
void
pdgstrs3d_symldl (superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
           dScalePermstruct_t * ScalePermstruct,
           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, double *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info)
{
    gridinfo_t *grid = &(grid3d->grid2d);

    *info = 0;
    if (n < 0) *info = -1;
    else if (nrhs < 0) *info = -9;
    if (*info) {
        pxerr_dist("PDGSTRS_SYMLDL", grid, -*info);
        return;
    }
    if (nrhs == 0) return;
    pdgstrs3d_symldl_distributed(options, n, LUstruct, ScalePermstruct,
                                 trf3Dpartition, grid3d, B, m_loc,
                                 fst_row, ldb, nrhs, SOLVEstruct, stat, info);
    return;
}                               /* pdgstrs3d_symldl */
