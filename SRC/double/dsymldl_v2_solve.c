/*! @file
 * \brief LDL-native solve for the SymFact GPU3D v2 factor path.
 */
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "superlu_ddefs.h"
#include "superlu_upacked.h"
#include "dsymldl_v2_nvshmem_solve.h"
#include "dsymldl_v2_literature_forward.h"

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

static int_t
pdReDistribute3d_B_to_X_symv2(double *B, int_t m_loc, int nrhs, int_t ldb,
                         int_t fst_row, int_t * ilsum, double *x,
                         dScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist,
                         dtrf3Dpartition_t *trf3Dpartition,
                         gridinfo3d_t * grid3d, dSOLVEstruct_t * SOLVEstruct)
{
    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *sdispls_nrhs, *rdispls, *rdispls_nrhs;
    int *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *perm_r, *perm_c;
    int_t *send_ibuf, *recv_ibuf;
    double *send_dbuf, *recv_dbuf;
    int_t *xsup, *supno;
    int_t i, ii, irow, gbi, jj, k, knsupc, l, lk;
    int p, procs;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

    MPI_Comm_size(grid3d->comm, &procs);
    perm_r = ScalePermstruct->perm_r;
    perm_c = ScalePermstruct->perm_c;
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    SendCnt = gstrs_comm->B_to_X_SendCnt;
    SendCnt_nrhs = gstrs_comm->B_to_X_SendCnt + procs;
    RecvCnt = gstrs_comm->B_to_X_SendCnt + 2 * procs;
    RecvCnt_nrhs = gstrs_comm->B_to_X_SendCnt + 3 * procs;
    sdispls = gstrs_comm->B_to_X_SendCnt + 4 * procs;
    sdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 5 * procs;
    rdispls = gstrs_comm->B_to_X_SendCnt + 6 * procs;
    rdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 7 * procs;
    ptr_to_ibuf = gstrs_comm->ptr_to_ibuf;
    ptr_to_dbuf = gstrs_comm->ptr_to_dbuf;

    k = sdispls[procs - 1] + SendCnt[procs - 1];
    l = rdispls[procs - 1] + RecvCnt[procs - 1];
    if (!(send_ibuf = intMalloc_dist(k + l)))
        ABORT("Malloc fails for SymV2 send_ibuf[].");
    recv_ibuf = send_ibuf + k;
    if (!(send_dbuf = doubleMalloc_dist((k + l) * (size_t) nrhs)))
        ABORT("Malloc fails for SymV2 send_dbuf[].");
    recv_dbuf = send_dbuf + k * nrhs;

    for (p = 0; p < procs; ++p) {
        ptr_to_ibuf[p] = sdispls[p];
        ptr_to_dbuf[p] = sdispls[p] * nrhs;
    }

    if (!grid3d->zscp.Iam) {
        for (i = 0, l = fst_row; i < m_loc; ++i, ++l) {
            irow = perm_c[perm_r[l]];
            gbi = BlockNum(irow);
            p = pdgstrs3d_symv2_owner(trf3Dpartition, gbi);
            k = ptr_to_ibuf[p];
            send_ibuf[k] = irow;
            k = ptr_to_dbuf[p];
            for (int_t j = 0; j < nrhs; ++j)
                send_dbuf[k++] = B[i + j * ldb];
            ++ptr_to_ibuf[p];
            ptr_to_dbuf[p] += nrhs;
        }
    }

    MPI_Alltoallv(send_ibuf, SendCnt, sdispls, mpi_int_t,
                  recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid3d->comm);
    MPI_Alltoallv(send_dbuf, SendCnt_nrhs, sdispls_nrhs, MPI_DOUBLE,
                  recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, MPI_DOUBLE,
                  grid3d->comm);

    ii = 0;
    for (p = 0; p < procs; ++p) {
        jj = rdispls_nrhs[p];
        for (int_t i = 0; i < RecvCnt[p]; ++i) {
            irow = recv_ibuf[ii];
            k = BlockNum(irow);
            knsupc = SuperSize(k);
            lk = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
            l = X_BLK(lk);
            x[l - XK_H] = k;
            irow -= FstBlockC(k);
            for (int_t j = 0; j < nrhs; ++j)
                x[l + irow + j * knsupc] = recv_dbuf[jj++];
            ++ii;
        }
    }

    SUPERLU_FREE(send_ibuf);
    SUPERLU_FREE(send_dbuf);
    return 0;
}

static int_t
pdReDistribute3d_X_to_B_symv2(int_t n, double *B, int_t m_loc, int_t ldb,
                         int_t fst_row, int nrhs, double *x, int_t * ilsum,
                         dScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist,
                         dtrf3Dpartition_t *trf3Dpartition,
                         gridinfo3d_t * grid3d, dSOLVEstruct_t * SOLVEstruct)
{
    int_t i, irow, jj, k, knsupc, nsupers, l, lk;
    int_t *xsup;
    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *rdispls, *sdispls_nrhs, *rdispls_nrhs;
    int *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *send_ibuf, *recv_ibuf;
    double *send_dbuf, *recv_dbuf;
    int iam, p, q, procs;

    xsup = Glu_persist->xsup;
    nsupers = Glu_persist->supno[n - 1] + 1;
    iam = grid3d->iam;
    MPI_Comm_size(grid3d->comm, &procs);
    int_t *row_to_proc = SOLVEstruct->row_to_proc;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

    SendCnt = gstrs_comm->X_to_B_SendCnt;
    SendCnt_nrhs = gstrs_comm->X_to_B_SendCnt + procs;
    RecvCnt = gstrs_comm->X_to_B_SendCnt + 2 * procs;
    RecvCnt_nrhs = gstrs_comm->X_to_B_SendCnt + 3 * procs;
    sdispls = gstrs_comm->X_to_B_SendCnt + 4 * procs;
    sdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 5 * procs;
    rdispls = gstrs_comm->X_to_B_SendCnt + 6 * procs;
    rdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 7 * procs;
    ptr_to_ibuf = gstrs_comm->ptr_to_ibuf;
    ptr_to_dbuf = gstrs_comm->ptr_to_dbuf;

    k = sdispls[procs - 1] + SendCnt[procs - 1];
    l = rdispls[procs - 1] + RecvCnt[procs - 1];
    if (!(send_ibuf = intMalloc_dist(k + l)))
        ABORT("Malloc fails for SymV2 X send_ibuf[].");
    recv_ibuf = send_ibuf + k;
    if (!(send_dbuf = doubleMalloc_dist((k + l) * nrhs)))
        ABORT("Malloc fails for SymV2 X send_dbuf[].");
    recv_dbuf = send_dbuf + k * nrhs;

    for (p = 0; p < procs; ++p) {
        ptr_to_ibuf[p] = sdispls[p];
        ptr_to_dbuf[p] = sdispls_nrhs[p];
    }

    for (k = 0; k < nsupers; ++k) {
        p = pdgstrs3d_symv2_owner(trf3Dpartition, k);
        if (iam != p)
            continue;
        knsupc = SuperSize(k);
        lk = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
        irow = FstBlockC(k);
        l = X_BLK(lk);
        for (i = 0; i < knsupc; ++i) {
            q = row_to_proc[irow];
            jj = ptr_to_ibuf[q];
            send_ibuf[jj] = irow;
            jj = ptr_to_dbuf[q];
            for (int_t j = 0; j < nrhs; ++j)
                send_dbuf[jj++] = x[l + i + j * knsupc];
            ++ptr_to_ibuf[q];
            ptr_to_dbuf[q] += nrhs;
            ++irow;
        }
    }

    MPI_Alltoallv(send_ibuf, SendCnt, sdispls, mpi_int_t,
                  recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid3d->comm);
    MPI_Alltoallv(send_dbuf, SendCnt_nrhs, sdispls_nrhs, MPI_DOUBLE,
                  recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, MPI_DOUBLE,
                  grid3d->comm);

    int_t total_recv = rdispls[procs - 1] + RecvCnt[procs - 1];
    for (i = 0, k = 0; i < total_recv; ++i) {
        irow = recv_ibuf[i] - fst_row;
        for (int_t j = 0; j < nrhs; ++j)
            B[irow + j * ldb] = recv_dbuf[k++];
    }

    SUPERLU_FREE(send_ibuf);
    SUPERLU_FREE(send_dbuf);
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
pdgstrs3d_symldl_dgemm(const char *transa, const char *transb,
                       int_t m_in, int_t n_in, int_t k_in,
                       double alpha, double *a, int_t lda_in,
                       double *b, int_t ldb_in,
                       double beta, double *c, int_t ldc_in)
{
    int m = pdgstrs3d_symldl_count_to_int(m_in, "SymLDL BLAS m");
    int n = pdgstrs3d_symldl_count_to_int(n_in, "SymLDL BLAS n");
    int k = pdgstrs3d_symldl_count_to_int(k_in, "SymLDL BLAS k");
    int lda = pdgstrs3d_symldl_count_to_int(lda_in, "SymLDL BLAS lda");
    int ldb = pdgstrs3d_symldl_count_to_int(ldb_in, "SymLDL BLAS ldb");
    int ldc = pdgstrs3d_symldl_count_to_int(ldc_in, "SymLDL BLAS ldc");

    if (m == 0 || n == 0)
        return;

#if defined (USE_VENDOR_BLAS)
    dgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb,
           &beta, c, &ldc, 1, 1);
#else
    dgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb,
           &beta, c, &ldc);
#endif
}

static void
pdgstrs3d_symldl_counts_to_displs(int nprocs, int *counts, int *displs,
                                  int *total)
{
    int p;
    displs[0] = 0;
    *total = counts[0];
    for (p = 1; p < nprocs; ++p) {
        displs[p] = displs[p - 1] + counts[p - 1];
        *total += counts[p];
    }
}

static void
pdgstrs3d_symldl_grow_int_t_buffer(int_t **buffer, int_t *capacity,
                                   int_t need, const char *what)
{
    if (need <= *capacity)
        return;
    if (*buffer)
        SUPERLU_FREE(*buffer);
    if (!(*buffer = intMalloc_dist(need)))
        ABORT(what);
    *capacity = need;
}

static void
pdgstrs3d_symldl_grow_int_buffer(int **buffer, int_t *capacity,
                                 int_t need, const char *what)
{
    if (need <= *capacity)
        return;
    if (*buffer)
        SUPERLU_FREE(*buffer);
    if (!(*buffer = (int *) SUPERLU_MALLOC(pdgstrs3d_checked_product(
              (size_t) need, sizeof(int), what))))
        ABORT(what);
    *capacity = need;
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

static void
pdgstrs3d_symldl_grow_request_buffer(MPI_Request **buffer, int_t *capacity,
                                     int_t need, const char *what)
{
    if (need <= *capacity)
        return;
    if (*buffer)
        SUPERLU_FREE(*buffer);
    if (!(*buffer = (MPI_Request *) SUPERLU_MALLOC(
              pdgstrs3d_checked_product((size_t) need,
                                        sizeof(MPI_Request), what))))
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
    int_t node;
    int_t pos;
} pdgstrs3d_symldl_node_order_t;

typedef struct {
    MPI_Comm comm;
    int active;
    int rank;
    int nprocs;
    int *global_to_local;
} pdgstrs3d_symldl_tree_comm_t;

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
    int active;
    int nprocs;
    int total_send;
    int total_recv;
    int total_send_vals;
    int total_recv_vals;
    int needs_xk;
    int xk_receiver_count;
    int has_diag;
    int diag_rank_count;
    int *counts;
    int *send_counts;
    int *send_displs;
    int *recv_counts;
    int *recv_displs;
    int *send_val_counts;
    int *send_val_displs;
    int *recv_val_counts;
    int *recv_val_displs;
    int *row_to_send_pos;
    int *send_seq;
    int *xk_receivers;
    int *diag_ranks;
    int_t *recv_rows;
} pdgstrs3d_symldl_comm_meta_t;

typedef struct {
    int_t nlevels;
    int_t *level_ptr;
    int_t *nodes;
} pdgstrs3d_symldl_level_schedule_t;

typedef struct {
    double *values;
    unsigned char *valid;
    int_t row_count;
} pdgstrs3d_symldl_x_cache_t;

typedef struct {
    double *x;
    double *xk_buf;
    double *diag_send_buf;
    double *diag_buf;
    double *delta_send_buf;
    double *delta_buf;
    double *gemm_buf;
    double *rhs_buf;
    double *send_vals_buf;
    double *recv_vals_buf;
    double *row_values_buf;
    double *request_values_buf;
    double *recv_request_values_buf;
    double *delta_recv_buf;
    MPI_Request *comm_reqs;
    struct pdgstrs3d_symldl_node_ctx_s *level_ctxs;
    int_t x_cap;
    int_t xk_cap;
    int_t diag_send_cap;
    int_t diag_cap;
    int_t delta_send_cap;
    int_t delta_cap;
    int_t gemm_cap;
    int_t rhs_cap;
    int_t send_vals_cap;
    int_t recv_vals_cap;
    int_t row_values_cap;
    int_t request_values_cap;
    int_t recv_request_values_cap;
    int_t delta_recv_cap;
    int_t comm_reqs_cap;
    int_t level_ctxs_cap;
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
    int superlu_n_gemm;
    int numForests;
    int_t max_panel_block_rows;
    Glu_persist_t *Glu_persist;
    dLocalLU_t *Llu;
    dtrf3Dpartition_t *trf3Dpartition;
    int *supernodeMask;
    int *diag_owner;
    int_t *solve_order;
    pdgstrs3d_symldl_level_schedule_t solve_schedule;
    pdgstrs3d_symldl_tree_comm_t *tree_comms;
    pdgstrs3d_symldl_panel_meta_t *panel_meta;
    pdgstrs3d_symldl_comm_meta_t *comm_meta;
    pdgstrs3d_symldl_x_cache_t *x_cache;
    pdgstrs3d_symldl_workspace_t work;
    void *nvshmem_state;
    void *literature_forward_state;
    void *factor_gpu_handle;
    double cpu_blas_ops;
    double cpu_blas_calls;
    double host_panel_copy_time;
    double x_cache_fill_time;
    double x_cache_replicated_bytes;
    double x_cache_avoided_request_bytes;
    double x_cache_hits;
    double x_cache_misses;
    double x_cache_panels;
    int factor_gpu_synchronized;
    int literature_forward_mode;
    int reused;
} pdgstrs3d_symldl_solve_meta_t;

typedef struct {
    double metadata;
    double workspace;
    double b_to_x;
    double forward_xk;
    double forward_compute;
    double forward_values;
    double forward_apply;
    double diag_comm;
    double diag_compute;
    double x_cache_fill;
    double backward_values;
    double backward_compute;
    double backward_delta;
    double x_to_b;
    double gpu_h2d;
    double gpu_compute;
    double gpu_d2h;
} pdgstrs3d_symldl_timer_t;

typedef struct pdgstrs3d_symldl_node_ctx_s {
    int active;
    int_t k;
    int_t ksupc;
    int xk_count;
    int delta_count;
    int root_rank;
    int solve_rank;
    MPI_Comm solve_comm;
    pdgstrs3d_symldl_panel_meta_t *kmeta;
    pdgstrs3d_symldl_comm_meta_t *cmeta;
    double *xk_buf;
    double *send_vals;
    double *recv_vals;
    double *row_values;
    double *request_values;
    double *recv_request_values;
    double *delta_send_buf;
    double *delta_buf;
    double *delta_recv_buf;
} pdgstrs3d_symldl_node_ctx_t;

static void
pdgstrs3d_symldl_reset_solve_stats(pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL)
        return;
    meta->cpu_blas_ops = 0.0;
    meta->cpu_blas_calls = 0.0;
    meta->host_panel_copy_time = 0.0;
    meta->x_cache_fill_time = 0.0;
    meta->x_cache_replicated_bytes = 0.0;
    meta->x_cache_avoided_request_bytes = 0.0;
    meta->x_cache_hits = 0.0;
    meta->x_cache_misses = 0.0;
    meta->x_cache_panels = 0.0;
    meta->factor_gpu_synchronized = 0;
}

static void
pdgstrs3d_symldl_note_cpu_blas(pdgstrs3d_symldl_solve_meta_t *meta,
                               double ops)
{
    if (meta == NULL)
        return;
    meta->cpu_blas_ops += ops;
    meta->cpu_blas_calls += 1.0;
}

static void
pdgstrs3d_symldl_note_cpu_direct_panel(pdgstrs3d_symldl_solve_meta_t *meta,
                                       double ops, double calls)
{
    if (meta == NULL || calls == 0.0)
        return;
    meta->cpu_blas_ops += ops;
    meta->cpu_blas_calls += calls;
}

static double
pdgstrs3d_symldl_cpu_panel_min_ops(pdgstrs3d_symldl_solve_meta_t *meta)
{
    return meta != NULL && meta->superlu_n_gemm >= 0
               ? (double) meta->superlu_n_gemm
               : 0.0;
}

static double
pdgstrs3d_symldl_panel_solve_ops(pdgstrs3d_symldl_panel_meta_t *kmeta,
                                 int_t ksupc, int nrhs);

static void
pdgstrs3d_symldl_cpu_gemm(pdgstrs3d_symldl_solve_meta_t *meta,
                          const char *transa, const char *transb,
                          int_t m, int_t n, int_t kdim,
                          double alpha, double *a, int_t lda,
                          double *b, int_t ldb,
                          double beta, double *c, int_t ldc)
{
    double ops = 2.0 * (double) m * (double) n * (double) kdim;
    pdgstrs3d_symldl_note_cpu_blas(meta, ops);
    pdgstrs3d_symldl_dgemm(transa, transb, m, n, kdim, alpha, a, lda,
                           b, ldb, beta, c, ldc);
}

static int
pdgstrs3d_symldl_use_direct_cpu_panel(pdgstrs3d_symldl_solve_meta_t *meta,
                                      double ops, int nrhs)
{
    (void) nrhs;
    return ops < pdgstrs3d_symldl_cpu_panel_min_ops(meta);
}

static void
pdgstrs3d_symldl_cpu_forward_panel_direct(
    pdgstrs3d_symldl_panel_meta_t *kmeta,
    pdgstrs3d_symldl_comm_meta_t *cmeta, int_t ksupc, int nrhs,
    const double *xk, double *send_vals)
{
    double *lusup = kmeta->lusup;
    int_t nsupr = kmeta->nsupr;

    if (nrhs == 1) {
        for (int_t block = 0; block < kmeta->nblocks; ++block) {
            int_t nbrow = kmeta->block_nbrow[block];
            int_t row_start = kmeta->block_row_start[block];
            int_t luptr = kmeta->block_luptr[block];
            for (int_t r = 0; r < nbrow; ++r) {
                double sum = 0.0;
                double *a = &lusup[luptr + r];
                for (int_t c = 0; c < ksupc; ++c)
                    sum += a[c * nsupr] * xk[c];
                send_vals[cmeta->row_to_send_pos[row_start + r]] = -sum;
            }
        }
        return;
    }

    for (int_t block = 0; block < kmeta->nblocks; ++block) {
        int_t nbrow = kmeta->block_nbrow[block];
        int_t row_start = kmeta->block_row_start[block];
        int_t luptr = kmeta->block_luptr[block];
        for (int_t r = 0; r < nbrow; ++r) {
            int pos = cmeta->row_to_send_pos[row_start + r];
            double *a = &lusup[luptr + r];
            for (int rhs = 0; rhs < nrhs; ++rhs) {
                double sum = 0.0;
                const double *xr = &xk[(int_t) rhs * ksupc];
                for (int_t c = 0; c < ksupc; ++c)
                    sum += a[c * nsupr] * xr[c];
                send_vals[(int_t) pos * nrhs + rhs] = -sum;
            }
        }
    }
}

static void
pdgstrs3d_symldl_cpu_backward_panel_direct(
    pdgstrs3d_symldl_panel_meta_t *kmeta, int_t ksupc, int nrhs,
    const double *row_values, double *delta_send)
{
    double *lusup = kmeta->lusup;
    int_t nsupr = kmeta->nsupr;

    if (nrhs == 1) {
        for (int_t block = 0; block < kmeta->nblocks; ++block) {
            int_t nbrow = kmeta->block_nbrow[block];
            int_t row_start = kmeta->block_row_start[block];
            int_t luptr = kmeta->block_luptr[block];
            for (int_t c = 0; c < ksupc; ++c) {
                double sum = 0.0;
                double *a = &lusup[luptr + c * nsupr];
                for (int_t r = 0; r < nbrow; ++r)
                    sum += a[r] * row_values[row_start + r];
                delta_send[c] -= sum;
            }
        }
        return;
    }

    for (int_t block = 0; block < kmeta->nblocks; ++block) {
        int_t nbrow = kmeta->block_nbrow[block];
        int_t row_start = kmeta->block_row_start[block];
        int_t luptr = kmeta->block_luptr[block];
        for (int rhs = 0; rhs < nrhs; ++rhs) {
            const double *rows = &row_values[(int_t) row_start * nrhs + rhs];
            double *delta = &delta_send[(int_t) rhs * ksupc];
            for (int_t c = 0; c < ksupc; ++c) {
                double sum = 0.0;
                double *a = &lusup[luptr + c * nsupr];
                for (int_t r = 0; r < nbrow; ++r)
                    sum += a[r] * rows[(int_t) r * nrhs];
                delta[c] -= sum;
            }
        }
    }
}

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
pdgstrs3d_symldl_node_order_cmp(const void *a, const void *b)
{
    const pdgstrs3d_symldl_node_order_t *oa =
        (const pdgstrs3d_symldl_node_order_t *) a;
    const pdgstrs3d_symldl_node_order_t *ob =
        (const pdgstrs3d_symldl_node_order_t *) b;

    if (oa->pos < ob->pos) return -1;
    if (oa->pos > ob->pos) return 1;
    if (oa->node < ob->node) return -1;
    if (oa->node > ob->node) return 1;
    return 0;
}

static int_t *
pdgstrs3d_symldl_tree_order(int_t nsupers,
                            dtrf3Dpartition_t *trf3Dpartition,
                            gridinfo3d_t *grid3d)
{
    int_t *order;
    int_t *local_pos = NULL;
    int_t *global_pos = NULL;
    pdgstrs3d_symldl_node_order_t *entries = NULL;
    int_t sentinel = (int_t) INT_MAX;

    if (trf3Dpartition == NULL ||
        trf3Dpartition->sForests == NULL ||
        trf3Dpartition->myTreeIdxs == NULL ||
        trf3Dpartition->myZeroTrIdxs == NULL)
        ABORT("SymLDL solve requires an LDL-native forest schedule.");

    if (!(order = intMalloc_dist(nsupers)))
        ABORT("Malloc fails for SymLDL solve order.");
    if (!(local_pos = intMalloc_dist(nsupers)) ||
        !(global_pos = intMalloc_dist(nsupers)) ||
        !(entries = (pdgstrs3d_symldl_node_order_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) nsupers,
                              sizeof(pdgstrs3d_symldl_node_order_t),
                              "SymLDL tree solve order"))))
        ABORT("Malloc fails for SymLDL tree solve order.");

    for (int_t k = 0; k < nsupers; ++k)
        local_pos[k] = sentinel;

    int maxLvl = trf3Dpartition->maxLvl > 0
                     ? trf3Dpartition->maxLvl
                     : log2i(grid3d->zscp.Np) + 1;
    int_t pos = 0;

    for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {
        if (trf3Dpartition->myZeroTrIdxs[ilvl])
            continue;
        int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
        sForest_t *sforest = trf3Dpartition->sForests[tree];
        if (sforest == NULL)
            continue;
        for (int_t k0 = 0; k0 < sforest->nNodes; ++k0) {
            int_t node = sforest->nodeList[k0];
            if (node >= 0 && node < nsupers && local_pos[node] == sentinel)
                local_pos[node] = pos;
            ++pos;
        }
    }

    MPI_Allreduce(local_pos, global_pos,
                  pdgstrs3d_symldl_count_to_int(nsupers,
                                                "SymLDL tree solve order"),
                  mpi_int_t, MPI_MIN, grid3d->comm);
    for (int_t k = 0; k < nsupers; ++k) {
        entries[k].node = k;
        entries[k].pos = global_pos[k];
    }
    qsort(entries, (size_t) nsupers, sizeof(*entries),
          pdgstrs3d_symldl_node_order_cmp);
    for (int_t k = 0; k < nsupers; ++k)
        order[k] = entries[k].node;

    SUPERLU_FREE(entries);
    SUPERLU_FREE(global_pos);
    SUPERLU_FREE(local_pos);
    return order;
}

typedef struct {
    int_t node;
    int_t level;
    int_t order;
    double cost;
} pdgstrs3d_symldl_sched_entry_t;

static int
pdgstrs3d_symldl_sched_entry_cmp(const void *a, const void *b)
{
    const pdgstrs3d_symldl_sched_entry_t *ea =
        (const pdgstrs3d_symldl_sched_entry_t *) a;
    const pdgstrs3d_symldl_sched_entry_t *eb =
        (const pdgstrs3d_symldl_sched_entry_t *) b;

    if (ea->level < eb->level) return -1;
    if (ea->level > eb->level) return 1;
    if (ea->cost > eb->cost) return -1;
    if (ea->cost < eb->cost) return 1;
    if (ea->order < eb->order) return -1;
    if (ea->order > eb->order) return 1;
    if (ea->node < eb->node) return -1;
    if (ea->node > eb->node) return 1;
    return 0;
}

static void
pdgstrs3d_symldl_level_schedule_free(pdgstrs3d_symldl_level_schedule_t *schedule)
{
    if (schedule == NULL)
        return;
    if (schedule->nodes) SUPERLU_FREE(schedule->nodes);
    if (schedule->level_ptr) SUPERLU_FREE(schedule->level_ptr);
    memset(schedule, 0, sizeof(*schedule));
}

static pdgstrs3d_symldl_level_schedule_t
pdgstrs3d_symldl_level_schedule_create(
    int_t nsupers, int_t *solve_order,
    pdgstrs3d_symldl_panel_meta_t *panel_meta,
    gridinfo3d_t *grid3d, Glu_persist_t *Glu_persist)
{
    pdgstrs3d_symldl_level_schedule_t schedule;
    int_t *local_edges = NULL;
    int_t *edges = NULL;
    int *recv_counts = NULL;
    int *recv_displs = NULL;
    int_t *order_pos = NULL;
    int_t *edge_counts = NULL;
    int_t *edge_ptr = NULL;
    int_t *edge_next = NULL;
    int_t *edge_succ = NULL;
    int_t *indegree = NULL;
    int_t *topo_queue = NULL;
    int_t *levels = NULL;
    double *local_cost = NULL;
    double *global_cost = NULL;
    pdgstrs3d_symldl_sched_entry_t *entries = NULL;
    int global_nprocs;
    int local_pair_count_i;
    int total_pair_count = 0;
    int valid_schedule = 1;
    int global_valid_schedule = 1;
    int invalid_kind = 0;
    int_t invalid_pred = -1;
    int_t invalid_succ = -1;
    int_t invalid_pred_pos = -1;
    int_t invalid_succ_pos = -1;
    int_t *xsup = Glu_persist->xsup;
    int_t local_edges_count = 0;
    int_t local_edges_fill = 0;
    int_t total_edges = 0;

    memset(&schedule, 0, sizeof(schedule));
    MPI_Comm_size(grid3d->comm, &global_nprocs);

    for (int_t k = 0; k < nsupers; ++k)
        if (panel_meta[k].has_panel)
            local_edges_count += panel_meta[k].nblocks;

    if (local_edges_count > 0) {
        if (!(local_edges = intMalloc_dist(2 * local_edges_count)))
            ABORT("Malloc fails for SymLDL solve dependency edges.");
    }

    if (!(local_cost = doubleCalloc_dist(nsupers)) ||
        !(global_cost = doubleCalloc_dist(nsupers)))
        ABORT("Calloc fails for SymLDL solve cost metadata.");

    for (int_t k = 0; k < nsupers; ++k) {
        pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
        if (!kmeta->has_panel)
            continue;
        int_t ksupc = SuperSize(k);
        local_cost[k] += (double) ksupc * (double) ksupc;
        for (int_t block = 0; block < kmeta->nblocks; ++block) {
            int_t row_start = kmeta->block_row_start[block];
            int_t nbrow = kmeta->block_nbrow[block];
            if (nbrow <= 0)
                continue;
            int_t succ = Glu_persist->supno[kmeta->rows[row_start]];
            local_edges[2 * local_edges_fill] = k;
            local_edges[2 * local_edges_fill + 1] = succ;
            ++local_edges_fill;
            local_cost[k] += (double) nbrow * (double) ksupc;
        }
    }
    local_edges_count = local_edges_fill;

    if (!(recv_counts = (int *) SUPERLU_MALLOC(
              pdgstrs3d_checked_product((size_t) global_nprocs, sizeof(int),
                                        "SymLDL solve edge counts"))) ||
        !(recv_displs = (int *) SUPERLU_MALLOC(
              pdgstrs3d_checked_product((size_t) global_nprocs, sizeof(int),
                                        "SymLDL solve edge displacements"))))
        ABORT("Malloc fails for SymLDL solve edge metadata.");

    local_pair_count_i = pdgstrs3d_symldl_count_to_int(
        2 * local_edges_count, "SymLDL solve dependency edge count");
    MPI_Allgather(&local_pair_count_i, 1, MPI_INT, recv_counts, 1, MPI_INT,
                  grid3d->comm);
    pdgstrs3d_symldl_counts_to_displs(global_nprocs, recv_counts,
                                      recv_displs, &total_pair_count);
    if (total_pair_count % 2 != 0)
        ABORT("SymLDL solve dependency edge metadata is malformed.");
    total_edges = total_pair_count / 2;

    if (total_edges > 0) {
        if (!(edges = intMalloc_dist(2 * total_edges)))
            ABORT("Malloc fails for SymLDL solve global dependency edges.");
    }

    {
        int_t dummy_edge[2] = {0, 0};
        MPI_Allgatherv(local_edges_count ? local_edges : dummy_edge,
                       local_pair_count_i, mpi_int_t,
                       total_edges ? edges : dummy_edge,
                       recv_counts, recv_displs, mpi_int_t, grid3d->comm);
    }
    MPI_Allreduce(local_cost, global_cost,
                  pdgstrs3d_symldl_count_to_int(nsupers,
                                                "SymLDL solve cost metadata"),
                  MPI_DOUBLE, MPI_SUM, grid3d->comm);

    if (!(order_pos = intMalloc_dist(nsupers)) ||
        !(edge_counts = intCalloc_dist(nsupers)) ||
        !(edge_ptr = intMalloc_dist(nsupers + 1)) ||
        !(indegree = intCalloc_dist(nsupers)) ||
        !(topo_queue = intMalloc_dist(nsupers)) ||
        !(levels = intCalloc_dist(nsupers)) ||
        !(entries = (pdgstrs3d_symldl_sched_entry_t *) SUPERLU_MALLOC(
              pdgstrs3d_checked_product((size_t) nsupers, sizeof(*entries),
                                        "SymLDL solve level entries"))))
        ABORT("Malloc fails for SymLDL solve level metadata.");

    for (int_t pos = 0; pos < nsupers; ++pos)
        order_pos[pos] = -1;
    for (int_t pos = 0; pos < nsupers; ++pos) {
        int_t node = solve_order[pos];
        if (node < 0 || node >= nsupers ||
            (node >= 0 && node < nsupers && order_pos[node] >= 0)) {
            if (!invalid_kind) {
                invalid_kind = 3;
                invalid_pred = node;
                invalid_pred_pos = pos;
            }
            valid_schedule = 0;
            continue;
        }
        order_pos[node] = pos;
    }
    MPI_Allreduce(&valid_schedule, &global_valid_schedule, 1, MPI_INT,
                  MPI_MIN, grid3d->comm);
    if (!global_valid_schedule) {
        int global_rank = -1;
        MPI_Comm_rank(grid3d->comm, &global_rank);
        if (!valid_schedule) {
            fprintf(stderr,
                    "SymLDL solve forest order is invalid on rank %d: "
                    "kind=%d node=%lld pos=%lld nsupers=%lld\n",
                    global_rank, invalid_kind, (long long) invalid_pred,
                    (long long) invalid_pred_pos, (long long) nsupers);
            fflush(stderr);
        }
        ABORT("SymLDL solve forest order is invalid.");
    }

    valid_schedule = 1;
    global_valid_schedule = 1;
    invalid_kind = 0;
    invalid_pred = -1;
    invalid_succ = -1;
    invalid_pred_pos = -1;
    invalid_succ_pos = -1;
    for (int_t e = 0; e < total_edges; ++e) {
        int_t pred = edges[2 * e];
        int_t succ = edges[2 * e + 1];
        if (pred < 0 || pred >= nsupers || succ < 0 || succ >= nsupers) {
            if (!invalid_kind) {
                invalid_kind = 1;
                invalid_pred = pred;
                invalid_succ = succ;
            }
            valid_schedule = 0;
            continue;
        }
        if (pred == succ) {
            if (!invalid_kind) {
                invalid_kind = 2;
                invalid_pred = pred;
                invalid_succ = succ;
                invalid_pred_pos = order_pos[pred];
                invalid_succ_pos = order_pos[succ];
            }
            valid_schedule = 0;
            continue;
        }
        ++edge_counts[pred];
        ++indegree[succ];
    }
    MPI_Allreduce(&valid_schedule, &global_valid_schedule, 1, MPI_INT,
                  MPI_MIN, grid3d->comm);
    if (!global_valid_schedule) {
        int global_rank = -1;
        MPI_Comm_rank(grid3d->comm, &global_rank);
        if (!valid_schedule) {
            fprintf(stderr,
                    "SymLDL solve dependency graph is invalid on rank %d: "
                    "kind=%d pred=%lld succ=%lld pred_pos=%lld succ_pos=%lld "
                    "nsupers=%lld total_edges=%lld\n",
                    global_rank, invalid_kind, (long long) invalid_pred,
                    (long long) invalid_succ, (long long) invalid_pred_pos,
                    (long long) invalid_succ_pos, (long long) nsupers,
                    (long long) total_edges);
            fflush(stderr);
        }
        ABORT("SymLDL solve dependency graph is invalid.");
    }

    edge_ptr[0] = 0;
    for (int_t k = 0; k < nsupers; ++k)
        edge_ptr[k + 1] = edge_ptr[k] + edge_counts[k];
    if (!(edge_succ = intMalloc_dist(SUPERLU_MAX(total_edges, (int_t) 1))) ||
        !(edge_next = intMalloc_dist(nsupers)))
        ABORT("Malloc fails for SymLDL solve adjacency.");
    memcpy(edge_next, edge_ptr,
           pdgstrs3d_checked_alloc_bytes(nsupers, sizeof(int_t),
                                         "SymLDL solve adjacency cursor"));
    for (int_t e = 0; e < total_edges; ++e) {
        int_t pred = edges[2 * e];
        int_t succ = edges[2 * e + 1];
        edge_succ[edge_next[pred]++] = succ;
    }

    int_t max_level = 0;
    int_t queue_head = 0;
    int_t queue_tail = 0;
    for (int_t k = 0; k < nsupers; ++k)
        if (indegree[k] == 0)
            topo_queue[queue_tail++] = k;

    while (queue_head < queue_tail) {
        int_t k = topo_queue[queue_head++];
        max_level = SUPERLU_MAX(max_level, levels[k]);
        for (int_t p = edge_ptr[k]; p < edge_ptr[k + 1]; ++p) {
            int_t succ = edge_succ[p];
            levels[succ] = SUPERLU_MAX(levels[succ], levels[k] + 1);
            --indegree[succ];
            if (indegree[succ] == 0)
                topo_queue[queue_tail++] = succ;
        }
    }
    if (queue_tail != nsupers) {
        int global_rank = -1;
        int_t first_blocked = -1;
        int_t first_indegree = -1;
        MPI_Comm_rank(grid3d->comm, &global_rank);
        for (int_t k = 0; k < nsupers; ++k) {
            if (indegree[k] > 0) {
                first_blocked = k;
                first_indegree = indegree[k];
                break;
            }
        }
        fprintf(stderr,
                "SymLDL solve dependency graph is cyclic on rank %d: "
                "processed=%lld nsupers=%lld first_blocked=%lld "
                "remaining_indegree=%lld total_edges=%lld\n",
                global_rank, (long long) queue_tail, (long long) nsupers,
                (long long) first_blocked, (long long) first_indegree,
                (long long) total_edges);
        fflush(stderr);
        ABORT("SymLDL solve dependency graph is cyclic.");
    }

    schedule.nlevels = max_level + 1;
    if (!(schedule.level_ptr = intCalloc_dist(schedule.nlevels + 1)) ||
        !(schedule.nodes = intMalloc_dist(nsupers)))
        ABORT("Malloc fails for SymLDL solve level schedule.");

    for (int_t k = 0; k < nsupers; ++k) {
        entries[k].node = k;
        entries[k].level = levels[k];
        entries[k].order = order_pos[k];
        entries[k].cost = global_cost[k];
        ++schedule.level_ptr[levels[k] + 1];
    }
    for (int_t level = 0; level < schedule.nlevels; ++level)
        schedule.level_ptr[level + 1] += schedule.level_ptr[level];
    qsort(entries, (size_t) nsupers, sizeof(*entries),
          pdgstrs3d_symldl_sched_entry_cmp);
    for (int_t i = 0; i < nsupers; ++i)
        schedule.nodes[i] = entries[i].node;

cleanup:
    if (entries) SUPERLU_FREE(entries);
    if (topo_queue) SUPERLU_FREE(topo_queue);
    if (indegree) SUPERLU_FREE(indegree);
    if (edge_next) SUPERLU_FREE(edge_next);
    if (edge_succ) SUPERLU_FREE(edge_succ);
    if (levels) SUPERLU_FREE(levels);
    if (edge_ptr) SUPERLU_FREE(edge_ptr);
    if (edge_counts) SUPERLU_FREE(edge_counts);
    if (order_pos) SUPERLU_FREE(order_pos);
    if (global_cost) SUPERLU_FREE(global_cost);
    if (local_cost) SUPERLU_FREE(local_cost);
    if (edges) SUPERLU_FREE(edges);
    if (recv_displs) SUPERLU_FREE(recv_displs);
    if (recv_counts) SUPERLU_FREE(recv_counts);
    if (local_edges) SUPERLU_FREE(local_edges);
    return schedule;
}

static int
pdgstrs3d_symldl_rank_to_tree_rank(pdgstrs3d_symldl_tree_comm_t *tree_comm,
                                   int global_rank)
{
    int out;

    if (tree_comm == NULL)
        ABORT("SymLDL solve requires a tree communicator.");
    out = tree_comm->global_to_local[global_rank];
    if (out < 0)
        ABORT("SymLDL solve tree communicator is missing a required rank.");
    return out;
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

static int
pdgstrs3d_symldl_num_forests(dtrf3Dpartition_t *trf3Dpartition,
                             gridinfo3d_t *grid3d)
{
    int maxLvl = trf3Dpartition && trf3Dpartition->maxLvl > 0
                     ? trf3Dpartition->maxLvl
                     : log2i(grid3d->zscp.Np) + 1;
    return (1 << maxLvl) - 1;
}

static pdgstrs3d_symldl_tree_comm_t *
pdgstrs3d_symldl_tree_comms_create(dtrf3Dpartition_t *trf3Dpartition,
                                   pdgstrs3d_symldl_panel_meta_t *panel_meta,
                                   int *diag_owner, int_t nsupers,
                                   gridinfo3d_t *grid3d, int global_nprocs)
{
    pdgstrs3d_symldl_tree_comm_t *tree_comms;
    int *local_active = NULL;
    int *global_active = NULL;
    int maxLvl;
    int numForests;
    int global_rank;

    if (trf3Dpartition == NULL ||
        trf3Dpartition->supernode2treeMap == NULL ||
        trf3Dpartition->myTreeIdxs == NULL ||
        trf3Dpartition->myZeroTrIdxs == NULL)
        ABORT("SymLDL solve requires LDL-native tree communicator metadata.");

    maxLvl = trf3Dpartition->maxLvl > 0
                 ? trf3Dpartition->maxLvl
                 : log2i(grid3d->zscp.Np) + 1;
    numForests = (1 << maxLvl) - 1;
    MPI_Comm_rank(grid3d->comm, &global_rank);

    size_t active_count = pdgstrs3d_checked_product((size_t) numForests,
                                                    (size_t) global_nprocs,
                                                    "SymLDL tree active ranks");
    int_t active_count_t = pdgstrs3d_checked_size_to_int_t(
        active_count, "SymLDL tree active ranks");
    int active_count_i = pdgstrs3d_symldl_count_to_int(
        active_count_t, "SymLDL tree active ranks");
    if (!(local_active = (int *) SUPERLU_MALLOC(
              pdgstrs3d_checked_alloc_bytes(active_count_t, sizeof(int),
                                            "SymLDL tree active ranks"))) ||
        !(global_active = (int *) SUPERLU_MALLOC(
              pdgstrs3d_checked_alloc_bytes(active_count_t, sizeof(int),
                                            "SymLDL tree active ranks"))))
        ABORT("Calloc fails for SymLDL tree active rank metadata.");
    memset(local_active, 0,
           pdgstrs3d_checked_alloc_bytes(active_count_t, sizeof(int),
                                         "SymLDL tree active ranks"));
    memset(global_active, 0,
           pdgstrs3d_checked_alloc_bytes(active_count_t, sizeof(int),
                                         "SymLDL tree active ranks"));

    for (int tree = 0; tree < numForests; ++tree) {
        if (grid3d->zscp.Iam == 0)
            local_active[tree * global_nprocs + global_rank] = 1;
    }

    for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {
        if (!trf3Dpartition->myZeroTrIdxs[ilvl]) {
            int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
            if (tree >= 0 && tree < numForests)
                local_active[tree * global_nprocs + global_rank] = 1;
        }
    }

    if (diag_owner != NULL) {
        for (int_t k = 0; k < nsupers; ++k) {
            int_t tree = trf3Dpartition->supernode2treeMap[k];
            int owner = diag_owner[k];
            if (tree >= 0 && tree < numForests &&
                owner >= 0 && owner < global_nprocs)
                local_active[tree * global_nprocs + owner] = 1;
        }
    }

    if (panel_meta != NULL) {
        for (int_t k = 0; k < nsupers; ++k) {
            int_t tree = trf3Dpartition->supernode2treeMap[k];
            pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
            if (tree < 0 || tree >= numForests ||
                !pdgstrs3d_symldl_local_panel_active(trf3Dpartition,
                                                     kmeta, k))
                continue;
            local_active[tree * global_nprocs + global_rank] = 1;
            for (int_t row = 0; row < kmeta->row_count; ++row) {
                int dest = kmeta->row_dest_global[row];
                if (dest >= 0 && dest < global_nprocs)
                    local_active[tree * global_nprocs + dest] = 1;
            }
        }
    }

    MPI_Allreduce(local_active, global_active, active_count_i, MPI_INT,
                  MPI_MAX, grid3d->comm);

    if (!(tree_comms = (pdgstrs3d_symldl_tree_comm_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) numForests,
                              sizeof(pdgstrs3d_symldl_tree_comm_t),
                              "SymLDL tree communicators"))))
        ABORT("Malloc fails for SymLDL tree communicators.");

    for (int tree = 0; tree < numForests; ++tree) {
        int active = global_active[tree * global_nprocs + global_rank];

        tree_comms[tree].comm = MPI_COMM_NULL;
        tree_comms[tree].active = 0;
        tree_comms[tree].rank = -1;
        tree_comms[tree].nprocs = 0;
        tree_comms[tree].global_to_local = NULL;

        MPI_Comm_split(grid3d->comm, active ? 0 : MPI_UNDEFINED,
                       global_rank, &tree_comms[tree].comm);
        if (active) {
            int *local_to_global;

            tree_comms[tree].active = 1;
            MPI_Comm_rank(tree_comms[tree].comm, &tree_comms[tree].rank);
            MPI_Comm_size(tree_comms[tree].comm, &tree_comms[tree].nprocs);
            if (!(tree_comms[tree].global_to_local =
                      (int *) SUPERLU_MALLOC(pdgstrs3d_checked_product(
                          (size_t) global_nprocs, sizeof(int),
                          "SymLDL tree rank map"))) ||
                !(local_to_global = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product((size_t) tree_comms[tree].nprocs,
                                                sizeof(int),
                                                "SymLDL tree rank list"))))
                ABORT("Malloc fails for SymLDL tree rank maps.");
            for (int p = 0; p < global_nprocs; ++p)
                tree_comms[tree].global_to_local[p] = -1;
            MPI_Allgather(&global_rank, 1, MPI_INT, local_to_global, 1,
                          MPI_INT, tree_comms[tree].comm);
            for (int p = 0; p < tree_comms[tree].nprocs; ++p)
                tree_comms[tree].global_to_local[local_to_global[p]] = p;
            SUPERLU_FREE(local_to_global);
        }
    }

    SUPERLU_FREE(global_active);
    SUPERLU_FREE(local_active);
    return tree_comms;
}

static void
pdgstrs3d_symldl_tree_comms_free_count(pdgstrs3d_symldl_tree_comm_t *tree_comms,
                                       int numForests)
{
    if (tree_comms == NULL)
        return;

    for (int tree = 0; tree < numForests; ++tree) {
        if (tree_comms[tree].global_to_local)
            SUPERLU_FREE(tree_comms[tree].global_to_local);
        if (tree_comms[tree].comm != MPI_COMM_NULL)
            MPI_Comm_free(&tree_comms[tree].comm);
    }
    SUPERLU_FREE(tree_comms);
}

static void
pdgstrs3d_symldl_tree_comms_free(pdgstrs3d_symldl_tree_comm_t *tree_comms,
                                 dtrf3Dpartition_t *trf3Dpartition,
                                 gridinfo3d_t *grid3d)
{
    pdgstrs3d_symldl_tree_comms_free_count(
        tree_comms, pdgstrs3d_symldl_num_forests(trf3Dpartition, grid3d));
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
pdgstrs3d_symldl_comm_meta_set_count_views(pdgstrs3d_symldl_comm_meta_t *meta,
                                           int nprocs)
{
    meta->send_counts = meta->counts;
    meta->send_displs = meta->send_counts + nprocs;
    meta->recv_counts = meta->send_displs + nprocs;
    meta->recv_displs = meta->recv_counts + nprocs;
    meta->send_val_counts = meta->recv_displs + nprocs;
    meta->send_val_displs = meta->send_val_counts + nprocs;
    meta->recv_val_counts = meta->send_val_displs + nprocs;
    meta->recv_val_displs = meta->recv_val_counts + nprocs;
}

static pdgstrs3d_symldl_comm_meta_t *
pdgstrs3d_symldl_comm_meta_create(int_t nsupers,
                                  pdgstrs3d_symldl_panel_meta_t *panel_meta,
                                  dtrf3Dpartition_t *trf3Dpartition,
                                  pdgstrs3d_symldl_tree_comm_t *tree_comms,
                                  gridinfo3d_t *grid3d, int global_nprocs,
                                  int nrhs, int *diag_owner)
{
    pdgstrs3d_symldl_comm_meta_t *meta;
    int_t *send_rows_buf = NULL;
    int_t send_rows_cap = 0;
    int *fill_counts = NULL;

    if (!(meta = (pdgstrs3d_symldl_comm_meta_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) nsupers,
                              sizeof(pdgstrs3d_symldl_comm_meta_t),
                              "SymLDL communication schedule"))))
        ABORT("Malloc fails for SymLDL communication schedule.");
    if (!(fill_counts = (int *) SUPERLU_MALLOC(pdgstrs3d_checked_product(
              (size_t) global_nprocs, sizeof(int),
              "SymLDL communication schedule workspace"))))
        ABORT("Malloc fails for SymLDL communication schedule workspace.");

    for (int_t k = 0; k < nsupers; ++k) {
        int_t tree = (trf3Dpartition && trf3Dpartition->supernode2treeMap)
                         ? trf3Dpartition->supernode2treeMap[k] : -1;
        pdgstrs3d_symldl_tree_comm_t *tree_comm =
            (tree_comms && tree >= 0) ? &tree_comms[tree] : NULL;
        MPI_Comm solve_comm;
        int comm_nprocs;
        pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
        pdgstrs3d_symldl_comm_meta_t *cmeta = &meta[k];
        int total_send = 0;
        int total_recv = 0;
        int root_rank;
        int local_need_xk;
        int local_has_diag;
        int local_panel_active;

        cmeta->active = 0;
        cmeta->nprocs = 0;
        cmeta->total_send = 0;
        cmeta->total_recv = 0;
        cmeta->total_send_vals = 0;
        cmeta->total_recv_vals = 0;
        cmeta->needs_xk = 0;
        cmeta->xk_receiver_count = 0;
        cmeta->has_diag = 0;
        cmeta->diag_rank_count = 0;
        cmeta->counts = NULL;
        cmeta->send_counts = NULL;
        cmeta->send_displs = NULL;
        cmeta->recv_counts = NULL;
        cmeta->recv_displs = NULL;
        cmeta->send_val_counts = NULL;
        cmeta->send_val_displs = NULL;
        cmeta->recv_val_counts = NULL;
        cmeta->recv_val_displs = NULL;
        cmeta->row_to_send_pos = NULL;
        cmeta->send_seq = NULL;
        cmeta->xk_receivers = NULL;
        cmeta->diag_ranks = NULL;
        cmeta->recv_rows = NULL;

        if (tree_comm == NULL)
            ABORT("SymLDL solve communication metadata is missing a tree communicator.");
        if (!tree_comm->active)
            continue;

        solve_comm = tree_comm->comm;
        comm_nprocs = tree_comm->nprocs;
        cmeta->active = 1;
        cmeta->nprocs = comm_nprocs;
        if (!(cmeta->counts = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) comm_nprocs,
                                            8 * sizeof(int),
                                            "SymLDL communication schedule counts"))))
            ABORT("Malloc fails for SymLDL communication schedule counts.");
        pdgstrs3d_symldl_comm_meta_set_count_views(cmeta, comm_nprocs);
        for (int p = 0; p < 8 * comm_nprocs; ++p)
            cmeta->counts[p] = 0;

        root_rank = pdgstrs3d_symldl_rank_to_tree_rank(tree_comm,
                                                       diag_owner[k]);
        local_panel_active =
            pdgstrs3d_symldl_local_panel_active(trf3Dpartition, kmeta, k);
        local_need_xk = (local_panel_active && kmeta->row_count > 0);
        cmeta->needs_xk = local_need_xk;
        MPI_Allgather(&local_need_xk, 1, MPI_INT, fill_counts, 1, MPI_INT,
                      solve_comm);
        for (int p = 0; p < comm_nprocs; ++p)
            if (fill_counts[p] && p != root_rank)
                ++cmeta->xk_receiver_count;
        if (cmeta->xk_receiver_count > 0) {
            int receiver = 0;
            if (!(cmeta->xk_receivers = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product(
                          (size_t) cmeta->xk_receiver_count, sizeof(int),
                          "SymLDL Xk receiver list"))))
                ABORT("Malloc fails for SymLDL Xk receiver list.");
            for (int p = 0; p < comm_nprocs; ++p)
                if (fill_counts[p] && p != root_rank)
                    cmeta->xk_receivers[receiver++] = p;
        }

        local_has_diag = kmeta->has_diag;
        cmeta->has_diag = local_has_diag;
        MPI_Allgather(&local_has_diag, 1, MPI_INT, fill_counts, 1, MPI_INT,
                      solve_comm);
        for (int p = 0; p < comm_nprocs; ++p)
            if (fill_counts[p])
                ++cmeta->diag_rank_count;
        if (cmeta->diag_rank_count > 0) {
            int owner = 0;
            if (!(cmeta->diag_ranks = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product(
                          (size_t) cmeta->diag_rank_count, sizeof(int),
                          "SymLDL diagonal rank list"))))
                ABORT("Malloc fails for SymLDL diagonal rank list.");
            for (int p = 0; p < comm_nprocs; ++p)
                if (fill_counts[p])
                    cmeta->diag_ranks[owner++] = p;
        }

        if (local_panel_active && kmeta->row_count > 0) {
            int row_count = pdgstrs3d_symldl_count_to_int(
                kmeta->row_count, "SymLDL communication schedule rows");
            if (!(cmeta->row_to_send_pos = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product((size_t) row_count,
                                                sizeof(int),
                                                "SymLDL row send map"))))
                ABORT("Malloc fails for SymLDL row send map.");
            for (int_t row = 0; row < kmeta->row_count; ++row) {
                int dest = pdgstrs3d_symldl_rank_to_tree_rank(
                    tree_comm, kmeta->row_dest_global[row]);
                ++cmeta->send_counts[dest];
            }
        }

        pdgstrs3d_symldl_counts_to_displs(comm_nprocs,
                                          cmeta->send_counts,
                                          cmeta->send_displs,
                                          &total_send);
        MPI_Alltoall(cmeta->send_counts, 1, MPI_INT,
                     cmeta->recv_counts, 1, MPI_INT, solve_comm);
        pdgstrs3d_symldl_counts_to_displs(comm_nprocs,
                                          cmeta->recv_counts,
                                          cmeta->recv_displs,
                                          &total_recv);
        cmeta->total_send = total_send;
        cmeta->total_recv = total_recv;

        for (int p = 0; p < comm_nprocs; ++p) {
            cmeta->send_val_counts[p] = cmeta->send_counts[p] * nrhs;
            cmeta->recv_val_counts[p] = cmeta->recv_counts[p] * nrhs;
        }
        pdgstrs3d_symldl_counts_to_displs(comm_nprocs,
                                          cmeta->send_val_counts,
                                          cmeta->send_val_displs,
                                          &cmeta->total_send_vals);
        pdgstrs3d_symldl_counts_to_displs(comm_nprocs,
                                          cmeta->recv_val_counts,
                                          cmeta->recv_val_displs,
                                          &cmeta->total_recv_vals);

        if (total_send > 0) {
            if (!(cmeta->send_seq = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product((size_t) total_send,
                                                sizeof(int),
                                                "SymLDL row send sequence"))))
                ABORT("Malloc fails for SymLDL row send sequence.");
            pdgstrs3d_symldl_grow_int_t_buffer(
                &send_rows_buf, &send_rows_cap, total_send,
                "Malloc fails for SymLDL communication schedule rows.");
            for (int p = 0; p < comm_nprocs; ++p)
                fill_counts[p] = 0;
            for (int_t row = 0; row < kmeta->row_count; ++row) {
                int row_i = pdgstrs3d_symldl_count_to_int(
                    row, "SymLDL row send position");
                int dest = pdgstrs3d_symldl_rank_to_tree_rank(
                    tree_comm, kmeta->row_dest_global[row]);
                int pos = cmeta->send_displs[dest] + fill_counts[dest]++;
                send_rows_buf[pos] = kmeta->rows[row];
                cmeta->row_to_send_pos[row_i] = pos;
                cmeta->send_seq[pos] = row_i;
            }
        }
        if (total_recv > 0) {
            if (!(cmeta->recv_rows = intMalloc_dist(total_recv)))
                ABORT("Malloc fails for SymLDL received row schedule.");
        }

        MPI_Alltoallv(send_rows_buf, cmeta->send_counts, cmeta->send_displs,
                      mpi_int_t, cmeta->recv_rows, cmeta->recv_counts,
                      cmeta->recv_displs, mpi_int_t, solve_comm);
    }

    if (send_rows_buf) SUPERLU_FREE(send_rows_buf);
    SUPERLU_FREE(fill_counts);
    return meta;
}

static void
pdgstrs3d_symldl_comm_meta_free(pdgstrs3d_symldl_comm_meta_t *meta,
                                int_t nsupers)
{
    if (meta == NULL)
        return;

    for (int_t k = 0; k < nsupers; ++k) {
        if (meta[k].recv_rows) SUPERLU_FREE(meta[k].recv_rows);
        if (meta[k].diag_ranks) SUPERLU_FREE(meta[k].diag_ranks);
        if (meta[k].xk_receivers) SUPERLU_FREE(meta[k].xk_receivers);
        if (meta[k].send_seq) SUPERLU_FREE(meta[k].send_seq);
        if (meta[k].row_to_send_pos) SUPERLU_FREE(meta[k].row_to_send_pos);
        if (meta[k].counts) SUPERLU_FREE(meta[k].counts);
    }
    SUPERLU_FREE(meta);
}

static pdgstrs3d_symldl_x_cache_t *
pdgstrs3d_symldl_x_cache_create(int_t nsupers,
                                pdgstrs3d_symldl_panel_meta_t *panel_meta,
                                int nrhs)
{
    pdgstrs3d_symldl_x_cache_t *cache;

    if (!(cache = (pdgstrs3d_symldl_x_cache_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) nsupers,
                              sizeof(pdgstrs3d_symldl_x_cache_t),
                              "SymLDL replicated X cache"))))
        ABORT("Malloc fails for SymLDL replicated X cache.");
    memset(cache, 0, pdgstrs3d_checked_alloc_bytes(
           nsupers, sizeof(pdgstrs3d_symldl_x_cache_t),
           "SymLDL replicated X cache"));

    for (int_t k = 0; k < nsupers; ++k) {
        int_t row_count = panel_meta[k].has_panel ? panel_meta[k].row_count : 0;
        if (row_count <= 0)
            continue;
        int_t value_count = pdgstrs3d_checked_workspace_count(
            row_count, nrhs, 0, 0, "SymLDL replicated X cache values");
        cache[k].row_count = row_count;
        if (!(cache[k].values = doubleMalloc_dist(value_count)) ||
            !(cache[k].valid = (unsigned char *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_alloc_bytes(
                      row_count, sizeof(unsigned char),
                      "SymLDL replicated X cache valid flags"))))
            ABORT("Malloc fails for SymLDL replicated X cache entries.");
    }

    return cache;
}

static void
pdgstrs3d_symldl_x_cache_free(pdgstrs3d_symldl_x_cache_t *cache,
                              int_t nsupers)
{
    if (cache == NULL)
        return;
    for (int_t k = 0; k < nsupers; ++k) {
        if (cache[k].valid) SUPERLU_FREE(cache[k].valid);
        if (cache[k].values) SUPERLU_FREE(cache[k].values);
    }
    SUPERLU_FREE(cache);
}

static int
pdgstrs3d_symldl_node_in_tree(pdgstrs3d_symldl_solve_meta_t *meta,
                              int_t k, int tree)
{
    dtrf3Dpartition_t *trf3Dpartition =
        meta != NULL ? meta->trf3Dpartition : NULL;
    if (trf3Dpartition == NULL || trf3Dpartition->supernode2treeMap == NULL)
        ABORT("SymLDL replicated X cache requires supernode tree metadata.");
    return trf3Dpartition->supernode2treeMap[k] == tree;
}

static double
pdgstrs3d_symldl_x_cache_fill(pdgstrs3d_symldl_solve_meta_t *meta,
                              double *x, int nrhs, int_t *ilsum,
                              gridinfo3d_t *grid3d,
                              pdgstrs3d_symldl_node_ctx_t *ctxs,
                              int_t nctx)
{
    dtrf3Dpartition_t *trf3Dpartition = meta->trf3Dpartition;
    Glu_persist_t *Glu_persist = meta->Glu_persist;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    double start = SuperLU_timer_();
    double replicated_bytes = 0.0;
    double avoided_bytes = 0.0;
    double hits = 0.0;
    double misses = 0.0;
    double panels = 0.0;
    double dummy = 0.0;

    for (int_t ci = 0; ci < nctx; ++ci) {
        pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
        pdgstrs3d_symldl_panel_meta_t *kmeta;
        pdgstrs3d_symldl_x_cache_t *cache;
        if (!ctx->active)
            continue;
        kmeta = ctx->kmeta;
        cache = &meta->x_cache[ctx->k];
        if (!kmeta->has_panel || kmeta->row_count <= 0 ||
            cache->valid == NULL)
            continue;
        memset(cache->valid, 0,
               pdgstrs3d_checked_alloc_bytes(
                   cache->row_count, sizeof(unsigned char),
                   "SymLDL replicated X cache valid flags"));
    }

    for (int tree = 0; tree < meta->numForests; ++tree) {
        pdgstrs3d_symldl_tree_comm_t *tree_comm = &meta->tree_comms[tree];
        int nprocs;
        int *send_counts = NULL;
        int *recv_counts = NULL;
        int *send_displs = NULL;
        int *recv_displs = NULL;
        int *send_cursor = NULL;
        int *recv_cursor = NULL;
        double *send_buf = NULL;
        double *recv_buf = NULL;
        int total_send = 0;
        int total_recv = 0;

        if (!tree_comm->active)
            continue;
        nprocs = tree_comm->nprocs;
        if (!(send_counts = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache send counts"))) ||
            !(recv_counts = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache recv counts"))) ||
            !(send_displs = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache send displs"))) ||
            !(recv_displs = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache recv displs"))) ||
            !(send_cursor = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache send cursor"))) ||
            !(recv_cursor = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache recv cursor"))))
            ABORT("Malloc fails for SymLDL X cache communication counts.");
        for (int p = 0; p < nprocs; ++p) {
            send_counts[p] = 0;
            recv_counts[p] = 0;
        }

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            int_t k = ctx->k;
            if (!ctx->active)
                continue;
            if (!cmeta->active ||
                !pdgstrs3d_symldl_node_in_tree(meta, k, tree))
                continue;
            for (int p = 0; p < nprocs; ++p) {
                send_counts[p] += cmeta->recv_val_counts[p];
                recv_counts[p] += cmeta->send_val_counts[p];
            }
        }

        pdgstrs3d_symldl_counts_to_displs(nprocs, send_counts, send_displs,
                                          &total_send);
        pdgstrs3d_symldl_counts_to_displs(nprocs, recv_counts, recv_displs,
                                          &total_recv);
        if (total_send > 0 && !(send_buf = doubleMalloc_dist(total_send)))
            ABORT("Malloc fails for SymLDL X cache send buffer.");
        if (total_recv > 0 && !(recv_buf = doubleMalloc_dist(total_recv)))
            ABORT("Malloc fails for SymLDL X cache recv buffer.");
        for (int p = 0; p < nprocs; ++p) {
            send_cursor[p] = 0;
            recv_cursor[p] = 0;
        }

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            int_t k = ctx->k;
            if (!ctx->active)
                continue;
            if (!cmeta->active ||
                !pdgstrs3d_symldl_node_in_tree(meta, k, tree))
                continue;
            for (int p = 0; p < nprocs; ++p) {
                int row_base = cmeta->recv_displs[p];
                int val_base = send_displs[p] + send_cursor[p];
                for (int j = 0; j < cmeta->recv_counts[p]; ++j) {
                    int_t grow = cmeta->recv_rows[row_base + j];
                    int_t gsup = BlockNum(grow);
                    int_t rel = grow - FstBlockC(gsup);
                    int_t gsupsz = SuperSize(gsup);
                    int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, gsup);
                    double *xg = &x[X_BLK(lk)];
                    for (int rhs = 0; rhs < nrhs; ++rhs)
                        send_buf[val_base + j * nrhs + rhs] =
                            xg[rel + (int_t) rhs * gsupsz];
                }
                send_cursor[p] += cmeta->recv_val_counts[p];
            }
        }

        MPI_Alltoallv(total_send ? send_buf : &dummy, send_counts,
                      send_displs, MPI_DOUBLE,
                      total_recv ? recv_buf : &dummy, recv_counts,
                      recv_displs, MPI_DOUBLE, tree_comm->comm);

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            int_t k = ctx->k;
            pdgstrs3d_symldl_x_cache_t *cache = &meta->x_cache[k];
            if (!ctx->active)
                continue;
            if (!cmeta->active ||
                !pdgstrs3d_symldl_node_in_tree(meta, k, tree) ||
                !kmeta->has_panel || kmeta->row_count <= 0)
                continue;
            if (cache->values == NULL || cache->valid == NULL ||
                cache->row_count != kmeta->row_count)
                ABORT("SymLDL X cache metadata is inconsistent.");
            for (int p = 0; p < nprocs; ++p) {
                int row_base = cmeta->send_displs[p];
                int val_base = recv_displs[p] + recv_cursor[p];
                for (int j = 0; j < cmeta->send_counts[p]; ++j) {
                    int seq = cmeta->send_seq[row_base + j];
                    for (int rhs = 0; rhs < nrhs; ++rhs)
                        cache->values[(int_t) seq * nrhs + rhs] =
                            recv_buf[val_base + j * nrhs + rhs];
                    cache->valid[seq] = 1;
                }
                recv_cursor[p] += cmeta->send_val_counts[p];
            }
        }

        replicated_bytes += (double) total_send * (double) sizeof(double);
        avoided_bytes += (double) total_recv * (double) sizeof(double);

        if (send_buf) SUPERLU_FREE(send_buf);
        if (recv_buf) SUPERLU_FREE(recv_buf);
        SUPERLU_FREE(recv_cursor);
        SUPERLU_FREE(send_cursor);
        SUPERLU_FREE(recv_displs);
        SUPERLU_FREE(send_displs);
        SUPERLU_FREE(recv_counts);
        SUPERLU_FREE(send_counts);
    }

    for (int_t ci = 0; ci < nctx; ++ci) {
        pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
        pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
        pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
        int_t k = ctx->k;
        pdgstrs3d_symldl_x_cache_t *cache = &meta->x_cache[k];
        if (!ctx->active)
            continue;
        if (!kmeta->has_panel || kmeta->row_count <= 0 ||
            !cmeta->active || cmeta->total_send <= 0)
            continue;
        ++panels;
        for (int_t row = 0; row < kmeta->row_count; ++row) {
            if (cache->valid[row]) {
                hits += (double) nrhs;
            } else {
                misses += (double) nrhs;
                fprintf(stderr,
                        "SymLDL X cache missing row on rank %d: "
                        "panel=%lld local_row=%lld global_row=%lld\n",
                        grid3d->iam, (long long) k, (long long) row,
                        (long long) kmeta->rows[row]);
                fflush(stderr);
            }
        }
    }
    if (misses != 0.0)
        ABORT("SymLDL X cache coverage is incomplete.");

    double elapsed = SuperLU_timer_() - start;
    meta->x_cache_fill_time += elapsed;
    meta->x_cache_replicated_bytes += replicated_bytes;
    meta->x_cache_avoided_request_bytes += avoided_bytes;
    meta->x_cache_hits += hits;
    meta->x_cache_misses += misses;
    meta->x_cache_panels += panels;
    (void) xsup;
    (void) supno;
    return elapsed;
}

static int_t
pdgstrs3d_symldl_panel_meta_max_block_rows(pdgstrs3d_symldl_panel_meta_t *meta,
                                           int_t nsupers)
{
    int_t max_rows = 1;

    if (meta == NULL)
        return max_rows;
    for (int_t k = 0; k < nsupers; ++k)
        for (int_t block = 0; block < meta[k].nblocks; ++block)
            max_rows = SUPERLU_MAX(max_rows, meta[k].block_nbrow[block]);
    return max_rows;
}

static void
pdgstrs3d_symldl_workspace_free(pdgstrs3d_symldl_workspace_t *work)
{
    if (work == NULL)
        return;
    if (work->recv_request_values_buf) SUPERLU_FREE(work->recv_request_values_buf);
    if (work->request_values_buf) SUPERLU_FREE(work->request_values_buf);
    if (work->row_values_buf) SUPERLU_FREE(work->row_values_buf);
    if (work->recv_vals_buf) SUPERLU_FREE(work->recv_vals_buf);
    if (work->send_vals_buf) SUPERLU_FREE(work->send_vals_buf);
    if (work->delta_recv_buf) SUPERLU_FREE(work->delta_recv_buf);
    if (work->comm_reqs) SUPERLU_FREE(work->comm_reqs);
    if (work->level_ctxs) SUPERLU_FREE(work->level_ctxs);
    if (work->rhs_buf) SUPERLU_FREE(work->rhs_buf);
    if (work->gemm_buf) SUPERLU_FREE(work->gemm_buf);
    if (work->delta_buf) SUPERLU_FREE(work->delta_buf);
    if (work->delta_send_buf) SUPERLU_FREE(work->delta_send_buf);
    if (work->diag_buf) SUPERLU_FREE(work->diag_buf);
    if (work->diag_send_buf) SUPERLU_FREE(work->diag_send_buf);
    if (work->xk_buf) SUPERLU_FREE(work->xk_buf);
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
pdgstrs3d_symldl_workspace_prepare(pdgstrs3d_symldl_solve_meta_t *meta,
                                   int_t x_count, int_t maxsup, int nrhs,
                                   int global_nprocs)
{
    pdgstrs3d_symldl_workspace_t *work = &meta->work;
    int_t panel_rows = SUPERLU_MAX(meta->max_panel_block_rows, (int_t) 1);
    int_t pivot_count = pdgstrs3d_checked_workspace_count(
        SUPERLU_MAX(maxsup, (int_t) 1), nrhs, 0, 0,
        "3D SymLDL pivot workspace");
    int_t panel_count = pdgstrs3d_checked_workspace_count(
        panel_rows, nrhs, 0, 0, "3D SymLDL panel workspace");
    int_t comm_req_count = pdgstrs3d_checked_workspace_count(
        2, global_nprocs, 0, 0, "SymLDL communication requests");

    if (panel_count <= 0)
        panel_count = 1;
    if (comm_req_count <= 0)
        comm_req_count = 1;

    pdgstrs3d_symldl_workspace_prepare_x(meta, x_count);
    pdgstrs3d_symldl_grow_double_buffer(&work->xk_buf, &work->xk_cap,
                                        pivot_count,
                                        "Malloc fails for SymLDL pivot buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->diag_send_buf,
                                        &work->diag_send_cap, pivot_count,
                                        "Malloc fails for SymLDL diagonal send buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->diag_buf, &work->diag_cap,
                                        pivot_count,
                                        "Malloc fails for SymLDL diagonal buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->delta_send_buf,
                                        &work->delta_send_cap, pivot_count,
                                        "Malloc fails for SymLDL delta send buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->delta_buf, &work->delta_cap,
                                        pivot_count,
                                        "Malloc fails for SymLDL delta buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->gemm_buf, &work->gemm_cap,
                                        panel_count,
                                        "Malloc fails for SymLDL GEMM buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->rhs_buf, &work->rhs_cap,
                                        panel_count,
                                        "Malloc fails for SymLDL RHS buffer.");
    pdgstrs3d_symldl_grow_request_buffer(&work->comm_reqs,
                                         &work->comm_reqs_cap,
                                         comm_req_count,
                                         "Malloc fails for SymLDL communication requests.");
}

static void
pdgstrs3d_symldl_sync_factor_gpu(pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL || meta->factor_gpu_handle == NULL ||
        meta->factor_gpu_synchronized)
        return;
#if defined(GPU_ACC)
    dSymLDLFactorGPUSynchronize((dLUgpu_Handle) meta->factor_gpu_handle);
#else
    ABORT("SymLDL V2 solve received GPU factor state in a non-CUDA build.");
#endif
    meta->factor_gpu_synchronized = 1;
}

static void
pdgstrs3d_symldl_prepare_host_factor_panels(
    pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL || meta->panel_meta == NULL ||
        meta->factor_gpu_handle == NULL)
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
    meta->host_panel_copy_time += SuperLU_timer_() - t;
}

static int
pdgstrs3d_symldl_use_literature_forward(
    pdgstrs3d_symldl_solve_meta_t *meta)
{
#if defined(GPU_ACC)
    return meta != NULL && meta->literature_forward_mode &&
           meta->superlu_acc_offload && meta->factor_gpu_handle != NULL &&
           dSymLDLLiteratureForwardAvailable();
#else
    (void) meta;
    return 0;
#endif
}

static int
pdgstrs3d_symldl_use_nvshmem(pdgstrs3d_symldl_solve_meta_t *meta)
{
#if defined(GPU_ACC)
    return meta != NULL && !meta->literature_forward_mode &&
           meta->superlu_acc_offload &&
           meta->factor_gpu_handle != NULL &&
           dSymLDLNVSHMEMSolveAvailable();
#else
    (void) meta;
    return 0;
#endif
}

static void
pdgstrs3d_symldl_literature_forward_prepare(
    pdgstrs3d_symldl_solve_meta_t *meta, int_t n, int_t nlb,
    int_t ldalsum, int nrhs, gridinfo3d_t *grid3d)
{
    if (meta == NULL || !meta->literature_forward_mode)
        return;
    if (!pdgstrs3d_symldl_use_literature_forward(meta))
        ABORT("SymLDL literature forward solve requires GPU offload and NVSHMEM.");
    if (meta->literature_forward_state != NULL)
        return;
#if defined(GPU_ACC)
    pdgstrs3d_symldl_sync_factor_gpu(meta);
    int_t panel_count = meta->trf3Dpartition->symV2LocalPanelCount;
    dSymLDLLiteraturePanelDesc *panels = panel_count > 0
        ? (dSymLDLLiteraturePanelDesc *) SUPERLU_MALLOC(
              pdgstrs3d_checked_alloc_bytes(
                  panel_count, sizeof(*panels),
                  "SymLDL literature forward panel descriptors"))
        : NULL;
    if (panel_count > 0 && panels == NULL)
        ABORT("Malloc fails for SymLDL literature forward panels.");

    for (int_t lp = 0; lp < panel_count; ++lp) {
        int_t k = meta->trf3Dpartition->symV2LocalPanelGids[lp];
        int_t *lsub = meta->Llu->Lrowind_bc_ptr[lp];
        int_t *lloc = meta->Llu->Lindval_loc_bc_ptr[lp];
        panels[lp].gid = k;
        panels[lp].lsub = lsub;
        panels[lp].lloc = lloc;
        panels[lp].lsub_count = 0;
        panels[lp].lloc_count = 0;
        panels[lp].value_count = 0;
        panels[lp].host_values = NULL;
        panels[lp].values = NULL;
        if (lsub == NULL)
            continue;
        panels[lp].lsub_count = BC_HEADER +
            lsub[0] * LB_DESCRIPTOR + lsub[1];
        panels[lp].lloc_count = 3 * lsub[0];
        panels[lp].value_count = lsub[1] *
            (meta->Glu_persist->xsup[k + 1] - meta->Glu_persist->xsup[k]);
        panels[lp].host_values = meta->Llu->Lnzval_bc_ptr[lp];
        int_t device_count = 0;
        if (dSymLDLFactorGPUGetPanel(
                (dLUgpu_Handle) meta->factor_gpu_handle, k,
                &panels[lp].values, &device_count) != 0 ||
            panels[lp].values == NULL ||
            device_count < panels[lp].value_count)
            ABORT("SymLDL literature forward could not access a retained L panel.");
    }

    int_t x_count = pdgstrs3d_checked_workspace_count(
        ldalsum, nrhs, nlb, XK_H,
        "SymLDL literature forward X workspace");
    int_t lsum_count = pdgstrs3d_checked_workspace_count(
        ldalsum, nrhs, nlb, LSUM_H,
        "SymLDL literature forward sum workspace");
    meta->literature_forward_state = dSymLDLLiteratureForwardCreate(
        n, meta->nsupers, nrhs, x_count, lsum_count, panel_count,
        panels, meta->Glu_persist->xsup, meta->Llu->ilsum,
        meta->trf3Dpartition, grid3d);
    if (panels != NULL)
        SUPERLU_FREE(panels);
    if (meta->literature_forward_state == NULL)
        ABORT("SymLDL literature forward setup failed.");
#else
    (void) n;
    (void) nlb;
    (void) ldalsum;
    (void) nrhs;
    (void) grid3d;
    ABORT("SymLDL literature forward requires a CUDA build.");
#endif
}

static void
pdgstrs3d_symldl_nvshmem_prepare(
    pdgstrs3d_symldl_solve_meta_t *meta, int_t n, int_t nlb,
    int_t ldalsum, int nrhs, gridinfo3d_t *grid3d)
{
    if (!pdgstrs3d_symldl_use_nvshmem(meta))
        return;
    if (grid3d == NULL)
        ABORT("SymLDL NVSHMEM solve metadata is missing.");
    if (meta->nvshmem_state != NULL)
        return;
#if defined(GPU_ACC)
    pdgstrs3d_symldl_sync_factor_gpu(meta);
    int rank;
    MPI_Comm_rank(grid3d->comm, &rank);
    int_t *xsup = meta->Glu_persist->xsup;
    int_t *supno = meta->Glu_persist->supno;
    int_t *ilsum = meta->Llu->ilsum;
    int_t panel_count = 0;
    int_t block_count = 0;
    int_t row_count = 0;

    for (int_t k = 0; k < meta->nsupers; ++k) {
        pdgstrs3d_symldl_panel_meta_t *panel = &meta->panel_meta[k];
        int panel_active = pdgstrs3d_symldl_local_panel_active(
            meta->trf3Dpartition, panel, k);
        if (!panel->has_panel || (!panel_active && meta->diag_owner[k] != rank))
            continue;
        ++panel_count;
        if (panel_active) {
            block_count = pdgstrs3d_checked_size_to_int_t(
                (size_t) block_count + (size_t) panel->nblocks,
                "SymLDL NVSHMEM block metadata");
            row_count = pdgstrs3d_checked_size_to_int_t(
                (size_t) row_count + (size_t) panel->row_count,
                "SymLDL NVSHMEM row metadata");
        }
    }

    dSymLDLNVPanelDesc *panels = panel_count > 0
        ? (dSymLDLNVPanelDesc *) SUPERLU_MALLOC(
              pdgstrs3d_checked_alloc_bytes(panel_count, sizeof(*panels),
                                            "SymLDL NVSHMEM panels"))
        : NULL;
    dSymLDLNVBlockDesc *blocks = block_count > 0
        ? (dSymLDLNVBlockDesc *) SUPERLU_MALLOC(
              pdgstrs3d_checked_alloc_bytes(block_count, sizeof(*blocks),
                                            "SymLDL NVSHMEM blocks"))
        : NULL;
    int_t *rows = row_count > 0 ? intMalloc_dist(row_count) : NULL;
    int_t *x_offsets = intMalloc_dist(meta->nsupers);
    int_t *lsum_offsets = intMalloc_dist(meta->nsupers);
    if ((panel_count > 0 && panels == NULL) ||
        (block_count > 0 && blocks == NULL) ||
        (row_count > 0 && rows == NULL) ||
        x_offsets == NULL || lsum_offsets == NULL)
        ABORT("Malloc fails for SymLDL NVSHMEM solve metadata.");

    for (int_t k = 0; k < meta->nsupers; ++k) {
        int_t local = meta->trf3Dpartition->symV2RowLocalIndex[k];
        if (local >= 0) {
            x_offsets[k] = X_BLK(local);
            lsum_offsets[k] = LSUM_BLK(local);
        } else {
            x_offsets[k] = -1;
            lsum_offsets[k] = -1;
        }
    }

    int_t panel_pos = 0;
    int_t block_pos = 0;
    int_t row_pos = 0;
    for (int_t k = 0; k < meta->nsupers; ++k) {
        pdgstrs3d_symldl_panel_meta_t *source = &meta->panel_meta[k];
        int panel_active = pdgstrs3d_symldl_local_panel_active(
            meta->trf3Dpartition, source, k);
        if (!source->has_panel || (!panel_active && meta->diag_owner[k] != rank))
            continue;

        double *device_values = NULL;
        int_t device_count = 0;
        if (dSymLDLFactorGPUGetPanel(
                (dLUgpu_Handle) meta->factor_gpu_handle, k,
                &device_values, &device_count) != 0 ||
            device_values == NULL || device_count < source->lusup_count)
            ABORT("SymLDL NVSHMEM solve could not access a factor panel.");

        dSymLDLNVPanelDesc *target = &panels[panel_pos++];
        target->gid = k;
        target->width = SuperSize(k);
        target->nsupr = source->nsupr;
        target->diag_luptr = source->diag_luptr;
        target->block_begin = block_pos;
        target->block_count = panel_active ? source->nblocks : 0;
        target->value_count = source->lusup_count;
        target->owner = meta->diag_owner[k];
        target->values = device_values;

        for (int_t b = 0; panel_active && b < source->nblocks; ++b) {
            int_t source_row = source->block_row_start[b];
            int_t nbrow = source->block_nbrow[b];
            int_t target_gid = BlockNum(source->rows[source_row]);
            dSymLDLNVBlockDesc *block = &blocks[block_pos++];
            block->panel_id = panel_pos - 1;
            block->target_gid = target_gid;
            block->luptr = source->block_luptr[b];
            block->nbrow = nbrow;
            block->row_begin = row_pos;
            for (int_t r = 0; r < nbrow; ++r) {
                int_t grow = source->rows[source_row + r];
                if (BlockNum(grow) != target_gid)
                    ABORT("SymLDL NVSHMEM block spans multiple supernodes.");
                rows[row_pos++] = grow;
            }
        }
    }
    if (panel_pos != panel_count || block_pos != block_count ||
        row_pos != row_count)
        ABORT("SymLDL NVSHMEM solve metadata size is inconsistent.");

    int_t x_count = pdgstrs3d_checked_workspace_count(
        ldalsum, nrhs, nlb, XK_H, "SymLDL NVSHMEM X workspace");
    int_t lsum_count = pdgstrs3d_checked_workspace_count(
        ldalsum, nrhs, nlb, LSUM_H, "SymLDL NVSHMEM sum workspace");
    meta->nvshmem_state = dSymLDLNVSHMEMSolveCreate(
        n, meta->nsupers, nrhs, x_count, lsum_count,
        panel_count, panels, block_count, blocks, row_count, rows,
        xsup, meta->diag_owner, x_offsets, lsum_offsets,
        grid3d->zscp.Np, grid3d->comm);

    if (lsum_offsets) SUPERLU_FREE(lsum_offsets);
    if (x_offsets) SUPERLU_FREE(x_offsets);
    if (rows) SUPERLU_FREE(rows);
    if (blocks) SUPERLU_FREE(blocks);
    if (panels) SUPERLU_FREE(panels);
    if (meta->nvshmem_state == NULL)
        ABORT("SymLDL NVSHMEM solve setup failed.");
#else
    (void) n;
    (void) nlb;
    (void) ldalsum;
    (void) nrhs;
    ABORT("SymLDL NVSHMEM solve requires a CUDA build.");
#endif
}

static void
pdgstrs3d_symldl_nvshmem_take_timers(
    pdgstrs3d_symldl_solve_meta_t *meta,
    pdgstrs3d_symldl_timer_t *timer)
{
    double h2d = 0.0;
    double forward = 0.0;
    double d2h = 0.0;
    double forward_phase = 0.0;
    double diagonal_phase = 0.0;
    double backward_phase = 0.0;
    if (meta == NULL || meta->nvshmem_state == NULL || timer == NULL)
        return;
    dSymLDLNVSHMEMSolveTakeTimers(meta->nvshmem_state,
                                  &h2d, &forward, &d2h);
    dSymLDLNVSHMEMSolveTakePhaseTimers(meta->nvshmem_state,
                                      &forward_phase, &diagonal_phase,
                                      &backward_phase);
    timer->gpu_h2d += h2d;
    timer->gpu_compute += forward;
    timer->gpu_d2h += d2h;
    timer->forward_compute += forward_phase;
    timer->diag_compute += diagonal_phase;
    timer->backward_compute += backward_phase;
}

static void
pdgstrs3d_symldl_literature_take_timers(
    pdgstrs3d_symldl_solve_meta_t *meta,
    pdgstrs3d_symldl_timer_t *timer)
{
    double setup = 0.0;
    double forward = 0.0;
    double sparse_reduce = 0.0;
    double sparse_broadcast = 0.0;
    double h2d = 0.0;
    double d2h = 0.0;
    if (meta == NULL || meta->literature_forward_state == NULL ||
        timer == NULL)
        return;
    dSymLDLLiteratureForwardTakeTimers(
        meta->literature_forward_state, &setup, &forward,
        &sparse_reduce, &sparse_broadcast, &h2d, &d2h);
    timer->workspace += setup;
    timer->forward_compute += forward;
    timer->forward_values += sparse_reduce;
    timer->forward_apply += sparse_broadcast;
    timer->gpu_h2d += h2d;
    timer->gpu_compute += forward;
    timer->gpu_d2h += d2h;
}

static void
pdgstrs3d_symldl_solve_meta_destroy(pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL)
        return;
    if (meta->literature_forward_state)
        dSymLDLLiteratureForwardDestroy(meta->literature_forward_state);
    if (meta->nvshmem_state)
        dSymLDLNVSHMEMSolveDestroy(meta->nvshmem_state);
#if defined(GPU_ACC)
    if (meta->factor_gpu_handle)
        dDestroyLUgpuHandle((dLUgpu_Handle) meta->factor_gpu_handle);
#else
    if (meta->factor_gpu_handle)
        ABORT("SymLDL V2 solve cannot destroy GPU factor state in a non-CUDA build.");
#endif
    pdgstrs3d_symldl_workspace_free(&meta->work);
    pdgstrs3d_symldl_level_schedule_free(&meta->solve_schedule);
    pdgstrs3d_symldl_x_cache_free(meta->x_cache, meta->nsupers);
    pdgstrs3d_symldl_comm_meta_free(meta->comm_meta, meta->nsupers);
    pdgstrs3d_symldl_panel_meta_free(meta->panel_meta, meta->nsupers);
    pdgstrs3d_symldl_tree_comms_free_count(meta->tree_comms,
                                           meta->numForests);
    if (meta->diag_owner) SUPERLU_FREE(meta->diag_owner);
    if (meta->solve_order) SUPERLU_FREE(meta->solve_order);
    SUPERLU_FREE(meta);
}

void
pdgstrs3d_symldl_finalize(dSOLVEstruct_t *SOLVEstruct)
{
    if (SOLVEstruct == NULL || SOLVEstruct->symldl_v2_solve_meta == NULL)
    {
        if (SOLVEstruct != NULL && SOLVEstruct->symldl_v2_factor_handle != NULL) {
#if defined(GPU_ACC)
            dDestroyLUgpuHandle(
                (dLUgpu_Handle) SOLVEstruct->symldl_v2_factor_handle);
#else
            ABORT("SymLDL V2 solve cannot destroy GPU factor state in a non-CUDA build.");
#endif
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
                                  int superlu_n_gemm)
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
           meta->literature_forward_mode ==
               pdgstrs3d_symldl_env_enabled(
                   "GPU3DV2_SYM_SOLVE_LITERATURE_FORWARD") &&
           meta->superlu_acc_offload == superlu_acc_offload &&
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
                                int superlu_acc_offload, int superlu_n_gemm)
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
    meta->literature_forward_mode = pdgstrs3d_symldl_env_enabled(
        "GPU3DV2_SYM_SOLVE_LITERATURE_FORWARD");
    meta->superlu_acc_offload = superlu_acc_offload;
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
    if (!pdgstrs3d_symldl_use_nvshmem(meta)) {
        meta->solve_order = pdgstrs3d_symldl_tree_order(
            nsupers, trf3Dpartition, grid3d);
        meta->tree_comms = pdgstrs3d_symldl_tree_comms_create(
            trf3Dpartition, meta->panel_meta, meta->diag_owner,
            nsupers, grid3d, global_nprocs);
        meta->numForests = meta->tree_comms
                               ? pdgstrs3d_symldl_num_forests(
                                     trf3Dpartition, grid3d)
                               : 0;
        meta->max_panel_block_rows =
            pdgstrs3d_symldl_panel_meta_max_block_rows(
                meta->panel_meta, nsupers);
        meta->solve_schedule = pdgstrs3d_symldl_level_schedule_create(
            nsupers, meta->solve_order, meta->panel_meta, grid3d,
            Glu_persist);
        meta->comm_meta = pdgstrs3d_symldl_comm_meta_create(
            nsupers, meta->panel_meta, trf3Dpartition, meta->tree_comms,
            grid3d, global_nprocs, nrhs, meta->diag_owner);
        meta->x_cache = pdgstrs3d_symldl_x_cache_create(
            nsupers, meta->panel_meta, nrhs);
    }
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
        sp_ienv_dist(10, options), sp_ienv_dist(7, options));
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
pdgstrs3d_symldl_forward_xk(double *xk, int count, int rank, int root,
                            pdgstrs3d_symldl_comm_meta_t *meta,
                            MPI_Comm comm)
{
    if (rank == root) {
        for (int i = 0; i < meta->xk_receiver_count; ++i)
            MPI_Send(xk, count, MPI_DOUBLE, meta->xk_receivers[i], Xk, comm);
    } else if (meta->needs_xk) {
        MPI_Recv(xk, count, MPI_DOUBLE, root, Xk, comm, MPI_STATUS_IGNORE);
    }
}

static void
pdgstrs3d_symldl_exchange_double(double *send_buf, int *send_counts,
                                 int *send_displs, double *recv_buf,
                                 int *recv_counts, int *recv_displs,
                                 int nprocs, int rank, int tag,
                                 MPI_Comm comm, MPI_Request *reqs)
{
    int nreq = 0;

    for (int p = 0; p < nprocs; ++p) {
        if (p == rank) {
            if (send_counts[p] > 0)
                memcpy(&recv_buf[recv_displs[p]], &send_buf[send_displs[p]],
                       (size_t) send_counts[p] * sizeof(double));
        } else if (recv_counts[p] > 0) {
            MPI_Irecv(&recv_buf[recv_displs[p]], recv_counts[p], MPI_DOUBLE,
                      p, tag, comm, &reqs[nreq++]);
        }
    }
    for (int p = 0; p < nprocs; ++p) {
        if (p != rank && send_counts[p] > 0)
            MPI_Isend(&send_buf[send_displs[p]], send_counts[p], MPI_DOUBLE,
                      p, tag, comm, &reqs[nreq++]);
    }
    if (nreq > 0)
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

static void
pdgstrs3d_symldl_post_xk(double *xk, int count, int rank, int root,
                         pdgstrs3d_symldl_comm_meta_t *meta,
                         MPI_Comm comm, MPI_Request *reqs, int *nreq)
{
    if (rank == root) {
        for (int i = 0; i < meta->xk_receiver_count; ++i)
            MPI_Isend(xk, count, MPI_DOUBLE, meta->xk_receivers[i], Xk,
                      comm, &reqs[(*nreq)++]);
    } else if (meta->needs_xk) {
        MPI_Irecv(xk, count, MPI_DOUBLE, root, Xk, comm,
                  &reqs[(*nreq)++]);
    }
}

static void
pdgstrs3d_symldl_post_exchange_double(double *send_buf, int *send_counts,
                                      int *send_displs, double *recv_buf,
                                      int *recv_counts, int *recv_displs,
                                      int nprocs, int rank, int tag,
                                      MPI_Comm comm, MPI_Request *reqs,
                                      int *nreq)
{
    for (int p = 0; p < nprocs; ++p) {
        if (p == rank) {
            if (send_counts[p] > 0)
                memcpy(&recv_buf[recv_displs[p]], &send_buf[send_displs[p]],
                       (size_t) send_counts[p] * sizeof(double));
        } else if (recv_counts[p] > 0) {
            MPI_Irecv(&recv_buf[recv_displs[p]], recv_counts[p], MPI_DOUBLE,
                      p, tag, comm, &reqs[(*nreq)++]);
        }
    }
    for (int p = 0; p < nprocs; ++p) {
        if (p != rank && send_counts[p] > 0)
            MPI_Isend(&send_buf[send_displs[p]], send_counts[p], MPI_DOUBLE,
                      p, tag, comm, &reqs[(*nreq)++]);
    }
}

static void
pdgstrs3d_symldl_reduce_delta(double *delta_send, double *delta, int count,
                              int rank, int root,
                              pdgstrs3d_symldl_comm_meta_t *meta,
                              MPI_Comm comm)
{
    if (rank == root) {
        if (meta->needs_xk)
            for (int i = 0; i < count; ++i)
                delta[i] += delta_send[i];
        for (int sender = 0; sender < meta->xk_receiver_count; ++sender) {
            MPI_Recv(delta_send, count, MPI_DOUBLE,
                     meta->xk_receivers[sender], RD_U, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; ++i)
                delta[i] += delta_send[i];
        }
    } else if (meta->needs_xk) {
        MPI_Send(delta_send, count, MPI_DOUBLE, root, RD_U, comm);
    }
}

static void
pdgstrs3d_symldl_post_reduce_delta(pdgstrs3d_symldl_node_ctx_t *ctx,
                                   MPI_Request *reqs, int *nreq)
{
    pdgstrs3d_symldl_comm_meta_t *meta = ctx->cmeta;
    int count = ctx->delta_count;

    if (ctx->solve_rank == ctx->root_rank) {
        if (meta->needs_xk)
            for (int i = 0; i < count; ++i)
                ctx->delta_buf[i] += ctx->delta_send_buf[i];
        for (int sender = 0; sender < meta->xk_receiver_count; ++sender) {
            MPI_Irecv(&ctx->delta_recv_buf[(size_t) sender * (size_t) count],
                      count, MPI_DOUBLE, meta->xk_receivers[sender], RD_U,
                      ctx->solve_comm, &reqs[(*nreq)++]);
        }
    } else if (meta->needs_xk) {
        MPI_Isend(ctx->delta_send_buf, count, MPI_DOUBLE, ctx->root_rank,
                  RD_U, ctx->solve_comm, &reqs[(*nreq)++]);
    }
}

static void
pdgstrs3d_symldl_finish_reduce_delta(pdgstrs3d_symldl_node_ctx_t *ctx)
{
    pdgstrs3d_symldl_comm_meta_t *meta = ctx->cmeta;
    int count = ctx->delta_count;

    if (ctx->solve_rank != ctx->root_rank)
        return;
    for (int sender = 0; sender < meta->xk_receiver_count; ++sender) {
        double *recv = &ctx->delta_recv_buf[(size_t) sender * (size_t) count];
        for (int i = 0; i < count; ++i)
            ctx->delta_buf[i] += recv[i];
    }
}

static int_t
pdgstrs3d_symldl_count_sum(int_t total, int_t add, const char *what)
{
    return pdgstrs3d_checked_workspace_count(total, 1, add, 1, what);
}

static pdgstrs3d_symldl_node_ctx_t *
pdgstrs3d_symldl_workspace_prepare_level_contexts(
    pdgstrs3d_symldl_workspace_t *work, int_t nctx, const char *what)
{
    int_t need = SUPERLU_MAX(nctx, (int_t) 1);
    size_t bytes = pdgstrs3d_checked_product((size_t) need,
                                             sizeof(*work->level_ctxs),
                                             what);

    if (need > work->level_ctxs_cap) {
        if (work->level_ctxs)
            SUPERLU_FREE(work->level_ctxs);
        work->level_ctxs =
            (pdgstrs3d_symldl_node_ctx_t *) SUPERLU_MALLOC(bytes);
        if (work->level_ctxs == NULL)
            ABORT(what);
        work->level_ctxs_cap = need;
    }
    memset(work->level_ctxs, 0, bytes);
    return work->level_ctxs;
}

static void
pdgstrs3d_symldl_timer_print(pdgstrs3d_symldl_timer_t *timer,
                             pdgstrs3d_symldl_solve_meta_t *meta,
                             gridinfo3d_t *grid3d)
{
    enum { SYMLDL_TIMER_COUNT = 25 };
    double local[SYMLDL_TIMER_COUNT];
    double maxv[SYMLDL_TIMER_COUNT];
    double sumv[SYMLDL_TIMER_COUNT];
    struct { double val; int rank; } local_max[SYMLDL_TIMER_COUNT];
    struct { double val; int rank; } global_max[SYMLDL_TIMER_COUNT];
    int rank, nprocs;

    local[0] = timer->metadata;
    local[1] = timer->workspace;
    local[2] = timer->b_to_x;
    local[3] = timer->forward_xk;
    local[4] = timer->forward_compute;
    local[5] = timer->forward_values;
    local[6] = timer->forward_apply;
    local[7] = timer->diag_comm;
    local[8] = timer->diag_compute;
    local[9] = timer->x_cache_fill;
    local[10] = timer->backward_values;
    local[11] = timer->backward_compute;
    local[12] = timer->backward_delta;
    local[13] = timer->x_to_b;
    local[14] = timer->gpu_h2d;
    local[15] = timer->gpu_compute;
    local[16] = timer->gpu_d2h;
    local[17] = meta != NULL ? meta->cpu_blas_ops : 0.0;
    local[18] = meta != NULL ? meta->cpu_blas_calls : 0.0;
    local[19] = meta != NULL ? meta->host_panel_copy_time : 0.0;
    local[20] = meta != NULL ? meta->x_cache_replicated_bytes : 0.0;
    local[21] = meta != NULL ? meta->x_cache_avoided_request_bytes : 0.0;
    local[22] = meta != NULL ? meta->x_cache_hits : 0.0;
    local[23] = meta != NULL ? meta->x_cache_misses : 0.0;
    local[24] = meta != NULL ? meta->x_cache_panels : 0.0;

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
            "metadata", "workspace", "B_to_X", "forward_xk",
            "forward_compute", "forward_values", "forward_apply",
            "diag_comm", "diag_compute", "x_cache_fill", "backward_values",
            "backward_compute", "backward_delta", "X_to_B",
            "gpu_h2d", "gpu_compute", "gpu_d2h",
            "cpu_blas_ops", "cpu_blas_calls", "host_panel_copy",
            "x_cache_bytes", "x_cache_avoided", "x_cache_hits",
            "x_cache_misses", "x_cache_panels"
        };
        printf("SymFact GPU3D V2 solve timing (max_rank / avg_rank / max_rank_id):\n");
        for (int i = 0; i < SYMLDL_TIMER_COUNT; ++i)
            printf("  %-17s %.6f / %.6f / %d\n", names[i], maxv[i],
                   sumv[i] / (double) nprocs, global_max[i].rank);
    }
}

static void
pdgstrs3d_symldl_distributed(superlu_dist_options_t *options, int_t n,
           dLUstruct_t * LUstruct, dScalePermstruct_t * ScalePermstruct,
           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, double *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    MPI_Comm global_comm = grid3d->comm;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    int_t *ilsum = Llu->ilsum;
    int *supernodeMask = trf3Dpartition ? trf3Dpartition->supernodeMask : NULL;
    int_t nsupers = supno[n - 1] + 1;
    int_t nlb = (trf3Dpartition != NULL &&
                 trf3Dpartition->symV2RowLocalIndex != NULL)
                    ? SUPERLU_MAX((int_t) 1,
                                  trf3Dpartition->symV2LocalRowCount)
                    : CEILING(nsupers, grid->nprow);
    int_t ldalsum = Llu->ldalsum;
    int_t x_count;
    int_t maxsup = 0;
    int global_rank, global_nprocs;
    int superlu_acc_offload = sp_ienv_dist(10, options);
    int superlu_n_gemm = sp_ienv_dist(7, options);
    pdgstrs3d_symldl_solve_meta_t *solve_meta;
    int *diag_owner;
    int_t *solve_order;
    pdgstrs3d_symldl_level_schedule_t *solve_schedule;
    pdgstrs3d_symldl_tree_comm_t *tree_comms;
    pdgstrs3d_symldl_panel_meta_t *panel_meta;
    pdgstrs3d_symldl_comm_meta_t *comm_meta;
    pdgstrs3d_symldl_workspace_t *workspace;
    double *x;
    double *xk_buf;
    double *diag_send_buf;
    double *diag_buf;
    double *delta_send_buf;
    double *delta_buf;
    double *gemm_buf;
    double *rhs_buf;
    MPI_Request *comm_reqs;
    double tx, tx_st;
    double ttmp;
    pdgstrs3d_symldl_timer_t symldl_timer;

    MPI_Comm_rank(global_comm, &global_rank);
    MPI_Comm_size(global_comm, &global_nprocs);
    memset(&symldl_timer, 0, sizeof(symldl_timer));

    for (int_t k = 0; k < nsupers; ++k)
        maxsup = SUPERLU_MAX(maxsup, SuperSize(k));

    x_count = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb, XK_H,
                                                "3D SymLDL solve x workspace");

    ttmp = SuperLU_timer_();
    solve_meta = (pdgstrs3d_symldl_solve_meta_t *)
        SOLVEstruct->symldl_v2_solve_meta;
    if (!pdgstrs3d_symldl_solve_meta_valid(
            solve_meta, n, nsupers, nrhs, Glu_persist, Llu, trf3Dpartition,
            supernodeMask, grid3d, global_nprocs, superlu_acc_offload,
            superlu_n_gemm))
        ABORT("SymLDL solve requires prebuilt V2 solve metadata.");
    solve_meta->reused = 1;
    pdgstrs3d_symldl_reset_solve_stats(solve_meta);
    diag_owner = solve_meta->diag_owner;
    solve_order = solve_meta->solve_order;
    solve_schedule = &solve_meta->solve_schedule;
    tree_comms = solve_meta->tree_comms;
    panel_meta = solve_meta->panel_meta;
    comm_meta = solve_meta->comm_meta;
    symldl_timer.metadata = SuperLU_timer_() - ttmp;

    ttmp = SuperLU_timer_();
    if (pdgstrs3d_symldl_use_nvshmem(solve_meta)) {
        pdgstrs3d_symldl_workspace_prepare_x(solve_meta, x_count);
        pdgstrs3d_symldl_nvshmem_prepare(
            solve_meta, n, nlb, ldalsum, nrhs, grid3d);
    } else {
        pdgstrs3d_symldl_workspace_prepare(
            solve_meta, x_count, maxsup, nrhs, global_nprocs);
        pdgstrs3d_symldl_prepare_host_factor_panels(solve_meta);
        pdgstrs3d_symldl_literature_forward_prepare(
            solve_meta, n, nlb, ldalsum, nrhs, grid3d);
    }
    workspace = &solve_meta->work;
    x = workspace->x;
    xk_buf = workspace->xk_buf;
    diag_send_buf = workspace->diag_send_buf;
    diag_buf = workspace->diag_buf;
    delta_send_buf = workspace->delta_send_buf;
    delta_buf = workspace->delta_buf;
    gemm_buf = workspace->gemm_buf;
    rhs_buf = workspace->rhs_buf;
    comm_reqs = workspace->comm_reqs;
    symldl_timer.workspace = SuperLU_timer_() - ttmp;

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    xtrsTimer_t xtrsTimer;
    initTRStimer(&xtrsTimer, grid);

    tx = SuperLU_timer_();
    pdReDistribute3d_B_to_X_symv2(B, m_loc, nrhs, ldb, fst_row, ilsum, x,
                                  ScalePermstruct, Glu_persist,
                                  trf3Dpartition, grid3d, SOLVEstruct);
    xtrsTimer.t_pxReDistribute_B_to_X = SuperLU_timer_() - tx;
    symldl_timer.b_to_x = xtrsTimer.t_pxReDistribute_B_to_X;

    MPI_Barrier(grid3d->comm);
    tx_st = SuperLU_timer_();

    if (solve_meta->literature_forward_state != NULL) {
        if (dSymLDLLiteratureForwardInitializeRHS(
                solve_meta->literature_forward_state, x, x_count) != 0)
            ABORT("SymLDL literature forward RHS initialization failed.");
        if (dSymLDLLiteratureForwardSolve(
                solve_meta->literature_forward_state, x, x_count) != 0)
            ABORT("SymLDL literature forward solve failed.");
        if (dSymLDLLiteratureForwardSparseAllreduce(
                solve_meta->literature_forward_state, x, x_count) != 0)
            ABORT("SymLDL literature sparse Z allreduce failed.");
        xtrsTimer.t_forwardSolve = SuperLU_timer_() - tx_st;
        goto symldl_diagonal_solve;
    }

    if (solve_meta->nvshmem_state != NULL) {
        if (dSymLDLNVSHMEMForward(solve_meta->nvshmem_state,
                                  x, x_count) != 0)
            ABORT("SymLDL NVSHMEM forward solve failed.");
        xtrsTimer.t_forwardSolve = SuperLU_timer_() - tx_st;
        goto symldl_diagonal_solve;
    }

    /* Forward solve with unit-lower L. X lives only on z=0 diagonal owners. */
    for (int_t level = 0; level < solve_schedule->nlevels; ++level) {
        int_t level_begin = solve_schedule->level_ptr[level];
        int_t level_end = solve_schedule->level_ptr[level + 1];
        int_t nctx = level_end - level_begin;
        int_t max_level_reqs = 1;
        int_t level_xk_count = 0;
        int_t level_send_vals_count = 0;
        int_t level_recv_vals_count = 0;
        int_t xk_offset = 0;
        int_t send_vals_offset = 0;
        int_t recv_vals_offset = 0;
        pdgstrs3d_symldl_node_ctx_t *ctxs =
            pdgstrs3d_symldl_workspace_prepare_level_contexts(
                workspace, nctx, "Malloc fails for SymLDL forward level contexts.");

        for (int_t ci = 0; ci < nctx; ++ci) {
            int_t k = solve_schedule->nodes[level_begin + ci];
            int_t tree = (trf3Dpartition && trf3Dpartition->supernode2treeMap)
                             ? trf3Dpartition->supernode2treeMap[k] : -1;
            pdgstrs3d_symldl_tree_comm_t *tree_comm =
                (tree_comms && tree >= 0) ? &tree_comms[tree] : NULL;
            pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
            pdgstrs3d_symldl_comm_meta_t *cmeta = &comm_meta[k];
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            int_t ksupc = SuperSize(k);

            if (tree_comm == NULL)
                ABORT("SymLDL solve is missing tree metadata for a forward panel.");
            if (!tree_comm->active || !cmeta->active)
                continue;

            ctx->active = 1;
            ctx->k = k;
            ctx->ksupc = ksupc;
            ctx->xk_count = pdgstrs3d_symldl_count_to_int(
                ksupc * nrhs, "SymLDL pivot block");
            ctx->root_rank = pdgstrs3d_symldl_rank_to_tree_rank(
                tree_comm, diag_owner[k]);
            ctx->solve_rank = tree_comm->rank;
            ctx->solve_comm = tree_comm->comm;
            ctx->kmeta = kmeta;
            ctx->cmeta = cmeta;

            level_xk_count = pdgstrs3d_symldl_count_sum(
                level_xk_count, ctx->xk_count, "SymLDL forward pivot buffer");
            if (cmeta->total_send > 0)
                level_send_vals_count = pdgstrs3d_symldl_count_sum(
                    level_send_vals_count, cmeta->total_send_vals,
                    "SymLDL forward send values");
            if (cmeta->total_recv > 0)
                level_recv_vals_count = pdgstrs3d_symldl_count_sum(
                    level_recv_vals_count, cmeta->total_recv_vals,
                    "SymLDL forward recv values");
            max_level_reqs += cmeta->xk_receiver_count + 1 +
                              2 * cmeta->nprocs;
        }

        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->xk_buf, &workspace->xk_cap,
            SUPERLU_MAX(level_xk_count, (int_t) 1),
            "Malloc fails for SymLDL forward pivot buffer.");
        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->send_vals_buf, &workspace->send_vals_cap,
            SUPERLU_MAX(level_send_vals_count, (int_t) 1),
            "Malloc fails for SymLDL forward send values.");
        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->recv_vals_buf, &workspace->recv_vals_cap,
            SUPERLU_MAX(level_recv_vals_count, (int_t) 1),
            "Malloc fails for SymLDL forward recv values.");
        pdgstrs3d_symldl_grow_request_buffer(
            &workspace->comm_reqs, &workspace->comm_reqs_cap,
            max_level_reqs, "Malloc fails for SymLDL forward level requests.");
        comm_reqs = workspace->comm_reqs;

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            int_t ksupc = ctx->ksupc;

            if (!ctx->active)
                continue;

            ctx->xk_buf = &workspace->xk_buf[xk_offset];
            xk_offset += ctx->xk_count;
            if (cmeta->total_send > 0) {
                ctx->send_vals = &workspace->send_vals_buf[send_vals_offset];
                send_vals_offset += cmeta->total_send_vals;
            }
            if (cmeta->total_recv > 0) {
                ctx->recv_vals = &workspace->recv_vals_buf[recv_vals_offset];
                recv_vals_offset += cmeta->total_recv_vals;
            }

            if (global_rank == diag_owner[ctx->k]) {
                int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, ctx->k);
                double *xk = &x[X_BLK(lk)];
                for (int rhs = 0; rhs < nrhs; ++rhs)
                    for (int_t i = 0; i < ksupc; ++i)
                        ctx->xk_buf[i + (int_t) rhs * ksupc] =
                            xk[i + (int_t) rhs * ksupc];
            }
        }

        ttmp = SuperLU_timer_();
        int nreq_xk = 0;
        for (int_t ci = 0; ci < nctx; ++ci)
            if (ctxs[ci].active)
                pdgstrs3d_symldl_post_xk(ctxs[ci].xk_buf,
                                         ctxs[ci].xk_count,
                                         ctxs[ci].solve_rank,
                                         ctxs[ci].root_rank,
                                         ctxs[ci].cmeta,
                                         ctxs[ci].solve_comm,
                                         comm_reqs, &nreq_xk);
        if (nreq_xk > 0)
            MPI_Waitall(nreq_xk, comm_reqs, MPI_STATUSES_IGNORE);
        symldl_timer.forward_xk += SuperLU_timer_() - ttmp;

        ttmp = SuperLU_timer_();
        double forward_direct_ops = 0.0;
        double forward_direct_calls = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(+:forward_direct_ops,forward_direct_calls)
#endif
        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active || !kmeta->has_panel || cmeta->total_send <= 0)
                continue;
            double panel_ops =
                pdgstrs3d_symldl_panel_solve_ops(kmeta, ctx->ksupc, nrhs);
            if (!pdgstrs3d_symldl_use_direct_cpu_panel(
                    solve_meta, panel_ops, nrhs))
                continue;
            pdgstrs3d_symldl_cpu_forward_panel_direct(
                kmeta, cmeta, ctx->ksupc, nrhs, ctx->xk_buf,
                ctx->send_vals);
            forward_direct_ops += panel_ops;
            forward_direct_calls += 1.0;
        }
        pdgstrs3d_symldl_note_cpu_direct_panel(
            solve_meta, forward_direct_ops, forward_direct_calls);
        stat->ops[SOLVE] += forward_direct_ops;

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active || !kmeta->has_panel || cmeta->total_send <= 0)
                continue;
            double panel_ops =
                pdgstrs3d_symldl_panel_solve_ops(kmeta, ctx->ksupc, nrhs);
            if (pdgstrs3d_symldl_use_direct_cpu_panel(
                    solve_meta, panel_ops, nrhs))
                continue;
            double *lusup = kmeta->lusup;
            int_t nsupr = kmeta->nsupr;
            for (int_t block = 0; block < kmeta->nblocks; ++block) {
                int_t nbrow = kmeta->block_nbrow[block];
                int_t row_start = kmeta->block_row_start[block];
                int_t luptr = kmeta->block_luptr[block];
                pdgstrs3d_symldl_cpu_gemm(
                    solve_meta, "N", "N", nbrow, nrhs, ctx->ksupc,
                    1.0, &lusup[luptr], nsupr, ctx->xk_buf, ctx->ksupc,
                    0.0, gemm_buf, nbrow);
                for (int_t r = 0; r < nbrow; ++r) {
                    int_t row = row_start + r;
                    int pos = cmeta->row_to_send_pos[row];
                    for (int rhs = 0; rhs < nrhs; ++rhs)
                        ctx->send_vals[pos * nrhs + rhs] =
                            -gemm_buf[r + (int_t) rhs * nbrow];
                }
                stat->ops[SOLVE] += 2.0 * (double) nbrow *
                                    (double) nrhs * (double) ctx->ksupc;
            }
        }
        symldl_timer.forward_compute += SuperLU_timer_() - ttmp;

        ttmp = SuperLU_timer_();
        int nreq_values = 0;
        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active)
                continue;
            pdgstrs3d_symldl_post_exchange_double(
                ctx->send_vals, cmeta->send_val_counts,
                cmeta->send_val_displs, ctx->recv_vals,
                cmeta->recv_val_counts, cmeta->recv_val_displs,
                cmeta->nprocs, ctx->solve_rank, LSUM,
                ctx->solve_comm, comm_reqs, &nreq_values);
        }
        if (nreq_values > 0)
            MPI_Waitall(nreq_values, comm_reqs, MPI_STATUSES_IGNORE);
        symldl_timer.forward_values += SuperLU_timer_() - ttmp;

        ttmp = SuperLU_timer_();
        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active)
                continue;
            for (int row = 0; row < cmeta->total_recv; ++row) {
                int_t grow = cmeta->recv_rows[row];
                int_t gsup = BlockNum(grow);
                if (global_rank == diag_owner[gsup]) {
                    int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, gsup);
                    int_t rel = grow - FstBlockC(gsup);
                    int_t gsupsz = SuperSize(gsup);
                    double *xg = &x[X_BLK(lk)];
                    for (int rhs = 0; rhs < nrhs; ++rhs)
                        xg[rel + (int_t) rhs * gsupsz] +=
                            ctx->recv_vals[row * nrhs + rhs];
                }
            }
        }
        symldl_timer.forward_apply += SuperLU_timer_() - ttmp;

    }
    xtrsTimer.t_forwardSolve = SuperLU_timer_() - tx_st;
    xk_buf = workspace->xk_buf;

symldl_diagonal_solve:
    ;
    const char *forward_dump_prefix = getenv(
        "GPU3DV2_SYM_SOLVE_FORWARD_DUMP_PREFIX");
    if (forward_dump_prefix != NULL && forward_dump_prefix[0] != '\0') {
        char filename[4096];
        int length = snprintf(filename, sizeof(filename), "%s.rank%06d.tsv",
                              forward_dump_prefix, global_rank);
        if (length < 0 || (size_t) length >= sizeof(filename))
            ABORT("SymLDL forward diagnostic filename is too long.");
        FILE *dump = fopen(filename, "w");
        if (dump == NULL)
            ABORT("Could not open SymLDL forward diagnostic output.");
        for (int_t k = 0; k < nsupers; ++k) {
            if (global_rank != diag_owner[k])
                continue;
            int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
            int_t width = SuperSize(k);
            double *xk = &x[X_BLK(lk)];
            fprintf(dump, "%lld\t%lld", (long long) k, (long long) width);
            for (int rhs = 0; rhs < nrhs; ++rhs)
                for (int_t i = 0; i < width; ++i)
                    fprintf(dump, "\t%.17e",
                            xk[i + (int_t) rhs * width]);
            fputc('\n', dump);
        }
        if (fclose(dump) != 0)
            ABORT("Could not write SymLDL forward diagnostic output.");
    }
    if (pdgstrs3d_symldl_env_enabled(
            "GPU3DV2_SYM_SOLVE_FORWARD_DIAGNOSTICS")) {
        double local_sum = 0.0;
        double local_abs = 0.0;
        double local_sq = 0.0;
        double local_max = 0.0;
        int local_nonfinite = 0;
        int_t local_count = 0;
        for (int_t k = 0; k < nsupers; ++k) {
            if (global_rank != diag_owner[k])
                continue;
            int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
            int_t width = SuperSize(k);
            double *xk = &x[X_BLK(lk)];
            for (int rhs = 0; rhs < nrhs; ++rhs) {
                for (int_t i = 0; i < width; ++i) {
                    double value = xk[i + (int_t) rhs * width];
                    if (!isfinite(value)) {
                        ++local_nonfinite;
                        continue;
                    }
                    local_sum += value;
                    local_abs += fabs(value);
                    local_sq += value * value;
                    local_max = SUPERLU_MAX(local_max, fabs(value));
                    ++local_count;
                }
            }
        }
        double global_values[4];
        double local_values[4] = {
            local_sum, local_abs, local_sq, local_max
        };
        int global_nonfinite = 0;
        int_t global_count = 0;
        MPI_Allreduce(local_values, global_values, 3, MPI_DOUBLE, MPI_SUM,
                      global_comm);
        MPI_Allreduce(local_values + 3, global_values + 3, 1, MPI_DOUBLE,
                      MPI_MAX, global_comm);
        MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT,
                      MPI_SUM, global_comm);
        MPI_Allreduce(&local_count, &global_count, 1, mpi_int_t, MPI_SUM,
                      global_comm);
        if (global_rank == 0) {
            fprintf(stderr,
                    "SymLDL forward diagnostics: count=%lld nonfinite=%d "
                    "sum=%.17e abs=%.17e norm=%.17e max=%.17e\n",
                    (long long) global_count, global_nonfinite,
                    global_values[0], global_values[1],
                    sqrt(global_values[2]), global_values[3]);
            fflush(stderr);
        }
    }
    if (solve_meta->nvshmem_state != NULL) {
        if (dSymLDLNVSHMEMDiagonal(solve_meta->nvshmem_state,
                                  x, x_count) != 0)
            ABORT("SymLDL NVSHMEM diagonal solve failed.");
        for (int_t k = 0; k < nsupers; ++k) {
            if (global_rank == diag_owner[k]) {
                int_t ksupc = SuperSize(k);
                stat->ops[SOLVE] += 2.0 * (double) ksupc *
                                    (double) ksupc * (double) nrhs;
            }
        }
        goto symldl_backward_solve;
    }
    /* Dense diagonal apply z = D^{-1} y on the canonical diagonal owner. */
    tx = SuperLU_timer_();
    for (int_t level = 0; level < solve_schedule->nlevels; ++level) {
    for (int_t ko = solve_schedule->level_ptr[level];
         ko < solve_schedule->level_ptr[level + 1]; ++ko) {
        int_t k = solve_schedule->nodes[ko];
        int_t tree = (trf3Dpartition && trf3Dpartition->supernode2treeMap)
                         ? trf3Dpartition->supernode2treeMap[k] : -1;
        pdgstrs3d_symldl_tree_comm_t *tree_comm =
            (tree_comms && tree >= 0) ? &tree_comms[tree] : NULL;
        pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
        pdgstrs3d_symldl_comm_meta_t *cmeta = &comm_meta[k];
        int_t ksupc = SuperSize(k);
        int block_count = pdgstrs3d_symldl_count_to_int(ksupc * nrhs,
                                                        "SymLDL diagonal block");

        if (tree_comm == NULL)
            ABORT("SymLDL solve is missing tree metadata for a diagonal block.");
        if (!tree_comm->active || !cmeta->active)
            continue;
        if (global_rank != diag_owner[k])
            continue;
        if (!kmeta->has_diag)
            ABORT("SymLDL solve diagonal owner is missing inverse diagonal block.");

        int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
        double *xk = &x[X_BLK(lk)];
        for (int rhs = 0; rhs < nrhs; ++rhs)
            for (int_t i = 0; i < ksupc; ++i)
                xk_buf[i + (int_t) rhs * ksupc] =
                    xk[i + (int_t) rhs * ksupc];

        ttmp = SuperLU_timer_();
        pdgstrs3d_symldl_cpu_gemm(
            solve_meta, "N", "N", ksupc, nrhs, ksupc,
            1.0, &kmeta->lusup[kmeta->diag_luptr], kmeta->nsupr,
            xk_buf, ksupc, 0.0, diag_buf, ksupc);
        symldl_timer.diag_compute += SuperLU_timer_() - ttmp;
        stat->ops[SOLVE] += 2.0 * (double) ksupc * (double) ksupc *
                            (double) nrhs;

        for (int rhs = 0; rhs < nrhs; ++rhs)
            for (int_t i = 0; i < ksupc; ++i)
                xk[i + (int_t) rhs * ksupc] =
                    diag_buf[i + (int_t) rhs * ksupc];
    }
    }
    (void) tx;

symldl_backward_solve:
    ;
    if (solve_meta->nvshmem_state != NULL) {
        tx = SuperLU_timer_();
        if (dSymLDLNVSHMEMBackward(solve_meta->nvshmem_state,
                                   x, x_count) != 0)
            ABORT("SymLDL NVSHMEM backward solve failed.");
        for (int_t k = 0; k < nsupers; ++k) {
            pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
            if (!kmeta->has_panel ||
                !pdgstrs3d_symldl_local_panel_active(
                    trf3Dpartition, kmeta, k))
                continue;
            stat->ops[SOLVE] +=
                pdgstrs3d_symldl_panel_solve_ops(
                    kmeta, SuperSize(k), nrhs);
        }
        xtrsTimer.t_backwardSolve = SuperLU_timer_() - tx;
        goto symldl_backward_done;
    }
    /* Backward solve with L^T using per-tree replicated row values. */
    tx = SuperLU_timer_();
    for (int_t level = solve_schedule->nlevels; level > 0; --level) {
        int_t level_begin = solve_schedule->level_ptr[level - 1];
        int_t level_end = solve_schedule->level_ptr[level];
        int_t nctx = level_end - level_begin;
        int_t max_level_reqs = 1;
        int_t level_delta_count = 0;
        int_t level_delta_recv_count = 0;
        int_t delta_offset = 0;
        int_t delta_recv_offset = 0;
        pdgstrs3d_symldl_node_ctx_t *ctxs =
            pdgstrs3d_symldl_workspace_prepare_level_contexts(
                workspace, nctx, "Malloc fails for SymLDL backward level contexts.");

        for (int_t ci = 0; ci < nctx; ++ci) {
            int_t k = solve_schedule->nodes[level_begin + ci];
            int_t tree = (trf3Dpartition && trf3Dpartition->supernode2treeMap)
                             ? trf3Dpartition->supernode2treeMap[k] : -1;
            pdgstrs3d_symldl_tree_comm_t *tree_comm =
                (tree_comms && tree >= 0) ? &tree_comms[tree] : NULL;
            pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
            pdgstrs3d_symldl_comm_meta_t *cmeta = &comm_meta[k];
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            int_t ksupc = SuperSize(k);

            if (tree_comm == NULL)
                ABORT("SymLDL solve is missing tree metadata for a backward panel.");
            if (!tree_comm->active || !cmeta->active)
                continue;

            ctx->active = 1;
            ctx->k = k;
            ctx->ksupc = ksupc;
            ctx->delta_count = pdgstrs3d_symldl_count_to_int(
                ksupc * nrhs, "SymLDL backward delta");
            ctx->root_rank = pdgstrs3d_symldl_rank_to_tree_rank(
                tree_comm, diag_owner[k]);
            ctx->solve_rank = tree_comm->rank;
            ctx->solve_comm = tree_comm->comm;
            ctx->kmeta = kmeta;
            ctx->cmeta = cmeta;

            level_delta_count = pdgstrs3d_symldl_count_sum(
                level_delta_count, ctx->delta_count,
                "SymLDL backward delta buffers");
            if (ctx->solve_rank == ctx->root_rank &&
                cmeta->xk_receiver_count > 0) {
                int_t recv_count = pdgstrs3d_checked_workspace_count(
                    cmeta->xk_receiver_count, ctx->delta_count, 0, 0,
                    "SymLDL backward delta receive buffer");
                level_delta_recv_count = pdgstrs3d_symldl_count_sum(
                    level_delta_recv_count, recv_count,
                    "SymLDL backward delta receive buffer");
            }

            max_level_reqs += cmeta->xk_receiver_count + 1;
        }

        ttmp = SuperLU_timer_();
        symldl_timer.x_cache_fill += pdgstrs3d_symldl_x_cache_fill(
            solve_meta, x, nrhs, ilsum, grid3d, ctxs, nctx);

        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->delta_send_buf, &workspace->delta_send_cap,
            SUPERLU_MAX(level_delta_count, (int_t) 1),
            "Malloc fails for SymLDL backward delta buffers.");
        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->delta_buf, &workspace->delta_cap,
            SUPERLU_MAX(level_delta_count, (int_t) 1),
            "Malloc fails for SymLDL backward delta buffers.");
        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->delta_recv_buf, &workspace->delta_recv_cap,
            SUPERLU_MAX(level_delta_recv_count, (int_t) 1),
            "Malloc fails for SymLDL backward delta receive buffer.");
        pdgstrs3d_symldl_grow_request_buffer(
            &workspace->comm_reqs, &workspace->comm_reqs_cap,
            max_level_reqs, "Malloc fails for SymLDL backward level requests.");
        comm_reqs = workspace->comm_reqs;
        pdgstrs3d_symldl_zero_double_buffer(
            workspace->delta_send_buf, level_delta_count,
            "SymLDL backward delta send buffer");
        pdgstrs3d_symldl_zero_double_buffer(
            workspace->delta_buf, level_delta_count,
            "SymLDL backward delta buffer");

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;

            if (!ctx->active)
                continue;

            ctx->delta_send_buf = &workspace->delta_send_buf[delta_offset];
            ctx->delta_buf = &workspace->delta_buf[delta_offset];
            delta_offset += ctx->delta_count;
            if (ctx->solve_rank == ctx->root_rank &&
                cmeta->xk_receiver_count > 0) {
                int_t recv_count = pdgstrs3d_checked_workspace_count(
                    cmeta->xk_receiver_count, ctx->delta_count, 0, 0,
                    "SymLDL backward delta receive buffer");
                ctx->delta_recv_buf =
                    &workspace->delta_recv_buf[delta_recv_offset];
                delta_recv_offset += recv_count;
            }
            if (kmeta->has_panel && cmeta->total_send > 0) {
                pdgstrs3d_symldl_x_cache_t *cache =
                    &solve_meta->x_cache[ctx->k];
                if (cache->values == NULL ||
                    cache->row_count != kmeta->row_count)
                    ABORT("SymLDL backward solve missing replicated X cache.");
                ctx->row_values = cache->values;
            }
        }

        ttmp = SuperLU_timer_();
        double backward_direct_ops = 0.0;
        double backward_direct_calls = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(+:backward_direct_ops,backward_direct_calls)
#endif
        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active || !kmeta->has_panel || cmeta->total_send <= 0)
                continue;
            double panel_ops =
                pdgstrs3d_symldl_panel_solve_ops(kmeta, ctx->ksupc, nrhs);
            if (!pdgstrs3d_symldl_use_direct_cpu_panel(
                    solve_meta, panel_ops, nrhs))
                continue;
            pdgstrs3d_symldl_cpu_backward_panel_direct(
                kmeta, ctx->ksupc, nrhs, ctx->row_values,
                ctx->delta_send_buf);
            backward_direct_ops += panel_ops;
            backward_direct_calls += 1.0;
        }
        pdgstrs3d_symldl_note_cpu_direct_panel(
            solve_meta, backward_direct_ops, backward_direct_calls);
        stat->ops[SOLVE] += backward_direct_ops;

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active || !kmeta->has_panel || cmeta->total_send <= 0)
                continue;
            double panel_ops =
                pdgstrs3d_symldl_panel_solve_ops(kmeta, ctx->ksupc, nrhs);
            if (pdgstrs3d_symldl_use_direct_cpu_panel(
                    solve_meta, panel_ops, nrhs))
                continue;
            double *lusup = kmeta->lusup;
            int_t nsupr = kmeta->nsupr;
            int seq = 0;
            for (int_t block = 0; block < kmeta->nblocks; ++block) {
                int_t nbrow = kmeta->block_nbrow[block];
                int_t luptr = kmeta->block_luptr[block];
                for (int rhs = 0; rhs < nrhs; ++rhs)
                    for (int_t r = 0; r < nbrow; ++r)
                        rhs_buf[r + (int_t) rhs * nbrow] =
                            ctx->row_values[(seq + r) * nrhs + rhs];
                pdgstrs3d_symldl_cpu_gemm(
                    solve_meta, "T", "N", ctx->ksupc, nrhs, nbrow,
                    -1.0, &lusup[luptr], nsupr, rhs_buf, nbrow,
                    1.0, ctx->delta_send_buf, ctx->ksupc);
                seq += nbrow;
                stat->ops[SOLVE] += 2.0 * (double) nbrow *
                                    (double) nrhs * (double) ctx->ksupc;
            }
        }
        symldl_timer.backward_compute += SuperLU_timer_() - ttmp;

        ttmp = SuperLU_timer_();
        int nreq_delta = 0;
        for (int_t ci = 0; ci < nctx; ++ci)
            if (ctxs[ci].active)
                pdgstrs3d_symldl_post_reduce_delta(&ctxs[ci],
                                                   comm_reqs, &nreq_delta);
        if (nreq_delta > 0)
            MPI_Waitall(nreq_delta, comm_reqs, MPI_STATUSES_IGNORE);
        for (int_t ci = 0; ci < nctx; ++ci)
            if (ctxs[ci].active)
                pdgstrs3d_symldl_finish_reduce_delta(&ctxs[ci]);
        symldl_timer.backward_delta += SuperLU_timer_() - ttmp;

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            if (!ctx->active || global_rank != diag_owner[ctx->k])
                continue;
            int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, ctx->k);
            double *xk = &x[X_BLK(lk)];
            for (int rhs = 0; rhs < nrhs; ++rhs)
                for (int_t c = 0; c < ctx->ksupc; ++c)
                    xk[c + (int_t) rhs * ctx->ksupc] +=
                        ctx->delta_buf[c + (int_t) rhs * ctx->ksupc];
        }

    }
    xtrsTimer.t_backwardSolve = SuperLU_timer_() - tx;

symldl_backward_done:
    MPI_Barrier(grid3d->comm);
    stat->utime[SOLVE] = SuperLU_timer_() - tx_st;

    tx = SuperLU_timer_();
    pdReDistribute3d_X_to_B_symv2(n, B, m_loc, ldb, fst_row, nrhs, x,
                                  ilsum, ScalePermstruct, Glu_persist,
                                  trf3Dpartition, grid3d, SOLVEstruct);
    xtrsTimer.t_pxReDistribute_X_to_B = SuperLU_timer_() - tx;
    symldl_timer.x_to_b = xtrsTimer.t_pxReDistribute_X_to_B;

    reduceStat(SOLVE, stat, grid3d);
    pdgstrs3d_symldl_nvshmem_take_timers(solve_meta, &symldl_timer);
    pdgstrs3d_symldl_literature_take_timers(solve_meta, &symldl_timer);
    if (pdgstrs3d_symldl_env_enabled("GPU3DV2_SYM_SOLVE_TIMING"))
        pdgstrs3d_symldl_timer_print(&symldl_timer, solve_meta, grid3d);

#if ( PRNTlevel >= 1 )
    printTRStimer(&xtrsTimer, grid3d);
#endif

    return;
}

/*! \brief
 *
 *   Experimental LDL-native solve for the SymFact GPU3D v2 factor path.
 *   This uses the tree-scheduled distributed solve for all process grids,
 *   including the 1x1x1 local case, while keeping the existing B<->X
 *   redistribution contract.
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
