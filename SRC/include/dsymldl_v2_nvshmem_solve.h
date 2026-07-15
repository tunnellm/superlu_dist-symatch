#ifndef DSYMLDL_V2_NVSHMEM_SOLVE_H
#define DSYMLDL_V2_NVSHMEM_SOLVE_H

#include "superlu_ddefs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int_t gid;
    int_t width;
    int_t nsupr;
    int_t diag_luptr;
    int has_diag;
    int_t block_begin;
    int_t block_count;
    int_t value_count;
    int active;
    double *values;
} dSymLDLNVPanelDesc;

typedef struct {
    int_t panel_id;
    int_t target_gid;
    int_t luptr;
    int_t nbrow;
    int_t row_begin;
} dSymLDLNVBlockDesc;

typedef void *dSymLDLNVSHMEMSolveHandle;

int dSymLDLNVSHMEMSolveAvailable(void);

dSymLDLNVSHMEMSolveHandle dSymLDLNVSHMEMSolveCreate(
    int_t n, int_t nsupers, int nrhs, int_t x_count, int_t lsum_count,
    int_t panel_count, const dSymLDLNVPanelDesc *panels,
    int_t block_count, const dSymLDLNVBlockDesc *blocks,
    int_t row_count, const int_t *rows,
    const int_t *xsup, const int_t *ilsum,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d);

int dSymLDLNVSHMEMForward(dSymLDLNVSHMEMSolveHandle handle,
                          double *x, int_t x_count);

int dSymLDLNVSHMEMDiagonal(dSymLDLNVSHMEMSolveHandle handle,
                           double *x, int_t x_count);

int dSymLDLNVSHMEMBackward(dSymLDLNVSHMEMSolveHandle handle,
                           double *x, int_t x_count);

void dSymLDLNVSHMEMSolveTakeTimers(
    dSymLDLNVSHMEMSolveHandle handle, double *setup, double *forward,
    double *sparse_reduce, double *sparse_broadcast, double *diagonal,
    double *backward, double *h2d, double *d2h);

void dSymLDLNVSHMEMSolveDestroy(dSymLDLNVSHMEMSolveHandle handle);

#ifdef __cplusplus
}
#endif

#endif
