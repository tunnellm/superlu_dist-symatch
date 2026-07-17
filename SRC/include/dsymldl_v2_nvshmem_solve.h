#ifndef DSYMLDL_V2_NVSHMEM_SOLVE_H
#define DSYMLDL_V2_NVSHMEM_SOLVE_H

#include "dsymldl_v2_solve_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef dSymLDLPanelDesc dSymLDLNVPanelDesc;
typedef dSymLDLBlockDesc dSymLDLNVBlockDesc;

typedef void *dSymLDLNVSHMEMSolveHandle;

int dSymLDLNVSHMEMSolveAvailable(void);

dSymLDLNVSHMEMSolveHandle dSymLDLNVSHMEMSolveCreate(
    int nrhs, int_t x_count, int_t lsum_count,
    const dSymLDLSolveGraph *graph,
    const dSymLDLNVPanelDesc *device_panels,
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
