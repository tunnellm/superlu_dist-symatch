/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

#ifndef DSYMLDL_V2_LITERATURE_FORWARD_H
#define DSYMLDL_V2_LITERATURE_FORWARD_H

#include "superlu_ddefs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int_t gid;
    int_t lsub_count;
    int_t lloc_count;
    int_t value_count;
    const int_t *lsub;
    const int_t *lloc;
    const double *host_values;
    double *values;
} dSymLDLLiteraturePanelDesc;

typedef void *dSymLDLLiteratureForwardHandle;

int dSymLDLLiteratureForwardAvailable(void);

dSymLDLLiteratureForwardHandle dSymLDLLiteratureForwardCreate(
    int_t n, int_t nsupers, int nrhs, int_t x_count, int_t lsum_count,
    int_t panel_count, const dSymLDLLiteraturePanelDesc *panels,
    const int_t *xsup, const int_t *ilsum,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d);

int dSymLDLLiteratureForwardInitializeRHS(
    dSymLDLLiteratureForwardHandle handle, double *x, int_t x_count);

int dSymLDLLiteratureForwardSolve(
    dSymLDLLiteratureForwardHandle handle, double *x, int_t x_count);

int dSymLDLLiteratureForwardSparseAllreduce(
    dSymLDLLiteratureForwardHandle handle, double *x, int_t x_count);

void dSymLDLLiteratureForwardTakeTimers(
    dSymLDLLiteratureForwardHandle handle, double *setup, double *forward,
    double *sparse_reduce, double *sparse_broadcast, double *h2d,
    double *d2h);

void dSymLDLLiteratureForwardDestroy(
    dSymLDLLiteratureForwardHandle handle);

#ifdef __cplusplus
}
#endif

#endif
