/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Non-CUDA stubs for the literature-faithful SymLDL forward solve.
 */

#include "dsymldl_v2_literature_forward.h"

int
dSymLDLLiteratureForwardAvailable(void)
{
    return 0;
}

dSymLDLLiteratureForwardHandle
dSymLDLLiteratureForwardCreate(
    int_t n, int_t nsupers, int nrhs, int_t x_count, int_t lsum_count,
    int_t panel_count, const dSymLDLLiteraturePanelDesc *panels,
    int_t block_count, const dSymLDLLiteratureBlockDesc *blocks,
    int_t row_count, const int_t *rows,
    const int_t *xsup, const int_t *ilsum,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d)
{
    (void) n;
    (void) nsupers;
    (void) nrhs;
    (void) x_count;
    (void) lsum_count;
    (void) panel_count;
    (void) panels;
    (void) block_count;
    (void) blocks;
    (void) row_count;
    (void) rows;
    (void) xsup;
    (void) ilsum;
    (void) trf3Dpartition;
    (void) grid3d;
    return NULL;
}

int
dSymLDLLiteratureForwardInitializeRHS(
    dSymLDLLiteratureForwardHandle handle, double *x, int_t x_count)
{
    (void) handle;
    (void) x;
    (void) x_count;
    return -1;
}

int
dSymLDLLiteratureForwardSolve(
    dSymLDLLiteratureForwardHandle handle, double *x, int_t x_count)
{
    (void) handle;
    (void) x;
    (void) x_count;
    return -1;
}

int
dSymLDLLiteratureForwardSparseAllreduce(
    dSymLDLLiteratureForwardHandle handle, double *x, int_t x_count)
{
    (void) handle;
    (void) x;
    (void) x_count;
    return -1;
}

void
dSymLDLLiteratureForwardTakeTimers(
    dSymLDLLiteratureForwardHandle handle, double *setup, double *forward,
    double *sparse_reduce, double *sparse_broadcast, double *h2d,
    double *d2h)
{
    (void) handle;
    if (setup != NULL) *setup = 0.0;
    if (forward != NULL) *forward = 0.0;
    if (sparse_reduce != NULL) *sparse_reduce = 0.0;
    if (sparse_broadcast != NULL) *sparse_broadcast = 0.0;
    if (h2d != NULL) *h2d = 0.0;
    if (d2h != NULL) *d2h = 0.0;
}

void
dSymLDLLiteratureForwardDestroy(
    dSymLDLLiteratureForwardHandle handle)
{
    (void) handle;
}
