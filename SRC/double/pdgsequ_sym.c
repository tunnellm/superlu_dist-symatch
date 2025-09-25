/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Computes row and column scalings
 *
 * File name:	pdgsequ.c
 * History:     Modified from LAPACK routine DGEEQU
 */
#include <math.h>
#include "superlu_ddefs.h"

/*! \brief

 <pre>
    Purpose
    =======

    PDGSEQU computes row and column scalings intended to equilibrate an
    M-by-N sparse matrix A and reduce its condition number. R returns the row
    scale factors and C the column scale factors, chosen to try to make
    the largest element in each row and column of the matrix B with
    elements B(i,j)=R(i)*A(i,j)*C(j) have absolute value 1.

    R(i) and C(j) are restricted to be between SMLNUM = smallest safe
    number and BIGNUM = largest safe number.  Use of these scaling
    factors is not guaranteed to reduce the condition number of A but
    works well in practice.

    See supermatrix.h for the definition of 'SuperMatrix' structure.

    Arguments
    =========

    A       (input) SuperMatrix*
            The matrix of dimension (A->nrow, A->ncol) whose equilibration
            factors are to be computed. The type of A can be:
            Stype = SLU_NR_loc; Dtype = SLU_D; Mtype = SLU_GE.

    R       (output) double*, size A->nrow
            If INFO = 0 or INFO > M, R contains the row scale factors
            for A.

    C       (output) double*, size A->ncol
            If INFO = 0,  C contains the column scale factors for A.

    ROWCND  (output) double*
            If INFO = 0 or INFO > M, ROWCND contains the ratio of the
            smallest R(i) to the largest R(i).  If ROWCND >= 0.1 and
            AMAX is neither too large nor too small, it is not worth
            scaling by R.

    COLCND  (output) double*
            If INFO = 0, COLCND contains the ratio of the smallest
            C(i) to the largest C(i).  If COLCND >= 0.1, it is not
            worth scaling by C.

    AMAX    (output) double*
            Absolute value of largest matrix element.  If AMAX is very
            close to overflow or very close to underflow, the matrix
            should be scaled.

    INFO    (output) int*
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i,  and i is
                  <= M:  the i-th row of A is exactly zero
                  >  M:  the (i-M)-th column of A is exactly zero

    GRID    (input) gridinof_t*
            The 2D process mesh.
    =====================================================================
</pre>
*/

void
pdgsequ_sym(SuperMatrix *A, double *r, double *c, gridinfo_t *grid, int *iinfo, char *equed)
{

#define PRECISION   (0.0001)
    int iam = grid->iam;
    /* Quick return if possible */
    if (A->nrow <= 0 || A->ncol <= 0) {
        *(unsigned char *)equed = 'N';
        return;
    }

    int_t i, j;
    double rowcnd, colcnd, amax, rowcri, colcri;
    double *iter_c, *iter_r;
    char iter_equed[1];
    *iter_equed = 'N';

    if ( !(iter_r = doubleMalloc_dist(A->nrow)) )
        ABORT("SUPERLU_MALLOC fails for iter_r[]");
    if ( !(iter_c = doubleMalloc_dist(A->ncol)) )
        ABORT("SUPERLU_MALLOC fails for iter_c[]");

    for (i = 0; i < A->nrow; ++i) { iter_r[i] = r[i]; r[i] = 1.0; }
    for (j = 0; j < A->ncol; ++j) { iter_c[j] = c[j]; c[j] = 1.0; }

    int converged = 0;
    int equi_iter = 0;

    *(unsigned char *)equed = 'N';
    while (converged == 0) {
        /* Compute the row and column scalings. */
        pdgsequ_new(A, iter_r, iter_c, &rowcnd, &colcnd, &rowcri, &colcri, &amax, iinfo, grid);
        if ( *iinfo > 0 ) {
            if ( *iinfo <= A->nrow ) {
                fprintf(stderr, "The %d-th row of A in %d-th iteration of MC77 is exactly zero\n", (int) *iinfo, equi_iter);
            } else {
                fprintf(stderr, "The %d-th column of A in %d-th iteration of MC77 is exactly zero\n", (int) (*iinfo-(A->ncol)), equi_iter);
            }
        } else if ( *iinfo < 0 ) return;

        /* Equilibrate matrix A if it is badly-scaled.
           A <-- diag(R)*A*diag(C)                     */
        pdlaqgs(A, iter_r, iter_c, rowcnd, colcnd, amax, iter_equed);

        if(!iam){
            printf("In iteration %d, rowcnd is %f, colcnd is %f.\n", equi_iter, rowcnd, colcnd);
            fflush(stdout);
        }

        if ( strncmp(iter_equed, "R", 1)==0 ) {
            for (i = 0; i < A->nrow; ++i) { r[i] *= iter_r[i]; }

            if ( ( strncmp(equed, "R", 1)==0 ) || ( strncmp(equed, "N", 1)==0 ) ) {
                *(unsigned char *)equed = 'R';
            } else {
                *(unsigned char *)equed = 'B';
            }
        } else if ( strncmp(iter_equed, "C", 1)==0 ) {
            for (j = 0; j < A->ncol; ++j) { c[j] *= iter_c[j]; }

            if ( ( strncmp(equed, "C", 1)==0 ) || ( strncmp(equed, "N", 1)==0 ) ) {
                *(unsigned char *)equed = 'C';
            } else {
                *(unsigned char *)equed = 'B';
            }
        } else if ( strncmp(iter_equed, "B", 1)==0 ) {
            for (i = 0; i < A->nrow; ++i) { r[i] *= iter_r[i]; }
            for (j = 0; j < A->ncol; ++j) { c[j] *= iter_c[j]; }

            *(unsigned char *)equed = 'B';
        } else {
            converged = 1;
        }

        equi_iter++;

        if ( (rowcri <= PRECISION) && (colcri <= PRECISION) ) {
            converged = 1;
        }
    }
    if(!iam){
        printf("MC77 takes %d iterations to converge.\n", equi_iter);
        fflush(stdout);
    }

    SUPERLU_FREE(iter_r);
    SUPERLU_FREE(iter_c);

    return;

} /* pdgsequ */
