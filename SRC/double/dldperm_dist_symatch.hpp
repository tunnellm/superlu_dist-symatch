#pragma once

#include "superlu_ddefs.h"

// symatch headers
#include "gmodel.hpp"
#include "gutil.hpp"
#include "tdef.hpp"
#include "util.hpp"
#include "wrmatch.hpp"


int
dldperm_dist_symatch(int job, int n, int_t nnz, int_t colptr[], int_t adjncy[],
					 double nzval[], int_t *perm, double u[], double v[],
					 int_t *n_crs, int_t **crs_vrts, int_t **fine_vids);

int
coarsen_graph(SuperMatrix *G, SuperMatrix *Gc, int_t n_crs, int_t *crs_vrts);

int
coarsen_graph_v2(SuperMatrix *G, SuperMatrix *Gc, int_t n_crs, int_t *crs_vrts);

void
apply_perm_sym(int n, int_t nnz, int_t *colptr, int_t *adjncy, double *nzval,
			   int_t *p);


int
is_symmetric(int n, int_t nnz, int_t *colptr, int_t *adjncy, double *nzval);

int
is_postorder(int n, int *parents);
