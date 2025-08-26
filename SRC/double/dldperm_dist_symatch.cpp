#include <cassert>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "dldperm_dist_symatch.hpp"

using std::cout;	using std::endl;	using std::string;	using std::cerr;
using std::vector;	using std::tuple;	using std::unordered_map;

// #define DBG_MATCHING

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 *   DLDPERM finds a row permutation so that the matrix has large
 *   entries on the diagonal.
 *
 * Arguments
 * =========
 *
 * job    (input) int
 *        Control the action. Possible values for JOB are:
 *        = 1 : Compute a row permutation of the matrix so that the
 *              permuted matrix has as many entries on its diagonal as
 *              possible. The values on the diagonal are of arbitrary size.
 *              HSL subroutine MC21A/AD is used for this.
 *        = 2 : Compute a row permutation of the matrix so that the smallest
 *              value on the diagonal of the permuted matrix is maximized.
 *        = 3 : Compute a row permutation of the matrix so that the smallest
 *              value on the diagonal of the permuted matrix is maximized.
 *              The algorithm differs from the one used for JOB = 2 and may
 *              have quite a different performance.
 *        = 4 : Compute a row permutation of the matrix so that the sum
 *              of the diagonal entries of the permuted matrix is maximized.
 *        = 5 : Compute a row permutation of the matrix so that the product
 *              of the diagonal entries of the permuted matrix is maximized
 *              and vectors to scale the matrix so that the nonzero diagonal
 *              entries of the permuted matrix are one in absolute value and
 *              all the off-diagonal entries are less than or equal to one in
 *              absolute value.
 *        Restriction: 1 <= JOB <= 5.
 *
 * n      (input) int
 *        The order of the matrix.
 *
 * nnz    (input) int
 *        The number of nonzeros in the matrix.
 *
 * adjncy (input) int*, of size nnz
 *        The adjacency structure of the matrix, which contains the row
 *        indices of the nonzeros.
 *
 * colptr (input) int*, of size n+1
 *        The pointers to the beginning of each column in ADJNCY.
 *
 * nzval  (input) double*, of size nnz
 *        The nonzero values of the matrix. nzval[k] is the value of
 *        the entry corresponding to adjncy[k].
 *        It is not used if job = 1.
 *
 * perm   (output) int*, of size n
 *        The permutation vector. perm[i] = j means row i in the
 *        original matrix is in row j of the permuted matrix.
 *
 * u      (output) double*, of size n
 *        If job = 5, the natural logarithms of the row scaling factors.
 *
 * v      (output) double*, of size n
 *        If job = 5, the natural logarithms of the column scaling factors.
 *        The scaled matrix B has entries b_ij = a_ij * exp(u_i + v_j).
 * </pre>
 */

int
dldperm_dist_symatch
(
    int		  job,
	int		  n,
	int_t	  nnz,
	int_t	  colptr[],
	int_t	  adjncy[],
	double	  nzval[],
	int_t	 *perm,
	double	  R1[],
	double	  C1[],
	int_t	 *n_crs,
	int_t	**crs_vrts

)
{
	cout << string(80, '=') << endl;


	// Create a matrix market object from CSC.
	// Unfilled members: svals, header.
	MatMarket_t<int_t, double>	mm;
	mm.fmt		  = MatMarket_t<int_t, double>::Coordinate;
	mm.type		  = MatMarket_t<int_t, double>::Real;
	mm.storage	  = MatMarket_t<int_t, double>::Symmetric;	// assumed
	mm.nr		  = mm.nc = n;
	mm.nnzs		  = nnz;
	mm.vals_exist = true;

	uint64_t ndiags = 0;
	mm.nelems = 0;
	for (int_t c = 0; c < n; ++c)
	{
		for (int_t rptr = colptr[c]; rptr < colptr[c+1]; ++rptr)
		{
			int_t	r = adjncy[rptr];
			double	v = nzval[rptr];

			if (r == c)
				++ndiags;

			if (r <= c)			// U
			{
				mm.rids.push_back(r);
				mm.cids.push_back(c);
				mm.vals.push_back(v);
				++(mm.nelems);
			}
		}
	}

	assert(((nnz - ndiags) & 0x1) == 0x0);
	cout << "#nnzs " << mm.nnzs << " #diags " << ndiags << endl;


	// Form the graph model for matching.
	GModel *gm = new StandardSymDiag(&mm);
	gm->form_graph(true);
	auto *g = gm->g;


	cout << "formed the graph model, running the matching..." << endl;


	// check the suitor library's global params
	if (g->nv > max_n || g->xadj[g->nv]+1 > max_m)
	{
		cerr << "matching lib's max_n and max_m variables too small "
			 << max_n << " " << max_m
			 << " graph nv " << g->nv
			 << " adj size " << g->xadj[g->nv]
			 << " .ABORT."
			 << endl;
		exit(EXIT_FAILURE);
	}


	// WrMatch *wrm = new WgtGPASeq(g);
	WrMatch *wrm = new WgtSuitorSeqFor(g);
	wrm->match();

	fprintf(stdout, "matching cost %lf\n", wrm->cost());
	fprintf(stdout, "matching cost (prod) %lf\n", wrm->cost_prod());


	#ifdef DBG_MATCHING
	ofstream outfile;
	outfile.open("debug-output");

	outfile << "=== Constructed graph === \n";
	for (int64_t v = 0; v < g->nv; ++v)
	{
		outfile << v << ": ";
		for (int_t i = g->xadj[v]; i < g->xadj[v+1]; ++i)
			outfile << "(" << g->adj[i] << ","
					<< g->ew[i] << ") ";
		outfile << "\n";
	}
	outfile << "\n\n\n";

	outfile << "=== Matching (1-based indices) ===\n";
	#endif


	// Compute the permutation
	vector<tuple<int_t, int_t, double>> m_sorted;
	wrm->sort_matching(m_sorted, false);
	int_t curidx = 0;
	*n_crs = 0;
	for (auto &t : m_sorted)
	{
		int_t v = get<0>(t);
		int_t u = get<1>(t);
		assert(v != 0);

		#ifdef DBG_MATCHING
		outfile << v << " " << u << " " << get<2>(t) << "\n";
		#endif

		if (u == 0)				// unmatched vertex
		{
			if (v <= mm.nr)
			{
				++(*n_crs);
				perm[v-1] = curidx++;
			}
			continue;
		}

		++(*n_crs);

		perm[min(v, u)-1] = curidx++;
		if (u <= mm.nr && v <= mm.nr)
			perm[max(v, u)-1] = curidx++;
	}


	#ifdef DBG_MATCHING
	outfile << "\n\n\n";

	outfile << "=== Permutation (perm_r) ===\n";
	for (int_t v = 0; v < mm.nr; ++v)
		outfile << v << " -> " << perm[v] << "\n";

	outfile << "\n\n\n";
	#endif


	// 2nd pass - coarsening information.
	if (*n_crs > 0)
		*crs_vrts  = (int_t *)malloc(sizeof(**crs_vrts) * (*n_crs));

	int_t	crs_idx = 0;
	for (auto &t : m_sorted)
	{
		int_t v = get<0>(t);
		int_t u = get<1>(t);

		if (u == 0)				// unmatched vertex
		{
			if (v <= mm.nr)
			{
				// assert(crs_idx < (*n_crs));
				(*crs_vrts)[crs_idx++] = 1;
			}
			continue;
		}

		// assert(crs_idx < (*n_crs));
		(*crs_vrts)[crs_idx] = 1;
		if (u <= mm.nr && v <= mm.nr)
			(*crs_vrts)[crs_idx] = 2;
		++crs_idx;
	}


	#ifdef DBG_MATCHING
	outfile << "=== Coarsening info ===\n";
	outfile << "n_crs " << *n_crs << "\n";

	for (int_t i = 0; i < *n_crs; ++i)
		outfile << i << " " << (*crs_vrts)[i] << "\n";

	outfile << "\n\n\n";
	outfile.close();
	#endif

	
	cout << "#2x2 " << n-(*n_crs) << " #1x1 " << 2*(*n_crs)-n << "\n";
	cout << "Number of coarse vertices " << *n_crs << "\n";
	cout << string(80, '=') << endl;

	

	delete gm;
	delete wrm;



	return 0;
}





int
coarsen_graph
(
    SuperMatrix *G,
	SuperMatrix *Gc,			// output coarsened graph
	int_t		 n_crs,
	int_t		*crs_vrts
)
{
	assert(G->Stype == SLU_NC &&
		   "Coarsening only implemented for storage type SLU_NC.\n");

	cout << "Coarsening the graph..." << endl;

	NCformat	*Gstore	= (NCformat *) G->Store;
	int_t		*colptr = Gstore->colptr;
	int_t		*rowind = Gstore->rowind;
	double		*nzval	= (double *)Gstore->nzval;
	int_t		 nr		= G->nrow;
	int_t		 nc		= G->ncol;

	assert(nr == nc && "Matrix should be square.\n");


	int_t	*v_cid = new int_t[nr];
	int_t	 v	   = 0;
	for (int_t cv = 0; cv < n_crs; ++cv)
	{
		assert(crs_vrts[cv] == 1 || crs_vrts[cv] == 2);
		for (int_t i = 0; i < crs_vrts[cv]; ++i)
		{
			assert(v < nr);
			v_cid[v++] = cv;
		}
	}


	assert(v == nr);


	// Find the neighbors of coarse vertices and their edge weights.
	vector<unordered_map<int_t, double>> crs_ngbrs(n_crs);
	int_t crs_nnz = 0;
	for (int_t c = 0; c < nc; ++c)
	{
		int_t c_c = v_cid[c];
		assert(c_c >= 0 && c_c < n_crs);
		for (int_t rptr = colptr[c]; rptr < colptr[c+1]; ++rptr)
		{
			int_t	r	= rowind[rptr];
			int_t	r_c = v_cid[r];
			assert(r_c >= 0 && r_c < n_crs);
			if (crs_ngbrs[c_c].find(r_c) == crs_ngbrs[c_c].end())
			{
				crs_ngbrs[c_c].insert({r_c, nzval[rptr]});
				++crs_nnz;
			}
		}
	}

	cout << "#nnzs in the coarse graph " << crs_nnz << endl;


	// Form the coarse graph.
	// @WARNING Do not make direct assignment for store type.
	Gc->Stype			 = G->Stype;
	Gc->Dtype			 = G->Dtype;
	Gc->Mtype			 = G->Mtype;
	Gc->nrow			 = n_crs;
	Gc->ncol			 = n_crs;
	Gc->Store			 = (NCformat *) SUPERLU_MALLOC(sizeof(NCformat));
	NCformat	*Gcstore = (NCformat *) Gc->Store;

	Gcstore->nnz	   = crs_nnz;
	Gcstore->nzval	   = (double *) doubleMalloc_dist(crs_nnz);
	double	*crs_nzval = (double *) Gcstore->nzval;
	Gcstore->rowind	   = (int_t *) intMalloc_dist(crs_nnz);
	// Gcstore->rowind = (int_t *) SUPERLU_MALLOC((crs_nnz)*sizeof(int_t));
	Gcstore->colptr	   = (int_t *) intMalloc_dist(n_crs+1);
	// Gcstore->colptr = (int_t *) SUPERLU_MALLOC((n_crs+1)*sizeof(int_t));

	Gcstore->colptr[0] = 0;
	int_t curnz = 0;
	for (int_t cv = 0; cv < n_crs; ++cv)
	{
		Gcstore->colptr[cv+1] = Gcstore->colptr[cv] + crs_ngbrs[cv].size();
		for (auto &el : crs_ngbrs[cv])
		{
			assert(curnz < crs_nnz);
			Gcstore->rowind[curnz] = el.first;
			crs_nzval[curnz]	   = el.second;
			++curnz;
		}
	}


	#ifdef DBG_MATCHING
	ofstream outfile("debug-output", std::ios_base::app);
	outfile << "coarse graph #rows/cols " << n_crs
			<< " #nnz " << crs_nnz << endl;
	outfile << "=== Coarse matrix ===\n";
	for (int_t c = 0; c < n_crs; ++c)
	{
		outfile << c << ": ";
		for (int_t rptr = Gcstore->colptr[c]; rptr < Gcstore->colptr[c+1];
			 ++rptr)
			outfile << "(" << Gcstore->rowind[rptr] << ","
					<< crs_nzval[rptr] << ") ";
		outfile << "\n";
	}

	outfile << "\n\n\n";

	outfile << "=== fine to coarse vertex info ===" << endl;
	for (int_t c = 0; c < nc; ++c)
		outfile << c << " " << v_cid[c] << "\n";

	outfile << "\n\n\n";
	#endif


	delete [] v_cid;



	return 0;
}





int
coarsen_graph_v2
(
    SuperMatrix *G,
	SuperMatrix *Gc,			// output coarsened graph
	int_t		 n_crs,
	int_t		*crs_vrts
)
{
	assert(G->Stype == SLU_NC &&
		   "Coarsening only implemented for storage type SLU_NC.\n");

	cout << "Coarsening the graph..." << endl;

	NCformat	*Gstore	= (NCformat *) G->Store;
	int_t		*colptr = Gstore->colptr;
	int_t		*rowind = Gstore->rowind;
	double		*nzval	= (double *)Gstore->nzval;
	int_t		 nr		= G->nrow;
	int_t		 nc		= G->ncol;

	assert(nr == nc && "Matrix should be square.\n");

	
	// coarse graph
	Gc->Stype			 = G->Stype;
	Gc->Dtype			 = G->Dtype;
	Gc->Mtype			 = G->Mtype;
	Gc->nrow			 = n_crs;
	Gc->ncol			 = n_crs;
	Gc->Store			 = (NCformat *) SUPERLU_MALLOC(sizeof(NCformat));
	NCformat	*Gcstore = (NCformat *) Gc->Store;
	Gcstore->colptr		 = (int_t *) intMalloc_dist(n_crs+2);


	// aux
	int_t *work1 = new int_t[n_crs];
	int_t *work2 = new int_t[n_crs];

	
	// fine to coarse vertex
	int_t	*v_cid = new int_t[nr];
	int_t	 v	   = 0;
	for (int_t cv = 0; cv < n_crs; ++cv)
	{
		assert(crs_vrts[cv] == 1 || crs_vrts[cv] == 2);
		for (int_t i = 0; i < crs_vrts[cv]; ++i)
		{
			assert(v < nr);
			v_cid[v++] = cv;
		}

		work1[cv] = 0;
	}

	assert(v == nr);


	// determine xptr for the coarse graph
	Gcstore->colptr[0] = Gcstore->colptr[1] = 0;
	v				   = 0;		// current fine vertex
	for (int_t cv = 0; cv < n_crs; ++cv)
	{
		int_t n_crs_ngbrs = 0;
		for (int_t i = 0; i < crs_vrts[cv]; ++i)
		{
			for (int_t rptr = colptr[v]; rptr < colptr[v+1]; ++rptr)
			{
				int_t	r	  = rowind[rptr];
				int_t	r_crs = v_cid[r];
				assert(r_crs >= 0 && r_crs < n_crs);
				if (work1[r_crs] == 0)
				{
					work1[r_crs] = 1;
					work2[n_crs_ngbrs++] = r_crs;
				}
			}

			++v;
		}

		Gcstore->colptr[cv+2]  = Gcstore->colptr[cv+1] + n_crs_ngbrs;
		// crs_nnz				  += n_crs_ngbrs;
		for (int_t i = 0; i < n_crs_ngbrs; ++i)
		{
			work1[work2[i]] = 0;
		}
	}


	// coarse graph
	int_t	 crs_nnz   = Gcstore->colptr[n_crs+1];
	Gcstore->nnz	   = crs_nnz;
	Gcstore->nzval	   = (double *) doubleMalloc_dist(crs_nnz);
	double	*crs_nzval = (double *) Gcstore->nzval;
	Gcstore->rowind	   = (int_t *) intMalloc_dist(crs_nnz);


	cout << "#nnzs in the coarse graph " << crs_nnz << endl;


	// fill ptr and val
	v				   = 0;		// current fine vertex
	for (int_t cv = 0; cv < n_crs; ++cv)
	{
		int_t n_crs_ngbrs = 0;
		for (int_t i = 0; i < crs_vrts[cv]; ++i)
		{
			for (int_t rptr = colptr[v]; rptr < colptr[v+1]; ++rptr)
			{
				int_t	r	  = rowind[rptr];
				int_t	r_crs = v_cid[r];
				assert(r_crs >= 0 && r_crs < n_crs);
				if (work1[r_crs] == 0)
				{
					work1[r_crs]						   = 1;
					work2[n_crs_ngbrs++]				   = r_crs;
					Gcstore->rowind[Gcstore->colptr[cv+1]] = r_crs;
					crs_nzval[Gcstore->colptr[cv+1]]	   = nzval[rptr];
					++(Gcstore->colptr[cv+1]);
				}
			}

			++v;
		}

		for (int_t i = 0; i < n_crs_ngbrs; ++i)
		{
			work1[work2[i]] = 0;
		}
	}


	#ifdef DBG_MATCHING
	ofstream outfile("debug-output", std::ios_base::app);
	outfile << "coarse graph #rows/cols " << n_crs
			<< " #nnz " << crs_nnz << endl;
	outfile << "=== Coarse matrix ===\n";
	for (int_t c = 0; c < n_crs; ++c)
	{
		outfile << c << ": ";
		for (int_t rptr = Gcstore->colptr[c]; rptr < Gcstore->colptr[c+1];
			 ++rptr)
			outfile << "(" << Gcstore->rowind[rptr] << ","
					<< crs_nzval[rptr] << ") ";
		outfile << "\n";
	}

	outfile << "\n\n\n";

	outfile << "=== fine to coarse vertex info ===" << endl;
	for (int_t c = 0; c < nc; ++c)
		outfile << c << " " << v_cid[c] << "\n";

	outfile << "\n\n\n";
	#endif


	delete [] v_cid;
	delete [] work1;
	delete [] work2;



	return 0;	
}





// @ASK Must the row indices be sorted? - OK NO
void
apply_perm_sym
(
    int		 n,
	int_t	 nnz,
	int_t	*colptr,
	int_t	*adjncy,
	double	*nzval,
	int_t   *p					// permutation
)
{
	#ifdef DBG_MATCHING
	static int x = 0;
	ofstream outfile("debug-output", std::ios_base::app);
	outfile << "=== Permutation === " << x << "\n";
	for (int_t c = 0; c < n; ++c)
		outfile << c << " " << p[c] << "\n";

	outfile << "\n\n\n";
	#endif


	// Permuted matrix data.
	int_t	*colptr_p = (int_t *) intMalloc_dist(n+1);
	int_t	*adjncy_p = (int_t *) intMalloc_dist(nnz);
	double	*nzval_p  = (double *) doubleMalloc_dist(nnz);

	// Compute permuted colptr.
	colptr_p[0] = 0;
	for (int_t c = 0; c < n; ++c)
	    colptr_p[p[c]+1] = colptr[c+1] - colptr[c]; // column count

	for (int_t c = 0; c < n; ++c)  // prefix sum
		colptr_p[c+1] += colptr_p[c];

	// Permute the matrix.
	for (int_t c = 0; c < n; ++c)
	{
		int_t tmp = colptr_p[p[c]];
		for (int_t rptr = colptr[c]; rptr < colptr[c+1]; ++rptr, ++tmp)
		{
			adjncy_p[tmp] = p[adjncy[rptr]];
			nzval_p[tmp]  = nzval[rptr];
		}
	}


	memcpy(colptr, colptr_p, sizeof(*colptr_p) * (n+1));
	memcpy(adjncy, adjncy_p, sizeof(*adjncy_p) * nnz);
	memcpy(nzval, nzval_p, sizeof(*nzval_p) * nnz);


	#ifdef DBG_MATCHING
	outfile << "=== Permuted matrix === " << x << "\n";
	for (int_t c = 0; c < n; ++c)
	{
		outfile << c << ": ";
		for (int_t rptr = colptr[c]; rptr < colptr[c+1]; ++rptr)
			outfile << "(" << adjncy[rptr] << ","
					<< nzval[rptr] << ") ";
		outfile << "\n";
	}

	outfile << "\n\n\n";
	++x;
	#endif


	SUPERLU_FREE(colptr_p);
	SUPERLU_FREE(adjncy_p);
	SUPERLU_FREE(nzval_p);

	return;
}





int
is_symmetric
(
 	int		 n,
	int_t	 nnz,
	int_t	*colptr,
	int_t	*adjncy,
	double	*nzval
)
{
	#ifdef DBG_MATCHING
	ofstream outfile("debug-output", std::ios_base::app);
	outfile << "Checking whether the matrix is symmetric (pattern) ...\n";
	#endif

	int is_symmetric = 1;
	for (int_t c = 0; c < n; ++c)
	{
		for (int_t rptr = colptr[c]; rptr < colptr[c+1]; ++rptr)
		{
			int_t r = adjncy[rptr];
			if (r == c)
				continue;

			int found = 0;
			for (int_t rptr2 = colptr[r]; rptr2 < colptr[r+1]; ++rptr2)
			{
				if (adjncy[rptr2] == c)
				{
					found = 1;
					break;
				}
			}

			is_symmetric = is_symmetric && found;

			if (!found)
			{
				#ifdef DBG_MATCHING
				outfile << "Symmetricity breaks for ("
						<< r << "," << c << "\n";
				#endif
			}
		}
	}

	#ifdef DBG_MATCHING
	outfile << "Result " << is_symmetric << "\n";
	outfile << "\n\n\n";
	#endif


	return is_symmetric;
}





static
int
walk_postorder
(
 	vector<vector<int>> &ch,
 	int 				 v
)
{
	if (ch[v].empty())			// leaf
		return 1;

	int max_ch = -1, res = -1;
	for (auto u : ch[v])
	{
		res = res && walk_postorder(ch, u);
		max_ch = max(max_ch, u);
	}

	assert(max_ch != -1 && res != -1);

	if (v == max_ch+1)
		return res && 1;
	else
		cout << "post-order walk breaks at subtree rooted at " << v << endl;

	return 0;
}





int
is_postorder
(
 	int	 n,
 	int *parents
)
{
	int root = -1;

	// Need children information for each node.
	vector<vector<int>> ch(n);
	for (int v = 0; v < n; ++v)
	{
		if (parents[v] == n)
			root = v;
		else
			ch[parents[v]].push_back(v);
	}


	#ifdef DBG_MATCHING
	ofstream outfile("debug-output", std::ios_base::app);
	outfile << "=== tree post-order ===\n";
	outfile << "root " << root << "\n";
	for (int v = 0; v < n; ++v)
	{
		outfile << v << ": ";
		for (auto u : ch[v])
			outfile << u << " ";
		outfile << "\n";
	}
	outfile.close();
	#endif

	assert(root != -1 &&
		   "Could not find root in the given tree!\n");

	int res = walk_postorder(ch, root);
	cout << "is tree post-order " << res << endl;


	return res;
}
