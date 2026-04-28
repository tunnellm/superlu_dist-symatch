#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dldperm_dist_symatch.hpp"

using std::cout;	using std::endl;	using std::string;	using std::cerr;
using std::vector;	using std::tuple;	using std::unordered_map;
using std::hash;	using std::pair;	using std::unordered_set;
using std::max;	using std::min;	using std::abs;

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

// temporary debugging - delete after ensuring
SyMatch::Graph<int_t, double> *g_dbg;
SyMatch::Graph<int_t, double> *g_dbg_v1;


struct
timer
{
	std::string								name;
	int64_t									val = 0;
	std::chrono::steady_clock::time_point	tp;

	
	//--------------------------------------------------------------------------


	timer ()
	{
	}



	
	timer (const std::string &s) :
		name(s)
	{
	}

	


	void
	start_timer ()
	{
		tp = std::chrono::steady_clock::now();
	}




	void
	reset_timer ()
	{
		val = 0;
	}




	void
	stop_timer ()
	{
		val += std::chrono::duration_cast<std::chrono::microseconds>
			(std::chrono::steady_clock::now() - tp).count();
	}

	


	std::string
	getstr ()
	{
		std::stringstream ss;
		ss << name << " "
		   << std::fixed << std::setprecision(3)
		   << static_cast<double>(val) / 1e+3
		   << " msec"; 

		return ss.str();
	}
};




struct
pair_hash
{
	std::size_t
	operator () (const pair<int_t, int_t> &arg) const
	{
		return hash<int_t>{}(arg.first) ^ hash<int_t>{}(arg.second);
	}
};






int
dldperm_dist_symatch_v1
(
    int			  job,
	int			  n,
	int_t		  nnz,
	int_t		  colptr[],
	int_t		  adjncy[],
	double		  nzval[],
	int_t		 *perm,
	crs_info_t	 *crs_info
)
{
	timer tmr_sym_all("sym-all");
	timer tmr_gm_form("gm-form");
	timer tmr_match("match");
	timer tmr_perm("perm");
	timer tmr_crsinf("crs-info");
	

	cout << string(80, '=') << endl;

	tmr_sym_all.start_timer();
	tmr_gm_form.start_timer();

	// form directly from sparse matrix
	SyMatch::GModel *gm =
		new SyMatch::StandardSymDiag(n, colptr, adjncy, nzval);
	gm->form_graph(true);
	auto *g = gm->g;
	// g_dbg_v1 = gm->g;				// @DELETE

	tmr_gm_form.stop_timer();

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

	tmr_match.start_timer();

	// WrMatch *wrm = new WgtGPASeq(g);
	SyMatch::WrMatch *wrm = new SyMatch::WgtSuitorSeqFor(g);
	wrm->match();

	tmr_match.stop_timer();

	// fprintf(stdout, "matching cost %lf\n", wrm->cost());
	// fprintf(stdout, "matching cost (prod) %lf\n", wrm->cost_prod());

	// ofstream out_graph;
	// out_graph.open("graph");
	// out_graph << g->nv << " " << g->xadj[g->nv]/2 << "\n";
	// for (int64_t v = 0; v < g->nv; ++v)
	// {
	// 	for (int_t i = g->xadj[v]; i < g->xadj[v+1]; ++i)
	// 	{
	// 		if (v < g->adj[i])
	// 			out_graph << v << " " << g->adj[i] << " " << g->ew[i] << "\n";
	// 	}
	// }
	// out_graph.close();
	// exit(37);


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

	tmr_perm.start_timer();

	// Compute the permutation
	int32_t *m		= wrm->p;	// raw match	
	int_t	 curidx = 0;
	crs_info->n_crs	= 0;
	// fine to crs mapping (for after Pr is applied)
	crs_info->ftoc	= (int_t *) malloc(sizeof(*(crs_info->ftoc)) * n);
	for (int_t v = 1; v <= n; ++v)
	{
		int_t u = m[v];
		
		#ifdef DBG_MATCHING
		outfile << v << " " << u << "\n";
		#endif

		if (u == 0 || u > n) // singleton
		{
			crs_info->ftoc[curidx] = crs_info->n_crs;
			++(crs_info->n_crs);
			perm[v-1] = curidx++;			
		}
		else if (v < u)
		{
			crs_info->ftoc[curidx]	 = crs_info->n_crs;
			crs_info->ftoc[curidx+1] = crs_info->n_crs;
			++(crs_info->n_crs);
			perm[v-1] = curidx++;
			perm[u-1] = curidx++;
		}
	}

	tmr_perm.stop_timer();

	#ifdef DBG_MATCHING
	outfile << "\n\n\n";

	outfile << "=== Permutation (perm_r) ===\n";
	for (int_t v = 0; v < mm.nr; ++v)
		outfile << v << " -> " << perm[v] << "\n";

	outfile << "\n\n\n";
	#endif


	tmr_crsinf.start_timer();

	// 2nd pass - coarsening information.
	if (crs_info->n_crs > 0)
		crs_info->crs_vrts =
			(int_t *)malloc(sizeof(*(crs_info->crs_vrts)) * (crs_info->n_crs));	

	int_t crs_idx = 0;
	for (int_t v = 1; v <= n; ++v)
	{
		int_t u = m[v];

		if (u == 0 || u > n) // singleton
			(crs_info->crs_vrts)[crs_idx++] = 1;
		else if (v < u)
			(crs_info->crs_vrts)[crs_idx++] = 2;
	}

	tmr_crsinf.stop_timer();
	tmr_sym_all.stop_timer();


	#ifdef DBG_MATCHING
	outfile << "=== Coarsening info ===\n";
	outfile << "n_crs " << *n_crs << "\n";

	for (int_t i = 0; i < *n_crs; ++i)
		outfile << i << " " << (*crs_vrts)[i] << "\n";

	outfile << "\n\n\n";
	outfile.close();
	#endif

	
	cout << "#2x2 " << n-(crs_info->n_crs)
		 << " #1x1 " << 2*(crs_info->n_crs)-n << "\n";
	cout << "Number of coarse vertices " << crs_info->n_crs << "\n";
	fprintf(stdout, "matching cost %lf\n", wrm->cost());
	cout << string(80, '=') << endl;
	

	delete gm;
	delete wrm;


	cout << "time:\n"
		 << "  " << tmr_gm_form.getstr() << "\n"
		 << "  " << tmr_match.getstr() << "\n"
		 << "  " << tmr_perm.getstr() << "\n"
		 << "  " << tmr_crsinf.getstr() << "\n"
		 << "  " << tmr_sym_all.getstr()
		 << endl;



	return 0;
}





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
	int_t	 *n_crs,
	int_t	**crs_vrts
)
{
	timer tmr_sym_all("sym-all");
	timer tmr_csc_mm("csc->mm");
	timer tmr_gm_form("gm-form");
	timer tmr_match("match");
	timer tmr_perm("perm");
	timer tmr_crsinf("crs-info");

	
	cout << string(80, '=') << endl;

	tmr_sym_all.start_timer();
	tmr_csc_mm.start_timer();

	// Create a matrix market object from CSC.
	// Unfilled members: svals, header.
	SyMatch::MatMarket_t<int_t, double>	mm;
	mm.fmt		  = SyMatch::MatMarket_t<int_t, double>::Coordinate;
	mm.type		  = SyMatch::MatMarket_t<int_t, double>::Real;
	mm.storage	  = SyMatch::MatMarket_t<int_t, double>::Symmetric;	// assumed
	mm.nr		  = mm.nc = n;
	mm.nnzs		  = nnz;
	mm.vals_exist = true;

	uint64_t ndiags = 0, ndiags0 = 0;
	mm.nelems = 0;
	for (int_t c = 0; c < n; ++c)
	{
		for (int_t rptr = colptr[c]; rptr < colptr[c+1]; ++rptr)
		{
			int_t	r = adjncy[rptr];
			double	v = nzval[rptr];

			if (r == c)
			{
				++ndiags;
				if (v == 0.0)
					++ndiags0;
			}

			if (r <= c && v != 0.0)			// U
			{
				mm.rids.push_back(r);
				mm.cids.push_back(c);
				mm.vals.push_back(v);
				++(mm.nelems);
			}
		}
	}

	tmr_csc_mm.stop_timer();

	assert(((nnz - ndiags) & 0x1) == 0x0);
	cout << "#nnzs " << mm.nnzs << " #diags " << ndiags
		 << " #diags0 " << ndiags0 << endl;

	tmr_gm_form.start_timer();

	// Form the graph model for matching.
	SyMatch::GModel *gm = new SyMatch::StandardSymDiag(&mm);
	gm->form_graph(true);
	auto *g = gm->g;
	// g_dbg = gm->g;				// @DELETE

	tmr_gm_form.stop_timer();

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

	tmr_match.start_timer();

	// WrMatch *wrm = new WgtGPASeq(g);
	SyMatch::WrMatch *wrm = new SyMatch::WgtSuitorSeqFor(g);
	wrm->match();

	tmr_match.stop_timer();

	// fprintf(stdout, "matching cost (prod) %lf\n", wrm->cost_prod());

	// ofstream out_graph;
	// out_graph.open("graph");
	// out_graph << g->nv << " " << g->xadj[g->nv]/2 << "\n";
	// for (int64_t v = 0; v < g->nv; ++v)
	// {
	// 	for (int_t i = g->xadj[v]; i < g->xadj[v+1]; ++i)
	// 	{
	// 		if (v < g->adj[i])
	// 			out_graph << v << " " << g->adj[i] << " " << g->ew[i] << "\n";
	// 	}
	// }
	// out_graph.close();
	// exit(37);


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

	tmr_perm.start_timer();

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

	tmr_perm.stop_timer();

	#ifdef DBG_MATCHING
	outfile << "\n\n\n";

	outfile << "=== Permutation (perm_r) ===\n";
	for (int_t v = 0; v < mm.nr; ++v)
		outfile << v << " -> " << perm[v] << "\n";

	outfile << "\n\n\n";
	#endif


	tmr_crsinf.start_timer();

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

	tmr_crsinf.stop_timer();
	tmr_sym_all.stop_timer();


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
	fprintf(stdout, "matching cost %lf\n", wrm->cost());
	cout << string(80, '=') << endl;
	

	delete gm;
	delete wrm;


	cout << "time:\n"
		 << "  " << tmr_csc_mm.getstr() << "\n"
		 << "  " << tmr_gm_form.getstr() << "\n"
		 << "  " << tmr_match.getstr() << "\n"
		 << "  " << tmr_perm.getstr() << "\n"
		 << "  " << tmr_crsinf.getstr() << "\n"
		 << "  " << tmr_sym_all.getstr()
		 << endl;



	return 0;
}





// GPU version, SUMAC
// int
// dldperm_dist_symatch_g
// (
//     int			  job,
// 	int			  n,
// 	int_t		  nnz,
// 	int_t		  colptr[],
// 	int_t		  adjncy[],
// 	double		  nzval[],
// 	int_t		 *perm,
// 	crs_info_t	 *crs_info
// )
// {
	


// 	return 0;
// }





int
coarsen_graph
(
    SuperMatrix *G,
	SuperMatrix *Gc,			// output coarsened graph
	int_t		 n_crs,
	int_t		*crs_vrts
)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	assert(G->Stype == SLU_NC &&
		   "Coarsening only implemented for storage type SLU_NC.\n");

	if (rank == 0)
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

	if (rank == 0)
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
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	assert(G->Stype == SLU_NC &&
		   "Coarsening only implemented for storage type SLU_NC.\n");

	if (rank == 0)
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


	if (rank == 0)
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





int
coarsen_graph_v3
(
    SuperMatrix *G,
	SuperMatrix *Gc,			// output coarsened graph
	crs_info_t	*crs_info
)
{
	// fine graph
	NCformat	*Gstore	= (NCformat *) G->Store;
	int_t		*colptr = Gstore->colptr;
	int_t		*rowind = Gstore->rowind;
	double		*nzval	= (double *)Gstore->nzval;
	int_t		 nr		= G->nrow;
	int_t		 nc		= G->ncol;
	int_t		 nnz    = Gstore->nnz;

	assert(nr == nc && "Matrix should be square.\n");

	int_t n_crs = crs_info->n_crs;
	int_t *ftoc = crs_info->ftoc;

	// coarse graph
	Gc->Stype			   = G->Stype;
	Gc->Dtype			   = G->Dtype;
	Gc->Mtype			   = G->Mtype;
	Gc->nrow			   = n_crs;
	Gc->ncol			   = n_crs;
	Gc->Store			   = (NCformat *) SUPERLU_MALLOC(sizeof(NCformat));
	NCformat	*Gcstore   = (NCformat *) Gc->Store;
	Gcstore->colptr		   = (int_t *) intMalloc_dist(n_crs+1);
	Gcstore->nzval		   = (double *) doubleMalloc_dist(nnz);
	Gcstore->rowind		   = (int_t *) intMalloc_dist(nnz);
	
	// tables for forming coarse adj lists
	int_t mask = (1<<13) - 1;
	int_t *ht, *dt;
	ht = (int_t *)malloc(sizeof(*ht) * (mask+1));
	dt = (int_t *)malloc(sizeof(*dt) * (n_crs));
	for (int_t i = 0; i < mask+1; ++i)
		ht[i] = -1;
	for (int_t i = 0; i < n_crs; ++i)
		dt[i] = -1;

	int_t	 c_nnz	  = 0;
	int_t	 c_n	  = 0;
	int_t	*c_colptr = Gcstore->colptr;
	int_t	*c_rowind = Gcstore->rowind;
	double	*c_nzval  = (double *) Gcstore->nzval;
	c_colptr[0]		  = 0;

	int_t	v = 0, u;			// fine vertices
	int_t	t;
	int_t	hash, cadjlen, hval;
	int_t	rbeg, rend;
	for (int_t cv = 0; cv < n_crs; ++cv)
	{
		cadjlen		   = 0;
		u			   = -1;
		int_t	adjlen = colptr[v+1] - colptr[v];
		if (crs_info->crs_vrts[cv] == 2)
		{
			u		= v+1;
			adjlen += colptr[u+1] - colptr[u];
		}
		

		if (adjlen < (mask>>2))	// ht
		{	
			rbeg = colptr[v];
			rend = colptr[v+1];
			for (int_t i = rbeg; i < rend; ++i)
			{
				t = ftoc[rowind[i]]; // coarse neighbor
				for (hash = t & mask;
					 ht[hash] != -1 && c_rowind[ht[hash]] != t;
					 hash = (hash+1) & mask);

				if ((hval = ht[hash]) == -1)
				{
					c_rowind[cadjlen] = t;
					// c_nzval[cadjlen]  = nzval[i];
					ht[hash]		  = cadjlen++;
				}
				// else
				// {
				// 	c_nzval[hval] += nzval[i];
				// }
			}


			if (u != -1)
			{
				rbeg = colptr[u];
				rend = colptr[u+1];
				for (int_t i = rbeg; i < rend; ++i)
				{
					t = ftoc[rowind[i]]; // coarse neighbor
					for (hash = t & mask;
						 ht[hash] != -1 && c_rowind[ht[hash]] != t;
						 hash = (hash+1) & mask);

					if ((hval = ht[hash]) == -1)
					{
						c_rowind[cadjlen] = t;
						// c_nzval[cadjlen]  = nzval[i];
						ht[hash]		  = cadjlen++;
					}
					// else
					// {
					// 	c_nzval[hval] += nzval[i];
					// }
				}
			}

			// clear table in reverse order
			// @TODO record used hash slots into a list to clear without hashing
			for (int_t i = cadjlen-1; i >= 0; --i)
			{
				t = c_rowind[i];
				for (hash = t & mask;
					 c_rowind[ht[hash]] != t;
					 hash = (hash+1) & mask);
				ht[hash] = -1;
			}
		}
		else					// dt
		{
			rbeg = colptr[v];
			rend = colptr[v+1];
			for (int_t i = rbeg; i < rend; ++i)
			{
				t = ftoc[rowind[i]];
				if ((hval = dt[t]) == -1)
				{
					c_rowind[cadjlen] = t;
					// c_nzval[cadjlen]  = nzval[i];
					dt[t]			  = cadjlen++;
				}
				// else
				// {
				// 	c_nzval[hval] += nzval[i];
				// }
			}

			if (u != -1)
			{
				rbeg = colptr[u];
				rend = colptr[u+1];

				for (int_t i = rbeg; i < rend; ++i)
				{
					t = ftoc[rowind[i]];
					if ((hval = dt[t]) == -1)
					{
						c_rowind[cadjlen] = t;
						// c_nzval[cadjlen]  = nzval[i];
						dt[t]			  = cadjlen++;
					}
					// else
					// {
					// 	c_nzval[hval] += nzval[i];
					// }
				}
			}

			// clear table
			for (int_t i = 0; i < cadjlen; ++i)
				dt[c_rowind[i]] = -1;
		}


		v			   += crs_info->crs_vrts[cv];
		c_rowind	   += cadjlen;
		// c_nzval		   += cadjlen;
		c_nnz		   += cadjlen;
		c_colptr[cv+1]	= c_nnz;		
	}


	Gcstore->nnz = c_nnz;

	cout << "#nnzs in the coarse graph " << c_nnz << endl;



	free(ht);
	free(dt);

	return 0;
}





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





// similar speed with apply_perm_sym
void
apply_perm_sym_v2
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
	int_t	*pinv	  = (int_t *) intMalloc_dist(n);

	// Compute permuted colptr.
	colptr_p[0] = 0;
	for (int_t c = 0; c < n; ++c)
	{
	    // colptr_p[p[c]+1] = colptr[c+1] - colptr[c]; // column count
		pinv[p[c]] = c;
	}

	// for (int_t c = 0; c < n; ++c)  // prefix sum
	// 	colptr_p[c+1] += colptr_p[c];

	// Permute the matrix.
	for (int_t c = 0; c < n; ++c)
	{
		colptr_p[c+1] = colptr_p[c] + (colptr[pinv[c]+1] - colptr[pinv[c]]);		
		int_t tmp = colptr[pinv[c]];
		for (int_t rptr = colptr_p[c]; rptr < colptr_p[c+1]; ++rptr, ++tmp)
		{
			adjncy_p[rptr] = p[adjncy[tmp]];
			nzval_p[rptr]  = nzval[tmp];
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
	SUPERLU_FREE(pinv);

	return;
}





void
apply_perm_sym_pattern
(
    int		 n,
	int_t	 nnz,
	int_t	*colptr,
	int_t	*adjncy,
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
			adjncy_p[tmp] = p[adjncy[rptr]];
	}


	memcpy(colptr, colptr_p, sizeof(*colptr_p) * (n+1));
	memcpy(adjncy, adjncy_p, sizeof(*adjncy_p) * nnz);
	

	#ifdef DBG_MATCHING
	outfile << "=== Permuted matrix === " << x << "\n";
	for (int_t c = 0; c < n; ++c)
	{
		outfile << c << ": ";
		for (int_t rptr = colptr[c]; rptr < colptr[c+1]; ++rptr)
			outfile << "(" << adjncy[rptr] << ") ";
		outfile << "\n";
	}

	outfile << "\n\n\n";
	++x;
	#endif


	SUPERLU_FREE(colptr_p);
	SUPERLU_FREE(adjncy_p);

	
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
				return 0;
			}
		}
	}

	#ifdef DBG_MATCHING
	outfile << "Result " << is_symmetric << "\n";
	outfile << "\n\n\n";
	#endif


	return is_symmetric;
}





void
is_symmetric_v2
(
 	int		 n,
	int_t	 nnz,
	int_t	*colptr,
	int_t	*adjncy,
	double	*nzval
)
{
	auto is_equal_eps =
		[] (double a, double b,
			double eps = 100 * DBL_EPSILON,
			double abs_th = DBL_MIN)
		-> bool
		{
			if (a == b)
				return true;

			auto diff = abs(a - b);
			auto norm = min((abs(a) + abs(b)),
							std::numeric_limits<double>::max());

			return diff < max(abs_th, eps * norm);
		};
	
	unordered_map<int_t, double> *S = new unordered_map<int_t, double>[n];
	bool is_sym_pattern = true;
	bool is_sym_numeric = true;
	
	for (int_t c = 0; c < n; ++c)
	{
		for (int_t rptr = colptr[c]; rptr < colptr[c+1]; ++rptr)
		{
			int_t	r = adjncy[rptr];
			double	v = nzval[rptr];
			S[c].insert({r, v});
		}
	}


	for (int_t c = 0; c < n; ++c)
	{
		for (int_t rptr = colptr[c]; rptr < colptr[c+1]; ++rptr)
		{
			int_t	r = adjncy[rptr];
			double	v = nzval[rptr];

			auto tmp = S[r].find(c);
			if (tmp == S[r].end())
			{
				is_sym_pattern = false;
				is_sym_numeric = false;
				goto L01;
			}
			else
			{
				// if (tmp->second != v)
				if (!is_equal_eps(tmp->second, v))
				{
					// cout << "(" << c << ", " << r << ") "
					// 	 << tmp->second << " " << v << endl;
					is_sym_numeric = false;
				}
			}
		}
	}

 L01:

	cout << "symmetric pattern " << is_sym_pattern << " "
		 << "numeric " << is_sym_numeric
		 << endl;

	delete [] S;


	return;
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





void
ensure_graphs ()
{
	auto is_equal_eps =
		[] (double a, double b,
			double eps = 100 * DBL_EPSILON,
			double abs_th = DBL_MIN)
		-> bool
		{
			if (a == b)
				return true;

			auto diff = abs(a - b);
			auto norm = min((abs(a) + abs(b)),
							std::numeric_limits<double>::max());

			return diff < max(abs_th, eps * norm);
		};

	
	auto *g1 = g_dbg;
	auto *g2 = g_dbg_v1;
	
	if (g1->nv != g2->nv)
	{
		cout << g1->nv << " " << g2->nv << endl;
		return;
	}

	if (g1->nedges != g2->nedges)
	{
		cout << g1->nedges << " " << g2->nedges << endl;
		return;
	}
	

	for (int_t v = 0; v < g1->nv; ++v)
	{
		unordered_map<pair<int_t, int_t>, double, pair_hash> S;
		for (int_t eptr = g1->xadj[v]; eptr < g1->xadj[v+1]; ++eptr)
		{
			int_t	u = g1->adj[eptr];
			double	w = g1->ew[eptr];
			S.insert({{v, u}, w});
		}

		if ((g1->xadj[v+1]-g1->xadj[v]) != (g2->xadj[v+1]-g2->xadj[v]))
		{
			cout << "adj list sizes differ for vertex " << v << " "
				 << (g1->xadj[v+1]-g1->xadj[v])
				 << " "
				 << (g2->xadj[v+1]-g2->xadj[v])
				 << endl;
			return;				
		}

		for (int_t eptr = g2->xadj[v]; eptr < g2->xadj[v+1]; ++eptr)
		{
			int_t	u = g2->adj[eptr];
			double	w = g2->ew[eptr];

			auto tmp = S.find({v, u});
			if (tmp == S.end())
			{
				cout << "(" << v << ", " << u << ") does not exist in g1."
					 << endl;
				return;
			}

			// if (tmp->second != w)
			if (!is_equal_eps(tmp->second, w))
			{
				cout << "weights differ for edge (" << v << ", " << u << ") "
					 << tmp->second << " " << w
					 << endl;
				return;
			}

			S.erase(tmp);
		}

		if (!S.empty())
		{
			cout << "S not empty, vertex " << v
				 << " size after comparison " << S.size() << endl;
			return;
		}
	}
	

	cout << "graphs are the same" << endl;
	


	return;
}
