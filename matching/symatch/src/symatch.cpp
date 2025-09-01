#include <cassert>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "gutil.hpp"
#include "util.hpp"

#include "gmodel.hpp"
#include "tdef.hpp"
#include "wrmatch.hpp"

using std::cout;	using std::endl;	using std::string;	using std::vector;
using std::tuple;	using std::cerr;




int
main
(
    int		  argc,
	char	**argv
)
{
	params_t params;
	params.mmfile = string(argv[1]);

	// read and convert 
	MatMarket_t<VIDX_T, EW_T> mm;
	mmread(params.mmfile, &mm);

	// GModel *gm = new StandardSym(&mm);
	// GModel *gm = new RepGrDiagCon(&mm);
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

	// g->printx(true);

	WrMatch *wrm_gpa = new WgtGPASeq(g);
	wrm_gpa->match();
	printf("matching cost (GPA) sum %lf prod %lf\n",
		   wrm_gpa->cost(), wrm_gpa->cost_prod());
	// vector<tuple<VIDX_T, VIDX_T, EW_T>> m_gpa_sorted;
	// wrm_gpa->sort_matching(m_gpa_sorted, false);
	// cout << "sorted matching" << endl;
	// for (auto &t : m_gpa_sorted)
	// 	cout << get<0>(t) << " " << get<1>(t) << " " << get<2>(t) << endl;

	// return 0;

	cout << string(80, '-') << endl;

	WrMatch *wrm = new WgtSuitorSeqFor(g);
	wrm->match();
	printf("matching cost (Suitor) sum %lf prod %lf\n",
		   wrm->cost(), wrm->cost_prod());
	// vector<tuple<VIDX_T, VIDX_T, EW_T>> m_sorted;	
	// wrm->sort_matching(m_sorted, true);	
	// wrm->sort_matching(m_sorted, false);
	// cout << "sorted matching" << endl;
	// for (auto &t : m_sorted)
	// 	cout << get<0>(t) << " " << get<1>(t) << " " << get<2>(t) << endl;

	// below here needs to be changed when superlu is integrated

	// @OGUZ-TODO when superlu is integrated, change this
	// the indices of a pair of matched vertices are ordered by their indices
	// perm[v] = x -> new idx of v is x
	// vector<VIDX_T> perm(g->nv);	// 0-based
	// vector<VIDX_T> perm(mm.nr);	// 0-based
	// gm->permute(m_sorted, perm);

	
	// cout << "perm" << endl;
	// for (VIDX_T v = 0; v < mm.nr; ++v)
	// 	cout << v << " " << perm[v] << endl;

	// exit(0);

	// string outfile = params.mmfile.substr(0, params.mmfile.size()-4) +
	// 	"-stdsymdiag-perm.mtx";
	// mmwrite(mm, outfile, &perm);


	// exit(0);
	

	// VIDX_T card = wrm->cardinality();
	// cout << "matching cardinality " << card
	// 	 << " perc (%) " << (static_cast<double>(card) / g->nv * 100.0)
	// 	 << endl;

	// uint64_t card = gm->cardinality(m_sorted);
	// cout << "matching cardinality " << card
	// 	 << " perc (%) " << (static_cast<double>(card) / mm.nr * 100.0)
	// 	 << endl;
	

	delete wrm_gpa;
	delete wrm;
	delete gm;
	


	return (EXIT_SUCCESS);
}

