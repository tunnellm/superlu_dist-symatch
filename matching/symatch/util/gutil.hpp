#ifndef _GUTIL_HPP_
#define _GUTIL_HPP_

#include <algorithm>
#include <numeric>
#include <typeinfo>

#include "bigraph.hpp"
#include "graph.hpp"
#include "types.hpp"

using std::cout;	using std::cerr;	using std::endl;	using std::sort;
using std::iota;




namespace
SyMatch
{

// if type of graph is bipartite, the graph will be an undirected bipartite
//   graph whether it is square (symmetric or general) or rectangular
// otherwise
//   mm is symmetric -> undirected graph
//   mm is square general -> directed graph
//   mm is rectangular -> undirected bipartite graph (g must be bipartite)

// |---------------------------------|
// |               input graph type	 |
// |mat-type |    regular  bipartite |
// |---------------------------------|
// |     sym | undirected undirected |
// |  sq-gen |   directed undirected |
// |    rect |        N/A undirected |
// |---------------------------------|
template <typename IDX_T,		// mm
		  typename VAL_T,
		  typename VIDX_T,		// graph
		  typename EW_T>
void
mm_to_gr
(
    const MatMarket_t<IDX_T, VAL_T> &mm,
    Graph<VIDX_T, EW_T>				*g
)
{
	// vertex and matrix market elem indices must match
	using std::is_same;
	static_assert(is_same<VIDX_T, IDX_T>::value == true);

	// if mm is rectangular, g must be bipartite
	if ((mm.nr != mm.nc) &&
		!(dynamic_cast<BiGraph<VIDX_T, EW_T> *>(g)))
	{
		cerr << "FATAL: Cannot create regular graph from rectangular matrix. "
			 << "The graph must be bipartite."
			 << endl;
		exit(1);
	}

	bool					 is_bipartite = false;
	BiGraph<VIDX_T, EW_T>	*bg			  = NULL;
	if (dynamic_cast<BiGraph<VIDX_T, EW_T> *>(g))
	{
		is_bipartite = true;
		bg			 = dynamic_cast<BiGraph<VIDX_T, EW_T> *>(g);
	}
	
	g->gt = Graph<VIDX_T, EW_T>::Directed;
	if (is_bipartite ||
		mm.storage == MatMarket_t<IDX_T, VAL_T>::Symmetric)
		g->gt = Graph<VIDX_T, EW_T>::Undirected;
	bool	is_directed = (g->gt == Graph<VIDX_T, EW_T>::Directed);
	bool	is_sym		= (mm.storage == MatMarket_t<IDX_T, VAL_T>::Symmetric);
	
	g->wgtd_edges = mm.vals_exist;

	cout << "creating graph from Matrix Market data" << endl;
	cout << "graph is ";
	if (is_bipartite)
		cout << "bipartite" << endl;
	else
		cout << "regular" << endl;	


	// number of vertices
	g->nv = mm.nr;
	if (is_bipartite)
	{
		g->nv	+= mm.nc;
		bg->nvl	 = mm.nr;
		bg->nvr  = mm.nc;
	}

	// find size of adj lists
	g->xadj.resize(g->nv+2, static_cast<uint64_t>(0));
	for (uint64_t el = 0; el < mm.nelems; ++el)
	{
		IDX_T r = mm.rids[el];
		IDX_T c = mm.cids[el];

		++(g->xadj[r+2]);
		if (is_sym && (r != c)) // undirected
			++(g->xadj[c+2]);

		if (is_bipartite)		// bipartite & undirected
		{
			VIDX_T rbeg = bg->nvl;
			++(g->xadj[rbeg+c+2]);
			if (is_sym && (r != c))
				++(g->xadj[rbeg+r+2]);
		}
	}

	// prefix sum to get beg/end pointers
	for (VIDX_T i = 2; i < g->nv+2; ++i)
		g->xadj[i] += g->xadj[i-1];

	// allocate
	g->adj.resize(g->xadj[g->nv+1]);
	if (g->wgtd_edges)
		g->ew.resize(g->xadj[g->nv+1]);

	// fill adj and ew, finalize xadj
	for (uint64_t el = 0; el < mm.nelems; ++el)
	{
		IDX_T	r = mm.rids[el];
		IDX_T	c = mm.cids[el];
		EW_T	val;
		if (g->wgtd_edges)
			val = static_cast<EW_T>(mm.vals[el]); // VAL_T -> EW_T

		VIDX_T rbeg = 0;		// in case bipartite
		if (is_bipartite)
			rbeg = bg->nvl;

		g->adj[g->xadj[r+1]] = c + rbeg;
		if (g->wgtd_edges)
			g->ew[g->xadj[r+1]] = val;
		++(g->xadj[r+1]);

		if (is_sym && (r != c))
		{
			g->adj[g->xadj[c+1]] = r + rbeg;
			if (g->wgtd_edges)
				g->ew[g->xadj[c+1]] = val;
			++(g->xadj[c+1]);
		}

		if (is_bipartite)		// bipartite & undirected
		{
			g->adj[g->xadj[c+rbeg+1]] = r;
			if (g->wgtd_edges)
				g->ew[g->xadj[c+rbeg+1]] = val;
			++(g->xadj[c+rbeg+1]);

			if (is_sym && (r != c))
			{
				g->adj[g->xadj[r+rbeg+1]] = c;
				if (g->wgtd_edges)
					g->ew[g->xadj[r+rbeg+1]] = val;
				++(g->xadj[r+rbeg+1]);
			}
		}		
	}
	

	// #edges
	if (!is_bipartite && is_directed) // each edge appears only once in adj
		g->nedges = g->xadj[g->nv];
	else if (is_bipartite && !is_directed) // each edge appears twice
										   // and there are no self-loops
		g->nedges = g->xadj[g->nv]/2;
	else if (!is_bipartite && !is_directed) // self-loops appear once,
											// others appear twice
	{
		uint64_t nself_loops = 0;
		g->nedges = 0;
		for (VIDX_T v = 0; v < g->nv; ++v)
		{
			for (uint64_t eptr = g->xadj[v]; eptr < g->xadj[v+1]; ++eptr)
			{
				VIDX_T u = g->adj[eptr];
				if (u == v)
					++nself_loops;
				else
					++(g->nedges);
			}
		}

		g->nedges /= 2;
		g->nedges += nself_loops;
	}


	cout << "#vertices " << g->nv
		 << " #edges " << g->nedges
		 << endl;


	// handle sorted adj
	if (g->adj_sorted)
		g->sort_adj();

	
	return;
}

}

#endif
