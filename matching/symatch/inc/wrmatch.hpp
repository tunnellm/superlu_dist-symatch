/**
 * @file
 *	wrmatch.hpp
 *
 * @author
 *	Oguz Selvitopi
 *
 * @date
 *	None
 *
 * @brief
 *	Wraps the algorithms in the matching lib to provide a simpler interface
 *
 * @todo
 *
 * @note
 *  The indices and edge weights are hard-coded (int and double) in the library.
 *  The matching library uses 1-based indexing.
 */

#ifndef _WRMATCH_HPP_
#define _WRMATCH_HPP_

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include "wframe2-inc.h"		// matching library

#include "gutil.hpp"

using std::cout;	using std::vector;	using std::tuple;	using std::get;
using std::make_tuple;	using std::sort;	using std::count_if;
using std::swap;

typedef int		WM_VIDX_T;
typedef double	WM_EW_T;



// forward declarations for the matching library
#ifdef __cplusplus
extern "C" {
#endif

// file: sweight.c
void sweight(int n, int *ver, int *edges, int *s, double *ws, double *weight,
			 int *init);

// other util
double cost_matching(int n, int *ver, int *edges, double *weight, int *match);

double cost_matching_prod(int n, int *ver, int *edges, double *weight,
						  int *match);

#ifdef __cplusplus
}
#endif





// void sweight(int n,int *ver,int *edges,int *s,double *ws,double *weight,int *init)


class
WrMatch
{

public:

	WrMatch (Graph<WM_VIDX_T, WM_EW_T> *g)
	{
		n = g->nv;

		ver = (WM_VIDX_T *) malloc(sizeof(*ver) * (n+2));
		for (WM_VIDX_T v = 0; v < n+1; ++v)
			ver[v+1] = (WM_VIDX_T)(g->xadj[v]);	// uint64_t -> WM_VIDX_T

		int xadj_len = g->xadj[g->nv];
		edges  = (WM_VIDX_T *) malloc(sizeof(*edges) * xadj_len);
		weight = (WM_EW_T *) malloc(sizeof(*weight) * xadj_len);

		for (WM_VIDX_T e = 0; e < xadj_len; ++e)
		{
			edges[e]  = g->adj[e] + 1; // 1-indexing
			weight[e] = fabs(g->ew[e]); // magnitude
		}

		p = (WM_VIDX_T *) malloc(sizeof(*p) * (n+2));
	}




	virtual void match () = 0;




	double
	cost ()
	{
		return cost_matching(n, ver, edges, weight, p);
	}




	double
	cost_prod ()
	{
		return cost_matching_prod(n, ver, edges, weight, p);
	}




	// puts unmatched vertices at the end if unmatched_at_beg flag is false
	void
	sort_matching
	(
	    vector<tuple<WM_VIDX_T, WM_VIDX_T, WM_EW_T> > &m_sorted,
		bool	unmatched_at_beg = false
	)
	{
		if (n == 0)
			return;
		
		vector<bool> mark(n+1, false);
		WM_VIDX_T n_unmatched = 0;
		
		for (WM_VIDX_T v = 1; v <= n; ++v)
		{
			WM_EW_T w = 0.0;
			
			if (p[v] == 0)		// unmatched vertex
			{
				++n_unmatched;
				m_sorted.push_back({v, 0, w});
				continue;
			}

			if (mark[v])
				continue;

			mark[v]	   = true;
			mark[p[v]] = true;
			
			for (WM_VIDX_T eptr = ver[v]; eptr < ver[v+1]; ++eptr)
			{
				WM_VIDX_T u = edges[eptr];
				if (u == p[v])
				{
					w = weight[eptr];
					break;
				}
			}

			m_sorted.push_back({v, p[v], w});
		}


		// gather the unmatched vertices at the beg
		if (n_unmatched > 0 && unmatched_at_beg)
		{
			WM_VIDX_T i = 0, j;

			// find one unmatched vertex and put at the beginning
			while (i < m_sorted.size() && get<1>(m_sorted[i]) != 0)
				++i;
			swap(m_sorted[0], m_sorted[i]);

			i = 1, j = 0;
			while (i < m_sorted.size())
			{
				if (get<1>(m_sorted[i]) == 0)
				{
					++j;
					swap(m_sorted[j], m_sorted[i]);
				}
				++i;
			}
		}


		auto f_comp =
			[] (const tuple<WM_VIDX_T, WM_VIDX_T, WM_EW_T> &t1,
				const tuple<WM_VIDX_T, WM_VIDX_T, WM_EW_T> &t2)
			{
				return get<2>(t1) > get<2>(t2);
			};

		WM_VIDX_T offset = 0;
		if (unmatched_at_beg)
			offset = n_unmatched;
			
		sort(m_sorted.begin()+offset, m_sorted.end(), f_comp);


		
		return;
	}




	WM_VIDX_T
	cardinality ()
	{
		return
			count_if (p+1, p+n+1,
					  [] (int v) { return v != 0; }
					  );
	}




	virtual
	~WrMatch ()
	{
		// cout << "WrMatch destructor" << endl;
		free(ver);
		free(edges);
		free(weight);
		free(p);
	}




public:

	// base data needed by all
	WM_VIDX_T	 n;				// #vertices
	WM_VIDX_T	*ver;			// xadj
	WM_VIDX_T	*edges;			// adj
	WM_EW_T		*weight;		// edge weights
	WM_VIDX_T	*p;				// matching info
};





class
WgtSuitorSeqFor : public WrMatch
{

public:

	WgtSuitorSeqFor (Graph<WM_VIDX_T, WM_EW_T> *g) :
		WrMatch(g)
	{
		p_init = (WM_VIDX_T *) malloc(sizeof(*p_init) * (n+2));
		ws	   = (WM_EW_T *) malloc(sizeof(*ws) * (max_n+1));
	}




	void
	match ()
	{
		sweight(n, ver, edges, p, ws, weight, p_init);
	}




	virtual
	~WgtSuitorSeqFor ()
	{
		// cout << "WgtSuitorSeqFor destructor" << endl;
		free(p_init);
		free(ws);
	}
		




public:

	// needs init match and suitor weight
	WM_VIDX_T	*p_init;
	WM_EW_T		*ws;
};



#endif
