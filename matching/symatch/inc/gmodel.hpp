/**
 * @file
 *	gmodel.hpp
 *
 * @author
 *	Oguz Selvitopi
 *
 * @date
 *	None
 *
 * @brief
 *	Graph models for modeling matrices for the purpose of matching
 *
 * @todo
 *
 * @note
 */


#ifndef _GMODEL_H_
#define _GMODEL_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gutil.hpp"
#include "tdef.hpp"

using std::vector;	using std::tuple;	using std::min;	using std::max;
using std::get;	using std::cout;	using std::endl;	using std::cerr;
using std::sort;	using std::fabs;	using std::unordered_set;
using std::pair;	using std::make_pair;	using std::unordered_map;
using std::log;




namespace
SyMatch
{

class
GModel
{

public:

	// directly from a sparse matrix
	int32_t	 n;
	int32_t *xind = nullptr;
	int32_t *ind  = nullptr;
	double	*val  = nullptr;

	MatMarket_t<VIDX_T, EW_T>	*mm = nullptr; // in
	Graph<VIDX_T, EW_T>			*g;	 // out


	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------


public:

	GModel (MatMarket_t<VIDX_T, EW_T> *mmptr) :
		mm(mmptr), g(NULL)
	{
	}

	


	GModel (int32_t m, int32_t *ind_ptr, int32_t *inds, double *vals) :
		n(m), xind(ind_ptr), ind(inds), val(vals), g(NULL)
	{
	}




	// graph model formation
	virtual void form_graph (bool use_diag_wgts = false) = 0;




	// permutes the graph for a given matching
	virtual
	void
	permute (vector<tuple<VIDX_T, VIDX_T, EW_T>> &m_sorted,
			 vector<VIDX_T> 					 &perm
		) = 0;



	// computes cardinality for a given matching on the graph model
	virtual
	uint64_t
	cardinality (vector<tuple<VIDX_T, VIDX_T, EW_T>> &m_sorted) = 0;

	
	

	virtual
	~GModel ()
	{
		if (g != NULL)
			delete g;
	}
};





// Standard graph model for symmetric matrices
// Removes self-loops
class
StandardSym : public GModel
{

public:

	StandardSym (MatMarket_t<VIDX_T, EW_T> *mmptr) :
		GModel(mmptr)
	{
	}



	
	void
	form_graph (bool use_diag_wgts)
	{
		g = new Graph<VIDX_T, EW_T>();

		// use the one already provided in gutil
		mm_to_gr(*mm, g);
		g->remove_self_loops();
	}




	void
	permute
	(
	    vector<tuple<VIDX_T, VIDX_T, EW_T>> &m_sorted,
		vector<VIDX_T>					    &perm
	)
	{
		VIDX_T curidx = 0;
		for (auto &t : m_sorted)
		{
			VIDX_T v = get<0>(t);
			VIDX_T u = get<1>(t);
			assert(v != 0);

			if (u == 0)				// unmatched vertex
			{
				perm[v-1] = curidx++;
				continue;
			}

			perm[min(v, u)-1] = curidx++;
			perm[max(v, u)-1] = curidx++;
		}
	}




	// NOT-IMPLEMENTED
	uint64_t
	cardinality (vector<tuple<VIDX_T, VIDX_T, EW_T>> &m_sorted)
	{
		return 0;
	}




	virtual
	~StandardSym ()
	{
	}



public:
	
};





// For each row/col i, we have two nodes vi and v'i
// If Aij exists we connect vi with vj and v'i with v'j
// Finally, there exists an edge between each node that represents the
// same pair, i.e., vi and v'i
// Note that the diagonals are enforced, it does not matter whether
// they are in the graph or not
class
RepGrDiagCon : public GModel
{

public:

	RepGrDiagCon (MatMarket_t<VIDX_T, EW_T> *mmptr) :
		GModel(mmptr)
	{
	}

	


	void
	form_graph (bool use_diag_wgts)
	{
		// valid only for symmetric matrices
		if (mm->storage != MatMarket_t<VIDX_T, EW_T>::Symmetric)
		{
			cerr << "RepGrDiagCon can only be formed for symmetric matrices."
				 << endl;
			exit(EXIT_FAILURE);
		}
		
		g = new Graph<VIDX_T, EW_T>();
		g->gt = Graph<VIDX_T, EW_T>::Undirected;
		g->wgtd_edges = mm->vals_exist;

		unordered_map<VIDX_T, EW_T> diags; // in case use_diag_wgts is set
		

		cout << "creating replicated graph with diag connections from "
			 << "Matrix Market data" << endl;


		// number of vertices
		g->nv = 2 * (mm->nr);
		VIDX_T offset = mm->nr;	// for G'

		
		// find size of adj lists
		g->xadj.resize(g->nv+2, static_cast<uint64_t>(0));
		for (uint64_t el = 0; el < mm->nelems; ++el)
		{
			VIDX_T r = mm->rids[el];
			VIDX_T c = mm->cids[el];

			if (r == c)
			{
				if (use_diag_wgts && g->wgtd_edges)
					diags.insert({r, fabs(static_cast<EW_T>(mm->vals[el]))});
					
				continue;
			}

			++(g->xadj[r+2]);
			++(g->xadj[c+2]);
			++(g->xadj[r+offset+2]);
			++(g->xadj[c+offset+2]);
		}

		
		// edges for diag connections across two graphs
		for (VIDX_T i = 0; i < mm->nr; ++i)
		{
			++(g->xadj[i+2]);
			++(g->xadj[i+offset+2]);
		}

		
		// prefix sum to get beg/end pointers
		for (VIDX_T i = 2; i < g->nv+2; ++i)
			g->xadj[i] += g->xadj[i-1];

		
		// allocate
		g->adj.resize(g->xadj[g->nv+1]);
		if (g->wgtd_edges)
			g->ew.resize(g->xadj[g->nv+1]);

		
		// fill adj and ew, finalize xadj
		for (uint64_t el = 0; el < mm->nelems; ++el)
		{
			VIDX_T	r = mm->rids[el];
			VIDX_T	c = mm->cids[el];
			EW_T	val;
			if (g->wgtd_edges)
				val = fabs(static_cast<EW_T>(mm->vals[el]));

			if (r == c)
				continue;

			// G
			g->adj[g->xadj[r+1]] = c;
			if (g->wgtd_edges)
				g->ew[g->xadj[r+1]] = val;
			++(g->xadj[r+1]);

			g->adj[g->xadj[c+1]] = r;
			if (g->wgtd_edges)
				g->ew[g->xadj[c+1]] = val;
			++(g->xadj[c+1]);

			// G'
			g->adj[g->xadj[r+offset+1]] = c + offset;
			if (g->wgtd_edges)
				g->ew[g->xadj[r+offset+1]] = val;
			++(g->xadj[r+offset+1]);

			g->adj[g->xadj[c+offset+1]] = r + offset;
			if (g->wgtd_edges)
				g->ew[g->xadj[c+offset+1]] = val;
			++(g->xadj[c+offset+1]);
		}

		
		// edges for diag connections across two graphs
		for (VIDX_T i = 0; i < mm->nr; ++i)
		{
			// G
			g->adj[g->xadj[i+1]] = i + offset;
			if (g->wgtd_edges)
			{
				g->ew[g->xadj[i+1]] = 0; // set 0
				if (use_diag_wgts)
				{
					auto tmp = diags.find(i);
					if (tmp != diags.end())
						g->ew[g->xadj[i+1]] = tmp->second;
				}
					
				
			}
			++(g->xadj[i+1]);

			// G'
			g->adj[g->xadj[i+offset+1]] = i;
			if (g->wgtd_edges)
			{
				g->ew[g->xadj[i+offset+1]] = 0; // set 0
				if (use_diag_wgts)
				{
					auto tmp = diags.find(i);
					if (tmp != diags.end())
						g->ew[g->xadj[i+offset+1]] = tmp->second;
				}
			}
			++(g->xadj[i+offset+1]);
		}


		g->nedges = g->xadj[g->nv];
		
		cout << "#vertices " << g->nv
		 << " #edges " << g->nedges
		 << endl;


		// handle sorted adj
		if (g->adj_sorted)
			g->sort_adj(true, true); // desc and use ew
	}




	// m_sorted is 1-based
	void
	permute
	(
	    vector<tuple<VIDX_T, VIDX_T, EW_T>> &m_sorted,
		vector<VIDX_T>					    &perm
	)
	{
		struct
			pair_hash
		{
			inline std::size_t operator() (const std::pair<int,int> & v) const
			{
				return v.first*31+v.second;
			}
		};
		
		VIDX_T offset = mm->nr;	// for G'

		int nm_g	 = 0;
		int nm_gp	 = 0;
		int nm_cross = 0;
		int num_g    = 0;
		int num_gp   = 0;
		int nm_id    = 0;
		int num_id   = 0;

		// need membership test to check identical matches/unmatches in G and G'
		unordered_set<pair<VIDX_T, VIDX_T>, pair_hash> mset;
		unordered_set<VIDX_T>  umset;
		for (auto &t : m_sorted)
		{
			VIDX_T v = get<0>(t);
			VIDX_T u = get<1>(t);
			assert(v != 0);

			if (u == 0)
			{
				umset.insert(v);
				continue;
			}

			mset.insert(make_pair(min(v, u), max(v, u)));
		}

		

		for (auto &t : m_sorted)
		{
			VIDX_T v = get<0>(t);
			VIDX_T u = get<1>(t);
			assert(v != 0);

			if (u == 0)
			{
				if (v <= offset)
					++num_g;
				else
					++num_gp;

				VIDX_T tmp = (v <= offset ? v+offset : v-offset);
				if (umset.find(tmp) != umset.end())
					++num_id;
					
				continue;
			}

			if ((v <= offset && u > offset) ||
				(v > offset && u <= offset))
			{
				++nm_cross;
				assert((v <= offset && u == v+offset) ||
					   (v >= offset && u == v-offset)); // must be diag
				continue;
			}

			// assert((v <= offset && u <= offset) ||
			// 	   (v > offset && u > offset));

			if (v <= offset)
			{
				assert(u <= offset);
				++nm_g;

				if (mset.find(make_pair(min(v+offset, u+offset),
										max(v+offset, u+offset)))
					!= mset.end())
					++nm_id;
				
			}
			else if (v > offset)
			{
				assert(u > offset);
				++nm_gp;

				if (mset.find(make_pair(min(v-offset, u-offset),
										max(v-offset, u-offset)))
					!= mset.end())
					++nm_id;
			}
			else
				cout << "shouldn't be here" << endl;
		}


		nm_id  /= 2;
		num_id /= 2;
		cout << "total #matched " << (nm_g + nm_gp)
			 << " #unmatched " << (num_g + num_gp)
			 << "\n"
			 << "G : #matched " << nm_g
			 << " #unmatched " << num_g
			 << "\n"
			 << "G': #matched " << nm_gp
			 << " #unmatched " << num_gp
			 << "\n"
			 << "#cross " << nm_cross
			 << "\n"
			 << "#identical matched " << nm_id << " unmatched " << num_id
			 << "\n"
			 << "G : unique matched " << (nm_g-nm_id)
			 << " #unmatched " << (num_g-num_id)
			 << "\n"
			 << "G': unique matched " << (nm_gp-nm_id)
			 << " #unmatched " << (num_gp-num_id)
			 << endl;

		if ((nm_g - nm_id > 0) || (num_g - num_id > 0) ||
			(nm_gp - nm_id > 0) || (num_gp - num_id > 0))
		{
			cout << "Warning: there are non-symmetric matched vertices. "
				"Aborting.\n" << endl;
			exit(1);
		}


		// reorder
		VIDX_T curidx = 0;
		for (auto &t : m_sorted)
		{
			VIDX_T v = get<0>(t);
			VIDX_T u = get<1>(t);
			assert(v != 0);

			if (u == 0)			// unmatched
			{
				if (v <= offset)
					perm[v-1] = curidx++;				
					
				continue;
			}

			assert(v != 0);

			if (v > offset && u > offset)	// will be processed when v <= offset
				continue;

			if (u > offset)	// v in G, u in G'
			{
				assert(v == u-offset);
				perm[v-1] = curidx++;
			}
			else				// v and u in G
			{
				perm[min(v, u)-1] = curidx++;
				perm[max(v, u)-1] = curidx++;
			}			
		}
	}




	// NOT-IMPLEMENTED
	uint64_t
	cardinality (vector<tuple<VIDX_T, VIDX_T, EW_T>> &m_sorted)
	{
		return 0;
	}




	virtual
	~RepGrDiagCon ()
	{
	}


	

public:
	
};





// Standard symmetric graph model enhanced with additional vertices
// for diagonals
// A self-loop is represented with two vertices and an edge between them
// For row i, we have vi as normal
// We add another vertex v'i and connect it to vi whether the diag
// exists or not
// Its weight is set to 0 if it does not exist
class
StandardSymDiag : public GModel
{

public:

	StandardSymDiag (MatMarket_t<VIDX_T, EW_T> *mmptr) :
		GModel(mmptr)
	{
	}

	


	StandardSymDiag (int32_t m, int32_t *ind_ptr, int32_t *inds, double *vals) :
		GModel(m, ind_ptr, inds, vals)
	{
	}

	



	void form_graph (bool use_diag_wgts)
	{
		if (mm != nullptr)
			form_graph_mm(use_diag_wgts);
		else if (xind != nullptr)
			form_graph_sp();

		return;
	}
	



	// single-pass
	// matching produces error on this graph (i is matched j but j is
	// not matched i)
	void
	form_graph_sp2 ()
	{
		g = new Graph<VIDX_T, EW_T>();
		g->gt = Graph<VIDX_T, EW_T>::Undirected;
		g->wgtd_edges = true;;
		g->adj_sorted = false;

		int32_t m = xind[n];	// original nnzs
		cout << "n " << n << " m " << m << endl;

		g->nv = 2 * n;
		g->xadj.resize(g->nv+1, static_cast<uint64_t>(0));
		g->adj.resize(m+n);		// upper bound
		g->ew.resize(m+n);		// upper bound

		int32_t i	   = 0;
		int32_t diag_i = m;
		for (int32_t c = 0; c < n; ++c)
		{
			int32_t j = g->xadj[c];
			for (int32_t rptr = xind[c]; rptr < xind[c+1]; ++rptr)
			{
				int32_t r = ind[rptr];
				double	v = val[rptr];

				if (v == 0.0)
					continue;

				v = fabs(v);

				if (r == c)
				{
					g->adj[j]	   = r + n;
					g->ew[j]	   = v;

					// diag vertex
					g->xadj[n+1+r] = 1;
					g->adj[diag_i] = r;
					g->ew[diag_i]  = v;
					++diag_i;
				}
				else
				{
					g->adj[j] = r;
					g->ew[j]  = 2*v;
				}

				++j;
			}

			g->xadj[c+1] = j;
		}


		cout << "#edges " << g->xadj[n]
			 << " eliminated " << (m - g->xadj[n])
			 << " #diag-edges " << (diag_i - m)
			 << endl;

		i				= 0;
		int32_t m_diags = diag_i - m;
		int32_t offset	= g->xadj[n];
		assert(m_diags <= n);
		for (int32_t v = n+1; v <= 2*n; ++v)
		{
			g->xadj[v] += g->xadj[v-1];

			if (i < m_diags)
			{
				g->adj[offset+i] = g->adj[m+i];
				g->ew[offset+i]	 = g->ew[m+i];
				++i;
			}
		}

		g->nedges = g->xadj[g->nv];

		cout << "#vertices " << g->nv
			 << " #edges " << g->nedges
			 << endl;

		// g->printx("matrix_ACTIVSg10k_AC_00.mm_sp2");


		return;
	}




	// assumes symmetry and weights
	void
	form_graph_sp ()
	{
		g = new Graph<VIDX_T, EW_T>();
		g->gt = Graph<VIDX_T, EW_T>::Undirected;
		g->wgtd_edges = true;;
		g->adj_sorted = false;

		g->nv = 2 * n;

		// find size of adj lists
		g->xadj.resize(g->nv+2, static_cast<uint64_t>(0));
		for (int32_t c = 0; c < n; ++c)
		{
			for (int32_t rptr = xind[c]; rptr < xind[c+1]; ++rptr)
			{
				int32_t r = ind[rptr];
				double	v = val[rptr];

				if (v == 0.0)
					continue;

				if (r < c)
				{
					++(g->xadj[r+2]);
					++(g->xadj[c+2]);
				}
				else if (r == c)
				{
					++(g->xadj[r+2]);
					++(g->xadj[n+r+2]);
				}
			}
		}

		
		
		// prefix sum to get beg/end pointers
		for (int32_t i = 2; i < g->nv+2; ++i)
			g->xadj[i] += g->xadj[i-1];


		// allocate
		g->adj.resize(g->xadj[g->nv+1]);
		g->ew.resize(g->xadj[g->nv+1]);

		
		// fill adj and ew, finalize xadj
		for (int32_t c = 0; c < n; ++c)
		{
			for (int32_t rptr = xind[c]; rptr < xind[c+1]; ++rptr)
			{
				int32_t r = ind[rptr];
				double	v = val[rptr];

				if (v == 0.0)
					continue;

				v = fabs(v);

				if (r < c)
				{
					g->adj[g->xadj[r+1]] = c;
					g->ew[g->xadj[r+1]]	 = 2*v;
					++(g->xadj[r+1]);
					g->adj[g->xadj[c+1]] = r;
					g->ew[g->xadj[c+1]]	 = 2*v;
					++(g->xadj[c+1]);
				}
				else if (r == c)
				{
					g->adj[g->xadj[r+1]] = c+n;
					g->ew[g->xadj[r+1]]	 = v;
					++(g->xadj[r+1]);
					g->adj[g->xadj[c+n+1]] = r;
					g->ew[g->xadj[c+n+1]]  = v;
					++(g->xadj[c+n+1]);
				}
			}
		}

		
		g->nedges = g->xadj[g->nv];

		// double ew_sum = 0.0;
		// for (uint64_t i = 0; i < g->nedges; ++i)
		// 	ew_sum += g->ew[i];
		
		cout << "#vertices " << g->nv
			 << " #edges " << g->nedges
			// << " ewsum " << ew_sum
			 << endl;

		// g->printx("matrix_ACTIVSg10k_AC_00.mm_sp");

		return;
	}



	
	void
	form_graph_mm (bool use_diag_wgts)
	{
		std::chrono::steady_clock::time_point tbeg;
		
		// valid only for symmetric matrices
		if (mm->storage != MatMarket_t<VIDX_T, EW_T>::Symmetric)
		{
			cerr << "StandardSymDiag can only be formed for symmetric matrices."
				 << endl;
			exit(EXIT_FAILURE);
		}

		g = new Graph<VIDX_T, EW_T>();
		g->gt = Graph<VIDX_T, EW_T>::Undirected;
		g->wgtd_edges = mm->vals_exist;
		g->adj_sorted = false;

		unordered_map<VIDX_T, EW_T> diags; // in case use_diag_wgts is set
		

		cout << "creating standard symmetric graph model with diagonals from "
			 << "Matrix Market data" << endl;


		// number of vertices
		g->nv = 2 * (mm->nr);

		tbeg = std::chrono::steady_clock::now();

		// find size of adj lists
		g->xadj.resize(g->nv+2, static_cast<uint64_t>(0));
		for (uint64_t el = 0; el < mm->nelems; ++el)
		{
			VIDX_T r = mm->rids[el];
			VIDX_T c = mm->cids[el];

			if (r == c)
			{
				if (use_diag_wgts && g->wgtd_edges)
					diags.insert({r, fabs(static_cast<EW_T>(mm->vals[el]))});
					
				continue;
			}

			++(g->xadj[r+2]);
			++(g->xadj[c+2]);
		}

		int64_t t1 = std::chrono::duration_cast<std::chrono::microseconds>
			(std::chrono::steady_clock::now() - tbeg).count();
		tbeg = std::chrono::steady_clock::now();

		// edges for diag connections
		for (VIDX_T i = 0; i < mm->nr; ++i)
		{
			if (diags.find(i) != diags.end())
			{
				++(g->xadj[i+2]);
				++(g->xadj[mm->nr+i+2]);
			}
		}

		int64_t t2 = std::chrono::duration_cast<std::chrono::microseconds>
			(std::chrono::steady_clock::now() - tbeg).count();
		tbeg = std::chrono::steady_clock::now();

		// prefix sum to get beg/end pointers
		for (VIDX_T i = 2; i < g->nv+2; ++i)
			g->xadj[i] += g->xadj[i-1];

		int64_t t3 = std::chrono::duration_cast<std::chrono::microseconds>
			(std::chrono::steady_clock::now() - tbeg).count();
		tbeg = std::chrono::steady_clock::now();

		// allocate
		g->adj.resize(g->xadj[g->nv+1]);
		if (g->wgtd_edges)
			g->ew.resize(g->xadj[g->nv+1]);

		// fill adj and ew, finalize xadj
		for (uint64_t el = 0; el < mm->nelems; ++el)
		{
			VIDX_T	r = mm->rids[el];
			VIDX_T	c = mm->cids[el];
			EW_T	val;
			if (g->wgtd_edges)
				val = fabs(static_cast<EW_T>(mm->vals[el]));

			if (r == c)
				continue;

			g->adj[g->xadj[r+1]] = c;
			if (g->wgtd_edges)
				g->ew[g->xadj[r+1]] = 2*val;
			++(g->xadj[r+1]);

			g->adj[g->xadj[c+1]] = r;
			if (g->wgtd_edges)
				g->ew[g->xadj[c+1]] = 2*val;
			++(g->xadj[c+1]);
		}

		int64_t t4 = std::chrono::duration_cast<std::chrono::microseconds>
			(std::chrono::steady_clock::now() - tbeg).count();
		tbeg = std::chrono::steady_clock::now();

		// edges for diag connections
		for (VIDX_T i = 0; i < mm->nr; ++i)
		{
			auto tmp = diags.find(i);
			if (tmp == diags.end())
				continue;
			
			g->adj[g->xadj[i+1]] = i + mm->nr;
			g->adj[g->xadj[mm->nr+i+1]] = i;
			if (g->wgtd_edges && use_diag_wgts)
			{
				g->ew[g->xadj[i+1]]		   = tmp->second;
				g->ew[g->xadj[mm->nr+i+1]] = tmp->second;
			}
			
			++(g->xadj[i+1]);
			++(g->xadj[mm->nr+i+1]);
		}

		int64_t t5 = std::chrono::duration_cast<std::chrono::microseconds>
			(std::chrono::steady_clock::now() - tbeg).count();
		tbeg = std::chrono::steady_clock::now();

		g->nedges = g->xadj[g->nv];

		// handle sorted adj
		if (g->adj_sorted)
			g->sort_adj(true, true); // desc and use ew

		int64_t t6 = std::chrono::duration_cast<std::chrono::microseconds>
			(std::chrono::steady_clock::now() - tbeg).count();


		auto f_usec_to_msec = [] (int64_t arg) -> double
		{
			return static_cast<double>(arg) / 1e+3;
		};
		
		cout << "t1 " << f_usec_to_msec(t1) << "\n"
			 << "t2 " << f_usec_to_msec(t2) << "\n"
			 << "t3 " << f_usec_to_msec(t3) << "\n"
			 << "t4 " << f_usec_to_msec(t4) << "\n"
			 << "t5 " << f_usec_to_msec(t5) << "\n"
			 << "t6 " << f_usec_to_msec(t6)
			 << endl;


		// double ew_sum = 0.0;
		// for (uint64_t i = 0; i < g->nedges; ++i)
		// 	ew_sum += g->ew[i];
		
		cout << "#vertices " << g->nv
			 << " #edges " << g->nedges
			 // << " ewsum " << ew_sum
			 << endl;

		// g->printx("matrix_ACTIVSg10k_AC_00.mm_mm");


		return;
	}
	


	
	// m_sorted is 1-based
	void
	permute
	(
	    vector<tuple<VIDX_T, VIDX_T, EW_T>> &m_sorted,
		vector<VIDX_T>					    &perm
	)
	{
		VIDX_T curidx = 0;
		for (auto &t : m_sorted)
		{
			VIDX_T v = get<0>(t);
			VIDX_T u = get<1>(t);
			assert(v != 0);

			if (u == 0)				// unmatched vertex
			{
				if (v <= mm->nr)
					perm[v-1] = curidx++;
				continue;
			}

			perm[min(v, u)-1] = curidx++;
			if (u <= mm->nr && v <= mm->nr)
				perm[max(v, u)-1] = curidx++;			
		}


		return;
	}




	
	// NOT-IMPLEMENTED
	uint64_t
	cardinality (vector<tuple<VIDX_T, VIDX_T, EW_T>> &m_sorted)
	{
		uint64_t	n_matched	= 0;
		uint64_t	n_unmatched = 0;
		for (auto &t : m_sorted)
		{
			VIDX_T v = get<0>(t);
			VIDX_T u = get<1>(t);
			assert(v != 0);

			if (u == 0)				// unmatched vertex
			{
				if (v <= mm->nr)
					++n_unmatched;
				continue;
			}

			
			if (u <= mm->nr && v <= mm->nr)
				n_matched += 2;
			else
				++n_unmatched;	// diag match counted as unmatched			
		}


		cout << n_unmatched << " " << n_matched << endl;


		return n_matched;
	}

	


	virtual
	~StandardSymDiag ()
	{
	}
	
};

}

#endif
