#ifndef _GRAPH_HPP_
#define _GRAPH_HPP_

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using std::vector;	using std::cout;	using std::endl;	using std::string;
using std::setw;	using std::stringstream;	using std::iota;
using std::sort;	using std::transform;	using std::pair;
using std::ostream;	using std::ofstream;




template <typename VIDX_T,
		  typename EW_T = int32_t>
class Graph
{

public:

	enum GType
	{
		Directed,
		Undirected
	};



	// ctor
	Graph (
	    bool	adj_sorted = true
		) :
		nv(0),
		nedges(0),
		adj_sorted(adj_sorted)
	{
	}



	
	virtual
	void
	printx (const char *fname = nullptr, bool one_based = false)
	{
		int tmp = 0;
		if (one_based)
			tmp = 1;

		ofstream ofs;
		ostream &os = fname ? (ofs.open(fname), ofs) : cout;
		
		os << string(80, '*') << endl;
		os << "#vertices " << this->nv
		   << " #edges " << this->nedges
		   << " type " << (this->gt == Directed ? "Directed" : "Undirected")
		   << " ew " << (this->wgtd_edges ? "yes" : "no")
		   << " sorted-adj " << (this->adj_sorted ? "yes" : "no")
		   << endl;

		os << "adj-list:" << endl;
		for (VIDX_T v = 0; v < this->nv; ++v)
		{
			// sorted printing
			vector<pair<VIDX_T, EW_T>> curadj;
			for (uint64_t eptr = this->xadj[v]; eptr < this->xadj[v+1]; ++eptr)
			{
				if (this->wgtd_edges)
					curadj.push_back({this->adj[eptr], this->ew[eptr]});
				else
					curadj.push_back({this->adj[eptr], 0x0});
			}
			sort(curadj.begin(), curadj.end());

			os << setw(4) << v+tmp
			   << " [" << xadj[v] << ", " << xadj[v+1] << ")"
			   << " :";
			stringstream ss;
			for (auto &el: curadj)
			{
				ss << " ";
				if (this->wgtd_edges)
					ss << "(";
				ss << setw(5) << el.first;
				if (this->wgtd_edges)
					ss << ", " << std::fixed << std::setprecision(7)
					   << el.second << ")";
			}
			os << ss.str() << endl;
			
			
			// os << setw(4) << v+tmp
			//    << " [" << xadj[v] << ", " << xadj[v+1] << ")"
			//    << " :";
			// stringstream ss;
			// for (uint64_t eptr = this->xadj[v]; eptr < this->xadj[v+1]; ++eptr)
			// {
			// 	VIDX_T u = this->adj[eptr];
			// 	ss << " ";
			// 	if (this->wgtd_edges)
			// 		ss << "(";
			// 	ss << setw(5) << u+tmp;
			// 	if (this->wgtd_edges)
			// 		ss << ", " << this->ew[eptr] << ")";
			// }
			// os << ss.str() << endl;
		}

		os << string(80, '*') << endl;

		if (fname != nullptr)
			ofs.close();
	}

	


	void
	remove_self_loops ();



	
	void
	sort_adj (bool desc = false, bool use_ew = false);

	



	virtual ~Graph() = default;

	

public:

	VIDX_T				nv;
	uint64_t			nedges;
	vector<uint64_t>	xadj;	// nv+1
	vector<VIDX_T>		adj;
	vector<EW_T>		ew;
	bool				wgtd_edges;
	bool				adj_sorted;
	GType				gt;
	
};




// removes self loops in place
// assumes there can be at most one self loop per vertex
template<typename VIDX_T,
		 typename EW_T>
void
Graph<VIDX_T, EW_T>::remove_self_loops
(
)
{
	cout << "removing self-loops... ";
	uint64_t	prev		= 0;
	VIDX_T		nself_loops = 0;
	for (VIDX_T v = 0; v < nv; ++v)
	{
		bool		self_loop  = false;
		uint64_t	beg		   = xadj[v] + nself_loops;
		uint64_t	end		   = xadj[v+1];
		while (beg < end)
		{
			VIDX_T u = adj[beg];
			if (u == v)			// self-loop
			{
				self_loop = true;
			}
			else
			{
				adj[prev] = adj[beg];
				if (wgtd_edges)
					ew[prev] = ew[beg];
				++prev;
			}
			++beg;
		}

		if (self_loop)
			++nself_loops;

		xadj[v+1] -= nself_loops;
	}

	nedges -= nself_loops;

	cout << " removed " << nself_loops << " self-loops."
		 << " new number of edges " << nedges
		 << endl;
}





template <typename VIDX_T,
	 	  typename EW_T>
void
Graph<VIDX_T, EW_T>::sort_adj
(
    bool desc,
    bool use_ew
)
{
	// printx(true);

	if (use_ew && !wgtd_edges)
	{
		cout << "graph does not have edge weights, no need to sort." << endl;
		return;
	}


	cout << "sorting adj lists ("
		 << (use_ew ? "edge weights, " : "indices, ")
		 << (desc ? "desc" : "asc")
		 << ")" << endl;


	vector<pair<VIDX_T, EW_T>> tmp;
	EW_T dummy;
	for (uint64_t i = 0; i < xadj[nv]; ++i)
	{
		if (wgtd_edges)
			dummy = ew[i];

		tmp.push_back({adj[i], dummy});
	}

	
	auto f_comp_idx_asc =
		[&] (pair<VIDX_T, EW_T> &p1, pair<VIDX_T, EW_T> &p2)
		{
			return p1.first < p2.first;
		};
	auto f_comp_idx_desc =
		[&] (pair<VIDX_T, EW_T> &p1, pair<VIDX_T, EW_T> &p2)
		{
			return p1.first > p2.first;
		};
	auto f_comp_ew_asc =
		[&] (pair<VIDX_T, EW_T> &p1, pair<VIDX_T, EW_T> &p2)
		{
			return p1.second < p2.second;
		};
	auto f_comp_ew_desc =
		[&] (pair<VIDX_T, EW_T> &p1, pair<VIDX_T, EW_T> &p2)
		{
			return p1.second > p2.second;
		};

	
	if (!use_ew && !desc)
	{
		for (VIDX_T v = 0; v < nv; ++v)
			sort(tmp.begin() + xadj[v],
				 tmp.begin() + xadj[v+1],
				 f_comp_idx_asc);
	}
	else if (!use_ew && desc)
	{
		for (VIDX_T v = 0; v < nv; ++v)
			sort(tmp.begin() + xadj[v],
				 tmp.begin() + xadj[v+1],
				 f_comp_idx_desc);
	}
	else if (use_ew && !desc)
	{
		for (VIDX_T v = 0; v < nv; ++v)
			sort(tmp.begin() + xadj[v],
				 tmp.begin() + xadj[v+1],
				 f_comp_ew_asc);
	}
	else
	{
		for (VIDX_T v = 0; v < nv; ++v)
			sort(tmp.begin() + xadj[v],
				 tmp.begin() + xadj[v+1],
				 f_comp_ew_desc);
	}


	// copy back
	for (uint64_t i = 0; i < xadj[nv]; ++i)
	{
		adj[i] = tmp[i].first;
		if (wgtd_edges)
			ew[i] = tmp[i].second;
	}
	


	// printx(true);
	// exit(0);

	return;
}
	



#endif
