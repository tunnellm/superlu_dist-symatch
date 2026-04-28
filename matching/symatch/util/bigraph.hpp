#ifndef _BIGRAPH_HPP_
#define _BIGRAPH_HPP_

#include "graph.hpp"



namespace
SyMatch
{

template <typename VIDX_T,
		  typename EW_T = int32_t>
class BiGraph : public Graph<VIDX_T, EW_T>
{

public:

	// ctor
	BiGraph (
	    bool adj_sorted = true
		) :
		Graph<VIDX_T, EW_T>::Graph(adj_sorted)
	{
	}
	
		

	virtual
	void
	printx ()
	{
		cout << string(80, '*') << endl;
		cout << "#vertices " << this->nv
			 << " #edges " << this->nedges
			 << " type " << (this->gt == Graph<VIDX_T, EW_T>::Directed
							 ? "Directed" : "Undirected")
			 << " ew " << (this->wgtd_edges ? "yes" : "no")
			 << " sorted-adj " << (this->adj_sorted ? "yes" : "no")
			 << " bipartite"
			 << " #lvertices " << this->nvl
			 << " #rvertices " << this->nvr
			 << endl;

		cout << "adj-list (L):" << endl;
		for (VIDX_T v = 0; v < this->nvl; ++v)
		{
			cout << setw(4) << v << " :";
			stringstream ss;
			for (uint64_t eptr = this->xadj[v]; eptr < this->xadj[v+1]; ++eptr)
			{
				VIDX_T u = this->adj[eptr];
				ss << " ";
				if (this->wgtd_edges)
					ss << "(";
				ss << setw(5) << u;
				if (this->wgtd_edges)
					ss << ", " << this->ew[eptr] << ")";
			}
			cout << ss.str() << endl;
		}

		cout << "adj-list (R):" << endl;
		for (VIDX_T v = this->nvl; v < this->nv; ++v)
		{
			cout << setw(4) << v << " :";
			stringstream ss;
			for (uint64_t eptr = this->xadj[v]; eptr < this->xadj[v+1]; ++eptr)
			{
				VIDX_T u = this->adj[eptr];
				ss << " ";
				if (this->wgtd_edges)
					ss << "(";
				ss << setw(5) << u;
				if (this->wgtd_edges)
					ss << ", " << this->ew[eptr] << ")";
			}
			cout << ss.str() << endl;
		}

		cout << string(80, '*') << endl;
	}
	

public:

	VIDX_T			nvl;		// # of vertices on the left
	VIDX_T			nvr;		// # of vertices on the right
	
};

}




#endif
