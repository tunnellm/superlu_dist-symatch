#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

using std::vector;	using std::tuple;	using std::string;




namespace
SyMatch
{

// Matrix market information
template <typename IDX_T,
		  typename VAL_T>
struct
MatMarket_t
{

	enum Format
	{
		Coordinate,				// sparse
		Array					// dense - NOT IMPLEMENTED
	};


	
	enum Type
	{
		Real,
		Complex,				// NOT IMPLEMENTED
		Pattern,
		Integer
	};



	enum Storage
	{
		General,
		Hermitian,				// NOT IMPLEMENTED
		Symmetric,
		Skew					// NOT IMPLEMENTED
	};




	// matrix properties
	Format	fmt;
	Type	type;
	Storage storage;

	// matrix itself
	uint64_t		nr, nc;
	uint64_t		nelems;		// #entries in the file
	uint64_t		nnzs;		// computed
	vector<IDX_T>	rids;		// 0-based
	vector<IDX_T>	cids;		// 0-based
	vector<VAL_T>	vals;
	vector<string>  svals;		// avoid playing with numerics
	bool			vals_exist;

	// extras
	// not filled yet - don't use diags
	vector<tuple<bool, bool, VAL_T> > diags; // (idx exists, val exists, val)
	string header;
	
};

}

#endif
