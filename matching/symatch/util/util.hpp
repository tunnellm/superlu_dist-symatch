#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "types.hpp"

using std::string;	using std::strstr;	using std::cerr;	using std::endl;
using std::strncmp;	using std::cout;	using std::ifstream;	using std::stod;
using std::stringstream;	using std::move;	using std::ofstream;




namespace
SyMatch
{
	
template <typename IDX_T,
		  typename VAL_T>
void
mmread
(
    string						 fname,
	MatMarket_t<IDX_T, VAL_T>	*mm
)
{
	const int	max_line_len = 1<<16;
	ifstream	infile;
	char		line[max_line_len];

	infile.exceptions(ifstream::badbit);
	try
	{
		infile.open(fname.c_str());
	}
	catch (const ifstream::failure &e)
	{
		cerr << "FATAL: Cannot open file " << fname << endl;
		exit(1);
	}	

	// header
	infile.getline(line, max_line_len);
	mm->header.append(line); mm->header.push_back('\n');
	if (strncmp(line, "%%MatrixMarket", 14))
	{
		cerr << "Matrix market header incorrect." << endl;
		exit(1);
	}

	mm->fmt		= MatMarket_t<IDX_T, VAL_T>::Coordinate;
	mm->type	= MatMarket_t<IDX_T, VAL_T>::Pattern;
	mm->storage = MatMarket_t<IDX_T, VAL_T>::General;

	// format
	if (strstr(line, "array"))
	{
		cerr << "\"array\" dense format not supported." << endl;
		exit(1);
	}

	// type
	if (strstr(line, "real"))
	{
		mm->type = MatMarket_t<IDX_T, VAL_T>::Real;
		cout << "Matrix type is real." << endl;
	}
	
	if (strstr(line, "complex"))
	{
		cout << "\"complex\" matrix type not supported. Will read the "
			 << "matrix as \"pattern\"." << endl;
	}

	if (strstr(line, "pattern"))
	{
		mm->type = MatMarket_t<IDX_T, VAL_T>::Pattern;
		cout << "Matrix type is pattern." << endl;
	}

	if (strstr(line, "integer"))
	{
		mm->type = MatMarket_t<IDX_T, VAL_T>::Integer;
		cout << "Matrix type is integer." << endl;
	}

	// storage
	if (strstr(line, "hermitian"))
	{
		cerr << "\"hermitian\" storage not supported." << endl;
		exit(1);
	}

	if (strstr(line, "skew"))
	{
		cerr << "\"skew\" storage not supported." << endl;
		exit(1);
	}

	if (strstr(line, "symmetric"))
	{
		mm->storage = MatMarket_t<IDX_T, VAL_T>::Symmetric;
		cout << "Matrix storage is symmetric." << endl;
	}

	if (strstr(line, "general"))
	{
		mm->storage = MatMarket_t<IDX_T, VAL_T>::General;
		cout << "Matrix storage is general." << endl;
	}

	mm->vals_exist = false;
	if ((mm->type != MatMarket_t<IDX_T, VAL_T>::Pattern))
		mm->vals_exist = true;
	mm->nnzs = 0;

	// read till first char not '%'
	char c = infile.get();
    while (c == '%')
	{
		mm->header.push_back(c);
		infile.getline(line, max_line_len);
		mm->header.append(line); mm->header.push_back('\n');
		c = infile.get();		
	}
	infile.unget();

	infile >> mm->nr >> mm->nc >> mm->nelems;
	uint64_t nlines_read = 0;
	while (nlines_read++ < mm->nelems)
	{
		IDX_T r, c;
		infile >> r >> c;
		
		++(mm->nnzs);
		if ((mm->storage == MatMarket_t<IDX_T, VAL_T>::Symmetric) &&
			(r != c))
			++(mm->nnzs);

		mm->rids.push_back(r-1);
		mm->cids.push_back(c-1);
		if (mm->vals_exist)
		{
			string s;
			infile >> s;
			mm->svals.push_back(s);

			double vd = std::stod(s);
			// VAL_T v;
			// infile >> v;			
			mm->vals.push_back(static_cast<VAL_T>(vd));
		}		
	}
	
	cout << "matrix " << fname
		 << " nrows " << mm->nr
		 << " ncols " << mm->nc
		 << " nelems " << mm->nelems
		 << " nnzs " << mm->nnzs
		 << endl;

	infile.close();


	return;
}





// writes matrix market struct to file
// can give permutation where if perm[v] = u, v in the matrix market file will
// be u in the output
//   symmetric permutation (both rows and cols are permuted)
template <typename IDX_T,
		  typename VAL_T>
void
mmwrite
(
    MatMarket_t<IDX_T, VAL_T>	&mm,
	string						&fname,
    vector<IDX_T>				*perm = nullptr
)
{
	ofstream outfile(fname, ofstream::out);
	outfile << mm.header;

	outfile << mm.nr << " " << mm.nc << " " << mm.nelems << endl;
	for (uint64_t el = 0; el < mm.nelems; ++el)
	{
		IDX_T r = mm.rids[el];
		IDX_T c = mm.cids[el];

		if (perm == nullptr)
			outfile << r+1 << " " << c+1 << " ";
		else
			outfile << (*perm)[r]+1 << " " << (*perm)[c]+1 << " ";

		if (mm.vals_exist)
			outfile << mm.svals[el]; // numerics

		outfile << "\n";
	}


	outfile.close();
}

}

#endif
