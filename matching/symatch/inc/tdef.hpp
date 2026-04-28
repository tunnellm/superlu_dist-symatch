/**
 * @file
 *	tdef.hpp
 *
 * @author
 *	Oguz Selvitopi
 *
 * @date
 *	None
 *
 * @brief
 *	typedefs and global variables
 *
 * @todo
 *
 * @note
 *
 */

#ifndef _TDEF_HPP_
#define _TDEF_HPP_

#include <string>

typedef int		VIDX_T;
typedef double	EW_T;




namespace
SyMatch
{

enum
match_alg_t
{
	SUITOR_WGT_SEQ_PLAIN		// Suitor based weighted matching. Using a
								// for-loop as outer loop
};



struct
params_t
{
	std::string mmfile;
};

}

#endif
