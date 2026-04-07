#ifndef HSL_MC80D_H
#define HSL_MC80D_H

#include <stdbool.h>

#ifndef mc80_default_control
#define mc80_control mc80_control_d
#define mc80_info mc80_info_d
#define mc80_default_control mc80_default_control_d
#define mc80_order mc80_order_d
#endif

typedef double mc80pkgtype_d_;


struct
mc80_control_d
{
    int f_arrays;
	bool action;
	bool unmatched_scale_zero;
	bool unmatched_last;
};


struct
mc80_info_d
{
    int compress_rank;
    int flag;
    int flag68;
    int max_cycle;
    int struct_rank;
    int stat;   
};


void mc80_default_control_d(struct mc80_control *control);
void mc80_order_d(int ord, int n, const int *ptr, const int *row,
				  const mc80pkgtype_d_ *cval, int *order,
				  const struct mc80_control *control,
				  struct mc80_info *info,
				  int *perm,
				  mc80pkgtype_d_ *scale);
void mc80_order_full_d(int ord, int n, const int *ptr, const int *row,
					   const mc80pkgtype_d_ *cval, int *order,
					   const struct mc80_control *control,
					   struct mc80_info *info,
					   int *perm,
					   mc80pkgtype_d_ *scale);

#endif
