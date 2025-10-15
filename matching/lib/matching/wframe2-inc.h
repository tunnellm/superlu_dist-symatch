#ifndef _WFRAME2_INC_H_
#define _WFRAME2_INC_H_


typedef struct { int x,y; } edge;

typedef struct { int x,y;
                 double w; } wed;

typedef struct { int x;
                 double w; } neig;
#ifndef true
#define true  (1)
#endif

#ifndef false
#define false (0)
#endif

#ifndef TRUE
#define TRUE  (1)
#endif

#ifndef FALSE
#define FALSE (0)
#endif

/* #define max_n  68000000 */
/* #define max_m 761000000 */
#define max_n  6800000
#define max_m 99000000

#define max_experiment 200   // Maximum different algorithms that can be tested
#define max_conf 20          // Maximum different thread configurations that can be tested

#define max_graphs 20   // Maximum number of graphs in input file

#define max_threads 500


#endif
