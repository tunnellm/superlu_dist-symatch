#include "wframe2-inc.h"
#include "wframe2.h"	// Include files and define variables
#include "weight2_onesuitor.c"  // Code for suitor with only one value to give weight and id of suitor

#include "rama.c"               // Suitor + Rama type post-processing
#include "roma.c"               // Roma type post-processing
#include "rema.c"               // Roma type post-processing

#include "seq1.c" 		// Sequential unweighted greedy matching,  vertex based 
#include "eseq1.c"		// Sequential unweighted greedy matching,  edge based 

#include "wseq1.c" 		// Sequential weighted localy greedy matching, vertex based
#include "wseq2.c" 		// Sequential weighted localy greedy matching, vertex based, but in random order
#include "wseq3.c" 		// Same as above, but code has been optimized with prefetching directives

#include "ver1.c"               // Parallel unweighted matching, vertex based using verification to check correctness
#include "edge1.c"              // Parallel unweighted matching, edge based using verification to check correctness
#include "edge2.c"              // Optimized version of edge1
#include "lock1.c"              // Parallel unweighted matching, edge based using locking to assure correctness
#include "weight1.c"            // First version of parallel suitor code, not very efficient
#include "weight2.c"            // Final version of parallel suitor code

#include "sweight.c" 		// Suitor based weighted matching. Using a for-loop as outer loop
#include "sweight1.c" 		// Suitor based weighted matching. Using sorted adjacency lists
#include "sweight7.c" 		// Suitor based weighted matching. Optimized loop.
#include "sweight3.c" 		// Suitor based weighted matching. Using a while-loop as outer loop
#include "sweight4.c" 		// Two rounds of suitor based matching, followed by dyn. prog 
#include "sweight5.c" 		// Same as above, but now with separate routine for dyn. prog
#include "sweight6.c" 		// Two rounds of localy greedy matching (wseq), followed by dyn. prog
#include "sweight8.c" 		// Same as sweight5, but now with separate routine for cyclic dyn. prog
#include "sweight9.c" 		// Same as sweight6, but now with separate routine for cyclic dyn. prog
#include "sweight10.c" 		// Two level algorithm with DP, assuming sorted neightbor lists
#include "sweight11.c" 		// Suitor based algorithm with compression of neighbor lists 
#include "mcw1.c" 		    // Suitor based algorithm using a stack as outer loop 
#include "mcw2.c" 		    // Suitor based algorithm using a circular queue as outer loop 
#include "mcw3.c" 		    // Suitor based algorithm using a queue as outer loop, implemented with two arrays 
#include "pdp.c" 		// Two level parallel suitor algorithm with DP
#include "spdp.c" 		// Two level parallel suitor algorithm with DP
#include "psweight1.c" 		// Parallel suitor based weighted matching. Using sorted adjacency lists
#include "localmax.c" 		// Round based local max algorithm
#include "plocalmax.c" 		// Parallel round based local max algorithm, using a global edge list
#include "plocalmax1.c" 	// Parallel round based local max algorithm, using local edge lists
#include "greedy.c"             // Sequential greedy algorithm on sorted edge list
#include "MB1List.c"        // Parallel Manne-Bisseling algorithm, 1 list
#include "MBLocal.c"        // Parallel Manne-Bisseling algorithm, local list
#include "bmatching.c"      // 2-matching followed by DP
#include "bcmatching.c"      // 2-matching followed by DP, using compressed neigbor lists
#include "pstack1.c"        // Parallel McVittie and Wilson algorithm using a stack
#include "pqueue1.c"        // Parallel Gale-Shapley algorithm using a queue

#include "bweight1.c"       // Parallel code for computing a 2-matching, currently does not produce a 1-matching

#include "gpa.c" 		// Create paths and cycles by adding in edges (by decreasing weight), followed by dyn. prog
#include "pth1.c" 		// 2 rounds of Path growing algorithm followed by dynamic programming
#include "path.c" 		// Path growing algorithm followed by dynamic programming

#include "wsort.c"		// Sorts edge lists of a graph by decreasing weight. Uses insertion sort
#include "mmio.c"               // Matrix market routines for reading files
// #include <numa.h>


// Routine for comparing the weight of two edges
// Used for sorting edges by increasing weight in qsort()

static int compedge(const void *m1, const void *m2) {
    wed *v1 = (wed *) m1; 
    wed *v2 = (wed *) m2; 
    return (int) (v1->w < v2->w);
}

static int compneig(const void *m1, const void *m2) {
    neig *v1 = (neig *) m1; 
    neig *v2 = (neig *) m2; 
    if  (v1->w < v2->w)
      return true;
    if  (v1->w > v2->w)
      return false;
    return (int) (v1->x < v2->x);
}


int read_bin_graph(int *n,int *m,char *f_name,int **ver,int **edges,edge **e,
                   double **weight,wed **we,neig **swe,int **where) {

  int *count;
  int *start;
  int i;
  int num_edges;
  FILE *fp;
  int x,y;
  double v;

  fp = fopen(f_name,"r");
  if (fp == NULL) {
    printf("Could not open file %s \n",f_name);
    return(false);
  }

  fread(n,sizeof(int),1,fp);
  fread(m,sizeof(int),1,fp);


  start = (int *) malloc(sizeof(int)*((*n)+1));
  if (start == NULL) {
    printf("Unable to allocate space for start-array in read_bin_graph \n");
    return(false);
  }

// Used for calculating where the edge list of each vertex should be stored in the compressed edge list

  count = (int *) malloc(sizeof(int)*((*n)+1));
  if (count == NULL) {
    printf("Unable to allocate space for count-array in read_bin_graph \n");
    return(false);
  }

// Start counting of degrees by setting counters to zero
  for(i=1;i<=*n;i++) {
    count[i] = 0;
  }
  
  fread(*we,(*m)*(2*sizeof(int)+sizeof(double)),1,fp);
  

  num_edges = 0;
  
  

  for(i=0;i<*m;i++) {

    x = (*we)[num_edges].x;
    y = (*we)[num_edges].y;
    v = (*we)[num_edges].w;

/*
    (*e)[num_edges].x = x; // Store edges
    (*e)[num_edges].y = y;
*/

    count[x]++; // Get vertex degrees
    count[y]++;

    num_edges++;
    if (num_edges > *m) {
      printf("Have set num_edges to %d while i=%d \n",num_edges,i);
      return;
    }
  }


  
  *m = num_edges;  // Make sure m is the correct number of edges

// Find starting positions in edge list for each vertex
  start[1] = 0;
  (*ver)[1] = 0;
  for(i=2;i<=*n+1;i++) {
    start[i] = start[i-1]+count[i-1];
    (*ver)[i] = start[i];
  }

// Place edges in edge lists, once for each endpoint

  
  for(i=0;i<*m;i++) {
    x = (*we)[i].x;
    y = (*we)[i].y;
    v = (*we)[i].w;
      
    if ((x == 0) || (y == 0)) {
      printf("edge %d: %d and %d are neighbors, numbering starts from 1! \n",i,x,y);
      return(false);
    }

    (*where)[start[x]] = start[y];
    (*where)[start[y]] = start[x];

    (*edges)[start[x]] = y;
    (*weight)[start[x]] = v;
    start[x]++;

    (*edges)[start[y]] = x;
    (*weight)[start[y]] = v;
    start[y]++;

  }

  
  fclose(fp);
  free(start);
  free(count);
  

  return(true);
}

int read_graph(int *n,int *m,char *f_name,int **ver,int **edges,edge **e,
               double **weight,wed **we,neig **swe,int *count, int *start,int **where) {

// Note reuse of p as count and p1 as start

  FILE *fp;
  long int i;
  int x,y,z;
  double v,w;
  MM_typecode matcode;
  int cols;

  int mxdeg = 0;
  //int *count;
  //int *start;

  // printf("Opening file %s \n",f_name);

// First make sure graph is of the right type

  fp = fopen(f_name,"r");
  if (fp == NULL) {
    printf("Could not open file %s \n",f_name);
    return(false);
  }

  if (mm_read_banner(fp, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    return(false);
  }

  if ((mm_read_mtx_crd_size(fp, &cols, n, m)) !=0) {
    printf("Could not read size of graph.\n");
    return(false);
  }

  if (*m > max_m) {
    printf("Graph is too large. Asking for %d edges, max is %d \n",*m,max_m);
    return(false);
  }
  if (*n > max_n) {
    printf("Graph is too large. Asking for %d vertices, max is %d \n",*n,max_n);
    return(false);
  }

  if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode) || !mm_is_sparse(matcode) || !mm_is_symmetric(matcode)) {
    printf("The program can only read files that are sparse symmmetric matrices in coordinate format! \n");
    return(false);
  }


// Start counting of degrees by setting counters to zero
  for(i=1;i<=*n;i++) {
    count[i] = 0;
  }

  int num_edges = 0;

//  printf("Number of possible edges is %d \n",*m);
  
// Read inn the edges
  if (mm_is_real(matcode)) {
    // printf("Real matrix, Starting to read %d edges \n",*m);
    srand48(time(NULL));
    for(i=0;i<*m;i++) {
      fscanf(fp,"%d %d %lf",&x,&y,&v); // Use this line if there is exactly one double weight 
    //  fscanf(fp,"%d %d %lf %lf",&x,&y,&v,&w); // Use this line for complex weights
      if (x != y) { // Avoid self-edges
/*
        (*e)[num_edges].x = x; // Store edges
        (*e)[num_edges].y = y;
*/

        (*we)[num_edges].x = x;
        (*we)[num_edges].y = y;
//        int intv = (int) fabs(v);
//        (*we)[num_edges].w = (double) intv; 
        (*we)[num_edges].w = fabs(v);     // Take absolute value of edge
    
        if (fabs(v) < 0.0)  {
          printf("error reading data,got %f \n",fabs(v));
          return;
        }
         (*we)[num_edges].w = 1.0 + (double)rand()/100000.0;
      //  (*we)[num_edges].w = (double)rand();
        count[x]++; // Get vertex degrees
        count[y]++;

        num_edges++;
        if (num_edges > *m) {
          printf("Have set num_edges to %d while i=%d \n",num_edges,i);
          return;
        }
      }
    }
  }
  else if (mm_is_integer(matcode)) {
    printf("Integer matrix, Starting to read %d edges \n",*m);
    srand48(time(NULL));
    for(i=0;i<*m;i++) {
      fscanf(fp,"%d %d %lf",&x,&y,&v); // Use this line if there is exactly one double weight 
    
      if (x != y) { // Avoid self-edges
/*
        (*e)[num_edges].x = x; // Store edges
        (*e)[num_edges].y = y;
*/

        (*we)[num_edges].x = x;
        (*we)[num_edges].y = y;
        (*we)[num_edges].w = fabs(v) + drand48()/1000.0;     // Take absolute value of edge

        if (fabs(v) < 0.0)  {
          printf("error reading data,got %f \n",fabs(v));
          return;
        }

        count[x]++; // Get vertex degrees
        count[y]++;

        num_edges++;
        if (num_edges > *m) {
          printf("Have set num_edges to %d while i=%d \n",num_edges,i);
          return;
        }
      }
    }
  }
  else
  {          // Symbolic matrix
//    srand48(time(NULL));
    srand48(0);
    printf("Symbolic matrix \n");
    for(i=0;i<*m;i++) {
      fscanf(fp,"%d %d",&x,&y);
      if (x != y) { // Avoid self-edges
/*
        (*e)[num_edges].x = x; // Store edges
        (*e)[num_edges].y = y;
*/

        (*we)[num_edges].x = x;
        (*we)[num_edges].y = y;

        (*we)[num_edges].w = drand48();  // Using random value

        count[x]++; // Get vertex degrees
        count[y]++;

        num_edges++;
      }
    }
  }
//   printf("original edges %d, used edges %d \n",*m,num_edges);
  printf("*  |V| = %d, |E| = %d \n",*n,num_edges);
  printf("************************************************\n");

  *m = num_edges;  // Make sure m is the correct number of edges

// Find starting positions in edge list for each vertex
  start[1] = 0;
  (*ver)[1] = 0;
  for(i=2;i<=*n+1;i++) {
    start[i] = start[i-1]+count[i-1];
    (*ver)[i] = start[i];
  }

// Place edges in edge lists, once for each endpoint

  for(i=0;i<*m;i++) {
    x = (*we)[i].x;
    y = (*we)[i].y;
    v = (*we)[i].w;

    if ((x == 0) || (y == 0)) {
      printf("edge %d: %d and %d are neighbors, numbering starts from 1! \n",i,x,y);
      return(false);
    }

    (*where)[start[x]] = start[y];
    (*where)[start[y]] = start[x];

    (*edges)[start[x]] = y;
    (*weight)[start[x]] = v;
    start[x]++;

    (*edges)[start[y]] = x;
    (*weight)[start[y]] = v;
    start[y]++;

  }


  int maxnode;
//  printf("ver1 = %d ver2=%d ver3=%d \n",(*ver)[1],(*ver)[2],(*ver)[3]);
  for(i=1;i<=*n;i++) {
    if (((*ver)[i+1] - (*ver)[i]) > mxdeg) {
      mxdeg = (*ver)[i+1] - (*ver)[i];
      maxnode = i;
    }
  }
/*
  printf("Vertex %d has maximum degree %d \n",maxnode,mxdeg);
  printf("Average degree is %d \n",((*m) * 2)/(*n));
*/


//  printf("Done read_graph, freeing up memory \n");
  fclose(fp);
  // free(count);
  // free(start);
  return(true);
}

int read_general_graph(int *n,int *m,char *f_name,int **ver,int **edges,edge **e,
                       double **weight,wed **we,neig **swe,int *count, int *start) {

// Note reuse of p as count and p1 as start

  FILE *fp;
  long int i;
  int x,y,z;
  double v,w;
  MM_typecode matcode;
  int cols;

  //int *count;
  //int *start;

  // printf("Opening file %s \n",f_name);

// First make sure graph is of the right type

  fp = fopen(f_name,"r");
  if (fp == NULL) {
    printf("Could not open file %s \n",f_name);
    return(false);
  }

  if (mm_read_banner(fp, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    return(false);
  }

  if ((mm_read_mtx_crd_size(fp, &cols, n, m)) !=0) {
    printf("Could not read size of graph.\n");
    return(false);
  }

// If this is an unsymmetric matrix then we only keep the square part.
  if (cols < *n)
    *n = cols;

/*
  if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode) || !mm_is_sparse(matcode) || !mm_is_symmetric(matcode)) {
    printf("The program can only read files that are sparse symmmetric matrices in coordinate format! \n");
    return(false);
  }
*/


// Start counting of degrees by setting counters to zero
  for(i=1;i<=*n;i++) {
    count[i] = 0;
  }

  int num_edges = 0;

//  printf("Number of possible edges is %d \n",*m);
  
// Read inn the edges
  if (mm_is_real(matcode)) {
//    printf("Starting to read %d edges \n",*m);
    for(i=0;i<*m;i++) {
      fscanf(fp,"%d %d %lf",&x,&y,&v); // Use this line if there is exactly one double weight 
      // fscanf(fp,"%d %d %lf %lf",&x,&y,&v,&w); // Use this line for complex weights
      if ((x < y) && (x <= *n) && (y <= *n)) { // Only use edges in square part
        (*e)[num_edges].x = x; // Store edges
        (*e)[num_edges].y = y;

        (*we)[num_edges].x = x;
        (*we)[num_edges].y = y;
//        int intv = (int) fabs(v);
//        (*we)[num_edges].w = (double) intv; 
        (*we)[num_edges].w = fabs(v);
//        (*we)[num_edges].w = drand48();
        if (fabs(v) < 0.0)  {
          printf("error reading data,got %f \n",fabs(v));
          return;
        }
      //   (*we)[num_edges].w = 1.0;
      //  (*we)[num_edges].w = (double)rand();

        count[x]++; // Get vertex degrees
        count[y]++;

        num_edges++;
        if (num_edges > *m) {
          printf("Have set num_edges to %d while i=%d \n",num_edges,i);
          return;
        }
      }
    }
  }
  else {          // Symbolic matrix
    printf("Trouble ahead, the code now assumes weighted graphs \n");
    for(i=0;i<*m;i++) {
      fscanf(fp,"%d %d",&x,&y);
      if (x != y) { // Avoid self-edges
        (*e)[num_edges].x = x; // Store edges
        (*e)[num_edges].y = y;

        count[x]++; // Get vertex degrees
        count[y]++;

        num_edges++;
      }
    }
  }
//   printf("original edges %d, used edges %d \n",*m,num_edges);
  printf("*  |V| = %d, |E| = %d \n",*n,num_edges);
  printf("************************************************\n");

  *m = num_edges;  // Make sure m is the correct number of edges

// Find starting positions in edge list for each vertex
  start[1] = 0;
  (*ver)[1] = 0;
  for(i=2;i<=*n+1;i++) {
    start[i] = start[i-1]+count[i-1];
    (*ver)[i] = start[i];
  }

// Place edges in edge lists, once for each endpoint

  for(i=0;i<*m;i++) {
    x = (*we)[i].x;
    y = (*we)[i].y;
    v = (*we)[i].w;
    if ((x == 0) || (y == 0)) {
      printf("edge %d: %d and %d are neighbors, numbering starts from 1! \n",i,x,y);
      return(false);
    }

    (*edges)[start[x]] = y;
    (*weight)[start[x]] = v;
    start[x]++;

    (*edges)[start[y]] = x;
    (*weight)[start[y]] = v;
    start[y]++;

  }
//  printf("Done read_graph, freeing up memory \n");
  fclose(fp);
  // free(count);
  // free(start);
  return(true);
}

// Get input. This is given as a file name to the executable. 
// This file should contain the following information (per line):
// Number of input graphs
// Number of times each configuration is run, the program reports the best running time out of these
// Number of configurations, followed by the number of threads in each configuration
// One line giving the complete name of each graph file that is to be run

int get_input(int argc, char *argv[],int *n_graphs,int *n_runs,int *n_conf,int **conf,char ***name) {

  FILE *rf;	// File pointer
  int i;

  if (argc != 2) {
    printf("Give data file name as first input parameter!\n");
    return(false);
  }

  printf("************************************************\n");
  printf("*       Reading setup from %10s          *\n",argv[1]);
  printf("************************************************\n");

// Opening file containing file names for graphs

  rf = fopen(argv[1],"r");
  if (rf == NULL) {
    printf("Cannot open data file: %s \n",argv[1]);
    return(false);
  }


// Get number of graphs to read
  fscanf(rf,"%d",n_graphs);

  *name =  malloc(sizeof(char *)*(*n_graphs));  // Allocate one pointer for each graph 

  if (*name == NULL) {
    printf("Unable to allocate space for names of %d graphs\n",max_graphs);
    return(0);
  }

// Get number of runs per configuration
  fscanf(rf,"%d",n_runs);

// Get number of thread configurations
  fscanf(rf,"%d",n_conf);
  *conf = (int *) malloc(sizeof(int)*(*n_conf));  // Allocate space for configurations


  if (*conf == NULL) {
    printf("Unable to allocate memory for %d different thread configurations \n",n_conf);
    return(0);
  }

// Get the different configurations
  for(i=0;i< *n_conf;i++) {
    fscanf(rf,"%d",&(*conf)[i]);
  }

// Get the different file names
  for(i=0;i< *n_graphs;i++) {
    (*name)[i] = (char *) malloc(sizeof(char)*100);
    if ((*name)[i] == NULL) {
      printf("Unable to allocate memory for graph name %d \n",i);
      return(0);
    }
// Read name of graph
    fscanf(rf,"%s",(*name)[i]);
  }

  fclose(rf);
  return(true);
}


int allocate_memory(int size,int **n,int **m,double ***timer,double ***cost,double ****p_timer,double ****p_cost,int n_conf) {


  *n = (int *) malloc(sizeof(int)*size);  // List holding the number of vertices in each graph

  if (*n == NULL) {
    printf("Unable to allocate memory for n[] in allocate_memory() \n");
    return(false);
  }

  *m = (int *) malloc(sizeof(int)*size);  // List holding the number of edges in each graph
  if (*m == NULL) {
    printf("Unable to allocate memory for m[] in allocate_memory() \n");
    return(false);
  }

  *timer = malloc(sizeof(double *)*(max_experiment));  // Allocate one pointer for each sequential experiment
  if (*timer == NULL) {
    printf("Unable to allocate memory for timer in allocate_memory() \n");
    return(false);
  }

  *cost = malloc(sizeof(double *)*(max_experiment));  // Allocate one pointer for each sequential experiment
  if (*timer == NULL) {
    printf("Unable to allocate memory for cost in allocate_memory() \n");
    return(false);
  }

  // For each experiment, allocate one number for each graph
  int i;
  for(i=0;i<max_experiment;i++) {
    (*cost)[i] = (double *) malloc(sizeof(double)*size);
    if ((*cost)[i] == NULL) {
      printf("Unable to allocate memory for cost %d in allocate_memory() \n",i);
      return(false);
    }
    (*timer)[i] = (double *) malloc(sizeof(double)*size);
    if ((*timer)[i] == NULL) {
      printf("Unable to allocate memory for timer %d in allocate_memory() \n",i);
      return(false);
    }
    int j;
    for(j=0;j<size;j++)          // Set each timer to -1
      (*timer)[i][j] = -1.0;
  }

  *p_timer = malloc(sizeof(double **)*(max_experiment));  // Allocate one pointer for each parallel experiment
  *p_cost  = malloc(sizeof(double **)*(max_experiment));  // Allocate one pointer for each parallel experiment

  if (*p_timer == NULL) {
    printf("Unable to allocate memory for p_timer in allocate_memory() \n");
    return(false);
  }
  if (*p_cost == NULL) {
    printf("Unable to allocate memory for p_cost in allocate_memory() \n");
    return(false);
  }
  // For each experiment, allocate "size" pointers for each graph
  for(i=0;i<max_experiment;i++) {
    (*p_timer)[i] = (double **) malloc(sizeof(double *)*size);
    (*p_cost)[i]  = (double **) malloc(sizeof(double *)*size);
    if ((*p_timer)[i] == NULL) {
      printf("Unable to allocate memory for p_timer %d in allocate_memory() \n",i);
      return(false);
    }
    if ((*p_cost)[i] == NULL) {
      printf("Unable to allocate memory for p_cost %d in allocate_memory() \n",i);
      return(false);
    }
    int j;
    for(j=0;j<size;j++) {
      (*p_timer)[i][j] = (double *)  malloc(sizeof(double)*n_conf);
      (*p_cost)[i][j] = (double *)  malloc(sizeof(double)*n_conf);
      if ((*p_timer)[i][j] == NULL) {
        printf("Unable to allocate memory for p_timer %d in allocate_memory(),%d \n",i,j);
        return(false);
      }
      if ((*p_cost)[i][j] == NULL) {
        printf("Unable to allocate memory for p_cost %d,%d in allocate memory \n",i,j);
        return(false);
      }
      int k;
      for(k=0;k<n_conf;k++) {
        (*p_timer)[i][j][k] = -1.0;
        (*p_cost)[i][j][k] = -1.0;
      }
    }
  }
  return(true);
}


int prepare_output(FILE **wf,int n_conf,int conf[]) {

// Open file for writing of results
  *wf = fopen("results.m","w");
  if (*wf == NULL) {
    printf("Unable to open results.m for writing \n");
    return(false);
  }

// print the thread configurations to file
  fprintf(*wf,"x = [");
  int i;
  for(i=0;i<n_conf;i++) 
    fprintf(*wf,"%d ",conf[i]);
  fprintf(*wf,"];\n");

  return(true);
}

// Verify that p[] defines a legal matching

int verify_Wmatching(int n,int *ver,int *edges,int *p,double *ws) {

  int i;

  for(i=1;i<=n;i++) {
    if ((p[i] < 0) || (p[i] > n)) {
      printf("p[%d] = %d, while n = %d \n",i,p[i],n);
      return(false);
    }
    if ((p[i] != 0) && (p[p[i]] != i)) {
      printf("p[%d] = %d, while p[%d] = %d \n",i,p[i],p[i],p[p[i]]); 
      return(false);
    }
    if ((p[i] != 0) && (ws[p[i]] != ws[i])) {
      printf("p[%d] = %d, and p[%d] = %d, but ws[%d]=%lf and ws[%d]=%lf \n",i,p[i],p[i],p[p[i]],i,ws[i],p[i],ws[p[i]]); 
      return(false);
    }
  }
  return(true);
}
// Verify that p[] defines a legal matching

int verify_matching(int n,int *ver,int *edges,int *p) {

  int i;

  for(i=1;i<=n;i++) {
    if ((p[i] < 0) || (p[i] > n)) {
      printf("p[%d] = %d, while n = %d \n",i,p[i],n);
      return(false);
    }
    if ((p[i] != 0) && (p[p[i]] != i)) {
      printf("p[%d] = %d, while p[%d] = %d \n",i,p[i],p[i],p[p[i]]); 
      return(false);
    }
  }
  return(true);
}

// Compute a random ordering of the vertices

int random_order(int n,int *order) {

  int l;

// Compute a random ordering of the vertices
  for(l=1;l<=n;l++) {
    order[l] = l;
  }
  for(l=1;l<n;l++) {
    long int x = random() % (long int) (n-l+1);
    x++;
    int tmp = order[n-l+1];
    order[n-l+1] = order[x];
    order[x] = tmp;
  }
  return(true);
}

// Allocating memory specifically for this graph

int allocate_graph_memory(neig **swe,double **weight,edge **e,wed **we,int **ver,int **edges,int n,int **next,int **used,int **p,int **p1,int **p2,int **p3) {		

// The compressed edge list with the weight, needed for sorting using qsort()

  long temp_long = 2;
  temp_long = temp_long * max_m * sizeof(neig);

  // The edge list with the weight
  printf("MAX_M = %d \n",max_m); fflush(stdout);


  *swe = (neig *) malloc(temp_long);
  if (*swe == NULL) {
    printf("Unable to allocate space for swe-array in allocate_graph_memory() \n");
    return(false);
  }

  *we = (wed *) malloc(max_m * sizeof(wed));
  if (*we == NULL) {
    printf("Unable to allocate space for we-array in allocate_graph_memory() \n");
    return(false);
  }


// 'weight' contains the weight of each edge, stored in the same way as the edge-lists
  *weight= (double *) malloc(2*sizeof(double)*(max_m));
  if (*weight== NULL) {
    printf("Unable to allocate space for weight-array in allocate_graph_memory() \n");
    return(false);
  }
// The raw edge list is stored in e, i.e. in the same order as it was read in

/*  This part is removed to ensure space for parallel algorithms.
 *  Must be returned to run unweighted code
 
  *e = (edge *) malloc(sizeof(edge)*(max_m));
  if (*e == NULL) {
    printf("Unable to allocate space for e-array in allocate_graph_memory() \n");
    return(false);
  }
*/

  *ver = (int *) malloc(sizeof(int)*(2+n));
  if (*ver == NULL) {
    printf("Unable to allocate space for vertex-array in allocate_graph_memory() \n");
    return(false);
  }

  *edges = (int *) malloc(2*sizeof(int)*(max_m));
  if (*edges == NULL) {
    printf("Unable to allocate space for edges-array in allocate_graph_memory \n");
    return(false);
  }

  *p = (int *) malloc(sizeof(int)*(n+2));	// Storage for final matching
  if (*p == NULL) {
    printf("Unable to allocate space for p[] in allocate_graph_memory() \n");
    return(false);
  }

  *p1 = (int *) malloc(sizeof(int)*(n+2));	// Storage for first matching in two level algorithms
  if (*p1 == NULL) {
    printf("Unable to allocate space for p1[] in allocate_graph_memory() \n");
    return(false);
  }

  *p2 = (int *) malloc(sizeof(int)*(n+2));	// Storage for second matching in two level algorithms
  if (*p2 == NULL) {
    printf("Unable to allocate space for p2[] in allocate_graph_memory() \n");
    return(false);
  }

  *p3 = (int *) malloc(sizeof(int)*(n+2));	// Even more storage...
  if (*p3 == NULL) {
    printf("Unable to allocate space for p3[] in allocate_graph_memory() \n");
    return(false);
  }

  *next = (int *) malloc(sizeof(int)*n*2);	// Pointers used in suitor based algorithm with sorted edge lists
  if (*next == NULL) {
    printf("Unable to allocate space for next[] in allocate_graph_memory() \n");
    return(false);
  }

  *used = (int *) malloc(sizeof(int)*(n+2));
  if (*used == NULL) {
    printf("Unable to allocate space for used[] in allocate_graph_memory() \n");
    return(false);
  }
  return(true);
}

// Calculate the cost of the matching

double cost_and_correct_matching(int n,int *ver,int *edges,double *weight,int *match,double *ws) {

  double glob_sum = 0.0;
  int i,k;
  
  for(i=1;i<=n;i++) {
    //printf("%d is matched with %d \n",i,match[i]);
    if ((match[i] != 0) && (i < match[i])) { // Only use vertices that are matched and that have lower index then their partner
      for(k=ver[i];k<ver[i+1];k++) {         // Loop through neighbors of vertex i
        if (edges[k] == match[i]) {
          glob_sum += weight[k];
          ws[i] = weight[k];
          ws[match[i]] = weight[k];
          if (match[edges[k]] != i) 
            printf("Error in cost_matching: %d is matched with %d but %d is matched with %d \n",i,match[i],edges[k],match[edges[k]]);
        } // if
      } // for k
    } // if
  } // for i
 
  return(glob_sum);
}

// Calculate the cost of the matching

double cost_matching(int n,int *ver,int *edges,double *weight,int *match) {

  double glob_sum = 0.0;
  int i,k;
  
  for(i=1;i<=n;i++) {
    //printf("%d is matched with %d \n",i,match[i]);
    if ((match[i] != 0) && (i < match[i])) { // Only use vertices that are matched and that have lower index then their partner
      for(k=ver[i];k<ver[i+1];k++) {         // Loop through neighbors of vertex i
        if (edges[k] == match[i]) {
          glob_sum += weight[k];
          if (match[edges[k]] != i) 
            printf("Error in cost_matching: %d is matched with %d but %d is matched with %d \n",i,match[i],edges[k],match[edges[k]]);
        } // if
      } // for k
    } // if
  } // for i
 
  return(glob_sum);
}
void store_p_value(FILE *wf,char *s,double **values,int n,int n_conf) {
  int i,j;
  fprintf(wf,"%s = [",s); 
  for(i=0;i<n;i++) {
    for(j=0;j<n_conf;j++)  {
      fprintf(wf,"%lf ",values[i][j]);
    }
  }
  fprintf(wf,"];\n");
}

void store_value(FILE *wf,char *s,double *values,int n) {
  int i;
  fprintf(wf,"%s = [",s); 
  for(i=0;i<n;i++)  {
    fprintf(wf,"%lf ",values[i]);
  }
  fprintf(wf,"];\n");
}
