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

int main(int argc, char *argv[]) {

	/* @OGUZ-EDIT numa disabled */
  /* if (numa_available() < 0) { */
  /*   printf("No NUMA support available on this system.\n"); */
  /*   exit(1); */
  /* } */
  /* numa_set_interleave_mask(numa_all_nodes_ptr); */


// Primary graph data structure, using compressed neighbor lists.
// ver stores pointers into the edges list. Note that numbering of both vertices and edges both start at 1

  int *ver;          // Pointers to edge lists
  int *edges;        // Edge lists
  double *weight;    // Corresponding edge weights

// Edge lists where each neighbor list has been sorted by decreasing weight

  int *s_edges;      // Sorted edge lists
  double *s_weight;  // Sorted corresponding edge weights

// Vertex lists used in various algorithms

  int *next;         
  int *used;        // Keeps track of which vertices has been used in the dynamic programming

// Vertex lists used for holding the matching

  int *p;	     // If p(i) = j then i is matched with j, this is used in all the algorithms
  int *p1;           // Same as p, but used for first round of two round algorithms
  int *p2;           // Same as p, but used for second round of two round algorithms
  int *p3;           // Same as p, but used for second round of two round algorithms

// Vertex lists for suitor algorithm

  double *ws1;       // If ws1(i) = w, then the weight of the best suitor of i is w
  double *ws2;       // Same as ws1, but used in two round algorithm

// Raw edge lists in the order it was read from file

  edge *e;           // List of all edges, no weight, one struct for each edge
  wed *we;	     // Same as e, but now with weights
  neig *swe;	     // List of neighbor + weight 

//  omp_lock_t *nlocks;
  int tlist[max_threads];  // List for storing data for each thread (in plocalmax)

  int read_graph(),get_input(),allocate_memory(),read_general_graph();
  int read_bin_graph();
  double cost_matching();
  double cost_and_correct_matching();
  void store_value();
  void store_p_value();

  int i,j;          // Loop indices
  double mt1,mt2;   // Used for timing of each algorithm
  int xx;
  int UGreedyVertex,UGreedyEdge;
  int WGreedyVertex,WRandomGreedyVertex,WInorderGreedyVertex;
  int WPrefetchRandomGreedyVertex,WLocalMaxSequential;
  int WSuitorSequential,WSuitorCompEdgeSequential;
  int WSuitorWhileSequential,SortNeighborListsSequential;
  int WSuitorSortSequential,WSuitorSortWhileSequential;
  int W2RSuitorDPSequential,W2RSuitorDPCycSequential;
  int W2RGreedyDPSequential,SortEdgesSequential;
  int WGPASequential,WPGASequential;
  int W2RPGADPSequential,W2RSuitorDPSortEdgeSequential;
  int UGreedyVertexVerifyParallel;
  int UGreedyEdgeVerifyParallel;
  int UGreedyEdgeVerifyPrefetchParallel;
  int UGreedyEdgeLockParallel;
  int WSuitorStackParallel;
  int WSuitorParallel;
  int WStackParallel;
  int WQueueParallel;
  int WLocalMax1ListParallel;
  int WLocalMaxLocalListParallel;
  int WSuitorSortParallel;
  int W2RSuitorDPParallel;
  int W2RSuitorSortParallel;
  int SortEdgeListsParallel;
  int WGreedySequential;
  int WMB1ListParallel;
  int WMBLocalParallel;
  int WBmatching;
  int WBcmatching;
  int W2MatchingParallel;
  int WSuitorStackSequential;
  int WSuitorQueueSequential;
  int WSuitorDQueueSequential;
  int WRama;
  int WRoma;
  int WRema;

// Unweighted sequential algorithms
  UGreedyVertex                 = false;   /* Unweighted Greedy Vertex-based  */
  UGreedyEdge                   = false;   /* Unweighted Greedy Edge-based  */

// Weighted sequential algorithms
  WGreedyVertex                 = false;   /* Weighted Greedy Vertex-based  */
  WRandomGreedyVertex           = false;   /* Weighted Greedy Vertex-based, but in random order */
  WInorderGreedyVertex          = false;   /* Weighted Greedy Vertex-based, but in order */
  WPrefetchRandomGreedyVertex   = false;   /* Weighted Greedy Vertex-based with random order and prefetching */

  WLocalMaxSequential           = false;   /* *Weighted Round based dominant edge (Birn et al.) */
  WSuitorSequential             = true;    /* *Weighted Standard suitor, using a for loop outermost */
  WRama                         = false;    /* *Rama type post-processing */
  WRoma                         = false;    /* *Roma type post-processing */
  WRema                         = false;   /* *Rema type post-processing */
  WSuitorCompEdgeSequential     = false;   /* Weighted suitor with compressed edge lists, will destroy edge and weight lists */
  WSuitorWhileSequential        = false;   /* Weighted suitor with while loop as outermost loop */
  WSuitorStackSequential        = false;    /* Weighted suitor using a stack as outer loop */
  WSuitorQueueSequential        = false;    /* Weighted suitor using a circular queue as outer loop */
  WSuitorDQueueSequential       = false;    /* Weighted suitor using a queue as outer loop, implemented with two arrays */
  SortNeighborListsSequential   = false;   /* *Sort neighbor lists using qsort */
  WSuitorSortSequential         = false;   /* *Weighted suitor with sorted neighbor lists */
  WSuitorSortWhileSequential    = false;   /* Weighted suitor with sorted neighbor lists + outer while loop */

  W2RSuitorDPSequential         = true;   /* Weighted 2 round suitor w/ one routine for DP */
  W2RSuitorDPCycSequential      = false;    /* *Weighted 2 round suitor w/separate routines for path DP and cycle DP */
  W2RGreedyDPSequential         = false;   /* *Weighted 2 round greedy w/DP */
  SortEdgesSequential           = true;    /* *Sort all the edges into one list */
  WPGASequential                = true;   /* *Path growing followed by DP */
  WGPASequential                = true;   /* *Weighted GPA algorithm */
  W2RPGADPSequential            = false;   /* *Weighted 2 round PGA followed by DP */
  W2RSuitorDPSortEdgeSequential = false;   /* *Weighted 2 round Suitor + DP, assuming sorted edge lists */
  WGreedySequential             = true;   /* *Weighted greedy on sorted edge list */
  WBmatching                    = false;    /* 2-matching followed by DP */
  WBcmatching                   = false;    /* 2-matching followed by DP  using compressed neighbor lists*/
  W2MatchingParallel            = false;

  UGreedyVertexVerifyParallel       = false; /* Unweighted Greedy Vertex Verification Parallel */
  UGreedyEdgeVerifyParallel         = false; /* Unweighted Greedy Edge Verification Parallel */
  UGreedyEdgeVerifyPrefetchParallel = false; /* Unweighted Greedy Edge Verification w/prefetch Parallel */
  UGreedyEdgeLockParallel           = false; /* Unweighted Greedy Edge Locking Parallel */

  WSuitorStackParallel              = false; /* Weighted Suitor using stack Parallel */
  WSuitorParallel                   = true; /* Weighted Suitor Parallel (standard) */
  WStackParallel                    = false; /* Weighted Suitor Parallel (using stack) */
  WQueueParallel                    = false; /* Weighted Suitor Parallel (using queue) */
  WLocalMax1ListParallel            = false; /* Weighted Local Max Parallel w/1 global list */
  WLocalMaxLocalListParallel        = false; /* Weighted Local Max Parallel w/local lists */
  WSuitorSortParallel               = false; /* Weighted Suitor w/sorted adjacency lists Parallel */
  W2RSuitorDPParallel               = false; /* Weighted 2 round Suitor w/DP Parallel */
  W2RSuitorSortParallel             = false; /* Weighted 2 round Suitor w/DP + sorted adj lists Parallel */
  SortEdgeListsParallel             = false; /* Sort neighbor lists in parallel */
  WMB1ListParallel                  = false;  /* Weighted Manne-Bisseling alg w/1 global list */
  WMBLocalParallel                  = false;  /* Weighted Manne-Bisseling alg w/local lists */

  if (!get_input(argc,argv,&n_graphs,&n_runs,&n_conf,&conf,&name))	// Get input and check that it is in order
    return(false);

  if (!prepare_output(&wf,n_conf,conf)) {				// Prepare output file
    return(false);
  }

  if (!allocate_memory(n_graphs,&n,&m,&timer,&cost,&p_timer,&p_cost,n_conf)) {		// Allocating memory
    return(false);
  }



// Run the main loop, processing each graph, first sequentially and then in parallel

  if (!allocate_graph_memory(&swe,&weight,&e,&we, &ver,&edges,max_n,&next,&used,&p,&p1,&p2,&p3)) {			// Allocating memory specifically for this graph
    return(false);
  }

  int *where;
  where = (int *) malloc(2*sizeof(int)*(max_m));
  if (where == NULL) {
    printf("Unable to allocate space for where array \n");
    return(false);
  }

  ws1 = (double *) malloc(sizeof(double)*(max_n+1));
  ws2 = (double *) malloc(sizeof(double)*(max_n+1));
  if ((ws1 == NULL) || (ws2 == NULL)) {
    printf("Unable to allocate space for ws-arrays \n");
    return(false);
  }
/*
  s_edges = (int *) malloc(sizeof(int)*max_m);
  if (s_edges == NULL) {
    printf("Unable to allocate memory for s_edges in wframe3 \n");
    return;
  }
  s_weight = (double *) malloc(sizeof(double)*max_m);
  if (s_weight == NULL) {
    printf("Unable to allocate memory for s_weight in wframe3 \n");
    return;
  }
*/
//  nlocks = (omp_lock_t *) malloc((n[i]+1)*sizeof(omp_lock_t));

  for(i=0;i<n_graphs;i++) {

    printf("\n");
    printf("************************************************\n");
    printf("*  Graph %2d: %20s             *\n",i,name[i]);


// Reading the i'th graph

    if (!read_graph(&(n[i]),&(m[i]),name[i],&ver,&edges,&e,&weight,&we,&swe,p,p1,&where)) {
      printf("Problem reading graph %s \n",name[i]);
      return(false);
    }

/*

    printf("ver1 = %d, ver2= %d ver3=%d \n",ver[1],ver[2],ver[3]);



//
    if (!read_general_graph(&(n[i]),&(m[i]),name[i],&ver,&edges,&e,&weight,&we,&swe,p,p1)) {
      printf("Problem reading graph %s \n",name[i]);
      return(false);
    }


    if (!read_bin_graph(&(n[i]),&(m[i]),name[i],&ver,&edges,&e,&weight,&we,&swe,&where)) {
      printf("Problem reading graph %s \n",name[i]);
      return(false);
    }
*/


    printf("*  |V| = %9d, |E| = %10d           *\n",n[i],m[i]);
    printf("************************************************\n");

//    printf("Done reading graph \n");
/*
    int xxx = ver[n[i]+1];
    printf("Asking for %d integers \n",xxx);
    s_edges = (int *) malloc(sizeof(int)*xxx);
    printf("Got %d integers \n",xxx);
    if (s_edges == NULL) {
      printf("Unable to allocate memory for s_edges in wframe2 \n");
      return;
    }
    printf("Asking for %d doubles \n",ver[n[i]+1]);
    s_weight = (double *) malloc(sizeof(double)*(ver[n[i]+1]));
    if (s_weight == NULL) {
      printf("Unable to allocate memory for s_weight in wframe2 \n");
      return;
    }
    printf("Got them \n");
 */   

/*
    if (!allocate_graph_memory(n[i],&next,&used,&p,&p1,&p2)) {			// Allocating memory specifically for this graph
      return(false);
    }
*/


// List of best local suitor for each vertex in the weighted algorithm
// Why is this not in allocate_graph_memory?
/*
    ws1 = (double *) malloc(sizeof(double)*((*n)+1));
    ws2 = (double *) malloc(sizeof(double)*((*n)+1));
    if ((ws1 == NULL) || (ws2 == NULL)) {
      printf("Unable to allocate space for ws-arrays \n");
      return(false);
    }
*/

// Run sequential code 

    printf("\n************************************************\n");
    printf("***           Sequential algorithms          ***\n");
    printf("************************************************\n");

    int l;
    printf("***                                          ***\n");
    printf("***           Unweighted algorithms          ***\n");
    printf("***                                          ***\n");

    if (UGreedyVertex) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        seq1(n[i],ver,edges,p);	// Sequential unweighted greedy vertex based matching
        mt2 = omp_get_wtime(); 
        if ((timer[0][i] < 0) || (mt2-mt1 < timer[0][i]))
          timer[0][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("seq1() did not produce a legal matching \n");
        exit(false);
      }

      printf("************************************************\n");
      printf("*  Unweighted greedy vertex based              *\n");
      printf("*  Time = %8.6lf seconds                     *\n",timer[0][i]);
      printf("************************************************\n");
    }


    if (UGreedyEdge) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        eseq1(n[i],m[i],e,p);	// Sequential unweighted greedy edge based matching
        mt2 = omp_get_wtime(); 
        if ((timer[1][i] < 0) || (mt2-mt1 < timer[1][i]))
          timer[1][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("eseq1() did not produce a legal matching \n");
        exit(false);
      }

      printf("************************************************\n");
      printf("*  Unweighted greedy edge based                *\n");
      printf("*  Time = %8.6lf seconds                     *\n",timer[1][i]);
      printf("************************************************\n");
    }

    printf("***                                          ***\n");
    printf("***            Weighted algorithms           ***\n");
    printf("***                                          ***\n");

    if (WGreedyVertex) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        wseq1(n[i],ver,edges,p,weight); // Sequential weighted localy greedy matching, vertex based
        mt2 = omp_get_wtime(); 
        if ((timer[2][i] < 0) || (mt2-mt1 < timer[2][i]))
          timer[2][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("wseq1() did not produce a legal matching \n");
        exit(false);
      }

      cost[2][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Greedy vertex based matching                *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[2][i],cost[2][i]);
      printf("************************************************\n");
    }

// Compute a random ordering of the vertices
// This is used to traverse the vertices of the graph in a random order


    if (WRandomGreedyVertex) {
      if (!random_order(n[i],next)) {
        printf("Problem in random_order() \n");
        exit(false);
      }

      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        wseq2(n[i],ver,edges,p,weight,next); // Sequential weighted localy greedy matching, vertex based, but in random order
        mt2 = omp_get_wtime(); 
        if ((timer[3][i] < 0) || (mt2-mt1 < timer[3][i]))
          timer[3][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("wseq2() did not produce a legal matching \n");
        exit(false);
      }

      cost[3][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Random Greedy vertex based matching         *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[3][i],cost[3][i]);
      printf("************************************************\n");
    }


    if (WInorderGreedyVertex) {
// Do the same as above but now in-order

      for(l=1;l<n[i];l++)
        next[l] = l;

      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime();
        wseq2(n[i],ver,edges,p,weight,next); // Sequential weighted localy greedy matching, vertex based
        mt2 = omp_get_wtime();
        if ((timer[21][i] < 0) || (mt2-mt1 < timer[21][i]))
          timer[21][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("wseq2() did not produce a legal matching \n");
        exit(false);
      }

      cost[21][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching


      printf("************************************************\n");
      printf("*  Local maximal weight matching               *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[21][i],cost[21][i]);
      printf("************************************************\n");
    }


    if (WPrefetchRandomGreedyVertex) {

      if (!random_order(n[i],next)) {
        printf("Problem in random_order() \n");
        exit(false);
      }

      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        wseq3(n[i],ver,edges,p,weight,next); // Same as random order, but code has been optimized with prefetching directives
        mt2 = omp_get_wtime(); 
        if ((timer[4][i] < 0) || (mt2-mt1 < timer[4][i]))
          timer[4][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("wseq3() did not produce a legal matching \n");
        exit(false);
      }
      cost[4][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Random Greedy vertex based with Prefetching *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[4][i],cost[4][i]);
      printf("************************************************\n");
    }

    wed *edgel;


    if (WLocalMaxSequential) {
      
      edgel = (wed *) malloc(2+ver[n[i]+1]* sizeof(wed));
      if (edgel == NULL) {
        printf("Unable to allocate memory for edgel in main() \n");
        return(false);
      }

      for(l=0;l<n_runs;l++) {

// Put the edges in a list that the algorithm can alter

        for(j=0;j<m[i];j++) {
          edgel[j].x = we[j].x;
          edgel[j].y = we[j].y;
          edgel[j].w = we[j].w;
        }

        mt1 = omp_get_wtime(); 
        localmax(n[i],m[i],edgel,p,ws1,p1);     // Round based local max (Birn, Sanders etc)
//      localmax(n[i],m[i],we,p,ws1,p1);     // Round based local max (Birn, Sanders etc) This version destroys the "we" list
        mt2 = omp_get_wtime(); 
        if ((timer[22][i] < 0) || (mt2-mt1 < timer[22][i]))
          timer[22][i] = mt2-mt1;
      }


      if (!verify_matching(n[i],ver,edges,p)) {
        printf("localmax() did not produce a legal matching \n");
        exit(false);
      }

      cost[22][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Round based local max algorithm             *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[22][i],cost[22][i]);
      printf("************************************************\n");
    }


    if (WSuitorSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        sweight(n[i],ver,edges,p,ws1,weight,p1);     // Suitor based weighted matching. Using a for-loop as outer loop
        mt2 = omp_get_wtime(); 
        if ((timer[5][i] < 0) || (mt2-mt1 < timer[5][i]))
          timer[5][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("sweight() did not produce a legal matching \n");
        exit(false);
      }

      cost[5][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Suitor algorithm                            *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[5][i],cost[5][i]);
      printf("************************************************\n");

      if (WRama) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime(); 
        rama(n[i],ver,edges,p2,ws2,weight,p1);     // Rama type post-processing
        mt2 = omp_get_wtime(); 
        timer[28][i] = mt2-mt1;
        cost[28][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
      if (WRoma) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime(); 
        roma(n[i],ver,edges,p2,ws2,weight,p1);     // Roma type post-processing
        mt2 = omp_get_wtime(); 
        timer[29][i] = mt2-mt1;
        cost[29][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
      if (WRema) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        rema(n[i],ver,edges,p2,ws2,weight,p1);     // Rema type post-processing
        mt2 = omp_get_wtime();
        timer[30][i] = mt2-mt1;
        cost[30][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
    }



    if (WSuitorCompEdgeSequential) {
      int *nn = (int *) malloc((n[i]+3)*sizeof(int));

      for(l=0;l<1;l++) {  // Can only be run once in current form

        for(j=0;j<=n[i]+1;j++) 
        nn[j] = ver[j];

        mt1 = omp_get_wtime(); 
        sweight11(n[i],ver,edges,p,ws1,weight,nn);     // Suitor based weighted matching. Using a for-loop as outer loop
        mt2 = omp_get_wtime(); 
        if ((timer[19][i] < 0) || (mt2-mt1 < timer[19][i]))
          timer[19][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("sweight11() did not produce a legal matching \n");
        exit(false);
      }

      cost[19][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Suitor algorithm with compressed edge lists *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[19][i],cost[19][i]);
      printf("*                                              *\n");
      printf("*  NB NB weight and edges arrays are destroyed *\n");
      printf("************************************************\n");
    }

    if (WSuitorWhileSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        sweight3(n[i],ver,edges,p,ws1,weight);   // Suitor based weighted matching. Using a while-loop as outer loop
        mt2 = omp_get_wtime(); 
        if ((timer[6][i] < 0) || (mt2-mt1 < timer[6][i]))
          timer[6][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("sweight3() did not produce a legal matching \n");
        exit(false);
      }

      cost[6][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Suitor algorithm with while loop            *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[6][i],cost[6][i]);
      printf("************************************************\n");
    }

    if (WSuitorStackSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        mcw1(n[i],ver,edges,p,ws1,next,weight);   // Suitor based weighted matching. Using a stack as outer loop
        mt2 = omp_get_wtime(); 
        if ((timer[25][i] < 0) || (mt2-mt1 < timer[25][i]))
          timer[25][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("mcw1() did not produce a legal matching \n");
        exit(false);
      }

      cost[25][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Suitor algorithm with stack                 *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[25][i],cost[25][i]);
      printf("************************************************\n");
    }

    if (WSuitorQueueSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        mcw2(n[i],ver,edges,p,ws1,next,weight);   // Suitor based weighted matching. Using a queue as outer loop
        mt2 = omp_get_wtime(); 
        if ((timer[26][i] < 0) || (mt2-mt1 < timer[26][i]))
          timer[26][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("mcw2() did not produce a legal matching \n");
        exit(false);
      }

      cost[26][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Suitor algorithm with queue                 *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[26][i],cost[26][i]);
      printf("************************************************\n");
    }

    if (WSuitorDQueueSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        mcw3(n[i],ver,edges,p,ws1,next,used,weight);   // Suitor based weighted matching. Using a two array queue
        mt2 = omp_get_wtime(); 
        if ((timer[27][i] < 0) || (mt2-mt1 < timer[27][i]))
          timer[27][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("mcw3() did not produce a legal matching \n");
        exit(false);
      }

      cost[27][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Suitor algorithm with two array queue       *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[27][i],cost[27][i]);
      printf("************************************************\n");
    }

    if (SortNeighborListsSequential) {

// Copy the neighbor list + weights into list of structs (needed for sorting using qsort)

      for(j=0;j<ver[n[i]+1];j++) {
        swe[j].w = weight[j];
        swe[j].x = edges[j];
      }

      mt1 = omp_get_wtime(); 
//    wsort(n[i],ver,edges,weight,&s_edges,&s_weight);  // Sort the edge list of each vertex by decreasing weight

      for(j=1;j<=n[i];j++) {
        qsort(&(swe[ver[j]]),ver[j+1]-ver[j],sizeof(neig),compneig); 		 // Sort the edge list of each vertex by decreasing weight
      }


      mt2 = omp_get_wtime(); 
      timer[20][i] = mt2-mt1;


 //  Check if sorting is correct
 /*
      int ferdig = false;
      for(xx=ver[j]+1;xx<ver[j+1];xx++) {
        if (swe[xx].w > swe[xx-1].w) {
          printf("%d, unsorted w(%d)=%lf > w(%d)=%lf \n",j,xx,swe[xx].w,xx-1,swe[xx-1].w);  
          ferdig = true;
        }
      }
      if (ferdig)
        return;


*/

      printf("************************************************\n");
      printf("*  Neighbor list sorting                       *\n");
      printf("*  Time = %8.6lf seconds                     *\n",timer[20][i]);
      printf("************************************************\n");
    }


    if (WSuitorSortSequential) {
      if (!SortNeighborListsSequential) {
        printf("Trying to run suitor algorithm with sorted adjacency lists without sorting first!!!\n");
        return;
      }
      int xxx = ver[n[i]+1];
      s_edges = (int *) malloc(sizeof(int)*xxx);
 
      if (s_edges == NULL) {
        printf("Unable to allocate memory for s_edges in wframe3 \n");
        return;
      }
    
      s_weight = (double *) malloc(sizeof(double)*(ver[n[i]+1]));
      if (s_weight == NULL) {
        printf("Unable to allocate memory for s_weight in wframe3 \n");
        return;
      }

      for(j=0;j<ver[n[i]+1];j++) {
        s_edges[j] = swe[j].x;
        s_weight[j] = swe[j].w;
      }

      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        sweight1(n[i],ver,s_edges,p,ws1,s_weight,next);      // Suitor algorithm assuming sorted neighbor lists
//      sweight1(n[i],ver,edges,p,ws1,weight,next);      // Suitor algorithm assuming sorted neighbor lists   NB assumes edges and weight are sorted!!!
        mt2 = omp_get_wtime(); 
        if ((timer[13][i] < 0) || (mt2-mt1 < timer[13][i]))
          timer[13][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("sweight1() did not produce a legal matching \n");
        exit(false);
      }
      cost[13][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Suitor algorithm with sorted edge lists     *\n"); 
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[13][i],cost[13][i]);
      printf("************************************************\n");
    }


    if (WSuitorSortWhileSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        sweight7(n[i],ver,s_edges,p,ws1,s_weight,next);  // Suitor based code, using sorted adjacency lists
        mt2 = omp_get_wtime(); 
        if ((timer[14][i] < 0) || (mt2-mt1 < timer[14][i]))
          timer[14][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("sweight7() did not produce a legal matching \n");
        exit(false);
      }
      cost[14][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Suitor with sorted edge lists + while loop  *\n"); 
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[14][i],cost[14][i]);
      printf("************************************************\n");
    }


// Next follows the algorithms using dynamic programming for post-processing

    printf("************************************************\n");
    printf("***            Algorithms using DP           ***\n");
    printf("************************************************\n");
/*
    for(l=0;l<n_runs;l++) {
      mt1 = omp_get_wtime(); 
      sweight4(n[i],ver,edges,p,ws1,p2,ws2,weight,used);      // Two rounds of suitor based matching, followed by dyn. prog 
      mt2 = omp_get_wtime(); 				     // This does not produce a complete matching, only the weight
      if ((timer[7][i] < 0) || (mt2-mt1 < timer[7][i]))
        timer[7][i] = mt2-mt1;
    }

    printf("Two round sequential weighted matching with dynamic programming took %lf seconds\n",timer[7][i]);
*/

    if (W2RSuitorDPSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        sweight5(n[i],ver,edges,p1,ws1,p2,ws2,weight,used,p);      // Same as above, but now with separate routine for dyn. prog
        mt2 = omp_get_wtime(); 
        if ((timer[8][i] < 0) || (mt2-mt1 < timer[8][i]))
          timer[8][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("sweight5() did not produce a legal matching \n");
        exit(false);
      }

      cost[8][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  2R Suitor with one routine for DP           *\n"); 
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[8][i],cost[8][i]);
      printf("************************************************\n");
    }

    if (WBmatching) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        bmatching(n[i],ver,edges,p1,ws1,p2,ws2,weight,used,p);      // Same as above, but now with separate routine for cycle dyn. prog
        mt2 = omp_get_wtime(); 
        if ((timer[24][i] < 0) || (mt2-mt1 < timer[24][i]))
          timer[24][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("bmatching() did not produce a legal matching \n");
        exit(false);
      }

      cost[24][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  2-matching with path DP + cycle DP          *\n"); 
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[24][i],cost[24][i]);
      printf("************************************************\n");
    }


    if (W2RSuitorDPCycSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        sweight8(n[i],ver,edges,p1,ws1,p2,ws2,weight,used,p);      // Same as above, but now with separate routine for cycle dyn. prog


        mt2 = omp_get_wtime(); 
        if ((timer[15][i] < 0) || (mt2-mt1 < timer[15][i]))
          timer[15][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("sweight8() did not produce a legal matching \n");
        exit(false);
      }

      cost[15][i] = cost_and_correct_matching(n[i],ver,edges,weight,p,ws1);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Two round suitor with path DP + cycle DP    *\n"); 
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[15][i],cost[15][i]);
      printf("************************************************\n");

      if (WRama) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        rama(n[i],ver,edges,p2,ws2,weight,p1);     // Rama type post-processing
        mt2 = omp_get_wtime();
        timer[31][i] = mt2-mt1;
        cost[31][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
      if (WRoma) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        roma(n[i],ver,edges,p2,ws2,weight,p1);     // Roma type post-processing
        mt2 = omp_get_wtime();
        timer[32][i] = mt2-mt1;
        cost[32][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
      if (WRema) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        rema(n[i],ver,edges,p2,ws2,weight,p1);     // Rema type post-processing
        mt2 = omp_get_wtime();
        timer[33][i] = mt2-mt1;
        cost[33][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
    }
 
        
/*
    for(l=0;l<n_runs;l++) {
      mt1 = omp_get_wtime(); 
      sweight6(n[i],ver,edges,p1,ws1,p2,ws2,weight,used,p);      // Two rounds of localy greedy matching (wseq), followed by dyn. prog
      mt2 = omp_get_wtime(); 
      if ((timer[9][i] < 0) || (mt2-mt1 < timer[9][i]))
        timer[9][i] = mt2-mt1;
    }

    if (!verify_matching(n[i],ver,edges,p)) {
      printf("sweight6() did not produce a legal matching \n");
      exit(false);
    }

    cost[9][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

    printf("************************************************\n");
    printf("*a)Two round local greedy algorithm  with DP   *\n"); 
    printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[9][i],cost[9][i]);
    printf("************************************************\n");
*/


    if (W2RGreedyDPSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime();
        sweight9(n[i],ver,edges,p1,ws1,p2,ws2,weight,used,p);      // Two rounds of localy greedy matching (wseq), followed by dyn. prog
        mt2 = omp_get_wtime();
        if ((timer[16][i] < 0) || (mt2-mt1 < timer[16][i]))
          timer[16][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("sweight9() did not produce a legal matching \n");
        exit(false);
      }

      cost[16][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Two round local greedy algorithm with DP    *\n"); 
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[16][i],cost[16][i]);
      printf("************************************************\n");
    }

    // printf("Sorting %d numbers \n",m[i]);


    if (SortEdgesSequential) {
      mt1 = omp_get_wtime(); 
      qsort(we,m[i],sizeof(wed),compedge);	// Sort the edges by decreasing weight
      mt2 = omp_get_wtime(); 
      timer[10][i] = mt2 - mt1;


      for(l=0;l<m[i]-1;l++) {  			// Check that the edges are sorted
        if (we[l].w < we[l+1].w) {
          printf("Data is not sorted in descending order \n");
          printf("%d =  %lf   < %d =  %lf \n",l,we[l].w,l+1,we[l+1].w);
          exit(0);
        }
      }

      printf("************************************************\n");
      printf("*  Sort all the edges                          *\n"); 
      printf("*  Time = %8.6lf seconds                     *\n",timer[10][i]);
      printf("************************************************\n");
    }

    if (WGreedySequential) {
      if (!SortEdgesSequential) {
        printf("Trying to run Greedy without sorting the edges first! \n");
        return;
      }
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        greedy(n[i],m[i],we,p);	// Run greedy on sorted edge list
        mt2 = omp_get_wtime(); 
        if ((timer[23][i] < 0) || (mt2-mt1 < timer[23][i]))
          timer[23][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("greedy() did not produce a legal matching \n");
        exit(false);
      }

      cost[23][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Greedy algorithm                            *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[23][i],cost[23][i]);
      printf("************************************************\n");
    }

    if (WGPASequential) {
      if (!SortEdgesSequential) {
        printf("Trying to run GPA without sorting the edges first! \n");
        return;
      }
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        gpa(n[i],m[i],we,ver,edges,weight,p);	// Create paths and cycles by adding in edges (by decreasing weight), followed by dyn. prog
        mt2 = omp_get_wtime(); 
        if ((timer[11][i] < 0) || (mt2-mt1 < timer[11][i]))
          timer[11][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("GPA() did not produce a legal matching \n");
        exit(false);
      }

      cost[11][i] = cost_and_correct_matching(n[i],ver,edges,weight,p,ws1);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  GPA algorithm with dynamic programming      *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[11][i],cost[11][i]);
      printf("************************************************\n");

      if (WRama) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        rama(n[i],ver,edges,p2,ws2,weight,p1);     // Rama type post-processing
        mt2 = omp_get_wtime();
        timer[37][i] = mt2-mt1;
        cost[37][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
      if (WRoma) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        roma(n[i],ver,edges,p2,ws2,weight,p1);     // Roma type post-processing
        mt2 = omp_get_wtime();
        timer[38][i] = mt2-mt1;
        cost[38][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
      if (WRema) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        rema(n[i],ver,edges,p2,ws2,weight,p1);     // Rema type post-processing
        mt2 = omp_get_wtime();
        timer[39][i] = mt2-mt1;
        cost[39][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }

    }

    if (WPGASequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        path(n[i],ver,edges,weight,p);		// Path growing algorithm followed by dynamic programming
        mt2 = omp_get_wtime(); 
        if ((timer[12][i] < 0) || (mt2-mt1 < timer[12][i]))
          timer[12][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("path() did not produce a legal matching \n");
        exit(false);
      }

      cost[12][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Path growing algorithm with DP (PGA')       *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[12][i],cost[12][i]);
      printf("************************************************\n");

      if (WRama) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        rama(n[i],ver,edges,p2,ws2,weight,p1);     // Rama type post-processing
        mt2 = omp_get_wtime();
        timer[34][i] = mt2-mt1;
        cost[34][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
      if (WRoma) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        roma(n[i],ver,edges,p2,ws2,weight,p1);     // Roma type post-processing
        mt2 = omp_get_wtime();
        timer[35][i] = mt2-mt1;
        cost[35][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
      if (WRema) {
        int c;
        for(c=0;c<=n[i];c++) {
          p2[c] = p[c];
          ws2[c] = ws1[c];
        }
        mt1 = omp_get_wtime();
        rema(n[i],ver,edges,p2,ws2,weight,p1);     // Rema type post-processing
        mt2 = omp_get_wtime();
        timer[36][i] = mt2-mt1;
        cost[36][i] = cost_matching(n[i],ver,edges,weight,p2);  // Get the cost of the matching
      }
    }

    if (W2RPGADPSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        pth1(n[i],ver,edges,weight,p);		// 2 Round path growing algorithm followed by dynamic programming
        mt2 = omp_get_wtime(); 
        if ((timer[17][i] < 0) || (mt2-mt1 < timer[17][i]))
          timer[17][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("pth1() did not produce a legal matching \n");
        exit(false);
      }

      cost[17][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Two round path growing algorithm + DP       *\n");
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[17][i],cost[17][i]);
      printf("************************************************\n");
    }

    if (W2RSuitorDPSortEdgeSequential) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime();
        sweight10(n[i],ver,s_edges,p1,ws1,p2,ws2,s_weight,next,used,p);
//      sweight10(n[i],ver,edges,p1,ws1,p2,ws2,weight,next,used,p);   // NB assumes edges and weight are sorted
        mt2 = omp_get_wtime();
        if ((timer[18][i] < 0) || (mt2-mt1 < timer[18][i]))
          timer[18][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("sweight10() did not produce a legal matching \n");
        exit(false);
      }
      cost[18][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching

      printf("************************************************\n");
      printf("*  Two round suitor + DP with sorted lists     *\n"); 
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[18][i],cost[18][i]);
      printf("************************************************\n");
    }

    if (WBcmatching) {
      for(l=0;l<n_runs;l++) {
        mt1 = omp_get_wtime(); 
        bcmatching(n[i],ver,edges,p1,ws1,p2,ws2,weight,used,p);      // Same as above, but now with separate routine for cycle dyn. prog
        mt2 = omp_get_wtime(); 
        if ((timer[25][i] < 0) || (mt2-mt1 < timer[25][i]))
          timer[25][i] = mt2-mt1;
      }

      if (!verify_matching(n[i],ver,edges,p)) {
        printf("bcmatching() did not produce a legal matching \n");
        exit(false);
      }

//      cost[25][i] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
      cost[25][i] = cost[24][i];

      printf("************************************************\n");
      printf("*  2-matching with compression and  DP         *\n"); 
      printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",timer[25][i],cost[25][i]);
      printf("************************************************\n");
    }

// Allocate space for locks

    omp_lock_t *nlocks;
    nlocks = (omp_lock_t *) malloc((n[i]+1)*sizeof(omp_lock_t));

    printf("************************************************\n");
    printf("***            Parallel algorithms           ***\n");
    printf("************************************************\n");

// Run parallel algorithm
// One iteration for each thread configuration
    for(j=0;j<n_conf;j++) {
      printf("************************************************\n");
      printf("*       Configuration %2d, using %2d threads     *\n",j,conf[j]);
      printf("************************************************\n");
      omp_set_num_threads(conf[j]);  // Set number of threads in this configuration

#pragma omp parallel 
      {
        int k;
        int threads = omp_get_num_threads();
        int my_id = omp_get_thread_num();
      

        if (threads != conf[j]) {
          printf("***** Did not get the correct numer of threads! Wanted %d, got %d *****\n",conf[i],threads);
        }

// Now we are ready to run the parallel algorithms
// Run as many times as required, only store timings for best run

#pragma omp barrier

        if (SortEdgeListsParallel) {
// Time how long time it takes to sort the individual edge lists
// Copy the neighbor list + weights into list of structs (needed for sorting using qsort)

          int b;
#pragma omp for schedule(static) private(b)
          for(b=0;b<ver[n[i]+1];b++) {
            swe[b].w = weight[b];
            swe[b].x = edges[b];
          }
         
#pragma omp barrier
#pragma omp master
          { mt1 = omp_get_wtime(); }
// #pragma omp for schedule(dynamic,700) private(b)
#pragma omp for schedule(static) private(b)
          for(b=1;b<=n[i];b++) {
            qsort(&(swe[ver[b]]),ver[b+1]-ver[b],sizeof(neig),compneig);               // Sort the edge list of each vertex by decreasing weight
          }
          
#pragma omp master
          { mt2 = omp_get_wtime();
            if ((p_timer[10][i][j] < 0) || (mt2-mt1 < p_timer[10][i][j]))
              p_timer[10][i][j] = mt2-mt1;
          }
        }

// Allocating space for variables needed in the parallel code

        int *path1;
        double *weight1;
        wed *new_list;

        new_list = (wed *) malloc(sizeof(wed) * m[i]);  // Allocate local memory to store edge list
        if (new_list == NULL)
          printf("Unable to allocate space for new_list \n");
 
        path1 = (int *) malloc(max_n * sizeof(int));
        if (path1 == NULL)
          printf("Unable to allocate space for path1 \n");

        weight1 = (double *) malloc(max_n * sizeof(double));
        if (weight1 == NULL)
          printf("Unable to allocate space for weight1 \n");

#pragma omp barrier
        for(k=0;k<n_runs;k++) {	  // Run all experiments n_runs times, select best run time as final

          if (UGreedyVertexVerifyParallel) {
#pragma omp barrier
#pragma omp master
            { mt1 = omp_get_wtime(); }
            verify(n[i],ver,edges,p,next);   // Run vertex based verification code
#pragma omp master
            { mt2 = omp_get_wtime();
              if ((p_timer[0][i][j] < 0) || (mt2-mt1 < p_timer[0][i][j]))
                p_timer[0][i][j] = mt2-mt1;

              if (!verify_matching(n[i],ver,edges,p)) {
                printf("verify() did not produce a legal matching \n");
                exit(false);
              }
            } // master
          }

          if (UGreedyEdgeVerifyParallel) {
#pragma omp barrier
#pragma omp master
            { mt1 = omp_get_wtime(); }
            edgec1(n[i],m[i],e,p,next);   // Run edge based verification code
#pragma omp master
            { mt2 = omp_get_wtime(); 
              if ((p_timer[1][i][j] < 0) || (mt2-mt1 < p_timer[1][i][j]))
                p_timer[1][i][j] = mt2-mt1;   // Note that the current version of edgec does not return the actual matching
            }
          }


          if (UGreedyEdgeVerifyPrefetchParallel) {
#pragma omp barrier
#pragma omp master
            {  mt1 = omp_get_wtime(); }
            edgec2(n[i],m[i],e,p,next);  // The same as above but with prefetching directives
#pragma omp master
            {  mt2 = omp_get_wtime(); 
               if ((p_timer[11][i][j] < 0) || (mt2-mt1 < p_timer[11][i][j]))
                 p_timer[11][i][j] = mt2-mt1;
            }
          }

          if (UGreedyEdgeLockParallel) {
#pragma omp barrier
#pragma omp master
            { mt1 = omp_get_wtime(); }
            locking1(n[i],m[i],e,p,nlocks);  // Run edge based locking code
#pragma omp master
            { mt2 = omp_get_wtime(); 
              if ((p_timer[2][i][j] < 0) || (mt2-mt1 < p_timer[2][i][j]))
                p_timer[2][i][j] = mt2-mt1; // Note the current version of locking1 does not return the actual matching
            }
          }

// Running weighted algorithms
          if (WMBLocalParallel) {  // Manne-Bisseling algorithm with local lists
#pragma omp barrier
#pragma omp master
            { mt1 = omp_get_wtime(); }
            MBLocal(n[i],ver,edges,p,ws1,nlocks,weight,path1,tlist,p1,p2,weight1);    

#pragma omp master
            { mt2 = omp_get_wtime(); 
              if ((p_timer[13][i][j] < 0) || (mt2-mt1 < p_timer[13][i][j]))
                p_timer[13][i][j] = mt2-mt1;

              if (!verify_matching(n[i],ver,edges,p)) {
                printf("MBLocal() did not produce a legal matching \n");
                exit(false);
              }
              p_cost[13][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }


          if (WMB1ListParallel) {  // Manne-Bisseling algorithm with one list
#pragma omp barrier
#pragma omp master
            { mt1 = omp_get_wtime(); }
            MB1List(n[i],ver,edges,p,ws1,nlocks,weight,path1,tlist,p1,p2,p3);    

#pragma omp master
            { mt2 = omp_get_wtime(); 
              if ((p_timer[12][i][j] < 0) || (mt2-mt1 < p_timer[12][i][j]))
                p_timer[12][i][j] = mt2-mt1;

              if (!verify_matching(n[i],ver,edges,p)) {
                printf("MB1List() did not produce a legal matching \n");
                exit(false);
              }
              p_cost[12][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }

          if (WSuitorStackParallel) {  // Suitor algorithm where the vertices are first placed in a stack
#pragma omp barrier
#pragma omp master
            { mt1 = omp_get_wtime(); }
            weighted(n[i],ver,edges,p,ws1,next,nlocks,weight);

#pragma omp master
            { mt2 = omp_get_wtime(); 
              if ((p_timer[3][i][j] < 0) || (mt2-mt1 < p_timer[3][i][j]))
                p_timer[3][i][j] = mt2-mt1;

              if (!verify_matching(n[i],ver,edges,p)) {
                printf("weighted() did not produce a legal matching \n");
                exit(false);
              }
              p_cost[3][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }



          if (WSuitorParallel) {  // Standard Parallel Suitor algorithm
#pragma omp barrier
#pragma omp master
            { mt1 = omp_get_wtime(); }
             weighted2(n[i],ver,edges,p,ws1,nlocks,weight);    // regular code 
            // weighted2_os(n[i],ver,edges,p,ws1,nlocks,weight,where);    // using only one variable to hold the suitor
#pragma omp master
            { 
			  
              mt2 = omp_get_wtime(); 
              if ((p_timer[4][i][j] < 0) || (mt2-mt1 < p_timer[4][i][j]))
                p_timer[4][i][j] = mt2-mt1;

/*
              int xy;
              for(xy=1;xy<=n[i];xy++)                 // Extra code for weighted2_os !!!
                p[xy] = edges[p[xy]];
*/

              if (!verify_matching(n[i],ver,edges,p)) {
                printf("weighted2() did not produce a legal matching \n");
                exit(false);
              }
              p_cost[4][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }

          if (WStackParallel) {  // Parallel Suitor algorithm using stack
#pragma omp barrier
#pragma omp master
            { mt1 = omp_get_wtime(); }
             pstack1(n[i],ver,edges,p,next,ws1,nlocks,weight);    // McVittie and Wilson algorithm in parallel
#pragma omp master
            { 
			  
              mt2 = omp_get_wtime(); 
              if ((p_timer[15][i][j] < 0) || (mt2-mt1 < p_timer[15][i][j]))
                p_timer[15][i][j] = mt2-mt1;

/*
              int xy;
              for(xy=1;xy<=n[i];xy++)                 // Extra code for weighted2_os !!!
                p[xy] = edges[p[xy]];
*/

              if (!verify_matching(n[i],ver,edges,p)) {
                printf("pstack1() did not produce a legal matching \n");
                exit(false);
              }
              p_cost[15][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }

          if (WQueueParallel) {  // Parallel Suitor algorithm using queue
#pragma omp barrier
#pragma omp master
            { mt1 = omp_get_wtime(); }
             pqueue1(n[i],ver,edges,p,next,used,ws1,nlocks,weight);    // Gale and Shapley algorithm in parallel
#pragma omp master
            { 
			  
              mt2 = omp_get_wtime(); 
              if ((p_timer[16][i][j] < 0) || (mt2-mt1 < p_timer[16][i][j]))
                p_timer[16][i][j] = mt2-mt1;

/*
              int xy;
              for(xy=1;xy<=n[i];xy++)                 // Extra code for weighted2_os !!!
                p[xy] = edges[p[xy]];
*/

              if (!verify_matching(n[i],ver,edges,p)) {
                printf("pqueue1() did not produce a legal matching \n");
                exit(false);
              }
              p_cost[16][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }

          if (W2RSuitorDPParallel) {
// Running 2 stage weighted Suitor algorithm with dynamic programming

#pragma omp barrier
#pragma omp master
			{ 
              mt1 = omp_get_wtime(); }
               // pdp(n[i],ver,edges,p,ws1,nlocks,weight);
            
            pdp(n[i],ver,edges,p1,ws1,p2,ws2,weight,used,p,nlocks,next,path1,weight1);  
            
#pragma omp master
            { mt2 = omp_get_wtime(); 
              if ((p_timer[5][i][j] < 0) || (mt2-mt1 < p_timer[5][i][j]))
                p_timer[5][i][j] = mt2-mt1;

              if (!verify_matching(n[i],ver,edges,p)) {
                printf("pdp() did not produce a legal matching \n");
                exit(false);
              }
              p_cost[5][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }

          if (W2MatchingParallel) {
// Running 2-matching algorithm 

#pragma omp barrier
#pragma omp master
			{ 
              mt1 = omp_get_wtime(); }
               // pdp(n[i],ver,edges,p,ws1,nlocks,weight);
            
            bweight1(n[i],ver,edges,p1,ws1,p2,ws2,weight,used,p,nlocks,next,path1,weight1);  
            
#pragma omp master
            { mt2 = omp_get_wtime(); 
              if ((p_timer[14][i][j] < 0) || (mt2-mt1 < p_timer[14][i][j]))
                p_timer[14][i][j] = mt2-mt1;
/*
              if (!verify_matching(n[i],ver,edges,p)) {
                printf("pdp() did not produce a legal matching \n");
                exit(false);
              }
              p_cost[5][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
*/
            }
          }


          if (WSuitorSortParallel) {
// Running parallel suitor algorithm with sorted adjacency lists

#pragma omp barrier
#pragma omp master
			{ 
              mt1 = omp_get_wtime(); }
            
            psweight1(n[i],ver,s_edges,p,ws1,s_weight,next,nlocks);
           
#pragma omp master
            { mt2 = omp_get_wtime();
              if ((p_timer[6][i][j] < 0) || (mt2-mt1 < p_timer[6][i][j]))
                p_timer[6][i][j] = mt2-mt1;

              if (!verify_matching(n[i],ver,edges,p)) {
                printf("psweight1() did not produce a legal matching \n");
                exit(false);
              }

              p_cost[6][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }


          if (W2RSuitorSortParallel) {
// Running 2 stage weighted Suitor algorithm assuming sorted edge lists followed by dynamic programming
#pragma omp barrier
#pragma omp master
			{ 
              mt1 = omp_get_wtime(); }
               // pdp(n[i],ver,edges,p,ws1,nlocks,weight);
               // Trenger en int og en double n_max
              
              spdp(n[i],ver,s_edges,p1,ws1,p2,ws2,s_weight,used,p,nlocks,next,path1,weight1);
              
#pragma omp master
            { 
			  mt2 = omp_get_wtime();
              if ((p_timer[7][i][j] < 0) || (mt2-mt1 < p_timer[7][i][j]))
                p_timer[7][i][j] = mt2-mt1;

			  
              if (!verify_matching(n[i],ver,edges,p)) {
                printf("spdp() did not produce a legal matching \n");
                exit(false);
              }
              p_cost[7][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
			  
            }
          }



          if (WLocalMax1ListParallel) {   // Local Max with one global list
            if (!WLocalMaxSequential) {
#pragma omp master
              printf("Running the parallel Local Max algorithm requires that the sequential one is executed first \n");
              exit(false);
            }
#pragma omp barrier
            int jj;
#pragma omp master
            for(jj=0;jj<m[i];jj++) {       // Put the edges in a list that the algorithm can alter
              edgel[jj].x = we[jj].x;
              edgel[jj].y = we[jj].y;
              edgel[jj].w = we[jj].w;
            }
#pragma omp barrier
#pragma omp master
			{ 
              mt1 = omp_get_wtime(); }
            
            plocalmax(n[i],m[i],edgel,p,ws1,p1,nlocks,tlist,new_list);     // Parallel round based local max (Birn, Sanders etc)
            
#pragma omp master
            { 
              mt2 = omp_get_wtime();
            
              if ((p_timer[8][i][j] < 0) || (mt2-mt1 < p_timer[8][i][j])) {
                p_timer[8][i][j] = mt2-mt1;
              }
              if (!verify_matching(n[i],ver,edges,p)) {
                printf("plocalmax() did not produce a legal matching \n");
                exit(false);
              }

              p_cost[8][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }

          if (WLocalMaxLocalListParallel) { 
            int jj;
#pragma omp barrier
#pragma omp master
            for(jj=0;jj<m[i];jj++) {          // Put the edges in a list that the algorithm can alter
              edgel[jj].x = we[jj].x;
              edgel[jj].y = we[jj].y;
              edgel[jj].w = we[jj].w;
            }
#pragma omp barrier
#pragma omp master
			{ 
              mt1 = omp_get_wtime(); }
            
            plocalmax1(n[i],m[i],edgel,p,ws1,p1,nlocks,tlist,new_list);     // Parallel round based local max (Birn, Sanders etc)
           
            
#pragma omp master
            { 
              mt2 = omp_get_wtime();
            
              if ((p_timer[9][i][j] < 0) || (mt2-mt1 < p_timer[9][i][j])) {
                p_timer[9][i][j] = mt2-mt1;
              }
              if (!verify_matching(n[i],ver,edges,p)) {
                printf("plocalmax1() did not produce a legal matching \n");
                exit(false);
              }

              p_cost[9][i][j] = cost_matching(n[i],ver,edges,weight,p);  // Get the cost of the matching
            }
          }




#pragma omp barrier

        } // End of k iterations over same configuration 

       
        free(weight1);
        free(path1);
        free(new_list);

        

       
#pragma omp master
        { 
/*
          printf("\n**** Unweighted algorithms ****\n");
*/
          if (UGreedyVertexVerifyParallel) {
            printf("************************************************\n");
            printf("*  Parallel unweighted vertex algorithm        *\n");
            printf("*  Time = %8.6lf seconds                     *\n",p_timer[0][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[0][i]/p_timer[0][i][j],p_timer[0][i][0]/p_timer[0][i][j]);
            printf("************************************************\n");
          }

          if (UGreedyEdgeVerifyParallel) {
            printf("************************************************\n");
            printf("*  Parallel unweighted edge algorithm          *\n");
            printf("*  Time = %8.6lf seconds                     *\n",p_timer[1][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[1][i]/p_timer[1][i][j],p_timer[1][i][0]/p_timer[1][i][j]);
            printf("************************************************\n");
          }

          if (UGreedyEdgeVerifyPrefetchParallel) {
            printf("************************************************\n");
            printf("*  Parallel unweighted edge algorithm +prefetch*\n");
            printf("*  Time = %8.6lf seconds                     *\n",p_timer[11][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[1][i]/p_timer[11][i][j],p_timer[11][i][0]/p_timer[11][i][j]);
            printf("************************************************\n");
          }

          if (UGreedyEdgeLockParallel) {
            printf("************************************************\n");
            printf("*  Parallel unweighted edge locking algorithm  *\n");
            printf("*  Time = %8.6lf seconds                     *\n",p_timer[2][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[1][i]/p_timer[2][i][j],p_timer[2][i][0]/p_timer[2][i][j]);
            printf("************************************************\n");
          }

//          printf("\n**** Weighted algorithms ****\n");

//         printf("Weighted code took %lf seconds, absolute speedup = %4.2lf, relative speedup = %4.2lf  \n",p_timer[3][i][j],
//                  timer[6][i]/p_timer[3][i][j],p_timer[3][i][0]/p_timer[3][i][j]);

          if (WMBLocalParallel) {
            printf("************************************************\n");
            printf("*  Parallel MB algorithm  with local list      *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[13][i][j],p_cost[13][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[5][i]/p_timer[13][i][j],p_timer[13][i][0]/p_timer[13][i][j]);
            printf("************************************************\n");
          }

          if (WMB1ListParallel) {
            printf("************************************************\n");
            printf("*  Parallel MB algorithm  with 1 list          *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[12][i][j],p_cost[12][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[5][i]/p_timer[12][i][j],p_timer[12][i][0]/p_timer[12][i][j]);
            printf("************************************************\n");
          }

          if (WSuitorStackParallel) {
            printf("************************************************\n");
            printf("*  Parallel suitor algorithm  with stack       *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[3][i][j],p_cost[3][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[5][i]/p_timer[3][i][j],p_timer[3][i][0]/p_timer[3][i][j]);
            printf("************************************************\n");
          }

          if (WSuitorParallel) {
            printf("************************************************\n");
            printf("*  Parallel suitor algorithm                   *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[4][i][j],p_cost[4][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[5][i]/p_timer[4][i][j],p_timer[4][i][0]/p_timer[4][i][j]);
            printf("************************************************\n");
          }

          if (WStackParallel) {
            printf("************************************************\n");
            printf("*  Parallel McVittie and Wilson algorithm      *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[15][i][j],p_cost[15][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[5][i]/p_timer[15][i][j],p_timer[15][i][0]/p_timer[15][i][j]);
            printf("************************************************\n");
          }

          if (WQueueParallel) {
            printf("************************************************\n");
            printf("*  Parallel Gale and Shapley algorithm         *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[16][i][j],p_cost[16][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[5][i]/p_timer[16][i][j],p_timer[16][i][0]/p_timer[16][i][j]);
            printf("************************************************\n");
          }

          if (W2RSuitorDPParallel) {
            printf("************************************************\n");
            printf("*  Parallel 2 round Suitor  with DP            *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[5][i][j],p_cost[5][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[15][i]/p_timer[5][i][j],p_timer[5][i][0]/p_timer[5][i][j]);
            printf("************************************************\n");
          }

          if (WSuitorSortParallel) {
            printf("************************************************\n");
            printf("*  Parallel Suitor with sorted edge lists      *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[6][i][j],p_cost[6][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[13][i]/p_timer[6][i][j],p_timer[6][i][0]/p_timer[6][i][j]);
            printf("************************************************\n");
          }

          if (W2RSuitorSortParallel) {
            printf("************************************************\n");
            printf("*  Parallel 2R Suitor w/sorted edge lists + DP *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[7][i][j],p_cost[7][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[18][i]/p_timer[7][i][j],p_timer[7][i][0]/p_timer[7][i][j]);
            printf("************************************************\n");
          }

          if (WLocalMax1ListParallel) {
            printf("************************************************\n");
            printf("*  Parallel local max algorithm, 1 list        *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[8][i][j],p_cost[8][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[22][i]/p_timer[8][i][j],p_timer[8][i][0]/p_timer[8][i][j]);
            printf("************************************************\n");
          }

          if (WLocalMaxLocalListParallel) {
            printf("************************************************\n");
            printf("*  Parallel local max with local edge lists    *\n");
            printf("*  Time = %8.6lf seconds, Weight = %8.1lf  *\n",p_timer[9][i][j],p_cost[9][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[22][i]/p_timer[9][i][j],p_timer[9][i][0]/p_timer[9][i][j]);
            printf("************************************************\n");
          }

          if (SortEdgeListsParallel) {
            printf("************************************************\n");
            printf("*  Parallel sorting of edge lists              *\n");
            printf("*  Time = %8.6lf seconds                     *\n",p_timer[10][i][j]);
            printf("*  Speedup = %4.2lf, Relative speedup = %4.2lf     *\n",timer[20][i]/p_timer[10][i][j],p_timer[10][i][0]/p_timer[10][i][j]);
            printf("************************************************\n");
          }

/*
          printf("Parallel suitor algorithm with sorted lists took %lf seconds, absolute speedup = %4.2lf, relative speedup = %4.2lf, cost = %e \n",
                  p_timer[6][i][j],timer[13][i]/p_timer[6][i][j],p_timer[6][i][0]/p_timer[6][i][j],p_cost[6][i][j]);
*/
/*
*/
        }

      }  // End of parallel section


    }  // End of different thread configurations

      
//    free(next);
//    free(nlocks);   // Free up space used for locks
//    free(ver);
//    free(edges);
//    free(e);
//    free(weight);
//    free(we);
//    free(swe);
//    free(used);
//    free(p);
//    free(p1);
//    free(p2);
//    free(ws1);
//    free(ws2);
//    free(s_edges);
//    free(s_weight);
//    printf("Done freeing space\n");


  } // Loop over graphs


// Print results to file
//
// name[i] 	Name of file i (including the .mtx extension)

  char *tc;         // Name of current graph without the .mtx extension

  fprintf(wf,"name = {");       // Print the names of the graphs
  for(i=0;i<n_graphs;i++)  {
    tc = strtok(name[i],"."); 		// Remove .mtx extension from name
    fprintf(wf,"'%s' ",tc);
    if (i != n_graphs-1)
      fprintf(wf,",");
  }
  fprintf(wf,"};\n");

  fprintf(wf,"n = [");       // Print the number of vertices of the graphs
  for(i=0;i<n_graphs;i++)  {
    fprintf(wf,"%d ",n[i]);
  }
  fprintf(wf,"];\n");

  fprintf(wf,"nz = [");       // Print the number of non-zeros of the graphs
  for(i=0;i<n_graphs;i++)  {
    fprintf(wf,"%d ",m[i]);
  }
  fprintf(wf,"];\n");

// Sequential timings
// ******************
// timer[0][]	Sequential unweighted greedy vertex based matching
// timer[1][]	Sequential unweighted greedy edge based matching
// timer[3][]	Sequential weighted localy greedy matching, vertex based, but in random order
// timer[4][]	Same as above, but code has been optimized with prefetching directives
// timer[5][] 	Suitor based weighted matching. Using a for-loop as outer loop
// timer[6][]	Suitor based weighted matching. Using a while-loop as outer loop
// timer[7][]	Two rounds of suitor based matching, followed by dyn. prog, only weight, no matching
// timer[8][]	Same as above, but now with separate routine for dyn. prog, will give matching.
// timer[9][]	Two rounds of localy greedy matching (wseq), followed by dyn. prog
// timer[10][]	Sort all of the edges by decreasing weight
// timer[11][]	GPA algorithm with dynamic programming
// timer[12][]	Path growing algorithm followed by dynamic programming
// timer[13][]	Suitor-based weighted matching, using sorted edge lists
// timer[14][] 	Suitor based code, using sorted adjacency lists, "optimized"
// timer[15][]	Two round suitor code with separate routine for cycle dynamic programming
// timer[16][]	Two rounds of localy greedy matching (wseq), followed by separate cyclic dyn. prog, compare with [9]
// timer[17][]	2-level Path growing algorithm followed by dynamic programming
// timer[18][]	2 round suitor-based weighted matching + DP, using sorted edge lists
// timer[20][]	Sorting of edge list of each vertex
// timer[21][]	Greedy weighted algorithm, considering vertices in order, compare with [3]
// timer[22][]	Round based local max algorithm
// timer[23][]	Greedy weighted algorithm on sorted edge list. 
// timer[24][]  2-matching followed by DP
// timer[25][]  Suitor using stack
// timer[28][]  Rama after Suitor
// timer[29][]  Roma after Suitor
// timer[30][]  Rema after Suitor
//
// Parallel timings, all timings gives the best run
// ****************
// ** Unweighted algorithms **
// p_timer[0][i][conf]	Unweighted greedy vertex based matching algorithm
// p_timer[1][i][conf]	Unweighted edge based verification code, does not return matching
// p_timer[2][i][conf]	Unweighted edge based locking code
// ** Weighted algorithms **
// p_timer[3][i][conf]	??
// p_timer[4][i][conf]	Parallel suitor algorithm
// p_timer[5][i][conf]	2 stage weighted suitor algorithm with dynamic programming
// p_timer[6][i][conf]	Parallel suitor algorithm with sorted adjacency lists
// p_timer[7][i][conf]	2 stage weighted suitor algorithm with sorted edge lists followed by dynamic programming
// p_timer[8][i][conf]	parallel local max algorithm
// p_timer[9][i][conf]	parallel local max algorithm with local edge lists
// p_timer[10][i][conf]	Sorting of edge lists
// p_timer[11][i][conf]	Parallel unweighted edge algorithm with prefetch
// p_timer[12][i][conf]	Parallel MB algorithm with 1 list 
// p_timer[13][i][conf]	Parallel MB algorithm with 1 list 
// p_timer[14][i][conf]	Parallel 2-matching algorithm 
// p_timer[15][i][conf]	Parallel McVittie and Wilson algorithm
//

  if (UGreedyVertex) 
    store_value(wf,"TimeUGreedyVertexSequential",timer[0],n_graphs);    // Print timing results for unweighted sequential greedy vertex algorithm
    
  if (UGreedyEdge) 
    store_value(wf,"TimeUGreedyEdgeSequential",timer[1],n_graphs);    // Print timing results for unweighted sequential greedy edge algorithm

  if (WBcmatching) {
    store_value(wf,"TimeWBcmatching",timer[25],n_graphs);    // Print timing results for 2-matching with compression
    store_value(wf,"CostWBcmatching",cost[25],n_graphs);     // Print cost results for 2-matching with compression
  }
  if (WBmatching) {
    store_value(wf,"TimeWBmatching",timer[24],n_graphs);    // Print timing results for 2-matching
    store_value(wf,"CostWBmatching",cost[24],n_graphs);     // Print cost results for 2-matching
  }

  if (WGreedyVertex) {
    store_value(wf,"TimeWGreedyVertexSequential",timer[2],n_graphs);    // Print timing results for weighted sequential greedy vertex algorithm
    store_value(wf,"CostWGreedyVertexSequential",cost[2],n_graphs);     // Print cost results for weighted sequential greedy vertex algorithm
  }

  if (WRandomGreedyVertex) {
    store_value(wf,"TimeWRandomGreedyVertexSequential",timer[3],n_graphs);    // Print timing results for weighted sequential greedy vertex algorithm
    store_value(wf,"CostWRandomGreedyVertexSequential",cost[3],n_graphs);     // Print cost results for weighted sequential greedy vertex algorithm
  } 

  if (WPrefetchRandomGreedyVertex) {
    store_value(wf,"TimeGreedyPreFSequential",timer[4],n_graphs); // Print timing results for sequential greedy algorithm
    store_value(wf,"CostGreedyPreFSequential",cost[4],n_graphs);  // Print cost results for sequential greedy algorithm
  } 

  if (WSuitorSequential) {
    store_value(wf,"TimeSuitorSequential",timer[5],n_graphs); // Print timing results for sequential suitor algorithm
    store_value(wf,"CostSuitorSequential",cost[5],n_graphs);  // Print cost results for sequential suitor algorithm

    if (WRama) {
      store_value(wf,"TimeRamaSuitorSequential",timer[28],n_graphs); // Print timing results for Rama after sequential suitor algorithm
      store_value(wf,"CostRamaSuitorSequential",cost[28],n_graphs);  // Print cost results for Rama after sequential suitor algorithm
    }
    if (WRoma) {
      store_value(wf,"TimeRomaSuitorSequential",timer[29],n_graphs); // Print timing results for Roma after sequential suitor algorithm
      store_value(wf,"CostRomaSuitorSequential",cost[29],n_graphs);  // Print cost results for Roma after sequential suitor algorithm
    }
    if (WRema) {
      store_value(wf,"TimeRemaSuitorSequential",timer[30],n_graphs); // Print timing results for Rema after sequential suitor algorithm
      store_value(wf,"CostRemaSuitorSequential",cost[30],n_graphs);  // Print cost results for Rema after sequential suitor algorithm
    }
  }

  if (WSuitorStackSequential) {
    store_value(wf,"TimeSuitorStackSequential",timer[25],n_graphs); // Print timing results for sequential suitor algorithm w/stack
    store_value(wf,"CostSuitorStackSequential",cost[25],n_graphs);  // Print cost results for sequential suitor algorithm w/stack
  }

  if (WSuitorQueueSequential) {
    store_value(wf,"TimeSuitorQueueSequential",timer[26],n_graphs); // Print timing results for sequential suitor algorithm w/queue
    store_value(wf,"CostSuitorQueueSequential",cost[26],n_graphs);  // Print cost results for sequential suitor algorithm w/queue
  }

  if (WSuitorDQueueSequential) {
    store_value(wf,"TimeSuitorDQueueSequential",timer[27],n_graphs); // Print timing results for sequential suitor algorithm w/d-queue
    store_value(wf,"CostSuitorDQueueSequential",cost[27],n_graphs);  // Print cost results for sequential suitor algorithm w/d-queue
  }

  if (WSuitorWhileSequential) {
    store_value(wf,"TimeSuitorWhileSequential",timer[6],n_graphs); // Print timing results for sequential suitor algorithm w/while loop
    store_value(wf,"CostSuitorWhileSequential",cost[6],n_graphs);  // Print cost results for sequential suitor algorithm w/while loop
  }

  if (W2RSuitorDPSequential) {
    store_value(wf,"TimeSuitorWhileSequential",timer[8],n_graphs); // Print timing results for 2 round sequential suitor algorithm w/ one routine for DP
    store_value(wf,"CostSuitorWhileSequential",cost[8],n_graphs);  // Print cost results for 2 round sequential suitor algorithm w/ one routine for DP
  }

  if (SortEdgesSequential) {
    store_value(wf,"TimeEdgeSortSequential",timer[10],n_graphs); // Print timing results for sorting all edges
  }

  if (WGPASequential) {
    store_value(wf,"TimeGPASequential",timer[11],n_graphs);      // Print timing results for sequential GPA algorithm
    store_value(wf,"CostGPASequential",cost[11],n_graphs);       // Print cost results for sequential GPA algorithm

    if (WRama) {
      store_value(wf,"TimeRamaGPASequential",timer[37],n_graphs); // Print timing results for Rama after sequential suitor algorithm
      store_value(wf,"CostRamaGPASequential",cost[37],n_graphs);  // Print cost results for Rama after sequential suitor algorithm
    }
    if (WRoma) {
      store_value(wf,"TimeRomaGPASequential",timer[38],n_graphs); // Print timing results for Roma after sequential suitor algorithm
      store_value(wf,"CostRomaGPASequential",cost[38],n_graphs);  // Print cost results for Roma after sequential suitor algorithm
    }
    if (WRema) {
      store_value(wf,"TimeRemaGPASequential",timer[39],n_graphs); // Print timing results for Rema after sequential suitor algorithm
      store_value(wf,"CostRemaGPASequential",cost[39],n_graphs);  // Print cost results for Rema after sequential suitor algorithm
    }
  }

  if (WPGASequential) {
    store_value(wf,"TimePathGrowSequential",timer[12],n_graphs); // Print timing results for sequential Path growing algorithm
    store_value(wf,"CostPathGrowSequential",cost[12],n_graphs);  // Print cost results for sequential Path growing algorithm

    if (WRama) {
      store_value(wf,"TimeRamaPGASequential",timer[34],n_graphs); // Print timing results for Rama after sequential suitor algorithm
      store_value(wf,"CostRamaPGASequential",cost[34],n_graphs);  // Print cost results for Rama after sequential suitor algorithm
    }
    if (WRoma) {
      store_value(wf,"TimeRomaPGASequential",timer[35],n_graphs); // Print timing results for Roma after sequential suitor algorithm
      store_value(wf,"CostRomaPGASequential",cost[35],n_graphs);  // Print cost results for Roma after sequential suitor algorithm
    }
    if (WRema) {
      store_value(wf,"TimeRemaPGASequential",timer[36],n_graphs); // Print timing results for Rema after sequential suitor algorithm
      store_value(wf,"CostRemaPGASequential",cost[36],n_graphs);  // Print cost results for Rema after sequential suitor algorithm
    }
  }

  if (WSuitorSortSequential) {
    store_value(wf,"TimeSortedSuitorSequential",timer[13],n_graphs); // Print timing results for sequential suitor algorithm with sorted edge lists
    store_value(wf,"CostSortedSuitorSequential",cost[13],n_graphs);  // Print cost results for sequential suitor algorithm with sorted edge lists
  }

  if (W2RSuitorDPCycSequential) {
    store_value(wf,"Time2RSuitorSequential",timer[15],n_graphs);  // Print timing results for 2 round sequential suitor algorithm
    store_value(wf,"Cost2RSuitorSequential",cost[15],n_graphs);   // Print cost results for 2 round sequential suitor algorithm

    if (WRama) {
      store_value(wf,"TimeRama2RSuitorSequential",timer[31],n_graphs); // Print timing results for Rama after 2r-sequential suitor algorithm
      store_value(wf,"CostRama2RSuitorSequential",cost[31],n_graphs);  // Print cost results for Rama after 2r-sequential suitor algorithm
    }
    if (WRoma) {
      store_value(wf,"TimeRoma2RSuitorSequential",timer[32],n_graphs); // Print timing results for Roma after 2r-sequential suitor algorithm
      store_value(wf,"CostRoma2RSuitorSequential",cost[32],n_graphs);  // Print cost results for Roma after 2r-sequential suitor algorithm
    }
    if (WRema) {
      store_value(wf,"TimeRema2RSuitorSequential",timer[33],n_graphs); // Print timing results for Rema after 2r-sequential suitor algorithm
      store_value(wf,"CostRema2RSuitorSequential",cost[33],n_graphs);  // Print cost results for Rema after 2r-sequential suitor algorithm
    }
  }

  if (W2RGreedyDPSequential) {
    store_value(wf,"Time2RGreedySequential2",timer[16],n_graphs); // Print timing results for 2 round locally greedy + DP
    store_value(wf,"Cost2RGreedySequential2",cost[16],n_graphs);  // Print cost results for 2 round locally greedy + DP
  }

  if (W2RPGADPSequential) {
    store_value(wf,"Time2RPathSequential",timer[17],n_graphs); // Print timing results for 2 round Path growing
    store_value(wf,"Cost2RPathSequential",cost[17],n_graphs);  // Print cost results for 2 round Path growing
  }

  if (W2RSuitorDPSortEdgeSequential) {
    store_value(wf,"Time2RSuitorSortedSequential",timer[18],n_graphs); // Print timing results for 2 round suitor with sorted lists + DP
    store_value(wf,"Cost2RSuitorSortedSequential",cost[18],n_graphs);  // Print cost results for 2 round suitor with sorted lists + DP
  }

  if (WSuitorCompEdgeSequential) {
    store_value(wf,"TimeSuitorCompEdgeSequential",timer[19],n_graphs); // Print timing results for sequential suitor algorithm w/compressed edges
    store_value(wf,"CostSuitorCompEdgeSequential",cost[19],n_graphs);  // Print cost results for sequential suitor algorithm w/compressed edges
  }

  if (SortNeighborListsSequential) {
    store_value(wf,"TimeEdgeListsSortSequential",timer[20],n_graphs); // Print timing results for sorting each edge list
  }

  if (WInorderGreedyVertex) {
    store_value(wf,"TimeGreedySequential",timer[21],n_graphs); // Print timing results for sequential greedy algorithm
    store_value(wf,"CostGreedySequential",cost[21],n_graphs);  // Print cost results for sequential greedy algorithm
  } 

  if (WLocalMaxSequential) {
    store_value(wf,"TimeBirnSequential",timer[22],n_graphs); // Print timing results for sequential Birn algorithm
    store_value(wf,"CostBirnSequential",cost[22],n_graphs);  // Print cost results for sequential Birn algorithm
  }

  if (WGreedySequential) {
    store_value(wf,"TimeGreedySortedSequential",timer[23],n_graphs); // Print timing results for greedy on sorted edge list
    store_value(wf,"CostGreedySortedSequential",cost[23],n_graphs);  // Print cost results for greedy on sorted edge list
  }

/*
  store_value(wf,"Time2RGreedySequential1",timer[9],n_graphs);    // Print timing results for first 2R sequential greedy algorithm
  store_value(wf,"Cost2RGreedySequential1",cost[9],n_graphs);     // Print cost results for first 2R sequential greedy algorithm
*/




// Print out parallel values
//
  if (UGreedyVertexVerifyParallel) {
    store_p_value(wf,"TimeUVertexVerifyParallel",p_timer[0],n_graphs,n_conf); // Print timing results for parallel unweigted vertex alg
  }
  if (UGreedyEdgeVerifyParallel) {
    store_p_value(wf,"TimeUEdgeVerifyParallel",p_timer[1],n_graphs,n_conf); // Print timing results for parallel unweigted edge alg
  }
  if (UGreedyEdgeVerifyPrefetchParallel) {
    store_p_value(wf,"TimeUEdgeVerifyParallel",p_timer[11],n_graphs,n_conf); // Print timing results for parallel unweigted edge prefetch alg
  }
  if (UGreedyEdgeLockParallel) {
    store_p_value(wf,"TimeUEdgeLockParallel",p_timer[2],n_graphs,n_conf); // Print timing results for parallel unweigted edge lock alg
  }

  if (WMB1ListParallel) { 
    store_p_value(wf,"TimeMB1ListParallel",p_timer[12],n_graphs,n_conf); // Print timing results for parallel MB algorithm
    store_p_value(wf,"CostMB1ListParallel",p_cost[12],n_graphs,n_conf);  // Print cost results for parallel MB algorithm
  }

  if (WMBLocalParallel) { 
    store_p_value(wf,"TimeMBLocalParallel",p_timer[13],n_graphs,n_conf); // Print timing results for parallel MB algorithm
    store_p_value(wf,"CostMBLocalParallel",p_cost[13],n_graphs,n_conf);  // Print cost results for parallel MB algorithm
  }
  
  if (WSuitorParallel) { 
    store_p_value(wf,"TimeSuitorParallel",p_timer[4],n_graphs,n_conf); // Print timing results for parallel suitor algorithm
    store_p_value(wf,"CostSuitorParallel",p_cost[4],n_graphs,n_conf);  // Print cost results for parallel suitor algorithm
  }
  
  if (WStackParallel) { 
    store_p_value(wf,"TimeSuitorStackParallel",p_timer[15],n_graphs,n_conf); // Print timing results for parallel suitor algorithm using stack
    store_p_value(wf,"CostSuitorStackParallel",p_cost[15],n_graphs,n_conf);  // Print cost results for parallel suitor algorithm using stack
  }

  if (WQueueParallel) { 
    store_p_value(wf,"TimeSuitorQueueParallel",p_timer[16],n_graphs,n_conf); // Print timing results for parallel suitor algorithm using queue
    store_p_value(wf,"CostSuitorQueueParallel",p_cost[16],n_graphs,n_conf);  // Print cost results for parallel suitor algorithm queue
  }
  
  if (W2RSuitorDPParallel) { 
    store_p_value(wf,"Time2RSuitorParallel",p_timer[5],n_graphs,n_conf); // Print timing results for 2 round parallel suitor algorithm
    store_p_value(wf,"Cost2RSuitorParallel",p_cost[5],n_graphs,n_conf);  // Print cost results for 2 round parallel suitor algorithm
  }

  if (WSuitorSortParallel) { 
    store_p_value(wf,"TimeSortedSuitorParallel",p_timer[6],n_graphs,n_conf); // Print timing results for parallel suitor algorithm
    store_p_value(wf,"CostSortedSuitorParallel",p_cost[6],n_graphs,n_conf);  // Print cost results for parallel suitor algorithm
  }

  if (W2RSuitorSortParallel) { 
    store_p_value(wf,"Time2RSortedSuitorParallel",p_timer[7],n_graphs,n_conf); // Print timing results for parallel suitor algorithm
    store_p_value(wf,"Cost2RSortedSuitorParallel",p_cost[7],n_graphs,n_conf);  // Print cost results for parallel suitor algorithm
  }

  if (WLocalMax1ListParallel) { 
    store_p_value(wf,"TimeLocalMax1LParallel",p_timer[8],n_graphs,n_conf); // Print timing results for parallel local max w/1 list
    store_p_value(wf,"CostLocalMax1LParallel",p_cost[8],n_graphs,n_conf);  // Print cost results for parallel local max w/1 list
  }

  if (WLocalMaxLocalListParallel) { 
    store_p_value(wf,"TimeLocalMaxLLParallel",p_timer[9],n_graphs,n_conf); // Print timing results for parallel local max w/local list
    store_p_value(wf,"CostLocalMaxLLParallel",p_cost[9],n_graphs,n_conf);  // Print cost results for parallel local max w/local list
  }

  if (SortEdgeListsParallel) {
    store_p_value(wf,"TimeSortEdgeListParallel",p_timer[10],n_graphs,n_conf); // Print timing results for sorting edge lists in paralle
  }

  if (W2MatchingParallel) {
    store_p_value(wf,"Time2MatchParallel",p_timer[14],n_graphs,n_conf); // Print timing results for parallel 2-matching
//    store_p_value(wf,"Cost2MatchParallel",p_cost[14],n_graphs,n_conf);  // Print cost results for parallel 2-matching
  }
/*
  fprintf(wf,"s_vx = [");       // Print results for sequential vertex based matching to file
  for(i=0;i<n_graphs;i++)  {
    fprintf(wf,"%lf ",timer[0][i]);
  }
  fprintf(wf,"];\n");

  fprintf(wf,"s_ed = [");       // Print results for sequential edge based matching to file
  for(i=0;i<n_graphs;i++)  {
    fprintf(wf,"%lf ",timer[1][i]);
  }
  fprintf(wf,"];\n");

  fprintf(wf,"p_vx = [");       // Print results for parallel vertex based matching to file
  for(i=0;i<n_graphs;i++)  {
    for(j=0;j<n_conf;j++)  {
      fprintf(wf,"%lf ",t_ver[i][j]);
    }
    if (i != n_graphs-1)
      fprintf(wf,";\n");
  }
  fprintf(wf,"];\n");

  fprintf(wf,"p_ex = [");       // Print results for parallel edge based matching to file
  for(i=0;i<n_graphs;i++)  {
    for(j=0;j<n_conf;j++)  {
      fprintf(wf,"%lf ",t_edge[i][j]);
    }
    if (i != n_graphs-1)
      fprintf(wf,";\n");
  }
  fprintf(wf,"];\n");

  fprintf(wf,"p_lc = [");       // Print results for parallel lock based matching to file
  for(i=0;i<n_graphs;i++)  {
    for(j=0;j<n_conf;j++)  {
      fprintf(wf,"%lf ",t_lock[i][j]);
    }
    if (i != n_graphs-1)
      fprintf(wf,";\n");
  }
  fprintf(wf,"];\n");
*/
  fclose(wf);
  printf("************************************************\n");
  printf("***               Normal exit                ***\n");
  printf("************************************************\n");
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
	  /* printf("%d %d %lf ", x, y ,v); */
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

		/* printf("%lf ", (*we)[num_edges].w); */
    
        if (fabs(v) < 0.0)  {
          printf("error reading data,got %f \n",fabs(v));
          return;
        }

		/* @OGUZ-EDIT commented out random edge weight */
         /* (*we)[num_edges].w = 1.0 + (double)rand()/100000.0; */
		 
      //  (*we)[num_edges].w = (double)rand();
        count[x]++; // Get vertex degrees
        count[y]++;

		/* printf("%lf \n", (*we)[num_edges].w); */

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
