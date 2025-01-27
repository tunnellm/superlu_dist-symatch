// 2-augmenting code
//
// This code assumes that s contains a matching with ws giving the weight
//

int random_order(int n,int *order);

void roma(int n,int *ver,int *edges,int *s,double *ws,double *weight,int *p1) {
  void roma_driver();
  double cost_matching();

  double old_cost = cost_matching(n,ver,edges,weight,s);     // Get the cost of the old matching
  printf("Cost before roma is %8.1lf \n",old_cost);

  double mt1 = omp_get_wtime();

  roma_driver(n,ver,edges,s,ws,weight,p1);                  // Roma type post-processing

  double mt2 = omp_get_wtime();
  double timeRoma = mt2 - mt1;

  double new_cost = cost_matching(n,ver,edges,weight,s);     // Get the cost of the matching

  printf("************************************************\n");
  printf("*  After roma %8.1lf, improvement %4.2lf %%   *\n", new_cost,(new_cost-old_cost)/old_cost*100.0);
  printf("*  Time rama %8.6lf                         *\n", timeRoma);
  printf("************************************************\n");
}

void roma_driver(int n,int *ver,int *edges,int *s,double *ws,double *weight,int *init) {
  int i,j,x;
  void addEdgeR();
  double cost_matching();

  int *mark;
  double *cw;
  int ip = 0;
  int s1p = 0;
  double wip = 0.0;
  double ws1p = 0.0;
  int s1 = 0;
  double bestWeight;

  int *rQ; //,*wQ,*round;

  int debug = false;
  int probRound = 1;

  rQ = (int *) malloc(sizeof(int)*(n+2));
  //wQ = (int *) malloc(sizeof(int)*2*(n+1));
  mark = (int *) malloc(sizeof(int)*(n+2));
  cw = (double *) malloc(sizeof(double)*(n+2));
  //round = (int *) malloc(sizeof(int)*(n+1));


  int rLast = n;        // The number of vertices to process in the current round

  int curRound = 0;     // The number of the current round

  int maxRounds = 8;

  for(curRound=0;curRound<maxRounds;curRound++) {     // Do at most maxRounds
    printf("Round %d, %d items \n",curRound,rLast);

//    double old_cost = cost_matching(n,ver,edges,weight,s);     // Get the cost of the old matching

    int nrChanged = 0;

    for(i=0;i<=n;i++) {   // Should start from 0
      mark[i] = 0;        // Used to find 4-cycles, set to v if vertex i is reachable from vertex v
    }

// Compute a random permutation of 1..n and store in rQ (position 0 is not used)
    random_order(n,rQ);

    for(i=1;i<=n;i++) {   // Iterate through the vertices in rQ
      int v = rQ[i];
      s1 = s[v];             // Processing edge (v,s1), note s1 could be 0 if v is not matched

      if ((debug) && (curRound == probRound))
        if (i==1)
          printf("%d is matched with %d \n",v,s1);

// First find the best 4-cycle involving (v,s1)
      bestWeight = ws[v];   // To be used it must have larger gain than w(v,s[v])

      // printf("(%d,%d): Initial bestWeight = %lf \n",v,s1,bestWeight);

      if (s1) {             // Only need this if v is matched
        for(j=ver[v];j<ver[v+1];j++) {  // Mark every neighbor of v
          x  = edges[j];
          mark[x] = v;
          cw[x] = weight[j];  // Store the weight of (v,x) with x
        }

        for(j=ver[s1];j<ver[s1+1];j++) {  // Check every neighbor of s1
          x  = edges[j];
          if (x == v)
            continue;

          if (mark[s[x]] == v) {  // If x is not matched then s[x] == 0, note v != 0

          // Have found 4-cycle v,s1,x,s[x], now check weight

            if (weight[j] - ws[x] + cw[s[x]] > bestWeight) {  // Check for new best weight
              bestWeight = weight[j] - ws[x] + cw[s[x]];
              if ((debug) && (curRound == probRound)) {
                printf("Setting bestWeight to %lf - %lf + %lf = %lf \n",weight[j],ws[x],cw[s[x]],bestWeight);
                printf("Cycle is now %d, %d, %d, %d \n",v,s1,x,s[x]);
              }
              ip  = s[x];       // Store v´s best partner
              wip = cw[s[x]];   // Store w(v,ip)
              s1p = x;          // Store s1´s best partner
              ws1p = weight[j]; // Store w(s1,x)
            }
          }
        } // Neighbors of s1
      } // if s1

    // Done with the 4-cycles
    // Have found a positive gain cycle if bestWeight > ws[i]
    // Now find the best 2-augmenting path including v

    // printf("%d: %d is matched to %d with value %lf \n",i,i,s[i],ws[i]);
    // printf("%d: Best cycle gives %lf compared to existing value %lf \n",i,bestWeight, ws[i]);

      double gain = 0.0;         // How much can be gained from a particular arm

      int p1 = 0;                // First vertex of best arm
      double w1 = 0.0;           // Weight of best arm
      double wp1;                // Weight of first edge of best arm

      int p2 = 0;                // First vertex of second best arm
      double w2 = 0.0;           // Weight of second best arm
      double wp2;                // Weight of first edge of second best arm
      double up1,uw1,uwp1;

    // First find the two best arms from v

      if ((debug) && (curRound == probRound))
        if (i==0)
          printf("BestWeight is now %lf \n",bestWeight);

      for(j=ver[v];j<ver[v+1];j++) {  // Check every neighbor of v
        x  = edges[j];
        if (x == s[v])  // No need to check existing partner of v
          continue;

// Compute gain of new arm v - x - s[x]  (s[x] could be 0, but then s[x] = 0.0)

        gain = weight[j] - ws[x];  // we add w(v,x) - w(x,s(x)) to the matching
                                   // Note, if x is not matched then ws[x] == 0.0

        if (gain <= w2)     // Must have gain larger than second best
          continue;

// Keep the two best arms,

        if (gain > w1) { // If new best
          w2 = w1;       // The best arm now becomes second best
          p2 = p1;       // First vertex
          wp2 = wp1;     // Weight of first edge

          w1 = gain;       // Weight of best arm
          p1 = x;          // First vertex
          wp1 = weight[j]; // Weight of first edge
        }
        else  { // If not best, then it is second best
          w2 = gain;
          p2 = x;
          wp2 = weight[j];
        }
      } // loop over neighbors of v

      if ((debug) && (curRound == probRound))
        printf("Best arm from %d has weight %lf and goes to vertex %d \n",v,w1,p1);

      // It could be that w1 is better than the best cycle
      // This also prevents negative weights arms from s1
      if (w1 > bestWeight) {
        bestWeight = w1;
        ip  = p1;          // Store v´s best partnr
        wip = wp1;         // Store the weight of (v,ip)

        if ((debug) && (curRound == probRound))
          printf("Best arm : (v=%d,%d),Setting bestWeight to %lf= %lf - %lf \n",v,p1,bestWeight,wp1,ws[p1]);
      }

    // Note that if v has no other neighbor than s[v] then w1 = 0.0

    // Now find the best arm of s[v], only makes sense if v is matched (s1 != 0)
      if (s1) {

        for(j=ver[s1];j<ver[s1+1];j++) {  // Check every neighbor of s1
          x  = edges[j];
          if (x == v)
            continue;      // No point in checking v

          gain = weight[j] - ws[x];  // we add w(s1,x) - w(x,s(x)) to the matching
                                     // Note, if x is not matched then ws[x] == 0.0

          if (gain < 0.0)  // Must have positive gain to be used
            continue;

      // Which arm of v to use (note there can be 0,1 or 2 arms)
      // Know that x != 0, but p1 or p2 could be undefined (i.e. 0)

          if (x == p1) { // Check for triangel v - s1 - x==p1 - v
            up1 = p2;     // Using p2
            uw1 = w2;
            uwp1 = wp2;
          } else {
            up1 = p1;     // Using p1
            uw1 = w1;
            uwp1 = wp1;
          }

          // p1, w1, wp1 now denotes the arm we use, which might be the second best one
          // If no arm then p1 == 0 and w1 == 0.0

          if (gain + uw1 > bestWeight) {
            bestWeight = gain + uw1;
            if ((debug) && (curRound == probRound)) {
              printf("gain from s1 arm =  %lf - %lf = %lf \n",weight[j],ws[x],gain);
              printf("(s1=%d,%d): Setting bestWeight to %lf+%lf = %lf \n",s1,x,gain,uw1,bestWeight);
            }
            ip  = up1;          // Store v´s best partnr
            s1p = x;           // Store s1´s best partner
            wip = uwp1;         // Store the weight of (v,ip)
            ws1p = weight[j];  // Store the weight of (s1,s1p)
          }
        } // loop over neighbors of s1
      } // if v is matched

   // Have possible path ip - v - s1 - s1p

   // Have found positive path if bestPWeight > ws[v]

   // Check if the best gain is larger than what we loose ws[v]=w(v,s1), if v is unmatched then ws[v]==0.0

      if (bestWeight > ws[v]) {  // True if we found 1 or 2 arms with positive gain larger than ws[v]
        if (ip == s1) {
          printf("Error,vertex %d in queue v = %d is already matched with %d \n",i,v,s[v]);
          printf("when trying to match %d with %d \n",v,ip);
          printf("bestWeight=%lf and ws[v]= %lf \n",bestWeight,ws[v]);
          for(i=ver[v];i<ver[v+1];i++)
            if (edges[i]==s[v])
              printf("True edge weight: %lf \n",weight[i]);
          exit(0);
        }
        nrChanged++;
        if (ip)  // != 0 if there is an arm from v
          addEdgeR(s,ws,v,ip,wip);
        if (s1p) // != 0 if there is an arm from s1
          addEdgeR(s,ws,s1,s1p,ws1p);
      }

      // double new_cost = cost_matching(n,ver,edges,weight,s);     // Get the cost of the old matching

      //if (new_cost < old_cost) {
      //  printf("old cost is %8.1lf while new cost is %8.1lf \n",old_cost,new_cost);
      //  printf("Processed vertex in place %d with index %d \n",i,v);
      //  exit(0);
      //}
    } // Loop over all vertices

    // Swap the read and the write list
//    int *t = rQ;
//    rQ = wQ;
//    wQ = t;
//    rLast = wLast;
    printf("*** Round %d, Numbers changed %d \n",curRound,nrChanged);
    double cm = cost_matching(n,ver,edges,weight,s);
    printf("Current cost of matching is %8.1lf \n",cm);
    if (nrChanged == 0)
      break;

  } // while rQ is not empty
}

void addEdgeR(int *s,double *ws,int x,int y,double newW) {
   // printf("Replacing (%d,%d) and (%d,%d)=(%d,%d) \n",x,s[x],y,s[y],s[y],s[s[y]]);
   // printf("of weight %lf=%lf  and %lf=%lf \n",ws[x],ws[s[x]],ws[y],ws[s[y]]);
   // printf("Matching %d and %d of weight %lf \n",x,y,newW);
  s[s[x]] = 0;
  s[s[y]] = 0;
  ws[s[x]] = 0.0;
  ws[s[y]] = 0.0;
  int a = s[x];
  int b = s[y];
  s[x] = y;
  s[y] = x;
  ws[x] = newW;
  ws[y] = newW;
/*
  if ((a != 0) && (round[a] <= curRound)) {
    round[a] = curRound+1;
    wQ[*wLast] = a;
    *wLast = *wLast + 1;
  }
  if ((b != 0) && (round[b] <= curRound)) {
    round[b] = curRound+1;
    wQ[*wLast] = b;
    *wLast = *wLast + 1;
  }
*/
}
