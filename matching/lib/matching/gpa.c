// The GPA algorithm by Maue and Sanders

void gpa(int n,int m,wed *we,int *ver,int *edges,double *weight,int *match) {

  int *parity;
  int *other;
  int *right;
  int *left;
  double *wr;
  double *wl;
  int v,w;
  int *path1;
  int *path2;
  double *weight1;
  double *weight2;
  double dyn_prog();
  int i,k;
  int current, next;
  int tmp;
  int length1,length2;
  int partner;
  double two_weight = 0.0;

  parity = (int *) malloc((n+1) * sizeof(int));
  other  = (int *) malloc((n+1) * sizeof(int));
  right  = (int *) malloc((n+1) * sizeof(int));
  left   = (int *) malloc((n+1) * sizeof(int));
  wr     = (double *) malloc((n+1) * sizeof(double));
  wl     = (double *) malloc((n+1) * sizeof(double));

  path1   = (int *) malloc((n+1) * sizeof(int));
  path2   = (int *) malloc((n+1) * sizeof(int));
  weight1 = (double *) malloc((n+1) * sizeof(double));
  weight2 = (double *) malloc((n+1) * sizeof(double));

// Iterate through the edges (in order). For each edge check if it completes an even length cycle or
// if it connects two paths.


// First initialize variables for each vertex
  for(i=1;i<=n;i++) {
    parity[i] = true;
    other[i] = i;
    right[i] = 0;
    left[i] = 0;
    match[i] = 0;
  }

// Then iterate through the edges

  for(i=0;i<m;i++) {
    v = we[i].x;
    w = we[i].y;

// If either of v and w is not a path endpoint then skip this edge
    if (((right[v] != 0) && (left[v] != 0)) || ((right[w] != 0) && (left[w] != 0)))
      continue;

// Check if (v,w) completes a cycle
    if (other[v] == w) {  // we have a cycle

      if (parity[v])  // If we create an odd cycle then skip to next edge
        continue;

      two_weight += we[i].w;

      parity[v] = true; // Set that this cycle is even
      parity[w] = true;

// Create the cycle
      if (right[v] == 0) {  // First link v to w, must find which pointer to use
        right[v] = w;
        wr[v] = we[i].w;  // Store the weight of the right edge of v
      }
      else {
        left[v] = w;
        wl[v] = we[i].w;  // Store the weight of the left edge of v
      }

      if (right[w] == 0) {  // Now connect w to v
        right[w] = v;
        wr[w] = we[i].w;
      }
      else {
        left[w] = v;
        wl[w] = we[i].w;
      }
      other[v] = -1;  // Set that this is a cycle
      continue;
    }  // End treatment of cycle

// (v,w) does not complete a cycle, v and w must be endpoints, ok to merge lists

    two_weight += we[i].w;

    parity[other[v]] = parity[v]^parity[w];  // Performing logical XOR on the parity
    parity[other[w]] = parity[other[v]];     // Storing the new parity with the new endpoints

    int tmp = other[v];
    other[tmp] = other[w];  // Update the remaining endpoints
    other[other[w]] = tmp;

// Link the lists, first v to w and then w to v
    if (right[v] == 0) {
      right[v] = w;
      wr[v] = we[i].w;
    }
    else {
      left[v] = w;
      wl[v] = we[i].w;
    }
    if (right[w] == 0) {
      right[w] = v;
      wr[w] = we[i].w;
    }
    else {
      left[w] = v;
      wl[w] = we[i].w;
    }

  } // End loop over m edges


// Now pick out the paths and cycles and do dynamic programming on them
  for(i=1;i<=n;i++) {
    if (other[i] == -1) {// This is a cycle

// Processing the first vertex

      path1[0] = i;
      weight1[0] = wr[i];
      tmp = i;
      current = right[i]; // Initially start by moving to the right
      length1 = 0;

// Now tmp and current point to two consecutive nodes on the cycle, tmp has been processed.

// Pick up rest of cycle
      while (current != i) {
        length1++;
        path1[length1] = current;    // Put current in the list
        if (right[current] == tmp) { // Moving next forward to v(i+1)
          next = left[current];
          weight1[length1] = wl[current];  // Picking up weight of edge (current,next)
        }
        else {
          next = right[current];
          weight1[length1] = wr[current];
        }
        tmp = current;
        current = next;

      } // End while

// Add vertex i at the end of path1
      length1++;
      path1[length1] = current;   // Placing vertex i at the end of the list

// Now do dynamic programming
      cycle_dyn_prog(length1,path1,weight1,match);

      continue;
    } // End treatment of cycle

    if ((right[i] != 0) && (left[i] !=0)) // This is not an endpoint, skip it
      continue;

    if ((right[i] == 0) && (left[i] ==0)) // This is a singleton, skip it
      continue;

    if (other[i] == -2) // This is a path that has been handled from the other endpoint
      continue;

// We now have a path starting in i, must pick up every vertex and mark other endpoint with -2
    current = i;
    if (right[i] == 0) {
      next = left[i];
      weight1[0] = wl[i];
    }
    else {
      next = right[i];
      weight1[0] = wr[i];
    }
    length1 = 0;
    while (current != other[i]) { // Move through the path picking up vertices and edge weights
      path1[length1] = current;
      length1++;
      int tmp = next;
      if (right[next] != current) {
        next = right[next];
        weight1[length1] = wr[tmp];
      }
      else {
        next = left[next];
        weight1[length1] = wl[tmp];
      }
      current = tmp;
    }
    path1[length1] = current; // Add the final vertex
    other[current] = -2;     // Mark last vertex so we only traverse path in one direction

    dyn_prog(length1,path1,weight1,match);

  }

// Match any node that was left unmatched after the DP

  for(i=1;i<=n;i++) {
    if (match[i] == 0) {
      double best = -1.0;
      for(k=ver[i];k<ver[i+1];k++) {
        int y = edges[k];
        if ((match[y] == 0) && (weight[k] > best)) {
          best = weight[k];
          partner = y;
        }
      }
      if (best > 0.0) {
        match[i] = partner;
        match[partner] = i;
      }
    }
  }

  free(parity);
  free(other);
  free(right);
  free(left);
  free(wr);
  free(wl);
  free(path1);
  free(path2);
  free(weight1);
  free(weight2);

  // printf("GPA: Weight of 2-matching is %f \n",two_weight);
}
