// weighted algorithm
//
// Optimized main loop
// Two round matching followed by dynamic programming
// This time the dynamic programming is done in a separate routine
// One routine for the paths, and one routine for the cycles

void cycle_dyn_prog(int l1,int *path1,double *weight1,int *match);

void sweight8(int n,int *ver,int *edges,int *s,double *ws,int *s2, double *ws2,double *weight, int *used,int *match) {


  int i,j;
  int partner;     // Prospective candidate to match with
  int x, done, next_vertex;
  int count,current;
  int *path1;
  int l1,l2;
  double *weight1;
  double cum_weight = 0.0;
  double dyn_prog();


  path1 = (int *) malloc(n * sizeof(int));
  weight1 = (double *) malloc(n * sizeof(double));

// Start of matching algorithm

  for(i=0;i<=n;i++) {
    s[i] = 0;           // Set that no node is trying to match with i
    ws[i] = 0.0;        // Set the current weight of best suitor edge to 0

    s2[i] = 0;           // Set that no node is trying to match with i
    ws2[i] = 0.0;        // Set the current weight of best suitor edge to 0

    used[i] = false;     // Set that node i has not been used in the dynamic programming part
    match[i] = 0;
  }
//  count = 0;

// First round of matching

  i = 1;
  current = i;

  while (current != n+1) {
    double heaviest = ws[current];          // Find the heaviest candidate that does not have a better offer
    partner = s[current];                  // No point in trying a partner worse than the best suitor

    for(j=ver[current];j<ver[current+1];j++) { // Loop over neighbors of the current vertex
      int y = edges[j];            // y is the neighbor of the current vertex
      if ((weight[j]>heaviest) && (ws[y]<weight[j])) {// Check if w(current,y) is the best so far, and if it is a better option for y
        heaviest = weight[j];      // Store the weight of the heaviest edge found so far
        partner = y;               // Store the name of the associated neighbor
      } // loop over neighbors
    }


    if (heaviest > 0) {            // True if there is a new partner
      if (s[partner] != 0) {       // True if partner already had a suitor
        next_vertex = s[partner];  // Pick up the old suitor and continue
      }
      else {
        i = i + 1;                 // Move to the next vertex
        next_vertex = i;
      }
      s[partner] = current;        // current is now the current suitor of s
      ws[partner] = heaviest;      // The weight of the edge (current,partner)
    }
    else {
      i = i + 1;                   // Move to the next vertex
      next_vertex = i;
    }
    current = next_vertex;         // Update current vertex with the next one to work on
  } // loop over vertices


// Second round of matching

  i = 1;
  current = i;
  while (current != n+1) {
    double heaviest = ws2[current];          // Find the heaviest candidate that does not have a better offer
    partner = s2[current];                  // No point in trying a partner worse than the best suitor

    for(j=ver[current];j<ver[current+1];j++) { // Loop over neighbors of the current vertex
      int y = edges[j];            // y is the neighbor of the current vertex
      // if (y == s[current]) continue;
      if ((weight[j]>heaviest) && (ws2[y]<weight[j]) && (y != s[current])) {// Check if w(current,y) is the best so far, and if it is a better option for y. It must also be different from the vertex selected in stage 1
        heaviest = weight[j];      // Store the weight of the heaviest edge found so far
        partner = y;               // Store the name of the associated neighbor
      } // loop over neighbors
    }


    if (heaviest > 0) {            // True if there is a new partner
      if (s2[partner] != 0) {       // True if partner already had a suitor
        next_vertex = s2[partner];  // Pick up the old suitor and continue
      }
      else {
        i = i + 1;                 // Move to the next vertex
        next_vertex = i;
      }
      s2[partner] = current;        // current is now the current suitor of s
      ws2[partner] = heaviest;      // The weight of the edge (current,partner)
    }
    else {
      i = i + 1;                   // Move to the next vertex
      next_vertex = i;
    }
    current = next_vertex;         // Update current vertex with the next one to work on
  } // loop over vertices


// Starting dynamic programming

  for(i=1;i<=n;i++) {  // Check each vertex as a starting point of a path
    if (used[i]) continue;  // If this vertex has been used previously then skip it

    if ((s2[i] == 0) && (s[i] != 0)) {  // Starting with a level 1 edge
      l1 = 0;  			// Keep track of the length of this path, starting from 0
      path1[l1] = i;  		// Store the first vertex
      weight1[l1] = ws[i];	// Store the weight of the first edge
      used[i] = true;		// Mark the vertex as used
      current = s[i];   	// Move to the next vertex

      while (true) {   // Standing at the endpoint of an edge from the level 1 matching
        used[current] = true;   // Set the current vertex as used
        l1++;			// Increase the length of the path
        path1[l1] = current;  	// Store this vertex
        if (s2[current] == 0)   // If the path does not continue then move to next path
          break;
        weight1[l1] = ws2[current];	// Store the weight of the next edge

        current = s2[current];    // Move along the path using an edge from the level 2 matching
        used[current] = true;     // Mark new vertex as used
        l1++;
        path1[l1] = current;      // Store the next vertex
        if (s[current] == 0)      // If the path ends, move to next path
          break;
        weight1[l1] = ws[current];	// Store the weight of the next edge

        current = s[current]; // Move to the next vertex on the path
      } // End while

      dyn_prog(l1,path1,weight1,match);

    } // End if
    else if ((s2[i] != 0) && (s[i] == 0)) {  // Starting with a level 2 edge
      l1 = 0;  			// Keep track of the length of this path, starting from 0
      path1[l1] = i;  		// Store the first vertex
      weight1[l1] = ws2[i];	// Store the weight of the first edge
      used[i] = true;		// Mark the vertex as used
      current = s2[i];   	// Move to the next vertex

      while (true) {   // Standing at the endpoint of an edge from the level 2 matching
        used[current] = true;   // Set the current vertex as used
        l1++;			// Increase the length of the path
        path1[l1] = current;  	// Store this vertex
        if (s[current] == 0)    // If the path does not continue then move to next path
          break;
        weight1[l1] = ws[current];	// Store the weight of the next edge

        current = s[current];     // Move along the path using an edge from the level 2 matching
        used[current] = true;     // Mark new vertex as used
        l1++;
        path1[l1] = current;      // Store the next vertex
        if (s2[current] == 0)      // If the path ends, move to next path
          break;
        weight1[l1] = ws2[current];	// Store the weight of the next edge

        current = s2[current]; // Move to the next vertex on the path
      } // End while

      dyn_prog(l1,path1,weight1,match);

    }
    else if ((s2[i] == 0) && (s[i] == 0)) {  // Found single vertex
      used[i] = true;
    }
  }

// Now look for the cycles
  int nr_cycle = 0;
  for(i=1;i<=n;i++) {  // Check each vertex as a starting point of a path
    if (used[i]) continue;  // If this vertex has been used previously then skip it

    nr_cycle++;
    current = i;
    l1 = 0;

// Ready to start loop

    while (!used[current]) {   // Process two edges (vi,v(i+1)) and (v(i+1),v(i+2)), at a time until we reach the start of the cycle

// Put vertex v_i and edge (vi,v(i+1)) in the path

      used[current] = true;    		// Set vi as used
      path1[l1] = current;		// Put vi in path1
      weight1[l1] = ws[current];	// Put weight of (vi,v(i+1)) in the path

      current = s[current];    		// Move to v(i+1)

      l1++;				// Increase number of vertices
      used[current] = true;    		// Set v(i+1) as used
      path1[l1] = current;		// Put v(i+1) in path1
      weight1[l1] = ws2[current];	// Put weight of (v(i+1),v(i+2) in the path

      current = s2[current]; 		// Move to v(i+2)
      l1++;
    } // End while loop

    path1[l1] = path1[0];		// Put the first vertex last to complete the cycle


// Now do dynamic programming on the cycle
    cycle_dyn_prog(l1,path1,weight1,match);

  } // End of loop over vertices that looks for unprocessed cycles

  int k;
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

  free(path1);
  free(weight1);

}

// Dynamic programming on a weighted cycle to find the heaviest matching
// The path contains l1 vertices, stored in path1[]
// and l1 edges, the weights of these are stored in match[].
// The final edge goes from vertex path1[l1-1] to vertex path1[0].

void cycle_dyn_prog(int l1,int *path1,double *weight1,int *match) {

  double temp;
  double w1_old = 0.0;
  double w1_new = weight1[0];
  double w2_old = 0.0;
  double w2_new = weight1[1];
  int k;
  int in1[l1];
  int in2[l1];

// The first path (#1) ignores the last edge.
// The second path (#2) ignores the first edge.

  in1[0] = true;   // The 0'th edge is in the first solution
  in2[0] = true;   // The 1st edge is in the second solution

  for(k=1;k<l1-1;k++) {
// Process the first path
    if (weight1[k] + w1_old > w1_new) {
      temp = weight1[k] + w1_old;
      w1_old = w1_new;
      w1_new = temp;
      in1[k] = true;   // The k'th edge is in the solution
    }
    else {
      w1_old = w1_new;
      in1[k] = false;  // The k'th edge is out of the solution
    }
// Process the second path
    if (weight1[k+1] + w2_old > w2_new) {
      temp = weight1[k+1] + w2_old;
      w2_old = w2_new;
      w2_new = temp;
      in2[k] = true;   // The k+1'th edge is in the solution
    }
    else {
      w2_old = w2_new;
      in2[k] = false;  // The k+1'th edge is out of the solution
    }
  }

// Now w1_new contains the cost of the path where the last edge is ignored
// while w2_new contains the cost of the path where the first edge is ignored


// Now match up the vertices that belong to the final solution

  if (w1_new > w2_new) {
    k = l1-2;
    while (k >= 0) {
      if (in1[k]) {
        match[path1[k]] = path1[k+1];
        match[path1[k+1]] = path1[k];
        k = k - 2;
      }
      else {
        k = k - 1;
      }
    }
  }
  else {
    k = l1-2;
    while (k >= 0) {
      if (in2[k]) {
        match[path1[k+1]] = path1[k+2];
        match[path1[k+2]] = path1[k+1];
        k = k - 2;
      }
      else {
        k = k - 1;
      }
    }
  }

}
