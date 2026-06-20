/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/




/*! @file
 * \brief Solves a system of distributed linear equations A*X = B with a
 * general N-by-N matrix A using the LU factors computed previously.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 15, 2008
 * September 18, 2018  version 6.0
 * February 8, 2019  version 6.1.1
 * </pre>
 */
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "superlu_ddefs.h"
#include "superlu_upacked.h"
#define ISEND_IRECV

static size_t
pdgstrs3d_checked_product(size_t a, size_t b, const char *what)
{
    (void) what;
    if (a != 0 && b > ((size_t)-1) / a)
        ABORT("Workspace size overflows allocation size.");
    return a * b;
}

static int_t
pdgstrs3d_checked_workspace_count(int_t a, int_t b, int_t c, int_t d,
                                  const char *what)
{
    if (a < 0 || b < 0 || c < 0 || d < 0)
        ABORT("Negative workspace size.");

    size_t first = pdgstrs3d_checked_product((size_t) a, (size_t) b, what);
    size_t second = pdgstrs3d_checked_product((size_t) c, (size_t) d, what);
    if (first > ((size_t)-1) - second)
        ABORT("Workspace size overflows allocation size.");

    size_t total = first + second;
    int_t out = (int_t) total;
    if (out < 0 || (size_t) out != total)
        ABORT("Workspace size overflows int_t.");
    return out;
}

static int_t
pdgstrs3d_checked_size_to_int_t(size_t count, const char *what)
{
    (void) what;
    int_t out = (int_t) count;
    if (out < 0 || (size_t) out != count)
        ABORT("Workspace size overflows int_t.");
    return out;
}

static size_t
pdgstrs3d_checked_alloc_bytes(int_t count, size_t elem_size,
                              const char *what)
{
    if (count < 0)
        ABORT("Negative allocation size.");
    size_t n = (size_t) count;
    if ((int_t) n != count)
        ABORT("Allocation size overflows int_t.");
    return pdgstrs3d_checked_product(n, elem_size, what);
}

// Broadcast the RHS to all grids from grid 0
int_t dtrs_B_init3d(int_t nsupers, double* x, int nrhs, dLUstruct_t * LUstruct,
	gridinfo3d_t *grid3d)
{

	gridinfo_t * grid = &(grid3d->grid2d);
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );

	for (int_t k = 0; k < nsupers; ++k)
	{
		/* code */
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow && mycol == kcol)
		{
			int_t lk = LBi(k, grid);
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize(k);
			MPI_Bcast( &x[ii - XK_H], knsupc * nrhs + XK_H, MPI_DOUBLE, 0, grid3d->zscp.comm);

		}
	}

	return 0;
}

// Broadcast the RHS to all grids from grid 0. Once received, every grid zeros out certain subvectors to allow for the new 3D solve.
int_t dtrs_B_init3d_newsolve(int_t nsupers, double* x, int nrhs, dLUstruct_t * LUstruct,
	gridinfo3d_t *grid3d, dtrf3Dpartition_t*  trf3Dpartition)
{

	gridinfo_t * grid = &(grid3d->grid2d);
    int_t myGrid = grid3d->zscp.Iam;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );
    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    double zero = 0.0;
    double* xtmp;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t Pr = grid->nprow;
    int_t nlb = CEILING (nsupers, Pr);    /* Number of local block rows. */

    int_t x_count = pdgstrs3d_checked_workspace_count(Llu->ldalsum, nrhs,
                                                      nlb, XK_H,
                                                      "3D solve xtmp workspace");
    if (!(xtmp = doubleCalloc_dist (x_count)))
    ABORT ("Malloc fails for xtmp[].");

	for (int_t k = 0; k < nsupers; ++k)
	{
		/* code */
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow && mycol == kcol)
		{
			int_t lk = LBi(k, grid);
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize(k);
            MPI_Bcast( &x[ii - XK_H], knsupc * nrhs + XK_H, MPI_DOUBLE, 0, grid3d->zscp.comm);
            for (int_t i=0; i<XK_H; ++i){
                xtmp[ii-XK_H+i] = x[ii - XK_H+i];
            }
            for (int_t i=0; i<knsupc * nrhs; ++i){
                xtmp[ii+i] = x[ii+i];
                x[ii+i] = zero;
            }
		}
	}


    // fill corresponding RHSs
    for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        // printf("gana grid3d->zscp.iam %5d ilvl %5d myZeroTrIdxs[ilvl] %5d myTreeIdxs[ilvl] %5d\n",grid3d->zscp.Iam, ilvl, myZeroTrIdxs[ilvl],myTreeIdxs[ilvl]);
        if (!myZeroTrIdxs[ilvl])
        {
            int_t tree = myTreeIdxs[ilvl];
            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];
            /*main loop over all the super nodes*/
            if (sforest)
            {
                int_t nnodes = sforest->nNodes ;
	            int_t *nodeList = sforest->nodeList ;
                for (int_t k0 = 0; k0 < nnodes; ++k0)
	            {
		            int_t k = nodeList[k0];
                    int_t krow = PROW (k, grid);
                    int_t kcol = PCOL (k, grid);

                    if (myrow == krow && mycol == kcol)
                    {
                        int_t lk = LBi(k, grid);
                        int_t ii = X_BLK (lk);
                        int_t knsupc = SuperSize(k);
                        for(int_t i=0; i<knsupc * nrhs; ++i)
                            x[ii +i]= xtmp[ii+i];
                    }
                }
            }
        }
    }
    SUPERLU_FREE (xtmp);
	return 0;
}

// #ifdef HAVE_NVSHMEM
/*global variables for nvshmem, is it safe to be put them here? */
double *dready_x, *dready_lsum;
// #endif

int dtrs_compute_communication_structure(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                           dScalePermstruct_t * ScalePermstruct,
                           int* supernodeMask, gridinfo_t *grid, SuperLUStat_t * stat)
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int kr,kc,nlb,nub;
    int nsupers = Glu_persist->supno[n - 1] + 1;
    int_t *rowcounts, *colcounts, **rowlists, **collists, *tmpglo;
    int_t  *lsub, *lloc;
    int_t idx_i, lptr1_tmp, ib, jb, jj;
    int   *displs, *recvcounts, count, nbg;

    kr = CEILING( nsupers, grid->nprow);/* Number of local block rows */
    kc = CEILING( nsupers, grid->npcol);/* Number of local block columns */
    int_t iam=grid->iam;
    int nprocs = grid->nprow * grid->npcol;
    int_t myrow = MYROW( iam, grid );
    int_t mycol = MYCOL( iam, grid );
    int_t *ActiveFlag;
    int *ranks;
    superlu_scope_t *rscp = &grid->rscp;
    superlu_scope_t *cscp = &grid->cscp;
    int rank_cnt,rank_cnt_ref,Root;
    int_t Iactive,gb,pr,pc,nb, idx_n;

	C_Tree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
	C_Tree  *LRtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
	C_Tree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
	C_Tree  *URtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
	int msgsize;

    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t *usub;
    double *lnzval;


    int_t len, len1, len2, len3, nrbl;


	double **Lnzval_bc_ptr=Llu->Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc) */
	double *Lnzval_bc_dat;  /* size sum of sizes of Lnzval_bc_ptr[lk])                 */
    long int *Lnzval_bc_offset;  /* size ceil(NSUPERS/Pc)                 */

	int_t  **Lrowind_bc_ptr=Llu->Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc) */
	int_t *Lrowind_bc_dat;  /* size sum of sizes of Lrowind_bc_ptr[lk])                 */
    long int *Lrowind_bc_offset;  /* size ceil(NSUPERS/Pc)                 */

	int_t  **Lindval_loc_bc_ptr=Llu->Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
	int_t *Lindval_loc_bc_dat;  /* size sum of sizes of Lindval_loc_bc_ptr[lk])                 */
    long int *Lindval_loc_bc_offset;  /* size ceil(NSUPERS/Pc)                 */

    double **Linv_bc_ptr=Llu->Linv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
	double *Linv_bc_dat;  /* size sum of sizes of Linv_bc_ptr[lk])                 */
    long int *Linv_bc_offset;  /* size ceil(NSUPERS/Pc)                 */
    double **Uinv_bc_ptr=Llu->Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
	double *Uinv_bc_dat;  /* size sum of sizes of Uinv_bc_ptr[lk])                 */
    long int *Uinv_bc_offset;  /* size ceil(NSUPERS/Pc) */


	double **Unzval_br_ptr=Llu->Unzval_br_ptr;  /* size ceil(NSUPERS/Pr) */
	double *Unzval_br_dat;  /* size sum of sizes of Unzval_br_ptr[lk])                 */
	long int *Unzval_br_offset;  /* size ceil(NSUPERS/Pr)    */
	int_t  **Ufstnz_br_ptr=Llu->Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr) */
    int_t   *Ufstnz_br_dat;  /* size sum of sizes of Ufstnz_br_ptr[lk])                 */
    long int *Ufstnz_br_offset;  /* size ceil(NSUPERS/Pr)    */

    Ucb_indptr_t *Ucb_inddat;
    long int *Ucb_indoffset;
    int_t  **Ucb_valptr = Llu->Ucb_valptr;
    int_t  *Ucb_valdat;
    long int *Ucb_valoffset;
    int *h_recv_cnt;
    int *h_recv_cnt_u;

    /* Reconstruct the global L structure and compute the communication metadata */

    if ( !(tmpglo = intCalloc_dist(nsupers)) )
		ABORT("Calloc fails for tmpglo[].");
    if (!(recvcounts = (int *) SUPERLU_MALLOC (SUPERLU_MAX (grid->npcol, grid->nprow) * sizeof(int))))
        ABORT ("SUPERLU_MALLOC fails for recvcounts.");
    if (!(displs = (int *) SUPERLU_MALLOC (SUPERLU_MAX (grid->npcol, grid->nprow) * sizeof(int))))
        ABORT ("SUPERLU_MALLOC fails for displs.");


    /* gather information about the global L structure */

	if ( !(rowcounts = intCalloc_dist(kc)) )
		ABORT("Calloc fails for rowcounts[].");
	if ( !(colcounts = intCalloc_dist(kr)) )
		ABORT("Calloc fails for colcounts[].");

	if ( !(rowlists = (int_t**)SUPERLU_MALLOC(kc * sizeof(int_t*))) )
		fprintf(stderr, "Malloc fails for rowlists[].");
	if ( !(collists = (int_t**)SUPERLU_MALLOC(kr * sizeof(int_t*))) )
		fprintf(stderr, "Malloc fails for collists[].");

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                lsub = Llu->Lrowind_bc_ptr[lk];
                lloc = Llu->Lindval_loc_bc_ptr[lk];
                if(lsub){
                    nlb = lsub[0];
                    idx_i = nlb;
                    for (int_t lb = 0; lb < nlb; ++lb){
                        lptr1_tmp = lloc[lb+idx_i];
                        ib = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                        if(supernodeMask[ib]>0){
                            rowcounts[lk]++;
                            int_t lib = LBi( ib, grid ); /* Local block number, row-wise. */
                            colcounts[lib]++;
                        }
                    }
                }
            }
		}
	}

    for (int_t j=0; j<kc; j++){
        if(rowcounts[j]>0){
            if ( !(rowlists[j] = intCalloc_dist(rowcounts[j])) )
                ABORT("Calloc fails for rowlists[j].");
        }else{
            rowlists[j] = NULL;
        }
        rowcounts[j]=0;
    }
    for (int_t i=0; i<kr; i++){
        if(colcounts[i]>0){
            if ( !(collists[i] = intCalloc_dist(colcounts[i])) )
                ABORT("Calloc fails for collists[i].");
        }else{
            collists[i] = NULL;
        }
        colcounts[i]=0;
    }

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                lsub = Llu->Lrowind_bc_ptr[lk];
                lloc = Llu->Lindval_loc_bc_ptr[lk];
                if(lsub){
                    nlb = lsub[0];
                    idx_i = nlb;
                    for (int_t lb = 0; lb < nlb; ++lb){
                        lptr1_tmp = lloc[lb+idx_i];
                        ib = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                        if(supernodeMask[ib]>0){
                            rowlists[lk][rowcounts[lk]++]=ib;
                            int_t lib = LBi( ib, grid ); /* Local block number, row-wise. */
                            collists[lib][colcounts[lib]++]=jb;
                        }
                    }
                }
            }
		}
	}

    /* broadcast tree for L*/

	if ( !(ActiveFlag = intCalloc_dist(grid->nprow*2)) )
		ABORT("Calloc fails for ActiveFlag[].");
	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->nprow * sizeof(int))) )
		ABORT("Malloc fails for ranks[].");
	if ( !(LBtree_ptr = (C_Tree*)SUPERLU_MALLOC(kc * sizeof(C_Tree))) )
		ABORT("Malloc fails for LBtree_ptr[].");
	for (int_t lk = 0; lk <kc ; ++lk) {
		C_BcTree_Nullify(&LBtree_ptr[lk]);
	}

#ifdef GPU_ACC
#ifdef HAVE_NVSHMEM
    if ( !(mystatus = (int*)SUPERLU_MALLOC(kc * sizeof(int))) )
        ABORT("Malloc fails for mystatus[].");
    if ( !(h_nfrecv = (int*)SUPERLU_MALLOC( 3* sizeof(int))) )
        ABORT("Malloc fails for h_nfrecv[].");
    if ( !(h_nfrecvmod = (int*)SUPERLU_MALLOC( 4 * sizeof(int))) )
        ABORT("Malloc fails for h_nfrecvmod[].");
    h_nfrecvmod[3]=0;
    //printf("(%d)k=%d\n",iam,k);
	for (int i=0;i<kc;i++){
        mystatus[i]=1;
	}
#endif
#endif

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                // printf("iam %5d jb %5d \n",iam, jb);
                // fflush(stdout);
                pc = PCOL( jb, grid );
                count = rowcounts[lk];
                MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, cscp->comm);
                displs[0] = 0;
                nbg=0;
                for(int i=0; i<grid->nprow; ++i)
                {
                    nbg +=recvcounts[i];
                }
                if(nbg>0){
                    for(int i=0; i<grid->nprow-1; ++i)
                    {
                        displs[i+1] = displs[i] + recvcounts[i];
                    }
                    MPI_Allgatherv(rowlists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, cscp->comm);
                }
                for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j]=3*nsupers;
                for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j+grid->nprow]=j;
                for (int_t j=0;j<grid->nprow;++j)ranks[j]=-1;

                for (int_t i = 0; i < nbg; ++i) {
                    gb = tmpglo[i];
                    pr = PROW( gb, grid );
                    ActiveFlag[pr]=SUPERLU_MIN(ActiveFlag[pr],gb);
                } /* for i ... */

                Root=-1;
                Iactive = 0;
                for (int_t j=0;j<grid->nprow;++j){
                    if(ActiveFlag[j]!=3*nsupers){
                    gb = ActiveFlag[j];
                    pr = PROW( gb, grid );
                    if(gb==jb)Root=pr;
                    if(myrow==pr)Iactive=1;
                    }
                }

                quickSortM(ActiveFlag,0,grid->nprow-1,grid->nprow,0,2);

                if(Iactive==1){
                    // printf("iam %5d jb %5d Root %5d \n",iam, jb,Root);
                    // fflush(stdout);
                    assert( Root>-1 );
                    rank_cnt = 1;
                    ranks[0]=Root;
                    for (int_t j = 0; j < grid->nprow; ++j){
                        if(ActiveFlag[j]!=3*nsupers && ActiveFlag[j+grid->nprow]!=Root){
                            ranks[rank_cnt]=ActiveFlag[j+grid->nprow];
                            ++rank_cnt;
                        }
                    }

                    if(rank_cnt>1){

                        for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                            ranks[ii] = PNUM( ranks[ii], pc, grid );

                        msgsize = SuperSize( jb );
                        int needrecv=0;
                        C_BcTree_Create_nv(&LBtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd',&needrecv);
                        //C_BcTree_Create(&LBtree_ptr[ljb], grid->comm, ranks, rank_cnt, msgsize, 'd');
                        //printf("(%d) HOST create:ljb=%d,msg=%d,needrecv=%d\n",iam,ljb,mysendmsg_num,needrecv);
                        #ifdef GPU_ACC
                        #ifdef HAVE_NVSHMEM
                        if (needrecv==1) {
                            mystatus[lk]=0;
                            //printf("(%d) Col %d need one msg %d\n",iam, ljb,mystatus[ljb]);
                            //fflush(stdout);
                        }
                        #endif
                        #endif
                        LBtree_ptr[lk].tag_=BC_L;

                        // printf("iam %5d btree rank_cnt %5d \n",iam,rank_cnt);
                        // fflush(stdout);

                    }
                }
            }
        }
    }
	SUPERLU_FREE(ActiveFlag);
	SUPERLU_FREE(ranks);


    /* reduction tree for L*/
	if ( !(LRtree_ptr = (C_Tree*)SUPERLU_MALLOC(kr * sizeof(C_Tree))) )
		ABORT("Malloc fails for LRtree_ptr[].");
	if ( !(ActiveFlag = intCalloc_dist(grid->npcol*2)) )
		ABORT("Calloc fails for ActiveFlag[].");
	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->npcol * sizeof(int))) )
		ABORT("Malloc fails for ranks[].");
	for (int_t lk = 0; lk <kr ; ++lk) {
		C_RdTree_Nullify(&LRtree_ptr[lk]);
	}
#ifdef GPU_ACC
#ifdef HAVE_NVSHMEM
    if ( !(mystatusmod = (int*)SUPERLU_MALLOC(2*kr * sizeof(int))) )
        ABORT("Malloc fails for mystatusmod[].");
	if ( !(h_recv_cnt = (int*)SUPERLU_MALLOC(kr * sizeof(int))) )
        ABORT("Malloc fails for mystatusmod[].");

	int nfrecvmod=0;
	for (int i=0;i<kr;i++){
        h_recv_cnt[i]=0;
	}
	for (int i=0;i<2*kr;i++) mystatusmod[i]=1;
#endif
#endif

	for (int_t lk=0;lk<kr;++lk){
		ib = myrow+lk*grid->nprow;  /* not sure */
		if(ib<nsupers){
            if(supernodeMask[ib]>0){
                pr = PROW( ib, grid );

                count = colcounts[lk];
                MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, rscp->comm);
                displs[0] = 0;
                nbg=0;
                for(int i=0; i<grid->npcol; ++i)
                {
                    nbg +=recvcounts[i];
                }
                if(nbg>0){
                    for(int i=0; i<grid->npcol-1; ++i)
                    {
                        displs[i+1] = displs[i] + recvcounts[i];
                    }
                    MPI_Allgatherv(collists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, rscp->comm);
                }
                for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j]=-3*nsupers;
                for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j+grid->npcol]=j;
                for (int_t j=0;j<grid->npcol;++j)ranks[j]=-1;

                for (int_t j = 0; j < nbg; ++j) {
                    gb = tmpglo[j];
                    pc = PCOL( gb, grid );
                    ActiveFlag[pc]=SUPERLU_MAX(ActiveFlag[pc],gb);
                } /* for j ... */

                Root=-1;
                Iactive = 0;

                for (int_t j=0;j<grid->npcol;++j){
                    if(ActiveFlag[j]!=-3*nsupers){
                    jb = ActiveFlag[j];
                    pc = PCOL( jb, grid );
                    if(jb==ib)Root=pc;
                    if(mycol==pc)Iactive=1;
                    }
                }

                quickSortM(ActiveFlag,0,grid->npcol-1,grid->npcol,1,2);

                if(Iactive==1){
                    assert( Root>-1 );
                    rank_cnt = 1;
                    ranks[0]=Root;
                    for (int_t j = 0; j < grid->npcol; ++j){
                        if(ActiveFlag[j]!=-3*nsupers && ActiveFlag[j+grid->npcol]!=Root){
                            ranks[rank_cnt]=ActiveFlag[j+grid->npcol];
                            ++rank_cnt;
                        }
                    }
                    if(rank_cnt>1){
                        for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                            ranks[ii] = PNUM( pr, ranks[ii], grid );
                        msgsize = SuperSize( ib );
                        // C_RdTree_Create(&LRtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd');

                        int needrecvrd=0;
                        int needsendrd=0;
                        C_RdTree_Create_nv(&LRtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd', &needrecvrd, &needsendrd);
                        //C_RdTree_Create(&LRtree_ptr[lib], grid->comm, ranks, rank_cnt, msgsize, 'd');
                        #ifdef GPU_ACC
                        #ifdef HAVE_NVSHMEM
                        h_nfrecvmod[3]+=needsendrd;
                        if (needrecvrd!=0) {
                            mystatusmod[lk*2]=0;
                            mystatusmod[lk*2+1]=0;
                            h_recv_cnt[lk]=needrecvrd;
                            //printf("(%d) on CPU, lib=%d, cnt=%d\n",iam,lib,LRtree_ptr[lib].destCnt_);
                            nfrecvmod+=needrecvrd;
                        }
                        #endif
                        #endif
                        LRtree_ptr[lk].tag_=RD_L;
                    }
                }
            }
        }
    }
	SUPERLU_FREE(ActiveFlag);
	SUPERLU_FREE(ranks);




    for (int_t j=0; j<kc; j++){
        if(rowlists[j]){
            SUPERLU_FREE(rowlists[j]);
        }
    }
    for (int_t i=0; i<kr; i++){
        if(collists[i]){
            SUPERLU_FREE(collists[i]);
        }
    }
    SUPERLU_FREE(rowcounts);
    SUPERLU_FREE(colcounts);
    SUPERLU_FREE(rowlists);
    SUPERLU_FREE(collists);


    /* gather information about the global U structure */

	if ( !(rowcounts = intCalloc_dist(kc)) )
		ABORT("Calloc fails for rowcounts[].");
	if ( !(colcounts = intCalloc_dist(kr)) )
		ABORT("Calloc fails for colcounts[].");

	if ( !(rowlists = (int_t**)SUPERLU_MALLOC(kc * sizeof(int_t*))) )
		fprintf(stderr, "Malloc fails for rowlists[].");
	if ( !(collists = (int_t**)SUPERLU_MALLOC(kr * sizeof(int_t*))) )
		fprintf(stderr, "Malloc fails for collists[].");

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                nub = Urbs[lk];      /* Number of U blocks in block column lk */
                for (int_t ub = 0; ub < nub; ++ub){
                    int_t lib = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
                    ib = lib * grid->nprow + myrow;/* Global block number, row-wise. */
                    if(supernodeMask[ib]>0){
                        rowcounts[lk]++;
                        colcounts[lib]++;
                    }
                }
            }
		}
	}

    for (int_t j=0; j<kc; j++){
        if(rowcounts[j]>0){
            if ( !(rowlists[j] = intCalloc_dist(rowcounts[j])) )
                ABORT("Calloc fails for rowlists[j].");
        }else{
            rowlists[j] = NULL;
        }
        rowcounts[j]=0;
    }
    for (int_t i=0; i<kr; i++){
        if(colcounts[i]>0){
            if ( !(collists[i] = intCalloc_dist(colcounts[i])) )
                ABORT("Calloc fails for collists[i].");
        }else{
            collists[i] = NULL;
        }
        colcounts[i]=0;
    }

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                nub = Urbs[lk];      /* Number of U blocks in block column lk */
                for (int_t ub = 0; ub < nub; ++ub){
                    int_t lib = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
                    ib = lib * grid->nprow + myrow;/* Global block number, row-wise. */
                    if(supernodeMask[ib]>0){
                        rowlists[lk][rowcounts[lk]++]=ib;
                        collists[lib][colcounts[lib]++]=jb;
                    }
                }
            }
		}
	}



    /* broadcast tree for U*/
	if ( !(UBtree_ptr = (C_Tree*)SUPERLU_MALLOC(kc * sizeof(C_Tree))) )
		ABORT("Malloc fails for UBtree_ptr[].");
	if ( !(ActiveFlag = intCalloc_dist(grid->nprow*2)) )
		ABORT("Calloc fails for ActiveFlag[].");
	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->nprow * sizeof(int))) )
		ABORT("Malloc fails for ranks[].");
	for (int_t lk = 0; lk <kc ; ++lk) {
		C_BcTree_Nullify(&UBtree_ptr[lk]);
	}
#ifdef GPU_ACC
#ifdef HAVE_NVSHMEM
	if ( !(mystatus_u = (int*)SUPERLU_MALLOC(kc * sizeof(int))) )
        ABORT("Malloc fails for mystatus_u[].");
    if ( !(h_nfrecv_u = (int*)SUPERLU_MALLOC( 3* sizeof(int))) )
        ABORT("Malloc fails for h_nfrecv_u[].");
    if ( !(h_nfrecvmod_u = (int*)SUPERLU_MALLOC( 4* sizeof(int))) )
        ABORT("Malloc fails for h_nfrecvmod_u[].");
    h_nfrecvmod_u[3]=0;

	for (int i=0;i<kc;i++){
		mystatus_u[i]=1;
	}
#endif
#endif


    /* update bsendx_plist with the supernode mask. Note that fsendx_plist doesn't require updates */
    for (int_t lk=0;lk<kc;++lk){
        jb = mycol+lk*grid->npcol;  /* not sure */
        if(jb<nsupers){
        int_t krow = PROW(jb, grid);
        int_t kcol = PCOL(jb, grid);
        if (myrow == krow && mycol == kcol){
        for (int_t pr=0;pr<grid->nprow;++pr){
            Llu->bsendx_plist[lk][pr]=  SLU_EMPTY;
        }
        }
        }
    }

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                pc = PCOL( jb, grid );
                count = rowcounts[lk];
                MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, cscp->comm);
                displs[0] = 0;
                nbg=0;
                for(int i=0; i<grid->nprow; ++i)
                {
                    nbg +=recvcounts[i];
                }
                if(nbg>0){
                    for(int i=0; i<grid->nprow-1; ++i)
                    {
                        displs[i+1] = displs[i] + recvcounts[i];
                    }
                    MPI_Allgatherv(rowlists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, cscp->comm);
                }

                for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j]=-3*nsupers;
                for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j+grid->nprow]=j;
                for (int_t j=0;j<grid->nprow;++j)ranks[j]=-1;

                for (int_t i = 0; i < nbg; ++i) {
                    gb = tmpglo[i];
                    pr = PROW( gb, grid );
                    ActiveFlag[pr]=SUPERLU_MAX(ActiveFlag[pr],gb);
                } /* for i ... */

                pr = PROW( jb, grid ); // take care of diagonal node stored as L
                ActiveFlag[pr]=SUPERLU_MAX(ActiveFlag[pr],jb);

                Root=-1;
                Iactive = 0;
                for (int_t j=0;j<grid->nprow;++j){
                    if(ActiveFlag[j]!=-3*nsupers){
                    gb = ActiveFlag[j];
                    pr = PROW( gb, grid );
                    if(gb==jb)Root=pr;
                    if(myrow==pr)Iactive=1;
                    if(myrow!=pr && myrow == PROW(jb, grid)) /* update bsendx_plist with the supernode mask */
                        Llu->bsendx_plist[lk][pr]=YES;
                    }
                }

                quickSortM(ActiveFlag,0,grid->nprow-1,grid->nprow,1,2);

                if(Iactive==1){
                    assert( Root>-1 );
                    rank_cnt = 1;
                    ranks[0]=Root;
                    for (int_t j = 0; j < grid->nprow; ++j){
                        if(ActiveFlag[j]!=-3*nsupers && ActiveFlag[j+grid->nprow]!=Root){
                            ranks[rank_cnt]=ActiveFlag[j+grid->nprow];
                            ++rank_cnt;
                        }
                    }
                    if(rank_cnt>1){
                        for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                            ranks[ii] = PNUM( ranks[ii], pc, grid );
                        msgsize = SuperSize( jb );
                        // C_BcTree_Create(&UBtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd');

                        int needrecv=0;
                        C_BcTree_Create_nv(&UBtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd',&needrecv);
                        //C_BcTree_Create(&UBtree_ptr[ljb], grid->comm, ranks, rank_cnt, msgsize, 'd');
                        #ifdef GPU_ACC
                        #ifdef HAVE_NVSHMEM
                        if (needrecv==1) {
                            mystatus_u[lk]=0;
                            //printf("(%d) Col %d need one msg %d\n",iam, ljb,mystatus[ljb]);
                        }
                        #endif
                        #endif
                        UBtree_ptr[lk].tag_=BC_U;
                    }
                }
            }
        }
    }
	SUPERLU_FREE(ActiveFlag);
	SUPERLU_FREE(ranks);


    /* reduction tree for U*/
	if ( !(URtree_ptr = (C_Tree*)SUPERLU_MALLOC(kr * sizeof(C_Tree))) )
		ABORT("Malloc fails for URtree_ptr[].");
	if ( !(ActiveFlag = intCalloc_dist(grid->npcol*2)) )
		ABORT("Calloc fails for ActiveFlag[].");
	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->npcol * sizeof(int))) )
		ABORT("Malloc fails for ranks[].");
	for (int_t lk = 0; lk <kr ; ++lk) {
		C_RdTree_Nullify(&URtree_ptr[lk]);
	}

    #ifdef GPU_ACC
    #ifdef HAVE_NVSHMEM
	if ( !(mystatusmod_u = (int*)SUPERLU_MALLOC(2*kr * sizeof(int))) )
        ABORT("Malloc fails for mystatusmod_u[].");
    if ( !(h_recv_cnt_u = (int*)SUPERLU_MALLOC(kr * sizeof(int))) )
        ABORT("Malloc fails for h_recv_cnt_u[].");

    int nbrecvmod=0;
	for (int i=0;i<kr;i++){
        h_recv_cnt_u[i]=0;
	}
	for (int i=0;i<2*kr;i++) mystatusmod_u[i]=1;
    #endif
    #endif

	for (int_t lk=0;lk<kr;++lk){
		ib = myrow+lk*grid->nprow;  /* not sure */
		if(ib<nsupers){
            if(supernodeMask[ib]>0){
                pr = PROW( ib, grid );

                count = colcounts[lk];
                MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, rscp->comm);
                displs[0] = 0;
                nbg=0;
                for(int i=0; i<grid->npcol; ++i)
                {
                    nbg +=recvcounts[i];
                }
                if(nbg>0){
                    for(int i=0; i<grid->npcol-1; ++i)
                    {
                        displs[i+1] = displs[i] + recvcounts[i];
                    }
                    MPI_Allgatherv(collists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, rscp->comm);
                }
                for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j]=3*nsupers;
                for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j+grid->npcol]=j;
                for (int_t j=0;j<grid->npcol;++j)ranks[j]=-1;

                for (int_t j = 0; j < nbg; ++j) {
                    gb = tmpglo[j];
                    pc = PCOL( gb, grid );
                    ActiveFlag[pc]=SUPERLU_MIN(ActiveFlag[pc],gb);
                } /* for j ... */
                pc = PCOL( ib, grid ); // take care of diagonal node stored as L
                ActiveFlag[pc]=SUPERLU_MIN(ActiveFlag[pc],ib);

                Root=-1;
                Iactive = 0;

                for (int_t j=0;j<grid->npcol;++j){
                    if(ActiveFlag[j]!=3*nsupers){
                    jb = ActiveFlag[j];
                    pc = PCOL( jb, grid );
                    if(jb==ib)Root=pc;
                    if(mycol==pc)Iactive=1;
                    }
                }

                quickSortM(ActiveFlag,0,grid->npcol-1,grid->npcol,0,2);

                if(Iactive==1){
                    assert( Root>-1 );
                    rank_cnt = 1;
                    ranks[0]=Root;
                    for (int_t j = 0; j < grid->npcol; ++j){
                        if(ActiveFlag[j]!=3*nsupers && ActiveFlag[j+grid->npcol]!=Root){
                            ranks[rank_cnt]=ActiveFlag[j+grid->npcol];
                            ++rank_cnt;
                        }
                    }
                    if(rank_cnt>1){
                        for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                            ranks[ii] = PNUM( pr, ranks[ii], grid );
                        msgsize = SuperSize( ib );
                        // C_RdTree_Create(&URtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd');

                        int needrecvrd=0;
                        int needsendrd=0;
                        C_RdTree_Create_nv(&URtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd', &needrecvrd,&needsendrd);
                        //C_RdTree_Create(&URtree_ptr[lib], grid->comm, ranks, rank_cnt, msgsize, 'd');
                        #ifdef GPU_ACC
                        #ifdef HAVE_NVSHMEM
                        h_nfrecvmod_u[3] +=needsendrd;
                        if (needrecvrd!=0) {
                            mystatusmod_u[lk*2]=0;
                            mystatusmod_u[lk*2+1]=0;
                            h_recv_cnt_u[lk]=needrecvrd;
                            //printf("(%d) on CPU, lib=%d, cnt=%d\n",iam,lib,LRtree_ptr[lib].destCnt_);
                            nbrecvmod+=needrecvrd;
                        }
                        #endif
                        #endif
                        URtree_ptr[lk].tag_=RD_U;
                    }
                }
            }
        }
    }
	SUPERLU_FREE(ActiveFlag);
	SUPERLU_FREE(ranks);

    for (int_t j=0; j<kc; j++){
        if(rowlists[j]){
            SUPERLU_FREE(rowlists[j]);
        }
    }
    for (int_t i=0; i<kr; i++){
        if(collists[i]){
            SUPERLU_FREE(collists[i]);
        }
    }
    SUPERLU_FREE(rowcounts);
    SUPERLU_FREE(colcounts);
    SUPERLU_FREE(rowlists);
    SUPERLU_FREE(collists);

    SUPERLU_FREE(tmpglo);
    SUPERLU_FREE(recvcounts);
    SUPERLU_FREE(displs);



    // ////////////////////////////////////////////////////
    // // use contignous memory for the L meta data
    // int_t k = kc;/* Number of local block columns */
    // long int Lnzval_bc_cnt=0;
    // long int Lrowind_bc_cnt=0;
    // long int Lindval_loc_bc_cnt=0;
	// long int Linv_bc_cnt=0;
	// long int Uinv_bc_cnt=0;

	// if ( !(Lnzval_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Lnzval_bc_offset[].");
	// }
	// Lnzval_bc_offset[k-1] = -1;

	// if ( !(Lrowind_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Lrowind_bc_offset[].");
	// }
	// Lrowind_bc_offset[k-1] = -1;
	// if ( !(Lindval_loc_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Lindval_loc_bc_offset[].");
	// }
	// Lindval_loc_bc_offset[k-1] = -1;
	// if ( !(Linv_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Linv_bc_offset[].");
	// }
	// Linv_bc_offset[k-1] = -1;
	// if ( !(Uinv_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Uinv_bc_offset[].");
	// }
	// Uinv_bc_offset[k-1] = -1;


    // for (int_t lk=0;lk<k;++lk){
    //     jb = mycol+lk*grid->npcol;  /* not sure */
	//     lsub = Lrowind_bc_ptr[lk];
	//     lloc = Lindval_loc_bc_ptr[lk];
	//     lnzval = Lnzval_bc_ptr[lk];

    //     Linv_bc_offset[lk] = -1;
    //     Uinv_bc_offset[lk] = -1;
    //     Lrowind_bc_offset[lk]=-1;
    //     Lindval_loc_bc_offset[lk]=-1;
    //     Lnzval_bc_offset[lk]=-1;

    //     if(lsub){
    //         nrbl  =   lsub[0]; /*number of L blocks */
    //         len   = lsub[1];   /* LDA of the nzval[] */
    //         len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
    //         int_t nsupc = SuperSize(jb);
    //         len2  = nsupc * len;
    //         len3 = nrbl*3;
    //         Lnzval_bc_offset[lk]=len2;
    //         Lnzval_bc_cnt += Lnzval_bc_offset[lk];

    //         Lrowind_bc_offset[lk]=len1;
    //         Lrowind_bc_cnt += Lrowind_bc_offset[lk];

	// 		Lindval_loc_bc_offset[lk]=nrbl*3;
	// 		Lindval_loc_bc_cnt += Lindval_loc_bc_offset[lk];

    //         int_t krow = PROW( jb, grid );
	// 		if(myrow==krow){   /* diagonal block */
	// 			Linv_bc_offset[lk]=nsupc*nsupc;
	// 			Linv_bc_cnt += Linv_bc_offset[lk];
	// 			Uinv_bc_offset[lk]=nsupc*nsupc;
	// 			Uinv_bc_cnt += Uinv_bc_offset[lk];
	// 		}else{
	// 			Linv_bc_offset[lk] = -1;
	// 			Uinv_bc_offset[lk] = -1;
	// 		}

    //     }
    // }

	// Linv_bc_cnt +=1; // safe guard
	// Uinv_bc_cnt +=1;
	// Lrowind_bc_cnt +=1;
	// Lindval_loc_bc_cnt +=1;
	// Lnzval_bc_cnt +=1;
	// if ( !(Linv_bc_dat =
	// 			(double*)SUPERLU_MALLOC(Linv_bc_cnt * sizeof(double))) ) {
	// 	fprintf(stderr, "Malloc fails for Linv_bc_dat[].");
	// }
	// if ( !(Uinv_bc_dat =
	// 			(double*)SUPERLU_MALLOC(Uinv_bc_cnt * sizeof(double))) ) {
	// 	fprintf(stderr, "Malloc fails for Uinv_bc_dat[].");
	// }

	// if ( !(Lrowind_bc_dat =
	// 			(int_t*)SUPERLU_MALLOC(Lrowind_bc_cnt * sizeof(int_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Lrowind_bc_dat[].");
	// }
	// if ( !(Lindval_loc_bc_dat =
	// 			(int_t*)SUPERLU_MALLOC(Lindval_loc_bc_cnt * sizeof(int_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Lindval_loc_bc_dat[].");
	// }
	// if ( !(Lnzval_bc_dat =
	// 			(double*)SUPERLU_MALLOC(Lnzval_bc_cnt * sizeof(double))) ) {
	// 	fprintf(stderr, "Malloc fails for Lnzval_bc_dat[].");
	// }


	// /* use contingous memory for Linv_bc_ptr, Uinv_bc_ptr, Lrowind_bc_ptr, Lnzval_bc_ptr*/
	// Linv_bc_cnt=0;
	// Uinv_bc_cnt=0;
	// Lrowind_bc_cnt=0;
	// Lnzval_bc_cnt=0;
	// Lindval_loc_bc_cnt=0;
	// long int tmp_cnt;
	// for (jb = 0; jb < k; ++jb) { /* for each block column ... */
	// 	if(Linv_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Linv_bc_offset[jb]; ++jj) {
	// 			Linv_bc_dat[Linv_bc_cnt+jj]=Linv_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Linv_bc_ptr[jb]);
	// 		Linv_bc_ptr[jb]=&Linv_bc_dat[Linv_bc_cnt];
	// 		tmp_cnt = Linv_bc_offset[jb];
	// 		Linv_bc_offset[jb]=Linv_bc_cnt;
	// 		Linv_bc_cnt+=tmp_cnt;
	// 	}

	// 	if(Uinv_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Uinv_bc_offset[jb]; ++jj) {
	// 			Uinv_bc_dat[Uinv_bc_cnt+jj]=Uinv_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Uinv_bc_ptr[jb]);
	// 		Uinv_bc_ptr[jb]=&Uinv_bc_dat[Uinv_bc_cnt];
	// 		tmp_cnt = Uinv_bc_offset[jb];
	// 		Uinv_bc_offset[jb]=Uinv_bc_cnt;
	// 		Uinv_bc_cnt+=tmp_cnt;
	// 	}


	// 	if(Lrowind_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Lrowind_bc_offset[jb]; ++jj) {
	// 			Lrowind_bc_dat[Lrowind_bc_cnt+jj]=Lrowind_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Lrowind_bc_ptr[jb]);
	// 		Lrowind_bc_ptr[jb]=&Lrowind_bc_dat[Lrowind_bc_cnt];
	// 		tmp_cnt = Lrowind_bc_offset[jb];
	// 		Lrowind_bc_offset[jb]=Lrowind_bc_cnt;
	// 		Lrowind_bc_cnt+=tmp_cnt;
	// 	}

	// 	if(Lnzval_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Lnzval_bc_offset[jb]; ++jj) {
	// 			Lnzval_bc_dat[Lnzval_bc_cnt+jj]=Lnzval_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Lnzval_bc_ptr[jb]);
	// 		Lnzval_bc_ptr[jb]=&Lnzval_bc_dat[Lnzval_bc_cnt];
	// 		tmp_cnt = Lnzval_bc_offset[jb];
	// 		Lnzval_bc_offset[jb]=Lnzval_bc_cnt;
	// 		Lnzval_bc_cnt+=tmp_cnt;
	// 	}

	// 	if(Lindval_loc_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Lindval_loc_bc_offset[jb]; ++jj) {
	// 			Lindval_loc_bc_dat[Lindval_loc_bc_cnt+jj]=Lindval_loc_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Lindval_loc_bc_ptr[jb]);
	// 		Lindval_loc_bc_ptr[jb]=&Lindval_loc_bc_dat[Lindval_loc_bc_cnt];
	// 		tmp_cnt = Lindval_loc_bc_offset[jb];
	// 		Lindval_loc_bc_offset[jb]=Lindval_loc_bc_cnt;
	// 		Lindval_loc_bc_cnt+=tmp_cnt;
	// 	}
	// }



    // // use contignous memory for the U meta data
    // k = kr;/* Number of local block rows */
    // long int Unzval_br_cnt=0;
    // long int Ufstnz_br_cnt=0;
    // long int Ucb_indcnt=0;
    // long int Ucb_valcnt=0;

	// if ( !(Unzval_br_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Unzval_br_offset[].");
	// }
	// Unzval_br_offset[k-1] = -1;
	// if ( !(Ufstnz_br_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Ufstnz_br_offset[].");
	// }
	// Ufstnz_br_offset[k-1] = -1;

    // int_t Pc = grid->npcol;
    // nub = CEILING (nsupers, Pc);
	// if ( !(Ucb_valoffset =
	// 			(long int*)SUPERLU_MALLOC(nub * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Ucb_valoffset[].");
	// }
	// Ucb_valoffset[nub-1] = -1;
	// if ( !(Ucb_indoffset =
	// 			(long int*)SUPERLU_MALLOC(nub * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Ucb_indoffset[].");
	// }
	// Ucb_indoffset[nub-1] = -1;

    // for (int_t lk=0;lk<k;++lk){
    //     ib = myrow+lk*grid->nprow;  /* not sure */
	//     usub =  Ufstnz_br_ptr[lk];
    //     Unzval_br_offset[lk]=-1;
    //     Ufstnz_br_offset[lk]=-1;
    //     if(usub){
    //         int_t lenv = usub[1];
	// 	    int_t lens = usub[2];
    //         Unzval_br_offset[lk]=lenv;
	// 	    Unzval_br_cnt += Unzval_br_offset[lk];
    //         Ufstnz_br_offset[lk]=lens;
    //         Ufstnz_br_cnt += Ufstnz_br_offset[lk];
    //     }
    // }

	// /* Set up the vertical linked lists for the row blocks.
	//    One pass of the skeleton graph of U. */
	// for (int_t lb = 0; lb < kc; ++lb) {
	// 	if ( Urbs[lb] ) { /* Not an empty block column. */
	// 		Ucb_indoffset[lb]=Urbs[lb];
	// 		Ucb_indcnt += Ucb_indoffset[lb];
	// 		Ucb_valoffset[lb]=Urbs[lb];
	// 		Ucb_valcnt += Ucb_valoffset[lb];
	// 	}else{
	// 		Ucb_valoffset[lb]=-1;
	// 		Ucb_indoffset[lb]=-1;
	// 	}
	// }

	// Unzval_br_cnt +=1; // safe guard
	// Ufstnz_br_cnt +=1;
	// Ucb_valcnt +=1;
	// Ucb_indcnt +=1;
	// if ( !(Unzval_br_dat =
	// 			(double*)SUPERLU_MALLOC(Unzval_br_cnt * sizeof(double))) ) {
	// 	fprintf(stderr, "Malloc fails for Lnzval_bc_dat[].");
	// }
	// if ( !(Ufstnz_br_dat =
	// 			(int_t*)SUPERLU_MALLOC(Ufstnz_br_cnt * sizeof(int_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Ufstnz_br_dat[].");
	// }
	// if ( !(Ucb_valdat =
	// 			(int_t*)SUPERLU_MALLOC(Ucb_valcnt * sizeof(int_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Ucb_valdat[].");
	// }
	// if ( !(Ucb_inddat =
	// 			(Ucb_indptr_t*)SUPERLU_MALLOC(Ucb_indcnt * sizeof(Ucb_indptr_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Ucb_inddat[].");
	// }


	// /* use contingous memory for Unzval_br_ptr, Ufstnz_br_ptr, Ucb_valptr */
	// k = CEILING( nsupers, grid->nprow );/* Number of local block rows */
	// Unzval_br_cnt=0;
	// Ufstnz_br_cnt=0;
	// for (int_t lb = 0; lb < k; ++lb) { /* for each block row ... */
	// 	if(Unzval_br_ptr[lb]!=NULL){
	// 		for (jj = 0; jj < Unzval_br_offset[lb]; ++jj) {
	// 			Unzval_br_dat[Unzval_br_cnt+jj]=Unzval_br_ptr[lb][jj];
	// 		}
	// 		SUPERLU_FREE(Unzval_br_ptr[lb]);
	// 		Unzval_br_ptr[lb]=&Unzval_br_dat[Unzval_br_cnt];
	// 		tmp_cnt = Unzval_br_offset[lb];
	// 		Unzval_br_offset[lb]=Unzval_br_cnt;
	// 		Unzval_br_cnt+=tmp_cnt;
	// 	}

	// 	if(Ufstnz_br_ptr[lb]!=NULL){
	// 		for (jj = 0; jj < Ufstnz_br_offset[lb]; ++jj) {
	// 			Ufstnz_br_dat[Ufstnz_br_cnt+jj]=Ufstnz_br_ptr[lb][jj];
	// 		}
	// 		SUPERLU_FREE(Ufstnz_br_ptr[lb]);
	// 		Ufstnz_br_ptr[lb]=&Ufstnz_br_dat[Ufstnz_br_cnt];
	// 		tmp_cnt = Ufstnz_br_offset[lb];
	// 		Ufstnz_br_offset[lb]=Ufstnz_br_cnt;
	// 		Ufstnz_br_cnt+=tmp_cnt;
	// 	}
	// }

	// k = CEILING( nsupers, grid->npcol );/* Number of local block columns */
	// Ucb_valcnt=0;
	// Ucb_indcnt=0;
	// for (int_t lb = 0; lb < k; ++lb) { /* for each block row ... */
	// 	if(Ucb_valptr[lb]!=NULL){
	// 		for (jj = 0; jj < Ucb_valoffset[lb]; ++jj) {
	// 			Ucb_valdat[Ucb_valcnt+jj]=Ucb_valptr[lb][jj];
	// 		}
	// 		SUPERLU_FREE(Ucb_valptr[lb]);
	// 		Ucb_valptr[lb]=&Ucb_valdat[Ucb_valcnt];
	// 		tmp_cnt = Ucb_valoffset[lb];
	// 		Ucb_valoffset[lb]=Ucb_valcnt;
	// 		Ucb_valcnt+=tmp_cnt;
	// 	}
	// 	if(Ucb_indptr[lb]!=NULL){
	// 		for (jj = 0; jj < Ucb_indoffset[lb]; ++jj) {
	// 			Ucb_inddat[Ucb_indcnt+jj]=Ucb_indptr[lb][jj];
	// 		}
	// 		SUPERLU_FREE(Ucb_indptr[lb]);
	// 		Ucb_indptr[lb]=&Ucb_inddat[Ucb_indcnt];
	// 		tmp_cnt = Ucb_indoffset[lb];
	// 		Ucb_indoffset[lb]=Ucb_indcnt;
	// 		Ucb_indcnt+=tmp_cnt;
	// 	}
	// }

	// Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
	// Llu->Lrowind_bc_dat = Lrowind_bc_dat;
	// Llu->Lrowind_bc_offset = Lrowind_bc_offset;
	// Llu->Lrowind_bc_cnt = Lrowind_bc_cnt;

	// Llu->Lindval_loc_bc_ptr = Lindval_loc_bc_ptr;
	// Llu->Lindval_loc_bc_dat = Lindval_loc_bc_dat;
	// Llu->Lindval_loc_bc_offset = Lindval_loc_bc_offset;
	// Llu->Lindval_loc_bc_cnt = Lindval_loc_bc_cnt;

	// Llu->Lnzval_bc_ptr = Lnzval_bc_ptr;
	// Llu->Lnzval_bc_dat = Lnzval_bc_dat;
	// Llu->Lnzval_bc_offset = Lnzval_bc_offset;
	// Llu->Lnzval_bc_cnt = Lnzval_bc_cnt;

	// Llu->Linv_bc_ptr = Linv_bc_ptr;
	// Llu->Linv_bc_dat = Linv_bc_dat;
	// Llu->Linv_bc_offset = Linv_bc_offset;
	// Llu->Linv_bc_cnt = Linv_bc_cnt;

	// Llu->Uinv_bc_ptr = Uinv_bc_ptr;
	// Llu->Uinv_bc_dat = Uinv_bc_dat;
	// Llu->Uinv_bc_offset = Uinv_bc_offset;
	// Llu->Uinv_bc_cnt = Uinv_bc_cnt;


	Llu->Ufstnz_br_ptr = Ufstnz_br_ptr;
    // Llu->Ufstnz_br_dat = Ufstnz_br_dat;
    // Llu->Ufstnz_br_offset = Ufstnz_br_offset;
    // Llu->Ufstnz_br_cnt = Ufstnz_br_cnt;

	Llu->Unzval_br_ptr = Unzval_br_ptr;
	// Llu->Unzval_br_dat = Unzval_br_dat;
	// Llu->Unzval_br_offset = Unzval_br_offset;
	// Llu->Unzval_br_cnt = Unzval_br_cnt;

	Llu->Ucb_indptr = Ucb_indptr;
	// Llu->Ucb_inddat = Ucb_inddat;
	// Llu->Ucb_indoffset = Ucb_indoffset;
	// Llu->Ucb_indcnt = Ucb_indcnt;
	Llu->Ucb_valptr = Ucb_valptr;
	// Llu->Ucb_valdat = Ucb_valdat;
	// Llu->Ucb_valoffset = Ucb_valoffset;
	// Llu->Ucb_valcnt = Ucb_valcnt;


	Llu->LRtree_ptr = LRtree_ptr;
	Llu->LBtree_ptr = LBtree_ptr;
	Llu->URtree_ptr = URtree_ptr;
	Llu->UBtree_ptr = UBtree_ptr;


    Llu->nbcol_masked=0;
	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
        if(jb<nsupers){
            if(supernodeMask[jb]==1){ // only record the columns performed on GPU
               Llu->nbcol_masked++;
            }
        }
    }
 	if ( !(Llu->bcols_masked =
				(int*)SUPERLU_MALLOC(Llu->nbcol_masked * sizeof(int))) ) {
		fprintf(stderr, "Malloc fails for nbcol_masked[].");
	}
    Llu->nbcol_masked=0;
	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
        if(jb<nsupers){
            if(supernodeMask[jb]==1){ // only record the columns performed on GPU
               Llu->bcols_masked[Llu->nbcol_masked++]=lk;
            }
        }
    }
    // printf("Llu->nbcol_masked: %10d\n",Llu->nbcol_masked);
    // fflush(stdout);


#ifdef GPU_ACC
    if (get_acc_solve()){
	checkGPU(gpuMalloc( (void**)&Llu->d_bcols_masked, Llu->nbcol_masked * sizeof(int)));
	checkGPU(gpuMemcpy(Llu->d_bcols_masked, Llu->bcols_masked, Llu->nbcol_masked * sizeof(int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_xsup, (n+1) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_xsup, xsup, (n+1) * sizeof(int_t), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_LRtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree)));
	checkGPU(gpuMalloc( (void**)&Llu->d_LBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree)));
	checkGPU(gpuMalloc( (void**)&Llu->d_URtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree)));
	checkGPU(gpuMalloc( (void**)&Llu->d_UBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree)));
	checkGPU(gpuMemcpy(Llu->d_LRtree_ptr, Llu->LRtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(Llu->d_LBtree_ptr, Llu->LBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(Llu->d_URtree_ptr, Llu->URtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(Llu->d_UBtree_ptr, Llu->UBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lrowind_bc_dat, (Llu->Lrowind_bc_cnt) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_Lrowind_bc_dat, Llu->Lrowind_bc_dat, (Llu->Lrowind_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lindval_loc_bc_dat, (Llu->Lindval_loc_bc_cnt) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_Lindval_loc_bc_dat, Llu->Lindval_loc_bc_dat, (Llu->Lindval_loc_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lrowind_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Lrowind_bc_offset, Llu->Lrowind_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lindval_loc_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Lindval_loc_bc_offset, Llu->Lindval_loc_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lnzval_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Lnzval_bc_offset, Llu->Lnzval_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
    checkGPU(gpuMalloc( (void**)&Llu->d_grid, sizeof(gridinfo_t)));
    checkGPU(gpuMemcpy(Llu->d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));

	// some dummy allocation to avoid checking whether they are null pointers later
#if 0
	checkGPU(gpuMalloc( (void**)&Llu->d_Ucolind_bc_dat, sizeof(int_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Ucolind_bc_offset, sizeof(int64_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Unzval_bc_dat, sizeof(double)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Unzval_bc_offset, sizeof(int64_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uindval_loc_bc_dat, sizeof(int_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uindval_loc_bc_offset, sizeof(int_t)));
#else
    Llu->d_Ucolind_bc_dat=NULL;
    Llu->d_Ucolind_br_dat=NULL;
    Llu->d_Ucolind_bc_offset=NULL;
    Llu->d_Ucolind_br_offset=NULL;
    Llu->d_Uind_br_dat=NULL;
    Llu->d_Uind_br_offset=NULL;
    Llu->d_Unzval_bc_dat=NULL;
    Llu->d_Unzval_bc_offset=NULL;
    Llu->d_Unzval_br_new_dat=NULL;
    Llu->d_Unzval_br_new_offset=NULL;
    Llu->d_Uindval_loc_bc_dat=NULL;
    Llu->d_Uindval_loc_bc_offset=NULL;
#endif


	checkGPU(gpuMalloc( (void**)&Llu->d_Linv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Linv_bc_offset, Llu->Linv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uinv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Uinv_bc_offset, Llu->Uinv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_ilsum, (CEILING( nsupers, grid->nprow )+1) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_ilsum, Llu->ilsum, (CEILING( nsupers, grid->nprow )+1) * sizeof(int_t), gpuMemcpyHostToDevice));


	/* gpuMemcpy for the following is performed in pxgssvx/pxgssvx3d */
	checkGPU(gpuMalloc( (void**)&Llu->d_Lnzval_bc_dat, (Llu->Lnzval_bc_cnt) * sizeof(double)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Linv_bc_dat, (Llu->Linv_bc_cnt) * sizeof(double)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uinv_bc_dat, (Llu->Uinv_bc_cnt) * sizeof(double)));


       /* nvshmem related*/
#ifdef HAVE_NVSHMEM
	checkGPU(gpuMalloc( (void**)&d_recv_cnt, CEILING(nsupers, grid->nprow) * sizeof(int)));
	checkGPU(gpuMemcpy(d_recv_cnt, h_recv_cnt,  CEILING(nsupers, grid->nprow) * sizeof(int), gpuMemcpyHostToDevice));
        checkGPU(gpuMalloc( (void**)&d_recv_cnt_u, CEILING(nsupers, grid->nprow) * sizeof(int)));
        checkGPU(gpuMemcpy(d_recv_cnt_u, h_recv_cnt_u,  CEILING(nsupers, grid->nprow) * sizeof(int), gpuMemcpyHostToDevice));
#endif
    }
#ifdef HAVE_NVSHMEM
    SUPERLU_FREE(h_recv_cnt);
    SUPERLU_FREE(h_recv_cnt_u);
#endif
#endif /* end ifdef GPU_ACC */

    // /* recompute fmod, bmod */
	// for (int_t i = 0; i < kc; ++i)
	// 	Llu->fmod[i] = 0;


	// for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
	// 	jb = mycol+lk*grid->npcol;  /* not sure */
	// 	if(jb<nsupers){
    //         if(supernodeMask[jb]>0)
    //         {
    //             int_t krow = PROW (jb, grid);
    //             int_t kcol = PCOL (jb, grid);

    //             int_t* lsub = Lrowind_bc_ptr[lk];
    //             int_t* lloc = LUstruct->Llu->Lindval_loc_bc_ptr[lk];
    //             if(lsub){
    //             if(lsub[0]>0){
    //                 if(myrow==krow){
    //                     nb = lsub[0] - 1;
    //                     idx_n = 1;
    //                     idx_i = nb+2;
    //                 }else{
    //                     nb = lsub[0];
    //                     idx_n = 0;
    //                     idx_i = nb;
    //                 }
    //                 for (int_t lb=0;lb<nb;lb++){
    //                     int_t lik = lloc[lb+idx_n]; /* Local block number, row-wise. */
    //                     lptr1_tmp = lloc[lb+idx_i];
    //                     ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
    //                     if(supernodeMask[ik])
    //                         Llu->fmod[lik] +=1;
    //                 }
    //             }
    //             }
    //         }
    //     }
    // }

    return 0;
} // end dtrs_compute_communication_structure





// int dtrs_compute_communication_structure_sym(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
//                            dScalePermstruct_t * ScalePermstruct,
//                            int* supernodeMask, gridinfo_t *grid, SuperLUStat_t * stat)
// {
//     Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
//     int kr,kc,nlb,nub;
//     int nsupers = Glu_persist->supno[n - 1] + 1;
//     int_t *rowcounts, *colcounts, **rowlists, **collists, *tmpglo;
//     int_t  *lsub, *lloc;
//     int_t idx_i, lptr1_tmp, ib, jb, jj;
//     int   *displs, *recvcounts, count, nbg;

//     kr = CEILING( nsupers, grid->nprow);/* Number of local block rows */
//     kc = CEILING( nsupers, grid->npcol);/* Number of local block columns */
//     int_t iam=grid->iam;
//     int nprocs = grid->nprow * grid->npcol;
//     int_t myrow = MYROW( iam, grid );
//     int_t mycol = MYCOL( iam, grid );
//     int_t *ActiveFlag;
//     int *ranks;
//     superlu_scope_t *rscp = &grid->rscp;
//     superlu_scope_t *cscp = &grid->cscp;
//     int rank_cnt,rank_cnt_ref,Root;
//     int_t Iactive,gb,pr,pc,nb, idx_n;

// 	C_Tree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
// 	C_Tree  *LRtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
// 	C_Tree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
// 	C_Tree  *URtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
// 	int msgsize;

//     dLocalLU_t *Llu = LUstruct->Llu;
//     int_t* xsup = Glu_persist->xsup;
//     int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
//     Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
//     int_t *usub;
//     double *lnzval;


//     int_t len, len1, len2, len3, nrbl;


// 	double **Lnzval_bc_ptr=Llu->Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc) */
// 	double *Lnzval_bc_dat;  /* size sum of sizes of Lnzval_bc_ptr[lk])                 */
//     long int *Lnzval_bc_offset;  /* size ceil(NSUPERS/Pc)                 */

// 	int_t  **Lrowind_bc_ptr=Llu->Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc) */
// 	int_t *Lrowind_bc_dat;  /* size sum of sizes of Lrowind_bc_ptr[lk])                 */
//     long int *Lrowind_bc_offset;  /* size ceil(NSUPERS/Pc)                 */

// 	int_t  **Lindval_loc_bc_ptr=Llu->Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
// 	int_t *Lindval_loc_bc_dat;  /* size sum of sizes of Lindval_loc_bc_ptr[lk])                 */
//     long int *Lindval_loc_bc_offset;  /* size ceil(NSUPERS/Pc)                 */

//     double **Linv_bc_ptr=Llu->Linv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
// 	double *Linv_bc_dat;  /* size sum of sizes of Linv_bc_ptr[lk])                 */
//     long int *Linv_bc_offset;  /* size ceil(NSUPERS/Pc)                 */
//     double **Uinv_bc_ptr=Llu->Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
// 	double *Uinv_bc_dat;  /* size sum of sizes of Uinv_bc_ptr[lk])                 */
//     long int *Uinv_bc_offset;  /* size ceil(NSUPERS/Pc) */


// 	double **Unzval_br_ptr=Llu->Unzval_br_ptr;  /* size ceil(NSUPERS/Pr) */
// 	double *Unzval_br_dat;  /* size sum of sizes of Unzval_br_ptr[lk])                 */
// 	long int *Unzval_br_offset;  /* size ceil(NSUPERS/Pr)    */
// 	int_t  **Ufstnz_br_ptr=Llu->Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr) */
//     int_t   *Ufstnz_br_dat;  /* size sum of sizes of Ufstnz_br_ptr[lk])                 */
//     long int *Ufstnz_br_offset;  /* size ceil(NSUPERS/Pr)    */

//     Ucb_indptr_t *Ucb_inddat;
//     long int *Ucb_indoffset;
//     int_t  **Ucb_valptr = Llu->Ucb_valptr;
//     int_t  *Ucb_valdat;
//     long int *Ucb_valoffset;
//     int *h_recv_cnt;
//     int *h_recv_cnt_u;

//     /* Reconstruct the global L structure and compute the communication metadata */

//     if ( !(tmpglo = intCalloc_dist(nsupers)) )
// 		ABORT("Calloc fails for tmpglo[].");
//     if (!(recvcounts = (int *) SUPERLU_MALLOC (SUPERLU_MAX (grid->npcol, grid->nprow) * sizeof(int))))
//         ABORT ("SUPERLU_MALLOC fails for recvcounts.");
//     if (!(displs = (int *) SUPERLU_MALLOC (SUPERLU_MAX (grid->npcol, grid->nprow) * sizeof(int))))
//         ABORT ("SUPERLU_MALLOC fails for displs.");


//     /* gather information about the global L structure */

// 	if ( !(rowcounts = intCalloc_dist(kc)) )
// 		ABORT("Calloc fails for rowcounts[].");
// 	if ( !(colcounts = intCalloc_dist(kr)) )
// 		ABORT("Calloc fails for colcounts[].");

// 	if ( !(rowlists = (int_t**)SUPERLU_MALLOC(kc * sizeof(int_t*))) )
// 		fprintf(stderr, "Malloc fails for rowlists[].");
// 	if ( !(collists = (int_t**)SUPERLU_MALLOC(kr * sizeof(int_t*))) )
// 		fprintf(stderr, "Malloc fails for collists[].");

// 	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
// 		jb = mycol+lk*grid->npcol;  /* not sure */
// 		if(jb<nsupers){
//             if(supernodeMask[jb]>0){
//                 lsub = Llu->Lrowind_bc_ptr[lk];
//                 lloc = Llu->Lindval_loc_bc_ptr[lk];
//                 if(lsub){
//                     nlb = lsub[0];
//                     idx_i = nlb;
//                     for (int_t lb = 0; lb < nlb; ++lb){
//                         lptr1_tmp = lloc[lb+idx_i];
//                         ib = lsub[lptr1_tmp]; /* Global block number, row-wise. */
//                         if(supernodeMask[ib]>0){
//                             rowcounts[lk]++;
//                             int_t lib = LBi( ib, grid ); /* Local block number, row-wise. */
//                             colcounts[lib]++;
//                         }
//                     }
//                 }
//             }
// 		}
// 	}

//     for (int_t j=0; j<kc; j++){
//         if(rowcounts[j]>0){
//             if ( !(rowlists[j] = intCalloc_dist(rowcounts[j])) )
//                 ABORT("Calloc fails for rowlists[j].");
//         }else{
//             rowlists[j] = NULL;
//         }
//         rowcounts[j]=0;
//     }
//     for (int_t i=0; i<kr; i++){
//         if(colcounts[i]>0){
//             if ( !(collists[i] = intCalloc_dist(colcounts[i])) )
//                 ABORT("Calloc fails for collists[i].");
//         }else{
//             collists[i] = NULL;
//         }
//         colcounts[i]=0;
//     }

// 	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
// 		jb = mycol+lk*grid->npcol;  /* not sure */
// 		if(jb<nsupers){
//             if(supernodeMask[jb]>0){
//                 lsub = Llu->Lrowind_bc_ptr[lk];
//                 lloc = Llu->Lindval_loc_bc_ptr[lk];
//                 if(lsub){
//                     nlb = lsub[0];
//                     idx_i = nlb;
//                     for (int_t lb = 0; lb < nlb; ++lb){
//                         lptr1_tmp = lloc[lb+idx_i];
//                         ib = lsub[lptr1_tmp]; /* Global block number, row-wise. */
//                         if(supernodeMask[ib]>0){
//                             rowlists[lk][rowcounts[lk]++]=ib;
//                             int_t lib = LBi( ib, grid ); /* Local block number, row-wise. */
//                             collists[lib][colcounts[lib]++]=jb;
//                         }
//                     }
//                 }
//             }
// 		}
// 	}

//     /* broadcast tree for L and reduction tree for L^T*/

// 	if ( !(ActiveFlag = intCalloc_dist(grid->nprow*2)) )
// 		ABORT("Calloc fails for ActiveFlag[].");
// 	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->nprow * sizeof(int))) )
// 		ABORT("Malloc fails for ranks[].");
// 	if ( !(LBtree_ptr = (C_Tree*)SUPERLU_MALLOC(kc * sizeof(C_Tree))) )
// 		ABORT("Malloc fails for LBtree_ptr[].");
// 	for (int_t lk = 0; lk <kc ; ++lk) {
// 		C_BcTree_Nullify(&LBtree_ptr[lk]);
// 	}

// 	if ( !(URtree_ptr = (C_Tree*)SUPERLU_MALLOC(kc * sizeof(C_Tree))) )
// 		ABORT("Malloc fails for URtree_ptr[].");
// 	for (int_t lk = 0; lk <kc ; ++lk) {
// 		C_RdTree_Nullify(&URtree_ptr[lk]);
// 	}


// 	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
// 		jb = mycol+lk*grid->npcol;  /* not sure */
// 		if(jb<nsupers){
//             if(supernodeMask[jb]>0){
//                 // printf("iam %5d jb %5d \n",iam, jb);
//                 // fflush(stdout);
//                 pc = PCOL( jb, grid );
//                 count = rowcounts[lk];
//                 MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, cscp->comm);
//                 displs[0] = 0;
//                 nbg=0;
//                 for(int i=0; i<grid->nprow; ++i)
//                 {
//                     nbg +=recvcounts[i];
//                 }
//                 if(nbg>0){
//                     for(int i=0; i<grid->nprow-1; ++i)
//                     {
//                         displs[i+1] = displs[i] + recvcounts[i];
//                     }
//                     MPI_Allgatherv(rowlists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, cscp->comm);
//                 }
//                 for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j]=3*nsupers;
//                 for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j+grid->nprow]=j;
//                 for (int_t j=0;j<grid->nprow;++j)ranks[j]=-1;

//                 for (int_t i = 0; i < nbg; ++i) {
//                     gb = tmpglo[i];
//                     pr = PROW( gb, grid );
//                     ActiveFlag[pr]=SUPERLU_MIN(ActiveFlag[pr],gb);
//                 } /* for i ... */

//                 Root=-1;
//                 Iactive = 0;
//                 for (int_t j=0;j<grid->nprow;++j){
//                     if(ActiveFlag[j]!=3*nsupers){
//                     gb = ActiveFlag[j];
//                     pr = PROW( gb, grid );
//                     if(gb==jb)Root=pr;
//                     if(myrow==pr)Iactive=1;
//                     }
//                 }

//                 quickSortM(ActiveFlag,0,grid->nprow-1,grid->nprow,0,2);

//                 if(Iactive==1){
//                     // printf("iam %5d jb %5d Root %5d \n",iam, jb,Root);
//                     // fflush(stdout);
//                     assert( Root>-1 );
//                     rank_cnt = 1;
//                     ranks[0]=Root;
//                     for (int_t j = 0; j < grid->nprow; ++j){
//                         if(ActiveFlag[j]!=3*nsupers && ActiveFlag[j+grid->nprow]!=Root){
//                             ranks[rank_cnt]=ActiveFlag[j+grid->nprow];
//                             ++rank_cnt;
//                         }
//                     }

//                     if(rank_cnt>1){

//                         for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
//                             ranks[ii] = PNUM( ranks[ii], pc, grid );

//                         msgsize = SuperSize( jb );
//                         int needrecv=0;
//                         C_BcTree_Create_nv(&LBtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd',&needrecv);
//                         LBtree_ptr[lk].tag_=BC_L;

//                         int needrecvrd=0;
//                         int needsendrd=0;
//                         C_RdTree_Create_nv(&URtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd', &needrecvrd, &needsendrd);
//                         URtree_ptr[lk].tag_=RD_U;
                        
//                         // printf("iam %5d btree rank_cnt %5d \n",iam,rank_cnt);
//                         // fflush(stdout);

//                     }
//                 }
//             }
//         }
//     }
// 	SUPERLU_FREE(ActiveFlag);
// 	SUPERLU_FREE(ranks);


//     /* reduction tree for L and broadcast tree for L^T*/
// 	if ( !(LRtree_ptr = (C_Tree*)SUPERLU_MALLOC(kr * sizeof(C_Tree))) )
// 		ABORT("Malloc fails for LRtree_ptr[].");
// 	if ( !(ActiveFlag = intCalloc_dist(grid->npcol*2)) )
// 		ABORT("Calloc fails for ActiveFlag[].");
// 	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->npcol * sizeof(int))) )
// 		ABORT("Malloc fails for ranks[].");
// 	for (int_t lk = 0; lk <kr ; ++lk) {
// 		C_RdTree_Nullify(&LRtree_ptr[lk]);
// 	}
// 	if ( !(UBtree_ptr = (C_Tree*)SUPERLU_MALLOC(kr * sizeof(C_Tree))) )
// 		ABORT("Malloc fails for UBtree_ptr[].");
// 	for (int_t lk = 0; lk <kr ; ++lk) {
// 		C_BcTree_Nullify(&UBtree_ptr[lk]);
// 	}


// // #ifdef GPU_ACC
// // #ifdef HAVE_NVSHMEM
// //     if ( !(mystatusmod = (int*)SUPERLU_MALLOC(2*kr * sizeof(int))) )
// //         ABORT("Malloc fails for mystatusmod[].");
// // 	if ( !(h_recv_cnt = (int*)SUPERLU_MALLOC(kr * sizeof(int))) )
// //         ABORT("Malloc fails for mystatusmod[].");

// // 	int nfrecvmod=0;
// // 	for (int i=0;i<kr;i++){
// //         h_recv_cnt[i]=0;
// // 	}
// // 	for (int i=0;i<2*kr;i++) mystatusmod[i]=1;
// // #endif
// // #endif

// 	for (int_t lk=0;lk<kr;++lk){
// 		ib = myrow+lk*grid->nprow;  /* not sure */
// 		if(ib<nsupers){
//             if(supernodeMask[ib]>0){
//                 pr = PROW( ib, grid );

//                 count = colcounts[lk];
//                 MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, rscp->comm);
//                 displs[0] = 0;
//                 nbg=0;
//                 for(int i=0; i<grid->npcol; ++i)
//                 {
//                     nbg +=recvcounts[i];
//                 }
//                 if(nbg>0){
//                     for(int i=0; i<grid->npcol-1; ++i)
//                     {
//                         displs[i+1] = displs[i] + recvcounts[i];
//                     }
//                     MPI_Allgatherv(collists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, rscp->comm);
//                 }
//                 for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j]=-3*nsupers;
//                 for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j+grid->npcol]=j;
//                 for (int_t j=0;j<grid->npcol;++j)ranks[j]=-1;

//                 for (int_t j = 0; j < nbg; ++j) {
//                     gb = tmpglo[j];
//                     pc = PCOL( gb, grid );
//                     ActiveFlag[pc]=SUPERLU_MAX(ActiveFlag[pc],gb);
//                 } /* for j ... */

//                 Root=-1;
//                 Iactive = 0;

//                 for (int_t j=0;j<grid->npcol;++j){
//                     if(ActiveFlag[j]!=-3*nsupers){
//                     jb = ActiveFlag[j];
//                     pc = PCOL( jb, grid );
//                     if(jb==ib)Root=pc;
//                     if(mycol==pc)Iactive=1;
//                     }
//                 }

//                 quickSortM(ActiveFlag,0,grid->npcol-1,grid->npcol,1,2);

//                 if(Iactive==1){
//                     assert( Root>-1 );
//                     rank_cnt = 1;
//                     ranks[0]=Root;
//                     for (int_t j = 0; j < grid->npcol; ++j){
//                         if(ActiveFlag[j]!=-3*nsupers && ActiveFlag[j+grid->npcol]!=Root){
//                             ranks[rank_cnt]=ActiveFlag[j+grid->npcol];
//                             ++rank_cnt;
//                         }
//                     }
//                     if(rank_cnt>1){
//                         for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
//                             ranks[ii] = PNUM( pr, ranks[ii], grid );
//                         msgsize = SuperSize( ib );
//                         // C_RdTree_Create(&LRtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd');

//                         int needrecvrd=0;
//                         int needsendrd=0;
//                         C_RdTree_Create_nv(&LRtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd', &needrecvrd, &needsendrd);
//                         LRtree_ptr[lk].tag_=RD_L;
                        
//                         int needrecv=0;
//                         C_BcTree_Create_nv(&UBtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 'd',&needrecv);
//                         UBtree_ptr[lk].tag_=BC_U;   
                                             
//                     }
//                 }
//             }
//         }
//     }
// 	SUPERLU_FREE(ActiveFlag);
// 	SUPERLU_FREE(ranks);




//     for (int_t j=0; j<kc; j++){
//         if(rowlists[j]){
//             SUPERLU_FREE(rowlists[j]);
//         }
//     }
//     for (int_t i=0; i<kr; i++){
//         if(collists[i]){
//             SUPERLU_FREE(collists[i]);
//         }
//     }
//     SUPERLU_FREE(rowcounts);
//     SUPERLU_FREE(colcounts);
//     SUPERLU_FREE(rowlists);
//     SUPERLU_FREE(collists);


//     SUPERLU_FREE(tmpglo);
//     SUPERLU_FREE(recvcounts);
//     SUPERLU_FREE(displs);


// 	Llu->Ufstnz_br_ptr = Ufstnz_br_ptr;
// 	Llu->Unzval_br_ptr = Unzval_br_ptr;
// 	Llu->Ucb_indptr = Ucb_indptr;
// 	Llu->Ucb_valptr = Ucb_valptr;

// 	Llu->LRtree_ptr = LRtree_ptr;
// 	Llu->LBtree_ptr = LBtree_ptr;
// 	Llu->URtree_ptr = URtree_ptr;
// 	Llu->UBtree_ptr = UBtree_ptr;


//     Llu->nbcol_masked=0;
// 	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
// 		jb = mycol+lk*grid->npcol;  /* not sure */
//         if(jb<nsupers){
//             if(supernodeMask[jb]==1){ // only record the columns performed on GPU
//                Llu->nbcol_masked++;
//             }
//         }
//     }
//  	if ( !(Llu->bcols_masked =
// 				(int*)SUPERLU_MALLOC(Llu->nbcol_masked * sizeof(int))) ) {
// 		fprintf(stderr, "Malloc fails for nbcol_masked[].");
// 	}
//     Llu->nbcol_masked=0;
// 	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
// 		jb = mycol+lk*grid->npcol;  /* not sure */
//         if(jb<nsupers){
//             if(supernodeMask[jb]==1){ // only record the columns performed on GPU
//                Llu->bcols_masked[Llu->nbcol_masked++]=lk;
//             }
//         }
//     }
//     // printf("Llu->nbcol_masked: %10d\n",Llu->nbcol_masked);
//     // fflush(stdout);


// #ifdef GPU_ACC
//     if (get_acc_solve()){
// 	checkGPU(gpuMalloc( (void**)&Llu->d_bcols_masked, Llu->nbcol_masked * sizeof(int)));
// 	checkGPU(gpuMemcpy(Llu->d_bcols_masked, Llu->bcols_masked, Llu->nbcol_masked * sizeof(int), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_xsup, (n+1) * sizeof(int_t)));
// 	checkGPU(gpuMemcpy(Llu->d_xsup, xsup, (n+1) * sizeof(int_t), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_LRtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_LBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_URtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_UBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree)));
// 	checkGPU(gpuMemcpy(Llu->d_LRtree_ptr, Llu->LRtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMemcpy(Llu->d_LBtree_ptr, Llu->LBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMemcpy(Llu->d_URtree_ptr, Llu->URtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMemcpy(Llu->d_UBtree_ptr, Llu->UBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Lrowind_bc_dat, (Llu->Lrowind_bc_cnt) * sizeof(int_t)));
// 	checkGPU(gpuMemcpy(Llu->d_Lrowind_bc_dat, Llu->Lrowind_bc_dat, (Llu->Lrowind_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Lindval_loc_bc_dat, (Llu->Lindval_loc_bc_cnt) * sizeof(int_t)));
// 	checkGPU(gpuMemcpy(Llu->d_Lindval_loc_bc_dat, Llu->Lindval_loc_bc_dat, (Llu->Lindval_loc_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Lrowind_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
// 	checkGPU(gpuMemcpy(Llu->d_Lrowind_bc_offset, Llu->Lrowind_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Lindval_loc_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
// 	checkGPU(gpuMemcpy(Llu->d_Lindval_loc_bc_offset, Llu->Lindval_loc_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Lnzval_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
// 	checkGPU(gpuMemcpy(Llu->d_Lnzval_bc_offset, Llu->Lnzval_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
//     checkGPU(gpuMalloc( (void**)&Llu->d_grid, sizeof(gridinfo_t)));
//     checkGPU(gpuMemcpy(Llu->d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));

// 	// some dummy allocation to avoid checking whether they are null pointers later
// #if 0
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Ucolind_bc_dat, sizeof(int_t)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Ucolind_bc_offset, sizeof(int64_t)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Unzval_bc_dat, sizeof(double)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Unzval_bc_offset, sizeof(int64_t)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Uindval_loc_bc_dat, sizeof(int_t)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Uindval_loc_bc_offset, sizeof(int_t)));
// #else
//     Llu->d_Ucolind_bc_dat=NULL;
//     Llu->d_Ucolind_br_dat=NULL;
//     Llu->d_Ucolind_bc_offset=NULL;
//     Llu->d_Ucolind_br_offset=NULL;
//     Llu->d_Uind_br_dat=NULL;
//     Llu->d_Uind_br_offset=NULL;
//     Llu->d_Unzval_bc_dat=NULL;
//     Llu->d_Unzval_bc_offset=NULL;
//     Llu->d_Unzval_br_new_dat=NULL;
//     Llu->d_Unzval_br_new_offset=NULL;
//     Llu->d_Uindval_loc_bc_dat=NULL;
//     Llu->d_Uindval_loc_bc_offset=NULL;
// #endif


// 	checkGPU(gpuMalloc( (void**)&Llu->d_Linv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
// 	checkGPU(gpuMemcpy(Llu->d_Linv_bc_offset, Llu->Linv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Uinv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
// 	checkGPU(gpuMemcpy(Llu->d_Uinv_bc_offset, Llu->Uinv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_ilsum, (CEILING( nsupers, grid->nprow )+1) * sizeof(int_t)));
// 	checkGPU(gpuMemcpy(Llu->d_ilsum, Llu->ilsum, (CEILING( nsupers, grid->nprow )+1) * sizeof(int_t), gpuMemcpyHostToDevice));


// 	/* gpuMemcpy for the following is performed in pxgssvx/pxgssvx3d */
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Lnzval_bc_dat, (Llu->Lnzval_bc_cnt) * sizeof(double)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Linv_bc_dat, (Llu->Linv_bc_cnt) * sizeof(double)));
// 	checkGPU(gpuMalloc( (void**)&Llu->d_Uinv_bc_dat, (Llu->Uinv_bc_cnt) * sizeof(double)));


//        /* nvshmem related*/
// // #ifdef HAVE_NVSHMEM
// // 	checkGPU(gpuMalloc( (void**)&d_recv_cnt, CEILING(nsupers, grid->nprow) * sizeof(int)));
// // 	checkGPU(gpuMemcpy(d_recv_cnt, h_recv_cnt,  CEILING(nsupers, grid->nprow) * sizeof(int), gpuMemcpyHostToDevice));
// //         checkGPU(gpuMalloc( (void**)&d_recv_cnt_u, CEILING(nsupers, grid->nprow) * sizeof(int)));
// //         checkGPU(gpuMemcpy(d_recv_cnt_u, h_recv_cnt_u,  CEILING(nsupers, grid->nprow) * sizeof(int), gpuMemcpyHostToDevice));
// // #endif
//     }
// // #ifdef HAVE_NVSHMEM
// //     SUPERLU_FREE(h_recv_cnt);
// //     SUPERLU_FREE(h_recv_cnt_u);
// // #endif
// #endif /* end ifdef GPU_ACC */


//     return 0;
// } // end dtrs_compute_communication_structure_sym





int_t dtrs_x_reduction_newsolve(int_t nsupers, double* x, int nrhs, dLUstruct_t * LUstruct, gridinfo3d_t *grid3d, dtrf3Dpartition_t*  trf3Dpartition, double* recvbuf, xtrsTimer_t *xtrsTimer)

{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t myGrid = grid3d->zscp.Iam;
	int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
	int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;

	for (int_t ilvl = 1; ilvl < maxLvl ; ++ilvl)
	{
        if(!myZeroTrIdxs[ilvl-1]){ // this ensures the number of grids in communication is reduced by half every level down
            int_t sender, receiver;
            int_t tree = myTreeIdxs[ilvl];
            sForest_t** sForests = trf3Dpartition->sForests;
            sForest_t* sforest = sForests[tree];

            if ((myGrid % (1 << ilvl)) == 0)
            {
                sender = myGrid + (1 << (ilvl-1));
                receiver = myGrid;
            }
            else
            {
                sender = myGrid;
                receiver = myGrid - (1 << (ilvl-1));
            }
            int_t tr =  tree;
            for (int_t alvl = ilvl; alvl < maxLvl; alvl++)
            {
                /* code */
                // printf("myGrid %5d tr %5d sender %5d receiver %5d\n",myGrid,tr, sender, receiver);
                // fflush(stdout);
                dreduceSolvedX_newsolve(tr, sender, receiver, x, nrhs,  trf3Dpartition, LUstruct, grid3d, recvbuf, xtrsTimer);
                tr=(tr+1)/2-1;

            }
        }
	}

	return 0;
}



int_t dtrs_x_broadcast_newsolve(int_t nsupers, double* x, int nrhs, dLUstruct_t * LUstruct, gridinfo3d_t *grid3d, dtrf3Dpartition_t*  trf3Dpartition, double* recvbuf, xtrsTimer_t *xtrsTimer)

{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t myGrid = grid3d->zscp.Iam;
	int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
	int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;

	for (int_t ilvl = maxLvl-1; ilvl >0 ; --ilvl)
	{
        if(!myZeroTrIdxs[ilvl-1]){ // this ensures the number of grids in communication is doubled every level down
            int_t sender, receiver;
            int_t tree = myTreeIdxs[ilvl];
            if ((myGrid % (1 << ilvl)) == 0)
            {
                sender = myGrid;
                receiver = myGrid + (1 << (ilvl-1));
            }
            else
            {
                sender = myGrid - (1 << (ilvl-1));
                receiver = myGrid ;
            }
            int_t tr =  tree;
            for (int_t alvl = ilvl; alvl < maxLvl; alvl++)
            {
                // /* code */
                // printf("myGrid %5d tr %5d sender %5d receiver %5d\n",myGrid,tr, sender, receiver);
                // fflush(stdout);

                dp2pSolvedX3d(tr, sender, receiver, x, nrhs,  trf3Dpartition, LUstruct, grid3d, xtrsTimer);
                tr=(tr+1)/2-1;

            }
        }
	}

	return 0;
}




int_t dreduceSolvedX_newsolve(int_t treeId, int_t sender, int_t receiver, double* x, int nrhs,
                      dtrf3Dpartition_t*  trf3Dpartition, dLUstruct_t* LUstruct, gridinfo3d_t* grid3d, double* recvbuf, xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;
	sForest_t* sforest = sForests[treeId];
	if (!sforest) return 0;
	int_t nnodes = sforest->nNodes ;
	int_t *nodeList = sforest->nodeList ;

	gridinfo_t * grid = &(grid3d->grid2d);
	int_t myGrid = grid3d->zscp.Iam;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );
    double zero = 0.0;

	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow && mycol == kcol)
		{
			int_t lk = LBi(k, grid);
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize(k);
			if (myGrid == sender)
			{
				/* code */
				MPI_Send( &x[ii], knsupc * nrhs, MPI_DOUBLE, receiver, k,  grid3d->zscp.comm);
                for(int_t i=0; i<knsupc * nrhs; i++){
                    x[ii+i]=zero;
                }
                xtrsTimer->trsDataSendZ += knsupc * nrhs;
            }
			else
			{
				MPI_Status status;
				MPI_Recv( recvbuf, knsupc * nrhs, MPI_DOUBLE, sender, k, grid3d->zscp.comm, &status );
                for(int_t i=0; i<knsupc * nrhs; i++){
                    x[ii+i]+=recvbuf[i];
                }
                xtrsTimer->trsDataRecvZ += knsupc * nrhs;
			}
		}
	}

	return 0;
}




// Gather the solution vector from all grids to grid 0
int_t dtrs_X_gather3d(double* x, int nrhs, dtrf3Dpartition_t*  trf3Dpartition,
                     dLUstruct_t* LUstruct,
                     gridinfo3d_t* grid3d, xtrsTimer_t *xtrsTimer)

{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t myGrid = grid3d->zscp.Iam;
	int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;

	for (int_t ilvl = 0; ilvl < maxLvl - 1; ++ilvl)
	{
		int_t sender, receiver;
		if (!myZeroTrIdxs[ilvl])
		{
			if ((myGrid % (1 << (ilvl + 1))) == 0)
			{
				sender = myGrid + (1 << ilvl);
				receiver = myGrid;
			}
			else
			{
				sender = myGrid;
				receiver = myGrid - (1 << ilvl);
			}
			for (int_t alvl = 0; alvl <= ilvl; alvl++)
			{
				int_t diffLvl  = ilvl - alvl;
				int_t numTrees = 1 << diffLvl;
				int_t blvl = maxLvl - alvl - 1;
				int_t st = (1 << blvl) - 1 + (sender >> alvl);

				for (int_t tr = st; tr < st + numTrees; ++tr)
				{
					/* code */
					dp2pSolvedX3d(tr, sender, receiver, x, nrhs,  trf3Dpartition, LUstruct, grid3d, xtrsTimer);
				}
			}

		}
	}

	return 0;
}


int_t dp2pSolvedX3d(int_t treeId, int_t sender, int_t receiver, double* x, int nrhs,
                      dtrf3Dpartition_t*  trf3Dpartition, dLUstruct_t* LUstruct, gridinfo3d_t* grid3d, xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;
	sForest_t* sforest = sForests[treeId];
	if (!sforest) return 0;
	int_t nnodes = sforest->nNodes ;
	int_t *nodeList = sforest->nodeList ;

	gridinfo_t * grid = &(grid3d->grid2d);
	int_t myGrid = grid3d->zscp.Iam;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );


	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow && mycol == kcol)
		{
			int_t lk = LBi(k, grid);
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize(k);
			if (myGrid == sender)
			{
				/* code */
				MPI_Send( &x[ii], knsupc * nrhs, MPI_DOUBLE, receiver, k,  grid3d->zscp.comm);
                xtrsTimer->trsDataSendZ += knsupc * nrhs;
            }
			else
			{
				MPI_Status status;
				MPI_Recv( &x[ii], knsupc * nrhs, MPI_DOUBLE, sender, k, grid3d->zscp.comm, &status );
                xtrsTimer->trsDataRecvZ += knsupc * nrhs;
            }
		}
	}

	return 0;
}


int_t dfsolveReduceLsum3d(int_t treeId, int_t sender, int_t receiver, double* lsum, double* recvbuf, int nrhs,
                         dtrf3Dpartition_t*  trf3Dpartition, dLUstruct_t* LUstruct, gridinfo3d_t* grid3d ,
                         xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;
	sForest_t* sforest = sForests[treeId];
	if (!sforest) return 0;
	int_t nnodes = sforest->nNodes ;
	int_t *nodeList = sforest->nodeList ;

	gridinfo_t * grid = &(grid3d->grid2d);
	int_t myGrid = grid3d->zscp.Iam;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );


	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow )
		{
			int_t lk = LBi(k, grid);
			int_t knsupc = SuperSize(k);
			if (myGrid == sender)
			{
				/* code */
				int_t ii = LSUM_BLK (lk);
				double* lsum_k = &lsum[ii];
				superlu_scope_t *scp = &grid->rscp;
				MPI_Reduce( lsum_k, recvbuf, knsupc * nrhs,
				            MPI_DOUBLE, MPI_SUM, kcol, scp->comm);
				xtrsTimer->trsDataSendXY += knsupc * nrhs;
				xtrsTimer->trsDataRecvXY += knsupc * nrhs;
				if (mycol == kcol)
				{
					MPI_Send( recvbuf, knsupc * nrhs, MPI_DOUBLE, receiver, k,  grid3d->zscp.comm);
					xtrsTimer->trsDataSendZ += knsupc * nrhs;
				}
			}
			else
			{
				if (mycol == kcol)
				{
					MPI_Status status;
					MPI_Recv( recvbuf, knsupc * nrhs, MPI_DOUBLE, sender, k, grid3d->zscp.comm, &status );
					xtrsTimer->trsDataRecvZ += knsupc * nrhs;
					int_t ii = LSUM_BLK (lk);
					double* dest = &lsum[ii];
					double* tempv = recvbuf;
					for (int_t j = 0; j < nrhs; ++j)
					{
						for (int_t i = 0; i < knsupc; ++i)
                            dest[i + j * knsupc] += tempv[i + j * knsupc];
                    }
				}

			}
		}
	}

	return 0;
}



int_t dbsolve_Xt_bcast(int_t ilvl, dxT_struct *xT_s, int nrhs, dtrf3Dpartition_t*  trf3Dpartition,
                     dLUstruct_t * LUstruct,gridinfo3d_t* grid3d , xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t *ilsum = Llu->ilsum;
    int_t* xsup = Glu_persist->xsup;

	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	gridinfo_t * grid = &(grid3d->grid2d);
	int_t myGrid = grid3d->zscp.Iam;
		int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );


	double *xT = xT_s->xT;
	int_t *ilsumT = xT_s->ilsumT;
	int_t ldaspaT = xT_s->ldaspaT;


	int_t sender, receiver;

	if ((myGrid % (1 << (ilvl + 1))) == 0)
	{
		receiver = myGrid + (1 << ilvl);
		sender = myGrid;
	}
	else
	{
		receiver = myGrid;
		sender = myGrid - (1 << ilvl);
	}

	for (int_t alvl = ilvl + 1; alvl < maxLvl; ++alvl)
	{
		/* code */

		int_t treeId = trf3Dpartition->myTreeIdxs[alvl];
		sForest_t* sforest = trf3Dpartition->sForests[treeId];
		if (sforest)
		{
			/* code */
			int_t nnodes = sforest->nNodes;
			int_t* nodeList = sforest->nodeList;
			for (int_t k0 = 0; k0 < nnodes ; ++k0)
			{
				/* code */
				int_t k = nodeList[k0];
				int_t krow = PROW (k, grid);
				int_t kcol = PCOL (k, grid);
				int_t knsupc = SuperSize (k);
				if (myGrid == sender)
				{
					/* code */
					if (mycol == kcol &&   myrow == krow)
					{

						int_t lk = LBj (k, grid);
						int_t ii = XT_BLK (lk);
						double* xk = &xT[ii];
						MPI_Send( xk, knsupc * nrhs, MPI_DOUBLE, receiver, k,
						           grid3d->zscp.comm);
						xtrsTimer->trsDataSendZ += knsupc * nrhs;

					}
				}
				else
				{
					if (mycol == kcol)
					{
						/* code */
						if (myrow == krow )
						{
							/* code */
							int_t lk = LBj (k, grid);
							int_t ii = XT_BLK (lk);
							double* xk = &xT[ii];
							MPI_Status status;
							MPI_Recv( xk, knsupc * nrhs, MPI_DOUBLE, sender,k,
							           grid3d->zscp.comm, &status);
							xtrsTimer->trsDataRecvZ += knsupc * nrhs;
						}
						dbCastXk2Pck( k,  xT_s,  nrhs, LUstruct, grid, xtrsTimer);
					}

				}

			}
		}
	}


	return 0;
}




int_t dlsumForestFsolve(int_t k,
                       double *lsum, double *x, double* rtemp,  dxT_struct *xT_s, int    nrhs,
                       dLUstruct_t * LUstruct,
                       dtrf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat)
{
	gridinfo_t * grid = &(grid3d->grid2d);
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t* xsup = Glu_persist->xsup;

	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t lk = LBj( k, grid ); /* Local block number, column-wise. */
	int_t *lsub = Llu->Lrowind_bc_ptr[lk];
	if (!lsub) return 0;
	double* lusup = Llu->Lnzval_bc_ptr[lk];
	int nsupr = lsub[1];
	int_t nlb = lsub[0];
	int_t lptr = BC_HEADER;
	int_t luptr = 0;
	int_t krow = PROW (k, grid);
	int knsupc = SuperSize(k);
	if (myrow == krow)
	{
		/* code */
		nlb = lsub[0] - 1;
		lptr +=  LB_DESCRIPTOR + knsupc;
		luptr += knsupc;
	}

	double *xT = xT_s->xT;
	int_t *ilsumT = xT_s->ilsumT;
	int_t ldaspaT = xT_s->ldaspaT;


	int_t *ilsum = Llu->ilsum;
	int_t ii = XT_BLK (lk);
	double* xk = &xT[ii];
	for (int_t lb = 0; lb < nlb; ++lb)
	{
		int_t ik = lsub[lptr]; /* Global block number, row-wise. */
		int nbrow = lsub[lptr + 1];
        double alpha = 1.0;
        double beta = 0.0;
#ifdef _CRAY
		SGEMM( ftcs2, ftcs2, &nbrow, &nrhs, &knsupc,
		       &alpha, &lusup[luptr], &nsupr, xk,
		       &knsupc, &beta, rtemp, &nbrow );
#elif defined (USE_VENDOR_BLAS)
		dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
		        &alpha, &lusup[luptr], &nsupr, xk,
		        &knsupc, &beta, rtemp, &nbrow, 1, 1 );
#else
		dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
		        &alpha, &lusup[luptr], &nsupr, xk,
		        &knsupc, &beta, rtemp, &nbrow );
#endif
		stat->ops[SOLVE] += 2 * nbrow * nrhs * knsupc + nbrow * nrhs;

		int_t lk = LBi( ik, grid ); /* Local block number, row-wise. */
		int_t iknsupc = SuperSize( ik );
		int_t il = LSUM_BLK( lk );
		double* dest = &lsum[il];
		lptr += LB_DESCRIPTOR;
		int_t rel = xsup[ik]; /* Global row index of block ik. */
		for (int_t i = 0; i < nbrow; ++i)
		{
			int_t irow = lsub[lptr++] - rel; /* Relative row. */
			for (int_t j = 0; j < nrhs; ++j)
                dest[irow + j * iknsupc] -= rtemp[i + j * nbrow];
		}
		luptr += nbrow;
	}

	return 0;
}



int_t dnonLeafForestForwardSolve3d( int_t treeId,  dLUstruct_t * LUstruct,
                                   dScalePermstruct_t * ScalePermstruct,
                                   dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                                   double * x, double * lsum,
                                   dxT_struct *xT_s,
                                   double * recvbuf, double* rtemp,
                                   MPI_Request * send_req,
                                   int nrhs,
                                   dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{

	sForest_t** sForests = trf3Dpartition->sForests;

	sForest_t* sforest = sForests[treeId];
	if (!sforest)
	{
		/* code */
		return 0;
	}
	int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
	if (nnodes < 1) return 1;
	int_t *perm_c_supno = sforest->nodeList ;
	gridinfo_t * grid = &(grid3d->grid2d);

	dLocalLU_t *Llu = LUstruct->Llu;
	int_t *ilsum = Llu->ilsum;

	int_t* xsup =  LUstruct->Glu_persist->xsup;

	double *xT = xT_s->xT;
	int_t *ilsumT = xT_s->ilsumT;
	int_t ldaspaT = xT_s->ldaspaT;

	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);

	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = perm_c_supno[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);
		// printf("doing %d \n", k);
		/**
		 * Pkk(Yk) = sumOver_PrK (Yk)
		 */
		if (myrow == krow )
		{
			double tx = SuperLU_timer_();
			dlsumReducePrK(k, x, lsum, recvbuf, nrhs, LUstruct, grid,xtrsTimer);
			// xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
			xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
		}

		if (mycol == kcol )
		{
			int_t lk = LBi (k, grid); /* Local block number, row-wise. */
			int_t ii = X_BLK (lk);
			if (myrow == krow )
			{
				/* Diagonal process. */
				double tx = SuperLU_timer_();
				dlocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				int_t lkj = LBj (k, grid);
				int_t jj = XT_BLK (lkj);
				int_t knsupc = SuperSize(k);
				memcpy(&xT[jj], &x[ii], knsupc * nrhs * sizeof(double) );
				xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
			}                       /* if diagonal process ... */
			/*
			 * Send Xk to process column Pc[k].
			 */
			double tx = SuperLU_timer_();
			dbCastXk2Pck( k,  xT_s,  nrhs, LUstruct, grid, xtrsTimer);
			xtrsTimer->tfs_comm += SuperLU_timer_() - tx;

			/*
			 * Perform local block modifications: lsum[i] -= U_i,k * X[k]
			 * where i is in current sforest
			 */
			tx = SuperLU_timer_();
			dlsumForestFsolve(k, lsum, x, rtemp, xT_s, nrhs,
			                 LUstruct, trf3Dpartition, grid3d, stat);
			xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
		}
	}                           /* for k ... */
	return 0;
}


int_t dleafForestForwardSolve3d(superlu_dist_options_t *options, int_t treeId, int_t n,  dLUstruct_t * LUstruct,
                               dScalePermstruct_t * ScalePermstruct,
                               dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                               double * x, double * lsum, double * recvbuf, double* rtemp,
                               MPI_Request * send_req,
                               int nrhs,
                               dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;

	sForest_t* sforest = sForests[treeId];
	if (!sforest) return 0;
	int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
	if (nnodes < 1)
	{
		return 1;
	}
	gridinfo_t * grid = &(grid3d->grid2d);
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t* xsup = Glu_persist->xsup;
	int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
	int_t nsupers = Glu_persist->supno[n - 1] + 1;
	int_t Pr = grid->nprow;
	int_t nlb = CEILING (nsupers, Pr);

	treeTopoInfo_t* treeTopoInfo = &sforest->topoInfo;
	int_t* eTreeTopLims = treeTopoInfo->eTreeTopLims;
	int_t *nodeList = sforest->nodeList ;


	int_t knsupc = sp_ienv_dist (3,options);
	int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);

	int **fsendx_plist = Llu->fsendx_plist;
	int_t* ilsum = Llu->ilsum;

	int* fmod = getfmodLeaf(nlb, LUstruct->Llu->fmod);
	int* frecv = getfrecvLeaf(sforest, nlb, fmod, LUstruct->Llu->mod_bit, grid);
	Llu->frecv = frecv;
	int  nfrecvx = getNfrecvxLeaf(sforest, LUstruct->Llu->Lrowind_bc_ptr, grid);
	int nleaf = 0;
	int_t nfrecvmod = getNfrecvmodLeaf(&nleaf, sforest, frecv, fmod,  grid);
    int_t myGrid = grid3d->zscp.Iam;
    // printf("igrid %5d, iam %5d, nfrecvx %5d, nfrecvmod %5d, nleaf %5d\n",myGrid,iam,nfrecvx,nfrecvmod,nleaf);


	/* factor the leaf to being the factorization*/
	for (int_t k0 = 0; k0 < nnodes && nleaf; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);
		if (myrow == krow && mycol == kcol)
		{
			/* Diagonal process */
			int_t knsupc = SuperSize (k);
			int_t lk = LBi (k, grid);
			if (frecv[lk] == 0 && fmod[lk] == 0)
			{
				double tx = SuperLU_timer_();
				fmod[lk] = -1;  /* Do not solve X[k] in the future. */
				int_t ii = X_BLK (lk);

				int_t lkj = LBj (k, grid); /* Local block number, column-wise. */
				int_t* lsub = Lrowind_bc_ptr[lkj];
				dlocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				diBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, fsendx_plist, send_req, LUstruct, grid,xtrsTimer);
				nleaf--;
				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				int_t nb = lsub[0] - 1;
				int_t lptr = BC_HEADER + LB_DESCRIPTOR + knsupc;
				int_t luptr = knsupc; /* Skip diagonal block L(k,k). */

				dlsum_fmod_leaf (treeId, trf3Dpartition, lsum, x, &x[ii], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
				xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
			}
		}                       /* if diagonal process ... */
	}


	while (nfrecvx || nfrecvmod)
	{
		/* While not finished. */
		/* Receive a message. */
		MPI_Status status;
		double tx = SuperLU_timer_();
		MPI_Recv (recvbuf, maxrecvsz, MPI_DOUBLE,
		          MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status);
		xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
		int_t k = *recvbuf;
		xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
		tx = SuperLU_timer_();
		switch (status.MPI_TAG)
		{
		case Xk:
		{
			--nfrecvx;
			int_t lk = LBj (k, grid); /* Local block number, column-wise. */
			int_t *lsub = Lrowind_bc_ptr[lk];

			if (lsub)
			{
				int_t nb = lsub[0];
				int_t lptr = BC_HEADER;
				int_t luptr = 0;
				int_t knsupc = SuperSize (k);

				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				dlsum_fmod_leaf (treeId, trf3Dpartition, lsum, x, &recvbuf[XK_H], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
			}                   /* if lsub */

			break;
		}

		case LSUM:             /* Receiver must be a diagonal process */
		{
			--nfrecvmod;
			int_t lk = LBi (k, grid); /* Local block number, row-wise. */
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize (k);
			double* tempv = &recvbuf[LSUM_H];
			for (int_t j = 0; j < nrhs; ++j)
			{
				for (int_t i = 0; i < knsupc; ++i)
					x[i + ii + j * knsupc] += tempv[i + j * knsupc];
			}

			if ((--frecv[lk]) == 0 && fmod[lk] == 0)
			{
				fmod[lk] = -1;  /* Do not solve X[k] in the future. */
				lk = LBj (k, grid); /* Local block number, column-wise. */
				int_t *lsub = Lrowind_bc_ptr[lk];
				dlocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				/*
				  * Send Xk to process column Pc[k].
				  */
				diBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, fsendx_plist, send_req, LUstruct, grid, xtrsTimer);
				/*
				 * Perform local block modifications.
				 */
				int_t nb = lsub[0] - 1;
				int_t lptr = BC_HEADER + LB_DESCRIPTOR + knsupc;
				int_t luptr = knsupc; /* Skip diagonal block L(k,k). */

				dlsum_fmod_leaf (treeId, trf3Dpartition, lsum, x, &x[ii], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
			}                   /* if */

			break;
		}

		default:
		{
			// printf ("(%2d) Recv'd wrong message tag %4d\n", status.MPI_TAG);
			break;
		}

		}                       /* switch */
		xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
	}                           /* while not finished ... */
	SUPERLU_FREE (fmod);
	SUPERLU_FREE (frecv);
	double tx = SuperLU_timer_();
	for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
	{
		MPI_Status status;
		MPI_Wait (&send_req[i], &status);
	}
	Llu->SolveMsgSent = 0;
	xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
	MPI_Barrier (grid->comm);
	return 0;
}




void dlsum_fmod_leaf (
    int_t treeId,
    dtrf3Dpartition_t*  trf3Dpartition,
    double *lsum,    /* Sum of local modifications.                        */
    double *x,       /* X array (local)                                    */
    double *xk,      /* X[k].                                              */
    double *rtemp,   /* Result of full matrix-vector multiply.             */
    int   nrhs,      /* Number of right-hand sides.                        */
    int   knsupc,    /* Size of supernode k.                               */
    int_t k,         /* The k-th component of X.                           */
    int *fmod,     /* Modification count for L-solve.                    */
    int_t nlb,       /* Number of L blocks.                                */
    int_t lptr,      /* Starting position in lsub[*].                      */
    int_t luptr,     /* Starting position in lusup[*].                     */
    int_t *xsup,
    gridinfo_t *grid,
    dLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat,xtrsTimer_t *xtrsTimer)

{
    double alpha = 1.0;
    double beta = 0.0;
	double *lusup, *lusup1;
	double *dest;
	int    iam, iknsupc, myrow, nbrow, nsupr, nsupr1, p, pi;
	int_t  i, ii, ik, il, ikcol, irow, j, lb, lk, lib, rel;
	int_t  *lsub, *lsub1, nlb1, lptr1, luptr1;
	int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
	int  *frecv = Llu->frecv;
	int  **fsendx_plist = Llu->fsendx_plist;
	MPI_Status status;
	int test_flag;

#if ( PROFlevel>=1 )
	double t1, t2;
	float msg_vol = 0, msg_cnt = 0;
#endif
#if ( PROFlevel>=1 )
	TIC(t1);
#endif

	iam = grid->iam;
	myrow = MYROW( iam, grid );
	lk = LBj( k, grid ); /* Local block number, column-wise. */
	lsub = Llu->Lrowind_bc_ptr[lk];
	lusup = Llu->Lnzval_bc_ptr[lk];
	nsupr = lsub[1];

	for (lb = 0; lb < nlb; ++lb)
	{
		ik = lsub[lptr]; /* Global block number, row-wise. */
		nbrow = lsub[lptr + 1];
#ifdef _CRAY
		SGEMM( ftcs2, ftcs2, &nbrow, &nrhs, &knsupc,
		       &alpha, &lusup[luptr], &nsupr, xk,
		       &knsupc, &beta, rtemp, &nbrow );
#elif defined (USE_VENDOR_BLAS)
		dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
		        &alpha, &lusup[luptr], &nsupr, xk,
		        &knsupc, &beta, rtemp, &nbrow, 1, 1 );
#else
		dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
		        &alpha, &lusup[luptr], &nsupr, xk,
		        &knsupc, &beta, rtemp, &nbrow );
#endif
		stat->ops[SOLVE] += 2 * nbrow * nrhs * knsupc + nbrow * nrhs;

		lk = LBi( ik, grid ); /* Local block number, row-wise. */
		iknsupc = SuperSize( ik );
		il = LSUM_BLK( lk );
		dest = &lsum[il];
		lptr += LB_DESCRIPTOR;
		rel = xsup[ik]; /* Global row index of block ik. */
		for (i = 0; i < nbrow; ++i)
		{
			irow = lsub[lptr++] - rel; /* Relative row. */
			RHS_ITERATE(j)
            dest[irow + j * iknsupc] -= rtemp[i + j * nbrow];
		}
		luptr += nbrow;

#if ( PROFlevel>=1 )
		TOC(t2, t1);
		stat->utime[SOL_GEMM] += t2;
#endif


		if ( (--fmod[lk]) == 0  )   /* Local accumulation done. */
		{
			if (trf3Dpartition->supernode2treeMap[ik] == treeId)
			{
				ikcol = PCOL( ik, grid );
				p = PNUM( myrow, ikcol, grid );
				if ( iam != p )
				{
#ifdef ISEND_IRECV
					MPI_Isend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
					           MPI_DOUBLE, p, LSUM, grid->comm,
					           &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
					MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
					           MPI_DOUBLE, p, LSUM, grid->comm );
#else
					MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
					          MPI_DOUBLE, p, LSUM, grid->comm );
#endif
#endif
					xtrsTimer->trsDataSendXY += iknsupc * nrhs + LSUM_H;
				}
				else     /* Diagonal process: X[i] += lsum[i]. */
				{
					ii = X_BLK( lk );
					RHS_ITERATE(j)
					for (i = 0; i < iknsupc; ++i)
					    x[i + ii + j * iknsupc] += lsum[i + il + j * iknsupc];

					if ( frecv[lk] == 0 )   /* Becomes a leaf node. */
					{
						fmod[lk] = -1; /* Do not solve X[k] in the future. */


						lk = LBj( ik, grid );/* Local block number, column-wise. */
						lsub1 = Llu->Lrowind_bc_ptr[lk];
						lusup1 = Llu->Lnzval_bc_ptr[lk];
						nsupr1 = lsub1[1];
#ifdef _CRAY
						STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &iknsupc, &nrhs, &alpha,
						      lusup1, &nsupr1, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
						dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
						       lusup1, &nsupr1, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
						dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
						       lusup1, &nsupr1, &x[ii], &iknsupc);
#endif


						stat->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;

						/*
						 * Send Xk to process column Pc[k].
						 */
						for (p = 0; p < grid->nprow; ++p)
						{
							if ( fsendx_plist[lk][p] != SLU_EMPTY )
							{
								pi = PNUM( p, ikcol, grid );
#ifdef ISEND_IRECV
								MPI_Isend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
								           MPI_DOUBLE, pi, Xk, grid->comm,
								           &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
								MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
								           MPI_DOUBLE, pi, Xk, grid->comm );
#else
								MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
								          MPI_DOUBLE, pi, Xk, grid->comm );
#endif
#endif

							}
						}
						xtrsTimer->trsDataSendXY += iknsupc * nrhs + XK_H;
						/*
						 * Perform local block modifications.
						 */
						nlb1 = lsub1[0] - 1;
						lptr1 = BC_HEADER + LB_DESCRIPTOR + iknsupc;
						luptr1 = iknsupc; /* Skip diagonal block L(I,I). */

						dlsum_fmod_leaf(treeId, trf3Dpartition,
						                lsum, x, &x[ii], rtemp, nrhs, iknsupc, ik,
						                fmod, nlb1, lptr1, luptr1, xsup,
						                grid, Llu, send_req, stat,xtrsTimer);
					} /* if frecv[lk] == 0 */
				} /* if iam == p */
			}
		}/* if fmod[lk] == 0 */

	} /* for lb ... */

} /* dLSUM_FMOD_LEAF */


int_t dleafForestForwardSolve3d_newsolve(superlu_dist_options_t *options, int_t n,  dLUstruct_t * LUstruct,
                               dScalePermstruct_t * ScalePermstruct,
                               dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                               double * x, double * lsum, double * recvbuf, double* rtemp,
                               MPI_Request * send_req,
                               int nrhs,
                               dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;

	// sForest_t* sforest = sForests[treeId];
	// if (!sforest) return 0;
	// int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
	// if (nnodes < 1)
	// {
	// 	return 1;
	// }
	gridinfo_t * grid = &(grid3d->grid2d);
	int_t iam = grid->iam;
    int_t myGrid = grid3d->zscp.Iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t* xsup = Glu_persist->xsup;
	int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
	int_t nsupers = Glu_persist->supno[n - 1] + 1;
	int_t Pr = grid->nprow;
	int_t nlb = CEILING (nsupers, Pr);
    int* supernodeMask = trf3Dpartition->supernodeMask;

	// treeTopoInfo_t* treeTopoInfo = &sforest->topoInfo;
	// int_t* eTreeTopLims = treeTopoInfo->eTreeTopLims;
	// int_t *nodeList = sforest->nodeList ;


	int_t knsupc = sp_ienv_dist (3,options);
	int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);

	int **fsendx_plist = Llu->fsendx_plist;
	int_t* ilsum = Llu->ilsum;

	int* fmod = getfmod_newsolve(nlb, nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);
	int* frecv = getfrecv_newsolve(nsupers, supernodeMask, nlb, fmod, LUstruct->Llu->mod_bit, grid);
	Llu->frecv = frecv;
	int  nfrecvx = getNfrecvx_newsolve(nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);
	int  nleaf = 0;
	int_t nfrecvmod = getNfrecvmod_newsolve(&nleaf, nsupers, supernodeMask, frecv, fmod,  grid);
    // printf("igrid %5d, iam %5d, nfrecvx %5d, nfrecvmod %5d, nleaf %5d\n",myGrid,iam,nfrecvx,nfrecvmod,nleaf);

	/* factor the leaf to being the factorization*/
	for (int_t k = 0; k < nsupers && nleaf; ++k)
	{
        if(supernodeMask[k]>0){
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);
		if (myrow == krow && mycol == kcol)
		{
			/* Diagonal process */
			int_t knsupc = SuperSize (k);
			int_t lk = LBi (k, grid);
			if (frecv[lk] == 0 && fmod[lk] == 0)
			{
				double tx = SuperLU_timer_();
				fmod[lk] = -1;  /* Do not solve X[k] in the future. */
				int_t ii = X_BLK (lk);

				int_t lkj = LBj (k, grid); /* Local block number, column-wise. */
				int_t* lsub = Lrowind_bc_ptr[lkj];
				dlocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				diBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, fsendx_plist, send_req, LUstruct, grid,xtrsTimer);
				nleaf--;
				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				int_t nb = lsub[0] - 1;
				int_t lptr = BC_HEADER + LB_DESCRIPTOR + knsupc;
				int_t luptr = knsupc; /* Skip diagonal block L(k,k). */

				dlsum_fmod_leaf_newsolve (trf3Dpartition, lsum, x, &x[ii], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
				xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
			}
		}                       /* if diagonal process ... */
        }
	}


	while (nfrecvx || nfrecvmod)
	{
		/* While not finished. */
		/* Receive a message. */
		MPI_Status status;
		double tx = SuperLU_timer_();
		MPI_Recv (recvbuf, maxrecvsz, MPI_DOUBLE,
		          MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status);
		xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
		int_t k = *recvbuf;
		xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
		tx = SuperLU_timer_();
		switch (status.MPI_TAG)
		{
		case Xk:
		{
			--nfrecvx;
			int_t lk = LBj (k, grid); /* Local block number, column-wise. */
			int_t *lsub = Lrowind_bc_ptr[lk];

			if (lsub)
			{
				int_t nb = lsub[0];
				int_t lptr = BC_HEADER;
				int_t luptr = 0;
				int_t knsupc = SuperSize (k);

				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				dlsum_fmod_leaf_newsolve (trf3Dpartition, lsum, x, &recvbuf[XK_H], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
			}                   /* if lsub */

			break;
		}

		case LSUM:             /* Receiver must be a diagonal process */
		{
			--nfrecvmod;
			int_t lk = LBi (k, grid); /* Local block number, row-wise. */
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize (k);
			double* tempv = &recvbuf[LSUM_H];
			for (int_t j = 0; j < nrhs; ++j)
			{
				for (int_t i = 0; i < knsupc; ++i)
					x[i + ii + j * knsupc] += tempv[i + j * knsupc];
			}

			if ((--frecv[lk]) == 0 && fmod[lk] == 0)
			{
				fmod[lk] = -1;  /* Do not solve X[k] in the future. */
				lk = LBj (k, grid); /* Local block number, column-wise. */
				int_t *lsub = Lrowind_bc_ptr[lk];
				dlocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				/*
				  * Send Xk to process column Pc[k].
				  */
				diBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, fsendx_plist, send_req, LUstruct, grid, xtrsTimer);
				/*
				 * Perform local block modifications.
				 */
				int_t nb = lsub[0] - 1;
				int_t lptr = BC_HEADER + LB_DESCRIPTOR + knsupc;
				int_t luptr = knsupc; /* Skip diagonal block L(k,k). */

				dlsum_fmod_leaf_newsolve (trf3Dpartition, lsum, x, &x[ii], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
			}                   /* if */

			break;
		}

		default:
		{
			// printf ("(%2d) Recv'd wrong message tag %4d\n", status.MPI_TAG);
			break;
		}

		}                       /* switch */
		xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
	}                           /* while not finished ... */
	SUPERLU_FREE (fmod);
	SUPERLU_FREE (frecv);
	double tx = SuperLU_timer_();
	for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
	{
		MPI_Status status;
		MPI_Wait (&send_req[i], &status);
	}
	Llu->SolveMsgSent = 0;
	xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
	MPI_Barrier (grid->comm);
	return 0;
}





void dForwardSolve3d_newsolve_reusepdgstrs(superlu_dist_options_t *options, int_t n,  dLUstruct_t * LUstruct,
                               dScalePermstruct_t * ScalePermstruct,
                               int*  supernodeMask, gridinfo3d_t *grid3d,
                               double * x, double * lsum,
                               int nrhs,
                               dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    int_t myGrid = grid3d->zscp.Iam;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    double alpha = 1.0;
    double beta = 0.0;
    double zero = 0.0;
    double *lusup, *dest;
    double *recvbuf, *recvbuf_on, *tempv,
            *recvbufall, *recvbuf_BC_fwd, *recvbuf0, *xin, *recvbuf_BC_gpu,*recvbuf_RD_gpu;
    double *rtemp, *rtemp_loc; /* Result of full matrix-vector multiply. */
    double *Linv; /* Inverse of diagonal block */
    double *Uinv; /* Inverse of diagonal block */
    int *ipiv;
    int_t *leaf_send;
    int_t nleaf_send, nleaf_send_tmp;
    int_t *root_send;
    int_t nroot_send, nroot_send_tmp;
    int_t  **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
        /*-- Data structures used for broadcast and reduction trees. --*/
    C_Tree  *LBtree_ptr = Llu->LBtree_ptr;
    C_Tree  *LRtree_ptr = Llu->LRtree_ptr;
    C_Tree  *UBtree_ptr = Llu->UBtree_ptr;
    C_Tree  *URtree_ptr = Llu->URtree_ptr;
    int_t  *Urbs1; /* Number of row blocks in each block column of U. */
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t  kcol, krow, mycol, myrow;
    int_t  i, ii, il, j, jj, k, kk, lb, ljb, lk, lib, lptr, luptr, gb, nn;
    int_t  nb, nlb,nlb_nodiag, nub, nsupers, nsupers_j, nsupers_i,maxsuper;
    int_t  *xsup, *supno, *lsub, *usub;
    int_t  *ilsum;    /* Starting position of each supernode in lsum (LOCAL)*/
    int    Pc, Pr, iam;
    int    knsupc, nsupr, nprobe;
    int    nbtree, nrtree, outcount;
    int    ldalsum;   /* Number of lsum entries locally owned. */
    int    maxrecvsz, p, pi;
    int_t  **Lrowind_bc_ptr;
    double **Lnzval_bc_ptr;
    double **Linv_bc_ptr;
    double **Uinv_bc_ptr;
    double sum;
    MPI_Status status,status_on,statusx,statuslsum;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
    SuperLUStat_t **stat_loc;

    double tmax;
    	/*-- Counts used for L-solve --*/
    int  *fmod;         /* Modification count for L-solve --
    			 Count the number of local block products to
    			 be summed into lsum[lk]. */
	int_t *fmod_sort;
	int_t *order;
	//int_t *order1;
	//int_t *order2;
    int fmod_tmp;
    int  **fsendx_plist = Llu->fsendx_plist;
    int  nfrecvx_buf=0;
    int *frecv;        /* Count of lsum[lk] contributions to be received
    			 from processes in this row.
    			 It is only valid on the diagonal processes. */
    int  frecv_tmp;
    int  nfrecvmod = 0; /* Count of total modifications to be recv'd. */
    int  nfrecv = 0; /* Count of total messages to be recv'd. */
    int  nbrecv = 0; /* Count of total messages to be recv'd. */
    int  nleaf = 0, nroot = 0;
    int  nleaftmp = 0, nroottmp = 0;
    int_t  msgsize;
        /*-- Counts used for U-solve --*/
    int  *bmod;         /* Modification count for U-solve. */
    int  bmod_tmp;
    int  **bsendx_plist = Llu->bsendx_plist;
    int  nbrecvx = Llu->nbrecvx; /* Number of X components to be recv'd. */
    int  nbrecvx_buf=0;
    int  *brecv;        /* Count of modifications to be recv'd from
    			 processes in this row. */
    int_t  nbrecvmod = 0; /* Count of total modifications to be recv'd. */
    int_t flagx,flaglsum,flag;
    int_t *LBTree_active, *LRTree_active, *LBTree_finish, *LRTree_finish, *leafsups, *rootsups;
    int_t TAG;
    double t1_sol, t2_sol, t;
#if ( DEBUGlevel>=2 )
    int_t Ublocks = 0;
#endif

    int_t gik,iklrow,fnz;

    int *mod_bit = Llu->mod_bit; /* flag contribution from each row block */
    int INFO, pad;
    int_t tmpresult;

    // #if ( PROFlevel>=1 )
    double t1, t2, t3;
    float msg_vol = 0, msg_cnt = 0;
    // #endif

    int_t msgcnt[4]; /* Count the size of the message xfer'd in each buffer:
		      *     0 : transferred in Lsub_buf[]
		      *     1 : transferred in Lval_buf[]
		      *     2 : transferred in Usub_buf[]
		      *     3 : transferred in Uval_buf[]
		      */
    int iword = sizeof (int_t);
    int dword = sizeof (double);
    int Nwork;
    int_t procs = grid->nprow * grid->npcol;
    yes_no_t done;
    yes_no_t startforward;
    int nbrow;
    int_t  ik, rel, idx_r, jb, nrbl, irow, pc,iknsupc;
    int_t  lptr1_tmp, idx_i, idx_v,m;
    int_t ready;
    int thread_id = 0;
    yes_no_t empty;
    int_t sizelsum,sizertemp,aln_d,aln_i;
    aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    int num_thread = 1;
	int_t cnt1,cnt2;
    double tx;



#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK)

#if ( PRNTlevel>=1 )

    if (get_acc_solve()) /* GPU trisolve*/
    {
    iam = grid->iam;
	if ( !iam) printf(".. GPU trisolve\n");
	fflush(stdout);
    }
#endif

	const int nwrp_block = 1; /* number of warps in each block */
	const int warp_size = 32; /* number of threads per warp*/
	gpuStream_t sid=0;
	int gid=0;
	gridinfo_t *d_grid = NULL;
	double *d_x = NULL;
	double *d_lsum = NULL;
    int  *d_fmod = NULL;
#endif


// cudaProfilerStart();
    maxsuper = sp_ienv_dist(3, options);

#ifdef _OPENMP
#pragma omp parallel default(shared)
    {
    	if (omp_get_thread_num () == 0) {
    		num_thread = omp_get_num_threads ();
    	}
    }
#else
	num_thread=1;
#endif

    // MPI_Barrier( grid->comm );
    t1_sol = SuperLU_timer_();
    t = SuperLU_timer_();


    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW( iam, grid );
    mycol = MYCOL( iam, grid );
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = supno[n-1] + 1;
    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    Linv_bc_ptr = Llu->Linv_bc_ptr;
    Uinv_bc_ptr = Llu->Uinv_bc_ptr;
    nlb = CEILING( nsupers, Pr ); /* Number of local block rows. */

    stat->utime[SOL_COMM] = 0.0;
    stat->utime[SOL_GEMM] = 0.0;
    stat->utime[SOL_TRSM] = 0.0;
    stat->utime[SOL_TOT] = 0.0;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter dForwardSolve3d_newsolve_reusepdgstrs()");
#endif

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    /* Save the count to be altered so it can be used by
       subsequent call to PDGSTRS. */

/* skip fmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
	fmod = getfmod_newsolve(nlb, nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);
}
	int  nfrecvx = getNfrecvx_newsolve(nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);

    if ( !(frecv = int32Calloc_dist(nlb)) )
	ABORT("Calloc fails for frecv[].");
    Llu->frecv = frecv;

    if ( !(leaf_send = intMalloc_dist((CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i)) )
	ABORT("Malloc fails for leaf_send[].");
    nleaf_send=0;


#ifdef _CRAY
    ftcs1 = _cptofcd("L", strlen("L"));
    ftcs2 = _cptofcd("N", strlen("N"));
    ftcs3 = _cptofcd("U", strlen("U"));
#endif


    /* Obtain ilsum[] and ldalsum for process column 0. */
    ilsum = Llu->ilsum;
    ldalsum = Llu->ldalsum;

    /* Allocate working storage. */
    knsupc = sp_ienv_dist(3, options);
    maxrecvsz = pdgstrs3d_checked_workspace_count(knsupc, nrhs, 1,
                                                  SUPERLU_MAX( XK_H, LSUM_H ),
                                                  "3D forward recvbuf");
    sizelsum = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb, LSUM_H,
                                                 "3D forward lsum workspace");
    if (aln_d > 1) {
        size_t lsum_aligned = (size_t) sizelsum;
        size_t addend = (size_t) aln_d - 1;
        if (lsum_aligned > ((size_t)-1) - addend)
            ABORT("Workspace size overflows allocation size.");
        lsum_aligned = ((lsum_aligned + addend) / (size_t) aln_d) * (size_t) aln_d;
        sizelsum = pdgstrs3d_checked_size_to_int_t(lsum_aligned,
                                                   "3D forward lsum workspace");
    }
    int_t x_count = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb,
                                                      XK_H,
                                                      "3D forward x workspace");
    sizertemp = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, 0, 0,
                                                  "3D forward rtemp workspace");
    if (aln_d > 1) {
        size_t rtemp_aligned = (size_t) sizertemp;
        size_t addend = (size_t) aln_d - 1;
        if (rtemp_aligned > ((size_t)-1) - addend)
            ABORT("Workspace size overflows allocation size.");
        rtemp_aligned = ((rtemp_aligned + addend) / (size_t) aln_d) * (size_t) aln_d;
        sizertemp = pdgstrs3d_checked_size_to_int_t(rtemp_aligned,
                                                    "3D forward rtemp workspace");
    }



/* skip rtemp on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    size_t rtemp_thread_count = pdgstrs3d_checked_product((size_t) sizertemp,
                                                          (size_t) num_thread,
                                                          "3D forward rtemp workspace");
    int_t rtemp_thread_count_i = pdgstrs3d_checked_size_to_int_t(rtemp_thread_count,
                                                                 "3D forward rtemp workspace");
    if (rtemp_thread_count > ((size_t)-1) - 1)
        ABORT("Workspace size overflows allocation size.");
    size_t rtemp_alloc_bytes = pdgstrs3d_checked_product(rtemp_thread_count + 1,
                                                         sizeof(double),
                                                         "3D forward rtemp workspace");
    if ( !(rtemp = (double*)SUPERLU_MALLOC(rtemp_alloc_bytes)) )
	ABORT("Malloc fails for rtemp[].");
#ifdef _OPENMP
#pragma omp parallel default(shared) private(ii)
    {
	int thread_id=omp_get_thread_num();
	for ( ii=0; ii<sizertemp; ii++ )
		rtemp[thread_id*sizertemp+ii]=zero;
    }
#else
    for ( ii=0; ii<rtemp_thread_count_i; ii++ )
	rtemp[ii]=zero;
#endif
}


    if ( !(stat_loc = (SuperLUStat_t**) SUPERLU_MALLOC(num_thread*sizeof(SuperLUStat_t*))) )
	ABORT("Malloc fails for stat_loc[].");

    for ( i=0; i<num_thread; i++) {
	stat_loc[i] = (SuperLUStat_t*)SUPERLU_MALLOC(sizeof(SuperLUStat_t));
	PStatInit(stat_loc[i]);
    }

    // /* Set up the headers in lsum[]. */
    // for (k = 0; k < nsupers; ++k) {
	// krow = PROW( k, grid );
	// if ( myrow == krow ) {
	//     lk = LBi( k, grid );   /* Local block number. */
	//     il = LSUM_BLK( lk );
	    // lsum[il - LSUM_H] = k; /* Block number prepended in the header. */
	// }
    // }

	/* ---------------------------------------------------------
	   Initialize the async Bcast trees on all processes.
	   --------------------------------------------------------- */
	nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */

	nbtree = 0;
	for (lk=0;lk<nsupers_j;++lk){
		if(LBtree_ptr[lk].empty_==NO){
			// printf("LBtree_ptr lk %5d\n",lk);
			if(C_BcTree_IsRoot(&LBtree_ptr[lk])==NO){
				nbtree++;
				if(LBtree_ptr[lk].destCnt_>0)nfrecvx_buf++;
			}
		}
	}



	nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	if ( !(	leafsups = (int_t*)intCalloc_dist(nsupers_i)) )
		ABORT("Calloc fails for leafsups.");

	nrtree = 0;
	nleaf=0;
	nfrecvmod=0;


	for (lk=0;lk<nsupers_j;++lk){
		if(LBtree_ptr[lk].empty_==NO){
            xtrsTimer->trsDataSendXY  += LBtree_ptr[lk].msgSize_*nrhs+XK_H;
		}
    }
	for (lk=0;lk<nsupers_i;++lk){
		if(LRtree_ptr[lk].empty_==NO){
            xtrsTimer->trsDataSendXY  += LRtree_ptr[lk].msgSize_*nrhs+LSUM_H;
		}
    }


    /* skip fmod,leafsups,nleaf on CPU if using GPU solve*/
    if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
        if(procs==1){
            for (lk=0;lk<nsupers_i;++lk){
                gb = myrow+lk*grid->nprow;  /* not sure */
                if(gb<nsupers){
                        if (fmod[lk*aln_i]==0 && supernodeMask[gb]){
                                leafsups[nleaf]=gb;
                                ++nleaf;
                        }
                }
            }
        }else{
            for (lk=0;lk<nsupers_i;++lk){
                if(LRtree_ptr[lk].empty_==NO){
                        nrtree++;
                        // RdTree_allocateRequest(LRtree_ptr[lk],'d');
                        frecv[lk] = LRtree_ptr[lk].destCnt_;
                        nfrecvmod += frecv[lk];
                }else{
                        gb = myrow+lk*grid->nprow;  /* not sure */
                        if(gb<nsupers){
                                kcol = PCOL( gb, grid );
                                if(mycol==kcol) { /* Diagonal process */
                                    /* skip fmod,leafsups,nleaf on CPU if using GPU solve*/
                                    if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
                                        if (fmod[lk*aln_i]==0 && supernodeMask[gb]){
                                                leafsups[nleaf]=gb;
                                                ++nleaf;
                                        }
                                    }
                                }
                        }
                }
            }
        }
    }else{
        if(procs>1){
            for (lk=0;lk<nsupers_i;++lk){
                if(LRtree_ptr[lk].empty_==NO){
                    nrtree++;
                    // RdTree_allocateRequest(LRtree_ptr[lk],'d');
                    gb = myrow+lk*grid->nprow;  /* not sure */
                    if (supernodeMask[gb]==1){
                        frecv[lk] = LRtree_ptr[lk].destCnt_;
                        nfrecvmod += frecv[lk];
                    }
                }
            }
        }
    }

/* skip fmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
	for (i = 0; i < nlb; ++i) fmod[i*aln_i] += frecv[i];
}
    size_t recvbuf_BC_count = pdgstrs3d_checked_product((size_t) maxrecvsz,
                                                        (size_t) nfrecvx + 1,
                                                        "3D forward recvbuf_BC");
    size_t recvbuf_BC_bytes = pdgstrs3d_checked_product(recvbuf_BC_count,
                                                        sizeof(double),
                                                        "3D forward recvbuf_BC");
	if ( !(recvbuf_BC_fwd = (double*)SUPERLU_MALLOC(recvbuf_BC_bytes)) )  // this needs to be optimized for 1D row mapping
		ABORT("Malloc fails for recvbuf_BC_fwd[].");
	nfrecvx_buf=0;

	log_memory(nlb*aln_i*iword+nlb*iword+(CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*2.0*iword+ nsupers_i*iword + sizelsum*num_thread * dword + (ldalsum * nrhs + nlb * XK_H) *dword + (sizertemp*num_thread + 1)*dword+maxrecvsz*(nfrecvx+1)*dword, stat);	//account for fmod, frecv, leaf_send, root_send, leafsups, recvbuf_BC_fwd	, lsum, x, rtemp


#if ( DEBUGlevel>=2 )
	printf("(%2d) nfrecvx %4d,  nfrecvmod %4d,  nleaf %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nfrecvx, nfrecvmod, nleaf, nbtree, nrtree);
	fflush(stdout);
#endif

// #if ( PRNTlevel>=1 )
#if 0
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Grid %3d: Setup L-solve time\t%8.4f\n", myGrid, t);
	fflush(stdout);
	MPI_Barrier( grid->comm );
	t = SuperLU_timer_();
#endif

#if ( VAMPIR>=1 )
	// VT_initialize();
	VT_traceon();
#endif

#ifdef USE_VTUNE
	__SSC_MARK(0x111);// start SDE tracing, note uses 2 underscores
	__itt_resume(); // start VTune, again use 2 underscores
#endif

	/* ---------------------------------------------------------
	   Solve the leaf nodes first by all the diagonal processes.
	   --------------------------------------------------------- */
#if ( DEBUGlevel>=2 )
	printf("(%2d) nleaf %4d\n", iam, nleaf);
	fflush(stdout);
#endif

	// ii = X_BLK( 0 );
	// knsupc = SuperSize( 0 );
	// for (i=0 ; i<knsupc*nrhs ; i++){
	    // printf("x_l: %f\n",x[ii+i]);
	// fflush(stdout);
	// }


#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK)

    if (get_acc_solve()) /* GPU trisolve*/
    {
// #if 0 /* CPU trisolve*/

// #if HAVE_CUDA
// cudaProfilerStart();
// #elif defined(HAVE_HIP)
// roctracer_mark("before HIP LaunchKernel");
// roctxMark("before hipLaunchKernel");
// roctxRangePush("hipLaunchKernel");
// #endif

#if ( PROFlevel>=1 )
    t = SuperLU_timer_();
#endif

    d_fmod=SOLVEstruct->d_fmod;
    d_lsum=SOLVEstruct->d_lsum;
	d_x=SOLVEstruct->d_x;
	d_grid=Llu->d_grid;

	checkGPU(gpuMemcpy(d_fmod, SOLVEstruct->d_fmod_save, nlb * sizeof(int), gpuMemcpyDeviceToDevice));
    checkGPU(gpuMemcpy(d_lsum, SOLVEstruct->d_lsum_save,
                       pdgstrs3d_checked_alloc_bytes(sizelsum, sizeof(double),
                                                     "3D forward lsum workspace"),
                       gpuMemcpyDeviceToDevice));
	checkGPU(gpuMemcpy(d_x, x,
                       pdgstrs3d_checked_alloc_bytes(x_count, sizeof(double),
                                                     "3D forward x workspace"),
                       gpuMemcpyHostToDevice));

	k = CEILING( nsupers, grid->npcol);/* Number of local block columns divided by #warps per block used as number of thread blocks*/
	knsupc = sp_ienv_dist(3, options);

    if(procs>1){ /* only nvshmem needs the following*/
    #ifdef HAVE_NVSHMEM
    checkGPU(gpuMemcpy(d_status, mystatus, k * sizeof(int), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_statusmod, mystatusmod, 2* nlb * sizeof(int), gpuMemcpyHostToDevice));
	//for(int i=0;i<2*nlb;i++) printf("(%d),mystatusmod[%d]=%d\n",iam,i,mystatusmod[i]);
	checkGPU(gpuMemset(flag_rd_q, 0, RDMA_FLAG_SIZE * nlb * 2 * sizeof(int)));
    checkGPU(gpuMemset(flag_bc_q, 0, RDMA_FLAG_SIZE * (k+1)  * sizeof(int)));
	checkGPU(gpuMemset(dready_x, 0, maxrecvsz*CEILING( nsupers, grid->npcol) * sizeof(double)));
    checkGPU(gpuMemset(dready_lsum, 0, 2*maxrecvsz*CEILING( nsupers, grid->nprow) * sizeof(double)));
    checkGPU(gpuMemset(d_msgnum, 0, h_nfrecv[1] * sizeof(int)));
	//printf("2-(%d) maxrecvsz=%d,dready_x=%d, dready_lsum=%d,RDMA_FLAG_SIZE=%d,k=%d,nlb=%d\n",iam,maxrecvsz,maxrecvsz*CEILING( nsupers, grid->npcol),2*maxrecvsz*CEILING( nsupers, grid->nprow),RDMA_FLAG_SIZE,k,nlb);
	//fflush(stdout);
    // MUST have this barrier, otherwise the code hang.
	MPI_Barrier( grid->comm );
    #endif
    }

	// k -> Llu->nbcol_masked ???
    int nblock_loc;
    if(procs==1){
        nblock_loc=Llu->nbcol_masked;
    }else{
        nblock_loc=k;
    }
	dlsum_fmod_inv_gpu_wrap(nblock_loc,nlb,DIM_X,DIM_Y,d_lsum,d_x,nrhs,knsupc,nsupers,d_fmod,Llu->d_LBtree_ptr,Llu->d_LRtree_ptr,Llu->d_ilsum,Llu->d_Lrowind_bc_dat, Llu->d_Lrowind_bc_offset, Llu->d_Lnzval_bc_dat, Llu->d_Lnzval_bc_offset, Llu->d_Linv_bc_dat, Llu->d_Linv_bc_offset, Llu->d_Lindval_loc_bc_dat, Llu->d_Lindval_loc_bc_offset,Llu->d_xsup,Llu->d_bcols_masked, d_grid,
                         maxrecvsz,
	                        flag_bc_q, flag_rd_q, dready_x, dready_lsum, my_flag_bc, my_flag_rd, d_nfrecv, h_nfrecv,
	                        d_status,d_colnum,d_mynum, d_mymaskstart,d_mymasklength,
	                        d_nfrecvmod,d_statusmod,d_colnummod,d_mynummod,d_mymaskstartmod,d_mymasklengthmod,d_recv_cnt,d_msgnum,d_flag_mod,procs);
	checkGPU(gpuMemcpy(x, d_x,
                       pdgstrs3d_checked_alloc_bytes(x_count, sizeof(double),
                                                     "3D forward x workspace"),
                       gpuMemcpyDeviceToHost));


#if ( PROFlevel>=1 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Grid %3d: around L kernel time\t%8.4f\n", myGrid, t);
#endif

	stat_loc[0]->ops[SOLVE]+=Llu->Lnzval_bc_cnt*nrhs*2; // YL: this is a rough estimate

    } else
    
#endif /* match #if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK) */
    { /* CPU trisolve */

tx = SuperLU_timer_();

#ifdef _OPENMP
#pragma omp parallel default (shared)
{
int thread_id = omp_get_thread_num();
#else
{
thread_id=0;
#endif
		{

            if (Llu->inv == 1) { /* Diagonal is inverted. */

#ifdef _OPENMP
#pragma	omp	for firstprivate(nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Linv,i,lib,rtemp_loc,nleaf_send_tmp) nowait
#endif
		for (jj=0;jj<nleaf;jj++){
		    k=leafsups[jj];

// #ifdef _OPENMP
// #pragma omp task firstprivate (k,nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,knsupc,lk,luptr,lsub,nsupr,lusup,thread_id,t1,t2,Linv,i,lib,rtemp_loc)
// #endif
   		    {

#if ( PROFlevel>=1 )
					TIC(t1);
#endif
					rtemp_loc = &rtemp[sizertemp* thread_id];


					knsupc = SuperSize( k );
					lk = LBi( k, grid );

					ii = X_BLK( lk );
					lk = LBj( k, grid ); /* Local block number, column-wise. */
					lsub = Lrowind_bc_ptr[lk];
					lusup = Lnzval_bc_ptr[lk];

					nsupr = lsub[1];

					Linv = Linv_bc_ptr[lk];
#ifdef _CRAY
					SGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
					dgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
					dgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc );
#endif

					for (i=0 ; i<knsupc*nrhs ; i++){
				        x[ii+i] = rtemp_loc[i];
					}
							// printf("\n");
							// printf("k: %5d\n",k);
					// for (i=0 ; i<knsupc*nrhs ; i++){
                        // printf("x_l: %f\n",x[ii+i]);
					// fflush(stdout);
					// }


#if ( PROFlevel>=1 )
					TOC(t2, t1);
					stat_loc[thread_id]->utime[SOL_TRSM] += t2;

#endif

			        stat_loc[thread_id]->ops[SOLVE] += knsupc * (knsupc - 1) * nrhs;



					// --nleaf;
#if ( DEBUGlevel>=2 )
					printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

					/*
					 * Send Xk to process column Pc[k].
					 */

					if(LBtree_ptr[lk].empty_==NO){
						lib = LBi( k, grid ); /* Local block number, row-wise. */
						ii = X_BLK( lib );

#ifdef _OPENMP
#pragma omp atomic capture
#endif
						nleaf_send_tmp = ++nleaf_send;
						leaf_send[(nleaf_send_tmp-1)*aln_i] = lk;
						// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],'d');
					}
				}
			}
	} else { /* Diagonal is not inverted. */
#ifdef _OPENMP
#pragma	omp	for firstprivate (nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Linv,i,lib,rtemp_loc,nleaf_send_tmp) nowait
#endif
	    for (jj=0;jj<nleaf;jj++) {
		k=leafsups[jj];
		{

#if ( PROFlevel>=1 )
		    TIC(t1);
#endif
		    rtemp_loc = &rtemp[sizertemp* thread_id];

		    knsupc = SuperSize( k );
		    lk = LBi( k, grid );

		    ii = X_BLK( lk );
		    lk = LBj( k, grid ); /* Local block number, column-wise. */
		    lsub = Lrowind_bc_ptr[lk];
		    lusup = Lnzval_bc_ptr[lk];

		    nsupr = lsub[1];

#ifdef _CRAY
   		    STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
				lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
		    dtrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
				lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
 		    dtrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
					lusup, &nsupr, &x[ii], &knsupc);
#endif

#if ( PROFlevel>=1 )
		    TOC(t2, t1);
		    stat_loc[thread_id]->utime[SOL_TRSM] += t2;

#endif

            stat_loc[thread_id]->ops[SOLVE] += knsupc * (knsupc - 1) * nrhs;

		    // --nleaf;
#if ( DEBUGlevel>=2 )
		    printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

		    /*
		     * Send Xk to process column Pc[k].
		     */

		    if (LBtree_ptr[lk].empty_==NO) {
			lib = LBi( k, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lib );

#ifdef _OPENMP
#pragma omp atomic capture
#endif
			nleaf_send_tmp = ++nleaf_send;
			leaf_send[(nleaf_send_tmp-1)*aln_i] = lk;
		    }
		    } /* end a block */
		} /* end for jj ... */
	    } /* end else ... diagonal is not invedted */
	  }
	} /* end omp parallel */

	jj=0;

#if ( DEBUGlevel>=2 )
	printf("(%2d) end solving nleaf %4d\n", iam, nleaf);
	fflush(stdout);
#endif

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
#else
	{
#endif

#ifdef _OPENMP
#pragma omp master
#endif
		    {

#ifdef _OPENMP
#if defined __GNUC__  && !defined __NVCOMPILER
#pragma	omp taskloop private (k,ii,lk,thread_id) num_tasks(num_thread*8) nogroup
#endif
#endif

			for (jj=0;jj<nleaf;jj++){
			    k=leafsups[jj];

			    {
#ifdef _OPENMP
				thread_id=omp_get_thread_num();
#else
				thread_id=0;
#endif

				/* Diagonal process */
				lk = LBi( k, grid );
				ii = X_BLK( lk );
				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				dlsum_fmod_inv(lsum, x, &x[ii], rtemp, nrhs, k, fmod, xsup, grid, Llu, stat_loc, leaf_send, &nleaf_send,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);
			    }

			} /* for jj ... */
		    }

		}

			for (i=0;i<nleaf_send;i++){
				lk = leaf_send[i*aln_i];
				if(lk>=0){ // this is a bcast forwarding
					gb = mycol+lk*grid->npcol;  /* not sure */
					lib = LBi( gb, grid ); /* Local block number, row-wise. */
					ii = X_BLK( lib );
					// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'d')*nrhs+XK_H,'d');
					C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H);

				}else{ // this is a reduce forwarding
					lk = -lk - 1;
					il = LSUM_BLK( lk );
					// RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(LRtree_ptr[lk],'d')*nrhs+LSUM_H,'d');
					C_RdTree_forwardMessageSimple(&LRtree_ptr[lk],&lsum[il - LSUM_H ],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H);
				}
			}
        xtrsTimer->tfs_compute += SuperLU_timer_() - tx;


#ifdef USE_VTUNE
	__itt_pause(); // stop VTune
	__SSC_MARK(0x222); // stop SDE tracing
#endif

			/* -----------------------------------------------------------
			   Compute the internal nodes asynchronously by all processes.
			   ----------------------------------------------------------- */

#ifdef _OPENMP
#pragma omp parallel default (shared)
			{
	int thread_id = omp_get_thread_num();
#else
	{
	thread_id=0;
#endif

#ifdef _OPENMP
#pragma omp master
#endif
				{
					for ( nfrecv =0; nfrecv<nfrecvx+nfrecvmod;nfrecv++) { /* While not finished. */
						thread_id = 0;
#if ( PROFlevel>=1 )
						TIC(t1);
						// msgcnt[1] = maxrecvsz;
#endif

						recvbuf0 = &recvbuf_BC_fwd[nfrecvx_buf*maxrecvsz];
                        double tx = SuperLU_timer_();
						/* Receive a message. */
						MPI_Recv( recvbuf0, maxrecvsz, MPI_DOUBLE,
								MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status );
                        xtrsTimer->tfs_comm += SuperLU_timer_() - tx;

						// MPI_Irecv(recvbuf0,maxrecvsz,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,grid->comm,&req);
						// ready=0;
						// while(ready==0){
						// MPI_Test(&req,&ready,&status);
						// #pragma omp taskyield
						// }

#if ( PROFlevel>=1 )
						TOC(t2, t1);
						stat_loc[thread_id]->utime[SOL_COMM] += t2;

						msg_cnt += 1;
						msg_vol += maxrecvsz * dword;
#endif

						{
                            double tx = SuperLU_timer_();
			                k = *recvbuf0;

#if ( DEBUGlevel>=2 )
							printf("(%2d) Recv'd block %d, tag %2d\n", iam, k, status.MPI_TAG);
#endif

							if(status.MPI_TAG==BC_L){
                                xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
								// --nfrecvx;
								nfrecvx_buf++;
								{
									lk = LBj( k, grid );    /* local block number */

									if(LBtree_ptr[lk].destCnt_>0){

										// BcTree_forwardMessageSimple(LBtree_ptr[lk],recvbuf0,BcTree_GetMsgSize(LBtree_ptr[lk],'d')*nrhs+XK_H,'d');
										C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], recvbuf0, LBtree_ptr[lk].msgSize_*nrhs+XK_H);
										// nfrecvx_buf++;
									}

									/*
									 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
									 */

									lk = LBj( k, grid ); /* Local block number, column-wise. */
									lsub = Lrowind_bc_ptr[lk];
									lusup = Lnzval_bc_ptr[lk];
									if ( lsub ) {
										krow = PROW( k, grid );
										if(myrow==krow){
											nb = lsub[0] - 1;
											knsupc = SuperSize( k );
											ii = X_BLK( LBi( k, grid ) );
											xin = &x[ii];
										}else{
											nb   = lsub[0];
											knsupc = SuperSize( k );
											xin = &recvbuf0[XK_H] ;
										}

										dlsum_fmod_inv_master(lsum, x, xin, rtemp, nrhs, knsupc, k,
												fmod, nb, xsup, grid, Llu,
												stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);

									} /* if lsub */
								}

							}else if(status.MPI_TAG==RD_L){
                                xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + LSUM_H;
								// --nfrecvmod;
								lk = LBi( k, grid ); /* Local block number, row-wise. */

								knsupc = SuperSize( k );
								tempv = &recvbuf0[LSUM_H];
								il = LSUM_BLK( lk );
								RHS_ITERATE(j) {
									for (i = 0; i < knsupc; ++i)
					                    lsum[i + il + j*knsupc + thread_id*sizelsum] += tempv[i + j*knsupc];

								}

								// #ifdef _OPENMP
								// #pragma omp atomic capture
								// #endif
								fmod_tmp=--fmod[lk*aln_i];
								{
									thread_id = 0;
									rtemp_loc = &rtemp[sizertemp* thread_id];
									if ( fmod_tmp==0 ) {
										if(C_RdTree_IsRoot(&LRtree_ptr[lk])==YES){
											// ii = X_BLK( lk );
											knsupc = SuperSize( k );
											for (ii=1;ii<num_thread;ii++)
												for (jj=0;jj<knsupc*nrhs;jj++)
						                            lsum[il + jj ] += lsum[il + jj + ii*sizelsum];


											ii = X_BLK( lk );
											RHS_ITERATE(j)
												for (i = 0; i < knsupc; ++i)
					                                x[i + ii + j*knsupc] += lsum[i + il + j*knsupc];

											// fmod[lk] = -1; /* Do not solve X[k] in the future. */
											lk = LBj( k, grid ); /* Local block number, column-wise. */
											lsub = Lrowind_bc_ptr[lk];
											lusup = Lnzval_bc_ptr[lk];
											nsupr = lsub[1];

#if ( PROFlevel>=1 )
											TIC(t1);
#endif

											if(Llu->inv == 1){
												Linv = Linv_bc_ptr[lk];
#ifdef _CRAY
												SGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
														&alpha, Linv, &knsupc, &x[ii],
														&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
												dgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
														&alpha, Linv, &knsupc, &x[ii],
														&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
												dgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
														&alpha, Linv, &knsupc, &x[ii],
														&knsupc, &beta, rtemp_loc, &knsupc );
#endif
												for (i=0 ; i<knsupc*nrhs ; i++){
				                                    x[ii+i] = rtemp_loc[i];
												}
											}
											else{
#ifdef _CRAY
												STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
														lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
												dtrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
														lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
												dtrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
														lusup, &nsupr, &x[ii], &knsupc);
#endif
											}

#if ( PROFlevel>=1 )
											TOC(t2, t1);
											stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif

			                                stat_loc[thread_id]->ops[SOLVE] += knsupc * (knsupc - 1) * nrhs;

#if ( DEBUGlevel>=2 )
											printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

											/*
											 * Send Xk to process column Pc[k].
											 */
											if(LBtree_ptr[lk].empty_==NO){
												// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'d')*nrhs+XK_H,'d');
												C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H);
											}


											/*
											 * Perform local block modifications.
											 */
											lk = LBj( k, grid ); /* Local block number, column-wise. */
											lsub = Lrowind_bc_ptr[lk];
											lusup = Lnzval_bc_ptr[lk];
											if ( lsub ) {
												krow = PROW( k, grid );
												nb = lsub[0] - 1;
												knsupc = SuperSize( k );
												ii = X_BLK( LBi( k, grid ) );
												xin = &x[ii];
												dlsum_fmod_inv_master(lsum, x, xin, rtemp, nrhs, knsupc, k,
														fmod, nb, xsup, grid, Llu,
														stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);
											} /* if lsub */
											// }

									}else{

										il = LSUM_BLK( lk );
										knsupc = SuperSize( k );

										for (ii=1;ii<num_thread;ii++)
											for (jj=0;jj<knsupc*nrhs;jj++)
                                                lsum[il + jj ] += lsum[il + jj + ii*sizelsum];
										// RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il-LSUM_H],RdTree_GetMsgSize(LRtree_ptr[lk],'d')*nrhs+LSUM_H,'d');
										C_RdTree_forwardMessageSimple(&LRtree_ptr[lk],&lsum[il - LSUM_H ],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H);
									}

								}

							}
						} /* check Tag */
					    xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
                    }

				} /* while not finished ... */

			}
		} // end of parallel
	}  /* end CPU trisolve */


// #if ( PRNTlevel>=1 )
#if 0
		t = SuperLU_timer_() - t;
		stat->utime[SOL_TOT] += t;
		// if ( !iam ) {
		// 	printf(".. L-solve time\t%8.4f\n", t);
		// 	fflush(stdout);
		// }


		MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE,
				MPI_MAX, 0, grid->comm);
		if ( !iam ) {
			printf(".. Grid %3d: L-solve time (MAX) \t%8.4f\n", myGrid, tmax);
			fflush(stdout);
		}


		t = SuperLU_timer_();
#endif


// stat->utime[SOLVE] = SuperLU_timer_() - t1_sol;

#if ( DEBUGlevel==2 )
		{
		  printf("(%d) .. After L-solve: y =\n", iam); fflush(stdout);
			for (i = 0, k = 0; k < nsupers; ++k) {
				krow = PROW( k, grid );
				kcol = PCOL( k, grid );
				if ( myrow == krow && mycol == kcol ) { /* Diagonal process */
					knsupc = SuperSize( k );
					lk = LBi( k, grid );
					ii = X_BLK( lk );
					for (j = 0; j < knsupc; ++j)
						printf("\t(%d)\t%4d\t%.10f\n", iam, xsup[k]+j, x[ii+j]);
					fflush(stdout);
				}
				MPI_Barrier( grid->comm );
			}
		}
#endif
/* skip fmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
		SUPERLU_FREE(fmod);
}
		SUPERLU_FREE(frecv);
		SUPERLU_FREE(leaf_send);
		SUPERLU_FREE(leafsups);
		SUPERLU_FREE(recvbuf_BC_fwd);
		log_memory(-nlb*aln_i*iword-nlb*iword-(CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*iword- nsupers_i*iword -maxrecvsz*(nfrecvx+1)*dword, stat);	//account for fmod, frecv, leaf_send, leafsups, recvbuf_BC_fwd

		for (lk=0;lk<nsupers_j;++lk){
			if(LBtree_ptr[lk].empty_==NO){
				// if(BcTree_IsRoot(LBtree_ptr[lk],'d')==YES){
				// BcTree_waitSendRequest(LBtree_ptr[lk],'d');
				C_BcTree_waitSendRequest(&LBtree_ptr[lk]);
				// }
				// deallocate requests here
			}
		}

		for (lk=0;lk<nsupers_i;++lk){
			if(LRtree_ptr[lk].empty_==NO){
				C_RdTree_waitSendRequest(&LRtree_ptr[lk]);
				// deallocate requests here
			}
		}
		// MPI_Barrier( grid->comm );

#if ( VAMPIR>=1 )
		VT_traceoff();
		VT_finalize();
#endif

		double tmp1=0;
		double tmp2=0;
		double tmp3=0;
		double tmp4=0;
		for(i=0;i<num_thread;i++){
			tmp1 = SUPERLU_MAX(tmp1,stat_loc[i]->utime[SOL_TRSM]);
			tmp2 = SUPERLU_MAX(tmp2,stat_loc[i]->utime[SOL_GEMM]);
			tmp3 = SUPERLU_MAX(tmp3,stat_loc[i]->utime[SOL_COMM]);
			tmp4 += stat_loc[i]->ops[SOLVE];
#if ( PRNTlevel>=2 )
			if(iam==0)printf("thread %5d gemm %9.5f\n",i,stat_loc[i]->utime[SOL_GEMM]);
#endif
		}


		stat->utime[SOL_TRSM] += tmp1;
		stat->utime[SOL_GEMM] += tmp2;
		stat->utime[SOL_COMM] += tmp3;
		stat->ops[SOLVE]+= tmp4;


		/* Deallocate storage. */
		for(i=0;i<num_thread;i++){
			PStatFree(stat_loc[i]);
			SUPERLU_FREE(stat_loc[i]);
		}
		SUPERLU_FREE(stat_loc);

/* skip rtemp on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
		SUPERLU_FREE(rtemp);
}
		// SUPERLU_FREE(lsum);
		// SUPERLU_FREE(x);

		// MPI_Barrier( grid->comm );


#if ( PROFlevel>=2 )
		{
			float msg_vol_max, msg_vol_sum, msg_cnt_max, msg_cnt_sum;

			MPI_Reduce (&msg_cnt, &msg_cnt_sum,
					1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
			MPI_Reduce (&msg_cnt, &msg_cnt_max,
					1, MPI_FLOAT, MPI_MAX, 0, grid->comm);
			MPI_Reduce (&msg_vol, &msg_vol_sum,
					1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
			MPI_Reduce (&msg_vol, &msg_vol_max,
					1, MPI_FLOAT, MPI_MAX, 0, grid->comm);
			if (!iam) {
				printf ("\tPDGSTRS comm stat:"
						"\tAvg\tMax\t\tAvg\tMax\n"
						"\t\t\tCount:\t%.0f\t%.0f\tVol(MB)\t%.2f\t%.2f\n",
						msg_cnt_sum / Pr / Pc, msg_cnt_max,
						msg_vol_sum / Pr / Pc * 1e-6, msg_vol_max * 1e-6);
			}
		}
#endif

    stat->utime[SOLVE] = SuperLU_timer_() - t1_sol;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit dForwardSolve3d_newsolve_reusepdgstrs()");
#endif


#if ( PRNTlevel>=2 )
	    float for_lu, total, max, avg, temp;
		superlu_dist_mem_usage_t num_mem_usage;

	    dQuerySpace_dist(n, LUstruct, grid, stat, &num_mem_usage);
	    temp = num_mem_usage.total;

	    MPI_Reduce( &temp, &max,
		       1, MPI_FLOAT, MPI_MAX, 0, grid->comm );
	    MPI_Reduce( &temp, &avg,
		       1, MPI_FLOAT, MPI_SUM, 0, grid->comm );
            if (!iam) {
		printf("\n** Memory Usage **********************************\n");
                printf("** Total highmark (MB):\n"
		       "    Sum-of-all : %8.2f | Avg : %8.2f  | Max : %8.2f\n",
		       avg * 1e-6,
		       avg / grid->nprow / grid->npcol * 1e-6,
		       max * 1e-6);
		printf("**************************************************\n");
		fflush(stdout);
            }
#endif

// cudaProfilerStop();

    return;
} /* dForwardSolve3d_newsolve_reusepdgstrs */







void dlsum_fmod_leaf_newsolve (
    dtrf3Dpartition_t*  trf3Dpartition,
    double *lsum,    /* Sum of local modifications.                        */
    double *x,       /* X array (local)                                    */
    double *xk,      /* X[k].                                              */
    double *rtemp,   /* Result of full matrix-vector multiply.             */
    int   nrhs,      /* Number of right-hand sides.                        */
    int   knsupc,    /* Size of supernode k.                               */
    int_t k,         /* The k-th component of X.                           */
    int *fmod,     /* Modification count for L-solve.                    */
    int_t nlb,       /* Number of L blocks.                                */
    int_t lptr,      /* Starting position in lsub[*].                      */
    int_t luptr,     /* Starting position in lusup[*].                     */
    int_t *xsup,
    gridinfo_t *grid,
    dLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat,xtrsTimer_t *xtrsTimer)

{
    double alpha = 1.0;
    double beta = 0.0;
	double *lusup, *lusup1;
	double *dest;
	int    iam, iknsupc, myrow, nbrow, nsupr, nsupr1, p, pi;
	int_t  i, ii, ik, il, ikcol, irow, j, lb, lk, lib, rel;
	int_t  *lsub, *lsub1, nlb1, lptr1, luptr1;
	int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
	int  *frecv = Llu->frecv;
	int  **fsendx_plist = Llu->fsendx_plist;
	MPI_Status status;
	int test_flag;

#if ( PROFlevel>=1 )
	double t1, t2;
	float msg_vol = 0, msg_cnt = 0;
#endif
#if ( PROFlevel>=1 )
	TIC(t1);
#endif

	iam = grid->iam;
	myrow = MYROW( iam, grid );
	lk = LBj( k, grid ); /* Local block number, column-wise. */
	lsub = Llu->Lrowind_bc_ptr[lk];
	lusup = Llu->Lnzval_bc_ptr[lk];
	nsupr = lsub[1];

	for (lb = 0; lb < nlb; ++lb)
	{
		ik = lsub[lptr]; /* Global block number, row-wise. */

        if (trf3Dpartition->supernodeMask[ik])
        {

		nbrow = lsub[lptr + 1];
#ifdef _CRAY
		SGEMM( ftcs2, ftcs2, &nbrow, &nrhs, &knsupc,
		       &alpha, &lusup[luptr], &nsupr, xk,
		       &knsupc, &beta, rtemp, &nbrow );
#elif defined (USE_VENDOR_BLAS)
		dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
		        &alpha, &lusup[luptr], &nsupr, xk,
		        &knsupc, &beta, rtemp, &nbrow, 1, 1 );
#else
		dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
		        &alpha, &lusup[luptr], &nsupr, xk,
		        &knsupc, &beta, rtemp, &nbrow );
#endif
		stat->ops[SOLVE] += 2 * nbrow * nrhs * knsupc + nbrow * nrhs;

		lk = LBi( ik, grid ); /* Local block number, row-wise. */
		iknsupc = SuperSize( ik );
		il = LSUM_BLK( lk );
		dest = &lsum[il];
		lptr += LB_DESCRIPTOR;
		rel = xsup[ik]; /* Global row index of block ik. */
		for (i = 0; i < nbrow; ++i)
		{
			irow = lsub[lptr++] - rel; /* Relative row. */
			RHS_ITERATE(j)
                dest[irow + j * iknsupc] -= rtemp[i + j * nbrow];
		}
		luptr += nbrow;

#if ( PROFlevel>=1 )
		TOC(t2, t1);
		stat->utime[SOL_GEMM] += t2;
#endif


		if ( (--fmod[lk]) == 0  )   /* Local accumulation done. */
		{
            ikcol = PCOL( ik, grid );
            p = PNUM( myrow, ikcol, grid );
            if ( iam != p )
            {
#ifdef ISEND_IRECV
                MPI_Isend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
                            MPI_DOUBLE, p, LSUM, grid->comm,
                            &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
                MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
                            MPI_DOUBLE, p, LSUM, grid->comm );
#else
                MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
                            MPI_DOUBLE, p, LSUM, grid->comm );
#endif
#endif
                xtrsTimer->trsDataSendXY += iknsupc * nrhs + LSUM_H;
            }
            else     /* Diagonal process: X[i] += lsum[i]. */
            {
                ii = X_BLK( lk );
                RHS_ITERATE(j)
                for (i = 0; i < iknsupc; ++i)
                    x[i + ii + j * iknsupc] += lsum[i + il + j * iknsupc];
                if ( frecv[lk] == 0 )   /* Becomes a leaf node. */
                {
                    fmod[lk] = -1; /* Do not solve X[k] in the future. */


                    lk = LBj( ik, grid );/* Local block number, column-wise. */
                    lsub1 = Llu->Lrowind_bc_ptr[lk];
                    lusup1 = Llu->Lnzval_bc_ptr[lk];
                    nsupr1 = lsub1[1];
#ifdef _CRAY
                    STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &iknsupc, &nrhs, &alpha,
                            lusup1, &nsupr1, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
                    dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
                            lusup1, &nsupr1, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
                    dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
                            lusup1, &nsupr1, &x[ii], &iknsupc);
#endif


                    stat->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;

                    /*
                        * Send Xk to process column Pc[k].
                        */
                    for (p = 0; p < grid->nprow; ++p)
                    {
                        if ( fsendx_plist[lk][p] != SLU_EMPTY )
                        {
                            pi = PNUM( p, ikcol, grid );
#ifdef ISEND_IRECV
                            MPI_Isend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
                                        MPI_DOUBLE, pi, Xk, grid->comm,
                                        &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
                            MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
                                        MPI_DOUBLE, pi, Xk, grid->comm );
#else
                            MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
                                        MPI_DOUBLE, pi, Xk, grid->comm );
#endif
#endif

                        }
                    }
                    xtrsTimer->trsDataSendXY += iknsupc * nrhs + XK_H;
                    /*
                        * Perform local block modifications.
                        */
                    nlb1 = lsub1[0] - 1;
                    lptr1 = BC_HEADER + LB_DESCRIPTOR + iknsupc;
                    luptr1 = iknsupc; /* Skip diagonal block L(I,I). */

                    dlsum_fmod_leaf_newsolve(trf3Dpartition,
                                    lsum, x, &x[ii], rtemp, nrhs, iknsupc, ik,
                                    fmod, nlb1, lptr1, luptr1, xsup,
                                    grid, Llu, send_req, stat,xtrsTimer);
                } /* if frecv[lk] == 0 */
            } /* if iam == p */
		}/* if fmod[lk] == 0 */

	    }
    }/* for lb ... */
} /* dLSUM_FMOD_LEAF */




int_t dlasum_bmod_Tree(int_t  pTree, int_t cTree, double *lsum, double *x,
                       dxT_struct *xT_s,
                       int    nrhs, dlsumBmod_buff_t* lbmod_buf,
                       dLUstruct_t * LUstruct,
                       dtrf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    sForest_t* pforest = trf3Dpartition->sForests[pTree];
    sForest_t* cforest = trf3Dpartition->sForests[cTree];
    if (!pforest || !cforest) return 0;

    int_t nnodes = pforest->nNodes;
    if (nnodes < 1) return 0;
    int_t* nodeList =  pforest->nodeList;
    int_t iam = grid->iam;
    int_t mycol = MYCOL( iam, grid );
    for (int_t k0 = 0; k0 < nnodes; ++k0)
    {
        /* code */
        int_t k = nodeList[k0];
        int_t kcol = PCOL (k, grid);
        if (mycol == kcol)
        {
            /* code */
            dlsumForestBsolve(k, cTree, lsum, x, xT_s, nrhs, lbmod_buf,
                             LUstruct, trf3Dpartition, grid3d, stat);
        }
    }
    return 0;
}


int_t dinitLsumBmod_buff(int_t ns, int nrhs, dlsumBmod_buff_t* lbmod_buf)
{
    lbmod_buf->tX = SUPERLU_MALLOC(ns * nrhs * sizeof(double));
    lbmod_buf->tU = SUPERLU_MALLOC(ns * ns * sizeof(double));
    lbmod_buf->indCols = SUPERLU_MALLOC(ns * sizeof(int_t));
    return 0;
}

int_t dfreeLsumBmod_buff(dlsumBmod_buff_t* lbmod_buf)
{
    SUPERLU_FREE(lbmod_buf->tX);
    SUPERLU_FREE(lbmod_buf->tU);
    SUPERLU_FREE(lbmod_buf->indCols);
    return 0;
}


int dpackUblock(int ldu, int_t* indCols,
                 int_t knsupc, int_t iklrow,  int_t* usub,
                 double* tempu, double* uval )
{
    double zero = 0.0;
    int ncols = 0;
    for (int_t jj = 0; jj < knsupc; ++jj)
    {

        int_t segsize = iklrow - usub[jj];
        if ( segsize )
        {
            int_t lead_zero = ldu - segsize;
            for (int_t i = 0; i < lead_zero; ++i) tempu[i] = zero;
            tempu += lead_zero;
            for (int_t i = 0; i < segsize; ++i)
            {
                tempu[i] = uval[i];
            }

            uval += segsize;
            tempu += segsize;
            indCols[ncols] = jj;
            ncols++;
        }

    } /* for jj ... */

    return ncols;
}


int_t dpackXbmod( int_t knsupc, int_t ncols, int_t nrhs, int_t* indCols, double* xk, double* tempx)
{

    for (int_t j = 0; j < nrhs; ++j)
    {
        double* dest = &tempx[j * ncols];
        double* y = &xk[j * knsupc];

        for (int_t jj = 0; jj < ncols; ++jj)
        {
            dest[jj] = y[indCols[jj]];
        } /* for jj ... */
    }

    return 0;
}

int_t dlsumBmod(int_t gik, int_t gjk, int nrhs, dlsumBmod_buff_t* lbmod_buf,
               int_t* usub,  double* uval,
               double* xk, double* lsum, int_t* xsup, SuperLUStat_t * stat)
{

    int_t* indCols = lbmod_buf->indCols;
    double* tempu = lbmod_buf->tU;
    double* tempx = lbmod_buf->tX;
    int iknsupc = (int)SuperSize( gik );
    int_t knsupc = SuperSize( gjk );
    int_t iklrow = FstBlockC( gik + 1 );
    int ldu = getldu(knsupc, iklrow,
                       usub // use &usub[i]
                      );

    int ncols = dpackUblock(ldu, indCols, knsupc, iklrow, usub,
                             tempu, uval );

    double alpha = -1.0;
    double beta = 1.0;
    double* X;

    if (ncols < knsupc)
    {
        /* code */
        dpackXbmod(knsupc, ncols, nrhs, indCols, xk, tempx);
        X = tempx;
    }
    else
    {
        X = xk;
    }

    double* V = &lsum[iknsupc - ldu];


#if defined (USE_VENDOR_BLAS)
	dgemm_("N", "N", &ldu, &nrhs, &ncols, &alpha,
	tempu, &ldu,
	X, &ncols, &beta, V, &iknsupc, 1, 1);
#else
	dgemm_("N", "N", &ldu, &nrhs, &ncols, &alpha,
	tempu, &ldu,
	X, &ncols, &beta, V, &iknsupc);
#endif





    stat->ops[SOLVE] += 2 * ldu * nrhs * ncols;
    return 0;
}

int_t dlsumForestBsolve(int_t k, int_t treeId,
                       double *lsum, double *x,  dxT_struct *xT_s, int    nrhs, dlsumBmod_buff_t* lbmod_buf,
                       dLUstruct_t * LUstruct,
                       dtrf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t iam = grid->iam;
    int_t myrow = MYROW( iam, grid );
    int_t knsupc = SuperSize( k );
    double *xT = xT_s->xT;
    int_t *ilsumT = xT_s->ilsumT;
    int_t ldaspaT = xT_s->ldaspaT;

    int_t lk = LBj( k, grid ); /* Local block number, column-wise. */
    int_t nub = Urbs[lk];      /* Number of U blocks in block column lk */
    int_t *ilsum = Llu->ilsum;
    int_t ii = XT_BLK (lk);
    double* xk = &xT[ii];
    for (int_t ub = 0; ub < nub; ++ub)
    {
        int_t ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
        int_t gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */

        if (trf3Dpartition->supernode2treeMap[gik] == treeId)
        {
            int_t* usub = Llu->Ufstnz_br_ptr[ik];
            double* uval = Llu->Unzval_br_ptr[ik];
            int_t i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
            i += UB_DESCRIPTOR;
            int_t il = LSUM_BLK( ik );
#if 1
            dlsumBmod(gik, k, nrhs, lbmod_buf,
                     &usub[i], &uval[Ucb_valptr[lk][ub]], xk,
                     &lsum[il], xsup, stat);
#else
            int_t iknsupc = SuperSize( gik );
            int_t ikfrow = FstBlockC( gik );
            int_t iklrow = FstBlockC( gik + 1 );

            for (int_t j = 0; j < nrhs; ++j)
            {
                double* dest = &lsum[il + j * iknsupc];
                double* y = &xk[j * knsupc];
                int_t uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
                for (int_t jj = 0; jj < knsupc; ++jj)
                {
                    int_t fnz = usub[i + jj];
                    if ( fnz < iklrow )
                    {
                        /* Nonzero segment. */
                        /* AXPY */
                        for (int_t irow = fnz; irow < iklrow; ++irow)
                            dest[irow - ikfrow] -= uval[uptr++] * y[jj];
                        stat->ops[SOLVE] += 2 * (iklrow - fnz);
                    }
                } /* for jj ... */
            } /*for (int_t j = 0;*/
#endif

        }

    }
    return 0;
}


int_t  dbCastXk2Pck  (int_t k, dxT_struct *xT_s, int nrhs,
                     dLUstruct_t * LUstruct, gridinfo_t * grid, xtrsTimer_t *xtrsTimer)
{
    /*
     * Send Xk to process column Pc[k].
     */

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t *ilsum = Llu->ilsum;
    int_t* xsup = Glu_persist->xsup;

    double *xT = xT_s->xT;
    int_t *ilsumT = xT_s->ilsumT;
    int_t ldaspaT = xT_s->ldaspaT;

    int_t lk = LBj (k, grid);
    int_t ii = XT_BLK (lk);
    double* xk = &xT[ii];
    superlu_scope_t *scp = &grid->cscp;
    int_t knsupc = SuperSize (k);
    int_t krow = PROW (k, grid);
    MPI_Bcast( xk, knsupc * nrhs, MPI_DOUBLE, krow,
               scp->comm);

    xtrsTimer->trsDataRecvXY  += knsupc * nrhs;
    xtrsTimer->trsDataSendXY  += knsupc * nrhs;
    return 0;
}

int_t  dlsumReducePrK (int_t k, double*x, double* lsum, double* recvbuf, int nrhs,
                      dLUstruct_t * LUstruct, gridinfo_t * grid, xtrsTimer_t *xtrsTimer)
{
    /*
     * Send Xk to process column Pc[k].
     */

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t *ilsum = Llu->ilsum;
    int_t* xsup = Glu_persist->xsup;

    int_t knsupc = SuperSize (k);
    int_t lk = LBi (k, grid);
    int_t iam = grid->iam;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);

    int_t ii = LSUM_BLK (lk);
    double* lsum_k = &lsum[ii];
    superlu_scope_t *scp = &grid->rscp;
    MPI_Reduce( lsum_k, recvbuf, knsupc * nrhs,
                MPI_DOUBLE, MPI_SUM, kcol, scp->comm);

    xtrsTimer->trsDataRecvXY  += knsupc * nrhs;
    xtrsTimer->trsDataSendXY  += knsupc * nrhs;

    if (mycol == kcol)
    {
        int_t ii = X_BLK( lk );
        double* dest = &x[ii];
        double* tempv = recvbuf;
        for (int_t j = 0; j < nrhs; ++j)
        {
            for (int_t i = 0; i < knsupc; ++i)
                x[i + ii + j * knsupc] += tempv[i + j * knsupc];
        }
    }

    return 0;
}

int_t dnonLeafForestBackSolve3d( int_t treeId,  dLUstruct_t * LUstruct,
                                dScalePermstruct_t * ScalePermstruct,
                                dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                                double * x, double * lsum,
                                dxT_struct *xT_s,
                                double * recvbuf,
                                MPI_Request * send_req,
                                int nrhs, dlsumBmod_buff_t* lbmod_buf,
                                dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    sForest_t** sForests = trf3Dpartition->sForests;

    sForest_t* sforest = sForests[treeId];
    if (!sforest)
    {
        /* code */
        return 0;
    }
    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1) return 1;
    int_t *perm_c_supno = sforest->nodeList ;
    gridinfo_t * grid = &(grid3d->grid2d);

    dLocalLU_t *Llu = LUstruct->Llu;
    int_t *ilsum = Llu->ilsum;

    int_t* xsup =  LUstruct->Glu_persist->xsup;

    double *xT = xT_s->xT;
    int_t *ilsumT = xT_s->ilsumT;
    int_t ldaspaT = xT_s->ldaspaT;

    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);

    for (int_t k0 = nnodes - 1; k0 >= 0; --k0)
    {
        int_t k = perm_c_supno[k0];
        int_t krow = PROW (k, grid);
        int_t kcol = PCOL (k, grid);
        // printf("doing %d \n", k);
        /**
         * Pkk(Yk) = sumOver_PrK (Yk)
         */
        if (myrow == krow )
        {
            double tx = SuperLU_timer_();
            dlsumReducePrK(k, x, lsum, recvbuf, nrhs, LUstruct, grid, xtrsTimer);
            xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
        }

        if (mycol == kcol )
        {
            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            int_t ii = X_BLK (lk);
            if (myrow == krow )
            {
                double tx = SuperLU_timer_();
                /* Diagonal process. */
                dlocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                int_t lkj = LBj (k, grid);
                int_t jj = XT_BLK (lkj);
                int_t knsupc = SuperSize(k);
                memcpy(&xT[jj], &x[ii], knsupc * nrhs * sizeof(double) );
                xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
            }                       /* if diagonal process ... */

            /*
             * Send Xk to process column Pc[k].
             */
            double tx = SuperLU_timer_();
            dbCastXk2Pck( k,  xT_s,  nrhs, LUstruct, grid,xtrsTimer);
            xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
            /*
             * Perform local block modifications: lsum[i] -= U_i,k * X[k]
             * where i is in current sforest
             */
            tx = SuperLU_timer_();
            dlsumForestBsolve(k, treeId, lsum, x, xT_s, nrhs, lbmod_buf,
                             LUstruct, trf3Dpartition, grid3d, stat);
            xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
        }
    }                           /* for k ... */
    return 0;
}



int_t dleafForestBackSolve3d(superlu_dist_options_t *options, int_t treeId, int_t n,  dLUstruct_t * LUstruct,
                            dScalePermstruct_t * ScalePermstruct,
                            dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                            double * x, double * lsum, double * recvbuf,
                            MPI_Request * send_req,
                            int nrhs, dlsumBmod_buff_t* lbmod_buf,
                            dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{

    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    sForest_t* sforest = trf3Dpartition->sForests[treeId];
    if (!sforest) return 0;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    // double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int_t *ilsum = Llu->ilsum;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t myGrid = grid3d->zscp.Iam;
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t knsupc = sp_ienv_dist (3,options);
    int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);

    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1) return 1;
    int_t *perm_c_supno = sforest->nodeList ;

    int **bsendx_plist = Llu->bsendx_plist;
    int_t Pr = grid->nprow;
    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t nlb = CEILING (nsupers, Pr);
    int* bmod =  getBmod3d(treeId, nlb, sforest, xsup, Llu->Ufstnz_br_ptr, trf3Dpartition->supernode2treeMap, grid);
    // for (int_t l=0;l<nsupers;l++)
        // printf("iam %5d lk %5d bmod %5d \n",grid->iam,l,bmod[l]);
    int* brecv = getBrecvTree(nlb, sforest, bmod, grid);
    Llu->brecv = brecv;

    int_t nbrecvmod = 0;
    int nroot = getNrootUsolveTree(&nbrecvmod, sforest, brecv, bmod, grid);
    int nbrecvx = getNbrecvX(sforest, Urbs, grid);
    // printf("igrid %5d, iam %5d, nbrecvx %5d, nbrecvmod %5d, nroot %5d\n",myGrid,iam,nbrecvx,nbrecvmod,nroot);

    /*before starting the solve; intialize the 3d lsum*/

    for (int_t k0 = nnodes - 1; k0 >= 0 ; --k0)
    {
        int_t k = perm_c_supno[k0];
        int_t krow = PROW (k, grid);
        int_t kcol = PCOL (k, grid);
        if (myrow == krow)
        {
            /* Diagonal process. */

            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            if (bmod[lk] == 0)
            {
                /* code */
                int_t il = LSUM_BLK( lk );
                int_t knsupc = SuperSize(k);
                if (mycol != kcol)
                {
                    /* code */
                    int_t p = PNUM( myrow, kcol, grid );
                    MPI_Isend( &lsum[il - LSUM_H], knsupc * nrhs + LSUM_H,
                               MPI_DOUBLE, p, LSUM, grid->comm,
                               &send_req[Llu->SolveMsgSent++] );
                    xtrsTimer->trsDataSendXY += knsupc * nrhs + LSUM_H;
                }
                else
                {
                    int_t ii = X_BLK( lk );
                    double* dest = &x[ii];
                    for (int_t j = 0; j < nrhs; ++j)
                        for (int_t i = 0; i < knsupc; ++i)
                            dest[i + j * knsupc] += lsum[i + il + j * knsupc];

                    if (brecv[lk] == 0 )
                    {
                        double tx = SuperLU_timer_();
                        bmod[lk] = -1;  /* Do not solve X[k] in the future. */

                        int_t ii = X_BLK (lk);
                        int_t lkj = LBj (k, grid); /* Local block number, column-wise */

                        dlocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                        --nroot;
                        /*
                         * Send Xk to process column Pc[k].
                         */
                        diBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, bsendx_plist, send_req, LUstruct, grid,xtrsTimer);
                        /*
                         * Perform local block modifications: lsum[i] -= U_i,k * X[k]
                         */
                        if (Urbs[lkj])
                            dlsum_bmod_GG (lsum, x, &x[ii], nrhs, lbmod_buf,  k, bmod, Urbs,
                                           Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                           send_req, stat,xtrsTimer);
                        xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
                    }                   /* if root ... */
                }
            }

        }                       /* if diagonal process ... */
    }                           /* for k ... */
    while (nbrecvx || nbrecvmod)
    {
        /* While not finished. */

        /* Receive a message. */
        MPI_Status status;
        double tx = SuperLU_timer_();
        MPI_Recv (recvbuf, maxrecvsz, MPI_DOUBLE,
                  MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status);
        xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
		int_t k = *recvbuf;

        tx = SuperLU_timer_();
        switch (status.MPI_TAG)
        {
        case Xk:
        {
            --nbrecvx;
            xtrsTimer->trsDataRecvXY += SuperSize(k)*nrhs + XK_H;
            /*
             * Perform local block modifications:
             *         lsum[i] -= U_i,k * X[k]
             */
            dlsum_bmod_GG (lsum, x, &recvbuf[XK_H], nrhs, lbmod_buf, k, bmod, Urbs,
                           Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                           send_req, stat,xtrsTimer);
            break;
        }
        case LSUM:             /* Receiver must be a diagonal process */
        {
            --nbrecvmod;
            xtrsTimer->trsDataRecvXY += SuperSize(k)*nrhs + LSUM_H;
            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            int_t ii = X_BLK (lk);
            int_t knsupc = SuperSize (k);
            double* tempv = &recvbuf[LSUM_H];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
					x[i + ii + j * knsupc] += tempv[i + j * knsupc];
            }

            if ((--brecv[lk]) == 0 && bmod[lk] == 0)
            {
                bmod[lk] = -1;  /* Do not solve X[k] in the future. */
                int_t lk = LBj (k, grid); /* Local block number, column-wise. */
                // int_t* lsub = Lrowind_bc_ptr[lk];
                dlocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                diBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, bsendx_plist, send_req, LUstruct, grid,xtrsTimer);
                if (Urbs[lk])
                    dlsum_bmod_GG (lsum, x, &x[ii], nrhs, lbmod_buf, k, bmod, Urbs,
                                   Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                   send_req, stat,xtrsTimer);
            }                   /* if becomes solvable */

            break;
        }
        }                       /* switch */
        xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
    }                           /* while not finished ... */

    double tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    SUPERLU_FREE(bmod);
    SUPERLU_FREE(brecv);
    Llu->SolveMsgSent = 0;
    xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
    return 0;

}



int_t dleafForestBackSolve3d_newsolve(superlu_dist_options_t *options, int_t n,  dLUstruct_t * LUstruct,
                            dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                            double * x, double * lsum, double * recvbuf,
                            MPI_Request * send_req,
                            int nrhs, dlsumBmod_buff_t* lbmod_buf,
                            dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{

    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    // sForest_t* sforest = trf3Dpartition->sForests[treeId];
    // if (!sforest) return 0;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    // double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int_t *ilsum = Llu->ilsum;
    int_t iam = grid->iam;
    int_t myGrid = grid3d->zscp.Iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);

    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t knsupc = sp_ienv_dist (3,options);
    int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);

    // int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    // if (nnodes < 1) return 1;
    // int_t *perm_c_supno = sforest->nodeList ;

    int **bsendx_plist = Llu->bsendx_plist;
    int_t Pr = grid->nprow;
    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t nlb = CEILING (nsupers, Pr);
    int* supernodeMask = trf3Dpartition->supernodeMask;


    int* bmod=  getBmod3d_newsolve(nlb, nsupers, supernodeMask, xsup, Llu->Ufstnz_br_ptr, grid);
    // for (int_t l=0;l<nsupers;l++)
    //     printf("iam %5d lk %5d bmod %5d \n",grid->iam,l,bmod[l]);
    int* brecv = getBrecvTree_newsolve(nlb, nsupers, supernodeMask, bmod, grid);
    Llu->brecv = brecv;

    int_t nbrecvmod = 0;
    int nroot= getNrootUsolveTree_newsolve(&nbrecvmod, nsupers, supernodeMask, brecv, bmod, grid);
    int nbrecvx= getNbrecvX_newsolve(nsupers, supernodeMask, Urbs, Ucb_indptr, grid);

    // printf("igrid %5d, iam %5d, nbrecvx %5d, nbrecvmod %5d, nroot %5d\n",myGrid,iam,nbrecvx,nbrecvmod,nroot);

    /*before starting the solve; intialize the 3d lsum*/

    for (int_t k = nsupers - 1; k >= 0 && nroot; --k)
    {
        if(supernodeMask[k]>0){
        int_t krow = PROW (k, grid);
        int_t kcol = PCOL (k, grid);
        if (myrow == krow && mycol == kcol)
        {
            /* Diagonal process. */

            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            if (bmod[lk] == 0)
            {
                /* code */
                int_t il = LSUM_BLK( lk );
                int_t knsupc = SuperSize(k);
                {
                int_t ii = X_BLK( lk );
                if (brecv[lk] == 0 )
                {
                    double tx = SuperLU_timer_();
                    bmod[lk] = -1;  /* Do not solve X[k] in the future. */

                    int_t ii = X_BLK (lk);
                    int_t lkj = LBj (k, grid); /* Local block number, column-wise */


                    // if(4327==k)
                    // for(int_t i=0;i<knsupc;i++)
                    // printf("before xk root: lk %5d, k %5d, x[ii] %15.6f iam %5d knsupc %5d tree %5d \n",lk,k,x[ii+i],grid3d->iam,knsupc,trf3Dpartition->supernode2treeMap[k]);


                    dlocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                    --nroot;

                    int_t knsupc = SuperSize(k);

                    // // // for(int_t i=0;i<knsupc;i++)
                    // if(4327==k)
                    // for(int_t i=0;i<knsupc;i++)
                    // printf("check xk root: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+i],grid3d->iam);


                    /*
                        * Send Xk to process column Pc[k].
                        */
                    diBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, bsendx_plist, send_req, LUstruct, grid,xtrsTimer);
                    /*
                        * Perform local block modifications: lsum[i] -= U_i,k * X[k]
                        */
                    if (Urbs[lkj])
                        dlsum_bmod_GG_newsolve (trf3Dpartition, lsum, x, &x[ii], nrhs, lbmod_buf,  k, bmod, Urbs,
                                        Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                        send_req, stat,xtrsTimer);
                    xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
                }                   /* if root ... */
                }
            }
        }                       /* if diagonal process ... */
        }                       /* if(supernodeMask[k]) */
    }                           /* for k ... */
    while (nbrecvx || nbrecvmod)
    {
        /* While not finished. */

        /* Receive a message. */
        MPI_Status status;
        double tx = SuperLU_timer_();
        MPI_Recv (recvbuf, maxrecvsz, MPI_DOUBLE,
                  MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status);
        xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
		int_t k = *recvbuf;

        tx = SuperLU_timer_();
        switch (status.MPI_TAG)
        {
        case Xk:
        {
            --nbrecvx;
            xtrsTimer->trsDataRecvXY += SuperSize(k)*nrhs + XK_H;
            /*
             * Perform local block modifications:
             *         lsum[i] -= U_i,k * X[k]
             */
            dlsum_bmod_GG_newsolve (trf3Dpartition, lsum, x, &recvbuf[XK_H], nrhs, lbmod_buf, k, bmod, Urbs,
                           Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                           send_req, stat,xtrsTimer);
            break;
        }
        case LSUM:             /* Receiver must be a diagonal process */
        {
            --nbrecvmod;
            xtrsTimer->trsDataRecvXY += SuperSize(k)*nrhs + LSUM_H;
            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            int_t ii = X_BLK (lk);
            int_t knsupc = SuperSize (k);
            double* tempv = &recvbuf[LSUM_H];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
					x[i + ii + j * knsupc] += tempv[i + j * knsupc];
            }

            if ((--brecv[lk]) == 0 && bmod[lk] == 0)
            {
                bmod[lk] = -1;  /* Do not solve X[k] in the future. */
                int_t lk = LBj (k, grid); /* Local block number, column-wise. */
                // int_t* lsub = Lrowind_bc_ptr[lk];
                dlocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                diBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, bsendx_plist, send_req, LUstruct, grid,xtrsTimer);
                if (Urbs[lk])
                    dlsum_bmod_GG_newsolve (trf3Dpartition, lsum, x, &x[ii], nrhs, lbmod_buf, k, bmod, Urbs,
                                   Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                   send_req, stat,xtrsTimer);
            }                   /* if becomes solvable */

            break;
        }
        }                       /* switch */
        xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
    }                           /* while not finished ... */

    double tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    Llu->SolveMsgSent = 0;
    xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
    SUPERLU_FREE(bmod);
    SUPERLU_FREE(brecv);
    return 0;

}





void dBackSolve3d_newsolve_reusepdgstrs(superlu_dist_options_t *options, int_t n,  dLUstruct_t * LUstruct,
                               int*  supernodeMask, gridinfo3d_t *grid3d,
                               double * x, double * lsum,
                               int nrhs,
                               dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    int_t myGrid = grid3d->zscp.Iam;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    double alpha = 1.0;
    double beta = 0.0;
    double zero = 0.0;
    double *lusup, *dest;
    double *recvbuf, *recvbuf_on, *tempv,
            *recvbufall, *recvbuf_BC_fwd, *recvbuf0, *xin, *recvbuf_BC_gpu,*recvbuf_RD_gpu;
    double *rtemp, *rtemp_loc; /* Result of full matrix-vector multiply. */
    double *Linv; /* Inverse of diagonal block */
    double *Uinv; /* Inverse of diagonal block */
    int *ipiv;
    int_t *leaf_send;
    int_t nleaf_send, nleaf_send_tmp;
    int_t *root_send;
    int_t nroot_send, nroot_send_tmp;
    int_t  **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
        /*-- Data structures used for broadcast and reduction trees. --*/
    C_Tree  *LBtree_ptr = Llu->LBtree_ptr;
    C_Tree  *LRtree_ptr = Llu->LRtree_ptr;
    C_Tree  *UBtree_ptr = Llu->UBtree_ptr;
    C_Tree  *URtree_ptr = Llu->URtree_ptr;
    int_t  *Urbs1; /* Number of row blocks in each block column of U. */
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t  kcol, krow, mycol, myrow;
    int_t  i, ii, il, j, jj, k, kk, lb, ljb, lk, lib, lptr, luptr, gb, nn;
    int_t  nb, nlb,nlb_nodiag, nub, nsupers, nsupers_j, nsupers_i,maxsuper;
    int_t  *xsup, *supno, *lsub, *usub;
    int_t  *ilsum;    /* Starting position of each supernode in lsum (LOCAL)*/
    int    Pc, Pr, iam;
    int    knsupc, nsupr, nprobe;
    int    nbtree, nrtree, outcount;
    int    ldalsum;   /* Number of lsum entries locally owned. */
    int    maxrecvsz, p, pi;
    int_t  **Lrowind_bc_ptr;
    double **Lnzval_bc_ptr;
    double **Linv_bc_ptr;
    double **Uinv_bc_ptr;
    double sum;
    MPI_Status status,status_on,statusx,statuslsum;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
    SuperLUStat_t **stat_loc;

    double tmax;
    	/*-- Counts used for L-solve --*/
    int  *fmod;         /* Modification count for L-solve --
    			 Count the number of local block products to
    			 be summed into lsum[lk]. */
	int_t *fmod_sort;
	int_t *order;
	//int_t *order1;
	//int_t *order2;
    int fmod_tmp;
    int  **fsendx_plist = Llu->fsendx_plist;
    int  nfrecvx_buf=0;
    int *frecv;        /* Count of lsum[lk] contributions to be received
    			 from processes in this row.
    			 It is only valid on the diagonal processes. */
    int  frecv_tmp;
    int  nfrecvmod = 0; /* Count of total modifications to be recv'd. */
    int  nfrecv = 0; /* Count of total messages to be recv'd. */
    int  nbrecv = 0; /* Count of total messages to be recv'd. */
    int  nleaf = 0, nroot = 0;
    int  nleaftmp = 0, nroottmp = 0;
    int_t  msgsize;
        /*-- Counts used for U-solve --*/
    int  *bmod;         /* Modification count for U-solve. */
    int  bmod_tmp;
    int  **bsendx_plist = Llu->bsendx_plist;
    int  nbrecvx = Llu->nbrecvx; /* Number of X components to be recv'd. */
    int  nbrecvx_buf=0;
    int  *brecv;        /* Count of modifications to be recv'd from
    			 processes in this row. */
    int_t  nbrecvmod = 0; /* Count of total modifications to be recv'd. */
    int_t flagx,flaglsum,flag;
    int_t *LBTree_active, *LRTree_active, *LBTree_finish, *LRTree_finish, *leafsups, *rootsups;
    int_t TAG;
    double t1_sol, t2_sol, t;
#if ( DEBUGlevel>=2 )
    int_t Ublocks = 0;
#endif

    int_t gik,iklrow,fnz;

    int *mod_bit = Llu->mod_bit; /* flag contribution from each row block */
    int INFO, pad;
    int_t tmpresult;

    // #if ( PROFlevel>=1 )
    double t1, t2, t3;
    float msg_vol = 0, msg_cnt = 0;
    // #endif

    int_t msgcnt[4]; /* Count the size of the message xfer'd in each buffer:
		      *     0 : transferred in Lsub_buf[]
		      *     1 : transferred in Lval_buf[]
		      *     2 : transferred in Usub_buf[]
		      *     3 : transferred in Uval_buf[]
		      */
    int iword = sizeof (int_t);
    int dword = sizeof (double);
    int Nwork;
    int_t procs = grid->nprow * grid->npcol;
    yes_no_t done;
    yes_no_t startforward;
    int nbrow;
    int_t  ik, rel, idx_r, jb, nrbl, irow, pc,iknsupc;
    int_t  lptr1_tmp, idx_i, idx_v,m;
    int_t ready;
    int thread_id = 0;
    yes_no_t empty;
    int_t sizelsum,sizertemp,aln_d,aln_i;
    aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    int num_thread = 1;
	int_t cnt1,cnt2;
    double tx;

#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK)

	const int nwrp_block = 1; /* number of warps in each block */
	const int warp_size = 32; /* number of threads per warp*/
	gpuStream_t sid=0;
	int gid=0;
	gridinfo_t *d_grid = NULL;
	double *d_x = NULL;
	double *d_lsum = NULL;
    int  *d_bmod = NULL;
#endif


// cudaProfilerStart();
    maxsuper = sp_ienv_dist(3, options);

#ifdef _OPENMP
#pragma omp parallel default(shared)
    {
    	if (omp_get_thread_num () == 0) {
    		num_thread = omp_get_num_threads ();
    	}
    }
#else
	num_thread=1;
#endif

    // MPI_Barrier( grid->comm );
    t1_sol = SuperLU_timer_();
    t = SuperLU_timer_();


    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW( iam, grid );
    mycol = MYCOL( iam, grid );
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = supno[n-1] + 1;
    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    Linv_bc_ptr = Llu->Linv_bc_ptr;
    Uinv_bc_ptr = Llu->Uinv_bc_ptr;
    nlb = CEILING( nsupers, Pr ); /* Number of local block rows. */

    // stat->utime[SOL_COMM] = 0.0;
    // stat->utime[SOL_GEMM] = 0.0;
    // stat->utime[SOL_TRSM] = 0.0;
    // stat->utime[SOL_TOT] = 0.0;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter dBackSolve3d_newsolve_reusepdgstrs()");
#endif

    // stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    /* Save the count to be altered so it can be used by
       subsequent call to PDGSTRS. */

    if ( !(root_send = intMalloc_dist((CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i)) )
	ABORT("Malloc fails for root_send[].");
    nroot_send=0;

#ifdef _CRAY
    ftcs1 = _cptofcd("L", strlen("L"));
    ftcs2 = _cptofcd("N", strlen("N"));
    ftcs3 = _cptofcd("U", strlen("U"));
#endif


    /* Obtain ilsum[] and ldalsum for process column 0. */
    ilsum = Llu->ilsum;
    ldalsum = Llu->ldalsum;

    /* Allocate working storage. */
    knsupc = sp_ienv_dist(3, options);
    maxrecvsz = pdgstrs3d_checked_workspace_count(knsupc, nrhs, 1,
                                                  SUPERLU_MAX( XK_H, LSUM_H ),
                                                  "3D back recvbuf");
    sizelsum = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb, LSUM_H,
                                                 "3D back lsum workspace");
    if (aln_d > 1) {
        size_t lsum_aligned = (size_t) sizelsum;
        size_t addend = (size_t) aln_d - 1;
        if (lsum_aligned > ((size_t)-1) - addend)
            ABORT("Workspace size overflows allocation size.");
        lsum_aligned = ((lsum_aligned + addend) / (size_t) aln_d) * (size_t) aln_d;
        sizelsum = pdgstrs3d_checked_size_to_int_t(lsum_aligned,
                                                   "3D back lsum workspace");
    }
    int_t x_count = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb,
                                                      XK_H,
                                                      "3D back x workspace");
    sizertemp = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, 0, 0,
                                                  "3D back rtemp workspace");
    if (aln_d > 1) {
        size_t rtemp_aligned = (size_t) sizertemp;
        size_t addend = (size_t) aln_d - 1;
        if (rtemp_aligned > ((size_t)-1) - addend)
            ABORT("Workspace size overflows allocation size.");
        rtemp_aligned = ((rtemp_aligned + addend) / (size_t) aln_d) * (size_t) aln_d;
        sizertemp = pdgstrs3d_checked_size_to_int_t(rtemp_aligned,
                                                    "3D back rtemp workspace");
    }

/* skip rtemp on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    size_t rtemp_thread_count = pdgstrs3d_checked_product((size_t) sizertemp,
                                                          (size_t) num_thread,
                                                          "3D back rtemp workspace");
    int_t rtemp_thread_count_i = pdgstrs3d_checked_size_to_int_t(rtemp_thread_count,
                                                                 "3D back rtemp workspace");
    if (rtemp_thread_count > ((size_t)-1) - 1)
        ABORT("Workspace size overflows allocation size.");
    size_t rtemp_alloc_bytes = pdgstrs3d_checked_product(rtemp_thread_count + 1,
                                                         sizeof(double),
                                                         "3D back rtemp workspace");
    if ( !(rtemp = (double*)SUPERLU_MALLOC(rtemp_alloc_bytes)) )
	ABORT("Malloc fails for rtemp[].");
#ifdef _OPENMP
#pragma omp parallel default(shared) private(ii)
    {
	int thread_id=omp_get_thread_num();
	for ( ii=0; ii<sizertemp; ii++ )
		rtemp[thread_id*sizertemp+ii]=zero;
    }
#else
    for ( ii=0; ii<rtemp_thread_count_i; ii++ )
	rtemp[ii]=zero;
#endif
}

    if ( !(stat_loc = (SuperLUStat_t**) SUPERLU_MALLOC(num_thread*sizeof(SuperLUStat_t*))) )
	ABORT("Malloc fails for stat_loc[].");

    for ( i=0; i<num_thread; i++) {
	stat_loc[i] = (SuperLUStat_t*)SUPERLU_MALLOC(sizeof(SuperLUStat_t));
	PStatInit(stat_loc[i]);
    }


	/* ---------------------------------------------------------
	   Initialize the async Bcast trees on all processes.
	   --------------------------------------------------------- */
	nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */


/* skip bmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    bmod=  getBmod3d_newsolve(nlb, nsupers, supernodeMask, xsup, Llu->Ufstnz_br_ptr, grid);
}
    nbrecvx= getNbrecvX_newsolve(nsupers, supernodeMask, Urbs, Ucb_indptr, grid);


		/* Save the count to be altered so it can be used by
		   subsequent call to PDGSTRS. */
		if ( !(brecv = int32Calloc_dist(nlb)) )
			ABORT("Calloc fails for brecv[].");
		Llu->brecv = brecv;

		/* Re-initialize lsum to zero. Each block header is already in place. */


/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
#ifdef _OPENMP
#pragma omp parallel default(shared) private(ii)
	{
		int thread_id = omp_get_thread_num();
		for(ii=0;ii<sizelsum;ii++)
			lsum[thread_id*sizelsum+ii]=zero;
	}
    /* Set up the headers in lsum[]. */
    for (k = 0; k < nsupers; ++k) {
	krow = PROW( k, grid );
	if ( myrow == krow ) {
	    lk = LBi( k, grid );   /* Local block number. */
	    il = LSUM_BLK( lk );
	    lsum[il - LSUM_H] = k; /* Block number prepended in the header. */
	}
    }

#else
	for (k = 0; k < nsupers; ++k) {
		krow = PROW( k, grid );
		if ( myrow == krow ) {
			knsupc = SuperSize( k );
			lk = LBi( k, grid );
			il = LSUM_BLK( lk );
			dest = &lsum[il];

			for (jj = 0; jj < num_thread; ++jj) {
				RHS_ITERATE(j) {
					for (i = 0; i < knsupc; ++i) dest[i + j*knsupc + jj*sizelsum] = zero;
				}
			}
		}
	}
#endif
}


#if ( DEBUGlevel>=2 )
		for (p = 0; p < Pr*Pc; ++p) {
			if (iam == p) {
				printf("(%2d) .. Ublocks %d\n", iam, Ublocks);
				for (lb = 0; lb < nub; ++lb) {
					printf("(%2d) Local col %2d: # row blocks %2d\n",
							iam, lb, Urbs[lb]);
					if ( Urbs[lb] ) {
						for (i = 0; i < Urbs[lb]; ++i)
							printf("(%2d) .. row blk %2d:\
									lbnum %d, indpos %d, valpos %d\n",
									iam, i,
									Ucb_indptr[lb][i].lbnum,
									Ucb_indptr[lb][i].indpos,
									Ucb_valptr[lb][i]);
					}
				}
			}
			MPI_Barrier( grid->comm );
		}
		for (p = 0; p < Pr*Pc; ++p) {
			if ( iam == p ) {
				printf("\n(%d) bsendx_plist[][]", iam);
				for (lb = 0; lb < nub; ++lb) {
					printf("\n(%d) .. local col %2d: ", iam, lb);
					for (i = 0; i < Pr; ++i)
						printf("%4d", bsendx_plist[lb][i]);
				}
				printf("\n");
			}
			MPI_Barrier( grid->comm );
		}
#endif /* DEBUGlevel */


	/* ---------------------------------------------------------
	   Initialize the async Bcast trees on all processes.
	   --------------------------------------------------------- */
	nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */

	nbtree = 0;
	for (lk=0;lk<nsupers_j;++lk){
		if(UBtree_ptr[lk].empty_==NO){
			// printf("UBtree_ptr lk %5d\n",lk);
			if(C_BcTree_IsRoot(&UBtree_ptr[lk])==NO){
				nbtree++;
				if(UBtree_ptr[lk].destCnt_>0)nbrecvx_buf++;
			}
			// BcTree_allocateRequest(UBtree_ptr[lk],'d');
		}
	}

	nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	if ( !(	rootsups = (int_t*)intCalloc_dist(nsupers_i)) )
		ABORT("Calloc fails for rootsups.");


	for (lk=0;lk<nsupers_j;++lk){
		if(UBtree_ptr[lk].empty_==NO){
            xtrsTimer->trsDataSendXY  += UBtree_ptr[lk].msgSize_*nrhs+XK_H;
		}
    }
	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk].empty_==NO){
            xtrsTimer->trsDataSendXY  += URtree_ptr[lk].msgSize_*nrhs+LSUM_H;
		}
    }

/* skip bmod/rootsups/nroot on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
	nrtree = 0;
	nroot=0;
	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk].empty_==NO){
			// printf("here lk %5d myid %5d\n",lk,iam);
			// fflush(stdout);
			nrtree++;
			// RdTree_allocateRequest(URtree_ptr[lk],'d');
			brecv[lk] = URtree_ptr[lk].destCnt_;
			nbrecvmod += brecv[lk];
		}else{
			gb = myrow+lk*grid->nprow;  /* not sure */
			if(gb<nsupers){
				kcol = PCOL( gb, grid );
				if(mycol==kcol) { /* Diagonal process */
					if (bmod[lk*aln_i]==0 && supernodeMask[gb]>0){
						rootsups[nroot]=gb;
						++nroot;
					}
                }
            }
        }
    }
}else{
	nrtree = 0;
	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk].empty_==NO){
			// printf("here lk %5d myid %5d\n",lk,iam);
			// fflush(stdout);
			nrtree++;
			// RdTree_allocateRequest(URtree_ptr[lk],'d');
			brecv[lk] = URtree_ptr[lk].destCnt_;
			nbrecvmod += brecv[lk];
		}
    }
}

/* skip bmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
	for (i = 0; i < nlb; ++i) bmod[i*aln_i] += brecv[i];
	// for (i = 0; i < nlb; ++i)printf("bmod[i]: %5d\n",bmod[i]);
}

    size_t recvbuf_BC_count = pdgstrs3d_checked_product((size_t) maxrecvsz,
                                                        (size_t) nbrecvx + 1,
                                                        "3D back recvbuf_BC");
    size_t recvbuf_BC_bytes = pdgstrs3d_checked_product(recvbuf_BC_count,
                                                        sizeof(double),
                                                        "3D back recvbuf_BC");
	if ( !(recvbuf_BC_fwd = (double*)SUPERLU_MALLOC(recvbuf_BC_bytes)) )  // this needs to be optimized for 1D row mapping
		ABORT("Malloc fails for recvbuf_BC_fwd[].");
	nbrecvx_buf=0;

	log_memory(nlb*aln_i*iword+nlb*iword + nsupers_i*iword + maxrecvsz*(nbrecvx+1)*dword, stat);	//account for bmod, brecv, rootsups, recvbuf_BC_fwd

#if ( DEBUGlevel>=2 )
	printf("(%2d) nbrecvx %4d,  nbrecvmod %4d,  nroot %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nbrecvx, nbrecvmod, nroot, nbtree, nrtree);
	fflush(stdout);
#endif

// #if ( PRNTlevel>=1 )
#if 0
	t = SuperLU_timer_() - t;
    if ( !iam) printf(".. Grid %3d: Setup U-solve time\t%8.4f\n", myGrid, t);
	fflush(stdout);
	MPI_Barrier( grid->comm );
	t = SuperLU_timer_();
#endif

		/*
		 * Solve the roots first by all the diagonal processes.
		 */
#if ( DEBUGlevel>=2 )
		printf("(%2d) nroot %4d\n", iam, nroot);
		fflush(stdout);
#endif





if (get_acc_solve()){  /* GPU trisolve*/
#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK)
// #if 0 /* CPU trisolve*/


#if ( PROFlevel>=1 )
    t = SuperLU_timer_();
#endif

    d_bmod=SOLVEstruct->d_bmod;
    d_lsum=SOLVEstruct->d_lsum;
	d_x=SOLVEstruct->d_x;
	d_grid=Llu->d_grid;

	checkGPU(gpuMemcpy(d_bmod, SOLVEstruct->d_bmod_save, nlb * sizeof(int), gpuMemcpyDeviceToDevice));
    checkGPU(gpuMemcpy(d_lsum, SOLVEstruct->d_lsum_save,
                       pdgstrs3d_checked_alloc_bytes(sizelsum, sizeof(double),
                                                     "3D back lsum workspace"),
                       gpuMemcpyDeviceToDevice));
    checkGPU(gpuMemcpy(d_x, x,
                       pdgstrs3d_checked_alloc_bytes(x_count, sizeof(double),
                                                     "3D back x workspace"),
                       gpuMemcpyHostToDevice));

	k = CEILING( nsupers, grid->npcol);/* Number of local block columns divided by #warps per block used as number of thread blocks*/
	knsupc = sp_ienv_dist(3, options);

    if(procs>1){ /* only nvshmem needs the following*/
    #ifdef HAVE_NVSHMEM
    checkGPU(gpuMemcpy(d_status, mystatus_u, k * sizeof(int), gpuMemcpyHostToDevice));
    checkGPU(gpuMemcpy(d_statusmod, mystatusmod_u, 2* nlb * sizeof(int), gpuMemcpyHostToDevice));
    //for(int i=0;i<2*nlb;i++) printf("(%d),mystatusmod[%d]=%d\n",iam,i,mystatusmod[i]);
    checkGPU(gpuMemset(flag_rd_q, 0, RDMA_FLAG_SIZE * nlb * 2 * sizeof(int)));
    checkGPU(gpuMemset(flag_bc_q, 0, RDMA_FLAG_SIZE * (k+1)  * sizeof(int)));
    checkGPU(gpuMemset(dready_x, 0, maxrecvsz*CEILING( nsupers, grid->npcol) * sizeof(double)));
    checkGPU(gpuMemset(dready_lsum, 0, 2*maxrecvsz*CEILING( nsupers, grid->nprow) * sizeof(double)));
    checkGPU(gpuMemset(d_msgnum, 0, h_nfrecv_u[1] * sizeof(int)));
    // MUST have this barrier, otherwise the code hang.
	MPI_Barrier( grid->comm );
    #endif
    }

    dlsum_bmod_inv_gpu_wrap(options, k,nlb,DIM_X,DIM_Y,d_lsum,d_x,nrhs,knsupc,nsupers,d_bmod,
                        Llu->d_UBtree_ptr,Llu->d_URtree_ptr,
                        Llu->d_ilsum,Llu->d_Ucolind_bc_dat,Llu->d_Ucolind_bc_offset,Llu->d_Ucolind_br_dat,Llu->d_Ucolind_br_offset,
                        Llu->d_Uind_br_dat,Llu->d_Uind_br_offset,
                        Llu->d_Unzval_bc_dat,Llu->d_Unzval_bc_offset,Llu->d_Unzval_br_new_dat,Llu->d_Unzval_br_new_offset,
                        Llu->d_Uinv_bc_dat,Llu->d_Uinv_bc_offset,
                        Llu->d_Uindval_loc_bc_dat,Llu->d_Uindval_loc_bc_offset,
                        Llu->d_xsup,d_grid,
                        maxrecvsz, flag_bc_q, flag_rd_q, dready_x, dready_lsum,
                        my_flag_bc, my_flag_rd,
                        d_nfrecv_u, h_nfrecv_u, d_status, d_colnum_u, d_mynum_u,
                        d_mymaskstart_u,d_mymasklength_u,
                        d_nfrecvmod_u, d_statusmod, d_colnummod_u, d_mynummod_u,
                        d_mymaskstartmod_u, d_mymasklengthmod_u,
                        d_recv_cnt_u, d_msgnum, d_flag_mod_u, procs);


    checkGPU(gpuMemcpy(x, d_x,
                       pdgstrs3d_checked_alloc_bytes(x_count, sizeof(double),
                                                     "3D back x workspace"),
                       gpuMemcpyDeviceToHost));


#if ( PROFlevel>=1 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Grid %3d: around U kernel time\t%8.4f\n", myGrid, t);
#endif

	stat_loc[0]->ops[SOLVE]+=Llu->Unzval_br_cnt*nrhs*2; // YL: this is a rough estimate
#endif
}else{  /* CPU trisolve*/


tx = SuperLU_timer_();

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
#else
	{
#endif
#ifdef _OPENMP
#pragma omp master
#endif
		{
#ifdef _OPENMP
#if defined __GNUC__  && !defined __NVCOMPILER
#pragma	omp	taskloop firstprivate (nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,jj,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Uinv,i,lib,rtemp_loc,nroot_send_tmp,thread_id) nogroup
#endif
#endif
		for (jj=0;jj<nroot;jj++){
			k=rootsups[jj];

#if ( PROFlevel>=1 )
			TIC(t1);
#endif
#ifdef _OPENMP
			thread_id=omp_get_thread_num();
#else
			thread_id=0;
#endif

			rtemp_loc = &rtemp[sizertemp* thread_id];



			knsupc = SuperSize( k );
			lk = LBi( k, grid ); /* Local block number, row-wise. */

			// bmod[lk] = -1;       /* Do not solve X[k] in the future. */
			ii = X_BLK( lk );
			lk = LBj( k, grid ); /* Local block number, column-wise */
			lsub = Lrowind_bc_ptr[lk];
			lusup = Lnzval_bc_ptr[lk];
			nsupr = lsub[1];


			if(Llu->inv == 1){

				Uinv = Uinv_bc_ptr[lk];
#ifdef _CRAY
				SGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
				dgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
				dgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#endif
				for (i=0 ; i<knsupc*nrhs ; i++){
				    x[ii+i] = rtemp_loc[i];
				}
			}else{
#ifdef _CRAY
				STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
				dtrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
				dtrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc);
#endif
			}

#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif
			stat_loc[thread_id]->ops[SOLVE] += knsupc * (knsupc + 1) * nrhs;

#if ( DEBUGlevel>=2 )
			printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

			/*
			 * Send Xk to process column Pc[k].
			 */

			if(UBtree_ptr[lk].empty_==NO){
#ifdef _OPENMP
#pragma omp atomic capture
#endif
				nroot_send_tmp = ++nroot_send;
				root_send[(nroot_send_tmp-1)*aln_i] = lk;

			}
		} /* for k ... */
	}
}


#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
#else
	{
#endif
#ifdef _OPENMP
#pragma omp master
#endif
		{
#ifdef _OPENMP
#if defined __GNUC__  && !defined __NVCOMPILER
#pragma	omp	taskloop private (ii,jj,k,lk,thread_id) nogroup
#endif
#endif
		for (jj=0;jj<nroot;jj++){
			k=rootsups[jj];
			lk = LBi( k, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lk );
			lk = LBj( k, grid ); /* Local block number, column-wise */
#ifdef _OPENMP
			thread_id=omp_get_thread_num();
#else
			thread_id=0;
#endif
			/*
			 * Perform local block modifications: lsum[i] -= U_i,k * X[k]
			 */
			if ( Urbs[lk] )
				dlsum_bmod_inv(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
						Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
						stat_loc, root_send, &nroot_send, sizelsum,sizertemp,thread_id,num_thread);

		} /* for k ... */

	}
}

for (i=0;i<nroot_send;i++){
	lk = root_send[(i)*aln_i];
	if(lk>=0){ // this is a bcast forwarding
		gb = mycol+lk*grid->npcol;  /* not sure */
		lib = LBi( gb, grid ); /* Local block number, row-wise. */
		ii = X_BLK( lib );
		// BcTree_forwardMessageSimple(UBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk],'d')*nrhs+XK_H,'d');
		C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], &x[ii - XK_H], UBtree_ptr[lk].msgSize_*nrhs+XK_H);
	}else{ // this is a reduce forwarding
		lk = -lk - 1;
		il = LSUM_BLK( lk );
		// RdTree_forwardMessageSimple(URtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(URtree_ptr[lk],'d')*nrhs+LSUM_H,'d');
		C_RdTree_forwardMessageSimple(&URtree_ptr[lk],&lsum[il - LSUM_H ],URtree_ptr[lk].msgSize_*nrhs+LSUM_H);
	}
}
xtrsTimer->tbs_compute += SuperLU_timer_() - tx;

		/*
		 * Compute the internal nodes asychronously by all processes.
		 */

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
	int thread_id=omp_get_thread_num();
#else
	{
	thread_id=0;
#endif
#ifdef _OPENMP
#pragma omp master
#endif
		for ( nbrecv =0; nbrecv<nbrecvx+nbrecvmod;nbrecv++) { /* While not finished. */

			// printf("iam %4d nbrecv %4d nbrecvx %4d nbrecvmod %4d\n", iam, nbrecv, nbrecvxnbrecvmod);
			// fflush(stdout);



			thread_id = 0;
#if ( PROFlevel>=1 )
			TIC(t1);
#endif

			recvbuf0 = &recvbuf_BC_fwd[nbrecvx_buf*maxrecvsz];
            double tx = SuperLU_timer_();
			/* Receive a message. */
			MPI_Recv( recvbuf0, maxrecvsz, MPI_DOUBLE,
					MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status );
            xtrsTimer->tbs_comm += SuperLU_timer_() - tx;

#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat_loc[thread_id]->utime[SOL_COMM] += t2;

			msg_cnt += 1;
			msg_vol += maxrecvsz * dword;
#endif

			k = *recvbuf0;
#if ( DEBUGlevel>=2 )
			printf("(%2d) Recv'd block %d, tag %2d\n", iam, k, status.MPI_TAG);
			fflush(stdout);
#endif
            tx = SuperLU_timer_();
			if(status.MPI_TAG==BC_U){
                xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
				// --nfrecvx;
				nbrecvx_buf++;

				lk = LBj( k, grid );    /* local block number */

				if(UBtree_ptr[lk].destCnt_>0){

					// BcTree_forwardMessageSimple(UBtree_ptr[lk],recvbuf0,BcTree_GetMsgSize(UBtree_ptr[lk],'d')*nrhs+XK_H,'d');
					C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], recvbuf0, UBtree_ptr[lk].msgSize_*nrhs+XK_H);
					// nfrecvx_buf++;
				}

				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */

				lk = LBj( k, grid ); /* Local block number, column-wise. */
				dlsum_bmod_inv_master(lsum, x, &recvbuf0[XK_H], rtemp, nrhs, k, bmod, Urbs,
						Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
						stat_loc, sizelsum,sizertemp,thread_id,num_thread);
			}else if(status.MPI_TAG==RD_U){
                xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + LSUM_H;
				lk = LBi( k, grid ); /* Local block number, row-wise. */

				knsupc = SuperSize( k );
				tempv = &recvbuf0[LSUM_H];
				il = LSUM_BLK( lk );
				RHS_ITERATE(j) {
					for (i = 0; i < knsupc; ++i)
					    lsum[i + il + j*knsupc + thread_id*sizelsum] += tempv[i + j*knsupc];

				}
			// #ifdef _OPENMP
			// #pragma omp atomic capture
			// #endif
				bmod_tmp=--bmod[lk*aln_i];
				thread_id = 0;
				rtemp_loc = &rtemp[sizertemp* thread_id];
				if ( bmod_tmp==0 ) {
					if(C_RdTree_IsRoot(&URtree_ptr[lk])==YES){

						knsupc = SuperSize( k );
						for (ii=1;ii<num_thread;ii++)
							for (jj=0;jj<knsupc*nrhs;jj++)
					            lsum[il+ jj ] += lsum[il + jj + ii*sizelsum];


						ii = X_BLK( lk );
						RHS_ITERATE(j)
							for (i = 0; i < knsupc; ++i)
					            x[i + ii + j*knsupc] += lsum[i + il + j*knsupc];

						lk = LBj( k, grid ); /* Local block number, column-wise. */
						lsub = Lrowind_bc_ptr[lk];
						lusup = Lnzval_bc_ptr[lk];
						nsupr = lsub[1];

						if(Llu->inv == 1){

							Uinv = Uinv_bc_ptr[lk];

#ifdef _CRAY
							SGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
									&alpha, Uinv, &knsupc, &x[ii],
									&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
							dgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
									&alpha, Uinv, &knsupc, &x[ii],
									&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
							dgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
									&alpha, Uinv, &knsupc, &x[ii],
									&knsupc, &beta, rtemp_loc, &knsupc );
#endif

							for (i=0 ; i<knsupc*nrhs ; i++){
				                x[ii+i] = rtemp_loc[i];
							}
						}else{
#ifdef _CRAY
							STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
							dtrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
							dtrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &knsupc);
#endif
						}

#if ( PROFlevel>=1 )
							TOC(t2, t1);
							stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif
							stat_loc[thread_id]->ops[SOLVE] += knsupc * (knsupc + 1) * nrhs;

#if ( DEBUGlevel>=2 )
						printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

						/*
						 * Send Xk to process column Pc[k].
						 */
						if(UBtree_ptr[lk].empty_==NO){
							// BcTree_forwardMessageSimple(UBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk],'d')*nrhs+XK_H,'d');
							C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], &x[ii - XK_H], UBtree_ptr[lk].msgSize_*nrhs+XK_H);
						}


						/*
						 * Perform local block modifications:
						 *         lsum[i] -= U_i,k * X[k]
						 */
						if ( Urbs[lk] )
							dlsum_bmod_inv_master(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
									Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
									stat_loc, sizelsum,sizertemp,thread_id,num_thread);

					}else{
						il = LSUM_BLK( lk );
						knsupc = SuperSize( k );

						for (ii=1;ii<num_thread;ii++)
							for (jj=0;jj<knsupc*nrhs;jj++)
					            lsum[il+ jj ] += lsum[il + jj + ii*sizelsum];

						// RdTree_forwardMessageSimple(URtree_ptr[lk],&lsum[il-LSUM_H],RdTree_GetMsgSize(URtree_ptr[lk],'d')*nrhs+LSUM_H,'d');
						C_RdTree_forwardMessageSimple(&URtree_ptr[lk],&lsum[il - LSUM_H ],URtree_ptr[lk].msgSize_*nrhs+LSUM_H);
					}

				}
			}
            xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
		} /* while not finished ... */
	}

    }

// #if ( PRNTlevel>=1 )
#if 0
		t = SuperLU_timer_() - t;
		stat->utime[SOL_TOT] += t;
		// if ( !iam ) printf(".. U-solve time\t%8.4f\n", t);
		MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE,
				MPI_MAX, 0, grid->comm);
		if ( !iam ) {
			printf(".. Grid %3d: U-solve time (MAX) \t%8.4f\n", myGrid, tmax);
			fflush(stdout);
		}
		t = SuperLU_timer_();
#endif


#if ( DEBUGlevel>=2 )
		{
			double *x_col;
			int diag;
			printf("\n(%d) .. After U-solve: x (ON DIAG PROCS) = \n", iam);
			ii = 0;
			for (k = 0; k < nsupers; ++k) {
				knsupc = SuperSize( k );
				krow = PROW( k, grid );
				kcol = PCOL( k, grid );
				diag = PNUM( krow, kcol, grid);
				if ( iam == diag ) { /* Diagonal process. */
					lk = LBi( k, grid );
					jj = X_BLK( lk );
					x_col = &x[jj];
					RHS_ITERATE(j) {
						for (i = 0; i < knsupc; ++i) { /* X stored in blocks */
							printf("\t(%d)\t%4d\t%.10f\n",
									iam, xsup[k]+i, x_col[i]);
						}
						x_col += knsupc;
					}
				}
				ii += knsupc;
			} /* for k ... */
		}
#endif





		double tmp1=0;
		double tmp2=0;
		double tmp3=0;
		double tmp4=0;
		for(i=0;i<num_thread;i++){
			tmp1 = SUPERLU_MAX(tmp1,stat_loc[i]->utime[SOL_TRSM]);
			tmp2 = SUPERLU_MAX(tmp2,stat_loc[i]->utime[SOL_GEMM]);
			tmp3 = SUPERLU_MAX(tmp3,stat_loc[i]->utime[SOL_COMM]);
			tmp4 += stat_loc[i]->ops[SOLVE];
#if ( PRNTlevel>=2 )
			if(iam==0)printf("thread %5d gemm %9.5f\n",i,stat_loc[i]->utime[SOL_GEMM]);
#endif
		}


		stat->utime[SOL_TRSM] += tmp1;
		stat->utime[SOL_GEMM] += tmp2;
		stat->utime[SOL_COMM] += tmp3;
		stat->ops[SOLVE]+= tmp4;


		/* Deallocate storage. */
		for(i=0;i<num_thread;i++){
			PStatFree(stat_loc[i]);
			SUPERLU_FREE(stat_loc[i]);
		}
		SUPERLU_FREE(stat_loc);
/* skip rtemp on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
		SUPERLU_FREE(rtemp);
}

/* skip bmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
		SUPERLU_FREE(bmod);
}
		SUPERLU_FREE(brecv);
		SUPERLU_FREE(root_send);

		SUPERLU_FREE(rootsups);
		SUPERLU_FREE(recvbuf_BC_fwd);

		log_memory(-nlb*aln_i*iword-nlb*iword - nsupers_i*iword - (CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*iword - maxrecvsz*(nbrecvx+1)*dword - sizelsum*num_thread * dword - (ldalsum * nrhs + nlb * XK_H) *dword - (sizertemp*num_thread + 1)*dword, stat);	//account for bmod, brecv, root_send, rootsups, recvbuf_BC_fwd,rtemp,lsum,x

		for (lk=0;lk<nsupers_j;++lk){
			if(UBtree_ptr[lk].empty_==NO){
				// if(BcTree_IsRoot(LBtree_ptr[lk],'d')==YES){
				C_BcTree_waitSendRequest(&UBtree_ptr[lk]);
				// }
				// deallocate requests here
			}
		}

		for (lk=0;lk<nsupers_i;++lk){
			if(URtree_ptr[lk].empty_==NO){
				C_RdTree_waitSendRequest(&URtree_ptr[lk]);
				// deallocate requests here
			}
		}
		// MPI_Barrier( grid->comm );


#if ( PROFlevel>=2 )
		{
			float msg_vol_max, msg_vol_sum, msg_cnt_max, msg_cnt_sum;

			MPI_Reduce (&msg_cnt, &msg_cnt_sum,
					1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
			MPI_Reduce (&msg_cnt, &msg_cnt_max,
					1, MPI_FLOAT, MPI_MAX, 0, grid->comm);
			MPI_Reduce (&msg_vol, &msg_vol_sum,
					1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
			MPI_Reduce (&msg_vol, &msg_vol_max,
					1, MPI_FLOAT, MPI_MAX, 0, grid->comm);
			if (!iam) {
				printf ("\tPDGSTRS comm stat:"
						"\tAvg\tMax\t\tAvg\tMax\n"
						"\t\t\tCount:\t%.0f\t%.0f\tVol(MB)\t%.2f\t%.2f\n",
						msg_cnt_sum / Pr / Pc, msg_cnt_max,
						msg_vol_sum / Pr / Pc * 1e-6, msg_vol_max * 1e-6);
			}
		}
#endif

    stat->utime[SOLVE] = SuperLU_timer_() - t1_sol;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit dBackSolve3d_newsolve_reusepdgstrs()");
#endif


#if ( PRNTlevel>=2 )
	    float for_lu, total, max, avg, temp;
		superlu_dist_mem_usage_t num_mem_usage;

	    dQuerySpace_dist(n, LUstruct, grid, stat, &num_mem_usage);
	    temp = num_mem_usage.total;

	    MPI_Reduce( &temp, &max,
		       1, MPI_FLOAT, MPI_MAX, 0, grid->comm );
	    MPI_Reduce( &temp, &avg,
		       1, MPI_FLOAT, MPI_SUM, 0, grid->comm );
            if (!iam) {
		printf("\n** Memory Usage **********************************\n");
                printf("** Total highmark (MB):\n"
		       "    Sum-of-all : %8.2f | Avg : %8.2f  | Max : %8.2f\n",
		       avg * 1e-6,
		       avg / grid->nprow / grid->npcol * 1e-6,
		       max * 1e-6);
		printf("**************************************************\n");
		fflush(stdout);
            }
#endif

// cudaProfilerStop();

    return;
} /* dBackSolve3d_newsolve_reusepdgstrs */

/************************************************************************/

/************************************************************************/
void dlsum_bmod_GG (
    double *lsum,        /* Sum of local modifications.                    */
    double *x,           /* X array (local).                               */
    double *xk,          /* X[k].                                          */
    int    nrhs,          /* Number of right-hand sides.                    */
    dlsumBmod_buff_t* lbmod_buf,
    int_t  k,            /* The k-th component of X.                       */
    int  *bmod,        /* Modification count for L-solve.                */
    int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
    Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
    int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
    int_t  *xsup,
    gridinfo_t *grid,
    dLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat
    , xtrsTimer_t *xtrsTimer)
{
    // printf("bmodding %d\n", k);
    /*
     * Purpose
     * =======
     *   Perform local block modifications: lsum[i] -= U_i,k * X[k].
     */
    double alpha = 1.0;
    double beta = 0.0;
    int    iam, iknsupc, knsupc, myrow, nsupr, p, pi;
    int_t  fnz, gik, gikcol, i, ii, ik, ikfrow, iklrow, il, irow,
           j, jj, lk, lk1, nub, ub, uptr;
    int_t  *usub;
    double *uval, *dest, *y;
    int_t  *lsub;
    double *lusup;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int  *brecv = Llu->brecv;
    int  **bsendx_plist = Llu->bsendx_plist;
    MPI_Status status;
    int test_flag;

    iam = grid->iam;
    myrow = MYROW( iam, grid );
    knsupc = SuperSize( k );
    lk = LBj( k, grid ); /* Local block number, column-wise. */
    nub = Urbs[lk];      /* Number of U blocks in block column lk */

    for (ub = 0; ub < nub; ++ub)
    {
        ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
        usub = Llu->Ufstnz_br_ptr[ik];
        uval = Llu->Unzval_br_ptr[ik];
        i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
        i += UB_DESCRIPTOR;
        il = LSUM_BLK( ik );
        gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
        iknsupc = SuperSize( gik );
#if 1
        dlsumBmod(gik, k, nrhs, lbmod_buf,
                 &usub[i], &uval[Ucb_valptr[lk][ub]], xk,
                 &lsum[il], xsup, stat);
#else

        ikfrow = FstBlockC( gik );
        iklrow = FstBlockC( gik + 1 );

        for (int_t j = 0; j < nrhs; ++j)
        {
            dest = &lsum[il + j * iknsupc];
            y = &xk[j * knsupc];
            uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
            for (jj = 0; jj < knsupc; ++jj)
            {
                fnz = usub[i + jj];
                if ( fnz < iklrow )   /* Nonzero segment. */
                {
                    /* AXPY */
                    for (irow = fnz; irow < iklrow; ++irow)
                        dest[irow - ikfrow] -= uval[uptr++] * y[jj];
                    stat->ops[SOLVE] += 2 * (iklrow - fnz);
                }
            } /* for jj ... */
        } /*for (int_t j = 0;*/
#endif
        // printf(" updating %d  %d  \n",ik, bmod[ik] );
        if ( (--bmod[ik]) == 0 )   /* Local accumulation done. */
        {
            // printf("Local accumulation done %d  %d, brecv[ik]=%d  ",ik, bmod[ik],brecv[ik] );
            gikcol = PCOL( gik, grid );
            p = PNUM( myrow, gikcol, grid );
            if ( iam != p )
            {
#ifdef ISEND_IRECV
                MPI_Isend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
                           MPI_DOUBLE, p, LSUM, grid->comm,
                           &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
                MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
                           MPI_DOUBLE, p, LSUM, grid->comm );
#else
                MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
                          MPI_DOUBLE, p, LSUM, grid->comm );
#endif
#endif
#if ( DEBUGlevel>=2 )
                printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
                       iam, lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H, p);
#endif
                xtrsTimer->trsDataSendXY += iknsupc * nrhs + LSUM_H;
            }
            else     /* Diagonal process: X[i] += lsum[i]. */
            {
                ii = X_BLK( ik );
                dest = &x[ii];
                for (int_t j = 0; j < nrhs; ++j)
                    for (i = 0; i < iknsupc; ++i)
                        dest[i + j * iknsupc] += lsum[i + il + j * iknsupc];
                if ( !brecv[ik] )   /* Becomes a leaf node. */
                {
                    bmod[ik] = -1; /* Do not solve X[k] in the future. */
                    lk1 = LBj( gik, grid ); /* Local block number. */
                    lsub = Llu->Lrowind_bc_ptr[lk1];
                    lusup = Llu->Lnzval_bc_ptr[lk1];
                    nsupr = lsub[1];
#ifdef _CRAY
                    STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &iknsupc, &nrhs, &alpha,
                          lusup, &nsupr, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
                    dtrsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
                           lusup, &nsupr, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
                    dtrsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
                           lusup, &nsupr, &x[ii], &iknsupc);
#endif
                    stat->ops[SOLVE] += iknsupc * (iknsupc + 1) * nrhs;
#if ( DEBUGlevel>=2 )
                    printf("(%2d) Solve X[%2d]\n", iam, gik);
#endif

                    /*
                     * Send Xk to process column Pc[k].
                     */
                    for (p = 0; p < grid->nprow; ++p)
                    {
                        if ( bsendx_plist[lk1][p] != SLU_EMPTY )
                        {
                            pi = PNUM( p, gikcol, grid );
#ifdef ISEND_IRECV
                            MPI_Isend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
                                       MPI_DOUBLE, pi, Xk, grid->comm,
                                       &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
                            MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
                                       MPI_DOUBLE, pi, Xk, grid->comm );
#else
                            MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
                                      MPI_DOUBLE, pi, Xk, grid->comm );
#endif
#endif
#if ( DEBUGlevel>=2 )
                            printf("(%2d) Sent X[%2.0f] to P %2d\n",
                                   iam, x[ii - XK_H], pi);
#endif
                        }
                    }
                    xtrsTimer->trsDataSendXY += iknsupc * nrhs + XK_H;
                    /*
                     * Perform local block modifications.
                     */
                    if ( Urbs[lk1] )
                        dlsum_bmod_GG(lsum, x, &x[ii], nrhs, lbmod_buf, gik, bmod, Urbs,
                                      Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                      send_req, stat,xtrsTimer);
                } /* if brecv[ik] == 0 */
            }
        } /* if bmod[ik] == 0 */

    } /* for ub ... */

} /* dlSUM_BMOD */




/************************************************************************/
void dlsum_bmod_GG_newsolve (
    dtrf3Dpartition_t*  trf3Dpartition,
    double *lsum,        /* Sum of local modifications.                    */
    double *x,           /* X array (local).                               */
    double *xk,          /* X[k].                                          */
    int    nrhs,          /* Number of right-hand sides.                    */
    dlsumBmod_buff_t* lbmod_buf,
    int_t  k,            /* The k-th component of X.                       */
    int  *bmod,        /* Modification count for L-solve.                */
    int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
    Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
    int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
    int_t  *xsup,
    gridinfo_t *grid,
    dLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat
    , xtrsTimer_t *xtrsTimer)
{
    // printf("bmodding %d\n", k);
    /*
     * Purpose
     * =======
     *   Perform local block modifications: lsum[i] -= U_i,k * X[k].
     */
    double alpha = 1.0;
    double beta = 0.0;
    int    iam, iknsupc, knsupc, myrow, nsupr, p, pi;
    int_t  fnz, gik, gikcol, i, ii, ik, ikfrow, iklrow, il, irow,
           j, jj, lk, lk1, nub, ub, uptr;
    int_t  *usub;
    double *uval, *dest, *y;
    int_t  *lsub;
    double *lusup;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int  *brecv = Llu->brecv;
    int  **bsendx_plist = Llu->bsendx_plist;
    MPI_Status status;
    int test_flag;

    iam = grid->iam;
    myrow = MYROW( iam, grid );
    knsupc = SuperSize( k );
    lk = LBj( k, grid ); /* Local block number, column-wise. */
    nub = Urbs[lk];      /* Number of U blocks in block column lk */

    for (ub = 0; ub < nub; ++ub)
    {
        ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
        gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
        if (trf3Dpartition->supernodeMask[gik]>0)
        {
        usub = Llu->Ufstnz_br_ptr[ik];
        uval = Llu->Unzval_br_ptr[ik];
        i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
        i += UB_DESCRIPTOR;
        il = LSUM_BLK( ik );
        iknsupc = SuperSize( gik );
#if 1
        dlsumBmod(gik, k, nrhs, lbmod_buf,
                 &usub[i], &uval[Ucb_valptr[lk][ub]], xk,
                 &lsum[il], xsup, stat);
#else

        ikfrow = FstBlockC( gik );
        iklrow = FstBlockC( gik + 1 );

        for (int_t j = 0; j < nrhs; ++j)
        {
            dest = &lsum[il + j * iknsupc];
            y = &xk[j * knsupc];
            uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
            for (jj = 0; jj < knsupc; ++jj)
            {
                fnz = usub[i + jj];
                if ( fnz < iklrow )   /* Nonzero segment. */
                {
                    /* AXPY */
                    for (irow = fnz; irow < iklrow; ++irow)
                        dest[irow - ikfrow] -= uval[uptr++] * y[jj];
                    stat->ops[SOLVE] += 2 * (iklrow - fnz);
                }
            } /* for jj ... */
        } /*for (int_t j = 0;*/
#endif
        // printf(" updating %d  %d  \n",ik, bmod[ik] );
        if ( (--bmod[ik]) == 0 )   /* Local accumulation done. */
        {
            // printf("Local accumulation done %d  %d, brecv[ik]=%d  ",ik, bmod[ik],brecv[ik] );
            gikcol = PCOL( gik, grid );
            p = PNUM( myrow, gikcol, grid );
            if ( iam != p )
            {
#ifdef ISEND_IRECV
                MPI_Isend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
                           MPI_DOUBLE, p, LSUM, grid->comm,
                           &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
                MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
                           MPI_DOUBLE, p, LSUM, grid->comm );
#else
                MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
                          MPI_DOUBLE, p, LSUM, grid->comm );
#endif
#endif
#if ( DEBUGlevel>=2 )
                printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
                       iam, lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H, p);
#endif
                xtrsTimer->trsDataSendXY += iknsupc * nrhs + LSUM_H;
            }
            else     /* Diagonal process: X[i] += lsum[i]. */
            {
                ii = X_BLK( ik );
                dest = &x[ii];
                for (int_t j = 0; j < nrhs; ++j)
                    for (i = 0; i < iknsupc; ++i)
                    dest[i + j * iknsupc] += lsum[i + il + j * iknsupc];

                if ( !brecv[ik] )   /* Becomes a leaf node. */
                {
                    bmod[ik] = -1; /* Do not solve X[k] in the future. */
                    lk1 = LBj( gik, grid ); /* Local block number. */
                    lsub = Llu->Lrowind_bc_ptr[lk1];
                    lusup = Llu->Lnzval_bc_ptr[lk1];
                    nsupr = lsub[1];
#ifdef _CRAY
                    STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &iknsupc, &nrhs, &alpha,
                          lusup, &nsupr, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
                    dtrsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
                           lusup, &nsupr, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
                    dtrsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
                           lusup, &nsupr, &x[ii], &iknsupc);
#endif
                    stat->ops[SOLVE] += iknsupc * (iknsupc + 1) * nrhs;
#if ( DEBUGlevel>=2 )
                    printf("(%2d) Solve X[%2d]\n", iam, gik);
#endif

                    /*
                     * Send Xk to process column Pc[k].
                     */
                    for (p = 0; p < grid->nprow; ++p)
                    {
                        if ( bsendx_plist[lk1][p] != SLU_EMPTY )
                        {
                            pi = PNUM( p, gikcol, grid );
#ifdef ISEND_IRECV
                            MPI_Isend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
                                       MPI_DOUBLE, pi, Xk, grid->comm,
                                       &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
                            MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
                                       MPI_DOUBLE, pi, Xk, grid->comm );
#else
                            MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
                                      MPI_DOUBLE, pi, Xk, grid->comm );
#endif
#endif
#if ( DEBUGlevel>=2 )
                            printf("(%2d) Sent X[%2.0f] to P %2d\n",
                                   iam, x[ii - XK_H], pi);
#endif
                        }
                    }
                    xtrsTimer->trsDataSendXY += iknsupc * nrhs + XK_H;
                    /*
                     * Perform local block modifications.
                     */
                    if ( Urbs[lk1] )
                        dlsum_bmod_GG_newsolve(trf3Dpartition, lsum, x, &x[ii], nrhs, lbmod_buf, gik, bmod, Urbs,
                                      Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                      send_req, stat,xtrsTimer);
                } /* if brecv[ik] == 0 */
            }
        } /* if bmod[ik] == 0 */
        } /* if (trf3Dpartition->supernodeMask[gik]>0) */
    } /* for ub ... */

} /* dlsum_bmod_GG_newsolve */



/*
 * Sketch of the algorithm for L-solve:
 * =======================
 *
 * Self-scheduling loop:
 *
 *   while ( not finished ) { .. use message counter to control
 *
 *      reveive a message;
 *
 * 	if ( message is Xk ) {
 * 	    perform local block modifications into lsum[];
 *                 lsum[i] -= L_i,k * X[k]
 *          if all local updates done, Isend lsum[] to diagonal process;
 *
 *      } else if ( message is LSUM ) { .. this must be a diagonal process
 *          accumulate LSUM;
 *          if ( all LSUM are received ) {
 *              perform triangular solve for Xi;
 *              Isend Xi down to the current process column;
 *              perform local block modifications into lsum[];
 *          }
 *      }
 *   }
 *
 *
 * Auxiliary data structures: lsum[] / ilsum (pointer to lsum array)
 * =======================
 *
 * lsum[] array (local)
 *   + lsum has "nrhs" columns, row-wise is partitioned by supernodes
 *   + stored by row blocks, column wise storage within a row block
 *   + prepend a header recording the global block number.
 *
 *         lsum[]                        ilsum[nsupers + 1]
 *
 *         -----
 *         | | |  <- header of size 2     ---
 *         --------- <--------------------| |
 *         | | | | |			  ---
 * 	   | | | | |	      |-----------| |
 *         | | | | | 	      |           ---
 *	   ---------          |   |-------| |
 *         | | |  <- header   |   |       ---
 *         --------- <--------|   |  |----| |
 *         | | | | |		  |  |    ---
 * 	   | | | | |              |  |
 *         | | | | |              |  |
 *	   ---------              |  |
 *         | | |  <- header       |  |
 *         --------- <------------|  |
 *         | | | | |                 |
 * 	   | | | | |                 |
 *         | | | | |                 |
 *	   --------- <---------------|
 */

/*#define ISEND_IRECV*/

/*
 * Function prototypes
 */
#ifdef _CRAY
fortran void STRSM(_fcd, _fcd, _fcd, _fcd, int*, int*, double*,
		   double*, int*, double*, int*);
_fcd ftcs1;
_fcd ftcs2;
_fcd ftcs3;
#endif





int_t dlocalSolveXkYk( trtype_t trtype, int_t k, double* x, int nrhs,
                      dLUstruct_t * LUstruct, gridinfo_t * grid,
                      SuperLUStat_t * stat)
{
    // printf("Solving %d \n",k );
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    double alpha = 1.0;
    int_t* xsup = Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int knsupc = (int)SuperSize (k);
    int_t lk = LBj (k, grid); /* Local block number, column-wise */
    int_t *lsub = Lrowind_bc_ptr[lk];
    double* lusup = Lnzval_bc_ptr[lk];
    int nsupr = (int) lsub[1];

    if (trtype == UPPER_TRI)
    {
        /* upper triangular matrix */
#ifdef _CRAY
        STRSM (ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
               lusup, &nsupr, x, &knsupc);
#elif defined (USE_VENDOR_BLAS)
        dtrsm_ ("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
                lusup, &nsupr, x, &knsupc, 1, 1, 1, 1);
#else
        dtrsm_ ("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
                lusup, &nsupr, x, &knsupc);
#endif
    }
    else
    {
        /* lower triangular matrix */
#ifdef _CRAY
        STRSM (ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
               lusup, &nsupr, x, &knsupc);
#elif defined (USE_VENDOR_BLAS)
        dtrsm_ ("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
                lusup, &nsupr, x, &knsupc, 1, 1, 1, 1);
#else
        dtrsm_ ("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
                lusup, &nsupr, x, &knsupc);
#endif
    }
    stat->ops[SOLVE] += knsupc * (knsupc + 1) * nrhs;
    return 0;
}

int_t diBcastXk2Pck(int_t k, double* x, int nrhs,
                   int** sendList, MPI_Request *send_req,
                   dLUstruct_t * LUstruct, gridinfo_t * grid,xtrsTimer_t *xtrsTimer)
{
    /*
     * Send Xk to process column Pc[k].
     */
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t Pr = grid->nprow;
    int_t knsupc = SuperSize (k);
    int_t lk = LBj (k, grid);
    int_t kcol = PCOL (k, grid);
    for (int_t p = 0; p < Pr; ++p)
    {
        if (sendList[lk][p] != SLU_EMPTY)
        {
            int_t pi = PNUM (p, kcol, grid);

            MPI_Isend (x, knsupc * nrhs + XK_H,
                       MPI_DOUBLE, pi, Xk, grid->comm,
                       &send_req[Llu->SolveMsgSent++]);

        }
    }

    xtrsTimer->trsDataSendXY += (double) SuperSize(k)*nrhs + XK_H;
    // printf("Data sent so far =%g and in this round= %g \n",xtrsTimer->trsDataSendXY, (double) SuperSize(k)*nrhs + XK_H );

    return 0;
}

/*! \brief
 *
 * <pre>
 * Purpose
 *
 *   Re-distribute B on the diagonal processes of the 2D process mesh (only on grid 0).
 *
 * Note
 *
 *   This routine can only be called after the routine pxgstrs_init(),
 *   in which the structures of the send and receive buffers are set up.
 *
 * Arguments
 *
  *
 * B      (input) double*
 *        The distributed right-hand side matrix of the possibly
 *        equilibrated system.
 *
 * m_loc  (input) int (local)
 *        The local row dimension of matrix B.
 *
 * nrhs   (input) int (global)
 *        Number of right-hand sides.
 *
 * ldb    (input) int (local)
 *        Leading dimension of matrix B.
 *
 * fst_row (input) int (global)
 *        The row number of B's first row in the global matrix.
 *
 * ilsum  (input) int* (global)
 *        Starting position of each supernode in a full array.
 *
 * x      (output) double*
 *        The solution vector. It is valid only on the diagonal processes.
 *
 * ScalePermstruct (input) dScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * SOLVEstruct (input) dSOLVEstruct_t*
 *        Contains the information for the communication during the
 *        solution phase.
 *
 * Return value

 * </pre>
 */

int_t
pdReDistribute3d_B_to_X (double *B, int_t m_loc, int nrhs, int_t ldb,
                         int_t fst_row, int_t * ilsum, double *x,
                         dScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist,
                         gridinfo3d_t * grid3d, dSOLVEstruct_t * SOLVEstruct)
{
    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *sdispls_nrhs, *rdispls, *rdispls_nrhs;
    int *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *perm_r, *perm_c;     /* row and column permutation vectors */
    int_t *send_ibuf, *recv_ibuf;
    double *send_dbuf, *recv_dbuf;
    int_t *xsup, *supno;
    int_t i, ii, irow, gbi, jj, k, knsupc, l, lk;
    int p, procs;
    gridinfo_t * grid = &(grid3d->grid2d);
    if (!grid3d->zscp.Iam)
    {
        pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

#if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter pdReDistribute3d_B_to_X()");
#endif

        /* ------------------------------------------------------------
           INITIALIZATION.
           ------------------------------------------------------------ */
        perm_r = ScalePermstruct->perm_r;
        perm_c = ScalePermstruct->perm_c;
        procs = grid->nprow * grid->npcol;
        xsup = Glu_persist->xsup;
        supno = Glu_persist->supno;
        SendCnt = gstrs_comm->B_to_X_SendCnt;
        SendCnt_nrhs = gstrs_comm->B_to_X_SendCnt + procs;
        RecvCnt = gstrs_comm->B_to_X_SendCnt + 2 * procs;
        RecvCnt_nrhs = gstrs_comm->B_to_X_SendCnt + 3 * procs;
        sdispls = gstrs_comm->B_to_X_SendCnt + 4 * procs;
        sdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 5 * procs;
        rdispls = gstrs_comm->B_to_X_SendCnt + 6 * procs;
        rdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 7 * procs;
        ptr_to_ibuf = gstrs_comm->ptr_to_ibuf;
        ptr_to_dbuf = gstrs_comm->ptr_to_dbuf;

        /* ------------------------------------------------------------
           NOW COMMUNICATE THE ACTUAL DATA.
           ------------------------------------------------------------ */
        k = sdispls[procs - 1] + SendCnt[procs - 1];    /* Total number of sends */
        l = rdispls[procs - 1] + RecvCnt[procs - 1];    /* Total number of receives */
        if (!(send_ibuf = intMalloc_dist (k + l)))
            ABORT ("Malloc fails for send_ibuf[].");
        recv_ibuf = send_ibuf + k;
        if (!(send_dbuf = doubleMalloc_dist ((k + l) * (size_t) nrhs)))
            ABORT ("Malloc fails for send_dbuf[].");
        recv_dbuf = send_dbuf + k * nrhs;

        for (p = 0; p < procs; ++p)
        {
            ptr_to_ibuf[p] = sdispls[p];
            ptr_to_dbuf[p] = sdispls[p] * nrhs;
        }

        /* Copy the row indices and values to the send buffer. */
        for (i = 0, l = fst_row; i < m_loc; ++i, ++l)
        {
            irow = perm_c[perm_r[l]];   /* Row number in Pc*Pr*B */
            gbi = BlockNum (irow);
            p = PNUM (PROW (gbi, grid), PCOL (gbi, grid), grid);    /* Diagonal process */
            k = ptr_to_ibuf[p];
            send_ibuf[k] = irow;
            k = ptr_to_dbuf[p];
            for (int_t j = 0; j < nrhs; ++j)
            {
                /* RHS is stored in row major in the buffer. */
                send_dbuf[k++] = B[i + j * ldb];
            }
            ++ptr_to_ibuf[p];
            ptr_to_dbuf[p] += nrhs;
        }

        /* Communicate the (permuted) row indices. */
        MPI_Alltoallv (send_ibuf, SendCnt, sdispls, mpi_int_t,
                       recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid->comm);

        /* Communicate the numerical values. */
        MPI_Alltoallv (send_dbuf, SendCnt_nrhs, sdispls_nrhs, MPI_DOUBLE,
                       recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, MPI_DOUBLE,
                       grid->comm);

        /* ------------------------------------------------------------
           Copy buffer into X on the diagonal processes.
           ------------------------------------------------------------ */
        ii = 0;
        for (p = 0; p < procs; ++p)
        {
            jj = rdispls_nrhs[p];
            for (int_t i = 0; i < RecvCnt[p]; ++i)
            {
                /* Only the diagonal processes do this; the off-diagonal processes
                   have 0 RecvCnt. */
                irow = recv_ibuf[ii];   /* The permuted row index. */
                k = BlockNum (irow);
                knsupc = SuperSize (k);
                lk = LBi (k, grid); /* Local block number. */
                l = X_BLK (lk);
			    x[l - XK_H] = k;      /* Block number prepended in the header. */
                irow = irow - FstBlockC (k);    /* Relative row number in X-block */
                for (int_t j = 0; j < nrhs; ++j)
                {
                    x[l + irow + j * knsupc] = recv_dbuf[jj++];
                }
                ++ii;
            }
        }

        SUPERLU_FREE (send_ibuf);
        SUPERLU_FREE (send_dbuf);
    }
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid->iam, "Exit pdReDistribute3d_B_to_X()");
#endif
    return 0;
}                               /* pdReDistribute3d_B_to_X */

/*! \brief
 *
 * <pre>
 * Purpose
 *
 *   Re-distribute X on the diagonal processes to B distributed on all
 *   the processes (only on grid 0)
 *
 * Note
 *
 *   This routine can only be called after the routine pxgstrs_init(),
 *   in which the structures of the send and receive buffers are set up.
 * </pre>
 */

int_t
pdReDistribute3d_X_to_B (int_t n, double *B, int_t m_loc, int_t ldb,
                         int_t fst_row, int nrhs, double *x, int_t * ilsum,
                         dScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist, gridinfo3d_t * grid3d,
                         dSOLVEstruct_t * SOLVEstruct)
{
    int_t i, ii, irow,  jj, k, knsupc, nsupers, l, lk;
    int_t *xsup, *supno;
    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *rdispls, *sdispls_nrhs, *rdispls_nrhs;
    int *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *send_ibuf, *recv_ibuf;
    double *send_dbuf, *recv_dbuf;
    int iam, p, q, pkk, procs;
    int_t num_diag_procs, *diag_procs;
    gridinfo_t * grid = &(grid3d->grid2d);
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid->iam, "Enter pdReDistribute_X_to_B()");
#endif

    /* ------------------------------------------------------------
       INITIALIZATION.
       ------------------------------------------------------------ */
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = Glu_persist->supno[n - 1] + 1;
    iam = grid->iam;
    procs = grid->nprow * grid->npcol;
    if (!grid3d->zscp.Iam)
    {
        int_t *row_to_proc = SOLVEstruct->row_to_proc;  /* row-process mapping */
        pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

        SendCnt = gstrs_comm->X_to_B_SendCnt;
        SendCnt_nrhs = gstrs_comm->X_to_B_SendCnt + procs;
        RecvCnt = gstrs_comm->X_to_B_SendCnt + 2 * procs;
        RecvCnt_nrhs = gstrs_comm->X_to_B_SendCnt + 3 * procs;
        sdispls = gstrs_comm->X_to_B_SendCnt + 4 * procs;
        sdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 5 * procs;
        rdispls = gstrs_comm->X_to_B_SendCnt + 6 * procs;
        rdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 7 * procs;
        ptr_to_ibuf = gstrs_comm->ptr_to_ibuf;
        ptr_to_dbuf = gstrs_comm->ptr_to_dbuf;

        k = sdispls[procs - 1] + SendCnt[procs - 1];    /* Total number of sends */
        l = rdispls[procs - 1] + RecvCnt[procs - 1];    /* Total number of receives */
        if (!(send_ibuf = intMalloc_dist (k + l)))
            ABORT ("Malloc fails for send_ibuf[].");
        recv_ibuf = send_ibuf + k;
        if (!(send_dbuf = doubleMalloc_dist ((k + l) * nrhs)))
            ABORT ("Malloc fails for send_dbuf[].");
        recv_dbuf = send_dbuf + k * nrhs;
        for (p = 0; p < procs; ++p)
        {
            ptr_to_ibuf[p] = sdispls[p];
            ptr_to_dbuf[p] = sdispls_nrhs[p];
        }
        num_diag_procs = SOLVEstruct->num_diag_procs;
        diag_procs = SOLVEstruct->diag_procs;

        for (p = 0; p < num_diag_procs; ++p)
        {
            /* For all diagonal processes. */
            pkk = diag_procs[p];
            if (iam == pkk)
            {
                for (k = p; k < nsupers; k += num_diag_procs)
                {
                    knsupc = SuperSize (k);
                    lk = LBi (k, grid); /* Local block number */
                    irow = FstBlockC (k);
                    l = X_BLK (lk);
                    for (i = 0; i < knsupc; ++i)
                    {

                        ii = irow;

                        q = row_to_proc[ii];
                        jj = ptr_to_ibuf[q];
                        send_ibuf[jj] = ii;
                        jj = ptr_to_dbuf[q];
                        for (int_t j = 0; j < nrhs; ++j)
                        {
                            /* RHS stored in row major in buffer. */
                            send_dbuf[jj++] = x[l + i + j * knsupc];
                        }
                        ++ptr_to_ibuf[q];
                        ptr_to_dbuf[q] += nrhs;
                        ++irow;
                    }
                }
            }
        }

        /* ------------------------------------------------------------
           COMMUNICATE THE (PERMUTED) ROW INDICES AND NUMERICAL VALUES.
           ------------------------------------------------------------ */
        MPI_Alltoallv (send_ibuf, SendCnt, sdispls, mpi_int_t,
                       recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid->comm);
        MPI_Alltoallv (send_dbuf, SendCnt_nrhs, sdispls_nrhs, MPI_DOUBLE,
                       recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, MPI_DOUBLE,
                       grid->comm);

        /* ------------------------------------------------------------
           COPY THE BUFFER INTO B.
           ------------------------------------------------------------ */
        for (i = 0, k = 0; i < m_loc; ++i)
        {
            irow = recv_ibuf[i];
            irow -= fst_row;        /* Relative row number */
            for (int_t j = 0; j < nrhs; ++j)
            {
                /* RHS is stored in row major in the buffer. */
                B[irow + j * ldb] = recv_dbuf[k++];
            }
        }

        SUPERLU_FREE (send_ibuf);
        SUPERLU_FREE (send_dbuf);
    }
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid->iam, "Exit pdReDistribute_X_to_B()");
#endif
    return 0;

}                               /* pdReDistribute_X_to_B */

static int
pdgstrs3d_symv2_owner(dtrf3Dpartition_t *trf3Dpartition, int_t k)
{
    if (trf3Dpartition == NULL ||
        trf3Dpartition->symV2DiagOwner == NULL)
        ABORT("SymFact V2 solve redistribution requires owner tables.");
    return trf3Dpartition->symV2DiagOwner[k];
}

static int_t
pdgstrs3d_symv2_row_index(dtrf3Dpartition_t *trf3Dpartition, int_t k)
{
    int_t lk;
    if (trf3Dpartition == NULL ||
        trf3Dpartition->symV2RowLocalIndex == NULL)
        ABORT("SymFact V2 solve redistribution requires local row indexes.");
    lk = trf3Dpartition->symV2RowLocalIndex[k];
    if (lk < 0)
        ABORT("SymFact V2 solve redistribution missing local row index.");
    return lk;
}

static int_t
pdReDistribute3d_B_to_X_symv2(double *B, int_t m_loc, int nrhs, int_t ldb,
                         int_t fst_row, int_t * ilsum, double *x,
                         dScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist,
                         dtrf3Dpartition_t *trf3Dpartition,
                         gridinfo3d_t * grid3d, dSOLVEstruct_t * SOLVEstruct)
{
    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *sdispls_nrhs, *rdispls, *rdispls_nrhs;
    int *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *perm_r, *perm_c;
    int_t *send_ibuf, *recv_ibuf;
    double *send_dbuf, *recv_dbuf;
    int_t *xsup, *supno;
    int_t i, ii, irow, gbi, jj, k, knsupc, l, lk;
    int p, procs;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

    MPI_Comm_size(grid3d->comm, &procs);
    perm_r = ScalePermstruct->perm_r;
    perm_c = ScalePermstruct->perm_c;
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    SendCnt = gstrs_comm->B_to_X_SendCnt;
    SendCnt_nrhs = gstrs_comm->B_to_X_SendCnt + procs;
    RecvCnt = gstrs_comm->B_to_X_SendCnt + 2 * procs;
    RecvCnt_nrhs = gstrs_comm->B_to_X_SendCnt + 3 * procs;
    sdispls = gstrs_comm->B_to_X_SendCnt + 4 * procs;
    sdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 5 * procs;
    rdispls = gstrs_comm->B_to_X_SendCnt + 6 * procs;
    rdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 7 * procs;
    ptr_to_ibuf = gstrs_comm->ptr_to_ibuf;
    ptr_to_dbuf = gstrs_comm->ptr_to_dbuf;

    k = sdispls[procs - 1] + SendCnt[procs - 1];
    l = rdispls[procs - 1] + RecvCnt[procs - 1];
    if (!(send_ibuf = intMalloc_dist(k + l)))
        ABORT("Malloc fails for SymV2 send_ibuf[].");
    recv_ibuf = send_ibuf + k;
    if (!(send_dbuf = doubleMalloc_dist((k + l) * (size_t) nrhs)))
        ABORT("Malloc fails for SymV2 send_dbuf[].");
    recv_dbuf = send_dbuf + k * nrhs;

    for (p = 0; p < procs; ++p) {
        ptr_to_ibuf[p] = sdispls[p];
        ptr_to_dbuf[p] = sdispls[p] * nrhs;
    }

    if (!grid3d->zscp.Iam) {
        for (i = 0, l = fst_row; i < m_loc; ++i, ++l) {
            irow = perm_c[perm_r[l]];
            gbi = BlockNum(irow);
            p = pdgstrs3d_symv2_owner(trf3Dpartition, gbi);
            k = ptr_to_ibuf[p];
            send_ibuf[k] = irow;
            k = ptr_to_dbuf[p];
            for (int_t j = 0; j < nrhs; ++j)
                send_dbuf[k++] = B[i + j * ldb];
            ++ptr_to_ibuf[p];
            ptr_to_dbuf[p] += nrhs;
        }
    }

    MPI_Alltoallv(send_ibuf, SendCnt, sdispls, mpi_int_t,
                  recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid3d->comm);
    MPI_Alltoallv(send_dbuf, SendCnt_nrhs, sdispls_nrhs, MPI_DOUBLE,
                  recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, MPI_DOUBLE,
                  grid3d->comm);

    ii = 0;
    for (p = 0; p < procs; ++p) {
        jj = rdispls_nrhs[p];
        for (int_t i = 0; i < RecvCnt[p]; ++i) {
            irow = recv_ibuf[ii];
            k = BlockNum(irow);
            knsupc = SuperSize(k);
            lk = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
            l = X_BLK(lk);
            x[l - XK_H] = k;
            irow -= FstBlockC(k);
            for (int_t j = 0; j < nrhs; ++j)
                x[l + irow + j * knsupc] = recv_dbuf[jj++];
            ++ii;
        }
    }

    SUPERLU_FREE(send_ibuf);
    SUPERLU_FREE(send_dbuf);
    return 0;
}

static int_t
pdReDistribute3d_X_to_B_symv2(int_t n, double *B, int_t m_loc, int_t ldb,
                         int_t fst_row, int nrhs, double *x, int_t * ilsum,
                         dScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist,
                         dtrf3Dpartition_t *trf3Dpartition,
                         gridinfo3d_t * grid3d, dSOLVEstruct_t * SOLVEstruct)
{
    int_t i, irow, jj, k, knsupc, nsupers, l, lk;
    int_t *xsup;
    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *rdispls, *sdispls_nrhs, *rdispls_nrhs;
    int *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *send_ibuf, *recv_ibuf;
    double *send_dbuf, *recv_dbuf;
    int iam, p, q, procs;

    xsup = Glu_persist->xsup;
    nsupers = Glu_persist->supno[n - 1] + 1;
    iam = grid3d->iam;
    MPI_Comm_size(grid3d->comm, &procs);
    int_t *row_to_proc = SOLVEstruct->row_to_proc;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

    SendCnt = gstrs_comm->X_to_B_SendCnt;
    SendCnt_nrhs = gstrs_comm->X_to_B_SendCnt + procs;
    RecvCnt = gstrs_comm->X_to_B_SendCnt + 2 * procs;
    RecvCnt_nrhs = gstrs_comm->X_to_B_SendCnt + 3 * procs;
    sdispls = gstrs_comm->X_to_B_SendCnt + 4 * procs;
    sdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 5 * procs;
    rdispls = gstrs_comm->X_to_B_SendCnt + 6 * procs;
    rdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 7 * procs;
    ptr_to_ibuf = gstrs_comm->ptr_to_ibuf;
    ptr_to_dbuf = gstrs_comm->ptr_to_dbuf;

    k = sdispls[procs - 1] + SendCnt[procs - 1];
    l = rdispls[procs - 1] + RecvCnt[procs - 1];
    if (!(send_ibuf = intMalloc_dist(k + l)))
        ABORT("Malloc fails for SymV2 X send_ibuf[].");
    recv_ibuf = send_ibuf + k;
    if (!(send_dbuf = doubleMalloc_dist((k + l) * nrhs)))
        ABORT("Malloc fails for SymV2 X send_dbuf[].");
    recv_dbuf = send_dbuf + k * nrhs;

    for (p = 0; p < procs; ++p) {
        ptr_to_ibuf[p] = sdispls[p];
        ptr_to_dbuf[p] = sdispls_nrhs[p];
    }

    for (k = 0; k < nsupers; ++k) {
        p = pdgstrs3d_symv2_owner(trf3Dpartition, k);
        if (iam != p)
            continue;
        knsupc = SuperSize(k);
        lk = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
        irow = FstBlockC(k);
        l = X_BLK(lk);
        for (i = 0; i < knsupc; ++i) {
            q = row_to_proc[irow];
            jj = ptr_to_ibuf[q];
            send_ibuf[jj] = irow;
            jj = ptr_to_dbuf[q];
            for (int_t j = 0; j < nrhs; ++j)
                send_dbuf[jj++] = x[l + i + j * knsupc];
            ++ptr_to_ibuf[q];
            ptr_to_dbuf[q] += nrhs;
            ++irow;
        }
    }

    MPI_Alltoallv(send_ibuf, SendCnt, sdispls, mpi_int_t,
                  recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid3d->comm);
    MPI_Alltoallv(send_dbuf, SendCnt_nrhs, sdispls_nrhs, MPI_DOUBLE,
                  recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, MPI_DOUBLE,
                  grid3d->comm);

    int_t total_recv = rdispls[procs - 1] + RecvCnt[procs - 1];
    for (i = 0, k = 0; i < total_recv; ++i) {
        irow = recv_ibuf[i] - fst_row;
        for (int_t j = 0; j < nrhs; ++j)
            B[irow + j * ldb] = recv_dbuf[k++];
    }

    SUPERLU_FREE(send_ibuf);
    SUPERLU_FREE(send_dbuf);
    return 0;
}


/*! \brief
 *
 * <pre>
 * Purpose
 *
 *
 * PDGSTRS solves a system of distributed linear equations
 * A*X = B with a general N-by-N matrix A using the LU factorization
 * computed by PDGSTRF.
 * If the equilibration, and row and column permutations were performed,
 * the LU factorization was performed for A1 where
 *     A1 = Pc*Pr*diag(R)*A*diag(C)*Pc^T = L*U
 * and the linear system solved is
 *     A1 * Y = Pc*Pr*B1, where B was overwritten by B1 = diag(R)*B, and
 * the permutation to B1 by Pc*Pr is applied internally in this routine.
 *
 * Arguments
 *
 *
 * n      (input) int (global)
 *        The order of the system of linear equations.
 *
 * LUstruct (input) dLUstruct_t*
 *        The distributed data structures storing L and U factors.
 *        The L and U factors are obtained from PDGSTRF for
 *        the possibly scaled and permuted matrix A.
 *        See superlu_ddefs.h for the definition of 'dLUstruct_t'.
 *        A may be scaled and permuted into A1, so that
 *        A1 = Pc*Pr*diag(R)*A*diag(C)*Pc^T = L*U
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh. It contains the MPI communicator, the number
 *        of process rows (NPROW), the number of process columns (NPCOL),
 *        and my process rank. It is an input argument to all the
 *        parallel routines.
 *        Grid can be initialized by subroutine SUPERLU_GRIDINIT.
 *        See superlu_defs.h for the definition of 'gridinfo_t'.
 *
 * B      (input/output) double*
 *        On entry, the distributed right-hand side matrix of the possibly
 *        equilibrated system. That is, B may be overwritten by diag(R)*B.
 *        On exit, the distributed solution matrix Y of the possibly
 *        equilibrated system if info = 0, where Y = Pc*diag(C)^(-1)*X,
 *        and X is the solution of the original system.
 *
 * m_loc  (input) int (local)
 *        The local row dimension of matrix B.
 *
 * fst_row (input) int (global)
 *        The row number of B's first row in the global matrix.
 *
 * ldb    (input) int (local)
 *        The leading dimension of matrix B.
 *
 * nrhs   (input) int (global)
 *        Number of right-hand sides.
 *
 * SOLVEstruct (input) dSOLVEstruct_t* (global)
 *        Contains the information for the communication during the
 *        solution phase.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the triangular solves.
 *        See util.h for the definition of 'SuperLUStat_t'.
 *
 * info   (output) int*
 *     = 0: successful exit
 *     < 0: if info = -i, the i-th argument had an illegal value
 * </pre>
 */

void
pdgstrs3d (superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
           dScalePermstruct_t * ScalePermstruct,
           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, double *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info)
{
    // printf("Using pdgstr3d ..\n");
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;

    double *lsum;               /* Local running sum of the updates to B-components */
    double *x;                  /* X component at step k. */
    /* NOTE: x and lsum are of same size. */

    double *recvbuf;


    int_t iam,  mycol, myrow;
    int_t i, k;
    int_t  nlb, nsupers;
    int_t *xsup, *supno;
    int_t *ilsum;               /* Starting position of each supernode in lsum (LOCAL) */
    int_t Pc, Pr;
    int knsupc;
    int ldalsum;                /* Number of lsum entries locally owned. */
    int maxrecvsz;
    int_t **Lrowind_bc_ptr;
    double **Lnzval_bc_ptr;
    MPI_Status status;
    MPI_Request *send_req;


    double t;
#if ( DEBUGlevel>=2 )
    int_t Ublocks = 0;
#endif



    t = SuperLU_timer_ ();

    /* Test input parameters. */
    *info = 0;
    if ( n < 0 ) *info = -1;
    else if ( nrhs < 0 ) *info = -9;
    if ( *info ) {
	pxerr_dist("PDGSTRS", grid, -*info);
	return;
    }
#ifdef _CRAY
    ftcs1 = _cptofcd ("L", strlen ("L"));
    ftcs2 = _cptofcd ("N", strlen ("N"));
    ftcs3 = _cptofcd ("U", strlen ("U"));
#endif

    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW (iam, grid);
    mycol = MYCOL (iam, grid);
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = supno[n - 1] + 1;
    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    nlb = CEILING (nsupers, Pr);    /* Number of local block rows. */
    int_t nub = CEILING (nsupers, Pc);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Enter pdgstrs3d()");
#endif

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;


    k = SUPERLU_MAX (Llu->nfsendx, Llu->nbsendx) + nlb;
    if (!
            (send_req =
                 (MPI_Request *) SUPERLU_MALLOC (k * sizeof (MPI_Request))))
        ABORT ("Malloc fails for send_req[].");




    /* Obtain ilsum[] and ldalsum for process column 0. */


    ilsum = Llu->ilsum;
    ldalsum = Llu->ldalsum;

    /* Allocate working storage. */
    knsupc = sp_ienv_dist (3,options);
    maxrecvsz = pdgstrs3d_checked_workspace_count(knsupc, nrhs, 1,
                                                  SUPERLU_MAX (XK_H, LSUM_H),
                                                  "3D solve recvbuf");
    int_t lsum_count = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb,
                                                         LSUM_H,
                                                         "3D solve lsum workspace");
    int_t x_count = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb,
                                                      XK_H,
                                                      "3D solve x workspace");
    if (!
            (lsum = doubleCalloc_dist (lsum_count)))
        ABORT ("Calloc fails for lsum[].");
    if (!(x = doubleMalloc_dist (x_count)))
        ABORT ("Malloc fails for x[].");
    if (!(recvbuf = doubleMalloc_dist (maxrecvsz)))
        ABORT ("Malloc fails for recvbuf[].");

    /**
     * Initializing xT
     */

    int_t* ilsumT = SUPERLU_MALLOC (sizeof(int_t) * (nub + 1));
    int_t ldaspaT = 0;
    ilsumT[0] = 0;
    for (int_t jb = 0; jb < nsupers; ++jb)
    {
        if ( mycol == PCOL( jb, grid ) )
        {
            int_t i = SuperSize( jb );
            ldaspaT += i;
            int_t ljb = LBj( jb, grid );
            ilsumT[ljb + 1] = ilsumT[ljb] + i;
        }
    }
    double* xT;
    int_t xT_count = pdgstrs3d_checked_workspace_count(ldaspaT, nrhs, nub,
                                                       XK_H,
                                                       "3D solve xT workspace");
    if (!(xT = doubleMalloc_dist (xT_count)))
        ABORT ("Malloc fails for xT[].");
    /**
     * Setup the headers for xT
     */
    for (int_t jb = 0; jb < nsupers; ++jb)
    {
        if ( mycol == PCOL( jb, grid ) )
        {
            int_t ljb = LBj( jb, grid );
            int_t jj = XT_BLK (ljb);

	        xT[jj] = jb;

        }
    }

    dxT_struct xT_s;
    xT_s.xT = xT;
    xT_s.ilsumT = ilsumT;
    xT_s.ldaspaT = ldaspaT;

    xtrsTimer_t xtrsTimer;

    initTRStimer(&xtrsTimer, grid);
    double tx = SuperLU_timer_();
    /* Redistribute B into X on the diagonal processes. */
    pdReDistribute3d_B_to_X_symv2(B, m_loc, nrhs, ldb, fst_row, ilsum, x,
                                  ScalePermstruct, Glu_persist,
                                  trf3Dpartition, grid3d, SOLVEstruct);

    xtrsTimer.t_pxReDistribute_B_to_X = SuperLU_timer_() - tx;

    /*---------------------------------------------------
     * Forward solve Ly = b.
     *---------------------------------------------------*/

    dtrs_B_init3d(nsupers, x, nrhs, LUstruct, grid3d);

    MPI_Barrier (grid3d->comm);
    tx = SuperLU_timer_();
    stat->utime[SOLVE] = 0.0;
    double tx_st= SuperLU_timer_();


    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 printf("before pdgsTrForwardSolve3d: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }

    pdgsTrForwardSolve3d(options, n,  LUstruct, ScalePermstruct, trf3Dpartition, grid3d, x,  lsum, &xT_s,
                          recvbuf, send_req,  nrhs, SOLVEstruct,  stat, &xtrsTimer);
    // pdgsTrForwardSolve3d_2d( n,  LUstruct, ScalePermstruct, trf3Dpartition, grid3d, x,  lsum, &xT_s,
    //                          recvbuf, send_req,  nrhs, SOLVEstruct,  stat, info);
    xtrsTimer.t_forwardSolve = SuperLU_timer_() - tx;



    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 // for(int_t i=0;i<knsupc;i++)
    //                 printf("check x after L solve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }

    /*---------------------------------------------------
     * Back solve Ux = y.
     *
     * The Y components from the forward solve is already
     * on the diagonal processes.
     *---------------------------------------------------*/
    tx = SuperLU_timer_();
    pdgsTrBackSolve3d(options, n,  LUstruct, ScalePermstruct, trf3Dpartition, grid3d, x,  lsum, &xT_s,
                       recvbuf, send_req,  nrhs, SOLVEstruct,  stat, &xtrsTimer);

    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 // for(int_t i=0;i<knsupc;i++)
    //                 printf("check x after U solve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }

    xtrsTimer.t_backwardSolve = SuperLU_timer_() - tx;
    MPI_Barrier (grid3d->comm);
    stat->utime[SOLVE] = SuperLU_timer_ () - tx_st;
    dtrs_X_gather3d(x, nrhs, trf3Dpartition, LUstruct, grid3d, &xtrsTimer);
    tx = SuperLU_timer_();
    pdReDistribute3d_X_to_B_symv2(n, B, m_loc, ldb, fst_row, nrhs, x,
                                  ilsum, ScalePermstruct, Glu_persist,
                                  trf3Dpartition, grid3d, SOLVEstruct);

    xtrsTimer.t_pxReDistribute_X_to_B = SuperLU_timer_() - tx;

    /**
     * Reduce the Solve flops from all the grids to grid zero
     */
    reduceStat(SOLVE, stat, grid3d);
    /* Deallocate storage. */
    SUPERLU_FREE (lsum);
    SUPERLU_FREE (x);
    SUPERLU_FREE (recvbuf);
    SUPERLU_FREE (ilsumT);
    SUPERLU_FREE (xT);


    /*for (i = 0; i < Llu->SolveMsgSent; ++i) MPI_Request_free(&send_req[i]); */

    for (i = 0; i < Llu->SolveMsgSent; ++i)
        MPI_Wait (&send_req[i], &status);
    SUPERLU_FREE (send_req);

    MPI_Barrier (grid->comm);

#if ( PRNTlevel >= 1)
    printTRStimer(&xtrsTimer, grid3d);
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Exit pdgstrs3d()");
#endif

    return;
}                               /* pdgstrs3d */


void
pdgstrs3d_newsolve (superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
           dScalePermstruct_t * ScalePermstruct,
           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, double *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info)
{
    // printf("Using pdgstr3d ..\n");
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;

    double *lsum;               /* Local running sum of the updates to B-components */
    double *x;                  /* X component at step k. */
    /* NOTE: x and lsum are of same size. */

    double *recvbuf;
    double zero = 0.0;


    int_t iam,  mycol, myrow;
    int_t i, k, ii;
    int_t  nlb, nsupers;
    int_t *xsup, *supno;
    int_t *ilsum;               /* Starting position of each supernode in lsum (LOCAL) */
    int_t Pc, Pr;
    int knsupc;
    int ldalsum;                /* Number of lsum entries locally owned. */
    int maxrecvsz;
    int_t **Lrowind_bc_ptr;
    double **Lnzval_bc_ptr;
    MPI_Status status;
    MPI_Request *send_req;


    double t;
#if ( DEBUGlevel>=2 )
    int_t Ublocks = 0;
#endif

    t = SuperLU_timer_ ();

    /* Test input parameters. */
    *info = 0;
    if ( n < 0 ) *info = -1;
    else if ( nrhs < 0 ) *info = -9;
    if ( *info ) {
	pxerr_dist("PDGSTRS", grid, -*info);
	return;
    }
#ifdef _CRAY
    ftcs1 = _cptofcd ("L", strlen ("L"));
    ftcs2 = _cptofcd ("N", strlen ("N"));
    ftcs3 = _cptofcd ("U", strlen ("U"));
#endif

    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW (iam, grid);
    mycol = MYCOL (iam, grid);
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = supno[n - 1] + 1;
    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    nlb = CEILING (nsupers, Pr);    /* Number of local block rows. */
    int_t nub = CEILING (nsupers, Pc);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Enter pdgstrs3d_newsolve()");
#endif

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    k = SUPERLU_MAX (Llu->nfsendx, Llu->nbsendx) + nlb;
 /* skip send_req on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    if (!(send_req =
                 (MPI_Request *) SUPERLU_MALLOC (k * sizeof (MPI_Request))))
        ABORT ("Malloc fails for send_req[].");
}

    /* Obtain ilsum[] and ldalsum for process column 0. */


    ilsum = Llu->ilsum;
    ldalsum = Llu->ldalsum;

    /* Allocate working storage. */
    knsupc = sp_ienv_dist (3,options);
    maxrecvsz = pdgstrs3d_checked_workspace_count(knsupc, nrhs, 1,
                                                  SUPERLU_MAX (XK_H, LSUM_H),
                                                  "3D new solve recvbuf");


    int_t sizelsum,sizertemp,aln_d,aln_i;
    aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    sizelsum = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb, LSUM_H,
                                                 "3D new solve lsum workspace");
    if (aln_d > 1) {
        size_t lsum_aligned = (size_t) sizelsum;
        size_t addend = (size_t) aln_d - 1;
        if (lsum_aligned > ((size_t)-1) - addend)
            ABORT("Workspace size overflows allocation size.");
        lsum_aligned = ((lsum_aligned + addend) / (size_t) aln_d) * (size_t) aln_d;
        sizelsum = pdgstrs3d_checked_size_to_int_t(lsum_aligned,
                                                   "3D new solve lsum workspace");
    }
    int_t x_count = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb,
                                                      XK_H,
                                                      "3D new solve x workspace");

    int num_thread = 1;
#ifdef _OPENMP
#pragma omp parallel default(shared)
    {
    	if (omp_get_thread_num () == 0) {
    		num_thread = omp_get_num_threads ();
    	}
    }
#else
	num_thread=1;
#endif

#if ( PRNTlevel>=1 )
    if( grid3d->iam==0 ) {
	printf("num_thread: %5d\n", num_thread);
	fflush(stdout);
    }
#endif



/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    size_t lsum_thread_count = pdgstrs3d_checked_product((size_t) sizelsum,
                                                         (size_t) num_thread,
                                                         "3D new solve lsum workspace");
    int_t lsum_thread_count_i = pdgstrs3d_checked_size_to_int_t(lsum_thread_count,
                                                                "3D new solve lsum workspace");
    size_t lsum_thread_bytes = pdgstrs3d_checked_product(lsum_thread_count,
                                                         sizeof(double),
                                                         "3D new solve lsum workspace");
#ifdef _OPENMP
    if ( !(lsum = (double*)SUPERLU_MALLOC(lsum_thread_bytes)))
	ABORT("Malloc fails for lsum[].");
#pragma omp parallel default(shared) private(ii)
    {
	int thread_id = omp_get_thread_num(); //mjc
	for (ii=0; ii<sizelsum; ii++)
    	    lsum[thread_id*sizelsum+ii]=zero;
    }
#else
    if ( !(lsum = (double*)SUPERLU_MALLOC(lsum_thread_bytes)))
  	    ABORT("Malloc fails for lsum[].");
    for ( ii=0; ii < lsum_thread_count_i; ii++ )
	lsum[ii]=zero;
#endif
}

    /* intermediate solution x[] vector has same structure as lsum[], see leading comment */
    if ( !(x = doubleCalloc_dist(x_count)) )
	ABORT("Calloc fails for x[].");
    if (!(recvbuf = doubleMalloc_dist (maxrecvsz)))
        ABORT ("Malloc fails for recvbuf[].");

    xtrsTimer_t xtrsTimer;

    initTRStimer(&xtrsTimer, grid);
    double tx = SuperLU_timer_();
    /* Redistribute B into X on the diagonal processes. */
    pdReDistribute3d_B_to_X(B, m_loc, nrhs, ldb, fst_row, ilsum, x,
                            ScalePermstruct, Glu_persist, grid3d, SOLVEstruct);

    xtrsTimer.t_pxReDistribute_B_to_X = SuperLU_timer_() - tx;

    /*---------------------------------------------------
     * Forward solve Ly = b.
     *---------------------------------------------------*/

    dtrs_B_init3d_newsolve(nsupers, x, nrhs, LUstruct, grid3d, trf3Dpartition);

    MPI_Barrier (grid3d->comm);
    tx = SuperLU_timer_();
    stat->utime[SOLVE] = 0.0;
    double tx_st= SuperLU_timer_();


    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 printf("before pdgsTrForwardSolve3d_newsolve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }



    pdgsTrForwardSolve3d_newsolve(options, n,  LUstruct, ScalePermstruct, trf3Dpartition, grid3d, x,  lsum,
                          recvbuf, send_req,  nrhs, SOLVEstruct,  stat, &xtrsTimer);
    xtrsTimer.t_forwardSolve = SuperLU_timer_() - tx;

    // printf("Llu->SolveMsgSent %10d size %10d\n",Llu->SolveMsgSent,SUPERLU_MAX (Llu->nfsendx, Llu->nbsendx) + nlb);
    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 printf("before dtrs_x_reduction_newsolve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }

    tx = SuperLU_timer_();
    dtrs_x_reduction_newsolve(nsupers, x, nrhs, LUstruct, grid3d, trf3Dpartition, recvbuf, &xtrsTimer);
    dtrs_x_broadcast_newsolve(nsupers, x, nrhs, LUstruct, grid3d, trf3Dpartition, recvbuf, &xtrsTimer);
    xtrsTimer.trs_comm_z += SuperLU_timer_() - tx;

    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    ////                 if(grid3d->iam==7)
    //                 printf("check x after L solve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }

    /*---------------------------------------------------
     * Back solve Ux = y.
     *
     * The Y components from the forward solve is already
     * on the diagonal processes.
     *---------------------------------------------------*/
    tx = SuperLU_timer_();
    pdgsTrBackSolve3d_newsolve(options, n,  LUstruct, trf3Dpartition, grid3d, x,  lsum,
                       recvbuf, send_req,  nrhs, SOLVEstruct,  stat, &xtrsTimer);

    // printf("pdgsTrBackSolve3d_newsolve Llu->SolveMsgSent %10d size %10d\n",Llu->SolveMsgSent,SUPERLU_MAX (Llu->nfsendx, Llu->nbsendx) + nlb);
    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 // for(int_t i=0;i<knsupc;i++)
    ////                 if(grid3d->iam==7)
    //                 printf("check x after U solve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }


    xtrsTimer.t_backwardSolve = SuperLU_timer_() - tx;
    MPI_Barrier (grid3d->comm);
    stat->utime[SOLVE] = SuperLU_timer_ () - tx_st;
    dtrs_X_gather3d(x, nrhs, trf3Dpartition, LUstruct, grid3d, &xtrsTimer);
    tx = SuperLU_timer_();
    pdReDistribute3d_X_to_B(n, B, m_loc, ldb, fst_row, nrhs, x, ilsum,
                            ScalePermstruct, Glu_persist, grid3d, SOLVEstruct);

    xtrsTimer.t_pxReDistribute_X_to_B = SuperLU_timer_() - tx;

    /**
     * Reduce the Solve flops from all the grids to grid zero
     */
    reduceStat(SOLVE, stat, grid3d);
    /* Deallocate storage. */

/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    SUPERLU_FREE (lsum);
}
    SUPERLU_FREE (x);
    SUPERLU_FREE (recvbuf);


/* skip send_req on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    /*for (i = 0; i < Llu->SolveMsgSent; ++i) MPI_Request_free(&send_req[i]); */
    for (i = 0; i < Llu->SolveMsgSent; ++i)
        MPI_Wait (&send_req[i], &status);
    SUPERLU_FREE (send_req);
}
    // MPI_Barrier (grid->comm);

#if ( PRNTlevel >= 1 )
    printTRStimer(&xtrsTimer, grid3d);
#endif    
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Exit pdgstrs3d_newsolve()");
#endif

    return;
}                               /* pdgstrs3d_newsolve */


static int
pdgstrs3d_symldl_env_enabled(const char *name)
{
    const char *value = getenv(name);
    return value != NULL && atoi(value) != 0;
}

static int
pdgstrs3d_symldl_env_is_set(const char *name)
{
    return getenv(name) != NULL;
}

static int
pdgstrs3d_symldl_count_to_int(int_t count, const char *what)
{
    int out = (int) count;
    (void) what;
    if (count < 0 || (int_t) out != count)
        ABORT("SymLDL solve MPI count overflows int.");
    return out;
}

static void
pdgstrs3d_symldl_dgemm(const char *transa, const char *transb,
                       int_t m_in, int_t n_in, int_t k_in,
                       double alpha, double *a, int_t lda_in,
                       double *b, int_t ldb_in,
                       double beta, double *c, int_t ldc_in)
{
    int m = pdgstrs3d_symldl_count_to_int(m_in, "SymLDL BLAS m");
    int n = pdgstrs3d_symldl_count_to_int(n_in, "SymLDL BLAS n");
    int k = pdgstrs3d_symldl_count_to_int(k_in, "SymLDL BLAS k");
    int lda = pdgstrs3d_symldl_count_to_int(lda_in, "SymLDL BLAS lda");
    int ldb = pdgstrs3d_symldl_count_to_int(ldb_in, "SymLDL BLAS ldb");
    int ldc = pdgstrs3d_symldl_count_to_int(ldc_in, "SymLDL BLAS ldc");

    if (m == 0 || n == 0)
        return;

#if defined (USE_VENDOR_BLAS)
    dgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb,
           &beta, c, &ldc, 1, 1);
#else
    dgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb,
           &beta, c, &ldc);
#endif
}

static void
pdgstrs3d_symldl_counts_to_displs(int nprocs, int *counts, int *displs,
                                  int *total)
{
    int p;
    displs[0] = 0;
    *total = counts[0];
    for (p = 1; p < nprocs; ++p) {
        displs[p] = displs[p - 1] + counts[p - 1];
        *total += counts[p];
    }
}

static void
pdgstrs3d_symldl_grow_int_t_buffer(int_t **buffer, int_t *capacity,
                                   int_t need, const char *what)
{
    if (need <= *capacity)
        return;
    if (*buffer)
        SUPERLU_FREE(*buffer);
    if (!(*buffer = intMalloc_dist(need)))
        ABORT(what);
    *capacity = need;
}

static void
pdgstrs3d_symldl_grow_int_buffer(int **buffer, int_t *capacity,
                                 int_t need, const char *what)
{
    if (need <= *capacity)
        return;
    if (*buffer)
        SUPERLU_FREE(*buffer);
    if (!(*buffer = (int *) SUPERLU_MALLOC(pdgstrs3d_checked_product(
              (size_t) need, sizeof(int), what))))
        ABORT(what);
    *capacity = need;
}

static void
pdgstrs3d_symldl_grow_double_buffer(double **buffer, int_t *capacity,
                                    int_t need, const char *what)
{
    if (need <= *capacity)
        return;
    if (*buffer)
        SUPERLU_FREE(*buffer);
    if (!(*buffer = doubleMalloc_dist(need)))
        ABORT(what);
    *capacity = need;
}

static void
pdgstrs3d_symldl_grow_request_buffer(MPI_Request **buffer, int_t *capacity,
                                     int_t need, const char *what)
{
    if (need <= *capacity)
        return;
    if (*buffer)
        SUPERLU_FREE(*buffer);
    if (!(*buffer = (MPI_Request *) SUPERLU_MALLOC(
              pdgstrs3d_checked_product((size_t) need,
                                        sizeof(MPI_Request), what))))
        ABORT(what);
    *capacity = need;
}

static int *
pdgstrs3d_symldl_diag_owners(int_t nsupers,
                             dtrf3Dpartition_t *trf3Dpartition,
                             gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    MPI_Comm solve_comm = grid3d->comm;
    int rank;
    int *local_owner;
    int *owner;
    int_t k;
    size_t owner_bytes = pdgstrs3d_checked_product((size_t) nsupers,
                                                   sizeof(int),
                                                   "SymLDL owner table");

    if (trf3Dpartition != NULL && trf3Dpartition->symV2DiagOwner != NULL) {
        if (!(owner = (int *) SUPERLU_MALLOC(owner_bytes)))
            ABORT("Malloc fails for SymLDL owner table.");
        memcpy(owner, trf3Dpartition->symV2DiagOwner, owner_bytes);
        return owner;
    }

    MPI_Comm_rank(solve_comm, &rank);
    if (!(local_owner = (int *) SUPERLU_MALLOC(owner_bytes)))
        ABORT("Malloc fails for SymLDL owner workspace.");
    if (!(owner = (int *) SUPERLU_MALLOC(owner_bytes)))
        ABORT("Malloc fails for SymLDL owner table.");

    for (k = 0; k < nsupers; ++k) {
        int diag_2d_rank = PNUM(PROW(k, grid), PCOL(k, grid), grid);
        local_owner[k] = (grid3d->zscp.Iam == 0 && grid->iam == diag_2d_rank)
                             ? rank
                             : INT_MAX;
    }

    MPI_Allreduce(local_owner, owner, pdgstrs3d_symldl_count_to_int(nsupers,
                  "SymLDL owner table"), MPI_INT, MPI_MIN, solve_comm);
    for (k = 0; k < nsupers; ++k)
        if (owner[k] == INT_MAX)
            ABORT("SymLDL solve could not identify a diagonal owner.");

    SUPERLU_FREE(local_owner);
    return owner;
}

typedef struct {
    int_t node;
    int_t pos;
} pdgstrs3d_symldl_node_order_t;

typedef struct {
    MPI_Comm comm;
    int active;
    int rank;
    int nprocs;
    int *global_to_local;
} pdgstrs3d_symldl_tree_comm_t;

typedef struct {
    int has_panel;
    int has_diag;
    int_t nsupr;
    int_t diag_luptr;
    int_t nblocks;
    int_t row_count;
    int_t lusup_count;
    double *lusup;
    int_t *block_luptr;
    int_t *block_nbrow;
    int_t *block_row_start;
    int_t *rows;
    int *row_dest_global;
} pdgstrs3d_symldl_panel_meta_t;

typedef struct {
    int active;
    int nprocs;
    int total_send;
    int total_recv;
    int total_send_vals;
    int total_recv_vals;
    int needs_xk;
    int xk_receiver_count;
    int has_diag;
    int diag_rank_count;
    int *counts;
    int *send_counts;
    int *send_displs;
    int *recv_counts;
    int *recv_displs;
    int *send_val_counts;
    int *send_val_displs;
    int *recv_val_counts;
    int *recv_val_displs;
    int *row_to_send_pos;
    int *send_seq;
    int *xk_receivers;
    int *diag_ranks;
    int_t *recv_rows;
} pdgstrs3d_symldl_comm_meta_t;

typedef struct {
    int_t nlevels;
    int_t *level_ptr;
    int_t *nodes;
} pdgstrs3d_symldl_level_schedule_t;

typedef struct {
    double *values;
    unsigned char *valid;
    int_t row_count;
} pdgstrs3d_symldl_x_cache_t;

typedef struct {
    double *x;
    double *xk_buf;
    double *diag_send_buf;
    double *diag_buf;
    double *delta_send_buf;
    double *delta_buf;
    double *gemm_buf;
    double *rhs_buf;
    double *send_vals_buf;
    double *recv_vals_buf;
    double *row_values_buf;
    double *request_values_buf;
    double *recv_request_values_buf;
    double *delta_recv_buf;
    MPI_Request *comm_reqs;
    struct pdgstrs3d_symldl_node_ctx_s *level_ctxs;
    int_t x_cap;
    int_t xk_cap;
    int_t diag_send_cap;
    int_t diag_cap;
    int_t delta_send_cap;
    int_t delta_cap;
    int_t gemm_cap;
    int_t rhs_cap;
    int_t send_vals_cap;
    int_t recv_vals_cap;
    int_t row_values_cap;
    int_t request_values_cap;
    int_t recv_request_values_cap;
    int_t delta_recv_cap;
    int_t comm_reqs_cap;
    int_t level_ctxs_cap;
} pdgstrs3d_symldl_workspace_t;

typedef struct {
    int_t n;
    int_t nsupers;
    int nrhs;
    int global_nprocs;
    int nprow;
    int npcol;
    int znp;
    int superlu_acc_offload;
    int superlu_n_gemm;
    int numForests;
    int_t max_panel_block_rows;
    Glu_persist_t *Glu_persist;
    dLocalLU_t *Llu;
    dtrf3Dpartition_t *trf3Dpartition;
    int *supernodeMask;
    int *diag_owner;
    int_t *solve_order;
    pdgstrs3d_symldl_level_schedule_t solve_schedule;
    pdgstrs3d_symldl_tree_comm_t *tree_comms;
    pdgstrs3d_symldl_panel_meta_t *panel_meta;
    pdgstrs3d_symldl_comm_meta_t *comm_meta;
    pdgstrs3d_symldl_x_cache_t *x_cache;
    pdgstrs3d_symldl_workspace_t work;
    void *gpu_state;
    void *factor_gpu_handle;
    double cpu_blas_ops;
    double gpu_blas_ops;
    double gpu_panel_ops;
    double below_threshold_ops;
    double cpu_blas_calls;
    double gpu_blas_calls;
    double gpu_panel_calls;
    double below_threshold_calls;
    double host_panel_copy_time;
    double gpu_panel_import_time;
    double gpu_schedule_upload_time;
    double x_cache_fill_time;
    double x_cache_replicated_bytes;
    double x_cache_avoided_request_bytes;
    double x_cache_hits;
    double x_cache_misses;
    double x_cache_panels;
    int factor_gpu_synchronized;
    int reused;
} pdgstrs3d_symldl_solve_meta_t;

typedef struct {
    double metadata;
    double workspace;
    double b_to_x;
    double forward_xk;
    double forward_compute;
    double forward_values;
    double forward_apply;
    double diag_comm;
    double diag_compute;
    double x_cache_fill;
    double backward_values;
    double backward_compute;
    double backward_delta;
    double x_to_b;
    double gpu_h2d;
    double gpu_compute;
    double gpu_d2h;
} pdgstrs3d_symldl_timer_t;

typedef struct pdgstrs3d_symldl_node_ctx_s {
    int active;
    int_t k;
    int_t ksupc;
    int xk_count;
    int delta_count;
    int root_rank;
    int solve_rank;
    MPI_Comm solve_comm;
    pdgstrs3d_symldl_panel_meta_t *kmeta;
    pdgstrs3d_symldl_comm_meta_t *cmeta;
    double *xk_buf;
    double *send_vals;
    double *recv_vals;
    double *row_values;
    double *request_values;
    double *recv_request_values;
    double *delta_send_buf;
    double *delta_buf;
    double *delta_recv_buf;
} pdgstrs3d_symldl_node_ctx_t;

static int
pdgstrs3d_symldl_gpu_solve_allowed(pdgstrs3d_symldl_solve_meta_t *meta);

static double
pdgstrs3d_symldl_gpu_solve_min_ops(pdgstrs3d_symldl_solve_meta_t *meta);

static void
pdgstrs3d_symldl_reset_offload_stats(pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL)
        return;
    meta->cpu_blas_ops = 0.0;
    meta->gpu_blas_ops = 0.0;
    meta->gpu_panel_ops = 0.0;
    meta->below_threshold_ops = 0.0;
    meta->cpu_blas_calls = 0.0;
    meta->gpu_blas_calls = 0.0;
    meta->gpu_panel_calls = 0.0;
    meta->below_threshold_calls = 0.0;
    meta->host_panel_copy_time = 0.0;
    meta->gpu_panel_import_time = 0.0;
    meta->gpu_schedule_upload_time = 0.0;
    meta->x_cache_fill_time = 0.0;
    meta->x_cache_replicated_bytes = 0.0;
    meta->x_cache_avoided_request_bytes = 0.0;
    meta->x_cache_hits = 0.0;
    meta->x_cache_misses = 0.0;
    meta->x_cache_panels = 0.0;
    meta->factor_gpu_synchronized = 0;
}

static void
pdgstrs3d_symldl_note_cpu_blas(pdgstrs3d_symldl_solve_meta_t *meta,
                               double ops)
{
    if (meta == NULL)
        return;
    meta->cpu_blas_ops += ops;
    meta->cpu_blas_calls += 1.0;
    if (pdgstrs3d_symldl_gpu_solve_allowed(meta) &&
        ops < pdgstrs3d_symldl_gpu_solve_min_ops(meta)) {
        meta->below_threshold_ops += ops;
        meta->below_threshold_calls += 1.0;
    }
}

static void
pdgstrs3d_symldl_note_gpu_blas(pdgstrs3d_symldl_solve_meta_t *meta,
                               double ops)
{
    if (meta == NULL)
        return;
    meta->gpu_blas_ops += ops;
    meta->gpu_blas_calls += 1.0;
}

static void
pdgstrs3d_symldl_note_gpu_panel(pdgstrs3d_symldl_solve_meta_t *meta,
                                double ops)
{
    if (meta == NULL)
        return;
    meta->gpu_panel_ops += ops;
    meta->gpu_panel_calls += 1.0;
}

static void
pdgstrs3d_symldl_note_cpu_direct_panel(pdgstrs3d_symldl_solve_meta_t *meta,
                                       double ops, double calls)
{
    if (meta == NULL || calls == 0.0)
        return;
    meta->cpu_blas_ops += ops;
    meta->cpu_blas_calls += calls;
    if (pdgstrs3d_symldl_gpu_solve_allowed(meta)) {
        meta->below_threshold_ops += ops;
        meta->below_threshold_calls += calls;
    }
}

static int
pdgstrs3d_symldl_gpu_solve_allowed(pdgstrs3d_symldl_solve_meta_t *meta)
{
    const char *override_name = "GPU3DV2_SYM_SOLVE_GPU";
    if (meta == NULL || !meta->superlu_acc_offload)
        return 0;
    if (pdgstrs3d_symldl_env_is_set(override_name))
        return pdgstrs3d_symldl_env_enabled(override_name);
    return 1;
}

static double
pdgstrs3d_symldl_gpu_solve_min_ops(pdgstrs3d_symldl_solve_meta_t *meta)
{
    return meta != NULL && meta->superlu_n_gemm >= 0
               ? (double) meta->superlu_n_gemm
               : 0.0;
}

static int
pdgstrs3d_symldl_should_gpu_ops(pdgstrs3d_symldl_solve_meta_t *meta,
                                double ops)
{
    return pdgstrs3d_symldl_gpu_solve_allowed(meta) &&
           ops >= pdgstrs3d_symldl_gpu_solve_min_ops(meta);
}

static double
pdgstrs3d_symldl_panel_solve_ops(pdgstrs3d_symldl_panel_meta_t *kmeta,
                                 int_t ksupc, int nrhs);

static double
pdgstrs3d_symldl_diag_solve_ops(pdgstrs3d_symldl_panel_meta_t *kmeta,
                                int_t ksupc, int nrhs);

static void
pdgstrs3d_symldl_dispatch_gemm(pdgstrs3d_symldl_solve_meta_t *meta,
                               int_t panel, int_t a_offset,
                               const char *transa, const char *transb,
                               int_t m, int_t n, int_t kdim,
                               double alpha, double *a, int_t lda,
                               double *b, int_t ldb,
                               double beta, double *c, int_t ldc)
{
    double ops = 2.0 * (double) m * (double) n * (double) kdim;
    if (pdgstrs3d_symldl_should_gpu_ops(meta, ops)) {
#if defined(GPU_ACC)
        if (meta == NULL || meta->gpu_state == NULL)
            ABORT("SymLDL GPU solve state is missing for eligible GPU work.");
        int ierr = dSymLDLSolveGPUGemm((dSymLDLSolveGPU_Handle) meta->gpu_state,
                                       panel, a_offset, transa[0], transb[0],
                                       m, n, kdim, alpha, lda, b, ldb,
                                       beta, c, ldc);
        if (ierr != 0)
            ABORT("SymLDL GPU solve GEMM failed.");
        pdgstrs3d_symldl_note_gpu_blas(meta, ops);
        (void) a;
        return;
#else
        ABORT("SymLDL GPU solve requires a CUDA build.");
#endif
    }

    pdgstrs3d_symldl_note_cpu_blas(meta, ops);
    pdgstrs3d_symldl_dgemm(transa, transb, m, n, kdim, alpha, a, lda,
                           b, ldb, beta, c, ldc);
}

static int
pdgstrs3d_symldl_try_gpu_forward_panel(pdgstrs3d_symldl_solve_meta_t *meta,
                                       int_t panel,
                                       pdgstrs3d_symldl_panel_meta_t *kmeta,
                                       int_t ksupc, int nrhs,
                                       const double *xk, int total_send,
                                       double *send_vals)
{
    if (!pdgstrs3d_symldl_gpu_solve_allowed(meta))
        return 0;
    if (!pdgstrs3d_symldl_should_gpu_ops(meta,
            pdgstrs3d_symldl_panel_solve_ops(kmeta, ksupc, nrhs)))
        return 0;
#if defined(GPU_ACC)
    if (meta == NULL || meta->gpu_state == NULL)
        ABORT("SymLDL GPU solve state is missing for eligible forward panel work.");
    if (kmeta == NULL || !kmeta->has_panel || kmeta->nblocks <= 0 ||
        total_send <= 0)
        return 1;
    int ierr = dSymLDLSolveGPUForwardPanel(
        (dSymLDLSolveGPU_Handle) meta->gpu_state, panel, ksupc, nrhs,
        kmeta->nsupr, kmeta->nblocks, kmeta->block_luptr,
        kmeta->block_nbrow, kmeta->block_row_start, xk, total_send,
        send_vals);
    if (ierr != 0)
        ABORT("SymLDL GPU solve forward panel failed.");
    pdgstrs3d_symldl_note_gpu_panel(
        meta, pdgstrs3d_symldl_panel_solve_ops(kmeta, ksupc, nrhs));
    return 1;
#else
    (void) meta;
    (void) panel;
    (void) kmeta;
    (void) ksupc;
    (void) nrhs;
    (void) xk;
    (void) total_send;
    (void) send_vals;
    ABORT("SymLDL GPU solve requires a CUDA build.");
    return 0;
#endif
}

static int
pdgstrs3d_symldl_try_gpu_backward_panel(pdgstrs3d_symldl_solve_meta_t *meta,
                                        int_t panel,
                                        pdgstrs3d_symldl_panel_meta_t *kmeta,
                                        int_t ksupc, int nrhs,
                                        const double *row_values,
                                        double *delta_send)
{
    if (!pdgstrs3d_symldl_gpu_solve_allowed(meta))
        return 0;
    if (!pdgstrs3d_symldl_should_gpu_ops(meta,
            pdgstrs3d_symldl_panel_solve_ops(kmeta, ksupc, nrhs)))
        return 0;
#if defined(GPU_ACC)
    if (meta == NULL || meta->gpu_state == NULL)
        ABORT("SymLDL GPU solve state is missing for eligible backward panel work.");
    if (kmeta == NULL || !kmeta->has_panel || kmeta->nblocks <= 0 ||
        kmeta->row_count <= 0)
        return 1;
    int ierr = dSymLDLSolveGPUBackwardPanel(
        (dSymLDLSolveGPU_Handle) meta->gpu_state, panel, ksupc, nrhs,
        kmeta->nsupr, kmeta->nblocks, kmeta->block_luptr,
        kmeta->block_nbrow, kmeta->row_count, row_values, delta_send);
    if (ierr != 0)
        ABORT("SymLDL GPU solve backward panel failed.");
    pdgstrs3d_symldl_note_gpu_panel(
        meta, pdgstrs3d_symldl_panel_solve_ops(kmeta, ksupc, nrhs));
    return 1;
#else
    (void) meta;
    (void) panel;
    (void) kmeta;
    (void) ksupc;
    (void) nrhs;
    (void) row_values;
    (void) delta_send;
    ABORT("SymLDL GPU solve requires a CUDA build.");
    return 0;
#endif
}

static int
pdgstrs3d_symldl_use_direct_cpu_panel(pdgstrs3d_symldl_solve_meta_t *meta,
                                      double ops, int nrhs)
{
    (void) nrhs;
    return ops < pdgstrs3d_symldl_gpu_solve_min_ops(meta);
}

static void
pdgstrs3d_symldl_cpu_forward_panel_direct(
    pdgstrs3d_symldl_panel_meta_t *kmeta,
    pdgstrs3d_symldl_comm_meta_t *cmeta, int_t ksupc, int nrhs,
    const double *xk, double *send_vals)
{
    double *lusup = kmeta->lusup;
    int_t nsupr = kmeta->nsupr;

    if (nrhs == 1) {
        for (int_t block = 0; block < kmeta->nblocks; ++block) {
            int_t nbrow = kmeta->block_nbrow[block];
            int_t row_start = kmeta->block_row_start[block];
            int_t luptr = kmeta->block_luptr[block];
            for (int_t r = 0; r < nbrow; ++r) {
                double sum = 0.0;
                double *a = &lusup[luptr + r];
                for (int_t c = 0; c < ksupc; ++c)
                    sum += a[c * nsupr] * xk[c];
                send_vals[cmeta->row_to_send_pos[row_start + r]] = -sum;
            }
        }
        return;
    }

    for (int_t block = 0; block < kmeta->nblocks; ++block) {
        int_t nbrow = kmeta->block_nbrow[block];
        int_t row_start = kmeta->block_row_start[block];
        int_t luptr = kmeta->block_luptr[block];
        for (int_t r = 0; r < nbrow; ++r) {
            int pos = cmeta->row_to_send_pos[row_start + r];
            double *a = &lusup[luptr + r];
            for (int rhs = 0; rhs < nrhs; ++rhs) {
                double sum = 0.0;
                const double *xr = &xk[(int_t) rhs * ksupc];
                for (int_t c = 0; c < ksupc; ++c)
                    sum += a[c * nsupr] * xr[c];
                send_vals[(int_t) pos * nrhs + rhs] = -sum;
            }
        }
    }
}

static void
pdgstrs3d_symldl_cpu_backward_panel_direct(
    pdgstrs3d_symldl_panel_meta_t *kmeta, int_t ksupc, int nrhs,
    const double *row_values, double *delta_send)
{
    double *lusup = kmeta->lusup;
    int_t nsupr = kmeta->nsupr;

    if (nrhs == 1) {
        for (int_t block = 0; block < kmeta->nblocks; ++block) {
            int_t nbrow = kmeta->block_nbrow[block];
            int_t row_start = kmeta->block_row_start[block];
            int_t luptr = kmeta->block_luptr[block];
            for (int_t c = 0; c < ksupc; ++c) {
                double sum = 0.0;
                double *a = &lusup[luptr + c * nsupr];
                for (int_t r = 0; r < nbrow; ++r)
                    sum += a[r] * row_values[row_start + r];
                delta_send[c] -= sum;
            }
        }
        return;
    }

    for (int_t block = 0; block < kmeta->nblocks; ++block) {
        int_t nbrow = kmeta->block_nbrow[block];
        int_t row_start = kmeta->block_row_start[block];
        int_t luptr = kmeta->block_luptr[block];
        for (int rhs = 0; rhs < nrhs; ++rhs) {
            const double *rows = &row_values[(int_t) row_start * nrhs + rhs];
            double *delta = &delta_send[(int_t) rhs * ksupc];
            for (int_t c = 0; c < ksupc; ++c) {
                double sum = 0.0;
                double *a = &lusup[luptr + c * nsupr];
                for (int_t r = 0; r < nbrow; ++r)
                    sum += a[r] * rows[(int_t) r * nrhs];
                delta[c] -= sum;
            }
        }
    }
}

static double
pdgstrs3d_symldl_panel_solve_ops(pdgstrs3d_symldl_panel_meta_t *kmeta,
                                 int_t ksupc, int nrhs)
{
    double ops = 0.0;
    if (kmeta == NULL)
        return 0.0;
    for (int_t block = 0; block < kmeta->nblocks; ++block)
        ops += 2.0 * (double) kmeta->block_nbrow[block] *
               (double) nrhs * (double) ksupc;
    return ops;
}

static double
pdgstrs3d_symldl_diag_solve_ops(pdgstrs3d_symldl_panel_meta_t *kmeta,
                                int_t ksupc, int nrhs)
{
    if (kmeta == NULL || !kmeta->has_diag)
        return 0.0;
    return 2.0 * (double) ksupc * (double) ksupc * (double) nrhs;
}

static int
pdgstrs3d_symldl_node_order_cmp(const void *a, const void *b)
{
    const pdgstrs3d_symldl_node_order_t *oa =
        (const pdgstrs3d_symldl_node_order_t *) a;
    const pdgstrs3d_symldl_node_order_t *ob =
        (const pdgstrs3d_symldl_node_order_t *) b;

    if (oa->pos < ob->pos) return -1;
    if (oa->pos > ob->pos) return 1;
    if (oa->node < ob->node) return -1;
    if (oa->node > ob->node) return 1;
    return 0;
}

static int_t *
pdgstrs3d_symldl_tree_order(int_t nsupers,
                            dtrf3Dpartition_t *trf3Dpartition,
                            gridinfo3d_t *grid3d)
{
    int_t *order;
    int_t *local_pos = NULL;
    int_t *global_pos = NULL;
    pdgstrs3d_symldl_node_order_t *entries = NULL;
    int_t sentinel = (int_t) INT_MAX;

    if (trf3Dpartition == NULL ||
        trf3Dpartition->sForests == NULL ||
        trf3Dpartition->myTreeIdxs == NULL ||
        trf3Dpartition->myZeroTrIdxs == NULL)
        ABORT("SymLDL solve requires an LDL-native forest schedule.");

    if (!(order = intMalloc_dist(nsupers)))
        ABORT("Malloc fails for SymLDL solve order.");
    if (!(local_pos = intMalloc_dist(nsupers)) ||
        !(global_pos = intMalloc_dist(nsupers)) ||
        !(entries = (pdgstrs3d_symldl_node_order_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) nsupers,
                              sizeof(pdgstrs3d_symldl_node_order_t),
                              "SymLDL tree solve order"))))
        ABORT("Malloc fails for SymLDL tree solve order.");

    for (int_t k = 0; k < nsupers; ++k)
        local_pos[k] = sentinel;

    int maxLvl = trf3Dpartition->maxLvl > 0
                     ? trf3Dpartition->maxLvl
                     : log2i(grid3d->zscp.Np) + 1;
    int_t pos = 0;

    for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {
        if (trf3Dpartition->myZeroTrIdxs[ilvl])
            continue;
        int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
        sForest_t *sforest = trf3Dpartition->sForests[tree];
        if (sforest == NULL)
            continue;
        for (int_t k0 = 0; k0 < sforest->nNodes; ++k0) {
            int_t node = sforest->nodeList[k0];
            if (node >= 0 && node < nsupers && local_pos[node] == sentinel)
                local_pos[node] = pos;
            ++pos;
        }
    }

    MPI_Allreduce(local_pos, global_pos,
                  pdgstrs3d_symldl_count_to_int(nsupers,
                                                "SymLDL tree solve order"),
                  mpi_int_t, MPI_MIN, grid3d->comm);
    for (int_t k = 0; k < nsupers; ++k) {
        entries[k].node = k;
        entries[k].pos = global_pos[k];
    }
    qsort(entries, (size_t) nsupers, sizeof(*entries),
          pdgstrs3d_symldl_node_order_cmp);
    for (int_t k = 0; k < nsupers; ++k)
        order[k] = entries[k].node;

    SUPERLU_FREE(entries);
    SUPERLU_FREE(global_pos);
    SUPERLU_FREE(local_pos);
    return order;
}

typedef struct {
    int_t node;
    int_t level;
    int_t order;
    double cost;
} pdgstrs3d_symldl_sched_entry_t;

static int
pdgstrs3d_symldl_sched_entry_cmp(const void *a, const void *b)
{
    const pdgstrs3d_symldl_sched_entry_t *ea =
        (const pdgstrs3d_symldl_sched_entry_t *) a;
    const pdgstrs3d_symldl_sched_entry_t *eb =
        (const pdgstrs3d_symldl_sched_entry_t *) b;

    if (ea->level < eb->level) return -1;
    if (ea->level > eb->level) return 1;
    if (ea->cost > eb->cost) return -1;
    if (ea->cost < eb->cost) return 1;
    if (ea->order < eb->order) return -1;
    if (ea->order > eb->order) return 1;
    if (ea->node < eb->node) return -1;
    if (ea->node > eb->node) return 1;
    return 0;
}

static void
pdgstrs3d_symldl_level_schedule_free(pdgstrs3d_symldl_level_schedule_t *schedule)
{
    if (schedule == NULL)
        return;
    if (schedule->nodes) SUPERLU_FREE(schedule->nodes);
    if (schedule->level_ptr) SUPERLU_FREE(schedule->level_ptr);
    memset(schedule, 0, sizeof(*schedule));
}

static pdgstrs3d_symldl_level_schedule_t
pdgstrs3d_symldl_level_schedule_create(
    int_t nsupers, int_t *solve_order,
    pdgstrs3d_symldl_panel_meta_t *panel_meta,
    gridinfo3d_t *grid3d, Glu_persist_t *Glu_persist)
{
    pdgstrs3d_symldl_level_schedule_t schedule;
    int_t *local_edges = NULL;
    int_t *edges = NULL;
    int *recv_counts = NULL;
    int *recv_displs = NULL;
    int_t *order_pos = NULL;
    int_t *edge_counts = NULL;
    int_t *edge_ptr = NULL;
    int_t *edge_next = NULL;
    int_t *edge_succ = NULL;
    int_t *indegree = NULL;
    int_t *topo_queue = NULL;
    int_t *levels = NULL;
    double *local_cost = NULL;
    double *global_cost = NULL;
    pdgstrs3d_symldl_sched_entry_t *entries = NULL;
    int global_nprocs;
    int local_pair_count_i;
    int total_pair_count = 0;
    int valid_schedule = 1;
    int global_valid_schedule = 1;
    int invalid_kind = 0;
    int_t invalid_pred = -1;
    int_t invalid_succ = -1;
    int_t invalid_pred_pos = -1;
    int_t invalid_succ_pos = -1;
    int_t *xsup = Glu_persist->xsup;
    int_t local_edges_count = 0;
    int_t local_edges_fill = 0;
    int_t total_edges = 0;

    memset(&schedule, 0, sizeof(schedule));
    MPI_Comm_size(grid3d->comm, &global_nprocs);

    for (int_t k = 0; k < nsupers; ++k)
        if (panel_meta[k].has_panel)
            local_edges_count += panel_meta[k].nblocks;

    if (local_edges_count > 0) {
        if (!(local_edges = intMalloc_dist(2 * local_edges_count)))
            ABORT("Malloc fails for SymLDL solve dependency edges.");
    }

    if (!(local_cost = doubleCalloc_dist(nsupers)) ||
        !(global_cost = doubleCalloc_dist(nsupers)))
        ABORT("Calloc fails for SymLDL solve cost metadata.");

    for (int_t k = 0; k < nsupers; ++k) {
        pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
        if (!kmeta->has_panel)
            continue;
        int_t ksupc = SuperSize(k);
        local_cost[k] += (double) ksupc * (double) ksupc;
        for (int_t block = 0; block < kmeta->nblocks; ++block) {
            int_t row_start = kmeta->block_row_start[block];
            int_t nbrow = kmeta->block_nbrow[block];
            if (nbrow <= 0)
                continue;
            int_t succ = Glu_persist->supno[kmeta->rows[row_start]];
            local_edges[2 * local_edges_fill] = k;
            local_edges[2 * local_edges_fill + 1] = succ;
            ++local_edges_fill;
            local_cost[k] += (double) nbrow * (double) ksupc;
        }
    }
    local_edges_count = local_edges_fill;

    if (!(recv_counts = (int *) SUPERLU_MALLOC(
              pdgstrs3d_checked_product((size_t) global_nprocs, sizeof(int),
                                        "SymLDL solve edge counts"))) ||
        !(recv_displs = (int *) SUPERLU_MALLOC(
              pdgstrs3d_checked_product((size_t) global_nprocs, sizeof(int),
                                        "SymLDL solve edge displacements"))))
        ABORT("Malloc fails for SymLDL solve edge metadata.");

    local_pair_count_i = pdgstrs3d_symldl_count_to_int(
        2 * local_edges_count, "SymLDL solve dependency edge count");
    MPI_Allgather(&local_pair_count_i, 1, MPI_INT, recv_counts, 1, MPI_INT,
                  grid3d->comm);
    pdgstrs3d_symldl_counts_to_displs(global_nprocs, recv_counts,
                                      recv_displs, &total_pair_count);
    if (total_pair_count % 2 != 0)
        ABORT("SymLDL solve dependency edge metadata is malformed.");
    total_edges = total_pair_count / 2;

    if (total_edges > 0) {
        if (!(edges = intMalloc_dist(2 * total_edges)))
            ABORT("Malloc fails for SymLDL solve global dependency edges.");
    }

    {
        int_t dummy_edge[2] = {0, 0};
        MPI_Allgatherv(local_edges_count ? local_edges : dummy_edge,
                       local_pair_count_i, mpi_int_t,
                       total_edges ? edges : dummy_edge,
                       recv_counts, recv_displs, mpi_int_t, grid3d->comm);
    }
    MPI_Allreduce(local_cost, global_cost,
                  pdgstrs3d_symldl_count_to_int(nsupers,
                                                "SymLDL solve cost metadata"),
                  MPI_DOUBLE, MPI_SUM, grid3d->comm);

    if (!(order_pos = intMalloc_dist(nsupers)) ||
        !(edge_counts = intCalloc_dist(nsupers)) ||
        !(edge_ptr = intMalloc_dist(nsupers + 1)) ||
        !(indegree = intCalloc_dist(nsupers)) ||
        !(topo_queue = intMalloc_dist(nsupers)) ||
        !(levels = intCalloc_dist(nsupers)) ||
        !(entries = (pdgstrs3d_symldl_sched_entry_t *) SUPERLU_MALLOC(
              pdgstrs3d_checked_product((size_t) nsupers, sizeof(*entries),
                                        "SymLDL solve level entries"))))
        ABORT("Malloc fails for SymLDL solve level metadata.");

    for (int_t pos = 0; pos < nsupers; ++pos)
        order_pos[pos] = -1;
    for (int_t pos = 0; pos < nsupers; ++pos) {
        int_t node = solve_order[pos];
        if (node < 0 || node >= nsupers ||
            (node >= 0 && node < nsupers && order_pos[node] >= 0)) {
            if (!invalid_kind) {
                invalid_kind = 3;
                invalid_pred = node;
                invalid_pred_pos = pos;
            }
            valid_schedule = 0;
            continue;
        }
        order_pos[node] = pos;
    }
    MPI_Allreduce(&valid_schedule, &global_valid_schedule, 1, MPI_INT,
                  MPI_MIN, grid3d->comm);
    if (!global_valid_schedule) {
        int global_rank = -1;
        MPI_Comm_rank(grid3d->comm, &global_rank);
        if (!valid_schedule) {
            fprintf(stderr,
                    "SymLDL solve forest order is invalid on rank %d: "
                    "kind=%d node=%lld pos=%lld nsupers=%lld\n",
                    global_rank, invalid_kind, (long long) invalid_pred,
                    (long long) invalid_pred_pos, (long long) nsupers);
            fflush(stderr);
        }
        ABORT("SymLDL solve forest order is invalid.");
    }

    valid_schedule = 1;
    global_valid_schedule = 1;
    invalid_kind = 0;
    invalid_pred = -1;
    invalid_succ = -1;
    invalid_pred_pos = -1;
    invalid_succ_pos = -1;
    for (int_t e = 0; e < total_edges; ++e) {
        int_t pred = edges[2 * e];
        int_t succ = edges[2 * e + 1];
        if (pred < 0 || pred >= nsupers || succ < 0 || succ >= nsupers) {
            if (!invalid_kind) {
                invalid_kind = 1;
                invalid_pred = pred;
                invalid_succ = succ;
            }
            valid_schedule = 0;
            continue;
        }
        if (pred == succ) {
            if (!invalid_kind) {
                invalid_kind = 2;
                invalid_pred = pred;
                invalid_succ = succ;
                invalid_pred_pos = order_pos[pred];
                invalid_succ_pos = order_pos[succ];
            }
            valid_schedule = 0;
            continue;
        }
        ++edge_counts[pred];
        ++indegree[succ];
    }
    MPI_Allreduce(&valid_schedule, &global_valid_schedule, 1, MPI_INT,
                  MPI_MIN, grid3d->comm);
    if (!global_valid_schedule) {
        int global_rank = -1;
        MPI_Comm_rank(grid3d->comm, &global_rank);
        if (!valid_schedule) {
            fprintf(stderr,
                    "SymLDL solve dependency graph is invalid on rank %d: "
                    "kind=%d pred=%lld succ=%lld pred_pos=%lld succ_pos=%lld "
                    "nsupers=%lld total_edges=%lld\n",
                    global_rank, invalid_kind, (long long) invalid_pred,
                    (long long) invalid_succ, (long long) invalid_pred_pos,
                    (long long) invalid_succ_pos, (long long) nsupers,
                    (long long) total_edges);
            fflush(stderr);
        }
        ABORT("SymLDL solve dependency graph is invalid.");
    }

    edge_ptr[0] = 0;
    for (int_t k = 0; k < nsupers; ++k)
        edge_ptr[k + 1] = edge_ptr[k] + edge_counts[k];
    if (!(edge_succ = intMalloc_dist(SUPERLU_MAX(total_edges, (int_t) 1))) ||
        !(edge_next = intMalloc_dist(nsupers)))
        ABORT("Malloc fails for SymLDL solve adjacency.");
    memcpy(edge_next, edge_ptr,
           pdgstrs3d_checked_alloc_bytes(nsupers, sizeof(int_t),
                                         "SymLDL solve adjacency cursor"));
    for (int_t e = 0; e < total_edges; ++e) {
        int_t pred = edges[2 * e];
        int_t succ = edges[2 * e + 1];
        edge_succ[edge_next[pred]++] = succ;
    }

    int_t max_level = 0;
    int_t queue_head = 0;
    int_t queue_tail = 0;
    for (int_t k = 0; k < nsupers; ++k)
        if (indegree[k] == 0)
            topo_queue[queue_tail++] = k;

    while (queue_head < queue_tail) {
        int_t k = topo_queue[queue_head++];
        max_level = SUPERLU_MAX(max_level, levels[k]);
        for (int_t p = edge_ptr[k]; p < edge_ptr[k + 1]; ++p) {
            int_t succ = edge_succ[p];
            levels[succ] = SUPERLU_MAX(levels[succ], levels[k] + 1);
            --indegree[succ];
            if (indegree[succ] == 0)
                topo_queue[queue_tail++] = succ;
        }
    }
    if (queue_tail != nsupers) {
        int global_rank = -1;
        int_t first_blocked = -1;
        int_t first_indegree = -1;
        MPI_Comm_rank(grid3d->comm, &global_rank);
        for (int_t k = 0; k < nsupers; ++k) {
            if (indegree[k] > 0) {
                first_blocked = k;
                first_indegree = indegree[k];
                break;
            }
        }
        fprintf(stderr,
                "SymLDL solve dependency graph is cyclic on rank %d: "
                "processed=%lld nsupers=%lld first_blocked=%lld "
                "remaining_indegree=%lld total_edges=%lld\n",
                global_rank, (long long) queue_tail, (long long) nsupers,
                (long long) first_blocked, (long long) first_indegree,
                (long long) total_edges);
        fflush(stderr);
        ABORT("SymLDL solve dependency graph is cyclic.");
    }

    schedule.nlevels = max_level + 1;
    if (!(schedule.level_ptr = intCalloc_dist(schedule.nlevels + 1)) ||
        !(schedule.nodes = intMalloc_dist(nsupers)))
        ABORT("Malloc fails for SymLDL solve level schedule.");

    for (int_t k = 0; k < nsupers; ++k) {
        entries[k].node = k;
        entries[k].level = levels[k];
        entries[k].order = order_pos[k];
        entries[k].cost = global_cost[k];
        ++schedule.level_ptr[levels[k] + 1];
    }
    for (int_t level = 0; level < schedule.nlevels; ++level)
        schedule.level_ptr[level + 1] += schedule.level_ptr[level];
    qsort(entries, (size_t) nsupers, sizeof(*entries),
          pdgstrs3d_symldl_sched_entry_cmp);
    for (int_t i = 0; i < nsupers; ++i)
        schedule.nodes[i] = entries[i].node;

cleanup:
    if (entries) SUPERLU_FREE(entries);
    if (topo_queue) SUPERLU_FREE(topo_queue);
    if (indegree) SUPERLU_FREE(indegree);
    if (edge_next) SUPERLU_FREE(edge_next);
    if (edge_succ) SUPERLU_FREE(edge_succ);
    if (levels) SUPERLU_FREE(levels);
    if (edge_ptr) SUPERLU_FREE(edge_ptr);
    if (edge_counts) SUPERLU_FREE(edge_counts);
    if (order_pos) SUPERLU_FREE(order_pos);
    if (global_cost) SUPERLU_FREE(global_cost);
    if (local_cost) SUPERLU_FREE(local_cost);
    if (edges) SUPERLU_FREE(edges);
    if (recv_displs) SUPERLU_FREE(recv_displs);
    if (recv_counts) SUPERLU_FREE(recv_counts);
    if (local_edges) SUPERLU_FREE(local_edges);
    return schedule;
}

static int
pdgstrs3d_symldl_rank_to_tree_rank(pdgstrs3d_symldl_tree_comm_t *tree_comm,
                                   int global_rank)
{
    int out;

    if (tree_comm == NULL)
        ABORT("SymLDL solve requires a tree communicator.");
    out = tree_comm->global_to_local[global_rank];
    if (out < 0)
        ABORT("SymLDL solve tree communicator is missing a required rank.");
    return out;
}

static int
pdgstrs3d_symldl_num_forests(dtrf3Dpartition_t *trf3Dpartition,
                             gridinfo3d_t *grid3d)
{
    int maxLvl = trf3Dpartition && trf3Dpartition->maxLvl > 0
                     ? trf3Dpartition->maxLvl
                     : log2i(grid3d->zscp.Np) + 1;
    return (1 << maxLvl) - 1;
}

static pdgstrs3d_symldl_tree_comm_t *
pdgstrs3d_symldl_tree_comms_create(dtrf3Dpartition_t *trf3Dpartition,
                                   pdgstrs3d_symldl_panel_meta_t *panel_meta,
                                   int *diag_owner, int_t nsupers,
                                   gridinfo3d_t *grid3d, int global_nprocs)
{
    pdgstrs3d_symldl_tree_comm_t *tree_comms;
    int *local_active = NULL;
    int *global_active = NULL;
    int maxLvl;
    int numForests;
    int global_rank;

    if (trf3Dpartition == NULL ||
        trf3Dpartition->supernode2treeMap == NULL ||
        trf3Dpartition->myTreeIdxs == NULL ||
        trf3Dpartition->myZeroTrIdxs == NULL)
        ABORT("SymLDL solve requires LDL-native tree communicator metadata.");

    maxLvl = trf3Dpartition->maxLvl > 0
                 ? trf3Dpartition->maxLvl
                 : log2i(grid3d->zscp.Np) + 1;
    numForests = (1 << maxLvl) - 1;
    MPI_Comm_rank(grid3d->comm, &global_rank);

    size_t active_count = pdgstrs3d_checked_product((size_t) numForests,
                                                    (size_t) global_nprocs,
                                                    "SymLDL tree active ranks");
    int_t active_count_t = pdgstrs3d_checked_size_to_int_t(
        active_count, "SymLDL tree active ranks");
    int active_count_i = pdgstrs3d_symldl_count_to_int(
        active_count_t, "SymLDL tree active ranks");
    if (!(local_active = (int *) SUPERLU_MALLOC(
              pdgstrs3d_checked_alloc_bytes(active_count_t, sizeof(int),
                                            "SymLDL tree active ranks"))) ||
        !(global_active = (int *) SUPERLU_MALLOC(
              pdgstrs3d_checked_alloc_bytes(active_count_t, sizeof(int),
                                            "SymLDL tree active ranks"))))
        ABORT("Calloc fails for SymLDL tree active rank metadata.");
    memset(local_active, 0,
           pdgstrs3d_checked_alloc_bytes(active_count_t, sizeof(int),
                                         "SymLDL tree active ranks"));
    memset(global_active, 0,
           pdgstrs3d_checked_alloc_bytes(active_count_t, sizeof(int),
                                         "SymLDL tree active ranks"));

    for (int tree = 0; tree < numForests; ++tree) {
        if (grid3d->zscp.Iam == 0)
            local_active[tree * global_nprocs + global_rank] = 1;
    }

    for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {
        if (!trf3Dpartition->myZeroTrIdxs[ilvl]) {
            int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
            if (tree >= 0 && tree < numForests)
                local_active[tree * global_nprocs + global_rank] = 1;
        }
    }

    if (diag_owner != NULL) {
        for (int_t k = 0; k < nsupers; ++k) {
            int_t tree = trf3Dpartition->supernode2treeMap[k];
            int owner = diag_owner[k];
            if (tree >= 0 && tree < numForests &&
                owner >= 0 && owner < global_nprocs)
                local_active[tree * global_nprocs + owner] = 1;
        }
    }

    if (panel_meta != NULL) {
        for (int_t k = 0; k < nsupers; ++k) {
            int_t tree = trf3Dpartition->supernode2treeMap[k];
            pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
            if (tree < 0 || tree >= numForests || !kmeta->has_panel ||
                grid3d->zscp.Iam != 0)
                continue;
            local_active[tree * global_nprocs + global_rank] = 1;
            for (int_t row = 0; row < kmeta->row_count; ++row) {
                int dest = kmeta->row_dest_global[row];
                if (dest >= 0 && dest < global_nprocs)
                    local_active[tree * global_nprocs + dest] = 1;
            }
        }
    }

    MPI_Allreduce(local_active, global_active, active_count_i, MPI_INT,
                  MPI_MAX, grid3d->comm);

    if (!(tree_comms = (pdgstrs3d_symldl_tree_comm_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) numForests,
                              sizeof(pdgstrs3d_symldl_tree_comm_t),
                              "SymLDL tree communicators"))))
        ABORT("Malloc fails for SymLDL tree communicators.");

    for (int tree = 0; tree < numForests; ++tree) {
        int active = global_active[tree * global_nprocs + global_rank];

        tree_comms[tree].comm = MPI_COMM_NULL;
        tree_comms[tree].active = 0;
        tree_comms[tree].rank = -1;
        tree_comms[tree].nprocs = 0;
        tree_comms[tree].global_to_local = NULL;

        MPI_Comm_split(grid3d->comm, active ? 0 : MPI_UNDEFINED,
                       global_rank, &tree_comms[tree].comm);
        if (active) {
            int *local_to_global;

            tree_comms[tree].active = 1;
            MPI_Comm_rank(tree_comms[tree].comm, &tree_comms[tree].rank);
            MPI_Comm_size(tree_comms[tree].comm, &tree_comms[tree].nprocs);
            if (!(tree_comms[tree].global_to_local =
                      (int *) SUPERLU_MALLOC(pdgstrs3d_checked_product(
                          (size_t) global_nprocs, sizeof(int),
                          "SymLDL tree rank map"))) ||
                !(local_to_global = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product((size_t) tree_comms[tree].nprocs,
                                                sizeof(int),
                                                "SymLDL tree rank list"))))
                ABORT("Malloc fails for SymLDL tree rank maps.");
            for (int p = 0; p < global_nprocs; ++p)
                tree_comms[tree].global_to_local[p] = -1;
            MPI_Allgather(&global_rank, 1, MPI_INT, local_to_global, 1,
                          MPI_INT, tree_comms[tree].comm);
            for (int p = 0; p < tree_comms[tree].nprocs; ++p)
                tree_comms[tree].global_to_local[local_to_global[p]] = p;
            SUPERLU_FREE(local_to_global);
        }
    }

    SUPERLU_FREE(global_active);
    SUPERLU_FREE(local_active);
    return tree_comms;
}

static void
pdgstrs3d_symldl_tree_comms_free_count(pdgstrs3d_symldl_tree_comm_t *tree_comms,
                                       int numForests)
{
    if (tree_comms == NULL)
        return;

    for (int tree = 0; tree < numForests; ++tree) {
        if (tree_comms[tree].global_to_local)
            SUPERLU_FREE(tree_comms[tree].global_to_local);
        if (tree_comms[tree].comm != MPI_COMM_NULL)
            MPI_Comm_free(&tree_comms[tree].comm);
    }
    SUPERLU_FREE(tree_comms);
}

static void
pdgstrs3d_symldl_tree_comms_free(pdgstrs3d_symldl_tree_comm_t *tree_comms,
                                 dtrf3Dpartition_t *trf3Dpartition,
                                 gridinfo3d_t *grid3d)
{
    pdgstrs3d_symldl_tree_comms_free_count(
        tree_comms, pdgstrs3d_symldl_num_forests(trf3Dpartition, grid3d));
}

static pdgstrs3d_symldl_panel_meta_t *
pdgstrs3d_symldl_panel_meta_create(int_t nsupers, dLocalLU_t *Llu,
                                   Glu_persist_t *Glu_persist, gridinfo_t *grid,
                                   dtrf3Dpartition_t *trf3Dpartition,
                                   int *supernodeMask, int *diag_owner)
{
    pdgstrs3d_symldl_panel_meta_t *meta;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;

    if (!(meta = (pdgstrs3d_symldl_panel_meta_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) nsupers,
                              sizeof(pdgstrs3d_symldl_panel_meta_t),
                              "SymLDL panel metadata"))))
        ABORT("Malloc fails for SymLDL panel metadata.");

    for (int_t k = 0; k < nsupers; ++k) {
        meta[k].has_panel = 0;
        meta[k].has_diag = 0;
        meta[k].nsupr = 0;
        meta[k].diag_luptr = 0;
        meta[k].nblocks = 0;
        meta[k].row_count = 0;
        meta[k].lusup_count = 0;
        meta[k].lusup = NULL;
        meta[k].block_luptr = NULL;
        meta[k].block_nbrow = NULL;
        meta[k].block_row_start = NULL;
        meta[k].rows = NULL;
        meta[k].row_dest_global = NULL;

        int use_symv2_owner =
            trf3Dpartition != NULL &&
            trf3Dpartition->symV2PanelRoot != NULL &&
            trf3Dpartition->symV2PanelLocalIndex != NULL;
        int panel_root = use_symv2_owner
                             ? trf3Dpartition->symV2PanelRoot[k]
                             : PCOL(k, grid);

        if ((supernodeMask != NULL && !supernodeMask[k]) ||
            MYCOL(grid->iam, grid) != panel_root)
            continue;

        int_t lk_col = use_symv2_owner
                           ? trf3Dpartition->symV2PanelLocalIndex[k]
                           : LBj(k, grid);
        if (lk_col < 0)
            ABORT("SymLDL solve missing local L panel index.");
        int_t *lsub = Llu->Lrowind_bc_ptr[lk_col];
        double *lusup = Llu->Lnzval_bc_ptr[lk_col];
        if (lsub == NULL || lusup == NULL)
            continue;

        int_t lptr = BC_HEADER;
        int_t luptr = 0;
        int_t nblocks = 0;
        int_t row_count = 0;

        for (int_t lb = 0; lb < lsub[0]; ++lb) {
            int_t ik = lsub[lptr];
            int_t nbrow = lsub[lptr + 1];
            if (ik == k) {
                if (nbrow != SuperSize(k))
                    ABORT("SymLDL solve diagonal block has an unexpected size.");
                meta[k].has_diag = 1;
                meta[k].diag_luptr = luptr;
            } else {
                ++nblocks;
                row_count += nbrow;
            }
            lptr += LB_DESCRIPTOR + nbrow;
            luptr += nbrow;
        }

        meta[k].has_panel = 1;
        meta[k].nsupr = lsub[1];
        meta[k].lusup = lusup;
        meta[k].nblocks = nblocks;
        meta[k].row_count = row_count;
        meta[k].lusup_count = pdgstrs3d_checked_workspace_count(
            lsub[1], SuperSize(k), 0, 0, "SymLDL L panel values");

        if (nblocks > 0) {
            if (!(meta[k].block_luptr = intMalloc_dist(nblocks)) ||
                !(meta[k].block_nbrow = intMalloc_dist(nblocks)) ||
                !(meta[k].block_row_start = intMalloc_dist(nblocks)))
                ABORT("Malloc fails for SymLDL block metadata.");
        }
        if (row_count > 0) {
            if (!(meta[k].rows = intMalloc_dist(row_count)) ||
                !(meta[k].row_dest_global = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product((size_t) row_count,
                                                sizeof(int),
                                                "SymLDL row destination metadata"))))
                ABORT("Malloc fails for SymLDL row metadata.");
        }

        lptr = BC_HEADER;
        luptr = 0;
        int_t block = 0;
        int_t row = 0;
        for (int_t lb = 0; lb < lsub[0]; ++lb) {
            int_t ik = lsub[lptr];
            int_t nbrow = lsub[lptr + 1];
            int_t rows = lptr + LB_DESCRIPTOR;
            if (ik != k) {
                meta[k].block_luptr[block] = luptr;
                meta[k].block_nbrow[block] = nbrow;
                meta[k].block_row_start[block] = row;
                for (int_t r = 0; r < nbrow; ++r) {
                    int_t grow = lsub[rows + r];
                    meta[k].rows[row] = grow;
                    meta[k].row_dest_global[row] = diag_owner[BlockNum(grow)];
                    ++row;
                }
                ++block;
            }
            lptr += LB_DESCRIPTOR + nbrow;
            luptr += nbrow;
        }
    }

    return meta;
}

static void
pdgstrs3d_symldl_panel_meta_free(pdgstrs3d_symldl_panel_meta_t *meta,
                                 int_t nsupers)
{
    if (meta == NULL)
        return;

    for (int_t k = 0; k < nsupers; ++k) {
        if (meta[k].row_dest_global) SUPERLU_FREE(meta[k].row_dest_global);
        if (meta[k].rows) SUPERLU_FREE(meta[k].rows);
        if (meta[k].block_row_start) SUPERLU_FREE(meta[k].block_row_start);
        if (meta[k].block_nbrow) SUPERLU_FREE(meta[k].block_nbrow);
        if (meta[k].block_luptr) SUPERLU_FREE(meta[k].block_luptr);
    }
    SUPERLU_FREE(meta);
}

static void
pdgstrs3d_symldl_comm_meta_set_count_views(pdgstrs3d_symldl_comm_meta_t *meta,
                                           int nprocs)
{
    meta->send_counts = meta->counts;
    meta->send_displs = meta->send_counts + nprocs;
    meta->recv_counts = meta->send_displs + nprocs;
    meta->recv_displs = meta->recv_counts + nprocs;
    meta->send_val_counts = meta->recv_displs + nprocs;
    meta->send_val_displs = meta->send_val_counts + nprocs;
    meta->recv_val_counts = meta->send_val_displs + nprocs;
    meta->recv_val_displs = meta->recv_val_counts + nprocs;
}

static pdgstrs3d_symldl_comm_meta_t *
pdgstrs3d_symldl_comm_meta_create(int_t nsupers,
                                  pdgstrs3d_symldl_panel_meta_t *panel_meta,
                                  dtrf3Dpartition_t *trf3Dpartition,
                                  pdgstrs3d_symldl_tree_comm_t *tree_comms,
                                  gridinfo3d_t *grid3d, int global_nprocs,
                                  int nrhs, int *diag_owner)
{
    pdgstrs3d_symldl_comm_meta_t *meta;
    int_t *send_rows_buf = NULL;
    int_t send_rows_cap = 0;
    int *fill_counts = NULL;

    if (!(meta = (pdgstrs3d_symldl_comm_meta_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) nsupers,
                              sizeof(pdgstrs3d_symldl_comm_meta_t),
                              "SymLDL communication schedule"))))
        ABORT("Malloc fails for SymLDL communication schedule.");
    if (!(fill_counts = (int *) SUPERLU_MALLOC(pdgstrs3d_checked_product(
              (size_t) global_nprocs, sizeof(int),
              "SymLDL communication schedule workspace"))))
        ABORT("Malloc fails for SymLDL communication schedule workspace.");

    for (int_t k = 0; k < nsupers; ++k) {
        int_t tree = (trf3Dpartition && trf3Dpartition->supernode2treeMap)
                         ? trf3Dpartition->supernode2treeMap[k] : -1;
        pdgstrs3d_symldl_tree_comm_t *tree_comm =
            (tree_comms && tree >= 0) ? &tree_comms[tree] : NULL;
        MPI_Comm solve_comm;
        int comm_nprocs;
        pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
        pdgstrs3d_symldl_comm_meta_t *cmeta = &meta[k];
        int total_send = 0;
        int total_recv = 0;
        int root_rank;
        int local_need_xk;
        int local_has_diag;
        int local_panel_active;

        cmeta->active = 0;
        cmeta->nprocs = 0;
        cmeta->total_send = 0;
        cmeta->total_recv = 0;
        cmeta->total_send_vals = 0;
        cmeta->total_recv_vals = 0;
        cmeta->needs_xk = 0;
        cmeta->xk_receiver_count = 0;
        cmeta->has_diag = 0;
        cmeta->diag_rank_count = 0;
        cmeta->counts = NULL;
        cmeta->send_counts = NULL;
        cmeta->send_displs = NULL;
        cmeta->recv_counts = NULL;
        cmeta->recv_displs = NULL;
        cmeta->send_val_counts = NULL;
        cmeta->send_val_displs = NULL;
        cmeta->recv_val_counts = NULL;
        cmeta->recv_val_displs = NULL;
        cmeta->row_to_send_pos = NULL;
        cmeta->send_seq = NULL;
        cmeta->xk_receivers = NULL;
        cmeta->diag_ranks = NULL;
        cmeta->recv_rows = NULL;

        if (tree_comm == NULL)
            ABORT("SymLDL solve communication metadata is missing a tree communicator.");
        if (!tree_comm->active)
            continue;

        solve_comm = tree_comm->comm;
        comm_nprocs = tree_comm->nprocs;
        cmeta->active = 1;
        cmeta->nprocs = comm_nprocs;
        if (!(cmeta->counts = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) comm_nprocs,
                                            8 * sizeof(int),
                                            "SymLDL communication schedule counts"))))
            ABORT("Malloc fails for SymLDL communication schedule counts.");
        pdgstrs3d_symldl_comm_meta_set_count_views(cmeta, comm_nprocs);
        for (int p = 0; p < 8 * comm_nprocs; ++p)
            cmeta->counts[p] = 0;

        root_rank = pdgstrs3d_symldl_rank_to_tree_rank(tree_comm,
                                                       diag_owner[k]);
        local_panel_active = kmeta->has_panel && grid3d->zscp.Iam == 0;
        local_need_xk = (local_panel_active && kmeta->row_count > 0);
        cmeta->needs_xk = local_need_xk;
        MPI_Allgather(&local_need_xk, 1, MPI_INT, fill_counts, 1, MPI_INT,
                      solve_comm);
        for (int p = 0; p < comm_nprocs; ++p)
            if (fill_counts[p] && p != root_rank)
                ++cmeta->xk_receiver_count;
        if (cmeta->xk_receiver_count > 0) {
            int receiver = 0;
            if (!(cmeta->xk_receivers = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product(
                          (size_t) cmeta->xk_receiver_count, sizeof(int),
                          "SymLDL Xk receiver list"))))
                ABORT("Malloc fails for SymLDL Xk receiver list.");
            for (int p = 0; p < comm_nprocs; ++p)
                if (fill_counts[p] && p != root_rank)
                    cmeta->xk_receivers[receiver++] = p;
        }

        local_has_diag = kmeta->has_diag;
        cmeta->has_diag = local_has_diag;
        MPI_Allgather(&local_has_diag, 1, MPI_INT, fill_counts, 1, MPI_INT,
                      solve_comm);
        for (int p = 0; p < comm_nprocs; ++p)
            if (fill_counts[p])
                ++cmeta->diag_rank_count;
        if (cmeta->diag_rank_count > 0) {
            int owner = 0;
            if (!(cmeta->diag_ranks = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product(
                          (size_t) cmeta->diag_rank_count, sizeof(int),
                          "SymLDL diagonal rank list"))))
                ABORT("Malloc fails for SymLDL diagonal rank list.");
            for (int p = 0; p < comm_nprocs; ++p)
                if (fill_counts[p])
                    cmeta->diag_ranks[owner++] = p;
        }

        if (local_panel_active && kmeta->row_count > 0) {
            int row_count = pdgstrs3d_symldl_count_to_int(
                kmeta->row_count, "SymLDL communication schedule rows");
            if (!(cmeta->row_to_send_pos = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product((size_t) row_count,
                                                sizeof(int),
                                                "SymLDL row send map"))))
                ABORT("Malloc fails for SymLDL row send map.");
            for (int_t row = 0; row < kmeta->row_count; ++row) {
                int dest = pdgstrs3d_symldl_rank_to_tree_rank(
                    tree_comm, kmeta->row_dest_global[row]);
                ++cmeta->send_counts[dest];
            }
        }

        pdgstrs3d_symldl_counts_to_displs(comm_nprocs,
                                          cmeta->send_counts,
                                          cmeta->send_displs,
                                          &total_send);
        MPI_Alltoall(cmeta->send_counts, 1, MPI_INT,
                     cmeta->recv_counts, 1, MPI_INT, solve_comm);
        pdgstrs3d_symldl_counts_to_displs(comm_nprocs,
                                          cmeta->recv_counts,
                                          cmeta->recv_displs,
                                          &total_recv);
        cmeta->total_send = total_send;
        cmeta->total_recv = total_recv;

        for (int p = 0; p < comm_nprocs; ++p) {
            cmeta->send_val_counts[p] = cmeta->send_counts[p] * nrhs;
            cmeta->recv_val_counts[p] = cmeta->recv_counts[p] * nrhs;
        }
        pdgstrs3d_symldl_counts_to_displs(comm_nprocs,
                                          cmeta->send_val_counts,
                                          cmeta->send_val_displs,
                                          &cmeta->total_send_vals);
        pdgstrs3d_symldl_counts_to_displs(comm_nprocs,
                                          cmeta->recv_val_counts,
                                          cmeta->recv_val_displs,
                                          &cmeta->total_recv_vals);

        if (total_send > 0) {
            if (!(cmeta->send_seq = (int *) SUPERLU_MALLOC(
                      pdgstrs3d_checked_product((size_t) total_send,
                                                sizeof(int),
                                                "SymLDL row send sequence"))))
                ABORT("Malloc fails for SymLDL row send sequence.");
            pdgstrs3d_symldl_grow_int_t_buffer(
                &send_rows_buf, &send_rows_cap, total_send,
                "Malloc fails for SymLDL communication schedule rows.");
            for (int p = 0; p < comm_nprocs; ++p)
                fill_counts[p] = 0;
            for (int_t row = 0; row < kmeta->row_count; ++row) {
                int row_i = pdgstrs3d_symldl_count_to_int(
                    row, "SymLDL row send position");
                int dest = pdgstrs3d_symldl_rank_to_tree_rank(
                    tree_comm, kmeta->row_dest_global[row]);
                int pos = cmeta->send_displs[dest] + fill_counts[dest]++;
                send_rows_buf[pos] = kmeta->rows[row];
                cmeta->row_to_send_pos[row_i] = pos;
                cmeta->send_seq[pos] = row_i;
            }
        }
        if (total_recv > 0) {
            if (!(cmeta->recv_rows = intMalloc_dist(total_recv)))
                ABORT("Malloc fails for SymLDL received row schedule.");
        }

        MPI_Alltoallv(send_rows_buf, cmeta->send_counts, cmeta->send_displs,
                      mpi_int_t, cmeta->recv_rows, cmeta->recv_counts,
                      cmeta->recv_displs, mpi_int_t, solve_comm);
    }

    if (send_rows_buf) SUPERLU_FREE(send_rows_buf);
    SUPERLU_FREE(fill_counts);
    return meta;
}

static void
pdgstrs3d_symldl_comm_meta_free(pdgstrs3d_symldl_comm_meta_t *meta,
                                int_t nsupers)
{
    if (meta == NULL)
        return;

    for (int_t k = 0; k < nsupers; ++k) {
        if (meta[k].recv_rows) SUPERLU_FREE(meta[k].recv_rows);
        if (meta[k].diag_ranks) SUPERLU_FREE(meta[k].diag_ranks);
        if (meta[k].xk_receivers) SUPERLU_FREE(meta[k].xk_receivers);
        if (meta[k].send_seq) SUPERLU_FREE(meta[k].send_seq);
        if (meta[k].row_to_send_pos) SUPERLU_FREE(meta[k].row_to_send_pos);
        if (meta[k].counts) SUPERLU_FREE(meta[k].counts);
    }
    SUPERLU_FREE(meta);
}

static pdgstrs3d_symldl_x_cache_t *
pdgstrs3d_symldl_x_cache_create(int_t nsupers,
                                pdgstrs3d_symldl_panel_meta_t *panel_meta,
                                int nrhs)
{
    pdgstrs3d_symldl_x_cache_t *cache;

    if (!(cache = (pdgstrs3d_symldl_x_cache_t *)
              SUPERLU_MALLOC(pdgstrs3d_checked_product((size_t) nsupers,
                              sizeof(pdgstrs3d_symldl_x_cache_t),
                              "SymLDL replicated X cache"))))
        ABORT("Malloc fails for SymLDL replicated X cache.");
    memset(cache, 0, pdgstrs3d_checked_alloc_bytes(
           nsupers, sizeof(pdgstrs3d_symldl_x_cache_t),
           "SymLDL replicated X cache"));

    for (int_t k = 0; k < nsupers; ++k) {
        int_t row_count = panel_meta[k].has_panel ? panel_meta[k].row_count : 0;
        if (row_count <= 0)
            continue;
        int_t value_count = pdgstrs3d_checked_workspace_count(
            row_count, nrhs, 0, 0, "SymLDL replicated X cache values");
        cache[k].row_count = row_count;
        if (!(cache[k].values = doubleMalloc_dist(value_count)) ||
            !(cache[k].valid = (unsigned char *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_alloc_bytes(
                      row_count, sizeof(unsigned char),
                      "SymLDL replicated X cache valid flags"))))
            ABORT("Malloc fails for SymLDL replicated X cache entries.");
    }

    return cache;
}

static void
pdgstrs3d_symldl_x_cache_free(pdgstrs3d_symldl_x_cache_t *cache,
                              int_t nsupers)
{
    if (cache == NULL)
        return;
    for (int_t k = 0; k < nsupers; ++k) {
        if (cache[k].valid) SUPERLU_FREE(cache[k].valid);
        if (cache[k].values) SUPERLU_FREE(cache[k].values);
    }
    SUPERLU_FREE(cache);
}

static int
pdgstrs3d_symldl_node_in_tree(pdgstrs3d_symldl_solve_meta_t *meta,
                              int_t k, int tree)
{
    dtrf3Dpartition_t *trf3Dpartition =
        meta != NULL ? meta->trf3Dpartition : NULL;
    if (trf3Dpartition == NULL || trf3Dpartition->supernode2treeMap == NULL)
        ABORT("SymLDL replicated X cache requires supernode tree metadata.");
    return trf3Dpartition->supernode2treeMap[k] == tree;
}

static double
pdgstrs3d_symldl_x_cache_fill(pdgstrs3d_symldl_solve_meta_t *meta,
                              double *x, int nrhs, int_t *ilsum,
                              gridinfo3d_t *grid3d,
                              pdgstrs3d_symldl_node_ctx_t *ctxs,
                              int_t nctx)
{
    dtrf3Dpartition_t *trf3Dpartition = meta->trf3Dpartition;
    Glu_persist_t *Glu_persist = meta->Glu_persist;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    double start = SuperLU_timer_();
    double replicated_bytes = 0.0;
    double avoided_bytes = 0.0;
    double hits = 0.0;
    double misses = 0.0;
    double panels = 0.0;
    double dummy = 0.0;

    for (int_t ci = 0; ci < nctx; ++ci) {
        pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
        pdgstrs3d_symldl_panel_meta_t *kmeta;
        pdgstrs3d_symldl_x_cache_t *cache;
        if (!ctx->active)
            continue;
        kmeta = ctx->kmeta;
        cache = &meta->x_cache[ctx->k];
        if (!kmeta->has_panel || kmeta->row_count <= 0 ||
            cache->valid == NULL)
            continue;
        memset(cache->valid, 0,
               pdgstrs3d_checked_alloc_bytes(
                   cache->row_count, sizeof(unsigned char),
                   "SymLDL replicated X cache valid flags"));
    }

    for (int tree = 0; tree < meta->numForests; ++tree) {
        pdgstrs3d_symldl_tree_comm_t *tree_comm = &meta->tree_comms[tree];
        int nprocs;
        int *send_counts = NULL;
        int *recv_counts = NULL;
        int *send_displs = NULL;
        int *recv_displs = NULL;
        int *send_cursor = NULL;
        int *recv_cursor = NULL;
        double *send_buf = NULL;
        double *recv_buf = NULL;
        int total_send = 0;
        int total_recv = 0;

        if (!tree_comm->active)
            continue;
        nprocs = tree_comm->nprocs;
        if (!(send_counts = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache send counts"))) ||
            !(recv_counts = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache recv counts"))) ||
            !(send_displs = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache send displs"))) ||
            !(recv_displs = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache recv displs"))) ||
            !(send_cursor = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache send cursor"))) ||
            !(recv_cursor = (int *) SUPERLU_MALLOC(
                  pdgstrs3d_checked_product((size_t) nprocs, sizeof(int),
                                            "SymLDL X cache recv cursor"))))
            ABORT("Malloc fails for SymLDL X cache communication counts.");
        for (int p = 0; p < nprocs; ++p) {
            send_counts[p] = 0;
            recv_counts[p] = 0;
        }

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            int_t k = ctx->k;
            if (!ctx->active)
                continue;
            if (!cmeta->active ||
                !pdgstrs3d_symldl_node_in_tree(meta, k, tree))
                continue;
            for (int p = 0; p < nprocs; ++p) {
                send_counts[p] += cmeta->recv_val_counts[p];
                recv_counts[p] += cmeta->send_val_counts[p];
            }
        }

        pdgstrs3d_symldl_counts_to_displs(nprocs, send_counts, send_displs,
                                          &total_send);
        pdgstrs3d_symldl_counts_to_displs(nprocs, recv_counts, recv_displs,
                                          &total_recv);
        if (total_send > 0 && !(send_buf = doubleMalloc_dist(total_send)))
            ABORT("Malloc fails for SymLDL X cache send buffer.");
        if (total_recv > 0 && !(recv_buf = doubleMalloc_dist(total_recv)))
            ABORT("Malloc fails for SymLDL X cache recv buffer.");
        for (int p = 0; p < nprocs; ++p) {
            send_cursor[p] = 0;
            recv_cursor[p] = 0;
        }

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            int_t k = ctx->k;
            if (!ctx->active)
                continue;
            if (!cmeta->active ||
                !pdgstrs3d_symldl_node_in_tree(meta, k, tree))
                continue;
            for (int p = 0; p < nprocs; ++p) {
                int row_base = cmeta->recv_displs[p];
                int val_base = send_displs[p] + send_cursor[p];
                for (int j = 0; j < cmeta->recv_counts[p]; ++j) {
                    int_t grow = cmeta->recv_rows[row_base + j];
                    int_t gsup = BlockNum(grow);
                    int_t rel = grow - FstBlockC(gsup);
                    int_t gsupsz = SuperSize(gsup);
                    int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, gsup);
                    double *xg = &x[X_BLK(lk)];
                    for (int rhs = 0; rhs < nrhs; ++rhs)
                        send_buf[val_base + j * nrhs + rhs] =
                            xg[rel + (int_t) rhs * gsupsz];
                }
                send_cursor[p] += cmeta->recv_val_counts[p];
            }
        }

        MPI_Alltoallv(total_send ? send_buf : &dummy, send_counts,
                      send_displs, MPI_DOUBLE,
                      total_recv ? recv_buf : &dummy, recv_counts,
                      recv_displs, MPI_DOUBLE, tree_comm->comm);

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            int_t k = ctx->k;
            pdgstrs3d_symldl_x_cache_t *cache = &meta->x_cache[k];
            if (!ctx->active)
                continue;
            if (!cmeta->active ||
                !pdgstrs3d_symldl_node_in_tree(meta, k, tree) ||
                !kmeta->has_panel || kmeta->row_count <= 0)
                continue;
            if (cache->values == NULL || cache->valid == NULL ||
                cache->row_count != kmeta->row_count)
                ABORT("SymLDL X cache metadata is inconsistent.");
            for (int p = 0; p < nprocs; ++p) {
                int row_base = cmeta->send_displs[p];
                int val_base = recv_displs[p] + recv_cursor[p];
                for (int j = 0; j < cmeta->send_counts[p]; ++j) {
                    int seq = cmeta->send_seq[row_base + j];
                    for (int rhs = 0; rhs < nrhs; ++rhs)
                        cache->values[(int_t) seq * nrhs + rhs] =
                            recv_buf[val_base + j * nrhs + rhs];
                    cache->valid[seq] = 1;
                }
                recv_cursor[p] += cmeta->send_val_counts[p];
            }
        }

        replicated_bytes += (double) total_send * (double) sizeof(double);
        avoided_bytes += (double) total_recv * (double) sizeof(double);

        if (send_buf) SUPERLU_FREE(send_buf);
        if (recv_buf) SUPERLU_FREE(recv_buf);
        SUPERLU_FREE(recv_cursor);
        SUPERLU_FREE(send_cursor);
        SUPERLU_FREE(recv_displs);
        SUPERLU_FREE(send_displs);
        SUPERLU_FREE(recv_counts);
        SUPERLU_FREE(send_counts);
    }

    for (int_t ci = 0; ci < nctx; ++ci) {
        pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
        pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
        pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
        int_t k = ctx->k;
        pdgstrs3d_symldl_x_cache_t *cache = &meta->x_cache[k];
        if (!ctx->active)
            continue;
        if (!kmeta->has_panel || kmeta->row_count <= 0 ||
            !cmeta->active || cmeta->total_send <= 0)
            continue;
        ++panels;
        for (int_t row = 0; row < kmeta->row_count; ++row) {
            if (cache->valid[row]) {
                hits += (double) nrhs;
            } else {
                misses += (double) nrhs;
                fprintf(stderr,
                        "SymLDL X cache missing row on rank %d: "
                        "panel=%lld local_row=%lld global_row=%lld\n",
                        grid3d->iam, (long long) k, (long long) row,
                        (long long) kmeta->rows[row]);
                fflush(stderr);
            }
        }
    }
    if (misses != 0.0)
        ABORT("SymLDL X cache coverage is incomplete.");

    double elapsed = SuperLU_timer_() - start;
    meta->x_cache_fill_time += elapsed;
    meta->x_cache_replicated_bytes += replicated_bytes;
    meta->x_cache_avoided_request_bytes += avoided_bytes;
    meta->x_cache_hits += hits;
    meta->x_cache_misses += misses;
    meta->x_cache_panels += panels;
    (void) xsup;
    (void) supno;
    return elapsed;
}

static int_t
pdgstrs3d_symldl_panel_meta_max_block_rows(pdgstrs3d_symldl_panel_meta_t *meta,
                                           int_t nsupers)
{
    int_t max_rows = 1;

    if (meta == NULL)
        return max_rows;
    for (int_t k = 0; k < nsupers; ++k)
        for (int_t block = 0; block < meta[k].nblocks; ++block)
            max_rows = SUPERLU_MAX(max_rows, meta[k].block_nbrow[block]);
    return max_rows;
}

static void
pdgstrs3d_symldl_workspace_free(pdgstrs3d_symldl_workspace_t *work)
{
    if (work == NULL)
        return;
    if (work->recv_request_values_buf) SUPERLU_FREE(work->recv_request_values_buf);
    if (work->request_values_buf) SUPERLU_FREE(work->request_values_buf);
    if (work->row_values_buf) SUPERLU_FREE(work->row_values_buf);
    if (work->recv_vals_buf) SUPERLU_FREE(work->recv_vals_buf);
    if (work->send_vals_buf) SUPERLU_FREE(work->send_vals_buf);
    if (work->delta_recv_buf) SUPERLU_FREE(work->delta_recv_buf);
    if (work->comm_reqs) SUPERLU_FREE(work->comm_reqs);
    if (work->level_ctxs) SUPERLU_FREE(work->level_ctxs);
    if (work->rhs_buf) SUPERLU_FREE(work->rhs_buf);
    if (work->gemm_buf) SUPERLU_FREE(work->gemm_buf);
    if (work->delta_buf) SUPERLU_FREE(work->delta_buf);
    if (work->delta_send_buf) SUPERLU_FREE(work->delta_send_buf);
    if (work->diag_buf) SUPERLU_FREE(work->diag_buf);
    if (work->diag_send_buf) SUPERLU_FREE(work->diag_send_buf);
    if (work->xk_buf) SUPERLU_FREE(work->xk_buf);
    if (work->x) SUPERLU_FREE(work->x);
    memset(work, 0, sizeof(*work));
}

static void
pdgstrs3d_symldl_zero_double_buffer(double *buffer, int_t count,
                                    const char *what)
{
    if (buffer == NULL || count <= 0)
        return;
    memset(buffer, 0, pdgstrs3d_checked_alloc_bytes(count, sizeof(double),
                                                    what));
}

static void
pdgstrs3d_symldl_workspace_prepare(pdgstrs3d_symldl_solve_meta_t *meta,
                                   int_t x_count, int_t maxsup, int nrhs,
                                   int global_nprocs)
{
    pdgstrs3d_symldl_workspace_t *work = &meta->work;
    int_t panel_rows = SUPERLU_MAX(meta->max_panel_block_rows, (int_t) 1);
    int_t pivot_count = pdgstrs3d_checked_workspace_count(
        SUPERLU_MAX(maxsup, (int_t) 1), nrhs, 0, 0,
        "3D SymLDL pivot workspace");
    int_t panel_count = pdgstrs3d_checked_workspace_count(
        panel_rows, nrhs, 0, 0, "3D SymLDL panel workspace");
    int_t comm_req_count = pdgstrs3d_checked_workspace_count(
        2, global_nprocs, 0, 0, "SymLDL communication requests");

    if (x_count <= 0)
        x_count = 1;
    if (panel_count <= 0)
        panel_count = 1;
    if (comm_req_count <= 0)
        comm_req_count = 1;

    pdgstrs3d_symldl_grow_double_buffer(&work->x, &work->x_cap, x_count,
                                        "Malloc fails for x[].");
    pdgstrs3d_symldl_grow_double_buffer(&work->xk_buf, &work->xk_cap,
                                        pivot_count,
                                        "Malloc fails for SymLDL pivot buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->diag_send_buf,
                                        &work->diag_send_cap, pivot_count,
                                        "Malloc fails for SymLDL diagonal send buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->diag_buf, &work->diag_cap,
                                        pivot_count,
                                        "Malloc fails for SymLDL diagonal buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->delta_send_buf,
                                        &work->delta_send_cap, pivot_count,
                                        "Malloc fails for SymLDL delta send buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->delta_buf, &work->delta_cap,
                                        pivot_count,
                                        "Malloc fails for SymLDL delta buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->gemm_buf, &work->gemm_cap,
                                        panel_count,
                                        "Malloc fails for SymLDL GEMM buffer.");
    pdgstrs3d_symldl_grow_double_buffer(&work->rhs_buf, &work->rhs_cap,
                                        panel_count,
                                        "Malloc fails for SymLDL RHS buffer.");
    pdgstrs3d_symldl_grow_request_buffer(&work->comm_reqs,
                                         &work->comm_reqs_cap,
                                         comm_req_count,
                                         "Malloc fails for SymLDL communication requests.");

    pdgstrs3d_symldl_zero_double_buffer(work->x, x_count,
                                        "3D SymLDL solve x workspace");
}

static int
pdgstrs3d_symldl_panel_needs_host_values(
    pdgstrs3d_symldl_solve_meta_t *meta,
    pdgstrs3d_symldl_panel_meta_t *kmeta, int_t ksupc, int nrhs)
{
    if (kmeta == NULL || !kmeta->has_panel)
        return 0;
    if (kmeta->has_diag &&
        !pdgstrs3d_symldl_should_gpu_ops(
            meta, pdgstrs3d_symldl_diag_solve_ops(kmeta, ksupc, nrhs)))
        return 1;
    if (kmeta->row_count > 0 &&
        !pdgstrs3d_symldl_should_gpu_ops(
            meta, pdgstrs3d_symldl_panel_solve_ops(kmeta, ksupc, nrhs)))
        return 1;
    return 0;
}

static int_t
pdgstrs3d_symldl_super_size_from_meta(
    pdgstrs3d_symldl_solve_meta_t *meta, int_t k)
{
    int_t *meta_xsup = (meta != NULL && meta->Glu_persist != NULL)
                           ? meta->Glu_persist->xsup : NULL;
    if (meta_xsup == NULL || k < 0 || k >= meta->nsupers)
        ABORT("SymLDL solve missing supernode size metadata.");
    return meta_xsup[k + 1] - meta_xsup[k];
}

static void
pdgstrs3d_symldl_sync_factor_gpu(pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL || meta->factor_gpu_handle == NULL ||
        meta->factor_gpu_synchronized)
        return;
    dSymLDLFactorGPUSynchronize((dLUgpu_Handle) meta->factor_gpu_handle);
    meta->factor_gpu_synchronized = 1;
}

static void
pdgstrs3d_symldl_prepare_host_factor_panels(
    pdgstrs3d_symldl_solve_meta_t *meta, int nrhs)
{
    if (meta == NULL || meta->panel_meta == NULL)
        return;

    int need_any_host_panel = 0;
    for (int_t k = 0; k < meta->nsupers; ++k) {
        pdgstrs3d_symldl_panel_meta_t *kmeta = &meta->panel_meta[k];
        int_t ksupc = pdgstrs3d_symldl_super_size_from_meta(meta, k);
        if (pdgstrs3d_symldl_panel_needs_host_values(
                meta, kmeta, ksupc, nrhs)) {
            need_any_host_panel = 1;
            break;
        }
    }
    if (!need_any_host_panel)
        return;
    if (meta->factor_gpu_handle == NULL)
        ABORT("SymLDL V2 solve requires retained factor GPU state for CPU-scheduled panel work.");

    pdgstrs3d_symldl_sync_factor_gpu(meta);
    double t = SuperLU_timer_();
    for (int_t k = 0; k < meta->nsupers; ++k) {
        pdgstrs3d_symldl_panel_meta_t *kmeta = &meta->panel_meta[k];
        int_t ksupc = pdgstrs3d_symldl_super_size_from_meta(meta, k);
        if (!pdgstrs3d_symldl_panel_needs_host_values(
                meta, kmeta, ksupc, nrhs))
            continue;
        if (dSymLDLFactorGPUCopyPanelToHost(
                (dLUgpu_Handle) meta->factor_gpu_handle, k) != 0)
            ABORT("Failed to copy CPU-scheduled SymLDL panel from factor GPU state.");
    }
    meta->host_panel_copy_time += SuperLU_timer_() - t;
}

static void
pdgstrs3d_symldl_gpu_prepare(pdgstrs3d_symldl_solve_meta_t *meta,
                             int_t maxsup, int nrhs, gridinfo3d_t *grid3d)
{
    if (!pdgstrs3d_symldl_gpu_solve_allowed(meta))
        return;
#if defined(GPU_ACC)
    if (meta == NULL)
        ABORT("SymLDL GPU solve metadata is missing.");
    if (meta->gpu_state != NULL)
        return;
    int_t *xsup = meta->Glu_persist->xsup;

    int has_gpu_work = 0;
    for (int_t k = 0; k < meta->nsupers; ++k) {
        pdgstrs3d_symldl_panel_meta_t *kmeta = &meta->panel_meta[k];
        int_t ksupc = SuperSize(k);
        if ((kmeta->has_panel &&
             pdgstrs3d_symldl_should_gpu_ops(meta,
                 pdgstrs3d_symldl_panel_solve_ops(kmeta, ksupc, nrhs))) ||
            pdgstrs3d_symldl_should_gpu_ops(meta,
                pdgstrs3d_symldl_diag_solve_ops(kmeta, ksupc, nrhs))) {
            has_gpu_work = 1;
            break;
        }
    }
    if (!has_gpu_work)
        return;
    if (meta->factor_gpu_handle == NULL)
        ABORT("SymLDL V2 GPU solve requires retained factor GPU state.");
    pdgstrs3d_symldl_sync_factor_gpu(meta);

    dSymLDLSolveGPU_Handle handle =
        dSymLDLSolveGPUCreate(meta->nsupers, maxsup,
                              meta->max_panel_block_rows, nrhs, grid3d);
    if (handle == NULL)
        ABORT("Failed to create SymLDL GPU solve state.");

    for (int_t k = 0; k < meta->nsupers; ++k) {
        pdgstrs3d_symldl_panel_meta_t *kmeta = &meta->panel_meta[k];
        if (!kmeta->has_panel || kmeta->lusup == NULL ||
            kmeta->lusup_count <= 0)
            continue;
        int_t ksupc = SuperSize(k);
        int use_panel_gpu = pdgstrs3d_symldl_should_gpu_ops(meta,
            pdgstrs3d_symldl_panel_solve_ops(kmeta, ksupc, nrhs));
        int use_diag_gpu = pdgstrs3d_symldl_should_gpu_ops(meta,
            pdgstrs3d_symldl_diag_solve_ops(kmeta, ksupc, nrhs));
        if (!use_panel_gpu && !use_diag_gpu)
            continue;
        double t_panel = SuperLU_timer_();
        if (dSymLDLSolveGPUAttachFactorPanel(
                handle, (dLUgpu_Handle) meta->factor_gpu_handle, k) != 0)
            ABORT("Failed to attach SymLDL factor GPU panel to solve state.");
        meta->gpu_panel_import_time += SuperLU_timer_() - t_panel;
        if (use_panel_gpu && kmeta->row_count > 0 && meta->comm_meta != NULL) {
            pdgstrs3d_symldl_comm_meta_t *cmeta = &meta->comm_meta[k];
            if (cmeta->row_to_send_pos != NULL) {
                double t_sched = SuperLU_timer_();
                if (dSymLDLSolveGPUSetPanelSchedule(handle, k,
                                                    cmeta->row_to_send_pos,
                                                    kmeta->row_count,
                                                    kmeta->nblocks,
                                                    kmeta->block_luptr,
                                                    kmeta->block_nbrow,
                                                    kmeta->block_row_start) != 0)
                    ABORT("Failed to copy SymLDL solve panel schedule to GPU.");
                meta->gpu_schedule_upload_time += SuperLU_timer_() - t_sched;
            }
        }
    }
    meta->gpu_state = (void *) handle;
#else
    (void) meta;
    (void) maxsup;
    (void) nrhs;
    (void) grid3d;
    ABORT("SymLDL GPU solve requires a CUDA build.");
#endif
}

static void
pdgstrs3d_symldl_gpu_take_timers(pdgstrs3d_symldl_solve_meta_t *meta,
                                 pdgstrs3d_symldl_timer_t *timer)
{
#if defined(GPU_ACC)
    double h2d = 0.0;
    double compute = 0.0;
    double d2h = 0.0;
    if (meta == NULL || meta->gpu_state == NULL || timer == NULL)
        return;
    dSymLDLSolveGPUTakeTimers((dSymLDLSolveGPU_Handle) meta->gpu_state,
                              &h2d, &compute, &d2h);
    timer->gpu_h2d += h2d;
    timer->gpu_compute += compute;
    timer->gpu_d2h += d2h;
#else
    (void) meta;
    (void) timer;
#endif
}

static void
pdgstrs3d_symldl_solve_meta_destroy(pdgstrs3d_symldl_solve_meta_t *meta)
{
    if (meta == NULL)
        return;
#if defined(GPU_ACC)
    if (meta->gpu_state)
        dSymLDLSolveGPUDestroy((dSymLDLSolveGPU_Handle) meta->gpu_state);
#endif
    if (meta->factor_gpu_handle)
        dDestroyLUgpuHandle((dLUgpu_Handle) meta->factor_gpu_handle);
    pdgstrs3d_symldl_workspace_free(&meta->work);
    pdgstrs3d_symldl_level_schedule_free(&meta->solve_schedule);
    pdgstrs3d_symldl_x_cache_free(meta->x_cache, meta->nsupers);
    pdgstrs3d_symldl_comm_meta_free(meta->comm_meta, meta->nsupers);
    pdgstrs3d_symldl_panel_meta_free(meta->panel_meta, meta->nsupers);
    pdgstrs3d_symldl_tree_comms_free_count(meta->tree_comms,
                                           meta->numForests);
    if (meta->diag_owner) SUPERLU_FREE(meta->diag_owner);
    if (meta->solve_order) SUPERLU_FREE(meta->solve_order);
    SUPERLU_FREE(meta);
}

void
pdgstrs3d_symldl_finalize(dSOLVEstruct_t *SOLVEstruct)
{
    if (SOLVEstruct == NULL || SOLVEstruct->symldl_v2_solve_meta == NULL)
    {
        if (SOLVEstruct != NULL && SOLVEstruct->symldl_v2_factor_handle != NULL) {
            dDestroyLUgpuHandle(
                (dLUgpu_Handle) SOLVEstruct->symldl_v2_factor_handle);
            SOLVEstruct->symldl_v2_factor_handle = NULL;
        }
        return;
    }
    pdgstrs3d_symldl_solve_meta_destroy(
        (pdgstrs3d_symldl_solve_meta_t *) SOLVEstruct->symldl_v2_solve_meta);
    SOLVEstruct->symldl_v2_solve_meta = NULL;
    SOLVEstruct->symldl_v2_factor_handle = NULL;
}

static int
pdgstrs3d_symldl_solve_meta_valid(pdgstrs3d_symldl_solve_meta_t *meta,
                                  int_t n, int_t nsupers, int nrhs,
                                  Glu_persist_t *Glu_persist,
                                  dLocalLU_t *Llu,
                                  dtrf3Dpartition_t *trf3Dpartition,
                                  int *supernodeMask, gridinfo3d_t *grid3d,
                                  int global_nprocs, int superlu_acc_offload,
                                  int superlu_n_gemm)
{
    gridinfo_t *grid = &(grid3d->grid2d);

    return meta != NULL &&
           meta->n == n &&
           meta->nsupers == nsupers &&
           meta->nrhs == nrhs &&
           meta->global_nprocs == global_nprocs &&
           meta->nprow == grid->nprow &&
           meta->npcol == grid->npcol &&
           meta->znp == grid3d->zscp.Np &&
           meta->superlu_acc_offload == superlu_acc_offload &&
           meta->superlu_n_gemm == superlu_n_gemm &&
           meta->Glu_persist == Glu_persist &&
           meta->Llu == Llu &&
           meta->trf3Dpartition == trf3Dpartition &&
           meta->supernodeMask == supernodeMask;
}

static pdgstrs3d_symldl_solve_meta_t *
pdgstrs3d_symldl_solve_meta_get(dSOLVEstruct_t *SOLVEstruct, int_t n,
                                int_t nsupers, int nrhs,
                                dLUstruct_t *LUstruct,
                                dtrf3Dpartition_t *trf3Dpartition,
                                int *supernodeMask, gridinfo_t *grid,
                                gridinfo3d_t *grid3d, int global_nprocs,
                                int superlu_acc_offload, int superlu_n_gemm)
{
    pdgstrs3d_symldl_solve_meta_t *meta =
        (pdgstrs3d_symldl_solve_meta_t *) SOLVEstruct->symldl_v2_solve_meta;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;

    if (pdgstrs3d_symldl_solve_meta_valid(meta, n, nsupers, nrhs,
                                          Glu_persist, Llu, trf3Dpartition,
                                          supernodeMask, grid3d,
                                          global_nprocs,
                                          superlu_acc_offload,
                                          superlu_n_gemm)) {
        if (SOLVEstruct->symldl_v2_factor_handle != NULL) {
            if (meta->factor_gpu_handle != NULL)
                ABORT("SymLDL solve metadata already owns a factor GPU handle.");
            meta->factor_gpu_handle = SOLVEstruct->symldl_v2_factor_handle;
            SOLVEstruct->symldl_v2_factor_handle = NULL;
        }
        meta->reused = 1;
        return meta;
    }

    void *pending_factor_gpu_handle = SOLVEstruct->symldl_v2_factor_handle;
    SOLVEstruct->symldl_v2_factor_handle = NULL;
    pdgstrs3d_symldl_finalize(SOLVEstruct);
    SOLVEstruct->symldl_v2_factor_handle = pending_factor_gpu_handle;
    if (!(meta = (pdgstrs3d_symldl_solve_meta_t *)
              SUPERLU_MALLOC(sizeof(pdgstrs3d_symldl_solve_meta_t))))
        ABORT("Malloc fails for SymLDL solve metadata.");
    memset(meta, 0, sizeof(*meta));
    meta->n = n;
    meta->nsupers = nsupers;
    meta->nrhs = nrhs;
    meta->global_nprocs = global_nprocs;
    meta->nprow = grid->nprow;
    meta->npcol = grid->npcol;
    meta->znp = grid3d->zscp.Np;
    meta->superlu_acc_offload = superlu_acc_offload;
    meta->superlu_n_gemm = superlu_n_gemm;
    if (getenv("GPU3DV2_TRACE")) {
        fprintf(stderr,
                "[sym-v2-trace] rank %d: metadata taking factor handle %p\n",
                grid3d->iam, SOLVEstruct->symldl_v2_factor_handle);
        fflush(stderr);
    }
    meta->factor_gpu_handle = SOLVEstruct->symldl_v2_factor_handle;
    SOLVEstruct->symldl_v2_factor_handle = NULL;
    meta->Glu_persist = Glu_persist;
    meta->Llu = Llu;
    meta->trf3Dpartition = trf3Dpartition;
    meta->supernodeMask = supernodeMask;
    meta->diag_owner =
        pdgstrs3d_symldl_diag_owners(nsupers, trf3Dpartition, grid3d);
    meta->solve_order = pdgstrs3d_symldl_tree_order(nsupers,
                                                    trf3Dpartition, grid3d);
    meta->panel_meta = pdgstrs3d_symldl_panel_meta_create(nsupers, Llu,
                                                          Glu_persist, grid,
                                                          trf3Dpartition,
                                                          supernodeMask,
                                                          meta->diag_owner);
    meta->tree_comms = pdgstrs3d_symldl_tree_comms_create(trf3Dpartition,
                                                          meta->panel_meta,
                                                          meta->diag_owner,
                                                          nsupers, grid3d,
                                                          global_nprocs);
    meta->numForests = meta->tree_comms
                           ? pdgstrs3d_symldl_num_forests(trf3Dpartition,
                                                          grid3d)
                           : 0;
    meta->max_panel_block_rows =
        pdgstrs3d_symldl_panel_meta_max_block_rows(meta->panel_meta, nsupers);
    meta->solve_schedule =
        pdgstrs3d_symldl_level_schedule_create(nsupers, meta->solve_order,
                                               meta->panel_meta, grid3d,
                                               Glu_persist);
    meta->comm_meta = pdgstrs3d_symldl_comm_meta_create(nsupers,
                                                        meta->panel_meta,
                                                        trf3Dpartition,
                                                        meta->tree_comms,
                                                        grid3d,
                                                        global_nprocs, nrhs,
                                                        meta->diag_owner);
    meta->x_cache = pdgstrs3d_symldl_x_cache_create(nsupers,
                                                    meta->panel_meta, nrhs);
    meta->reused = 0;
    SOLVEstruct->symldl_v2_solve_meta = meta;
    return meta;
}

void
pdgstrs3d_symldl_init_meta(superlu_dist_options_t *options, int_t n, int nrhs,
                           dLUstruct_t *LUstruct,
                           dtrf3Dpartition_t *trf3Dpartition,
                           gridinfo3d_t *grid3d,
                           dSOLVEstruct_t *SOLVEstruct)
{
    if (options == NULL || LUstruct == NULL || LUstruct->Glu_persist == NULL ||
        LUstruct->Llu == NULL || trf3Dpartition == NULL ||
        grid3d == NULL || SOLVEstruct == NULL)
        ABORT("SymLDL solve metadata initialization received invalid arguments.");

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int *supernodeMask = trf3Dpartition->supernodeMask;
    int global_nprocs;

    MPI_Comm_size(grid3d->comm, &global_nprocs);
    (void) pdgstrs3d_symldl_solve_meta_get(
        SOLVEstruct, n, nsupers, nrhs, LUstruct, trf3Dpartition,
        supernodeMask, &grid3d->grid2d, grid3d, global_nprocs,
        sp_ienv_dist(10, options), sp_ienv_dist(7, options));
}

static void
pdgstrs3d_symldl_forward_xk(double *xk, int count, int rank, int root,
                            pdgstrs3d_symldl_comm_meta_t *meta,
                            MPI_Comm comm)
{
    if (rank == root) {
        for (int i = 0; i < meta->xk_receiver_count; ++i)
            MPI_Send(xk, count, MPI_DOUBLE, meta->xk_receivers[i], Xk, comm);
    } else if (meta->needs_xk) {
        MPI_Recv(xk, count, MPI_DOUBLE, root, Xk, comm, MPI_STATUS_IGNORE);
    }
}

static void
pdgstrs3d_symldl_exchange_double(double *send_buf, int *send_counts,
                                 int *send_displs, double *recv_buf,
                                 int *recv_counts, int *recv_displs,
                                 int nprocs, int rank, int tag,
                                 MPI_Comm comm, MPI_Request *reqs)
{
    int nreq = 0;

    for (int p = 0; p < nprocs; ++p) {
        if (p == rank) {
            if (send_counts[p] > 0)
                memcpy(&recv_buf[recv_displs[p]], &send_buf[send_displs[p]],
                       (size_t) send_counts[p] * sizeof(double));
        } else if (recv_counts[p] > 0) {
            MPI_Irecv(&recv_buf[recv_displs[p]], recv_counts[p], MPI_DOUBLE,
                      p, tag, comm, &reqs[nreq++]);
        }
    }
    for (int p = 0; p < nprocs; ++p) {
        if (p != rank && send_counts[p] > 0)
            MPI_Isend(&send_buf[send_displs[p]], send_counts[p], MPI_DOUBLE,
                      p, tag, comm, &reqs[nreq++]);
    }
    if (nreq > 0)
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

static void
pdgstrs3d_symldl_post_xk(double *xk, int count, int rank, int root,
                         pdgstrs3d_symldl_comm_meta_t *meta,
                         MPI_Comm comm, MPI_Request *reqs, int *nreq)
{
    if (rank == root) {
        for (int i = 0; i < meta->xk_receiver_count; ++i)
            MPI_Isend(xk, count, MPI_DOUBLE, meta->xk_receivers[i], Xk,
                      comm, &reqs[(*nreq)++]);
    } else if (meta->needs_xk) {
        MPI_Irecv(xk, count, MPI_DOUBLE, root, Xk, comm,
                  &reqs[(*nreq)++]);
    }
}

static void
pdgstrs3d_symldl_post_exchange_double(double *send_buf, int *send_counts,
                                      int *send_displs, double *recv_buf,
                                      int *recv_counts, int *recv_displs,
                                      int nprocs, int rank, int tag,
                                      MPI_Comm comm, MPI_Request *reqs,
                                      int *nreq)
{
    for (int p = 0; p < nprocs; ++p) {
        if (p == rank) {
            if (send_counts[p] > 0)
                memcpy(&recv_buf[recv_displs[p]], &send_buf[send_displs[p]],
                       (size_t) send_counts[p] * sizeof(double));
        } else if (recv_counts[p] > 0) {
            MPI_Irecv(&recv_buf[recv_displs[p]], recv_counts[p], MPI_DOUBLE,
                      p, tag, comm, &reqs[(*nreq)++]);
        }
    }
    for (int p = 0; p < nprocs; ++p) {
        if (p != rank && send_counts[p] > 0)
            MPI_Isend(&send_buf[send_displs[p]], send_counts[p], MPI_DOUBLE,
                      p, tag, comm, &reqs[(*nreq)++]);
    }
}

static void
pdgstrs3d_symldl_reduce_delta(double *delta_send, double *delta, int count,
                              int rank, int root,
                              pdgstrs3d_symldl_comm_meta_t *meta,
                              MPI_Comm comm)
{
    if (rank == root) {
        if (meta->needs_xk)
            for (int i = 0; i < count; ++i)
                delta[i] += delta_send[i];
        for (int sender = 0; sender < meta->xk_receiver_count; ++sender) {
            MPI_Recv(delta_send, count, MPI_DOUBLE,
                     meta->xk_receivers[sender], RD_U, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; ++i)
                delta[i] += delta_send[i];
        }
    } else if (meta->needs_xk) {
        MPI_Send(delta_send, count, MPI_DOUBLE, root, RD_U, comm);
    }
}

static void
pdgstrs3d_symldl_post_reduce_delta(pdgstrs3d_symldl_node_ctx_t *ctx,
                                   MPI_Request *reqs, int *nreq)
{
    pdgstrs3d_symldl_comm_meta_t *meta = ctx->cmeta;
    int count = ctx->delta_count;

    if (ctx->solve_rank == ctx->root_rank) {
        if (meta->needs_xk)
            for (int i = 0; i < count; ++i)
                ctx->delta_buf[i] += ctx->delta_send_buf[i];
        for (int sender = 0; sender < meta->xk_receiver_count; ++sender) {
            MPI_Irecv(&ctx->delta_recv_buf[(size_t) sender * (size_t) count],
                      count, MPI_DOUBLE, meta->xk_receivers[sender], RD_U,
                      ctx->solve_comm, &reqs[(*nreq)++]);
        }
    } else if (meta->needs_xk) {
        MPI_Isend(ctx->delta_send_buf, count, MPI_DOUBLE, ctx->root_rank,
                  RD_U, ctx->solve_comm, &reqs[(*nreq)++]);
    }
}

static void
pdgstrs3d_symldl_finish_reduce_delta(pdgstrs3d_symldl_node_ctx_t *ctx)
{
    pdgstrs3d_symldl_comm_meta_t *meta = ctx->cmeta;
    int count = ctx->delta_count;

    if (ctx->solve_rank != ctx->root_rank)
        return;
    for (int sender = 0; sender < meta->xk_receiver_count; ++sender) {
        double *recv = &ctx->delta_recv_buf[(size_t) sender * (size_t) count];
        for (int i = 0; i < count; ++i)
            ctx->delta_buf[i] += recv[i];
    }
}

static int_t
pdgstrs3d_symldl_count_sum(int_t total, int_t add, const char *what)
{
    return pdgstrs3d_checked_workspace_count(total, 1, add, 1, what);
}

static pdgstrs3d_symldl_node_ctx_t *
pdgstrs3d_symldl_workspace_prepare_level_contexts(
    pdgstrs3d_symldl_workspace_t *work, int_t nctx, const char *what)
{
    int_t need = SUPERLU_MAX(nctx, (int_t) 1);
    size_t bytes = pdgstrs3d_checked_product((size_t) need,
                                             sizeof(*work->level_ctxs),
                                             what);

    if (need > work->level_ctxs_cap) {
        if (work->level_ctxs)
            SUPERLU_FREE(work->level_ctxs);
        work->level_ctxs =
            (pdgstrs3d_symldl_node_ctx_t *) SUPERLU_MALLOC(bytes);
        if (work->level_ctxs == NULL)
            ABORT(what);
        work->level_ctxs_cap = need;
    }
    memset(work->level_ctxs, 0, bytes);
    return work->level_ctxs;
}

static void
pdgstrs3d_symldl_timer_print(pdgstrs3d_symldl_timer_t *timer,
                             pdgstrs3d_symldl_solve_meta_t *meta,
                             gridinfo3d_t *grid3d)
{
    enum { SYMLDL_TIMER_COUNT = 33 };
    double local[SYMLDL_TIMER_COUNT];
    double maxv[SYMLDL_TIMER_COUNT];
    double sumv[SYMLDL_TIMER_COUNT];
    struct { double val; int rank; } local_max[SYMLDL_TIMER_COUNT];
    struct { double val; int rank; } global_max[SYMLDL_TIMER_COUNT];
    int rank, nprocs;

    local[0] = timer->metadata;
    local[1] = timer->workspace;
    local[2] = timer->b_to_x;
    local[3] = timer->forward_xk;
    local[4] = timer->forward_compute;
    local[5] = timer->forward_values;
    local[6] = timer->forward_apply;
    local[7] = timer->diag_comm;
    local[8] = timer->diag_compute;
    local[9] = timer->x_cache_fill;
    local[10] = timer->backward_values;
    local[11] = timer->backward_compute;
    local[12] = timer->backward_delta;
    local[13] = timer->x_to_b;
    local[14] = timer->gpu_h2d;
    local[15] = timer->gpu_compute;
    local[16] = timer->gpu_d2h;
    local[17] = meta != NULL ? meta->cpu_blas_ops : 0.0;
    local[18] = meta != NULL ? meta->gpu_blas_ops : 0.0;
    local[19] = meta != NULL ? meta->gpu_panel_ops : 0.0;
    local[20] = meta != NULL ? meta->below_threshold_ops : 0.0;
    local[21] = meta != NULL ? meta->cpu_blas_calls : 0.0;
    local[22] = meta != NULL ? meta->gpu_blas_calls : 0.0;
    local[23] = meta != NULL ? meta->gpu_panel_calls : 0.0;
    local[24] = meta != NULL ? meta->below_threshold_calls : 0.0;
    local[25] = meta != NULL ? meta->host_panel_copy_time : 0.0;
    local[26] = meta != NULL ? meta->gpu_panel_import_time : 0.0;
    local[27] = meta != NULL ? meta->gpu_schedule_upload_time : 0.0;
    local[28] = meta != NULL ? meta->x_cache_replicated_bytes : 0.0;
    local[29] = meta != NULL ? meta->x_cache_avoided_request_bytes : 0.0;
    local[30] = meta != NULL ? meta->x_cache_hits : 0.0;
    local[31] = meta != NULL ? meta->x_cache_misses : 0.0;
    local[32] = meta != NULL ? meta->x_cache_panels : 0.0;

    MPI_Comm_rank(grid3d->comm, &rank);
    MPI_Comm_size(grid3d->comm, &nprocs);
    for (int i = 0; i < SYMLDL_TIMER_COUNT; ++i) {
        local_max[i].val = local[i];
        local_max[i].rank = rank;
    }
    MPI_Reduce(local, maxv, SYMLDL_TIMER_COUNT, MPI_DOUBLE, MPI_MAX, 0,
               grid3d->comm);
    MPI_Reduce(local, sumv, SYMLDL_TIMER_COUNT, MPI_DOUBLE, MPI_SUM, 0,
               grid3d->comm);
    MPI_Reduce(local_max, global_max, SYMLDL_TIMER_COUNT, MPI_DOUBLE_INT,
               MPI_MAXLOC, 0, grid3d->comm);

    if (rank == 0) {
        static const char *names[SYMLDL_TIMER_COUNT] = {
            "metadata", "workspace", "B_to_X", "forward_xk",
            "forward_compute", "forward_values", "forward_apply",
            "diag_comm", "diag_compute", "x_cache_fill", "backward_values",
            "backward_compute", "backward_delta", "X_to_B",
            "gpu_h2d", "gpu_compute", "gpu_d2h",
            "cpu_blas_ops", "gpu_blas_ops", "gpu_panel_ops",
            "below_thresh_ops", "cpu_blas_calls", "gpu_blas_calls",
            "gpu_panel_calls", "below_thresh_calls",
            "host_panel_copy", "gpu_panel_attach", "gpu_schedule_h2d",
            "x_cache_bytes", "x_cache_avoided", "x_cache_hits",
            "x_cache_misses", "x_cache_panels"
        };
        printf("SymFact GPU3D V2 solve timing (max_rank / avg_rank / max_rank_id):\n");
        for (int i = 0; i < SYMLDL_TIMER_COUNT; ++i)
            printf("  %-17s %.6f / %.6f / %d\n", names[i], maxv[i],
                   sumv[i] / (double) nprocs, global_max[i].rank);
    }
}

static void
pdgstrs3d_symldl_distributed(superlu_dist_options_t *options, int_t n,
           dLUstruct_t * LUstruct, dScalePermstruct_t * ScalePermstruct,
           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, double *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    MPI_Comm global_comm = grid3d->comm;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    int_t *ilsum = Llu->ilsum;
    int *supernodeMask = trf3Dpartition ? trf3Dpartition->supernodeMask : NULL;
    int_t nsupers = supno[n - 1] + 1;
    int_t nlb = (trf3Dpartition != NULL &&
                 trf3Dpartition->symV2RowLocalIndex != NULL)
                    ? SUPERLU_MAX((int_t) 1,
                                  trf3Dpartition->symV2LocalRowCount)
                    : CEILING(nsupers, grid->nprow);
    int_t ldalsum = Llu->ldalsum;
    int_t x_count;
    int_t maxsup = 0;
    int global_rank, global_nprocs;
    int superlu_acc_offload = sp_ienv_dist(10, options);
    int superlu_n_gemm = sp_ienv_dist(7, options);
    pdgstrs3d_symldl_solve_meta_t *solve_meta;
    int *diag_owner;
    int_t *solve_order;
    pdgstrs3d_symldl_level_schedule_t *solve_schedule;
    pdgstrs3d_symldl_tree_comm_t *tree_comms;
    pdgstrs3d_symldl_panel_meta_t *panel_meta;
    pdgstrs3d_symldl_comm_meta_t *comm_meta;
    pdgstrs3d_symldl_workspace_t *workspace;
    double *x;
    double *xk_buf;
    double *diag_send_buf;
    double *diag_buf;
    double *delta_send_buf;
    double *delta_buf;
    double *gemm_buf;
    double *rhs_buf;
    MPI_Request *comm_reqs;
    double tx, tx_st;
    double ttmp;
    pdgstrs3d_symldl_timer_t symldl_timer;

    MPI_Comm_rank(global_comm, &global_rank);
    MPI_Comm_size(global_comm, &global_nprocs);
    memset(&symldl_timer, 0, sizeof(symldl_timer));

    for (int_t k = 0; k < nsupers; ++k)
        maxsup = SUPERLU_MAX(maxsup, SuperSize(k));

    x_count = pdgstrs3d_checked_workspace_count(ldalsum, nrhs, nlb, XK_H,
                                                "3D SymLDL solve x workspace");

    ttmp = SuperLU_timer_();
    solve_meta = (pdgstrs3d_symldl_solve_meta_t *)
        SOLVEstruct->symldl_v2_solve_meta;
    if (!pdgstrs3d_symldl_solve_meta_valid(
            solve_meta, n, nsupers, nrhs, Glu_persist, Llu, trf3Dpartition,
            supernodeMask, grid3d, global_nprocs, superlu_acc_offload,
            superlu_n_gemm))
        ABORT("SymLDL solve requires prebuilt V2 solve metadata.");
    solve_meta->reused = 1;
    pdgstrs3d_symldl_reset_offload_stats(solve_meta);
    diag_owner = solve_meta->diag_owner;
    solve_order = solve_meta->solve_order;
    solve_schedule = &solve_meta->solve_schedule;
    tree_comms = solve_meta->tree_comms;
    panel_meta = solve_meta->panel_meta;
    comm_meta = solve_meta->comm_meta;
    symldl_timer.metadata = SuperLU_timer_() - ttmp;

    ttmp = SuperLU_timer_();
    pdgstrs3d_symldl_workspace_prepare(solve_meta, x_count, maxsup, nrhs,
                                       global_nprocs);
    pdgstrs3d_symldl_prepare_host_factor_panels(solve_meta, nrhs);
    pdgstrs3d_symldl_gpu_prepare(solve_meta, maxsup, nrhs, grid3d);
    workspace = &solve_meta->work;
    x = workspace->x;
    xk_buf = workspace->xk_buf;
    diag_send_buf = workspace->diag_send_buf;
    diag_buf = workspace->diag_buf;
    delta_send_buf = workspace->delta_send_buf;
    delta_buf = workspace->delta_buf;
    gemm_buf = workspace->gemm_buf;
    rhs_buf = workspace->rhs_buf;
    comm_reqs = workspace->comm_reqs;
    symldl_timer.workspace = SuperLU_timer_() - ttmp;

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    xtrsTimer_t xtrsTimer;
    initTRStimer(&xtrsTimer, grid);

    tx = SuperLU_timer_();
    pdReDistribute3d_B_to_X_symv2(B, m_loc, nrhs, ldb, fst_row, ilsum, x,
                                  ScalePermstruct, Glu_persist,
                                  trf3Dpartition, grid3d, SOLVEstruct);
    xtrsTimer.t_pxReDistribute_B_to_X = SuperLU_timer_() - tx;
    symldl_timer.b_to_x = xtrsTimer.t_pxReDistribute_B_to_X;

    MPI_Barrier(grid3d->comm);
    tx_st = SuperLU_timer_();

    /* Forward solve with unit-lower L. X lives only on z=0 diagonal owners. */
    for (int_t level = 0; level < solve_schedule->nlevels; ++level) {
        int_t level_begin = solve_schedule->level_ptr[level];
        int_t level_end = solve_schedule->level_ptr[level + 1];
        int_t nctx = level_end - level_begin;
        int_t max_level_reqs = 1;
        int_t level_xk_count = 0;
        int_t level_send_vals_count = 0;
        int_t level_recv_vals_count = 0;
        int_t xk_offset = 0;
        int_t send_vals_offset = 0;
        int_t recv_vals_offset = 0;
        pdgstrs3d_symldl_node_ctx_t *ctxs =
            pdgstrs3d_symldl_workspace_prepare_level_contexts(
                workspace, nctx, "Malloc fails for SymLDL forward level contexts.");

        for (int_t ci = 0; ci < nctx; ++ci) {
            int_t k = solve_schedule->nodes[level_begin + ci];
            int_t tree = (trf3Dpartition && trf3Dpartition->supernode2treeMap)
                             ? trf3Dpartition->supernode2treeMap[k] : -1;
            pdgstrs3d_symldl_tree_comm_t *tree_comm =
                (tree_comms && tree >= 0) ? &tree_comms[tree] : NULL;
            pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
            pdgstrs3d_symldl_comm_meta_t *cmeta = &comm_meta[k];
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            int_t ksupc = SuperSize(k);

            if (tree_comm == NULL)
                ABORT("SymLDL solve is missing tree metadata for a forward panel.");
            if (!tree_comm->active || !cmeta->active)
                continue;

            ctx->active = 1;
            ctx->k = k;
            ctx->ksupc = ksupc;
            ctx->xk_count = pdgstrs3d_symldl_count_to_int(
                ksupc * nrhs, "SymLDL pivot block");
            ctx->root_rank = pdgstrs3d_symldl_rank_to_tree_rank(
                tree_comm, diag_owner[k]);
            ctx->solve_rank = tree_comm->rank;
            ctx->solve_comm = tree_comm->comm;
            ctx->kmeta = kmeta;
            ctx->cmeta = cmeta;

            level_xk_count = pdgstrs3d_symldl_count_sum(
                level_xk_count, ctx->xk_count, "SymLDL forward pivot buffer");
            if (cmeta->total_send > 0)
                level_send_vals_count = pdgstrs3d_symldl_count_sum(
                    level_send_vals_count, cmeta->total_send_vals,
                    "SymLDL forward send values");
            if (cmeta->total_recv > 0)
                level_recv_vals_count = pdgstrs3d_symldl_count_sum(
                    level_recv_vals_count, cmeta->total_recv_vals,
                    "SymLDL forward recv values");
            max_level_reqs += cmeta->xk_receiver_count + 1 +
                              2 * cmeta->nprocs;
        }

        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->xk_buf, &workspace->xk_cap,
            SUPERLU_MAX(level_xk_count, (int_t) 1),
            "Malloc fails for SymLDL forward pivot buffer.");
        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->send_vals_buf, &workspace->send_vals_cap,
            SUPERLU_MAX(level_send_vals_count, (int_t) 1),
            "Malloc fails for SymLDL forward send values.");
        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->recv_vals_buf, &workspace->recv_vals_cap,
            SUPERLU_MAX(level_recv_vals_count, (int_t) 1),
            "Malloc fails for SymLDL forward recv values.");
        pdgstrs3d_symldl_grow_request_buffer(
            &workspace->comm_reqs, &workspace->comm_reqs_cap,
            max_level_reqs, "Malloc fails for SymLDL forward level requests.");
        comm_reqs = workspace->comm_reqs;

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            int_t ksupc = ctx->ksupc;

            if (!ctx->active)
                continue;

            ctx->xk_buf = &workspace->xk_buf[xk_offset];
            xk_offset += ctx->xk_count;
            if (cmeta->total_send > 0) {
                ctx->send_vals = &workspace->send_vals_buf[send_vals_offset];
                send_vals_offset += cmeta->total_send_vals;
            }
            if (cmeta->total_recv > 0) {
                ctx->recv_vals = &workspace->recv_vals_buf[recv_vals_offset];
                recv_vals_offset += cmeta->total_recv_vals;
            }

            if (global_rank == diag_owner[ctx->k]) {
                int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, ctx->k);
                double *xk = &x[X_BLK(lk)];
                for (int rhs = 0; rhs < nrhs; ++rhs)
                    for (int_t i = 0; i < ksupc; ++i)
                        ctx->xk_buf[i + (int_t) rhs * ksupc] =
                            xk[i + (int_t) rhs * ksupc];
            }
        }

        ttmp = SuperLU_timer_();
        int nreq_xk = 0;
        for (int_t ci = 0; ci < nctx; ++ci)
            if (ctxs[ci].active)
                pdgstrs3d_symldl_post_xk(ctxs[ci].xk_buf,
                                         ctxs[ci].xk_count,
                                         ctxs[ci].solve_rank,
                                         ctxs[ci].root_rank,
                                         ctxs[ci].cmeta,
                                         ctxs[ci].solve_comm,
                                         comm_reqs, &nreq_xk);
        if (nreq_xk > 0)
            MPI_Waitall(nreq_xk, comm_reqs, MPI_STATUSES_IGNORE);
        symldl_timer.forward_xk += SuperLU_timer_() - ttmp;

        ttmp = SuperLU_timer_();
        double forward_direct_ops = 0.0;
        double forward_direct_calls = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(+:forward_direct_ops,forward_direct_calls)
#endif
        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active || !kmeta->has_panel || cmeta->total_send <= 0)
                continue;
            double panel_ops =
                pdgstrs3d_symldl_panel_solve_ops(kmeta, ctx->ksupc, nrhs);
            if (pdgstrs3d_symldl_should_gpu_ops(solve_meta, panel_ops) ||
                !pdgstrs3d_symldl_use_direct_cpu_panel(
                    solve_meta, panel_ops, nrhs))
                continue;
            pdgstrs3d_symldl_cpu_forward_panel_direct(
                kmeta, cmeta, ctx->ksupc, nrhs, ctx->xk_buf,
                ctx->send_vals);
            forward_direct_ops += panel_ops;
            forward_direct_calls += 1.0;
        }
        pdgstrs3d_symldl_note_cpu_direct_panel(
            solve_meta, forward_direct_ops, forward_direct_calls);
        stat->ops[SOLVE] += forward_direct_ops;

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active || !kmeta->has_panel || cmeta->total_send <= 0)
                continue;
            double panel_ops =
                pdgstrs3d_symldl_panel_solve_ops(kmeta, ctx->ksupc, nrhs);
            if (!pdgstrs3d_symldl_should_gpu_ops(solve_meta, panel_ops) &&
                pdgstrs3d_symldl_use_direct_cpu_panel(
                    solve_meta, panel_ops, nrhs))
                continue;
            if (pdgstrs3d_symldl_try_gpu_forward_panel(
                    solve_meta, ctx->k, kmeta, ctx->ksupc, nrhs,
                    ctx->xk_buf, cmeta->total_send, ctx->send_vals)) {
                stat->ops[SOLVE] +=
                    panel_ops;
                continue;
            }
            double *lusup = kmeta->lusup;
            int_t nsupr = kmeta->nsupr;
            for (int_t block = 0; block < kmeta->nblocks; ++block) {
                int_t nbrow = kmeta->block_nbrow[block];
                int_t row_start = kmeta->block_row_start[block];
                int_t luptr = kmeta->block_luptr[block];
                pdgstrs3d_symldl_dispatch_gemm(solve_meta, ctx->k, luptr,
                                       "N", "N", nbrow, nrhs, ctx->ksupc,
                                       1.0, &lusup[luptr], nsupr,
                                       ctx->xk_buf, ctx->ksupc,
                                       0.0, gemm_buf, nbrow);
                for (int_t r = 0; r < nbrow; ++r) {
                    int_t row = row_start + r;
                    int pos = cmeta->row_to_send_pos[row];
                    for (int rhs = 0; rhs < nrhs; ++rhs)
                        ctx->send_vals[pos * nrhs + rhs] =
                            -gemm_buf[r + (int_t) rhs * nbrow];
                }
                stat->ops[SOLVE] += 2.0 * (double) nbrow *
                                    (double) nrhs * (double) ctx->ksupc;
            }
        }
        symldl_timer.forward_compute += SuperLU_timer_() - ttmp;

        ttmp = SuperLU_timer_();
        int nreq_values = 0;
        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active)
                continue;
            pdgstrs3d_symldl_post_exchange_double(
                ctx->send_vals, cmeta->send_val_counts,
                cmeta->send_val_displs, ctx->recv_vals,
                cmeta->recv_val_counts, cmeta->recv_val_displs,
                cmeta->nprocs, ctx->solve_rank, LSUM,
                ctx->solve_comm, comm_reqs, &nreq_values);
        }
        if (nreq_values > 0)
            MPI_Waitall(nreq_values, comm_reqs, MPI_STATUSES_IGNORE);
        symldl_timer.forward_values += SuperLU_timer_() - ttmp;

        ttmp = SuperLU_timer_();
        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active)
                continue;
            for (int row = 0; row < cmeta->total_recv; ++row) {
                int_t grow = cmeta->recv_rows[row];
                int_t gsup = BlockNum(grow);
                if (global_rank == diag_owner[gsup]) {
                    int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, gsup);
                    int_t rel = grow - FstBlockC(gsup);
                    int_t gsupsz = SuperSize(gsup);
                    double *xg = &x[X_BLK(lk)];
                    for (int rhs = 0; rhs < nrhs; ++rhs)
                        xg[rel + (int_t) rhs * gsupsz] +=
                            ctx->recv_vals[row * nrhs + rhs];
                }
            }
        }
        symldl_timer.forward_apply += SuperLU_timer_() - ttmp;

    }
    xtrsTimer.t_forwardSolve = SuperLU_timer_() - tx_st;
    xk_buf = workspace->xk_buf;

    /* Dense diagonal apply z = D^{-1} y on the canonical diagonal owner. */
    tx = SuperLU_timer_();
    for (int_t level = 0; level < solve_schedule->nlevels; ++level) {
    for (int_t ko = solve_schedule->level_ptr[level];
         ko < solve_schedule->level_ptr[level + 1]; ++ko) {
        int_t k = solve_schedule->nodes[ko];
        int_t tree = (trf3Dpartition && trf3Dpartition->supernode2treeMap)
                         ? trf3Dpartition->supernode2treeMap[k] : -1;
        pdgstrs3d_symldl_tree_comm_t *tree_comm =
            (tree_comms && tree >= 0) ? &tree_comms[tree] : NULL;
        pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
        pdgstrs3d_symldl_comm_meta_t *cmeta = &comm_meta[k];
        int_t ksupc = SuperSize(k);
        int block_count = pdgstrs3d_symldl_count_to_int(ksupc * nrhs,
                                                        "SymLDL diagonal block");

        if (tree_comm == NULL)
            ABORT("SymLDL solve is missing tree metadata for a diagonal block.");
        if (!tree_comm->active || !cmeta->active)
            continue;
        if (global_rank != diag_owner[k])
            continue;
        if (!kmeta->has_diag)
            ABORT("SymLDL solve diagonal owner is missing inverse diagonal block.");

        int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, k);
        double *xk = &x[X_BLK(lk)];
        for (int rhs = 0; rhs < nrhs; ++rhs)
            for (int_t i = 0; i < ksupc; ++i)
                xk_buf[i + (int_t) rhs * ksupc] =
                    xk[i + (int_t) rhs * ksupc];

        ttmp = SuperLU_timer_();
        pdgstrs3d_symldl_dispatch_gemm(solve_meta, k, kmeta->diag_luptr,
                               "N", "N", ksupc, nrhs, ksupc,
                               1.0, &kmeta->lusup[kmeta->diag_luptr],
                               kmeta->nsupr, xk_buf, ksupc,
                               0.0, diag_buf, ksupc);
        symldl_timer.diag_compute += SuperLU_timer_() - ttmp;
        stat->ops[SOLVE] += 2.0 * (double) ksupc * (double) ksupc *
                            (double) nrhs;

        for (int rhs = 0; rhs < nrhs; ++rhs)
            for (int_t i = 0; i < ksupc; ++i)
                xk[i + (int_t) rhs * ksupc] =
                    diag_buf[i + (int_t) rhs * ksupc];
    }
    }
    (void) tx;

    /* Backward solve with L^T using per-tree replicated row values. */
    tx = SuperLU_timer_();
    for (int_t level = solve_schedule->nlevels; level > 0; --level) {
        int_t level_begin = solve_schedule->level_ptr[level - 1];
        int_t level_end = solve_schedule->level_ptr[level];
        int_t nctx = level_end - level_begin;
        int_t max_level_reqs = 1;
        int_t level_delta_count = 0;
        int_t level_delta_recv_count = 0;
        int_t delta_offset = 0;
        int_t delta_recv_offset = 0;
        pdgstrs3d_symldl_node_ctx_t *ctxs =
            pdgstrs3d_symldl_workspace_prepare_level_contexts(
                workspace, nctx, "Malloc fails for SymLDL backward level contexts.");

        for (int_t ci = 0; ci < nctx; ++ci) {
            int_t k = solve_schedule->nodes[level_begin + ci];
            int_t tree = (trf3Dpartition && trf3Dpartition->supernode2treeMap)
                             ? trf3Dpartition->supernode2treeMap[k] : -1;
            pdgstrs3d_symldl_tree_comm_t *tree_comm =
                (tree_comms && tree >= 0) ? &tree_comms[tree] : NULL;
            pdgstrs3d_symldl_panel_meta_t *kmeta = &panel_meta[k];
            pdgstrs3d_symldl_comm_meta_t *cmeta = &comm_meta[k];
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            int_t ksupc = SuperSize(k);

            if (tree_comm == NULL)
                ABORT("SymLDL solve is missing tree metadata for a backward panel.");
            if (!tree_comm->active || !cmeta->active)
                continue;

            ctx->active = 1;
            ctx->k = k;
            ctx->ksupc = ksupc;
            ctx->delta_count = pdgstrs3d_symldl_count_to_int(
                ksupc * nrhs, "SymLDL backward delta");
            ctx->root_rank = pdgstrs3d_symldl_rank_to_tree_rank(
                tree_comm, diag_owner[k]);
            ctx->solve_rank = tree_comm->rank;
            ctx->solve_comm = tree_comm->comm;
            ctx->kmeta = kmeta;
            ctx->cmeta = cmeta;

            level_delta_count = pdgstrs3d_symldl_count_sum(
                level_delta_count, ctx->delta_count,
                "SymLDL backward delta buffers");
            if (ctx->solve_rank == ctx->root_rank &&
                cmeta->xk_receiver_count > 0) {
                int_t recv_count = pdgstrs3d_checked_workspace_count(
                    cmeta->xk_receiver_count, ctx->delta_count, 0, 0,
                    "SymLDL backward delta receive buffer");
                level_delta_recv_count = pdgstrs3d_symldl_count_sum(
                    level_delta_recv_count, recv_count,
                    "SymLDL backward delta receive buffer");
            }

            max_level_reqs += cmeta->xk_receiver_count + 1;
        }

        ttmp = SuperLU_timer_();
        symldl_timer.x_cache_fill += pdgstrs3d_symldl_x_cache_fill(
            solve_meta, x, nrhs, ilsum, grid3d, ctxs, nctx);

        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->delta_send_buf, &workspace->delta_send_cap,
            SUPERLU_MAX(level_delta_count, (int_t) 1),
            "Malloc fails for SymLDL backward delta buffers.");
        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->delta_buf, &workspace->delta_cap,
            SUPERLU_MAX(level_delta_count, (int_t) 1),
            "Malloc fails for SymLDL backward delta buffers.");
        pdgstrs3d_symldl_grow_double_buffer(
            &workspace->delta_recv_buf, &workspace->delta_recv_cap,
            SUPERLU_MAX(level_delta_recv_count, (int_t) 1),
            "Malloc fails for SymLDL backward delta receive buffer.");
        pdgstrs3d_symldl_grow_request_buffer(
            &workspace->comm_reqs, &workspace->comm_reqs_cap,
            max_level_reqs, "Malloc fails for SymLDL backward level requests.");
        comm_reqs = workspace->comm_reqs;
        pdgstrs3d_symldl_zero_double_buffer(
            workspace->delta_send_buf, level_delta_count,
            "SymLDL backward delta send buffer");
        pdgstrs3d_symldl_zero_double_buffer(
            workspace->delta_buf, level_delta_count,
            "SymLDL backward delta buffer");

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;

            if (!ctx->active)
                continue;

            ctx->delta_send_buf = &workspace->delta_send_buf[delta_offset];
            ctx->delta_buf = &workspace->delta_buf[delta_offset];
            delta_offset += ctx->delta_count;
            if (ctx->solve_rank == ctx->root_rank &&
                cmeta->xk_receiver_count > 0) {
                int_t recv_count = pdgstrs3d_checked_workspace_count(
                    cmeta->xk_receiver_count, ctx->delta_count, 0, 0,
                    "SymLDL backward delta receive buffer");
                ctx->delta_recv_buf =
                    &workspace->delta_recv_buf[delta_recv_offset];
                delta_recv_offset += recv_count;
            }
            if (kmeta->has_panel && cmeta->total_send > 0) {
                pdgstrs3d_symldl_x_cache_t *cache =
                    &solve_meta->x_cache[ctx->k];
                if (cache->values == NULL ||
                    cache->row_count != kmeta->row_count)
                    ABORT("SymLDL backward solve missing replicated X cache.");
                ctx->row_values = cache->values;
            }
        }

        ttmp = SuperLU_timer_();
        double backward_direct_ops = 0.0;
        double backward_direct_calls = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(+:backward_direct_ops,backward_direct_calls)
#endif
        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active || !kmeta->has_panel || cmeta->total_send <= 0)
                continue;
            double panel_ops =
                pdgstrs3d_symldl_panel_solve_ops(kmeta, ctx->ksupc, nrhs);
            if (pdgstrs3d_symldl_should_gpu_ops(solve_meta, panel_ops) ||
                !pdgstrs3d_symldl_use_direct_cpu_panel(
                    solve_meta, panel_ops, nrhs))
                continue;
            pdgstrs3d_symldl_cpu_backward_panel_direct(
                kmeta, ctx->ksupc, nrhs, ctx->row_values,
                ctx->delta_send_buf);
            backward_direct_ops += panel_ops;
            backward_direct_calls += 1.0;
        }
        pdgstrs3d_symldl_note_cpu_direct_panel(
            solve_meta, backward_direct_ops, backward_direct_calls);
        stat->ops[SOLVE] += backward_direct_ops;

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            pdgstrs3d_symldl_panel_meta_t *kmeta = ctx->kmeta;
            pdgstrs3d_symldl_comm_meta_t *cmeta = ctx->cmeta;
            if (!ctx->active || !kmeta->has_panel || cmeta->total_send <= 0)
                continue;
            double panel_ops =
                pdgstrs3d_symldl_panel_solve_ops(kmeta, ctx->ksupc, nrhs);
            if (!pdgstrs3d_symldl_should_gpu_ops(solve_meta, panel_ops) &&
                pdgstrs3d_symldl_use_direct_cpu_panel(
                    solve_meta, panel_ops, nrhs))
                continue;
            if (pdgstrs3d_symldl_try_gpu_backward_panel(
                    solve_meta, ctx->k, kmeta, ctx->ksupc, nrhs,
                    ctx->row_values, ctx->delta_send_buf)) {
                stat->ops[SOLVE] +=
                    panel_ops;
                continue;
            }
            double *lusup = kmeta->lusup;
            int_t nsupr = kmeta->nsupr;
            int seq = 0;
            for (int_t block = 0; block < kmeta->nblocks; ++block) {
                int_t nbrow = kmeta->block_nbrow[block];
                int_t luptr = kmeta->block_luptr[block];
                for (int rhs = 0; rhs < nrhs; ++rhs)
                    for (int_t r = 0; r < nbrow; ++r)
                        rhs_buf[r + (int_t) rhs * nbrow] =
                            ctx->row_values[(seq + r) * nrhs + rhs];
                pdgstrs3d_symldl_dispatch_gemm(solve_meta, ctx->k, luptr,
                                       "T", "N", ctx->ksupc, nrhs, nbrow,
                                       -1.0, &lusup[luptr], nsupr,
                                       rhs_buf, nbrow,
                                       1.0, ctx->delta_send_buf, ctx->ksupc);
                seq += nbrow;
                stat->ops[SOLVE] += 2.0 * (double) nbrow *
                                    (double) nrhs * (double) ctx->ksupc;
            }
        }
        symldl_timer.backward_compute += SuperLU_timer_() - ttmp;

        ttmp = SuperLU_timer_();
        int nreq_delta = 0;
        for (int_t ci = 0; ci < nctx; ++ci)
            if (ctxs[ci].active)
                pdgstrs3d_symldl_post_reduce_delta(&ctxs[ci],
                                                   comm_reqs, &nreq_delta);
        if (nreq_delta > 0)
            MPI_Waitall(nreq_delta, comm_reqs, MPI_STATUSES_IGNORE);
        for (int_t ci = 0; ci < nctx; ++ci)
            if (ctxs[ci].active)
                pdgstrs3d_symldl_finish_reduce_delta(&ctxs[ci]);
        symldl_timer.backward_delta += SuperLU_timer_() - ttmp;

        for (int_t ci = 0; ci < nctx; ++ci) {
            pdgstrs3d_symldl_node_ctx_t *ctx = &ctxs[ci];
            if (!ctx->active || global_rank != diag_owner[ctx->k])
                continue;
            int_t lk = pdgstrs3d_symv2_row_index(trf3Dpartition, ctx->k);
            double *xk = &x[X_BLK(lk)];
            for (int rhs = 0; rhs < nrhs; ++rhs)
                for (int_t c = 0; c < ctx->ksupc; ++c)
                    xk[c + (int_t) rhs * ctx->ksupc] +=
                        ctx->delta_buf[c + (int_t) rhs * ctx->ksupc];
        }

    }
    xtrsTimer.t_backwardSolve = SuperLU_timer_() - tx;

    MPI_Barrier(grid3d->comm);
    stat->utime[SOLVE] = SuperLU_timer_() - tx_st;

    tx = SuperLU_timer_();
    pdReDistribute3d_X_to_B_symv2(n, B, m_loc, ldb, fst_row, nrhs, x,
                                  ilsum, ScalePermstruct, Glu_persist,
                                  trf3Dpartition, grid3d, SOLVEstruct);
    xtrsTimer.t_pxReDistribute_X_to_B = SuperLU_timer_() - tx;
    symldl_timer.x_to_b = xtrsTimer.t_pxReDistribute_X_to_B;

    reduceStat(SOLVE, stat, grid3d);
    pdgstrs3d_symldl_gpu_take_timers(solve_meta, &symldl_timer);
    if (pdgstrs3d_symldl_env_enabled("GPU3DV2_SYM_SOLVE_TIMING"))
        pdgstrs3d_symldl_timer_print(&symldl_timer, solve_meta, grid3d);

#if ( PRNTlevel >= 1 )
    printTRStimer(&xtrsTimer, grid3d);
#endif

    return;
}

/*! \brief
 *
 *   Experimental LDL-native solve for the SymFact GPU3D v2 factor path.
 *   This uses the tree-scheduled distributed solve for all process grids,
 *   including the 1x1x1 local case, while keeping the existing B<->X
 *   redistribution contract.
 */
void
pdgstrs3d_symldl (superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
           dScalePermstruct_t * ScalePermstruct,
           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, double *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info)
{
    gridinfo_t *grid = &(grid3d->grid2d);

    *info = 0;
    if (n < 0) *info = -1;
    else if (nrhs < 0) *info = -9;
    if (*info) {
        pxerr_dist("PDGSTRS_SYMLDL", grid, -*info);
        return;
    }
    if (nrhs == 0) return;
    pdgstrs3d_symldl_distributed(options, n, LUstruct, ScalePermstruct,
                                 trf3Dpartition, grid3d, B, m_loc,
                                 fst_row, ldb, nrhs, SOLVEstruct, stat, info);
    return;
}                               /* pdgstrs3d_symldl */




int_t pdgsTrForwardSolve3d(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                           dScalePermstruct_t * ScalePermstruct,
                           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                           double *x3d, double *lsum3d,
                           dxT_struct *xT_s,
                           double * recvbuf,
                           MPI_Request * send_req, int nrhs,
                           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    double zero = 0.0;
    int_t* xsup = Glu_persist->xsup;

    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;

    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    int_t *ilsum = Llu->ilsum;

    int_t knsupc = sp_ienv_dist (3,options);
    int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);
    double* rtemp;
    if (!(rtemp = doubleCalloc_dist (maxrecvsz)))
        ABORT ("Malloc fails for rtemp[].");

    /**
     *  Loop over all the levels from root to leaf
     */
    int_t ii = 0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t knsupc = SuperSize (k);
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid); /* Local block number. */
            int_t il = LSUM_BLK (lk);
	        lsum3d[il - LSUM_H] = k; /* Block number prepended in the header. */

        }
        ii += knsupc;
    }

    /*initilize lsum to zero*/
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t knsupc = SuperSize (k);
            int_t lk = LBi (k, grid);
            int_t il = LSUM_BLK (lk);
            double* dest = &lsum3d[il];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
                    dest[i + j * knsupc] = zero;
            }
        }
    }


    Llu->SolveMsgSent = 0;
    for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        double tx = SuperLU_timer_();
        /* if I participate in this level */
        if (!myZeroTrIdxs[ilvl])
        {
            int_t tree = myTreeIdxs[ilvl];

            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

            /*main loop over all the super nodes*/
            if (sforest)
            {
                if (ilvl == 0)
                    dleafForestForwardSolve3d(options, tree, n, LUstruct,
                                              ScalePermstruct, trf3Dpartition, grid3d,
                                              x3d,  lsum3d, recvbuf, rtemp,
                                              send_req,  nrhs, SOLVEstruct,  stat, xtrsTimer);
                else
                    dnonLeafForestForwardSolve3d(tree, LUstruct,
                                                ScalePermstruct, trf3Dpartition, grid3d,  x3d,  lsum3d, xT_s, recvbuf, rtemp,
                                                send_req, nrhs, SOLVEstruct,  stat, xtrsTimer);

            }
            if (ilvl != maxLvl - 1)
            {
                /* code */
                int_t myGrid = grid3d->zscp.Iam;


                int_t sender, receiver;
                if ((myGrid % (1 << (ilvl + 1))) == 0)
                {
                    sender = myGrid + (1 << ilvl);
                    receiver = myGrid;
                }
                else
                {

                    sender = myGrid;
                    receiver = myGrid - (1 << ilvl);
                }
                double tx = SuperLU_timer_();
                for (int_t alvl = ilvl + 1; alvl <  maxLvl; ++alvl)
                {
                    /* code */
                    int_t treeId = myTreeIdxs[alvl];
                    dfsolveReduceLsum3d(treeId, sender, receiver, lsum3d, recvbuf, nrhs,
                                       trf3Dpartition, LUstruct, grid3d,xtrsTimer );
                }
                xtrsTimer->trs_comm_z += SuperLU_timer_() - tx;
            }
        }
        xtrsTimer->tfs_tree[ilvl] = SuperLU_timer_() - tx;
    }

    double tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    Llu->SolveMsgSent = 0;
    xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
    SUPERLU_FREE(rtemp);


    return 0;
}



int_t pdgsTrForwardSolve3d_newsolve(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                           dScalePermstruct_t * ScalePermstruct,
                           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                           double *x3d, double *lsum3d,
                           double * recvbuf,
                           MPI_Request * send_req, int nrhs,
                           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    double zero = 0.0;
    int_t* xsup = Glu_persist->xsup;

    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;

    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    int_t *ilsum = Llu->ilsum;

    int_t knsupc = sp_ienv_dist (3,options);
    int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);
    double* rtemp;
    if (!(rtemp = doubleCalloc_dist (maxrecvsz)))
        ABORT ("Malloc fails for rtemp[].");

/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    int_t ii = 0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t knsupc = SuperSize (k);
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid); /* Local block number. */
            int_t il = LSUM_BLK (lk);
	        lsum3d[il - LSUM_H] = k; /* Block number prepended in the header. */
        }
        ii += knsupc;
    }

    /*initilize lsum to zero*/
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t knsupc = SuperSize (k);
            int_t lk = LBi (k, grid);
            int_t il = LSUM_BLK (lk);
            double* dest = &lsum3d[il];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
                    dest[i + j * knsupc] = zero;
            }
        }
    }
}

    Llu->SolveMsgSent = 0;

    double tx = SuperLU_timer_();

if (get_new3dsolvetreecomm()){
    dForwardSolve3d_newsolve_reusepdgstrs(options, n, LUstruct,
                                ScalePermstruct, trf3Dpartition->supernodeMask, grid3d,
                                x3d, lsum3d, nrhs, SOLVEstruct, stat, xtrsTimer);
}else{
    dleafForestForwardSolve3d_newsolve(options, n, LUstruct,
                                ScalePermstruct, trf3Dpartition, grid3d,
                                x3d,  lsum3d, recvbuf, rtemp,
                                send_req,  nrhs, SOLVEstruct,  stat, xtrsTimer);
}


    xtrsTimer->tfs_tree[0] = SuperLU_timer_() - tx;
    tx = SuperLU_timer_();
/* skip send_req on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
}
    Llu->SolveMsgSent = 0;
    xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
    SUPERLU_FREE(rtemp);
    return 0;
}





int_t pdgsTrBackSolve3d(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                        dScalePermstruct_t * ScalePermstruct,
                        dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                        double *x3d, double *lsum3d,
                        dxT_struct *xT_s,
                        double * recvbuf,
                        MPI_Request * send_req, int nrhs,
                        dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    // printf("Using pdgsTrBackSolve3d_2d \n");

    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    double zero = 0.0;
    int_t* xsup = Glu_persist->xsup;

    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;

    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    int_t *ilsum = Llu->ilsum;

    /**
     *  Loop over all the levels from root to leaf
     */

    /*initilize lsum to zero*/
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t knsupc = SuperSize (k);
            int_t lk = LBi (k, grid);
            int_t il = LSUM_BLK (lk);
            double* dest = &lsum3d[il];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
                    dest[i + j * knsupc] = zero;
            }
        }
    }

    /**
     * Adding dlsumBmod_buff_t* lbmod_buf
     */

    dlsumBmod_buff_t lbmod_buf;
    int_t nsupc = sp_ienv_dist (3,options);
    dinitLsumBmod_buff(nsupc, nrhs, &lbmod_buf);

    int_t numTrees = 2 * grid3d->zscp.Np - 1;
    int_t nLeafTrees = grid3d->zscp.Np;
    Llu->SolveMsgSent = 0;
    for (int_t ilvl = maxLvl - 1; ilvl >= 0  ; --ilvl)
    {
        /* code */
        double tx = SuperLU_timer_();
        if (!myZeroTrIdxs[ilvl])
        {
            double tx = SuperLU_timer_();
            dbsolve_Xt_bcast(ilvl, xT_s, nrhs, trf3Dpartition,
                            LUstruct, grid3d,xtrsTimer );
            xtrsTimer->trs_comm_z += SuperLU_timer_() - tx;


            int_t tree = myTreeIdxs[ilvl];

            int_t trParent = (tree + 1) / 2  - 1;
            tx = SuperLU_timer_();
            while (trParent > -1 )
            {
                dlasum_bmod_Tree(trParent, tree, lsum3d, x3d,  xT_s, nrhs, &lbmod_buf,
                                 LUstruct, trf3Dpartition, grid3d, stat);
                trParent = (trParent + 1) / 2 - 1;

            }
            xtrsTimer->tbs_compute += SuperLU_timer_() - tx;


            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

            /*main loop over all the super nodes*/
            if (sforest)
            {
                if (ilvl == 0)
                    dleafForestBackSolve3d(options, tree, n, LUstruct,
                                           ScalePermstruct, trf3Dpartition, grid3d, x3d,  lsum3d, recvbuf,
                                           send_req,  nrhs, &lbmod_buf,
                                            SOLVEstruct,  stat, xtrsTimer);
                else
                    dnonLeafForestBackSolve3d(tree, LUstruct,
                                             ScalePermstruct, trf3Dpartition, grid3d,  x3d,  lsum3d, xT_s, recvbuf,
                                             send_req, nrhs, &lbmod_buf,
                                              SOLVEstruct,  stat, xtrsTimer);


            }
        }
        xtrsTimer->tbs_tree[ilvl] = SuperLU_timer_() - tx;
    }
    double tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
    Llu->SolveMsgSent = 0;
    dfreeLsumBmod_buff(&lbmod_buf);

    return 0;
}




int_t pdgsTrBackSolve3d_newsolve(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                        dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                        double *x3d, double *lsum3d,
                        double * recvbuf,
                        MPI_Request * send_req, int nrhs,
                        dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{

    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    double zero = 0.0;
    int_t* xsup = Glu_persist->xsup;

    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;

    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    int_t *ilsum = Llu->ilsum;


/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    /*initilize lsum to zero*/
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t knsupc = SuperSize (k);
            int_t lk = LBi (k, grid);
            int_t il = LSUM_BLK (lk);
            double* dest = &lsum3d[il];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
                    dest[i + j * knsupc] = zero;
            }
        }
    }
}

    /**
     * Adding dlsumBmod_buff_t* lbmod_buf
     */

    dlsumBmod_buff_t lbmod_buf;
    int_t nsupc = sp_ienv_dist (3,options);
    dinitLsumBmod_buff(nsupc, nrhs, &lbmod_buf);

    Llu->SolveMsgSent = 0;
    double tx = SuperLU_timer_();

if (get_new3dsolvetreecomm()){
    dBackSolve3d_newsolve_reusepdgstrs(options, n, LUstruct,
                                trf3Dpartition->supernodeMask, grid3d,
                                x3d, lsum3d, nrhs, SOLVEstruct, stat, xtrsTimer);
}else{
    dleafForestBackSolve3d_newsolve(options, n, LUstruct, trf3Dpartition, grid3d, x3d,  lsum3d, recvbuf,
                            send_req,  nrhs, &lbmod_buf, SOLVEstruct,  stat, xtrsTimer);
}

    xtrsTimer->tbs_tree[0] = SuperLU_timer_() - tx;

    tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
    Llu->SolveMsgSent = 0;

    dfreeLsumBmod_buff(&lbmod_buf);

    return 0;
}
