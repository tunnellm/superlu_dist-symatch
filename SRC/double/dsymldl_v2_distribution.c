#include "superlu_ddefs.h"
#include <stdlib.h>
#include <string.h>

static int dSymV2CheckPartition(dtrf3Dpartition_t *trf3Dpart)
{
    return trf3Dpart != NULL &&
           trf3Dpart->symV2PanelRoot != NULL &&
           trf3Dpart->symV2DiagRoot != NULL &&
           trf3Dpart->symV2PanelLocalIndex != NULL &&
           trf3Dpart->symV2RowLocalIndex != NULL;
}

static void dSymV2DistributeTrace(gridinfo3d_t *grid3d, const char *msg)
{
    const char *env = getenv("GPU3DV2_TRACE");
    if (env == NULL || env[0] == '\0' || env[0] == '0')
        return;
    fprintf(stderr, "[sym-v2-trace] rank %d: distribute %s\n",
            grid3d ? grid3d->iam : -1, msg);
    fflush(stderr);
}

static void dSymV2MoveDiagBlockFirst(int_t *index, int_t *loc,
                                     int_t nrbl, int_t jb)
{
    int_t diag_pos = -1;

    for (int_t block = 0; block < nrbl; ++block)
    {
        int_t lptr = loc[block + nrbl];
        if (index[lptr] == jb)
        {
            diag_pos = block;
            break;
        }
    }

    if (diag_pos < 0)
        ABORT("SymFact V2 L-only distribution missing diagonal block.");

    if (diag_pos == 0)
        return;

    for (int_t field = 0; field < 3; ++field)
    {
        int_t base = field * nrbl;
        int_t tmp = loc[base];
        loc[base] = loc[base + diag_pos];
        loc[base + diag_pos] = tmp;
    }
}

static int dSymV2PanelRoot(dtrf3Dpartition_t *trf3Dpart, int_t k)
{
    return trf3Dpart->symV2PanelRoot[k];
}

static int dSymV2DiagRoot(dtrf3Dpartition_t *trf3Dpart, int_t k)
{
    return trf3Dpart->symV2DiagRoot[k];
}

static int_t dSymV2PanelLocalIndex(dtrf3Dpartition_t *trf3Dpart, int_t k)
{
    return trf3Dpart->symV2PanelLocalIndex[k];
}

static int_t dSymV2RowLocalIndex(dtrf3Dpartition_t *trf3Dpart, int_t k)
{
    return trf3Dpart->symV2RowLocalIndex[k];
}

static int dSymV2EntryOwner(dtrf3Dpartition_t *trf3Dpart, int_t *supno,
                            gridinfo_t *grid, int_t *irow, int_t *jcol)
{
    int_t gbi = BlockNum(*irow);
    int_t gbj = BlockNum(*jcol);

    if (gbi < gbj)
    {
        int_t tmp = *irow;
        *irow = *jcol;
        *jcol = tmp;
        tmp = gbi;
        gbi = gbj;
        gbj = tmp;
    }

    return PNUM(dSymV2DiagRoot(trf3Dpart, gbi),
                dSymV2PanelRoot(trf3Dpart, gbj), grid);
}

static int_t dSymV2RedistributeLowerToLDL(dtrf3Dpartition_t *trf3Dpart,
                                          int_t *supno,
                                          int_t n, gridinfo_t *grid,
                                          int_t **colptr, int_t **rowind,
                                          double **a)
{
    int iam = grid->iam;
    int procs = grid->nprow * grid->npcol;
    int_t nnz_old = (*colptr != NULL) ? (*colptr)[n] : 0;
    int_t *nnzToRecv = intCalloc_dist(2 * procs);
    int_t *nnzToSend = nnzToRecv + procs;
    int_t *ptr_to_send = NULL;
    int_t **ia_send = NULL;
    double **aij_send = NULL;
    int_t *index = NULL;
    double *nzval = NULL;
    int_t *ia = NULL;
    int_t *ja = NULL;
    double *aij = NULL;
    int_t *itemp = NULL;
    double *dtemp = NULL;
    MPI_Request *send_req = NULL;
    MPI_Status status;
    int_t maxnnzToRecv = 0;
    int_t SendCnt = 0;
    int_t RecvCnt = 0;
    int_t nnz_loc = 0;

    if (!dSymV2CheckPartition(trf3Dpart))
        ABORT("SymFact V2 LDL distribution metadata is not initialized.");

    for (int_t j = 0; j < n; ++j)
    {
        for (int_t p0 = (*colptr)[j]; p0 < (*colptr)[j + 1]; ++p0)
        {
            int_t irow = (*rowind)[p0];
            int_t jcol = j;
            int dest = dSymV2EntryOwner(trf3Dpart, supno, grid, &irow, &jcol);
            ++nnzToSend[dest];
        }
    }

    MPI_Alltoall(nnzToSend, 1, mpi_int_t, nnzToRecv, 1, mpi_int_t,
                 grid->comm);

    for (int p = 0; p < procs; ++p)
    {
        if (p != iam)
        {
            SendCnt += nnzToSend[p];
            RecvCnt += nnzToRecv[p];
            maxnnzToRecv = SUPERLU_MAX(maxnnzToRecv, nnzToRecv[p]);
        }
        else
        {
            nnz_loc += nnzToRecv[p];
        }
    }

    int_t total = nnz_loc + RecvCnt;
    if (total > 0)
    {
        if (!(ia = intMalloc_dist(2 * total)))
            ABORT("Malloc fails for SymFact V2 redistributed ia[].");
        if (!(aij = doubleMalloc_dist(total)))
            ABORT("Malloc fails for SymFact V2 redistributed aij[].");
        ja = ia + total;
    }

    if (procs > 1)
    {
        if (!(send_req = (MPI_Request *)
                  SUPERLU_MALLOC(2 * procs * sizeof(MPI_Request))))
            ABORT("Malloc fails for SymFact V2 send_req[].");
        if (!(ia_send = (int_t **) SUPERLU_MALLOC(procs * sizeof(int_t *))))
            ABORT("Malloc fails for SymFact V2 ia_send[].");
        if (!(aij_send = (double **) SUPERLU_MALLOC(procs * sizeof(double *))))
            ABORT("Malloc fails for SymFact V2 aij_send[].");
        for (int p = 0; p < procs; ++p)
        {
            ia_send[p] = NULL;
            aij_send[p] = NULL;
            send_req[p] = MPI_REQUEST_NULL;
            send_req[procs + p] = MPI_REQUEST_NULL;
        }
        if (SendCnt > 0)
        {
            if (!(index = intMalloc_dist(2 * SendCnt)))
                ABORT("Malloc fails for SymFact V2 send index[].");
            if (!(nzval = doubleMalloc_dist(SendCnt)))
                ABORT("Malloc fails for SymFact V2 send values[].");
        }
        if (!(ptr_to_send = intCalloc_dist(procs)))
            ABORT("Calloc fails for SymFact V2 ptr_to_send[].");
        if (maxnnzToRecv > 0)
        {
            if (!(itemp = intMalloc_dist(2 * maxnnzToRecv)))
                ABORT("Malloc fails for SymFact V2 recv index[].");
            if (!(dtemp = doubleMalloc_dist(maxnnzToRecv)))
                ABORT("Malloc fails for SymFact V2 recv values[].");
        }

        int_t ip = 0;
        int_t vp = 0;
        for (int p = 0; p < procs; ++p)
        {
            if (p != iam)
            {
                if (nnzToSend[p] > 0)
                    ia_send[p] = &index[ip];
                ip += 2 * nnzToSend[p];
                if (nnzToSend[p] > 0)
                    aij_send[p] = &nzval[vp];
                vp += nnzToSend[p];
            }
        }
    }

    nnz_loc = 0;
    for (int_t j = 0; j < n; ++j)
    {
        for (int_t p0 = (*colptr)[j]; p0 < (*colptr)[j + 1]; ++p0)
        {
            int_t irow = (*rowind)[p0];
            int_t jcol = j;
            int dest = dSymV2EntryOwner(trf3Dpart, supno, grid, &irow, &jcol);
            if (dest != iam)
            {
                int_t pos = ptr_to_send[dest]++;
                ia_send[dest][pos] = irow;
                ia_send[dest][pos + nnzToSend[dest]] = jcol;
                aij_send[dest][pos] = (*a)[p0];
            }
            else
            {
                ia[nnz_loc] = irow;
                ja[nnz_loc] = jcol;
                aij[nnz_loc] = (*a)[p0];
                ++nnz_loc;
            }
        }
    }

    if (procs > 1)
    {
        for (int p = 0; p < procs; ++p)
        {
            if (p != iam && nnzToSend[p] > 0)
            {
                MPI_Isend(ia_send[p], 2 * nnzToSend[p], mpi_int_t,
                          p, iam, grid->comm, &send_req[p]);
                MPI_Isend(aij_send[p], nnzToSend[p], MPI_DOUBLE,
                          p, iam + procs, grid->comm,
                          &send_req[procs + p]);
            }
        }

        for (int p = 0; p < procs; ++p)
        {
            if (p != iam && nnzToRecv[p] > 0)
            {
                MPI_Recv(itemp, 2 * nnzToRecv[p], mpi_int_t,
                         p, p, grid->comm, &status);
                MPI_Recv(dtemp, nnzToRecv[p], MPI_DOUBLE,
                         p, p + procs, grid->comm, &status);
                for (int_t i = 0; i < nnzToRecv[p]; ++i)
                {
                    ia[nnz_loc] = itemp[i];
                    ja[nnz_loc] = itemp[i + nnzToRecv[p]];
                    aij[nnz_loc] = dtemp[i];
                    ++nnz_loc;
                }
            }
        }

        for (int p = 0; p < procs; ++p)
        {
            if (p != iam && nnzToSend[p] > 0)
            {
                MPI_Wait(&send_req[p], &status);
                MPI_Wait(&send_req[procs + p], &status);
            }
        }
    }

    SUPERLU_FREE(*colptr);
    if (nnz_old > 0)
    {
        SUPERLU_FREE(*rowind);
        SUPERLU_FREE(*a);
    }

    if (!(*colptr = intCalloc_dist(n + 1)))
        ABORT("Calloc fails for SymFact V2 redistributed colptr[].");
    for (int_t i = 0; i < nnz_loc; ++i)
        ++(*colptr)[ja[i]];

    int_t running = 0;
    for (int_t j = 0; j < n; ++j)
    {
        int_t count = (*colptr)[j];
        (*colptr)[j] = running;
        running += count;
    }
    (*colptr)[n] = running;

    if (nnz_loc > 0)
    {
        if (!(*rowind = intMalloc_dist(nnz_loc)))
            ABORT("Malloc fails for SymFact V2 redistributed rowind[].");
        if (!(*a = doubleMalloc_dist(nnz_loc)))
            ABORT("Malloc fails for SymFact V2 redistributed values[].");
        for (int_t i = 0; i < nnz_loc; ++i)
        {
            int_t j = ja[i];
            int_t pos = (*colptr)[j]++;
            (*rowind)[pos] = ia[i];
            (*a)[pos] = aij[i];
        }
        for (int_t j = n; j > 0; --j)
            (*colptr)[j] = (*colptr)[j - 1];
        (*colptr)[0] = 0;
    }
    else
    {
        *rowind = NULL;
        *a = NULL;
    }

    SUPERLU_FREE(nnzToRecv);
    if (ia)
        SUPERLU_FREE(ia);
    if (aij)
        SUPERLU_FREE(aij);
    if (procs > 1)
    {
        SUPERLU_FREE(send_req);
        SUPERLU_FREE(ia_send);
        SUPERLU_FREE(aij_send);
        if (SendCnt > 0)
        {
            SUPERLU_FREE(index);
            SUPERLU_FREE(nzval);
        }
        SUPERLU_FREE(ptr_to_send);
        if (maxnnzToRecv > 0)
        {
            SUPERLU_FREE(itemp);
            SUPERLU_FREE(dtemp);
        }
    }

    return 0;
}

static float
dSymV2Distribute3d_LDL_impl(superlu_dist_options_t *options, int_t n,
        SuperMatrix *A, dScalePermstruct_t *ScalePermstruct,
        Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct,
        gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    dtrf3Dpartition_t *trf3Dpart = LUstruct->trf3Dpart;
    SupernodeToGridMap_t *superGridMap = trf3Dpart->superGridMap;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int iam = grid->iam;
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    int_t nsupers = supno[n - 1] + 1;
    int_t *lsub = Glu_freeable->lsub;
    int_t *xlsub = Glu_freeable->xlsub;
    int_t *xa = NULL;
    int_t *asub = NULL;
    double *a = NULL;
    int_t mybufmax[NBUFFERS];
    int_t iword = sizeof(int_t);
    int_t dword = sizeof(double);
    float mem_use = 0.0;
    float memTRS = 0.0;

    if (options->Fact == SamePattern_SameRowPerm)
        ABORT("SymFact GPU3DVERSION=2 LDL distribution does not support SamePattern_SameRowPerm yet.");
    if (!dSymV2CheckPartition(trf3Dpart))
        ABORT("SymFact GPU3DVERSION=2 LDL distribution metadata is not initialized.");

    for (int_t i = 0; i < NBUFFERS; ++i)
        mybufmax[i] = 0;

    int_t sym_panel_count = trf3Dpart->symV2LocalPanelCount;
    int_t sym_row_count = trf3Dpart->symV2LocalRowCount;
    int_t sym_panel_alloc = SUPERLU_MAX((int_t)1, sym_panel_count);
    int_t sym_row_alloc = SUPERLU_MAX((int_t)1, sym_row_count);

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Enter dSymV2Distribute3d_LDL_impl()");
#endif

    dSymV2DistributeTrace(grid3d, "V2 LDL before dReDistribute_A");
    dReDistribute_A(A, ScalePermstruct, Glu_freeable, xsup, supno,
                    grid, &xa, &asub, &a);
    dSymV2DistributeTrace(grid3d, "V2 LDL after dReDistribute_A");
    dSymV2RedistributeLowerToLDL(trf3Dpart, supno, n, grid, &xa, &asub, &a);
    dSymV2DistributeTrace(grid3d, "V2 LDL after lower-to-LDL redistribution");

    int *ToRecv = (int *)SUPERLU_MALLOC(nsupers * sizeof(int));
    if (!ToRecv)
        ABORT("Malloc fails for SymFact V2 ToRecv[].");
    for (int_t i = 0; i < nsupers; ++i)
        ToRecv[i] = 0;

    int **ToSendR = (int **)SUPERLU_MALLOC(sym_panel_alloc * sizeof(int *));
    if (!ToSendR)
        ABORT("Malloc fails for SymFact V2 ToSendR[].");
    int_t len = sym_panel_alloc * grid->npcol;
    int *index1 = int32Malloc_dist(len);
    if (!index1)
        ABORT("Malloc fails for SymFact V2 ToSendR[0].");
    for (int_t i = 0; i < len; ++i)
        index1[i] = SLU_EMPTY;
    for (int_t i = 0, j = 0; i < sym_panel_alloc; ++i, j += grid->npcol)
        ToSendR[i] = &index1[j];

    int *ToSendD = SUPERLU_MALLOC(sym_row_alloc * sizeof(int));
    if (!ToSendD)
        ABORT("Malloc fails for SymFact V2 ToSendD[].");
    for (int_t i = 0; i < sym_row_alloc; ++i)
        ToSendD[i] = NO;

    int_t *ilsum = intMalloc_dist(sym_row_alloc + 1);
    if (!ilsum)
        ABORT("Malloc fails for SymFact V2 ilsum[].");
    int_t ldaspa = 0;
    ilsum[0] = 0;
    for (int_t lb = 0; lb < sym_row_count; ++lb)
    {
        int_t gb = trf3Dpart->symV2LocalRowGids[lb];
        int_t nrow = SuperSize(gb);
        ldaspa += nrow;
        ilsum[lb + 1] = ilsum[lb] + nrow;
    }

    int_t *rb_marker = intCalloc_dist(sym_row_alloc);
    int_t *Lrb_length = intCalloc_dist(sym_row_alloc);
    int_t *Lrb_number = intMalloc_dist(sym_row_alloc);
    int_t *Lrb_indptr = intMalloc_dist(sym_row_alloc);
    int_t *Lrb_valptr = intMalloc_dist(sym_row_alloc);
    double *dense = doubleCalloc_dist(ldaspa * sp_ienv_dist(3, options));
    if (!rb_marker || !Lrb_length || !Lrb_number || !Lrb_indptr ||
        !Lrb_valptr || !dense)
        ABORT("Malloc fails for SymFact V2 L scratch.");

    int *fmod = int32Calloc_dist(sym_row_alloc);
    if (!fmod)
        ABORT("Calloc fails for SymFact V2 fmod[].");

    double **Lnzval_bc_ptr =
        (double **)SUPERLU_MALLOC(sym_panel_alloc * sizeof(double *));
    int_t **Lrowind_bc_ptr =
        (int_t **)SUPERLU_MALLOC(sym_panel_alloc * sizeof(int_t *));
    int_t **Lindval_loc_bc_ptr =
        (int_t **)SUPERLU_MALLOC(sym_panel_alloc * sizeof(int_t *));
    double **Linv_bc_ptr =
        (double **)SUPERLU_MALLOC(sym_panel_alloc * sizeof(double *));
    double **Uinv_bc_ptr =
        (double **)SUPERLU_MALLOC(sym_panel_alloc * sizeof(double *));
    if (!Lnzval_bc_ptr || !Lrowind_bc_ptr || !Lindval_loc_bc_ptr ||
        !Linv_bc_ptr || !Uinv_bc_ptr)
        ABORT("Malloc fails for SymFact V2 L/D pointer arrays.");
    for (int_t i = 0; i < sym_panel_alloc; ++i)
    {
        Lnzval_bc_ptr[i] = NULL;
        Lrowind_bc_ptr[i] = NULL;
        Lindval_loc_bc_ptr[i] = NULL;
        Linv_bc_ptr[i] = NULL;
        Uinv_bc_ptr[i] = NULL;
    }

    int **fsendx_plist =
        (int **)SUPERLU_MALLOC(sym_panel_alloc * sizeof(int *));
    if (!fsendx_plist)
        ABORT("Malloc fails for SymFact V2 fsendx_plist[].");
    len = sym_panel_alloc * grid->nprow;
    index1 = int32Malloc_dist(len);
    if (!index1)
        ABORT("Malloc fails for SymFact V2 fsendx_plist[0].");
    for (int_t i = 0; i < len; ++i)
        index1[i] = SLU_EMPTY;
    for (int_t i = 0, j = 0; i < sym_panel_alloc; ++i, j += grid->nprow)
        fsendx_plist[i] = &index1[j];

    int nfrecvx = 0;
    int nfsendx = 0;
    mem_use += sym_panel_alloc * sizeof(int_t *) +
               (sym_panel_alloc * grid->npcol + nsupers) * iword;
    mem_use += sym_row_alloc * iword;
    mem_use += 5.0 * sym_row_alloc * iword +
               ldaspa * sp_ienv_dist(3, options) * dword;
    mem_use += sym_panel_alloc * sizeof(double *) * 3.0 +
               sym_panel_alloc * sizeof(int_t *) * 2.0 +
               len * iword;
    memTRS += sym_panel_alloc * sizeof(int_t *) +
              2.0 * sym_panel_alloc * sizeof(double *);

    dSymV2DistributeTrace(grid3d, "V2 LDL before L/D structure propagation");

    for (int_t jb = 0; jb < nsupers; ++jb)
    {
        int pc = dSymV2PanelRoot(trf3Dpart, jb);
        if (mycol != pc)
            continue;

        int_t fsupc = FstBlockC(jb);
        int_t nsupc = SuperSize(jb);
        int_t ljb = dSymV2PanelLocalIndex(trf3Dpart, jb);
        if (ljb < 0)
            continue;

        for (int_t j = fsupc, dense_col = 0; j < FstBlockC(jb + 1);
             ++j, dense_col += ldaspa)
        {
            for (int_t p = xa[j]; p < xa[j + 1]; ++p)
            {
                int_t irow = asub[p];
                int_t gb = BlockNum(irow);
                if (gb < jb)
                    continue;
                int pr = dSymV2DiagRoot(trf3Dpart, gb);
                if (myrow != pr)
                    continue;
                int_t lb = dSymV2RowLocalIndex(trf3Dpart, gb);
                if (lb < 0)
                    continue;
                int_t spa_row = ilsum[lb] + irow - FstBlockC(gb);
                dense[dense_col + spa_row] = a[p];
            }
        }

        int jbrow = dSymV2DiagRoot(trf3Dpart, jb);
        int_t nrbl = 0;
        len = 0;
        int kseen = 0;
        int_t istart = xlsub[fsupc];
        for (int_t p = istart; p < xlsub[fsupc + 1]; ++p)
        {
            int_t irow = lsub[p];
            int_t gb = BlockNum(irow);
            int pr = dSymV2DiagRoot(trf3Dpart, gb);
            if (pr != jbrow && myrow == jbrow &&
                fsendx_plist[ljb][pr] == SLU_EMPTY)
            {
                fsendx_plist[ljb][pr] = YES;
                ++nfsendx;
            }
            if (myrow == pr)
            {
                int_t lb = dSymV2RowLocalIndex(trf3Dpart, gb);
                if (lb < 0)
                    continue;
                if (rb_marker[lb] <= jb)
                {
                    rb_marker[lb] = jb + 1;
                    Lrb_length[lb] = 1;
                    Lrb_number[nrbl++] = gb;
                    if (gb != jb)
                        ++fmod[lb];
                    if (kseen == 0 && myrow != jbrow)
                    {
                        ++nfrecvx;
                        kseen = 1;
                    }
                }
                else
                {
                    ++Lrb_length[lb];
                }
                ++len;
            }
        }

        if (nrbl > 0 && superGridMap[jb] != NOT_IN_GRID)
        {
            int_t len1 = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
            int_t *index = intMalloc_dist(len1);
            double *lusup =
                (double *)SUPERLU_MALLOC(len * nsupc * sizeof(double));
            if (!index || !lusup)
                ABORT("Malloc fails for SymFact V2 L panel.");
            Lindval_loc_bc_ptr[ljb] = intCalloc_dist(nrbl * 3);
            if (!Lindval_loc_bc_ptr[ljb])
                ABORT("Malloc fails for SymFact V2 L local map.");

            int krow = dSymV2DiagRoot(trf3Dpart, jb);
            if (myrow == krow)
            {
                Linv_bc_ptr[ljb] =
                    (double *)SUPERLU_MALLOC(nsupc * nsupc * sizeof(double));
                Uinv_bc_ptr[ljb] =
                    (double *)SUPERLU_MALLOC(nsupc * nsupc * sizeof(double));
                if (!Linv_bc_ptr[ljb] || !Uinv_bc_ptr[ljb])
                    ABORT("Malloc fails for SymFact V2 diagonal inverse blocks.");
            }

            mybufmax[0] = SUPERLU_MAX(mybufmax[0], len1);
            mybufmax[1] = SUPERLU_MAX(mybufmax[1], len * nsupc);
            mybufmax[4] = SUPERLU_MAX(mybufmax[4], len);
            mem_use += len * nsupc * dword + len1 * iword;
            memTRS += nrbl * 3.0 * iword;
            if (myrow == krow)
                memTRS += 2.0 * nsupc * nsupc * dword;

            index[0] = nrbl;
            index[1] = len;
            int_t next_lind = BC_HEADER;
            int_t next_lval = 0;
            for (int_t ib = 0; ib < nrbl; ++ib)
            {
                int_t gb = Lrb_number[ib];
                int_t lb = dSymV2RowLocalIndex(trf3Dpart, gb);
                if (lb < 0)
                    ABORT("SymFact V2 LDL distribution missing local row index.");
                int_t block_len = Lrb_length[lb];
                Lindval_loc_bc_ptr[ljb][ib] = lb;
                Lindval_loc_bc_ptr[ljb][ib + nrbl] = next_lind;
                Lindval_loc_bc_ptr[ljb][ib + nrbl * 2] = next_lval;
                Lrb_length[lb] = 0;
                index[next_lind++] = gb;
                index[next_lind++] = block_len;
                Lrb_indptr[lb] = next_lind;
                Lrb_valptr[lb] = next_lval;
                next_lind += block_len;
                next_lval += block_len;
            }

            for (int_t p = istart; p < xlsub[fsupc + 1]; ++p)
            {
                int_t irow = lsub[p];
                int_t gb = BlockNum(irow);
                int pr = dSymV2DiagRoot(trf3Dpart, gb);
                if (myrow != pr)
                    continue;
                int_t lb = dSymV2RowLocalIndex(trf3Dpart, gb);
                if (lb < 0)
                    continue;
                int_t ipos = Lrb_indptr[lb]++;
                index[ipos] = irow;
                int_t vpos = Lrb_valptr[lb]++;
                int_t spa_row = ilsum[lb] + irow - FstBlockC(gb);
                for (int_t j = 0, dense_col = 0; j < nsupc;
                     ++j, dense_col += ldaspa)
                {
                    lusup[vpos] = dense[dense_col + spa_row];
                    dense[dense_col + spa_row] = 0.0;
                    vpos += len;
                }
            }

            if (nrbl > 1)
            {
                if (myrow == krow)
                {
                    dSymV2MoveDiagBlockFirst(index,
                                             Lindval_loc_bc_ptr[ljb],
                                             nrbl, jb);
                    if (nrbl > 2)
                        quickSortM(&Lindval_loc_bc_ptr[ljb][1], 0,
                                   nrbl - 2, nrbl, 0, 3);
                }
                else
                {
                    quickSortM(Lindval_loc_bc_ptr[ljb], 0,
                               nrbl - 1, nrbl, 0, 3);
                }
            }

            int_t *index_srt = intMalloc_dist(len1);
            double *lusup_srt =
                (double *)SUPERLU_MALLOC(len * nsupc * sizeof(double));
            if (!index_srt || !lusup_srt)
                ABORT("Malloc fails for SymFact V2 sorted L panel.");

            int_t idx_indx = BC_HEADER;
            int_t idx_lusup = 0;
            for (int_t jj = 0; jj < BC_HEADER; ++jj)
                index_srt[jj] = index[jj];
            for (int_t ib = 0; ib < nrbl; ++ib)
            {
                int_t loc = Lindval_loc_bc_ptr[ljb][ib + nrbl];
                int_t nbrow = index[loc + 1];
                for (int_t jj = 0; jj < LB_DESCRIPTOR + nbrow; ++jj)
                    index_srt[idx_indx++] = index[loc + jj];
                Lindval_loc_bc_ptr[ljb][ib + nrbl] =
                    idx_indx - LB_DESCRIPTOR - nbrow;

                for (int_t jj = 0; jj < nbrow; ++jj)
                {
                    int_t dst = idx_lusup;
                    int_t src =
                        Lindval_loc_bc_ptr[ljb][ib + nrbl * 2] + jj;
                    for (int_t col = 0; col < nsupc; ++col)
                    {
                        lusup_srt[dst] = lusup[src];
                        dst += len;
                        src += len;
                    }
                    ++idx_lusup;
                }
                Lindval_loc_bc_ptr[ljb][ib + nrbl * 2] =
                    idx_lusup - nbrow;
            }

            SUPERLU_FREE(lusup);
            SUPERLU_FREE(index);
            if (superGridMap[jb] == IN_GRID_ZERO)
                memset(lusup_srt, 0, len * nsupc * sizeof(double));
            Lrowind_bc_ptr[ljb] = index_srt;
            Lnzval_bc_ptr[ljb] = lusup_srt;
        }
        else
        {
            for (int_t p = istart; p < xlsub[fsupc + 1]; ++p)
            {
                int_t irow = lsub[p];
                int_t gb = BlockNum(irow);
                int pr = dSymV2DiagRoot(trf3Dpart, gb);
                if (myrow != pr)
                    continue;
                int_t lb = dSymV2RowLocalIndex(trf3Dpart, gb);
                if (lb < 0)
                    continue;
                int_t spa_row = ilsum[lb] + irow - FstBlockC(gb);
                for (int_t j = 0, dense_col = 0; j < nsupc;
                     ++j, dense_col += ldaspa)
                    dense[dense_col + spa_row] = 0.0;
            }
            Lrowind_bc_ptr[ljb] = NULL;
            Lnzval_bc_ptr[ljb] = NULL;
            Lindval_loc_bc_ptr[ljb] = NULL;
            Linv_bc_ptr[ljb] = NULL;
            Uinv_bc_ptr[ljb] = NULL;
        }
    }

    Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
    Llu->Lindval_loc_bc_ptr = Lindval_loc_bc_ptr;
    Llu->Lnzval_bc_ptr = Lnzval_bc_ptr;
    Llu->Ufstnz_br_ptr = NULL;
    Llu->Unzval_br_ptr = NULL;
    Llu->Unnz = NULL;
    Llu->ToRecv = ToRecv;
    Llu->ToSendD = ToSendD;
    Llu->ToSendR = ToSendR;
    Llu->fmod = fmod;
    Llu->fsendx_plist = fsendx_plist;
    Llu->nfrecvx = nfrecvx;
    Llu->nfsendx = nfsendx;
    Llu->bmod = NULL;
    Llu->bsendx_plist = NULL;
    Llu->nbrecvx = 0;
    Llu->nbsendx = 0;
    Llu->ilsum = ilsum;
    Llu->ldalsum = ldaspa;
    Llu->Linv_bc_ptr = Linv_bc_ptr;
    Llu->Uinv_bc_ptr = Uinv_bc_ptr;
    Llu->Urbs = NULL;
    Llu->Ucb_indptr = NULL;
    Llu->Ucb_valptr = NULL;
    Llu->Send_CommL = NULL;
    Llu->Recv_CommL = NULL;

    SUPERLU_FREE(rb_marker);
    SUPERLU_FREE(Lrb_length);
    SUPERLU_FREE(Lrb_number);
    SUPERLU_FREE(Lrb_indptr);
    SUPERLU_FREE(Lrb_valptr);
    SUPERLU_FREE(dense);
    mem_use -= 5.0 * sym_row_alloc * iword +
               ldaspa * sp_ienv_dist(3, options) * dword;

    dSymV2DistributeTrace(grid3d, "V2 LDL before bufmax allreduce");
    MPI_Allreduce(mybufmax, Llu->bufmax, NBUFFERS, mpi_int_t,
                  MPI_MAX, grid->comm);
    dSymV2DistributeTrace(grid3d, "V2 LDL after bufmax allreduce");

    Llu->mod_bit = int32Malloc_dist(sym_row_alloc);
    if (!Llu->mod_bit)
        ABORT("Malloc fails for SymFact V2 mod_bit[].");

    if (xa != NULL && xa[A->ncol] > 0)
    {
        SUPERLU_FREE(asub);
        SUPERLU_FREE(a);
    }
    if (xa != NULL)
        SUPERLU_FREE(xa);

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Exit dSymV2Distribute3d_LDL_impl()");
#endif

    return mem_use + memTRS;
}

float
dSymV2Distribute3d(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
        dScalePermstruct_t *ScalePermstruct,
        Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct,
        gridinfo3d_t *grid3d)
{
    if (options == NULL || options->SymFact != YES)
        ABORT("dSymV2Distribute3d requires SymFact=YES.");

    return dSymV2Distribute3d_LDL_impl(options, n, A, ScalePermstruct,
                                       Glu_freeable, LUstruct, grid3d);
}
