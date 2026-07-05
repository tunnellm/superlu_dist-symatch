#pragma once 
#include <algorithm>
#include <iostream>
#include <cassert>
#include "superlu_defs.h"
#include "luAuxStructTemplated.hpp"
#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#include "xlupanels_GPU.cuh"
#endif
#include "lupanels.hpp"  //unneeded??
#include "xlupanels.hpp"
#include "symldl_v2_lupanels_impl.hpp"
#include "xlupanels_diag_buffers_impl.hpp"
#include "xlupanels_gpu_panel_access_impl.hpp"
#include "xlupanels_constructor_impl.hpp"
#include "superlu_blas.hpp"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurComplementUpdate(
    int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

#pragma omp parallel for
    for (size_t ij = 0; ij < (nlb - st_lb) * nub; ij++)
    {
        int_t ii = ij / nub + st_lb;
        int_t jj = ij % nub;
        blockUpdate(k, ii, jj, lpanel, upanel);
    }

    return 0;
}

// should be called from an openMP region
template <typename Ftype>
int_t *xLUstruct_t<Ftype>::computeIndirectMap(indirectMapType direction, int_t srcLen, int_t *srcVec,
                                         int_t dstLen, int_t *dstVec)
{
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        return srcVec;
    }
    int_t thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif
    int_t *dstIdx = indirect + thread_id * ldt;
    for (int_t i = 0; i < dstLen; i++)
    {
        // if(thread_id < dstLen)
        dstIdx[dstVec[i]] = i;
    }

    int_t *RCmap = (direction == ROW_MAP) ? indirectRow : indirectCol;
    RCmap += thread_id * ldt;

    for (int_t i = 0; i < srcLen; i++)
    {
        RCmap[i] = dstIdx[srcVec[i]];
    }

    return RCmap;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dScatter(int_t m, int_t n,
                              int_t gi, int_t gj,
                              Ftype *Src, int_t ldsrc,
                              int_t *srcRowList, int_t *srcColList)
{

    Ftype *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    if (gj > gi) // its in upanel
    {
        int li = g2lRow(gi);
        int lj = uPanelVec[li].find(gj);
        Dst = uPanelVec[li].blkPtr(lj);
        lddst = supersize(gi);
        dstRowLen = supersize(gi);
        dstRowList = NULL;
        dstColLen = uPanelVec[li].nbcol(lj);
        dstColList = uPanelVec[li].colList(lj);
        // std::cout<<li<<" "<<lj<<" Dst[0] is"<<Dst[0] << "\n";
    }
    else
    {
        int lj = g2lCol(gj);
        int li = lPanelVec[lj].find(gi);
        Dst = lPanelVec[lj].blkPtr(li);
        lddst = lPanelVec[lj].LDA();
        dstRowLen = lPanelVec[lj].nbrow(li);
        dstRowList = lPanelVec[lj].rowList(li);
        dstColLen = supersize(gj);
        dstColList = NULL;
    }

    // compute source row to dest row mapping
    int_t *rowS2D = computeIndirectMap(ROW_MAP, m, srcRowList,
                                       dstRowLen, dstRowList);

    // compute source col to dest col mapping
    int_t *colS2D = computeIndirectMap(COL_MAP, n, srcColList,
                                       dstColLen, dstColList);

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::packedU2skyline(LUStruct_type<Ftype> *LUstruct)
{

    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    Ftype **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            int_t globalId = i * Pr + myrow;
            uPanelVec[i].packed2skyline(globalId, Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
        }
    }

    return 0;
}

int numProcsPerNode(MPI_Comm baseCommunicator);
// int numProcsPerNode(MPI_Comm baseCommunicator)
// {
//     MPI_Comm sharedComm;
//     MPI_Comm_split_type(baseCommunicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedComm);
//     int count = 0;
//     MPI_Comm_size(sharedComm, &count);
//     return count;
// }


template <typename Ftype>
int_t xLUstruct_t<Ftype>::lookAheadUpdate(
    int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t laILoc = lpanel.find(laIdx);
    int_t nub = upanel.nblocks();
    int_t laJLoc = upanel.find(laIdx);

#pragma omp parallel
    {
        /*Next lpanelUpdate*/
#pragma omp for nowait
        for (size_t ii = st_lb; ii < nlb; ii++)
        {
            int_t jj = laJLoc;
            if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
                blockUpdate(k, ii, jj, lpanel, upanel);
        }

        /*Next upanelUpdate*/
#pragma omp for nowait
        for (size_t jj = 0; jj < nub; jj++)
        {
            int_t ii = laILoc;
            if (laILoc != GLOBAL_BLOCK_NOT_FOUND && jj != laJLoc)
                blockUpdate(k, ii, jj, lpanel, upanel);
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::blockUpdate(int_t k,
                                 int_t ii, int_t jj, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    int thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif

    Ftype *V = bigV + thread_id * ldt * ldt;

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    superlu_gemm<Ftype>("N", "N",
                  lpanel.nbrow(ii), upanel.nbcol(jj), supersize(k), alpha,
                  lpanel.blkPtr(ii), lpanel.LDA(),
                  upanel.blkPtr(jj), upanel.LDA(), beta,
                  V, lpanel.nbrow(ii));

    // now do the scatter
    int_t ib = lpanel.gid(ii);
    int_t jb = upanel.gid(jj);

    dScatter(lpanel.nbrow(ii), upanel.nbcol(jj),
             ib, jb, V, lpanel.nbrow(ii),
             lpanel.rowList(ii), upanel.colList(jj));
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurCompUpdateExcludeOne(
    int_t k, int_t ex, // suypernodes to be excluded
    xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = upanel.find(ex);

#pragma omp parallel for
    for (size_t ij = 0; ij < (nlb - st_lb) * nub; ij++)
    {
        int_t ii = ij / nub + st_lb;
        int_t jj = ij % nub;

        if (ii != exILoc && jj != exJLoc)
            blockUpdate(k, ii, jj, lpanel, upanel);
    }
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dDiagFactorPanelSolve(int_t k, int_t offset, diagFactBufs_type<Ftype>**dFBufs)
{

    int_t ksupc = SuperSize(k);
    /*=======   Diagonal Factorization      ======*/
    if (iam == procIJ(k, k))
    {
        lPanelVec[g2lCol(k)].diagFactor(k, dFBufs[offset]->BlockUFactor, ksupc,
                                        thresh, xsup, options, stat, info);
        lPanelVec[g2lCol(k)].packDiagBlock(dFBufs[offset]->BlockLFactor, ksupc);
    }

    /*=======   Diagonal Broadcast          ======*/
    if (myrow == krow(k))
        MPI_Bcast((void *)dFBufs[offset]->BlockLFactor, ksupc * ksupc,
                  MPI_DOUBLE, kcol(k), (grid->rscp).comm);
    if (mycol == kcol(k))
        MPI_Bcast((void *)dFBufs[offset]->BlockUFactor, ksupc * ksupc,
                  MPI_DOUBLE, krow(k), (grid->cscp).comm);

    /*=======   Panel Update                ======*/
    if (myrow == krow(k))
        uPanelVec[g2lRow(k)].panelSolve(ksupc, dFBufs[offset]->BlockLFactor, ksupc);

    if (mycol == kcol(k))
        lPanelVec[g2lCol(k)].panelSolve(ksupc, dFBufs[offset]->BlockUFactor, ksupc);

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dPanelBcast(int_t k, int_t offset)
{
    /*=======   Panel Broadcast             ======*/
    xupanel_t<Ftype> k_upanel(UidxRecvBufs[offset], UvalRecvBufs[offset]);
    xlpanel_t<Ftype> k_lpanel(LidxRecvBufs[offset], LvalRecvBufs[offset]);
    if (myrow == krow(k))
        k_upanel = uPanelVec[g2lRow(k)];

    if (mycol == kcol(k))
        k_lpanel = lPanelVec[g2lCol(k)];

    if (UidxSendCounts[k] > 0)
    {
        MPI_Bcast(k_upanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
        MPI_Bcast(k_upanel.val, UvalSendCounts[k], MPI_DOUBLE, krow(k), grid3d->cscp.comm);
    }

    if (LidxSendCounts[k] > 0)
    {
        MPI_Bcast(k_lpanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
        MPI_Bcast(k_lpanel.val, LvalSendCounts[k], MPI_DOUBLE, kcol(k), grid3d->rscp.comm);
    }
    return 0;
}
