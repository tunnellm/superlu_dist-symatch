/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Literature-faithful NVSHMEM forward solve for native SymLDL.
 *
 * The WAIT/SOLVE kernels and two-stream launch structure are copied from
 * pdgstrs_lsum_cuda.cu. SymLDL-specific storage and ownership adapters are
 * kept outside the copied execution protocol.
 */

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "dsymldl_v2_literature_forward.h"

#ifndef BLK_M
#define BLK_M DIM_X*4
#define BLK_N DIM_Y*4
#define BLK_K 1024/(BLK_M)
#endif

#ifndef CACHELINE
#define CACHELINE 64
#endif

#ifndef RDMA_FLAG_SIZE
#define RDMA_FLAG_SIZE 2
#endif

#ifndef d_atomicAdd
#define d_atomicAdd atomicAdd
#endif

#ifdef HAVE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                        \
    do {                                                                        \
        cudaError_t result_ = (stmt);                                           \
        if (result_ != cudaSuccess) {                                           \
            fprintf(stderr, "[%s:%d] CUDA failed: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(result_));                               \
            ABORT("SymLDL literature forward CUDA operation failed.");        \
        }                                                                       \
    } while (0)
#endif


/******************************************************************************/
static __device__
void symldl_literature_gemm_device(
        int M, int N, int K,
        int blx, int bly,
        const double* __restrict__ A, int LDA,
        const double* __restrict__ B, int LDB,
        double rC[THR_N][THR_M],
        double alpha, double beta)
{
    // #if (__CUDA_ARCH__ >= 200)
    int idx = threadIdx_x;  // thread's m dimension
    int idy = threadIdx_y;  // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    // int blx = blockIdx_x;   // block's m dimension
    // int bly = blockIdx_y;   // block's n dimension

    __shared__ double sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ double sB[BLK_N][BLK_K+1];      // +1 always required

    // Registers for the innermost loop
    double rA[THR_M];
    double rB[THR_N];

    double ra[BLK_K/DIM_YA+1][BLK_M/DIM_XA];
    double rb[BLK_N/DIM_YB][BLK_K/DIM_XB+1];

    const double *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
    const double *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    int boundA = (LDA*(K-1) + M) - ( blx*BLK_M  + idyA*LDA + idxA ) -1;
    int boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;

    int m, n, k, kk;
    double zero = 0.0;

    // Zero C
#pragma unroll
    for (n = 0; n < THR_N; n++)
#pragma unroll
            for (m = 0; m < THR_M; m++)
                rC[n][m] = zero;

#pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YA)
#pragma unroll
            for (m = 0; m < BLK_M; m += DIM_XA)
                sA[n+idyA][m+idxA] = fetch(A, m, n, boundA);

#pragma unroll
    for (n = 0; n < BLK_N; n += DIM_YB)
#pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XB)
                sB[n+idyB][m+idxB] = fetch(B, m, n, boundB);


// #pragma unroll
//     for (n = 0; n < BLK_N; n += DIM_YB)
// #pragma unroll
//             for (m = 0; m < BLK_K; m += DIM_XB){
//                 int_t nn = min(n+idyB,N-1);
//                 int_t mm = min(m+idxB,K-1);
//                 // sB[n+idyB][m+idxB] = B[nn*LDB+mm+bly*BLK_N*LDB + idyB*LDB + idxB];
//                 sB[n+idyB][m+idxB] = B[bly*BLK_N*LDB + nn*LDB+mm];
//             }


    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K*LDA;
        boundA  -= BLK_K*LDA;

        offs_dB += BLK_K;
        boundB  -= BLK_K;

#pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
#pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

#pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
#pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);

// #pragma unroll
//         for (n = 0; n < BLK_N/DIM_YB; n++)
// #pragma unroll
//                 for (m = 0; m < BLK_K/DIM_XB; m++){
//                     int_t nn = min(n*DIM_YB+idyB,N-1);
//                     int_t mm = min(m*DIM_XB+idxB+kk+BLK_K,K-1);
//                     rb[n][m] = B[nn*LDB+mm+bly*BLK_N*LDB];
//                 }



        // Multiply
#pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
#pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
#pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
#pragma unroll
            for (n = 0; n < THR_N; n++) {
#pragma unroll
                for (m = 0; m < THR_M; m++) {
                    fma(rA[m], rB[n], rC[n][m]);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
#pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];

#pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
#pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];

        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
#pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
#pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
#pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
#pragma unroll
        for (n = 0; n < THR_N; n++) {
#pragma unroll
            for (m = 0; m < THR_M; m++) {
                fma(rA[m], rB[n], rC[n][m]);

            }
        }
    }

    // Store C regs->dev
    // if( beta == make_FloatingPoint_t(0.0,0.0) ) {
    // #pragma unroll
    // for (n = 0; n < THR_N; n++) {
    // int_t coord_dCn = bly*BLK_N + n*DIM_Y + idy;
    // #pragma unroll
    // for (m = 0; m < THR_M; m++) {
    // int_t coord_dCm = blx*BLK_M + m*DIM_X + idx;
    // if (coord_dCm < M && coord_dCn < N) {
    // int_t offsC = coord_dCn*LDC + coord_dCm;

    // double &regC = rC[n][m];
    // double &memC = C[offsC];

    // // memC = mul(alpha, regC);
    // }
    // }
    // }
    // } else {
    // #pragma unroll
    // for (n = 0; n < THR_N; n++) {
    // int_t coord_dCn = bly*BLK_N + n*DIM_Y + idy;
    // #pragma unroll
    // for (m = 0; m < THR_M; m++) {
    // int_t coord_dCm = blx*BLK_M + m*DIM_X + idx;
    // if (coord_dCm < M && coord_dCn < N) {
    // int_t offsC = coord_dCn*LDC + coord_dCm;

    // double &regC = rC[n][m];
    // double &memC = C[offsC];

    // // memC = add(mul(alpha, regC), mul(beta, memC));
    // }
    // }
    // }
    // }
    // #endif /* (__CUDA_ARCH__ >= 200) */
}



/******************************************************************************/
__device__ void symldl_literature_bcast_forward_device(C_Tree* tree,  volatile uint64_t* flag_bc_q,  int* my_flag_bc, int mype, int tid,double* dready_x, int maxrecvsz){
#ifdef HAVE_NVSHMEM
//int BCsendoffset;
    uint64_t sig = 1;
    int data_ofset=my_flag_bc[0]*maxrecvsz;
    for( int idxRecv = 0; idxRecv < tree->destCnt_; ++idxRecv ) {
        int iProc = tree->myDests_[idxRecv];
        nvshmemx_double_put_signal_nbi_block((double*)&dready_x[data_ofset], (double*)&dready_x[data_ofset], my_flag_bc[1],(uint64_t*)(flag_bc_q + my_flag_bc[0]), sig, NVSHMEM_SIGNAL_SET,iProc);
    }
#endif
}
__device__ void symldl_literature_reduce_forward_device(C_Tree* Tree, volatile uint64_t* flag_rd_q, int* my_flag_rd, int mype, int bid, int tid, double* dready_lsum, int maxrecvsz, int myroot){
    #ifdef HAVE_NVSHMEM
        int data_ofset,sig_ofset;
        uint64_t sig = 1;
        if (Tree->myIdx % 2 == 0) {
            sig_ofset = my_flag_rd[0] * 2;
            data_ofset = my_flag_rd[0] * maxrecvsz * 2;
        } else {
            sig_ofset = my_flag_rd[0] * 2 + 1;
            data_ofset = my_flag_rd[0] * maxrecvsz * 2 + maxrecvsz;
        }
        nvshmem_double_put_signal_nbi((double*)&dready_lsum[data_ofset],(double*)&dready_lsum[my_flag_rd[0]*maxrecvsz*2],my_flag_rd[1],(uint64_t*)flag_rd_q+sig_ofset, sig, NVSHMEM_SIGNAL_SET, myroot);


    // ////forward to my root if I have received everything
    // //double sum = 0;

    // //for (int i = my_flag_rd[0]*maxrecvsz*2; i < my_flag_rd[0]*maxrecvsz*2 + my_flag_rd[1]; i++) {
    // //    //printf("(%d), data, %d\n",mype,i);
    // //    printf("(%d), data, %d,%lf\n", mype, i, dready_lsum[i]);
    // //    sum += dready_lsum[i];
    // //}

    // //printf("---- Start RD Thread (%d,%d,%d), forwardMessage, forwardDevice, send to %d, "
    // //       "lib=%d,size=%d,  dataoffset=%d,maxrecvsz=%d\n",
    // //       mype, bid,tid, myroot,
    // //       my_flag_rd[0], my_flag_rd[1], data_ofset,maxrecvsz);
    //     nvshmem_double_put_nbi(&dready_lsum[data_ofset],&dready_lsum[my_flag_rd[0]*maxrecvsz*2],my_flag_rd[1],myroot);
    // //printf("---- END RD Thread (%d,%d,%d), forwardMessage, forwardDevice, send to %d, "
    // //       "lib=%d,size=%d,  dataoffset=%d,maxrecvsz=%d\n",
    // //       mype, bid,tid, myroot,
    // //       my_flag_rd[0], my_flag_rd[1], data_ofset,maxrecvsz);
    //     nvshmem_fence();
    //     int sig=1;
    //     nvshmemx_signal_op((uint64_t*)flag_rd_q+sig_ofset, sig, NVSHMEM_SIGNAL_SET, myroot);
    // //printf("Tsend:%d,%d,%d,%d,%d\n",
    // //       mype, my_flag_rd[0],data_ofset, sig_ofset,my_flag_rd[1]);
    #endif
}
__global__ void symldl_literature_wait_bcrd
        (
                int nrhs,
                C_Tree  *LRtree_ptr,
                int_t maxrecvsz,
                int mype,
                uint64_t* flag_bc_q,
                uint64_t* flag_rd_q,
                double* dready_x,
                double* dready_lsum,
                int* my_flag_bc,
                int* my_flag_rd,
                int* d_nfrecv,
                int* d_status,
                int* d_colnum,
                int* d_mynum,
                int* d_mymaskstart,
                int* d_mymasklength,
                int* d_nfrecvmod,
                int* d_statusmod,
                int* d_colnummod,
                int* d_mynummod,
                int* d_mymaskstartmod,
                int* d_mymasklengthmod,
                int* d_recv_cnt,
                int* d_msgnum,
                int* d_flag_mod,
                double *lsum,    /* Sum of local modifications.                        */
                int *fmod,     /* Modification count for L-solve.                    */
                gridinfo_t *grid,
                int_t *xsup,
                int_t *ilsum,
                int_t *row_gids,
                int nbrow_loc,
                int_t  nsupers
        ) {
#ifdef HAVE_NVSHMEM
    int bid = blockIdx.x;
//int global_id= blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int WAIT_NUM_THREADS = d_nfrecv[1]; //*d_nfrecv[2];
//if (tid==0) printf("(%d) WAIT_NUM_THREADS=%d,tot_wait_col=%d\n",mype,WAIT_NUM_THREADS,d_nfrecv[0]);
#ifdef _USE_SUMMIT
     if (bid < 4) { // for BC recv
	tid=bid*WAIT_NUM_THREADS+tid;
	WAIT_NUM_THREADS=WAIT_NUM_THREADS*4;
#else
     if (bid == 0) { // for BC recv
#endif
   //if (tid < WAIT_NUM_THREADS) { // for BC recv
   //if (tid==0) printf("(%d) WAIT_NUM_THREADS=%d,bc_tot_wait_col=%d\n",mype,WAIT_NUM_THREADS,d_nfrecv[0]);
       if (WAIT_NUM_THREADS >= d_nfrecv[0]) {
           if (tid < d_nfrecv[0]) {
               nvshmem_signal_wait_until((uint64_t *) (flag_bc_q + d_colnum[tid]), NVSHMEM_CMP_EQ, 1);
               d_status[d_colnum[tid]] = 1;
               //printf("WAIT1 (%d,%d) msg arrived in col %d\n", mype, tid, d_colnum[tid]);
           }
       } else {
           int delta = d_nfrecv[0] % WAIT_NUM_THREADS;
           if (tid < delta) {
               d_mynum[tid] = d_nfrecv[0] / WAIT_NUM_THREADS + 1;
           } else {
               d_mynum[tid] = d_nfrecv[0] / WAIT_NUM_THREADS;
           }
           __syncthreads();
           d_mymaskstart[tid] = 0;
           for (int i = 0; i < tid; i++) {
               d_mymaskstart[tid] += d_mynum[i];
           }
           d_mymasklength[tid] = d_colnum[d_mymaskstart[tid] + d_mynum[tid] - 1] - d_colnum[d_mymaskstart[tid]] + 1;
           __syncthreads();
           //printf("WAIT2 (%d,%d) mynum=%d, start=%d,%d length=%d\n",mype,tid,d_mynum[tid],d_mymaskstart[tid],d_colnum[d_mymaskstart[tid]],d_mymasklength[tid]);

           for (int i = 0; i < d_mynum[tid]; i++) {
               int wm_val = nvshmem_uint64_wait_until_any(flag_bc_q + d_colnum[d_mymaskstart[tid]], d_mymasklength[tid],
                                                       d_status + d_colnum[d_mymaskstart[tid]], NVSHMEM_CMP_EQ, 1);
               d_status[d_colnum[d_mymaskstart[tid]] + wm_val] = 1;
               //printf("WAIT2 (%d,%d) msg arrived in col %d, i=%d\n",mype,tid,d_colnum[d_mymaskstart[tid]] + wm_val, i);
           }
       }
   }
//if (tid==0) printf("(%d,%d,%d) WAIT EXIT\n",mype,bid,tid);
#ifdef _USE_SUMMIT
    if (bid >=4) { // for RD recv
        tid=(bid-4)*WAIT_NUM_THREADS+tid;
        WAIT_NUM_THREADS=WAIT_NUM_THREADS*4;
#else
    if (bid == 1) { // for RD recv
#endif
        //if (tid==0) printf("RD---(%d) WAIT_NUM_THREADS=%d,tot_wait_col=%d\n",mype,WAIT_NUM_THREADS,d_nfrecvmod[1]);
       int j, lib, k, knsupc, il, cnt;
       int fmod_tmp, aln_i;

       aln_i = 1;
    //   double temp;
       if (WAIT_NUM_THREADS >= d_nfrecvmod[1]) { // one thread wait for one col
           if (tid < d_nfrecvmod[1]) {
               //printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]]);
               for (int i = 0; i < d_recv_cnt[d_colnummod[tid]]; i++) {
                   //printf("(%d,%d,%d) I'm waiting for %d/%d msg in row %d.wait_off=%d,%d,status=%d,%d\n",mype, bid,tid, i,d_recv_cnt[d_colnummod[tid]],d_colnummod[tid],d_colnummod[tid]*2, d_colnummod[tid]*2+1,d_statusmod[d_colnummod[tid]*2], d_statusmod[d_colnummod[tid]*2+1]);
           //printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d,i=%d,wait_off=%d,%d,status=%d,%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]],i,d_colnummod[tid]*2, d_colnummod[tid]*2+1,d_statusmod[d_colnummod[tid]*2], d_statusmod[d_colnummod[tid]*2+1]);
                   int wm_val = nvshmem_uint64_wait_until_any(flag_rd_q + d_colnummod[tid] * 2, 2,
                                                           d_statusmod + d_colnummod[tid] * 2, NVSHMEM_CMP_EQ, 1);
                   d_statusmod[d_colnummod[tid] * 2 + wm_val] = 1;
                   lib = (d_colnummod[tid] * 2 + wm_val) / 2;
                   k = row_gids[lib];
                   knsupc = SuperSize(k);
                   il = LSUM_BLK(lib);
                   cnt = LRtree_ptr[lib].destCnt_;
                   //printf("recv1,%d,%d,%d,%d\n",
                   //       mype,d_colnummod[d_mymaskstartmod[tid]]*2,wm_val,lib);
                   //printf("(%d,%d,%d),idx=%d,lib=%d,cnt=%d,i=%d/%d\n", mype, bid, tid,
                   //       d_colnummod[tid] * 2 + wm_val, lib, cnt,i,d_recv_cnt[d_colnummod[tid]]);
                   if (d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1] == cnt) {
                       //double tmp_sum = 0;
                       int ii = 0;
                       if (cnt == 2) {
                           for (ii = 0; ii < cnt; ++ii) {
                               //tmp_sum = 0;
                               RHS_ITERATE(j) {
                                   for (int aab = 0; aab < knsupc; ++aab) {
                                       //temp=d_atomicAdd(&lsum[il+i + j*knsupc], dready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                       d_atomicAdd(&lsum[il + aab + j * knsupc],
                                                        dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab +
                                                                   j * knsupc]);
                                       //tmp_sum += dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                       //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,sum=%lf,dready_lsum[%d]=%f\n", mype, bid, tid,
                                       //       lib, k, ii, tmp_sum,
                                       //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                       //       dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                                   }

                                   // atomic return old val
                                   fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                                   //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d, tmp_sum=%lf\n", mype, bid, tid, lib, k,
                                   //       tmp_sum,fmod_tmp, tmp_sum);
                                   //printf("sum2-(%d,%d,%d),lib=%d,k=%d,fmod_tmp=%d\n", mype, bid, tid, lib, k,fmod_tmp);
                               }
                           }
                       }
                       if (cnt == 1) {
                           if (flag_rd_q[lib * 2 + 1] == 1) ii = 1;
                           //tmp_sum = 0;
                           RHS_ITERATE(j) {
                               for (int aab = 0; aab < knsupc; ++aab) {
                                   //temp=d_atomicAdd(&lsum[il+i + j*knsupc], dready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                   d_atomicAdd(&lsum[il + aab + j * knsupc],
                                                    dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                                   //tmp_sum += dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                   //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,sum=%lf,dready_lsum[%d]=%lf\n", mype, bid, tid, lib, k, ii,
                                   //       tmp_sum,maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc,
                                   //       dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                               }

                           }
                           // atomic return old val
                           fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                           //printf("sum1-(%d,%d,%d),lib=%d,k=%d,fmod_tmp=%d\n", mype, bid, tid, lib, k,fmod_tmp);
                           //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,fmod_tmp);
                       }

                       if (fmod_tmp == 1) {// forward RD
                           //senddone[lk]=1;
                           if (LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_) {
                               //cnt=LRtree_ptr[lib].msgSize_;
                               my_flag_rd[lib * RDMA_FLAG_SIZE] = lib;
                               my_flag_rd[lib * RDMA_FLAG_SIZE + 1] = LRtree_ptr[lib].msgSize_;
                               //double tmp_sum=0;
                               RHS_ITERATE(j) {
                                   for (int aab = 0; aab < knsupc; aab++) {
                                       dready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc] = lsum[il + aab + j * knsupc];
                                       //tmp_sum += dready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc];
                                       //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,dready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
                                       //       k * maxrecvsz * 2 + i +j * knsupc,
                                       //       dready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

                                   }
                               }
                               //printf("(%d,%d,%d),in wait lib=%d,k=%d,myflagrd=%d,%d\n", mype, bid, tid, lib, k,
                               //       my_flag_rd[k * RDMA_FLAG_SIZE], my_flag_rd[k * RDMA_FLAG_SIZE + 1]);
                               //int temp_mysendcout=atomicAdd(&d_flag_mod[0], 1);
                               //int temp_flag_mod=atomicExch(&d_flag_mod[temp_mysendcout+1],lib);
                               //printf("iam=%d in wait,lib=%d,%d,%d, pos=%d, temp %d,%d\n",mype,lib,k, d_flag_mod[temp_mysendcout+1], temp_mysendcout+1, temp_mysendcout,temp_flag_mod);
                               //printf("iam=%d in wait,lib=%d,%d,%d, pos=%d, temp %d,%d, sum=%lf\n",mype,lib,k, d_flag_mod[temp_mysendcout+1], temp_mysendcout+1, temp_mysendcout,temp_flag_mod, tmp_sum);
                               symldl_literature_reduce_forward_device(&LRtree_ptr[lib], flag_rd_q,
                                                                    &my_flag_rd[RDMA_FLAG_SIZE * lib], mype, bid, tid,
                                                                    &dready_lsum[0], maxrecvsz,LRtree_ptr[lib].myRoot_ );
                           }
                       }
                   }
               }//for
           }
       } else {
           int delta = d_nfrecvmod[1] % WAIT_NUM_THREADS;
           //int mynum = d_nfrecvmod[1] / WAIT_NUM_THREADS;
           //int mystart = tid*(mynum+1);
           //if (tid < delta) {
           //    mynum = mynum + 1;
           //    //d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS+1;
           //}else{
           //    mystart = (delta*(mynum+1))+(tid-delta)*mynum;
           //}
           //int mymasklength=(d_colnummod[mystart + mynum - 1] - d_colnummod[mystart]+1)*2;

           if (tid < delta){
               d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS + 1;
           }else {
               d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS;
           }
           __syncthreads();

           d_mymaskstartmod[tid] = 0;
           d_mymasklengthmod[tid] = 0;
           d_msgnum[tid] = 0;

           ////d_mymaskstartmod: start offset of d_colnummod
           for (int i = 0; i < tid; i++) {
               d_mymaskstartmod[tid] += d_mynummod[i];
               //printf("(%d,%d,%d),i=%d,d_mynummod=%d,d_mymaskstartmod=%d\n",
               //       mype,bid,tid,i,
               //       d_mynummod[i],d_mymaskstartmod[tid]);
           }
           __syncthreads();

           for (int i = d_mymaskstartmod[tid]; i < d_mymaskstartmod[tid] + d_mynummod[tid]; i++) {
               d_msgnum[tid] += d_recv_cnt[d_colnummod[i]];
               //printf("(%d,%d,%d),i=%d,d_recv_cnt=%d\n",mype,bid,tid,i,d_recv_cnt[d_colnummod[i]]);
           }
           d_mymasklengthmod[tid] = (d_colnummod[d_mymaskstartmod[tid] + d_mynummod[tid] - 1]
                                     - d_colnummod[d_mymaskstartmod[tid]]+1)*2;

           //printf("(%d,%d,%d) waitcol=%d,msgnum=%d,masklength=%d,start=%d\n",mype,bid,tid,
           //                   d_mynummod[tid],d_msgnum[tid],
           //                   d_mymasklengthmod[tid],d_mymaskstartmod[tid]);

           //printf("(%d,%d), mynum=%d,%d,mystart=%d,%d, mylength=%d,%d\n",mype,tid,d_mynummod[tid], mynum,d_mymaskstartmod[tid], mystart, d_mymasklengthmod[tid],mymasklength);
           for (int i = 0; i < d_msgnum[tid]; i++) {
               //printf("(%d,%d,%d)--before wait any,i=%d/%d\n",mype,bid,tid,i,d_msgnum[tid]);
               int wm_val = nvshmem_uint64_wait_until_any(&flag_rd_q[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                                       d_mymasklengthmod[tid],
                                                       &d_statusmod[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                                       NVSHMEM_CMP_EQ, 1);
               d_statusmod[d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val] = 1;
               lib = (d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val) / 2;
                   //printf("(%d,%d,%d),idx=%d,lib=%d,cnt=%d,i=%d/%d\n", mype, bid, tid,
                   //       d_colnummod[d_mymaskstartmod[tid]] * 2 + wm_val, lib, cnt,i,d_msgnum[tid]);
               //printf("recv,%d,%d,%d,%d,%d\n",
               //       mype,tid,d_colnummod[d_mymaskstartmod[tid]]*2,wm_val,lib);
               k = row_gids[lib];
               knsupc = SuperSize(k);
               il = LSUM_BLK(lib);
               cnt = LRtree_ptr[lib].destCnt_;
               //printf("HERE2-(%d,%d,%d),lib=%d,k=%d,wm_val=%d,cnt=%d,%d, mycnt=%d\n", mype, bid, tid, lib, k,
               //       wm_val,cnt,d_recv_cnt[lib],d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1]);

               if (d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1] == cnt) {
                   //double tmp_sum = 0;
                   int ii = 0;
                   if (cnt == 2) {
                       for (ii = 0; ii < cnt; ++ii) {
                           //tmp_sum = 0;
                           RHS_ITERATE(j) {
                               for (int aab = 0; aab < knsupc; aab++) {
                                   //temp=d_atomicAdd(&lsum[il+i + j*knsupc], dready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                   d_atomicAdd(&lsum[il + aab + j * knsupc],
                                                    dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab +
                                                               j * knsupc]);
                                   //tmp_sum += dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                   //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,dready_lsum[%d]=%f\n", mype, bid, tid,
                                   //       lib, k, ii,
                                   //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                   //       dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                               }

                               // atomic return old val
                               fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                               //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k,tmp_sum,fmod_tmp);
                           }
                       }
                   }
                   if (cnt == 1) {
                       if (flag_rd_q[lib * 2 + 1] == 1) ii = 1;
                       RHS_ITERATE(j) {
                           for (int aab = 0; aab < knsupc; ++aab) {
                               d_atomicAdd(&lsum[il + aab + j * knsupc],
                                                dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                               //tmp_sum += dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                               //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,dready_lsum[%d]=%f\n", mype, bid, tid, lib, k, ii,
                               //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                               //       dready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                           }

                       }
                       // atomic return old val
                       fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                       //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,fmod_tmp);
                   }

                   if (fmod_tmp == 1) {// forward RD
                       //printf("sum1-(%d,%d,%d),lib=%d, myRoot=%d\n", mype, bid, tid, lib,LRtree_ptr[lib].myRoot_);
                       if (LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_) {
                           my_flag_rd[lib * RDMA_FLAG_SIZE] = lib;
                           my_flag_rd[lib * RDMA_FLAG_SIZE + 1] = LRtree_ptr[lib].msgSize_;
                           RHS_ITERATE(j) {
                               for (int aab = 0; aab < knsupc; aab++) {
                                   dready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc] = lsum[il + aab + j * knsupc];
                                   //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,dready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
                                   //       k * maxrecvsz * 2 + i +j * knsupc,
                                   //       dready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

                               }
                           }
                           //printf("(%d,%d,%d),in wait lib=%d,k=%d,myflagrd=%d,%d\n", mype, bid, tid, lib, k,
                           //       my_flag_rd[k * RDMA_FLAG_SIZE], my_flag_rd[k * RDMA_FLAG_SIZE + 1]);
                           //int temp_mysendcout=atomicAdd(&d_flag_mod[0], 1);
                           //int temp_flag_mod=atomicExch(&d_flag_mod[temp_mysendcout+1],lib);
                           //printf("iam=%d in wait2,lib=%d,%d,%d, pos=%d, temp %d,%d\n",mype,lib,k, d_flag_mod[temp_mysendcout+1], temp_mysendcout+1, temp_mysendcout,temp_flag_mod);
                           symldl_literature_reduce_forward_device(&LRtree_ptr[lib], flag_rd_q,
                                                                &my_flag_rd[RDMA_FLAG_SIZE * lib], mype, bid, tid,
                                                                &dready_lsum[0], maxrecvsz,LRtree_ptr[lib].myRoot_);
                       }
                   }
               }
           }//for
       } // else WAIT_NUM_THREAD<recv
   }
#if 0
    if (bid==2){
        int tot_threads=blockDim.x * blockDim.y;
        //if (tid==0){
        //    printf("iam=%d, len=%d, tot_threads=%d\n",mype,d_nfrecvmod[3],tot_threads);
        //}

        //if (d_nfrecvmod[3]==0) return;
        int lk=-1,k=-1,iam=-1,myroot=-1,myrank=-1;
        int mycol,myrow,lib;
        __shared__ int recv_num, finish_num;
        __shared__ int cur_send_num;
        recv_num = finish_num = 0;

        for (int i=1; i<d_nfrecvmod[3]+1;i=i+cur_send_num){

            if (tid==0){
                int tmp, tmp1;
                //printf("iam=%d,i=%d, count=%d\n",mype,i,d_flag_mod[0]);
                do {
                    tmp = d_flag_mod[0];
                    //tmp1 == d_flag_mod[tmp];
                    __threadfence();
                    //msg_recv=d_status[gc];
                    //msg_recv=flag_bc_q[gc];
                } while (tmp == finish_num);

                recv_num=tmp;
            }
            __syncthreads();
            cur_send_num=recv_num-finish_num;
            finish_num=recv_num;
            //if (cur_send_num==1) {
            //    lk=d_flag_mod[i];
            //    iam = grid->iam;
            //    mycol = MYCOL(iam, grid);
            //    myrow = MYROW(iam, grid);
            //    k = myrow + lk * grid->nprow; // global block row
            //    myroot=LRtree_ptr[lk].myRoot_;
            //    myrank=LRtree_ptr[lk].myRank_;
            //    //if (tid==0) printf("B,(%d,%d) loop=%d, recv_num=%d,cur_send_num=%d, k=%d, to %d\n",mype,tid,i, recv_num,cur_send_num,lk, myroot);
            //    //__syncthreads();
            //    dC_RdTree_forwardMessageBlock_Device(&LRtree_ptr[lk], (int*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &dready_lsum[0],maxrecvsz,myroot);
            //    //__syncthreads();
            //    //if (tid==0) printf("B Done,(%d,%d) loop=%d, recv_num=%d,cur_send_num=%d\n",mype,tid,i, recv_num,cur_send_num);
            //}else if ((cur_send_num <= tot_threads/32) && (cur_send_num >1)){
            if ((cur_send_num <= tot_threads/32)){
                if (tid/32 < cur_send_num){
                    lk=d_flag_mod[i+tid/32];
                    //if (tid%32==0) printf("-- Warp, (%d,%d) i=%d, recv_num=%d,cur_send_num=%d, lk=%d, size=%d\n",mype,tid,i, recv_num,cur_send_num,lk,my_flag_rd[RDMA_FLAG_SIZE*k+1]);
                    iam = grid->iam;
                    mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lk * grid->nprow; // global block row
                    myroot=LRtree_ptr[lk].myRoot_;
                    myrank=LRtree_ptr[lk].myRank_;
                    //if (tid%32==0) printf("W, (%d,%d) loop=%d, recv_num=%d,cur_send_num=%d, lk=%d, to %d\n",mype,tid,i, recv_num,cur_send_num, lk, myroot);
                    dC_RdTree_forwardMessageWarp_Device(&LRtree_ptr[lk], (uint64_t*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &dready_lsum[0],maxrecvsz,myroot);
                    //if (tid%32==0) printf("W Done, (%d,%d) loop=%d, recv_num=%d,cur_send_num=%d, lk=%d\n",mype,tid,i, recv_num,cur_send_num,lk);
                }
            }else if ((cur_send_num > tot_threads/32) && (cur_send_num <= tot_threads)){
                __syncthreads();
                if (tid < cur_send_num){
                    lk=d_flag_mod[i+tid];
                    iam = grid->iam;
                    mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lk * grid->nprow; // global block row
                    myroot=LRtree_ptr[lk].myRoot_;
                    myrank=LRtree_ptr[lk].myRank_;
                    //printf("-- Thread, (%d,%d) i=%d, recv_num=%d,cur_send_num=%d, lk=%d, size=%d\n",mype,tid,i, recv_num,cur_send_num,lk,my_flag_rd[RDMA_FLAG_SIZE*k+1]);
                    dC_RdTree_forwardMessageThread_Device(&LRtree_ptr[lk], (uint64_t*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &dready_lsum[0],maxrecvsz,myroot);
                    //printf("T Done,(%d,%d) recv_num=%d,cur_send_num=%d\n",mype,tid, recv_num,cur_send_num);
                }
            }else if (cur_send_num > tot_threads){
                int delta=cur_send_num%tot_threads;
                int mynum=cur_send_num/tot_threads;
                int myoffset=0;
                if (tid < delta){
                    myoffset=tid*(mynum+1);
                }else{
                    myoffset=(delta*(mynum+1))+(tid-delta)*mynum;
                }

                for(int j=0;j<mynum;j++){
                    lk=d_flag_mod[i+myoffset+j];
                    iam = grid->iam;
                    mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lk * grid->nprow; // global block row
                    myroot=LRtree_ptr[lk].myRoot_;
                    myrank=LRtree_ptr[lk].myRank_;
                    //printf("-- Threadloop, (%d,%d) i=%d, recv_num=%d,cur_send_num=%d, lk=%d, size=%d\n",mype,tid,i, recv_num,cur_send_num,lk,my_flag_rd[RDMA_FLAG_SIZE*k+1]);
                    dC_RdTree_forwardMessageThread_Device(&LRtree_ptr[lk], (uint64_t*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &dready_lsum[0],maxrecvsz,myroot);
                    //printf("Ts Done,(%d,%d) loop=%d (%d,%d) recv_num=%d,cur_send_num=%d, k=%d, to %d\n",mype, tid,i, j, mynum,recv_num,cur_send_num,lk,myroot);
                }
            }

            __syncthreads();

            //if (tid==0) printf("iam=%d,tid=%d,i=%d, recv_num=%d,cur_send_num=%d\n",mype,tid,i,recv_num,cur_send_num);
        }
    }
#endif
#endif
}

__global__ void symldl_literature_forward_kernel
/************************************************************************/
        (
                int nbcol_loc,
                int nblock_ex,
                double *lsum,    /* Sum of local modifications.                        */
                double *x,       /* X array (local)                                    */
                int   nrhs,      /* Number of right-hand sides.                        */
                int   maxsup,      /* Max supernode size.                        */
                int_t   nsupers,      /* Number of total supernodes.                        */
                int *fmod,     /* Modification count for L-solve.                    */
                C_Tree  *LBtree_ptr,
                C_Tree  *LRtree_ptr,
                int_t *ilsum,
                int_t *Lrowind_bc_dat,
                long int *Lrowind_bc_offset,
                double **Lnzval_bc_ptr,
                int_t *Lindval_loc_bc_dat,
                long int *Lindval_loc_bc_offset,
                int_t *xsup,
                gridinfo_t *grid,
                int_t *panel_gids,
                int_t *row_gids,
                int_t *row_local_index,
                int *diag_roots,
                int_t maxrecvsz,
                int mype,
                volatile uint64_t* flag_bc_q,
                volatile uint64_t* flag_rd_q,
                double* dready_x,
                double* dready_lsum,
                int* my_flag_bc,
                int* my_flag_rd,
                int* d_nfrecv,
                volatile int* d_status,
                volatile int* d_statusmod,
                int* d_flag_mod
        )
{
    double zero = 0.0, alpha = 1.0, beta = 0.0;
    double *lusup;
    int    iam, iknsupc, myrow, krow, nbrow, nbrow1, nsupr, m;
    int_t  k,i, l,ii, ik, il, irow, j, lb, lk, rel, lib;
    int_t  *lsub, *lloc;
    int_t  luptr_tmp1,lptr1_tmp, idx_i, idx_v, fmod_tmp;
    //__shared__ double rtemp_loc[128];
    double temp1;
    int_t lptr;      /* Starting position in lsub[*].                      */
    int aln_i;
    // aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    int   knsupc;    /* Size of supernode k.                               */
    int_t nlb;       /* Number of L blocks.                                */

    int bid=blockIdx_x;
    int_t tmp;
    int tid = threadIdx_x + threadIdx_y * blockDim_x;
// int_t lock = 0;
    const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/
    double rC[THR_N][THR_M];
    int idx = threadIdx_x;  // thread's m dimension
    int idy = threadIdx_y;  // thread's n dimension
    int_t ni,mi;
    int cnt;

    if (Lrowind_bc_offset[bid] == -1) {
        return;
    }

    int gc;

    lk = bid;
    iam = grid->iam;
    myrow = MYROW(iam, grid);
    gc = panel_gids[lk];
    if (gc < 0 || gc >= nsupers) return;
    k = gc;

    knsupc = SuperSize(k);
    lsub = &Lrowind_bc_dat[Lrowind_bc_offset[lk]];
    iam = grid->iam;
    krow = diag_roots[k];
    lusup = Lnzval_bc_ptr[lk];
    lloc = &Lindval_loc_bc_dat[Lindval_loc_bc_offset[lk]];
    nsupr = lsub[1];

    if (myrow == krow) {
        nlb = lsub[0] - 1;
	// idx_n = 1;
        idx_i = nlb + 2;
        idx_v = 2 * nlb + 3;
        // luptr_tmp = lloc[idx_v];
        m = nsupr - knsupc;
    } else {
        nlb = lsub[0];
        // idx_n = 0;
        idx_i = nlb;
        idx_v = 2 * nlb;
        // luptr_tmp = lloc[idx_v];
        m = nsupr;
    }

    if (myrow == krow) {   /* diagonal block performs trsm and forward the message*/

        if (tid == 0) {  /*only the first thread in a block handles the lock */
            //printf("(%d) iam bid=%d,enter solve--2, wait lock,gc=%d\n",mype,bid,gc);
            //printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i],myrow,krow);
            // for (i=0 ; i<maxsup ; i++){
            // rtemp_loc[i]=0.0;
            // }

            lib = row_local_index[k];
            do {
                tmp = fmod[lib * aln_i];
                __threadfence();
            } while (tmp > 0);
        }
        __syncthreads();
        //if(tid==0) printf("(%d) iam bid=%d,enter solve--2, unlock,gc=%d\n",mype,bid,gc);


        lib = row_local_index[k];
        il = LSUM_BLK(lib);
        ii = X_BLK(lib);

        RHS_ITERATE(j)
            for (i = tid; i < knsupc; i += block_size) {
                //d_atomicAdd(&dready_x[0],lsum[i + il + j * knsupc]);
                x[i + ii + j*knsupc] += lsum[i + il + j*knsupc];

            }
        __syncthreads();
        //if(tid==0) printf("(%d,%d,%d),CHECKING k=%d,gc=%d,checksum=%lf\n",mype,bid,tid,k,gc,dready_x[0]);
        //if(tid==0) printf("(%d) iam bid=%d,enter solve--3,gc=%d\n",mype,bid,gc);

        RHS_ITERATE(j)for (i = tid; i < knsupc; i += block_size)
                dready_x[i + maxrecvsz * lk + j * knsupc] = x[i + ii + j * knsupc];

        __syncthreads();
    } else {   /* off-diagonal block forward the message*/
        /* waiting for the x subvector and forward*/
        //YL: only the first thread in a block spin-waits for the coming x subvector message using NVSHMEM, put the message into dready_x[maxrecvsz*lk]
        volatile uint64_t msg_recv = 0;
        if (tid == 0) {
            //printf("in solve WAIT1 (%d,%d) wait for col %d,flag=%d\n", mype, bid, gc,flag_bc_q[gc]);
            //nvshmem_signal_wait_until((int *) flag_bc_q + gc, NVSHMEM_CMP_EQ, 1);
            do {
                msg_recv = flag_bc_q[lk];
                //msg_recv=d_status[gc];
                //msg_recv=flag_bc_q[gc];
                __threadfence();
            } while (msg_recv != 1);
            //printf("(%d,%d,%d,%d) in compute kernel, I have msg=%d,sz=%d,ofset=%d\n",mype,bid,tid,gc,msg_recv,LBtree_ptr[lk].msgSize_*nrhs+XK_H,maxrecvsz*lk);
            //double sum=0;
            //for (int myi=0;myi<LBtree_ptr[lk].msgSize_*nrhs+XK_H;myi++){
            //    sum+=dready_x[maxrecvsz*lk+myi];
            //}
            //printf("(%d,%d,%d), gc=%d,lk=%d, sum=%lf\n",mype,bid,tid,gc,lk,sum);
        }
        __syncthreads();
    }
    __syncthreads();

//YL: only the first thread in a block forwards the x subvector using NVSHMEM
    cnt = LBtree_ptr[lk].destCnt_;
    if (cnt > 0) {
        //cnt=LBtree_ptr[lk].msgSize_;
        my_flag_bc[lk * RDMA_FLAG_SIZE] = lk;
        my_flag_bc[lk * RDMA_FLAG_SIZE + 1] = LBtree_ptr[lk].msgSize_ * nrhs + XK_H;
        symldl_literature_bcast_forward_device(&LBtree_ptr[lk], flag_bc_q, &my_flag_bc[lk * RDMA_FLAG_SIZE],
                                             mype, tid, &dready_x[0], maxrecvsz);
        //printf("(%d,%d,%d), lk=%d, gc=%d\n",mype,bid,tid,lk,gc);
        //symldl_literature_bcast_forward_device(&LBtree_ptr[lk],&dready_x[maxrecvsz*lk],cnt*nrhs+XK_H);
    }
    int keep_lk = lk;
    __syncthreads();

    if (nlb > 0) {

        lib = row_local_index[k];
        ii = X_BLK(lib);

        if (nrhs == 1) {
            luptr_tmp1 = lloc[idx_v];
            lb = 0;
            nbrow = 0;
            lptr1_tmp = lloc[lb + idx_i];
            lptr = lptr1_tmp + 2;
            nbrow1 = lsub[lptr1_tmp + 1];
            ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
            rel = xsup[ik]; /* Global row index of block ik. */
            lk = row_local_index[ik];
            iknsupc = SuperSize(ik);
            il = LSUM_BLK(lk);

            for (i = tid; i < m; i += block_size) {
                while (nbrow + lsub[lptr1_tmp + 1] <= i) {
                    lb++;
                    nbrow += lsub[lptr1_tmp + 1];
                    lptr1_tmp = lloc[lb + idx_i];
                    lptr = lptr1_tmp + 2;
                    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                    rel = xsup[ik]; /* Global row index of block ik. */
                    lk = row_local_index[ik];
                    iknsupc = SuperSize(ik);
                    il = LSUM_BLK(lk);
                }

                irow = lsub[lptr + i - nbrow] - rel; /* Relative row. */
                RHS_ITERATE(j) {
                    temp1 = zero;
                    for (l = 0; l < knsupc; l++) {
                        temp1 += lusup[luptr_tmp1 + l * nsupr + i] * dready_x[l + maxrecvsz * keep_lk + j * knsupc];
                        //temp1+= lusup[luptr_tmp1+l*nsupr+i]*x[ii+j*knsupc+l];
                    }
                    d_atomicAdd(&lsum[il + irow + j * iknsupc], -temp1);
                    //printf("(%d,%d,%d),lsum[%d]=%f\n",mype,bid,tid,il+irow + j*iknsupc,lsum[il+irow + j*iknsupc]);
                }

                //  irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
                //  if(i==nbrow+lsub[lptr1_tmp+1]-1){
                //   fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
                //   // __threadfence();
                //  }


            }
            __syncthreads();

            luptr_tmp1 = lloc[idx_v];
            lb = 0;
            nbrow = 0;
            lptr1_tmp = lloc[lb + idx_i];
            lptr = lptr1_tmp + 2;
            nbrow1 = lsub[lptr1_tmp + 1];
            ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
            rel = xsup[ik]; /* Global row index of block ik. */
            lk = row_local_index[ik];
            iknsupc = SuperSize(ik);
            il = LSUM_BLK(lk);
            // gr=myrow + lk * grid->nprow;
            //knsupc = SuperSize(gr);

            for (i = tid; i < m; i += block_size) {
                while (nbrow + lsub[lptr1_tmp + 1] <= i) {
                    lb++;
                    nbrow += lsub[lptr1_tmp + 1];
                    lptr1_tmp = lloc[lb + idx_i];
                    lptr = lptr1_tmp + 2;
                    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                    rel = xsup[ik]; /* Global row index of block ik. */
                    lk = row_local_index[ik];
                    iknsupc = SuperSize(ik);
                    il = LSUM_BLK(lk);
                }
                //if (ik==15) printf("(%d) iam bid=%d,enter solve--3,fmod=%d\n",mype,bid,fmod_tmp);

                irow = lsub[lptr + i - nbrow] - rel; /* Relative row. */
                if (i == nbrow + lsub[lptr1_tmp + 1] - 1) {
                    // atomic return old val, omp return new val
                    fmod_tmp = atomicSub(&fmod[lk * aln_i], 1);
                    // __threadfence();
                    if(fmod_tmp==1) {// forward RD
                        //senddone[lk]=1;
                        if(LRtree_ptr[lk].myRoot_ != LRtree_ptr[lk].myRank_){
                            //cnt=LRtree_ptr[lib].msgSize_;

                            my_flag_rd[lk*RDMA_FLAG_SIZE]=lk;
                            my_flag_rd[lk*RDMA_FLAG_SIZE+1]=LRtree_ptr[lk].msgSize_;
                            //double tmp_sum=0;
                            RHS_ITERATE(j) {
                                for (int aab = 0; aab < iknsupc; aab++) {
                                    dready_lsum[lk * maxrecvsz * 2 + aab +j * iknsupc] = lsum[il + aab +j * iknsupc];
                                    //tmp_sum += dready_lsum[lk * maxrecvsz * 2 + aab +j * iknsupc];
                                    //printf("data3-(%d,%d,%d),lib=%d,k=%d,%d,i=%d,sum=%lf,dready_lsum[%d]=%lf, size=%d\n", mype, bid, tid, lk, gr,ik, i, tmp_sum,
                                    //       lk * maxrecvsz * 2 + aab +j * iknsupc,
                                    //       dready_lsum[lk * maxrecvsz * 2 + aab +j * iknsupc],my_flag_rd[ik*RDMA_FLAG_SIZE+1]);

                                }
                            }
                            // int temp_mysendcout=atomicAdd(&d_flag_mod[0], 1);
                            // int temp_flag_mod=atomicExch(&d_flag_mod[temp_mysendcout+1],lk);
                            //printf("iam=%d in solve,lib=%d,%d,%d, "
                            //       "pos=%d, temp %d,%d, "
                            //       "maxrecvsz=%d\n",mype,lk,k, d_flag_mod[temp_mysendcout+1],
                            //       temp_mysendcout+1,
                            //       temp_mysendcout,temp_flag_mod,
                            //       maxrecvsz);
                            //printf("(%d,%d,%d) in solve,lib=%d,gr=%d,ik=%d,myflagrd=%d,%d\n",mype,bid,tid,lk,gr,ik,my_flag_rd[ik*RDMA_FLAG_SIZE],my_flag_rd[ik*RDMA_FLAG_SIZE+1]);
                            symldl_literature_reduce_forward_device(&LRtree_ptr[lk], flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*lk], mype, bid, tid, &dready_lsum[0],maxrecvsz,LRtree_ptr[lk].myRoot_);
                        }
                    }
                }
            }
            //__syncthreads();

        } else {
            for (lb = 0; lb < nlb; lb++) {
                luptr_tmp1 = lloc[lb + idx_v];

                // nbrow=0;
                // lptr1_tmp = lloc[lb+idx_i];
                // nbrow += lsub[lptr1_tmp+1];


                lib = row_local_index[k];
                ii = X_BLK(lib);

                lptr1_tmp = lloc[lb + idx_i];
                lptr = lptr1_tmp + 2;
                nbrow1 = lsub[lptr1_tmp + 1];
                ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                rel = xsup[ik]; /* Global row index of block ik. */

                lk = row_local_index[ik];

                iknsupc = SuperSize(ik);
                il = LSUM_BLK(lk);
                for (int blx = 0; blx * BLK_M < nbrow1; blx++) {
                    for (int bly = 0; bly * BLK_N < nrhs; bly++) {
                        symldl_literature_gemm_device(nbrow1, nrhs, knsupc, blx, bly,
                                               &lusup[luptr_tmp1], nsupr, &dready_x[maxrecvsz * keep_lk], knsupc, rC,
                                               alpha, beta);
#pragma unroll
                        for (ni = 0; ni < THR_N; ni++) {
                            int coord_dCn = bly * BLK_N + ni * DIM_Y + idy;
#pragma unroll
                            for (mi = 0; mi < THR_M; mi++) {
                                int coord_dCm = blx * BLK_M + mi * DIM_X + idx;
                                if (coord_dCm < nbrow1 && coord_dCn < nrhs) {
                                    irow = lsub[lptr + coord_dCm] - rel; /* Relative row. */
                                    double &regC = rC[ni][mi];
                                    d_atomicAdd(&lsum[il + irow + coord_dCn * iknsupc], -regC);


                                }
                            }
                        }
                    }
                }
                if (tid == 0) fmod_tmp = atomicSub(&fmod[lk * aln_i], 1);


            }

        }//if(nrhs==1)
    } /* if nlb>0*/

} /* symldl_literature_forward_kernel */

static int
symldl_literature_forward_wrap(
    int nbcol_loc, int nbrow_loc, double *lsum, double *x, int nrhs,
    int maxsup, int_t nsupers, int *fmod, C_Tree *LBtree_ptr,
    C_Tree *LRtree_ptr, int_t *ilsum, int_t *Lrowind_bc_dat,
    long int *Lrowind_bc_offset, double **Lnzval_bc_ptr,
    int_t *Lindval_loc_bc_dat, long int *Lindval_loc_bc_offset,
    int_t *xsup, gridinfo_t *grid, int_t *panel_gids, int_t *row_gids,
    int_t *row_local_index, int *diag_roots, int_t maxrecvsz,
    uint64_t *flag_bc_q, uint64_t *flag_rd_q, double *dready_x,
    double *dready_lsum, int *my_flag_bc, int *my_flag_rd,
    int *d_nfrecv, int *h_nfrecv, int *d_status, int *d_colnum,
    int *d_mynum, int *d_mymaskstart, int *d_mymasklength,
    int *d_nfrecvmod, int *d_statusmod, int *d_colnummod,
    int *d_mynummod, int *d_mymaskstartmod, int *d_mymasklengthmod,
    int *d_recv_cnt, int *d_msgnum, int *d_flag_mod)
{
#ifndef HAVE_NVSHMEM
    (void) nbcol_loc; (void) nbrow_loc; (void) lsum; (void) x;
    (void) nrhs; (void) maxsup; (void) nsupers; (void) fmod;
    (void) LBtree_ptr; (void) LRtree_ptr; (void) ilsum;
    (void) Lrowind_bc_dat; (void) Lrowind_bc_offset;
    (void) Lnzval_bc_ptr; (void) Lindval_loc_bc_dat;
    (void) Lindval_loc_bc_offset; (void) xsup; (void) grid;
    (void) panel_gids; (void) row_gids; (void) row_local_index;
    (void) diag_roots; (void) maxrecvsz; (void) flag_bc_q;
    (void) flag_rd_q; (void) dready_x; (void) dready_lsum;
    (void) my_flag_bc; (void) my_flag_rd; (void) d_nfrecv;
    (void) h_nfrecv; (void) d_status; (void) d_colnum;
    (void) d_mynum; (void) d_mymaskstart; (void) d_mymasklength;
    (void) d_nfrecvmod; (void) d_statusmod; (void) d_colnummod;
    (void) d_mynummod; (void) d_mymaskstartmod;
    (void) d_mymasklengthmod; (void) d_recv_cnt; (void) d_msgnum;
    (void) d_flag_mod;
    return -1;
#else
    int nblock_ex = CEILING(nbrow_loc, ((DIM_X * DIM_Y) / 32));
    int mype = nvshmem_my_pe();
    cudaStream_t stream[2] = {0, 0};
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));

    cudaFuncAttributes cuattr;
    CUDA_CHECK(cudaFuncGetAttributes(&cuattr,
                                     symldl_literature_forward_kernel));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize,
                                  cuattr.localSizeBytes));

    int min_grid_size = 0;
    int wait_threads = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &wait_threads,
        (const void *) symldl_literature_wait_bcrd, 0, 0));
    (void) min_grid_size;
    if (wait_threads < h_nfrecv[1]) {
        h_nfrecv[1] = wait_threads;
        CUDA_CHECK(cudaMemcpy(d_nfrecv, h_nfrecv, 3 * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    dim3 wait_grid(h_nfrecv[2]);
    dim3 wait_block(h_nfrecv[1]);
    dim3 solve_block(DIM_X, DIM_Y);
    void *args[] = {
        &nrhs, &LRtree_ptr, &maxrecvsz, &mype, &flag_bc_q, &flag_rd_q,
        &dready_x, &dready_lsum, &my_flag_bc, &my_flag_rd, &d_nfrecv,
        &d_status, &d_colnum, &d_mynum, &d_mymaskstart, &d_mymasklength,
        &d_nfrecvmod, &d_statusmod, &d_colnummod, &d_mynummod,
        &d_mymaskstartmod, &d_mymasklengthmod, &d_recv_cnt, &d_msgnum,
        &d_flag_mod, &lsum, &fmod, &grid, &xsup, &ilsum, &row_gids,
        &nbrow_loc, &nsupers
    };

    int status = nvshmemx_collective_launch(
        (const void *) symldl_literature_wait_bcrd, wait_grid, wait_block,
        args, 0, stream[0]);
    if (status != NVSHMEMX_SUCCESS)
        return -1;

    if (nbcol_loc > 0) {
        dim3 solve_grid(nbcol_loc);
        symldl_literature_forward_kernel<<<solve_grid, solve_block, 0,
                                           stream[1]>>>(
            nbcol_loc, nblock_ex, lsum, x, nrhs, maxsup, nsupers, fmod,
            LBtree_ptr, LRtree_ptr, ilsum, Lrowind_bc_dat,
            Lrowind_bc_offset, Lnzval_bc_ptr, Lindval_loc_bc_dat,
            Lindval_loc_bc_offset, xsup, grid, panel_gids, row_gids,
            row_local_index, diag_roots, maxrecvsz, mype, flag_bc_q,
            flag_rd_q, dready_x, dready_lsum, my_flag_bc, my_flag_rd,
            d_nfrecv, d_status, d_statusmod, d_flag_mod);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    return 0;
#endif
}

namespace {

struct dSymLDLLiteratureForwardState {
    int_t n;
    int_t nsupers;
    int nrhs;
    int_t x_count;
    int_t lsum_count;
    int_t panel_count;
    int_t row_count;
    int_t panel_slots;
    int_t row_slots;
    int maxsup;
    int maxrecvsz;
    int rank;
    int nprocs;
    int diagnostics;
    int sparse_reduce_calls;
    int sparse_broadcast_calls;
    int expected_bcast_recvs;
    int expected_reduce_recvs;
    int expected_reduce_sends;
    int h_nfrecv[3];
    int h_nfrecvmod[4];
    double setup_time;
    double forward_time;
    double sparse_reduce_time;
    double sparse_broadcast_time;
    double h2d_time;
    double d2h_time;
    dtrf3Dpartition_t *trf3Dpartition;
    gridinfo3d_t *grid3d;
    gridinfo_t *grid;
    MPI_Comm layer_comm;
    const int_t *host_xsup;
    const int_t *host_ilsum;

    std::vector<int> h_fmod;
    std::vector<int> h_status;
    std::vector<int> h_statusmod;
    std::vector<int> h_recv_cnt;
    std::vector<int> h_colnum;
    std::vector<int> h_colnummod;
    std::vector<int_t> h_panel_gids;
    std::vector<int_t> h_row_gids;

    double *d_x;
    double *d_lsum;
    int *d_fmod;
    C_Tree *d_bcast_trees;
    C_Tree *d_reduce_trees;
    int_t *d_ilsum;
    int_t *d_xsup;
    int_t *d_lsub;
    long int *d_lsub_offsets;
    int_t *d_lloc;
    long int *d_lloc_offsets;
    double **d_panel_values;
    int_t *d_panel_gids;
    int_t *d_row_gids;
    int_t *d_row_local_index;
    int *d_diag_roots;
    gridinfo_t *d_grid;
    int *d_nfrecv;
    int *d_status;
    int *d_colnum;
    int *d_mynum;
    int *d_mymaskstart;
    int *d_mymasklength;
    int *d_nfrecvmod;
    int *d_statusmod;
    int *d_colnummod;
    int *d_mynummod;
    int *d_mymaskstartmod;
    int *d_mymasklengthmod;
    int *d_recv_cnt;
    int *d_msgnum;
    int *d_flag_mod;

    uint64_t *flag_bc_q;
    uint64_t *flag_rd_q;
    double *ready_x;
    double *ready_lsum;
    int *my_flag_bc;
    int *my_flag_rd;
};

static int_t
symldl_literature_super_size(
    const dSymLDLLiteratureForwardState *state, int_t k)
{
    return state->host_xsup[k + 1] - state->host_xsup[k];
}

static int_t
symldl_literature_x_block(
    const dSymLDLLiteratureForwardState *state, int_t local_row)
{
    return state->host_ilsum[local_row] * state->nrhs +
           (local_row + 1) * XK_H;
}

static int symldl_literature_nvshmem_initialized = 0;

static int
symldl_literature_env_enabled(const char *name)
{
    const char *value = getenv(name);
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

template <typename T>
static void
symldl_literature_cuda_alloc(T **ptr, size_t count)
{
    size_t n = std::max((size_t) 1, count);
    CUDA_CHECK(cudaMalloc((void **) ptr, n * sizeof(T)));
}

template <typename T>
static void
symldl_literature_cuda_copy(T **ptr, const T *source, size_t count)
{
    symldl_literature_cuda_alloc(ptr, count);
    if (count > 0)
        CUDA_CHECK(cudaMemcpy(*ptr, source, count * sizeof(T),
                              cudaMemcpyHostToDevice));
}

static void
symldl_literature_cuda_free(void *ptr)
{
    if (ptr != NULL)
        cudaFree(ptr);
}

static void
symldl_literature_state_free(dSymLDLLiteratureForwardState *state)
{
    if (state == NULL)
        return;

    symldl_literature_cuda_free(state->d_x);
    symldl_literature_cuda_free(state->d_lsum);
    symldl_literature_cuda_free(state->d_fmod);
    symldl_literature_cuda_free(state->d_bcast_trees);
    symldl_literature_cuda_free(state->d_reduce_trees);
    symldl_literature_cuda_free(state->d_ilsum);
    symldl_literature_cuda_free(state->d_xsup);
    symldl_literature_cuda_free(state->d_lsub);
    symldl_literature_cuda_free(state->d_lsub_offsets);
    symldl_literature_cuda_free(state->d_lloc);
    symldl_literature_cuda_free(state->d_lloc_offsets);
    symldl_literature_cuda_free(state->d_panel_values);
    symldl_literature_cuda_free(state->d_panel_gids);
    symldl_literature_cuda_free(state->d_row_gids);
    symldl_literature_cuda_free(state->d_row_local_index);
    symldl_literature_cuda_free(state->d_diag_roots);
    symldl_literature_cuda_free(state->d_grid);
    symldl_literature_cuda_free(state->d_nfrecv);
    symldl_literature_cuda_free(state->d_status);
    symldl_literature_cuda_free(state->d_colnum);
    symldl_literature_cuda_free(state->d_mynum);
    symldl_literature_cuda_free(state->d_mymaskstart);
    symldl_literature_cuda_free(state->d_mymasklength);
    symldl_literature_cuda_free(state->d_nfrecvmod);
    symldl_literature_cuda_free(state->d_statusmod);
    symldl_literature_cuda_free(state->d_colnummod);
    symldl_literature_cuda_free(state->d_mynummod);
    symldl_literature_cuda_free(state->d_mymaskstartmod);
    symldl_literature_cuda_free(state->d_mymasklengthmod);
    symldl_literature_cuda_free(state->d_recv_cnt);
    symldl_literature_cuda_free(state->d_msgnum);
    symldl_literature_cuda_free(state->d_flag_mod);
#ifdef HAVE_NVSHMEM
    if (state->flag_bc_q) nvshmem_free(state->flag_bc_q);
    if (state->flag_rd_q) nvshmem_free(state->flag_rd_q);
    if (state->ready_x) nvshmem_free(state->ready_x);
    if (state->ready_lsum) nvshmem_free(state->ready_lsum);
    if (state->my_flag_bc) nvshmem_free(state->my_flag_bc);
    if (state->my_flag_rd) nvshmem_free(state->my_flag_rd);
#endif
    delete state;
}

static void
symldl_literature_panel_targets(
    const dSymLDLLiteraturePanelDesc &panel,
    const int_t *row_local_index, int_t source_gid,
    std::vector<int_t> &targets)
{
    if (panel.lsub == NULL)
        return;
    int_t lptr = BC_HEADER;
    for (int_t lb = 0; lb < panel.lsub[0]; ++lb) {
        int_t target = panel.lsub[lptr];
        int_t nbrow = panel.lsub[lptr + 1];
        if (target != source_gid) {
            if (target < 0 || row_local_index[target] < 0)
                ABORT("SymLDL literature forward panel targets a row outside its Z layer.");
            targets.push_back(target);
        }
        lptr += LB_DESCRIPTOR + nbrow;
    }
    if (lptr > panel.lsub_count)
        ABORT("SymLDL literature forward panel metadata is truncated.");
}

static void
symldl_literature_validate_panel(
    const dSymLDLLiteratureForwardState *state,
    const dSymLDLLiteraturePanelDesc &panel)
{
    int myrow = MYROW(state->grid->iam, state->grid);
    int root = state->trf3Dpartition->symV2DiagRoot[panel.gid];
    if (panel.lsub == NULL) {
        if (myrow == root)
            ABORT("SymLDL literature forward diagonal panel is missing.");
        return;
    }
    if (panel.lloc == NULL || panel.values == NULL ||
        panel.lsub_count < BC_HEADER || panel.lsub[0] <= 0 ||
        panel.lsub[1] <= 0 ||
        panel.lloc_count != 3 * panel.lsub[0])
        ABORT("SymLDL literature forward panel storage is incomplete.");

    int_t lptr = BC_HEADER;
    for (int_t lb = 0; lb < panel.lsub[0]; ++lb) {
        if (lptr + LB_DESCRIPTOR > panel.lsub_count)
            ABORT("SymLDL literature forward panel metadata is truncated.");
        int_t target = panel.lsub[lptr];
        int_t nbrow = panel.lsub[lptr + 1];
        if (target < 0 || target >= state->nsupers || nbrow <= 0 ||
            lptr + LB_DESCRIPTOR + nbrow > panel.lsub_count)
            ABORT("SymLDL literature forward panel block is invalid.");
        if (state->trf3Dpartition->symV2RowLocalIndex[target] < 0)
            ABORT("SymLDL literature forward panel targets a row outside its Z layer.");
        lptr += LB_DESCRIPTOR + nbrow;
    }
    if (lptr != panel.lsub_count)
        ABORT("SymLDL literature forward panel metadata length is inconsistent.");

    if (myrow == root) {
        if (panel.lsub[BC_HEADER] != panel.gid)
            ABORT("SymLDL literature forward diagonal block is not first.");
    } else {
        for (int_t lb = 0, ptr = BC_HEADER; lb < panel.lsub[0]; ++lb) {
            if (panel.lsub[ptr] == panel.gid)
                ABORT("SymLDL literature forward diagonal block is on the wrong process row.");
            ptr += LB_DESCRIPTOR + panel.lsub[ptr + 1];
        }
    }
}

static void
symldl_literature_build_bcast_trees(
    dSymLDLLiteratureForwardState *state,
    const dSymLDLLiteraturePanelDesc *panels,
    std::vector<C_Tree> &trees)
{
    gridinfo_t *grid = state->grid;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int Pr = grid->nprow;
    std::vector<int> counts((size_t) Pr);
    std::vector<int> displs((size_t) Pr);

    trees.resize((size_t) state->panel_count);
    state->h_status.assign((size_t) state->panel_count, 1);
    for (int_t lp = 0; lp < state->panel_count; ++lp) {
        C_BcTree_Nullify(&trees[(size_t) lp]);
        int_t k = state->h_panel_gids[(size_t) lp];
        std::vector<int_t> local_targets;
        symldl_literature_panel_targets(
            panels[lp], state->trf3Dpartition->symV2RowLocalIndex, k,
            local_targets);
        int count = (int) local_targets.size();
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      grid->cscp.comm);
        int total = 0;
        for (int pr = 0; pr < Pr; ++pr) {
            displs[(size_t) pr] = total;
            total += counts[(size_t) pr];
        }
        std::vector<int_t> all_targets((size_t) std::max(1, total));
        MPI_Allgatherv(local_targets.data(), count, mpi_int_t,
                       all_targets.data(), counts.data(), displs.data(),
                       mpi_int_t, grid->cscp.comm);

        std::vector<int_t> first((size_t) Pr, 3 * state->nsupers);
        for (int pr = 0; pr < Pr; ++pr)
            for (int pos = displs[(size_t) pr];
                 pos < displs[(size_t) pr] + counts[(size_t) pr]; ++pos)
                first[(size_t) pr] = std::min(first[(size_t) pr],
                                               all_targets[(size_t) pos]);
        int root = state->trf3Dpartition->symV2DiagRoot[k];
        first[(size_t) root] = std::min(first[(size_t) root], k);

        std::vector<std::pair<int_t, int> > ordered;
        for (int pr = 0; pr < Pr; ++pr)
            if (pr != root && first[(size_t) pr] != 3 * state->nsupers)
                ordered.push_back(std::make_pair(first[(size_t) pr], pr));
        std::sort(ordered.begin(), ordered.end());
        std::vector<int> ranks;
        ranks.push_back(PNUM(root, mycol, grid));
        for (size_t i = 0; i < ordered.size(); ++i)
            ranks.push_back(PNUM(ordered[i].second, mycol, grid));

        int active = first[(size_t) myrow] != 3 * state->nsupers;
        if (active && ranks.size() > 1) {
            int needrecv = 0;
            C_BcTree_Create_nv(&trees[(size_t) lp], state->layer_comm,
                               ranks.data(), (int) ranks.size(),
                               symldl_literature_super_size(state, k), 'd',
                               &needrecv);
            trees[(size_t) lp].tag_ = BC_L;
            if (needrecv) {
                state->h_status[(size_t) lp] = 0;
                state->h_colnum.push_back((int) lp);
                ++state->expected_bcast_recvs;
            }
        }
    }
}

static void
symldl_literature_build_reduce_trees(
    dSymLDLLiteratureForwardState *state,
    const dSymLDLLiteraturePanelDesc *panels,
    std::vector<C_Tree> &trees)
{
    gridinfo_t *grid = state->grid;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int Pc = grid->npcol;
    std::vector<std::vector<int_t> > local_sources(
        (size_t) state->row_count);

    state->h_fmod.assign((size_t) state->row_count, 0);
    for (int_t lp = 0; lp < state->panel_count; ++lp) {
        int_t source = state->h_panel_gids[(size_t) lp];
        std::vector<int_t> targets;
        symldl_literature_panel_targets(
            panels[lp], state->trf3Dpartition->symV2RowLocalIndex,
            source, targets);
        for (size_t i = 0; i < targets.size(); ++i) {
            int_t lr = state->trf3Dpartition->symV2RowLocalIndex[
                targets[i]];
            local_sources[(size_t) lr].push_back(source);
            ++state->h_fmod[(size_t) lr];
        }
    }

    trees.resize((size_t) state->row_count);
    state->h_statusmod.assign((size_t) state->row_count * 2, 1);
    state->h_recv_cnt.assign((size_t) state->row_count, 0);
    std::vector<int> counts((size_t) Pc);
    std::vector<int> displs((size_t) Pc);
    for (int_t lr = 0; lr < state->row_count; ++lr) {
        C_RdTree_Nullify(&trees[(size_t) lr]);
        int_t k = state->h_row_gids[(size_t) lr];
        int count = (int) local_sources[(size_t) lr].size();
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      grid->rscp.comm);
        int total = 0;
        for (int pc = 0; pc < Pc; ++pc) {
            displs[(size_t) pc] = total;
            total += counts[(size_t) pc];
        }
        std::vector<int_t> all_sources((size_t) std::max(1, total));
        MPI_Allgatherv(local_sources[(size_t) lr].data(), count, mpi_int_t,
                       all_sources.data(), counts.data(), displs.data(),
                       mpi_int_t, grid->rscp.comm);
        if (total == 0)
            continue;

        std::vector<int_t> last((size_t) Pc, -3 * state->nsupers);
        for (int pc = 0; pc < Pc; ++pc)
            for (int pos = displs[(size_t) pc];
                 pos < displs[(size_t) pc] + counts[(size_t) pc]; ++pos)
                last[(size_t) pc] = std::max(last[(size_t) pc],
                                              all_sources[(size_t) pos]);
        int root = state->trf3Dpartition->symV2PanelRoot[k];
        last[(size_t) root] = std::max(last[(size_t) root], k);

        std::vector<std::pair<int_t, int> > ordered;
        for (int pc = 0; pc < Pc; ++pc)
            if (pc != root && last[(size_t) pc] != -3 * state->nsupers)
                ordered.push_back(std::make_pair(last[(size_t) pc], pc));
        std::sort(ordered.begin(), ordered.end(),
                  [](const std::pair<int_t, int> &a,
                     const std::pair<int_t, int> &b) {
                      return a.first > b.first;
                  });
        std::vector<int> ranks;
        ranks.push_back(PNUM(myrow, root, grid));
        for (size_t i = 0; i < ordered.size(); ++i)
            ranks.push_back(PNUM(myrow, ordered[i].second, grid));

        int active = last[(size_t) mycol] != -3 * state->nsupers;
        if (active && ranks.size() > 1) {
            int needrecv = 0;
            int needsend = 0;
            C_RdTree_Create_nv(&trees[(size_t) lr], state->layer_comm,
                               ranks.data(), (int) ranks.size(),
                               symldl_literature_super_size(state, k), 'd',
                               &needrecv, &needsend);
            trees[(size_t) lr].tag_ = RD_L;
            state->expected_reduce_sends += needsend;
            if (needrecv > 0) {
                state->h_statusmod[(size_t) lr * 2] = 0;
                state->h_statusmod[(size_t) lr * 2 + 1] = 0;
                state->h_recv_cnt[(size_t) lr] = needrecv;
                state->h_colnummod.push_back((int) lr);
                state->expected_reduce_recvs += needrecv;
                state->h_fmod[(size_t) lr] += needrecv;
            }
        }
    }
}

static int
symldl_literature_z_reduce_tree(
    dSymLDLLiteratureForwardState *state, int_t tree_id, int_t sender,
    int_t receiver, double *x, double *recvbuf)
{
    sForest_t *forest = state->trf3Dpartition->sForests[tree_id];
    if (forest == NULL)
        return 0;
    int myz = state->grid3d->zscp.Iam;
    int myrow = MYROW(state->grid->iam, state->grid);
    int mycol = MYCOL(state->grid->iam, state->grid);
    for (int_t pos = 0; pos < forest->nNodes; ++pos) {
        int_t k = forest->nodeList[pos];
        if (myrow != state->trf3Dpartition->symV2DiagRoot[k] ||
            mycol != state->trf3Dpartition->symV2PanelRoot[k])
            continue;
        int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
        if (lr < 0)
            ABORT("SymLDL sparse Z reduction is missing a local row.");
        int_t ii = symldl_literature_x_block(state, lr);
        int count = (int) (symldl_literature_super_size(state, k) *
                           state->nrhs);
        if (myz == sender) {
            MPI_Send(&x[ii], count, MPI_DOUBLE, (int) receiver, (int) k,
                     state->grid3d->zscp.comm);
            std::fill(&x[ii], &x[ii] + count, 0.0);
        } else if (myz == receiver) {
            MPI_Recv(recvbuf, count, MPI_DOUBLE, (int) sender, (int) k,
                     state->grid3d->zscp.comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < count; ++i)
                x[ii + i] += recvbuf[i];
        }
    }
    return 0;
}

static int
symldl_literature_z_bcast_tree(
    dSymLDLLiteratureForwardState *state, int_t tree_id, int_t sender,
    int_t receiver, double *x)
{
    sForest_t *forest = state->trf3Dpartition->sForests[tree_id];
    if (forest == NULL)
        return 0;
    int myz = state->grid3d->zscp.Iam;
    int myrow = MYROW(state->grid->iam, state->grid);
    int mycol = MYCOL(state->grid->iam, state->grid);
    for (int_t pos = 0; pos < forest->nNodes; ++pos) {
        int_t k = forest->nodeList[pos];
        if (myrow != state->trf3Dpartition->symV2DiagRoot[k] ||
            mycol != state->trf3Dpartition->symV2PanelRoot[k])
            continue;
        int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
        if (lr < 0)
            ABORT("SymLDL sparse Z broadcast is missing a local row.");
        int_t ii = symldl_literature_x_block(state, lr);
        int count = (int) (symldl_literature_super_size(state, k) *
                           state->nrhs);
        if (myz == sender)
            MPI_Send(&x[ii], count, MPI_DOUBLE, (int) receiver, (int) k,
                     state->grid3d->zscp.comm);
        else if (myz == receiver)
            MPI_Recv(&x[ii], count, MPI_DOUBLE, (int) sender, (int) k,
                     state->grid3d->zscp.comm, MPI_STATUS_IGNORE);
    }
    return 0;
}

static int
symldl_literature_check_forward(
    dSymLDLLiteratureForwardState *state)
{
    if (!state->diagnostics)
        return 0;
    std::vector<int> fmod((size_t) state->row_count);
    std::vector<int> status((size_t) state->panel_count);
    std::vector<int> statusmod((size_t) state->row_count * 2);
    if (!fmod.empty())
        CUDA_CHECK(cudaMemcpy(fmod.data(), state->d_fmod,
                              fmod.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    if (!status.empty())
        CUDA_CHECK(cudaMemcpy(status.data(), state->d_status,
                              status.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    if (!statusmod.empty())
        CUDA_CHECK(cudaMemcpy(statusmod.data(), state->d_statusmod,
                              statusmod.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    int blocked = 0;
    int bcast_missing = 0;
    int reduce_missing = 0;
    for (size_t i = 0; i < fmod.size(); ++i)
        blocked += fmod[i] != 0;
    for (size_t i = 0; i < state->h_colnum.size(); ++i)
        bcast_missing += status[(size_t) state->h_colnum[i]] != 1;
    for (size_t i = 0; i < state->h_colnummod.size(); ++i) {
        int lr = state->h_colnummod[i];
        int got = statusmod[(size_t) lr * 2] +
                  statusmod[(size_t) lr * 2 + 1];
        reduce_missing += got != state->h_recv_cnt[(size_t) lr];
    }
    fprintf(stderr,
            "SymLDL literature forward trace rank=%d z=%d layer_npes=%d "
            "bcast_expected=%d bcast_missing=%d reduce_expected=%d "
            "reduce_missing=%d fmod_nonzero=%d\n",
            state->grid3d->iam, state->grid3d->zscp.Iam, state->nprocs,
            state->expected_bcast_recvs, bcast_missing,
            state->expected_reduce_recvs, reduce_missing, blocked);
    fflush(stderr);
    return blocked || bcast_missing || reduce_missing ? -1 : 0;
}

} /* namespace */

extern "C" int
dSymLDLLiteratureForwardAvailable(void)
{
#ifdef HAVE_NVSHMEM
    return 1;
#else
    return 0;
#endif
}

static unsigned long long
symldl_literature_panel_structure_hash(
    const int_t *lsub, int_t lsub_count)
{
    const unsigned long long offset = 1469598103934665603ULL;
    const unsigned long long prime = 1099511628211ULL;
    unsigned long long hash = offset;
    if (lsub == NULL || lsub_count <= 0)
        return 0;
    for (int_t i = 0; i < lsub_count; ++i) {
        unsigned long long value = (unsigned long long) lsub[i];
        for (size_t byte = 0; byte < sizeof(value); ++byte) {
            hash ^= (value >> (8 * byte)) & 0xffULL;
            hash *= prime;
        }
    }
    return hash;
}

/*
 * The 3D factorization leaves IN_GRID_ZERO panels as partial workspaces.
 * Algorithm 1 instead solves with L_z, whose ancestor panels are replicas of
 * the finalized panel on the unique IN_GRID_AIJ layer.  Refresh those existing
 * retained device panels in place before constructing the copied LU solver
 * view.  This adds no persistent factor storage.
 */
static void
symldl_literature_replicate_final_panels(
    int_t nsupers, int_t panel_count,
    const dSymLDLLiteraturePanelDesc *panels,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d)
{
    if (grid3d->zscp.Np <= 1)
        return;
    if (nsupers > INT_MAX)
        ABORT("SymLDL literature forward panel replication exceeds MPI count limits.");

    gridinfo_t *grid = &grid3d->grid2d;
    int mycol = MYCOL(grid->iam, grid);
    int myz = grid3d->zscp.Iam;
    std::vector<int> local_source((size_t) nsupers, INT_MAX);
    std::vector<int> source((size_t) nsupers, INT_MAX);
    std::vector<int> local_active((size_t) nsupers, 0);
    std::vector<int> active((size_t) nsupers, 0);
    std::vector<int_t> local_value_count((size_t) nsupers, -1);
    std::vector<int_t> value_count((size_t) nsupers, -1);
    std::vector<int_t> local_lsub_count((size_t) nsupers, -1);
    std::vector<int_t> lsub_count((size_t) nsupers, -1);
    std::vector<unsigned long long> local_lsub_hash(
        (size_t) nsupers, 0);
    std::vector<unsigned long long> lsub_hash((size_t) nsupers, 0);

    for (int_t k = 0; k < nsupers; ++k) {
        if (trf3Dpartition->superGridMap[k] == IN_GRID_AIJ) {
            local_source[(size_t) k] = myz;
            local_active[(size_t) k] = 1;
        }
        if (mycol != trf3Dpartition->symV2PanelRoot[k] ||
            trf3Dpartition->superGridMap[k] != IN_GRID_AIJ)
            continue;
        int_t lp = trf3Dpartition->symV2PanelLocalIndex[k];
        if (lp < 0 || lp >= panel_count || panels[lp].gid != k)
            ABORT("SymLDL literature forward is missing an active panel descriptor.");
        local_value_count[(size_t) k] = panels[lp].value_count;
        local_lsub_count[(size_t) k] = panels[lp].lsub_count;
        local_lsub_hash[(size_t) k] =
            symldl_literature_panel_structure_hash(
                panels[lp].lsub, panels[lp].lsub_count);
    }

    MPI_Allreduce(local_source.data(), source.data(), (int) nsupers,
                  MPI_INT, MPI_MIN, grid3d->zscp.comm);
    MPI_Allreduce(local_active.data(), active.data(), (int) nsupers,
                  MPI_INT, MPI_SUM, grid3d->zscp.comm);
    MPI_Allreduce(local_value_count.data(), value_count.data(),
                  (int) nsupers, mpi_int_t, MPI_MAX,
                  grid3d->zscp.comm);
    MPI_Allreduce(local_lsub_count.data(), lsub_count.data(),
                  (int) nsupers, mpi_int_t, MPI_MAX,
                  grid3d->zscp.comm);
    MPI_Allreduce(local_lsub_hash.data(), lsub_hash.data(),
                  (int) nsupers, MPI_UNSIGNED_LONG_LONG, MPI_BOR,
                  grid3d->zscp.comm);

    int_t max_values = 0;
    for (int_t k = 0; k < nsupers; ++k) {
        if (active[(size_t) k] != 1 || source[(size_t) k] == INT_MAX)
            ABORT("SymLDL literature forward requires one active Z panel owner.");
        if (mycol == trf3Dpartition->symV2PanelRoot[k]) {
            if (value_count[(size_t) k] < 0 ||
                lsub_count[(size_t) k] < 0)
                ABORT("SymLDL literature forward is missing source panel metadata.");
            max_values = std::max(max_values, value_count[(size_t) k]);
        }
    }

    std::vector<double> values((size_t) std::max((int_t) 1, max_values));
    for (int_t k = 0; k < nsupers; ++k) {
        if (mycol != trf3Dpartition->symV2PanelRoot[k])
            continue;
        int_t count = value_count[(size_t) k];
        if (count > INT_MAX)
            ABORT("SymLDL literature forward panel exceeds MPI count limits.");
        int_t lp = trf3Dpartition->symV2PanelLocalIndex[k];
        int local_panel =
            trf3Dpartition->superGridMap[k] != NOT_IN_GRID;

        if (local_panel) {
            if (lp < 0 || lp >= panel_count || panels[lp].gid != k)
                ABORT("SymLDL literature forward is missing a replicated panel descriptor.");
            if (panels[lp].value_count != count ||
                panels[lp].lsub_count != lsub_count[(size_t) k] ||
                symldl_literature_panel_structure_hash(
                    panels[lp].lsub, panels[lp].lsub_count) !=
                    lsub_hash[(size_t) k])
                ABORT("SymLDL literature forward replicated panel structure differs across Z.");
        } else if (lp >= 0) {
            ABORT("SymLDL literature forward has a panel outside its local forest.");
        }

        if (myz == source[(size_t) k] && count > 0) {
            if (!local_panel || panels[lp].host_values == NULL)
                ABORT("SymLDL literature forward source panel values are unavailable.");
            std::copy(panels[lp].host_values,
                      panels[lp].host_values + count, values.begin());
        }
        MPI_Bcast(values.data(), (int) count, MPI_DOUBLE,
                  source[(size_t) k], grid3d->zscp.comm);

        if (trf3Dpartition->superGridMap[k] == IN_GRID_ZERO && count > 0) {
            if (panels[lp].values == NULL)
                ABORT("SymLDL literature forward target panel values are unavailable.");
            CUDA_CHECK(cudaMemcpy(panels[lp].values, values.data(),
                                  (size_t) count * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
    }
}

extern "C" dSymLDLLiteratureForwardHandle
dSymLDLLiteratureForwardCreate(
    int_t n, int_t nsupers, int nrhs, int_t x_count, int_t lsum_count,
    int_t panel_count, const dSymLDLLiteraturePanelDesc *panels,
    const int_t *xsup, const int_t *ilsum,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d)
{
#ifndef HAVE_NVSHMEM
    (void) n; (void) nsupers; (void) nrhs; (void) x_count;
    (void) lsum_count; (void) panel_count; (void) panels; (void) xsup;
    (void) ilsum; (void) trf3Dpartition; (void) grid3d;
    return NULL;
#else
    if (n <= 0 || nsupers <= 0 || nrhs <= 0 || x_count <= 0 ||
        lsum_count <= 0 || panel_count < 0 ||
        (panel_count > 0 && panels == NULL) ||
        xsup == NULL || ilsum == NULL || trf3Dpartition == NULL ||
        grid3d == NULL || trf3Dpartition->symV2PanelLocalIndex == NULL ||
        trf3Dpartition->symV2RowLocalIndex == NULL ||
        (panel_count > 0 &&
         trf3Dpartition->symV2LocalPanelGids == NULL) ||
        (trf3Dpartition->symV2LocalRowCount > 0 &&
         trf3Dpartition->symV2LocalRowGids == NULL) ||
        trf3Dpartition->symV2DiagRoot == NULL ||
        trf3Dpartition->symV2PanelRoot == NULL ||
        trf3Dpartition->symV2DiagOwner == NULL ||
        trf3Dpartition->superGridMap == NULL)
        return NULL;

    double setup_start = SuperLU_timer_();
    dSymLDLLiteratureForwardState *state =
        new dSymLDLLiteratureForwardState();
    state->n = n;
    state->nsupers = nsupers;
    state->nrhs = nrhs;
    state->x_count = x_count;
    state->lsum_count = lsum_count;
    state->panel_count = panel_count;
    state->row_count = trf3Dpartition->symV2LocalRowCount;
    state->trf3Dpartition = trf3Dpartition;
    state->grid3d = grid3d;
    state->grid = &grid3d->grid2d;
    state->layer_comm = state->grid->comm;
    state->host_xsup = xsup;
    state->host_ilsum = ilsum;
    state->diagnostics = symldl_literature_env_enabled(
        "GPU3DV2_SYM_SOLVE_LITERATURE_FORWARD_DIAGNOSTICS");
    MPI_Comm_rank(state->layer_comm, &state->rank);
    MPI_Comm_size(state->layer_comm, &state->nprocs);

    if (panel_count != trf3Dpartition->symV2LocalPanelCount)
        ABORT("SymLDL literature forward panel count is inconsistent.");
    symldl_literature_replicate_final_panels(
        nsupers, panel_count, panels, trf3Dpartition, grid3d);
    int local_counts[2] = {(int) panel_count, (int) state->row_count};
    int max_counts[2] = {0, 0};
    MPI_Allreduce(local_counts, max_counts, 2, MPI_INT, MPI_MAX,
                  state->layer_comm);
    state->panel_slots = std::max(1, max_counts[0]);
    state->row_slots = std::max(1, max_counts[1]);
    for (int_t k = 0; k < nsupers; ++k)
        state->maxsup = std::max(state->maxsup, (int) SuperSize(k));
    state->maxrecvsz = state->maxsup * nrhs +
                       std::max((int) XK_H, (int) LSUM_H);

    if (panel_count > 0)
        state->h_panel_gids.assign(
            trf3Dpartition->symV2LocalPanelGids,
            trf3Dpartition->symV2LocalPanelGids + panel_count);
    if (state->row_count > 0)
        state->h_row_gids.assign(
            trf3Dpartition->symV2LocalRowGids,
            trf3Dpartition->symV2LocalRowGids + state->row_count);
    for (int_t lp = 0; lp < panel_count; ++lp)
        if (panels[lp].gid != state->h_panel_gids[(size_t) lp])
            ABORT("SymLDL literature forward panels are not in local-index order.");

    std::vector<C_Tree> bcast_trees;
    std::vector<C_Tree> reduce_trees;
    symldl_literature_build_bcast_trees(state, panels, bcast_trees);
    symldl_literature_build_reduce_trees(state, panels, reduce_trees);

    std::vector<long int> lsub_offsets((size_t) panel_count, -1);
    std::vector<long int> lloc_offsets((size_t) panel_count, -1);
    std::vector<int_t> lsub_data;
    std::vector<int_t> lloc_data;
    std::vector<double *> panel_values((size_t) panel_count, NULL);
    int panel_diagnostics = symldl_literature_env_enabled(
        "GPU3DV2_SYM_SOLVE_LITERATURE_PANEL_DIAGNOSTICS");
    for (int_t lp = 0; lp < panel_count; ++lp) {
        symldl_literature_validate_panel(state, panels[lp]);
        if (panels[lp].lsub == NULL)
            continue;
        if (panels[lp].lloc == NULL || panels[lp].values == NULL ||
            panels[lp].lsub_count <= 0 || panels[lp].lloc_count <= 0)
            ABORT("SymLDL literature forward panel storage is incomplete.");
        lsub_offsets[(size_t) lp] = (long int) lsub_data.size();
        lloc_offsets[(size_t) lp] = (long int) lloc_data.size();
        lsub_data.insert(lsub_data.end(), panels[lp].lsub,
                         panels[lp].lsub + panels[lp].lsub_count);
        lloc_data.insert(lloc_data.end(), panels[lp].lloc,
                         panels[lp].lloc + panels[lp].lloc_count);
        panel_values[(size_t) lp] = panels[lp].values;

        int_t k = panels[lp].gid;
        if (panel_diagnostics &&
            trf3Dpartition->supernode2treeMap[k] == 0) {
            std::vector<double> values((size_t) panels[lp].value_count);
            CUDA_CHECK(cudaMemcpy(values.data(), panels[lp].values,
                                  values.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            int myrow = MYROW(state->grid->iam, state->grid);
            int row_begin = myrow == trf3Dpartition->symV2DiagRoot[k]
                                ? symldl_literature_super_size(state, k)
                                : 0;
            int_t nsupr = panels[lp].lsub[1];
            int_t width = symldl_literature_super_size(state, k);
            double sum = 0.0;
            double sum_abs = 0.0;
            double sum_sq = 0.0;
            double max_abs = 0.0;
            long long count = 0;
            for (int_t col = 0; col < width; ++col) {
                for (int_t row = row_begin; row < nsupr; ++row) {
                    double value = values[(size_t) col * (size_t) nsupr +
                                          (size_t) row];
                    sum += value;
                    sum_abs += fabs(value);
                    sum_sq += value * value;
                    max_abs = std::max(max_abs, fabs(value));
                    ++count;
                }
            }
            fprintf(stderr,
                    "SymLDL literature root-panel trace: rank=%d z=%d "
                    "layer_rank=%d row=%d col=%d k=%lld map=%d "
                    "count=%lld sum=%.17e abs=%.17e norm=%.17e "
                    "max=%.17e\n",
                    grid3d->iam, grid3d->zscp.Iam, state->rank,
                    MYROW(state->grid->iam, state->grid),
                    MYCOL(state->grid->iam, state->grid), (long long) k,
                    (int) trf3Dpartition->superGridMap[k], count, sum,
                    sum_abs, sqrt(sum_sq), max_abs);
            fflush(stderr);
        }
    }

    if (!symldl_literature_nvshmem_initialized) {
        nv_init_wrapper(state->layer_comm);
        symldl_literature_nvshmem_initialized = 1;
    }
    if (nvshmem_my_pe() != state->rank ||
        nvshmem_n_pes() != state->nprocs ||
        state->nprocs != state->grid->nprow * state->grid->npcol) {
        if (state->rank == 0)
            fprintf(stderr,
                    "SymLDL literature forward requires layer-local NVSHMEM rank space.\n");
        symldl_literature_state_free(state);
        return NULL;
    }

    symldl_literature_cuda_alloc(&state->d_x, (size_t) x_count);
    symldl_literature_cuda_alloc(&state->d_lsum, (size_t) lsum_count);
    symldl_literature_cuda_copy(&state->d_fmod, state->h_fmod.data(),
                                state->h_fmod.size());
    symldl_literature_cuda_copy(&state->d_bcast_trees,
                                bcast_trees.data(), bcast_trees.size());
    symldl_literature_cuda_copy(&state->d_reduce_trees,
                                reduce_trees.data(), reduce_trees.size());
    symldl_literature_cuda_copy(&state->d_ilsum, ilsum,
                                (size_t) state->row_count + 1);
    symldl_literature_cuda_copy(&state->d_xsup, xsup,
                                (size_t) nsupers + 1);
    symldl_literature_cuda_copy(&state->d_lsub, lsub_data.data(),
                                lsub_data.size());
    symldl_literature_cuda_copy(&state->d_lsub_offsets,
                                lsub_offsets.data(), lsub_offsets.size());
    symldl_literature_cuda_copy(&state->d_lloc, lloc_data.data(),
                                lloc_data.size());
    symldl_literature_cuda_copy(&state->d_lloc_offsets,
                                lloc_offsets.data(), lloc_offsets.size());
    symldl_literature_cuda_copy(&state->d_panel_values,
                                panel_values.data(), panel_values.size());
    symldl_literature_cuda_copy(&state->d_panel_gids,
                                state->h_panel_gids.data(),
                                state->h_panel_gids.size());
    symldl_literature_cuda_copy(&state->d_row_gids,
                                state->h_row_gids.data(),
                                state->h_row_gids.size());
    symldl_literature_cuda_copy(
        &state->d_row_local_index,
        trf3Dpartition->symV2RowLocalIndex, (size_t) nsupers);
    symldl_literature_cuda_copy(&state->d_diag_roots,
                                trf3Dpartition->symV2DiagRoot,
                                (size_t) nsupers);
    symldl_literature_cuda_copy(&state->d_grid, state->grid, 1);

    state->h_nfrecv[0] = (int) state->h_colnum.size();
#ifdef _USE_SUMMIT
    state->h_nfrecv[1] = 32;
    state->h_nfrecv[2] = 8;
#else
    state->h_nfrecv[1] = 1024;
    state->h_nfrecv[2] = 2;
#endif
    state->h_nfrecvmod[0] = state->expected_reduce_recvs;
    state->h_nfrecvmod[1] = (int) state->h_colnummod.size();
    state->h_nfrecvmod[2] = state->h_nfrecv[2];
    state->h_nfrecvmod[3] = state->expected_reduce_sends;
    symldl_literature_cuda_copy(&state->d_nfrecv, state->h_nfrecv, 3);
    symldl_literature_cuda_copy(&state->d_status,
                                state->h_status.data(),
                                state->h_status.size());
    symldl_literature_cuda_copy(&state->d_colnum,
                                state->h_colnum.data(),
                                state->h_colnum.size());
    symldl_literature_cuda_alloc(&state->d_mynum, 1024);
    symldl_literature_cuda_alloc(&state->d_mymaskstart, 1024);
    symldl_literature_cuda_alloc(&state->d_mymasklength, 1024);
    symldl_literature_cuda_copy(&state->d_nfrecvmod,
                                state->h_nfrecvmod, 4);
    symldl_literature_cuda_copy(&state->d_statusmod,
                                state->h_statusmod.data(),
                                state->h_statusmod.size());
    symldl_literature_cuda_copy(&state->d_colnummod,
                                state->h_colnummod.data(),
                                state->h_colnummod.size());
    symldl_literature_cuda_alloc(&state->d_mynummod, 1024);
    symldl_literature_cuda_alloc(&state->d_mymaskstartmod, 1024);
    symldl_literature_cuda_alloc(&state->d_mymasklengthmod, 1024);
    symldl_literature_cuda_copy(&state->d_recv_cnt,
                                state->h_recv_cnt.data(),
                                state->h_recv_cnt.size());
    symldl_literature_cuda_alloc(&state->d_msgnum, 1024);
    symldl_literature_cuda_alloc(&state->d_flag_mod,
                                 (size_t) state->expected_reduce_sends + 1);

    size_t flag_bc_count = (size_t) RDMA_FLAG_SIZE *
                           ((size_t) state->panel_slots + 1);
    size_t flag_rd_count = (size_t) RDMA_FLAG_SIZE *
                           (size_t) state->row_slots * 2;
    size_t ready_x_count = (size_t) state->maxrecvsz *
                           (size_t) state->panel_slots;
    size_t ready_lsum_count = (size_t) 2 * (size_t) state->maxrecvsz *
                              (size_t) state->row_slots;
    state->flag_bc_q = (uint64_t *) nvshmem_malloc(
        flag_bc_count * sizeof(uint64_t));
    state->flag_rd_q = (uint64_t *) nvshmem_malloc(
        flag_rd_count * sizeof(uint64_t));
    state->ready_x = (double *) nvshmem_malloc(
        ready_x_count * sizeof(double));
    state->ready_lsum = (double *) nvshmem_malloc(
        ready_lsum_count * sizeof(double));
    state->my_flag_bc = (int *) nvshmem_malloc(
        flag_bc_count * sizeof(int));
    state->my_flag_rd = (int *) nvshmem_malloc(
        flag_rd_count * sizeof(int));
    if (state->flag_bc_q == NULL || state->flag_rd_q == NULL ||
        state->ready_x == NULL || state->ready_lsum == NULL ||
        state->my_flag_bc == NULL || state->my_flag_rd == NULL) {
        symldl_literature_state_free(state);
        return NULL;
    }
    CUDA_CHECK(cudaMemset(state->my_flag_bc, 0,
                          flag_bc_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(state->my_flag_rd, 0,
                          flag_rd_count * sizeof(int)));
    state->setup_time = SuperLU_timer_() - setup_start;
    return (dSymLDLLiteratureForwardHandle) state;
#endif
}

extern "C" int
dSymLDLLiteratureForwardInitializeRHS(
    dSymLDLLiteratureForwardHandle handle, double *x, int_t x_count)
{
    dSymLDLLiteratureForwardState *state =
        (dSymLDLLiteratureForwardState *) handle;
    if (state == NULL || x == NULL || x_count != state->x_count)
        return -1;

    int myrow = MYROW(state->grid->iam, state->grid);
    int mycol = MYCOL(state->grid->iam, state->grid);
    std::vector<double> x_source((size_t) x_count, 0.0);
    std::vector<double> block(
        (size_t) state->maxsup * (size_t) state->nrhs + XK_H);

    /*
     * This is the selected-owner equivalent of dtrs_B_init3d_newsolve():
     * first preserve and zero every local replica, then restore RHS blocks
     * by the forest levels active on this Z layer.
     */
    for (int_t k = 0; k < state->nsupers; ++k) {
        if (myrow != state->trf3Dpartition->symV2DiagRoot[k] ||
            mycol != state->trf3Dpartition->symV2PanelRoot[k])
            continue;
        int candidate = state->grid3d->iam ==
                        state->trf3Dpartition->symV2DiagOwner[k]
                            ? state->grid3d->zscp.Iam : INT_MAX;
        int source_z = INT_MAX;
        MPI_Allreduce(&candidate, &source_z, 1, MPI_INT, MPI_MIN,
                      state->grid3d->zscp.comm);
        if (source_z == INT_MAX)
            return -1;
        int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
        int_t width = symldl_literature_super_size(state, k);
        int count = (int) (width * state->nrhs + XK_H);
        if (state->grid3d->zscp.Iam == source_z) {
            if (lr < 0)
                return -1;
            int_t ii = symldl_literature_x_block(state, lr);
            std::copy(&x[ii - XK_H], &x[ii - XK_H] + count,
                      block.begin());
        }
        MPI_Bcast(block.data(), count, MPI_DOUBLE, source_z,
                  state->grid3d->zscp.comm);
        if (lr >= 0) {
            int_t ii = symldl_literature_x_block(state, lr);
            std::copy(block.begin(), block.begin() + count,
                      &x_source[ii - XK_H]);
            std::copy(block.begin(), block.begin() + XK_H, &x[ii - XK_H]);
            std::fill(&x[ii], &x[ii] + width * state->nrhs, 0.0);
        }
    }

    int_t max_level = log2i(state->grid3d->zscp.Np) + 1;
    int_t *tree_ids = state->trf3Dpartition->myTreeIdxs;
    int_t *zero_ids = state->trf3Dpartition->myZeroTrIdxs;
    for (int_t level = 0; level < max_level; ++level) {
        if (zero_ids[level])
            continue;
        sForest_t *forest = state->trf3Dpartition->sForests[tree_ids[level]];
        if (forest == NULL)
            continue;
        for (int_t pos = 0; pos < forest->nNodes; ++pos) {
            int_t k = forest->nodeList[pos];
            if (myrow != state->trf3Dpartition->symV2DiagRoot[k] ||
                mycol != state->trf3Dpartition->symV2PanelRoot[k])
                continue;
            int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
            if (lr < 0)
                ABORT("SymLDL literature forward RHS forest is missing a local row.");
            int_t ii = symldl_literature_x_block(state, lr);
            int_t count = symldl_literature_super_size(state, k) *
                          state->nrhs;
            std::copy(&x_source[ii], &x_source[ii] + count, &x[ii]);
        }
    }
    return 0;
}

extern "C" int
dSymLDLLiteratureForwardSolve(
    dSymLDLLiteratureForwardHandle handle, double *x, int_t x_count)
{
    dSymLDLLiteratureForwardState *state =
        (dSymLDLLiteratureForwardState *) handle;
    if (state == NULL || x == NULL || x_count != state->x_count)
        return -1;
#ifndef HAVE_NVSHMEM
    return -1;
#else
    double phase_start = SuperLU_timer_();
    double copy_start = SuperLU_timer_();
    CUDA_CHECK(cudaMemcpy(state->d_x, x,
                          (size_t) x_count * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state->d_lsum, 0,
                          (size_t) state->lsum_count * sizeof(double)));
    if (!state->h_fmod.empty())
        CUDA_CHECK(cudaMemcpy(state->d_fmod, state->h_fmod.data(),
                              state->h_fmod.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    if (!state->h_status.empty())
        CUDA_CHECK(cudaMemcpy(state->d_status, state->h_status.data(),
                              state->h_status.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    if (!state->h_statusmod.empty())
        CUDA_CHECK(cudaMemcpy(state->d_statusmod,
                              state->h_statusmod.data(),
                              state->h_statusmod.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state->flag_bc_q, 0,
                          (size_t) RDMA_FLAG_SIZE *
                          ((size_t) state->panel_slots + 1) *
                          sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(state->flag_rd_q, 0,
                          (size_t) RDMA_FLAG_SIZE *
                          (size_t) state->row_slots * 2 *
                          sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(state->ready_x, 0,
                          (size_t) state->maxrecvsz *
                          (size_t) state->panel_slots * sizeof(double)));
    CUDA_CHECK(cudaMemset(state->ready_lsum, 0,
                          2 * (size_t) state->maxrecvsz *
                          (size_t) state->row_slots * sizeof(double)));
    CUDA_CHECK(cudaMemset(state->d_msgnum, 0,
                          (size_t) state->h_nfrecv[1] * sizeof(int)));
    state->h2d_time += SuperLU_timer_() - copy_start;

    MPI_Barrier(state->layer_comm);
    if (symldl_literature_forward_wrap(
            (int) state->panel_count, (int) state->row_count,
            state->d_lsum, state->d_x, state->nrhs, state->maxsup,
            state->nsupers, state->d_fmod, state->d_bcast_trees,
            state->d_reduce_trees, state->d_ilsum, state->d_lsub,
            state->d_lsub_offsets, state->d_panel_values, state->d_lloc,
            state->d_lloc_offsets, state->d_xsup, state->d_grid,
            state->d_panel_gids, state->d_row_gids,
            state->d_row_local_index, state->d_diag_roots,
            state->maxrecvsz, state->flag_bc_q, state->flag_rd_q,
            state->ready_x, state->ready_lsum, state->my_flag_bc,
            state->my_flag_rd, state->d_nfrecv, state->h_nfrecv,
            state->d_status, state->d_colnum, state->d_mynum,
            state->d_mymaskstart, state->d_mymasklength,
            state->d_nfrecvmod, state->d_statusmod,
            state->d_colnummod, state->d_mynummod,
            state->d_mymaskstartmod, state->d_mymasklengthmod,
            state->d_recv_cnt, state->d_msgnum, state->d_flag_mod) != 0)
        return -1;

    copy_start = SuperLU_timer_();
    CUDA_CHECK(cudaMemcpy(x, state->d_x,
                          (size_t) x_count * sizeof(double),
                          cudaMemcpyDeviceToHost));
    state->d2h_time += SuperLU_timer_() - copy_start;
    state->forward_time += SuperLU_timer_() - phase_start;
    return symldl_literature_check_forward(state);
#endif
}

extern "C" int
dSymLDLLiteratureForwardSparseAllreduce(
    dSymLDLLiteratureForwardHandle handle, double *x, int_t x_count)
{
    dSymLDLLiteratureForwardState *state =
        (dSymLDLLiteratureForwardState *) handle;
    if (state == NULL || x == NULL || x_count != state->x_count ||
        state->sparse_reduce_calls != 0 ||
        state->sparse_broadcast_calls != 0)
        return -1;

    std::vector<double> recvbuf(
        (size_t) state->maxsup * (size_t) state->nrhs);
    int_t max_level = log2i(state->grid3d->zscp.Np) + 1;
    int_t myz = state->grid3d->zscp.Iam;
    int_t *tree_ids = state->trf3Dpartition->myTreeIdxs;
    int_t *zero_ids = state->trf3Dpartition->myZeroTrIdxs;

    if (state->diagnostics) {
        double local_sum = 0.0;
        double local_abs = 0.0;
        double local_sq = 0.0;
        double local_max = 0.0;
        long long local_count = 0;
        int local_nonfinite = 0;
        int myrow = MYROW(state->grid->iam, state->grid);
        int mycol = MYCOL(state->grid->iam, state->grid);
        for (int_t k = 0; k < state->nsupers; ++k) {
            if (myrow != state->trf3Dpartition->symV2DiagRoot[k] ||
                mycol != state->trf3Dpartition->symV2PanelRoot[k])
                continue;
            int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
            if (lr < 0)
                continue;
            int_t ii = symldl_literature_x_block(state, lr);
            int_t count = symldl_literature_super_size(state, k) *
                          state->nrhs;
            for (int_t i = 0; i < count; ++i) {
                double value = x[ii + i];
                if (!isfinite(value)) {
                    ++local_nonfinite;
                    continue;
                }
                local_sum += value;
                local_abs += fabs(value);
                local_sq += value * value;
                local_max = std::max(local_max, fabs(value));
                ++local_count;
            }
        }
        double layer_sum = 0.0;
        double layer_abs = 0.0;
        double layer_sq = 0.0;
        double layer_max = 0.0;
        long long layer_count = 0;
        int layer_nonfinite = 0;
        MPI_Reduce(&local_sum, &layer_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
                   state->layer_comm);
        MPI_Reduce(&local_abs, &layer_abs, 1, MPI_DOUBLE, MPI_SUM, 0,
                   state->layer_comm);
        MPI_Reduce(&local_sq, &layer_sq, 1, MPI_DOUBLE, MPI_SUM, 0,
                   state->layer_comm);
        MPI_Reduce(&local_max, &layer_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                   state->layer_comm);
        MPI_Reduce(&local_count, &layer_count, 1, MPI_LONG_LONG, MPI_SUM, 0,
                   state->layer_comm);
        MPI_Reduce(&local_nonfinite, &layer_nonfinite, 1, MPI_INT, MPI_SUM,
                   0, state->layer_comm);
        if (state->rank == 0) {
            fprintf(stderr,
                    "SymLDL literature pre-sparse trace: z=%d count=%lld "
                    "nonfinite=%d sum=%.17e abs=%.17e norm=%.17e "
                    "max=%.17e\n",
                    state->grid3d->zscp.Iam, layer_count, layer_nonfinite,
                    layer_sum, layer_abs, sqrt(layer_sq), layer_max);
            fflush(stderr);
        }
    }

    double start = SuperLU_timer_();
    ++state->sparse_reduce_calls;
    for (int_t level = 1; level < max_level; ++level) {
        if (zero_ids[level - 1])
            continue;
        int_t sender;
        int_t receiver;
        int_t tree = tree_ids[level];
        if ((myz % (1 << level)) == 0) {
            sender = myz + (1 << (level - 1));
            receiver = myz;
        } else {
            sender = myz;
            receiver = myz - (1 << (level - 1));
        }
        int_t current = tree;
        for (int_t ancestor = level; ancestor < max_level; ++ancestor) {
            symldl_literature_z_reduce_tree(
                state, current, sender, receiver, x, recvbuf.data());
            current = (current + 1) / 2 - 1;
        }
    }
    state->sparse_reduce_time += SuperLU_timer_() - start;

    start = SuperLU_timer_();
    ++state->sparse_broadcast_calls;
    for (int_t level = max_level - 1; level > 0; --level) {
        if (zero_ids[level - 1])
            continue;
        int_t sender;
        int_t receiver;
        int_t tree = tree_ids[level];
        if ((myz % (1 << level)) == 0) {
            sender = myz;
            receiver = myz + (1 << (level - 1));
        } else {
            sender = myz - (1 << (level - 1));
            receiver = myz;
        }
        int_t current = tree;
        for (int_t ancestor = level; ancestor < max_level; ++ancestor) {
            symldl_literature_z_bcast_tree(
                state, current, sender, receiver, x);
            current = (current + 1) / 2 - 1;
        }
    }
    state->sparse_broadcast_time += SuperLU_timer_() - start;

    if (state->diagnostics) {
        double local_replica_error = 0.0;
        int myrow = MYROW(state->grid->iam, state->grid);
        int mycol = MYCOL(state->grid->iam, state->grid);
        for (int_t k = 0; k < state->nsupers; ++k) {
            if (myrow != state->trf3Dpartition->symV2DiagRoot[k] ||
                mycol != state->trf3Dpartition->symV2PanelRoot[k])
                continue;
            int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
            int active = lr >= 0;
            int active_count = 0;
            MPI_Allreduce(&active, &active_count, 1, MPI_INT, MPI_SUM,
                          state->grid3d->zscp.comm);
            for (int rhs = 0; rhs < state->nrhs; ++rhs) {
                int_t width = symldl_literature_super_size(state, k);
                for (int_t i = 0; i < width; ++i) {
                    double value = active
                        ? x[symldl_literature_x_block(state, lr) + i +
                            (int_t) rhs * width]
                        : 0.0;
                    double sum = 0.0;
                    double max_value = -DBL_MAX;
                    double min_value = DBL_MAX;
                    double selected_max = active ? value : -DBL_MAX;
                    double selected_min = active ? value : DBL_MAX;
                    MPI_Allreduce(&value, &sum, 1, MPI_DOUBLE, MPI_SUM,
                                  state->grid3d->zscp.comm);
                    MPI_Allreduce(&selected_max, &max_value, 1, MPI_DOUBLE,
                                  MPI_MAX, state->grid3d->zscp.comm);
                    MPI_Allreduce(&selected_min, &min_value, 1, MPI_DOUBLE,
                                  MPI_MIN, state->grid3d->zscp.comm);
                    (void) sum;
                    if (active_count > 1)
                        local_replica_error = std::max(
                            local_replica_error, fabs(max_value - min_value));
                }
            }
        }
        double global_replica_error = 0.0;
        MPI_Allreduce(&local_replica_error, &global_replica_error, 1,
                      MPI_DOUBLE, MPI_MAX, state->grid3d->comm);
        if (state->grid3d->iam == 0) {
            fprintf(stderr,
                    "SymLDL literature sparse-Z trace: reduce_calls=%d "
                    "broadcast_calls=%d replica_error=%.17e\n",
                    state->sparse_reduce_calls,
                    state->sparse_broadcast_calls,
                    global_replica_error);
            fflush(stderr);
        }
        if (global_replica_error != 0.0)
            return -1;
    }
    return 0;
}

extern "C" void
dSymLDLLiteratureForwardTakeTimers(
    dSymLDLLiteratureForwardHandle handle, double *setup, double *forward,
    double *sparse_reduce, double *sparse_broadcast, double *h2d,
    double *d2h)
{
    dSymLDLLiteratureForwardState *state =
        (dSymLDLLiteratureForwardState *) handle;
    if (setup) *setup = state ? state->setup_time : 0.0;
    if (forward) *forward = state ? state->forward_time : 0.0;
    if (sparse_reduce)
        *sparse_reduce = state ? state->sparse_reduce_time : 0.0;
    if (sparse_broadcast)
        *sparse_broadcast = state ? state->sparse_broadcast_time : 0.0;
    if (h2d) *h2d = state ? state->h2d_time : 0.0;
    if (d2h) *d2h = state ? state->d2h_time : 0.0;
}

extern "C" void
dSymLDLLiteratureForwardDestroy(
    dSymLDLLiteratureForwardHandle handle)
{
    symldl_literature_state_free(
        (dSymLDLLiteratureForwardState *) handle);
}
