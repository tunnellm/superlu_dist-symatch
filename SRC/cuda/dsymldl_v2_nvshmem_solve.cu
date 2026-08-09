/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief NVSHMEM 3D L-D-L^T solve for native SymLDL.
 *
 * The forward WAIT/SOLVE kernels and two-stream launch structure follow
 * pdgstrs_lsum_cuda.cu. The backward phase traverses the shared factor graph
 * in reverse and applies retained L blocks as transposed. SymLDL-specific
 * storage and ownership adapters stay outside the execution protocol.
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
#include <unistd.h>

#include "dsymldl_v2_nvshmem_solve.h"

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
            ABORT("SymLDL NVSHMEM 3D forward CUDA operation failed.");        \
        }                                                                       \
    } while (0)
#endif


/******************************************************************************/
static __device__
void symldl_nvshmem_gemm_device(
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
__device__ void symldl_nvshmem_bcast_forward_device(C_Tree* tree,  volatile uint64_t* flag_bc_q,  int* my_flag_bc, int mype, int tid,double* dready_x, int maxrecvsz){
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
__device__ void symldl_nvshmem_reduce_forward_device(C_Tree* Tree, volatile uint64_t* flag_rd_q, int* my_flag_rd, int mype, int bid, int tid, double* dready_lsum, int maxrecvsz, int myroot){
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
__device__ void symldl_nvshmem_wait_bcrd_device
        (
                int wait_bid,
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
    int bid = wait_bid;
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

                                   //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d, tmp_sum=%lf\n", mype, bid, tid, lib, k,
                                   //       tmp_sum,fmod_tmp, tmp_sum);
                                   //printf("sum2-(%d,%d,%d),lib=%d,k=%d,fmod_tmp=%d\n", mype, bid, tid, lib, k,fmod_tmp);
                               }
                               // One dependency is consumed per reduction message.
                               __threadfence();
                               fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
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
                           __threadfence();
                           fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                           //printf("sum1-(%d,%d,%d),lib=%d,k=%d,fmod_tmp=%d\n", mype, bid, tid, lib, k,fmod_tmp);
                           //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,fmod_tmp);
                       }

                       if (fmod_tmp == 1) {// forward RD
                           __threadfence();
                           //senddone[lk]=1;
                           if (LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_) {
                               //cnt=LRtree_ptr[lib].msgSize_;
                               my_flag_rd[lib * RDMA_FLAG_SIZE] = lib;
                               my_flag_rd[lib * RDMA_FLAG_SIZE + 1] =
                                   LRtree_ptr[lib].msgSize_ * nrhs;
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
                               symldl_nvshmem_reduce_forward_device(&LRtree_ptr[lib], flag_rd_q,
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

                               //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k,tmp_sum,fmod_tmp);
                           }
                           // One dependency is consumed per reduction message.
                           __threadfence();
                           fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
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
                       __threadfence();
                       fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                       //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,fmod_tmp);
                   }

                   if (fmod_tmp == 1) {// forward RD
                       __threadfence();
                       //printf("sum1-(%d,%d,%d),lib=%d, myRoot=%d\n", mype, bid, tid, lib,LRtree_ptr[lib].myRoot_);
                       if (LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_) {
                           my_flag_rd[lib * RDMA_FLAG_SIZE] = lib;
                           my_flag_rd[lib * RDMA_FLAG_SIZE + 1] =
                               LRtree_ptr[lib].msgSize_ * nrhs;
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
                           symldl_nvshmem_reduce_forward_device(&LRtree_ptr[lib], flag_rd_q,
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

__global__ void symldl_nvshmem_wait_bcrd
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
                double *lsum,
                int *fmod,
                gridinfo_t *grid,
                int_t *xsup,
                int_t *ilsum,
                int_t *row_gids,
                int nbrow_loc,
                int_t nsupers
        )
{
    symldl_nvshmem_wait_bcrd_device(
        blockIdx.x, nrhs, LRtree_ptr, maxrecvsz, mype, flag_bc_q,
        flag_rd_q, dready_x, dready_lsum, my_flag_bc, my_flag_rd,
        d_nfrecv, d_status, d_colnum, d_mynum, d_mymaskstart,
        d_mymasklength, d_nfrecvmod, d_statusmod, d_colnummod,
        d_mynummod, d_mymaskstartmod, d_mymasklengthmod, d_recv_cnt,
        d_msgnum, d_flag_mod, lsum, fmod, grid, xsup, ilsum, row_gids,
        nbrow_loc, nsupers);
}

__global__ void symldl_nvshmem_forward_kernel
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
                const dSymLDLNVPanelDesc *panels,
                const dSymLDLNVBlockDesc *blocks,
                const int_t *rows,
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
    int    iam, iknsupc, myrow, krow, nbrow1, nsupr;
    int_t  k,i, l,ii, ik, il, irow, j, lb, lk, rel, lib;
    int_t  fmod_tmp;
    //__shared__ double rtemp_loc[128];
    double temp1;
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

    const dSymLDLNVPanelDesc panel = panels[bid];
    if (!panel.active || panel.values == NULL) {
        return;
    }

    int gc;

    lk = bid;
    iam = grid->iam;
    myrow = MYROW(iam, grid);
    gc = panel_gids[lk];
    if (gc < 0 || gc >= nsupers) return;
    k = gc;

    knsupc = panel.width;
    iam = grid->iam;
    krow = diag_roots[k];
    lusup = panel.values;
    nsupr = panel.nsupr;

    nlb = panel.block_count;

    if (myrow == krow) {   /* Unit-lower diagonal row completes and forwards x_k. */

        if (tid == 0) {  /*only the first thread in a block handles the lock */
            //printf("(%d) iam bid=%d,enter solve--2, wait lock,gc=%d\n",mype,bid,gc);
            //printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i],myrow,krow);
            // for (i=0 ; i<maxsup ; i++){
            // rtemp_loc[i]=0.0;
            // }

            lib = row_local_index[k];
            do {
                tmp = atomicAdd(&fmod[lib * aln_i], 0);
            } while (tmp > 0);
            __threadfence();
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
        symldl_nvshmem_bcast_forward_device(&LBtree_ptr[lk], flag_bc_q, &my_flag_bc[lk * RDMA_FLAG_SIZE],
                                             mype, tid, &dready_x[0], maxrecvsz);
        //printf("(%d,%d,%d), lk=%d, gc=%d\n",mype,bid,tid,lk,gc);
        //symldl_nvshmem_bcast_forward_device(&LBtree_ptr[lk],&dready_x[maxrecvsz*lk],cnt*nrhs+XK_H);
    }
    int keep_lk = lk;
    __syncthreads();

    if (nlb > 0) {
        int_t first_block = panel.block_begin;
        __shared__ int send_reduce;
        if (nrhs == 1) {
            int_t first_luptr = blocks[first_block].luptr;
            int_t update_rows = 0;
            for (lb = 0; lb < nlb; ++lb)
                update_rows += blocks[first_block + lb].nbrow;

            for (i = tid; i < update_rows; i += block_size) {
                int_t block_row = 0;
                lb = 0;
                while (block_row + blocks[first_block + lb].nbrow <= i) {
                    block_row += blocks[first_block + lb].nbrow;
                    ++lb;
                }
                const dSymLDLNVBlockDesc block =
                    blocks[first_block + lb];
                ik = block.target_gid;
                lk = row_local_index[ik];
                if (lk < 0)
                    return;
                iknsupc = SuperSize(ik);
                il = LSUM_BLK(lk);
                rel = xsup[ik];
                irow = rows[block.row_begin + i - block_row] - rel;
                temp1 = zero;
                for (l = 0; l < knsupc; ++l)
                    temp1 += lusup[first_luptr + i + l * nsupr] *
                             dready_x[l + maxrecvsz * keep_lk];
                d_atomicAdd(&lsum[il + irow], -temp1);
            }
            __syncthreads();

            for (i = tid; i < update_rows; i += block_size) {
                int_t block_row = 0;
                lb = 0;
                while (block_row + blocks[first_block + lb].nbrow <= i) {
                    block_row += blocks[first_block + lb].nbrow;
                    ++lb;
                }
                const dSymLDLNVBlockDesc block =
                    blocks[first_block + lb];
                if (i != block_row + block.nbrow - 1)
                    continue;
                ik = block.target_gid;
                lk = row_local_index[ik];
                if (lk < 0)
                    return;
                iknsupc = SuperSize(ik);
                il = LSUM_BLK(lk);
                __threadfence();
                fmod_tmp = atomicSub(&fmod[lk * aln_i], 1);
                if (fmod_tmp == 1 &&
                    LRtree_ptr[lk].myRoot_ != LRtree_ptr[lk].myRank_) {
                    my_flag_rd[lk * RDMA_FLAG_SIZE] = lk;
                    my_flag_rd[lk * RDMA_FLAG_SIZE + 1] =
                        LRtree_ptr[lk].msgSize_ * nrhs;
                    for (int aab = 0; aab < iknsupc; ++aab)
                        dready_lsum[lk * maxrecvsz * 2 + aab] =
                            lsum[il + aab];
                    symldl_nvshmem_reduce_forward_device(
                        &LRtree_ptr[lk], flag_rd_q,
                        &my_flag_rd[RDMA_FLAG_SIZE * lk], mype, bid, tid,
                        &dready_lsum[0], maxrecvsz,
                        LRtree_ptr[lk].myRoot_);
                }
            }
        } else {
            for (lb = 0; lb < nlb; ++lb) {
                const dSymLDLNVBlockDesc block =
                    blocks[first_block + lb];
                ik = block.target_gid;
                lk = row_local_index[ik];
                if (lk < 0)
                    return;
                iknsupc = SuperSize(ik);
                il = LSUM_BLK(lk);
                rel = xsup[ik];
                nbrow1 = block.nbrow;
                for (int blx = 0; blx * BLK_M < nbrow1; ++blx) {
                    for (int bly = 0; bly * BLK_N < nrhs; ++bly) {
                        symldl_nvshmem_gemm_device(
                            nbrow1, nrhs, knsupc, blx, bly,
                            &lusup[block.luptr], nsupr,
                            &dready_x[maxrecvsz * keep_lk], knsupc,
                            rC, alpha, beta);
#pragma unroll
                        for (ni = 0; ni < THR_N; ++ni) {
                            int coord_dCn = bly * BLK_N + ni * DIM_Y + idy;
#pragma unroll
                            for (mi = 0; mi < THR_M; ++mi) {
                                int coord_dCm = blx * BLK_M + mi * DIM_X + idx;
                                if (coord_dCm < nbrow1 && coord_dCn < nrhs) {
                                    irow = rows[block.row_begin + coord_dCm] - rel;
                                    d_atomicAdd(
                                        &lsum[il + irow +
                                              coord_dCn * iknsupc],
                                        -rC[ni][mi]);
                                }
                            }
                        }
                    }
                }
                __syncthreads();
                if (tid == 0) {
                    __threadfence();
                    fmod_tmp = atomicSub(&fmod[lk * aln_i], 1);
                    send_reduce = fmod_tmp == 1 &&
                        LRtree_ptr[lk].myRoot_ != LRtree_ptr[lk].myRank_;
                }
                __syncthreads();
                if (send_reduce) {
                    RHS_ITERATE(j)
                        for (i = tid; i < iknsupc; i += block_size)
                            dready_lsum[lk * maxrecvsz * 2 + i +
                                        j * iknsupc] =
                                lsum[il + i + j * iknsupc];
                    __syncthreads();
                    if (tid == 0) {
                        my_flag_rd[lk * RDMA_FLAG_SIZE] = lk;
                        my_flag_rd[lk * RDMA_FLAG_SIZE + 1] =
                            LRtree_ptr[lk].msgSize_ * nrhs;
                        symldl_nvshmem_reduce_forward_device(
                            &LRtree_ptr[lk], flag_rd_q,
                            &my_flag_rd[RDMA_FLAG_SIZE * lk], mype, bid,
                            tid, &dready_lsum[0], maxrecvsz,
                            LRtree_ptr[lk].myRoot_);
                    }
                }
                __syncthreads();
            }
        }
    } /* if nlb>0*/

} /* symldl_nvshmem_forward_kernel */

static int
symldl_nvshmem_forward_wrap(
    int nbcol_loc, int nbrow_loc, double *lsum, double *x, int nrhs,
    int maxsup, int_t nsupers, int *fmod, C_Tree *LBtree_ptr,
    C_Tree *LRtree_ptr, int_t *ilsum,
    const dSymLDLNVPanelDesc *panels,
    const dSymLDLNVBlockDesc *blocks, const int_t *rows,
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
    (void) panels; (void) blocks; (void) rows; (void) xsup; (void) grid;
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
                                     symldl_nvshmem_forward_kernel));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize,
                                  cuattr.localSizeBytes));

    int min_grid_size = 0;
    int wait_threads = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &wait_threads,
        (const void *) symldl_nvshmem_wait_bcrd, 0, 0));
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
        (const void *) symldl_nvshmem_wait_bcrd, wait_grid, wait_block,
        args, 0, stream[0]);
    if (status != NVSHMEMX_SUCCESS) {
        for (int i = 0; i < 2; ++i)
            CUDA_CHECK(cudaStreamDestroy(stream[i]));
        return -1;
    }

    if (nbcol_loc > 0) {
        dim3 solve_grid(nbcol_loc);
        symldl_nvshmem_forward_kernel<<<solve_grid, solve_block, 0,
                                           stream[1]>>>(
            nbcol_loc, nblock_ex, lsum, x, nrhs, maxsup, nsupers, fmod,
            LBtree_ptr, LRtree_ptr, ilsum, panels, blocks, rows,
            xsup, grid, panel_gids, row_gids,
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

/*
 * Backward elimination uses the forward solve's WAIT/SOLVE protocol with the
 * process-grid directions exchanged.  A source row i broadcasts x_i across
 * its process row; retained blocks L(i,k) then accumulate -L(i,k)^T x_i into
 * panel k, and those partial sums reduce down panel k's process column.
 */
__global__ void symldl_nvshmem_diagonal_kernel(
    int panel_count, int nrhs,
    const dSymLDLNVPanelDesc *panels,
    const int_t *row_local_index, const int_t *ilsum,
    const double *input, double *x)
{
    int lp = blockIdx.x;
    if (lp >= panel_count)
        return;
    const dSymLDLNVPanelDesc panel = panels[lp];
    if (!panel.active || !panel.has_diag || panel.values == NULL)
        return;
    int_t lr = row_local_index[panel.gid];
    if (lr < 0)
        return;
    int_t offset = X_BLK(lr);
    int count = panel.width * nrhs;
    for (int out = threadIdx.x; out < count; out += blockDim.x) {
        int rhs = out / panel.width;
        int row = out - rhs * panel.width;
        double value = 0.0;
        for (int col = 0; col < panel.width; ++col)
            value += panel.values[panel.diag_luptr + row +
                                  col * panel.nsupr] *
                     input[offset + col + rhs * panel.width];
        x[offset + row + rhs * panel.width] = value;
    }
}

static __device__ __forceinline__ void
symldl_nvshmem_backward_reduce_send(
    int target_slot, int width, int nrhs, int_t target_sum,
    int maxrecvsz, C_Tree *reduce_trees, volatile uint64_t *flag_rd_q,
    double *dready_lsum, const double *lsum)
{
#ifdef HAVE_NVSHMEM
    C_Tree *tree = reduce_trees + target_slot;
    if (tree->empty_ == YES || tree->myRoot_ == tree->myRank_)
        return;
    int count = width * nrhs;
    int source = target_slot * maxrecvsz * 2;
    int slot = tree->myIdx & 1;
    int destination = source + (slot ? maxrecvsz : 0);
    for (int i = 0; i < count; ++i)
        dready_lsum[source + i] = lsum[target_sum + i];
    nvshmem_double_put_signal_nbi(
        dready_lsum + destination, dready_lsum + source, count,
        (uint64_t *) flag_rd_q + 2 * target_slot + slot, 1,
        NVSHMEM_SIGNAL_SET,
        tree->myRoot_);
#else
    (void) target_slot; (void) width; (void) nrhs; (void) target_sum;
    (void) maxrecvsz; (void) reduce_trees; (void) flag_rd_q;
    (void) dready_lsum; (void) lsum;
#endif
}

__device__ void symldl_nvshmem_backward_source(
    int source_slot, int worker_id, double *lsum, double *x, int nrhs,
    int_t nsupers, int *bmod, C_Tree *bcast_trees,
    C_Tree *reduce_trees, const int_t *row_ilsum,
    const int_t *target_ilsum,
    const dSymLDLNVPanelDesc *panels,
    const dSymLDLNVBlockDesc *blocks, const int_t *rows,
    const int_t *source_edge_offsets, const int_t *source_edge_ids,
    const int *source_active, int_t *xsup, gridinfo_t *grid,
    int_t *source_gids, int_t *row_local_index,
    int_t *panel_local_index, int *panel_roots, int_t maxrecvsz,
    int mype, volatile uint64_t *flag_bc_q,
    volatile uint64_t *flag_rd_q, double *dready_x,
    double *dready_lsum, int *my_flag_bc, int *my_flag_rd)
{
#ifdef HAVE_NVSHMEM
    if (source_slot < 0 || !source_active[source_slot])
        return;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_size = blockDim.x * blockDim.y;
    int_t source = source_gids[source_slot];
    if (source < 0 || source >= nsupers)
        return;
    int mycol = MYCOL(grid->iam, grid);
    int_t source_width = SuperSize(source);
    int_t source_row = row_local_index[source];
    int_t source_x = source_row >= 0
                         ? row_ilsum[source_row] * nrhs +
                               (source_row + 1) * XK_H
                         : -1;
    int owner = mycol == panel_roots[source];

    if (owner) {
        int_t target_slot = panel_local_index[source];
        if (source_row < 0 || target_slot < 0)
            return;
        if (tid == 0) {
            while (atomicAdd(&bmod[target_slot], 0) > 0) {
                __threadfence();
                __nanosleep(64);
            }
        }
        __syncthreads();
        /* The final dependency may have been completed by another CTA. */
        __threadfence();
        __syncthreads();
        int_t target_sum = target_ilsum[target_slot] * nrhs +
                           (target_slot + 1) * LSUM_H;
        for (int rhs = 0; rhs < nrhs; ++rhs)
            for (int_t row = tid; row < source_width; row += block_size)
                x[source_x + row + (int_t) rhs * source_width] +=
                    lsum[target_sum + row + (int_t) rhs * source_width];
        __syncthreads();
        for (int rhs = 0; rhs < nrhs; ++rhs)
            for (int_t row = tid; row < source_width; row += block_size)
                dready_x[(int_t) source_slot * maxrecvsz + row +
                         (int_t) rhs * source_width] =
                    x[source_x + row + (int_t) rhs * source_width];
        __syncthreads();
    } else {
        if (tid == 0) {
            volatile uint64_t arrived = 0;
            do {
                arrived = flag_bc_q[source_slot];
                __threadfence();
            } while (arrived != 1);
        }
        __syncthreads();
    }

    if (bcast_trees[source_slot].destCnt_ > 0) {
        if (tid == 0) {
            my_flag_bc[source_slot * RDMA_FLAG_SIZE] = source_slot;
            my_flag_bc[source_slot * RDMA_FLAG_SIZE + 1] =
                bcast_trees[source_slot].msgSize_ * nrhs + XK_H;
        }
        __syncthreads();
        symldl_nvshmem_bcast_forward_device(
            &bcast_trees[source_slot], flag_bc_q,
            &my_flag_bc[source_slot * RDMA_FLAG_SIZE], mype, tid,
            dready_x, maxrecvsz);
    }
    __syncthreads();
    int_t edge_begin = source_edge_offsets[source_slot];
    int_t edge_end = source_edge_offsets[source_slot + 1];
    int_t source_first = xsup[source];
    __shared__ int send_reduce;
    for (int_t pos = edge_begin; pos < edge_end; ++pos) {
        int_t edge = source_edge_ids[pos];
        const dSymLDLNVBlockDesc block = blocks[edge];
        const dSymLDLNVPanelDesc panel = panels[block.panel_id];
        int_t target_slot = block.panel_id;
        int_t target_sum = target_ilsum[target_slot] * nrhs +
                           (target_slot + 1) * LSUM_H;
        int outputs = panel.width * nrhs;
        for (int out = tid; out < outputs; out += block_size) {
            int rhs = out / panel.width;
            int col = out - rhs * panel.width;
            double value = 0.0;
            for (int_t row = 0; row < block.nbrow; ++row) {
                int_t relative = rows[block.row_begin + row] - source_first;
                value += panel.values[block.luptr + row +
                                      (int_t) col * panel.nsupr] *
                         dready_x[(int_t) source_slot * maxrecvsz + relative +
                                  (int_t) rhs * source_width];
            }
            d_atomicAdd(lsum + target_sum + col +
                            (int_t) rhs * panel.width,
                        -value);
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            int old = atomicSub(bmod + target_slot, 1);
            send_reduce = old == 1 &&
                          reduce_trees[target_slot].myRoot_ !=
                              reduce_trees[target_slot].myRank_;
            if (send_reduce) {
                __threadfence();
                symldl_nvshmem_backward_reduce_send(
                    target_slot, panel.width, nrhs, target_sum, maxrecvsz,
                    reduce_trees, flag_rd_q, dready_lsum, lsum);
            }
        }
        __syncthreads();
    }
#else
    (void) source_slot; (void) worker_id; (void) lsum; (void) x;
    (void) nrhs;
    (void) nsupers; (void) bmod; (void) bcast_trees;
    (void) reduce_trees; (void) row_ilsum; (void) target_ilsum;
    (void) panels;
    (void) blocks; (void) rows; (void) source_edge_offsets;
    (void) source_edge_ids;
    (void) source_active; (void) xsup;
    (void) grid; (void) source_gids; (void) row_local_index;
    (void) panel_local_index; (void) panel_roots; (void) maxrecvsz;
    (void) mype; (void) flag_bc_q; (void) flag_rd_q;
    (void) dready_x; (void) dready_lsum; (void) my_flag_bc;
    (void) my_flag_rd;
#endif
}

__global__ void symldl_nvshmem_backward_kernel(
    int source_count, int *next_source,
    double *lsum, double *x, int nrhs, int_t nsupers, int *bmod,
    C_Tree *bcast_trees, C_Tree *reduce_trees, int_t *row_ilsum,
    int_t *target_ilsum,
    const dSymLDLNVPanelDesc *panels,
    const dSymLDLNVBlockDesc *blocks, const int_t *rows,
    const int_t *source_edge_offsets, const int_t *source_edge_ids,
    const int *source_active, int_t *xsup, gridinfo_t *grid,
    int_t *source_gids, int_t *row_local_index,
    int_t *panel_local_index, int *panel_roots, int_t maxrecvsz,
    int mype, uint64_t *flag_bc_q, uint64_t *flag_rd_q,
    double *dready_x, double *dready_lsum,
    int *my_flag_bc, int *my_flag_rd)
{
#ifdef HAVE_NVSHMEM
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ int source_position;
    for (;;) {
        if (tid == 0)
            source_position = atomicAdd(next_source, 1);
        __syncthreads();
        if (source_position >= source_count)
            return;
        int source_slot = source_count - 1 - source_position;
        symldl_nvshmem_backward_source(
            source_slot, source_position, lsum, x, nrhs, nsupers, bmod,
            bcast_trees, reduce_trees, row_ilsum, target_ilsum, panels,
            blocks, rows, source_edge_offsets, source_edge_ids,
            source_active, xsup, grid, source_gids, row_local_index,
            panel_local_index, panel_roots,
            maxrecvsz, mype, flag_bc_q, flag_rd_q, dready_x, dready_lsum,
            my_flag_bc, my_flag_rd);
        __syncthreads();
    }
#else
    (void) source_count; (void) next_source;
    (void) lsum; (void) x; (void) nrhs; (void) nsupers; (void) bmod;
    (void) bcast_trees; (void) reduce_trees; (void) row_ilsum;
    (void) target_ilsum;
    (void) panels; (void) blocks; (void) rows;
    (void) source_edge_offsets; (void) source_edge_ids;
    (void) source_active; (void) xsup; (void) grid;
    (void) source_gids; (void) row_local_index;
    (void) panel_local_index; (void) panel_roots; (void) maxrecvsz;
    (void) mype; (void) flag_bc_q; (void) flag_rd_q;
    (void) dready_x; (void) dready_lsum; (void) my_flag_bc;
    (void) my_flag_rd;
#endif
}

static int
symldl_nvshmem_backward_wrap(
    int source_count, int target_count,
    double *lsum, double *x, int nrhs,
    int maxsup, int_t nsupers, int *next_source, int worker_blocks,
    int *bmod, C_Tree *bcast_trees,
    C_Tree *reduce_trees, int_t *row_ilsum, int_t *target_ilsum,
    const dSymLDLNVPanelDesc *panels,
    const dSymLDLNVBlockDesc *blocks, const int_t *rows,
    const int_t *source_edge_offsets, const int_t *source_edge_ids,
    const int *source_active, int_t *xsup, gridinfo_t *grid,
    int_t *source_gids, int_t *target_gids, int_t *row_local_index,
    int_t *panel_local_index, int *panel_roots, int_t maxrecvsz,
    uint64_t *flag_bc_q, uint64_t *flag_rd_q, double *dready_x,
    double *dready_lsum, int *my_flag_bc, int *my_flag_rd,
    int *d_nfrecv, int *h_nfrecv, int *d_status, int *d_colnum,
    int *d_mynum, int *d_mymaskstart, int *d_mymasklength,
    int *d_nfrecvmod, int *d_statusmod, int *d_colnummod,
    int *d_mynummod, int *d_mymaskstartmod, int *d_mymasklengthmod,
    int *d_recv_cnt, int *d_msgnum, int *d_flag_mod)
{
#ifndef HAVE_NVSHMEM
    (void) source_count; (void) target_count;
    (void) lsum; (void) x;
    (void) nrhs; (void) maxsup; (void) nsupers; (void) next_source;
    (void) worker_blocks; (void) bmod;
    (void) bcast_trees; (void) reduce_trees; (void) row_ilsum;
    (void) target_ilsum;
    (void) panels; (void) blocks; (void) rows;
    (void) source_edge_offsets; (void) source_edge_ids;
    (void) source_active; (void) xsup; (void) grid;
    (void) source_gids; (void) target_gids; (void) row_local_index;
    (void) panel_local_index; (void) panel_roots; (void) maxrecvsz;
    (void) flag_bc_q; (void) flag_rd_q; (void) dready_x;
    (void) dready_lsum; (void) my_flag_bc; (void) my_flag_rd;
    (void) d_nfrecv; (void) h_nfrecv; (void) d_status;
    (void) d_colnum; (void) d_mynum; (void) d_mymaskstart;
    (void) d_mymasklength; (void) d_nfrecvmod; (void) d_statusmod;
    (void) d_colnummod; (void) d_mynummod; (void) d_mymaskstartmod;
    (void) d_mymasklengthmod; (void) d_recv_cnt; (void) d_msgnum;
    (void) d_flag_mod;
    return -1;
#else
    int mype = nvshmem_my_pe();
    int waiter_blocks = h_nfrecv[2];
    if (waiter_blocks < 2 || h_nfrecv[1] <= 0 || worker_blocks <= 0)
        return -1;
    cudaStream_t streams[2] = {NULL, NULL};
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &streams[i], cudaStreamNonBlocking));

    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, symldl_nvshmem_backward_kernel));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize,
                                  attributes.localSizeBytes));

    int min_grid_size = 0;
    int wait_threads = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &wait_threads,
        (const void *) symldl_nvshmem_wait_bcrd, 0, 0));
    (void) min_grid_size;
    if (wait_threads < h_nfrecv[1]) {
        h_nfrecv[1] = wait_threads;
        CUDA_CHECK(cudaMemcpy(d_nfrecv, h_nfrecv, 3 * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    dim3 wait_grid(waiter_blocks);
    dim3 wait_block(h_nfrecv[1]);
    void *wait_args[] = {
        &nrhs, &reduce_trees, &maxrecvsz, &mype, &flag_bc_q, &flag_rd_q,
        &dready_x, &dready_lsum, &my_flag_bc, &my_flag_rd, &d_nfrecv,
        &d_status, &d_colnum, &d_mynum, &d_mymaskstart, &d_mymasklength,
        &d_nfrecvmod, &d_statusmod, &d_colnummod, &d_mynummod,
        &d_mymaskstartmod, &d_mymasklengthmod, &d_recv_cnt, &d_msgnum,
        &d_flag_mod, &lsum, &bmod, &grid, &xsup, &target_ilsum,
        &target_gids, &target_count, &nsupers
    };
    int status = nvshmemx_collective_launch(
        (const void *) symldl_nvshmem_wait_bcrd, wait_grid, wait_block,
        wait_args, 0, streams[0]);
    if (status != NVSHMEMX_SUCCESS) {
        for (int i = 0; i < 2; ++i)
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        return -1;
    }

    if (source_count > 0) {
        dim3 solve_grid(std::min(source_count, worker_blocks));
        dim3 solve_block(DIM_X, DIM_Y);
        symldl_nvshmem_backward_kernel<<<
            solve_grid, solve_block, 0, streams[1]>>>(
                source_count, next_source, lsum, x, nrhs, nsupers, bmod,
                bcast_trees, reduce_trees, row_ilsum, target_ilsum, panels,
                blocks, rows, source_edge_offsets, source_edge_ids,
                source_active, xsup, grid, source_gids, row_local_index,
                panel_local_index, panel_roots,
                maxrecvsz, mype, flag_bc_q, flag_rd_q, dready_x,
                dready_lsum, my_flag_bc, my_flag_rd);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    return 0;
#endif
}

namespace {

struct dSymLDLNVSHMEMSharedState {
    int references;
    int_t n;
    int_t nsupers;
    int nrhs;
    int_t x_count;
    int_t lsum_capacity;
    int_t panel_count;
    int_t block_count;
    int_t factor_row_count;
    int_t row_count;
    int_t backward_lsum_count;
    int panel_slots;
    int row_slots;
    int transport_slots;
    int maxsup;
    int maxrecvsz;
    dtrf3Dpartition_t *trf3Dpartition;
    gridinfo3d_t *grid3d;
    gridinfo_t *grid;
    const dSymLDLSolveGraph *graph;
    const int_t *host_xsup;
    const int_t *host_ilsum;

    /* Device panel bindings mirror graph topology with GPU value pointers. */
    std::vector<dSymLDLNVPanelDesc> h_panels;

    double *d_x;
    double *d_lsum;
    int_t *d_ilsum;
    int_t *d_xsup;
    dSymLDLNVPanelDesc *d_panels;
    dSymLDLNVBlockDesc *d_blocks;
    int_t *d_rows;
    int_t *d_panel_gids;
    int_t *d_row_gids;
    int_t *d_source_edge_offsets;
    int_t *d_source_edge_ids;
    int_t *d_target_ilsum;
    int_t *d_row_local_index;
    int_t *d_panel_local_index;
    int *d_diag_roots;
    int *d_panel_roots;
    gridinfo_t *d_grid;

    uint64_t *flag_bc_q;
    uint64_t *flag_rd_q;
    double *ready_x;
    double *ready_lsum;
    int *my_flag_bc;
    int *my_flag_rd;
};

struct dSymLDLNVSHMEMForwardState {
    dSymLDLNVSHMEMSharedState *shared;
    int_t n;
    int_t nsupers;
    int nrhs;
    int_t x_count;
    int_t lsum_count;
    int_t panel_count;
    int_t block_count;
    int_t factor_row_count;
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
    int panel_replication_batches;
    int panel_replication_sources;
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
    const int_t *h_panel_gids;
    const int_t *h_row_gids;
    std::vector<int_t> h_rhs_gids;
    std::vector<int_t> h_rhs_offsets;

    double *d_x;
    double *d_lsum;
    int *d_fmod;
    C_Tree *d_bcast_trees;
    C_Tree *d_reduce_trees;
    int_t *d_ilsum;
    int_t *d_xsup;
    dSymLDLNVPanelDesc *d_panels;
    dSymLDLNVBlockDesc *d_blocks;
    int_t *d_rows;
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

struct dSymLDLNVSHMEMBackwardState {
    dSymLDLNVSHMEMSharedState *shared;
    int_t n;
    int_t nsupers;
    int nrhs;
    int_t x_count;
    int_t panel_count;
    int_t block_count;
    int_t factor_row_count;
    int_t source_count;
    int_t target_count;
    int_t source_slots;
    int_t target_slots;
    int maxsup;
    int maxrecvsz;
    int worker_blocks;
    int rank;
    int nprocs;
    int diagnostics;
    int diagonal_ready;
    int expected_bcast_recvs;
    int expected_reduce_recvs;
    int expected_reduce_sends;
    int h_nfrecv[3];
    int h_nfrecvmod[4];
    double setup_time;
    double h2d_time;
    double diagonal_time;
    double backward_time;
    double d2h_time;
    dtrf3Dpartition_t *trf3Dpartition;
    gridinfo3d_t *grid3d;
    gridinfo_t *grid;
    MPI_Comm layer_comm;
    const int_t *host_xsup;
    const int_t *host_ilsum;

    std::vector<int> h_bmod;
    std::vector<int> h_status;
    std::vector<int> h_statusmod;
    std::vector<int> h_recv_cnt;
    std::vector<int> h_colnum;
    std::vector<int> h_colnummod;
    std::vector<int> h_source_active;

    double *d_x;
    double *d_diag_input;
    double *d_lsum;
    int *d_bmod;
    C_Tree *d_bcast_trees;
    C_Tree *d_reduce_trees;
    int_t *d_global_ilsum;
    int_t *d_target_ilsum;
    int_t *d_xsup;
    dSymLDLNVPanelDesc *d_panels;
    dSymLDLNVBlockDesc *d_blocks;
    int_t *d_rows;
    int_t *d_source_gids;
    int_t *d_target_gids;
    int_t *d_source_edge_offsets;
    int_t *d_source_edge_ids;
    int *d_source_active;
    int_t *d_row_local_index;
    int_t *d_panel_local_index;
    int *d_panel_roots;
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
    int *d_next_source;

    uint64_t *flag_bc_q;
    uint64_t *flag_rd_q;
    double *ready_x;
    double *ready_lsum;
    int *my_flag_bc;
    int *my_flag_rd;
};

struct dSymLDLNVSHMEMSolveState {
    dSymLDLNVSHMEMForwardState *forward;
    dSymLDLNVSHMEMBackwardState *backward;
};

static int_t
symldl_nvshmem_super_size(
    const dSymLDLNVSHMEMForwardState *state, int_t k)
{
    return state->host_xsup[k + 1] - state->host_xsup[k];
}

static int_t
symldl_nvshmem_x_block(
    const dSymLDLNVSHMEMForwardState *state, int_t local_row)
{
    return state->host_ilsum[local_row] * state->nrhs +
           (local_row + 1) * XK_H;
}

#ifdef HAVE_NVSHMEM
static int symldl_nvshmem_nvshmem_initialized = 0;
static int symldl_nvshmem_shared_state_count = 0;
#endif

static int
symldl_nvshmem_env_enabled(const char *name)
{
    const char *value = getenv(name);
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool
symldl_nvshmem_range_valid(int_t begin, int_t count, int_t extent)
{
    return begin >= 0 && count >= 0 && extent >= 0 && begin <= extent &&
           count <= extent - begin;
}

static size_t
symldl_nvshmem_checked_product(size_t a, size_t b, const char *message)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        ABORT(message);
    return a * b;
}

template <typename T>
static void
symldl_nvshmem_cuda_alloc(T **ptr, size_t count)
{
    size_t n = std::max((size_t) 1, count);
    CUDA_CHECK(cudaMalloc((void **) ptr, n * sizeof(T)));
}

template <typename T>
static void
symldl_nvshmem_cuda_copy(T **ptr, const T *source, size_t count)
{
    symldl_nvshmem_cuda_alloc(ptr, count);
    if (count > 0)
        CUDA_CHECK(cudaMemcpy(*ptr, source, count * sizeof(T),
                              cudaMemcpyHostToDevice));
}

static void
symldl_nvshmem_cuda_free(void *ptr)
{
    if (ptr != NULL)
        cudaFree(ptr);
}

static dSymLDLNVSHMEMSharedState *
symldl_nvshmem_shared_create(
    int nrhs, int_t x_count, int_t lsum_count,
    const dSymLDLSolveGraph *graph,
    const dSymLDLNVPanelDesc *device_panels,
    dtrf3Dpartition_t *partition,
    gridinfo3d_t *grid3d)
{
    dSymLDLNVSHMEMSharedState *shared =
        new dSymLDLNVSHMEMSharedState();
    shared->references = 1;
    shared->n = graph->n;
    shared->nsupers = graph->nsupers;
    shared->nrhs = nrhs;
    shared->x_count = x_count;
    shared->lsum_capacity = lsum_count;
    shared->panel_count = graph->panel_count;
    shared->block_count = graph->block_count;
    shared->factor_row_count = graph->factor_row_count;
    shared->row_count = graph->row_count;
    shared->trf3Dpartition = partition;
    shared->grid3d = grid3d;
    shared->grid = &grid3d->grid2d;
    shared->graph = graph;
    shared->host_xsup = graph->xsup;
    shared->host_ilsum = graph->ilsum;

    if (graph->panel_count > 0)
        shared->h_panels.assign(
            device_panels, device_panels + graph->panel_count);
    size_t compact_count =
        (size_t) graph->backward_lsum_rows * (size_t) nrhs +
        ((size_t) graph->panel_count + 1) * (size_t) LSUM_H;
    if (compact_count > (size_t) std::numeric_limits<int_t>::max())
        ABORT("SymLDL NVSHMEM 3D backward sum workspace overflows.");
    shared->backward_lsum_count = (int_t) compact_count;
    shared->lsum_capacity = std::max(
        shared->lsum_capacity, shared->backward_lsum_count);

    int local_counts[2] = {
        (int) shared->panel_count, (int) shared->row_count};
    int max_counts[2] = {0, 0};
    MPI_Allreduce(local_counts, max_counts, 2, MPI_INT, MPI_MAX,
                  shared->grid->comm);
    shared->panel_slots = std::max(1, max_counts[0]);
    shared->row_slots = std::max(1, max_counts[1]);
    shared->transport_slots = std::max(
        shared->panel_slots, shared->row_slots);
    shared->maxsup = graph->maxsup;
    shared->maxrecvsz = shared->maxsup * nrhs +
                        std::max((int) XK_H, (int) LSUM_H);

    symldl_nvshmem_cuda_alloc(&shared->d_x, (size_t) x_count);
    symldl_nvshmem_cuda_alloc(
        &shared->d_lsum, (size_t) shared->lsum_capacity);
    symldl_nvshmem_cuda_copy(
        &shared->d_ilsum, graph->ilsum, (size_t) shared->row_count + 1);
    symldl_nvshmem_cuda_copy(
        &shared->d_xsup, graph->xsup, (size_t) graph->nsupers + 1);
    symldl_nvshmem_cuda_copy(
        &shared->d_panels, shared->h_panels.data(),
        shared->h_panels.size());
    symldl_nvshmem_cuda_copy(
        &shared->d_blocks, graph->blocks, (size_t) graph->block_count);
    symldl_nvshmem_cuda_copy(
        &shared->d_rows, graph->rows, (size_t) graph->factor_row_count);
    symldl_nvshmem_cuda_copy(
        &shared->d_panel_gids, graph->panel_gids,
        (size_t) graph->panel_count);
    symldl_nvshmem_cuda_copy(
        &shared->d_row_gids, graph->row_gids,
        (size_t) graph->row_count);
    symldl_nvshmem_cuda_copy(
        &shared->d_source_edge_offsets, graph->source_edge_offsets,
        (size_t) graph->row_count + 1);
    symldl_nvshmem_cuda_copy(
        &shared->d_source_edge_ids, graph->source_edge_ids,
        (size_t) graph->block_count);
    symldl_nvshmem_cuda_copy(
        &shared->d_target_ilsum, graph->target_ilsum,
        (size_t) graph->panel_count + 1);
    symldl_nvshmem_cuda_copy(
        &shared->d_row_local_index, graph->row_local_index,
        (size_t) graph->nsupers);
    symldl_nvshmem_cuda_copy(
        &shared->d_panel_local_index, graph->panel_local_index,
        (size_t) graph->nsupers);
    symldl_nvshmem_cuda_copy(
        &shared->d_diag_roots, graph->diag_roots,
        (size_t) graph->nsupers);
    symldl_nvshmem_cuda_copy(
        &shared->d_panel_roots, graph->panel_roots,
        (size_t) graph->nsupers);
    symldl_nvshmem_cuda_copy(&shared->d_grid, shared->grid, 1);

#ifdef HAVE_NVSHMEM
    if (!symldl_nvshmem_nvshmem_initialized) {
        nv_init_wrapper(shared->grid->comm);
        symldl_nvshmem_nvshmem_initialized = 1;
    }
    ++symldl_nvshmem_shared_state_count;
    int rank = 0;
    int nprocs = 0;
    MPI_Comm_rank(shared->grid->comm, &rank);
    MPI_Comm_size(shared->grid->comm, &nprocs);
    if (nvshmem_my_pe() != rank || nvshmem_n_pes() != nprocs ||
        nprocs != shared->grid->nprow * shared->grid->npcol) {
        if (rank == 0)
            fprintf(stderr,
                    "SymLDL NVSHMEM 3D solve requires layer-local NVSHMEM rank space.\n");
        return shared;
    }

    size_t flag_bc_count = (size_t) RDMA_FLAG_SIZE *
                           ((size_t) shared->transport_slots + 1);
    size_t flag_rd_count = (size_t) RDMA_FLAG_SIZE *
                           (size_t) shared->transport_slots * 2;
    size_t ready_x_count = (size_t) shared->maxrecvsz *
                           (size_t) shared->transport_slots;
    size_t ready_lsum_count = (size_t) 2 * (size_t) shared->maxrecvsz *
                              (size_t) shared->transport_slots;
    shared->flag_bc_q = (uint64_t *) nvshmem_malloc(
        flag_bc_count * sizeof(uint64_t));
    shared->flag_rd_q = (uint64_t *) nvshmem_malloc(
        flag_rd_count * sizeof(uint64_t));
    shared->ready_x = (double *) nvshmem_malloc(
        ready_x_count * sizeof(double));
    shared->ready_lsum = (double *) nvshmem_malloc(
        ready_lsum_count * sizeof(double));
    shared->my_flag_bc = (int *) nvshmem_malloc(
        flag_bc_count * sizeof(int));
    shared->my_flag_rd = (int *) nvshmem_malloc(
        flag_rd_count * sizeof(int));
    if (shared->flag_bc_q == NULL || shared->flag_rd_q == NULL ||
        shared->ready_x == NULL || shared->ready_lsum == NULL ||
        shared->my_flag_bc == NULL || shared->my_flag_rd == NULL)
        return shared;
    CUDA_CHECK(cudaMemset(
        shared->my_flag_bc, 0, flag_bc_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(
        shared->my_flag_rd, 0, flag_rd_count * sizeof(int)));
#endif
    return shared;
}

static void
symldl_nvshmem_shared_retain(dSymLDLNVSHMEMSharedState *shared)
{
    if (shared == NULL || shared->references <= 0)
        ABORT("SymLDL NVSHMEM 3D shared state is invalid.");
    ++shared->references;
}

static void
symldl_nvshmem_shared_release(dSymLDLNVSHMEMSharedState *shared)
{
    if (shared == NULL)
        return;
    if (shared->references <= 0)
        ABORT("SymLDL NVSHMEM 3D shared state reference count is invalid.");
    if (--shared->references != 0)
        return;
    symldl_nvshmem_cuda_free(shared->d_x);
    symldl_nvshmem_cuda_free(shared->d_lsum);
    symldl_nvshmem_cuda_free(shared->d_ilsum);
    symldl_nvshmem_cuda_free(shared->d_xsup);
    symldl_nvshmem_cuda_free(shared->d_panels);
    symldl_nvshmem_cuda_free(shared->d_blocks);
    symldl_nvshmem_cuda_free(shared->d_rows);
    symldl_nvshmem_cuda_free(shared->d_panel_gids);
    symldl_nvshmem_cuda_free(shared->d_row_gids);
    symldl_nvshmem_cuda_free(shared->d_source_edge_offsets);
    symldl_nvshmem_cuda_free(shared->d_source_edge_ids);
    symldl_nvshmem_cuda_free(shared->d_target_ilsum);
    symldl_nvshmem_cuda_free(shared->d_row_local_index);
    symldl_nvshmem_cuda_free(shared->d_panel_local_index);
    symldl_nvshmem_cuda_free(shared->d_diag_roots);
    symldl_nvshmem_cuda_free(shared->d_panel_roots);
    symldl_nvshmem_cuda_free(shared->d_grid);
#ifdef HAVE_NVSHMEM
    if (shared->flag_bc_q) nvshmem_free(shared->flag_bc_q);
    if (shared->flag_rd_q) nvshmem_free(shared->flag_rd_q);
    if (shared->ready_x) nvshmem_free(shared->ready_x);
    if (shared->ready_lsum) nvshmem_free(shared->ready_lsum);
    if (shared->my_flag_bc) nvshmem_free(shared->my_flag_bc);
    if (shared->my_flag_rd) nvshmem_free(shared->my_flag_rd);
    if (symldl_nvshmem_shared_state_count <= 0)
        ABORT("SymLDL NVSHMEM shared state ownership is invalid.");
    if (--symldl_nvshmem_shared_state_count == 0) {
        nvshmem_finalize();
        symldl_nvshmem_nvshmem_initialized = 0;
    }
#endif
    delete shared;
}

static void
symldl_nvshmem_state_free(dSymLDLNVSHMEMForwardState *state)
{
    if (state == NULL)
        return;

    symldl_nvshmem_cuda_free(state->d_fmod);
    symldl_nvshmem_cuda_free(state->d_bcast_trees);
    symldl_nvshmem_cuda_free(state->d_reduce_trees);
    symldl_nvshmem_cuda_free(state->d_nfrecv);
    symldl_nvshmem_cuda_free(state->d_status);
    symldl_nvshmem_cuda_free(state->d_colnum);
    symldl_nvshmem_cuda_free(state->d_mynum);
    symldl_nvshmem_cuda_free(state->d_mymaskstart);
    symldl_nvshmem_cuda_free(state->d_mymasklength);
    symldl_nvshmem_cuda_free(state->d_nfrecvmod);
    symldl_nvshmem_cuda_free(state->d_statusmod);
    symldl_nvshmem_cuda_free(state->d_colnummod);
    symldl_nvshmem_cuda_free(state->d_mynummod);
    symldl_nvshmem_cuda_free(state->d_mymaskstartmod);
    symldl_nvshmem_cuda_free(state->d_mymasklengthmod);
    symldl_nvshmem_cuda_free(state->d_recv_cnt);
    symldl_nvshmem_cuda_free(state->d_msgnum);
    symldl_nvshmem_cuda_free(state->d_flag_mod);
    symldl_nvshmem_shared_release(state->shared);
    delete state;
}

static void
symldl_nvshmem_backward_state_free(
    dSymLDLNVSHMEMBackwardState *state)
{
    if (state == NULL)
        return;
    symldl_nvshmem_cuda_free(state->d_diag_input);
    symldl_nvshmem_cuda_free(state->d_bmod);
    symldl_nvshmem_cuda_free(state->d_bcast_trees);
    symldl_nvshmem_cuda_free(state->d_reduce_trees);
    symldl_nvshmem_cuda_free(state->d_source_active);
    symldl_nvshmem_cuda_free(state->d_nfrecv);
    symldl_nvshmem_cuda_free(state->d_status);
    symldl_nvshmem_cuda_free(state->d_colnum);
    symldl_nvshmem_cuda_free(state->d_mynum);
    symldl_nvshmem_cuda_free(state->d_mymaskstart);
    symldl_nvshmem_cuda_free(state->d_mymasklength);
    symldl_nvshmem_cuda_free(state->d_nfrecvmod);
    symldl_nvshmem_cuda_free(state->d_statusmod);
    symldl_nvshmem_cuda_free(state->d_colnummod);
    symldl_nvshmem_cuda_free(state->d_mynummod);
    symldl_nvshmem_cuda_free(state->d_mymaskstartmod);
    symldl_nvshmem_cuda_free(state->d_mymasklengthmod);
    symldl_nvshmem_cuda_free(state->d_recv_cnt);
    symldl_nvshmem_cuda_free(state->d_msgnum);
    symldl_nvshmem_cuda_free(state->d_flag_mod);
    symldl_nvshmem_cuda_free(state->d_next_source);
    symldl_nvshmem_shared_release(state->shared);
    delete state;
}

static void
symldl_nvshmem_panel_targets(
    const dSymLDLNVPanelDesc &panel,
    const dSymLDLNVBlockDesc *blocks,
    const int_t *row_local_index, int_t nsupers, int_t source_gid,
    std::vector<int_t> &targets)
{
    if (!panel.active || panel.values == NULL)
        return;
    for (int_t lb = 0; lb < panel.block_count; ++lb) {
        const dSymLDLNVBlockDesc &block =
            blocks[panel.block_begin + lb];
        int_t target = block.target_gid;
        if (target != source_gid) {
            if (target < 0 || target >= nsupers ||
                row_local_index[target] < 0)
                ABORT("SymLDL NVSHMEM 3D forward panel targets a row outside its Z layer.");
            targets.push_back(target);
        }
    }
}

static void
symldl_nvshmem_validate_panel(
    const dSymLDLNVSHMEMForwardState *state,
    int_t panel_id,
    const dSymLDLNVPanelDesc &panel,
    const dSymLDLNVBlockDesc *blocks,
    const int_t *rows)
{
    if (panel.gid < 0 || panel.gid >= state->nsupers || panel.width <= 0 ||
        panel.width != symldl_nvshmem_super_size(state, panel.gid))
        ABORT("SymLDL NVSHMEM 3D forward panel descriptor is invalid.");
    int myrow = MYROW(state->grid->iam, state->grid);
    int root = state->trf3Dpartition->symV2DiagRoot[panel.gid];
    if (!panel.active) {
        if (panel.values != NULL || panel.block_count != 0 || panel.has_diag)
            ABORT("SymLDL NVSHMEM 3D forward inactive panel has storage.");
        return;
    }
    if (!symldl_nvshmem_range_valid(
            panel.block_begin, panel.block_count, state->block_count))
        ABORT("SymLDL NVSHMEM 3D forward panel block range is invalid.");
    if (panel.values == NULL || panel.nsupr <= 0 || panel.value_count < 0) {
        if (myrow == root)
            ABORT("SymLDL NVSHMEM 3D forward diagonal panel is missing.");
        return;
    }
    size_t panel_values = symldl_nvshmem_checked_product(
        (size_t) panel.nsupr, (size_t) panel.width,
        "SymLDL NVSHMEM 3D forward panel size overflows.");
    if ((size_t) panel.value_count < panel_values) {
        if (myrow == root)
            ABORT("SymLDL NVSHMEM 3D forward diagonal panel is missing.");
        return;
    }
    int_t previous_target = -1;
    for (int_t lb = 0; lb < panel.block_count; ++lb) {
        const dSymLDLNVBlockDesc &block =
            blocks[panel.block_begin + lb];
        if (block.panel_id != panel_id ||
            block.target_gid < 0 || block.target_gid >= state->nsupers ||
            block.target_gid <= previous_target || block.nbrow <= 0 ||
            !symldl_nvshmem_range_valid(
                block.row_begin, block.nbrow, state->factor_row_count) ||
            !symldl_nvshmem_range_valid(
                block.luptr, block.nbrow, panel.nsupr))
            ABORT("SymLDL NVSHMEM 3D forward panel block is invalid.");
        previous_target = block.target_gid;
        size_t last_column = symldl_nvshmem_checked_product(
            (size_t) (panel.width - 1), (size_t) panel.nsupr,
            "SymLDL NVSHMEM 3D forward panel offset overflows.");
        size_t block_end = last_column + (size_t) block.luptr +
                           (size_t) block.nbrow;
        if (block_end > (size_t) panel.value_count)
            ABORT("SymLDL NVSHMEM 3D forward panel block exceeds its values.");
        if (state->trf3Dpartition->symV2RowLocalIndex[
                block.target_gid] < 0)
            ABORT("SymLDL NVSHMEM 3D forward panel targets a row outside its Z layer.");
        for (int_t r = 0; r < block.nbrow; ++r) {
            int_t grow = rows[block.row_begin + r];
            if (grow < state->host_xsup[block.target_gid] ||
                grow >= state->host_xsup[block.target_gid + 1])
                ABORT("SymLDL NVSHMEM 3D forward row metadata is inconsistent.");
        }
    }

    if (myrow == root) {
        if (!panel.has_diag || panel.diag_luptr < 0 ||
            !symldl_nvshmem_range_valid(
                panel.diag_luptr, panel.width, panel.nsupr))
            ABORT("SymLDL NVSHMEM 3D forward diagonal panel is invalid.");
    } else if (panel.has_diag) {
        ABORT("SymLDL NVSHMEM 3D forward diagonal block is on the wrong process row.");
    }
    for (int_t lb = 0; lb < panel.block_count; ++lb)
        if (blocks[panel.block_begin + lb].target_gid == panel.gid)
            ABORT("SymLDL NVSHMEM 3D forward update descriptors include the diagonal block.");
}

static void
symldl_nvshmem_build_bcast_trees(
    dSymLDLNVSHMEMForwardState *state,
    const dSymLDLNVPanelDesc *panels,
    const dSymLDLNVBlockDesc *blocks,
    std::vector<C_Tree> &trees)
{
    gridinfo_t *grid = state->grid;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int Pr = grid->nprow;
    const int_t missing = 3 * state->nsupers;
    if (state->panel_count > INT_MAX)
        ABORT("SymLDL NVSHMEM 3D forward has too many panel slots.");
    std::vector<int_t> local_first((size_t) state->panel_count, missing);
    std::vector<int_t> all_first(
        (size_t) Pr * (size_t) state->panel_count, missing);

    trees.resize((size_t) state->panel_count);
    state->h_status.assign((size_t) state->panel_count, 1);
    for (int_t lp = 0; lp < state->panel_count; ++lp) {
        C_BcTree_Nullify(&trees[(size_t) lp]);
        int_t k = state->h_panel_gids[(size_t) lp];
        std::vector<int_t> local_targets;
        symldl_nvshmem_panel_targets(
            panels[lp], blocks,
            state->trf3Dpartition->symV2RowLocalIndex, state->nsupers, k,
            local_targets);
        for (size_t i = 0; i < local_targets.size(); ++i)
            local_first[(size_t) lp] = std::min(
                local_first[(size_t) lp], local_targets[i]);
        int root = state->trf3Dpartition->symV2DiagRoot[k];
        if (panels[lp].active && myrow == root)
            local_first[(size_t) lp] = std::min(local_first[(size_t) lp], k);
    }

    MPI_Allgather(local_first.data(), (int) state->panel_count, mpi_int_t,
                  all_first.data(), (int) state->panel_count, mpi_int_t,
                  grid->cscp.comm);

    for (int_t lp = 0; lp < state->panel_count; ++lp) {
        int_t k = state->h_panel_gids[(size_t) lp];
        int root = state->trf3Dpartition->symV2DiagRoot[k];
        std::vector<int_t> first((size_t) Pr, missing);
        for (int pr = 0; pr < Pr; ++pr)
            first[(size_t) pr] =
                all_first[(size_t) pr * (size_t) state->panel_count +
                          (size_t) lp];

        std::vector<std::pair<int_t, int> > ordered;
        for (int pr = 0; pr < Pr; ++pr)
            if (pr != root && first[(size_t) pr] != missing)
                ordered.push_back(std::make_pair(first[(size_t) pr], pr));
        if (first[(size_t) root] == missing && !ordered.empty())
            ABORT("SymLDL NVSHMEM 3D forward broadcast root is inactive.");
        std::sort(ordered.begin(), ordered.end());
        std::vector<int> ranks;
        ranks.push_back(PNUM(root, mycol, grid));
        for (size_t i = 0; i < ordered.size(); ++i)
            ranks.push_back(PNUM(ordered[i].second, mycol, grid));

        int active = first[(size_t) myrow] != missing;
        if (active && ranks.size() > 1) {
            int needrecv = 0;
            C_BcTree_Create_nv(&trees[(size_t) lp], state->layer_comm,
                               ranks.data(), (int) ranks.size(),
                               symldl_nvshmem_super_size(state, k), 'd',
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
symldl_nvshmem_build_reduce_trees(
    dSymLDLNVSHMEMForwardState *state,
    const dSymLDLNVPanelDesc *panels,
    const dSymLDLNVBlockDesc *blocks,
    std::vector<C_Tree> &trees)
{
    gridinfo_t *grid = state->grid;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int Pc = grid->npcol;
    const int_t missing = -3 * state->nsupers;
    if (state->row_count > INT_MAX)
        ABORT("SymLDL NVSHMEM 3D forward has too many row slots.");
    std::vector<int_t> local_last((size_t) state->row_count, missing);
    std::vector<int_t> all_last(
        (size_t) Pc * (size_t) state->row_count, missing);

    state->h_fmod.assign((size_t) state->row_count, 0);
    for (int_t lp = 0; lp < state->panel_count; ++lp) {
        int_t source = state->h_panel_gids[(size_t) lp];
        std::vector<int_t> targets;
        symldl_nvshmem_panel_targets(
            panels[lp], blocks,
            state->trf3Dpartition->symV2RowLocalIndex,
            state->nsupers, source, targets);
        for (size_t i = 0; i < targets.size(); ++i) {
            int_t lr = state->trf3Dpartition->symV2RowLocalIndex[
                targets[i]];
            local_last[(size_t) lr] = std::max(
                local_last[(size_t) lr], source);
            ++state->h_fmod[(size_t) lr];
        }
    }

    MPI_Allgather(local_last.data(), (int) state->row_count, mpi_int_t,
                  all_last.data(), (int) state->row_count, mpi_int_t,
                  grid->rscp.comm);

    trees.resize((size_t) state->row_count);
    state->h_statusmod.assign((size_t) state->row_count * 2, 1);
    state->h_recv_cnt.assign((size_t) state->row_count, 0);
    for (int_t lr = 0; lr < state->row_count; ++lr) {
        C_RdTree_Nullify(&trees[(size_t) lr]);
        int_t k = state->h_row_gids[(size_t) lr];
        std::vector<int_t> last((size_t) Pc, missing);
        for (int pc = 0; pc < Pc; ++pc)
            last[(size_t) pc] =
                all_last[(size_t) pc * (size_t) state->row_count +
                         (size_t) lr];
        int total = 0;
        for (int pc = 0; pc < Pc; ++pc)
            total += last[(size_t) pc] != missing;
        if (total == 0)
            continue;
        int root = state->trf3Dpartition->symV2PanelRoot[k];
        last[(size_t) root] = std::max(last[(size_t) root], k);

        std::vector<std::pair<int_t, int> > ordered;
        for (int pc = 0; pc < Pc; ++pc)
            if (pc != root && last[(size_t) pc] != missing)
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

        int active = last[(size_t) mycol] != missing;
        if (active && ranks.size() > 1) {
            int needrecv = 0;
            int needsend = 0;
            C_RdTree_Create_nv(&trees[(size_t) lr], state->layer_comm,
                               ranks.data(), (int) ranks.size(),
                               symldl_nvshmem_super_size(state, k), 'd',
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
symldl_nvshmem_z_reduce_tree(
    dSymLDLNVSHMEMForwardState *state, int_t tree_id, int_t sender,
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
        int_t ii = symldl_nvshmem_x_block(state, lr);
        int count = (int) (symldl_nvshmem_super_size(state, k) *
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
symldl_nvshmem_z_bcast_tree(
    dSymLDLNVSHMEMForwardState *state, int_t tree_id, int_t sender,
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
        int_t ii = symldl_nvshmem_x_block(state, lr);
        int count = (int) (symldl_nvshmem_super_size(state, k) *
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

static void
symldl_nvshmem_build_backward_bcast_trees(
    dSymLDLNVSHMEMBackwardState *state,
    std::vector<C_Tree> &trees)
{
    gridinfo_t *grid = state->grid;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int Pc = grid->npcol;
    const int_t missing = -3 * state->nsupers;
    std::vector<int_t> local_last((size_t) state->source_count, missing);
    std::vector<int_t> all_last(
        (size_t) Pc * (size_t) state->source_count, missing);

    for (int_t slot = 0; slot < state->source_count; ++slot) {
        int_t source = state->shared->graph->row_gids[(size_t) slot];
        int root = state->trf3Dpartition->symV2PanelRoot[source];
        if (mycol == root)
            local_last[(size_t) slot] = source;
        if (state->shared->graph->source_edge_offsets[(size_t) slot + 1] >
            state->shared->graph->source_edge_offsets[(size_t) slot])
            local_last[(size_t) slot] = std::max(
                local_last[(size_t) slot],
                state->shared->graph->row_gids[(size_t) slot]);
    }

    MPI_Allgather(local_last.data(), (int) state->source_count, mpi_int_t,
                  all_last.data(), (int) state->source_count, mpi_int_t,
                  grid->rscp.comm);

    trees.resize((size_t) state->source_count);
    state->h_status.assign((size_t) state->source_count, 1);
    state->h_source_active.assign((size_t) state->source_count, 0);
    for (int_t slot = 0; slot < state->source_count; ++slot) {
        C_BcTree_Nullify(&trees[(size_t) slot]);
        int_t source = state->shared->graph->row_gids[(size_t) slot];
        int root = state->trf3Dpartition->symV2PanelRoot[source];
        std::vector<std::pair<int_t, int> > ordered;
        for (int pc = 0; pc < Pc; ++pc) {
            int_t last = all_last[(size_t) pc * (size_t) state->source_count +
                                  (size_t) slot];
            if (pc != root && last != missing)
                ordered.push_back(std::make_pair(last, pc));
        }
        std::sort(ordered.begin(), ordered.end(),
                  [](const std::pair<int_t, int> &a,
                     const std::pair<int_t, int> &b) {
                      return a.first > b.first;
                  });
        std::vector<int> ranks;
        ranks.push_back(PNUM(myrow, root, grid));
        for (size_t i = 0; i < ordered.size(); ++i)
            ranks.push_back(PNUM(myrow, ordered[i].second, grid));
        int active = local_last[(size_t) slot] != missing;
        state->h_source_active[(size_t) slot] = active;
        if (active && ranks.size() > 1) {
            int needrecv = 0;
            C_BcTree_Create_nv(&trees[(size_t) slot], state->layer_comm,
                               ranks.data(), (int) ranks.size(),
                               state->host_xsup[source + 1] -
                                   state->host_xsup[source],
                               'd', &needrecv);
            trees[(size_t) slot].tag_ = BC_U;
            if (needrecv) {
                state->h_status[(size_t) slot] = 0;
                state->h_colnum.push_back((int) slot);
                ++state->expected_bcast_recvs;
            }
        }
    }
}

static void
symldl_nvshmem_build_backward_reduce_trees(
    dSymLDLNVSHMEMBackwardState *state,
    const dSymLDLNVPanelDesc *panels,
    std::vector<C_Tree> &trees)
{
    gridinfo_t *grid = state->grid;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int Pr = grid->nprow;
    const int_t missing = -3 * state->nsupers;
    std::vector<int_t> local_last((size_t) state->target_count, missing);
    std::vector<int_t> all_last(
        (size_t) Pr * (size_t) state->target_count, missing);
    state->h_bmod.assign((size_t) state->target_count, 0);
    for (int_t slot = 0; slot < state->target_count; ++slot) {
        const dSymLDLNVPanelDesc &panel = panels[slot];
        state->h_bmod[(size_t) slot] = (int) panel.block_count;
        if (panel.block_count > 0)
            local_last[(size_t) slot] = std::max(
                local_last[(size_t) slot], panel.gid + 1);
        if (myrow == state->trf3Dpartition->symV2DiagRoot[panel.gid])
            local_last[(size_t) slot] = std::max(
                local_last[(size_t) slot], panel.gid);
    }
    MPI_Allgather(local_last.data(), (int) state->target_count, mpi_int_t,
                  all_last.data(), (int) state->target_count, mpi_int_t,
                  grid->cscp.comm);

    trees.resize((size_t) state->target_count);
    state->h_statusmod.assign((size_t) state->target_count * 2, 1);
    state->h_recv_cnt.assign((size_t) state->target_count, 0);
    for (int_t slot = 0; slot < state->target_count; ++slot) {
        C_RdTree_Nullify(&trees[(size_t) slot]);
        int_t target = state->shared->graph->panel_gids[(size_t) slot];
        int root = state->trf3Dpartition->symV2DiagRoot[target];
        std::vector<std::pair<int_t, int> > ordered;
        for (int pr = 0; pr < Pr; ++pr) {
            int_t last = all_last[(size_t) pr * (size_t) state->target_count +
                                  (size_t) slot];
            if (pr != root && last != missing)
                ordered.push_back(std::make_pair(last, pr));
        }
        std::sort(ordered.begin(), ordered.end(),
                  [](const std::pair<int_t, int> &a,
                     const std::pair<int_t, int> &b) {
                      return a.first > b.first;
                  });
        std::vector<int> ranks;
        ranks.push_back(PNUM(root, mycol, grid));
        for (size_t i = 0; i < ordered.size(); ++i)
            ranks.push_back(PNUM(ordered[i].second, mycol, grid));
        int active = local_last[(size_t) slot] != missing;
        if (active && ranks.size() > 1) {
            int needrecv = 0;
            int needsend = 0;
            C_RdTree_Create_nv(&trees[(size_t) slot], state->layer_comm,
                               ranks.data(), (int) ranks.size(),
                               state->host_xsup[target + 1] -
                                   state->host_xsup[target],
                               'd', &needrecv, &needsend);
            trees[(size_t) slot].tag_ = RD_U;
            state->expected_reduce_sends += needsend;
            if (needrecv > 0) {
                state->h_statusmod[(size_t) slot * 2] = 0;
                state->h_statusmod[(size_t) slot * 2 + 1] = 0;
                state->h_recv_cnt[(size_t) slot] = needrecv;
                state->h_colnummod.push_back((int) slot);
                state->expected_reduce_recvs += needrecv;
                state->h_bmod[(size_t) slot] += needrecv;
            }
        }
    }
}

static void
symldl_nvshmem_tree_from_graph(
    const dSymLDLTreeNode *node, MPI_Comm comm, int my_rank,
    int msg_size, int tag, C_Tree *tree)
{
    C_BcTree_Nullify(tree);
    tree->myIdx = -1;
    if (node == NULL || !node->active ||
        (node->parent_rank < 0 && node->child_count == 0))
        return;
    if (node->rank_index < 0 || node->root_rank < 0 ||
        node->child_count > 2 || msg_size <= 0)
        ABORT("SymLDL NVSHMEM solve tree descriptor is invalid.");

    tree->comm_ = comm;
    tree->myRoot_ = node->parent_rank >= 0 ? node->parent_rank : my_rank;
    tree->destCnt_ = node->child_count;
    tree->myDests_[0] = node->children[0];
    tree->myDests_[1] = node->children[1];
    tree->myRank_ = my_rank;
    tree->msgSize_ = msg_size;
    tree->tag_ = tag;
    tree->empty_ = NO;
    tree->type_ = MPI_DOUBLE;
    tree->myIdx = node->rank_index;
    tree->sendRequests_[0] = MPI_REQUEST_NULL;
    tree->sendRequests_[1] = MPI_REQUEST_NULL;
}

static void
symldl_nvshmem_build_forward_trees_from_graph(
    dSymLDLNVSHMEMForwardState *state,
    std::vector<C_Tree> &bcast_trees,
    std::vector<C_Tree> &reduce_trees)
{
    const dSymLDLSolveGraph *graph = state->shared->graph;
    bcast_trees.resize((size_t) graph->panel_count);
    reduce_trees.resize((size_t) graph->row_count);
    state->h_status.assign((size_t) graph->panel_count, 1);
    state->h_statusmod.assign((size_t) graph->row_count * 2, 1);
    state->h_recv_cnt.assign((size_t) graph->row_count, 0);
    state->h_fmod.assign((size_t) graph->row_count, 0);
    if (graph->row_count > 0)
        std::copy(
            graph->forward_local_dependencies,
            graph->forward_local_dependencies + graph->row_count,
            state->h_fmod.begin());

    for (int_t slot = 0; slot < graph->panel_count; ++slot) {
        int_t gid = graph->panel_gids[slot];
        const dSymLDLTreeNode *node = &graph->forward_bcast[slot];
        symldl_nvshmem_tree_from_graph(
            node, state->layer_comm, state->rank,
            (int) (graph->xsup[gid + 1] - graph->xsup[gid]), BC_L,
            &bcast_trees[(size_t) slot]);
        if (!bcast_trees[(size_t) slot].empty_ &&
            node->parent_rank >= 0) {
            state->h_status[(size_t) slot] = 0;
            state->h_colnum.push_back((int) slot);
            ++state->expected_bcast_recvs;
        }
    }

    for (int_t slot = 0; slot < graph->row_count; ++slot) {
        int_t gid = graph->row_gids[slot];
        const dSymLDLTreeNode *node = &graph->forward_reduce[slot];
        symldl_nvshmem_tree_from_graph(
            node, state->layer_comm, state->rank,
            (int) (graph->xsup[gid + 1] - graph->xsup[gid]), RD_L,
            &reduce_trees[(size_t) slot]);
        state->expected_reduce_sends += node->parent_rank >= 0;
        if (!reduce_trees[(size_t) slot].empty_ &&
            node->child_count > 0) {
            state->h_statusmod[(size_t) slot * 2] = 0;
            state->h_statusmod[(size_t) slot * 2 + 1] = 0;
            state->h_recv_cnt[(size_t) slot] = node->child_count;
            state->h_colnummod.push_back((int) slot);
            state->expected_reduce_recvs += node->child_count;
            state->h_fmod[(size_t) slot] += node->child_count;
        }
    }
}

static void
symldl_nvshmem_build_backward_trees_from_graph(
    dSymLDLNVSHMEMBackwardState *state,
    std::vector<C_Tree> &bcast_trees,
    std::vector<C_Tree> &reduce_trees)
{
    const dSymLDLSolveGraph *graph = state->shared->graph;
    bcast_trees.resize((size_t) graph->row_count);
    reduce_trees.resize((size_t) graph->panel_count);
    state->h_status.assign((size_t) graph->row_count, 1);
    state->h_source_active.assign((size_t) graph->row_count, 0);
    state->h_statusmod.assign((size_t) graph->panel_count * 2, 1);
    state->h_recv_cnt.assign((size_t) graph->panel_count, 0);
    state->h_bmod.assign((size_t) graph->panel_count, 0);
    if (graph->panel_count > 0)
        std::copy(
            graph->backward_local_dependencies,
            graph->backward_local_dependencies + graph->panel_count,
            state->h_bmod.begin());

    for (int_t slot = 0; slot < graph->row_count; ++slot) {
        int_t gid = graph->row_gids[slot];
        const dSymLDLTreeNode *node = &graph->backward_bcast[slot];
        state->h_source_active[(size_t) slot] = node->active;
        symldl_nvshmem_tree_from_graph(
            node, state->layer_comm, state->rank,
            (int) (graph->xsup[gid + 1] - graph->xsup[gid]), BC_U,
            &bcast_trees[(size_t) slot]);
        if (!bcast_trees[(size_t) slot].empty_ &&
            node->parent_rank >= 0) {
            state->h_status[(size_t) slot] = 0;
            state->h_colnum.push_back((int) slot);
            ++state->expected_bcast_recvs;
        }
    }

    for (int_t slot = 0; slot < graph->panel_count; ++slot) {
        int_t gid = graph->panel_gids[slot];
        const dSymLDLTreeNode *node = &graph->backward_reduce[slot];
        symldl_nvshmem_tree_from_graph(
            node, state->layer_comm, state->rank,
            (int) (graph->xsup[gid + 1] - graph->xsup[gid]), RD_U,
            &reduce_trees[(size_t) slot]);
        state->expected_reduce_sends += node->parent_rank >= 0;
        if (!reduce_trees[(size_t) slot].empty_ &&
            node->child_count > 0) {
            state->h_statusmod[(size_t) slot * 2] = 0;
            state->h_statusmod[(size_t) slot * 2 + 1] = 0;
            state->h_recv_cnt[(size_t) slot] = node->child_count;
            state->h_colnummod.push_back((int) slot);
            state->expected_reduce_recvs += node->child_count;
            state->h_bmod[(size_t) slot] += node->child_count;
        }
    }
}

static int
symldl_nvshmem_backward_value_trace(
    dSymLDLNVSHMEMBackwardState *state, const char *phase)
{
    const char *trace_gid_text = getenv(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_BACKWARD_TRACE_GID");
    const char *trace_peer_text = getenv(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_BACKWARD_TRACE_PEER_GID");
    int_t trace_gids[2] = {-1, -1};
    if (trace_gid_text != NULL && trace_gid_text[0] != '\0')
        trace_gids[0] = (int_t) strtoll(trace_gid_text, NULL, 10);
    if (trace_peer_text != NULL && trace_peer_text[0] != '\0')
        trace_gids[1] = (int_t) strtoll(trace_peer_text, NULL, 10);
    int mycol = MYCOL(state->grid->iam, state->grid);
    for (int which = 0; which < 2; ++which) {
        int_t gid = trace_gids[which];
        if (gid < 0 || gid >= state->nsupers ||
            mycol != state->trf3Dpartition->symV2PanelRoot[gid])
            continue;
        int_t row_slot =
            state->trf3Dpartition->symV2RowLocalIndex[gid];
        if (row_slot < 0)
            continue;
        int_t width = state->host_xsup[gid + 1] - state->host_xsup[gid];
        int_t offset = state->host_ilsum[row_slot] * state->nrhs +
                       (row_slot + 1) * XK_H;
        int_t count = width * state->nrhs;
        std::vector<double> values((size_t) count);
        CUDA_CHECK(cudaMemcpy(values.data(), state->d_x + offset,
                              (size_t) count * sizeof(double),
                              cudaMemcpyDeviceToHost));
        double sum = 0.0;
        double norm = 0.0;
        for (int_t i = 0; i < count; ++i) {
            sum += values[(size_t) i];
            norm += values[(size_t) i] * values[(size_t) i];
        }
        fprintf(stderr,
                "SymLDL NVSHMEM 3D backward target values phase=%s "
                "rank=%d z=%d gid=%lld count=%lld sum=%.17e "
                "norm=%.17e first=%.17e\n",
                phase, state->grid3d->iam, state->grid3d->zscp.Iam,
                (long long) gid, (long long) count, sum, sqrt(norm),
                count > 0 ? values[0] : 0.0);
        fflush(stderr);
    }
    if (!state->diagnostics)
        return 0;
    std::vector<double> host_x((size_t) state->x_count);
    CUDA_CHECK(cudaMemcpy(host_x.data(), state->d_x,
                          host_x.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double local[4] = {0.0, 0.0, 0.0, 0.0};
    int_t local_count = 0;
    for (int_t slot = 0; slot < state->source_count; ++slot) {
        int_t gid = state->shared->graph->row_gids[(size_t) slot];
        if (mycol != state->trf3Dpartition->symV2PanelRoot[gid])
            continue;
        int_t lr = state->trf3Dpartition->symV2RowLocalIndex[gid];
        if (lr < 0)
            ABORT("SymLDL NVSHMEM 3D backward diagnostic row is missing.");
        int_t width = state->host_xsup[gid + 1] - state->host_xsup[gid];
        int_t offset = state->host_ilsum[lr] * state->nrhs +
                       (lr + 1) * XK_H;
        int_t count = width * state->nrhs;
        if (offset < 0 || count < 0 ||
            offset > state->x_count || count > state->x_count - offset)
            ABORT("SymLDL NVSHMEM 3D backward diagnostic X range is invalid.");
        for (int_t i = 0; i < count; ++i) {
            double value = host_x[(size_t) (offset + i)];
            local[0] += value;
            local[1] += fabs(value);
            local[2] += value * value;
            local[3] = std::max(local[3], fabs(value));
        }
        local_count += count;
    }
    double global[4] = {0.0, 0.0, 0.0, 0.0};
    int_t global_count = 0;
    MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_SUM,
                  state->layer_comm);
    MPI_Allreduce(local + 3, global + 3, 1, MPI_DOUBLE, MPI_MAX,
                  state->layer_comm);
    MPI_Allreduce(&local_count, &global_count, 1, mpi_int_t, MPI_SUM,
                  state->layer_comm);
    if (state->rank == 0) {
        fprintf(stderr,
                "SymLDL NVSHMEM 3D backward values phase=%s count=%lld "
                "sum=%.17e abs=%.17e norm=%.17e max=%.17e\n",
                phase, (long long) global_count, global[0], global[1],
                sqrt(global[2]), global[3]);
        fflush(stderr);
    }
    return 0;
}

static int
symldl_nvshmem_check_backward(
    dSymLDLNVSHMEMBackwardState *state)
{
    if (!state->diagnostics)
        return 0;
    std::vector<int> bmod((size_t) state->target_count);
    std::vector<int> status((size_t) state->source_count);
    std::vector<int> statusmod((size_t) state->target_count * 2);
    if (!bmod.empty())
        CUDA_CHECK(cudaMemcpy(bmod.data(), state->d_bmod,
                              bmod.size() * sizeof(int),
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
    for (size_t i = 0; i < bmod.size(); ++i)
        blocked += bmod[i] != 0;
    for (size_t i = 0; i < state->h_colnum.size(); ++i)
        bcast_missing += status[(size_t) state->h_colnum[i]] != 1;
    for (size_t i = 0; i < state->h_colnummod.size(); ++i) {
        int slot = state->h_colnummod[i];
        int got = statusmod[(size_t) slot * 2] +
                  statusmod[(size_t) slot * 2 + 1];
        reduce_missing += got != state->h_recv_cnt[(size_t) slot];
    }
    fprintf(stderr,
            "SymLDL NVSHMEM 3D backward trace rank=%d z=%d layer_npes=%d "
            "bcast_expected=%d bcast_missing=%d reduce_expected=%d "
            "reduce_missing=%d bmod_nonzero=%d\n",
            state->grid3d->iam, state->grid3d->zscp.Iam, state->nprocs,
            state->expected_bcast_recvs, bcast_missing,
            state->expected_reduce_recvs, reduce_missing, blocked);
    fflush(stderr);
    return blocked || bcast_missing || reduce_missing ? -1 : 0;
}

static int
symldl_nvshmem_check_forward(
    dSymLDLNVSHMEMForwardState *state)
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
            "SymLDL NVSHMEM 3D forward trace rank=%d z=%d layer_npes=%d "
            "bcast_expected=%d bcast_missing=%d reduce_expected=%d "
            "reduce_missing=%d fmod_nonzero=%d\n",
            state->grid3d->iam, state->grid3d->zscp.Iam, state->nprocs,
            state->expected_bcast_recvs, bcast_missing,
            state->expected_reduce_recvs, reduce_missing, blocked);
    fflush(stderr);
    return blocked || bcast_missing || reduce_missing ? -1 : 0;
}

} /* namespace */

/*
 * The 3D factorization leaves IN_GRID_ZERO panels as partial workspaces.
 * Algorithm 1 instead solves with L_z, whose ancestor panels are replicas of
 * the finalized panel on the unique IN_GRID_AIJ layer. Refresh retained panels
 * in bounded batches; no panel-wise collective or persistent factor copy is
 * introduced.
 */
static void
symldl_nvshmem_replicate_final_panels(
    dSymLDLNVSHMEMForwardState *state,
    const dSymLDLNVPanelDesc *panels,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d)
{
    if (grid3d->zscp.Np <= 1)
        return;
    if (state->nsupers > INT_MAX)
        ABORT("SymLDL NVSHMEM 3D forward has too many supernodes.");

    int myz = grid3d->zscp.Iam;
    const int_t missing_count = std::numeric_limits<int_t>::max();
    std::vector<int> local_source_count((size_t) state->nsupers, 0);
    std::vector<int> source_count((size_t) state->nsupers, 0);
    std::vector<int> local_source((size_t) state->nsupers, INT_MAX);
    std::vector<int> source((size_t) state->nsupers, INT_MAX);
    std::vector<int_t> local_min_count(
        (size_t) state->nsupers, missing_count);
    std::vector<int_t> min_count((size_t) state->nsupers, missing_count);
    std::vector<int_t> local_max_count((size_t) state->nsupers, -1);
    std::vector<int_t> value_count((size_t) state->nsupers, -1);
    for (int_t k = 0; k < state->nsupers; ++k) {
        int_t lp = trf3Dpartition->symV2PanelLocalIndex[k];
        if (lp < 0)
            continue;
        if (lp >= state->panel_count || panels[lp].gid != k)
            ABORT("SymLDL NVSHMEM 3D forward local panel index is inconsistent.");
        if (!panels[lp].active || panels[lp].values == NULL)
            continue;
        if (panels[lp].value_count <= 0)
            ABORT("SymLDL NVSHMEM 3D forward retained panel is empty.");
        local_min_count[(size_t) k] = panels[lp].value_count;
        local_max_count[(size_t) k] = panels[lp].value_count;
        if (trf3Dpartition->superGridMap[k] == IN_GRID_AIJ) {
            local_source_count[(size_t) k] = 1;
            local_source[(size_t) k] = myz;
        }
    }
    MPI_Allreduce(local_source_count.data(), source_count.data(),
                  (int) state->nsupers, MPI_INT, MPI_SUM,
                  grid3d->zscp.comm);
    MPI_Allreduce(local_source.data(), source.data(), (int) state->nsupers,
                  MPI_INT, MPI_MIN, grid3d->zscp.comm);
    MPI_Allreduce(local_min_count.data(), min_count.data(),
                  (int) state->nsupers, mpi_int_t, MPI_MIN,
                  grid3d->zscp.comm);
    MPI_Allreduce(local_max_count.data(), value_count.data(),
                  (int) state->nsupers, mpi_int_t, MPI_MAX,
                  grid3d->zscp.comm);

    int source_panels = 0;
    int_t first_panel = state->nsupers;
    for (int_t k = 0; k < state->nsupers; ++k) {
        if (value_count[(size_t) k] < 0)
            continue;
        if (source_count[(size_t) k] != 1 || source[(size_t) k] == INT_MAX)
            ABORT("SymLDL NVSHMEM 3D forward finalized panel source is not unique.");
        if (min_count[(size_t) k] != value_count[(size_t) k])
            ABORT("SymLDL NVSHMEM 3D forward panel layouts differ across Z layers.");
        ++source_panels;
        first_panel = std::min(first_panel, k);
    }
    state->panel_replication_sources = source_panels;
    if (first_panel == state->nsupers)
        return;

    const size_t batch_capacity = 8U * 1024U * 1024U;
    double *batch = NULL;
    CUDA_CHECK(cudaMallocHost((void **) &batch,
                              batch_capacity * sizeof(double)));
    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int_t k = first_panel;
    int_t panel_offset = 0;
    while (k < state->nsupers) {
        struct Segment { int_t gid; int_t offset; int_t count; };
        std::vector<Segment> segments;
        size_t used = 0;
        while (k < state->nsupers && used < batch_capacity) {
            int_t count = value_count[(size_t) k];
            if (count < 0) {
                ++k;
                panel_offset = 0;
                continue;
            }
            int_t remaining = count - panel_offset;
            if (remaining <= 0) {
                ++k;
                panel_offset = 0;
                continue;
            }
            int_t take = (int_t) std::min(
                (size_t) remaining, batch_capacity - used);
            segments.push_back(Segment{k, panel_offset, take});
            used += (size_t) take;
            panel_offset += take;
            if (panel_offset == count) {
                ++k;
                panel_offset = 0;
            }
        }
        if (used == 0)
            continue;
        memset(batch, 0, used * sizeof(double));
        size_t batch_offset = 0;
        for (size_t s = 0; s < segments.size(); ++s) {
            const Segment &segment = segments[s];
            int_t local_panel = trf3Dpartition->symV2PanelLocalIndex[
                segment.gid];
            if (myz == source[(size_t) segment.gid]) {
                if (local_panel < 0 || local_panel >= state->panel_count ||
                    panels[local_panel].gid != segment.gid ||
                    !panels[local_panel].active ||
                    panels[local_panel].values == NULL ||
                    !symldl_nvshmem_range_valid(
                        segment.offset, segment.count,
                        panels[local_panel].value_count))
                    ABORT("SymLDL NVSHMEM 3D forward source panel storage is unavailable.");
                CUDA_CHECK(cudaMemcpyAsync(
                    batch + batch_offset,
                    panels[local_panel].values + segment.offset,
                    (size_t) segment.count * sizeof(double),
                    cudaMemcpyDeviceToHost, stream));
            }
            batch_offset += (size_t) segment.count;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Allreduce(MPI_IN_PLACE, batch, (int) used, MPI_DOUBLE, MPI_SUM,
                      grid3d->zscp.comm);
        batch_offset = 0;
        for (size_t s = 0; s < segments.size(); ++s) {
            const Segment &segment = segments[s];
            int_t local_panel = trf3Dpartition->symV2PanelLocalIndex[
                segment.gid];
            if (local_panel >= 0) {
                if (local_panel >= state->panel_count ||
                    panels[local_panel].gid != segment.gid)
                    ABORT("SymLDL NVSHMEM 3D forward target panel index is inconsistent.");
                const dSymLDLNVPanelDesc &panel = panels[local_panel];
                if (!panel.active || panel.values == NULL ||
                    panel.value_count != value_count[(size_t) segment.gid] ||
                    !symldl_nvshmem_range_valid(
                        segment.offset, segment.count, panel.value_count))
                    ABORT("SymLDL NVSHMEM 3D forward target panel storage is unavailable.");
                CUDA_CHECK(cudaMemcpyAsync(
                    panel.values + segment.offset, batch + batch_offset,
                    (size_t) segment.count * sizeof(double),
                    cudaMemcpyHostToDevice, stream));
            }
            batch_offset += (size_t) segment.count;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++state->panel_replication_batches;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(batch));
}

static dSymLDLNVSHMEMForwardState *
symldl_nvshmem_forward_create(
    int nrhs, int_t x_count, int_t lsum_count,
    const dSymLDLSolveGraph *graph,
    const dSymLDLNVPanelDesc *device_panels,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d)
{
#ifndef HAVE_NVSHMEM
    (void) nrhs; (void) x_count; (void) lsum_count; (void) graph;
    (void) device_panels; (void) trf3Dpartition; (void) grid3d;
    return NULL;
#else
    int_t n = graph != NULL ? graph->n : 0;
    int_t nsupers = graph != NULL ? graph->nsupers : 0;
    int_t panel_count = graph != NULL ? graph->panel_count : 0;
    int_t block_count = graph != NULL ? graph->block_count : 0;
    int_t factor_row_count = graph != NULL ? graph->factor_row_count : 0;
    const dSymLDLNVPanelDesc *panels = device_panels;
    const dSymLDLNVBlockDesc *blocks = graph != NULL ? graph->blocks : NULL;
    const int_t *rows = graph != NULL ? graph->rows : NULL;
    const int_t *xsup = graph != NULL ? graph->xsup : NULL;
    const int_t *ilsum = graph != NULL ? graph->ilsum : NULL;
    if (n <= 0 || nsupers <= 0 || nrhs <= 0 || x_count <= 0 ||
        lsum_count <= 0 || panel_count < 0 ||
        (panel_count > 0 && panels == NULL) ||
        block_count < 0 || (block_count > 0 && blocks == NULL) ||
        factor_row_count < 0 ||
        (factor_row_count > 0 && rows == NULL) ||
        xsup == NULL || ilsum == NULL || graph == NULL ||
        trf3Dpartition == NULL ||
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
    dSymLDLNVSHMEMForwardState *state =
        new dSymLDLNVSHMEMForwardState();
    state->n = n;
    state->nsupers = nsupers;
    state->nrhs = nrhs;
    state->x_count = x_count;
    state->lsum_count = lsum_count;
    state->panel_count = panel_count;
    state->block_count = block_count;
    state->factor_row_count = factor_row_count;
    state->row_count = trf3Dpartition->symV2LocalRowCount;
    state->trf3Dpartition = trf3Dpartition;
    state->grid3d = grid3d;
    state->grid = &grid3d->grid2d;
    state->layer_comm = state->grid->comm;
    state->host_xsup = xsup;
    state->host_ilsum = ilsum;
    state->diagnostics = symldl_nvshmem_env_enabled(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_FORWARD_DIAGNOSTICS");
    MPI_Comm_rank(state->layer_comm, &state->rank);
    MPI_Comm_size(state->layer_comm, &state->nprocs);

    if (panel_count != trf3Dpartition->symV2LocalPanelCount)
        ABORT("SymLDL NVSHMEM 3D forward panel count is inconsistent.");
    state->h_panel_gids = graph->panel_gids;
    state->h_row_gids = graph->row_gids;
    for (int_t lp = 0; lp < panel_count; ++lp)
        if (panels[lp].gid != state->h_panel_gids[(size_t) lp])
            ABORT("SymLDL NVSHMEM 3D forward panels are not in local-index order.");
    for (int_t lp = 0; lp < panel_count; ++lp)
        symldl_nvshmem_validate_panel(
            state, lp, panels[lp], blocks, rows);

    symldl_nvshmem_replicate_final_panels(
        state, panels, trf3Dpartition, grid3d);
    state->shared = symldl_nvshmem_shared_create(
        nrhs, x_count, lsum_count, graph, panels,
        trf3Dpartition, grid3d);
    if (state->shared == NULL || state->shared->flag_bc_q == NULL ||
        state->shared->flag_rd_q == NULL ||
        state->shared->ready_x == NULL ||
        state->shared->ready_lsum == NULL ||
        state->shared->my_flag_bc == NULL ||
        state->shared->my_flag_rd == NULL) {
        symldl_nvshmem_state_free(state);
        return NULL;
    }
    state->panel_slots = state->shared->panel_slots;
    state->row_slots = state->shared->row_slots;
    state->maxsup = state->shared->maxsup;
    state->maxrecvsz = state->shared->maxrecvsz;

    int myrow = MYROW(state->grid->iam, state->grid);
    int mycol = MYCOL(state->grid->iam, state->grid);
    state->h_rhs_offsets.push_back(0);
    for (int_t k = 0; k < nsupers; ++k) {
        if (myrow != trf3Dpartition->symV2DiagRoot[k] ||
            mycol != trf3Dpartition->symV2PanelRoot[k])
            continue;
        int_t width = symldl_nvshmem_super_size(state, k);
        size_t count = (size_t) width * (size_t) nrhs + (size_t) XK_H;
        size_t next = (size_t) state->h_rhs_offsets.back() + count;
        if (next > (size_t) std::numeric_limits<int_t>::max())
            ABORT("SymLDL NVSHMEM 3D forward RHS metadata overflows.");
        state->h_rhs_gids.push_back(k);
        state->h_rhs_offsets.push_back((int_t) next);
    }

    std::vector<C_Tree> bcast_trees;
    std::vector<C_Tree> reduce_trees;
    symldl_nvshmem_build_forward_trees_from_graph(
        state, bcast_trees, reduce_trees);

    int panel_diagnostics = symldl_nvshmem_env_enabled(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_PANEL_DIAGNOSTICS");
    for (int_t lp = 0; lp < panel_count; ++lp) {
        if (!panels[lp].active || panels[lp].values == NULL)
            continue;

        int_t k = panels[lp].gid;
        if (panel_diagnostics &&
            trf3Dpartition->supernode2treeMap[k] == 0) {
            std::vector<double> values((size_t) panels[lp].value_count);
            CUDA_CHECK(cudaMemcpy(values.data(), panels[lp].values,
                                  values.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            int myrow = MYROW(state->grid->iam, state->grid);
            int row_begin = myrow == trf3Dpartition->symV2DiagRoot[k]
                                ? symldl_nvshmem_super_size(state, k)
                                : 0;
            int_t nsupr = panels[lp].nsupr;
            int_t width = symldl_nvshmem_super_size(state, k);
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
                    "SymLDL NVSHMEM 3D root-panel trace: rank=%d z=%d "
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
    if (state->diagnostics) {
        fprintf(stderr,
                "SymLDL NVSHMEM 3D setup trace: rank=%d z=%d "
                "panels=%lld blocks=%lld rows=%lld local_rows=%lld "
                "replica_sources=%d replica_batches=%d bcast_recvs=%d "
                "reduce_recvs=%d reduce_sends=%d\n",
                grid3d->iam, grid3d->zscp.Iam,
                (long long) panel_count, (long long) block_count,
                (long long) factor_row_count, (long long) state->row_count,
                state->panel_replication_sources,
                state->panel_replication_batches,
                state->expected_bcast_recvs,
                state->expected_reduce_recvs,
                state->expected_reduce_sends);
        fflush(stderr);
    }

    state->d_x = state->shared->d_x;
    state->d_lsum = state->shared->d_lsum;
    state->d_ilsum = state->shared->d_ilsum;
    state->d_xsup = state->shared->d_xsup;
    state->d_panels = state->shared->d_panels;
    state->d_blocks = state->shared->d_blocks;
    state->d_rows = state->shared->d_rows;
    state->d_panel_gids = state->shared->d_panel_gids;
    state->d_row_gids = state->shared->d_row_gids;
    state->d_row_local_index = state->shared->d_row_local_index;
    state->d_diag_roots = state->shared->d_diag_roots;
    state->d_grid = state->shared->d_grid;
    state->flag_bc_q = state->shared->flag_bc_q;
    state->flag_rd_q = state->shared->flag_rd_q;
    state->ready_x = state->shared->ready_x;
    state->ready_lsum = state->shared->ready_lsum;
    state->my_flag_bc = state->shared->my_flag_bc;
    state->my_flag_rd = state->shared->my_flag_rd;

    symldl_nvshmem_cuda_copy(&state->d_fmod, state->h_fmod.data(),
                                state->h_fmod.size());
    symldl_nvshmem_cuda_copy(&state->d_bcast_trees,
                                bcast_trees.data(), bcast_trees.size());
    symldl_nvshmem_cuda_copy(&state->d_reduce_trees,
                                reduce_trees.data(), reduce_trees.size());
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
    symldl_nvshmem_cuda_copy(&state->d_nfrecv, state->h_nfrecv, 3);
    symldl_nvshmem_cuda_copy(&state->d_status,
                                state->h_status.data(),
                                state->h_status.size());
    symldl_nvshmem_cuda_copy(&state->d_colnum,
                                state->h_colnum.data(),
                                state->h_colnum.size());
    symldl_nvshmem_cuda_alloc(&state->d_mynum, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_mymaskstart, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_mymasklength, 1024);
    symldl_nvshmem_cuda_copy(&state->d_nfrecvmod,
                                state->h_nfrecvmod, 4);
    symldl_nvshmem_cuda_copy(&state->d_statusmod,
                                state->h_statusmod.data(),
                                state->h_statusmod.size());
    symldl_nvshmem_cuda_copy(&state->d_colnummod,
                                state->h_colnummod.data(),
                                state->h_colnummod.size());
    symldl_nvshmem_cuda_alloc(&state->d_mynummod, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_mymaskstartmod, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_mymasklengthmod, 1024);
    symldl_nvshmem_cuda_copy(&state->d_recv_cnt,
                                state->h_recv_cnt.data(),
                                state->h_recv_cnt.size());
    symldl_nvshmem_cuda_alloc(&state->d_msgnum, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_flag_mod,
                                 (size_t) state->expected_reduce_sends + 1);

    state->setup_time = SuperLU_timer_() - setup_start;
    return (dSymLDLNVSHMEMForwardState *) state;
#endif
}

static int
symldl_nvshmem_forward_initialize_rhs(
    dSymLDLNVSHMEMForwardState * handle, double *x, int_t x_count)
{
    dSymLDLNVSHMEMForwardState *state =
        (dSymLDLNVSHMEMForwardState *) handle;
    if (state == NULL || x == NULL || x_count != state->x_count)
        return -1;

    double setup_start = SuperLU_timer_();
    int_t packed_count = state->h_rhs_offsets.empty()
                             ? 0
                             : state->h_rhs_offsets.back();
    std::vector<double> x_source((size_t) packed_count, 0.0);

    /*
     * This is the selected-owner equivalent of dtrs_B_init3d_newsolve():
     * collect the unique diagonal-owner blocks in one Z operation, zero every
     * replica, then restore only the forest nodes active on this layer.
     */
    for (size_t pos = 0; pos < state->h_rhs_gids.size(); ++pos) {
        int_t k = state->h_rhs_gids[pos];
        int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
        int_t count = state->h_rhs_offsets[pos + 1] -
                      state->h_rhs_offsets[pos];
        if (state->grid3d->iam ==
            state->trf3Dpartition->symV2DiagOwner[k]) {
            if (lr < 0)
                return -1;
            int_t ii = symldl_nvshmem_x_block(state, lr);
            std::copy(&x[ii - XK_H], &x[ii - XK_H] + count,
                      &x_source[(size_t) state->h_rhs_offsets[pos]]);
        }
    }
    const int_t collective_chunk = 8 * 1024 * 1024;
    for (int_t begin = 0; begin < packed_count; begin += collective_chunk) {
        int_t count = std::min(collective_chunk, packed_count - begin);
        MPI_Allreduce(MPI_IN_PLACE, x_source.data() + begin, (int) count,
                      MPI_DOUBLE, MPI_SUM, state->grid3d->zscp.comm);
    }

    std::fill(x, x + x_count, 0.0);
    for (size_t pos = 0; pos < state->h_rhs_gids.size(); ++pos) {
        int_t k = state->h_rhs_gids[pos];
        int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
        if (lr >= 0) {
            int_t ii = symldl_nvshmem_x_block(state, lr);
            std::copy(&x_source[(size_t) state->h_rhs_offsets[pos]],
                      &x_source[(size_t) state->h_rhs_offsets[pos]] + XK_H,
                      &x[ii - XK_H]);
        }
    }

    int_t max_level = log2i(state->grid3d->zscp.Np) + 1;
    int_t *tree_ids = state->trf3Dpartition->myTreeIdxs;
    int_t *zero_ids = state->trf3Dpartition->myZeroTrIdxs;
    int myrow = MYROW(state->grid->iam, state->grid);
    int mycol = MYCOL(state->grid->iam, state->grid);
    for (int_t level = 0; level < max_level; ++level) {
        if (zero_ids[level])
            continue;
        sForest_t *forest = state->trf3Dpartition->sForests[
            tree_ids[level]];
        if (forest == NULL)
            continue;
        for (int_t node = 0; node < forest->nNodes; ++node) {
            int_t k = forest->nodeList[node];
            if (myrow != state->trf3Dpartition->symV2DiagRoot[k] ||
                mycol != state->trf3Dpartition->symV2PanelRoot[k])
                continue;
            int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
            if (lr < 0)
                ABORT("SymLDL NVSHMEM 3D forward RHS forest is missing a local row.");
            std::vector<int_t>::const_iterator it = std::lower_bound(
                state->h_rhs_gids.begin(), state->h_rhs_gids.end(), k);
            if (it == state->h_rhs_gids.end() || *it != k)
                ABORT("SymLDL NVSHMEM 3D forward RHS forest is missing packed data.");
            size_t pos = (size_t) (it - state->h_rhs_gids.begin());
            int_t ii = symldl_nvshmem_x_block(state, lr);
            int_t count = symldl_nvshmem_super_size(state, k) *
                          state->nrhs;
            const double *source =
                &x_source[(size_t) state->h_rhs_offsets[pos] + XK_H];
            std::copy(source, source + count, &x[ii]);
        }
    }
    state->setup_time += SuperLU_timer_() - setup_start;
    return 0;
}

static int
symldl_nvshmem_forward_solve(
    dSymLDLNVSHMEMForwardState * handle, double *x, int_t x_count)
{
    dSymLDLNVSHMEMForwardState *state =
        (dSymLDLNVSHMEMForwardState *) handle;
    if (state == NULL || x == NULL || x_count != state->x_count)
        return -1;
#ifndef HAVE_NVSHMEM
    return -1;
#else
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
    double forward_start = SuperLU_timer_();
    if (symldl_nvshmem_forward_wrap(
            (int) state->panel_count, (int) state->row_count,
            state->d_lsum, state->d_x, state->nrhs, state->maxsup,
            state->nsupers, state->d_fmod, state->d_bcast_trees,
            state->d_reduce_trees, state->d_ilsum, state->d_panels,
            state->d_blocks, state->d_rows, state->d_xsup, state->d_grid,
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
    state->forward_time += SuperLU_timer_() - forward_start;

    copy_start = SuperLU_timer_();
    CUDA_CHECK(cudaMemcpy(x, state->d_x,
                          (size_t) x_count * sizeof(double),
                          cudaMemcpyDeviceToHost));
    state->d2h_time += SuperLU_timer_() - copy_start;
    return symldl_nvshmem_check_forward(state);
#endif
}

static int
symldl_nvshmem_forward_sparse_allreduce(
    dSymLDLNVSHMEMForwardState * handle, double *x, int_t x_count)
{
    dSymLDLNVSHMEMForwardState *state =
        (dSymLDLNVSHMEMForwardState *) handle;
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
            int_t ii = symldl_nvshmem_x_block(state, lr);
            int_t count = symldl_nvshmem_super_size(state, k) *
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
                    "SymLDL NVSHMEM 3D pre-sparse trace: z=%d count=%lld "
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
            symldl_nvshmem_z_reduce_tree(
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
            symldl_nvshmem_z_bcast_tree(
                state, current, sender, receiver, x);
            current = (current + 1) / 2 - 1;
        }
    }
    state->sparse_broadcast_time += SuperLU_timer_() - start;

    if (state->diagnostics) {
        double local_replica_error = 0.0;
        size_t packed_count = 0;
        for (size_t pos = 0; pos < state->h_rhs_gids.size(); ++pos) {
            int_t k = state->h_rhs_gids[pos];
            packed_count += (size_t) symldl_nvshmem_super_size(state, k) *
                            (size_t) state->nrhs;
        }
        std::vector<double> replica_max(packed_count, -DBL_MAX);
        std::vector<double> replica_min(packed_count, DBL_MAX);
        size_t packed_offset = 0;
        int local_nonfinite = 0;
        for (size_t pos = 0; pos < state->h_rhs_gids.size(); ++pos) {
            int_t k = state->h_rhs_gids[pos];
            int_t lr = state->trf3Dpartition->symV2RowLocalIndex[k];
            int_t width = symldl_nvshmem_super_size(state, k);
            for (int rhs = 0; rhs < state->nrhs; ++rhs) {
                for (int_t i = 0; i < width; ++i) {
                    if (lr >= 0) {
                        double value = x[
                            symldl_nvshmem_x_block(state, lr) + i +
                            (int_t) rhs * width];
                        replica_max[packed_offset] = value;
                        replica_min[packed_offset] = value;
                        local_nonfinite += !isfinite(value);
                    }
                    ++packed_offset;
                }
            }
        }
        const size_t collective_chunk = 8U * 1024U * 1024U;
        for (size_t begin = 0; begin < packed_count;
             begin += collective_chunk) {
            int count = (int) std::min(collective_chunk,
                                       packed_count - begin);
            MPI_Allreduce(MPI_IN_PLACE, replica_max.data() + begin, count,
                          MPI_DOUBLE, MPI_MAX, state->grid3d->zscp.comm);
            MPI_Allreduce(MPI_IN_PLACE, replica_min.data() + begin, count,
                          MPI_DOUBLE, MPI_MIN, state->grid3d->zscp.comm);
        }
        int missing_replica = 0;
        for (size_t i = 0; i < packed_count; ++i) {
            if (replica_max[i] == -DBL_MAX || replica_min[i] == DBL_MAX) {
                missing_replica = 1;
                continue;
            }
            local_replica_error = std::max(
                local_replica_error, fabs(replica_max[i] - replica_min[i]));
        }
        double global_replica_error = 0.0;
        int global_nonfinite = 0;
        int global_missing = 0;
        MPI_Allreduce(&local_replica_error, &global_replica_error, 1,
                      MPI_DOUBLE, MPI_MAX, state->grid3d->comm);
        MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT,
                      MPI_SUM, state->grid3d->comm);
        MPI_Allreduce(&missing_replica, &global_missing, 1, MPI_INT,
                      MPI_MAX, state->grid3d->comm);
        if (state->grid3d->iam == 0) {
            fprintf(stderr,
                    "SymLDL NVSHMEM 3D sparse-Z trace: reduce_calls=%d "
                    "broadcast_calls=%d replica_error=%.17e "
                    "nonfinite=%d missing=%d\n",
                    state->sparse_reduce_calls,
                    state->sparse_broadcast_calls,
                    global_replica_error, global_nonfinite, global_missing);
            fflush(stderr);
        }
        if (global_replica_error != 0.0 || global_nonfinite != 0 ||
            global_missing != 0)
            return -1;
    }

    return 0;
}

static void
symldl_nvshmem_forward_take_timers(
    dSymLDLNVSHMEMForwardState * handle, double *setup, double *forward,
    double *sparse_reduce, double *sparse_broadcast, double *h2d,
    double *d2h)
{
    dSymLDLNVSHMEMForwardState *state =
        (dSymLDLNVSHMEMForwardState *) handle;
    if (setup) *setup = state ? state->setup_time : 0.0;
    if (forward) *forward = state ? state->forward_time : 0.0;
    if (sparse_reduce)
        *sparse_reduce = state ? state->sparse_reduce_time : 0.0;
    if (sparse_broadcast)
        *sparse_broadcast = state ? state->sparse_broadcast_time : 0.0;
    if (h2d) *h2d = state ? state->h2d_time : 0.0;
    if (d2h) *d2h = state ? state->d2h_time : 0.0;
}

static void
symldl_nvshmem_forward_destroy(
    dSymLDLNVSHMEMForwardState * handle)
{
    symldl_nvshmem_state_free(
        (dSymLDLNVSHMEMForwardState *) handle);
}

static dSymLDLNVSHMEMBackwardState *
symldl_nvshmem_backward_create(
    dSymLDLNVSHMEMForwardState * forward_handle)
{
#ifndef HAVE_NVSHMEM
    (void) forward_handle;
    return NULL;
#else
    dSymLDLNVSHMEMForwardState *forward =
        (dSymLDLNVSHMEMForwardState *) forward_handle;
    if (forward == NULL || forward->shared == NULL)
        return NULL;
    dSymLDLNVSHMEMSharedState *shared = forward->shared;
    if (shared->panel_count > INT_MAX || shared->row_count > INT_MAX ||
        shared->flag_bc_q == NULL || shared->flag_rd_q == NULL ||
        shared->ready_x == NULL || shared->ready_lsum == NULL ||
        shared->my_flag_bc == NULL || shared->my_flag_rd == NULL)
        ABORT("SymLDL NVSHMEM 3D backward local metadata is inconsistent.");

    int_t n = shared->n;
    int_t nsupers = shared->nsupers;
    int nrhs = shared->nrhs;
    int_t x_count = shared->x_count;
    int_t panel_count = shared->panel_count;
    int_t block_count = shared->block_count;
    int_t factor_row_count = shared->factor_row_count;
    const dSymLDLNVPanelDesc *panels = shared->h_panels.data();
    const dSymLDLNVBlockDesc *blocks = shared->graph->blocks;
    const int_t *rows = shared->graph->rows;
    const int_t *xsup = shared->host_xsup;
    const int_t *ilsum = shared->host_ilsum;
    dtrf3Dpartition_t *trf3Dpartition = shared->trf3Dpartition;
    gridinfo3d_t *grid3d = shared->grid3d;

    double setup_start = SuperLU_timer_();
    dSymLDLNVSHMEMBackwardState *state =
        new dSymLDLNVSHMEMBackwardState();
    symldl_nvshmem_shared_retain(shared);
    state->shared = shared;
    state->n = n;
    state->nsupers = nsupers;
    state->nrhs = nrhs;
    state->x_count = x_count;
    state->panel_count = panel_count;
    state->block_count = block_count;
    state->factor_row_count = factor_row_count;
    state->source_count = trf3Dpartition->symV2LocalRowCount;
    state->target_count = panel_count;
    state->trf3Dpartition = trf3Dpartition;
    state->grid3d = grid3d;
    state->grid = &grid3d->grid2d;
    state->layer_comm = state->grid->comm;
    state->host_xsup = xsup;
    state->host_ilsum = ilsum;
    state->diagnostics = symldl_nvshmem_env_enabled(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_BACKWARD_DIAGNOSTICS");
    MPI_Comm_rank(state->layer_comm, &state->rank);
    MPI_Comm_size(state->layer_comm, &state->nprocs);

    std::vector<C_Tree> bcast_trees;
    std::vector<C_Tree> reduce_trees;
    symldl_nvshmem_build_backward_trees_from_graph(
        state, bcast_trees, reduce_trees);

    const char *trace_gid_text = getenv(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_BACKWARD_TRACE_GID");
    const char *trace_peer_text = getenv(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_BACKWARD_TRACE_PEER_GID");
    auto trace_backward_gid = [&](int_t trace_gid, int_t trace_peer) {
        if (trace_gid >= 0 && trace_gid < nsupers) {
            int_t row_slot = trf3Dpartition->symV2RowLocalIndex[trace_gid];
            int_t panel_slot =
                trf3Dpartition->symV2PanelLocalIndex[trace_gid];
            int_t source_edges = row_slot >= 0
                                     ? shared->graph->source_edge_offsets[
                                           (size_t) row_slot + 1] -
                                           shared->graph->source_edge_offsets[
                                               (size_t) row_slot]
                                     : 0;
            int peer_blocks = 0;
            if (panel_slot >= 0 && panel_slot < panel_count) {
                const dSymLDLNVPanelDesc &panel = panels[panel_slot];
                for (int_t b = 0; b < panel.block_count; ++b) {
                    const dSymLDLNVBlockDesc &block =
                        blocks[panel.block_begin + b];
                    peer_blocks +=
                        block.target_gid == trace_peer;
                    fprintf(stderr,
                            "SymLDL NVSHMEM 3D backward panel edge rank=%d "
                            "z=%d panel_gid=%lld target_gid=%lld rows=%lld\n",
                            grid3d->iam, grid3d->zscp.Iam,
                            (long long) trace_gid,
                            (long long) block.target_gid,
                            (long long) block.nbrow);
                }
            }
            if (row_slot >= 0) {
                int_t begin = shared->graph->source_edge_offsets[(size_t) row_slot];
                int_t end =
                    shared->graph->source_edge_offsets[(size_t) row_slot + 1];
                for (int_t pos = begin; pos < end; ++pos) {
                    int_t edge = shared->graph->source_edge_ids[(size_t) pos];
                    fprintf(stderr,
                            "SymLDL NVSHMEM 3D backward source edge rank=%d "
                            "z=%d source_gid=%lld panel_gid=%lld rows=%lld\n",
                            grid3d->iam, grid3d->zscp.Iam,
                            (long long) trace_gid,
                            (long long) panels[blocks[edge].panel_id].gid,
                            (long long) blocks[edge].nbrow);
                }
            }
            const C_Tree *bcast = row_slot >= 0
                                      ? &bcast_trees[(size_t) row_slot]
                                      : NULL;
            const C_Tree *reduce = panel_slot >= 0
                                       ? &reduce_trees[(size_t) panel_slot]
                                       : NULL;
            fprintf(stderr,
                    "SymLDL NVSHMEM 3D backward gid trace rank=%d z=%d "
                    "row=%d col=%d gid=%lld peer=%lld map=%d "
                    "diag_root=%d panel_root=%d row_slot=%lld "
                    "panel_slot=%lld source_edges=%lld panel_blocks=%lld "
                    "peer_blocks=%d source_active=%d bmod=%d "
                    "bcast_empty=%d bcast_parent=%d bcast_rank=%d "
                    "bcast_index=%d bcast_children=%d reduce_empty=%d "
                    "reduce_parent=%d reduce_rank=%d reduce_index=%d "
                    "reduce_children=%d\n",
                    grid3d->iam, grid3d->zscp.Iam,
                    MYROW(state->grid->iam, state->grid),
                    MYCOL(state->grid->iam, state->grid),
                    (long long) trace_gid, (long long) trace_peer,
                    trf3Dpartition->superGridMap[trace_gid],
                    trf3Dpartition->symV2DiagRoot[trace_gid],
                    trf3Dpartition->symV2PanelRoot[trace_gid],
                    (long long) row_slot, (long long) panel_slot,
                    (long long) source_edges,
                    (long long) (panel_slot >= 0 && panel_slot < panel_count
                                     ? panels[panel_slot].block_count : 0),
                    peer_blocks,
                    row_slot >= 0
                        ? state->h_source_active[(size_t) row_slot] : 0,
                    panel_slot >= 0
                        ? state->h_bmod[(size_t) panel_slot] : 0,
                    bcast != NULL ? bcast->empty_ : -1,
                    bcast != NULL ? bcast->myRoot_ : -1,
                    bcast != NULL ? bcast->myRank_ : -1,
                    bcast != NULL ? bcast->myIdx : -1,
                    bcast != NULL ? bcast->destCnt_ : -1,
                    reduce != NULL ? reduce->empty_ : -1,
                    reduce != NULL ? reduce->myRoot_ : -1,
                    reduce != NULL ? reduce->myRank_ : -1,
                    reduce != NULL ? reduce->myIdx : -1,
                    reduce != NULL ? reduce->destCnt_ : -1);
            fflush(stderr);
        }
    };
    if (trace_gid_text != NULL && trace_gid_text[0] != '\0') {
        int_t trace_gid = (int_t) strtoll(trace_gid_text, NULL, 10);
        int_t trace_peer = trace_peer_text != NULL && trace_peer_text[0] != '\0'
                               ? (int_t) strtoll(trace_peer_text, NULL, 10)
                               : -1;
        trace_backward_gid(trace_gid, trace_peer);
        if (trace_peer >= 0 && trace_peer != trace_gid)
            trace_backward_gid(trace_peer, trace_gid);
    }

    if (state->diagnostics) {
        long long local_edges = 0;
        long long initial_bmod_sum = 0;
        int active_sources = 0;
        int nonzero_targets = 0;
        int max_bmod = 0;
        for (int_t slot = 0; slot < state->source_count; ++slot) {
            local_edges +=
                shared->graph->source_edge_offsets[(size_t) slot + 1] -
                shared->graph->source_edge_offsets[(size_t) slot];
            active_sources += state->h_source_active[(size_t) slot] != 0;
        }
        for (int_t slot = 0; slot < state->target_count; ++slot) {
            int value = state->h_bmod[(size_t) slot];
            initial_bmod_sum += value;
            nonzero_targets += value != 0;
            max_bmod = std::max(max_bmod, value);
        }
        fprintf(stderr,
                "SymLDL NVSHMEM 3D backward setup rank=%d z=%d "
                "sources=%lld active_sources=%d targets=%lld edges=%lld "
                "bcast_recvs=%d reduce_recvs=%d reduce_sends=%d "
                "bmod_sum=%lld bmod_nonzero=%d bmod_max=%d\n",
                state->grid3d->iam, state->grid3d->zscp.Iam,
                (long long) state->source_count, active_sources,
                (long long) state->target_count, local_edges,
                state->expected_bcast_recvs, state->expected_reduce_recvs,
                state->expected_reduce_sends, initial_bmod_sum,
                nonzero_targets, max_bmod);
        fflush(stderr);
        if (local_edges != state->block_count)
            ABORT("SymLDL NVSHMEM 3D backward edge count is inconsistent.");
    }

    state->source_slots = shared->row_slots;
    state->target_slots = shared->panel_slots;
    state->maxsup = shared->maxsup;
    state->maxrecvsz = shared->maxrecvsz;

    state->d_x = shared->d_x;
    state->d_lsum = shared->d_lsum;
    state->d_global_ilsum = shared->d_ilsum;
    state->d_target_ilsum = shared->d_target_ilsum;
    state->d_xsup = shared->d_xsup;
    state->d_panels = shared->d_panels;
    state->d_blocks = shared->d_blocks;
    state->d_rows = shared->d_rows;
    state->d_source_gids = shared->d_row_gids;
    state->d_target_gids = shared->d_panel_gids;
    state->d_source_edge_offsets = shared->d_source_edge_offsets;
    state->d_source_edge_ids = shared->d_source_edge_ids;
    state->d_row_local_index = shared->d_row_local_index;
    state->d_panel_local_index = shared->d_panel_local_index;
    state->d_panel_roots = shared->d_panel_roots;
    state->d_grid = shared->d_grid;
    state->flag_bc_q = shared->flag_bc_q;
    state->flag_rd_q = shared->flag_rd_q;
    state->ready_x = shared->ready_x;
    state->ready_lsum = shared->ready_lsum;
    state->my_flag_bc = shared->my_flag_bc;
    state->my_flag_rd = shared->my_flag_rd;

    symldl_nvshmem_cuda_alloc(&state->d_diag_input, (size_t) x_count);
    symldl_nvshmem_cuda_copy(&state->d_bmod, state->h_bmod.data(),
                                state->h_bmod.size());
    symldl_nvshmem_cuda_copy(&state->d_bcast_trees,
                                bcast_trees.data(), bcast_trees.size());
    symldl_nvshmem_cuda_copy(&state->d_reduce_trees,
                                reduce_trees.data(), reduce_trees.size());
    symldl_nvshmem_cuda_copy(&state->d_source_active,
                                state->h_source_active.data(),
                                state->h_source_active.size());

    state->h_nfrecv[0] = (int) state->h_colnum.size();
#ifdef _USE_SUMMIT
    state->h_nfrecv[1] = 32;
    state->h_nfrecv[2] = 8;
#else
    state->h_nfrecv[1] = 1024;
    state->h_nfrecv[2] = 2;
#endif
    int device = 0;
    int multiprocessors = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, device));
    if (multiprocessors <= state->h_nfrecv[2])
        ABORT("SymLDL NVSHMEM 3D backward has no resident worker capacity.");
    state->worker_blocks = multiprocessors - state->h_nfrecv[2];
    const char *worker_blocks_text = getenv(
        "GPU3DV2_SYM_SOLVE_NVSHMEM_BACKWARD_WORKER_BLOCKS");
    if (worker_blocks_text != NULL && worker_blocks_text[0] != '\0') {
        int requested = atoi(worker_blocks_text);
        if (requested <= 0)
            ABORT("SymLDL NVSHMEM 3D backward worker count must be positive.");
        state->worker_blocks = std::min(state->worker_blocks, requested);
    }
    state->h_nfrecvmod[0] = state->expected_reduce_recvs;
    state->h_nfrecvmod[1] = (int) state->h_colnummod.size();
    state->h_nfrecvmod[2] = state->h_nfrecv[2];
    state->h_nfrecvmod[3] = state->expected_reduce_sends;
    symldl_nvshmem_cuda_copy(&state->d_nfrecv, state->h_nfrecv, 3);
    symldl_nvshmem_cuda_copy(&state->d_status, state->h_status.data(),
                                state->h_status.size());
    symldl_nvshmem_cuda_copy(&state->d_colnum, state->h_colnum.data(),
                                state->h_colnum.size());
    symldl_nvshmem_cuda_alloc(&state->d_mynum, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_mymaskstart, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_mymasklength, 1024);
    symldl_nvshmem_cuda_copy(&state->d_nfrecvmod,
                                state->h_nfrecvmod, 4);
    symldl_nvshmem_cuda_copy(&state->d_statusmod,
                                state->h_statusmod.data(),
                                state->h_statusmod.size());
    symldl_nvshmem_cuda_copy(&state->d_colnummod,
                                state->h_colnummod.data(),
                                state->h_colnummod.size());
    symldl_nvshmem_cuda_alloc(&state->d_mynummod, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_mymaskstartmod, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_mymasklengthmod, 1024);
    symldl_nvshmem_cuda_copy(&state->d_recv_cnt,
                                state->h_recv_cnt.data(),
                                state->h_recv_cnt.size());
    symldl_nvshmem_cuda_alloc(&state->d_msgnum, 1024);
    symldl_nvshmem_cuda_alloc(&state->d_flag_mod,
                                 (size_t) state->expected_reduce_sends + 1);
    symldl_nvshmem_cuda_alloc(&state->d_next_source, 1);

    state->setup_time = SuperLU_timer_() - setup_start;
    return (dSymLDLNVSHMEMBackwardState *) state;
#endif
}

static int
symldl_nvshmem_diagonal(
    dSymLDLNVSHMEMBackwardState *state, double *x, int_t x_count)
{
    if (state == NULL || x == NULL || x_count != state->x_count ||
        state->diagonal_ready)
        return -1;
#ifndef HAVE_NVSHMEM
    return -1;
#else
    double start = SuperLU_timer_();
    CUDA_CHECK(cudaMemcpy(state->d_x, x, (size_t) x_count * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state->d_lsum, 0,
                          (size_t) state->shared->backward_lsum_count *
                              sizeof(double)));
    if (!state->h_bmod.empty())
        CUDA_CHECK(cudaMemcpy(state->d_bmod, state->h_bmod.data(),
                              state->h_bmod.size() * sizeof(int),
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
    size_t flag_bc_count = (size_t) RDMA_FLAG_SIZE *
                           ((size_t) state->source_slots + 1);
    size_t flag_rd_count = (size_t) RDMA_FLAG_SIZE *
                           (size_t) state->target_slots * 2;
    CUDA_CHECK(cudaMemset(state->flag_bc_q, 0,
                          flag_bc_count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(state->flag_rd_q, 0,
                          flag_rd_count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(state->ready_x, 0,
                          (size_t) state->maxrecvsz *
                              (size_t) state->source_slots * sizeof(double)));
    CUDA_CHECK(cudaMemset(state->ready_lsum, 0,
                          2 * (size_t) state->maxrecvsz *
                              (size_t) state->target_slots * sizeof(double)));
    CUDA_CHECK(cudaMemset(state->d_msgnum, 0,
                          (size_t) state->h_nfrecv[1] * sizeof(int)));
    CUDA_CHECK(cudaMemset(state->d_next_source, 0, sizeof(int)));
    state->h2d_time += SuperLU_timer_() - start;

    start = SuperLU_timer_();
    CUDA_CHECK(cudaMemcpy(state->d_diag_input, state->d_x,
                          (size_t) x_count * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    if (state->panel_count > 0) {
        symldl_nvshmem_diagonal_kernel<<<
            (int) state->panel_count, 256>>>(
                (int) state->panel_count, state->nrhs, state->d_panels,
                state->d_row_local_index, state->d_global_ilsum,
                state->d_diag_input, state->d_x);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    symldl_nvshmem_backward_value_trace(state, "post-diagonal");
    state->diagonal_time += SuperLU_timer_() - start;
    state->diagonal_ready = 1;
    return 0;
#endif
}

static int
symldl_nvshmem_backward_solve(
    dSymLDLNVSHMEMBackwardState *state, double *x, int_t x_count)
{
    if (state == NULL || x == NULL || x_count != state->x_count ||
        !state->diagonal_ready)
        return -1;
#ifndef HAVE_NVSHMEM
    return -1;
#else
    MPI_Barrier(state->layer_comm);
    double start = SuperLU_timer_();
    if (symldl_nvshmem_backward_wrap(
            (int) state->source_count, (int) state->target_count,
            state->d_lsum, state->d_x,
            state->nrhs, state->maxsup,
            state->nsupers, state->d_next_source, state->worker_blocks,
            state->d_bmod, state->d_bcast_trees,
            state->d_reduce_trees, state->d_global_ilsum,
            state->d_target_ilsum,
            state->d_panels, state->d_blocks, state->d_rows,
            state->d_source_edge_offsets, state->d_source_edge_ids,
            state->d_source_active, state->d_xsup, state->d_grid,
            state->d_source_gids,
            state->d_target_gids,
            state->d_row_local_index, state->d_panel_local_index,
            state->d_panel_roots, state->maxrecvsz, state->flag_bc_q,
            state->flag_rd_q, state->ready_x, state->ready_lsum,
            state->my_flag_bc, state->my_flag_rd, state->d_nfrecv,
            state->h_nfrecv, state->d_status, state->d_colnum,
            state->d_mynum, state->d_mymaskstart,
            state->d_mymasklength, state->d_nfrecvmod,
            state->d_statusmod, state->d_colnummod,
            state->d_mynummod, state->d_mymaskstartmod,
            state->d_mymasklengthmod, state->d_recv_cnt,
            state->d_msgnum, state->d_flag_mod) != 0)
        return -1;
    state->backward_time += SuperLU_timer_() - start;
    symldl_nvshmem_backward_value_trace(state, "post-backward");

    start = SuperLU_timer_();
    CUDA_CHECK(cudaMemcpy(x, state->d_x, (size_t) x_count * sizeof(double),
                          cudaMemcpyDeviceToHost));
    state->d2h_time += SuperLU_timer_() - start;
    state->diagonal_ready = 0;
    return symldl_nvshmem_check_backward(state);
#endif
}

static void
symldl_nvshmem_backward_take_timers(
    dSymLDLNVSHMEMBackwardState * handle, double *setup, double *h2d,
    double *diagonal, double *backward, double *d2h)
{
    dSymLDLNVSHMEMBackwardState *state =
        (dSymLDLNVSHMEMBackwardState *) handle;
    if (setup) *setup = state ? state->setup_time : 0.0;
    if (h2d) *h2d = state ? state->h2d_time : 0.0;
    if (diagonal) *diagonal = state ? state->diagonal_time : 0.0;
    if (backward) *backward = state ? state->backward_time : 0.0;
    if (d2h) *d2h = state ? state->d2h_time : 0.0;
}

static void
symldl_nvshmem_backward_destroy(
    dSymLDLNVSHMEMBackwardState * handle)
{
    symldl_nvshmem_backward_state_free(
        (dSymLDLNVSHMEMBackwardState *) handle);
}

extern "C" int
dSymLDLNVSHMEMSolveAvailable(void)
{
#ifdef HAVE_NVSHMEM
    return 1;
#else
    return 0;
#endif
}

extern "C" dSymLDLNVSHMEMSolveHandle
dSymLDLNVSHMEMSolveCreate(
    int nrhs, int_t x_count, int_t lsum_count,
    const dSymLDLSolveGraph *graph,
    const dSymLDLNVPanelDesc *device_panels,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d)
{
#ifndef HAVE_NVSHMEM
    (void) nrhs; (void) x_count; (void) lsum_count; (void) graph;
    (void) device_panels; (void) trf3Dpartition; (void) grid3d;
    return NULL;
#else
    dSymLDLNVSHMEMSolveState *state = new dSymLDLNVSHMEMSolveState();
    state->forward = symldl_nvshmem_forward_create(
        nrhs, x_count, lsum_count, graph, device_panels,
        trf3Dpartition, grid3d);
    if (state->forward == NULL) {
        delete state;
        return NULL;
    }
    state->backward = symldl_nvshmem_backward_create(state->forward);
    if (state->backward == NULL) {
        symldl_nvshmem_forward_destroy(state->forward);
        delete state;
        return NULL;
    }
    return (dSymLDLNVSHMEMSolveHandle) state;
#endif
}

extern "C" int
dSymLDLNVSHMEMForward(
    dSymLDLNVSHMEMSolveHandle handle, double *x, int_t x_count)
{
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    if (state == NULL || state->forward == NULL)
        return -1;
    if (symldl_nvshmem_forward_initialize_rhs(
            state->forward, x, x_count) != 0 ||
        symldl_nvshmem_forward_solve(
            state->forward, x, x_count) != 0)
        return -1;
    return symldl_nvshmem_forward_sparse_allreduce(
        state->forward, x, x_count);
}

extern "C" int
dSymLDLNVSHMEMDiagonal(
    dSymLDLNVSHMEMSolveHandle handle, double *x, int_t x_count)
{
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    return state != NULL && state->backward != NULL
               ? symldl_nvshmem_diagonal(state->backward, x, x_count)
               : -1;
}

extern "C" int
dSymLDLNVSHMEMBackward(
    dSymLDLNVSHMEMSolveHandle handle, double *x, int_t x_count)
{
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    return state != NULL && state->backward != NULL
               ? symldl_nvshmem_backward_solve(
                     state->backward, x, x_count)
               : -1;
}

extern "C" void
dSymLDLNVSHMEMSolveTakeTimers(
    dSymLDLNVSHMEMSolveHandle handle, double *setup, double *forward,
    double *sparse_reduce, double *sparse_broadcast, double *diagonal,
    double *backward, double *h2d, double *d2h)
{
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    double forward_setup = 0.0;
    double forward_h2d = 0.0;
    double forward_d2h = 0.0;
    double backward_setup = 0.0;
    double backward_h2d = 0.0;
    double backward_d2h = 0.0;
    if (state != NULL) {
        symldl_nvshmem_forward_take_timers(
            state->forward, &forward_setup, forward, sparse_reduce,
            sparse_broadcast, &forward_h2d, &forward_d2h);
        symldl_nvshmem_backward_take_timers(
            state->backward, &backward_setup, &backward_h2d, diagonal,
            backward, &backward_d2h);
    } else {
        if (forward) *forward = 0.0;
        if (sparse_reduce) *sparse_reduce = 0.0;
        if (sparse_broadcast) *sparse_broadcast = 0.0;
        if (diagonal) *diagonal = 0.0;
        if (backward) *backward = 0.0;
    }
    if (setup) *setup = forward_setup + backward_setup;
    if (h2d) *h2d = forward_h2d + backward_h2d;
    if (d2h) *d2h = forward_d2h + backward_d2h;
}

extern "C" void
dSymLDLNVSHMEMSolveDestroy(dSymLDLNVSHMEMSolveHandle handle)
{
    dSymLDLNVSHMEMSolveState *state =
        (dSymLDLNVSHMEMSolveState *) handle;
    if (state == NULL)
        return;
    symldl_nvshmem_backward_destroy(state->backward);
    symldl_nvshmem_forward_destroy(state->forward);
    delete state;
}
