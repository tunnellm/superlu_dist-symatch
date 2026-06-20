#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "superlu_ddefs.h"   // superlu_defs.h ??
#include "lu_common.hpp"
#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#include "xlupanels_GPU.cuh"
#include "gpuCommon.hpp"
#endif
#include "commWrapper.hpp"
#include "anc25d.hpp"
#include "luAuxStructTemplated.hpp"
// class lpanelGPU_t;
// class upanelGPU_t;
#define GLOBAL_BLOCK_NOT_FOUND -1
// it can be templatized for Ftype and complex Ftype


template <typename Ftype>
class xlpanel_t
{
public:
    int_t *index;
    Ftype *val;
    // ifdef GPU acceraleration

#ifdef HAVE_CUDA
    xlpanelGPU_t<Ftype> gpuPanel;
#endif
    // bool isDiagIncluded;

    xlpanel_t(int_t k, int_t *lsub, Ftype *nzval, int_t *xsup, int_t isDiagIncluded);

    // default constuctor
#ifdef HAVE_CUDA
    xlpanel_t() : gpuPanel(NULL, NULL)
    {
        index = NULL;
        val = NULL;
    }
#else
    xlpanel_t()
    {
        index = NULL;
        val = NULL;
    }
#endif

    xlpanel_t(int_t *index_, Ftype *val_) : index(index_), val(val_) { return; };

    // index[0] is number of blocks
    int_t nblocks()
    {
        return index[0];
    }
    // number of rows
    int_t nzrows() { return index[1]; }
    int_t haveDiag() { return index[2]; }
    int_t ncols() { return index[3]; }

    // global block id of k-th block in the panel
    int_t gid(int_t k)
    {
        return index[LPANEL_HEADER_SIZE + k];
    }

    // number of rows in the k-th block
    int_t nbrow(int_t k)
    {
        return index[LPANEL_HEADER_SIZE + nblocks() + k + 1] - index[LPANEL_HEADER_SIZE + nblocks() + k];
    }

    //
    int_t stRow(int k)
    {
        return index[LPANEL_HEADER_SIZE + nblocks() + k];
    }
    // row
    int_t *rowList(int_t k)
    {
        // LPANEL_HEADER
        // nblocks() : blocks list
        // nblocks()+1 : blocks st_points
        // index[LPANEL_HEADER_SIZE + nblocks() + k] statrting of the block
        return &index[LPANEL_HEADER_SIZE +
                      2 * nblocks() + 1 + index[LPANEL_HEADER_SIZE + nblocks() + k]];
    }

    Ftype *blkPtr(int_t k)
    {
        return &val[index[LPANEL_HEADER_SIZE + nblocks() + k]];
    }

    size_t blkPtrOffset(int_t k)
    {
        return index[LPANEL_HEADER_SIZE + nblocks() + k];
    }

    int_t LDA() { return index[1]; }
    int_t find(int_t k);
    // for L panel I don't need any special transformation function
    int_t panelSolve(int_t ksupsz, Ftype *DiagBlk, int_t LDD);
    int_t panelSolveSymmetric(int_t ksupsz, Ftype *DiagBlk, int_t LDD,
                              Ftype *Work, int_t LDWork);
    int_t diagFactor(int_t k, Ftype *UBlk, int_t LDU, threshPivValType<Ftype> thresh, int_t *xsup,
                     superlu_dist_options_t *options, SuperLUStat_t *stat, int *info);
    int_t packDiagBlock(Ftype *DiagLBlk, int_t LDD);
    int_t isEmpty() { return index == NULL; }
    int_t nzvalSize()
    {
        if (index == NULL)
            return 0;
        return ncols() * nzrows();
    }

    int_t indexSize()
    {
        if (index == NULL)
            return 0;
        return LPANEL_HEADER_SIZE + 2 * nblocks() + 1 + nzrows();
    }

    size_t totalSize()
    {
        return sizeof(int_t) * indexSize() + sizeof(Ftype) * nzvalSize();
    }

    // return the maximal iEnd such that stRow(iEnd)-stRow(iSt) < maxRow;
    int getEndBlock(int iSt, int maxRows);

    // ~lpanel_t()
    // {
    //     SUPERLU_FREE(index);
    //     // SUPERLU_FREE(val);
    // }

#ifdef HAVE_CUDA
    xlpanelGPU_t<Ftype> copyToGPU();
    xlpanelGPU_t<Ftype> copyToGPU(void *basePtr); // when we are doing a single allocation
    int checkGPU();
    int copyBackToGPU();

    int_t panelSolveGPU(cublasHandle_t handle, cudaStream_t cuStream,
                        int_t ksupsz,
                        Ftype *DiagBlk, // device pointer
                        int_t LDD);
    int_t panelSolveSymmetricGPU(cublasHandle_t handle, cudaStream_t cuStream,
                                 int_t ksupsz,
                                 Ftype *DiagBlk, // device pointer
                                 int_t LDD,
                                 Ftype *Work, // device pointer
                                 int_t LDWork);

    int_t diagFactorPackDiagBlockGPU(int_t k,
                                     Ftype *UBlk, int_t LDU,     // CPU pointers
                                     Ftype *DiagLBlk, int_t LDD, // CPU pointers
                                     Ftype thresh, int_t *xsup,
                                     superlu_dist_options_t *options,
                                     SuperLUStat_t *stat, int *info);
    int_t diagFactorCuSolver(int_t k,
                             cusolverDnHandle_t cusolverH, cudaStream_t cuStream,
                             Ftype *dWork, int *dInfo,   // GPU pointers
                             Ftype *dDiagBuf, int_t LDD, // GPU pointers
                             threshPivValType<Ftype> thresh, int_t *xsup,
                             superlu_dist_options_t *options,
                             SuperLUStat_t *stat, int *info);

    Ftype *blkPtrGPU(int k)
    {
        return &gpuPanel.val[blkPtrOffset(k)];
    }

    xlpanel_t(int_t *index_, Ftype *val_, int_t *indexGPU, Ftype *valGPU) : index(index_), val(val_), gpuPanel(indexGPU, valGPU)
    {
        return;
    };
    int_t copyFromGPU();
#endif
};

template <typename Ftype>
class xupanel_t
{
public:
    int_t *index;
    Ftype *val;
#ifdef HAVE_CUDA
    // xupanelGPU_t<Ftype>* upanelGPU;
    xupanelGPU_t<Ftype> gpuPanel;
#endif

    // xupanel_t(int_t *usub, Ftype *uval);
    xupanel_t(int_t k, int_t *usub, Ftype *uval, int_t *xsup);
#ifdef HAVE_CUDA
    xupanel_t() : gpuPanel(NULL, NULL)
    {
        index = NULL;
        val = NULL;
    }
#else
    xupanel_t()
    {
        index = NULL;
        val = NULL;
    }
#endif
    // constructing from recevied index and val
    xupanel_t(int_t *index_, Ftype *val_) : index(index_), val(val_) { return; };
    // index[0] is number of blocks
    int_t nblocks()
    {
        return index[0];
    }
    // number of rows
    int_t nzcols()
    {
        if (index == NULL)
            return 0; /* Sherry added this check. 2/22/2023 */
        return index[1];
    }
    int_t LDA() { return index[2]; } // is also supersize of that coloumn

    // global block id of k-th block in the panel
    int_t gid(int_t k)
    {
        return index[UPANEL_HEADER_SIZE + k];
    }

    // number of rows in the k-th block
    int_t nbcol(int_t k)
    {
        return index[UPANEL_HEADER_SIZE + nblocks() + k + 1] - index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    // row
    int_t *colList(int_t k)
    {
        // UPANEL_HEADER
        // nblocks() : blocks list
        // nblocks()+1 : blocks st_points
        // index[UPANEL_HEADER_SIZE + nblocks() + k] statrting of the block
        return &index[UPANEL_HEADER_SIZE +
                      2 * nblocks() + 1 + index[UPANEL_HEADER_SIZE + nblocks() + k]];
    }

    Ftype *blkPtr(int_t k)
    {
        return &val[LDA() * index[UPANEL_HEADER_SIZE + nblocks() + k]];
    }

    size_t blkPtrOffset(int_t k)
    {
        return LDA() * index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    // for U panel
    // int_t packed2skyline(int_t* usub, Ftype* uval );
    int_t packed2skyline(int_t k, int_t *usub, Ftype *uval, int_t *xsup);
    int_t loadFromSkyline(int_t k, int_t *usub, Ftype *uval, int_t *xsup);
    int_t panelSolve(int_t ksupsz, Ftype *DiagBlk, int_t LDD);
    int_t diagFactor(int_t k, Ftype *UBlk, int_t LDU, Ftype thresh, int_t *xsup,
                     superlu_dist_options_t *options,
                     SuperLUStat_t *stat, int *info);

    // Ftype* blkPtr(int_t k);
    // int_t LDA();
    int_t find(int_t k);
    int_t isEmpty() { return index == NULL; }
    int_t nzvalSize()
    {
        if (index == NULL)
            return 0;
        return LDA() * nzcols();
    }

    int_t indexSize()
    {
        if (index == NULL)
            return 0;
        return UPANEL_HEADER_SIZE + 2 * nblocks() + 1 + nzcols();
    }
    size_t totalSize()
    {
        return sizeof(int_t) * indexSize() + sizeof(Ftype) * nzvalSize();
    }
    int_t checkCorrectness()
    {
        if (index == NULL)
        {
            std::cout << "## Warning: Empty Panel"
                      << "\n";
            return 0;
        }
        int_t alternateNzcols = index[UPANEL_HEADER_SIZE + 2 * nblocks()];
        // std::cout<<nblocks()<<"  nzcols "<<nzcols()<<" alternate nzcols "<< alternateNzcols << "\n";
        if (nzcols() != alternateNzcols)
        {
            printf("Error 175\n");
            exit(-1);
        }

        return UPANEL_HEADER_SIZE + 2 * nblocks() + 1 + nzcols();
    }

    int_t stCol(int k)
    {
        return index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    int getEndBlock(int jSt, int maxCols);

    // ~upanel_t()
    // {
    //     SUPERLU_FREE(index);
    //     SUPERLU_FREE(val);
    // }

#ifdef HAVE_CUDA
    xupanelGPU_t<Ftype> copyToGPU();
    int_t ensureGPUValueStorage();
    // TODO: implement with baseptr
    xupanelGPU_t<Ftype> copyToGPU(void *basePtr);
    int copyBackToGPU();

    int_t panelSolveGPU(cublasHandle_t handle, cudaStream_t cuStream,
                        int_t ksupsz, Ftype *DiagBlk, int_t LDD);
    int checkGPU();

    Ftype *blkPtrGPU(int k)
    {
        return &gpuPanel.val[blkPtrOffset(k)];
    }

    xupanel_t(int_t *index_, Ftype *val_, int_t *indexGPU, Ftype *valGPU) : index(index_), val(val_), gpuPanel(indexGPU, valGPU)
    {
        return;
    };
    int_t copyFromGPU();
#endif
};

// Defineing GPU data types
// lapenGPU_t has exact same structure has lapanel_t but
// the pointers are on GPU
template <typename Ftype>
struct xLUstruct_t
{
    xlpanel_t<Ftype> *lPanelVec;
    xupanel_t<Ftype> *uPanelVec;
    gridinfo3d_t *grid3d;
    gridinfo_t *grid;
    int_t iam, Pc, Pr, myrow, mycol, ldt;
    int_t *xsup;
    int_t nsupers;
    // variables for scattering ldt*THREAD_Size
    int nThreads;
    int_t *indirect, *indirectRow, *indirectCol;
    Ftype *bigV; // size = THREAD_Size*ldt*ldt
    int *isNodeInMyGrid;
    threshPivValType<Ftype> thresh;
    int *info; 
    // TODO: get it from environment
    int numDiagBufs = 32; /* Sherry: not fixed yet */

    // Add SCT_t here
    SCT_t *SCT;
    superlu_dist_options_t *options;
    SuperLUStat_t *stat;
    LUStruct_type<Ftype> *LUstructPtr;
    int *symL2UOrders;
    Ftype *symFactWork;
    int *symFactIPIV;
    int64_t symFactWorkSize;
    int symFactTagUb;
    int symGPU3DContract = 0;
    int symGPU3DVersion = 1;
    double symContractValidateTol = 1.0e8;
    int64_t symContract1Accepted = 0;
    int64_t symContract1Fallbacks = 0;
    double symContract1MaxResid = 0.0;
    std::vector<Ftype *> symV2DiagBlocks;
    std::vector<Ftype *> symV2DiagBlocksGPU;
    enum SymV2SetupProfileId
    {
        SYM_V2_SETUP_NODE_MASK = 0,
        SYM_V2_SETUP_PANEL_VEC_BUILD,
        SYM_V2_SETUP_SEND_COUNT_EXCHANGE,
        SYM_V2_SETUP_PARTNER_SCRATCH_SIZE,
        SYM_V2_SETUP_CPU_WORKSPACE_ALLOC,
        SYM_V2_SETUP_RECV_BUFFER_ALLOC,
        SYM_V2_SETUP_DIAG_BUFFER_ALLOC,
        SYM_V2_SETUP_INIT_SYM_WORKSPACE,
        SYM_V2_SETUP_SYM_CPU_WORKSPACE,
        SYM_V2_SETUP_PARTNER_SEND_MAP_BUILD,
        SYM_V2_SETUP_PARTNER_SEND_GPU_ALLOC_COPY,
        SYM_V2_SETUP_PARTNER_RECV_COUNT_ALLREDUCE,
        SYM_V2_SETUP_PARTNER_META_ALLGATHER,
        SYM_V2_SETUP_PARTNER_RECV_MAP_BUILD,
        SYM_V2_SETUP_SET_GPU_TOTAL,
        SYM_V2_SETUP_GPU_MEM_ESTIMATE,
        SYM_V2_SETUP_COPY_L_PANELS_TO_GPU,
        SYM_V2_SETUP_SYM_V2_INDEX_COPY,
        SYM_V2_SETUP_GPU_PANEL_STRUCT_COPY,
        SYM_V2_SETUP_GPU_DIAG_FACTOR_SETUP,
        SYM_V2_SETUP_PER_STREAM_BUFFER_ALLOC,
        SYM_V2_SETUP_DFBUF_GEMMBUF_ALLOC,
        SYM_V2_SETUP_STREAM_HANDLE_CREATE,
        SYM_V2_SETUP_DIAG_PREFETCH_ALLOC,
        SYM_V2_SETUP_DEVICE_STRUCT_COPY,
        SYM_V2_SETUP_COUNT
    };

    int symV2SetupProfileEnabled = 0;
    int symV2SetupProfilePrinted = 0;
    double symV2SetupProfileTime[SYM_V2_SETUP_COUNT] = {};
    long long symV2SetupProfileCount[SYM_V2_SETUP_COUNT] = {};

    bool symV2SetupProfileActive() const
    {
        return symV2SetupProfileEnabled != 0;
    }

    void symV2SetupProfileAdd(SymV2SetupProfileId id, double elapsed)
    {
        if (!symV2SetupProfileActive())
            return;
        symV2SetupProfileTime[id] += elapsed;
        symV2SetupProfileCount[id] += 1;
    }

    struct SymV2SetupProfileScope
    {
        xLUstruct_t<Ftype> *owner;
        SymV2SetupProfileId id;
        bool active;
        double start;

        SymV2SetupProfileScope(xLUstruct_t<Ftype> *owner_,
                               SymV2SetupProfileId id_)
            : owner(owner_), id(id_),
              active(owner_ != NULL && owner_->symV2SetupProfileActive()),
              start(active ? SuperLU_timer_() : 0.0) {}

        ~SymV2SetupProfileScope()
        {
            if (active)
                owner->symV2SetupProfileAdd(id, SuperLU_timer_() - start);
        }
    };

    void printSymV2SetupProfile();
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    enum SymGPU3DTimingId
    {
        SYM_GPU3D_T_L2U_START = 0,
        SYM_GPU3D_T_DIAG_D2H,
        SYM_GPU3D_T_DIAG_D2H_COPY,
        SYM_GPU3D_T_DIAG_D2H_WAIT,
        SYM_GPU3D_T_DIAG_PREFETCH_ISSUE,
        SYM_GPU3D_T_DIAG_PREFETCH_WAIT,
        SYM_GPU3D_T_CPU_SYTRF,
        SYM_GPU3D_T_GPU_SYTRI,
        SYM_GPU3D_T_GPU_SYTRI_VALIDATE,
        SYM_GPU3D_T_CPU_SYTRI,
        SYM_GPU3D_T_DIAG_PACK,
        SYM_GPU3D_T_DIAG_BCAST,
        SYM_GPU3D_T_INV_H2D,
        SYM_GPU3D_T_LDIAG_D2D,
        SYM_GPU3D_T_LPANEL_TRANSFORM,
        SYM_GPU3D_T_L2U_FINISH,
        SYM_GPU3D_T_LFRAG_EXCHANGE_TOTAL,
        SYM_GPU3D_T_LFRAG_PACK_ISSUE,
        SYM_GPU3D_T_LFRAG_D2H_STAGE_ISSUE,
        SYM_GPU3D_T_LFRAG_PACK_STAGE_SYNC,
        SYM_GPU3D_T_LFRAG_MPI_RECV_WAIT,
        SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
        SYM_GPU3D_T_LFRAG_ASSEMBLE_ISSUE,
        SYM_GPU3D_T_LFRAG_SEND_WAIT,
        SYM_GPU3D_T_LFRAG_STREAM_SYNC,
        SYM_GPU3D_T_PANEL_BCAST,
        SYM_GPU3D_T_PANEL_BCAST_MPI,
        SYM_GPU3D_T_PANEL_INDEX_D2H,
        SYM_GPU3D_T_PANEL_BCAST_SINGLETON,
        SYM_GPU3D_T_SCHUR_UPDATE,
        SYM_GPU3D_T_SCHUR_SYNC,
        SYM_GPU3D_T_LOOKAHEAD_UPDATE,
        SYM_GPU3D_T_LOOKAHEAD_SYNC,
        SYM_GPU3D_T_EXCLUDE_UPDATE,
        SYM_GPU3D_T_SCHED_LOOKAHEAD_DISPATCH,
        SYM_GPU3D_T_SCHED_PREFETCH_READY,
        SYM_GPU3D_T_SCHED_FACTOR_DISPATCH,
        SYM_GPU3D_T_SCHED_BCAST_ADVANCE,
        SYM_GPU3D_T_SCHED_FINAL_SYNC,
        SYM_GPU3D_T_INITIAL_FACTOR_DISPATCH,
        SYM_GPU3D_T_INITIAL_PANEL_BCAST,
        SYM_GPU3D_T_FACTOR_TREE_WALL,
        SYM_GPU3D_T_COUNT
    };

    enum SymGPU3DStatId
    {
        SYM_GPU3D_S_FACTOR_TREES = 0,
        SYM_GPU3D_S_FACTOR_NODES,
        SYM_GPU3D_S_INITIAL_FACTOR_NODES,
        SYM_GPU3D_S_PARENT_FACTOR_NODES,
        SYM_GPU3D_S_LOOKAHEAD_UPDATES,
        SYM_GPU3D_S_EXCLUDE_UPDATES,
        SYM_GPU3D_S_PANEL_BCASTS,
        SYM_GPU3D_S_PANEL_BCAST_BYTES,
        SYM_GPU3D_S_PANEL_BCAST_MPI_BYTES,
        SYM_GPU3D_S_PANEL_INDEX_D2H_BYTES,
        SYM_GPU3D_S_L2U_LOCAL_BYTES,
        SYM_GPU3D_S_L2U_SEND_BYTES,
        SYM_GPU3D_S_L2U_RECV_BYTES,
        SYM_GPU3D_S_L2U_HOST_STAGING_BYTES,
        SYM_GPU3D_S_L2U_CUDA_AWARE_SEND_BYTES,
        SYM_GPU3D_S_DIAG_D2H_BYTES,
        SYM_GPU3D_S_DIAG_PREFETCH_HITS,
        SYM_GPU3D_S_DIAG_PREFETCH_MISSES,
        SYM_GPU3D_S_DIAG_PREFETCH_ISSUES,
        SYM_GPU3D_S_SCHED_WINDOWS,
        SYM_GPU3D_S_SCHED_WINDOW_NODES,
        SYM_GPU3D_S_SCHED_READY_BCASTS,
        SYM_GPU3D_S_SCHED_MAX_WINDOW,
        SYM_GPU3D_S_SCHED_MAX_NUM_LA,
        SYM_GPU3D_S_COUNT
    };

    double symGPU3DTime[SYM_GPU3D_T_COUNT] = {};
    long long symGPU3DCount[SYM_GPU3D_T_COUNT] = {};
    long long symGPU3DStat[SYM_GPU3D_S_COUNT] = {};

    void symTimingAdd(SymGPU3DTimingId id, double elapsed)
    {
        symGPU3DTime[id] += elapsed;
        symGPU3DCount[id] += 1;
    }

    void symStatAdd(SymGPU3DStatId id, long long value = 1)
    {
        symGPU3DStat[id] += value;
    }

    void symStatMax(SymGPU3DStatId id, long long value)
    {
        if (value > symGPU3DStat[id])
            symGPU3DStat[id] = value;
    }

    struct SymTimingScope
    {
        xLUstruct_t<Ftype> *owner;
        SymGPU3DTimingId id;
        double start;

        SymTimingScope(xLUstruct_t<Ftype> *owner_, SymGPU3DTimingId id_)
            : owner(owner_), id(id_), start(SuperLU_timer_()) {}

        ~SymTimingScope()
        {
            if (owner != NULL)
                owner->symTimingAdd(id, SuperLU_timer_() - start);
        }
    };

    void printSymGPU3DTiming();
#endif
#ifdef HAVE_CUDA
    std::vector<double *> symL2USendBufsGPU;
    std::vector<int_t *> symL2USendMapsGPU;
    std::vector<double *> symV2PartnerLSendBufsGPU;
    std::vector<int_t *> symL2LSendMapsGPU;
    double *symV2PartnerLSendBufPoolGPU;
    int_t *symL2LSendMapPoolGPU;
    int_t *symV2PartnerLRecvMapPoolGPU;
    size_t symV2PartnerLSendBufPoolCount;
    size_t symL2LSendMapPoolCount;
    size_t symV2PartnerLRecvMapPoolCount;
    std::vector<std::vector<int_t> > symL2LSendMeta;
    std::vector<std::vector<double> > symV2PartnerLHostSendBufs;
    std::vector<int> symV2PartnerLSendSizes;
    std::vector<int> symV2PartnerLRecvSizes;
    std::vector<std::vector<int_t> > symV2PartnerLRecvIndex;
    std::vector<std::vector<int_t> > symV2PartnerLRecvMap;
    std::vector<int_t *> symV2PartnerLRecvMapsGPU;
    std::vector<int_t *> symL2ULocalMapsGPU;
    std::vector<int> symPanelReadyEventIds;
    std::vector<Ftype *> symDiagPrefetchBufs;
    std::vector<cudaEvent_t> symDiagPrefetchDoneEvents;
    std::vector<int> symDiagPrefetchEventIds;
    std::vector<int_t> symDiagPrefetchNodes;
#endif

    // Adding more variables for factorization
    trf3dpartitionType<Ftype> *trf3Dpartition;
    int_t maxLvl;
    int maxLeafNodes; /* Sherry added 12/31/22. Computed in xLUstruct_t constructor */

    diagFactBufs_type<Ftype>** dFBufs; /* stores L and U diagonal blocks */
    int superlu_acc_offload;
    // myNodeCount,
    // treePerm
    // myZeroTrIdxs
    // sForests
    // myTreeIdxs
    // gEtreeInfo

    // buffers for communication
    int_t maxLvalCount = 0;
    int_t maxLidxCount = 0;
    int_t maxSymPartnerLvalCount = 0;
    int_t maxSymPartnerLidxCount = 0;
    int_t maxUvalCount = 0;
    int_t maxUidxCount = 0;
    std::vector<Ftype *> diagFactBufs; /* stores diagonal blocks,
                       each one is a normal dense matrix.
                    Sherry: where are they free'd ?? */
    std::vector<Ftype *> LvalRecvBufs;
    std::vector<Ftype *> UvalRecvBufs;
    std::vector<Ftype *> symPartnerLvalRecvBufs;
    std::vector<int_t *> LidxRecvBufs;
    std::vector<int_t *> UidxRecvBufs;
    std::vector<int_t *> symPartnerLidxRecvBufs;

    // send and recv count for 2d comm
    std::vector<int_t> LvalSendCounts;
    std::vector<int_t> UvalSendCounts;
    std::vector<int_t> LidxSendCounts;
    std::vector<int_t> UidxSendCounts;

    //
    #pragma warning disabling bcastStruct
    #if 0
    std::vector<bcastStruct> bcastDiagRow;
    std::vector<bcastStruct> bcastDiagCol;
    std::vector<bcastStruct> bcastLval;
    std::vector<bcastStruct> bcastUval;
    std::vector<bcastStruct> bcastLidx;
    std::vector<bcastStruct> bcastUidx;
    #endif 

    int_t krow(int_t k) { return k % Pr; }
    int_t kcol(int_t k) { return k % Pc; }
    int_t procIJ(int_t i, int_t j) { return PNUM(krow(i), kcol(j), grid); }
    int_t symV2PanelRoot(int_t k);
    int_t symV2DiagRoot(int_t k);
    int_t symV2DiagProc(int_t k);
    int_t symV2PanelIndex(int_t k);
    int_t symV2RowIndex(int_t k);
    int_t symV2PanelCount();
    int_t symV2RowCount();
    int_t symV2PanelGid(int_t local_index);
    int_t symV2RowGid(int_t local_index);
    int_t supersize(int_t k) { return xsup[k + 1] - xsup[k]; }
    int_t g2lRow(int_t k) { return k / Pr; }
    int_t g2lCol(int_t k) { return k / Pc; }
    bool useSymV2Solve() const
    {
        return options != NULL && options->SymFact == YES &&
               symGPU3DVersion == 2;
    }
    bool needsUPanelStorage() const
    {
        return !useSymV2Solve();
    }
    bool symV2ScheduleActive() const;
    int_t symV2ForestLevelCount() const;
    template <typename PtrT>
    static void superluFreeIfAllocated(PtrT *&ptr)
    {
        if (ptr != NULL)
        {
            SUPERLU_FREE(ptr);
            ptr = NULL;
        }
    }
    void freeRecvBuffers(bool include_u_buffers)
    {
        int nlook = (options != NULL) ? options->num_lookaheads : 0;
        for (int i = 0; i < nlook; i++)
        {
            superluFreeIfAllocated(LvalRecvBufs[i]);
            if (include_u_buffers)
                superluFreeIfAllocated(UvalRecvBufs[i]);
            superluFreeIfAllocated(symPartnerLvalRecvBufs[i]);
            superluFreeIfAllocated(LidxRecvBufs[i]);
            if (include_u_buffers)
                superluFreeIfAllocated(UidxRecvBufs[i]);
            superluFreeIfAllocated(symPartnerLidxRecvBufs[i]);
        }
    }

    anc25d_t anc25d;
    // For GPU acceleration
    xLUstructGPU_t<Ftype> *dA_gpu; // pointing to memory on GPU
    xLUstructGPU_t<Ftype> A_gpu;   // pointing to memory accessible on CPU

    /////////////////////////////////////////////////////////////////
    // Intermediate for flat batched
    /////////////////////////////////////////////////////////////////
    dLocalLU_t *host_Llu;
    dLocalLU_t d_localLU;

    int *d_lblock_gid_dat, **d_lblock_gid_ptrs;
    int *d_lblock_start_dat, **d_lblock_start_ptrs;
    int64_t *d_lblock_gid_offsets, *d_lblock_start_offsets;
    int64_t total_l_blocks, total_start_size;

    void computeLBlockData();
    /////////////////////////////////////////////////////////////////

    enum indirectMapType
    {
        ROW_MAP,
        COL_MAP
    };

    /**
     *          C O N / D E S - T R U C T O R S
     */
    xLUstruct_t(int_t nsupers, int_t ldt_, trf3dpartitionType<Ftype> *trf3Dpartition,
                  LUStruct_type<Ftype> *LUstruct, gridinfo3d_t *grid3d,
                  SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                  threshPivValType<Ftype> thresh_, int *info_);

    ~xLUstruct_t()
    {
#define XLU_V2_DTOR_TRACE(msg_) do { \
        const char *xlu_trace_env_ = std::getenv("GPU3DV2_TRACE"); \
        if (xlu_trace_env_ != NULL && xlu_trace_env_[0] != '\0' && \
            xlu_trace_env_[0] != '0') { \
            std::fprintf(stderr, "[sym-v2-trace] rank %d: destructor %s\n", \
                         (grid3d != NULL) ? grid3d->iam : -1, (msg_)); \
            std::fflush(stderr); \
        } \
    } while (0)
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        if (options != NULL && options->SymFact == YES)
            printSymGPU3DTiming();
#endif

        XLU_V2_DTOR_TRACE("begin");

        /* Yang: Deallocate the lPanelVec[i] and uPanelVec[i] here instead of using destructors ~lpanel_t or ~upanel_t,
        as xlpanel_t/upanel_t is used for holding temporary communication buffers as well. Note that lPanelVec[i].val is not deallocated here as it's pointing to the L data in the C code*/

        XLU_V2_DTOR_TRACE("before lPanelVec free");
        for (int_t i = 0; i < symV2PanelCount(); ++i)
            if (symV2PanelGid(i) < nsupers &&
                isNodeInMyGrid[symV2PanelGid(i)] == 1)
            {
                if (lPanelVec[i].index)
                    SUPERLU_FREE(lPanelVec[i].index);
                // SUPERLU_FREE(lPanelVec[i].val);
            }
        XLU_V2_DTOR_TRACE("after lPanelVec free");

        XLU_V2_DTOR_TRACE("before uPanelVec free");
        if (uPanelVec != NULL)
            for (int_t i = 0; i < symV2RowCount(); ++i)
                if (symV2RowGid(i) < nsupers &&
                    isNodeInMyGrid[symV2RowGid(i)] == 1)
                {
                    if (uPanelVec[i].index)
                        SUPERLU_FREE(uPanelVec[i].index);
                    if (uPanelVec[i].val)
                        SUPERLU_FREE(uPanelVec[i].val);
                }
        XLU_V2_DTOR_TRACE("after uPanelVec free");

        delete[] lPanelVec;
        if (uPanelVec != NULL)
            delete[] uPanelVec;
        XLU_V2_DTOR_TRACE("after panel vector delete");

        /* free diagonal L and U blocks */
        // dfreeDiagFactBufsArr(maxLeafNodes, dFBufs);
        XLU_V2_DTOR_TRACE("before freeDiagFactBufsArr");
        freeDiagFactBufsArr(numDiagBufs, dFBufs);
        XLU_V2_DTOR_TRACE("after freeDiagFactBufsArr");
        XLU_V2_DTOR_TRACE("before freeSymFactWorkspace");
        freeSymFactWorkspace();
        XLU_V2_DTOR_TRACE("after freeSymFactWorkspace");
        XLU_V2_DTOR_TRACE("before symV2DiagBlocks free");
        for (size_t i = 0; i < symV2DiagBlocks.size(); ++i)
            if (symV2DiagBlocks[i] != NULL)
                SUPERLU_FREE(symV2DiagBlocks[i]);
#ifdef HAVE_CUDA
        for (size_t i = 0; i < symV2DiagBlocksGPU.size(); ++i)
            if (symV2DiagBlocksGPU[i] != NULL)
                cudaFree(symV2DiagBlocksGPU[i]);
#endif
        XLU_V2_DTOR_TRACE("after symV2DiagBlocks free");

        XLU_V2_DTOR_TRACE("before CPU scratch free");
        SUPERLU_FREE(bigV);
        SUPERLU_FREE(indirect);
        SUPERLU_FREE(indirectRow);
        SUPERLU_FREE(indirectCol);
        XLU_V2_DTOR_TRACE("after CPU scratch free");

        int i;
        XLU_V2_DTOR_TRACE("before recv buffers free");
        freeRecvBuffers(needsUPanelStorage());
        XLU_V2_DTOR_TRACE("after recv buffers free");

        XLU_V2_DTOR_TRACE("before diagFactBufs free");
        for (i = 0; i < numDiagBufs; i++)
            SUPERLU_FREE(diagFactBufs[i]);
        XLU_V2_DTOR_TRACE("after diagFactBufs free");

        /* Sherry added the following, which comes from batch setup */
        superlu_acc_offload = sp_ienv_dist(10, options); //get_acc_offload();
        if (superlu_acc_offload)
        {
            XLU_V2_DTOR_TRACE("before GPU scratch free");
            // printf(".. free batch buffers\n");  fflush(stdout);
            SUPERLU_FREE(A_gpu.dFBufs);
            SUPERLU_FREE(A_gpu.gpuGemmBuffs);

            for (int stream = 0; stream < A_gpu.numCudaStreams; stream++)
            {
                cudaEventDestroy(A_gpu.panelReadyEvents[stream]);
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
                cudaEventDestroy(A_gpu.diagD2HStartEvents[stream]);
                cudaEventDestroy(A_gpu.diagD2HEndEvents[stream]);
#endif
                if (A_gpu.cuSolveHandles[stream] != NULL)
                    cusolverDnDestroy(A_gpu.cuSolveHandles[stream]);
                cublasDestroy(A_gpu.cuHandles[stream]);
                cublasDestroy(A_gpu.lookAheadLHandle[stream]);
                cublasDestroy(A_gpu.lookAheadUHandle[stream]);
            }
            if (A_gpu.symV2PanelLocalIndex != NULL)
                cudaFree(A_gpu.symV2PanelLocalIndex);
            XLU_V2_DTOR_TRACE("after GPU scratch free");
        }

        XLU_V2_DTOR_TRACE("before isNodeInMyGrid free");
        SUPERLU_FREE(isNodeInMyGrid);
        XLU_V2_DTOR_TRACE("end");
#undef XLU_V2_DTOR_TRACE

    } /* end destructor xLUstruct_t */

    /**
     *           Compute Functions
     */
    int_t pdgstrf3d();
    int_t pdgstrf3dSymV2();
    int_t dSchurComplementUpdate(int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel);
    int_t *computeIndirectMap(indirectMapType direction, int_t srcLen, int_t *srcVec,
                              int_t dstLen, int_t *dstVec);

    int_t dScatter(int_t m, int_t n,
                   int_t gi, int_t gj,
                   Ftype *V, int_t ldv,
                   int_t *srcRowList, int_t *srcColList);

    int_t lookAheadUpdate(
        int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel);
    int_t blockUpdate(int_t k,
                      int_t ii, int_t jj, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel);
    int_t dSchurCompUpdateExcludeOne(
        int_t k, int_t ex, // suypernodes to be excluded
        xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel);
    int_t dSymV2LFragmentExchangeHost(int_t k, int_t offset,
                                      int raw_values = 0);
    int_t dSymSchurCompUpdatePartLL(
        int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
        int_t k, xlpanel_t<Ftype> &lpanel);
    int_t dSymSchurCompUpLimitedMemLL(
        int_t lStart, int_t lEnd,
        int_t jStart, int_t jEnd,
        int_t k, xlpanel_t<Ftype> &lpanel);
    int_t dSymLookAheadUpdateLL(
        int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel);
    int_t dSymSchurCompUpdateExcludeOneLL(
        int_t k, int_t ex, xlpanel_t<Ftype> &lpanel);
    int_t dSymSchurCompUpdatePartWithLFragments(
        int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
        int_t k, xlpanel_t<Ftype> &lpanel,
        int_t *frag_index, Ftype *frag_val);
    int_t dSymSchurCompUpLimitedMemWithLFragments(
        int_t lStart, int_t lEnd,
        int_t fragStart, int_t fragEnd,
        int_t k, xlpanel_t<Ftype> &lpanel,
        int_t *frag_index, Ftype *frag_val);
    int_t dSymLookAheadUpdateWithLFragments(
        int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel,
        int_t *frag_index, Ftype *frag_val);
    int_t dSymSchurCompUpdateExcludeOneWithLFragments(
        int_t k, int_t ex, xlpanel_t<Ftype> &lpanel,
        int_t *frag_index, Ftype *frag_val);

    int_t dsparseTreeFactor(
        sForest_t *sforest,
        diagFactBufs_type<Ftype>** dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);

    int dsparseTreeFactorBatchGPU(
        sForest_t *sforest,
        diagFactBufs_type<Ftype>** dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);

    diagFactBufs_type<Ftype>** initDiagFactBufsArr(int_t mxLeafNode, int_t ldt);

    // Helper routine to marshall batch LU data into the device data in A_gpu
    void marshallBatchedLUData(int k_st, int k_end, int_t *perm_c_supno);
    void marshallBatchedBufferCopyData(int k_st, int k_end, int_t *perm_c_supno);
    void marshallBatchedTRSMUData(int k_st, int k_end, int_t *perm_c_supno);
    void marshallBatchedTRSMLData(int k_st, int k_end, int_t *perm_c_supno);
    void marshallBatchedSCUData(int k_st, int k_end, int_t *perm_c_supno);
    void initSCUMarshallData(int k_st, int k_end, int_t *perm_c_supno);
    int marshallSCUBatchedDataInner(int k_st, int k_end, int_t *perm_c_supno);
    int marshallSCUBatchedDataOuter(int k_st, int k_end, int_t *perm_c_supno);
    void dFactBatchSolve(int k_st, int k_end, int_t *perm_c_supno);

    //
    int_t dDiagFactorPanelSolve(int_t k, int_t offset, diagFactBufs_type<Ftype>** dFBufs);
    int_t dSymDiagFactorPanelSolve(int_t k, int_t handle_offset, int_t buffer_offset,
                                   diagFactBufs_type<Ftype>** dFBufs);
    int_t dSymStartL2U(int_t k, int_t stream_offset = 0);
    int_t dSymFinishL2U(int_t k);
#ifdef HAVE_CUDA
    int_t dSymStartDiagPrefetch(int_t k, int_t stream_offset);
#endif
    int initSymFactWorkspace();
    int freeSymFactWorkspace();
    int ensureSymFactWorkSize(int64_t minSize);
    int_t dPanelBcast(int_t k, int_t offset);
    int_t dsparseTreeFactorBaseline(
        sForest_t *sforest,
        diagFactBufs_type<Ftype>** dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);

    int_t packedU2skyline(LUStruct_type<Ftype> *LUstruct);

    int_t ancestorReduction3d(int_t ilvl, int_t *myNodeCount,
                              int_t **treePerm);

    int_t zSendLPanel(int_t k0, int_t receiverGrid);
    int_t zRecvLPanel(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta);
    int_t zSendUPanel(int_t k0, int_t receiverGrid);
    int_t zRecvUPanel(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta);

    int_t dAncestorFactorBaseline(
        int_t alvl,
        sForest_t *sforest,
        diagFactBufs_type<Ftype> **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);

    int_t dAncestorFactor(
        int_t alvl,
        sForest_t *sforest,
        diagFactBufs_type<Ftype> **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);
    // GPU related functions
#ifdef HAVE_CUDA
    int_t setLUstruct_GPU();
    int_t dsparseTreeFactorGPU(
        sForest_t *sforest,
        diagFactBufs_type<Ftype> **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);
    int_t dsparseTreeFactorGPUBaseline(
        sForest_t *sforest,
        diagFactBufs_type<Ftype> **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);

    int_t dAncestorFactorBaselineGPU(
        int_t alvl,
        sForest_t *sforest,
        diagFactBufs_type<Ftype> **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);

    int_t dSchurComplementUpdateGPU(
        int streamId,
        int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel);
    int_t dSchurCompUpdatePartGPU(
        int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
        int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel,
        cublasHandle_t handle, cudaStream_t cuStream,
        Ftype *gemmBuff);
    int_t lookAheadUpdateGPU(
        int streamId,
        int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel);
    int_t dSchurCompUpLimitedMem(
        int streamId,
        int_t lStart, int_t lEnd,
        int_t uStart, int_t uEnd,
        int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel);
    int_t dSchurCompUpdateExcludeOneGPU(
        int streamId,
        int_t k, int_t ex, // suypernodes to be excluded
        xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel);
    int_t dSymSchurCompUpdatePartWithLFragmentsGPU(
        int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
        int_t k, xlpanel_t<Ftype> &lpanel,
        const int_t *frag_index, Ftype *frag_val,
        cublasHandle_t handle, cudaStream_t cuStream,
        Ftype *gemmBuff);
    int_t dSymSchurCompUpLimitedMemWithLFragmentsGPU(
        int_t lStart, int_t lEnd,
        int_t fragStart, int_t fragEnd,
        int_t k, xlpanel_t<Ftype> &lpanel,
        const int_t *frag_index, Ftype *frag_val,
        cublasHandle_t handle, cudaStream_t cuStream,
        Ftype *gemmBuff);
    int_t dSymLookAheadUpdateWithLFragmentsGPU(
        int streamId,
        int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel);
    int_t dSymSchurCompUpdateExcludeOneWithLFragmentsGPU(
        int streamId,
        int_t k, int_t ex, xlpanel_t<Ftype> &lpanel);
    int_t dSymSchurCompUpdatePartLLGPU(
        int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
        int_t k, xlpanel_t<Ftype> &lpanel,
        cublasHandle_t handle, cudaStream_t cuStream,
        Ftype *rawBlock, Ftype *gemmBuff);
    int_t dSymSchurCompUpLimitedMemLLGPU(
        int_t lStart, int_t lEnd,
        int_t jStart, int_t jEnd,
        int_t k, xlpanel_t<Ftype> &lpanel,
        cublasHandle_t handle, cudaStream_t cuStream,
        Ftype *rawBlock, Ftype *gemmBuff);
    int_t dSymLookAheadUpdateLLGPU(
        int streamId,
        int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel);
    int_t dSymSchurCompUpdateExcludeOneLLGPU(
        int streamId,
        int_t k, int_t ex, xlpanel_t<Ftype> &lpanel);

    int_t dDiagFactorPanelSolveGPU(int_t k, int_t offset, diagFactBufs_type<Ftype>** dFBufs);
    int_t dPanelBcastGPU(int_t k, int_t offset);
    int_t dSymStartL2UGPU(int_t k, int_t stream_offset);
    int_t dSymV2ComputePartnerScratchSize(LUStruct_type<Ftype> *LUstruct);
    int_t dSymV2LFragmentExchangeGPU(int_t k, int_t stream_offset);

    int_t ancestorReduction3dGPU(int_t ilvl, int_t *myNodeCount,
                                 int_t **treePerm);
    int_t zSendLPanelGPU(int_t k0, int_t receiverGrid);
    int_t zRecvLPanelGPU(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta);
    int_t zSendUPanelGPU(int_t k0, int_t receiverGrid);
    int_t zRecvUPanelGPU(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta);
    int_t copyLUGPUtoHost();
    int_t copyLUHosttoGPU();
    int_t checkGPU();

    // some more helper functions
    xupanel_t<Ftype> getKUpanel(int_t k, int_t offset);
    xlpanel_t<Ftype> getKLpanel(int_t k, int_t offset);
    int_t SyncLookAheadUpdate(int streamId);

    Ftype *gpuLvalBasePtr, *gpuUvalBasePtr;
    int_t *gpuLidxBasePtr, *gpuUidxBasePtr;
    size_t gpuLvalSize, gpuUvalSize, gpuLidxSize, gpuUidxSize;

    xlpanelGPU_t<Ftype> *copyLpanelsToGPU();
    xupanelGPU_t<Ftype> *copyUpanelsToGPU();

    int freeDiagFactBufsArr(int_t num_bufs, diagFactBufs_type<Ftype>** dFBufs);

    // to perform diagFactOn GPU
    int_t dDFactPSolveGPU(int_t k, int_t offset, diagFactBufs_type<Ftype>** dFBufs);
    int_t dDFactPSolveGPU(int_t k, int_t handle_offset, int buffer_offset, diagFactBufs_type<Ftype>** dFBufs);
#endif
};

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2PanelRoot(int_t k)
{
    return kcol(k);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2DiagRoot(int_t k)
{
    return krow(k);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2DiagProc(int_t k)
{
    return PNUM(symV2DiagRoot(k), symV2PanelRoot(k), grid);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2PanelIndex(int_t k)
{
    return g2lCol(k);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2RowIndex(int_t k)
{
    return g2lRow(k);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2PanelCount()
{
    return CEILING(nsupers, Pc);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2RowCount()
{
    return CEILING(nsupers, Pr);
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2PanelGid(int_t local_index)
{
    return local_index * Pc + mycol;
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2RowGid(int_t local_index)
{
    return local_index * Pr + myrow;
}

template <typename Ftype>
inline bool xLUstruct_t<Ftype>::symV2ScheduleActive() const
{
    return false;
}

template <typename Ftype>
inline int_t xLUstruct_t<Ftype>::symV2ForestLevelCount() const
{
    return log2i(grid3d->zscp.Np) + 1;
}

template <>
inline bool xLUstruct_t<double>::symV2ScheduleActive() const
{
    return useSymV2Solve() && trf3Dpartition != NULL &&
           trf3Dpartition->symV2ScheduleEnabled;
}

template <>
inline int_t xLUstruct_t<double>::symV2ForestLevelCount() const
{
    return symV2ScheduleActive() ? trf3Dpartition->maxLvl
                                 : log2i(grid3d->zscp.Np) + 1;
}

template <>
inline int_t xLUstruct_t<double>::symV2PanelRoot(int_t k)
{
    if (useSymV2Solve())
    {
        if (trf3Dpartition == NULL ||
            trf3Dpartition->symV2PanelRoot == NULL)
            ABORT("SymFact V2 LDL owner metadata is not initialized.");
        return trf3Dpartition->symV2PanelRoot[k];
    }
    return kcol(k);
}

template <>
inline int_t xLUstruct_t<double>::symV2DiagRoot(int_t k)
{
    if (useSymV2Solve())
    {
        if (trf3Dpartition == NULL ||
            trf3Dpartition->symV2DiagRoot == NULL)
            ABORT("SymFact V2 LDL owner metadata is not initialized.");
        return trf3Dpartition->symV2DiagRoot[k];
    }
    return krow(k);
}

template <>
inline int_t xLUstruct_t<double>::symV2PanelIndex(int_t k)
{
    if (useSymV2Solve())
    {
        if (trf3Dpartition == NULL ||
            trf3Dpartition->symV2PanelLocalIndex == NULL)
            ABORT("SymFact V2 LDL local panel metadata is not initialized.");
        return trf3Dpartition->symV2PanelLocalIndex[k];
    }
    return g2lCol(k);
}

template <>
inline int_t xLUstruct_t<double>::symV2RowIndex(int_t k)
{
    if (useSymV2Solve())
    {
        if (trf3Dpartition == NULL ||
            trf3Dpartition->symV2RowLocalIndex == NULL)
            ABORT("SymFact V2 LDL local row metadata is not initialized.");
        return trf3Dpartition->symV2RowLocalIndex[k];
    }
    return g2lRow(k);
}

template <>
inline int_t xLUstruct_t<double>::symV2PanelCount()
{
    return useSymV2Solve() ? trf3Dpartition->symV2LocalPanelCount
                           : CEILING(nsupers, Pc);
}

template <>
inline int_t xLUstruct_t<double>::symV2RowCount()
{
    return useSymV2Solve() ? trf3Dpartition->symV2LocalRowCount
                           : CEILING(nsupers, Pr);
}

template <>
inline int_t xLUstruct_t<double>::symV2PanelGid(int_t local_index)
{
    return useSymV2Solve() ? trf3Dpartition->symV2LocalPanelGids[local_index]
                           : local_index * Pc + mycol;
}

template <>
inline int_t xLUstruct_t<double>::symV2RowGid(int_t local_index)
{
    return useSymV2Solve() ? trf3Dpartition->symV2LocalRowGids[local_index]
                           : local_index * Pr + myrow;
}
