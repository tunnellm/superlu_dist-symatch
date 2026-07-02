#pragma once
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "superlu_ddefs.h"   // superlu_defs.h ??
#include "lu_common.hpp"
#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#include "xlupanels_GPU.cuh"
#include "gpuCommon.hpp"
#include "gpu_mpi_utils.hpp"
#endif
#include "commWrapper.hpp"
#include "anc25d.hpp"
#include "luAuxStructTemplated.hpp"
// class lpanelGPU_t;
// class upanelGPU_t;
#define GLOBAL_BLOCK_NOT_FOUND -1

#ifdef HAVE_CUDA
struct SymV2RowDownSendSegmentGPU
{
    size_t map_offset;
    int_t nrows;
    int_t dst_row_offset;
};
#endif

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
        size_t idxSize = sizeof(int_t) * indexSize();
        const size_t align = alignof(Ftype);
        const size_t mask = align - 1;
        if (align > 1)
            idxSize = (idxSize + mask) & ~mask;
        return idxSize + sizeof(Ftype) * nzvalSize();
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
        size_t idxSize = sizeof(int_t) * indexSize();
        const size_t align = alignof(Ftype);
        const size_t mask = align - 1;
        if (align > 1)
            idxSize = (idxSize + mask) & ~mask;
        return idxSize + sizeof(Ftype) * nzvalSize();
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
    long long symV2ProbePairs = 0;
    long long symV2ProbePresentPairs = 0;
    long long symV2ProbeDirectPairs = 0;
    long long symV2ProbeFlops = 0;
    long long symV2ProbeDirectFlops = 0;
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
        SYM_V2_SETUP_RECV_CACHE_BUILD,
        SYM_V2_SETUP_PARTNER_RECV_INDEX_BUILD,
        SYM_V2_SETUP_PARTNER_RECV_LOOKUP_BUILD,
        SYM_V2_SETUP_ROW_RECV_INDEX_BUILD,
        SYM_V2_SETUP_ROW_RECV_LOOKUP_BUILD,
        SYM_V2_SETUP_EXACT_DEMAND_BUILD,
        SYM_V2_SETUP_EXACT_SEND_MAP_INDEX,
        SYM_V2_SETUP_EXACT_SEND_MAP_BUILD,
        SYM_V2_SETUP_ROW_COMPACT_DEMAND_BUILD,
        SYM_V2_SETUP_ROW_COMPACT_RECV_META_BUILD,
        SYM_V2_SETUP_ROW_COMPACT_SEND_MAP_BUILD,
        SYM_V2_SETUP_ROW_COMPACT_SIZE_CHECK,
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
    void printSymV2FrontProbe();
    void symV2ProbeLLRange(int_t k, xlpanel_t<Ftype> &lpanel,
                           int_t iSt, int_t iEnd,
                           int_t jSt, int_t jEnd,
                           const std::vector<int_t> *frag);
    enum SymV2FactorProfileId
    {
        SYM_V2_FACTOR_TREE_WALL = 0,
        SYM_V2_FACTOR_INITIAL_FACTOR_DISPATCH,
        SYM_V2_FACTOR_INITIAL_PANEL_BCAST,
        SYM_V2_FACTOR_SCHED_LOOKAHEAD_DISPATCH,
        SYM_V2_FACTOR_LOOKAHEAD_UPDATE,
        SYM_V2_FACTOR_LOOKAHEAD_SYNC,
        SYM_V2_FACTOR_SCHED_FACTOR_DISPATCH,
        SYM_V2_FACTOR_PARENT_FACTOR,
        SYM_V2_FACTOR_EXCLUDE_UPDATE,
        SYM_V2_FACTOR_BCAST_ADVANCE,
        SYM_V2_FACTOR_FINAL_SYNC,
        SYM_V2_FACTOR_DIAG_PANEL_SOLVE,
        SYM_V2_FACTOR_PANEL_BCAST,
        SYM_V2_FACTOR_PARTNER_L_EXCHANGE,
        SYM_V2_FACTOR_COUNT
    };

    int symV2FactorProfileEnabled = 0;
    int symV2FactorProfilePrinted = 0;
    double symV2FactorProfileTime[SYM_V2_FACTOR_COUNT] = {};
    long long symV2FactorProfileCount[SYM_V2_FACTOR_COUNT] = {};

    enum SymV2PayloadProfileId
    {
        SYM_V2_PAYLOAD_PANEL_CALL = 0,
        SYM_V2_PAYLOAD_PANEL_MPI,
        SYM_V2_PAYLOAD_PARTNER_CALL,
        SYM_V2_PAYLOAD_PARTNER_MPI_SEND,
        SYM_V2_PAYLOAD_PARTNER_MPI_RECV,
        SYM_V2_PAYLOAD_PARTNER_SELF,
        SYM_V2_PAYLOAD_ROWFRAG_CALL,
        SYM_V2_PAYLOAD_ROWFRAG_MPI_SEND,
        SYM_V2_PAYLOAD_ROWFRAG_MPI_RECV,
        SYM_V2_PAYLOAD_ROWFRAG_HOST_STAGING,
        SYM_V2_PAYLOAD_ROWFRAG_SELF,
        SYM_V2_PAYLOAD_COUNT
    };

    enum SymV2PayloadProfileBin
    {
        SYM_V2_PAYLOAD_ZERO = 0,
        SYM_V2_PAYLOAD_LE_1K,
        SYM_V2_PAYLOAD_LE_4K,
        SYM_V2_PAYLOAD_LE_16K,
        SYM_V2_PAYLOAD_LE_64K,
        SYM_V2_PAYLOAD_LE_256K,
        SYM_V2_PAYLOAD_LE_1M,
        SYM_V2_PAYLOAD_LE_4M,
        SYM_V2_PAYLOAD_LE_16M,
        SYM_V2_PAYLOAD_GT_16M,
        SYM_V2_PAYLOAD_BIN_COUNT
    };

    long long symV2PayloadProfileCount
        [SYM_V2_PAYLOAD_COUNT][SYM_V2_PAYLOAD_BIN_COUNT] = {};
    long long symV2PayloadProfileBytes
        [SYM_V2_PAYLOAD_COUNT][SYM_V2_PAYLOAD_BIN_COUNT] = {};
    long long symV2PayloadProfileMaxBytes
        [SYM_V2_PAYLOAD_COUNT][SYM_V2_PAYLOAD_BIN_COUNT] = {};

    enum SymV2ProfileScalarId
    {
        SYM_V2_PROFILE_GPU_USABLE_BYTES = 0,
        SYM_V2_PROFILE_GPU_PERSISTENT_BYTES,
        SYM_V2_PROFILE_GPU_DELAYED_METADATA_BYTES,
        SYM_V2_PROFILE_GPU_PER_STREAM_BASE_BYTES,
        SYM_V2_PROFILE_GPU_PER_STREAM_BYTES,
        SYM_V2_PROFILE_GPU_GEMM_BUFFER_BYTES,
        SYM_V2_PROFILE_GPU_GEMM_SHRINK_BYTES,
        SYM_V2_PROFILE_GPU_STREAMS,
        SYM_V2_PROFILE_GPU_RAW_W_CACHE_BYTES,
        SYM_V2_PROFILE_GPU_PARTNER_VALUE_BYTES,
        SYM_V2_PROFILE_GPU_PARTNER_INDEX_BYTES,
        SYM_V2_PROFILE_GPU_ROW_STAGE_BYTES,
        SYM_V2_PROFILE_GPU_ROW_RECV_VALUE_BYTES,
        SYM_V2_PROFILE_GPU_ROW_INDEX_BYTES,
        SYM_V2_PROFILE_GPU_ROW_SEND_STAGE_BYTES,
        SYM_V2_PROFILE_GPU_ROW_STAGE_REUSE,
        SYM_V2_PROFILE_GPU_ROW_STAGE_CHOSEN_BYTES,
        SYM_V2_PROFILE_GPU_DIAG_BYTES,
        SYM_V2_PROFILE_ROW_CURRENT_RECV_VALUES,
        SYM_V2_PROFILE_ROW_SPARSE_SEND_VALUES,
        SYM_V2_PROFILE_ROW_SPARSE_RECV_VALUES,
        SYM_V2_PROFILE_ROW_SAVED_RECV_VALUES,
        SYM_V2_PROFILE_ROW_DEMAND_RECORDS,
        SYM_V2_PROFILE_ROW_SEND_MESSAGES,
        SYM_V2_PROFILE_ROW_RECV_MESSAGES,
        SYM_V2_PROFILE_COUNT
    };

    long long symV2ProfileScalar[SYM_V2_PROFILE_COUNT] = {};

    void symV2ProfileScalarSet(SymV2ProfileScalarId id, long long value)
    {
        if (id < 0 || id >= SYM_V2_PROFILE_COUNT)
            return;
        symV2ProfileScalar[id] = value;
    }

    void symV2ProfileScalarAdd(SymV2ProfileScalarId id, long long value)
    {
        if (id < 0 || id >= SYM_V2_PROFILE_COUNT)
            return;
        symV2ProfileScalar[id] += value;
    }

    bool symV2FactorProfileActive() const
    {
        return symV2FactorProfileEnabled != 0;
    }

    void symV2FactorProfileAdd(SymV2FactorProfileId id, double elapsed)
    {
        if (!symV2FactorProfileActive())
            return;
        symV2FactorProfileTime[id] += elapsed;
        symV2FactorProfileCount[id] += 1;
    }

    int symV2PayloadProfileBin(long long bytes) const
    {
        if (bytes <= 0)
            return SYM_V2_PAYLOAD_ZERO;
        if (bytes <= 1024LL)
            return SYM_V2_PAYLOAD_LE_1K;
        if (bytes <= 4LL * 1024LL)
            return SYM_V2_PAYLOAD_LE_4K;
        if (bytes <= 16LL * 1024LL)
            return SYM_V2_PAYLOAD_LE_16K;
        if (bytes <= 64LL * 1024LL)
            return SYM_V2_PAYLOAD_LE_64K;
        if (bytes <= 256LL * 1024LL)
            return SYM_V2_PAYLOAD_LE_256K;
        if (bytes <= 1024LL * 1024LL)
            return SYM_V2_PAYLOAD_LE_1M;
        if (bytes <= 4LL * 1024LL * 1024LL)
            return SYM_V2_PAYLOAD_LE_4M;
        if (bytes <= 16LL * 1024LL * 1024LL)
            return SYM_V2_PAYLOAD_LE_16M;
        return SYM_V2_PAYLOAD_GT_16M;
    }

    void symV2PayloadProfileAdd(SymV2PayloadProfileId id, long long bytes)
    {
        if (!symV2FactorProfileActive())
            return;
        if (id < 0 || id >= SYM_V2_PAYLOAD_COUNT)
            return;
        if (bytes < 0)
            bytes = 0;
        int bin = symV2PayloadProfileBin(bytes);
        symV2PayloadProfileCount[id][bin] += 1;
        symV2PayloadProfileBytes[id][bin] += bytes;
        if (bytes > symV2PayloadProfileMaxBytes[id][bin])
            symV2PayloadProfileMaxBytes[id][bin] = bytes;
    }

    struct SymV2FactorProfileScope
    {
        xLUstruct_t<Ftype> *owner;
        SymV2FactorProfileId id;
        bool active;
        double start;

        SymV2FactorProfileScope(xLUstruct_t<Ftype> *owner_,
                                SymV2FactorProfileId id_)
            : owner(owner_), id(id_),
              active(owner_ != NULL && owner_->symV2FactorProfileActive()),
              start(active ? SuperLU_timer_() : 0.0) {}

        ~SymV2FactorProfileScope()
        {
            if (active)
                owner->symV2FactorProfileAdd(id, SuperLU_timer_() - start);
        }
    };

    void printSymV2FactorProfile();
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
        SYM_GPU3D_T_LFRAG_RECV_POST,
        SYM_GPU3D_T_LFRAG_MPI_RECV_WAIT,
        SYM_GPU3D_T_LFRAG_H2D_STAGE_ISSUE,
        SYM_GPU3D_T_LFRAG_ASSEMBLE_ISSUE,
        SYM_GPU3D_T_LFRAG_SEND_POST,
        SYM_GPU3D_T_LFRAG_SEND_WAIT,
        SYM_GPU3D_T_LFRAG_STREAM_SYNC,
        SYM_GPU3D_T_PARTNER_LFRAG_PACK_ISSUE,
        SYM_GPU3D_T_PARTNER_LFRAG_D2H_STAGE_ISSUE,
        SYM_GPU3D_T_PARTNER_LFRAG_PACK_STAGE_SYNC,
        SYM_GPU3D_T_PARTNER_LFRAG_RECV_POST,
        SYM_GPU3D_T_PARTNER_LFRAG_MPI_RECV_WAIT,
        SYM_GPU3D_T_PARTNER_LFRAG_H2D_STAGE_ISSUE,
        SYM_GPU3D_T_PARTNER_LFRAG_ASSEMBLE_ISSUE,
        SYM_GPU3D_T_PARTNER_LFRAG_SEND_POST,
        SYM_GPU3D_T_PARTNER_LFRAG_SEND_WAIT,
        SYM_GPU3D_T_ROW_LFRAG_PACK_ISSUE,
        SYM_GPU3D_T_ROW_LFRAG_D2H_STAGE_ISSUE,
        SYM_GPU3D_T_ROW_LFRAG_PACK_STAGE_SYNC,
        SYM_GPU3D_T_ROW_LFRAG_RECV_POST,
        SYM_GPU3D_T_ROW_LFRAG_MPI_RECV_WAIT,
        SYM_GPU3D_T_ROW_LFRAG_H2D_STAGE_ISSUE,
        SYM_GPU3D_T_ROW_LFRAG_ASSEMBLE_ISSUE,
        SYM_GPU3D_T_ROW_LFRAG_SEND_POST,
        SYM_GPU3D_T_ROW_LFRAG_SEND_WAIT,
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
        SYM_GPU3D_T_SCHED_LOOKAHEAD_BOOKKEEP,
        SYM_GPU3D_T_SCHED_FACTOR_BOOKKEEP,
        SYM_GPU3D_T_SCHED_BCAST_BOOKKEEP,
        SYM_GPU3D_T_SCHED_FINAL_SYNC_BOOKKEEP,
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
        SYM_GPU3D_S_L2U_RECV_REQUESTS,
        SYM_GPU3D_S_L2U_SEND_REQUESTS,
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

    void symTimingAddBoth(SymGPU3DTimingId aggregate,
                          SymGPU3DTimingId specific,
                          double elapsed)
    {
        symTimingAdd(aggregate, elapsed);
        symTimingAdd(specific, elapsed);
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
    std::vector<double *> symV2PartnerLExactSendBufsGPU;
    std::vector<int_t *> symV2PartnerLExactSendMapsGPU;
    std::vector<double *> symV2RowFragExactSendBufsGPU;
    std::vector<int_t *> symV2RowFragExactSendMapsGPU;
    double *symV2PartnerLSendBufPoolGPU;
    int_t *symL2LSendMapPoolGPU;
    double *symV2PartnerLExactSendBufPoolGPU;
    int_t *symV2PartnerLExactSendMapPoolGPU;
    double *symV2RowFragExactSendBufPoolGPU;
    int_t *symV2RowFragExactSendMapPoolGPU;
    int_t *symV2PartnerLRecvMapPoolGPU;
    int_t *symV2RowFragRecvMapPoolGPU;
// SYM_V2_PC2_PHASE4_XLU_ROW_DOWN_GPU_POOL_BEGIN
    int_t *symV2RowDownSendMapPoolGPU;
// SYM_V2_PC2_PHASE4_XLU_ROW_DOWN_GPU_POOL_END

    size_t symV2PartnerLSendBufPoolCount;
    size_t symL2LSendMapPoolCount;
    size_t symV2PartnerLExactSendBufPoolCount;
    size_t symV2PartnerLExactSendMapPoolCount;
    size_t symV2RowFragExactSendBufPoolCount;
    size_t symV2RowFragExactSendMapPoolCount;
    size_t symV2PartnerLRecvMapPoolCount;
    size_t symV2RowFragRecvMapPoolCount;
// SYM_V2_PC2_PHASE4_XLU_ROW_DOWN_GPU_COUNT_BEGIN
    size_t symV2RowDownSendMapPoolCount;
    std::vector<int_t *> symV2RowDownSendMapsGPU;
// SYM_V2_PC2_PHASE4_XLU_ROW_DOWN_GPU_COUNT_END

    std::vector<std::vector<int_t> > symL2LSendMeta;
    std::vector<std::vector<double> > symV2PartnerLHostSendBufs;
    std::vector<double *> symV2PartnerLHostSendBufsPinned;
    std::vector<std::vector<double> > symV2PartnerLExactHostSendBufs;
    std::vector<double *> symV2PartnerLExactHostSendBufsPinned;
    std::vector<std::vector<double> > symV2RowFragExactHostSendBufs;
    std::vector<double *> symV2RowFragExactHostSendBufsPinned;
    std::vector<std::vector<int_t> > symL2USendMapsHost;
    std::vector<std::vector<int_t> > symL2ULocalMapsHost;
    std::vector<size_t> symV2PartnerLMapOffsets;
    std::vector<int_t> symV2PartnerLPackedMaps;
    std::vector<int_t> symV2PartnerLExactSendMapsHost;
    std::vector<size_t> symV2PartnerLExactSendMapOffsets;
    std::vector<int_t> symV2RowFragExactSendMapsHost;
    std::vector<size_t> symV2RowFragExactSendMapOffsets;
    std::vector<int> symV2RowDirectSendSizes;
    std::vector<size_t> symV2RowDirectSendMapOffsets;
    std::vector<int_t> symV2RowDirectSendMapsHost;
    std::vector<std::vector<int_t> > symV2RowDirectSendBlocksHost;
    std::vector<std::vector<int_t> > symV2RowDirectSendMapScratchHost;
// SYM_V2_PC2_PHASE3_XLU_ROW_DOWN_PLAN_BEGIN
    struct SymV2RowDownSeg
    {
        int_t gid;
        int_t chunk_pc;
        int_t nrows;
        int_t dst_row_offset;
        int_t value_count;
        size_t map_offset;
    };
    std::vector<int> symV2RowDownSendSizes;
    std::vector<size_t> symV2RowDownSendMapOffsets;
    std::vector<int_t> symV2RowDownSendMapsHost;
    SymV2RowDownSendSegmentGPU *symV2RowDownSendSegPoolGPU = NULL;
    size_t symV2RowDownSendSegPoolCount = 0;
    std::vector<SymV2RowDownSendSegmentGPU> symV2RowDownSendSegsHost;
    std::vector<size_t> symV2RowDownSendSegOffsets;
    std::vector<int> symV2RowDownSendSegCounts;
    std::vector<SymV2RowDownSendSegmentGPU *> symV2RowDownSendSegsGPU;
    std::vector<size_t> symV2RowDownSegOffsets;
    std::vector<SymV2RowDownSeg> symV2RowDownSegs;
    std::vector<int> symV2RowDownRecvSizes;
    std::vector<unsigned char> symV2RowDownPlanReady;
    long long symV2RowDownSparseSendValues = 0;
    long long symV2RowDownSparseRecvValues = 0;
    long long symV2RowDownCurrentRecvValues = 0;
    long long symV2RowDownDemandRecords = 0;
    long long symV2RowDownSendMessages = 0;
    long long symV2RowDownRecvMessages = 0;
    double symV2RowDownSetupSeconds = 0.0;
// SYM_V2_PC2_PHASE3_XLU_ROW_DOWN_PLAN_END

    double *symV2PartnerLHostSendPoolPinned = NULL;
    Ftype *symV2PartnerLHostRecvPoolPinned = NULL;
    Ftype *symV2RowFragHostRecvPoolPinned = NULL;
// SYM_V2_PC2_PHASE1_XLU_SEND_POOL_BEGIN
    Ftype *symV2RowFragHostSendPoolPinned = NULL;
// SYM_V2_PC2_PHASE1_XLU_SEND_POOL_END

    size_t symV2PartnerLHostSendPoolPinnedCount = 0;
    size_t symV2PartnerLHostRecvPoolPinnedCount = 0;
    size_t symV2RowFragHostRecvPoolPinnedCount = 0;
// SYM_V2_PC2_PHASE1_XLU_SEND_POOL_COUNT_BEGIN
    size_t symV2RowFragHostSendPoolPinnedCount = 0;
// SYM_V2_PC2_PHASE1_XLU_SEND_POOL_COUNT_END

    int symV2PartnerLHostRecvPinned = 0;
    int symV2RowFragHostRecvPinned = 0;
// SYM_V2_PC2_PHASE1_XLU_SEND_PINNED_BEGIN
    int symV2RowFragHostSendPinned = 0;
// SYM_V2_PC2_PHASE1_XLU_SEND_PINNED_END

    std::vector<size_t> symV2PartnerLHostSendScratchOffsets;
    std::vector<int> symV2ExchangeSendSizesScratch;
    std::vector<int> symV2ExchangeRecvSizesScratch;
    std::vector<int> symV2ExchangeRecvOffsetsScratch;
    std::vector<MPI_Request> symV2ExchangeRecvReqsScratch;
    std::vector<MPI_Request> symV2ExchangeSendReqsScratch;
    std::vector<int> symV2ExchangeRecvPeersScratch;
    std::vector<int> symV2ExchangeWaitIndicesScratch;
    std::vector<MPI_Status> symV2ExchangeWaitStatusesScratch;
// SYM_V2_PC2_PHASE1_XLU_SEND_SCRATCH_BEGIN
    std::vector<int> symV2RowFragSendCountsScratch;
    std::vector<int> symV2RowFragSendOffsetsScratch;
    std::vector<MPI_Request> symV2RowFragSendReqsScratch;
// SYM_V2_PC2_PHASE1_XLU_SEND_SCRATCH_END

    std::vector<int> symV2PartnerLSendSizes;
    std::vector<int> symV2PartnerLExactSendSizes;
    std::vector<int> symV2RowFragExactSendSizes;
    std::vector<unsigned char> symV2PartnerLSendRowActive;
    std::vector<unsigned char> symV2RowFragSendActive;
    std::vector<unsigned char> symV2PartnerLPrepacked;
    std::vector<int> symV2PartnerLRecvSizes;
    std::vector<std::vector<int_t> > symV2PartnerLRecvIndex;
    std::vector<std::vector<int_t> > symV2PartnerLRecvIndexBySrc;
    std::vector<std::vector<int_t> > symV2PartnerLRecvMap;
    std::vector<size_t> symV2PartnerLRecvMapOffsets;
    std::vector<int_t *> symV2PartnerLRecvMapsGPU;
    std::vector<int> symV2RowFragRecvSizes;
    std::vector<std::vector<int_t> > symV2RowFragRecvIndex;
    std::vector<std::vector<int_t> > symV2RowFragRecvMap;
    std::vector<size_t> symV2RowFragRecvMapOffsets;
    std::vector<int_t *> symV2RowFragRecvMapsGPU;
    std::vector<int_t *> symL2ULocalMapsGPU;
    std::vector<int> symPanelReadyEventIds;
    std::vector<unsigned char> symV2UsePcFragmentSchur;
    std::vector<int_t> symV2RawPanelNodes;
// SYM_V2_PC2_PHASE6_XLU_EXCHANGE_STATE_BEGIN
    struct SymV2RowExchangeState
    {
        std::vector<MPI_Request> send_reqs;
        std::vector<MPI_Request> recv_reqs;
        std::vector<int> send_counts;
        std::vector<int> send_offsets;
        std::vector<int> recv_counts;
        std::vector<int> recv_offsets;
#ifdef HAVE_CUDA
        cudaEvent_t pack_done;
        cudaEvent_t d2h_done;
        cudaEvent_t h2d_done;
#endif
        int_t active_k;
        int active;
        SymV2RowExchangeState()
#ifdef HAVE_CUDA
            : pack_done(NULL), d2h_done(NULL), h2d_done(NULL),
              active_k(-1), active(0) {}
#else
            : active_k(-1), active(0) {}
#endif
    };
    std::vector<SymV2RowExchangeState> symV2RowExchangeStates;
// SYM_V2_PC2_PHASE6_XLU_EXCHANGE_STATE_END

// SYM_V2_PCFRAG_TASKFLOW_STATE_BEGIN
    enum SymV2PcFragPieceKind
    {
        SYM_V2_PCFRAG_PIECE_ROW = 0,
        SYM_V2_PCFRAG_PIECE_PARTNER = 1
    };

    enum SymV2PcFragTaskMode
    {
        SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL = 1,
        SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW = 2,
        SYM_V2_PCFRAG_TASK_EXCLUDE = 4,
        SYM_V2_PCFRAG_TASK_FULL = 8
    };

    enum SymV2PcFragTaskStreamKind
    {
        SYM_V2_PCFRAG_TASK_STREAM_NONE = 0,
        SYM_V2_PCFRAG_TASK_STREAM_MAIN = 1,
        SYM_V2_PCFRAG_TASK_STREAM_LOOKAHEAD_L = 2,
        SYM_V2_PCFRAG_TASK_STREAM_LOOKAHEAD_U = 3,
        SYM_V2_PCFRAG_TASK_STREAM_COUNT = 4
    };

    enum SymV2PcFragTaskGemmResourceKind
    {
        SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_NONE = 0,
        SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_MAIN = 1,
        SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_LOOKAHEAD_L = 2,
        SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_COUNT = 3
    };

    struct SymV2PcFragOutputKey
    {
        int_t gj;
        int_t gi;
        int_t local_panel_j;
        int_t local_block_i;
        int_t output_id;

        SymV2PcFragOutputKey()
            : gj(GLOBAL_BLOCK_NOT_FOUND), gi(GLOBAL_BLOCK_NOT_FOUND),
              local_panel_j(GLOBAL_BLOCK_NOT_FOUND),
              local_block_i(GLOBAL_BLOCK_NOT_FOUND),
              output_id(GLOBAL_BLOCK_NOT_FOUND)
        {
        }

        SymV2PcFragOutputKey(int_t gj_, int_t gi_)
            : gj(gj_), gi(gi_), local_panel_j(GLOBAL_BLOCK_NOT_FOUND),
              local_block_i(GLOBAL_BLOCK_NOT_FOUND),
              output_id(GLOBAL_BLOCK_NOT_FOUND)
        {
        }

        bool equals(const SymV2PcFragOutputKey &other) const
        {
            return gj == other.gj && gi == other.gi &&
                   local_panel_j == other.local_panel_j &&
                   local_block_i == other.local_block_i;
        }

        bool operator<(const SymV2PcFragOutputKey &other) const
        {
            if (gj != other.gj)
                return gj < other.gj;
            if (gi != other.gi)
                return gi < other.gi;
            if (local_panel_j != other.local_panel_j)
                return local_panel_j < other.local_panel_j;
            return local_block_i < other.local_block_i;
        }
    };

    struct SymV2PcFragPieceDesc
    {
        int_t k;
        unsigned char kind;
        int piece_id;
        int_t frag_blk_begin;
        int_t frag_blk_end;
        int_t frag_row_offset;
        int_t nblocks;
        int_t gid_first;
        int_t gid_last;
        int_t nrows;
        int_t ksupc;
        int_t lda;
        int_t index_count;
        int_t value_count;
        int_t filled_rows;
        std::vector<int_t> h_index;
        int_t *d_index;
        Ftype *d_val;
        unsigned char ready;
        unsigned char released;
        int pending_consumers;
#ifdef HAVE_CUDA
        cudaEvent_t ready_event;
        cudaEvent_t done_event;
#endif

        SymV2PcFragPieceDesc()
            : k(-1), kind(SYM_V2_PCFRAG_PIECE_ROW), piece_id(-1),
              frag_blk_begin(0), frag_blk_end(0), frag_row_offset(0),
              nblocks(0),
              gid_first(GLOBAL_BLOCK_NOT_FOUND),
              gid_last(GLOBAL_BLOCK_NOT_FOUND), nrows(0), ksupc(0),
              lda(0), index_count(0), value_count(0), filled_rows(0),
              d_index(NULL), d_val(NULL),
              ready(0), released(0), pending_consumers(0)
#ifdef HAVE_CUDA
              , ready_event(NULL), done_event(NULL)
#endif
        {
        }
    };

    struct SymV2PcFragTaskDesc
    {
        int_t k;
        int task_id;
        int row_piece;
        int partner_piece;
        int_t row_piece_blk_begin;
        int_t row_piece_blk_end;
        int_t partner_piece_blk_begin;
        int_t partner_piece_blk_end;
        int_t gemm_m;
        int_t gemm_n;
        int_t gemm_k;
        unsigned char mode_mask;
        int scatter_group;
        int lookahead_col_gid_index;
        int lookahead_row_gid_index;
        int output_begin;
        int output_count;
        unsigned char launched;
        unsigned char complete;
        unsigned char launch_stream_kind;
        unsigned char gemm_resource_kind;
#ifdef HAVE_CUDA
        cudaEvent_t done_event;
#endif

        SymV2PcFragTaskDesc()
            : k(-1), task_id(-1), row_piece(-1), partner_piece(-1),
              row_piece_blk_begin(0), row_piece_blk_end(0),
              partner_piece_blk_begin(0), partner_piece_blk_end(0),
              gemm_m(0), gemm_n(0), gemm_k(0), mode_mask(0),
              scatter_group(-1), lookahead_col_gid_index(-1),
              lookahead_row_gid_index(-1), output_begin(0),
              output_count(0), launched(0), complete(0),
              launch_stream_kind(SYM_V2_PCFRAG_TASK_STREAM_NONE),
              gemm_resource_kind(SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_NONE)
#ifdef HAVE_CUDA
              , done_event(NULL)
#endif
        {
        }
    };

    struct SymV2PcFragPairTaskEntry
    {
        int row_piece;
        int partner_piece;
        int task_id;

        SymV2PcFragPairTaskEntry()
            : row_piece(-1), partner_piece(-1), task_id(-1)
        {
        }

        SymV2PcFragPairTaskEntry(int row_piece_, int partner_piece_,
                                 int task_id_)
            : row_piece(row_piece_), partner_piece(partner_piece_),
              task_id(task_id_)
        {
        }

        bool operator<(const SymV2PcFragPairTaskEntry &other) const
        {
            if (row_piece != other.row_piece)
                return row_piece < other.row_piece;
            return partner_piece < other.partner_piece;
        }
    };

    struct SymV2PcFragLaunchedTaskGroup
    {
        int group_id;
        int task_begin;
        int task_count;
        unsigned char launch_stream_kind;
        unsigned char gemm_resource_kind;
#ifdef HAVE_CUDA
        cudaEvent_t done_event;
        int_t *d_group_index_pool;
        Ftype *d_group_value_pool;
#endif
        size_t group_index_pool_capacity;
        size_t group_value_pool_capacity;

        SymV2PcFragLaunchedTaskGroup()
            : group_id(-1), task_begin(0), task_count(0),
              launch_stream_kind(SYM_V2_PCFRAG_TASK_STREAM_NONE),
              gemm_resource_kind(SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_NONE)
#ifdef HAVE_CUDA
              , done_event(NULL), d_group_index_pool(NULL),
              d_group_value_pool(NULL)
#endif
              , group_index_pool_capacity(0), group_value_pool_capacity(0)
        {
        }
    };

    struct SymV2PcFragGidCounterMap
    {
        struct Entry
        {
            int_t first;
            int second;

            Entry() : first(-1), second(0) {}
            Entry(int_t gid, int value) : first(gid), second(value) {}

            bool operator<(const Entry &other) const
            {
                return first < other.first;
            }
        };

        std::vector<Entry> entries;
        typedef typename std::vector<Entry>::iterator iterator;
        typedef typename std::vector<Entry>::const_iterator const_iterator;

        void clear() { entries.clear(); }
        void reserve(size_t count) { entries.reserve(count); }
        size_t size() const { return entries.size(); }
        iterator end() { return entries.end(); }
        const_iterator end() const { return entries.end(); }

        void assign_zero_from(const SymV2PcFragGidCounterMap &degrees)
        {
            entries.resize(degrees.entries.size());
            for (size_t i = 0; i < degrees.entries.size(); ++i)
                entries[i] = Entry(degrees.entries[i].first, 0);
        }

        iterator find(int_t gid)
        {
            iterator it = lower_bound(gid);
            return (it != entries.end() && it->first == gid) ? it
                                                             : entries.end();
        }

        const_iterator find(int_t gid) const
        {
            const_iterator it = lower_bound(gid);
            return (it != entries.end() && it->first == gid) ? it
                                                             : entries.end();
        }

        int &operator[](int_t gid)
        {
            iterator it = lower_bound(gid);
            if (it == entries.end() || it->first != gid)
                it = entries.insert(it, Entry(gid, 0));
            return it->second;
        }

        int index_of(int_t gid) const
        {
            const_iterator it = lower_bound(gid);
            if (it == entries.end() || it->first != gid)
                return -1;
            size_t idx = static_cast<size_t>(it - entries.begin());
            if (idx > static_cast<size_t>(std::numeric_limits<int>::max()))
                return -1;
            return static_cast<int>(idx);
        }

        int_t gid_at(int index) const
        {
            return entries[static_cast<size_t>(index)].first;
        }

        int &value_at(int index)
        {
            return entries[static_cast<size_t>(index)].second;
        }

        const int &value_at(int index) const
        {
            return entries[static_cast<size_t>(index)].second;
        }

      private:
        iterator lower_bound(int_t gid)
        {
            return std::lower_bound(entries.begin(), entries.end(),
                                    Entry(gid, 0));
        }

        const_iterator lower_bound(int_t gid) const
        {
            return std::lower_bound(entries.begin(), entries.end(),
                                    Entry(gid, 0));
        }
    };

    struct SymV2PcFragGidTaskQueueMap
    {
        struct Entry
        {
            int_t first;
            std::vector<int> second;

            Entry() : first(-1), second() {}
            Entry(int_t gid) : first(gid), second() {}

            bool operator<(const Entry &other) const
            {
                return first < other.first;
            }
        };

        std::vector<Entry> entries;
        typedef typename std::vector<Entry>::iterator iterator;
        typedef typename std::vector<Entry>::const_iterator const_iterator;

        void clear() { entries.clear(); }
        void reserve(size_t count) { entries.reserve(count); }
        size_t size() const { return entries.size(); }
        iterator end() { return entries.end(); }
        const_iterator end() const { return entries.end(); }

        void assign_from_degrees(const SymV2PcFragGidCounterMap &degrees)
        {
            entries.clear();
            entries.reserve(degrees.entries.size());
            for (typename SymV2PcFragGidCounterMap::const_iterator it =
                     degrees.entries.begin();
                 it != degrees.entries.end(); ++it)
            {
                entries.push_back(Entry(it->first));
                if (it->second < 0)
                    ABORT("GPU3DV2_PCFRAG_TASKFLOW runnable gid degree is negative.");
                entries.back().second.reserve(
                    static_cast<size_t>(it->second));
            }
        }

        iterator find(int_t gid)
        {
            iterator it = lower_bound(gid);
            return (it != entries.end() && it->first == gid) ? it
                                                             : entries.end();
        }

        const_iterator find(int_t gid) const
        {
            const_iterator it = lower_bound(gid);
            return (it != entries.end() && it->first == gid) ? it
                                                             : entries.end();
        }

        std::vector<int> &operator[](int_t gid)
        {
            iterator it = lower_bound(gid);
            if (it == entries.end() || it->first != gid)
                it = entries.insert(it, Entry(gid));
            return it->second;
        }

        int_t gid_at(int index) const
        {
            return entries[static_cast<size_t>(index)].first;
        }

        std::vector<int> &value_at(int index)
        {
            return entries[static_cast<size_t>(index)].second;
        }

        const std::vector<int> &value_at(int index) const
        {
            return entries[static_cast<size_t>(index)].second;
        }

      private:
        iterator lower_bound(int_t gid)
        {
            return std::lower_bound(entries.begin(), entries.end(),
                                    Entry(gid));
        }

        const_iterator lower_bound(int_t gid) const
        {
            return std::lower_bound(entries.begin(), entries.end(),
                                    Entry(gid));
        }
    };

    struct SymV2PcFragPanelTaskState
    {
        int_t k;
        int stream_offset;
        unsigned char initialized;
        unsigned char exchange_posted;
        unsigned char closed;
        std::vector<SymV2PcFragPieceDesc> row_pieces;
        std::vector<SymV2PcFragPieceDesc> partner_pieces;
        std::vector<SymV2PcFragTaskDesc> tasks;
        std::vector<SymV2PcFragOutputKey> task_output_pool;
        std::vector<int> row_block_piece;
        std::vector<int> partner_block_piece;
        std::vector<SymV2PcFragPairTaskEntry> pair_task_entries;
        std::vector<int> row_piece_task_offsets;
        std::vector<int> row_piece_task_ids;
        std::vector<int> partner_piece_task_offsets;
        std::vector<int> partner_piece_task_ids;
        std::vector<unsigned char> task_ready_inputs;
        std::vector<unsigned char> task_enqueued;
        std::vector<int> runnable_task_ids;
        std::vector<int> runnable_task_ids_by_mode[16];
        SymV2PcFragGidTaskQueueMap runnable_lookahead_col_by_gid;
        SymV2PcFragGidTaskQueueMap runnable_lookahead_row_by_gid;
        SymV2PcFragGidCounterMap incomplete_lookahead_col_members_by_gid;
        SymV2PcFragGidCounterMap incomplete_lookahead_row_members_by_gid;
        SymV2PcFragGidCounterMap launched_lookahead_col_members_by_gid;
        SymV2PcFragGidCounterMap launched_lookahead_row_members_by_gid;
        SymV2PcFragGidCounterMap launched_lookahead_col_members_by_gid_by_stream[
            SYM_V2_PCFRAG_TASK_STREAM_COUNT];
        SymV2PcFragGidCounterMap launched_lookahead_row_members_by_gid_by_stream[
            SYM_V2_PCFRAG_TASK_STREAM_COUNT];
        std::vector<int> launched_task_ids_by_stream[
            SYM_V2_PCFRAG_TASK_STREAM_COUNT];
        std::vector<SymV2PcFragLaunchedTaskGroup>
            launched_task_groups_by_stream[SYM_V2_PCFRAG_TASK_STREAM_COUNT];
        std::vector<int> launched_group_task_ids;
        int launched_task_pending_by_stream[
            SYM_V2_PCFRAG_TASK_STREAM_COUNT];
        int launched_task_pending_mode_by_stream[
            SYM_V2_PCFRAG_TASK_STREAM_COUNT][16];
        std::set<SymV2PcFragOutputKey> active_output_key_set;
        int active_output_lock_count;
        int task_event_poll_skip[SYM_V2_PCFRAG_TASK_STREAM_COUNT];
        int task_event_poll_backoff[SYM_V2_PCFRAG_TASK_STREAM_COUNT];
        int incomplete_task_count;
        int producer_tasks_launched;
        unsigned char producer_launch_cap_reported;
        unsigned char producer_exchange_active;
        unsigned char producer_exchange_pending;
        unsigned char producer_stream_pending;
        unsigned char output_conflicts_possible;
        unsigned char group_scratch_in_use;
        size_t row_pieces_ready_count;
        size_t partner_pieces_ready_count;
        int producer_partner_recv_remaining;
        int producer_row_recv_remaining;
        int producer_ksupc;
        std::vector<MPI_Request> producer_partner_recv_reqs;
        std::vector<int> producer_partner_recv_prs;
        std::vector<int> producer_partner_recv_sizes;
        std::vector<int> producer_partner_recv_offsets;
        std::vector<unsigned char> producer_partner_recv_done;
        std::vector<unsigned char> producer_partner_progressive_assembled;
        std::vector<MPI_Request> producer_row_recv_reqs;
        std::vector<int> producer_row_recv_pcs;
        std::vector<int> producer_row_recv_sizes;
        std::vector<int> producer_row_recv_offsets;
        std::vector<unsigned char> producer_row_recv_done;
        std::vector<MPI_Request> producer_send_reqs;
        std::vector<int> producer_progress_indices;
        std::vector<MPI_Status> producer_progress_statuses;
#ifdef HAVE_CUDA
        Ftype *producer_partner_recv_host_values;
        Ftype *producer_row_recv_host_values;
        Ftype *producer_partner_send_host_values;
        Ftype *producer_row_send_host_values;
        cudaEvent_t producer_last_ready_event;
        int_t *d_index_pool;
        Ftype *d_value_pool;
        int_t *d_group_index_pool;
        Ftype *d_group_value_pool;
#endif
        size_t producer_partner_recv_host_capacity;
        size_t producer_row_recv_host_capacity;
        size_t producer_partner_send_host_capacity;
        size_t producer_row_send_host_capacity;
        size_t index_pool_capacity;
        size_t value_pool_capacity;
        size_t group_index_pool_capacity;
        size_t group_value_pool_capacity;
        size_t index_pool_used;
        size_t value_pool_used;

        SymV2PcFragPanelTaskState()
            : k(-1), stream_offset(-1), initialized(0),
              exchange_posted(0), closed(0), incomplete_task_count(0),
              producer_tasks_launched(0), producer_launch_cap_reported(0),
              producer_exchange_active(0), producer_exchange_pending(0),
              producer_stream_pending(0), output_conflicts_possible(0),
              group_scratch_in_use(0), active_output_lock_count(0),
              row_pieces_ready_count(0), partner_pieces_ready_count(0),
              producer_partner_recv_remaining(0),
              producer_row_recv_remaining(0), producer_ksupc(0)
#ifdef HAVE_CUDA
              , producer_partner_recv_host_values(NULL),
              producer_row_recv_host_values(NULL),
              producer_partner_send_host_values(NULL),
              producer_row_send_host_values(NULL),
              producer_last_ready_event(NULL),
              d_index_pool(NULL), d_value_pool(NULL),
              d_group_index_pool(NULL), d_group_value_pool(NULL)
#endif
              , producer_partner_recv_host_capacity(0),
              producer_row_recv_host_capacity(0),
              producer_partner_send_host_capacity(0),
              producer_row_send_host_capacity(0),
              index_pool_capacity(0), value_pool_capacity(0),
              group_index_pool_capacity(0), group_value_pool_capacity(0),
              index_pool_used(0), value_pool_used(0)
        {
            for (int i = 0; i < SYM_V2_PCFRAG_TASK_STREAM_COUNT; ++i)
            {
                task_event_poll_skip[i] = 0;
                task_event_poll_backoff[i] = 0;
                launched_task_pending_by_stream[i] = 0;
                for (int mask = 0; mask < 16; ++mask)
                    launched_task_pending_mode_by_stream[i][mask] = 0;
            }
        }

        void note_piece_ready(unsigned char kind, int piece_id)
        {
            const std::vector<int> &offsets =
                (kind == SYM_V2_PCFRAG_PIECE_ROW)
                    ? row_piece_task_offsets
                    : partner_piece_task_offsets;
            const std::vector<int> &task_ids =
                (kind == SYM_V2_PCFRAG_PIECE_ROW)
                    ? row_piece_task_ids
                    : partner_piece_task_ids;
            if (piece_id < 0 ||
                static_cast<size_t>(piece_id + 1) >= offsets.size())
                return;
            int begin = offsets[static_cast<size_t>(piece_id)];
            int end = offsets[static_cast<size_t>(piece_id + 1)];
            if (begin < 0 || end < begin ||
                static_cast<size_t>(end) > task_ids.size())
                ABORT("GPU3DV2_PCFRAG_TASKFLOW piece task CSR is invalid.");
            for (int i = begin; i < end; ++i)
            {
                int tid = task_ids[static_cast<size_t>(i)];
                if (tid < 0 || static_cast<size_t>(tid) >= tasks.size())
                    continue;
                size_t pos = static_cast<size_t>(tid);
                if (pos >= task_ready_inputs.size() ||
                    pos >= task_enqueued.size())
                    continue;
                if (task_ready_inputs[pos] < 2)
                    ++task_ready_inputs[pos];
                if (task_ready_inputs[pos] == 2 &&
                    !task_enqueued[pos])
                {
                    if (runnable_task_ids.size() >=
                        runnable_task_ids.capacity())
                        ABORT("GPU3DV2_PCFRAG_TASKFLOW runnable queue capacity is undersized.");
                    runnable_task_ids.push_back(tid);
                    unsigned char mode_mask = tasks[pos].mode_mask;
                    for (int mask = 1; mask < 16; mask <<= 1)
                    {
                        if ((mode_mask & mask) == 0)
                            continue;
                        if (runnable_task_ids_by_mode[mask].size() >=
                            runnable_task_ids_by_mode[mask].capacity())
                            ABORT("GPU3DV2_PCFRAG_TASKFLOW mode runnable queue capacity is undersized.");
                        runnable_task_ids_by_mode[mask].push_back(tid);
                    }
                    if (mode_mask & SYM_V2_PCFRAG_TASK_LOOKAHEAD_COL)
                    {
                        if (tasks[pos].output_count == 1 &&
                            tasks[pos].lookahead_col_gid_index >= 0)
                        {
                            int idx = tasks[pos].lookahead_col_gid_index;
                            size_t out_pos =
                                static_cast<size_t>(tasks[pos].output_begin);
                            if (out_pos >= task_output_pool.size())
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW task output pool index is invalid.");
                            if (static_cast<size_t>(idx) >=
                                    runnable_lookahead_col_by_gid.size() ||
                                runnable_lookahead_col_by_gid.gid_at(idx) !=
                                    task_output_pool[out_pos].gj)
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead column runnable compact gid is invalid.");
                            std::vector<int> &queue =
                                runnable_lookahead_col_by_gid.value_at(idx);
                            if (queue.size() >= queue.capacity())
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead column runnable queue capacity is undersized.");
                            queue.push_back(tid);
                        }
                        else
                        {
                            for (int o = 0; o < tasks[pos].output_count; ++o)
                            {
                                size_t out_pos =
                                    static_cast<size_t>(
                                        tasks[pos].output_begin + o);
                                if (out_pos >= task_output_pool.size())
                                    ABORT("GPU3DV2_PCFRAG_TASKFLOW task output pool index is invalid.");
                                int_t gid = task_output_pool[out_pos].gj;
                                auto it =
                                    runnable_lookahead_col_by_gid.find(gid);
                                if (it == runnable_lookahead_col_by_gid.end())
                                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead column runnable map is missing a gid.");
                                if (it->second.size() >= it->second.capacity())
                                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead column runnable queue capacity is undersized.");
                                it->second.push_back(tid);
                            }
                        }
                    }
                    if (mode_mask & SYM_V2_PCFRAG_TASK_LOOKAHEAD_ROW)
                    {
                        if (tasks[pos].output_count == 1 &&
                            tasks[pos].lookahead_row_gid_index >= 0)
                        {
                            int idx = tasks[pos].lookahead_row_gid_index;
                            size_t out_pos =
                                static_cast<size_t>(tasks[pos].output_begin);
                            if (out_pos >= task_output_pool.size())
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW task output pool index is invalid.");
                            if (static_cast<size_t>(idx) >=
                                    runnable_lookahead_row_by_gid.size() ||
                                runnable_lookahead_row_by_gid.gid_at(idx) !=
                                    task_output_pool[out_pos].gi)
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead row runnable compact gid is invalid.");
                            std::vector<int> &queue =
                                runnable_lookahead_row_by_gid.value_at(idx);
                            if (queue.size() >= queue.capacity())
                                ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead row runnable queue capacity is undersized.");
                            queue.push_back(tid);
                        }
                        else
                        {
                            for (int o = 0; o < tasks[pos].output_count; ++o)
                            {
                                size_t out_pos =
                                    static_cast<size_t>(
                                        tasks[pos].output_begin + o);
                                if (out_pos >= task_output_pool.size())
                                    ABORT("GPU3DV2_PCFRAG_TASKFLOW task output pool index is invalid.");
                                int_t gid = task_output_pool[out_pos].gi;
                                auto it =
                                    runnable_lookahead_row_by_gid.find(gid);
                                if (it == runnable_lookahead_row_by_gid.end())
                                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead row runnable map is missing a gid.");
                                if (it->second.size() >= it->second.capacity())
                                    ABORT("GPU3DV2_PCFRAG_TASKFLOW lookahead row runnable queue capacity is undersized.");
                                it->second.push_back(tid);
                            }
                        }
                    }
                    task_enqueued[pos] = 1;
                }
            }
        }

        void reset()
        {
            k = -1;
            stream_offset = -1;
            initialized = 0;
            exchange_posted = 0;
            closed = 0;
            row_pieces.clear();
            partner_pieces.clear();
            tasks.clear();
            task_output_pool.clear();
            row_block_piece.clear();
            partner_block_piece.clear();
            pair_task_entries.clear();
            row_piece_task_offsets.clear();
            row_piece_task_ids.clear();
            partner_piece_task_offsets.clear();
            partner_piece_task_ids.clear();
            task_ready_inputs.clear();
            task_enqueued.clear();
            runnable_task_ids.clear();
            for (int mask = 0; mask < 16; ++mask)
                runnable_task_ids_by_mode[mask].clear();
            runnable_lookahead_col_by_gid.clear();
            runnable_lookahead_row_by_gid.clear();
            incomplete_lookahead_col_members_by_gid.clear();
            incomplete_lookahead_row_members_by_gid.clear();
            launched_lookahead_col_members_by_gid.clear();
            launched_lookahead_row_members_by_gid.clear();
            for (int i = 0; i < SYM_V2_PCFRAG_TASK_STREAM_COUNT; ++i)
            {
                launched_lookahead_col_members_by_gid_by_stream[i].clear();
                launched_lookahead_row_members_by_gid_by_stream[i].clear();
                launched_task_ids_by_stream[i].clear();
                launched_task_groups_by_stream[i].clear();
                launched_task_pending_by_stream[i] = 0;
                for (int mask = 0; mask < 16; ++mask)
                    launched_task_pending_mode_by_stream[i][mask] = 0;
            }
            launched_group_task_ids.clear();
            active_output_key_set.clear();
            active_output_lock_count = 0;
            for (int i = 0; i < SYM_V2_PCFRAG_TASK_STREAM_COUNT; ++i)
            {
                task_event_poll_skip[i] = 0;
                task_event_poll_backoff[i] = 0;
            }
            incomplete_task_count = 0;
            producer_tasks_launched = 0;
            producer_launch_cap_reported = 0;
            producer_exchange_active = 0;
            producer_exchange_pending = 0;
            producer_stream_pending = 0;
            output_conflicts_possible = 0;
            group_scratch_in_use = 0;
            row_pieces_ready_count = 0;
            partner_pieces_ready_count = 0;
            producer_partner_recv_remaining = 0;
            producer_row_recv_remaining = 0;
            producer_ksupc = 0;
            producer_partner_recv_reqs.clear();
            producer_partner_recv_prs.clear();
            producer_partner_recv_sizes.clear();
            producer_partner_recv_offsets.clear();
            producer_partner_recv_done.clear();
            producer_partner_progressive_assembled.clear();
            producer_row_recv_reqs.clear();
            producer_row_recv_pcs.clear();
            producer_row_recv_sizes.clear();
            producer_row_recv_offsets.clear();
            producer_row_recv_done.clear();
            producer_send_reqs.clear();
            // Async-core progress scratch is setup-sized and intentionally
            // kept across panel resets so MPI_Testsome/Waitsome never grows
            // temporary vectors during factorization.
#ifdef HAVE_CUDA
            producer_partner_recv_host_values = NULL;
            producer_row_recv_host_values = NULL;
            producer_partner_send_host_values = NULL;
            producer_row_send_host_values = NULL;
            producer_last_ready_event = NULL;
            d_index_pool = NULL;
            d_value_pool = NULL;
            d_group_index_pool = NULL;
            d_group_value_pool = NULL;
#endif
            producer_partner_recv_host_capacity = 0;
            producer_row_recv_host_capacity = 0;
            producer_partner_send_host_capacity = 0;
            producer_row_send_host_capacity = 0;
            index_pool_capacity = 0;
            value_pool_capacity = 0;
            group_index_pool_capacity = 0;
            group_value_pool_capacity = 0;
            index_pool_used = 0;
            value_pool_used = 0;
        }
    };

    struct SymV2PcFragGemmResourceState
    {
#ifdef HAVE_CUDA
        cudaEvent_t tail_event;
#endif
        unsigned char recorded;
        int owner_stream_id;
        int resource_kind;
        int active_task_id;
        long long waits;
        long long updates;

        SymV2PcFragGemmResourceState()
#ifdef HAVE_CUDA
            : tail_event(NULL),
#else
            :
#endif
              recorded(0), owner_stream_id(-1),
              resource_kind(SYM_V2_PCFRAG_TASK_GEMM_RESOURCE_NONE),
              active_task_id(-1), waits(0), updates(0)
        {
        }
    };

    struct SymV2PcFragTaskflowStats
    {
        long long row_pieces_created;
        long long partner_pieces_created;
        long long row_pieces_ready;
        long long partner_pieces_ready;
        long long tasks_planned;
        long long tasks_launched;
        long long tasks_completed;
        long long tasks_completed_async_core;
        long long task_tiled_block_pairs;
        long long task_tiled_gemm_tiles;
        long long task_completion_event_queries;
        long long task_completion_event_query_skips;
        long long task_completion_event_waits;
        long long task_completion_poll_calls;
        long long task_completion_poll_task_scans;
        long long task_completion_poll_required_seen;
        long long task_completion_drain_poll_calls;
        long long task_completion_drain_task_scans;
        long long task_completion_drain_required_seen;
        long long tasks_blocked_row;
        long long tasks_blocked_partner;
        long long tasks_blocked_output;
        long long scatter_conflict_waits;
        long long output_locks_acquired;
        long long output_lock_high_water;
        long long output_locks_released_by_event;
        long long output_locks_released_by_launch_sync;
        long long tasks_launched_progress;
        long long tasks_launched_eager_full;
        long long tasks_launched_lookahead;
        long long tasks_launched_exclude;
        long long tasks_launched_full;
        long long dispatch_calls_lookahead;
        long long dispatch_calls_exclude;
        long long dispatch_calls_full;
        long long drain_calls_lookahead;
        long long drain_calls_exclude;
        long long drain_calls_full;
        long long drain_incomplete_tasks;
        long long taskflow_entries;
        long long legacy_wrapper_aborts;
        long long early_task_launches_before_full_panel_ready;
        long long arena_value_high_water;
        long long arena_index_high_water;
        long long arena_pinned_high_water;
        long long arena_index_prewarm_blocks;
        long long arena_value_prewarm_blocks;
        long long arena_pinned_prewarm_blocks;
        long long arena_event_prewarm_blocks;
        long long arena_index_late_allocs;
        long long arena_value_late_allocs;
        long long arena_pinned_late_allocs;
        long long arena_event_late_allocs;
        long long producer_recv_wait_calls;
        long long producer_send_wait_calls;
        long long producer_send_boundary_wait_calls;
        long long producer_send_nonboundary_wait_calls;
        long long producer_mpi_wait_requests;
        long long producer_returns;
        long long producer_returns_all_pieces_ready;
        long long producer_returns_incomplete_pieces;
        long long producer_return_unready_pieces;
        long long producer_returns_all_tasks_complete;
        long long producer_returns_incomplete_tasks;
        long long producer_return_incomplete_task_sum;
        long long producer_task_launch_cap_hits;
        long long producer_task_launch_cap_deferred;
        long long producer_exchange_progress_calls;
        long long producer_exchange_drain_calls;
        long long producer_recv_test_calls;
        long long producer_recv_test_completions;
        long long producer_send_test_calls;
        long long producer_send_test_completions;
        long long producer_returns_with_pending_recvs;
        long long final_progress_rounds;
        long long final_progress_tasks_launched;
        long long final_progress_tasks_completed;
        long long final_predrain_rounds;
        long long final_predrain_dispatch_calls;
        long long final_predrain_tasks_launched;
        long long task_launch_stream_syncs;
        long long gemm_resource_tail_waits;
        long long gemm_resource_tail_updates;
        long long global_output_lock_conflicts;
        long long global_output_locks_acquired;
        long long global_output_locks_released;
        long long global_output_locks_live;
        long long gemm_resource_live_recorded;
        long long producer_exchange_stream_syncs;
        long long producer_recv_pinned_posts;
        long long producer_recv_pageable_posts;
        long long producer_progress_vector_growths;
        long long task_completion_event_successes;
        long long grouped_dispatch_attempts;
        long long grouped_launches;
        long long grouped_task_members;
        long long grouped_candidate_scans;
        long long grouped_single_fallbacks;
        long long grouped_completed_pair_fallbacks;
        long long grouped_output_conflict_fallbacks;
        long long grouped_capacity_fallbacks;
        long long grouped_scratch_busy_deferrals;
        long long grouped_pending_cap_deferrals;

        SymV2PcFragTaskflowStats()
            : row_pieces_created(0), partner_pieces_created(0),
              row_pieces_ready(0), partner_pieces_ready(0),
              tasks_planned(0), tasks_launched(0), tasks_completed(0),
              tasks_completed_async_core(0), task_tiled_block_pairs(0),
              task_tiled_gemm_tiles(0), task_completion_event_queries(0),
              task_completion_event_query_skips(0),
              task_completion_event_waits(0),
              task_completion_poll_calls(0),
              task_completion_poll_task_scans(0),
              task_completion_poll_required_seen(0),
              task_completion_drain_poll_calls(0),
              task_completion_drain_task_scans(0),
              task_completion_drain_required_seen(0),
              tasks_blocked_row(0), tasks_blocked_partner(0),
              tasks_blocked_output(0), scatter_conflict_waits(0),
              output_locks_acquired(0), output_lock_high_water(0),
              output_locks_released_by_event(0),
              output_locks_released_by_launch_sync(0),
              tasks_launched_progress(0), tasks_launched_eager_full(0),
              tasks_launched_lookahead(0), tasks_launched_exclude(0),
              tasks_launched_full(0), dispatch_calls_lookahead(0),
              dispatch_calls_exclude(0), dispatch_calls_full(0),
              drain_calls_lookahead(0), drain_calls_exclude(0),
              drain_calls_full(0), drain_incomplete_tasks(0),
              taskflow_entries(0), legacy_wrapper_aborts(0),
              early_task_launches_before_full_panel_ready(0),
              arena_value_high_water(0), arena_index_high_water(0),
              arena_pinned_high_water(0),
              arena_index_prewarm_blocks(0),
              arena_value_prewarm_blocks(0),
              arena_pinned_prewarm_blocks(0),
              arena_event_prewarm_blocks(0),
              arena_index_late_allocs(0), arena_value_late_allocs(0),
              arena_pinned_late_allocs(0), arena_event_late_allocs(0),
              producer_recv_wait_calls(0),
              producer_send_wait_calls(0),
              producer_send_boundary_wait_calls(0),
              producer_send_nonboundary_wait_calls(0),
              producer_mpi_wait_requests(0),
              producer_returns(0), producer_returns_all_pieces_ready(0),
              producer_returns_incomplete_pieces(0),
              producer_return_unready_pieces(0),
              producer_returns_all_tasks_complete(0),
              producer_returns_incomplete_tasks(0),
              producer_return_incomplete_task_sum(0),
              producer_task_launch_cap_hits(0),
              producer_task_launch_cap_deferred(0),
              producer_exchange_progress_calls(0),
              producer_exchange_drain_calls(0),
              producer_recv_test_calls(0),
              producer_recv_test_completions(0),
              producer_send_test_calls(0),
              producer_send_test_completions(0),
              producer_returns_with_pending_recvs(0),
              final_progress_rounds(0), final_progress_tasks_launched(0),
              final_progress_tasks_completed(0),
              final_predrain_rounds(0), final_predrain_dispatch_calls(0),
              final_predrain_tasks_launched(0),
              task_launch_stream_syncs(0),
              gemm_resource_tail_waits(0),
              gemm_resource_tail_updates(0),
              global_output_lock_conflicts(0),
              global_output_locks_acquired(0),
              global_output_locks_released(0),
              global_output_locks_live(0),
              gemm_resource_live_recorded(0),
              producer_exchange_stream_syncs(0),
              producer_recv_pinned_posts(0),
              producer_recv_pageable_posts(0),
              producer_progress_vector_growths(0),
              task_completion_event_successes(0),
              grouped_dispatch_attempts(0),
              grouped_launches(0), grouped_task_members(0),
              grouped_candidate_scans(0), grouped_single_fallbacks(0),
              grouped_completed_pair_fallbacks(0),
              grouped_output_conflict_fallbacks(0),
              grouped_capacity_fallbacks(0),
              grouped_scratch_busy_deferrals(0),
              grouped_pending_cap_deferrals(0)
        {
        }
    };

    enum SymV2PcFragTaskflowProfileIndex
    {
        SYM_V2_PCFRAG_TASKFLOW_ROW_PIECES_CREATED = 0,
        SYM_V2_PCFRAG_TASKFLOW_PARTNER_PIECES_CREATED,
        SYM_V2_PCFRAG_TASKFLOW_ROW_PIECES_READY,
        SYM_V2_PCFRAG_TASKFLOW_PARTNER_PIECES_READY,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_PLANNED,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_LAUNCHED,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_COMPLETED,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_COMPLETED_ASYNC_CORE,
        SYM_V2_PCFRAG_TASKFLOW_TASK_TILED_BLOCK_PAIRS,
        SYM_V2_PCFRAG_TASKFLOW_TASK_TILED_GEMM_TILES,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_EVENT_QUERIES,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_EVENT_QUERY_SKIPS,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_EVENT_WAITS,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_POLL_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_POLL_TASK_SCANS,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_POLL_REQUIRED_SEEN,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_DRAIN_POLL_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_DRAIN_TASK_SCANS,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_DRAIN_REQUIRED_SEEN,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_BLOCKED_ROW,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_BLOCKED_PARTNER,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_BLOCKED_OUTPUT,
        SYM_V2_PCFRAG_TASKFLOW_SCATTER_CONFLICT_WAITS,
        SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_ACQUIRED,
        SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCK_HIGH_WATER,
        SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_RELEASED_BY_EVENT,
        SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_RELEASED_BY_LAUNCH_SYNC,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_LAUNCHED_PROGRESS,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_LAUNCHED_EAGER_FULL,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_LAUNCHED_LOOKAHEAD,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_LAUNCHED_EXCLUDE,
        SYM_V2_PCFRAG_TASKFLOW_TASKS_LAUNCHED_FULL,
        SYM_V2_PCFRAG_TASKFLOW_DISPATCH_CALLS_LOOKAHEAD,
        SYM_V2_PCFRAG_TASKFLOW_DISPATCH_CALLS_EXCLUDE,
        SYM_V2_PCFRAG_TASKFLOW_DISPATCH_CALLS_FULL,
        SYM_V2_PCFRAG_TASKFLOW_DRAIN_CALLS_LOOKAHEAD,
        SYM_V2_PCFRAG_TASKFLOW_DRAIN_CALLS_EXCLUDE,
        SYM_V2_PCFRAG_TASKFLOW_DRAIN_CALLS_FULL,
        SYM_V2_PCFRAG_TASKFLOW_DRAIN_INCOMPLETE_TASKS,
        SYM_V2_PCFRAG_TASKFLOW_TASKFLOW_ENTRIES,
        SYM_V2_PCFRAG_TASKFLOW_LEGACY_WRAPPER_ABORTS,
        SYM_V2_PCFRAG_TASKFLOW_EARLY_TASK_LAUNCHES_BEFORE_FULL_PANEL_READY,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_VALUE_HIGH_WATER,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_INDEX_HIGH_WATER,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_PINNED_HIGH_WATER,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_INDEX_PREWARM_BLOCKS,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_VALUE_PREWARM_BLOCKS,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_PINNED_PREWARM_BLOCKS,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_EVENT_PREWARM_BLOCKS,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_INDEX_LATE_ALLOCS,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_VALUE_LATE_ALLOCS,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_PINNED_LATE_ALLOCS,
        SYM_V2_PCFRAG_TASKFLOW_ARENA_EVENT_LATE_ALLOCS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_WAIT_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_WAIT_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_BOUNDARY_WAIT_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_NONBOUNDARY_WAIT_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_MPI_WAIT_REQUESTS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RETURNS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RETURNS_ALL_PIECES_READY,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RETURNS_INCOMPLETE_PIECES,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RETURN_UNREADY_PIECES,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RETURNS_ALL_TASKS_COMPLETE,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RETURNS_INCOMPLETE_TASKS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RETURN_INCOMPLETE_TASK_SUM,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_TASK_LAUNCH_CAP_HITS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_TASK_LAUNCH_CAP_DEFERRED,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_EXCHANGE_PROGRESS_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_EXCHANGE_DRAIN_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_TEST_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_TEST_COMPLETIONS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_TEST_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_TEST_COMPLETIONS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RETURNS_WITH_PENDING_RECVS,
        SYM_V2_PCFRAG_TASKFLOW_FINAL_PROGRESS_ROUNDS,
        SYM_V2_PCFRAG_TASKFLOW_FINAL_PROGRESS_TASKS_LAUNCHED,
        SYM_V2_PCFRAG_TASKFLOW_FINAL_PROGRESS_TASKS_COMPLETED,
        SYM_V2_PCFRAG_TASKFLOW_FINAL_PREDRAIN_ROUNDS,
        SYM_V2_PCFRAG_TASKFLOW_FINAL_PREDRAIN_DISPATCH_CALLS,
        SYM_V2_PCFRAG_TASKFLOW_FINAL_PREDRAIN_TASKS_LAUNCHED,
        SYM_V2_PCFRAG_TASKFLOW_TASK_LAUNCH_STREAM_SYNCS,
        SYM_V2_PCFRAG_TASKFLOW_GEMM_RESOURCE_TAIL_WAITS,
        SYM_V2_PCFRAG_TASKFLOW_GEMM_RESOURCE_TAIL_UPDATES,
        SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCK_CONFLICTS,
        SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCKS_ACQUIRED,
        SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCKS_RELEASED,
        SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCKS_LIVE,
        SYM_V2_PCFRAG_TASKFLOW_GEMM_RESOURCE_LIVE_RECORDED,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_EXCHANGE_STREAM_SYNCS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_PINNED_POSTS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_PAGEABLE_POSTS,
        SYM_V2_PCFRAG_TASKFLOW_PRODUCER_PROGRESS_VECTOR_GROWTHS,
        SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_EVENT_SUCCESSES,
        SYM_V2_PCFRAG_TASKFLOW_PROFILE_COUNT
    };

#ifdef HAVE_CUDA
    struct SymV2PcFragGpuIndexBlock
    {
        int_t *ptr;
        size_t capacity;

        SymV2PcFragGpuIndexBlock() : ptr(NULL), capacity(0) {}
        SymV2PcFragGpuIndexBlock(int_t *ptr_, size_t capacity_)
            : ptr(ptr_), capacity(capacity_) {}
    };

    struct SymV2PcFragGpuValueBlock
    {
        Ftype *ptr;
        size_t capacity;

        SymV2PcFragGpuValueBlock() : ptr(NULL), capacity(0) {}
        SymV2PcFragGpuValueBlock(Ftype *ptr_, size_t capacity_)
            : ptr(ptr_), capacity(capacity_) {}
    };

    struct SymV2PcFragHostValueBlock
    {
        Ftype *ptr;
        size_t capacity;

        SymV2PcFragHostValueBlock() : ptr(NULL), capacity(0) {}
        SymV2PcFragHostValueBlock(Ftype *ptr_, size_t capacity_)
            : ptr(ptr_), capacity(capacity_) {}
    };
#endif

    std::vector<SymV2PcFragPanelTaskState> symV2PcFragTaskStates;
    std::set<SymV2PcFragOutputKey> symV2PcFragTaskflowGlobalOutputLocks;
    std::vector<int_t> symV2PcFragTaskflowOutputPanelOffsets;
    std::vector<unsigned char> symV2PcFragTaskflowGlobalOutputLockState;
    long long symV2PcFragTaskflowGlobalOutputLocksLive = 0;
#ifdef HAVE_CUDA
    std::vector<SymV2PcFragGemmResourceState>
        symV2PcFragTaskflowGemmResources;
    std::vector<cudaEvent_t> symV2PcFragTaskflowEventPool;
    std::vector<SymV2PcFragGpuIndexBlock>
        symV2PcFragTaskflowIndexBlockPool;
    std::vector<SymV2PcFragGpuValueBlock>
        symV2PcFragTaskflowValueBlockPool;
    std::vector<SymV2PcFragHostValueBlock>
        symV2PcFragTaskflowPinnedBlockPool;
#endif
    SymV2PcFragTaskflowStats symV2PcFragTaskflowStats;

    void symV2PcFragTaskflowPrintProfile()
    {
        if (!superlu_sym_v2_pcfrag_taskflow())
            return;
        symV2PcFragTaskflowStats.global_output_locks_live =
            symV2PcFragTaskflowGlobalOutputLocksLive +
            static_cast<long long>(
                symV2PcFragTaskflowGlobalOutputLocks.size());
        long long live_recorded = 0;
#ifdef HAVE_CUDA
        for (size_t i = 0; i < symV2PcFragTaskflowGemmResources.size(); ++i)
            if (symV2PcFragTaskflowGemmResources[i].recorded)
                ++live_recorded;
#endif
        symV2PcFragTaskflowStats.gemm_resource_live_recorded =
            live_recorded;
        long long local[SYM_V2_PCFRAG_TASKFLOW_PROFILE_COUNT] = {
            symV2PcFragTaskflowStats.row_pieces_created,
            symV2PcFragTaskflowStats.partner_pieces_created,
            symV2PcFragTaskflowStats.row_pieces_ready,
            symV2PcFragTaskflowStats.partner_pieces_ready,
            symV2PcFragTaskflowStats.tasks_planned,
            symV2PcFragTaskflowStats.tasks_launched,
            symV2PcFragTaskflowStats.tasks_completed,
            symV2PcFragTaskflowStats.tasks_completed_async_core,
            symV2PcFragTaskflowStats.task_tiled_block_pairs,
            symV2PcFragTaskflowStats.task_tiled_gemm_tiles,
            symV2PcFragTaskflowStats.task_completion_event_queries,
            symV2PcFragTaskflowStats.task_completion_event_query_skips,
            symV2PcFragTaskflowStats.task_completion_event_waits,
            symV2PcFragTaskflowStats.task_completion_poll_calls,
            symV2PcFragTaskflowStats.task_completion_poll_task_scans,
            symV2PcFragTaskflowStats.task_completion_poll_required_seen,
            symV2PcFragTaskflowStats.task_completion_drain_poll_calls,
            symV2PcFragTaskflowStats.task_completion_drain_task_scans,
            symV2PcFragTaskflowStats.task_completion_drain_required_seen,
            symV2PcFragTaskflowStats.tasks_blocked_row,
            symV2PcFragTaskflowStats.tasks_blocked_partner,
            symV2PcFragTaskflowStats.tasks_blocked_output,
            symV2PcFragTaskflowStats.scatter_conflict_waits,
            symV2PcFragTaskflowStats.output_locks_acquired,
            symV2PcFragTaskflowStats.output_lock_high_water,
            symV2PcFragTaskflowStats.output_locks_released_by_event,
            symV2PcFragTaskflowStats.output_locks_released_by_launch_sync,
            symV2PcFragTaskflowStats.tasks_launched_progress,
            symV2PcFragTaskflowStats.tasks_launched_eager_full,
            symV2PcFragTaskflowStats.tasks_launched_lookahead,
            symV2PcFragTaskflowStats.tasks_launched_exclude,
            symV2PcFragTaskflowStats.tasks_launched_full,
            symV2PcFragTaskflowStats.dispatch_calls_lookahead,
            symV2PcFragTaskflowStats.dispatch_calls_exclude,
            symV2PcFragTaskflowStats.dispatch_calls_full,
            symV2PcFragTaskflowStats.drain_calls_lookahead,
            symV2PcFragTaskflowStats.drain_calls_exclude,
            symV2PcFragTaskflowStats.drain_calls_full,
            symV2PcFragTaskflowStats.drain_incomplete_tasks,
            symV2PcFragTaskflowStats.taskflow_entries,
            symV2PcFragTaskflowStats.legacy_wrapper_aborts,
            symV2PcFragTaskflowStats.early_task_launches_before_full_panel_ready,
            symV2PcFragTaskflowStats.arena_value_high_water,
            symV2PcFragTaskflowStats.arena_index_high_water,
            symV2PcFragTaskflowStats.arena_pinned_high_water,
            symV2PcFragTaskflowStats.arena_index_prewarm_blocks,
            symV2PcFragTaskflowStats.arena_value_prewarm_blocks,
            symV2PcFragTaskflowStats.arena_pinned_prewarm_blocks,
            symV2PcFragTaskflowStats.arena_event_prewarm_blocks,
            symV2PcFragTaskflowStats.arena_index_late_allocs,
            symV2PcFragTaskflowStats.arena_value_late_allocs,
            symV2PcFragTaskflowStats.arena_pinned_late_allocs,
            symV2PcFragTaskflowStats.arena_event_late_allocs,
            symV2PcFragTaskflowStats.producer_recv_wait_calls,
            symV2PcFragTaskflowStats.producer_send_wait_calls,
            symV2PcFragTaskflowStats.producer_send_boundary_wait_calls,
            symV2PcFragTaskflowStats.producer_send_nonboundary_wait_calls,
            symV2PcFragTaskflowStats.producer_mpi_wait_requests,
            symV2PcFragTaskflowStats.producer_returns,
            symV2PcFragTaskflowStats.producer_returns_all_pieces_ready,
            symV2PcFragTaskflowStats.producer_returns_incomplete_pieces,
            symV2PcFragTaskflowStats.producer_return_unready_pieces,
            symV2PcFragTaskflowStats.producer_returns_all_tasks_complete,
            symV2PcFragTaskflowStats.producer_returns_incomplete_tasks,
            symV2PcFragTaskflowStats.producer_return_incomplete_task_sum,
            symV2PcFragTaskflowStats.producer_task_launch_cap_hits,
            symV2PcFragTaskflowStats.producer_task_launch_cap_deferred,
            symV2PcFragTaskflowStats.producer_exchange_progress_calls,
            symV2PcFragTaskflowStats.producer_exchange_drain_calls,
            symV2PcFragTaskflowStats.producer_recv_test_calls,
            symV2PcFragTaskflowStats.producer_recv_test_completions,
            symV2PcFragTaskflowStats.producer_send_test_calls,
            symV2PcFragTaskflowStats.producer_send_test_completions,
            symV2PcFragTaskflowStats.producer_returns_with_pending_recvs,
            symV2PcFragTaskflowStats.final_progress_rounds,
            symV2PcFragTaskflowStats.final_progress_tasks_launched,
            symV2PcFragTaskflowStats.final_progress_tasks_completed,
            symV2PcFragTaskflowStats.final_predrain_rounds,
            symV2PcFragTaskflowStats.final_predrain_dispatch_calls,
            symV2PcFragTaskflowStats.final_predrain_tasks_launched,
            symV2PcFragTaskflowStats.task_launch_stream_syncs,
            symV2PcFragTaskflowStats.gemm_resource_tail_waits,
            symV2PcFragTaskflowStats.gemm_resource_tail_updates,
            symV2PcFragTaskflowStats.global_output_lock_conflicts,
            symV2PcFragTaskflowStats.global_output_locks_acquired,
            symV2PcFragTaskflowStats.global_output_locks_released,
            symV2PcFragTaskflowStats.global_output_locks_live,
            symV2PcFragTaskflowStats.gemm_resource_live_recorded,
            symV2PcFragTaskflowStats.producer_exchange_stream_syncs,
            symV2PcFragTaskflowStats.producer_recv_pinned_posts,
            symV2PcFragTaskflowStats.producer_recv_pageable_posts,
            symV2PcFragTaskflowStats.producer_progress_vector_growths,
            symV2PcFragTaskflowStats.task_completion_event_successes
        };
        long long global[SYM_V2_PCFRAG_TASKFLOW_PROFILE_COUNT] = {};
        long long local_group[10] = {
            symV2PcFragTaskflowStats.grouped_dispatch_attempts,
            symV2PcFragTaskflowStats.grouped_launches,
            symV2PcFragTaskflowStats.grouped_task_members,
            symV2PcFragTaskflowStats.grouped_candidate_scans,
            symV2PcFragTaskflowStats.grouped_single_fallbacks,
            symV2PcFragTaskflowStats.grouped_completed_pair_fallbacks,
            symV2PcFragTaskflowStats.grouped_output_conflict_fallbacks,
            symV2PcFragTaskflowStats.grouped_capacity_fallbacks,
            symV2PcFragTaskflowStats.grouped_scratch_busy_deferrals,
            symV2PcFragTaskflowStats.grouped_pending_cap_deferrals
        };
        long long global_group[10] = {};
        if (grid3d != NULL)
        {
            MPI_Reduce(local, global,
                       SYM_V2_PCFRAG_TASKFLOW_PROFILE_COUNT,
                       MPI_LONG_LONG, MPI_SUM, 0, grid3d->comm);
            MPI_Reduce(local_group, global_group, 10, MPI_LONG_LONG,
                       MPI_SUM, 0, grid3d->comm);
            if (grid3d->iam != 0)
                return;
        }
        else
        {
            for (int i = 0;
                 i < SYM_V2_PCFRAG_TASKFLOW_PROFILE_COUNT; ++i)
                global[i] = local[i];
            for (int i = 0; i < 10; ++i)
                global_group[i] = local_group[i];
        }
        std::printf(
            "SymFact V2 Pc-fragment taskflow profile: "
            "row_pieces_created=%lld partner_pieces_created=%lld "
            "row_pieces_ready=%lld partner_pieces_ready=%lld "
            "tasks_planned=%lld tasks_launched=%lld tasks_completed=%lld "
            "tasks_completed_async_core=%lld "
            "task_tiled_block_pairs=%lld "
            "task_tiled_gemm_tiles=%lld "
            "task_completion_event_queries=%lld "
            "task_completion_event_query_skips=%lld "
            "task_completion_event_waits=%lld "
            "task_completion_poll_calls=%lld "
            "task_completion_poll_task_scans=%lld "
            "task_completion_poll_required_seen=%lld "
            "task_completion_drain_poll_calls=%lld "
            "task_completion_drain_task_scans=%lld "
            "task_completion_drain_required_seen=%lld "
            "tasks_blocked_row=%lld tasks_blocked_partner=%lld "
            "tasks_blocked_output=%lld scatter_conflict_waits=%lld "
            "output_locks_acquired=%lld output_lock_high_water=%lld "
            "output_locks_released_by_event=%lld "
            "output_locks_released_by_launch_sync=%lld "
            "tasks_launched_progress=%lld tasks_launched_eager_full=%lld "
            "tasks_launched_lookahead=%lld tasks_launched_exclude=%lld "
            "tasks_launched_full=%lld "
            "dispatch_calls_lookahead=%lld dispatch_calls_exclude=%lld "
            "dispatch_calls_full=%lld "
            "drain_calls_lookahead=%lld drain_calls_exclude=%lld "
            "drain_calls_full=%lld drain_incomplete_tasks=%lld "
            "taskflow_entries=%lld legacy_wrapper_aborts=%lld "
            "early_task_launches_before_full_panel_ready=%lld "
            "arena_value_high_water=%lld arena_index_high_water=%lld "
            "arena_pinned_high_water=%lld "
            "arena_index_prewarm_blocks=%lld "
            "arena_value_prewarm_blocks=%lld "
            "arena_pinned_prewarm_blocks=%lld "
            "arena_event_prewarm_blocks=%lld "
            "arena_index_late_allocs=%lld "
            "arena_value_late_allocs=%lld "
            "arena_pinned_late_allocs=%lld "
            "arena_event_late_allocs=%lld "
            "producer_recv_wait_calls=%lld producer_send_wait_calls=%lld "
            "producer_send_boundary_wait_calls=%lld "
            "producer_send_nonboundary_wait_calls=%lld "
            "producer_mpi_wait_requests=%lld "
            "producer_returns=%lld "
            "producer_returns_all_pieces_ready=%lld "
            "producer_returns_incomplete_pieces=%lld "
            "producer_return_unready_pieces=%lld "
            "producer_returns_all_tasks_complete=%lld "
            "producer_returns_incomplete_tasks=%lld "
            "producer_return_incomplete_task_sum=%lld "
            "producer_task_launch_cap_hits=%lld "
            "producer_task_launch_cap_deferred=%lld "
            "producer_exchange_progress_calls=%lld "
            "producer_exchange_drain_calls=%lld "
            "producer_recv_test_calls=%lld "
            "producer_recv_test_completions=%lld "
            "producer_send_test_calls=%lld "
            "producer_send_test_completions=%lld "
            "producer_returns_with_pending_recvs=%lld "
            "final_progress_rounds=%lld "
            "final_progress_tasks_launched=%lld "
            "final_progress_tasks_completed=%lld "
            "final_predrain_rounds=%lld "
            "final_predrain_dispatch_calls=%lld "
            "final_predrain_tasks_launched=%lld "
            "task_launch_stream_syncs=%lld "
            "gemm_resource_tail_waits=%lld "
            "gemm_resource_tail_updates=%lld "
            "global_output_lock_conflicts=%lld "
            "global_output_locks_acquired=%lld "
            "global_output_locks_released=%lld "
            "global_output_locks_live=%lld "
            "gemm_resource_live_recorded=%lld "
            "producer_exchange_stream_syncs=%lld "
            "producer_recv_pinned_posts=%lld "
            "producer_recv_pageable_posts=%lld "
            "producer_progress_vector_growths=%lld "
            "task_completion_event_successes=%lld\n",
            global[0], global[1], global[2], global[3], global[4],
            global[5], global[6], global[7], global[8], global[9],
            global[10], global[11], global[12], global[13], global[14],
            global[15], global[16], global[17], global[18], global[19],
            global[20], global[21], global[22], global[23], global[24],
            global[25], global[26], global[27], global[28], global[29],
            global[30], global[31], global[32], global[33], global[34],
            global[35], global[36], global[37], global[38], global[39],
            global[40], global[41], global[42], global[43], global[44],
            global[45], global[46], global[47], global[48], global[49],
            global[50], global[51], global[52], global[53], global[54],
            global[55], global[56], global[57], global[58], global[59],
            global[60], global[61], global[62], global[63], global[64],
            global[65], global[66], global[67], global[68], global[69],
            global[70], global[71], global[72], global[73], global[74],
            global[75], global[76], global[77], global[78], global[79],
            global[80], global[81], global[82], global[83], global[84],
            global[85], global[86], global[87], global[88], global[89],
            global[90], global[91], global[92]);
        std::printf(
            "SymFact V2 Pc-fragment taskflow grouping: "
            "attempts=%lld launches=%lld task_members=%lld "
            "candidate_scans=%lld single_fallbacks=%lld "
            "completed_pair_fallbacks=%lld "
            "output_conflict_fallbacks=%lld "
            "capacity_fallbacks=%lld "
            "scratch_busy_deferrals=%lld "
            "pending_cap_deferrals=%lld\n",
            global_group[0], global_group[1], global_group[2],
            global_group[3], global_group[4], global_group[5],
            global_group[6], global_group[7], global_group[8],
            global_group[9]);
        if (superlu_sym_v2_pcfrag_taskflow_async_core())
        {
            long long late_allocs =
                global[SYM_V2_PCFRAG_TASKFLOW_ARENA_INDEX_LATE_ALLOCS] +
                global[SYM_V2_PCFRAG_TASKFLOW_ARENA_VALUE_LATE_ALLOCS] +
                global[SYM_V2_PCFRAG_TASKFLOW_ARENA_PINNED_LATE_ALLOCS] +
                global[SYM_V2_PCFRAG_TASKFLOW_ARENA_EVENT_LATE_ALLOCS];
            long long non_async_task_completions =
                global[SYM_V2_PCFRAG_TASKFLOW_TASKS_COMPLETED] -
                global[SYM_V2_PCFRAG_TASKFLOW_TASKS_COMPLETED_ASYNC_CORE];
            if (non_async_task_completions < 0)
                non_async_task_completions = 0;
            long long producer_send_wait_mismatch =
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_WAIT_CALLS] -
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_BOUNDARY_WAIT_CALLS] -
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_NONBOUNDARY_WAIT_CALLS];
            if (producer_send_wait_mismatch < 0)
                producer_send_wait_mismatch = -producer_send_wait_mismatch;
            long long output_lock_release_mismatch =
                global[SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_ACQUIRED] -
                global[SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_RELEASED_BY_EVENT] -
                global[SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_RELEASED_BY_LAUNCH_SYNC];
            if (output_lock_release_mismatch < 0)
                output_lock_release_mismatch = -output_lock_release_mismatch;
            long long global_output_lock_release_mismatch =
                global[SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCKS_ACQUIRED] -
                global[SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCKS_RELEASED];
            if (global_output_lock_release_mismatch < 0)
                global_output_lock_release_mismatch =
                    -global_output_lock_release_mismatch;
            long long gemm_tail_update_mismatch =
                global[SYM_V2_PCFRAG_TASKFLOW_TASKS_LAUNCHED] -
                global[SYM_V2_PCFRAG_TASKFLOW_GEMM_RESOURCE_TAIL_UPDATES];
            if (gemm_tail_update_mismatch < 0)
                gemm_tail_update_mismatch = -gemm_tail_update_mismatch;
            long long task_completion_event_success_mismatch =
                global[SYM_V2_PCFRAG_TASKFLOW_TASKS_COMPLETED_ASYNC_CORE] -
                global[SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_EVENT_SUCCESSES];
            if (task_completion_event_success_mismatch < 0)
                task_completion_event_success_mismatch =
                    -task_completion_event_success_mismatch;
            if (superlu_sym_v2_pcfrag_taskflow_async_grouped_dispatch())
            {
                gemm_tail_update_mismatch = 0;
                task_completion_event_success_mismatch = 0;
            }
            std::printf(
                "SymFact V2 Pc-fragment taskflow async-core contract: "
                "late_allocs=%lld event_waits=%lld "
                "producer_recv_wait_calls=%lld legacy_wrapper_aborts=%lld "
                "non_async_task_completions=%lld "
                "producer_send_wait_calls=%lld "
                "producer_send_boundary_wait_calls=%lld "
                "producer_send_nonboundary_wait_calls=%lld "
                "producer_send_wait_mismatch=%lld "
                "task_launch_stream_syncs=%lld "
                "output_locks_acquired=%lld "
                "output_locks_released_by_event=%lld "
                "output_locks_released_by_launch_sync=%lld "
                "output_lock_release_mismatch=%lld "
                "gemm_resource_tail_waits=%lld "
                "gemm_resource_tail_updates=%lld "
                "gemm_tail_update_mismatch=%lld "
                "global_output_lock_conflicts=%lld "
                "global_output_locks_acquired=%lld "
                "global_output_locks_released=%lld "
                "global_output_lock_release_mismatch=%lld "
                "global_output_locks_live=%lld "
                "gemm_resource_live_recorded=%lld "
                "producer_exchange_stream_syncs=%lld "
                "producer_recv_pinned_posts=%lld "
                "producer_recv_pageable_posts=%lld "
                "producer_progress_vector_growths=%lld "
                "task_completion_event_successes=%lld "
                "task_completion_event_success_mismatch=%lld\n",
                late_allocs,
                global[SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_EVENT_WAITS],
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_WAIT_CALLS],
                global[SYM_V2_PCFRAG_TASKFLOW_LEGACY_WRAPPER_ABORTS],
                non_async_task_completions,
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_WAIT_CALLS],
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_BOUNDARY_WAIT_CALLS],
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_NONBOUNDARY_WAIT_CALLS],
                producer_send_wait_mismatch,
                global[SYM_V2_PCFRAG_TASKFLOW_TASK_LAUNCH_STREAM_SYNCS],
                global[SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_ACQUIRED],
                global[SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_RELEASED_BY_EVENT],
                global[SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_RELEASED_BY_LAUNCH_SYNC],
                output_lock_release_mismatch,
                global[SYM_V2_PCFRAG_TASKFLOW_GEMM_RESOURCE_TAIL_WAITS],
                global[SYM_V2_PCFRAG_TASKFLOW_GEMM_RESOURCE_TAIL_UPDATES],
                gemm_tail_update_mismatch,
                global[SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCK_CONFLICTS],
                global[SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCKS_ACQUIRED],
                global[SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCKS_RELEASED],
                global_output_lock_release_mismatch,
                global[SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCKS_LIVE],
                global[SYM_V2_PCFRAG_TASKFLOW_GEMM_RESOURCE_LIVE_RECORDED],
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_EXCHANGE_STREAM_SYNCS],
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_PINNED_POSTS],
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_PAGEABLE_POSTS],
                global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_PROGRESS_VECTOR_GROWTHS],
                global[SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_EVENT_SUCCESSES],
                task_completion_event_success_mismatch);
            if (superlu_sym_v2_pcfrag_taskflow_async_core_check() &&
                (late_allocs != 0 ||
                 global[SYM_V2_PCFRAG_TASKFLOW_TASK_COMPLETION_EVENT_WAITS] != 0 ||
                 global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_WAIT_CALLS] != 0 ||
                 global[SYM_V2_PCFRAG_TASKFLOW_LEGACY_WRAPPER_ABORTS] != 0 ||
                 non_async_task_completions != 0 ||
                 global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_SEND_NONBOUNDARY_WAIT_CALLS] != 0 ||
                 producer_send_wait_mismatch != 0 ||
                 global[SYM_V2_PCFRAG_TASKFLOW_TASK_LAUNCH_STREAM_SYNCS] != 0 ||
                 global[SYM_V2_PCFRAG_TASKFLOW_OUTPUT_LOCKS_RELEASED_BY_LAUNCH_SYNC] != 0 ||
                 output_lock_release_mismatch != 0 ||
                 gemm_tail_update_mismatch != 0 ||
                 global_output_lock_release_mismatch != 0 ||
                 global[SYM_V2_PCFRAG_TASKFLOW_GLOBAL_OUTPUT_LOCKS_LIVE] != 0 ||
                 global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_RECV_PAGEABLE_POSTS] != 0 ||
                 global[SYM_V2_PCFRAG_TASKFLOW_PRODUCER_PROGRESS_VECTOR_GROWTHS] != 0 ||
                 task_completion_event_success_mismatch != 0))
                ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE_CHECK detected a contract violation.");
        }
        std::fflush(stdout);
    }

    void symV2PcFragTaskflowFinalizeResources()
    {
        if (!superlu_sym_v2_pcfrag_taskflow_async_core())
            return;
        if (!symV2PcFragTaskflowGlobalOutputLocks.empty() ||
            symV2PcFragTaskflowGlobalOutputLocksLive != 0)
            ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE finalization found live global output locks.");
        for (size_t i = 0; i < symV2PcFragTaskflowGemmResources.size(); ++i)
        {
            SymV2PcFragGemmResourceState &res =
                symV2PcFragTaskflowGemmResources[i];
            if (res.active_task_id != -1)
                ABORT("GPU3DV2_PCFRAG_TASKFLOW_ASYNC_CORE finalization found active GEMM resource state.");
            res.recorded = 0;
            res.active_task_id = -1;
        }
        symV2PcFragTaskflowStats.global_output_locks_live = 0;
        symV2PcFragTaskflowStats.gemm_resource_live_recorded = 0;
    }
// SYM_V2_PCFRAG_TASKFLOW_STATE_END

    void *symV2LPanelArenaGPU = NULL;
    void *symV2StreamArenaGPU = NULL;
    void *symV2GemmArenaGPU = NULL;
    size_t symV2LPanelArenaBytes = 0;
    size_t symV2StreamArenaBytes = 0;
    size_t symV2GemmArenaBytes = 0;
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
    int_t maxSymPartnerLSendStageCount = 0;
    int_t maxSymV2RowFragStageCount = 0;
    int_t maxSymV2RowFragValRecvCount = 0;
    int_t maxSymV2RowFragIdxRecvCount = 0;
    int_t maxSymV2RowFragValSendCount = 0;
    int_t maxUvalCount = 0;
    int_t maxUidxCount = 0;
    std::vector<Ftype *> diagFactBufs; /* stores diagonal blocks,
                       each one is a normal dense matrix.
                    Sherry: where are they free'd ?? */
    std::vector<Ftype *> LvalRecvBufs;
    std::vector<Ftype *> UvalRecvBufs;
    std::vector<Ftype *> symPartnerLvalRecvBufs;
    std::vector<Ftype *> symV2RowFragHostRecvBufs;
// SYM_V2_PC2_PHASE1_XLU_SEND_BUFS_BEGIN
    std::vector<Ftype *> symV2RowFragHostSendBufs;
// SYM_V2_PC2_PHASE1_XLU_SEND_BUFS_END

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
#ifdef HAVE_CUDA
        const bool pooled_partner_recv =
            symV2PartnerLHostRecvPoolPinned != NULL;
        const bool pooled_row_recv =
            symV2RowFragHostRecvPoolPinned != NULL;
#endif
        for (int i = 0; i < nlook; i++)
        {
            superluFreeIfAllocated(LvalRecvBufs[i]);
            if (include_u_buffers)
                superluFreeIfAllocated(UvalRecvBufs[i]);
#ifdef HAVE_CUDA
            if (pooled_partner_recv)
            {
                /* All lookahead entries alias the one synchronous scratch pool. */
                symPartnerLvalRecvBufs[i] = NULL;
            }
            else if (symV2PartnerLHostRecvPinned &&
                     symPartnerLvalRecvBufs[i] != NULL)
            {
                gpuErrchk(cudaFreeHost(symPartnerLvalRecvBufs[i]));
                symPartnerLvalRecvBufs[i] = NULL;
            }
            else
#endif
            {
                superluFreeIfAllocated(symPartnerLvalRecvBufs[i]);
            }
#ifdef HAVE_CUDA
            if (i < static_cast<int>(symV2RowFragHostRecvBufs.size()))
            {
                if (pooled_row_recv)
                {
                    symV2RowFragHostRecvBufs[i] = NULL;
                }
                else if (symV2RowFragHostRecvPinned &&
                         symV2RowFragHostRecvBufs[i] != NULL)
                {
                    gpuErrchk(cudaFreeHost(symV2RowFragHostRecvBufs[i]));
                    symV2RowFragHostRecvBufs[i] = NULL;
                }
                else
                {
                    superluFreeIfAllocated(symV2RowFragHostRecvBufs[i]);
                }
            }
#else
            if (i < static_cast<int>(symV2RowFragHostRecvBufs.size()))
                superluFreeIfAllocated(symV2RowFragHostRecvBufs[i]);
#endif
            superluFreeIfAllocated(LidxRecvBufs[i]);
            if (include_u_buffers)
                superluFreeIfAllocated(UidxRecvBufs[i]);
            superluFreeIfAllocated(symPartnerLidxRecvBufs[i]);
        }
#ifdef HAVE_CUDA
        if (symV2PartnerLHostRecvPoolPinned != NULL)
        {
            gpuErrchk(cudaFreeHost(symV2PartnerLHostRecvPoolPinned));
            symV2PartnerLHostRecvPoolPinned = NULL;
        }
        symV2PartnerLHostRecvPoolPinnedCount = 0;
        if (symV2RowFragHostRecvPoolPinned != NULL)
        {
            gpuErrchk(cudaFreeHost(symV2RowFragHostRecvPoolPinned));
            symV2RowFragHostRecvPoolPinned = NULL;
        }
        symV2RowFragHostRecvPoolPinnedCount = 0;
#endif
#ifdef HAVE_CUDA
        symV2PartnerLHostRecvPinned = 0;
        symV2RowFragHostRecvPinned = 0;
#endif
        symV2RowFragHostRecvBufs.clear();
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
#ifdef HAVE_CUDA
        if (options != NULL && options->SymFact == YES)
            symV2PcFragTaskflowPrintProfile();
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
                if (A_gpu.symV2RowFragStageBufs[stream] != NULL &&
                    symV2StreamArenaGPU == NULL)
                    cudaFree(A_gpu.symV2RowFragStageBufs[stream]);
                if (A_gpu.symV2RowFragValRecvBufs[stream] != NULL &&
                    symV2StreamArenaGPU == NULL)
                    cudaFree(A_gpu.symV2RowFragValRecvBufs[stream]);
                if (A_gpu.symV2RowFragIdxRecvBufs[stream] != NULL &&
                    symV2StreamArenaGPU == NULL)
                    cudaFree(A_gpu.symV2RowFragIdxRecvBufs[stream]);
                if (A_gpu.symV2RowFragSendMapStageBufs[stream] != NULL &&
                    symV2StreamArenaGPU == NULL)
                    cudaFree(A_gpu.symV2RowFragSendMapStageBufs[stream]);
                if (A_gpu.symV2RawPanelReadyEvents[stream] != NULL)
                    cudaEventDestroy(A_gpu.symV2RawPanelReadyEvents[stream]);
                if (A_gpu.symV2RawPanelBufs[stream] != NULL &&
                    symV2StreamArenaGPU == NULL)
                    cudaFree(A_gpu.symV2RawPanelBufs[stream]);
                cudaEventDestroy(A_gpu.symV2PartnerLPackReadyEvents[stream]);
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
            if (symV2StreamArenaGPU != NULL)
            {
                cudaFree(symV2StreamArenaGPU);
                symV2StreamArenaGPU = NULL;
                symV2StreamArenaBytes = 0;
            }
            if (symV2GemmArenaGPU != NULL)
            {
                cudaFree(symV2GemmArenaGPU);
                symV2GemmArenaGPU = NULL;
                symV2GemmArenaBytes = 0;
            }
            if (symV2LPanelArenaGPU != NULL)
            {
                cudaFree(symV2LPanelArenaGPU);
                symV2LPanelArenaGPU = NULL;
                symV2LPanelArenaBytes = 0;
            }
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
    size_t symV2DelayedGpuMetadataBytes() const;
    int materializeSymFactGpuMetadata();
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
    int_t dSymSchurCompUpdatePartDualFragmentsGPU(
        int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
        int_t k,
        const std::vector<int_t> &row_frag,
        const std::vector<int_t> &col_frag,
        int_t *row_frag_index, Ftype *row_frag_val,
        int_t *col_frag_index, Ftype *col_frag_val,
        cublasHandle_t handle, cudaStream_t cuStream,
        Ftype *gemmBuff);
    int_t dSymSchurCompUpdateTaskDualPieceGroupGPU(
        int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
        int_t k,
        const std::vector<int_t> &row_frag,
        const std::vector<int_t> &col_frag,
        int_t *row_frag_index, Ftype *row_frag_val,
        int_t *col_frag_index, Ftype *col_frag_val,
        cublasHandle_t handle, cudaStream_t cuStream,
        Ftype *gemmBuff);
    int_t dSymSchurCompUpdateTaskDualPiecesGPU(
        int_t k,
        const std::vector<int_t> &row_piece,
        const std::vector<int_t> &col_piece,
        int_t *row_piece_index, Ftype *row_piece_val,
        int_t *col_piece_index, Ftype *col_piece_val,
        cublasHandle_t handle, cudaStream_t cuStream,
        Ftype *gemmBuff);
    int_t dSymSchurCompUpLimitedMemDualFragmentsGPU(
        int_t rowStart, int_t rowEnd,
        int_t colStart, int_t colEnd,
        int_t k,
        const std::vector<int_t> &row_frag,
        const std::vector<int_t> &col_frag,
        int_t *row_frag_index, Ftype *row_frag_val,
        int_t *col_frag_index, Ftype *col_frag_val,
        cublasHandle_t handle, cudaStream_t cuStream,
        Ftype *gemmBuff);
    int_t dSymLookAheadUpdateDualFragmentsGPU(
        int streamId, int_t k, int_t laIdx);
    int_t dSymSchurCompUpdateExcludeOneDualFragmentsGPU(
        int streamId, int_t k, int_t ex);
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
    int_t dSymV2PrepackLFragmentsGPU(int_t k, int_t stream_offset);
    int_t dSymV2LFragmentExchangeGPU(int_t k, int_t stream_offset);
    bool symV2UsePcFragmentSchurPanel(int_t k) const;
    bool symV2UsePcFragmentTaskflowPanel(int_t k) const;
    int_t dSymV2PcFragTaskflowBeginGPU(int_t k, int_t stream_offset);
    int_t dSymV2PcFragTaskflowAssembleOwnedPiecesGPU(
        int_t k, unsigned char kind, const Ftype *stage,
        const std::vector<int_t> &recv_map, int_t ksupc,
        cudaStream_t stream);
    int_t dSymV2PcFragTaskflowProgressExchangeGPU(int_t k, int drain);
    int_t dSymV2PcFragTaskflowProgressGPU(int_t k, int budget);
    int_t dSymV2PcFragTaskflowDispatchGPU(
        int streamId, int_t k, unsigned mode_mask, int_t mode_gid, int drain);
    int_t dSymV2PcFragTaskflowDrainGPU(
        int_t k, unsigned mode_mask, int_t mode_gid);
    int_t dSymV2PcFragTaskflowReleaseGPU(int_t k);

    int_t ancestorReduction3dGPU(int_t ilvl, int_t *myNodeCount,
                                 int_t **treePerm);
    int_t dSymV2BatchAncestorReduceGPU(int_t numNodes, int_t *nodeList,
                                       int_t sender, int_t receiver,
                                       int_t reduction_level);
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
