#include "mpi.h"
// #include "cublasDefs.hhandle, "
#include <limits>
#include "lupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

static inline int symldl_v2_reduction_mpi_count(int_t count)
{
    if (count < 0 ||
        count > static_cast<int_t>(std::numeric_limits<int>::max()))
        ABORT("Panel reduction count exceeds MPI limit.");
    return static_cast<int>(count);
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::ancestorReduction3dGPU(int_t ilvl, int_t *myNodeCount,
                                         int_t **treePerm)
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int_t myGrid = grid3d->zscp.Iam;

#if (DEBUGlevel >= 1)
    printf(".maxLvl %d\n", maxLvl); fflush(stdout);
    CHECK_MALLOC(grid3d->iam, "Enter ancestorReduction3dGPU()");
#endif
	
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

    /*Reduce all the ancestors*/
    for (int_t alvl = ilvl + 1; alvl < maxLvl; ++alvl)
    {
        /* code */
        // int_t atree = myTreeIdxs[alvl];
        int_t numNodes = myNodeCount[alvl];
        int_t *nodeList = treePerm[alvl];
        double treduce = SuperLU_timer_();
        

        /*first setting the L blocks to zero*/
        for (int_t node = 0; node < numNodes; ++node) /* for each block column ... */
        {
            int_t k0 = nodeList[node];

            if (myGrid == sender)
            {
                zSendLPanelGPU(k0, receiver);
                zSendUPanelGPU(k0, receiver);
            }
            else
            {
                Ftype alpha = one<Ftype>(); Ftype beta = one<Ftype>(); 

                zRecvLPanelGPU(k0, sender, alpha, beta);
                zRecvUPanelGPU(k0, sender, alpha, beta);
            }
        }
        cudaStreamSynchronize(A_gpu.cuStreams[0]) ;
        // return 0;
        SCT->ancsReduce += SuperLU_timer_() - treduce;
    }
    
#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Exit ancestorReduction3dGPU()");
#endif
    
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zSendLPanelGPU(int_t k0, int_t receiverGrid)
{
    const int_t panel_root = useSymV2Solve() ? symV2PanelRoot(k0)
                                             : kcol(k0);
	if (mycol == panel_root)
	{
		int_t lk = useSymV2Solve() ? symV2PanelIndex(k0)
                                   : g2lCol(k0);
        int_t panel_count = useSymV2Solve() ? symV2PanelCount()
                                            : CEILING(nsupers, Pc);
        if (lk >= 0 && lk < panel_count &&
            !lPanelVec[lk].isEmpty())
		{
            int mpi_count =
                symldl_v2_reduction_mpi_count(lPanelVec[lk].nzvalSize());
            superlu_gpu_mpi_send(lPanelVec[lk].blkPtrGPU(0),
                                 lPanelVec[lk].blkPtr(0), sizeof(Ftype),
                                 mpi_count, get_mpi_type<Ftype>(),
                                 receiverGrid, k0, grid3d->zscp.comm);
			SCT->commVolRed += lPanelVec[lk].nzvalSize() * sizeof(Ftype);
		}
	}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zRecvLPanelGPU(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta)
{
    const int_t panel_root = useSymV2Solve() ? symV2PanelRoot(k0)
                                             : kcol(k0);
    if (mycol == panel_root)
	{
		int_t lk = useSymV2Solve() ? symV2PanelIndex(k0)
                                   : g2lCol(k0);
        int_t panel_count = useSymV2Solve() ? symV2PanelCount()
                                            : CEILING(nsupers, Pc);
        if (lk >= 0 && lk < panel_count &&
            !lPanelVec[lk].isEmpty())
		{
            
            MPI_Status status;
            int mpi_count =
                symldl_v2_reduction_mpi_count(lPanelVec[lk].nzvalSize());
            superlu_gpu_mpi_recv(A_gpu.LvalRecvBufs[0], LvalRecvBufs[0],
                                 sizeof(Ftype), mpi_count,
                                 get_mpi_type<Ftype>(), senderGrid, k0,
                                 grid3d->zscp.comm, &status);

			/*reduce the updates*/
            cublasHandle_t handle=A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            cublasSetStream(handle, cuStream);
            myCublasScal<Ftype>(handle, lPanelVec[lk].nzvalSize(), &alpha, lPanelVec[lk].blkPtrGPU(0), 1);
            myCublasAxpy<Ftype>(handle, lPanelVec[lk].nzvalSize(), &beta, A_gpu.LvalRecvBufs[0], 1, lPanelVec[lk].blkPtrGPU(0), 1);
			cudaStreamSynchronize(cuStream);
            
		}
	}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zSendUPanelGPU(int_t k0, int_t receiverGrid)
{
    if (useSymV2Solve())
        return 0;
    
	if (myrow == krow(k0))
	{
        int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{
            int mpi_count =
                symldl_v2_reduction_mpi_count(uPanelVec[lk].nzvalSize());
            superlu_gpu_mpi_send(uPanelVec[lk].blkPtrGPU(0),
                                 uPanelVec[lk].blkPtr(0), sizeof(Ftype),
                                 mpi_count, get_mpi_type<Ftype>(),
                                 receiverGrid, k0, grid3d->zscp.comm);
			SCT->commVolRed += uPanelVec[lk].nzvalSize() * sizeof(Ftype);
		}
	}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zRecvUPanelGPU(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta)
{
    if (useSymV2Solve())
        return 0;

    if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{

            MPI_Status status;
            int mpi_count =
                symldl_v2_reduction_mpi_count(uPanelVec[lk].nzvalSize());
            superlu_gpu_mpi_recv(A_gpu.UvalRecvBufs[0], UvalRecvBufs[0],
                                 sizeof(Ftype), mpi_count,
                                 get_mpi_type<Ftype>(), senderGrid, k0,
                                 grid3d->zscp.comm, &status);

			/*reduce the updates*/
            cublasHandle_t handle=A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            cublasSetStream(handle, cuStream);
			myCublasScal<Ftype>(handle, uPanelVec[lk].nzvalSize(), &alpha, uPanelVec[lk].blkPtrGPU(0), 1);
			myCublasAxpy<Ftype>(handle, uPanelVec[lk].nzvalSize(), &beta, A_gpu.UvalRecvBufs[0], 1, uPanelVec[lk].blkPtrGPU(0), 1);
            cudaStreamSynchronize(cuStream);
		}
	}
	return 0;
}

#endif
