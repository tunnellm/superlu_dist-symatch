#include "mpi.h"
// #include "cublasDefs.hhandle, "
#include "lupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "gpu_mpi_utils.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
int_t xLUstruct_t<Ftype>::ancestorReduction3dGPU(int_t ilvl, int_t *myNodeCount,
                                         int_t **treePerm)
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int_t myGrid = grid3d->zscp.Iam;
    bool sym_v2_l_only =
        (options != NULL && options->SymFact == YES && symGPU3DVersion == 2);

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
                if (!sym_v2_l_only)
                    zSendUPanelGPU(k0, receiver);
            }
            else
            {
                Ftype alpha = one<Ftype>(); Ftype beta = one<Ftype>(); 

                zRecvLPanelGPU(k0, sender, alpha, beta);
                if (!sym_v2_l_only)
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
    int_t panel_col = useSymV2Solve() ? symV2PanelRoot(k0) : kcol(k0);

	if (mycol == panel_col)
	{
		int_t lk = useSymV2Solve() ? symV2PanelIndex(k0) : g2lCol(k0);
        if (lk < 0)
            return 0;
        if (!lPanelVec[lk].isEmpty())
		{
            superlu_gpu_mpi_send(lPanelVec[lk].blkPtrGPU(0), LvalRecvBufs[0],
                    sizeof(Ftype), static_cast<int>(lPanelVec[lk].nzvalSize()),
                    get_mpi_type<Ftype>(), receiverGrid, k0, grid3d->zscp.comm);
				SCT->commVolRed += lPanelVec[lk].nzvalSize() * sizeof(Ftype);
			}
		}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zRecvLPanelGPU(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta)
{
    int_t panel_col = useSymV2Solve() ? symV2PanelRoot(k0) : kcol(k0);

    if (mycol == panel_col)
	{
		int_t lk = useSymV2Solve() ? symV2PanelIndex(k0) : g2lCol(k0);
        if (lk < 0)
            return 0;
        if (!lPanelVec[lk].isEmpty())
		{
            
            MPI_Status status;
				superlu_gpu_mpi_recv(A_gpu.LvalRecvBufs[0], LvalRecvBufs[0],
						 sizeof(Ftype), static_cast<int>(lPanelVec[lk].nzvalSize()),
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
    
	if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{
            superlu_gpu_mpi_send(uPanelVec[lk].blkPtrGPU(0), UvalRecvBufs[0],
                    sizeof(Ftype), static_cast<int>(uPanelVec[lk].nzvalSize()),
                    get_mpi_type<Ftype>(), receiverGrid, k0, grid3d->zscp.comm);
				SCT->commVolRed += uPanelVec[lk].nzvalSize() * sizeof(Ftype);
			}
		}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zRecvUPanelGPU(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta)
{
    if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{

            MPI_Status status;
				superlu_gpu_mpi_recv(A_gpu.UvalRecvBufs[0], UvalRecvBufs[0],
						 sizeof(Ftype), static_cast<int>(uPanelVec[lk].nzvalSize()),
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
