#pragma once

template <typename Ftype>
xLUstruct_t<Ftype>::~xLUstruct_t()
{

    /* Yang: Deallocate the lPanelVec[i] and uPanelVec[i] here instead of using destructors ~lpanel_t or ~upanel_t,
    as xlpanel_t/upanel_t is used for holding temporary communication buffers as well. Note that lPanelVec[i].val is not deallocated here as it's pointing to the L data in the C code*/

    int_t localPanelCount = symV2PanelCount();
    for (int_t i = 0; i < localPanelCount; ++i)
    {
        int_t gid = symV2PanelGid(i);
        if (gid < nsupers && isNodeInMyGrid != NULL &&
            isNodeInMyGrid[gid] == 1)
        {
            if (lPanelVec[i].index)
                SUPERLU_FREE(lPanelVec[i].index);
            // SUPERLU_FREE(lPanelVec[i].val);
        }
    }

    if (uPanelVec != NULL)
    {
        int_t localRowCount = symV2RowCount();
        for (int_t i = 0; i < localRowCount; ++i)
        {
            int_t gid = symV2RowGid(i);
            if (gid < nsupers && isNodeInMyGrid != NULL &&
                isNodeInMyGrid[gid] == 1)
            {
                if (uPanelVec[i].index)
                    SUPERLU_FREE(uPanelVec[i].index);
                if (uPanelVec[i].val)
                    SUPERLU_FREE(uPanelVec[i].val);
            }
        }
    }

    delete[] lPanelVec;
    if (uPanelVec != NULL)
        delete[] uPanelVec;

    symV2FreeDiagBlocks();

    /* free diagonal L and U blocks */
    // dfreeDiagFactBufsArr(maxLeafNodes, dFBufs);
    freeDiagFactBufsArr(numDiagBufs, dFBufs);

    SUPERLU_FREE(bigV);
    SUPERLU_FREE(indirect);
    SUPERLU_FREE(indirectRow);
    SUPERLU_FREE(indirectCol);

    int i;
    for (i = 0; i < options->num_lookaheads; i++)
    {
        SUPERLU_FREE(LvalRecvBufs[i]);
        SUPERLU_FREE(UvalRecvBufs[i]);
        SUPERLU_FREE(LidxRecvBufs[i]);
        SUPERLU_FREE(UidxRecvBufs[i]);
        symV2FreeStreamHostBuffers(i);
    }

    for (i = 0; i < numDiagBufs; i++)
        SUPERLU_FREE(diagFactBufs[i]);

    /* Sherry added the following, which comes from batch setup */
    superlu_acc_offload = sp_ienv_dist(10, options); //get_acc_offload();
    if (superlu_acc_offload)
    {
        // printf(".. free batch buffers\n");  fflush(stdout);
        SUPERLU_FREE(A_gpu.dFBufs);
        SUPERLU_FREE(A_gpu.gpuGemmBuffs);

#ifdef HAVE_CUDA
        symV2FreeGpuStorage();
#endif

        for (int stream = 0; stream < A_gpu.numCudaStreams; stream++)
        {
            cusolverDnDestroy(A_gpu.cuSolveHandles[stream]);
            cublasDestroy(A_gpu.cuHandles[stream]);
            cublasDestroy(A_gpu.lookAheadLHandle[stream]);
            cublasDestroy(A_gpu.lookAheadUHandle[stream]);
        }
    }

    SUPERLU_FREE(isNodeInMyGrid);

} /* end destructor xLUstruct_t */
