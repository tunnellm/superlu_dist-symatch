#pragma once

// Host/device panel synchronization helpers for xLUstruct_t.

template <typename Ftype>
int_t xLUstruct_t<Ftype>::copyLUGPUtoHost()
{
    if (useSymV2Solve())
    {
        for (int_t i = 0; i < symV2PanelCount(); ++i)
            if (symV2PanelGid(i) < nsupers &&
                isNodeInMyGrid[symV2PanelGid(i)] == 1)
                lPanelVec[i].copyFromGPU();

        if (needsUPanelStorage())
        {
            if (uPanelVec == NULL)
                ABORT("U host panel storage is missing.");
            for (int_t i = 0; i < symV2RowCount(); ++i)
                if (symV2RowGid(i) < nsupers &&
                    isNodeInMyGrid[symV2RowGid(i)] == 1)
                    uPanelVec[i].copyFromGPU();
        }
        return 0;
    }

    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
            lPanelVec[i].copyFromGPU();

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            uPanelVec[i].copyFromGPU();
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::copyLUHosttoGPU()
{
    if (useSymV2Solve())
    {
        for (int_t i = 0; i < symV2PanelCount(); ++i)
            if (symV2PanelGid(i) < nsupers &&
                isNodeInMyGrid[symV2PanelGid(i)] == 1)
                lPanelVec[i].copyBackToGPU();

        if (needsUPanelStorage())
        {
            if (uPanelVec == NULL)
                ABORT("U host panel storage is missing.");
            if (A_gpu.uPanelVec == NULL)
                ABORT("U GPU panel storage is missing.");
            for (int_t i = 0; i < symV2RowCount(); ++i)
                if (symV2RowGid(i) < nsupers &&
                    isNodeInMyGrid[symV2RowGid(i)] == 1)
                    uPanelVec[i].copyBackToGPU();
        }
        return 0;
    }

    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
            lPanelVec[i].copyBackToGPU();

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            uPanelVec[i].copyBackToGPU();
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::checkGPU()
{
    if (useSymV2Solve())
    {
        for (int_t i = 0; i < symV2PanelCount(); ++i)
            lPanelVec[i].checkGPU();

        if (needsUPanelStorage())
        {
            if (uPanelVec == NULL)
                ABORT("U host panel storage is missing.");
            if (A_gpu.uPanelVec == NULL)
                ABORT("U GPU panel storage is missing.");
            for (int_t i = 0; i < symV2RowCount(); ++i)
                uPanelVec[i].checkGPU();
        }

        std::cout << "Checking LU struct completed succesfully"
                  << "\n";
        return 0;
    }

    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
        lPanelVec[i].checkGPU();

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        uPanelVec[i].checkGPU();

    std::cout << "Checking LU struct completed succesfully"
              << "\n";
    return 0;
}
