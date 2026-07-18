#pragma once

#include "anc25d.hpp"

inline MPI_Comm *anc25d_t::initComm(gridinfo3d_t *grid3d)
{
    int maxLvl = log2i(grid3d->zscp.Np) + 1;
    int myGrid = grid3d->zscp.Iam;
    MPI_Comm *zCommOut = (MPI_Comm *) SUPERLU_MALLOC(
        (maxLvl - 1) * sizeof(MPI_Comm));
    MPI_Comm zComm = grid3d->zscp.comm;
    for (int_t alvl = 0; alvl < maxLvl - 1; ++alvl)
    {
        int lvlCommSize = 1 << (alvl + 1);
        int lvlCommBase = (myGrid / lvlCommSize) * lvlCommSize;
        int lvlCommRank = myGrid - lvlCommBase;
        MPI_Comm lvlComm;
        MPI_Comm_split(zComm, lvlCommBase, lvlCommRank, &lvlComm);
        zCommOut[alvl] = lvlComm;
    }
    return zCommOut;
}
