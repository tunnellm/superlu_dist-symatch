#pragma once

#include <vector>

// Sparse host-buffer to GPU value-copy helpers.

template <typename Ftype>
__global__ void indirectCopy(Ftype *dest, Ftype *src, int_t *idx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dest[idx[i]] = src[i];
}

template <typename Ftype>
void copyToGPU(Ftype *gpuValBasePtr, std::vector<Ftype> &valBufferPacked,
               std::vector<int_t> &valIdx)
{
    int nnzCount = valBufferPacked.size();
    int_t gpuLvalSizePacked = nnzCount * sizeof(Ftype);
    int_t gpuLidxSizePacked = nnzCount * sizeof(int_t);

    Ftype *dlvalPacked;
    int_t *dlidxPacked;
    gpuErrchk(cudaMalloc(&dlvalPacked, gpuLvalSizePacked));
    gpuErrchk(cudaMalloc(&dlidxPacked, gpuLidxSizePacked));

    gpuErrchk(cudaMemcpy(dlvalPacked, valBufferPacked.data(),
                         gpuLvalSizePacked, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dlidxPacked, valIdx.data(), gpuLidxSizePacked,
                         cudaMemcpyHostToDevice));

    const int ThreadblockSize = 256;
    int nThreadBlocks = (nnzCount + ThreadblockSize - 1) / ThreadblockSize;
    indirectCopy<<<nThreadBlocks, ThreadblockSize>>>(
        gpuValBasePtr, dlvalPacked, dlidxPacked, nnzCount);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(dlvalPacked));
    gpuErrchk(cudaFree(dlidxPacked));
}

template <typename Ftype>
void copyToGPU_Sparse(Ftype *gpuValBasePtr, Ftype *valBuffer,
                      int_t gpuLvalSize)
{
    int numFtypes = gpuLvalSize / sizeof(Ftype);
    std::vector<Ftype> valBufferPacked;
    std::vector<int_t> valIdx;
    for (int_t i = 0; i < numFtypes; i++)
    {
        if (valBuffer[i] != 0)
        {
            valBufferPacked.push_back(valBuffer[i]);
            valIdx.push_back(i);
        }
    }
    printf("%d non-zero elements in the panel, wrt original=%d\n",
           valBufferPacked.size(), numFtypes);
    copyToGPU(gpuValBasePtr, valBufferPacked, valIdx);
}
