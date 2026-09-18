
module load PrgEnv-gnu
module load gcc-native/12.3
module load cmake
module load cudatoolkit
module load nccl
module load cray-libsci
# module load nvshmem/2.11.0
export MAGMA_ROOT=/global/cfs/cdirs/m2957/lib/magma_master
export PTSCOTCH_ROOT=/global/cfs/cdirs/m2957/lib/lib/PrgEnv-gnu/scotch-7.0.10
NVSHMEM_HOME=/global/cfs/cdirs/m2957/lib/lib/PrgEnv-gnu/nvshmem_src_2.8.0-3/build/
cmake .. \
  -DCMAKE_C_FLAGS="-O2 -std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0 -DAdd_" \
  -DCMAKE_CXX_FLAGS="-O2 -std=c++17 -DMULTIPHASE -DPRINT -DNGPU=1 -DUSE_32BIT_GRAPH" \
  -DCMAKE_Fortran_FLAGS="-O2" \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_Fortran_COMPILER=ftn \
  -DXSDK_ENABLE_Fortran=ON \
  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DTPL_ENABLE_LAPACKLIB=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DTPL_ENABLE_CUDALIB=ON \
  -DCMAKE_CUDA_FLAGS="-I${NVSHMEM_HOME}/include -I${MPICH_DIR}/include -ccbin=CC" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_INSTALL_LIBDIR=./lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DTPL_ENABLE_MAGMALIB=OFF \
  -DTPL_MAGMA_INCLUDE_DIRS="${MAGMA_ROOT}/include" \
  -DTPL_MAGMA_LIBRARIES="${MAGMA_ROOT}/lib/libmagma.so" \
  -DTPL_BLAS_LIBRARIES="$CFS/m2957/lib/lib/PrgEnv-gnu/OpenBLAS/build/install/lib//libopenblas.so" \
  -DTPL_LAPACK_LIBRARIES="$CFS/m2957/lib/lib/PrgEnv-gnu/OpenBLAS/build/install/lib//libopenblas.so" \
  -DXSDK_INDEX_SIZE=32 \
  -DTPL_ENABLE_PARMETISLIB=ON \
  -DTPL_PARMETIS_INCLUDE_DIRS="${PTSCOTCH_ROOT}/include" \
  -DTPL_PARMETIS_LIBRARIES="${PTSCOTCH_ROOT}/lib/libptscotchparmetisv3.a;${PTSCOTCH_ROOT}/lib/libscotchmetisv5.a;${PTSCOTCH_ROOT}/lib/libptscotch.a;${PTSCOTCH_ROOT}/lib/libscotch.a;${PTSCOTCH_ROOT}/lib/libptscotcherr.a;${PTSCOTCH_ROOT}/lib/libscotcherr.a;m" \
  -DTPL_ENABLE_COMBBLASLIB=OFF \
  -DTPL_ENABLE_NVSHMEM=OFF \
  -DTPL_NVSHMEM_LIBRARIES="-L${CUDA_HOME}/lib64/stubs/ -lnvidia-ml -L/usr/lib64 -lgdrapi -lstdc++ -L/opt/cray/libfabric/1.22.0/lib64 -lfabric -L${NVSHMEM_HOME}/lib -lnvshmem" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DMPIEXEC_NUMPROC_FLAG=-n \
  -DMPIEXEC_EXECUTABLE=/usr/bin/srun \
  -DMPIEXEC_MAX_NUMPROCS=16 \
  -DTPL_ENABLE_SYMATCHLIB=ON \
  -DTPL_ENABLE_SUMAC=ON \
  -DTPL_ENABLE_MC80=ON \
  -DTPL_SYMATCH_INCLUDE_DIRS="$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/symatch/inc;$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/symatch/util;$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/matching;$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/sumac" \
  -DTPL_SYMATCH_LIBRARIES="$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/matching/lib/libsuitor.a;${NCCL_HOME}/lib/libnccl.so;$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/sumac/libsumac.a" \
  -DTPL_MC80_INCLUDE_DIRS="$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/hsl_mc80-1.1.4/include" \
  -DTPL_MC80_LIBRARIES="$CFS/m2957/liuyangz/my_research/superlu_dist-symatch/matching/lib/hsl_mc80-1.1.4/build/lib/libhsl_mc80.a"

make pddrive -j16
make pddrive3d -j16
make pddrive3d-sym -j16
make pzdrive3d -j16