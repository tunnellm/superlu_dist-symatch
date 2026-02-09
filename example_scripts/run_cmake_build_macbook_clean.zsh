
#!/bin/zsh
cd ../
export BREWPATH=/usr/local/Cellar
export SUPERLUROOT=$PWD

# Detect package version numbers from homebrew (the upgraded package versions from the homebrew commands above).
pythonversion=$(ls ${BREWPATH}/python@3.9/ | sort -V | tail -n 1)
gccversion=$(ls ${BREWPATH}/gcc/ | sort -V | tail -n 1)
openblasversion=$(ls ${BREWPATH}/openblas/ | sort -V | tail -n 1)
libeventversion=$(ls ${BREWPATH}/libevent/ | sort -V | tail -n 1)
hwlocversion=$(ls ${BREWPATH}/hwloc/ | sort -V | tail -n 1)

echo "Detected Python version: ${pythonversion}"
echo "Detected GCC version: ${gccversion}"
echo "Detected OpenBLAS version: ${openblasversion}"
echo "Detected libevent version: ${libeventversion}"
echo "Detected hwloc version: ${hwlocversion}"


export BLAS_LIB=$BREWPATH/openblas/$openblasversion/lib/libblas.dylib
export LAPACK_LIB=$BREWPATH/openblas/$openblasversion/lib/liblapack.dylib  

export ParMETIS_DIR=$SUPERLUROOT/parmetis-4.0.3/install/
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export PARMETIS_LIBRARIES="$ParMETIS_DIR/lib/libparmetis.dylib;$ParMETIS_DIR/lib/libmetis.dylib"

export LD_LIBRARY_PATH=$ParMETIS_DIR/lib/:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ParMETIS_DIR/lib/:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$ParMETIS_DIR/lib/:$DYLD_LIBRARY_PATH



OPENMPFLAG=fopenmp
CC=$BREWPATH/gcc/$gccversion/bin/gcc-15
FTN=$BREWPATH/gcc/$gccversion/bin/gfortran-15
CXX=$BREWPATH/gcc/$gccversion/bin/g++-15

export MPICC="$SUPERLUROOT/openmpi-5.0.6/bin/mpicc"
export MPICXX="$SUPERLUROOT/openmpi-5.0.6/bin/mpicxx"
export MPIF90="$SUPERLUROOT/openmpi-5.0.6/bin/mpif90"
export PATH=$SUPERLUROOT/openmpi-5.0.6/bin:$PATH
export LD_LIBRARY_PATH=$SUPERLUROOT/openmpi-5.0.6/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$SUPERLUROOT/openmpi-5.0.6/lib:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$SUPERLUROOT/openmpi-5.0.6/lib/:$DYLD_LIBRARY_PATH

export LD_LIBRARY_PATH=$SUPERLUROOT/matching/lib/matching/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$SUPERLUROOT/matching/lib/matching/lib:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$SUPERLUROOT/matching/lib/matching/lib:$DYLD_LIBRARY_PATH


alias python=$BREWPATH/python@3.9/$pythonversion/bin/python3.9  # this makes sure virtualenv uses the correct python version
alias pip=$BREWPATH/python@3.9/$pythonversion/bin/pip3.9

# cd $SUPERLUROOT
# rm -rf openmpi-5.0.6*
# wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.6.tar.bz2
# bzip2 -d openmpi-5.0.6.tar.bz2
# tar -xvf openmpi-5.0.6.tar 
# cd openmpi-5.0.6/ 
# # ./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CXX F77=$FTN FC=$FTN --enable-mpi1-compatibility --disable-dlopen
# ./configure --prefix=$PWD CC=$CC CXX=$CXX F77=$FTN FC=$FTN --with-libevent=/usr/local/Cellar/libevent/$libeventversion --with-hwloc=/usr/local/Cellar/hwloc/$hwlocversion
# make -j8
# make install


# cd $SUPERLUROOT
# rm -rf parmetis_4.0.3*
# wget https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/parmetis/4.0.3-4/parmetis_4.0.3.orig.tar.gz
# tar -xf parmetis_4.0.3.orig.tar.gz
# cd parmetis-4.0.3/
# sed -i '' '/set(ParMETIS_LIBRARY_TYPE SHARED)/a\
# set(METIS_LIBRARY_TYPE SHARED)
# ' CMakeLists.txt
# mkdir -p install
# make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
# make install > make_parmetis_install.log 2>&1
# cd ../
# cp $PWD/parmetis-4.0.3/build/$(echo "$(uname -s)-$(uname -m)")/libmetis/libmetis.dylib $PWD/parmetis-4.0.3/install/lib/.
# cp $PWD/parmetis-4.0.3/metis/include/metis.h $PWD/parmetis-4.0.3/install/include/.


cd $SUPERLUROOT
mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
    -DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
    -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
    -DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch" \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_COMPILER=$MPICXX \
    -DCMAKE_C_COMPILER=$MPICC \
    -DCMAKE_Fortran_COMPILER=$MPIF90 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DTPL_ENABLE_LAPACKLIB=ON \
    -DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
    -DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
    -DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
    -DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES \
    -DTPL_ENABLE_SYMATCHLIB=ON \
    -DTPL_SYMATCH_INCLUDE_DIRS="$SUPERLUROOT/matching/symatch/inc;$SUPERLUROOT/matching/symatch/util;$SUPERLUROOT/matching/lib/matching" \
    -DTPL_SYMATCH_LIBRARIES="$SUPERLUROOT/matching/lib/matching/lib/libsuitor.dylib"

make pddrive
make pddrive3d
make pddrive3d-sym

