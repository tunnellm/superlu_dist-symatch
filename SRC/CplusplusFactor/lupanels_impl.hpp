#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>

#include "superlu_defs.h"
#include "luAuxStructTemplated.hpp"
#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#include "xlupanels_GPU.cuh"
#endif
#include "lupanels.hpp"
#include "xlupanels.hpp"
#include "symldl_v2_lupanels_impl.hpp"
#include "xlupanels_diag_buffers_impl.hpp"
#include "xlupanels_gpu_panel_access_impl.hpp"
#include "xlupanels_constructor_impl.hpp"
#include "xlupanels_cpu_update_impl.hpp"
#include "xlupanels_cpu_panel_factor_impl.hpp"
#include "superlu_blas.hpp"

int numProcsPerNode(MPI_Comm baseCommunicator);
