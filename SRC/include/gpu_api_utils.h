/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 * Modified:
 *     May 22, 2022        version 8.0.0
 * </pre>
 */

#ifndef gpu_api_utils_H
#define gpu_api_utils_H

#ifdef GPU_ACC

#include <stdint.h>

#include "gpu_wrapper.h"
typedef struct LUstruct_gpu_  LUstruct_gpu;  // Sherry - not in this distribution

/* Checkpoint-based, device-wide usage from gpuMemGetInfo().  The peak is the
 * largest explicit sample, not a continuously monitored high-water mark. */
typedef struct {
    uint64_t total_bytes;
    uint64_t baseline_used_bytes;
    uint64_t factor_end_used_bytes;
    uint64_t peak_used_bytes;
    int device;
    int samples;
    int query_failures;
    int factor_end_recorded;
    int valid;
} superlu_gpu_memory_stats_t;

#ifdef __cplusplus
extern "C" {
#endif
extern void DisplayHeader();
extern const char* gpublasGetErrorString(gpublasStatus_t status);
extern gpuError_t checkGPU(gpuError_t);
extern gpublasStatus_t checkGPUblas(gpublasStatus_t);
extern gpublasHandle_t create_handle ();
extern void destroy_handle (gpublasHandle_t handle);
extern int superlu_gpu_memory_tracker_start(void);
extern int superlu_gpu_memory_tracker_enabled(void);
extern int superlu_gpu_memory_tracker_sample(void);
extern int superlu_gpu_memory_tracker_mark_factor_end(void);
extern int superlu_gpu_memory_tracker_stop(
    superlu_gpu_memory_stats_t *stats);
#ifdef __cplusplus
  }
#endif

#endif // end GPU_ACC
#endif 
