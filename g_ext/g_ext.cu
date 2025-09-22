#include <chrono>
#include <thread>
#include <assert.h>
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <cmath>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <unordered_set>
#include <utility> 
#include <algorithm>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "g_ext.cuh"

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// G_EXT fucntions

__global__ void GPU_print_test() {
    printf("threadIdx.x = %d, hello world!\n", threadIdx.x);
}

void G_EXT::gpu_print() {
    printf("Get inside G_EXT::gpu_print, starts to launch GPU_print_test...\n");
    GPU_print_test <<< 1, 32 >>>();
    cudaDeviceSynchronize(); 
    printf("\ngpu_equality_saturator::gpu_print test ends...\n");
}
