#pragma once
#include "xeg.cuh"
#include <cuda/atomic>

// Error checking macro for GPU 
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

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t cuda_status__ = (call);                                  \
    if (cuda_status__ != cudaSuccess) {                                  \
      fprintf(stderr,                                                    \
              "CUDA API error: %s at %s:%d\n",                           \
              cudaGetErrorString(cuda_status__),                         \
              __FILE__,                                                  \
              __LINE__);                                                 \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)    
 
namespace {
constexpr uint32_t kGpuThreads = 512u;
constexpr uint32_t kGpuReachabilityThreads = 256u;

// Runtime batching and levelization options live in gpu_config.hpp.
// Detailed per-iteration diagnostics add transfers and remain disabled in
// production timings; total DAG timing is always reported.
constexpr bool kEnableGpuDagIterationLog = false;

template <typename T>
void cuda_malloc_array(T*& ptr, size_t count)
{
  ptr = nullptr;

  if (count == 0) {
    return;
  }

  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&ptr),
    count * sizeof(T)));
}

template <typename T>
void cuda_free_and_null(T*& ptr) noexcept
{
  if (ptr == nullptr) {
    return;
  }

  cudaError_t err = cudaFree(ptr);

  if (err != cudaSuccess) {
    fprintf(
      stderr,
      "Warning: cudaFree failed: %s\n",
      cudaGetErrorString(err));
  }

  ptr = nullptr;
}
template<class F> struct GpuScopeExit {
  F release;
  ~GpuScopeExit() { release(); }
};
template<class F> GpuScopeExit<F> gpu_scope_exit(F release) { return {release}; }
} // namespace
