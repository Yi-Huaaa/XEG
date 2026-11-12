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

// Exact overlap insertion preserves the original 256-lane
// transposed-batch policy.
constexpr uint32_t kGpuDagExactBatchSize = 256u; // default: 256; 0802

// Bitmap overlap insertion uses fewer concurrent candidate
// lanes to stabilize solution quality. This does not change
// the CUDA threads assigned to each Block candidate.
constexpr uint32_t kGpuDagBitmapBatchSize = 32u;

static_assert(
  kGpuDagExactBatchSize > 0u &&
  kGpuDagExactBatchSize <= 1024u,
  "Exact DAG batch size must be a valid CUDA block size");

static_assert(
  kGpuDagBitmapBatchSize > 0u,
  "Bitmap DAG batch size must be positive");

// Compile-time controller for GPU DAG levelization.
//
// Keep this false while verifying that controller separation
// preserves the original host-driven implementation.
// Change it to true only after the persistent path compiles
// and the host-driven baseline still passes all tests.
constexpr bool kEnableGpuDagHybridLevelization = true;

// Single-block persistent levelization parameters.
constexpr uint32_t kGpuDagPersistentThreads = 512u;

// Enter persistent mode only after the current frontier and
// the preceding frontiers have remained at or below this width.
constexpr uint32_t
  kGpuDagPersistentEnterFrontierThreshold = 512u; // 256 and 512 get similiar runtime 

// Require this many consecutive small frontiers before entering
// persistent mode. This prevents switching because of one short,
// temporary dip in frontier width.
constexpr uint32_t
  kGpuDagPersistentRequiredSmallLevels = 8u; // 8 is good

// The persistent kernel processes frontiers up to this width.
// A wider frontier is left unprocessed and returned to the host,
// which resumes with the existing multi-block kernel.
//
// With 512 threads, 1024 nodes require at most two node rounds.
constexpr uint32_t
  kGpuDagPersistentEmergencyExitThreshold = 1024u;

static_assert(
  kGpuDagPersistentThreads > 0u &&
  kGpuDagPersistentThreads <= 1024u,
  "Persistent block size must be a valid CUDA block size");

static_assert(
  kGpuDagPersistentThreads % 32u == 0u,
  "Persistent block size should be a whole number of warps");

static_assert(
  kGpuDagPersistentEnterFrontierThreshold > 0u,
  "Persistent entry threshold must be positive");

static_assert(
  kGpuDagPersistentRequiredSmallLevels > 0u,
  "Persistent entry streak must be positive");

static_assert(
  kGpuDagPersistentEmergencyExitThreshold >=
    kGpuDagPersistentEnterFrontierThreshold,
  "Emergency threshold must not be below entry threshold");

// Enables detailed GPU DAG diagnostics:
//   1. per-iteration proposal/survival/rollback logs
//   2. core/filter runtime breakdown
//
// GPU DAG total runtime and the outer-loop termination copy
// remain enabled regardless of this flag.
constexpr bool kEnableGpuDagIterationLog = false;

// ============================================================
// Batch low-activity controller
//
// A completed batch iteration is considered low activity when:
//
//   survived * 100 <= historical_peak * low_activity_percent
//
// The low-activity condition is reached only after:
//   1. at least minimum_iterations have completed
//   2. low activity has been observed for patience
//      consecutive iterations
//
// The insertion policy decides what happens after the
// condition is reached:
//
//   FullBatch:
//     ignore the condition
//
//   BatchEarlyStop:
//     stop DAG construction
//
//   BatchThenDirect:
//     switch subsequent iterations to direct insertion
//
// These constants affect only the host-side outer-loop
// controller. They do not modify insertion, levelization,
// finalization, or filtering kernels.
// ============================================================
constexpr uint32_t
  kGpuDagLowActivityMinimumIterations = 4u;

constexpr uint32_t
  kGpuDagLowActivityPercent = 3u;

constexpr uint32_t
  kGpuDagLowActivityPatience = 2u;

static_assert(
  kGpuDagLowActivityMinimumIterations > 0u,
  "Low-activity minimum iterations must be positive");

static_assert(
  kGpuDagLowActivityPercent > 0u &&
  kGpuDagLowActivityPercent <= 100u,
  "Low-activity percentage must be in [1, 100]");

static_assert(
  kGpuDagLowActivityPatience > 0u,
  "Low-activity patience must be positive");

static_assert(
  kGpuDagLowActivityMinimumIterations >=
    kGpuDagLowActivityPatience,
  "Minimum iterations should cover the low-activity streak");

// ============================================================
// Bitmap direct-mode iteration limit
//
// This limit applies only when:
//
//   1. the configured DAG kernel is Warp or Block
//   2. bitmap overlap detection is enabled
//   3. the controller has entered Direct mode
//
// Exact overlap detection remains convergence-driven and is
// not limited by this parameter.
// ============================================================
constexpr uint32_t
  kGpuDagBitmapMaxDirectIterations = 5u; // default setting = 5

static_assert(
  kGpuDagBitmapMaxDirectIterations > 0u,
  "Bitmap direct iteration limit must be positive");

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
