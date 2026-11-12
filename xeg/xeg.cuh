#ifndef XEG_CUH
#define XEG_CUH
#include <algorithm>
#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <list>
#include <thread>
#include <queue>
#include <utility>
#include <atomic>
#include <chrono>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/strong_components.hpp>
#include <nlohmann/json.hpp>
#include <limits>
#include <omp.h>
#include <cstdint> // For UINT32_MAX
#include <cuda_runtime.h>
#include <limits.h>
#include <random> // for dump new json
#include <cmath>



/* ILP Solver */
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/base/logging.h" // For LOG(INFO)

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "cuopt_solver.hpp"
#include "cpu_extraction.hpp"
#include "gpu_config.hpp"

enum class DagKernel : uint32_t {
  Thread = 0,
  Warp   = 1,
  Block  = 2,
};

// Selects the ILP solver backend.
//
// CpSat:
//   Use the existing OR-Tools CP-SAT implementation.
//
// CuOpt:
//   Use the NVIDIA cuOpt implementation when XEG is built
//   with cuOpt support.
enum class IlpBackend : uint32_t {
  CpSat = 0u,
  CuOpt = 1u,
};


inline IlpBackend parse_ilp_backend(
  const uint32_t value)
{
  switch (value) {
    case 0u:
      return IlpBackend::CpSat;

    case 1u:
      return IlpBackend::CuOpt;

    default:
      throw std::invalid_argument(
        "Invalid ILP_BACKEND value: " +
        std::to_string(value));
  }
}


inline const char* ilp_backend_name(
  const IlpBackend backend)
{
  switch (backend) {
    case IlpBackend::CpSat:
      return "CP-SAT";

    case IlpBackend::CuOpt:
      return "cuOpt";

    default:
      throw std::invalid_argument(
        "Invalid ILP backend");
  }
}

namespace xeg_detail {

enum class GpuDagPersistentExitReason : uint32_t
{
  Completed = 0u,
  FrontierTooWide = 1u,
  InvalidQueueState = 2u
};

// One compact device-to-host snapshot returned by the
// single-block persistent levelization kernel.
//
// The persistent kernel receives its initial frontier through
// kernel arguments. This buffer is written only when the kernel
// completes or exits back to the host.
//
// Keeping the input in kernel arguments avoids an additional
// host-to-device state copy before every persistent launch.
struct alignas(16) GpuDagPersistentState
{
  uint32_t queue_begin;
  uint32_t queue_count;
  int32_t current_level;
  GpuDagPersistentExitReason exit_reason;
};

static_assert(
  sizeof(GpuDagPersistentState) == 16u,
  "GpuDagPersistentState must remain a compact 16-byte POD");

}  // namespace xeg_detail

class XEG {

private:
  enum class CpsatInitialSolutionMode : uint32_t {
    GpuTree = 0u,
    Random = 1u,
  };

public:
  XEG() = default;
  XEG(const XEG&) = delete;
  XEG& operator=(const XEG&) = delete;
  ~XEG();
  // Read input file
  void read(const std::string input_path) {
    _gpu_free_buffers();
    _read_input(input_path); 
  }

  // Dump the current in-memory e-graph using the same JSON
  // structure accepted by _read_input().
  //
  // The output directory is fixed to:
  //
  //   /home/ychung79/gpu_extraction/ablation_benchmarks
  //
  // The generated filename has the form:
  //
  //   <input-stem>_sf<multiplier>_<index>.json
  //
  // Example:
  //
  //   mul112.rename_sf2_1.json
  //
  // Existing files are never intentionally overwritten. The
  // smallest available positive index is selected.
  //
  // Returns the complete path of the generated JSON file.
  std::string dump_output_degree_exp_json(
    const std::string& input_path,
    const uint32_t output_degree_multiplier);
  // Read the original e-graph and then construct the isolated
  // output-degree ablation graph.
  //
  // output_degree_multiplier:
  //   1 -> keep the original e-node-to-e-class edge count
  //   2 -> target 2x the original edge count
  //   8 -> target 8x the original edge count
  //
  // The actual multiplier may be smaller when legal
  // source-target pairs become difficult to find.
  //
  // This interface must be called on a fresh XEG object.
  void read_output_degree_exp(
    const std::string& input_path,
    const uint32_t output_degree_multiplier,
    const uint32_t random_seed = 42u,
    const bool dump_augmented_json = false)
  {
    if (!_ende_name.empty()) {
      throw std::logic_error(
        "read_output_degree_exp requires a fresh XEG object");
    }

    if (output_degree_multiplier == 0u) {
      throw std::invalid_argument(
        "Output-degree multiplier must be at least 1");
    }

    // Reuse the complete production read and pruning path.
    _read_input(input_path);

    // _read_input() currently reports a file-open failure and
    // returns. Prevent the ablation post-processing from running
    // on an uninitialized object.
    if (_ende_name.empty()) {
      throw std::runtime_error(
        "Output-degree experiment received an empty "
        "or unreadable input graph");
    }

    _augment_output_degree_for_exp(
      output_degree_multiplier,
      random_seed);

    // Dump only after augmentation and reconstruction have
    // completed. At this point _ende_child,
    // _ende_child_ptr, and all derived graph data represent
    // the final augmented graph.
    if (dump_augmented_json) {
      const std::string dumped_path =
        dump_output_degree_exp_json(
          input_path,
          output_degree_multiplier);

      std::cout
        << "[Output-Degree Exp] dumped_json="
        << dumped_path
        << "\n";
    }
  }

  // ============================================================
  // Unit-test interface
  //
  // These wrappers expose selected internal GPU operations for
  // deterministic unit tests. They are not used by run().
  // ============================================================
  struct CsrGraphSnapshot {
    std::vector<uint32_t> forward_adj;
    std::vector<uint32_t> forward_ptr;
    std::vector<uint32_t> inverse_adj;
    std::vector<uint32_t> inverse_ptr;
  };
  
  struct GpuInsertionResult {
    std::vector<uint32_t> selected_state;
    std::vector<int> level;
    uint32_t u0 = 0;
  };

  struct GpuFinalizeResult {
    std::vector<uint32_t> selected_state;
    std::vector<uint32_t> residual_degree;
    std::vector<int> level;
    int num_levels = 0;
    uint32_t u0 = 0;
  };  
  
  struct GpuDagConstructionResult {
    std::vector<uint32_t> selected_state;
    std::vector<uint32_t> residual_degree;
    std::vector<int> level;
    std::vector<uint32_t> in_subgraph;
    int num_levels = 0;
    uint32_t final_u0 = 0;
  };

  struct GpuRootToLeafFilterResult {
    std::vector<uint32_t> selected_state;
    std::vector<uint32_t> from_roots;
    std::vector<uint32_t> from_leaves;
    std::vector<uint32_t> in_subgraph;
  };

  // Selects the levelization implementation used only by
  // unit-test wrappers.
  //
  // Configured:
  //   Follow the production compile-time dispatcher in
  //   _gpu_dag_levelize().
  //
  // HostDriven:
  //   Directly invoke the original host-controlled
  //   implementation.
  //
  // Hybrid:
  //   Directly invoke the hybrid persistent implementation.
  //
  // This enum is never consulted by run().
  enum class GpuLevelizationBackend : uint32_t {
    Configured = 0u,
    HostDriven = 1u,
    Hybrid = 2u,
  };

  // Selects the GPU DAG insertion policy.
  //
  // FullBatch:
  //   Use the batch insertion kernel until the original
  //   u0 == 0 convergence condition is reached.
  //
  // BatchEarlyStop:
  //   Use batch insertion and stop when the low-activity
  //   condition is triggered.
  //
  // BatchThenDirect:
  //   Start with batch insertion. When the low-activity
  //   condition is triggered, switch permanently to the
  //   one-thread-per-e-node direct insertion kernel and
  //   continue until u0 == 0.
  enum class GpuDagInsertionPolicy : uint32_t {
    FullBatch = 0u,
    BatchEarlyStop = 1u,
    BatchThenDirect = 2u,
  };

  struct GpuLevelizationResult {
    std::vector<uint32_t> selected_state;
    std::vector<uint32_t> residual_degree;
    std::vector<int> level;

    int num_levels = 0;

    // Test-only execution metadata.
    //
    // These values are derived by inspecting the existing
    // persistent-state buffer after a unit-test call. No
    // counters or branches are added to the production
    // levelization implementation.
    bool persistent_path_used = false;
    bool persistent_completed = false;
    bool persistent_exited_to_host = false;
  };

  CsrGraphSnapshot
  unittest_get_csr_snapshot() const;

  // Invoke the production ILP backend dispatcher with an empty
  // model. This wrapper exists only to verify backend routing.
  void unittest_solve_ilp_backend(
    IlpBackend ilp_backend,
    CuoptMode cuopt_mode = CuoptMode::Presolve);

  // Run exactly one batch insertion phase.
  // This wrapper is used only by unit tests.
  GpuInsertionResult unittest_gpu_insert_batch_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion);

  // Run exactly one one-thread-per-e-node direct insertion
  // phase. This wrapper is used only by unit tests.
  GpuInsertionResult unittest_gpu_insert_direct_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion);

  // Run exactly one Warp-Batch insertion phase.
  //
  // One 32-lane warp acts as one transposed candidate lane.
  // Multiple independent candidate warps may share one CUDA
  // block according to threads_per_block.
  //
  // This wrapper is used only by unit tests.
  GpuInsertionResult
  unittest_gpu_insert_warp_batch_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion,
    uint32_t threads_per_block,
    uint32_t bitmap_bits = 0u);

  // Run exactly one Warp-Direct insertion phase.
  //
  // One 32-lane warp evaluates exactly one candidate e-node.
  // Multiple independent candidate warps may share one CUDA
  // block according to threads_per_block.
  //
  // This wrapper is used only by unit tests.
  GpuInsertionResult
  unittest_gpu_insert_warp_direct_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion,
    uint32_t threads_per_block,
    uint32_t bitmap_bits = 0u);

  // Run exactly one Block-Batch insertion phase.
  //
  // One CUDA block acts as one transposed candidate lane.
  // Threads inside the block cooperatively evaluate one
  // candidate at a time.
  //
  // This wrapper is used only by unit tests.
  GpuInsertionResult
  unittest_gpu_insert_block_batch_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion,
    uint32_t threads_per_candidate,
    uint32_t bitmap_bits = 0u);

  // Run exactly one Block-Direct insertion phase.
  //
  // One CUDA block evaluates one candidate e-node.
  // Threads inside the block cooperatively traverse the
  // candidate's parent and child adjacency.
  //
  // This wrapper is used only by unit tests.
  GpuInsertionResult
  unittest_gpu_insert_block_direct_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion,
    uint32_t threads_per_candidate,
    uint32_t bitmap_bits = 0u);

  GpuFinalizeResult
  unittest_gpu_levelize_and_finalize(
    const std::vector<uint32_t>& selected_state,
    bool is_top_down_levelization,
    uint32_t initial_u0);

  GpuDagConstructionResult
  unittest_gpu_construct_dag(
    const std::vector<uint32_t>& selected_state);

  // Run the production GPU DAG controller starting from the
  // GPU tree produced by _gpu_tree_construction().
  //
  // This wrapper exists only for unit tests. It does not add
  // branches, counters, or callbacks to the production
  // controller.
  //
  // A fresh XEG object must be used so _dvc_picked_endes
  // still contains the unmodified GPU tree when DAG
  // construction begins.
  GpuDagConstructionResult
  unittest_gpu_construct_dag_from_tree_with_policy(
    GpuDagInsertionPolicy insertion_policy);

  GpuLevelizationResult unittest_gpu_levelize(
    const std::vector<uint32_t>& selected_state,
    bool is_top_down);

  // Run fixed-mask GPU levelization with an explicitly selected
  // backend.
  //
  // This bypasses the production compile-time dispatcher and
  // exists only so one unit-test binary can compare the original
  // and hybrid implementations using the exact same input mask.
  GpuLevelizationResult
  unittest_gpu_levelize_with_backend(
    const std::vector<uint32_t>& selected_state,
    bool is_top_down,
    GpuLevelizationBackend backend);

  GpuRootToLeafFilterResult
  unittest_gpu_filter_root_to_leaf(
    const std::vector<uint32_t>& selected_state);

  /* run */
  double round_to(double value, double precision = 1.0) {
    return std::round(value / precision) * precision;
  }
  
  void run(
    const bool ILP_solver = false,
    const double ILP_time_limit = 360,
    DagKernel dag_kernel = DagKernel::Thread,
    uint32_t num_thread_dag_constrution = 1u,
    uint32_t dag_bitmap_bits = 0u,
    IlpBackend ilp_backend = IlpBackend::CpSat,
    CuoptMode cuopt_mode = CuoptMode::Presolve)
  {
    _run_impl(
      ILP_solver,
      ILP_time_limit,
      dag_kernel,
      num_thread_dag_constrution,
      dag_bitmap_bits,
      CpsatInitialSolutionMode::GpuTree,
      0u,
      ilp_backend,
      cuopt_mode);
  }

  // CPU-only construction and CP-SAT. A zero time limit skips the solver.
  // Thread count controls construction only; solver_workers is independent.
  nlohmann::json run_cpu(int threads = 1, double time_limit = 360.0,
                         int solver_workers = 8, double deterministic_limit = 0.0);

  // GPU construction with checked candidates and the same CP-SAT/fallback
  // policy as run_cpu. A zero time limit runs construction and checks only.
  nlohmann::json run_gpu(double time_limit = 360.0, DagKernel kernel = DagKernel::Thread,
                         uint32_t threads = 1, uint32_t bitmap_bits = 0,
                         int solver_workers = 8, double deterministic_limit = 0.0,
                         const xeg::GpuDagOptions& options = {});

  // Run XEG-D on the complete post-read e-graph.
  //
  // CPU tree construction provides the CP-SAT warm start.
  // This path does not perform CPU or GPU DAG construction.
  void run_xeg_d(
    const double ILP_time_limit = 86400.0);

  void run_xeg_t_exp(
    const double ILP_time_limit = 360,
    DagKernel dag_kernel = DagKernel::Thread,
    uint32_t num_thread_dag_constrution = 1u,
    uint32_t dag_bitmap_bits = 0u,
    uint32_t random_seed = 42u,
    IlpBackend ilp_backend = IlpBackend::CpSat)
  {
    _run_impl(
      true,
      ILP_time_limit,
      dag_kernel,
      num_thread_dag_constrution,
      dag_bitmap_bits,
      CpsatInitialSolutionMode::Random,
      random_seed,
      ilp_backend,
      CuoptMode::Presolve);
  }

private:
  void _gpu_free_buffers() noexcept;
  void _gpu_allocate_graph_buffers();
  cudaError_t _gpu_copy_control(void* destination, const void* source, size_t bytes);
  void* _gpu_control = nullptr;
  int _gpu_num_levels = 0;
  xeg::GpuDagOptions _gpu_options;
  nlohmann::json _run_gpu_pipeline(
    double time_limit, DagKernel dag_kernel, uint32_t num_thread_dag_constrution,
    uint32_t dag_bitmap_bits, CpsatInitialSolutionMode initial_solution_mode,
    uint32_t random_seed, IlpBackend ilp_backend, CuoptMode cuopt_mode,
    int solver_workers, double deterministic_limit, const xeg::GpuDagOptions& options = {});

  double _gpu_tree_ms = 0, _gpu_dag_ms = 0;
  xeg::CpuGraph _cpu_graph() const;
  nlohmann::json _solve_checked_candidates(
    double time_limit, const std::vector<uint8_t>& candidates,
    const std::vector<uint8_t>& seed, const std::vector<uint32_t>& hint_best,
    const std::vector<bool>& hint_selected, int solver_workers,
    double deterministic_limit, bool acyclic);

  int _solver_workers = 0;
  double _solver_deterministic_limit = 0.0;
  bool _solver_omit_level_constraints = false;
  bool _solver_validate = false;
  nlohmann::json _solver_result;
  /* Read */
  void _read_input(const std::string input_path);
  // Output-degree ablation only.
  //
  // Randomly adds e-node -> strict-leaf-e-class child
  // references until the requested total edge multiplier is
  // reached or 100ly adds e-node -> strict-leaf-e-class child
  // references until the requested total consecutive random attempts fail.
  void _augment_output_degree_for_exp(
    uint32_t output_degree_multiplier,
    uint32_t random_seed);

  // Rebuild only the graph data derived from _ende_child and
  // _ende_child_ptr after the ablation modifies those arrays.
  //
  // The production _prune_dead_nodes() path remains unchanged.
  void _rebuild_derived_graph_data_for_output_degree_exp();  
  void _prune_dead_nodes(
    const uint32_t tmp_num_ecls,
    const uint32_t tmp_num_ende,
    const std::vector<std::string>& tmp_ende_name, 
    const std::vector<std::string>& tmp_ende_op, 
    const std::vector<float>& tmp_ende_cost, 
    const std::vector<uint32_t>& tmp_ende_ecls,
    const std::vector<uint32_t>& tmp_ende_child, 
    const std::vector<uint32_t>& tmp_ende_child_ptr, 
    const std::vector<uint32_t>& tmp_root_eclass_id);
  void _dump_to_egg_json(
    const std::string& out_path,
    uint32_t num_ecls,
    uint32_t num_ende,
    const std::vector<std::string>& ende_name,
    const std::vector<std::string>& ende_op,
    const std::vector<float>& ende_cost,
    const std::vector<uint32_t>& ende_ecls,
    const std::vector<uint32_t>& ende_child,
    const std::vector<uint32_t>& ende_child_ptr,
    const std::vector<uint32_t>& root_eclass_id);
  /* ASPLOS rebuttal starts */
  // skewness
  double _compute_skewness(const std::vector<uint32_t>& degrees);
  void _print_degree_stats();
  /* ASPLOS rebuttal ends */

  /* Run */
  void _run_impl(
    bool ILP_solver,
    double ILP_time_limit,
    DagKernel dag_kernel,
    uint32_t num_thread_dag_constrution,
    uint32_t dag_bitmap_bits,
    CpsatInitialSolutionMode initial_solution_mode,
    uint32_t random_seed,
    IlpBackend ilp_backend,
    CuoptMode cuopt_mode);

  void _build_random_cpsat_initial_solution(
    uint32_t random_seed,
    const std::vector<bool>& picked_endes_gpu,
    std::vector<uint32_t>& random_best_ende_of_ecls,
    std::vector<bool>& random_picked_endes) const;

  /* Tree Level Relaxation */
  // CPU version
  std::pair<double, double> _cpu_tree_construction(
    const std::vector<float>& tmp_ende_cost,
    std::vector<double>& cost_of_ecls,
    std::vector<double>& cost_of_ende,
    std::vector<uint32_t>& best_ende_of_ecls,
    std::vector<bool>& picked_endes); 
  // GPU version
  std::pair<double, double> _gpu_tree_construction();

  /* DAG Construction */
  // CPU version 
  void _cpu_DAG_construction(std::vector<bool>& picked_endes);
  // Validate the runtime Warp configuration.
  //
  // One candidate always uses exactly one 32-lane warp.
  // This parameter controls the total number of CUDA threads
  // per block, and therefore the number of independent
  // candidate warps placed in one block.
  //
  // Valid values are positive multiples of 32, subject to the
  // current CUDA device's maxThreadsPerBlock limit.
  void _validate_warp_threads_per_block(
    uint32_t threads_per_block) const;

  // Validate the runtime Warp bitmap size.
  //
  // 0:
  //   Use the existing exact Warp implementation.
  //
  // 32:
  //   Use the existing register-based Warp Bitmap-32
  //   implementation.
  //
  // 64:
  //   Use the existing register-based Warp Bitmap-64
  //   implementation.
  //
  // 128, 256, 512, 1024, 2048, 4096:
  //   Use one runtime-sized shared-memory bitmap per warp.
  //
  // Every nonzero bitmap size must be a power of two.
  void _validate_warp_bitmap_bits(
    uint32_t bitmap_bits) const;

  // Validate the runtime Block configuration.
  //
  // The initial shared-memory reduction supports:
  //
  //   64, 128, 256, 512, 1024
  //
  // subject to the current CUDA device limit.
  void _validate_block_threads_per_candidate(
    uint32_t threads_per_candidate) const;
  // Validate the runtime Block bitmap size.
  //
  // 0:
  //   Disable bitmap and use the existing exact Block kernels.
  //
  // Nonzero:
  //   Must be a power of two in [32, 4096].
  //
  // Valid values:
  //
  //   0, 32, 64, 128, 256, 512, 1024, 2048, 4096
  void _validate_block_bitmap_bits(
    uint32_t bitmap_bits) const;    
  // GPU DAG construction controller.
  //
  // FullBatch:
  //   Use batch insertion until u0 == 0.
  //
  // BatchEarlyStop:
  //   Stop when the batch low-activity condition is triggered.
  //
  // BatchThenDirect:
  //   Switch from batch insertion to direct insertion when the
  //   low-activity condition is triggered, then continue until
  //   u0 == 0.
  void _gpu_DAG_construction(
    GpuDagInsertionPolicy insertion_policy);
  // Single-block transposed-batch live optimistic insertion.
  //
  // One call scans all e-node slots inside one single-block
  // kernel. Exact Thread insertion processes transposed batches
  // sequentially using the configured exact batch size.
  //
  // Candidate IDs are assigned using a deterministic
  // full-transpose mapping so neighboring IDs are spread
  // across different batches.
  //
  // Tentative state-2 nodes published by earlier work may be
  // observed by later work in the same insertion phase.
  //
  // Levelization and finalization occur only after the entire
  // insertion kernel completes.
  //
  // is_top_down_insertion = true:
  //   require at least one selected parent
  //   proposed level = max parent level + 1
  //
  // is_top_down_insertion = false:
  //   require at least one selected child
  //   proposed level = min child level - 1
  void _gpu_dag_insert_batch(
    bool is_top_down_insertion);

  // One-thread-per-e-node direct live optimistic insertion.
  //
  // One CUDA thread evaluates exactly one e-node:
  //
  //   u = blockIdx.x * blockDim.x + threadIdx.x
  //
  // The candidate is published directly as tentative state 2.
  // Existing levelization and finalization validate or roll back
  // the insertion.
  void _gpu_dag_insert_direct(
    bool is_top_down_insertion);

  // Warp-cooperative transposed-batch insertion.
  //
  // One 32-lane warp acts as one transposed candidate lane.
  // Multiple independent candidate warps may share one CUDA
  // block.
  void _gpu_dag_insert_warp_batch(
    bool is_top_down_insertion,
    uint32_t threads_per_block);

  // Warp-cooperative direct insertion.
  //
  // One 32-lane warp evaluates exactly one candidate e-node.
  // Multiple independent candidate warps may share one CUDA
  // block.
  void _gpu_dag_insert_warp_direct(
    bool is_top_down_insertion,
    uint32_t threads_per_block);

  // Warp-cooperative transposed-batch insertion using one
  // fixed 32-bit register bitmap per warp.
  //
  // All nonzero Warp bitmap implementations use the same
  // Strategy-A1 hash and probe policy as the Block bitmap
  // implementation:
  //
  //   1. mix the e-node ID
  //   2. set one bit in the lower half
  //   3. set one bit in the upper half
  //
  // A selected child is treated as an approximate overlap only
  // when both Strategy-A1 probe bits are present.
  //
  // Bitmap-32 stores both probes in one uint32_t register.
  void _gpu_dag_insert_warp_bitmap32_batch(
    bool is_top_down_insertion,
    uint32_t threads_per_block);

  // Warp-cooperative direct insertion using the same fixed
  // 32-bit register bitmap and Strategy-A1 probe policy.
  void _gpu_dag_insert_warp_bitmap32_direct(
    bool is_top_down_insertion,
    uint32_t threads_per_block);

  // Warp-cooperative transposed-batch insertion using one
  // fixed 64-bit register bitmap represented by two uint32_t
  // words per warp.
  //
  // The lower Strategy-A1 probe is stored in the first register
  // word. The upper Strategy-A1 probe is stored in the second
  // register word.
  void _gpu_dag_insert_warp_bitmap64_batch(
    bool is_top_down_insertion,
    uint32_t threads_per_block);

  // Warp-cooperative direct insertion using the same fixed
  // 64-bit register bitmap and Strategy-A1 probe policy.
  void _gpu_dag_insert_warp_bitmap64_direct(
    bool is_top_down_insertion,
    uint32_t threads_per_block);

  // Warp-cooperative transposed-batch insertion using one
  // runtime-sized shared-memory bitmap per warp.
  //
  // This path is used only for:
  //
  //   128, 256, 512, 1024, 4096 bits
  //
  // Every nonzero Warp bitmap size uses the same Strategy-A1
  // hash and two-probe policy as the Block bitmap
  // implementation.
  //
  // Warp Bitmap-32 and Bitmap-64 remain register-based.
  // Warp Bitmap-128...4096 use one disjoint dynamic
  // shared-memory bitmap segment per warp.
  void _gpu_dag_insert_warp_bitmap_shared_batch(
    bool is_top_down_insertion,
    uint32_t threads_per_block,
    uint32_t bitmap_mask,
    uint32_t bitmap_words);

  // Warp-cooperative direct insertion using one runtime-sized
  // shared-memory bitmap per warp and the common Strategy-A1
  // hash and two-probe policy.
  //
  // One warp evaluates one candidate e-node. Multiple candidate
  // warps may share one CUDA block, but every warp owns a
  // disjoint shared-memory bitmap segment.
  void _gpu_dag_insert_warp_bitmap_shared_direct(
    bool is_top_down_insertion,
    uint32_t threads_per_block,
    uint32_t bitmap_mask,
    uint32_t bitmap_words);

  // Block-cooperative transposed-batch insertion.
  //
  // One CUDA block acts as one candidate lane. Each block
  // sequentially processes its transposed candidate sequence.
  void _gpu_dag_insert_block_batch(
    bool is_top_down_insertion,
    uint32_t threads_per_candidate);

  // Block-cooperative direct insertion.
  //
  // One CUDA block evaluates exactly one candidate e-node.
  void _gpu_dag_insert_block_direct(
    bool is_top_down_insertion,
    uint32_t threads_per_candidate);
  // Block-cooperative transposed-batch insertion using the
  // runtime-sized shared-memory Strategy-A1 bitmap.
  //
  // Strategy A1 first mixes each selected, level-valid parent
  // e-node ID. It then sets two bitmap bits:
  //
  //   one bit in the lower half of the bitmap
  //   one bit in the upper half of the bitmap
  //
  // A selected, level-valid child is treated as an approximate
  // overlap only when both of its corresponding bits are set.
  //
  // A two-bit hit may still be caused by hash collisions and is
  // conservatively rejected without exact verification.
  //
  // The existing exact Block-Batch implementation remains
  // unchanged and is selected when BITMAP_BITS == 0.
  void _gpu_dag_insert_block_bitmap_batch(
    bool is_top_down_insertion,
    uint32_t threads_per_candidate,
    uint32_t bitmap_mask,
    uint32_t bitmap_words);

  // Block-cooperative direct insertion using the same
  // runtime-sized shared-memory Strategy-A1 bitmap as the
  // Block-Batch implementation.
  //
  // Only selected, level-valid parents set their two A1 bits.
  // Only selected, level-valid children probe those two bits.
  //
  // One CUDA block evaluates one candidate.
  void _gpu_dag_insert_block_bitmap_direct(
    bool is_top_down_insertion,
    uint32_t threads_per_candidate,
    uint32_t bitmap_mask,
    uint32_t bitmap_words);

  // Select Thread, Warp, or Block insertion while preserving
  // the controller's independent Batch/Direct schedule.
  void _gpu_dag_insert_configured(
    bool use_direct_mode,
    bool is_top_down_insertion);

  // Finalize tentative state-2 nodes after levelization.
  // Finalize tentative state-2 nodes after levelization.
  //
  // residual degree == 0:
  //   state 2 -> state 1
  //
  // residual degree > 0:
  //   state 2 -> state 0
  //   level = -1
  //   u0--
  void _gpu_dag_finalize_tentative(
    bool is_top_down_levelization);    
  // GPU Kahn levelization using one append-only queue.
  // is_top_down = true:
  //   roots -> leaves
  // is_top_down = false:
  //   leaves -> roots, followed by level reversal
  void _gpu_dag_levelize(bool is_top_down);
  // Original host-controlled implementation.
  //
  // Every frontier is processed by one multi-block kernel launch,
  // followed by one device-to-host queue-size copy.
  void _gpu_dag_levelize_host_driven(
    const bool is_top_down);

  // Hybrid implementation.
  //
  // Early or wide frontiers remain host-driven and multi-block.
  // After a stable sequence of small frontiers, the remaining
  // levels are processed by one single-block persistent kernel.
  void _gpu_dag_levelize_hybrid(
    const bool is_top_down);

  // Shared implementation for the two public Warp insertion
  // unit-test wrappers.
  GpuInsertionResult
  _unittest_gpu_insert_warp_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion,
    bool use_direct_mode,
    uint32_t threads_per_block,
    uint32_t bitmap_bits);

  // Shared implementation for the two public Block insertion
  // unit-test wrappers.
  GpuInsertionResult
  _unittest_gpu_insert_block_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion,
    bool use_direct_mode,
    uint32_t threads_per_candidate,
    uint32_t bitmap_bits);

  // Shared implementation used only by the public
  // levelization unit-test wrappers.
  //
  // All additional backend selection, sentinel state, and
  // result inspection remain outside the production execution
  // path.
  GpuLevelizationResult
  _unittest_gpu_levelize_impl(
    const std::vector<uint32_t>& selected_state,
    bool is_top_down,
    GpuLevelizationBackend backend);

  // GPU DAG memory lifecycle
  void _gpu_allocate_dag_buffers();
  void _gpu_initialize_dag_buffers();
  void _gpu_free_dag_buffers() noexcept;

  // Device-to-host copy helpers
  void _copy_pickedende_back(
    std::vector<bool>& picked_endes_out);

  void _copy_best_ende_of_ecls_back(
    std::vector<uint32_t>& best_ende_of_ecls_out);

  // Mark selected e-nodes reachable from a set of starts.
  //
  // starts and adjacency use the same pruned e-node IDs as
  // _dvc_picked_endes and the flattened GPU CSR.
  void _gpu_mark_reachable(
    const uint32_t* starts,
    uint32_t num_starts,
    const uint32_t* adjacency,
    const uint32_t* adjacency_ptr,
    uint32_t* visited);

  // Compute:
  //
  // in_subgraph[u] =
  //   selected[u] && from_roots[u] && from_leaves[u]
  void _gpu_build_in_subgraph();

  // Match the CPU root-to-leaf filtering semantics without
  // modifying _dvc_picked_endes.
  void _gpu_filter_root_to_leaf();

  /* ILP solver */ 
  void _solve_ilp(
    IlpBackend backend,
    double time_limit,
    const std::vector<std::vector<uint32_t>>& ecls_its_ende,
    const std::vector<uint32_t>& ende_child,
    const std::vector<uint32_t>& ende_child_ptr,
    const std::vector<uint32_t>& root_eclass_id,
    const std::vector<float>& ende_cost,
    const std::vector<uint32_t>& best_ende_of_ecls,
    const std::vector<bool>& picked_endes,
    CuoptMode cuopt_mode);

  void _cpsat_solver(
    const double time_limit,
    const std::vector<std::vector<uint32_t>>& ecls_its_ende, 
    const std::vector<uint32_t>& ende_child,
    const std::vector<uint32_t>& ende_child_ptr,
    const std::vector<uint32_t>& root_eclass_id,
    const std::vector<float>& ende_cost,
    const std::vector<uint32_t>& best_ende_of_ecls,
    const std::vector<bool>& picked_endes);

  /* Private members */
  // Read in from .json, and prune the dead eclasses/enodes
  uint32_t _num_ecls = 0; // nuber of eclasses
  uint32_t _num_ende = 0; // number of enodes
  std::vector<std::string> _ende_name; 
  std::vector<std::string> _ende_op; 
  std::vector<float> _ende_cost; 
  std::vector<uint32_t> _ende_ecls; // the eclass that enode belonsg to; 
  std::vector<uint32_t> _ende_child; // enode point to its child eclass
  std::vector<uint32_t> _ende_child_ptr; // enode point to its child eclass
  std::vector<uint32_t> _root_eclass_id;

  std::vector<std::vector<uint32_t>> _ecls_its_ende; // enodes under that eclasses
  std::vector<uint32_t> _ecls_its_ende_1D; // compress _ecls_its_ende into 1D
  std::vector<uint32_t> _ecls_its_ende_1D_ptr;
  std::vector<uint32_t> _all_root_ende_1D; // store all of the enodes in 1D

  // For construction of new graph G 
  std::vector<std::unordered_set<uint32_t>> _global_adj;
  std::vector<std::unordered_set<uint32_t>> _global_inv_adj;
  std::vector<uint32_t> _global_adj_1D;
  std::vector<uint32_t> _global_adj_1D_ptr;
  std::vector<uint32_t> _global_inv_adj_1D;
  std::vector<uint32_t> _global_inv_adj_1D_ptr;
  std::vector<bool> _picked_adj; // for picked enode: set as true, pointer based on _global_adj_1D_ptr
  std::vector<bool> _picked_inv_adj; // for picked enode: set as true, pointer based on _global_inv_adj_1D_ptr
  std::vector<bool> _tmp_picked_adj;
  std::vector<bool> _tmp_picked_inv_adj;
    
  // For new graph G
  std::vector<uint32_t> _indegree;
  std::vector<uint32_t> _inv_indegree;
  int _max_level = -1;  // init as -1
  std::vector<bool> _in_subgraph;
  std::vector<uint32_t> _roots;
  std::vector<uint32_t> _leaves;
  std::vector<std::vector<uint32_t>> _ecls_its_ende_newG;
  std::vector<std::vector<uint32_t>> _ecls_its_ende_newG_GPU;

  
  // READ IN argv/argc
  DagKernel _dag_kernel = DagKernel::Thread;

  // Runtime GPU DAG execution width.
  //
  // Thread:
  //   must be 1; one thread evaluates one candidate.
  //
  // Warp:
  //   total CUDA threads per block; each candidate still uses
  //   exactly one 32-lane warp.
  //
  // Block:
  //   number of CUDA threads cooperatively evaluating one
  //   candidate.
  //
  // The existing member name is preserved to avoid an unrelated
  // interface refactor.
  uint32_t _dag_threads_per_candidate = 1;

  // Runtime GPU DAG bitmap configuration.
  //
  // Thread:
  //   0 only.
  //
  // Warp:
  //   0
  //     -> Exact Warp implementation without hashing
  //
  //   32, 64
  //     -> register-based Warp Bitmap implementations
  //
  //   128, 256, 512, 1024, 2048, 4096
  //     -> one runtime-sized shared-memory bitmap per warp
  //
  //   Every nonzero Warp bitmap size uses the same Strategy-A1
  //   mixed two-probe hash policy as the Block bitmap
  //   implementation.
  // Block:
  //   0    -> existing Exact Block implementation
  //   32..4096 power-of-two
  //        -> shared-memory Bitmap Block implementation
  //
  // A nonzero bitmap performs approximate overlap detection
  // between:
  //
  //   selected, level-valid parents
  //   selected, level-valid children
  //
  // A hash collision may conservatively reject a candidate.
  uint32_t _dag_bitmap_bits = 0u;

  // Valid only when _dag_bitmap_bits > 0.
  //
  // Warp Bitmap-32/64 implementations use their fixed masks
  // internally:
  //
  //   32 bits -> 31
  //   64 bits -> 63
  //
  // Warp Shared-Bitmap-128...4096 and Block Bitmap
  // implementations receive this runtime mask.
  //
  // Every nonzero bitmap implementation passes the applicable
  // mask to the common Strategy-A1 probe generator.
  //
  // Examples:
  //
  //   32 bits   -> mask 31
  //   64 bits   -> mask 63
  //   4096 bits -> mask 4095
  uint32_t _dag_bitmap_mask = 0u;

  // Number of uint32_t words:
  //
  //   words = bits / 32
  //
  // Warp Bitmap-32/64 fixed-size register kernels do not require
  // this runtime word count because their register layout is
  // known at compile time.
  //
  // Warp Shared-Bitmap-128...4096 and Block Bitmap kernels use
  // this value to size dynamic shared memory.
  uint32_t _dag_bitmap_words = 0u;

  // Print function
  void _print_input() const;
  void _print_adj(const std::vector<std::unordered_set<uint32_t>>& adj) const;
  void _print_read_in_data(const bool init,
                          const uint32_t tmp_num_ecls,
                          const uint32_t tmp_num_ende,
                          const std::vector<std::string>& tmp_ende_name, 
                          const std::vector<std::string>& tmp_ende_op, 
                          const std::vector<float>& tmp_ende_cost, 
                          const std::unordered_map<std::string, uint32_t>& tmp_ecls_name_to_id,
                          const std::vector<uint32_t>& tmp_ende_ecls,
                          const std::vector<uint32_t>& tmp_ende_child, 
                          const std::vector<uint32_t>& tmp_ende_child_ptr, 
                          const std::vector<uint32_t>& tmp_root_eclass_id) const;
  void _print_degree(const bool is_td, const std::vector<bool>& picked_endes) const;
  void _print_levelization(const bool is_td, const std::vector<int>& level,
                          const std::vector<bool>& picked_endes);

  // 0515 revision
  std::vector<bool> _new_inserted; 
  // ASPLOS rebuttal Q2: wasted work counters
  std::atomic<uint64_t> _total_insert_count{0};
  std::atomic<uint64_t> _total_revert_count{0};  
  std::atomic<uint64_t> _total_bitmap_fp_count{0};
  std::atomic<uint64_t> _total_overlap_check_count{0};  
  bool _insertion2(const bool is_td, 
    std::vector<bool>& picked_endes,
    std::vector<int>& level);
  void _BFS2(
    const bool is_td, 
    std::vector<uint32_t>& deg, 
    const std::vector<bool>& picked_endes,
    const std::vector<uint32_t>& global_adj_1D,
    const std::vector<uint32_t>& global_adj_1D_ptr,
    const std::vector<bool>& picked_adj,
    std::vector<int>& level);

  // ============================================================
  // Existing GPU tree/shared memory
  //
  // These names and ownership are preserved because
  // _gpu_tree_construction() already uses them.
  // Do not rename or remove them during the DAG rewrite.
  // ============================================================
  uint32_t* _dvc_best_ende_of_ecls = nullptr;
  uint32_t* _dvc_update = nullptr;
  uint32_t* _dvc_tree_locks = nullptr;

  uint32_t* _dvc_root_eclass_id = nullptr;

  uint32_t* _dvc_picked_endes = nullptr;

  uint32_t* _dvc_q = nullptr;
  uint32_t* _dvc_q_sz = nullptr;

  // These buffers are allocated and initialized by
  // _gpu_tree_construction(), then reused by GPU DAG construction.
  uint32_t* _dvc_global_adj_1D = nullptr;
  uint32_t* _dvc_global_adj_1D_ptr = nullptr;
  uint32_t* _dvc_global_inv_adj_1D = nullptr;
  uint32_t* _dvc_global_inv_adj_1D_ptr = nullptr;

  uint32_t* _dvc_indegree = nullptr;
  uint32_t* _dvc_inv_indegree = nullptr;


  int* _dvc_level = nullptr;
  int* _dvc_max_level = nullptr;

  uint32_t* _dvc_in_subgraph = nullptr;
  uint32_t* _dvc_roots = nullptr;
  uint32_t* _dvc_leaves = nullptr;
  uint32_t* _dvc_from_roots = nullptr;
  uint32_t* _dvc_from_leaves = nullptr;

  // ============================================================
  // New GPU DAG-only scratch memory
  //
  // These are the only buffers managed by
  // _gpu_allocate_dag_buffers() / _gpu_free_dag_buffers().
  // ============================================================
  uint32_t* _dvc_dag_new_inserted = nullptr;

  uint32_t* _dvc_dag_insert_decision = nullptr;
  int* _dvc_dag_candidate_level = nullptr;
  uint32_t* _dvc_dag_u0 = nullptr;
  xeg_detail::GpuDagPersistentState*
    _dvc_dag_persistent_state = nullptr;  
}; // class XEG


#endif

