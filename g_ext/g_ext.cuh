#ifndef EXT_GPU_H
#define EXT_GPU_H
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

/* ILP Solver */
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/base/logging.h" // For LOG(INFO)

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

enum class DagKernel : uint32_t {
  Thread = 0,
  Warp   = 1,
  Block  = 2,
};

namespace g_ext_detail {

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

}  // namespace g_ext_detail

class G_EXT {

public:
  // Read input file
  void read(const std::string input_path) {
    _read_input(input_path); 
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
    uint32_t threads_per_candidate);

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
    uint32_t threads_per_candidate);

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
  // A fresh G_EXT object must be used so _dvc_picked_endes
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
  
  void run(const bool ILP_solver = false, 
          const double ILP_time_limit = 360,
          DagKernel dag_kernel = DagKernel::Thread,
          uint32_t num_thread_dag_constrution = 1) 
  {
    switch (dag_kernel) {
      case DagKernel::Thread:
        if (num_thread_dag_constrution != 1) {
          throw std::invalid_argument(
            "Thread DAG kernel requires exactly one thread per candidate");
        }
        break;

      case DagKernel::Warp:
        throw std::logic_error(
          "Warp-level GPU DAG construction is not implemented yet");

      case DagKernel::Block:
        _validate_block_threads_per_candidate(
          num_thread_dag_constrution);
        break;

      default:
        throw std::invalid_argument(
          "Invalid DagKernel value");
    }

    _dag_kernel = dag_kernel;
    _dag_threads_per_candidate = num_thread_dag_constrution;

    // CPU version
    std::vector<double> cost_of_ecls; // Final cost of each eclass
    std::vector<double> cost_of_ende; // Final cost of each enode
    std::vector<uint32_t> best_ende_of_ecls; // enode idx with smallest cost in that eclass     
    std::vector<bool> picked_endes;
    std::chrono::duration<double> duration;
    auto start = std::chrono::steady_clock::now(); 
      auto [best_tree_cost_cpu, dag_cost_cpu] = _cpu_tree_construction(
        _ende_cost, 
        cost_of_ecls, 
        cost_of_ende, 
        best_ende_of_ecls, 
        picked_endes);
    auto end = std::chrono::steady_clock::now();
    duration = end - start;
    std::cout << "CPU tree construction runtime: " << 
      round_to((duration.count()) * 1000, 0.01) << " (ms)\n";
    printf("CPU version cost:\n\tunder tree code model = %.2lf, \
      \n\tunder DAG cost model = %.2lf\n", 
      best_tree_cost_cpu, dag_cost_cpu);

    std::vector<bool> picked_endes_tmp = picked_endes;

    start = std::chrono::steady_clock::now();

    _cpu_DAG_construction(picked_endes);
    // _cpu_DAG_construction_mt_bitmap(picked_endes);

    end = std::chrono::steady_clock::now();
    duration = end - start;

    std::cout
      << "CPU DAG construction runtime: "
      << round_to(
           duration.count() * 1000,
           0.01)
      << " (ms)\n";

    // Debug statistics are outside the measured runtime.
    uint32_t cpu_tree_selected = 0u;
    uint32_t cpu_dag_selected = 0u;
    uint32_t cpu_filtered = 0u;

    for (bool selected : picked_endes_tmp) {
      cpu_tree_selected += selected ? 1u : 0u;
    }

    for (bool selected : picked_endes) {
      cpu_dag_selected += selected ? 1u : 0u;
    }

    for (bool selected : _in_subgraph) {
      cpu_filtered += selected ? 1u : 0u;
    }

    printf(
      "[CPU DAG Debug] tree=%u, expanded=%u, filtered=%u\n",
      cpu_tree_selected,
      cpu_dag_selected,
      cpu_filtered);    

    // if (ILP_solver) {
    //   _cpsat_solver(
    //     ILP_time_limit, 
    //     _ecls_its_ende_newG, 
    //     _ende_child, 
    //     _ende_child_ptr, 
    //     _root_eclass_id, 
    //     _ende_cost,
    //     best_ende_of_ecls, 
    //     picked_endes_tmp); 
    // }

    // GPU version starts
    auto [best_tree_cost_gpu, dag_cost_gpu] =
      _gpu_tree_construction();

    printf(
      "GPU version cost:\n"
      "\tunder tree cost model = %.2lf,\n"
      "\tunder DAG cost model = %.2lf\n",
      best_tree_cost_gpu,
      dag_cost_gpu);

    // Save the GPU tree solution before DAG construction
    // expands _dvc_picked_endes.
    std::vector<uint32_t> best_ende_of_ecls_gpu;
    std::vector<bool> picked_endes_gpu;

    if (ILP_solver) {
      _copy_best_ende_of_ecls_back(
        best_ende_of_ecls_gpu);

      _copy_pickedende_back(
        picked_endes_gpu);

      uint32_t gpu_tree_selected = 0u;

      for (bool selected : picked_endes_gpu) {
        gpu_tree_selected += selected ? 1u : 0u;
      }

      printf(
        "[GPU DAG Debug] tree=%u\n",
        gpu_tree_selected);
    }


    _gpu_allocate_dag_buffers();
    _gpu_initialize_dag_buffers();

    // GPU DAG insertion policy.
    //
    // FullBatch:
    //   Use batch insertion until full convergence.
    //
    // BatchEarlyStop:
    //   Use batch insertion and stop when the low-activity
    //   condition is triggered.
    //
    // BatchThenDirect:
    //   Start with batch insertion, then switch permanently
    //   to direct insertion when the low-activity condition
    //   is triggered.
    constexpr GpuDagInsertionPolicy
      gpu_dag_insertion_policy =
        GpuDagInsertionPolicy::BatchThenDirect;

    // _dvc_picked_endes still contains the GPU tree produced by
    // _gpu_tree_construction().
    _gpu_DAG_construction(
      gpu_dag_insertion_policy);

    if (ILP_solver) {
      // Copy the final GPU root-to-leaf filtering mask.
      std::vector<uint32_t> in_subgraph_gpu(
        _num_ende,
        0u);

      if (_num_ende > 0u) {
        const cudaError_t err = cudaMemcpy(
          in_subgraph_gpu.data(),
          _dvc_in_subgraph,
          static_cast<size_t>(_num_ende) *
            sizeof(uint32_t),
          cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
          _gpu_free_dag_buffers();

          throw std::runtime_error(
            std::string(
              "Failed to copy GPU in_subgraph: ") +
            cudaGetErrorString(err));
        }
      }

      // Copy the expanded GPU DAG selection for debugging.
      // This happens after the GPU DAG timer has stopped.
      std::vector<bool> gpu_dag_selected;
      _copy_pickedende_back(
        gpu_dag_selected);

      uint32_t gpu_dag_selected_count = 0u;
      uint32_t gpu_filtered_count = 0u;

      for (bool selected : gpu_dag_selected) {
        gpu_dag_selected_count +=
          selected ? 1u : 0u;
      }

      for (uint32_t selected : in_subgraph_gpu) {
        gpu_filtered_count +=
          selected != 0u ? 1u : 0u;
      }

      printf(
        "[GPU DAG Debug] expanded=%u, filtered=%u\n",
        gpu_dag_selected_count,
        gpu_filtered_count);

      // Reconstruct the GPU DAG search space for ILP.
      _ecls_its_ende_newG_GPU.assign(
        _num_ecls,
        std::vector<uint32_t>());

      for (uint32_t ecls = 0u;
           ecls < _num_ecls;
           ++ecls) {
        const auto& original_enodes =
          _ecls_its_ende[ecls];

        auto& gpu_dag_enodes =
          _ecls_its_ende_newG_GPU[ecls];

        gpu_dag_enodes.reserve(
          original_enodes.size());

        for (uint32_t ende : original_enodes) {
          if (in_subgraph_gpu[ende] != 0u) {
            gpu_dag_enodes.push_back(ende);
          }
        }
      }
    }


    // Free only GPU DAG scratch memory.
    _gpu_free_dag_buffers();

    if (ILP_solver) {
      _cpsat_solver(
        ILP_time_limit,
        _ecls_its_ende_newG_GPU,
        _ende_child,
        _ende_child_ptr,
        _root_eclass_id,
        _ende_cost,
        best_ende_of_ecls_gpu,
        picked_endes_gpu);
    }
    // // GPU version ends
  }

private:
  /* Read */
  void _read_input(const std::string input_path);
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
  bool _insertion_bitmap_mt(
    const bool is_td, 
    std::vector<bool>& picked_endes,
    std::vector<int>& level);

  void _cpu_DAG_construction_mt_bitmap(std::vector<bool>& picked_endes);
  /* ASPLOS rebuttal ends */

  /* Run */
  /* Tree Level Relaxation */
  // CPU version
  std::pair<double, double> _cpu_tree_construction(
    const std::vector<float>& tmp_ende_cost,
    std::vector<double>& cost_of_ecls,
    std::vector<double>& cost_of_ende,
    std::vector<uint32_t>& best_ende_of_ecls,
    std::vector<bool>& picked_endes); 
  double _compute_dag_cost(
    const std::vector<uint32_t>& best_ende_of_ecls,
    std::vector<bool>& picked_endes,
    std::vector<std::vector<uint32_t>>& ende_picked_by_ende);
  void _init_picked_adj(const std::vector<bool>& picked_endes);
  // GPU version
  std::pair<double, double> _gpu_tree_construction(); 

  /* DAG Construction */
  // CPU version 
  void _cpu_DAG_construction(std::vector<bool>& picked_endes);
  void _construct_deg(
    const std::vector<bool>& picked_endes,
    const std::vector<double>& cost_of_ende); 
  void _top_down_levelize(
    std::vector<bool>& picked_endes,
    std::vector<int>& level);
  void _bottom_up_levelize(
    std::vector<bool>& picked_endes,
    std::vector<int>& level);
    void _update_deg(
      const bool need_to_update, 
      const std::vector<bool>& picked_endes);
    void _BFS(
      const bool is_td, 
      std::vector<uint32_t>& deg, 
      std::vector<bool>& picked_endes, 
      const std::vector<uint32_t>& global_adj_1D,
      const std::vector<uint32_t>& global_adj_1D_ptr,
      const std::vector<bool>& picked_adj,
      std::vector<int>& level);
  bool _insertion(
    const bool is_td, 
    std::vector<bool>& picked_endes,
    std::vector<int>& level);
  void _update_picked_adj(
    const uint32_t fm_ende, 
    const uint32_t to_ende);
  void _clear_redundant_endes(std::vector<bool>& picked_endes);
    void _mark_reachable(
      const std::vector<uint32_t>& starts,
      const std::vector<uint32_t>& global_adj_1D,
      const std::vector<uint32_t>& global_adj_1D_ptr,
      const std::vector<bool>& picked_adj,      
      std::vector<bool>& visited,
      const std::vector<bool>& picked_endes);
    void get_nodes_in_subgraph(
      const std::vector<uint32_t>& roots,
      const std::vector<uint32_t>& leaves,
      const std::vector<bool>& picked_endes);

  // Validate the runtime Block configuration.
  //
  // The initial shared-memory reduction supports:
  //
  //   64, 128, 256, 512, 1024
  //
  // subject to the current CUDA device limit.
  void _validate_block_threads_per_candidate(
    uint32_t threads_per_candidate) const;
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
  // kernel. The kernel processes transposed batches
  // sequentially using kGpuDagBatchSize lanes.
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

  // Select Thread or Block insertion while preserving the
  // controller's independent Batch/Direct schedule.
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

  // Shared implementation for the two public Block insertion
  // unit-test wrappers.
  GpuInsertionResult
  _unittest_gpu_insert_block_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion,
    bool use_direct_mode,
    uint32_t threads_per_candidate);

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
  uint32_t _num_ecls; // nuber of eclasses 
  uint32_t _num_ende; // number of enodes
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
  uint32_t _dag_threads_per_candidate = 1;

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
  bool* _dvc_update = nullptr;

  uint32_t* _dvc_ecls_its_ende_1D = nullptr;
  uint32_t* _dvc_ecls_its_ende_1D_ptr = nullptr;
  uint32_t* _dvc_root_eclass_id = nullptr;
  uint32_t* _dvc_ende_picked_by_ende = nullptr;

  uint32_t* _dvc_picked_endes = nullptr;

  uint32_t* _dvc_q = nullptr;
  uint32_t* _dvc_q_sz = nullptr;
  uint32_t* _dvc_q_sp = nullptr;
  uint32_t* _dvc_q_num = nullptr;

  // These buffers are allocated and initialized by
  // _gpu_tree_construction(), then reused by GPU DAG construction.
  uint32_t* _dvc_global_adj_1D = nullptr;
  uint32_t* _dvc_global_adj_1D_ptr = nullptr;
  uint32_t* _dvc_global_inv_adj_1D = nullptr;
  uint32_t* _dvc_global_inv_adj_1D_ptr = nullptr;

  uint32_t* _dvc_indegree = nullptr;
  uint32_t* _dvc_inv_indegree = nullptr;

  uint32_t* _dvc_picked_adj = nullptr;
  uint32_t* _dvc_picked_inv_adj = nullptr;
  uint32_t* _dvc_tmp_picked_adj = nullptr;
  uint32_t* _dvc_tmp_picked_inv_adj = nullptr;

  int* _dvc_level = nullptr;
  uint32_t* _dvc_all_root_ende_1D = nullptr;
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
  g_ext_detail::GpuDagPersistentState*
    _dvc_dag_persistent_state = nullptr;  
}; // class G_EXT


#endif

