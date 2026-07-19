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

  struct GpuLevelizationResult {
    std::vector<uint32_t> selected_state;
    std::vector<uint32_t> residual_degree;
    std::vector<int> level;
    int num_levels = 0;
  };

  CsrGraphSnapshot
  unittest_get_csr_snapshot() const;
  
  GpuInsertionResult unittest_gpu_insert_once(
    const std::vector<uint32_t>& selected_state,
    const std::vector<int>& level,
    bool is_top_down_insertion);

  GpuLevelizationResult unittest_gpu_levelize(
    const std::vector<uint32_t>& selected_state,
    bool is_top_down);

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
        throw std::logic_error(
          "Block-level GPU DAG construction is not implemented yet");

      default:
        throw std::invalid_argument("Invalid DagKernel value");
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
    std::cout << "CPU DAG construction runtime: " << 
      round_to((duration.count()) * 1000, 0.01) << " (ms)\n";
    if (ILP_solver) {
      _cpsat_solver(
        ILP_time_limit, 
        _ecls_its_ende_newG, 
        _ende_child, 
        _ende_child_ptr, 
        _root_eclass_id, 
        _ende_cost,
        best_ende_of_ecls, 
        picked_endes_tmp); 
    }

    // GPU version starts
    auto [best_tree_cost_gpu, dag_cost_gpu] =
      _gpu_tree_construction();

    printf(
      "GPU version cost:\n"
      "\tunder tree cost model = %.2lf,\n"
      "\tunder DAG cost model = %.2lf\n",
      best_tree_cost_gpu,
      dag_cost_gpu);

    // Preserve GPU tree result independently from the CPU oracle.
    std::vector<bool> gpu_tree_selected;
    std::vector<uint32_t> gpu_best_ende_of_ecls;

    _copy_pickedende_back(gpu_tree_selected);
    _copy_best_ende_of_ecls_back(gpu_best_ende_of_ecls);

    // Allocate and initialize only DAG-construction memory.
    _gpu_allocate_dag_buffers();
    _gpu_initialize_dag_buffers();

    // Currently this is allowed to be a no-op stub.
    _gpu_DAG_construction();

    std::vector<bool> gpu_dag_selected;
    _copy_pickedende_back(gpu_dag_selected);

    // 之後為了效能要刪掉
    if (gpu_tree_selected != gpu_dag_selected) {
      throw std::logic_error(
        "No-op GPU DAG construction modified the GPU tree solution");
    }

    // Only free memory owned exclusively by the new DAG implementation.
    _gpu_free_dag_buffers();
    // GPU version ends    

    // if (ILP_solver) {
    //   _cpsat_solver(
    //     ILP_time_limit, 
    //     _ecls_its_ende_newG_GPU, 
    //     _ende_child, 
    //     _ende_child_ptr, 
    //     _root_eclass_id, 
    //     _ende_cost,
    //     best_ende_of_ecls, 
    //     picked_endes_tmp);
    // }
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
  // GPU version
  void _gpu_DAG_construction();
  // Thread-level live optimistic insertion.
  //
  // is_top_down_insertion = true:
  //   require at least one selected parent
  //   proposed level = max parent level + 1
  //
  // is_top_down_insertion = false:
  //   require at least one selected child
  //   proposed level = min child level - 1
  void _gpu_dag_insert_thread(
    bool is_top_down_insertion);  
  // GPU Kahn levelization using one append-only queue.
  // is_top_down = true:
  //   roots -> leaves
  // is_top_down = false:
  //   leaves -> roots, followed by level reversal
  void _gpu_dag_levelize(bool is_top_down);

  // GPU DAG memory lifecycle
  void _gpu_allocate_dag_buffers();
  void _gpu_initialize_dag_buffers();
  void _gpu_free_dag_buffers() noexcept;

  // Device-to-host copy helpers
  void _copy_pickedende_back(
    std::vector<bool>& picked_endes_out);

  void _copy_best_ende_of_ecls_back(
    std::vector<uint32_t>& best_ende_of_ecls_out);

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
}; // class G_EXT


#endif




