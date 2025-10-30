#ifndef EXT_GPU_H
#define EXT_GPU_H
#include <climits>
#include <fstream>
#include <iostream>
#include <string>
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


/* ILP Solver */
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/base/logging.h" // For LOG(INFO)

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define GPU_VERSION

class G_EXT {

public:
  // Read input file
  void read(const std::string input_path) {
    // Tempory memory 
    _read_input(input_path); 
  }

  /* run */
  double round_to(double value, double precision = 1.0) {
    return std::round(value / precision) * precision;
  }
  
  void run(const bool ILP_solver = false, 
          const double ILP_time_limit = 360
) {
    printf("Hello bibi\n");

    // CPU version
    std::vector<double> cost_of_ecls; // Final cost of each eclass
    std::vector<double> cost_of_ende; // Final cost of each enode
    std::vector<uint32_t> best_ende_of_ecls; // enode idx with smallest cost in that eclass     
    std::vector<bool> picked_endes;
    std::chrono::duration<double> duration;
    auto start = std::chrono::steady_clock::now();
      auto [best_tree_cost_cpu, dag_cost_cpu] = _cpu_best_tree(_ende_cost, cost_of_ecls, 
                                                              cost_of_ende, best_ende_of_ecls, 
                                                              picked_endes);
    auto end = std::chrono::steady_clock::now();
    duration = end - start;
    std::cout << "CPU Relaxation Runtime: " << round_to((duration.count()) * 1000, 0.01) << " (ms)\n";
    printf("CPU RESULTS: best_tree_cost_cpu = %.2lf, dag_cost_cpu = %.2lf\n", 
      best_tree_cost_cpu, dag_cost_cpu);
    
    // if (ILP_solver) {
    //   _cpsat_solver(ILP_time_limit, 
    //                 _ecls_its_ende, 
    //                 _ende_child, 
    //                 _ende_child_ptr, 
    //                 _root_eclass_id, 
    //                 _ende_cost,
    //                 best_ende_of_ecls, 
    //                 picked_endes);
    // }
    
    std::vector<bool> picked_endes_tmp = picked_endes;
    start = std::chrono::steady_clock::now();
      _cpu_reconstruct_graph(picked_endes); 
    end = std::chrono::steady_clock::now();
    duration = end - start;
    std::cout << "CPU Reconstruct Graph Runtime: " << round_to((duration.count()) * 1000, 0.01) << " (ms)\n";
    
    // if (ILP_solver) {
    //   _cpsat_solver(ILP_time_limit, 
    //                 _ecls_its_ende_newG, 
    //                 _ende_child, 
    //                 _ende_child_ptr, 
    //                 _root_eclass_id, 
    //                 _ende_cost,
    //                 best_ende_of_ecls, 
    //                 picked_endes_tmp);
    // }

  #ifdef GPU_VERSION
    // GPU version
    // 計時我放在 function 裡面
    auto [best_tree_cost_gpu, dag_cost_gpu] = _gpu_best_tree();
    printf("GPU RESULTS: best_tree_cost_gpu = %.2lf, dag_cost_gpu = %.2lf\n", 
              best_tree_cost_gpu, dag_cost_gpu);

  float gpu_reconst_Time = 0.;
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);    
  cudaEventRecord(startGPU);
      _gpu_reconstruct_graph(); 
  cudaEventRecord(stopGPU);
  cudaEventSynchronize(stopGPU);
  cudaEventElapsedTime(&gpu_reconst_Time, startGPU, stopGPU);                                    
  printf("GPU Reconstruct Graph Runtime: %.2f (ms)\n", gpu_reconst_Time);



    #endif
  }

private:
  /* Read */
  void _read_input(const std::string input_path);
  void _prune_dead_nodes(const uint32_t tmp_num_ecls,
    const uint32_t tmp_num_ende,
    const std::vector<std::string>& tmp_ende_name, 
    const std::vector<std::string>& tmp_ende_op, 
    const std::vector<float>& tmp_ende_cost, 
    const std::vector<uint32_t>& tmp_ende_ecls,
    const std::vector<uint32_t>& tmp_ende_child, 
    const std::vector<uint32_t>& tmp_ende_child_ptr, 
    const std::vector<uint32_t>& tmp_root_eclass_id);
  
  /* Run */
  /* Tree Level Relaxation */
  // CPU version
  std::pair<double, double> _cpu_best_tree(const std::vector<float>& tmp_ende_cost,
    std::vector<double>& cost_of_ecls,
    std::vector<double>& cost_of_ende,
    std::vector<uint32_t>& best_ende_of_ecls,
    std::vector<bool>& picked_endes); 
  double _compute_dag_cost(const std::vector<uint32_t>& best_ende_of_ecls,
    std::vector<bool>& picked_endes,
    std::vector<std::vector<uint32_t>>& ende_picked_by_ende);
  void _init_picked_adj(const std::vector<bool>& picked_endes);
  // GPU version
  std::pair<double, double> _gpu_best_tree();


  /* DAG Construction */
  // CPU version
  void _cpu_reconstruct_graph(std::vector<bool>& picked_endes);
  void _construct_deg(const std::vector<bool>& picked_endes,
    const std::vector<double>& cost_of_ende); 
  void _top_down_levelize(const std::vector<bool>& picked_endes,
    std::vector<int>& level);
  void _bottom_up_levelize(const std::vector<bool>& picked_endes,
    std::vector<int>& level);
    void _update_deg(const bool need_to_update, 
      const std::vector<bool>& picked_endes);
    void _BFS(const bool is_td, std::vector<uint32_t>& deg, 
      const std::vector<bool>& picked_endes,
      const std::vector<uint32_t>& global_adj_1D,
      const std::vector<uint32_t>& global_adj_1D_ptr,
      const std::vector<bool>& picked_adj,
      std::vector<int>& level);
  bool _insertion(const bool is_td, 
    std::vector<bool>& picked_endes,
    std::vector<int>& level);
  void _update_picked_adj(const uint32_t fm_ende, const uint32_t to_ende);
  void _clear_redundant_endes(std::vector<bool>& picked_endes);
    void mark_reachable(const std::vector<uint32_t>& starts,
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
  void _gpu_reconstruct_graph();

  /* ILP solver */ 
  void _cpsat_solver(const double time_limit,
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

  std::vector<std::vector<uint32_t>> _ecls_its_ende_newG;

  // global Device memory 
  uint32_t* _dvc_best_ende_of_ecls;
  bool* _dvc_update;
  uint32_t* _dvc_ecls_its_ende_1D; // 有用了
  uint32_t* _dvc_ecls_its_ende_1D_ptr; // 目前還沒有用
  uint32_t* _dvc_root_eclass_id;
  uint32_t* _dvc_ende_picked_by_ende;
  uint32_t* _dvc_picked_endes; // act bool
  uint32_t* _dvc_q; // queue
  uint32_t* _dvc_q_sz; // size of queue
  uint32_t* _dvc_q_sp; // start point of queue

  uint32_t* _dvc_global_adj_1D;
  uint32_t* _dvc_global_adj_1D_ptr;
  uint32_t* _dvc_global_inv_adj_1D;
  uint32_t* _dvc_global_inv_adj_1D_ptr;
  uint32_t* _dvc_indegree;
  uint32_t* _dvc_inv_indegree;
  uint32_t* _dvc_picked_adj;
  uint32_t* _dvc_picked_inv_adj;
  uint32_t* _dvc_tmp_picked_adj;
  uint32_t* _dvc_tmp_picked_inv_adj; 
  int *_dvc_level;

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
}; // class G_EXT
#endif


