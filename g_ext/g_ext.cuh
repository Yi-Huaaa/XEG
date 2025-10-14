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
      auto [best_tree_cost_cpu, dag_cost_cpu] = _cpu_best_tree(_ende_cost, cost_of_ecls, cost_of_ende, best_ende_of_ecls, picked_endes);
      // double cpu_dag_refine = _cpu_dag_refinement(cost_of_ecls, cost_of_ende, best_ende_of_ecls, picked_endes);

    auto end = std::chrono::steady_clock::now();
    duration = end - start;
    std::cout << "CPU Relaxation Runtime: " << round_to((duration.count()) * 1000, 0.01) << " (ms)\n";
    double cpu_dag_refine = 0.;
    printf("CPU RESULTS: best_tree_cost_cpu = %.2lf, dag_cost_cpu = %.2lf, cpu_dag_refine = %.2lf\n", 
              best_tree_cost_cpu, dag_cost_cpu, cpu_dag_refine);

    _reconstruct_graph(cost_of_ecls, cost_of_ende, 
                      best_ende_of_ecls, picked_endes); 


    // // GPU version
    // float gpuTime = 0.;
    // cudaEvent_t startGPU, stopGPU;
    // cudaEventCreate(&startGPU);
    // cudaEventCreate(&stopGPU);    
    // cudaEventRecord(startGPU);
    //   auto [best_tree_cost_gpu, dag_cost_gpu] = _gpu_best_tree();
    // cudaEventRecord(stopGPU);
    // cudaEventSynchronize(stopGPU);
    // cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);                                    
    // printf("GPU Runtime: %.2f (ms)\n", gpuTime);
    // printf("GPU RESULTS: best_tree_cost_gpu = %.2lf, dag_cost_gpu = %.2lf\n", 
    //           best_tree_cost_gpu, dag_cost_gpu);

    // if (ILP_solver) {
    //   _cpsat_solver(ILP_time_limit, 
    //                 cost_of_ecls, cost_of_ende, best_ende_of_ecls, picked_endes);
    // }

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
  std::pair<double, double> _cpu_best_tree(const std::vector<float>& tmp_ende_cost,
                                            std::vector<double>& cost_of_ecls,
                                            std::vector<double>& cost_of_ende,
                                            std::vector<uint32_t>& best_ende_of_ecls,
                                            std::vector<bool>& picked_endes); 
  double _compute_dag_cost(const std::vector<uint32_t>& best_ende_of_ecls,
                          std::vector<bool>& picked_endes);
  std::pair<double, double> _gpu_best_tree();
  /* DAG level relxation */
  void _reconstruct_graph(std::vector<double>& cost_of_ecls,
                          std::vector<double>& cost_of_ende,
                          std::vector<uint32_t>& best_ende_of_ecls,
                          std::vector<bool>& picked_endes);
  void _construct_deg(const std::vector<bool>& picked_endes,
                      const std::vector<double>& cost_of_ende); 
  void _top_down_levelize(const std::vector<bool>& picked_endes,
                          std::vector<int>& level,
                          std::vector<std::vector<uint32_t>>& ende_in_level);    
  void _bottom_up_levelize(const std::vector<bool>& picked_endes,
                          std::vector<int>& level,
                          std::vector<std::vector<uint32_t>>& ende_in_level);
    void _update_deg(const bool is_td, const std::vector<bool>& picked_endes);
    void _BFS(const bool is_td, std::vector<uint32_t>& deg, 
              const std::vector<bool>& picked_endes,
              const std::vector<std::unordered_set<uint32_t>>& adj,
              std::vector<int>& level,
              std::vector<std::vector<uint32_t>>& ende_in_level);
  bool _top_down_insertion(std::vector<bool>& picked_endes,
                          std::vector<int>& level,
                          std::vector<std::vector<uint32_t>>& ende_in_level);
  bool _bottom_up_insertion(std::vector<bool>& picked_endes,
                          std::vector<int>& level,
                          std::vector<std::vector<uint32_t>>& ende_in_level);  
  /* ILP solver */ 
  void _cpsat_solver(const double time_limit,
                    const std::vector<double>& cost_of_ecls,
                    const std::vector<double>& cost_of_ende,
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

  // For refinement 
  std::vector<uint32_t> _ecls_parnt; // eclass point to its parent enode
  std::vector<uint32_t> _ecls_parnt_ptr; // eclass point to its parent enode
  std::vector<std::unordered_set<uint32_t>> _adj;
  std::vector<std::unordered_set<uint32_t>> _inv_adj;
  std::vector<uint32_t> _indegree;
  std::vector<uint32_t> _inv_indegree;
  std::vector<std::vector<uint32_t>> _par_endes; // it may have multiple parent enodes 
  std::vector<uint32_t> _tar_ende; // only one target enode
  std::vector<std::vector<uint32_t>> _chd_ende; // it may have multiple children enode


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
                          const std::vector<std::vector<uint32_t>>& ende_in_level) const;
}; // class G_EXT
#endif


