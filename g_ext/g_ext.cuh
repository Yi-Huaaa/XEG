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
    _prune_dead_nodes();
  }

  /* run */
  double round_to(double value, double precision = 1.0) {
    return std::round(value / precision) * precision;
  }
  
  void run(const bool ILP_solver = false, 
          const double ILP_time_limit = 360
) {
    printf("Hellow bibi\n");
    // std::vector<float> csr_ende_cost;
    // std::vector<uint32_t> csr_ende_ecls;
    // std::vector<uint32_t> csr_ende_child;
    // std::vector<uint32_t> csr_ende_child_ptr;
    // std::vector<std::vector<uint32_t>> ecls_its_ende;

    // _preprocess(csr_ende_cost, csr_ende_ecls, csr_ende_child, csr_ende_child_ptr, ecls_its_ende); 
    
    // // CPU version
    // std::vector<double> cost_of_ecls; // Final cost of each eclass
    // std::vector<double> cost_of_ende; // Final cost of each enode
    // std::vector<uint32_t> best_ende_of_ecls; // enode idx with smallest cost in that eclass     
    // std::vector<bool> picked_endes;
    // std::chrono::duration<double> duration;
    // auto start = std::chrono::steady_clock::now();
    //   auto [best_tree_cost_cpu, dag_cost_cpu] = _cpu_relaxation(_queue_list.size(), 
    //                                                             csr_ende_cost, csr_ende_ecls,
    //                                                             csr_ende_child, csr_ende_child_ptr,
    //                                                             cost_of_ecls, cost_of_ende, best_ende_of_ecls, picked_endes);
    //   double cpu_dag_refine = _cpu_dag_refinement(_queue_list.size(), 
    //                                               csr_ende_cost, csr_ende_ecls,
    //                                               csr_ende_child, csr_ende_child_ptr,
    //                                               cost_of_ecls, cost_of_ende, best_ende_of_ecls, picked_endes,
    //                                               dag_cost_cpu); 

    // auto end = std::chrono::steady_clock::now();
    // duration = end - start;
    // std::cout << "CPU Relaxation Runtime: " << round_to((duration.count()) * 1000, 0.01) << " (ms)\n";
    // printf("CPU RESULTS: best_tree_cost_cpu = %.2lf, dag_cost_cpu = %.2lf, cpu_dag_refine = %.2lf\n", 
    //           best_tree_cost_cpu, dag_cost_cpu, cpu_dag_refine);

    // // // GPU version
    // // float gpuTime = 0.;
    // // cudaEvent_t startGPU, stopGPU;
    // // cudaEventCreate(&startGPU);
    // // cudaEventCreate(&stopGPU);    
    // // cudaEventRecord(startGPU);
    // //   auto [best_tree_cost_gpu, dag_cost_gpu] = _gpu_relaxation(_queue_list.size(), csr_ende_cost, csr_ende_ecls,
    // //                                                             csr_ende_child, csr_ende_child_ptr);
    // // cudaEventRecord(stopGPU);
    // // cudaEventSynchronize(stopGPU);
    // // cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);                                    
    // // printf("GPU Runtime: %.2f (ms)\n", gpuTime);
    // // printf("GPU RESULTS: best_tree_cost_gpu = %.2lf, dag_cost_gpu = %.2lf\n", 
    // //           best_tree_cost_gpu, dag_cost_gpu);

    // if (ILP_solver) {
    //   _cpsat_solver(ILP_time_limit, 
    //                 csr_ende_cost, csr_ende_ecls,
    //                 csr_ende_child, csr_ende_child_ptr, ecls_its_ende, 
    //                 cost_of_ecls, cost_of_ende, best_ende_of_ecls, picked_endes);
    // }

  }

  // ILP Solver functions 
  void SimpleSatProgram();

private:
  /* Read */
  void _read_input(const std::string input_path);
  void _prune_dead_nodes();
  
  /* Run */
  void _preprocess(std::vector<float>& csr_ende_cost, 
                  std::vector<uint32_t>& csr_ende_ecls,
                  std::vector<uint32_t>& csr_ende_child,
                  std::vector<uint32_t>& csr_ende_child_ptr,
                  std::vector<std::vector<uint32_t>>& ecls_its_ende); 
    void _construct_alive_graph(std::unordered_map<uint32_t, uint32_t>& ecls_id_to_alive_id, 
                                std::vector<std::unordered_set<uint32_t>>& alive_adj);
    void _find_SCC(const std::vector<std::unordered_set<uint32_t>>& adj,
                   std::unordered_map<uint32_t, std::vector<uint32_t>>& comp_nodes);
    void _construct_superGraph(const std::unordered_map<uint32_t, std::vector<uint32_t>>& comp_nodes,
                              const std::vector<std::unordered_set<uint32_t>>& alive_adj,
                              std::unordered_map<uint32_t, uint32_t>& alive_id_to_super_adj,
                              std::vector<std::unordered_set<uint32_t>>& super_adj);
    void _levelize_from_leaves(const std::vector<std::unordered_set<uint32_t>>& super_adj,
                              std::vector<uint32_t>& levelize_leaves_to_roots);
    void _get_queue_list(const std::vector<uint32_t>& levelize_leaves_to_roots,
                        const std::unordered_map<uint32_t, uint32_t>& ecls_id_to_alive_id,
                        const std::unordered_map<uint32_t, uint32_t>& alive_id_to_super_adj);
    void _CSR (const std::vector<uint32_t>& queue, 
              const std::vector<float>& ende_cost, 
              const std::vector<uint32_t>& ende_ecls,
              const std::vector<uint32_t>& ende_child,
              const std::vector<uint32_t>& ende_child_ptr,
              std::vector<float>& csr_ende_cost, 
              std::vector<uint32_t>& csr_ende_ecls,
              std::vector<uint32_t>& csr_ende_child,
              std::vector<uint32_t>& csr_ende_child_ptr,
              std::vector<std::vector<uint32_t>>& ecls_its_ende);

  /* Tree Level Relaxation */
  std::pair<double, double> _cpu_relaxation(const uint32_t N, const std::vector<float>& csr_ende_cost, 
                                            const std::vector<uint32_t>& csr_ende_ecls,
                                            const std::vector<uint32_t>& csr_ende_child,
                                            const std::vector<uint32_t>& csr_ende_child_ptr,
                                            std::vector<double>& cost_of_ecls,
                                            std::vector<double>& cost_of_ende,
                                            std::vector<uint32_t>& best_ende_of_ecls,
                                            std::vector<bool>& picked_endes); 
  double _compute_dag_cost(const std::vector<uint32_t>& csr_ende_ecls,
                          const std::vector<float>& csr_ende_cost, 
                          const std::vector<uint32_t>& csr_ende_child,
                          const std::vector<uint32_t>& csr_ende_child_ptr,
                          const std::vector<uint32_t>& best_ende_of_ecls,
                          std::vector<bool>& picked_endes);
  double _cpu_dag_refinement(const uint32_t N, const std::vector<float>& csr_ende_cost, 
                            const std::vector<uint32_t>& csr_ende_ecls,
                            const std::vector<uint32_t>& csr_ende_child,
                            const std::vector<uint32_t>& csr_ende_child_ptr,
                            std::vector<double>& cost_of_ecls,
                            std::vector<double>& cost_of_ende,
                            std::vector<uint32_t>& best_ende_of_ecls,
                            std::vector<bool>& picked_endes,
                            const double init_dag_cost);
  std::pair<double, double> _gpu_relaxation(const uint32_t N, const std::vector<float>& csr_ende_cost, 
                                            const std::vector<uint32_t>& csr_ende_ecls,
                                            const std::vector<uint32_t>& csr_ende_child,
                                            const std::vector<uint32_t>& csr_ende_child_ptr);
  
  
  /* ILP solver */ 
  void _cpsat_solver(const double time_limit,
                    const std::vector<float>& csr_ende_cost, 
                    const std::vector<uint32_t>& csr_ende_ecls,
                    const std::vector<uint32_t>& csr_ende_child,
                    const std::vector<uint32_t>& csr_ende_child_ptr,
                    const std::vector<std::vector<uint32_t>>& ecls_its_ende,
                    const std::vector<double>& cost_of_ecls,
                    const std::vector<double>& cost_of_ende,
                    const std::vector<uint32_t>& best_ende_of_ecls,
                    const std::vector<bool>& picked_endes);


  /* Private members */
  // read in from .json
  uint32_t _num_ecls; // nuber of eclasses 
  uint32_t _num_ende; // number of enodes
  std::vector<std::string> _ende_name; // "Alive + Dead"
  std::vector<std::string> _ende_op; // "Alive + Dead"
  std::vector<float> _ende_cost; // "Alive + Dead"
  std::unordered_map<std::string, uint32_t> _ecls_name_to_id; // "Alive + Dead"
  std::vector<uint32_t> _ende_ecls; // the eclass that enode belonsg to; // "Alive + Dead"
  std::vector<uint32_t> _ende_child; // "Alive + Dead"
  std::vector<uint32_t> _ende_child_ptr; // "Alive + Dead"
  std::vector<uint32_t> _root_eclass_id;// "Alive + Dead"

  // Prune dead enodes info
  uint32_t _num_alive_ecls; // nuber of "Alive" eclasses   
  uint32_t _num_alive_ende; // number of "Alive" enodes
  std::vector<bool> _alive_ecls;

  // queue list for GPU brute-force algorithm 
  std::vector<uint32_t> _queue_list;

  // Print function
  void _print_input() const;
  void _print_adj(const std::vector<std::unordered_set<uint32_t>>& adj) const;


}; // class G_EXT
#endif


