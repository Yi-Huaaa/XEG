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

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

__global__ void GPU_print_test(); 

class G_EXT {

public:
  void gpu_public_test() { _gpu_private_test(); }
  void gpu_print();
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
  
  void run() {
    std::vector<float> csr_ende_cost;
    std::vector<uint32_t> csr_ende_ecls;
    std::vector<uint32_t> csr_ende_child;
    std::vector<uint32_t> csr_ende_child_ptr;

    _preprocess(csr_ende_cost, csr_ende_ecls, csr_ende_child, csr_ende_child_ptr); 
    
    // CPU version
    std::chrono::duration<double> duration;
    auto start = std::chrono::steady_clock::now();
      float res_cpu = _cpu_relaxation(_queue_list.size(), csr_ende_cost, csr_ende_ecls,
                                      csr_ende_child, csr_ende_child_ptr);
    auto end = std::chrono::steady_clock::now();
    duration = end - start;
    std::cout << "CPU Runtime (ms): " << round_to((duration.count()) * 1000, 0.001) << " ms\n";

    // GPU version
    float gpuTime = 0.;
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);    
    cudaEventRecord(startGPU);
      float res_gpu = _gpu_relaxation(_queue_list.size(), csr_ende_cost, csr_ende_ecls,
                                      csr_ende_child, csr_ende_child_ptr);
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);                                    
    printf("GPU Runtime (ms) : %f (ms)\n", gpuTime);
    printf("FINAL COST = %.2lf, %.2lf\n", res_cpu, res_gpu);
  }

private:
  void _gpu_private_test() { printf("\nclass G_EXT: Hello world CUDA function!\n"); }
  /* Read */
  void _read_input(const std::string input_path);

  void _prune_dead_nodes();
  
  /* Run */
  void _preprocess(std::vector<float>& csr_ende_cost, 
                  std::vector<uint32_t>& csr_ende_ecls,
                  std::vector<uint32_t>& csr_ende_child,
                  std::vector<uint32_t>& csr_ende_child_ptr); 
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
              std::vector<uint32_t>& csr_ende_child_ptr);

  /* Relaxation */
  float _cpu_relaxation(const uint32_t N, const std::vector<float>& csr_ende_cost, 
                        const std::vector<uint32_t>& csr_ende_ecls,
                        const std::vector<uint32_t>& csr_ende_child,
                        const std::vector<uint32_t>& csr_ende_child_ptr); 
  float _gpu_relaxation(const uint32_t N, const std::vector<float>& csr_ende_cost, 
                        const std::vector<uint32_t>& csr_ende_ecls,
                        const std::vector<uint32_t>& csr_ende_child,
                        const std::vector<uint32_t>& csr_ende_child_ptr); 

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


