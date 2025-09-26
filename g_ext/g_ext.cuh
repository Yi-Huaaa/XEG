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

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

__global__ void GPU_print_test(); 

class G_EXT {

public:
  void gpu_public_test() { _gpu_private_test(); }
  void gpu_print();
  // Read input file
  void read(const std::string input_path); 

  /* run */
  void run() {
    _SCC();
  }

private:
  void _gpu_private_test() { printf("\nclass G_EXT: Hello world CUDA function!\n"); }
  /* run */
  void _SCC(); 
  void _construct_graph(std::vector<std::unordered_set<uint32_t>> &adj);
  void _find_SCC(const std::vector<std::unordered_set<uint32_t>>& adj);
  void _count_dead_nodes();
  void _rebuild_graph(const std::vector<std::unordered_set<uint32_t>>& adj);
  void _gpu_relaxation(); 

  // read in from .json
  uint32_t _num_ende; // number of enodes
  uint32_t _num_ecls; // nuber of eclasses 
  std::vector<std::string>  _ende_name; 
  std::vector<std::string>  _ende_op; 
  std::vector<float>  _ende_cost; 
  std::unordered_map<std::string, uint32_t> ecls_name_to_id; 
  std::vector<uint32_t>  _ende_ecls; // the eclass that enode belonsg to
  std::vector<uint32_t>  _ende_child; 
  std::vector<uint32_t>  _ende_child_ptr; 
  std::vector<uint32_t> _root_eclass_id;


  // Print function
  void _print_input() const;
  void _print_adj(const std::vector<std::unordered_set<uint32_t>>& adj) const;


}; // class G_EXT
#endif


