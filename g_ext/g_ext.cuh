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
  void read(const std::string input_path) {
    // Tempory memory 
    uint32_t num_ende = 0, num_ecls = 0;

    _read_input(input_path, 
                &num_ende, 
                &num_ecls); 

    _prune_dead_nodes(&num_ende, 
                      &num_ecls);
  }

  /* run */
  void run() {
    _preprocess();
    _gpu_relaxation();
  }

private:
  void _gpu_private_test() { printf("\nclass G_EXT: Hello world CUDA function!\n"); }
  /* Read */
  void _read_input(const std::string input_path, 
                  uint32_t* num_ende,
                  uint32_t* num_ecls);

  void _prune_dead_nodes(uint32_t* num_ende,
                        uint32_t* num_ecls);
  
  /* Run */
  void _preprocess(); 
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
    void _get_queue_list(std::vector<uint32_t>& levelize_leaves_to_roots,
                        std::unordered_map<uint32_t, uint32_t>& ecls_id_to_alive_id,
                        std::unordered_map<uint32_t, std::vector<uint32_t>>& comp_nodes,
                        std::unordered_map<uint32_t, uint32_t>& alive_id_to_super_adj);

  void _gpu_relaxation(); 

  // read in from .json
  uint32_t _num_ecls; // nuber of eclasses 
  uint32_t _num_ende; // number of enodes
  std::vector<std::string>  _ende_name; // "Alive + Dead"
  std::vector<std::string>  _ende_op; // "Alive + Dead"
  std::vector<float>  _ende_cost; // "Alive + Dead"
  std::unordered_map<std::string, uint32_t> _ecls_name_to_id; // "Alive + Dead"
  std::vector<uint32_t>  _ende_ecls; // the eclass that enode belonsg to; // "Alive + Dead"
  std::vector<uint32_t>  _ende_child; // "Alive + Dead"
  std::vector<uint32_t>  _ende_child_ptr; // "Alive + Dead"
  std::vector<uint32_t> _root_eclass_id;// "Alive + Dead"

  // Prune dead enodes info
  uint32_t _num_alive_ecls; // nuber of "Alive" eclasses   
  uint32_t _num_alive_ende; // number of "Alive" enodes
  std::vector<bool> _alive_ecls;

  // queue list for GPU brute-force algorithm 



  // Print function
  void _print_input() const;
  void _print_adj(const std::vector<std::unordered_set<uint32_t>>& adj) const;


}; // class G_EXT
#endif


