#include <chrono>
#include <thread>
#include <assert.h>
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <cmath>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <unordered_set>
#include <utility> 
#include <algorithm>
#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "g_ext.cuh"

using json = nlohmann::json; // parse JSON json j; ifs >> j;

// error checking macro
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

// G_EXT fucntions

__global__ void GPU_print_test() {
    printf("threadIdx.x = %d, hello world!\n", threadIdx.x);
}

void G_EXT::gpu_print() {
    printf("Get inside G_EXT::gpu_print, starts to launch GPU_print_test...\n");
    GPU_print_test <<< 1, 32 >>>();
    cudaDeviceSynchronize(); 
    printf("\ngpu_equality_saturator::gpu_print test ends...\n");
}

void G_EXT::read(const std::string input_path) {
  std::ifstream ifs(input_path);
  if (!ifs.is_open()) {
    std::cerr << "Error: cannot open file " << input_path << std::endl;
    return;
  }

  // parse JSON
  json j;
  ifs >> j;

  auto nodes = j["nodes"]; // get number of nodes object
  _num_ende = nodes.size();
  _ende_name.reserve(_num_ende);
  _ende_op.reserve(_num_ende);
  _ende_cost.reserve(_num_ende);
  _ende_ecls.reserve(_num_ende);
  _ende_child_ptr.reserve(_num_ende + 1);

  // lambda function for eclass
  uint32_t ecls_counter = 0;
  auto get_ecls_id = [&](const std::string &ecls_name) {
    if (ecls_name_to_id.find(ecls_name) == ecls_name_to_id.end()) {
      ecls_name_to_id[ecls_name] = ecls_counter++;
    }
    return ecls_name_to_id[ecls_name];
  };

  // 遍歷所有 enode
  _ende_child_ptr.push_back(0); // CSR 起始位置
  for (auto &[ename, node] : nodes.items()) {
    std::string op = node["op"];
    float cost = node["cost"];
    std::string ecls_name = node["eclass"];
    uint32_t ecls_id = get_ecls_id(ecls_name);

    _ende_name.push_back(ename);     // enode 的 id (e.g., "0__0")
    _ende_op.push_back(op);          // operator string
    _ende_cost.push_back(cost);      // cost
    _ende_ecls.push_back(ecls_id);   // enode's eclass

    // children
    for (auto &ch : node["children"]) {
      std::string child_enode = ch.get<std::string>();
      std::string child_ecls = nodes[child_enode]["eclass"];
      uint32_t child_ecls_id = get_ecls_id(child_ecls);
      _ende_child.push_back(child_ecls_id);
    }

    _ende_child_ptr.push_back(_ende_child.size());
  }

  _num_ecls = ecls_counter; // get number of eclasses

  // roots
  for (auto &r : j["root_eclasses"]) {
    std::string ecls_name = r.get<std::string>();
    _root_eclass_id.push_back(get_ecls_id(ecls_name));
  }
}


void G_EXT::print_input() {
  std::cout << "\n========== G_EXT Internal State ==========\n";
  std::cout << "_num_ende: " << _num_ende << ", " << "_num_ecls: " << _num_ecls << "\n\n";

  // std::cout << "Enodes info:\n";
  // for (uint32_t i = 0; i < _num_ende; i++) {
  //   std::cout << "  enode[" << i << "]: "
  //             << "name=" << _ende_name[i] 
  //             << ", op=" << _ende_op[i] 
  //             << ", cost=" << _ende_cost[i]
  //             << ", eclass=" << _ende_ecls[i]
  //             << "\n";

  //   uint32_t start = _ende_child_ptr[i];
  //   uint32_t end   = _ende_child_ptr[i + 1];
  //   std::cout << "    children: ";
  //   for (uint32_t j = start; j < end; j++) {
  //     std::cout << _ende_child[j] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // std::cout << "\nEclass name -> id map:\n";
  // for (auto &kv : ecls_name_to_id) {
  //   std::cout << "  " << kv.first << " -> " << kv.second << "\n";
  // }

  // std::cout << "\nRoot eclasses:\n";
  // for (auto r : _root_eclass_id) {
  //   std::cout << "  " << r << "\n";
  // }

  // std::cout << "==========================================\n\n";
}
