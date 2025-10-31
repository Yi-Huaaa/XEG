#include <chrono>
#include <iostream>
#include <g_ext/G_EXT.hpp>
#include <vector>
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/base/logging.h" // For LOG(INFO)

int main(int argc, char *argv[]) {
  // // Timer
  std::chrono::duration<double> duration_read;
  
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);
  if (argc < 2) {
    std::cerr << "Usage example: ./main/simple ~/extraction-gym/data/egg/math_diff_same.json\n";
    return 1;
  }
  
  // Input file 
  std::string input_path = argv[1];
  G_EXT g_ext;

  auto start = std::chrono::steady_clock::now();
    g_ext.read(input_path); 
  auto end = std::chrono::steady_clock::now();
  duration_read = end - start;
  std::cout << "Read Time: " << g_ext.round_to((duration_read.count()) * 1000, 0.001) << " (ms)\n";

    g_ext.run(true, 50); 

  return 0;
}