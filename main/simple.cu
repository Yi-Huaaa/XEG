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
    std::cerr << "Usage: ./main/simple <input_file> [DAG_KERNEL] [NUM_THREAD_DAG_CONSTR]\n";
    std::cerr << "Example: ./main/simple ~/data/math_diff.json 1 1024\n";
    return 1;
  }
  
  // === Parse arguments ===
  std::string input_path = argv[1];
  int DAG_KERNEL = (argc > 2) ? std::stoi(argv[2]) : 0;
  int NUM_THREAD_DAG_CONSTR = (argc > 3) ? std::stoi(argv[3]) : 256;
  // int EXTEND_MUL = (argc > 4) ? std::stoi(argv[4]) : 128;

  std::cout << "-------------------------------------\n";
  std::cout << "Input file: " << input_path << "\n";
  std::cout << "DAG_KERNEL: " << DAG_KERNEL << "\n";
  std::cout << "NUM_THREAD_DAG_CONSTR: " << NUM_THREAD_DAG_CONSTR << "\n";
  // std::cout << "EXTEND_MUL: " << EXTEND_MUL << "\n";
  std::cout << "-------------------------------------\n";


  G_EXT g_ext;

  auto start = std::chrono::steady_clock::now();
    g_ext.read(input_path); 
  auto end = std::chrono::steady_clock::now();
  duration_read = end - start;
  std::cout << "Read Time: " << g_ext.round_to((duration_read.count()) * 1000, 0.001) << " (ms)\n";


    // (use ILP, ILP tim limit (sec), DAG_KERNEL, NUM_THREAD_DAG_CONSTR)
    // g_ext.run(false, 360, DAG_KERNEL, NUM_THREAD_DAG_CONSTR);
    g_ext.run(true, 360, DAG_KERNEL, NUM_THREAD_DAG_CONSTR);



  return 0;
}