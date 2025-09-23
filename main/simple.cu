#include <chrono>
#include <iostream>
#include <g_ext/G_EXT.hpp>

double round_to(double value, double precision = 1.0) {
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  // // Timer
  std::chrono::duration<double> duration;
  float gpuTime = 0.;
  
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);
  if (argc < 2) {
    std::cerr << "Usage example: ./main/simple ~/extraction-gym/data/egg/math_diff_same.json\n";
    return 1;
  }
  
  // Input file 
  std::string input_path = argv[1];


  cudaEventRecord(startGPU);
  auto start = std::chrono::steady_clock::now();
  
    G_EXT g_ext;
    g_ext.gpu_public_test();
    g_ext.gpu_print();
  
  auto end = std::chrono::steady_clock::now();
  duration = end - start;
  cudaEventRecord(stopGPU);
  cudaEventSynchronize(stopGPU);
  cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);

  std::cout << "GPU Time: " << round_to((duration.count()) * 1000, 0.001) << " ms\n";
  printf("GPU Sort Time (ms) : %f (ms)\n", gpuTime);


  return 0;
}