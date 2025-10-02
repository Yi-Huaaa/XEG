#include <chrono>
#include <iostream>
#include <g_ext/G_EXT.hpp>

double round_to(double value, double precision = 1.0) {
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  // // Timer
  std::chrono::duration<double> duration_read, duration_run_cpu;
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
  G_EXT g_ext;

  auto start = std::chrono::steady_clock::now();
    g_ext.read(input_path); 
  auto end = std::chrono::steady_clock::now();
  duration_read = end - start;
  std::cout << "Read Time: " << round_to((duration_read.count()) * 1000, 0.001) << " ms\n";

  // start = std::chrono::steady_clock::now();
    float res_cpu = g_ext.run_cpu(); 
  // end = std::chrono::steady_clock::now();
  // duration_run_cpu = end - start;

  // cudaEventRecord(startGPU);
    float res_gpu = g_ext.run_gpu(); 
  // cudaEventRecord(stopGPU);
  // cudaEventSynchronize(stopGPU);
  // cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);

  printf("FINAL COST = %.2lf, %.2lf\n", res_cpu, res_gpu);
  // std::cout << "CPU Runtime (ms): " << round_to((duration_run_cpu.count()) * 1000, 0.001) << " ms\n";
  // printf("GPU Runtime (ms) : %f (ms)\n", gpuTime);


  return 0;
}