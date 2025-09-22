#include <chrono>
#include <iostream>
#include <g_ext/G_EXT.hpp>


double round_to(double value, double precision = 1.0) {
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  // Timer
  std::chrono::duration<double> duration;

  // From user's perspective: test GPU function 
  G_EXT g_ext;
  auto start = std::chrono::steady_clock::now();
    g_ext.gpu_public_test();
    g_ext.gpu_print();
  auto end = std::chrono::steady_clock::now();
  duration = end - start;
  std::cout << "GPU Time: " << round_to((duration.count()) * 1000, 0.001) << " ms\n";


  return 0;
}