#include <chrono>
#include <iostream>
#include <g_ext/G_EXT.hpp>


double round_to(double value, double precision = 1.0) {
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  // Timer
  std::chrono::duration<double> duration;
  if (argc < 2) {
    std::cerr << "Usage example: ./main/simple ~/extraction-gym/data/egg/math_diff_same.json\n";
    return 1;
  }
  
  // Input file 
  std::string input_path = argv[1];


  // Call function 
  G_EXT g_ext;
  auto start = std::chrono::steady_clock::now();
  // std::vector<std::string> all_sexprs = egraph.EqnToSexprAll(eqn_path);
    g_ext.gpu_public_test();
    g_ext.gpu_print();
  auto end = std::chrono::steady_clock::now();
  duration = end - start;
  std::cout << "GPU Time: " << round_to((duration.count()) * 1000, 0.001) << " ms\n";

  return 0;
}