#include <chrono>
#include <iostream>
#include <g_ext/G_EXT.hpp>

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/base/logging.h" // For LOG(INFO)
#include <iostream>
#include <vector>

using namespace operations_research;
using namespace sat;

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/base/logging.h" // For LOG(INFO)
#include <iostream>
#include <vector>

namespace operations_research {
namespace sat {

void SimpleSatProgram() {
    // 1. Create the CP-SAT model.
    CpModelBuilder cp_model;

    // 2. Define the variables.
    // Each variable 'a', 'b', 'c' will be an integer in the range [0, 100].
    const Domain domain(0, 100);
    IntVar a = cp_model.NewIntVar(domain).WithName("a");
    IntVar b = cp_model.NewIntVar(domain).WithName("b");
    IntVar c = cp_model.NewIntVar(domain).WithName("c");

    // 3. Define the constraints.
    // The sum of 'a', 'b', and 'c' must be equal to 200.
    cp_model.AddEquality(a + b + c, 200);

    // 4. Solve the model.
    CpSolverResponse response = Solve(cp_model.Build());

    // 5. Interpret the results.
    if (response.status() == CpSolverStatus::OPTIMAL ||
        response.status() == CpSolverStatus::FEASIBLE) {
        LOG(INFO) << "Solution found:";
        LOG(INFO) << "a = " << SolutionIntegerValue(response, a);
        LOG(INFO) << "b = " << SolutionIntegerValue(response, b);
        LOG(INFO) << "c = " << SolutionIntegerValue(response, c);
    } else {
        LOG(INFO) << "No solution found.";
    }
}

}  // namespace sat
}  // namespace operations_research

int main() {
    operations_research::sat::SimpleSatProgram();
    return EXIT_SUCCESS;
}

// double round_to(double value, double precision = 1.0) {
//   return std::round(value / precision) * precision;
// }

// int main(int argc, char *argv[]) {
//   // // Timer
//   std::chrono::duration<double> duration_read;
  
//   cudaEvent_t startGPU, stopGPU;
//   cudaEventCreate(&startGPU);
//   cudaEventCreate(&stopGPU);
//   if (argc < 2) {
//     std::cerr << "Usage example: ./main/simple ~/extraction-gym/data/egg/math_diff_same.json\n";
//     return 1;
//   }
  
//   // Input file 
//   std::string input_path = argv[1];
//   G_EXT g_ext;

//   auto start = std::chrono::steady_clock::now();
//     g_ext.read(input_path); 
//   auto end = std::chrono::steady_clock::now();
//   duration_read = end - start;
//   std::cout << "Read Time: " << round_to((duration_read.count()) * 1000, 0.001) << " ms\n";

//     g_ext.run(); 


//   // ===== OR-Tools CP-SAT test starts =====
//   printf("bibi test\n");



//  return 0;
// }