#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <cuopt/linear_programming/cuopt_c.h>

#include <array>
#include <cmath>

TEST_CASE("cuOpt C API solves a two-variable MILP") {
  cuOptOptimizationProblem problem = nullptr;
  cuOptSolverSettings settings = nullptr;
  cuOptSolution solution = nullptr;

  constexpr cuopt_int_t num_variables = 2;
  constexpr cuopt_int_t num_constraints = 2;

  std::array<cuopt_int_t, 3> row_offsets{0, 2, 4};
  std::array<cuopt_int_t, 4> column_indices{0, 1, 0, 1};
  std::array<cuopt_float_t, 4> values{3.0, 4.0, 2.7, 10.1};
  std::array<cuopt_float_t, 2> objective{-0.2, 0.1};
  std::array<cuopt_float_t, 2> constraint_lower{
    -CUOPT_INFINITY,
    -CUOPT_INFINITY};
  std::array<cuopt_float_t, 2> constraint_upper{5.4, 4.9};
  std::array<cuopt_float_t, 2> variable_lower{0.0, 0.0};
  std::array<cuopt_float_t, 2> variable_upper{
    CUOPT_INFINITY,
    CUOPT_INFINITY};
  std::array<char, 2> variable_types{CUOPT_INTEGER, CUOPT_CONTINUOUS};

  cuopt_int_t status = cuOptCreateRangedProblem(
    num_constraints,
    num_variables,
    CUOPT_MINIMIZE,
    0.0,
    objective.data(),
    row_offsets.data(),
    column_indices.data(),
    values.data(),
    constraint_lower.data(),
    constraint_upper.data(),
    variable_lower.data(),
    variable_upper.data(),
    variable_types.data(),
    &problem);
  REQUIRE(status == CUOPT_SUCCESS);

  status = cuOptCreateSolverSettings(&settings);
  REQUIRE(status == CUOPT_SUCCESS);

  status = cuOptSetFloatParameter(
    settings,
    CUOPT_MIP_ABSOLUTE_TOLERANCE,
    1.0e-4);
  REQUIRE(status == CUOPT_SUCCESS);

  status = cuOptSolve(problem, settings, &solution);
  REQUIRE(status == CUOPT_SUCCESS);

  cuopt_int_t termination_status = 0;
  status = cuOptGetTerminationStatus(solution, &termination_status);
  REQUIRE(status == CUOPT_SUCCESS);
  CHECK(termination_status == CUOPT_TERMINATION_STATUS_OPTIMAL);

  cuopt_float_t objective_value = 0.0;
  status = cuOptGetObjectiveValue(solution, &objective_value);
  REQUIRE(status == CUOPT_SUCCESS);
  CHECK(objective_value == doctest::Approx(-0.2).epsilon(1.0e-6));

  std::array<cuopt_float_t, 2> primal_solution{};
  status = cuOptGetPrimalSolution(solution, primal_solution.data());
  REQUIRE(status == CUOPT_SUCCESS);
  CHECK(primal_solution[0] == doctest::Approx(1.0).epsilon(1.0e-6));
  CHECK(primal_solution[1] == doctest::Approx(0.0).epsilon(1.0e-6));

  cuOptDestroySolution(&solution);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}
