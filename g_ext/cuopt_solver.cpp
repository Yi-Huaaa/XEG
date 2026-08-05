#include "cuopt_solver.hpp"

#include "xeg_milp_model.hpp"

#ifdef XEG_ENABLE_CUOPT
#include <cuopt/linear_programming/cuopt_c.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


#ifdef XEG_ENABLE_CUOPT

namespace {

void check_cuopt_status(
  const cuopt_int_t status,
  const char* operation)
{
  if (status != CUOPT_SUCCESS) {
    throw std::runtime_error(
      std::string(operation) +
      " failed with cuOpt status " +
      std::to_string(status));
  }
}


cuopt_int_t to_cuopt_int(
  const size_t value,
  const char* description)
{
  if (value >
      static_cast<size_t>(
        std::numeric_limits<cuopt_int_t>::max())) {
    throw std::runtime_error(
      std::string(description) +
      " exceeds cuOpt integer range");
  }

  return static_cast<cuopt_int_t>(value);
}


cuopt_float_t to_cuopt_bound(
  const double bound)
{
  if (std::isinf(bound)) {
    return bound > 0.0
      ? CUOPT_INFINITY
      : -CUOPT_INFINITY;
  }

  return static_cast<cuopt_float_t>(bound);
}


struct CuoptResources {
  cuOptOptimizationProblem problem = nullptr;
  cuOptSolverSettings settings = nullptr;
  cuOptSolution solution = nullptr;

  ~CuoptResources()
  {
    if (solution != nullptr) {
      cuOptDestroySolution(&solution);
    }

    if (settings != nullptr) {
      cuOptDestroySolverSettings(&settings);
    }

    if (problem != nullptr) {
      cuOptDestroyProblem(&problem);
    }
  }
};


bool termination_may_have_solution(
  const cuopt_int_t termination_status)
{
  return
    termination_status ==
      CUOPT_TERMINATION_STATUS_OPTIMAL ||
    termination_status ==
      CUOPT_TERMINATION_STATUS_PRIMAL_FEASIBLE ||
    termination_status ==
      CUOPT_TERMINATION_STATUS_FEASIBLE_FOUND ||
    termination_status ==
      CUOPT_TERMINATION_STATUS_TIME_LIMIT ||
    termination_status ==
      CUOPT_TERMINATION_STATUS_ITERATION_LIMIT ||
    termination_status ==
      CUOPT_TERMINATION_STATUS_CONCURRENT_LIMIT ||
    termination_status ==
      CUOPT_TERMINATION_STATUS_WORK_LIMIT;
}


bool termination_is_limit(
  const cuopt_int_t termination_status)
{
  return
    termination_status ==
      CUOPT_TERMINATION_STATUS_TIME_LIMIT ||
    termination_status ==
      CUOPT_TERMINATION_STATUS_ITERATION_LIMIT ||
    termination_status ==
      CUOPT_TERMINATION_STATUS_CONCURRENT_LIMIT ||
    termination_status ==
      CUOPT_TERMINATION_STATUS_WORK_LIMIT;
}

}  // namespace

#endif


CuoptSolveResult solve_with_cuopt(
  const double time_limit,
  const std::vector<std::vector<uint32_t>>& ecls_its_ende,
  const std::vector<uint32_t>& ende_child,
  const std::vector<uint32_t>& ende_child_ptr,
  const std::vector<uint32_t>& root_eclass_id,
  const std::vector<float>& ende_cost,
  const std::vector<uint32_t>& best_ende_of_ecls,
  const std::vector<bool>& picked_endes)
{
#ifndef XEG_ENABLE_CUOPT
  (void)time_limit;
  (void)ecls_its_ende;
  (void)ende_child;
  (void)ende_child_ptr;
  (void)root_eclass_id;
  (void)ende_cost;
  (void)best_ende_of_ecls;
  (void)picked_endes;

  throw std::runtime_error(
    "XEG was built without cuOpt support");
#else
  // Patch 1C-A intentionally does not use warm-start data.
  (void)best_ende_of_ecls;
  (void)picked_endes;

  const XegMilpModel model =
    build_xeg_milp_model(
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost);

  const cuopt_int_t num_variables =
    to_cuopt_int(
      model.variables.size(),
      "number of variables");

  const cuopt_int_t num_constraints =
    to_cuopt_int(
      model.constraints.size(),
      "number of constraints");

  std::vector<cuopt_float_t> objective_coefficients;
  std::vector<cuopt_float_t> variable_lower_bounds;
  std::vector<cuopt_float_t> variable_upper_bounds;
  std::vector<char> variable_types;

  objective_coefficients.reserve(model.variables.size());
  variable_lower_bounds.reserve(model.variables.size());
  variable_upper_bounds.reserve(model.variables.size());
  variable_types.reserve(model.variables.size());

  for (const XegMilpVariable& variable :
       model.variables) {
    objective_coefficients.push_back(
      static_cast<cuopt_float_t>(
        variable.objective_coefficient));

    variable_lower_bounds.push_back(
      to_cuopt_bound(variable.lower_bound));

    variable_upper_bounds.push_back(
      to_cuopt_bound(variable.upper_bound));

    variable_types.push_back(
      variable.is_integer
        ? CUOPT_INTEGER
        : CUOPT_CONTINUOUS);
  }

  std::vector<cuopt_int_t> row_offsets;
  std::vector<cuopt_int_t> column_indices;
  std::vector<cuopt_float_t> matrix_coefficients;
  std::vector<cuopt_float_t> constraint_lower_bounds;
  std::vector<cuopt_float_t> constraint_upper_bounds;

  row_offsets.reserve(model.constraints.size() + 1u);
  constraint_lower_bounds.reserve(model.constraints.size());
  constraint_upper_bounds.reserve(model.constraints.size());

  row_offsets.push_back(0);

  for (const XegMilpConstraint& constraint :
       model.constraints) {
    std::vector<std::pair<uint32_t, double>> terms;

    terms.reserve(constraint.terms.size());

    for (const XegMilpTerm& term : constraint.terms) {
      if (term.variable >= model.variables.size()) {
        throw std::runtime_error(
          "XEG MILP constraint references an invalid variable");
      }

      terms.emplace_back(
        term.variable,
        term.coefficient);
    }

    std::sort(
      terms.begin(),
      terms.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
      });

    size_t term_index = 0u;

    while (term_index < terms.size()) {
      const uint32_t variable = terms[term_index].first;
      double coefficient = 0.0;

      while (term_index < terms.size() &&
             terms[term_index].first == variable) {
        coefficient += terms[term_index].second;
        ++term_index;
      }

      if (coefficient == 0.0) {
        continue;
      }

      column_indices.push_back(
        to_cuopt_int(variable, "variable index"));

      matrix_coefficients.push_back(
        static_cast<cuopt_float_t>(coefficient));
    }

    row_offsets.push_back(
      to_cuopt_int(
        matrix_coefficients.size(),
        "number of matrix nonzeros"));

    constraint_lower_bounds.push_back(
      to_cuopt_bound(constraint.lower_bound));

    constraint_upper_bounds.push_back(
      to_cuopt_bound(constraint.upper_bound));
  }

  CuoptResources resources;

  check_cuopt_status(
    cuOptCreateRangedProblem(
      num_constraints,
      num_variables,
      CUOPT_MINIMIZE,
      0.0,
      objective_coefficients.data(),
      row_offsets.data(),
      column_indices.data(),
      matrix_coefficients.data(),
      constraint_lower_bounds.data(),
      constraint_upper_bounds.data(),
      variable_lower_bounds.data(),
      variable_upper_bounds.data(),
      variable_types.data(),
      &resources.problem),
    "cuOptCreateRangedProblem");

  check_cuopt_status(
    cuOptCreateSolverSettings(&resources.settings),
    "cuOptCreateSolverSettings");

  if (time_limit > 0.0) {
    check_cuopt_status(
      cuOptSetFloatParameter(
        resources.settings,
        CUOPT_TIME_LIMIT,
        static_cast<cuopt_float_t>(time_limit)),
      "cuOptSetFloatParameter(CUOPT_TIME_LIMIT)");
  }

  check_cuopt_status(
    cuOptSetIntegerParameter(
      resources.settings,
      CUOPT_RANDOM_SEED,
      42),
    "cuOptSetIntegerParameter(CUOPT_RANDOM_SEED)");

  check_cuopt_status(
    cuOptSolve(
      resources.problem,
      resources.settings,
      &resources.solution),
    "cuOptSolve");

  cuopt_int_t termination_status =
    CUOPT_TERMINATION_STATUS_NO_TERMINATION;

  check_cuopt_status(
    cuOptGetTerminationStatus(
      resources.solution,
      &termination_status),
    "cuOptGetTerminationStatus");

  CuoptSolveResult result;

  if (termination_status ==
        CUOPT_TERMINATION_STATUS_INFEASIBLE ||
      termination_status ==
        CUOPT_TERMINATION_STATUS_UNBOUNDED ||
      termination_status ==
        CUOPT_TERMINATION_STATUS_UNBOUNDED_OR_INFEASIBLE) {
    return result;
  }

  if (!termination_may_have_solution(
        termination_status)) {
    throw std::runtime_error(
      "cuOpt terminated without a usable solution; status " +
      std::to_string(termination_status));
  }

  cuopt_float_t objective_value = 0.0;

  const cuopt_int_t objective_status =
    cuOptGetObjectiveValue(
      resources.solution,
      &objective_value);

  if (objective_status != CUOPT_SUCCESS) {
    if (termination_is_limit(termination_status)) {
      return result;
    }

    check_cuopt_status(
      objective_status,
      "cuOptGetObjectiveValue");
  }

  std::vector<cuopt_float_t>
    cuopt_primal(model.variables.size(), 0.0);

  const cuopt_int_t primal_status =
    cuOptGetPrimalSolution(
      resources.solution,
      cuopt_primal.data());

  if (primal_status != CUOPT_SUCCESS) {
    if (termination_is_limit(termination_status)) {
      return result;
    }

    check_cuopt_status(
      primal_status,
      "cuOptGetPrimalSolution");
  }

  std::vector<double> primal_solution;
  primal_solution.reserve(cuopt_primal.size());

  for (const cuopt_float_t value : cuopt_primal) {
    primal_solution.push_back(
      static_cast<double>(value));
  }

  const XegMilpValidationResult validation =
    validate_xeg_milp_solution(
      model,
      primal_solution,
      static_cast<double>(objective_value));

  if (!validation.valid) {
    throw std::runtime_error(
      "cuOpt returned an invalid XEG solution: " +
      validation.message);
  }

  result.has_feasible_solution = true;
  result.objective_value =
    static_cast<double>(objective_value);

  result.recomputed_objective =
    validation.recomputed_objective;

  const uint32_t invalid_variable =
    std::numeric_limits<uint32_t>::max();

  for (uint32_t enode = 0u;
       enode < model.selected_var_by_enode.size();
       ++enode) {
    const uint32_t variable =
      model.selected_var_by_enode.at(enode);

    if (variable == invalid_variable) {
      continue;
    }

    if (primal_solution.at(variable) > 0.5) {
      result.selected_enodes.push_back(enode);
    }
  }

  return result;
#endif
}