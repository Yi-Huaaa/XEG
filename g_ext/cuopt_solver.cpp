#include "cuopt_solver.hpp"

#include "xeg_milp_model.hpp"

#ifdef XEG_ENABLE_CUOPT
#include <cuopt/linear_programming/cuopt_c.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
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

struct CuoptMipStart {
  std::vector<double> primal;
  double objective_value = 0.0;
  uint32_t selected_enodes = 0u;
};


CuoptMipStart build_gpu_tree_mip_start(
  const XegMilpModel& model,
  const std::vector<std::vector<uint32_t>>&
    ecls_its_ende,
  const std::vector<uint32_t>& ende_child,
  const std::vector<uint32_t>& ende_child_ptr,
  const std::vector<uint32_t>& best_ende_of_ecls,
  const std::vector<bool>& picked_endes)
{
  if (best_ende_of_ecls.size() !=
      static_cast<size_t>(model.num_eclasses)) {
    throw std::invalid_argument(
      "GPU tree representative size does not match "
      "the number of e-classes");
  }

  if (picked_endes.size() !=
      static_cast<size_t>(model.num_enodes)) {
    throw std::invalid_argument(
      "GPU tree selected-mask size does not match "
      "the number of e-nodes");
  }

  const uint32_t invalid_index =
    std::numeric_limits<uint32_t>::max();

  CuoptMipStart result;

  result.primal.assign(
    model.variables.size(),
    0.0);

  std::vector<uint32_t> enode_eclass(
    model.num_enodes,
    invalid_index);

  for (uint32_t eclass = 0u;
       eclass < model.num_eclasses;
       ++eclass) {
    for (const uint32_t enode :
         ecls_its_ende.at(eclass)) {
      if (enode >= model.num_enodes) {
        throw std::invalid_argument(
          "GPU tree references an invalid e-node");
      }

      if (enode_eclass.at(enode) !=
          invalid_index) {
        throw std::invalid_argument(
          "An e-node appears in more than one e-class");
      }

      enode_eclass.at(enode) = eclass;
    }
  }

  std::vector<uint32_t> selected_by_eclass(
    model.num_eclasses,
    invalid_index);

  std::vector<bool> active_eclasses(
    model.num_eclasses,
    false);

  uint32_t active_eclass_count = 0u;

  for (uint32_t enode = 0u;
       enode < model.num_enodes;
       ++enode) {
    if (!picked_endes.at(enode)) {
      continue;
    }

    const uint32_t selected_variable =
      model.selected_var_by_enode.at(enode);

    if (selected_variable == invalid_index) {
      throw std::invalid_argument(
        "GPU tree selected e-node is absent from "
        "the DAG search space");
    }

    const uint32_t parent_eclass =
      enode_eclass.at(enode);

    if (parent_eclass == invalid_index) {
      throw std::invalid_argument(
        "GPU tree selected e-node has no e-class");
    }

    if (best_ende_of_ecls.at(parent_eclass) !=
        enode) {
      throw std::invalid_argument(
        "GPU tree representative does not match "
        "the selected e-node");
    }

    if (selected_by_eclass.at(parent_eclass) !=
        invalid_index) {
      throw std::invalid_argument(
        "GPU tree selects more than one e-node "
        "in an e-class");
    }

    const uint32_t active_variable =
      model.active_var_by_eclass.at(parent_eclass);

    if (active_variable == invalid_index) {
      throw std::invalid_argument(
        "GPU tree selects an e-node from an "
        "inactive e-class");
    }

    selected_by_eclass.at(parent_eclass) =
      enode;

    active_eclasses.at(parent_eclass) = true;

    result.primal.at(selected_variable) = 1.0;
    result.primal.at(active_variable) = 1.0;

    ++active_eclass_count;
    ++result.selected_enodes;
  }

  std::vector<std::vector<uint32_t>>
    dependencies(model.num_eclasses);

  for (uint32_t parent_eclass = 0u;
       parent_eclass < model.num_eclasses;
       ++parent_eclass) {
    if (!active_eclasses.at(parent_eclass)) {
      continue;
    }

    const uint32_t enode =
      selected_by_eclass.at(parent_eclass);

    const uint32_t child_begin =
      ende_child_ptr.at(enode);

    const uint32_t child_end =
      ende_child_ptr.at(enode + 1u);

    for (uint32_t child_index = child_begin;
         child_index < child_end;
         ++child_index) {
      const uint32_t child_eclass =
        ende_child.at(child_index);

      if (child_eclass >= model.num_eclasses) {
        throw std::invalid_argument(
          "GPU tree selected e-node references "
          "an invalid child e-class");
      }

      if (!active_eclasses.at(child_eclass)) {
        throw std::invalid_argument(
          "GPU tree selected dependency child "
          "is not active");
      }

      dependencies.at(parent_eclass).push_back(
        child_eclass);
    }

    std::vector<uint32_t>& children =
      dependencies.at(parent_eclass);

    std::sort(
      children.begin(),
      children.end());

    children.erase(
      std::unique(
        children.begin(),
        children.end()),
      children.end());
  }

  std::vector<uint32_t> indegree(
    model.num_eclasses,
    0u);

  for (uint32_t parent_eclass = 0u;
       parent_eclass < model.num_eclasses;
       ++parent_eclass) {
    for (const uint32_t child_eclass :
         dependencies.at(parent_eclass)) {
      ++indegree.at(child_eclass);
    }
  }

  std::vector<uint32_t> ready_eclasses;

  for (uint32_t eclass = 0u;
       eclass < model.num_eclasses;
       ++eclass) {
    if (active_eclasses.at(eclass) &&
        indegree.at(eclass) == 0u) {
      ready_eclasses.push_back(eclass);
    }
  }

  std::vector<uint32_t> levels(
    model.num_eclasses,
    0u);

  size_t ready_index = 0u;
  uint32_t processed_eclasses = 0u;

  while (ready_index < ready_eclasses.size()) {
    const uint32_t parent_eclass =
      ready_eclasses.at(ready_index);

    ++ready_index;
    ++processed_eclasses;

    for (const uint32_t child_eclass :
         dependencies.at(parent_eclass)) {
      levels.at(child_eclass) =
        std::max(
          levels.at(child_eclass),
          levels.at(parent_eclass) + 1u);

      uint32_t& child_indegree =
        indegree.at(child_eclass);

      --child_indegree;

      if (child_indegree == 0u) {
        ready_eclasses.push_back(child_eclass);
      }
    }
  }

  if (processed_eclasses !=
      active_eclass_count) {
    throw std::invalid_argument(
      "GPU tree selected e-nodes contain a cycle");
  }

  for (uint32_t eclass = 0u;
       eclass < model.num_eclasses;
       ++eclass) {
    const uint32_t level_variable =
      model.level_var_by_eclass.at(eclass);

    if (level_variable == invalid_index) {
      throw std::runtime_error(
        "XEG MILP model has no level variable "
        "for an e-class");
    }

    result.primal.at(level_variable) =
      static_cast<double>(levels.at(eclass));
  }

  for (size_t variable = 0u;
       variable < model.variables.size();
       ++variable) {
    result.objective_value +=
      model.variables.at(variable).
        objective_coefficient *
      result.primal.at(variable);
  }

  const XegMilpValidationResult validation =
    validate_xeg_milp_solution(
      model,
      result.primal,
      result.objective_value);

  if (!validation.valid) {
    throw std::invalid_argument(
      "Invalid cuOpt GPU-tree MIP start: " +
      validation.message);
  }

  return result;
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
  const std::vector<bool>& picked_endes,
  const CuoptMode cuopt_mode)
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
  (void)cuopt_mode;

  throw std::runtime_error(
    "XEG was built without cuOpt support");
#else
  // The warm-start inputs are required only in
  // GPU-tree warm-start mode.
  switch (cuopt_mode) {
    case CuoptMode::Presolve:
      break;

    case CuoptMode::GpuTreeWarmStart:
      if (picked_endes.size() !=
          ende_cost.size()) {
        throw std::invalid_argument(
          "GPU tree selected-mask size does not match "
          "the number of e-nodes");
      }

      if (best_ende_of_ecls.size() !=
          ecls_its_ende.size()) {
        throw std::invalid_argument(
          "GPU tree representative size does not match "
          "the number of e-classes");
      }

      break;

    default:
      throw std::invalid_argument(
        "Invalid cuOpt mode");
  }

  const XegMilpModel model =
    build_xeg_milp_model(
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost);

  CuoptMipStart mip_start;

  if (cuopt_mode ==
      CuoptMode::GpuTreeWarmStart) {
    mip_start =
      build_gpu_tree_mip_start(
        model,
        ecls_its_ende,
        ende_child,
        ende_child_ptr,
        best_ende_of_ecls,
        picked_endes);
  }

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

  CuoptSolveResult result;

  std::vector<cuopt_float_t>
    cuopt_mip_start;

  if (cuopt_mode ==
      CuoptMode::GpuTreeWarmStart) {
    check_cuopt_status(
      cuOptSetIntegerParameter(
        resources.settings,
        CUOPT_PRESOLVE,
        0),
      "cuOptSetIntegerParameter(CUOPT_PRESOLVE)");

    cuopt_mip_start.assign(
      mip_start.primal.begin(),
      mip_start.primal.end());

    check_cuopt_status(
      cuOptAddMIPStart(
        resources.settings,
        cuopt_mip_start.data(),
        num_variables),
      "cuOptAddMIPStart");

    result.mip_start_submitted = true;
    result.mip_start_objective =
      mip_start.objective_value;
  }

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

  if (cuopt_mode ==
      CuoptMode::GpuTreeWarmStart) {
    std::printf(
      "[cuOpt Mode] presolve=off, "
      "mip_start=GPU-tree, "
      "objective=%.0f, "
      "selected_enodes=%u\n",
      mip_start.objective_value,
      mip_start.selected_enodes);
  }
  else {
    std::printf(
      "[cuOpt Mode] presolve=on, "
      "mip_start=off\n");
  }

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