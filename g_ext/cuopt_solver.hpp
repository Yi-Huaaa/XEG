#ifndef XEG_CUOPT_SOLVER_HPP
#define XEG_CUOPT_SOLVER_HPP

#include <cstdint>
#include <vector>

struct CuoptSolveResult {
  bool has_feasible_solution = false;
  double objective_value = 0.0;
  double recomputed_objective = 0.0;
  std::vector<uint32_t> selected_enodes;
};

CuoptSolveResult solve_with_cuopt(
  double time_limit,
  const std::vector<std::vector<uint32_t>>& ecls_its_ende,
  const std::vector<uint32_t>& ende_child,
  const std::vector<uint32_t>& ende_child_ptr,
  const std::vector<uint32_t>& root_eclass_id,
  const std::vector<float>& ende_cost,
  const std::vector<uint32_t>& best_ende_of_ecls,
  const std::vector<bool>& picked_endes);


#endif