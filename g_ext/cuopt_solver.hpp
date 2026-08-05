#ifndef XEG_CUOPT_SOLVER_HPP
#define XEG_CUOPT_SOLVER_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

enum class CuoptMode : uint32_t {
  Presolve = 0u,
  GpuTreeWarmStart = 1u,
};


inline CuoptMode parse_cuopt_mode(
  const uint32_t value)
{
  switch (value) {
    case 0u:
      return CuoptMode::Presolve;

    case 1u:
      return CuoptMode::GpuTreeWarmStart;

    default:
      throw std::invalid_argument(
        "Invalid CUOPT_MODE value: " +
        std::to_string(value));
  }
}


inline const char* cuopt_mode_name(
  const CuoptMode mode)
{
  switch (mode) {
    case CuoptMode::Presolve:
      return "Presolve";

    case CuoptMode::GpuTreeWarmStart:
      return "GPU-tree warm start";

    default:
      throw std::invalid_argument(
        "Invalid cuOpt mode");
  }
}

struct CuoptSolveResult {
  bool has_feasible_solution = false;
  double objective_value = 0.0;
  double recomputed_objective = 0.0;

  bool mip_start_submitted = false;
  double mip_start_objective = 0.0;

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
  const std::vector<bool>& picked_endes,
  CuoptMode cuopt_mode = CuoptMode::Presolve);


#endif