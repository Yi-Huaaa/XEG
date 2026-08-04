#include "cuopt_solver.hpp"

#include <stdexcept>


void solve_with_cuopt(
  const double time_limit,
  const std::vector<std::vector<uint32_t>>& ecls_its_ende,
  const std::vector<uint32_t>& ende_child,
  const std::vector<uint32_t>& ende_child_ptr,
  const std::vector<uint32_t>& root_eclass_id,
  const std::vector<float>& ende_cost,
  const std::vector<uint32_t>& best_ende_of_ecls,
  const std::vector<bool>& picked_endes)
{
  // These inputs are intentionally unused until the XEG MILP
  // formulation is translated to cuOpt in the next patch.
  (void)time_limit;
  (void)ecls_its_ende;
  (void)ende_child;
  (void)ende_child_ptr;
  (void)root_eclass_id;
  (void)ende_cost;
  (void)best_ende_of_ecls;
  (void)picked_endes;

#ifdef XEG_ENABLE_CUOPT
  throw std::runtime_error(
    "cuOpt XEG formulation is not implemented in this patch");
#else
  throw std::runtime_error(
    "XEG was built without cuOpt support");
#endif
}