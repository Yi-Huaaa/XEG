#ifndef XEG_MILP_MODEL_HPP
#define XEG_MILP_MODEL_HPP

#include <cstdint>
#include <string>
#include <vector>


enum class XegMilpVariableKind : uint32_t {
  ActiveEClass = 0u,
  SelectedENode = 1u,
  Level = 2u,
};


struct XegMilpVariable {
  XegMilpVariableKind kind;
  uint32_t source_id;

  double lower_bound;
  double upper_bound;
  double objective_coefficient;

  bool is_integer;
};


struct XegMilpTerm {
  uint32_t variable;
  double coefficient;
};


struct XegMilpConstraint {
  std::vector<XegMilpTerm> terms;

  double lower_bound;
  double upper_bound;
};


struct XegMilpModel {
  uint32_t num_eclasses = 0u;
  uint32_t num_enodes = 0u;

  std::vector<XegMilpVariable> variables;
  std::vector<XegMilpConstraint> constraints;

  std::vector<uint32_t> active_var_by_eclass;
  std::vector<uint32_t> selected_var_by_enode;
  std::vector<uint32_t> level_var_by_eclass;
};


struct XegMilpValidationResult {
  bool valid = false;
  double recomputed_objective = 0.0;
  std::string message;
};


XegMilpModel build_xeg_milp_model(
  const std::vector<std::vector<uint32_t>>& ecls_its_ende,
  const std::vector<uint32_t>& ende_child,
  const std::vector<uint32_t>& ende_child_ptr,
  const std::vector<uint32_t>& root_eclass_id,
  const std::vector<float>& ende_cost);


XegMilpValidationResult validate_xeg_milp_solution(
  const XegMilpModel& model,
  const std::vector<double>& primal_solution,
  double reported_objective,
  double tolerance = 1.0e-6);


#endif