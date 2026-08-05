#include "xeg_milp_model.hpp"

#include <limits>
#include <utility>

XegMilpModel build_xeg_milp_model(
  const std::vector<std::vector<uint32_t>>& ecls_its_ende,
  const std::vector<uint32_t>& ende_child,
  const std::vector<uint32_t>& ende_child_ptr,
  const std::vector<uint32_t>& root_eclass_id,
  const std::vector<float>& ende_cost)
{
  // These inputs are used by later formulation steps.
  // The first test contains no dependency edges.
  (void)ende_child;
  (void)ende_child_ptr;

  XegMilpModel model;

  model.num_eclasses =
    static_cast<uint32_t>(ecls_its_ende.size());

  model.num_enodes =
    static_cast<uint32_t>(ende_cost.size());

  const uint32_t invalid_variable =
    std::numeric_limits<uint32_t>::max();

  model.active_var_by_eclass.assign(
    model.num_eclasses,
    invalid_variable);

  model.selected_var_by_enode.assign(
    model.num_enodes,
    invalid_variable);

  model.level_var_by_eclass.assign(
    model.num_eclasses,
    invalid_variable);

  // Active variables exist only for alive e-classes, matching
  // the current CP-SAT formulation.
  for (uint32_t cid = 0u;
       cid < model.num_eclasses;
       ++cid) {
    if (ecls_its_ende[cid].empty()) {
      continue;
    }

    model.active_var_by_eclass[cid] =
      static_cast<uint32_t>(model.variables.size());

    model.variables.push_back({
      XegMilpVariableKind::ActiveEClass,
      cid,
      0.0,
      1.0,
      0.0,
      true,
    });
  }

  // Selected e-node variables.
  for (uint32_t cid = 0u;
       cid < model.num_eclasses;
       ++cid) {
    for (const uint32_t ende : ecls_its_ende[cid]) {
      model.selected_var_by_enode.at(ende) =
        static_cast<uint32_t>(model.variables.size());

      model.variables.push_back({
        XegMilpVariableKind::SelectedENode,
        ende,
        0.0,
        1.0,
        static_cast<double>(
          static_cast<int>(ende_cost.at(ende))),
        true,
      });
    }
  }

  // CP-SAT currently creates one level variable for every
  // e-class, including dead e-classes.
  for (uint32_t cid = 0u;
       cid < model.num_eclasses;
       ++cid) {
    model.level_var_by_eclass[cid] =
      static_cast<uint32_t>(model.variables.size());

    model.variables.push_back({
      XegMilpVariableKind::Level,
      cid,
      0.0,
      static_cast<double>(model.num_eclasses),
      0.0,
      true,
    });
  }

  // For each alive e-class:
  //
  //   sum(selected e-nodes) - active[e-class] = 0
  for (uint32_t cid = 0u;
       cid < model.num_eclasses;
       ++cid) {
    if (ecls_its_ende[cid].empty()) {
      continue;
    }

    XegMilpConstraint constraint;

    constraint.lower_bound = 0.0;
    constraint.upper_bound = 0.0;

    constraint.terms.push_back({
      model.active_var_by_eclass.at(cid),
      -1.0,
    });

    for (const uint32_t ende : ecls_its_ende[cid]) {
      constraint.terms.push_back({
        model.selected_var_by_enode.at(ende),
        1.0,
      });
    }

    model.constraints.push_back(
      std::move(constraint));
  }

  // Root e-classes must be active.
  for (const uint32_t root : root_eclass_id) {
    XegMilpConstraint constraint;

    constraint.lower_bound = 1.0;
    constraint.upper_bound = 1.0;

    constraint.terms.push_back({
      model.active_var_by_eclass.at(root),
      1.0,
    });

    model.constraints.push_back(
      std::move(constraint));
  }

  return model;
}