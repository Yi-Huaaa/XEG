#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <g_ext/xeg_milp_model.hpp>

#include <cstdint>
#include <vector>
#include <limits>

namespace {

double coefficient_of(
  const XegMilpConstraint& constraint,
  const uint32_t variable)
{
  for (const XegMilpTerm& term : constraint.terms) {
    if (term.variable == variable) {
      return term.coefficient;
    }
  }

  return 0.0;
}

}  // namespace


DOCTEST_TEST_CASE(
  "XEG MILP model builds a rooted alternative choice")
{
  // E-class 0 is the root and contains two leaf e-nodes:
  //
  //   e-node 0: cost 5.9 -> current CP-SAT semantics use 5
  //   e-node 1: cost 2.1 -> current CP-SAT semantics use 2
  //
  // The model must activate e-class 0 and select exactly one
  // of the two e-nodes.
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u, 1u},
    };

  const std::vector<uint32_t>
    ende_child;

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      0u,
      0u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      5.9f,
      2.1f,
    };

  const XegMilpModel model =
    build_xeg_milp_model(
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost);

  DOCTEST_CHECK_EQ(
    model.num_eclasses,
    1u);

  DOCTEST_CHECK_EQ(
    model.num_enodes,
    2u);

  // active[0], selected[0], selected[1], level[0]
  DOCTEST_CHECK_EQ(
    model.variables.size(),
    4u);

  // One selection equality and one root equality.
  DOCTEST_CHECK_EQ(
    model.constraints.size(),
    2u);

  const uint32_t active_0 =
    model.active_var_by_eclass.at(0u);

  const uint32_t selected_0 =
    model.selected_var_by_enode.at(0u);

  const uint32_t selected_1 =
    model.selected_var_by_enode.at(1u);

  const uint32_t level_0 =
    model.level_var_by_eclass.at(0u);

  DOCTEST_CHECK_EQ(
    model.variables.at(active_0).kind,
    XegMilpVariableKind::ActiveEClass);

  DOCTEST_CHECK_EQ(
    model.variables.at(selected_0).kind,
    XegMilpVariableKind::SelectedENode);

  DOCTEST_CHECK_EQ(
    model.variables.at(selected_1).kind,
    XegMilpVariableKind::SelectedENode);

  DOCTEST_CHECK_EQ(
    model.variables.at(level_0).kind,
    XegMilpVariableKind::Level);

  DOCTEST_CHECK_EQ(
    model.variables.at(selected_0).objective_coefficient,
    5.0);

  DOCTEST_CHECK_EQ(
    model.variables.at(selected_1).objective_coefficient,
    2.0);

  const XegMilpConstraint& selection =
    model.constraints.at(0u);

  DOCTEST_CHECK_EQ(
    selection.lower_bound,
    0.0);

  DOCTEST_CHECK_EQ(
    selection.upper_bound,
    0.0);

  DOCTEST_CHECK_EQ(
    coefficient_of(selection, active_0),
    -1.0);

  DOCTEST_CHECK_EQ(
    coefficient_of(selection, selected_0),
    1.0);

  DOCTEST_CHECK_EQ(
    coefficient_of(selection, selected_1),
    1.0);

  const XegMilpConstraint& root =
    model.constraints.at(1u);

  DOCTEST_CHECK_EQ(
    root.lower_bound,
    1.0);

  DOCTEST_CHECK_EQ(
    root.upper_bound,
    1.0);

  DOCTEST_CHECK_EQ(
    coefficient_of(root, active_0),
    1.0);
}

DOCTEST_TEST_CASE(
  "selected e-node activates its child e-class")
{
  // e-class 0:
  //   e-node 0 -> child e-class 1
  //
  // e-class 1:
  //   e-node 1 is a leaf
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u},
      {1u},
    };

  const std::vector<uint32_t>
    ende_child{
      1u,
    };

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      1u,
      1u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      3.0f,
      2.0f,
    };

  const XegMilpModel model =
    build_xeg_milp_model(
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost);

  // Expected constraint order:
  //   0: selection for e-class 0
  //   1: selection for e-class 1
  //   2: selected[0] <= active[1]
  //   3: active[0] = 1
  //   4: selected[0] =>
  //        level[1] - level[0] >= 1
  DOCTEST_REQUIRE_EQ(
    model.constraints.size(),
    5u);

  const uint32_t selected_0 =
    model.selected_var_by_enode.at(0u);

  const uint32_t active_1 =
    model.active_var_by_eclass.at(1u);

  const XegMilpConstraint& dependency =
    model.constraints.at(2u);

  DOCTEST_CHECK_EQ(
    dependency.lower_bound,
    -std::numeric_limits<double>::infinity());

  DOCTEST_CHECK_EQ(
    dependency.upper_bound,
    0.0);

  // selected[0] - active[1] <= 0
  DOCTEST_CHECK_EQ(
    coefficient_of(dependency, selected_0),
    1.0);

  DOCTEST_CHECK_EQ(
    coefficient_of(dependency, active_1),
    -1.0);

  const uint32_t level_0 =
    model.level_var_by_eclass.at(0u);

  const uint32_t level_1 =
    model.level_var_by_eclass.at(1u);

  const XegMilpConstraint& acyclic =
    model.constraints.at(4u);

  // num_eclasses = 2, so M = 3:
  //
  // level[1] - level[0] - 3 * selected[0] >= -2
  DOCTEST_CHECK_EQ(
    acyclic.lower_bound,
    -2.0);

  DOCTEST_CHECK_EQ(
    acyclic.upper_bound,
    std::numeric_limits<double>::infinity());

  DOCTEST_CHECK_EQ(
    coefficient_of(acyclic, level_1),
    1.0);

  DOCTEST_CHECK_EQ(
    coefficient_of(acyclic, level_0),
    -1.0);

  DOCTEST_CHECK_EQ(
    coefficient_of(acyclic, selected_0),
    -3.0);
}

DOCTEST_TEST_CASE(
  "e-node with a dead child cannot be selected")
{
  // e-class 0:
  //   e-node 0 -> child e-class 1
  //
  // e-class 1 is dead: it contains no candidate e-node.
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u},
      {},
    };

  const std::vector<uint32_t>
    ende_child{
      1u,
    };

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      1u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      1.0f,
    };

  const XegMilpModel model =
    build_xeg_milp_model(
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost);

  // Expected constraint order:
  //   0: selection for e-class 0
  //   1: selected[0] = 0 because its child is dead
  //   2: active[0] = 1
  DOCTEST_REQUIRE_EQ(
    model.constraints.size(),
    3u);

  const uint32_t selected_0 =
    model.selected_var_by_enode.at(0u);

  const XegMilpConstraint& disabled =
    model.constraints.at(1u);

  DOCTEST_CHECK_EQ(
    disabled.lower_bound,
    0.0);

  DOCTEST_CHECK_EQ(
    disabled.upper_bound,
    0.0);

  DOCTEST_CHECK_EQ(
    coefficient_of(disabled, selected_0),
    1.0);
}

DOCTEST_TEST_CASE(
  "validator accepts a feasible rooted alternative solution")
{
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u, 1u},
    };

  const std::vector<uint32_t> ende_child;

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      0u,
      0u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      5.9f,
      2.1f,
    };

  const XegMilpModel model =
    build_xeg_milp_model(
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost);

  // Variable order:
  //   active[0]   = 1
  //   selected[0] = 0
  //   selected[1] = 1
  //   level[0]    = 0
  const std::vector<double>
    primal_solution{
      1.0,
      0.0,
      1.0,
      0.0,
    };

  const XegMilpValidationResult validation =
    validate_xeg_milp_solution(
      model,
      primal_solution,
      2.0);

  DOCTEST_CHECK(validation.valid);

  DOCTEST_CHECK_EQ(
    validation.recomputed_objective,
    2.0);
}

DOCTEST_TEST_CASE(
  "validator rejects a variable outside its bounds")
{
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u, 1u},
    };

  const std::vector<uint32_t> ende_child;

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      0u,
      0u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      5.0f,
      2.0f,
    };

  const XegMilpModel model =
    build_xeg_milp_model(
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost);

  // active[0] = 2 violates its [0, 1] bound.
  // The reported objective still matches the selected e-node,
  // so the failure must come from bounds validation.
  const std::vector<double>
    primal_solution{
      2.0,
      0.0,
      1.0,
      0.0,
    };

  const XegMilpValidationResult validation =
    validate_xeg_milp_solution(
      model,
      primal_solution,
      2.0);

  DOCTEST_CHECK_FALSE(validation.valid);
}

DOCTEST_TEST_CASE(
  "validator rejects a fractional integer variable")
{
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u, 1u},
    };

  const std::vector<uint32_t> ende_child;

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      0u,
      0u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      5.0f,
      2.0f,
    };

  const XegMilpModel model =
    build_xeg_milp_model(
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost);

  // active[0] = 1
  // selected[0] = 0.5
  // selected[1] = 0.5
  // level[0] = 0
  //
  // All values are within bounds and the objective is correct,
  // but selected variables must be integer.
  const std::vector<double>
    primal_solution{
      1.0,
      0.5,
      0.5,
      0.0,
    };

  const XegMilpValidationResult validation =
    validate_xeg_milp_solution(
      model,
      primal_solution,
      3.5);

  DOCTEST_CHECK_FALSE(validation.valid);
}

DOCTEST_TEST_CASE(
  "validator rejects a violated MILP constraint")
{
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u, 1u},
    };

  const std::vector<uint32_t> ende_child;

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      0u,
      0u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      5.0f,
      2.0f,
    };

  const XegMilpModel model =
    build_xeg_milp_model(
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost);

  // active[0] = 1, but neither e-node is selected.
  //
  // Bounds and integrality are valid, but this violates:
  //
  //   selected[0] + selected[1] - active[0] = 0
  const std::vector<double>
    primal_solution{
      1.0,
      0.0,
      0.0,
      0.0,
    };

  const XegMilpValidationResult validation =
    validate_xeg_milp_solution(
      model,
      primal_solution,
      0.0);

  DOCTEST_CHECK_FALSE(validation.valid);
}