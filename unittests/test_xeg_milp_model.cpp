#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <g_ext/xeg_milp_model.hpp>

#include <cstdint>
#include <vector>


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