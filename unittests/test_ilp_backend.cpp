#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <g_ext/cuopt_solver.hpp>
#include <g_ext/g_ext.cuh>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


DOCTEST_TEST_CASE(
  "ILP backend value zero selects CP-SAT")
{
  const IlpBackend backend =
    parse_ilp_backend(0u);

  const bool is_cpsat =
    backend == IlpBackend::CpSat;

  DOCTEST_CHECK_EQ(
    is_cpsat,
    true);
}


DOCTEST_TEST_CASE(
  "ILP backend value one selects cuOpt")
{
  const IlpBackend backend =
    parse_ilp_backend(1u);

  const bool is_cuopt =
    backend == IlpBackend::CuOpt;

  DOCTEST_CHECK_EQ(
    is_cuopt,
    true);
}


DOCTEST_TEST_CASE(
  "Invalid ILP backend values are rejected")
{
  DOCTEST_CHECK_THROWS_WITH_AS(
    parse_ilp_backend(2u),
    "Invalid ILP_BACKEND value: 2",
    std::invalid_argument);
}


DOCTEST_TEST_CASE(
  "ILP backend names are stable")
{
  const std::string cpsat_name =
    ilp_backend_name(
      IlpBackend::CpSat);

  const std::string cuopt_name =
    ilp_backend_name(
      IlpBackend::CuOpt);

  DOCTEST_CHECK_EQ(
    cpsat_name,
    "CP-SAT");

  DOCTEST_CHECK_EQ(
    cuopt_name,
    "cuOpt");
}


DOCTEST_TEST_CASE(
  "G_EXT run accepts a trailing ILP backend")
{
  using RunFunction =
    void (G_EXT::*)(
      bool,
      double,
      DagKernel,
      uint32_t,
      uint32_t,
      IlpBackend);

  constexpr bool has_expected_signature =
    std::is_same_v<
      decltype(&G_EXT::run),
      RunFunction>;

  DOCTEST_CHECK_EQ(
    has_expected_signature,
    true);
}


DOCTEST_TEST_CASE(
  "G_EXT XEG-T experiment accepts a trailing ILP backend")
{
  using RunXegTFunction =
    void (G_EXT::*)(
      double,
      DagKernel,
      uint32_t,
      uint32_t,
      uint32_t,
      IlpBackend);

  constexpr bool has_expected_signature =
    std::is_same_v<
      decltype(&G_EXT::run_xeg_t_exp),
      RunXegTFunction>;

  DOCTEST_CHECK_EQ(
    has_expected_signature,
    true);
}


DOCTEST_TEST_CASE(
  "cuOpt solves a rooted alternative choice")
{
#ifdef XEG_ENABLE_CUOPT
  // Root e-class 0 contains two leaf e-nodes.
  // The optimal choice is e-node 1 with cost 2.
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
      5.0f,
      2.0f,
    };

  // Patch 1C-A intentionally ignores warm-start data.
  const std::vector<uint32_t>
    best_ende_of_ecls{
      0u,
    };

  const std::vector<bool>
    picked_endes{
      true,
      false,
    };

  const CuoptSolveResult result =
    solve_with_cuopt(
      10.0,
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost,
      best_ende_of_ecls,
      picked_endes);

  DOCTEST_CHECK(
    result.has_feasible_solution);

  DOCTEST_CHECK_EQ(
    result.objective_value,
    doctest::Approx(2.0));

  DOCTEST_CHECK_EQ(
    result.recomputed_objective,
    doctest::Approx(2.0));

  DOCTEST_REQUIRE_EQ(
    result.selected_enodes.size(),
    1u);

  DOCTEST_CHECK_EQ(
    result.selected_enodes.at(0u),
    1u);
#else
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende;

  const std::vector<uint32_t>
    ende_child;

  const std::vector<uint32_t>
    ende_child_ptr;

  const std::vector<uint32_t>
    root_eclass_id;

  const std::vector<float>
    ende_cost;

  const std::vector<uint32_t>
    best_ende_of_ecls;

  const std::vector<bool>
    picked_endes;

  DOCTEST_CHECK_THROWS_WITH_AS(
    solve_with_cuopt(
      1.0,
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost,
      best_ende_of_ecls,
      picked_endes),
    "XEG was built without cuOpt support",
    std::runtime_error);
#endif
}


DOCTEST_TEST_CASE(
  "G_EXT dispatcher routes cuOpt backend to cuOpt solver boundary")
{
  G_EXT extension;

#ifdef XEG_ENABLE_CUOPT
  DOCTEST_CHECK_NOTHROW(
    extension.unittest_solve_ilp_backend(
      IlpBackend::CuOpt));
#else
  DOCTEST_CHECK_THROWS_WITH_AS(
    extension.unittest_solve_ilp_backend(
      IlpBackend::CuOpt),
    "XEG was built without cuOpt support",
    std::runtime_error);
#endif
}


#ifdef XEG_ENABLE_CUOPT

DOCTEST_TEST_CASE(
  "cuOpt avoids a cyclic solution")
{
  // e-class 0 is the root:
  //   e-node 0 -> e-class 1, cost 0
  //   e-node 1 -> leaf,      cost 5
  //
  // e-class 1:
  //   e-node 2 -> e-class 0, cost 0
  //   e-node 3 -> leaf,      cost 1
  //
  // Selecting e-node 0 and e-node 2 would create a cycle
  // with objective 0, so it must be rejected.
  //
  // The optimal acyclic solution is e-node 0 + e-node 3,
  // with objective 1.
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u, 1u},
      {2u, 3u},
    };

  const std::vector<uint32_t>
    ende_child{
      1u,
      0u,
    };

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      1u,
      1u,
      2u,
      2u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      0.0f,
      5.0f,
      0.0f,
      1.0f,
    };

  // Patch 1C-A intentionally ignores warm-start data.
  const std::vector<uint32_t>
    best_ende_of_ecls{
      1u,
      3u,
    };

  const std::vector<bool>
    picked_endes{
      false,
      true,
      false,
      false,
    };

  const CuoptSolveResult result =
    solve_with_cuopt(
      10.0,
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost,
      best_ende_of_ecls,
      picked_endes);

  DOCTEST_REQUIRE(
    result.has_feasible_solution);

  DOCTEST_CHECK_EQ(
    result.objective_value,
    doctest::Approx(1.0));

  DOCTEST_CHECK_EQ(
    result.recomputed_objective,
    doctest::Approx(1.0));

  DOCTEST_CHECK(
    std::find(
      result.selected_enodes.begin(),
      result.selected_enodes.end(),
      0u) != result.selected_enodes.end());

  DOCTEST_CHECK(
    std::find(
      result.selected_enodes.begin(),
      result.selected_enodes.end(),
      3u) != result.selected_enodes.end());

  DOCTEST_CHECK(
    std::find(
      result.selected_enodes.begin(),
      result.selected_enodes.end(),
      2u) == result.selected_enodes.end());
}


DOCTEST_TEST_CASE(
  "cuOpt reports an unavoidable cycle as infeasible")
{
  // e-class 0 is the root:
  //   e-node 0 -> e-class 1
  //
  // e-class 1:
  //   e-node 1 -> e-class 0
  //
  // Each e-class has only one candidate, so the cycle
  // cannot be avoided.
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u},
      {1u},
    };

  const std::vector<uint32_t>
    ende_child{
      1u,
      0u,
    };

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      1u,
      2u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      0.0f,
      0.0f,
    };

  // Patch 1C-A intentionally ignores warm-start data.
  const std::vector<uint32_t>
    best_ende_of_ecls{
      0u,
      1u,
    };

  const std::vector<bool>
    picked_endes{
      true,
      true,
    };

  const CuoptSolveResult result =
    solve_with_cuopt(
      10.0,
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost,
      best_ende_of_ecls,
      picked_endes);

  DOCTEST_CHECK_FALSE(
    result.has_feasible_solution);

  DOCTEST_CHECK(
    result.selected_enodes.empty());
}

#endif