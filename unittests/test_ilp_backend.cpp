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
  "cuOpt mode value zero selects presolve")
{
  const CuoptMode mode =
    parse_cuopt_mode(0u);

  DOCTEST_CHECK_EQ(
    mode,
    CuoptMode::Presolve);
}


DOCTEST_TEST_CASE(
  "cuOpt mode value one selects GPU tree warm start")
{
  const CuoptMode mode =
    parse_cuopt_mode(1u);

  DOCTEST_CHECK_EQ(
    mode,
    CuoptMode::GpuTreeWarmStart);
}


DOCTEST_TEST_CASE(
  "Invalid cuOpt mode values are rejected")
{
  DOCTEST_CHECK_THROWS_WITH_AS(
    parse_cuopt_mode(2u),
    "Invalid CUOPT_MODE value: 2",
    std::invalid_argument);
}


DOCTEST_TEST_CASE(
  "cuOpt mode names are stable")
{
  const std::string presolve_name =
    cuopt_mode_name(
      CuoptMode::Presolve);

  const std::string warm_start_name =
    cuopt_mode_name(
      CuoptMode::GpuTreeWarmStart);

  DOCTEST_CHECK_EQ(
    presolve_name,
    "Presolve");

  DOCTEST_CHECK_EQ(
    warm_start_name,
    "GPU-tree warm start");
}

#ifdef XEG_ENABLE_CUOPT

DOCTEST_TEST_CASE(
  "cuOpt GPU-tree warm start validates selected-mask size")
{
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u},
    };

  const std::vector<uint32_t>
    ende_child;

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      0u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      1.0f,
    };

  const std::vector<uint32_t>
    best_ende_of_ecls{
      0u,
    };

  // Deliberately invalid: the model contains one e-node.
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
      picked_endes,
      CuoptMode::GpuTreeWarmStart),
    "GPU tree selected-mask size does not match "
    "the number of e-nodes",
    std::invalid_argument);
}

DOCTEST_TEST_CASE(
  "cuOpt GPU-tree warm start validates representative size")
{
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u},
    };

  const std::vector<uint32_t>
    ende_child;

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      0u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      1.0f,
    };

  // Invalid: the model contains one e-class.
  const std::vector<uint32_t>
    best_ende_of_ecls;

  const std::vector<bool>
    picked_endes{
      true,
    };

  DOCTEST_CHECK_THROWS_WITH_AS(
    solve_with_cuopt(
      1.0,
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost,
      best_ende_of_ecls,
      picked_endes,
      CuoptMode::GpuTreeWarmStart),
    "GPU tree representative size does not match "
    "the number of e-classes",
    std::invalid_argument);
}

DOCTEST_TEST_CASE(
  "cuOpt solve result defaults to no MIP start")
{
  const CuoptSolveResult result;

  DOCTEST_CHECK_EQ(
    result.mip_start_submitted,
    false);

  DOCTEST_CHECK(
    result.mip_start_objective ==
    doctest::Approx(0.0));
}

DOCTEST_TEST_CASE(
  "cuOpt submits GPU-tree dependency-chain MIP start")
{
  // e-class 0 (root):
  //   e-node 0 -> e-class 1, cost 2
  //
  // e-class 1:
  //   e-node 1 -> leaf, cost 3
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
      2.0f,
      3.0f,
    };

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
      picked_endes,
      CuoptMode::GpuTreeWarmStart);

  DOCTEST_REQUIRE(
    result.has_feasible_solution);

  DOCTEST_CHECK(
    result.mip_start_submitted);

  DOCTEST_CHECK(
    result.mip_start_objective ==
    doctest::Approx(5.0));

  DOCTEST_CHECK(
    result.objective_value ==
    doctest::Approx(5.0));
}

DOCTEST_TEST_CASE(
  "cuOpt can improve a worse GPU-tree MIP start")
{
  // One root e-class with two leaf candidates:
  //
  // e-node 0: cost 5, selected by GPU tree
  // e-node 1: cost 2, better MILP solution
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
      picked_endes,
      CuoptMode::GpuTreeWarmStart);

  DOCTEST_REQUIRE(
    result.has_feasible_solution);

  DOCTEST_CHECK(
    result.mip_start_submitted);

  DOCTEST_CHECK(
    result.mip_start_objective ==
    doctest::Approx(5.0));

  DOCTEST_CHECK(
    result.objective_value ==
    doctest::Approx(2.0));

  DOCTEST_REQUIRE_EQ(
    result.selected_enodes.size(),
    1u);

  DOCTEST_CHECK_EQ(
    result.selected_enodes.at(0),
    1u);
}

DOCTEST_TEST_CASE(
  "cuOpt rejects GPU-tree e-node absent from DAG search space")
{
  // The canonical DAG search space contains only e-node 0.
  //
  // e-node 1 still exists in the original e-node arrays and is
  // selected by the GPU tree snapshot, so the start is invalid.
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u},
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
      1.0f,
      2.0f,
    };

  const std::vector<uint32_t>
    best_ende_of_ecls{
      1u,
    };

  const std::vector<bool>
    picked_endes{
      false,
      true,
    };

  DOCTEST_CHECK_THROWS_WITH_AS(
    solve_with_cuopt(
      10.0,
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost,
      best_ende_of_ecls,
      picked_endes,
      CuoptMode::GpuTreeWarmStart),
    "GPU tree selected e-node is absent from "
    "the DAG search space",
    std::invalid_argument);
}

DOCTEST_TEST_CASE(
  "cuOpt rejects cyclic GPU-tree MIP start")
{
  // The DAG search space itself is feasible because each
  // e-class also has a leaf alternative.
  //
  // The GPU tree snapshot selects:
  //   e-node 0: e-class 0 -> e-class 1
  //   e-node 1: e-class 1 -> e-class 0
  //
  // This selected start is cyclic and must be rejected.
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u, 2u},
      {1u, 3u},
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
      2u,
      2u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      1.0f,
      1.0f,
      2.0f,
      2.0f,
    };

  const std::vector<uint32_t>
    best_ende_of_ecls{
      0u,
      1u,
    };

  const std::vector<bool>
    picked_endes{
      true,
      true,
      false,
      false,
    };

  DOCTEST_CHECK_THROWS_WITH_AS(
    solve_with_cuopt(
      10.0,
      ecls_its_ende,
      ende_child,
      ende_child_ptr,
      root_eclass_id,
      ende_cost,
      best_ende_of_ecls,
      picked_endes,
      CuoptMode::GpuTreeWarmStart),
    "GPU tree selected e-nodes contain a cycle",
    std::invalid_argument);
}

#endif

DOCTEST_TEST_CASE(
  "G_EXT run accepts a trailing cuOpt mode")
{
  using RunFunction =
    void (G_EXT::*)(
      bool,
      double,
      DagKernel,
      uint32_t,
      uint32_t,
      IlpBackend,
      CuoptMode);

  constexpr bool has_expected_signature =
    std::is_same_v<
      decltype(&G_EXT::run),
      RunFunction>;

  DOCTEST_CHECK_EQ(
    has_expected_signature,
    true);
}

DOCTEST_TEST_CASE(
  "cuOpt solver accepts a trailing solve mode")
{
  using SolveWithCuoptFunction =
    CuoptSolveResult (*)(
      double,
      const std::vector<std::vector<uint32_t>>&,
      const std::vector<uint32_t>&,
      const std::vector<uint32_t>&,
      const std::vector<uint32_t>&,
      const std::vector<float>&,
      const std::vector<uint32_t>&,
      const std::vector<bool>&,
      CuoptMode);

  constexpr bool has_expected_signature =
    std::is_same_v<
      decltype(&solve_with_cuopt),
      SolveWithCuoptFunction>;

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

  // Presolve mode intentionally ignores warm-start data.
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
  "G_EXT dispatcher accepts a trailing cuOpt mode")
{
  using DispatcherFunction =
    void (G_EXT::*)(
      IlpBackend,
      CuoptMode);

  constexpr bool has_expected_signature =
    std::is_same_v<
      decltype(
        &G_EXT::unittest_solve_ilp_backend),
      DispatcherFunction>;

  DOCTEST_CHECK_EQ(
    has_expected_signature,
    true);
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

  // Presolve mode intentionally ignores warm-start data.
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

  // Presolve mode intentionally ignores warm-start data.
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