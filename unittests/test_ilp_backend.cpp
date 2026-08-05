#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <g_ext/g_ext.cuh>
#include <g_ext/cuopt_solver.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

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
  "cuOpt solver boundary reports its implementation status")
{
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

#ifdef XEG_ENABLE_CUOPT
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
    "cuOpt XEG formulation is not implemented in this patch",
    std::runtime_error);
#else
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
  DOCTEST_CHECK_THROWS_WITH_AS(
    extension.unittest_solve_ilp_backend(
      IlpBackend::CuOpt),
    "cuOpt XEG formulation is not implemented in this patch",
    std::runtime_error);
#else
  DOCTEST_CHECK_THROWS_WITH_AS(
    extension.unittest_solve_ilp_backend(
      IlpBackend::CuOpt),
    "XEG was built without cuOpt support",
    std::runtime_error);
#endif
}
