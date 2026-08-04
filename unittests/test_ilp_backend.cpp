#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <g_ext/g_ext.cuh>

#include <cstdint>
#include <stdexcept>
#include <string>


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