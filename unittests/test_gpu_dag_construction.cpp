/* trunk-ignore-all(clang-format) */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <g_ext/G_EXT.hpp>

namespace {

std::string test_data_path(
  const std::string& filename)
{
  return std::string(G_EXT_TEST_DATA_DIR)
    + "/"
    + filename;
}

template <typename T>
void check_vector_equal(
  const std::vector<T>& actual,
  const std::vector<T>& expected)
{
  DOCTEST_REQUIRE(
    actual.size() == expected.size());

  for (size_t i = 0; i < actual.size(); ++i) {
    DOCTEST_CAPTURE(i);

    DOCTEST_CHECK(
      actual[i] == expected[i]);
  }
}

void check_all_zero(
  const std::vector<uint32_t>& values)
{
  for (size_t i = 0; i < values.size(); ++i) {
    DOCTEST_CAPTURE(i);

    DOCTEST_CHECK(
      values[i] == 0u);
  }
}

void check_no_tentative_state(
  const std::vector<uint32_t>& selected_state)
{
  for (size_t i = 0;
       i < selected_state.size();
       ++i) {
    DOCTEST_CAPTURE(i);

    DOCTEST_CHECK(
      selected_state[i] != 2u);
  }
}

}  // namespace


TEST_CASE(
  "GPU DAG construction inserts a node during the bottom-up phase")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path("diamond.json"));

  // Selected subgraph before insertion:
  //
  //     n0
  //      |
  //     n1
  //      |
  //     n3
  //
  // n2 is initially unselected.
  //
  // Initial top-down levels:
  //   n0 = 0
  //   n1 = 1
  //   n3 = 2
  //
  // Candidate n2 has:
  //   parent n0 at level 0
  //   child  n3 at level 2
  //
  // Bottom-up insertion assigns:
  //   level[n2] = 2 - 1 = 1
  const std::vector<uint32_t> initial_state = {
    1u, 1u, 0u, 1u
  };

  const auto result =
    g_ext.unittest_gpu_construct_dag(
      initial_state);

  check_vector_equal(
    result.selected_state,
    std::vector<uint32_t>{
      1u, 1u, 1u, 1u
    });

  check_vector_equal(
    result.level,
    std::vector<int>{
      0, 1, 1, 2
    });

  check_all_zero(
    result.residual_degree);

  check_no_tentative_state(
    result.selected_state);

  DOCTEST_CHECK(result.num_levels == 3);
  DOCTEST_CHECK(result.final_u0 == 0u);
}


TEST_CASE(
  "GPU DAG construction inserts a node during the top-down phase")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path("single_edge.json"));

  // Initial selected graph:
  //
  // n0 -> n1
  //
  // n0 is selected and n1 is not.
  //
  // n1 has no selected child, so the first bottom-up
  // insertion rejects it.
  //
  // After bottom-up levelization, top-down insertion sees
  // selected parent n0 and inserts n1.
  const std::vector<uint32_t> initial_state = {
    1u, 0u
  };

  const auto result =
    g_ext.unittest_gpu_construct_dag(
      initial_state);

  check_vector_equal(
    result.selected_state,
    std::vector<uint32_t>{
      1u, 1u
    });

  check_vector_equal(
    result.level,
    std::vector<int>{
      0, 1
    });

  check_all_zero(
    result.residual_degree);

  check_no_tentative_state(
    result.selected_state);

  DOCTEST_CHECK(result.num_levels == 2);
  DOCTEST_CHECK(result.final_u0 == 0u);
}


TEST_CASE(
  "GPU DAG construction leaves a complete DAG unchanged")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path("chain.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 1u, 1u
  };

  const auto result =
    g_ext.unittest_gpu_construct_dag(
      initial_state);

  check_vector_equal(
    result.selected_state,
    initial_state);

  check_vector_equal(
    result.level,
    std::vector<int>{
      0, 1, 2
    });

  check_all_zero(
    result.residual_degree);

  check_no_tentative_state(
    result.selected_state);

  DOCTEST_CHECK(result.num_levels == 3);
  DOCTEST_CHECK(result.final_u0 == 0u);
}