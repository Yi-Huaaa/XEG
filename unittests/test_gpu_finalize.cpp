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
  DOCTEST_REQUIRE(actual.size() == expected.size());

  for (size_t i = 0; i < actual.size(); ++i) {
    DOCTEST_CAPTURE(i);
    DOCTEST_CHECK(actual[i] == expected[i]);
  }
}

void check_all_zero(
  const std::vector<uint32_t>& values)
{
  for (size_t i = 0; i < values.size(); ++i) {
    DOCTEST_CAPTURE(i);
    DOCTEST_CHECK(values[i] == 0u);
  }
}

}  // namespace


TEST_CASE(
  "GPU finalization promotes acyclic tentative nodes")
{
  const std::vector<uint32_t> initial_state = {
    1u, 2u, 1u
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;

    g_ext.read(
      test_data_path("chain.json"));

    const auto result =
      g_ext.unittest_gpu_levelize_and_finalize(
        initial_state,
        is_top_down,
        1u);

    check_vector_equal(
      result.selected_state,
      std::vector<uint32_t>{1u, 1u, 1u});

    check_vector_equal(
      result.level,
      std::vector<int>{0, 1, 2});

    check_all_zero(
      result.residual_degree);

    DOCTEST_CHECK(result.num_levels == 3);

    // The tentative node survived, so it remains counted
    // as a successful insertion.
    DOCTEST_CHECK(result.u0 == 1u);
  }
}


TEST_CASE(
  "GPU finalization rolls back tentative cycle nodes")
{
  const std::vector<uint32_t> initial_state = {
    1u, 1u, 1u, 1u,
    2u, 2u, 2u
  };

  const std::vector<uint32_t> expected_state = {
    1u, 1u, 1u, 1u,
    0u, 0u, 0u
  };

  const std::vector<uint32_t>
    expected_residual_degree = {
      0u, 0u, 0u, 0u,
      1u, 1u, 1u
    };

  const std::vector<int> expected_level = {
    0, 1, 1, 2,
    -1, -1, -1
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;

    g_ext.read(
      test_data_path("mixed_cycle.json"));

    const auto result =
      g_ext.unittest_gpu_levelize_and_finalize(
        initial_state,
        is_top_down,
        3u);

    check_vector_equal(
      result.selected_state,
      expected_state);

    check_vector_equal(
      result.residual_degree,
      expected_residual_degree);

    check_vector_equal(
      result.level,
      expected_level);

    DOCTEST_CHECK(result.num_levels == 3);

    // All three tentative cycle nodes were rolled back.
    DOCTEST_CHECK(result.u0 == 0u);
  }
}


TEST_CASE(
  "GPU finalization does not remove stable cycle nodes")
{
  const std::vector<uint32_t> initial_state = {
    1u, 2u
  };

  const std::vector<uint32_t> expected_state = {
    1u, 0u
  };

  const std::vector<uint32_t>
    expected_residual_degree = {
      1u, 1u
    };

  const std::vector<int> expected_level = {
    -1, -1
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;

    g_ext.read(
      test_data_path("overlap_cycle.json"));

    const auto result =
      g_ext.unittest_gpu_levelize_and_finalize(
        initial_state,
        is_top_down,
        1u);

    check_vector_equal(
      result.selected_state,
      expected_state);

    check_vector_equal(
      result.residual_degree,
      expected_residual_degree);

    check_vector_equal(
      result.level,
      expected_level);

    DOCTEST_CHECK(result.num_levels == 0);

    // Only the state-2 node was rolled back.
    DOCTEST_CHECK(result.u0 == 0u);
  }
}


TEST_CASE(
  "GPU rollback preserves insertions counted by an earlier phase")
{
  const std::vector<uint32_t> initial_state = {
    1u, 2u
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;

    g_ext.read(
      test_data_path("overlap_cycle.json"));

    const auto result =
      g_ext.unittest_gpu_levelize_and_finalize(
        initial_state,
        is_top_down,
        4u);

    check_vector_equal(
      result.selected_state,
      std::vector<uint32_t>{1u, 0u});

    // One current tentative node rolled back.
    // Three previously counted surviving insertions remain.
    DOCTEST_CHECK(result.u0 == 3u);
  }
}