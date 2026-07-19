/* trunk-ignore-all(clang-format) */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <g_ext/G_EXT.hpp>

namespace {

std::string test_data_path(const std::string& filename)
{
  return std::string(G_EXT_TEST_DATA_DIR) + "/" + filename;
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


TEST_CASE("GPU levelization handles a chain")
{
  const std::vector<uint32_t> selected_state = {
    1u, 1u, 1u
  };

  const std::vector<int> expected_level = {
    0, 1, 2
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;
    g_ext.read(test_data_path("chain.json"));

    const auto result =
      g_ext.unittest_gpu_levelize(
        selected_state,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      selected_state);

    check_vector_equal(
      result.level,
      expected_level);

    check_all_zero(
      result.residual_degree);

    DOCTEST_CHECK(result.num_levels == 3);
  }
}


TEST_CASE("GPU levelization handles a diamond")
{
  const std::vector<uint32_t> selected_state = {
    1u, 1u, 1u, 1u
  };

  const std::vector<int> expected_level = {
    0, 1, 1, 2
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;
    g_ext.read(test_data_path("diamond.json"));

    const auto result =
      g_ext.unittest_gpu_levelize(
        selected_state,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      selected_state);

    check_vector_equal(
      result.level,
      expected_level);

    check_all_zero(
      result.residual_degree);

    DOCTEST_CHECK(result.num_levels == 3);
  }
}


TEST_CASE("GPU levelization ignores unselected e-nodes")
{
  // Diamond with n2 unselected:
  //
  // n0 -> n1 -> n3
  //
  // n2 must remain at level -1.
  const std::vector<uint32_t> selected_state = {
    1u, 1u, 0u, 1u
  };

  const std::vector<int> expected_level = {
    0, 1, -1, 2
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;
    g_ext.read(test_data_path("diamond.json"));

    const auto result =
      g_ext.unittest_gpu_levelize(
        selected_state,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      selected_state);

    check_vector_equal(
      result.level,
      expected_level);

    check_all_zero(
      result.residual_degree);

    DOCTEST_CHECK(result.num_levels == 3);
  }
}


TEST_CASE("GPU levelization treats state 2 as selected")
{
  const std::vector<uint32_t> selected_state = {
    1u, 2u, 1u
  };

  const std::vector<int> expected_level = {
    0, 1, 2
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;
    g_ext.read(test_data_path("chain.json"));

    const auto result =
      g_ext.unittest_gpu_levelize(
        selected_state,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      selected_state);

    check_vector_equal(
      result.level,
      expected_level);

    check_all_zero(
      result.residual_degree);

    DOCTEST_CHECK(result.num_levels == 3);
  }
}


TEST_CASE("GPU levelization leaves cycle nodes unresolved")
{
  const std::vector<uint32_t> selected_state = {
    1u, 1u, 1u
  };

  const std::vector<int> expected_level = {
    -1, -1, -1
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;
    g_ext.read(test_data_path("cycle.json"));

    const auto result =
      g_ext.unittest_gpu_levelize(
        selected_state,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      selected_state);

    check_vector_equal(
      result.level,
      expected_level);

    DOCTEST_REQUIRE(result.residual_degree.size() == 3);

    for (size_t i = 0;
         i < result.residual_degree.size();
         ++i) {
      DOCTEST_CAPTURE(i);
      DOCTEST_CHECK(result.residual_degree[i] > 0u);
    }

    DOCTEST_CHECK(result.num_levels == 0);
  }
}

TEST_CASE(
  "GPU levelization handles an asymmetric multi-root DAG")
{
  const std::vector<uint32_t> selected_state = {
    1u, 1u, 1u, 1u, 1u, 1u
  };

  const std::vector<int> expected_td_level = {
    0, 0, 1, 1, 2, 3
  };

  const std::vector<int> expected_bu_level = {
    0, 1, 1, 2, 2, 3
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;
    g_ext.read(
      test_data_path("asymmetric_dag.json"));

    const auto result =
      g_ext.unittest_gpu_levelize(
        selected_state,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      selected_state);

    if (is_top_down) {
      check_vector_equal(
        result.level,
        expected_td_level);
    }
    else {
      check_vector_equal(
        result.level,
        expected_bu_level);
    }

    check_all_zero(
      result.residual_degree);

    DOCTEST_CHECK(result.num_levels == 4);
  }
}

TEST_CASE(
  "GPU levelization processes a DAG while retaining a cycle")
{
  const std::vector<uint32_t> selected_state = {
    1u, 1u, 1u, 1u, 1u, 1u, 1u
  };

  const std::vector<int> expected_level = {
    0, 1, 1, 2, -1, -1, -1
  };

  const std::vector<uint32_t>
    expected_residual_degree = {
      0u, 0u, 0u, 0u, 1u, 1u, 1u
    };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    G_EXT g_ext;
    g_ext.read(
      test_data_path("mixed_cycle.json"));

    const auto result =
      g_ext.unittest_gpu_levelize(
        selected_state,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      selected_state);

    check_vector_equal(
      result.level,
      expected_level);

    check_vector_equal(
      result.residual_degree,
      expected_residual_degree);

    DOCTEST_CHECK(result.num_levels == 3);
  }
}