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
  REQUIRE(actual.size() == expected.size());

  for (size_t i = 0; i < actual.size(); ++i) {
    CAPTURE(i);
    CHECK(actual[i] == expected[i]);
  }
}

}  // namespace


TEST_CASE(
  "GPU top-down insertion uses a selected parent")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path("single_edge.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 0u
  };

  const std::vector<int> initial_level = {
    0, -1
  };

  const auto result =
    g_ext.unittest_gpu_insert_once(
      initial_state,
      initial_level,
      true);

  check_vector_equal(
    result.selected_state,
    std::vector<uint32_t>{1u, 2u});

  check_vector_equal(
    result.level,
    std::vector<int>{0, 1});

  CHECK(result.u0 == 1u);
}


TEST_CASE(
  "GPU bottom-up insertion uses a selected child")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path("single_edge.json"));

  const std::vector<uint32_t> initial_state = {
    0u, 1u
  };

  const std::vector<int> initial_level = {
    -1, 1
  };

  const auto result =
    g_ext.unittest_gpu_insert_once(
      initial_state,
      initial_level,
      false);

  check_vector_equal(
    result.selected_state,
    std::vector<uint32_t>{2u, 1u});

  check_vector_equal(
    result.level,
    std::vector<int>{0, 1});

  CHECK(result.u0 == 1u);
}


TEST_CASE(
  "GPU insertion rejects nodes without selected neighbors")
{
  for (const bool is_top_down : {true, false}) {
    CAPTURE(is_top_down);

    G_EXT g_ext;

    g_ext.read(
      test_data_path("single_edge.json"));

    const std::vector<uint32_t> initial_state = {
      0u, 0u
    };

    const std::vector<int> initial_level = {
      -1, -1
    };

    const auto result =
      g_ext.unittest_gpu_insert_once(
        initial_state,
        initial_level,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      initial_state);

    check_vector_equal(
      result.level,
      initial_level);

    CHECK(result.u0 == 0u);
  }
}


TEST_CASE(
  "GPU insertion rejects a parent-child level conflict")
{
  const std::vector<uint32_t> initial_state = {
    1u, 0u, 1u
  };

  const std::vector<int> initial_level = {
    2, -1, 1
  };

  for (const bool is_top_down : {true, false}) {
    CAPTURE(is_top_down);

    G_EXT g_ext;

    g_ext.read(
      test_data_path("chain.json"));

    const auto result =
      g_ext.unittest_gpu_insert_once(
        initial_state,
        initial_level,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      initial_state);

    check_vector_equal(
      result.level,
      initial_level);

    CHECK(result.u0 == 0u);
  }
}


TEST_CASE(
  "GPU insertion rejects selected parent-child overlap")
{
  const std::vector<uint32_t> initial_state = {
    1u, 0u
  };

  const std::vector<int> initial_level = {
    0, -1
  };

  for (const bool is_top_down : {true, false}) {
    CAPTURE(is_top_down);

    G_EXT g_ext;

    g_ext.read(
      test_data_path("overlap_cycle.json"));

    const auto result =
      g_ext.unittest_gpu_insert_once(
        initial_state,
        initial_level,
        is_top_down);

    check_vector_equal(
      result.selected_state,
      initial_state);

    check_vector_equal(
      result.level,
      initial_level);

    CHECK(result.u0 == 0u);
  }
}


TEST_CASE(
  "GPU insertion treats state 2 as a selected neighbor")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path("single_edge.json"));

  const std::vector<uint32_t> initial_state = {
    2u, 0u
  };

  const std::vector<int> initial_level = {
    0, -1
  };

  const auto result =
    g_ext.unittest_gpu_insert_once(
      initial_state,
      initial_level,
      true);

  check_vector_equal(
    result.selected_state,
    std::vector<uint32_t>{2u, 2u});

  check_vector_equal(
    result.level,
    std::vector<int>{0, 1});

  CHECK(result.u0 == 1u);
}


TEST_CASE(
  "GPU insertion ignores selected neighbors with invalid levels")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path("single_edge.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 0u
  };

  const std::vector<int> initial_level = {
    -1, -1
  };

  const auto result =
    g_ext.unittest_gpu_insert_once(
      initial_state,
      initial_level,
      true);

  check_vector_equal(
    result.selected_state,
    initial_state);

  check_vector_equal(
    result.level,
    initial_level);

  CHECK(result.u0 == 0u);
}