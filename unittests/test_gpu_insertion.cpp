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

  DOCTEST_CHECK(result.u0 == 1u);
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

  DOCTEST_CHECK(result.u0 == 1u);
}


TEST_CASE(
  "GPU insertion rejects nodes without selected neighbors")
{
  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

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

    DOCTEST_CHECK(result.u0 == 0u);
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
    DOCTEST_CAPTURE(is_top_down);

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

    DOCTEST_CHECK(result.u0 == 0u);
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
    DOCTEST_CAPTURE(is_top_down);

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

    DOCTEST_CHECK(result.u0 == 0u);
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

  DOCTEST_CHECK(result.u0 == 1u);
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

  DOCTEST_CHECK(result.u0 == 0u);
}

TEST_CASE(
  "GPU insertion handles multiple independent candidates")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path(
      "parallel_candidates.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 1u, 1u, 0u, 0u
  };

  const std::vector<int> initial_level = {
    0, 1, 2, -1, -1
  };

  const auto result =
    g_ext.unittest_gpu_insert_once(
      initial_state,
      initial_level,
      false);

  check_vector_equal(
    result.selected_state,
    std::vector<uint32_t>{
      1u, 1u, 1u, 2u, 2u
    });

  check_vector_equal(
    result.level,
    std::vector<int>{
      0, 1, 2, 1, 1
    });

  DOCTEST_CHECK(result.u0 == 2u);
}


TEST_CASE(
  "GPU live optimistic insertion permits valid concurrent outcomes")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path(
      "optimistic_cycle_pair.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 1u, 1u, 0u, 0u
  };

  const std::vector<int> initial_level = {
    0, 1, 2, -1, -1
  };

  const auto result =
    g_ext.unittest_gpu_insert_once(
      initial_state,
      initial_level,
      false);

  DOCTEST_REQUIRE(
    result.selected_state.size() == 5u);

  DOCTEST_REQUIRE(
    result.level.size() == 5u);

  // Original stable nodes must remain unchanged.
  DOCTEST_CHECK(
    result.selected_state[0] == 1u);

  DOCTEST_CHECK(
    result.selected_state[1] == 1u);

  DOCTEST_CHECK(
    result.selected_state[2] == 1u);

  DOCTEST_CHECK(result.level[0] == 0);
  DOCTEST_CHECK(result.level[1] == 1);
  DOCTEST_CHECK(result.level[2] == 2);

  // Finalization has not run yet. Each candidate is either
  // still unselected or has tentatively published state 2.
  const bool candidate_3_has_valid_state =
    result.selected_state[3] == 0u ||
    result.selected_state[3] == 2u;

  const bool candidate_4_has_valid_state =
    result.selected_state[4] == 0u ||
    result.selected_state[4] == 2u;

  DOCTEST_CHECK(
    candidate_3_has_valid_state);

  DOCTEST_CHECK(
    candidate_4_has_valid_state);

  const uint32_t tentative_count =
    static_cast<uint32_t>(
      result.selected_state[3] == 2u)
    +
    static_cast<uint32_t>(
      result.selected_state[4] == 2u);

  // At least one candidate must publish. Depending on the
  // thread interleaving, either one or both can publish.
  DOCTEST_CHECK(tentative_count >= 1u);
  DOCTEST_CHECK(tentative_count <= 2u);

  DOCTEST_CHECK(
    result.u0 == tentative_count);

  for (size_t candidate = 3;
       candidate <= 4;
       ++candidate) {
    DOCTEST_CAPTURE(candidate);

    if (result.selected_state[candidate] == 2u) {
      DOCTEST_CHECK(
        result.level[candidate] == 1);
    }
    else {
      DOCTEST_CHECK(
        result.level[candidate] == -1);
    }
  }
}

