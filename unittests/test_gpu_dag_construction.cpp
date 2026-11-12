/* trunk-ignore-all(clang-format) */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <xeg/XEG.hpp>

namespace {

std::string test_data_path(
  const std::string& filename)
{
  return std::string(XEG_TEST_DATA_DIR)
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


void check_selected_edges_respect_levels(
  const XEG::CsrGraphSnapshot& csr,
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& level)
{
  DOCTEST_REQUIRE(
    selected_state.size() == level.size());

  DOCTEST_REQUIRE(
    csr.forward_ptr.size()
      == selected_state.size() + 1);

  for (size_t u = 0;
       u < selected_state.size();
       ++u) {
    if (selected_state[u] == 0u) {
      continue;
    }

    DOCTEST_CAPTURE(u);

    DOCTEST_REQUIRE(level[u] >= 0);

    const uint32_t begin =
      csr.forward_ptr[u];

    const uint32_t end =
      csr.forward_ptr[u + 1];

    DOCTEST_REQUIRE(begin <= end);

    DOCTEST_REQUIRE(
      end <= csr.forward_adj.size());

    for (uint32_t index = begin;
         index < end;
         ++index) {
      const uint32_t v =
        csr.forward_adj[index];

      DOCTEST_CAPTURE(v);

      DOCTEST_REQUIRE(
        v < selected_state.size());

      if (selected_state[v] == 0u) {
        continue;
      }

      DOCTEST_REQUIRE(level[v] >= 0);

      // Every selected edge must move from a smaller
      // top-down level to a larger top-down level.
      DOCTEST_CHECK(
        level[u] < level[v]);
    }
  }
}

void check_in_subgraph_is_selected_subset(
  const std::vector<uint32_t>& selected_state,
  const std::vector<uint32_t>& in_subgraph)
{
  DOCTEST_REQUIRE(
    in_subgraph.size() ==
    selected_state.size());

  for (size_t u = 0u;
       u < selected_state.size();
       ++u) {
    DOCTEST_CAPTURE(u);
    DOCTEST_CHECK(in_subgraph[u] <= 1u);

    if (in_subgraph[u] != 0u) {
      DOCTEST_CHECK(
        selected_state[u] != 0u);
    }
  }
}

}  // namespace


TEST_CASE(
  "GPU DAG construction inserts a node during the bottom-up phase")
{
  XEG xeg;

  xeg.read(
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
    xeg.unittest_gpu_construct_dag(
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

  check_in_subgraph_is_selected_subset(
    result.selected_state,
    result.in_subgraph);
}


TEST_CASE(
  "GPU DAG construction inserts a node during the top-down phase")
{
  XEG xeg;

  xeg.read(
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
    xeg.unittest_gpu_construct_dag(
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

  check_in_subgraph_is_selected_subset(
    result.selected_state,
    result.in_subgraph);
}


TEST_CASE(
  "GPU DAG construction leaves a complete DAG unchanged")
{
  XEG xeg;

  xeg.read(
    test_data_path("chain.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 1u, 1u
  };

  const auto result =
    xeg.unittest_gpu_construct_dag(
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

  check_in_subgraph_is_selected_subset(
    result.selected_state,
    result.in_subgraph);
}

TEST_CASE(
  "GPU DAG construction inserts multiple independent candidates")
{
  XEG xeg;

  xeg.read(
    test_data_path(
      "parallel_candidates.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 1u, 1u, 0u, 0u
  };

  const auto result =
    xeg.unittest_gpu_construct_dag(
      initial_state);

  check_vector_equal(
    result.selected_state,
    std::vector<uint32_t>{
      1u, 1u, 1u, 1u, 1u
    });

  check_vector_equal(
    result.level,
    std::vector<int>{
      0, 1, 2, 1, 1
    });

  check_all_zero(
    result.residual_degree);

  check_no_tentative_state(
    result.selected_state);

  const auto csr =
    xeg.unittest_get_csr_snapshot();

  check_selected_edges_respect_levels(
    csr,
    result.selected_state,
    result.level);

  DOCTEST_CHECK(result.num_levels == 3);
  DOCTEST_CHECK(result.final_u0 == 0u);

  check_in_subgraph_is_selected_subset(
    result.selected_state,
    result.in_subgraph);
}


TEST_CASE(
  "GPU DAG construction resolves concurrent tentative cycle safely")
{
  XEG xeg;

  xeg.read(
    test_data_path(
      "optimistic_cycle_pair.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 1u, 1u, 0u, 0u
  };

  const auto result =
    xeg.unittest_gpu_construct_dag(
      initial_state);

  DOCTEST_REQUIRE(
    result.selected_state.size() == 5u);

  DOCTEST_REQUIRE(
    result.level.size() == 5u);

  // The original stable DAG must be preserved.
  DOCTEST_CHECK(
    result.selected_state[0] == 1u);

  DOCTEST_CHECK(
    result.selected_state[1] == 1u);

  DOCTEST_CHECK(
    result.selected_state[2] == 1u);

  DOCTEST_CHECK(result.level[0] == 0);
  DOCTEST_CHECK(result.level[1] == 1);
  DOCTEST_CHECK(result.level[2] == 2);

  // Complete DAG construction must not leave state 2.
  check_no_tentative_state(
    result.selected_state);

  const bool candidate_3_has_valid_state =
    result.selected_state[3] == 0u ||
    result.selected_state[3] == 1u;

  const bool candidate_4_has_valid_state =
    result.selected_state[4] == 0u ||
    result.selected_state[4] == 1u;

  DOCTEST_CHECK(
    candidate_3_has_valid_state);

  DOCTEST_CHECK(
    candidate_4_has_valid_state);

  const uint32_t surviving_candidates =
    static_cast<uint32_t>(
      result.selected_state[3] == 1u)
    +
    static_cast<uint32_t>(
      result.selected_state[4] == 1u);

  // Keeping both candidates would retain n3 <-> n4.
  // Zero or one candidate may legally survive.
  DOCTEST_CHECK(
    surviving_candidates <= 1u);

  for (size_t candidate = 3;
       candidate <= 4;
       ++candidate) {
    DOCTEST_CAPTURE(candidate);

    if (result.selected_state[candidate] == 1u) {
      DOCTEST_CHECK(
        result.level[candidate] == 1);
    }
    else {
      DOCTEST_CHECK(
        result.level[candidate] == -1);
    }
  }

  check_all_zero(
    result.residual_degree);

  const auto csr =
    xeg.unittest_get_csr_snapshot();

  check_selected_edges_respect_levels(
    csr,
    result.selected_state,
    result.level);

  DOCTEST_CHECK(result.num_levels == 3);
  DOCTEST_CHECK(result.final_u0 == 0u);

  check_in_subgraph_is_selected_subset(
    result.selected_state,
    result.in_subgraph);
}

TEST_CASE(
  "GPU root-to-leaf filtering removes both kinds of redundant nodes")
{
  XEG xeg;

  xeg.read(
    test_data_path(
      "filter_branches.json"));

  const std::vector<uint32_t> selected_state = {
    1u,  // n0: selected root
    1u,  // n1: valid middle
    1u,  // n2: reachable from root, but no selected path to leaf
    1u,  // n3: selected global leaf
    0u,  // n4
    0u,  // n5
    1u,  // n6: disconnected selected component
    1u,  // n7: disconnected selected global leaf
    0u   // n8: unselected root alternative
  };

  const auto result =
    xeg.unittest_gpu_filter_root_to_leaf(
      selected_state);

  // Filtering must not modify the DAG-construction result.
  check_vector_equal(
    result.selected_state,
    selected_state);

  check_vector_equal(
    result.from_roots,
    std::vector<uint32_t>{
      1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u
    });

  check_vector_equal(
    result.from_leaves,
    std::vector<uint32_t>{
      1u, 1u, 0u, 1u, 0u, 0u, 1u, 1u, 0u
    });

  check_vector_equal(
    result.in_subgraph,
    std::vector<uint32_t>{
      1u, 1u, 0u, 1u, 0u, 0u, 0u, 0u, 0u
    });
}

TEST_CASE(
  "GPU root-to-leaf filtering retains a complete selected graph")
{
  XEG xeg;

  xeg.read(
    test_data_path(
      "filter_branches.json"));

  const std::vector<uint32_t> selected_state = {
    1u, 1u, 1u, 1u, 1u,
    1u, 1u, 1u, 1u
  };

  const auto result =
    xeg.unittest_gpu_filter_root_to_leaf(
      selected_state);

  const std::vector<uint32_t> all_selected(
    9u,
    1u);

  check_vector_equal(
    result.selected_state,
    all_selected);

  check_vector_equal(
    result.from_roots,
    all_selected);

  check_vector_equal(
    result.from_leaves,
    all_selected);

  check_vector_equal(
    result.in_subgraph,
    all_selected);
}

TEST_CASE(
  "GPU DAG construction filters a disconnected selected component")
{
  XEG xeg;

  xeg.read(
    test_data_path(
      "filter_branches.json"));

  // Initial selected graph:
  //
  // n0 -> n1 -> n3
  //  |
  //  +--> n2
  //
  // n6 -> n7 is selected but disconnected from every
  // selected root because its root alternative n8 is unselected.
  //
  // Full DAG construction should insert n4 and n5:
  //
  // n0 -> n2 -> n4 -> n5
  //
  // Filtering must then retain n0..n5 while excluding the
  // disconnected selected component n6 -> n7.
  const std::vector<uint32_t> initial_state = {
    1u,  // n0
    1u,  // n1
    1u,  // n2
    1u,  // n3
    0u,  // n4
    0u,  // n5
    1u,  // n6
    1u,  // n7
    0u   // n8
  };

  const auto result =
    xeg.unittest_gpu_construct_dag(
      initial_state);

  // DAG construction preserves the disconnected selected
  // component and inserts the missing root-connected branch.
  check_vector_equal(
    result.selected_state,
    std::vector<uint32_t>{
      1u, 1u, 1u, 1u, 1u,
      1u, 1u, 1u, 0u
    });

  // Root-to-leaf filtering does not modify selected_state.
  // It records only nodes lying on a selected path from a
  // selected root to a global leaf.
  check_vector_equal(
    result.in_subgraph,
    std::vector<uint32_t>{
      1u, 1u, 1u, 1u, 1u,
      1u, 0u, 0u, 0u
    });

  check_in_subgraph_is_selected_subset(
    result.selected_state,
    result.in_subgraph);

  for (uint32_t state :
       result.selected_state) {
    DOCTEST_CHECK(state != 2u);
  }

  DOCTEST_CHECK(
    result.final_u0 == 0u);
}