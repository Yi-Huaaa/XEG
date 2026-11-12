/* trunk-ignore-all(clang-format) */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
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
  DOCTEST_REQUIRE(actual.size() == expected.size());

  for (size_t i = 0; i < actual.size(); ++i) {
    DOCTEST_CAPTURE(i);
    DOCTEST_CHECK(actual[i] == expected[i]);
  }
}

std::string indexed_name(
  const char prefix,
  const uint32_t index)
{
  std::ostringstream stream;

  stream
    << prefix
    << std::setfill('0')
    << std::setw(6)
    << index;

  return stream.str();
}


class TemporaryFanoutGraph
{
public:
  explicit TemporaryFanoutGraph(
    const uint32_t total_nodes)
    : path_(
        "/tmp/xeg_batch_boundary_"
        + std::to_string(total_nodes)
        + ".json")
  {
    if (total_nodes < 2u) {
      throw std::invalid_argument(
        "TemporaryFanoutGraph requires "
        "at least two e-nodes");
    }

    std::ofstream output(path_);

    if (!output.is_open()) {
      throw std::runtime_error(
        "Failed to create temporary batch test graph");
    }

    const std::string root_node =
      indexed_name('n', 0u);

    const std::string root_eclass =
      indexed_name('c', 0u);

    output
      << "{\n"
      << "  \"nodes\": {\n"
      << "    \"" << root_node << "\": {\n"
      << "      \"op\": \"root\",\n"
      << "      \"cost\": 1,\n"
      << "      \"eclass\": \""
      << root_eclass
      << "\",\n"
      << "      \"children\": [";

    for (uint32_t node = 1u;
         node < total_nodes;
         ++node) {
      if (node > 1u) {
        output << ", ";
      }

      output
        << "\""
        << indexed_name('n', node)
        << "\"";
    }

    output
      << "]\n"
      << "    }";

    for (uint32_t node = 1u;
         node < total_nodes;
         ++node) {
      output
        << ",\n"
        << "    \""
        << indexed_name('n', node)
        << "\": {\n"
        << "      \"op\": \"candidate\",\n"
        << "      \"cost\": 1,\n"
        << "      \"eclass\": \""
        << indexed_name('c', node)
        << "\",\n"
        << "      \"children\": []\n"
        << "    }";
    }

    output
      << "\n"
      << "  },\n"
      << "  \"root_eclasses\": [\""
      << root_eclass
      << "\"]\n"
      << "}\n";

    output.close();

    if (!output) {
      std::remove(path_.c_str());

      throw std::runtime_error(
        "Failed while writing temporary "
        "batch test graph");
    }
  }

  ~TemporaryFanoutGraph()
  {
    std::remove(path_.c_str());
  }

  TemporaryFanoutGraph(
    const TemporaryFanoutGraph&) = delete;

  TemporaryFanoutGraph&
  operator=(
    const TemporaryFanoutGraph&) = delete;

  const std::string& path() const
  {
    return path_;
  }

private:
  std::string path_;
};

}  // namespace


TEST_CASE(
  "GPU batch top-down insertion uses a selected parent")
{
  XEG xeg;

  xeg.read(
    test_data_path("single_edge.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 0u
  };

  const std::vector<int> initial_level = {
    0, -1
  };

  const auto result =
    xeg.unittest_gpu_insert_batch_once(
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
  "GPU batch bottom-up insertion uses a selected child")
{
  XEG xeg;

  xeg.read(
    test_data_path("single_edge.json"));

  const std::vector<uint32_t> initial_state = {
    0u, 1u
  };

  const std::vector<int> initial_level = {
    -1, 1
  };

  const auto result =
    xeg.unittest_gpu_insert_batch_once(
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
  "GPU batch insertion rejects nodes without selected neighbors")
{
  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    XEG xeg;

    xeg.read(
      test_data_path("single_edge.json"));

    const std::vector<uint32_t> initial_state = {
      0u, 0u
    };

    const std::vector<int> initial_level = {
      -1, -1
    };

    const auto result =
      xeg.unittest_gpu_insert_batch_once(
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
  "GPU batch insertion rejects a parent-child level conflict")
{
  const std::vector<uint32_t> initial_state = {
    1u, 0u, 1u
  };

  const std::vector<int> initial_level = {
    2, -1, 1
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    XEG xeg;

    xeg.read(
      test_data_path("chain.json"));

    const auto result =
      xeg.unittest_gpu_insert_batch_once(
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
  "GPU batch insertion rejects selected parent-child overlap")
{
  const std::vector<uint32_t> initial_state = {
    1u, 0u
  };

  const std::vector<int> initial_level = {
    0, -1
  };

  for (const bool is_top_down : {true, false}) {
    DOCTEST_CAPTURE(is_top_down);

    XEG xeg;

    xeg.read(
      test_data_path("overlap_cycle.json"));

    const auto result =
      xeg.unittest_gpu_insert_batch_once(
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
  "GPU batch insertion treats state 2 as a selected neighbor")
{
  XEG xeg;

  xeg.read(
    test_data_path("single_edge.json"));

  const std::vector<uint32_t> initial_state = {
    2u, 0u
  };

  const std::vector<int> initial_level = {
    0, -1
  };

  const auto result =
    xeg.unittest_gpu_insert_batch_once(
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
  "GPU batch insertion ignores selected neighbors with invalid levels")
{
  XEG xeg;

  xeg.read(
    test_data_path("single_edge.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 0u
  };

  const std::vector<int> initial_level = {
    -1, -1
  };

  const auto result =
    xeg.unittest_gpu_insert_batch_once(
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
  "GPU batch insertion handles multiple independent candidates")
{
  XEG xeg;

  xeg.read(
    test_data_path(
      "parallel_candidates.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 1u, 1u, 0u, 0u
  };

  const std::vector<int> initial_level = {
    0, 1, 2, -1, -1
  };

  const auto result =
    xeg.unittest_gpu_insert_batch_once(
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
  "GPU batch live optimistic insertion permits valid concurrent outcomes")
{
  XEG xeg;

  xeg.read(
    test_data_path(
      "optimistic_cycle_pair.json"));

  const std::vector<uint32_t> initial_state = {
    1u, 1u, 1u, 0u, 0u
  };

  const std::vector<int> initial_level = {
    0, 1, 2, -1, -1
  };

  const auto result =
    xeg.unittest_gpu_insert_batch_once(
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

TEST_CASE(
  "GPU batch insertion covers transposed batch boundaries")
{
  // total_nodes includes:
  //   1 selected root
  //   total_nodes - 1 unselected candidates
  //
  // These sizes cover:
  //   255: below one complete 256-thread batch
  //   256: exactly one complete batch
  //   257: one complete batch plus one node
  //   513: more than two complete batch ranges
  for (const uint32_t total_nodes :
       {255u, 256u, 257u, 513u}) {
    DOCTEST_CAPTURE(total_nodes);

    TemporaryFanoutGraph graph(
      total_nodes);

    XEG xeg;

    xeg.read(
      graph.path());

    std::vector<uint32_t> initial_state(
      total_nodes,
      0u);

    std::vector<int> initial_level(
      total_nodes,
      -1);

    initial_state[0] = 1u;
    initial_level[0] = 0;

    const auto result =
      xeg.unittest_gpu_insert_batch_once(
        initial_state,
        initial_level,
        true);

    DOCTEST_REQUIRE(
      result.selected_state.size()
        == total_nodes);

    DOCTEST_REQUIRE(
      result.level.size()
        == total_nodes);

    // The selected root must remain stable.
    DOCTEST_CHECK(
      result.selected_state[0] == 1u);

    DOCTEST_CHECK(
      result.level[0] == 0);

    // Every candidate has the root as a selected parent.
    // Top-down insertion should therefore publish every
    // candidate as tentative state 2 at level 1.
    for (uint32_t node = 1u;
         node < total_nodes;
         ++node) {
      DOCTEST_CAPTURE(node);

      DOCTEST_CHECK(
        result.selected_state[node] == 2u);

      DOCTEST_CHECK(
        result.level[node] == 1);
    }

    DOCTEST_CHECK(
      result.u0 == total_nodes - 1u);
  }
}