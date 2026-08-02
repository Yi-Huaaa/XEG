#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <g_ext/g_ext.cuh>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

constexpr int kTestNodeIdWidth = 10;

constexpr uint32_t kProbeSearchBegin = 2u;
constexpr uint32_t kProbeSearchLimit = 200000u;

constexpr uint32_t kCandidateParentLevel = 1u;
constexpr uint32_t kCandidateChildLevel = 3u;
constexpr int kExpectedCandidateLevel = 2;

const std::vector<uint32_t> kSupportedBitmapBits{
  32u,
  64u,
  128u,
  256u,
  512u,
  1024u,
  2048u
};

struct Edge
{
  uint32_t parent;
  uint32_t child;
};

struct NodePair
{
  uint32_t parent;
  uint32_t child;
};

struct ProbeBits
{
  uint32_t first;
  uint32_t second;
};

struct CandidateState
{
  std::vector<uint32_t> selected_state;
  std::vector<int> level;
};

struct StrategyGraphCase
{
  uint32_t bitmap_bits;

  NodePair full_collision;
  NodePair first_probe_only_collision;
  NodePair second_probe_only_collision;

  uint32_t full_collision_candidate;
  uint32_t first_probe_candidate;
  uint32_t second_probe_candidate;

  uint32_t node_count;
  std::vector<Edge> edges;
};

enum class PartialProbeMatch
{
  First,
  Second
};


std::string node_name(
  const uint32_t node)
{
  std::ostringstream stream;

  stream
    << "n"
    << std::setfill('0')
    << std::setw(kTestNodeIdWidth)
    << node;

  return stream.str();
}


std::string eclass_name(
  const uint32_t node)
{
  std::ostringstream stream;

  stream
    << "e"
    << std::setfill('0')
    << std::setw(kTestNodeIdWidth)
    << node;

  return stream.str();
}


// Host-side copy of gpu_dag_bitmap_mix32().
//
// uint32_t arithmetic intentionally wraps modulo 2^32.
uint32_t strategy_a1_mix32(
  uint32_t value)
{
  value ^=
    value >> 16u;

  value *=
    0x7feb352du;

  value ^=
    value >> 15u;

  value *=
    0x846ca68bu;

  value ^=
    value >> 16u;

  return value;
}


// Host-side copy of gpu_dag_bitmap_a1_probe_bits().
ProbeBits strategy_a1_probe_bits(
  const uint32_t node,
  const uint32_t bitmap_bits)
{
  if (bitmap_bits < 32u ||
      bitmap_bits > 2048u ||
      (bitmap_bits &
       (bitmap_bits - 1u)) != 0u) {
    throw std::invalid_argument(
      "Invalid Strategy-A1 bitmap size");
  }

  const uint32_t half_bits =
    bitmap_bits >> 1u;

  const uint32_t half_mask =
    half_bits - 1u;

  const uint32_t mixed =
    strategy_a1_mix32(
      node);

  ProbeBits result;

  result.first =
    mixed &
    half_mask;

  result.second =
    half_bits +
    ((mixed >> 16u) &
     half_mask);

  return result;
}


uint64_t probe_pair_key(
  const ProbeBits& probe)
{
  return
    (
      static_cast<uint64_t>(
        probe.first) << 32u
    ) |
    static_cast<uint64_t>(
      probe.second);
}


bool probe_pairs_equal(
  const ProbeBits& lhs,
  const ProbeBits& rhs)
{
  return
    lhs.first == rhs.first &&
    lhs.second == rhs.second;
}


bool contains_node(
  const std::unordered_set<uint32_t>& excluded,
  const uint32_t node)
{
  return
    excluded.find(node) !=
    excluded.end();
}


NodePair find_full_probe_collision(
  const uint32_t bitmap_bits,
  const std::unordered_set<uint32_t>& excluded)
{
  std::unordered_map<uint64_t, uint32_t>
    first_node_for_probe;

  first_node_for_probe.reserve(
    kProbeSearchLimit / 2u);

  for (uint32_t node = kProbeSearchBegin;
       node < kProbeSearchLimit;
       ++node) {
    if (contains_node(
          excluded,
          node)) {
      continue;
    }

    const ProbeBits probe =
      strategy_a1_probe_bits(
        node,
        bitmap_bits);

    const uint64_t key =
      probe_pair_key(
        probe);

    const auto insertion =
      first_node_for_probe.emplace(
        key,
        node);

    if (!insertion.second) {
      const uint32_t previous =
        insertion.first->second;

      if (previous != node &&
          !contains_node(
            excluded,
            previous)) {
        return {
          previous,
          node
        };
      }
    }
  }

  throw std::runtime_error(
    "Failed to find a full Strategy-A1 collision");
}


NodePair find_partial_probe_collision(
  const uint32_t bitmap_bits,
  const PartialProbeMatch match,
  const std::unordered_set<uint32_t>& excluded)
{
  // key:
  //   the probe position that must match
  //
  // value.first:
  //   first node that produced the key
  //
  // value.second:
  //   the other probe position, which must differ
  std::unordered_map<
    uint32_t,
    std::pair<uint32_t, uint32_t>>
      first_node_for_probe;

  first_node_for_probe.reserve(
    kProbeSearchLimit / 2u);

  for (uint32_t node = kProbeSearchBegin;
       node < kProbeSearchLimit;
       ++node) {
    if (contains_node(
          excluded,
          node)) {
      continue;
    }

    const ProbeBits probe =
      strategy_a1_probe_bits(
        node,
        bitmap_bits);

    const uint32_t key =
      match == PartialProbeMatch::First
        ? probe.first
        : probe.second;

    const uint32_t other_probe =
      match == PartialProbeMatch::First
        ? probe.second
        : probe.first;

    const auto insertion =
      first_node_for_probe.emplace(
        key,
        std::make_pair(
          node,
          other_probe));

    if (!insertion.second) {
      const uint32_t previous_node =
        insertion.first->second.first;

      const uint32_t previous_other_probe =
        insertion.first->second.second;

      // Same requested probe, but different other probe:
      //
      //   First:
      //     same first, different second
      //
      //   Second:
      //     same second, different first
      if (previous_other_probe !=
          other_probe) {
        return {
          previous_node,
          node
        };
      }
    }
  }

  throw std::runtime_error(
    "Failed to find a partial Strategy-A1 collision");
}


uint32_t find_noncolliding_node(
  const uint32_t bitmap_bits,
  const ProbeBits& target_probe,
  const std::unordered_set<uint32_t>& excluded)
{
  for (uint32_t node = kProbeSearchBegin;
       node < kProbeSearchLimit;
       ++node) {
    if (contains_node(
          excluded,
          node)) {
      continue;
    }

    const ProbeBits probe =
      strategy_a1_probe_bits(
        node,
        bitmap_bits);

    if (!probe_pairs_equal(
          probe,
          target_probe)) {
      return node;
    }
  }

  throw std::runtime_error(
    "Failed to find a noncolliding Strategy-A1 node");
}


std::vector<uint32_t>
find_nodes_avoiding_probe_positions(
  const uint32_t bitmap_bits,
  const ProbeBits& target_probe,
  const uint32_t requested_count,
  const std::unordered_set<uint32_t>& excluded)
{
  std::vector<uint32_t> result;

  result.reserve(
    requested_count);

  for (uint32_t node = kProbeSearchBegin;
       node < kProbeSearchLimit &&
       result.size() <
         static_cast<size_t>(
           requested_count);
       ++node) {
    if (contains_node(
          excluded,
          node)) {
      continue;
    }

    const ProbeBits probe =
      strategy_a1_probe_bits(
        node,
        bitmap_bits);

    // Strategy A1 always places first and second in disjoint
    // halves. By avoiding each corresponding target position,
    // the union of all safe-parent bits cannot contain either
    // target bit.
    if (probe.first ==
          target_probe.first ||
        probe.second ==
          target_probe.second) {
      continue;
    }

    result.push_back(
      node);
  }

  if (result.size() !=
      static_cast<size_t>(
        requested_count)) {
    throw std::runtime_error(
      "Failed to find enough safe Strategy-A1 parents");
  }

  return result;
}


class TemporaryGraphFile
{
public:
  TemporaryGraphFile(
    const uint32_t num_nodes,
    const std::vector<Edge>& edges)
  {
    std::vector<std::vector<uint32_t>>
      children_by_parent(
        num_nodes);

    for (const Edge& edge : edges) {
      if (edge.parent >= num_nodes ||
          edge.child >= num_nodes) {
        throw std::invalid_argument(
          "Temporary graph edge is outside node range");
      }

      children_by_parent[
        edge.parent].push_back(
          edge.child);
    }

    static std::atomic<uint32_t>
      next_file_id{0u};

    const auto timestamp =
      std::chrono::steady_clock::now()
        .time_since_epoch()
        .count();

    const uint32_t file_id =
      next_file_id.fetch_add(
        1u,
        std::memory_order_relaxed);

    path_ =
      std::filesystem::temp_directory_path() /
      (
        "g_ext_warp_bitmap_a1_" +
        std::to_string(timestamp) +
        "_" +
        std::to_string(file_id) +
        ".json"
      );

    nlohmann::json nodes =
      nlohmann::json::object();

    for (uint32_t node = 0u;
         node < num_nodes;
         ++node) {
      nlohmann::json children =
        nlohmann::json::array();

      for (const uint32_t child :
           children_by_parent[node]) {
        // g_ext.cu interprets string children as child e-node
        // names and then resolves their eclasses.
        children.push_back(
          node_name(child));
      }

      nodes[node_name(node)] = {
        {"op", "test"},
        {"cost", 1.0},
        {"eclass", eclass_name(node)},
        {"children", std::move(children)}
      };
    }

    // Keep every one-node eclass alive and preserve the
    // zero-padded numeric e-node ordering after pruning.
    nlohmann::json roots =
      nlohmann::json::array();

    for (uint32_t node = 0u;
         node < num_nodes;
         ++node) {
      roots.push_back(
        eclass_name(node));
    }

    nlohmann::json document;

    document["nodes"] =
      std::move(nodes);

    document["root_eclasses"] =
      std::move(roots);

    std::ofstream output(path_);

    if (!output.is_open()) {
      throw std::runtime_error(
        "Failed to create temporary test graph");
    }

    output
      << document.dump(2)
      << '\n';

    if (!output.good()) {
      throw std::runtime_error(
        "Failed to write temporary test graph");
    }
  }

  TemporaryGraphFile(
    const TemporaryGraphFile&) = delete;

  TemporaryGraphFile&
  operator=(
    const TemporaryGraphFile&) = delete;

  ~TemporaryGraphFile()
  {
    std::error_code error;

    std::filesystem::remove(
      path_,
      error);
  }

  std::string path() const
  {
    return path_.string();
  }

private:
  std::filesystem::path path_;
};


StrategyGraphCase make_strategy_graph_case(
  const uint32_t bitmap_bits)
{
  std::unordered_set<uint32_t>
    excluded{
      0u,
      1u
    };

  const NodePair full_collision =
    find_full_probe_collision(
      bitmap_bits,
      excluded);

  excluded.insert(
    full_collision.parent);

  excluded.insert(
    full_collision.child);

  const NodePair first_probe_collision =
    find_partial_probe_collision(
      bitmap_bits,
      PartialProbeMatch::First,
      excluded);

  excluded.insert(
    first_probe_collision.parent);

  excluded.insert(
    first_probe_collision.child);

  const NodePair second_probe_collision =
    find_partial_probe_collision(
      bitmap_bits,
      PartialProbeMatch::Second,
      excluded);

  excluded.insert(
    second_probe_collision.parent);

  excluded.insert(
    second_probe_collision.child);

  const uint32_t maximum_data_node =
    std::max({
      full_collision.parent,
      full_collision.child,
      first_probe_collision.parent,
      first_probe_collision.child,
      second_probe_collision.parent,
      second_probe_collision.child
    });

  const uint32_t full_candidate =
    maximum_data_node + 1u;

  const uint32_t first_candidate =
    maximum_data_node + 2u;

  const uint32_t second_candidate =
    maximum_data_node + 3u;

  StrategyGraphCase result;

  result.bitmap_bits =
    bitmap_bits;

  result.full_collision =
    full_collision;

  result.first_probe_only_collision =
    first_probe_collision;

  result.second_probe_only_collision =
    second_probe_collision;

  result.full_collision_candidate =
    full_candidate;

  result.first_probe_candidate =
    first_candidate;

  result.second_probe_candidate =
    second_candidate;

  result.node_count =
    second_candidate + 1u;

  result.edges = {
    {
      full_collision.parent,
      full_candidate
    },
    {
      full_candidate,
      full_collision.child
    },
    {
      first_probe_collision.parent,
      first_candidate
    },
    {
      first_candidate,
      first_probe_collision.child
    },
    {
      second_probe_collision.parent,
      second_candidate
    },
    {
      second_candidate,
      second_probe_collision.child
    }
  };

  return result;
}


CandidateState make_candidate_state(
  const uint32_t node_count,
  const uint32_t candidate,
  const uint32_t parent,
  const uint32_t child)
{
  if (candidate >= node_count ||
      parent >= node_count ||
      child >= node_count) {
    throw std::invalid_argument(
      "Candidate-state node is outside graph range");
  }

  if (candidate == parent ||
      candidate == child ||
      parent == child) {
    throw std::invalid_argument(
      "Candidate, parent, and child must be distinct");
  }

  CandidateState result;

  result.selected_state.assign(
    node_count,
    0u);

  result.level.assign(
    node_count,
    -1);

  result.selected_state[parent] =
    1u;

  result.selected_state[child] =
    1u;

  result.level[parent] =
    static_cast<int>(
      kCandidateParentLevel);

  result.level[child] =
    static_cast<int>(
      kCandidateChildLevel);

  return result;
}


void check_result_equal(
  const G_EXT::GpuInsertionResult& lhs,
  const G_EXT::GpuInsertionResult& rhs)
{
  DOCTEST_CHECK(
    lhs.selected_state ==
    rhs.selected_state);

  DOCTEST_CHECK(
    lhs.level ==
    rhs.level);

  DOCTEST_CHECK(
    lhs.u0 ==
    rhs.u0);
}


void check_candidate_accepted(
  const G_EXT::GpuInsertionResult& result,
  const uint32_t candidate)
{
  DOCTEST_REQUIRE(
    candidate <
    result.selected_state.size());

  DOCTEST_REQUIRE(
    candidate <
    result.level.size());

  DOCTEST_CHECK(
    result.selected_state[candidate] ==
    2u);

  DOCTEST_CHECK(
    result.level[candidate] ==
    kExpectedCandidateLevel);

  DOCTEST_CHECK(
    result.u0 ==
    1u);
}


void check_candidate_rejected(
  const G_EXT::GpuInsertionResult& result,
  const uint32_t candidate)
{
  DOCTEST_REQUIRE(
    candidate <
    result.selected_state.size());

  DOCTEST_REQUIRE(
    candidate <
    result.level.size());

  DOCTEST_CHECK(
    result.selected_state[candidate] ==
    0u);

  DOCTEST_CHECK(
    result.level[candidate] ==
    -1);

  DOCTEST_CHECK(
    result.u0 ==
    0u);
}

}  // namespace


DOCTEST_TEST_CASE(
  "Warp bitmap accepts every supported runtime size")
{
  constexpr uint32_t node_count =
    8u;

  const TemporaryGraphFile graph(
    node_count,
    {});

  G_EXT extension;

  extension.read(
    graph.path());

  const std::vector<uint32_t>
    selected_state(
      node_count,
      1u);

  const std::vector<int>
    level(
      node_count,
      0);

  for (const uint32_t bitmap_bits :
       {
         0u,
         32u,
         64u,
         128u,
         256u,
         512u,
         1024u,
         2048u
       }) {
    DOCTEST_CAPTURE(
      bitmap_bits);

    DOCTEST_CHECK_NOTHROW(
      extension
        .unittest_gpu_insert_warp_direct_once(
          selected_state,
          level,
          true,
          64u,
          bitmap_bits));
  }

  for (const uint32_t bitmap_bits :
       {
         1u,
         16u,
         96u,
         192u,
         4096u
       }) {
    DOCTEST_CAPTURE(
      bitmap_bits);

    DOCTEST_CHECK_THROWS_AS(
      extension
        .unittest_gpu_insert_warp_direct_once(
          selected_state,
          level,
          true,
          64u,
          bitmap_bits),
      std::invalid_argument);
  }
}


DOCTEST_TEST_CASE(
  "Every nonzero Warp bitmap uses the same Strategy-A1 "
  "two-probe policy as Block")
{
  for (const uint32_t bitmap_bits :
       kSupportedBitmapBits) {
    DOCTEST_CAPTURE(
      bitmap_bits);

    const StrategyGraphCase test_case =
      make_strategy_graph_case(
        bitmap_bits);

    const TemporaryGraphFile graph(
      test_case.node_count,
      test_case.edges);

    G_EXT extension;

    extension.read(
      graph.path());

    // ==========================================================
    // Full two-probe collision
    //
    // Exact overlap sees different e-node IDs and accepts.
    //
    // Strategy A1 sees both probe positions and rejects.
    // ==========================================================
    {
      const CandidateState input =
        make_candidate_state(
          test_case.node_count,
          test_case.full_collision_candidate,
          test_case.full_collision.parent,
          test_case.full_collision.child);

      const ProbeBits parent_probe =
        strategy_a1_probe_bits(
          test_case.full_collision.parent,
          bitmap_bits);

      const ProbeBits child_probe =
        strategy_a1_probe_bits(
          test_case.full_collision.child,
          bitmap_bits);

      DOCTEST_REQUIRE(
        test_case.full_collision.parent !=
        test_case.full_collision.child);

      DOCTEST_REQUIRE(
        probe_pairs_equal(
          parent_probe,
          child_probe));

      const auto exact =
        extension
          .unittest_gpu_insert_warp_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            0u);

      const auto warp_bitmap =
        extension
          .unittest_gpu_insert_warp_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      const auto block_bitmap =
        extension
          .unittest_gpu_insert_block_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      check_candidate_accepted(
        exact,
        test_case.full_collision_candidate);

      check_candidate_rejected(
        warp_bitmap,
        test_case.full_collision_candidate);

      check_candidate_rejected(
        block_bitmap,
        test_case.full_collision_candidate);

      check_result_equal(
        warp_bitmap,
        block_bitmap);
    }

    // ==========================================================
    // Only the lower-half probe collides.
    //
    // Strategy A1 requires both probes, so the candidate must
    // remain accepted.
    // ==========================================================
    {
      const CandidateState input =
        make_candidate_state(
          test_case.node_count,
          test_case.first_probe_candidate,
          test_case
            .first_probe_only_collision
            .parent,
          test_case
            .first_probe_only_collision
            .child);

      const ProbeBits parent_probe =
        strategy_a1_probe_bits(
          test_case
            .first_probe_only_collision
            .parent,
          bitmap_bits);

      const ProbeBits child_probe =
        strategy_a1_probe_bits(
          test_case
            .first_probe_only_collision
            .child,
          bitmap_bits);

      DOCTEST_REQUIRE(
        parent_probe.first ==
        child_probe.first);

      DOCTEST_REQUIRE(
        parent_probe.second !=
        child_probe.second);

      const auto warp_bitmap =
        extension
          .unittest_gpu_insert_warp_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      const auto block_bitmap =
        extension
          .unittest_gpu_insert_block_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      check_candidate_accepted(
        warp_bitmap,
        test_case.first_probe_candidate);

      check_candidate_accepted(
        block_bitmap,
        test_case.first_probe_candidate);

      check_result_equal(
        warp_bitmap,
        block_bitmap);
    }

    // ==========================================================
    // Only the upper-half probe collides.
    //
    // This must also remain accepted.
    // ==========================================================
    {
      const CandidateState input =
        make_candidate_state(
          test_case.node_count,
          test_case.second_probe_candidate,
          test_case
            .second_probe_only_collision
            .parent,
          test_case
            .second_probe_only_collision
            .child);

      const ProbeBits parent_probe =
        strategy_a1_probe_bits(
          test_case
            .second_probe_only_collision
            .parent,
          bitmap_bits);

      const ProbeBits child_probe =
        strategy_a1_probe_bits(
          test_case
            .second_probe_only_collision
            .child,
          bitmap_bits);

      DOCTEST_REQUIRE(
        parent_probe.first !=
        child_probe.first);

      DOCTEST_REQUIRE(
        parent_probe.second ==
        child_probe.second);

      const auto warp_bitmap =
        extension
          .unittest_gpu_insert_warp_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      const auto block_bitmap =
        extension
          .unittest_gpu_insert_block_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      check_candidate_accepted(
        warp_bitmap,
        test_case.second_probe_candidate);

      check_candidate_accepted(
        block_bitmap,
        test_case.second_probe_candidate);

      check_result_equal(
        warp_bitmap,
        block_bitmap);
    }
  }
}


DOCTEST_TEST_CASE(
  "Unselected and invalid-level parents do not populate "
  "Strategy-A1 Warp bitmaps")
{
  for (const uint32_t bitmap_bits :
       kSupportedBitmapBits) {
    DOCTEST_CAPTURE(
      bitmap_bits);

    std::unordered_set<uint32_t>
      excluded{
        0u,
        1u
      };

    const NodePair collision =
      find_full_probe_collision(
        bitmap_bits,
        excluded);

    excluded.insert(
      collision.parent);

    excluded.insert(
      collision.child);

    const ProbeBits child_probe =
      strategy_a1_probe_bits(
        collision.child,
        bitmap_bits);

    const uint32_t valid_parent =
      find_noncolliding_node(
        bitmap_bits,
        child_probe,
        excluded);

    const uint32_t candidate =
      std::max({
        collision.parent,
        collision.child,
        valid_parent
      }) + 1u;

    const uint32_t node_count =
      candidate + 1u;

    const std::vector<Edge> edges{
      {
        collision.parent,
        candidate
      },
      {
        valid_parent,
        candidate
      },
      {
        candidate,
        collision.child
      }
    };

    const TemporaryGraphFile graph(
      node_count,
      edges);

    G_EXT extension;

    extension.read(
      graph.path());

    // ----------------------------------------------------------
    // The colliding parent is unselected.
    // ----------------------------------------------------------
    {
      CandidateState input;

      input.selected_state.assign(
        node_count,
        0u);

      input.level.assign(
        node_count,
        -1);

      input.selected_state[valid_parent] =
        1u;

      input.selected_state[collision.child] =
        1u;

      input.level[valid_parent] =
        static_cast<int>(
          kCandidateParentLevel);

      input.level[collision.child] =
        static_cast<int>(
          kCandidateChildLevel);

      const auto warp_result =
        extension
          .unittest_gpu_insert_warp_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      const auto block_result =
        extension
          .unittest_gpu_insert_block_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      check_candidate_accepted(
        warp_result,
        candidate);

      check_candidate_accepted(
        block_result,
        candidate);

      check_result_equal(
        warp_result,
        block_result);
    }

    // ----------------------------------------------------------
    // The colliding parent is selected but has level -1.
    // ----------------------------------------------------------
    {
      CandidateState input;

      input.selected_state.assign(
        node_count,
        0u);

      input.level.assign(
        node_count,
        -1);

      input.selected_state[valid_parent] =
        1u;

      input.selected_state[collision.parent] =
        1u;

      input.selected_state[collision.child] =
        1u;

      input.level[valid_parent] =
        static_cast<int>(
          kCandidateParentLevel);

      // collision.parent intentionally remains level -1.

      input.level[collision.child] =
        static_cast<int>(
          kCandidateChildLevel);

      const auto warp_result =
        extension
          .unittest_gpu_insert_warp_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      const auto block_result =
        extension
          .unittest_gpu_insert_block_direct_once(
            input.selected_state,
            input.level,
            true,
            64u,
            bitmap_bits);

      check_candidate_accepted(
        warp_result,
        candidate);

      check_candidate_accepted(
        block_result,
        candidate);

      check_result_equal(
        warp_result,
        block_result);
    }
  }
}


DOCTEST_TEST_CASE(
  "Warp Bitmap Batch and Direct use identical Strategy-A1 "
  "behavior")
{
  for (const uint32_t bitmap_bits :
       kSupportedBitmapBits) {
    DOCTEST_CAPTURE(
      bitmap_bits);

    const StrategyGraphCase test_case =
      make_strategy_graph_case(
        bitmap_bits);

    const TemporaryGraphFile graph(
      test_case.node_count,
      test_case.edges);

    G_EXT extension;

    extension.read(
      graph.path());

    // Use a one-probe-only collision. Strategy A1 must accept
    // it, making this test sensitive to both the hash policy and
    // the complete candidate publication result.
    const CandidateState input =
      make_candidate_state(
        test_case.node_count,
        test_case.first_probe_candidate,
        test_case
          .first_probe_only_collision
          .parent,
        test_case
          .first_probe_only_collision
          .child);

    for (const bool is_top_down_insertion :
         {
           true,
           false
         }) {
      DOCTEST_CAPTURE(
        is_top_down_insertion);

      const auto batch =
        extension
          .unittest_gpu_insert_warp_batch_once(
            input.selected_state,
            input.level,
            is_top_down_insertion,
            64u,
            bitmap_bits);

      const auto direct =
        extension
          .unittest_gpu_insert_warp_direct_once(
            input.selected_state,
            input.level,
            is_top_down_insertion,
            64u,
            bitmap_bits);

      check_result_equal(
        batch,
        direct);

      check_candidate_accepted(
        batch,
        test_case.first_probe_candidate);
    }
  }
}


DOCTEST_TEST_CASE(
  "Warp-local Strategy-A1 bitmaps do not leak between "
  "warps in one CUDA block")
{
  // Cover both register and shared-memory implementations.
  for (const uint32_t bitmap_bits :
       {
         32u,
         64u,
         128u,
         2048u
       }) {
    DOCTEST_CAPTURE(
      bitmap_bits);

    std::unordered_set<uint32_t>
      excluded{
        0u,
        1u
      };

    const NodePair collision =
      find_full_probe_collision(
        bitmap_bits,
        excluded);

    excluded.insert(
      collision.parent);

    excluded.insert(
      collision.child);

    const ProbeBits target_probe =
      strategy_a1_probe_bits(
        collision.parent,
        bitmap_bits);

    // Candidate 1 receives a deliberately long parent scan.
    //
    // None of these parents sets either of the target child
    // probe positions. Therefore candidate 1's own bitmap cannot
    // reject its child.
    const std::vector<uint32_t>
      safe_parents =
        find_nodes_avoiding_probe_positions(
          bitmap_bits,
          target_probe,
          512u,
          excluded);

    uint32_t maximum_node =
      std::max(
        collision.parent,
        collision.child);

    for (const uint32_t parent :
         safe_parents) {
      maximum_node =
        std::max(
          maximum_node,
          parent);
    }

    const uint32_t node_count =
      maximum_node + 1u;

    std::vector<Edge> edges{
      // Candidate 0 has a local full Strategy-A1 collision.
      {
        collision.parent,
        0u
      },
      {
        0u,
        collision.child
      },

      // Candidate 1 probes the same child probe pair but its own
      // parents deliberately avoid both target positions.
      {
        1u,
        collision.child
      }
    };

    for (const uint32_t parent :
         safe_parents) {
      edges.push_back({
        parent,
        1u
      });
    }

    const TemporaryGraphFile graph(
      node_count,
      edges);

    G_EXT extension;

    extension.read(
      graph.path());

    CandidateState input;

    input.selected_state.assign(
      node_count,
      0u);

    input.level.assign(
      node_count,
      -1);

    input.selected_state[collision.parent] =
      1u;

    input.level[collision.parent] =
      static_cast<int>(
        kCandidateParentLevel);

    input.selected_state[collision.child] =
      1u;

    input.level[collision.child] =
      static_cast<int>(
        kCandidateChildLevel);

    for (const uint32_t parent :
         safe_parents) {
      input.selected_state[parent] =
        1u;

      input.level[parent] =
        static_cast<int>(
          kCandidateParentLevel);
    }

    for (const uint32_t threads_per_block :
         {
           64u,
           128u
         }) {
      DOCTEST_CAPTURE(
        threads_per_block);

      const auto result =
        extension
          .unittest_gpu_insert_warp_direct_once(
            input.selected_state,
            input.level,
            true,
            threads_per_block,
            bitmap_bits);

      // Candidate 0 sees its own complete probe collision.
      DOCTEST_CHECK(
        result.selected_state[0u] ==
        0u);

      DOCTEST_CHECK(
        result.level[0u] ==
        -1);

      // Candidate 1 has no complete collision in its own
      // register/shared-memory bitmap.
      //
      // It must not observe candidate 0's two probe bits.
      DOCTEST_CHECK(
        result.selected_state[1u] ==
        2u);

      DOCTEST_CHECK(
        result.level[1u] ==
        kExpectedCandidateLevel);

      DOCTEST_CHECK(
        result.u0 ==
        1u);
    }
  }
}


DOCTEST_TEST_CASE(
  "BITMAP_BITS zero remains Exact Warp without Strategy-A1 "
  "hash rejection")
{
  const StrategyGraphCase test_case =
    make_strategy_graph_case(
      32u);

  const TemporaryGraphFile graph(
    test_case.node_count,
    test_case.edges);

  G_EXT extension;

  extension.read(
    graph.path());

  const CandidateState input =
    make_candidate_state(
      test_case.node_count,
      test_case.full_collision_candidate,
      test_case.full_collision.parent,
      test_case.full_collision.child);

  // Both e-node IDs produce the same Strategy-A1 probes, but
  // Exact Warp compares the actual IDs and must accept them.
  const auto implicit_exact =
    extension
      .unittest_gpu_insert_warp_direct_once(
        input.selected_state,
        input.level,
        true,
        64u);

  const auto explicit_exact =
    extension
      .unittest_gpu_insert_warp_direct_once(
        input.selected_state,
        input.level,
        true,
        64u,
        0u);

  check_result_equal(
    implicit_exact,
    explicit_exact);

  check_candidate_accepted(
    implicit_exact,
    test_case.full_collision_candidate);
}