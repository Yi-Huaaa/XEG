#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <g_ext/G_EXT.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

std::string make_indexed_name(
  const char prefix,
  const uint32_t index)
{
  std::ostringstream stream;

  stream
    << prefix
    << std::setw(2)
    << std::setfill('0')
    << index;

  return stream.str();
}


// Create this graph:
//
//   parent ID 1
//       |
//       v
//   candidate ID 0
//       |
//       v
//   child ID 33
//
// Candidate 0 therefore has:
//
//   selected parent ID = 1, level = 0
//   selected child  ID = 33, level = 2
//
// Exact overlap:
//   1 != 33
//
// 32-bit bitmap:
//
//   1  & 31 = 1
//   33 & 31 = 1
//
// Therefore the 32-bit bitmap conservatively rejects candidate
// 0 because of a deliberate hash collision.
//
// Bitmap sizes >=64 do not collide for this pair.
nlohmann::json make_collision_graph()
{
  constexpr uint32_t kNumNodes =
    34u;

  nlohmann::json graph;

  graph["nodes"] =
    nlohmann::json::object();

  graph["root_eclasses"] =
    nlohmann::json::array();

  for (uint32_t index = 0u;
       index < kNumNodes;
       ++index) {
    const std::string node_name =
      make_indexed_name(
        'n',
        index);

    const std::string eclass_name =
      make_indexed_name(
        'e',
        index);

    nlohmann::json node;

    node["op"] =
      "test";

    node["cost"] =
      1.0;

    node["eclass"] =
      eclass_name;

    node["children"] =
      nlohmann::json::array();

    graph["nodes"][node_name] =
      node;

    // Keep every eclass alive so the pruned e-node IDs remain:
    //
    //   n00 -> 0
    //   n01 -> 1
    //   ...
    //   n33 -> 33
    graph["root_eclasses"].push_back(
      eclass_name);
  }

  // parent ID 1 -> candidate ID 0
  graph["nodes"]["n01"]["children"].push_back(
    "n00");

  // candidate ID 0 -> child ID 33
  graph["nodes"]["n00"]["children"].push_back(
    "n33");

  return graph;
}


// Create this graph:
//
//   node 1 -> candidate 0
//   candidate 0 -> node 1
//
// Node 1 appears in both the parent and child CSR segments of
// candidate 0.
nlohmann::json make_true_overlap_graph()
{
  nlohmann::json graph;

  graph["nodes"] =
    nlohmann::json::object();

  graph["root_eclasses"] =
    nlohmann::json::array({
      "e00",
      "e01"
    });

  graph["nodes"]["n00"] = {
    {"op", "candidate"},
    {"cost", 1.0},
    {"eclass", "e00"},
    {"children", nlohmann::json::array({"n01"})}
  };

  graph["nodes"]["n01"] = {
    {"op", "neighbor"},
    {"cost", 1.0},
    {"eclass", "e01"},
    {"children", nlohmann::json::array({"n00"})}
  };

  return graph;
}


class TemporaryJsonFile
{
public:
  explicit TemporaryJsonFile(
    const nlohmann::json& graph)
  {
    static std::atomic<uint64_t>
      sequence{0u};

    const uint64_t timestamp =
      static_cast<uint64_t>(
        std::chrono::high_resolution_clock::
          now().
          time_since_epoch().
          count());

    const uint64_t suffix =
      sequence.fetch_add(
        1u);

    std::ostringstream filename;

    filename
      << "g_ext_block_bitmap_"
      << timestamp
      << "_"
      << suffix
      << ".json";

    path_ =
      std::filesystem::temp_directory_path() /
      filename.str();

    std::ofstream output(
      path_);

    if (!output.is_open()) {
      throw std::runtime_error(
        "Failed to create temporary JSON test file");
    }

    output
      << graph.dump(2)
      << "\n";
  }

  ~TemporaryJsonFile()
  {
    std::error_code error;

    std::filesystem::remove(
      path_,
      error);
  }

  TemporaryJsonFile(
    const TemporaryJsonFile&) = delete;

  TemporaryJsonFile& operator=(
    const TemporaryJsonFile&) = delete;

  const std::string path() const
  {
    return path_.string();
  }

private:
  std::filesystem::path path_;
};


std::vector<uint32_t>
make_collision_selected_state()
{
  std::vector<uint32_t>
    selected_state(
      34u,
      0u);

  // Selected parent ID 1.
  selected_state[1u] =
    1u;

  // Selected child ID 33.
  selected_state[33u] =
    1u;

  return selected_state;
}


std::vector<int>
make_collision_level()
{
  std::vector<int>
    level(
      34u,
      -1);

  level[1u] =
    0;

  level[33u] =
    2;

  return level;
}


void check_candidate_accepted(
  const G_EXT::GpuInsertionResult& result)
{
  REQUIRE(
    result.selected_state.size() >
    0u);

  REQUIRE(
    result.level.size() >
    0u);

  CHECK(
    result.selected_state[0u] ==
    2u);

  CHECK(
    result.level[0u] ==
    1);

  CHECK(
    result.u0 ==
    1u);
}


void check_candidate_rejected(
  const G_EXT::GpuInsertionResult& result)
{
  REQUIRE(
    result.selected_state.size() >
    0u);

  REQUIRE(
    result.level.size() >
    0u);

  CHECK(
    result.selected_state[0u] ==
    0u);

  CHECK(
    result.level[0u] ==
    -1);

  CHECK(
    result.u0 ==
    0u);
}

}  // namespace


TEST_CASE(
  "Block-Direct supports every requested bitmap size")
{
  TemporaryJsonFile graph_file(
    make_collision_graph());

  G_EXT g_ext;

  g_ext.read(
    graph_file.path());

  const std::vector<uint32_t> selected_state =
    make_collision_selected_state();

  const std::vector<int> level =
    make_collision_level();

  constexpr std::array<uint32_t, 8u>
    bitmap_sizes{
      0u,
      32u,
      64u,
      128u,
      256u,
      512u,
      1024u,
      2048u
    };

  for (const uint32_t bitmap_bits :
       bitmap_sizes) {
    CAPTURE(
      bitmap_bits);

    const auto result =
      g_ext.unittest_gpu_insert_block_direct_once(
        selected_state,
        level,
        true,
        64u,
        bitmap_bits);

    if (bitmap_bits == 32u) {
      // Deliberate collision:
      //
      //   1 & 31  == 1
      //   33 & 31 == 1
      check_candidate_rejected(
        result);
    }
    else {
      // Exact mode and every bitmap >=64 distinguish IDs 1
      // and 33.
      check_candidate_accepted(
        result);
    }
  }
}


TEST_CASE(
  "Block-Batch uses the same bitmap collision semantics")
{
  TemporaryJsonFile graph_file(
    make_collision_graph());

  G_EXT g_ext;

  g_ext.read(
    graph_file.path());

  const std::vector<uint32_t> selected_state =
    make_collision_selected_state();

  const std::vector<int> level =
    make_collision_level();

  SUBCASE(
    "BITMAP_BITS=0 uses the exact Block-Batch path")
  {
    const auto result =
      g_ext.unittest_gpu_insert_block_batch_once(
        selected_state,
        level,
        true,
        64u,
        0u);

    check_candidate_accepted(
      result);
  }

  SUBCASE(
    "32-bit bitmap conservatively rejects the collision")
  {
    const auto result =
      g_ext.unittest_gpu_insert_block_batch_once(
        selected_state,
        level,
        true,
        64u,
        32u);

    check_candidate_rejected(
      result);
  }

  SUBCASE(
    "64-bit bitmap distinguishes IDs 1 and 33")
  {
    const auto result =
      g_ext.unittest_gpu_insert_block_batch_once(
        selected_state,
        level,
        true,
        64u,
        64u);

    check_candidate_accepted(
      result);
  }

  SUBCASE(
    "2048-bit bitmap distinguishes IDs 1 and 33")
  {
    const auto result =
      g_ext.unittest_gpu_insert_block_batch_once(
        selected_state,
        level,
        true,
        64u,
        2048u);

    check_candidate_accepted(
      result);
  }
}


TEST_CASE(
  "True parent-child overlap remains rejected")
{
  TemporaryJsonFile graph_file(
    make_true_overlap_graph());

  G_EXT g_ext;

  g_ext.read(
    graph_file.path());

  const std::vector<uint32_t>
    selected_state{
      0u,
      1u
    };

  const std::vector<int>
    level{
      -1,
      0
    };

  constexpr std::array<uint32_t, 3u>
    bitmap_sizes{
      0u,
      32u,
      2048u
    };

  for (const uint32_t bitmap_bits :
       bitmap_sizes) {
    CAPTURE(
      bitmap_bits);

    const auto batch_result =
      g_ext.unittest_gpu_insert_block_batch_once(
        selected_state,
        level,
        true,
        64u,
        bitmap_bits);

    check_candidate_rejected(
      batch_result);

    const auto direct_result =
      g_ext.unittest_gpu_insert_block_direct_once(
        selected_state,
        level,
        true,
        64u,
        bitmap_bits);

    check_candidate_rejected(
      direct_result);
  }
}


TEST_CASE(
  "Invalid Block bitmap sizes are rejected")
{
  TemporaryJsonFile graph_file(
    make_collision_graph());

  G_EXT g_ext;

  g_ext.read(
    graph_file.path());

  const std::vector<uint32_t> selected_state =
    make_collision_selected_state();

  const std::vector<int> level =
    make_collision_level();

  constexpr std::array<uint32_t, 5u>
    invalid_bitmap_sizes{
      1u,
      16u,
      31u,
      96u,
      4096u
    };

  for (const uint32_t bitmap_bits :
       invalid_bitmap_sizes) {
    CAPTURE(
      bitmap_bits);

    CHECK_THROWS_AS(
      g_ext.unittest_gpu_insert_block_direct_once(
        selected_state,
        level,
        true,
        64u,
        bitmap_bits),
      std::invalid_argument);

    CHECK_THROWS_AS(
      g_ext.unittest_gpu_insert_block_batch_once(
        selected_state,
        level,
        true,
        64u,
        bitmap_bits),
      std::invalid_argument);
  }
}