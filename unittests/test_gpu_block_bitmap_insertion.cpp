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
#include <stdexcept>
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


// Create an indexed graph with one e-node per e-class.
//
// Every e-class is also listed as a root so that pruning keeps
// all e-nodes and preserves the intended IDs:
//
//   n00 -> e-node 0
//   n01 -> e-node 1
//   ...
nlohmann::json make_indexed_graph(
  const uint32_t num_nodes)
{
  nlohmann::json graph;

  graph["nodes"] =
    nlohmann::json::object();

  graph["root_eclasses"] =
    nlohmann::json::array();

  for (uint32_t index = 0u;
       index < num_nodes;
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

    graph["root_eclasses"].push_back(
      eclass_name);
  }

  return graph;
}


// Create this graph:
//
//   selected parent ID 1
//            |
//            v
//      candidate ID 0
//            |
//            v
//   selected child ID 33
//
// Candidate 0 has:
//
//   selected parent ID = 1,  level = 0
//   selected child  ID = 33, level = 2
//
// Exact overlap:
//
//   1 != 33
//
// 32-bit bitmap:
//
//   1  & 31 = 1
//   33 & 31 = 1
//
// Therefore a 32-bit bitmap conservatively rejects candidate 0
// because of a real hash collision between one selected valid
// parent and one selected valid child.
//
// Bitmap sizes >= 64 distinguish IDs 1 and 33.
nlohmann::json make_selected_parent_collision_graph()
{
  constexpr uint32_t kNumNodes =
    34u;

  nlohmann::json graph =
    make_indexed_graph(
      kNumNodes);

  // Parent ID 1 -> candidate ID 0.
  graph["nodes"]["n01"]["children"].push_back(
    "n00");

  // Candidate ID 0 -> child ID 33.
  graph["nodes"]["n00"]["children"].push_back(
    "n33");

  return graph;
}


// Create this graph:
//
//   unselected parent ID 1 ----+
//                              |
//   selected parent ID 2 ------+
//                              v
//                       candidate ID 0
//                              |
//                              v
//                       selected child ID 33
//
// Runtime state:
//
//   parent 1:
//     selected_state = 0
//     level = -1
//
//   parent 2:
//     selected_state = 1
//     level = 0
//
//   child 33:
//     selected_state = 1
//     level = 2
//
// Exact selected-parent/selected-child overlap:
//
//   {2} intersection {33} = empty
//
// Candidate 0 must therefore be accepted.
//
// However, the incorrect implementation inserts every static
// parent into the bitmap:
//
//   static parent 1 hashes to bit 1
//   selected child 33 hashes to bit 1
//
// That incorrect behavior rejects candidate 0 with a 32-bit
// bitmap. This graph is the main regression test for the bug.
nlohmann::json make_unselected_parent_collision_graph()
{
  constexpr uint32_t kNumNodes =
    34u;

  nlohmann::json graph =
    make_indexed_graph(
      kNumNodes);

  // Unselected parent ID 1 -> candidate ID 0.
  graph["nodes"]["n01"]["children"].push_back(
    "n00");

  // Selected parent ID 2 -> candidate ID 0.
  graph["nodes"]["n02"]["children"].push_back(
    "n00");

  // Candidate ID 0 -> selected child ID 33.
  graph["nodes"]["n00"]["children"].push_back(
    "n33");

  return graph;
}


// This graph has the same static topology as the unselected
// parent collision graph:
//
//   parent ID 1 --------+
//                       |
//   parent ID 2 --------+
//                       v
//                candidate ID 0
//                       |
//                       v
//                child ID 33
//
// Runtime state differs:
//
//   parent 1:
//     selected_state = 1
//     level = -1
//
//   parent 2:
//     selected_state = 1
//     level = 0
//
//   child 33:
//     selected_state = 1
//     level = 2
//
// The exact implementation ignores parent 1 because its level
// is invalid. Therefore the effective selected valid parent set
// is still:
//
//   {2}
//
// Candidate 0 must be accepted even with a 32-bit bitmap.
//
// The bitmap must not include a parent whose level is negative.
nlohmann::json make_invalid_level_parent_collision_graph()
{
  return
    make_unselected_parent_collision_graph();
}


// Create this graph:
//
//   node ID 1 -> candidate ID 0
//   candidate ID 0 -> node ID 1
//
// Node 1 appears in both the parent and child CSR segments of
// candidate 0.
//
// When node 1 is selected with a valid level, candidate 0 must
// be rejected by both the exact implementation and every bitmap
// implementation.
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

  std::string path() const
  {
    return path_.string();
  }

private:
  std::filesystem::path path_;
};


std::vector<uint32_t>
make_selected_parent_collision_state()
{
  std::vector<uint32_t>
    selected_state(
      34u,
      0u);

  // Selected valid parent ID 1.
  selected_state[1u] =
    1u;

  // Selected valid child ID 33.
  selected_state[33u] =
    1u;

  return selected_state;
}


std::vector<int>
make_selected_parent_collision_level()
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


std::vector<uint32_t>
make_unselected_parent_collision_state()
{
  std::vector<uint32_t>
    selected_state(
      34u,
      0u);

  // Parent ID 1 remains unselected.
  selected_state[1u] =
    0u;

  // Selected valid parent ID 2.
  selected_state[2u] =
    1u;

  // Selected valid child ID 33.
  selected_state[33u] =
    1u;

  return selected_state;
}


std::vector<int>
make_unselected_parent_collision_level()
{
  std::vector<int>
    level(
      34u,
      -1);

  level[2u] =
    0;

  level[33u] =
    2;

  return level;
}


std::vector<uint32_t>
make_invalid_level_parent_collision_state()
{
  std::vector<uint32_t>
    selected_state(
      34u,
      0u);

  // Parent ID 1 is selected, but its level remains invalid.
  selected_state[1u] =
    1u;

  // Parent ID 2 is selected and valid.
  selected_state[2u] =
    1u;

  // Child ID 33 is selected and valid.
  selected_state[33u] =
    1u;

  return selected_state;
}


std::vector<int>
make_invalid_level_parent_collision_level()
{
  std::vector<int>
    level(
      34u,
      -1);

  // Parent ID 1 intentionally remains at level -1.
  level[1u] =
    -1;

  level[2u] =
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


void run_block_batch_and_direct(
  G_EXT& g_ext,
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& level,
  const uint32_t threads_per_candidate,
  const uint32_t bitmap_bits,
  const bool expect_accepted)
{
  CAPTURE(
    threads_per_candidate);

  CAPTURE(
    bitmap_bits);

  const auto batch_result =
    g_ext.unittest_gpu_insert_block_batch_once(
      selected_state,
      level,
      true,
      threads_per_candidate,
      bitmap_bits);

  if (expect_accepted) {
    check_candidate_accepted(
      batch_result);
  }
  else {
    check_candidate_rejected(
      batch_result);
  }

  const auto direct_result =
    g_ext.unittest_gpu_insert_block_direct_once(
      selected_state,
      level,
      true,
      threads_per_candidate,
      bitmap_bits);

  if (expect_accepted) {
    check_candidate_accepted(
      direct_result);
  }
  else {
    check_candidate_rejected(
      direct_result);
  }
}

}  // namespace


TEST_CASE(
  "Selected valid parent-child hash collisions are "
  "conservatively rejected")
{
  TemporaryJsonFile graph_file(
    make_selected_parent_collision_graph());

  G_EXT g_ext;

  g_ext.read(
    graph_file.path());

  const std::vector<uint32_t> selected_state =
    make_selected_parent_collision_state();

  const std::vector<int> level =
    make_selected_parent_collision_level();

  constexpr std::array<uint32_t, 2u>
    thread_counts{
      64u,
      128u
    };

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

  for (const uint32_t threads_per_candidate :
       thread_counts) {
    for (const uint32_t bitmap_bits :
         bitmap_sizes) {
      // Exact mode accepts because IDs 1 and 33 differ.
      //
      // A 32-bit bitmap rejects because both selected valid
      // IDs hash to bit 1.
      //
      // Every larger bitmap distinguishes the two IDs.
      const bool expect_accepted =
        bitmap_bits != 32u;

      run_block_batch_and_direct(
        g_ext,
        selected_state,
        level,
        threads_per_candidate,
        bitmap_bits,
        expect_accepted);
    }
  }
}


TEST_CASE(
  "Unselected parent IDs do not populate the Block bitmap")
{
  TemporaryJsonFile graph_file(
    make_unselected_parent_collision_graph());

  G_EXT g_ext;

  g_ext.read(
    graph_file.path());

  const std::vector<uint32_t> selected_state =
    make_unselected_parent_collision_state();

  const std::vector<int> level =
    make_unselected_parent_collision_level();

  constexpr std::array<uint32_t, 2u>
    thread_counts{
      64u,
      128u
    };

  constexpr std::array<uint32_t, 4u>
    bitmap_sizes{
      0u,
      32u,
      64u,
      2048u
    };

  for (const uint32_t threads_per_candidate :
       thread_counts) {
    for (const uint32_t bitmap_bits :
         bitmap_sizes) {
      // Effective selected valid sets:
      //
      //   parents = {2}
      //   children = {33}
      //
      // They do not overlap and do not collide in any supported
      // bitmap size tested here.
      //
      // Parent ID 1 is unselected and must not affect the
      // result even though:
      //
      //   1 & 31  == 1
      //   33 & 31 == 1
      run_block_batch_and_direct(
        g_ext,
        selected_state,
        level,
        threads_per_candidate,
        bitmap_bits,
        true);
    }
  }
}


TEST_CASE(
  "Parents with invalid levels do not populate the Block bitmap")
{
  TemporaryJsonFile graph_file(
    make_invalid_level_parent_collision_graph());

  G_EXT g_ext;

  g_ext.read(
    graph_file.path());

  const std::vector<uint32_t> selected_state =
    make_invalid_level_parent_collision_state();

  const std::vector<int> level =
    make_invalid_level_parent_collision_level();

  constexpr std::array<uint32_t, 2u>
    thread_counts{
      64u,
      128u
    };

  for (const uint32_t threads_per_candidate :
       thread_counts) {
    // Parent ID 1 is selected but has level -1.
    //
    // Exact Block insertion ignores it. Bitmap Block insertion
    // must also ignore it.
    //
    // The effective selected valid sets are:
    //
    //   parents = {2}
    //   children = {33}
    //
    // Candidate 0 must therefore be accepted even though
    // invalid parent ID 1 would collide with child ID 33 in a
    // 32-bit bitmap.
    run_block_batch_and_direct(
      g_ext,
      selected_state,
      level,
      threads_per_candidate,
      32u,
      true);
  }
}


TEST_CASE(
  "True selected parent-child overlap remains rejected")
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

  constexpr std::array<uint32_t, 2u>
    thread_counts{
      64u,
      128u
    };

  constexpr std::array<uint32_t, 4u>
    bitmap_sizes{
      0u,
      32u,
      1024u,
      2048u
    };

  for (const uint32_t threads_per_candidate :
       thread_counts) {
    for (const uint32_t bitmap_bits :
         bitmap_sizes) {
      run_block_batch_and_direct(
        g_ext,
        selected_state,
        level,
        threads_per_candidate,
        bitmap_bits,
        false);
    }
  }
}


TEST_CASE(
  "Invalid Block bitmap sizes are rejected")
{
  TemporaryJsonFile graph_file(
    make_selected_parent_collision_graph());

  G_EXT g_ext;

  g_ext.read(
    graph_file.path());

  const std::vector<uint32_t> selected_state =
    make_selected_parent_collision_state();

  const std::vector<int> level =
    make_selected_parent_collision_level();

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