#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <g_ext/g_ext.cuh>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

constexpr uint32_t kNodeCount = 80u;

struct Edge
{
  uint32_t parent;
  uint32_t child;
};

std::string node_name(
  const uint32_t node)
{
  std::ostringstream stream;

  stream
    << "n"
    << std::setfill('0')
    << std::setw(3)
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
    << std::setw(3)
    << node;

  return stream.str();
}

std::vector<Edge> make_test_edges()
{
  return {
    // Candidate 0:
    //
    //   selected parent = 4
    //   selected child  = 36
    //
    // 4 != 36
    // 4 & 31 == 36 & 31
    // 4 & 63 != 36 & 63
    //
    // Therefore:
    //   Exact -> accept
    //   32-bit -> reject
    //   64-bit -> accept
    {4u, 0u},
    {0u, 36u},

    // Candidate 1:
    //
    // No collision for either bitmap size.
    {5u, 1u},
    {1u, 6u},

    // Candidate 2:
    //
    //   selected parent = 7
    //   selected child  = 71
    //
    // 7 != 71
    // 7 & 31 == 71 & 31
    // 7 & 63 == 71 & 63
    //
    // Therefore:
    //   Exact -> accept
    //   32-bit -> reject
    //   64-bit -> reject
    {7u, 2u},
    {2u, 71u},

    // Candidate 3:
    //
    // Selected valid parent:
    //   8
    //
    // Unselected parent:
    //   68
    //
    // Selected child:
    //   4
    //
    // 68 collides with 4 under both 32-bit and 64-bit
    // hashing, but parent 68 is unselected and must not enter
    // either bitmap.
    {8u, 3u},
    {68u, 3u},
    {3u, 4u},

    // Candidate 9:
    //
    // Selected valid parent:
    //   10
    //
    // Selected parent with invalid level:
    //   69
    //
    // Selected child:
    //   5
    //
    // 69 collides with 5 under both 32-bit and 64-bit
    // hashing, but parent 69 has level -1 and must not enter
    // either bitmap.
    {10u, 9u},
    {69u, 9u},
    {9u, 5u},

    // Candidate 11:
    //
    // Node 12 appears in both the parent and child segments.
    {12u, 11u},
    {11u, 12u},

    // Candidate 13:
    //
    // Ordinary non-collision candidate used for Batch/Direct
    // parity testing.
    {14u, 13u},
    {13u, 15u}
  };
}

class TemporaryGraphFile
{
public:
  TemporaryGraphFile(
    const uint32_t num_nodes,
    const std::vector<Edge>& edges)
  {
    for (const Edge& edge : edges) {
      if (edge.parent >= num_nodes ||
          edge.child >= num_nodes) {
        throw std::invalid_argument(
          "Temporary graph edge is outside node range");
      }
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
        "g_ext_warp_bitmap_" +
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

      for (const Edge& edge : edges) {
        if (edge.parent == node) {
          // g_ext.cu interprets string children as child
          // e-node names and then looks up their eclasses.
          children.push_back(
            node_name(edge.child));
        }
      }

      nodes[node_name(node)] = {
        {"op", "test"},
        {"cost", 1.0},
        {"eclass", eclass_name(node)},
        {"children", std::move(children)}
      };
    }

    // Make every unique eclass alive so pruning preserves the
    // zero-padded node ordering:
    //
    //   n000 -> e-node ID 0
    //   n001 -> e-node ID 1
    //   ...
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

std::vector<uint32_t>
make_selected_state(
  const std::initializer_list<uint32_t>
    unselected_nodes)
{
  std::vector<uint32_t> selected_state(
    kNodeCount,
    1u);

  for (const uint32_t node :
       unselected_nodes) {
    if (node >= kNodeCount) {
      throw std::invalid_argument(
        "Unselected test node is outside node range");
    }

    selected_state[node] =
      0u;
  }

  return selected_state;
}

std::vector<int>
make_level_state(
  const std::initializer_list<uint32_t>
    unselected_nodes)
{
  std::vector<int> level(
    kNodeCount,
    0);

  for (const uint32_t node :
       unselected_nodes) {
    if (node >= kNodeCount) {
      throw std::invalid_argument(
        "Unselected test node is outside node range");
    }

    level[node] =
      -1;
  }

  return level;
}

void check_result_equal(
  const G_EXT::GpuInsertionResult& lhs,
  const G_EXT::GpuInsertionResult& rhs)
{
  CHECK(lhs.selected_state ==
        rhs.selected_state);

  CHECK(lhs.level ==
        rhs.level);

  CHECK(lhs.u0 ==
        rhs.u0);
}

}  // namespace

TEST_CASE(
  "Warp bitmap accepts only 0, 32, and 64 bits")
{
  const TemporaryGraphFile graph(
    kNodeCount,
    make_test_edges());

  G_EXT extension;
  extension.read(graph.path());

  const auto selected_state =
    make_selected_state({});

  const auto level =
    make_level_state({});

  CHECK_NOTHROW(
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        0u));

  CHECK_NOTHROW(
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        32u));

  CHECK_NOTHROW(
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        64u));

  CHECK_THROWS_AS(
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        16u),
    std::invalid_argument);

  CHECK_THROWS_AS(
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        128u),
    std::invalid_argument);
}

TEST_CASE(
  "Warp Bitmap-32 rejects a 32-bit collision that "
  "Warp Bitmap-64 distinguishes")
{
  const TemporaryGraphFile graph(
    kNodeCount,
    make_test_edges());

  G_EXT extension;
  extension.read(graph.path());

  auto selected_state =
    make_selected_state({0u});

  auto level =
    make_level_state({0u});

  level[4u] =
    1;

  level[36u] =
    3;

  const auto exact =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        0u);

  const auto bitmap32 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        32u);

  const auto bitmap64 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        64u);

  REQUIRE(exact.selected_state.size() ==
          kNodeCount);

  REQUIRE(bitmap32.selected_state.size() ==
          kNodeCount);

  REQUIRE(bitmap64.selected_state.size() ==
          kNodeCount);

  CHECK(exact.selected_state[0u] ==
        2u);

  CHECK(exact.level[0u] ==
        2);

  CHECK(exact.u0 ==
        1u);

  CHECK(bitmap32.selected_state[0u] ==
        0u);

  CHECK(bitmap32.level[0u] ==
        -1);

  CHECK(bitmap32.u0 ==
        0u);

  CHECK(bitmap64.selected_state[0u] ==
        2u);

  CHECK(bitmap64.level[0u] ==
        2);

  CHECK(bitmap64.u0 ==
        1u);
}

TEST_CASE(
  "Warp Bitmap-64 conservatively rejects a 64-bit collision")
{
  const TemporaryGraphFile graph(
    kNodeCount,
    make_test_edges());

  G_EXT extension;
  extension.read(graph.path());

  auto selected_state =
    make_selected_state({2u});

  auto level =
    make_level_state({2u});

  level[7u] =
    1;

  level[71u] =
    3;

  const auto exact =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        0u);

  const auto bitmap32 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        32u);

  const auto bitmap64 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        64u);

  CHECK(exact.selected_state[2u] ==
        2u);

  CHECK(exact.level[2u] ==
        2);

  CHECK(exact.u0 ==
        1u);

  CHECK(bitmap32.selected_state[2u] ==
        0u);

  CHECK(bitmap32.level[2u] ==
        -1);

  CHECK(bitmap32.u0 ==
        0u);

  CHECK(bitmap64.selected_state[2u] ==
        0u);

  CHECK(bitmap64.level[2u] ==
        -1);

  CHECK(bitmap64.u0 ==
        0u);
}

TEST_CASE(
  "Unselected parents do not populate Warp bitmaps")
{
  const TemporaryGraphFile graph(
    kNodeCount,
    make_test_edges());

  G_EXT extension;
  extension.read(graph.path());

  // Candidate 3 and parent 68 are both unselected.
  //
  // Parent 68 has no selected parent, so it cannot be inserted
  // by this top-down insertion phase.
  auto selected_state =
    make_selected_state({
      3u,
      68u
    });

  auto level =
    make_level_state({
      3u,
      68u
    });

  level[8u] =
    1;

  level[4u] =
    3;

  const auto bitmap32 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        32u);

  const auto bitmap64 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        64u);

  CHECK(bitmap32.selected_state[3u] ==
        2u);

  CHECK(bitmap32.level[3u] ==
        2);

  CHECK(bitmap32.selected_state[68u] ==
        0u);

  CHECK(bitmap32.u0 ==
        1u);

  CHECK(bitmap64.selected_state[3u] ==
        2u);

  CHECK(bitmap64.level[3u] ==
        2);

  CHECK(bitmap64.selected_state[68u] ==
        0u);

  CHECK(bitmap64.u0 ==
        1u);
}

TEST_CASE(
  "Selected parents with invalid levels do not populate "
  "Warp bitmaps")
{
  const TemporaryGraphFile graph(
    kNodeCount,
    make_test_edges());

  G_EXT extension;
  extension.read(graph.path());

  auto selected_state =
    make_selected_state({9u});

  auto level =
    make_level_state({9u});

  level[10u] =
    1;

  // Parent 69 is selected, but its invalid level excludes it
  // from both the level summary and the bitmap.
  level[69u] =
    -1;

  level[5u] =
    3;

  const auto bitmap32 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        32u);

  const auto bitmap64 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        64u);

  CHECK(bitmap32.selected_state[9u] ==
        2u);

  CHECK(bitmap32.level[9u] ==
        2);

  CHECK(bitmap32.u0 ==
        1u);

  CHECK(bitmap64.selected_state[9u] ==
        2u);

  CHECK(bitmap64.level[9u] ==
        2);

  CHECK(bitmap64.u0 ==
        1u);
}

TEST_CASE(
  "A true selected parent-child overlap is rejected by "
  "all Warp implementations")
{
  const TemporaryGraphFile graph(
    kNodeCount,
    make_test_edges());

  G_EXT extension;
  extension.read(graph.path());

  auto selected_state =
    make_selected_state({11u});

  auto level =
    make_level_state({11u});

  level[12u] =
    2;

  const auto exact =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        0u);

  const auto bitmap32 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        32u);

  const auto bitmap64 =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        64u);

  CHECK(exact.selected_state[11u] ==
        0u);

  CHECK(bitmap32.selected_state[11u] ==
        0u);

  CHECK(bitmap64.selected_state[11u] ==
        0u);

  CHECK(exact.u0 ==
        0u);

  CHECK(bitmap32.u0 ==
        0u);

  CHECK(bitmap64.u0 ==
        0u);
}

TEST_CASE(
  "Warp-local bitmaps do not leak between warps in one block")
{
  const TemporaryGraphFile graph(
    kNodeCount,
    make_test_edges());

  G_EXT extension;
  extension.read(graph.path());

  auto selected_state =
    make_selected_state({
      0u,
      1u
    });

  auto level =
    make_level_state({
      0u,
      1u
    });

  // Candidate 0:
  //   32-bit collision -> reject.
  level[4u] =
    1;

  level[36u] =
    3;

  // Candidate 1:
  //   no collision -> accept.
  level[5u] =
    1;

  level[6u] =
    3;

  SUBCASE(
    "Two warps in one 64-thread block")
  {
    const auto result =
      extension
        .unittest_gpu_insert_warp_direct_once(
          selected_state,
          level,
          true,
          64u,
          32u);

    CHECK(result.selected_state[0u] ==
          0u);

    CHECK(result.selected_state[1u] ==
          2u);

    CHECK(result.level[1u] ==
          2);

    CHECK(result.u0 ==
          1u);
  }

  SUBCASE(
    "Four warps in one 128-thread block")
  {
    const auto result =
      extension
        .unittest_gpu_insert_warp_direct_once(
          selected_state,
          level,
          true,
          128u,
          32u);

    CHECK(result.selected_state[0u] ==
          0u);

    CHECK(result.selected_state[1u] ==
          2u);

    CHECK(result.level[1u] ==
          2);

    CHECK(result.u0 ==
          1u);
  }
}

TEST_CASE(
  "Warp Bitmap Batch and Direct match for one active candidate")
{
  const TemporaryGraphFile graph(
    kNodeCount,
    make_test_edges());

  G_EXT extension;
  extension.read(graph.path());

  auto selected_state =
    make_selected_state({13u});

  auto level =
    make_level_state({13u});

  level[14u] =
    1;

  level[15u] =
    3;

  for (const uint32_t bitmap_bits :
       {32u, 64u}) {
    SUBCASE(
      bitmap_bits == 32u
        ? "Bitmap-32"
        : "Bitmap-64")
    {
      SUBCASE(
        "Top-down")
      {
        const auto batch =
          extension
            .unittest_gpu_insert_warp_batch_once(
              selected_state,
              level,
              true,
              64u,
              bitmap_bits);

        const auto direct =
          extension
            .unittest_gpu_insert_warp_direct_once(
              selected_state,
              level,
              true,
              64u,
              bitmap_bits);

        check_result_equal(
          batch,
          direct);

        CHECK(batch.selected_state[13u] ==
              2u);

        CHECK(batch.level[13u] ==
              2);

        CHECK(batch.u0 ==
              1u);
      }

      SUBCASE(
        "Bottom-up")
      {
        const auto batch =
          extension
            .unittest_gpu_insert_warp_batch_once(
              selected_state,
              level,
              false,
              64u,
              bitmap_bits);

        const auto direct =
          extension
            .unittest_gpu_insert_warp_direct_once(
              selected_state,
              level,
              false,
              64u,
              bitmap_bits);

        check_result_equal(
          batch,
          direct);

        CHECK(batch.selected_state[13u] ==
              2u);

        CHECK(batch.level[13u] ==
              2);

        CHECK(batch.u0 ==
              1u);
      }
    }
  }
}

TEST_CASE(
  "The original four-argument Warp API still selects "
  "the Exact Warp implementation")
{
  const TemporaryGraphFile graph(
    kNodeCount,
    make_test_edges());

  G_EXT extension;
  extension.read(graph.path());

  auto selected_state =
    make_selected_state({0u});

  auto level =
    make_level_state({0u});

  level[4u] =
    1;

  level[36u] =
    3;

  const auto implicit_exact =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u);

  const auto explicit_exact =
    extension
      .unittest_gpu_insert_warp_direct_once(
        selected_state,
        level,
        true,
        64u,
        0u);

  check_result_equal(
    implicit_exact,
    explicit_exact);

  CHECK(implicit_exact.selected_state[0u] ==
        2u);

  CHECK(implicit_exact.level[0u] ==
        2);

  CHECK(implicit_exact.u0 ==
        1u);
}