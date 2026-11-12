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

enum class BlockInsertionMode : uint32_t
{
  Batch = 0u,
  Direct = 1u
};


const char* block_mode_name(
  const BlockInsertionMode mode)
{
  switch (mode) {
    case BlockInsertionMode::Batch:
      return "batch";

    case BlockInsertionMode::Direct:
      return "direct";
  }

  return "invalid";
}


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

  for (size_t i = 0u;
       i < actual.size();
       ++i) {
    DOCTEST_CAPTURE(i);

    DOCTEST_CHECK(
      actual[i] == expected[i]);
  }
}


XEG::GpuInsertionResult run_block_insertion(
  XEG& xeg,
  const BlockInsertionMode mode,
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& level,
  const bool is_top_down_insertion,
  const uint32_t threads_per_candidate)
{
  switch (mode) {
    case BlockInsertionMode::Batch:
      return
        xeg.unittest_gpu_insert_block_batch_once(
          selected_state,
          level,
          is_top_down_insertion,
          threads_per_candidate);

    case BlockInsertionMode::Direct:
      return
        xeg.unittest_gpu_insert_block_direct_once(
          selected_state,
          level,
          is_top_down_insertion,
          threads_per_candidate);
  }

  throw std::logic_error(
    "Invalid BlockInsertionMode");
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
  TemporaryFanoutGraph(
    const BlockInsertionMode mode,
    const uint32_t total_nodes)
    : path_(
        "/tmp/xeg_block_"
        + std::string(block_mode_name(mode))
        + "_boundary_"
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
        "Failed to create temporary Block test graph");
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
        "Block test graph");
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
  "GPU Block insertion handles a selected parent and child")
{
  for (const BlockInsertionMode mode : {
         BlockInsertionMode::Batch,
         BlockInsertionMode::Direct}) {
    for (const uint32_t threads_per_candidate : {
           64u, 128u, 256u}) {
      DOCTEST_CAPTURE(
        block_mode_name(mode));

      DOCTEST_CAPTURE(
        threads_per_candidate);

      {
        XEG xeg;

        xeg.read(
          test_data_path("single_edge.json"));

        const auto result =
          run_block_insertion(
            xeg,
            mode,
            std::vector<uint32_t>{1u, 0u},
            std::vector<int>{0, -1},
            true,
            threads_per_candidate);

        check_vector_equal(
          result.selected_state,
          std::vector<uint32_t>{1u, 2u});

        check_vector_equal(
          result.level,
          std::vector<int>{0, 1});

        DOCTEST_CHECK(
          result.u0 == 1u);
      }

      {
        XEG xeg;

        xeg.read(
          test_data_path("single_edge.json"));

        const auto result =
          run_block_insertion(
            xeg,
            mode,
            std::vector<uint32_t>{0u, 1u},
            std::vector<int>{-1, 1},
            false,
            threads_per_candidate);

        check_vector_equal(
          result.selected_state,
          std::vector<uint32_t>{2u, 1u});

        check_vector_equal(
          result.level,
          std::vector<int>{0, 1});

        DOCTEST_CHECK(
          result.u0 == 1u);
      }
    }
  }
}


TEST_CASE(
  "GPU Block insertion rejects candidates without a selected neighbor")
{
  for (const BlockInsertionMode mode : {
         BlockInsertionMode::Batch,
         BlockInsertionMode::Direct}) {
    for (const bool is_top_down : {
           true, false}) {
      DOCTEST_CAPTURE(
        block_mode_name(mode));

      DOCTEST_CAPTURE(
        is_top_down);

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
        run_block_insertion(
          xeg,
          mode,
          initial_state,
          initial_level,
          is_top_down,
          64u);

      check_vector_equal(
        result.selected_state,
        initial_state);

      check_vector_equal(
        result.level,
        initial_level);

      DOCTEST_CHECK(
        result.u0 == 0u);
    }
  }
}


TEST_CASE(
  "GPU Block insertion rejects a parent-child level conflict")
{
  for (const BlockInsertionMode mode : {
         BlockInsertionMode::Batch,
         BlockInsertionMode::Direct}) {
    for (const bool is_top_down : {
           true, false}) {
      DOCTEST_CAPTURE(
        block_mode_name(mode));

      DOCTEST_CAPTURE(
        is_top_down);

      XEG xeg;

      xeg.read(
        test_data_path("chain.json"));

      const std::vector<uint32_t> initial_state = {
        1u, 0u, 1u
      };

      const std::vector<int> initial_level = {
        2, -1, 1
      };

      const auto result =
        run_block_insertion(
          xeg,
          mode,
          initial_state,
          initial_level,
          is_top_down,
          64u);

      check_vector_equal(
        result.selected_state,
        initial_state);

      check_vector_equal(
        result.level,
        initial_level);

      DOCTEST_CHECK(
        result.u0 == 0u);
    }
  }
}


TEST_CASE(
  "GPU Block insertion rejects selected parent-child overlap")
{
  for (const BlockInsertionMode mode : {
         BlockInsertionMode::Batch,
         BlockInsertionMode::Direct}) {
    for (const bool is_top_down : {
           true, false}) {
      DOCTEST_CAPTURE(
        block_mode_name(mode));

      DOCTEST_CAPTURE(
        is_top_down);

      XEG xeg;

      xeg.read(
        test_data_path("overlap_cycle.json"));

      const std::vector<uint32_t> initial_state = {
        1u, 0u
      };

      const std::vector<int> initial_level = {
        0, -1
      };

      const auto result =
        run_block_insertion(
          xeg,
          mode,
          initial_state,
          initial_level,
          is_top_down,
          64u);

      check_vector_equal(
        result.selected_state,
        initial_state);

      check_vector_equal(
        result.level,
        initial_level);

      DOCTEST_CHECK(
        result.u0 == 0u);
    }
  }
}


TEST_CASE(
  "GPU Block insertion treats state 2 as selected")
{
  for (const BlockInsertionMode mode : {
         BlockInsertionMode::Batch,
         BlockInsertionMode::Direct}) {
    DOCTEST_CAPTURE(
      block_mode_name(mode));

    XEG xeg;

    xeg.read(
      test_data_path("single_edge.json"));

    const auto result =
      run_block_insertion(
        xeg,
        mode,
        std::vector<uint32_t>{2u, 0u},
        std::vector<int>{0, -1},
        true,
        64u);

    check_vector_equal(
      result.selected_state,
      std::vector<uint32_t>{2u, 2u});

    check_vector_equal(
      result.level,
      std::vector<int>{0, 1});

    DOCTEST_CHECK(
      result.u0 == 1u);
  }
}


TEST_CASE(
  "GPU Block insertion ignores selected neighbors with invalid levels")
{
  for (const BlockInsertionMode mode : {
         BlockInsertionMode::Batch,
         BlockInsertionMode::Direct}) {
    DOCTEST_CAPTURE(
      block_mode_name(mode));

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
      run_block_insertion(
        xeg,
        mode,
        initial_state,
        initial_level,
        true,
        64u);

    check_vector_equal(
      result.selected_state,
      initial_state);

    check_vector_equal(
      result.level,
      initial_level);

    DOCTEST_CHECK(
      result.u0 == 0u);
  }
}


TEST_CASE(
  "GPU Block insertion handles independent candidates")
{
  for (const BlockInsertionMode mode : {
         BlockInsertionMode::Batch,
         BlockInsertionMode::Direct}) {
    DOCTEST_CAPTURE(
      block_mode_name(mode));

    XEG xeg;

    xeg.read(
      test_data_path(
        "parallel_candidates.json"));

    const auto result =
      run_block_insertion(
        xeg,
        mode,
        std::vector<uint32_t>{
          1u, 1u, 1u, 0u, 0u
        },
        std::vector<int>{
          0, 1, 2, -1, -1
        },
        false,
        64u);

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

    DOCTEST_CHECK(
      result.u0 == 2u);
  }
}


TEST_CASE(
  "GPU Block insertion covers Batch and Direct boundaries")
{
  for (const BlockInsertionMode mode : {
         BlockInsertionMode::Batch,
         BlockInsertionMode::Direct}) {
    const std::vector<uint32_t> total_node_counts =
      mode == BlockInsertionMode::Batch
        ? std::vector<uint32_t>{
            255u, 256u, 257u, 513u
          }
        : std::vector<uint32_t>{
            511u, 512u, 513u, 1025u
          };

    for (const uint32_t total_nodes :
         total_node_counts) {
      DOCTEST_CAPTURE(
        block_mode_name(mode));

      DOCTEST_CAPTURE(
        total_nodes);

      TemporaryFanoutGraph graph(
        mode,
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
        run_block_insertion(
          xeg,
          mode,
          initial_state,
          initial_level,
          true,
          64u);

      DOCTEST_REQUIRE(
        result.selected_state.size()
          == total_nodes);

      DOCTEST_REQUIRE(
        result.level.size()
          == total_nodes);

      DOCTEST_CHECK(
        result.selected_state[0] == 1u);

      DOCTEST_CHECK(
        result.level[0] == 0);

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
        result.u0 ==
          total_nodes - 1u);
    }
  }
}


TEST_CASE(
  "GPU Block insertion rejects invalid thread counts")
{
  for (const BlockInsertionMode mode : {
         BlockInsertionMode::Batch,
         BlockInsertionMode::Direct}) {
    for (const uint32_t invalid_threads : {
           32u, 96u, 2048u}) {
      DOCTEST_CAPTURE(
        block_mode_name(mode));

      DOCTEST_CAPTURE(
        invalid_threads);

      XEG xeg;

      xeg.read(
        test_data_path("single_edge.json"));

      DOCTEST_CHECK_THROWS_AS(
        run_block_insertion(
          xeg,
          mode,
          std::vector<uint32_t>{1u, 0u},
          std::vector<int>{0, -1},
          true,
          invalid_threads),
        std::invalid_argument);
    }
  }
}

TEST_CASE("Block batches alternate inactive and active candidates across warps") {
  // All blocks skip an already-selected candidate before processing a live
  // candidate in the next transposed batch: shared flags must not be reused early.
  TemporaryFanoutGraph graph(BlockInsertionMode::Batch, 513u);
  XEG xeg; xeg.read(graph.path());
  for (const uint32_t bits : {0u, 512u}) {
    xeg::GpuDagOptions options;
    const uint32_t batch = bits ? options.bitmap_batch_size : options.exact_batch_size;
    std::vector<uint32_t> state(513u, 0u);
    std::vector<int> levels(513u, -1);
    state[0] = 1u; levels[0] = 0;
    uint32_t expected = 0;
    for (uint32_t u = 1; u < state.size(); ++u) {
      if ((u / batch) % 2 == 0) { state[u] = 1u; levels[u] = 1; }
      else ++expected;
    }
    for (const uint32_t threads : {64u, 128u, 256u}) {
      const auto result = xeg.unittest_gpu_insert_block_batch_once(
        state, levels, true, threads, bits);
      DOCTEST_CHECK(result.u0 == expected);
      for (uint32_t u = 1; u < state.size(); ++u) {
        DOCTEST_CHECK(result.selected_state[u] == (state[u] ? 1u : 2u));
        DOCTEST_CHECK(result.level[u] == 1);
      }
    }
  }
}
