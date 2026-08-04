/* trunk-ignore-all(clang-format) */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

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


void check_all_zero(
  const std::vector<uint32_t>& values)
{
  for (size_t i = 0u;
       i < values.size();
       ++i) {
    DOCTEST_CAPTURE(i);

    DOCTEST_CHECK(
      values[i] == 0u);
  }
}


void check_no_tentative_state(
  const std::vector<uint32_t>& selected_state)
{
  for (size_t i = 0u;
       i < selected_state.size();
       ++i) {
    DOCTEST_CAPTURE(i);

    DOCTEST_CHECK(
      selected_state[i] != 2u);
  }
}


void check_selected_edges_respect_levels(
  const G_EXT::CsrGraphSnapshot& csr,
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& level)
{
  DOCTEST_REQUIRE(
    selected_state.size() == level.size());

  DOCTEST_REQUIRE(
    csr.forward_ptr.size()
      == selected_state.size() + 1u);

  for (size_t u = 0u;
       u < selected_state.size();
       ++u) {
    if (selected_state[u] == 0u) {
      continue;
    }

    DOCTEST_CAPTURE(u);

    DOCTEST_REQUIRE(
      level[u] >= 0);

    const uint32_t begin =
      csr.forward_ptr[u];

    const uint32_t end =
      csr.forward_ptr[u + 1u];

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

      DOCTEST_REQUIRE(
        level[v] >= 0);

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
    in_subgraph.size()
      == selected_state.size());

  for (size_t u = 0u;
       u < selected_state.size();
       ++u) {
    DOCTEST_CAPTURE(u);

    DOCTEST_CHECK(
      in_subgraph[u] <= 1u);

    if (in_subgraph[u] != 0u) {
      DOCTEST_CHECK(
        selected_state[u] != 0u);
    }
  }
}


// Capture printf()/stdout output produced by the production
// GPU DAG controller.
//
// This helper is compiled only into this unit-test executable.
// It does not change or instrument the g_ext library.
template <typename Function>
std::string capture_stdout(
  Function function)
{
  std::fflush(stdout);

  FILE* temporary_file =
    std::tmpfile();

  if (temporary_file == nullptr) {
    throw std::runtime_error(
      "Failed to create temporary stdout file");
  }

  const int stdout_fd =
    ::fileno(stdout);

  const int temporary_fd =
    ::fileno(temporary_file);

  const int saved_stdout_fd =
    ::dup(stdout_fd);

  if (saved_stdout_fd == -1) {
    std::fclose(temporary_file);

    throw std::runtime_error(
      "Failed to duplicate stdout");
  }

  if (::dup2(
        temporary_fd,
        stdout_fd) == -1) {
    ::close(saved_stdout_fd);
    std::fclose(temporary_file);

    throw std::runtime_error(
      "Failed to redirect stdout");
  }

  try {
    function();
    std::fflush(stdout);
  }
  catch (...) {
    std::fflush(stdout);

    (void)::dup2(
      saved_stdout_fd,
      stdout_fd);

    ::close(saved_stdout_fd);
    std::fclose(temporary_file);

    throw;
  }

  if (::dup2(
        saved_stdout_fd,
        stdout_fd) == -1) {
    ::close(saved_stdout_fd);
    std::fclose(temporary_file);

    throw std::runtime_error(
      "Failed to restore stdout");
  }

  ::close(saved_stdout_fd);

  std::rewind(temporary_file);

  std::string output;

  char buffer[4096];

  while (true) {
    const size_t count =
      std::fread(
        buffer,
        1u,
        sizeof(buffer),
        temporary_file);

    output.append(
      buffer,
      count);

    if (count < sizeof(buffer)) {
      if (std::ferror(temporary_file) != 0) {
        std::fclose(temporary_file);

        throw std::runtime_error(
          "Failed to read captured stdout");
      }

      break;
    }
  }

  std::fclose(temporary_file);

  return output;
}


struct ControllerSummary
{
  std::string policy;
  uint32_t iterations = 0u;
  uint32_t batch_iterations = 0u;
  uint32_t direct_iterations = 0u;
  std::string reason;
  uint32_t peak = 0u;
  uint32_t last = 0u;
  uint32_t low_activity_threshold = 0u;
  uint32_t low_activity_streak = 0u;
  std::string switched;
  uint32_t switch_iteration = 0u;
  uint32_t switch_survived = 0u;
};


uint32_t parse_uint32(
  const std::ssub_match& match)
{
  return static_cast<uint32_t>(
    std::stoul(match.str()));
}


ControllerSummary parse_controller_summary(
  const std::string& output)
{
  const std::regex summary_pattern(
    R"(\[GPU DAG Outer Loop\] policy=([^,]+), iterations=([0-9]+), batch_iterations=([0-9]+), direct_iterations=([0-9]+), reason=([^,]+), peak=([0-9]+), last=([0-9]+), low_activity_threshold=([0-9]+), low_activity_streak=([0-9]+), switched=([^,]+), switch_iteration=([0-9]+), switch_survived=([0-9]+))");

  std::smatch match;

  if (!std::regex_search(
        output,
        match,
        summary_pattern)) {
    throw std::runtime_error(
      "GPU DAG controller summary was not found");
  }

  ControllerSummary summary;

  summary.policy =
    match[1].str();

  summary.iterations =
    parse_uint32(match[2]);

  summary.batch_iterations =
    parse_uint32(match[3]);

  summary.direct_iterations =
    parse_uint32(match[4]);

  summary.reason =
    match[5].str();

  summary.peak =
    parse_uint32(match[6]);

  summary.last =
    parse_uint32(match[7]);

  summary.low_activity_threshold =
    parse_uint32(match[8]);

  summary.low_activity_streak =
    parse_uint32(match[9]);

  summary.switched =
    match[10].str();

  summary.switch_iteration =
    parse_uint32(match[11]);

  summary.switch_survived =
    parse_uint32(match[12]);

  return summary;
}

}  // namespace


TEST_CASE(
  "GPU batch-then-direct controller does not switch after immediate convergence")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path("chain.json"));

  G_EXT::GpuDagConstructionResult result;

  const std::string output =
    capture_stdout(
      [&]() {
        result =
          g_ext
            .unittest_gpu_construct_dag_from_tree_with_policy(
              G_EXT::GpuDagInsertionPolicy::
                BatchThenDirect);
      });

  DOCTEST_INFO(output);

  const ControllerSummary summary =
    parse_controller_summary(output);

  DOCTEST_CHECK(
    summary.policy == "batch-then-direct");

  DOCTEST_CHECK(
    summary.iterations == 1u);

  DOCTEST_CHECK(
    summary.batch_iterations == 1u);

  DOCTEST_CHECK(
    summary.direct_iterations == 0u);

  DOCTEST_CHECK(
    summary.iterations ==
      summary.batch_iterations
      + summary.direct_iterations);

  DOCTEST_CHECK(
    summary.reason == "converged");

  DOCTEST_CHECK(
    summary.last == 0u);

  DOCTEST_CHECK(
    summary.switched == "no");

  DOCTEST_CHECK(
    summary.switch_iteration == 0u);

  DOCTEST_CHECK(
    summary.switch_survived == 0u);

  check_vector_equal(
    result.selected_state,
    std::vector<uint32_t>{
      1u, 1u, 1u
    });

  check_vector_equal(
    result.level,
    std::vector<int>{
      0, 1, 2
    });

  check_vector_equal(
    result.in_subgraph,
    std::vector<uint32_t>{
      1u, 1u, 1u
    });

  check_all_zero(
    result.residual_degree);

  check_no_tentative_state(
    result.selected_state);

  DOCTEST_CHECK(
    result.num_levels == 3);

  DOCTEST_CHECK(
    result.final_u0 == 0u);
}


TEST_CASE(
  "GPU batch-then-direct controller reaches full convergence")
{
  G_EXT g_ext;

  g_ext.read(
    test_data_path(
      "batch_then_direct_switch.json"));

  G_EXT::GpuDagConstructionResult result;

  const std::string output =
    capture_stdout(
      [&]() {
        result =
          g_ext
            .unittest_gpu_construct_dag_from_tree_with_policy(
              G_EXT::GpuDagInsertionPolicy::
                BatchThenDirect);
      });

  DOCTEST_INFO(output);

  const ControllerSummary summary =
    parse_controller_summary(output);

  // Verify that the real production controller used the
  // intended policy.
  DOCTEST_CHECK(
    summary.policy == "batch-then-direct");

  // Batch mode must always execute first.
  DOCTEST_CHECK(
    summary.batch_iterations > 0u);

  DOCTEST_CHECK(
    summary.iterations ==
      summary.batch_iterations
      + summary.direct_iterations);

  DOCTEST_CHECK(
    summary.peak > 0u);

  // GPU execution is nondeterministic. The batch phase may
  // either trigger the switch to direct mode or converge before
  // the switching condition is reached.
  const bool has_valid_switched_value =
    summary.switched == "yes" ||
    summary.switched == "no";

  DOCTEST_REQUIRE_EQ(
    has_valid_switched_value,
    true);

  if (summary.switched == "yes") {
    // Once the controller switches, every later iteration uses
    // direct mode. Therefore the switch iteration equals the
    // total number of completed batch iterations.
    DOCTEST_CHECK(
      summary.direct_iterations > 0u);

    DOCTEST_CHECK(
      summary.switch_iteration ==
        summary.batch_iterations);

    DOCTEST_CHECK(
      summary.switch_survived > 0u);

    DOCTEST_CHECK(
      summary.low_activity_streak >= 2u);
  }
  else {
    // The batch phase reached full convergence before the
    // low-activity switching condition was triggered.
    DOCTEST_CHECK(
      summary.direct_iterations == 0u);

    DOCTEST_CHECK(
      summary.switch_iteration == 0u);

    DOCTEST_CHECK(
      summary.switch_survived == 0u);
  }

  // BatchThenDirect must continue until the original u0 == 0
  // convergence condition, regardless of whether a switch was
  // needed.
  DOCTEST_CHECK(
    summary.reason == "converged");

  DOCTEST_CHECK(
    summary.last == 0u);

  DOCTEST_CHECK(
    result.final_u0 == 0u);

  DOCTEST_REQUIRE(
    !result.selected_state.empty());

  check_no_tentative_state(
    result.selected_state);

  // The controller may finish immediately after tentative
  // nodes are rolled back. In that case, result.level and
  // result.residual_degree still describe the final
  // pre-rollback levelization pass rather than the final
  // stable selected graph.
  //
  // Validate the final selected mask independently using a
  // fresh object and the original host-driven levelizer.
  // This is test-only validation and does not alter the
  // production controller.
  G_EXT final_graph_validator;

  final_graph_validator.read(
    test_data_path(
      "batch_then_direct_switch.json"));

  const auto final_levelization =
    final_graph_validator
      .unittest_gpu_levelize_with_backend(
        result.selected_state,
        true,
        G_EXT::GpuLevelizationBackend::
          HostDriven);

  // Levelization must not change the final selected mask.
  check_vector_equal(
    final_levelization.selected_state,
    result.selected_state);

  // After re-levelizing the final stable graph, every selected
  // node must be processed by Kahn levelization. Therefore all
  // residual degrees must be zero.
  check_all_zero(
    final_levelization.residual_degree);

  DOCTEST_CHECK(
    final_levelization.num_levels > 0);

  const auto csr =
    final_graph_validator
      .unittest_get_csr_snapshot();

  check_selected_edges_respect_levels(
    csr,
    result.selected_state,
    final_levelization.level);

  check_in_subgraph_is_selected_subset(
    result.selected_state,
    result.in_subgraph);

  uint32_t filtered_node_count = 0u;

  for (const uint32_t in_subgraph :
       result.in_subgraph) {
    filtered_node_count +=
      in_subgraph != 0u ? 1u : 0u;
  }

  DOCTEST_CHECK(
    filtered_node_count > 0u);
}