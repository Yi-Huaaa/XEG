#include <xeg/XEG.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kChainLength = 64u;
constexpr uint32_t kShortPrefixLength = 7u;

[[noreturn]] void fail(
  const std::string& message)
{
  throw std::runtime_error(message);
}

void expect(
  const bool condition,
  const std::string& message)
{
  if (!condition) {
    fail(message);
  }
}

template <typename T>
void expect_vector_equal(
  const std::vector<T>& actual,
  const std::vector<T>& expected,
  const std::string& field_name)
{
  if (actual.size() != expected.size()) {
    std::ostringstream oss;

    oss
      << field_name
      << " size mismatch: actual="
      << actual.size()
      << ", expected="
      << expected.size();

    fail(oss.str());
  }

  for (size_t index = 0u;
       index < actual.size();
       ++index) {
    if (actual[index] != expected[index]) {
      std::ostringstream oss;

      oss
        << field_name
        << " mismatch at index "
        << index
        << ": actual="
        << actual[index]
        << ", expected="
        << expected[index];

      fail(oss.str());
    }
  }
}

std::string make_node_name(
  const uint32_t index)
{
  std::ostringstream oss;

  oss
    << "n"
    << std::setw(3)
    << std::setfill('0')
    << index;

  return oss.str();
}

class TemporaryChainJson
{
public:
  explicit TemporaryChainJson(
    const uint32_t num_nodes)
  {
    const auto timestamp =
      std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count();

    _path =
      "/tmp/xeg_gpu_dag_hybrid_fixed_mask_" +
      std::to_string(timestamp) +
      ".json";

    nlohmann::json graph;

    graph["nodes"] =
      nlohmann::json::object();

    for (uint32_t node_index = 0u;
         node_index < num_nodes;
         ++node_index) {
      nlohmann::json node;

      node["op"] = "chain";
      node["cost"] = 1.0;
      node["eclass"] = node_index;

      if (node_index + 1u < num_nodes) {
        node["children"] =
          nlohmann::json::array(
            {node_index + 1u});
      }
      else {
        node["children"] =
          nlohmann::json::array();
      }

      graph["nodes"][
        make_node_name(node_index)] =
          std::move(node);
    }

    graph["root_eclasses"] =
      nlohmann::json::array({0u});

    std::ofstream output(_path);

    if (!output.is_open()) {
      fail(
        "Failed to create temporary chain JSON: " +
        _path);
    }

    output << graph.dump(2);

    if (!output.good()) {
      fail(
        "Failed to write temporary chain JSON: " +
        _path);
    }
  }

  TemporaryChainJson(
    const TemporaryChainJson&) = delete;

  TemporaryChainJson& operator=(
    const TemporaryChainJson&) = delete;

  ~TemporaryChainJson()
  {
    if (!_path.empty()) {
      std::remove(_path.c_str());
    }
  }

  const std::string& path() const
  {
    return _path;
  }

private:
  std::string _path;
};

void expect_same_levelization(
  const XEG::GpuLevelizationResult& host_result,
  const XEG::GpuLevelizationResult& hybrid_result,
  const std::string& test_name)
{
  expect_vector_equal(
    hybrid_result.selected_state,
    host_result.selected_state,
    test_name + ": selected_state");

  expect_vector_equal(
    hybrid_result.residual_degree,
    host_result.residual_degree,
    test_name + ": residual_degree");

  expect_vector_equal(
    hybrid_result.level,
    host_result.level,
    test_name + ": level");

  if (hybrid_result.num_levels !=
      host_result.num_levels) {
    std::ostringstream oss;

    oss
      << test_name
      << ": num_levels mismatch: hybrid="
      << hybrid_result.num_levels
      << ", host="
      << host_result.num_levels;

    fail(oss.str());
  }
}

void expect_chain_result(
  const XEG::GpuLevelizationResult& result,
  const uint32_t total_nodes,
  const uint32_t selected_prefix_length,
  const std::string& test_name)
{
  expect(
    result.selected_state.size() ==
      static_cast<size_t>(total_nodes),
    test_name + ": unexpected selected_state size");

  expect(
    result.residual_degree.size() ==
      static_cast<size_t>(total_nodes),
    test_name + ": unexpected residual_degree size");

  expect(
    result.level.size() ==
      static_cast<size_t>(total_nodes),
    test_name + ": unexpected level size");

  expect(
    result.num_levels ==
      static_cast<int>(selected_prefix_length),
    test_name + ": unexpected num_levels");

  for (uint32_t node = 0u;
       node < total_nodes;
       ++node) {
    const bool is_selected =
      node < selected_prefix_length;

    const uint32_t expected_state =
      is_selected ? 1u : 0u;

    const int expected_level =
      is_selected
        ? static_cast<int>(node)
        : -1;

    if (result.selected_state[node] !=
        expected_state) {
      std::ostringstream oss;

      oss
        << test_name
        << ": selected_state["
        << node
        << "]="
        << result.selected_state[node]
        << ", expected="
        << expected_state;

      fail(oss.str());
    }

    if (result.residual_degree[node] != 0u) {
      std::ostringstream oss;

      oss
        << test_name
        << ": residual_degree["
        << node
        << "]="
        << result.residual_degree[node]
        << ", expected=0";

      fail(oss.str());
    }

    if (result.level[node] !=
        expected_level) {
      std::ostringstream oss;

      oss
        << test_name
        << ": level["
        << node
        << "]="
        << result.level[node]
        << ", expected="
        << expected_level;

      fail(oss.str());
    }
  }
}

void run_long_chain_test(
  XEG& graph,
  const bool is_top_down)
{
  const std::string direction =
    is_top_down ? "TD" : "BU";

  const std::string test_name =
    "long-chain-" + direction;

  const std::vector<uint32_t>
    selected_state(
      kChainLength,
      1u);

  const auto host_result =
    graph.unittest_gpu_levelize_with_backend(
      selected_state,
      is_top_down,
      XEG::GpuLevelizationBackend::
        HostDriven);

  const auto hybrid_result =
    graph.unittest_gpu_levelize_with_backend(
      selected_state,
      is_top_down,
      XEG::GpuLevelizationBackend::
        Hybrid);

  expect_same_levelization(
    host_result,
    hybrid_result,
    test_name);

  expect_chain_result(
    host_result,
    kChainLength,
    kChainLength,
    test_name + "-host");

  expect_chain_result(
    hybrid_result,
    kChainLength,
    kChainLength,
    test_name + "-hybrid");

  expect(
    !host_result.persistent_path_used,
    test_name +
      ": host-driven backend unexpectedly used "
      "persistent path");

  expect(
    hybrid_result.persistent_path_used,
    test_name +
      ": hybrid backend did not enter persistent path");

  expect(
    hybrid_result.persistent_completed,
    test_name +
      ": persistent path did not complete the remaining "
      "frontiers");

  expect(
    !hybrid_result.persistent_exited_to_host,
    test_name +
      ": persistent path unexpectedly returned a wide "
      "frontier to the host");

  std::cout
    << "[PASS] "
    << test_name
    << ": exact host/hybrid match; "
    << "persistent path used and completed\n";
}

void run_short_prefix_test(
  XEG& graph,
  const bool is_top_down)
{
  const std::string direction =
    is_top_down ? "TD" : "BU";

  const std::string test_name =
    "short-prefix-" + direction;

  std::vector<uint32_t>
    selected_state(
      kChainLength,
      0u);

  std::fill(
    selected_state.begin(),
    selected_state.begin() +
      kShortPrefixLength,
    1u);

  const auto host_result =
    graph.unittest_gpu_levelize_with_backend(
      selected_state,
      is_top_down,
      XEG::GpuLevelizationBackend::
        HostDriven);

  const auto hybrid_result =
    graph.unittest_gpu_levelize_with_backend(
      selected_state,
      is_top_down,
      XEG::GpuLevelizationBackend::
        Hybrid);

  expect_same_levelization(
    host_result,
    hybrid_result,
    test_name);

  expect_chain_result(
    host_result,
    kChainLength,
    kShortPrefixLength,
    test_name + "-host");

  expect_chain_result(
    hybrid_result,
    kChainLength,
    kShortPrefixLength,
    test_name + "-hybrid");

  expect(
    !host_result.persistent_path_used,
    test_name +
      ": host-driven backend unexpectedly used "
      "persistent path");

  expect(
    !hybrid_result.persistent_path_used,
    test_name +
      ": hybrid backend entered persistent mode even "
      "though the graph has fewer than 8 active levels");

  expect(
    !hybrid_result.persistent_completed,
    test_name +
      ": persistent_completed should be false when "
      "persistent mode was never entered");

  expect(
    !hybrid_result.persistent_exited_to_host,
    test_name +
      ": persistent_exited_to_host should be false "
      "when persistent mode was never entered");

  std::cout
    << "[PASS] "
    << test_name
    << ": exact host/hybrid match; "
    << "persistent path correctly not used\n";
}

}  // namespace

int main()
{
  try {
    TemporaryChainJson chain_json(
      kChainLength);

    XEG graph;
    graph.read(chain_json.path());

    const auto csr =
      graph.unittest_get_csr_snapshot();

    expect(
      csr.forward_ptr.size() ==
        static_cast<size_t>(kChainLength) + 1u,
      "The generated chain was pruned or parsed "
      "with an unexpected number of e-nodes");

    // All 64 selected nodes produce 64 one-node frontiers.
    //
    // With:
    //   enter threshold = 512
    //   required small levels = 8
    //
    // The hybrid backend must enter persistent mode after
    // observing eight consecutive small frontiers.
    run_long_chain_test(
      graph,
      true);

    run_long_chain_test(
      graph,
      false);
    
    // Selecting only the first seven nodes produces seven
    // active levels, so the eight-level entry streak cannot
    // be reached.
    run_short_prefix_test(
      graph,
      true);

    run_short_prefix_test(
      graph,
      false);

    std::cout
      << "All GPU DAG hybrid fixed-mask "
      << "levelization tests passed.\n";

    return 0;
  }
  catch (const std::exception& error) {
    std::cerr
      << "[FAIL] "
      << error.what()
      << "\n";

    return 1;
  }
}