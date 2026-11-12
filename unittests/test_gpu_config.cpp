#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <xeg/gpu_config.hpp>
#include <filesystem>
#include <fstream>

TEST_CASE("Embedded GPU defaults match every measured design and the unlisted fallback") {
  const auto path = std::filesystem::path(__FILE__).parent_path().parent_path()
    / "configs/gpu/87-designs.json";
  std::ifstream input(path);
  DOCTEST_REQUIRE(input.good());
  nlohmann::json document;
  input >> document;
  DOCTEST_REQUIRE(document["designs"].size() == 87);
  for (const auto& [design, overrides] : document["designs"].items()) {
    (void)overrides;
    DOCTEST_CHECK(xeg::default_gpu_profile(design).to_json() ==
                  xeg::resolve_gpu_profile(document, design).to_json());
  }
  const auto fallback = xeg::default_gpu_profile("unlisted/example");
  DOCTEST_CHECK(fallback.to_json() == xeg::resolve_gpu_profile(document, "unlisted/example").to_json());
  DOCTEST_CHECK(fallback.kernel == 2);
  DOCTEST_CHECK(fallback.dag_threads == 64);
  DOCTEST_CHECK(fallback.bitmap_bits == 512);
}

TEST_CASE("GPU profiles inherit defaults and override only the chosen design") {
  const nlohmann::json config = {
    {"schema_version", 1},
    {"defaults", {{"kernel", 2}, {"dag_threads", 64}, {"bitmap_bits", 512}}},
    {"designs", {{"boolE/mul32_map.rename", {{"bitmap_batch_size", 8}, {"bitmap_max_direct_iterations", 10}}}}}
  };
  const auto defaults = xeg::resolve_gpu_profile(config, "unlisted");
  DOCTEST_CHECK(defaults.dag.bitmap_batch_size == 32);
  DOCTEST_CHECK(defaults.dag.exact_batch_size == 256);
  DOCTEST_CHECK(defaults.dag.low_activity_percent == 3);
  DOCTEST_CHECK(defaults.dag.hybrid_levelization);
  const auto selected = xeg::resolve_gpu_profile(config, "boolE/mul32_map.rename");
  DOCTEST_CHECK(selected.dag.bitmap_batch_size == 8);
  DOCTEST_CHECK(selected.dag.bitmap_max_direct_iterations == 10);
  DOCTEST_CHECK(selected.bitmap_bits == 512);
  DOCTEST_CHECK(selected.dag_threads == 64);
  const auto roundtrip = xeg::resolve_gpu_profile({{"schema_version", 1}, {"defaults", selected.to_json()}}, "");
  DOCTEST_CHECK(roundtrip.to_json() == selected.to_json());
}

TEST_CASE("GPU profiles reject typos and unsafe launch or controller values") {
  for (const nlohmann::json values : {
      nlohmann::json{{"bitmap_batch_size", 0}}, {{"exact_batch_size", 1025}},
      {{"bitmap_batch_size", -1}}, {{"bitmap_batch_size", 8.5}},
      {{"bitmap_batch_size", 4294967296ULL}}, {{"bitmap_batch_szie", 8}},
      {{"hybrid_levelization", 1}}, {{"persistent_threads", 33}},
      {{"persistent_enter_frontier", 2048}}, {{"persistent_small_levels", 0}},
      {{"low_activity_percent", 101}}, {{"low_activity_patience", 5}},
      {{"bitmap_max_direct_iterations", 0}}, {{"insertion_policy", 3}},
      {{"kernel", 3}}, {{"kernel", 0}, {"bitmap_bits", 512}},
      {{"kernel", 2}, {"dag_threads", 48}}, {{"kernel", 2}, {"dag_threads", 64}, {"bitmap_bits", 96}}}) {
    DOCTEST_CHECK_THROWS_AS(xeg::resolve_gpu_profile({{"schema_version", 1}, {"defaults", values}}, ""), std::invalid_argument);
    // Validate even unused design entries, so a bad manifest cannot hide.
    DOCTEST_CHECK_THROWS_AS(xeg::resolve_gpu_profile({{"schema_version", 1}, {"designs", {{"other", values}}}}, "selected"), std::invalid_argument);
  }
  DOCTEST_CHECK_THROWS_AS(xeg::resolve_gpu_profile({{"defaults", nlohmann::json::object()}}, ""), std::invalid_argument);
  DOCTEST_CHECK_THROWS_AS(xeg::resolve_gpu_profile({{"schema_version", 2}}, ""), std::invalid_argument);
}
