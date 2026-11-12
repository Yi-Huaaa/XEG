#include "gpu_config.hpp"
#include "gpu_default_profile.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace xeg {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
bool power_of_two(uint32_t value) { return value && !(value & (value - 1)); }
uint32_t unsigned_value(const nlohmann::json& value, const std::string& key) {
  if (!value.is_number_integer() || value < 0 || value > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("GPU option must be a uint32 integer: " + key);
  return value.get<uint32_t>();
}
void apply(GpuProfile& profile, const nlohmann::json& values) {
  require(values.is_object(), "GPU profile must be an object");
  for (const auto& [key, value] : values.items()) {
    if (key == "hybrid_levelization") {
      require(value.is_boolean(), "hybrid_levelization must be boolean");
      profile.dag.hybrid_levelization = value.get<bool>();
      continue;
    }
    uint32_t* destination = nullptr;
    if (key == "kernel") destination = &profile.kernel;
    else if (key == "dag_threads") destination = &profile.dag_threads;
    else if (key == "bitmap_bits") destination = &profile.bitmap_bits;
#define XEG_GPU_OPTION(name) else if (key == #name) destination = &profile.dag.name
    XEG_GPU_OPTION(insertion_policy);
    XEG_GPU_OPTION(exact_batch_size);
    XEG_GPU_OPTION(bitmap_batch_size);
    XEG_GPU_OPTION(low_activity_minimum_iterations);
    XEG_GPU_OPTION(low_activity_percent);
    XEG_GPU_OPTION(low_activity_patience);
    XEG_GPU_OPTION(bitmap_max_direct_iterations);
    XEG_GPU_OPTION(persistent_threads);
    XEG_GPU_OPTION(persistent_enter_frontier);
    XEG_GPU_OPTION(persistent_small_levels);
    XEG_GPU_OPTION(persistent_exit_frontier);
#undef XEG_GPU_OPTION
    else throw std::invalid_argument("Unknown GPU option: " + key);
    *destination = unsigned_value(value, key);
  }
}
}  // namespace

void GpuDagOptions::validate() const {
  require(insertion_policy <= 2, "insertion_policy must be 0, 1, or 2");
  require(exact_batch_size >= 1 && exact_batch_size <= 1024, "exact_batch_size must be in [1, 1024]");
  require(bitmap_batch_size >= 1 && bitmap_batch_size <= 1024, "bitmap_batch_size must be in [1, 1024]");
  require(low_activity_percent >= 1 && low_activity_percent <= 100, "low_activity_percent must be in [1, 100]");
  require(low_activity_patience > 0 && low_activity_minimum_iterations >= low_activity_patience,
          "low_activity_minimum_iterations must cover positive low_activity_patience");
  require(bitmap_max_direct_iterations > 0, "bitmap_max_direct_iterations must be positive");
  require(persistent_threads >= 32 && persistent_threads <= 1024 && persistent_threads % 32 == 0,
          "persistent_threads must be a multiple of 32 in [32, 1024]");
  require(persistent_enter_frontier > 0 && persistent_exit_frontier >= persistent_enter_frontier,
          "persistent_exit_frontier must cover positive persistent_enter_frontier");
  require(persistent_small_levels > 0, "persistent_small_levels must be positive");
}

nlohmann::json GpuDagOptions::to_json() const {
  nlohmann::json result = nlohmann::json::object();
#define XEG_GPU_OPTION(name) result[#name] = name
  XEG_GPU_OPTION(insertion_policy);
  XEG_GPU_OPTION(exact_batch_size);
  XEG_GPU_OPTION(bitmap_batch_size);
  XEG_GPU_OPTION(low_activity_minimum_iterations);
  XEG_GPU_OPTION(low_activity_percent);
  XEG_GPU_OPTION(low_activity_patience);
  XEG_GPU_OPTION(bitmap_max_direct_iterations);
  XEG_GPU_OPTION(hybrid_levelization);
  XEG_GPU_OPTION(persistent_threads);
  XEG_GPU_OPTION(persistent_enter_frontier);
  XEG_GPU_OPTION(persistent_small_levels);
  XEG_GPU_OPTION(persistent_exit_frontier);
#undef XEG_GPU_OPTION
  return result;
}

void GpuProfile::validate() const {
  dag.validate();
  require(kernel <= 2, "kernel must be 0 (Thread), 1 (Warp), or 2 (Block)");
  if (kernel == 0) {
    require(dag_threads == 1 && bitmap_bits == 0, "Thread kernel requires dag_threads=1 and bitmap_bits=0");
  } else {
    require(dag_threads <= 1024 && (kernel == 1 ? dag_threads >= 32 && dag_threads % 32 == 0
                                                             : dag_threads >= 64 && power_of_two(dag_threads)),
            "Invalid dag_threads for Warp/Block kernel");
    require(bitmap_bits == 0 || (bitmap_bits >= 32 && bitmap_bits <= 4096 && power_of_two(bitmap_bits)),
            "bitmap_bits must be 0 or a power of two in [32, 4096]");
  }
}

nlohmann::json GpuProfile::to_json() const {
  auto result = dag.to_json();
  result["kernel"] = kernel;
  result["dag_threads"] = dag_threads;
  result["bitmap_bits"] = bitmap_bits;
  return result;
}

GpuProfile resolve_gpu_profile(const nlohmann::json& document, const std::string& design) {
  require(document.is_object(), "GPU configuration must be an object");
  for (const auto& [key, value] : document.items()) {
    (void)value;
    require(key == "schema_version" || key == "defaults" || key == "designs", "Unknown GPU configuration section");
  }
  require(document.contains("schema_version") && document["schema_version"].is_number_integer() &&
          document["schema_version"] == 1, "GPU configuration requires schema_version=1");
  GpuProfile defaults;
  if (document.contains("defaults")) apply(defaults, document["defaults"]);
  defaults.validate();
  auto result = defaults;
  if (document.contains("designs")) {
    require(document["designs"].is_object(), "GPU designs must be an object");
    for (const auto& [key, values] : document["designs"].items()) {
      auto profile = defaults;
      apply(profile, values);
      profile.validate();
      if (key == design) { result = profile; result.matched = true; }
    }
  }
  return result;
}

GpuProfile load_gpu_profile(const std::string& path, const std::string& design) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open GPU configuration: " + path);
  nlohmann::json document;
  input >> document;
  require(document.contains("defaults") && document["defaults"].is_object(),
          "Custom GPU configuration requires defaults with kernel, dag_threads, and bitmap_bits");
  for (const char* key : {"kernel", "dag_threads", "bitmap_bits"})
    require(document["defaults"].contains(key),
            "Custom GPU defaults must explicitly specify kernel, dag_threads, and bitmap_bits");
  return resolve_gpu_profile(document, design);
}

GpuProfile default_gpu_profile(const std::string& design) {
  static const auto document = nlohmann::json::parse(xeg_default_gpu_profiles);
  return resolve_gpu_profile(document, design);
}
}  // namespace xeg
