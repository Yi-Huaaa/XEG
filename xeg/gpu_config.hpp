#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

namespace xeg {

// Defaults preserve the original CUDA batching and levelization controller.
// These are host launch/controller values; kernels retain their algorithms.
struct GpuDagOptions {
  uint32_t insertion_policy = 2;  // FullBatch=0, BatchEarlyStop=1, BatchThenDirect=2.
  uint32_t exact_batch_size = 256;
  uint32_t bitmap_batch_size = 32;
  uint32_t low_activity_minimum_iterations = 4;
  uint32_t low_activity_percent = 3;
  uint32_t low_activity_patience = 2;
  uint32_t bitmap_max_direct_iterations = 5;
  bool hybrid_levelization = true;
  uint32_t persistent_threads = 512;
  uint32_t persistent_enter_frontier = 512;
  uint32_t persistent_small_levels = 8;
  uint32_t persistent_exit_frontier = 1024;

  void validate() const;
  nlohmann::json to_json() const;
};

struct GpuProfile {
  uint32_t kernel = 0;
  uint32_t dag_threads = 1;
  uint32_t bitmap_bits = 0;
  GpuDagOptions dag;

  void validate() const;
  nlohmann::json to_json() const;
};

// An absent design inherits defaults. Unknown option names and invalid types
// fail explicitly, so a misspelled experiment setting cannot silently run.
GpuProfile resolve_gpu_profile(const nlohmann::json& document, const std::string& design);
GpuProfile load_gpu_profile(const std::string& path, const std::string& design);

}  // namespace xeg
