#pragma once

#include <xeg/gpu_config.hpp>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <limits>

// Shared by all CUDA CLI entry points; solver-specific arguments stay with
// their existing drivers. Explicit geometry still overrides the profile.
struct GpuProfileArgs {
  std::vector<char*> positional;
  std::string config_path, design, solver_seconds, workers;
  bool profile_matched = false;

  GpuProfileArgs(int argc, char** argv, bool solver_options = false) : positional{argv[0]} {
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--config" || argument == "--design" ||
          (solver_options && (argument == "--solver-seconds" || argument == "--workers"))) {
        if (++i == argc) throw std::invalid_argument("Missing value after " + argument);
        auto& value = argument == "--config" ? config_path : argument == "--design" ? design :
                      argument == "--solver-seconds" ? solver_seconds : workers;
        if (!value.empty()) throw std::invalid_argument("Duplicate option: " + argument);
        value = argv[i];
        if (value.empty()) throw std::invalid_argument("Empty value after " + argument);
      } else if (argument.rfind("--", 0) == 0 && argument != "--report") {
        throw std::invalid_argument("Unknown option: " + argument);
      } else positional.push_back(argv[i]);
    }
  }

  xeg::GpuProfile resolve(const std::string& input_path) {
    const bool explicit_design = !design.empty();
    if (design.empty()) {
      const auto input = std::filesystem::absolute(input_path).lexically_normal();
      design = input.parent_path().filename().string() + "/" + input.stem().string();
    }
    auto result = config_path.empty() ? xeg::default_gpu_profile(design)
                                     : xeg::load_gpu_profile(config_path, design);
    profile_matched = result.matched;
    if (explicit_design && !profile_matched)
      throw std::invalid_argument("Unknown explicit --design: " + design);
    std::cout << "[GPU Settings] design=" << design << " profile_matched="
              << (profile_matched ? "true" : "false") << '\n';
    return result;
  }
  // An explicit kernel starts from canonical geometry; further positional
  // values override it. With no geometry arguments the profile stays intact.
  static xeg::GpuProfile geometry(xeg::GpuProfile profile, int argc, char** argv) {
    auto integer = [](const char* text) {
      const std::string value(text);
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("GPU geometry must be a nonnegative integer");
      const auto result = std::stoull(value);
      if (result > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("GPU geometry exceeds uint32 range");
      return static_cast<uint32_t>(result);
    };
    if (argc > 2) {
      profile.kernel = integer(argv[2]);
      profile.dag_threads = profile.kernel == 0 ? 1 : profile.kernel == 1 ? 32 : 64;
      profile.bitmap_bits = profile.kernel == 2 ? 512 : 0;
    }
    if (argc > 3) profile.dag_threads = integer(argv[3]);
    if (argc > 4) profile.bitmap_bits = integer(argv[4]);
    profile.validate();
    return profile;
  }
};
