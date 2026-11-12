#pragma once

#include <xeg/gpu_config.hpp>
#include <filesystem>
#include <stdexcept>
#include <vector>

// Shared by all CUDA CLI entry points; solver-specific arguments stay with
// their existing drivers. Explicit geometry still overrides the profile.
struct GpuProfileArgs {
  std::vector<char*> positional;
  std::string config_path, design;

  GpuProfileArgs(int argc, char** argv) : positional{argv[0]} {
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--config" || argument == "--design") {
        if (++i == argc) throw std::invalid_argument("Missing value after " + argument);
        auto& value = argument == "--config" ? config_path : design;
        if (!value.empty()) throw std::invalid_argument("Duplicate option: " + argument);
        value = argv[i];
        if (value.empty()) throw std::invalid_argument("Empty value after " + argument);
      } else if (argument.rfind("--", 0) == 0 && argument != "--report") {
        throw std::invalid_argument("Unknown option: " + argument);
      } else positional.push_back(argv[i]);
    }
  }

  xeg::GpuProfile resolve(const std::string& input_path) {
    if (design.empty()) {
      const std::filesystem::path input(input_path);
      design = input.parent_path().filename().string() + "/" + input.stem().string();
    }
    return config_path.empty() ? xeg::default_gpu_profile(design)
                               : xeg::load_gpu_profile(config_path, design);
  }
};
