#include <xeg/XEG.hpp>
#include <iomanip>
#include "gpu_profile_args.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
      << " INPUT [KERNEL] [DAG_THREADS] [BITMAP_BITS] [ILP_SECONDS=360] [SOLVER_WORKERS=8] [REPORT.json] [REPEATS=1] [DETERMINISTIC_SECONDS=0]\n"
      << "Options: --config FILE.json --design FAMILY/STEM --no-profile --report FILE.json\n"
      << "Defaults: measured design profile (unlisted: Block/64/512). --no-profile restores Thread/1/0.\nKERNEL: 0=Thread, 1=Warp, 2=Block. ILP_SECONDS=0 skips solving.\n";
    return 2;
  }
  try {
    GpuProfileArgs profile_args(argc, argv);
    std::vector<char*> args{argv[0]};
    std::string report_path;
    const auto& positional = profile_args.positional;
    for (size_t i = 1; i < positional.size(); ++i) {
      const std::string argument = positional[i];
      if (argument == "--report") {
        if (++i == positional.size()) throw std::invalid_argument("Missing value after --report");
        if (!report_path.empty()) throw std::invalid_argument("Duplicate --report");
        report_path = positional[i];
      } else if (argument.rfind("--", 0) == 0) {
        throw std::invalid_argument("Unknown option: " + argument);
      } else args.push_back(positional[i]);
    }
    if (args.size() < 2 || args.size() > 10) throw std::invalid_argument("Invalid number of positional arguments");
    argc = static_cast<int>(args.size());
    argv = args.data();
    const auto profile = profile_args.resolve(argv[1]);
    auto integer = [](const char* text, bool zero) {
      size_t end = 0; const int value = std::stoi(text,&end);
      if (end != std::string(text).size() || value < (zero ? 0 : 1))
        throw std::invalid_argument("Invalid integer argument");
      return value;
    };
    auto number = [](const char* text) {
      size_t end = 0; const double value = std::stod(text,&end);
      if (end != std::string(text).size()) throw std::invalid_argument("Invalid numeric argument");
      return value;
    };
    const auto kernel = static_cast<DagKernel>(argc > 2 ? integer(argv[2],true) : profile.kernel);
    const int threads = argc > 3 ? integer(argv[3],false) : profile.dag_threads;
    const int bits = argc > 4 ? integer(argv[4],true) : profile.bitmap_bits;
    const double seconds = argc > 5 ? number(argv[5]) : 360;
    const int workers = argc > 6 ? integer(argv[6],false) : 8;
    const int repeats = argc > 8 ? integer(argv[8],false) : 1;
    const double deterministic = argc > 9 ? number(argv[9]) : 0;
    if (argc > 7) {
      if (!report_path.empty()) throw std::invalid_argument("Report specified both positionally and with --report");
      report_path = argv[7];
    }
    XEG graph; graph.read(argv[1]);
    auto reports = nlohmann::json::array();
    for (int i = 0; i < repeats; ++i) {
      auto report = graph.run_gpu(seconds,kernel,threads,bits,workers,deterministic,profile.dag);
      report["input"] = argv[1]; report["repeat"] = i;
      report["configuration_source"] = !profile_args.enabled ? "original-defaults"
        : profile_args.config_path.empty() ? "built-in-profile" : "file";
      if (profile_args.enabled) report["design_profile"] = profile_args.design;
      if (!profile_args.config_path.empty()) report["configuration_file"] = profile_args.config_path;
      auto summary = report;
      for (const char* key : {"solver", "solution"})
        if (summary.contains(key)) summary[key].erase("selected_enodes");
      std::cout << "[GPU Result] " << summary.dump() << '\n';
      reports.push_back(std::move(report));
      if (!report_path.empty()) {
        std::ofstream output(report_path);
        if (!output) throw std::runtime_error("Cannot open report output");
        output << std::setw(2) << reports << '\n';
        if (!output) throw std::runtime_error("Cannot write report output");
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "GPU extraction failed: " << error.what() << '\n';
    return 1;
  }
}
