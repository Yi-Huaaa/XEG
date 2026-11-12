#include <xeg/XEG.hpp>
#include <iomanip>

int main(int argc, char** argv) {
  if (argc < 2 || argc > 10) {
    std::cerr << "Usage: " << argv[0]
      << " INPUT [KERNEL=0] [DAG_THREADS=1] [BITMAP_BITS=0] [ILP_SECONDS=360] [SOLVER_WORKERS=8] [REPORT.json] [REPEATS=1] [DETERMINISTIC_SECONDS=0]\n"
      << "KERNEL: 0=Thread, 1=Warp, 2=Block. ILP_SECONDS=0 skips solving.\n";
    return 2;
  }
  try {
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
    const auto kernel = static_cast<DagKernel>(argc > 2 ? integer(argv[2],true) : 0);
    const int threads = argc > 3 ? integer(argv[3],false) : 1;
    const int bits = argc > 4 ? integer(argv[4],true) : 0;
    const double seconds = argc > 5 ? number(argv[5]) : 360;
    const int workers = argc > 6 ? integer(argv[6],false) : 8;
    const int repeats = argc > 8 ? integer(argv[8],false) : 1;
    const double deterministic = argc > 9 ? number(argv[9]) : 0;
    XEG graph; graph.read(argv[1]);
    auto reports = nlohmann::json::array();
    for (int i = 0; i < repeats; ++i) {
      auto report = graph.run_gpu(seconds,kernel,threads,bits,workers,deterministic);
      report["input"] = argv[1]; report["repeat"] = i;
      auto summary = report;
      for (const char* key : {"solver", "solution"})
        if (summary.contains(key)) summary[key].erase("selected_enodes");
      std::cout << "[GPU Result] " << summary.dump() << '\n';
      reports.push_back(std::move(report));
      if (argc > 7) {
        std::ofstream output(argv[7]);
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
