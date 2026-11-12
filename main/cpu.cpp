#include <xeg/XEG.hpp>
#include <iomanip>

int main(int argc, char** argv) {
  if (argc < 2 || argc > 8) {
    std::cerr << "Usage: " << argv[0]
              << " INPUT [THREADS=1] [ILP_SECONDS=360] [SOLVER_WORKERS=8] [REPORT.json] [REPEATS=1] [DETERMINISTIC_SECONDS=0]\n"
              << "ILP_SECONDS=0 runs construction and validation only.\n";
    return 2;
  }
  try {
    auto integer = [](const char* s) {
      size_t end = 0;
      const int value = std::stoi(s,&end);
      if (end != std::string(s).size() || value < 1) throw std::invalid_argument("Expected a positive integer");
      return value;
    };
    const int threads = argc > 2 ? integer(argv[2]) : 1;
    size_t end = 0;
    const double seconds = argc > 3 ? std::stod(argv[3],&end) : 360;
    if (argc > 3 && end != std::string(argv[3]).size()) throw std::invalid_argument("Invalid time limit");
    const int workers = argc > 4 ? integer(argv[4]) : 8;
    const int repeats = argc > 6 ? integer(argv[6]) : 1;
    end = 0;
    const double deterministic = argc > 7 ? std::stod(argv[7],&end) : 0;
    if (argc > 7 && end != std::string(argv[7]).size()) throw std::invalid_argument("Invalid deterministic limit");
    XEG graph;
    graph.read(argv[1]);
    auto reports = nlohmann::json::array();
    for (int i=0;i<repeats;++i) {
      auto report = graph.run_cpu(threads,seconds,workers,deterministic);
      report["input"] = argv[1]; report["repeat"] = i;
      auto summary = report;
      for (const char* key : {"solver", "solution"})
        if (summary.contains(key)) summary[key].erase("selected_enodes");
      std::cout << "[CPU Result] " << summary.dump() << '\n';
      reports.push_back(std::move(report));
      if (argc > 5) {
        std::ofstream output(argv[5]);
        if (!output) throw std::runtime_error("Cannot open report output");
        output << std::setw(2) << reports << '\n';
        if (!output) throw std::runtime_error("Cannot write report output");
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "CPU extraction failed: " << error.what() << '\n';
    return 1;
  }
}
