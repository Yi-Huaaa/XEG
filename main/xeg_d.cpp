#include <chrono>
#include <iostream>
#include <string>

#include <xeg/XEG.hpp>

int main(
  int argc,
  char* argv[])
{
  if (argc != 2) {
    std::cerr
      << "Usage:\n"
      << "  "
      << argv[0]
      << " <input_file>\n";

    return 1;
  }

  const std::string input_path =
    argv[1];

  std::cout
    << "-------------------------------------\n"
    << "Mode: XEG-D\n"
    << "Input file: "
    << input_path
    << "\n"
    << "CP-SAT time limit: 86400 seconds\n"
    << "Search space: complete post-read e-graph\n"
    << "Warm start: single-thread CPU tree construction\n"
    << "-------------------------------------\n";

  XEG xeg;

  const auto read_start =
    std::chrono::steady_clock::now();

  xeg.read(input_path);

  const auto read_end =
    std::chrono::steady_clock::now();

  const std::chrono::duration<double>
    read_duration =
      read_end - read_start;

  std::cout
    << "Read Time: "
    << xeg.round_to(
         read_duration.count() * 1000.0,
         0.001)
    << " (ms)\n";

  xeg.run_xeg_d();

  return 0;
}
