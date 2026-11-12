#include <chrono>
#include "gpu_profile_args.hpp"
#include <iostream>
#include <xeg/XEG.hpp>
#include <vector>
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/base/logging.h" // For LOG(INFO)

int main(int argc, char *argv[]) {
  try {
  GpuProfileArgs profile_args(argc, argv);
  argc = static_cast<int>(profile_args.positional.size());
  argv = profile_args.positional.data();
  // Timer
  std::chrono::duration<double> duration_read;
  
  if (argc < 2 || argc > 5) {
    std::cerr
      << "Defaults: measured GPU design profiles. Options: --config FILE.json --design FAMILY/STEM --no-profile\n"
      << "Usage:\n"
      << "  ./main/xeg_t "
      << "<input_file> "
      << "[DAG_KERNEL] "
      << "[NUM_THREAD_DAG_CONSTR] "
      << "[BITMAP_BITS]\n\n"
      << "DAG_KERNEL:\n"
      << "  0: Thread\n"
      << "  1: Warp\n"
      << "  2: Block\n\n"
      << "BITMAP_BITS:\n"
      << "  Thread: 0\n"
      << "  Warp:   0, 32, 64, 128, 256, 512, 1024, 2048, 4096\n"
      << "  Block:  0, 32, 64, 128, 256, 512, 1024, 2048, 4096\n"
      << "\n"
      << "  0 uses exact overlap detection.\n"
      << "  A nonzero value uses approximate bitmap overlap detection.\n\n"
      << "Examples:\n"
      << "  ./main/xeg_t input.json 0 1 0\n"
      << "  ./main/xeg_t input.json 1 128 0\n"
      << "  ./main/xeg_t input.json 1 128 32\n"
      << "  ./main/xeg_t input.json 1 128 64\n"
      << "  ./main/xeg_t input.json 2 128 0\n"
      << "  ./main/xeg_t input.json 2 128 64\n";

    return 1;
  }
  
  // === Parse arguments ===
  std::string input_path = argv[1];
  const auto profile = profile_args.resolve(input_path);
  int dag_kernel_value = (argc > 2) ? std::stoi(argv[2]) : static_cast<int>(profile.kernel);

  if (dag_kernel_value < 0 || dag_kernel_value > 2) {
    std::cerr
      << "Invalid DAG_KERNEL: " << dag_kernel_value << "\n"
      << "Valid values are:\n"
      << "  0: Thread\n"
      << "  1: Warp\n"
      << "  2: Block\n";
    return 1;
  }

  DagKernel dag_kernel = static_cast<DagKernel>(dag_kernel_value);

  uint32_t num_thread_dag_constr = (argc > 3)
      ? static_cast<uint32_t>(std::stoul(argv[3]))
      : profile.dag_threads;
  uint32_t bitmap_bits = (argc > 4)
      ? static_cast<uint32_t>(std::stoul(argv[4]))
      : profile.bitmap_bits;
      
  bool valid_bitmap_bits =
    false;

  switch (dag_kernel) {
    case DagKernel::Thread:
      // Thread insertion supports only exact overlap detection.
      valid_bitmap_bits =
        bitmap_bits == 0u;

      if (!valid_bitmap_bits) {
        std::cerr
          << "Invalid BITMAP_BITS for Thread DAG kernel: "
          << bitmap_bits << "\n"
          << "Valid value is:\n"
          << "  0\n";

        return 1;
      }

      break;

    case DagKernel::Warp:
      // Warp insertion supports the existing exact path and the
      // two fixed-size register bitmap implementations.
      valid_bitmap_bits =
        bitmap_bits == 0u ||
        (
          bitmap_bits >= 32u &&
          bitmap_bits <= 4096u &&
          (
            bitmap_bits &
            (bitmap_bits - 1u)
          ) == 0u
        );

      if (!valid_bitmap_bits) {
        std::cerr
          << "Invalid BITMAP_BITS for Block DAG kernel: "
          << bitmap_bits << "\n"
          << "Valid values are:\n"
          << "  0, 32, 64, 128, 256, 512, 1024, 2048, 4096\n";

        return 1;
      }

      break;

    case DagKernel::Block:
      // Block insertion supports a runtime-sized power-of-two
      // shared-memory bitmap from 32 through 4096 bits.
      valid_bitmap_bits =
        bitmap_bits == 0u ||
        (
          bitmap_bits >= 32u &&
          bitmap_bits <= 4096u &&
          (
            bitmap_bits &
            (bitmap_bits - 1u)
          ) == 0u
        );

      if (!valid_bitmap_bits) {
        std::cerr
          << "Invalid BITMAP_BITS for Block DAG kernel: "
          << bitmap_bits << "\n"
          << "Valid values are:\n"
          << "  0, 32, 64, 128, 256, 512, 1024, 2048, 4096\n";

        return 1;
      }

      break;
  }
  
  std::cout << "-------------------------------------\n";
  std::cout << "Input file: " << input_path << "\n";
  std::cout << "DAG_KERNEL: " << dag_kernel_value << "\n";
  std::cout << "NUM_THREAD_DAG_CONSTR: " << num_thread_dag_constr << "\n";  
  std::cout
    << "BITMAP_BITS: "
    << bitmap_bits;

  if (bitmap_bits == 0u) {
    std::cout
      << " (Exact overlap detection)";
  }
  else if (dag_kernel == DagKernel::Warp) {
    std::cout
      << " (Register-based Warp Bitmap-"
      << bitmap_bits
      << ")";
  }
  else {
    std::cout
      << " (Shared-memory Block Bitmap-"
      << bitmap_bits
      << ")";
  }

  std::cout << "\n";
  std::cout << "-------------------------------------\n";


  XEG xeg;

  auto start = std::chrono::steady_clock::now();
    xeg.read(input_path);
  auto end = std::chrono::steady_clock::now();
  duration_read = end - start;
  std::cout << "Read Time: " << xeg.round_to((duration_read.count()) * 1000, 0.001) << " (ms)\n";

  // Run XEG-T exp 
  xeg.set_gpu_dag_options(profile.dag);
  xeg.run_xeg_t_exp(
    360,
    dag_kernel,
    num_thread_dag_constr,
    bitmap_bits,
    333u);

  return 0;
  } catch (const std::exception& error) {
    std::cerr << "GPU extraction failed: " << error.what() << '\n';
    return 1;
  }
}

