#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <g_ext/G_EXT.hpp>

namespace {

// The output-degree augmentation must be deterministic so that
// different DAG kernels and bitmap configurations see exactly
// the same augmented input graph.
constexpr uint32_t
  kOutputDegreeAblationRandomSeed = 42u;

uint32_t parse_uint32_argument(
  const char* text,
  const char* argument_name)
{
  if (text == nullptr) {
    throw std::invalid_argument(
      std::string(argument_name) +
      " is missing");
  }

  const std::string value_string(text);

  if (value_string.empty()) {
    throw std::invalid_argument(
      std::string(argument_name) +
      " must not be empty");
  }

  // std::stoul may accept a leading minus sign and convert it
  // into an unsigned value. Reject negative input explicitly.
  if (value_string.front() == '-') {
    throw std::invalid_argument(
      std::string(argument_name) +
      " must not be negative");
  }

  size_t parsed_characters = 0u;

  const unsigned long parsed_value =
    std::stoul(
      value_string,
      &parsed_characters,
      10);

  if (parsed_characters !=
      value_string.size()) {
    throw std::invalid_argument(
      std::string(argument_name) +
      " must be an unsigned integer");
  }

  if (parsed_value >
      static_cast<unsigned long>(
        std::numeric_limits<uint32_t>::max())) {
    throw std::out_of_range(
      std::string(argument_name) +
      " exceeds uint32_t range");
  }

  return static_cast<uint32_t>(
    parsed_value);
}


void print_usage(
  const char* program_name)
{
  std::cerr
    << "Usage:\n"
    << "  "
    << program_name
    << " <input_file>"
    << " <DAG_KERNEL>"
    << " <NUM_THREAD_DAG_CONSTR>"
    << " <BITMAP_BITS>"
    << " <OUTPUT_DEGREE_MULTIPLIER>\n\n"

    << "DAG_KERNEL:\n"
    << "  0: Thread\n"
    << "  1: Warp\n"
    << "  2: Block\n\n"

    << "BITMAP_BITS:\n"
    << "  Thread: 0\n"
    << "  Warp:   0, 32, 64\n"
    << "  Block:  0, 32, 64, 128, 256, 512, 1024, 2048\n\n"

    << "OUTPUT_DEGREE_MULTIPLIER:\n"
    << "  Positive integer multiplier applied by the\n"
    << "  output-degree ablation preprocessing step.\n"
    << "  Typical values: 2, 4, 8, 16, 32, 64\n\n"

    << "The random seed is fixed internally at "
    << kOutputDegreeAblationRandomSeed
    << " for reproducibility.\n\n"

    << "Examples:\n"
    << "  "
    << program_name
    << " input.json 0 1 0 8\n"

    << "  "
    << program_name
    << " input.json 1 128 64 16\n"

    << "  "
    << program_name
    << " input.json 2 128 64 32\n";
}


bool validate_bitmap_bits(
  const DagKernel dag_kernel,
  const uint32_t bitmap_bits)
{
  switch (dag_kernel) {
    case DagKernel::Thread:
      return
        bitmap_bits == 0u;

    case DagKernel::Warp:
      return
        bitmap_bits == 0u ||
        bitmap_bits == 32u ||
        bitmap_bits == 64u;

    case DagKernel::Block:
      if (bitmap_bits == 0u) {
        return true;
      }

      return
        bitmap_bits >= 32u &&
        bitmap_bits <= 2048u &&
        (
          bitmap_bits &
          (bitmap_bits - 1u)
        ) == 0u;
  }

  return false;
}


void print_bitmap_description(
  const DagKernel dag_kernel,
  const uint32_t bitmap_bits)
{
  std::cout
    << "BITMAP_BITS: "
    << bitmap_bits;

  if (bitmap_bits == 0u) {
    std::cout
      << " (Exact overlap detection)";
  }
  else if (dag_kernel ==
           DagKernel::Warp) {
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
}

}  // namespace


int main(
  int argc,
  char* argv[])
{
  // Required arguments:
  //
  //   argv[1]: input file
  //   argv[2]: DAG kernel
  //   argv[3]: DAG execution width
  //   argv[4]: bitmap size
  //   argv[5]: output-degree multiplier
  //
  // The random seed is intentionally not exposed through the
  // command line. Every run uses the same fixed seed.
  if (argc != 6) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    const std::string input_path =
      argv[1];

    const uint32_t dag_kernel_value =
      parse_uint32_argument(
        argv[2],
        "DAG_KERNEL");

    if (dag_kernel_value > 2u) {
      std::cerr
        << "Invalid DAG_KERNEL: "
        << dag_kernel_value
        << "\n"
        << "Valid values are:\n"
        << "  0: Thread\n"
        << "  1: Warp\n"
        << "  2: Block\n";

      return 1;
    }

    const DagKernel dag_kernel =
      static_cast<DagKernel>(
        dag_kernel_value);

    const uint32_t
      num_thread_dag_constr =
        parse_uint32_argument(
          argv[3],
          "NUM_THREAD_DAG_CONSTR");

    const uint32_t bitmap_bits =
      parse_uint32_argument(
        argv[4],
        "BITMAP_BITS");

    const uint32_t
      output_degree_multiplier =
        parse_uint32_argument(
          argv[5],
          "OUTPUT_DEGREE_MULTIPLIER");

    if (output_degree_multiplier == 0u) {
      std::cerr
        << "Invalid OUTPUT_DEGREE_MULTIPLIER: 0\n"
        << "The multiplier must be greater than zero.\n";

      return 1;
    }

    if (!validate_bitmap_bits(
          dag_kernel,
          bitmap_bits)) {
      std::cerr
        << "Invalid BITMAP_BITS="
        << bitmap_bits
        << " for DAG_KERNEL="
        << dag_kernel_value
        << "\n";

      switch (dag_kernel) {
        case DagKernel::Thread:
          std::cerr
            << "Valid value: 0\n";
          break;

        case DagKernel::Warp:
          std::cerr
            << "Valid values: 0, 32, 64\n";
          break;

        case DagKernel::Block:
          std::cerr
            << "Valid values: "
            << "0, 32, 64, 128, 256, "
            << "512, 1024, 2048\n";
          break;
      }

      return 1;
    }

    std::cout
      << "-------------------------------------\n";

    std::cout
      << "Experiment: Output-degree ablation\n";

    std::cout
      << "Input file: "
      << input_path
      << "\n";

    std::cout
      << "DAG_KERNEL: "
      << dag_kernel_value
      << "\n";

    std::cout
      << "NUM_THREAD_DAG_CONSTR: "
      << num_thread_dag_constr
      << "\n";

    print_bitmap_description(
      dag_kernel,
      bitmap_bits);

    std::cout
      << "OUTPUT_DEGREE_MULTIPLIER: "
      << output_degree_multiplier
      << "x\n";

    std::cout
      << "RANDOM_SEED: "
      << kOutputDegreeAblationRandomSeed
      << " (fixed)\n";

    std::cout
      << "-------------------------------------\n";

    G_EXT g_ext;

    // ==========================================================
    // Ablation-specific read and graph augmentation
    //
    // This is the only semantic difference from simple:
    //
    //   simple:
    //     g_ext.read(input_path)
    //
    //   output-degree ablation:
    //     g_ext.read_output_degree_exp(...)
    //
    // read_output_degree_exp() first performs the original read
    // and pruning, then augments the e-node -> leaf-e-class
    // child relationships and rebuilds every affected graph
    // representation.
    // ==========================================================
    const auto read_start =
      std::chrono::steady_clock::now();

    g_ext.read_output_degree_exp(
      input_path,
      output_degree_multiplier,
      kOutputDegreeAblationRandomSeed);

    const auto read_end =
      std::chrono::steady_clock::now();

    const std::chrono::duration<double>
      read_duration =
        read_end -
        read_start;

    std::cout
      << "Ablation Read Time: "
      << g_ext.round_to(
           read_duration.count() * 1000.0,
           0.001)
      << " (ms)\n";

    // Reuse the production XEG execution path without any
    // ablation-specific branch inside run().
    //
    // Therefore, after preprocessing, tree construction, DAG
    // construction, filtering, and CP-SAT are exactly the same
    // implementations used by the normal simple executable.
    g_ext.run(
      false,
      360,
      dag_kernel,
      num_thread_dag_constr,
      bitmap_bits);

    return 0;
  }
  catch (const std::exception& error) {
    std::cerr
      << "Output-degree ablation failed: "
      << error.what()
      << "\n";

    return 1;
  }
}