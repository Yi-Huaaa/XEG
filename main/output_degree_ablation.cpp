#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <g_ext/G_EXT.hpp>

namespace {

// Default seed used only by the legacy build-and-run mode.
//
// Dump-only mode receives its random seed explicitly from the
// command line so multiple independently generated graph
// instances can be created for the same input and multiplier.
constexpr uint32_t
  kDefaultOutputDegreeAblationRandomSeed = 113u;

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
    << "Usage:\n\n"

    << "  Dump-only mode:\n"
    << "  "
    << program_name
    << " --dump"
    << " <input_file>"
    << " <OUTPUT_DEGREE_MULTIPLIER>"
    << " <RANDOM_SEED>\n\n"

    << "  Legacy build-and-run mode:\n"
    << "  "
    << program_name
    << " <input_file>"
    << " <DAG_KERNEL>"
    << " <NUM_THREAD_DAG_CONSTR>"
    << " <BITMAP_BITS>"
    << " <OUTPUT_DEGREE_MULTIPLIER>\n\n"

    << "Dump-only mode:\n"
    << "  Reads and augments the graph, writes the generated\n"
    << "  JSON file, and exits without calling run().\n\n"

    << "  Output directory:\n"
    << "    /home/ychung79/gpu_extraction/"
    << "ablation_benchmarks\n\n"

    << "DAG_KERNEL:\n"
    << "  0: Thread\n"
    << "  1: Warp\n"
    << "  2: Block\n\n"

    << "BITMAP_BITS:\n"
    << "  Thread: 0\n"
    << "  Warp:   0, 32, 64, 128, 256, "
    << "512, 1024, 2048\n"
    << "  Block:  0, 32, 64, 128, 256, "
    << "512, 1024, 2048\n\n"

    << "OUTPUT_DEGREE_MULTIPLIER:\n"
    << "  Positive integer multiplier applied by the\n"
    << "  output-degree ablation preprocessing step.\n"
    << "  Typical values: 2, 4, 8, 16, 32, 64\n\n"

    << "RANDOM_SEED:\n"
    << "  Required in dump-only mode. Different seeds normally\n"
    << "  generate different augmented graphs. Reusing the same\n"
    << "  input, multiplier, and seed reproduces the same graph.\n\n"

    << "Legacy build-and-run mode uses default seed "
    << kDefaultOutputDegreeAblationRandomSeed
    << ".\n\n"

    << "Examples:\n"

    << "  "
    << program_name
    << " --dump input.json 2 42\n"

    << "  "
    << program_name
    << " --dump input.json 2 43\n\n"

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
  // Supported command-line modes:
  //
  // Dump-only:
  //
  //   argv[1]: --dump
  //   argv[2]: original input file
  //   argv[3]: output-degree multiplier
  //   argv[4]: random seed
  //
  // Legacy build-and-run:
  //
  //   argv[1]: input file
  //   argv[2]: DAG kernel
  //   argv[3]: DAG execution width
  //   argv[4]: bitmap size
  //   argv[5]: output-degree multiplier
  //
  // Dump-only mode receives an explicit random seed so several
  // independently generated graph instances can be produced.
  // Legacy mode continues to use the compile-time default seed.
  const bool dump_only_mode =
    argc >= 2 &&
    std::string(argv[1]) ==
      "--dump";

  const bool valid_argument_count =
    dump_only_mode
      ? argc == 5
      : argc == 6;

  if (!valid_argument_count) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    // ==========================================================
    // Dump-only mode
    //
    // This mode constructs one augmented graph, serializes it,
    // and exits before tree or DAG construction begins.
    // ==========================================================
    if (dump_only_mode) {
      const std::string input_path =
        argv[2];

      const uint32_t
        output_degree_multiplier =
          parse_uint32_argument(
            argv[3],
            "OUTPUT_DEGREE_MULTIPLIER");

      const uint32_t random_seed =
        parse_uint32_argument(
          argv[4],
          "RANDOM_SEED");

      if (output_degree_multiplier == 0u) {
        std::cerr
          << "Invalid OUTPUT_DEGREE_MULTIPLIER: 0\n"
          << "The multiplier must be greater than zero.\n";

        return 1;
      }

      std::cout
        << "-------------------------------------\n";

      std::cout
        << "Experiment: Output-degree graph generation\n";

      std::cout
        << "Mode: Dump only\n";

      std::cout
        << "Input file: "
        << input_path
        << "\n";

      std::cout
        << "OUTPUT_DEGREE_MULTIPLIER: "
        << output_degree_multiplier
        << "x\n";

      std::cout
        << "RANDOM_SEED: "
        << random_seed
        << " (command line)\n";

      std::cout
        << "Output directory: "
        << "/home/ychung79/gpu_extraction/"
        << "ablation_benchmarks\n";

      std::cout
        << "-------------------------------------\n";

      G_EXT g_ext;

      const auto read_start =
        std::chrono::steady_clock::now();

      g_ext.read_output_degree_exp(
        input_path,
        output_degree_multiplier,
        random_seed,
        true);

      const auto read_end =
        std::chrono::steady_clock::now();

      const std::chrono::duration<double>
        read_duration =
          read_end -
          read_start;

      std::cout
        << "Ablation Build and Dump Time: "
        << g_ext.round_to(
             read_duration.count() * 1000.0,
             0.001)
        << " (ms)\n";

      // Intentionally do not call g_ext.run().
      return 0;
    }

    // ==========================================================
    // Existing legacy build-and-run mode
    // ==========================================================
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
            << "Valid values: "
            << "0, 32, 64, 128, 256, "
            << "512, 1024, 2048\n";
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
      << kDefaultOutputDegreeAblationRandomSeed
      << " (legacy default)\n";

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
      kDefaultOutputDegreeAblationRandomSeed,
      false);

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