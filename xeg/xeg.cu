#include "gpu_common.cuh"
#include "cuopt_solver.hpp"

#include <cuda/atomic>

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>

#include <sys/resource.h>
#include <unistd.h>

#define FAST_BITMAP_CHECK true // Use bitmap strategy or not

/* DEBUG */
// #define PRINT_LOG

#define ENABLE_CHECK
#ifdef ENABLE_CHECK
  #define MY_CHECK(cond, msg) \
    do { \
      if (!(cond)) { \
        std::cerr << "Check failed: " << msg \
                  << " (in " << __FILE__ << ":" << __LINE__ << ")\n"; \
        std::exit(EXIT_FAILURE); \
      } \
    } while (0)
#else
  #define MY_CHECK(cond, msg) ((void)0)
#endif

namespace {

// ============================================================
// Output-degree ablation random-search stopping rule
//
// A successful edge insertion resets this counter to zero.
// An invalid or duplicate source-target sample increments it.
//
// Reaching this limit means the random search has stalled. It
// does not mathematically prove that every legal pair has been
// exhausted.
// ============================================================
constexpr uint32_t
  kOutputDegreeExpMaxConsecutiveFailures = 100u;

static_assert(
  kOutputDegreeExpMaxConsecutiveFailures > 0u,
  "Output-degree failure limit must be positive");

// Fixed output directory for generated output-degree ablation
// benchmark JSON files.
// Note: when open source this part must be updated!!!
constexpr char
  kOutputDegreeExpDumpDirectory[] =
    "/home/ychung79/gpu_extraction/"
    "ablation_benchmarks";

}  // namespace

void XEG::_validate_warp_threads_per_block(
  const uint32_t threads_per_block) const
{
  if (threads_per_block < 32u) {
    throw std::invalid_argument(
      "Warp DAG kernel requires at least "
      "32 threads per CUDA block");
  }

  if ((threads_per_block % 32u) != 0u) {
    throw std::invalid_argument(
      "Warp DAG kernel requires a whole number "
      "of 32-lane warps per CUDA block");
  }

  if (threads_per_block > 1024u) {
    throw std::invalid_argument(
      "Warp DAG kernel supports at most "
      "1024 threads per CUDA block");
  }

  int device = 0;

  cudaError_t status =
    cudaGetDevice(&device);

  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(
        "Failed to query current CUDA device: ")
      + cudaGetErrorString(status));
  }

  cudaDeviceProp properties{};

  status =
    cudaGetDeviceProperties(
      &properties,
      device);

  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(
        "Failed to query CUDA device properties: ")
      + cudaGetErrorString(status));
  }

  if (properties.warpSize != 32) {
    throw std::runtime_error(
      "Warp DAG kernel requires a CUDA device "
      "with a 32-thread warp");
  }

  if (threads_per_block >
      static_cast<uint32_t>(
        properties.maxThreadsPerBlock)) {
    throw std::invalid_argument(
      "Warp DAG threads per block exceed "
      "the CUDA device maximum");
  }
}

void XEG::_validate_warp_bitmap_bits(
  const uint32_t bitmap_bits) const
{
  if (bitmap_bits == 0u) {
    return;
  }

  const bool is_power_of_two =
    (bitmap_bits &
     (bitmap_bits - 1u)) == 0u;

  if (bitmap_bits < 32u ||
      bitmap_bits > 4096u ||
      !is_power_of_two) {
    throw std::invalid_argument(
      "Warp DAG bitmap size must be one of: "
      "0, 32, 64, 128, 256, 512, 1024, 2048, 4096");
  }
}

void XEG::_validate_block_threads_per_candidate(
  const uint32_t threads_per_candidate) const
{
  if (threads_per_candidate < 64u) {
    throw std::invalid_argument(
      "Block DAG kernel requires at least "
      "64 threads per candidate");
  }

  if ((threads_per_candidate &
       (threads_per_candidate - 1u)) != 0u) {
    throw std::invalid_argument(
      "Block DAG kernel requires a power-of-two "
      "thread count");
  }

  if (threads_per_candidate > 1024u) {
    throw std::invalid_argument(
      "Block DAG kernel supports at most "
      "1024 threads per candidate");
  }

  int device = 0;

  cudaError_t status =
    cudaGetDevice(&device);

  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(
        "Failed to query current CUDA device: ")
      + cudaGetErrorString(status));
  }

  cudaDeviceProp properties{};

  status =
    cudaGetDeviceProperties(
      &properties,
      device);

  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(
        "Failed to query CUDA device properties: ")
      + cudaGetErrorString(status));
  }

  if (threads_per_candidate >
      static_cast<uint32_t>(
        properties.maxThreadsPerBlock)) {
    throw std::invalid_argument(
      "Block DAG threads per candidate exceed "
      "the CUDA device maximum");
  }
}

void XEG::_validate_block_bitmap_bits(
  const uint32_t bitmap_bits) const
{
  if (bitmap_bits == 0u) {
    return;
  }

  const bool is_power_of_two =
    (bitmap_bits &
     (bitmap_bits - 1u)) == 0u;

  if (bitmap_bits < 32u ||
      bitmap_bits > 4096u ||
      !is_power_of_two) {
    throw std::invalid_argument(
      "Block DAG bitmap size must be one of: "
      "0, 32, 64, 128, 256, 512, 1024, 2048, 4096");
  }
}

void XEG::_solve_ilp(
  const IlpBackend backend,
  const double time_limit,
  const std::vector<std::vector<uint32_t>>& ecls_its_ende,
  const std::vector<uint32_t>& ende_child,
  const std::vector<uint32_t>& ende_child_ptr,
  const std::vector<uint32_t>& root_eclass_id,
  const std::vector<float>& ende_cost,
  const std::vector<uint32_t>& best_ende_of_ecls,
  const std::vector<bool>& picked_endes,
  const CuoptMode cuopt_mode)
{
  switch (backend) {
    case IlpBackend::CpSat:
      _cpsat_solver(
        time_limit,
        ecls_its_ende,
        ende_child,
        ende_child_ptr,
        root_eclass_id,
        ende_cost,
        best_ende_of_ecls,
        picked_endes);
      return;

    case IlpBackend::CuOpt:
      solve_with_cuopt(
        time_limit,
        ecls_its_ende,
        ende_child,
        ende_child_ptr,
        root_eclass_id,
        ende_cost,
        best_ende_of_ecls,
        picked_endes,
        cuopt_mode);
      return;

    default:
      throw std::invalid_argument(
        "Invalid ILP backend");
  }
}


void XEG::unittest_solve_ilp_backend(
  const IlpBackend ilp_backend,
  const CuoptMode cuopt_mode)
{
  // Minimal valid MILP:
  // root e-class 0 contains two leaf e-nodes.
  const std::vector<std::vector<uint32_t>>
    ecls_its_ende{
      {0u, 1u},
    };

  const std::vector<uint32_t>
    ende_child;

  const std::vector<uint32_t>
    ende_child_ptr{
      0u,
      0u,
      0u,
    };

  const std::vector<uint32_t>
    root_eclass_id{
      0u,
    };

  const std::vector<float>
    ende_cost{
      5.0f,
      2.0f,
    };

  const std::vector<uint32_t>
    best_ende_of_ecls{
      0u,
    };

  const std::vector<bool>
    picked_endes{
      true,
      false,
    };

  _solve_ilp(
    ilp_backend,
    10.0,
    ecls_its_ende,
    ende_child,
    ende_child_ptr,
    root_eclass_id,
    ende_cost,
    best_ende_of_ecls,
    picked_endes,
    cuopt_mode);
}

void XEG::run_xeg_d(
  const double ILP_time_limit)
{
  std::vector<double> cost_of_ecls;
  std::vector<double> cost_of_ende;
  std::vector<uint32_t> best_ende_of_ecls;
  std::vector<bool> picked_endes;

  const auto start =
    std::chrono::steady_clock::now();

  const auto [best_tree_cost_cpu, dag_cost_cpu] =
    _cpu_tree_construction(
      _ende_cost,
      cost_of_ecls,
      cost_of_ende,
      best_ende_of_ecls,
      picked_endes);

  const auto end =
    std::chrono::steady_clock::now();

  const std::chrono::duration<double> duration =
    end - start;

  std::cout
    << "CPU tree construction runtime: "
    << round_to(
         duration.count() * 1000,
         0.01)
    << " (ms)\n";

  printf(
    "CPU version cost:\n"
    "\tunder tree cost model = %.2lf,\n"
    "\tunder DAG cost model = %.2lf\n",
    best_tree_cost_cpu,
    dag_cost_cpu);

  printf(
    "[XEG-D] CP-SAT input: complete post-read e-graph\n");

  printf(
    "[XEG-D] CP-SAT time limit: %.2lf seconds\n",
    ILP_time_limit);

  _cpsat_solver(
    ILP_time_limit,
    _ecls_its_ende,
    _ende_child,
    _ende_child_ptr,
    _root_eclass_id,
    _ende_cost,
    best_ende_of_ecls,
    picked_endes);
}

void XEG::_run_impl(
  const bool ILP_solver,
  const double ILP_time_limit,
  const DagKernel dag_kernel,
  const uint32_t num_thread_dag_constrution,
  const uint32_t dag_bitmap_bits,
  const CpsatInitialSolutionMode initial_solution_mode,
  const uint32_t random_seed,
  const IlpBackend ilp_backend,
  const CuoptMode cuopt_mode)
{
  auto report = _run_gpu_pipeline(ILP_solver ? ILP_time_limit : 0.0,
    dag_kernel, num_thread_dag_constrution, dag_bitmap_bits,
    initial_solution_mode, random_seed, ilp_backend, cuopt_mode, 8, 0.0);
  for (const char* key : {"solver", "solution"})
    if (report.contains(key)) report[key].erase("selected_enodes");
  std::cout << "[GPU Result] " << report.dump() << '\n';
}

void XEG::_build_random_cpsat_initial_solution(
  const uint32_t random_seed,
  const std::vector<bool>& picked_endes_gpu,
  std::vector<uint32_t>& random_best_ende_of_ecls,
  std::vector<bool>& random_picked_endes) const
{
  if (picked_endes_gpu.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "picked_endes_gpu size does not match "
      "the number of e-nodes");
  }

  if (_ecls_its_ende.size() !=
      static_cast<size_t>(_num_ecls)) {
    throw std::logic_error(
      "Original e-class membership is not initialized");
  }

  if (_ecls_its_ende_newG_GPU.size() !=
      static_cast<size_t>(_num_ecls)) {
    throw std::logic_error(
      "GPU-pruned CP-SAT search space is not initialized");
  }

  random_best_ende_of_ecls.assign(
    _num_ecls,
    UINT32_MAX);

  // XEG-T requirement:
  // begin with the GPU-tree picked mask.
  random_picked_endes =
    picked_endes_gpu;

  std::mt19937 random_generator(
    random_seed);

  uint32_t representatives_in_search_space =
    0u;

  uint32_t representatives_absent_from_search_space =
    0u;

  uint64_t representative_checksum =
    1469598103934665603ull;

  for (uint32_t ecls = 0u;
       ecls < _num_ecls;
       ++ecls) {
    // Random representatives must be selected from the
    // original post-read, pre-DAG-construction e-class.
    const auto& original_enodes =
      _ecls_its_ende[ecls];

    if (original_enodes.empty()) {
      throw std::logic_error(
        "XEG-T cannot choose a representative "
        "from an empty e-class");
    }

    std::uniform_int_distribution<size_t>
      choose_index(
        0u,
        original_enodes.size() - 1u);

    const uint32_t random_ende =
      original_enodes[
        choose_index(random_generator)];

    if (random_ende >= _num_ende) {
      throw std::logic_error(
        "XEG-T selected an out-of-range e-node");
    }

    if (_ende_ecls[random_ende] != ecls) {
      throw std::logic_error(
        "XEG-T selected an e-node from the wrong e-class");
    }

    random_best_ende_of_ecls[ecls] =
      random_ende;

    representative_checksum ^=
      static_cast<uint64_t>(random_ende);

    representative_checksum *=
      1099511628211ull;

    // Keep the copied GPU-tree mask, then add every random
    // representative to it.
    random_picked_endes[random_ende] =
      true;

    // Diagnostic only. Do not resample when absent.
    const auto& pruned_enodes =
      _ecls_its_ende_newG_GPU[ecls];

    const bool representative_is_present =
      std::find(
        pruned_enodes.begin(),
        pruned_enodes.end(),
        random_ende) !=
      pruned_enodes.end();

    if (representative_is_present) {
      ++representatives_in_search_space;
    }
    else {
      ++representatives_absent_from_search_space;
    }
  }

  uint32_t unique_random_representatives =
    0u;

  std::vector<uint32_t> representative_seen(
    _num_ende,
    0u);

  for (uint32_t random_ende :
       random_best_ende_of_ecls) {
    if (representative_seen[random_ende] == 0u) {
      representative_seen[random_ende] = 1u;
      ++unique_random_representatives;
    }
  }

  printf(
    "[XEG-T] random_seed=%u, "
    "representatives=%u, "
    "unique=%u, "
    "in_pruned_search_space=%u, "
    "absent_from_pruned_search_space=%u, "
    "checksum=%llu\n",
    random_seed,
    _num_ecls,
    unique_random_representatives,
    representatives_in_search_space,
    representatives_absent_from_search_space,
    static_cast<unsigned long long>(
      representative_checksum));
}

// XEG fucntions
void XEG::_read_input(
  const std::string input_path) 
{
  std::ifstream ifs(input_path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open input file: " + input_path);
  }

  // Parse JSON
  nlohmann::json j; ifs >> j;

  // A new input replaces the previous graph, including derived CSR arrays.
  _ende_name.clear();
  _ende_op.clear();
  _ende_cost.clear();
  _ende_ecls.clear();
  _ende_child.clear();
  _ende_child_ptr.clear();
  _root_eclass_id.clear();
  _ecls_its_ende.clear();
  _ecls_its_ende_1D.clear();
  _ecls_its_ende_1D_ptr.clear();
  _all_root_ende_1D.clear();
  _global_adj.clear();
  _global_inv_adj.clear();
  _global_adj_1D.clear();
  _global_adj_1D_ptr.clear();
  _global_inv_adj_1D.clear();
  _global_inv_adj_1D_ptr.clear();
  _picked_adj.clear();
  _picked_inv_adj.clear();
  _tmp_picked_adj.clear();
  _tmp_picked_inv_adj.clear();
  _indegree.clear();
  _inv_indegree.clear();
  _in_subgraph.clear();
  _roots.clear();
  _leaves.clear();
  _ecls_its_ende_newG.clear();
  _ecls_its_ende_newG_GPU.clear();
  _new_inserted.clear();
  _num_ende = 0; _num_ecls = 0; _max_level = -1;


  // Read in from .json
  uint32_t tmp_num_ecls; // nuber of eclasses 
  uint32_t tmp_num_ende; // number of enodes
  std::vector<std::string> tmp_ende_name; 
  std::vector<std::string> tmp_ende_op; 
  std::vector<float> tmp_ende_cost; 
  std::unordered_map<std::string, uint32_t> tmp_ecls_name_to_id; 
  std::vector<uint32_t> tmp_ende_ecls; // the eclass that enode belonsg to; 
  std::vector<uint32_t> tmp_ende_child; 
  std::vector<uint32_t> tmp_ende_child_ptr; 
  std::vector<uint32_t> tmp_root_eclass_id;

  auto nodes = j["nodes"]; // get number of nodes object
  tmp_num_ende = nodes.size();
  tmp_ende_name.reserve(tmp_num_ende);
  tmp_ende_op.reserve(tmp_num_ende);
  tmp_ende_cost.reserve(tmp_num_ende);
  tmp_ende_ecls.reserve(tmp_num_ende);
  tmp_ende_child_ptr.reserve(tmp_num_ende + 1);

  // lambda function for eclass
  uint32_t ecls_counter = 0;
  auto get_ecls_id = [&](const std::string &ecls_name, std::unordered_map<std::string, uint32_t>& mm) {
    if (mm.find(ecls_name) == mm.end()) {
      mm[ecls_name] = ecls_counter++;
    }
    return mm[ecls_name];
  };

  // enodes
  tmp_ende_child_ptr.push_back(0);
  for (auto &[ename, node] : nodes.items()) {
    std::string op = node["op"];
    float cost = node["cost"];
    
    // eclass
    std::string ecls_name;
    if (node["eclass"].is_string()) {
      ecls_name = node["eclass"].get<std::string>();
    } else if (node["eclass"].is_number_integer()) {
      ecls_name = std::to_string(node["eclass"].get<int>());
    } else {
      throw std::runtime_error("Unsupported eclass type in JSON (must be string or int)");
    }
    uint32_t ecls_id = get_ecls_id(ecls_name, tmp_ecls_name_to_id);

    tmp_ende_name.push_back(ename);     // enode's id (e.g., "0__0")
    tmp_ende_op.push_back(op);          // operator string
    tmp_ende_cost.push_back(cost);      // cost
    tmp_ende_ecls.push_back(ecls_id);   // enode's eclass

    // children
    for (auto &ch : node["children"]) {
      if (ch.is_string()) {
        // === egg / extraction-gym format ===
        std::string child_enode = ch.get<std::string>();
        std::string child_ecls = nodes[child_enode]["eclass"];
        uint32_t child_ecls_id = get_ecls_id(child_ecls, tmp_ecls_name_to_id);
        tmp_ende_child.push_back(child_ecls_id);
      } else if (ch.is_number_integer()) {
        // === eboost / BoolE format ===
        std::string child_ecls = std::to_string(ch.get<int>());
        uint32_t child_ecls_id = get_ecls_id(child_ecls, tmp_ecls_name_to_id);
        tmp_ende_child.push_back(child_ecls_id);
      } else {
        throw std::runtime_error("Unsupported child type in JSON (must be string or int; format please ref github: extraction_gym)");
      }
    }

    tmp_ende_child_ptr.push_back(tmp_ende_child.size());
  }

  tmp_num_ecls = ecls_counter; // get number of eclasses

  // roots
  for (auto &r : j["root_eclasses"]) {
    std::string ecls_name;
    if (r.is_string()) {
      ecls_name = r.get<std::string>();
    } else if (r.is_number_integer()) {
      ecls_name = std::to_string(r.get<int>());
    } else {
      throw std::runtime_error("Unsupported root_eclass type (must be string or int)");
    }
    tmp_root_eclass_id.push_back(get_ecls_id(ecls_name, tmp_ecls_name_to_id));
  }

  MY_CHECK(tmp_ende_name.size() == tmp_num_ende, "!= num_ende 1");
  MY_CHECK(tmp_ende_op.size() == tmp_num_ende, "!= num_ende 2");
  MY_CHECK(tmp_ende_cost.size() == tmp_num_ende, "!= num_ende 3");
  MY_CHECK(tmp_ende_ecls.size() == tmp_num_ende, "!= num_ende 4");

#ifdef PRINT_LOG
  _print_read_in_data(true,
    tmp_num_ecls,
    tmp_num_ende,
    tmp_ende_name,
    tmp_ende_op,
    tmp_ende_cost,
    tmp_ecls_name_to_id,
    tmp_ende_ecls,
    tmp_ende_child,
    tmp_ende_child_ptr,
    tmp_root_eclass_id); 
  printf("\n"); 
  printf("Init #eclass = %u, #enodes = %u\n", tmp_num_ecls, tmp_num_ende);
#endif   
  _prune_dead_nodes(tmp_num_ecls,
    tmp_num_ende,
    tmp_ende_name,
    tmp_ende_op,
    tmp_ende_cost,
    tmp_ende_ecls,
    tmp_ende_child,
    tmp_ende_child_ptr,
    tmp_root_eclass_id);
}

void XEG::_prune_dead_nodes(
  const uint32_t tmp_num_ecls,
  const uint32_t tmp_num_ende,
  const std::vector<std::string>& tmp_ende_name, 
  const std::vector<std::string>& tmp_ende_op, 
  const std::vector<float>& tmp_ende_cost, 
  const std::vector<uint32_t>& tmp_ende_ecls,
  const std::vector<uint32_t>& tmp_ende_child, 
  const std::vector<uint32_t>& tmp_ende_child_ptr, 
  const std::vector<uint32_t>& tmp_root_eclass_id) 
{
#ifdef PRINT_LOG
  printf("XEG::_prune_dead_nodes()\n");
#endif

  // Prune dead nodes 
  std::vector<bool> alive_ecls;
  alive_ecls.resize(tmp_num_ecls, false); // init
  for (uint32_t ende_idx = 0; ende_idx < tmp_num_ende; ++ende_idx) {
    uint32_t start = tmp_ende_child_ptr[ende_idx];
    uint32_t end   = tmp_ende_child_ptr[ende_idx+1];
    // uint32_t ecls = tmp_ende_ecls[ende_idx];
    for (uint32_t j = start; j < end; j++) {
      alive_ecls[tmp_ende_child[j]] = true;
    }
  }
  for (uint32_t r : tmp_root_eclass_id) {
    alive_ecls[r] = true;
  }

  // Reconstruct all things 
  std::vector<uint32_t> new_ecls_idx_mapping; 
  new_ecls_idx_mapping.resize(tmp_num_ecls, UINT32_MAX); 
  std::unordered_set<uint32_t> dead_ecls_idx; 
  _num_ecls = 0;
  for (uint32_t ecls = 0; ecls < alive_ecls.size(); ecls++) {
    if (alive_ecls[ecls]) {
      new_ecls_idx_mapping[ecls] = _num_ecls; 
      _num_ecls++;
    } else {
      dead_ecls_idx.insert(ecls);
    }
  }

  _num_ende = 0;
  std::vector<uint32_t> new_ende_idx_mapping; 
  new_ende_idx_mapping.resize(tmp_num_ende, UINT32_MAX); 
  for (uint32_t ende = 0; ende < tmp_num_ende; ende++) {
    const uint32_t which_ecls = tmp_ende_ecls[ende];
    if (alive_ecls[which_ecls]) {
      new_ende_idx_mapping[ende] = _num_ende;
      _num_ende++;
    } 
  }

  _ende_name.resize(_num_ende);
  _ende_op.resize(_num_ende);
  _ende_cost.resize(_num_ende, UINT32_MAX);
  _ende_ecls.resize(_num_ende, UINT32_MAX);
  _ecls_its_ende.resize(_num_ecls, std::vector<uint32_t>());
  _ende_child_ptr.push_back(0);

  uint32_t acc = 0;
  for (uint32_t old_ende = 0; old_ende < tmp_num_ende; old_ende++) {
    const uint32_t old_ecls = tmp_ende_ecls[old_ende];
    if (alive_ecls[old_ecls]) {
      #ifdef PRINT_LOG
        std::cout << "  enode[" << old_ende << "]: "
          << "name=" << tmp_ende_name[old_ende] 
          << ", op=" << tmp_ende_op[old_ende] 
          << ", cost=" << tmp_ende_cost[old_ende]
          << ", eclass=" << tmp_ende_ecls[old_ende]
          << "\n";
      #endif
      uint32_t new_ende_idx = new_ende_idx_mapping[old_ende];
      _ende_name[new_ende_idx] = tmp_ende_name[old_ende];
      _ende_op[new_ende_idx] = tmp_ende_op[old_ende];
      _ende_cost[new_ende_idx] = tmp_ende_cost[old_ende];

      uint32_t new_ecls_idx = new_ecls_idx_mapping[old_ecls];
      _ende_ecls[new_ende_idx] = new_ecls_idx;
      _ecls_its_ende[new_ecls_idx].push_back(new_ende_idx);

      uint32_t start = tmp_ende_child_ptr[old_ende];
      uint32_t end   = tmp_ende_child_ptr[old_ende+1];
      for (uint32_t jj = start; jj < end; jj++) {
        uint32_t old_chd_ecls = tmp_ende_child[jj]; 
        if (alive_ecls[old_chd_ecls]) {
          uint32_t new_chd_ecls = new_ecls_idx_mapping[old_chd_ecls];
          _ende_child.push_back(new_chd_ecls);
          acc++;
        }
      }
      _ende_child_ptr.push_back(acc);
    }
  }

  for (uint32_t r = 0; r < tmp_root_eclass_id.size(); r++) {
    uint32_t new_r = new_ecls_idx_mapping[tmp_root_eclass_id[r]];
    _root_eclass_id.push_back(new_r); 
  }

  MY_CHECK((_num_ecls+dead_ecls_idx.size()) == tmp_num_ecls, "_num_ecls+dead_ecls_idx.size()");
  MY_CHECK(_root_eclass_id.size() == tmp_root_eclass_id.size(), "_root_eclass_id.size()");
  
  // Compress _ecls_its_ende to 1D 
  uint32_t acc_ptr = 0;
  _ecls_its_ende_1D_ptr.push_back(acc_ptr);
  for (uint32_t ecls = 0; ecls < _ecls_its_ende.size(); ecls++) {
    for (uint32_t ende : _ecls_its_ende[ecls]) {
      _ecls_its_ende_1D.push_back(ende);
      acc_ptr++;
    }
    _ecls_its_ende_1D_ptr.push_back(acc_ptr);
  }

  for (uint32_t r = 0; r < _root_eclass_id.size(); r++) {
    uint32_t ecls = _root_eclass_id[r];
    for (uint32_t ende : _ecls_its_ende[ecls]) {
      _all_root_ende_1D.push_back(ende);
    }
  }


#ifdef PRINT_LOG  
  for (const auto& elem : dead_ecls_idx) { std::cout << "elem: " << elem << "\n"; }
  
  for (uint32_t i = 0; i < new_ecls_idx_mapping.size(); i++) {
    printf("ori eclsIdx = %u, new_ecls_idx = %u\n", i, new_ecls_idx_mapping[i]);
  } printf("\n"); 
  
  for (uint32_t i = 0; i < new_ende_idx_mapping.size(); i++) {
    printf("ori endeIdx = %u, new_ende_idx = %u\n", i, new_ende_idx_mapping[i]);
  } printf("\n"); 

  printf("_num_ende = %u, _num_ecls = %u\n", _num_ende, _num_ecls);

  std::unordered_map<std::string, uint32_t> _ecls_name_to_id;
  _print_read_in_data(false, 
    _num_ecls,
    _num_ende,
    _ende_name,
    _ende_op,
    _ende_cost,
    _ecls_name_to_id,
    _ende_ecls,
    _ende_child,
    _ende_child_ptr,
    _root_eclass_id);   
#endif 
  
  // Step 1: gather raw (possibly duplicated) adjacency per e-node
  std::vector<std::vector<uint32_t>> raw_adj(_num_ende);
  std::vector<std::vector<uint32_t>> raw_inv_adj(_num_ende);

  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    uint32_t start = _ende_child_ptr[ende];
    uint32_t end   = _ende_child_ptr[ende+1];
    for (uint32_t jj = start; jj < end; jj++) {
      uint32_t chd_ecls = _ende_child[jj];
      for (uint32_t ende2 : _ecls_its_ende[chd_ecls]) {
        // remove self-loop
        if (ende != ende2) {
          raw_adj[ende].push_back(ende2);
          raw_inv_adj[ende2].push_back(ende);
        }
      }
    }
  }

  // Step 2: sort + unique (dedupe) each list, then compress directly to 1D CSR
  _global_adj_1D_ptr.push_back(0);
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    std::sort(raw_adj[ende].begin(), raw_adj[ende].end());
    raw_adj[ende].erase(std::unique(raw_adj[ende].begin(), raw_adj[ende].end()), raw_adj[ende].end());
    _global_adj_1D.insert(_global_adj_1D.end(), raw_adj[ende].begin(), raw_adj[ende].end());
    _global_adj_1D_ptr.push_back(_global_adj_1D.size());
  }

  _global_inv_adj_1D_ptr.push_back(0);
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    std::sort(raw_inv_adj[ende].begin(), raw_inv_adj[ende].end());
    raw_inv_adj[ende].erase(std::unique(raw_inv_adj[ende].begin(), raw_inv_adj[ende].end()), raw_inv_adj[ende].end());
    _global_inv_adj_1D.insert(_global_inv_adj_1D.end(), raw_inv_adj[ende].begin(), raw_inv_adj[ende].end());
    _global_inv_adj_1D_ptr.push_back(_global_inv_adj_1D.size());
  }

  // Init size of _picked_adj and _picked_inv_adj
  _picked_adj.resize(_global_adj_1D.size(), false); 
  _picked_inv_adj.resize(_global_inv_adj_1D.size(), false);
  _tmp_picked_adj.resize(_global_adj_1D.size(), false); 
  _tmp_picked_inv_adj.resize(_global_inv_adj_1D.size(), false);  

#ifdef PRINT_LOG
  printf("_global_adj_1D mapping (ende Id -> ende Id):\n");
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    uint32_t start = _global_adj_1D_ptr[ende];
    uint32_t end   = _global_adj_1D_ptr[ende + 1];
    printf("fm_node = %u, to nodes = [", ende);
    for (uint32_t k = start; k < end; k++) {
      printf("%u, ", _global_adj_1D[k]);
    }
    printf("]\n");
  }
  printf("_global_inv_adj_1D mapping (ende Id -> ende Id):\n");
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    uint32_t start = _global_inv_adj_1D_ptr[ende];
    uint32_t end   = _global_inv_adj_1D_ptr[ende + 1];
    printf("fm_node = %u, to nodes = [", ende);
    for (uint32_t k = start; k < end; k++) {
      printf("%u, ", _global_inv_adj_1D[k]);
    }
    printf("]\n");
  }

  printf("_num_ende = %u\n", _num_ende);
  printf("\n");  
#endif

  for (uint32_t r_ecls : _root_eclass_id) {
    for (uint32_t r : _ecls_its_ende[r_ecls]) {
      _roots.push_back(r); 
    }
  }

  // JUST USED FOR CHECK
  MY_CHECK(_all_root_ende_1D.size() == _roots.size(), "_all_root_ende_1D.size() == _roots.size()");
  for (uint32_t i = 0; i < _all_root_ende_1D.size(); i++) {
    if (_all_root_ende_1D[i] != _roots[i]) {
      exit(1);
    }
  }

  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    uint32_t out_degree = _global_adj_1D_ptr[ende+1] - _global_adj_1D_ptr[ende];
    if (out_degree == 0) {
      _leaves.push_back(ende); 
    }
  }

// // 0813 check

//   // ============================================================
//   // BIBI DEBUG: check whether the pruned e-graph contains
//   // a zero-local-cost cycle.
//   //
//   // Graph node:
//   //   e-class
//   //
//   // Directed edge A -> B:
//   //   there exists a zero-cost e-node in e-class A whose
//   //   child contains e-class B.
//   // ============================================================

//   uint32_t bibi_check_zero_cost_enode = 0u;
//   uint32_t bibi_check_neg_cost_enode = 0u;

//   // (child e-class, e-node producing this dependency)
//   std::vector<
//     std::vector<std::pair<uint32_t, uint32_t>>>
//     zero_cost_adj(_num_ecls);

//   for (uint32_t ende = 0u;
//        ende < _num_ende;
//        ++ende) {

//     if (_ende_cost[ende] < 0.0f) {
//       ++bibi_check_neg_cost_enode;
//     }

//     if (_ende_cost[ende] != 0.0f) {
//       continue;
//     }

//     ++bibi_check_zero_cost_enode;

//     const uint32_t src_ecls =
//       _ende_ecls[ende];

//     const uint32_t start =
//       _ende_child_ptr[ende];

//     const uint32_t end =
//       _ende_child_ptr[ende + 1u];

//     for (uint32_t pos = start;
//          pos < end;
//          ++pos) {

//       const uint32_t dst_ecls =
//         _ende_child[pos];

//       zero_cost_adj[src_ecls].push_back(
//         {dst_ecls, ende});
//     }
//   }

//   // DFS color:
//   //   0 = unvisited
//   //   1 = currently visiting
//   //   2 = finished
//   std::vector<uint8_t>
//     color(_num_ecls, 0u);

//   std::vector<uint32_t>
//     parent_ecls(
//       _num_ecls,
//       UINT32_MAX);

//   // e-node used for parent_ecls[v] -> v
//   std::vector<uint32_t>
//     parent_ende(
//       _num_ecls,
//       UINT32_MAX);

//   bool has_zero_cost_cycle = false;

//   uint32_t cycle_start =
//     UINT32_MAX;

//   uint32_t cycle_end =
//     UINT32_MAX;

//   uint32_t cycle_back_ende =
//     UINT32_MAX;

//   auto dfs =
//     [&](auto&& self,
//         const uint32_t u) -> bool {

//       color[u] = 1u;

//       for (const auto& edge :
//            zero_cost_adj[u]) {

//         const uint32_t v =
//           edge.first;

//         const uint32_t ende =
//           edge.second;

//         if (color[v] == 0u) {
//           parent_ecls[v] = u;
//           parent_ende[v] = ende;

//           if (self(self, v)) {
//             return true;
//           }
//         }
//         else if (color[v] == 1u) {
//           // Found:
//           //
//           //   v -> ... -> u -> v
//           cycle_start = v;
//           cycle_end = u;
//           cycle_back_ende = ende;

//           return true;
//         }
//       }

//       color[u] = 2u;

//       return false;
//     };

//   for (uint32_t ecls = 0u;
//        ecls < _num_ecls;
//        ++ecls) {

//     if (color[ecls] != 0u) {
//       continue;
//     }

//     if (dfs(dfs, ecls)) {
//       has_zero_cost_cycle = true;
//       break;
//     }
//   }

//   printf(
//     "bibi_check_zero_cost_enode = %u, "
//     "bibi_check_neg_cost_enode = %u\n",
//     bibi_check_zero_cost_enode,
//     bibi_check_neg_cost_enode);

//   printf(
//     "bibi_check_zero_cost_cycle = %s\n",
//     has_zero_cost_cycle
//       ? "YES"
//       : "NO");

//   // Print one concrete cycle for debugging.
//   if (has_zero_cost_cycle) {

//     std::vector<uint32_t>
//       cycle_eclasses;

//     uint32_t cur =
//       cycle_end;

//     cycle_eclasses.push_back(cur);

//     while (cur != cycle_start) {
//       cur = parent_ecls[cur];

//       MY_CHECK(
//         cur != UINT32_MAX,
//         "broken zero-cost cycle parent chain");

//       cycle_eclasses.push_back(cur);
//     }

//     std::reverse(
//       cycle_eclasses.begin(),
//       cycle_eclasses.end());

//     printf(
//       "ZERO-COST CYCLE FOUND:\n");

//     // Tree edges on:
//     //
//     // cycle_start -> ... -> cycle_end
//     for (size_t i = 0;
//          i + 1u < cycle_eclasses.size();
//          ++i) {

//       const uint32_t src =
//         cycle_eclasses[i];

//       const uint32_t dst =
//         cycle_eclasses[i + 1u];

//       const uint32_t ende =
//         parent_ende[dst];

//       printf(
//         "  eclass %u -- enode %u (%s, cost=%.6f) --> eclass %u\n",
//         src,
//         ende,
//         _ende_name[ende].c_str(),
//         _ende_cost[ende],
//         dst);
//     }

//     // Closing edge:
//     //
//     // cycle_end -> cycle_start
//     printf(
//       "  eclass %u -- enode %u (%s, cost=%.6f) --> eclass %u\n",
//       cycle_end,
//       cycle_back_ende,
//       _ende_name[cycle_back_ende].c_str(),
//       _ende_cost[cycle_back_ende],
//       cycle_start);
//   }

//   exit(1);

// // 0813 check 

}

void XEG::_rebuild_derived_graph_data_for_output_degree_exp()
{
  if (_ende_child_ptr.size() !=
      static_cast<size_t>(_num_ende) + 1u) {
    throw std::logic_error(
      "Output-degree experiment found an invalid "
      "e-node child-pointer array");
  }

  if (_ecls_its_ende.size() !=
      static_cast<size_t>(_num_ecls)) {
    throw std::logic_error(
      "Output-degree experiment found invalid "
      "e-class membership data");
  }

  // Re-expand:
  //
  //   e-node -> child e-class
  //
  // into:
  //
  //   e-node -> every e-node in the child e-class
  //
  // This intentionally matches the existing production
  // construction in _prune_dead_nodes().
  std::vector<std::vector<uint32_t>>
    raw_adj(_num_ende);

  std::vector<std::vector<uint32_t>>
    raw_inv_adj(_num_ende);

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    const uint32_t child_begin =
      _ende_child_ptr[ende];

    const uint32_t child_end =
      _ende_child_ptr[ende + 1u];

    for (uint32_t position = child_begin;
         position < child_end;
         ++position) {
      const uint32_t child_eclass =
        _ende_child[position];

      if (child_eclass >= _num_ecls) {
        throw std::logic_error(
          "Output-degree experiment generated an "
          "out-of-range child e-class");
      }

      for (uint32_t child_ende :
           _ecls_its_ende[child_eclass]) {
        // Preserve the production self-loop filtering
        // behavior for the expanded e-node graph.
        if (ende == child_ende) {
          continue;
        }

        raw_adj[ende].push_back(
          child_ende);

        raw_inv_adj[child_ende].push_back(
          ende);
      }
    }
  }

  // Remove every CSR array built by the original read before
  // constructing the augmented versions.
  _global_adj.clear();
  _global_inv_adj.clear();

  _global_adj_1D.clear();
  _global_adj_1D_ptr.clear();

  _global_inv_adj_1D.clear();
  _global_inv_adj_1D_ptr.clear();

  _global_adj_1D_ptr.reserve(
    static_cast<size_t>(_num_ende) + 1u);

  _global_inv_adj_1D_ptr.reserve(
    static_cast<size_t>(_num_ende) + 1u);

  // Forward CSR.
  _global_adj_1D_ptr.push_back(0u);

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    auto& neighbors =
      raw_adj[ende];

    std::sort(
      neighbors.begin(),
      neighbors.end());

    neighbors.erase(
      std::unique(
        neighbors.begin(),
        neighbors.end()),
      neighbors.end());

    _global_adj_1D.insert(
      _global_adj_1D.end(),
      neighbors.begin(),
      neighbors.end());

    if (_global_adj_1D.size() >
        std::numeric_limits<uint32_t>::max()) {
      throw std::overflow_error(
        "Output-degree forward CSR exceeds uint32_t");
    }

    _global_adj_1D_ptr.push_back(
      static_cast<uint32_t>(
        _global_adj_1D.size()));
  }

  // Inverse CSR.
  _global_inv_adj_1D_ptr.push_back(0u);

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    auto& neighbors =
      raw_inv_adj[ende];

    std::sort(
      neighbors.begin(),
      neighbors.end());

    neighbors.erase(
      std::unique(
        neighbors.begin(),
        neighbors.end()),
      neighbors.end());

    _global_inv_adj_1D.insert(
      _global_inv_adj_1D.end(),
      neighbors.begin(),
      neighbors.end());

    if (_global_inv_adj_1D.size() >
        std::numeric_limits<uint32_t>::max()) {
      throw std::overflow_error(
        "Output-degree inverse CSR exceeds uint32_t");
    }

    _global_inv_adj_1D_ptr.push_back(
      static_cast<uint32_t>(
        _global_inv_adj_1D.size()));
  }

  // The edge masks are pointer-indexed, so their sizes must
  // match the newly constructed CSR arrays.
  _picked_adj.assign(
    _global_adj_1D.size(),
    false);

  _picked_inv_adj.assign(
    _global_inv_adj_1D.size(),
    false);

  _tmp_picked_adj.assign(
    _global_adj_1D.size(),
    false);

  _tmp_picked_inv_adj.assign(
    _global_inv_adj_1D.size(),
    false);

  // These arrays are reconstructed by tree/DAG construction,
  // but reset them here so no original-read state remains.
  _indegree.assign(
    _num_ende,
    0u);

  _inv_indegree.assign(
    _num_ende,
    0u);

  _new_inserted.assign(
    _num_ende,
    false);

  _in_subgraph.assign(
    _num_ende,
    false);

  _ecls_its_ende_newG.clear();
  _ecls_its_ende_newG_GPU.clear();

  _max_level =
    -1;

  // Recompute global leaf e-nodes from the augmented expanded
  // graph.
  //
  // Because only original non-leaf e-nodes are modified, the
  // leaf set should normally remain unchanged. Recomputing it
  // avoids relying on that assumption.
  _leaves.clear();

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    const uint32_t out_degree =
      _global_adj_1D_ptr[ende + 1u] -
      _global_adj_1D_ptr[ende];

    if (out_degree == 0u) {
      _leaves.push_back(
        ende);
    }
  }

  // The following structures remain unchanged because the
  // experiment does not add/remove e-nodes or e-classes:
  //
  //   _ende_name
  //   _ende_op
  //   _ende_cost
  //   _ende_ecls
  //   _ecls_its_ende
  //   _ecls_its_ende_1D
  //   _ecls_its_ende_1D_ptr
  //   _root_eclass_id
  //   _all_root_ende_1D
  //   _roots
}

void XEG::_augment_output_degree_for_exp(
  const uint32_t output_degree_multiplier,
  const uint32_t random_seed)
{
  if (output_degree_multiplier == 0u) {
    throw std::invalid_argument(
      "Output-degree multiplier must be at least 1");
  }

  if (_ende_child_ptr.size() !=
      static_cast<size_t>(_num_ende) + 1u) {
    throw std::logic_error(
      "Output-degree experiment requires initialized "
      "e-node child CSR");
  }

  if (_ende_ecls.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::logic_error(
      "Output-degree experiment requires initialized "
      "e-node-to-e-class mapping");
  }

  if (_ecls_its_ende.size() !=
      static_cast<size_t>(_num_ecls)) {
    throw std::logic_error(
      "Output-degree experiment requires initialized "
      "e-class membership");
  }

  const uint64_t original_child_reference_count =
    static_cast<uint64_t>(
      _ende_child.size());

  if (original_child_reference_count >
      std::numeric_limits<uint64_t>::max() /
        output_degree_multiplier) {
    throw std::overflow_error(
      "Requested output-degree multiplier overflows uint64_t");
  }

  const uint64_t target_child_reference_count =
    original_child_reference_count *
    static_cast<uint64_t>(
      output_degree_multiplier);

  // _ende_child_ptr stores uint32_t positions, so the final
  // number of child references must remain representable.
  if (target_child_reference_count >
      std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(
      "Requested output-degree graph exceeds uint32_t "
      "child-pointer capacity");
  }

  const uint64_t original_expanded_edge_count =
    static_cast<uint64_t>(
      _global_adj_1D.size());

  uint32_t original_max_output_degree =
    0u;

  // Sources are fixed from the original post-pruning graph.
  //
  // A source must already be a non-leaf e-node before any
  // artificial edge is added.
  std::vector<uint32_t>
    non_leaf_enodes;

  non_leaf_enodes.reserve(
    _num_ende);

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    const uint32_t degree =
      _ende_child_ptr[ende + 1u] -
      _ende_child_ptr[ende];

    if (degree >
        original_max_output_degree) {
      original_max_output_degree =
        degree;
    }

    if (degree > 0u) {
      non_leaf_enodes.push_back(
        ende);
    }
  }

  // Use the strict leaf-e-class definition:
  //
  //   the e-class is non-empty
  //   every e-node inside it has zero child e-classes
  //
  // Because these target e-nodes are never eligible sources,
  // adding source -> strict-leaf-e-class edges cannot create a
  // path back to the source.
  std::vector<uint32_t>
    strict_leaf_eclasses;

  strict_leaf_eclasses.reserve(
    _num_ecls);

  for (uint32_t eclass = 0u;
       eclass < _num_ecls;
       ++eclass) {
    const auto& member_enodes =
      _ecls_its_ende[eclass];

    // Never create a dependency on an empty e-class.
    if (member_enodes.empty()) {
      continue;
    }

    bool every_member_is_leaf =
      true;

    for (uint32_t ende :
         member_enodes) {
      const bool is_leaf =
        _ende_child_ptr[ende] ==
        _ende_child_ptr[ende + 1u];

      if (!is_leaf) {
        every_member_is_leaf =
          false;

        break;
      }
    }

    if (every_member_is_leaf) {
      strict_leaf_eclasses.push_back(
        eclass);
    }
  }

  // Store only newly added children.
  //
  // Original child ordering and duplicate operands must remain
  // untouched. For example, an original e-node with children
  // [C, C] may represent two operands and must not be changed
  // into [C].
  struct AddedChildData
  {
    std::vector<uint32_t> ordered_targets;
    std::unordered_set<uint32_t> target_lookup;
  };

  std::unordered_map<uint32_t, AddedChildData>
    added_children_by_source;

  uint64_t current_child_reference_count =
    original_child_reference_count;

  uint64_t successfully_added_edges =
    0u;

  uint64_t random_attempts =
    0u;

  uint32_t consecutive_failures =
    0u;

  uint64_t rejected_same_eclass =
    0u;

  uint64_t rejected_original_child =
    0u;

  uint64_t rejected_added_child =
    0u;

  // Deterministic checksum of the successfully added edge
  // sequence. Runs with the same input, multiplier, and seed
  // should print the same checksum regardless of DAG kernel.
  uint64_t augmentation_checksum =
    1469598103934665603ull;

  std::mt19937 random_generator(
    random_seed);

  if (current_child_reference_count <
        target_child_reference_count &&
      !non_leaf_enodes.empty() &&
      !strict_leaf_eclasses.empty()) {
    std::uniform_int_distribution<size_t>
      choose_source(
        0u,
        non_leaf_enodes.size() - 1u);

    std::uniform_int_distribution<size_t>
      choose_target(
        0u,
        strict_leaf_eclasses.size() - 1u);

    while (current_child_reference_count <
             target_child_reference_count &&
           consecutive_failures <
             kOutputDegreeExpMaxConsecutiveFailures) {
      ++random_attempts;

      const uint32_t source_ende =
        non_leaf_enodes[
          choose_source(
            random_generator)];

      const uint32_t target_eclass =
        strict_leaf_eclasses[
          choose_target(
            random_generator)];

      bool valid_pair =
        true;

      // Defensive direct self-loop prevention:
      //
      // source e-node u belongs to e-class C
      // adding u -> C would make u depend on its own e-class.
      if (_ende_ecls[source_ende] ==
          target_eclass) {
        valid_pair =
          false;

        ++rejected_same_eclass;
      }

      // Do not add a target e-class that already appears in the
      // source's original child list.
      if (valid_pair) {
        const uint32_t child_begin =
          _ende_child_ptr[source_ende];

        const uint32_t child_end =
          _ende_child_ptr[source_ende + 1u];

        for (uint32_t position = child_begin;
             position < child_end;
             ++position) {
          if (_ende_child[position] ==
              target_eclass) {
            valid_pair =
              false;

            ++rejected_original_child;

            break;
          }
        }
      }

      // Do not add the same artificial source-target edge more
      // than once.
      if (valid_pair) {
        const auto source_it =
          added_children_by_source.find(
            source_ende);

        if (source_it !=
              added_children_by_source.end() &&
            source_it->second.target_lookup.find(
              target_eclass) !=
              source_it->second.target_lookup.end()) {
          valid_pair =
            false;

          ++rejected_added_child;
        }
      }

      if (!valid_pair) {
        ++consecutive_failures;

        continue;
      }

      auto& added_data =
        added_children_by_source[
          source_ende];

      added_data.ordered_targets.push_back(
        target_eclass);

      added_data.target_lookup.insert(
        target_eclass);

      ++successfully_added_edges;
      ++current_child_reference_count;

      // A success proves that random sampling is still making
      // progress, so restart the consecutive-failure window.
      consecutive_failures =
        0u;

      const uint64_t edge_key =
        (
          static_cast<uint64_t>(
            source_ende) << 32u
        ) |
        static_cast<uint64_t>(
          target_eclass);

      augmentation_checksum ^=
        edge_key;

      augmentation_checksum *=
        1099511628211ull;
    }
  }

  const uint64_t final_child_reference_count =
    original_child_reference_count +
    successfully_added_edges;

  if (final_child_reference_count >
      std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(
      "Augmented child-reference array exceeds uint32_t");
  }

  // Reconstruct _ende_child and _ende_child_ptr while preserving:
  //
  //   1. every original child
  //   2. original child order
  //   3. original duplicate operands
  //   4. random insertion order of new target e-classes
  std::vector<uint32_t>
    augmented_ende_child;

  augmented_ende_child.reserve(
    static_cast<size_t>(
      final_child_reference_count));

  std::vector<uint32_t>
    augmented_ende_child_ptr;

  augmented_ende_child_ptr.reserve(
    static_cast<size_t>(_num_ende) + 1u);

  augmented_ende_child_ptr.push_back(
    0u);

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    const uint32_t original_begin =
      _ende_child_ptr[ende];

    const uint32_t original_end =
      _ende_child_ptr[ende + 1u];

    augmented_ende_child.insert(
      augmented_ende_child.end(),
      _ende_child.begin() + original_begin,
      _ende_child.begin() + original_end);

    const auto added_it =
      added_children_by_source.find(
        ende);

    if (added_it !=
        added_children_by_source.end()) {
      const auto& added_targets =
        added_it->second.ordered_targets;

      augmented_ende_child.insert(
        augmented_ende_child.end(),
        added_targets.begin(),
        added_targets.end());
    }

    if (augmented_ende_child.size() >
        std::numeric_limits<uint32_t>::max()) {
      throw std::overflow_error(
        "Augmented e-node child pointer exceeds uint32_t");
    }

    augmented_ende_child_ptr.push_back(
      static_cast<uint32_t>(
        augmented_ende_child.size()));
  }

  if (augmented_ende_child.size() !=
      static_cast<size_t>(
        final_child_reference_count)) {
    throw std::logic_error(
      "Augmented child-reference count is inconsistent");
  }

  _ende_child.swap(
    augmented_ende_child);

  _ende_child_ptr.swap(
    augmented_ende_child_ptr);

  // Reconstruct every data structure whose content or size
  // depends on e-node -> child-e-class references.
  //
  // This is executed even for multiplier=1, allowing a 1x
  // ablation run to validate the reconstruction path.
  _rebuild_derived_graph_data_for_output_degree_exp();

  uint32_t final_max_output_degree =
    0u;

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    const uint32_t degree =
      _ende_child_ptr[ende + 1u] -
      _ende_child_ptr[ende];

    if (degree >
        final_max_output_degree) {
      final_max_output_degree =
        degree;
    }
  }

  const double original_average_output_degree =
    _num_ende == 0u
      ? 0.0
      : static_cast<double>(
          original_child_reference_count) /
        static_cast<double>(
          _num_ende);

  const double final_average_output_degree =
    _num_ende == 0u
      ? 0.0
      : static_cast<double>(
          final_child_reference_count) /
        static_cast<double>(
          _num_ende);

  const double actual_multiplier =
    original_child_reference_count == 0u
      ? 0.0
      : static_cast<double>(
          final_child_reference_count) /
        static_cast<double>(
          original_child_reference_count);

  const bool target_reached =
    final_child_reference_count >=
    target_child_reference_count;

  const char* stop_reason =
    "target-reached";

  if (!target_reached) {
    if (non_leaf_enodes.empty()) {
      stop_reason =
        "no-non-leaf-source";
    }
    else if (strict_leaf_eclasses.empty()) {
      stop_reason =
        "no-strict-leaf-eclass";
    }
    else {
      stop_reason =
        "100-consecutive-failures";
    }
  }

  printf(
    "[Output-Degree Exp] "
    "seed=%u, "
    "requested_multiplier=%u, "
    "actual_multiplier=%.6f, "
    "stop_reason=%s\n",
    random_seed,
    output_degree_multiplier,
    actual_multiplier,
    stop_reason);

  printf(
    "[Output-Degree Exp] "
    "original_child_references=%llu, "
    "target_child_references=%llu, "
    "final_child_references=%llu, "
    "added=%llu\n",
    static_cast<unsigned long long>(
      original_child_reference_count),
    static_cast<unsigned long long>(
      target_child_reference_count),
    static_cast<unsigned long long>(
      final_child_reference_count),
    static_cast<unsigned long long>(
      successfully_added_edges));

  printf(
    "[Output-Degree Exp] "
    "non_leaf_sources=%zu, "
    "strict_leaf_eclasses=%zu, "
    "attempts=%llu, "
    "final_consecutive_failures=%u\n",
    non_leaf_enodes.size(),
    strict_leaf_eclasses.size(),
    static_cast<unsigned long long>(
      random_attempts),
    consecutive_failures);

  printf(
    "[Output-Degree Exp] "
    "rejected_same_eclass=%llu, "
    "rejected_original_child=%llu, "
    "rejected_added_child=%llu\n",
    static_cast<unsigned long long>(
      rejected_same_eclass),
    static_cast<unsigned long long>(
      rejected_original_child),
    static_cast<unsigned long long>(
      rejected_added_child));

  printf(
    "[Output-Degree Exp] "
    "original_avg_degree=%.6f, "
    "final_avg_degree=%.6f, "
    "original_max_degree=%u, "
    "final_max_degree=%u\n",
    original_average_output_degree,
    final_average_output_degree,
    original_max_output_degree,
    final_max_output_degree);

  printf(
    "[Output-Degree Exp] "
    "original_expanded_edges=%llu, "
    "final_expanded_edges=%zu, "
    "checksum=%llu\n",
    static_cast<unsigned long long>(
      original_expanded_edge_count),
    _global_adj_1D.size(),
    static_cast<unsigned long long>(
      augmentation_checksum));
}

std::string XEG::dump_output_degree_exp_json(
  const std::string& input_path,
  const uint32_t output_degree_multiplier)
{
  namespace fs =
    std::filesystem;

  if (_ende_name.empty() ||
      _num_ende == 0u) {
    throw std::logic_error(
      "dump_output_degree_exp_json requires "
      "an initialized non-empty graph");
  }

  if (output_degree_multiplier == 0u) {
    throw std::invalid_argument(
      "Output-degree multiplier must be at least 1");
  }

  const fs::path output_directory(
    kOutputDegreeExpDumpDirectory);

  std::error_code filesystem_error;

  // Create the complete directory hierarchy when it does not
  // already exist.
  fs::create_directories(
    output_directory,
    filesystem_error);

  if (filesystem_error) {
    throw std::runtime_error(
      std::string(
        "Failed to create output-degree dump directory: ") +
      output_directory.string() +
      " (" +
      filesystem_error.message() +
      ")");
  }

  filesystem_error.clear();

  const bool output_path_is_directory =
    fs::is_directory(
      output_directory,
      filesystem_error);

  if (filesystem_error) {
    throw std::runtime_error(
      std::string(
        "Failed to inspect output-degree dump directory: ") +
      output_directory.string() +
      " (" +
      filesystem_error.message() +
      ")");
  }

  if (!output_path_is_directory) {
    throw std::runtime_error(
      std::string(
        "Output-degree dump path exists but is not "
        "a directory: ") +
      output_directory.string());
  }

  const fs::path original_input_path(
    input_path);

  // For:
  //
  //   mul112.rename.json
  //
  // filename().stem() returns:
  //
  //   mul112.rename
  const std::string input_stem =
    original_input_path
      .filename()
      .stem()
      .string();

  if (input_stem.empty()) {
    throw std::invalid_argument(
      "Cannot derive an output filename from "
      "the input path");
  }

  for (uint32_t file_index = 1u;
       ;
       ++file_index) {
    const std::string output_filename =
      input_stem +
      "_sf" +
      std::to_string(
        output_degree_multiplier) +
      "_" +
      std::to_string(
        file_index) +
      ".json";

    const fs::path output_path =
      output_directory /
      output_filename;

    filesystem_error.clear();

    const bool output_already_exists =
      fs::exists(
        output_path,
        filesystem_error);

    if (filesystem_error) {
      throw std::runtime_error(
        std::string(
          "Failed to inspect candidate output file: ") +
        output_path.string() +
        " (" +
        filesystem_error.message() +
        ")");
    }

    if (!output_already_exists) {
      try {
        _dump_to_egg_json(
          output_path.string(),
          _num_ecls,
          _num_ende,
          _ende_name,
          _ende_op,
          _ende_cost,
          _ende_ecls,
          _ende_child,
          _ende_child_ptr,
          _root_eclass_id);
      }
      catch (...) {
        // Do not leave a partial JSON file behind if writing
        // fails after the file has been created.
        std::error_code remove_error;

        fs::remove(
          output_path,
          remove_error);

        throw;
      }

      return output_path.string();
    }

    if (file_index ==
        std::numeric_limits<uint32_t>::max()) {
      throw std::overflow_error(
        "Output-degree dump file index exceeded uint32_t");
    }
  }
}

void XEG::_dump_to_egg_json(
  const std::string& out_path,
  const uint32_t num_ecls,
  const uint32_t num_ende,
  const std::vector<std::string>& ende_name,
  const std::vector<std::string>& ende_op,
  const std::vector<float>& ende_cost,
  const std::vector<uint32_t>& ende_ecls,
  const std::vector<uint32_t>& ende_child,
  const std::vector<uint32_t>& ende_child_ptr,
  const std::vector<uint32_t>& root_eclass_id)
{
  if (num_ecls == 0u ||
      num_ende == 0u) {
    throw std::logic_error(
      "Cannot dump an empty e-graph");
  }

  if (ende_name.size() !=
        static_cast<size_t>(num_ende) ||
      ende_op.size() !=
        static_cast<size_t>(num_ende) ||
      ende_cost.size() !=
        static_cast<size_t>(num_ende) ||
      ende_ecls.size() !=
        static_cast<size_t>(num_ende)) {
    throw std::logic_error(
      "Cannot dump e-graph because e-node arrays "
      "have inconsistent sizes");
  }

  if (ende_child_ptr.size() !=
      static_cast<size_t>(num_ende) + 1u) {
    throw std::logic_error(
      "Cannot dump e-graph because the child-pointer "
      "array has an invalid size");
  }

  if (ende_child_ptr.front() != 0u) {
    throw std::logic_error(
      "Cannot dump e-graph because the first "
      "child pointer is not zero");
  }

  if (ende_child_ptr.back() !=
      ende_child.size()) {
    throw std::logic_error(
      "Cannot dump e-graph because the final "
      "child pointer does not match the child array");
  }

  // Validate every CSR range before using it.
  for (uint32_t ende = 0u;
       ende < num_ende;
       ++ende) {
    const uint32_t child_begin =
      ende_child_ptr[ende];

    const uint32_t child_end =
      ende_child_ptr[ende + 1u];

    if (child_begin > child_end ||
        child_end > ende_child.size()) {
      throw std::logic_error(
        "Cannot dump e-graph because an e-node has "
        "an invalid child range");
    }
  }

  // The in-memory representation stores:
  //
  //   e-node -> child e-class ID
  //
  // The string-child JSON format stores:
  //
  //   e-node -> one e-node name belonging to the child e-class
  //
  // Select the first e-node in every e-class as its canonical
  // JSON representative.
  std::vector<uint32_t>
    representative_ende_of_eclass(
      num_ecls,
      UINT32_MAX);

  for (uint32_t ende = 0u;
       ende < num_ende;
       ++ende) {
    const uint32_t eclass =
      ende_ecls[ende];

    if (eclass >= num_ecls) {
      throw std::logic_error(
        "Cannot dump e-graph because an e-node "
        "belongs to an out-of-range e-class");
    }

    if (representative_ende_of_eclass[eclass] ==
        UINT32_MAX) {
      representative_ende_of_eclass[eclass] =
        ende;
    }
  }

  nlohmann::json output_json;

  output_json["nodes"] =
    nlohmann::json::object();

  output_json["root_eclasses"] =
    nlohmann::json::array();

  for (uint32_t ende = 0u;
       ende < num_ende;
       ++ende) {
    const std::string& node_name =
      ende_name[ende];

    if (node_name.empty()) {
      throw std::logic_error(
        "Cannot dump an e-node with an empty name");
    }

    if (output_json["nodes"].contains(
          node_name)) {
      throw std::logic_error(
        std::string(
          "Cannot dump duplicate e-node name: ") +
        node_name);
    }

    nlohmann::json node_json;

    node_json["children"] =
      nlohmann::json::array();

    const uint32_t child_begin =
      ende_child_ptr[ende];

    const uint32_t child_end =
      ende_child_ptr[ende + 1u];

    for (uint32_t position = child_begin;
         position < child_end;
         ++position) {
      const uint32_t child_eclass =
        ende_child[position];

      if (child_eclass >= num_ecls) {
        throw std::logic_error(
          "Cannot dump e-graph because a child "
          "e-class is out of range");
      }

      const uint32_t representative_ende =
        representative_ende_of_eclass[
          child_eclass];

      if (representative_ende ==
          UINT32_MAX) {
        throw std::logic_error(
          "Cannot dump e-graph because a referenced "
          "child e-class has no e-node");
      }

      node_json["children"].push_back(
        ende_name[
          representative_ende]);
    }

    node_json["cost"] =
      ende_cost[ende];

    // The original external e-class names are not retained after
    // _read_input() and pruning. Use the current dense e-class
    // ID as the serialized e-class name.
    node_json["eclass"] =
      std::to_string(
        ende_ecls[ende]);

    node_json["op"] =
      ende_op[ende];

    output_json["nodes"][node_name] =
      std::move(
        node_json);
  }

  for (uint32_t root_eclass :
       root_eclass_id) {
    if (root_eclass >= num_ecls) {
      throw std::logic_error(
        "Cannot dump e-graph because a root "
        "e-class is out of range");
    }

    if (representative_ende_of_eclass[
          root_eclass] ==
        UINT32_MAX) {
      throw std::logic_error(
        "Cannot dump e-graph because a root "
        "e-class has no e-node");
    }

    output_json["root_eclasses"].push_back(
      std::to_string(
        root_eclass));
  }

  std::ofstream output_stream(
    out_path,
    std::ios::out |
    std::ios::trunc);

  if (!output_stream.is_open()) {
    throw std::runtime_error(
      std::string(
        "Failed to open output JSON file: ") +
      out_path);
  }

  output_stream
    << std::setw(2)
    << output_json
    << '\n';

  output_stream.flush();

  if (!output_stream.good()) {
    throw std::runtime_error(
      std::string(
        "Failed while writing output JSON file: ") +
      out_path);
  }
}

// =========================== TREE/DAG functions =========================== // 

xeg::CpuGraph XEG::_cpu_graph() const {
  return {_ende_cost, _ende_ecls, _ende_child, _ende_child_ptr,
          _ecls_its_ende, _root_eclass_id,
          _global_adj_1D, _global_adj_1D_ptr,
          _global_inv_adj_1D, _global_inv_adj_1D_ptr};
}

std::pair<double, double> XEG::_cpu_tree_construction(
  const std::vector<float>& costs, std::vector<double>& class_cost,
  std::vector<double>& node_cost, std::vector<uint32_t>& best,
  std::vector<bool>& selected)
{
  const auto graph = _cpu_graph();
  xeg::CpuGraph view{costs, graph.node_class, graph.children, graph.child_ptr,
                    graph.members, graph.roots, graph.forward, graph.forward_ptr,
                    graph.inverse, graph.inverse_ptr};
  auto result = xeg::cpu_tree(view, 1);
  class_cost = std::move(result.class_cost);
  node_cost = std::move(result.node_cost);
  best = std::move(result.best);
  selected.assign(result.selected.begin(), result.selected.end());
  return {result.tree_cost, result.dag_cost};
}

void XEG::_cpu_DAG_construction(std::vector<bool>& selected) {
  auto result = xeg::cpu_dag(_cpu_graph(),
      std::vector<uint8_t>(selected.begin(), selected.end()), 1);
  selected.assign(result.selected.begin(), result.selected.end());
  _in_subgraph.assign(result.in_subgraph.begin(), result.in_subgraph.end());
  _ecls_its_ende_newG.assign(_num_ecls, {});
  for (uint32_t c = 0; c < _num_ecls; ++c)
    for (uint32_t u : _ecls_its_ende[c])
      if (result.in_subgraph[u]) _ecls_its_ende_newG[c].push_back(u);
  std::cout << "CPU DAG iterations = " << result.iterations << '\n';
}

nlohmann::json XEG::run_cpu(int threads, double time_limit, int solver_workers, double deterministic_limit) {
  if (threads < 1 || solver_workers < 1 || !std::isfinite(time_limit) || time_limit < 0)
    throw std::invalid_argument("Invalid CPU threads, solver workers or time limit");
  if (!std::isfinite(deterministic_limit) || deterministic_limit < 0 ||
      (deterministic_limit > 0 && (solver_workers != 1 || time_limit == 0)))
    throw std::invalid_argument("Deterministic solving requires one solver worker and positive limits");
  if (_ende_cost.empty()) throw std::invalid_argument("CPU extraction requires a nonempty graph");
  using Clock = std::chrono::steady_clock;
  auto graph = _cpu_graph();
  auto start = Clock::now();
  auto tree = xeg::cpu_tree(graph, threads);
  auto tree_end = Clock::now();
  auto dag = xeg::cpu_dag(graph, tree.selected, threads);
  auto dag_end = Clock::now();
  // Validation is outside the timed construction regions.
  xeg::validate_cpu_solution(graph, tree.selected);
  xeg::validate_cpu_dag(graph, dag.selected);
  xeg::validate_cpu_candidates(graph, dag.in_subgraph);
  for (uint32_t u = 0; u < _num_ende; ++u)
    if (tree.selected[u] && !dag.in_subgraph[u])
      throw std::runtime_error("CPU DAG filtering removed a tree node");
  const auto validation_end = Clock::now();
  auto hash = [](const auto& values) {
    uint64_t result = 14695981039346656037ull;
    for (auto value : values) { result ^= static_cast<uint64_t>(value); result *= 1099511628211ull; }
    return std::to_string(result);
  };
  const double tree_ms = std::chrono::duration<double, std::milli>(tree_end-start).count();
  const double dag_ms = std::chrono::duration<double, std::milli>(dag_end-tree_end).count();
  nlohmann::json report = {
    {"threads",threads}, {"solver_workers",solver_workers}, {"time_limit_s",time_limit},
    {"deterministic_limit",deterministic_limit},
    {"tree_ms",tree_ms}, {"dag_ms",dag_ms}, {"tree_cost",tree.tree_cost},
    {"validation_ms",std::chrono::duration<double,std::milli>(validation_end-dag_end).count()},
    {"tree_dag_cost",tree.dag_cost}, {"tree_iterations",tree.iterations},
    {"dag_iterations",dag.iterations}, {"proposals",dag.proposals},
    {"neighbor_updates",dag.neighbor_updates}, {"reverted",dag.reverted},
    {"invalid_candidates_removed",dag.invalid_candidates_removed},
    {"tree_hash",hash(tree.selected)}, {"best_hash",hash(tree.best)},
    {"dag_hash",hash(dag.selected)}, {"filtered_hash",hash(dag.in_subgraph)},
    {"selected",std::count(dag.selected.begin(),dag.selected.end(),uint8_t{1})},
    {"filtered",std::count(dag.in_subgraph.begin(),dag.in_subgraph.end(),uint8_t{1})},
    {"validated",true}
  };
  std::cout << "CPU tree construction runtime: " << tree_ms << " (ms)\n"
            << "CPU DAG construction runtime: " << dag_ms << " (ms)\n";
  if (time_limit > 0) {
    report.update(_solve_checked_candidates(time_limit, dag.in_subgraph, tree.selected,
      tree.best, std::vector<bool>(tree.selected.begin(), tree.selected.end()),
      solver_workers, deterministic_limit, true));
  }
  report["cpu_pipeline_ms"] = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
  return report;
}

void XEG::_print_adj(
  const std::vector<std::unordered_set<uint32_t>>& adj) const 
{
  for (uint32_t fm = 0; fm < adj.size(); fm++) {
    printf("fm_node = %u, to nodes = [", fm); 
    for (const auto& elem : adj[fm]) {
      printf("%u, ", elem);
    }
    printf("]\n");
  }
}

void XEG::_print_read_in_data(
  const bool init,
  const uint32_t tmp_num_ecls,
  const uint32_t tmp_num_ende,
  const std::vector<std::string>& tmp_ende_name, 
  const std::vector<std::string>& tmp_ende_op, 
  const std::vector<float>& tmp_ende_cost, 
  const std::unordered_map<std::string, uint32_t>& tmp_ecls_name_to_id,
  const std::vector<uint32_t>& tmp_ende_ecls,
  const std::vector<uint32_t>& tmp_ende_child, 
  const std::vector<uint32_t>& tmp_ende_child_ptr, 
  const std::vector<uint32_t>& tmp_root_eclass_id) const 
{
  std::cout << "\n========== XEG Internal State ==========\n";
  std::cout << "num_ende: " << tmp_num_ende << ", " << "num_ecls: " << tmp_num_ecls << "\n\n";

  std::cout << "Enodes info:\n";
  for (uint32_t i = 0; i < tmp_num_ende; i++) {
    std::cout << "  enode[" << i << "]: "
              << "name=" << tmp_ende_name[i] 
              << ", op=" << tmp_ende_op[i] 
              << ", cost=" << tmp_ende_cost[i]
              << ", eclass=" << tmp_ende_ecls[i]
              << "\n";

    uint32_t start = tmp_ende_child_ptr[i];
    uint32_t end   = tmp_ende_child_ptr[i + 1];
    std::cout << "    children: ";
    for (uint32_t jj = start; jj < end; jj++) {
      std::cout << tmp_ende_child[jj] << " ";
    }
    std::cout << "\n";
  }

  if (init) {
    std::cout << "\nEclass name -> id map:\n";
    for (auto &kv : tmp_ecls_name_to_id) {
      std::cout << "  " << kv.first << " -> " << kv.second << "\n";
    }
  }

  std::cout << "\nRoot eclasses:\n";
  for (auto r : tmp_root_eclass_id) {
    std::cout << "  " << r << "\n";
  }
}

void XEG::_print_degree(
  const bool is_td, 
  const std::vector<bool>& picked_endes) const 
{
  const std::string a = (is_td) ? ("Top-Down _indegree\n") : ("Bottom-up _inv_indegree\n");
  const std::string b = (is_td) ? ("_indegree") : ("_inv_indegree");
  const std::vector<uint32_t>& vec = (is_td) ? _indegree : _inv_indegree;
  
  std::cout << "XEG::_print_degree(): " << a ;
  for (uint32_t i = 0; i < vec.size(); i++) {
    if (picked_endes[i]) {
      std::cout << b << "[" << i << "] = " << vec[i] << " (picked: T)\n";
    } else {
      std::cout << b << "[" << i << "] = " << vec[i] << " (picked: F)\n";
    }
  }
  printf("\n");
}

void XEG::_print_levelization(
  const bool is_td, 
  const std::vector<int>& level,
  const std::vector<bool>& picked_endes) 
{
  const std::string a = (is_td) ? ("Top-Down\n") : ("Bottom-up\n");
  
  std::cout << "XEG::_print_levelization(): " << a ;
  printf("=== Levelization result ===\n");
 
   // Construct ende_in_level
  std::vector<std::vector<uint32_t>> ende_in_level;
  ende_in_level.resize(_max_level, std::vector<uint32_t>());
  for (uint32_t i = 0; i < level.size(); i++) {
    if (!picked_endes[i]) continue;
    int lvl = level[i]; 
    ende_in_level[lvl].push_back(i); 
  }
  for (uint32_t lvl = 0; lvl < ende_in_level.size(); lvl++) {
    printf("level %u: ", lvl); 
    for (uint32_t ende = 0; ende < ende_in_level[lvl].size(); ende++) {
      printf("%u, ", ende_in_level[lvl][ende]); 
    } printf("\n"); 
  }
  printf("\n"); 
}
