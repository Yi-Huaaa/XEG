#include "g_ext.cuh"

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

// Error checking macro for GPU 
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t cuda_status__ = (call);                                  \
    if (cuda_status__ != cudaSuccess) {                                  \
      fprintf(stderr,                                                    \
              "CUDA API error: %s at %s:%d\n",                           \
              cudaGetErrorString(cuda_status__),                         \
              __FILE__,                                                  \
              __LINE__);                                                 \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)    
 
namespace {

constexpr uint32_t kGpuThreads = 512u;
constexpr uint32_t kGpuReachabilityThreads = 256u;
constexpr uint32_t kGpuDagBatchSize = 256u; // Note: this must be 256 to ensure the solution quality

// Compile-time controller for GPU DAG levelization.
//
// Keep this false while verifying that controller separation
// preserves the original host-driven implementation.
// Change it to true only after the persistent path compiles
// and the host-driven baseline still passes all tests.
constexpr bool kEnableGpuDagHybridLevelization = true;

// Single-block persistent levelization parameters.
constexpr uint32_t kGpuDagPersistentThreads = 512u;

// Enter persistent mode only after the current frontier and
// the preceding frontiers have remained at or below this width.
constexpr uint32_t
  kGpuDagPersistentEnterFrontierThreshold = 512u; // 256 and 512 get similiar runtime 

// Require this many consecutive small frontiers before entering
// persistent mode. This prevents switching because of one short,
// temporary dip in frontier width.
constexpr uint32_t
  kGpuDagPersistentRequiredSmallLevels = 8u;

// The persistent kernel processes frontiers up to this width.
// A wider frontier is left unprocessed and returned to the host,
// which resumes with the existing multi-block kernel.
//
// With 512 threads, 1024 nodes require at most two node rounds.
constexpr uint32_t
  kGpuDagPersistentEmergencyExitThreshold = 1024u;

static_assert(
  kGpuDagPersistentThreads > 0u &&
  kGpuDagPersistentThreads <= 1024u,
  "Persistent block size must be a valid CUDA block size");

static_assert(
  kGpuDagPersistentThreads % 32u == 0u,
  "Persistent block size should be a whole number of warps");

static_assert(
  kGpuDagPersistentEnterFrontierThreshold > 0u,
  "Persistent entry threshold must be positive");

static_assert(
  kGpuDagPersistentRequiredSmallLevels > 0u,
  "Persistent entry streak must be positive");

static_assert(
  kGpuDagPersistentEmergencyExitThreshold >=
    kGpuDagPersistentEnterFrontierThreshold,
  "Emergency threshold must not be below entry threshold");

// Enables detailed GPU DAG diagnostics:
//   1. per-iteration proposal/survival/rollback logs
//   2. core/filter runtime breakdown
//
// GPU DAG total runtime and the outer-loop termination copy
// remain enabled regardless of this flag.
constexpr bool kEnableGpuDagIterationLog = false;

// ============================================================
// Batch low-activity controller
//
// A completed batch iteration is considered low activity when:
//
//   survived * 100 <= historical_peak * low_activity_percent
//
// The low-activity condition is reached only after:
//   1. at least minimum_iterations have completed
//   2. low activity has been observed for patience
//      consecutive iterations
//
// The insertion policy decides what happens after the
// condition is reached:
//
//   FullBatch:
//     ignore the condition
//
//   BatchEarlyStop:
//     stop DAG construction
//
//   BatchThenDirect:
//     switch subsequent iterations to direct insertion
//
// These constants affect only the host-side outer-loop
// controller. They do not modify insertion, levelization,
// finalization, or filtering kernels.
// ============================================================
constexpr uint32_t
  kGpuDagLowActivityMinimumIterations = 4u;

constexpr uint32_t
  kGpuDagLowActivityPercent = 3u;

constexpr uint32_t
  kGpuDagLowActivityPatience = 2u;

static_assert(
  kGpuDagLowActivityMinimumIterations > 0u,
  "Low-activity minimum iterations must be positive");

static_assert(
  kGpuDagLowActivityPercent > 0u &&
  kGpuDagLowActivityPercent <= 100u,
  "Low-activity percentage must be in [1, 100]");

static_assert(
  kGpuDagLowActivityPatience > 0u,
  "Low-activity patience must be positive");

static_assert(
  kGpuDagLowActivityMinimumIterations >=
    kGpuDagLowActivityPatience,
  "Minimum iterations should cover the low-activity streak");

template <typename T>
void cuda_malloc_array(T*& ptr, size_t count)
{
  ptr = nullptr;

  if (count == 0) {
    return;
  }

  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&ptr),
    count * sizeof(T)));
}

template <typename T>
void cuda_free_and_null(T*& ptr) noexcept
{
  if (ptr == nullptr) {
    return;
  }

  cudaError_t err = cudaFree(ptr);

  if (err != cudaSuccess) {
    fprintf(
      stderr,
      "Warning: cudaFree failed: %s\n",
      cudaGetErrorString(err));
  }

  ptr = nullptr;
}
}  // namespace


// G_EXT fucntions
void G_EXT::_read_input(
  const std::string input_path) 
{
  std::ifstream ifs(input_path);
  if (!ifs.is_open()) {
    std::cerr << "Error: cannot open file " 
      << input_path << std::endl;
    return;
  }

  // Parse JSON
  nlohmann::json j; ifs >> j;

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

void G_EXT::_prune_dead_nodes(
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
  printf("G_EXT::_prune_dead_nodes()\n");
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


}

// =========================== TREE/DAG functions =========================== // 

std::pair<double, double> G_EXT::_cpu_tree_construction(
  const std::vector<float>& tmp_ende_cost,
  std::vector<double>& cost_of_ecls,
  std::vector<double>& cost_of_ende,
  std::vector<uint32_t>& best_ende_of_ecls, 
  std::vector<bool>& picked_endes)
{
  bool h_update = true;
  cost_of_ecls.resize(_num_ecls, std::numeric_limits<double>::infinity());
  cost_of_ende.resize(_num_ende, std::numeric_limits<double>::infinity());
  best_ende_of_ecls.resize(_num_ecls, UINT32_MAX);

  while (h_update) {
    h_update = false;
    {
      for (uint32_t t_idx = 0; t_idx < _num_ende; t_idx++) {
        double acc_cost = tmp_ende_cost[t_idx];
        uint32_t start = _ende_child_ptr[t_idx];
        uint32_t end   = _ende_child_ptr[t_idx+1];
        for (uint32_t jj = start; jj < end; jj++) {
          acc_cost += cost_of_ecls[_ende_child[jj]];
        }
        cost_of_ende[t_idx] = (acc_cost < cost_of_ende[t_idx]) ? (acc_cost) : (cost_of_ende[t_idx]);
        const uint32_t which_ecls = _ende_ecls[t_idx];

        { // #pragma omp critical
          if (acc_cost < cost_of_ecls[which_ecls]) {
            cost_of_ecls[which_ecls] = acc_cost;
            best_ende_of_ecls[which_ecls] = t_idx;
            h_update = true;
          }
        }
      }
    }
  }

  // === Phase 2: reconstruct best picks ===
  std::vector<std::vector<uint32_t>> ende_picked_by_ende(_num_ende);
  for (uint32_t t_idx = 0; t_idx < _num_ende; t_idx++) {
    std::vector<uint32_t> picks;
    uint32_t start = _ende_child_ptr[t_idx];
    uint32_t end   = _ende_child_ptr[t_idx + 1];
    picks.reserve(end - start);
    for (uint32_t jj = start; jj < end; jj++) {
      uint32_t chd_ecls = _ende_child[jj];
      picks.push_back(best_ende_of_ecls[chd_ecls]);
    }
    ende_picked_by_ende[t_idx] = std::move(picks);
  }

#ifdef PRINT_LOG
  printf("\ncost_of_ecls:\n");
  for (size_t i = 0; i < cost_of_ecls.size(); i++) {
    printf("  [%zu] = %.1f\n", i, cost_of_ecls[i]);
  }

  printf("\ncost_of_ende:\n");
  for (size_t i = 0; i < cost_of_ende.size(); i++) {
    printf("  [%zu] = %.1f\n", i, cost_of_ende[i]);
  }
  printf("\n");  

  printf("\nbest_ende_of_ecls:\n");
  for (size_t i = 0; i < best_ende_of_ecls.size(); i++) {
    printf("  [%zu] = %u\n", i, best_ende_of_ecls[i]);
  }
  printf("\n");

  printf("\nende_picked_by_ende:\n");
  for (size_t i = 0; i < ende_picked_by_ende.size(); i++) {
    printf("  [%zu]: ", i);
    for (size_t j = 0; j < ende_picked_by_ende[i].size(); j++) {
      printf("%u, ", ende_picked_by_ende[i][j]);
    }
    printf("\n");  
  }
  printf("\n");  
#endif
  // Get best tree cost 
  double tree_ret = 0.0; 
  for (uint32_t r = 0; r < _root_eclass_id.size(); r++) {
    tree_ret += cost_of_ecls[_root_eclass_id[r]];
  }

  double dag_ret = _compute_dag_cost(best_ende_of_ecls, picked_endes, ende_picked_by_ende); 
  
  // construct the masks of adj and in_adj based on picked e-nodes
  _init_picked_adj(picked_endes);

  return std::make_pair(tree_ret, dag_ret);
}

double G_EXT::_compute_dag_cost(
  const std::vector<uint32_t>& best_ende_of_ecls,
  std::vector<bool>& picked_endes,
  std::vector<std::vector<uint32_t>>& ende_picked_by_ende) 
{
  // Get best dag cost 
  double dag_ret = 0.0; 
  picked_endes.resize(_num_ende, false); 

  std::queue<uint32_t> q;

  // enqueue all root enodes
  for (uint32_t rid : _root_eclass_id) {
    MY_CHECK(rid < _num_ecls, "rid < _num_ecls");
    uint32_t root_ende = best_ende_of_ecls[rid];
    MY_CHECK(root_ende < _num_ende, "root_ende < _num_ende"); 
    q.push(root_ende);
    picked_endes[root_ende] = true;
  }

  // BFS traversal over enodes
  while (!q.empty()) {
    uint32_t cur = q.front();
    dag_ret += _ende_cost[cur];
    q.pop();

    for (uint32_t chd : ende_picked_by_ende[cur]) {
      MY_CHECK(chd < _num_ende, "chd < _num_ende");
      if (!picked_endes[chd]) {
        picked_endes[chd] = true;
        q.push(chd);
      }
    }
  }

#ifdef PRINT_LOG
  printf("\nActivated enodes in DAG traversal:\n");
  for (size_t i = 0; i < picked_endes.size(); i++) {
    if (picked_endes[i]) {
      printf("  enode[%zu] (eclass %u, cost %.1f)\n", 
             i, _ende_ecls[i], _ende_cost[i]);
    }
  }
#endif
  return dag_ret;
}

void G_EXT::_init_picked_adj(
  const std::vector<bool>& picked_endes) 
{
  _indegree.resize(_num_ende, 0);
  _inv_indegree.resize(_num_ende, 0);

  // init _picked_adj
  for (uint32_t fm_ende = 0; fm_ende < _num_ende; fm_ende++) {
    if (!picked_endes[fm_ende]) continue; 
    uint32_t start = _global_adj_1D_ptr[fm_ende];
    uint32_t end = _global_adj_1D_ptr[fm_ende+1];
    for (uint32_t k = start; k < end; k++) {
      const uint32_t to_ende = _global_adj_1D[k];
      if (!picked_endes[to_ende]) continue;
      _picked_adj[k] = true;
      _indegree[to_ende]++;
      _inv_indegree[fm_ende]++;
    }
  }

  // init _picked_inv_adj
  for (uint32_t fm_ende = 0; fm_ende < _num_ende; fm_ende++) {
    if (!picked_endes[fm_ende]) continue; 
    uint32_t start = _global_inv_adj_1D_ptr[fm_ende];
    uint32_t end = _global_inv_adj_1D_ptr[fm_ende+1];
    for (uint32_t k = start; k < end; k++) {
      const uint32_t to_ende = _global_inv_adj_1D[k];
      if (!picked_endes[to_ende]) continue;
      _picked_inv_adj[k] = true;
    }
  }

#ifdef PRINT_LOG
  printf("\n===== _picked_adj =====\n");
  for (uint32_t fm_ende = 0; fm_ende < _num_ende; fm_ende++) {
    uint32_t start = _global_adj_1D_ptr[fm_ende];
    uint32_t end   = _global_adj_1D_ptr[fm_ende + 1];
    bool has = false;
    for (uint32_t k = start; k < end; k++) {
      if (_picked_adj[k]) {
        if (!has) {
          printf("from %u:", fm_ende);
          has = true;
        }
        printf(" %u", _global_adj_1D[k]);
      }
    }
    if (has) printf("\n");
  }

  printf("\n===== _picked_inv_adj =====\n");
  for (uint32_t fm_ende = 0; fm_ende < _num_ende; fm_ende++) {
    uint32_t start = _global_inv_adj_1D_ptr[fm_ende];
    uint32_t end   = _global_inv_adj_1D_ptr[fm_ende + 1];
    bool has = false;
    for (uint32_t k = start; k < end; k++) {
      if (_picked_inv_adj[k]) {
        if (!has) {
          printf("to %u:", fm_ende);
          has = true;
        }
        printf(" %u", _global_inv_adj_1D[k]);
      }
    }
    if (has) printf("\n");
  }

  _print_degree(true, picked_endes);  
  _print_degree(false, picked_endes);    
#endif
}

void G_EXT::_cpu_DAG_construction(
  std::vector<bool>& picked_endes)
{
#ifdef PRINT_LOG
  printf("picked_endes\n");
  for (uint32_t i = 0; i < picked_endes.size(); i++) {
    if (picked_endes[i]) {
      std::cout << "picked_endes[" << i << "]: TRUE\n"; 
    }
  }
  for (uint32_t i = 0; i < picked_endes.size(); i++) {
    if (!picked_endes[i]) {
      std::cout << "picked_endes[" << i << "]: FALSE\n"; 
    }
  }
#endif

  bool updated = true; 
  std::vector<int> level;
  level.resize(_num_ende, -1); // init, needed
  _new_inserted.assign(_num_ende, false); // init, needed
  int itr = 0;
  while(updated) {
    // bottom-up levelization && top-down insertion
    _update_deg(updated, picked_endes); 
    _bottom_up_levelize(picked_endes, level);
    bool tp = _insertion(true, picked_endes, level);

    // top-down levelization && bottom-up insertion
    _update_deg(tp, picked_endes); 
    _top_down_levelize(picked_endes, level);
    bool bu = _insertion(false, picked_endes, level);
   
    updated = tp || bu; 
    
    #ifdef PRINT_LOG
      std::cout << "Get into while loop again? " << updated << "\n==========================================\n"; 
    #endif
    itr++;
  }
  
  printf("CPU DAG iterations = %d\n", itr);
  
  // Clear redundant enodes
  _clear_redundant_endes(picked_endes);

  // Re-constructing data for second times ILP solver
  _ecls_its_ende_newG.resize(_ecls_its_ende.size(), std::vector<uint32_t>());
  for (uint32_t ecls = 0; ecls < _ecls_its_ende.size(); ecls++) {
    for (uint32_t ende : _ecls_its_ende[ecls]) {
      if (!_in_subgraph[ende]) continue;
      _ecls_its_ende_newG[ecls].push_back(ende);
    }
  }
}

void G_EXT::_update_deg(
  const bool need_to_update, 
  const std::vector<bool>& picked_endes) 
{
#ifdef PRINT_LOG 
  printf("\nInside G_EXT::_update_deg()\n");
  printf("Pre:\n"); 
  _print_degree(true, picked_endes);
  _print_degree(false, picked_endes);
#endif
  if (!need_to_update) return; 

  _indegree.assign(_num_ende, 0);
  _inv_indegree.assign(_num_ende, 0);

  for (uint32_t fm_ende = 0; fm_ende < _num_ende; fm_ende++) {
    if (!picked_endes[fm_ende]) continue; 
    for (uint32_t k = _global_adj_1D_ptr[fm_ende]; k < _global_adj_1D_ptr[fm_ende+1]; k++) {
      if (!_picked_adj[k]) continue;
      const uint32_t to_ende = _global_adj_1D[k];
      _indegree[to_ende]++;
      _inv_indegree[fm_ende]++;
    }
  }

#ifdef PRINT_LOG
  printf("Post:\n"); 
  _print_degree(true, picked_endes);
  _print_degree(false, picked_endes);
#endif  
}

void G_EXT::_BFS(
  const bool is_td, std::vector<uint32_t>& deg, 
  std::vector<bool>& picked_endes, // <-- 拿掉 const
  const std::vector<uint32_t>& global_adj_1D,
  const std::vector<uint32_t>& global_adj_1D_ptr,
  const std::vector<bool>& picked_adj,
  std::vector<int>& level)
{
  // lambda
  auto get_rev_level = [&](const uint32_t LEVEL, uint32_t a) -> uint32_t {
    return LEVEL - a - 1;
  };

  // init queue (roots || leaves)
  std::queue<uint32_t> q;
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if ((deg[ende] == 0) && picked_endes[ende]) {
      q.push(ende);
    }
  } 

  // BFS: top-down levelization 
  int cur_level = 0;
  while (!q.empty()) {
    size_t layer = q.size();
    for (size_t t = 0; t < layer; ++t) {
      uint32_t u = q.front(); q.pop();
      level[u] = cur_level;
      for (uint32_t k = global_adj_1D_ptr[u]; k < global_adj_1D_ptr[u+1]; k++) {
        if (!picked_adj[k]) continue;
        const uint32_t v = global_adj_1D[k];
        if (deg[v] > 0 && --deg[v] == 0) {
          q.push(v);
        }        
      }
    }
    ++cur_level;
  }

  // update _max_level
  _max_level = cur_level; 

  // cycle revert; Note: actually here can be optimized; cannot in parallel, but can be optimized; 
  // but I haven't optimize the implementation here; that is life.
  for (uint32_t i = 0; i < _num_ende; i++) {
    if (_new_inserted[i] && deg[i] > 0) {
      picked_endes[i] = false;
      _new_inserted[i] = false;
      level[i] = -1; 

      // clear mask, e-node to child e-nodes
      for (uint32_t k = _global_adj_1D_ptr[i]; k < _global_adj_1D_ptr[i+1]; k++) {
        _picked_adj[k] = false;
        uint32_t chd = _global_adj_1D[k];
        // chd's inv_adj to e-node i -> false
        for (uint32_t kk = _global_inv_adj_1D_ptr[chd]; kk < _global_inv_adj_1D_ptr[chd+1]; kk++) {
          if (_global_inv_adj_1D[kk] == i) {
            _picked_inv_adj[kk] = false;
            break;
          }
        }
      }

      // clear mask, e-node to parent e-nodes
      for (uint32_t k = _global_inv_adj_1D_ptr[i]; k < _global_inv_adj_1D_ptr[i+1]; k++) {
        _picked_inv_adj[k] = false;
        uint32_t par = _global_inv_adj_1D[k];
        // par's adj to e-node i -> false
        for (uint32_t kk = _global_adj_1D_ptr[par]; kk < _global_adj_1D_ptr[par+1]; kk++) {
          if (_global_adj_1D[kk] == i) {
            _picked_adj[kk] = false;
            break;
          }
        }
      }

      #ifdef PRINT_LOG
        printf("[Cycle Revert] Node %u removed due to cycle!\n", i);
      #endif
    }
  }

  // reverse assigned level, just for bottom-up levelization
  if (!is_td) {
    for (uint32_t i = 0; i < level.size(); i++) {
      if (level[i] != -1) {
        level[i] = get_rev_level(_max_level, level[i]);
      }
    }
  }

#ifdef PRINT_LOG
  printf("_max_level = %d, cur_level = %d\n", _max_level, cur_level); 
#endif
}

void G_EXT::_top_down_levelize(
  std::vector<bool>& picked_endes,
  std::vector<int>& level) 
{
#ifdef PRINT_LOG
  printf("\nG_EXT::_top_down_levelize()\n");
#endif

  // Top-down BFS levelization
  level.assign(_num_ende, -1); 
  _BFS(true, _indegree, picked_endes, 
      _global_adj_1D, _global_adj_1D_ptr, _picked_adj, level);

#ifdef PRINT_LOG
  _print_levelization(true, level, picked_endes);
#endif 
}

void G_EXT::_bottom_up_levelize(
  std::vector<bool>& picked_endes,
  std::vector<int>& level) 
{
#ifdef PRINT_LOG
  printf("\nG_EXT::_bottom_up_levelize()\n");
#endif

  // Bottom-up BFS levelization
  level.assign(_num_ende, -1); 
  _BFS(false, _inv_indegree, picked_endes, 
      _global_inv_adj_1D, _global_inv_adj_1D_ptr, _picked_inv_adj, level);

#ifdef PRINT_LOG
  _print_levelization(false, level, picked_endes);
#endif 
}

bool G_EXT::_insertion(const bool is_td, 
  std::vector<bool>& picked_endes,
  std::vector<int>& level) 
{
#ifdef PRINT_LOG
  if (is_td) {
    printf("\nG_EXT::_top_down_insertion()\n");
  } else {
    printf("\nG_EXT::_bottom_up_insertion()\n");
  }
#endif

  bool update = false;

  // Step 1: 初始化 cycle revert 標記
  // 將所有 _new_inserted 設為 false，只捕捉「這一輪」剛被平行的 kGpuThreads 插入的點
  _new_inserted.assign(_num_ende, false);

  // Step 3: 解放平行運算
  // 使用 dynamic 排程，因為每個節點的 degree 不同，負載可能不均
  // #pragma omp parallel for schedule(dynamic)
  // #pragma omp parallel for schedule(static, 16)
  #pragma omp parallel for
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if (picked_endes[ende]) {
      continue;
    }

    // Step 2: 使用 Local Vector 取代全域的 _tmp_picked_adj
    // 這將運算完全鎖在 CPU 的 L1/L2 Cache 內，大幅降低 Memory Bandwidth 壓力
    bool has_par = false, has_chd = false;
    int max_par_lvl = -1, min_chd_lvl = INT_MAX;
    
    std::vector<uint32_t> valid_pars;
    std::vector<uint32_t> valid_chds;

    // 尋找 max parent level
    const uint32_t p_start = _global_inv_adj_1D_ptr[ende];
    const uint32_t p_end   = _global_inv_adj_1D_ptr[ende+1];
    for (uint32_t k = p_start; k < p_end; k++) {
      uint32_t par_ende = _global_inv_adj_1D[k];
      if (!picked_endes[par_ende]) continue;
      
      has_par = true;
      int par_lvl = level[par_ende];
      valid_pars.push_back(par_ende);
      
      if (max_par_lvl < par_lvl) {
        max_par_lvl = par_lvl; 
      }
    }

    // 尋找 min child level
    const uint32_t c_start = _global_adj_1D_ptr[ende];
    const uint32_t c_end   = _global_adj_1D_ptr[ende+1];
    for (uint32_t k = c_start; k < c_end; k++) {
      uint32_t chd_ende = _global_adj_1D[k];      
      if (!picked_endes[chd_ende]) continue;
      
      has_chd = true;
      int chd_lvl = level[chd_ende];
      valid_chds.push_back(chd_ende);
      
      if (min_chd_lvl > chd_lvl) {
        min_chd_lvl = chd_lvl;
      }
    }

    // Check whether form a cycle (區域變數 Overlap Check)
    bool cond = (is_td) ? (has_par) : (has_chd);
    int given_lvl = (is_td) ? (max_par_lvl + 1) : (min_chd_lvl - 1);
    
    bool has_overlap_b = false;
    for (uint32_t p : valid_pars) {
      for (uint32_t c : valid_chds) {
        if (p == c) {
          has_overlap_b = true;
          break;
        }
      }
      if (has_overlap_b) break;
    }

    bool valid = (cond) && (max_par_lvl < min_chd_lvl) && (!has_overlap_b) && (given_lvl >= 0);
    
    // Step 3 (續): 只做輕量級的狀態更新，完全不碰 _picked_adj 這些 Mask
    if (valid) {
      update = true; 
      // 因為每個 thread 處理不同的 ende，直接寫入陣列是 thread-safe 的
      picked_endes[ende] = true;
      level[ende] = given_lvl;
      _new_inserted[ende] = true; // 標記為剛插入，交給接下來的 BFS 把關
      
      #ifdef PRINT_LOG
        printf("! valid ende = %u, max_par_lvl = %d, min_chd_lvl = %d, given_lvl = %d\n", 
               ende, max_par_lvl, min_chd_lvl, given_lvl);
      #endif
    } 
  } // End of parallel for

  // Step 4: 統一更新 Mask (批次重建)
  // 如果這一輪有任何節點被成功加入，我們就呼叫你已經寫好的函數，一口氣把全域 Edge Masks 建好
  if (update) {
    _init_picked_adj(picked_endes);
  }

  return update;
}


void G_EXT::_update_picked_adj(const uint32_t fm_ende, const uint32_t to_ende) 
{
  // bool check = false; // only used for check and debug, can be revmoed 
  for (uint32_t k = _global_adj_1D_ptr[fm_ende]; k < _global_adj_1D_ptr[fm_ende+1]; k++) {
    const uint32_t tar_ende = _global_adj_1D[k];
    if (to_ende == tar_ende) {
      _picked_adj[k] = true;
      // check = true;
      break;
    }
  }
  // MY_CHECK(check == true, "check == true 0"); check = false;
  for (uint32_t k = _global_inv_adj_1D_ptr[to_ende]; k < _global_inv_adj_1D_ptr[to_ende+1]; k++) {
    const uint32_t tar_ende = _global_inv_adj_1D[k];
    if (fm_ende == tar_ende) {
      _picked_inv_adj[k] = true;
      // check = true;
      break;
    }
  }
  // MY_CHECK(check == true, "check == true 1"); 
}

void G_EXT::_clear_redundant_endes(std::vector<bool>& picked_endes) 
{
#ifdef PRINT_LOG
  printf("Insides G_EXT::_clear_redundant_endes() \n");
#endif  



#ifdef PRINT_LOG
  printf("_roots: ");
  for (uint32_t r = 0; r < _roots.size(); r++) {
    if (picked_endes[_roots[r]]) {
      printf("%u, ", _roots[r]); 
    }
  } printf("\n"); 

  printf("_leavs: ");
  for (uint32_t r = 0; r < _leaves.size(); r++) {
    if (picked_endes[_leaves[r]]) {
      printf("%u, ", _leaves[r]); 
    }
  } printf("\n");   
  printf("picked_endes now:  ");
  for (uint32_t i = 0; i < picked_endes.size(); i++) {
    if (picked_endes[i]) {
      printf("%u, ", i);
    } 
  } printf("\n");
#endif 

  get_nodes_in_subgraph(_roots, _leaves, picked_endes);

#ifdef PRINT_LOG
  printf("enodes in subgraph:  ");
  for (uint32_t ende = 0; ende < _in_subgraph.size(); ende++) {
    if (_in_subgraph[ende]) {
      printf("%u, ", ende);
    }
  } 
  printf("\n");  
#endif
}

void G_EXT::_mark_reachable(const std::vector<uint32_t>& starts,
  const std::vector<uint32_t>& global_adj_1D,
  const std::vector<uint32_t>& global_adj_1D_ptr,
  const std::vector<bool>& picked_adj,
  std::vector<bool>& visited,
  const std::vector<bool>& picked_endes) 
{
  // new version
  std::queue<uint32_t> q;
  for (uint32_t s : starts) {
    if (picked_endes[s] && !visited[s]) {
      q.push(s);
      visited[s] = true;
    }
  }
  while (!q.empty()) {
    uint32_t u = q.front(); q.pop();
    for (uint32_t k = global_adj_1D_ptr[u]; k < global_adj_1D_ptr[u+1]; k++) {
      if (!picked_adj[k]) continue;
      const uint32_t v = global_adj_1D[k];
      MY_CHECK(picked_endes[v] == true, "picked_endes[v] == true");
      if (!visited[v]) {
        visited[v] = true;
        q.push(v);
      }
    }
  }
}

void G_EXT::get_nodes_in_subgraph(
    const std::vector<uint32_t>& roots,
    const std::vector<uint32_t>& leaves,
    const std::vector<bool>& picked_endes)
{
  std::vector<bool> from_roots(_num_ende, false);
  std::vector<bool> to_leaves(_num_ende, false);
  _in_subgraph.resize(_num_ende, false);

  // 1. reachable from roots
  _mark_reachable(roots, 
                _global_adj_1D, _global_adj_1D_ptr, _picked_adj, 
                from_roots, picked_endes);

  // 2. can reach leaves
  _mark_reachable(leaves, 
                _global_inv_adj_1D, _global_inv_adj_1D_ptr, _picked_inv_adj, 
                to_leaves, picked_endes);

  // 3. intersection
  for (size_t i = 0; i < _num_ende; ++i) {
    _in_subgraph[i] = from_roots[i] && to_leaves[i];
  }
}

bool G_EXT::_insertion_bitmap_mt(const bool is_td, 
  std::vector<bool>& picked_endes,
  std::vector<int>& level) 
{
  bool update = false;
  _new_inserted.assign(_num_ende, false);

  #pragma omp parallel for
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if (picked_endes[ende]) {
      continue;
    }

    bool has_par = false, has_chd = false;
    int max_par_lvl = -1, min_chd_lvl = INT_MAX;

    // 每個 thread 各自的 local bitmap（跟 GPU kernel 的 shared bitmap 邏輯對應）
    const uint32_t BITMAP_SIZE = 256;
    uint8_t chd_flags[BITMAP_SIZE] = {0};


    // ============================================================
    // Step 1: 掃 child edges，建立 bitmap，順便找 min child level
    // ============================================================
    const uint32_t c_start = _global_adj_1D_ptr[ende];
    const uint32_t c_end   = _global_adj_1D_ptr[ende+1];
    for (uint32_t k = c_start; k < c_end; k++) {
      uint32_t chd_ende = _global_adj_1D[k];
      if (!picked_endes[chd_ende]) continue;

      has_chd = true;
      int chd_lvl = level[chd_ende];
      if (min_chd_lvl > chd_lvl) {
        min_chd_lvl = chd_lvl;
      }
      chd_flags[chd_ende & (BITMAP_SIZE - 1)] = 1; 
    }

    // ============================================================
    // Step 2: 掃 parent edges，找 max parent level，同時用 bitmap 檢查 overlap
    // ============================================================
    const uint32_t p_start = _global_inv_adj_1D_ptr[ende];
    const uint32_t p_end   = _global_inv_adj_1D_ptr[ende+1];
    bool has_overlap_b = false;

    for (uint32_t k = p_start; k < p_end; k++) {
      uint32_t par_ende = _global_inv_adj_1D[k];
      if (!picked_endes[par_ende]) continue;

      has_par = true;
      int par_lvl = level[par_ende];
      if (max_par_lvl < par_lvl) {
        max_par_lvl = par_lvl;
      }

      // bitmap overlap check（跟 GPU kernel 完全對應的邏輯）
      if (chd_flags[par_ende & (BITMAP_SIZE - 1)]) {
        has_overlap_b = true;
      }
    }

    // ============================================================
    // Step 3: 判斷有效性（跟 GPU kernel 完全一致）
    // ============================================================
    bool cond = (is_td) ? (has_par) : (has_chd);
    int given_lvl = (is_td) ? (max_par_lvl + 1) : (min_chd_lvl - 1);
    bool valid = (cond) && (max_par_lvl < min_chd_lvl) && (!has_overlap_b) && (given_lvl >= 0);

    // ===== 唯一一個 if (valid) 區塊，內部用 macro 決定要不要多做計數 =====
    if (valid) {
      update = true;
      picked_endes[ende] = true;
      level[ende] = given_lvl;
      _new_inserted[ende] = true;

      #ifdef PRINT_LOG
        printf("! valid ende = %u, max_par_lvl = %d, min_chd_lvl = %d, given_lvl = %d\n", 
               ende, max_par_lvl, min_chd_lvl, given_lvl);
      #endif
    }

  } // End of parallel for

  if (update) {
    _init_picked_adj(picked_endes);
  }
  return update;
}

void G_EXT::_cpu_DAG_construction_mt_bitmap(
  std::vector<bool>& picked_endes)
{
  uint32_t num_iter = 0;
  bool updated = true; 
  std::vector<int> level;
  level.resize(_num_ende, -1);
  _new_inserted.assign(_num_ende, false);

  while(updated) {
    _update_deg(updated, picked_endes); 
    _bottom_up_levelize(picked_endes, level);   // 沿用既有、已驗證正確的 CPU levelization+revert
    bool tp = _insertion_bitmap_mt(true, picked_endes, level); // 只有這裡換成 bitmap 版本

    _update_deg(tp, picked_endes); 
    _top_down_levelize(picked_endes, level);
    bool bu = _insertion_bitmap_mt(false, picked_endes, level);

    updated = tp || bu; 
    num_iter++;

    // if (num_iter > 1000) {  // 安全閥，避免真的卡死
    //   std::cerr << "[WARNING] CPU+bitmap version exceeded 1000 iterations, likely livelock\n";
    //   break;
    // }
  }

  // Clear redundant enodes
  _clear_redundant_endes(picked_endes);

  // Re-constructing data for second times ILP solver
  _ecls_its_ende_newG.resize(_ecls_its_ende.size(), std::vector<uint32_t>());
  for (uint32_t ecls = 0; ecls < _ecls_its_ende.size(); ecls++) {
    for (uint32_t ende : _ecls_its_ende[ecls]) {
      if (!_in_subgraph[ende]) continue;
      _ecls_its_ende_newG[ecls].push_back(ende);
    }
  }
}

// =========================== GPU functions =========================== // 


__device__ double atomicMinDouble(double* addr, double val) {
  unsigned long long int* addr_as_ull = (unsigned long long int*)addr;
  unsigned long long int old = *addr_as_ull, assumed;

  do {
    assumed = old;
    double old_val = __longlong_as_double(assumed);
    if (old_val <= val) break;
    old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__global__ void _gpu_tree_construction_kernel(const uint32_t num_endes, 
  const float* ende_cost, 
  const uint32_t* ende_ecls,
  const uint32_t* ende_child,
  const uint32_t* ende_child_ptr,
  bool* update, 
  double* cost_of_ecls,
  double* cost_of_ende,
  uint32_t* best_ende_of_ecls) 
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num_endes) return;

  // float acc_cost = ende_cost[t_idx];
  double acc_cost = (double)ende_cost[t_idx]; // promote from float
  uint32_t start = ende_child_ptr[t_idx];
  uint32_t end   = ende_child_ptr[t_idx+1];
  for (uint32_t jj = start; jj < end; jj++) {
    acc_cost += cost_of_ecls[ende_child[jj]];
  }

  cost_of_ende[t_idx] = fmin(cost_of_ende[t_idx], acc_cost);
  const uint32_t which_ecls = ende_ecls[t_idx];

  double old = atomicMinDouble(&cost_of_ecls[which_ecls], acc_cost);
  if (acc_cost < old) {
    best_ende_of_ecls[which_ecls] = t_idx;
    *update = true; // No need to be atomic
  }  
}

__global__ void _resolve_best_ende(
  const uint32_t num_ecls,
  const uint32_t* ecls_its_ende_1D,
  const uint32_t* ecls_its_ende_1D_ptr,
  const double*   cost_of_ecls,
  const double*   cost_of_ende,
  uint32_t*       best_ende_of_ecls)
{
  uint32_t ecls = blockIdx.x * blockDim.x + threadIdx.x;
  if (ecls >= num_ecls) return;

  const uint32_t start = ecls_its_ende_1D_ptr[ecls];
  const uint32_t end   = ecls_its_ende_1D_ptr[ecls+1];
  const double btendecost = cost_of_ende[best_ende_of_ecls[ecls]];
 
  for (uint32_t k = start; k < end; k++) {
    uint32_t ende = ecls_its_ende_1D[k];
    if (btendecost > cost_of_ende[ende]) { 
      best_ende_of_ecls[ecls] = ende;
    }
  }
}

__global__ void _get_tree_cost(
  const uint32_t size, 
  const double* cost_of_ecls,
  const uint32_t* root_eclass_id,
  double* tree_ret) 
{
  *tree_ret = 0.0; // init
  for (uint32_t r = 0; r < size; r++) {
    *tree_ret += cost_of_ecls[root_eclass_id[r]];
  }
}

__global__ void _init_q(
  const uint32_t sz, 
  const uint32_t* root_eclass_id,
  const uint32_t* best_ende_of_ecls,
  uint32_t* q,
  uint32_t* q_sz,
  uint32_t* q_num,
  uint32_t* picked_endes)
{
  uint32_t pos = 0;
  for (uint32_t i = 0; i < sz; i++) {
    uint32_t ecls = root_eclass_id[i];
    uint32_t best_ende = best_ende_of_ecls[ecls];
    picked_endes[best_ende] = 1u;
    q[pos] = best_ende;
    pos++;
  }
  *q_sz = pos;
  *q_num = pos;
}

__global__ void _get_dag_cost(
  const uint32_t num, 
  const float* ende_cost,
  const uint32_t* ende_child,
  const uint32_t* ende_child_ptr,
  const uint32_t* best_ende_of_ecls,
  uint32_t* q,
  uint32_t* q_sz,
  const uint32_t* q_sp,
  uint32_t* picked_endes, // No bool atomicCAS
  double* dag_ret)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num) return;
  uint32_t cur = q[*q_sp + t_idx];
  atomicAdd(dag_ret, static_cast<double>(ende_cost[cur]));
  picked_endes[cur] = 1u; // set as true
  
  uint32_t start = ende_child_ptr[cur];
  uint32_t end   = ende_child_ptr[cur + 1];

  for (uint32_t chd = start; chd < end; chd++) {
    uint32_t chd_ecls = ende_child[chd];
    uint32_t chd_ende = best_ende_of_ecls[chd_ecls];
    // 確保只有一次 push
    if (atomicCAS(&picked_endes[chd_ende], 0u, 1u) == 0u) {
      uint32_t pos = atomicAdd(q_sz, 1);
      q[pos] = chd_ende;
    }
  }
}

__global__ void _print_gpu_degree(const uint32_t n0, 
  const uint32_t n1, 
  const uint32_t* dvc_picked_endes,
  const uint32_t* dvc_indegree,
  const uint32_t* dvc_inv_indegree) 
{
  printf("\n==================== _print_gpu_degree ====================\n");
  printf("  dvc_indegree:\n");
  for (uint32_t i = 0; i < n0; i++) {
    if (dvc_picked_endes[i]) {
      printf("    dvc_indegree[%u] = %u (picked: T)\n", i, dvc_indegree[i]);
    } else {
      printf("    dvc_indegree[%u] = %u (picked: F)\n", i, dvc_indegree[i]);
    }
  }
  printf("  dvc_inv_indegree:\n");
  for (uint32_t i = 0; i < n1; i++) {
    if (dvc_picked_endes[i]) {
      printf("    dvc_indegree[%u] = %u (picked: T)\n", i, dvc_inv_indegree[i]);
    } else {
      printf("    dvc_indegree[%u] = %u (picked: F)\n", i, dvc_inv_indegree[i]);
    }
  }
  printf("\n");
}

__global__ void _print_gpu_init_picked_adj(
  const uint32_t n0, 
  const uint32_t n1, 
  const uint32_t n2, 
  const uint32_t n3, 
  const uint32_t* dvc_picked_endes,
  const uint32_t* dvc_indegree,
  const uint32_t* dvc_inv_indegree,
  const uint32_t* dvc_picked_adj,
  const uint32_t* dvc_picked_inv_adj,
  const uint32_t* dvc_global_adj_1D_ptr, 
  const uint32_t* dvc_global_inv_adj_1D_ptr,
  const uint32_t* dvc_global_adj_1D,
  const uint32_t* dvc_global_inv_adj_1D)
{
  printf("_print_gpu_init_picked_adj\n");
  printf("\n===== dvc_picked_adj =====\n");
  for (uint32_t fm_ende = 0; fm_ende < n2; fm_ende++) {
    uint32_t start = dvc_global_adj_1D_ptr[fm_ende];
    uint32_t end   = dvc_global_adj_1D_ptr[fm_ende + 1];
    bool has = false;
    for (uint32_t k = start; k < end; k++) {
      if (dvc_picked_adj[k]) {
        if (!has) {
          printf("from %u:", fm_ende);
          has = true;
        }
        printf(" %u", dvc_global_adj_1D[k]);
      }
    }
    if (has) printf("\n");
  }

  printf("\n===== dvc_picked_inv_adj =====\n");
  for (uint32_t fm_ende = 0; fm_ende < n3; fm_ende++) {
    uint32_t start = dvc_global_inv_adj_1D_ptr[fm_ende];
    uint32_t end   = dvc_global_inv_adj_1D_ptr[fm_ende + 1];
    bool has = false;
    for (uint32_t k = start; k < end; k++) {
      if (dvc_picked_inv_adj[k]) {
        if (!has) {
          printf("to %u:", fm_ende);
          has = true;
        }
        printf(" %u", dvc_global_inv_adj_1D[k]);
      }
    }
    if (has) printf("\n");
  }
}

__global__ void _gpu_warm_up(
  const uint32_t num_ende, 
  int* dvc_level)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num_ende) return;  
  dvc_level[t_idx] = -1;
}


std::pair<double, double> G_EXT::_gpu_tree_construction() 
{
  int blocks = (_num_ende + kGpuThreads - 1) / kGpuThreads;

  // Local memory 
  float* dvc_ende_cost; 
  uint32_t* dvc_ende_ecls;
  uint32_t* dvc_ende_child;
  uint32_t* dvc_ende_child_ptr;
  double* dvc_cost_of_ecls; 
  double* dvc_cost_of_ende; 
  double* dvc_tree_ret;
  double* dvc_dag_ret;
  
  // GPU relaxation 
  bool h_update = true;
  std::vector<double> cost_of_ecls, cost_of_ende; 
  std::vector<uint32_t> best_ende_of_ecls;
  cost_of_ecls.resize(_num_ecls, std::numeric_limits<double>::infinity());
  cost_of_ende.resize(_num_ende, std::numeric_limits<double>::infinity());
  best_ende_of_ecls.resize(_num_ecls, 0);
  // Get best tree cost
  cudaMalloc(&dvc_ende_cost, _ende_cost.size()*sizeof(float));
  cudaMalloc(&dvc_ende_ecls, _ende_ecls.size()*sizeof(uint32_t));
  cudaMalloc(&dvc_ende_child, _ende_child.size()*sizeof(uint32_t));
  cudaMalloc(&dvc_ende_child_ptr, _ende_child_ptr.size()*sizeof(uint32_t));
  cudaMalloc(&dvc_cost_of_ecls, cost_of_ecls.size()*sizeof(double));
  cudaMalloc(&dvc_cost_of_ende, cost_of_ende.size()*sizeof(double));
  cudaMalloc(&dvc_tree_ret, 1*sizeof(double));
  cudaMalloc(&dvc_dag_ret, 1*sizeof(double));
  cudaMalloc(&_dvc_best_ende_of_ecls, best_ende_of_ecls.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_update, 1*sizeof(bool));
  cudaMalloc(&_dvc_ecls_its_ende_1D, _ecls_its_ende_1D.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_ecls_its_ende_1D_ptr, _ecls_its_ende_1D_ptr.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_root_eclass_id, _root_eclass_id.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_ende_picked_by_ende, _ende_child.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_picked_endes, _num_ende*sizeof(uint32_t)); // act bool
  cudaMalloc(&_dvc_q, _num_ende*sizeof(uint32_t));
  cudaMalloc(&_dvc_q_sz, 1*sizeof(uint32_t));
  cudaMalloc(&_dvc_q_sp, 1*sizeof(uint32_t));
  cudaMalloc(&_dvc_q_num, 1*sizeof(uint32_t));
  cudaMalloc(&_dvc_indegree, _num_ende*sizeof(uint32_t));
  cudaMalloc(&_dvc_inv_indegree, _num_ende*sizeof(uint32_t));
  // new G
  cudaMalloc(&_dvc_global_adj_1D, _global_adj_1D.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_global_adj_1D_ptr, _global_adj_1D_ptr.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_global_inv_adj_1D, _global_inv_adj_1D.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_global_inv_adj_1D_ptr, _global_inv_adj_1D_ptr.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_picked_adj, _global_adj_1D.size()*sizeof(uint32_t)); // set as 0
  cudaMalloc(&_dvc_picked_inv_adj, _global_inv_adj_1D.size()*sizeof(uint32_t)); // set as 0
  cudaMalloc(&_dvc_tmp_picked_adj, _global_adj_1D.size()*sizeof(uint32_t)); // set as 0
  cudaMalloc(&_dvc_tmp_picked_inv_adj, _global_inv_adj_1D.size()*sizeof(uint32_t)); // set as 0
  cudaMalloc(&_dvc_level, _num_ende*sizeof(int));
  cudaMalloc(&_dvc_all_root_ende_1D, _all_root_ende_1D.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_max_level, 1*sizeof(int));
  cudaMalloc(&_dvc_in_subgraph, _num_ende*sizeof(uint32_t));
  cudaMalloc(&_dvc_roots, _roots.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_leaves, _leaves.size()*sizeof(uint32_t));
  cudaMalloc(&_dvc_from_roots, _num_ende*sizeof(uint32_t));
  cudaMalloc(&_dvc_from_leaves, _num_ende*sizeof(uint32_t));
  cudaCheckErrors("cudaMalloc failed");

  // Get best tree cost
  cudaMemcpy(dvc_ende_cost, _ende_cost.data(), _ende_cost.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_ende_ecls, _ende_ecls.data(), _ende_ecls.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_ende_child, _ende_child.data(), _ende_child.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_ende_child_ptr, _ende_child_ptr.data(), _ende_child_ptr.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_cost_of_ecls, cost_of_ecls.data(), cost_of_ecls.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_cost_of_ende, cost_of_ende.data(), cost_of_ende.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(dvc_tree_ret, 0, 1*sizeof(double));
  cudaMemset(dvc_dag_ret, 0, 1*sizeof(double));
  cudaMemcpy(_dvc_best_ende_of_ecls, best_ende_of_ecls.data(), best_ende_of_ecls.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_dvc_update, &h_update, 1*sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(_dvc_ecls_its_ende_1D, _ecls_its_ende_1D.data(), _ecls_its_ende_1D.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_dvc_ecls_its_ende_1D_ptr, _ecls_its_ende_1D_ptr.data(), _ecls_its_ende_1D_ptr.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_dvc_root_eclass_id, _root_eclass_id.data(), _root_eclass_id.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemset(_dvc_ende_picked_by_ende, 0, _ende_child.size()*sizeof(uint32_t));
  cudaMemset(_dvc_picked_endes, 0, _num_ende*sizeof(uint32_t)); // false
  cudaMemset(_dvc_q, 0, _num_ende*sizeof(uint32_t));
  cudaMemset(_dvc_q_sz, 0, 1*sizeof(uint32_t));
  cudaMemset(_dvc_q_sp, 0, 1*sizeof(uint32_t));
  cudaMemset(_dvc_q_num, 0, 1*sizeof(uint32_t));
  cudaMemset(_dvc_indegree, 0, _num_ende*sizeof(uint32_t));
  cudaMemset(_dvc_inv_indegree, 0, _num_ende*sizeof(uint32_t));
  // new G
  cudaMemcpy(_dvc_global_adj_1D, _global_adj_1D.data(), _global_adj_1D.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_dvc_global_adj_1D_ptr, _global_adj_1D_ptr.data(), _global_adj_1D_ptr.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_dvc_global_inv_adj_1D, _global_inv_adj_1D.data(), _global_inv_adj_1D.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_dvc_global_inv_adj_1D_ptr, _global_inv_adj_1D_ptr.data(), _global_inv_adj_1D_ptr.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemset(_dvc_picked_adj, 0, _global_adj_1D.size()*sizeof(uint32_t));
  cudaMemset(_dvc_picked_inv_adj, 0, _global_inv_adj_1D.size()*sizeof(uint32_t));
  cudaMemset(_dvc_tmp_picked_adj, 0, _global_adj_1D.size()*sizeof(uint32_t));
  cudaMemset(_dvc_tmp_picked_inv_adj, 0, _global_inv_adj_1D.size()*sizeof(uint32_t));

  cudaMemcpy(_dvc_all_root_ende_1D, _all_root_ende_1D.data(), _all_root_ende_1D.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemset(_dvc_max_level, 0, 1*sizeof(int));
  cudaMemset(_dvc_in_subgraph, 0, _num_ende*sizeof(uint32_t)); // false
  cudaMemcpy(_dvc_roots, _roots.data(), _roots.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_dvc_leaves, _leaves.data(), _leaves.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemset(_dvc_from_roots, 0, _num_ende*sizeof(uint32_t)); // false
  cudaMemset(_dvc_from_leaves, 0, _num_ende*sizeof(uint32_t)); // false
  cudaCheckErrors("cudaMemcpy failed");


  _gpu_warm_up<<<blocks,kGpuThreads>>>(_num_ende, _dvc_level); // Note: place this GPU kernel here as the warm-up for GPU; w/o this, the kernel runtime below will be extremely long. Need to remrmber!

  float gpuTime = 0.;
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);    
  cudaEventRecord(startGPU);

  blocks = (_num_ende + kGpuThreads - 1) / kGpuThreads;
  while (h_update) {
    h_update = false;
    cudaMemcpy(_dvc_update, &h_update, sizeof(bool), cudaMemcpyHostToDevice);
    _gpu_tree_construction_kernel<<<blocks, kGpuThreads>>>(_num_ende,
      dvc_ende_cost,
      dvc_ende_ecls,
      dvc_ende_child,
      dvc_ende_child_ptr,
      _dvc_update,
      dvc_cost_of_ecls,
      dvc_cost_of_ende,
      _dvc_best_ende_of_ecls);
    cudaCheckErrors("_gpu_tree_construction_kernel failed");
    cudaMemcpy(&h_update, _dvc_update, sizeof(bool), cudaMemcpyDeviceToHost);
  }
  
  // Get the best tree cost on GPU
  _get_tree_cost<<<1, 1>>>(_root_eclass_id.size(), dvc_cost_of_ecls, _dvc_root_eclass_id, dvc_tree_ret);

  // Get best tree cost 
  double tree_ret = 0.0; 
  cudaMemcpy(&tree_ret, dvc_tree_ret, 1*sizeof(double), cudaMemcpyDeviceToHost);
  
  blocks = (_num_ecls + kGpuThreads - 1) / kGpuThreads;
  _resolve_best_ende<<<blocks, kGpuThreads>>>(
    _num_ecls,
    _dvc_ecls_its_ende_1D,
    _dvc_ecls_its_ende_1D_ptr,
    dvc_cost_of_ecls,
    dvc_cost_of_ende,
    _dvc_best_ende_of_ecls); 
  cudaCheckErrors("_resolve_best_ende failed");

  // Get the best dag cost on GPU
  _init_q<<<1,1>>>(_root_eclass_id.size(),
    _dvc_root_eclass_id,
    _dvc_best_ende_of_ecls,
    _dvc_q, _dvc_q_sz, _dvc_q_num,
    _dvc_picked_endes);    
  cudaCheckErrors("_init_q failed");

  double dag_ret = 0.0;

  uint32_t q_sz = 0;
  cudaMemcpy(&q_sz, _dvc_q_sz, 1*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  uint32_t q_sp = 0;
  uint32_t num = q_sz - q_sp;

  while (num > 0) {
    cudaMemcpy(_dvc_q_sp, &q_sp, sizeof(uint32_t), cudaMemcpyHostToDevice);
    blocks = (num + kGpuThreads - 1) / kGpuThreads;
    _get_dag_cost<<<blocks, kGpuThreads>>>(
      num,
      dvc_ende_cost,
      dvc_ende_child,
      dvc_ende_child_ptr,
      _dvc_best_ende_of_ecls,
      _dvc_q,
      _dvc_q_sz,
      _dvc_q_sp,
      _dvc_picked_endes,
      dvc_dag_ret);
    cudaCheckErrors("_get_dag_cost failed");
    cudaMemcpy(&q_sz, _dvc_q_sz, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    q_sp += num;
    num = q_sz - q_sp;
  }
  cudaMemcpy(&dag_ret, dvc_dag_ret, sizeof(double), cudaMemcpyDeviceToHost);
  cudaCheckErrors("DAG cost memcpy failed");

  cudaEventRecord(stopGPU);
  cudaEventSynchronize(stopGPU);
  cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);                                    
  printf("GPU tree construction runtime: %.2f (ms)\n", gpuTime);

  return std::make_pair(tree_ret, dag_ret);
}

// One GPU thread evaluates one e-node slot in each sequential
// transposed batch.
//
// selected_state:
//   0 = unselected
//   1 = stable selected
//   2 = tentative selected
//
// Candidate IDs are assigned by:
//
//   u = threadIdx.x * num_batches + batch_index
//
// The host launches this kernel only once. Each lane walks
// through its transposed candidate sequence:
//
//   u = lane * num_batches + batch_index
//
// There is currently no explicit synchronization boundary
// between consecutive batch_index values. Tentative state and
// level updates are published live and may be observed by other
// threads during the same insertion phase.
//
// Candidate evaluation treats both state 1 and state 2 as selected.
// Later batches can observe tentative nodes inserted by earlier
// batches in the same insertion phase.
//
// Levelization/finalization occurs only after all batches finish.
__global__ void _gpu_dag_insert_batch_kl(
  const uint32_t num_endes,
  const uint32_t num_batches,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  // The original implementation launched one kernel for every
  // batch_index. This loop performs the same batch sequence
  // inside one persistent single-block kernel.
  for (uint32_t batch_index = 0u;
       batch_index < num_batches;
       ++batch_index) {
    // All candidate-level early exits must leave only this
    // do-while block. They must not return from the kernel,
    // because every thread must reach the batch barrier below.
    do {
      // Full-transpose assignment:
      //
      //   contiguous mapping:
      //     u = batch_index * batch_size + lane
      //
      //   transposed mapping:
      //     u = lane * num_batches + batch_index
      //
      // This is identical to the previous host-launched
      // batch mapping.
      const uint32_t u =
        threadIdx.x * num_batches +
        batch_index;

      // Some lanes in the final transposed batch may not map
      // to a valid e-node. They still have to participate in
      // the batch-level __syncthreads().
      if (u >= num_endes) {
        break;
      }

      // Atomic load because selected_state is updated live by
      // other threads in this block.
      if (atomicAdd(
            &selected_state[u],
            0u) != 0u) {
        break;
      }

      uint32_t parent_position =
        inverse_adj_ptr[u];

      const uint32_t parent_end =
        inverse_adj_ptr[u + 1u];

      uint32_t child_position =
        forward_adj_ptr[u];

      const uint32_t child_end =
        forward_adj_ptr[u + 1u];

      bool has_selected_parent = false;
      bool has_selected_child = false;
      bool has_selected_overlap = false;

      int max_parent_level = -1;
      int min_child_level = INT_MAX;

      // Both CSR segments are sorted and unique.
      //
      // Merge them with two pointers so that:
      //   neighbor only in inverse CSR -> process as parent
      //   neighbor only in forward CSR -> process as child
      //   neighbor in both             -> exact overlap check
      //
      // Complexity:
      //   O(parent_degree + child_degree)
      while (parent_position < parent_end ||
             child_position < child_end) {
        const bool take_parent_only =
          child_position >= child_end ||
          (parent_position < parent_end &&
           inverse_adj[parent_position] <
             forward_adj[child_position]);

        const bool take_child_only =
          parent_position >= parent_end ||
          (child_position < child_end &&
           forward_adj[child_position] <
             inverse_adj[parent_position]);

        if (take_parent_only) {
          const uint32_t parent =
            inverse_adj[parent_position];

          const uint32_t parent_state =
            atomicAdd(
              &selected_state[parent],
              0u);

          if (parent_state != 0u) {
            const int parent_level =
              atomicAdd(
                &level[parent],
                0);

            if (parent_level >= 0) {
              has_selected_parent = true;

              if (parent_level >
                  max_parent_level) {
                max_parent_level =
                  parent_level;
              }
            }
          }

          ++parent_position;
          continue;
        }

        if (take_child_only) {
          const uint32_t child =
            forward_adj[child_position];

          const uint32_t child_state =
            atomicAdd(
              &selected_state[child],
              0u);

          if (child_state != 0u) {
            const int child_level =
              atomicAdd(
                &level[child],
                0);

            if (child_level >= 0) {
              has_selected_child = true;

              if (child_level <
                  min_child_level) {
                min_child_level =
                  child_level;
              }
            }
          }

          ++child_position;
          continue;
        }

        // The same e-node appears in both sorted segments.
        const uint32_t neighbor =
          inverse_adj[parent_position];

        const uint32_t neighbor_state =
          atomicAdd(
            &selected_state[neighbor],
            0u);

        if (neighbor_state != 0u) {
          const int neighbor_level =
            atomicAdd(
              &level[neighbor],
              0);

          if (neighbor_level >= 0) {
            has_selected_parent = true;
            has_selected_child = true;
            has_selected_overlap = true;

            if (neighbor_level >
                max_parent_level) {
              max_parent_level =
                neighbor_level;
            }

            if (neighbor_level <
                min_child_level) {
              min_child_level =
                neighbor_level;
            }
          }
        }

        ++parent_position;
        ++child_position;
      }

      if (has_selected_overlap) {
        break;
      }

      if (max_parent_level >=
          min_child_level) {
        break;
      }

      int proposed_level = -1;

      if (is_top_down_insertion) {
        if (!has_selected_parent) {
          break;
        }

        proposed_level =
          max_parent_level + 1;
      }
      else {
        if (!has_selected_child) {
          break;
        }

        proposed_level =
          min_child_level - 1;
      }

      if (proposed_level < 0) {
        break;
      }

      // Publish the proposed level before publishing state 2.
      //
      // Keep __threadfence() because another thread in the same
      // batch may observe state 2 before the end-of-batch
      // __syncthreads(). The level must already be visible when
      // state 2 becomes visible.
      level[u] = proposed_level;

      // __threadfence(); // Note: open this will have worse soln Q than close it. So we close it

      // Exactly one successful committer increments u0.
      if (atomicCAS(
            &selected_state[u],
            0u,
            2u) == 0u) {
        atomicAdd(
          u0,
          1u);
      }
    } while (false);

    // No explicit block barrier is used here. This preserves
    // the currently evaluated live optimistic behavior.
    //
    // Enabling this barrier changes the order in which
    // tentative candidates become observable and has been
    // observed to change solution quality.
    // __syncthreads(); // Note: open this will have worse soln Q than close it. So we close it
  }
}

void G_EXT::_gpu_dag_insert_batch(
  const bool is_top_down_insertion)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_global_adj_1D == nullptr ||
      _dvc_global_adj_1D_ptr == nullptr ||
      _dvc_global_inv_adj_1D == nullptr ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_batch requires "
      "initialized GPU DAG buffers");
  }

  // Keep the original batch-count calculation.
  //
  // kGpuDagBatchSize is both:
  //   1. the number of threads in the single block
  //   2. the number of candidate lanes per transposed batch
  const uint32_t num_batches =
    (_num_ende +
     kGpuDagBatchSize - 1u) /
    kGpuDagBatchSize;

  // Launch one single-block kernel for the entire insertion
  // phase. The kernel processes all batch_index values
  // sequentially and synchronizes after every batch.
  _gpu_dag_insert_batch_kl
    <<<1u, kGpuDagBatchSize>>>(
      _num_ende,
      num_batches,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_batch_kl failed");
}

// ============================================================
// Direct one-thread-per-e-node live optimistic insertion
//
// One CUDA thread evaluates exactly one e-node:
//
//   u = blockIdx.x * blockDim.x + threadIdx.x
//
// Unlike the batch insertion kernel, this kernel has no
// internal batch loop. The full e-node range is evaluated in one
// multi-block launch.
//
// selected_state:
//   0 = unselected
//   1 = stable selected
//   2 = tentative selected
//
// Both states 1 and 2 are treated as selected while evaluating
// neighbors. A successful candidate is published as state 2.
// The existing levelization/finalization flow later validates
// or rolls back the tentative insertion.
// ============================================================
__global__ void _gpu_dag_insert_direct_kl(
  const uint32_t num_endes,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t u =
    blockIdx.x * blockDim.x +
    threadIdx.x;

  if (u >= num_endes) {
    return;
  }

  // selected_state is updated live by other GPU threads.
  if (atomicAdd(
        &selected_state[u],
        0u) != 0u) {
    return;
  }

  uint32_t parent_position =
    inverse_adj_ptr[u];

  const uint32_t parent_end =
    inverse_adj_ptr[u + 1u];

  uint32_t child_position =
    forward_adj_ptr[u];

  const uint32_t child_end =
    forward_adj_ptr[u + 1u];

  bool has_selected_parent = false;
  bool has_selected_child = false;
  bool has_selected_overlap = false;

  int max_parent_level = -1;
  int min_child_level = INT_MAX;

  // The forward and inverse CSR segments are sorted and unique.
  //
  // Merge the two segments so parent/child processing and exact
  // overlap detection require:
  //
  //   O(parent_degree + child_degree)
  while (parent_position < parent_end ||
         child_position < child_end) {
    const bool take_parent_only =
      child_position >= child_end ||
      (parent_position < parent_end &&
       inverse_adj[parent_position] <
         forward_adj[child_position]);

    const bool take_child_only =
      parent_position >= parent_end ||
      (child_position < child_end &&
       forward_adj[child_position] <
         inverse_adj[parent_position]);

    if (take_parent_only) {
      const uint32_t parent =
        inverse_adj[parent_position];

      const uint32_t parent_state =
        atomicAdd(
          &selected_state[parent],
          0u);

      if (parent_state != 0u) {
        const int parent_level =
          atomicAdd(
            &level[parent],
            0);

        if (parent_level >= 0) {
          has_selected_parent = true;

          if (parent_level >
              max_parent_level) {
            max_parent_level =
              parent_level;
          }
        }
      }

      ++parent_position;
      continue;
    }

    if (take_child_only) {
      const uint32_t child =
        forward_adj[child_position];

      const uint32_t child_state =
        atomicAdd(
          &selected_state[child],
          0u);

      if (child_state != 0u) {
        const int child_level =
          atomicAdd(
            &level[child],
            0);

        if (child_level >= 0) {
          has_selected_child = true;

          if (child_level <
              min_child_level) {
            min_child_level =
              child_level;
          }
        }
      }

      ++child_position;
      continue;
    }

    // The same selected e-node appears in both the parent and
    // child CSR segments. Inserting u would immediately violate
    // the local overlap condition.
    const uint32_t neighbor =
      inverse_adj[parent_position];

    const uint32_t neighbor_state =
      atomicAdd(
        &selected_state[neighbor],
        0u);

    if (neighbor_state != 0u) {
      const int neighbor_level =
        atomicAdd(
          &level[neighbor],
          0);

      if (neighbor_level >= 0) {
        has_selected_parent = true;
        has_selected_child = true;
        has_selected_overlap = true;

        if (neighbor_level >
            max_parent_level) {
          max_parent_level =
            neighbor_level;
        }

        if (neighbor_level <
            min_child_level) {
          min_child_level =
            neighbor_level;
        }
      }
    }

    ++parent_position;
    ++child_position;
  }

  if (has_selected_overlap) {
    return;
  }

  if (max_parent_level >=
      min_child_level) {
    return;
  }

  int proposed_level = -1;

  if (is_top_down_insertion) {
    if (!has_selected_parent) {
      return;
    }

    proposed_level =
      max_parent_level + 1;
  }
  else {
    if (!has_selected_child) {
      return;
    }

    proposed_level =
      min_child_level - 1;
  }

  if (proposed_level < 0) {
    return;
  }

  // Match the current batch insertion live publication
  // behavior.
  //
  // No global synchronization or thread fence is introduced.
  // Existing levelization/finalization validates the resulting
  // tentative graph.
  level[u] = proposed_level;

  if (atomicCAS(
        &selected_state[u],
        0u,
        2u) == 0u) {
    atomicAdd(
      u0,
      1u);
  }
}

void G_EXT::_gpu_dag_insert_direct(
  const bool is_top_down_insertion)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_global_adj_1D == nullptr ||
      _dvc_global_adj_1D_ptr == nullptr ||
      _dvc_global_inv_adj_1D == nullptr ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_direct requires "
      "initialized GPU DAG buffers");
  }

  const uint32_t blocks =
    (_num_ende +
     kGpuThreads - 1u) /
    kGpuThreads;

  _gpu_dag_insert_direct_kl
    <<<blocks, kGpuThreads>>>(
      _num_ende,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_direct_kl failed");
}

// Construct the selected-subgraph degree and append all
// zero-degree selected e-nodes to the initial queue.
//
// selected_state:
//   0 = unselected
//   1 = stable selected
//   2 = tentative selected
//
// Any nonzero state participates in levelization.
__global__ void _gpu_dag_init_levelization_queue_kl(
  const uint32_t num_endes,
  const uint32_t* selected_state,
  const uint32_t* incoming_adj,
  const uint32_t* incoming_adj_ptr,
  uint32_t* degree,
  int* level,
  uint32_t* queue,
  uint32_t* queue_size)
{
  const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;

  if (u >= num_endes) {
    return;
  }

  level[u] = -1;

  if (selected_state[u] == 0u) {
    degree[u] = 0u;
    return;
  }

  uint32_t selected_degree = 0u;

  const uint32_t start = incoming_adj_ptr[u];
  const uint32_t end = incoming_adj_ptr[u + 1];

  for (uint32_t k = start; k < end; ++k) {
    const uint32_t neighbor = incoming_adj[k];

    if (selected_state[neighbor] != 0u) {
      ++selected_degree;
    }
  }

  degree[u] = selected_degree;

  if (selected_degree == 0u) {
    const uint32_t position =
      atomicAdd(queue_size, 1u);

    queue[position] = u;
  }
}

// Process exactly one queue slice:
//
//   queue[queue_begin,
//         queue_begin + queue_count)
//
// Newly discovered zero-degree nodes are appended after the
// current queue tail and are processed by the next launch.
__global__ void _gpu_dag_process_queue_slice_kl(
  const uint32_t queue_count,
  const uint32_t queue_begin,
  uint32_t* queue,
  uint32_t* queue_size,
  const uint32_t* selected_state,
  const uint32_t* traversal_adj,
  const uint32_t* traversal_adj_ptr,
  uint32_t* degree,
  int* level,
  const int current_level)
{
  const uint32_t index =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= queue_count) {
    return;
  }

  const uint32_t u =
    queue[queue_begin + index];

  level[u] = current_level;

  const uint32_t start = traversal_adj_ptr[u];
  const uint32_t end = traversal_adj_ptr[u + 1];

  for (uint32_t k = start; k < end; ++k) {
    const uint32_t v = traversal_adj[k];

    if (selected_state[v] == 0u) {
      continue;
    }

    const uint32_t old_degree =
      atomicSub(&degree[v], 1u);

    if (old_degree == 1u) {
      const uint32_t position =
        atomicAdd(queue_size, 1u);

      queue[position] = v;
    }
  }
}

// Process a sequence of complete Kahn frontiers inside one
// persistent single-block kernel.
//
// The initial frontier is:
//
//   queue[initial_queue_begin,
//         initial_queue_begin + initial_queue_count)
//
// Each frontier is a strict snapshot. Nodes appended while the
// current frontier is processed are not consumed until every
// thread finishes the current level and reaches __syncthreads().
//
// The kernel exits in one of three ways:
//
//   Completed:
//     No next frontier remains. max_level is final.
//
//   FrontierTooWide:
//     The next frontier has not been processed. The host may
//     resume from the returned queue_begin, queue_count, and
//     current_level with the original multi-block kernel.
//
//   InvalidQueueState:
//     An append-only queue invariant was violated.
__global__ void _gpu_dag_process_persistent_tail_kl(
  const uint32_t num_endes,
  const uint32_t initial_queue_begin,
  const uint32_t initial_queue_count,
  const int32_t initial_current_level,
  const uint32_t emergency_exit_threshold,
  uint32_t* queue,
  uint32_t* queue_size,
  const uint32_t* selected_state,
  const uint32_t* traversal_adj,
  const uint32_t* traversal_adj_ptr,
  uint32_t* degree,
  int* level,
  int* max_level,
  g_ext_detail::GpuDagPersistentState* persistent_state)
{
  __shared__ uint32_t frontier_begin;
  __shared__ uint32_t frontier_count;
  __shared__ int32_t current_level;
  __shared__ uint32_t invalid_queue_state;

  if (threadIdx.x == 0u) {
    frontier_begin = initial_queue_begin;
    frontier_count = initial_queue_count;
    current_level = initial_current_level;
    invalid_queue_state = 0u;
  }

  __syncthreads();

  while (true) {
    // All threads observe the same shared frontier_count, so all
    // threads take the same return branch.
    if (frontier_count == 0u) {
      if (threadIdx.x == 0u) {
        persistent_state->queue_begin =
          frontier_begin;

        persistent_state->queue_count =
          0u;

        persistent_state->current_level =
          current_level;

        persistent_state->exit_reason =
          g_ext_detail::GpuDagPersistentExitReason::Completed;

        *max_level = current_level;
      }

      return;
    }

    // Leave this frontier unprocessed. The host resumes from
    // exactly this queue range with the multi-block kernel.
    if (frontier_count >
        emergency_exit_threshold) {
      if (threadIdx.x == 0u) {
        persistent_state->queue_begin =
          frontier_begin;

        persistent_state->queue_count =
          frontier_count;

        persistent_state->current_level =
          current_level;

        persistent_state->exit_reason =
          g_ext_detail::GpuDagPersistentExitReason::
            FrontierTooWide;
      }

      return;
    }

    // One block may process more nodes than it has threads.
    //
    // With 512 threads:
    //   1...512 nodes   -> one node round
    //   513...1024      -> two node rounds
    for (uint32_t index = threadIdx.x;
         index < frontier_count;
         index += blockDim.x) {
      const uint32_t u =
        queue[frontier_begin + index];

      level[u] = current_level;

      const uint32_t start =
        traversal_adj_ptr[u];

      const uint32_t end =
        traversal_adj_ptr[u + 1u];

      for (uint32_t k = start; k < end; ++k) {
        const uint32_t v =
          traversal_adj[k];

        if (selected_state[v] == 0u) {
          continue;
        }

        const uint32_t old_degree =
          atomicSub(&degree[v], 1u);

        if (old_degree == 1u) {
          const uint32_t position =
            atomicAdd(queue_size, 1u);

          queue[position] = v;
        }
      }
    }

    // Every node and edge in the current frontier must finish
    // before thread 0 snapshots the newly appended queue tail.
    __syncthreads();

    if (threadIdx.x == 0u) {
      const uint32_t new_queue_size =
        *queue_size;

      const uint32_t next_queue_begin =
        frontier_begin + frontier_count;

      // The queue is append-only and every selected node can
      // enter it at most once.
      if (new_queue_size < next_queue_begin ||
          new_queue_size > num_endes) {
        invalid_queue_state = 1u;

        persistent_state->queue_begin =
          frontier_begin;

        persistent_state->queue_count =
          frontier_count;

        persistent_state->current_level =
          current_level;

        persistent_state->exit_reason =
          g_ext_detail::GpuDagPersistentExitReason::
            InvalidQueueState;
      }
      else {
        frontier_begin =
          next_queue_begin;

        frontier_count =
          new_queue_size - next_queue_begin;

        ++current_level;
      }
    }

    // Publish the next frontier snapshot to every thread.
    __syncthreads();

    if (invalid_queue_state != 0u) {
      return;
    }
  }
}

__global__ void _gpu_dag_reverse_level_kl(
  const uint32_t num_endes,
  const uint32_t* selected_state,
  int* level,
  const int num_levels)
{
  const uint32_t u =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (u >= num_endes ||
      selected_state[u] == 0u ||
      level[u] < 0) {
    return;
  }

  level[u] =
    num_levels - level[u] - 1;
}

// Finalize tentative nodes after one complete Kahn
// levelization pass.
//
// selected_state:
//   0 = unselected
//   1 = stable selected
//   2 = tentative selected
//
// A tentative node with residual degree greater than zero
// was not processed by Kahn levelization and is rolled back.
//
// A tentative node with zero residual degree survived and
// becomes a stable selected node.
__global__ void _gpu_dag_finalize_tentative_kl(
  const uint32_t num_endes,
  uint32_t* selected_state,
  const uint32_t* residual_degree,
  int* level,
  uint32_t* u0)
{
  const uint32_t u =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (u >= num_endes) {
    return;
  }

  if (selected_state[u] != 2u) {
    return;
  }

  if (residual_degree[u] == 0u) {
    selected_state[u] = 1u;
    return;
  }

  selected_state[u] = 0u;
  level[u] = -1;

  // u0 counts successful insertions in the current outer
  // iteration. A rolled-back tentative insertion must be
  // removed from that count.
  atomicSub(u0, 1u);
}

// Seed an append-only reachability queue.
//
// Only selected start nodes participate. atomicCAS on
// visited guarantees that each e-node is appended at most once,
// even if the start list contains duplicates.
__global__ void _gpu_reachability_seed_kl(
  const uint32_t num_endes,
  const uint32_t num_starts,
  const uint32_t* starts,
  const uint32_t* selected_state,
  uint32_t* visited,
  uint32_t* queue,
  uint32_t* queue_size)
{
  const uint32_t index =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_starts) {
    return;
  }

  const uint32_t u = starts[index];

  if (u >= num_endes) {
    return;
  }

  if (selected_state[u] == 0u) {
    return;
  }

  if (atomicCAS(
        &visited[u],
        0u,
        1u) == 0u) {
    const uint32_t position =
      atomicAdd(queue_size, 1u);

    queue[position] = u;
  }
}


// Process one append-only reachability queue slice:
//
//   queue[queue_begin,
//         queue_begin + queue_count)
//
// Newly discovered selected nodes are appended after the
// current queue tail. They are processed by the next launch.
//
// selected_state is immutable during root-to-leaf filtering,
// so ordinary device loads are sufficient.
__global__ void _gpu_reachability_process_queue_slice_kl(
  const uint32_t queue_count,
  const uint32_t queue_begin,
  uint32_t* queue,
  uint32_t* queue_size,
  const uint32_t* selected_state,
  const uint32_t* adjacency,
  const uint32_t* adjacency_ptr,
  uint32_t* visited)
{
  const uint32_t index =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= queue_count) {
    return;
  }

  const uint32_t u =
    queue[queue_begin + index];

  const uint32_t start =
    adjacency_ptr[u];

  const uint32_t end =
    adjacency_ptr[u + 1u];

  for (uint32_t k = start; k < end; ++k) {
    const uint32_t v = adjacency[k];

    if (selected_state[v] == 0u) {
      continue;
    }

    // Only the first thread that changes visited[v]
    // from 0 to 1 may append v to the queue.
    if (atomicCAS(
          &visited[v],
          0u,
          1u) == 0u) {
      const uint32_t position =
        atomicAdd(queue_size, 1u);

      queue[position] = v;
    }
  }
}


// Match the CPU get_nodes_in_subgraph() semantics.
//
// Do not modify selected_state. The final filtered graph is
// represented separately by in_subgraph.
__global__ void _gpu_build_in_subgraph_kl(
  const uint32_t num_endes,
  const uint32_t* selected_state,
  const uint32_t* from_roots,
  const uint32_t* from_leaves,
  uint32_t* in_subgraph)
{
  const uint32_t u =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (u >= num_endes) {
    return;
  }

  in_subgraph[u] =
    selected_state[u] != 0u &&
    from_roots[u] != 0u &&
    from_leaves[u] != 0u
      ? 1u
      : 0u;
}

void G_EXT::_gpu_dag_levelize(
  const bool is_top_down)
{
  if constexpr (kEnableGpuDagHybridLevelization) {
    _gpu_dag_levelize_hybrid(is_top_down);
  }
  else {
    _gpu_dag_levelize_host_driven(is_top_down);
  }
}

void G_EXT::_gpu_dag_levelize_host_driven(
  const bool is_top_down)
{
  // Top-down:
  //   incoming dependencies = selected parents
  //   traversal = parent -> child
  //
  // Bottom-up:
  //   incoming dependencies = selected children
  //   traversal = child -> parent

  const uint32_t* incoming_adj =
    is_top_down
      ? _dvc_global_inv_adj_1D
      : _dvc_global_adj_1D;

  const uint32_t* incoming_adj_ptr =
    is_top_down
      ? _dvc_global_inv_adj_1D_ptr
      : _dvc_global_adj_1D_ptr;

  const uint32_t* traversal_adj =
    is_top_down
      ? _dvc_global_adj_1D
      : _dvc_global_inv_adj_1D;

  const uint32_t* traversal_adj_ptr =
    is_top_down
      ? _dvc_global_adj_1D_ptr
      : _dvc_global_inv_adj_1D_ptr;

  uint32_t* degree =
    is_top_down
      ? _dvc_indegree
      : _dvc_inv_indegree;

  // Reuse the append-only queue allocated by tree construction.
  // Queue contents do not need to be cleared.
  CUDA_CHECK(cudaMemset(
    _dvc_q_sz,
    0,
    sizeof(uint32_t)));

  const uint32_t node_blocks =
    (_num_ende + kGpuThreads - 1) / kGpuThreads;

  _gpu_dag_init_levelization_queue_kl
    <<<node_blocks, kGpuThreads>>>(
      _num_ende,
      _dvc_picked_endes,
      incoming_adj,
      incoming_adj_ptr,
      degree,
      _dvc_level,
      _dvc_q,
      _dvc_q_sz);

  cudaCheckErrors(
    "_gpu_dag_init_levelization_queue_kl failed");

  uint32_t queue_size = 0u;

  CUDA_CHECK(cudaMemcpy(
    &queue_size,
    _dvc_q_sz,
    sizeof(uint32_t),
    cudaMemcpyDeviceToHost));

  uint32_t queue_begin = 0u;
  uint32_t queue_count = queue_size;
  int current_level = 0;

  while (queue_count > 0u) {
    const uint32_t queue_blocks =
      (queue_count + kGpuThreads - 1) / kGpuThreads;

    _gpu_dag_process_queue_slice_kl
      <<<queue_blocks, kGpuThreads>>>(
        queue_count,
        queue_begin,
        _dvc_q,
        _dvc_q_sz,
        _dvc_picked_endes,
        traversal_adj,
        traversal_adj_ptr,
        degree,
        _dvc_level,
        current_level);

    cudaCheckErrors(
      "_gpu_dag_process_queue_slice_kl failed");

    uint32_t new_queue_size = 0u;

    CUDA_CHECK(cudaMemcpy(
      &new_queue_size,
      _dvc_q_sz,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    if (new_queue_size > _num_ende) {
      throw std::logic_error(
        "GPU DAG levelization queue exceeded number of e-nodes");
    }

    queue_begin += queue_count;
    queue_size = new_queue_size;
    queue_count = queue_size - queue_begin;
    ++current_level;
  }

  CUDA_CHECK(cudaMemcpy(
    _dvc_max_level,
    &current_level,
    sizeof(int),
    cudaMemcpyHostToDevice));

  if (!is_top_down && current_level > 0) {
    _gpu_dag_reverse_level_kl
      <<<node_blocks, kGpuThreads>>>(
        _num_ende,
        _dvc_picked_endes,
        _dvc_level,
        current_level);

    cudaCheckErrors(
      "_gpu_dag_reverse_level_kl failed");
  }
}

void G_EXT::_gpu_dag_levelize_hybrid(
  const bool is_top_down)
{
  if (_dvc_dag_persistent_state == nullptr) {
    throw std::logic_error(
      "_gpu_dag_levelize_hybrid requires "
      "an allocated persistent-state buffer");
  }

  // Top-down:
  //   incoming dependencies = selected parents
  //   traversal = parent -> child
  //
  // Bottom-up:
  //   incoming dependencies = selected children
  //   traversal = child -> parent

  const uint32_t* incoming_adj =
    is_top_down
      ? _dvc_global_inv_adj_1D
      : _dvc_global_adj_1D;

  const uint32_t* incoming_adj_ptr =
    is_top_down
      ? _dvc_global_inv_adj_1D_ptr
      : _dvc_global_adj_1D_ptr;

  const uint32_t* traversal_adj =
    is_top_down
      ? _dvc_global_adj_1D
      : _dvc_global_inv_adj_1D;

  const uint32_t* traversal_adj_ptr =
    is_top_down
      ? _dvc_global_adj_1D_ptr
      : _dvc_global_inv_adj_1D_ptr;

  uint32_t* degree =
    is_top_down
      ? _dvc_indegree
      : _dvc_inv_indegree;

  // Reuse the append-only queue allocated by tree construction.
  // Queue contents do not need to be cleared.
  CUDA_CHECK(cudaMemset(
    _dvc_q_sz,
    0,
    sizeof(uint32_t)));

  const uint32_t node_blocks =
    (_num_ende + kGpuThreads - 1u) /
    kGpuThreads;

  // Degree construction and initial zero-degree queue remain
  // multi-block. This is unchanged from the original version.
  _gpu_dag_init_levelization_queue_kl
    <<<node_blocks, kGpuThreads>>>(
      _num_ende,
      _dvc_picked_endes,
      incoming_adj,
      incoming_adj_ptr,
      degree,
      _dvc_level,
      _dvc_q,
      _dvc_q_sz);

  cudaCheckErrors(
    "_gpu_dag_init_levelization_queue_kl failed");

  uint32_t initial_queue_size = 0u;

  CUDA_CHECK(cudaMemcpy(
    &initial_queue_size,
    _dvc_q_sz,
    sizeof(uint32_t),
    cudaMemcpyDeviceToHost));

  if (initial_queue_size > _num_ende) {
    throw std::logic_error(
      "GPU DAG initial levelization queue exceeded "
      "the number of e-nodes");
  }

  uint32_t queue_begin = 0u;
  uint32_t queue_count = initial_queue_size;

  int current_level = 0;

  uint32_t consecutive_small_levels = 0u;

  // The persistent kernel writes _dvc_max_level only when it
  // completes all remaining frontiers. If the host finishes the
  // traversal, the host writes _dvc_max_level as before.
  bool device_max_level_is_final = false;

  while (queue_count > 0u) {
    if (queue_count <=
        kGpuDagPersistentEnterFrontierThreshold) {
      ++consecutive_small_levels;
    }
    else {
      consecutive_small_levels = 0u;
    }

    // Enter persistent mode on the Nth consecutive small
    // frontier. The current frontier has not yet been processed.
    if (consecutive_small_levels >=
        kGpuDagPersistentRequiredSmallLevels) {
      if constexpr (kEnableGpuDagIterationLog) {
        printf(
          "[GPU DAG Levelize] dir=%s, "
          "enter persistent, "
          "level=%d, frontier=%u, streak=%u\n",
          is_top_down ? "TD" : "BU",
          current_level,
          queue_count,
          consecutive_small_levels);
      }

      // Initial state is passed through kernel arguments, not
      // copied into the state buffer. This avoids one H2D copy.
      _gpu_dag_process_persistent_tail_kl
        <<<1u, kGpuDagPersistentThreads>>>(
          _num_ende,
          queue_begin,
          queue_count,
          static_cast<int32_t>(current_level),
          kGpuDagPersistentEmergencyExitThreshold,
          _dvc_q,
          _dvc_q_sz,
          _dvc_picked_endes,
          traversal_adj,
          traversal_adj_ptr,
          degree,
          _dvc_level,
          _dvc_max_level,
          _dvc_dag_persistent_state);

      cudaCheckErrors(
        "_gpu_dag_process_persistent_tail_kl failed");

      // One compact synchronization after the entire persistent
      // segment, rather than one synchronization per level.
      g_ext_detail::GpuDagPersistentState
        host_state{};

      CUDA_CHECK(cudaMemcpy(
        &host_state,
        _dvc_dag_persistent_state,
        sizeof(host_state),
        cudaMemcpyDeviceToHost));

      if (host_state.current_level < 0) {
        throw std::logic_error(
          "GPU DAG persistent levelization returned "
          "a negative current level");
      }

      if (static_cast<uint32_t>(
            host_state.current_level) >
          _num_ende) {
        throw std::logic_error(
          "GPU DAG persistent levelization returned "
          "too many levels");
      }

      if (host_state.queue_begin > _num_ende ||
          host_state.queue_count >
            _num_ende - host_state.queue_begin) {
        throw std::logic_error(
          "GPU DAG persistent levelization returned "
          "an invalid queue range");
      }

      queue_begin =
        host_state.queue_begin;

      queue_count =
        host_state.queue_count;

      current_level =
        static_cast<int>(
          host_state.current_level);

      if (host_state.exit_reason ==
          g_ext_detail::GpuDagPersistentExitReason::
            Completed) {
        if (queue_count != 0u) {
          throw std::logic_error(
            "Completed persistent levelization "
            "returned a non-empty frontier");
        }

        device_max_level_is_final = true;

        if constexpr (kEnableGpuDagIterationLog) {
          printf(
            "[GPU DAG Levelize] dir=%s, "
            "persistent completed, "
            "num_levels=%d\n",
            is_top_down ? "TD" : "BU",
            current_level);
        }

        break;
      }

      if (host_state.exit_reason ==
          g_ext_detail::GpuDagPersistentExitReason::
            FrontierTooWide) {
        if (queue_count <=
            kGpuDagPersistentEmergencyExitThreshold) {
          throw std::logic_error(
            "Persistent levelization reported a wide "
            "frontier below the emergency threshold");
        }

        if constexpr (kEnableGpuDagIterationLog) {
          printf(
            "[GPU DAG Levelize] dir=%s, "
            "persistent exit=frontier_too_wide, "
            "level=%d, frontier=%u\n",
            is_top_down ? "TD" : "BU",
            current_level,
            queue_count);
        }

        // The returned frontier has not been processed.
        // Process it using the original multi-block path.
        consecutive_small_levels = 0u;
        continue;
      }

      throw std::logic_error(
        "GPU DAG persistent levelization detected "
        "an invalid append-only queue state");
    }

    // ==========================================================
    // Original host-driven multi-block processing
    //
    // This block intentionally matches the original
    // _gpu_dag_levelize() implementation.
    // ==========================================================
    const uint32_t queue_blocks =
      (queue_count + kGpuThreads - 1u) /
      kGpuThreads;

    _gpu_dag_process_queue_slice_kl
      <<<queue_blocks, kGpuThreads>>>(
        queue_count,
        queue_begin,
        _dvc_q,
        _dvc_q_sz,
        _dvc_picked_endes,
        traversal_adj,
        traversal_adj_ptr,
        degree,
        _dvc_level,
        current_level);

    cudaCheckErrors(
      "_gpu_dag_process_queue_slice_kl failed");

    uint32_t new_queue_size = 0u;

    CUDA_CHECK(cudaMemcpy(
      &new_queue_size,
      _dvc_q_sz,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    const uint32_t processed_queue_end =
      queue_begin + queue_count;

    if (new_queue_size < processed_queue_end) {
      throw std::logic_error(
        "GPU DAG levelization append-only queue shrank");
    }

    if (new_queue_size > _num_ende) {
      throw std::logic_error(
        "GPU DAG levelization queue exceeded "
        "the number of e-nodes");
    }

    queue_begin =
      processed_queue_end;

    queue_count =
      new_queue_size - queue_begin;

    ++current_level;
  }

  // When traversal ended in the host-driven path, preserve the
  // original H2D max-level update.
  //
  // When the persistent kernel completed the traversal, it
  // already wrote the final value and this copy is skipped.
  if (!device_max_level_is_final) {
    CUDA_CHECK(cudaMemcpy(
      _dvc_max_level,
      &current_level,
      sizeof(int),
      cudaMemcpyHostToDevice));
  }

  // Bottom-up reverse remains the existing multi-block kernel.
  if (!is_top_down && current_level > 0) {
    _gpu_dag_reverse_level_kl
      <<<node_blocks, kGpuThreads>>>(
        _num_ende,
        _dvc_picked_endes,
        _dvc_level,
        current_level);

    cudaCheckErrors(
      "_gpu_dag_reverse_level_kl failed");
  }
}

void G_EXT::_gpu_dag_finalize_tentative(
  const bool is_top_down_levelization)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_indegree == nullptr ||
      _dvc_inv_indegree == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_finalize_tentative requires "
      "initialized GPU DAG buffers");
  }

  const uint32_t* residual_degree =
    is_top_down_levelization
      ? _dvc_indegree
      : _dvc_inv_indegree;

  const uint32_t blocks =
    (_num_ende + kGpuThreads - 1) / kGpuThreads;

  _gpu_dag_finalize_tentative_kl
    <<<blocks, kGpuThreads>>>(
      _num_ende,
      _dvc_picked_endes,
      residual_degree,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_finalize_tentative_kl failed");
}

void G_EXT::_gpu_mark_reachable(
  const uint32_t* starts,
  const uint32_t num_starts,
  const uint32_t* adjacency,
  const uint32_t* adjacency_ptr,
  uint32_t* visited)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_q == nullptr ||
      _dvc_q_sz == nullptr ||
      adjacency_ptr == nullptr ||
      visited == nullptr) {
    throw std::logic_error(
      "_gpu_mark_reachable requires initialized "
      "GPU graph and queue buffers");
  }

  // A null adjacency data pointer is legal when the graph has
  // no edges because every CSR segment is empty. adjacency_ptr
  // remains required.
  if (num_starts > 0u && starts == nullptr) {
    throw std::logic_error(
      "_gpu_mark_reachable received a null start buffer");
  }

  // Queue contents do not need to be cleared. Reset only the
  // append position.
  CUDA_CHECK(cudaMemset(
    _dvc_q_sz,
    0,
    sizeof(uint32_t)));

  if (num_starts == 0u) {
    return;
  }

  const uint32_t seed_blocks =
    (num_starts +
     kGpuReachabilityThreads - 1u) /
    kGpuReachabilityThreads;

  _gpu_reachability_seed_kl
    <<<seed_blocks, kGpuReachabilityThreads>>>(
      _num_ende,
      num_starts,
      starts,
      _dvc_picked_endes,
      visited,
      _dvc_q,
      _dvc_q_sz);

  cudaCheckErrors(
    "_gpu_reachability_seed_kl failed");

  uint32_t queue_size = 0u;

  CUDA_CHECK(cudaMemcpy(
    &queue_size,
    _dvc_q_sz,
    sizeof(uint32_t),
    cudaMemcpyDeviceToHost));

  if (queue_size > _num_ende) {
    throw std::logic_error(
      "GPU reachability seed queue exceeded "
      "the number of e-nodes");
  }

  uint32_t queue_begin = 0u;

  while (queue_begin < queue_size) {
    const uint32_t queue_count =
      queue_size - queue_begin;

    const uint32_t queue_blocks =
      (queue_count +
       kGpuReachabilityThreads - 1u) /
      kGpuReachabilityThreads;

    _gpu_reachability_process_queue_slice_kl
      <<<queue_blocks, kGpuReachabilityThreads>>>(
        queue_count,
        queue_begin,
        _dvc_q,
        _dvc_q_sz,
        _dvc_picked_endes,
        adjacency,
        adjacency_ptr,
        visited);

    cudaCheckErrors(
      "_gpu_reachability_process_queue_slice_kl failed");

    uint32_t new_queue_size = 0u;

    CUDA_CHECK(cudaMemcpy(
      &new_queue_size,
      _dvc_q_sz,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    if (new_queue_size < queue_size) {
      throw std::logic_error(
        "GPU reachability append-only queue shrank");
    }

    if (new_queue_size > _num_ende) {
      throw std::logic_error(
        "GPU reachability queue exceeded "
        "the number of e-nodes");
    }

    // Everything before the old queue tail has now been
    // processed. Newly appended entries form the next slice.
    queue_begin = queue_size;
    queue_size = new_queue_size;
  }
}

void G_EXT::_gpu_build_in_subgraph()
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_from_roots == nullptr ||
      _dvc_from_leaves == nullptr ||
      _dvc_in_subgraph == nullptr) {
    throw std::logic_error(
      "_gpu_build_in_subgraph requires initialized "
      "GPU filtering buffers");
  }

  const uint32_t blocks =
    (_num_ende +
     kGpuReachabilityThreads - 1u) /
    kGpuReachabilityThreads;

  _gpu_build_in_subgraph_kl
    <<<blocks, kGpuReachabilityThreads>>>(
      _num_ende,
      _dvc_picked_endes,
      _dvc_from_roots,
      _dvc_from_leaves,
      _dvc_in_subgraph);

  cudaCheckErrors(
    "_gpu_build_in_subgraph_kl failed");
}

void G_EXT::_gpu_filter_root_to_leaf()
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_global_adj_1D_ptr == nullptr ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_from_roots == nullptr ||
      _dvc_from_leaves == nullptr ||
      _dvc_in_subgraph == nullptr) {
    throw std::logic_error(
      "_gpu_filter_root_to_leaf requires initialized "
      "GPU graph and filtering buffers");
  }

  if (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) {
    throw std::logic_error(
      "GPU forward adjacency buffer is missing");
  }

  if (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) {
    throw std::logic_error(
      "GPU inverse adjacency buffer is missing");
  }

  if (!_roots.empty() &&
      _dvc_roots == nullptr) {
    throw std::logic_error(
      "GPU root buffer is missing");
  }

  if (!_leaves.empty() &&
      _dvc_leaves == nullptr) {
    throw std::logic_error(
      "GPU leaf buffer is missing");
  }

  // Each traversal owns a separate visited array.
  CUDA_CHECK(cudaMemset(
    _dvc_from_roots,
    0,
    static_cast<size_t>(_num_ende) *
      sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(
    _dvc_from_leaves,
    0,
    static_cast<size_t>(_num_ende) *
      sizeof(uint32_t)));

  // Forward traversal:
  //
  // selected roots -> selected descendants
  _gpu_mark_reachable(
    _dvc_roots,
    static_cast<uint32_t>(_roots.size()),
    _dvc_global_adj_1D,
    _dvc_global_adj_1D_ptr,
    _dvc_from_roots);

  // Reverse traversal:
  //
  // selected global leaves -> selected ancestors
  _gpu_mark_reachable(
    _dvc_leaves,
    static_cast<uint32_t>(_leaves.size()),
    _dvc_global_inv_adj_1D,
    _dvc_global_inv_adj_1D_ptr,
    _dvc_from_leaves);

  _gpu_build_in_subgraph();
}

void G_EXT::_gpu_allocate_dag_buffers()
{
  // Validate the shared buffers produced by GPU tree construction.
  if (_dvc_picked_endes == nullptr ||
      _dvc_q == nullptr ||
      _dvc_q_sz == nullptr ||
      _dvc_global_adj_1D_ptr == nullptr ||      
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_indegree == nullptr ||
      _dvc_inv_indegree == nullptr ||
      _dvc_level == nullptr ||
      _dvc_max_level == nullptr ||
      _dvc_in_subgraph == nullptr ||
      _dvc_from_roots == nullptr ||
      _dvc_from_leaves == nullptr) {
    throw std::logic_error(
      "GPU DAG construction requires initialized GPU tree/shared buffers");
  }

  if (!_roots.empty() && _dvc_roots == nullptr) {
    throw std::logic_error(
      "GPU root buffer was not initialized by GPU tree construction");
  }

  if (!_leaves.empty() && _dvc_leaves == nullptr) {
    throw std::logic_error(
      "GPU leaf buffer was not initialized by GPU tree construction");
  }

  // Safe if DAG allocation is requested more than once.
  // This function only frees DAG-owned scratch memory.
  _gpu_free_dag_buffers();

  cuda_malloc_array(_dvc_dag_new_inserted, _num_ende);
  cuda_malloc_array(_dvc_dag_insert_decision, _num_ende);
  cuda_malloc_array(_dvc_dag_candidate_level, _num_ende);
  cuda_malloc_array(_dvc_dag_u0, 1);

  cuda_malloc_array(
    _dvc_dag_persistent_state,
    1u);
}

void G_EXT::_gpu_initialize_dag_buffers()
{
  if (_dvc_dag_new_inserted == nullptr ||
      _dvc_dag_insert_decision == nullptr ||
      _dvc_dag_candidate_level == nullptr ||
      _dvc_dag_u0 == nullptr ||
      _dvc_dag_persistent_state == nullptr)
  {
    throw std::logic_error(
      "_gpu_initialize_dag_buffers called before DAG allocation");
  }

  // Reinitialize shared working state.
  //
  // Do not modify _dvc_picked_endes:
  // it contains the tree solution produced by Algorithm 1.
  CUDA_CHECK(cudaMemset(_dvc_level, 0xFF,
    static_cast<size_t>(_num_ende) * sizeof(int)));

  CUDA_CHECK(cudaMemset(_dvc_indegree, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_inv_indegree, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_max_level, 0,
    sizeof(int)));

  // Initialize DAG-only insertion state.
  CUDA_CHECK(cudaMemset(_dvc_dag_new_inserted, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_dag_insert_decision, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_dag_candidate_level, 0xFF,
    static_cast<size_t>(_num_ende) * sizeof(int)));

  CUDA_CHECK(cudaMemset(_dvc_dag_u0, 0,
    sizeof(uint32_t)));
  
  CUDA_CHECK(cudaMemset(
    _dvc_dag_persistent_state,
    0,
    sizeof(g_ext_detail::GpuDagPersistentState)));    

  // Reuse the filtering buffers already allocated by tree construction.
  CUDA_CHECK(cudaMemset(_dvc_from_roots, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_from_leaves, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_in_subgraph, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));
}

void G_EXT::_gpu_free_dag_buffers() noexcept
{
  cuda_free_and_null(_dvc_dag_persistent_state);
  cuda_free_and_null(_dvc_dag_u0);
  cuda_free_and_null(_dvc_dag_candidate_level);
  cuda_free_and_null(_dvc_dag_insert_decision);
  cuda_free_and_null(_dvc_dag_new_inserted);
}

void G_EXT::_gpu_DAG_construction(
  const GpuDagInsertionPolicy insertion_policy)
{
  // ============================================================
  // Total DAG timing
  //
  // Total runtime is always measured and printed, regardless of
  // kEnableGpuDagIterationLog, because it is the primary metric
  // used for performance optimization.
  // ============================================================
  float gpu_dag_total_time_ms = 0.0f;

  cudaEvent_t start_gpu = nullptr;
  cudaEvent_t stop_gpu = nullptr;

  CUDA_CHECK(cudaEventCreate(&start_gpu));
  CUDA_CHECK(cudaEventCreate(&stop_gpu));

  // ============================================================
  // Optional detailed timing
  //
  // These objects are used only when detailed GPU DAG logging
  // is enabled. [[maybe_unused]] is a compile-time annotation and
  // has no runtime overhead.
  // ============================================================
  [[maybe_unused]] float gpu_dag_core_time_ms = 0.0f;
  [[maybe_unused]] float gpu_filter_time_ms = 0.0f;

  [[maybe_unused]] cudaEvent_t after_core_gpu = nullptr;

  if constexpr (kEnableGpuDagIterationLog) {
    CUDA_CHECK(cudaEventCreate(&after_core_gpu));
  }

  // Start total DAG construction timing.
  CUDA_CHECK(cudaEventRecord(start_gpu));

  // ============================================================
  // Algorithm 2 initialization
  //
  // Initial top-down levelization of the GPU tree result.
  // States 1 and 2 are both treated as selected.
  // At this point only states 0 and 1 should exist.
  // ============================================================
  _gpu_dag_levelize(true);

  uint32_t host_u0 = 1u;

  // Total number of completed bottom-up + top-down outer
  // iterations.
  uint32_t outer_iteration = 0u;

  // Number of completed iterations using each insertion mode.
  uint32_t batch_iterations = 0u;
  uint32_t direct_iterations = 0u;

  // Largest survived insertion count observed during batch
  // insertion.
  uint32_t peak_survived = 0u;

  // Number of consecutive batch iterations whose survived
  // insertion count is at most kGpuDagLowActivityPercent of
  // the historical batch peak.
  uint32_t low_activity_streak = 0u;

  // false:
  //   use the transposed-batch insertion kernel.
  //
  // true:
  //   use the one-thread-per-e-node direct insertion kernel.
  bool use_direct_mode = false;

  // BatchEarlyStop terminates immediately when the
  // low-activity condition is reached.
  bool batch_early_stop_triggered = false;

  // BatchThenDirect records the Batch -> Direct transition.
  bool switched_to_direct = false;
  uint32_t switch_iteration = 0u;
  uint32_t switch_survived = 0u;

  while (host_u0 > 0u) {
    // Keep one insertion mode for both the bottom-up and
    // top-down phases of this complete outer iteration.
    const bool iteration_uses_direct =
      use_direct_mode;

    // u0 counts the insertions that survive during the current
    // outer iteration.
    CUDA_CHECK(cudaMemset(
      _dvc_dag_u0,
      0,
      sizeof(uint32_t)));

    // ==========================================================
    // Bottom-up insertion
    // ==========================================================
    if (iteration_uses_direct) {
      _gpu_dag_insert_direct(false);
    }
    else {
      _gpu_dag_insert_batch(false);
    }

    // This value is required only for detailed logging.
    [[maybe_unused]] uint32_t bottom_up_proposed = 0u;

    if constexpr (kEnableGpuDagIterationLog) {
      CUDA_CHECK(cudaMemcpy(
        &bottom_up_proposed,
        _dvc_dag_u0,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost));
    }

    _gpu_dag_levelize(false);
    _gpu_dag_finalize_tentative(false);

    // This value is required only for detailed logging.
    [[maybe_unused]] uint32_t bottom_up_survived = 0u;

    if constexpr (kEnableGpuDagIterationLog) {
      CUDA_CHECK(cudaMemcpy(
        &bottom_up_survived,
        _dvc_dag_u0,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost));
    }

    // ==========================================================
    // Top-down insertion
    // ==========================================================
    if (iteration_uses_direct) {
      _gpu_dag_insert_direct(true);
    }
    else {
      _gpu_dag_insert_batch(true);
    }

    // This value is required only for detailed logging.
    [[maybe_unused]] uint32_t after_top_down_insert = 0u;

    if constexpr (kEnableGpuDagIterationLog) {
      CUDA_CHECK(cudaMemcpy(
        &after_top_down_insert,
        _dvc_dag_u0,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost));
    }

    _gpu_dag_levelize(true);
    _gpu_dag_finalize_tentative(true);

    // ==========================================================
    // Required outer-loop termination copy
    //
    // This copy is part of the algorithm, not merely logging.
    // It must remain enabled even when iteration logging is off.
    // ==========================================================
    uint32_t after_both_phases = 0u;

    CUDA_CHECK(cudaMemcpy(
      &after_both_phases,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    host_u0 = after_both_phases;
    ++outer_iteration;

    if (iteration_uses_direct) {
      ++direct_iterations;
    }
    else {
      ++batch_iterations;
    }

    // ==========================================================
    // Optional per-iteration diagnostics
    // ==========================================================
    if constexpr (kEnableGpuDagIterationLog) {
      const uint32_t top_down_proposed =
        after_top_down_insert -
        bottom_up_survived;

      const uint32_t top_down_survived =
        after_both_phases -
        bottom_up_survived;

      const uint32_t bottom_up_rollback =
        bottom_up_proposed -
        bottom_up_survived;

      const uint32_t top_down_rollback =
        top_down_proposed -
        top_down_survived;

      printf(
        "[GPU DAG] iteration=%u, mode=%s, "
        "BU proposed=%u survived=%u rollback=%u, "
        "TD proposed=%u survived=%u rollback=%u, "
        "total survived=%u\n",
        outer_iteration,
        iteration_uses_direct
          ? "direct"
          : "batch",
        bottom_up_proposed,
        bottom_up_survived,
        bottom_up_rollback,
        top_down_proposed,
        top_down_survived,
        top_down_rollback,
        after_both_phases);
    }

    // ==========================================================
    // Batch low-activity controller
    //
    // This controller is evaluated only while the current
    // iteration uses the batch insertion kernel.
    //
    // FullBatch:
    //   Ignore the low-activity condition.
    //
    // BatchEarlyStop:
    //   Stop DAG construction when the condition is reached.
    //
    // BatchThenDirect:
    //   Switch the next outer iteration to direct insertion.
    //   The current iteration has already completed both
    //   levelization and finalization phases.
    //
    // after_both_phases is already copied for the original
    // u0 == 0 convergence condition, so no additional D2H
    // copy is introduced.
    // ==========================================================
    if (!iteration_uses_direct) {
      if (after_both_phases > peak_survived) {
        peak_survived = after_both_phases;
      }

      if (insertion_policy !=
          GpuDagInsertionPolicy::FullBatch) {
        const bool is_low_activity =
          after_both_phases > 0u &&
          static_cast<uint64_t>(
            after_both_phases) * 100u <=
          static_cast<uint64_t>(
            peak_survived) *
            kGpuDagLowActivityPercent;

        if (is_low_activity) {
          ++low_activity_streak;
        }
        else {
          low_activity_streak = 0u;
        }

        const bool low_activity_condition_reached =
          outer_iteration >=
            kGpuDagLowActivityMinimumIterations &&
          low_activity_streak >=
            kGpuDagLowActivityPatience;

        if (low_activity_condition_reached) {
          if (insertion_policy ==
              GpuDagInsertionPolicy::BatchEarlyStop) {
            batch_early_stop_triggered = true;
            break;
          }

          // BatchThenDirect:
          //
          // Switch only after one complete batch iteration.
          // The next outer iteration begins in direct mode.
          use_direct_mode = true;
          switched_to_direct = true;
          switch_iteration = outer_iteration;
          switch_survived = after_both_phases;

          if constexpr (kEnableGpuDagIterationLog) {
            printf(
              "[GPU DAG Controller] "
              "switch batch->direct after iteration=%u, "
              "survived=%u, peak=%u, "
              "low_activity_streak=%u\n",
              switch_iteration,
              switch_survived,
              peak_survived,
              low_activity_streak);
          }
        }
      }
    }
  }

  
  // ============================================================
  // End of DAG construction core
  //
  // Record the core/filter boundary only when detailed logging
  // is enabled.
  // ============================================================
  if constexpr (kEnableGpuDagIterationLog) {
    CUDA_CHECK(cudaEventRecord(after_core_gpu));
  }

  // Match the CPU post-processing semantics:
  //
  // Keep _dvc_picked_endes unchanged and separately mark selected
  // e-nodes lying on a selected root-to-global-leaf path.
  _gpu_filter_root_to_leaf();

  // ============================================================
  // Stop total DAG timing
  // ============================================================
  CUDA_CHECK(cudaEventRecord(stop_gpu));
  CUDA_CHECK(cudaEventSynchronize(stop_gpu));

  // Total runtime is always calculated.
  CUDA_CHECK(cudaEventElapsedTime(
    &gpu_dag_total_time_ms,
    start_gpu,
    stop_gpu));

  // Detailed core/filter timing is calculated only when enabled.
  if constexpr (kEnableGpuDagIterationLog) {
    CUDA_CHECK(cudaEventElapsedTime(
      &gpu_dag_core_time_ms,
      start_gpu,
      after_core_gpu));

    CUDA_CHECK(cudaEventElapsedTime(
      &gpu_filter_time_ms,
      after_core_gpu,
      stop_gpu));

    printf(
      "GPU DAG core runtime: %.2f (ms)\n",
      gpu_dag_core_time_ms);

    printf(
      "GPU root-to-leaf filtering runtime: %.2f (ms)\n",
      gpu_filter_time_ms);
  }

  const uint32_t low_activity_threshold =
    static_cast<uint32_t>(
      (static_cast<uint64_t>(
         peak_survived) *
       kGpuDagLowActivityPercent) /
      100u);

  const char* insertion_policy_name = "unknown";

  switch (insertion_policy) {
    case GpuDagInsertionPolicy::FullBatch:
      insertion_policy_name = "full-batch";
      break;

    case GpuDagInsertionPolicy::BatchEarlyStop:
      insertion_policy_name = "batch-early-stop";
      break;

    case GpuDagInsertionPolicy::BatchThenDirect:
      insertion_policy_name = "batch-then-direct";
      break;
  }

  printf(
    "[GPU DAG Outer Loop] "
    "policy=%s, "
    "iterations=%u, "
    "batch_iterations=%u, "
    "direct_iterations=%u, "
    "reason=%s, "
    "peak=%u, "
    "last=%u, "
    "low_activity_threshold=%u, "
    "low_activity_streak=%u, "
    "switched=%s, "
    "switch_iteration=%u, "
    "switch_survived=%u\n",
    insertion_policy_name,
    outer_iteration,
    batch_iterations,
    direct_iterations,
    batch_early_stop_triggered
      ? "low-activity-stop"
      : "converged",
    peak_survived,
    host_u0,
    low_activity_threshold,
    low_activity_streak,
    switched_to_direct
      ? "yes"
      : "no",
    switch_iteration,
    switch_survived);

  // Total runtime is always printed.
  printf(
    "GPU DAG total runtime: %.2f (ms)\n",
    gpu_dag_total_time_ms);

  // ============================================================
  // Destroy timing events
  // ============================================================
  if constexpr (kEnableGpuDagIterationLog) {
    CUDA_CHECK(cudaEventDestroy(after_core_gpu));
  }

  CUDA_CHECK(cudaEventDestroy(start_gpu));
  CUDA_CHECK(cudaEventDestroy(stop_gpu));
}

G_EXT::CsrGraphSnapshot
G_EXT::unittest_get_csr_snapshot() const
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_get_csr_snapshot requires "
      "read() to be called first");
  }

  if (_global_adj_1D_ptr.size()
        != static_cast<size_t>(_num_ende) + 1 ||
      _global_inv_adj_1D_ptr.size()
        != static_cast<size_t>(_num_ende) + 1) {
    throw std::logic_error(
      "CSR pointer arrays are not initialized");
  }

  CsrGraphSnapshot result;

  result.forward_adj =
    _global_adj_1D;

  result.forward_ptr =
    _global_adj_1D_ptr;

  result.inverse_adj =
    _global_inv_adj_1D;

  result.inverse_ptr =
    _global_inv_adj_1D_ptr;

  return result;
}

G_EXT::GpuInsertionResult
G_EXT::unittest_gpu_insert_batch_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_insert_batch_once requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  if (input_level.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "level size does not match "
      "the number of e-nodes");
  }

  for (uint32_t ende = 0;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }
  }

  // GPU tree construction currently owns allocation of
  // the shared immutable CSR and working buffers.
  if (_dvc_picked_endes == nullptr) {
    (void)_gpu_tree_construction();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
      _dvc_level,
      input_level.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
      _dvc_dag_u0,
      0,
      sizeof(uint32_t)));

    _gpu_dag_insert_batch(
      is_top_down_insertion);

    GpuInsertionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

G_EXT::GpuInsertionResult
G_EXT::unittest_gpu_insert_direct_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_insert_direct_once requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  if (input_level.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "level size does not match "
      "the number of e-nodes");
  }

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }
  }

  // GPU tree construction owns allocation of the shared
  // immutable CSR and working buffers.
  if (_dvc_picked_endes == nullptr) {
    (void)_gpu_tree_construction();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
      _dvc_level,
      input_level.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
      _dvc_dag_u0,
      0,
      sizeof(uint32_t)));

    _gpu_dag_insert_direct(
      is_top_down_insertion);

    GpuInsertionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

G_EXT::GpuFinalizeResult
G_EXT::unittest_gpu_levelize_and_finalize(
  const std::vector<uint32_t>& selected_state,
  const bool is_top_down_levelization,
  const uint32_t initial_u0)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_levelize_and_finalize "
      "requires read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  uint32_t tentative_count = 0u;

  for (uint32_t ende = 0;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }

    if (selected_state[ende] == 2u) {
      ++tentative_count;
    }
  }

  // Every rolled-back tentative node decrements u0 once.
  // Therefore initial_u0 must be large enough to represent
  // all currently tentative nodes.
  if (initial_u0 < tentative_count) {
    throw std::invalid_argument(
      "initial_u0 must be at least the number "
      "of tentative state-2 nodes");
  }

  // GPU tree construction currently owns allocation of
  // the immutable CSR and shared working buffers.
  if (_dvc_picked_endes == nullptr) {
    (void)_gpu_tree_construction();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
      _dvc_dag_u0,
      &initial_u0,
      sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    _gpu_dag_levelize(
      is_top_down_levelization);

    _gpu_dag_finalize_tentative(
      is_top_down_levelization);

    GpuFinalizeResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.residual_degree.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    const uint32_t* degree_device =
      is_top_down_levelization
        ? _dvc_indegree
        : _dvc_inv_indegree;

    CUDA_CHECK(cudaMemcpy(
      result.residual_degree.data(),
      degree_device,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.num_levels,
      _dvc_max_level,
      sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

G_EXT::GpuDagConstructionResult
G_EXT::unittest_gpu_construct_dag(
  const std::vector<uint32_t>& selected_state)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_construct_dag requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  // Full DAG construction starts from the stable tree
  // produced by Algorithm 1. State 2 must not be present
  // before the first insertion phase.
  for (uint32_t ende = 0;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 1u) {
      throw std::invalid_argument(
        "initial DAG construction state must "
        "contain only 0 or 1");
    }
  }

  // GPU tree construction currently owns allocation of
  // immutable CSR and shared GPU working buffers.
  if (_dvc_picked_endes == nullptr) {
    (void)_gpu_tree_construction();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    // Preserve the original full-batch, full-convergence
    // behavior expected by the existing DAG correctness tests.
    _gpu_DAG_construction(
      GpuDagInsertionPolicy::FullBatch);

    GpuDagConstructionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.residual_degree.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    result.in_subgraph.resize(
      _num_ende,
      0u);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    // _gpu_DAG_construction() finishes with top-down
    // levelization, so _dvc_indegree contains the final
    // top-down residual degree.
    CUDA_CHECK(cudaMemcpy(
      result.residual_degree.data(),
      _dvc_indegree,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.in_subgraph.data(),
      _dvc_in_subgraph,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.num_levels,
      _dvc_max_level,
      sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.final_u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

G_EXT::GpuDagConstructionResult
G_EXT::unittest_gpu_construct_dag_from_tree_with_policy(
  const GpuDagInsertionPolicy insertion_policy)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_construct_dag_from_tree_with_policy "
      "requires read() to be called first");
  }

  // Reject invalid enum values before entering the production
  // controller. This validation exists only in the unit-test
  // wrapper and does not change the production path.
  switch (insertion_policy) {
    case GpuDagInsertionPolicy::FullBatch:
    case GpuDagInsertionPolicy::BatchEarlyStop:
    case GpuDagInsertionPolicy::BatchThenDirect:
      break;

    default:
      throw std::invalid_argument(
        "Invalid GPU DAG insertion policy");
  }

  // This wrapper must start from an unmodified GPU tree.
  //
  // Reusing an object after another GPU unit-test operation
  // could leave _dvc_picked_endes containing a modified DAG
  // rather than the original tree.
  if (_dvc_picked_endes != nullptr) {
    throw std::logic_error(
      "unittest_gpu_construct_dag_from_tree_with_policy "
      "requires a fresh G_EXT object");
  }

  // Build the same GPU tree used by the production run().
  //
  // _gpu_initialize_dag_buffers() intentionally does not
  // modify _dvc_picked_endes, so the tree selection remains
  // the initial state of DAG construction.
  (void)_gpu_tree_construction();

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    // Execute the real production controller with the policy
    // explicitly selected by the unit test.
    _gpu_DAG_construction(
      insertion_policy);

    GpuDagConstructionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.residual_degree.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    result.in_subgraph.resize(
      _num_ende,
      0u);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    // _gpu_DAG_construction() finishes with top-down
    // levelization, so _dvc_indegree contains the final
    // top-down residual degree.
    CUDA_CHECK(cudaMemcpy(
      result.residual_degree.data(),
      _dvc_indegree,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.in_subgraph.data(),
      _dvc_in_subgraph,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.num_levels,
      _dvc_max_level,
      sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.final_u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

G_EXT::GpuRootToLeafFilterResult
G_EXT::unittest_gpu_filter_root_to_leaf(
  const std::vector<uint32_t>& selected_state)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_filter_root_to_leaf requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  // Root-to-leaf filtering runs after DAG construction has
  // finalized every tentative node. Therefore its input must
  // contain only stable states 0 and 1.
  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 1u) {
      throw std::invalid_argument(
        "root-to-leaf filtering input must "
        "contain only states 0 or 1");
    }
  }

  // GPU tree construction owns the immutable CSR, roots,
  // leaves, queue, and filtering buffers.
  if (_dvc_picked_endes == nullptr) {
    (void)_gpu_tree_construction();
  }

  if (_num_ende > 0u) {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  }

  _gpu_filter_root_to_leaf();

  GpuRootToLeafFilterResult result;

  result.selected_state.resize(
    _num_ende,
    0u);

  result.from_roots.resize(
    _num_ende,
    0u);

  result.from_leaves.resize(
    _num_ende,
    0u);

  result.in_subgraph.resize(
    _num_ende,
    0u);

  if (_num_ende > 0u) {
    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.from_roots.data(),
      _dvc_from_roots,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.from_leaves.data(),
      _dvc_from_leaves,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.in_subgraph.data(),
      _dvc_in_subgraph,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  }

  return result;
}

G_EXT::GpuLevelizationResult
G_EXT::unittest_gpu_levelize(
  const std::vector<uint32_t>& selected_state,
  const bool is_top_down)
{
  // Preserve the existing unit-test API and behavior.
  //
  // This follows the same compile-time production dispatcher
  // selected by kEnableGpuDagHybridLevelization.
  return _unittest_gpu_levelize_impl(
    selected_state,
    is_top_down,
    GpuLevelizationBackend::Configured);
}

G_EXT::GpuLevelizationResult
G_EXT::unittest_gpu_levelize_with_backend(
  const std::vector<uint32_t>& selected_state,
  const bool is_top_down,
  const G_EXT::GpuLevelizationBackend backend)
{
  return _unittest_gpu_levelize_impl(
    selected_state,
    is_top_down,
    backend);
}

G_EXT::GpuLevelizationResult
G_EXT::_unittest_gpu_levelize_impl(
  const std::vector<uint32_t>& selected_state,
  const bool is_top_down,
  const G_EXT::GpuLevelizationBackend backend)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "_unittest_gpu_levelize_impl requires read() "
      "to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }
  }

  // GPU tree construction currently owns allocation of the
  // immutable CSR and shared GPU working buffers.
  //
  // This is executed only on the first unit-test call for this
  // G_EXT object. Later backend comparisons reuse the exact same
  // graph and shared buffers.
  if (_dvc_picked_endes == nullptr) {
    (void)_gpu_tree_construction();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    // Every backend receives the exact selected-state mask
    // supplied by the unit test.
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    // ----------------------------------------------------------
    // Persistent-path sentinel
    //
    // _gpu_initialize_dag_buffers() normally initializes this
    // buffer to zero. That is not sufficient for detecting use
    // because Completed is enum value zero.
    //
    // Therefore, write values that cannot be produced by a valid
    // persistent-kernel return. If the persistent kernel runs,
    // it overwrites all four fields before returning.
    //
    // This extra copy occurs only inside this unit-test helper.
    // ----------------------------------------------------------
    const g_ext_detail::GpuDagPersistentState
      persistent_sentinel{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<int32_t>::min(),
        g_ext_detail::GpuDagPersistentExitReason::
          InvalidQueueState
      };

    CUDA_CHECK(cudaMemcpy(
      _dvc_dag_persistent_state,
      &persistent_sentinel,
      sizeof(persistent_sentinel),
      cudaMemcpyHostToDevice));

    // Direct backend selection is restricted to this unit-test
    // helper. The production dispatcher remains unchanged.
    switch (backend) {
      case GpuLevelizationBackend::Configured:
        _gpu_dag_levelize(is_top_down);
        break;

      case GpuLevelizationBackend::HostDriven:
        _gpu_dag_levelize_host_driven(is_top_down);
        break;

      case GpuLevelizationBackend::Hybrid:
        _gpu_dag_levelize_hybrid(is_top_down);
        break;

      default:
        throw std::invalid_argument(
          "Invalid GPU levelization backend");
    }

    GpuLevelizationResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.residual_degree.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    const uint32_t* degree_device =
      is_top_down
        ? _dvc_indegree
        : _dvc_inv_indegree;

    CUDA_CHECK(cudaMemcpy(
      result.residual_degree.data(),
      degree_device,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.num_levels,
      _dvc_max_level,
      sizeof(int),
      cudaMemcpyDeviceToHost));

    // ----------------------------------------------------------
    // Detect whether the persistent kernel was actually used.
    //
    // Host-driven levelization never touches this buffer.
    // Hybrid levelization touches it only after entering the
    // persistent path.
    // ----------------------------------------------------------
    g_ext_detail::GpuDagPersistentState
      observed_persistent_state{};

    CUDA_CHECK(cudaMemcpy(
      &observed_persistent_state,
      _dvc_dag_persistent_state,
      sizeof(observed_persistent_state),
      cudaMemcpyDeviceToHost));

    const bool persistent_state_is_sentinel =
      observed_persistent_state.queue_begin ==
        persistent_sentinel.queue_begin &&
      observed_persistent_state.queue_count ==
        persistent_sentinel.queue_count &&
      observed_persistent_state.current_level ==
        persistent_sentinel.current_level &&
      observed_persistent_state.exit_reason ==
        persistent_sentinel.exit_reason;

    result.persistent_path_used =
      !persistent_state_is_sentinel;

    if (result.persistent_path_used) {
      result.persistent_completed =
        observed_persistent_state.exit_reason ==
          g_ext_detail::GpuDagPersistentExitReason::
            Completed;

      result.persistent_exited_to_host =
        observed_persistent_state.exit_reason ==
          g_ext_detail::GpuDagPersistentExitReason::
            FrontierTooWide;
    }

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}


void G_EXT::_copy_pickedende_back(
  std::vector<bool>& picked_endes_out)
{
  if (_dvc_picked_endes == nullptr) {
    throw std::logic_error(
      "_copy_pickedende_back called before GPU tree construction");
  }

  std::vector<uint32_t> host_selected(_num_ende, 0);

  cudaError_t err = cudaMemcpy(
    host_selected.data(),
    _dvc_picked_endes,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t),
    cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    throw std::runtime_error(
      std::string("Failed to copy selected e-nodes: ") +
      cudaGetErrorString(err));
  }

  picked_endes_out.assign(_num_ende, false);

  for (uint32_t ende = 0; ende < _num_ende; ++ende) {
    picked_endes_out[ende] = (host_selected[ende] != 0);
  }
}

void G_EXT::_copy_best_ende_of_ecls_back(
  std::vector<uint32_t>& best_ende_of_ecls_out)
{
  if (_dvc_best_ende_of_ecls == nullptr) {
    throw std::logic_error(
      "_copy_best_ende_of_ecls_back called before GPU tree construction");
  }

  best_ende_of_ecls_out.resize(_num_ecls);

  cudaError_t err = cudaMemcpy(
    best_ende_of_ecls_out.data(),
    _dvc_best_ende_of_ecls,
    static_cast<size_t>(_num_ecls) * sizeof(uint32_t),
    cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    throw std::runtime_error(
      std::string("Failed to copy best e-nodes: ") +
      cudaGetErrorString(err));
  }
}

// =========================== ILP Solver functions =========================== // 


using namespace operations_research;
using namespace sat;
void G_EXT::_cpsat_solver(
  const double time_limit,
  const std::vector<std::vector<uint32_t>>& ecls_its_ende, 
  const std::vector<uint32_t>& ende_child,
  const std::vector<uint32_t>& ende_child_ptr,
  const std::vector<uint32_t>& root_eclass_id,
  const std::vector<float>& ende_cost,
  const std::vector<uint32_t>& best_ende_of_ecls,
  const std::vector<bool>& picked_endes
) {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_cpsat_solver() Time_limit = %.2lf (sec)\n", time_limit);

  printf("ecls_its_ende\n");
  for (uint32_t i = 0; i < ecls_its_ende.size(); i++) {
    bool alive = (ecls_its_ende[i].size() == 0) ? (false) : (true); 
    if (alive) {
      printf("eclass %u, activate; with enodes: ", i); 
    } else {
      printf("eclass %u, deactivate;", i); 
    }
    for (uint32_t j = 0; j < ecls_its_ende[i].size(); j++) {
      printf("%u, ", ecls_its_ende[i][j]); 
    }
    printf("\n");  
  }
  std::vector<std::pair<uint32_t, uint32_t>> dep_log; // (parent_ecls, child_ecls)
  std::vector<std::tuple<uint32_t, uint32_t, int>> cost_log; // (eclass, enode, int_cost)
  std::vector<uint32_t> root_log;
#endif
uint32_t total_enodes = 0;
uint32_t alive_eclasses = 0;
uint32_t single_choice_eclasses = 0;

  // 20260616: ★ 新增：顯示 formulation 後的規模
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
      const auto& enodes = ecls_its_ende[cid];
      if (enodes.empty()) continue;
      alive_eclasses++;
      total_enodes += enodes.size();
      if (enodes.size() == 1) single_choice_eclasses++;
  }

  printf("[DAG Stats] Alive eclasses: %u\n", alive_eclasses);
  printf("[DAG Stats] Total enodes: %u\n", total_enodes);
  printf("[DAG Stats] Avg enodes per eclass: %.2f\n", 
        (float)total_enodes / alive_eclasses);
  printf("[DAG Stats] Single-choice eclasses: %u (%.1f%%)\n", 
        single_choice_eclasses, 
        100.0f * single_choice_eclasses / alive_eclasses);
  // 20260616: ★ 新增：顯示 formulation 後的規模

  std::unordered_map<uint32_t, BoolVar> active; // ClassId
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, BoolVar>> nodes_vars; // ClassId / NodeId

  CpModelBuilder cp_model;
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
    const auto& enodes = ecls_its_ende[cid];
    if (enodes.empty()) continue;  // skip dead eclasses
    
    active[cid] = cp_model.NewBoolVar();
    for (uint32_t ende_idx : enodes) {
      nodes_vars[cid].insert({ende_idx, cp_model.NewBoolVar()});
    }
    
    LinearExpr sum_nodes;
    for (const auto &v : nodes_vars[cid]) {
      sum_nodes += v.second;
    }
    cp_model.AddEquality(sum_nodes, active[cid]);
  }

#ifdef PRINT_LOG
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
    const auto& enodes = ecls_its_ende[cid];
    if (enodes.empty()) continue;
    printf("EClass %u: %zu enodes\n", cid, enodes.size());
  }
#endif
  
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
    const auto& enodes = ecls_its_ende[cid];
    if (enodes.empty()) continue;

    const auto& vars = nodes_vars[cid];
    for (uint32_t ende_idx : enodes) {
      uint32_t start = ende_child_ptr[ende_idx];
      uint32_t end   = ende_child_ptr[ende_idx+1];
      for (uint32_t jj = start; jj < end; jj++) {
        uint32_t chd_ecls_idx = ende_child[jj];
        if (active.find(chd_ecls_idx) == active.end()) { // 這個 if branch 是搭配我們先建立的 graph 用的
          // 該 child eclass 沒 enode，代表這個 enode 不可用
          cp_model.AddEquality(vars.at(ende_idx), 0);
          continue;
        }
        cp_model.AddLessOrEqual(vars.at(ende_idx), active[chd_ecls_idx]);
        #ifdef PRINT_LOG
          // log (parent enode → child_ecls)
          dep_log.emplace_back(ende_idx, chd_ecls_idx);
        #endif
      }
    }
  }

#ifdef PRINT_LOG
  printf("==== Dependency Constraints (node → child eclass) ====\n");
  for (const auto& [parent_ecls, child_ecls] : dep_log) {
    printf("  parent enodeIdx=%u -> child_ecls=%u\n", parent_ecls, child_ecls);
  }
  printf("======================================================\n\n");
#endif

  // Part 3. objective function
  LinearExpr objective;
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
    const auto& enodes = ecls_its_ende[cid];
    if (enodes.empty()) continue;

    const auto& vars = nodes_vars[cid];
    for (uint32_t ende_idx : enodes) {
      // const Node &node = egraph[ende_idx];
      const double cost = ende_cost[ende_idx];
      int int_cost = static_cast<int>(cost);
      if (int_cost != 0) {
        if (int_cost == 1) {
          objective += vars.at(ende_idx);
        } else {
          objective += int_cost * vars.at(ende_idx);
        }
        #ifdef PRINT_LOG
          cost_log.emplace_back(cid, ende_idx, int_cost);
        #endif
      }
    }
  }
  cp_model.Minimize(objective);

  // Part 4. active root e-classes
  for (uint32_t r = 0; r < root_eclass_id.size(); r++) {
    uint32_t root_idx = root_eclass_id[r];
    cp_model.AddEquality(active[root_idx], 1);
    #ifdef PRINT_LOG
      root_log.push_back(root_idx);
    #endif  
  }  

  // 20260616: ★ 新增：顯示 formulation 後的規模
  {
    const CpModelProto& proto = cp_model.Proto();
    std::cout << "[After formulation] Variables: " << proto.variables_size()
              << ", Constraints: " << proto.constraints_size() << "\n";
  }  
  // 20260616: ★ 新增：顯示 formulation 後的規模

#ifdef PRINT_LOG
  printf("==== Objective Cost Terms (active nodes) ====\n");
  for (const auto& [cid, ende_idx, int_cost] : cost_log) {
    printf("  eclass=%u, enode=%u, cost=%d\n", cid, ende_idx, int_cost);
  }
  printf("Total contributing terms: %zu\n", cost_log.size());
  printf("=================================================\n\n");

  printf("==== Root E-Class Activation Constraints ====\n");
  for (uint32_t rid : root_log) {
    printf("  Root eclass=%u forced active\n", rid);
  }
  printf("Total roots: %zu\n", root_log.size());
  printf("=============================================\n\n");
#endif

  // Part 5. 
  int num_classes = ecls_its_ende.size();
  std::unordered_map<uint32_t, IntVar, std::hash<uint32_t>> level;
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
    level[cid] = cp_model.NewIntVar(Domain(0, num_classes));
  }

  // Part 6. constraints
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
    const auto& enodes = ecls_its_ende[cid];
    if (enodes.empty()) continue;

    const auto& vars = nodes_vars[cid];
    for (uint32_t ende_idx : enodes) {
      uint32_t start = ende_child_ptr[ende_idx];
      uint32_t end   = ende_child_ptr[ende_idx+1];
      for (uint32_t jj = start; jj < end; jj++) {
        uint32_t child_cid = ende_child[jj];
        cp_model.AddGreaterOrEqual(LinearExpr(level[child_cid]) - level[cid], 1)
            .OnlyEnforceIf(vars.at(ende_idx));
      }
    }
  }  

#ifdef PRINT_LOG
  uint32_t hint_count_1 = 0;
  uint32_t hint_count_0 = 0;
  printf("==== Warm Start Hints (from CPU best_ende_of_ecls and picked_endes) ====\n");
  printf("%-10s %-10s %-10s %-10s\n", "Eclass", "EnodeIdx", "Hint", "Remark");
#endif

  // Part 8. warm start hint
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
    const auto& enodes = ecls_its_ende[cid];
    if (enodes.empty()) continue;
    const auto& vars = nodes_vars[cid];

    uint32_t best_idx = best_ende_of_ecls[cid];
    if (vars.find(best_idx) == vars.end()) continue;

    for (uint32_t ende_idx : enodes) {
      if (vars.find(ende_idx) == vars.end()) continue;

      int hint_val = (ende_idx == best_idx && picked_endes[best_idx]) ? 1 : 0;
      cp_model.AddHint(vars.at(ende_idx), hint_val);

      #ifdef PRINT_LOG
        if (hint_val == 1) {
          hint_count_1++;
          printf("%-10u %-10u %-10d %-10s\n", cid, ende_idx, 1, "<= picked best enode");
        } else {
          hint_count_0++;
          printf("%-10u %-10u %-10d %-10s\n", cid, ende_idx, 0, "");
        }
      #endif
    }
  }

  // // 20260616: ★ 新增：顯示 formulation 後的規模
  // // warm start 品質報告
  // {
  //     const CpModelProto& proto = cp_model.Proto();
  //     int num_hints = proto.solution_hint().vars_size();
  //     int total_vars = proto.variables_size();
  //     int total_cons = proto.constraints_size();
      
  //     std::cout << "[Warm Start Summary]\n";
  //     std::cout << "  Model size   : " << total_vars << " vars, " << total_cons << " constraints\n";
  //     std::cout << "  Hints set    : " << num_hints << " / " << total_vars << " vars\n";
  //     std::cout << "  Hint coverage: " 
  //               << (100.0 * num_hints / std::max(total_vars, 1)) << "%\n";
  // }
  // // 20260616: ★ 新增：顯示 formulation 後的規模

#ifdef PRINT_LOG
  printf("---------------------------------------------------------\n");
  printf("Total hints set to 1: %u\n", hint_count_1);
  printf("Total hints set to 0: %u\n", hint_count_0);
  printf("Grand total hints:    %u\n", hint_count_1 + hint_count_0);
  printf("=========================================================\n\n");
#endif

  // // 20260616: ★ 新增：顯示 formulation 後的規模
  // const CpModelProto& proto = cp_model.Proto();

  // Model model;
  // int num_solutions = 0;
  // auto start = std::chrono::high_resolution_clock::now();
  // model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& r) {
  //     auto current_time = std::chrono::high_resolution_clock::now();
  //     double elapsed_seconds =
  //         std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start).count();

  //     std::cout << "Incumbent Solution: "
  //               << elapsed_seconds << "s; objective = "
  //               << r.objective_value()
  //               << "; vars = " << proto.variables_size()        // ★
  //               << ", cons = " << proto.constraints_size()      // ★
  //               << std::endl;
  //     num_solutions++;
  // }));
  // // 20260616: ★ 新增：顯示 formulation 後的規模

  // Part 9. 
  Model model;
  int num_solutions = 0;
  auto start = std::chrono::high_resolution_clock::now();
  model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& r) {
      auto current_time = std::chrono::high_resolution_clock::now();
      double elapsed_seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start).count();

      std::cout << "Incumbent Solution: "
                << elapsed_seconds << "s; objective = "
                << r.objective_value() << std::endl;
      num_solutions++;
  })); // model.Add

  // // 20260616: ★ 新增：顯示 formulation 後的規模
  // // set time limit
  // if (time_limit > 0.0) {
  //   SatParameters parameters;
  //   parameters.set_max_time_in_seconds(time_limit);
  //   parameters.set_random_seed(42);
  //   parameters.set_randomize_search(true);
  //   // parameters.set_max_time_in_seconds(3600.0);
  //   parameters.set_log_search_progress(true);  // for full log
  //   model.Add(NewSatParameters(parameters));
  // }  
  // // 20260616: ★ 新增：顯示 formulation 後的規模

  // set time limit
  if (time_limit > 0.0) {
    SatParameters parameters;
    parameters.set_max_time_in_seconds(time_limit);
    parameters.set_random_seed(42);
    parameters.set_randomize_search(true);
    // parameters.set_max_time_in_seconds(3600.0);
    // parameters.set_log_search_progress(true);  // for full log
    model.Add(NewSatParameters(parameters));
  }


  // start solving ILP
  const CpSolverResponse response = SolveCpModel(cp_model.Build(), &model);

  // print results
  if (response.status() == CpSolverStatus::OPTIMAL || response.status() == CpSolverStatus::FEASIBLE) {
    std::cout << "=========================================\n";
    std::cout << "Solver finished.\n";
    std::cout << "Objective value = " << response.objective_value() << "\n";
    std::cout << "Runtime = " << response.wall_time() << " (s)\n";
    std::cout << "Feasible solutions found = " << num_solutions << "\n";

    // // Optional: print all active e-nodes
    // std::cout << "==== Active enodes (solution) ====\n";
    // for (const auto& [cid, node_map] : nodes_vars) {
    //   for (const auto& [node_id, var] : node_map) {
    //     bool value = SolutionBooleanValue(response, var);
    //     if (value) {
    //       std::cout << "EClass " << cid
    //                 << " -> enode " << node_id
    //                 << " (cost " << ende_cost[node_id] << ")\n";
    //     }
    //   }
    // }
    std::cout << "=========================================\n";
  } else {
    std::cout << "No feasible solution found.\n";
  }
}

// =========================== Print functions =========================== // 
void G_EXT::_print_adj(
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

void G_EXT::_print_read_in_data(
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
  std::cout << "\n========== G_EXT Internal State ==========\n";
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

void G_EXT::_print_degree(
  const bool is_td, 
  const std::vector<bool>& picked_endes) const 
{
  const std::string a = (is_td) ? ("Top-Down _indegree\n") : ("Bottom-up _inv_indegree\n");
  const std::string b = (is_td) ? ("_indegree") : ("_inv_indegree");
  const std::vector<uint32_t>& vec = (is_td) ? _indegree : _inv_indegree;
  
  std::cout << "G_EXT::_print_degree(): " << a ;
  for (uint32_t i = 0; i < vec.size(); i++) {
    if (picked_endes[i]) {
      std::cout << b << "[" << i << "] = " << vec[i] << " (picked: T)\n";
    } else {
      std::cout << b << "[" << i << "] = " << vec[i] << " (picked: F)\n";
    }
  }
  printf("\n");
}

void G_EXT::_print_levelization(
  const bool is_td, 
  const std::vector<int>& level,
  const std::vector<bool>& picked_endes) 
{
  const std::string a = (is_td) ? ("Top-Down\n") : ("Bottom-up\n");
  
  std::cout << "G_EXT::_print_levelization(): " << a ;
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

