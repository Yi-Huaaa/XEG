#include "g_ext.cuh"

// #define PRINT_LOG
// #define CHECK_CYCLE
// #define BIBI_CHECK

#define ENABLE_CHECK
// Error checking macro for CPU 
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


// G_EXT fucntions
void G_EXT::_read_input(const std::string input_path) 
{
  std::ifstream ifs(input_path);
  if (!ifs.is_open()) {
    std::cerr << "Error: cannot open file " << input_path << std::endl;
    return;
  }

  // Parse JSON
  nlohmann::json j;
  ifs >> j;

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

    tmp_ende_name.push_back(ename);     // enode 的 id (e.g., "0__0")
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
        throw std::runtime_error("Unsupported child type in JSON (must be string or int)");
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

  // printf
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

void G_EXT::_prune_dead_nodes(const uint32_t tmp_num_ecls,
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
    uint32_t ecls = tmp_ende_ecls[ende_idx];
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
  
#ifdef CHECK_CYCLE
  // ======================================
  // SCC 檢查：確認 prune 後的 e-graph 是否仍有 cycle
  // ======================================
  {
    std::vector<std::vector<uint32_t>> adj(_num_ecls);
    // 建立 eclass-level adjacency
    for (uint32_t ende = 0; ende < _num_ende; ++ende) {
      uint32_t start = _ende_child_ptr[ende];
      uint32_t end   = _ende_child_ptr[ende + 1];
      uint32_t parent_ecls = _ende_ecls[ende];
      for (uint32_t jj = start; jj < end; ++jj) {
        uint32_t child_ecls = _ende_child[jj];
        adj[parent_ecls].push_back(child_ecls);
      }
    }

    // Tarjan's algorithm
    uint32_t index = 0;
    std::vector<int> dfn(_num_ecls, -1), low(_num_ecls, -1);
    std::vector<bool> in_stack(_num_ecls, false);
    std::stack<uint32_t> st;
    std::vector<std::vector<uint32_t>> scc_list;

    std::function<void(uint32_t)> tarjan = [&](uint32_t u) {
      dfn[u] = low[u] = index++;
      st.push(u);
      in_stack[u] = true;

      for (uint32_t v : adj[u]) {
        if (dfn[v] == -1) {
          tarjan(v);
          low[u] = std::min(low[u], low[v]);
        } else if (in_stack[v]) {
          low[u] = std::min(low[u], dfn[v]);
        }
      }

      if (low[u] == dfn[u]) {
        std::vector<uint32_t> scc;
        while (true) {
          uint32_t w = st.top();
          st.pop();
          in_stack[w] = false;
          scc.push_back(w);
          if (w == u) break;
        }
        if (scc.size() > 1) { // real cycle
          scc_list.push_back(scc);
        }
      }
    };

    for (uint32_t i = 0; i < _num_ecls; ++i) {
      if (dfn[i] == -1) tarjan(i);
    }

    if (!scc_list.empty()) {
      std::cout << "⚠️  Found " << scc_list.size() << " strongly connected components (cycles) after pruning.\n";
      for (size_t i = 0; i < scc_list.size(); ++i) {
        std::cout << "  SCC " << i << " (size=" << scc_list[i].size() << "): ";
        for (uint32_t e : scc_list[i]) std::cout << e << " ";
        std::cout << "\n";
      }
    } else {
      std::cout << "✅  Graph is acyclic (no SCC found after pruning).\n";
    }
  }
#endif

}

// =========================== TREE/DAG functions =========================== // 


std::pair<double, double> G_EXT::_cpu_best_tree(const std::vector<float>& tmp_ende_cost,
                                                std::vector<double>& cost_of_ecls,
                                                std::vector<double>& cost_of_ende,
                                                std::vector<uint32_t>& best_ende_of_ecls, 
                                                std::vector<bool>& picked_endes)
{
#ifdef PRINT_LOG
  printf("Inside G_EXT::_cpu_best_tree()\n"); 
#endif
  // CPU relaxation 
  bool h_update = true;
  cost_of_ecls.resize(_num_ecls, std::numeric_limits<double>::infinity());
  cost_of_ende.resize(_num_ende, std::numeric_limits<double>::infinity());
  best_ende_of_ecls.resize(_num_ecls, 0);
 
  while (h_update) {
    h_update = false;

    #pragma omp parallel
    {
      #pragma omp for
      for (uint32_t t_idx = 0; t_idx < _num_ende; t_idx++) {
        double acc_cost = tmp_ende_cost[t_idx];
        uint32_t start = _ende_child_ptr[t_idx];
        uint32_t end   = _ende_child_ptr[t_idx+1];
        for (uint32_t jj = start; jj < end; jj++) {
          acc_cost += cost_of_ecls[_ende_child[jj]];
        }

        cost_of_ende[t_idx] = (acc_cost < cost_of_ende[t_idx]) ? (acc_cost) : (cost_of_ende[t_idx]);

        const uint32_t which_ecls = _ende_ecls[t_idx];

        #pragma omp critical
        {
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
  _indegree.resize(_num_ende, 0);
  _inv_indegree.resize(_num_ende, 0);
  _adj.resize(_num_ende, std::unordered_set<uint32_t>()); // ende -> ende
  _inv_adj.resize(_num_ende, std::unordered_set<uint32_t>()); // ende -> ende

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
      _adj[cur].insert(chd);
      _inv_adj[chd].insert(cur);

      if (!picked_endes[chd]) {
        picked_endes[chd] = true;
        q.push(chd);
      }
    }
  }

  for (uint32_t u = 0; u < _num_ende; ++u) {
    for (uint32_t v : _adj[u]) {
      _indegree[v]++;
      _inv_indegree[u]++;
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
  printf("_adj:\n");
  _print_adj(_adj);

  printf("\n_inv_adj:\n");
  _print_adj(_inv_adj);
  printf("\n");
  _print_degree(true, picked_endes);  
  _print_degree(false, picked_endes);  
#endif

  return dag_ret;
}



__device__ float atomicMinFloat(float* addr, float val) {
  int* addr_as_i = (int*)addr;
  int old = *addr_as_i, assumed;

  do {
    assumed = old;
    float old_f = __int_as_float(assumed);
    if (old_f <= val) break; 
    old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
  } while (assumed != old);

  return __int_as_float(old);
}

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

__global__ void relax_enodes_kernel(const uint32_t num_endes, 
                                    const float* _ende_cost, 
                                    const uint32_t* _ende_ecls,
                                    const uint32_t* _ende_child,
                                    const uint32_t* _ende_child_ptr,
                                    bool* update, 
                                    double* cost_of_ecls,
                                    double* cost_of_ende,
                                    uint32_t* best_ende_of_ecls) 
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num_endes) return;

  // float acc_cost = _ende_cost[t_idx];
  double acc_cost = (double)_ende_cost[t_idx]; // promote from float
  uint32_t start = _ende_child_ptr[t_idx];
  uint32_t end   = _ende_child_ptr[t_idx+1];
  for (uint32_t jj = start; jj < end; jj++) {
    acc_cost += cost_of_ecls[_ende_child[jj]];
  }

  cost_of_ende[t_idx] = fmin(cost_of_ende[t_idx], acc_cost);
  const uint32_t which_ecls = _ende_ecls[t_idx];

  // float old = atomicMinFloat(&cost_of_ecls[which_ecls], acc_cost);
  double old = atomicMinDouble(&cost_of_ecls[which_ecls], acc_cost);
  if (acc_cost < old) {
    best_ende_of_ecls[which_ecls] = t_idx;
    *update = true; // No need to be atomic
  }  
}


std::pair<double, double> G_EXT::_gpu_best_tree() 
{
#ifdef PRINT_LOG
  printf("Inside G_EXT::_gpu_best_tree()\n"); 
#endif
  // CPU relaxation 
  bool h_update = true;
  std::vector<double> cost_of_ecls, cost_of_ende; 
  std::vector<uint32_t> best_ende_of_ecls;
  cost_of_ecls.resize(_num_ecls, std::numeric_limits<double>::infinity());
  cost_of_ende.resize(_num_ende, std::numeric_limits<double>::infinity());
  best_ende_of_ecls.resize(_num_ecls, 0);

  // Device memory 
  float* dvc_ende_cost; 
  uint32_t* dvc_ende_ecls;
  uint32_t* dvc_ende_child;
  uint32_t* dvc_ende_child_ptr;
  bool* dvc_update;
  double* dvc_cost_of_ecls; 
  double* dvc_cost_of_ende; 
  uint32_t* dvc_best_ende_of_ecls;

  cudaMalloc(&dvc_ende_cost, _ende_cost.size() * sizeof(float));
  cudaMalloc(&dvc_ende_ecls, _ende_ecls.size() * sizeof(uint32_t));
  cudaMalloc(&dvc_ende_child, _ende_child.size() * sizeof(uint32_t));
  cudaMalloc(&dvc_ende_child_ptr, _ende_child_ptr.size() * sizeof(uint32_t));
  cudaMalloc(&dvc_update, 1 * sizeof(bool));
  cudaMalloc(&dvc_cost_of_ecls, cost_of_ecls.size() * sizeof(double));
  cudaMalloc(&dvc_cost_of_ende, cost_of_ende.size() * sizeof(double));
  cudaMalloc(&dvc_best_ende_of_ecls, best_ende_of_ecls.size() * sizeof(uint32_t));
  cudaCheckErrors("cudaMalloc failed");

  cudaMemcpy(dvc_ende_cost, _ende_cost.data(), _ende_cost.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_ende_ecls, _ende_ecls.data(), _ende_ecls.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_ende_child, _ende_child.data(), _ende_child.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_ende_child_ptr, _ende_child_ptr.data(), _ende_child_ptr.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_update, &h_update, 1*sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_cost_of_ecls, cost_of_ecls.data(), cost_of_ecls.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_cost_of_ende, cost_of_ende.data(), cost_of_ende.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_best_ende_of_ecls, best_ende_of_ecls.data(), best_ende_of_ecls.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy failed");

  const int threads = 1024;
  const int blocks = (_num_ende + threads - 1) / threads;

  while (h_update) {
    h_update = false;
    cudaMemcpy(dvc_update, &h_update, sizeof(bool), cudaMemcpyHostToDevice);
    relax_enodes_kernel<<<blocks, threads>>>(_num_ende,
                                            dvc_ende_cost,
                                            dvc_ende_ecls,
                                            dvc_ende_child,
                                            dvc_ende_child_ptr,
                                            dvc_update,
                                            dvc_cost_of_ecls,
                                            dvc_cost_of_ende,
                                            dvc_best_ende_of_ecls);

    cudaCheckErrors("relax_enodes_kernel failed");
    cudaMemcpy(&h_update, dvc_update, sizeof(bool), cudaMemcpyDeviceToHost);
  }

  // // 我在把 cost_of_ende[t_idx] = fmin(cost_of_ende[t_idx], acc_cost);
  // //   best_ende_of_ecls[which_ecls] = t_idx;
  // //   讓他保證正確比較重要
  // printf("bibi check\n");
  // uint32_t ll = 0; 
  // for (uint32_t i = 0; i < _ecls_its_ende.size(); i++) {
  //   uint32_t L = _ecls_its_ende[i].size(); 
  //   ll = (L > ll) ? (L) : (ll);
  // }
  // printf("bibi check ll = %u\n", ll);


  // copy data back to CPU 
  cudaMemcpy(cost_of_ecls.data(), dvc_cost_of_ecls, cost_of_ecls.size()*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cost_of_ende.data(), dvc_cost_of_ende, cost_of_ende.size()*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(best_ende_of_ecls.data(), dvc_best_ende_of_ecls, best_ende_of_ecls.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost);

  // Get best tree cost 
  double tree_ret = 0.0; 
  for (uint32_t r = 0; r < _root_eclass_id.size(); r++) {
    tree_ret += cost_of_ecls[_root_eclass_id[r]];
  }
  
  // Get best dag cost 
  double dag_ret = 0.0;
  std::vector<bool> picked_endes; 
  picked_endes.resize(_num_ende, false); 

  std::vector<bool> visited_eclasses(_num_ecls, false);
  std::queue<uint32_t> q;

  // enqueue all roots
  for (uint32_t rid : _root_eclass_id) {
    if (rid < _num_ecls) {
      q.push(rid);
      visited_eclasses[rid] = true;
    }
  }

  while (!q.empty()) {
    uint32_t cur_ecls = q.front();
    q.pop();

    uint32_t ende_idx = best_ende_of_ecls[cur_ecls];
    if (ende_idx >= _num_ende) continue;

    if (!picked_endes[ende_idx]) {
      picked_endes[ende_idx] = true;
      dag_ret += _ende_cost[ende_idx];
    }

    uint32_t start = _ende_child_ptr[ende_idx];
    uint32_t end   = _ende_child_ptr[ende_idx + 1];
    for (uint32_t jj = start; jj < end; jj++) {
      uint32_t child_ecls = _ende_child[jj];
      if (!visited_eclasses[child_ecls]) {
        visited_eclasses[child_ecls] = true;
        q.push(child_ecls);
      }
    }
  }

  return std::make_pair(tree_ret, dag_ret);
}

void G_EXT::_reconstruct_graph(std::vector<double>& cost_of_ecls,
                                std::vector<double>& cost_of_ende,
                                std::vector<uint32_t>& best_ende_of_ecls,
                                std::vector<bool>& picked_endes) 
{
#ifdef PRINT_LOG
  printf("\nInside G_EXT::_reconstruct_graph()\n"); 
  // Reconstruct graph that can be levelization
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
  printf("\nConstruct _global_adj, _global_inv_adj :\n"); 
#endif

#ifdef BIBI_CHECK
  std::vector<bool> bibi_picked_endes_0 = picked_endes;
  uint32_t bibi_acc_0 = 0;
  for (uint32_t i = 0; i < picked_endes.size(); i++) {
    if (picked_endes[i]) {
      bibi_acc_0++;
    }
  }
#endif

  // Construct _global_adj, _global_inv_adj --> There maybe cycle inside _global_adj and _global_inv_adj
  _global_adj.resize(_num_ende, std::unordered_set<uint32_t>()); // ende -> ende
  _global_inv_adj.resize(_num_ende, std::unordered_set<uint32_t>()); // ende -> ende
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    uint32_t start = _ende_child_ptr[ende]; 
    uint32_t end = _ende_child_ptr[ende+1];
    for (uint32_t jj = start; jj < end; jj++) {
      uint32_t chd_ecls = _ende_child[jj];
      for (uint32_t ende2 : _ecls_its_ende[chd_ecls]) {
        // remove self-loop
        if (ende != ende2) {
          // update _global_adj 
          _global_adj[ende].insert(ende2); 
          // update inv_adjs
          _global_inv_adj[ende2].insert(ende);
        }
      }
    }
  }
#ifdef PRINT_LOG
  printf("_global_adj mapping (ende Id -> ende Id):\n");
  _print_adj(_global_adj);
  printf("_global_inv_adj mapping (ende Id -> ende Id):\n");
  _print_adj(_global_inv_adj);
  printf("\n\n============================== START TO _reconstruct_graph... ==============================\n");
#endif

// TODO: 
  // // Add more roots enodes
  for (uint32_t r_ecls : _root_eclass_id) {
    for (uint32_t r : _ecls_its_ende[r_ecls]) {
      picked_endes[r] = true;  // 把所有 roots enode 都拉成 true
    }
  }

  bool updated = true; 
  std::vector<int> level;
  level.resize(_num_ende, -1); // init, needed

  while(updated) {
    // bottom-up levelization && top-down insertion
    _update_deg(updated, picked_endes); 
    _bottom_up_levelize(picked_endes, level);
    bool tp = _insertion(true, picked_endes, level);
    #ifdef PRINT_LOG
      std::cout << "_top_down_insertion: updated? " << tp <<"\n\n"; 
    #endif

    // top-down levelization && bottom-up insertion
    _update_deg(tp, picked_endes); 
    _top_down_levelize(picked_endes, level);
    bool bu = _insertion(false, picked_endes, level);
    #ifdef PRINT_LOG
      std::cout << "_bottom_up_insertion: updated? " << bu <<"\n\n"; 
    #endif
   
    updated = tp || bu; 
    
    #ifdef PRINT_LOG
      std::cout << "Get into while loop again? " << updated << "\n==========================================\n"; 
    #endif
  }

#ifdef BIBI_CHECK
  uint32_t bibi_acc_1 = 0;
  for (uint32_t i = 0; i < picked_endes.size(); i++) {
    if (picked_endes[i]) {
      bibi_acc_1++;
    }
  }
  printf("Successfully ends, _max_level = %d, acc = %u, _num_ende = %u, (acc/_num_ende) = %.3lf%\n", 
          _max_level, bibi_acc_1, _num_ende, ((double)bibi_acc_1/(double)_num_ende)*100); 
#endif  

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

#ifdef PRINT_LOG
  printf("_adj\n");
  _print_adj(_adj);
  printf("\n");
  printf("_inv_adj\n");
  _print_adj(_inv_adj);
  printf("\n");
#endif
#ifdef BIBI_CHECK
  uint32_t bibi_acc_2 = 0;
  for (uint32_t i = 0; i < _in_subgraph.size(); i++) {
    if (_in_subgraph[i]) {
      bibi_acc_2++;
    }
  }
  printf("After removing redundant enodes: _max_level = %d, acc = %u, _num_ende = %u, (acc/_num_ende) = %.3lf%\n", 
          _max_level, bibi_acc_2, _num_ende, ((double)bibi_acc_2/(double)_num_ende)*100); 
  MY_CHECK(bibi_acc_0 <= bibi_acc_1, "bibi_acc_0 <= bibi_acc_1");
  MY_CHECK(bibi_acc_2 <= bibi_acc_1, "bibi_acc_2 <= bibi_acc_1");
  MY_CHECK(bibi_acc_0 <= bibi_acc_2, "bibi_acc_0 <= bibi_acc_2");
  for (uint32_t i = 0; i < bibi_picked_endes_0.size(); i++) {
    if (bibi_picked_endes_0[i]) {
      MY_CHECK(_in_subgraph[i] == bibi_picked_endes_0[i], "_in_subgraph[i] == bibi_picked_endes_0[i]");
    }
  }
#endif
}

void G_EXT::_update_deg(const bool need_to_update, const std::vector<bool>& picked_endes) 
{
#ifdef PRINT_LOG 
  printf("\nInside G_EXT::_update_deg()\n");
  printf("Pre:\n"); 
  _print_degree(true, picked_endes);
  _print_degree(false, picked_endes);
#endif
  if (!need_to_update) return; 
  // init 
  _indegree.assign(_num_ende, 0);
  _inv_indegree.assign(_num_ende, 0);  

  for (uint32_t fm_ende = 0; fm_ende < _num_ende; fm_ende++) {
    if (!picked_endes[fm_ende]) continue;

    for (uint32_t to_ende : _adj[fm_ende]) {
      if (!picked_endes[to_ende]) continue;
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


void G_EXT::_update_deg_and_adjs(const std::vector<bool>& picked) 
{
  #ifdef PRINT_LOG 
    printf("\nInside G_EXT::_update_deg_and_adjs()\n");
    printf("Pre:\n"); 
    _print_degree(true, picked);
    _print_degree(false, picked);
    printf("\n"); 
    printf("_adj\n");
    _print_adj(_adj);
    printf("\n");
    printf("_inv_adj\n");
    _print_adj(_inv_adj);
    printf("\n");  
  #endif

  // init 
  _indegree.assign(_num_ende, 0);
  _inv_indegree.assign(_num_ende, 0);  

  for (uint32_t u = 0; u < _adj.size(); u++) {
    if (!picked[u]) {
      _adj[u].clear();   // 該 node 不在子圖內，移除所有邊
      continue;
    }

    for (auto it = _adj[u].begin(); it != _adj[u].end(); ) {
      if (!picked[*it]) it = _adj[u].erase(it);
      else ++it;
    }
  }

  for (uint32_t v = 0; v < _inv_adj.size(); v++) {
    if (!picked[v]) {
      _inv_adj[v].clear();
      continue;
    }

    for (auto it = _inv_adj[v].begin(); it != _inv_adj[v].end(); ) {
      if (!picked[*it]) it = _inv_adj[v].erase(it);
      else ++it;
    }
  }

  for (uint32_t u = 0; u < _adj.size(); u++) {
    if (!picked[u]) continue;
    for (uint32_t v : _adj[u]) {
      _indegree[v]++;
      _inv_indegree[u]++;
    }
  }

  #ifdef PRINT_LOG
    printf("Post:\n"); 
    _print_degree(true, picked);
    _print_degree(false, picked);
    printf("\n"); 
    printf("_adj\n");
    _print_adj(_adj);
    printf("\n");
    printf("_inv_adj\n");
    _print_adj(_inv_adj);
    printf("\n");    
  #endif  
}

void G_EXT::_BFS(const bool is_td, std::vector<uint32_t>& deg, 
                const std::vector<bool>& picked_endes,
                const std::vector<std::unordered_set<uint32_t>>& adj,
                std::vector<int>& level)
{
#ifdef PRINT_LOG
  printf("\nInside G_EXT::_BFS()\n");
#endif 
  // lambda
  auto get_rev_level = [&](const uint32_t LEVEL, uint32_t a) -> uint32_t {
    return LEVEL - a - 1;
  };

  // init queue (roots || leaves)
  std::queue<uint32_t> q;
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if ((deg[ende] == 0) && picked_endes[ende]) {
      q.push(ende);
      #ifdef PRINT_LOG  
        printf("  Queue init push in enode %u\n", ende);
      #endif
    }
  } 
  #ifdef PRINT_LOG  
    printf("\n");
  #endif

  // BFS: top-down levelization 
  int cur_level = 0;
  while (!q.empty()) {
    size_t layer = q.size();
    for (size_t t = 0; t < layer; ++t) {
      uint32_t u = q.front(); q.pop();
      level[u] = cur_level;
      #ifdef PRINT_LOG
        printf("  Processing enode %u\n", u);
      #endif
      for (uint32_t v : adj[u]) {
        if (!picked_endes[v]) continue;
        #ifdef PRINT_LOG
          printf("   child enode %u, with deg[%u] = %u\n", v, v, deg[v]); 
        #endif
        if (deg[v] > 0 && --deg[v] == 0) {
          q.push(v);
          #ifdef PRINT_LOG
            printf("    Push enode %u (indegree->0)\n", v);
          #endif
        }
      }
    }
    ++cur_level;
  }
  // update _max_level
  _max_level = cur_level; 

  if (!is_td) {
    for (uint32_t i = 0; i < level.size(); i++) {
      level[i] = get_rev_level(_max_level, level[i]);
      MY_CHECK(level[i] >= 0, "level[i] >= 0,");
    }
  }

#ifdef PRINT_LOG
  printf("_max_level = %d, cur_level = %d\n", _max_level, cur_level); 
#endif
}

void G_EXT::_top_down_levelize(const std::vector<bool>& picked_endes,
                              std::vector<int>& level) 
{
#ifdef PRINT_LOG
  printf("\nG_EXT::_top_down_levelize()\n");
#endif

  // Top-down BFS levelization
  level.assign(_num_ende, -1); 
  _BFS(true, _indegree, picked_endes, _adj, level);

#ifdef PRINT_LOG
  _print_levelization(true, level, picked_endes);
#endif 
}

void G_EXT::_bottom_up_levelize(const std::vector<bool>& picked_endes,
                                std::vector<int>& level) 
{
#ifdef PRINT_LOG
  printf("\nG_EXT::_bottom_up_levelize()\n");
#endif

  // Bottom-up BFS levelization
  level.assign(_num_ende, -1); 
  _BFS(false, _inv_indegree, picked_endes, _inv_adj, level);

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
  // Lambda
  auto has_overlap = [&](const std::vector<uint32_t>& a,
                        const std::vector<uint32_t>& b) -> bool {
    for (uint32_t x : a)
      for (uint32_t y : b)
        if (x == y)
          return true;
    return false;
  };

  bool update = false;

  // Try all candidates
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if (picked_endes[ende]) continue;
    #ifdef PRINT_LOG
      printf("\n* Processing ende = %u\n", ende);
    #endif
    std::vector<uint32_t> valid_pars;
    std::vector<uint32_t> valid_chds;
    // _top_down_insertion(): 
      // If not picked, check whether it can be added to this graph
      // Conditions:
      //   1. At least 1 parent enode exists (located @ MAX(parent_level))
      //   2. At least 0 chldrn enode exists (located @ min(chldrn_level))
      //   3. parent_level <= chldrn_level; can equal since children can be push to lower level at the next levelization
    // _bottom_up_insertion(): 
      // If not picked, check whether it can be added to this graph
      // Conditions:
      //   1. At least 0 parent enode exists (located @ MAX(parent_level))
      //   2. At least 1 chldrn enode exists (located @ min(chldrn_level))
      //   3. parent_level <= chldrn_level; can equal since children can be push to lower level at the next levelization
    bool has_par = false, has_chd = false;
    int max_par_lvl = -1, min_chd_lvl = INT_MAX;
    uint32_t max_par_ende = UINT32_MAX, min_chd_ende = UINT32_MAX; // Note: init values here are not important!
    // Find max parent level 
    #ifdef PRINT_LOG
      printf("Find max parent level\n"); 
    #endif
    for (uint32_t par_ende : _global_inv_adj[ende]) {
      if (!picked_endes[par_ende]) continue;
      int par_lvl = level[par_ende];
      valid_pars.push_back(par_ende);
      if (max_par_lvl < par_lvl) {
        #ifdef PRINT_LOG
          printf("  pre: par_ende = %u, par_lvl = %d, max_par_lvl = %d\n", par_ende, par_lvl, max_par_lvl);
        #endif
        MY_CHECK(par_lvl >= 0, "par_lvl >= 0"); 
        has_par = true; 
        max_par_lvl = par_lvl; 
        max_par_ende = par_ende;
        #ifdef PRINT_LOG
          printf("  pos: par_ende = %u, par_lvl = %d, max_par_lvl = %d\n", par_ende, par_lvl, max_par_lvl);
       #endif
      }
    }
    // Find min child level
    #ifdef PRINT_LOG
      printf("Find min child level\n"); 
    #endif
    for (uint32_t chd_ende : _global_adj[ende]) {
      if (!picked_endes[chd_ende]) continue;
      // TODO: 這裏其實可能會造成搜索範圍變小，因為實際上，我們可以選擇不把該 parent and enode edge 加進來 QWQ
      // 但我們現在就是很暴力的，把所有可以是 parents 的 enode 都和 target enode 連上線的概念
      int chd_lvl = level[chd_ende];
      if (chd_lvl < 0) {
        printf("WTF: \n");
        std::cout << "chd_ende: " << chd_ende << ", chd_lvl = " << chd_lvl << "\n";
        exit(1); 
      }      
      MY_CHECK(chd_lvl >= 0, "chd_lvl >= 0"); 
      valid_chds.push_back(chd_ende);
      if (min_chd_lvl > chd_lvl) {
        #ifdef PRINT_LOG
          printf("  pre: chd_ende = %u, chd_lvl = %d, min_chd_lvl = %d\n", chd_ende, chd_lvl, min_chd_lvl);
        #endif 
        has_chd = true; 
        min_chd_lvl = chd_lvl;
        min_chd_ende = chd_ende; 
        #ifdef PRINT_LOG
          printf("  pos: chd_ende = %u, chd_lvl = %d, min_chd_lvl = %d\n", chd_ende, chd_lvl, min_chd_lvl);
        #endif 
      }
    }
    
    // Check whether form a cycle
    bool cond = (is_td) ? (has_par) : (has_chd);
    int given_lvl = (is_td) ? (max_par_lvl + 1) : (min_chd_lvl - 1);
    // given_lvl == 0 的時候就有可能把非 root 的 enode 加到 level 0，這個時候就不能再繼續往上加上去否則 level 就會壞掉
    // 因此我們會停在 given level = 0 的狀況下
    bool valid = (cond) && (max_par_lvl < min_chd_lvl) && (!has_overlap(valid_pars, valid_chds)) && (given_lvl >= 0);
    if (valid) {
      MY_CHECK(given_lvl > -1, "given_lvl > -1");
      // if (given_lvl < -1) {
      //   printf("given_lvl = %d, max_par_lvl = %d, min_chd_lvl = %d\n", given_lvl, max_par_lvl, min_chd_lvl);
      //   std::cout << "is_td: " << is_td << ", has_par: " << has_par << ", has_chd: " << has_chd << "\n"; 
      //   exit(1);
      // }

      // Update 
      picked_endes[ende] = true;
      level[ende] = given_lvl;
      // update _adj and _inv_adj
      for (uint32_t par_ende : valid_pars) {
        _adj[par_ende].insert(ende);
        _inv_adj[ende].insert(par_ende);
      }
      for (uint32_t chd_ende : valid_chds) {
        _adj[ende].insert(chd_ende);
        _inv_adj[chd_ende].insert(ende);
      }
      update = true; 

      #ifdef PRINT_LOG
        printf("! valid ende = %u, max_par_lvl = %d (max_par_ende = %u), min_chd_lvl = %d (min_chd_ende = %u), given_lvl = %d\n", 
                ende, 
                max_par_lvl, max_par_ende,
                min_chd_lvl, min_chd_ende,
                given_lvl);
      #endif      
    }
  }

  return update;
}

void G_EXT::_clear_redundant_endes(std::vector<bool>& picked_endes) 
{
#ifdef PRINT_LOG
  printf("Insides G_EXT::_clear_redundant_endes() \n");
#endif  

  std::vector<uint32_t> roots;
  std::vector<uint32_t> leaves;
  for (uint32_t r_ecls : _root_eclass_id) {
    for (uint32_t r : _ecls_its_ende[r_ecls]) {
      roots.push_back(r); 
    }
  }
  for (uint32_t ende = 0; ende < _global_adj.size(); ende++) {
    if (_global_adj[ende].size() == 0) {
      leaves.push_back(ende); 
    }
  }

#ifdef PRINT_LOG
  printf("r_roots: ");
  for (uint32_t r = 0; r < roots.size(); r++) {
    if (picked_endes[roots[r]]) {
      printf("%u, ", roots[r]); 
    }
  } printf("\n"); 

  printf("r_leavs: ");
  for (uint32_t r = 0; r < leaves.size(); r++) {
    if (picked_endes[leaves[r]]) {
      printf("%u, ", leaves[r]); 
    }
  } printf("\n");   
  printf("picked_endes now:  ");
  for (uint32_t i = 0; i < picked_endes.size(); i++) {
    if (picked_endes[i]) {
      printf("%u, ", i);
    } 
  } printf("\n");
#endif 

  get_nodes_in_subgraph(roots, leaves, _adj, _inv_adj, picked_endes);

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

void G_EXT::mark_reachable(const std::vector<uint32_t>& starts,
                    const std::vector<std::unordered_set<uint32_t>>& adj,
                    std::vector<bool>& visited,
                    const std::vector<bool>& picked_endes) {
  std::queue<uint32_t> q;
  for (uint32_t s : starts) {
    if (s < adj.size() && picked_endes[s] && !visited[s]) {
      q.push(s);
      visited[s] = true;
    }
  }

  while (!q.empty()) {
    uint32_t u = q.front(); q.pop();
    for (uint32_t v : adj[u]) {
      if (v < adj.size() && picked_endes[v] && !visited[v]) {
        visited[v] = true;
        q.push(v);
      }
    }
  }
}

void G_EXT::get_nodes_in_subgraph(
    const std::vector<uint32_t>& roots,
    const std::vector<uint32_t>& leaves,
    const std::vector<std::unordered_set<uint32_t>>& adj,
    const std::vector<std::unordered_set<uint32_t>>& inv_adj,
    const std::vector<bool>& picked_endes)
{
  size_t N = adj.size();
  std::vector<bool> from_roots(N, false);
  std::vector<bool> to_leaves(N, false);
  _in_subgraph.resize(N, false);

  // 1. reachable from roots
  mark_reachable(roots, adj, from_roots, picked_endes);

  // 2. can reach leaves
  mark_reachable(leaves, inv_adj, to_leaves, picked_endes);

  // 3. intersection
  for (size_t i = 0; i < N; ++i) {
    _in_subgraph[i] = from_roots[i] && to_leaves[i];
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
  
  // 对于每个节点，如果其被选中，则要求所有其子 e-class 也必须被激活
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

  // Part 3. 目标函数：最小化所有选中节点的成本
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

  // Part 4. 强制根 e-class 激活
  for (uint32_t r = 0; r < root_eclass_id.size(); r++) {
    uint32_t root_idx = root_eclass_id[r];
    cp_model.AddEquality(active[root_idx], 1);
    #ifdef PRINT_LOG
      root_log.push_back(root_idx);
    #endif  
  }  

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

  // Part 5. 为每个 e-class 创建层次变量，用于防止环路（范围 [0, num_classes]）
  int num_classes = ecls_its_ende.size();
  std::unordered_map<uint32_t, IntVar, std::hash<uint32_t>> level;
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
    level[cid] = cp_model.NewIntVar(Domain(0, num_classes));
  }

  // Part 6. 添加约束：如果一个节点被选中，则其所有子 e-class 的层次至少比当前 e-class 高 1
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


  // Part 7. 读取 zero_node.mst 文件，将对应节点变量固定为 0
  // Skip it now, this is the pruning part from e-boost, e-boost prunes more enodes with setting larger alpha

// #ifdef PRINT_LOG
//   printf("==== Level Variable Allocation (防环层级变量建立) ====\n");
//   printf("Total e-classes with level vars: %zu\n", level_log.size());
//   for (uint32_t cid : level_log) {
//     printf("  eclass=%u, level domain=[0,%d]\n", cid, num_classes);
//   }
//   printf("======================================================\n\n");

//   printf("==== Level Constraints (防环约束) ====\n");
//   printf("Format: parent_ecls -> child_ecls (enode_idx)\n");
//   for (const auto& [parent_cid, child_cid, ende_idx] : level_constr_log) {
//     printf("  %u -> %u (enode=%u)\n", parent_cid, child_cid, ende_idx);
//   }
//   printf("Total constraints: %zu\n", level_constr_log.size());
//   printf("======================================\n\n");
// #endif

  // Part 7. 读取 zero_node.mst 文件，将对应节点变量固定为 0
  // Skip zero_node.mst pruning
  // Add warm-start hint from CPU relaxation

#ifdef PRINT_LOG
  uint32_t hint_count_1 = 0;
  uint32_t hint_count_0 = 0;
  printf("==== Warm Start Hints (from CPU best_ende_of_ecls and picked_endes) ====\n");
  printf("%-10s %-10s %-10s %-10s\n", "Eclass", "EnodeIdx", "Hint", "Remark");
#endif


  // Part 8. 添加 warm start hint：依據 picked_endes
  for (uint32_t cid = 0; cid < ecls_its_ende.size(); ++cid) {
    const auto& enodes = ecls_its_ende[cid];
    if (enodes.empty()) continue;
    const auto& vars = nodes_vars[cid];

    uint32_t best_idx = best_ende_of_ecls[cid];
    if (vars.find(best_idx) == vars.end()) continue;

    // 為該 eclass 的所有 enode 設 hint (picked = 1, 其他 = 0)
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

#ifdef PRINT_LOG
  printf("---------------------------------------------------------\n");
  printf("Total hints set to 1: %u\n", hint_count_1);
  printf("Total hints set to 0: %u\n", hint_count_0);
  printf("Grand total hints:    %u\n", hint_count_1 + hint_count_0);
  printf("=========================================================\n\n");
#endif

  // Part 9. 设置求解器观察器、参数并求解
  Model model;
  int num_solutions = 0;
  auto start = std::chrono::high_resolution_clock::now();

  // 可選：觀察者，用來打印目前最好的解
  model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& r) {
      auto current_time = std::chrono::high_resolution_clock::now();
      double elapsed_seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start).count();

      std::cout << "Incumbent Solution: "
                << elapsed_seconds << "s; objective = "
                << r.objective_value() << std::endl;
      num_solutions++;
  })); // model.Add

  // 設定求解時間上限
  if (time_limit > 0.0) {
    SatParameters parameters;
    parameters.set_max_time_in_seconds(time_limit);
    // parameters.set_max_time_in_seconds(3600.0);
    // parameters.set_log_search_progress(true);  // 如果想看完整 solver log 可開
    model.Add(NewSatParameters(parameters));
  }

  // 真正開始求解
  const CpSolverResponse response = SolveCpModel(cp_model.Build(), &model);

  // 結果輸出
  if (response.status() == CpSolverStatus::OPTIMAL || response.status() == CpSolverStatus::FEASIBLE) {
    std::cout << "=========================================\n";
    std::cout << "Solver finished.\n";
    std::cout << "Objective value = " << response.objective_value() << "\n";
    std::cout << "Runtime = " << response.wall_time() << " (s)\n";
    std::cout << "Feasible solutions found = " << num_solutions << "\n";

    // // Optional: 印出所有 active 節點
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

void G_EXT::_print_adj(const std::vector<std::unordered_set<uint32_t>>& adj) const 
{
  for (uint32_t fm = 0; fm < adj.size(); fm++) {
    printf("fm_node = %u, to nodes = [", fm); 
    for (const auto& elem : adj[fm]) {
      printf("%u, ", elem);
    }
    printf("]\n");
  }
}

void G_EXT::_print_read_in_data(const bool init,
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

void G_EXT::_print_degree(const bool is_td, const std::vector<bool>& picked_endes) const 
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

void G_EXT::_print_levelization(const bool is_td, const std::vector<int>& level,
                                const std::vector<bool>& picked_endes) 
{
  const std::string a = (is_td) ? ("Top-Down\n") : ("Bottom-up\n");
  
  std::cout << "G_EXT::_print_levelization(): " << a ;
  printf("=== Levelization result ===\n");
  // for (uint32_t ende = 0; ende < _num_ende; ende++) {
  //   if (level[ende] >= 0)
  //     printf("ENode %u -> Level %d\n", ende, level[ende]);
  // }
 
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
