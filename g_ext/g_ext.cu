#include "g_ext.cuh"

#define PRINT_LOG

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

__global__ void GPU_print_test() {
    printf("threadIdx.x = %d, hello world!\n", threadIdx.x);
}

void G_EXT::gpu_print() {
    printf("Get inside G_EXT::gpu_print, starts to launch GPU_print_test...\n");
    GPU_print_test <<< 1, 32 >>>();
    cudaDeviceSynchronize(); 
    printf("\ngpu_equality_saturator::gpu_print test ends...\n");
}

void G_EXT::_read_input(const std::string input_path, 
                        uint32_t* num_ende,
                        uint32_t* num_ecls) {
  std::ifstream ifs(input_path);
  if (!ifs.is_open()) {
    std::cerr << "Error: cannot open file " << input_path << std::endl;
    return;
  }

  // parse JSON
  nlohmann::json j;
  ifs >> j;

  auto nodes = j["nodes"]; // get number of nodes object
  _num_ende = nodes.size();
  _ende_name.reserve(_num_ende);
  _ende_op.reserve(_num_ende);
  _ende_cost.reserve(_num_ende);
  _ende_ecls.reserve(_num_ende);
  _ende_child_ptr.reserve(_num_ende + 1);

  // lambda function for eclass
  uint32_t ecls_counter = 0;
  auto get_ecls_id = [&](const std::string &ecls_name, std::unordered_map<std::string, uint32_t>& mm) {
    if (mm.find(ecls_name) == mm.end()) {
      mm[ecls_name] = ecls_counter++;
    }
    return mm[ecls_name];
  };

  // enodes
  _ende_child_ptr.push_back(0); // CSR starts
  for (auto &[ename, node] : nodes.items()) {
    std::string op = node["op"];
    float cost = node["cost"];
    std::string ecls_name = node["eclass"];
    uint32_t ecls_id = get_ecls_id(ecls_name, _ecls_name_to_id);

    _ende_name.push_back(ename);     // enode 的 id (e.g., "0__0")
    _ende_op.push_back(op);          // operator string
    _ende_cost.push_back(cost);      // cost
    _ende_ecls.push_back(ecls_id);   // enode's eclass

    // children
    for (auto &ch : node["children"]) {
      std::string child_enode = ch.get<std::string>();
      std::string child_ecls = nodes[child_enode]["eclass"];
      uint32_t child_ecls_id = get_ecls_id(child_ecls, _ecls_name_to_id);
      _ende_child.push_back(child_ecls_id);
    }

    _ende_child_ptr.push_back(_ende_child.size());
  }

  _num_ecls = ecls_counter; // get number of eclasses

  // roots
  for (auto &r : j["root_eclasses"]) {
    std::string ecls_name = r.get<std::string>();
    _root_eclass_id.push_back(get_ecls_id(ecls_name, _ecls_name_to_id));
  }

  MY_CHECK(_ende_name.size() == _num_ende, "!= num_ende 1");
  MY_CHECK(_ende_op.size() == _num_ende, "!= num_ende 2");
  MY_CHECK(_ende_cost.size() == _num_ende, "!= num_ende 3");
  MY_CHECK(_ende_ecls.size() == _num_ende, "!= num_ende 4");

  // printf
#ifdef PRINT_LOG
  std::cout << "\n========== G_EXT Internal State ==========\n";
  std::cout << "num_ende: " << _num_ende << ", " << "num_ecls: " << _num_ecls << "\n\n";

  std::cout << "Enodes info:\n";
  for (uint32_t i = 0; i < _num_ende; i++) {
    std::cout << "  enode[" << i << "]: "
              << "name=" << _ende_name[i] 
              << ", op=" << _ende_op[i] 
              << ", cost=" << _ende_cost[i]
              << ", eclass=" << _ende_ecls[i]
              << "\n";

    uint32_t start = _ende_child_ptr[i];
    uint32_t end   = _ende_child_ptr[i + 1];
    std::cout << "    children: ";
    for (uint32_t jj = start; jj < end; jj++) {
      std::cout << _ende_child[jj] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nEclass name -> id map:\n";
  for (auto &kv : _ecls_name_to_id) {
    std::cout << "  " << kv.first << " -> " << kv.second << "\n";
  }

  std::cout << "\nRoot eclasses:\n";
  for (auto r : _root_eclass_id) {
    std::cout << "  " << r << "\n";
  }
#endif   
}

void G_EXT::_prune_dead_nodes(uint32_t* num_ende,
                              uint32_t* num_ecls) {
  // Prune dead nodes 
  _alive_ecls.resize(_num_ecls, false); // init
  for (uint32_t ende_idx = 0; ende_idx < _num_ende; ++ende_idx) {
    uint32_t start = _ende_child_ptr[ende_idx];
    uint32_t end   = _ende_child_ptr[ende_idx+1];
    uint32_t ecls = _ende_ecls[ende_idx];
    for (uint32_t j = start; j < end; j++) {
      _alive_ecls[_ende_child[j]] = true;
    }
  }
  for (uint32_t r : _root_eclass_id) {
    _alive_ecls[r] = true;
  }

  // Set _num_alive_ende and _num_alive_ecls
  uint32_t num_dead_nodes = 0; 
  for (uint32_t i = 0; i < _alive_ecls.size(); i++) {
    if (!_alive_ecls[i]) {
      num_dead_nodes++; 
    }
  }
  _num_alive_ecls = _num_ecls - num_dead_nodes; 
  
  _num_alive_ende = 0;
  for (uint32_t i = 0; i < _ende_ecls.size(); i++) {
    uint32_t ecls = _ende_ecls[i];
    if (_alive_ecls[ecls]) {
      _num_alive_ende++;
    }
  }

#ifdef PRINT_LOG
  printf("Inside _prune_dead_nodes():\n"); 
  printf("Dead nodes: {");
  for (uint32_t i = 0; i < _alive_ecls.size(); i++) {
    if (!_alive_ecls[i]) {
      printf("%u, ", i); 
    }
  }
  printf("}\n"); 
  printf("_num_ecls = %u, _num_ende = %u, _ende_ecls.size() = %lu\n", _num_ecls, _num_ende, _ende_ecls.size());
  printf("_num_alive_ecls = %u, _num_alive_ende = %u\n", _num_alive_ecls, _num_alive_ende);
#endif 
}

void G_EXT::_preprocess() {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_preprocess()\n");
#endif

  // Construct adj for the following step to find the SCC 
  std::unordered_map<uint32_t, uint32_t> ecls_id_to_alive_id;
  std::vector<std::unordered_set<uint32_t>> alive_adj;
  _construct_alive_graph(ecls_id_to_alive_id, alive_adj); 

  // Find SCC
  std::unordered_map<uint32_t, std::vector<uint32_t>> comp_nodes;
  _find_SCC(alive_adj, comp_nodes);

  // Construct supernode adj 
  std::unordered_map<uint32_t, uint32_t> alive_id_to_super_adj;
  std::vector<std::unordered_set<uint32_t>> super_adj;
  _construct_superGraph(comp_nodes, alive_adj, alive_id_to_super_adj, super_adj);

  // Levelize supergraph
  std::vector<uint32_t> levelize_leaves_to_roots; 
  _levelize_from_leaves(super_adj, levelize_leaves_to_roots); 

  // Construct queue list 
  _get_queue_list(levelize_leaves_to_roots, ecls_id_to_alive_id, alive_id_to_super_adj); 
}

void G_EXT::_construct_alive_graph(std::unordered_map<uint32_t, uint32_t>& ecls_id_to_alive_id, 
                                  std::vector<std::unordered_set<uint32_t>>& alive_adj) {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_construct_alive_graph()\n"); 
#endif

  uint32_t acc = 0; 
  for (uint32_t ecls = 0; ecls < _alive_ecls.size(); ecls++) {
    if (_alive_ecls[ecls]) {
      ecls_id_to_alive_id[ecls] = acc; 
      acc++; 
    }
  }
  MY_CHECK(acc = _num_alive_ecls, "acc = _num_alive_ecls");

#ifdef PRINT_LOG
  std::cout << "\necls_id_to_alive_id:\n";
  for (auto &kv : ecls_id_to_alive_id) {
    std::cout << "  " << kv.first << " -> " << kv.second << "\n";
  }
#endif

  alive_adj.resize(_num_alive_ecls);
  for (uint32_t ende_idx = 0; ende_idx < _num_ende; ++ende_idx) {
    uint32_t start = _ende_child_ptr[ende_idx];
    uint32_t end   = _ende_child_ptr[ende_idx+1];
    uint32_t ecls = _ende_ecls[ende_idx];
    uint32_t new_ecls = ecls_id_to_alive_id[ecls];

    for (uint32_t j = start; j < end; j++) {
      uint32_t cd_ecls_idx = _ende_child[j];
      uint32_t new_cd_ecls_idx = ecls_id_to_alive_id[cd_ecls_idx];
      if (_alive_ecls[ecls] && (new_ecls != new_cd_ecls_idx)) {
        alive_adj[new_ecls].insert(new_cd_ecls_idx); 
      }
    }
  }

#ifdef PRINT_LOG
  printf("alive_adj: \n"); 
  _print_adj(alive_adj); 
  printf("\n"); 
#endif  
}

void G_EXT::_find_SCC(const std::vector<std::unordered_set<uint32_t>>& adj,
                      std::unordered_map<uint32_t, std::vector<uint32_t>>& comp_nodes) {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_find_SCC()\n"); 
#endif
  using namespace boost;

  typedef adjacency_list<vecS, vecS, directedS> Graph;
  Graph g(adj.size());

  for (size_t u = 0; u < adj.size(); u++) {
    for (auto v : adj[u]) {
      add_edge(u, v, g);
    }
  }

  std::vector<uint32_t> component(num_vertices(g));
  uint32_t num = strong_components(
    g, make_iterator_property_map(component.begin(), get(vertex_index, g))
  );
  
  for (size_t i = 0; i < component.size(); i++) {
    comp_nodes[component[i]].push_back(i);
  }

#ifdef PRINT_LOG  
  std::cout << "\n========== comp_nodes dump ==========\n";
  for (const auto& kv : comp_nodes) {
    uint32_t comp_id = kv.first;
    const auto& nodes = kv.second;
    std::cout << "Component " << comp_id << " (size=" << nodes.size() << "): { ";
    for (size_t i = 0; i < nodes.size(); i++) {
      std::cout << nodes[i];
      if (i + 1 < nodes.size()) std::cout << ", ";
    }
    std::cout << " }\n";
  }
  std::cout << "=====================================\n\n";  

  std::vector<std::vector<uint32_t>> real_sccs; // only used for check and debug
  std::vector<uint32_t> singletons; // only used for check and debug
  for (auto &kv : comp_nodes) {
    if (kv.second.size() > 1) {
      real_sccs.push_back(kv.second);
    } else {
      singletons.push_back(kv.second[0]);
    }
  }
  uint32_t acc = 0; // only used for check and debug
  if (!real_sccs.empty()) {
    std::cout << "Totally have " << real_sccs.size() << " SCCs in this graph\n";
    for (size_t i = 0; i < real_sccs.size(); i++) {
      std::cout << "SCC " << (i+1) << " (size = " << real_sccs[i].size() << ")"<< " {";
      for (size_t j = 0; j < real_sccs[i].size(); j++) {
        std::cout << real_sccs[i][j];
        if (j + 1 < real_sccs[i].size()) std::cout << ", ";
        acc++;
      }
      std::cout << "}\n";
    }
    if (!singletons.empty()) {
      std::cout << "The remaining nodes (not in SCCs) (size = " << singletons.size() << ") \n";
      acc += singletons.size();
    }
  } else {
    std::cout << "This is a DAG (no SCC)\n\n";
    acc = _num_alive_ecls;
  }
  MY_CHECK(_num_alive_ecls == acc, "_num_alive_ecls == acc");
  printf("\n");
#endif
}

void G_EXT::_construct_superGraph(const std::unordered_map<uint32_t, std::vector<uint32_t>>& comp_nodes,
                                  const std::vector<std::unordered_set<uint32_t>>& alive_adj,
                                  std::unordered_map<uint32_t, uint32_t>& alive_id_to_super_adj,
                                  std::vector<std::unordered_set<uint32_t>>& super_adj) {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_construct_superGraph\n"); 
#endif

  // Construct alive_id_to_super_adj
  for (const auto& kv : comp_nodes) {
    uint32_t comp_id = kv.first;
    const auto& nodes = kv.second;
    for (size_t i = 0; i < nodes.size(); i++) {
      alive_id_to_super_adj[nodes[i]] = comp_id;
    }
  }
  
  // Construct super_adj
  super_adj.resize(comp_nodes.size());
  for (uint32_t ecls = 0; ecls < alive_adj.size(); ecls++) {
    uint32_t new_ecls = alive_id_to_super_adj[ecls];
    for (uint32_t cd_ecls : alive_adj[ecls]) {
      uint32_t new_cd_ecls_idx = alive_id_to_super_adj[cd_ecls];
      if (new_ecls != new_cd_ecls_idx)
        super_adj[new_ecls].insert(new_cd_ecls_idx);
    }
  }

#ifdef PRINT_LOG
  std::cout << "\nalive_id_to_super_adj:\n";
  for (auto &kv : alive_id_to_super_adj) {
    std::cout << "  " << kv.first << " -> " << kv.second << "\n";
  }
  printf("super_adj:\n"); 
  _print_adj(super_adj);
  printf("\n"); 
#endif
}

void G_EXT::_levelize_from_leaves(const std::vector<std::unordered_set<uint32_t>>& super_adj,
                                  std::vector<uint32_t>& levelize_leaves_to_roots) {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_levelize_from_leaves\n"); 
#endif
  // Make live easier
  const size_t N = super_adj.size();

  // Construct in-degree; Note: We want sort levelize from leaves to roots! 
  std::vector<uint32_t> indeg(N, 0);
  for (size_t u = 0; u < N; u++) {
    for (auto v : super_adj[u]) {
      indeg[v]++;
    }
  }

  std::queue<uint32_t> q;
  for (size_t i = 0; i < N; i++) {
    if (indeg[i] == 0) { q.push(i); }
  }

  // BFS
  while (!q.empty()) {
    uint32_t u = q.front(); 
    q.pop();
    levelize_leaves_to_roots.push_back(u);

    for (auto v : super_adj[u]) {
      indeg[v]--;
      if (indeg[v] == 0) { q.push(v); }
    }
  }
  
  // Note: We want from leaves to roots
  std::reverse(levelize_leaves_to_roots.begin(), levelize_leaves_to_roots.end());  
  MY_CHECK(levelize_leaves_to_roots.size() == N, "levelize_leaves_to_roots.size() == N");

#ifdef PRINT_LOG
  printf("indeg:\n"); 
  for (uint32_t i = 0; i < indeg.size(); i++) {
    printf("Node[%u]'s indeg = %u\n", i, indeg[i]);
  }

  std::cout << "Levelized order (leaf -> root): ";
  for (auto n : levelize_leaves_to_roots) std::cout << n << " ";
  std::cout << "\n";
  printf("\n"); 
#endif
}


void G_EXT::_get_queue_list(const std::vector<uint32_t>& levelize_leaves_to_roots,
                            const std::unordered_map<uint32_t, uint32_t>& ecls_id_to_alive_id,
                            const std::unordered_map<uint32_t, uint32_t>& alive_id_to_super_adj) {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_get_queue_list\n"); 
#endif
  std::vector<uint32_t> super_idx_to_level; // give each super node one level
  super_idx_to_level.resize(levelize_leaves_to_roots.size()); 
  for (uint32_t i = 0; i < levelize_leaves_to_roots.size(); i++) {
    super_idx_to_level[levelize_leaves_to_roots[i]] = i;
    MY_CHECK(levelize_leaves_to_roots[i] < levelize_leaves_to_roots.size(), "levelize_leaves_to_roots[i] < levelize_leaves_to_roots.size()"); 
  }
#ifdef PRINT_LOG
  printf("super_idx_to_level:\n"); 
  for (uint32_t i = 0; i < super_idx_to_level.size(); i++) {
    std::cout << "  super_idx " << i << " -> level " << super_idx_to_level[i] << "\n";
  }
#endif 

  uint32_t acc = 0;
  std::vector<std::vector<uint32_t>> _queue_list_2D; // list
  _queue_list_2D.resize(levelize_leaves_to_roots.size(), std::vector<uint32_t>());
  for (uint32_t ende_idx = 0; ende_idx < _num_ende; ende_idx++) {
    uint32_t ecls = _ende_ecls[ende_idx];
    if (_alive_ecls[ecls]) {
      auto it_alive = ecls_id_to_alive_id.find(ecls);
      uint32_t alive_idx = it_alive->second;
      auto it_super = alive_id_to_super_adj.find(alive_idx);
      uint32_t super_idx = it_super->second;
      MY_CHECK(it_alive != ecls_id_to_alive_id.end(), "ecls not found in ecls_id_to_alive_id");
      MY_CHECK(it_super != alive_id_to_super_adj.end(), "alive_idx not found in alive_id_to_super_adj");
      uint32_t level = super_idx_to_level[super_idx];
      _queue_list_2D[level].push_back(ende_idx);
      acc++;
      #ifdef PRINT_LOG 
        printf("ende_idx = %u, ecls = %u, alive_idx = %u, super_idx = %u, level = %u\n", 
                ende_idx, ecls, alive_idx, super_idx, level); 
      #endif
    } else {
      #ifdef PRINT_LOG 
        printf("ende_idx = %u, ecls = %u, dead eclass\n", ende_idx, ecls); 
      #endif
    }    
  }
  MY_CHECK(acc == _num_alive_ende, "acc == _num_alive_ende");

  _queue_list.reserve(acc);
  for (const auto& inner : _queue_list_2D) {
    _queue_list.insert(_queue_list.end(), inner.begin(), inner.end());
  }

#ifdef PRINT_LOG 
  std::cout << "_queue_list (size=" << _queue_list.size() << ", with _num_alive_ende=" << _num_alive_ende << "): ";
  for (size_t i = 0; i < _queue_list.size(); i++) {
    std::cout << _queue_list[i];
    if (i + 1 < _queue_list.size()) std::cout << ", ";
  }
  std::cout << std::endl;
#endif
}

void G_EXT::_cpu_relaxation() {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_gpu_relaxation()\n"); 
#endif
  // Make all of the data into CSR format 
  uint32_t N = _queue_list.size();
  std::vector<float> csr_ende_cost; csr_ende_cost.resize(N, 0.0); 
  std::vector<uint32_t> csr_ende_ecls; csr_ende_ecls.resize(N, 0); 
  std::vector<uint32_t> csr_ende_child;
  std::vector<uint32_t> csr_ende_child_ptr; csr_ende_child_ptr.resize(N+1); 
  std::vector<uint32_t> csr_root_eclass_id(_root_eclass_id); // copy

  // Init 
  uint32_t acc = 0;
  for (uint32_t n = 0; n < N; n++) {
    uint32_t ende = _queue_list[n];
    csr_ende_cost[n] = _ende_cost[ende];
    csr_ende_ecls[n] = _ende_ecls[ende];
    csr_ende_child_ptr[n] = acc; 
    uint32_t start = _ende_child_ptr[ende];
    uint32_t end   = _ende_child_ptr[ende + 1];
    for (uint32_t jj = start; jj < end; jj++) {
      uint32_t val = _ende_child[jj]; 
      csr_ende_child.push_back(val);
      acc++;
    }
  }
  csr_ende_child_ptr[N] = acc; 

#ifdef PRINT_LOG
  printf("\n"); 
  printf("=== CSR Data Dump ===\n");
  printf("csr_ende_cost:\n");
  for (size_t i = 0; i < csr_ende_cost.size(); i++) {
    printf("  [%zu] = %.1f\n", i, csr_ende_cost[i]);
  }

  printf("\ncsr_ende_ecls:\n");
  for (size_t i = 0; i < csr_ende_ecls.size(); i++) {
    printf("  [%zu] = %u\n", i, csr_ende_ecls[i]);
  }

  printf("\ncsr_ende_child_ptr:\n");
  for (size_t i = 0; i < csr_ende_child_ptr.size(); i++) {
    printf("  [%zu] = %u\n", i, csr_ende_child_ptr[i]);
  }

  printf("\ncsr_ende_child:\n");
  for (size_t i = 0; i < csr_ende_child.size(); i++) {
    printf("  [%zu] = %u\n", i, csr_ende_child[i]);
  }

  printf("\ncsr_root_eclass_id:\n");
  for (size_t i = 0; i < csr_root_eclass_id.size(); i++) {
    printf("  [%zu] = %u\n", i, csr_root_eclass_id[i]);
  }
  
  printf("=====================\n");
  printf("\n"); 
#endif 

  // CPU relaxation 
  TODO here /
}

void G_EXT::_gpu_relaxation() {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_gpu_relaxation()\n"); 
#endif
  // TODO here 

}



// =========================== Print functions =========================== // 

void G_EXT::_print_adj(const std::vector<std::unordered_set<uint32_t>>& adj) const {
  for (uint32_t fm = 0; fm < adj.size(); fm++) {
    printf("fm_node = %u, to nodes = [", fm); 
    for (const auto& elem : adj[fm]) {
      printf("%u, ", elem);
    }
    printf("]\n");
  }
}