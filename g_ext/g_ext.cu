#include "g_ext.cuh"

// #define PRINT_LOG

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
void G_EXT::_read_input(const std::string input_path
) {
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
    // std::string ecls_name = node["eclass"];
    // uint32_t ecls_id = get_ecls_id(ecls_name, _ecls_name_to_id);
    
    // eclass
    std::string ecls_name;
    if (node["eclass"].is_string()) {
      ecls_name = node["eclass"].get<std::string>();
    } else if (node["eclass"].is_number_integer()) {
      ecls_name = std::to_string(node["eclass"].get<int>());
    } else {
      throw std::runtime_error("Unsupported eclass type in JSON (must be string or int)");
    }
    uint32_t ecls_id = get_ecls_id(ecls_name, _ecls_name_to_id);

    _ende_name.push_back(ename);     // enode 的 id (e.g., "0__0")
    _ende_op.push_back(op);          // operator string
    _ende_cost.push_back(cost);      // cost
    _ende_ecls.push_back(ecls_id);   // enode's eclass

    // // children
    // for (auto &ch : node["children"]) {
    //   std::string child_enode = ch.get<std::string>();
    //   std::string child_ecls = nodes[child_enode]["eclass"];
    //   uint32_t child_ecls_id = get_ecls_id(child_ecls, _ecls_name_to_id);
    //   _ende_child.push_back(child_ecls_id);
    // }

    // children
    for (auto &ch : node["children"]) {
      if (ch.is_string()) {
        // === egg / extraction-gym format ===
        std::string child_enode = ch.get<std::string>();
        std::string child_ecls = nodes[child_enode]["eclass"];
        uint32_t child_ecls_id = get_ecls_id(child_ecls, _ecls_name_to_id);
        _ende_child.push_back(child_ecls_id);
      } else if (ch.is_number_integer()) {
        // === eboost / BoolE format ===
        std::string child_ecls = std::to_string(ch.get<int>());
        uint32_t child_ecls_id = get_ecls_id(child_ecls, _ecls_name_to_id);
        _ende_child.push_back(child_ecls_id);
      } else {
        throw std::runtime_error("Unsupported child type in JSON (must be string or int)");
      }
    }

    _ende_child_ptr.push_back(_ende_child.size());
  }

  _num_ecls = ecls_counter; // get number of eclasses

  // // roots
  // for (auto &r : j["root_eclasses"]) {
  //   std::string ecls_name = r.get<std::string>();
  //   _root_eclass_id.push_back(get_ecls_id(ecls_name, _ecls_name_to_id));
  // }

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

void G_EXT::_prune_dead_nodes() {
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

void G_EXT::_preprocess(std::vector<float>& csr_ende_cost, 
                        std::vector<uint32_t>& csr_ende_ecls,
                        std::vector<uint32_t>& csr_ende_child,
                        std::vector<uint32_t>& csr_ende_child_ptr,
                        std::vector<std::vector<uint32_t>>& ecls_its_ende
) {
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

  // Store things in CSR
  _CSR(_queue_list, 
      _ende_cost, _ende_ecls, 
      _ende_child, _ende_child_ptr,
      csr_ende_cost, csr_ende_ecls,
      csr_ende_child, csr_ende_child_ptr, ecls_its_ende);
}

void G_EXT::_construct_alive_graph(std::unordered_map<uint32_t, uint32_t>& ecls_id_to_alive_id, 
                                  std::vector<std::unordered_set<uint32_t>>& alive_adj
) {
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
                      std::unordered_map<uint32_t, std::vector<uint32_t>>& comp_nodes
) {
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
  // printf("_num_alive_ecls = %u, acc = %u\n", _num_alive_ecls, acc);
  MY_CHECK(_num_alive_ecls == acc, "_num_alive_ecls == acc");
  printf("\n");
#endif
}

void G_EXT::_construct_superGraph(const std::unordered_map<uint32_t, std::vector<uint32_t>>& comp_nodes,
                                  const std::vector<std::unordered_set<uint32_t>>& alive_adj,
                                  std::unordered_map<uint32_t, uint32_t>& alive_id_to_super_adj,
                                  std::vector<std::unordered_set<uint32_t>>& super_adj
) {
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
                                  std::vector<uint32_t>& levelize_leaves_to_roots
) {
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
                            const std::unordered_map<uint32_t, uint32_t>& alive_id_to_super_adj
) {
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

void G_EXT::_CSR(const std::vector<uint32_t>& queue, 
                const std::vector<float>& ende_cost, 
                const std::vector<uint32_t>& ende_ecls,
                const std::vector<uint32_t>& ende_child,
                const std::vector<uint32_t>& ende_child_ptr,
                std::vector<float>& csr_ende_cost, 
                std::vector<uint32_t>& csr_ende_ecls,
                std::vector<uint32_t>& csr_ende_child,
                std::vector<uint32_t>& csr_ende_child_ptr,
                std::vector<std::vector<uint32_t>>& ecls_its_ende
) {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_CSR()\n"); 
#endif
  // Make all of the data into CSR format 

  uint32_t N = queue.size();
  // printf("N = %u\n", N);
  csr_ende_cost.resize(N, 0.0); 
  csr_ende_ecls.resize(N, 0); 
  csr_ende_child_ptr.resize(N+1); 
  ecls_its_ende.resize(_num_ecls, std::vector<uint32_t>());

  // Init 
  uint32_t acc = 0;
  for (uint32_t n = 0; n < N; n++) {
    uint32_t ende = queue[n];
    csr_ende_cost[n] = ende_cost[ende];
    csr_ende_ecls[n] = ende_ecls[ende];
    ecls_its_ende[ende_ecls[ende]].push_back(n);
    csr_ende_child_ptr[n] = acc; 
    uint32_t start = ende_child_ptr[ende];
    uint32_t end   = ende_child_ptr[ende + 1];
    for (uint32_t jj = start; jj < end; jj++) {
      uint32_t val = ende_child[jj]; 
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

  printf("=====================\n");
  printf("\n"); 
#endif 

}

std::pair<double, double> G_EXT::_cpu_relaxation(const uint32_t N, const std::vector<float>& csr_ende_cost, 
                            const std::vector<uint32_t>& csr_ende_ecls,
                            const std::vector<uint32_t>& csr_ende_child,
                            const std::vector<uint32_t>& csr_ende_child_ptr,
                            std::vector<double>& cost_of_ecls,
                            std::vector<double>& cost_of_ende,
                            std::vector<uint32_t>& best_ende_of_ecls, std::vector<bool>& picked_endes
) {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_cpu_relaxation()\n"); 
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
      for (uint32_t t_idx = 0; t_idx < N; t_idx++) {
        double acc_cost = csr_ende_cost[t_idx];
        uint32_t start = csr_ende_child_ptr[t_idx];
        uint32_t end   = csr_ende_child_ptr[t_idx+1];
        for (uint32_t jj = start; jj < end; jj++) {
          acc_cost += cost_of_ecls[csr_ende_child[jj]];
        }

        cost_of_ende[t_idx] = (acc_cost < cost_of_ende[t_idx]) ? (acc_cost) : (cost_of_ende[t_idx]);

        const uint32_t which_ecls = csr_ende_ecls[t_idx];

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
#endif

  // Get best tree cost 
  double tree_ret = 0.0; 
  for (uint32_t r = 0; r < _root_eclass_id.size(); r++) {
    tree_ret += cost_of_ecls[_root_eclass_id[r]];
  }
  
  // Get best dag cost 
  double dag_ret = 0.0; 
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
      dag_ret += csr_ende_cost[ende_idx];
    }

    uint32_t start = csr_ende_child_ptr[ende_idx];
    uint32_t end   = csr_ende_child_ptr[ende_idx + 1];
    for (uint32_t jj = start; jj < end; jj++) {
      uint32_t child_ecls = csr_ende_child[jj];
      if (!visited_eclasses[child_ecls]) {
        visited_eclasses[child_ecls] = true;
        q.push(child_ecls);
      }
    }
  }

#ifdef PRINT_LOG
  printf("\nActivated enodes in DAG traversal:\n");
  for (size_t i = 0; i < picked_endes.size(); i++) {
    if (picked_endes[i]) {
      printf("  enode[%zu] (eclass %u, cost %.1f)\n", 
             i, csr_ende_ecls[i], csr_ende_cost[i]);
    }
  }
  printf("DAG total active enodes: %zu\n", 
         std::count(picked_endes.begin(), picked_endes.end(), true));
#endif

  
  return std::make_pair(tree_ret, dag_ret);
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

__global__ void relax_enodes_kernel(const uint32_t N, 
                                    const uint32_t num_endes, 
                                    const float* csr_ende_cost, 
                                    const uint32_t* csr_ende_ecls,
                                    const uint32_t* csr_ende_child,
                                    const uint32_t* csr_ende_child_ptr,
                                    bool* update, double* cost_of_ecls
) {
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num_endes) return;

  // float acc_cost = csr_ende_cost[t_idx];
  double acc_cost = (double)csr_ende_cost[t_idx]; // promote from float
  uint32_t start = csr_ende_child_ptr[t_idx];
  uint32_t end   = csr_ende_child_ptr[t_idx+1];
  for (uint32_t jj = start; jj < end; jj++) {
    acc_cost += cost_of_ecls[csr_ende_child[jj]];
  }

  const uint32_t which_ecls = csr_ende_ecls[t_idx];

  // float old = atomicMinFloat(&cost_of_ecls[which_ecls], acc_cost);
  double old = atomicMinDouble(&cost_of_ecls[which_ecls], acc_cost);
  if (acc_cost < old) {
    *update = true; // No need to be atomic
  }  
}

double G_EXT::_gpu_relaxation(const uint32_t N, 
                              const std::vector<float>& csr_ende_cost, 
                              const std::vector<uint32_t>& csr_ende_ecls,
                              const std::vector<uint32_t>& csr_ende_child,
                              const std::vector<uint32_t>& csr_ende_child_ptr
) {
#ifdef PRINT_LOG
  printf("Inside G_EXT::_gpu_relaxation()\n"); 
#endif
  // CPU relaxation 
  bool h_update = true;
  std::vector<double> cost_of_ecls; 
  cost_of_ecls.resize(_num_ecls, std::numeric_limits<double>::infinity());

  // Device memory 
  float* dvc_csr_ende_cost; 
  uint32_t* dvc_csr_ende_ecls;
  uint32_t* dvc_csr_ende_child;
  uint32_t* dvc_csr_ende_child_ptr;
  bool* dvc_update;
  double* dvc_cost_of_ecls; 

  cudaMalloc(&dvc_csr_ende_cost, csr_ende_cost.size() * sizeof(float));
  cudaMalloc(&dvc_csr_ende_ecls, csr_ende_ecls.size() * sizeof(uint32_t));
  cudaMalloc(&dvc_csr_ende_child, csr_ende_child.size() * sizeof(uint32_t));
  cudaMalloc(&dvc_csr_ende_child_ptr, csr_ende_child_ptr.size() * sizeof(uint32_t));
  cudaMalloc(&dvc_update, 1 * sizeof(bool));
  cudaMalloc(&dvc_cost_of_ecls, cost_of_ecls.size() * sizeof(double));
  cudaCheckErrors("cudaMalloc failed");

  cudaMemcpy(dvc_csr_ende_cost, csr_ende_cost.data(), csr_ende_cost.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_csr_ende_ecls, csr_ende_ecls.data(), csr_ende_ecls.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_csr_ende_child, csr_ende_child.data(), csr_ende_child.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_csr_ende_child_ptr, csr_ende_child_ptr.data(), csr_ende_child_ptr.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_update, &h_update, 1*sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_cost_of_ecls, cost_of_ecls.data(), cost_of_ecls.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy failed");

  const int threads = 1024;
  const int blocks = (_num_alive_ende + threads - 1) / threads;

  while (h_update) {
    h_update = false;
    cudaMemcpy(dvc_update, &h_update, sizeof(bool), cudaMemcpyHostToDevice);

    relax_enodes_kernel <<< blocks, threads >>> (N, _num_alive_ende,
                                                dvc_csr_ende_cost,
                                                dvc_csr_ende_ecls,
                                                dvc_csr_ende_child,
                                                dvc_csr_ende_child_ptr,
                                                dvc_update,
                                                dvc_cost_of_ecls); 
    cudaCheckErrors("relax_enodes_kernel failed");
    cudaMemcpy(&h_update, dvc_update, sizeof(bool), cudaMemcpyDeviceToHost);
  }

  // copy data back to CPU 
  cudaMemcpy(cost_of_ecls.data(), dvc_cost_of_ecls, cost_of_ecls.size()*sizeof(double), cudaMemcpyDeviceToHost);

  double ret = 0.0; 
  for (uint32_t r = 0; r < _root_eclass_id.size(); r++) {
    ret += cost_of_ecls[_root_eclass_id[r]];
  }

  return ret;
}


// =========================== ILP Solver functions =========================== // 
using namespace operations_research;
using namespace sat;

void G_EXT::SimpleSatProgram() {
  // 1. Create the CP-SAT model.
  CpModelBuilder cp_model;

  // 2. Define the variables.
  const Domain domain(0, 100);
  IntVar a = cp_model.NewIntVar(domain).WithName("a");
  IntVar b = cp_model.NewIntVar(domain).WithName("b");
  IntVar c = cp_model.NewIntVar(domain).WithName("c");

  // 3. Define the constraints.
  cp_model.AddEquality(a + b + c, 200);

  // 4. Solve the model.
  CpSolverResponse response = Solve(cp_model.Build());

  // 5. Interpret the results.
  if (response.status() == CpSolverStatus::OPTIMAL ||
    response.status() == CpSolverStatus::FEASIBLE) {
    LOG(INFO) << "Solution found:";
    LOG(INFO) << "a = " << SolutionIntegerValue(response, a);
    LOG(INFO) << "b = " << SolutionIntegerValue(response, b);
    LOG(INFO) << "c = " << SolutionIntegerValue(response, c);
  } else {
    LOG(INFO) << "No solution found.";
  }
}


void G_EXT::_cpsat_solver(
  const double time_limit,
  const std::vector<float>& csr_ende_cost, 
  const std::vector<uint32_t>& csr_ende_ecls,
  const std::vector<uint32_t>& csr_ende_child,
  const std::vector<uint32_t>& csr_ende_child_ptr,
  const std::vector<std::vector<uint32_t>>& ecls_its_ende,
  const std::vector<double>& cost_of_ecls,
  const std::vector<double>& cost_of_ende,
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
  // std::vector<uint32_t> level_log;
  // std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> level_constr_log; // (parent_ecls, child_ecls, ende_idx)

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
      uint32_t start = csr_ende_child_ptr[ende_idx];
      uint32_t end   = csr_ende_child_ptr[ende_idx+1];
      for (uint32_t jj = start; jj < end; jj++) {
        uint32_t chd_ecls_idx = csr_ende_child[jj];
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
      const double cost = csr_ende_cost[ende_idx];
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
  for (uint32_t r = 0; r < _root_eclass_id.size(); r++) {
    uint32_t root_idx = _root_eclass_id[r];
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
      uint32_t start = csr_ende_child_ptr[ende_idx];
      uint32_t end   = csr_ende_child_ptr[ende_idx+1];
      for (uint32_t jj = start; jj < end; jj++) {
        uint32_t child_cid = csr_ende_child[jj];
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
    //                 << " (cost " << csr_ende_cost[node_id] << ")\n";
    //     }
    //   }
    // }
    // std::cout << "=========================================\n";
  } else {
    std::cout << "No feasible solution found.\n";
  }
}


// =========================== Print functions =========================== // 

void G_EXT::_print_adj(const std::vector<std::unordered_set<uint32_t>>& adj
) const {
  for (uint32_t fm = 0; fm < adj.size(); fm++) {
    printf("fm_node = %u, to nodes = [", fm); 
    for (const auto& elem : adj[fm]) {
      printf("%u, ", elem);
    }
    printf("]\n");
  }
}