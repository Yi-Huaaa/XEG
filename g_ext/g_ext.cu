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

void G_EXT::_get_superGraph() {
  printf("Inside G_EXT::_get_superGraph()\n");

  // Construct adj for the following step to find the SCC 
  std::unordered_map<uint32_t, uint32_t> ecls_id_to_tmp_id;
  std::vector<std::unordered_set<uint32_t>> alive_adj;
  _construct_alive_graph(ecls_id_to_tmp_id, alive_adj); 

  // Find SCC
  _find_SCC(alive_adj); 

  // Construct supernode adj 
  _construct_superGraph();
}

void G_EXT::_construct_alive_graph(std::unordered_map<uint32_t, uint32_t>& ecls_id_to_tmp_id, 
                                  std::vector<std::unordered_set<uint32_t>>& alive_adj) {
  printf("Inside G_EXT::_construct_alive_graph()\n"); 

  uint32_t acc = 0; 
  for (uint32_t ecls = 0; ecls < _alive_ecls.size(); ecls++) {
    if (_alive_ecls[ecls]) {
      ecls_id_to_tmp_id[ecls] = acc; 
      acc++; 
    }
  }
  MY_CHECK(acc = _num_alive_ecls, "acc = _num_alive_ecls");
#ifdef PRINT_LOG
  std::cout << "\necls_id_to_tmp_id:\n";
  for (auto &kv : ecls_id_to_tmp_id) {
    std::cout << "  " << kv.first << " -> " << kv.second << "\n";
  }
#endif

  // alive_adj.resize(_num_alive_ecls);
  // for (uint32_t ende_idx = 0; ende_idx < _num_ende; ++ende_idx) {
  //   uint32_t start = _ende_child_ptr[ende_idx];
  //   uint32_t end   = _ende_child_ptr[ende_idx+1];
  //   uint32_t ecls = _ende_ecls[ende_idx];
  //   uint32_t new_ecls = ecls_id_to_tmp_id[ecls];

  //   for (uint32_t j = start; j < end; j++) {
  //     uint32_t cd_ecls_idx = _ende_child[j];
  //     uint32_t new_cd_ecls_idx = ecls_id_to_tmp_id[cd_ecls_idx];
  //     if (_alive_ecls[ecls] && (new_ecls != new_cd_ecls_idx)) {
  //       alive_adj[new_ecls].insert(new_cd_ecls_idx); 
  //     }
  //   }
  // }

#ifdef PRINT_LOG
  // printf("alive_adj: \n"); 
  // _print_adj(alive_adj); 
#endif  
}

void G_EXT::_find_SCC(const std::vector<std::unordered_set<uint32_t>>& alive_adj) {
  printf("Inside G_EXT::_find_SCC()\n"); 
}

void G_EXT::_construct_superGraph() {
  printf("Inside G_EXT::_construct_superGraph\n"); 
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