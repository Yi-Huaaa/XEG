// #include <nlohmann/json.hpp>
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

__global__ void GPU_print_test() {
    printf("threadIdx.x = %d, hello world!\n", threadIdx.x);
}

void G_EXT::gpu_print() {
    printf("Get inside G_EXT::gpu_print, starts to launch GPU_print_test...\n");
    GPU_print_test <<< 1, 32 >>>();
    cudaDeviceSynchronize(); 
    printf("\ngpu_equality_saturator::gpu_print test ends...\n");
}

void G_EXT::read(const std::string input_path) {
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
  auto get_ecls_id = [&](const std::string &ecls_name) {
    if (ecls_name_to_id.find(ecls_name) == ecls_name_to_id.end()) {
      ecls_name_to_id[ecls_name] = ecls_counter++;
    }
    return ecls_name_to_id[ecls_name];
  };

  // enodes
  _ende_child_ptr.push_back(0); // CSR starts
  for (auto &[ename, node] : nodes.items()) {
    std::string op = node["op"];
    float cost = node["cost"];
    std::string ecls_name = node["eclass"];
    uint32_t ecls_id = get_ecls_id(ecls_name);

    _ende_name.push_back(ename);     // enode 的 id (e.g., "0__0")
    _ende_op.push_back(op);          // operator string
    _ende_cost.push_back(cost);      // cost
    _ende_ecls.push_back(ecls_id);   // enode's eclass

    // children
    for (auto &ch : node["children"]) {
      std::string child_enode = ch.get<std::string>();
      std::string child_ecls = nodes[child_enode]["eclass"];
      uint32_t child_ecls_id = get_ecls_id(child_ecls);
      _ende_child.push_back(child_ecls_id);
    }

    _ende_child_ptr.push_back(_ende_child.size());
  }

  _num_ecls = ecls_counter; // get number of eclasses

  // roots
  for (auto &r : j["root_eclasses"]) {
    std::string ecls_name = r.get<std::string>();
    _root_eclass_id.push_back(get_ecls_id(ecls_name));
  }

#ifdef PRINT_LOG
  _print_input(); 
#endif   
}


void G_EXT::_SCC() {
#ifdef PRINT_LOG
  printf("Inside void G_EXT::_SCC()\n"); 
#endif 

  std::vector<std::unordered_set<uint32_t>> adj; // should be enodes; adjacency list
  adj.resize(_num_ecls); 
  
  // Cconstruct DAG, from -> to
  _construct_graph(adj);
  
  // Find SCC
  _find_SCC(adj); 
}

void G_EXT::_construct_graph(std::vector<std::unordered_set<uint32_t>> &adj) {
#ifdef PRINT_LOG
  printf("Inside void G_EXT::_construct_graph()\n");
#endif 

  // Each of enode is a hypergraph, we need to flatten them into multiple edges
  uint32_t acc = 0; 
  for (uint32_t ende_idx = 0; ende_idx < _num_ende; ende_idx++) {
    uint32_t num_child = _ende_child_ptr[ende_idx+1] - _ende_child_ptr[ende_idx];
    uint32_t ecls = _ende_ecls[ende_idx];
    for (uint32_t chd = 0; chd < num_child; chd++) {
      uint32_t cd_ecls_idx = _ende_child[acc];
      if (ecls != cd_ecls_idx) {
        adj[ecls].insert(cd_ecls_idx); 
      }
      acc++;
    }
  }

  MY_CHECK(acc == _ende_child.size(), "acc != _ende_child.size()");

#ifdef PRINT_LOG
  _print_adj(adj);
#endif
}


void G_EXT::_find_SCC(const std::vector<std::unordered_set<uint32_t>>& adj) {
#ifdef PRINT_LOG
  printf("Inside void G_EXT::_find_SCC()\n"); 
#endif
  using namespace boost;
  uint32_t acc = 0;

  typedef adjacency_list<vecS, vecS, directedS> Graph;
  Graph g(adj.size());

  for (size_t u = 0; u < adj.size(); u++) {
    for (auto v : adj[u]) {
      add_edge(u, v, g);
    }
  }

  std::vector<int> component(num_vertices(g));
  int num = strong_components(
    g, make_iterator_property_map(component.begin(), get(vertex_index, g))
  );

  std::unordered_map<int, std::vector<int>> comp_nodes;
  for (size_t i = 0; i < component.size(); i++) {
    comp_nodes[component[i]].push_back(i);
  }

  std::vector<std::vector<int>> real_sccs;
  std::vector<int> singletons;
  for (auto &kv : comp_nodes) {
    if (kv.second.size() > 1) {
      real_sccs.push_back(kv.second);
    } else {
      singletons.push_back(kv.second[0]);
    }
  }

  std::cout << "_num_ende: " << _num_ende 
            << ", _num_ecls: " << _num_ecls << "\n";

  if (!real_sccs.empty()) {
    std::cout << "Totally have " << real_sccs.size() << " SCCs in this graph\n";
    for (size_t i = 0; i < real_sccs.size(); i++) {
      std::cout << "SCC " << (i+1) << "(size = " << real_sccs[i].size() << ")"<< " {";
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
    acc = _num_ecls;
  }

  // printf("_num_ecls = %u, acc = %u\n", _num_ecls, acc); 
  MY_CHECK(_num_ecls == acc, "_num_ecls == acc");
}






// =========================== Print functions =========================== // 

void G_EXT::_print_input() const {
  std::cout << "\n========== G_EXT Internal State ==========\n";
  std::cout << "_num_ende: " << _num_ende << ", " << "_num_ecls: " << _num_ecls << "\n\n";

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
    for (uint32_t j = start; j < end; j++) {
      std::cout << _ende_child[j] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nEclass name -> id map:\n";
  for (auto &kv : ecls_name_to_id) {
    std::cout << "  " << kv.first << " -> " << kv.second << "\n";
  }

  std::cout << "\nRoot eclasses:\n";
  for (auto r : _root_eclass_id) {
    std::cout << "  " << r << "\n";
  }

  MY_CHECK(_ende_name.size() == _num_ende, "!= _num_ende 1");
  MY_CHECK(_ende_op.size() == _num_ende, "!= _num_ende 2");
  MY_CHECK(_ende_cost.size() == _num_ende, "!= _num_ende 3");
  MY_CHECK(_ende_ecls.size() == _num_ende, "!= _num_ende 4");

  std::cout << "==========================================\n\n";
}

void G_EXT::_print_adj(const std::vector<std::unordered_set<uint32_t>>& adj) const {
  printf("adj:\n"); 
  for (uint32_t fm = 0; fm < adj.size(); fm++) {
    printf("fm_node = %u, to nodes = [", fm); 
    for (const auto& elem : adj[fm]) {
      printf("%u, ", elem);
    }
    printf("]\n");
  }  
}