#include "g_ext.cuh"

#define threads 1024 // num GPU threads
#define FAST_BITMAP_CHECK true // Use bitmap strategy or not

/* ASPLOS_REBUTTAL */ 
// #define ASPLOS_REBUTTAL
// #define ASPLOS_REBUTTAL_SKEWNESS
// #define ASPLOS_REBUTTAL_DATA_TRANSFER_TIME
// #define ASPLOS_REBUTTAL_NUMBER_ITERATIONS
// #define ASPLOS_REBUTTAL_DAG_SHAPE
#define ASPLOS_REBUTTAL_WASTED_INSERTION_TEST
#define ASPLOS_REBUTTAL_BITMAP_FALSE_POSITIVE_RATE


/* DEBUG */
// #define PRINT_LOG
// #define CHECK_CYCLE // check for read input

/* GENERATE NEW DBENCHMARK for SmoothE */
// #define DUMP_NEW_JSON

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

#ifdef DUMP_NEW_JSON
  std::string out_path = input_path + ".egg.json";
  _dump_to_egg_json(out_path, tmp_num_ecls, tmp_num_ende,
    tmp_ende_name, tmp_ende_op, tmp_ende_cost,
    tmp_ende_ecls, tmp_ende_child, tmp_ende_child_ptr,
    tmp_root_eclass_id);
#endif

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
  
#ifdef ASPLOS_REBUTTAL_SKEWNESS  
  _print_degree_stats();  // Experiment 1. skewness
#endif 
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
  
#ifdef ASPLOS_REBUTTAL_DATA_TRANSFER_TIME
  auto t_parse_start = std::chrono::high_resolution_clock::now();
#endif 

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
  
#ifdef CHECK_CYCLE
  // ======================================
  // SCC check
  // ======================================
  {
    std::vector<std::vector<uint32_t>> adj(_num_ecls);
    // construct eclass-level adjacency
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
      std::cout << "Found " << scc_list.size() << " strongly connected components (cycles) after pruning.\n";
      for (size_t i = 0; i < scc_list.size(); ++i) {
        std::cout << "  SCC " << i << " (size=" << scc_list[i].size() << "): ";
        for (uint32_t e : scc_list[i]) std::cout << e << " ";
        std::cout << "\n";
      }
    } else {
      std::cout << "Graph is acyclic (no SCC found after pruning).\n";
    }
  }
#endif

//   // Construct _global_adj, _global_inv_adj --> There maybe cycle inside _global_adj and _global_inv_adj
//   _global_adj.resize(_num_ende, std::unordered_set<uint32_t>()); // ende -> ende
//   _global_inv_adj.resize(_num_ende, std::unordered_set<uint32_t>()); // ende -> ende
//   for (uint32_t ende = 0; ende < _num_ende; ende++) {
//     uint32_t start = _ende_child_ptr[ende]; 
//     uint32_t end = _ende_child_ptr[ende+1];
//     for (uint32_t jj = start; jj < end; jj++) {
//       uint32_t chd_ecls = _ende_child[jj];
//       for (uint32_t ende2 : _ecls_its_ende[chd_ecls]) {
//         // remove self-loop
//         if (ende != ende2) {
//           _global_adj[ende].insert(ende2); // update _global_adj 
//           _global_inv_adj[ende2].insert(ende); // update inv_adjs
//         }
//       }
//     }
//   }

//   // Compress _global_adj and _global_inv_adj to 1D 
//   acc_ptr = 0;
//   _global_adj_1D_ptr.push_back(acc_ptr);
//   for (uint32_t ende = 0; ende < _global_adj.size(); ende++) {
//     for (uint32_t to_ende : _global_adj[ende]) {
//       _global_adj_1D.push_back(to_ende);
//       acc_ptr++;
//     }
//     _global_adj_1D_ptr.push_back(acc_ptr);
//   }
  
//   acc_ptr = 0;
//   _global_inv_adj_1D_ptr.push_back(acc_ptr);
//   for (uint32_t ende = 0; ende < _global_inv_adj.size(); ende++) {
//     for (uint32_t to_ende : _global_inv_adj[ende]) {
//       _global_inv_adj_1D.push_back(to_ende);
//       acc_ptr++;
//     }
//     _global_inv_adj_1D_ptr.push_back(acc_ptr);
//   }

//   // Init size of _picked_adj and _picked_inv_adj
//   _picked_adj.resize(_global_adj_1D.size(), false); 
//   _picked_inv_adj.resize(_global_inv_adj_1D.size(), false);
//   _tmp_picked_adj.resize(_global_adj_1D.size(), false); 
//   _tmp_picked_inv_adj.resize(_global_inv_adj_1D.size(), false);  

// #ifdef PRINT_LOG
//   printf("_global_adj mapping (ende Id -> ende Id):\n");
//   _print_adj(_global_adj);
//   printf("_global_inv_adj mapping (ende Id -> ende Id):\n");
//   _print_adj(_global_inv_adj);

//   printf("_num_ende = %u\n", _num_ende);
//   printf("_global_adj:\n");
//   for (uint32_t ende = 0; ende < _global_adj.size(); ende++) {
//     printf("_global_adj[%u] (sz = %lu) = [", ende, _global_adj.size());
//     for (uint32_t to_ende : _global_adj[ende]) {
//       printf("%u, ", to_ende);
//     }
//     printf("]\n");
//   }
//   printf("\n");  
// #endif

//   for (uint32_t r_ecls : _root_eclass_id) {
//     for (uint32_t r : _ecls_its_ende[r_ecls]) {
//       _roots.push_back(r); 
//     }
//   }

//   // JUST USED FOR CHECK
//   MY_CHECK(_all_root_ende_1D.size() == _roots.size(), "_all_root_ende_1D.size() == _roots.size()");
//   for (uint32_t i = 0; i < _all_root_ende_1D.size(); i++) {
//     if (_all_root_ende_1D[i] != _roots[i]) {
//       exit(1);
//     }
//   }

//   for (uint32_t ende = 0; ende < _global_adj.size(); ende++) {
//     if (_global_adj[ende].size() == 0) {
//       _leaves.push_back(ende); 
//     }
//   }  
// Construct _global_adj_1D, _global_inv_adj_1D directly (CSR format),
  // avoiding std::unordered_set overhead for large e-graphs.
  // Note: There maybe cycle inside _global_adj_1D and _global_inv_adj_1D

  // 20260706 Claude optimized data preprocessing version starts 
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
  // 20260706 Claude optimized data preprocessing version ends 


#ifdef ASPLOS_REBUTTAL_DATA_TRANSFER_TIME
  auto t_parse_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> d = t_parse_end - t_parse_start;
  std::cout << "CSR flattern time: " << round_to((d.count()) * 1000, 0.001) << " (ms)\n";
#endif 

}

#ifdef ASPLOS_REBUTTAL_SKEWNESS
  double G_EXT::_compute_skewness(const std::vector<uint32_t>& degrees) {
    if (degrees.empty()) return 0.0;

    double n = static_cast<double>(degrees.size());
    double mean = 0.0;
    for (uint32_t d : degrees) mean += d;
    mean /= n;

    double m2 = 0.0, m3 = 0.0;
    for (uint32_t d : degrees) {
      double diff = static_cast<double>(d) - mean;
      m2 += diff * diff;
      m3 += diff * diff * diff;
    }
    m2 /= n;
    m3 /= n;

    double stddev = std::sqrt(m2);
    if (stddev < 1e-12) return 0.0; // 避免除以 0（所有 degree 都一樣時）

    return m3 / (stddev * stddev * stddev);
  }

  void G_EXT::_print_degree_stats() {
    std::vector<uint32_t> deg_NC(_num_ende);
    std::vector<uint32_t> deg_NN(_num_ende);

    for (uint32_t i = 0; i < _num_ende; i++) {
      uint32_t start = _ende_child_ptr[i];
      uint32_t end   = _ende_child_ptr[i+1];
      deg_NC[i] = end - start;  // e-node -> child e-class 的數量

      uint32_t nn_count = 0;
      for (uint32_t jj = start; jj < end; jj++) {
        uint32_t child_ecls = _ende_child[jj];
        nn_count += _ecls_its_ende[child_ecls].size();  // 該 child e-class 裡有幾個 e-node
      }
      deg_NN[i] = nn_count;  // e-node -> e-node 的展開數量（不去重）
    }

    // 算 average（全部 e-node）
    double sum_NC = 0.0, sum_NN = 0.0;
    for (uint32_t i = 0; i < _num_ende; i++) {
      sum_NC += deg_NC[i];
      sum_NN += deg_NN[i];
    }
    double avg_NC = sum_NC / static_cast<double>(_num_ende);
    double avg_NN = sum_NN / static_cast<double>(_num_ende);

    double skew_NC = _compute_skewness(deg_NC);
    double skew_NN = _compute_skewness(deg_NN);

    // 找出 max 的位置（e-node index），而不只是數值
    uint32_t max_NC_idx = static_cast<uint32_t>(std::distance(deg_NC.begin(), std::max_element(deg_NC.begin(), deg_NC.end())));
    uint32_t max_NN_idx = static_cast<uint32_t>(std::distance(deg_NN.begin(), std::max_element(deg_NN.begin(), deg_NN.end())));
    uint32_t max_NC = deg_NC[max_NC_idx];
    uint32_t max_NN = deg_NN[max_NN_idx];

    printf("[Degree Stats] avg(d_NC) = %.4f, avg(d_NN) = %.4f\n", avg_NC, avg_NN);
    printf("[Degree Stats] skewness(d_NC) = %.4f, skewness(d_NN) = %.4f\n", skew_NC, skew_NN);
    printf("[Degree Stats] max(d_NC) = %u (ende_idx=%u), max(d_NN) = %u (ende_idx=%u)\n",
          max_NC, max_NC_idx, max_NN, max_NN_idx);

    // ===== 檢查這個 max degree 的 e-node 是否為 root =====
    std::unordered_set<uint32_t> root_set(_roots.begin(), _roots.end()); // 轉成 set，查詢 O(1)
    auto is_root_ende = [&](uint32_t ende_idx) -> bool {
      return root_set.find(ende_idx) != root_set.end();
    };

    bool max_NC_is_root = is_root_ende(max_NC_idx);
    bool max_NN_is_root = is_root_ende(max_NN_idx);

    printf("[Degree Stats] max(d_NC) ende is root? %s\n", max_NC_is_root ? "YES" : "NO");
    printf("[Degree Stats] max(d_NN) ende is root? %s\n", max_NN_is_root ? "YES" : "NO");
    printf("[Degree Stats] max(d_NC) ende belongs to eclass=%u\n", _ende_ecls[max_NC_idx]);
    printf("[Degree Stats] max(d_NN) ende belongs to eclass=%u\n", _ende_ecls[max_NN_idx]);

    // ===== 新增：排除所有 root e-node 之後，重新計算 avg / skewness / max =====
    std::vector<uint32_t> deg_NC_noroot, deg_NN_noroot;
    deg_NC_noroot.reserve(_num_ende);
    deg_NN_noroot.reserve(_num_ende);
    for (uint32_t i = 0; i < _num_ende; i++) {
      if (root_set.find(i) != root_set.end()) continue; // 排除 root
      deg_NC_noroot.push_back(deg_NC[i]);
      deg_NN_noroot.push_back(deg_NN[i]);
    }

    double avg_NC_noroot = 0.0, avg_NN_noroot = 0.0;
    double skew_NC_noroot = 0.0, skew_NN_noroot = 0.0;
    uint32_t max_NC_noroot = 0, max_NN_noroot = 0;

    if (!deg_NC_noroot.empty()) {
      double sum_NC_nr = 0.0, sum_NN_nr = 0.0;
      for (size_t i = 0; i < deg_NC_noroot.size(); i++) {
        sum_NC_nr += deg_NC_noroot[i];
        sum_NN_nr += deg_NN_noroot[i];
      }
      avg_NC_noroot = sum_NC_nr / static_cast<double>(deg_NC_noroot.size());
      avg_NN_noroot = sum_NN_nr / static_cast<double>(deg_NN_noroot.size());

      skew_NC_noroot = _compute_skewness(deg_NC_noroot);
      skew_NN_noroot = _compute_skewness(deg_NN_noroot);

      max_NC_noroot = *std::max_element(deg_NC_noroot.begin(), deg_NC_noroot.end());
      max_NN_noroot = *std::max_element(deg_NN_noroot.begin(), deg_NN_noroot.end());
    }

    printf("[Degree Stats][excl. root] avg(d_NC) = %.4f, avg(d_NN) = %.4f\n", avg_NC_noroot, avg_NN_noroot);
    printf("[Degree Stats][excl. root] skewness(d_NC) = %.4f, skewness(d_NN) = %.4f\n", skew_NC_noroot, skew_NN_noroot);
    printf("[Degree Stats][excl. root] max(d_NC) = %u, max(d_NN) = %u\n", max_NC_noroot, max_NN_noroot);
    printf("[Degree Stats][excl. root] num e-nodes excluded (root count) = %zu\n", _roots.size());

    // ===== 新增：把每個 e-node 的 degree dump 到檔案（單一檔案，直接覆蓋）=====
    std::ofstream ofs("rebuttal_exp/degree_skew_exp.out"); // 預設 trunc mode，每次重跑會覆蓋舊檔
    ofs << "# NUM_ENDE: " << _num_ende << "\n";
    ofs << "# AVG_NC: " << avg_NC << " AVG_NN: " << avg_NN << "\n";
    ofs << "# SKEW_NC: " << skew_NC << " SKEW_NN: " << skew_NN << "\n";
    ofs << "# MAX_NC: " << max_NC << " MAX_NN: " << max_NN << "\n";
    ofs << "# MAX_NC_IS_ROOT: " << (max_NC_is_root ? 1 : 0)
        << " MAX_NN_IS_ROOT: " << (max_NN_is_root ? 1 : 0) << "\n";
    ofs << "# AVG_NC_NOROOT: " << avg_NC_noroot << " AVG_NN_NOROOT: " << avg_NN_noroot << "\n";
    ofs << "# SKEW_NC_NOROOT: " << skew_NC_noroot << " SKEW_NN_NOROOT: " << skew_NN_noroot << "\n";
    ofs << "# MAX_NC_NOROOT: " << max_NC_noroot << " MAX_NN_NOROOT: " << max_NN_noroot << "\n";
    ofs << "# deg_NC deg_NN\n";
    for (uint32_t i = 0; i < _num_ende; i++) {
      ofs << deg_NC[i] << " " << deg_NN[i] << "\n";
    }
    ofs.close();
  }
#endif 

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
  
  #ifdef ASPLOS_REBUTTAL_NUMBER_ITERATIONS
    uint32_t num_iter = 0;
  #endif

  bool updated = true; 
  std::vector<int> level;
  level.resize(_num_ende, -1); // init, needed
  _new_inserted.assign(_num_ende, false); // init, needed
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
    #ifdef ASPLOS_REBUTTAL_NUMBER_ITERATIONS
      num_iter++;
    #endif
  }
  
  #ifdef ASPLOS_REBUTTAL_NUMBER_ITERATIONS
    std::cout << "CPU num_iter = " << num_iter << "\n";
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

#ifdef ASPLOS_REBUTTAL_DAG_SHAPE
  // Compute depth/width of the final DAG passed to the ILP solver
  // (based on a fresh top-down levelization restricted to _in_subgraph nodes)
  {
    std::vector<int> final_level(_num_ende, -1);
    std::vector<uint32_t> final_indeg(_num_ende, 0);

    // build in-degree restricted to final subgraph nodes and picked_adj edges
    for (uint32_t fm_ende = 0; fm_ende < _num_ende; fm_ende++) {
      if (!_in_subgraph[fm_ende]) continue;
      uint32_t start = _global_adj_1D_ptr[fm_ende];
      uint32_t end   = _global_adj_1D_ptr[fm_ende + 1];
      for (uint32_t k = start; k < end; k++) {
        if (!_picked_adj[k]) continue;
        uint32_t to_ende = _global_adj_1D[k];
        if (!_in_subgraph[to_ende]) continue;
        final_indeg[to_ende]++;
      }
    }

    std::queue<uint32_t> q;
    for (uint32_t i = 0; i < _num_ende; i++) {
      if (_in_subgraph[i] && final_indeg[i] == 0) q.push(i);
    }

    int cur_level = 0;
    std::vector<uint32_t> width_per_level;
    while (!q.empty()) {
      size_t layer_size = q.size();
      width_per_level.push_back(static_cast<uint32_t>(layer_size));
      for (size_t t = 0; t < layer_size; t++) {
        uint32_t u = q.front(); q.pop();
        final_level[u] = cur_level;
        uint32_t start = _global_adj_1D_ptr[u];
        uint32_t end   = _global_adj_1D_ptr[u + 1];
        for (uint32_t k = start; k < end; k++) {
          if (!_picked_adj[k]) continue;
          uint32_t v = _global_adj_1D[k];
          if (!_in_subgraph[v]) continue;
          if (--final_indeg[v] == 0) q.push(v);
        }
      }
      cur_level++;
    }

    uint32_t depth = static_cast<uint32_t>(cur_level);
    uint32_t max_width = 0;
    uint64_t total_nodes_in_subgraph = 0;
    for (uint32_t w : width_per_level) {
      if (w > max_width) max_width = w;
      total_nodes_in_subgraph += w;
    }
    double avg_width = (depth > 0)
      ? (static_cast<double>(total_nodes_in_subgraph) / depth)
      : 0.0;

    std::cout << "[DAG Shape] depth = " << depth
               << ", max_width = " << max_width
               << ", avg_width = " << avg_width
               << ", total_nodes_in_final_subgraph = " << total_nodes_in_subgraph
               << "\n";
  }
#endif  
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
      #ifdef ASPLOS_REBUTTAL_WASTED_INSERTION_TEST
        _total_revert_count.fetch_add(1, std::memory_order_relaxed);
      #endif 

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
  // 將所有 _new_inserted 設為 false，只捕捉「這一輪」剛被平行的 threads 插入的點
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
    const uint32_t BITMAP_SIZE = 64;
    uint8_t chd_flags[BITMAP_SIZE] = {0};

    #ifdef ASPLOS_REBUTTAL_BITMAP_FALSE_POSITIVE_RATE
      std::vector<uint32_t> valid_chds; // 新增：保留精確版比對用
      std::vector<uint32_t> valid_pars; // 用來記錄哪些 parent 通過初步篩選
    #endif 

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

      #ifdef ASPLOS_REBUTTAL_BITMAP_FALSE_POSITIVE_RATE
        valid_chds.push_back(chd_ende); // 新增
      #endif 
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

      #ifdef ASPLOS_REBUTTAL_BITMAP_FALSE_POSITIVE_RATE
        valid_pars.push_back(par_ende); // 新增
      #endif 
    }

    // ============================================================
    // Step 3: 判斷有效性（跟 GPU kernel 完全一致）
    // ============================================================
    bool cond = (is_td) ? (has_par) : (has_chd);
    int given_lvl = (is_td) ? (max_par_lvl + 1) : (min_chd_lvl - 1);
    bool valid = (cond) && (max_par_lvl < min_chd_lvl) && (!has_overlap_b) && (given_lvl >= 0);

    #ifdef ASPLOS_REBUTTAL_BITMAP_FALSE_POSITIVE_RATE
      // ===== 精確比對（ground truth）=====
      bool has_overlap_exact = false;
      for (uint32_t p : valid_pars) {
        for (uint32_t c : valid_chds) {
          if (p == c) { has_overlap_exact = true; break; }
        }
        if (has_overlap_exact) break;
      }
      if (has_overlap_b && !has_overlap_exact) {
        _total_bitmap_fp_count.fetch_add(1, std::memory_order_relaxed);
      }
      _total_overlap_check_count.fetch_add(1, std::memory_order_relaxed);
    #endif 


    // ===== 唯一一個 if (valid) 區塊，內部用 macro 決定要不要多做計數 =====
    if (valid) {
      update = true;
      picked_endes[ende] = true;
      level[ende] = given_lvl;
      _new_inserted[ende] = true;

      #if defined(ASPLOS_REBUTTAL_WASTED_INSERTION_TEST) || defined(ASPLOS_REBUTTAL_BITMAP_FALSE_POSITIVE_RATE)
        _total_insert_count.fetch_add(1, std::memory_order_relaxed);
      #endif

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
  #ifdef ASPLOS_REBUTTAL_WASTED_INSERTION_TEST
    _total_insert_count.store(0);
    _total_revert_count.store(0);
  #endif
  #ifdef ASPLOS_REBUTTAL_BITMAP_FALSE_POSITIVE_RATE
    _total_bitmap_fp_count.store(0);
    _total_overlap_check_count.store(0);
  #endif
  
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

    printf("[CPU+bitmap] iter %u: tp=%d, bu=%d\n", num_iter, tp, bu);

    if (num_iter > 1000) {  // 安全閥，避免真的卡死
      std::cerr << "[WARNING] CPU+bitmap version exceeded 1000 iterations, likely livelock\n";
      break;
    }
  }
  std::cout << "[CPU+bitmap] Total iterations = " << num_iter << "\n";

  #ifdef ASPLOS_REBUTTAL_WASTED_INSERTION_TEST
    // ===== Q2: wasted work report =====
    uint64_t total_insert = _total_insert_count.load();
    uint64_t total_revert = _total_revert_count.load();
    double wasted_pct_of_insert = (total_insert > 0) 
      ? (100.0 * static_cast<double>(total_revert) / static_cast<double>(total_insert)) 
      : 0.0;

    std::cout << "[Q2 Wasted Work] total_insert = " << total_insert
              << ", total_revert = " << total_revert
              << ", wasted_pct_of_insert = " << wasted_pct_of_insert << "%\n";  
  #endif

#ifdef ASPLOS_REBUTTAL_BITMAP_FALSE_POSITIVE_RATE
    uint64_t total_fp = _total_bitmap_fp_count.load();
    uint64_t total_check = _total_overlap_check_count.load();
    double fp_pct = (total_check > 0) 
      ? (100.0 * static_cast<double>(total_fp) / static_cast<double>(total_check)) 
      : 0.0;
    std::cout << "[Q4 Bitmap FP] total_fp = " << total_fp 
              << ", total_check = " << total_check 
              << ", fp_pct = " << fp_pct << "%\n";
  #endif  
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

__global__ void _relax_enodes_kernel(const uint32_t num_endes, 
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

__global__ void _gpu_init_picked_adj (
  const uint32_t num_endes, 
  const uint32_t* dvc_global_adj_1D,
  const uint32_t* dvc_global_adj_1D_ptr,
  const uint32_t* dvc_global_inv_adj_1D,
  const uint32_t* dvc_global_inv_adj_1D_ptr,
  const uint32_t* dvc_picked_endes,
  uint32_t* dvc_picked_adj, 
  uint32_t* dvc_picked_inv_adj,
  uint32_t* dvc_indegree, 
  uint32_t* dvc_inv_indegree) 
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num_endes) return;

  if (!dvc_picked_endes[t_idx]) return;
  // init dvc_picked_adj
  uint32_t start = dvc_global_adj_1D_ptr[t_idx];
  uint32_t end = dvc_global_adj_1D_ptr[t_idx+1];
  for (uint32_t k = start; k < end; k++) {
    const uint32_t to_ende = dvc_global_adj_1D[k];
    if (dvc_picked_endes[to_ende]) {
      dvc_picked_adj[k] = 1; // true
      atomicAdd(&dvc_indegree[to_ende], 1);
      atomicAdd(&dvc_inv_indegree[t_idx], 1);
    }
  }

  // init dvc_picked_inv_adj
  start = dvc_global_inv_adj_1D_ptr[t_idx];
  end = dvc_global_inv_adj_1D_ptr[t_idx+1];
  for (uint32_t k = start; k < end; k++) {
    const uint32_t to_ende = dvc_global_inv_adj_1D[k];
    if (dvc_picked_endes[to_ende]) {
      dvc_picked_inv_adj[k] = 1; // true
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
#ifdef ASPLOS_REBUTTAL_DATA_TRANSFER_TIME
  float H2D_runtime = 0.;
  cudaEvent_t startH2D_runtime, stopH2D_runtime;
  cudaEventCreate(&startH2D_runtime);
  cudaEventCreate(&stopH2D_runtime);    
  cudaEventRecord(startH2D_runtime);
#endif

  int blocks = (_num_ende + threads - 1) / threads;

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

 #ifdef ASPLOS_REBUTTAL_DATA_TRANSFER_TIME 
  cudaEventRecord(stopH2D_runtime);
  cudaEventSynchronize(stopH2D_runtime);
  cudaEventElapsedTime(&H2D_runtime, startH2D_runtime, stopH2D_runtime);                                    
  printf("H2D_runtime runtime: %.2f (ms)\n", H2D_runtime);
#endif

  _gpu_warm_up<<<blocks,threads>>>(_num_ende, _dvc_level); // Note: place this GPU kernel here as the warm-up for GPU; w/o this, the kernel runtime below will be extremely long. Need to remrmber!

  float gpuTime = 0.;
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);    
  cudaEventRecord(startGPU);

  blocks = (_num_ende + threads - 1) / threads;
  while (h_update) {
    h_update = false;
    cudaMemcpy(_dvc_update, &h_update, sizeof(bool), cudaMemcpyHostToDevice);
    _relax_enodes_kernel<<<blocks, threads>>>(_num_ende,
      dvc_ende_cost,
      dvc_ende_ecls,
      dvc_ende_child,
      dvc_ende_child_ptr,
      _dvc_update,
      dvc_cost_of_ecls,
      dvc_cost_of_ende,
      _dvc_best_ende_of_ecls);
    cudaCheckErrors("_relax_enodes_kernel failed");
    cudaMemcpy(&h_update, _dvc_update, sizeof(bool), cudaMemcpyDeviceToHost);
  }
  
  // Get the best tree cost on GPU
  _get_tree_cost<<<1, 1>>>(_root_eclass_id.size(), dvc_cost_of_ecls, _dvc_root_eclass_id, dvc_tree_ret);

  // Get best tree cost 
  double tree_ret = 0.0; 
  cudaMemcpy(&tree_ret, dvc_tree_ret, 1*sizeof(double), cudaMemcpyDeviceToHost);
  
  blocks = (_num_ecls + threads - 1) / threads;
  _resolve_best_ende<<<blocks, threads>>>(
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
    blocks = (num + threads - 1) / threads;
    _get_dag_cost<<<blocks, threads>>>(
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
  // printf("GPU tree construction runtime: %.2f (ms)\n", gpuTime);

  return std::make_pair(tree_ret, dag_ret);
}

void G_EXT::_gpu_DAG_construction() {
#ifdef PRINT_LOG
  printf("\nInside _gpu_DAG_construction()\n");
#endif

  int blocks = (_num_ende + threads - 1) / threads;
  _gpu_init_picked_adj<<<blocks,threads>>>(
    _num_ende, 
    _dvc_global_adj_1D,
    _dvc_global_adj_1D_ptr,
    _dvc_global_inv_adj_1D,
    _dvc_global_inv_adj_1D_ptr,
    _dvc_picked_endes,
    _dvc_picked_adj, _dvc_picked_inv_adj,
    _dvc_indegree, _dvc_inv_indegree);
  cudaCheckErrors("_gpu_init_picked_adj failed");
  
  cudaMalloc(&_dvc_base_indegree, _num_ende * sizeof(uint32_t));
  cudaMalloc(&_dvc_base_inv_indegree, _num_ende * sizeof(uint32_t));
  cudaMalloc(&_dvc_new_inserted, _num_ende * sizeof(bool));
  cudaMemset(_dvc_new_inserted, 0, _num_ende * sizeof(bool));

  _gpu_update_deg_init_level(true, true);
  cudaMemcpy(_dvc_base_indegree, _dvc_indegree, _num_ende * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(_dvc_base_inv_indegree, _dvc_inv_indegree, _num_ende * sizeof(uint32_t), cudaMemcpyDeviceToDevice);


#ifdef PRINT_LOG
  _print_gpu_init_picked_adj<<<1,1>>>(_num_ende,
    _num_ende,
    _global_adj_1D.size(),
    _global_adj_1D.size(),
    _dvc_picked_endes,
    _dvc_indegree,
    _dvc_inv_indegree,
    _dvc_picked_adj,
    _dvc_picked_inv_adj,
    _dvc_global_adj_1D_ptr, 
    _dvc_global_inv_adj_1D_ptr,
    _dvc_global_adj_1D,
    _dvc_global_inv_adj_1D); // debug
  cudaCheckErrors("_print_gpu_init_picked_adj failed"); // debug 
  cudaDeviceSynchronize(); // debug
  _print_gpu_degree<<<1,1>>>(_num_ende, _num_ende, _dvc_picked_endes, _dvc_indegree, _dvc_inv_indegree);
  cudaCheckErrors("_print_gpu_degree failed"); // debug 
  cudaDeviceSynchronize(); // debug
#endif

  float gpu_reconst_Time = 0.;
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);    
  cudaEventRecord(startGPU);

  #ifdef ASPLOS_REBUTTAL_NUMBER_ITERATIONS
    uint32_t num_iter = 0;
  #endif

  bool updated = true;
  bool dvc_update_flag = false; 

  while (updated) {
    _gpu_update_deg_init_level(updated, false);
    _gpu_BFS(
      false, 
      _dvc_inv_indegree, 
      _dvc_picked_endes, 
      _dvc_global_inv_adj_1D, 
      _dvc_global_inv_adj_1D_ptr, 
      _dvc_picked_inv_adj, 
      _dvc_level);
      
    cudaMemcpy(_dvc_update, &dvc_update_flag, sizeof(bool), cudaMemcpyHostToDevice);
    bool tp = _gpu_insertion(true);
    
    _gpu_update_deg_init_level(tp, false);
    _gpu_BFS(
      true, 
      _dvc_indegree, 
      _dvc_picked_endes, 
      _dvc_global_adj_1D, 
      _dvc_global_adj_1D_ptr, 
      _dvc_picked_adj, 
      _dvc_level);
      
    cudaMemcpy(_dvc_update, &dvc_update_flag, sizeof(bool), cudaMemcpyHostToDevice);
    bool bu = _gpu_insertion(false);

    updated = tp || bu;
    #ifdef ASPLOS_REBUTTAL_NUMBER_ITERATIONS
      num_iter++;
      printf("Now num_iter = %d\n", num_iter);
    #endif  
  }  

  std::vector<uint32_t> in_subgraph_gpu; 
  in_subgraph_gpu.resize(_num_ende, 0);
  cudaMemcpy(in_subgraph_gpu.data(), _dvc_picked_endes, _num_ende*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  
  uint32_t aaa = 0;
  for (uint32_t i = 0; i < in_subgraph_gpu.size(); ++i ) {
    if (in_subgraph_gpu[i]) aaa++;
  }
  printf("#alive e-nodes after GPU DAG construction = %u\n", aaa);

  // Re-constructing data for second times ILP solver
  _ecls_its_ende_newG_GPU.resize(_ecls_its_ende.size(), std::vector<uint32_t>());
  for (uint32_t ecls = 0; ecls < _ecls_its_ende.size(); ecls++) {
    for (uint32_t ende : _ecls_its_ende[ecls]) {
      if (!_in_subgraph[ende]) continue;
      _ecls_its_ende_newG_GPU[ecls].push_back(ende);
    }
  }

  cudaEventRecord(stopGPU);
  cudaEventSynchronize(stopGPU);
  cudaEventElapsedTime(&gpu_reconst_Time, startGPU, stopGPU);
  
  #ifdef ASPLOS_REBUTTAL_NUMBER_ITERATIONS
    std::cout << "GPU num_iter = " << num_iter << "\n";
  #endif
  printf("GPU DAG construction runtime: %.2f (ms)\n", gpu_reconst_Time);  
}

__global__ void _update_deg_kl2 (
  const uint32_t num_endes, 
  const uint32_t* dvc_global_adj_1D,
  const uint32_t* dvc_global_adj_1D_ptr,
  const uint32_t* dvc_picked_endes,
  const uint32_t* dvc_picked_adj, 
  uint32_t* dvc_indegree, uint32_t* dvc_inv_indegree) 
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ((t_idx >= num_endes) || (!dvc_picked_endes[t_idx])) return;

  // assign dvc_indegree and dvc_inv_indegree
  for (uint32_t k = dvc_global_adj_1D_ptr[t_idx]; k < dvc_global_adj_1D_ptr[t_idx+1]; k++) {
    if (!dvc_picked_adj[k]) continue;
    const uint32_t to_ende = dvc_global_adj_1D[k];
    if (dvc_picked_endes[to_ende]) {
      atomicAdd(&dvc_indegree[to_ende], 1);
      atomicAdd(&dvc_inv_indegree[t_idx], 1);
    } 
  }
}

void G_EXT::_gpu_update_deg_init_level2(const bool need_to_update) 
{
  if (!need_to_update) return;

  // Init 
  cudaMemset(_dvc_indegree, 0, _num_ende*sizeof(uint32_t));
  cudaMemset(_dvc_inv_indegree, 0, _num_ende*sizeof(uint32_t));
  cudaMemset(_dvc_level, 0xFF, _num_ende*sizeof(int));
  
  int blocks = (_num_ende + threads - 1) / threads;
  _update_deg_kl2<<<blocks, threads>>>(
    _num_ende,
    _dvc_global_adj_1D,
    _dvc_global_adj_1D_ptr,
    _dvc_picked_endes,
    _dvc_picked_adj,
    _dvc_indegree,
    _dvc_inv_indegree);
  cudaCheckErrors("_update_deg_kl2 failed");

#ifdef PRINT_LOG
  _print_gpu_degree<<<1,1>>>(_num_ende, _num_ende, _dvc_picked_endes, _dvc_indegree, _dvc_inv_indegree);
  cudaCheckErrors("_print_gpu_degree failed"); // debug 
  cudaDeviceSynchronize(); // debug  
#endif
}

__global__ void _update_deg_kl(
  const uint32_t num,
  const bool is_init,
  const uint32_t* dvc_picked_endes,
  const bool* dvc_new_inserted,
  const uint32_t* dvc_global_adj_1D_ptr,
  const uint32_t* dvc_global_adj_1D,
  const uint32_t* dvc_picked_adj,
  const uint32_t* dvc_global_inv_adj_1D_ptr,
  const uint32_t* dvc_global_inv_adj_1D,
  const uint32_t* dvc_picked_inv_adj,
  uint32_t* dvc_indegree,
  uint32_t* dvc_inv_indegree)
{
  uint32_t fm_ende = blockIdx.x * blockDim.x + threadIdx.x;
  if (fm_ende >= num) return;

  if (is_init) {
    if (!dvc_picked_endes[fm_ende]) return;
    for (uint32_t k = dvc_global_adj_1D_ptr[fm_ende]; k < dvc_global_adj_1D_ptr[fm_ende+1]; k++) {
      if (!dvc_picked_adj[k]) continue;
      const uint32_t to_ende = dvc_global_adj_1D[k];
      atomicAdd(&dvc_indegree[to_ende], 1);
      atomicAdd(&dvc_inv_indegree[fm_ende], 1);
    }
  } else {
    if (!dvc_new_inserted[fm_ende]) return; // only update those for newly inserted e-nodes

    // assign dvc_indegree and dvc_inv_indegree
    for (uint32_t k = dvc_global_adj_1D_ptr[fm_ende]; k < dvc_global_adj_1D_ptr[fm_ende+1]; k++) {
      const uint32_t to_ende = dvc_global_adj_1D[k];
      if (dvc_picked_endes[to_ende]) {
        atomicAdd(&dvc_indegree[to_ende], 1);
        atomicAdd(&dvc_inv_indegree[fm_ende], 1);
      } 
    }    
  }
}

// void G_EXT::_gpu_update_deg_init_level(
//   const bool need_to_update, 
//   const bool is_init) 
// {
//   if (!need_to_update) return;

//   cudaMemset(_dvc_level, 0xFF, _num_ende*sizeof(int)); 

//   if (is_init) {
//     cudaMemset(_dvc_indegree, 0, _num_ende*sizeof(uint32_t));
//     cudaMemset(_dvc_inv_indegree, 0, _num_ende*sizeof(uint32_t));
//   } else {
//     cudaMemcpy(_dvc_indegree, _dvc_base_indegree, _num_ende*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
//     cudaMemcpy(_dvc_inv_indegree, _dvc_base_inv_indegree, _num_ende*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
//   }

//   int blocks = (_num_ende + threads - 1) / threads;
//   _update_deg_kl<<<blocks, threads>>>(
//     _num_ende,
//     is_init,
//     _dvc_picked_endes,
//     _dvc_new_inserted,
//     _dvc_global_adj_1D_ptr,
//     _dvc_global_adj_1D,
//     _dvc_picked_adj,
//     _dvc_global_inv_adj_1D_ptr,
//     _dvc_global_inv_adj_1D,
//     _dvc_picked_inv_adj,
//     _dvc_indegree,
//     _dvc_inv_indegree
//   );
//   cudaCheckErrors("_update_deg_kl failed");
// }

__global__ void _update_deg_full_kl(
  const uint32_t num_ende,
  const uint32_t* dvc_picked_endes,
  const uint32_t* dvc_global_adj_1D_ptr,
  const uint32_t* dvc_global_adj_1D,
  const uint32_t* dvc_picked_adj,
  uint32_t* dvc_indegree,
  uint32_t* dvc_inv_indegree)
{
  uint32_t fm_ende = blockIdx.x * blockDim.x + threadIdx.x;
  if (fm_ende >= num_ende) return;
  if (!dvc_picked_endes[fm_ende]) return;
  uint32_t start = dvc_global_adj_1D_ptr[fm_ende];
  uint32_t end   = dvc_global_adj_1D_ptr[fm_ende + 1];
  for (uint32_t k = start; k < end; k++) {
    if (!dvc_picked_adj[k]) continue;
    const uint32_t to_ende = dvc_global_adj_1D[k];
    atomicAdd(&dvc_indegree[to_ende], 1);
    atomicAdd(&dvc_inv_indegree[fm_ende], 1);
  }
}

void G_EXT::_gpu_update_deg_init_level(
  const bool need_to_update, 
const bool is_init) 
{
  if (!need_to_update) return;
  cudaMemset(_dvc_level, 0xFF, _num_ende*sizeof(int));
  cudaMemset(_dvc_indegree, 0, _num_ende*sizeof(uint32_t));
  cudaMemset(_dvc_inv_indegree, 0, _num_ende*sizeof(uint32_t));
  int blocks = (_num_ende + threads - 1) / threads;
  _update_deg_full_kl<<<blocks, threads>>>(
    _num_ende, _dvc_picked_endes,
    _dvc_global_adj_1D_ptr, _dvc_global_adj_1D, _dvc_picked_adj,
    _dvc_indegree, _dvc_inv_indegree);
  cudaCheckErrors("_update_deg_full_kl failed");
}

__global__ void _init_q1(const uint32_t num_ende, 
  const uint32_t* dvc_indegree,
  const uint32_t* dvc_picked_endes,
  uint32_t* q,
  uint32_t* q_sz,
  uint32_t* q_num)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num_ende) return;

  if ((dvc_indegree[t_idx] == 0) && dvc_picked_endes[t_idx]) {
    uint32_t pos = atomicAdd(q_sz, 1);
    q[pos] = t_idx; // Note: t_idx is ende
  }
  *q_num = *q_sz;
}

__global__ void _gpu_cycle_revert_kl(
  const uint32_t num_ende,
  bool* dvc_new_inserted,
  const uint32_t* dvc_deg,
  uint32_t* dvc_picked_endes,
  int* dvc_level,
  const uint32_t* dvc_global_adj_1D_ptr,
  const uint32_t* dvc_global_adj_1D,
  const uint32_t* dvc_global_inv_adj_1D_ptr,
  const uint32_t* dvc_global_inv_adj_1D,
  uint32_t* dvc_picked_adj,
  uint32_t* dvc_picked_inv_adj)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_ende) return;
  if (!(dvc_new_inserted[i] && dvc_deg[i] > 0)) return;

  dvc_picked_endes[i] = 0;
  dvc_new_inserted[i] = false;
  dvc_level[i] = -1;

  uint32_t start = dvc_global_adj_1D_ptr[i];
  uint32_t end   = dvc_global_adj_1D_ptr[i + 1];
  for (uint32_t k = start; k < end; k++) {
    dvc_picked_adj[k] = 0;
    uint32_t chd = dvc_global_adj_1D[k];
    uint32_t cs = dvc_global_inv_adj_1D_ptr[chd];
    uint32_t ce = dvc_global_inv_adj_1D_ptr[chd + 1];
    for (uint32_t kk = cs; kk < ce; kk++) {
      if (dvc_global_inv_adj_1D[kk] == i) { dvc_picked_inv_adj[kk] = 0; break; }
    }
  }

  uint32_t p_start = dvc_global_inv_adj_1D_ptr[i];
  uint32_t p_end   = dvc_global_inv_adj_1D_ptr[i + 1];
  for (uint32_t k = p_start; k < p_end; k++) {
    dvc_picked_inv_adj[k] = 0;
    uint32_t par = dvc_global_inv_adj_1D[k];
    uint32_t ps = dvc_global_adj_1D_ptr[par];
    uint32_t pe = dvc_global_adj_1D_ptr[par + 1];
    for (uint32_t kk = ps; kk < pe; kk++) {
      if (dvc_global_adj_1D[kk] == i) { dvc_picked_adj[kk] = 0; break; }
    }
  }
}

__global__ void _gpu_BFS_kl(
  const uint32_t num, 
  const uint32_t* dvc_global_adj_1D_ptr,
  const uint32_t* dvc_global_adj_1D,
  const uint32_t q_sp,
  uint32_t* q,
  uint32_t* q_sz,
  int* level,
  const int cur_level,
  const uint32_t* dvc_picked_adj,
  uint32_t* deg)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num) return;

  uint32_t u = q[q_sp + t_idx];
  level[u] = cur_level;
  
  for (uint32_t k = dvc_global_adj_1D_ptr[u]; k < dvc_global_adj_1D_ptr[u+1]; k++) {
    if (!dvc_picked_adj[k]) continue;
    const uint32_t v = dvc_global_adj_1D[k];
    uint32_t old_deg = atomicSub(&deg[v], 1);
    if (old_deg == 1) {
      uint32_t pos = atomicAdd(q_sz, 1);
      q[pos] = v;
    }        
  }
}

__global__ void _reverse_level(  
  const uint32_t num, 
  const int cur_level,
  int* level) 
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num) return;
  
  if (level[t_idx] != -1) {
    level[t_idx] = cur_level - level[t_idx] - 1;
  }
}

void G_EXT::_gpu_BFS(
  const bool is_td, 
  uint32_t* deg, 
  const uint32_t* picked_endes,
  const uint32_t* global_adj_1D,
  const uint32_t* global_adj_1D_ptr,
  const uint32_t* picked_adj,
  int* level)
{
  cudaMemset(_dvc_q, 0, _num_ende*sizeof(uint32_t));
  cudaMemset(_dvc_q_sz, 0, 1*sizeof(uint32_t));

  // init queue
  int blocks = (_num_ende + threads - 1) / threads;
  _init_q1<<<blocks,threads>>>(
    _num_ende, 
    deg,
    picked_endes,
    _dvc_q, _dvc_q_sz, _dvc_q_num);
  cudaCheckErrors("_init_q1 failed");
  
  // BFS: top-down levelization 
  uint32_t q_sz = 0;
  cudaMemcpy(&q_sz, _dvc_q_sz, 1*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  uint32_t q_sp = 0;
  uint32_t num = q_sz - q_sp;  
  int cur_level = 0;

  while (num > 0) {
    blocks = (num + threads - 1) / threads;
    _gpu_BFS_kl<<<blocks, threads>>>(
      num,
      global_adj_1D_ptr,
      global_adj_1D,
      q_sp,
      _dvc_q,
      _dvc_q_sz,
      level,
      cur_level, 
      picked_adj,
      deg);
    cudaCheckErrors("_gpu_BFS_kl failed");

    cudaMemcpy(&q_sz, _dvc_q_sz, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    q_sp += num;
    num = q_sz - q_sp;
    cur_level++; 
  }

  _max_level = cur_level;

  // ===== cycle revert：永遠用「絕對」正向/反向 adjacency（member variable），與 is_td 無關 =====
  {
    int rblocks = (_num_ende + threads - 1) / threads;
    _gpu_cycle_revert_kl<<<rblocks, threads>>>(
      _num_ende, _dvc_new_inserted, deg, _dvc_picked_endes, level,
      _dvc_global_adj_1D_ptr, _dvc_global_adj_1D,           // <-- 永遠是正向
      _dvc_global_inv_adj_1D_ptr, _dvc_global_inv_adj_1D,   // <-- 永遠是反向
      _dvc_picked_adj, _dvc_picked_inv_adj);
    cudaCheckErrors("_gpu_cycle_revert_kl failed");
  }

  if (!is_td) {
    blocks = (_num_ende + threads - 1) / threads;
    _reverse_level<<<blocks,threads>>>(_num_ende, _max_level, level);
    cudaCheckErrors("_reverse_level failed");
  }
}

__global__ void _gpu_insertion_kl(
  const bool is_td,
  const uint32_t num_ende, 
  uint32_t* dvc_picked_endes,
  const uint32_t* dvc_global_adj_1D_ptr,
  const uint32_t* dvc_global_adj_1D,
  const uint32_t* dvc_global_inv_adj_1D_ptr,
  const uint32_t* dvc_global_inv_adj_1D,
  uint32_t* dvc_tmp_picked_adj,
  uint32_t* dvc_tmp_picked_inv_adj,
  uint32_t* dvc_picked_adj,
  uint32_t* dvc_picked_inv_adj,
  int* dvc_level,
  bool* update)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num_ende) return;
  const uint32_t ende = t_idx; 
  if (dvc_picked_endes[ende]) return;

  bool has_par = false, has_chd = false;
  int max_par_lvl = -1, min_chd_lvl = INT_MAX;
  // Find max parent level 
  for (uint32_t k = dvc_global_inv_adj_1D_ptr[ende]; k < dvc_global_inv_adj_1D_ptr[ende+1]; k++) {
    uint32_t par_ende = dvc_global_inv_adj_1D[k];
    if (!dvc_picked_endes[par_ende]) continue;
    int par_lvl = dvc_level[par_ende];
    dvc_tmp_picked_inv_adj[k] = true; 
    if (max_par_lvl < par_lvl) {
      has_par = true; 
      max_par_lvl = par_lvl; 
    }
  }

  // Find min child level
  for (uint32_t k = dvc_global_adj_1D_ptr[ende]; k < dvc_global_adj_1D_ptr[ende+1]; k++) {
    uint32_t chd_ende = dvc_global_adj_1D[k];      
    if (!dvc_picked_endes[chd_ende]) continue;
    int chd_lvl = dvc_level[chd_ende];
    dvc_tmp_picked_adj[k] = true;
    if (min_chd_lvl > chd_lvl) {
      has_chd = true; 
      min_chd_lvl = chd_lvl;
    }
  }

  // Check whether form a cycle
  bool cond = (is_td) ? (has_par) : (has_chd);
  int given_lvl = (is_td) ? (max_par_lvl + 1) : (min_chd_lvl - 1);
  bool has_overlap_b = false;
  for (uint32_t k = dvc_global_inv_adj_1D_ptr[ende]; k < dvc_global_inv_adj_1D_ptr[ende+1]; k++) {
    if(!dvc_tmp_picked_inv_adj[k]) continue;
    const uint32_t par_ende = dvc_global_inv_adj_1D[k];      
    for (uint32_t kk = dvc_global_adj_1D_ptr[ende]; kk < dvc_global_adj_1D_ptr[ende+1]; kk++) {
      if(!dvc_tmp_picked_adj[kk]) continue;
      const uint32_t chd_ende = dvc_global_adj_1D[kk];
      if (par_ende == chd_ende) has_overlap_b = true;
      break;
    }      
  }
  bool valid = (cond) && (max_par_lvl < min_chd_lvl) && (!has_overlap_b) && (given_lvl >= 0);
  if (valid) {
    // Update 
    dvc_picked_endes[ende] = true;
    dvc_level[ende] = given_lvl;
    *update = true; 

    // update _adj and _inv_adj
    for (uint32_t k = dvc_global_inv_adj_1D_ptr[ende]; k < dvc_global_inv_adj_1D_ptr[ende+1]; k++) {
      if(!dvc_tmp_picked_inv_adj[k]) continue;
      const uint32_t par_ende = dvc_global_inv_adj_1D[k];
      dvc_picked_inv_adj[k] = true;
      for (uint32_t kk = dvc_global_adj_1D_ptr[par_ende]; kk < dvc_global_adj_1D_ptr[par_ende+1]; kk++) {
        const uint32_t tar_ende = dvc_global_adj_1D[kk];
        if (ende == tar_ende) {
          dvc_picked_adj[kk] = true;
          break;
        }
      }
    }

    for (uint32_t k = dvc_global_adj_1D_ptr[ende]; k < dvc_global_adj_1D_ptr[ende+1]; k++) {
      if(!dvc_tmp_picked_adj[k]) continue;
      const uint32_t chd_ende = dvc_global_adj_1D[k];
      dvc_picked_adj[k] = true;
      for (uint32_t kk = dvc_global_inv_adj_1D_ptr[chd_ende]; kk < dvc_global_inv_adj_1D_ptr[chd_ende+1]; kk++) {
        const uint32_t tar_ende = dvc_global_inv_adj_1D[kk];
        if (ende == tar_ende) {
          dvc_picked_inv_adj[kk] = true;
          break;
        }
      }
    }         
  } 
}

// bitmap version
#if FAST_BITMAP_CHECK
  __inline__ __device__ int warpReduceMax(int val) {
    return __reduce_max_sync(0xffffffff, val);
  }

  __inline__ __device__ int warpReduceMin(int val) {
    return __reduce_min_sync(0xffffffff, val);
  }

  __global__ void _gpu_insertion_kl_warp(
    const bool is_td,
    const uint32_t num_ende, 
    uint32_t* dvc_picked_endes,
    const uint32_t* dvc_global_adj_1D_ptr,
    const uint32_t* dvc_global_adj_1D,
    const uint32_t* dvc_global_inv_adj_1D_ptr,
    const uint32_t* dvc_global_inv_adj_1D,
    uint32_t* dvc_tmp_picked_adj,
    uint32_t* dvc_tmp_picked_inv_adj,
    uint32_t* dvc_picked_adj,
    uint32_t* dvc_picked_inv_adj,
    int* dvc_level,
    bool* update)
  {
    // one warp per ende
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = global_tid / 32;
    const uint32_t lane_id = threadIdx.x % 32;

    if (warp_id >= num_ende) return;
    const uint32_t ende = warp_id;
    if (dvc_picked_endes[ende]) return;

    bool has_par = false, has_chd = false;
    int max_par_lvl = -1, min_chd_lvl = INT_MAX;

    // find max parent level
    uint32_t p_start = dvc_global_inv_adj_1D_ptr[ende];
    uint32_t p_end   = dvc_global_inv_adj_1D_ptr[ende + 1];
    for (uint32_t k = p_start + lane_id; k < p_end; k += 32) {
      uint32_t par_ende = dvc_global_inv_adj_1D[k];
      if (!dvc_picked_endes[par_ende]) continue;
      int par_lvl = dvc_level[par_ende];
      dvc_tmp_picked_inv_adj[k] = true;
      has_par = true;
      max_par_lvl = max(max_par_lvl, par_lvl);
    }
    max_par_lvl = warpReduceMax(max_par_lvl);
    has_par = __any_sync(0xffffffff, has_par);

    // find min child level + build bitmap
    uint64_t child_bitmap = 0ull;
    uint32_t c_start = dvc_global_adj_1D_ptr[ende];
    uint32_t c_end   = dvc_global_adj_1D_ptr[ende + 1];
    for (uint32_t k = c_start + lane_id; k < c_end; k += 32) {
      uint32_t chd_ende = dvc_global_adj_1D[k];
      if (!dvc_picked_endes[chd_ende]) continue;
      int chd_lvl = dvc_level[chd_ende];
      dvc_tmp_picked_adj[k] = true;
      has_chd = true;
      min_chd_lvl = min(min_chd_lvl, chd_lvl);

      // map child e-node to bitmap bit (use hash to compress)
      uint32_t bit_idx = chd_ende & 63; // mod 64
      child_bitmap |= (1ull << bit_idx);
    }

    // combine all warp threads' bitmaps
    min_chd_lvl = warpReduceMin(min_chd_lvl);
    has_chd = __any_sync(0xffffffff, has_chd);
    child_bitmap = __reduce_or_sync(0xffffffff, child_bitmap);

    // cycle check with bitmap
    bool cond = is_td ? has_par : has_chd;
    int given_lvl = is_td ? (max_par_lvl + 1) : (min_chd_lvl - 1);

    bool local_overlap = false;
    for (uint32_t k = p_start + lane_id; k < p_end && !local_overlap; k += 32) {
      if (!dvc_tmp_picked_inv_adj[k]) continue;
      const uint32_t par_ende = dvc_global_inv_adj_1D[k];
      uint32_t bit_idx = par_ende & 63;
      if (child_bitmap & (1ull << bit_idx)) {
        local_overlap = true;
        break;
      }
    }

    bool has_overlap_b = __any_sync(0xffffffff, local_overlap);
    bool valid = cond && (max_par_lvl < min_chd_lvl)
              && !has_overlap_b && (given_lvl >= 0);

    bool warp_valid = __any_sync(0xffffffff, valid);

    // updates
    if (warp_valid) {
      if (lane_id == 0) {
        dvc_picked_endes[ende] = true;
        dvc_level[ende] = given_lvl;
        *update = true;
      }

      // update inv_adj
      for (uint32_t k = p_start + lane_id; k < p_end; k += 32) {
        if (!dvc_tmp_picked_inv_adj[k]) continue;
        const uint32_t par_ende = dvc_global_inv_adj_1D[k];
        dvc_picked_inv_adj[k] = true;
        uint32_t par_start = dvc_global_adj_1D_ptr[par_ende];
        uint32_t par_end   = dvc_global_adj_1D_ptr[par_ende + 1];
        for (uint32_t kk = par_start; kk < par_end; kk++) {
          const uint32_t tar_ende = dvc_global_adj_1D[kk];
          if (tar_ende == ende) {
            dvc_picked_adj[kk] = true;
            break;
          }
        }
      }

      // update adj
      for (uint32_t k = c_start + lane_id; k < c_end; k += 32) {
        if (!dvc_tmp_picked_adj[k]) continue;
        const uint32_t chd_ende = dvc_global_adj_1D[k];
        dvc_picked_adj[k] = true;
        uint32_t chd_start = dvc_global_inv_adj_1D_ptr[chd_ende];
        uint32_t chd_end   = dvc_global_inv_adj_1D_ptr[chd_ende + 1];
        for (uint32_t kk = chd_start; kk < chd_end; kk++) {
          const uint32_t tar_ende = dvc_global_inv_adj_1D[kk];
          if (tar_ende == ende) {
            dvc_picked_inv_adj[kk] = true;
            break;
          }
        }
      }
    }
  }

#else // slower original version
  __inline__ __device__ int warpReduceMax(int val) {
    return __reduce_max_sync(0xffffffff, val);
  }

  __inline__ __device__ int warpReduceMin(int val) {
    return __reduce_min_sync(0xffffffff, val);
  }

  __global__ void _gpu_insertion_kl_warp(
    const bool is_td,
    const uint32_t num_ende, 
    uint32_t* dvc_picked_endes,
    const uint32_t* dvc_global_adj_1D_ptr,
    const uint32_t* dvc_global_adj_1D,
    const uint32_t* dvc_global_inv_adj_1D_ptr,
    const uint32_t* dvc_global_inv_adj_1D,
    uint32_t* dvc_tmp_picked_adj,
    uint32_t* dvc_tmp_picked_inv_adj,
    uint32_t* dvc_picked_adj,
    uint32_t* dvc_picked_inv_adj,
    int* dvc_level,
    bool* update)
  {
    // one warp per ende
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = global_tid / 32;
    const uint32_t lane_id = threadIdx.x % 32;

    if (warp_id >= num_ende) return;
    const uint32_t ende = warp_id;
    if (dvc_picked_endes[ende]) return;

    bool has_par = false, has_chd = false;
    int max_par_lvl = -1, min_chd_lvl = INT_MAX;
    // Find max parent level 
    uint32_t p_start = dvc_global_inv_adj_1D_ptr[ende];
    uint32_t p_end   = dvc_global_inv_adj_1D_ptr[ende + 1];
    for (uint32_t k = p_start + lane_id; k < p_end; k += 32) {
      uint32_t par_ende = dvc_global_inv_adj_1D[k];
      if (!dvc_picked_endes[par_ende]) continue;
      int par_lvl = dvc_level[par_ende];
      dvc_tmp_picked_inv_adj[k] = true;
      has_par = true;
      max_par_lvl = max(max_par_lvl, par_lvl);
    }

    // reduction to get max parent level across warp
    max_par_lvl = warpReduceMax(max_par_lvl);

    // warp broadcast: has_par is true if any thread has true
    unsigned int par_mask = __ballot_sync(0xffffffff, has_par);
    has_par = par_mask != 0;

    // Find min child level
    uint32_t c_start = dvc_global_adj_1D_ptr[ende];
    uint32_t c_end   = dvc_global_adj_1D_ptr[ende + 1];
    for (uint32_t k = c_start + lane_id; k < c_end; k += 32) {
      uint32_t chd_ende = dvc_global_adj_1D[k];
      if (!dvc_picked_endes[chd_ende]) continue;
      int chd_lvl = dvc_level[chd_ende];
      dvc_tmp_picked_adj[k] = true;
      has_chd = true;
      min_chd_lvl = min(min_chd_lvl, chd_lvl);
    }

    // reduction to get min child level across warp
    min_chd_lvl = warpReduceMin(min_chd_lvl);
    unsigned int chd_mask = __ballot_sync(0xffffffff, has_chd);
    has_chd = chd_mask != 0;

    // --- compute cond and given_lvl (same as before, all lanes do the same)
    bool cond = (is_td) ? (has_par) : (has_chd);
    int given_lvl = (is_td) ? (max_par_lvl + 1) : (min_chd_lvl - 1);
    bool local_overlap = false;

    for (uint32_t k = p_start + lane_id; k < p_end && !local_overlap; k += 32) {
      if (!dvc_tmp_picked_inv_adj[k]) continue;
      const uint32_t par_ende = dvc_global_inv_adj_1D[k];
      for (uint32_t kk = c_start; kk < c_end; kk++) {
        if (!dvc_tmp_picked_adj[kk]) continue;
        const uint32_t chd_ende = dvc_global_adj_1D[kk];
        if (par_ende == chd_ende) {
          local_overlap = true;
          break;
        }
      }
    }

    unsigned overlap_mask = __ballot_sync(0xffffffff, local_overlap);
    bool has_overlap_b = overlap_mask != 0;
    bool valid = (cond) && (max_par_lvl < min_chd_lvl)
                && (!has_overlap_b) && (given_lvl >= 0);

    unsigned valid_mask = __ballot_sync(0xffffffff, valid);
    bool warp_valid = (valid_mask != 0);

    if (warp_valid) {
      if (lane_id == 0) {
        dvc_picked_endes[ende] = true;
        dvc_level[ende] = given_lvl;
        *update = true;
      }

      for (uint32_t k = p_start + lane_id; k < p_end; k += 32) {
        if (!dvc_tmp_picked_inv_adj[k]) continue;
        const uint32_t par_ende = dvc_global_inv_adj_1D[k];
        dvc_picked_inv_adj[k] = true;
        uint32_t par_start = dvc_global_adj_1D_ptr[par_ende];
        uint32_t par_end   = dvc_global_adj_1D_ptr[par_ende + 1];
        for (uint32_t kk = par_start; kk < par_end; kk++) {
          const uint32_t tar_ende = dvc_global_adj_1D[kk];
          if (tar_ende == ende) {
            dvc_picked_adj[kk] = true;
            break;
          }
        }
      }

      for (uint32_t k = c_start + lane_id; k < c_end; k += 32) {
        if (!dvc_tmp_picked_adj[k]) continue;
        const uint32_t chd_ende = dvc_global_adj_1D[k];
        dvc_picked_adj[k] = true;
        uint32_t chd_start = dvc_global_inv_adj_1D_ptr[chd_ende];
        uint32_t chd_end   = dvc_global_inv_adj_1D_ptr[chd_ende + 1];
        for (uint32_t kk = chd_start; kk < chd_end; kk++) {
          const uint32_t tar_ende = dvc_global_inv_adj_1D[kk];
          if (tar_ende == ende) {
            dvc_picked_inv_adj[kk] = true;
            break;
          }
        }
      }
    }
  }
#endif


// __inline__ __device__ int blockReduceMax(int val) {
//   static __shared__ int shared[NUM_THREAD_DAG_CONSTR / 32]; // one per warp
//   int lane = threadIdx.x % 32;
//   int wid  = threadIdx.x / 32;
//   // Warp reduce first
//   for (int offset = 16; offset > 0; offset /= 2)
//     val = max(val, __shfl_down_sync(0xffffffff, val, offset));
//   // Write warp results to shared memory
//   if (lane == 0) shared[wid] = val;
//   __syncthreads();
//   // Read from first warp only
//   val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : INT_MIN;
//   // Final reduction within first warp
//   if (wid == 0) {
//     for (int offset = 16; offset > 0; offset /= 2)
//       val = max(val, __shfl_down_sync(0xffffffff, val, offset));
//   }
//   return val;
// }

// __inline__ __device__ int blockReduceMin(int val) {
//   static __shared__ int shared[NUM_THREAD_DAG_CONSTR / 32];
//   int lane = threadIdx.x % 32;
//   int wid  = threadIdx.x / 32;
//   for (int offset = 16; offset > 0; offset /= 2)
//     val = min(val, __shfl_down_sync(0xffffffff, val, offset));
//   if (lane == 0) shared[wid] = val;
//   __syncthreads();
//   val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : INT_MAX;
//   if (wid == 0) {
//     for (int offset = 16; offset > 0; offset /= 2)
//       val = min(val, __shfl_down_sync(0xffffffff, val, offset));
//   }
//   return val;
// }

// __global__ void _gpu_insertion_kl_block(
//   const bool is_td,
//   const uint32_t num_ende,
//   uint32_t* dvc_picked_endes,
//   const uint32_t* dvc_global_adj_1D_ptr,
//   const uint32_t* dvc_global_adj_1D,
//   const uint32_t* dvc_global_inv_adj_1D_ptr,
//   const uint32_t* dvc_global_inv_adj_1D,
//   uint32_t* dvc_tmp_picked_adj,
//   uint32_t* dvc_tmp_picked_inv_adj,
//   uint32_t* dvc_picked_adj,
//   uint32_t* dvc_picked_inv_adj,
//   int* dvc_level,
//   bool* update)
// {
//   const uint32_t ende = blockIdx.x;
//   if (ende >= num_ende) return;
//   if (dvc_picked_endes[ende]) return;

//   int tid = threadIdx.x;
//   bool has_par = false, has_chd = false;
//   int max_par_lvl = -1, min_chd_lvl = INT_MAX;

//   // --------------------------
//   // find max parent level
//   // --------------------------
//   uint32_t p_start = dvc_global_inv_adj_1D_ptr[ende];
//   uint32_t p_end   = dvc_global_inv_adj_1D_ptr[ende + 1];
//   for (uint32_t k = p_start + tid; k < p_end; k += blockDim.x) {
//     uint32_t par_ende = dvc_global_inv_adj_1D[k];
//     if (!dvc_picked_endes[par_ende]) continue;
//     int par_lvl = dvc_level[par_ende];
//     dvc_tmp_picked_inv_adj[k] = true;
//     has_par = true;
//     max_par_lvl = max(max_par_lvl, par_lvl);
//   }
//   max_par_lvl = blockReduceMax(max_par_lvl);
//   unsigned int par_mask = __ballot_sync(0xffffffff, has_par);
//   has_par = par_mask != 0;

//   // --------------------------
//   // find min child level
//   // --------------------------
//   uint32_t c_start = dvc_global_adj_1D_ptr[ende];
//   uint32_t c_end   = dvc_global_adj_1D_ptr[ende + 1];
//   for (uint32_t k = c_start + tid; k < c_end; k += blockDim.x) {
//     uint32_t chd_ende = dvc_global_adj_1D[k];
//     if (!dvc_picked_endes[chd_ende]) continue;
//     int chd_lvl = dvc_level[chd_ende];
//     dvc_tmp_picked_adj[k] = true;
//     has_chd = true;
//     min_chd_lvl = min(min_chd_lvl, chd_lvl);
//   }
//   min_chd_lvl = blockReduceMin(min_chd_lvl);
//   unsigned int chd_mask = __ballot_sync(0xffffffff, has_chd);
//   has_chd = chd_mask != 0;

//   // --------------------------
//   // cycle check
//   // --------------------------
//   bool cond = (is_td) ? (has_par) : (has_chd);
//   int given_lvl = (is_td) ? (max_par_lvl + 1) : (min_chd_lvl - 1);
//   bool local_overlap = false;
//   for (uint32_t k = p_start + tid; k < p_end && !local_overlap; k += blockDim.x) {
//     if (!dvc_tmp_picked_inv_adj[k]) continue;
//     const uint32_t par_ende = dvc_global_inv_adj_1D[k];
//     for (uint32_t kk = c_start; kk < c_end; kk++) {
//       if (!dvc_tmp_picked_adj[kk]) continue;
//       const uint32_t chd_ende = dvc_global_adj_1D[kk];
//       if (par_ende == chd_ende) { local_overlap = true; break; }
//     }
//   }

//   unsigned overlap_mask = __ballot_sync(0xffffffff, local_overlap);
//   bool has_overlap_b = overlap_mask != 0;

//   bool valid = (cond) && (max_par_lvl < min_chd_lvl)
//               && (!has_overlap_b) && (given_lvl >= 0);

//   // unsigned valid_mask = __ballot_sync(0xffffffff, valid);
//   // bool block_valid = (valid_mask != 0);
//   bool block_valid = __any_sync(0xffffffff, valid);


//   // --------------------------
//   // updates
//   // --------------------------
//   if (block_valid) {
//     if (tid == 0) {
//       dvc_picked_endes[ende] = true;
//       dvc_level[ende] = given_lvl;
//       *update = true;
//     }

//     // update inv_adj
//     for (uint32_t k = p_start + tid; k < p_end; k += blockDim.x) {
//       if (!dvc_tmp_picked_inv_adj[k]) continue;
//       const uint32_t par_ende = dvc_global_inv_adj_1D[k];
//       dvc_picked_inv_adj[k] = true;
//       uint32_t par_start = dvc_global_adj_1D_ptr[par_ende];
//       uint32_t par_end   = dvc_global_adj_1D_ptr[par_ende + 1];
//       for (uint32_t kk = par_start; kk < par_end; kk++) {
//         const uint32_t tar_ende = dvc_global_adj_1D[kk];
//         if (tar_ende == ende) {
//           dvc_picked_adj[kk] = true;
//           break;
//         }
//       }
//     }

//     // update adj
//     for (uint32_t k = c_start + tid; k < c_end; k += blockDim.x) {
//       if (!dvc_tmp_picked_adj[k]) continue;
//       const uint32_t chd_ende = dvc_global_adj_1D[k];
//       dvc_picked_adj[k] = true;
//       uint32_t chd_start = dvc_global_inv_adj_1D_ptr[chd_ende];
//       uint32_t chd_end   = dvc_global_inv_adj_1D_ptr[chd_ende + 1];
//       for (uint32_t kk = chd_start; kk < chd_end; kk++) {
//         const uint32_t tar_ende = dvc_global_inv_adj_1D[kk];
//         if (tar_ende == ende) {
//           dvc_picked_inv_adj[kk] = true;
//           break;
//         }
//       }
//     }
//   }
// }

// ============================================================
// === Faster intra-block reductions (merged max+min) =========
// ============================================================

__inline__ __device__ void blockReduceMaxMin(int& max_val, int& min_val) {
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  // --- Warp-level reduce ---
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    min_val = min(min_val, __shfl_down_sync(0xffffffff, min_val, offset));
  }

  // --- Warp results to shared ---
  __shared__ int warp_max[32];
  __shared__ int warp_min[32];
  if (lane == 0) {
    warp_max[wid] = max_val;
    warp_min[wid] = min_val;
  }
  __syncthreads();

  // --- Final reduction by warp 0 only ---
  if (wid == 0) {
    max_val = (lane < (blockDim.x >> 5)) ? warp_max[lane] : INT_MIN;
    min_val = (lane < (blockDim.x >> 5)) ? warp_min[lane] : INT_MAX;
    for (int offset = 16; offset > 0; offset >>= 1) {
      max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
      min_val = min(min_val, __shfl_down_sync(0xffffffff, min_val, offset));
    }
  }
  // Broadcast final values to all threads in block
  max_val = __shfl_sync(0xffffffff, max_val, 0);
  min_val = __shfl_sync(0xffffffff, min_val, 0);
}

// ============================================================
// === Optimized _gpu_insertion_kl_block_fast =================
// ============================================================
__global__ void _gpu_insertion_kl_block(
  const bool is_td,
  const uint32_t num_ende,
  uint32_t* dvc_picked_endes,
  const uint32_t* dvc_global_adj_1D_ptr,
  const uint32_t* dvc_global_adj_1D,
  const uint32_t* dvc_global_inv_adj_1D_ptr,
  const uint32_t* dvc_global_inv_adj_1D,
  uint32_t* dvc_tmp_picked_adj,
  uint32_t* dvc_tmp_picked_inv_adj,
  uint32_t* dvc_picked_adj,
  uint32_t* dvc_picked_inv_adj,
  int* dvc_level,
  bool* dvc_new_inserted,
  bool* update)
{
  const uint32_t ende = blockIdx.x;
  if (ende >= num_ende) return;
  if (dvc_picked_endes[ende]) return;

  const int tid = threadIdx.x;
  bool has_par = false, has_chd = false;
  int max_par_lvl = -1, min_chd_lvl = INT_MAX;

  // ============================================================
  // === 1. Scan parent edges ===================================
  // ============================================================
  uint32_t p_start = dvc_global_inv_adj_1D_ptr[ende];
  uint32_t p_end   = dvc_global_inv_adj_1D_ptr[ende + 1];

  for (uint32_t k = p_start + tid; k < p_end; k += blockDim.x) {
    uint32_t par_ende = dvc_global_inv_adj_1D[k];
    if (!dvc_picked_endes[par_ende]) continue;
    has_par = true;
    int par_lvl = dvc_level[par_ende];
    dvc_tmp_picked_inv_adj[k] = true;
    max_par_lvl = max(max_par_lvl, par_lvl);
  }

  // ============================================================
  // === 2. Scan child edges ====================================
  // ============================================================
  uint32_t c_start = dvc_global_adj_1D_ptr[ende];
  uint32_t c_end   = dvc_global_adj_1D_ptr[ende + 1];

  for (uint32_t k = c_start + tid; k < c_end; k += blockDim.x) {
    uint32_t chd_ende = dvc_global_adj_1D[k];
    if (!dvc_picked_endes[chd_ende]) continue;
    has_chd = true;
    int chd_lvl = dvc_level[chd_ende];
    dvc_tmp_picked_adj[k] = true;
    min_chd_lvl = min(min_chd_lvl, chd_lvl);
  }

  // ============================================================
  // === 3. Combined reduction for max/min ======================
  // ============================================================
  blockReduceMaxMin(max_par_lvl, min_chd_lvl);

  has_par = __any_sync(0xffffffff, has_par);
  has_chd = __any_sync(0xffffffff, has_chd);

  // ============================================================
  // === 4. Cycle detection (optional shared bitmap) ============
  // ============================================================
  bool local_overlap = false;
#if FAST_BITMAP_CHECK
  // --- Shared bitmap trick (lower bits as hash) ---
  const uint32_t bitMap_size = 256;
  __shared__ uint8_t chd_flags[bitMap_size];
  for (int i = tid; i < bitMap_size; i += blockDim.x) 
    chd_flags[i] = 0;
  __syncthreads();

  for (uint32_t k = c_start + tid; k < c_end; k += blockDim.x)
    if (dvc_tmp_picked_adj[k])
      chd_flags[dvc_global_adj_1D[k] & 255] = 1;
  __syncthreads();

  for (uint32_t k = p_start + tid; k < p_end; k += blockDim.x)
    if (dvc_tmp_picked_inv_adj[k] && chd_flags[dvc_global_inv_adj_1D[k] & 255])
      local_overlap = true;
      
#else
  // --- Default O(deg^2) safe check ---
  for (uint32_t k = p_start + tid; k < p_end && !local_overlap; k += blockDim.x) {
    if (!dvc_tmp_picked_inv_adj[k]) continue;
    const uint32_t par_ende = dvc_global_inv_adj_1D[k];
    for (uint32_t kk = c_start; kk < c_end; kk++) {
      if (!dvc_tmp_picked_adj[kk]) continue;
      if (par_ende == dvc_global_adj_1D[kk]) { local_overlap = true; break; }
    }
  }
#endif

  bool has_overlap_b = __any_sync(0xffffffff, local_overlap);

  // ============================================================
  // === 5. Compute validity & early exit =======================
  // ============================================================
  bool cond = is_td ? has_par : has_chd;
  int given_lvl = is_td ? (max_par_lvl + 1) : (min_chd_lvl - 1);
  bool valid = cond && (max_par_lvl < min_chd_lvl)
              && !has_overlap_b && (given_lvl >= 0);

  bool block_valid = __any_sync(0xffffffff, valid);
  if (!block_valid) return;   // early break

  // ============================================================
  // === 6. Apply updates =======================================
  // ============================================================
  if (tid == 0) {
    dvc_picked_endes[ende] = true;
    dvc_level[ende] = given_lvl;
    *update = true;
    dvc_new_inserted[ende] = true;
  }

  // --- update inv_adj ---
  for (uint32_t k = p_start + tid; k < p_end; k += blockDim.x) {
    if (!dvc_tmp_picked_inv_adj[k]) continue;
    const uint32_t par_ende = dvc_global_inv_adj_1D[k];
    dvc_picked_inv_adj[k] = true;
    uint32_t ps = dvc_global_adj_1D_ptr[par_ende];
    uint32_t pe = dvc_global_adj_1D_ptr[par_ende + 1];
    for (uint32_t kk = ps; kk < pe; kk++) {
      if (dvc_global_adj_1D[kk] == ende) {
        dvc_picked_adj[kk] = true;
        break;
      }
    }
  }

  // --- update adj ---
  for (uint32_t k = c_start + tid; k < c_end; k += blockDim.x) {
    if (!dvc_tmp_picked_adj[k]) continue;
    const uint32_t chd_ende = dvc_global_adj_1D[k];
    dvc_picked_adj[k] = true;
    uint32_t cs = dvc_global_inv_adj_1D_ptr[chd_ende];
    uint32_t ce = dvc_global_inv_adj_1D_ptr[chd_ende + 1];
    for (uint32_t kk = cs; kk < ce; kk++) {
      if (dvc_global_inv_adj_1D[kk] == ende) {
        dvc_picked_inv_adj[kk] = true;
        break;
      }
    }
  }
}


bool G_EXT::_gpu_insertion(const bool is_td)
{
#ifdef PRINT_LOG
  if (is_td) {
    printf("\n_gpu_insertion: _top_down_insertion()\n");
  } else {
    printf("\n_gpu_insertion: _bottom_up_insertion()\n");
  }
#endif

  bool updated = false;
  
  cudaMemset(_dvc_tmp_picked_adj, 0, _global_adj_1D.size()*sizeof(uint32_t));
  cudaMemset(_dvc_tmp_picked_inv_adj, 0, _global_inv_adj_1D.size()*sizeof(uint32_t));
  cudaMemset(_dvc_new_inserted, 0, _num_ende * sizeof(bool));

  if (DAG_KERNEL == 0) {
    // 一個 thread 負責一個 enode
    int blocks = (_num_ende + threads - 1) / threads;
    _gpu_insertion_kl<<<blocks,threads>>>(
      is_td,
      _num_ende, 
      _dvc_picked_endes,
      _dvc_global_adj_1D_ptr,
      _dvc_global_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_tmp_picked_adj,
      _dvc_tmp_picked_inv_adj,
      _dvc_picked_adj,
      _dvc_picked_inv_adj,    
      _dvc_level,
      _dvc_update);
    cudaCheckErrors("_gpu_insertion_kl failed");
  } else if (DAG_KERNEL == 1) {
    // 一個 warp 負責一個 enode
    int blocks = ((_num_ende*32) + threads - 1) / threads;
    _gpu_insertion_kl_warp<<<blocks,threads>>>(
      is_td,
      _num_ende, 
      _dvc_picked_endes,
      _dvc_global_adj_1D_ptr,
      _dvc_global_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_tmp_picked_adj,
      _dvc_tmp_picked_inv_adj,
      _dvc_picked_adj,
      _dvc_picked_inv_adj,    
      _dvc_level,
      _dvc_update);
    cudaCheckErrors("_gpu_insertion_kl_warp failed");  
  } else { // DAG_KERNEL == 2
    // 一個 block 負責一個 enode
    _gpu_insertion_kl_block<<<_num_ende, NUM_THREAD_DAG_CONSTR>>>(
      is_td,
      _num_ende,
      _dvc_picked_endes,
      _dvc_global_adj_1D_ptr,
      _dvc_global_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_tmp_picked_adj,
      _dvc_tmp_picked_inv_adj,
      _dvc_picked_adj,
      _dvc_picked_inv_adj,
      _dvc_level,
      _dvc_new_inserted,
      _dvc_update);
    cudaCheckErrors("_gpu_insertion_kl_block failed");
  }

  cudaMemcpy(&updated, _dvc_update, 1*sizeof(bool), cudaMemcpyDeviceToHost);

  return updated;
}

__global__ void _gpu_intersection(const uint32_t num_ende,
  uint32_t* dvc_from_roots,
  uint32_t* dvc_to_leaves,
  uint32_t* dvc_in_subgraph)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num_ende) return;
  dvc_in_subgraph[t_idx] = dvc_from_roots[t_idx] && dvc_to_leaves[t_idx];
}

void G_EXT::_gpu_clear_redundant_endes() 
{
  // 1. reachable from _roots
  _gpu_mark_reachable(
    _roots,
    _dvc_roots,
    _dvc_global_adj_1D,
    _dvc_global_adj_1D_ptr,
    _dvc_picked_adj,
    _dvc_from_roots);

  // // 2. can reach _leaves
  // _gpu_mark_reachable(
  //   _leaves,
  //   _dvc_leaves,
  //   _dvc_global_inv_adj_1D,
  //   _dvc_global_inv_adj_1D_ptr,
  //   _dvc_picked_inv_adj,
  //   _dvc_from_leaves);

  // // 3. intersection
  // int blocks = (_num_ende + threads - 1) / threads;
  // _gpu_intersection<<<blocks,threads>>>(
  //   _num_ende, 
  //   _dvc_from_roots, 
  //   _dvc_from_leaves, 
  //   _dvc_in_subgraph);
  // cudaCheckErrors("_gpu_intersection failed");
}

__global__ void _init_q2(const uint32_t sz, 
  const uint32_t* starts,
  const uint32_t* dvc_picked_endes,
  uint32_t* visited,
  uint32_t* q,
  uint32_t* q_sz)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= sz) return;
  
  const uint32_t s = starts[t_idx];
  if (dvc_picked_endes[s] && !visited[s]) {
    q[t_idx] = s;
    visited[s] = true;
  }
  if (t_idx == 0) 
    *q_sz = sz;
}

__global__ void _gpu_mark_reachable_kl(const uint32_t num,
  const uint32_t* dvc_global_adj_1D_ptr,
  const uint32_t* dvc_global_adj_1D,
  const uint32_t* dvc_picked_adj,
  uint32_t* dvc_visited,
  uint32_t* q_sp, 
  uint32_t* q,
  uint32_t* q_sz)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num) return;

  uint32_t u = q[*q_sp + t_idx];
  for (uint32_t k = dvc_global_adj_1D_ptr[u]; k < dvc_global_adj_1D_ptr[u+1]; k++) {
    if (!dvc_picked_adj[k]) continue;
    const uint32_t v = dvc_global_adj_1D[k];
    uint32_t old = atomicCAS(&dvc_visited[v], 0u, 1u); // 0: false; 1: true
    if (old == 0u) {
      uint32_t pos = atomicAdd(q_sz, 1);
      q[pos] = v;
    }     
  }
}

void G_EXT::_gpu_mark_reachable(
  const std::vector<uint32_t>& starts_v,
  const uint32_t* starts,
  const uint32_t* global_adj_1D,
  const uint32_t* global_adj_1D_ptr,
  const uint32_t* picked_adj,
  uint32_t* visited)
{
  cudaMemset(_dvc_q, 0, _num_ende*sizeof(uint32_t));
  cudaMemset(_dvc_q_sz, 0, 1*sizeof(uint32_t));
  cudaMemset(_dvc_q_sp, 0, 1*sizeof(uint32_t));

  // init queue
  const uint32_t start_N = starts_v.size();
  int blocks = (start_N + threads - 1) / threads;
  _init_q2<<<blocks,threads>>>(start_N, 
    starts,
    _dvc_picked_endes,
    visited,
    _dvc_q, 
    _dvc_q_sz);
  cudaCheckErrors("_init_q2 failed");

  // Actually is BFS
  uint32_t q_sz = start_N;
  uint32_t q_sp = 0;
  uint32_t num = q_sz - q_sp;

  while (num > 0) {
    cudaMemcpy(_dvc_q_sp, &q_sp, sizeof(uint32_t), cudaMemcpyHostToDevice); // start point
    blocks = (num + threads - 1) / threads;

    _gpu_mark_reachable_kl<<<blocks, threads>>>(
      num,
      global_adj_1D_ptr,
      global_adj_1D,
      picked_adj,
      visited,
      _dvc_q_sp, 
      _dvc_q,
      _dvc_q_sz);
    cudaCheckErrors("_gpu_mark_reachable_kl failed");

    cudaMemcpy(&q_sz, _dvc_q_sz, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    q_sp += num;
    num = q_sz - q_sp;
  }
}

void G_EXT::_copy_pickedende_back(std::vector<bool>& picked_endes_tmp)
{
  std::vector<uint32_t> tmp; tmp.resize(_num_ende, 0);
  picked_endes_tmp.assign(_num_ende, false);
  cudaMemcpy(tmp.data(), _dvc_picked_endes, _num_ende*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  for (uint32_t i = 0; i < _num_ende; i++) {
    if (tmp[i]) {
      picked_endes_tmp[i] = true;
    }
  }
}

void G_EXT::_copy_best_ende_of_ecls_back(std::vector<uint32_t>& best_ende_of_ecls)
{
  cudaMemcpy(best_ende_of_ecls.data(), _dvc_best_ende_of_ecls, best_ende_of_ecls.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost);
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


#ifdef DUMP_NEW_JSON
  #include <random>
  #include <fstream>
  #include <iomanip>
  #include <nlohmann/json.hpp>

  void G_EXT::_dump_to_egg_json(
    const std::string& out_path,
    uint32_t num_ecls,
    uint32_t num_ende,
    const std::vector<std::string>& ende_name,
    const std::vector<std::string>& ende_op,
    const std::vector<float>& ende_cost,
    const std::vector<uint32_t>& ende_ecls,
    const std::vector<uint32_t>& ende_child,
    const std::vector<uint32_t>& ende_child_ptr,
    const std::vector<uint32_t>& root_eclass_id)
  {
    std::vector<std::vector<uint32_t>> ecls_its_ende(num_ecls);
    for (uint32_t i = 0; i < num_ende; ++i) {
      ecls_its_ende[ende_ecls[i]].push_back(i);
    }

    nlohmann::json j_out; nlohmann::json j_nodes; std::mt19937 rng(42); 

    for (uint32_t ende = 0; ende < num_ende; ++ende) {
      std::string node_name = ende_name[ende];

      nlohmann::json node_json;
      node_json["op"] = ende_op[ende];
      node_json["cost"] = ende_cost[ende];

      node_json["eclass"] = std::to_string(ende_ecls[ende]);

      std::vector<std::string> children;
      for (uint32_t k = ende_child_ptr[ende]; k < ende_child_ptr[ende + 1]; ++k) {
        uint32_t child_ecls = ende_child[k];

        if (ecls_its_ende[child_ecls].empty()) continue;

        std::uniform_int_distribution<size_t> dist(0, ecls_its_ende[child_ecls].size() - 1);
        uint32_t rep_ende = ecls_its_ende[child_ecls][dist(rng)];

        children.push_back(ende_name[rep_ende]);
      }

      node_json["children"] = children;
      j_nodes[node_name] = node_json;
    }

    std::vector<std::string> root_eclasses;
    for (uint32_t r : root_eclass_id) {
      root_eclasses.push_back(std::to_string(r));
    }

    j_out["nodes"] = j_nodes;
    j_out["root_eclasses"] = root_eclasses;

    std::ofstream ofs(out_path);
    ofs << std::setw(2) << j_out << std::endl;
    std::cout << "[INFO] Wrote egg-compatible JSON to: " << out_path << std::endl;
  }
#endif





// __device__ float atomicMinFloat(float* addr, float val) {
//   int* addr_as_i = (int*)addr;
//   int old = *addr_as_i, assumed;

//   do {
//     assumed = old;
//     float old_f = __int_as_float(assumed);
//     if (old_f <= val) break; 
//     old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
//   } while (assumed != old);

//   return __int_as_float(old);
// }


// __global__ void _get_ende_picked_by_ende(const uint32_t num_ende, 
//   const uint32_t* ende_child,
//   const uint32_t* ende_child_ptr,
//   const uint32_t* best_ende_of_ecls, 
//   uint32_t* ende_picked_by_ende) 
// {
//   uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (t_idx >= num_ende) return;
//   uint32_t start = ende_child_ptr[t_idx];
//   uint32_t end   = ende_child_ptr[t_idx+1];
//   for (uint32_t jj = start; jj < end; jj++) {
//     uint32_t chd_ecls = ende_child[jj];
//     ende_picked_by_ende[jj] = best_ende_of_ecls[chd_ecls]; // Note: 因為每個 eclass 都要選到一個 enode, 因此 size 會和 ende_child 一樣
//   }
// }

// __global__ void _print_dvc_ende_picked_by_ende(const uint32_t sz, 
//   const uint32_t* ende_picked_by_ende) 
// {
//   printf("ende_picked_by_ende: ");
//   for (uint32_t i = 0; i < sz; i++) {
//     printf("%u, ", ende_picked_by_ende[i]); 
//   }
//   printf("\n");
// }

// bool G_EXT::_insertion2(const bool is_td, 
//   std::vector<bool>& picked_endes,
//   std::vector<int>& level) 
// {
// #ifdef PRINT_LOG
//   if (is_td) {
//     printf("\nG_EXT::_top_down_insertion()\n");
//   } else {
//     printf("\nG_EXT::_bottom_up_insertion()\n");
//   }
// #endif
//   bool update = false;
//   _tmp_picked_adj.assign(_global_adj_1D.size(), false);
//   _tmp_picked_inv_adj.assign(_global_inv_adj_1D.size(), false);

//   // omp_set_num_threads(8); // 這裏 threads = 8 比較快
//   // uint32_t acc = 0;  // used for check 

//   // Try all candidates
//   // #pragma omp for
//   for (uint32_t ende = 0; ende < _num_ende; ende++) {
//     if (picked_endes[ende]) continue;
//     #ifdef PRINT_LOG
//       printf("\n* Processing ende = %u\n", ende);
//     #endif

//     // auto start = std::chrono::steady_clock::now();
//     // _top_down_insertion(): 
//       // If not picked, check whether it can be added to this graph
//       // Conditions:
//       //   1. At least 1 parent enode exists (located @ MAX(parent_level))
//       //   2. At least 0 chldrn enode exists (located @ min(chldrn_level))
//       //   3. parent_level <= chldrn_level; can equal since children can be push to lower level at the next levelization
//     // _bottom_up_insertion(): 
//       // If not picked, check whether it can be added to this graph
//       // Conditions:
//       //   1. At least 0 parent enode exists (located @ MAX(parent_level))
//       //   2. At least 1 chldrn enode exists (located @ min(chldrn_level))
//       //   3. parent_level <= chldrn_level; can equal since children can be push to lower level at the next levelization
//     bool has_par = false, has_chd = false;
//     int max_par_lvl = -1, min_chd_lvl = INT_MAX;
//     #ifdef PRINT_LOG
//       uint32_t max_par_ende = UINT32_MAX, min_chd_ende = UINT32_MAX; // Note: init values here are not important!
//     #endif
//     // Find max parent level 
//     #ifdef PRINT_LOG
//       printf("Find max parent level\n"); 
//     #endif
//     for (uint32_t k = _global_inv_adj_1D_ptr[ende]; k < _global_inv_adj_1D_ptr[ende+1]; k++) {
//       uint32_t par_ende = _global_inv_adj_1D[k];
//       if (!picked_endes[par_ende]) continue;
//       int par_lvl = level[par_ende];
//       _tmp_picked_inv_adj[k] = true;
//       if (max_par_lvl < par_lvl) {
//         // #ifdef PRINT_LOG
//         //   printf("  pre: par_ende = %u, par_lvl = %d, max_par_lvl = %d\n", par_ende, par_lvl, max_par_lvl);
//         // #endif
//         MY_CHECK(par_lvl >= 0, "par_lvl >= 0"); 
//         has_par = true; 
//         max_par_lvl = par_lvl; 
//         #ifdef PRINT_LOG
//           max_par_ende = par_ende;
//         #endif
//         // #ifdef PRINT_LOG
//         //   printf("  pos: par_ende = %u, par_lvl = %d, max_par_lvl = %d\n", par_ende, par_lvl, max_par_lvl);
//         // #endif
//       }
//     }


//     // Find min child level
//     #ifdef PRINT_LOG
//       printf("Find min child level\n"); 
//     #endif
//     for (uint32_t k = _global_adj_1D_ptr[ende]; k < _global_adj_1D_ptr[ende+1]; k++) {
//       uint32_t chd_ende = _global_adj_1D[k];      
//       if (!picked_endes[chd_ende]) continue;
//       int chd_lvl = level[chd_ende];
//       // if (chd_lvl < 0) {
//       //   printf("WTF: \n");
//       //   std::cout << "chd_ende: " << chd_ende << ", chd_lvl = " << chd_lvl << "\n";
//       //   exit(1);
//       // }
//       MY_CHECK(chd_lvl >= 0, "chd_lvl >= 0"); 
//       _tmp_picked_adj[k] = true;
//       if (min_chd_lvl > chd_lvl) {
//         // #ifdef PRINT_LOG
//         //   printf("  pre: chd_ende = %u, chd_lvl = %d, min_chd_lvl = %d\n", chd_ende, chd_lvl, min_chd_lvl);
//         // #endif
//         has_chd = true; 
//         min_chd_lvl = chd_lvl;
//         #ifdef PRINT_LOG
//           min_chd_ende = chd_ende; 
//         #endif
//         // #ifdef PRINT_LOG
//         //   printf("  pos: chd_ende = %u, chd_lvl = %d, min_chd_lvl = %d\n", chd_ende, chd_lvl, min_chd_lvl);
//         // #endif
//       }
//     }

//     // Check whether form a cycle
//     bool cond = (is_td) ? (has_par) : (has_chd);
//     int given_lvl = (is_td) ? (max_par_lvl + 1) : (min_chd_lvl - 1);
//     // given_lvl == 0 的時候就有可能把非 root 的 enode 加到 level 0，這個時候就不能再繼續往上加上去否則 level 就會壞掉
//     // 因此我們會停在 given level = 0 的狀況下
//     bool has_overlap_b = false;
//     for (uint32_t k = _global_inv_adj_1D_ptr[ende]; k < _global_inv_adj_1D_ptr[ende+1]; k++) {
//       if(!_tmp_picked_inv_adj[k]) continue;
//       const uint32_t par_ende = _global_inv_adj_1D[k];      
//       for (uint32_t kk = _global_adj_1D_ptr[ende]; kk < _global_adj_1D_ptr[ende+1]; kk++) {
//         if(!_tmp_picked_adj[kk]) continue;
//         const uint32_t chd_ende = _global_adj_1D[kk];
//         if (par_ende == chd_ende) has_overlap_b = true;
//         break;
//       }      
//     }
//     bool valid = (cond) && (max_par_lvl < min_chd_lvl) && (!has_overlap_b) && (given_lvl >= 0);
//     // Critical zone
//     if (valid) {
//       MY_CHECK(given_lvl > -1, "given_lvl > -1");
//       // acc++; // used for check 

//       // Update 
//       picked_endes[ende] = true;
//       level[ende] = given_lvl;
//       update = true; 

//       // update _adj and _inv_adj
//       for (uint32_t k = _global_inv_adj_1D_ptr[ende]; k < _global_inv_adj_1D_ptr[ende+1]; k++) {
//         if(!_tmp_picked_inv_adj[k]) continue;
//         const uint32_t par_ende = _global_inv_adj_1D[k];
//         // _update_picked_adj(par_ende, ende); 
//         _picked_inv_adj[k] = true;
//         for (uint32_t kk = _global_adj_1D_ptr[par_ende]; kk < _global_adj_1D_ptr[par_ende+1]; kk++) {
//           const uint32_t tar_ende = _global_adj_1D[kk];
//           if (ende == tar_ende) {
//             _picked_adj[kk] = true;
//             break;
//           }
//         }
//       }

//       for (uint32_t k = _global_adj_1D_ptr[ende]; k < _global_adj_1D_ptr[ende+1]; k++) {
//         if(!_tmp_picked_adj[k]) continue;
//         const uint32_t chd_ende = _global_adj_1D[k];
//         // _update_picked_adj(ende, chd_ende); 
//         _picked_adj[k] = true;
//         for (uint32_t kk = _global_inv_adj_1D_ptr[chd_ende]; kk < _global_inv_adj_1D_ptr[chd_ende+1]; kk++) {
//           const uint32_t tar_ende = _global_inv_adj_1D[kk];
//           if (ende == tar_ende) {
//             _picked_inv_adj[kk] = true;
//             break;
//           }
//         }
//       }         

//       #ifdef PRINT_LOG
//         printf("! valid ende = %u, max_par_lvl = %d (max_par_ende = %u), min_chd_lvl = %d (min_chd_ende = %u), given_lvl = %d\n", 
//                 ende, 
//                 max_par_lvl, max_par_ende,
//                 min_chd_lvl, min_chd_ende,
//                 given_lvl);
//       #endif
//     } 
//   }
//   // printf("acc %u\n", acc);
  
//   return update;
// }

// void G_EXT::_BFS2(
//   const bool is_td, std::vector<uint32_t>& deg, 
//   const std::vector<bool>& picked_endes,
//   const std::vector<uint32_t>& global_adj_1D,
//   const std::vector<uint32_t>& global_adj_1D_ptr,
//   const std::vector<bool>& picked_adj,
//   std::vector<int>& level)
// {
//   // lambda
//   auto get_rev_level = [&](const uint32_t LEVEL, uint32_t a) -> uint32_t {
//     return LEVEL - a - 1;
//   };

//   // init queue (roots || leaves)
//   std::queue<uint32_t> q;
//   for (uint32_t ende = 0; ende < _num_ende; ende++) {
//     if ((deg[ende] == 0) && picked_endes[ende]) {
//       q.push(ende);
//     }
//   } 

//   // BFS: top-down levelization 
//   int cur_level = 0;
//   while (!q.empty()) {
//     size_t layer = q.size();
//     for (size_t t = 0; t < layer; ++t) {
//       uint32_t u = q.front(); q.pop();
//       level[u] = cur_level;
//       for (uint32_t k = global_adj_1D_ptr[u]; k < global_adj_1D_ptr[u+1]; k++) {
//         if (!picked_adj[k]) continue;
//         const uint32_t v = global_adj_1D[k];
//         if (deg[v] > 0 && --deg[v] == 0) {
//           q.push(v);
//         }        
//       }
//     }
//     ++cur_level;
//   }

//   // update _max_level
//   _max_level = cur_level; 

//   if (!is_td) {
//     for (uint32_t i = 0; i < level.size(); i++) {
//       if (level[i] != -1) {
//         level[i] = get_rev_level(_max_level, level[i]);
//       }
//     }
//   }

// #ifdef PRINT_LOG
//   printf("_max_level = %d, cur_level = %d\n", _max_level, cur_level); 
// #endif
// }



// __global__ void _reverse_level(  
//   const uint32_t num, 
//   const int* cur_level,
//   int* level) 
// {
//   uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (t_idx >= num) return;
  
//   if (level[t_idx] != -1) {
//     level[t_idx] = *cur_level - level[t_idx] - 1;
//   }
// }


// __global__ void _gpu_BFS_kl(
//   const uint32_t num, 
//   const uint32_t* dvc_global_adj_1D_ptr,
//   const uint32_t* dvc_global_adj_1D,
//   const uint32_t* q_sp,
//   uint32_t* q,
//   uint32_t* q_sz,
//   int* level,
//   int* cur_level,
//   const uint32_t* dvc_picked_adj,
//   uint32_t* deg)
// {
//   uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (t_idx >= num) return;

//   uint32_t u = q[*q_sp + t_idx];
//   level[u] = *cur_level;
//   for (uint32_t k = dvc_global_adj_1D_ptr[u]; k < dvc_global_adj_1D_ptr[u+1]; k++) {
//     if (!dvc_picked_adj[k]) continue;
//     const uint32_t v = dvc_global_adj_1D[k];
//     uint32_t old_deg = atomicSub(&deg[v], 1);
//     if (old_deg == 1) {
//       uint32_t pos = atomicAdd(q_sz, 1);
//       q[pos] = v;
//     }        
//   }
// }


// void G_EXT::_gpu_BFS(const bool is_td, 
//   uint32_t* deg, 
//   const uint32_t* picked_endes,
//   const uint32_t* global_adj_1D,
//   const uint32_t* global_adj_1D_ptr,
//   const uint32_t* picked_adj,
//   int* level)
// {
//   // clear queue 
//   cudaMemset(_dvc_q, 0, _num_ende*sizeof(uint32_t));
//   cudaMemset(_dvc_q_sz, 0, 1*sizeof(uint32_t));
//   cudaMemset(_dvc_q_sp, 0, 1*sizeof(uint32_t));
//   cudaMemset(_dvc_max_level, 0, 1*sizeof(int));

//   // init queue
//   int blocks = (_num_ende + threads - 1) / threads;
//   _init_q1<<<blocks,threads>>>(_num_ende, 
//     deg,
//     picked_endes,
//     _dvc_q, _dvc_q_sz, _dvc_q_num);
//   cudaCheckErrors("_init_q1 failed");
  
//   // BFS: top-down levelization 
//   uint32_t q_sz = 0;
//   cudaMemcpy(&q_sz, _dvc_q_sz, 1*sizeof(uint32_t), cudaMemcpyDeviceToHost);
//   uint32_t q_sp = 0;
//   uint32_t num = q_sz - q_sp;  
//   int cur_level = 0;

//   while (num > 0) {
//     cudaMemcpy(_dvc_q_sp, &q_sp, sizeof(uint32_t), cudaMemcpyHostToDevice); // start point
//     blocks = (num + threads - 1) / threads;

//     _gpu_BFS_kl<<<blocks, threads>>>(
//       num,
//       global_adj_1D_ptr,
//       global_adj_1D,
//       _dvc_q_sp, 
//       _dvc_q,
//       _dvc_q_sz,
//       level,
//       _dvc_max_level,
//       picked_adj,
//       deg);
//     cudaCheckErrors("_gpu_BFS_kl failed");

//     cudaMemcpy(&q_sz, _dvc_q_sz, sizeof(uint32_t), cudaMemcpyDeviceToHost);
//     q_sp += num;
//     num = q_sz - q_sp;
//     cur_level++; 
//     cudaMemcpy(_dvc_max_level, &cur_level, sizeof(int), cudaMemcpyHostToDevice);
//   }

//   if (!is_td) {
//     blocks = (_num_ende + threads - 1) / threads;
//     _reverse_level<<<blocks,threads>>>(_num_ende, _dvc_max_level, level);
//     cudaCheckErrors("_reverse_level failed");
//   }

// #ifdef PRINT_LOG
//   cudaMemcpy(&_max_level, _dvc_max_level, sizeof(int), cudaMemcpyDeviceToHost);
//   cudaDeviceSynchronize();
//   std::vector<uint32_t> picked_endes_cpu; picked_endes_cpu.resize(_num_ende, 0);
//   std::vector<int> level_cpu; level_cpu.resize(_num_ende, -1);
//   cudaMemcpy(picked_endes_cpu.data(), picked_endes, _num_ende*sizeof(uint32_t), cudaMemcpyDeviceToHost);
//   cudaMemcpy(level_cpu.data(), level, _num_ende*sizeof(int), cudaMemcpyDeviceToHost);
//   cudaDeviceSynchronize();
//   std::vector<bool> picked_endes_cpu_bool; picked_endes_cpu_bool.resize(_num_ende, false);
//   for (uint32_t i = 0; i < _num_ende; i++) {
//     if (picked_endes_cpu[i] == 1) {
//       picked_endes_cpu_bool[i] = true;
//     }
//   }
//   printf("==================== GPU _print_levelization ====================\n");
//   _print_levelization(is_td, level_cpu, picked_endes_cpu_bool);
// #endif
// }

