
void G_EXT::_reconstruct_graph(std::vector<double> cost_of_ecls,
                                std::vector<double> cost_of_ende,
                                std::vector<uint32_t> best_ende_of_ecls,
                                std::vector<bool> picked_endes) 
{
  printf("Inside G_EXT::_reconstruct_graph()\n"); 
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

  // Construct inv adj
  std::vector<std::unordered_set<uint32_t>> adj;
  std::vector<std::unordered_set<uint32_t>> inv_adj;
  std::vector<std::vector<uint32_t>> tmp_inv_adj;
  adj.resize(_num_ende, std::unordered_set<uint32_t>()); // ende -> ende
  inv_adj.resize(_num_ende, std::unordered_set<uint32_t>()); // ende -> ende
  tmp_inv_adj.resize(_num_ende, std::vector<uint32_t>());   
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    uint32_t start = _ende_child_ptr[ende]; 
    uint32_t end = _ende_child_ptr[ende+1];
    for (uint32_t jj = start; jj < end; jj++) {
      uint32_t chd_ecls = _ende_child[jj];
      for (uint32_t ende2 = 0; ende2 < _ecls_its_ende[chd_ecls].size(); ende2++) {
        uint32_t ende2ID = _ecls_its_ende[chd_ecls][ende2];
        // update adj 
        adj[ende].insert(ende2ID); 
        // update inv_adjs
        inv_adj[ende2ID].insert(ende);
        if (picked_endes[ende] && picked_endes[ende2ID]) {
          tmp_inv_adj[ende2ID].push_back(ende);
        }
      }
    }
  }
  printf("adj mapping (ende Id -> ende Id):\n");
  for (uint32_t ende = 0; ende < adj.size(); ende++) {
    printf("  adj[%u]:", ende);
    for (auto ende : adj[ende]) {
      printf(" %u", ende);
    }
    printf("\n");
  }
  printf("inv_adj mapping (ende Id -> ende Id):\n");
  for (uint32_t ende = 0; ende < inv_adj.size(); ende++) {
    printf("  inv_adj[%u]:", ende);
    for (auto ende : inv_adj[ende]) {
      printf(" %u", ende);
    }
    printf("\n");
  }

  // levelize the new graph with the picked enode 
  std::vector<uint32_t> indegree(_num_ende, 0);
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if (!picked_endes[ende]) continue;
    uint32_t start = _ende_child_ptr[ende];
    uint32_t end = _ende_child_ptr[ende + 1];
    for (uint32_t jj = start; jj < end; jj++) {
      uint32_t chd_ecls = _ende_child[jj];
      for (uint32_t i = 0; i < _ecls_its_ende[chd_ecls].size(); i++) {
        uint32_t chd_ende = _ecls_its_ende[chd_ecls][i]; 
        if (picked_endes[chd_ende]) {
          indegree[chd_ende]++; 
        }
      }
    }
  }
  for (uint32_t i = 0; i < indegree.size(); i++) {
    if (picked_endes[i]) {
      printf("indegree of enode[%u] = %u (picked: T)\n", i, indegree[i]);
    } else {
      printf("indegree of enode[%u] = %u (picked: F)\n", i, indegree[i]);
    }
  }

  // init queue
  std::queue<uint32_t> q;
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if ((indegree[ende] == 0) && picked_endes[ende]) {
      q.push(ende);
      // printf("Queue init push in %u\n", ende);
    }
  } printf("\n"); 

  
  std::vector<int> level(_num_ende, -1);
  std::vector<std::vector<uint32_t>> ende_in_level;
  ende_in_level.push_back(std::vector<uint32_t>()); 
  // BFS（由 roots：indegree==0 的 picked enodes 出發）
  int cur_level = 0;
  while (!q.empty()) {
    size_t layer = q.size();
    for (size_t t = 0; t < layer; ++t) {
      uint32_t u = q.front(); q.pop();
      level[u] = cur_level;
      ende_in_level[cur_level].push_back(u); 
      // printf("Current enode %u, level[%u] = %u\n", u, u, level[u]);

      uint32_t start = _ende_child_ptr[u];
      uint32_t end   = _ende_child_ptr[u + 1];
      for (uint32_t jj = start; jj < end; ++jj) {
        uint32_t chd_ecls = _ende_child[jj];
        const auto& vec = _ecls_its_ende[chd_ecls];
        for (uint32_t k = 0; k < vec.size(); ++k) {
          uint32_t v = vec[k];                // child enode
          if (!picked_endes[v]) continue;     // 只考慮 picked
          if (indegree[v] > 0 && --indegree[v] == 0) {
            q.push(v);
            // printf("Push %u (indegree->0)\n", v);
          }
        }
      }
    }
    ++cur_level;  // 層切換
    if (!q.empty())
      ende_in_level.push_back(std::vector<uint32_t>()); 
  }
  std::vector<uint32_t> tmp_indegree(_num_ende, 0);
  for (uint32_t chd_ende = 0; chd_ende < tmp_inv_adj.size(); chd_ende++) {
    for (uint32_t p_ende = 0; p_ende < tmp_inv_adj[chd_ende].size(); p_ende++) {
      uint32_t to_ende = tmp_inv_adj[chd_ende][p_ende]; 
      tmp_indegree[to_ende]++;
    }
  }  
  printf("=== Levelization result ===\n");
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if (level[ende] >= 0)
      printf("ENode %u -> Level %d\n", ende, level[ende]);
  }
  for (uint32_t lvl = 0; lvl < ende_in_level.size(); lvl++) {
    printf("level %u: ", lvl); 
    for (uint32_t ende = 0; ende < ende_in_level[lvl].size(); ende++) {
      printf("%u, ", ende_in_level[lvl][ende]); 
    } printf("\n"); 
  }
  printf("tmp_inv_adj mapping (ende Id -> ende Id):\n");
  for (uint32_t ende = 0; ende < tmp_inv_adj.size(); ende++) {
    printf("  tmp_inv_adj[%u]:", ende);
    for (auto ende : tmp_inv_adj[ende]) {
      printf(" %u", ende);
    }
    printf("\n");
  }
  for (uint32_t i = 0; i < tmp_indegree.size(); i++) {
    printf("enode %u, tmp_indegree[%u] = %u\n", i, i, tmp_indegree[i]); 
  }

  // reorder ende_in_level, make it as lower as possible
  // std::vector<int> level(_num_ende, -1);
  uint32_t LEVEL = ende_in_level.size(); 
  level.clear(); 
  std::vector<std::vector<uint32_t>> low_ende_in_level; 
  low_ende_in_level.resize(LEVEL, std::vector<uint32_t>()); 
  auto get_rev_level = [&](uint32_t LEVEL, uint32_t a) -> uint32_t {
    return LEVEL - a;
  };
  // init queue
  std::queue<uint32_t>().swap(q); // Clears the queue
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if ((tmp_indegree[ende] == 0) && picked_endes[ende]) {
      q.push(ende);
      printf("Queue init push in %u\n", ende);
    }
  } printf("\n"); 

  cur_level = 0;
  while (!q.empty()) {
    size_t layer = q.size();
    for (size_t t = 0; t < layer; ++t) {
      uint32_t u = q.front(); q.pop();
      int rev_lvl = get_rev_level(LEVEL-1, cur_level);
      level[u] = rev_lvl;
      low_ende_in_level[rev_lvl].push_back(u); 
      for (uint32_t v : tmp_inv_adj[u]) {
        if (tmp_indegree[v] > 0 && --tmp_indegree[v] == 0) {
          q.push(v);
        }
      }
    }
    ++cur_level;  // 層切換
  }
  printf("=== Lower Levelization result ===\n");
  for (uint32_t ende = 0; ende < _num_ende; ende++) {
    if (level[ende] >= 0)
      printf("ENode %u -> Level %d\n", ende, level[ende]);
  }
  for (uint32_t lvl = 0; lvl < low_ende_in_level.size(); lvl++) {
    printf("level %u: ", lvl); 
    for (uint32_t ende = 0; ende < low_ende_in_level[lvl].size(); ende++) {
      printf("%u, ", low_ende_in_level[lvl][ende]); 
    } printf("\n"); 
  }

  // bottom-up levelization
  // top-down insertion
  // top-down levelization
  // bottom-up insertion

}
