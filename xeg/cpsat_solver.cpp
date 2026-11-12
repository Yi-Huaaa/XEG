#include "xeg.cuh"
#include <condition_variable>
#include <mutex>
#include <iomanip>
#include <sys/resource.h>
#include <unistd.h>

namespace {
  // used to test cpsat peak memory start
uint64_t get_peak_rss_kb()
{
  struct rusage usage {};

  if (getrusage(
        RUSAGE_SELF,
        &usage) != 0) {
    return 0u;
  }

  return static_cast<uint64_t>(
    usage.ru_maxrss);
}
constexpr std::chrono::seconds
  kCpsatMemorySampleInterval{10};


void write_cpsat_memory_sample(
  std::ofstream& memory_log,
  const std::chrono::steady_clock::time_point& start_time)
{
  const auto now =
    std::chrono::steady_clock::now();

  const double elapsed_seconds =
    std::chrono::duration<double>(
      now - start_time).count();

  const uint64_t peak_rss_kb =
    get_peak_rss_kb();

  const double peak_rss_gib =
    static_cast<double>(
      peak_rss_kb) /
    (1024.0 * 1024.0);

  memory_log
    << std::fixed
    << std::setprecision(3)
    << "elapsed_s = "
    << elapsed_seconds
    << ", peak_rss_kb = "
    << peak_rss_kb
    << ", peak_rss_gib = "
    << peak_rss_gib
    << '\n';

  memory_log.flush();
}
// used to test cpsat peak memory end


}

using namespace operations_research;
using namespace sat;
void XEG::_cpsat_solver(
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
  printf("Inside XEG::_cpsat_solver() Time_limit = %.2lf (sec)\n", time_limit);

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

  // General e-graphs require explicit class levels. The CPU DAG path may omit
  // them only after candidate acyclicity, self-dependencies and closure checks.
  if (!_cpu_omit_level_constraints) {
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
  // Model model;
  // int num_solutions = 0;
  // auto start = std::chrono::high_resolution_clock::now();
  Model model;
  int num_solutions = 0;
  const auto start =
    std::chrono::steady_clock::now();  

  const pid_t process_id =
    getpid();

  const std::string memory_log_path =
    "cpsat_memory_" +
    std::to_string(process_id) +
    ".log";

  std::ofstream memory_log(
    memory_log_path,
    std::ios::out |
    std::ios::trunc);

  if (!memory_log.is_open()) {
    throw std::runtime_error(
      "Failed to open CP-SAT memory log: " +
      memory_log_path);
  }

  std::cout
    << "[CP-SAT Memory] Log file: "
    << memory_log_path
    << std::endl;

  memory_log
    << "CP-SAT Memory Monitor\n"
    << "PID = "
    << process_id
    << '\n'
    << "sample_interval_s = "
    << kCpsatMemorySampleInterval.count()
    << '\n'
    << '\n';

  memory_log.flush();    
  model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& r) {
      const auto current_time =
        std::chrono::steady_clock::now();
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
    if (_cpu_solver_workers > 0) parameters.set_num_search_workers(_cpu_solver_workers);
    if (_cpu_deterministic_limit > 0) parameters.set_max_deterministic_time(_cpu_deterministic_limit);
    // parameters.set_max_time_in_seconds(3600.0);
    // parameters.set_log_search_progress(true);  // for full log
    model.Add(NewSatParameters(parameters));
  }


  // // start solving ILP
  // const CpSolverResponse response = SolveCpModel(cp_model.Build(), &model);
  // ============================================================
  // CP-SAT memory monitor
  // ============================================================

  std::atomic<bool> stop_memory_monitor{
    false
  };

  std::mutex memory_monitor_mutex;

  std::condition_variable memory_monitor_cv;

  // Record one sample immediately before CP-SAT starts.
  write_cpsat_memory_sample(
    memory_log,
    start);

  std::thread memory_monitor(
    [&]() {
      std::unique_lock<std::mutex>
        lock(memory_monitor_mutex);

      while (true) {
        const bool should_stop =
          memory_monitor_cv.wait_for(
            lock,
            kCpsatMemorySampleInterval,
            [&]() {
              return stop_memory_monitor.load(
                std::memory_order_relaxed);
            });

        if (should_stop) {
          break;
        }

        // Do not hold the monitor mutex while reading/writing
        // the memory sample.
        lock.unlock();

        write_cpsat_memory_sample(
          memory_log,
          start);

        lock.lock();
      }
    });

  // start solving ILP
  const CpSolverResponse response =
    SolveCpModel(
      cp_model.Build(),
      &model);

  // Stop the memory monitor immediately after CP-SAT returns.
  stop_memory_monitor.store(
    true,
    std::memory_order_relaxed);

  memory_monitor_cv.notify_one();

  memory_monitor.join();

  // Record one final sample at solver termination.
  write_cpsat_memory_sample(
    memory_log,
    start);

  memory_log.close();  

  // // print results
  // if (response.status() == CpSolverStatus::OPTIMAL || response.status() == CpSolverStatus::FEASIBLE) {
  //   std::cout << "=========================================\n";
  //   std::cout << "Solver finished.\n";
  //   std::cout << "Objective value = " << response.objective_value() << "\n";
  //   std::cout << "Runtime = " << response.wall_time() << " (s)\n";
  //   std::cout << "Feasible solutions found = " << num_solutions << "\n";

  //   // // Optional: print all active e-nodes
  //   // std::cout << "==== Active enodes (solution) ====\n";
  //   // for (const auto& [cid, node_map] : nodes_vars) {
  //   //   for (const auto& [node_id, var] : node_map) {
  //   //     bool value = SolutionBooleanValue(response, var);
  //   //     if (value) {
  //   //       std::cout << "EClass " << cid
  //   //                 << " -> enode " << node_id
  //   //                 << " (cost " << ende_cost[node_id] << ")\n";
  //   //     }
  //   //   }
  //   // }
  //   std::cout << "=========================================\n";
  // } else {
  //   std::cout << "No feasible solution found.\n";
  // }
  // Print final CP-SAT result.
  const char* solver_status = "UNKNOWN";

  switch (response.status()) {
    case CpSolverStatus::OPTIMAL:
      solver_status = "OPTIMAL";
      break;

    case CpSolverStatus::FEASIBLE:
      solver_status = "FEASIBLE";
      break;

    case CpSolverStatus::INFEASIBLE:
      solver_status = "INFEASIBLE";
      break;

    case CpSolverStatus::MODEL_INVALID:
      solver_status = "MODEL_INVALID";
      break;

    case CpSolverStatus::UNKNOWN:
      solver_status = "UNKNOWN";
      break;
  }

  if (_cpu_validate_solver) {
    _cpu_solver_result = {{"status",solver_status}, {"wall_time_s",response.wall_time()},
      {"level_constraints",!_cpu_omit_level_constraints},
      {"model_variables",cp_model.Proto().variables_size()},
      {"model_constraints",cp_model.Proto().constraints_size()}};
    if (response.status() == CpSolverStatus::OPTIMAL || response.status() == CpSolverStatus::FEASIBLE) {
      std::vector<uint8_t> selected(_num_ende,0);
      for (const auto& entry : nodes_vars)
        for (const auto& node : entry.second)
          if (SolutionBooleanValue(response,node.second)) selected[node.first] = 1;
      const double objective = xeg::validate_cpu_solution(_cpu_graph(),selected,true);
      if (std::abs(objective-response.objective_value()) > 1e-6)
        throw std::runtime_error("CP-SAT objective does not match independently evaluated extraction");
      _cpu_solver_result["objective"] = objective;
      _cpu_solver_result["bound"] = response.best_objective_bound();
      _cpu_solver_result["validated"] = true;
      auto names = nlohmann::json::array();
      for (uint32_t u=0;u<_num_ende;++u) if (selected[u]) names.push_back(_ende_name[u]);
      _cpu_solver_result["selected_enodes"] = std::move(names);
    }
  }

  std::cout << "=========================================\n";
  std::cout << "Solver finished.\n";
  std::cout << "Solver status = " << solver_status << "\n";
  std::cout << "Runtime = " << response.wall_time() << " (s)\n";
  std::cout << "Feasible solutions found = "
            << num_solutions
            << "\n";

  if (
    response.status() == CpSolverStatus::OPTIMAL ||
    response.status() == CpSolverStatus::FEASIBLE)
  {
    const double objective =
      response.objective_value();

    const double best_bound =
      response.best_objective_bound();

    const double relative_gap =
      std::abs(objective - best_bound) /
      std::max(1.0, std::abs(objective));

    std::cout << "Objective value = "
              << objective
              << "\n";

    std::cout << "Best objective bound = "
              << best_bound
              << "\n";

    std::cout << "Relative gap = "
              << relative_gap
              << "\n";
  } else {
    std::cout << "No feasible solution found.\n";
  }

  std::cout << "=========================================\n";
}

// =========================== Print functions =========================== // 
