#include "gpu_common.cuh"

nlohmann::json XEG::run_gpu(double time_limit, DagKernel kernel, uint32_t threads,
                           uint32_t bitmap_bits, int solver_workers, double deterministic_limit) {
  return _run_gpu_pipeline(time_limit, kernel, threads, bitmap_bits,
    CpsatInitialSolutionMode::GpuTree, 0, IlpBackend::CpSat, CuoptMode::Presolve,
    solver_workers, deterministic_limit);
}

nlohmann::json XEG::_run_gpu_pipeline(
  double time_limit, DagKernel dag_kernel, uint32_t num_thread_dag_constrution,
  uint32_t dag_bitmap_bits, CpsatInitialSolutionMode initial_solution_mode,
  uint32_t random_seed, IlpBackend ilp_backend, CuoptMode cuopt_mode,
  int solver_workers, double deterministic_limit)
{
  if (!std::isfinite(time_limit) || time_limit < 0 || solver_workers < 1)
    throw std::invalid_argument("Invalid GPU solver workers or time limit");
  if (!std::isfinite(deterministic_limit) || deterministic_limit < 0 ||
      (deterministic_limit > 0 && (solver_workers != 1 || time_limit == 0)))
    throw std::invalid_argument("Deterministic solving requires one solver worker and positive limits");
  switch (dag_kernel) {
    case DagKernel::Thread:
      if (num_thread_dag_constrution != 1u) {
        throw std::invalid_argument(
          "Thread DAG kernel requires exactly "
          "one thread per candidate");
      }

      if (dag_bitmap_bits != 0u) {
        throw std::invalid_argument(
          "Thread DAG kernel requires "
          "BITMAP_BITS=0");
      }

      break;

    case DagKernel::Warp:
      _validate_warp_threads_per_block(
        num_thread_dag_constrution);

      _validate_warp_bitmap_bits(
        dag_bitmap_bits);

      break;

    case DagKernel::Block:
      _validate_block_threads_per_candidate(
        num_thread_dag_constrution);

      _validate_block_bitmap_bits(
        dag_bitmap_bits);

      break;

    default:
      throw std::invalid_argument(
        "Invalid DagKernel value");
  }

  _dag_kernel =
    dag_kernel;

  _dag_threads_per_candidate =
    num_thread_dag_constrution;

  _dag_bitmap_bits =
    dag_bitmap_bits;

  _dag_bitmap_mask =
    dag_bitmap_bits == 0u
      ? 0u
      : dag_bitmap_bits - 1u;

  _dag_bitmap_words =
    dag_bitmap_bits / 32u;


  using Clock = std::chrono::steady_clock;
  auto elapsed = [](auto from, auto to) { return std::chrono::duration<double,std::milli>(to-from).count(); };
  auto hash = [](const auto& values) {
    uint64_t h = 14695981039346656037ull;
    for (auto v : values) { h ^= static_cast<uint64_t>(v); h *= 1099511628211ull; }
    return std::to_string(h);
  };
  auto release = gpu_scope_exit([&] { _gpu_free_buffers(); });
  const auto graph = _cpu_graph();
  const auto start = Clock::now();
  const auto costs = _gpu_tree_construction();
  const auto tree_end = Clock::now();
  std::vector<uint32_t> best;
  _copy_best_ende_of_ecls_back(best);
  std::vector<uint32_t> raw(_num_ende);
  CUDA_CHECK(cudaMemcpy(raw.data(), _dvc_picked_endes,
                        raw.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));
  std::vector<uint8_t> seed(raw.size());
  for (size_t u = 0; u < raw.size(); ++u) seed[u] = raw[u] != 0;
  const double seed_cost = xeg::validate_cpu_solution(graph, seed);
  if (std::abs(seed_cost-costs.second) > 1e-6 * std::max(1.0, seed_cost))
    throw std::runtime_error("GPU tree DAG cost does not match its selected nodes");
  const auto seed_checked = Clock::now();

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();
  const auto dag_start = Clock::now();
  _gpu_DAG_construction(GpuDagInsertionPolicy::BatchThenDirect);
  const auto dag_end = Clock::now();
  CUDA_CHECK(cudaMemcpy(raw.data(), _dvc_in_subgraph,
                        raw.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));
  _gpu_free_dag_buffers();
  std::vector<uint8_t> candidates(raw.size());
  for (size_t u = 0; u < raw.size(); ++u) candidates[u] = raw[u] != 0;
  const auto copied = Clock::now();
  // Root-to-global-leaf intersection alone does not ensure child closure.
  // This removes only choices that cannot belong to any legal solution.
  const auto removed = xeg::prune_cpu_candidates(graph, candidates, 1);
  const auto pruned = Clock::now();
  const bool acyclic = xeg::validate_cpu_candidates(graph, candidates, false);
  for (uint32_t u = 0; u < _num_ende; ++u)
    if (seed[u] && !candidates[u]) throw std::runtime_error("GPU DAG filtering removed a tree node");
  const auto checked = Clock::now();

  nlohmann::json report = {
    {"kernel",static_cast<uint32_t>(dag_kernel)}, {"dag_threads",num_thread_dag_constrution},
    {"bitmap_bits",dag_bitmap_bits}, {"insertion_policy","batch-then-direct"},
    {"solver_workers",solver_workers}, {"time_limit_s",time_limit}, {"deterministic_limit",deterministic_limit},
    {"tree_ms",_gpu_tree_ms}, {"dag_ms",_gpu_dag_ms},
    {"tree_wall_ms",elapsed(start,tree_end)}, {"dag_wall_ms",elapsed(dag_start,dag_end)},
    {"seed_validation_ms",elapsed(tree_end,seed_checked)}, {"candidate_copy_ms",elapsed(dag_end,copied)},
    {"candidate_prune_ms",elapsed(copied,pruned)}, {"candidate_validation_ms",elapsed(pruned,checked)},
    {"postprocess_ms",elapsed(dag_end,checked)},
    {"tree_cost",costs.first}, {"tree_dag_cost",costs.second}, {"seed_valid",true},
    {"invalid_candidates_removed",removed}, {"candidate_acyclic",acyclic}, {"validated",true},
    {"tree_hash",hash(seed)}, {"best_hash",hash(best)}, {"filtered_hash",hash(candidates)},
    {"filtered",std::count(candidates.begin(),candidates.end(),uint8_t{1})}
  };
  if (time_limit > 0) {
    // XEG-T keeps its randomized hint experiment; the legal seed is retained
    // separately for final fallback, independent of which hints are supplied.
    const std::vector<bool> picked(seed.begin(), seed.end());
    auto hint_best = best;
    auto hint_selected = picked;
    if (initial_solution_mode == CpsatInitialSolutionMode::Random || ilp_backend == IlpBackend::CuOpt) {
      _ecls_its_ende_newG_GPU.assign(_num_ecls, {});
      for (uint32_t c = 0; c < _num_ecls; ++c)
        for (uint32_t u : _ecls_its_ende[c])
          if (candidates[u]) _ecls_its_ende_newG_GPU[c].push_back(u);
    }
    if (initial_solution_mode == CpsatInitialSolutionMode::Random)
      _build_random_cpsat_initial_solution(random_seed, picked, hint_best, hint_selected);
    if (ilp_backend == IlpBackend::CpSat) {
      report.update(_solve_checked_candidates(time_limit, candidates, seed, hint_best,
                    hint_selected, solver_workers, deterministic_limit, acyclic));
    } else {
      // The optional cuOpt backend retains its own level-constrained model.
      _solve_ilp(ilp_backend, time_limit, _ecls_its_ende_newG_GPU, _ende_child,
        _ende_child_ptr, _root_eclass_id, _ende_cost, hint_best, hint_selected, cuopt_mode);
    }
  }
  _gpu_free_buffers();
  report["gpu_pipeline_ms"] = elapsed(start,Clock::now());
  return report;
}
