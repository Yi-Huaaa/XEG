#include "cpu_extraction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <omp.h>

namespace xeg {
namespace {
constexpr uint32_t absent = std::numeric_limits<uint32_t>::max();
constexpr double infinity = std::numeric_limits<double>::infinity();
void check_threads(int threads) {
  if (threads < 1) throw std::invalid_argument("CPU thread count must be positive");
}

// All concurrent accesses to a shared scalar use relaxed atomics. The
// OpenMP barriers delimit complete relaxation rounds; no representative is
// published alongside a cost update.
template<bool Parallel, class T> T load(const T& value) {
  if constexpr (!Parallel) return value;
  T result;
  __atomic_load(&value, &result, __ATOMIC_RELAXED);
  return result;
}
template<bool Parallel, class T> bool lower(T& value, T candidate) {
  if constexpr (!Parallel) {
    if (candidate >= value) return false;
    value = candidate; return true;
  } else {
    T old = load<true>(value);
    while (candidate < old) {
      if (__atomic_compare_exchange(&value, &old, &candidate, true,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return true;
    }
    return false;
  }
}

template<bool Parallel>
void relax_tree(const CpuGraph& g, int threads, CpuTree& result) {
  const size_t n = g.cost.size();
  const uint64_t missing = std::numeric_limits<uint64_t>::max();
  const uint64_t stride = n + 1;
  std::vector<uint64_t> rank(g.members.size(), missing);
  int changed = 1;
  bool again = true;
  bool need_rank = true;
  #pragma omp parallel num_threads(threads) if(Parallel) shared(changed, again)
  {
    while (true) {
      #pragma omp single
      { again = changed != 0; changed = 0; if (again) ++result.iterations; }
      if (!again) break;
      #pragma omp for schedule(static) reduction(|:changed)
      for (size_t u = 0; u < n; ++u) {
        const uint32_t c = g.node_class[u];
        const double bound = load<Parallel>(result.class_cost[c]);
        double cost = g.cost[u];
        // Nonnegative costs make this an exact lower bound, including leaves.
        if (cost >= bound) continue;
        for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1] && cost < bound; ++k)
          cost += load<Parallel>(result.class_cost[g.children[k]]);
        if (lower<Parallel>(result.class_cost[c], cost)) {
          changed = 1;
          if constexpr (!Parallel) result.best[c] = u;
        }
      }
    }
    #pragma omp for schedule(static)
    for (size_t u = 0; u < n; ++u) {
      double cost = g.cost[u];
      for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k)
        cost += result.class_cost[g.children[k]];
      result.node_cost[u] = cost;
    }
    #pragma omp single
    {
      // For exact integer sums the ordered serial relaxation already records
      // the same first-optimal witness recovered by logical ranks. Keep that
      // fast path; rounded/nonintegral costs use the common witness recovery.
      if constexpr (!Parallel) {
        need_rank = !std::all_of(g.cost.begin(),g.cost.end(),
            [](float c) { return std::floor(c) == c; }) ||
          !std::all_of(result.class_cost.begin(),result.class_cost.end(),
            [](double c) { return !std::isfinite(c) || c < 9007199254740992.0; });
      }
      changed = need_rank ? 1 : 0;
      if (need_rank) std::fill(result.best.begin(), result.best.end(), absent);
    }
    // Recover a finite witness only after costs have converged. Ranks encode
    // (logical ordered sweep, node ID). An optimal candidate can be discovered
    // in its latest child's sweep iff its ID comes after that child; otherwise
    // it is discovered in the next sweep. Taking the least rank reproduces
    // ordered relaxation's first optimal witness (for exact additive costs),
    // while making tie-breaking independent of thread scheduling. Each chosen
    // node has a strictly larger rank than all its children, even at zero cost.
    while (true) {
      #pragma omp single
      { again = changed != 0; changed = 0; }
      if (!again) break;
      #pragma omp for schedule(static) reduction(|:changed)
      for (size_t u = 0; u < n; ++u) {
        const uint32_t c = g.node_class[u];
        if (!std::isfinite(result.node_cost[u]) || result.node_cost[u] != result.class_cost[c]) continue;
        uint64_t latest = 0;
        for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k)
          latest = std::max(latest, load<Parallel>(rank[g.children[k]]));
        if (latest == missing) continue;
        uint64_t candidate = (latest / stride) * stride + u + 1;
        if (candidate <= latest) candidate += stride;
        changed |= lower<Parallel>(rank[c], candidate);
      }
    }
    #pragma omp for schedule(static)
    for (size_t c = 0; c < g.members.size(); ++c)
      if (rank[c] != missing) result.best[c] = static_cast<uint32_t>(rank[c] % stride - 1);
  }
}

// Kahn traversal on the induced e-node graph. Each degree belongs to one
// vertex; only decrements from a wide frontier require atomic operations.
// Narrow frontiers run serially to avoid one OpenMP launch per graph level.
struct Levelizer {
  const CpuGraph& g;
  int threads;
  std::vector<uint32_t> degree, queue;
  std::vector<int> level;
  explicit Levelizer(const CpuGraph& graph, int count)
      : g(graph), threads(count), degree(g.cost.size()),
        queue(g.cost.size()), level(g.cost.size()) {}

  size_t run(const std::vector<uint8_t>& selected, bool top_down) {
    const size_t n = selected.size();
    const auto& incoming = top_down ? g.inverse : g.forward;
    const auto& incoming_ptr = top_down ? g.inverse_ptr : g.forward_ptr;
    const auto& outgoing = top_down ? g.forward : g.inverse;
    const auto& outgoing_ptr = top_down ? g.forward_ptr : g.inverse_ptr;
    #pragma omp parallel for num_threads(threads) schedule(static) if(n >= 2048)
    for (size_t u = 0; u < n; ++u) {
      uint32_t d = 0;
      if (selected[u])
        for (uint32_t k = incoming_ptr[u]; k < incoming_ptr[u + 1]; ++k)
          d += selected[incoming[k]] != 0;
      degree[u] = d;
      level[u] = -1;
    }
    size_t end = 0;
    for (uint32_t u = 0; u < n; ++u)
      if (selected[u] && !degree[u]) queue[end++] = u;
    size_t begin = 0;
    int depth = 0;
    while (begin < end) {
      const size_t frontier_end = end;
      if (threads > 1 && frontier_end - begin >= 1024) {
        #pragma omp parallel num_threads(threads)
        {
          std::vector<uint32_t> local;
          #pragma omp for schedule(static)
          for (size_t i = begin; i < frontier_end; ++i) {
            const uint32_t u = queue[i];
            level[u] = depth;
            for (uint32_t k = outgoing_ptr[u]; k < outgoing_ptr[u + 1]; ++k) {
              const uint32_t v = outgoing[k];
              if (selected[v] && __atomic_fetch_sub(&degree[v], 1u, __ATOMIC_RELAXED) == 1u)
                local.push_back(v);
            }
          }
          size_t offset;
          #pragma omp atomic capture
          { offset = end; end += local.size(); }
          std::copy(local.begin(), local.end(), queue.begin() + offset);
        }
      } else {
        for (size_t i = begin; i < frontier_end; ++i) {
          const uint32_t u = queue[i];
          level[u] = depth;
          for (uint32_t k = outgoing_ptr[u]; k < outgoing_ptr[u + 1]; ++k) {
            const uint32_t v = outgoing[k];
            if (selected[v] && --degree[v] == 0) queue[end++] = v;
          }
        }
      }
      begin = frontier_end;
      ++depth;
    }
    if (!top_down) {
      #pragma omp parallel for num_threads(threads) schedule(static) if(n >= 2048)
      for (size_t u = 0; u < n; ++u)
        if (level[u] >= 0) level[u] = depth - level[u] - 1;
    }
    return end;
  }
};

struct Bounds {
  int parent = -1;
  int child = std::numeric_limits<int>::max();
  int proposed_level(bool top_down) const {
    // A shared selected parent/child is rejected by this same strict bound.
    if (parent >= child) return -1;
    if (top_down) return parent >= 0 ? parent + 1 : -1;
    return child != std::numeric_limits<int>::max() ? child - 1 : -1;
  }
};
Bounds bounds(const CpuGraph& g, uint32_t u,
              const std::vector<uint8_t>& selected, const std::vector<int>& level) {
  Bounds result;
  auto parents = [&]() {
    for (uint32_t k = g.inverse_ptr[u]; k < g.inverse_ptr[u + 1]; ++k) {
      const uint32_t v = g.inverse[k];
      if (selected[v]) result.parent = std::max(result.parent, level[v]);
      if (result.parent >= result.child) break;
    }
  };
  auto children = [&]() {
    for (uint32_t k = g.forward_ptr[u]; k < g.forward_ptr[u + 1]; ++k) {
      const uint32_t v = g.forward[k];
      if (selected[v]) result.child = std::min(result.child, level[v]);
      if (result.parent >= result.child) break;
    }
  };
  // Scan the shorter side first. Once the bounds conflict, later additions
  // can only tighten them, so even a cached rejected candidate needs no more
  // adjacency scanning during this insertion pass.
  if (g.inverse_ptr[u + 1] - g.inverse_ptr[u] < g.forward_ptr[u + 1] - g.forward_ptr[u]) {
    parents(); children();
  } else {
    children(); parents();
  }
  return result;
}

std::vector<uint8_t> reachable(const CpuGraph& g,
    const std::vector<uint8_t>& selected, bool from_roots) {
  std::vector<uint8_t> visited(selected.size(), 0);
  std::vector<uint32_t> queue;
  queue.reserve(selected.size());
  auto add = [&](uint32_t u) {
    if (selected[u] && !visited[u]) { visited[u] = 1; queue.push_back(u); }
  };
  if (from_roots) {
    for (uint32_t c : g.roots) for (uint32_t u : g.members[c]) add(u);
  } else {
    for (uint32_t u = 0; u < selected.size(); ++u)
      if (g.forward_ptr[u] == g.forward_ptr[u + 1]) add(u);
  }
  const auto& edges = from_roots ? g.forward : g.inverse;
  const auto& ptr = from_roots ? g.forward_ptr : g.inverse_ptr;
  for (size_t i = 0; i < queue.size(); ++i) {
    const uint32_t u = queue[i];
    for (uint32_t k = ptr[u]; k < ptr[u + 1]; ++k) add(edges[k]);
  }
  return visited;
}
} // namespace

CpuTree cpu_tree(const CpuGraph& g, int threads) {
  check_threads(threads);
  const size_t n = g.cost.size(), classes = g.members.size();
  for (float c : g.cost)
    if (!std::isfinite(c) || c < 0)
      throw std::invalid_argument("CPU tree requires finite nonnegative node costs");
  CpuTree result;
  result.class_cost.assign(classes, infinity);
  result.node_cost.assign(n, infinity);
  result.best.assign(classes, absent);
  result.selected.assign(n, 0);
  if (threads > 1 && n >= 2048) relax_tree<true>(g,threads,result);
  else relax_tree<false>(g,1,result);
  std::vector<uint32_t> queue;
  auto add_class = [&](uint32_t c) {
    if (result.best[c] == absent) throw std::runtime_error("Root dependency has no finite extraction");
    const uint32_t u = result.best[c];
    if (!result.selected[u]) { result.selected[u] = 1; queue.push_back(u); }
  };
  for (uint32_t c : g.roots) { result.tree_cost += result.class_cost[c]; add_class(c); }
  for (size_t i = 0; i < queue.size(); ++i) {
    const uint32_t u = queue[i];
    result.dag_cost += g.cost[u];
    for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k) add_class(g.children[k]);
  }
  return result;
}

CpuDag cpu_dag(const CpuGraph& g, const std::vector<uint8_t>& tree_selected, int threads) {
  check_threads(threads);
  if (tree_selected.size() != g.cost.size()) throw std::invalid_argument("CPU DAG mask size mismatch");
  CpuDag result;
  result.selected = tree_selected;
  const size_t n = g.cost.size();
  Levelizer levels(g, threads);
  const size_t initial = std::count(tree_selected.begin(), tree_selected.end(), uint8_t{1});
  if (levels.run(result.selected, false) != initial)
    throw std::invalid_argument("CPU DAG seed contains a cycle");
  const bool parallel = threads > 1 && n >= 2048;
  std::vector<Bounds> cached(parallel ? n : 0);
  std::vector<uint8_t> tentative(n);
  auto insert = [&](bool top_down) {
    std::fill(tentative.begin(), tentative.end(), 0);
    if (parallel) {
      // Evaluate neighbor bounds against a read-only snapshot in parallel.
      #pragma omp parallel for num_threads(threads) schedule(static)
      for (size_t u = 0; u < n; ++u)
        if (!result.selected[u]) cached[u] = bounds(g, u, result.selected, levels.level);
    }
    size_t added = 0;
    for (uint32_t u = 0; u < n; ++u) {
      if (result.selected[u]) continue;
      ++result.proposals;
      const int proposed = (parallel ? cached[u] : bounds(g,u,result.selected,levels.level)).proposed_level(top_down);
      if (proposed < 0) continue;
      result.selected[u] = tentative[u] = 1;
      levels.level[u] = proposed;
      ++added;
      if (!parallel) continue;
      // Ordered commit preserves the sequential algorithm. A new node changes
      // only its neighbors' bounds; updating those bounds avoids rescanning
      // whole adjacency lists for every invalidated speculative candidate.
      for (uint32_t k = g.forward_ptr[u]; k < g.forward_ptr[u + 1]; ++k) {
        const uint32_t v = g.forward[k];
        if (v > u && !result.selected[v]) {
          cached[v].parent = std::max(cached[v].parent, proposed);
          ++result.neighbor_updates;
        }
      }
      for (uint32_t k = g.inverse_ptr[u]; k < g.inverse_ptr[u + 1]; ++k) {
        const uint32_t v = g.inverse[k];
        if (v > u && !result.selected[v]) {
          cached[v].child = std::min(cached[v].child, proposed);
          ++result.neighbor_updates;
        }
      }
    }
    // Validate before any tentative nodes become stable, including the final
    // insertion pass. Rollback must be followed by fresh levels: residuals
    // also contain stable vertices downstream of a rejected cycle.
    levels.run(result.selected, top_down);
    size_t rejected = 0;
    for (size_t u = 0; u < n; ++u)
      if (tentative[u] && levels.level[u] < 0) {
        result.selected[u] = 0; ++rejected;
      }
    result.reverted += rejected;
    if (rejected) levels.run(result.selected, top_down);
    return added - rejected;
  };
  size_t added;
  do {
    ++result.iterations;
    added = insert(true);
    // insert(true) has already produced top-down levels.
    added += insert(false);
  } while (added);
  auto forward = reachable(g, result.selected, true);
  auto inverse = reachable(g, result.selected, false);
  result.in_subgraph.resize(n);
  #pragma omp parallel for num_threads(threads) schedule(static) if(n >= 2048)
  for (size_t u = 0; u < n; ++u)
    result.in_subgraph[u] = forward[u] && inverse[u];
  result.invalid_candidates_removed = prune_cpu_candidates(g, result.in_subgraph, threads);
  if (result.invalid_candidates_removed)
    result.in_subgraph = reachable(g, result.in_subgraph, true);
  return result;
}

size_t prune_cpu_candidates(const CpuGraph& g, std::vector<uint8_t>& candidates, int threads) {
  check_threads(threads);
  if (candidates.size() != g.cost.size()) throw std::invalid_argument("Candidate mask size mismatch");
  const size_t n = candidates.size(), classes = g.members.size();
  std::vector<uint32_t> choices(classes, 0);
  #pragma omp parallel for num_threads(threads) schedule(static) if(classes >= 2048)
  for (size_t c = 0; c < classes; ++c)
    for (uint32_t u : g.members[c]) choices[c] += candidates[u] != 0;
  std::vector<uint8_t> queued(n, 0);
  #pragma omp parallel for num_threads(threads) schedule(static) if(n >= 2048)
  for (size_t u = 0; u < n; ++u) if (candidates[u])
    for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k)
      if (!choices[g.children[k]] || g.children[k] == g.node_class[u]) {
        queued[u] = 1; break;
      }
  std::vector<uint32_t> queue;
  for (uint32_t u = 0; u < n; ++u) if (queued[u]) queue.push_back(u);
  if (queue.empty()) return 0;

  // Build class-to-parent incidence only when repair is needed. This uses
  // original child references, not the much larger expanded e-node graph.
  std::vector<size_t> ptr(classes + 1, 0);
  for (uint32_t u = 0; u < n; ++u) if (candidates[u])
    for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k) ++ptr[g.children[k] + 1];
  std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
  auto cursor = ptr;
  std::vector<uint32_t> parents(ptr.back());
  for (uint32_t u = 0; u < n; ++u) if (candidates[u])
    for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k)
      parents[cursor[g.children[k]]++] = u;
  for (size_t i = 0; i < queue.size(); ++i) {
    const uint32_t u = queue[i], c = g.node_class[u];
    candidates[u] = 0;
    if (--choices[c]) continue;
    for (size_t k = ptr[c]; k < ptr[c + 1]; ++k) {
      const uint32_t parent = parents[k];
      if (!queued[parent]) { queued[parent] = 1; queue.push_back(parent); }
    }
  }
  return queue.size();
}

bool validate_cpu_candidates(const CpuGraph& g, const std::vector<uint8_t>& candidates, bool require_acyclic) {
  if (candidates.size() != g.cost.size()) throw std::invalid_argument("Candidate mask size mismatch");
  std::vector<uint32_t> choices(g.members.size(), 0);
  for (uint32_t u = 0; u < candidates.size(); ++u) if (candidates[u]) ++choices[g.node_class[u]];
  std::vector<uint32_t> degree(g.members.size(), 0), queue;
  for (uint32_t c : g.roots)
    if (!choices[c]) throw std::runtime_error("Candidate graph has no root choice");
  for (uint32_t u = 0; u < candidates.size(); ++u) if (candidates[u])
    for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k) {
      if (!choices[g.children[k]]) throw std::runtime_error("Candidate has an unavailable child class");
      if (g.children[k] == g.node_class[u]) throw std::runtime_error("Candidate depends on its own class");
      ++degree[g.children[k]];
    }
  // Every candidate points to every available choice of each child class.
  // After excluding self-class dependencies, a class cycle exists iff the
  // expanded candidate-node graph has a cycle. Traverse compact child refs
  // instead of scanning the much larger expanded CSR twice.
  size_t active = 0;
  for (uint32_t c = 0; c < choices.size(); ++c) if (choices[c]) {
    ++active;
    if (!degree[c]) queue.push_back(c);
  }
  for (size_t i = 0; i < queue.size(); ++i)
    for (uint32_t u : g.members[queue[i]]) if (candidates[u])
      for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k)
        if (--degree[g.children[k]] == 0) queue.push_back(g.children[k]);
  const bool acyclic = queue.size() == active;
  if (require_acyclic && !acyclic) throw std::runtime_error("Selected e-node graph contains a cycle");
  return acyclic;
}

void validate_cpu_dag(const CpuGraph& g, const std::vector<uint8_t>& selected) {
  if (selected.size() != g.cost.size()) throw std::invalid_argument("Solution mask size mismatch");
  Levelizer levels(g, 1);
  const size_t count = std::count_if(selected.begin(), selected.end(), [](uint8_t v) { return v != 0; });
  if (levels.run(selected, true) != count) throw std::runtime_error("Selected e-node graph contains a cycle");
}

double validate_cpu_solution(const CpuGraph& g, const std::vector<uint8_t>& selected, bool integer_cost) {
  if (selected.size() != g.cost.size()) throw std::invalid_argument("Solution mask size mismatch");
  std::vector<uint32_t> best(g.members.size(), absent), degree(g.members.size(), 0);
  double cost = 0;
  for (uint32_t u = 0; u < selected.size(); ++u) if (selected[u]) {
    uint32_t c = g.node_class[u];
    if (best[c] != absent) throw std::runtime_error("Multiple selected nodes in one e-class");
    best[c] = u;
    cost += integer_cost ? static_cast<int>(g.cost[u]) : g.cost[u];
  }
  for (uint32_t c : g.roots)
    if (best[c] == absent) throw std::runtime_error("Missing root e-class");
  for (uint32_t u = 0; u < selected.size(); ++u) if (selected[u])
    for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k) {
      const uint32_t c = g.children[k];
      if (best[c] == absent) throw std::runtime_error("Missing child e-class");
      ++degree[c];
    }
  std::vector<uint32_t> queue;
  size_t active = 0;
  for (uint32_t c = 0; c < best.size(); ++c) if (best[c] != absent) {
    ++active;
    if (!degree[c]) queue.push_back(c);
  }
  for (size_t i = 0; i < queue.size(); ++i) {
    const uint32_t u = best[queue[i]];
    for (uint32_t k = g.child_ptr[u]; k < g.child_ptr[u + 1]; ++k)
      if (--degree[g.children[k]] == 0) queue.push_back(g.children[k]);
  }
  if (queue.size() != active) throw std::runtime_error("Extracted e-class graph contains a cycle");
  return cost;
}
} // namespace xeg
