#ifndef XEG_CPU_EXTRACTION_HPP
#define XEG_CPU_EXTRACTION_HPP

#include <cstdint>
#include <cstddef>
#include <vector>

namespace xeg {
// Non-owning views of the post-read graph. Child entries are e-class IDs;
// forward/inverse entries are sorted, unique e-node IDs.
struct CpuGraph {
  const std::vector<float>& cost;
  const std::vector<uint32_t>& node_class;
  const std::vector<uint32_t>& children;
  const std::vector<uint32_t>& child_ptr;
  const std::vector<std::vector<uint32_t>>& members;
  const std::vector<uint32_t>& roots;
  const std::vector<uint32_t>& forward;
  const std::vector<uint32_t>& forward_ptr;
  const std::vector<uint32_t>& inverse;
  const std::vector<uint32_t>& inverse_ptr;
};

struct CpuTree {
  std::vector<double> class_cost, node_cost;
  std::vector<uint32_t> best;
  std::vector<uint8_t> selected;
  double tree_cost = 0, dag_cost = 0;
  uint32_t iterations = 0;
};
struct CpuDag {
  // selected is the expansion before filtering; in_subgraph is the final
  // reachable, child-closed candidate set supplied to the solver.
  std::vector<uint8_t> selected, in_subgraph;
  uint32_t iterations = 0;
  uint64_t proposals = 0, neighbor_updates = 0, reverted = 0;
  uint64_t invalid_candidates_removed = 0;
};

CpuTree cpu_tree(const CpuGraph& graph, int threads);
CpuDag cpu_dag(const CpuGraph& graph,
               const std::vector<uint8_t>& tree_selected, int threads);
// Independent checks; throw on invalid roots, choices, dependencies or cycles.
// The cost of an extracted DAG counts each selected e-node exactly once.
double validate_cpu_solution(const CpuGraph& graph,
                             const std::vector<uint8_t>& selected,
                             bool integer_cost = false);
void validate_cpu_dag(const CpuGraph& graph,
                      const std::vector<uint8_t>& selected);
// Removing the last available choice of a class invalidates its dependents.
size_t prune_cpu_candidates(const CpuGraph& graph, std::vector<uint8_t>& candidates,
                            int threads);
// With require_acyclic=false, validate closure and report whether class levels
// are redundant; cyclic candidates can still use the general CP-SAT model.
bool validate_cpu_candidates(const CpuGraph& graph,
                             const std::vector<uint8_t>& candidates,
                             bool require_acyclic = true);
} // namespace xeg
#endif
