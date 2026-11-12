#include "gpu_common.cuh"

// One GPU thread evaluates one e-node slot in each sequential
// transposed batch.
//
// selected_state:
//   0 = unselected
//   1 = stable selected
//   2 = tentative selected
//
// Candidate IDs are assigned by:
//
//   u = threadIdx.x * num_batches + batch_index
//
// The host launches this kernel only once. Each lane walks
// through its transposed candidate sequence:
//
//   u = lane * num_batches + batch_index
//
// There is currently no explicit synchronization boundary
// between consecutive batch_index values. Tentative state and
// level updates are published live and may be observed by other
// threads during the same insertion phase.
//
// Candidate evaluation treats both state 1 and state 2 as selected.
// Later batches can observe tentative nodes inserted by earlier
// batches in the same insertion phase.
//
// Levelization/finalization occurs only after all batches finish.
__global__ void _gpu_dag_insert_batch_kl(
  const uint32_t num_endes,
  const uint32_t num_batches,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  // The original implementation launched one kernel for every
  // batch_index. This loop performs the same batch sequence
  // inside one persistent single-block kernel.
  for (uint32_t batch_index = 0u;
       batch_index < num_batches;
       ++batch_index) {
    // All candidate-level early exits must leave only this
    // do-while block. They must not return from the kernel,
    // because every thread must reach the batch barrier below.
    do {
      // Full-transpose assignment:
      //
      //   contiguous mapping:
      //     u = batch_index * batch_size + lane
      //
      //   transposed mapping:
      //     u = lane * num_batches + batch_index
      //
      // This is identical to the previous host-launched
      // batch mapping.
      const uint32_t u =
        threadIdx.x * num_batches +
        batch_index;

      // Some lanes in the final transposed batch may not map
      // to a valid e-node. They still have to participate in
      // the batch-level __syncthreads().
      if (u >= num_endes) {
        break;
      }

      // Atomic load because selected_state is updated live by
      // other threads in this block.
      if (atomicAdd(
            &selected_state[u],
            0u) != 0u) {
        break;
      }

      uint32_t parent_position =
        inverse_adj_ptr[u];

      const uint32_t parent_end =
        inverse_adj_ptr[u + 1u];

      uint32_t child_position =
        forward_adj_ptr[u];

      const uint32_t child_end =
        forward_adj_ptr[u + 1u];

      bool has_selected_parent = false;
      bool has_selected_child = false;
      bool has_selected_overlap = false;

      int max_parent_level = -1;
      int min_child_level = INT_MAX;

      // Both CSR segments are sorted and unique.
      //
      // Merge them with two pointers so that:
      //   neighbor only in inverse CSR -> process as parent
      //   neighbor only in forward CSR -> process as child
      //   neighbor in both             -> exact overlap check
      //
      // Complexity:
      //   O(parent_degree + child_degree)
      while (parent_position < parent_end ||
             child_position < child_end) {
        const bool take_parent_only =
          child_position >= child_end ||
          (parent_position < parent_end &&
           inverse_adj[parent_position] <
             forward_adj[child_position]);

        const bool take_child_only =
          parent_position >= parent_end ||
          (child_position < child_end &&
           forward_adj[child_position] <
             inverse_adj[parent_position]);

        if (take_parent_only) {
          const uint32_t parent =
            inverse_adj[parent_position];

          const uint32_t parent_state =
            atomicAdd(
              &selected_state[parent],
              0u);

          if (parent_state != 0u) {
            const int parent_level =
              atomicAdd(
                &level[parent],
                0);

            if (parent_level >= 0) {
              has_selected_parent = true;

              if (parent_level >
                  max_parent_level) {
                max_parent_level =
                  parent_level;
              }
            }
          }

          ++parent_position;
          continue;
        }

        if (take_child_only) {
          const uint32_t child =
            forward_adj[child_position];

          const uint32_t child_state =
            atomicAdd(
              &selected_state[child],
              0u);

          if (child_state != 0u) {
            const int child_level =
              atomicAdd(
                &level[child],
                0);

            if (child_level >= 0) {
              has_selected_child = true;

              if (child_level <
                  min_child_level) {
                min_child_level =
                  child_level;
              }
            }
          }

          ++child_position;
          continue;
        }

        // The same e-node appears in both sorted segments.
        const uint32_t neighbor =
          inverse_adj[parent_position];

        const uint32_t neighbor_state =
          atomicAdd(
            &selected_state[neighbor],
            0u);

        if (neighbor_state != 0u) {
          const int neighbor_level =
            atomicAdd(
              &level[neighbor],
              0);

          if (neighbor_level >= 0) {
            has_selected_parent = true;
            has_selected_child = true;
            has_selected_overlap = true;

            if (neighbor_level >
                max_parent_level) {
              max_parent_level =
                neighbor_level;
            }

            if (neighbor_level <
                min_child_level) {
              min_child_level =
                neighbor_level;
            }
          }
        }

        ++parent_position;
        ++child_position;
      }

      if (has_selected_overlap) {
        break;
      }

      if (max_parent_level >=
          min_child_level) {
        break;
      }

      int proposed_level = -1;

      if (is_top_down_insertion) {
        if (!has_selected_parent) {
          break;
        }

        proposed_level =
          max_parent_level + 1;
      }
      else {
        if (!has_selected_child) {
          break;
        }

        proposed_level =
          min_child_level - 1;
      }

      if (proposed_level < 0) {
        break;
      }

      // Publish the proposed level before publishing state 2.
      //
      // Keep __threadfence() because another thread in the same
      // batch may observe state 2 before the end-of-batch
      // __syncthreads(). The level must already be visible when
      // state 2 becomes visible.
      level[u] = proposed_level;

      // __threadfence(); // Note: open this will have worse soln Q than close it. So we close it

      // Exactly one successful committer increments u0.
      if (atomicCAS(
            &selected_state[u],
            0u,
            2u) == 0u) {
        atomicAdd(
          u0,
          1u);
      }
    } while (false);

    // No explicit block barrier is used here. This preserves
    // the currently evaluated live optimistic behavior.
    //
    // Enabling this barrier changes the order in which
    // tentative candidates become observable and has been
    // observed to change solution quality.
    // __syncthreads(); // Note: open this will have worse soln Q than close it. So we close it
  }
}

void XEG::_gpu_dag_insert_batch(
  const bool is_top_down_insertion)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_batch requires "
      "initialized GPU DAG buffers");
  }

  // Keep the original exact batch-count calculation.
  //
  // kGpuDagExactBatchSize is both:
  //   1. the number of threads in the single block
  //   2. the number of candidate lanes per transposed batch
  const uint32_t num_batches =
    (_num_ende +
     kGpuDagExactBatchSize - 1u) /
    kGpuDagExactBatchSize;

  // Launch one single-block kernel for the entire insertion
  // phase. The kernel processes all batch_index values
  // sequentially and synchronizes after every batch.
  _gpu_dag_insert_batch_kl
    <<<1u, kGpuDagExactBatchSize>>>(
      _num_ende,
      num_batches,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_batch_kl failed");
}

// ============================================================
// Direct one-thread-per-e-node live optimistic insertion
//
// One CUDA thread evaluates exactly one e-node:
//
//   u = blockIdx.x * blockDim.x + threadIdx.x
//
// Unlike the batch insertion kernel, this kernel has no
// internal batch loop. The full e-node range is evaluated in one
// multi-block launch.
//
// selected_state:
//   0 = unselected
//   1 = stable selected
//   2 = tentative selected
//
// Both states 1 and 2 are treated as selected while evaluating
// neighbors. A successful candidate is published as state 2.
// The existing levelization/finalization flow later validates
// or rolls back the tentative insertion.
// ============================================================
__global__ void _gpu_dag_insert_direct_kl(
  const uint32_t num_endes,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t u =
    blockIdx.x * blockDim.x +
    threadIdx.x;

  if (u >= num_endes) {
    return;
  }

  // selected_state is updated live by other GPU threads.
  if (atomicAdd(
        &selected_state[u],
        0u) != 0u) {
    return;
  }

  uint32_t parent_position =
    inverse_adj_ptr[u];

  const uint32_t parent_end =
    inverse_adj_ptr[u + 1u];

  uint32_t child_position =
    forward_adj_ptr[u];

  const uint32_t child_end =
    forward_adj_ptr[u + 1u];

  bool has_selected_parent = false;
  bool has_selected_child = false;
  bool has_selected_overlap = false;

  int max_parent_level = -1;
  int min_child_level = INT_MAX;

  // The forward and inverse CSR segments are sorted and unique.
  //
  // Merge the two segments so parent/child processing and exact
  // overlap detection require:
  //
  //   O(parent_degree + child_degree)
  while (parent_position < parent_end ||
         child_position < child_end) {
    const bool take_parent_only =
      child_position >= child_end ||
      (parent_position < parent_end &&
       inverse_adj[parent_position] <
         forward_adj[child_position]);

    const bool take_child_only =
      parent_position >= parent_end ||
      (child_position < child_end &&
       forward_adj[child_position] <
         inverse_adj[parent_position]);

    if (take_parent_only) {
      const uint32_t parent =
        inverse_adj[parent_position];

      const uint32_t parent_state =
        atomicAdd(
          &selected_state[parent],
          0u);

      if (parent_state != 0u) {
        const int parent_level =
          atomicAdd(
            &level[parent],
            0);

        if (parent_level >= 0) {
          has_selected_parent = true;

          if (parent_level >
              max_parent_level) {
            max_parent_level =
              parent_level;
          }
        }
      }

      ++parent_position;
      continue;
    }

    if (take_child_only) {
      const uint32_t child =
        forward_adj[child_position];

      const uint32_t child_state =
        atomicAdd(
          &selected_state[child],
          0u);

      if (child_state != 0u) {
        const int child_level =
          atomicAdd(
            &level[child],
            0);

        if (child_level >= 0) {
          has_selected_child = true;

          if (child_level <
              min_child_level) {
            min_child_level =
              child_level;
          }
        }
      }

      ++child_position;
      continue;
    }

    // The same selected e-node appears in both the parent and
    // child CSR segments. Inserting u would immediately violate
    // the local overlap condition.
    const uint32_t neighbor =
      inverse_adj[parent_position];

    const uint32_t neighbor_state =
      atomicAdd(
        &selected_state[neighbor],
        0u);

    if (neighbor_state != 0u) {
      const int neighbor_level =
        atomicAdd(
          &level[neighbor],
          0);

      if (neighbor_level >= 0) {
        has_selected_parent = true;
        has_selected_child = true;
        has_selected_overlap = true;

        if (neighbor_level >
            max_parent_level) {
          max_parent_level =
            neighbor_level;
        }

        if (neighbor_level <
            min_child_level) {
          min_child_level =
            neighbor_level;
        }
      }
    }

    ++parent_position;
    ++child_position;
  }

  if (has_selected_overlap) {
    return;
  }

  if (max_parent_level >=
      min_child_level) {
    return;
  }

  int proposed_level = -1;

  if (is_top_down_insertion) {
    if (!has_selected_parent) {
      return;
    }

    proposed_level =
      max_parent_level + 1;
  }
  else {
    if (!has_selected_child) {
      return;
    }

    proposed_level =
      min_child_level - 1;
  }

  if (proposed_level < 0) {
    return;
  }

  // Match the current batch insertion live publication
  // behavior.
  //
  // No global synchronization or thread fence is introduced.
  // Existing levelization/finalization validates the resulting
  // tentative graph.
  level[u] = proposed_level;

  if (atomicCAS(
        &selected_state[u],
        0u,
        2u) == 0u) {
    atomicAdd(
      u0,
      1u);
  }
}

void XEG::_gpu_dag_insert_direct(
  const bool is_top_down_insertion)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_direct requires "
      "initialized GPU DAG buffers");
  }  

  const uint32_t blocks =
    (_num_ende +
     kGpuThreads - 1u) /
    kGpuThreads;

  _gpu_dag_insert_direct_kl
    <<<blocks, kGpuThreads>>>(
      _num_ende,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_direct_kl failed");
}

// ============================================================
// Block-cooperative GPU DAG insertion
//
// Block-Batch:
//   one CUDA block acts as one transposed candidate lane
//
// Block-Direct:
//   one CUDA block evaluates one candidate
//
// Both kernels reuse the same candidate evaluator.
//
// Parent and child adjacency are scanned cooperatively.
// Exact overlap detection is folded into those scans:
//
//   shorter segment:
//     cooperative scan
//
//   longer segment:
//     binary search
//
// There is no separate third overlap pass.
// ============================================================

constexpr uint32_t
  kGpuDagBlockHasSelectedParent =
    1u << 0u;

constexpr uint32_t
  kGpuDagBlockHasSelectedChild =
    1u << 1u;

constexpr uint32_t
  kGpuDagBlockHasSelectedOverlap =
    1u << 2u;


struct GpuDagBlockThreadScratch
{
  int max_parent_level;
  int min_child_level;
  uint32_t flags;
};


static_assert(
  sizeof(GpuDagBlockThreadScratch) == 12u,
  "Unexpected Block insertion scratch size");


__device__ __forceinline__
bool gpu_dag_sorted_segment_contains(
  const uint32_t* adjacency,
  const uint32_t segment_begin,
  const uint32_t segment_end,
  const uint32_t target)
{
  uint32_t begin =
    segment_begin;

  uint32_t end =
    segment_end;

  while (begin < end) {
    const uint32_t middle =
      begin + (end - begin) / 2u;

    const uint32_t value =
      adjacency[middle];

    if (value < target) {
      begin =
        middle + 1u;
    }
    else {
      end =
        middle;
    }
  }

  return
    begin < segment_end &&
    adjacency[begin] == target;
}


__device__ __forceinline__
void gpu_dag_process_block_candidate(
  const uint32_t u,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0,
  GpuDagBlockThreadScratch* scratch,
  uint32_t* candidate_active)
{
  const uint32_t lane =
    threadIdx.x;

  // Every thread in this CUDA block evaluates the same
  // candidate.
  //
  // Thread 0 snapshots whether this candidate is currently
  // unselected. The final atomicCAS still resolves races with
  // other candidate blocks.
  if (lane == 0u) {
    *candidate_active =
      atomicAdd(
        &selected_state[u],
        0u) == 0u
        ? 1u
        : 0u;
  }

  __syncthreads();

  // This return is uniform because every thread reads the same
  // shared candidate_active value.
  if (*candidate_active == 0u) {
    return;
  }

  const uint32_t parent_begin =
    inverse_adj_ptr[u];

  const uint32_t parent_end =
    inverse_adj_ptr[u + 1u];

  const uint32_t child_begin =
    forward_adj_ptr[u];

  const uint32_t child_end =
    forward_adj_ptr[u + 1u];

  const uint32_t parent_count =
    parent_end - parent_begin;

  const uint32_t child_count =
    child_end - child_begin;

  // Scan only the shorter side for overlap checks.
  //
  // Equal-size segments use the parent side.
  const bool check_overlap_from_parents =
    parent_count <= child_count;

  GpuDagBlockThreadScratch local{
    -1,
    INT_MAX,
    0u
  };

  // ----------------------------------------------------------
  // Cooperative selected-parent scan
  //
  // Every thread processes:
  //
  //   parent_begin + lane
  //   parent_begin + lane + blockDim.x
  //   ...
  //
  // When the parent segment is shorter, overlap detection is
  // folded directly into this scan.
  // ----------------------------------------------------------
  for (uint32_t position =
         parent_begin + lane;
       position < parent_end;
       position += blockDim.x) {
    const uint32_t parent =
      inverse_adj[position];

    const uint32_t parent_state =
      atomicAdd(
        &selected_state[parent],
        0u);

    if (parent_state == 0u) {
      continue;
    }

    const int parent_level =
      atomicAdd(
        &level[parent],
        0);

    if (parent_level < 0) {
      continue;
    }

    local.flags |=
      kGpuDagBlockHasSelectedParent;

    if (parent_level >
        local.max_parent_level) {
      local.max_parent_level =
        parent_level;
    }

    // Only selected parents with valid levels require an
    // overlap lookup.
    //
    // Because the same e-node ID has one shared state and one
    // shared level, finding this ID in the child CSR proves an
    // exact selected parent-child overlap.
    if (check_overlap_from_parents &&
        gpu_dag_sorted_segment_contains(
          forward_adj,
          child_begin,
          child_end,
          parent)) {
      local.flags |=
        kGpuDagBlockHasSelectedParent |
        kGpuDagBlockHasSelectedChild |
        kGpuDagBlockHasSelectedOverlap;

      // This parent is also the same selected child.
      if (parent_level <
          local.min_child_level) {
        local.min_child_level =
          parent_level;
      }
    }
  }

  // ----------------------------------------------------------
  // Cooperative selected-child scan
  //
  // When the child segment is shorter, overlap detection is
  // folded directly into this scan.
  // ----------------------------------------------------------
  for (uint32_t position =
         child_begin + lane;
       position < child_end;
       position += blockDim.x) {
    const uint32_t child =
      forward_adj[position];

    const uint32_t child_state =
      atomicAdd(
        &selected_state[child],
        0u);

    if (child_state == 0u) {
      continue;
    }

    const int child_level =
      atomicAdd(
        &level[child],
        0);

    if (child_level < 0) {
      continue;
    }

    local.flags |=
      kGpuDagBlockHasSelectedChild;

    if (child_level <
        local.min_child_level) {
      local.min_child_level =
        child_level;
    }

    // Only selected children with valid levels require an
    // overlap lookup.
    if (!check_overlap_from_parents &&
        gpu_dag_sorted_segment_contains(
          inverse_adj,
          parent_begin,
          parent_end,
          child)) {
      local.flags |=
        kGpuDagBlockHasSelectedParent |
        kGpuDagBlockHasSelectedChild |
        kGpuDagBlockHasSelectedOverlap;

      // This child is also the same selected parent.
      if (child_level >
          local.max_parent_level) {
        local.max_parent_level =
          child_level;
      }
    }
  }

  // Publish this thread's local summary into dynamic shared
  // memory.
  scratch[lane] =
    local;

  __syncthreads();

  // ----------------------------------------------------------
  // Power-of-two shared-memory reduction
  //
  // The validator guarantees blockDim.x is one of:
  //
  //   64, 128, 256, 512, 1024
  //
  // Reduction combines:
  //
  //   maximum selected-parent level
  //   minimum selected-child level
  //   parent/child/overlap flags
  // ----------------------------------------------------------
  for (uint32_t stride =
         blockDim.x / 2u;
       stride > 0u;
       stride >>= 1u) {
    if (lane < stride) {
      const GpuDagBlockThreadScratch other =
        scratch[lane + stride];

      if (other.max_parent_level >
          scratch[lane].max_parent_level) {
        scratch[lane].max_parent_level =
          other.max_parent_level;
      }

      if (other.min_child_level <
          scratch[lane].min_child_level) {
        scratch[lane].min_child_level =
          other.min_child_level;
      }

      scratch[lane].flags |=
        other.flags;
    }

    __syncthreads();
  }

  // Thread 0 evaluates the same candidate legality conditions
  // used by the existing Thread insertion kernels.
  if (lane == 0u) {
    const GpuDagBlockThreadScratch summary =
      scratch[0];

    const bool has_selected_parent =
      (summary.flags &
       kGpuDagBlockHasSelectedParent) != 0u;

    const bool has_selected_child =
      (summary.flags &
       kGpuDagBlockHasSelectedChild) != 0u;

    const bool has_selected_overlap =
      (summary.flags &
       kGpuDagBlockHasSelectedOverlap) != 0u;

    bool valid =
      !has_selected_overlap &&
      summary.max_parent_level <
        summary.min_child_level;

    int proposed_level = -1;

    if (valid &&
        is_top_down_insertion) {
      valid =
        has_selected_parent;

      if (valid) {
        proposed_level =
          summary.max_parent_level + 1;
      }
    }
    else if (valid) {
      valid =
        has_selected_child;

      if (valid) {
        proposed_level =
          summary.min_child_level - 1;
      }
    }

    valid =
      valid &&
      proposed_level >= 0;

    if (valid) {
      // Preserve the existing live optimistic publication
      // order:
      //
      //   1. publish proposed level
      //   2. publish tentative state 2
      //
      // Do not add __threadfence().
      level[u] =
        proposed_level;

      if (atomicCAS(
            &selected_state[u],
            0u,
            2u) == 0u) {
        atomicAdd(
          u0,
          1u);
      }
    }
  }

  // Block-Batch reuses this dynamic shared-memory array for the
  // next candidate owned by the same block.
  //
  // This is only block-local synchronization. It does not
  // synchronize different candidate blocks.
  __syncthreads();
}

__global__ void _gpu_dag_insert_block_batch_kl(
  const uint32_t num_endes,
  const uint32_t num_batches,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  extern __shared__
    GpuDagBlockThreadScratch scratch[];

  __shared__ uint32_t candidate_active;

  // One CUDA block corresponds to one transposed candidate
  // lane.
  //
  // Candidate mapping:
  //
  //   u = blockIdx.x * num_batches + batch_index
  for (uint32_t batch_index = 0u;
       batch_index < num_batches;
       ++batch_index) {
    const uint32_t u =
      blockIdx.x * num_batches +
      batch_index;

    // u is identical for all threads in this CUDA block.
    // Therefore this branch is uniform and safe around the
    // candidate evaluator's __syncthreads().
    if (u >= num_endes) {
      continue;
    }

    gpu_dag_process_block_candidate(
      u,
      is_top_down_insertion,
      selected_state,
      forward_adj,
      forward_adj_ptr,
      inverse_adj,
      inverse_adj_ptr,
      level,
      u0,
      scratch,
      &candidate_active);
  }
}

__global__ void _gpu_dag_insert_block_direct_kl(
  const uint32_t num_endes,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t u =
    blockIdx.x;

  if (u >= num_endes) {
    return;
  }

  extern __shared__
    GpuDagBlockThreadScratch scratch[];

  __shared__ uint32_t candidate_active;

  gpu_dag_process_block_candidate(
    u,
    is_top_down_insertion,
    selected_state,
    forward_adj,
    forward_adj_ptr,
    inverse_adj,
    inverse_adj_ptr,
    level,
    u0,
    scratch,
    &candidate_active);
}

void XEG::_gpu_dag_insert_block_batch(
  const bool is_top_down_insertion,
  const uint32_t threads_per_candidate)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_block_batch requires "
      "initialized GPU DAG buffers");
  }

  // Preserve the existing 256-lane transposed Batch mapping.
  const uint32_t num_batches =
    (_num_ende +
     kGpuDagExactBatchSize - 1u) /
    kGpuDagExactBatchSize;

  // Launch only candidate-lane blocks that own at least one
  // valid candidate.
  //
  // candidate_lanes is at most kGpuDagExactBatchSize.
  const uint32_t candidate_lanes =
    (_num_ende +
     num_batches - 1u) /
    num_batches;

  const size_t shared_bytes =
    static_cast<size_t>(
      threads_per_candidate) *
    sizeof(GpuDagBlockThreadScratch);

  _gpu_dag_insert_block_batch_kl
    <<<candidate_lanes,
       threads_per_candidate,
       shared_bytes>>>(
      _num_ende,
      num_batches,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_block_batch_kl failed");
}

void XEG::_gpu_dag_insert_block_direct(
  const bool is_top_down_insertion,
  const uint32_t threads_per_candidate)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_block_direct requires "
      "initialized GPU DAG buffers");
  }

  const size_t shared_bytes =
    static_cast<size_t>(
      threads_per_candidate) *
    sizeof(GpuDagBlockThreadScratch);

  // One CUDA block per candidate e-node.
  _gpu_dag_insert_block_direct_kl
    <<<_num_ende,
       threads_per_candidate,
       shared_bytes>>>(
      _num_ende,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_block_direct_kl failed");
}

// ============================================================
// Block-cooperative Bitmap GPU DAG insertion
//
// This is the only Block Bitmap implementation.
//
// BITMAP_BITS == 0:
//   Existing exact Block kernels remain unchanged.
//
// BITMAP_BITS > 0:
//   Existing Block Bitmap Batch and Direct kernels use
//   Strategy A1 directly.
//
// Strategy A1:
//
//   1. Mix each selected, level-valid parent e-node ID.
//   2. Split the bitmap into equal lower and upper halves.
//   3. Set one mixed bit in each half.
//   4. A selected, level-valid child reports overlap only when
//      both corresponding bits are present.
//
// The two probes always occupy different halves of the bitmap.
// This reduces the probability that one ordinary modulo
// collision rejects a candidate.
//
// A two-probe hit may still be a false positive because its two
// bits may have been populated by different parents. Such a hit
// is conservatively rejected without exact verification.
//
// One runtime-sized parent bitmap remains stored in dynamic
// shared memory for each CUDA block.
//
// Shared-memory layout:
//
//   GpuDagBlockThreadScratch scratch[blockDim.x]
//   uint32_t parent_bitmap[bitmap_words]
//
// No new global allocation, kernel, launcher, runtime branch,
// or synchronization barrier is introduced.
// ============================================================

// ============================================================
// Common Strategy-A1 bitmap hash and shared-memory operations
//
// Every nonzero Block and Warp bitmap implementation uses the
// same hash and probe-position generator.
//
// Storage remains implementation-specific:
//
//   Warp Bitmap-32:
//     one uint32_t register
//
//   Warp Bitmap-64:
//     two uint32_t registers
//
//   Warp Bitmap-128...4096:
//     one shared-memory bitmap segment per warp
//
//   Block Bitmap-32...4096:
//     one shared-memory bitmap per CUDA block
//
// The common Strategy-A1 policy is:
//
//   1. mix the 32-bit e-node ID
//   2. choose one bit in the lower half
//   3. choose one bit in the upper half
//   4. report an approximate overlap only when both bits hit
// ============================================================

// Mix one 32-bit e-node ID before extracting bitmap positions.
//
// Arithmetic is intentionally uint32_t, so multiplication wraps
// modulo 2^32.
__device__ __forceinline__
uint32_t gpu_dag_bitmap_mix32(
  uint32_t value)
{
  value ^=
    value >> 16u;

  value *=
    0x7feb352du;

  value ^=
    value >> 15u;

  value *=
    0x846ca68bu;

  value ^=
    value >> 16u;

  return value;
}


// Strategy A1 assigns exactly two physical bitmap positions:
//
//   first:
//     one bit in the lower half
//
//   second:
//     one bit in the upper half
//
// Because the two probes belong to disjoint halves, they are
// always two different physical bitmap positions.
struct GpuDagBitmapA1ProbeBits
{
  uint32_t first;
  uint32_t second;
};


__device__ __forceinline__
GpuDagBitmapA1ProbeBits
gpu_dag_bitmap_a1_probe_bits(
  const uint32_t node,
  const uint32_t bitmap_mask)
{
  // Existing host configuration stores:
  //
  //   bitmap_mask = bitmap_bits - 1
  //
  // Therefore the runtime bitmap size can be recovered without
  // adding another kernel argument.
  const uint32_t bitmap_bits =
    bitmap_mask + 1u;

  const uint32_t half_bits =
    bitmap_bits >> 1u;

  const uint32_t half_mask =
    half_bits - 1u;

  const uint32_t mixed =
    gpu_dag_bitmap_mix32(
      node);

  GpuDagBitmapA1ProbeBits result;

  // Lower-half position:
  //
  //   [0, half_bits)
  result.first =
    mixed &
    half_mask;

  // Upper-half position:
  //
  //   [half_bits, bitmap_bits)
  result.second =
    half_bits +
    ((mixed >> 16u) &
     half_mask);

  return result;
}


// Insert one Strategy-A1 probe pair into a uint32_t-word bitmap.
//
// This helper is shared by:
//
//   Block shared-memory bitmap
//   Warp shared-memory bitmap
//
// Multiple cooperating threads may update the same word, so
// both updates use atomicOr().
__device__ __forceinline__
void gpu_dag_bitmap_shared_insert(
  uint32_t* bitmap,
  const uint32_t node,
  const uint32_t bitmap_mask)
{
  const GpuDagBitmapA1ProbeBits probe =
    gpu_dag_bitmap_a1_probe_bits(
      node,
      bitmap_mask);

  const uint32_t first_word =
    probe.first >> 5u;

  const uint32_t first_word_mask =
    1u << (probe.first & 31u);

  const uint32_t second_word =
    probe.second >> 5u;

  const uint32_t second_word_mask =
    1u << (probe.second & 31u);

  atomicOr(
    &bitmap[first_word],
    first_word_mask);

  atomicOr(
    &bitmap[second_word],
    second_word_mask);
}


// Probe one Strategy-A1 pair in a uint32_t-word bitmap.
//
// A candidate is conservatively rejected only when both probes
// hit. The two bits may still have been populated by different
// parents, so this remains an approximate overlap test.
__device__ __forceinline__
bool gpu_dag_bitmap_shared_contains(
  const uint32_t* bitmap,
  const uint32_t node,
  const uint32_t bitmap_mask)
{
  const GpuDagBitmapA1ProbeBits probe =
    gpu_dag_bitmap_a1_probe_bits(
      node,
      bitmap_mask);

  const uint32_t first_word =
    probe.first >> 5u;

  const uint32_t first_word_mask =
    1u << (probe.first & 31u);

  const uint32_t second_word =
    probe.second >> 5u;

  const uint32_t second_word_mask =
    1u << (probe.second & 31u);

  const bool first_hit =
    (bitmap[first_word] &
     first_word_mask) != 0u;

  const bool second_hit =
    (bitmap[second_word] &
     second_word_mask) != 0u;

  return
    first_hit &&
    second_hit;
}

template <typename T>
__device__ __forceinline__
T gpu_atomic_load_relaxed(
  T* ptr)
{
  cuda::atomic_ref<
    T,
    cuda::thread_scope_device> ref(*ptr);

  return ref.load(
    cuda::memory_order_relaxed);
}

template <typename T>
__device__ __forceinline__
void gpu_atomic_store_relaxed(
  T* ptr,
  const T value)
{
  cuda::atomic_ref<
    T,
    cuda::thread_scope_device> ref(*ptr);

  ref.store(
    value,
    cuda::memory_order_relaxed);
}

__device__ __forceinline__
void gpu_dag_process_block_bitmap_candidate(
  const uint32_t u,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0,
  GpuDagBlockThreadScratch* scratch,
  uint32_t* candidate_active,
  uint32_t* parent_bitmap,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  const uint32_t lane =
    threadIdx.x;

  // ----------------------------------------------------------
  // Candidate activity and bitmap initialization
  //
  // Thread 0 snapshots candidate activity.
  //
  // At the same time, all threads cooperatively clear the
  // runtime-sized bitmap.
  //
  // The following __syncthreads() publishes both:
  //
  //   candidate_active
  //   cleared parent_bitmap
  //
  // Therefore bitmap initialization does not require its own
  // additional barrier.
  // ----------------------------------------------------------
  if (lane == 0u) {
    *candidate_active =
      selected_state[u] == 0u
        ? 1u
        : 0u;
  }

  for (uint32_t word = lane;
       word < bitmap_words;
       word += blockDim.x) {
    parent_bitmap[word] =
      0u;
  }

  __syncthreads();

  // Uniform return because all threads read the same shared
  // candidate_active value.
  if (*candidate_active == 0u) {
    return;
  }

  const uint32_t parent_begin =
    inverse_adj_ptr[u];

  const uint32_t parent_end =
    inverse_adj_ptr[u + 1u];

  const uint32_t child_begin =
    forward_adj_ptr[u];

  const uint32_t child_end =
    forward_adj_ptr[u + 1u];

  GpuDagBlockThreadScratch local{
    -1,
    INT_MAX,
    0u
  };

  // ----------------------------------------------------------
  // Cooperative selected-parent scan
  //
  // Match the exact Block evaluator's participating set:
  //
  //   selected_state[parent] != 0
  //   level[parent] >= 0
  //
  // Only a parent satisfying both conditions is:
  //
  //   1. counted as a selected parent
  //   2. used to update max_parent_level
  //   3. inserted into the approximate overlap bitmap
  //
  // Unselected parents and parents with invalid levels must not
  // populate the bitmap. Otherwise they can create artificial
  // false-positive overlap rejections unrelated to the exact
  // selected parent-child sets.
  // ----------------------------------------------------------
  for (uint32_t position =
         parent_begin + lane;
       position < parent_end;
       position += blockDim.x) {
    const uint32_t parent =
      inverse_adj[position];

    const uint32_t parent_state =
      atomicAdd(
        &selected_state[parent],
        0u);

    if (parent_state == 0u) {
      continue;
    }

    const int parent_level =
      atomicAdd(
        &level[parent],
        0);

    if (parent_level < 0) {
      continue;
    }

    // const uint32_t parent_state =
    //   atomicAdd(
    //     &selected_state[parent],
    //     0u);

    // const int parent_level =
    //   level[parent];

    // if (parent_level < 0) {
    //   continue;
    // }

    // Only selected parents with valid levels participate in
    // Strategy-A1 approximate overlap detection.
    //
    // gpu_dag_bitmap_shared_insert() mixes this parent ID and
    // sets one bit in each half of the bitmap.
    //
    // This call must remain after both checks above. Moving it
    // before either check allows inactive or invalid parents to
    // pollute the bitmap and reject otherwise legal candidates.
    gpu_dag_bitmap_shared_insert(
      parent_bitmap,
      parent,
      bitmap_mask);

    local.flags |=
      kGpuDagBlockHasSelectedParent;

    if (parent_level >
        local.max_parent_level) {
      local.max_parent_level =
        parent_level;
    }
  }

  // Every parent bitmap update must complete before any child
  // thread probes the bitmap.
  //
  // This is the one additional block-wide barrier required by
  // the Bitmap evaluator.
  __syncthreads();

  // ----------------------------------------------------------
  // Cooperative child scan
  //
  // Preserve the existing selected-child semantics:
  //
  //   selected_state[child] != 0
  //   level[child] >= 0
  //
  // Only such children contribute to:
  //
  //   has_selected_child
  //   min_child_level
  //   overlap detection
  // ----------------------------------------------------------
  for (uint32_t position =
         child_begin + lane;
       position < child_end;
       position += blockDim.x) {
    const uint32_t child =
      forward_adj[position];

    // const uint32_t child_state =
    //   atomicAdd(
    //     &selected_state[child],
    //     0u);

    // if (child_state == 0u) {
    //   continue;
    // }

    // const int child_level =
    //   atomicAdd(
    //     &level[child],
    //     0);

    // if (child_level < 0) {
    //   continue;
    // }

    const uint32_t child_state =
      selected_state[child];

    if (child_state == 0u) {
      continue;
    }

    const int child_level =
      level[child];

    if (child_level < 0) {
      continue;
    }    

    local.flags |=
      kGpuDagBlockHasSelectedChild;

    if (child_level <
        local.min_child_level) {
      local.min_child_level =
        child_level;
    }

    // Strategy-A1 approximate overlap detection.
    //
    // The child ID is mixed using the same function as the
    // parents. The candidate is rejected only when both the
    // lower-half and upper-half probes hit.
    //
    // A two-probe hit may be:
    //
    //   true overlap
    //   two-bit hash collision
    //   two bits populated by different parents
    //
    // No exact verification is performed.
    if (gpu_dag_bitmap_shared_contains(
          parent_bitmap,
          child,
          bitmap_mask)) {
      local.flags |=
        kGpuDagBlockHasSelectedOverlap;
    }
  }

  // ----------------------------------------------------------
  // The remainder is intentionally copied from the existing
  // exact Block evaluator.
  //
  // Do not factor this into a shared helper during the first
  // implementation, because that would modify the Exact path.
  // ----------------------------------------------------------
  scratch[lane] =
    local;

  __syncthreads();

  for (uint32_t stride =
         blockDim.x / 2u;
       stride > 0u;
       stride >>= 1u) {
    if (lane < stride) {
      const GpuDagBlockThreadScratch other =
        scratch[lane + stride];

      if (other.max_parent_level >
          scratch[lane].max_parent_level) {
        scratch[lane].max_parent_level =
          other.max_parent_level;
      }

      if (other.min_child_level <
          scratch[lane].min_child_level) {
        scratch[lane].min_child_level =
          other.min_child_level;
      }

      scratch[lane].flags |=
        other.flags;
    }

    __syncthreads();
  }

  if (lane == 0u) {
    const GpuDagBlockThreadScratch summary =
      scratch[0];

    const bool has_selected_parent =
      (summary.flags &
       kGpuDagBlockHasSelectedParent) != 0u;

    const bool has_selected_child =
      (summary.flags &
       kGpuDagBlockHasSelectedChild) != 0u;

    const bool has_selected_overlap =
      (summary.flags &
       kGpuDagBlockHasSelectedOverlap) != 0u;

    bool valid =
      !has_selected_overlap &&
      summary.max_parent_level <
        summary.min_child_level;

    int proposed_level =
      -1;

    if (valid &&
        is_top_down_insertion) {
      valid =
        has_selected_parent;

      if (valid) {
        proposed_level =
          summary.max_parent_level + 1;
      }
    }
    else if (valid) {
      valid =
        has_selected_child;

      if (valid) {
        proposed_level =
          summary.min_child_level - 1;
      }
    }

    valid =
      valid &&
      proposed_level >= 0;

    if (valid) {
      // Preserve the existing live optimistic publication
      // order.
      //
      // Do not add __threadfence().
      level[u] =
        proposed_level;

      if (atomicCAS(
            &selected_state[u],
            0u,
            2u) == 0u) {
        atomicAdd(
          u0,
          1u);
      }
    }
    // if (valid) {
    //   gpu_atomic_store_relaxed(
    //     &level[u],
    //     proposed_level);

    //   gpu_atomic_store_relaxed(
    //     &selected_state[u],
    //     2u);

    //   // Many candidate blocks update the same counter.
    //   atomicAdd(
    //     u0,
    //     1u);
    // }    
  }

  // Required by Block-Batch before the same CUDA block reuses
  // its scratch and bitmap storage for the next candidate.
  __syncthreads();
}

__global__ void _gpu_dag_insert_block_bitmap_batch_kl(
  const uint32_t num_endes,
  const uint32_t num_batches,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  extern __shared__
    unsigned char dynamic_shared_memory[];

  auto* scratch =
    reinterpret_cast<
      GpuDagBlockThreadScratch*>(
        dynamic_shared_memory);

  auto* parent_bitmap =
    reinterpret_cast<uint32_t*>(
      scratch + blockDim.x);

  __shared__ uint32_t candidate_active;

  // Preserve the existing Block-Batch transposed mapping:
  //
  //   u = blockIdx.x * num_batches + batch_index
  for (uint32_t batch_index = 0u;
       batch_index < num_batches;
       ++batch_index) {
    const uint32_t u =
      blockIdx.x * num_batches +
      batch_index;

    if (u >= num_endes) {
      continue;
    }

    gpu_dag_process_block_bitmap_candidate(
      u,
      is_top_down_insertion,
      selected_state,
      forward_adj,
      forward_adj_ptr,
      inverse_adj,
      inverse_adj_ptr,
      level,
      u0,
      scratch,
      &candidate_active,
      parent_bitmap,
      bitmap_mask,
      bitmap_words);
  }
}


__global__ void _gpu_dag_insert_block_bitmap_direct_kl(
  const uint32_t num_endes,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  const uint32_t u =
    blockIdx.x;

  if (u >= num_endes) {
    return;
  }

  extern __shared__
    unsigned char dynamic_shared_memory[];

  auto* scratch =
    reinterpret_cast<
      GpuDagBlockThreadScratch*>(
        dynamic_shared_memory);

  auto* parent_bitmap =
    reinterpret_cast<uint32_t*>(
      scratch + blockDim.x);

  __shared__ uint32_t candidate_active;

  gpu_dag_process_block_bitmap_candidate(
    u,
    is_top_down_insertion,
    selected_state,
    forward_adj,
    forward_adj_ptr,
    inverse_adj,
    inverse_adj_ptr,
    level,
    u0,
    scratch,
    &candidate_active,
    parent_bitmap,
    bitmap_mask,
    bitmap_words);
}


void XEG::_gpu_dag_insert_block_bitmap_batch(
  const bool is_top_down_insertion,
  const uint32_t threads_per_candidate,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_block_bitmap_batch requires "
      "initialized GPU DAG buffers");
  }

  if (bitmap_words == 0u ||
      bitmap_words > 128u) {
    throw std::logic_error(
      "_gpu_dag_insert_block_bitmap_batch received "
      "an invalid bitmap word count");
  }

  // Bitmap Block insertion limits the number of concurrent
  // transposed candidate lanes to improve solution stability.
  // This does not change threads_per_candidate; a 64-thread
  // Block candidate still launches with 64 CUDA threads.
  const uint32_t num_batches =
    (_num_ende +
     kGpuDagBitmapBatchSize - 1u) /
    kGpuDagBitmapBatchSize;

  const uint32_t candidate_lanes =
    (_num_ende +
     num_batches - 1u) /
    num_batches;

  const size_t scratch_bytes =
    static_cast<size_t>(
      threads_per_candidate) *
    sizeof(GpuDagBlockThreadScratch);

  const size_t bitmap_bytes =
    static_cast<size_t>(
      bitmap_words) *
    sizeof(uint32_t);

  const size_t shared_bytes =
    scratch_bytes +
    bitmap_bytes;

  _gpu_dag_insert_block_bitmap_batch_kl
    <<<candidate_lanes,
       threads_per_candidate,
       shared_bytes>>>(
      _num_ende,
      num_batches,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0,
      bitmap_mask,
      bitmap_words);

  cudaCheckErrors(
    "_gpu_dag_insert_block_bitmap_batch_kl failed");
}


void XEG::_gpu_dag_insert_block_bitmap_direct(
  const bool is_top_down_insertion,
  const uint32_t threads_per_candidate,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_block_bitmap_direct requires "
      "initialized GPU DAG buffers");
  }

  if (bitmap_words == 0u ||
      bitmap_words > 128u) {
    throw std::logic_error(
      "_gpu_dag_insert_block_bitmap_direct received "
      "an invalid bitmap word count");
  }

  const size_t scratch_bytes =
    static_cast<size_t>(
      threads_per_candidate) *
    sizeof(GpuDagBlockThreadScratch);

  const size_t bitmap_bytes =
    static_cast<size_t>(
      bitmap_words) *
    sizeof(uint32_t);

  const size_t shared_bytes =
    scratch_bytes +
    bitmap_bytes;

  _gpu_dag_insert_block_bitmap_direct_kl
    <<<_num_ende,
       threads_per_candidate,
       shared_bytes>>>(
      _num_ende,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0,
      bitmap_mask,
      bitmap_words);

  cudaCheckErrors(
    "_gpu_dag_insert_block_bitmap_direct_kl failed");
}

// ============================================================
// Warp-cooperative GPU DAG insertion
//
// Warp-Batch:
//   one 32-lane warp acts as one transposed candidate lane
//
// Warp-Direct:
//   one 32-lane warp evaluates one candidate
//
// Multiple independent candidate warps may share one CUDA
// block. Warps do not communicate with each other.
//
// Candidate-local values remain in registers. Warp shuffle
// primitives replace the Block implementation's shared-memory
// scratch and block-wide reduction.
// ============================================================

constexpr uint32_t
  kGpuDagWarpSize =
    32u;

constexpr unsigned int
  kGpuDagWarpFullMask =
    0xffffffffu;

constexpr uint32_t
  kGpuDagWarpHasSelectedParent =
    1u << 0u;

constexpr uint32_t
  kGpuDagWarpHasSelectedChild =
    1u << 1u;

constexpr uint32_t
  kGpuDagWarpHasSelectedOverlap =
    1u << 2u;

__device__ __forceinline__
void gpu_dag_process_warp_candidate(
  const uint32_t u,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t lane =
    threadIdx.x &
    (kGpuDagWarpSize - 1u);

  // ----------------------------------------------------------
  // Warp-uniform candidate activity check
  //
  // Lane 0 snapshots the candidate state. The result is then
  // broadcast to all 32 lanes.
  //
  // The final atomicCAS remains responsible for publication.
  // ----------------------------------------------------------
  uint32_t candidate_active = 0u;

  if (lane == 0u) {
    candidate_active =
      atomicAdd(
        &selected_state[u],
        0u) == 0u
        ? 1u
        : 0u;
  }

  candidate_active =
    __shfl_sync(
      kGpuDagWarpFullMask,
      candidate_active,
      0);

  // This return is warp-uniform because every lane receives
  // the same lane-0 broadcast value.
  if (candidate_active == 0u) {
    return;
  }

  const uint32_t parent_begin =
    inverse_adj_ptr[u];

  const uint32_t parent_end =
    inverse_adj_ptr[u + 1u];

  const uint32_t child_begin =
    forward_adj_ptr[u];

  const uint32_t child_end =
    forward_adj_ptr[u + 1u];

  const uint32_t parent_count =
    parent_end - parent_begin;

  const uint32_t child_count =
    child_end - child_begin;

  // Match the Block evaluator exactly.
  //
  // Equal-sized segments check overlap from the parent side.
  const bool check_overlap_from_parents =
    parent_count <= child_count;

  // These scalar values should normally remain in registers.
  int local_max_parent_level =
    -1;

  int local_min_child_level =
    INT_MAX;

  uint32_t local_flags =
    0u;

  // ----------------------------------------------------------
  // Cooperative selected-parent scan
  //
  // lane 0:
  //   parent_begin + 0, +32, +64, ...
  //
  // lane 1:
  //   parent_begin + 1, +33, +65, ...
  // ----------------------------------------------------------
  for (uint32_t position =
         parent_begin + lane;
       position < parent_end;
       position += kGpuDagWarpSize) {
    const uint32_t parent =
      inverse_adj[position];

    const uint32_t parent_state =
      atomicAdd(
        &selected_state[parent],
        0u);

    if (parent_state == 0u) {
      continue;
    }

    const int parent_level =
      atomicAdd(
        &level[parent],
        0);

    if (parent_level < 0) {
      continue;
    }

    local_flags |=
      kGpuDagWarpHasSelectedParent;

    if (parent_level >
        local_max_parent_level) {
      local_max_parent_level =
        parent_level;
    }

    // Reuse the Block implementation's exact sorted-CSR
    // binary-search helper.
    if (check_overlap_from_parents &&
        gpu_dag_sorted_segment_contains(
          forward_adj,
          child_begin,
          child_end,
          parent)) {
      local_flags |=
        kGpuDagWarpHasSelectedParent |
        kGpuDagWarpHasSelectedChild |
        kGpuDagWarpHasSelectedOverlap;

      if (parent_level <
          local_min_child_level) {
        local_min_child_level =
          parent_level;
      }
    }
  }

  // ----------------------------------------------------------
  // Cooperative selected-child scan
  // ----------------------------------------------------------
  for (uint32_t position =
         child_begin + lane;
       position < child_end;
       position += kGpuDagWarpSize) {
    const uint32_t child =
      forward_adj[position];

    const uint32_t child_state =
      atomicAdd(
        &selected_state[child],
        0u);

    if (child_state == 0u) {
      continue;
    }

    const int child_level =
      atomicAdd(
        &level[child],
        0);

    if (child_level < 0) {
      continue;
    }

    local_flags |=
      kGpuDagWarpHasSelectedChild;

    if (child_level <
        local_min_child_level) {
      local_min_child_level =
        child_level;
    }

    if (!check_overlap_from_parents &&
        gpu_dag_sorted_segment_contains(
          inverse_adj,
          parent_begin,
          parent_end,
          child)) {
      local_flags |=
        kGpuDagWarpHasSelectedParent |
        kGpuDagWarpHasSelectedChild |
        kGpuDagWarpHasSelectedOverlap;

      if (child_level >
          local_max_parent_level) {
        local_max_parent_level =
          child_level;
      }
    }
  }

  // ----------------------------------------------------------
  // Register-based warp reduction
  //
  // This replaces:
  //
  //   scratch[lane] = local
  //   __syncthreads()
  //   shared-memory tree reduction
  //
  // with five register-shuffle stages:
  //
  //   16, 8, 4, 2, 1
  // ----------------------------------------------------------
  for (uint32_t offset =
         kGpuDagWarpSize / 2u;
       offset > 0u;
       offset >>= 1u) {
    const int other_max_parent_level =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_max_parent_level,
        offset);

    const int other_min_child_level =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_min_child_level,
        offset);

    const uint32_t other_flags =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_flags,
        offset);

    if (other_max_parent_level >
        local_max_parent_level) {
      local_max_parent_level =
        other_max_parent_level;
    }

    if (other_min_child_level <
        local_min_child_level) {
      local_min_child_level =
        other_min_child_level;
    }

    local_flags |=
      other_flags;
  }

  // After the reduction, lane 0 owns the whole candidate
  // summary.
  if (lane == 0u) {
    const bool has_selected_parent =
      (local_flags &
       kGpuDagWarpHasSelectedParent) != 0u;

    const bool has_selected_child =
      (local_flags &
       kGpuDagWarpHasSelectedChild) != 0u;

    const bool has_selected_overlap =
      (local_flags &
       kGpuDagWarpHasSelectedOverlap) != 0u;

    // Match gpu_dag_process_block_candidate exactly.
    bool valid =
      !has_selected_overlap &&
      local_max_parent_level <
        local_min_child_level;

    int proposed_level =
      -1;

    if (valid &&
        is_top_down_insertion) {
      valid =
        has_selected_parent;

      if (valid) {
        proposed_level =
          local_max_parent_level + 1;
      }
    }
    else if (valid) {
      valid =
        has_selected_child;

      if (valid) {
        proposed_level =
          local_min_child_level - 1;
      }
    }

    valid =
      valid &&
      proposed_level >= 0;

    if (valid) {
      // Preserve the existing live optimistic publication
      // order.
      //
      // Do not add __threadfence().
      level[u] =
        proposed_level;

      if (atomicCAS(
            &selected_state[u],
            0u,
            2u) == 0u) {
        atomicAdd(
          u0,
          1u);
      }
    }
  }

  // Warp-local synchronization only.
  //
  // This ensures all lanes finish this candidate before
  // Warp-Batch starts evaluating the next candidate owned by
  // the same warp.
  //
  // Never replace this with __syncthreads(), because other
  // warps in the same CUDA block may have returned or may be
  // processing different candidates.
  __syncwarp(
    kGpuDagWarpFullMask);
}

__global__ void _gpu_dag_insert_warp_batch_kl(
  const uint32_t num_endes,
  const uint32_t num_batches,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t warp_in_block =
    threadIdx.x / kGpuDagWarpSize;

  const uint32_t warps_per_block =
    blockDim.x / kGpuDagWarpSize;

  const uint32_t global_warp =
    blockIdx.x * warps_per_block +
    warp_in_block;

  // Match Block-Batch candidate-lane count.
  const uint32_t candidate_lanes =
    (num_endes +
     num_batches - 1u) /
    num_batches;

  // This condition is identical for all 32 lanes in the warp.
  if (global_warp >= candidate_lanes) {
    return;
  }

  // One warp acts as one transposed candidate lane.
  //
  // This mapping is exactly the Block-Batch mapping with:
  //
  //   blockIdx.x
  //
  // replaced by:
  //
  //   global_warp
  for (uint32_t batch_index = 0u;
       batch_index < num_batches;
       ++batch_index) {
    const uint32_t u =
      global_warp * num_batches +
      batch_index;

    // Uniform for the entire warp.
    if (u >= num_endes) {
      continue;
    }

    gpu_dag_process_warp_candidate(
      u,
      is_top_down_insertion,
      selected_state,
      forward_adj,
      forward_adj_ptr,
      inverse_adj,
      inverse_adj_ptr,
      level,
      u0);
  }
}

__global__ void _gpu_dag_insert_warp_direct_kl(
  const uint32_t num_endes,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t warp_in_block =
    threadIdx.x / kGpuDagWarpSize;

  const uint32_t warps_per_block =
    blockDim.x / kGpuDagWarpSize;

  const uint32_t global_warp =
    blockIdx.x * warps_per_block +
    warp_in_block;

  // One warp evaluates exactly one candidate.
  const uint32_t u =
    global_warp;

  // Uniform for the whole warp.
  if (u >= num_endes) {
    return;
  }

  gpu_dag_process_warp_candidate(
    u,
    is_top_down_insertion,
    selected_state,
    forward_adj,
    forward_adj_ptr,
    inverse_adj,
    inverse_adj_ptr,
    level,
    u0);
}

void XEG::_gpu_dag_insert_warp_batch(
  const bool is_top_down_insertion,
  const uint32_t threads_per_block)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_batch requires "
      "initialized GPU DAG buffers");
  }

  // Keep the existing 256-lane transposed Batch mapping.
  // Bitmap Block insertion limits the number of concurrent
  // transposed candidate lanes to improve solution stability.
  // This does not change threads_per_candidate; a 64-thread
  // Block candidate still launches with 64 CUDA threads.
  const uint32_t num_batches =
    (_num_ende +
     kGpuDagExactBatchSize - 1u) /
    kGpuDagExactBatchSize;

  const uint32_t candidate_lanes =
    (_num_ende +
     num_batches - 1u) /
    num_batches;

  const uint32_t warps_per_block =
    threads_per_block /
    kGpuDagWarpSize;

  const uint32_t blocks =
    (candidate_lanes +
     warps_per_block - 1u) /
    warps_per_block;

  _gpu_dag_insert_warp_batch_kl
    <<<blocks,
       threads_per_block>>>(
      _num_ende,
      num_batches,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_warp_batch_kl failed");
}

void XEG::_gpu_dag_insert_warp_direct(
  const bool is_top_down_insertion,
  const uint32_t threads_per_block)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_direct requires "
      "initialized GPU DAG buffers");
  }

  const uint32_t warps_per_block =
    threads_per_block /
    kGpuDagWarpSize;

  const uint32_t blocks =
    (_num_ende +
     warps_per_block - 1u) /
    warps_per_block;

  _gpu_dag_insert_warp_direct_kl
    <<<blocks,
       threads_per_block>>>(
      _num_ende,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_warp_direct_kl failed");
}

// ============================================================
// Warp-cooperative Register Bitmap GPU DAG insertion
//
// This section contains only the existing fixed-size register
// implementations.
//
// BITMAP_BITS == 0:
//   Existing Exact Warp evaluator, kernels, and launchers.
//
// BITMAP_BITS == 32:
//   Existing register-based Warp Bitmap-32 implementation.
//
// BITMAP_BITS == 64:
//   Existing register-based Warp Bitmap-64 implementation.
//
// BITMAP_BITS >= 128:
//   Implemented separately by the runtime-sized per-warp
//   shared-memory path below.
//
// For Bitmap-32 and Bitmap-64, each warp owns its bitmap
// entirely in registers. Different warps in the same CUDA block
// do not share bitmap state.
//
// No shared memory, atomicOr(), dynamic shared memory, or
// __syncthreads() is used by these fixed-size register paths.
//
// Only parents satisfying:
//
//   selected_state[parent] != 0
//   level[parent] >= 0
//
// populate a bitmap.
//
// Only children satisfying:
//
//   selected_state[child] != 0
//   level[child] >= 0
//
// probe the bitmap.
//
// A two-probe Strategy-A1 hit is conservatively treated as
// overlap without exact verification.
// ============================================================

__device__ __forceinline__
uint32_t gpu_dag_warp_or_reduce_and_broadcast(
  uint32_t value)
{
  // Reduce every lane's register fragment into lane 0.
  for (uint32_t offset =
         kGpuDagWarpSize / 2u;
       offset > 0u;
       offset >>= 1u) {
    value |=
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        value,
        offset);
  }

  // Every lane needs the complete parent bitmap during the
  // following selected-child scan.
  return __shfl_sync(
    kGpuDagWarpFullMask,
    value,
    0);
}

constexpr uint32_t
  kGpuDagWarpBitmap32Mask =
    31u;

constexpr uint32_t
  kGpuDagWarpBitmap64Mask =
    63u;

// ============================================================
// Warp Bitmap-32 candidate evaluator
// ============================================================

__device__ __forceinline__
void gpu_dag_process_warp_bitmap32_candidate(
  const uint32_t u,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t lane =
    threadIdx.x &
    (kGpuDagWarpSize - 1u);

  // ----------------------------------------------------------
  // Warp-uniform candidate activity check
  // ----------------------------------------------------------
  uint32_t candidate_active =
    0u;

  if (lane == 0u) {
    candidate_active =
      atomicAdd(
        &selected_state[u],
        0u) == 0u
        ? 1u
        : 0u;
  }

  candidate_active =
    __shfl_sync(
      kGpuDagWarpFullMask,
      candidate_active,
      0);

  if (candidate_active == 0u) {
    return;
  }

  const uint32_t parent_begin =
    inverse_adj_ptr[u];

  const uint32_t parent_end =
    inverse_adj_ptr[u + 1u];

  const uint32_t child_begin =
    forward_adj_ptr[u];

  const uint32_t child_end =
    forward_adj_ptr[u + 1u];

  int local_max_parent_level =
    -1;

  int local_min_child_level =
    INT_MAX;

  uint32_t local_flags =
    0u;

  // Each lane initially owns one fragment of the candidate's
  // 32-bit parent bitmap.
  uint32_t local_parent_bitmap =
    0u;

  // ----------------------------------------------------------
  // Cooperative selected-parent scan
  //
  // A parent enters the bitmap only after both checks:
  //
  //   selected_state[parent] != 0
  //   level[parent] >= 0
  // ----------------------------------------------------------
  for (uint32_t position =
         parent_begin + lane;
       position < parent_end;
       position += kGpuDagWarpSize) {
    const uint32_t parent =
      inverse_adj[position];

    const uint32_t parent_state =
      atomicAdd(
        &selected_state[parent],
        0u);

    if (parent_state == 0u) {
      continue;
    }

    const int parent_level =
      atomicAdd(
        &level[parent],
        0);

    if (parent_level < 0) {
      continue;
    }

    local_flags |=
      kGpuDagWarpHasSelectedParent;

    if (parent_level >
        local_max_parent_level) {
      local_max_parent_level =
        parent_level;
    }

    const GpuDagBitmapA1ProbeBits probe =
      gpu_dag_bitmap_a1_probe_bits(
        parent,
        kGpuDagWarpBitmap32Mask);

    // Bitmap-32 stores both Strategy-A1 probes in the same
    // uint32_t register:
    //
    //   first  -> bits 0...15
    //   second -> bits 16...31
    local_parent_bitmap |=
      1u << probe.first;

    local_parent_bitmap |=
      1u << probe.second;
  }

  // Combine all 32 lane-local fragments, then broadcast the
  // complete bitmap back to every lane.
  const uint32_t parent_bitmap =
    gpu_dag_warp_or_reduce_and_broadcast(
      local_parent_bitmap);

  // ----------------------------------------------------------
  // Cooperative selected-child scan
  //
  // Only selected children with valid levels probe the parent
  // bitmap.
  // ----------------------------------------------------------
  for (uint32_t position =
         child_begin + lane;
       position < child_end;
       position += kGpuDagWarpSize) {
    const uint32_t child =
      forward_adj[position];

    const uint32_t child_state =
      atomicAdd(
        &selected_state[child],
        0u);

    if (child_state == 0u) {
      continue;
    }

    const int child_level =
      atomicAdd(
        &level[child],
        0);

    if (child_level < 0) {
      continue;
    }

    local_flags |=
      kGpuDagWarpHasSelectedChild;

    if (child_level <
        local_min_child_level) {
      local_min_child_level =
        child_level;
    }

    const GpuDagBitmapA1ProbeBits probe =
      gpu_dag_bitmap_a1_probe_bits(
        child,
        kGpuDagWarpBitmap32Mask);

    const uint32_t first_mask =
      1u << probe.first;

    const uint32_t second_mask =
      1u << probe.second;

    const bool first_hit =
      (parent_bitmap &
       first_mask) != 0u;

    const bool second_hit =
      (parent_bitmap &
       second_mask) != 0u;

    // Strategy A1 reports overlap only when both probes hit.
    if (first_hit &&
        second_hit) {
      local_flags |=
        kGpuDagWarpHasSelectedOverlap;
    }
  }

  // ----------------------------------------------------------
  // Reduce the candidate summary into lane 0.
  //
  // This intentionally duplicates the existing Exact Warp
  // reduction instead of modifying or factoring that path.
  // ----------------------------------------------------------
  for (uint32_t offset =
         kGpuDagWarpSize / 2u;
       offset > 0u;
       offset >>= 1u) {
    const int other_max_parent_level =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_max_parent_level,
        offset);

    const int other_min_child_level =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_min_child_level,
        offset);

    const uint32_t other_flags =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_flags,
        offset);

    if (other_max_parent_level >
        local_max_parent_level) {
      local_max_parent_level =
        other_max_parent_level;
    }

    if (other_min_child_level <
        local_min_child_level) {
      local_min_child_level =
        other_min_child_level;
    }

    local_flags |=
      other_flags;
  }

  if (lane == 0u) {
    const bool has_selected_parent =
      (local_flags &
       kGpuDagWarpHasSelectedParent) != 0u;

    const bool has_selected_child =
      (local_flags &
       kGpuDagWarpHasSelectedChild) != 0u;

    const bool has_selected_overlap =
      (local_flags &
       kGpuDagWarpHasSelectedOverlap) != 0u;

    bool valid =
      !has_selected_overlap &&
      local_max_parent_level <
        local_min_child_level;

    int proposed_level =
      -1;

    if (valid &&
        is_top_down_insertion) {
      valid =
        has_selected_parent;

      if (valid) {
        proposed_level =
          local_max_parent_level + 1;
      }
    }
    else if (valid) {
      valid =
        has_selected_child;

      if (valid) {
        proposed_level =
          local_min_child_level - 1;
      }
    }

    valid =
      valid &&
      proposed_level >= 0;

    if (valid) {
      // Preserve the existing live optimistic publication
      // order.
      //
      // Do not add __threadfence().
      level[u] =
        proposed_level;

      if (atomicCAS(
            &selected_state[u],
            0u,
            2u) == 0u) {
        atomicAdd(
          u0,
          1u);
      }
    }
  }

  // Required before Warp-Batch reuses the same warp's register
  // state for its next candidate.
  __syncwarp(
    kGpuDagWarpFullMask);
}


// ============================================================
// Warp Bitmap-64 candidate evaluator
// ============================================================

__device__ __forceinline__
void gpu_dag_process_warp_bitmap64_candidate(
  const uint32_t u,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t lane =
    threadIdx.x &
    (kGpuDagWarpSize - 1u);

  uint32_t candidate_active =
    0u;

  if (lane == 0u) {
    candidate_active =
      atomicAdd(
        &selected_state[u],
        0u) == 0u
        ? 1u
        : 0u;
  }

  candidate_active =
    __shfl_sync(
      kGpuDagWarpFullMask,
      candidate_active,
      0);

  if (candidate_active == 0u) {
    return;
  }

  const uint32_t parent_begin =
    inverse_adj_ptr[u];

  const uint32_t parent_end =
    inverse_adj_ptr[u + 1u];

  const uint32_t child_begin =
    forward_adj_ptr[u];

  const uint32_t child_end =
    forward_adj_ptr[u + 1u];

  int local_max_parent_level =
    -1;

  int local_min_child_level =
    INT_MAX;

  uint32_t local_flags =
    0u;

  // The 64-bit bitmap is represented by two independent
  // uint32_t register words.
  uint32_t local_parent_bitmap_low =
    0u;

  uint32_t local_parent_bitmap_high =
    0u;

  // ----------------------------------------------------------
  // Cooperative selected-parent scan
  // ----------------------------------------------------------
  for (uint32_t position =
         parent_begin + lane;
       position < parent_end;
       position += kGpuDagWarpSize) {
    const uint32_t parent =
      inverse_adj[position];

    const uint32_t parent_state =
      atomicAdd(
        &selected_state[parent],
        0u);

    if (parent_state == 0u) {
      continue;
    }

    const int parent_level =
      atomicAdd(
        &level[parent],
        0);

    if (parent_level < 0) {
      continue;
    }

    local_flags |=
      kGpuDagWarpHasSelectedParent;

    if (parent_level >
        local_max_parent_level) {
      local_max_parent_level =
        parent_level;
    }

    const GpuDagBitmapA1ProbeBits probe =
      gpu_dag_bitmap_a1_probe_bits(
        parent,
        kGpuDagWarpBitmap64Mask);

    // For Bitmap-64, Strategy A1 guarantees:
    //
    //   first  -> lower half [0, 32)
    //   second -> upper half [32, 64)
    local_parent_bitmap_low |=
      1u << probe.first;

    local_parent_bitmap_high |=
      1u << (probe.second - 32u);
  }

  const uint32_t parent_bitmap_low =
    gpu_dag_warp_or_reduce_and_broadcast(
      local_parent_bitmap_low);

  const uint32_t parent_bitmap_high =
    gpu_dag_warp_or_reduce_and_broadcast(
      local_parent_bitmap_high);

  // ----------------------------------------------------------
  // Cooperative selected-child scan
  // ----------------------------------------------------------
  for (uint32_t position =
         child_begin + lane;
       position < child_end;
       position += kGpuDagWarpSize) {
    const uint32_t child =
      forward_adj[position];

    const uint32_t child_state =
      atomicAdd(
        &selected_state[child],
        0u);

    if (child_state == 0u) {
      continue;
    }

    const int child_level =
      atomicAdd(
        &level[child],
        0);

    if (child_level < 0) {
      continue;
    }

    local_flags |=
      kGpuDagWarpHasSelectedChild;

    if (child_level <
        local_min_child_level) {
      local_min_child_level =
        child_level;
    }

    const GpuDagBitmapA1ProbeBits probe =
      gpu_dag_bitmap_a1_probe_bits(
        child,
        kGpuDagWarpBitmap64Mask);

    const uint32_t first_mask =
      1u << probe.first;

    const uint32_t second_mask =
      1u << (probe.second - 32u);

    const bool first_hit =
      (parent_bitmap_low &
       first_mask) != 0u;

    const bool second_hit =
      (parent_bitmap_high &
       second_mask) != 0u;

    if (first_hit &&
        second_hit) {
      local_flags |=
        kGpuDagWarpHasSelectedOverlap;
    }
  }

  // ----------------------------------------------------------
  // Reduce candidate summary into lane 0.
  // ----------------------------------------------------------
  for (uint32_t offset =
         kGpuDagWarpSize / 2u;
       offset > 0u;
       offset >>= 1u) {
    const int other_max_parent_level =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_max_parent_level,
        offset);

    const int other_min_child_level =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_min_child_level,
        offset);

    const uint32_t other_flags =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_flags,
        offset);

    if (other_max_parent_level >
        local_max_parent_level) {
      local_max_parent_level =
        other_max_parent_level;
    }

    if (other_min_child_level <
        local_min_child_level) {
      local_min_child_level =
        other_min_child_level;
    }

    local_flags |=
      other_flags;
  }

  if (lane == 0u) {
    const bool has_selected_parent =
      (local_flags &
       kGpuDagWarpHasSelectedParent) != 0u;

    const bool has_selected_child =
      (local_flags &
       kGpuDagWarpHasSelectedChild) != 0u;

    const bool has_selected_overlap =
      (local_flags &
       kGpuDagWarpHasSelectedOverlap) != 0u;

    bool valid =
      !has_selected_overlap &&
      local_max_parent_level <
        local_min_child_level;

    int proposed_level =
      -1;

    if (valid &&
        is_top_down_insertion) {
      valid =
        has_selected_parent;

      if (valid) {
        proposed_level =
          local_max_parent_level + 1;
      }
    }
    else if (valid) {
      valid =
        has_selected_child;

      if (valid) {
        proposed_level =
          local_min_child_level - 1;
      }
    }

    valid =
      valid &&
      proposed_level >= 0;

    if (valid) {
      level[u] =
        proposed_level;

      if (atomicCAS(
            &selected_state[u],
            0u,
            2u) == 0u) {
        atomicAdd(
          u0,
          1u);
      }
    }
  }

  __syncwarp(
    kGpuDagWarpFullMask);
}


// ============================================================
// Warp Bitmap-32 kernels
// ============================================================

__global__
void _gpu_dag_insert_warp_bitmap32_batch_kl(
  const uint32_t num_endes,
  const uint32_t num_batches,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t warp_in_block =
    threadIdx.x /
    kGpuDagWarpSize;

  const uint32_t warps_per_block =
    blockDim.x /
    kGpuDagWarpSize;

  const uint32_t global_warp =
    blockIdx.x * warps_per_block +
    warp_in_block;

  const uint32_t candidate_lanes =
    (num_endes +
     num_batches - 1u) /
    num_batches;

  // Uniform for all lanes in this warp.
  if (global_warp >=
      candidate_lanes) {
    return;
  }

  for (uint32_t batch_index = 0u;
       batch_index < num_batches;
       ++batch_index) {
    const uint32_t u =
      global_warp * num_batches +
      batch_index;

    // Uniform for all lanes in this warp.
    if (u >= num_endes) {
      continue;
    }

    gpu_dag_process_warp_bitmap32_candidate(
      u,
      is_top_down_insertion,
      selected_state,
      forward_adj,
      forward_adj_ptr,
      inverse_adj,
      inverse_adj_ptr,
      level,
      u0);
  }
}


__global__
void _gpu_dag_insert_warp_bitmap32_direct_kl(
  const uint32_t num_endes,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t warp_in_block =
    threadIdx.x /
    kGpuDagWarpSize;

  const uint32_t warps_per_block =
    blockDim.x /
    kGpuDagWarpSize;

  const uint32_t global_warp =
    blockIdx.x * warps_per_block +
    warp_in_block;

  const uint32_t u =
    global_warp;

  if (u >= num_endes) {
    return;
  }

  gpu_dag_process_warp_bitmap32_candidate(
    u,
    is_top_down_insertion,
    selected_state,
    forward_adj,
    forward_adj_ptr,
    inverse_adj,
    inverse_adj_ptr,
    level,
    u0);
}


// ============================================================
// Warp Bitmap-64 kernels
// ============================================================

__global__
void _gpu_dag_insert_warp_bitmap64_batch_kl(
  const uint32_t num_endes,
  const uint32_t num_batches,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t warp_in_block =
    threadIdx.x /
    kGpuDagWarpSize;

  const uint32_t warps_per_block =
    blockDim.x /
    kGpuDagWarpSize;

  const uint32_t global_warp =
    blockIdx.x * warps_per_block +
    warp_in_block;

  const uint32_t candidate_lanes =
    (num_endes +
     num_batches - 1u) /
    num_batches;

  if (global_warp >=
      candidate_lanes) {
    return;
  }

  for (uint32_t batch_index = 0u;
       batch_index < num_batches;
       ++batch_index) {
    const uint32_t u =
      global_warp * num_batches +
      batch_index;

    if (u >= num_endes) {
      continue;
    }

    gpu_dag_process_warp_bitmap64_candidate(
      u,
      is_top_down_insertion,
      selected_state,
      forward_adj,
      forward_adj_ptr,
      inverse_adj,
      inverse_adj_ptr,
      level,
      u0);
  }
}


__global__
void _gpu_dag_insert_warp_bitmap64_direct_kl(
  const uint32_t num_endes,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0)
{
  const uint32_t warp_in_block =
    threadIdx.x /
    kGpuDagWarpSize;

  const uint32_t warps_per_block =
    blockDim.x /
    kGpuDagWarpSize;

  const uint32_t global_warp =
    blockIdx.x * warps_per_block +
    warp_in_block;

  const uint32_t u =
    global_warp;

  if (u >= num_endes) {
    return;
  }

  gpu_dag_process_warp_bitmap64_candidate(
    u,
    is_top_down_insertion,
    selected_state,
    forward_adj,
    forward_adj_ptr,
    inverse_adj,
    inverse_adj_ptr,
    level,
    u0);
}


// ============================================================
// Warp Bitmap-32 host launchers
// ============================================================

void XEG::_gpu_dag_insert_warp_bitmap32_batch(
  const bool is_top_down_insertion,
  const uint32_t threads_per_block)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap32_batch requires "
      "initialized GPU DAG buffers");
  }

  // Warp Bitmap-32 limits the number of concurrent
  // transposed candidate lanes to improve solution stability.
  //
  // This changes only the candidate-lane batch mapping.
  // threads_per_block and the number of CUDA threads assigned
  // to each candidate warp remain unchanged.
  const uint32_t num_batches =
    (_num_ende +
     kGpuDagBitmapBatchSize - 1u) /
    kGpuDagBitmapBatchSize;

  const uint32_t candidate_lanes =
    (_num_ende +
     num_batches - 1u) /
    num_batches;

  const uint32_t warps_per_block =
    threads_per_block /
    kGpuDagWarpSize;

  const uint32_t blocks =
    (candidate_lanes +
     warps_per_block - 1u) /
    warps_per_block;

  _gpu_dag_insert_warp_bitmap32_batch_kl
    <<<blocks,
       threads_per_block>>>(
      _num_ende,
      num_batches,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_warp_bitmap32_batch_kl failed");
}


void XEG::_gpu_dag_insert_warp_bitmap32_direct(
  const bool is_top_down_insertion,
  const uint32_t threads_per_block)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap32_direct requires "
      "initialized GPU DAG buffers");
  }

  const uint32_t warps_per_block =
    threads_per_block /
    kGpuDagWarpSize;

  const uint32_t blocks =
    (_num_ende +
     warps_per_block - 1u) /
    warps_per_block;

  _gpu_dag_insert_warp_bitmap32_direct_kl
    <<<blocks,
       threads_per_block>>>(
      _num_ende,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_warp_bitmap32_direct_kl failed");
}


// ============================================================
// Warp Bitmap-64 host launchers
// ============================================================

void XEG::_gpu_dag_insert_warp_bitmap64_batch(
  const bool is_top_down_insertion,
  const uint32_t threads_per_block)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap64_batch requires "
      "initialized GPU DAG buffers");
  }

  // Warp Bitmap-64 limits the number of concurrent
  // transposed candidate lanes to improve solution stability.
  //
  // This changes only the candidate-lane batch mapping.
  // threads_per_block and the number of CUDA threads assigned
  // to each candidate warp remain unchanged.
  const uint32_t num_batches =
    (_num_ende +
     kGpuDagBitmapBatchSize - 1u) /
    kGpuDagBitmapBatchSize;

  const uint32_t candidate_lanes =
    (_num_ende +
     num_batches - 1u) /
    num_batches;

  const uint32_t warps_per_block =
    threads_per_block /
    kGpuDagWarpSize;

  const uint32_t blocks =
    (candidate_lanes +
     warps_per_block - 1u) /
    warps_per_block;

  _gpu_dag_insert_warp_bitmap64_batch_kl
    <<<blocks,
       threads_per_block>>>(
      _num_ende,
      num_batches,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_warp_bitmap64_batch_kl failed");
}


void XEG::_gpu_dag_insert_warp_bitmap64_direct(
  const bool is_top_down_insertion,
  const uint32_t threads_per_block)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap64_direct requires "
      "initialized GPU DAG buffers");
  }

  const uint32_t warps_per_block =
    threads_per_block /
    kGpuDagWarpSize;

  const uint32_t blocks =
    (_num_ende +
     warps_per_block - 1u) /
    warps_per_block;

  _gpu_dag_insert_warp_bitmap64_direct_kl
    <<<blocks,
       threads_per_block>>>(
      _num_ende,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_insert_warp_bitmap64_direct_kl failed");
}

// ============================================================
// Warp-cooperative Shared-Memory Bitmap GPU DAG insertion
//
// This path is used only for:
//
//   BITMAP_BITS = 128, 256, 512, 1024, 2048, 4096
//
// The existing Warp Bitmap-32 and Bitmap-64 paths remain
// register-based and are not modified.
//
// Dynamic shared-memory layout:
//
//   uint32_t warp_parent_bitmaps
//     [warps_per_block][bitmap_words]
//
// Every warp receives one disjoint bitmap segment.
//
// Warps synchronize only with __syncwarp(). No __syncthreads()
// is used because different warps operate on independent bitmap
// segments and may process different candidates.
//
// Every nonzero Warp bitmap size uses the same Strategy-A1
// policy as the Block bitmap implementation:
//
//   1. mix the e-node ID
//   2. set one bit in the lower half
//   3. set one bit in the upper half
//
// A selected child is conservatively treated as an overlap only
// when both corresponding probe bits are present.
//
// No exact verification is performed after a two-probe hit.
// ============================================================


__device__ __forceinline__
void gpu_dag_process_warp_bitmap_shared_candidate(
  const uint32_t u,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0,
  uint32_t* parent_bitmap,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  const uint32_t lane =
    threadIdx.x &
    (kGpuDagWarpSize - 1u);

  // ----------------------------------------------------------
  // Warp-uniform candidate activity check
  // ----------------------------------------------------------
  uint32_t candidate_active =
    0u;

  if (lane == 0u) {
    candidate_active =
      atomicAdd(
        &selected_state[u],
        0u) == 0u
        ? 1u
        : 0u;
  }

  candidate_active =
    __shfl_sync(
      kGpuDagWarpFullMask,
      candidate_active,
      0);

  // Every lane observes the same broadcast value.
  if (candidate_active == 0u) {
    return;
  }

  // ----------------------------------------------------------
  // Clear only this warp's private bitmap segment.
  //
  // 128 bits:
  //   4 uint32_t words
  //
  // 4096 bits:
  //   128 uint32_t words
  //
  // With 32 lanes, each lane clears at most two words.
  // ----------------------------------------------------------
  for (uint32_t word = lane;
       word < bitmap_words;
       word += kGpuDagWarpSize) {
    parent_bitmap[word] =
      0u;
  }

  // Publish the cleared shared-memory words to this warp.
  __syncwarp(
    kGpuDagWarpFullMask);

  const uint32_t parent_begin =
    inverse_adj_ptr[u];

  const uint32_t parent_end =
    inverse_adj_ptr[u + 1u];

  const uint32_t child_begin =
    forward_adj_ptr[u];

  const uint32_t child_end =
    forward_adj_ptr[u + 1u];

  int local_max_parent_level =
    -1;

  int local_min_child_level =
    INT_MAX;

  uint32_t local_flags =
    0u;

  // ----------------------------------------------------------
  // Cooperative selected-parent scan
  // ----------------------------------------------------------
  for (uint32_t position =
         parent_begin + lane;
       position < parent_end;
       position += kGpuDagWarpSize) {
    const uint32_t parent =
      inverse_adj[position];

    const uint32_t parent_state =
      atomicAdd(
        &selected_state[parent],
        0u);

    if (parent_state == 0u) {
      continue;
    }

    const int parent_level =
      atomicAdd(
        &level[parent],
        0);

    if (parent_level < 0) {
      continue;
    }

    local_flags |=
      kGpuDagWarpHasSelectedParent;

    if (parent_level >
        local_max_parent_level) {
      local_max_parent_level =
        parent_level;
    }

    // Only selected parents with valid levels populate the
    // approximate overlap bitmap.
    gpu_dag_bitmap_shared_insert(
      parent_bitmap,
      parent,
      bitmap_mask);
  }

  // Every shared-memory atomicOr must finish before any lane
  // begins probing selected children.
  __syncwarp(
    kGpuDagWarpFullMask);

  // ----------------------------------------------------------
  // Cooperative selected-child scan
  // ----------------------------------------------------------
  for (uint32_t position =
         child_begin + lane;
       position < child_end;
       position += kGpuDagWarpSize) {
    const uint32_t child =
      forward_adj[position];

    const uint32_t child_state =
      atomicAdd(
        &selected_state[child],
        0u);

    if (child_state == 0u) {
      continue;
    }

    const int child_level =
      atomicAdd(
        &level[child],
        0);

    if (child_level < 0) {
      continue;
    }

    local_flags |=
      kGpuDagWarpHasSelectedChild;

    if (child_level <
        local_min_child_level) {
      local_min_child_level =
        child_level;
    }

    if (gpu_dag_bitmap_shared_contains(
          parent_bitmap,
          child,
          bitmap_mask)) {
      local_flags |=
        kGpuDagWarpHasSelectedOverlap;
    }
  }

  // ----------------------------------------------------------
  // Register-based warp reduction
  //
  // Match the existing Exact Warp and Warp Bitmap-32/64
  // implementations.
  // ----------------------------------------------------------
  for (uint32_t offset =
         kGpuDagWarpSize / 2u;
       offset > 0u;
       offset >>= 1u) {
    const int other_max_parent_level =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_max_parent_level,
        offset);

    const int other_min_child_level =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_min_child_level,
        offset);

    const uint32_t other_flags =
      __shfl_down_sync(
        kGpuDagWarpFullMask,
        local_flags,
        offset);

    if (other_max_parent_level >
        local_max_parent_level) {
      local_max_parent_level =
        other_max_parent_level;
    }

    if (other_min_child_level <
        local_min_child_level) {
      local_min_child_level =
        other_min_child_level;
    }

    local_flags |=
      other_flags;
  }

  // ----------------------------------------------------------
  // Lane 0 decides whether the candidate is legal.
  // ----------------------------------------------------------
  if (lane == 0u) {
    const bool has_selected_parent =
      (local_flags &
       kGpuDagWarpHasSelectedParent) != 0u;

    const bool has_selected_child =
      (local_flags &
       kGpuDagWarpHasSelectedChild) != 0u;

    const bool has_selected_overlap =
      (local_flags &
       kGpuDagWarpHasSelectedOverlap) != 0u;

    bool valid =
      !has_selected_overlap &&
      local_max_parent_level <
        local_min_child_level;

    int proposed_level =
      -1;

    if (valid &&
        is_top_down_insertion) {
      valid =
        has_selected_parent;

      if (valid) {
        proposed_level =
          local_max_parent_level + 1;
      }
    }
    else if (valid) {
      valid =
        has_selected_child;

      if (valid) {
        proposed_level =
          local_min_child_level - 1;
      }
    }

    valid =
      valid &&
      proposed_level >= 0;

    if (valid) {
      // Preserve the existing live optimistic publication
      // order:
      //
      //   1. level
      //   2. tentative state 2
      //
      // Do not add __threadfence().
      level[u] =
        proposed_level;

      if (atomicCAS(
            &selected_state[u],
            0u,
            2u) == 0u) {
        atomicAdd(
          u0,
          1u);
      }
    }
  }

  // Warp-Batch may reuse this warp's bitmap segment for its
  // next candidate.
  __syncwarp(
    kGpuDagWarpFullMask);
}

__global__
void _gpu_dag_insert_warp_bitmap_shared_batch_kl(
  const uint32_t num_endes,
  const uint32_t num_batches,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  const uint32_t warp_in_block =
    threadIdx.x /
    kGpuDagWarpSize;

  const uint32_t warps_per_block =
    blockDim.x /
    kGpuDagWarpSize;

  const uint32_t global_warp =
    blockIdx.x * warps_per_block +
    warp_in_block;

  const uint32_t candidate_lanes =
    (num_endes +
     num_batches - 1u) /
    num_batches;

  // Uniform for all 32 lanes in this warp.
  if (global_warp >=
      candidate_lanes) {
    return;
  }

  extern __shared__
    uint32_t warp_parent_bitmaps[];

  // Every warp receives a disjoint bitmap segment.
  uint32_t* parent_bitmap =
    warp_parent_bitmaps +
    warp_in_block * bitmap_words;

  // Preserve the current Warp Bitmap transposed mapping:
  //
  //   u = global_warp * num_batches + batch_index
  for (uint32_t batch_index = 0u;
       batch_index < num_batches;
       ++batch_index) {
    const uint32_t u =
      global_warp * num_batches +
      batch_index;

    // Uniform for all lanes in this warp.
    if (u >= num_endes) {
      continue;
    }

    gpu_dag_process_warp_bitmap_shared_candidate(
      u,
      is_top_down_insertion,
      selected_state,
      forward_adj,
      forward_adj_ptr,
      inverse_adj,
      inverse_adj_ptr,
      level,
      u0,
      parent_bitmap,
      bitmap_mask,
      bitmap_words);
  }
}

__global__
void _gpu_dag_insert_warp_bitmap_shared_direct_kl(
  const uint32_t num_endes,
  const bool is_top_down_insertion,
  uint32_t* selected_state,
  const uint32_t* forward_adj,
  const uint32_t* forward_adj_ptr,
  const uint32_t* inverse_adj,
  const uint32_t* inverse_adj_ptr,
  int* level,
  uint32_t* u0,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  const uint32_t warp_in_block =
    threadIdx.x /
    kGpuDagWarpSize;

  const uint32_t warps_per_block =
    blockDim.x /
    kGpuDagWarpSize;

  const uint32_t global_warp =
    blockIdx.x * warps_per_block +
    warp_in_block;

  const uint32_t u =
    global_warp;

  // Uniform for all 32 lanes in this warp.
  if (u >= num_endes) {
    return;
  }

  extern __shared__
    uint32_t warp_parent_bitmaps[];

  // Every warp receives a disjoint bitmap segment.
  uint32_t* parent_bitmap =
    warp_parent_bitmaps +
    warp_in_block * bitmap_words;

  gpu_dag_process_warp_bitmap_shared_candidate(
    u,
    is_top_down_insertion,
    selected_state,
    forward_adj,
    forward_adj_ptr,
    inverse_adj,
    inverse_adj_ptr,
    level,
    u0,
    parent_bitmap,
    bitmap_mask,
    bitmap_words);
}

void XEG::_gpu_dag_insert_warp_bitmap_shared_batch(
  const bool is_top_down_insertion,
  const uint32_t threads_per_block,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap_shared_batch "
      "requires initialized GPU DAG buffers");
  }

  // Shared Warp Bitmap is used only for:
  //
  //   128 bits  -> 4 words
  //   ...
  //   4096 bits -> 64 words
  if (bitmap_words < 4u ||
      bitmap_words > 128u) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap_shared_batch "
      "received an invalid bitmap word count");
  }

  const uint32_t bitmap_bits =
    bitmap_mask + 1u;

  if (bitmap_bits !=
      bitmap_words * 32u) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap_shared_batch "
      "received inconsistent bitmap parameters");
  }

  // Match the current Bitmap Warp Batch mapping.
  const uint32_t num_batches =
    (_num_ende +
     kGpuDagBitmapBatchSize - 1u) /
    kGpuDagBitmapBatchSize;

  const uint32_t candidate_lanes =
    (_num_ende +
     num_batches - 1u) /
    num_batches;

  const uint32_t warps_per_block =
    threads_per_block /
    kGpuDagWarpSize;

  const uint32_t blocks =
    (candidate_lanes +
     warps_per_block - 1u) /
    warps_per_block;

  // One bitmap per warp:
  //
  //   shared_bytes =
  //     warps_per_block *
  //     bitmap_words *
  //     sizeof(uint32_t)
  //
  // Maximum:
  //   1024 threads/block = 32 warps
  //   4096 bits          = 128 words
  //
  //   32 * 128 * 4 = 16384 bytes
  const size_t shared_bytes =
    static_cast<size_t>(
      warps_per_block) *
    static_cast<size_t>(
      bitmap_words) *
    sizeof(uint32_t);

  _gpu_dag_insert_warp_bitmap_shared_batch_kl
    <<<blocks,
       threads_per_block,
       shared_bytes>>>(
      _num_ende,
      num_batches,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0,
      bitmap_mask,
      bitmap_words);

  cudaCheckErrors(
    "_gpu_dag_insert_warp_bitmap_shared_batch_kl "
    "failed");
}

void XEG::_gpu_dag_insert_warp_bitmap_shared_direct(
  const bool is_top_down_insertion,
  const uint32_t threads_per_block,
  const uint32_t bitmap_mask,
  const uint32_t bitmap_words)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) ||
      _dvc_global_adj_1D_ptr == nullptr ||
      (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap_shared_direct "
      "requires initialized GPU DAG buffers");
  }

  if (bitmap_words < 4u ||
      bitmap_words > 128u) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap_shared_direct "
      "received an invalid bitmap word count");
  }

  const uint32_t bitmap_bits =
    bitmap_mask + 1u;

  if (bitmap_bits !=
      bitmap_words * 32u) {
    throw std::logic_error(
      "_gpu_dag_insert_warp_bitmap_shared_direct "
      "received inconsistent bitmap parameters");
  }

  const uint32_t warps_per_block =
    threads_per_block /
    kGpuDagWarpSize;

  const uint32_t blocks =
    (_num_ende +
     warps_per_block - 1u) /
    warps_per_block;

  const size_t shared_bytes =
    static_cast<size_t>(
      warps_per_block) *
    static_cast<size_t>(
      bitmap_words) *
    sizeof(uint32_t);

  _gpu_dag_insert_warp_bitmap_shared_direct_kl
    <<<blocks,
       threads_per_block,
       shared_bytes>>>(
      _num_ende,
      is_top_down_insertion,
      _dvc_picked_endes,
      _dvc_global_adj_1D,
      _dvc_global_adj_1D_ptr,
      _dvc_global_inv_adj_1D,
      _dvc_global_inv_adj_1D_ptr,
      _dvc_level,
      _dvc_dag_u0,
      bitmap_mask,
      bitmap_words);

  cudaCheckErrors(
    "_gpu_dag_insert_warp_bitmap_shared_direct_kl "
    "failed");
}

void XEG::_gpu_dag_insert_configured(
  const bool use_direct_mode,
  const bool is_top_down_insertion)
{
  switch (_dag_kernel) {
    case DagKernel::Thread:
      if (use_direct_mode) {
        _gpu_dag_insert_direct(
          is_top_down_insertion);
      }
      else {
        _gpu_dag_insert_batch(
          is_top_down_insertion);
      }

      return;

    case DagKernel::Warp:
      switch (_dag_bitmap_bits) {
        case 0u:
          // Existing Exact Warp implementation.
          if (use_direct_mode) {
            _gpu_dag_insert_warp_direct(
              is_top_down_insertion,
              _dag_threads_per_candidate);
          }
          else {
            _gpu_dag_insert_warp_batch(
              is_top_down_insertion,
              _dag_threads_per_candidate);
          }

          return;

        case 32u:
          // Existing register-based Warp Bitmap-32
          // implementation.
          if (use_direct_mode) {
            _gpu_dag_insert_warp_bitmap32_direct(
              is_top_down_insertion,
              _dag_threads_per_candidate);
          }
          else {
            _gpu_dag_insert_warp_bitmap32_batch(
              is_top_down_insertion,
              _dag_threads_per_candidate);
          }

          return;

        case 64u:
          // Existing register-based Warp Bitmap-64
          // implementation.
          if (use_direct_mode) {
            _gpu_dag_insert_warp_bitmap64_direct(
              is_top_down_insertion,
              _dag_threads_per_candidate);
          }
          else {
            _gpu_dag_insert_warp_bitmap64_batch(
              is_top_down_insertion,
              _dag_threads_per_candidate);
          }

          return;

        case 128u:
        case 256u:
        case 512u:
        case 1024u:
        case 2048u:
        case 4096u:
          // Runtime-sized per-warp shared-memory bitmap
          // implementation.
          if (use_direct_mode) {
            _gpu_dag_insert_warp_bitmap_shared_direct(
              is_top_down_insertion,
              _dag_threads_per_candidate,
              _dag_bitmap_mask,
              _dag_bitmap_words);
          }
          else {
            _gpu_dag_insert_warp_bitmap_shared_batch(
              is_top_down_insertion,
              _dag_threads_per_candidate,
              _dag_bitmap_mask,
              _dag_bitmap_words);
          }

          return;

        default:
          throw std::logic_error(
            "Invalid Warp DAG bitmap configuration");
      }

    case DagKernel::Block:
      if (_dag_bitmap_bits == 0u) {
        // Existing exact Block implementation.
        if (use_direct_mode) {
          _gpu_dag_insert_block_direct(
            is_top_down_insertion,
            _dag_threads_per_candidate);
        }
        else {
          _gpu_dag_insert_block_batch(
            is_top_down_insertion,
            _dag_threads_per_candidate);
        }
      }
      else {
        // New isolated Bitmap Block implementation.
        if (use_direct_mode) {
          _gpu_dag_insert_block_bitmap_direct(
            is_top_down_insertion,
            _dag_threads_per_candidate,
            _dag_bitmap_mask,
            _dag_bitmap_words);
        }
        else {
          _gpu_dag_insert_block_bitmap_batch(
            is_top_down_insertion,
            _dag_threads_per_candidate,
            _dag_bitmap_mask,
            _dag_bitmap_words);
        }
      }

      return;    
  }

  throw std::logic_error(
    "Invalid DAG kernel configuration");
}



// Construct the selected-subgraph degree and append all
// zero-degree selected e-nodes to the initial queue.
//
// selected_state:
//   0 = unselected
//   1 = stable selected
//   2 = tentative selected
//
// Any nonzero state participates in levelization.
