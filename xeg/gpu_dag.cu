#include "gpu_common.cuh"

__global__ void _gpu_dag_init_levelization_queue_kl(
  const uint32_t num_endes,
  const uint32_t* selected_state,
  const uint32_t* incoming_adj,
  const uint32_t* incoming_adj_ptr,
  uint32_t* degree,
  int* level,
  uint32_t* queue,
  uint32_t* queue_size)
{
  const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;

  if (u >= num_endes) {
    return;
  }

  level[u] = -1;

  if (selected_state[u] == 0u) {
    degree[u] = 0u;
    return;
  }

  uint32_t selected_degree = 0u;

  const uint32_t start = incoming_adj_ptr[u];
  const uint32_t end = incoming_adj_ptr[u + 1];

  for (uint32_t k = start; k < end; ++k) {
    const uint32_t neighbor = incoming_adj[k];

    if (selected_state[neighbor] != 0u) {
      ++selected_degree;
    }
  }

  degree[u] = selected_degree;

  if (selected_degree == 0u) {
    const uint32_t position =
      atomicAdd(queue_size, 1u);

    queue[position] = u;
  }
}

// Process exactly one queue slice:
//
//   queue[queue_begin,
//         queue_begin + queue_count)
//
// Newly discovered zero-degree nodes are appended after the
// current queue tail and are processed by the next launch.
__global__ void _gpu_dag_process_queue_slice_kl(
  const uint32_t queue_count,
  const uint32_t queue_begin,
  uint32_t* queue,
  uint32_t* queue_size,
  const uint32_t* selected_state,
  const uint32_t* traversal_adj,
  const uint32_t* traversal_adj_ptr,
  uint32_t* degree,
  int* level,
  const int current_level)
{
  const uint32_t index =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= queue_count) {
    return;
  }

  const uint32_t u =
    queue[queue_begin + index];

  level[u] = current_level;

  const uint32_t start = traversal_adj_ptr[u];
  const uint32_t end = traversal_adj_ptr[u + 1];

  for (uint32_t k = start; k < end; ++k) {
    const uint32_t v = traversal_adj[k];

    if (selected_state[v] == 0u) {
      continue;
    }

    const uint32_t old_degree =
      atomicSub(&degree[v], 1u);

    if (old_degree == 1u) {
      const uint32_t position =
        atomicAdd(queue_size, 1u);

      queue[position] = v;
    }
  }
}

// Process a sequence of complete Kahn frontiers inside one
// persistent single-block kernel.
//
// The initial frontier is:
//
//   queue[initial_queue_begin,
//         initial_queue_begin + initial_queue_count)
//
// Each frontier is a strict snapshot. Nodes appended while the
// current frontier is processed are not consumed until every
// thread finishes the current level and reaches __syncthreads().
//
// The kernel exits in one of three ways:
//
//   Completed:
//     No next frontier remains. current_level is final.
//
//   FrontierTooWide:
//     The next frontier has not been processed. The host may
//     resume from the returned queue_begin, queue_count, and
//     current_level with the original multi-block kernel.
//
//   InvalidQueueState:
//     An append-only queue invariant was violated.
__global__ void _gpu_dag_process_persistent_tail_kl(
  const uint32_t num_endes,
  const uint32_t initial_queue_begin,
  const uint32_t initial_queue_count,
  const int32_t initial_current_level,
  const uint32_t emergency_exit_threshold,
  uint32_t* queue,
  uint32_t* queue_size,
  const uint32_t* selected_state,
  const uint32_t* traversal_adj,
  const uint32_t* traversal_adj_ptr,
  uint32_t* degree,
  int* level,
  xeg_detail::GpuDagPersistentState* persistent_state)
{
  __shared__ uint32_t frontier_begin;
  __shared__ uint32_t frontier_count;
  __shared__ int32_t current_level;
  __shared__ uint32_t invalid_queue_state;

  if (threadIdx.x == 0u) {
    frontier_begin = initial_queue_begin;
    frontier_count = initial_queue_count;
    current_level = initial_current_level;
    invalid_queue_state = 0u;
  }

  __syncthreads();

  while (true) {
    // All threads observe the same shared frontier_count, so all
    // threads take the same return branch.
    if (frontier_count == 0u) {
      if (threadIdx.x == 0u) {
        persistent_state->queue_begin =
          frontier_begin;

        persistent_state->queue_count =
          0u;

        persistent_state->current_level =
          current_level;

        persistent_state->exit_reason =
          xeg_detail::GpuDagPersistentExitReason::Completed;

      }

      return;
    }

    // Leave this frontier unprocessed. The host resumes from
    // exactly this queue range with the multi-block kernel.
    if (frontier_count >
        emergency_exit_threshold) {
      if (threadIdx.x == 0u) {
        persistent_state->queue_begin =
          frontier_begin;

        persistent_state->queue_count =
          frontier_count;

        persistent_state->current_level =
          current_level;

        persistent_state->exit_reason =
          xeg_detail::GpuDagPersistentExitReason::
            FrontierTooWide;
      }

      return;
    }

    // One block may process more nodes than it has threads.
    //
    // With 512 threads:
    //   1...512 nodes   -> one node round
    //   513...1024      -> two node rounds
    for (uint32_t index = threadIdx.x;
         index < frontier_count;
         index += blockDim.x) {
      const uint32_t u =
        queue[frontier_begin + index];

      level[u] = current_level;

      const uint32_t start =
        traversal_adj_ptr[u];

      const uint32_t end =
        traversal_adj_ptr[u + 1u];

      for (uint32_t k = start; k < end; ++k) {
        const uint32_t v =
          traversal_adj[k];

        if (selected_state[v] == 0u) {
          continue;
        }

        const uint32_t old_degree =
          atomicSub(&degree[v], 1u);

        if (old_degree == 1u) {
          const uint32_t position =
            atomicAdd(queue_size, 1u);

          queue[position] = v;
        }
      }
    }

    // Every node and edge in the current frontier must finish
    // before thread 0 snapshots the newly appended queue tail.
    __syncthreads();

    if (threadIdx.x == 0u) {
      const uint32_t new_queue_size =
        *queue_size;

      const uint32_t next_queue_begin =
        frontier_begin + frontier_count;

      // The queue is append-only and every selected node can
      // enter it at most once.
      if (new_queue_size < next_queue_begin ||
          new_queue_size > num_endes) {
        invalid_queue_state = 1u;

        persistent_state->queue_begin =
          frontier_begin;

        persistent_state->queue_count =
          frontier_count;

        persistent_state->current_level =
          current_level;

        persistent_state->exit_reason =
          xeg_detail::GpuDagPersistentExitReason::
            InvalidQueueState;
      }
      else {
        frontier_begin =
          next_queue_begin;

        frontier_count =
          new_queue_size - next_queue_begin;

        ++current_level;
      }
    }

    // Publish the next frontier snapshot to every thread.
    __syncthreads();

    if (invalid_queue_state != 0u) {
      return;
    }
  }
}

__global__ void _gpu_dag_reverse_level_kl(
  const uint32_t num_endes,
  const uint32_t* selected_state,
  int* level,
  const int num_levels)
{
  const uint32_t u =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (u >= num_endes ||
      selected_state[u] == 0u ||
      level[u] < 0) {
    return;
  }

  level[u] =
    num_levels - level[u] - 1;
}

// Finalize tentative nodes after one complete Kahn
// levelization pass.
//
// selected_state:
//   0 = unselected
//   1 = stable selected
//   2 = tentative selected
//
// A tentative node with residual degree greater than zero
// was not processed by Kahn levelization and is rolled back.
//
// A tentative node with zero residual degree survived and
// becomes a stable selected node.
__global__ void _gpu_dag_finalize_tentative_kl(
  const uint32_t num_endes,
  uint32_t* selected_state,
  const uint32_t* residual_degree,
  int* level,
  uint32_t* u0)
{
  const uint32_t u =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (u >= num_endes) {
    return;
  }

  if (selected_state[u] != 2u) {
    return;
  }

  if (residual_degree[u] == 0u) {
    selected_state[u] = 1u;
    return;
  }

  selected_state[u] = 0u;
  level[u] = -1;

  // u0 counts successful insertions in the current outer
  // iteration. A rolled-back tentative insertion must be
  // removed from that count.
  atomicSub(u0, 1u);
}

// Seed an append-only reachability queue.
//
// Only selected start nodes participate. atomicCAS on
// visited guarantees that each e-node is appended at most once,
// even if the start list contains duplicates.
__global__ void _gpu_reachability_seed_kl(
  const uint32_t num_endes,
  const uint32_t num_starts,
  const uint32_t* starts,
  const uint32_t* selected_state,
  uint32_t* visited,
  uint32_t* queue,
  uint32_t* queue_size)
{
  const uint32_t index =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_starts) {
    return;
  }

  const uint32_t u = starts[index];

  if (u >= num_endes) {
    return;
  }

  if (selected_state[u] == 0u) {
    return;
  }

  if (atomicCAS(
        &visited[u],
        0u,
        1u) == 0u) {
    const uint32_t position =
      atomicAdd(queue_size, 1u);

    queue[position] = u;
  }
}


// Process one append-only reachability queue slice:
//
//   queue[queue_begin,
//         queue_begin + queue_count)
//
// Newly discovered selected nodes are appended after the
// current queue tail. They are processed by the next launch.
//
// selected_state is immutable during root-to-leaf filtering,
// so ordinary device loads are sufficient.
__global__ void _gpu_reachability_process_queue_slice_kl(
  const uint32_t queue_count,
  const uint32_t queue_begin,
  uint32_t* queue,
  uint32_t* queue_size,
  const uint32_t* selected_state,
  const uint32_t* adjacency,
  const uint32_t* adjacency_ptr,
  uint32_t* visited)
{
  const uint32_t index =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= queue_count) {
    return;
  }

  const uint32_t u =
    queue[queue_begin + index];

  const uint32_t start =
    adjacency_ptr[u];

  const uint32_t end =
    adjacency_ptr[u + 1u];

  for (uint32_t k = start; k < end; ++k) {
    const uint32_t v = adjacency[k];

    if (selected_state[v] == 0u) {
      continue;
    }

    // Only the first thread that changes visited[v]
    // from 0 to 1 may append v to the queue.
    if (atomicCAS(
          &visited[v],
          0u,
          1u) == 0u) {
      const uint32_t position =
        atomicAdd(queue_size, 1u);

      queue[position] = v;
    }
  }
}


// Match the CPU get_nodes_in_subgraph() semantics.
//
// Do not modify selected_state. The final filtered graph is
// represented separately by in_subgraph.
__global__ void _gpu_build_in_subgraph_kl(
  const uint32_t num_endes,
  const uint32_t* selected_state,
  const uint32_t* from_roots,
  const uint32_t* from_leaves,
  uint32_t* in_subgraph)
{
  const uint32_t u =
    blockIdx.x * blockDim.x + threadIdx.x;

  if (u >= num_endes) {
    return;
  }

  in_subgraph[u] =
    selected_state[u] != 0u &&
    from_roots[u] != 0u &&
    from_leaves[u] != 0u
      ? 1u
      : 0u;
}

void XEG::_gpu_dag_levelize(
  const bool is_top_down)
{
  if (_gpu_options.hybrid_levelization) {
    _gpu_dag_levelize_hybrid(is_top_down);
  }
  else {
    _gpu_dag_levelize_host_driven(is_top_down);
  }
}

void XEG::_gpu_dag_levelize_host_driven(
  const bool is_top_down)
{
  // Top-down:
  //   incoming dependencies = selected parents
  //   traversal = parent -> child
  //
  // Bottom-up:
  //   incoming dependencies = selected children
  //   traversal = child -> parent

  const uint32_t* incoming_adj =
    is_top_down
      ? _dvc_global_inv_adj_1D
      : _dvc_global_adj_1D;

  const uint32_t* incoming_adj_ptr =
    is_top_down
      ? _dvc_global_inv_adj_1D_ptr
      : _dvc_global_adj_1D_ptr;

  const uint32_t* traversal_adj =
    is_top_down
      ? _dvc_global_adj_1D
      : _dvc_global_inv_adj_1D;

  const uint32_t* traversal_adj_ptr =
    is_top_down
      ? _dvc_global_adj_1D_ptr
      : _dvc_global_inv_adj_1D_ptr;

  uint32_t* degree =
    is_top_down
      ? _dvc_indegree
      : _dvc_inv_indegree;

  // Reuse the append-only queue allocated by tree construction.
  // Queue contents do not need to be cleared.
  CUDA_CHECK(cudaMemset(
    _dvc_q_sz,
    0,
    sizeof(uint32_t)));

  const uint32_t node_blocks =
    (_num_ende + kGpuThreads - 1) / kGpuThreads;

  _gpu_dag_init_levelization_queue_kl
    <<<node_blocks, kGpuThreads>>>(
      _num_ende,
      _dvc_picked_endes,
      incoming_adj,
      incoming_adj_ptr,
      degree,
      _dvc_level,
      _dvc_q,
      _dvc_q_sz);

  cudaCheckErrors(
    "_gpu_dag_init_levelization_queue_kl failed");

  uint32_t queue_size = 0u;

  CUDA_CHECK(_gpu_copy_control(&queue_size, _dvc_q_sz, sizeof(uint32_t)));

  uint32_t queue_begin = 0u;
  uint32_t queue_count = queue_size;
  int current_level = 0;

  while (queue_count > 0u) {
    const uint32_t queue_blocks =
      (queue_count + kGpuThreads - 1) / kGpuThreads;

    _gpu_dag_process_queue_slice_kl
      <<<queue_blocks, kGpuThreads>>>(
        queue_count,
        queue_begin,
        _dvc_q,
        _dvc_q_sz,
        _dvc_picked_endes,
        traversal_adj,
        traversal_adj_ptr,
        degree,
        _dvc_level,
        current_level);

    cudaCheckErrors(
      "_gpu_dag_process_queue_slice_kl failed");

    uint32_t new_queue_size = 0u;

    CUDA_CHECK(_gpu_copy_control(&new_queue_size, _dvc_q_sz, sizeof(uint32_t)));

    if (new_queue_size > _num_ende) {
      throw std::logic_error(
        "GPU DAG levelization queue exceeded number of e-nodes");
    }

    queue_begin += queue_count;
    queue_size = new_queue_size;
    queue_count = queue_size - queue_begin;
    ++current_level;
  }

  _gpu_num_levels = current_level;

  if (!is_top_down && current_level > 0) {
    _gpu_dag_reverse_level_kl
      <<<node_blocks, kGpuThreads>>>(
        _num_ende,
        _dvc_picked_endes,
        _dvc_level,
        current_level);

    cudaCheckErrors(
      "_gpu_dag_reverse_level_kl failed");
  }
}

void XEG::_gpu_dag_levelize_hybrid(
  const bool is_top_down)
{
  if (_dvc_dag_persistent_state == nullptr) {
    throw std::logic_error(
      "_gpu_dag_levelize_hybrid requires "
      "an allocated persistent-state buffer");
  }

  // Top-down:
  //   incoming dependencies = selected parents
  //   traversal = parent -> child
  //
  // Bottom-up:
  //   incoming dependencies = selected children
  //   traversal = child -> parent

  const uint32_t* incoming_adj =
    is_top_down
      ? _dvc_global_inv_adj_1D
      : _dvc_global_adj_1D;

  const uint32_t* incoming_adj_ptr =
    is_top_down
      ? _dvc_global_inv_adj_1D_ptr
      : _dvc_global_adj_1D_ptr;

  const uint32_t* traversal_adj =
    is_top_down
      ? _dvc_global_adj_1D
      : _dvc_global_inv_adj_1D;

  const uint32_t* traversal_adj_ptr =
    is_top_down
      ? _dvc_global_adj_1D_ptr
      : _dvc_global_inv_adj_1D_ptr;

  uint32_t* degree =
    is_top_down
      ? _dvc_indegree
      : _dvc_inv_indegree;

  // Reuse the append-only queue allocated by tree construction.
  // Queue contents do not need to be cleared.
  CUDA_CHECK(cudaMemset(
    _dvc_q_sz,
    0,
    sizeof(uint32_t)));

  const uint32_t node_blocks =
    (_num_ende + kGpuThreads - 1u) /
    kGpuThreads;

  // Degree construction and initial zero-degree queue remain
  // multi-block. This is unchanged from the original version.
  _gpu_dag_init_levelization_queue_kl
    <<<node_blocks, kGpuThreads>>>(
      _num_ende,
      _dvc_picked_endes,
      incoming_adj,
      incoming_adj_ptr,
      degree,
      _dvc_level,
      _dvc_q,
      _dvc_q_sz);

  cudaCheckErrors(
    "_gpu_dag_init_levelization_queue_kl failed");

  uint32_t initial_queue_size = 0u;

  CUDA_CHECK(_gpu_copy_control(&initial_queue_size, _dvc_q_sz, sizeof(uint32_t)));

  if (initial_queue_size > _num_ende) {
    throw std::logic_error(
      "GPU DAG initial levelization queue exceeded "
      "the number of e-nodes");
  }

  uint32_t queue_begin = 0u;
  uint32_t queue_count = initial_queue_size;

  int current_level = 0;

  uint32_t consecutive_small_levels = 0u;

  // The host retains the final level count without a diagnostic H2D copy.

  while (queue_count > 0u) {
    if (queue_count <=
        _gpu_options.persistent_enter_frontier) {
      ++consecutive_small_levels;
    }
    else {
      consecutive_small_levels = 0u;
    }

    // Enter persistent mode on the Nth consecutive small
    // frontier. The current frontier has not yet been processed.
    if (consecutive_small_levels >=
        _gpu_options.persistent_small_levels) {
      if constexpr (kEnableGpuDagIterationLog) {
        printf(
          "[GPU DAG Levelize] dir=%s, "
          "enter persistent, "
          "level=%d, frontier=%u, streak=%u\n",
          is_top_down ? "TD" : "BU",
          current_level,
          queue_count,
          consecutive_small_levels);
      }

      // Initial state is passed through kernel arguments, not
      // copied into the state buffer. This avoids one H2D copy.
      _gpu_dag_process_persistent_tail_kl
        <<<1u, _gpu_options.persistent_threads>>>(
          _num_ende,
          queue_begin,
          queue_count,
          static_cast<int32_t>(current_level),
          _gpu_options.persistent_exit_frontier,
          _dvc_q,
          _dvc_q_sz,
          _dvc_picked_endes,
          traversal_adj,
          traversal_adj_ptr,
          degree,
          _dvc_level,
          _dvc_dag_persistent_state);

      cudaCheckErrors(
        "_gpu_dag_process_persistent_tail_kl failed");

      // One compact synchronization after the entire persistent
      // segment, rather than one synchronization per level.
      xeg_detail::GpuDagPersistentState
        host_state{};

      CUDA_CHECK(_gpu_copy_control(&host_state, _dvc_dag_persistent_state, sizeof(host_state)));

      if (host_state.current_level < 0) {
        throw std::logic_error(
          "GPU DAG persistent levelization returned "
          "a negative current level");
      }

      if (static_cast<uint32_t>(
            host_state.current_level) >
          _num_ende) {
        throw std::logic_error(
          "GPU DAG persistent levelization returned "
          "too many levels");
      }

      if (host_state.queue_begin > _num_ende ||
          host_state.queue_count >
            _num_ende - host_state.queue_begin) {
        throw std::logic_error(
          "GPU DAG persistent levelization returned "
          "an invalid queue range");
      }

      queue_begin =
        host_state.queue_begin;

      queue_count =
        host_state.queue_count;

      current_level =
        static_cast<int>(
          host_state.current_level);

      if (host_state.exit_reason ==
          xeg_detail::GpuDagPersistentExitReason::
            Completed) {
        if (queue_count != 0u) {
          throw std::logic_error(
            "Completed persistent levelization "
            "returned a non-empty frontier");
        }


        if constexpr (kEnableGpuDagIterationLog) {
          printf(
            "[GPU DAG Levelize] dir=%s, "
            "persistent completed, "
            "num_levels=%d\n",
            is_top_down ? "TD" : "BU",
            current_level);
        }

        break;
      }

      if (host_state.exit_reason ==
          xeg_detail::GpuDagPersistentExitReason::
            FrontierTooWide) {
        if (queue_count <=
            _gpu_options.persistent_exit_frontier) {
          throw std::logic_error(
            "Persistent levelization reported a wide "
            "frontier below the emergency threshold");
        }

        if constexpr (kEnableGpuDagIterationLog) {
          printf(
            "[GPU DAG Levelize] dir=%s, "
            "persistent exit=frontier_too_wide, "
            "level=%d, frontier=%u\n",
            is_top_down ? "TD" : "BU",
            current_level,
            queue_count);
        }

        // The returned frontier has not been processed.
        // Process it using the original multi-block path.
        consecutive_small_levels = 0u;
        continue;
      }

      throw std::logic_error(
        "GPU DAG persistent levelization detected "
        "an invalid append-only queue state");
    }

    // ==========================================================
    // Original host-driven multi-block processing
    //
    // This block intentionally matches the original
    // _gpu_dag_levelize() implementation.
    // ==========================================================
    const uint32_t queue_blocks =
      (queue_count + kGpuThreads - 1u) /
      kGpuThreads;

    _gpu_dag_process_queue_slice_kl
      <<<queue_blocks, kGpuThreads>>>(
        queue_count,
        queue_begin,
        _dvc_q,
        _dvc_q_sz,
        _dvc_picked_endes,
        traversal_adj,
        traversal_adj_ptr,
        degree,
        _dvc_level,
        current_level);

    cudaCheckErrors(
      "_gpu_dag_process_queue_slice_kl failed");

    uint32_t new_queue_size = 0u;

    CUDA_CHECK(_gpu_copy_control(&new_queue_size, _dvc_q_sz, sizeof(uint32_t)));

    const uint32_t processed_queue_end =
      queue_begin + queue_count;

    if (new_queue_size < processed_queue_end) {
      throw std::logic_error(
        "GPU DAG levelization append-only queue shrank");
    }

    if (new_queue_size > _num_ende) {
      throw std::logic_error(
        "GPU DAG levelization queue exceeded "
        "the number of e-nodes");
    }

    queue_begin =
      processed_queue_end;

    queue_count =
      new_queue_size - queue_begin;

    ++current_level;
  }

  // Both paths already return the final level count to the host.
  _gpu_num_levels = current_level;

  // Bottom-up reverse remains the existing multi-block kernel.
  if (!is_top_down && current_level > 0) {
    _gpu_dag_reverse_level_kl
      <<<node_blocks, kGpuThreads>>>(
        _num_ende,
        _dvc_picked_endes,
        _dvc_level,
        current_level);

    cudaCheckErrors(
      "_gpu_dag_reverse_level_kl failed");
  }
}

void XEG::_gpu_dag_finalize_tentative(
  const bool is_top_down_levelization)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_indegree == nullptr ||
      _dvc_inv_indegree == nullptr ||
      _dvc_level == nullptr ||
      _dvc_dag_u0 == nullptr) {
    throw std::logic_error(
      "_gpu_dag_finalize_tentative requires "
      "initialized GPU DAG buffers");
  }

  const uint32_t* residual_degree =
    is_top_down_levelization
      ? _dvc_indegree
      : _dvc_inv_indegree;

  const uint32_t blocks =
    (_num_ende + kGpuThreads - 1) / kGpuThreads;

  _gpu_dag_finalize_tentative_kl
    <<<blocks, kGpuThreads>>>(
      _num_ende,
      _dvc_picked_endes,
      residual_degree,
      _dvc_level,
      _dvc_dag_u0);

  cudaCheckErrors(
    "_gpu_dag_finalize_tentative_kl failed");
}

void XEG::_gpu_mark_reachable(
  const uint32_t* starts,
  const uint32_t num_starts,
  const uint32_t* adjacency,
  const uint32_t* adjacency_ptr,
  uint32_t* visited)
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_q == nullptr ||
      _dvc_q_sz == nullptr ||
      adjacency_ptr == nullptr ||
      visited == nullptr) {
    throw std::logic_error(
      "_gpu_mark_reachable requires initialized "
      "GPU graph and queue buffers");
  }

  // A null adjacency data pointer is legal when the graph has
  // no edges because every CSR segment is empty. adjacency_ptr
  // remains required.
  if (num_starts > 0u && starts == nullptr) {
    throw std::logic_error(
      "_gpu_mark_reachable received a null start buffer");
  }

  // Queue contents do not need to be cleared. Reset only the
  // append position.
  CUDA_CHECK(cudaMemset(
    _dvc_q_sz,
    0,
    sizeof(uint32_t)));

  if (num_starts == 0u) {
    return;
  }

  const uint32_t seed_blocks =
    (num_starts +
     kGpuReachabilityThreads - 1u) /
    kGpuReachabilityThreads;

  _gpu_reachability_seed_kl
    <<<seed_blocks, kGpuReachabilityThreads>>>(
      _num_ende,
      num_starts,
      starts,
      _dvc_picked_endes,
      visited,
      _dvc_q,
      _dvc_q_sz);

  cudaCheckErrors(
    "_gpu_reachability_seed_kl failed");

  uint32_t queue_size = 0u;

  CUDA_CHECK(_gpu_copy_control(&queue_size, _dvc_q_sz, sizeof(uint32_t)));

  if (queue_size > _num_ende) {
    throw std::logic_error(
      "GPU reachability seed queue exceeded "
      "the number of e-nodes");
  }

  uint32_t queue_begin = 0u;

  while (queue_begin < queue_size) {
    const uint32_t queue_count =
      queue_size - queue_begin;

    const uint32_t queue_blocks =
      (queue_count +
       kGpuReachabilityThreads - 1u) /
      kGpuReachabilityThreads;

    _gpu_reachability_process_queue_slice_kl
      <<<queue_blocks, kGpuReachabilityThreads>>>(
        queue_count,
        queue_begin,
        _dvc_q,
        _dvc_q_sz,
        _dvc_picked_endes,
        adjacency,
        adjacency_ptr,
        visited);

    cudaCheckErrors(
      "_gpu_reachability_process_queue_slice_kl failed");

    uint32_t new_queue_size = 0u;

    CUDA_CHECK(_gpu_copy_control(&new_queue_size, _dvc_q_sz, sizeof(uint32_t)));

    if (new_queue_size < queue_size) {
      throw std::logic_error(
        "GPU reachability append-only queue shrank");
    }

    if (new_queue_size > _num_ende) {
      throw std::logic_error(
        "GPU reachability queue exceeded "
        "the number of e-nodes");
    }

    // Everything before the old queue tail has now been
    // processed. Newly appended entries form the next slice.
    queue_begin = queue_size;
    queue_size = new_queue_size;
  }
}

void XEG::_gpu_build_in_subgraph()
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_from_roots == nullptr ||
      _dvc_from_leaves == nullptr ||
      _dvc_in_subgraph == nullptr) {
    throw std::logic_error(
      "_gpu_build_in_subgraph requires initialized "
      "GPU filtering buffers");
  }

  const uint32_t blocks =
    (_num_ende +
     kGpuReachabilityThreads - 1u) /
    kGpuReachabilityThreads;

  _gpu_build_in_subgraph_kl
    <<<blocks, kGpuReachabilityThreads>>>(
      _num_ende,
      _dvc_picked_endes,
      _dvc_from_roots,
      _dvc_from_leaves,
      _dvc_in_subgraph);

  cudaCheckErrors(
    "_gpu_build_in_subgraph_kl failed");
}

void XEG::_gpu_filter_root_to_leaf()
{
  if (_num_ende == 0u) {
    return;
  }

  if (_dvc_picked_endes == nullptr ||
      _dvc_global_adj_1D_ptr == nullptr ||
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_from_roots == nullptr ||
      _dvc_from_leaves == nullptr ||
      _dvc_in_subgraph == nullptr) {
    throw std::logic_error(
      "_gpu_filter_root_to_leaf requires initialized "
      "GPU graph and filtering buffers");
  }

  if (!_global_adj_1D.empty() &&
      _dvc_global_adj_1D == nullptr) {
    throw std::logic_error(
      "GPU forward adjacency buffer is missing");
  }

  if (!_global_inv_adj_1D.empty() &&
      _dvc_global_inv_adj_1D == nullptr) {
    throw std::logic_error(
      "GPU inverse adjacency buffer is missing");
  }

  if (!_roots.empty() &&
      _dvc_roots == nullptr) {
    throw std::logic_error(
      "GPU root buffer is missing");
  }

  if (!_leaves.empty() &&
      _dvc_leaves == nullptr) {
    throw std::logic_error(
      "GPU leaf buffer is missing");
  }

  // Each traversal owns a separate visited array.
  CUDA_CHECK(cudaMemset(
    _dvc_from_roots,
    0,
    static_cast<size_t>(_num_ende) *
      sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(
    _dvc_from_leaves,
    0,
    static_cast<size_t>(_num_ende) *
      sizeof(uint32_t)));

  // Forward traversal:
  //
  // selected roots -> selected descendants
  _gpu_mark_reachable(
    _dvc_roots,
    static_cast<uint32_t>(_roots.size()),
    _dvc_global_adj_1D,
    _dvc_global_adj_1D_ptr,
    _dvc_from_roots);

  // Reverse traversal:
  //
  // selected global leaves -> selected ancestors
  _gpu_mark_reachable(
    _dvc_leaves,
    static_cast<uint32_t>(_leaves.size()),
    _dvc_global_inv_adj_1D,
    _dvc_global_inv_adj_1D_ptr,
    _dvc_from_leaves);

  _gpu_build_in_subgraph();
}

void XEG::_gpu_allocate_dag_buffers()
{
  // Validate the shared buffers produced by GPU tree construction.
  if (_dvc_picked_endes == nullptr ||
      _dvc_q == nullptr ||
      _dvc_q_sz == nullptr ||
      _dvc_global_adj_1D_ptr == nullptr ||      
      _dvc_global_inv_adj_1D_ptr == nullptr ||
      _dvc_indegree == nullptr ||
      _dvc_inv_indegree == nullptr ||
      _dvc_level == nullptr ||
      _dvc_in_subgraph == nullptr ||
      _dvc_from_roots == nullptr ||
      _dvc_from_leaves == nullptr) {
    throw std::logic_error(
      "GPU DAG construction requires initialized GPU tree/shared buffers");
  }

  if (!_roots.empty() && _dvc_roots == nullptr) {
    throw std::logic_error(
      "GPU root buffer was not initialized by GPU tree construction");
  }

  if (!_leaves.empty() && _dvc_leaves == nullptr) {
    throw std::logic_error(
      "GPU leaf buffer was not initialized by GPU tree construction");
  }

  // Safe if DAG allocation is requested more than once.
  // This function only frees DAG-owned scratch memory.
  _gpu_free_dag_buffers();

  cuda_malloc_array(_dvc_dag_new_inserted, _num_ende);
  cuda_malloc_array(_dvc_dag_insert_decision, _num_ende);
  cuda_malloc_array(_dvc_dag_candidate_level, _num_ende);
  cuda_malloc_array(_dvc_dag_u0, 1);

  cuda_malloc_array(
    _dvc_dag_persistent_state,
    1u);
}

void XEG::_gpu_initialize_dag_buffers()
{
  if (_dvc_dag_new_inserted == nullptr ||
      _dvc_dag_insert_decision == nullptr ||
      _dvc_dag_candidate_level == nullptr ||
      _dvc_dag_u0 == nullptr ||
      _dvc_dag_persistent_state == nullptr)
  {
    throw std::logic_error(
      "_gpu_initialize_dag_buffers called before DAG allocation");
  }

  // Reinitialize shared working state.
  //
  // Do not modify _dvc_picked_endes:
  // it contains the tree solution produced by Algorithm 1.
  CUDA_CHECK(cudaMemset(_dvc_level, 0xFF,
    static_cast<size_t>(_num_ende) * sizeof(int)));

  CUDA_CHECK(cudaMemset(_dvc_indegree, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_inv_indegree, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));


  // Initialize DAG-only insertion state.
  CUDA_CHECK(cudaMemset(_dvc_dag_new_inserted, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_dag_insert_decision, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_dag_candidate_level, 0xFF,
    static_cast<size_t>(_num_ende) * sizeof(int)));

  CUDA_CHECK(cudaMemset(_dvc_dag_u0, 0,
    sizeof(uint32_t)));
  
  CUDA_CHECK(cudaMemset(
    _dvc_dag_persistent_state,
    0,
    sizeof(xeg_detail::GpuDagPersistentState)));

  // Reuse the filtering buffers already allocated by tree construction.
  CUDA_CHECK(cudaMemset(_dvc_from_roots, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_from_leaves, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(_dvc_in_subgraph, 0,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t)));
}

void XEG::_gpu_free_dag_buffers() noexcept
{
  cuda_free_and_null(_dvc_dag_persistent_state);
  cuda_free_and_null(_dvc_dag_u0);
  cuda_free_and_null(_dvc_dag_candidate_level);
  cuda_free_and_null(_dvc_dag_insert_decision);
  cuda_free_and_null(_dvc_dag_new_inserted);
}

void XEG::_gpu_DAG_construction(
  const GpuDagInsertionPolicy insertion_policy)
{
  // ============================================================
  // Total DAG timing
  //
  // Total runtime is always measured and printed, regardless of
  // kEnableGpuDagIterationLog, because it is the primary metric
  // used for performance optimization.
  // ============================================================
  float gpu_dag_total_time_ms = 0.0f;

  cudaEvent_t start_gpu = nullptr;
  cudaEvent_t stop_gpu = nullptr;

  cudaEvent_t after_core_gpu = nullptr;
  auto release_events = gpu_scope_exit([&] {
    if (after_core_gpu) cudaEventDestroy(after_core_gpu);
    if (stop_gpu) cudaEventDestroy(stop_gpu);
    if (start_gpu) cudaEventDestroy(start_gpu);
  });
  CUDA_CHECK(cudaEventCreate(&start_gpu));
  CUDA_CHECK(cudaEventCreate(&stop_gpu));

  // ============================================================
  // Optional detailed timing
  //
  // These objects are used only when detailed GPU DAG logging
  // is enabled. [[maybe_unused]] is a compile-time annotation and
  // has no runtime overhead.
  // ============================================================
  [[maybe_unused]] float gpu_dag_core_time_ms = 0.0f;
  [[maybe_unused]] float gpu_filter_time_ms = 0.0f;


  if constexpr (kEnableGpuDagIterationLog) {
    CUDA_CHECK(cudaEventCreate(&after_core_gpu));
  }

  // Start total DAG construction timing.
  CUDA_CHECK(cudaEventRecord(start_gpu));

  // ============================================================
  // Algorithm 2 initialization
  //
  // Initial top-down levelization of the GPU tree result.
  // States 1 and 2 are both treated as selected.
  // At this point only states 0 and 1 should exist.
  // ============================================================
  _gpu_dag_levelize(true);

  uint32_t host_u0 = 1u;

  // Total number of completed bottom-up + top-down outer
  // iterations.
  uint32_t outer_iteration = 0u;

  // Number of completed iterations using each insertion mode.
  uint32_t batch_iterations = 0u;
  uint32_t direct_iterations = 0u;

  // Enable the finite Direct-mode limit only for the two bitmap
  // implementations:
  //
  //   Warp  + BITMAP_BITS > 0
  //   Block + BITMAP_BITS > 0
  //
  // Exact Warp, Exact Block, and Thread insertion continue until
  // the original host_u0 == 0 convergence condition is reached.
  const bool bitmap_direct_limit_enabled =
    _dag_bitmap_bits > 0u &&
    (_dag_kernel == DagKernel::Warp ||
    _dag_kernel == DagKernel::Block);

  // Largest survived insertion count observed during batch
  // insertion.
  uint32_t peak_survived = 0u;

  // Number of consecutive batch iterations whose survived
  // insertion count is at most _gpu_options.low_activity_percent of
  // the historical batch peak.
  uint32_t low_activity_streak = 0u;

  // false:
  //   use the transposed-batch insertion kernel.
  //
  // true:
  //   use the one-thread-per-e-node direct insertion kernel.
  bool use_direct_mode = false;

  // BatchEarlyStop terminates immediately when the
  // low-activity condition is reached.
  bool batch_early_stop_triggered = false;

  // Records termination caused by the Bitmap Direct-mode
  // iteration limit.
  //
  // This flag is set only when:
  //
  //   direct_iterations has reached the configured limit
  //   host_u0 is still greater than zero
  //
  // If the fifth Direct iteration naturally produces host_u0 ==
  // zero, termination is still reported as ordinary convergence.
  bool bitmap_direct_limit_triggered = false;

  // BatchThenDirect records the Batch -> Direct transition.
  bool switched_to_direct = false;
  uint32_t switch_iteration = 0u;
  uint32_t switch_survived = 0u;

  while (host_u0 > 0u) {
    // Bitmap Warp/Block Direct mode executes at most
    // _gpu_options.bitmap_max_direct_iterations complete outer
    // iterations.
    //
    // This check is placed at the beginning of the loop so that:
    //
    //   direct_iterations == 0 -> Direct iteration 1 may run
    //   direct_iterations == 4 -> Direct iteration 5 may run
    //   direct_iterations == 5 -> stop before iteration 6
    //
    // Exact modes never enter this branch because
    // bitmap_direct_limit_enabled is false.
    if (bitmap_direct_limit_enabled &&
        use_direct_mode &&
        direct_iterations >=
          _gpu_options.bitmap_max_direct_iterations) {
      bitmap_direct_limit_triggered = true;
      break;
    }
    // Keep one insertion mode for both the bottom-up and
    // top-down phases of this complete outer iteration.
    const bool iteration_uses_direct =
      use_direct_mode;

    // u0 counts the insertions that survive during the current
    // outer iteration.
    CUDA_CHECK(cudaMemset(
      _dvc_dag_u0,
      0,
      sizeof(uint32_t)));

    // ==========================================================
    // Bottom-up insertion
    // ==========================================================
    _gpu_dag_insert_configured(
      iteration_uses_direct,
      false);

    // This value is required only for detailed logging.
    [[maybe_unused]] uint32_t bottom_up_proposed = 0u;

    if constexpr (kEnableGpuDagIterationLog) {
      CUDA_CHECK(_gpu_copy_control(&bottom_up_proposed, _dvc_dag_u0, sizeof(uint32_t)));
    }

    _gpu_dag_levelize(false);
    _gpu_dag_finalize_tentative(false);

    // This value is required only for detailed logging.
    [[maybe_unused]] uint32_t bottom_up_survived = 0u;

    if constexpr (kEnableGpuDagIterationLog) {
      CUDA_CHECK(_gpu_copy_control(&bottom_up_survived, _dvc_dag_u0, sizeof(uint32_t)));
    }

    // ==========================================================
    // Top-down insertion
    // ==========================================================
    _gpu_dag_insert_configured(
      iteration_uses_direct,
      true);

    // This value is required only for detailed logging.
    [[maybe_unused]] uint32_t after_top_down_insert = 0u;

    if constexpr (kEnableGpuDagIterationLog) {
      CUDA_CHECK(_gpu_copy_control(&after_top_down_insert, _dvc_dag_u0, sizeof(uint32_t)));
    }

    _gpu_dag_levelize(true);
    _gpu_dag_finalize_tentative(true);

    // ==========================================================
    // Required outer-loop termination copy
    //
    // This copy is part of the algorithm, not merely logging.
    // It must remain enabled even when iteration logging is off.
    // ==========================================================
    uint32_t after_both_phases = 0u;

    CUDA_CHECK(_gpu_copy_control(&after_both_phases, _dvc_dag_u0, sizeof(uint32_t)));

    host_u0 = after_both_phases;
    ++outer_iteration;

    if (iteration_uses_direct) {
      ++direct_iterations;
    }
    else {
      ++batch_iterations;
    }

    // ==========================================================
    // Optional per-iteration diagnostics
    // ==========================================================
    if constexpr (kEnableGpuDagIterationLog) {
      const uint32_t top_down_proposed =
        after_top_down_insert -
        bottom_up_survived;

      const uint32_t top_down_survived =
        after_both_phases -
        bottom_up_survived;

      const uint32_t bottom_up_rollback =
        bottom_up_proposed -
        bottom_up_survived;

      const uint32_t top_down_rollback =
        top_down_proposed -
        top_down_survived;

      printf(
        "[GPU DAG] iteration=%u, mode=%s, "
        "BU proposed=%u survived=%u rollback=%u, "
        "TD proposed=%u survived=%u rollback=%u, "
        "total survived=%u\n",
        outer_iteration,
        iteration_uses_direct
          ? "direct"
          : "batch",
        bottom_up_proposed,
        bottom_up_survived,
        bottom_up_rollback,
        top_down_proposed,
        top_down_survived,
        top_down_rollback,
        after_both_phases);
    }

    // ==========================================================
    // Batch low-activity controller
    //
    // This controller is evaluated only while the current
    // iteration uses the batch insertion kernel.
    //
    // FullBatch:
    //   Ignore the low-activity condition.
    //
    // BatchEarlyStop:
    //   Stop DAG construction when the condition is reached.
    //
    // BatchThenDirect:
    //   Switch the next outer iteration to direct insertion.
    //   The current iteration has already completed both
    //   levelization and finalization phases.
    //
    // after_both_phases is already copied for the original
    // u0 == 0 convergence condition, so no additional D2H
    // copy is introduced.
    // ==========================================================
    if (!iteration_uses_direct) {
      if (after_both_phases > peak_survived) {
        peak_survived = after_both_phases;
      }

      if (insertion_policy !=
          GpuDagInsertionPolicy::FullBatch) {
        const bool is_low_activity =
          after_both_phases > 0u &&
          static_cast<uint64_t>(
            after_both_phases) * 100u <=
          static_cast<uint64_t>(
            peak_survived) *
            _gpu_options.low_activity_percent;

        if (is_low_activity) {
          ++low_activity_streak;
        }
        else {
          low_activity_streak = 0u;
        }

        const bool low_activity_condition_reached =
          outer_iteration >=
            _gpu_options.low_activity_minimum_iterations &&
          low_activity_streak >=
            _gpu_options.low_activity_patience;

        if (low_activity_condition_reached) {
          if (insertion_policy ==
              GpuDagInsertionPolicy::BatchEarlyStop) {
            batch_early_stop_triggered = true;
            break;
          }

          // BatchThenDirect:
          //
          // Switch only after one complete batch iteration.
          // The next outer iteration begins in direct mode.
          use_direct_mode = true;
          switched_to_direct = true;
          switch_iteration = outer_iteration;
          switch_survived = after_both_phases;

          if constexpr (kEnableGpuDagIterationLog) {
            printf(
              "[GPU DAG Controller] "
              "switch batch->direct after iteration=%u, "
              "survived=%u, peak=%u, "
              "low_activity_streak=%u\n",
              switch_iteration,
              switch_survived,
              peak_survived,
              low_activity_streak);
          }
        }
      }
    }
  }

  
  // ============================================================
  // End of DAG construction core
  //
  // Record the core/filter boundary only when detailed logging
  // is enabled.
  // ============================================================
  if constexpr (kEnableGpuDagIterationLog) {
    CUDA_CHECK(cudaEventRecord(after_core_gpu));
  }

  // Match the CPU post-processing semantics:
  //
  // Keep _dvc_picked_endes unchanged and separately mark selected
  // e-nodes lying on a selected root-to-global-leaf path.
  _gpu_filter_root_to_leaf();

  // ============================================================
  // Stop total DAG timing
  // ============================================================
  CUDA_CHECK(cudaEventRecord(stop_gpu));
  CUDA_CHECK(cudaEventSynchronize(stop_gpu));

  // Total runtime is always calculated.
  CUDA_CHECK(cudaEventElapsedTime(
    &gpu_dag_total_time_ms,
    start_gpu,
    stop_gpu));

  _gpu_dag_ms = gpu_dag_total_time_ms;
  // Detailed core/filter timing is calculated only when enabled.
  if constexpr (kEnableGpuDagIterationLog) {
    CUDA_CHECK(cudaEventElapsedTime(
      &gpu_dag_core_time_ms,
      start_gpu,
      after_core_gpu));

    CUDA_CHECK(cudaEventElapsedTime(
      &gpu_filter_time_ms,
      after_core_gpu,
      stop_gpu));

    printf(
      "GPU DAG core runtime: %.2f (ms)\n",
      gpu_dag_core_time_ms);

    printf(
      "GPU root-to-leaf filtering runtime: %.2f (ms)\n",
      gpu_filter_time_ms);
  }

  const uint32_t low_activity_threshold =
    static_cast<uint32_t>(
      (static_cast<uint64_t>(
         peak_survived) *
       _gpu_options.low_activity_percent) /
      100u);

  const char* insertion_policy_name = "unknown";

  switch (insertion_policy) {
    case GpuDagInsertionPolicy::FullBatch:
      insertion_policy_name = "full-batch";
      break;

    case GpuDagInsertionPolicy::BatchEarlyStop:
      insertion_policy_name = "batch-early-stop";
      break;

    case GpuDagInsertionPolicy::BatchThenDirect:
      insertion_policy_name = "batch-then-direct";
      break;
  }

  const char* termination_reason =
    "converged";

  if (batch_early_stop_triggered) {
    termination_reason =
      "low-activity-stop";
  }
  else if (bitmap_direct_limit_triggered) {
    termination_reason =
      "bitmap-direct-limit";
  }

  printf(
    "[GPU DAG Outer Loop] "
    "policy=%s, "
    "iterations=%u, "
    "batch_iterations=%u, "
    "direct_iterations=%u, "
    "reason=%s, "
    "peak=%u, "
    "last=%u, "
    "low_activity_threshold=%u, "
    "low_activity_streak=%u, "
    "switched=%s, "
    "switch_iteration=%u, "
    "switch_survived=%u\n",
    insertion_policy_name,
    outer_iteration,
    batch_iterations,
    direct_iterations,
    termination_reason,
    peak_survived,
    host_u0,
    low_activity_threshold,
    low_activity_streak,
    switched_to_direct
      ? "yes"
      : "no",
    switch_iteration,
    switch_survived);

  // Total runtime is always printed.
  printf(
    "GPU DAG total runtime: %.2f (ms)\n",
    gpu_dag_total_time_ms);

  // Events are released on normal return and on exceptions.
}
