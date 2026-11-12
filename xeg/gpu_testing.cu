#include "gpu_common.cuh"

XEG::CsrGraphSnapshot
XEG::unittest_get_csr_snapshot() const
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_get_csr_snapshot requires "
      "read() to be called first");
  }

  if (_global_adj_1D_ptr.size()
        != static_cast<size_t>(_num_ende) + 1 ||
      _global_inv_adj_1D_ptr.size()
        != static_cast<size_t>(_num_ende) + 1) {
    throw std::logic_error(
      "CSR pointer arrays are not initialized");
  }

  CsrGraphSnapshot result;

  result.forward_adj =
    _global_adj_1D;

  result.forward_ptr =
    _global_adj_1D_ptr;

  result.inverse_adj =
    _global_inv_adj_1D;

  result.inverse_ptr =
    _global_inv_adj_1D_ptr;

  return result;
}

XEG::GpuInsertionResult
XEG::unittest_gpu_insert_batch_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_insert_batch_once requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  if (input_level.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "level size does not match "
      "the number of e-nodes");
  }

  for (uint32_t ende = 0;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }
  }

  // GPU tree construction currently owns allocation of
  // the shared immutable CSR and working buffers.
  if (_dvc_picked_endes == nullptr) {
    _gpu_allocate_graph_buffers();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
      _dvc_level,
      input_level.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
      _dvc_dag_u0,
      0,
      sizeof(uint32_t)));

    _gpu_dag_insert_batch(
      is_top_down_insertion);

    GpuInsertionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

XEG::GpuInsertionResult
XEG::unittest_gpu_insert_direct_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_insert_direct_once requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  if (input_level.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "level size does not match "
      "the number of e-nodes");
  }

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }
  }

  // GPU tree construction owns allocation of the shared
  // immutable CSR and working buffers.
  if (_dvc_picked_endes == nullptr) {
    _gpu_allocate_graph_buffers();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
      _dvc_level,
      input_level.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
      _dvc_dag_u0,
      0,
      sizeof(uint32_t)));

    _gpu_dag_insert_direct(
      is_top_down_insertion);

    GpuInsertionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

XEG::GpuInsertionResult
XEG::_unittest_gpu_insert_warp_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion,
  const bool use_direct_mode,
  const uint32_t threads_per_block,
  const uint32_t bitmap_bits)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "_unittest_gpu_insert_warp_once requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  if (input_level.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "level size does not match "
      "the number of e-nodes");
  }

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }
  }

  // Validate before allocating or launching GPU work.
  _validate_warp_threads_per_block(
    threads_per_block);

  _validate_warp_bitmap_bits(
    bitmap_bits);

  if (_dvc_picked_endes == nullptr) {
    _gpu_allocate_graph_buffers();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
      _dvc_level,
      input_level.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
      _dvc_dag_u0,
      0,
      sizeof(uint32_t)));

    // Exercise the same dispatcher used by production.
    _dag_kernel =
      DagKernel::Warp;

    // In Warp mode this value means total CUDA threads/block.
    _dag_threads_per_candidate =
      threads_per_block;

    // Reset the complete runtime bitmap configuration on every
    // unit-test call.
    //
    // This prevents a previous Bitmap-32 or Bitmap-64 call from
    // leaking configuration into a later Exact Warp call that
    // reuses the same XEG object.
    _dag_bitmap_bits =
      bitmap_bits;

    _dag_bitmap_mask =
      bitmap_bits == 0u
        ? 0u
        : bitmap_bits - 1u;

    _dag_bitmap_words =
      bitmap_bits / 32u;

    _gpu_dag_insert_configured(
      use_direct_mode,
      is_top_down_insertion);

    GpuInsertionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

XEG::GpuInsertionResult
XEG::unittest_gpu_insert_warp_batch_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion,
  const uint32_t threads_per_block,
  const uint32_t bitmap_bits)
{
  return _unittest_gpu_insert_warp_once(
    selected_state,
    input_level,
    is_top_down_insertion,
    false,
    threads_per_block,
    bitmap_bits);
}

XEG::GpuInsertionResult
XEG::unittest_gpu_insert_warp_direct_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion,
  const uint32_t threads_per_block,
  const uint32_t bitmap_bits)
{
  return _unittest_gpu_insert_warp_once(
    selected_state,
    input_level,
    is_top_down_insertion,
    true,
    threads_per_block,
    bitmap_bits);
}

XEG::GpuInsertionResult
XEG::_unittest_gpu_insert_block_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion,
  const bool use_direct_mode,
  const uint32_t threads_per_candidate,
  const uint32_t bitmap_bits)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "_unittest_gpu_insert_block_once requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  if (input_level.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "level size does not match "
      "the number of e-nodes");
  }

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }
  }

  // Validate before allocating or launching GPU work.
  _validate_block_threads_per_candidate(
    threads_per_candidate);

  _validate_block_bitmap_bits(
    bitmap_bits);

  // GPU tree construction owns the shared immutable CSR and
  // working buffers.
  if (_dvc_picked_endes == nullptr) {
    _gpu_allocate_graph_buffers();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
      _dvc_level,
      input_level.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
      _dvc_dag_u0,
      0,
      sizeof(uint32_t)));

    // Exercise the same dispatcher used by the production
    // controller.
    _dag_kernel =
      DagKernel::Block;

    _dag_threads_per_candidate =
      threads_per_candidate;

    _dag_bitmap_bits =
      bitmap_bits;

    _dag_bitmap_mask =
      bitmap_bits == 0u
        ? 0u
        : bitmap_bits - 1u;

    _dag_bitmap_words =
      bitmap_bits / 32u;

    _gpu_dag_insert_configured(
      use_direct_mode,
      is_top_down_insertion);

    GpuInsertionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

XEG::GpuInsertionResult
XEG::unittest_gpu_insert_block_batch_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion,
  const uint32_t threads_per_candidate,
  const uint32_t bitmap_bits)
{
  return _unittest_gpu_insert_block_once(
    selected_state,
    input_level,
    is_top_down_insertion,
    false,
    threads_per_candidate,
    bitmap_bits);
}

XEG::GpuInsertionResult
XEG::unittest_gpu_insert_block_direct_once(
  const std::vector<uint32_t>& selected_state,
  const std::vector<int>& input_level,
  const bool is_top_down_insertion,
  const uint32_t threads_per_candidate,
  const uint32_t bitmap_bits)
{
  return _unittest_gpu_insert_block_once(
    selected_state,
    input_level,
    is_top_down_insertion,
    true,
    threads_per_candidate,
    bitmap_bits);
}

XEG::GpuFinalizeResult
XEG::unittest_gpu_levelize_and_finalize(
  const std::vector<uint32_t>& selected_state,
  const bool is_top_down_levelization,
  const uint32_t initial_u0)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_levelize_and_finalize "
      "requires read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  uint32_t tentative_count = 0u;

  for (uint32_t ende = 0;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }

    if (selected_state[ende] == 2u) {
      ++tentative_count;
    }
  }

  // Every rolled-back tentative node decrements u0 once.
  // Therefore initial_u0 must be large enough to represent
  // all currently tentative nodes.
  if (initial_u0 < tentative_count) {
    throw std::invalid_argument(
      "initial_u0 must be at least the number "
      "of tentative state-2 nodes");
  }

  // GPU tree construction currently owns allocation of
  // the immutable CSR and shared working buffers.
  if (_dvc_picked_endes == nullptr) {
    _gpu_allocate_graph_buffers();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
      _dvc_dag_u0,
      &initial_u0,
      sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    _gpu_dag_levelize(
      is_top_down_levelization);

    _gpu_dag_finalize_tentative(
      is_top_down_levelization);

    GpuFinalizeResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.residual_degree.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    const uint32_t* degree_device =
      is_top_down_levelization
        ? _dvc_indegree
        : _dvc_inv_indegree;

    CUDA_CHECK(cudaMemcpy(
      result.residual_degree.data(),
      degree_device,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.num_levels,
      _dvc_max_level,
      sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

XEG::GpuDagConstructionResult
XEG::unittest_gpu_construct_dag(
  const std::vector<uint32_t>& selected_state)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_construct_dag requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  // Full DAG construction starts from the stable tree
  // produced by Algorithm 1. State 2 must not be present
  // before the first insertion phase.
  for (uint32_t ende = 0;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 1u) {
      throw std::invalid_argument(
        "initial DAG construction state must "
        "contain only 0 or 1");
    }
  }

  // GPU tree construction currently owns allocation of
  // immutable CSR and shared GPU working buffers.
  if (_dvc_picked_endes == nullptr) {
    _gpu_allocate_graph_buffers();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    // Preserve the original full-batch, full-convergence
    // behavior expected by the existing DAG correctness tests.
    _gpu_DAG_construction(
      GpuDagInsertionPolicy::FullBatch);

    GpuDagConstructionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.residual_degree.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    result.in_subgraph.resize(
      _num_ende,
      0u);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    // _gpu_DAG_construction() finishes with top-down
    // levelization, so _dvc_indegree contains the final
    // top-down residual degree.
    CUDA_CHECK(cudaMemcpy(
      result.residual_degree.data(),
      _dvc_indegree,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.in_subgraph.data(),
      _dvc_in_subgraph,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.num_levels,
      _dvc_max_level,
      sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.final_u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

XEG::GpuDagConstructionResult
XEG::unittest_gpu_construct_dag_from_tree_with_policy(
  const GpuDagInsertionPolicy insertion_policy)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_construct_dag_from_tree_with_policy "
      "requires read() to be called first");
  }

  // Reject invalid enum values before entering the production
  // controller. This validation exists only in the unit-test
  // wrapper and does not change the production path.
  switch (insertion_policy) {
    case GpuDagInsertionPolicy::FullBatch:
    case GpuDagInsertionPolicy::BatchEarlyStop:
    case GpuDagInsertionPolicy::BatchThenDirect:
      break;

    default:
      throw std::invalid_argument(
        "Invalid GPU DAG insertion policy");
  }

  // This wrapper must start from an unmodified GPU tree.
  //
  // Reusing an object after another GPU unit-test operation
  // could leave _dvc_picked_endes containing a modified DAG
  // rather than the original tree.
  if (_dvc_picked_endes != nullptr) {
    throw std::logic_error(
      "unittest_gpu_construct_dag_from_tree_with_policy "
      "requires a fresh XEG object");
  }

  // Build the same GPU tree used by the production run().
  //
  // _gpu_initialize_dag_buffers() intentionally does not
  // modify _dvc_picked_endes, so the tree selection remains
  // the initial state of DAG construction.
  (void)_gpu_tree_construction();

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    // Execute the real production controller with the policy
    // explicitly selected by the unit test.
    _gpu_DAG_construction(
      insertion_policy);

    GpuDagConstructionResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.residual_degree.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    result.in_subgraph.resize(
      _num_ende,
      0u);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    // _gpu_DAG_construction() finishes with top-down
    // levelization, so _dvc_indegree contains the final
    // top-down residual degree.
    CUDA_CHECK(cudaMemcpy(
      result.residual_degree.data(),
      _dvc_indegree,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.in_subgraph.data(),
      _dvc_in_subgraph,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.num_levels,
      _dvc_max_level,
      sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.final_u0,
      _dvc_dag_u0,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}

XEG::GpuRootToLeafFilterResult
XEG::unittest_gpu_filter_root_to_leaf(
  const std::vector<uint32_t>& selected_state)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "unittest_gpu_filter_root_to_leaf requires "
      "read() to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  // Root-to-leaf filtering runs after DAG construction has
  // finalized every tentative node. Therefore its input must
  // contain only stable states 0 and 1.
  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 1u) {
      throw std::invalid_argument(
        "root-to-leaf filtering input must "
        "contain only states 0 or 1");
    }
  }

  // GPU tree construction owns the immutable CSR, roots,
  // leaves, queue, and filtering buffers.
  if (_dvc_picked_endes == nullptr) {
    _gpu_allocate_graph_buffers();
  }

  if (_num_ende > 0u) {
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  }

  _gpu_filter_root_to_leaf();

  GpuRootToLeafFilterResult result;

  result.selected_state.resize(
    _num_ende,
    0u);

  result.from_roots.resize(
    _num_ende,
    0u);

  result.from_leaves.resize(
    _num_ende,
    0u);

  result.in_subgraph.resize(
    _num_ende,
    0u);

  if (_num_ende > 0u) {
    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.from_roots.data(),
      _dvc_from_roots,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.from_leaves.data(),
      _dvc_from_leaves,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.in_subgraph.data(),
      _dvc_in_subgraph,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  }

  return result;
}

XEG::GpuLevelizationResult
XEG::unittest_gpu_levelize(
  const std::vector<uint32_t>& selected_state,
  const bool is_top_down)
{
  // Preserve the existing unit-test API and behavior.
  //
  // This follows the same compile-time production dispatcher
  // selected by kEnableGpuDagHybridLevelization.
  return _unittest_gpu_levelize_impl(
    selected_state,
    is_top_down,
    GpuLevelizationBackend::Configured);
}

XEG::GpuLevelizationResult
XEG::unittest_gpu_levelize_with_backend(
  const std::vector<uint32_t>& selected_state,
  const bool is_top_down,
  const XEG::GpuLevelizationBackend backend)
{
  return _unittest_gpu_levelize_impl(
    selected_state,
    is_top_down,
    backend);
}

XEG::GpuLevelizationResult
XEG::_unittest_gpu_levelize_impl(
  const std::vector<uint32_t>& selected_state,
  const bool is_top_down,
  const XEG::GpuLevelizationBackend backend)
{
  if (_ende_name.empty()) {
    throw std::logic_error(
      "_unittest_gpu_levelize_impl requires read() "
      "to be called first");
  }

  if (selected_state.size() !=
      static_cast<size_t>(_num_ende)) {
    throw std::invalid_argument(
      "selected_state size does not match "
      "the number of e-nodes");
  }

  for (uint32_t ende = 0u;
       ende < _num_ende;
       ++ende) {
    if (selected_state[ende] > 2u) {
      throw std::invalid_argument(
        "selected_state must contain only "
        "0, 1, or 2");
    }
  }

  // GPU tree construction currently owns allocation of the
  // immutable CSR and shared GPU working buffers.
  //
  // This is executed only on the first unit-test call for this
  // XEG object. Later backend comparisons reuse the exact same
  // graph and shared buffers.
  if (_dvc_picked_endes == nullptr) {
    _gpu_allocate_graph_buffers();
  }

  _gpu_allocate_dag_buffers();
  _gpu_initialize_dag_buffers();

  try {
    // Every backend receives the exact selected-state mask
    // supplied by the unit test.
    CUDA_CHECK(cudaMemcpy(
      _dvc_picked_endes,
      selected_state.data(),
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyHostToDevice));

    // ----------------------------------------------------------
    // Persistent-path sentinel
    //
    // _gpu_initialize_dag_buffers() normally initializes this
    // buffer to zero. That is not sufficient for detecting use
    // because Completed is enum value zero.
    //
    // Therefore, write values that cannot be produced by a valid
    // persistent-kernel return. If the persistent kernel runs,
    // it overwrites all four fields before returning.
    //
    // This extra copy occurs only inside this unit-test helper.
    // ----------------------------------------------------------
    const xeg_detail::GpuDagPersistentState
      persistent_sentinel{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<int32_t>::min(),
        xeg_detail::GpuDagPersistentExitReason::
          InvalidQueueState
      };

    CUDA_CHECK(cudaMemcpy(
      _dvc_dag_persistent_state,
      &persistent_sentinel,
      sizeof(persistent_sentinel),
      cudaMemcpyHostToDevice));

    // Direct backend selection is restricted to this unit-test
    // helper. The production dispatcher remains unchanged.
    switch (backend) {
      case GpuLevelizationBackend::Configured:
        _gpu_dag_levelize(is_top_down);
        break;

      case GpuLevelizationBackend::HostDriven:
        _gpu_dag_levelize_host_driven(is_top_down);
        break;

      case GpuLevelizationBackend::Hybrid:
        _gpu_dag_levelize_hybrid(is_top_down);
        break;

      default:
        throw std::invalid_argument(
          "Invalid GPU levelization backend");
    }

    GpuLevelizationResult result;

    result.selected_state.resize(
      _num_ende,
      0u);

    result.residual_degree.resize(
      _num_ende,
      0u);

    result.level.resize(
      _num_ende,
      -1);

    CUDA_CHECK(cudaMemcpy(
      result.selected_state.data(),
      _dvc_picked_endes,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    const uint32_t* degree_device =
      is_top_down
        ? _dvc_indegree
        : _dvc_inv_indegree;

    CUDA_CHECK(cudaMemcpy(
      result.residual_degree.data(),
      degree_device,
      static_cast<size_t>(_num_ende) *
        sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      result.level.data(),
      _dvc_level,
      static_cast<size_t>(_num_ende) *
        sizeof(int),
      cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
      &result.num_levels,
      _dvc_max_level,
      sizeof(int),
      cudaMemcpyDeviceToHost));

    // ----------------------------------------------------------
    // Detect whether the persistent kernel was actually used.
    //
    // Host-driven levelization never touches this buffer.
    // Hybrid levelization touches it only after entering the
    // persistent path.
    // ----------------------------------------------------------
    xeg_detail::GpuDagPersistentState
      observed_persistent_state{};

    CUDA_CHECK(cudaMemcpy(
      &observed_persistent_state,
      _dvc_dag_persistent_state,
      sizeof(observed_persistent_state),
      cudaMemcpyDeviceToHost));

    const bool persistent_state_is_sentinel =
      observed_persistent_state.queue_begin ==
        persistent_sentinel.queue_begin &&
      observed_persistent_state.queue_count ==
        persistent_sentinel.queue_count &&
      observed_persistent_state.current_level ==
        persistent_sentinel.current_level &&
      observed_persistent_state.exit_reason ==
        persistent_sentinel.exit_reason;

    result.persistent_path_used =
      !persistent_state_is_sentinel;

    if (result.persistent_path_used) {
      result.persistent_completed =
        observed_persistent_state.exit_reason ==
          xeg_detail::GpuDagPersistentExitReason::
            Completed;

      result.persistent_exited_to_host =
        observed_persistent_state.exit_reason ==
          xeg_detail::GpuDagPersistentExitReason::
            FrontierTooWide;
    }

    _gpu_free_dag_buffers();

    return result;
  }
  catch (...) {
    _gpu_free_dag_buffers();
    throw;
  }
}


