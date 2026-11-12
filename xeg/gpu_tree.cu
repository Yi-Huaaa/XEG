#include "gpu_common.cuh"
#include <cstring>

// =========================== GPU functions =========================== //


// Preserve one GPU thread per e-node and asynchronous cost relaxation.
// The per-class gate couples a strict cost improvement to its witness. Without
// it, two successful cost updates can publish their representatives in reverse
// order. Nonnegative costs plus strict improvement give acyclic witnesses even
// when zero costs or floating-point rounding make dependency costs tie.
__global__ void _gpu_tree_construction_kernel(
  uint32_t num_nodes, const float* costs, const uint32_t* node_class,
  const uint32_t* children, const uint32_t* child_ptr,
  uint32_t* update, double* class_cost, uint32_t* best, uint32_t* locks)
{
  const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= num_nodes) return;
  const uint32_t c = node_class[u];
  cuda::atomic_ref<double, cuda::thread_scope_device> own_cost(class_cost[c]);
  const double bound = own_cost.load(cuda::memory_order_relaxed);
  double candidate = costs[u];
  for (uint32_t k = child_ptr[u]; k < child_ptr[u + 1] && candidate < bound; ++k) {
    cuda::atomic_ref<double, cuda::thread_scope_device> child_cost(class_cost[children[k]]);
    candidate += child_cost.load(cuda::memory_order_relaxed);
  }
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> gate(locks[c]);
  while (candidate < own_cost.load(cuda::memory_order_relaxed)) {
    uint32_t unlocked = 0;
    if (gate.compare_exchange_strong(unlocked, 1u, cuda::memory_order_acquire,
                                     cuda::memory_order_relaxed)) {
      if (candidate < own_cost.load(cuda::memory_order_relaxed)) {
        best[c] = u;
        own_cost.store(candidate, cuda::memory_order_relaxed);
        atomicExch(update, 1u);
      }
      // Release within this branch before any lane retries: no warp can wait
      // for a lane that is holding the gate outside the retry loop.
      gate.store(0u, cuda::memory_order_release);
      break;
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
  uint32_t* picked_endes)
{
  uint32_t pos = 0;
  for (uint32_t i = 0; i < sz; i++) {
    uint32_t ecls = root_eclass_id[i];
    uint32_t best_ende = best_ende_of_ecls[ecls];
    if (picked_endes[best_ende]) continue;
    picked_endes[best_ende] = 1u;
    q[pos] = best_ende;
    pos++;
  }
  *q_sz = pos;
}

__global__ void _get_dag_cost(
  const uint32_t num,
  const float* ende_cost,
  const uint32_t* ende_child,
  const uint32_t* ende_child_ptr,
  const uint32_t* best_ende_of_ecls,
  uint32_t* q,
  uint32_t* q_sz,
  const uint32_t q_sp,
  uint32_t* picked_endes, // No bool atomicCAS
  double* dag_ret)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num) return;
  uint32_t cur = q[q_sp + t_idx];
  atomicAdd(dag_ret, static_cast<double>(ende_cost[cur]));

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

__global__ void _gpu_warm_up(
  const uint32_t num_ende,
  int* dvc_level)
{
  uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_idx >= num_ende) return;
  dvc_level[t_idx] = -1;
}


std::pair<double, double> XEG::_gpu_tree_construction()
{
  int blocks = (_num_ende + kGpuThreads - 1) / kGpuThreads;

  for (float cost : _ende_cost)
    if (!std::isfinite(cost) || cost < 0)
      throw std::invalid_argument("GPU extraction requires finite nonnegative costs");
  if (_num_ende == 0) throw std::invalid_argument("GPU extraction requires a nonempty graph");
  _gpu_allocate_graph_buffers();
  float* dvc_ende_cost = nullptr;
  uint32_t* dvc_ende_class = nullptr;
  uint32_t* dvc_ende_child = nullptr;
  uint32_t* dvc_ende_child_ptr = nullptr;
  double* dvc_cost_of_ecls = nullptr;
  double* dvc_tree_ret = nullptr;
  double* dvc_dag_ret = nullptr;
  auto release_local = gpu_scope_exit([&] {
    cuda_free_and_null(dvc_ende_cost);
    cuda_free_and_null(dvc_ende_class);
    cuda_free_and_null(dvc_ende_child);
    cuda_free_and_null(dvc_ende_child_ptr);
    cuda_free_and_null(dvc_cost_of_ecls);
    cuda_free_and_null(dvc_tree_ret);
    cuda_free_and_null(dvc_dag_ret);
  });

  // GPU relaxation
  uint32_t h_update = 1u;
  std::vector<double> cost_of_ecls;
  cost_of_ecls.resize(_num_ecls, std::numeric_limits<double>::infinity());
  cudaMalloc(&dvc_ende_cost, _ende_cost.size()*sizeof(float));
  cudaMalloc(&dvc_ende_class, _ende_ecls.size()*sizeof(uint32_t));
  cudaMalloc(&dvc_ende_child, _ende_child.size()*sizeof(uint32_t));
  cudaMalloc(&dvc_ende_child_ptr, _ende_child_ptr.size()*sizeof(uint32_t));
  cudaMalloc(&dvc_cost_of_ecls, cost_of_ecls.size()*sizeof(double));
  cudaMalloc(&dvc_tree_ret, 1*sizeof(double));
  cudaMalloc(&dvc_dag_ret, 1*sizeof(double));
  cudaMemcpy(dvc_ende_cost, _ende_cost.data(), _ende_cost.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_ende_class, _ende_ecls.data(), _ende_ecls.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_ende_child, _ende_child.data(), _ende_child.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_ende_child_ptr, _ende_child_ptr.data(), _ende_child_ptr.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dvc_cost_of_ecls, cost_of_ecls.data(), cost_of_ecls.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(dvc_tree_ret, 0, 1*sizeof(double));
  cudaMemset(dvc_dag_ret, 0, 1*sizeof(double));
  cudaCheckErrors("GPU tree cost buffer initialization failed");

  _gpu_warm_up<<<blocks,kGpuThreads>>>(_num_ende, _dvc_level); // Note: place this GPU kernel here as the warm-up for GPU; w/o this, the kernel runtime below will be extremely long. Need to remrmber!

  float gpuTime = 0.;
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);
  auto release_events = gpu_scope_exit([&] { cudaEventDestroy(startGPU); cudaEventDestroy(stopGPU); });
  cudaEventRecord(startGPU);

  blocks = (_num_ende + kGpuThreads - 1) / kGpuThreads;
  while (h_update) {
    h_update = 0u;
    CUDA_CHECK(cudaMemsetAsync(_dvc_update, 0, sizeof(h_update)));
    _gpu_tree_construction_kernel<<<blocks, kGpuThreads>>>(
      _num_ende, dvc_ende_cost, dvc_ende_class, dvc_ende_child, dvc_ende_child_ptr,
      _dvc_update, dvc_cost_of_ecls, _dvc_best_ende_of_ecls, _dvc_tree_locks);
    cudaCheckErrors("_gpu_tree_construction_kernel failed");
    CUDA_CHECK(_gpu_copy_control(&h_update, _dvc_update, sizeof(h_update)));
  }

  // Get the best tree cost on GPU
  _get_tree_cost<<<1, 1>>>(_root_eclass_id.size(), dvc_cost_of_ecls, _dvc_root_eclass_id, dvc_tree_ret);

  // Get best tree cost
  double tree_ret = 0.0;
  _gpu_copy_control(&tree_ret, dvc_tree_ret, 1*sizeof(double));

  if (!std::isfinite(tree_ret))
    throw std::runtime_error("GPU tree has no finite witness for every root");

  // Get the best dag cost on GPU
  _init_q<<<1,1>>>(_root_eclass_id.size(),
    _dvc_root_eclass_id,
    _dvc_best_ende_of_ecls,
    _dvc_q, _dvc_q_sz,
    _dvc_picked_endes);
  cudaCheckErrors("_init_q failed");

  double dag_ret = 0.0;

  uint32_t q_sz = 0;
  _gpu_copy_control(&q_sz, _dvc_q_sz, 1*sizeof(uint32_t));
  uint32_t q_sp = 0;
  uint32_t num = q_sz - q_sp;

  while (num > 0) {
    blocks = (num + kGpuThreads - 1) / kGpuThreads;
    _get_dag_cost<<<blocks, kGpuThreads>>>(
      num,
      dvc_ende_cost,
      dvc_ende_child,
      dvc_ende_child_ptr,
      _dvc_best_ende_of_ecls,
      _dvc_q,
      _dvc_q_sz,
      q_sp,
      _dvc_picked_endes,
      dvc_dag_ret);
    cudaCheckErrors("_get_dag_cost failed");
    _gpu_copy_control(&q_sz, _dvc_q_sz, sizeof(uint32_t));
    q_sp += num;
    num = q_sz - q_sp;
  }
  _gpu_copy_control(&dag_ret, dvc_dag_ret, sizeof(double));
  cudaCheckErrors("DAG cost memcpy failed");

  cudaEventRecord(stopGPU);
  cudaEventSynchronize(stopGPU);
  cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);
  _gpu_tree_ms = gpuTime;
  printf("GPU tree construction runtime: %.2f (ms)\n", gpuTime);

  return std::make_pair(tree_ret, dag_ret);
}

void XEG::_copy_pickedende_back(
  std::vector<bool>& picked_endes_out)
{
  if (_dvc_picked_endes == nullptr) {
    throw std::logic_error(
      "_copy_pickedende_back called before GPU tree construction");
  }

  std::vector<uint32_t> host_selected(_num_ende, 0);

  cudaError_t err = cudaMemcpy(
    host_selected.data(),
    _dvc_picked_endes,
    static_cast<size_t>(_num_ende) * sizeof(uint32_t),
    cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    throw std::runtime_error(
      std::string("Failed to copy selected e-nodes: ") +
      cudaGetErrorString(err));
  }

  picked_endes_out.assign(_num_ende, false);

  for (uint32_t ende = 0; ende < _num_ende; ++ende) {
    picked_endes_out[ende] = (host_selected[ende] != 0);
  }
}

void XEG::_copy_best_ende_of_ecls_back(
  std::vector<uint32_t>& best_ende_of_ecls_out)
{
  if (_dvc_best_ende_of_ecls == nullptr) {
    throw std::logic_error(
      "_copy_best_ende_of_ecls_back called before GPU tree construction");
  }

  best_ende_of_ecls_out.resize(_num_ecls);

  cudaError_t err = cudaMemcpy(
    best_ende_of_ecls_out.data(),
    _dvc_best_ende_of_ecls,
    static_cast<size_t>(_num_ecls) * sizeof(uint32_t),
    cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    throw std::runtime_error(
      std::string("Failed to copy best e-nodes: ") +
      cudaGetErrorString(err));
  }
}

// =========================== ILP Solver functions =========================== //



XEG::~XEG() { _gpu_free_buffers(); }

void XEG::_gpu_free_buffers() noexcept {
  if (_gpu_control) { cudaFreeHost(_gpu_control); _gpu_control = nullptr; }
  cuda_free_and_null(_dvc_best_ende_of_ecls);
  cuda_free_and_null(_dvc_update);
  cuda_free_and_null(_dvc_tree_locks);
  cuda_free_and_null(_dvc_root_eclass_id);
  cuda_free_and_null(_dvc_picked_endes);
  cuda_free_and_null(_dvc_q);
  cuda_free_and_null(_dvc_q_sz);
  cuda_free_and_null(_dvc_global_adj_1D);
  cuda_free_and_null(_dvc_global_adj_1D_ptr);
  cuda_free_and_null(_dvc_global_inv_adj_1D);
  cuda_free_and_null(_dvc_global_inv_adj_1D_ptr);
  cuda_free_and_null(_dvc_indegree);
  cuda_free_and_null(_dvc_inv_indegree);
  cuda_free_and_null(_dvc_level);
  cuda_free_and_null(_dvc_max_level);
  cuda_free_and_null(_dvc_in_subgraph);
  cuda_free_and_null(_dvc_roots);
  cuda_free_and_null(_dvc_leaves);
  cuda_free_and_null(_dvc_from_roots);
  cuda_free_and_null(_dvc_from_leaves);
  cuda_free_and_null(_dvc_dag_new_inserted);
  cuda_free_and_null(_dvc_dag_insert_decision);
  cuda_free_and_null(_dvc_dag_candidate_level);
  cuda_free_and_null(_dvc_dag_u0);
  cuda_free_and_null(_dvc_dag_persistent_state);
}

void XEG::_gpu_allocate_graph_buffers() {
  _gpu_free_buffers();
  CUDA_CHECK(cudaMallocHost(&_gpu_control, 32));
  CUDA_CHECK(cudaMalloc(&_dvc_best_ende_of_ecls, _num_ecls*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_update, 1*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_tree_locks, _num_ecls*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_root_eclass_id, _root_eclass_id.size()*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_picked_endes, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_q, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_q_sz, 1*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_indegree, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_inv_indegree, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_global_adj_1D, _global_adj_1D.size()*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_global_adj_1D_ptr, _global_adj_1D_ptr.size()*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_global_inv_adj_1D, _global_inv_adj_1D.size()*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_global_inv_adj_1D_ptr, _global_inv_adj_1D_ptr.size()*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_level, _num_ende*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&_dvc_max_level, 1*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&_dvc_in_subgraph, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_roots, _roots.size()*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_leaves, _leaves.size()*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_from_roots, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&_dvc_from_leaves, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(_dvc_best_ende_of_ecls, 0xff, _num_ecls*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(_dvc_update, 0, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(_dvc_tree_locks, 0, _num_ecls*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(_dvc_root_eclass_id, _root_eclass_id.data(), _root_eclass_id.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(_dvc_picked_endes, 0, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(_dvc_q, 0, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(_dvc_q_sz, 0, 1*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(_dvc_indegree, 0, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(_dvc_inv_indegree, 0, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(_dvc_global_adj_1D, _global_adj_1D.data(), _global_adj_1D.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(_dvc_global_adj_1D_ptr, _global_adj_1D_ptr.data(), _global_adj_1D_ptr.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(_dvc_global_inv_adj_1D, _global_inv_adj_1D.data(), _global_inv_adj_1D.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(_dvc_global_inv_adj_1D_ptr, _global_inv_adj_1D_ptr.data(), _global_inv_adj_1D_ptr.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(_dvc_max_level, 0, 1*sizeof(int)));
  CUDA_CHECK(cudaMemset(_dvc_in_subgraph, 0, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(_dvc_roots, _roots.data(), _roots.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(_dvc_leaves, _leaves.data(), _leaves.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(_dvc_from_roots, 0, _num_ende*sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(_dvc_from_leaves, 0, _num_ende*sizeof(uint32_t)));
}

// Queue sizes, progress flags, and the persistent-state snapshot are small,
// synchronous control messages. Reuse registered host storage instead of
// staging a pageable stack destination at each traversal frontier.
cudaError_t XEG::_gpu_copy_control(void* destination, const void* source, size_t bytes) {
  if (!_gpu_control || bytes > 32) throw std::logic_error("Invalid GPU control copy");
  CUDA_CHECK(cudaMemcpy(_gpu_control, source, bytes, cudaMemcpyDeviceToHost));
  std::memcpy(destination, _gpu_control, bytes);
  return cudaSuccess;
}
