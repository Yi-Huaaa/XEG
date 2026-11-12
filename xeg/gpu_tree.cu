#include "gpu_common.cuh"

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

__global__ void _gpu_tree_construction_kernel(const uint32_t num_endes,
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


std::pair<double, double> XEG::_gpu_tree_construction()
{
  int blocks = (_num_ende + kGpuThreads - 1) / kGpuThreads;

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


  _gpu_warm_up<<<blocks,kGpuThreads>>>(_num_ende, _dvc_level); // Note: place this GPU kernel here as the warm-up for GPU; w/o this, the kernel runtime below will be extremely long. Need to remrmber!

  float gpuTime = 0.;
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);    
  cudaEventRecord(startGPU);

  blocks = (_num_ende + kGpuThreads - 1) / kGpuThreads;
  while (h_update) {
    h_update = false;
    cudaMemcpy(_dvc_update, &h_update, sizeof(bool), cudaMemcpyHostToDevice);
    _gpu_tree_construction_kernel<<<blocks, kGpuThreads>>>(_num_ende,
      dvc_ende_cost,
      dvc_ende_ecls,
      dvc_ende_child,
      dvc_ende_child_ptr,
      _dvc_update,
      dvc_cost_of_ecls,
      dvc_cost_of_ende,
      _dvc_best_ende_of_ecls);
    cudaCheckErrors("_gpu_tree_construction_kernel failed");
    cudaMemcpy(&h_update, _dvc_update, sizeof(bool), cudaMemcpyDeviceToHost);
  }
  
  // Get the best tree cost on GPU
  _get_tree_cost<<<1, 1>>>(_root_eclass_id.size(), dvc_cost_of_ecls, _dvc_root_eclass_id, dvc_tree_ret);

  // Get best tree cost 
  double tree_ret = 0.0; 
  cudaMemcpy(&tree_ret, dvc_tree_ret, 1*sizeof(double), cudaMemcpyDeviceToHost);
  
  blocks = (_num_ecls + kGpuThreads - 1) / kGpuThreads;
  _resolve_best_ende<<<blocks, kGpuThreads>>>(
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
    blocks = (num + kGpuThreads - 1) / kGpuThreads;
    _get_dag_cost<<<blocks, kGpuThreads>>>(
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


