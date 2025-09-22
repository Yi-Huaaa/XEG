#ifndef EXT_GPU_H
#define EXT_GPU_H
#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <thread>
#include <queue>
#include <utility>
#include <atomic>
#include <chrono>
#include <set>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

__global__ void GPU_print_test(); 

class G_EXT {

public:
  void gpu_public_test() { gpu_private_test(); }
  void gpu_print();

private:
  void gpu_private_test() { printf("\nclass G_EXT: Hello world CUDA function!\n"); }


}; // class G_EXT  



#endif