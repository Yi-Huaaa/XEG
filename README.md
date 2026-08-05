# XEG

## Overview

XEG is an e-graph extraction system for equality saturation.
It takes a saturated e-graph with user-defined local costs and searches for a low-cost program under a DAG cost model.

The extraction flow has three main stages:

1. GPU tree construction builds an initial solution.
2. GPU DAG construction expands the search space while preserving acyclicity.
3. An integer solver searches the constructed space for a low-cost solution.

The project supports Google OR-Tools CP-SAT by default.
It also provides an optional NVIDIA cuOpt backend for users who want to try GPU based mixed integer optimization.

## Features

* GPU tree construction
* GPU DAG construction with Thread, Warp, and Block kernels
* Exact overlap detection and optional bitmap overlap detection
* Google OR-Tools CP-SAT backend
* Optional NVIDIA cuOpt backend
* CTest-based unit and integration tests

## Requirements

### Common Requirements

The following components are required for every build:

| Component | Purpose |
|---|---|
| Linux on x86_64 | Tested development platform |
| CMake 3.20 or newer | Build system |
| C++17 compiler | Host code |
| CUDA toolkit | GPU code |
| OpenMP | CPU parallel code |
| Protobuf | OR-Tools dependency |
| Google OR-Tools C++ | CP-SAT backend |

The project has been tested with CUDA 12.0.
The root `CMakeLists.txt` currently targets CUDA compute capability 8.6 with `sm_86`.
If your GPU uses a different compute capability, update both `CMAKE_CUDA_ARCHITECTURES` and the `-arch=sm_86` flag in the root `CMakeLists.txt`.

### OR-Tools and CP-SAT

OR-Tools is required for both build variants.
The cuOpt build still includes the CP-SAT backend.

The current CMake configuration expects OR-Tools C++ 9.7.2996 at `$HOME/or-tools_x86_64_Ubuntu-22.04_cpp_v9.7.2996`.

Download the OR-Tools C++ binary archive and extract it under your home directory so this path exists.

Useful links:

* [Google OR-Tools GitHub](https://github.com/google/or-tools)
* [OR-Tools C++ installation guide](https://developers.google.com/optimization/install/cpp)
* [OR-Tools 9.7 release](https://github.com/google/or-tools/releases/tag/v9.7)

### Optional NVIDIA cuOpt Backend

The cuOpt backend is optional.
Users who do not have cuOpt can still build and run XEG with CP-SAT.

This project uses the NVIDIA cuOpt C API package named `libcuopt`.
The current integration was tested with `libcuopt-cu12` version 26.6.0.

The NVIDIA cuOpt MILP solver is currently in beta and is under active development.
It may find good feasible solutions quickly, but it is not guaranteed to perform better than CP-SAT or cuOpt presolve for every problem.

Install `libcuopt` by following the official NVIDIA cuOpt C API instructions.
The exact install command may change between cuOpt releases.

Useful links:

* [NVIDIA cuOpt GitHub](https://github.com/NVIDIA/cuopt)
* [NVIDIA cuOpt installation guide](https://docs.nvidia.com/cuopt/user-guide/latest/install.html)
* [NVIDIA cuOpt C API quick start](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-c/quick-start.html)
* [NVIDIA cuOpt MILP features](https://docs.nvidia.com/cuopt/user-guide/latest/milp-features.html)
* [NVIDIA cuOpt system requirements](https://docs.nvidia.com/cuopt/user-guide/latest/system-requirements.html)

After installing `libcuopt` in the active Python or Conda environment, find its package root with:

```bash
CUOPT_ROOT="$(
  python -c \
    'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "libcuopt")'
)"

echo "$CUOPT_ROOT"
```

The detected directory must contain:

```text
libcuopt/
├── include/
└── lib64/
```

## Tested Environment

This table shows the environment used to test the current version of the project.
Other compatible systems may also work.

| Component | Tested version |
|---|---|
| Operating system | Ubuntu 24.04.4 LTS |
| Linux kernel | 6.8.0-134-generic |
| GPU | NVIDIA RTX A4000 |
| NVIDIA driver | 550.163.01 |
| CUDA compiler | CUDA 12.0, V12.0.140 |
| CMake | 3.28.3 |
| GCC and G++ | 13.3.0 |
| Python | 3.12.13 |
| OR-Tools C++ | 9.7.2996 |
| NVIDIA libcuopt | 26.6.0 |

## Build Options

XEG has two build variants.

| Build variant | CP-SAT | cuOpt | CMake setting |
|---|---:|---:|---|
| Standard build | Yes | No | `XEG_ENABLE_CUOPT=OFF` |
| cuOpt-enabled build | Yes | Yes | `XEG_ENABLE_CUOPT=ON` |

### Clone the Repository

```bash
git clone https://github.com/Yi-Huaaa/gpu_extractor.git
cd gpu_extractor
```

### Build Without cuOpt

This is the default build.
It requires OR-Tools but does not require cuOpt.

```bash
cmake \
  -S . \
  -B build-cpsat \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cpsat -j "$(nproc)"
```

### Build With cuOpt

Activate the environment that contains `libcuopt` before configuring the build.
For example:

```bash
conda activate YOUR_CUOPT_ENV
```

Find the `libcuopt` package root:

```bash
CUOPT_ROOT="$(
  python -c \
    'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "libcuopt")'
)"
```

Configure and build:

```bash
cmake \
  -S . \
  -B build-cuopt \
  -DCMAKE_BUILD_TYPE=Release \
  -DXEG_ENABLE_CUOPT=ON \
  -DCUOPT_ROOT="$CUOPT_ROOT"

cmake --build build-cuopt -j "$(nproc)"
```

## Run the Simple Driver

The main command-line driver is `main/simple`.
It reads one input JSON file and runs GPU tree construction, GPU DAG construction, and the selected integer solver.

### Command Syntax

```text
./<build-directory>/main/simple \
  <input_file> \
  [DAG_KERNEL] \
  [NUM_THREAD_DAG_CONSTR] \
  [BITMAP_BITS] \
  [ILP_BACKEND] \
  [CUOPT_MODE]
```

### Runtime Options

| Argument | Values | Default | Meaning |
|---|---|---:|---|
| `DAG_KERNEL` | `0`, `1`, `2` | `0` | Thread, Warp, or Block DAG construction |
| `NUM_THREAD_DAG_CONSTR` | Positive integer | `1` | Threads used by DAG construction |
| `BITMAP_BITS` | See table below | `0` | Exact or bitmap overlap detection |
| `ILP_BACKEND` | `0`, `1` | `0` | CP-SAT or cuOpt |
| `CUOPT_MODE` | `0`, `1` | `0` | cuOpt presolve or GPU tree warm start |

#### DAG Kernel

| Value | Kernel | Typical thread count |
|---:|---|---:|
| `0` | Thread | `1` |
| `1` | Warp | `128` |
| `2` | Block | `128` |

#### Bitmap Bits

| DAG kernel | Valid values |
|---|---|
| Thread | `0` |
| Warp | `0`, `32`, `64`, `128`, `256`, `512`, `1024`, `2048`, `4096` |
| Block | `0`, `32`, `64`, `128`, `256`, `512`, `1024`, `2048`, `4096` |

`BITMAP_BITS=0` uses exact overlap detection.
A nonzero value uses approximate bitmap overlap detection.

#### ILP Backend

| Value | Backend | Notes |
|---:|---|---|
| `0` | CP-SAT | Default backend |
| `1` | cuOpt | Requires a cuOpt-enabled build |

#### cuOpt Mode

Use `CUOPT_MODE` only with `ILP_BACKEND=1`.

| Value | Mode | Presolve | GPU tree MIP start | Notes |
|---:|---|---:|---:|---|
| `0` | Presolve | On | No | Default cuOpt mode |
| `1` | GPU tree warm start | Off | Yes | Starts from the GPU tree solution |

In mode 1, the current cuOpt 26.6 integration disables presolve before submitting the MIP start.
Mode 1 gives cuOpt an immediate feasible starting solution, but it may still perform worse than mode 0 on some inputs.
GPU DAG construction is not deterministic, so the warm-start objective may vary slightly between runs.

## Examples

Replace `path/to/input.json` with a valid XEG input file.

### Run With CP-SAT

```bash
./build-cpsat/main/simple \
  path/to/input.json \
  0 \
  1 \
  0 \
  0
```

### Run With cuOpt Presolve

```bash
./build-cuopt/main/simple \
  path/to/input.json \
  0 \
  1 \
  0 \
  1 \
  0
```

### Run With cuOpt GPU Tree Warm Start

```bash
./build-cuopt/main/simple \
  path/to/input.json \
  0 \
  1 \
  0 \
  1 \
  1
```

### Run a Warp Bitmap Configuration

```bash
./build-cpsat/main/simple \
  path/to/input.json \
  1 \
  128 \
  64 \
  0
```

## Run All Tests

### Standard Build

```bash
cmake --build build-cpsat -j "$(nproc)"

ctest \
  --test-dir build-cpsat \
  --output-on-failure
```

### cuOpt Build

Activate the environment that contains `libcuopt` before running the cuOpt tests.

```bash
conda activate YOUR_CUOPT_ENV

cmake --build build-cuopt -j "$(nproc)"

ctest \
  --test-dir build-cuopt \
  --output-on-failure
```

## Benchmarks

The full benchmark set is not included in this repository yet because the data is too large for standard GitHub storage.

Some of the smaller inputs used by this project come from the [Extraction Gym](https://github.com/egraphs-good/extraction-gym).

Users can also run `main/simple` with their own compatible JSON input files.

TODO: Add a `benchmark/` directory with a small public benchmark set and clear commands for repeatable runs.

## cuOpt Notes and Limitations

* cuOpt support is optional.
* The NVIDIA cuOpt MILP solver is currently in beta.
* cuOpt mode 0 uses presolve and does not submit a MIP start.
* cuOpt mode 1 disables presolve and submits the GPU tree solution as a MIP start.
* A feasible solution does not prove that the solution is optimal.
* Solver quality and runtime depend on the input, the selected mode, and the solver time limit.
* GPU DAG construction may produce slightly different search spaces between runs.

## Other Programs

The `main/` directory contains additional experiment programs.
Their detailed usage will be documented separately.

## References

* [Google OR-Tools](https://github.com/google/or-tools)
* [Google OR-Tools C++ installation](https://developers.google.com/optimization/install/cpp)
* [NVIDIA cuOpt](https://github.com/NVIDIA/cuopt)
* [NVIDIA cuOpt documentation](https://docs.nvidia.com/cuopt/user-guide/latest/)
* [Extraction Gym](https://github.com/egraphs-good/extraction-gym)
