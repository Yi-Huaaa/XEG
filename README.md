# XEG

## Overview

XEG is an e-graph extraction system for equality saturation.
It takes a saturated e-graph with user-defined local costs and searches for a low-cost program under a DAG cost model.

The extraction flow has three main stages:

1. CPU or GPU tree construction builds an initial solution.
2. The corresponding DAG construction expands the search space while preserving acyclicity.
3. An integer solver searches the constructed space for a low-cost solution.

The project supports Google OR-Tools CP-SAT by default.
It also provides an optional NVIDIA cuOpt backend for users who want to try GPU based mixed integer optimization.

## Features

* Deterministic CPU tree and DAG construction with OpenMP
* GPU tree construction
* GPU DAG construction with Thread, Warp, and Block kernels
* Exact overlap detection and optional bitmap overlap detection
* Google OR-Tools CP-SAT backend
* Optional NVIDIA cuOpt backend
* CTest-based unit and integration tests

## CPU extraction

Build and run the CPU pipeline with:

```bash
scripts/build.sh cpu
scripts/run_cpu.sh dataset/esyn/mul3.rename.json 8 360 8
```

The arguments after the input are construction threads, CP-SAT time limit in
seconds, and CP-SAT workers. Their defaults are `1`, `360`, and `8`. Use a zero
time limit to run construction and validation without solving. CPU execution
requires no GPU; the current shared-library build still requires a CUDA toolkit.

`scripts/run_cpu.sh` sets `OMP_PROC_BIND=false` and unsets `OMP_PLACES` for the
complete pipeline, including construction-only invocations of that wrapper.
With OpenMP binding enabled, the primary thread can be pinned before CP-SAT
starts, causing subsequently created solver workers to inherit one CPU.
For direct executable calls that include solving, use:

```bash
env -u OMP_PLACES OMP_PROC_BIND=false build/main/cpu INPUT.json 8 360 8 report.json
```

For construction-only measurements, binding can be set explicitly on
`scripts/benchmark_cpu.py`; its runs disable the solver. Do not carry those
binding settings into a combined construction/solver run.

The CPU tree uses nonnegative finite local costs and recovers an acyclic optimal
tree witness with deterministic tie-breaking. DAG insertion evaluates neighbor
bounds in parallel and commits candidates in node order, so changing construction
threads preserves the selected nodes and solver input. Small graphs and narrow
levelization frontiers use serial work to avoid synchronization overhead.
After reachability filtering, candidates with unavailable child classes are
removed recursively. Candidates that depend on their own e-class are excluded.
The original valid seed tree must survive this cleanup.
The final CPU candidate graph is checked for cycles and self-dependencies before
solving. This lets its CP-SAT model omit redundant integer level variables and
level constraints. The general e-graph solver paths retain their level model.

Each run writes `run.log`, `report.json`, and the solver's memory log under
`build/runs/`. The report contains separate tree/DAG times, construction hashes,
solver status, and selected e-node names for a feasible solution. Construction
times exclude JSON parsing, validation, and solving. The final solution is checked
for root coverage, one choice per active e-class, child closure, acyclicity, and
agreement with the solver's integer-cost objective. If the solver times out
without an incumbent, or its incumbent is more expensive than the valid seed,
`solution` retains the seed tree and labels its source `tree_fallback` with a
reason. The report preserves the original solver status and objective separately.
Reports also include `validation_ms`, `solver_pipeline_ms` (formulation, solving,
and solution checks), and `cpu_pipeline_ms` (the complete CPU path after parsing).

To repeat construction on a parsed graph without repeated JSON loading:

```bash
build/main/cpu INPUT.json 8 0 8 report.json 3
```

Use `python3 scripts/benchmark_cpu.py INPUT.json --output build/cpu-benchmark`
inside an allocation to compare thread counts and check construction hashes.

Keep solver workers and its time limit fixed when comparing construction thread
counts. CP-SAT's parallel search and a finite time budget can still change its
incumbent; deterministic construction does not promise identical timed solver
outcomes or reproduce every published cost from a single run.

For a fixed, reproducible final-cost baseline, use one solver worker and an
explicit deterministic work limit as the last argument:

```bash
build/main/cpu INPUT.json 1 360 1 baseline.json 1 10
build/main/cpu INPUT.json 8 360 1 parallel.json 1 10
python3 verification/verify_solution.py INPUT.json parallel.json
```

Both runs must finish before the wall-time safety cap. See
[`verification/README.md`](verification/README.md) for the independent checker
and interpretation of solver status and fallback solutions.

### Existing entry points

`xeg_d` uses the CPU compatibility wrapper with one construction thread.
`simple` and `xeg_t` run the checked GPU pipeline; they no longer run an unrelated
serial CPU comparison first. Use the dedicated `cpu` executable to measure CPU
construction and select its thread count.

The canonical acyclic witness can differ from the original serial seed when
floating-point costs tie. In the recorded 87-design comparison, 16 seed masks
changed on inputs with tree costs beyond the exact-integer range of double
precision. Nine designs have slower current single-thread tree-plus-DAG
construction than the original; the largest relative slowdown is `mul112_map`
(1570 ms to 2830 ms, with tree construction 766 ms to 2351 ms).
Consequently, XEG-D warm starts can differ from historical logs. These observations do not measure complete
XEG-D or GPU-pipeline runtime. See the [results index](results/cpu/20261112/README.md)
for the separately measured multi-threaded CPU results.

## GPU extraction and verification

```bash
scripts/build.sh gpu
scripts/run_gpu.sh dataset/esyn/mul3.rename.json 2 64 512 360 8
python3 verification/verify_solution.py dataset/esyn/mul3.rename.json build/runs/RUN/report.json
```

Arguments after the input are kernel (`0` Thread, `1` Warp, `2` Block), DAG threads,
bitmap bits, solver seconds, and solver workers. Defaults remain `0 1 0 360 8`;
the example explicitly selects Block/64/512. The existing batch/direct policy,
bitmap hash, batching, and levelization defaults are unchanged. GPU construction
remains nondeterministic; repeated runs can produce different candidate spaces.

Per-design profiles expose batch concurrency and controller parameters without
recompiling. The supplied manifest lists all 87 benchmark inputs; empty entries
inherit its Block/64/512 defaults. See the [profile format and parameter meanings](configs/gpu/README.md).

```bash
scripts/run_gpu.sh dataset/boolE/mul32_map.rename.json --config configs/gpu/87-designs.json
```

Explicit positional kernel/thread/bitmap arguments override the profile's
geometry. Reports record every effective DAG option alongside the solution.

Tree construction still relaxes double tree costs with one GPU thread per
e-node. A per-class gate couples each strict cost improvement to its representative.
This avoids
mismatched representatives and cyclic witnesses at zero-cost or rounded ties.
Repeated roots count once in the DAG objective. Device buffers and timing events
are released after use, including repeated extractions on one object.

The GPU seed is checked before DAG expansion. After the existing root-to-leaf
filter, impossible candidates are removed recursively, and the seed must survive.
CPU and GPU use the same CP-SAT settings, final-solution checks, and cheaper-seed
fallback. Class-level constraints are omitted only for a checked acyclic candidate
space; if a GPU candidate space is cyclic, the general level model is retained.
The optional cuOpt backend retains its existing level model and output interface.

`report.json` separates native `tree_ms` and `dag_ms` CUDA-event timings from
`seed_validation_ms`, candidate copy/pruning/validation times, and `postprocess_ms`.
`tree_wall_ms` includes allocations and transfers; `dag_wall_ms` covers the original
DAG controller. `gpu_pipeline_ms` includes GPU setup, checks, solving, and cleanup,
excluding input parsing. Solver reports include `model_build_ms` and `wall_time_s`.
Use the same workers, limits, and candidate input to isolate CP-SAT overhead:
CPU and GPU construction can supply different models even with identical settings.

The wrapper disables OpenMP binding so solver workers can use the allocation's
CPUs. Direct runs and repeated construction use:

```bash
env -u OMP_PLACES OMP_PROC_BIND=false build/main/gpu INPUT.json 2 64 512 0 8 report.json 3
```

The final optional argument is a deterministic solver work limit, requiring one
solver worker and a positive wall limit, as for the CPU executable. This fixes
solver work but does not make GPU candidate construction deterministic.

Implementation files separate `gpu_tree.cu`, `gpu_dag_insert.cu`, `gpu_dag.cu`,
`gpu_pipeline.cpp`, and `cpsat_solver.cpp`. `gpu_config.hpp` defines the controller
defaults; `gpu_common.cuh` holds CUDA helpers. Tests of individual GPU stages are
in `gpu_testing.cu`.

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

The project has been tested with CUDA 12.0 and CUDA 13.1.

CMake builds for the compute capability of the first local GPU, queried with `nvidia-smi`.
If no GPU can be queried, it falls back to compute capability 8.6.
To build for a different GPU, pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>` to CMake (for example `86` or `120`), or set `XEG_CUDA_ARCH` when using `scripts/build.sh`.

### OR-Tools and CP-SAT

OR-Tools is required for both build variants.
The cuOpt build still includes the CP-SAT backend.

The current CMake configuration expects OR-Tools C++ 9.7.2996 at `$HOME/or-tools_x86_64_Ubuntu-22.04_cpp_v9.7.2996`.

Download the OR-Tools C++ binary archive and extract it under your home directory so this path exists.
To use another location, set the `ORTOOLS_HOME` environment variable or pass `-DORTOOLS_HOME=<dir>` to CMake.

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

This table shows the environments used to test the current version of the project.
Other compatible systems may also work.
The cuOpt build was tested only on the first machine.

| Component | Machine 1 | Machine 2 |
|---|---|---|
| Operating system | Ubuntu 24.04.4 LTS | Ubuntu 22.04.5 LTS |
| Linux kernel | 6.8.0-134-generic | 6.8.0-90-generic |
| GPU | NVIDIA RTX A4000 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| NVIDIA driver | 550.163.01 | 580.126.09 |
| CUDA compiler | CUDA 12.0, V12.0.140 | CUDA 13.1, V13.1.115 |
| CMake | 3.28.3 | 3.22.1 |
| GCC and G++ | 13.3.0 | 12.3.0 |
| Python | 3.12.13 | Not used |
| OR-Tools C++ | 9.7.2996 | 9.7.2996 |
| NVIDIA libcuopt | 26.6.0 | Not tested |

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

### Build With the Script

`scripts/build.sh` configures a Release build without cuOpt in `build/` and builds `main/simple`.

```bash
scripts/build.sh                  # build main/simple
scripts/build.sh simple xeg_d     # build the listed targets
```

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `XEG_BUILD_DIR` | `build` | Build directory |
| `XEG_CUDA_ARCH` | First local GPU | CUDA architecture, for example `86` or `120` |
| `ORTOOLS_HOME` | `$HOME/or-tools_x86_64_Ubuntu-22.04_cpp_v9.7.2996` | OR-Tools C++ install directory |

The CMake commands below give the same builds without the script.

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

## Run With Scripts

The run scripts use `build/main/simple` from `scripts/build.sh`.
Both pass any arguments after the input options to `main/simple` unchanged (see [Runtime Options](#runtime-options)).
With no such arguments, they use `2 64 512`: the Block kernel, 64 threads per candidate, and a 512-bit bitmap.
This differs from the `main/simple` defaults (`0 1 0`).

### Run One Input

```bash
scripts/run.sh dataset/maxsat/maxcut-140-630-0.7-1.rename.json           # 2 64 512
scripts/run.sh dataset/maxsat/maxcut-140-630-0.7-1.rename.json 0 1 0     # Thread kernel
```

Each run writes to its own directory, `build/runs/<timestamp>_<input name>/`.
`run.log` holds the full console output.
The CP-SAT memory log is written to the same directory.

### Run Every Input in a Directory

```bash
scripts/run_all.sh                                # every *.json under dataset/
scripts/run_all.sh --dataset dataset/maxsat       # one benchmark directory
scripts/run_all.sh --dataset dataset/esyn 0 1 0   # one directory, Thread kernel
```

`run_all.sh` finds the input files when it starts, so it runs whatever `*.json` files are present.
Each input runs through `scripts/run.sh` in `build/runs/<timestamp>_all/<path under the dataset directory>/`.
A failing input does not stop the batch.
`summary.tsv` in the batch directory has one row per input with these columns:

- exit code;
- CP-SAT status;
- objective;
- solver runtime;
- wall time.

Inputs that run until the 360-second CP-SAT time limit make a full run take a long time.

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

For CPU-only execution after building, run the CPU test executables and the
independent Python checker tests:

```bash
for test in test_cpu_extraction test_cpu_solver test_csr_invariants test_ilp_backend test_xeg_milp_model; do
  build/unittests/"$test" || exit 1
done
python3 -m unittest verification.test_verify_solution
```

The complete CTest suite below also runs GPU tests and requires a GPU.

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

Published experiment reports, performance plots, and per-input costs are indexed
in [results/](results/README.md).

The `dataset/` directory holds the benchmark e-graphs, one subdirectory per benchmark:

- `boolE`
- `diospyros`
- `esyn`
- `flexc`
- `impress`
- `maxsat`
- `morphic`
- `rover`
- `tensat`

The six largest BoolE inputs are not included in the repository because of their size:

- `mul112`
- `mul128`
- `mul80_map`
- `mul96_map`
- `mul112_map`
- `mul128_map`

Some of the smaller inputs used by this project come from the [Extraction Gym](https://github.com/egraphs-good/extraction-gym).

Users can also run `main/simple` with their own compatible JSON input files.

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

The public C++ class is `XEG` (`#include <xeg/XEG.hpp>`). The CMake library
target and source directory are `xeg`; test/build options use the `XEG_` prefix.
