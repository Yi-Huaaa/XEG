# XEG

**Efficient and effective e-graph extraction on CPU and GPU**

[![Paper](https://img.shields.io/badge/paper-ASPLOS%202027-blue)](https://yi-huaaa.github.io/uploads/publications/2027-xeg-paper.pdf)
[![CUDA](https://img.shields.io/badge/GPU-CUDA-76B900?logo=nvidia&logoColor=white)](#requirements)
[![OpenMP](https://img.shields.io/badge/CPU-OpenMP-0071C5)](#cpu)

XEG extracts a low-cost program from a saturated e-graph under a DAG cost model,
where shared nodes are counted once. It constructs a seed tree, expands the
candidate space, and uses Google OR-Tools CP-SAT to select a solution.

This repository accompanies our ASPLOS 2027 paper
[*XEG: A GPU-parallel Algorithm for Efficient and Effective E-Graph
Extraction*](https://yi-huaaa.github.io/uploads/publications/2027-xeg-paper.pdf).
It includes OpenMP CPU and CUDA GPU implementations, scripts for running both,
and an independent solution checker. The implementation and GPU profiles have
evolved since the paper; [results/](results/README.md) distinguishes the paper's
numbers from measurements of the current code.

## Highlights

- **Parallel CPU construction.** Tree and DAG construction support multiple
  threads with deterministic candidate selection.
- **CUDA construction with measured profiles.** Thread, Warp, and Block kernels;
  the GPU runners select the supplied design profiles by default.
- **Shared CP-SAT solving.** CPU and GPU use the same solver formulation and
  final-solution checks, with a valid seed fallback when needed.
- **Independent verification.** A standalone Python checker validates selected
  nodes against the original input JSON.

See the [87-design CPU/GPU comparison](results/gpu/20261112/overall_runtime.md)
for recorded overall runtime, final cost, and cost ratios against paper XEG.

## Requirements

- Linux on x86_64, CMake **3.20+**, and a C++17 compiler with OpenMP.
- A CUDA toolkit and a compatible host compiler. The current shared-library
  build requires CUDA even when building the CPU executable; **CPU execution
  itself does not require a GPU**. GPU execution requires an NVIDIA GPU.
- [Google OR-Tools C++ 9.7.2996](https://github.com/google/or-tools/releases/tag/v9.7)
  and Protobuf development libraries.
- Python 3 for verification and benchmark scripts.

The recent CPU/GPU experiments used CUDA 12.9, GCC 12, and OR-Tools 9.7.2996;
full machine details are in [results/](results/README.md). NVIDIA cuOpt is an
[optional backend](docs/cuopt.md), not required for the commands below.

## Quick start

```bash
git clone https://github.com/Yi-Huaaa/gpu_extractor.git
cd gpu_extractor

# If OR-Tools is installed outside the default location:
# export ORTOOLS_HOME=/path/to/or-tools
scripts/build.sh cpu gpu
```

Extract a solution from the supplied BoolE `mul16` e-graph on either
implementation, then check the exported solution. Execute these commands
from the repository root; on a cluster, use an appropriate compute allocation.

### CPU

```bash
# 8 construction threads, 360-second CP-SAT limit, 8 solver workers
XEG_RUN_DIR=build/runs/example-cpu \
  scripts/run_cpu.sh dataset/boolE/mul16.rename.json 8 360 8

python3 verification/verify_solution.py \
  dataset/boolE/mul16.rename.json build/runs/example-cpu/report.json
```

### GPU

```bash
# Profiled GPU parameters, 360-second CP-SAT limit, 8 solver workers
XEG_RUN_DIR=build/runs/example-gpu \
  scripts/run_gpu.sh dataset/boolE/mul16.rename.json

python3 verification/verify_solution.py \
  dataset/boolE/mul16.rename.json build/runs/example-gpu/report.json
```

Both runners save `run.log`, `report.json`, and solver memory logs. Without
`XEG_RUN_DIR`, each run gets a timestamped directory under `build/runs/`.
`report.json` records final cost, solver status, selected e-nodes, and separate
construction timings. It is an array with one entry per repetition. Reusing an
explicit run directory replaces its `run.log` and `report.json`.

## How XEG works

1. **Tree construction** builds a valid seed solution using local node costs.
2. **DAG construction** adds candidate alternatives. Cleanup removes choices
   with impossible child dependencies and checks that the seed survives.
3. **CP-SAT extraction** selects a low-cost solution with root coverage, one
   choice per active e-class, and child closure. Level constraints are omitted
   only when the candidate graph has been checked acyclic; the final solution
   is checked again.

CPU construction is deterministic across thread counts. GPU construction and
parallel time-limited CP-SAT are nondeterministic, so repeated runs can produce
different final costs. Sharing solver code does not imply identical solver
runtime: the CPU and GPU can construct different candidate graphs.

## Usage

### CPU script

```text
scripts/run_cpu.sh INPUT.json [THREADS=1] [ILP_SECONDS=360] [SOLVER_WORKERS=8]
```

`THREADS` controls CPU tree/DAG construction; `SOLVER_WORKERS` independently
controls CP-SAT. Set `ILP_SECONDS=0` to run construction and validation only:

```bash
scripts/run_cpu.sh dataset/esyn/mul3.rename.json 8 0 8
```

### GPU script

```text
scripts/run_gpu.sh INPUT.json [KERNEL] [DAG_THREADS] [BITMAP_BITS] [ILP_SECONDS] [SOLVER_WORKERS]
                             [--config FILE.json] [--design FAMILY/STEM] [--no-profile]
```

The default is **Profiled GPU**, using
[`configs/gpu/87-designs.json`](configs/gpu/87-designs.json). A design key is its
parent directory and filename without `.json`, such as `esyn/mul3.rename`.
Unlisted designs inherit the common Block/64/512 settings. Solver defaults are
360 seconds and 8 workers.

```bash
# Replace the profile manifest, or select a profile for a renamed input:
scripts/run_gpu.sh dataset/esyn/mul3.rename.json --config configs/gpu/87-designs.json
scripts/run_gpu.sh path/to/input.json --design esyn/mul3.rename

# Fixed GPU: original common controller parameters, Block/64/512
scripts/run_gpu.sh dataset/esyn/mul3.rename.json 2 64 512 360 8 --no-profile

# Construction and validation only, with the selected profile's controller
scripts/run_gpu.sh dataset/esyn/mul3.rename.json 2 64 512 0 8
```

`KERNEL` is `0` (Thread), `1` (Warp), or `2` (Block). `DAG_THREADS` is the CUDA
execution width for insertion, not a CPU thread count or CP-SAT worker count.
`BITMAP_BITS=0` selects exact overlap detection. Explicit positional geometry
overrides the profile; `--no-profile` disables profiles entirely and restores
Thread/1/0 unless geometry is also specified.

The scripts read the JSON at runtime. Direct `gpu`, `simple`, and `xeg_t`
executables embed the same manifest at build time; rebuild to refresh it or
supply `--config`. See the [profile reference](configs/gpu/README.md) for all
parameters and the [program reference](main/README.md) for direct CLI syntax.

### Verification and reported metrics

The independent checker verifies root coverage, unique selection per class,
child closure, acyclicity, and the integer-cost objective. A successful
construction-only run has no final solver solution to check.

| Report field | Meaning |
|---|---|
| `solution.objective` | Final returned cost; includes every selected node |
| `solution.source` | `solver` or `tree_fallback` |
| `solver.status` | Solver outcome; `OPTIMAL` applies within the constructed candidate space |
| `tree_ms`, `dag_ms` | Separate construction times in milliseconds, excluding parsing and solving |
| `cpu_pipeline_ms` / `gpu_pipeline_ms` | Pipeline time after input parsing, including checks and solving |

CPU DAG timing includes candidate cleanup. GPU `tree_ms`/`dag_ms` are native
CUDA-event timings; setup, seed checks, and post-DAG cleanup are reported
separately. These phase timers are not interchangeable with overall runtime.

If CP-SAT finds no incumbent within its budget, or returns a more expensive
incumbent than the legal seed, XEG can return the seed as `tree_fallback`.
The solver's actual status and objective remain in `solver`. See
[verification/](verification/README.md) for details and deterministic CPU checks.

The run scripts disable OpenMP binding so CP-SAT workers do not inherit a
single-core affinity. For direct calls that solve, use
`env -u OMP_PLACES OMP_PROC_BIND=false` before the executable.

## Benchmarks and results

The [overall comparison](results/gpu/20261112/overall_runtime.md) places
**CPU 8 threads, Fixed GPU, and Profiled GPU** side by side for all 87 designs.
Each quality cell shows `final cost (cost ratio, status)`: the cost ratio uses
the paper's XEG final cost as its baseline, and lower is better. O means optimal
within that run's candidate space, F means feasible, and T means tree fallback.

| Resource | Contents |
|---|---|
| [Overall runtime and solution quality](results/gpu/20261112/overall_runtime.md) | CPU 8 threads / Fixed GPU / Profiled GPU, with cost ratios against paper XEG |
| [CPU results](results/cpu/20261112/README.md) | Quality versus the paper and construction timing across thread counts |
| [GPU results](results/gpu/20261112/README.md) | Original, Fixed, and Profiled GPU quality and construction timing |
| [GPU profiles](configs/gpu/87-designs.json) | Common defaults and per-design overrides |
| [Solution verification](verification/README.md) | Independent checker and CPU thread consistency checks |

The 87-design suite covers BoolE, Diospyros, E-Syn, FlexC, IMpress, MaxSAT,
E-morphic, Rover, and Tensat. The repository includes 81 JSON inputs under
`dataset/`; six large BoolE inputs are omitted from git: `mul112`, `mul128`,
`mul80_map`, `mul96_map`, `mul112_map`, and `mul128_map`.

Use `scripts/run_cpu.sh` and `scripts/run_gpu.sh` to run an individual design
with the settings shown above. GPU runs use the measured design profile by
default; append explicit geometry and `--no-profile` for the Fixed GPU settings.
To compare CPU construction thread counts without solver time:

```bash
python3 scripts/benchmark_cpu.py dataset/esyn/mul3.rename.json \
  --threads 1 2 4 8 16 32 --repeats 3 --output build/cpu-benchmark
```

Keep solver limits and worker counts fixed when comparing quality. Profiled GPU
results were selected from parameter trials; neither those profiles nor one
fresh run guarantee a particular published cost. Experiment budgets, timing
boundaries, and hardware are recorded with the results.

The overall-runtime table includes construction, in-program checks, and CP-SAT,
and excludes input parsing. Its CPU and GPU quality sweeps ran different numbers
of designs concurrently, so those times do not establish a controlled CPU/GPU
speedup. Use the separate construction tables for the recorded phase timings.

## Build options

`scripts/build.sh cpu gpu` builds both recommended drivers in `build/main/`.
Use `scripts/build.sh cpu` or `scripts/build.sh gpu` to build one target.
With no targets, the existing build helper builds the legacy `simple` driver.

| Variable | Default | Purpose |
|---|---|---|
| `XEG_BUILD_DIR` | `<repo>/build` | Build and runner executable directory |
| `XEG_CUDA_ARCH` | First detected GPU; `86` if none | CUDA compute architecture, e.g. `86` or `120` |
| `ORTOOLS_HOME` | `$HOME/or-tools_x86_64_Ubuntu-22.04_cpp_v9.7.2996` | OR-Tools installation |
| `XEG_RUN_DIR` | Timestamped directory under `<build>/runs` | Output directory for one scripted run |

CMake also accepts `-DORTOOLS_HOME=...` and
`-DCMAKE_CUDA_ARCHITECTURES=...` when invoked directly. The build script's
positional arguments are target names. For the optional NVIDIA cuOpt backend,
see [docs/cuopt.md](docs/cuopt.md).

## Tests

To build and run the complete test suite on a GPU machine:

```bash
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s verification -p 'test_*.py'
```

For CPU execution without a GPU, build and run `test_cpu_extraction` and
`test_cpu_solver`. The build still needs the CUDA toolkit:

```bash
scripts/build.sh test_cpu_extraction test_cpu_solver
ctest --test-dir build -R '^test_cpu_(extraction|solver)$' --output-on-failure
python3 -m unittest discover -s verification -p 'test_*.py'
```

## Repository layout

```text
xeg/           CPU/CUDA extraction, input reader, and solver implementations
main/          cpu, gpu, and legacy/experimental command-line programs
scripts/       build, CPU/GPU runners, and CPU construction benchmarks
configs/gpu/   measured GPU design profiles and parameter documentation
verification/  independent solution checker and thread consistency checks
unittests/     C++/CUDA tests and small input fixtures
dataset/       benchmark e-graphs
results/       published reports, plots, and per-design measurements
docs/          optional backend documentation
```

The public C++ class is `XEG` (`#include <xeg/XEG.hpp>`); the CMake library target
is `xeg`. Legacy XEG-D remains single-threaded, and its canonical seeds and
construction timings can differ from historical runs. See
[main/README.md](main/README.md) for compatibility notes and experimental tools,
and [docs/cuopt.md](docs/cuopt.md) for the optional cuOpt build.

## Citation

If you use XEG in your research, please cite our paper:

```bibtex
@inproceedings{chung2027xeg,
  author    = {Chung, Yi-Hua and Hsiao, Hao-Hsiang and Chiu, Cheng-Hsiang and
               Zhang, Boyang and Chang, Che and San Miguel, Joshua and
               Huang, Tsung-Wei},
  title     = {{XEG}: A {GPU}-parallel Algorithm for Efficient and Effective
               E-Graph Extraction},
  booktitle = {Proceedings of the 32nd ACM International Conference on
               Architectural Support for Programming Languages and Operating
               Systems, Volume 1},
  pages     = {1069--1085},
  year      = {2027},
  publisher = {ACM},
  doi       = {10.1145/3845814.3850040}
}
```

## Contact

For questions about XEG, please email Yi-Hua Chung at
[yihua.chung@wisc.edu](mailto:yihua.chung@wisc.edu).
