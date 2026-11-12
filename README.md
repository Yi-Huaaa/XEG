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

- **CPU and GPU construction.** OpenMP CPU and CUDA GPU implementations build
  the candidate space for extraction.
- **Profiled GPU (default settings).** The GPU runner automatically loads the
  supplied design profiles; common settings cover unlisted inputs.
- **Checked solutions.** Both implementations share CP-SAT solving and final
  checks, with a separate Python verifier for exported solutions.
- **87-design evaluation.** [Runtime and quality tables](results/gpu/20261112/overall_runtime.md)
  compare CPU, Profiled GPU (default settings), and Fixed GPU.

## Requirements

- Linux on x86_64, CMake **3.20+**, and a C++17 compiler with OpenMP.
- A CUDA toolkit and a compatible host compiler. The shared-library build
  requires CUDA even for the CPU executable. **CPU execution does not require
  a GPU**; GPU execution requires an NVIDIA GPU.
- [Google OR-Tools C++ 9.7.2996](https://github.com/google/or-tools/releases/tag/v9.7)
  and Protobuf development libraries.
- Python 3 for the independent verifier and benchmark scripts.

The recorded experiments used CUDA 12.9, GCC 12, and OR-Tools 9.7.2996.
See [build options](docs/usage.md#build-options) for installation paths and CUDA
architecture settings. NVIDIA cuOpt is an [optional backend](docs/cuopt.md).

## Quick start

```bash
git clone https://github.com/Yi-Huaaa/gpu_extractor.git
cd gpu_extractor

# If OR-Tools is installed outside the default location:
# export ORTOOLS_HOME=/path/to/or-tools
scripts/build.sh cpu gpu
```

Run these commands from the repository root, using an appropriate compute
allocation on a cluster. Both examples extract from the supplied BoolE `mul16`
e-graph and independently check the result.

### Profiled GPU (default settings)

```bash
XEG_RUN_DIR=build/runs/example-gpu \
  scripts/run_gpu.sh dataset/boolE/mul16.rename.json

python3 verification/verify_solution.py \
  dataset/boolE/mul16.rename.json build/runs/example-gpu/report.json
```

No optional arguments are needed. The runner loads
[`configs/gpu/87-designs.json`](configs/gpu/87-designs.json) and selects the input's
profile. CP-SAT defaults to **360 seconds and 8 workers**. Unlisted designs use
common Block / 64 / 512 geometry and the common controller settings.

### CPU

```bash
# 8 construction threads, 360-second CP-SAT limit, 8 solver workers
XEG_RUN_DIR=build/runs/example-cpu \
  scripts/run_cpu.sh dataset/boolE/mul16.rename.json 8 360 8

python3 verification/verify_solution.py \
  dataset/boolE/mul16.rename.json build/runs/example-cpu/report.json
```

CPU construction threads and CP-SAT workers are separate settings. Both runners
write `run.log`, `report.json`, and solver memory logs. If `XEG_RUN_DIR` is omitted,
each run gets a timestamped directory under `build/runs/`.

### Fixed GPU (comparison settings)

To use the common GPU parameters without per-design overrides:

```bash
scripts/run_gpu.sh dataset/boolE/mul16.rename.json 2 64 512 360 8 --no-profile
```

The five numbers mean **kernel, DAG threads, bitmap bits, solver seconds, and
solver workers**. Here `2` selects the Block kernel. **A solver limit of `0`
skips CP-SAT**; `2 64 512 0 8` is a construction-only run. `--no-profile` alone
restores Thread / 1 / 0, so keep `2 64 512` for Fixed GPU.

For custom profiles, construction-only runs, and parameter precedence, see
[run settings](docs/usage.md#gpu-settings).

## How XEG works

1. **Tree construction** produces a valid seed solution.
2. **DAG construction** adds alternatives, removes candidates with impossible
   child dependencies, and checks that the seed remains available.
3. **CP-SAT extraction** chooses a low-cost solution with root coverage, one
   choice per active e-class, and child closure. Level constraints are omitted
   only for a checked acyclic candidate graph; the final solution is checked again.

CPU construction is deterministic across thread counts. GPU construction and
parallel timed solving are nondeterministic. CPU and GPU share the solver code,
but their candidate spaces can differ in size, final cost, and solving time.

## Results

| Resource | Contents |
|---|---|
| [Overall runtime and solution quality](results/gpu/20261112/overall_runtime.md) | CPU 8 threads / Profiled GPU (default settings) / Fixed GPU across 87 designs |
| [CPU construction scaling](results/cpu/20261112/comparison.md) | Tree/DAG timings across thread counts and quality versus paper XEG |
| [Profiled GPU (default settings)](results/gpu/20261112/profiled.md) | Selected profiles, solution quality, and separate construction timings |
| [Fixed GPU](results/gpu/20261112/comparison.md) | Implementation comparison with common original parameters |

Quality cells in the overall table show `final cost (cost ratio, status)`.
The cost ratio uses **paper XEG's final cost** as the baseline; lower is better.
O means optimal within that run's candidate space, F means feasible, and T means
validated tree fallback.

Overall time includes construction, in-program checks, and CP-SAT, excluding
input parsing. The recorded CPU and GPU sweeps used different concurrent loads,
so the overall times are reference observations rather than a controlled speedup
comparison. Profile selection and timing limitations are explained with the results.

## Documentation

| Guide | Contents |
|---|---|
| [Running XEG](docs/usage.md) | CPU/GPU arguments, reports, benchmarks, build options, and tests |
| [GPU profiles](configs/gpu/README.md) | Per-design settings and all controller parameters |
| [Solution verification](verification/README.md) | Independent legality checks and deterministic CPU comparisons |
| [Program reference](main/README.md) | Direct CLI usage, legacy tools, and XEG-D compatibility notes |
| [Optional cuOpt backend](docs/cuopt.md) | Build instructions and solver modes |

The repository includes 81 of the 87 benchmark inputs. See the
[benchmark notes](docs/usage.md#benchmarks) for the six large inputs omitted from git.

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
