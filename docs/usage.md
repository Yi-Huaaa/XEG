# Running XEG

Start with the [quick start](../README.md#quick-start) for a complete CPU or GPU
run. This page covers optional settings, reported metrics, builds, and tests.

## CPU settings

```text
scripts/run_cpu.sh INPUT.json [THREADS=1] [ILP_SECONDS=360] [SOLVER_WORKERS=8]
```

`THREADS` controls CPU tree/DAG construction. `SOLVER_WORKERS` independently
controls CP-SAT. Set `ILP_SECONDS=0` to run construction and validation only:

```bash
scripts/run_cpu.sh dataset/esyn/mul3.rename.json 8 0 8
```

## GPU settings

The recommended GPU runner is `scripts/run_gpu.sh`:

```text
scripts/run_gpu.sh INPUT.json [KERNEL] [DAG_THREADS] [BITMAP_BITS] [ILP_SECONDS] [SOLVER_WORKERS]
                             [--config FILE.json] [--design FAMILY/STEM] [--solver-seconds SECONDS] [--workers N]
```

### Automatic settings

```bash
scripts/run_gpu.sh dataset/boolE/mul16.rename.json
scripts/run_gpu.sh path/to/your-design.json
```

Both commands use the same GPU implementation. The runner reads
[`configs/gpu/87-designs.json`](../configs/gpu/87-designs.json): matching designs
receive their saved overrides, and other designs use the common defaults.
For an unlisted input, the five default arguments are **`2 64 512 360 8`**.

### Optional manual arguments

```bash
scripts/run_gpu.sh path/to/your-design.json 2 128 1024 60 4
```

| Argument | Common default | Meaning |
|---|---:|---|
| `KERNEL` | `2` | Block kernel; `0` is Thread and `1` is Warp |
| `DAG_THREADS` | `64` | CUDA execution width for candidate insertion |
| `BITMAP_BITS` | `512` | Overlap bitmap size; `0` selects exact overlap detection |
| `ILP_SECONDS` | `360` | CP-SAT time limit in seconds |
| `SOLVER_WORKERS` | `8` | CP-SAT worker count |

Use `--solver-seconds SECONDS --workers N` to change solver limits without
overriding saved geometry. An explicit positional kernel starts from canonical
Thread/1/0, Warp/32/0, or Block/64/512 geometry; subsequent positional values
override it. Invalid geometry is rejected before reading the graph. Controller
settings still come from the configuration file. A solver limit of `0` skips
CP-SAT and runs construction and validation only:

```bash
scripts/run_gpu.sh dataset/boolE/mul16.rename.json --solver-seconds 0 --workers 8
```

### Custom JSON settings

```bash
scripts/run_gpu.sh path/to/your-design.json --config my-gpu-settings.json
```

The [JSON configuration guide](../configs/gpu/README.md#write-your-own-settings)
provides a complete example, explains `defaults` and per-design overrides, and
lists every valid parameter. A custom file replaces the bundled configuration.
If an input has been moved or renamed, `--design FAMILY/STEM` selects its saved
settings explicitly.

The scripts read JSON at runtime. Direct `gpu`, `simple`, and `xeg_t` executables
embed the bundled settings at build time; rebuild to refresh them or supply
`--config`. The legacy `scripts/run.sh` invokes `simple`, whose fourth and fifth
optional numbers are **ILP backend and cuOpt mode**, not solver seconds and
workers. See the [program reference](../main/README.md) for direct and legacy syntax.

## Reports and verification

Both recommended runners save `run.log`, `report.json`, and solver memory logs.
Without `XEG_RUN_DIR`, each run gets a timestamped directory under `build/runs/`.
Reusing an explicit run directory replaces its `run.log` and `report.json`.
The JSON report is an array with one entry per repetition.
CP-SAT currently writes `cpsat_memory_<pid>.log` in its working directory even
for direct executable calls; that directory must be writable. The runners
provide a writable run directory. Memory logging has not been made optional.

| Report field | Meaning |
|---|---|
| `profile_matched` (GPU CLI) | Whether automatic/explicit selection found a listed design; false means common defaults |
| `solution.objective` | Final returned cost, including every selected node |
| `solution.source` | `solver` or `tree_fallback` |
| `solver.status` | Solver outcome; `OPTIMAL` applies within the constructed candidate space |
| `tree_ms`, `dag_ms` | Separate construction times in milliseconds, excluding parsing and solving |
| `cpu_pipeline_ms` / `gpu_pipeline_ms` | Pipeline time after input parsing, including checks and solving |

CPU DAG timing includes candidate cleanup. GPU `tree_ms`/`dag_ms` are native
CUDA-event timings; setup, seed checks, and post-DAG cleanup are reported
separately. These phase timers are not interchangeable with overall runtime.

The independent checker verifies root coverage, unique selection per class,
child closure, acyclicity, and the integer-cost objective. Construction-only
runs have no final solver solution to check. If CP-SAT finds no incumbent, or
returns a more expensive incumbent than the legal seed, XEG can return that
seed as `tree_fallback`; `solver` retains the solver's actual result. See
[verification](../verification/README.md) for details and deterministic CPU checks.

The run scripts disable OpenMP binding so CP-SAT workers do not inherit a
single-core affinity. For direct calls that solve, use
`env -u OMP_PLACES OMP_PROC_BIND=false` before the executable.

## Benchmarks

The 87-design suite covers BoolE, Diospyros, E-Syn, FlexC, IMpress, MaxSAT,
E-morphic, Rover, and Tensat. All 87 inputs are included: 81 JSON files and six
compressed BoolE inputs. The CPU/GPU runners, batch runner, and CPU benchmark
helper automatically restore a missing `INPUT.json` from `INPUT.json.zip`.
Existing JSON files are kept unchanged. See [dataset/](../dataset/README.md) for
manual preparation before invoking executables directly.

To compare CPU construction thread counts without solver time:

```bash
python3 scripts/benchmark_cpu.py dataset/esyn/mul3.rename.json \
  --threads 1 2 4 8 16 32 --repeats 3 --output build/cpu-benchmark
```

Run benchmarks in an appropriate compute allocation. Keep solver budgets and
worker counts fixed when comparing solution quality. GPU construction and
parallel timed solving remain nondeterministic; profiles do not guarantee a
particular final cost on the next run. The [recorded results](../results/README.md)
include experiment settings and limitations.

## Build options

`scripts/build.sh cpu gpu` builds both recommended drivers in `build/main/`.
Use `scripts/build.sh cpu` or `scripts/build.sh gpu` for one target. With no
targets, the helper builds the legacy `simple` driver.

| Variable | Default | Purpose |
|---|---|---|
| `XEG_BUILD_DIR` | `<repo>/build` | Build and runner executable directory |
| `XEG_CUDA_ARCH` | First detected GPU; `86` if none | CUDA compute architecture, e.g. `86` or `120` |
| `ORTOOLS_HOME` | `$HOME/or-tools_x86_64_Ubuntu-22.04_cpp_v9.7.2996` | OR-Tools installation |
| `XEG_RUN_DIR` | Timestamped directory under `<build>/runs` | Output directory for one scripted run |

CMake also accepts `-DORTOOLS_HOME=...` and `-DCMAKE_CUDA_ARCHITECTURES=...` when
invoked directly. The build script's positional arguments are target names.
The optional NVIDIA cuOpt backend is documented in [docs/cuopt.md](cuopt.md).

## Tests

Build and run the complete test suite on a GPU machine:

```bash
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s verification -p 'test_*.py'
```

For CPU execution without a GPU, run the CPU tests. Building still requires
the CUDA toolkit:

```bash
scripts/build.sh test_cpu_extraction test_cpu_solver
ctest --test-dir build -R '^test_cpu_(extraction|solver)$' --output-on-failure
python3 -m unittest discover -s verification -p 'test_*.py'
```

## Library and experimental programs

The public C++ class is `XEG` (`#include <xeg/XEG.hpp>`); the CMake library target
is `xeg`. Legacy XEG-D remains single-threaded, and its canonical seeds and
construction timings can differ from historical runs. See the
[program reference](../main/README.md#xeg-d-and-compatibility) for compatibility
notes and experimental tools.
