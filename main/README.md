# XEG command-line programs

For scripted CPU/GPU runs, start with the [quick start](../README.md#quick-start).
All executables are built under `<build-directory>/main/`.

| Executable | Construction | Solver / purpose |
|---|---|---|
| `cpu` | OpenMP CPU tree and DAG | Shared checked CP-SAT pipeline |
| `gpu` | CUDA tree and DAG | Shared checked CP-SAT pipeline |
| `simple` | CUDA tree and DAG | Legacy CP-SAT/cuOpt driver |
| `xeg_t` | CUDA tree and DAG | XEG-T randomized CP-SAT hint experiment |
| `xeg_d` | Single-thread CPU tree | CP-SAT over the complete post-read e-graph |
| `output_degree_ablation` | Augmented input graph | Dump inputs or measure GPU construction |

## CPU driver

```text
build/main/cpu INPUT.json [THREADS=1] [ILP_SECONDS=360] [SOLVER_WORKERS=8]
                         [REPORT.json] [REPEATS=1] [DETERMINISTIC_SECONDS=0]
```

`THREADS` controls construction, independently of CP-SAT workers. A zero solver
limit skips solving. Repetitions reuse the parsed graph and rebuild construction
state. `REPORT.json` is an array with one entry per repetition, including selected
e-node names when a final solution is available.

```bash
# Three construction-only repetitions
build/main/cpu dataset/esyn/mul3.rename.json 8 0 8 cpu-timing.json 3

# Full extraction with an exported solution
env -u OMP_PLACES OMP_PROC_BIND=false \
  build/main/cpu dataset/esyn/mul3.rename.json 8 360 8 cpu-report.json
```

A positive deterministic solver limit requires one solver worker and a positive
wall limit. See [verification/](../verification/README.md) for reproducible CPU
comparisons; both runs must finish before the wall-time safety cap.

## GPU driver

```text
build/main/gpu INPUT.json [KERNEL] [DAG_THREADS] [BITMAP_BITS] [ILP_SECONDS=360]
                         [SOLVER_WORKERS=8] [REPORT.json] [REPEATS=1] [DETERMINISTIC_SECONDS=0]
                         [--solver-seconds SECONDS] [--workers N]
```

Known designs automatically use their saved parameters; unlisted inputs use
common Block/64/512 settings. Shared options for `gpu`, `simple`, and `xeg_t` are:

| Option | Meaning |
|---|---|
| `--config FILE.json` | Replace the built-in profile manifest |
| `--design FAMILY/STEM` | Select a profile explicitly when the input was moved or renamed |

`gpu` additionally accepts `--report FILE.json` instead of the positional report.
The report array includes the effective geometry/controller settings and final
selected e-nodes. As with CPU, zero solver seconds means construction only.

```bash
# Full extraction with automatic GPU settings
env -u OMP_PLACES OMP_PROC_BIND=false \
  build/main/gpu dataset/esyn/mul3.rename.json --report gpu-report.json

# Three construction-only repetitions with explicit Block/64/512 geometry
build/main/gpu dataset/esyn/mul3.rename.json 2 64 512 0 8 gpu-timing.json 3
```

A deterministic solver work limit fixes solver work, not the nondeterministic
GPU candidate construction. See [GPU profiles](../configs/gpu/README.md) for
all controller parameters and configuration precedence.

## Legacy GPU drivers

```text
build/main/simple INPUT.json [KERNEL] [DAG_THREADS] [BITMAP_BITS] [ILP_BACKEND=0] [CUOPT_MODE=0]
build/main/xeg_t INPUT.json [KERNEL] [DAG_THREADS] [BITMAP_BITS]
```

Both use the measured design profiles by default and accept the shared profile
options above. They run CUDA construction; they do not perform a separate CPU
construction comparison. CP-SAT uses a 360-second limit and 8 workers.
`xeg_t` randomizes its hint with seed 333 while retaining the legal GPU tree for
fallback. `simple` selects CP-SAT (`ILP_BACKEND=0`) or optional cuOpt (`1`);
see [cuOpt setup and modes](../docs/cuopt.md).

These legacy programs print their results to the console. Use `cpu` or `gpu`
for exported final solutions compatible with the independent Python checker.
For the legacy GPU batch runner:

```bash
scripts/build.sh simple
scripts/run_all.sh --dataset dataset/esyn
```

It invokes `scripts/run.sh` for each JSON input, retains per-input logs, and
writes `summary.tsv`. It is a GPU batch runner, not a CPU thread-scaling test.

### CUDA geometry

| Kernel | Value | Valid `DAG_THREADS` | Valid `BITMAP_BITS` |
|---|---:|---|---|
| Thread | `0` | `1` | `0` |
| Warp | `1` | Multiples of 32 from 32 through 1024 | `0`, or powers of two from 32 through 4096 |
| Block | `2` | `64`, `128`, `256`, `512`, `1024` | `0`, or powers of two from 32 through 4096 |

`BITMAP_BITS=0` means exact overlap detection. Defaults for unlisted inputs are Block/64/512; explicit geometry overrides
the selected values. To change controller settings, follow the
[JSON configuration guide](../configs/gpu/README.md#write-your-own-settings).

## XEG-D and compatibility

```bash
scripts/build.sh xeg_d
build/main/xeg_d dataset/esyn/mul3.rename.json
```

XEG-D uses a single-thread CPU tree warm start and solves over the complete
post-read e-graph, without DAG construction. Its default solver limit is 86400
seconds. Use `cpu` to select CPU construction thread counts.

Canonical acyclic seed recovery can choose a different tree when floating-point
costs tie. In the recorded CPU comparison, 16 original seed masks changed on
inputs with tree costs beyond double's exact-integer range; nine designs had
slower current single-thread construction than the original. The largest relative
slowdown was `mul112_map`: tree plus DAG 1570 ms to 2830 ms, including tree
766 ms to 2351 ms. These are construction measurements, not complete XEG-D
runtime. See the [CPU results](../results/cpu/20261112/README.md).

## Output-degree ablation

```text
build/main/output_degree_ablation --dump INPUT.json MULTIPLIER RANDOM_SEED
build/main/output_degree_ablation INPUT.json KERNEL DAG_THREADS BITMAP_BITS MULTIPLIER
```

Dump-only mode augments the graph and writes JSON without running extraction.
Output goes to `ablation_benchmarks/` under the current working directory.
Run from a writable directory when using this experimental tool. Names follow
`<input-stem>_sf<multiplier>_<index>.json`, using the smallest available index.

The second form augments the graph in memory using seed 113 and runs GPU tree
and DAG construction with solving disabled. It takes explicit geometry and uses
the original controller defaults, not automatic design profiles. This older
CLI accepts bitmap sizes only through 2048 bits. `MULTIPLIER` must be positive.
