# CPU parallel extraction — 2026-11-12

| Artifact | Contents |
|---|---|
| [87-design comparison tables](comparison.md) | Paper XEG versus CPU overall runtime and final cost, followed by separate CPU tree and DAG scaling tables |
| [Construction timing tables](construction.md) | Separate tree and DAG runtimes at 1, 2, 4, 8, 16, and 32 threads, with speedups relative to 1 thread |
| [Comparison data](comparison.csv) | Exact per-design values used by the comparison tables |
| [Construction summary data](construction_summary.csv) | Arithmetic and geometric mean speedups for each benchmark and across all 87 designs |
| [Paper XEG baseline](paper_xeg.csv) | XEG runtime and final cost transcribed from Table 8 |
| [Full report](REPORT.txt) | Implementation, measurement settings, correctness checks, results, and limitations |
| [Performance plot](performance.png) | CPU tree and DAG construction speedup across thread counts |
| [Per-input costs](quality.csv) | Paper, original CPU, delivered CPU, and fixed-budget verification costs and solver statuses |

![CPU construction performance across thread counts](performance.png)

The full report distinguishes construction timing, timed solver quality, and
fixed-budget reproducibility. It also records the remaining differences from the
paper's costs.

The existing `xeg_d`, `simple`, and `xeg_t` CPU stages still use one construction
thread. The recorded comparison has 16 changed seed masks and 9 designs with
slower single-thread tree-plus-DAG construction than the original. See the
[entry-point notes](../../../README.md#existing-entry-points) for the measured
example and implications for historical logs.

Construction measurements used `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.
These settings must not be reused for solving: the observed CP-SAT workers
inherited the OpenMP primary thread's single-core affinity. `scripts/run_cpu.sh`
now disables OpenMP binding; direct solver runs should use `OMP_PROC_BIND=false`
and unset `OMP_PLACES`, as shown in the root README.

`REPORT.txt`, `performance.png`, and `quality.csv` are exact copies of the experiment artifacts in
`build/plan/20261112_cpu_parallel/`. Other logs and evidence paths mentioned in
the report refer to that local experiment directory.
