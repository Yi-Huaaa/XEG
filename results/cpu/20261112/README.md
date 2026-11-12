# CPU parallel extraction — 2026-11-12

| Artifact | Contents |
|---|---|
| [Full report](REPORT.txt) | Implementation, measurement settings, correctness checks, results, and limitations |
| [Performance plot](performance.png) | CPU tree and DAG construction speedup across thread counts |
| [Per-input costs](quality.csv) | Paper, original CPU, delivered CPU, and fixed-budget verification costs and solver statuses |

![CPU construction performance across thread counts](performance.png)

The full report distinguishes construction timing, timed solver quality, and
fixed-budget reproducibility. It also records the remaining differences from the
paper's costs.

These three files are exact copies of the experiment artifacts in
`build/plan/20261112_cpu_parallel/`. Other logs and evidence paths mentioned in
the report refer to that local experiment directory.
