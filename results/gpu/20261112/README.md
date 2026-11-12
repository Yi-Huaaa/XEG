# GPU extraction results

- [Selected design profiles: quality and separate tree/DAG tables](profiled.md)
- [Fixed original parameters: implementation-only comparison](comparison.md)
- [Full report and experiment limits](REPORT.txt)
- [Per-design costs](quality.csv), [full measurements](profiles.csv), [profile settings](profiles.json)
- [All parameter-trial summaries, including rejected settings](profile_trials.csv)
- [Original INFEASIBLE diagnostics](infeasible_diagnostics.csv)

All 87 selected GPU solutions passed independent verification: 80 OPTIMAL, 7 FEASIBLE. Against the original GPU, native geometric-mean speedups are **1.145x tree** and **0.998x DAG**. 23 designs have slower tree timings and 44 have slower DAG timings; per-design results and added verification overhead are shown in the tables.

Selected quality versus the CPU reference: 17 lower, 51 equal, 19 higher; 0 without a reference answer. Profiles are chosen from measured trials and are not guaranteed optima or deterministic results. The fixed-parameter comparison remains separate to make the effect of tuning visible.

![GPU construction speedups](performance.png)

Validation: 139 C++/CUDA tests, 15 Python tests, 87 CPU regression comparisons, and GPU memcheck on tree/repeated-extraction tests. Raw experiment/review logs remain under `build/plan/20261112_gpu_alignment/` on the experiment server.

The CUDA CLI now uses these measured profiles by default. The tables retain their original recorded runs; this default-selection change does not rerun or replace the measurements. To request the Fixed GPU settings, pass `2 64 512 --no-profile`.
