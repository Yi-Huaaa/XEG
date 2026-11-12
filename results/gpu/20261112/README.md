# GPU extraction results

The [fixed-configuration comparison](comparison.md) covers all 87 designs with the original Block/64/512 settings. It contains separate solution-quality, tree-runtime, and DAG-runtime tables; [comparison.csv](comparison.csv) preserves unrounded measurements.

All 87 repaired GPU solutions passed the independent verifier: 81 OPTIMAL and 6 FEASIBLE. The original GPU returned 75 valid incumbents, 8 INFEASIBLE results, and 4 UNKNOWN results. Native geometric-mean speedups are 1.143x for tree construction and 0.979x for DAG construction. This fixed-parameter result does not meet the goal of improving both phases.

Design-profile experiments are still running. Their selected results will be published separately, with the trial-selection method and fresh construction timings. The current fixed-configuration table will remain available.

Correctness and regression checks completed so far: 139 C++/CUDA tests, 15 Python tests, all 87 CPU regression comparisons, and GPU memory checking on the tree/repeated-extraction tests (zero reported errors or leaked allocations). Raw logs, source snapshots, failed attempts, and reproduction scripts remain under `build/plan/20261112_gpu_alignment/` on the experiment server.
