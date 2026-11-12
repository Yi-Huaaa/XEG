# Independent solution verification

```bash
python3 verification/verify_solution.py INPUT.json REPORT.json
python3 -m unittest discover -s verification -p 'test_*.py'
python3 verification/check_threads.py INPUT.json --output build/verification
```

`check_threads.py` also accepts a dataset directory. It checks all six default
construction thread counts with one solver worker, verifies each solution, and
requires identical construction hashes, final costs, solver statuses, solution
sources, and fallback reasons. Its summary records status, source, and reason
alongside the independently checked objective. Run it inside a CPU
allocation for large datasets. Its timings are not a performance benchmark;
use `scripts/benchmark_cpu.py` for construction timing.

The checker uses only Python's standard library. It reads the original JSON and
exported e-node names, independently of XEG's dense IDs, CSR graph, filtering,
and C++ validators. It checks roots, one choice per class, every selected node's
child classes, acyclicity, and the integer objective. Repeated children and roots
count only once in the DAG objective. Costs follow the existing reader's float32
conversion and solver's integer truncation. Disconnected selected classes are
reported separately; they are allowed by the current solver formulation.

Candidate-DAG closure is checked inside the CPU pipeline before solving. Missing
child classes invalidate dependent candidates recursively. E-nodes that depend
on their own class are also removed. The valid seed tree must remain available.

The validated CPU solver omits class-level ordering variables. This is safe
because a selected cycle of e-classes would induce a cycle among their unique
selected e-nodes in the candidate graph. That graph is checked to be acyclic,
and candidates referring to their own class are removed explicitly. The solver
still enforces root selection, one choice per active class, and child closure.
The independent checker verifies acyclicity again on the actual final solution.

For reproducible final-cost comparisons, use one solver worker and a fixed
deterministic work limit (the last argument), for example:

```bash
build/main/cpu INPUT.json 1 360 1 baseline.json 1 10
build/main/cpu INPUT.json 8 360 1 parallel.json 1 10
python3 verification/verify_solution.py INPUT.json baseline.json
python3 verification/verify_solution.py INPUT.json parallel.json
```

The 360-second wall limit is a safety cap; comparisons must finish before it.
Compare `solution.objective`, construction hashes, and solver status. If solving
stops without an incumbent, `solution.source` explicitly records `tree_fallback`;
the solver status remains `UNKNOWN`. The seed is also retained when a short solve
finds a more expensive incumbent; `solution.reason` records that case while
`solver` preserves its actual status and objective. A hint is not guaranteed to
become the solver's incumbent. The fallback is the independently checked seed
tree, not a claim that the solver found or proved an optimum.
