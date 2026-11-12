# GPU design profiles

`87-designs.json` contains measured profiles for the 87 benchmark inputs: 33 override the common Block/64/512 configuration, and 54 inherit its defaults. The [results and selection protocol](../../results/gpu/20261112/profiled.md) show costs, separate construction timings, and rejected trials. These are selected observations, not guaranteed optima; GPU construction and timed parallel solving remain nondeterministic. Empty design objects use the common defaults.

```bash
scripts/run_gpu.sh dataset/boolE/mul32_map.rename.json --config configs/gpu/87-designs.json
# Direct executable; use --design if the input has been moved or renamed:
build/main/gpu input.json --config configs/gpu/87-designs.json --design boolE/mul32_map.rename --report report.json
```

The design key is the immediate parent directory plus the filename without `.json`. Resolution order is built-in defaults, configuration `defaults`, then the matching `designs` entry. Explicit positional kernel/thread/bitmap arguments override the resolved geometry. Without a configuration, the program retains Thread/1/0 and the original controller defaults. Unlisted inputs inherit the configuration defaults. Unknown keys, invalid values, and malformed entries fail before construction.

The JSON format uses the project's existing parser. Every report records the effective kernel, thread count, bitmap size, and all `dag_options`; changing a profile never changes the saved report.

| Option | Original default | Meaning |
|---|---:|---|
| `exact_batch_size` | 256 | Maximum concurrent candidate lanes in exact transposed batches; also the Thread batch's CUDA block size. Range 1–1024. |
| `bitmap_batch_size` | 32 | Maximum concurrent candidate lanes in bitmap transposed batches. Range 1–1024; independent of threads per candidate. |
| `insertion_policy` | 2 | 0: full batch until convergence; 1: stop on low activity; 2: switch from batch to direct on low activity. |
| `low_activity_minimum_iterations` | 4 | Minimum completed iterations before applying the low-activity rule. |
| `low_activity_percent` | 3 | Positive surviving insertions at or below this percentage of the observed peak count as low activity. Range 1–100. |
| `low_activity_patience` | 2 | Required consecutive low-activity iterations; positive and no greater than the minimum iteration count. |
| `bitmap_max_direct_iterations` | 5 | Positive limit on direct iterations for Warp/Block bitmap modes; exact modes run until convergence. |
| `hybrid_levelization` | true | Allow the existing persistent single-block levelization path. False uses the host-driven path. |
| `persistent_threads` | 512 | Persistent CUDA block size, a multiple of 32 from 32 through 1024. |
| `persistent_enter_frontier` | 512 | Maximum frontier width counted toward persistent entry. |
| `persistent_small_levels` | 8 | Consecutive small frontiers required before entering the persistent path. |
| `persistent_exit_frontier` | 1024 | A wider frontier returns control to the host; must be at least the entry threshold. |

For example, `"boolE/mul32_map.rename": {"bitmap_batch_size": 8}` changes only that design's bitmap concurrency. This is an illustration, not a measured recommendation. Smaller batches can preserve more compatible candidates but also increase construction work. GPU scheduling can change candidate selection between runs, so compare repeated runs with the same solver budget and validate every exported solution. Candidate count alone does not establish solution quality.

Correctness checks and CP-SAT constraints are independent of profiles. Each run validates its seed, removes candidates with impossible child dependencies, checks whether the candidate graph is acyclic, preserves the seed, and validates the final solution. Only a checked acyclic candidate graph permits omission of CP-SAT level constraints. Performance and quality measurements belong in `results/gpu/`.
