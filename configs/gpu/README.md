# GPU configuration

Run any input with one command:

```bash
scripts/run_gpu.sh path/to/your-design.json
```

The runner reads [`87-designs.json`](87-designs.json) automatically. It starts
with `defaults`, then applies a matching entry from `designs`. The bundled file
covers 87 inputs: 33 have overrides and 54 use the common defaults. Inputs not
listed in `designs` also use the common defaults. No separate execution mode is
needed.

The common settings are kernel `2` (Block), `64` DAG threads, and `512` bitmap
bits. CP-SAT separately defaults to `360` seconds and `8` workers. The saved
settings and [recorded results](../../results/gpu/20261112/profiled.md) come from
measured trials; they do not guarantee the same cost on every run.

## Write your own settings

Save the following as `my-gpu-settings.json`. JSON uses double-quoted keys and
does not allow comments or trailing commas.

```json
{
  "schema_version": 1,
  "defaults": {
    "kernel": 2,
    "dag_threads": 64,
    "bitmap_bits": 512,
    "bitmap_batch_size": 32
  },
  "designs": {
    "my-designs/adder": {
      "bitmap_batch_size": 8,
      "bitmap_max_direct_iterations": 10
    },
    "my-designs/multiplier": {
      "bitmap_bits": 2048
    }
  }
}
```

Then run:

```bash
scripts/run_gpu.sh my-designs/adder.json --config my-gpu-settings.json
```

- **`defaults`** supplies values shared by all designs in this file.
- **`designs`** supplies per-input overrides. The example adder keeps Block / 64 /
  512 but uses batch size 8 and at most 10 direct iterations. The multiplier
  changes only the bitmap size. An unlisted design uses `defaults` unchanged.
- The key is the **immediate parent folder plus filename without the final
  `.json`**. For `dataset/boolE/mul16.rename.json`, use `boolE/mul16.rename`.
- A custom file **replaces** the bundled file. To keep the existing 87 settings,
  copy `configs/gpu/87-designs.json` and edit the copy instead.
- Include `kernel`, `dag_threads`, and `bitmap_bits` explicitly in your custom
  `defaults`, as above. Controller fields omitted from `defaults` use the
  built-in values listed below. For compatibility, omitted geometry fields in
  a custom file fall back to Thread / 1 / 0; the bundled file explicitly sets
  Block / 64 / 512.

For one configuration shared by every input, remove the `designs` section or
set it to `{}`. The examples demonstrate the format; their parameter choices
are not measured recommendations for your input.

If you move or rename an input, select its settings explicitly:

```bash
scripts/run_gpu.sh renamed.json --config my-gpu-settings.json --design my-designs/adder
```

## Override arguments on the command line

```bash
scripts/run_gpu.sh my-designs/adder.json 2 128 1024 60 4 --config my-gpu-settings.json
```

The five numbers are **kernel, DAG threads, bitmap bits, solver seconds, solver
workers**. This example overrides the geometry to Block / 128 / 1024, keeps the
adder's JSON controller settings, and gives CP-SAT 60 seconds and 4 workers.
A solver limit of `0` skips solving.

Solver limits and worker counts belong to CLI arguments, not this JSON schema.
Unknown JSON keys, invalid types, and incompatible geometry are rejected before
construction, including invalid entries for designs not selected by this run.

## Geometry parameters

| Key | Values |
|---|---|
| `kernel` | `0` = Thread, `1` = Warp, `2` = Block |
| `dag_threads` | Thread: `1`; Warp: multiples of 32 from 32 to 1024; Block: `64`, `128`, `256`, `512`, `1024` |
| `bitmap_bits` | Thread: `0`; Warp/Block: `0` for exact overlap detection, or powers of two from 32 to 4096 |

`dag_threads` is the CUDA execution width for each candidate insertion, not a
CPU or solver thread count. A valid example for exact overlap detection is
`"kernel": 2, "dag_threads": 64, "bitmap_bits": 0`.

## DAG controller parameters

| Option | Built-in controller default | Meaning |
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


## Configuration loading and validation

`scripts/run_gpu.sh` and the legacy `scripts/run.sh` read the checkout's JSON
file at runtime. Direct `gpu`, `simple`, and `xeg_t` executables embed that file
at build time; rebuild after editing it, or use `--config` for runtime loading.
Explicit geometry arguments have highest priority. The report records the
actual kernel, DAG threads, bitmap size, and every `dag_options` value.

Correctness checks and CP-SAT constraints are independent of these choices.
Every run validates its seed, removes candidates with impossible child
dependencies, checks candidate acyclicity, preserves the seed, and validates the
final solution. GPU scheduling can change candidate selection between runs;
compare repeated runs with the same solver budget and verify exported solutions.
