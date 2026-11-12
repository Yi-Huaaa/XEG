# XEG Main Programs

This directory contains the GPU driver, the CPU driver, and experimental programs.

## 1. `simple.cu`

`simple.cu` runs the complete XEG flow on the original input e-graph.

### Usage

```bash
./build/main/simple \
  <input_file> \
  [DAG_KERNEL] \
  [NUM_THREAD_DAG_CONSTR] \
  [BITMAP_BITS]
```

Only `<input_file>` is required. The default values are:

```text
DAG_KERNEL = 0
NUM_THREAD_DAG_CONSTR = 1
BITMAP_BITS = 0
```

### Example

```bash
./build/main/simple input.json 1 128 64
```

## 2. `xeg_t.cpp`

`xeg_t.cpp` runs the XEG-T flow.

It uses the same input format and command line arguments as `simple.cu`. The main difference is that it calls `run_xeg_t_exp()` instead of `run()`.

### Usage

```bash
./build/main/xeg_t \
  <input_file> \
  [DAG_KERNEL] \
  [NUM_THREAD_DAG_CONSTR] \
  [BITMAP_BITS]
```

### Example

```bash
./build/main/xeg_t input.json 1 128 64
```

## 3. `output_degree_ablation.cpp`

`output_degree_ablation.cpp` is used for the output-degree ablation study in Figure 10.

It supports two execution modes.

### Dump-only mode

Dump-only mode reads the original e-graph, increases its output degree, writes the generated graph to a JSON file, and exits without running XEG.

### Usage

```bash
./build/main/output_degree_ablation \
  --dump \
  <input_file> \
  <OUTPUT_DEGREE_MULTIPLIER> \
  <RANDOM_SEED>
```

### Example

```bash
./build/main/output_degree_ablation \
  --dump \
  input.json \
  2 \
  42
```

Generated graphs are written to:

```text
/home/ychung79/gpu_extraction/ablation_benchmarks
```

The output filename follows this format:

```text
<input_name>_sf<multiplier>_<index>.json
```

For example:

```text
mul32.rename_sf2_1.json
```

Using the same input, multiplier, and random seed reproduces the same graph. Different random seeds normally generate different graph instances.

Example seeds used in the experiments are:

```text
42
113
2027
65537
```

### Build-and-run mode

Build-and-run mode reads the original graph, increases its output degree in memory, and immediately runs the complete XEG flow.

### Usage

```bash
./build/main/output_degree_ablation \
  <input_file> \
  <DAG_KERNEL> \
  <NUM_THREAD_DAG_CONSTR> \
  <BITMAP_BITS> \
  <OUTPUT_DEGREE_MULTIPLIER>
```

### Example

```bash
SEEDS=(42 113 2027 65537)
for n in 1 2 4 8 16 32 64 128; do
  echo "Scaling factor=$n"
  for seed in "${SEEDS[@]}"; do
    echo "seed=$seed"
    ./build/main/output_degree_ablation \
      --dump \
      "$INPUT" \
      "$n" \
      "$seed"
  done
done
```

This mode uses the default random seed defined in `output_degree_ablation.cpp`.

## Common Arguments

### `DAG_KERNEL`

```text
0: Thread
1: Warp
2: Block
```

### `NUM_THREAD_DAG_CONSTR`

This value controls the execution width used during DAG construction.

Typical values include:

```text
1 (Thread version only)
32 (Warp version only)
64 (Block version only)
128 (Block version only)
256 (Block version only)
512 (Block version only)
1024 (Block version only)
```

### `BITMAP_BITS`

```text
0: NO bitmap
32, 64, 128, 256, 512, 1024, 2048: Bitmap size
```

The Thread kernel supports only `BITMAP_BITS=0`.

### `OUTPUT_DEGREE_MULTIPLIER` (Only utilized in `output_degree_ablation.cpp`)

This value controls how much the total number of child references is increased.

Typical values include:

```text
2
4
8
16
32
64
```

## CPU driver (`cpu.cpp`)

```bash
build/main/cpu INPUT [THREADS=1] [ILP_SECONDS=360] [SOLVER_WORKERS=8] [REPORT.json] [REPEATS=1]
```

This runs CPU tree construction, CPU DAG construction, validation, and CP-SAT
on the CPU-created search space, using its original tree as the warm start.
`ILP_SECONDS=0` skips CP-SAT. Repeats reuse the parsed graph but rebuild all
construction state. See the root README for timing and report semantics.
