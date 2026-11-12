# Optional NVIDIA cuOpt backend

The standard CPU/GPU scripts use OR-Tools CP-SAT. The optional cuOpt C API
backend is exposed through `main/simple`; enabling it does not remove the
CP-SAT dependency. The recent CPU/GPU results use CP-SAT.

## Build

Install the NVIDIA `libcuopt` C API package following the
[official installation instructions](https://docs.nvidia.com/cuopt/user-guide/latest/install.html).
Activate the environment containing it, then locate the package:

```bash
CUOPT_ROOT="$(python -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "libcuopt")')"
```

`CUOPT_ROOT` must contain `include/cuopt/linear_programming/cuopt_c.h` and the
cuOpt shared library under `lib64/` or `lib/`. To use another installation,
set `CUOPT_ROOT` to that directory directly.

```bash
cmake -S . -B build-cuopt -DCMAKE_BUILD_TYPE=Release \
  -DXEG_ENABLE_CUOPT=ON -DCUOPT_ROOT="$CUOPT_ROOT"
cmake --build build-cuopt -j "$(nproc)"
```

The CMake finder adds the package's library directories and sibling CUDA
package libraries to the build runtime search path. See
[`FindcuOpt.cmake`](../cmake/FindcuOpt.cmake) for the lookup rules.

## Run

```text
build-cuopt/main/simple INPUT.json [KERNEL] [DAG_THREADS] [BITMAP_BITS] [ILP_BACKEND] [CUOPT_MODE]
                                  [--config FILE.json] [--design FAMILY/STEM]
```

| Setting | Value | Behavior |
|---|---:|---|
| `ILP_BACKEND` | `0` | CP-SAT (default) |
| `ILP_BACKEND` | `1` | cuOpt |
| `CUOPT_MODE` | `0` | Presolve enabled, no MIP start |
| `CUOPT_MODE` | `1` | Presolve disabled, GPU tree supplied as MIP start |

```bash
# cuOpt with presolve
build-cuopt/main/simple dataset/esyn/mul3.rename.json 2 64 512 1 0

# cuOpt with the GPU tree as a warm start
build-cuopt/main/simple dataset/esyn/mul3.rename.json 2 64 512 1 1
```

Both commands use the design profile's controller parameters, with explicit
Block/64/512 geometry. To change controller parameters, provide a custom
JSON file as described in the [configuration guide](../configs/gpu/README.md#write-your-own-settings).
`simple` uses a 360-second solver limit.

cuOpt retains its own level-constrained model and console output interface;
it does not use the CP-SAT report/fallback path. A feasible incumbent is not an
optimality proof, and the two cuOpt modes can have different costs and runtimes.
GPU candidate construction remains nondeterministic.

## Tests

Run tests with the cuOpt environment active and a GPU available:

```bash
ctest --test-dir build-cuopt --output-on-failure
```

The current CP-SAT validation report does not establish cuOpt correctness or
performance for a different installed cuOpt version. See the
[program reference](../main/README.md) and [results index](../results/README.md).
