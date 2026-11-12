# Benchmark inputs

The 87-design benchmark suite is included as 81 JSON files and six ZIP archives.
Each archive contains one JSON file with the same name, without the final `.zip`.

## Run an input

The CPU and GPU runners restore a missing JSON automatically:

```bash
scripts/run_gpu.sh dataset/boolE/mul128.rename.json
scripts/run_cpu.sh dataset/boolE/mul128.rename.json 8
```

You can also pass the `.json.zip` path directly. Existing JSON files are reused
without being overwritten. Extracted files stay next to their archives for later
runs; the six large JSON files are ignored by Git.

The legacy `scripts/run.sh`, batch `scripts/run_all.sh`, CPU benchmark script,
and `verification/check_threads.py` also prepare compressed inputs automatically.
Extraction happens before the executable starts and is outside its reported
construction and pipeline times.

## Prepare files manually

Direct C++ executables and `verification/verify_solution.py` read JSON files.
Before using them on a compressed input, prepare that input or the entire suite:

```bash
python3 scripts/prepare_input.py dataset/boolE/mul128.rename.json
python3 scripts/prepare_input.py --dataset dataset
```

The helper uses Python's standard library. It checks the archive member name
and ZIP CRC, writes a temporary file, and publishes the JSON only after
extraction succeeds. Concurrent runners never observe a partially extracted
JSON from this helper. Leave enough disk space for the uncompressed inputs.

## Compressed BoolE inputs

All six archives were unpacked into a fresh directory and their SHA-256 hashes
matched the original JSON files byte for byte. Sizes below use MiB (1024² bytes).

| Input | JSON size (MiB) | ZIP size (MiB) |
|---|---:|---:|
| [mul112.rename.json](boolE/mul112.rename.json.zip) | 81.48 | 6.30 |
| [mul128.rename.json](boolE/mul128.rename.json.zip) | 106.94 | 8.28 |
| [mul80_map.rename.json](boolE/mul80_map.rename.json.zip) | 67.73 | 5.19 |
| [mul96_map.rename.json](boolE/mul96_map.rename.json.zip) | 97.81 | 7.45 |
| [mul112_map.rename.json](boolE/mul112_map.rename.json.zip) | 133.33 | 10.26 |
| [mul128_map.rename.json](boolE/mul128_map.rename.json.zip) | 174.77 | 13.42 |
| **Total** | **662.06** | **50.89** |

## Sources and licensing status

Jiaqi Yin provided the full 87-design benchmark collection to Yi-Hua Chung. The
`.rename.json` files are the versions used in XEG's recorded evaluation. The
links below identify the originating work and public reference collections;
they do not assert byte-for-byte identity with the supplied files.

| Local family | Designs | Originating work | Public reference | License / redistribution permission |
|---|---:|---|---|---|
| `boolE/` | 16 | [BoolE](https://arxiv.org/abs/2504.05577) | [SmoothE dataset](https://github.com/cornell-zhang/SmoothE/tree/main/dataset/boole) | Redistributed with permission from Jiaqi Yin, including all six large ZIP archives |
| `diospyros/` | 13 | [Diospyros](https://doi.org/10.1145/3445814.3446707) | [SmoothE dataset](https://github.com/cornell-zhang/SmoothE/tree/main/dataset/diospyros) | Apache-2.0; older artifact is BSD-3-Clause |
| `esyn/` | 14 | [E-Syn](https://doi.org/10.1145/3649329.3656246) | [SmoothE dataset](https://github.com/cornell-zhang/SmoothE/tree/main/dataset/esyn) | Apache-2.0 |
| `flexc/` | 14 | [FlexC](https://arxiv.org/abs/2309.09112) | [SmoothE dataset](https://github.com/cornell-zhang/SmoothE/tree/main/dataset/flexc) | Apache-2.0; older artifact is BSD-3-Clause |
| `impress/` | 3 | [IMpress](https://ieeexplore.ieee.org/document/9786123) | [SmoothE dataset](https://github.com/cornell-zhang/SmoothE/tree/main/dataset/impress) | Apache-2.0; older artifact is BSD-3-Clause |
| `maxsat/` | 6 | [MaxSAT](https://doi.org/10.1145/3669940.3707262) | [SmoothE dataset](https://github.com/cornell-zhang/SmoothE/tree/main/dataset/maxsat) | Apache-2.0; older artifact is BSD-3-Clause |
| `morphic/` | 7 | [E-morphic](https://arxiv.org/abs/2504.11574) | [SmoothE dataset](https://github.com/cornell-zhang/SmoothE/tree/main/dataset/emorphic) | Apache-2.0 |
| `rover/` | 9 | [Rover](https://ieeexplore.ieee.org/document/10579443) | [SmoothE dataset](https://github.com/cornell-zhang/SmoothE/tree/main/dataset/rover) | Apache-2.0; older artifact is BSD-3-Clause |
| `tensat/` | 5 | [Tensat](https://proceedings.mlsys.org/paper_files/paper/2021/hash/cc427d934a7f6c0663e5923f49eba531-Abstract.html) | [SmoothE dataset](https://github.com/cornell-zhang/SmoothE/tree/main/dataset/tensat) | Apache-2.0; older artifact is BSD-3-Clause |

The current [SmoothE repository](https://github.com/cornell-zhang/SmoothE)
publishes these benchmark families and declares
[Apache-2.0](https://github.com/cornell-zhang/SmoothE/blob/main/LICENSE) at its root.
Its dataset tree has no separate license file. The older
[ASPLOS artifact](https://github.com/yaohuicai/smoothe-artifact/tree/main/dataset)
contains an earlier subset and declares
[BSD-3-Clause](https://github.com/yaohuicai/smoothe-artifact/blob/main/LICENSE.txt).
These are distinct published versions, not interchangeable license statements.
Copies of their notices are retained in [licenses/](licenses/).

SmoothE states that GitHub file-size limits prevent it from including all
BoolE inputs: it includes ten and links the six large cases to
[SeaSkysz/eboost_dataset](https://huggingface.co/datasets/SeaSkysz/eboost_dataset/tree/main/boole).
The [e-boost README](https://github.com/Yu-Maryland/e-boost#-benchmarks) also
links that complete collection.

**Release status:** the 16 BoolE inputs, including the six large compressed
archives, are included with Jiaqi Yin's permission. Sources and declared
reference-repository licenses for the other eight families are listed above.
XEG's [Apache-2.0 license](../LICENSE) applies to XEG code and does not relicense
this external collection.

## Acknowledgments

The BoolE inputs in `boolE/`, including the six large compressed archives, are
redistributed with the permission of Jiaqi Yin, first author of
[BoolE: Exact Symbolic Reasoning via Boolean Equality Saturation](https://arxiv.org/abs/2504.05577).
We sincerely thank Jiaqi Yin for sharing the full benchmark collection and
allowing us to include the BoolE inputs.
