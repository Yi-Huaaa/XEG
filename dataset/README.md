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

Benchmark data retain their respective licenses. The repository's [Apache 2.0
license](../LICENSE) applies to XEG code.
