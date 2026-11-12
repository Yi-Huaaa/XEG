# GPU XEG: fixed-configuration comparison

87 designs; Block kernel / 64 threads per candidate / 512-bit bitmap / BatchThenDirect. No GPU tuning parameters were changed.

Construction times are per-design medians of three separate-process runs on the same node (NVIDIA RTX PRO 6000 Blackwell Server Edition; AMD EPYC 9B45). Native CUDA-event boundaries match the original code. Input parsing, allocation, candidate cleanup/checks, and CP-SAT are excluded from the native tree/DAG times. Added checks and transfers are shown separately and retained in the CSV.

Quality uses one fixed-configuration run per design, 360 seconds and 8 CP-SAT workers. Four quality cases run concurrently; their solver wall times are not an isolated performance benchmark. GPU construction and parallel timed solving are nondeterministic. Paper costs are reference values carried over from the accepted [CPU comparison](../../cpu/20261112/comparison.md), with different tuning/repetition methodology.

Four large BoolE solves (`mul80`, `mul96`, `mul112`, and `mul128`) were killed when the initial job exceeded its 128 GiB allocation while running concurrent solves. Only those four cases were rerun with a 256 GiB allocation, keeping the executable, inputs, parameters, 360-second limit, 8 solver workers, and four-case concurrency unchanged. The other 83 completed results were retained. Failed attempts are preserved in the local experiment records and are not counted as solver answers.

## Solution quality

`OPTIMAL` proves the best cost inside that run’s candidate space. `FEASIBLE` has an incumbent without a proof of optimality. `UNKNOWN` has no solver incumbent; the current pipeline can return its independently validated seed (`tree_fallback`). Baseline `INFEASIBLE` means that constructed candidate space has no legal solution. A dash is not a zero cost.

| Benchmark | Design | Paper XEG cost | Baseline GPU cost | Baseline status | Current GPU cost | Current solver status | Source |
|---|---|---:|---:|---|---:|---|---|
| BoolE | mul16 | 1057 | 1057 | OPTIMAL | 1057 | OPTIMAL | solver |
| BoolE | mul32 | 4167 | 4170 | OPTIMAL | 4199 | OPTIMAL | solver |
| BoolE | mul48 | 9437 | 9404 | OPTIMAL | 9404 | OPTIMAL | solver |
| BoolE | mul64 | 16755 | 16589 | OPTIMAL | 16528 | OPTIMAL | solver |
| BoolE | mul80 | 25889 | — | UNKNOWN | 26052 | FEASIBLE | solver |
| BoolE | mul96 | 37199 | — | UNKNOWN | 37661 | FEASIBLE | solver |
| BoolE | mul112 | 53745 | — | UNKNOWN | 54339 | FEASIBLE | solver |
| BoolE | mul128 | 81902 | — | INFEASIBLE | 86310 | FEASIBLE | solver |
| BoolE | mul16map | 1653 | 1656 | OPTIMAL | 1649 | OPTIMAL | solver |
| BoolE | mul32map | 7259 | — | INFEASIBLE | 7270 | OPTIMAL | solver |
| BoolE | mul48map | 16885 | — | INFEASIBLE | 16968 | OPTIMAL | solver |
| BoolE | mul64map | 30502 | — | INFEASIBLE | 30398 | OPTIMAL | solver |
| BoolE | mul80map | 48023 | — | INFEASIBLE | 48166 | OPTIMAL | solver |
| BoolE | mul96map | 69333 | — | INFEASIBLE | 69502 | OPTIMAL | solver |
| BoolE | mul112map | 93567 | — | INFEASIBLE | 94571 | OPTIMAL | solver |
| BoolE | mul128map | 140830 | — | INFEASIBLE | 123876 | OPTIMAL | solver |
| ESyn | mul3 | 160 | 179 | OPTIMAL | 160 | OPTIMAL | solver |
| ESyn | c7552 | 6220 | 6235 | OPTIMAL | 6236 | OPTIMAL | solver |
| ESyn | b12 | 2955 | 3056 | OPTIMAL | 2955 | OPTIMAL | solver |
| ESyn | i7 | 1909 | 1909 | OPTIMAL | 1909 | OPTIMAL | solver |
| ESyn | C432 | 553 | 562 | OPTIMAL | 553 | OPTIMAL | solver |
| ESyn | c2670 | 2201 | 2203 | OPTIMAL | 2201 | OPTIMAL | solver |
| ESyn | C5315 | 4837 | 4842 | OPTIMAL | 4841 | OPTIMAL | solver |
| ESyn | mul5 | 599 | 773 | OPTIMAL | 607 | FEASIBLE | solver |
| ESyn | qdiv | 5381 | 5381 | OPTIMAL | 5381 | OPTIMAL | solver |
| ESyn | max | 9160 | 9155 | OPTIMAL | 9172 | OPTIMAL | solver |
| ESyn | bar | 9113 | 9113 | OPTIMAL | 9113 | OPTIMAL | solver |
| ESyn | cavlc | 1886 | 1905 | FEASIBLE | 1889 | FEASIBLE | solver |
| ESyn | adder | 3447 | 3449 | OPTIMAL | 3447 | OPTIMAL | solver |
| ESyn | frg2 | 2808 | 2801 | OPTIMAL | 2804 | OPTIMAL | solver |
| E-morphic | sin | 14231 | 14222 | OPTIMAL | 14230 | OPTIMAL | solver |
| E-morphic | arbiter | 35585 | 35687 | OPTIMAL | 35609 | OPTIMAL | solver |
| E-morphic | square | 53653 | 53663 | OPTIMAL | 53664 | OPTIMAL | solver |
| E-morphic | multiplier | 77821 | 77821 | OPTIMAL | 77821 | OPTIMAL | solver |
| E-morphic | log2 | 84211 | 84226 | OPTIMAL | 84235 | OPTIMAL | solver |
| E-morphic | mem_ctrl | 132982 | 132982 | OPTIMAL | 132984 | OPTIMAL | solver |
| E-morphic | div | 159358 | — | UNKNOWN | 159328 | OPTIMAL | solver |
| IMpress | mul512 | 149213 | 149213 | OPTIMAL | 149213 | OPTIMAL | solver |
| IMpress | mul1024 | 495629 | 495629 | OPTIMAL | 495629 | OPTIMAL | solver |
| IMpress | mul2048 | 1305960 | 3493317 | OPTIMAL | 3538018 | OPTIMAL | solver |
| Tensat | vgg | 468 | 468 | OPTIMAL | 468 | OPTIMAL | solver |
| Tensat | resnet50 | 398 | 398 | OPTIMAL | 398 | OPTIMAL | solver |
| Tensat | bert | 57 | 63 | OPTIMAL | 63 | OPTIMAL | solver |
| Tensat | nasneta | 1208 | 1215 | OPTIMAL | 1208 | OPTIMAL | solver |
| Tensat | nasrnn | 35 | 35 | OPTIMAL | 35 | OPTIMAL | solver |
| Flexc | flexc_75 | 35 | 10029 | OPTIMAL | 35 | OPTIMAL | solver |
| Flexc | flexc_67 | 36 | 36 | OPTIMAL | 36 | OPTIMAL | solver |
| Flexc | flexc_db | 36 | 10030 | OPTIMAL | 36 | OPTIMAL | solver |
| Flexc | flexc_86 | 36 | 10030 | OPTIMAL | 36 | OPTIMAL | solver |
| Flexc | flexc_4a | 36 | 10030 | OPTIMAL | 36 | OPTIMAL | solver |
| Flexc | flexc_03 | 36 | 10030 | OPTIMAL | 36 | OPTIMAL | solver |
| Flexc | flexc_7b | 95 | 95 | OPTIMAL | 95 | OPTIMAL | solver |
| Flexc | flexc_ff | 134 | 134 | OPTIMAL | 134 | OPTIMAL | solver |
| Flexc | flexc_b8 | 136 | 136 | OPTIMAL | 136 | OPTIMAL | solver |
| Flexc | flexc_a0 | 136 | 136 | OPTIMAL | 136 | OPTIMAL | solver |
| Flexc | flexc_85 | 136 | 136 | OPTIMAL | 136 | OPTIMAL | solver |
| Flexc | flexc_5c | 136 | 136 | OPTIMAL | 136 | OPTIMAL | solver |
| Flexc | flexc_fd | 105 | 10099 | OPTIMAL | 105 | OPTIMAL | solver |
| Flexc | flexc_fe | 88 | 88 | OPTIMAL | 88 | OPTIMAL | solver |
| Rover | box5r | 1819 | 1819 | OPTIMAL | 1819 | OPTIMAL | solver |
| Rover | box3r | 1701 | 1701 | OPTIMAL | 1701 | OPTIMAL | solver |
| Rover | fir8t5r | 5936 | 6025 | OPTIMAL | 5936 | OPTIMAL | solver |
| Rover | fir8t6r | 5936 | 5936 | OPTIMAL | 5936 | OPTIMAL | solver |
| Rover | fir8t7r | 5936 | 6171 | OPTIMAL | 5936 | OPTIMAL | solver |
| Rover | box4r | 1635 | 1794 | OPTIMAL | 1762 | OPTIMAL | solver |
| Rover | fir8t8r | 5936 | 5936 | OPTIMAL | 5936 | OPTIMAL | solver |
| Rover | mcm378r | 1165 | 1165 | OPTIMAL | 1165 | OPTIMAL | solver |
| Rover | mcm379r | 1165 | 1050 | OPTIMAL | 1165 | OPTIMAL | solver |
| Diospyros | vaddr7 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | vmjmzr14 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | vvamr23 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | vpmr23 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | vmacr21 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | vmm22r21 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | vmm22r26 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | specr26 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | vcv22r36 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | vmm23r58 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | vcv33r168 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | drcr2r18 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Diospyros | drcr18 | 1 | 1 | OPTIMAL | 1 | OPTIMAL | solver |
| Maxsat | maxcut39 | 141 | 141 | OPTIMAL | 141 | OPTIMAL | solver |
| Maxsat | maxcut1 | 141 | 141 | OPTIMAL | 141 | OPTIMAL | solver |
| Maxsat | s2v1 | 141 | 141 | OPTIMAL | 141 | OPTIMAL | solver |
| Maxsat | s2v3 | 141 | 141 | OPTIMAL | 141 | OPTIMAL | solver |
| Maxsat | hamming6 | 65 | 65 | OPTIMAL | 65 | OPTIMAL | solver |
| Maxsat | johnson8 | 71 | 71 | OPTIMAL | 71 | OPTIMAL | solver |

## Tree construction runtime

Speedup = original native time / current native time; larger is faster. Summary rows weight each design equally.

| Benchmark | Design | Baseline ms (speedup) | Current ms (speedup) |
|---|---|---:|---:|
| BoolE | mul16 | 1.461 (1.00x) | 1.673 (0.87x) |
| BoolE | mul32 | 3.825 (1.00x) | 3.424 (1.12x) |
| BoolE | mul48 | 5.672 (1.00x) | 5.618 (1.01x) |
| BoolE | mul64 | 7.592 (1.00x) | 8.657 (0.88x) |
| BoolE | mul80 | 9.550 (1.00x) | 11.811 (0.81x) |
| BoolE | mul96 | 12.406 (1.00x) | 15.698 (0.79x) |
| BoolE | mul112 | 18.210 (1.00x) | 20.119 (0.91x) |
| BoolE | mul128 | 21.858 (1.00x) | 25.327 (0.86x) |
| BoolE | mul16map | 2.295 (1.00x) | 2.158 (1.06x) |
| BoolE | mul32map | 5.730 (1.00x) | 5.238 (1.09x) |
| BoolE | mul48map | 8.536 (1.00x) | 8.951 (0.95x) |
| BoolE | mul64map | 11.434 (1.00x) | 13.664 (0.84x) |
| BoolE | mul80map | 14.578 (1.00x) | 18.432 (0.79x) |
| BoolE | mul96map | 22.629 (1.00x) | 24.519 (0.92x) |
| BoolE | mul112map | 26.884 (1.00x) | 32.972 (0.82x) |
| BoolE | mul128map | 32.826 (1.00x) | 42.192 (0.78x) |
| **BoolE** | **Arithmetic mean speedup** | **1.00x** | **0.91x** |
| **BoolE** | **Geometric mean speedup** | **1.00x** | **0.90x** |
| ESyn | mul3 | 0.697 (1.00x) | 0.726 (0.96x) |
| ESyn | c7552 | 3.312 (1.00x) | 2.943 (1.13x) |
| ESyn | b12 | 3.848 (1.00x) | 3.376 (1.14x) |
| ESyn | i7 | 2.207 (1.00x) | 1.928 (1.14x) |
| ESyn | C432 | 0.952 (1.00x) | 1.059 (0.90x) |
| ESyn | c2670 | 4.330 (1.00x) | 3.780 (1.15x) |
| ESyn | C5315 | 3.784 (1.00x) | 3.422 (1.11x) |
| ESyn | mul5 | 1.167 (1.00x) | 1.302 (0.90x) |
| ESyn | qdiv | 5.153 (1.00x) | 4.432 (1.16x) |
| ESyn | max | 9.570 (1.00x) | 10.037 (0.95x) |
| ESyn | bar | 4.216 (1.00x) | 3.671 (1.15x) |
| ESyn | cavlc | 0.950 (1.00x) | 0.923 (1.03x) |
| ESyn | adder | 6.005 (1.00x) | 7.203 (0.83x) |
| ESyn | frg2 | 4.498 (1.00x) | 4.037 (1.11x) |
| **ESyn** | **Arithmetic mean speedup** | **1.00x** | **1.05x** |
| **ESyn** | **Geometric mean speedup** | **1.00x** | **1.04x** |
| E-morphic | sin | 5.245 (1.00x) | 5.076 (1.03x) |
| E-morphic | arbiter | 8.879 (1.00x) | 7.405 (1.20x) |
| E-morphic | square | 10.344 (1.00x) | 10.567 (0.98x) |
| E-morphic | multiplier | 11.245 (1.00x) | 9.785 (1.15x) |
| E-morphic | log2 | 16.002 (1.00x) | 18.417 (0.87x) |
| E-morphic | mem_ctrl | 38.999 (1.00x) | 42.152 (0.93x) |
| E-morphic | div | 150.088 (1.00x) | 157.214 (0.95x) |
| **E-morphic** | **Arithmetic mean speedup** | **1.00x** | **1.02x** |
| **E-morphic** | **Geometric mean speedup** | **1.00x** | **1.01x** |
| IMpress | mul512 | 0.607 (1.00x) | 0.500 (1.21x) |
| IMpress | mul1024 | 0.677 (1.00x) | 0.599 (1.13x) |
| IMpress | mul2048 | 0.690 (1.00x) | 0.619 (1.11x) |
| **IMpress** | **Arithmetic mean speedup** | **1.00x** | **1.15x** |
| **IMpress** | **Geometric mean speedup** | **1.00x** | **1.15x** |
| Tensat | vgg | 1.140 (1.00x) | 1.088 (1.05x) |
| Tensat | resnet50 | 2.035 (1.00x) | 1.896 (1.07x) |
| Tensat | bert | 0.692 (1.00x) | 0.598 (1.16x) |
| Tensat | nasneta | 2.623 (1.00x) | 3.000 (0.87x) |
| Tensat | nasrnn | 1.595 (1.00x) | 1.382 (1.15x) |
| **Tensat** | **Arithmetic mean speedup** | **1.00x** | **1.06x** |
| **Tensat** | **Geometric mean speedup** | **1.00x** | **1.06x** |
| Flexc | flexc_75 | 0.436 (1.00x) | 0.423 (1.03x) |
| Flexc | flexc_67 | 0.441 (1.00x) | 0.442 (1.00x) |
| Flexc | flexc_db | 0.439 (1.00x) | 0.426 (1.03x) |
| Flexc | flexc_86 | 0.440 (1.00x) | 0.430 (1.02x) |
| Flexc | flexc_4a | 0.406 (1.00x) | 0.431 (0.94x) |
| Flexc | flexc_03 | 0.446 (1.00x) | 0.422 (1.06x) |
| Flexc | flexc_7b | 0.466 (1.00x) | 0.447 (1.04x) |
| Flexc | flexc_ff | 0.538 (1.00x) | 0.497 (1.08x) |
| Flexc | flexc_b8 | 0.515 (1.00x) | 0.492 (1.05x) |
| Flexc | flexc_a0 | 0.541 (1.00x) | 0.492 (1.10x) |
| Flexc | flexc_85 | 0.535 (1.00x) | 0.489 (1.09x) |
| Flexc | flexc_5c | 0.545 (1.00x) | 0.492 (1.11x) |
| Flexc | flexc_fd | 0.536 (1.00x) | 0.488 (1.10x) |
| Flexc | flexc_fe | 0.525 (1.00x) | 0.500 (1.05x) |
| **Flexc** | **Arithmetic mean speedup** | **1.00x** | **1.05x** |
| **Flexc** | **Geometric mean speedup** | **1.00x** | **1.05x** |
| Rover | box5r | 0.312 (1.00x) | 0.258 (1.21x) |
| Rover | box3r | 0.308 (1.00x) | 0.264 (1.17x) |
| Rover | fir8t5r | 0.512 (1.00x) | 0.427 (1.20x) |
| Rover | fir8t6r | 0.543 (1.00x) | 0.428 (1.27x) |
| Rover | fir8t7r | 0.572 (1.00x) | 0.471 (1.21x) |
| Rover | box4r | 0.450 (1.00x) | 0.331 (1.36x) |
| Rover | fir8t8r | 0.589 (1.00x) | 0.464 (1.27x) |
| Rover | mcm378r | 0.717 (1.00x) | 0.232 (3.08x) |
| Rover | mcm379r | 0.692 (1.00x) | 0.240 (2.88x) |
| **Rover** | **Arithmetic mean speedup** | **1.00x** | **1.63x** |
| **Rover** | **Geometric mean speedup** | **1.00x** | **1.51x** |
| Diospyros | vaddr7 | 0.156 (1.00x) | 0.109 (1.44x) |
| Diospyros | vmjmzr14 | 0.160 (1.00x) | 0.122 (1.31x) |
| Diospyros | vvamr23 | 0.157 (1.00x) | 0.124 (1.27x) |
| Diospyros | vpmr23 | 0.162 (1.00x) | 0.127 (1.27x) |
| Diospyros | vmacr21 | 0.173 (1.00x) | 0.129 (1.35x) |
| Diospyros | vmm22r21 | 0.173 (1.00x) | 0.128 (1.35x) |
| Diospyros | vmm22r26 | 0.187 (1.00x) | 0.137 (1.36x) |
| Diospyros | specr26 | 0.190 (1.00x) | 0.136 (1.39x) |
| Diospyros | vcv22r36 | 0.172 (1.00x) | 0.124 (1.39x) |
| Diospyros | vmm23r58 | 0.195 (1.00x) | 0.138 (1.41x) |
| Diospyros | vcv33r168 | 0.211 (1.00x) | 0.136 (1.55x) |
| Diospyros | drcr2r18 | 0.737 (1.00x) | 0.146 (5.05x) |
| Diospyros | drcr18 | 1.675 (1.00x) | 0.156 (10.75x) |
| **Diospyros** | **Arithmetic mean speedup** | **1.00x** | **2.38x** |
| **Diospyros** | **Geometric mean speedup** | **1.00x** | **1.78x** |
| Maxsat | maxcut39 | 0.290 (1.00x) | 0.287 (1.01x) |
| Maxsat | maxcut1 | 0.289 (1.00x) | 0.286 (1.01x) |
| Maxsat | s2v1 | 0.305 (1.00x) | 0.300 (1.01x) |
| Maxsat | s2v3 | 0.306 (1.00x) | 0.311 (0.98x) |
| Maxsat | hamming6 | 0.302 (1.00x) | 0.277 (1.09x) |
| Maxsat | johnson8 | 0.289 (1.00x) | 0.281 (1.03x) |
| **Maxsat** | **Arithmetic mean speedup** | **1.00x** | **1.02x** |
| **Maxsat** | **Geometric mean speedup** | **1.00x** | **1.02x** |
| **Across 87 designs** | **Arithmetic mean speedup** | **1.00x** | **1.28x** |
| **Across 87 designs** | **Geometric mean speedup** | **1.00x** | **1.14x** |

## DAG construction runtime

Speedup = original native time / current native time; larger is faster. Summary rows weight each design equally.
The final column adds seed checking, candidate transfer, pruning, and candidate validation; it is separate from the native DAG timer.

| Benchmark | Design | Baseline ms (speedup) | Current ms (speedup) | Added checks/transfers ms |
|---|---|---:|---:|---:|
| BoolE | mul16 | 27.082 (1.00x) | 25.617 (1.06x) | 0.563 |
| BoolE | mul32 | 65.336 (1.00x) | 62.416 (1.05x) | 2.017 |
| BoolE | mul48 | 106.621 (1.00x) | 106.767 (1.00x) | 4.499 |
| BoolE | mul64 | 157.030 (1.00x) | 155.530 (1.01x) | 7.997 |
| BoolE | mul80 | 208.876 (1.00x) | 212.488 (0.98x) | 12.557 |
| BoolE | mul96 | 291.139 (1.00x) | 280.490 (1.04x) | 18.247 |
| BoolE | mul112 | 362.451 (1.00x) | 358.151 (1.01x) | 26.087 |
| BoolE | mul128 | 451.023 (1.00x) | 452.632 (1.00x) | 36.303 |
| BoolE | mul16map | 21.513 (1.00x) | 18.800 (1.14x) | 0.819 |
| BoolE | mul32map | 62.655 (1.00x) | 72.319 (0.87x) | 3.127 |
| BoolE | mul48map | 104.483 (1.00x) | 126.489 (0.83x) | 7.772 |
| BoolE | mul64map | 170.938 (1.00x) | 191.069 (0.89x) | 14.038 |
| BoolE | mul80map | 267.163 (1.00x) | 270.006 (0.99x) | 23.075 |
| BoolE | mul96map | 338.419 (1.00x) | 369.135 (0.92x) | 33.863 |
| BoolE | mul112map | 444.875 (1.00x) | 487.090 (0.91x) | 48.863 |
| BoolE | mul128map | 565.201 (1.00x) | 618.820 (0.91x) | 67.923 |
| **BoolE** | **Arithmetic mean speedup** | **1.00x** | **0.98x** | **—** |
| **BoolE** | **Geometric mean speedup** | **1.00x** | **0.97x** | **—** |
| ESyn | mul3 | 20.947 (1.00x) | 40.140 (0.52x) | 0.415 |
| ESyn | c7552 | 12.028 (1.00x) | 11.836 (1.02x) | 0.574 |
| ESyn | b12 | 23.990 (1.00x) | 24.881 (0.96x) | 0.573 |
| ESyn | i7 | 29.883 (1.00x) | 28.959 (1.03x) | 0.517 |
| ESyn | C432 | 49.081 (1.00x) | 44.262 (1.11x) | 0.551 |
| ESyn | c2670 | 43.392 (1.00x) | 41.974 (1.03x) | 0.636 |
| ESyn | C5315 | 22.300 (1.00x) | 22.223 (1.00x) | 0.743 |
| ESyn | mul5 | 48.457 (1.00x) | 38.791 (1.25x) | 0.586 |
| ESyn | qdiv | 37.800 (1.00x) | 37.157 (1.02x) | 0.667 |
| ESyn | max | 37.575 (1.00x) | 39.115 (0.96x) | 0.927 |
| ESyn | bar | 18.028 (1.00x) | 17.649 (1.02x) | 0.738 |
| ESyn | cavlc | 25.165 (1.00x) | 27.501 (0.92x) | 0.930 |
| ESyn | adder | 56.297 (1.00x) | 56.085 (1.00x) | 1.199 |
| ESyn | frg2 | 49.275 (1.00x) | 49.281 (1.00x) | 1.321 |
| **ESyn** | **Arithmetic mean speedup** | **1.00x** | **0.99x** | **—** |
| **ESyn** | **Geometric mean speedup** | **1.00x** | **0.97x** | **—** |
| E-morphic | sin | 29.674 (1.00x) | 33.730 (0.88x) | 1.177 |
| E-morphic | arbiter | 31.794 (1.00x) | 32.668 (0.97x) | 1.327 |
| E-morphic | square | 42.550 (1.00x) | 43.580 (0.98x) | 5.142 |
| E-morphic | multiplier | 53.280 (1.00x) | 54.867 (0.97x) | 4.735 |
| E-morphic | log2 | 114.400 (1.00x) | 118.350 (0.97x) | 9.227 |
| E-morphic | mem_ctrl | 144.635 (1.00x) | 147.765 (0.98x) | 9.904 |
| E-morphic | div | 508.381 (1.00x) | 514.091 (0.99x) | 10.046 |
| **E-morphic** | **Arithmetic mean speedup** | **1.00x** | **0.96x** | **—** |
| **E-morphic** | **Geometric mean speedup** | **1.00x** | **0.96x** | **—** |
| IMpress | mul512 | 14.262 (1.00x) | 10.924 (1.31x) | 0.239 |
| IMpress | mul1024 | 35.413 (1.00x) | 32.828 (1.08x) | 0.655 |
| IMpress | mul2048 | 61.129 (1.00x) | 63.582 (0.96x) | 1.212 |
| **IMpress** | **Arithmetic mean speedup** | **1.00x** | **1.12x** | **—** |
| **IMpress** | **Geometric mean speedup** | **1.00x** | **1.11x** | **—** |
| Tensat | vgg | 7.994 (1.00x) | 7.748 (1.03x) | 0.128 |
| Tensat | resnet50 | 6.817 (1.00x) | 6.840 (1.00x) | 0.173 |
| Tensat | bert | 68.856 (1.00x) | 69.529 (0.99x) | 0.611 |
| Tensat | nasneta | 64.699 (1.00x) | 66.300 (0.98x) | 1.056 |
| Tensat | nasrnn | 69.632 (1.00x) | 58.316 (1.19x) | 0.876 |
| **Tensat** | **Arithmetic mean speedup** | **1.00x** | **1.04x** | **—** |
| **Tensat** | **Geometric mean speedup** | **1.00x** | **1.03x** | **—** |
| Flexc | flexc_75 | 21.239 (1.00x) | 19.671 (1.08x) | 0.148 |
| Flexc | flexc_67 | 19.637 (1.00x) | 21.452 (0.92x) | 0.154 |
| Flexc | flexc_db | 19.428 (1.00x) | 23.638 (0.82x) | 0.158 |
| Flexc | flexc_86 | 19.464 (1.00x) | 23.633 (0.82x) | 0.159 |
| Flexc | flexc_4a | 21.390 (1.00x) | 23.648 (0.90x) | 0.159 |
| Flexc | flexc_03 | 19.417 (1.00x) | 23.612 (0.82x) | 0.155 |
| Flexc | flexc_7b | 60.981 (1.00x) | 58.659 (1.04x) | 0.226 |
| Flexc | flexc_ff | 46.723 (1.00x) | 49.159 (0.95x) | 0.241 |
| Flexc | flexc_b8 | 47.205 (1.00x) | 47.348 (1.00x) | 0.247 |
| Flexc | flexc_a0 | 47.204 (1.00x) | 47.377 (1.00x) | 0.247 |
| Flexc | flexc_85 | 49.426 (1.00x) | 47.299 (1.04x) | 0.240 |
| Flexc | flexc_5c | 47.168 (1.00x) | 47.540 (0.99x) | 0.253 |
| Flexc | flexc_fd | 68.759 (1.00x) | 66.188 (1.04x) | 0.216 |
| Flexc | flexc_fe | 84.008 (1.00x) | 66.861 (1.26x) | 0.276 |
| **Flexc** | **Arithmetic mean speedup** | **1.00x** | **0.98x** | **—** |
| **Flexc** | **Geometric mean speedup** | **1.00x** | **0.97x** | **—** |
| Rover | box5r | 6.007 (1.00x) | 5.842 (1.03x) | 0.126 |
| Rover | box3r | 10.060 (1.00x) | 9.868 (1.02x) | 0.197 |
| Rover | fir8t5r | 19.512 (1.00x) | 23.011 (0.85x) | 0.329 |
| Rover | fir8t6r | 29.845 (1.00x) | 35.622 (0.84x) | 0.403 |
| Rover | fir8t7r | 38.640 (1.00x) | 45.367 (0.85x) | 0.470 |
| Rover | box4r | 77.264 (1.00x) | 87.963 (0.88x) | 0.582 |
| Rover | fir8t8r | 52.767 (1.00x) | 52.615 (1.00x) | 0.479 |
| Rover | mcm378r | 241.607 (1.00x) | 296.073 (0.82x) | 0.543 |
| Rover | mcm379r | 349.454 (1.00x) | 311.277 (1.12x) | 0.570 |
| **Rover** | **Arithmetic mean speedup** | **1.00x** | **0.93x** | **—** |
| **Rover** | **Geometric mean speedup** | **1.00x** | **0.93x** | **—** |
| Diospyros | vaddr7 | 0.937 (1.00x) | 0.811 (1.15x) | 0.053 |
| Diospyros | vmjmzr14 | 1.668 (1.00x) | 1.548 (1.08x) | 0.066 |
| Diospyros | vvamr23 | 1.978 (1.00x) | 1.897 (1.04x) | 0.062 |
| Diospyros | vpmr23 | 1.978 (1.00x) | 1.888 (1.05x) | 0.059 |
| Diospyros | vmacr21 | 2.271 (1.00x) | 2.264 (1.00x) | 0.065 |
| Diospyros | vmm22r21 | 3.185 (1.00x) | 3.164 (1.01x) | 0.069 |
| Diospyros | vmm22r26 | 4.533 (1.00x) | 4.584 (0.99x) | 0.075 |
| Diospyros | specr26 | 4.563 (1.00x) | 4.577 (1.00x) | 0.076 |
| Diospyros | vcv22r36 | 3.855 (1.00x) | 3.843 (1.00x) | 0.075 |
| Diospyros | vmm23r58 | 16.920 (1.00x) | 17.718 (0.95x) | 0.145 |
| Diospyros | vcv33r168 | 159.813 (1.00x) | 165.177 (0.97x) | 0.800 |
| Diospyros | drcr2r18 | 373.762 (1.00x) | 510.325 (0.73x) | 0.773 |
| Diospyros | drcr18 | 946.601 (1.00x) | 971.153 (0.97x) | 1.130 |
| **Diospyros** | **Arithmetic mean speedup** | **1.00x** | **1.00x** | **—** |
| **Diospyros** | **Geometric mean speedup** | **1.00x** | **0.99x** | **—** |
| Maxsat | maxcut39 | 1.456 (1.00x) | 1.421 (1.03x) | 0.114 |
| Maxsat | maxcut1 | 1.479 (1.00x) | 1.436 (1.03x) | 0.119 |
| Maxsat | s2v1 | 1.627 (1.00x) | 1.633 (1.00x) | 0.142 |
| Maxsat | s2v3 | 1.654 (1.00x) | 1.633 (1.01x) | 0.143 |
| Maxsat | hamming6 | 1.441 (1.00x) | 1.499 (0.96x) | 0.137 |
| Maxsat | johnson8 | 1.460 (1.00x) | 1.512 (0.97x) | 0.137 |
| **Maxsat** | **Arithmetic mean speedup** | **1.00x** | **1.00x** | **—** |
| **Maxsat** | **Geometric mean speedup** | **1.00x** | **1.00x** | **—** |
| **Across 87 designs** | **Arithmetic mean speedup** | **1.00x** | **0.99x** | **—** |
| **Across 87 designs** | **Geometric mean speedup** | **1.00x** | **0.98x** | **—** |
