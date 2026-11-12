# GPU XEG: measured design profiles

The [fixed-parameter comparison](comparison.md) measures the code changes with the original settings. This page separately evaluates the supplied [87-design profiles](../../../configs/gpu/87-designs.json).

Selection protocol: every design received a construction-only trial with `persistent_small_levels=1`; it was considered only with a measured DAG improvement above 10%. Designs whose fixed GPU cost exceeded the CPU reference or a valid original GPU result received six 60-second quality screens: bitmap batch 8, 16, 64, or 128, 2048-bit bitmap, and a 10-iteration bitmap direct limit. If the best screen still exceeded the lower reference cost by more than 3%, two additional screens used exact overlap with batch 32 and 4096-bit bitmap with batch 1. A lower-cost screen candidate (or an equal-cost OPTIMAL candidate that improves proof status or reduces observed solver time by more than 25%), optionally combined with the faster levelization setting, then received a full 360-second / 8-worker solve. For remaining gaps above 3%, up to two alternate profiles receive the full budget even when the 60-second screen has not beaten the 360-second fixed result. A profile is retained only if its validated full-budget result is no worse than the fixed run. Otherwise the original defaults and fixed result are retained.

For remaining CPU-reference gaps above 3% on large BoolE `mul112` and `mul128`, two additional full-budget trials test batch 256 with BatchEarlyStop and batch 1024 with a one-iteration direct limit. Both use minimum batch iterations 2, low-activity threshold 10%, and patience 1. A replacement must improve the validated full-budget cost.

A subsequent runtime pass preserves each selected insertion configuration and screens three levelization controllers: persistent entry/exit 2048/4096 with 128 threads, entry/exit 8192/16384 with 512 threads (both enter after one small frontier), and host-driven levelization. A candidate needs more than 10% DAG improvement in the isolated screen, a full-budget validated cost no worse than the previous selection, and more than 5% improvement in a fresh three-run median before replacement. Unchanged profiles retain their completed isolated measurements; CSV records both quality and timing sources.

These are **selected observed results**, with different numbers of trials per design. They are not an unbiased speed/quality trial, an exhaustive parameter optimum, or a guarantee of the same cost on the next run. GPU candidate construction and parallel timed solving remain nondeterministic. Selected timings are isolated medians of three runs; they also participate in the final controller selection, so this is not a held-out performance evaluation. Trial timings during concurrent solves are not used here.

## Solution quality

Paper and CPU values are references from the accepted CPU report. All selected GPU reports passed the independent original-JSON verifier. `OPTIMAL` applies within that run’s candidate graph; `FEASIBLE` lacks an optimality proof; `UNKNOWN` can return a valid `tree_fallback`. Original GPU dashes denote no incumbent, not zero cost.

| Benchmark | Design | Paper XEG | CPU 8t reference | Original GPU | Fixed GPU | Profiled GPU | Profiled status | Source |
|---|---|---:|---:|---:|---:|---:|---|---|
| BoolE | mul16 | 1057 | 1093 | 1057 | 1057 | 1057 | OPTIMAL | solver |
| BoolE | mul32 | 4167 | 4236 | 4170 | 4199 | 4165 | OPTIMAL | solver |
| BoolE | mul48 | 9437 | 9576 | 9404 | 9404 | 9404 | OPTIMAL | solver |
| BoolE | mul64 | 16755 | 17389 | 16589 | 16528 | 16528 | OPTIMAL | solver |
| BoolE | mul80 | 25889 | 25866 | — | 26052 | 26052 | FEASIBLE | solver |
| BoolE | mul96 | 37199 | 37171 | — | 37661 | 37661 | FEASIBLE | solver |
| BoolE | mul112 | 53745 | 51179 | — | 54339 | 53578 | FEASIBLE | solver |
| BoolE | mul128 | 81902 | 65934 | — | 86310 | 79129 | FEASIBLE | solver |
| BoolE | mul16map | 1653 | 1686 | 1656 | 1649 | 1649 | OPTIMAL | solver |
| BoolE | mul32map | 7259 | 7276 | — | 7270 | 7270 | OPTIMAL | solver |
| BoolE | mul48map | 16885 | 17288 | — | 16968 | 16968 | OPTIMAL | solver |
| BoolE | mul64map | 30502 | 30928 | — | 30398 | 30398 | OPTIMAL | solver |
| BoolE | mul80map | 48023 | 48059 | — | 48166 | 48166 | OPTIMAL | solver |
| BoolE | mul96map | 69333 | 69192 | — | 69502 | 69373 | OPTIMAL | solver |
| BoolE | mul112map | 93567 | 94285 | — | 94571 | 94571 | OPTIMAL | solver |
| BoolE | mul128map | 140830 | 126184 | — | 123876 | 123876 | OPTIMAL | solver |
| ESyn | mul3 | 160 | 153 | 179 | 160 | 154 | OPTIMAL | solver |
| ESyn | c7552 | 6220 | 6213 | 6235 | 6236 | 6198 | OPTIMAL | solver |
| ESyn | b12 | 2955 | 2918 | 3056 | 2955 | 2920 | OPTIMAL | solver |
| ESyn | i7 | 1909 | 1925 | 1909 | 1909 | 1909 | OPTIMAL | solver |
| ESyn | C432 | 553 | 556 | 562 | 553 | 553 | OPTIMAL | solver |
| ESyn | c2670 | 2201 | 2210 | 2203 | 2201 | 2201 | OPTIMAL | solver |
| ESyn | C5315 | 4837 | 4755 | 4842 | 4841 | 4818 | OPTIMAL | solver |
| ESyn | mul5 | 599 | 600 | 773 | 607 | 601 | FEASIBLE | solver |
| ESyn | qdiv | 5381 | 5421 | 5381 | 5381 | 5381 | OPTIMAL | solver |
| ESyn | max | 9160 | 9015 | 9155 | 9172 | 9145 | OPTIMAL | solver |
| ESyn | bar | 9113 | 9157 | 9113 | 9113 | 9113 | OPTIMAL | solver |
| ESyn | cavlc | 1886 | 1894 | 1905 | 1889 | 1889 | FEASIBLE | solver |
| ESyn | adder | 3447 | 3445 | 3449 | 3447 | 3445 | OPTIMAL | solver |
| ESyn | frg2 | 2808 | 2770 | 2801 | 2804 | 2804 | OPTIMAL | solver |
| E-morphic | sin | 14231 | 14102 | 14222 | 14230 | 14230 | OPTIMAL | solver |
| E-morphic | arbiter | 35585 | 35583 | 35687 | 35609 | 35587 | OPTIMAL | solver |
| E-morphic | square | 53653 | 53391 | 53663 | 53664 | 53655 | OPTIMAL | solver |
| E-morphic | multiplier | 77821 | 77821 | 77821 | 77821 | 77821 | OPTIMAL | solver |
| E-morphic | log2 | 84211 | 84122 | 84226 | 84235 | 84235 | OPTIMAL | solver |
| E-morphic | mem_ctrl | 132982 | 132982 | 132982 | 132984 | 132984 | OPTIMAL | solver |
| E-morphic | div | 159358 | 159328 | — | 159328 | 159328 | OPTIMAL | solver |
| IMpress | mul512 | 149213 | 149213 | 149213 | 149213 | 149213 | OPTIMAL | solver |
| IMpress | mul1024 | 495629 | 495629 | 495629 | 495629 | 495629 | OPTIMAL | solver |
| IMpress | mul2048 | 1305960 | 3671672 | 3493317 | 3538018 | 3242994 | OPTIMAL | solver |
| Tensat | vgg | 468 | 468 | 468 | 468 | 468 | OPTIMAL | solver |
| Tensat | resnet50 | 398 | 398 | 398 | 398 | 398 | OPTIMAL | solver |
| Tensat | bert | 57 | 57 | 63 | 63 | 58 | OPTIMAL | solver |
| Tensat | nasneta | 1208 | 1208 | 1215 | 1208 | 1208 | OPTIMAL | solver |
| Tensat | nasrnn | 35 | 35 | 35 | 35 | 35 | OPTIMAL | solver |
| Flexc | flexc_75 | 35 | 35 | 10029 | 35 | 35 | OPTIMAL | solver |
| Flexc | flexc_67 | 36 | 36 | 36 | 36 | 36 | OPTIMAL | solver |
| Flexc | flexc_db | 36 | 36 | 10030 | 36 | 36 | OPTIMAL | solver |
| Flexc | flexc_86 | 36 | 36 | 10030 | 36 | 36 | OPTIMAL | solver |
| Flexc | flexc_4a | 36 | 36 | 10030 | 36 | 36 | OPTIMAL | solver |
| Flexc | flexc_03 | 36 | 36 | 10030 | 36 | 36 | OPTIMAL | solver |
| Flexc | flexc_7b | 95 | 95 | 95 | 95 | 95 | OPTIMAL | solver |
| Flexc | flexc_ff | 134 | 134 | 134 | 134 | 134 | OPTIMAL | solver |
| Flexc | flexc_b8 | 136 | 136 | 136 | 136 | 136 | OPTIMAL | solver |
| Flexc | flexc_a0 | 136 | 136 | 136 | 136 | 136 | OPTIMAL | solver |
| Flexc | flexc_85 | 136 | 136 | 136 | 136 | 136 | OPTIMAL | solver |
| Flexc | flexc_5c | 136 | 136 | 136 | 136 | 136 | OPTIMAL | solver |
| Flexc | flexc_fd | 105 | 105 | 10099 | 105 | 105 | OPTIMAL | solver |
| Flexc | flexc_fe | 88 | 88 | 88 | 88 | 88 | OPTIMAL | solver |
| Rover | box5r | 1819 | 1819 | 1819 | 1819 | 1819 | OPTIMAL | solver |
| Rover | box3r | 1701 | 1701 | 1701 | 1701 | 1701 | OPTIMAL | solver |
| Rover | fir8t5r | 5936 | 5936 | 6025 | 5936 | 5936 | OPTIMAL | solver |
| Rover | fir8t6r | 5936 | 5936 | 5936 | 5936 | 5936 | OPTIMAL | solver |
| Rover | fir8t7r | 5936 | 5936 | 6171 | 5936 | 5936 | OPTIMAL | solver |
| Rover | box4r | 1635 | 1635 | 1794 | 1762 | 1635 | FEASIBLE | solver |
| Rover | fir8t8r | 5936 | 5936 | 5936 | 5936 | 5936 | OPTIMAL | solver |
| Rover | mcm378r | 1165 | 1165 | 1165 | 1165 | 1165 | OPTIMAL | solver |
| Rover | mcm379r | 1165 | 1165 | 1050 | 1165 | 1165 | OPTIMAL | solver |
| Diospyros | vaddr7 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | vmjmzr14 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | vvamr23 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | vpmr23 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | vmacr21 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | vmm22r21 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | vmm22r26 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | specr26 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | vcv22r36 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | vmm23r58 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | vcv33r168 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | drcr2r18 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Diospyros | drcr18 | 1 | 1 | 1 | 1 | 1 | OPTIMAL | solver |
| Maxsat | maxcut39 | 141 | 141 | 141 | 141 | 141 | OPTIMAL | solver |
| Maxsat | maxcut1 | 141 | 141 | 141 | 141 | 141 | OPTIMAL | solver |
| Maxsat | s2v1 | 141 | 141 | 141 | 141 | 141 | OPTIMAL | solver |
| Maxsat | s2v3 | 141 | 141 | 141 | 141 | 141 | OPTIMAL | solver |
| Maxsat | hamming6 | 65 | 65 | 65 | 65 | 65 | OPTIMAL | solver |
| Maxsat | johnson8 | 71 | 71 | 71 | 71 | 71 | OPTIMAL | solver |

## Tree construction

Native CUDA-event times in milliseconds; parsing, allocation, transfers/checks outside the original timer, and solving are excluded. Speedups use the original GPU as 1.00x. Each design has equal weight in mean rows. The fixed and profiled runs retain identical timing boundaries.

| Benchmark | Design | Original GPU | Fixed GPU | Profiled GPU |
|---|---|---:|---:|---:|
| BoolE | mul16 | 1.461 (1.00x) | 1.673 (0.87x) | 1.636 (0.89x) |
| BoolE | mul32 | 3.825 (1.00x) | 3.424 (1.12x) | 3.468 (1.10x) |
| BoolE | mul48 | 5.672 (1.00x) | 5.618 (1.01x) | 5.460 (1.04x) |
| BoolE | mul64 | 7.592 (1.00x) | 8.657 (0.88x) | 8.633 (0.88x) |
| BoolE | mul80 | 9.550 (1.00x) | 11.811 (0.81x) | 11.905 (0.80x) |
| BoolE | mul96 | 12.406 (1.00x) | 15.698 (0.79x) | 15.716 (0.79x) |
| BoolE | mul112 | 18.210 (1.00x) | 20.119 (0.91x) | 20.122 (0.90x) |
| BoolE | mul128 | 21.858 (1.00x) | 25.327 (0.86x) | 25.376 (0.86x) |
| BoolE | mul16map | 2.295 (1.00x) | 2.158 (1.06x) | 2.186 (1.05x) |
| BoolE | mul32map | 5.730 (1.00x) | 5.238 (1.09x) | 5.228 (1.10x) |
| BoolE | mul48map | 8.536 (1.00x) | 8.951 (0.95x) | 8.918 (0.96x) |
| BoolE | mul64map | 11.434 (1.00x) | 13.664 (0.84x) | 13.781 (0.83x) |
| BoolE | mul80map | 14.578 (1.00x) | 18.432 (0.79x) | 18.398 (0.79x) |
| BoolE | mul96map | 22.629 (1.00x) | 24.519 (0.92x) | 24.510 (0.92x) |
| BoolE | mul112map | 26.884 (1.00x) | 32.972 (0.82x) | 32.821 (0.82x) |
| BoolE | mul128map | 32.826 (1.00x) | 42.192 (0.78x) | 41.911 (0.78x) |
| **BoolE** | **Arithmetic mean speedup** | **1.00x** | **0.91x** | **0.91x** |
| **BoolE** | **Geometric mean speedup** | **1.00x** | **0.90x** | **0.90x** |
| ESyn | mul3 | 0.697 (1.00x) | 0.726 (0.96x) | 0.712 (0.98x) |
| ESyn | c7552 | 3.312 (1.00x) | 2.943 (1.13x) | 2.932 (1.13x) |
| ESyn | b12 | 3.848 (1.00x) | 3.376 (1.14x) | 3.405 (1.13x) |
| ESyn | i7 | 2.207 (1.00x) | 1.928 (1.14x) | 1.912 (1.15x) |
| ESyn | C432 | 0.952 (1.00x) | 1.059 (0.90x) | 1.055 (0.90x) |
| ESyn | c2670 | 4.330 (1.00x) | 3.780 (1.15x) | 3.756 (1.15x) |
| ESyn | C5315 | 3.784 (1.00x) | 3.422 (1.11x) | 3.476 (1.09x) |
| ESyn | mul5 | 1.167 (1.00x) | 1.302 (0.90x) | 1.325 (0.88x) |
| ESyn | qdiv | 5.153 (1.00x) | 4.432 (1.16x) | 4.387 (1.17x) |
| ESyn | max | 9.570 (1.00x) | 10.037 (0.95x) | 10.016 (0.96x) |
| ESyn | bar | 4.216 (1.00x) | 3.671 (1.15x) | 3.591 (1.17x) |
| ESyn | cavlc | 0.950 (1.00x) | 0.923 (1.03x) | 0.928 (1.02x) |
| ESyn | adder | 6.005 (1.00x) | 7.203 (0.83x) | 7.130 (0.84x) |
| ESyn | frg2 | 4.498 (1.00x) | 4.037 (1.11x) | 3.853 (1.17x) |
| **ESyn** | **Arithmetic mean speedup** | **1.00x** | **1.05x** | **1.05x** |
| **ESyn** | **Geometric mean speedup** | **1.00x** | **1.04x** | **1.05x** |
| E-morphic | sin | 5.245 (1.00x) | 5.076 (1.03x) | 4.979 (1.05x) |
| E-morphic | arbiter | 8.879 (1.00x) | 7.405 (1.20x) | 7.677 (1.16x) |
| E-morphic | square | 10.344 (1.00x) | 10.567 (0.98x) | 10.327 (1.00x) |
| E-morphic | multiplier | 11.245 (1.00x) | 9.785 (1.15x) | 9.701 (1.16x) |
| E-morphic | log2 | 16.002 (1.00x) | 18.417 (0.87x) | 18.404 (0.87x) |
| E-morphic | mem_ctrl | 38.999 (1.00x) | 42.152 (0.93x) | 41.939 (0.93x) |
| E-morphic | div | 150.088 (1.00x) | 157.214 (0.95x) | 174.822 (0.86x) |
| **E-morphic** | **Arithmetic mean speedup** | **1.00x** | **1.02x** | **1.00x** |
| **E-morphic** | **Geometric mean speedup** | **1.00x** | **1.01x** | **1.00x** |
| IMpress | mul512 | 0.607 (1.00x) | 0.500 (1.21x) | 0.505 (1.20x) |
| IMpress | mul1024 | 0.677 (1.00x) | 0.599 (1.13x) | 0.585 (1.16x) |
| IMpress | mul2048 | 0.690 (1.00x) | 0.619 (1.11x) | 0.612 (1.13x) |
| **IMpress** | **Arithmetic mean speedup** | **1.00x** | **1.15x** | **1.16x** |
| **IMpress** | **Geometric mean speedup** | **1.00x** | **1.15x** | **1.16x** |
| Tensat | vgg | 1.140 (1.00x) | 1.088 (1.05x) | 1.064 (1.07x) |
| Tensat | resnet50 | 2.035 (1.00x) | 1.896 (1.07x) | 1.946 (1.05x) |
| Tensat | bert | 0.692 (1.00x) | 0.598 (1.16x) | 0.602 (1.15x) |
| Tensat | nasneta | 2.623 (1.00x) | 3.000 (0.87x) | 2.996 (0.88x) |
| Tensat | nasrnn | 1.595 (1.00x) | 1.382 (1.15x) | 1.378 (1.16x) |
| **Tensat** | **Arithmetic mean speedup** | **1.00x** | **1.06x** | **1.06x** |
| **Tensat** | **Geometric mean speedup** | **1.00x** | **1.06x** | **1.05x** |
| Flexc | flexc_75 | 0.436 (1.00x) | 0.423 (1.03x) | 0.404 (1.08x) |
| Flexc | flexc_67 | 0.441 (1.00x) | 0.442 (1.00x) | 0.427 (1.03x) |
| Flexc | flexc_db | 0.439 (1.00x) | 0.426 (1.03x) | 0.412 (1.06x) |
| Flexc | flexc_86 | 0.440 (1.00x) | 0.430 (1.02x) | 0.426 (1.03x) |
| Flexc | flexc_4a | 0.406 (1.00x) | 0.431 (0.94x) | 0.420 (0.97x) |
| Flexc | flexc_03 | 0.446 (1.00x) | 0.422 (1.06x) | 0.413 (1.08x) |
| Flexc | flexc_7b | 0.466 (1.00x) | 0.447 (1.04x) | 0.437 (1.06x) |
| Flexc | flexc_ff | 0.538 (1.00x) | 0.497 (1.08x) | 0.492 (1.09x) |
| Flexc | flexc_b8 | 0.515 (1.00x) | 0.492 (1.05x) | 0.501 (1.03x) |
| Flexc | flexc_a0 | 0.541 (1.00x) | 0.492 (1.10x) | 0.488 (1.11x) |
| Flexc | flexc_85 | 0.535 (1.00x) | 0.489 (1.09x) | 0.489 (1.09x) |
| Flexc | flexc_5c | 0.545 (1.00x) | 0.492 (1.11x) | 0.501 (1.09x) |
| Flexc | flexc_fd | 0.536 (1.00x) | 0.488 (1.10x) | 0.480 (1.12x) |
| Flexc | flexc_fe | 0.525 (1.00x) | 0.500 (1.05x) | 0.498 (1.05x) |
| **Flexc** | **Arithmetic mean speedup** | **1.00x** | **1.05x** | **1.06x** |
| **Flexc** | **Geometric mean speedup** | **1.00x** | **1.05x** | **1.06x** |
| Rover | box5r | 0.312 (1.00x) | 0.258 (1.21x) | 0.258 (1.21x) |
| Rover | box3r | 0.308 (1.00x) | 0.264 (1.17x) | 0.262 (1.17x) |
| Rover | fir8t5r | 0.512 (1.00x) | 0.427 (1.20x) | 0.429 (1.19x) |
| Rover | fir8t6r | 0.543 (1.00x) | 0.428 (1.27x) | 0.428 (1.27x) |
| Rover | fir8t7r | 0.572 (1.00x) | 0.471 (1.21x) | 0.473 (1.21x) |
| Rover | box4r | 0.450 (1.00x) | 0.331 (1.36x) | 0.330 (1.36x) |
| Rover | fir8t8r | 0.589 (1.00x) | 0.464 (1.27x) | 0.469 (1.26x) |
| Rover | mcm378r | 0.717 (1.00x) | 0.232 (3.08x) | 0.236 (3.03x) |
| Rover | mcm379r | 0.692 (1.00x) | 0.240 (2.88x) | 0.241 (2.87x) |
| **Rover** | **Arithmetic mean speedup** | **1.00x** | **1.63x** | **1.62x** |
| **Rover** | **Geometric mean speedup** | **1.00x** | **1.51x** | **1.50x** |
| Diospyros | vaddr7 | 0.156 (1.00x) | 0.109 (1.44x) | 0.109 (1.44x) |
| Diospyros | vmjmzr14 | 0.160 (1.00x) | 0.122 (1.31x) | 0.127 (1.26x) |
| Diospyros | vvamr23 | 0.157 (1.00x) | 0.124 (1.27x) | 0.126 (1.24x) |
| Diospyros | vpmr23 | 0.162 (1.00x) | 0.127 (1.27x) | 0.127 (1.28x) |
| Diospyros | vmacr21 | 0.173 (1.00x) | 0.129 (1.35x) | 0.124 (1.40x) |
| Diospyros | vmm22r21 | 0.173 (1.00x) | 0.128 (1.35x) | 0.123 (1.40x) |
| Diospyros | vmm22r26 | 0.187 (1.00x) | 0.137 (1.36x) | 0.136 (1.37x) |
| Diospyros | specr26 | 0.190 (1.00x) | 0.136 (1.39x) | 0.136 (1.39x) |
| Diospyros | vcv22r36 | 0.172 (1.00x) | 0.124 (1.39x) | 0.121 (1.42x) |
| Diospyros | vmm23r58 | 0.195 (1.00x) | 0.138 (1.41x) | 0.148 (1.31x) |
| Diospyros | vcv33r168 | 0.211 (1.00x) | 0.136 (1.55x) | 0.139 (1.52x) |
| Diospyros | drcr2r18 | 0.737 (1.00x) | 0.146 (5.05x) | 0.144 (5.12x) |
| Diospyros | drcr18 | 1.675 (1.00x) | 0.156 (10.75x) | 0.154 (10.88x) |
| **Diospyros** | **Arithmetic mean speedup** | **1.00x** | **2.38x** | **2.39x** |
| **Diospyros** | **Geometric mean speedup** | **1.00x** | **1.78x** | **1.77x** |
| Maxsat | maxcut39 | 0.290 (1.00x) | 0.287 (1.01x) | 0.287 (1.01x) |
| Maxsat | maxcut1 | 0.289 (1.00x) | 0.286 (1.01x) | 0.287 (1.01x) |
| Maxsat | s2v1 | 0.305 (1.00x) | 0.300 (1.01x) | 0.302 (1.01x) |
| Maxsat | s2v3 | 0.306 (1.00x) | 0.311 (0.98x) | 0.312 (0.98x) |
| Maxsat | hamming6 | 0.302 (1.00x) | 0.277 (1.09x) | 0.280 (1.08x) |
| Maxsat | johnson8 | 0.289 (1.00x) | 0.281 (1.03x) | 0.280 (1.03x) |
| **Maxsat** | **Arithmetic mean speedup** | **1.00x** | **1.02x** | **1.02x** |
| **Maxsat** | **Geometric mean speedup** | **1.00x** | **1.02x** | **1.02x** |
| **Across 87 designs** | **Arithmetic mean speedup** | **1.00x** | **1.28x** | **1.28x** |
| **Across 87 designs** | **Geometric mean speedup** | **1.00x** | **1.14x** | **1.14x** |

## DAG construction

Native CUDA-event times in milliseconds; parsing, allocation, transfers/checks outside the original timer, and solving are excluded. Speedups use the original GPU as 1.00x. Each design has equal weight in mean rows. The fixed and profiled runs retain identical timing boundaries.
The final column includes seed checking and candidate copying, pruning, and checking; it is additional to the native construction times.

| Benchmark | Design | Original GPU | Fixed GPU | Profiled GPU | Added checks/transfers |
|---|---|---:|---:|---:|---:|
| BoolE | mul16 | 27.082 (1.00x) | 25.617 (1.06x) | 25.584 (1.06x) | 0.552 |
| BoolE | mul32 | 65.336 (1.00x) | 62.416 (1.05x) | 91.034 (0.72x) | 1.907 |
| BoolE | mul48 | 106.621 (1.00x) | 106.767 (1.00x) | 106.742 (1.00x) | 4.517 |
| BoolE | mul64 | 157.030 (1.00x) | 155.530 (1.01x) | 155.084 (1.01x) | 8.015 |
| BoolE | mul80 | 208.876 (1.00x) | 212.488 (0.98x) | 212.251 (0.98x) | 12.805 |
| BoolE | mul96 | 291.139 (1.00x) | 280.490 (1.04x) | 277.544 (1.05x) | 18.552 |
| BoolE | mul112 | 362.451 (1.00x) | 358.151 (1.01x) | 286.422 (1.27x) | 26.794 |
| BoolE | mul128 | 451.023 (1.00x) | 452.632 (1.00x) | 72.472 (6.22x) | 36.949 |
| BoolE | mul16map | 21.513 (1.00x) | 18.800 (1.14x) | 18.828 (1.14x) | 0.812 |
| BoolE | mul32map | 62.655 (1.00x) | 72.319 (0.87x) | 71.339 (0.88x) | 3.275 |
| BoolE | mul48map | 104.483 (1.00x) | 126.489 (0.83x) | 125.793 (0.83x) | 7.711 |
| BoolE | mul64map | 170.938 (1.00x) | 191.069 (0.89x) | 190.657 (0.90x) | 14.266 |
| BoolE | mul80map | 267.163 (1.00x) | 270.006 (0.99x) | 270.177 (0.99x) | 22.890 |
| BoolE | mul96map | 338.419 (1.00x) | 369.135 (0.92x) | 855.133 (0.40x) | 34.396 |
| BoolE | mul112map | 444.875 (1.00x) | 487.090 (0.91x) | 487.514 (0.91x) | 48.811 |
| BoolE | mul128map | 565.201 (1.00x) | 618.820 (0.91x) | 620.329 (0.91x) | 67.846 |
| **BoolE** | **Arithmetic mean speedup** | **1.00x** | **0.98x** | **1.27x** | **—** |
| **BoolE** | **Geometric mean speedup** | **1.00x** | **0.97x** | **1.03x** | **—** |
| ESyn | mul3 | 20.947 (1.00x) | 40.140 (0.52x) | 51.100 (0.41x) | 0.413 |
| ESyn | c7552 | 12.028 (1.00x) | 11.836 (1.02x) | 15.216 (0.79x) | 0.572 |
| ESyn | b12 | 23.990 (1.00x) | 24.881 (0.96x) | 21.565 (1.11x) | 0.561 |
| ESyn | i7 | 29.883 (1.00x) | 28.959 (1.03x) | 28.912 (1.03x) | 0.512 |
| ESyn | C432 | 49.081 (1.00x) | 44.262 (1.11x) | 45.246 (1.08x) | 0.532 |
| ESyn | c2670 | 43.392 (1.00x) | 41.974 (1.03x) | 41.952 (1.03x) | 0.643 |
| ESyn | C5315 | 22.300 (1.00x) | 22.223 (1.00x) | 28.087 (0.79x) | 0.739 |
| ESyn | mul5 | 48.457 (1.00x) | 38.791 (1.25x) | 43.076 (1.12x) | 0.577 |
| ESyn | qdiv | 37.800 (1.00x) | 37.157 (1.02x) | 37.176 (1.02x) | 0.675 |
| ESyn | max | 37.575 (1.00x) | 39.115 (0.96x) | 47.917 (0.78x) | 0.921 |
| ESyn | bar | 18.028 (1.00x) | 17.649 (1.02x) | 15.323 (1.18x) | 0.729 |
| ESyn | cavlc | 25.165 (1.00x) | 27.501 (0.92x) | 27.616 (0.91x) | 0.936 |
| ESyn | adder | 56.297 (1.00x) | 56.085 (1.00x) | 53.085 (1.06x) | 1.205 |
| ESyn | frg2 | 49.275 (1.00x) | 49.281 (1.00x) | 49.611 (0.99x) | 1.319 |
| **ESyn** | **Arithmetic mean speedup** | **1.00x** | **0.99x** | **0.95x** | **—** |
| **ESyn** | **Geometric mean speedup** | **1.00x** | **0.97x** | **0.93x** | **—** |
| E-morphic | sin | 29.674 (1.00x) | 33.730 (0.88x) | 33.726 (0.88x) | 1.179 |
| E-morphic | arbiter | 31.794 (1.00x) | 32.668 (0.97x) | 32.522 (0.98x) | 1.313 |
| E-morphic | square | 42.550 (1.00x) | 43.580 (0.98x) | 88.474 (0.48x) | 5.568 |
| E-morphic | multiplier | 53.280 (1.00x) | 54.867 (0.97x) | 54.770 (0.97x) | 5.036 |
| E-morphic | log2 | 114.400 (1.00x) | 118.350 (0.97x) | 118.495 (0.97x) | 9.339 |
| E-morphic | mem_ctrl | 144.635 (1.00x) | 147.765 (0.98x) | 196.416 (0.74x) | 10.692 |
| E-morphic | div | 508.381 (1.00x) | 514.091 (0.99x) | 514.214 (0.99x) | 9.930 |
| **E-morphic** | **Arithmetic mean speedup** | **1.00x** | **0.96x** | **0.86x** | **—** |
| **E-morphic** | **Geometric mean speedup** | **1.00x** | **0.96x** | **0.83x** | **—** |
| IMpress | mul512 | 14.262 (1.00x) | 10.924 (1.31x) | 9.724 (1.47x) | 0.234 |
| IMpress | mul1024 | 35.413 (1.00x) | 32.828 (1.08x) | 32.782 (1.08x) | 0.644 |
| IMpress | mul2048 | 61.129 (1.00x) | 63.582 (0.96x) | 144.326 (0.42x) | 1.224 |
| **IMpress** | **Arithmetic mean speedup** | **1.00x** | **1.12x** | **0.99x** | **—** |
| **IMpress** | **Geometric mean speedup** | **1.00x** | **1.11x** | **0.88x** | **—** |
| Tensat | vgg | 7.994 (1.00x) | 7.748 (1.03x) | 6.640 (1.20x) | 0.134 |
| Tensat | resnet50 | 6.817 (1.00x) | 6.840 (1.00x) | 5.799 (1.18x) | 0.172 |
| Tensat | bert | 68.856 (1.00x) | 69.529 (0.99x) | 87.083 (0.79x) | 0.634 |
| Tensat | nasneta | 64.699 (1.00x) | 66.300 (0.98x) | 66.334 (0.98x) | 1.152 |
| Tensat | nasrnn | 69.632 (1.00x) | 58.316 (1.19x) | 59.021 (1.18x) | 0.901 |
| **Tensat** | **Arithmetic mean speedup** | **1.00x** | **1.04x** | **1.07x** | **—** |
| **Tensat** | **Geometric mean speedup** | **1.00x** | **1.03x** | **1.05x** | **—** |
| Flexc | flexc_75 | 21.239 (1.00x) | 19.671 (1.08x) | 19.637 (1.08x) | 0.146 |
| Flexc | flexc_67 | 19.637 (1.00x) | 21.452 (0.92x) | 21.467 (0.91x) | 0.145 |
| Flexc | flexc_db | 19.428 (1.00x) | 23.638 (0.82x) | 18.538 (1.05x) | 0.154 |
| Flexc | flexc_86 | 19.464 (1.00x) | 23.633 (0.82x) | 23.650 (0.82x) | 0.155 |
| Flexc | flexc_4a | 21.390 (1.00x) | 23.648 (0.90x) | 18.529 (1.15x) | 0.157 |
| Flexc | flexc_03 | 19.417 (1.00x) | 23.612 (0.82x) | 23.663 (0.82x) | 0.152 |
| Flexc | flexc_7b | 60.981 (1.00x) | 58.659 (1.04x) | 58.673 (1.04x) | 0.216 |
| Flexc | flexc_ff | 46.723 (1.00x) | 49.159 (0.95x) | 49.224 (0.95x) | 0.241 |
| Flexc | flexc_b8 | 47.205 (1.00x) | 47.348 (1.00x) | 47.632 (0.99x) | 0.250 |
| Flexc | flexc_a0 | 47.204 (1.00x) | 47.377 (1.00x) | 47.272 (1.00x) | 0.241 |
| Flexc | flexc_85 | 49.426 (1.00x) | 47.299 (1.04x) | 47.303 (1.04x) | 0.250 |
| Flexc | flexc_5c | 47.168 (1.00x) | 47.540 (0.99x) | 47.446 (0.99x) | 0.243 |
| Flexc | flexc_fd | 68.759 (1.00x) | 66.188 (1.04x) | 66.047 (1.04x) | 0.207 |
| Flexc | flexc_fe | 84.008 (1.00x) | 66.861 (1.26x) | 66.992 (1.25x) | 0.270 |
| **Flexc** | **Arithmetic mean speedup** | **1.00x** | **0.98x** | **1.01x** | **—** |
| **Flexc** | **Geometric mean speedup** | **1.00x** | **0.97x** | **1.00x** | **—** |
| Rover | box5r | 6.007 (1.00x) | 5.842 (1.03x) | 5.209 (1.15x) | 0.127 |
| Rover | box3r | 10.060 (1.00x) | 9.868 (1.02x) | 9.853 (1.02x) | 0.191 |
| Rover | fir8t5r | 19.512 (1.00x) | 23.011 (0.85x) | 22.987 (0.85x) | 0.332 |
| Rover | fir8t6r | 29.845 (1.00x) | 35.622 (0.84x) | 35.854 (0.83x) | 0.404 |
| Rover | fir8t7r | 38.640 (1.00x) | 45.367 (0.85x) | 45.644 (0.85x) | 0.464 |
| Rover | box4r | 77.264 (1.00x) | 87.963 (0.88x) | 108.127 (0.71x) | 0.627 |
| Rover | fir8t8r | 52.767 (1.00x) | 52.615 (1.00x) | 52.441 (1.01x) | 0.481 |
| Rover | mcm378r | 241.607 (1.00x) | 296.073 (0.82x) | 296.028 (0.82x) | 0.535 |
| Rover | mcm379r | 349.454 (1.00x) | 311.277 (1.12x) | 153.532 (2.28x) | 0.394 |
| **Rover** | **Arithmetic mean speedup** | **1.00x** | **0.93x** | **1.06x** | **—** |
| **Rover** | **Geometric mean speedup** | **1.00x** | **0.93x** | **0.99x** | **—** |
| Diospyros | vaddr7 | 0.937 (1.00x) | 0.811 (1.15x) | 0.561 (1.67x) | 0.057 |
| Diospyros | vmjmzr14 | 1.668 (1.00x) | 1.548 (1.08x) | 1.119 (1.49x) | 0.069 |
| Diospyros | vvamr23 | 1.978 (1.00x) | 1.897 (1.04x) | 1.315 (1.50x) | 0.058 |
| Diospyros | vpmr23 | 1.978 (1.00x) | 1.888 (1.05x) | 1.314 (1.51x) | 0.060 |
| Diospyros | vmacr21 | 2.271 (1.00x) | 2.264 (1.00x) | 2.034 (1.12x) | 0.069 |
| Diospyros | vmm22r21 | 3.185 (1.00x) | 3.164 (1.01x) | 2.861 (1.11x) | 0.077 |
| Diospyros | vmm22r26 | 4.533 (1.00x) | 4.584 (0.99x) | 4.056 (1.12x) | 0.078 |
| Diospyros | specr26 | 4.563 (1.00x) | 4.577 (1.00x) | 4.061 (1.12x) | 0.070 |
| Diospyros | vcv22r36 | 3.855 (1.00x) | 3.843 (1.00x) | 3.432 (1.12x) | 0.076 |
| Diospyros | vmm23r58 | 16.920 (1.00x) | 17.718 (0.95x) | 17.717 (0.96x) | 0.154 |
| Diospyros | vcv33r168 | 159.813 (1.00x) | 165.177 (0.97x) | 165.199 (0.97x) | 0.782 |
| Diospyros | drcr2r18 | 373.762 (1.00x) | 510.325 (0.73x) | 510.447 (0.73x) | 0.771 |
| Diospyros | drcr18 | 946.601 (1.00x) | 971.153 (0.97x) | 971.162 (0.97x) | 1.135 |
| **Diospyros** | **Arithmetic mean speedup** | **1.00x** | **1.00x** | **1.18x** | **—** |
| **Diospyros** | **Geometric mean speedup** | **1.00x** | **0.99x** | **1.16x** | **—** |
| Maxsat | maxcut39 | 1.456 (1.00x) | 1.421 (1.03x) | 1.437 (1.01x) | 0.113 |
| Maxsat | maxcut1 | 1.479 (1.00x) | 1.436 (1.03x) | 1.445 (1.02x) | 0.119 |
| Maxsat | s2v1 | 1.627 (1.00x) | 1.633 (1.00x) | 1.604 (1.01x) | 0.147 |
| Maxsat | s2v3 | 1.654 (1.00x) | 1.633 (1.01x) | 1.640 (1.01x) | 0.143 |
| Maxsat | hamming6 | 1.441 (1.00x) | 1.499 (0.96x) | 1.498 (0.96x) | 0.135 |
| Maxsat | johnson8 | 1.460 (1.00x) | 1.512 (0.97x) | 1.511 (0.97x) | 0.137 |
| **Maxsat** | **Arithmetic mean speedup** | **1.00x** | **1.00x** | **1.00x** | **—** |
| **Maxsat** | **Geometric mean speedup** | **1.00x** | **1.00x** | **1.00x** | **—** |
| **Across 87 designs** | **Arithmetic mean speedup** | **1.00x** | **0.99x** | **1.07x** | **—** |
| **Across 87 designs** | **Geometric mean speedup** | **1.00x** | **0.98x** | **1.00x** | **—** |
