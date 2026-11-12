# CPU versus GPU overall runtime and solution quality — all 87 designs

**Runtime unit: milliseconds (ms). Lower runtime is faster; lower final cost is better. These are complete pipeline times from the existing quality experiments, not construction-only timings.**

> **Timing caveat: these observations do not establish a fair CPU/GPU speedup.** The CPU quality sweep ran up to 8 designs concurrently; GPU quality trials ran up to 4. Resource contention, different candidate graphs, and nondeterministic construction/search can affect wall time. The recorded times may therefore differ from standalone performance under matched conditions. No new benchmark runs were performed for this table.

## What is measured

- **CPU 8 threads:** the accepted CPU quality run, with 8 construction threads.
- **Fixed GPU:** the corrected CUDA implementation with common Block/64/512 parameters and the original controller settings.
- **Profiled GPU:** the selected per-design quality observations used in the [profiled comparison](profiled.md). These were selected from parameter trials for quality; they are not fresh runs or an independent performance evaluation.
- Every column uses **8 CP-SAT workers** and a **360-second solver limit**. This is a solver budget, not an overall-runtime cap; pipeline time can exceed 360000 ms.
- The value is the original report's `cpu_pipeline_ms` or `gpu_pipeline_ms`: tree construction, DAG construction, in-program validation/cleanup, model formulation, solving, and final solution handling. GPU pipeline time also includes setup and transfers within that interval. **Input loading/JSON parsing, output-file writing, and the separate Python verifier are excluded.**
- Each cell is **one retained quality-run observation**, not a three-run construction median. No construction time from another run was added to solver time.
- **54 designs reuse the same Fixed GPU observation in the Profiled GPU column.** Equal times for those designs are reused measurements, not independent repeated measurements. The remaining 33 Profiled GPU cells use selected tuning trials.
- Four large Fixed GPU solves (BoolE `mul80`, `mul96`, `mul112`, `mul128`) were retried with a 256 GiB allocation after the initial 128 GiB concurrent job ran out of memory; completed observations from the other 83 designs were retained. See the [full experiment report](REPORT.txt).

The three solution-quality columns show **final cost (x/b)**, where **b is the CPU 8-thread final cost for the same design** and x is the cost in that column. CPU is always `(1.00)`; a ratio below 1 means a lower cost than CPU, while a ratio above 1 means a higher cost. Ratios are rounded to two decimal places, so `(1.00)` can hide a small cost difference. Costs and runtimes come from the same retained run.

Final costs and solver outcomes can differ across implementations. The [detailed quality table](profiled.md#solution-quality) and [CSV](overall_runtime.csv) provide solver outcomes; the CSV also includes solution source, unrounded runtime, input hash, and report path for each observation.

Runtime values above 1000 ms are rounded to integers; values at or below 1000 ms show three decimal places. Numbers have no thousands separators. No speedup ratio is reported because these are unmatched quality-run observations.

## Recorded overall runtime and solution quality

| Benchmark / Design | CPU 8 threads runtime (ms) | Fixed GPU runtime (ms) | Profiled GPU runtime (ms) | CPU 8 threads cost (x/b) | Fixed GPU cost (x/b) | Profiled GPU cost (x/b) |
|---|---:|---:|---:|---:|---:|---:|
| BoolE / mul16 | 162.217 | 1330 | 1330 | 1093 (1.00) | 1057 (0.97) | 1057 (0.97) |
| BoolE / mul32 | 3486 | 15097 | 14132 | 4236 (1.00) | 4199 (0.99) | 4165 (0.98) |
| BoolE / mul48 | 13952 | 58898 | 58898 | 9576 (1.00) | 9404 (0.98) | 9404 (0.98) |
| BoolE / mul64 | 68312 | 101514 | 101514 | 17389 (1.00) | 16528 (0.95) | 16528 (0.95) |
| BoolE / mul80 | 13275 | 362660 | 362660 | 25866 (1.00) | 26052 (1.01) | 26052 (1.01) |
| BoolE / mul96 | 345960 | 363299 | 363299 | 37171 (1.00) | 37661 (1.01) | 37661 (1.01) |
| BoolE / mul112 | 241649 | 363970 | 364030 | 51179 (1.00) | 54339 (1.06) | 53578 (1.05) |
| BoolE / mul128 | 69147 | 366453 | 363504 | 65934 (1.00) | 86310 (1.31) | 79129 (1.20) |
| BoolE / mul16map | 102.247 | 240.053 | 240.053 | 1686 (1.00) | 1649 (0.98) | 1649 (0.98) |
| BoolE / mul32map | 565.950 | 1085 | 1085 | 7276 (1.00) | 7270 (1.00) | 7270 (1.00) |
| BoolE / mul48map | 1332 | 3405 | 3405 | 17288 (1.00) | 16968 (0.98) | 16968 (0.98) |
| BoolE / mul64map | 3003 | 7042 | 7042 | 30928 (1.00) | 30398 (0.98) | 30398 (0.98) |
| BoolE / mul80map | 5403 | 9843 | 9843 | 48059 (1.00) | 48166 (1.00) | 48166 (1.00) |
| BoolE / mul96map | 8266 | 12944 | 12675 | 69192 (1.00) | 69502 (1.00) | 69373 (1.00) |
| BoolE / mul112map | 15870 | 16163 | 16163 | 94285 (1.00) | 94571 (1.00) | 94571 (1.00) |
| BoolE / mul128map | 17996 | 24073 | 24073 | 126184 (1.00) | 123876 (0.98) | 123876 (0.98) |
| ESyn / mul3 | 360077 | 1665 | 3200 | 153 (1.00) | 160 (1.05) | 154 (1.01) |
| ESyn / c7552 | 376.509 | 389.480 | 386.253 | 6213 (1.00) | 6236 (1.00) | 6198 (1.00) |
| ESyn / b12 | 3334 | 1298 | 2207 | 2918 (1.00) | 2955 (1.01) | 2920 (1.00) |
| ESyn / i7 | 10705 | 9057 | 9057 | 1925 (1.00) | 1909 (0.99) | 1909 (0.99) |
| ESyn / C432 | 360090 | 4484 | 10734 | 556 (1.00) | 553 (0.99) | 553 (0.99) |
| ESyn / c2670 | 8715 | 5190 | 5190 | 2210 (1.00) | 2201 (1.00) | 2201 (1.00) |
| ESyn / C5315 | 5534 | 4382 | 4046 | 4755 (1.00) | 4841 (1.02) | 4818 (1.01) |
| ESyn / mul5 | 360140 | 360237 | 360224 | 600 (1.00) | 607 (1.01) | 601 (1.00) |
| ESyn / qdiv | 1022 | 493.138 | 493.138 | 5421 (1.00) | 5381 (0.99) | 5381 (0.99) |
| ESyn / max | 9811 | 12524 | 17285 | 9015 (1.00) | 9172 (1.02) | 9145 (1.01) |
| ESyn / bar | 9978 | 3242 | 8413 | 9157 (1.00) | 9113 (1.00) | 9113 (1.00) |
| ESyn / cavlc | 360197 | 360362 | 360362 | 1894 (1.00) | 1889 (1.00) | 1889 (1.00) |
| ESyn / adder | 17071 | 126667 | 125011 | 3445 (1.00) | 3447 (1.00) | 3445 (1.00) |
| ESyn / frg2 | 233967 | 140749 | 140749 | 2770 (1.00) | 2804 (1.01) | 2804 (1.01) |
| E-morphic / sin | 1652 | 1221 | 1221 | 14102 (1.00) | 14230 (1.01) | 14230 (1.01) |
| E-morphic / arbiter | 2224 | 2167 | 2310 | 35583 (1.00) | 35609 (1.00) | 35587 (1.00) |
| E-morphic / square | 7634 | 6482 | 7489 | 53391 (1.00) | 53664 (1.01) | 53655 (1.00) |
| E-morphic / multiplier | 11326 | 10981 | 10981 | 77821 (1.00) | 77821 (1.00) | 77821 (1.00) |
| E-morphic / log2 | 210225 | 83155 | 83155 | 84122 (1.00) | 84235 (1.00) | 84235 (1.00) |
| E-morphic / mem_ctrl | 17167 | 18798 | 14398 | 132982 (1.00) | 132984 (1.00) | 132984 (1.00) |
| E-morphic / div | 9162 | 9732 | 9732 | 159328 (1.00) | 159328 (1.00) | 159328 (1.00) |
| IMpress / mul512 | 18.333 | 106.837 | 110.234 | 149213 (1.00) | 149213 (1.00) | 149213 (1.00) |
| IMpress / mul1024 | 36.085 | 143.871 | 143.871 | 495629 (1.00) | 495629 (1.00) | 495629 (1.00) |
| IMpress / mul2048 | 128.845 | 164.222 | 254.299 | 3671672 (1.00) | 3538018 (0.96) | 3242994 (0.88) |
| Tensat / vgg | 8.371 | 285.748 | 114.129 | 468 (1.00) | 468 (1.00) | 468 (1.00) |
| Tensat / resnet50 | 9.828 | 102.684 | 113.797 | 398 (1.00) | 398 (1.00) | 398 (1.00) |
| Tensat / bert | 522.932 | 336.865 | 558.842 | 57 (1.00) | 63 (1.11) | 58 (1.02) |
| Tensat / nasneta | 110.068 | 528.354 | 528.354 | 1208 (1.00) | 1208 (1.00) | 1208 (1.00) |
| Tensat / nasrnn | 152.396 | 246.345 | 246.345 | 35 (1.00) | 35 (1.00) | 35 (1.00) |
| Flexc / flexc_75 | 13.705 | 149.319 | 149.319 | 35 (1.00) | 35 (1.00) | 35 (1.00) |
| Flexc / flexc_67 | 13.659 | 109.403 | 109.403 | 36 (1.00) | 36 (1.00) | 36 (1.00) |
| Flexc / flexc_db | 13.696 | 158.056 | 158.056 | 36 (1.00) | 36 (1.00) | 36 (1.00) |
| Flexc / flexc_86 | 10.656 | 162.135 | 162.135 | 36 (1.00) | 36 (1.00) | 36 (1.00) |
| Flexc / flexc_4a | 13.948 | 150.042 | 150.042 | 36 (1.00) | 36 (1.00) | 36 (1.00) |
| Flexc / flexc_03 | 13.986 | 152.013 | 152.013 | 36 (1.00) | 36 (1.00) | 36 (1.00) |
| Flexc / flexc_7b | 25.446 | 158.810 | 158.810 | 95 (1.00) | 95 (1.00) | 95 (1.00) |
| Flexc / flexc_ff | 24.691 | 146.431 | 146.431 | 134 (1.00) | 134 (1.00) | 134 (1.00) |
| Flexc / flexc_b8 | 35.488 | 145.700 | 145.700 | 136 (1.00) | 136 (1.00) | 136 (1.00) |
| Flexc / flexc_a0 | 29.236 | 147.433 | 147.433 | 136 (1.00) | 136 (1.00) | 136 (1.00) |
| Flexc / flexc_85 | 29.783 | 145.588 | 145.588 | 136 (1.00) | 136 (1.00) | 136 (1.00) |
| Flexc / flexc_5c | 36.140 | 140.890 | 140.890 | 136 (1.00) | 136 (1.00) | 136 (1.00) |
| Flexc / flexc_fd | 67.662 | 166.493 | 166.493 | 105 (1.00) | 105 (1.00) | 105 (1.00) |
| Flexc / flexc_fe | 29.785 | 165.257 | 165.257 | 88 (1.00) | 88 (1.00) | 88 (1.00) |
| Rover / box5r | 137.726 | 267.035 | 259.990 | 1819 (1.00) | 1819 (1.00) | 1819 (1.00) |
| Rover / box3r | 1001 | 717.360 | 717.360 | 1701 (1.00) | 1701 (1.00) | 1701 (1.00) |
| Rover / fir8t5r | 17008 | 2214 | 2214 | 5936 (1.00) | 5936 (1.00) | 5936 (1.00) |
| Rover / fir8t6r | 164615 | 4186 | 4186 | 5936 (1.00) | 5936 (1.00) | 5936 (1.00) |
| Rover / fir8t7r | 192382 | 251.098 | 251.098 | 5936 (1.00) | 5936 (1.00) | 5936 (1.00) |
| Rover / box4r | 360128 | 710.533 | 360304 | 1635 (1.00) | 1762 (1.08) | 1635 (1.00) |
| Rover / fir8t8r | 213702 | 281.271 | 281.271 | 5936 (1.00) | 5936 (1.00) | 5936 (1.00) |
| Rover / mcm378r | 81.461 | 531.378 | 531.378 | 1165 (1.00) | 1165 (1.00) | 1165 (1.00) |
| Rover / mcm379r | 79.878 | 467.493 | 242.376 | 1165 (1.00) | 1165 (1.00) | 1165 (1.00) |
| Diospyros / vaddr7 | 6.390 | 294.876 | 93.086 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / vmjmzr14 | 6.124 | 293.801 | 94.763 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / vvamr23 | 2.857 | 292.347 | 92.650 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / vpmr23 | 3.018 | 298.909 | 93.839 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / vmacr21 | 6.625 | 271.228 | 92.198 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / vmm22r21 | 3.497 | 217.762 | 95.836 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / vmm22r26 | 6.760 | 203.168 | 98.575 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / specr26 | 3.431 | 86.752 | 98.455 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / vcv22r36 | 4.021 | 105.142 | 94.643 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / vmm23r58 | 7.412 | 101.763 | 101.763 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / vcv33r168 | 29.021 | 263.262 | 263.262 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / drcr2r18 | 112.938 | 622.353 | 622.353 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Diospyros / drcr18 | 198.074 | 1100 | 1100 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| Maxsat / maxcut39 | 26.009 | 286.310 | 286.310 | 141 (1.00) | 141 (1.00) | 141 (1.00) |
| Maxsat / maxcut1 | 25.996 | 337.789 | 337.789 | 141 (1.00) | 141 (1.00) | 141 (1.00) |
| Maxsat / s2v1 | 45.004 | 291.839 | 291.839 | 141 (1.00) | 141 (1.00) | 141 (1.00) |
| Maxsat / s2v3 | 45.380 | 140.745 | 140.745 | 141 (1.00) | 141 (1.00) | 141 (1.00) |
| Maxsat / hamming6 | 59.050 | 226.627 | 226.627 | 65 (1.00) | 65 (1.00) | 65 (1.00) |
| Maxsat / johnson8 | 60.290 | 139.605 | 139.605 | 71 (1.00) | 71 (1.00) | 71 (1.00) |

## Sources

The CPU column matches the accepted [CPU quality-run pipeline times](../../cpu/20261112/comparison.md#quality-run). Fixed and Profiled GPU times come from the same quality reports that produced their published final costs. `profile_construction_pipeline_ms` in `profiles.csv` is solver-disabled and is **not** the source for this table.

Raw report paths in the CSV are relative to the repository on the experiment server. They remain under `build/plan/20261112_cpu_parallel/quality-delivered-cpu/`, `build/plan/20261112_gpu_alignment/v9-quality/`, and the selected `v9-tuning/` trial directories. Raw experiment logs remain local; the table and CSV are published for review.

[Back to GPU results](README.md)
