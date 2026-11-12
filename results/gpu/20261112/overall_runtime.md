# CPU versus GPU overall runtime and solution quality — all 87 designs

**Runtime unit: milliseconds (ms). Lower runtime is faster; lower final cost is better. These are complete pipeline times from the existing quality experiments, not construction-only timings.**

> **Timing caveat: these observations do not establish a fair CPU/GPU speedup.** The CPU quality sweep ran up to 8 designs concurrently; GPU quality trials ran up to 4. Resource contention, different candidate graphs, and nondeterministic construction/search can affect wall time. The recorded times may therefore differ from standalone performance under matched conditions. No new benchmark runs were performed for this table.

## What is measured

- **CPU 8 threads:** the accepted CPU quality run, with 8 construction threads.
- **Profiled GPU (default settings):** the selected per-design quality observations used in the [profiled comparison](profiled.md). These were selected from parameter trials for quality; they are not fresh runs or an independent performance evaluation.
- **Fixed GPU:** the corrected CUDA implementation with common Block/64/512 parameters and the original controller settings.
- Every column uses **8 CP-SAT workers** and a **360-second solver limit**. This is a solver budget, not an overall-runtime cap; pipeline time can exceed 360000 ms.
- The value is the original report's `cpu_pipeline_ms` or `gpu_pipeline_ms`: tree construction, DAG construction, in-program validation/cleanup, model formulation, solving, and final solution handling. GPU pipeline time also includes setup and transfers within that interval. **Input loading/JSON parsing, output-file writing, and the separate Python verifier are excluded.**
- Each cell is **one retained quality-run observation**, not a three-run construction median. No construction time from another run was added to solver time.
- **54 designs reuse the same Fixed GPU observation in the Profiled GPU column.** Equal times for those designs are reused measurements, not independent repeated measurements. The remaining 33 Profiled GPU cells use selected tuning trials.
- Four large Fixed GPU solves (BoolE `mul80`, `mul96`, `mul112`, `mul128`) were retried with a 256 GiB allocation after the initial 128 GiB concurrent job ran out of memory; completed observations from the other 83 designs were retained. See the [full experiment report](REPORT.txt).

The three solution-quality columns show **final cost (cost ratio, status)**, for example `1093 (1.03, O)`. For **CPU, Profiled GPU, and Fixed GPU alike**, the **cost ratio is the reported final cost divided by the paper's XEG final cost for the same design** (Table 8, PDF page 15; [paper baseline CSV](../../cpu/20261112/paper_xeg.csv)). A cost ratio below 1 means a lower cost than paper XEG, while a cost ratio above 1 means a higher cost. Cost ratios are rounded to two decimal places, so a displayed cost ratio of 1.00 can hide a small cost difference. Each measured cost and runtime still come from the same retained run.

**Status key:** **O = Optimal**, a legal CP-SAT solution proved optimal within that run's candidate graph; **F = Feasible**, a legal CP-SAT solution without an optimality proof; **T = Tree fallback**, the validated seed tree retained when CP-SAT finds no feasible solution or returns a higher-cost solution. T describes the final solution source, not a CP-SAT status. O does not imply a global optimum over the complete original e-graph. The retained runs in this table contain only O and F; none used T.

Final costs and solver outcomes can differ across implementations. The [detailed quality table](profiled.md#solution-quality) and [CSV](overall_runtime.csv) provide solver outcomes; the CSV also includes the paper final-cost baseline, solution source, unrounded runtime, input hash, and report path for each observation.

Runtime values above 1000 ms are rounded to integers; values at or below 1000 ms show three decimal places. Numbers have no thousands separators. No speedup ratio is reported because these are unmatched quality-run observations.

## Recorded overall runtime and solution quality

| Benchmark / Design | CPU 8 threads runtime (ms) | Profiled GPU (default settings) runtime (ms) | Fixed GPU runtime (ms) | CPU 8 threads cost (cost ratio, status) | Profiled GPU (default settings) cost (cost ratio, status) | Fixed GPU cost (cost ratio, status) |
|---|---:|---:|---:|---:|---:|---:|
| BoolE / mul16 | 162.217 | 1330 | 1330 | 1093 (1.03, O) | 1057 (1.00, O) | 1057 (1.00, O) |
| BoolE / mul32 | 3486 | 14132 | 15097 | 4236 (1.02, O) | 4165 (1.00, O) | 4199 (1.01, O) |
| BoolE / mul48 | 13952 | 58898 | 58898 | 9576 (1.01, O) | 9404 (1.00, O) | 9404 (1.00, O) |
| BoolE / mul64 | 68312 | 101514 | 101514 | 17389 (1.04, O) | 16528 (0.99, O) | 16528 (0.99, O) |
| BoolE / mul80 | 13275 | 362660 | 362660 | 25866 (1.00, O) | 26052 (1.01, F) | 26052 (1.01, F) |
| BoolE / mul96 | 345960 | 363299 | 363299 | 37171 (1.00, O) | 37661 (1.01, F) | 37661 (1.01, F) |
| BoolE / mul112 | 241649 | 364030 | 363970 | 51179 (0.95, O) | 53578 (1.00, F) | 54339 (1.01, F) |
| BoolE / mul128 | 69147 | 363504 | 366453 | 65934 (0.81, O) | 79129 (0.97, F) | 86310 (1.05, F) |
| BoolE / mul16map | 102.247 | 240.053 | 240.053 | 1686 (1.02, O) | 1649 (1.00, O) | 1649 (1.00, O) |
| BoolE / mul32map | 565.950 | 1085 | 1085 | 7276 (1.00, O) | 7270 (1.00, O) | 7270 (1.00, O) |
| BoolE / mul48map | 1332 | 3405 | 3405 | 17288 (1.02, O) | 16968 (1.00, O) | 16968 (1.00, O) |
| BoolE / mul64map | 3003 | 7042 | 7042 | 30928 (1.01, O) | 30398 (1.00, O) | 30398 (1.00, O) |
| BoolE / mul80map | 5403 | 9843 | 9843 | 48059 (1.00, O) | 48166 (1.00, O) | 48166 (1.00, O) |
| BoolE / mul96map | 8266 | 12675 | 12944 | 69192 (1.00, O) | 69373 (1.00, O) | 69502 (1.00, O) |
| BoolE / mul112map | 15870 | 16163 | 16163 | 94285 (1.01, O) | 94571 (1.01, O) | 94571 (1.01, O) |
| BoolE / mul128map | 17996 | 24073 | 24073 | 126184 (0.90, O) | 123876 (0.88, O) | 123876 (0.88, O) |
| ESyn / mul3 | 360077 | 3200 | 1665 | 153 (0.96, F) | 154 (0.96, O) | 160 (1.00, O) |
| ESyn / c7552 | 376.509 | 386.253 | 389.480 | 6213 (1.00, O) | 6198 (1.00, O) | 6236 (1.00, O) |
| ESyn / b12 | 3334 | 2207 | 1298 | 2918 (0.99, O) | 2920 (0.99, O) | 2955 (1.00, O) |
| ESyn / i7 | 10705 | 9057 | 9057 | 1925 (1.01, O) | 1909 (1.00, O) | 1909 (1.00, O) |
| ESyn / C432 | 360090 | 10734 | 4484 | 556 (1.01, F) | 553 (1.00, O) | 553 (1.00, O) |
| ESyn / c2670 | 8715 | 5190 | 5190 | 2210 (1.00, O) | 2201 (1.00, O) | 2201 (1.00, O) |
| ESyn / C5315 | 5534 | 4046 | 4382 | 4755 (0.98, O) | 4818 (1.00, O) | 4841 (1.00, O) |
| ESyn / mul5 | 360140 | 360224 | 360237 | 600 (1.00, F) | 601 (1.00, F) | 607 (1.01, F) |
| ESyn / qdiv | 1022 | 493.138 | 493.138 | 5421 (1.01, O) | 5381 (1.00, O) | 5381 (1.00, O) |
| ESyn / max | 9811 | 17285 | 12524 | 9015 (0.98, O) | 9145 (1.00, O) | 9172 (1.00, O) |
| ESyn / bar | 9978 | 8413 | 3242 | 9157 (1.00, O) | 9113 (1.00, O) | 9113 (1.00, O) |
| ESyn / cavlc | 360197 | 360362 | 360362 | 1894 (1.00, F) | 1889 (1.00, F) | 1889 (1.00, F) |
| ESyn / adder | 17071 | 125011 | 126667 | 3445 (1.00, O) | 3445 (1.00, O) | 3447 (1.00, O) |
| ESyn / frg2 | 233967 | 140749 | 140749 | 2770 (0.99, O) | 2804 (1.00, O) | 2804 (1.00, O) |
| E-morphic / sin | 1652 | 1221 | 1221 | 14102 (0.99, O) | 14230 (1.00, O) | 14230 (1.00, O) |
| E-morphic / arbiter | 2224 | 2310 | 2167 | 35583 (1.00, O) | 35587 (1.00, O) | 35609 (1.00, O) |
| E-morphic / square | 7634 | 7489 | 6482 | 53391 (1.00, O) | 53655 (1.00, O) | 53664 (1.00, O) |
| E-morphic / multiplier | 11326 | 10981 | 10981 | 77821 (1.00, O) | 77821 (1.00, O) | 77821 (1.00, O) |
| E-morphic / log2 | 210225 | 83155 | 83155 | 84122 (1.00, O) | 84235 (1.00, O) | 84235 (1.00, O) |
| E-morphic / mem_ctrl | 17167 | 14398 | 18798 | 132982 (1.00, O) | 132984 (1.00, O) | 132984 (1.00, O) |
| E-morphic / div | 9162 | 9732 | 9732 | 159328 (1.00, O) | 159328 (1.00, O) | 159328 (1.00, O) |
| IMpress / mul512 | 18.333 | 110.234 | 106.837 | 149213 (1.00, O) | 149213 (1.00, O) | 149213 (1.00, O) |
| IMpress / mul1024 | 36.085 | 143.871 | 143.871 | 495629 (1.00, O) | 495629 (1.00, O) | 495629 (1.00, O) |
| IMpress / mul2048 | 128.845 | 254.299 | 164.222 | 3671672 (2.81, O) | 3242994 (2.48, O) | 3538018 (2.71, O) |
| Tensat / vgg | 8.371 | 114.129 | 285.748 | 468 (1.00, O) | 468 (1.00, O) | 468 (1.00, O) |
| Tensat / resnet50 | 9.828 | 113.797 | 102.684 | 398 (1.00, O) | 398 (1.00, O) | 398 (1.00, O) |
| Tensat / bert | 522.932 | 558.842 | 336.865 | 57 (1.00, O) | 58 (1.02, O) | 63 (1.11, O) |
| Tensat / nasneta | 110.068 | 528.354 | 528.354 | 1208 (1.00, O) | 1208 (1.00, O) | 1208 (1.00, O) |
| Tensat / nasrnn | 152.396 | 246.345 | 246.345 | 35 (1.00, O) | 35 (1.00, O) | 35 (1.00, O) |
| Flexc / flexc_75 | 13.705 | 149.319 | 149.319 | 35 (1.00, O) | 35 (1.00, O) | 35 (1.00, O) |
| Flexc / flexc_67 | 13.659 | 109.403 | 109.403 | 36 (1.00, O) | 36 (1.00, O) | 36 (1.00, O) |
| Flexc / flexc_db | 13.696 | 158.056 | 158.056 | 36 (1.00, O) | 36 (1.00, O) | 36 (1.00, O) |
| Flexc / flexc_86 | 10.656 | 162.135 | 162.135 | 36 (1.00, O) | 36 (1.00, O) | 36 (1.00, O) |
| Flexc / flexc_4a | 13.948 | 150.042 | 150.042 | 36 (1.00, O) | 36 (1.00, O) | 36 (1.00, O) |
| Flexc / flexc_03 | 13.986 | 152.013 | 152.013 | 36 (1.00, O) | 36 (1.00, O) | 36 (1.00, O) |
| Flexc / flexc_7b | 25.446 | 158.810 | 158.810 | 95 (1.00, O) | 95 (1.00, O) | 95 (1.00, O) |
| Flexc / flexc_ff | 24.691 | 146.431 | 146.431 | 134 (1.00, O) | 134 (1.00, O) | 134 (1.00, O) |
| Flexc / flexc_b8 | 35.488 | 145.700 | 145.700 | 136 (1.00, O) | 136 (1.00, O) | 136 (1.00, O) |
| Flexc / flexc_a0 | 29.236 | 147.433 | 147.433 | 136 (1.00, O) | 136 (1.00, O) | 136 (1.00, O) |
| Flexc / flexc_85 | 29.783 | 145.588 | 145.588 | 136 (1.00, O) | 136 (1.00, O) | 136 (1.00, O) |
| Flexc / flexc_5c | 36.140 | 140.890 | 140.890 | 136 (1.00, O) | 136 (1.00, O) | 136 (1.00, O) |
| Flexc / flexc_fd | 67.662 | 166.493 | 166.493 | 105 (1.00, O) | 105 (1.00, O) | 105 (1.00, O) |
| Flexc / flexc_fe | 29.785 | 165.257 | 165.257 | 88 (1.00, O) | 88 (1.00, O) | 88 (1.00, O) |
| Rover / box5r | 137.726 | 259.990 | 267.035 | 1819 (1.00, O) | 1819 (1.00, O) | 1819 (1.00, O) |
| Rover / box3r | 1001 | 717.360 | 717.360 | 1701 (1.00, O) | 1701 (1.00, O) | 1701 (1.00, O) |
| Rover / fir8t5r | 17008 | 2214 | 2214 | 5936 (1.00, O) | 5936 (1.00, O) | 5936 (1.00, O) |
| Rover / fir8t6r | 164615 | 4186 | 4186 | 5936 (1.00, O) | 5936 (1.00, O) | 5936 (1.00, O) |
| Rover / fir8t7r | 192382 | 251.098 | 251.098 | 5936 (1.00, O) | 5936 (1.00, O) | 5936 (1.00, O) |
| Rover / box4r | 360128 | 360304 | 710.533 | 1635 (1.00, F) | 1635 (1.00, F) | 1762 (1.08, O) |
| Rover / fir8t8r | 213702 | 281.271 | 281.271 | 5936 (1.00, O) | 5936 (1.00, O) | 5936 (1.00, O) |
| Rover / mcm378r | 81.461 | 531.378 | 531.378 | 1165 (1.00, O) | 1165 (1.00, O) | 1165 (1.00, O) |
| Rover / mcm379r | 79.878 | 242.376 | 467.493 | 1165 (1.00, O) | 1165 (1.00, O) | 1165 (1.00, O) |
| Diospyros / vaddr7 | 6.390 | 93.086 | 294.876 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / vmjmzr14 | 6.124 | 94.763 | 293.801 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / vvamr23 | 2.857 | 92.650 | 292.347 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / vpmr23 | 3.018 | 93.839 | 298.909 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / vmacr21 | 6.625 | 92.198 | 271.228 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / vmm22r21 | 3.497 | 95.836 | 217.762 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / vmm22r26 | 6.760 | 98.575 | 203.168 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / specr26 | 3.431 | 98.455 | 86.752 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / vcv22r36 | 4.021 | 94.643 | 105.142 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / vmm23r58 | 7.412 | 101.763 | 101.763 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / vcv33r168 | 29.021 | 263.262 | 263.262 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / drcr2r18 | 112.938 | 622.353 | 622.353 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Diospyros / drcr18 | 198.074 | 1100 | 1100 | 1 (1.00, O) | 1 (1.00, O) | 1 (1.00, O) |
| Maxsat / maxcut39 | 26.009 | 286.310 | 286.310 | 141 (1.00, O) | 141 (1.00, O) | 141 (1.00, O) |
| Maxsat / maxcut1 | 25.996 | 337.789 | 337.789 | 141 (1.00, O) | 141 (1.00, O) | 141 (1.00, O) |
| Maxsat / s2v1 | 45.004 | 291.839 | 291.839 | 141 (1.00, O) | 141 (1.00, O) | 141 (1.00, O) |
| Maxsat / s2v3 | 45.380 | 140.745 | 140.745 | 141 (1.00, O) | 141 (1.00, O) | 141 (1.00, O) |
| Maxsat / hamming6 | 59.050 | 226.627 | 226.627 | 65 (1.00, O) | 65 (1.00, O) | 65 (1.00, O) |
| Maxsat / johnson8 | 60.290 | 139.605 | 139.605 | 71 (1.00, O) | 71 (1.00, O) | 71 (1.00, O) |

## Sources

The CPU column matches the accepted [CPU quality-run pipeline times](../../cpu/20261112/comparison.md#quality-run). Fixed and Profiled GPU times come from the same quality reports that produced their published final costs. `profile_construction_pipeline_ms` in `profiles.csv` is solver-disabled and is **not** the source for this table.

Raw report paths in the CSV are relative to the repository on the experiment server. They remain under `build/plan/20261112_cpu_parallel/quality-delivered-cpu/`, `build/plan/20261112_gpu_alignment/v9-quality/`, and the selected `v9-tuning/` trial directories. Raw experiment logs remain local; the table and CSV are published for review.

[Back to GPU results](README.md)
