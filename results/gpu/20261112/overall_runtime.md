# CPU versus GPU overall runtime — all 87 designs

**Unit: milliseconds (ms). Lower is faster for the recorded run. These are complete pipeline times from the existing quality experiments, not construction-only timings.**

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

Final costs and solver outcomes can differ across implementations. Read these times alongside the [quality table](profiled.md#solution-quality). The [CSV](overall_runtime.csv) includes the matching final cost, solver status, solution source, unrounded runtime, input hash, and report path for each observation.

Runtime values above 1000 ms are rounded to integers; values at or below 1000 ms show three decimal places. Numbers have no thousands separators. No speedup ratio is reported because these are unmatched quality-run observations.

## Recorded overall runtime

| Benchmark | Design | CPU 8 threads (ms) | Fixed GPU (ms) | Profiled GPU (ms) |
|---|---|---:|---:|---:|
| BoolE | mul16 | 162.217 | 1330 | 1330 |
| BoolE | mul32 | 3486 | 15097 | 14132 |
| BoolE | mul48 | 13952 | 58898 | 58898 |
| BoolE | mul64 | 68312 | 101514 | 101514 |
| BoolE | mul80 | 13275 | 362660 | 362660 |
| BoolE | mul96 | 345960 | 363299 | 363299 |
| BoolE | mul112 | 241649 | 363970 | 364030 |
| BoolE | mul128 | 69147 | 366453 | 363504 |
| BoolE | mul16map | 102.247 | 240.053 | 240.053 |
| BoolE | mul32map | 565.950 | 1085 | 1085 |
| BoolE | mul48map | 1332 | 3405 | 3405 |
| BoolE | mul64map | 3003 | 7042 | 7042 |
| BoolE | mul80map | 5403 | 9843 | 9843 |
| BoolE | mul96map | 8266 | 12944 | 12675 |
| BoolE | mul112map | 15870 | 16163 | 16163 |
| BoolE | mul128map | 17996 | 24073 | 24073 |
| ESyn | mul3 | 360077 | 1665 | 3200 |
| ESyn | c7552 | 376.509 | 389.480 | 386.253 |
| ESyn | b12 | 3334 | 1298 | 2207 |
| ESyn | i7 | 10705 | 9057 | 9057 |
| ESyn | C432 | 360090 | 4484 | 10734 |
| ESyn | c2670 | 8715 | 5190 | 5190 |
| ESyn | C5315 | 5534 | 4382 | 4046 |
| ESyn | mul5 | 360140 | 360237 | 360224 |
| ESyn | qdiv | 1022 | 493.138 | 493.138 |
| ESyn | max | 9811 | 12524 | 17285 |
| ESyn | bar | 9978 | 3242 | 8413 |
| ESyn | cavlc | 360197 | 360362 | 360362 |
| ESyn | adder | 17071 | 126667 | 125011 |
| ESyn | frg2 | 233967 | 140749 | 140749 |
| E-morphic | sin | 1652 | 1221 | 1221 |
| E-morphic | arbiter | 2224 | 2167 | 2310 |
| E-morphic | square | 7634 | 6482 | 7489 |
| E-morphic | multiplier | 11326 | 10981 | 10981 |
| E-morphic | log2 | 210225 | 83155 | 83155 |
| E-morphic | mem_ctrl | 17167 | 18798 | 14398 |
| E-morphic | div | 9162 | 9732 | 9732 |
| IMpress | mul512 | 18.333 | 106.837 | 110.234 |
| IMpress | mul1024 | 36.085 | 143.871 | 143.871 |
| IMpress | mul2048 | 128.845 | 164.222 | 254.299 |
| Tensat | vgg | 8.371 | 285.748 | 114.129 |
| Tensat | resnet50 | 9.828 | 102.684 | 113.797 |
| Tensat | bert | 522.932 | 336.865 | 558.842 |
| Tensat | nasneta | 110.068 | 528.354 | 528.354 |
| Tensat | nasrnn | 152.396 | 246.345 | 246.345 |
| Flexc | flexc_75 | 13.705 | 149.319 | 149.319 |
| Flexc | flexc_67 | 13.659 | 109.403 | 109.403 |
| Flexc | flexc_db | 13.696 | 158.056 | 158.056 |
| Flexc | flexc_86 | 10.656 | 162.135 | 162.135 |
| Flexc | flexc_4a | 13.948 | 150.042 | 150.042 |
| Flexc | flexc_03 | 13.986 | 152.013 | 152.013 |
| Flexc | flexc_7b | 25.446 | 158.810 | 158.810 |
| Flexc | flexc_ff | 24.691 | 146.431 | 146.431 |
| Flexc | flexc_b8 | 35.488 | 145.700 | 145.700 |
| Flexc | flexc_a0 | 29.236 | 147.433 | 147.433 |
| Flexc | flexc_85 | 29.783 | 145.588 | 145.588 |
| Flexc | flexc_5c | 36.140 | 140.890 | 140.890 |
| Flexc | flexc_fd | 67.662 | 166.493 | 166.493 |
| Flexc | flexc_fe | 29.785 | 165.257 | 165.257 |
| Rover | box5r | 137.726 | 267.035 | 259.990 |
| Rover | box3r | 1001 | 717.360 | 717.360 |
| Rover | fir8t5r | 17008 | 2214 | 2214 |
| Rover | fir8t6r | 164615 | 4186 | 4186 |
| Rover | fir8t7r | 192382 | 251.098 | 251.098 |
| Rover | box4r | 360128 | 710.533 | 360304 |
| Rover | fir8t8r | 213702 | 281.271 | 281.271 |
| Rover | mcm378r | 81.461 | 531.378 | 531.378 |
| Rover | mcm379r | 79.878 | 467.493 | 242.376 |
| Diospyros | vaddr7 | 6.390 | 294.876 | 93.086 |
| Diospyros | vmjmzr14 | 6.124 | 293.801 | 94.763 |
| Diospyros | vvamr23 | 2.857 | 292.347 | 92.650 |
| Diospyros | vpmr23 | 3.018 | 298.909 | 93.839 |
| Diospyros | vmacr21 | 6.625 | 271.228 | 92.198 |
| Diospyros | vmm22r21 | 3.497 | 217.762 | 95.836 |
| Diospyros | vmm22r26 | 6.760 | 203.168 | 98.575 |
| Diospyros | specr26 | 3.431 | 86.752 | 98.455 |
| Diospyros | vcv22r36 | 4.021 | 105.142 | 94.643 |
| Diospyros | vmm23r58 | 7.412 | 101.763 | 101.763 |
| Diospyros | vcv33r168 | 29.021 | 263.262 | 263.262 |
| Diospyros | drcr2r18 | 112.938 | 622.353 | 622.353 |
| Diospyros | drcr18 | 198.074 | 1100 | 1100 |
| Maxsat | maxcut39 | 26.009 | 286.310 | 286.310 |
| Maxsat | maxcut1 | 25.996 | 337.789 | 337.789 |
| Maxsat | s2v1 | 45.004 | 291.839 | 291.839 |
| Maxsat | s2v3 | 45.380 | 140.745 | 140.745 |
| Maxsat | hamming6 | 59.050 | 226.627 | 226.627 |
| Maxsat | johnson8 | 60.290 | 139.605 | 139.605 |

## Sources

The CPU column matches the accepted [CPU quality-run pipeline times](../../cpu/20261112/comparison.md#quality-run). Fixed and Profiled GPU times come from the same quality reports that produced their published final costs. `profile_construction_pipeline_ms` in `profiles.csv` is solver-disabled and is **not** the source for this table.

Raw report paths in the CSV are relative to the repository on the experiment server. They remain under `build/plan/20261112_cpu_parallel/quality-delivered-cpu/`, `build/plan/20261112_gpu_alignment/v9-quality/`, and the selected `v9-tuning/` trial directories. Raw experiment logs remain local; the table and CSV are published for review.

[Back to GPU results](README.md)
