#!/usr/bin/env bash
# Run GPU XEG and export a validated CP-SAT solution.
# Usage: scripts/run_gpu.sh INPUT [KERNEL=0] [DAG_THREADS=1] [BITMAP_BITS=0] [ILP_SECONDS=360] [SOLVER_WORKERS=8]
# KERNEL: 0=Thread, 1=Warp, 2=Block. ILP_SECONDS=0 skips solving.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 6 ]; then
  sed -n '2,4s/^# //p' "$0" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${XEG_BUILD_DIR:-${repo_root}/build}"
binary="$(realpath -m "${build_dir}/main/gpu")"
input="$(realpath "$1")"
if [ ! -x "$binary" ] || [ ! -f "$input" ]; then
  echo "error: input or GPU executable missing; build with scripts/build.sh gpu" >&2
  exit 1
fi
export OMP_PROC_BIND=false
unset OMP_PLACES
name="$(basename "$input" .json)"
run_dir="${XEG_RUN_DIR:-${build_dir}/runs/$(date +%Y%m%d_%H%M%S)_gpu_${name}_$$}"
mkdir -p "$run_dir"
run_dir="$(realpath "$run_dir")"
cd "$run_dir"
command=("$binary" "$input" "${2:-0}" "${3:-1}" "${4:-0}" "${5:-360}" "${6:-8}" "$run_dir/report.json")
{
  printf 'command: '; printf '%q ' "${command[@]}"; printf '\n'
  echo "started: $(date -Is)"
  "${command[@]}"
} 2>&1 | tee run.log
printf 'run directory: %s\n' "$run_dir"
