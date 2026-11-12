#!/usr/bin/env bash
# Run GPU XEG and export a validated CP-SAT solution.
# Usage: scripts/run_gpu.sh INPUT [KERNEL] [DAG_THREADS] [BITMAP_BITS] [ILP_SECONDS] [SOLVER_WORKERS] [--config FILE.json] [--design FAMILY/STEM] [--no-profile]
# KERNEL: 0=Thread, 1=Warp, 2=Block. ILP_SECONDS=0 skips solving.
set -euo pipefail
if [ "$#" -lt 1 ]; then
  sed -n '2,4s/^# //p' "$0" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${XEG_BUILD_DIR:-${repo_root}/build}"
binary="$(realpath -m "${build_dir}/main/gpu")"
input="$(realpath "$1")"
shift
# Resolve a relative configuration before entering the isolated run directory.
arguments=()
profile_selected=false
while [ "$#" -gt 0 ]; do
  if [ "$1" = --config ]; then
    if [ "$#" -lt 2 ]; then echo 'error: --config needs a path' >&2; exit 2; fi
    profile_selected=true
    arguments+=(--config "$(realpath "$2")")
    shift 2
  else
    if [ "$1" = --no-profile ]; then profile_selected=true; fi
    arguments+=("$1")
    shift
  fi
done
if ! "$profile_selected"; then
  arguments+=(--config "$repo_root/configs/gpu/87-designs.json")
fi
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
command=("$binary" "$input" "${arguments[@]}" --report "$run_dir/report.json")
{
  printf 'command: '; printf '%q ' "${command[@]}"; printf '\n'
  echo "started: $(date -Is)"
  "${command[@]}"
} 2>&1 | tee run.log
printf 'run directory: %s\n' "$run_dir"
