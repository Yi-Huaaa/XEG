#!/usr/bin/env bash
# Run CPU XEG and save validated results in an isolated directory.
# Usage: scripts/run_cpu.sh INPUT [THREADS=1] [ILP_SECONDS=360] [SOLVER_WORKERS=8]
# ILP_SECONDS=0 measures construction without solving.
# Environment: XEG_BUILD_DIR and XEG_RUN_DIR work as in scripts/run.sh.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
  sed -n '2,5s/^# //p' "$0" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${XEG_BUILD_DIR:-${repo_root}/build}"
binary="$(realpath -m "${build_dir}/main/cpu")"
input="$(python3 "$repo_root/scripts/prepare_input.py" "$1")"
if [ ! -x "$binary" ] || [ ! -f "$input" ]; then
  echo "error: input or CPU executable missing; build with scripts/build.sh cpu" >&2
  exit 1
fi
threads="${2:-1}"
seconds="${3:-360}"
workers="${4:-8}"
# OpenMP binding can pin the primary thread before CP-SAT starts; its workers
# would inherit that single-core affinity. Keep the complete pipeline unbound.
# For bound construction-only measurements, use scripts/benchmark_cpu.py.
export OMP_PROC_BIND=false
unset OMP_PLACES
name="$(basename "$input" .json)"
run_dir="${XEG_RUN_DIR:-${build_dir}/runs/$(date +%Y%m%d_%H%M%S)_cpu_${name}_$$}"
mkdir -p "$run_dir"
run_dir="$(realpath "$run_dir")"
cd "$run_dir"
{
  printf 'command: %q %q %q %q %q %q\n' "$binary" "$input" "$threads" "$seconds" "$workers" "$run_dir/report.json"
  echo "started: $(date -Is)"
  "$binary" "$input" "$threads" "$seconds" "$workers" "$run_dir/report.json"
} 2>&1 | tee run.log
printf 'run directory: %s\n' "$run_dir"
