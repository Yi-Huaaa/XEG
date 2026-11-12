#!/usr/bin/env bash
# Run XEG (main/simple) on one e-graph.
#
# Usage:
#   scripts/run.sh <input.json> [DAG_KERNEL [NUM_THREAD_DAG_CONSTR [BITMAP_BITS]]]
#
# Arguments after the input are passed to main/simple unchanged; see its
# usage message for the valid values. By default the measured design profile
# in configs/gpu/87-designs.json is loaded. --no-profile restores the original
# controller defaults; positional arguments override profile geometry.
#
# Each run gets its own directory, <build>/runs/<timestamp>_<input name>/,
# holding run.log (the full console output) and any files the solver
# writes to the working directory.
#
# Environment (optional):
#   XEG_BUILD_DIR   build directory (default: <repo>/build)
#   XEG_RUN_DIR     run directory to use instead of the default one
set -euo pipefail

if [ "$#" -lt 1 ]; then
  # Print the header comment as usage.
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${XEG_BUILD_DIR:-${repo_root}/build}"
binary="$(realpath -m "${build_dir}/main/simple")"
input="$(realpath "$1")"
shift
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

if [ ! -x "${binary}" ]; then
  echo "error: ${binary} not found; run scripts/build.sh first" >&2
  exit 1
fi
if [ ! -f "${input}" ]; then
  echo "error: input not found: ${input}" >&2
  exit 1
fi

export OMP_PROC_BIND=false
unset OMP_PLACES

name="$(basename "${input}" .json)"
run_dir="${XEG_RUN_DIR:-${build_dir}/runs/$(date +%Y%m%d_%H%M%S)_${name}}"
mkdir -p "${run_dir}"

cd "${run_dir}"
{
  printf 'command: '; printf '%q ' "${binary}" "${input}" "${arguments[@]}"; printf '\n'
  echo "started: $(date -Is)"
  "${binary}" "${input}" "${arguments[@]}"
} 2>&1 | tee run.log

echo "run directory: ${run_dir}"
