#!/usr/bin/env bash
# Run XEG (main/simple) on one e-graph.
#
# Usage:
#   scripts/run.sh <input.json> [DAG_KERNEL [NUM_THREAD_DAG_CONSTR [BITMAP_BITS]]]
#
# Arguments after the input are passed to main/simple unchanged; see its
# usage message for the valid values. With none, simple's defaults apply.
#
# Each run gets its own directory, <build>/runs/<timestamp>_<input name>/,
# holding run.log (the full console output) and any files the solver
# writes to the working directory.
#
# Environment (optional):
#   XEG_BUILD_DIR   build directory (default: <repo>/build)
set -euo pipefail

if [ "$#" -lt 1 ]; then
  sed -n '2,15p' "$0"
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${XEG_BUILD_DIR:-${repo_root}/build}"
binary="${build_dir}/main/simple"
input="$(realpath "$1")"
shift

if [ ! -x "${binary}" ]; then
  echo "error: ${binary} not found; run scripts/build.sh first" >&2
  exit 1
fi
if [ ! -f "${input}" ]; then
  echo "error: input not found: ${input}" >&2
  exit 1
fi

name="$(basename "${input}" .json)"
run_dir="${build_dir}/runs/$(date +%Y%m%d_%H%M%S)_${name}"
mkdir -p "${run_dir}"

cd "${run_dir}"
{
  echo "command: ${binary} ${input} $*"
  echo "started: $(date -Is)"
  "${binary}" "${input}" "$@"
} 2>&1 | tee run.log

echo "run directory: ${run_dir}"
