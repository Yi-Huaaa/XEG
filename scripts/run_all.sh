#!/usr/bin/env bash
# Run XEG (main/simple) on every e-graph under a dataset directory.
#
# Usage:
#   scripts/run_all.sh [--dataset DIR] [DAG_KERNEL [NUM_THREAD_DAG_CONSTR [BITMAP_BITS]]]
#
# Inputs are found when the script starts: every *.json under DIR
# (default: <repo>/dataset), in sorted order. No input name is fixed, so a
# full local dataset and a clone with only the published files both work.
# Pass a subdirectory, e.g. --dataset dataset/maxsat, to run one suite.
# Arguments after the options are passed to main/simple for every input.
#
# Each input runs through scripts/run.sh, in
#   <build>/runs/<timestamp>_all/<path under DIR>/run.log
# A failing input does not stop the batch. summary.tsv in the batch
# directory has one row per input.
#
# Environment (optional):
#   XEG_BUILD_DIR   build directory (default: <repo>/build)
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${XEG_BUILD_DIR:-${repo_root}/build}"
dataset_dir="${repo_root}/dataset"

if [ "${1:-}" = "--dataset" ]; then
  if [ "$#" -lt 2 ]; then
    echo "error: --dataset needs a directory" >&2
    exit 1
  fi
  dataset_dir="$2"
  shift 2
fi
if [ ! -d "${dataset_dir}" ]; then
  echo "error: dataset directory not found: ${dataset_dir}" >&2
  exit 1
fi
dataset_dir="$(realpath "${dataset_dir}")"

mapfile -t inputs < <(find "${dataset_dir}" -type f -name '*.json' | LC_ALL=C sort)
if [ "${#inputs[@]}" -eq 0 ]; then
  echo "error: no *.json under ${dataset_dir}" >&2
  exit 1
fi

batch_dir="${build_dir}/runs/$(date +%Y%m%d_%H%M%S)_all"
mkdir -p "${batch_dir}"
summary="${batch_dir}/summary.tsv"
printf 'input\texit_code\tsolver_status\tobjective\tsolver_runtime_s\twall_s\n' > "${summary}"

echo "dataset: ${dataset_dir} (${#inputs[@]} inputs)"
echo "simple arguments: ${*:-<defaults>}"
echo "batch directory: ${batch_dir}"

count=0
for input in "${inputs[@]}"; do
  count=$((count + 1))
  rel="${input#"${dataset_dir}"/}"
  run_dir="${batch_dir}/${rel%.json}"

  start=$(date +%s.%N)
  XEG_RUN_DIR="${run_dir}" "${repo_root}/scripts/run.sh" "${input}" "$@" > /dev/null 2>&1
  exit_code=$?
  wall=$(awk -v s="${start}" -v e="$(date +%s.%N)" 'BEGIN { printf "%.2f", e - s }')

  log="${run_dir}/run.log"
  status=$(awk -F' = ' '/^Solver status = / { print $2 }' "${log}" 2>/dev/null | tail -n 1)
  objective=$(awk -F' = ' '/^Objective value = / { print $2 }' "${log}" 2>/dev/null | tail -n 1)
  runtime=$(awk '/^Runtime = / { print $3 }' "${log}" 2>/dev/null | tail -n 1)

  printf '%s\t%s\t%s\t%s\t%s\t%.2f\n' \
    "${rel}" "${exit_code}" "${status:--}" "${objective:--}" "${runtime:--}" "${wall}" \
    >> "${summary}"
  printf '[%d/%d] %s  exit=%s  status=%s  objective=%s  wall=%.1fs\n' \
    "${count}" "${#inputs[@]}" "${rel}" "${exit_code}" "${status:--}" "${objective:--}" "${wall}"
done

echo "summary: ${summary}"
