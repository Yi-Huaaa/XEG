#!/bin/bash

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "[ERROR] run_xeg_d.sh does not accept command-line arguments."
    echo "[ERROR] Run it from the gpu_extraction repository root:"
    echo
    echo "  bash run_xeg_d.sh"
    exit 1
fi

BUILD_DIR="build-cpsat"
XEG_D_BIN="./${BUILD_DIR}/main/xeg_d"
OUT_DIR="results"

INPUT_FILES=(
    "$HOME/smoothe-artifact/dataset/eboost_morphic/log2.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE/mul32.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE/mul48.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE/mul64.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE_map/mul64_map.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE/mul80.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE_map/mul80_map.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE/mul96.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE_map/mul96_map.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE/mul112.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE_map/mul112_map.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_morphic/div.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE/mul128.rename.json"
    "$HOME/smoothe-artifact/dataset/eboost_boolE_map/mul128_map.rename.json"
)

if [ ! -d "$BUILD_DIR" ]; then
    echo "[ERROR] Missing build directory: $BUILD_DIR"
    echo "[ERROR] Run this script from the gpu_extraction repository root."
    exit 1
fi

if [ ! -x /usr/bin/time ]; then
    echo "[ERROR] Missing executable: /usr/bin/time"
    exit 1
fi

for input_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "$input_file" ]; then
        echo "[ERROR] Missing input file: $input_file"
        exit 1
    fi
done

echo "============================================================"
echo "[INFO] Building XEG-D executable"
echo "============================================================"

cmake --build "$BUILD_DIR" --target xeg_d -j "$(nproc)"

if [ ! -x "$XEG_D_BIN" ]; then
    echo "[ERROR] Missing executable after build: $XEG_D_BIN"
    exit 1
fi

BASE_RUN_ID="xeg_d_$(date +%Y%m%d_%H%M%S)"
RUN_ID="$BASE_RUN_ID"
RUN_ID_SUFFIX=1

while [ -e "$OUT_DIR/$RUN_ID" ]; do
    printf -v RUN_ID '%s_%02d' "$BASE_RUN_ID" "$RUN_ID_SUFFIX"
    RUN_ID_SUFFIX=$((RUN_ID_SUFFIX + 1))
done

RUN_OUT_DIR="$OUT_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_OUT_DIR/summary.log"

mkdir -p "$RUN_OUT_DIR"

batch_start_epoch="$(date +%s)"

{
    echo "============================================================"
    echo "XEG-D experiment"
    echo "============================================================"
    echo "Run ID           : $RUN_ID"
    echo "Applications     : ${#INPUT_FILES[@]}"
    echo "Executable       : $XEG_D_BIN"
    echo "CP-SAT limit     : 86400 seconds per application"
    echo "Start time       : $(date --iso-8601=seconds)"
    echo "Results directory: $RUN_OUT_DIR"
    echo "============================================================"
} | tee "$SUMMARY_LOG"

for ((index = 0; index < ${#INPUT_FILES[@]}; ++index)); do
    application_index=$((index + 1))
    input_file="${INPUT_FILES[$index]}"
    file_name="$(basename "$input_file")"
    file_stem="${file_name%.json}"

    printf -v index_label '%02d' "$application_index"

    log_file="$RUN_OUT_DIR/${index_label}_${file_stem}.log"

    start_epoch="$(date +%s)"

    {
        echo "============================================================"
        echo "XEG-D application"
        echo "============================================================"
        echo "Application       : $application_index/${#INPUT_FILES[@]}"
        echo "File name         : $file_name"
        echo "Input path        : $input_file"
        echo "Start time        : $(date --iso-8601=seconds)"
        echo "Command           : /usr/bin/time -v $XEG_D_BIN $input_file"
        echo "============================================================"
    } | tee "$log_file"

    echo
    echo "[INFO] [$application_index/${#INPUT_FILES[@]}] $file_name"
    echo "[INFO] Log: $log_file"

    set +e
    /usr/bin/time -v \
        "$XEG_D_BIN" \
        "$input_file" \
        2>&1 | tee -a "$log_file"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e

    command_status="${pipeline_status[0]}"
    tee_status="${pipeline_status[1]}"

    if [ "$tee_status" -ne 0 ]; then
        echo "[ERROR] Failed to write log file: $log_file" >&2
        exit "$tee_status"
    fi

    end_epoch="$(date +%s)"
    elapsed_seconds=$((end_epoch - start_epoch))

    if [ "$command_status" -eq 0 ]; then
        result="Process completed"
    else
        result="Process failed, continuing to next application"
    fi

    {
        echo "============================================================"
        echo "Process exit status : $command_status"
        echo "Wall time seconds   : $elapsed_seconds"
        echo "End time            : $(date --iso-8601=seconds)"
        echo "Result              : $result"
        echo "============================================================"
    } | tee -a "$log_file"

    {
        echo
        echo "Application       : $application_index/${#INPUT_FILES[@]}"
        echo "File name         : $file_name"
        echo "Log file          : $log_file"
        echo "Process status    : $command_status"
        echo "Wall time seconds : $elapsed_seconds"
        echo "Result            : $result"
    } >> "$SUMMARY_LOG"
done

batch_end_epoch="$(date +%s)"
batch_elapsed_seconds=$((batch_end_epoch - batch_start_epoch))

{
    echo
    echo "============================================================"
    echo "XEG-D experiment complete"
    echo "============================================================"
    echo "Applications       : ${#INPUT_FILES[@]}"
    echo "Wall time seconds  : $batch_elapsed_seconds"
    echo "End time           : $(date --iso-8601=seconds)"
    echo "Results directory  : $RUN_OUT_DIR"
    echo "============================================================"
} | tee -a "$SUMMARY_LOG"

echo
echo "[INFO] XEG-D experiment completed"
echo "[INFO] Results: $RUN_OUT_DIR"
echo "[INFO] Summary: $SUMMARY_LOG"
