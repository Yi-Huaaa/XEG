#!/bin/bash

set -euo pipefail

# Keep benchmark traversal order deterministic.
export LC_ALL=C

if [ "$#" -ne 0 ]; then
    echo "[ERROR] run_cuopt.sh does not accept command-line arguments."
    echo "[ERROR] Run it from the gpu_extraction repository root with:"
    echo
    echo "  bash run_cuopt.sh"
    exit 1
fi

# ============================================================
# Fixed experiment configuration
# ============================================================

DAG_KERNEL=2
DAG_THREADS=64
BITMAP_BITS=512
CPU_THREADS=32
ILP_BACKEND=1

PRESOLVE_MODE=0
WARM_START_MODE=1

BUILD_DIR="build-cuopt"
SIMPLE_BIN="./${BUILD_DIR}/main/simple"
OUT_DIR="results"

# ============================================================
# Input benchmark roots
# ============================================================

BENCH_ROOTS=(
    "$HOME/smoothe-artifact/dataset/eboost_boolE"
    "$HOME/smoothe-artifact/dataset/eboost_boolE_map"
    "$HOME/smoothe-artifact/dataset/eboost_esyn"
    "$HOME/smoothe-artifact/dataset/eboost_morphic"
    "$HOME/smoothe-artifact/dataset/eboost_smoothe_diospyros"
    "$HOME/smoothe-artifact/dataset/eboost_smoothe_impress"
    "$HOME/smoothe-artifact/dataset/eboost_smoothe_rover"
    "$HOME/smoothe-artifact/dataset/eboost_smoothe_tensat"
    "$HOME/smoothe-artifact/dataset/eboost_smoothe_flexc"
    "$HOME/smoothe-artifact/dataset/eboost_smoothe_maxsat"
)

# ============================================================
# Verify execution location and build the cuOpt executable once
# ============================================================

if [ ! -d "$BUILD_DIR" ]; then
    echo "[ERROR] Missing build directory: $BUILD_DIR"
    echo "[ERROR] Run this script from the gpu_extraction repository root."
    exit 1
fi

echo "============================================================"
echo "[INFO] Building cuOpt configuration"
echo "============================================================"

cmake --build "$BUILD_DIR" -j "$(nproc)"

if [ ! -x "$SIMPLE_BIN" ]; then
    echo "[ERROR] Missing executable after build: $SIMPLE_BIN"
    exit 1
fi

# ============================================================
# Collect all input JSON files once
# ============================================================

INPUT_FILES=()
DATASET_LABELS=()
MISSING_ROOTS=()

for root in "${BENCH_ROOTS[@]}"; do
    if [ ! -d "$root" ]; then
        MISSING_ROOTS+=("$root")
        continue
    fi

    dataset_label="$(basename "$root")"

    for input_file in "$root"/*.json; do
        [ -e "$input_file" ] || continue
        INPUT_FILES+=("$input_file")
        DATASET_LABELS+=("$dataset_label")
    done
done

if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
    echo "[ERROR] No JSON benchmark files were found."
    exit 1
fi

# ============================================================
# Create one timestamped result directory with exactly two logs
# ============================================================

BASE_RUN_ID="cuopt_$(date +%Y%m%d_%H%M%S)"
RUN_ID="$BASE_RUN_ID"
RUN_ID_SUFFIX=1

while [ -e "$OUT_DIR/$RUN_ID" ]; do
    printf -v RUN_ID '%s_%02d' "$BASE_RUN_ID" "$RUN_ID_SUFFIX"
    RUN_ID_SUFFIX=$((RUN_ID_SUFFIX + 1))
done

RUN_OUT_DIR="$OUT_DIR/$RUN_ID"
PRESOLVE_LOG="$RUN_OUT_DIR/cuopt_presolve.log"
WARM_START_LOG="$RUN_OUT_DIR/cuopt_gpu_tree_warm_start.log"

mkdir -p "$RUN_OUT_DIR"

# ============================================================
# Helpers
# ============================================================

cuopt_mode_name()
{
    case "$1" in
        0)
            printf '%s' "Presolve"
            ;;
        1)
            printf '%s' "GPU tree warm start"
            ;;
        *)
            printf '%s' "Unknown"
            ;;
    esac
}

initialize_mode_log()
{
    local mode="$1"
    local log_file="$2"
    local mode_name

    mode_name="$(cuopt_mode_name "$mode")"

    {
        echo "============================================================"
        echo "XEG cuOpt all-application runtime experiment"
        echo "============================================================"
        echo "Run ID           : $RUN_ID"
        echo "cuOpt mode        : $mode_name ($mode)"
        echo "DAG kernel       : Block ($DAG_KERNEL)"
        echo "DAG threads      : $DAG_THREADS"
        echo "CPU threads      : $CPU_THREADS"
        echo "BITMAP_BITS      : $BITMAP_BITS"
        echo "ILP backend      : cuOpt ($ILP_BACKEND)"
        echo "Applications     : ${#INPUT_FILES[@]}"
        echo "Start time       : $(date --iso-8601=seconds)"
        echo "Output file      : $log_file"
        echo "============================================================"

        if [ "${#MISSING_ROOTS[@]}" -gt 0 ]; then
            echo
            echo "Missing benchmark roots:"
            for missing_root in "${MISSING_ROOTS[@]}"; do
                echo "  $missing_root"
            done
        fi
    } > "$log_file"
}

run_one_application()
{
    local mode="$1"
    local log_file="$2"
    local application_index="$3"
    local input_file="$4"
    local dataset_label="$5"
    local mode_name
    local file_name
    local start_epoch
    local end_epoch
    local elapsed_seconds
    local command_status
    local tee_status
    local pipeline_status

    mode_name="$(cuopt_mode_name "$mode")"
    file_name="$(basename "$input_file")"
    start_epoch="$(date +%s)"

    {
        echo
        echo
        echo "############################################################"
        echo "APPLICATION_BEGIN"
        echo "############################################################"
        echo "Application       : $application_index/${#INPUT_FILES[@]}"
        echo "File name         : $file_name"
        echo "Input path        : $input_file"
        echo "Dataset           : $dataset_label"
        echo "cuOpt mode        : $mode_name ($mode)"
        echo "DAG kernel        : Block ($DAG_KERNEL)"
        echo "DAG threads       : $DAG_THREADS"
        echo "CPU threads       : $CPU_THREADS"
        echo "BITMAP_BITS       : $BITMAP_BITS"
        echo "ILP backend       : cuOpt ($ILP_BACKEND)"
        echo "Start time        : $(date --iso-8601=seconds)"
        echo "Command           : $SIMPLE_BIN $input_file $DAG_KERNEL $DAG_THREADS $BITMAP_BITS $ILP_BACKEND $mode"
        echo "------------------------------------------------------------"
    } | tee -a "$log_file"

    set +e
    OMP_NUM_THREADS="$CPU_THREADS" \
        "$SIMPLE_BIN" \
        "$input_file" \
        "$DAG_KERNEL" \
        "$DAG_THREADS" \
        "$BITMAP_BITS" \
        "$ILP_BACKEND" \
        "$mode" \
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

    {
        echo "------------------------------------------------------------"
        echo "Process exit status : $command_status"
        echo "Wall time seconds   : $elapsed_seconds"
        echo "End time            : $(date --iso-8601=seconds)"

        if [ "$command_status" -ne 0 ]; then
            echo "Result              : Process failed, continuing to next application"
        else
            echo "Result              : Process completed"
        fi

        echo "############################################################"
        echo "APPLICATION_END"
        echo "############################################################"
    } | tee -a "$log_file"
}

run_all_applications_for_mode()
{
    local mode="$1"
    local log_file="$2"
    local mode_name
    local mode_start_epoch
    local mode_end_epoch
    local mode_elapsed_seconds
    local index

    mode_name="$(cuopt_mode_name "$mode")"
    mode_start_epoch="$(date +%s)"

    initialize_mode_log "$mode" "$log_file"

    echo
    echo "============================================================"
    echo "[INFO] Starting cuOpt mode: $mode_name ($mode)"
    echo "[INFO] Applications: ${#INPUT_FILES[@]}"
    echo "[INFO] Log: $log_file"
    echo "============================================================"

    for ((index = 0; index < ${#INPUT_FILES[@]}; ++index)); do
        echo
        echo "[INFO] [$((index + 1))/${#INPUT_FILES[@]}] ${INPUT_FILES[$index]}"

        run_one_application \
            "$mode" \
            "$log_file" \
            "$((index + 1))" \
            "${INPUT_FILES[$index]}" \
            "${DATASET_LABELS[$index]}"
    done

    mode_end_epoch="$(date +%s)"
    mode_elapsed_seconds=$((mode_end_epoch - mode_start_epoch))

    {
        echo
        echo
        echo "============================================================"
        echo "MODE_COMPLETE"
        echo "============================================================"
        echo "cuOpt mode         : $mode_name ($mode)"
        echo "Applications       : ${#INPUT_FILES[@]}"
        echo "Wall time seconds  : $mode_elapsed_seconds"
        echo "End time           : $(date --iso-8601=seconds)"
        echo "============================================================"
    } | tee -a "$log_file"

    echo "[INFO] Completed cuOpt mode: $mode_name ($mode)"
}

# ============================================================
# Run presolve first, then GPU tree warm start
# ============================================================

echo
echo "============================================================"
echo "XEG cuOpt experiment summary"
echo "============================================================"
echo "Applications       : ${#INPUT_FILES[@]}"
echo "DAG kernel        : Block ($DAG_KERNEL)"
echo "DAG threads       : $DAG_THREADS"
echo "CPU threads       : $CPU_THREADS"
echo "BITMAP_BITS       : $BITMAP_BITS"
echo "ILP backend       : cuOpt ($ILP_BACKEND)"
echo "Results directory : $RUN_OUT_DIR"
echo "Presolve log      : $PRESOLVE_LOG"
echo "Warm-start log    : $WARM_START_LOG"
echo "============================================================"

run_all_applications_for_mode "$PRESOLVE_MODE" "$PRESOLVE_LOG"
run_all_applications_for_mode "$WARM_START_MODE" "$WARM_START_LOG"

echo
echo "============================================================"
echo "[INFO] cuOpt experiment completed"
echo "============================================================"
echo "Results directory : $RUN_OUT_DIR"
echo "Presolve log      : $PRESOLVE_LOG"
echo "Warm-start log    : $WARM_START_LOG"
echo "============================================================"