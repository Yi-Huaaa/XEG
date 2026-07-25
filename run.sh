#!/bin/bash

set -euo pipefail

# ============================================================
# Usage
#
#   bash run.sh
#
# Default settings:
#
#   DAG_KERNEL=2
#   DAG_THREADS=64
#   CPU_THREADS=32
#   NUM_REPEATS=3
#
# Override settings:
#
#   bash run.sh <DAG_KERNEL> <DAG_THREADS> <CPU_THREADS> <NUM_REPEATS>
#
# Examples:
#
#   Thread:
#     bash run.sh 0 1 32 3
#
#   Block 64:
#     bash run.sh 2 64 32 3
#
#   Block 128:
#     bash run.sh 2 128 32 3
#
#   Block 256:
#     bash run.sh 2 256 32 3
#
# DAG_KERNEL:
#
#   0 = Thread
#   1 = Warp  (not implemented)
#   2 = Block
# ============================================================


# ============================================================
# Runtime configuration
#
# Positional arguments override the default values.
# ============================================================

DAG_KERNEL="${1:-2}"
DAG_THREADS="${2:-64}"
CPU_THREADS="${3:-32}"
NUM_REPEATS="${4:-3}"

if [ "$#" -gt 4 ]; then
    echo "[ERROR] Too many command-line arguments"
    echo
    echo "Usage:"
    echo "  bash run.sh [DAG_KERNEL] [DAG_THREADS] [CPU_THREADS] [NUM_REPEATS]"
    exit 1
fi


# ============================================================
# Validate runtime configuration
# ============================================================

if ! [[ "$DAG_KERNEL" =~ ^[0-2]$ ]]; then
    echo "[ERROR] DAG_KERNEL must be 0, 1, or 2"
    exit 1
fi

if ! [[ "$DAG_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] DAG_THREADS must be a positive integer"
    exit 1
fi

if ! [[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] CPU_THREADS must be a positive integer"
    exit 1
fi

if ! [[ "$NUM_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_REPEATS must be a positive integer"
    exit 1
fi

case "$DAG_KERNEL" in
    0)
        DAG_KERNEL_NAME="Thread"

        if [ "$DAG_THREADS" -ne 1 ]; then
            echo "[ERROR] Thread DAG kernel requires DAG_THREADS=1"
            exit 1
        fi
        ;;

    1)
        DAG_KERNEL_NAME="Warp"

        if [ "$DAG_THREADS" -lt 32 ] ||
           [ "$DAG_THREADS" -gt 1024 ]; then
            echo "[ERROR] Warp DAG kernel requires DAG_THREADS between 32 and 1024"
            exit 1
        fi

        if (( DAG_THREADS % 32 != 0 )); then
            echo "[ERROR] Warp DAG kernel requires DAG_THREADS to be a multiple of 32"
            echo "[ERROR] Valid examples: 32, 64, 96, 128, 256"
            exit 1
        fi
        ;;

    2)
        DAG_KERNEL_NAME="Block"

        if [ "$DAG_THREADS" -lt 64 ] ||
           [ "$DAG_THREADS" -gt 1024 ]; then
            echo "[ERROR] Block DAG kernel requires DAG_THREADS between 64 and 1024"
            exit 1
        fi

        if (( (DAG_THREADS & (DAG_THREADS - 1)) != 0 )); then
            echo "[ERROR] Block DAG kernel requires a power-of-two DAG_THREADS value"
            echo "[ERROR] Valid examples: 64, 128, 256, 512, 1024"
            exit 1
        fi
        ;;
esac


# ============================================================
# Compile
#
# This script is expected to run inside:
#
#   ~/gpu_extraction/build
# ============================================================

echo "============================================================"
echo "[INFO] Building project"
echo "============================================================"

cmake ..
cmake --build . -j


# ============================================================
# Input benchmark roots
# ============================================================

BENCH_ROOTS=(
    # Old extraction-gym and e-boost paths
    # "$HOME/extraction-gym/data"
    # "$HOME/e-boost/benchmark/BoolE"
    # "$HOME/e-boost/benchmark/E-morphic"
    # "$HOME/e-boost/benchmark/SmootheE"
    # "$HOME/e-boost/benchmark/E-syn"

    # New dataset roots
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
# Output directory
#
# Example:
#
# results/
# └── kernel_2_dagthreads_128_cputhreads_32/
#     ├── repeat_1/
#     │   └── all_applications.log
#     ├── repeat_2/
#     │   └── all_applications.log
#     └── repeat_3/
#         └── all_applications.log
# ============================================================

OUT_DIR="results"

CONFIG_NAME="kernel_${DAG_KERNEL}_dagthreads_${DAG_THREADS}_cputhreads_${CPU_THREADS}"

CONFIG_OUT_DIR="$OUT_DIR/$CONFIG_NAME"

mkdir -p "$CONFIG_OUT_DIR"


# ============================================================
# Benchmark subfolders used by extraction-gym
# ============================================================

folders=(
    babble
    constructed
    dummy_examples
    eggcc-bril
    herbie
    maxsat
    set_covering
    diospyros
    egg
    flexc
    impress
    rover
    tensat
)


# ============================================================
# Run mode
#
#   0 = run all benchmark roots
#   1 = run all files in SINGLE_FOLDER
#   2 = run only SINGLE_FILE
#
# For full data collection, use:
#
#   RUN_MODE=0
# ============================================================

RUN_MODE=0


# ============================================================
# Single-folder / single-file configuration
#
# These are only used when RUN_MODE is 1 or 2.
# ============================================================

SINGLE_FOLDER="boole"
SINGLE_FILE="mul32.rename.json"

# SINGLE_FILE="c7552.rename.json"
# SINGLE_FILE="i7.rename.json"
# SINGLE_FILE="b12.rename.json"
# SINGLE_FILE="c2670.rename.json"
# SINGLE_FILE="C432.rename.json"
# SINGLE_FILE="mul3.rename.json"
# SINGLE_FILE="mul5.rename.json"
# SINGLE_FILE="bert.rename.json"
# SINGLE_FILE="box_filter_3iteration_egraph.rename.json"
# SINGLE_FILE="box_filter_4iteration_egraph.rename.json"
# SINGLE_FILE="vector_2d_conv_3x3_3x3_root_168.rename.json"
# SINGLE_FILE="sin.rename.json"
# SINGLE_FILE="mul112.rename.json"
# SINGLE_FILE="mul80.rename.json"
# SINGLE_FILE="mul48.rename.json"


# ============================================================
# Validate RUN_MODE
# ============================================================

if ! [[ "$RUN_MODE" =~ ^[0-2]$ ]]; then
    echo "[ERROR] RUN_MODE must be 0, 1, or 2"
    exit 1
fi


# ============================================================
# Print configuration
# ============================================================

echo
echo "============================================================"
echo "[INFO] Experiment configuration"
echo "============================================================"
echo "DAG kernel       : $DAG_KERNEL_NAME ($DAG_KERNEL)"
echo "DAG threads      : $DAG_THREADS"
echo "CPU threads      : $CPU_THREADS"
echo "Number of repeats: $NUM_REPEATS"
echo "Run mode         : $RUN_MODE"
echo "Output directory : $CONFIG_OUT_DIR"
echo "============================================================"
echo


# ============================================================
# Write the header of one repeat log
#
# The repeat log is truncated before data collection starts.
# Therefore, rerunning the same configuration replaces the
# previous repeat logs rather than appending duplicate data.
# ============================================================

initialize_repeat_log()
{
    : > "$CURRENT_REPEAT_LOG"

    {
        echo "============================================================"
        echo "GPU DAG experiment"
        echo "============================================================"
        echo "Repeat           : $CURRENT_REPEAT/$NUM_REPEATS"
        echo "DAG kernel       : $DAG_KERNEL_NAME ($DAG_KERNEL)"
        echo "DAG threads      : $DAG_THREADS"
        echo "CPU threads      : $CPU_THREADS"
        echo "Run mode         : $RUN_MODE"
        echo "Output file      : $CURRENT_REPEAT_LOG"
        echo "Start time       : $(date --iso-8601=seconds)"
        echo "============================================================"
        echo
    } >> "$CURRENT_REPEAT_LOG"
}


# ============================================================
# Run one benchmark
#
# Arguments:
#
#   $1 = input JSON file
#   $2 = dataset label
#
# All applications in the current repeat are appended to:
#
#   $CURRENT_REPEAT_LOG
#
# simple.cpp currently reads:
#
#   argv[1] = input file
#   argv[2] = DAG_KERNEL
#   argv[3] = DAG_THREADS
#
# CPU_THREADS is supplied through OMP_NUM_THREADS.
# ============================================================

run_benchmark()
{
    local input_file="$1"
    local dataset_label="$2"
    local file_name

    file_name="$(basename "$input_file")"

    RUN_COUNT=$((RUN_COUNT + 1))

    echo
    echo "------------------------------------------------------------"
    echo "[RUN $RUN_COUNT]"
    echo "Input       : $input_file"
    echo "Dataset     : $dataset_label"
    echo "Repeat      : $CURRENT_REPEAT/$NUM_REPEATS"
    echo "DAG kernel  : $DAG_KERNEL_NAME ($DAG_KERNEL)"
    echo "DAG threads : $DAG_THREADS"
    echo "CPU threads : $CPU_THREADS"
    echo "Output file : $CURRENT_REPEAT_LOG"
    echo "------------------------------------------------------------"

    {
        echo
        echo
        echo "############################################################"
        echo "APPLICATION_BEGIN"
        echo "############################################################"
        echo "Run index   : $RUN_COUNT"
        echo "File name   : $file_name"
        echo "Input path  : $input_file"
        echo "Dataset     : $dataset_label"
        echo "Repeat      : $CURRENT_REPEAT/$NUM_REPEATS"
        echo "DAG kernel  : $DAG_KERNEL_NAME ($DAG_KERNEL)"
        echo "DAG threads : $DAG_THREADS"
        echo "CPU threads : $CPU_THREADS"
        echo "Start time  : $(date --iso-8601=seconds)"
        echo "------------------------------------------------------------"

        OMP_NUM_THREADS="$CPU_THREADS" \
            ./main/simple \
            "$input_file" \
            "$DAG_KERNEL" \
            "$DAG_THREADS"

        echo "------------------------------------------------------------"
        echo "End time    : $(date --iso-8601=seconds)"
        echo "############################################################"
        echo "APPLICATION_END"
        echo "############################################################"
    } 2>&1 | tee -a "$CURRENT_REPEAT_LOG"
}


# ============================================================
# RUN_MODE=0
#
# Run all applications under all benchmark roots.
# ============================================================

run_all_benchmarks()
{
    local found_any=0
    local root
    local folder
    local input_dir
    local input_file
    local dataset_label

    echo "[INFO] Running all benchmark applications"

    for root in "${BENCH_ROOTS[@]}"; do
        if [ ! -d "$root" ]; then
            echo "[WARNING] Benchmark root does not exist: $root"

            {
                echo "[WARNING] Benchmark root does not exist: $root"
            } >> "$CURRENT_REPEAT_LOG"

            continue
        fi

        echo
        echo "============================================================"
        echo "[INFO] Benchmark root: $root"
        echo "============================================================"

        {
            echo
            echo "============================================================"
            echo "BENCHMARK_ROOT_BEGIN"
            echo "Root: $root"
            echo "============================================================"
        } >> "$CURRENT_REPEAT_LOG"

        if [[ "$root" == *"extraction-gym/data"* ]]; then
            for folder in "${folders[@]}"; do
                input_dir="$root/$folder"

                if [ ! -d "$input_dir" ]; then
                    continue
                fi

                echo "[INFO] Folder: $input_dir"

                for input_file in "$input_dir"/*.json; do
                    [ -e "$input_file" ] || continue

                    found_any=1
                    dataset_label="$(basename "$root")/$folder"

                    run_benchmark \
                        "$input_file" \
                        "$dataset_label"
                done
            done
        else
            dataset_label="$(basename "$root")"

            for input_file in "$root"/*.json; do
                [ -e "$input_file" ] || continue

                found_any=1

                run_benchmark \
                    "$input_file" \
                    "$dataset_label"
            done
        fi

        {
            echo
            echo "============================================================"
            echo "BENCHMARK_ROOT_END"
            echo "Root: $root"
            echo "============================================================"
        } >> "$CURRENT_REPEAT_LOG"
    done

    if [ "$found_any" -eq 0 ]; then
        echo "[ERROR] No JSON benchmark files were found"

        {
            echo
            echo "[ERROR] No JSON benchmark files were found"
        } >> "$CURRENT_REPEAT_LOG"

        return 1
    fi
}


# ============================================================
# RUN_MODE=1
#
# Find SINGLE_FOLDER in BENCH_ROOTS and run every JSON file
# inside that folder.
# ============================================================

run_single_folder()
{
    local root
    local input_dir
    local input_file
    local dataset_label
    local found_file=0

    for root in "${BENCH_ROOTS[@]}"; do
        input_dir="$root/$SINGLE_FOLDER"

        if [ ! -d "$input_dir" ]; then
            continue
        fi

        echo "[INFO] Running all files in: $input_dir"

        dataset_label="$(basename "$root")/$SINGLE_FOLDER"

        for input_file in "$input_dir"/*.json; do
            [ -e "$input_file" ] || continue

            found_file=1

            run_benchmark \
                "$input_file" \
                "$dataset_label"
        done

        if [ "$found_file" -eq 0 ]; then
            echo "[ERROR] No JSON files found in: $input_dir"

            {
                echo
                echo "[ERROR] No JSON files found in: $input_dir"
            } >> "$CURRENT_REPEAT_LOG"

            return 1
        fi

        return 0
    done

    echo "[ERROR] SINGLE_FOLDER=$SINGLE_FOLDER was not found"

    {
        echo
        echo "[ERROR] SINGLE_FOLDER=$SINGLE_FOLDER was not found"
    } >> "$CURRENT_REPEAT_LOG"

    return 1
}


# ============================================================
# RUN_MODE=2
#
# Search for SINGLE_FILE in two possible locations:
#
#   root/SINGLE_FOLDER/SINGLE_FILE
#   root/SINGLE_FILE
# ============================================================

run_single_file()
{
    local root
    local input_file
    local dataset_label

    for root in "${BENCH_ROOTS[@]}"; do
        input_file="$root/$SINGLE_FOLDER/$SINGLE_FILE"

        if [ -f "$input_file" ]; then
            dataset_label="$(basename "$root")/$SINGLE_FOLDER"

            echo "[INFO] Running single file: $input_file"

            run_benchmark \
                "$input_file" \
                "$dataset_label"

            return 0
        fi

        input_file="$root/$SINGLE_FILE"

        if [ -f "$input_file" ]; then
            dataset_label="$(basename "$root")"

            echo "[INFO] Running single file: $input_file"

            run_benchmark \
                "$input_file" \
                "$dataset_label"

            return 0
        fi
    done

    echo "[ERROR] File $SINGLE_FILE was not found in any benchmark root"

    {
        echo
        echo "[ERROR] File $SINGLE_FILE was not found in any benchmark root"
    } >> "$CURRENT_REPEAT_LOG"

    return 1
}


# ============================================================
# Main repeat loop
#
# Important loop order:
#
#   Repeat 1:
#       run every selected application
#       append all output to repeat_1/all_applications.log
#
#   Repeat 2:
#       run every selected application
#       append all output to repeat_2/all_applications.log
#
#   Repeat 3:
#       run every selected application
#       append all output to repeat_3/all_applications.log
#
# The complete dataset traversal is inside the repeat loop.
# ============================================================

for CURRENT_REPEAT in $(seq 1 "$NUM_REPEATS"); do
    CURRENT_REPEAT_OUT_DIR="$CONFIG_OUT_DIR/repeat_${CURRENT_REPEAT}"

    CURRENT_REPEAT_LOG="$CURRENT_REPEAT_OUT_DIR/all_applications.log"

    mkdir -p "$CURRENT_REPEAT_OUT_DIR"

    RUN_COUNT=0

    initialize_repeat_log

    echo
    echo "############################################################"
    echo "[INFO] Starting repeat $CURRENT_REPEAT/$NUM_REPEATS"
    echo "[INFO] Output file: $CURRENT_REPEAT_LOG"
    echo "############################################################"

    case "$RUN_MODE" in
        0)
            run_all_benchmarks
            ;;

        1)
            run_single_folder
            ;;

        2)
            run_single_file
            ;;
    esac

    {
        echo
        echo
        echo "============================================================"
        echo "REPEAT_COMPLETE"
        echo "============================================================"
        echo "Repeat                : $CURRENT_REPEAT/$NUM_REPEATS"
        echo "Applications executed : $RUN_COUNT"
        echo "End time              : $(date --iso-8601=seconds)"
        echo "============================================================"
    } >> "$CURRENT_REPEAT_LOG"

    echo
    echo "############################################################"
    echo "[INFO] Completed repeat $CURRENT_REPEAT/$NUM_REPEATS"
    echo "[INFO] Applications executed: $RUN_COUNT"
    echo "[INFO] Output file: $CURRENT_REPEAT_LOG"
    echo "############################################################"
done


# ============================================================
# Final summary
# ============================================================

echo
echo "============================================================"
echo "[INFO] Experiment completed"
echo "============================================================"
echo "DAG kernel       : $DAG_KERNEL_NAME ($DAG_KERNEL)"
echo "DAG threads      : $DAG_THREADS"
echo "CPU threads      : $CPU_THREADS"
echo "Completed repeats: $NUM_REPEATS"
echo "Results directory: $CONFIG_OUT_DIR"
echo "============================================================"


# ============================================================
# Personal notes
#
# Default:
#   bash run.sh
#
# Thread:
#   bash run.sh 0 1 32 3
#
# Block 64:
#   bash run.sh 2 64 32 3
#
# Block 128:
#   bash run.sh 2 128 32 3
#
# Block 256:
#   bash run.sh 2 256 32 3
#
# CPU single-thread version:
#   bash run.sh 2 128 1 3
#
# ctest from project root:
#   ctest --test-dir build --output-on-failure
#
# ctest under build folder:
#   ctest --output-on-failure
#
# Build:
#   cmake --build . -j
# 抓資料
# 例如後續要抓每個 application 的 GPU runtime，可以直接：
# grep -E \
#   'File name|GPU DAG total runtime' \
#   results/kernel_2_dagthreads_128_cputhreads_32/repeat_1/all_applications.log
# 要找所有 repeats：
# grep -R -E \
#   'File name|GPU DAG total runtime' \
#   results/kernel_2_dagthreads_128_cputhreads_32/
# ============================================================
