#!/bin/bash

set -euo pipefail

# Keep benchmark traversal order deterministic.
export LC_ALL=C


# ============================================================
# Interactive usage
#
# Run:
#
#   bash run.sh
#
# The script first asks for RUN_MODE:
#
#   0 = run every benchmark under BENCH_ROOTS
#       Save all logs under one timestamped experiment folder.
#
#   1 = run one input JSON file
#       Ask for the input filename interactively.
#       Print output to the terminal only.
#       Do not create or modify anything under results/.
#
# For RUN_MODE=1, the script asks for the input filename before
# asking for experiment configurations.
#
# Then enter one experiment configuration per line using exactly
# five integers:
#
#   DAG_KERNEL DAG_THREADS CPU_THREADS BITMAP_BITS NUM_REPEATS
#
# DAG_KERNEL:
#
#   0 = Thread
#   1 = Warp
#   2 = Block
#
# Valid DAG_THREADS values:
#
#   Thread: 1
#   Warp:   32, 64, 128, 256, 512, 1024
#   Block:  64, 128, 256, 512, 1024
#
# Valid BITMAP_BITS values:
#
#   Thread: 0
#   Warp:   0, 32, 64
#   Block:  0, 32, 64, 128, 256, 512, 1024, 2048
#
# BITMAP_BITS=0 selects exact overlap detection.
# A nonzero BITMAP_BITS value selects approximate bitmap
# overlap detection.
#
# Finish configuration input by pressing Enter on an empty line.
# The script prints a complete summary and asks for confirmation
# before building or running anything.
#
# Example 1: one Thread configuration
#
#   0 1 32 0 3
#
# Example 2: compare Exact Warp and Warp Bitmap versions
#
#   1 128 32 0  3
#   1 128 32 32 3
#   1 128 32 64 3
#
# Example 3: compare Thread, Warp, and Block
#
#   0 1   32 0   3
#   1 128 32 0   3
#   1 128 32 64  3
#   2 128 32 0   3
#   2 128 32 256 3
#
# For RUN_MODE=0, every configuration entered in one invocation
# shares one RUN_ID, for example:
#
# results/
# └── 20260725_173312/
#     ├── kernel_0_dagthreads_1_cputhreads_32_exact/
#     ├── kernel_1_dagthreads_128_cputhreads_32_exact/
#     ├── kernel_1_dagthreads_128_cputhreads_32_bitmap_32/
#     ├── kernel_1_dagthreads_128_cputhreads_32_bitmap_64/
#     ├── kernel_2_dagthreads_128_cputhreads_32_exact/
#     └── kernel_2_dagthreads_128_cputhreads_32_bitmap_256/
# ============================================================


# ============================================================
# Reject command-line configurations.
#
# This runner intentionally uses only the interactive interface
# described above so one invocation can collect multiple
# configurations under the same RUN_ID.
# ============================================================

if [ "$#" -ne 0 ]; then
    echo "[ERROR] This version of run.sh does not accept command-line arguments."
    echo "[ERROR] Run it interactively with:"
    echo
    echo "  bash run.sh"
    exit 1
fi


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
# Benchmark subfolders used by extraction-gym
#
# These are used only if an extraction-gym/data root is enabled
# in BENCH_ROOTS.
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
# Single-file configuration
#
# RUN_MODE=1 asks the user for SINGLE_FILE and searches for it in:
#
#   root/SINGLE_FOLDER/SINGLE_FILE
#   root/SINGLE_FILE
#
# RUN_MODE=1 validates the file before asking for configurations
# and never writes result files.
# ============================================================

SINGLE_FOLDER="boole"
SINGLE_FILE=""
SINGLE_INPUT_PATH=""
SINGLE_DATASET_LABEL=""

# Example input filenames:
#   c7552.rename.json
#   i7.rename.json
#   b12.rename.json
#   c2670.rename.json
#   C432.rename.json
#   mul3.rename.json
#   mul5.rename.json
#   mul32.rename.json
#   bert.rename.json
#   box_filter_3iteration_egraph.rename.json
#   box_filter_4iteration_egraph.rename.json
#   vector_2d_conv_3x3_3x3_root_168.rename.json
#   sin.rename.json
#   mul112.rename.json
#   mul80.rename.json
#   mul48.rename.json


# ============================================================
# General output configuration
# ============================================================

OUT_DIR="results"


# ============================================================
# Helper: return the display name of one DAG kernel.
# ============================================================

dag_kernel_name()
{
    case "$1" in
        0)
            printf '%s' "Thread"
            ;;

        1)
            printf '%s' "Warp"
            ;;

        2)
            printf '%s' "Block"
            ;;

        *)
            printf '%s' "Invalid"
            ;;
    esac
}


# ============================================================
# Helper: return a human-readable overlap detection name.
#
# Arguments:
#
#   $1 = BITMAP_BITS
#
# Examples:
#
#   0    -> Exact
#   32   -> Bitmap-32
#   1024 -> Bitmap-1024
# ============================================================

bitmap_mode_name()
{
    local bitmap_bits="$1"

    if [ "$bitmap_bits" -eq 0 ]; then
        printf '%s' "Exact"
    else
        printf 'Bitmap-%s' "$bitmap_bits"
    fi
}


# ============================================================
# Helper: return the folder-name suffix for one bitmap mode.
#
# Arguments:
#
#   $1 = BITMAP_BITS
#
# Examples:
#
#   0    -> exact
#   32   -> bitmap_32
#   1024 -> bitmap_1024
# ============================================================

bitmap_folder_tag()
{
    local bitmap_bits="$1"

    if [ "$bitmap_bits" -eq 0 ]; then
        printf '%s' "exact"
    else
        printf 'bitmap_%s' "$bitmap_bits"
    fi
}


# ============================================================
# Validate one configuration.
#
# Arguments:
#
#   $1 = DAG_KERNEL
#   $2 = DAG_THREADS
#   $3 = CPU_THREADS
#   $4 = BITMAP_BITS
#   $5 = NUM_REPEATS
#
# Return:
#
#   0 = valid
#   1 = invalid
# ============================================================

validate_configuration()
{
    local dag_kernel="$1"
    local dag_threads="$2"
    local cpu_threads="$3"
    local bitmap_bits="$4"
    local num_repeats="$5"

    if ! [[ "$dag_kernel" =~ ^[0-2]$ ]]; then
        echo "[ERROR] DAG_KERNEL must be 0, 1, or 2"
        return 1
    fi

    if ! [[ "$dag_threads" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] DAG_THREADS must be a positive integer"
        return 1
    fi

    if ! [[ "$cpu_threads" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] CPU_THREADS must be a positive integer"
        return 1
    fi

    if ! [[ "$bitmap_bits" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] BITMAP_BITS must be a nonnegative integer"
        return 1
    fi

    if ! [[ "$num_repeats" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] NUM_REPEATS must be a positive integer"
        return 1
    fi

    case "$dag_kernel" in
        0)
            if [ "$dag_threads" -ne 1 ]; then
                echo "[ERROR] Thread DAG kernel requires DAG_THREADS=1"
                return 1
            fi

            if [ "$bitmap_bits" -ne 0 ]; then
                echo "[ERROR] Thread DAG kernel requires BITMAP_BITS=0"
                return 1
            fi
            ;;

        1)
            case "$dag_threads" in
                32|64|128|256|512|1024)
                    ;;

                *)
                    echo "[ERROR] Warp DAG kernel requires one of these DAG_THREADS values:"
                    echo "        32, 64, 128, 256, 512, 1024"
                    return 1
                    ;;
            esac

            case "$bitmap_bits" in
                0|32|64)
                    ;;

                *)
                    echo "[ERROR] Warp DAG kernel requires one of these BITMAP_BITS values:"
                    echo "        0, 32, 64"
                    return 1
                    ;;
            esac
            ;;

        2)
            case "$dag_threads" in
                64|128|256|512|1024)
                    ;;

                *)
                    echo "[ERROR] Block DAG kernel requires one of these DAG_THREADS values:"
                    echo "        64, 128, 256, 512, 1024"
                    return 1
                    ;;
            esac

            case "$bitmap_bits" in
                0|32|64|128|256|512|1024|2048)
                    ;;

                *)
                    echo "[ERROR] Block DAG kernel requires one of these BITMAP_BITS values:"
                    echo "        0, 32, 64, 128, 256, 512, 1024, 2048"
                    return 1
                    ;;
            esac
            ;;
    esac

    return 0
}


# ============================================================
# Resolve one input filename for RUN_MODE=1.
#
# Arguments:
#
#   $1 = input JSON filename
#
# On success, this function sets:
#
#   SINGLE_INPUT_PATH
#   SINGLE_DATASET_LABEL
# ============================================================

resolve_single_file()
{
    local requested_file="$1"
    local root
    local candidate

    for root in "${BENCH_ROOTS[@]}"; do
        candidate="$root/$SINGLE_FOLDER/$requested_file"

        if [ -f "$candidate" ]; then
            SINGLE_INPUT_PATH="$candidate"
            SINGLE_DATASET_LABEL="$(basename "$root")/$SINGLE_FOLDER"
            return 0
        fi

        candidate="$root/$requested_file"

        if [ -f "$candidate" ]; then
            SINGLE_INPUT_PATH="$candidate"
            SINGLE_DATASET_LABEL="$(basename "$root")"
            return 0
        fi
    done

    return 1
}


# ============================================================
# Print the interactive user interface.
# ============================================================

print_usage_interface()
{
    cat <<'EOF_USAGE'
============================================================
GPU DAG Experiment Runner
============================================================

Step 1: Select one run mode.

  0 = Run all benchmark files.
      Logs are saved under one new timestamped folder:

        results/<RUN_ID>/

      Every configuration entered below shares the same RUN_ID.

  1 = Run one input JSON file.
      The script asks for the input filename first.
      Output is printed to the terminal only.
      Nothing is created or modified under results/.

Step 2 for RUN_MODE=1: Enter the input JSON filename.

  Example:

    mul32.rename.json

  The script searches for the file under the configured benchmark
  roots before accepting experiment configurations.

Step 3: Enter one configuration per line.

  Required format:

    DAG_KERNEL DAG_THREADS CPU_THREADS BITMAP_BITS NUM_REPEATS

  DAG_KERNEL:

    0 = Thread
    1 = Warp
    2 = Block

  Thread requirements:

    DAG_THREADS must be 1.
    BITMAP_BITS must be 0.

  Warp requirements:

    DAG_THREADS must be one of:

      32, 64, 128, 256, 512, 1024

    BITMAP_BITS must be one of:

      0, 32, 64

  Block requirements:

    DAG_THREADS must be one of:

      64, 128, 256, 512, 1024

    BITMAP_BITS must be one of:

      0, 32, 64, 128, 256, 512, 1024, 2048

  BITMAP_BITS:

    0       = Exact overlap detection
    nonzero = Approximate bitmap overlap detection

  An invalid configuration stops the script immediately.

Step 4: Press Enter on an empty line after the last configuration.

Step 5: Review the printed summary and enter y to start.

Example 1: one Thread configuration

  0 1 32 0 3

Example 2: compare Exact Warp and Warp Bitmap configurations

  1 128 32 0  3
  1 128 32 32 3
  1 128 32 64 3

Example 3: compare Thread, Warp, and Block

  0 1   32 0   3
  1 128 32 0   3
  1 128 32 64  3
  2 128 32 0   3
  2 128 32 256 3

Important:

  NUM_REPEATS belongs to one configuration. For example:

    2 128 32 256 3

  creates repeat_1, repeat_2, and repeat_3 inside the
  Block-128 Bitmap-256 configuration folder. It does not create
  three RUN_ID folders.

============================================================
EOF_USAGE
}


# ============================================================
# Interactive input
# ============================================================

print_usage_interface

echo

while true; do
    read -r -p \
        "Select RUN_MODE [0=all benchmarks, 1=single file]: " \
        RUN_MODE

    if [[ "$RUN_MODE" =~ ^[01]$ ]]; then
        break
    fi

    echo "[ERROR] RUN_MODE must be 0 or 1"
done


# ============================================================
# RUN_MODE=1 input-file prompt
#
# Ask for and validate the input file before reading configs.
# ============================================================

if [ "$RUN_MODE" -eq 1 ]; then
    echo
    read -r -p "Enter input JSON filename: " SINGLE_FILE

    if [[ -z "${SINGLE_FILE//[[:space:]]/}" ]]; then
        echo "[ERROR] Input filename cannot be empty"
        exit 1
    fi

    if [[ "$SINGLE_FILE" != *.json ]]; then
        echo "[ERROR] Input filename must end with .json"
        exit 1
    fi

    if ! resolve_single_file "$SINGLE_FILE"; then
        echo "[ERROR] File $SINGLE_FILE was not found in any benchmark root"
        echo "[ERROR] Searched both:"
        echo "        root/$SINGLE_FOLDER/$SINGLE_FILE"
        echo "        root/$SINGLE_FILE"
        exit 1
    fi

    echo "[FOUND] Input file: $SINGLE_INPUT_PATH"
fi


CONFIG_KERNELS=()
CONFIG_DAG_THREADS=()
CONFIG_CPU_THREADS=()
CONFIG_BITMAP_BITS=()
CONFIG_REPEATS=()
CONFIG_KEYS=()

echo
echo "Enter configurations one line at a time."
echo "Format: DAG_KERNEL DAG_THREADS CPU_THREADS BITMAP_BITS NUM_REPEATS"
echo "Press Enter on an empty line to finish."
echo


while true; do
    if ! read -r -p "> " configuration_line; then
        echo
        break
    fi

    # Stop on an empty or whitespace-only line.
    if [[ -z "${configuration_line//[[:space:]]/}" ]]; then
        break
    fi

    dag_kernel=""
    dag_threads=""
    cpu_threads=""
    bitmap_bits=""
    num_repeats=""
    extra_field=""

    read -r \
        dag_kernel \
        dag_threads \
        cpu_threads \
        bitmap_bits \
        num_repeats \
        extra_field \
        <<< "$configuration_line"

    if [ -z "$dag_kernel" ] ||
       [ -z "$dag_threads" ] ||
       [ -z "$cpu_threads" ] ||
       [ -z "$bitmap_bits" ] ||
       [ -z "$num_repeats" ] ||
       [ -n "$extra_field" ]; then
        echo "[ERROR] Each line must contain exactly five integers:"
        echo "        DAG_KERNEL DAG_THREADS CPU_THREADS BITMAP_BITS NUM_REPEATS"
        exit 1
    fi

    if ! validate_configuration \
        "$dag_kernel" \
        "$dag_threads" \
        "$cpu_threads" \
        "$bitmap_bits" \
        "$num_repeats"; then
        echo "[ERROR] Invalid configuration: $configuration_line"
        exit 1
    fi

    # BITMAP_BITS is part of the configuration identity.
    #
    # This allows Exact, Bitmap-32, Bitmap-64, and other bitmap
    # variants to use the same kernel/thread settings within one
    # experiment.
    configuration_key="${dag_kernel}_${dag_threads}_${cpu_threads}_${bitmap_bits}"
    duplicate_configuration=0

    for existing_key in "${CONFIG_KEYS[@]}"; do
        if [ "$existing_key" = "$configuration_key" ]; then
            duplicate_configuration=1
            break
        fi
    done

    if [ "$duplicate_configuration" -eq 1 ]; then
        echo "[ERROR] Duplicate configuration:"
        echo "        $dag_kernel $dag_threads $cpu_threads $bitmap_bits"
        echo "[ERROR] Enter each kernel/DAG-thread/CPU-thread/bitmap combination only once."
        exit 1
    fi

    CONFIG_KERNELS+=("$dag_kernel")
    CONFIG_DAG_THREADS+=("$dag_threads")
    CONFIG_CPU_THREADS+=("$cpu_threads")
    CONFIG_BITMAP_BITS+=("$bitmap_bits")
    CONFIG_REPEATS+=("$num_repeats")
    CONFIG_KEYS+=("$configuration_key")

    configuration_index="${#CONFIG_KERNELS[@]}"
    kernel_name="$(dag_kernel_name "$dag_kernel")"
    overlap_mode="$(bitmap_mode_name "$bitmap_bits")"

    echo "[ACCEPTED $configuration_index] $kernel_name: kernel=$dag_kernel, DAG threads=$dag_threads, CPU threads=$cpu_threads, overlap=$overlap_mode, BITMAP_BITS=$bitmap_bits, repeats=$num_repeats"
done


if [ "${#CONFIG_KERNELS[@]}" -eq 0 ]; then
    echo "[ERROR] No valid configurations were entered"
    exit 1
fi


# ============================================================
# Prepare one RUN_ID for the complete interactive experiment.
#
# RUN_MODE=0:
#   all configurations share this RUN_ID.
#
# RUN_MODE=1:
#   no output folder is created, so RUN_ID is not used.
# ============================================================

if [ "$RUN_MODE" -eq 0 ]; then
    SAVE_LOGS=1

    BASE_RUN_ID="$(date +%Y%m%d_%H%M%S)"
    RUN_ID="$BASE_RUN_ID"
    RUN_ID_SUFFIX=1

    # Avoid reusing an existing timestamp folder in the unlikely
    # event that two experiments begin during the same second.
    while [ -e "$OUT_DIR/$RUN_ID" ]; do
        printf -v RUN_ID \
            '%s_%02d' \
            "$BASE_RUN_ID" \
            "$RUN_ID_SUFFIX"

        RUN_ID_SUFFIX=$((RUN_ID_SUFFIX + 1))
    done

    RUN_OUT_DIR="$OUT_DIR/$RUN_ID"
else
    SAVE_LOGS=0
    RUN_ID=""
    RUN_OUT_DIR=""
fi


# ============================================================
# Confirmation summary
# ============================================================

echo
echo "============================================================"
echo "Experiment summary"
echo "============================================================"

if [ "$RUN_MODE" -eq 0 ]; then
    echo "Run mode       : 0 - all benchmark files"
    echo "Run ID         : $RUN_ID"
    echo "Results root   : $RUN_OUT_DIR"
else
    echo "Run mode       : 1 - single input file"
    echo "Input filename : $SINGLE_FILE"
    echo "Input path     : $SINGLE_INPUT_PATH"
    echo "Dataset        : $SINGLE_DATASET_LABEL"
    echo "Output         : terminal only"
    echo "Results folder : disabled"
fi

echo "Configurations : ${#CONFIG_KERNELS[@]}"
echo

for ((config_index = 0;
      config_index < ${#CONFIG_KERNELS[@]};
      ++config_index)); do
    dag_kernel="${CONFIG_KERNELS[$config_index]}"
    dag_threads="${CONFIG_DAG_THREADS[$config_index]}"
    cpu_threads="${CONFIG_CPU_THREADS[$config_index]}"
    bitmap_bits="${CONFIG_BITMAP_BITS[$config_index]}"
    num_repeats="${CONFIG_REPEATS[$config_index]}"

    kernel_name="$(dag_kernel_name "$dag_kernel")"
    overlap_mode="$(bitmap_mode_name "$bitmap_bits")"

    printf \
        '[%u] %-6s kernel=%s, DAG threads=%s, CPU threads=%s, overlap=%s, BITMAP_BITS=%s, repeats=%s\n' \
        "$((config_index + 1))" \
        "$kernel_name" \
        "$dag_kernel" \
        "$dag_threads" \
        "$cpu_threads" \
        "$overlap_mode" \
        "$bitmap_bits" \
        "$num_repeats"
done

echo "============================================================"

read -r -p "Start experiment? [y/N]: " confirmation

case "${confirmation,,}" in
    y|yes)
        ;;

    *)
        echo "[INFO] Experiment cancelled"
        exit 0
        ;;
esac


# ============================================================
# Compile once after the user confirms the complete experiment.
#
# This script is expected to run inside:
#
#   ~/gpu_extraction/build
# ============================================================

echo
echo "============================================================"
echo "[INFO] Building project"
echo "============================================================"

cmake ..
cmake --build . -j

if [ "$SAVE_LOGS" -eq 1 ]; then
    mkdir -p "$RUN_OUT_DIR"
fi


# ============================================================
# Write the header of one repeat log.
#
# Used only by RUN_MODE=0.
# ============================================================

initialize_repeat_log()
{
    if [ "$SAVE_LOGS" -eq 0 ]; then
        return
    fi

    : > "$CURRENT_REPEAT_LOG"

    {
        echo "============================================================"
        echo "GPU DAG experiment"
        echo "============================================================"
        echo "Run ID           : $RUN_ID"
        echo "Repeat           : $CURRENT_REPEAT/$NUM_REPEATS"
        echo "DAG kernel       : $DAG_KERNEL_NAME ($DAG_KERNEL)"
        echo "DAG threads      : $DAG_THREADS"
        echo "CPU threads      : $CPU_THREADS"
        echo "Overlap detection: $BITMAP_MODE"
        echo "BITMAP_BITS      : $BITMAP_BITS"
        echo "Run mode         : $RUN_MODE"
        echo "Output file      : $CURRENT_REPEAT_LOG"
        echo "Start time       : $(date --iso-8601=seconds)"
        echo "============================================================"
        echo
    } >> "$CURRENT_REPEAT_LOG"
}


# ============================================================
# Run one benchmark.
#
# Arguments:
#
#   $1 = input JSON file
#   $2 = dataset label
#
# RUN_MODE=0:
#   tee appends output to CURRENT_REPEAT_LOG.
#
# RUN_MODE=1:
#   CURRENT_REPEAT_LOG=/dev/null, so tee prints to the terminal
#   without storing output.
#
# simple.cpp currently reads:
#
#   argv[1] = input file
#   argv[2] = DAG_KERNEL
#   argv[3] = DAG_THREADS
#   argv[4] = BITMAP_BITS
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
    echo "Overlap     : $BITMAP_MODE"
    echo "BITMAP_BITS : $BITMAP_BITS"

    if [ "$SAVE_LOGS" -eq 1 ]; then
        echo "Output file : $CURRENT_REPEAT_LOG"
    else
        echo "Output      : terminal only"
    fi

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
        echo "Overlap     : $BITMAP_MODE"
        echo "BITMAP_BITS : $BITMAP_BITS"
        echo "Start time  : $(date --iso-8601=seconds)"
        echo "------------------------------------------------------------"

        OMP_NUM_THREADS="$CPU_THREADS" \
            ./main/simple \
            "$input_file" \
            "$DAG_KERNEL" \
            "$DAG_THREADS" \
            "$BITMAP_BITS"

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
# Run every application under every benchmark root.
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
# Run the input file that was resolved during interactive setup.
#
# No result directory or log file is created.
# ============================================================

run_single_file()
{
    echo "[INFO] Running single file: $SINGLE_INPUT_PATH"

    run_benchmark \
        "$SINGLE_INPUT_PATH" \
        "$SINGLE_DATASET_LABEL"
}


# ============================================================
# Run one complete configuration.
#
# Configuration order:
#
#   configuration 1
#     repeat 1
#     repeat 2
#     ...
#
#   configuration 2
#     repeat 1
#     repeat 2
#     ...
# ============================================================

run_configuration()
{
    local config_index="$1"

    DAG_KERNEL="${CONFIG_KERNELS[$config_index]}"
    DAG_THREADS="${CONFIG_DAG_THREADS[$config_index]}"
    CPU_THREADS="${CONFIG_CPU_THREADS[$config_index]}"
    BITMAP_BITS="${CONFIG_BITMAP_BITS[$config_index]}"
    NUM_REPEATS="${CONFIG_REPEATS[$config_index]}"

    DAG_KERNEL_NAME="$(dag_kernel_name "$DAG_KERNEL")"
    BITMAP_MODE="$(bitmap_mode_name "$BITMAP_BITS")"
    BITMAP_FOLDER_TAG="$(bitmap_folder_tag "$BITMAP_BITS")"

    CONFIG_NAME="kernel_${DAG_KERNEL}_dagthreads_${DAG_THREADS}_cputhreads_${CPU_THREADS}_${BITMAP_FOLDER_TAG}"

    if [ "$SAVE_LOGS" -eq 1 ]; then
        CONFIG_OUT_DIR="$RUN_OUT_DIR/$CONFIG_NAME"
        mkdir -p "$CONFIG_OUT_DIR"
    else
        CONFIG_OUT_DIR=""
    fi

    echo
    echo "============================================================"
    echo "[INFO] Starting configuration $((config_index + 1))/${#CONFIG_KERNELS[@]}"
    echo "============================================================"
    echo "DAG kernel       : $DAG_KERNEL_NAME ($DAG_KERNEL)"
    echo "DAG threads      : $DAG_THREADS"
    echo "CPU threads      : $CPU_THREADS"
    echo "Overlap detection: $BITMAP_MODE"
    echo "BITMAP_BITS      : $BITMAP_BITS"
    echo "Number of repeats: $NUM_REPEATS"

    if [ "$SAVE_LOGS" -eq 1 ]; then
        echo "Output directory : $CONFIG_OUT_DIR"
    else
        echo "Output            : terminal only"
    fi

    echo "============================================================"

    for ((CURRENT_REPEAT = 1;
          CURRENT_REPEAT <= NUM_REPEATS;
          ++CURRENT_REPEAT)); do
        if [ "$SAVE_LOGS" -eq 1 ]; then
            CURRENT_REPEAT_OUT_DIR="$CONFIG_OUT_DIR/repeat_${CURRENT_REPEAT}"
            CURRENT_REPEAT_LOG="$CURRENT_REPEAT_OUT_DIR/all_applications.log"

            mkdir -p "$CURRENT_REPEAT_OUT_DIR"
            initialize_repeat_log
        else
            CURRENT_REPEAT_OUT_DIR=""
            CURRENT_REPEAT_LOG="/dev/null"
        fi

        RUN_COUNT=0

        echo
        echo "############################################################"
        echo "[INFO] Starting repeat $CURRENT_REPEAT/$NUM_REPEATS"

        if [ "$SAVE_LOGS" -eq 1 ]; then
            echo "[INFO] Output file: $CURRENT_REPEAT_LOG"
        else
            echo "[INFO] Output: terminal only"
        fi

        echo "############################################################"

        case "$RUN_MODE" in
            0)
                run_all_benchmarks
                ;;

            1)
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
            echo "DAG kernel            : $DAG_KERNEL_NAME ($DAG_KERNEL)"
            echo "DAG threads           : $DAG_THREADS"
            echo "CPU threads           : $CPU_THREADS"
            echo "Overlap detection     : $BITMAP_MODE"
            echo "BITMAP_BITS           : $BITMAP_BITS"
            echo "End time              : $(date --iso-8601=seconds)"
            echo "============================================================"
        } >> "$CURRENT_REPEAT_LOG"

        echo
        echo "############################################################"
        echo "[INFO] Completed repeat $CURRENT_REPEAT/$NUM_REPEATS"
        echo "[INFO] Applications executed: $RUN_COUNT"

        if [ "$SAVE_LOGS" -eq 1 ]; then
            echo "[INFO] Output file: $CURRENT_REPEAT_LOG"
        else
            echo "[INFO] Output: terminal only"
        fi

        echo "############################################################"
    done

    echo
    echo "============================================================"
    echo "[INFO] Completed configuration $((config_index + 1))/${#CONFIG_KERNELS[@]}"
    echo "============================================================"
}


# ============================================================
# Execute every accepted configuration under the same RUN_ID.
# ============================================================

for ((config_index = 0;
      config_index < ${#CONFIG_KERNELS[@]};
      ++config_index)); do
    run_configuration "$config_index"
done


# ============================================================
# Final summary
# ============================================================

echo
echo "============================================================"
echo "[INFO] Experiment completed"
echo "============================================================"
echo "Run mode             : $RUN_MODE"
echo "Configurations       : ${#CONFIG_KERNELS[@]}"

if [ "$SAVE_LOGS" -eq 1 ]; then
    echo "Run ID               : $RUN_ID"
    echo "Results root         : $RUN_OUT_DIR"
else
    echo "Input file           : $SINGLE_INPUT_PATH"
    echo "Results directory    : disabled"
    echo "Log output           : terminal only"
fi

echo "============================================================"


# ============================================================
# Personal notes
#
# Start the interactive runner:
#   bash run.sh
#
# RUN_MODE:
#   0 = run all benchmark files and save logs
#   1 = ask for one input JSON file and print only to terminal
#
# RUN_MODE=1 interaction order:
#   1. Select RUN_MODE=1
#   2. Enter the input JSON filename
#   3. Enter one or more configurations
#   4. Press Enter on an empty line
#   5. Review the summary and enter y
#
# Configuration format:
#   DAG_KERNEL DAG_THREADS CPU_THREADS BITMAP_BITS NUM_REPEATS
#
# Thread Exact:
#   0 1 32 0 3
#
# Warp 32 threads, Exact:
#   1 32 32 0 3
#
# Warp 32 threads, Bitmap-32:
#   1 32 32 32 3
#
# Warp 32 threads, Bitmap-64:
#   1 32 32 64 3
#
# Warp 64 threads, Exact:
#   1 64 32 0 3
#
# Warp 64 threads, Bitmap-32:
#   1 64 32 32 3
#
# Warp 64 threads, Bitmap-64:
#   1 64 32 64 3
#
# Warp 128 threads, Exact:
#   1 128 32 0 3
#
# Warp 128 threads, Bitmap-32:
#   1 128 32 32 3
#
# Warp 128 threads, Bitmap-64:
#   1 128 32 64 3
#
# Warp 256 threads, Exact:
#   1 256 32 0 3
#
# Warp 256 threads, Bitmap-32:
#   1 256 32 32 3
#
# Warp 256 threads, Bitmap-64:
#   1 256 32 64 3
#
# Warp 512 threads, Exact:
#   1 512 32 0 3
#
# Warp 512 threads, Bitmap-32:
#   1 512 32 32 3
#
# Warp 512 threads, Bitmap-64:
#   1 512 32 64 3
#
# Warp 1024 threads, Exact:
#   1 1024 32 0 3
#
# Warp 1024 threads, Bitmap-32:
#   1 1024 32 32 3
#
# Warp 1024 threads, Bitmap-64:
#   1 1024 32 64 3
#
# Block 64 threads, Exact:
#   2 64 32 0 3
#
# Block 64 threads, Bitmap-32:
#   2 64 32 32 3
#
# Block 64 threads, Bitmap-64:
#   2 64 32 64 3
#
# Block 64 threads, Bitmap-128:
#   2 64 32 128 3
#
# Block 64 threads, Bitmap-256:
#   2 64 32 256 3
#
# Block 64 threads, Bitmap-512:
#   2 64 32 512 3
#
# Block 64 threads, Bitmap-1024:
#   2 64 32 1024 3
#
# Block 64 threads, Bitmap-2048:
#   2 64 32 2048 3
#
# Block 128 threads, Exact:
#   2 128 32 0 3
#
# Block 128 threads, Bitmap-32:
#   2 128 32 32 3
#
# Block 128 threads, Bitmap-64:
#   2 128 32 64 3
#
# Block 128 threads, Bitmap-128:
#   2 128 32 128 3
#
# Block 128 threads, Bitmap-256:
#   2 128 32 256 3
#
# Block 128 threads, Bitmap-512:
#   2 128 32 512 3
#
# Block 128 threads, Bitmap-1024:
#   2 128 32 1024 3
#
# Block 128 threads, Bitmap-2048:
#   2 128 32 2048 3
#
# Block 256 threads, Exact:
#   2 256 32 0 3
#
# Block 256 threads, Bitmap-32:
#   2 256 32 32 3
#
# Block 256 threads, Bitmap-64:
#   2 256 32 64 3
#
# Block 256 threads, Bitmap-128:
#   2 256 32 128 3
#
# Block 256 threads, Bitmap-256:
#   2 256 32 256 3
#
# Block 256 threads, Bitmap-512:
#   2 256 32 512 3
#
# Block 256 threads, Bitmap-1024:
#   2 256 32 1024 3
#
# Block 256 threads, Bitmap-2048:
#   2 256 32 2048 3
#
# Block 512 threads, Exact:
#   2 512 32 0 3
#
# Block 512 threads, Bitmap-32:
#   2 512 32 32 3
#
# Block 512 threads, Bitmap-64:
#   2 512 32 64 3
#
# Block 512 threads, Bitmap-128:
#   2 512 32 128 3
#
# Block 512 threads, Bitmap-256:
#   2 512 32 256 3
#
# Block 512 threads, Bitmap-512:
#   2 512 32 512 3
#
# Block 512 threads, Bitmap-1024:
#   2 512 32 1024 3
#
# Block 512 threads, Bitmap-2048:
#   2 512 32 2048 3
#
# Block 1024 threads, Exact:
#   2 1024 32 0 3
#
# Block 1024 threads, Bitmap-32:
#   2 1024 32 32 3
#
# Block 1024 threads, Bitmap-64:
#   2 1024 32 64 3
#
# Block 1024 threads, Bitmap-128:
#   2 1024 32 128 3
#
# Block 1024 threads, Bitmap-256:
#   2 1024 32 256 3
#
# Block 1024 threads, Bitmap-512:
#   2 1024 32 512 3
#
# Block 1024 threads, Bitmap-1024:
#   2 1024 32 1024 3
#
# Block 1024 threads, Bitmap-2048:
#   2 1024 32 2048 3
#
# CPU single-thread version:
#   2 128 1 0 3
#   2 128 1 256 3
#
# Example complete comparison entered in one invocation:
#   0 1   32 0    3
#
#   1 32  32 0    3
#   1 32  32 32   3
#   1 32  32 64   3
#
#   1 64  32 0    3
#   1 64  32 32   3
#   1 64  32 64   3
#
#   1 128 32 0    3
#   1 128 32 32   3
#   1 128 32 64   3
#
#   1 256 32 0    3
#   1 256 32 32   3
#   1 256 32 64   3
#
#   2 64  32 0    3
#   2 64  32 32   3
#   2 64  32 64   3
#   2 64  32 128  3
#   2 64  32 256  3
#   2 64  32 512  3
#   2 64  32 1024 3
#   2 64  32 2048 3
#
#   2 128 32 0    3
#   2 128 32 32   3
#   2 128 32 64   3
#   2 128 32 128  3
#   2 128 32 256  3
#   2 128 32 512  3
#   2 128 32 1024 3
#   2 128 32 2048 3
#
# All configurations above share one timestamped RUN_ID when
# RUN_MODE=0.
#
# Result folder examples:
#
#   results/<RUN_ID>/kernel_0_dagthreads_1_cputhreads_32_exact/
#
#   results/<RUN_ID>/kernel_1_dagthreads_128_cputhreads_32_exact/
#
#   results/<RUN_ID>/kernel_1_dagthreads_128_cputhreads_32_bitmap_32/
#
#   results/<RUN_ID>/kernel_1_dagthreads_128_cputhreads_32_bitmap_64/
#
#   results/<RUN_ID>/kernel_2_dagthreads_128_cputhreads_32_exact/
#
#   results/<RUN_ID>/kernel_2_dagthreads_128_cputhreads_32_bitmap_1024/
#
# ctest from project root:
#   ctest --test-dir build --output-on-failure
#
# ctest under build folder:
#   ctest --output-on-failure
#
# Build:
#   cmake --build . -j
#
# 抓資料
#
# 查看已有的 experiment IDs：
#   ls results/
#
# 例如：
#   RUN_ID=20260725_173312
#
# 例如後續要抓每個 application 的 GPU runtime，可以直接：
# grep -E \
#   'File name|GPU DAG total runtime' \
#   results/$RUN_ID/kernel_2_dagthreads_128_cputhreads_32_exact/repeat_1/all_applications.log
#
# 抓 Block Bitmap-1024 的結果：
# grep -E \
#   'File name|GPU DAG total runtime' \
#   results/$RUN_ID/kernel_2_dagthreads_128_cputhreads_32_bitmap_1024/repeat_1/all_applications.log
#
# 抓 Warp Bitmap-64 的結果：
# grep -E \
#   'File name|GPU DAG total runtime' \
#   results/$RUN_ID/kernel_1_dagthreads_128_cputhreads_32_bitmap_64/repeat_1/all_applications.log
#
# 要找該 configuration 的所有 repeats：
# grep -R -E \
#   'File name|GPU DAG total runtime' \
#   results/$RUN_ID/kernel_2_dagthreads_128_cputhreads_32_bitmap_1024/
#
# 要搜尋該次完整 experiment：
# grep -R -E \
#   'File name|GPU DAG total runtime' \
#   results/$RUN_ID/
#
# 完整的 cmake 所有東西：
# cmake -S . -B build
# cmake --build build -j "$(nproc)"
#
# 只 cmake simple：
# cmake --build build -j --target simple
#
# INPUT=~/smoothe-artifact/dataset/eboost_boolE/mul32.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_esyn/i7.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_esyn/c2670.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_esyn/mul3.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_diospyros/vector_2d_conv_3x3_3x3_root_168.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_diospyros/direct_recexpr_2_root_18.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_diospyros/direct_recexpr_root_18.rename.json
#
# INPUT=<your-input-json>
#
# Warp, Exact:
# ./build/main/simple "$INPUT" 1 32 0
# ./build/main/simple "$INPUT" 1 64 0
# ./build/main/simple "$INPUT" 1 128 0
#
# Warp, Bitmap-32:
# ./build/main/simple "$INPUT" 1 32 32
# ./build/main/simple "$INPUT" 1 64 32
# ./build/main/simple "$INPUT" 1 128 32
#
# Warp, Bitmap-64:
# ./build/main/simple "$INPUT" 1 32 64
# ./build/main/simple "$INPUT" 1 64 64
# ./build/main/simple "$INPUT" 1 128 64
#
# Thread, Exact only:
# ./build/main/simple "$INPUT" 0 1 0
#
# Block, Exact:
# ./build/main/simple "$INPUT" 2 64 0
#
# Block, Bitmap-32:
# ./build/main/simple "$INPUT" 2 64 32
#
# Block, Bitmap-64:
# ./build/main/simple "$INPUT" 2 64 64
#
# Block, Bitmap-128:
# ./build/main/simple "$INPUT" 2 64 128
#
# Block, Bitmap-256:
# ./build/main/simple "$INPUT" 2 64 256
#
# Block, Bitmap-512:
# ./build/main/simple "$INPUT" 2 64 512
#
# Block, Bitmap-1024:
# ./build/main/simple "$INPUT" 2 64 1024
#
# Block, Bitmap-2048:
# ./build/main/simple "$INPUT" 2 64 2048

# for log in \
#   results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_64/repeat_*/all_applications.log
# do
#   echo "===== $(basename "$(dirname "$log")") ====="
#   grep -h -E \
#     '^(Objective value = .+|No feasible solution found\.)$' \
#     "$log"
# done
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build -j "$(nproc)"

# ctest --test-dir build --output-on-failure
# mul32.rename.json
# 2 64 32 64 1
# ============================================================

# kernel_2_dagthreads_64_cputhreads_32_bitmap_64
# kernel_2_dagthreads_64_cputhreads_32_bitmap_128
# kernel_2_dagthreads_64_cputhreads_32_bitmap_256
# kernel_2_dagthreads_64_cputhreads_32_bitmap_512
# kernel_2_dagthreads_64_cputhreads_32_bitmap_1024

# kernel_2_dagthreads_128_cputhreads_32_bitmap_64
# kernel_2_dagthreads_128_cputhreads_32_bitmap_128
# kernel_2_dagthreads_128_cputhreads_32_bitmap_256
# kernel_2_dagthreads_128_cputhreads_32_bitmap_512
# kernel_2_dagthreads_128_cputhreads_32_bitmap_1024

# kernel_2_dagthreads_256_cputhreads_32_bitmap_64
# kernel_2_dagthreads_256_cputhreads_32_bitmap_128
# kernel_2_dagthreads_256_cputhreads_32_bitmap_256
# kernel_2_dagthreads_256_cputhreads_32_bitmap_512
# kernel_2_dagthreads_256_cputhreads_32_bitmap_1024


# for log in \
#   results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_*/all_applications.log
# do
#   echo "===== $(basename "$(dirname "$log")") ====="
#   grep -h -E \
#     '^(Objective value = .+|No feasible solution found\.)$' \
#     "$log"
# done

# for log in results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_{1,2,3}/all_applications.log
# do
#   awk -F'[: s]+' '
#     /APPLICATION_BEGIN/ { last=""; nofeas=0 }
#     /Incumbent Solution:/ { last=$3 }
#     /No feasible solution found\./ { nofeas=1 }
#     /APPLICATION_END/ {
#       if (nofeas) print "No feasible solution found."
#       else print last
#     }
#   ' "$log"
# done > tmp2.log

# grep -oP 'GPU tree construction runtime: \K[0-9.]+' results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_*/all_applications.log
# grep -oP 'GPU DAG total runtime: \K[0-9.]+' results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_*/all_applications.log
# grep -oP 'CPU DAG construction runtime: \K[0-9.]+' results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_*/all_applications.log


# 20260726 用的一些筆記
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j "$(nproc)"
# ctest --test-dir build --output-on-failure

# INPUT=~/smoothe-artifact/dataset/eboost_boolE/mul32.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_boolE/mul48.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_esyn/i7.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_esyn/C432.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_esyn/c2670.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_esyn/mul3.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_diospyros/vector_2d_conv_3x3_3x3_root_168.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_diospyros/direct_recexpr_2_root_18.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_diospyros/direct_recexpr_root_18.rename.json

# ./build/main/simple "$INPUT" 2 64 64

# for INPUT in ~/smoothe-artifact/dataset/eboost_esyn/*.rename.json; do echo "Running: $INPUT"; ./build/main/simple "$INPUT" 2 64 64; done

# INPUT=~/smoothe-artifact/dataset/eboost_boolE/mul32.rename.json

# for run in 1 2 3; do
#   echo "===== Exact Block run ${run} ====="
#   ./build/main/simple "$INPUT" 2 64 0
# done

# for run in 1 2 3; do
#   echo "===== Exact Block run ${run} ====="
#   ./build/main/simple "$INPUT" 2 64 64
# done

# INPUT=~/smoothe-artifact/dataset/eboost_boolE_map/mul16_map.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_boolE_map/mul32_map.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_boolE_map/mul48_map.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_tensat/bert.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_tensat/nasneta.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_rover/box_filter_3iteration_egraph.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_rover/box_filter_4iteration_egraph.rename.json (BITMAP 會爆)
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_rover/mcm_3_7_21_original_8iteration_egraph.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_rover/mcm_3_7_21_original_9iteration_egraph.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_diospyros/direct_recexpr_root_18.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_diospyros/direct_recexpr_2_root_18.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_diospyros/vector_2d_conv_3x3_3x3_root_168.rename.json (這個 bitmap 很厲害)
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_flexc/fdbbd205-1519-4880-9a24-07b2ea772d16.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_smoothe_flexc/fea54919-328c-4f00-aedc-2698d17914b6.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_morphic/div.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_boolE_map/mul112_map.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_boolE_map/mul128_map.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_boolE/mul112.rename.json
# INPUT=~/smoothe-artifact/dataset/eboost_boolE/mul128.rename.json


# 20260729
# for log in \
#   results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_*/all_applications.log
# do
#   echo "===== $(basename "$(dirname "$log")") ====="
#   grep -h -E \
#     '^(Objective value = .+|No feasible solution found\.)$' \
#     "$log"
# done

# for log in results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_{1,2,3}/all_applications.log
# do
#   awk -F'[: s]+' '
#     /APPLICATION_BEGIN/ { last=""; nofeas=0 }
#     /Incumbent Solution:/ { last=$3 }
#     /No feasible solution found\./ { nofeas=1 }
#     /APPLICATION_END/ {
#       if (nofeas) print "No feasible solution found."
#       else print last
#     }
#   ' "$log"
# done > tmp2.log

# cmake -S . -B build
# cmake --build build -j "$(nproc)"
# grep -oP 'GPU tree construction runtime: \K[0-9.]+' results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_*/all_applications.log
# grep -oP 'GPU DAG total runtime: \K[0-9.]+' results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_*/all_applications.log
# grep -oP 'CPU DAG construction runtime: \K[0-9.]+' results/20260726_235909/kernel_2_dagthreads_64_cputhreads_32_bitmap_512/repeat_*/all_applications.log
