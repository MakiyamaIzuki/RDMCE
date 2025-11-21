#!/bin/bash

# Set base paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${BASE_DIR}/logs"
DATASET_DIR="${BASE_DIR}/datasets/RDMCE"
DOGSTER_DATASET="${DATASET_DIR}/dogster.bin"

# Create log directory
mkdir -p "${LOG_DIR}"

# Define timeout duration in seconds (1 hour)
TIMEOUT_DURATION=3600

# List of RDMCE variants to test
RDMCE_VARIANTS=(
  "RDMCE-NP"
  "RDMCE-W4"
  "RDMCE-W8"
  "RDMCE-W12"
  "RDMCE-W16"
  "RDMCE-W20"
  "RDMCE-W24"
  "RDMCE-W28"
  "RDMCE-W32"
  "RDMCE-T16"
  "RDMCE-T24"
  "RDMCE-T32"
  "RDMCE-T48"
  "RDMCE-T64"
  "RDMCE-T96"
)

# GPU configurations for dogster test
GPU_CONFIGS=(
  "1:0"
  "2:0,1"
  "4:0,1,2,3"
  "8:0,1,2,3,4,5,6,7"
)

# Function to run a command and log output with timeout
run_and_log() {
    local cmd="$1"
    local log_file="$2"
    local dataset_name="$3"
    local method_name="$4"
    
    echo "[$(date)] Running ${method_name} on ${dataset_name}..."
    echo "[$(date)] ==== Running ${method_name} on ${dataset_name} ====" >> "${log_file}"
    echo "[$(date)] Command: ${cmd}" >> "${log_file}"
    echo "[$(date)] Timeout set to ${TIMEOUT_DURATION} seconds" >> "${log_file}"
    
    # Run command with timeout and capture time and output
    { time timeout ${TIMEOUT_DURATION} ${cmd}; } >> "${log_file}" 2>&1
    
    local exit_code=$?
    if [ ${exit_code} -eq 0 ]; then
        echo "[$(date)] Completed ${method_name} on ${dataset_name}"
        echo "[$(date)] ==== Completed ${method_name} on ${dataset_name} ====" >> "${log_file}"
        echo "" >> "${log_file}"
    elif [ ${exit_code} -eq 124 ]; then
        # Timeout command returns 124 when the command times out
        echo "[$(date)] TIMEOUT: ${method_name} on ${dataset_name} exceeded ${TIMEOUT_DURATION} seconds"
        echo "[$(date)] TIMEOUT: ${method_name} on ${dataset_name} exceeded ${TIMEOUT_DURATION} seconds" >> "${log_file}"
        echo "" >> "${log_file}"
    else
        echo "[$(date)] ERROR: Failed to run ${method_name} on ${dataset_name}"
        echo "[$(date)] ERROR: Failed to run ${method_name} on ${dataset_name} (Exit code: ${exit_code})" >> "${log_file}"
        echo "" >> "${log_file}"
    fi
}

# Process a specific RDMCE variant across all datasets
process_rdmce_variant() {
    local variant_name="$1"
    local variant_bin="${BASE_DIR}/bin/${variant_name}"
    local log_file="${LOG_DIR}/${variant_name}.log"
    
    # Check if the variant binary exists
    if [ ! -f "${variant_bin}" ]; then
        echo "[$(date)] ERROR: Binary for ${variant_name} not found at ${variant_bin}"
        echo "[$(date)] ERROR: Binary not found" > "${log_file}"
        return 1
    fi
    
    echo "[$(date)] Starting evaluation for ${variant_name} on all datasets..."
    echo "[$(date)] Starting ${variant_name} evaluations" > "${log_file}"
    
    # Process all dataset files
    for dataset in ${DATASET_DIR}/*.bin; do
        if [ -f "${dataset}" ]; then
            local dataset_name=$(basename "${dataset}" .bin)
            run_and_log "${variant_bin} -i ${dataset}" "${log_file}" "${dataset_name}" "${variant_name}"
        fi
    done
    
    echo "[$(date)] Completed all ${variant_name} evaluations" >> "${log_file}"
    echo "[$(date)] Results saved to ${log_file}"
}

# Test multi-GPU performance on dogster dataset
process_multigpu_test() {
    local log_file="${LOG_DIR}/rdmce_multigpu.log"
    local variant_bin="${BASE_DIR}/bin/RDMCE"
    
    # Check if the RDMCE binary exists
    if [ ! -f "${variant_bin}" ]; then
        echo "[$(date)] ERROR: Binary for RDMCE not found at ${variant_bin}"
        echo "[$(date)] ERROR: Binary not found" > "${log_file}"
        return 1
    fi
    
    echo "[$(date)] Starting multi-GPU evaluation on dogster dataset..."
    echo "[$(date)] Starting multi-GPU evaluations" > "${log_file}"
    
    # Test each GPU configuration
    for gpu_config in "${GPU_CONFIGS[@]}"; do
        IFS=':' read -r num_gpus gpu_ids <<< "${gpu_config}"
        echo "[$(date)] Testing with ${num_gpus} GPUs (IDs: ${gpu_ids})..."
        run_and_log "${variant_bin} -i ${DOGSTER_DATASET} -d ${gpu_ids}" "${log_file}" "dogster" "RDMCE-${num_gpus}GPUs"
    done
    
    echo "[$(date)] Completed all multi-GPU evaluations" >> "${log_file}"
    echo "[$(date)] Results saved to ${log_file}"
}

# Main execution
echo "[$(date)] Starting RDMCE variants and multi-GPU evaluation..."

# Process each RDMCE variant
for variant in "${RDMCE_VARIANTS[@]}"; do
    process_rdmce_variant "${variant}"
done

# Run multi-GPU test
# process_multigpu_test

echo "[$(date)] All evaluations completed. Results in ${LOG_DIR}"

