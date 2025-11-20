#!/bin/bash

# Set base paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${BASE_DIR}/logs"

# Create log directory
mkdir -p "${LOG_DIR}"

# Define log files for each method
RDMCE_LOG="${LOG_DIR}/rdmce_all.log"
G2_AIMD_LOG="${LOG_DIR}/g2_aimd_all.log"
MCE_GPU_PX_LOG="${LOG_DIR}/mce_gpu_px_all.log"
MCE_GPU_P_LOG="${LOG_DIR}/mce_gpu_p_all.log"

# Define timeout duration in seconds
TIMEOUT_DURATION=3600

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

# Process RDMCE
process_rdmce() {
    local dataset_file="$1"
    local dataset_name=$(basename "${dataset_file}" .bin)
    
    run_and_log "${BASE_DIR}/bin/RDMCE -i ${dataset_file}" "${RDMCE_LOG}" "${dataset_name}" "RDMCE"
}

# Process G2-AIMD
process_g2_aimd() {
    local dataset_file="$1"
    local dataset_name=$(basename "${dataset_file}" .bin)
    
    run_and_log "${BASE_DIR}/bin/bk -dg ${dataset_file} -h 2 -pn 1 -cn 1 -t 1 -qs 1" "${G2_AIMD_LOG}" "${dataset_name}" "G2-AIMD"
}

# Process MCE-GPU with different parameters
process_mce_gpu() {
    local dataset_file="$1"
    local dataset_name=$(basename "${dataset_file}" .bel)
    local param="$2"
    local log_file="${MCE_GPU_PX_LOG}"
    
    if [ "${param}" = "p" ]; then
        log_file="${MCE_GPU_P_LOG}"
    fi
    
    run_and_log "${BASE_DIR}/bin/mce_gpu -d 0 -p l2 -w wl -g ${dataset_file} -m mce -i ${param}" "${log_file}" "${dataset_name}" "MCE-GPU-${param^^}"
}

# Main execution
echo "[$(date)] Starting overall evaluation..."

# Clear previous log files if they exist
: > "${RDMCE_LOG}"
: > "${G2_AIMD_LOG}"
: > "${MCE_GPU_PX_LOG}"
: > "${MCE_GPU_P_LOG}"

# Process RDMCE datasets
echo "Processing RDMCE datasets..."
echo "[$(date)] Starting RDMCE evaluations" >> "${RDMCE_LOG}"
for dataset in ${BASE_DIR}/datasets/RDMCE/*.bin; do
    if [ -f "${dataset}" ]; then
        process_rdmce "${dataset}"
    fi
done
echo "[$(date)] Completed all RDMCE evaluations" >> "${RDMCE_LOG}"

# Process G2-AIMD datasets
echo "Processing G2-AIMD datasets..."
echo "[$(date)] Starting G2-AIMD evaluations" >> "${G2_AIMD_LOG}"
for dataset in ${BASE_DIR}/datasets/G2-AIMD/*.bin; do
    if [ -f "${dataset}" ]; then
        process_g2_aimd "${dataset}"
    fi
done
echo "[$(date)] Completed all G2-AIMD evaluations" >> "${G2_AIMD_LOG}"

# Process MCE-GPU datasets
echo "Processing MCE-GPU datasets..."
echo "[$(date)] Starting MCE-GPU-PX evaluations" >> "${MCE_GPU_PX_LOG}"
echo "[$(date)] Starting MCE-GPU-P evaluations" >> "${MCE_GPU_P_LOG}"

for dataset in ${BASE_DIR}/datasets/MCE-GPU/*.bel; do
    if [ -f "${dataset}" ]; then
        # Run with px parameter
        process_mce_gpu "${dataset}" "px"
        # Run with p parameter
        process_mce_gpu "${dataset}" "p"
    fi
done

echo "[$(date)] Completed all MCE-GPU-PX evaluations" >> "${MCE_GPU_PX_LOG}"
echo "[$(date)] Completed all MCE-GPU-P evaluations" >> "${MCE_GPU_P_LOG}"

echo "[$(date)] Overall evaluation completed. Results in ${LOG_DIR}"
echo "Log files created:"
echo "- ${RDMCE_LOG}"
echo "- ${G2_AIMD_LOG}"
echo "- ${MCE_GPU_PX_LOG}"
echo "- ${MCE_GPU_P_LOG}"