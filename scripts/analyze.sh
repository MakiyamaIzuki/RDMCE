#!/bin/bash

# RDMCE Log Analysis Script
# Extracts running time for each dataset from log files, uses X for failures
# Results saved in CSV format with specified dataset order

echo "=========================================="
echo "RDMCE Log Analysis Tool"
echo "=========================================="

# Ensure logs directory exists
LOG_DIR="/data/pz/RDMCE/logs"
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Cannot find logs directory $LOG_DIR"
    exit 1
fi

# Set output files directly in logs directory
CSV_FILE="$LOG_DIR/analysis_results.csv"
MULTI_GPU_FILE="$LOG_DIR/res_multiGPU.csv"

# Backup previous files if they exist
[ -f "$CSV_FILE" ] && cp "$CSV_FILE" "${CSV_FILE}.bak"
[ -f "$MULTI_GPU_FILE" ] && cp "$MULTI_GPU_FILE" "${MULTI_GPU_FILE}.bak"

# Define dataset list in specified order
datasets=(
    "flickr"
    "twitter-higgs"
    "wiki-link"
    "facebook"
    "stackoverflow"
    "web-ClueWeb09-50m"
    "soc-sinaweibo"
    "soc-orkut-dir"
    "com-orkut"
    "com-friendster"
    "wikipedia_link_en"
    "dogster"
)

# Define dataset abbreviations
declare -A dataset_abbr
dataset_abbr["flickr"]="fi"
dataset_abbr["twitter-higgs"]="hig"
dataset_abbr["wiki-link"]="wlk"
dataset_abbr["facebook"]="fb"
dataset_abbr["stackoverflow"]="st"
dataset_abbr["web-ClueWeb09-50m"]="cw9"
dataset_abbr["soc-sinaweibo"]="ssn"
dataset_abbr["soc-orkut-dir"]="okd"
dataset_abbr["com-orkut"]="or"
dataset_abbr["com-friendster"]="frd"
dataset_abbr["wikipedia_link_en"]="wen"
dataset_abbr["dogster"]="dg"

# Convert time format to seconds
time_to_seconds() {
    local time_str="$1"
    # Handle format like "0m30.5s" or "5m10.2s"
    if [[ "$time_str" =~ ^([0-9]+)m([0-9.]+)s$ ]]; then
        local minutes=${BASH_REMATCH[1]}
        local seconds=${BASH_REMATCH[2]}
        echo "scale=2; $minutes * 60 + $seconds" | bc
    # Handle format like "123.45s" or just "123"
    elif [[ "$time_str" =~ ^[0-9.]+$ ]] || [[ "$time_str" =~ ^[0-9.]+s$ ]]; then
        echo "$time_str" | sed 's/s$//'
    # Handle milliseconds format like "1234.56ms"
    elif [[ "$time_str" =~ ^([0-9.]+)ms$ ]]; then
        local ms=${BASH_REMATCH[1]}
        echo "scale=3; $ms / 1000" | bc
    else
        echo "$time_str"
    fi
}

# Analyze RDMCE log
analyze_rdmce_log() {
    local log_file="$1"
    local dataset="$2"
    
    # Get the specific dataset section
    local section=$(grep -A 500 "=== Running .* on $dataset ===" "$log_file" | grep -B 500 "=== Completed .* on $dataset ===" || grep -A 500 "=== Running .* on $dataset ===" "$log_file")
    
    # Check if section exists
    if [ -z "$section" ]; then
        return 1
    fi
    
    # Check if failed
    if echo "$section" | grep -q "ERROR: Failed to run"; then
        echo "X"
        return 0
    fi
    
    # Extract MCE GPU Time (only get the first match to avoid multiple values)
    local gpu_time=$(echo "$section" | grep -m 1 "MCE GPU Time:" | awk '{print $4}' | sed 's/[^0-9.]//g')
    if [ -n "$gpu_time" ]; then
        # Ensure it's a valid number
        if [[ "$gpu_time" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "$gpu_time"
            return 0
        fi
    fi
    
    echo "X"
}

# Analyze MCE-GPU log
analyze_mce_gpu_log() {
    local log_file="$1"
    local dataset="$2"
    
    # First check if the dataset run failed
    if grep -q "ERROR: Failed to run MCE-GPU-[PX]* on $dataset" "$log_file"; then
        echo "X"
        return 0
    fi
    
    # Use the 4-equals format to find the specific dataset section
    # This is the most accurate way to associate count time with the correct dataset
    local section=$(grep -A 500 "==== Running MCE-GPU-P on $dataset ====" "$log_file" | grep -B 500 "==== Completed MCE-GPU-P on $dataset ====" || grep -A 500 "==== Running MCE-GPU-P on $dataset ====" "$log_file")
    
    # If we can't find the P variant, try the PX variant
    if [ -z "$section" ]; then
        section=$(grep -A 500 "==== Running MCE-GPU-PX on $dataset ====" "$log_file" | grep -B 500 "==== Completed MCE-GPU-PX on $dataset ====" || grep -A 500 "==== Running MCE-GPU-PX on $dataset ====" "$log_file")
    fi
    
    # Extract count time from this specific section
    if [ -n "$section" ]; then
        local count_time=$(echo "$section" | grep "count time" | awk '{print $4}' | sed 's/s$//' | sed 's/[^0-9.]//g')
        if [ -n "$count_time" ] && [[ "$count_time" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "$count_time"
            return 0
        fi
    fi
    
    # For backwards compatibility, try the old method if the 4-equals format doesn't work
    local dataset_line=$(grep -n "Completed.*$dataset" "$log_file" | head -1 | cut -d':' -f1)
    if [ -n "$dataset_line" ]; then
        # Look backwards from the completion line
        local start_line=$((dataset_line - 200))
        if [ $start_line -lt 1 ]; then
            start_line=1
        fi
        
        # Extract count time from this range
        local count_time=$(sed -n "${start_line},${dataset_line}p" "$log_file" | grep "count time" | awk '{print $4}' | sed 's/s$//' | sed 's/[^0-9.]//g' | tail -1)
        
        if [ -n "$count_time" ] && [[ "$count_time" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "$count_time"
            return 0
        fi
    fi
    
    # If all else fails, return X
    echo "X"
}

# Analyze G2-AIMD log
analyze_g2_aimd_log() {
    local log_file="$1"
    local dataset="$2"
    
    # Get the specific dataset section with correct 4-equals format
    local section=$(grep -A 500 "==== Running G2-AIMD on $dataset ====" "$log_file" | grep -B 500 "==== Completed G2-AIMD on $dataset ====" || grep -A 500 "==== Running G2-AIMD on $dataset ====" "$log_file")
    
    # Check if section exists
    if [ -z "$section" ]; then
        return 1
    fi
    
    # Check if failed or timeout or Host Overflow error
    if echo "$section" | grep -q "ERROR\|Failed\|TIMEOUT\|Host Overflow"; then
        echo "X"
        return 0
    fi
    
    # Extract the last elapsed_time in the section (closest to the end)
    local elapsed_time=$(echo "$section" | grep "elapsed_time=" | tail -1 | awk -F'=' '{print $2}' | sed 's/ms$//' | sed 's/[^0-9.]//g')
    if [ -n "$elapsed_time" ]; then
        # Ensure it's a valid number
        if [[ "$elapsed_time" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            time_to_seconds "${elapsed_time}ms"  # Convert milliseconds to seconds
            return 0
        fi
    fi
    
    echo "X"
}

# Function to get result for a specific dataset and log file
get_dataset_result() {
    local dataset="$1"
    local log_file="$2"
    local log_type="$3"
    local time_result="-"
    
    case "$log_type" in
        "rdmce")
            time_result=$(analyze_rdmce_log "$log_file" "$dataset")
            if [ -z "$time_result" ]; then
                time_result="-"
            fi
            ;;
        "mce_gpu")
            time_result=$(analyze_mce_gpu_log "$log_file" "$dataset")
            if [ -z "$time_result" ]; then
                time_result="-"
            fi
            ;;
        "g2_aimd")
            # Special handling for g2-aimd
            # Hardcode X for dogster as requested
            if [ "$dataset" = "dogster" ]; then
                time_result="X"
            # For other datasets including wikipedia_link_en, use the 4-equals format matching
            elif grep -q "==== Running G2-AIMD on $dataset ====" "$log_file"; then
                time_result=$(analyze_g2_aimd_log "$log_file" "$dataset")
                # If analyze_g2_aimd_log returns empty or error, it means the dataset exists but didn't complete successfully
                if [ -z "$time_result" ] || [ "$time_result" = "X" ]; then
                    time_result="X"
                fi
            else
                # Dataset not found in log
                time_result="-"
            fi
            ;;
        *)
            # Try to determine automatically
            if grep -q "MCE GPU Time:" "$log_file"; then
                time_result=$(analyze_rdmce_log "$log_file" "$dataset")
            elif grep -q "count time" "$log_file"; then
                time_result=$(analyze_mce_gpu_log "$log_file" "$dataset")
            else
                time_result=$(analyze_g2_aimd_log "$log_file" "$dataset")
            fi
            if [ -z "$time_result" ]; then
                time_result="-"
            fi
            ;;
    esac
    
    echo "$time_result"
}

# Function to process rdmce_multigpu.log separately
process_multigpu_log() {
    local multigpu_log="$LOG_DIR/rdmce_multigpu.log"
    
    if [ ! -f "$multigpu_log" ]; then
        echo "Warning: rdmce_multigpu.log not found, cannot extract multi-GPU results"
        > "$MULTI_GPU_FILE"
        return 1
    fi
    
    echo "Processing multi-GPU log: $multigpu_log"
    
    # Create or clear the multi-GPU result file
    > "$MULTI_GPU_FILE"
    
    # Extract 1 GPU result - use more flexible search pattern
    local gpu1_time=$(grep -A 30 "RDMCE-1GPUs" "$multigpu_log" | grep "MCE GPU Time:" | awk '{print $4}')
    if [ -n "$gpu1_time" ]; then
        echo "1,$gpu1_time,,,,," >> "$MULTI_GPU_FILE"
        echo "Extracted 1 GPU time: $gpu1_time"
    else
        echo "1,,,,," >> "$MULTI_GPU_FILE"
        echo "Warning: Could not find 1 GPU time"
    fi
    
    # Extract 2 GPUs results - isolate the specific section
    # Find line numbers for 2 GPUs section
    local gpu2_start=$(grep -n "RDMCE-2GPUs" "$multigpu_log" | head -1 | cut -d':' -f1)
    local gpu2_end=$(grep -n "RDMCE-4GPUs" "$multigpu_log" | head -1 | cut -d':' -f1)
    if [ -n "$gpu2_start" ] && [ -n "$gpu2_end" ]; then
        # Extract only the 2 GPUs section
        local gpu2_section=$(sed -n "${gpu2_start},${gpu2_end}p" "$multigpu_log")
        # Extract lines like "0: 124.404", sort by device ID, then extract times
        local gpu2_times=$(echo "$gpu2_section" | grep -E '^[[:space:]]*[0-9]+:[[:space:]]*[0-9.]+' | sort -t':' -k1 -n | awk -F': ' '{print $2}' | tr '\n' ',')
        if [ -n "$gpu2_times" ]; then
            # Remove trailing comma and pad with commas to reach 8 fields
            gpu2_times="${gpu2_times%,}",,,,,
            # Take only the first 8 fields
            gpu2_times=$(echo "$gpu2_times" | cut -d',' -f1-8)
            echo "2,$gpu2_times" >> "$MULTI_GPU_FILE"
            echo "Extracted 2 GPUs times: $gpu2_times"
        else
            echo "2,,,,," >> "$MULTI_GPU_FILE"
            echo "Warning: Could not find 2 GPUs time values"
        fi
    else
        echo "2,,,,," >> "$MULTI_GPU_FILE"
        echo "Warning: Could not find 2 GPUs section boundaries"
    fi
    
    # Extract 4 GPUs results - isolate the specific section
    local gpu4_start=$(grep -n "RDMCE-4GPUs" "$multigpu_log" | head -1 | cut -d':' -f1)
    local gpu4_end=$(grep -n "RDMCE-8GPUs" "$multigpu_log" | head -1 | cut -d':' -f1)
    if [ -n "$gpu4_start" ] && [ -n "$gpu4_end" ]; then
        # Extract only the 4 GPUs section
        local gpu4_section=$(sed -n "${gpu4_start},${gpu4_end}p" "$multigpu_log")
        # Extract lines like "0: 72.2334", sort by device ID, then extract times
        local gpu4_times=$(echo "$gpu4_section" | grep -E '^[[:space:]]*[0-9]+:[[:space:]]*[0-9.]+' | sort -t':' -k1 -n | awk -F': ' '{print $2}' | tr '\n' ',')
        if [ -n "$gpu4_times" ]; then
            # Remove trailing comma and pad with commas
            gpu4_times="${gpu4_times%,}",,,,,
            # Take only the first 8 fields
            gpu4_times=$(echo "$gpu4_times" | cut -d',' -f1-8)
            echo "4,$gpu4_times" >> "$MULTI_GPU_FILE"
            echo "Extracted 4 GPUs times: $gpu4_times"
        else
            echo "4,,,,," >> "$MULTI_GPU_FILE"
            echo "Warning: Could not find 4 GPUs time values"
        fi
    else
        echo "4,,,,," >> "$MULTI_GPU_FILE"
        echo "Warning: Could not find 4 GPUs section boundaries"
    fi
    
    # Extract 8 GPUs results - isolate the specific section
    local gpu8_start=$(grep -n "RDMCE-8GPUs" "$multigpu_log" | head -1 | cut -d':' -f1)
    if [ -n "$gpu8_start" ]; then
        # Extract only the 8 GPUs section
        local gpu8_section=$(sed -n "${gpu8_start},$ p" "$multigpu_log")
        # Extract lines like "0: 47.3392", sort by device ID, then extract times
        local gpu8_times=$(echo "$gpu8_section" | grep -E '^[[:space:]]*[0-9]+:[[:space:]]*[0-9.]+' | sort -t':' -k1 -n | awk -F': ' '{print $2}' | tr '\n' ',')
        if [ -n "$gpu8_times" ]; then
            # Remove trailing comma and pad with commas if needed
            gpu8_times="${gpu8_times%,}"
            # Take only the first 8 fields
            gpu8_times=$(echo "$gpu8_times" | cut -d',' -f1-8)
            echo "8,$gpu8_times" >> "$MULTI_GPU_FILE"
            echo "Extracted 8 GPUs times: $gpu8_times"
        else
            echo "8,,,,," >> "$MULTI_GPU_FILE"
            echo "Warning: Could not find 8 GPUs time values"
        fi
    else
        echo "8,,,,," >> "$MULTI_GPU_FILE"
        echo "Warning: Could not find 8 GPUs section start"
    fi
    
    echo "Multi-GPU results saved to $MULTI_GPU_FILE"
}

# Function to format log name as requested
format_log_name() {
    local log_name="$1"
    
    # Remove .log extension
    log_name=${log_name%.log}
    
    # Replace specific names as requested
    case "$log_name" in
        "g2_aimd_all")
            echo "G2-AIMD"
            ;;
        "mce_gpu_px_all")
            echo "mce-gpu-PX"
            ;;
        "mce_gpu_p_all")
            echo "mce-gpu-P"
            ;;
        "rdmce_all")
            echo "RDMCE"
            ;;
        *)
            echo "$log_name"
            ;;
    esac
}

# Process multi-GPU log separately first
process_multigpu_log

# Get all log files except rdmce_multigpu.log
LOG_FILES=($(ls "$LOG_DIR"/*.log 2>/dev/null | grep -v "rdmce_multigpu.log"))
if [ ${#LOG_FILES[@]} -eq 0 ]; then
    echo "Error: No log files found in $LOG_DIR except rdmce_multigpu.log"
    exit 1
fi

# Create lists for log files and their types
log_names=()
formatted_log_names=()
log_types=()
for log_file in "${LOG_FILES[@]}"; do
    log_name=$(basename "$log_file")
    log_names+=($log_name)
    
    # Format log name as requested
    formatted_name=$(format_log_name "$log_name")
    formatted_log_names+=($formatted_name)
    
    # Determine log type
    log_type="unknown"
    if [[ "$log_name" == RDMCE*.log || "$log_name" == rdmce*.log ]]; then
        log_type="rdmce"
    elif [[ "$log_name" == mce_gpu*.log ]]; then
        log_type="mce_gpu"
    elif [[ "$log_name" == g2_aimd*.log ]]; then
        log_type="g2_aimd"
    fi
    log_types+=($log_type)
    
    echo "Found log file: $log_name -> $formatted_name (Type: $log_type)"
done

# Create CSV header
echo -n "Dataset,abbr" > "$CSV_FILE"
for formatted_name in "${formatted_log_names[@]}"; do
    echo -n ",$formatted_name" >> "$CSV_FILE"
done
echo >> "$CSV_FILE"

# Process each dataset in the specified order
for dataset in "${datasets[@]}"; do
    # Get abbreviation
    abbr=${dataset_abbr[$dataset]}
    
    # Create CSV row string
    csv_row="$dataset,$abbr"
    
    # Get results for each log file
    for i in "${!LOG_FILES[@]}"; do
        log_file=${LOG_FILES[$i]}
        log_type=${log_types[$i]}
        
        # Get result for this dataset and log file
        result=$(get_dataset_result "$dataset" "$log_file" "$log_type")
        
        # Default to '-' if result is empty
        if [ -z "$result" ]; then
            result="-"
        fi
        
        # Clean and validate the result
        if [ "$result" != "-" ] && [ "$result" != "X" ]; then
            # Remove any non-numeric characters (except decimal point)
            result=$(echo "$result" | sed 's/[^0-9.]//g')
            
            # Ensure it's a valid number and format to 6 decimal places
            if [[ "$result" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                result=$(printf "%.6f" "$result")
            else
                result="-"
            fi
        fi
        
        # Add to row
        csv_row="$csv_row,$result"
    done
    
    # Write to CSV
    echo "$csv_row" >> "$CSV_FILE"
    
    echo "Processed dataset: $dataset ($abbr)"
done

# Display the CSV result summary
echo "\nGenerated CSV results summary:"
echo "=========================================="
echo "Dataset count: ${#datasets[@]}"
echo "Log file count: ${#LOG_FILES[@]}"
echo "=========================================="
echo "First few lines of CSV output:"
head -5 "$CSV_FILE"

echo "\nMulti-GPU results (res_multiGPU.csv):"
cat "$MULTI_GPU_FILE"

echo "\n=========================================="
echo "Analysis completed!"
echo "CSV results saved to: $CSV_FILE"
echo "Multi-GPU results saved to: $MULTI_GPU_FILE"
echo "=========================================="