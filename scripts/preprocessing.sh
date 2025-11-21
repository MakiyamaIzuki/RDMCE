#!/bin/bash

# Preprocessing script for all graph datasets
# This script processes all .txt files in the datasets directory and generates:
# 1. RDMCE binary files in ./datasets/RDMCE/
# 2. MCE-GPU .bel files in ./datasets/MCE-GPU/
# 3. G2-AIMD preprocessed and k-core filtered files in ./datasets/G2-AIMD/

# Base directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="${SCRIPT_DIR}/.."
DATASET_DIR="${ROOT_DIR}/datasets"
# Create separate output directories for each algorithm
RDMCE_OUTPUT_DIR="${DATASET_DIR}/RDMCE"
MCEGPU_OUTPUT_DIR="${DATASET_DIR}/MCE-GPU"
G2AIMD_OUTPUT_DIR="${DATASET_DIR}/G2-AIMD"
BIN_DIR="${ROOT_DIR}/bin"
COMPILE_SCRIPT="${SCRIPT_DIR}/compile.sh"

# Create necessary output directories
mkdir -p "${RDMCE_OUTPUT_DIR}"
mkdir -p "${MCEGPU_OUTPUT_DIR}"
mkdir -p "${G2AIMD_OUTPUT_DIR}"
mkdir -p "${BIN_DIR}"  # Ensure bin directory exists

echo "Starting preprocessing for all datasets..."
echo "Checking for required executables..."

# Simple check and compile logic using bash to run the script
if [ ! -f "${BIN_DIR}/RDMCE" ]; then
    echo "RDMCE executable not found, compiling..."
    bash "${COMPILE_SCRIPT}" rdmce
    if [ ! -f "${BIN_DIR}/RDMCE" ]; then
        echo "✗ Failed to compile RDMCE"
        exit 1
    fi
fi

if [ ! -f "${BIN_DIR}/mce_gpu" ]; then
    echo "mce_gpu executable not found, compiling..."
    bash "${COMPILE_SCRIPT}" mce-gpu
    if [ ! -f "${BIN_DIR}/mce_gpu" ]; then
        echo "✗ Failed to compile mce_gpu"
        exit 1
    fi
fi

if [ ! -f "${BIN_DIR}/preprocess" ] || [ ! -f "${BIN_DIR}/kcore" ]; then
    echo "preprocess or kcore executable not found, compiling g2-aimd..."
    bash "${COMPILE_SCRIPT}" g2-aimd
    if [ ! -f "${BIN_DIR}/preprocess" ] || [ ! -f "${BIN_DIR}/kcore" ]; then
        echo "✗ Failed to compile preprocess or kcore"
        exit 1
    fi
fi

# Check if all required executables are available
if [ ! -f "${BIN_DIR}/RDMCE" ] || [ ! -f "${BIN_DIR}/mce_gpu" ] || [ ! -f "${BIN_DIR}/preprocess" ] || [ ! -f "${BIN_DIR}/kcore" ]; then
    echo "✗ Some required executables are missing or compilation failed."
    echo "Please check the compilation errors and try again."
    exit 1
fi

# Process each .txt file in the datasets directory
for txt_file in "${DATASET_DIR}"/*.txt; do
    # Skip if no .txt files found
    if [ ! -f "${txt_file}" ]; then
        echo "No .txt files found in ${DATASET_DIR}"
        continue
    fi
    
    # Extract filename without extension
    filename=$(basename "${txt_file}" .txt)
    echo "Processing ${filename}..."
    
    # Define output file paths
    rdmce_output="${RDMCE_OUTPUT_DIR}/${filename}.bin"
    mcegpu_output="${MCEGPU_OUTPUT_DIR}/${filename}.bel"
    g2aimd_bin="${G2AIMD_OUTPUT_DIR}/${filename}.bin"
    g2aimd_vbmap="${G2AIMD_OUTPUT_DIR}/${filename}.bin.2.hop.vbmap"
    
    # Check if all required output files already exist
    if [ -f "${rdmce_output}" ] && [ -f "${mcegpu_output}" ] && [ -f "${g2aimd_bin}" ] && [ -f "${g2aimd_vbmap}" ]; then
        echo "  - All preprocessing files for ${filename} already exist. Skipping..."
        continue
    fi
    
    # 1. Process for RDMCE
    if [ ! -f "${rdmce_output}" ]; then
        echo "  - Generating RDMCE binary file..."
        "${BIN_DIR}/RDMCE" -i "${txt_file} -c"
        if [ $? -eq 0 ]; then
            # Move the generated binary file to RDMCE output directory
            if [ -f "${DATASET_DIR}/${filename}.bin" ]; then
                mv "${DATASET_DIR}/${filename}.bin" "${RDMCE_OUTPUT_DIR}/"
                echo "    ✓ RDMCE file saved to ${rdmce_output}"
            else
                echo "    ✗ RDMCE binary file not found after processing"
            fi
        else
            echo "    ✗ Error processing ${filename} with RDMCE"
        fi
    else
        echo "  - RDMCE file already exists, skipping..."
    fi
    
    # 2. Process for MCE-GPU
    if [ ! -f "${mcegpu_output}" ]; then
        echo "  - Generating MCE-GPU .bel file..."
        "${BIN_DIR}/mce_gpu" -m convert -g "${txt_file}" -r "${mcegpu_output}"
        if [ $? -eq 0 ]; then
            echo "    ✓ MCE-GPU file saved to ${mcegpu_output}"
        else
            echo "    ✗ Error processing ${filename} with MCE-GPU"
        fi
    else
        echo "  - MCE-GPU file already exists, skipping..."
    fi
    
    # 3. Process for G2-AIMD
    if [ ! -f "${g2aimd_bin}" ] || [ ! -f "${g2aimd_vbmap}" ]; then
        echo "  - Generating G2-AIMD files..."
        
        # First run preprocess if bin file doesn't exist
        if [ ! -f "${g2aimd_bin}" ]; then
            "${BIN_DIR}/preprocess" -f "${txt_file}"
            if [ $? -eq 0 ]; then
                # Move the generated binary file to G2-AIMD output directory
                if [ -f "${DATASET_DIR}/${filename}.bin" ]; then
                    mv "${DATASET_DIR}/${filename}.bin" "${g2aimd_bin}"
                    echo "    ✓ Preprocessing complete, file saved to ${g2aimd_bin}"
                else
                    echo "    ✗ Preprocessed binary file not found"
                    continue
                fi
            else
                echo "    ✗ Error preprocessing ${filename} with G2-AIMD"
                continue
            fi
        else
            echo "    - Preprocess file already exists, skipping preprocess step"
        fi
        
        # Then run kcore on the preprocessed file if vbmap file doesn't exist
        if [ ! -f "${g2aimd_vbmap}" ]; then
            "${BIN_DIR}/kcore" -f "${g2aimd_bin}"
            if [ $? -eq 0 ]; then
                # Check if .vbmap file was generated and move it to the correct directory
                if [ -f "${DATASET_DIR}/${filename}.vbmap" ]; then
                    mv "${DATASET_DIR}/${filename}.vbmap" "${G2AIMD_OUTPUT_DIR}/"
                    echo "    ✓ Moved vbmap file to ${g2aimd_vbmap}"
                fi
                echo "    ✓ K-core decomposition complete"
            else
                echo "    ✗ Error running k-core decomposition"
            fi
        else
            echo "    - K-core file already exists, skipping kcore step"
        fi
    else
        echo "  - G2-AIMD files already exist, skipping..."
    fi
done

echo ""
echo "Preprocessing completed for all datasets."
echo "Output files organized in respective directories:"
echo "- RDMCE: ${RDMCE_OUTPUT_DIR}"
echo "- MCE-GPU: ${MCEGPU_OUTPUT_DIR}"
echo "- G2-AIMD: ${G2AIMD_OUTPUT_DIR}"