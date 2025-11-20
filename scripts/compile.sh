#!/bin/bash

# Base directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="${SCRIPT_DIR}/.."
BIN_DIR="${ROOT_DIR}/bin"

# Create bin directory if it doesn't exist
mkdir -p "${BIN_DIR}"

# Function to compile RDMCE variant with custom CMake definitions
compile_rdmce_variant() {
    local variant_name="$1"
    local cmake_defs="$2"
    local target_dir="${ROOT_DIR}/RDMCE"
    
    # Check if directory exists
    if [ ! -d "${target_dir}" ]; then
        echo "Error: RDMCE directory not found."
        return 1
    fi
    
    local build_dir="${target_dir}/build_${variant_name}"
    
    echo "Compiling ${variant_name}..."
    
    # Create and enter build directory
    mkdir -p "${build_dir}"
    cd "${build_dir}" || return 1
    
    # Build with custom definitions
    cmake ${cmake_defs} .. && make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Error: ${variant_name} compilation failed."
        return 1
    fi
    
    # Copy executable to bin with variant name
    if [ -f "RDMCE" ]; then
        chmod 755 "RDMCE"
        cp "RDMCE" "${BIN_DIR}/${variant_name}"
        echo "${variant_name} compiled successfully."
    else
        echo "Error: ${variant_name} executable not found."
        return 1
    fi
    
    # Clean up
    cd "${SCRIPT_DIR}"
    rm -rf "${build_dir}"
}

# Function to compile all RDMCE variants
compile_rdmce_variants() {
    echo "Compiling all RDMCE variants..."
    
    # NP type variant 
    compile_rdmce_variant "RDMCE-NP" "-UEXPAND_R"
    
    # W type variants
    compile_rdmce_variant "RDMCE-W4" "-DWARP_PER_BLOCK=1 -DBLOCK_PER_SM=4"
    compile_rdmce_variant "RDMCE-W8" "-DWARP_PER_BLOCK=2 -DBLOCK_PER_SM=4"
    compile_rdmce_variant "RDMCE-W12" "-DWARP_PER_BLOCK=3 -DBLOCK_PER_SM=4"
    compile_rdmce_variant "RDMCE-W16" "-DWARP_PER_BLOCK=4 -DBLOCK_PER_SM=4"
    compile_rdmce_variant "RDMCE-W20" "-DWARP_PER_BLOCK=5 -DBLOCK_PER_SM=4"
    compile_rdmce_variant "RDMCE-W24" "-DWARP_PER_BLOCK=6 -DBLOCK_PER_SM=4"
    compile_rdmce_variant "RDMCE-W28" "-DWARP_PER_BLOCK=7 -DBLOCK_PER_SM=4"
    compile_rdmce_variant "RDMCE-W32" "-DWARP_PER_BLOCK=8 -DBLOCK_PER_SM=4"
    
    # T type variants
    compile_rdmce_variant "RDMCE-NB" "-DTASK_SHARE_BOUND=11"
    compile_rdmce_variant "RDMCE-T16" "-DTASK_SHARE_BOUND=16"
    compile_rdmce_variant "RDMCE-T24" "-DTASK_SHARE_BOUND=24"
    compile_rdmce_variant "RDMCE-T32" "-DTASK_SHARE_BOUND=32"
    compile_rdmce_variant "RDMCE-T48" "-DTASK_SHARE_BOUND=48"
    compile_rdmce_variant "RDMCE-T64" "-DTASK_SHARE_BOUND=64"
    compile_rdmce_variant "RDMCE-T96" "-DTASK_SHARE_BOUND=96"
    
    echo "All RDMCE variants compilation completed."
}

# Function to compile standard RDMCE
compile_rdmce() {
    local variant="RDMCE"
    local target_dir="${ROOT_DIR}/${variant}"
    
    # Check if directory exists
    if [ ! -d "${target_dir}" ]; then
        echo "Error: ${variant} directory not found."
        return 1
    fi
    
    local build_dir="${target_dir}/build"
    
    echo "Compiling ${variant}..."
    
    # Create and enter build directory
    mkdir -p "${build_dir}"
    cd "${build_dir}" || return 1
    
    # Build
    cmake .. && make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Error: ${variant} compilation failed."
        return 1
    fi
    
    # Copy executable to bin
    if [ -f "${variant}" ]; then
        chmod 755 "${variant}"
        mv "${variant}" "${BIN_DIR}/"
        echo "${variant} compiled successfully."
    else
        echo "Error: ${variant} executable not found."
        return 1
    fi
    
    # Clean up
    cd "${SCRIPT_DIR}"
    rm -rf "${build_dir}"
}

# Function to compile mce-gpu
compile_mce_gpu() {
    local target="mce-gpu"
    local target_dir="${ROOT_DIR}/baselines/${target}"
    
    # Check if directory exists
    if [ ! -d "${target_dir}" ]; then
        echo "Error: ${target} directory not found."
        return 1
    fi
    
    local build_dir="${target_dir}/build"

    echo "Compiling ${target}..."
    
    mkdir -p "${build_dir}"
    cd "${build_dir}" || return 1

    # Build
    cmake .. && make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Error: ${target} compilation failed."
        return 1
    fi
    
    # Copy executable to bin
    if [ -f "parallel_mce_on_gpus" ]; then
        mv parallel_mce_on_gpus mce_gpu
        cp "mce_gpu" "${BIN_DIR}/"
        echo "${target} compiled successfully."
    else
        echo "Error: ${target} executable not found."
        return 1
    fi

    cd "${SCRIPT_DIR}"
    rm -rf "${build_dir}"
}

# Function to compile g2-aimd
compile_g2_aimd() {
    local target="g2-aimd"
    local target_dir="${ROOT_DIR}/baselines/${target}"
    local build_dir="${target_dir}/build"
    local build_bin_dir="${build_dir}/bin"
    
    # Check if directory exists
    if [ ! -d "${target_dir}" ]; then
        echo "Error: ${target} directory not found."
        return 1
    fi
    
    echo "Compiling ${target}..."
    
    # Create and enter build directory
    mkdir -p "${build_dir}"
    cd "${build_dir}" || return 1
    
    # Build using CMake
    cmake .. && make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Error: ${target} compilation failed."
        return 1
    fi
    
    # Copy required executables to bin directory
    local required_bins=('preprocess' 'kcore' 'bk')
    local success=0
    
    for bin in "${required_bins[@]}"; do
        if [ -f "${build_bin_dir}/${bin}" ]; then
            cp "${build_bin_dir}/${bin}" "${BIN_DIR}/"
            echo "Copied ${bin} to ${BIN_DIR}"
            success=1
        else
            echo "Warning: ${bin} executable not found in build directory"
        fi
    done
    
    if [ $success -eq 1 ]; then
        echo "${target} compiled successfully."
    else
        echo "Error: None of the required executables were found."
        return 1
    fi
    
    # Clean up and return to script directory
    cd "${SCRIPT_DIR}"
    rm -rf "${build_dir}"
}

# Main function to handle arguments
main() {
    if [ $# -eq 0 ]; then
        echo "Usage: $0 [rdmce|rdmce-variants|mce-gpu|g2-aimd|all]"
        echo "Compiles the specified component(s)."
        return 1
    fi
    
    local success=0
    
    while [ $# -gt 0 ]; do
        case "$1" in
            rdmce)
                compile_rdmce
                [ $? -ne 0 ] && success=1
                ;;
            rdmce-variants)
                compile_rdmce_variants
                [ $? -ne 0 ] && success=1
                ;;
            mce-gpu)
                compile_mce_gpu
                [ $? -ne 0 ] && success=1
                ;;
            g2-aimd)
                compile_g2_aimd
                [ $? -ne 0 ] && success=1
                ;;
            all)
                compile_rdmce
                compile_rdmce_variants
                compile_mce_gpu
                compile_g2_aimd
                [ $? -ne 0 ] && success=1
                ;;
            *)
                echo "Error: Unknown component '$1'"
                echo "Usage: $0 [rdmce|rdmce-variants|mce-gpu|g2-aimd|all]"
                return 1
                ;;
        esac
        shift
    done
    
    return $success
}

# Execute main function
main "$@"
exit $?