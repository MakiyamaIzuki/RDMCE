#!/bin/bash

# Base directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="${SCRIPT_DIR}/.."
BIN_DIR="${ROOT_DIR}/bin"

# Create bin directory if it doesn't exist
mkdir -p "${BIN_DIR}"

# Function to compile RDMCE and its variants
compile_rdmce() {
    local variant="RDMCE"
    local target_dir="${ROOT_DIR}/${variant}"
    
    # Check if directory exists (supports variants matching RDMCE*)
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
        cp "${variant}" "${BIN_DIR}/"
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
    
    # Check if directory exists
    if [ ! -d "${target_dir}" ]; then
        echo "Error: ${target} directory not found."
        return 1
    fi
    
    echo "Compiling ${target}..."
    
    # Compile BK application
    cd "${target_dir}/app_BK" || return 1
    make
    if [ $? -ne 0 ]; then
        echo "Error: ${target} BK application compilation failed."
        return 1
    fi
    
    # Copy executable to bin
    if [ -f "main" ]; then
        cp "main" "${BIN_DIR}/g2-aimd-bk"
    else
        echo "Error: ${target} BK executable not found."
        return 1
    fi
    
    # Compile gmatch application
    cd "${target_dir}/app_gmatch" || return 1
    make
    if [ $? -ne 0 ]; then
        echo "Error: ${target} gmatch application compilation failed."
        return 1
    fi
    
    # Copy executable to bin
    if [ -f "main" ]; then
        cp "main" "${BIN_DIR}/g2-aimd-gmatch"
        echo "${target} compiled successfully."
    else
        echo "Error: ${target} gmatch executable not found."
        return 1
    fi
}

# Main function to handle arguments
main() {
    if [ $# -eq 0 ]; then
        echo "Usage: $0 [rdmce|mce-gpu|g2-aimd|all]"
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
                compile_mce_gpu
                compile_g2_aimd
                [ $? -ne 0 ] && success=1
                ;;
            *)
                # Check for RDMCE variants
                if [[ "$1" == RDMCE* ]]; then
                    # For future extension with specific variants
                    echo "Compiling $1 variant..."
                    # Implementation for specific variants can be added here
                else
                    echo "Error: Unknown component '$1'"
                    echo "Usage: $0 [rdmce|mce-gpu|g2-aimd|all]"
                    return 1
                fi
                ;;
        esac
        shift
    done
    
    return $success
}

# Execute main function
main "$@"
exit $?