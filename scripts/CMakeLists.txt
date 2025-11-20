cmake_minimum_required(VERSION 3.10)
project(mce-gpu LANGUAGES CXX CUDA)

# Reduce CMake output verbosity
set(CMAKE_VERBOSE_MAKEFILE OFF)
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
set_property(GLOBAL PROPERTY ALLOW_DUPLICATE_CUSTOM_TARGETS ON)

# Set C++ standard to match Makefile
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add include directories
include_directories(include)

# Automatically detect and use native GPU architecture
set(CMAKE_CUDA_ARCHITECTURES native)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Source files
file(GLOB SOURCES "src/*.cu")

# Define executable
add_executable(parallel_mce_on_gpus ${SOURCES})

# Set compilation options
set_target_properties(parallel_mce_on_gpus PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF  # Match Makefile behavior
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF # Match Makefile behavior
)

# Add compiler flags to match Makefile
 target_compile_options(parallel_mce_on_gpus PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-w> # Suppress warnings
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-w> # Suppress warnings
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

# Link OpenMP library
 target_link_libraries(parallel_mce_on_gpus PRIVATE OpenMP::OpenMP_CXX)

# Add custom command to copy executable to source directory after build
add_custom_command(TARGET parallel_mce_on_gpus POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:parallel_mce_on_gpus> ${CMAKE_CURRENT_SOURCE_DIR}/parallel_mce_on_gpus
    COMMENT "Copying executable to source directory"
)