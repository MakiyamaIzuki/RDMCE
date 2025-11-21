# Root-Down Exposure for Maximal Clique Enumeration on GPUs

This repository contains the source code for the paper "Root-Down Exposure for Maximal Clique Enumeration on GPUs" accepted to PPoPP 2026.

## Project Structure

The project is organized as follows:

- **RDMCE/**: Contains the source code for our proposed Root-Down Maximal Clique Enumeration algorithm
  - **include/**: Header files defining data structures and interfaces
  - **src/**: Implementation files including CUDA kernels
  - **src/kernels/**: GPU kernel implementations, including BkPivotBitBalance and Multi-GPU variants

- **baselines/**: Contains implementations of comparison methods
  - **g2-aimd/**: G²-AIMD algorithm implementation (Git submodule)
  - **mce-gpu/**: MCE-GPU algorithm implementation (Git submodule)

- **scripts/**: Utility scripts for compilation and evaluation

- **datasets/**: Directory for storing test datasets

- **logs/**: Contains log files with execution time results

- **exp/**: Contains data and code for generating the figures presented in the paper

### Git Submodule Setup

This project uses Git submodules for the baseline implementations. After cloning the repository, initialize and update the submodules:

```bash
# Initialize and update all submodules
git submodule update --init --recursive
```

This command will fetch the source code for the g2-aimd and mce-gpu baselines into the baselines directory.

## RDMCE Quick Start

### Requirements

- **GPU Architecture**: NVIDIA GPU with architecture 7.0 or higher (e.g., RTX 4090, A100)
- **CUDA**: 12.0 or higher
- **GCC**: 9.0 or higher
- **CMake**: 3.25 or higher
- **Operating System**: Linux
- **Memory**: At least 64GB of system memory (required for large datasets like com-Friendster)

**Multi-GPU Usage Recommendation**: If multiple GPUs are available, it is recommended to allocate an independent GPU for each task to run in parallel. This will significantly save overall testing time, especially when there are many test entries including large datasets like com-Friendster (31GB), where each application needs to load data independently.

### Basic Testing

1. **Compile the code**:
   ```bash
   bash scripts/compile.sh rdmce
   ```
   This will compile the RDMCE implementation.
   
   **Estimated Time**: Compilation typically takes about 10 minutes.

2. **Run a simple example**:
   ```bash
   ./bin/RDMCE -i RDMCE/data/zachary.txt
   ```
   RDMCE supports standard COO (Coordinate List) plain text input format. Refer to `zachary.txt` for the expected format.

## Artifact Evaluation

### Requirements

- **Storage**: At least 300GB of disk space for datasets and preprocessed binary files
- **Multi-GPU Testing**: Machine with 8 NVIDIA RTX 4090 GPUs
- **Cross-Platform Testing**: Machines with A100 or H800 GPUs (used in the paper)
  - Note: Other NVIDIA GPUs with architecture 7.0+ should also work

### Dataset Download

```bash
# Download datasets (approximate time: 3 hours)
bash scripts/download_dataset.sh
```

The script downloads all required datasets for evaluation. You can modify the script to download specific datasets based on your needs by editing the file and commenting out unnecessary dataset downloads.

### Data Preprocessing

```bash
# Preprocess datasets (approximate time: 6 hours)
bash scripts/preprocessing.sh
```

This script converts text-format graphs to binary format for each MCE solution. Different algorithms have different binary formats, so the preprocessing step may take some time. For quicker testing, consider starting with smaller datasets like facebook or dogster.

### Experiments

1. **Compile the code**:
   ```bash
   # Compile all implementations (RDMCE and baselines)
   bash scripts/compile.sh all
   
   # Or compile specific implementations
   bash scripts/compile.sh rdmce
   bash scripts/compile.sh mce-gpu
   bash scripts/compile.sh g2-aimd
   ```

2. **Run overall evaluation**:
   ```bash
   bash scripts/overall_eval.sh  # Approximately 4 hours
   ```
   This script runs RDMCE and baselines (mce-gpu-P, mce-gpu-PX, G2-AIMD) on all test datasets and records execution times.

3. **Run breakdown evaluation**:
   ```bash
   bash scripts/breakdown_eval.sh  # Approximately 6 hours
   ```
   This script evaluates different RDMCE variants and GPU scaling performance.

4. **Generate experiment results**:
   After running the evaluation scripts, you can analyze the logs and generate structured results:

   ```bash
   # Analyze logs and generate structured results
   bash scripts/analyze.sh
   ```

   This script will automatically process all log files and aggregate results to:
   - `logs/analysis_results.csv`: Main performance comparison data
   - `logs/res_multiGPU.csv`: Multi-GPU scaling performance data

   To generate figures similar to those in the paper:

   ```bash
   cd exp
   python exp_overall.py    # Generate overall performance comparison
   python exp_breakdown.py  # Generate breakdown analysis
   python exp_multigpu.py   # Generate multi-GPU scaling charts
   ```

   These scripts will create PDF files with the performance charts.

   **Additional Notes for Memory Usage (Figure 9) and Cross-Platform Testing (Figure 13)**: These figures require manual data collection and configuration:
   
   ```bash
   # For memory usage monitoring
   python scripts/mem_prof.py
   ```
   
   After running the memory profiling script, manually record the memory usage data and update the corresponding configuration files in the `exp/` directory before generating the memory comparison figure (Figure 9). For cross-platform testing results (Figure 13), data collection must be performed manually on each platform, as memory and performance measurements are platform-dependent. Each platform should have its own configuration file in the `exp/` directory to ensure accurate figure generation.

If you have any questions, please contact pz@mail.tsinghua.edu.cn.