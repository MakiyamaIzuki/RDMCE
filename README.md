# Root-Down Exposure for Maximal Clique Enumeration on GPUs

This repository contains the source code for the paper "Root-Down Exposure for Maximal Clique Enumeration on GPUs" accepted to PPoPP 2026.

## Project Structure

The project is organized as follows:

- **RDMCE/**: Contains the source code for our proposed Root-Down Maximal Clique Enumeration algorithm
  - **include/**: Header files defining data structures and interfaces
  - **src/**: Implementation files including CUDA kernels
  - **src/kernels/**: GPU kernel implementations, including BkPivotBitBalance and Multi-GPU variants

- **baselines/**: Contains implementations of comparison methods
  - **g2-aimd/**: G²-AIMD algorithm implementation
  - **mce-gpu/**: MCE-GPU algorithm implementation

- **scripts/**: Utility scripts for compilation and evaluation

- **datasets/**: Directory for storing test datasets

- **logs/**: Contains log files with execution time results

- **exp/**: Contains data and code for generating the figures presented in the paper

## Data Preparation

### Dataset Download

Before running the tests, you need to download the datasets:

```bash
bash scripts/download_dataset.sh
```

**Important Notes:** 
- The dataset download process can take up to 3 hours as some datasets are very large. For example, `com-friendster.txt` is approximately 31GB.
- For testing purposes, it is recommended to download only the Facebook dataset first:
  ```bash
  # Edit download_dataset.sh to uncomment only the Facebook dataset
  bash scripts/download_dataset.sh
  ```

### Data Preprocessing

After downloading the datasets, you need to preprocess them to convert text-format graphs to binary format for each MCE solution:

```bash
bash scripts/preprocessing.sh
```

**Important Notes:** 
- The preprocessing step can take up to 6 hours to complete, depending on the number and size of datasets.
- This script will automatically compile required executables if they are not already available and process all downloaded datasets.

### Memory Requirements

**Important:** For Artificial evaluation, you will need at least 300GB of memory to store both the original and preprocessed graphs.

## Testing Workflow

To evaluate the performance of RDMCE and baselines, follow these steps:

1. **Compile the code**:
   ```bash
   bash scripts/compile.sh all
   ```
   This will compile RDMCE and all baseline implementations.

2. **Run evaluation script**:
   ```bash
   bash scripts/overall_eval.sh
   ```
   This script will automatically traverse valid datasets for all methods and execute them, recording results to log files.

3. **Run variants evaluation** (optional):
   ```bash
   bash scripts/breakdown_eval.sh
   ```
   This script evaluates different RDMCE variants and GPU scaling performance.

## Experimental Data and Figure Generation

The `exp/` directory contains:

- Raw performance data used in the paper
- Python scripts for generating the figures presented in the paper

To generate your own figures using new performance data:

1. Run the evaluation scripts to generate new log files in the `logs/` directory
2. Extract runtime information from the logs
3. Update the data files in the `exp/` directory
4. Run the plotting scripts to generate new figures

For example, to generate the overall performance comparison figure:
```bash
cd exp
python exp_overall.py
```

This will create a PDF file with the performance comparison chart.