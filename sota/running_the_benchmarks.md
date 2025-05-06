# Running the Benchmarks

This document provides a comprehensive guide for running the benchmarks in the `sota` directory to compare our LSM Tree implementation with industry-standard key-value stores.

## Prerequisites

Before running the benchmarks, ensure you have the following dependencies installed:

```bash
# Basic build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config git

# Compression libraries
sudo apt-get install -y libsnappy-dev zlib1g-dev libbz2-dev libzstd-dev liblz4-dev

# Additional dependencies
sudo apt-get install -y libgflags-dev liburing-dev libaio-dev

# Python dependencies for analysis
sudo apt-get install -y python3-pip python3-venv
```

## Benchmark Overview

The `sota` directory contains benchmarks for comparing our LSM Tree implementation against:

1. **RocksDB**: Facebook's LSM tree implementation
2. **SpeedB**: An enhanced fork of RocksDB with improved performance
3. **LevelDB**: Google's original LSM tree implementation
4. **LMDB**: A B+ tree-based key-value store
5. **WiredTiger**: MongoDB's storage engine with a hybrid approach
6. **TerarkDB**: ByteDance's storage engine with unique point lookup optimizations

## Running All Benchmarks

The simplest way to run all benchmarks is using the provided script:

```bash
# Make sure you're in the project root directory
cd /path/to/cs265-lsm-tree

# Run all benchmarks
./sota/run_all_benchmarks.sh
```

This script will:
1. Install all necessary dependencies
2. Build and configure each database
3. Run the benchmarks in sequence
4. Generate result files in the `sota` directory

## Running Individual Benchmarks

You can also run individual benchmarks to test specific databases:

### 1. RocksDB Benchmark

```bash
# Install RocksDB dependencies
./sota/install_rocksdb_deps.sh

# Run the benchmark
cargo bench --bench rocksdb_comparison --features use_rocksdb -- --verbose
```

The results will be saved to `sota/rocksdb_comparison_results.csv`.

### 2. SpeedB Benchmark

```bash
# Build and run SpeedB benchmark
./sota/benchmark_speedb.sh
```

The results will be saved to `sota/speedb_comparison_results.csv`.

### 3. LevelDB Benchmark

```bash
# Install LevelDB dependencies
./sota/install_leveldb_deps.sh

# Run the benchmark
./sota/benchmark_leveldb.sh
```

The results will be saved to `sota/leveldb_comparison_results.csv`.

### 4. LMDB Benchmark

```bash
# Install LMDB dependencies
./sota/install_lmdb_deps.sh

# Run the benchmark
./sota/benchmark_lmdb.sh
```

The results will be saved to `sota/lmdb_comparison_results.csv`.

### 5. WiredTiger Benchmark

```bash
# Install WiredTiger dependencies
./sota/install_wiredtiger_deps.sh

# Run the benchmark
./sota/benchmark_wiredtiger.sh
```

The results will be saved to `sota/wiredtiger_comparison_results.csv`.

### 6. TerarkDB Benchmark

```bash
# Install TerarkDB dependencies
./sota/install_terarkdb_deps.sh

# Run the benchmark
./sota/benchmark_terarkdb.sh
```

The results will be saved to `sota/terarkdb_comparison_results.csv`.

## Analyzing Benchmark Results

After running the benchmarks, you can analyze the results using:

```bash
./sota/run_analysis.sh
```

This script:
1. Sets up a Python environment using Poetry
2. Installs visualization dependencies
3. Processes the benchmark data
4. Generates visualizations and reports

The visualizations will be saved to the `sota/visualizations` directory.

## Troubleshooting Common Issues

### Missing Dependencies

If you encounter errors about missing libraries, run the individual dependency installation scripts:

```bash
# For RocksDB
./sota/install_rocksdb_deps.sh

# For LevelDB
./sota/install_leveldb_deps.sh

# For LMDB
./sota/install_lmdb_deps.sh

# For WiredTiger
./sota/install_wiredtiger_deps.sh

# For TerarkDB
./sota/install_terarkdb_deps.sh
```

### Compilation Errors

If you encounter compilation errors:

1. Make sure you have all required development libraries installed
2. Check if the correct version of Rust is being used (run `rustup update`)
3. Run `cargo clean` and try again

### TerarkDB Build Issues

TerarkDB might require additional dependencies:

```bash
sudo apt-get install -y libaio-dev
```

If the TerarkDB build fails, check:
1. The library was properly cloned from GitHub
2. All submodules were initialized
3. The build script has execute permissions

### Analysis Script Errors

If the analysis script fails:

1. Check if Poetry is installed (`curl -sSL https://install.python-poetry.org | python3 -`)
2. Make sure the Python version is at least 3.8
3. Try running `cd sota && poetry install --no-root` manually

## Interpreting the Results

After running the analysis, several files will be generated:

1. `sota/visualizations/throughput_by_operation.png`: Bar charts showing throughput for each operation type
2. `sota/visualizations/speedup_comparison.png`: Heatmaps showing relative performance
3. `sota/visualizations/summary_comparison.png`: Overall performance comparison
4. `sota/visualizations/speedup_report.csv`: Raw speedup data for detailed analysis

Additional benchmark details are available in the following files:

- `sota/comparison_results.md`: Detailed analysis of all benchmark results
- `sota/benchmark_summary.md`: High-level summary of the key findings

## Customizing Benchmarks

To customize workload sizes or other benchmark parameters:

1. For TerarkDB and WiredTiger, edit the respective benchmark scripts
2. For RocksDB, SpeedB, and LevelDB, modify the benchmark parameters in `benches/rocksdb_comparison.rs`
3. For LMDB, adjust settings in `benches/lmdb_comparison.rs`

## Exporting Results

To export the benchmark results for use in external tools:

1. The raw CSV files in the `sota` directory can be imported into spreadsheet software
2. The visualizations in the `sota/visualizations` directory are in PNG format
3. The speedup report CSV can be used for custom analysis

## Advanced: Running on Different Hardware

To run the benchmarks on different hardware:

1. Copy the entire repository to the target machine
2. Install the required dependencies
3. Run the same benchmark commands
4. Results will reflect the performance on that specific hardware

To compare across different hardware:
1. Keep the CSV files from each run
2. Run the analysis on each set of results separately
3. Compare the visualizations to see relative performance across hardware

## Understanding Benchmark Methodology

### Benchmark Design Principles

Our benchmark methodology follows these key principles to ensure scientific integrity:

1. **Real Database Integration**: All benchmarks use actual database implementations, not simulations or mocks.
   - Each database is properly installed using dedicated scripts
   - System dependencies are installed via package managers
   - For RocksDB and SpeedB, we integrate with official crates
   - For TerarkDB and WiredTiger, we create custom C/C++ wrappers

2. **Identical Workloads**: All databases are tested with identical workload patterns:
   - Small workload: 1,000 puts, 100 gets, 10 ranges, 10 deletes
   - Medium workload: 5,000 puts, 500 gets, 50 ranges, 50 deletes
   - Large workload: 100,000 puts, 10,000 gets, 1,000 ranges, 1,000 deletes

3. **Fair Comparison**: Steps taken to ensure fairness:
   - All databases use their recommended optimizations
   - The same key-value pair sizes and distributions
   - Identical hardware for all benchmarks
   - Multiple runs to account for variability

4. **Reproducible Results**: All benchmark code and scripts are:
   - Version controlled
   - Well-documented
   - Designed to be reproducible on different hardware

### Measurement Approach

Each benchmark measures:

1. **Put operations**: How fast data can be written
2. **Get operations**: How fast data can be read
3. **Range operations**: How fast ranges of data can be scanned
4. **Delete operations**: How fast data can be removed

### Database Integration Approach

We used different integration approaches based on each database's characteristics:

1. **RocksDB & SpeedB**: Direct Rust bindings using the `rocksdb` crate
2. **LevelDB**: Rust bindings using the `leveldb` crate
3. **LMDB**: Rust bindings using the `lmdb-rkv` crate
4. **TerarkDB**: Custom C++ wrapper around the TerarkDB API
5. **WiredTiger**: Custom C wrapper around the WiredTiger API

### Result Analysis

After collecting raw benchmark data:
1. Results are saved to standardized CSV files
2. Our Python analysis scripts process the data
3. Speedup ratios are calculated for each operation type
4. Visualizations highlight the relative performance
5. Reports summarize the key findings and insights

This comprehensive approach ensures that our performance claims are based on real, reproducible benchmarks against properly configured database systems.

## Setting Up Benchmark Environment

For optimal results:

1. Close other resource-intensive applications
2. Ensure consistent power settings (avoid thermal throttling)
3. Run benchmarks on the same storage device for fair comparison
4. Consider running each benchmark multiple times and averaging results