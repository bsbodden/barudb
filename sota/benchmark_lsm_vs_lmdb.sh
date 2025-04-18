#!/bin/bash
set -e

# Change to the project root directory
cd "$(dirname "$0")/.."
PROJ_ROOT=$(pwd)

# Install LMDB dependencies
cd $PROJ_ROOT
source ./sota/install_lmdb_deps.sh

# Create the results directory if it doesn't exist
mkdir -p ./sota/benchmark_results

# Clean any previous build artifacts
cargo clean

# Run the LMDB comparison benchmark
echo "Running LSM Tree vs LMDB benchmark..."

# Unset DB_NAME to use the defaults for each implementation
# LSM Tree will use "LSM Tree" and LMDB will use "LMDB"
unset DB_NAME

# Run the comparison benchmark
cargo bench --bench lmdb_comparison --features use_lmdb -- --verbose

# Run the analysis
cd sota
echo "Running benchmark analysis..."
./run_analysis.sh

echo "LSM Tree vs LMDB benchmarking completed successfully!"