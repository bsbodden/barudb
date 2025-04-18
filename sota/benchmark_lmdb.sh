#!/bin/bash
set -e

# Change to the project root directory
cd "$(dirname "$0")/.."
PROJ_ROOT=$(pwd)

# Install LMDB dependencies
cd $PROJ_ROOT
source ./sota/install_lmdb_deps.sh

# Set the database name for the benchmark results
export DB_NAME="LMDB"

# Create the results directory if it doesn't exist
mkdir -p ./sota/benchmark_results

# Clean any previous build artifacts
cargo clean

# Run the LMDB benchmark with the feature flag enabled
echo "Running LMDB benchmark..."
cargo bench --bench lmdb_comparison --features use_lmdb -- --verbose

# Run the analysis
cd sota
echo "Running benchmark analysis..."
./run_analysis.sh

echo "LMDB benchmarking completed successfully!"