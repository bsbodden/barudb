#!/bin/bash
set -e

# This script runs the RocksDB comparison benchmark but using SpeedB library instead

echo "==== Installing SpeedB ===="
# Check if SpeedB is already installed
if [ ! -d "speedb" ]; then
    echo "Cloning SpeedB repository..."
    git clone https://github.com/speedb-io/speedb.git
    cd speedb
    echo "Building SpeedB..."
    make static_lib -j$(nproc)
    cd ..
fi

# Directory where SpeedB is installed
SPEEDB_DIR=$(realpath ./speedb)
SPEEDB_LIB_DIR=$SPEEDB_DIR

echo "==== Setting up environment to use SpeedB ===="
# Create directory for our benchmark with SpeedB
mkdir -p speedb_benchmark
cd speedb_benchmark

# Tell the dynamic linker to use SpeedB instead of RocksDB
export LD_LIBRARY_PATH=$SPEEDB_LIB_DIR:$LD_LIBRARY_PATH

# Also set library path for the Rust rocksdb crate
export ROCKSDB_LIB_DIR=$SPEEDB_LIB_DIR
export ROCKSDB_STATIC=1

# Set the database name for the benchmark results
export DB_NAME="SpeedB"

echo "==== Running benchmark with SpeedB ===="
# Run the benchmark with the RocksDB feature (it will use SpeedB library)
cd ..
cargo bench --bench rocksdb_comparison --features use_rocksdb -- --verbose

# Rename the results file to indicate it's for SpeedB
mv sota/rocksdb_comparison_results.csv sota/speedb_comparison_results.csv

echo "==== Benchmark complete ===="
echo "Results saved to sota/speedb_comparison_results.csv"