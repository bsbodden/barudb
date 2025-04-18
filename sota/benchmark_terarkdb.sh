#!/bin/bash
set -e

# Change to the project root directory
cd "$(dirname "$0")/.."
PROJ_ROOT=$(pwd)
cd $PROJ_ROOT

echo "====================================================================="
echo "TerarkDB Benchmark"
echo "====================================================================="

# Check if TerarkDB is built
TERARKDB_DIR="$HOME/Code/hes/terarkdb"
TERARKDB_LIB="$TERARKDB_DIR/build/libterarkdb.a"

if [ ! -f "$TERARKDB_LIB" ]; then
    echo "TerarkDB library not found at $TERARKDB_LIB"
    echo "Please run the installation script first: ./sota/install_terarkdb_deps.sh"
    exit 1
fi

echo "Found TerarkDB library at: $TERARKDB_LIB"

# Create the results directory if it doesn't exist
mkdir -p ./sota/benchmark_results

# Ensure any existing faked results are removed
if [ -f "./sota/benchmark_results/terarkdb_comparison_results.csv" ]; then
    echo "Removing any existing TerarkDB benchmark results to ensure scientific integrity..."
    rm -f "./sota/benchmark_results/terarkdb_comparison_results.csv"
fi

# Compile the benchmark program
echo "Compiling TerarkDB benchmark program..."
g++ -std=c++17 -o ./sota/terarkdb_bench ./sota/terarkdb_bench.cpp \
    -I"$TERARKDB_DIR/include" -I"$TERARKDB_DIR" -I"$TERARKDB_DIR/utilities" \
    -L"$TERARKDB_DIR/build" \
    -lterarkdb -lpthread -ldl -lz -lbz2 -lsnappy -llz4 -lzstd -laio -O3

if [ $? -ne 0 ]; then
    echo "Failed to compile the benchmark program. Please check the errors above."
    exit 1
fi

# Set the database name
export DB_NAME="TerarkDB"

# Run the benchmark
echo "Running benchmark with $DB_NAME..."
./sota/terarkdb_bench 100000

# Check if the benchmark generated results
if [ ! -f "./sota/benchmark_results/terarkdb_comparison_results.csv" ]; then
    echo "ERROR: Benchmark did not generate results. Please check for errors."
    exit 1
fi

# Run the analysis
cd sota
echo "Running benchmark analysis..."
./run_analysis.sh

echo "====================================================================="
echo "BENCHMARK COMPLETE!"
echo "====================================================================="
echo ""
echo "Benchmarking completed successfully with $DB_NAME!"
echo "Results are available in sota/benchmark_results/terarkdb_comparison_results.csv"