#!/bin/bash
set -e

echo "=== Running all benchmarks for state-of-the-art database comparison ==="

# Change to the project root directory
cd "$(dirname "$0")/.."

# Run RocksDB benchmark
echo -e "\n=== Running RocksDB benchmark ==="
source ./sota/install_rocksdb_deps.sh
cargo bench --bench rocksdb_comparison --features use_rocksdb -- --verbose

# Run SpeedB benchmark
echo -e "\n=== Running SpeedB benchmark ==="
./sota/benchmark_speedb.sh

# Run LevelDB benchmark
echo -e "\n=== Running LevelDB benchmark ==="
./sota/benchmark_leveldb.sh

# Run LMDB benchmark
echo -e "\n=== Running LMDB benchmark ==="
./sota/benchmark_lmdb.sh

# Run WiredTiger benchmark
echo -e "\n=== Running WiredTiger benchmark ==="
./sota/benchmark_wiredtiger.sh

# Run TerarkDB benchmark
echo -e "\n=== Running TerarkDB benchmark ==="
./sota/benchmark_terarkdb.sh

# Run the analysis to generate visualizations
echo -e "\n=== Running benchmark analysis ==="
cd sota
./run_analysis.sh

echo -e "\n=== All benchmarks completed successfully! ==="
echo "Results and visualizations are available in the sota/visualizations directory"
echo "See sota/comparison_results.md for detailed analysis"