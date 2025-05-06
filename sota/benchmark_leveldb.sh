#!/bin/bash
set -e

echo "==== Installing LevelDB Dependencies ===="
cd "$(dirname "$0")"
./install_leveldb_deps.sh

echo "==== Running LevelDB Benchmark ===="
# Set the database name for the benchmark results
export DB_NAME="LevelDB"

# Run the benchmark with the LevelDB feature
cargo bench --bench leveldb_comparison --features use_leveldb -- --verbose

echo "==== Benchmark complete ===="
echo "Results saved to sota/leveldb_comparison_results.csv"

echo "==== Running Analysis ===="
# Run the analysis to generate visualizations
./sota/run_analysis.sh

echo "==== All Done! ===="
echo "Review the visualizations in sota/visualizations/ to see how our LSM tree compares to LevelDB and other databases."