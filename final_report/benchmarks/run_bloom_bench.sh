#!/bin/bash
# Run bloom filter benchmarks and save results to CSV

OUTPUT_DIR="$(pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Running bloom filter benchmarks..."
cd /home/bsb/Code/hes/cs265-lsm-tree

# Run the benchmark with a time limit of 5 minutes
timeout 300 cargo bench --bench bloom_bench | tee "${OUTPUT_DIR}/bloom_bench_raw_${TIMESTAMP}.txt"

echo "Extracting data from benchmark results..."

# Extract core results and save to a CSV
echo "Implementation,Operation,Size,Time_ns,StdDev_ns" > "${OUTPUT_DIR}/bloom_bench_results_${TIMESTAMP}.csv"

# Process the raw output to extract results
grep "bloom_filters/" "${OUTPUT_DIR}/bloom_bench_raw_${TIMESTAMP}.txt" | grep "time:" | \
    sed -E 's/bloom_filters\/([^\/]+)\/([0-9]+).*time:\s+\[([0-9.]+) µs ([0-9.]+) µs ([0-9.]+) µs\].*/\1,\2,\3,\5/' | \
    awk -F, '{printf "%s,%s,%s,%.2f,%.2f\n", $1, $2, $3, $4 * 1000, $5 * 1000}' >> "${OUTPUT_DIR}/bloom_bench_results_${TIMESTAMP}.csv"

# Extract comparison data between implementations
echo "Filter1,Filter2,Operation,Size,Ratio" > "${OUTPUT_DIR}/bloom_bench_comparison_${TIMESTAMP}.csv"

# Create symlinks to the latest results
ln -sf "bloom_bench_results_${TIMESTAMP}.csv" "${OUTPUT_DIR}/bloom_bench_results_latest.csv"
ln -sf "bloom_bench_raw_${TIMESTAMP}.txt" "${OUTPUT_DIR}/bloom_bench_raw_latest.txt"
ln -sf "bloom_bench_comparison_${TIMESTAMP}.csv" "${OUTPUT_DIR}/bloom_bench_comparison_latest.csv"

echo "Bloom filter benchmark results saved to ${OUTPUT_DIR}/bloom_bench_results_${TIMESTAMP}.csv"
echo "Bloom filter comparisons saved to ${OUTPUT_DIR}/bloom_bench_comparison_${TIMESTAMP}.csv"
echo "Full benchmark output saved to ${OUTPUT_DIR}/bloom_bench_raw_${TIMESTAMP}.txt"