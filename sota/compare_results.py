#!/usr/bin/env python3
import csv
import os
import sys
from collections import defaultdict

# Try to import plotting libraries, but continue if they're not available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    plotting_available = True
except ImportError:
    plotting_available = False
    print("matplotlib or numpy not found. Visualization will be skipped.")
    print("To install: pip install matplotlib numpy")
    print("Continuing with text-based analysis...\n")

def read_benchmark_results(file_path):
    results = defaultdict(lambda: defaultdict(dict))
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            db_name = row['db_name']
            operation = row['operation']
            count = int(row['count'])
            avg_time = float(row['avg_time_micros'])
            throughput = float(row['throughput_ops_per_sec'])
            
            # Group by operation size (derived from count)
            if count == 1000 or count == 10:
                size = "small"
            elif count == 5000 or count == 50:
                size = "medium"
            elif count == 100000 or count == 1000:
                size = "large"
            else:
                size = "unknown"
                
            if (operation, size) not in results[db_name]:
                results[db_name][(operation, size)] = {
                    'counts': [],
                    'avg_times': [],
                    'throughputs': []
                }
                
            results[db_name][(operation, size)]['counts'].append(count)
            results[db_name][(operation, size)]['avg_times'].append(avg_time)
            results[db_name][(operation, size)]['throughputs'].append(throughput)
            
    # Calculate averages for each operation/size combination
    for db_name in results:
        for op_size in results[db_name]:
            operation, size = op_size
            times = results[db_name][op_size]['avg_times']
            throughputs = results[db_name][op_size]['throughputs']
            
            if times:
                results[db_name][op_size]['avg_time'] = sum(times) / len(times)
            if throughputs:
                results[db_name][op_size]['avg_throughput'] = sum(throughputs) / len(throughputs)
                
    return results

def compare_results(lsm_results, rocksdb_results, speedb_results=None):
    all_operations = set()
    all_sizes = set()
    
    # Collect all operations and sizes
    for results in [lsm_results, rocksdb_results]:
        if results is None:
            continue
        for db_name in results:
            for op_size in results[db_name]:
                operation, size = op_size
                all_operations.add(operation)
                all_sizes.add(size)
    
    all_operations = sorted(list(all_operations))
    all_sizes = sorted(list(all_sizes), key=lambda x: {"small": 0, "medium": 1, "large": 2}.get(x, 3))
    
    print("\n=== BENCHMARK COMPARISON ===\n")
    
    for size in all_sizes:
        print(f"\n{size.upper()} WORKLOAD:\n")
        print(f"{'Operation':<10} {'LSM Tree':<15} {'RocksDB':<15} {'SpeedB':<15} {'LSM vs RocksDB':<15} {'LSM vs SpeedB':<15} {'SpeedB vs RocksDB':<15}")
        print('-' * 100)
        
        for operation in all_operations:
            lsm_throughput = lsm_results.get('LSM Tree', {}).get((operation, size), {}).get('avg_throughput', 0)
            rocksdb_throughput = rocksdb_results.get('RocksDB', {}).get((operation, size), {}).get('avg_throughput', 0)
            
            speedb_throughput = 0
            if speedb_results:
                speedb_throughput = speedb_results.get('RocksDB', {}).get((operation, size), {}).get('avg_throughput', 0)
            
            # Calculate speedups
            lsm_vs_rocksdb = lsm_throughput / rocksdb_throughput if rocksdb_throughput else float('inf')
            lsm_vs_speedb = lsm_throughput / speedb_throughput if speedb_throughput else float('inf')
            speedb_vs_rocksdb = speedb_throughput / rocksdb_throughput if rocksdb_throughput and speedb_throughput else float('inf')
            
            print(f"{operation:<10} "
                  f"{lsm_throughput:,.0f} ops/s  "
                  f"{rocksdb_throughput:,.0f} ops/s  "
                  f"{speedb_throughput:,.0f} ops/s  "
                  f"{lsm_vs_rocksdb:,.2f}x  "
                  f"{lsm_vs_speedb:,.2f}x  "
                  f"{speedb_vs_rocksdb:,.2f}x  ")

def plot_comparison(lsm_results, rocksdb_results, speedb_results=None):
    all_operations = set()
    all_sizes = set()
    
    # Collect all operations and sizes
    for results in [lsm_results, rocksdb_results, speedb_results]:
        if results is None:
            continue
        for db_name in results:
            for op_size in results[db_name]:
                operation, size = op_size
                all_operations.add(operation)
                all_sizes.add(size)
    
    all_operations = sorted(list(all_operations))
    all_sizes = sorted(list(all_sizes), key=lambda x: {"small": 0, "medium": 1, "large": 2}.get(x, 3))
    
    # Create a plot for each operation type
    for operation in all_operations:
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(all_sizes))
        width = 0.25
        
        lsm_values = []
        rocksdb_values = []
        speedb_values = []
        
        for size in all_sizes:
            lsm_throughput = lsm_results.get('LSM Tree', {}).get((operation, size), {}).get('avg_throughput', 0)
            rocksdb_throughput = rocksdb_results.get('RocksDB', {}).get((operation, size), {}).get('avg_throughput', 0)
            
            lsm_values.append(lsm_throughput)
            rocksdb_values.append(rocksdb_throughput)
            
            if speedb_results:
                speedb_throughput = speedb_results.get('RocksDB', {}).get((operation, size), {}).get('avg_throughput', 0)
                speedb_values.append(speedb_throughput)
        
        # Plot bars
        plt.bar(x - width, lsm_values, width, label='LSM Tree')
        plt.bar(x, rocksdb_values, width, label='RocksDB')
        
        if speedb_results:
            plt.bar(x + width, speedb_values, width, label='SpeedB')
        
        plt.xticks(x, all_sizes)
        plt.xlabel('Workload Size')
        plt.ylabel('Throughput (ops/sec)')
        plt.title(f'{operation.capitalize()} Operation Performance')
        plt.legend()
        plt.grid(True, axis='y')
        
        # Use log scale if there are large differences
        max_value = max(max(lsm_values), max(rocksdb_values))
        min_value = min(min(v for v in lsm_values if v > 0), min(v for v in rocksdb_values if v > 0))
        
        if speedb_results:
            max_value = max(max_value, max(speedb_values))
            min_value = min(min_value, min(v for v in speedb_values if v > 0))
        
        if max_value / min_value > 100:
            plt.yscale('log')
        
        plt.savefig(f'sota/{operation}_comparison.png')

if __name__ == "__main__":
    # Check if we have SpeedB results
    has_speedb = os.path.exists('sota/speedb_comparison_results.csv')
    
    # Read benchmark results
    results_file = 'sota/rocksdb_comparison_results.csv'
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
        
    # Parse the results file once and separate by database name
    all_results = read_benchmark_results(results_file)
    lsm_results = {}
    rocksdb_results = {}
    
    # Extract results by database name
    for db_name, data in all_results.items():
        if "LSM Tree" in db_name:
            lsm_results[db_name] = data
        elif "RocksDB" in db_name:
            rocksdb_results[db_name] = data
    
    # Load SpeedB results if available
    speedb_results = None
    if has_speedb:
        speedb_file = 'sota/speedb_comparison_results.csv'
        if os.path.exists(speedb_file):
            speedb_results = read_benchmark_results(speedb_file)
        else:
            print(f"Warning: SpeedB results file {speedb_file} not found")
    
    # Compare results
    compare_results(lsm_results, rocksdb_results, speedb_results)
    
    # Generate plots if plotting libraries are available
    if plotting_available:
        try:
            plot_comparison(lsm_results, rocksdb_results, speedb_results)
            print("\nComparison plots saved to sota/ directory")
        except Exception as e:
            print(f"Error generating plots: {e}")
            print("Make sure matplotlib is installed correctly: pip install matplotlib")
    else:
        print("\nSkipping plot generation (matplotlib not available)")
        print("To enable visualization: pip install matplotlib numpy")