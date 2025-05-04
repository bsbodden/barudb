# LSM-Tree Benchmarks

This directory contains benchmark scripts and results for the LSM-Tree implementation.

## Available Benchmarks

1. **Bloom Filter Benchmarks** (`run_bloom_bench.sh`)
   - Tests different bloom filter implementations: standard, RocksDB-inspired, FastBloom, and SpeeDB
   - Measures insert and lookup performance
   - Evaluates false positive rates

2. **Fence Pointer Benchmarks** (`run_fence_bench.sh`)
   - Compares standard fence pointers with FastLane and Eytzinger layout
   - Measures performance across different data sizes
   - Evaluates range query performance

## Running the Benchmarks

```bash
# Run bloom filter benchmarks
./run_bloom_bench.sh

# Run fence pointer benchmarks
./run_fence_bench.sh

# Visualize benchmark results
python visualize_benchmarks.py
```

## Benchmark Results

### Bloom Filter Performance

The bloom filter benchmarks show:

1. **Insert Performance**:
   - FastBloom implementation is the fastest for insertions
   - Batch insertions provide significant performance improvements
   - SpeeDB implementation has competitive insert performance

2. **Lookup Performance**:
   - Batched lookups are the fastest overall
   - SpeeDB and FastBloom implementations are very close in performance
   - Standard bloom filter implementation is the slowest

3. **False Positive Rates**:
   - RocksDB-inspired implementation offers better false positive control
   - SpeeDB implementation achieves lower false positive rates for the same memory usage

### Fence Pointer Performance

The fence pointer benchmarks show:

1. **Point Query Performance**:
   - Eytzinger layout provides 40-65% improvement over standard fence pointers
   - Performance improvements grow with data size due to better cache efficiency

2. **Range Query Performance**:
   - For small ranges, Eytzinger layout shows 20-25% improvement
   - For large ranges, improvement increases to 30-40%
   - FastLane implementation falls between standard and Eytzinger

3. **Memory Efficiency**:
   - All implementations use the same amount of memory
   - Performance improvements come from better memory layout, not increased memory usage