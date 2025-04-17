# Storage Implementation Comparison

This document provides a performance comparison between the two storage implementations for our LSM-tree:

1. **FileStorage**: A simple file-based approach that stores each run in a separate file
2. **LSFStorage**: A log-structured file approach that combines multiple runs into log segments

## Performance Summary

### Store Operations

| Data Size | FileStorage | LSFStorage | Ratio (LSF/File) |
|-----------|-------------|------------|------------------|
| 10 KV pairs | 87.158µs | 29.674µs | 0.34 |
| 100 KV pairs | 96.903µs | 79.408µs | 0.82 |
| 1000 KV pairs | 346.992µs | 419.966µs | 1.21 |
| 10000 KV pairs | 2.270575ms | 3.036134ms | 1.34 |

### Multiple Small Runs

| Run Count | FileStorage | LSFStorage | Ratio (LSF/File) |
|-----------|-------------|------------|------------------|
| 10 runs | 305.831µs | 75.937µs | 0.25 |
| 100 runs | 5.172342ms | 440.228µs | 0.09 |
| 1000 runs | 321.752212ms | 4.514932ms | 0.01 |

## Analysis

1. **Single Run Storage**:
   - For small runs (10-100 KV pairs), LSF is faster than file-based storage
   - For larger runs (1000-10000 KV pairs), file-based storage is slightly faster
   - This is likely due to LSF's overhead for maintaining log segments and index

2. **Multiple Run Storage**:
   - LSF is dramatically faster when storing many small runs
   - For 1000 runs, LSF is ~100x faster than file-based storage
   - This demonstrates LSF's major advantage: reduced file system overhead

3. **Run Loading**:
   - Not properly benchmarked due to serialization issues in the LSF implementation
   - In theory, LSF should be faster for loading runs when many runs are stored in the same log segment

## Implementation Differences

### FileStorage

- Creates one file per run
- Simple implementation, easy to understand
- Efficient for large runs
- No garbage collection needed
- High file system overhead for many small runs

### LSFStorage

- Combines multiple runs into log segments
- More complex implementation with index maintenance
- More efficient for many small runs
- Requires garbage collection/compaction
- Lower per-run overhead

## Next Steps

1. **Fix LSF Load Operation**: Resolve the checksum verification issues in the LSF implementation
2. **Implement Log Compaction**: Add garbage collection to reclaim space from deleted runs
3. **Add Memory-Mapped Storage**: Implement memory-mapped file approach for even faster access
4. **Add Tiered Storage**: Implement a tiered approach that combines in-memory and disk storage
5. **Conduct Comprehensive Benchmarks**: Compare all implementations under realistic workloads
