# Compression in LSM Tree

This document describes the compression options available in the LSM Tree implementation, how to configure them, and how to run experiments to compare different strategies.

## Available Compression Strategies

The LSM Tree implementation supports multiple compression strategies:

1. **NoopCompression**: No compression, acts as a baseline for performance comparisons.
2. **BitPackCompression**: Compresses integer data using bit-packing techniques, targeting data with small value ranges.
3. **DeltaCompression**: Stores differences between consecutive values, efficient for sequential or slowly changing data.
4. **DictionaryCompression**: Compresses repeated values using dictionary encoding, optimal for workloads with frequent value repetition.

## Configuring Compression

Compression can be configured at both the global and per-level granularity:

### Basic Configuration

```rust
let config = LSMConfig {
    // Enable compression
    compression: CompressionConfig {
        enabled: true,
        // Use bit-packing for all levels
        l0_default: CompressionType::BitPack,
        lower_level_default: CompressionType::BitPack,
        ..Default::default()
    },
    ..Default::default()
};
```

### Level-Specific Configuration

```rust
let mut level_types = vec![None; 10]; // Default for all levels
level_types[0] = Some(CompressionType::None); // No compression for L0
level_types[1] = Some(CompressionType::BitPack); // Bit-packing for L1

let config = LSMConfig {
    compression: CompressionConfig {
        enabled: true,
        level_types,
        ..Default::default()
    },
    ..Default::default()
};
```

### Adaptive Compression

For automatic selection of the best compression strategy based on data characteristics:

```rust
let config = LSMConfig {
    adaptive_compression: AdaptiveCompressionConfig {
        enabled: true,
        level_aware: true, // Use different strategies for different levels
        min_compression_ratio: 1.2, // Only compress if we get at least 20% improvement
        ..Default::default()
    },
    ..Default::default()
};
```

### Collecting Statistics

To gather statistics about compression performance:

```rust
let config = LSMConfig {
    // Enable compression statistics collection
    collect_compression_stats: true,
    ..Default::default()
};
```

## Running Compression Benchmarks

The implementation includes benchmarking tools to evaluate compression performance:

```bash
# Run the compression benchmarks
cargo bench --bench compression_bench

# Run specific bench for bit-packing only
cargo bench --bench compression_bench -- bit_pack
```

## Compression Statistics

During operation, each run maintains statistics about its compression performance. These can be accessed through the `compression_stats` field:

```rust
// Get compression stats for a run
let run = /* get run from level */;
if let Some(stats) = &run.compression_stats {
    println!("Strategy: {}", stats.strategy_name);
    println!("Compression ratio: {:.2}x", stats.compression_ratio);
    println!("Original size: {} bytes", stats.original_size);
    println!("Compressed size: {} bytes", stats.compressed_size);
    println!("Compression time: {:.3} ms", stats.compression_time_ms);
    println!("Decompression time: {:.3} ms", stats.decompression_time_ms);
}
```

## Design Considerations

### Performance vs. Space Tradeoffs

Different levels of the LSM tree have different performance characteristics:

* **L0**: Frequently accessed, benefits from faster decompression
* **L1-L2**: Moderate access frequency, balanced compression
* **L3+**: Less frequently accessed, higher compression ratio is beneficial

The default configuration uses:
- No compression for memtable (in-memory data)
- Bit-packing for L0 (good balance of speed and compression)
- Bit-packing for lower levels (with larger block sizes)

### Implementation Architecture

The compression system is built around three key components:

1. **CompressionStrategy Trait**: An interface implemented by all compression algorithms, providing:
   - `compress(&self, data: &[u8]) -> Result<Vec<u8>>`: Transforms uncompressed data into a compressed format.
   - `decompress(&self, data: &[u8]) -> Result<Vec<u8>>`: Reconstructs the original data from compressed data.
   - `estimate_compressed_size(&self, data: &[u8]) -> usize`: Provides a quick estimate of compression ratio.

2. **CompressionFactory**: Creates compression strategy instances based on type:
   ```rust
   // Create a compression strategy by type
   let compressor = CompressionFactory::create(CompressionType::BitPack);
   ```

3. **Block-level Integration**: Compression is integrated at the block level to maintain data integrity:
   - Each block is serialized with headers and data
   - Checksums are calculated to verify data integrity
   - The entire block (including headers) is compressed
   - Padding is applied to ensure data alignment for compression algorithms

### Block Size Impact

Compression efficiency increases with block size:

* **Small blocks** (256 entries): Lower latency, less compression
* **Medium blocks** (1024 entries): Good balance
* **Large blocks** (4096+ entries): Better compression, higher latency

The default block size is 1024 entries.

### Data Format Requirements

Key architectural decision in our implementation:

1. **Data Alignment**: All compression algorithms require data to be a multiple of 16 bytes (the size of a key-value pair).

2. **Padding Mechanism**: When serializing blocks, padding bytes are added after the checksum to maintain data integrity:
   ```rust
   // Calculate checksum before padding
   let checksum = xxhash_rust::xxh3::xxh3_64(&data);
   data.extend_from_slice(&checksum.to_le_bytes());
   
   // Add padding to align to 16-byte boundary
   let padding_needed = (16 - (data.len() % 16)) % 16;
   for _ in 0..padding_needed {
       data.push(0);
   }
   ```

3. **Checksum Validation**: During deserialization, the system calculates where the checksum should be based on block structure, not just at the end of the data:
   ```rust
   // Calculate expected data size from header information
   let expected_size = header_size + (entry_count * 16) + 8; // +8 for checksum
   let checksum_offset = expected_size - 8;
   
   // Calculate and verify checksum
   let computed_checksum = xxh3_64(&data[..checksum_offset]);
   let stored_checksum = u64::from_le_bytes(data[checksum_offset..checksum_offset+8]);
   
   // Validate the checksum
   if computed_checksum != stored_checksum {
       return Err(...);
   }
   ```

## Compression Algorithm Details

### BitPackCompression

BitPackCompression efficiently stores integers by using the minimum number of bits required:

1. **Key Insight**: Most datasets don't need the full 64 bits to represent values, especially when they have a small range.

2. **Algorithm**:
   - Find the minimum and maximum values in the dataset
   - Calculate the number of bits needed to represent the range (log2 of range)
   - Store the base value (minimum) and offsets using only the required bits
   - For sequential values, use a special encoding to further reduce size

3. **Optimizations**:
   - Blocks of values are processed together to amortize overhead
   - Automatic detection of sequential keys for additional compression
   - Special handling for repeated values

4. **Performance Characteristics**:
   - Best for: small-range integer data, sequential IDs, timestamp columns
   - Compression ratios: 5-10x for sequential data, 3-5x for small-range data
   - Speed: ~600-800µs compression, ~400-500µs decompression for 10K entries

### DeltaCompression

DeltaCompression encodes the differences between consecutive values:

1. **Key Insight**: In sorted data (like LSM-Tree keys), consecutive values often have small differences.

2. **Algorithm**:
   - Store the first value directly
   - For subsequent values, store only the difference from the previous value
   - Use variable-length encoding for the deltas to save even more space

3. **Optimizations**:
   - ZigZag encoding to efficiently represent signed differences
   - Variable-length encoding (similar to Protocol Buffers) for the deltas
   - Safe overflow handling for 64-bit values

4. **Performance Characteristics**:
   - Best for: sequential IDs, timestamps, sensor readings, anything with small step sizes
   - Compression ratios: 7-8x for sequential data, 4-5x for small-delta data
   - Speed: ~300-350µs compression, ~350-400µs decompression for 10K entries

### DictionaryCompression

DictionaryCompression replaces repeated values with indices into a dictionary:

1. **Key Insight**: Many real-world datasets contain significant value repetition.

2. **Algorithm**:
   - Count frequency of each key-value pair
   - Build a dictionary of the most common values
   - Replace values with dictionary indices when possible
   - Use a special marker for values not in the dictionary

3. **Optimizations**:
   - Frequency-based dictionary construction
   - Support for partial dictionary matches
   - Adaptive dictionary size based on data characteristics

4. **Performance Characteristics**:
   - Best for: log data, categorical data, data with high repetition
   - Compression ratios: 1.5-1.6x for repeated data, <1x for random data (slight overhead)
   - Speed: ~6-10ms compression, ~400-500µs decompression for 10K entries

## Experimental Results

Our integration tests compare compression strategies across different data patterns:

```
=== Compression Ratio Comparison ===

Sequential Data (160000 bytes):
  Delta compression: 20260 bytes, ratio: 7.90x, compress: 326.369µs, decompress: 357.91µs
  BitPack compression: 17682 bytes, ratio: 9.05x, compress: 648.442µs, decompress: 374.985µs
  Dictionary compression: 180100 bytes, ratio: 0.89x, compress: 9.572912ms, decompress: 481.001µs

Random Data (160000 bytes):
  Delta compression: 39854 bytes, ratio: 4.01x, compress: 326.825µs, decompress: 359.55µs
  BitPack compression: 41432 bytes, ratio: 3.86x, compress: 844.777µs, decompress: 609.717µs
  Dictionary compression: 180100 bytes, ratio: 0.89x, compress: 9.44623ms, decompress: 520.496µs

Repeated Data (160000 bytes):
  Delta compression: 20298 bytes, ratio: 7.88x, compress: 270.296µs, decompress: 322.951µs
  BitPack compression: 27780 bytes, ratio: 5.76x, compress: 790.615µs, decompress: 519.492µs
  Dictionary compression: 100100 bytes, ratio: 1.60x, compress: 6.434332ms, decompress: 404.174µs
```

### Key Findings

1. **BitPack** provides the best compression ratio for sequential data (9.05x)
2. **Delta** offers the fastest compression speed (270-326µs)
3. **Dictionary** excels only when data has significant repetition
4. **Dictionary** has the highest compression overhead (6-10ms)
5. All strategies achieve reasonable decompression speed (320-610µs)

## Running Tests and Experiments

We provide comprehensive testing infrastructure:

1. **Basic Unit Tests**:
   ```bash
   # Test all compression algorithms directly
   cargo test --test compression_test
   ```

2. **Integration Tests**:
   ```bash
   # Test LSM Tree with all compression types
   cargo test --test compression_test test_all_compression_types -- --nocapture
   ```

3. **Performance Benchmarks**:
   ```bash
   # Run complete benchmarks across all compression types and data patterns
   cargo test --test compression_test test_compression_strategies -- --ignored --nocapture
   ```

## Future Improvements

The compression system is designed to be extensible. Planned future improvements:

1. **SIMD acceleration** for bit-packing and delta encoding
2. **Hybrid compression** combining multiple strategies based on data patterns
3. **LZ4/Snappy integration** for general-purpose compression
4. **Adaptive block sizing** based on data characteristics
5. **Compression level selection** based on access patterns in different LSM tree levels

## Troubleshooting

Common issues:

1. **Slow compression**: Try reducing block size or using a faster algorithm
2. **Low compression ratio**: Check data patterns, bit-packing works best with sequential or limited-range data
3. **High memory usage**: Consider adjusting buffer sizes or block sizes