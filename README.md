# LSM Tree Database (Rust Implementation)

This repository contains a Rust implementation of an LSM tree key-value store, following the specifications from
Harvard's CS265 Systems project. The implementation provides a client-server architecture for managing key-value data
using a Log-Structured Merge Tree.

## Prerequisites

### Rust Installation

You'll need Rust installed on your system. If you haven't installed Rust yet,
see https://www.rust-lang.org/tools/install for installation options

The project requires Rust 1.81.0 or later.

## Building the Project

Clone the repository and build using cargo:

```bash
git clone [repository-url]
cd lsm-tree
cargo build --release
```

## Testing

The project includes various types of tests to verify functionality and performance.

### Running Standard Tests

To run all regular tests (excluding long-running tests):

```bash
cargo test
```

This will run all tests except those marked with `#[ignore]`.

### Running Workload Tests

The workload tests simulate real-world database usage but can take a long time to complete:

```bash
# Run the workload integration test
cargo test test_workload_execution -- --ignored
```

### Running Server Command Tests

The server command tests verify functionality of server commands like PrintStats and Load:

```bash
# Ensure server is running on port 8080 first
./run_server_tests.sh
```

This script will:
1. Build the server if needed
2. Start the server if not already running
3. Run tests that verify PrintStats and Load command functionality
4. Shut down the server (if it started it)

### Running Performance Tests

Performance tests evaluate various aspects of the system but are excluded from regular test runs:

```bash
# Run all performance tests
cargo test -- --ignored

# Run a specific performance test
cargo test compare_storage_implementations -- --ignored
```

## Running Benchmarks

The project includes several benchmarks for evaluating performance of different components.

### Running all benchmarks

To run all benchmarks:

```bash
cargo bench
```

### Running specific benchmarks

To run a specific benchmark:

```bash
# Run bloom filter benchmarks 
# Tests different bloom filter implementations (SpeedDB, RocksDB, etc.)
cargo bench --bench bloom_bench

# Run storage benchmarks (Criterion benchmark)
cargo bench --bench storage_bench

# Run simpler storage benchmark with clearer output
cargo run --release --bin storage_bench_simple
```

### FastLane Fence Pointer Benchmark

There's a standalone benchmark for evaluating FastLane vs Standard fence pointers:

```bash
# Build and run the FastLane benchmark
cargo run --release --bin benchmark_fastlane
```

This benchmark compares performance across three key distribution patterns:
- Sequential keys (e.g., timestamps, auto-incrementing IDs)
- Random keys (high entropy)
- Grouped keys (keys with common prefixes by group)

## Running the Server

To launch the server:

```bash
cargo run --release --bin server [OPTIONS]
```

### Server Options

| Option              | Default   | Description                                                         |
|---------------------|-----------|---------------------------------------------------------------------|
| `-e <error_rate>`   | 0.01      | Bloom filter error rate                                             |
| `-n <num_pages>`    | 1024      | Size of the buffer by number of disk pages                          |
| `-f <fanout>`       | 4         | LSM tree fanout                                                     |
| `-l <level_policy>` | "tiered"  | Compaction policy (options: tiered, leveled, lazy_leveled)          |
| `-t <threshold>`    | 4         | Compaction threshold (runs for tiered/lazy_leveled)                 |
| `-p <port>`         | 8080      | Port number                                                         |
| `-h`                | N/A       | Print help message                                                  |

## Running the Client

To launch the client:

```bash
cargo run --release --bin client [OPTIONS]
```

### Client Options

| Option      | Description                 |
|-------------|-----------------------------|
| `-p <port>` | Port number (default: 8080) |
| `-q`        | Quiet mode                  |

## Supported Commands

The database supports the following commands:

### Client Commands

| Command           | Description                                     | Example        |
|-------------------|-------------------------------------------------|----------------|
| `p <key> <value>` | Put a key-value pair                            | `p 10 42`      |
| `g <key>`         | Get value for key                               | `g 10`         |
| `r <start> <end>` | Range query                                     | `r 10 20`      |
| `d <key>`         | Delete key                                      | `d 10`         |
| `l <filename>`    | Load commands from file                         | `l "data.txt"` |
| `s`               | Print stats (tree config, storage stats, etc.)  | `s`            |
| `f`               | Force memtable flush to disk                    | `f`            |
| `q`               | Quit                                            | `q`            |

The `l` (Load) command processes a file containing multiple commands, with each command on a separate line. It supports `p` (Put) and `d` (Delete) commands in the file.

The `s` (PrintStats) command displays comprehensive statistics about the LSM tree, including:
- Storage type and compaction policy
- Buffer size and fanout configuration
- Total storage size and file count
- Level-by-level statistics (runs, blocks, entries)

### Server Commands

While the server is running, you can enter these commands in the server terminal:

| Command | Description                |
|---------|----------------------------|
| `bloom` | Print Bloom Filter summary |
| `quit`  | Quit server                |
| `help`  | Print help message         |

## Project Structure

```
lsm-tree/
├── Cargo.toml
├── src/
│   ├── bin/
│   │   ├── client.rs    # Client implementation
│   │   └── server.rs    # Server implementation
│   ├── bloom/           # Bloom filter implementations
│   │   ├── mod.rs       # Core Bloom filter code
│   │   ├── rocks_db.rs  # RocksDB filter implementation
│   │   └── speed_db.rs  # SpeedDB filter implementation
│   ├── compaction/      # Compaction policy implementations
│   │   ├── mod.rs       # Compaction framework & factory
│   │   ├── tiered.rs    # Tiered compaction policy
│   │   ├── leveled.rs   # Leveled compaction policy
│   │   └── lazy_leveled.rs # Hybrid lazy leveled compaction
│   ├── run/             # Run (data block) implementations
│   │   ├── block.rs     # Block structure
│   │   ├── compression.rs # Data compression
│   │   ├── compressed_fence.rs # Prefix-compressed fence pointers
│   │   ├── fastlane_fence.rs # FastLane-optimized fence pointers
│   │   ├── fence.rs     # Base fence pointers
│   │   ├── filter.rs    # Block-level filters
│   │   ├── lsf.rs       # Log-structured file storage
│   │   ├── mod.rs       # Run module definitions
│   │   ├── standard_fence.rs # Standard fence pointers
│   │   ├── storage.rs   # Storage interface
│   │   └── two_level_fence.rs # Two-level fence pointers
│   ├── command.rs       # Command parsing
│   ├── level.rs         # Level management
│   ├── lib.rs           # Shared library code
│   ├── lsm_tree.rs      # Main LSM tree implementation
│   ├── memtable.rs      # In-memory buffer
│   ├── test_helpers.rs  # Testing utilities
│   └── types.rs         # Type definitions
├── benches/             # Performance benchmarks
├── tests/               # Integration tests
│   ├── compaction_integration_test.rs # Tests for compaction policies
│   ├── compaction_policy_test.rs     # Unit tests for compaction
│   ├── lsm_tree_compaction_test.rs   # LSM tree with compaction
│   └── ... other test files
└── README.md
```

## Development Status

The implementation includes:

- Full LSM tree implementation with multi-level storage
- Memory buffer (memtable) for fast writes
- Disk-based runs with sorted key-value pairs
- Optimized Bloom filters for fast negative lookups
- Optimized fence pointers for efficient range queries
  - Cache-aligned memory layout for better CPU cache utilization
  - Hardware prefetching for reduced memory latency
  - Two-level sparse/dense indexing structure for memory efficiency
  - Prefix compression for maximized memory efficiency with numeric keys
  - FastLane memory layout for improved cache locality during lookups
- Monkey-optimized Bloom filters for memory efficiency
- Complete compaction framework with three policies:
  - Tiered compaction: Multiple runs per level, compact when threshold reached
  - Leveled compaction: Single run per level, compact on any new run
  - Lazy Leveled compaction: Hybrid approach with tiered behavior at level 0
- Pluggable component architecture for easy experimentation

## Monkey-Optimized Bloom Filters

Implemented Bloom filters that follow the optimization strategy described in the Monkey paper. This strategy allocates different numbers of bits per entry based on the level in the LSM tree:

- Higher levels (closer to the buffer) get more bits per entry
- Lower levels get fewer bits per entry
- The allocation follows an exponential decay pattern based on the fanout ratio

This optimization provides several benefits:
1. Reduces the overall memory footprint of Bloom filters
2. Maintains low false positive rates for frequently accessed levels
3. Accepts slightly higher false positive rates for rarely accessed levels
4. Scales appropriately based on the tree's fanout configuration

The implementation shows the following performance characteristics (with fanout=4):

| Level | Bits per Entry | Memory Usage (10K entries) | False Positive Rate |
|-------|----------------|----------------------------|---------------------|
| 0     | 52.43          | 65,536 bytes              | 0.0000%             |
| 1     | 26.21          | 32,768 bytes              | 0.0000%             |
| 2     | 13.11          | 16,384 bytes              | 0.0000%             |
| 3     | 6.55           | 8,192 bytes               | 0.0000%             |
| 4     | 3.28           | 4,096 bytes               | 0.1900%             |

The bit allocation formula follows an exponential decay: for a given level `i` and fanout `T`, 
bits per entry = 32.0 / T^(i/2), with a minimum of 2 bits per entry.

This approach demonstrates a memory reduction of ~94% from level 0 to level 4, while maintaining excellent false positive rates even in the lowest levels.

## Compaction Policies

The implementation provides three distinct compaction policies, each with different performance characteristics:

### Tiered Compaction

The tiered compaction policy allows multiple runs per level and triggers compaction when a level has accumulated a configured number of runs:

```rust
pub struct TieredCompactionPolicy {
    /// Number of runs that trigger compaction
    run_threshold: usize,
}
```

**Characteristics**:
- Lower write amplification (fewer rewrites during compaction)
- Higher read amplification (must check multiple runs per level)
- Good for write-heavy workloads
- Configurable threshold for controlling when compaction occurs

### Leveled Compaction

The leveled compaction policy maintains exactly one run per level and triggers compaction whenever there would be more than one run at any level:

```rust
pub struct LeveledCompactionPolicy {
    /// Size ratio threshold between levels (usually matches the fanout)
    size_ratio_threshold: usize,
}
```

**Characteristics**:
- Higher write amplification (more rewrites during compaction)
- Lower read amplification (at most one run per level)
- Good for read-heavy workloads
- More predictable query performance

### Lazy Leveled Compaction

A hybrid policy that combines tiered and leveled approaches:

```rust
pub struct LazyLeveledCompactionPolicy {
    /// Threshold for number of runs in level 0 before compaction
    run_threshold: usize,
}
```

**Characteristics**:
- Level 0: Behaves like tiered compaction (multiple runs allowed)
- Higher levels: Behaves like leveled compaction (single run per level)
- Balances read and write performance
- Good for mixed workloads
- Reduces write amplification while maintaining good read performance

All policies are implemented as plugins that satisfy the `CompactionPolicy` trait, making it easy to switch between them or implement new strategies:

```rust
pub trait CompactionPolicy: Send + Sync {
    fn should_compact(&self, level: &Level, level_num: usize) -> bool;
    fn select_runs_to_compact(&self, level: &Level) -> Vec<usize>;
    fn compact(...) -> Result<Run>;
    fn box_clone(&self) -> Box<dyn CompactionPolicy>;
}
```

The compaction policy can be selected when creating the LSM tree or through the command-line interface.

## Optimized Fence Pointers

The implementation includes state-of-the-art fence pointer optimizations to improve both lookup performance and memory efficiency:

### Cache-Aligned Memory Layout

Implemented fence pointers with explicit memory alignment to improve CPU cache utilization:

```rust
#[repr(align(16))]
pub struct FencePointer {
    pub min_key: Key,
    pub max_key: Key,
    pub block_index: usize,
}

#[repr(align(64))] // Align to typical cache line size
pub struct FencePointers {
    pub pointers: Vec<FencePointer>,
}
```

This alignment ensures that fence pointers are loaded efficiently from memory into CPU cache lines, reducing cache misses and improving overall lookup performance.

### Hardware Prefetching

On x86_64 platforms, uses explicit prefetching to load data into cache before it's needed:

```rustit
unsafe {
    _mm_prefetch(
        &self.pointers[prefetch_idx] as *const _ as *const i8,
        _MM_HINT_T0,
    );
}
```

Helps hide memory latency by anticipating what data will be needed next during searches.

### Two-Level Index Structure

For larger collections, implement a hierarchical index structure with:

1. **Sparse Index**: A small top-level index that directs searches to the appropriate section
2. **Dense Index**: A more detailed bottom-level index with the actual pointers

This approach reduces memory usage while maintaining good lookup performance, especially for large datasets. The sparse ratio (number of dense entries per sparse entry) is configurable and can be tuned based on the collection size and access patterns.

```rust
pub struct TwoLevelFencePointers {
    pub sparse: SparseIndex,
    pub dense: DenseIndex,
    pub sparse_ratio: usize,
}
```

The structure is automatically rebuilt as needed and maintains optimal performance across a wide range of collection sizes.

### Prefix Compression

For maximizing memory efficiency, especially with numerical keys, the implementation includes bit-level prefix compression:

1. **Bit-Level Grouping**: Groups keys with common high-order bits to maximize sharing
2. **Suffix Storage**: Stores only the unique suffix bits for each key in a group
3. **Adaptive Optimization**: Dynamically adjusts compression based on key distribution

```rust
pub struct PrefixGroup {
    pub common_bits_mask: u64,
    pub num_shared_bits: u8,
    pub entries: Vec<(u64, u64, usize)>, // (min_key_suffix, max_key_suffix, block_index)
}

pub struct CompressedFencePointers {
    pub groups: Vec<PrefixGroup>,
    pub min_key: Key,
    pub max_key: Key,
    pub target_group_size: usize,
}
```

This compression approach is particularly effective for:
- Sequential keys (e.g., timestamps or auto-incremented IDs)
- Keys with natural grouping patterns (e.g., data from different sources with distinct prefixes)
- Large fence pointer collections where memory efficiency is critical

The implementation shows significant memory reduction compared to standard fence pointers:
- Up to 70% memory reduction with sequential keys
- 30-50% reduction with grouped keys
- 10-20% reduction even with high-entropy random keys

An adaptive version (`AdaptivePrefixFencePointers`) periodically optimizes the compression based on the observed key distribution, further improving memory efficiency for dynamic workloads.

### FastLane Memory Layout

For optimizing lookup performance, especially for frequently traversed fence pointers, the implementation includes a FastLane memory layout:

1. **Lane Separation**: Segregates comparison data from value data in separate memory regions
2. **Explicit Prefetching**: Uses hardware prefetching for upcoming entries during binary search
3. **Cache-Friendly Access**: Organizes memory layout for better CPU cache utilization

```rust
pub struct FastLaneGroup {
    pub common_bits_mask: u64,
    pub num_shared_bits: u8,
    pub min_key_lane: Vec<u64>,    // Lane for min_key values
    pub max_key_lane: Vec<u64>,    // Lane for max_key values
    pub block_index_lane: Vec<usize>, // Lane for block indices
}

pub struct FastLaneFencePointers {
    pub groups: Vec<FastLaneGroup>,
    pub min_key: Key,
    pub max_key: Key,
    pub target_group_size: usize,
}
```

#### Benchmark Results

The FastLane implementation shows the following performance characteristics:

**Sequential Keys**:
- 100% key coverage
- Memory usage: 21.24% less than standard implementation
- Performance: 44.12% slower than standard implementation

**Random Keys**:
- 100% key coverage
- Memory usage: 3.32% more than standard implementation
- Performance: 1308.48% slower than standard implementation

**Grouped Keys**:
- 100% key coverage
- Memory usage: 1.07% less than standard implementation
- Performance: 94.55% slower than standard implementation

While the FastLane implementation provides perfect key coverage and good memory characteristics, the current performance is not optimal due to:

1. **Basic Implementation**: This is an initial implementation that prioritizes correctness and demonstrating the memory layout concept
2. **Missing Optimizations**: More aggressive inlining and specialized binary search routines could improve performance
3. **Hardware Limitations**: Optimal performance depends heavily on CPU features and cache characteristics

Future optimizations should focus on:
- **Specialized Binary Search**: Custom binary search algorithm optimized for the lane structure
- **SIMD Vectorization**: Using vector instructions to process multiple keys at once
- **Memory Alignment**: Ensuring optimal alignment for CPU cache lines
- **Branch Prediction Hints**: Adding hints to help CPU predict branch directions

The data-oriented layout provides these benefits through:
1. **Separating Comparison Data**: Keeps min/max key values in separate lanes
2. **Explicit Prefetching**: Loads data into cache before it's needed during binary search
3. **Cache-Friendly Grouping**: Organizes related data in memory for better cache utilization
4. **Reduced Cache Misses**: Lane-based layout minimizes cache misses during traversal

This optimization combines the memory efficiency of prefix compression with the performance benefits of data-oriented design. The implementation includes an adaptive variant (`AdaptiveFastLaneFencePointers`) that dynamically adjusts itself based on access patterns.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.