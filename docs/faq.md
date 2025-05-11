# Log-Structured Merge (LSM) Tree FAQ

This document addresses common questions about our LSM tree implementation, covering design decisions, architecture, performance considerations, and specific implementation details.

## General Questions

### What is this project?

This is an implementation of a Log-Structured Merge (LSM) tree-based key-value store written in Rust. It provides efficient storage with optimized write throughput, balanced read performance, and space efficiency. The implementation includes several advanced features like specialized Bloom filters, multiple compaction strategies, and novel fence pointer designs for efficient range queries.

### What are the main design goals?

The primary design goals were:

1. High write throughput (1M+ operations per second)
2. Reasonable read performance (50K+ operations per second)
3. Efficient range queries
4. Memory safety and concurrency
5. Efficient space utilization
6. Compatibility with modern hardware

### Why use Rust for this implementation?

Rust was chosen because it offers:

1. Memory safety without garbage collection
2. Thread safety through its ownership model
3. Zero-cost abstractions
4. High performance comparable to C/C++
5. Modern language features like traits and pattern matching
6. Growing ecosystem for systems programming

### How does this implementation compare to other LSM tree-based stores?

When compared to state-of-the-art systems like RocksDB, LevelDB, and others, this implementation:

1. Shows competitive write performance, exceeding 1M operations per second
2. Offers good read performance with 50K+ operations per second
3. Demonstrates excellent scalability with near-linear scaling up to 8 cores
4. Provides a more memory-safe architecture with fine-grained concurrency control
5. Implements novel optimizations like Eytzinger layout for fence pointers

## Architecture Questions

### What are the key components of the LSM tree implementation?

The key components include:

1. **Memtable**: In-memory buffer for writes (regular or lock-free implementation)
2. **Runs**: Immutable sorted data files stored on disk
3. **Levels**: Organized hierarchy of runs
4. **Bloom Filters**: Probabilistic data structures to quickly check key presence
5. **Fence Pointers**: Index structures to improve range query performance
6. **Block Cache**: Memory cache for frequently accessed blocks
7. **Compaction Policies**: Strategies for merging runs and levels

### How is the data organized in the LSM tree?

The data is organized in a hierarchical structure:

1. **Memtable**: All writes go to an in-memory buffer first
2. **Level 0**: Memtable is flushed to Level 0 when full
3. **Lower Levels**: Data is compacted into increasingly larger levels
4. Each level contains one or more runs (sorted files) depending on the compaction policy
5. Each run contains sorted key-value pairs organized into blocks

### How are concurrent operations handled?

Concurrency is handled through multiple mechanisms:

1. **Fine-grained locking**: Different components have their own locks
2. **Sharded structures**: Memory structures are divided into multiple shards
3. **Immutability**: Runs are immutable after creation, allowing lock-free reads
4. **Lock-free data structures**: Optional lock-free memtable and block cache
5. **RwLock**: Used to allow multiple readers with exclusive writer access
6. **Atomic operations**: Used for counters and flags

### What storage format is used for persistence?

The system uses a custom binary format optimized for both space efficiency and quick access:

1. Run files contain a header with metadata, bloom filters, and fence pointers
2. Data is organized into fixed-size blocks with headers
3. Each block contains sorted key-value pairs
4. Checksums are used to ensure data integrity
5. Optional compression is applied to blocks based on level and data characteristics

## Implementation Details

### What Bloom filter optimizations are implemented?

The Bloom filter implementation includes several optimizations:

1. **Double probing**: Each hash function sets two bits, reducing the number of hash functions needed
2. **Cache-aligned storage**: Bit arrays aligned to cache lines for improved access
3. **SIMD operations**: Vector instructions for batch operations when available
4. **Prefetching**: Memory prefetching to reduce cache miss penalties
5. **Dynamic sizing**: Monkey algorithm to allocate bits across levels optimally

### What are fence pointers and why are they important?

Fence pointers are index structures that improve range query performance:

1. They store min/max keys for each block to skip blocks not containing target keys
2. This implementation offers different fence pointer variants:
   - **Standard fence pointers**: Basic min/max key indexes
   - **FastLane fence pointers**: Multi-level skip lists for faster binary search
   - **Eytzinger layout**: Cache-optimized memory layout for binary search
   - **Compressed fence pointers**: Prefix compression for memory efficiency

### What compaction strategies are available?

The system supports multiple compaction strategies:

1. **Tiered compaction**: Allows multiple runs per level, merging when threshold reached
2. **Leveled compaction**: Maintains a single run per level for optimal read performance
3. **Lazy-leveled compaction**: Hybrid approach with tiered for Level 0, leveled for others
4. **Partial compaction**: Selectively compacts subsets of runs or key ranges

### How are recovery and crash resilience handled?

Recovery mechanisms include:

1. **Manifest files**: Track the LSM tree structure
2. **Metadata files**: Store information about each run
3. **Atomic operations**: File renames for atomic updates
4. **Write-ahead logging**: Records operations not yet flushed to disk
5. **Checksums**: Validate data integrity during recovery

### What compression techniques are supported?

Multiple compression algorithms are available and selected adaptively:

1. **LZ4**: Fast compression/decompression for hot data (Level 0)
2. **Snappy**: Balanced speed and compression for middle levels
3. **Zstd**: Better compression for cold data (deeper levels)
4. **Bit-packing**: Custom compression for numeric data
5. **Delta encoding**: Efficient compression for sequential keys

### How is memory management handled?

Memory is carefully managed through several techniques:

1. **Block caching**: LRU, LFU, or TinyLFU policies for block retention
2. **Memory-mapped files**: Using `mmap` for efficient random access
3. **Bloom filter sizing**: Optimal bit allocation based on level access patterns
4. **Fence pointer optimizations**: Memory-efficient index structures
5. **Rust's ownership model**: Ensures memory safety without garbage collection

## Performance Questions

### What are the performance characteristics for writes?

Write performance features:

1. Exceeds 1M operations per second on modern hardware
2. Batch operations for higher throughput
3. Efficiently handles sequential and random writes
4. Write amplification controlled through compaction policy choice
5. Near-linear scaling with more CPU cores

### What are the performance characteristics for reads?

Read performance features:

1. 50K+ operations per second for point queries
2. Optimized range query performance through fence pointers
3. Bloom filters reduce unnecessary disk I/O
4. Block cache improves performance for repeated access patterns
5. Read amplification bounded through level design

### How does the system scale with data size?

Scaling characteristics:

1. Write performance remains relatively stable as data size grows
2. Read performance degrades logarithmically with data size
3. Space efficiency improves with larger datasets
4. Compaction overhead increases with data size but is controlled through policies
5. Memory usage can be tuned for different hardware profiles

### What hardware configurations are recommended?

Recommended hardware:

1. Modern multi-core CPU (4+ cores recommended)
2. 16GB+ RAM for optimal performance
3. SSD storage for best latency characteristics
4. Systems with high I/O bandwidth benefit the most
5. CPU with good single-thread performance and SIMD support

## Tuning and Configuration

### What are the key configuration parameters?

Important configuration parameters:

1. **Memtable size**: Controls how frequently flushes occur
2. **Compaction policy**: Affects write amplification and read performance
3. **Bloom filter bits per key**: Trades memory for fewer disk reads
4. **Block size**: Affects I/O efficiency and cache utilization
5. **Cache size**: Controls memory usage vs. disk access frequency

### How should I tune for write-heavy workloads?

For write-heavy workloads:

1. Use **Tiered** or **Lazy-Leveled** compaction
2. Increase memtable size to reduce flush frequency
3. Consider the lock-free memtable implementation
4. Reduce Bloom filter size on deeper levels
5. Enable background compaction to prevent stalls

### How should I tune for read-heavy workloads?

For read-heavy workloads:

1. Use **Leveled** compaction for best read performance
2. Increase block cache size
3. Allocate more bits to Bloom filters
4. Consider Eytzinger fence pointers for range queries
5. Ensure compression is enabled for deeper levels

### How should I tune for mixed workloads?

For balanced workloads:

1. Use **Lazy-Leveled** compaction
2. Balance memtable size for moderate flush frequency
3. Ensure adequate block cache size
4. Maintain moderate Bloom filter density
5. Enable adaptive compression

### How should I tune for range query workloads?

For range query workloads:

1. Use **Leveled** compaction
2. Enable Eytzinger or FastLane fence pointers
3. Consider larger block sizes
4. Allocate more memory to block cache
5. Enable prefix compression for fence pointers

## Module-Specific Questions

### How does the bloom filter implementation work?

The Bloom filter (in `src/bloom/mod.rs`) is highly optimized:

1. Uses atomic operations for thread safety
2. Implements double probing (each hash function sets two bits)
3. Uses SIMD operations for batch processing
4. Handles multiple probe counts efficiently
5. Features prefetching for improved cache behavior

### How does the cache implementation work?

The block cache (`src/run/block_cache.rs` and `src/run/lock_free_block_cache.rs`):

1. Stores frequently accessed blocks in memory
2. Supports multiple eviction policies (LRU, LFU, TinyLFU)
3. Offers both traditional and lock-free implementations
4. Uses sharding to reduce contention
5. Monitors hit/miss statistics for optimization

### How do the compaction policies differ?

Compaction policies (`src/compaction/`) differ in:

1. **Tiered** (`tiered.rs`): Balances write performance with moderate reads
2. **Leveled** (`leveled.rs`): Optimizes read performance at cost of write amplification
3. **Lazy-Leveled** (`lazy_leveled.rs`): Hybrid approach balancing reads and writes
4. **Partial** (`partial_tiered.rs`): Selective compaction for reduced write amplification

### How do the memtable implementations differ?

Two memtable implementations are available:

1. **Sharded BTreeMap** (`src/memtable.rs`): Uses RwLocks for thread safety
2. **Lock-Free** (`src/lock_free_memtable.rs`): Uses lock-free data structures for concurrent access

Both implementations support:

- Key-value insertion and retrieval
- Range queries
- Consistent hashing for shard distribution
- Atomic size tracking

### How do fence pointers improve range query performance?

Fence pointers (`src/run/fence.rs`, `src/run/standard_fence.rs`, etc.) improve range queries by:

1. Storing min/max keys for each block
2. Allowing the system to skip blocks not containing target keys
3. Organizing index data for efficient binary search
4. Using specialized memory layouts for cache efficiency
5. Supporting compression for memory efficiency

## Advanced Features

### What is the Eytzinger layout for fence pointers?

The Eytzinger layout (`src/run/eytzinger_layout.rs`):

1. Reorganizes fence pointer entries using a breadth-first traversal of the binary search tree
2. Places elements in memory in the order they would be accessed during binary search
3. Significantly improves cache locality during binary search
4. Reduces cache misses by 30-70% compared to standard layout
5. Shows 2-2.5x faster range queries in benchmarks

### What is the lock-free block cache?

The lock-free block cache (`src/run/lock_free_block_cache.rs`):

1. Uses atomic operations instead of locks
2. Significantly reduces contention under high concurrency
3. Shows 2-5x better throughput than traditional cache under load
4. Supports advanced policies like TinyLFU with TTL
5. Maintains thread safety without locks

### How does the dynamic Bloom filter sizing work?

Dynamic Bloom filter sizing (based on the Monkey algorithm):

1. Allocates fewer bits to deeper levels that are accessed less frequently
2. Follows theoretical optimal bit distribution
3. Adjusts false positive rates based on access patterns
4. Monitors runtime performance for adaptation
5. Significantly reduces memory usage while maintaining performance

### What is the prefetching mechanism in Bloom filters?

The Bloom filter prefetching:

1. Issues CPU prefetch instructions for hash probe locations
2. Reduces cache miss penalties during lookup
3. Processes batches of keys with optimal memory access patterns
4. Provides 2-3x performance improvement for batch operations
5. Uses platform-specific optimizations when available

### How does the SIMD optimization in Bloom filters work?

SIMD (Single Instruction Multiple Data) optimization:

1. Uses wide (vector) instructions to process multiple hashes simultaneously
2. Leverages the `wide` crate for portable SIMD operations
3. Implements specialized paths for different probe counts
4. Reduces CPU cycles per key considerably
5. Maintains compatibility across different platforms

## Development and Testing

### How is the project tested?

The project has a comprehensive test suite including:

1. Unit tests for individual components
2. Integration tests for combined functionality
3. Performance benchmarks for optimization validation
4. Recovery and crash tests for reliability verification
5. Concurrent access tests for thread safety validation

### How can I run the benchmarks?

Benchmarks can be run using:

1. `cargo bench` for general performance tests
2. Specialized bench binaries in `src/bin/` for focused tests
3. The `benches/` directory contains comparison benchmarks
4. Parameters can be adjusted in benchmark code for different scenarios
5. Results are output in CSV format for analysis

### How can I contribute to the project?

To contribute:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass with `cargo test`
5. Submit a pull request with a clear description of changes

### Where can I find more documentation?

Additional documentation is available in:

1. The `docs/` directory for design docs and detailed explanations
2. Code comments throughout the source
3. The final report contains comprehensive architecture and performance analysis
4. Benchmark results in the `benches_history/` directory
5. The `final_report/` directory contains diagrams and additional documentation

### What future improvements are planned?

Planned improvements include:

1. Adaptive compaction policies that adjust based on workload
2. SIMD optimizations for more components
3. Custom file formats optimized for modern SSDs
4. Integration with persistent memory technologies
5. Distributed operation support
