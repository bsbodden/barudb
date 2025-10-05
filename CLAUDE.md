# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BarúDB**: A high-performance LSM tree key-value store written in Rust. Named after Volcán Barú, the highest peak in Panama. Originally developed as a research project for Harvard University's CS265 Data Systems course, implementing advanced LSM tree optimizations including Monkey-optimized Bloom filters, multiple compaction policies, and novel fence pointer designs.

**Rust Version**: 1.81.0 or later required

## Essential Commands

### Building
```bash
# Debug build
cargo build

# Release build (recommended for benchmarks and server)
cargo build --release
```

### Testing
```bash
# Run all tests (excludes long-running tests marked with #[ignore])
cargo test

# Run a specific test
cargo test test_name

# Run long-running workload tests
cargo test test_workload_execution -- --ignored

# Run server command tests (requires server running on port 8080)
./run_server_tests.sh

# Run all performance/ignored tests
cargo test -- --ignored
```

### Running Server and Client
```bash
# Start server (default port 8080)
cargo run --release --bin server

# Server with custom options
cargo run --release --bin server -- -p 8080 -f 4 -l tiered -t 4 -n 1024

# Start client
cargo run --release --bin client

# Client with options
cargo run --release --bin client -- -p 8080 -q
```

**Server Options:**
- `-e <error_rate>`: Bloom filter error rate (default: 0.01)
- `-n <num_pages>`: Buffer size in disk pages (default: 1024)
- `-f <fanout>`: LSM tree fanout (default: 4)
- `-l <policy>`: Compaction policy: tiered, leveled, lazy_leveled (default: tiered)
- `-t <threshold>`: Compaction threshold for runs (default: 4)
- `-p <port>`: Port number (default: 8080)

### Benchmarks
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench bloom_bench
cargo bench --bench storage_bench

# Simpler storage benchmark with clearer output
cargo run --release --bin storage_bench_simple

# FastLane fence pointer benchmark
cargo run --release --bin benchmark_fastlane
```

### Database Management
```bash
# Reset database to clean state
./reset_db.sh
```

## Architecture Overview

### Core Components

**LSM Tree Structure** (`src/lsm_tree.rs`):
- Main LSM tree implementation coordinating all components
- `LSMConfig`: Configuration for buffer size, storage type, fanout, compaction policy, compression
- Handles background compaction when enabled via `background_compaction` flag
- Supports both lock-free and standard memtable implementations via `use_lock_free_memtable` flag
- Supports both lock-free and standard block cache via `use_lock_free_block_cache` flag

**Storage Hierarchy**:
1. **Memtable** (`src/memtable.rs`, `src/lock_free_memtable.rs`): In-memory buffer for fast writes
   - Standard: Sharded memtable with RwLock-based concurrency
   - Lock-free: Crossbeam skiplist-based concurrent memtable (experimental)
2. **Levels** (`src/level.rs`): Organize runs by level with increasing size
3. **Runs** (`src/run/mod.rs`): Sorted immutable collections of key-value pairs on disk

### Run Module (`src/run/`)

The run module is the heart of disk storage, containing:

**Block Management**:
- `block.rs`: Block structure with configurable size
- `block_cache.rs`: Standard LRU/LFU block cache with eviction policies
- `lock_free_block_cache.rs`: Lock-free concurrent block cache (default)
- `cache_policies/`: LRU, LFU, and other eviction policies
- `lock_free_cache_policies/`: Lock-free versions of cache policies

**Compression** (`compression/`):
- `mod.rs`: Compression framework and factory
- `noop.rs`: No-op compression (default)
- `bit_pack.rs`: Bit packing compression
- `delta.rs`: Delta encoding
- `dictionary.rs`: Dictionary compression
- `lz4.rs`: LZ4 compression
- `snappy.rs`: Snappy compression
- Adaptive compression that selects best strategy based on data patterns

**Fence Pointers** (index structures for finding blocks):
- `standard_fence.rs`: Basic fence pointers with cache alignment
- `two_level_fence.rs`: Two-level sparse/dense indexing for large collections
- `compressed_fence.rs`: Prefix compression for numeric keys
- `eytzinger_layout.rs`: Modern high-performance Eytzinger (cache-oblivious) layout
- `adaptive_fastlane.rs`: Adaptive FastLane implementation
- Type aliases: `FastLaneFencePointers` = `EytzingerFencePointers` (current best)

**Storage Backends** (`storage.rs`, `lsf.rs`):
- `FileStorage`: Standard file-based storage
- `LSFStorage`: Log-structured file storage with write-ahead log and recovery
- `StorageFactory`: Creates storage instances based on `StorageType`
- All storage backends implement `RunStorage` trait

**Bloom Filters** (`filter.rs`):
- Monkey-optimized: Exponentially decreasing bits per entry by level
- Dynamic sizing support for adapting to workload
- Filter statistics tracking for false positive monitoring

### Compaction Policies (`src/compaction/`)

Pluggable compaction strategies implementing `CompactionPolicy` trait:

- `tiered.rs`: Multiple runs per level, compact when threshold reached (lower write amp, higher read amp)
- `leveled.rs`: Single run per level, compact on any new run (higher write amp, lower read amp)
- `lazy_leveled.rs`: Hybrid - tiered at level 0, leveled elsewhere (balanced)
- `partial_tiered.rs`: Partial tiered compaction optimization
- `mod.rs`: Factory pattern for creating policies

### Client-Server (`src/bin/`)

**Server** (`server.rs`):
- TCP server listening on configurable port (default 8080)
- Command parser delegates to LSM tree operations
- Background compaction thread if enabled
- Server commands: `bloom`, `quit`, `help`

**Client** (`client.rs`):
- Interactive REPL for database operations
- Supports: `p` (put), `g` (get), `r` (range), `d` (delete), `l` (load), `s` (stats), `f` (flush), `q` (quit)
- Load command: `l <filename>` processes batch commands from file

## Testing Structure

**Integration Tests** (`tests/`):
- `integration_test.rs`: Basic LSM tree operations
- `compaction_integration_test.rs`: Tests for all compaction policies
- `compression_test.rs`, `compression_comparison_test.rs`: Compression algorithm tests
- `recovery_reliability_test.rs`: Crash recovery and reliability tests
- `fence_pointer_bench_test.rs`, `fastlane_*_test.rs`: Fence pointer implementation tests
- `block_cache_test.rs`: Block cache and eviction policy tests
- `storage_comparison_test.rs`: Compare different storage backends
- `workload_test.rs`: Full workload simulation (marked with `#[ignore]`)

**Benchmarks** (`benches/`):
- `bloom_bench.rs`: Bloom filter performance (SpeedDB, RocksDB, etc.)
- `storage_bench.rs`: Criterion-based storage benchmarks
- `compression_bench.rs`: Compression algorithm benchmarks
- `block_cache_bench.rs`: Block cache performance
- Database comparison: `rocksdb_comparison`, `lmdb_comparison`, `leveldb_comparison`, `wiredtiger_comparison`

## Key Design Patterns

**Pluggable Architecture**:
- `CompactionPolicy` trait allows easy policy swapping
- `CompressionStrategy` trait for different compression algorithms
- `CachePolicy` trait for eviction strategies
- `RunStorage` trait for storage backends
- `FencePointersInterface` trait for index structures

**Configuration-Driven**:
- `LSMConfig` centralizes all configuration
- `CompressionConfig` and `AdaptiveCompressionConfig` for compression settings
- `BlockCacheConfig` for cache configuration
- Factory patterns (`CompactionFactory`, `StorageFactory`, `CompressionFactory`) create instances

**Concurrency**:
- Background compaction runs in separate thread
- Lock-free data structures: `LockFreeMemtable`, `LockFreeBlockCache`
- Standard sharded structures with RwLock for simpler cases
- Atomic statistics tracking throughout

**Recovery and Persistence**:
- LSF storage includes write-ahead log (WAL)
- Deterministic recovery sequence with flush points
- `sync_writes` flag controls synchronous vs async writes
- Recovery tests verify crash resilience

## Important Implementation Details

**Monkey Bloom Filter Optimization**:
- Bits per entry formula: `32.0 / T^(i/2)` where T=fanout, i=level
- Minimum 2 bits per entry at deepest levels
- ~94% memory reduction from level 0 to level 4 (with fanout=4)
- Dynamic resizing supported via `DynamicBloomFilterConfig`

**Fence Pointer Memory Layout**:
- `#[repr(align(16))]` on `FencePointer` for cache alignment
- Hardware prefetching on x86_64 using `_mm_prefetch`
- Eytzinger layout provides cache-oblivious binary search
- Prefix compression effective for sequential/grouped keys (up to 70% reduction)

**Storage Types**:
- File: Simple file-based storage
- LSF: Log-structured file with WAL and recovery
- MMap: Memory-mapped file storage (experimental)

**Compaction Selection**:
- Tiered: Best for write-heavy workloads
- Leveled: Best for read-heavy workloads
- Lazy Leveled: Best for mixed workloads

## Development Workflow

1. Make changes to source files
2. Run `cargo test` to verify basic functionality
3. Run specific test suites as needed
4. Use `./reset_db.sh` to clean database state between tests
5. Run benchmarks with `cargo bench` to measure performance impact
6. Test server/client interaction with `./run_server_tests.sh`

## Performance Considerations

- Always use `--release` flag for benchmarks and production server
- Block cache significantly improves read performance (enabled by default)
- Lock-free block cache is default and generally faster than standard cache
- Compression reduces disk I/O but adds CPU overhead
- Adaptive compression automatically selects best strategy
- Background compaction reduces latency spikes but uses extra CPU
- Fanout of 4 is generally optimal for balanced read/write performance
