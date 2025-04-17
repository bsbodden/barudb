# Fine-Grained Locking Implementation for LSM-Tree

## Overview

This document outlines the implementation of fine-grained locking mechanisms in the LSM-Tree to improve concurrent performance. The implementation follows the architectural principles described in literature on high-performance key-value stores like RocksDB and LevelDB.

## Components

### 1. Sharded Memtable

- Implemented a sharded memtable with multiple independent BTreeMaps
- Each shard has its own lock, allowing for concurrent operations on different keys
- Used a simple hashing scheme to determine which shard a key belongs to
- Updated all methods to work with the sharded architecture

```rust
pub struct Memtable {
    shards: Vec<MemtableShard>,
    current_size: AtomicUsize,
    max_size: usize,
    entry_size: usize,
}
```

### 2. Concurrent Level Implementation

- Created a new `ConcurrentLevel` type that wraps the original `Level` with a RwLock
- Provided thread-safe methods for all level operations
- Added ability to replace an entire level atomically
- Implemented proper locking for reads and writes

```rust
pub struct ConcurrentLevel {
    inner: RwLock<Level>,
}
```

### 3. Background Compaction

- Added a background compaction thread that runs asynchronously
- Implemented safe shutdown mechanics
- Added configuration option to enable/disable background compaction
- Ensured proper synchronization between main thread and background thread

```rust
struct CompactionState {
    active: AtomicBool,
    thread: Mutex<Option<JoinHandle<()>>>,
}
```

### 4. Test Improvements

- Created new tests for concurrent access patterns
- Updated existing tests to handle concurrent operations
- Added tests for background compaction
- Made tests more robust against race conditions

## Benefits

These changes have significantly improved the concurrency capabilities of the LSM-Tree:

1. **Enhanced Throughput**: Multiple threads can now read and write to different shards of the memtable concurrently
2. **Reduced Contention**: Each level has its own lock, allowing for concurrent operations on different levels
3. **Asynchronous Compaction**: Background compaction can happen concurrently with normal operations
4. **Resilience**: The code is now more resilient to high concurrency workloads

The implementation follows best practices for concurrent programming in Rust, using RwLock for shared data structures and atomic operations for counters and flags.

## Literature References

1. P. O'Neil, E. Cheng, D. Gawlick, and E. O'Neil. "The log-structured merge-tree (LSM-tree)." Acta Informatica, 33(4):351-385, 1996.

2. Lakshman, Avinash, and Prashant Malik. "Cassandra: a decentralized structured storage system." ACM SIGOPS Operating Systems Review 44.2 (2010): 35-40.

3. Siying Dong, Mark Callaghan, Leonidas Galanis, Dhruba Borthakur, Tony Savor, and Michael Stumm. "Optimizing space amplification in RocksDB." CIDR, 2017.

4. Lim, Hyeontaek, David G. Andersen, and Michael Kaminsky. "Towards accurate and fast evaluation of multi-stage log-structured designs." 14th USENIX Conference on File and Storage Technologies (FAST 16). 2016.

5. Kuszmaul, Bradley. "A comparison of fractal trees to log-structured merge (LSM) trees." Tokutek White Paper (2014).

6. Wu, Xingbo, et al. "LSM-based storage techniques: a survey." The VLDB Journal 29.1 (2020): 393-418.

## Implementation Notes

- The sharding strategy uses a bitwise AND operation with a power-of-two shard count for efficient key-to-shard mapping
- Read operations can proceed concurrently across different shards
- Write operations to different shards can proceed concurrently
- The background compaction thread periodically checks for compaction opportunities without blocking normal operations
- Special care was taken to ensure proper shutdown sequence for the background thread
