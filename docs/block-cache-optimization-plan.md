# Block Cache Optimization Mini-Plan

This document outlines a plan for optimizing the LSM-tree block cache to state-of-the-art levels while ensuring each change is small, incremental, and well-tested.

## Current State Analysis

The codebase currently has two block cache implementations:

1. **Standard BlockCache**: Uses `RwLock<HashMap>` with a mutex-protected LRU queue
2. **LockFreeBlockCache**: Uses `SkipMap` from crossbeam with atomic operations

Both implementations support:
- LRU eviction policy
- TTL-based expiration
- Statistics tracking
- IO batching

## Optimization Targets

Based on SOTA research and block cache designs in systems like RocksDB, LevelDB, and WiredTiger, we'll focus on:

1. **Improved Cache Policies**: Beyond simple LRU/TTL
2. **Prefetching and Hot Key Prediction**: Anticipating reads for better performance
3. **Memory Overhead Reduction**: More efficient metadata
4. **Concurrent Performance**: Better scaling with thread count
5. **Admission Control**: Smarter decision-making for cache entries
6. **Multi-Tiered Cache**: Hierarchical cache organization
7. **Content-Aware Caching**: Adapting to access patterns and data types

## Implementation Plan

### Phase 1: Advanced Cache Policies (Days 1-3)

#### 1.1: Implement TinyLFU / Segmented LRU (Day 1)

TinyLFU combines frequency and recency information and outperforms LRU in most workloads.

**Tasks:**
- [ ] Create `TinyLFUPolicy` struct with count-min sketch implementation
- [ ] Implement admission policy with frequency-based decisions
- [ ] Add configuration options for TinyLFU parameters
- [ ] Write tests comparing hit rates with standard LRU

**Files to modify:**
- Create new file: `src/run/cache_policies/tiny_lfu.rs`
- Modify: `src/run/block_cache.rs` to integrate the policy

**Testing:**
- Compare hit rates across different workload patterns
- Verify thread safety and correct operation under concurrency

#### 1.2: Add Configurable Eviction Policies (Day 2)

**Tasks:**
- [ ] Create a `CachePolicy` trait for pluggable cache policies
- [ ] Refactor existing LRU as a policy implementation
- [ ] Add configuration option to select policy type
- [ ] Write tests for policy selection and switching

**Files to modify:**
- Create new file: `src/run/cache_policies/mod.rs`
- Create new file: `src/run/cache_policies/lru.rs`
- Modify: `src/run/block_cache.rs` to use policy trait

#### 1.3: Ghost Cache for ARC-like Behavior (Day 3)

ARC (Adaptive Replacement Cache) uses ghost caches to track recently evicted items and adjust allocation between recency and frequency.

**Tasks:**
- [ ] Implement ghost cache tracking for recently evicted items
- [ ] Add adaptive sizing between recency and frequency components
- [ ] Write tests comparing performance against TinyLFU and LRU
- [ ] Add metrics for adaptation effectiveness

**Files to modify:**
- Create new file: `src/run/cache_policies/arc.rs`
- Modify: `src/run/cache_policies/mod.rs`

### Phase 2: Intelligent Prefetching and Access Prediction (Days 4-6)

#### 2.1: Sequential Prefetching (Day 4)

**Tasks:**
- [ ] Implement sequential access detection
- [ ] Add configurable prefetch window size
- [ ] Integrate with block loading to fetch ahead
- [ ] Add prefetch statistics tracking

**Files to modify:**
- Create new file: `src/run/prefetcher.rs`
- Modify: `src/run/block_cache.rs`
- Modify: `src/run/storage.rs` to handle prefetch requests

**Testing:**
- Test with sequential workloads to verify performance improvement
- Test with random workloads to ensure no performance degradation

#### 2.2: Priority-Based Cache Entries (Day 5)

**Tasks:**
- [ ] Add priority levels to cache entries (prefetched vs. explicitly requested)
- [ ] Modify eviction policies to consider priority
- [ ] Add statistics for priority-based operations
- [ ] Write tests for priority-aware eviction

**Files to modify:**
- Modify: `src/run/block_cache.rs`
- Modify: `src/run/cache_policies/mod.rs`

#### 2.3: ML-Inspired Hot Key Prediction (Day 6)

Inspired by Leaper from pushing_sota.md, implement a lightweight predictor for hot keys.

**Tasks:**
- [ ] Create access pattern analyzer to identify hotspots
- [ ] Implement simple forecasting for key access frequency
- [ ] Add proactive prefetching based on predictions
- [ ] Write tests comparing prediction accuracy and hit rate improvements

**Files to modify:**
- Create new file: `src/run/access_predictor.rs`
- Modify: `src/run/prefetcher.rs` to use predictions

### Phase 3: Memory Efficiency and Multi-Tier Organization (Days 7-9)

#### 3.1: Compact Block Metadata (Day 7)

**Tasks:**
- [ ] Refactor `CacheEntry` to use less memory per entry
- [ ] Implement shared metadata structures for blocks from the same run
- [ ] Add memory usage statistics
- [ ] Write tests for memory efficiency

**Files to modify:**
- Modify: `src/run/block_cache.rs`
- Create new file: `src/run/shared_metadata.rs`

#### 3.2: Multi-Tiered Cache Architecture (Day 8)

**Tasks:**
- [ ] Create a two-tier cache structure (hot/cold)
- [ ] Implement promotion/demotion between tiers
- [ ] Add statistics for tier operations
- [ ] Write tests for tier transitions

**Files to modify:**
- Create new file: `src/run/tiered_cache.rs`
- Modify: `src/run/mod.rs` to use tiered cache

#### 3.3: Content-Aware Caching (Day 9)

**Tasks:**
- [ ] Add block content classification (small vs. large, dense vs. sparse)
- [ ] Implement content-aware admission and eviction policies
- [ ] Add customizable weights for different content types
- [ ] Write tests comparing content-aware vs. standard policies

**Files to modify:**
- Create new file: `src/run/content_classifier.rs`
- Modify: `src/run/cache_policies/mod.rs` to support content awareness
- Modify: `src/run/block_cache.rs` to track content types

### Phase 4: Advanced Concurrency and Optimizations (Days 10-12)

#### 4.1: Work-Stealing I/O Thread Pool (Day 10)

As suggested in pushing_sota.md, implement a work-stealing I/O thread pool.

**Tasks:**
- [ ] Implement work-stealing thread pool for cache operations
- [ ] Add priority queues for different operation types
- [ ] Write tests for concurrent performance under various loads

**Files to modify:**
- Create new file: `src/run/io_pool.rs`
- Modify: `src/run/block_cache.rs` to use the I/O pool

#### 4.2: Vector-Aware Caching (Day 11)

**Tasks:**
- [ ] Implement batch cache line operations with SIMD optimizations
- [ ] Add vectorized comparison for fast cache lookup
- [ ] Write tests comparing vectorized vs. standard lookup performance

**Files to modify:**
- Create new file: `src/run/vector_cache.rs`
- Modify: `src/run/block_cache.rs` to use vectorized operations where applicable

#### 4.3: Advanced Shard-Aware Operations (Day 12)

**Tasks:**
- [ ] Optimize the lock-free implementation with improved sharding strategies
- [ ] Implement cross-shard operations with minimal contention
- [ ] Add statistics for cross-shard operations
- [ ] Write tests for concurrent performance

**Files to modify:**
- Modify: `src/run/lock_free_block_cache.rs`
- Create new file: `src/run/shard_aware_ops.rs`

### Phase 5: Integration and Tuning (Days 13-14)

#### 5.1: Adaptive Configuration (Day 13)

**Tasks:**
- [ ] Create auto-tuning framework for cache parameters
- [ ] Implement workload-aware parameter adjustment
- [ ] Add performance monitoring for auto-tuning feedback
- [ ] Write tests for adaptation to changing workloads

**Files to modify:**
- Create new file: `src/run/cache_tuner.rs`
- Modify: `src/run/block_cache.rs` to support auto-tuning

#### 5.2: Integration and Benchmarking (Day 14)

**Tasks:**
- [ ] Finalize integration of all cache optimizations
- [ ] Create comprehensive benchmarking suite
- [ ] Document performance characteristics and trade-offs
- [ ] Write comparative tests against baseline implementation

**Files to modify:**
- Create new file: `benches/advanced_cache_bench.rs`
- Update documentation with performance results

## Testing Strategy

For each change:

1. **Unit Tests**:
   - Test each component in isolation
   - Verify correct behavior under error conditions
   - Test thread safety

2. **Benchmark Tests**:
   - Measure hit rate across different workloads
   - Measure throughput and latency
   - Compare memory usage

3. **Integration Tests**:
   - Test with full LSM-tree operations
   - Verify correct interaction with other components
   - Test recovery and persistence

## Key Performance Metrics

We'll track these metrics for each change:

1. **Hit Rate**: Percentage of requests served from cache
2. **Latency**: Time to retrieve blocks (average and percentiles)
3. **Throughput**: Blocks served per second under load
4. **Memory Efficiency**: Bytes of memory per cached block
5. **Concurrency**: Performance scaling with thread count

## Implementation Priority

If time constraints arise, implement in this order:

1. TinyLFU/Segmented LRU Policy (highest ROI)
2. Ghost Cache/ARC Implementation
3. Sequential Prefetching
4. Multi-Tiered Cache Architecture
5. Memory Efficiency Improvements
6. Content-Aware Caching
7. Vector-Aware Optimizations
8. Work-Stealing Thread Pool
9. Adaptive Configuration (lowest priority)

## Success Criteria

The optimized block cache should demonstrate:

1. 25%+ improvement in hit rate compared to basic LRU
2. Linear scaling to at least 8 concurrent threads
3. 50%+ reduction in cache miss penalty through prefetching
4. Less than 5% memory overhead per cached block
5. All tests pass consistently
6. Dynamically adapts to different workload patterns