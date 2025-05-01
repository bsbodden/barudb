use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lsm_tree::run::{cache_policies::CachePolicyType, block_cache::{BlockCache, BlockCacheConfig}, 
                   Block, RunId, lock_free_block_cache::{LockFreeBlockCache, LockFreeBlockCacheConfig}};
use lsm_tree::run::lock_free_cache_policies::CachePriority;
use std::sync::Arc;
use std::time::Duration;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

// Helper functions for benchmark
fn create_test_block(key: i64, value: i64) -> Block {
    let mut block = Block::new();
    block.add_entry(key, value).unwrap();
    block.seal().unwrap();
    block
}

// Create both types of BlockKey
fn create_std_block_key(run_id_i: usize, block_idx: usize) -> lsm_tree::run::block_cache::BlockKey {
    lsm_tree::run::block_cache::BlockKey {
        run_id: RunId::new(0, run_id_i as u64),
        block_idx,
    }
}

fn create_lock_free_block_key(run_id_i: usize, block_idx: usize) -> lsm_tree::run::lock_free_block_cache::BlockKey {
    lsm_tree::run::lock_free_block_cache::BlockKey {
        run_id: RunId::new(0, run_id_i as u64),
        block_idx,
    }
}

// Test uniform access with LRU policy
fn bench_lru_uniform_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_cache_lru_uniform");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    
    let cache_sizes = [1000, 10000];
    
    for &size in &cache_sizes {
        let num_blocks = size * 2; // Double the cache size to ensure evictions
        
        // Standard cache
        let config = BlockCacheConfig {
            max_capacity: size,
            ttl: Duration::from_secs(3600),
            cleanup_interval: Duration::from_secs(60),
            policy_type: CachePolicyType::LRU,
        };
        
        let id = format!("standard_cache_{}", size);
        group.bench_with_input(
            BenchmarkId::new("get_hit_ratio", id), 
            &(size, num_blocks), 
            |b, &(capacity, num_blocks)| {
                let cache = BlockCache::new(config.clone());
                
                // Insert all blocks
                for i in 0..num_blocks {
                    let key = create_std_block_key(1, i);
                    let block = create_test_block(i as i64, (i * 10) as i64);
                    cache.insert(key, block).unwrap();
                }
                
                b.iter(|| {
                    // Simulate random uniform access to blocks
                    let mut hits = 0;
                    for i in 0..1000 {
                        let idx = i % num_blocks;
                        let key = create_std_block_key(1, idx);
                        if cache.get(&key).is_some() {
                            hits += 1;
                        }
                    }
                    black_box(hits)
                });
            },
        );
        
        // Lock-free cache
        let lock_free_config = LockFreeBlockCacheConfig {
            max_capacity: size,
            ttl: Duration::from_secs(3600),
            cleanup_interval: Duration::from_secs(60),
            policy_type: CachePolicyType::LRU,
        };
        
        let id = format!("lock_free_cache_{}", size);
        group.bench_with_input(
            BenchmarkId::new("get_hit_ratio", id), 
            &(size, num_blocks), 
            |b, &(capacity, num_blocks)| {
                let cache = LockFreeBlockCache::new(lock_free_config.clone());
                
                // Insert all blocks
                for i in 0..num_blocks {
                    let key = create_lock_free_block_key(1, i);
                    let block = create_test_block(i as i64, (i * 10) as i64);
                    cache.insert(key, block).unwrap();
                }
                
                b.iter(|| {
                    // Simulate random uniform access to blocks
                    let mut hits = 0;
                    for i in 0..1000 {
                        let idx = i % num_blocks;
                        let key = create_lock_free_block_key(1, idx);
                        if cache.get(&key).is_some() {
                            hits += 1;
                        }
                    }
                    black_box(hits)
                });
            },
        );
    }
    
    group.finish();
}

// Test with TinyLFU policy
fn bench_tinylfu_uniform_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_cache_tinylfu_uniform");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    
    let cache_sizes = [1000, 10000];
    
    for &size in &cache_sizes {
        let num_blocks = size * 2; // Double the cache size to ensure evictions
        
        // Standard cache with TinyLFU
        let config = BlockCacheConfig {
            max_capacity: size,
            ttl: Duration::from_secs(3600),
            cleanup_interval: Duration::from_secs(60),
            policy_type: CachePolicyType::TinyLFU,
        };
        
        let id = format!("standard_cache_{}", size);
        group.bench_with_input(
            BenchmarkId::new("get_hit_ratio", id), 
            &(size, num_blocks), 
            |b, &(capacity, num_blocks)| {
                let cache = BlockCache::new(config.clone());
                
                // Insert all blocks
                for i in 0..num_blocks {
                    let key = create_std_block_key(1, i);
                    let block = create_test_block(i as i64, (i * 10) as i64);
                    cache.insert(key, block).unwrap();
                }
                
                b.iter(|| {
                    // Simulate random uniform access to blocks
                    let mut hits = 0;
                    for i in 0..1000 {
                        // Use modulo to create skewed access pattern - lower indices more frequent
                        let idx = i % (num_blocks / 2) + (i % 10);
                        let key = create_std_block_key(1, idx % num_blocks);
                        if cache.get(&key).is_some() {
                            hits += 1;
                        }
                    }
                    black_box(hits)
                });
            },
        );
        
        // Lock-free cache with TinyLFU
        let lock_free_config = LockFreeBlockCacheConfig {
            max_capacity: size,
            ttl: Duration::from_secs(3600),
            cleanup_interval: Duration::from_secs(60),
            policy_type: CachePolicyType::TinyLFU,
        };
        
        let id = format!("lock_free_cache_{}", size);
        group.bench_with_input(
            BenchmarkId::new("get_hit_ratio", id), 
            &(size, num_blocks), 
            |b, &(capacity, num_blocks)| {
                let cache = LockFreeBlockCache::new(lock_free_config.clone());
                
                // Insert all blocks
                for i in 0..num_blocks {
                    let key = create_lock_free_block_key(1, i);
                    let block = create_test_block(i as i64, (i * 10) as i64);
                    cache.insert(key, block).unwrap();
                }
                
                b.iter(|| {
                    // Simulate random uniform access to blocks
                    let mut hits = 0;
                    for i in 0..1000 {
                        // Use modulo to create skewed access pattern - lower indices more frequent
                        let idx = i % (num_blocks / 2) + (i % 10);
                        let key = create_lock_free_block_key(1, idx % num_blocks);
                        if cache.get(&key).is_some() {
                            hits += 1;
                        }
                    }
                    black_box(hits)
                });
            },
        );
    }
    
    group.finish();
}

// Test TTL expiration
fn bench_ttl_expiration(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_cache_ttl");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    
    let cache_size = 5000;
    let num_blocks = cache_size;
    
    // Test with TinyLFUWithTTL
    let config = LockFreeBlockCacheConfig {
        max_capacity: cache_size,
        ttl: Duration::from_millis(50), // Very short TTL for testing
        cleanup_interval: Duration::from_millis(10),
        policy_type: CachePolicyType::TinyLFUWithTTL,
    };
    
    group.bench_with_input(
        BenchmarkId::new("ttl_cleanup_time", "tinylfu_ttl"),
        &(cache_size, num_blocks),
        |b, &(capacity, num_blocks)| {
            let cache = LockFreeBlockCache::new(config.clone());
            
            // Insert all blocks
            for i in 0..num_blocks {
                let key = create_lock_free_block_key(1, i);
                let block = create_test_block(i as i64, (i * 10) as i64);
                cache.insert(key, block).unwrap();
            }
            
            // Wait for TTL to expire
            thread::sleep(Duration::from_millis(60));
            
            b.iter(|| {
                // Manually trigger cleanup
                cache.force_cleanup();
                
                // Check how many items remain in the cache
                let mut remaining = 0;
                for i in 0..num_blocks {
                    let key = create_lock_free_block_key(1, i);
                    if cache.get(&key).is_some() {
                        remaining += 1;
                    }
                }
                black_box(remaining)
            });
        },
    );
    
    // Test with PriorityLFU
    let config = LockFreeBlockCacheConfig {
        max_capacity: cache_size,
        ttl: Duration::from_millis(50), // Very short TTL for testing
        cleanup_interval: Duration::from_millis(10),
        policy_type: CachePolicyType::PriorityLFU,
    };
    
    group.bench_with_input(
        BenchmarkId::new("ttl_cleanup_time", "priority_lfu"),
        &(cache_size, num_blocks),
        |b, &(capacity, num_blocks)| {
            let cache = LockFreeBlockCache::new(config.clone());
            
            // Insert all blocks
            for i in 0..num_blocks {
                let key = create_lock_free_block_key(1, i);
                let block = create_test_block(i as i64, (i * 10) as i64);
                cache.insert(key, block).unwrap();
            }
            
            // Wait for TTL to expire
            thread::sleep(Duration::from_millis(60));
            
            b.iter(|| {
                // Manually trigger cleanup
                cache.force_cleanup();
                
                // Check how many items remain in the cache
                let mut remaining = 0;
                for i in 0..num_blocks {
                    let key = create_lock_free_block_key(1, i);
                    if cache.get(&key).is_some() {
                        remaining += 1;
                    }
                }
                black_box(remaining)
            });
        },
    );
    
    group.finish();
}

// Test priority-based eviction
fn bench_priority_based_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_cache_priority");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    
    let cache_size = 1000;
    let num_blocks = cache_size * 2; // Double the cache size to ensure evictions
    
    let config = LockFreeBlockCacheConfig {
        max_capacity: cache_size,
        ttl: Duration::from_secs(3600),
        cleanup_interval: Duration::from_secs(60),
        policy_type: CachePolicyType::PriorityLFU,
    };
    
    group.bench_with_input(
        BenchmarkId::new("priority_eviction", "retention_by_priority"),
        &(cache_size, num_blocks),
        |b, &(capacity, num_blocks)| {
            let cache = LockFreeBlockCache::new(config.clone());
            
            // Insert initial blocks with varying priorities
            for i in 0..num_blocks/2 {
                let key = create_lock_free_block_key(1, i);
                let block = create_test_block(i as i64, (i * 10) as i64);
                cache.insert(key, block).unwrap();
                
                // Assign different priorities by modulo
                let priority = match i % 4 {
                    0 => CachePriority::Critical,  // 25% critical
                    1 => CachePriority::High,      // 25% high
                    2 => CachePriority::Normal,    // 25% normal
                    _ => CachePriority::Low,       // 25% low
                };
                
                cache.set_priority(&key, priority);
            }
            
            b.iter(|| {
                // Insert more items to trigger eviction
                let mut priorities_after_eviction = [0; 4]; // Count by priority
                
                // Add more items to cause eviction
                for i in num_blocks/2..num_blocks {
                    let key = create_lock_free_block_key(1, i);
                    let block = create_test_block(i as i64, (i * 10) as i64);
                    cache.insert(key, block).unwrap();
                    
                    // New items get Normal priority
                    cache.set_priority(&key, CachePriority::Normal);
                }
                
                // Count how many of each priority are left
                for i in 0..num_blocks/2 {
                    let key = create_lock_free_block_key(1, i);
                    
                    if let Some(priority) = cache.get_priority(&key) {
                        // Increment the appropriate counter based on priority
                        match priority {
                            CachePriority::Critical => priorities_after_eviction[0] += 1,
                            CachePriority::High => priorities_after_eviction[1] += 1,
                            CachePriority::Normal => priorities_after_eviction[2] += 1,
                            CachePriority::Low => priorities_after_eviction[3] += 1,
                        }
                    }
                }
                
                black_box(priorities_after_eviction)
            });
        },
    );
    
    group.finish();
}

// Test concurrent access
fn bench_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_cache_concurrent");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    
    let cache_size = 10000;
    let num_blocks = cache_size;
    let thread_count = 4;
    
    // Lock-free cache with PriorityLFU
    let config = LockFreeBlockCacheConfig {
        max_capacity: cache_size,
        ttl: Duration::from_secs(3600),
        cleanup_interval: Duration::from_secs(60),
        policy_type: CachePolicyType::PriorityLFU,
    };
    
    group.bench_with_input(
        BenchmarkId::new("concurrent_hit_rate", format!("priority_lfu_{}_threads", thread_count)),
        &(cache_size, num_blocks, thread_count),
        |b, &(capacity, num_blocks, threads)| {
            let cache = Arc::new(LockFreeBlockCache::new(config.clone()));
            
            // Insert all blocks
            for i in 0..num_blocks {
                let key = create_lock_free_block_key(1, i);
                let block = create_test_block(i as i64, (i * 10) as i64);
                cache.insert(key, block).unwrap();
            }
            
            b.iter(|| {
                let hit_counter = Arc::new(AtomicUsize::new(0));
                let total_counter = Arc::new(AtomicUsize::new(0));
                
                let mut handles = vec![];
                for thread_id in 0..threads {
                    let thread_cache = Arc::clone(&cache);
                    let thread_hits = Arc::clone(&hit_counter);
                    let thread_total = Arc::clone(&total_counter);
                    
                    let handle = thread::spawn(move || {
                        let ops_per_thread = 1000 / threads;
                        let mut local_hits = 0;
                        let mut local_ops = 0;
                        
                        for i in 0..ops_per_thread {
                            // Skewed access pattern based on thread ID
                            let is_write = (i + thread_id) % 5 == 0; // 20% writes
                            
                            if is_write {
                                // Write operation
                                let idx = (i * thread_id) % num_blocks;
                                let key = create_lock_free_block_key(1, idx);
                                let block = create_test_block(idx as i64, (idx * 10) as i64);
                                thread_cache.insert(key, block).unwrap();
                            } else {
                                // Read operation
                                let idx = (i * (thread_id + 1)) % num_blocks;
                                let key = create_lock_free_block_key(1, idx);
                                if thread_cache.get(&key).is_some() {
                                    local_hits += 1;
                                }
                            }
                            local_ops += 1;
                        }
                        
                        thread_hits.fetch_add(local_hits, Ordering::Relaxed);
                        thread_total.fetch_add(local_ops, Ordering::Relaxed);
                    });
                    
                    handles.push(handle);
                }
                
                // Wait for all threads to complete
                for handle in handles {
                    handle.join().unwrap();
                }
                
                let hit_rate = hit_counter.load(Ordering::Relaxed) as f64 / 
                               total_counter.load(Ordering::Relaxed) as f64;
                               
                black_box(hit_rate)
            });
        },
    );
    
    // Lock-free cache with TinyLFUWithTTL
    let config = LockFreeBlockCacheConfig {
        max_capacity: cache_size,
        ttl: Duration::from_secs(3600),
        cleanup_interval: Duration::from_secs(60),
        policy_type: CachePolicyType::TinyLFUWithTTL,
    };
    
    group.bench_with_input(
        BenchmarkId::new("concurrent_hit_rate", format!("tinylfu_ttl_{}_threads", thread_count)),
        &(cache_size, num_blocks, thread_count),
        |b, &(capacity, num_blocks, threads)| {
            let cache = Arc::new(LockFreeBlockCache::new(config.clone()));
            
            // Insert all blocks
            for i in 0..num_blocks {
                let key = create_lock_free_block_key(1, i);
                let block = create_test_block(i as i64, (i * 10) as i64);
                cache.insert(key, block).unwrap();
            }
            
            b.iter(|| {
                let hit_counter = Arc::new(AtomicUsize::new(0));
                let total_counter = Arc::new(AtomicUsize::new(0));
                
                let mut handles = vec![];
                for thread_id in 0..threads {
                    let thread_cache = Arc::clone(&cache);
                    let thread_hits = Arc::clone(&hit_counter);
                    let thread_total = Arc::clone(&total_counter);
                    
                    let handle = thread::spawn(move || {
                        let ops_per_thread = 1000 / threads;
                        let mut local_hits = 0;
                        let mut local_ops = 0;
                        
                        for i in 0..ops_per_thread {
                            // Skewed access pattern based on thread ID
                            let is_write = (i + thread_id) % 5 == 0; // 20% writes
                            
                            if is_write {
                                // Write operation
                                let idx = (i * thread_id) % num_blocks;
                                let key = create_lock_free_block_key(1, idx);
                                let block = create_test_block(idx as i64, (idx * 10) as i64);
                                thread_cache.insert(key, block).unwrap();
                            } else {
                                // Read operation
                                let idx = (i * (thread_id + 1)) % num_blocks;
                                let key = create_lock_free_block_key(1, idx);
                                if thread_cache.get(&key).is_some() {
                                    local_hits += 1;
                                }
                            }
                            local_ops += 1;
                        }
                        
                        thread_hits.fetch_add(local_hits, Ordering::Relaxed);
                        thread_total.fetch_add(local_ops, Ordering::Relaxed);
                    });
                    
                    handles.push(handle);
                }
                
                // Wait for all threads to complete
                for handle in handles {
                    handle.join().unwrap();
                }
                
                let hit_rate = hit_counter.load(Ordering::Relaxed) as f64 / 
                               total_counter.load(Ordering::Relaxed) as f64;
                               
                black_box(hit_rate)
            });
        },
    );
    
    group.finish();
}

criterion_group!(
    benches, 
    bench_ttl_expiration,
    bench_priority_based_eviction
);
criterion_main!(benches);