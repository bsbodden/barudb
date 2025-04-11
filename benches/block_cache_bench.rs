use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lsm_tree::run::{
    Block, BlockCache, BlockCacheConfig, BlockKey, Run, RunId
};
use std::time::Duration;

fn bench_block_cache(c: &mut Criterion) {
    // Print a header for better readability
    println!("\n==== Block Cache Benchmark ====");
    println!("Tests the performance improvement of using a block cache");
    
    let mut group = c.benchmark_group("block_cache_operations");
    
    // Create a block with test data
    let mut block = Block::new();
    for i in 0..1000 {
        block.add_entry(i, i * 10).unwrap();
    }
    block.seal().unwrap();
    
    // Create a cache with a small capacity
    let cache_config = BlockCacheConfig {
        max_capacity: 10,
        ttl: Duration::from_secs(60),
        cleanup_interval: Duration::from_secs(5),
    };
    let cache = BlockCache::new(cache_config);
    
    // Run ID for testing
    let run_id = RunId::new(0, 1);
    
    // Benchmark cache misses (first access)
    group.bench_function(BenchmarkId::new("cache", "miss"), |b| {
        b.iter(|| {
            // Create a new cache key every time to ensure it's a miss
            let key = BlockKey {
                run_id,
                block_idx: black_box(rand::random::<usize>() % 100),
            };
            cache.get(&key);
        })
    });
    
    // Insert a block into the cache
    let block_key = BlockKey {
        run_id,
        block_idx: 0,
    };
    cache.insert(block_key, block.clone()).unwrap();
    
    // Benchmark cache hits
    group.bench_function(BenchmarkId::new("cache", "hit"), |b| {
        b.iter(|| {
            cache.get(&block_key);
        })
    });
    
    // Benchmark direct block access (without cache)
    group.bench_function(BenchmarkId::new("block", "direct"), |b| {
        b.iter(|| {
            let key = black_box(500);
            block.get(&key);
        })
    });
    
    // Benchmark cached block access
    group.bench_function(BenchmarkId::new("block", "cached"), |b| {
        b.iter(|| {
            let cache_key = BlockKey {
                run_id,
                block_idx: 0,
            };
            if let Some(cached_block) = cache.get(&cache_key) {
                let key = black_box(500);
                cached_block.get(&key);
            }
        })
    });
    
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05)
        .without_plots();
    targets = bench_block_cache
}

criterion_main!(benches);