use lsm_tree::run::{
    Block, BlockCache, BlockCacheConfig, BlockKey, 
    FileStorage, Run, RunId, RunStorage, StorageOptions
};
use std::time::Instant;
use tempfile::tempdir;

#[test]
fn test_block_cache_basic_operations() {
    // Create a simple block cache with small capacity
    let config = BlockCacheConfig {
        max_capacity: 5,
        ttl: std::time::Duration::from_secs(60),
        ..Default::default()
    };
    
    let cache = BlockCache::new(config);
    
    // Create some test blocks
    let mut blocks = Vec::new();
    for i in 0..10 {
        let mut block = Block::new();
        block.add_entry(i, i * 10).unwrap();
        block.seal().unwrap();
        blocks.push(block);
    }
    
    // Create keys for the blocks
    let run_id = RunId::new(0, 1);
    let keys: Vec<_> = (0..10).map(|i| BlockKey { run_id, block_idx: i }).collect();
    
    // Insert blocks into cache
    for (i, block) in blocks.iter().take(5).enumerate() {
        cache.insert(keys[i], block.clone()).unwrap();
    }
    
    // Verify cache contains blocks 0-4
    for i in 0..5 {
        assert!(cache.get(&keys[i]).is_some());
    }
    
    // Verify cache does not contain blocks 5-9
    for i in 5..10 {
        assert!(cache.get(&keys[i]).is_none());
    }
    
    // Insert more blocks to exceed capacity (LRU should evict oldest)
    for (i, block) in blocks.iter().skip(5).enumerate() {
        cache.insert(keys[i + 5], block.clone()).unwrap();
    }
    
    // Verify cache now contains blocks 5-9
    for i in 5..10 {
        assert!(cache.get(&keys[i]).is_some());
    }
    
    // Verify some of the earlier blocks were evicted (at least one)
    let mut evicted_count = 0;
    for i in 0..5 {
        if cache.get(&keys[i]).is_none() {
            evicted_count += 1;
        }
    }
    
    assert!(evicted_count > 0, "Cache should have evicted at least one block");
    
    // Test cache stats
    let stats = cache.get_stats();
    println!("Cache stats: {:?}", stats);
    assert!(stats.hits > 0);
    assert!(stats.inserts >= 10);
    assert!(stats.capacity_evictions + stats.ttl_evictions >= 5);
    
    // Test clear
    let _ = cache.clear();
    for i in 0..10 {
        assert!(cache.get(&keys[i]).is_none());
    }
}

#[test]
fn test_block_cache_with_storage() {
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };

    // Create storage with a small cache (2 blocks)
    let cache_config = BlockCacheConfig {
        max_capacity: 2,
        ttl: std::time::Duration::from_secs(60),
        ..Default::default()
    };
    let storage = FileStorage::with_cache_config(options, cache_config).unwrap();
    
    // Create a run with multiple blocks
    let block_count = 10;
    let mut run = Run::new(vec![]);
    
    // Create multiple blocks with different data
    for i in 0..block_count {
        let mut block = Block::new();
        for j in 0..5 {
            let key = (i * 100 + j) as i64;
            block.add_entry(key, key * 10).unwrap();
        }
        block.seal().unwrap();
        run.blocks.push(block);
    }
    
    // Setup fence pointers
    for (i, block) in run.blocks.iter().enumerate() {
        run.fence_pointers.add(block.header.min_key, block.header.max_key, i);
    }
    
    // Store the run
    let run_id = storage.store_run(0, &run).unwrap();
    
    // First load - should be a cache miss
    let start = Instant::now();
    let block0 = storage.load_block(run_id, 0).unwrap();
    let first_load_time = start.elapsed();
    
    // Second load of same block - should be a cache hit
    let start = Instant::now();
    let block0_again = storage.load_block(run_id, 0).unwrap();
    let second_load_time = start.elapsed();
    
    println!("First load (cache miss): {:?}", first_load_time);
    println!("Second load (cache hit): {:?}", second_load_time);
    
    // Verify both blocks are the same
    assert_eq!(block0.get(&0), Some(0));
    assert_eq!(block0_again.get(&0), Some(0));
    
    // The cache hit should be significantly faster than the cache miss
    assert!(second_load_time < first_load_time);
    
    // Now load several blocks to exceed cache capacity
    let _block1 = storage.load_block(run_id, 1).unwrap();
    let _block2 = storage.load_block(run_id, 2).unwrap();
    let _block3 = storage.load_block(run_id, 3).unwrap();
    
    // Block 0 should have been evicted, so this should be a cache miss again
    let start = Instant::now();
    let _block0_third = storage.load_block(run_id, 0).unwrap();
    let third_load_time = start.elapsed();
    
    println!("Third load (cache miss): {:?}", third_load_time);
    
    // The third load should be slower than second load (cache miss vs hit)
    assert!(third_load_time > second_load_time);
    
    // Get cache stats from storage
    let cache = storage.get_cache();
    let stats = cache.get_stats();
    println!("Cache stats: {:?}", stats);
    
    // Verify stats show some hits and misses
    assert!(stats.hits > 0);
    assert!(stats.misses > 0);
    assert!(stats.capacity_evictions + stats.ttl_evictions > 0);
}

#[test]
fn test_block_cache_with_io_batching() {
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };

    // Create storage with a medium cache (20 blocks)
    let cache_config = BlockCacheConfig {
        max_capacity: 20,
        ttl: std::time::Duration::from_secs(60),
        ..Default::default()
    };
    let storage = FileStorage::with_cache_config(options, cache_config).unwrap();
    
    // Create a run with multiple blocks
    let block_count = 30;
    let mut run = Run::new(vec![]);
    
    // Create multiple blocks with different data
    for i in 0..block_count {
        let mut block = Block::new();
        for j in 0..5 {
            let key = (i * 100 + j) as i64;
            block.add_entry(key, key * 10).unwrap();
        }
        block.seal().unwrap();
        run.blocks.push(block);
    }
    
    // Setup fence pointers
    for (i, block) in run.blocks.iter().enumerate() {
        run.fence_pointers.add(block.header.min_key, block.header.max_key, i);
    }
    
    // Store the run
    let run_id = storage.store_run(0, &run).unwrap();
    
    // Test batch loading with cache
    let blocks_to_load: Vec<_> = (0..10).collect();
    
    // First batch load - should be all cache misses
    let start = Instant::now();
    let first_batch = storage.load_blocks_batch(run_id, &blocks_to_load).unwrap();
    let first_batch_time = start.elapsed();
    
    // Verify blocks
    for (idx, block) in first_batch.iter().enumerate() {
        let key = (idx * 100) as i64;
        assert_eq!(block.get(&key), Some(key * 10));
    }
    
    // Second batch load - should be all cache hits
    let start = Instant::now();
    let second_batch = storage.load_blocks_batch(run_id, &blocks_to_load).unwrap();
    let second_batch_time = start.elapsed();
    
    // Verify blocks again
    for (idx, block) in second_batch.iter().enumerate() {
        let key = (idx * 100) as i64;
        assert_eq!(block.get(&key), Some(key * 10));
    }
    
    println!("First batch load (cache misses): {:?}", first_batch_time);
    println!("Second batch load (cache hits): {:?}", second_batch_time);
    
    // The second batch should be faster (all cache hits)
    assert!(second_batch_time < first_batch_time, 
        "Second batch time ({:?}) should be faster than first batch time ({:?})", 
        second_batch_time, first_batch_time);
    
    // Now load more blocks to exceed cache capacity
    let new_blocks_to_load: Vec<_> = (10..30).collect();
    let _new_batch = storage.load_blocks_batch(run_id, &new_blocks_to_load).unwrap();
    
    // Some of the original blocks should be evicted, so this should include some cache misses
    let start = Instant::now();
    let third_batch = storage.load_blocks_batch(run_id, &blocks_to_load).unwrap();
    let third_batch_time = start.elapsed();
    
    println!("Third batch load (some cache misses): {:?}", third_batch_time);
    
    // Verify blocks again
    for (idx, block) in third_batch.iter().enumerate() {
        let key = (idx * 100) as i64;
        assert_eq!(block.get(&key), Some(key * 10));
    }
    
    // Get cache stats
    let cache = storage.get_cache();
    let stats = cache.get_stats();
    println!("Cache stats: {:?}", stats);
    
    // Verify stats show expected pattern
    assert!(stats.hits > 0);
    assert!(stats.misses > 0);
    assert!(stats.capacity_evictions + stats.ttl_evictions > 0);
}