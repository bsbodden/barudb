use barudb::run::{
    SimpleFastLaneFencePointers, TwoLevelFastLaneFencePointers, 
    StandardFencePointers, FastLaneFencePointers, FencePointersInterface
};
use barudb::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

#[test]
fn test_simple_fastlane() {
    // Create a new implementation
    let mut fps = SimpleFastLaneFencePointers::new();
    
    // Add some fence pointers
    fps.add(10, 20, 0);
    fps.add(25, 35, 1);
    fps.add(40, 50, 2);
    
    // Test finding blocks
    let result1 = fps.find_block_for_key(15);
    let result2 = fps.find_block_for_key(30);
    let result3 = fps.find_block_for_key(45);
    let result4 = fps.find_block_for_key(22); // Should be None
    
    assert_eq!(result1, Some(0));
    assert_eq!(result2, Some(1));
    assert_eq!(result3, Some(2));
    assert_eq!(result4, None);
    
    // Test range queries
    let range_blocks = fps.find_blocks_in_range(15, 45);
    
    // Should contain blocks 0, 1, and 2
    assert_eq!(range_blocks.len(), 3);
    assert!(range_blocks.contains(&0));
    assert!(range_blocks.contains(&1));
    assert!(range_blocks.contains(&2));
    
    // Test memory usage
    let memory = fps.memory_usage();
    assert!(memory > 0);
    
    println!("Simple FastLane test passed!");
}

#[test]
fn test_two_level_fastlane() {
    // Create a new implementation
    let mut fps = TwoLevelFastLaneFencePointers::with_partition_size(8);
    
    // Add many fence pointers to force partitioning
    for i in 0..100 {
        let min_key = i * 10;
        let max_key = min_key + 9;
        fps.add(min_key, max_key, i as usize);
    }
    
    // Verify all keys can be found
    for i in 0..100 {
        let key = i * 10 + 5; // Middle of each range
        let result = fps.find_block_for_key(key);
        assert_eq!(result, Some(i as usize), "Key {} should be found in block {}", key, i);
    }
    
    // Test range queries
    let range_blocks = fps.find_blocks_in_range(245, 275);
    
    // Should contain blocks 24, 25, 26, 27
    assert_eq!(range_blocks.len(), 4);
    assert!(range_blocks.contains(&24));
    assert!(range_blocks.contains(&25));
    assert!(range_blocks.contains(&26));
    assert!(range_blocks.contains(&27));
    
    println!("Two-Level FastLane test passed!");
}

#[test]
fn test_using_trait() {
    // Function that takes any FencePointersInterface implementation
    fn test_implementation(name: &str, mut fps: impl FencePointersInterface) {
        // Add some fence pointers
        fps.add(10, 20, 0);
        fps.add(25, 35, 1);
        fps.add(40, 50, 2);
        
        // Verify they can be found
        assert_eq!(fps.find_block_for_key(15), Some(0));
        assert_eq!(fps.find_block_for_key(30), Some(1));
        assert_eq!(fps.find_block_for_key(45), Some(2));
        
        println!("Implementation {} passed the trait test!", name);
    }
    
    // Test both implementations
    test_implementation("Simple", SimpleFastLaneFencePointers::new());
    test_implementation("TwoLevel", TwoLevelFastLaneFencePointers::new());
    test_implementation("Standard", StandardFencePointers::new());
    test_implementation("Original", FastLaneFencePointers::new());
}

#[test]
fn test_million_pattern() {
    // Create a new implementation
    let mut fps = SimpleFastLaneFencePointers::new();
    
    // Add fence pointers with million pattern
    for i in 1..10 {
        let min_key = i * 1_000_000;
        let max_key = min_key + 999;
        fps.add(min_key, max_key, i as usize);
    }
    
    // Test exact matches
    for i in 1..10 {
        let key = i * 1_000_000 + 500;
        let result = fps.find_block_for_key(key);
        assert_eq!(result, Some(i as usize), "Key {} should be found in block {}", key, i);
    }
    
    // Test range query across million boundaries
    let range_blocks = fps.find_blocks_in_range(1_500_000, 3_500_000);
    
    // Should include blocks 2 and 3
    assert!(range_blocks.contains(&2));
    assert!(range_blocks.contains(&3));
    
    println!("Million pattern test passed!");
}

#[test]
fn test_bench_comparison() {
    let dataset_size = 10_000;
    let lookup_count = 1_000;
    
    // Create dataset - sequential keys for simplicity
    let mut dataset = Vec::with_capacity(dataset_size);
    for i in 0..dataset_size {
        dataset.push((i as Key, (i as Key) + 10, i));
    }
    
    // Create lookup keys - mix of hits and misses
    let mut rng = StdRng::seed_from_u64(42);
    let mut lookup_keys = Vec::with_capacity(lookup_count);
    for _ in 0..lookup_count {
        lookup_keys.push(rng.random::<Key>() % (dataset_size as Key * 2)); // 50% hit rate
    }
    
    // Create implementations
    let mut standard_fps = StandardFencePointers::new();
    let mut simple_fps = SimpleFastLaneFencePointers::new();
    let mut two_level_fps = TwoLevelFastLaneFencePointers::with_partition_size(64);
    let mut original_fps = FastLaneFencePointers::new();
    
    // Add data to each
    for &(min_key, max_key, block_idx) in &dataset {
        standard_fps.add(min_key, max_key, block_idx);
        simple_fps.add(min_key, max_key, block_idx);
        two_level_fps.add(min_key, max_key, block_idx);
        original_fps.add(min_key, max_key, block_idx);
    }
    
    // Benchmark Standard
    let start = Instant::now();
    let mut std_hits = 0;
    for &key in &lookup_keys {
        if standard_fps.find_block_for_key(key).is_some() {
            std_hits += 1;
        }
    }
    let std_time = start.elapsed();
    
    // Benchmark Simple
    let start = Instant::now();
    let mut simple_hits = 0;
    for &key in &lookup_keys {
        if simple_fps.find_block_for_key(key).is_some() {
            simple_hits += 1;
        }
    }
    let simple_time = start.elapsed();
    
    // Benchmark TwoLevel
    let start = Instant::now();
    let mut two_level_hits = 0;
    for &key in &lookup_keys {
        if two_level_fps.find_block_for_key(key).is_some() {
            two_level_hits += 1;
        }
    }
    let two_level_time = start.elapsed();
    
    // Benchmark Original
    let start = Instant::now();
    let mut original_hits = 0;
    for &key in &lookup_keys {
        if original_fps.find_block_for_key(key).is_some() {
            original_hits += 1;
        }
    }
    let original_time = start.elapsed();
    
    // Print results
    println!("\n=== Point Query Performance Comparison ===");
    println!("Dataset size: {}, Lookups: {}", dataset_size, lookup_count);
    println!("Standard:      {:.6?}, {} hits", std_time, std_hits);
    println!("Simple:        {:.6?}, {} hits, {:.2}x speedup", 
             simple_time, simple_hits, std_time.as_secs_f64() / simple_time.as_secs_f64());
    println!("TwoLevel:      {:.6?}, {} hits, {:.2}x speedup", 
             two_level_time, two_level_hits, std_time.as_secs_f64() / two_level_time.as_secs_f64());
    println!("Original:      {:.6?}, {} hits, {:.2}x speedup", 
             original_time, original_hits, std_time.as_secs_f64() / original_time.as_secs_f64());
    
    // Print memory usage
    println!("\n=== Memory Usage Comparison ===");
    println!("Standard:      {} bytes", standard_fps.memory_usage());
    println!("Simple:        {} bytes", simple_fps.memory_usage());
    println!("TwoLevel:      {} bytes", two_level_fps.memory_usage());
    println!("Original:      {} bytes", original_fps.memory_usage());
}

#[test]
fn test_range_query_bench() {
    let dataset_size = 10_000;
    let range_count = 100;
    
    // Create dataset - sequential keys for simplicity
    let mut dataset = Vec::with_capacity(dataset_size);
    for i in 0..dataset_size {
        dataset.push((i as Key, (i as Key) + 10, i));
    }
    
    // Create implementations
    let mut standard_fps = StandardFencePointers::new();
    let mut simple_fps = SimpleFastLaneFencePointers::new();
    let mut two_level_fps = TwoLevelFastLaneFencePointers::with_partition_size(64);
    let mut original_fps = FastLaneFencePointers::new();
    
    // Add data to each
    for &(min_key, max_key, block_idx) in &dataset {
        standard_fps.add(min_key, max_key, block_idx);
        simple_fps.add(min_key, max_key, block_idx);
        two_level_fps.add(min_key, max_key, block_idx);
        original_fps.add(min_key, max_key, block_idx);
    }
    
    // Create range queries - cover approximately 5% of the key space each
    let mut rng = StdRng::seed_from_u64(42);
    let range_size = (dataset_size as f64 * 0.05) as Key;
    let mut ranges = Vec::with_capacity(range_count);
    
    for _ in 0..range_count {
        let start = rng.random::<Key>() % (dataset_size as Key - range_size);
        let end = start + range_size;
        ranges.push((start, end));
    }
    
    // Benchmark Standard
    let start = Instant::now();
    let mut std_total_blocks = 0;
    for &(start, end) in &ranges {
        let blocks = standard_fps.find_blocks_in_range(start, end);
        std_total_blocks += blocks.len();
    }
    let std_time = start.elapsed();
    
    // Benchmark Simple
    let start = Instant::now();
    let mut simple_total_blocks = 0;
    for &(start, end) in &ranges {
        let blocks = simple_fps.find_blocks_in_range(start, end);
        simple_total_blocks += blocks.len();
    }
    let simple_time = start.elapsed();
    
    // Benchmark TwoLevel
    let start = Instant::now();
    let mut two_level_total_blocks = 0;
    for &(start, end) in &ranges {
        let blocks = two_level_fps.find_blocks_in_range(start, end);
        two_level_total_blocks += blocks.len();
    }
    let two_level_time = start.elapsed();
    
    // Benchmark Original
    let start = Instant::now();
    let mut original_total_blocks = 0;
    for &(start, end) in &ranges {
        let blocks = original_fps.find_blocks_in_range(start, end);
        original_total_blocks += blocks.len();
    }
    let original_time = start.elapsed();
    
    // Print results
    println!("\n=== Range Query Performance Comparison ===");
    println!("Dataset size: {}, Range queries: {}", dataset_size, range_count);
    println!("Standard:      {:.6?}, {} total blocks", std_time, std_total_blocks);
    println!("Simple:        {:.6?}, {} total blocks, {:.2}x speedup", 
             simple_time, simple_total_blocks, std_time.as_secs_f64() / simple_time.as_secs_f64());
    println!("TwoLevel:      {:.6?}, {} total blocks, {:.2}x speedup", 
             two_level_time, two_level_total_blocks, std_time.as_secs_f64() / two_level_time.as_secs_f64());
    println!("Original:      {:.6?}, {} total blocks, {:.2}x speedup", 
             original_time, original_total_blocks, std_time.as_secs_f64() / original_time.as_secs_f64());
    
    // Check block consistency
    println!("\nBlock retrieval consistency:");
    println!("Simple/Standard:    {:.2}%", simple_total_blocks as f64 / std_total_blocks as f64 * 100.0);
    println!("TwoLevel/Standard:  {:.2}%", two_level_total_blocks as f64 / std_total_blocks as f64 * 100.0);
    println!("Original/Standard:  {:.2}%", original_total_blocks as f64 / std_total_blocks as f64 * 100.0);
}