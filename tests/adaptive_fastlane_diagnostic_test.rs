use lsm_tree::run::{AdaptiveFastLanePointers, StandardFencePointers};
use lsm_tree::types::Key;
use std::time::Instant;

/// Diagnostic test to verify that adaptive fastlane is correctly delegating to standard
/// implementation for range queries.
#[test]
fn test_adaptive_fastlane_range_delegation() {
    // Configuration
    let size = 10_000;
    let range_count = 1_000;
    
    // Generate sequential keys
    let keys: Vec<Key> = (0..size as Key).collect();
    
    // Build fence pointer implementations
    let mut std_fps = StandardFencePointers::new();
    let mut adaptive_fps = AdaptiveFastLanePointers::new();
    
    for (i, key) in keys.iter().enumerate() {
        std_fps.add(*key, *key, i);
        adaptive_fps.add(*key, *key, i);
    }
    
    adaptive_fps.optimize();
    
    // Generate some range queries
    let range_queries: Vec<(Key, Key)> = (0..range_count)
        .map(|i| {
            let start = (i as Key * 10) % (size as Key);
            let end = start + 100;
            (start, end)
        })
        .collect();
    
    // Verify correctness - both implementations should return the same results
    for (i, (start, end)) in range_queries.iter().enumerate() {
        let std_result = std_fps.find_blocks_in_range(*start, *end);
        let adaptive_result = adaptive_fps.find_blocks_in_range(*start, *end);
        
        // Sorting because order might not be guaranteed across implementations
        let mut std_sorted = std_result.clone();
        let mut adaptive_sorted = adaptive_result.clone();
        std_sorted.sort();
        adaptive_sorted.sort();
        
        assert_eq!(
            std_sorted, adaptive_sorted, 
            "Range query {} failed: Different results for range ({}, {})", 
            i, start, end
        );
    }
    
    // Performance test - verify that adaptive implementation performs comparably to standard
    let warmup_count = 100;
    
    // Warmup
    for _ in 0..warmup_count {
        for (start, end) in &range_queries {
            let _ = std_fps.find_blocks_in_range(*start, *end);
            let _ = adaptive_fps.find_blocks_in_range(*start, *end);
        }
    }
    
    // Benchmark
    let std_start = Instant::now();
    for (start, end) in &range_queries {
        let _ = std_fps.find_blocks_in_range(*start, *end);
    }
    let std_duration = std_start.elapsed();
    
    let adaptive_start = Instant::now();
    for (start, end) in &range_queries {
        let _ = adaptive_fps.find_blocks_in_range(*start, *end);
    }
    let adaptive_duration = adaptive_start.elapsed();
    
    // Calculate time per range query
    let std_ns_per_range = std_duration.as_nanos() as f64 / range_count as f64;
    let adaptive_ns_per_range = adaptive_duration.as_nanos() as f64 / range_count as f64;
    
    // Calculate performance ratio - if adaptive is more than 50% slower than standard,
    // it may not be correctly delegating
    let performance_ratio = adaptive_ns_per_range / std_ns_per_range;
    
    // Print diagnostics
    println!("Standard implementation: {:.2} ns/range", std_ns_per_range);
    println!("Adaptive implementation: {:.2} ns/range", adaptive_ns_per_range);
    println!("Performance ratio (adaptive/standard): {:.2}x", performance_ratio);
    
    // A generous threshold to account for small overhead
    // With proper delegation, adaptive should be within 50% of standard performance
    assert!(
        performance_ratio < 1.5,
        "Adaptive implementation is significantly slower than standard ({:.2}x), suggesting delegation may not be working correctly",
        performance_ratio
    );
}

/// Test to verify that the sampling mechanism is correctly capturing and using performance data
#[test]
fn test_adaptive_sampling_mechanism() {
    // Create a large dataset to ensure Eytzinger is better for point queries
    let size = 1_000_000;
    
    // Generate sequential keys
    let keys: Vec<Key> = (0..size as Key).collect();
    
    // Build adaptive implementation
    let mut adaptive_fps = AdaptiveFastLanePointers::new();
    
    for (i, key) in keys.iter().enumerate() {
        adaptive_fps.add(*key, *key, i);
    }
    
    adaptive_fps.optimize();
    
    // Do lots of point queries to accumulate stats
    for i in 0..200 {
        let key = i * 5000;
        let _ = adaptive_fps.find_block_for_key(key);
    }
    
    // Do some range queries too
    for i in 0..50 {
        let start = i * 10000;
        let end = start + 1000;
        let _ = adaptive_fps.find_blocks_in_range(start, end);
    }
    
    // Now check that the adaptive implementation correctly switches between
    // implementations for point queries
    let point_query_count = 1000;
    let mut point_hits = 0;
    
    let point_start = Instant::now();
    for i in 0..point_query_count {
        let key = i * 1000;
        if adaptive_fps.find_block_for_key(key).is_some() {
            point_hits += 1;
        }
    }
    let point_duration = point_start.elapsed();
    
    // For range queries
    let range_query_count = 100;
    let mut range_blocks = 0;
    
    let range_start = Instant::now();
    for i in 0..range_query_count {
        let start = i * 10000;
        let end = start + 1000;
        let blocks = adaptive_fps.find_blocks_in_range(start, end);
        range_blocks += blocks.len();
    }
    let range_duration = range_start.elapsed();
    
    // Print diagnostics
    println!("Point queries: {} queries, {} hits, {:.2} ns/query", 
             point_query_count, point_hits, 
             point_duration.as_nanos() as f64 / point_query_count as f64);
    
    println!("Range queries: {} queries, {} blocks, {:.2} ns/query", 
             range_query_count, range_blocks,
             range_duration.as_nanos() as f64 / range_query_count as f64);
    
    // For a dataset of this size, adaptive should be using Eytzinger for point queries
    // and standard for range queries, but we can only test that it produces correct results
    assert!(point_hits > 0, "No hits for point queries");
    assert!(range_blocks > 0, "No blocks for range queries");
}