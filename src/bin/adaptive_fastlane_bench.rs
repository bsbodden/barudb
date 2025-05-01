use lsm_tree::run::{StandardFencePointers, EytzingerFencePointers, AdaptiveFastLanePointers};
use lsm_tree::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

fn main() {
    println!("=== Adaptive FastLane Benchmark ===\n");
    
    // Test with various workload patterns
    test_pure_point_queries();
    test_pure_range_queries();
    test_mixed_workload();
}

fn test_pure_point_queries() {
    println!("\n==== Pure Point Query Workload ====");
    
    // Run benchmarks with different dataset sizes
    for &size in &[1_000, 10_000, 100_000, 1_000_000] {
        println!("\nBenchmarking with {} keys:", size);
        benchmark_point_queries(size, 100_000);
    }
}

fn test_pure_range_queries() {
    println!("\n==== Pure Range Query Workload ====");
    
    // Run benchmarks with different dataset sizes
    for &size in &[1_000, 10_000, 100_000, 1_000_000] {
        println!("\nBenchmarking with {} keys:", size);
        benchmark_range_queries(size, 1_000);
    }
}

fn test_mixed_workload() {
    println!("\n==== Mixed Workload (80% Point, 20% Range) ====");
    
    // Run benchmarks with different dataset sizes
    for &size in &[1_000, 10_000, 100_000, 1_000_000] {
        println!("\nBenchmarking with {} keys:", size);
        benchmark_mixed_workload(size, 100_000);
    }
}

fn benchmark_point_queries(size: usize, lookup_count: usize) {
    // Generate sequential keys
    let keys: Vec<Key> = (0..size as Key).collect();
    
    // Generate lookup keys with 50% hit rate
    let mut rng = StdRng::seed_from_u64(42);
    let lookup_keys: Vec<Key> = (0..lookup_count)
        .map(|_| rng.random_range(0..size as Key * 2))
        .collect();
    
    // Build fence pointer implementations
    let mut std_fps = StandardFencePointers::new();
    let mut eytzinger_fps = EytzingerFencePointers::new();
    let mut adaptive_fps = AdaptiveFastLanePointers::new();
    
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        std_fps.add(chunk[0], chunk[1], i);
        eytzinger_fps.add(chunk[0], chunk[1], i);
        adaptive_fps.add(chunk[0], chunk[1], i);
    }
    
    // Optimize all implementations
    eytzinger_fps.optimize();
    adaptive_fps.optimize();
    
    // Benchmark Standard implementation
    let std_start = Instant::now();
    let mut std_hits = 0;
    for key in &lookup_keys {
        if std_fps.find_block_for_key(*key).is_some() {
            std_hits += 1;
        }
    }
    let std_duration = std_start.elapsed();
    
    // Benchmark Eytzinger implementation
    let eytzinger_start = Instant::now();
    let mut eytzinger_hits = 0;
    for key in &lookup_keys {
        if eytzinger_fps.find_block_for_key(*key).is_some() {
            eytzinger_hits += 1;
        }
    }
    let eytzinger_duration = eytzinger_start.elapsed();
    
    // Benchmark Adaptive implementation
    let adaptive_start = Instant::now();
    let mut adaptive_hits = 0;
    for key in &lookup_keys {
        if adaptive_fps.find_block_for_key(*key).is_some() {
            adaptive_hits += 1;
        }
    }
    let adaptive_duration = adaptive_start.elapsed();
    
    // Calculate time per lookup
    let std_ns_per_lookup = std_duration.as_nanos() as f64 / lookup_count as f64;
    let eytzinger_ns_per_lookup = eytzinger_duration.as_nanos() as f64 / lookup_count as f64;
    let adaptive_ns_per_lookup = adaptive_duration.as_nanos() as f64 / lookup_count as f64;
    
    // Print results
    println!("Point Query Results ({} lookups):", lookup_count);
    println!("  Standard:  {:.2} ns/lookup ({} hits)", std_ns_per_lookup, std_hits);
    println!("  Eytzinger: {:.2} ns/lookup ({} hits)", eytzinger_ns_per_lookup, eytzinger_hits);
    println!("  Adaptive:  {:.2} ns/lookup ({} hits)", adaptive_ns_per_lookup, adaptive_hits);
    
    // Compare with Standard
    let eytzinger_vs_std = (std_ns_per_lookup - eytzinger_ns_per_lookup) / std_ns_per_lookup * 100.0;
    let adaptive_vs_std = (std_ns_per_lookup - adaptive_ns_per_lookup) / std_ns_per_lookup * 100.0;
    
    println!("  Eytzinger vs Standard: {:.2}% {}", 
        eytzinger_vs_std.abs(),
        if eytzinger_vs_std > 0.0 { "faster" } else { "slower" });
    println!("  Adaptive vs Standard:  {:.2}% {}", 
        adaptive_vs_std.abs(),
        if adaptive_vs_std > 0.0 { "faster" } else { "slower" });
}

fn benchmark_range_queries(size: usize, range_count: usize) {
    // Generate sequential keys
    let keys: Vec<Key> = (0..size as Key).collect();
    
    // Generate range queries
    let mut rng = StdRng::seed_from_u64(42);
    let range_size = 100; // Average range size
    let ranges: Vec<(Key, Key)> = (0..range_count)
        .map(|_| {
            let start = rng.random_range(0..size as Key - range_size);
            let end = start + rng.random_range(1..range_size);
            (start, end)
        })
        .collect();
    
    // Build fence pointer implementations
    let mut std_fps = StandardFencePointers::new();
    let mut eytzinger_fps = EytzingerFencePointers::new();
    let mut adaptive_fps = AdaptiveFastLanePointers::new();
    
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        std_fps.add(chunk[0], chunk[1], i);
        eytzinger_fps.add(chunk[0], chunk[1], i);
        adaptive_fps.add(chunk[0], chunk[1], i);
    }
    
    // Optimize all implementations
    eytzinger_fps.optimize();
    adaptive_fps.optimize();
    
    // Benchmark Standard implementation for range queries
    let std_start = Instant::now();
    let mut std_total_blocks = 0;
    for (start, end) in &ranges {
        let blocks = std_fps.find_blocks_in_range(*start, *end);
        std_total_blocks += blocks.len();
    }
    let std_duration = std_start.elapsed();
    
    // Benchmark Eytzinger implementation for range queries
    let eytzinger_start = Instant::now();
    let mut eytzinger_total_blocks = 0;
    for (start, end) in &ranges {
        let blocks = eytzinger_fps.find_blocks_in_range(*start, *end);
        eytzinger_total_blocks += blocks.len();
    }
    let eytzinger_duration = eytzinger_start.elapsed();
    
    // Benchmark Adaptive implementation for range queries
    let adaptive_start = Instant::now();
    let mut adaptive_total_blocks = 0;
    for (start, end) in &ranges {
        let blocks = adaptive_fps.find_blocks_in_range(*start, *end);
        adaptive_total_blocks += blocks.len();
    }
    let adaptive_duration = adaptive_start.elapsed();
    
    // Calculate time per range query
    let std_ns_per_range = std_duration.as_nanos() as f64 / range_count as f64;
    let eytzinger_ns_per_range = eytzinger_duration.as_nanos() as f64 / range_count as f64;
    let adaptive_ns_per_range = adaptive_duration.as_nanos() as f64 / range_count as f64;
    
    // Print results
    println!("Range Query Results ({} ranges):", range_count);
    println!("  Standard:  {:.2} ns/range ({} blocks)", std_ns_per_range, std_total_blocks);
    println!("  Eytzinger: {:.2} ns/range ({} blocks)", eytzinger_ns_per_range, eytzinger_total_blocks);
    println!("  Adaptive:  {:.2} ns/range ({} blocks)", adaptive_ns_per_range, adaptive_total_blocks);
    
    // Compare with Standard
    let eytzinger_vs_std = (std_ns_per_range - eytzinger_ns_per_range) / std_ns_per_range * 100.0;
    let adaptive_vs_std = (std_ns_per_range - adaptive_ns_per_range) / std_ns_per_range * 100.0;
    
    println!("  Eytzinger vs Standard: {:.2}% {}", 
        eytzinger_vs_std.abs(),
        if eytzinger_vs_std > 0.0 { "faster" } else { "slower" });
    println!("  Adaptive vs Standard:  {:.2}% {}", 
        adaptive_vs_std.abs(),
        if adaptive_vs_std > 0.0 { "faster" } else { "slower" });
}

fn benchmark_mixed_workload(size: usize, query_count: usize) {
    // Configuration
    let point_ratio = 0.8; // 80% point queries, 20% range queries
    let range_size = 100;
    
    // Generate sequential keys
    let keys: Vec<Key> = (0..size as Key).collect();
    
    // Build fence pointer implementations
    let mut std_fps = StandardFencePointers::new();
    let mut eytzinger_fps = EytzingerFencePointers::new();
    let mut adaptive_fps = AdaptiveFastLanePointers::new();
    
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        std_fps.add(chunk[0], chunk[1], i);
        eytzinger_fps.add(chunk[0], chunk[1], i);
        adaptive_fps.add(chunk[0], chunk[1], i);
    }
    
    // Optimize all implementations
    eytzinger_fps.optimize();
    adaptive_fps.optimize();
    
    // Generate mixed queries
    let mut rng = StdRng::seed_from_u64(42);
    let queries: Vec<Query> = (0..query_count)
        .map(|_| {
            if rng.random_bool(point_ratio) {
                // Point query
                let key = rng.random_range(0..size as Key * 2);
                Query::Point(key)
            } else {
                // Range query
                let start = rng.random_range(0..size as Key - range_size);
                let end = start + rng.random_range(1..range_size);
                Query::Range(start, end)
            }
        })
        .collect();
    
    // Benchmark Standard implementation
    let std_start = Instant::now();
    let mut std_results = 0;
    for query in &queries {
        match query {
            Query::Point(key) => {
                if std_fps.find_block_for_key(*key).is_some() {
                    std_results += 1;
                }
            },
            Query::Range(start, end) => {
                let blocks = std_fps.find_blocks_in_range(*start, *end);
                std_results += blocks.len();
            }
        }
    }
    let std_duration = std_start.elapsed();
    
    // Benchmark Eytzinger implementation
    let eytzinger_start = Instant::now();
    let mut eytzinger_results = 0;
    for query in &queries {
        match query {
            Query::Point(key) => {
                if eytzinger_fps.find_block_for_key(*key).is_some() {
                    eytzinger_results += 1;
                }
            },
            Query::Range(start, end) => {
                let blocks = eytzinger_fps.find_blocks_in_range(*start, *end);
                eytzinger_results += blocks.len();
            }
        }
    }
    let eytzinger_duration = eytzinger_start.elapsed();
    
    // Benchmark Adaptive implementation
    let adaptive_start = Instant::now();
    let mut adaptive_results = 0;
    for query in &queries {
        match query {
            Query::Point(key) => {
                if adaptive_fps.find_block_for_key(*key).is_some() {
                    adaptive_results += 1;
                }
            },
            Query::Range(start, end) => {
                let blocks = adaptive_fps.find_blocks_in_range(*start, *end);
                adaptive_results += blocks.len();
            }
        }
    }
    let adaptive_duration = adaptive_start.elapsed();
    
    // Calculate time per query
    let std_ns_per_query = std_duration.as_nanos() as f64 / query_count as f64;
    let eytzinger_ns_per_query = eytzinger_duration.as_nanos() as f64 / query_count as f64;
    let adaptive_ns_per_query = adaptive_duration.as_nanos() as f64 / query_count as f64;
    
    // Print results
    println!("Mixed Workload Results ({} total queries, {}% point, {}% range):", 
        query_count, (point_ratio * 100.0) as usize, ((1.0 - point_ratio) * 100.0) as usize);
    println!("  Standard:  {:.2} ns/query ({} results)", std_ns_per_query, std_results);
    println!("  Eytzinger: {:.2} ns/query ({} results)", eytzinger_ns_per_query, eytzinger_results);
    println!("  Adaptive:  {:.2} ns/query ({} results)", adaptive_ns_per_query, adaptive_results);
    
    // Compare with Standard
    let eytzinger_vs_std = (std_ns_per_query - eytzinger_ns_per_query) / std_ns_per_query * 100.0;
    let adaptive_vs_std = (std_ns_per_query - adaptive_ns_per_query) / std_ns_per_query * 100.0;
    
    println!("  Eytzinger vs Standard: {:.2}% {}", 
        eytzinger_vs_std.abs(),
        if eytzinger_vs_std > 0.0 { "faster" } else { "slower" });
    println!("  Adaptive vs Standard:  {:.2}% {}", 
        adaptive_vs_std.abs(),
        if adaptive_vs_std > 0.0 { "faster" } else { "slower" });
}

/// Query enum for mixed workload benchmark
enum Query {
    Point(Key),
    Range(Key, Key),
}