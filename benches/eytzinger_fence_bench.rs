use criterion::{criterion_group, criterion_main, Criterion};
use lsm_tree::run::{EytzingerFencePointers, StandardFencePointers, FastLaneFencePointers};
use lsm_tree::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

/// Manual benchmark to compare Eytzinger vs Standard and FastLane fence pointers
pub fn benchmark_eytzinger() -> (f64, f64) {
    println!("\n=== Eytzinger Fence Pointers Performance Comparison ===");
    
    // Configuration
    let size = 100_000;
    let lookup_count = 100_000;
    
    // Generate sequential keys
    println!("Generating {} sequential keys...", size);
    let keys: Vec<Key> = (0..size as Key).collect();
    
    // Generate lookup keys with 50% hit rate
    println!("Generating {} lookup keys...", lookup_count);
    let mut rng = StdRng::seed_from_u64(42);
    let lookup_keys: Vec<Key> = (0..lookup_count)
        .map(|_| rng.gen_range(0..size as Key * 2))
        .collect();
    
    // Build Standard fence pointers
    println!("Building standard fence pointers...");
    let mut std_fps = StandardFencePointers::new();
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        std_fps.add(chunk[0], chunk[1], i);
    }
    
    // Build FastLane fence pointers
    println!("Building FastLane fence pointers...");
    let mut fastlane_fps = FastLaneFencePointers::with_group_size(64);
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        fastlane_fps.add(chunk[0], chunk[1], i);
    }
    
    // Build Eytzinger fence pointers
    println!("Building Eytzinger fence pointers...");
    let mut eytzinger_fps = EytzingerFencePointers::new();
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        eytzinger_fps.add(chunk[0], chunk[1], i);
    }
    
    // Optimize structures
    println!("Optimizing fence pointer structures...");
    let optimized_fastlane = fastlane_fps.optimize();
    eytzinger_fps.optimize();
    
    // Benchmark Standard fence pointers
    println!("Testing standard fence pointers...");
    let std_start = Instant::now();
    let mut std_hits = 0;
    for key in &lookup_keys {
        if std_fps.find_block_for_key(*key).is_some() {
            std_hits += 1;
        }
    }
    let std_duration = std_start.elapsed();
    
    // Benchmark FastLane fence pointers
    println!("Testing FastLane fence pointers...");
    let fastlane_start = Instant::now();
    let mut fastlane_hits = 0;
    for key in &lookup_keys {
        if optimized_fastlane.find_block_for_key(*key).is_some() {
            fastlane_hits += 1;
        }
    }
    let fastlane_duration = fastlane_start.elapsed();
    
    // Benchmark Eytzinger fence pointers
    println!("Testing Eytzinger fence pointers...");
    let eytzinger_start = Instant::now();
    let mut eytzinger_hits = 0;
    for key in &lookup_keys {
        if eytzinger_fps.find_block_for_key(*key).is_some() {
            eytzinger_hits += 1;
        }
    }
    let eytzinger_duration = eytzinger_start.elapsed();
    
    // Calculate time per lookup
    let std_ns_per_lookup = std_duration.as_nanos() as f64 / lookup_count as f64;
    let fastlane_ns_per_lookup = fastlane_duration.as_nanos() as f64 / lookup_count as f64;
    let eytzinger_ns_per_lookup = eytzinger_duration.as_nanos() as f64 / lookup_count as f64;
    
    // Calculate improvement percentages
    let fastlane_improvement = (std_ns_per_lookup - fastlane_ns_per_lookup) / std_ns_per_lookup * 100.0;
    let eytzinger_improvement = (std_ns_per_lookup - eytzinger_ns_per_lookup) / std_ns_per_lookup * 100.0;
    let eytzinger_vs_fastlane = (fastlane_ns_per_lookup - eytzinger_ns_per_lookup) / fastlane_ns_per_lookup * 100.0;
    
    // Print results
    println!("\n=== Results ({} lookups) ===", lookup_count);
    println!("Standard Fence Pointers:");
    println!("  - Total time: {:.2?}", std_duration);
    println!("  - Time per lookup: {:.2} ns", std_ns_per_lookup);
    println!("  - Hits: {}/{} ({}%)", std_hits, lookup_count, std_hits as f64 / lookup_count as f64 * 100.0);
    
    println!("FastLane Fence Pointers:");
    println!("  - Total time: {:.2?}", fastlane_duration);
    println!("  - Time per lookup: {:.2} ns", fastlane_ns_per_lookup);
    println!("  - Hits: {}/{} ({}%)", fastlane_hits, lookup_count, fastlane_hits as f64 / lookup_count as f64 * 100.0);
    println!("  - Improvement over Standard: {:.2}% {}", 
        fastlane_improvement.abs(),
        if fastlane_improvement > 0.0 { "faster" } else { "slower" });
    
    println!("Eytzinger Fence Pointers:");
    println!("  - Total time: {:.2?}", eytzinger_duration);
    println!("  - Time per lookup: {:.2} ns", eytzinger_ns_per_lookup);
    println!("  - Hits: {}/{} ({}%)", eytzinger_hits, lookup_count, eytzinger_hits as f64 / lookup_count as f64 * 100.0);
    println!("  - Improvement over Standard: {:.2}% {}", 
        eytzinger_improvement.abs(),
        if eytzinger_improvement > 0.0 { "faster" } else { "slower" });
    println!("  - Improvement over FastLane: {:.2}% {}", 
        eytzinger_vs_fastlane.abs(),
        if eytzinger_vs_fastlane > 0.0 { "faster" } else { "slower" });
        
    // Memory comparison
    let std_memory = std::mem::size_of::<StandardFencePointers>() + 
                   std_fps.pointers.capacity() * std::mem::size_of::<(Key, Key, usize)>();
    let fastlane_memory = optimized_fastlane.memory_usage();
    let eytzinger_memory = eytzinger_fps.memory_usage();
    
    println!("\n=== Memory Usage ===");
    println!("Standard: {} bytes", std_memory);
    println!("FastLane: {} bytes", fastlane_memory);
    println!("Eytzinger: {} bytes", eytzinger_memory);
    
    // Compare memory usage
    let fastlane_memory_ratio = fastlane_memory as f64 / std_memory as f64;
    let eytzinger_memory_ratio = eytzinger_memory as f64 / std_memory as f64;
    
    println!("Memory ratio (FastLane vs Standard): {:.2}x", fastlane_memory_ratio);
    println!("Memory ratio (Eytzinger vs Standard): {:.2}x", eytzinger_memory_ratio);
    
    // Return the improvement percentages for criterion
    (eytzinger_improvement, eytzinger_vs_fastlane)
}

// Benchmark for range queries
pub fn benchmark_eytzinger_range() -> (f64, f64) {
    println!("\n=== Eytzinger Fence Pointers Range Query Performance ===");
    
    // Configuration
    let size = 100_000;
    let range_count = 1_000;
    let range_size = 100; // Average range size
    
    // Generate sequential keys
    println!("Generating {} sequential keys...", size);
    let keys: Vec<Key> = (0..size as Key).collect();
    
    // Generate range queries
    println!("Generating {} range queries...", range_count);
    let mut rng = StdRng::seed_from_u64(42);
    let ranges: Vec<(Key, Key)> = (0..range_count)
        .map(|_| {
            let start = rng.gen_range(0..size as Key - range_size);
            let end = start + rng.gen_range(1..range_size);
            (start, end)
        })
        .collect();
    
    // Build Standard fence pointers
    println!("Building standard fence pointers...");
    let mut std_fps = StandardFencePointers::new();
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        std_fps.add(chunk[0], chunk[1], i);
    }
    
    // Build FastLane fence pointers
    println!("Building FastLane fence pointers...");
    let mut fastlane_fps = FastLaneFencePointers::with_group_size(64);
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        fastlane_fps.add(chunk[0], chunk[1], i);
    }
    
    // Build Eytzinger fence pointers
    println!("Building Eytzinger fence pointers...");
    let mut eytzinger_fps = EytzingerFencePointers::new();
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        eytzinger_fps.add(chunk[0], chunk[1], i);
    }
    
    // Optimize structures
    println!("Optimizing fence pointer structures...");
    let optimized_fastlane = fastlane_fps.optimize();
    eytzinger_fps.optimize();
    
    // Benchmark Standard fence pointers for range queries
    println!("Testing standard fence pointers range queries...");
    let std_start = Instant::now();
    let mut std_total_blocks = 0;
    for (start, end) in &ranges {
        let blocks = std_fps.find_blocks_in_range(*start, *end);
        std_total_blocks += blocks.len();
    }
    let std_duration = std_start.elapsed();
    
    // Benchmark FastLane fence pointers for range queries
    println!("Testing FastLane fence pointers range queries...");
    let fastlane_start = Instant::now();
    let mut fastlane_total_blocks = 0;
    for (start, end) in &ranges {
        let blocks = optimized_fastlane.find_blocks_in_range(*start, *end);
        fastlane_total_blocks += blocks.len();
    }
    let fastlane_duration = fastlane_start.elapsed();
    
    // Benchmark Eytzinger fence pointers for range queries
    println!("Testing Eytzinger fence pointers range queries...");
    let eytzinger_start = Instant::now();
    let mut eytzinger_total_blocks = 0;
    for (start, end) in &ranges {
        let blocks = eytzinger_fps.find_blocks_in_range(*start, *end);
        eytzinger_total_blocks += blocks.len();
    }
    let eytzinger_duration = eytzinger_start.elapsed();
    
    // Calculate time per range query
    let std_ns_per_range = std_duration.as_nanos() as f64 / range_count as f64;
    let fastlane_ns_per_range = fastlane_duration.as_nanos() as f64 / range_count as f64;
    let eytzinger_ns_per_range = eytzinger_duration.as_nanos() as f64 / range_count as f64;
    
    // Calculate improvement percentages
    let fastlane_range_improvement = (std_ns_per_range - fastlane_ns_per_range) / std_ns_per_range * 100.0;
    let eytzinger_range_improvement = (std_ns_per_range - eytzinger_ns_per_range) / std_ns_per_range * 100.0;
    let eytzinger_vs_fastlane_range = (fastlane_ns_per_range - eytzinger_ns_per_range) / fastlane_ns_per_range * 100.0;
    
    // Print results
    println!("\n=== Range Query Results ({} ranges) ===", range_count);
    println!("Standard Fence Pointers:");
    println!("  - Total time: {:.2?}", std_duration);
    println!("  - Time per range query: {:.2} ns", std_ns_per_range);
    println!("  - Total blocks found: {}", std_total_blocks);
    
    println!("FastLane Fence Pointers:");
    println!("  - Total time: {:.2?}", fastlane_duration);
    println!("  - Time per range query: {:.2} ns", fastlane_ns_per_range);
    println!("  - Total blocks found: {}", fastlane_total_blocks);
    println!("  - Improvement over Standard: {:.2}% {}", 
        fastlane_range_improvement.abs(),
        if fastlane_range_improvement > 0.0 { "faster" } else { "slower" });
    
    println!("Eytzinger Fence Pointers:");
    println!("  - Total time: {:.2?}", eytzinger_duration);
    println!("  - Time per range query: {:.2} ns", eytzinger_ns_per_range);
    println!("  - Total blocks found: {}", eytzinger_total_blocks);
    println!("  - Improvement over Standard: {:.2}% {}", 
        eytzinger_range_improvement.abs(),
        if eytzinger_range_improvement > 0.0 { "faster" } else { "slower" });
    println!("  - Improvement over FastLane: {:.2}% {}", 
        eytzinger_vs_fastlane_range.abs(),
        if eytzinger_vs_fastlane_range > 0.0 { "faster" } else { "slower" });
    
    // Return the improvement percentages for criterion
    (eytzinger_range_improvement, eytzinger_vs_fastlane_range)
}

// Benchmark comparing performance across different dataset sizes
pub fn benchmark_eytzinger_scaling() {
    println!("\n=== Eytzinger Fence Pointers Scaling Performance ===");
    
    // Test with various dataset sizes
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let lookup_count = 100_000;
    
    println!("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}", 
        "Size", "Std (ns)", "FastLane (ns)", "Eytzinger (ns)", 
        "Eytz vs Std", "Eytz vs FL");
    
    for &size in &sizes {
        // Generate sequential keys
        let keys: Vec<Key> = (0..size as Key).collect();
        
        // Generate lookup keys with 50% hit rate
        let mut rng = StdRng::seed_from_u64(42);
        let lookup_keys: Vec<Key> = (0..lookup_count)
            .map(|_| rng.gen_range(0..size as Key * 2))
            .collect();
        
        // Build Standard fence pointers
        let mut std_fps = StandardFencePointers::new();
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 { continue; }
            std_fps.add(chunk[0], chunk[1], i);
        }
        
        // Build FastLane fence pointers
        let mut fastlane_fps = FastLaneFencePointers::with_group_size(64);
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 { continue; }
            fastlane_fps.add(chunk[0], chunk[1], i);
        }
        
        // Build Eytzinger fence pointers
        let mut eytzinger_fps = EytzingerFencePointers::new();
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 { continue; }
            eytzinger_fps.add(chunk[0], chunk[1], i);
        }
        
        // Optimize structures
        let optimized_fastlane = fastlane_fps.optimize();
        eytzinger_fps.optimize();
        
        // Benchmark Standard fence pointers
        let std_start = Instant::now();
        for key in &lookup_keys {
            let _ = std_fps.find_block_for_key(*key);
        }
        let std_duration = std_start.elapsed();
        let std_ns_per_lookup = std_duration.as_nanos() as f64 / lookup_count as f64;
        
        // Benchmark FastLane fence pointers
        let fastlane_start = Instant::now();
        for key in &lookup_keys {
            let _ = optimized_fastlane.find_block_for_key(*key);
        }
        let fastlane_duration = fastlane_start.elapsed();
        let fastlane_ns_per_lookup = fastlane_duration.as_nanos() as f64 / lookup_count as f64;
        
        // Benchmark Eytzinger fence pointers
        let eytzinger_start = Instant::now();
        for key in &lookup_keys {
            let _ = eytzinger_fps.find_block_for_key(*key);
        }
        let eytzinger_duration = eytzinger_start.elapsed();
        let eytzinger_ns_per_lookup = eytzinger_duration.as_nanos() as f64 / lookup_count as f64;
        
        // Calculate improvement percentages
        let eytzinger_vs_std = (std_ns_per_lookup - eytzinger_ns_per_lookup) / std_ns_per_lookup * 100.0;
        let eytzinger_vs_fastlane = (fastlane_ns_per_lookup - eytzinger_ns_per_lookup) / fastlane_ns_per_lookup * 100.0;
        
        println!("{:<10} {:<10.2} {:<10.2} {:<10.2} {:<10.2}% {:<10.2}%", 
            size, std_ns_per_lookup, fastlane_ns_per_lookup, eytzinger_ns_per_lookup,
            eytzinger_vs_std, eytzinger_vs_fastlane);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    // Run our manual benchmark in bench group
    let mut group = c.benchmark_group("Eytzinger Fence Pointers");
    
    group.bench_function("eytzinger_point_query", |b| {
        b.iter(|| benchmark_eytzinger())
    });
    
    group.bench_function("eytzinger_range_query", |b| {
        b.iter(|| benchmark_eytzinger_range())
    });
    
    group.finish();
    
    // Also run and print the scaling benchmark (not as part of criterion)
    benchmark_eytzinger_scaling();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);