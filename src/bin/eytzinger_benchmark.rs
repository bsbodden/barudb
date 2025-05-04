use lsm_tree::run::{EytzingerFencePointers, StandardFencePointers, FastLaneFencePointers};
use lsm_tree::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

fn main() {
    println!("\n=== Eytzinger Fence Pointers Performance Benchmark ===");
    
    // Run our benchmarks
    benchmark_eytzinger();
    benchmark_eytzinger_range();
    benchmark_eytzinger_scaling();
    benchmark_eytzinger_optimizations();
    benchmark_eytzinger_range_optimizations();
}

/// Manual benchmark to compare Eytzinger vs Standard and FastLane fence pointers
fn benchmark_eytzinger() -> (f64, f64) {
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
        .map(|_| rng.random_range(0..size as Key * 2))
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
    fastlane_fps.optimize();
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
        if fastlane_fps.find_block_for_key(*key).is_some() {
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
    let fastlane_memory = fastlane_fps.memory_usage();
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
    
    // Return the improvement percentages
    (eytzinger_improvement, eytzinger_vs_fastlane)
}

// Benchmark for range queries
fn benchmark_eytzinger_range() -> (f64, f64) {
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
            let start = rng.random_range(0..size as Key - range_size);
            let end = start + rng.random_range(1..range_size);
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
    fastlane_fps.optimize();
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
        let blocks = fastlane_fps.find_blocks_in_range(*start, *end);
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
    
    // Return the improvement percentages
    (eytzinger_range_improvement, eytzinger_vs_fastlane_range)
}

// Benchmark comparing performance across different dataset sizes
fn benchmark_eytzinger_scaling() {
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
            .map(|_| rng.random_range(0..size as Key * 2))
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
        fastlane_fps.optimize();
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
            let _ = fastlane_fps.find_block_for_key(*key);
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

/// Benchmark to compare different optimizations of the Eytzinger implementation
/// This benchmark measures the performance impact of the various optimizations we've added
fn benchmark_eytzinger_optimizations() {
    println!("\n=== Eytzinger Optimizations Performance Comparison ===");
    
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
        .map(|_| rng.random_range(0..size as Key * 2))
        .collect();
    
    // Build Standard fence pointers (as baseline)
    println!("Building standard fence pointers...");
    let mut std_fps = StandardFencePointers::new();
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        std_fps.add(chunk[0], chunk[1], i);
    }
    
    // Create different Eytzinger variants with specific optimizations enabled or disabled
    
    // 1. Basic Eytzinger without SIMD or cache alignment
    println!("Building basic Eytzinger fence pointers...");
    let mut basic_eytzinger = EytzingerFencePointers::new();
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        basic_eytzinger.add(chunk[0], chunk[1], i);
    }
    basic_eytzinger.optimize();
    
    // 2. Eytzinger with only SIMD enabled (no cache alignment)
    println!("Building SIMD-only Eytzinger fence pointers...");
    let mut simd_eytzinger = EytzingerFencePointers::with_simd(true);
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        simd_eytzinger.add(chunk[0], chunk[1], i);
    }
    simd_eytzinger.optimize();
    
    // 3. Eytzinger with all optimizations enabled
    println!("Building fully optimized Eytzinger fence pointers...");
    let mut full_eytzinger = EytzingerFencePointers::with_capacity(size / 2);
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        full_eytzinger.add(chunk[0], chunk[1], i);
    }
    full_eytzinger.optimize();
    
    // Benchmark Standard fence pointers (baseline)
    println!("Testing standard fence pointers...");
    let std_start = Instant::now();
    let mut std_hits = 0;
    for key in &lookup_keys {
        if std_fps.find_block_for_key(*key).is_some() {
            std_hits += 1;
        }
    }
    let std_duration = std_start.elapsed();
    
    // Benchmark basic Eytzinger
    println!("Testing basic Eytzinger fence pointers...");
    let basic_start = Instant::now();
    let mut basic_hits = 0;
    for key in &lookup_keys {
        if basic_eytzinger.find_block_for_key(*key).is_some() {
            basic_hits += 1;
        }
    }
    let basic_duration = basic_start.elapsed();
    
    // Benchmark SIMD-only Eytzinger
    println!("Testing SIMD-only Eytzinger fence pointers...");
    let simd_start = Instant::now();
    let mut simd_hits = 0;
    for key in &lookup_keys {
        if simd_eytzinger.find_block_for_key(*key).is_some() {
            simd_hits += 1;
        }
    }
    let simd_duration = simd_start.elapsed();
    
    // Benchmark fully optimized Eytzinger
    println!("Testing fully optimized Eytzinger fence pointers...");
    let full_start = Instant::now();
    let mut full_hits = 0;
    for key in &lookup_keys {
        if full_eytzinger.find_block_for_key(*key).is_some() {
            full_hits += 1;
        }
    }
    let full_duration = full_start.elapsed();
    
    // Calculate time per lookup
    let std_ns_per_lookup = std_duration.as_nanos() as f64 / lookup_count as f64;
    let basic_ns_per_lookup = basic_duration.as_nanos() as f64 / lookup_count as f64;
    let simd_ns_per_lookup = simd_duration.as_nanos() as f64 / lookup_count as f64;
    let full_ns_per_lookup = full_duration.as_nanos() as f64 / lookup_count as f64;
    
    // Calculate improvement percentages
    let basic_improvement = (std_ns_per_lookup - basic_ns_per_lookup) / std_ns_per_lookup * 100.0;
    let simd_improvement = (std_ns_per_lookup - simd_ns_per_lookup) / std_ns_per_lookup * 100.0;
    let full_improvement = (std_ns_per_lookup - full_ns_per_lookup) / std_ns_per_lookup * 100.0;
    
    // Calculate improvement from basic to full optimizations
    let optimization_improvement = (basic_ns_per_lookup - full_ns_per_lookup) / basic_ns_per_lookup * 100.0;
    
    // Print results
    println!("\n=== Results ({} lookups) ===", lookup_count);
    println!("Standard Fence Pointers:");
    println!("  - Total time: {:.2?}", std_duration);
    println!("  - Time per lookup: {:.2} ns", std_ns_per_lookup);
    println!("  - Hits: {}/{} ({}%)", std_hits, lookup_count, std_hits as f64 / lookup_count as f64 * 100.0);
    
    println!("Basic Eytzinger Fence Pointers:");
    println!("  - Total time: {:.2?}", basic_duration);
    println!("  - Time per lookup: {:.2} ns", basic_ns_per_lookup);
    println!("  - Hits: {}/{} ({}%)", basic_hits, lookup_count, basic_hits as f64 / lookup_count as f64 * 100.0);
    println!("  - Improvement over Standard: {:.2}% {}", 
        basic_improvement.abs(),
        if basic_improvement > 0.0 { "faster" } else { "slower" });
    
    println!("SIMD-only Eytzinger Fence Pointers:");
    println!("  - Total time: {:.2?}", simd_duration);
    println!("  - Time per lookup: {:.2} ns", simd_ns_per_lookup);
    println!("  - Hits: {}/{} ({}%)", simd_hits, lookup_count, simd_hits as f64 / lookup_count as f64 * 100.0);
    println!("  - Improvement over Standard: {:.2}% {}", 
        simd_improvement.abs(),
        if simd_improvement > 0.0 { "faster" } else { "slower" });
    
    println!("Fully Optimized Eytzinger Fence Pointers:");
    println!("  - Total time: {:.2?}", full_duration);
    println!("  - Time per lookup: {:.2} ns", full_ns_per_lookup);
    println!("  - Hits: {}/{} ({}%)", full_hits, lookup_count, full_hits as f64 / lookup_count as f64 * 100.0);
    println!("  - Improvement over Standard: {:.2}% {}", 
        full_improvement.abs(),
        if full_improvement > 0.0 { "faster" } else { "slower" });
    
    println!("\nOptimization Impact:");
    println!("  - Improvement from Basic to Fully Optimized: {:.2}% {}", 
        optimization_improvement.abs(),
        if optimization_improvement > 0.0 { "faster" } else { "slower" });
}

/// Benchmark specifically for range queries with different implementations
fn benchmark_eytzinger_range_optimizations() {
    println!("\n=== Eytzinger Range Query Optimizations Performance ===");
    
    // Configuration
    let size = 100_000;
    let range_count = 1_000;
    let range_size_avg = 100; // Average range size
    
    // Generate sequential keys
    println!("Generating {} sequential keys...", size);
    let keys: Vec<Key> = (0..size as Key).collect();
    
    // Generate range queries
    println!("Generating {} range queries...", range_count);
    let mut rng = StdRng::seed_from_u64(42);
    let ranges: Vec<(Key, Key)> = (0..range_count)
        .map(|_| {
            let start = rng.random_range(0..size as Key - range_size_avg);
            let range_len = rng.random_range(1..range_size_avg * 2); // Varied range sizes
            let end = start + range_len;
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
    
    // Build fully optimized Eytzinger fence pointers
    println!("Building optimized Eytzinger fence pointers...");
    let mut eytzinger_fps = EytzingerFencePointers::with_capacity(size / 2);
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        eytzinger_fps.add(chunk[0], chunk[1], i);
    }
    eytzinger_fps.optimize();
    
    // Benchmark Standard fence pointers
    println!("Testing standard fence pointers for range queries...");
    let std_start = Instant::now();
    let mut std_total_blocks = 0;
    for (start, end) in &ranges {
        let blocks = std_fps.find_blocks_in_range(*start, *end);
        std_total_blocks += blocks.len();
    }
    let std_duration = std_start.elapsed();
    
    // Benchmark optimized Eytzinger fence pointers
    println!("Testing optimized Eytzinger fence pointers for range queries...");
    let eytzinger_start = Instant::now();
    let mut eytzinger_total_blocks = 0;
    for (start, end) in &ranges {
        let blocks = eytzinger_fps.find_blocks_in_range(*start, *end);
        eytzinger_total_blocks += blocks.len();
    }
    let eytzinger_duration = eytzinger_start.elapsed();
    
    // Calculate time per range query
    let std_ns_per_range = std_duration.as_nanos() as f64 / range_count as f64;
    let eytzinger_ns_per_range = eytzinger_duration.as_nanos() as f64 / range_count as f64;
    
    // Calculate improvement percentage
    let improvement = (std_ns_per_range - eytzinger_ns_per_range) / std_ns_per_range * 100.0;
    
    // Print results
    println!("\n=== Range Query Results ({} ranges) ===", range_count);
    println!("Standard Fence Pointers:");
    println!("  - Total time: {:.2?}", std_duration);
    println!("  - Time per range query: {:.2} ns", std_ns_per_range);
    println!("  - Total blocks found: {}", std_total_blocks);
    
    println!("Optimized Eytzinger Fence Pointers:");
    println!("  - Total time: {:.2?}", eytzinger_duration);
    println!("  - Time per range query: {:.2} ns", eytzinger_ns_per_range);
    println!("  - Total blocks found: {}", eytzinger_total_blocks);
    println!("  - Improvement over Standard: {:.2}% {}", 
        improvement.abs(),
        if improvement > 0.0 { "faster" } else { "slower" });
    
    // Analyze different range sizes
    println!("\nAnalyzing performance by range size...");
    
    // Group ranges by size
    let mut small_ranges = Vec::new();  // 1-10 items
    let mut medium_ranges = Vec::new(); // 11-100 items
    let mut large_ranges = Vec::new();  // 101+ items
    
    for (start, end) in &ranges {
        let range_size = end - start;
        if range_size <= 10 {
            small_ranges.push((*start, *end));
        } else if range_size <= 100 {
            medium_ranges.push((*start, *end));
        } else {
            large_ranges.push((*start, *end));
        }
    }
    
    // Benchmark small ranges
    if !small_ranges.is_empty() {
        let std_start = Instant::now();
        let mut _std_blocks = 0;
        for (start, end) in &small_ranges {
            let blocks = std_fps.find_blocks_in_range(*start, *end);
            _std_blocks += blocks.len();
        }
        let std_small_duration = std_start.elapsed();
        
        let eytz_start = Instant::now();
        let mut _eytz_blocks = 0;
        for (start, end) in &small_ranges {
            let blocks = eytzinger_fps.find_blocks_in_range(*start, *end);
            _eytz_blocks += blocks.len();
        }
        let eytz_small_duration = eytz_start.elapsed();
        
        let std_ns = std_small_duration.as_nanos() as f64 / small_ranges.len() as f64;
        let eytz_ns = eytz_small_duration.as_nanos() as f64 / small_ranges.len() as f64;
        let small_improvement = (std_ns - eytz_ns) / std_ns * 100.0;
        
        println!("Small ranges (1-10 items, {} ranges):", small_ranges.len());
        println!("  - Standard: {:.2} ns per range", std_ns);
        println!("  - Eytzinger: {:.2} ns per range", eytz_ns);
        println!("  - Improvement: {:.2}% {}", 
            small_improvement.abs(),
            if small_improvement > 0.0 { "faster" } else { "slower" });
    }
    
    // Benchmark medium ranges
    if !medium_ranges.is_empty() {
        let std_start = Instant::now();
        let mut _std_blocks = 0;
        for (start, end) in &medium_ranges {
            let blocks = std_fps.find_blocks_in_range(*start, *end);
            _std_blocks += blocks.len();
        }
        let std_medium_duration = std_start.elapsed();
        
        let eytz_start = Instant::now();
        let mut _eytz_blocks = 0;
        for (start, end) in &medium_ranges {
            let blocks = eytzinger_fps.find_blocks_in_range(*start, *end);
            _eytz_blocks += blocks.len();
        }
        let eytz_medium_duration = eytz_start.elapsed();
        
        let std_ns = std_medium_duration.as_nanos() as f64 / medium_ranges.len() as f64;
        let eytz_ns = eytz_medium_duration.as_nanos() as f64 / medium_ranges.len() as f64;
        let medium_improvement = (std_ns - eytz_ns) / std_ns * 100.0;
        
        println!("Medium ranges (11-100 items, {} ranges):", medium_ranges.len());
        println!("  - Standard: {:.2} ns per range", std_ns);
        println!("  - Eytzinger: {:.2} ns per range", eytz_ns);
        println!("  - Improvement: {:.2}% {}", 
            medium_improvement.abs(),
            if medium_improvement > 0.0 { "faster" } else { "slower" });
    }
    
    // Benchmark large ranges
    if !large_ranges.is_empty() {
        let std_start = Instant::now();
        let mut _std_blocks = 0;
        for (start, end) in &large_ranges {
            let blocks = std_fps.find_blocks_in_range(*start, *end);
            _std_blocks += blocks.len();
        }
        let std_large_duration = std_start.elapsed();
        
        let eytz_start = Instant::now();
        let mut _eytz_blocks = 0;
        for (start, end) in &large_ranges {
            let blocks = eytzinger_fps.find_blocks_in_range(*start, *end);
            _eytz_blocks += blocks.len();
        }
        let eytz_large_duration = eytz_start.elapsed();
        
        let std_ns = std_large_duration.as_nanos() as f64 / large_ranges.len() as f64;
        let eytz_ns = eytz_large_duration.as_nanos() as f64 / large_ranges.len() as f64;
        let large_improvement = (std_ns - eytz_ns) / std_ns * 100.0;
        
        println!("Large ranges (101+ items, {} ranges):", large_ranges.len());
        println!("  - Standard: {:.2} ns per range", std_ns);
        println!("  - Eytzinger: {:.2} ns per range", eytz_ns);
        println!("  - Improvement: {:.2}% {}", 
            large_improvement.abs(),
            if large_improvement > 0.0 { "faster" } else { "slower" });
    }
}