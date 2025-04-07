use criterion::{criterion_group, criterion_main, Criterion};
use lsm_tree::run::{FastLaneFencePointers, StandardFencePointers};
use lsm_tree::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

/// Manual benchmark to directly compare FastLane vs Standard fence pointers
pub fn benchmark_fastlane() {
    println!("\n=== FastLane Fence Pointers Performance Comparison ===");
    
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
    let mut fastlane_fps = FastLaneFencePointers::with_group_size(16);
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        fastlane_fps.add(chunk[0], chunk[1], i);
    }
    
    // Optimize FastLane
    println!("Optimizing FastLane structure...");
    let optimized_fastlane = fastlane_fps.optimize();
    
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
    
    // Calculate time per lookup
    let std_ns_per_lookup = std_duration.as_nanos() as f64 / lookup_count as f64;
    let fastlane_ns_per_lookup = fastlane_duration.as_nanos() as f64 / lookup_count as f64;
    
    // Calculate improvement percentage
    let improvement = (std_ns_per_lookup - fastlane_ns_per_lookup) / std_ns_per_lookup * 100.0;
    
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
    
    println!("Performance Improvement:");
    println!("  - FastLane is {:.2}% {} than Standard", 
        improvement.abs(),
        if improvement > 0.0 { "faster" } else { "slower" });
        
    // Memory comparison
    let std_memory = std::mem::size_of::<StandardFencePointers>() + 
                    std_fps.pointers.capacity() * std::mem::size_of::<(Key, Key, usize)>();
    let fastlane_memory = optimized_fastlane.memory_usage();
    let memory_ratio = fastlane_memory as f64 / std_memory as f64;
    
    println!("\n=== Memory Usage ===");
    println!("Standard: {} bytes", std_memory);
    println!("FastLane: {} bytes", fastlane_memory);
    println!("Memory ratio: {:.2}x", memory_ratio);
    if memory_ratio > 1.0 {
        println!("FastLane uses {:.2}% more memory", (memory_ratio - 1.0) * 100.0);
    } else {
        println!("FastLane uses {:.2}% less memory", (1.0 - memory_ratio) * 100.0);
    }
}

fn criterion_benchmark(_c: &mut Criterion) {
    // Run our manual benchmark
    benchmark_fastlane();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);