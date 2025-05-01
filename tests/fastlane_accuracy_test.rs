use lsm_tree::run::{FastLaneFencePointers, StandardFencePointers};
use lsm_tree::types::Key;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

// Generate sequential keys (0, 1, 2, ...)
fn generate_sequential_keys(count: usize) -> Vec<Key> {
    (0..count as Key).collect()
}

// Generate random keys (randomly distributed across the entire Key range)
fn generate_random_keys(count: usize, seed: u64) -> Vec<Key> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(count);
    
    for _ in 0..count {
        keys.push(rng.random::<Key>());
    }
    
    keys.sort(); // Must be sorted for fence pointers
    keys
}

// Generate grouped keys (million pattern: X000000 + offset)
fn generate_grouped_keys(count: usize, num_groups: usize, seed: u64) -> Vec<Key> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(count);
    
    for _ in 0..count {
        let group = rng.random_range(0..num_groups) as Key;
        let offset = rng.random_range(0..1000) as Key;
        let key = group * 1_000_000 + offset;
        keys.push(key);
    }
    
    keys.sort(); // Must be sorted for fence pointers
    keys
}

// Create lookup keys that may or may not be in the existing keys
fn generate_lookup_keys(original_keys: &[Key], count: usize, hit_ratio: f64, seed: u64) -> Vec<Key> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut lookup_keys = Vec::with_capacity(count);
    
    for _ in 0..count {
        if rng.random_bool(hit_ratio) && !original_keys.is_empty() {
            // Select an existing key (should be a hit)
            let idx = rng.random_range(0..original_keys.len());
            lookup_keys.push(original_keys[idx]);
        } else {
            // Generate a random key (may or may not be a hit)
            lookup_keys.push(rng.random::<Key>());
        }
    }
    
    lookup_keys
}

// Build fence pointers from keys with a given chunk size
fn build_fence_pointers(keys: &[Key], chunk_size: usize) -> (StandardFencePointers, FastLaneFencePointers) {
    let mut standard = StandardFencePointers::new();
    let mut fastlane = FastLaneFencePointers::new();
    
    for (i, chunk) in keys.chunks(chunk_size).enumerate() {
        if chunk.len() < 2 { continue; }
        
        // Use min/max of the chunk
        let min_key = *chunk.iter().min().unwrap();
        let max_key = *chunk.iter().max().unwrap();
        
        standard.add(min_key, max_key, i);
        fastlane.add(min_key, max_key, i);
    }
    
    (standard, fastlane)
}

// Run a benchmark and report performance and hit rates
fn run_benchmark(key_type: &str, keys: Vec<Key>, lookup_count: usize) {
    println!("\n=== FastLane Accuracy Test: {} keys ===", key_type);
    let _key_count = keys.len();
    
    // Build fence pointers with chunk size of 2
    let (standard, fastlane) = build_fence_pointers(&keys, 2);
    
    // First test: lookup keys with 100% theoretical hit rate (keys from the original set)
    let perfect_lookup_keys = generate_lookup_keys(&keys, lookup_count, 1.0, 42);
    
    // Performance and hit rate test for 100% hit rate scenario
    let std_start = Instant::now();
    let mut std_hits = 0;
    for key in &perfect_lookup_keys {
        if standard.find_block_for_key(*key).is_some() {
            std_hits += 1;
        }
    }
    let std_duration = std_start.elapsed();
    
    let fl_start = Instant::now();
    let mut fl_hits = 0;
    for key in &perfect_lookup_keys {
        if fastlane.find_block_for_key(*key).is_some() {
            fl_hits += 1;
        }
    }
    let fl_duration = fl_start.elapsed();
    
    // Calculate performance metrics
    let std_time_per_lookup = std_duration.as_nanos() as f64 / lookup_count as f64;
    let fl_time_per_lookup = fl_duration.as_nanos() as f64 / lookup_count as f64;
    let performance_diff = (std_time_per_lookup - fl_time_per_lookup) / std_time_per_lookup * 100.0;
    
    // Report results for 100% hit rate scenario
    println!("\n== IDEAL SCENARIO (100% of keys should be found) ==");
    println!("STANDARD IMPLEMENTATION:");
    println!("  - Hit rate: {}/{} ({:.2}%)", std_hits, lookup_count, 
             std_hits as f64 / lookup_count as f64 * 100.0);
    println!("  - Time per lookup: {:.2} ns", std_time_per_lookup);
    
    println!("FASTLANE IMPLEMENTATION:");
    println!("  - Hit rate: {}/{} ({:.2}%)", fl_hits, lookup_count, 
             fl_hits as f64 / lookup_count as f64 * 100.0);
    println!("  - Time per lookup: {:.2} ns", fl_time_per_lookup);
    
    println!("PERFORMANCE COMPARISON:");
    if performance_diff > 0.0 {
        println!("  - FastLane is {:.2}% FASTER than Standard", performance_diff);
    } else {
        println!("  - FastLane is {:.2}% SLOWER than Standard", -performance_diff);
    }
    
    // Second test: lookup with mixed keys (50% from original, 50% random)
    let mixed_lookup_keys = generate_lookup_keys(&keys, lookup_count, 0.5, 43);
    
    // Performance and hit rate test for mixed scenario
    let std_start = Instant::now();
    let mut std_hits = 0;
    for key in &mixed_lookup_keys {
        if standard.find_block_for_key(*key).is_some() {
            std_hits += 1;
        }
    }
    let std_duration = std_start.elapsed();
    
    let fl_start = Instant::now();
    let mut fl_hits = 0;
    for key in &mixed_lookup_keys {
        if fastlane.find_block_for_key(*key).is_some() {
            fl_hits += 1;
        }
    }
    let fl_duration = fl_start.elapsed();
    
    // Calculate performance metrics
    let std_time_per_lookup = std_duration.as_nanos() as f64 / lookup_count as f64;
    let fl_time_per_lookup = fl_duration.as_nanos() as f64 / lookup_count as f64;
    let performance_diff = (std_time_per_lookup - fl_time_per_lookup) / std_time_per_lookup * 100.0;
    
    // Report results for mixed scenario
    println!("\n== REALISTIC SCENARIO (mix of existing and new keys) ==");
    println!("STANDARD IMPLEMENTATION:");
    println!("  - Hit rate: {}/{} ({:.2}%)", std_hits, lookup_count, 
             std_hits as f64 / lookup_count as f64 * 100.0);
    println!("  - Time per lookup: {:.2} ns", std_time_per_lookup);
    
    println!("FASTLANE IMPLEMENTATION:");
    println!("  - Hit rate: {}/{} ({:.2}%)", fl_hits, lookup_count, 
             fl_hits as f64 / lookup_count as f64 * 100.0);
    println!("  - Time per lookup: {:.2} ns", fl_time_per_lookup);
    
    println!("PERFORMANCE COMPARISON:");
    if performance_diff > 0.0 {
        println!("  - FastLane is {:.2}% FASTER than Standard", performance_diff);
    } else {
        println!("  - FastLane is {:.2}% SLOWER than Standard", -performance_diff);
    }
    
    // Additional metrics
    println!("\n== MEMORY USAGE ==");
    // Standard memory calculation
    let std_memory = std::mem::size_of::<StandardFencePointers>() + 
                     (standard.len() * std::mem::size_of::<(Key, Key, usize)>());
    
    // FastLane memory usage
    let fl_memory = std::mem::size_of::<FastLaneFencePointers>() +
                    fastlane.groups.iter().map(|g| {
                        std::mem::size_of::<u64>() + // common_bits_mask
                        std::mem::size_of::<u8>() +  // num_shared_bits
                        g.min_key_lane.capacity() * std::mem::size_of::<u64>() +
                        g.max_key_lane.capacity() * std::mem::size_of::<u64>() +
                        g.block_idx_lane.capacity() * std::mem::size_of::<usize>()
                    }).sum::<usize>();
    
    println!("STANDARD: {} bytes", std_memory);
    println!("FASTLANE: {} bytes", fl_memory);
    
    let memory_ratio = fl_memory as f64 / std_memory as f64;
    println!("MEMORY RATIO: {:.2}x", memory_ratio);
    
    if memory_ratio > 1.0 {
        println!("FastLane uses {:.2}% more memory", (memory_ratio - 1.0) * 100.0);
    } else {
        println!("FastLane uses {:.2}% less memory", (1.0 - memory_ratio) * 100.0);
    }
    
    // Structure info
    println!("\n== STRUCTURE INFO ==");
    println!("STANDARD: {} fence pointers", standard.len());
    println!("FASTLANE: {} total fence pointers in {} groups", 
             fastlane.len(), fastlane.groups.len());
    println!("Average group size: {:.2} entries", 
             fastlane.len() as f64 / fastlane.groups.len().max(1) as f64);
}

#[test]
fn test_fastlane_accuracy_sequential() {
    // Generate 100K sequential keys
    let keys = generate_sequential_keys(100_000);
    run_benchmark("sequential", keys, 10_000);
}

#[test]
fn test_fastlane_accuracy_random() {
    // Generate 100K random keys
    let keys = generate_random_keys(100_000, 42);
    run_benchmark("random", keys, 10_000);
}

#[test]
fn test_fastlane_accuracy_grouped() {
    // Generate 100K grouped keys with 100 groups
    let keys = generate_grouped_keys(100_000, 100, 42);
    run_benchmark("grouped", keys, 10_000);
}