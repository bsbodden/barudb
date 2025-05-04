use lsm_tree::run::{AdaptiveFastLanePointers, StandardFencePointers, EytzingerFencePointers};
use lsm_tree::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

fn main() {
    println!("\n=== Adaptive Fence Pointers Performance Benchmark ===");
    benchmark_adaptive_performance();
}

/// Test the adaptive fence pointers implementation with different dataset sizes
/// to verify that it selects the best implementation for each situation
fn benchmark_adaptive_performance() {
    println!("Testing adaptive implementation across different dataset sizes...\n");
    
    // Test with various dataset sizes
    let sizes = [1_000, 10_000, 100_000, 500_000, 1_000_000];
    let lookup_count = 100_000;
    
    println!("{:<10} {:<10} {:<10} {:<10} {:<15} {:<15} {:<15}", 
        "Size", "Std (ns)", "Eytzinger (ns)", "Adaptive (ns)", 
        "Adaptive vs Std", "Adaptive vs Eytz", "Impl Choice");
    
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
        
        // Build Eytzinger fence pointers
        let mut eytzinger_fps = EytzingerFencePointers::new();
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 { continue; }
            eytzinger_fps.add(chunk[0], chunk[1], i);
        }
        
        // Build Adaptive fence pointers
        let mut adaptive_fps = AdaptiveFastLanePointers::new();
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 { continue; }
            adaptive_fps.add(chunk[0], chunk[1], i);
        }
        
        // Optimize structures
        eytzinger_fps.optimize();
        adaptive_fps.optimize();
        
        // Benchmark Standard fence pointers
        let std_start = Instant::now();
        for key in &lookup_keys {
            let _ = std_fps.find_block_for_key(*key);
        }
        let std_duration = std_start.elapsed();
        let std_ns_per_lookup = std_duration.as_nanos() as f64 / lookup_count as f64;
        
        // Benchmark Eytzinger fence pointers
        let eytzinger_start = Instant::now();
        for key in &lookup_keys {
            let _ = eytzinger_fps.find_block_for_key(*key);
        }
        let eytzinger_duration = eytzinger_start.elapsed();
        let eytzinger_ns_per_lookup = eytzinger_duration.as_nanos() as f64 / lookup_count as f64;
        
        // Benchmark Adaptive fence pointers
        let adaptive_start = Instant::now();
        for key in &lookup_keys {
            let _ = adaptive_fps.find_block_for_key(*key);
        }
        let adaptive_duration = adaptive_start.elapsed();
        let adaptive_ns_per_lookup = adaptive_duration.as_nanos() as f64 / lookup_count as f64;
        
        // Calculate improvement percentages
        let adaptive_vs_std = (std_ns_per_lookup - adaptive_ns_per_lookup) / std_ns_per_lookup * 100.0;
        let adaptive_vs_eytzinger = (eytzinger_ns_per_lookup - adaptive_ns_per_lookup) / eytzinger_ns_per_lookup * 100.0;
        
        // Determine which implementation the adaptive one would choose (based on threshold=500,000)
        let impl_choice = if size >= 500_000 { "Eytzinger" } else { "Standard" };
        
        // Check if the adaptive implementation is making the right choice
        let actually_better = if impl_choice == "Eytzinger" {
            eytzinger_ns_per_lookup < std_ns_per_lookup
        } else {
            std_ns_per_lookup < eytzinger_ns_per_lookup
        };
        
        // Mark the impl_choice with ! if we're getting the decision wrong
        let impl_choice_marked = if actually_better {
            impl_choice.to_string()
        } else {
            format!("{}!", impl_choice)
        };
        
        println!("{:<10} {:<10.2} {:<10.2} {:<10.2} {:<15.2}% {:<15.2}% {:<15}", 
            size, std_ns_per_lookup, eytzinger_ns_per_lookup, adaptive_ns_per_lookup,
            adaptive_vs_std, adaptive_vs_eytzinger, impl_choice_marked);
    }
    
    // Now benchmark range queries with different sizes
    println!("\nTesting adaptive implementation for range queries...\n");
    
    let range_count = 1_000;
    let range_size = 100; // Average range size
    
    println!("{:<10} {:<10} {:<10} {:<10} {:<15} {:<15} {:<15}", 
        "Size", "Std (ns)", "Eytzinger (ns)", "Adaptive (ns)", 
        "Adaptive vs Std", "Adaptive vs Eytz", "Impl Choice");
    
    for &size in &sizes {
        // Generate sequential keys
        let keys: Vec<Key> = (0..size as Key).collect();
        
        // Generate range queries
        let mut rng = StdRng::seed_from_u64(42);
        let ranges: Vec<(Key, Key)> = (0..range_count)
            .map(|_| {
                let start = rng.random_range(0..size as Key - range_size);
                let end = start + rng.random_range(1..range_size);
                (start, end)
            })
            .collect();
        
        // Build Standard fence pointers
        let mut std_fps = StandardFencePointers::new();
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 { continue; }
            std_fps.add(chunk[0], chunk[1], i);
        }
        
        // Build Eytzinger fence pointers
        let mut eytzinger_fps = EytzingerFencePointers::new();
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 { continue; }
            eytzinger_fps.add(chunk[0], chunk[1], i);
        }
        
        // Build Adaptive fence pointers
        let mut adaptive_fps = AdaptiveFastLanePointers::new();
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 { continue; }
            adaptive_fps.add(chunk[0], chunk[1], i);
        }
        
        // Optimize structures
        eytzinger_fps.optimize();
        adaptive_fps.optimize();
        
        // Benchmark Standard fence pointers
        let std_start = Instant::now();
        let mut std_total_blocks = 0;
        for (start, end) in &ranges {
            let blocks = std_fps.find_blocks_in_range(*start, *end);
            std_total_blocks += blocks.len();
        }
        let std_duration = std_start.elapsed();
        let std_ns_per_range = std_duration.as_nanos() as f64 / range_count as f64;
        
        // Benchmark Eytzinger fence pointers
        let eytzinger_start = Instant::now();
        let mut eytzinger_total_blocks = 0;
        for (start, end) in &ranges {
            let blocks = eytzinger_fps.find_blocks_in_range(*start, *end);
            eytzinger_total_blocks += blocks.len();
        }
        let eytzinger_duration = eytzinger_start.elapsed();
        let eytzinger_ns_per_range = eytzinger_duration.as_nanos() as f64 / range_count as f64;
        
        // Benchmark Adaptive fence pointers
        let adaptive_start = Instant::now();
        let mut adaptive_total_blocks = 0;
        for (start, end) in &ranges {
            let blocks = adaptive_fps.find_blocks_in_range(*start, *end);
            adaptive_total_blocks += blocks.len();
        }
        let adaptive_duration = adaptive_start.elapsed();
        let adaptive_ns_per_range = adaptive_duration.as_nanos() as f64 / range_count as f64;
        
        // Check if block counts match
        if std_total_blocks != adaptive_total_blocks || eytzinger_total_blocks != adaptive_total_blocks {
            println!("WARNING: Block counts don't match! Std: {}, Eytzinger: {}, Adaptive: {}", 
                     std_total_blocks, eytzinger_total_blocks, adaptive_total_blocks);
        }
        
        // Calculate improvement percentages
        let adaptive_vs_std = (std_ns_per_range - adaptive_ns_per_range) / std_ns_per_range * 100.0;
        let adaptive_vs_eytzinger = (eytzinger_ns_per_range - adaptive_ns_per_range) / eytzinger_ns_per_range * 100.0;
        
        // Determine which implementation the adaptive one would choose (based on threshold=500,000)
        let impl_choice = if size >= 500_000 { "Eytzinger" } else { "Standard" };
        
        // Check if the adaptive implementation is making the right choice
        let actually_better = if impl_choice == "Eytzinger" {
            eytzinger_ns_per_range < std_ns_per_range
        } else {
            std_ns_per_range < eytzinger_ns_per_range
        };
        
        // Mark the impl_choice with ! if we're getting the decision wrong
        let impl_choice_marked = if actually_better {
            impl_choice.to_string()
        } else {
            format!("{}!", impl_choice)
        };
        
        println!("{:<10} {:<10.2} {:<10.2} {:<10.2} {:<15.2}% {:<15.2}% {:<15}", 
            size, std_ns_per_range, eytzinger_ns_per_range, adaptive_ns_per_range,
            adaptive_vs_std, adaptive_vs_eytzinger, impl_choice_marked);
    }
}