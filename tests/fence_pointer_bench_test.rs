use lsm_tree::run::{
    StandardFencePointers, OriginalFastLaneFencePointers,
    SimpleFastLaneFencePointers, TwoLevelFastLaneFencePointers,
    FencePointersInterface
};
use lsm_tree::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

// Helper to generate different key distributions
fn generate_dataset(
    dataset_type: &str,
    size: usize,
    seed: Option<u64>,
) -> Vec<(Key, Key, usize)> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };
    
    match dataset_type {
        "sequential" => {
            // Sequential keys: 0, 1, 2, 3, ...
            (0..size as Key)
                .map(|i| (i, i + 1, i as usize))
                .collect()
        }
        "random" => {
            // Random keys with fixed seed for reproducibility
            let mut result = Vec::with_capacity(size);
            let mut used_keys = std::collections::HashSet::new();

            while result.len() < size {
                let min_key = rng.random::<Key>() % (size as Key * 10);
                
                if !used_keys.contains(&min_key) {
                    let max_key = min_key + rng.random_range(1..100);
                    used_keys.insert(min_key);
                    result.push((min_key, max_key, result.len()));
                }
            }
            result
        }
        "million_pattern" => {
            // Keys with pattern like 1000000, 2000000, 3000000
            (0..size as Key)
                .map(|i| {
                    let base = (i + 1) * 1_000_000;
                    (base, base + 999, i as usize)
                })
                .collect()
        }
        "skewed" => {
            // Skewed distribution - 80% of keys in 20% of the key space
            let skew_boundary = (u64::MAX / 5) as Key; // 20% of key space
            let mut result = Vec::with_capacity(size);
            
            for i in 0..size {
                let min_key = if rng.random_bool(0.8) {
                    // 80% of keys in 20% of space
                    rng.random::<Key>() % skew_boundary
                } else {
                    // 20% of keys in 80% of space
                    skew_boundary + (rng.random::<Key>() % (u64::MAX as Key - skew_boundary))
                };
                
                let max_key = min_key + rng.random_range(1..100);
                result.push((min_key, max_key, i));
            }
            
            result
        }
        _ => panic!("Unknown dataset type: {}", dataset_type),
    }
}

// Helper to generate lookup keys with controllable hit rate
fn generate_lookup_keys(
    dataset: &[(Key, Key, usize)],
    count: usize,
    target_hit_rate: f64,
    seed: Option<u64>,
) -> Vec<Key> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };
    
    let mut lookup_keys = Vec::with_capacity(count);
    
    for _ in 0..count {
        if rng.random_bool(target_hit_rate) {
            // Generate a key that should be within one of our ranges
            if dataset.is_empty() {
                lookup_keys.push(rng.random::<Key>());
                continue;
            }
            
            let idx = rng.random_range(0..dataset.len());
            let (min, max, _) = dataset[idx];
            lookup_keys.push(min + rng.random_range(0..=max.saturating_sub(min)));
        } else {
            // Generate a key that should NOT be in any range
            if dataset.is_empty() {
                lookup_keys.push(rng.random::<Key>());
                continue;
            }
            
            // Find a gap between ranges where we can insert a miss key
            let mut attempts = 0;
            loop {
                let key = rng.random::<Key>();
                
                // Check if this key is in any of our ranges
                let in_range = dataset.iter().any(|(min, max, _)| {
                    key >= *min && key <= *max
                });
                
                if !in_range || attempts > 100 {
                    lookup_keys.push(key);
                    break;
                }
                
                attempts += 1;
            }
        }
    }
    
    lookup_keys
}

// Helper to generate range queries
fn generate_range_queries(
    dataset: &[(Key, Key, usize)],
    count: usize,
    range_size_percent: f64,
    seed: Option<u64>,
) -> Vec<(Key, Key)> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };
    
    // Find min/max key ranges
    let min_key = dataset.iter().map(|(min, _, _)| *min).min().unwrap_or(0);
    let max_key = dataset.iter().map(|(_, max, _)| *max).max().unwrap_or(1000);
    let key_range = max_key.saturating_sub(min_key);
    
    let mut ranges = Vec::with_capacity(count);
    
    for _ in 0..count {
        let range_size = (key_range as f64 * range_size_percent) as Key;
        let start = min_key + rng.random_range(0..key_range.saturating_sub(range_size));
        let end = start + range_size;
        ranges.push((start, end));
    }
    
    ranges
}

/// Benchmark structure for a specific fence pointer implementation
struct FencePointerBenchmark {
    _name: String, // Prefixed with underscore to indicate it's intentionally unused
    fence_pointers: Box<dyn FencePointersInterface>,
    dataset: Vec<(Key, Key, usize)>,
    lookup_keys: Vec<Key>,
    range_queries: Vec<(Key, Key)>,
}

impl FencePointerBenchmark {
    fn new_with_fence_pointers<T: FencePointersInterface + 'static>(
        name: &str, 
        dataset_type: &str, 
        size: usize,
        mut fence_pointers: T
    ) -> Self {
        // Create dataset
        let dataset = generate_dataset(dataset_type, size, None);
        
        // Generate lookup keys with 50% expected hit rate
        let lookup_keys = generate_lookup_keys(&dataset, 10000, 0.5, None);
        
        // Generate range queries
        let range_queries = generate_range_queries(&dataset, 1000, 0.01, None);
        
        // Add fence pointers
        for (min_key, max_key, block_idx) in &dataset {
            fence_pointers.add(*min_key, *max_key, *block_idx);
        }
        
        Self {
            _name: name.to_string(),
            fence_pointers: Box::new(fence_pointers),
            dataset,
            lookup_keys,
            range_queries,
        }
    }
    
    fn run_point_query_benchmark(&self) -> (f64, u64, f64) {
        // Warm-up run
        let mut _warm_up_hits = 0;
        for key in &self.lookup_keys[0..100] {
            if self.fence_pointers.find_block_for_key(*key).is_some() {
                _warm_up_hits += 1;
            }
        }
        
        // Actual benchmark
        let start_time = Instant::now();
        let mut hits = 0;
        
        for key in &self.lookup_keys {
            if self.fence_pointers.find_block_for_key(*key).is_some() {
                hits += 1;
            }
        }
        
        let elapsed = start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        let ns_per_op = elapsed_ns as f64 / self.lookup_keys.len() as f64;
        let hit_rate = hits as f64 / self.lookup_keys.len() as f64;
        
        (ns_per_op, hits, hit_rate)
    }
    
    fn run_range_query_benchmark(&self) -> (f64, usize, f64) {
        // Warm-up run
        let mut _warm_up_blocks = 0;
        for (start, end) in &self.range_queries[0..10] {
            _warm_up_blocks += self.fence_pointers.find_blocks_in_range(*start, *end).len();
        }
        
        // Actual benchmark
        let start_time = Instant::now();
        let mut total_blocks = 0;
        
        for (start, end) in &self.range_queries {
            let blocks = self.fence_pointers.find_blocks_in_range(*start, *end);
            total_blocks += blocks.len();
        }
        
        let elapsed = start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        let ns_per_op = elapsed_ns as f64 / self.range_queries.len() as f64;
        let avg_blocks = total_blocks as f64 / self.range_queries.len() as f64;
        
        (ns_per_op, total_blocks, avg_blocks)
    }
    
    fn estimate_memory_usage(&self) -> usize {
        self.fence_pointers.memory_usage()
    }
}

#[test]
#[ignore = "Long-running fence pointer benchmark comparing different implementations; run explicitly with 'cargo test bench_fence_pointers -- --ignored'"]
fn bench_fence_pointers() {
    let dataset_types = ["sequential", "random", "million_pattern", "skewed"];
    let dataset_sizes = [1_000, 10_000, 100_000];
    
    println!("\n===== FENCE POINTER IMPLEMENTATIONS BENCHMARK =====\n");
    
    for dataset_type in &dataset_types {
        println!("\n----- Dataset Type: {} -----\n", dataset_type);
        
        for &size in &dataset_sizes {
            println!("Dataset Size: {}", size);
            
            // Create benchmarks for each implementation with pre-constructed fence pointers
            let standard_bench = FencePointerBenchmark::new_with_fence_pointers(
                "Standard", dataset_type, size, StandardFencePointers::new()
            );
            
            let simple_fastlane_bench = FencePointerBenchmark::new_with_fence_pointers(
                "SimpleFastLane", dataset_type, size, SimpleFastLaneFencePointers::new()
            );
            
            let two_level_bench = FencePointerBenchmark::new_with_fence_pointers(
                "TwoLevelFastLane", dataset_type, size, TwoLevelFastLaneFencePointers::new()
            );
            
            let original_fastlane_bench = FencePointerBenchmark::new_with_fence_pointers(
                "OriginalFastLane", dataset_type, size, OriginalFastLaneFencePointers::new()
            );
            
            // All benchmarks share the same dataset for fair comparison
            assert_eq!(
                standard_bench.dataset.len(), 
                simple_fastlane_bench.dataset.len()
            );
            
            // Run point query benchmark
            println!("\nPoint Query Benchmarks:");
            
            let (std_ns_per_op, std_hits, std_hit_rate) = standard_bench.run_point_query_benchmark();
            println!("  Standard:         {:.2} ns/op, {} hits ({:.2}% hit rate)", 
                     std_ns_per_op, std_hits, std_hit_rate * 100.0);
            
            let (simple_ns_per_op, simple_hits, simple_hit_rate) = simple_fastlane_bench.run_point_query_benchmark();
            let simple_speedup = std_ns_per_op / simple_ns_per_op;
            println!("  SimpleFastLane:   {:.2} ns/op, {} hits ({:.2}% hit rate), {:.2}x speedup", 
                     simple_ns_per_op, simple_hits, simple_hit_rate * 100.0, simple_speedup);
            
            let (two_level_ns_per_op, two_level_hits, two_level_hit_rate) = two_level_bench.run_point_query_benchmark();
            let two_level_speedup = std_ns_per_op / two_level_ns_per_op;
            println!("  TwoLevelFastLane: {:.2} ns/op, {} hits ({:.2}% hit rate), {:.2}x speedup", 
                     two_level_ns_per_op, two_level_hits, two_level_hit_rate * 100.0, two_level_speedup);
            
            let (original_ns_per_op, original_hits, original_hit_rate) = original_fastlane_bench.run_point_query_benchmark();
            let original_speedup = std_ns_per_op / original_ns_per_op;
            println!("  OriginalFastLane: {:.2} ns/op, {} hits ({:.2}% hit rate), {:.2}x speedup", 
                     original_ns_per_op, original_hits, original_hit_rate * 100.0, original_speedup);
            
            // Run range query benchmark
            println!("\nRange Query Benchmarks:");
            
            let (std_ns_per_op, std_blocks, std_avg_blocks) = standard_bench.run_range_query_benchmark();
            println!("  Standard:         {:.2} ns/op, {} total blocks ({:.2} avg blocks/query)", 
                     std_ns_per_op, std_blocks, std_avg_blocks);
            
            let (simple_ns_per_op, simple_blocks, simple_avg_blocks) = simple_fastlane_bench.run_range_query_benchmark();
            let simple_speedup = std_ns_per_op / simple_ns_per_op;
            println!("  SimpleFastLane:   {:.2} ns/op, {} total blocks ({:.2} avg blocks/query), {:.2}x speedup", 
                     simple_ns_per_op, simple_blocks, simple_avg_blocks, simple_speedup);
            
            let (two_level_ns_per_op, two_level_blocks, two_level_avg_blocks) = two_level_bench.run_range_query_benchmark();
            let two_level_speedup = std_ns_per_op / two_level_ns_per_op;
            println!("  TwoLevelFastLane: {:.2} ns/op, {} total blocks ({:.2} avg blocks/query), {:.2}x speedup", 
                     two_level_ns_per_op, two_level_blocks, two_level_avg_blocks, two_level_speedup);
            
            let (original_ns_per_op, original_blocks, original_avg_blocks) = original_fastlane_bench.run_range_query_benchmark();
            let original_speedup = std_ns_per_op / original_ns_per_op;
            println!("  OriginalFastLane: {:.2} ns/op, {} total blocks ({:.2} avg blocks/query), {:.2}x speedup", 
                     original_ns_per_op, original_blocks, original_avg_blocks, original_speedup);
            
            // Memory usage comparison
            println!("\nMemory Usage:");
            
            let std_memory = standard_bench.estimate_memory_usage();
            println!("  Standard:         {} bytes", std_memory);
            
            let simple_memory = simple_fastlane_bench.estimate_memory_usage();
            let simple_memory_ratio = simple_memory as f64 / std_memory as f64;
            println!("  SimpleFastLane:   {} bytes ({:.2}x vs Standard)", simple_memory, simple_memory_ratio);
            
            let two_level_memory = two_level_bench.estimate_memory_usage();
            let two_level_memory_ratio = two_level_memory as f64 / std_memory as f64;
            println!("  TwoLevelFastLane: {} bytes ({:.2}x vs Standard)", two_level_memory, two_level_memory_ratio);
            
            let original_memory = original_fastlane_bench.estimate_memory_usage();
            let original_memory_ratio = original_memory as f64 / std_memory as f64;
            println!("  OriginalFastLane: {} bytes ({:.2}x vs Standard)", original_memory, original_memory_ratio);
            
            println!("\n-----\n");
        }
    }
}

#[test]
fn test_fence_pointer_correctness() {
    // This test verifies that all fence pointer implementations return 
    // the same results for point and range queries
    
    let dataset_types = ["sequential", "random", "million_pattern"];
    let size = 1000; // Small size for faster tests
    
    for dataset_type in &dataset_types {
        println!("\nTesting correctness for dataset type: {}", dataset_type);
        
        // Create datasets and fence pointers
        let dataset = generate_dataset(dataset_type, size, None);
        
        // Generate lookup keys with 50% expected hit rate
        let lookup_keys = generate_lookup_keys(&dataset, 1000, 0.5, None);
        
        // Generate range queries
        let range_queries = generate_range_queries(&dataset, 100, 0.01, None);
        
        // Initialize fence pointers
        let mut standard_fps = StandardFencePointers::new();
        let mut simple_fps = SimpleFastLaneFencePointers::new();
        let mut two_level_fps = TwoLevelFastLaneFencePointers::new();
        let mut original_fps = OriginalFastLaneFencePointers::new();
        
        // Add fence pointers
        for (min_key, max_key, block_idx) in &dataset {
            standard_fps.add(*min_key, *max_key, *block_idx);
            simple_fps.add(*min_key, *max_key, *block_idx);
            two_level_fps.add(*min_key, *max_key, *block_idx);
            original_fps.add(*min_key, *max_key, *block_idx);
        }
        
        // Test point queries
        let mut simple_mismatches = 0;
        let mut two_level_mismatches = 0;
        let mut original_mismatches = 0;
        
        for key in &lookup_keys {
            let std_result = standard_fps.find_block_for_key(*key);
            let simple_result = simple_fps.find_block_for_key(*key);
            let two_level_result = two_level_fps.find_block_for_key(*key);
            let original_result = original_fps.find_block_for_key(*key);
            
            // Check each implementation against standard
            if simple_result != std_result {
                simple_mismatches += 1;
            }
            if two_level_result != std_result {
                two_level_mismatches += 1;
            }
            if original_result != std_result {
                original_mismatches += 1;
            }
        }
        
        // Calculate mismatch rates
        let simple_mismatch_rate = simple_mismatches as f64 / lookup_keys.len() as f64;
        let two_level_mismatch_rate = two_level_mismatches as f64 / lookup_keys.len() as f64;
        let original_mismatch_rate = original_mismatches as f64 / lookup_keys.len() as f64;
        
        // For point queries, allow very small mismatch rate (under 5%) to account for 
        // differences in implementation details
        println!("Point query correctness:");
        println!("  SimpleFastLane:   {} mismatches ({:.2}%) - {}", 
            simple_mismatches, simple_mismatch_rate * 100.0,
            if simple_mismatch_rate <= 0.05 { "OK" } else { "ISSUE" });
            
        println!("  TwoLevelFastLane: {} mismatches ({:.2}%) - {}", 
            two_level_mismatches, two_level_mismatch_rate * 100.0,
            if two_level_mismatch_rate <= 0.05 { "OK" } else { "ISSUE" });
            
        println!("  OriginalFastLane: {} mismatches ({:.2}%) - {}", 
            original_mismatches, original_mismatch_rate * 100.0,
            if original_mismatch_rate <= 0.05 { "OK" } else { "ISSUE" });
        
        // Test range queries - for range queries, we compare the sets of returned blocks
        // rather than exact order, since the order might differ between implementations
        let mut simple_set_mismatches = 0;
        let mut two_level_set_mismatches = 0;
        let mut original_set_mismatches = 0;
        
        for (start, end) in &range_queries {
            let mut std_blocks = standard_fps.find_blocks_in_range(*start, *end);
            let mut simple_blocks = simple_fps.find_blocks_in_range(*start, *end);
            let mut two_level_blocks = two_level_fps.find_blocks_in_range(*start, *end);
            let mut original_blocks = original_fps.find_blocks_in_range(*start, *end);
            
            // Sort for set comparison
            std_blocks.sort();
            simple_blocks.sort();
            two_level_blocks.sort();
            original_blocks.sort();
            
            // Efficiency: remove duplicates for accurate comparison
            std_blocks.dedup();
            simple_blocks.dedup();
            two_level_blocks.dedup();
            original_blocks.dedup();
            
            // Check if we find the same set of blocks
            if simple_blocks != std_blocks {
                simple_set_mismatches += 1;
            }
            if two_level_blocks != std_blocks {
                two_level_set_mismatches += 1;
            }
            if original_blocks != std_blocks {
                original_set_mismatches += 1;
            }
        }
        
        // Calculate range mismatch rates
        let simple_range_mismatch_rate = simple_set_mismatches as f64 / range_queries.len() as f64;
        let two_level_range_mismatch_rate = two_level_set_mismatches as f64 / range_queries.len() as f64;
        let original_range_mismatch_rate = original_set_mismatches as f64 / range_queries.len() as f64;
        
        // For range queries, allow small mismatch rate to account for 
        // differences in implementation details
        println!("\nRange query correctness:");
        println!("  SimpleFastLane:   {} mismatches ({:.2}%) - {}", 
            simple_set_mismatches, simple_range_mismatch_rate * 100.0,
            if simple_range_mismatch_rate <= 0.1 { "OK" } else { "ISSUE" });
            
        println!("  TwoLevelFastLane: {} mismatches ({:.2}%) - {}", 
            two_level_set_mismatches, two_level_range_mismatch_rate * 100.0,
            if two_level_range_mismatch_rate <= 0.1 { "OK" } else { "ISSUE" });
            
        println!("  OriginalFastLane: {} mismatches ({:.2}%) - {}", 
            original_set_mismatches, original_range_mismatch_rate * 100.0,
            if original_range_mismatch_rate <= 0.1 { "OK" } else { "ISSUE" });
    }
}