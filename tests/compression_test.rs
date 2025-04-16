use lsm_tree::lsm_tree::{LSMConfig, LSMTree};
use lsm_tree::run::compression::{
    CompressionConfig, CompressionType, AdaptiveCompressionConfig, CompressionFactory
};
use lsm_tree::types::{Key, Value, CompactionPolicyType};
use std::time::{Duration, Instant};
use tempfile::tempdir;
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Generate sequential keys and values
fn generate_sequential_data(count: usize) -> Vec<(Key, Value)> {
    (0..count as Key).map(|i| (i, i * 10)).collect()
}

/// Generate random keys and values with fixed seed for reproducibility
fn generate_random_data(count: usize, seed: u64) -> Vec<(Key, Value)> {
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Use a more restricted range to avoid overflow issues in tests
    let mut keys = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);
    
    // Generate keys and values separately
    for _ in 0..count {
        keys.push(rng.gen_range(-100_000..100_000));
        values.push(rng.gen_range(-100_000..100_000));
    }
    
    // Ensure unique keys by deduplicating
    keys.sort();
    keys.dedup();
    
    // If we lost too many keys during deduplication, fill back up
    while keys.len() < count {
        let new_key = rng.gen_range(-100_000..100_000);
        if !keys.contains(&new_key) {
            keys.push(new_key);
        }
    }
    
    // Make sure values match back up with keys
    keys.sort();
    
    // Create pairs
    let mut data = Vec::with_capacity(count);
    for i in 0..count.min(keys.len()) {
        data.push((keys[i], values[i]));
    }
    
    // Final sort by key
    data.sort_by_key(|&(k, _)| k);
    data
}

/// Generate keys with repeated patterns (good for dictionary compression)
fn generate_repeated_data(count: usize) -> Vec<(Key, Value)> {
    let mut data = Vec::with_capacity(count);
    // Ensure at least 5 unique patterns
    let pattern_count = std::cmp::max(5, count / 20); 
    
    for i in 0..count {
        // Use modulo to create repetition in the data
        let pattern_idx = i % pattern_count;
        data.push((pattern_idx as Key, (pattern_idx * 10) as Value));
    }
    data
}

/// Generate keys with small deltas (good for delta compression)
fn generate_delta_data(count: usize) -> Vec<(Key, Value)> {
    let mut data = Vec::with_capacity(count);
    let mut rng = StdRng::seed_from_u64(42);
    
    let mut key = 0;
    let mut value = 0;
    
    for _ in 0..count {
        // Small deltas (1-5) between consecutive keys
        key += rng.gen_range(1..=5);
        value += rng.gen_range(1..=10);
        data.push((key, value));
    }
    data
}

/// Generate keys with small range (good for bit-packing)
fn generate_small_range_data(count: usize) -> Vec<(Key, Value)> {
    let mut rng = StdRng::seed_from_u64(43);
    let mut data: Vec<(Key, Value)> = (0..count)
        .map(|_| (rng.gen_range(0..1000) as Key, rng.gen_range(0..1000) as Value))
        .collect();
    data.sort_by_key(|&(k, _)| k);
    data
}

/// Create a temporary LSM tree with the specified compression configuration
fn create_lsm_tree(compression_config: CompressionConfig, 
                  adaptive_config: AdaptiveCompressionConfig) -> (LSMTree, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    
    let config = LSMConfig {
        buffer_size: 128,
        storage_path: temp_dir.path().to_path_buf(),
        compaction_policy: CompactionPolicyType::Tiered,
        compaction_threshold: 4,
        compression: compression_config,
        adaptive_compression: adaptive_config,
        collect_compression_stats: true,
        ..Default::default()
    };
    
    (LSMTree::with_config(config), temp_dir)
}

/// Benchmark LSM tree operations with the given data
fn benchmark_operations(
    lsm: &mut LSMTree, 
    data: &[(Key, Value)], 
    name: &str
) -> HashMap<String, Duration> {
    let mut results = HashMap::new();
    
    // Insert data
    let start = Instant::now();
    for &(key, value) in data {
        lsm.put(key, value).unwrap();
    }
    results.insert(format!("{}_put", name), start.elapsed());
    
    // Flush to trigger compression
    let start = Instant::now();
    lsm.flush_buffer_to_level0().unwrap();
    results.insert(format!("{}_flush", name), start.elapsed());
    
    // Get some values
    let start = Instant::now();
    for &(key, _) in data.iter().step_by(10) {
        let value = lsm.get(key);
        // Just verify we can retrieve the key
        assert!(value.is_some(), "Value for key {} not found", key);
    }
    results.insert(format!("{}_get", name), start.elapsed());
    
    // Range query
    if !data.is_empty() {
        let min_key = data.first().unwrap().0;
        let max_key = data.last().unwrap().0;
        
        // Avoid overflow by handling different cases
        let range_size = if max_key > min_key {
            (max_key - min_key) / 10
        } else {
            10 // Default size if min and max are too close
        };
        
        if range_size > 0 {
            let start = Instant::now();
            
            // Ensure we don't overflow when computing range bounds
            let range_start = if i64::MAX - (range_size * 2) <= min_key {
                min_key // Can't add without overflow
            } else {
                min_key + range_size * 2
            };
            
            let range_end = if i64::MAX - (range_size * 4) <= min_key {
                max_key // Can't add without overflow
            } else {
                min_key + range_size * 4
            };
            
            let range_result = lsm.range(range_start, range_end);
            results.insert(format!("{}_range", name), start.elapsed());
            
            // Verify range results
            for (key, _) in &range_result {
                assert!(*key >= range_start && *key < range_end, 
                       "Key {} is outside range [{}, {})", key, range_start, range_end);
                
                // Don't verify exact values since they may have changed during compaction
                // Just verify the key is returned within the range
            }
        }
    }
    
    results
}

/// Evaluate a single compression type with different data patterns
fn evaluate_compression_type(compression_type: CompressionType) {
    println!("\n=== Testing {:?} Compression ===", compression_type);
    
    // Create compression config
    let compression_config = CompressionConfig {
        enabled: true,
        l0_default: compression_type,
        lower_level_default: compression_type,
        level_types: vec![Some(compression_type); 10],
        ..Default::default()
    };
    
    let adaptive_config = AdaptiveCompressionConfig {
        enabled: false,
        ..Default::default()
    };
    
    // Create LSM trees for different data patterns
    let (mut sequential_lsm, _dir1) = create_lsm_tree(compression_config.clone(), adaptive_config.clone());
    let (mut random_lsm, _dir2) = create_lsm_tree(compression_config.clone(), adaptive_config.clone());
    let (mut repeated_lsm, _dir3) = create_lsm_tree(compression_config.clone(), adaptive_config.clone());
    let (mut delta_lsm, _dir4) = create_lsm_tree(compression_config.clone(), adaptive_config.clone());
    let (mut small_range_lsm, _dir5) = create_lsm_tree(compression_config, adaptive_config);
    
    // Generate test data
    let data_size = 10000;
    let sequential_data = generate_sequential_data(data_size);
    let random_data = generate_random_data(data_size, 42);
    let repeated_data = generate_repeated_data(data_size);
    let delta_data = generate_delta_data(data_size);
    let small_range_data = generate_small_range_data(data_size);
    
    // Benchmark each data pattern
    let sequential_results = benchmark_operations(&mut sequential_lsm, &sequential_data, "sequential");
    let random_results = benchmark_operations(&mut random_lsm, &random_data, "random");
    let repeated_results = benchmark_operations(&mut repeated_lsm, &repeated_data, "repeated");
    let delta_results = benchmark_operations(&mut delta_lsm, &delta_data, "delta");
    let small_range_results = benchmark_operations(&mut small_range_lsm, &small_range_data, "small_range");
    
    // Print results
    println!("Sequential Data Performance:");
    println!("  Put: {:?}", sequential_results.get("sequential_put").unwrap());
    println!("  Flush: {:?}", sequential_results.get("sequential_flush").unwrap());
    println!("  Get: {:?}", sequential_results.get("sequential_get").unwrap());
    if sequential_results.contains_key("sequential_range") {
        println!("  Range: {:?}", sequential_results.get("sequential_range").unwrap());
    }
    
    println!("Random Data Performance:");
    println!("  Put: {:?}", random_results.get("random_put").unwrap());
    println!("  Flush: {:?}", random_results.get("random_flush").unwrap());
    println!("  Get: {:?}", random_results.get("random_get").unwrap());
    if random_results.contains_key("random_range") {
        println!("  Range: {:?}", random_results.get("random_range").unwrap());
    }
    
    println!("Repeated Data Performance:");
    println!("  Put: {:?}", repeated_results.get("repeated_put").unwrap());
    println!("  Flush: {:?}", repeated_results.get("repeated_flush").unwrap());
    println!("  Get: {:?}", repeated_results.get("repeated_get").unwrap());
    if repeated_results.contains_key("repeated_range") {
        println!("  Range: {:?}", repeated_results.get("repeated_range").unwrap());
    }
    
    println!("Delta-friendly Data Performance:");
    println!("  Put: {:?}", delta_results.get("delta_put").unwrap());
    println!("  Flush: {:?}", delta_results.get("delta_flush").unwrap());
    println!("  Get: {:?}", delta_results.get("delta_get").unwrap());
    if delta_results.contains_key("delta_range") {
        println!("  Range: {:?}", delta_results.get("delta_range").unwrap());
    }
    
    println!("Small-range Data Performance:");
    println!("  Put: {:?}", small_range_results.get("small_range_put").unwrap());
    println!("  Flush: {:?}", small_range_results.get("small_range_flush").unwrap());
    println!("  Get: {:?}", small_range_results.get("small_range_get").unwrap());
    if small_range_results.contains_key("small_range_range") {
        println!("  Range: {:?}", small_range_results.get("small_range_range").unwrap());
    }
}

/// Evaluate adaptive compression with different data patterns
fn evaluate_adaptive_compression() {
    println!("\n=== Testing Adaptive Compression ===");
    
    // Create compression config
    let compression_config = CompressionConfig {
        enabled: true,
        ..Default::default()
    };
    
    let adaptive_config = AdaptiveCompressionConfig {
        enabled: true,
        level_aware: true,
        ..Default::default()
    };
    
    // Create LSM trees for different data patterns
    let (mut mixed_lsm, _dir) = create_lsm_tree(compression_config, adaptive_config);
    
    // Generate mixed test data with patterns suitable for different algorithms
    let sequential_data = generate_sequential_data(2000);
    let random_data = generate_random_data(2000, 44);
    let repeated_data = generate_repeated_data(2000);
    let delta_data = generate_delta_data(2000);
    let small_range_data = generate_small_range_data(2000);
    
    // Combine all data patterns into one dataset
    let mut mixed_data = Vec::new();
    mixed_data.extend_from_slice(&sequential_data);
    mixed_data.extend_from_slice(&random_data);
    mixed_data.extend_from_slice(&repeated_data);
    mixed_data.extend_from_slice(&delta_data);
    mixed_data.extend_from_slice(&small_range_data);
    
    // Sort by key for LSM tree
    mixed_data.sort_by_key(|&(k, _)| k);
    
    // Benchmark with mixed data
    let mixed_results = benchmark_operations(&mut mixed_lsm, &mixed_data, "mixed");
    
    // Print results
    println!("Mixed Data Performance with Adaptive Compression:");
    println!("  Put: {:?}", mixed_results.get("mixed_put").unwrap());
    println!("  Flush: {:?}", mixed_results.get("mixed_flush").unwrap());
    println!("  Get: {:?}", mixed_results.get("mixed_get").unwrap());
    if mixed_results.contains_key("mixed_range") {
        println!("  Range: {:?}", mixed_results.get("mixed_range").unwrap());
    }
}

/// Compare compression ratios for different strategies
fn compare_compression_ratios() {
    println!("\n=== Compression Ratio Comparison ===");
    
    // Generate test data
    let data_size = 10000;
    let sequential_data = generate_sequential_data(data_size);
    let random_data = generate_random_data(data_size, 45);
    let repeated_data = generate_repeated_data(data_size);
    let delta_data = generate_delta_data(data_size);
    let small_range_data = generate_small_range_data(data_size);
    
    // Convert to bytes for direct compression
    let sequential_bytes = data_to_bytes(&sequential_data);
    let random_bytes = data_to_bytes(&random_data);
    let repeated_bytes = data_to_bytes(&repeated_data);
    let delta_bytes = data_to_bytes(&delta_data);
    let small_range_bytes = data_to_bytes(&small_range_data);
    
    // Create compression strategies directly, not as boxed traits
    // Measure each type of compression separately
    println!("\nSequential Data ({} bytes):", sequential_bytes.len());
    measure_compression_type("Delta", &sequential_bytes, CompressionType::Delta);
    measure_compression_type("BitPack", &sequential_bytes, CompressionType::BitPack);
    measure_compression_type("Dictionary", &sequential_bytes, CompressionType::Dictionary);
    measure_compression_type("Lz4", &sequential_bytes, CompressionType::Lz4);
    measure_compression_type("Snappy", &sequential_bytes, CompressionType::Snappy);
    
    println!("\nRandom Data ({} bytes):", random_bytes.len());
    measure_compression_type("Delta", &random_bytes, CompressionType::Delta);
    measure_compression_type("BitPack", &random_bytes, CompressionType::BitPack);
    measure_compression_type("Dictionary", &random_bytes, CompressionType::Dictionary);
    measure_compression_type("Lz4", &random_bytes, CompressionType::Lz4);
    measure_compression_type("Snappy", &random_bytes, CompressionType::Snappy);
    
    println!("\nRepeated Data ({} bytes):", repeated_bytes.len());
    measure_compression_type("Delta", &repeated_bytes, CompressionType::Delta);
    measure_compression_type("BitPack", &repeated_bytes, CompressionType::BitPack);
    measure_compression_type("Dictionary", &repeated_bytes, CompressionType::Dictionary);
    measure_compression_type("Lz4", &repeated_bytes, CompressionType::Lz4);
    measure_compression_type("Snappy", &repeated_bytes, CompressionType::Snappy);
    
    println!("\nDelta-friendly Data ({} bytes):", delta_bytes.len());
    measure_compression_type("Delta", &delta_bytes, CompressionType::Delta);
    measure_compression_type("BitPack", &delta_bytes, CompressionType::BitPack);
    measure_compression_type("Dictionary", &delta_bytes, CompressionType::Dictionary);
    measure_compression_type("Lz4", &delta_bytes, CompressionType::Lz4);
    measure_compression_type("Snappy", &delta_bytes, CompressionType::Snappy);
    
    println!("\nSmall-range Data ({} bytes):", small_range_bytes.len());
    measure_compression_type("Delta", &small_range_bytes, CompressionType::Delta);
    measure_compression_type("BitPack", &small_range_bytes, CompressionType::BitPack);
    measure_compression_type("Dictionary", &small_range_bytes, CompressionType::Dictionary);
    measure_compression_type("Lz4", &small_range_bytes, CompressionType::Lz4);
    measure_compression_type("Snappy", &small_range_bytes, CompressionType::Snappy);
}

/// Measure compression performance for a specific compression type
fn measure_compression_type(
    name: &str,
    data: &[u8],
    compression_type: CompressionType
) {
    // Create the compression strategy
    let compression = CompressionFactory::create(compression_type);
    
    // Measure compression
    let start = Instant::now();
    let compressed = compression.compress(data).unwrap();
    let compress_time = start.elapsed();
    
    // Measure decompression
    let start = Instant::now();
    let decompressed = compression.decompress(&compressed).unwrap();
    let decompress_time = start.elapsed();
    
    // Verify decompression is correct
    assert_eq!(data, decompressed.as_slice(), "Decompressed data doesn't match original data");
    
    // Calculate compression ratio
    let ratio = data.len() as f64 / compressed.len() as f64;
    
    println!("  {} compression: {} bytes, ratio: {:.2}x, compress: {:?}, decompress: {:?}",
             name,
             compressed.len(),
             ratio,
             compress_time,
             decompress_time);
}

/// Convert key-value pairs to bytes
fn data_to_bytes(data: &[(Key, Value)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 16);
    for &(key, value) in data {
        bytes.extend_from_slice(&key.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Test with workload that stresses the adaptive compression
fn test_adaptive_workload() {
    println!("\n=== Adaptive Compression Workload Test ===");
    
    // Create LSM trees with different configurations
    let compression_config_none = CompressionConfig {
        enabled: true,
        l0_default: CompressionType::None,
        lower_level_default: CompressionType::None,
        level_types: vec![Some(CompressionType::None); 10],
        ..Default::default()
    };
    
    let compression_config_bitpack = CompressionConfig {
        enabled: true,
        l0_default: CompressionType::BitPack,
        lower_level_default: CompressionType::BitPack,
        level_types: vec![Some(CompressionType::BitPack); 10],
        ..Default::default()
    };
    
    let compression_config_adaptive = CompressionConfig {
        enabled: true,
        ..Default::default()
    };
    
    let adaptive_config = AdaptiveCompressionConfig {
        enabled: true,
        level_aware: true,
        ..Default::default()
    };
    
    let (mut none_lsm, _dir1) = create_lsm_tree(
        compression_config_none, 
        AdaptiveCompressionConfig { enabled: false, ..Default::default() }
    );
    
    let (mut bitpack_lsm, _dir2) = create_lsm_tree(
        compression_config_bitpack, 
        AdaptiveCompressionConfig { enabled: false, ..Default::default() }
    );
    
    let (mut adaptive_lsm, _dir3) = create_lsm_tree(
        compression_config_adaptive, 
        adaptive_config
    );
    
    // Create a workload that changes over time
    let mut workload = Vec::new();
    
    // Phase 1: Small range integers (good for BitPack)
    workload.extend_from_slice(&generate_small_range_data(5000));
    
    // Phase 2: Sequential data with small deltas (good for Delta)
    workload.extend_from_slice(&generate_delta_data(5000));
    
    // Phase 3: Repeated values (good for Dictionary)
    workload.extend_from_slice(&generate_repeated_data(5000));
    
    // Phase 4: Random data (challenging for compression)
    workload.extend_from_slice(&generate_random_data(5000, 46));
    
    // Sort the entire workload by key
    workload.sort_by_key(|&(k, _)| k);
    
    // Benchmark each LSM tree
    let none_results = benchmark_operations(&mut none_lsm, &workload, "none");
    let bitpack_results = benchmark_operations(&mut bitpack_lsm, &workload, "bitpack");
    let adaptive_results = benchmark_operations(&mut adaptive_lsm, &workload, "adaptive");
    
    // Print results
    println!("No Compression Performance:");
    println!("  Put: {:?}", none_results.get("none_put").unwrap());
    println!("  Flush: {:?}", none_results.get("none_flush").unwrap());
    println!("  Get: {:?}", none_results.get("none_get").unwrap());
    if none_results.contains_key("none_range") {
        println!("  Range: {:?}", none_results.get("none_range").unwrap());
    }
    
    println!("BitPack Compression Performance:");
    println!("  Put: {:?}", bitpack_results.get("bitpack_put").unwrap());
    println!("  Flush: {:?}", bitpack_results.get("bitpack_flush").unwrap());
    println!("  Get: {:?}", bitpack_results.get("bitpack_get").unwrap());
    if bitpack_results.contains_key("bitpack_range") {
        println!("  Range: {:?}", bitpack_results.get("bitpack_range").unwrap());
    }
    
    println!("Adaptive Compression Performance:");
    println!("  Put: {:?}", adaptive_results.get("adaptive_put").unwrap());
    println!("  Flush: {:?}", adaptive_results.get("adaptive_flush").unwrap());
    println!("  Get: {:?}", adaptive_results.get("adaptive_get").unwrap());
    if adaptive_results.contains_key("adaptive_range") {
        println!("  Range: {:?}", adaptive_results.get("adaptive_range").unwrap());
    }
}

#[test]
#[ignore = "Long-running benchmark test; run explicitly with 'cargo test --test compression_test -- --ignored'"]
fn test_compression_strategies() {
    // Compare direct compression ratios without LSM tree overhead
    compare_compression_ratios();
    
    // Test all compression types with the LSM tree
    evaluate_compression_type(CompressionType::None);
    evaluate_compression_type(CompressionType::Delta);
    evaluate_compression_type(CompressionType::BitPack);
    evaluate_compression_type(CompressionType::Dictionary);
    evaluate_compression_type(CompressionType::Lz4);
    evaluate_compression_type(CompressionType::Snappy);
    
    // Test adaptive compression
    evaluate_adaptive_compression();
    
    // Test with a changing workload
    test_adaptive_workload();
    
    println!("\nCompression comparison complete. Review the output to determine the best compression strategy for your workload.");
}

#[test]
fn test_all_compression_types() {
    // Test all compression types with LSM tree integration
    println!("\n=== Testing All Compression Types ===");
    
    for compression_type in [
        CompressionType::None,
        CompressionType::BitPack,
        CompressionType::Delta,
        CompressionType::Dictionary,
        CompressionType::Lz4,
        CompressionType::Snappy,
    ] {
        println!("\n--- Testing {:?} Compression ---", compression_type);
        
        // Create configuration with current compression type
        let compression_config = CompressionConfig {
            enabled: true,
            l0_default: compression_type,
            lower_level_default: compression_type,
            level_types: vec![Some(compression_type); 10],
            ..Default::default()
        };
        
        let adaptive_config = AdaptiveCompressionConfig {
            enabled: false,
            ..Default::default()
        };
        
        // Create LSM tree with this compression type
        let (mut lsm, _dir) = create_lsm_tree(compression_config, adaptive_config);
        
        // Use data optimized for each compression type
        let test_data = match compression_type {
            CompressionType::BitPack => generate_small_range_data(10),
            CompressionType::Delta => generate_delta_data(10),
            CompressionType::Dictionary => generate_repeated_data(10),
            _ => generate_sequential_data(10),
        };
        
        println!("Dataset: {} key-value pairs", test_data.len());
        
        // Insert data
        for &(key, value) in &test_data {
            lsm.put(key, value).unwrap();
        }
        
        // Flush to trigger compression
        lsm.flush_buffer_to_level0().unwrap();
        
        // Verify data retrievability
        for &(key, expected_value) in &test_data {
            let value = lsm.get(key);
            assert_eq!(value, Some(expected_value), "Failed to retrieve key {}", key);
        }
        
        println!("{:?} compression test passed successfully", compression_type);
    }
}

#[test]
fn test_basic_compression() {
    // Test direct compression first
    // This tests the underlying compression algorithms directly without LSM tree overhead
    let sequential_data = generate_sequential_data(100);
    println!("Testing direct compression of {} key-value pairs", sequential_data.len());
    
    let test_data = data_to_bytes(&sequential_data);
    assert_eq!(test_data.len() % 16, 0, "Test data size should be a multiple of 16 bytes");
    println!("Test data size: {} bytes", test_data.len());
    
    // Try different compression types directly
    for compression_type in [
        CompressionType::None,
        CompressionType::BitPack,
        CompressionType::Delta,
        CompressionType::Dictionary,
        CompressionType::Lz4,
        CompressionType::Snappy,
    ] {
        println!("Testing {:?} compression directly", compression_type);
        let compressor = CompressionFactory::create(compression_type);
        let compressed = compressor.compress(&test_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        // Verify the data can be compressed and decompressed correctly
        assert_eq!(test_data, decompressed, "Compression/decompression cycle failed for {:?}", compression_type);
    }
    
    // Now test with NoopCompression for LSM tree
    println!("Testing compression with LSM tree");
    let compression_config = CompressionConfig {
        enabled: true,
        l0_default: CompressionType::None,  // Use NoopCompression for LSM test
        lower_level_default: CompressionType::None,
        level_types: vec![Some(CompressionType::None); 10],
        ..Default::default()
    };
    
    let adaptive_config = AdaptiveCompressionConfig {
        enabled: false,
        ..Default::default()
    };
    
    // Create LSM tree with NoopCompression
    let (mut lsm, _dir) = create_lsm_tree(compression_config, adaptive_config);
    
    // Use a very small dataset for the LSM tree test
    let small_data = generate_sequential_data(5);
    println!("Testing LSM tree with {} key-value pairs", small_data.len());
    
    // Insert data
    for &(key, value) in &small_data {
        println!("Inserting key: {}, value: {}", key, value);
        lsm.put(key, value).unwrap();
    }
    
    // Flush to trigger compression
    println!("Flushing buffer to level 0");
    lsm.flush_buffer_to_level0().unwrap();
    
    // Verify data is retrievable
    for &(key, expected_value) in &small_data {
        let value = lsm.get(key);
        assert_eq!(value, Some(expected_value), "Failed to retrieve key {}", key);
    }
    
    // Test range query
    if !small_data.is_empty() {
        let min_key = small_data.first().unwrap().0;
        let max_key = small_data.last().unwrap().0;
        
        println!("Testing range query from {} to {}", min_key, max_key);
        if max_key > min_key {
            let mid_key = min_key + (max_key - min_key) / 2;
            
            let range_result = lsm.range(min_key, mid_key);
            println!("Range query returned {} results", range_result.len());
            
            // Verify range results
            for (key, value) in &range_result {
                assert!(*key >= min_key && *key < mid_key);
                
                // Find the expected value in the original data
                let expected = small_data.iter()
                    .find(|(k, _)| *k == *key)
                    .map(|(_, v)| *v);
                
                assert_eq!(Some(*value), expected, "Range query returned unexpected value for key {}", key);
            }
        }
    }
}