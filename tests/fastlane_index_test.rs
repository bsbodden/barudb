use barudb::run::{FastLaneFencePointers, StandardFencePointers};
use barudb::types::Key;

/// Test to debug the coverage issues with sequential keys
#[test]
fn test_fastlane_sequential_coverage() {
    // Generate a simple set of sequential keys
    let size = 1000;
    let keys: Vec<Key> = (0..size as Key).collect();
    
    // Build Standard fence pointers
    let mut std_fps = StandardFencePointers::new();
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        std_fps.add(chunk[0], chunk[1], i);
    }
    
    // Build FastLane fence pointers
    let mut fastlane_fps = FastLaneFencePointers::new();
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        fastlane_fps.add(chunk[0], chunk[1], i);
    }
    
    // Generate test lookup keys (every key in the original set)
    for key in 0..size as Key {
        let std_result = std_fps.find_block_for_key(key);
        let fastlane_result = fastlane_fps.find_block_for_key(key);
        
        // Check if both implementations found the key
        if std_result.is_some() && fastlane_result.is_none() {
            println!("Key {} found by Standard but not by FastLane", key);
            // Get the block details from Standard
            let std_block = std_result.unwrap();
            println!("  Standard block: {}", std_block);
            
            // Debug FastLane's partitions
            for (i, partition) in fastlane_fps.partitions.iter().enumerate() {
                // Check partition range
                if key >= partition.min_key && key <= partition.max_key {
                    println!("  Key {} is in partition {} range [{}, {}]",
                             key, i, partition.min_key, partition.max_key);
                }
                
                // Check individual entries
                for j in 0..partition.len() {
                    let min_key = partition.min_key_lane[j];
                    let max_key = partition.max_key_lane[j];
                    let block_idx = partition.block_index_lane[j];
                    
                    // If key is in range, it should have been found
                    if key >= min_key && key <= max_key {
                        println!("  Key {} SHOULD be in partition {}, entry {}: [{}, {}] -> {}", 
                                 key, i, j, min_key, max_key, block_idx);
                    }
                }
            }
        }
    }
    
    // Count how many keys are found by each implementation
    let mut std_found = 0;
    let mut fastlane_found = 0;
    
    for key in 0..size as Key {
        if std_fps.find_block_for_key(key).is_some() {
            std_found += 1;
        }
        if fastlane_fps.find_block_for_key(key).is_some() {
            fastlane_found += 1;
        }
    }
    
    println!("Standard found: {}/{} keys ({}%)", 
             std_found, size, std_found as f64 / size as f64 * 100.0);
    println!("FastLane found: {}/{} keys ({}%)", 
             fastlane_found, size, fastlane_found as f64 / size as f64 * 100.0);
    
    // The coverage should be similar for both implementations
    assert!(fastlane_found as f64 / std_found as f64 > 0.9, 
            "FastLane found significantly fewer keys than Standard");
}

/// Test to debug the coverage issues with grouped keys
#[test]
fn test_fastlane_grouped_coverage() {
    // This test will focus on a very small set of grouped keys to debug the problem
    // Generate a simple set of grouped keys - just 4 groups with 5 keys each
    let size = 20;
    let num_groups = 4;
    let mut keys = Vec::with_capacity(size);
    
    // Generate keys with known pattern
    for i in 0..size {
        let group = (i % num_groups) as Key;  // Groups 0, 1, 2, 3 
        let offset = (i / num_groups) as Key; // Offsets 0, 1, 2, 3, 4
        let key = group * 1_000_000 + offset;
        keys.push(key);
    }
    
    // Print the keys we're using
    println!("Test keys:");
    for key in &keys {
        println!("  {}", key);
    }
    
    // Build Standard fence pointers
    let mut std_fps = StandardFencePointers::new();
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        let min_key = *chunk.iter().min().unwrap();
        let max_key = *chunk.iter().max().unwrap();
        std_fps.add(min_key, max_key, i);
        println!("Added to Standard: [{}, {}] -> {}", min_key, max_key, i);
    }
    
    // Build FastLane fence pointers
    let mut fastlane_fps = FastLaneFencePointers::with_group_size(64);
    for (i, chunk) in keys.chunks(2).enumerate() {
        if chunk.len() < 2 { continue; }
        let min_key = *chunk.iter().min().unwrap();
        let max_key = *chunk.iter().max().unwrap();
        fastlane_fps.add(min_key, max_key, i);
        println!("Added to FastLane: [{}, {}] -> {}", min_key, max_key, i);
    }
    
    // Print FastLane structure info
    println!("\nFastLane structure:");
    println!("  Number of partitions: {}", fastlane_fps.partitions.len());
    println!("  Target partition size: {}", fastlane_fps.target_partition_size);
    
    // Examine each partition
    for (i, partition) in fastlane_fps.partitions.iter().enumerate() {
        println!("Partition {}: entries={}, range=[{}, {}]", 
                 i, partition.len(), partition.min_key, partition.max_key);
        
        // Print all entries for debugging
        for j in 0..partition.len() {
            let min_key = partition.min_key_lane[j];
            let max_key = partition.max_key_lane[j];
            let block_idx = partition.block_index_lane[j];
            
            println!("  Entry {}: [{}, {}] -> {}", 
                     j, min_key, max_key, block_idx);
        }
    }
    
    // Test looking up every key
    println!("\nTesting lookups for each key:");
    for &key in &keys {
        let std_result = std_fps.find_block_for_key(key);
        let fastlane_result = fastlane_fps.find_block_for_key(key);
        
        println!("Key {}: Standard={:?}, FastLane={:?}", 
                 key, std_result, fastlane_result);
                 
        // Debug grouped keys detection
        let key_bits = key as u64;
        let high_32_bits = key_bits >> 32;
        println!("  Key bits: 0x{:X}, high 32 bits: 0x{:X} ({})", 
                 key_bits, high_32_bits, high_32_bits);
        
        // If the key wasn't found in FastLane but was in Standard
        if std_result.is_some() && fastlane_result.is_none() {
            println!("  KEY NOT FOUND IN FASTLANE BUT FOUND IN STANDARD");
            
            // Print which partition should contain this key in FastLane
            for (i, partition) in fastlane_fps.partitions.iter().enumerate() {
                // Check partition range
                if key >= partition.min_key && key <= partition.max_key {
                    println!("  Key {} should be in partition {} range [{}, {}]",
                             key, i, partition.min_key, partition.max_key);
                }
                
                // For each entry in the partition
                for j in 0..partition.len() {
                    let min_key = partition.min_key_lane[j];
                    let max_key = partition.max_key_lane[j];
                    
                    // If this entry's range includes our key
                    if key >= min_key && key <= max_key {
                        println!("  Key {} SHOULD be in partition {}, entry {}: [{}, {}] -> {}", 
                                 key, i, j, min_key, max_key, partition.block_index_lane[j]);
                    }
                }
            }
        }
    }
    
    // Count how many keys are found by each implementation
    let mut std_found = 0;
    let mut fastlane_found = 0;
    
    for &key in &keys {
        if std_fps.find_block_for_key(key).is_some() {
            std_found += 1;
        }
        if fastlane_fps.find_block_for_key(key).is_some() {
            fastlane_found += 1;
        }
    }
    
    println!("\nSummary:");
    println!("Standard found: {}/{} keys ({}%)", 
             std_found, keys.len(), std_found as f64 / keys.len() as f64 * 100.0);
    println!("FastLane found: {}/{} keys ({}%)", 
             fastlane_found, keys.len(), fastlane_found as f64 / keys.len() as f64 * 100.0);
    
    // For our debug test, don't fail even if there's a difference
    // We'll use this diagnostic information to fix the implementation
}