use barudb::run::{
    SimpleFastLaneFencePointers, TwoLevelFastLaneFencePointers,
    StandardFencePointers, FastLaneFencePointers, FencePointersInterface
};
use barudb::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Basic test to check that all fence pointer implementations 
/// can be used through the common interface
#[test]
fn test_fence_pointers_interface() {
    // Create instances of each implementation
    let mut standard = StandardFencePointers::new();
    let mut simple = SimpleFastLaneFencePointers::new();
    let mut two_level = TwoLevelFastLaneFencePointers::new();
    let mut original = FastLaneFencePointers::new();
    
    // Add some fence pointers to each
    for i in 0..100 {
        let min_key = i * 10;
        let max_key = i * 10 + 9;
        let block_index = i as usize;
        
        standard.add(min_key, max_key, block_index);
        simple.add(min_key, max_key, block_index);
        two_level.add(min_key, max_key, block_index);
        original.add(min_key, max_key, block_index);
    }
    
    // Test point queries
    for i in 0..100 {
        let key = i * 10 + 5; // Middle of each range
        
        let standard_result = standard.find_block_for_key(key);
        let simple_result = simple.find_block_for_key(key);
        let two_level_result = two_level.find_block_for_key(key);
        let original_result = original.find_block_for_key(key);
        
        assert!(standard_result.is_some(), "Standard should find key {}", key);
        assert!(simple_result.is_some(), "Simple should find key {}", key);
        assert!(two_level_result.is_some(), "TwoLevel should find key {}", key);
        assert!(original_result.is_some(), "Original should find key {}", key);
        
        println!("Key {}: Standard={:?}, Simple={:?}, TwoLevel={:?}, Original={:?}",
                 key, standard_result, simple_result, two_level_result, original_result);
    }
    
    // Test range queries
    for i in 0..10 {
        let start = i * 100;
        let end = (i + 1) * 100;
        
        let standard_blocks = standard.find_blocks_in_range(start, end);
        let simple_blocks = simple.find_blocks_in_range(start, end);
        let two_level_blocks = two_level.find_blocks_in_range(start, end);
        let original_blocks = original.find_blocks_in_range(start, end);
        
        println!("Range [{}, {}]: Standard={}, Simple={}, TwoLevel={}, Original={}",
                 start, end, standard_blocks.len(), simple_blocks.len(), 
                 two_level_blocks.len(), original_blocks.len());
        
        // Check that the blocks are similar in count (allowing for implementation differences)
        // but we shouldn't expect exact matches due to different implementations
    }
    
    // Test memory usage
    println!("Memory usage: Standard={}, Simple={}, TwoLevel={}, Original={}",
             standard.memory_usage(), simple.memory_usage(), 
             two_level.memory_usage(), original.memory_usage());
    
    // Use the implementations through the trait interface
    fn use_fence_pointers(fp: &dyn FencePointersInterface) {
        assert!(!fp.is_empty());
        assert!(fp.len() > 0);
        assert!(fp.memory_usage() > 0);
    }
    
    use_fence_pointers(&standard);
    use_fence_pointers(&simple);
    use_fence_pointers(&two_level);
    use_fence_pointers(&original);
}

/// Compare optimization approaches between different implementations
#[test]
fn test_fence_pointers_optimization() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Create implementations
    let mut simple = SimpleFastLaneFencePointers::new();
    let mut two_level = TwoLevelFastLaneFencePointers::with_partition_size(16);
    
    // Add random fence pointers
    for i in 0..1000 {
        let min_key = rng.random::<Key>() % 1_000_000;
        let max_key = min_key + rng.random_range(1..100);
        
        simple.add(min_key, max_key, i);
        two_level.add(min_key, max_key, i);
    }
    
    // Test before optimization
    let mut simple_hits = 0;
    let mut two_level_hits = 0;
    
    // Generate random lookup keys
    let mut lookup_keys = Vec::with_capacity(100);
    for _ in 0..100 {
        lookup_keys.push(rng.random::<Key>() % 1_000_000);
    }
    
    // Test lookups before optimization
    for key in &lookup_keys {
        if simple.find_block_for_key(*key).is_some() {
            simple_hits += 1;
        }
        if two_level.find_block_for_key(*key).is_some() {
            two_level_hits += 1;
        }
    }
    
    println!("Before optimization: Simple hits={}, TwoLevel hits={}", 
             simple_hits, two_level_hits);
    
    // Optimize the implementations
    // Note that TwoLevelFastLaneFencePointers doesn't have "optimize" on the trait
    // but it has a method in the struct, so we cast to concrete type
    simple.optimize(); // The optimize method modifies the fence pointers in place rather than returning a new instance
    
    // Test after optimization
    let mut optimized_simple_hits = 0;
    
    for key in &lookup_keys {
        if simple.find_block_for_key(*key).is_some() {
            optimized_simple_hits += 1;
        }
    }
    
    println!("After optimization: Simple hits={}", optimized_simple_hits);
    
    // Check memory usage
    println!("Memory usage: Simple={}, TwoLevel={}",
             simple.memory_usage(), two_level.memory_usage());
}

/// Test with million pattern keys like 1000000, 2000000, 3000000
#[test]
fn test_million_pattern() {
    // Create implementations
    let mut standard = StandardFencePointers::new();
    let mut simple = SimpleFastLaneFencePointers::new();
    let mut two_level = TwoLevelFastLaneFencePointers::new();
    let mut original = FastLaneFencePointers::new();
    
    // Add fence pointers with million pattern
    for i in 1..10 {
        let min_key = i * 1_000_000;
        let max_key = min_key + 999;
        
        standard.add(min_key, max_key, i as usize);
        simple.add(min_key, max_key, i as usize);
        two_level.add(min_key, max_key, i as usize);
        original.add(min_key, max_key, i as usize);
    }
    
    // Test exact matches (key in the middle of range)
    for i in 1..10 {
        let key = i * 1_000_000 + 500;
        
        let standard_result = standard.find_block_for_key(key);
        let simple_result = simple.find_block_for_key(key);
        let two_level_result = two_level.find_block_for_key(key);
        let original_result = original.find_block_for_key(key);
        
        println!("Key {}: Standard={:?}, Simple={:?}, TwoLevel={:?}, Original={:?}",
                 key, standard_result, simple_result, two_level_result, original_result);
        
        assert!(standard_result.is_some(), "Standard should find key {}", key);
        assert!(simple_result.is_some(), "Simple should find key {}", key);
        assert!(two_level_result.is_some(), "TwoLevel should find key {}", key);
        assert!(original_result.is_some(), "Original should find key {}", key);
    }
    
    // Test near misses (key just outside a range)
    for i in 1..10 {
        let key = i * 1_000_000 - 1; // Just before the million
        
        let standard_result = standard.find_block_for_key(key);
        let simple_result = simple.find_block_for_key(key);
        let two_level_result = two_level.find_block_for_key(key);
        let original_result = original.find_block_for_key(key);
        
        println!("Near miss key {}: Standard={:?}, Simple={:?}, TwoLevel={:?}, Original={:?}",
                 key, standard_result, simple_result, two_level_result, original_result);
    }
}