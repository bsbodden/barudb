use lsm_tree::lsm_tree::{LSMTree, LSMConfig};
use lsm_tree::types::{CompactionPolicyType, StorageType};
use tempfile::tempdir;

fn create_test_tree(run_threshold: usize, policy_type: CompactionPolicyType) -> (LSMTree, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    
    // Create a configuration with compression disabled for recovery tests
    let mut compression_config = lsm_tree::run::CompressionConfig::default();
    compression_config.enabled = false; // Disable compression for recovery tests
    
    let mut adaptive_config = lsm_tree::run::AdaptiveCompressionConfig::default();
    adaptive_config.enabled = false; // Also disable adaptive compression
    
    let config = LSMConfig {
        buffer_size: 4, // Small buffer for testing
        storage_type: StorageType::File,
        storage_path: temp_dir.path().to_path_buf(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: policy_type,
        compaction_threshold: run_threshold,
        compression: compression_config,
        adaptive_compression: adaptive_config,
        collect_compression_stats: false, // Don't collect stats for tests
        background_compaction: false,
        use_lock_free_memtable: false,  // Use standard sharded memtable by default
        use_lock_free_block_cache: true, // Use lock-free block cache by default
    };
    (LSMTree::with_config(config), temp_dir)
}

#[test]
fn test_tiered_compaction_integration() {
    // Create a tree with a tiered compaction policy and a threshold of 2 runs
    let (mut tree, _temp_dir) = create_test_tree(2, CompactionPolicyType::Tiered);
    
    // Add data to fill the buffer and trigger a flush
    for i in 1..5 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Data should still be accessible
    assert_eq!(tree.get(1), Some(100));
    assert_eq!(tree.get(4), Some(400));
    
    // Add more data to trigger another flush and compaction
    for i in 5..9 {
        tree.put(i, i * 100).unwrap();
    }
    
    // All data should still be accessible
    assert_eq!(tree.get(1), Some(100));
    assert_eq!(tree.get(8), Some(800));
    
    // Add more data to see multi-level compaction
    for i in 9..13 {
        tree.put(i, i * 100).unwrap();
    }
    
    // All data should be accessible
    for i in 1..13 {
        assert_eq!(tree.get(i), Some(i * 100));
    }
    
    // Try a forced compaction
    tree.force_compact_all().unwrap();
    
    // Data should still be intact
    for i in 1..13 {
        assert_eq!(tree.get(i), Some(i * 100));
    }
}

#[test]
fn test_delete_with_compaction() {
    // Create a tree with a tiered compaction policy and a threshold of 2 runs
    let (mut tree, _temp_dir) = create_test_tree(2, CompactionPolicyType::Tiered);
    
    // Add some data
    for i in 1..5 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Delete some data
    tree.delete(2).unwrap();
    tree.delete(4).unwrap();
    
    // Add more data to trigger flush and compaction
    for i in 5..9 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Verify deletions persisted through compaction
    assert_eq!(tree.get(1), Some(100));
    assert_eq!(tree.get(2), None); // Deleted
    assert_eq!(tree.get(3), Some(300));
    assert_eq!(tree.get(4), None); // Deleted
    assert_eq!(tree.get(5), Some(500));
}

#[test]
fn test_leveled_compaction_integration() {
    // Create a tree with a leveled compaction policy and a threshold of 4
    let (mut tree, _temp_dir) = create_test_tree(4, CompactionPolicyType::Leveled);
    
    // Add data to fill the buffer and trigger a flush
    for i in 1..5 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Data should still be accessible
    assert_eq!(tree.get(1), Some(100));
    assert_eq!(tree.get(4), Some(400));
    
    // Add more data to trigger another flush and compaction
    // In leveled compaction, this should create a new run in L0, 
    // then compact it with any existing runs in L0 to L1
    for i in 5..9 {
        tree.put(i, i * 100).unwrap();
    }
    
    // All data should still be accessible
    assert_eq!(tree.get(1), Some(100));
    assert_eq!(tree.get(8), Some(800));
    
    // Add more data to see multi-level compaction
    for i in 9..17 {
        tree.put(i, i * 100).unwrap();
    }
    
    // All data should be accessible
    for i in 1..17 {
        assert_eq!(tree.get(i), Some(i * 100));
    }
    
    // Test updates with leveled compaction
    for i in 5..10 {
        tree.put(i, i * 200).unwrap(); // Double the values
    }
    
    // Verify updates were applied
    for i in 5..10 {
        assert_eq!(tree.get(i), Some(i * 200));
    }
    
    // Test deletions with leveled compaction
    tree.delete(2).unwrap();
    tree.delete(4).unwrap();
    tree.delete(15).unwrap();
    
    // Ensure deletes are visible
    assert_eq!(tree.get(2), None);
    assert_eq!(tree.get(4), None);
    assert_eq!(tree.get(15), None);
    
    // Try a forced compaction
    tree.force_compact_all().unwrap();
    
    // Verify all data is still intact
    assert_eq!(tree.get(1), Some(100));
    assert_eq!(tree.get(2), None); // Deleted
    assert_eq!(tree.get(3), Some(300));
    assert_eq!(tree.get(4), None); // Deleted
    for i in 5..10 {
        assert_eq!(tree.get(i), Some(i * 200)); // Updated values
    }
    for i in 10..15 {
        assert_eq!(tree.get(i), Some(i * 100));
    }
    assert_eq!(tree.get(15), None); // Deleted
    assert_eq!(tree.get(16), Some(1600));
}

#[test]
fn test_recovery_with_compaction() {
    // Create storage path for recovery test
    let temp_dir = tempdir().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    // Create first tree with tiered compaction but with specific path
    let mut config = lsm_tree::run::CompressionConfig::default();
    config.enabled = false; // Disable compression for recovery
    
    let mut adaptive_config = lsm_tree::run::AdaptiveCompressionConfig::default();
    adaptive_config.enabled = false;
    
    let config = LSMConfig {
        buffer_size: 4,
        storage_type: StorageType::File,
        storage_path: storage_path.clone(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::Tiered,
        compaction_threshold: 2,
        compression: config,
        adaptive_compression: adaptive_config,
        collect_compression_stats: false,
        background_compaction: false,
        use_lock_free_memtable: false,  // Use standard sharded memtable by default
        use_lock_free_block_cache: true, // Use lock-free block cache by default
    };
    
    let mut tree = LSMTree::with_config(config);
    
    // Add data and trigger compaction
    for i in 1..13 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Force a flush to ensure all data is on disk
    tree.flush_buffer_to_level0().unwrap();
    
    // Force compaction to ensure a clean state
    tree.force_compact_all().unwrap();
    
    // Drop the tree to ensure files are closed
    drop(tree);
    
    // Create a new tree with the same storage path to test recovery
    let mut config = lsm_tree::run::CompressionConfig::default();
    config.enabled = false; // Disable compression for recovery
    let mut adaptive_config = lsm_tree::run::AdaptiveCompressionConfig::default();
    adaptive_config.enabled = false;
    
    let config = LSMConfig {
        buffer_size: 4,
        storage_type: StorageType::File,
        storage_path: storage_path.clone(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::Tiered,
        compaction_threshold: 2,
        compression: config,
        adaptive_compression: adaptive_config,
        collect_compression_stats: false,
        background_compaction: false,
        use_lock_free_memtable: false,  // Use standard sharded memtable by default
        use_lock_free_block_cache: true, // Use lock-free block cache by default
    };
    
    let recovered_tree = LSMTree::with_config(config);
    
    // Test that all data was recovered
    for i in 1..13 {
        assert_eq!(recovered_tree.get(i), Some(i * 100), "Recovery failed for key {}", i);
    }
}

#[test]
fn test_recovery_with_leveled_compaction() {
    // Create storage path for recovery test
    let temp_dir = tempdir().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    // Create first tree with leveled compaction but with specific path
    let mut config = lsm_tree::run::CompressionConfig::default();
    config.enabled = false; // Disable compression for recovery
    
    let mut adaptive_config = lsm_tree::run::AdaptiveCompressionConfig::default();
    adaptive_config.enabled = false;
    
    let config = LSMConfig {
        buffer_size: 4,
        storage_type: StorageType::File,
        storage_path: storage_path.clone(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::Leveled,
        compaction_threshold: 4,
        compression: config,
        adaptive_compression: adaptive_config,
        collect_compression_stats: false,
        background_compaction: false,
        use_lock_free_memtable: false,  // Use standard sharded memtable by default
        use_lock_free_block_cache: true, // Use lock-free block cache by default
    };
    
    let mut tree = LSMTree::with_config(config);
        
    // Add data and trigger compaction
    for i in 1..21 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Add some updates and deletes
    tree.put(5, 555).unwrap();
    tree.put(10, 1010).unwrap();
    tree.delete(15).unwrap();
    
    // Force a flush to ensure all data is on disk
    tree.flush_buffer_to_level0().unwrap();
    
    // Force compaction to ensure a clean state
    tree.force_compact_all().unwrap();
    
    // Drop the tree to ensure files are closed
    drop(tree);
    
    // Create a new tree with the same storage path to test recovery
    let mut compression_config = lsm_tree::run::CompressionConfig::default();
    compression_config.enabled = false; // Disable compression for recovery
    
    let mut adaptive_config = lsm_tree::run::AdaptiveCompressionConfig::default();
    adaptive_config.enabled = false;
    
    let config = LSMConfig {
        buffer_size: 4,
        storage_type: StorageType::File,
        storage_path: storage_path,
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::Leveled,
        compaction_threshold: 4,
        compression: compression_config,
        adaptive_compression: adaptive_config,
        collect_compression_stats: false,
        background_compaction: false,
        use_lock_free_memtable: false,  // Use standard sharded memtable by default
        use_lock_free_block_cache: true, // Use lock-free block cache by default
    };
    
    let recovered_tree = LSMTree::with_config(config);
    
    // Test that all data was recovered, including updates and deletes
    for i in 1..5 {
        assert_eq!(recovered_tree.get(i), Some(i * 100), "Recovery failed for key {}", i);
    }
    assert_eq!(recovered_tree.get(5), Some(555), "Recovery failed for updated key 5");
    for i in 6..10 {
        assert_eq!(recovered_tree.get(i), Some(i * 100), "Recovery failed for key {}", i);
    }
    assert_eq!(recovered_tree.get(10), Some(1010), "Recovery failed for updated key 10");
    for i in 11..15 {
        assert_eq!(recovered_tree.get(i), Some(i * 100), "Recovery failed for key {}", i);
    }
    assert_eq!(recovered_tree.get(15), None, "Recovery failed for deleted key 15");
    for i in 16..21 {
        assert_eq!(recovered_tree.get(i), Some(i * 100), "Recovery failed for key {}", i);
    }
}

#[test]
fn test_lazy_leveled_compaction_integration() {
    // Create a tree with a lazy leveled compaction policy
    // Use a threshold of 3 for L0
    let (mut tree, _temp_dir) = create_test_tree(3, CompactionPolicyType::LazyLeveled);
    
    // Add data to fill the buffer and trigger a flush to L0
    for i in 1..5 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Data should still be accessible
    assert_eq!(tree.get(1), Some(100));
    assert_eq!(tree.get(4), Some(400));
    
    // Add more data to trigger another flush to L0
    // This will create a second run in L0
    for i in 5..9 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Add even more data to trigger a third flush to L0
    // This should now trigger compaction from L0 to L1 since we have 3 runs in L0
    for i in 9..13 {
        tree.put(i, i * 100).unwrap();
    }
    
    // All data should still be accessible
    for i in 1..13 {
        assert_eq!(tree.get(i), Some(i * 100));
    }
    
    // Add data that will go to different levels
    // This will cause more compactions and will test the "leveled" behavior
    // of higher levels
    for i in 13..25 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Test updates
    for i in 5..10 {
        tree.put(i, i * 200).unwrap(); // Double the values
    }
    
    // Test deletions
    tree.delete(2).unwrap();
    tree.delete(15).unwrap();
    
    // Force a compaction to ensure everything is properly merged
    tree.force_compact_all().unwrap();
    
    // Verify all data is correctly maintained
    // L1 and higher should have at most one run per level (leveled policy)
    assert_eq!(tree.get(1), Some(100));
    assert_eq!(tree.get(2), None); // Deleted
    assert_eq!(tree.get(3), Some(300));
    for i in 5..10 {
        assert_eq!(tree.get(i), Some(i * 200)); // Updated values
    }
    for i in 10..15 {
        assert_eq!(tree.get(i), Some(i * 100));
    }
    assert_eq!(tree.get(15), None); // Deleted
    for i in 16..25 {
        assert_eq!(tree.get(i), Some(i * 100));
    }
}

#[test]
fn test_recovery_with_lazy_leveled_compaction() {
    // Create storage path for recovery test
    let temp_dir = tempdir().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    // Create first tree with lazy leveled compaction but with specific path
    let mut config = lsm_tree::run::CompressionConfig::default();
    config.enabled = false; // Disable compression for recovery
    
    let mut adaptive_config = lsm_tree::run::AdaptiveCompressionConfig::default();
    adaptive_config.enabled = false;
    
    let config = LSMConfig {
        buffer_size: 4,
        storage_type: StorageType::File,
        storage_path: storage_path.clone(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::LazyLeveled,
        compaction_threshold: 3,
        compression: config,
        adaptive_compression: adaptive_config,
        collect_compression_stats: false,
        background_compaction: false,
        use_lock_free_memtable: false,  // Use standard sharded memtable by default
        use_lock_free_block_cache: true, // Use lock-free block cache by default
    };
    
    let mut tree = LSMTree::with_config(config);
        
    // Add data and trigger compaction
    for i in 1..21 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Add some updates and deletes
    tree.put(5, 555).unwrap();
    tree.put(10, 1010).unwrap();
    tree.delete(15).unwrap();
    
    // Force a flush to ensure all data is on disk
    tree.flush_buffer_to_level0().unwrap();
    
    // Force compaction to ensure a clean state
    tree.force_compact_all().unwrap();
    
    // Drop the tree to ensure files are closed
    drop(tree);
    
    // Create a new tree with the same storage path to test recovery
    let mut compression_config = lsm_tree::run::CompressionConfig::default();
    compression_config.enabled = false; // Disable compression for recovery
    
    let mut adaptive_config = lsm_tree::run::AdaptiveCompressionConfig::default();
    adaptive_config.enabled = false;
    
    let config = LSMConfig {
        buffer_size: 4,
        storage_type: StorageType::File,
        storage_path: storage_path,
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::LazyLeveled,
        compaction_threshold: 3, // Threshold for L0
        compression: compression_config,
        adaptive_compression: adaptive_config,
        collect_compression_stats: false,
        background_compaction: false,
        use_lock_free_memtable: false,  // Use standard sharded memtable by default
        use_lock_free_block_cache: true, // Use lock-free block cache by default
    };
    
    let recovered_tree = LSMTree::with_config(config);
    
    // Test that all data was recovered, including updates and deletes
    for i in 1..5 {
        assert_eq!(recovered_tree.get(i), Some(i * 100), "Recovery failed for key {}", i);
    }
    assert_eq!(recovered_tree.get(5), Some(555), "Recovery failed for updated key 5");
    for i in 6..10 {
        assert_eq!(recovered_tree.get(i), Some(i * 100), "Recovery failed for key {}", i);
    }
    assert_eq!(recovered_tree.get(10), Some(1010), "Recovery failed for updated key 10");
    for i in 11..15 {
        assert_eq!(recovered_tree.get(i), Some(i * 100), "Recovery failed for key {}", i);
    }
    assert_eq!(recovered_tree.get(15), None, "Recovery failed for deleted key 15");
    for i in 16..21 {
        assert_eq!(recovered_tree.get(i), Some(i * 100), "Recovery failed for key {}", i);
    }
}

#[test]
fn test_compare_compaction_policies() {
    // This test checks the differences in behavior between the three policies
    
    // Create trees with different compaction policies but same threshold
    let (mut tiered_tree, _temp_dir1) = create_test_tree(3, CompactionPolicyType::Tiered);
    let (mut leveled_tree, _temp_dir2) = create_test_tree(3, CompactionPolicyType::Leveled);
    let (mut lazy_tree, _temp_dir3) = create_test_tree(3, CompactionPolicyType::LazyLeveled);
    
    // Add same data to all trees
    for i in 1..20 {
        tiered_tree.put(i, i * 100).unwrap();
        leveled_tree.put(i, i * 100).unwrap();
        lazy_tree.put(i, i * 100).unwrap();
    }
    
    // Flush all buffers to L0
    tiered_tree.flush_buffer_to_level0().unwrap();
    leveled_tree.flush_buffer_to_level0().unwrap();
    lazy_tree.flush_buffer_to_level0().unwrap();
    
    // Force compaction
    tiered_tree.force_compact_all().unwrap();
    leveled_tree.force_compact_all().unwrap();
    lazy_tree.force_compact_all().unwrap();
    
    // Check that data is accessible in all trees
    for i in 1..20 {
        assert_eq!(tiered_tree.get(i), Some(i * 100));
        assert_eq!(leveled_tree.get(i), Some(i * 100));
        assert_eq!(lazy_tree.get(i), Some(i * 100));
    }
    
    // While the data is the same, the internal structure should be different:
    // - Tiered: Multiple runs per level, compact when threshold is reached
    // - Leveled: Single run per level, compact when > 1 run at any level
    // - Lazy Leveled: Multiple runs at L0 (like Tiered), single run per level elsewhere (like Leveled)
}