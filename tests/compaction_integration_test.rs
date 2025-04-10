use lsm_tree::lsm_tree::{LSMTree, LSMConfig};
use lsm_tree::types::CompactionPolicyType;
use tempfile::tempdir;

fn create_test_tree(run_threshold: usize, policy_type: CompactionPolicyType) -> (LSMTree, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    let config = LSMConfig {
        buffer_size: 4, // Small buffer for testing
        storage_type: "file".to_string(),
        storage_path: temp_dir.path().to_path_buf(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: policy_type,
        compaction_threshold: run_threshold,
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
    
    // Create first tree with tiered compaction
    let config = LSMConfig {
        buffer_size: 4,
        storage_type: "file".to_string(),
        storage_path: storage_path.clone(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::Tiered,
        compaction_threshold: 2,
    };
    
    // Create tree, add data, and force compaction
    {
        let mut tree = LSMTree::with_config(config.clone());
        
        // Add data and trigger compaction
        for i in 1..13 {
            tree.put(i, i * 100).unwrap();
        }
        
        // Force a flush to ensure all data is on disk
        tree.flush_buffer_to_level0().unwrap();
        
        // Force compaction to ensure a clean state
        tree.force_compact_all().unwrap();
    }
    
    // Create a new tree with the same storage path to test recovery
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
    
    // Create first tree with leveled compaction
    let config = LSMConfig {
        buffer_size: 4,
        storage_type: "file".to_string(),
        storage_path: storage_path.clone(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::Leveled,
        compaction_threshold: 4,
    };
    
    // Create tree, add data, and force compaction
    {
        let mut tree = LSMTree::with_config(config.clone());
        
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
    }
    
    // Create a new tree with the same storage path to test recovery
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