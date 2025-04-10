use lsm_tree::lsm_tree::{LSMTree, LSMConfig};
use lsm_tree::types::CompactionPolicyType;
use tempfile::tempdir;

fn create_test_tree(run_threshold: usize) -> (LSMTree, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    let config = LSMConfig {
        buffer_size: 4, // Small buffer for testing
        storage_type: "file".to_string(),
        storage_path: temp_dir.path().to_path_buf(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::Tiered,
        compaction_threshold: run_threshold,
    };
    (LSMTree::with_config(config), temp_dir)
}

#[test]
fn test_tiered_compaction_integration() {
    // Create a tree with a tiered compaction policy and a threshold of 2 runs
    let (mut tree, _temp_dir) = create_test_tree(2);
    
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
    let (mut tree, _temp_dir) = create_test_tree(2);
    
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
fn test_recovery_with_compaction() {
    // Create storage path for recovery test
    let temp_dir = tempdir().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    // Create first tree
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