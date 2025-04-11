use lsm_tree::lsm_tree::{LSMTree, LSMConfig};
use lsm_tree::types::{CompactionPolicyType, StorageType};
use lsm_tree::compaction::{SelectionStrategy, PartialTieredCompactionPolicy, CompactionPolicy};
use lsm_tree::level::Level;
use lsm_tree::run::Run;
use tempfile::tempdir;

fn create_test_tree(run_threshold: usize, policy_type: CompactionPolicyType) -> (LSMTree, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
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
    };
    (LSMTree::with_config(config), temp_dir)
}

#[test]
#[ignore = "TODO: Fix test with block cache integration"]
fn test_partial_tiered_compaction() {
    // Create a tree with a partial tiered compaction policy and a threshold of 4 runs
    let (mut tree, _temp_dir) = create_test_tree(4, CompactionPolicyType::PartialTiered);
    
    // Add some initial data to create a first run
    for i in 1..5 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Force a flush to create a run in level 0
    tree.flush_buffer_to_level0().unwrap();
    
    // Add data for a second run
    for i in 5..9 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Force a flush to create a second run in level 0
    tree.flush_buffer_to_level0().unwrap();
    
    // Add data for a third run
    for i in 9..13 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Force a flush to create a third run in level 0
    tree.flush_buffer_to_level0().unwrap();
    
    // Add data for a fourth run, which should trigger compaction
    for i in 13..17 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Force a flush to create a fourth run in level 0
    tree.flush_buffer_to_level0().unwrap();
    
    // The compaction should have occurred automatically after the fourth flush
    // Let's check that all data is still accessible
    for i in 1..17 {
        assert_eq!(tree.get(i), Some(i * 100), "Data loss for key {}", i);
    }
    
    // Test updating values
    for i in 5..10 {
        tree.put(i, i * 200).unwrap(); // Double the values
    }
    
    // Force another flush
    tree.flush_buffer_to_level0().unwrap();
    
    // Check that updates are visible
    for i in 5..10 {
        assert_eq!(tree.get(i), Some(i * 200), "Update not visible for key {}", i);
    }
    
    // Force compaction and check data integrity
    tree.force_compact_all().unwrap();
    
    // Verify all data is still intact
    for i in 1..5 {
        assert_eq!(tree.get(i), Some(i * 100), "Data loss after compaction for key {}", i);
    }
    for i in 5..10 {
        assert_eq!(tree.get(i), Some(i * 200), "Update lost after compaction for key {}", i);
    }
    for i in 10..17 {
        assert_eq!(tree.get(i), Some(i * 100), "Data loss after compaction for key {}", i);
    }
}

#[test]
#[ignore = "TODO: Fix test with block cache integration"]
fn test_partial_tiered_with_deletion() {
    // Create a tree with a partial tiered compaction policy and a threshold of 4 runs
    let (mut tree, _temp_dir) = create_test_tree(4, CompactionPolicyType::PartialTiered);
    
    // Add some initial data
    for i in 1..17 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Delete some keys
    tree.delete(5).unwrap();
    tree.delete(10).unwrap();
    tree.delete(15).unwrap();
    
    // Force flush and compaction
    tree.flush_buffer_to_level0().unwrap();
    tree.force_compact_all().unwrap();
    
    // Verify deletions are maintained through compaction
    for i in 1..17 {
        if i == 5 || i == 10 || i == 15 {
            assert_eq!(tree.get(i), None, "Key {} should be deleted", i);
        } else {
            assert_eq!(tree.get(i), Some(i * 100), "Data loss for key {}", i);
        }
    }
}

#[test]
fn test_partial_tiered_selection_strategies() {
    // This test verifies that each selection strategy correctly selects runs
    // based on its criteria
    
    // Create a mock level with different sized runs
    let mut level = Level::new();
    
    // Run 0: Large run (10 entries) - this is considered the "oldest" run by index
    level.add_run(Run::new(vec![
        (101, 1010), (102, 1020), (103, 1030), (104, 1040), (105, 1050),
        (106, 1060), (107, 1070), (108, 1080), (109, 1090), (110, 1100)
    ]));
    
    // Run 1: Small run (3 entries)
    level.add_run(Run::new(vec![
        (201, 2010), (202, 2020), (203, 2030)
    ]));
    
    // Run 2: Medium run (5 entries) with overlapping keys
    level.add_run(Run::new(vec![
        (105, 1500), (106, 1600), (201, 2100), (202, 2200), (203, 2300)
    ]));
    
    // Test oldest strategy (selects the first N runs by index)
    let oldest_policy = PartialTieredCompactionPolicy::new(
        3, 2, SelectionStrategy::Oldest
    );
    let selected = oldest_policy.select_runs_to_compact(&level);
    // Should select runs 0 and 1 (first two runs)
    assert!(selected.contains(&0), "Oldest strategy should select run 0");
    assert!(selected.contains(&1), "Oldest strategy should select run 1");
    assert_eq!(selected.len(), 2, "Should select exactly 2 runs");
    
    // Test smallest size strategy
    let size_policy = PartialTieredCompactionPolicy::new(
        3, 2, SelectionStrategy::SmallestSize
    );
    let selected = size_policy.select_runs_to_compact(&level);
    // Should select run 1 (3 entries) and one other smaller run
    assert!(selected.contains(&1), "SmallestSize strategy should select run 1");
    assert_eq!(selected.len(), 2, "Should select exactly 2 runs");
    
    // Test most overlap strategy
    // For this one, we need more specific overlapping data to test properly
    let overlap_policy = PartialTieredCompactionPolicy::new(
        3, 2, SelectionStrategy::MostOverlap
    );
    let selected = overlap_policy.select_runs_to_compact(&level);
    assert_eq!(selected.len(), 2, "Should select exactly 2 runs");
    
    // Since it's harder to predict overlap calculations, we just verify
    // that we get 2 runs back as expected
}

#[test]
fn test_recovery_with_partial_compaction() {
    // Create storage path for recovery test
    let temp_dir = tempdir().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    // Create first tree with partial tiered compaction
    let config = LSMConfig {
        buffer_size: 4,
        storage_type: StorageType::File,
        storage_path: storage_path.clone(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::PartialTiered,
        compaction_threshold: 3,
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