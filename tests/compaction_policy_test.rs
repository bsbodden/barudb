use lsm_tree::run::{Run, RunStorage, StorageFactory, StorageOptions};
use lsm_tree::level::Level;
use lsm_tree::types::{Key, Value, CompactionPolicyType};
use lsm_tree::compaction::{CompactionPolicy, TieredCompactionPolicy, CompactionFactory};
use tempfile::tempdir;

fn create_test_storage() -> (std::sync::Arc<dyn RunStorage>, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    let storage = StorageFactory::create("file", options).unwrap();
    (storage, temp_dir)
}

fn create_run_with_data(data: Vec<(Key, Value)>, level: usize) -> Run {
    Run::new_for_level(data, level, 4.0)
}

#[test]
fn test_tiered_compaction_should_compact() {
    // Create a tiered compaction policy with threshold of 2 runs per level
    let tiered_policy = TieredCompactionPolicy::new(2);
    
    // Create a level with 1 run - should not need compaction
    let mut level = Level::new();
    level.add_run(create_run_with_data(vec![(1, 100)], 0));
    assert!(!tiered_policy.should_compact(&level, 0));
    
    // Add another run - should now need compaction since threshold is 2
    level.add_run(create_run_with_data(vec![(2, 200)], 0));
    assert!(tiered_policy.should_compact(&level, 0));
    
    // Test factory creation
    let factory_policy = CompactionFactory::create(&CompactionPolicyType::Tiered.to_string(), 2).unwrap();
    assert!(factory_policy.should_compact(&level, 0));
}

#[test]
fn test_tiered_compaction_select_runs() {
    let tiered_policy = TieredCompactionPolicy::new(2);
    
    // Create a level with 3 runs
    let mut level = Level::new();
    
    // Add three runs with increasing key ranges
    level.add_run(create_run_with_data(vec![(1, 100), (2, 200)], 0));
    level.add_run(create_run_with_data(vec![(3, 300), (4, 400)], 0));
    level.add_run(create_run_with_data(vec![(5, 500), (6, 600)], 0));
    
    // Select runs to compact - tiered policy should select all runs
    let runs_to_compact = tiered_policy.select_runs_to_compact(&level);
    assert_eq!(runs_to_compact.len(), 3);
}

#[test]
fn test_tiered_compaction_merge() {
    let (storage, _temp_dir) = create_test_storage();
    let tiered_policy = TieredCompactionPolicy::new(2);
    
    // Create a level with runs to merge
    let mut level0 = Level::new();
    let mut level1 = Level::new();
    
    // Create runs with overlapping ranges
    let mut run1 = create_run_with_data(vec![(1, 100), (3, 300)], 0);
    let mut run2 = create_run_with_data(vec![(2, 200), (4, 400)], 0);
    
    // Store runs
    let id1 = run1.store(&*storage, 0).unwrap();
    let id2 = run2.store(&*storage, 0).unwrap();
    
    // Set run IDs for later cleanup
    run1.id = Some(id1);
    run2.id = Some(id2);
    
    // Add runs to level
    level0.add_run(run1);
    level0.add_run(run2);
    
    // Test compaction
    let merged_run = tiered_policy.compact(&level0, &mut level1, &*storage, 0, 1).unwrap();
    
    // Verify merged run properties
    assert_eq!(merged_run.entry_count(), 4);
    assert_eq!(merged_run.level, Some(1));
    
    // Verify merged run contains all data sorted correctly
    let range = merged_run.range(0, 5);
    assert_eq!(range, vec![(1, 100), (2, 200), (3, 300), (4, 400)]);
}