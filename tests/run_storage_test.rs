use lsm_tree::run::{
    BlockCacheConfig, Error, FileStorage, Run, RunId, RunStorage, 
    StorageFactory, StorageOptions
};
use lsm_tree::types::{Key, StorageType};
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn test_file_storage_basic() {
    // Create a temporary directory for the test
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage
    let storage = FileStorage::new(options).unwrap();
    
    // Create a run with some data
    let run = Run::new(vec![(1, 100), (2, 200), (3, 300)]);
    
    // Store the run
    let run_id = storage.store_run(0, &run).unwrap();
    
    // Check that the run exists
    assert!(storage.run_exists(run_id).unwrap());
    
    // Get the run's metadata
    let metadata = storage.get_run_metadata(run_id).unwrap();
    assert_eq!(metadata.id, run_id);
    assert_eq!(metadata.block_count, 1);
    assert_eq!(metadata.entry_count, 3);
    
    // Load the run
    let loaded_run = storage.load_run(run_id).unwrap();
    
    // Check that the loaded run has the correct data
    assert_eq!(loaded_run.blocks.len(), 1);
    assert_eq!(loaded_run.data.len(), 3);
    assert_eq!(loaded_run.get(1), Some(100));
    assert_eq!(loaded_run.get(2), Some(200));
    assert_eq!(loaded_run.get(3), Some(300));
    assert_eq!(loaded_run.get(4), None);
    
    // Run query
    let range = loaded_run.range(1, 3);
    assert_eq!(range, vec![(1, 100), (2, 200)]);
    
    // Get storage stats
    let stats = storage.get_stats().unwrap();
    assert_eq!(stats.runs_per_level[0], 1);
    assert_eq!(stats.blocks_per_level[0], 1);
    assert_eq!(stats.entries_per_level[0], 3);
    assert!(stats.total_size_bytes > 0);
    
    // Delete the run
    storage.delete_run(run_id).unwrap();
    
    // Check that the run no longer exists
    assert!(!storage.run_exists(run_id).unwrap());
}

#[test]
fn test_file_storage_multiple_runs() {
    // Create a temporary directory for the test
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage
    let storage = FileStorage::new(options).unwrap();
    
    // Create multiple runs with different data
    let run1 = Run::new(vec![(1, 100), (2, 200)]);
    let run2 = Run::new(vec![(3, 300), (4, 400)]);
    let run3 = Run::new(vec![(5, 500), (6, 600)]);
    
    // Store the runs in different levels
    let run1_id = storage.store_run(0, &run1).unwrap();
    let run2_id = storage.store_run(0, &run2).unwrap();
    let run3_id = storage.store_run(1, &run3).unwrap();
    
    // List runs in level 0
    let level0_runs = storage.list_runs(0).unwrap();
    assert_eq!(level0_runs.len(), 2);
    assert!(level0_runs.contains(&run1_id));
    assert!(level0_runs.contains(&run2_id));
    
    // List runs in level 1
    let level1_runs = storage.list_runs(1).unwrap();
    assert_eq!(level1_runs.len(), 1);
    assert_eq!(level1_runs[0], run3_id);
    
    // Get storage stats
    let stats = storage.get_stats().unwrap();
    assert_eq!(stats.runs_per_level.len(), 2);
    assert_eq!(stats.runs_per_level[0], 2);
    assert_eq!(stats.runs_per_level[1], 1);
    assert_eq!(stats.blocks_per_level[0], 2);
    assert_eq!(stats.blocks_per_level[1], 1);
    assert_eq!(stats.entries_per_level[0], 4);
    assert_eq!(stats.entries_per_level[1], 2);
}

#[test]
fn test_run_serialization() {
    // Create a run with test data
    let mut run = Run::new(vec![(1, 100), (2, 200), (3, 300)]);
    
    // Serialize the run
    let serialized = run.serialize().unwrap();
    
    // Deserialize the run
    let deserialized = Run::deserialize(&serialized).unwrap();
    
    // Check that the deserialized run has the correct data
    assert_eq!(deserialized.blocks.len(), 1);
    assert_eq!(deserialized.data.len(), 3);
    assert_eq!(deserialized.get(1), Some(100));
    assert_eq!(deserialized.get(2), Some(200));
    assert_eq!(deserialized.get(3), Some(300));
    assert_eq!(deserialized.get(4), None);
    
    // Check run bounds
    assert_eq!(deserialized.min_key(), Some(1));
    assert_eq!(deserialized.max_key(), Some(3));
    
    // Check entries
    assert_eq!(deserialized.entry_count(), 3);
}

#[test]
fn test_run_storage_delete_sequence() {
    // Create a temporary directory for the test
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage
    let storage = FileStorage::new(options).unwrap();
    
    // Create and store a run
    let run1 = Run::new(vec![(1, 100), (2, 200)]);
    let run1_id = storage.store_run(0, &run1).unwrap();
    
    // Delete the run
    storage.delete_run(run1_id).unwrap();
    
    // Create and store another run - should have sequence 1 again
    let run2 = Run::new(vec![(3, 300), (4, 400)]);
    let run2_id = storage.store_run(0, &run2).unwrap();
    
    // Sequence numbers should be the same (since we deleted the first run)
    assert_eq!(run2_id.sequence, 1);
}

#[test]
fn test_run_id_serialization() {
    // Create a run ID
    let run_id = RunId::new(5, 123456789);
    
    // Convert to string
    let s = run_id.to_string();
    assert_eq!(s, "L05_R0123456789");
    
    // Parse back from string
    let parsed = RunId::from_string(&s).unwrap();
    assert_eq!(parsed.level, 5);
    assert_eq!(parsed.sequence, 123456789);
}

#[test]
#[ignore = "Large data test with 1000 elements; run explicitly with 'cargo test test_large_run -- --ignored'"]
fn test_large_run() {
    // Create a temporary directory for the test
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage
    let storage = FileStorage::new(options).unwrap();
    
    // Create a run with a lot of data
    let mut large_data = Vec::new();
    for i in 0..1000 {
        large_data.push((i, i * 10));
    }
    
    let run = Run::new(large_data);
    
    // Store the run
    let run_id = storage.store_run(0, &run).unwrap();
    
    // Load the run
    let loaded_run = storage.load_run(run_id).unwrap();
    
    // Check that the loaded run has the correct data
    assert_eq!(loaded_run.data.len(), 1000);
    for i in 0..1000 {
        assert_eq!(loaded_run.get(i), Some(i * 10));
    }
}

#[test]
fn test_empty_run() {
    // Create a temporary directory for the test
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage
    let storage = FileStorage::new(options).unwrap();
    
    // Create an empty run
    let empty_run = Run::new(vec![]);
    
    // Store the run
    let run_id = storage.store_run(0, &empty_run).unwrap();
    
    // Load the run
    let loaded_run = storage.load_run(run_id).unwrap();
    
    // Check that the loaded run is empty
    assert_eq!(loaded_run.data.len(), 0);
    assert_eq!(loaded_run.blocks.len(), 0);
    assert_eq!(loaded_run.min_key(), None);
    assert_eq!(loaded_run.max_key(), None);
    assert_eq!(loaded_run.entry_count(), 0);
    
    // Test range query on empty run
    let range = loaded_run.range(0, 100);
    assert_eq!(range.len(), 0);
}

#[test]
fn test_run_validation() {
    // Create a valid run
    let mut run = Run::new(vec![(1, 100), (2, 200), (3, 300)]);
    
    // Validate should pass
    assert!(run.validate().is_ok());
    
    // Create a run with unsealed block (would require access to internal fields)
    // We can't easily test this case due to encapsulation, but the validation logic is there
    
    // Serialize and deserialize to ensure internal state is consistent
    let serialized = run.serialize().unwrap();
    let deserialized = Run::deserialize(&serialized).unwrap();
    assert!(deserialized.validate().is_ok());
}

#[test]
fn test_storage_factory() {
    // Create a temporary directory for the test
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage using factory
    let storage = StorageFactory::create_from_type(StorageType::File, options).unwrap();
    
    // Try to create a run
    let run = Run::new(vec![(1, 100), (2, 200)]);
    
    // Store it through the trait object
    let run_id = storage.store_run(0, &run).unwrap();
    
    // Verify it exists
    assert!(storage.run_exists(run_id).unwrap());
    
    // Try to create an invalid storage type
    let result = StorageFactory::create("invalid_type", StorageOptions::default());
    assert!(result.is_err());
    
    if let Err(Error::Storage(msg)) = result {
        assert!(msg.contains("Unknown storage type"));
    } else {
        panic!("Expected Storage error");
    }
}

#[test]
fn test_error_cases() {
    // Create a temporary directory for the test
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage
    let storage = FileStorage::new(options).unwrap();
    
    // Attempt to load a non-existent run
    let nonexistent_id = RunId::new(999, 999);
    let result = storage.load_run(nonexistent_id);
    assert!(result.is_err());
    
    if let Err(Error::Storage(msg)) = result {
        assert!(msg.contains("Run not found"));
    } else {
        panic!("Expected Storage error");
    }
    
    // Attempt to get metadata for a non-existent run
    let result = storage.get_run_metadata(nonexistent_id);
    assert!(result.is_err());
    
    // Attempt to delete a non-existent run (should not error, just be a no-op)
    let result = storage.delete_run(nonexistent_id);
    assert!(result.is_ok());
    
    // Create a run and then delete its files manually to cause an error
    let run = Run::new(vec![(1, 100)]);
    let run_id = storage.store_run(0, &run).unwrap();
    
    // Delete the data file but leave the metadata
    let run_path = temp_dir.path()
        .join("runs")
        .join("level_0")
        .join(format!("{}.bin", run_id.to_string()));
    std::fs::remove_file(run_path).unwrap();
    
    // Attempting to load should fail
    let result = storage.load_run(run_id);
    assert!(result.is_err());
}

#[test]
fn test_extreme_keys() {
    // Create a temporary directory for the test
    let temp_dir = tempdir().unwrap();
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage
    let storage = FileStorage::new(options).unwrap();
    
    // Create a run with extreme key values
    let run = Run::new(vec![
        (Key::MIN, 100),     // Minimum possible key
        (Key::MAX, 200),     // Maximum possible key
        (0, 300),            // Zero key
        (-1, 400),           // Negative key
    ]);
    
    // Store the run
    let run_id = storage.store_run(0, &run).unwrap();
    
    // Load the run
    let loaded_run = storage.load_run(run_id).unwrap();
    
    // Check extreme values
    assert_eq!(loaded_run.get(Key::MIN), Some(100));
    assert_eq!(loaded_run.get(Key::MAX), Some(200));
    assert_eq!(loaded_run.get(0), Some(300));
    assert_eq!(loaded_run.get(-1), Some(400));
    
    // Check run bounds
    assert_eq!(loaded_run.min_key(), Some(Key::MIN));
    assert_eq!(loaded_run.max_key(), Some(Key::MAX));
}

#[test]
fn test_create_if_missing_false() {
    // Create a non-existent path
    let random_path = format!("/tmp/nonexistent_path_{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos());
    
    let options = StorageOptions {
        base_path: random_path.into(),
        create_if_missing: false, // This should cause an error
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage should fail
    let result = FileStorage::new(options);
    assert!(result.is_err());
    
    if let Err(Error::Storage(msg)) = result {
        assert!(msg.contains("Base directory does not exist"));
    } else {
        panic!("Expected Storage error");
    }
}

#[test]
fn test_block_cache() {
    // Create a temporary directory for the test
    let temp_dir = tempdir().unwrap();
    
    // Create custom cache config for testing
    let cache_config = BlockCacheConfig {
        max_capacity: 10,
        ttl: Duration::from_secs(10),
        cleanup_interval: Duration::from_secs(1),
    };
    
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage with custom cache
    let storage = FileStorage::with_cache_config(options, cache_config).unwrap();
    
    // Create and store a run
    let run = Run::new(vec![(1, 100), (2, 200), (3, 300)]);
    let run_id = storage.store_run(0, &run).unwrap();
    
    // Load a block (which should cache it)
    let block1 = storage.load_block(run_id, 0).unwrap();
    
    // Verify we can get the value from the block
    assert_eq!(block1.get(&1), Some(100));
    
    // Get cache stats - should show 1 miss and 0 hits since this was the first load
    let stats = storage.get_cache().get_stats();
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 0);
    
    // Load the same block again - should be cached now
    let block2 = storage.load_block(run_id, 0).unwrap();
    
    // Verify we got the same data
    assert_eq!(block2.get(&1), Some(100));
    
    // Get cache stats - should show 1 miss and 1 hit now
    let stats = storage.get_cache().get_stats();
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 1);
}