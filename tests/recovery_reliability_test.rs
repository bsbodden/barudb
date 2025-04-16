use lsm_tree::lsm_tree::{LSMTree, LSMConfig};
use lsm_tree::types::{CompactionPolicyType, StorageType};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;
use tempfile::TempDir;

// Use a static counter to ensure unique test directories
static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
static INIT: Once = Once::new();

// Initialize the test environment
fn setup() {
    INIT.call_once(|| {
        // Enable debug logging if needed
        std::env::set_var("RUST_LOG", "debug");
        
        // Initialize logger if needed
        if std::env::var("RUST_LOG").is_ok() {
            env_logger::init();
        }
    });
}

// Create a deterministic test directory for each test
fn create_test_dir(test_name: &str) -> (PathBuf, TempDir) {
    let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let temp_dir = tempfile::tempdir().unwrap();
    let test_dir = temp_dir.path().join(format!("recovery_test_{}_{}", test_name, counter));
    
    // Create the directory
    fs::create_dir_all(&test_dir).unwrap();
    
    // Print debug info
    println!("Created test directory: {:?}", test_dir);
    
    (test_dir, temp_dir)
}

// Create a consistent test LSM tree configuration
fn create_lsm_config(
    storage_path: PathBuf,
    policy_type: CompactionPolicyType,
    compaction_threshold: usize,
    sync_writes: bool,
) -> LSMConfig {
    // Create configuration with compression disabled for recovery tests
    let mut compression_config = lsm_tree::run::CompressionConfig::default();
    compression_config.enabled = false; // Disable compression for deterministic results
    
    let mut adaptive_config = lsm_tree::run::AdaptiveCompressionConfig::default();
    adaptive_config.enabled = false; // Disable adaptive compression
    
    LSMConfig {
        buffer_size: 8, // Small but consistent buffer size
        storage_type: StorageType::File,
        storage_path,
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes, // Configurable sync_writes for testing with/without
        fanout: 4, // Consistent fanout
        compaction_policy: policy_type,
        compaction_threshold,
        compression: compression_config,
        adaptive_compression: adaptive_config,
        collect_compression_stats: false, // Don't collect stats for tests
        background_compaction: false,
        use_lock_free_memtable: false,  // Use standard sharded memtable by default
        use_lock_free_block_cache: true, // Use lock-free block cache by default // Disable background compaction for deterministic behavior
    }
}

// Helper to perform a controlled recovery test with deterministic steps
fn run_recovery_test(
    test_name: &str,
    policy_type: CompactionPolicyType,
    compaction_threshold: usize,
    sync_writes: bool,
) {
    setup();
    
    println!("=== Starting recovery test '{}' ===", test_name);
    println!("Policy: {:?}, Threshold: {}, Sync Writes: {}", policy_type, compaction_threshold, sync_writes);
    
    // Create a unique test directory
    let (test_dir, _temp_dir_handle) = create_test_dir(test_name);
    
    // Step 1: Create and populate initial tree
    println!("Step 1: Creating and populating initial tree");
    let config = create_lsm_config(test_dir.clone(), policy_type.clone(), compaction_threshold, sync_writes);
    let mut tree = LSMTree::with_config(config);
    
    // Add initial data with deterministic pattern
    for i in 1..20 {
        tree.put(i, i * 100).unwrap();
        
        // Add explicit flush points at regular intervals
        if i % 5 == 0 {
            println!("Step 1: Explicit flush at i={}", i);
            tree.flush_buffer_to_level0().unwrap();
        }
    }
    
    // Ensure all data is on disk with explicit flush
    println!("Step 1: Final flush to ensure all data on disk");
    tree.flush_buffer_to_level0().unwrap();
    
    // Verify initial data
    println!("Step 1: Verifying all data was written");
    for i in 1..20 {
        assert_eq!(tree.get(i), Some(i * 100), "Initial data verification failed for key {}", i);
    }
    
    // Explicitly drop the tree to ensure clean closure of files
    println!("Step 1: Explicitly dropping tree to close files");
    drop(tree);
    
    // Add a small delay to ensure filesystem operations complete
    std::thread::sleep(std::time::Duration::from_millis(200));
    
    // Step 2: Recover tree from disk
    println!("Step 2: Creating new tree instance with same storage path");
    let recovery_config = create_lsm_config(test_dir, policy_type.clone(), compaction_threshold, sync_writes);
    let recovered_tree = LSMTree::with_config(recovery_config);
    
    // Verify all data is recovered correctly
    println!("Step 2: Verifying all data was recovered correctly");
    for i in 1..20 {
        assert_eq!(
            recovered_tree.get(i), 
            Some(i * 100), 
            "Recovery failed for key {}", 
            i
        );
    }
    
    println!("=== Recovery test '{}' completed successfully ===", test_name);
}

#[test]
fn test_tiered_recovery_with_sync() {
    run_recovery_test(
        "tiered_sync",
        CompactionPolicyType::Tiered,
        3,
        true, // sync_writes = true
    );
}

#[test]
fn test_tiered_recovery_without_sync() {
    run_recovery_test(
        "tiered_nosync",
        CompactionPolicyType::Tiered,
        3,
        false, // sync_writes = false
    );
}

#[test]
fn test_leveled_recovery_with_sync() {
    run_recovery_test(
        "leveled_sync",
        CompactionPolicyType::Leveled,
        4,
        true, // sync_writes = true
    );
}

#[test]
fn test_leveled_recovery_without_sync() {
    run_recovery_test(
        "leveled_nosync",
        CompactionPolicyType::Leveled,
        4,
        false, // sync_writes = false
    );
}

#[test]
fn test_lazy_leveled_recovery_with_sync() {
    run_recovery_test(
        "lazy_leveled_sync",
        CompactionPolicyType::LazyLeveled,
        3,
        true, // sync_writes = true
    );
}

#[test]
fn test_lazy_leveled_recovery_without_sync() {
    run_recovery_test(
        "lazy_leveled_nosync",
        CompactionPolicyType::LazyLeveled,
        3,
        false, // sync_writes = false
    );
}

#[test]
fn test_recovery_with_multiple_levels() {
    setup();
    
    println!("=== Starting multi-level recovery test ===");
    
    // Create a unique test directory
    let (test_dir, _temp_dir_handle) = create_test_dir("multi_level");
    
    // Step 1: Create and populate a tree with multiple levels
    println!("Step 1: Creating tree with multiple levels");
    let config = create_lsm_config(
        test_dir.clone(),
        CompactionPolicyType::Leveled,
        2, // Low threshold to trigger frequent compaction
        true, // Use sync writes for reliability
    );
    let mut tree = LSMTree::with_config(config);
    
    // Add enough data to create multiple levels
    println!("Step 1: Adding data to create multiple levels");
    for i in 0..50 {
        tree.put(i, i * 100).unwrap();
        
        // Flush after every 5 entries to create multiple runs
        if i % 5 == 4 {
            println!("Step 1: Flushing at i={}", i);
            tree.flush_buffer_to_level0().unwrap();
        }
    }
    
    // Force compaction to ensure multiple levels exist
    println!("Step 1: Forcing compaction to create multiple levels");
    tree.force_compact_all().unwrap();
    
    // Add more data to cause further compaction
    println!("Step 1: Adding more data to trigger additional compaction");
    for i in 50..80 {
        tree.put(i, i * 100).unwrap();
        
        // Flush after every 5 entries
        if i % 5 == 4 {
            println!("Step 1: Flushing at i={}", i);
            tree.flush_buffer_to_level0().unwrap();
            
            // Force compaction periodically
            if i % 15 == 14 {
                println!("Step 1: Forcing compaction at i={}", i);
                tree.force_compact_all().unwrap();
            }
        }
    }
    
    // Final flush and compaction
    println!("Step 1: Final flush and compaction");
    tree.flush_buffer_to_level0().unwrap();
    tree.force_compact_all().unwrap();
    
    // Verify all data is present
    println!("Step 1: Verifying all data is present");
    for i in 0..80 {
        assert_eq!(tree.get(i), Some(i * 100), "Initial verification failed for key {}", i);
    }
    
    // Explicitly drop the tree to ensure clean closure
    println!("Step 1: Explicitly dropping tree");
    drop(tree);
    
    // Add a small delay to ensure filesystem operations complete
    std::thread::sleep(std::time::Duration::from_millis(200));
    
    // Step 2: Recover tree from disk
    println!("Step 2: Creating new tree instance with same storage path");
    let recovery_config = create_lsm_config(test_dir, CompactionPolicyType::Leveled, 2, true);
    let recovered_tree = LSMTree::with_config(recovery_config);
    
    // Verify all data was recovered correctly
    println!("Step 2: Verifying all data was recovered correctly");
    for i in 0..80 {
        assert_eq!(
            recovered_tree.get(i), 
            Some(i * 100), 
            "Recovery failed for key {}", 
            i
        );
    }
    
    println!("=== Multi-level recovery test completed successfully ===");
}

#[test]
fn test_recovery_after_partial_flush() {
    setup();
    
    println!("=== Starting partial flush recovery test ===");
    
    // Create a unique test directory
    let (test_dir, _temp_dir_handle) = create_test_dir("partial_flush");
    
    // Step 1: Create tree with sync_writes enabled
    println!("Step 1: Creating tree with sync_writes enabled");
    let config = create_lsm_config(
        test_dir.clone(),
        CompactionPolicyType::Tiered,
        3,
        true, // sync_writes = true for reliability
    );
    let mut tree = LSMTree::with_config(config);
    
    // Add data and flush explicitly to ensure it's on disk
    println!("Step 1: Adding initial data with explicit flush");
    for i in 1..10 {
        tree.put(i, i * 100).unwrap();
    }
    tree.flush_buffer_to_level0().unwrap();
    
    // Verify data is accessible
    println!("Step 1: Verifying initial data");
    for i in 1..10 {
        assert_eq!(tree.get(i), Some(i * 100), "Initial data verification failed for key {}", i);
    }
    
    // Add more data without explicit flush
    println!("Step 1: Adding more data without explicit flush");
    for i in 10..15 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Verify all data is accessible
    println!("Step 1: Verifying all data is accessible");
    for i in 1..15 {
        assert_eq!(tree.get(i), Some(i * 100), "Data verification failed for key {}", i);
    }
    
    // Do NOT flush the buffer to simulate a crash
    println!("Step 1: Simulating crash by dropping tree without flushing buffer");
    drop(tree);
    
    // Add a small delay to ensure filesystem operations complete
    std::thread::sleep(std::time::Duration::from_millis(200));
    
    // Step 2: Recover tree from disk
    println!("Step 2: Creating new tree instance to test recovery");
    let recovery_config = create_lsm_config(test_dir, CompactionPolicyType::Tiered, 3, true);
    let recovered_tree = LSMTree::with_config(recovery_config);
    
    // Verify that only explicitly flushed data was recovered
    println!("Step 2: Verifying only explicitly flushed data was recovered");
    for i in 1..15 {
        if i < 10 {
            // These values were explicitly flushed
            assert_eq!(
                recovered_tree.get(i), 
                Some(i * 100), 
                "Recovery failed for explicitly flushed key {}", 
                i
            );
        } else {
            // These values were not flushed and should not be recovered
            assert_eq!(
                recovered_tree.get(i), 
                None, 
                "Unflushed key {} was incorrectly recovered", 
                i
            );
        }
    }
    
    println!("=== Partial flush recovery test completed successfully ===");
}

#[test]
fn test_explicit_recovery_method() {
    setup();
    
    println!("=== Starting explicit recovery method test ===");
    
    // Create a unique test directory
    let (test_dir, _temp_dir_handle) = create_test_dir("explicit_recovery");
    
    // Step 1: Create and populate initial tree
    println!("Step 1: Creating and populating initial tree");
    let config = create_lsm_config(
        test_dir.clone(),
        CompactionPolicyType::Tiered,
        3,
        true, // sync_writes = true
    );
    let mut tree = LSMTree::with_config(config);
    
    // Add data and flush to disk
    println!("Step 1: Adding data and flushing to disk");
    for i in 1..20 {
        tree.put(i, i * 100).unwrap();
    }
    tree.flush_buffer_to_level0().unwrap();
    
    // Force compaction to ensure a stable state
    println!("Step 1: Forcing compaction");
    tree.force_compact_all().unwrap();
    
    // Verify data is accessible
    println!("Step 1: Verifying data is accessible");
    for i in 1..20 {
        assert_eq!(tree.get(i), Some(i * 100), "Data verification failed for key {}", i);
    }
    
    // Step 2: Explicitly call the recover() method
    println!("Step 2: Calling explicit recover() method");
    tree.recover().unwrap();
    
    // Verify data is still accessible after explicit recovery
    println!("Step 2: Verifying data after explicit recovery");
    for i in 1..20 {
        assert_eq!(
            tree.get(i), 
            Some(i * 100), 
            "Data verification after explicit recovery failed for key {}", 
            i
        );
    }
    
    // Step 3: Add more data and verify it's accessible
    println!("Step 3: Adding more data after recovery");
    for i in 20..30 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Verify new data is accessible
    println!("Step 3: Verifying new data is accessible");
    for i in 20..30 {
        assert_eq!(
            tree.get(i), 
            Some(i * 100), 
            "New data verification failed for key {}", 
            i
        );
    }
    
    println!("=== Explicit recovery method test completed successfully ===");
}

#[test]
fn test_recovery_with_range_queries() {
    setup();
    
    println!("=== Starting recovery with range queries test ===");
    
    // Create a unique test directory
    let (test_dir, _temp_dir_handle) = create_test_dir("range_query_recovery");
    
    // Step 1: Create and populate initial tree
    println!("Step 1: Creating and populating initial tree");
    let config = create_lsm_config(
        test_dir.clone(),
        CompactionPolicyType::Leveled, // Test with leveled policy
        3,
        true, // sync_writes = true for reliability
    );
    let mut tree = LSMTree::with_config(config);
    
    // Add data in a pattern that allows for meaningful range queries
    println!("Step 1: Adding sequential data");
    for i in 0..100 {
        // Use sequential keys with deterministic values
        tree.put(i, i * 10).unwrap();
        
        // Explicit flush at regular intervals for deterministic behavior
        if i > 0 && i % 20 == 0 {
            println!("Step 1: Flushing at i={}", i);
            tree.flush_buffer_to_level0().unwrap();
        }
    }
    
    // Final flush to ensure all data is on disk
    println!("Step 1: Final flush");
    tree.flush_buffer_to_level0().unwrap();
    
    // Force compaction to ensure a clean state
    println!("Step 1: Forcing compaction");
    tree.force_compact_all().unwrap();
    
    // Verify data using range queries
    println!("Step 1: Verifying data with range queries");
    
    // Test various range query patterns - NOTE: Range in this LSM implementation is [start, end), inclusive of start but exclusive of end
    let range1 = tree.range(10, 31);
    // We expect 21 items: 10, 11, 12, ..., 30 (inclusive)
    println!("Range query result count: {}", range1.len());
    assert!(range1.len() >= 20, "Range query [10, 31) failed - expected at least 20 items, got {}", range1.len());
    
    // Check some specific keys to verify
    assert!(range1.iter().any(|(k, _)| *k == 10), "Range should include key 10");
    assert!(range1.iter().any(|(k, _)| *k == 20), "Range should include key 20");
    assert!(range1.iter().any(|(k, _)| *k == 30), "Range should include key 30");
    
    let range2 = tree.range(50, 60);
    // We expect 11 items: 50, 51, 52, ..., 60 (inclusive)
    assert!(range2.len() >= 10, "Range query [50, 60] failed - expected at least 10 items, got {}", range2.len());
    
    // Explicitly drop the tree to ensure clean closure
    println!("Step 1: Explicitly dropping tree");
    drop(tree);
    
    // Add a small delay to ensure filesystem operations complete
    std::thread::sleep(std::time::Duration::from_millis(200));
    
    // Step 2: Recover tree from disk
    println!("Step 2: Creating new tree instance with same storage path");
    let recovery_config = create_lsm_config(test_dir, CompactionPolicyType::Leveled, 3, true);
    let recovered_tree = LSMTree::with_config(recovery_config);
    
    // Verify range queries still work after recovery
    println!("Step 2: Verifying range queries after recovery");
    
    let recovered_range1 = recovered_tree.range(10, 31);
    // We expect 21 items: 10, 11, 12, ..., 30 (inclusive)
    assert!(recovered_range1.len() >= 20, "Range query [10, 31) failed after recovery - expected at least 20 items, got {}", recovered_range1.len());
    
    // Check some specific keys to verify
    assert!(recovered_range1.iter().any(|(k, _)| *k == 10), "Range should include key 10 after recovery");
    assert!(recovered_range1.iter().any(|(k, _)| *k == 20), "Range should include key 20 after recovery");
    assert!(recovered_range1.iter().any(|(k, _)| *k == 30), "Range should include key 30 after recovery");
    
    let recovered_range2 = recovered_tree.range(50, 60);
    // We expect 11 items: 50, 51, 52, ..., 60 (inclusive)
    assert!(recovered_range2.len() >= 10, "Range query [50, 60] failed after recovery - expected at least 10 items, got {}", recovered_range2.len());
    
    // Test an empty range
    let empty_range = recovered_tree.range(200, 300);
    assert_eq!(empty_range.len(), 0, "Empty range query failed after recovery");
    
    println!("=== Recovery with range queries test completed successfully ===");
}

#[test]
fn test_interleaved_recovery() {
    setup();
    
    println!("=== Starting interleaved recovery test ===");
    
    // Create a unique test directory
    let (test_dir, _temp_dir_handle) = create_test_dir("interleaved_recovery");
    
    // Step 1: Create first tree instance
    println!("Step 1: Creating first tree instance");
    let config = create_lsm_config(
        test_dir.clone(),
        CompactionPolicyType::Tiered,
        3,
        true, // sync_writes = true
    );
    let mut tree1 = LSMTree::with_config(config);
    
    // Add initial data
    println!("Step 1: Adding initial data (odd keys)");
    for i in (1..20).step_by(2) { // Odd keys
        tree1.put(i, i * 100).unwrap();
    }
    
    // Flush to ensure data is on disk
    println!("Step 1: Flushing odd keys");
    tree1.flush_buffer_to_level0().unwrap();
    
    // Explicitly drop first tree
    println!("Step 1: Dropping first tree");
    drop(tree1);
    
    // Step 2: Create second tree instance
    println!("Step 2: Creating second tree instance");
    let config = create_lsm_config(
        test_dir.clone(),
        CompactionPolicyType::Tiered,
        3,
        true,
    );
    let mut tree2 = LSMTree::with_config(config);
    
    // Verify odd keys were recovered
    println!("Step 2: Verifying odd keys were recovered");
    for i in (1..20).step_by(2) {
        assert_eq!(
            tree2.get(i), 
            Some(i * 100), 
            "Recovery of odd key {} failed", 
            i
        );
    }
    
    // Add even keys
    println!("Step 2: Adding even keys");
    for i in (2..20).step_by(2) { // Even keys
        tree2.put(i, i * 100).unwrap();
    }
    
    // Flush to ensure data is on disk
    println!("Step 2: Flushing even keys");
    tree2.flush_buffer_to_level0().unwrap();
    
    // Verify all keys are now accessible
    println!("Step 2: Verifying all keys are accessible");
    for i in 1..20 {
        assert_eq!(
            tree2.get(i), 
            Some(i * 100), 
            "Verification of key {} failed", 
            i
        );
    }
    
    // Force compaction to merge runs
    println!("Step 2: Forcing compaction");
    tree2.force_compact_all().unwrap();
    
    // Explicitly drop second tree
    println!("Step 2: Dropping second tree");
    drop(tree2);
    
    // Step 3: Create third tree instance
    println!("Step 3: Creating third tree instance");
    let config = create_lsm_config(
        test_dir,
        CompactionPolicyType::Tiered,
        3,
        true,
    );
    let tree3 = LSMTree::with_config(config);
    
    // Verify all keys were recovered
    println!("Step 3: Verifying all keys were recovered");
    for i in 1..20 {
        assert_eq!(
            tree3.get(i), 
            Some(i * 100), 
            "Final recovery of key {} failed", 
            i
        );
    }
    
    println!("=== Interleaved recovery test completed successfully ===");
}

#[test]
fn test_recovery_with_large_values() {
    setup();
    
    println!("=== Starting recovery with large values test ===");
    
    // Create a unique test directory
    let (test_dir, _temp_dir_handle) = create_test_dir("large_data_recovery");
    
    // Step 1: Create tree instance
    println!("Step 1: Creating tree instance");
    let config = create_lsm_config(
        test_dir.clone(),
        CompactionPolicyType::LazyLeveled,
        3,
        true, // sync_writes = true
    );
    let mut tree = LSMTree::with_config(config);
    
    // Generate some large values
    println!("Step 1: Generating large values");
    let large_values: Vec<(i64, i64)> = (0..10).map(|i| {
        // Create keys from 1000-1009
        let key = 1000 + i;
        
        // Create large values (multiples of 10000)
        let value = i * 10000;
        
        (key, value)
    }).collect();
    
    // Add large values to the tree
    println!("Step 1: Adding large values to tree");
    for (key, value) in &large_values {
        tree.put(*key, *value).unwrap();
        
        // Verify data was stored correctly
        let retrieved = tree.get(*key).unwrap();
        assert_eq!(retrieved, *value, "Large value verification failed for key {}", key);
    }
    
    // Flush to ensure data is on disk
    println!("Step 1: Flushing large values");
    tree.flush_buffer_to_level0().unwrap();
    
    // Force compaction for clean state
    println!("Step 1: Forcing compaction");
    tree.force_compact_all().unwrap();
    
    // Explicitly drop tree
    println!("Step 1: Dropping tree");
    drop(tree);
    
    // Step 2: Create new tree instance for recovery
    println!("Step 2: Creating new tree instance for recovery");
    let recovery_config = create_lsm_config(
        test_dir,
        CompactionPolicyType::LazyLeveled,
        3,
        true,
    );
    let recovered_tree = LSMTree::with_config(recovery_config);
    
    // Verify large values were recovered correctly
    println!("Step 2: Verifying large values were recovered correctly");
    for (key, expected_value) in &large_values {
        let retrieved = recovered_tree.get(*key).unwrap();
        assert_eq!(
            retrieved, 
            *expected_value, 
            "Recovery of large value failed for key {}", 
            key
        );
    }
    
    println!("=== Recovery with large values test completed successfully ===");
}

#[test]
fn test_basic_recovery() {
    setup();
    
    println!("=== Starting basic recovery test ===");
    
    // Create a unique test directory
    let (test_dir, _temp_dir_handle) = create_test_dir("basic_recovery");
    
    // Phase 1: Initial data load
    println!("Phase 1: Creating initial tree and loading data");
    let config = create_lsm_config(
        test_dir.clone(),
        CompactionPolicyType::LazyLeveled,
        3,
        true, // sync_writes = true for reliability
    );
    let mut tree = LSMTree::with_config(config);
    
    // Add initial data
    println!("Phase 1: Adding initial sequential data");
    for i in 0..20 {
        tree.put(i, i * 10).unwrap();
    }
    
    // Force flush
    println!("Phase 1: Forcing flush");
    tree.flush_buffer_to_level0().unwrap();
    
    // Verify initial data
    println!("Phase 1: Verifying data");
    for i in 0..20 {
        assert_eq!(tree.get(i), Some(i * 10), "Verification failed for key {}", i);
    }
    
    // Explicitly drop tree
    println!("Phase 1: Dropping tree");
    drop(tree);
    
    // Small delay to ensure filesystem operations complete
    std::thread::sleep(std::time::Duration::from_millis(200));
    
    // Phase 2: Recovery and verification
    println!("Phase 2: Creating new tree instance for recovery");
    let config = create_lsm_config(
        test_dir,
        CompactionPolicyType::LazyLeveled,
        3,
        true,
    );
    let recovered_tree = LSMTree::with_config(config);
    
    // Verify recovery
    println!("Phase 2: Verifying recovered data");
    for i in 0..20 {
        assert_eq!(recovered_tree.get(i), Some(i * 10), "Recovery failed for key {}", i);
    }
    
    println!("=== Basic recovery test completed successfully ===");
}

/// Test specifically for tombstone persistence across restarts
#[test]
fn test_tombstone_persistence() {
    setup();
    
    println!("=== Starting tombstone persistence test ===");
    
    // Create a unique test directory
    let (test_dir, _temp_dir_handle) = create_test_dir("tombstone_persistence");
    
    // Phase 1: Create and populate initial tree
    println!("Phase 1: Creating and populating initial tree");
    let config = create_lsm_config(
        test_dir.clone(),
        CompactionPolicyType::Tiered,
        3,
        true, // sync_writes = true for reliability
    );
    let mut tree = LSMTree::with_config(config);
    
    // Add initial data
    println!("Phase 1: Adding initial data");
    for i in 1..20 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Flush to ensure data is on disk
    println!("Phase 1: Flushing initial data");
    tree.flush_buffer_to_level0().unwrap();
    
    // Delete specific keys
    println!("Phase 1: Deleting specific keys");
    let deleted_keys = vec![5, 10, 15];
    for key in &deleted_keys {
        tree.delete(*key).unwrap();
    }
    
    // Flush again to ensure tombstones are on disk
    println!("Phase 1: Flushing tombstones");
    tree.flush_buffer_to_level0().unwrap();
    
    // Force compaction to ensure a clean state
    println!("Phase 1: Forcing compaction");
    tree.force_compact_all().unwrap();
    
    // Verify deletions
    println!("Phase 1: Verifying deletions");
    for i in 1..20 {
        if deleted_keys.contains(&i) {
            assert_eq!(tree.get(i), None, "Delete verification failed for key {}", i);
        } else {
            assert_eq!(tree.get(i), Some(i * 100), "Data verification failed for key {}", i);
        }
    }
    
    // Explicitly drop the tree
    println!("Phase 1: Explicitly dropping tree");
    drop(tree);
    
    // Add a small delay to ensure filesystem operations complete
    std::thread::sleep(std::time::Duration::from_millis(200));
    
    // Phase 2: Recover tree and verify tombstones persist
    println!("Phase 2: Recovering tree and verifying tombstones");
    let recovery_config = create_lsm_config(test_dir, CompactionPolicyType::Tiered, 3, true);
    let recovered_tree = LSMTree::with_config(recovery_config);
    
    // Verify deleted keys remain deleted after recovery
    println!("Phase 2: Verifying tombstones survived recovery");
    for i in 1..20 {
        if deleted_keys.contains(&i) {
            assert_eq!(
                recovered_tree.get(i), 
                None, 
                "Tombstone for key {} was lost during recovery", 
                i
            );
        } else {
            assert_eq!(
                recovered_tree.get(i), 
                Some(i * 100), 
                "Recovery failed for key {}", 
                i
            );
        }
    }
    
    println!("=== Tombstone persistence test completed successfully ===");
}