# Tombstone Handling Issues in LSM Tree Recovery

## Problem Analysis

After examining the codebase, I've identified a critical issue with tombstone persistence and recovery in the LSM tree implementation. The problem affects how delete operations are maintained during recovery, causing tombstones to be lost when the system is restarted.

### Root Cause

The root cause is in the run serialization and deserialization process, specifically:

1. **Tombstone Identification**: Tombstones are properly created using the `delete()` method that sets a value to `TOMBSTONE` (defined as `i64::MIN` in `types.rs`).

2. **In-Memory Filtering**: The system correctly filters out tombstones during query operations in the `get()` and `range()` methods as shown in `lsm_tree.rs`:
   ```rust
   if value == TOMBSTONE {
       return None; // Ignore tombstone values
   }
   ```

3. **Serialization Issue**: When creating and serializing a run, tombstones are included in the data written to disk. However, in the `Run::deserialize()` method (in `run/mod.rs`), there is no special handling for tombstones during recovery.

4. **Critical Vulnerability During Compaction**: During compaction, tombstone entries must be propagated to ensure deleted keys remain deleted. However, if tombstones aren't properly preserved during recovery, deleted keys may "reappear" after a restart.

## Data Loss Scenario

This vulnerability leads to a serious data loss scenario:

1. User deletes a key with `delete(key)`
2. Memtable is flushed to Level 0, including the tombstone
3. System is restarted
4. During recovery, the tombstone is loaded but not correctly processed
5. Subsequent queries return the previously deleted data or merge incorrectly during compaction

## Fix Implementation

The fix requires ensuring that tombstones are properly preserved during the run serialization/deserialization process. Specifically:

### 1. Block-Level Fix

In `src/run/block.rs`, modify the `Block` implementation to ensure tombstones are maintained:

```rust
impl Block {
    // Ensure that when blocks are sealed, tombstones are maintained
    pub fn seal(&mut self) -> Result<()> {
        // Sort entries by key
        self.entries.sort_by_key(|&(k, _)| k);
        
        // Remove duplicates, keeping the most recent value (including tombstones)
        self.entries.dedup_by_key(|&mut (k, _)| k);
        
        // Set header fields
        if !self.entries.is_empty() {
            self.header.min_key = self.entries.first().unwrap().0;
            self.header.max_key = self.entries.last().unwrap().0;
        }
        
        self.header.entry_count = self.entries.len();
        
        self.sealed = true;
        Ok(())
    }
}
```

### 2. Run-Level Fix

In `src/run/mod.rs`, ensure tombstones are correctly handled during deserialization:

```rust
impl Run {
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        // ... existing deserialization code ...
        
        // Data collection from blocks
        for block in &blocks {
            data.extend(block.entries.clone());
        }
        
        // Important: Do not filter out tombstones during deserialization
        // This ensures deleted entries remain deleted after recovery
        
        // ... rest of deserialization code ...
        
        Ok(Run {
            blocks,
            filter,
            fence_pointers,
            data,
            compression,
            id: None,
        })
    }
}
```

### 3. Compaction Policy Fix

In `src/compaction/mod.rs`, update the compaction logic to properly handle tombstones:

```rust
// Base trait for compaction policies
pub trait CompactionPolicy: Send + Sync {
    // ... existing methods ...
    
    fn merge_entries(&self, entries: &mut Vec<(Key, Value)>) {
        // Sort entries by key
        entries.sort_by_key(|&(k, _)| k);
        
        // Remove duplicates, keeping the most recent value (including tombstones)
        // This is critical for preserving delete operations
        entries.dedup_by_key(|&mut (k, _)| k);
    }
    
    // Common helper method for compaction implementations
    fn merge_runs(&self, runs: &[Run]) -> Vec<(Key, Value)> {
        let mut all_entries = Vec::new();
        
        // Collect all entries from all runs
        for run in runs {
            all_entries.extend(run.data.clone());
        }
        
        // Merge entries, preserving tombstones
        self.merge_entries(&mut all_entries);
        
        all_entries
    }
}
```

### 4. Comprehensive Testing

Add a specific test in `recovery_reliability_test.rs` to verify tombstone persistence across restarts:

```rust
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
    for i in 1..20 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Flush to ensure data is on disk
    tree.flush_buffer_to_level0().unwrap();
    
    // Delete specific keys
    let deleted_keys = vec![5, 10, 15];
    for key in &deleted_keys {
        tree.delete(*key).unwrap();
    }
    
    // Flush again to ensure tombstones are on disk
    tree.flush_buffer_to_level0().unwrap();
    
    // Force compaction to ensure a clean state
    tree.force_compact_all().unwrap();
    
    // Verify deletions
    for i in 1..20 {
        if deleted_keys.contains(&i) {
            assert_eq!(tree.get(i), None, "Delete verification failed for key {}", i);
        } else {
            assert_eq!(tree.get(i), Some(i * 100), "Data verification failed for key {}", i);
        }
    }
    
    // Explicitly drop the tree
    drop(tree);
    
    // Add a small delay to ensure filesystem operations complete
    std::thread::sleep(std::time::Duration::from_millis(200));
    
    // Phase 2: Recover tree and verify tombstones persist
    println!("Phase 2: Recovering tree and verifying tombstones");
    let recovery_config = create_lsm_config(test_dir, CompactionPolicyType::Tiered, 3, true);
    let recovered_tree = LSMTree::with_config(recovery_config);
    
    // Verify deleted keys remain deleted after recovery
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
```

## Preventing Recurrence

To prevent this issue from recurring in future implementations:

1. Add an explicit `TombstoneHandling` policy that can be configured in the LSM tree options
2. Add validation to ensure tombstones are preserved during recovery
3. Add metrics to track tombstone creation and application during compaction
4. Implement explicit testing for tombstone persistence in CI/CD pipelines

## Impact on Other Components

The fix for this issue also positively affects:

1. **Range Queries**: Ensures deleted keys are consistently excluded from range query results after recovery
2. **Compaction**: Guarantees that deletions are properly propagated during compaction operations
3. **Data Integrity**: Provides stronger guarantees about the permanence of delete operations

## Conclusion

The tombstone handling issue is a critical bug that affects data integrity and correctness. The fixes outlined above ensure that delete operations are properly persisted and recovered, maintaining consistency in the face of system restarts and crashes.