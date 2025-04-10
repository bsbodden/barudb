use crate::level::Level;
use crate::memtable::Memtable;
use crate::run::{self, Run, RunStorage, StorageFactory, StorageOptions, StorageStats};
use crate::types::{Key, Result, Value, TOMBSTONE, CompactionPolicyType, StorageType};
use crate::compaction::{CompactionPolicy, CompactionFactory};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Configuration for LSM tree
#[derive(Debug, Clone)]
pub struct LSMConfig {
    /// Size of the memory buffer in pages
    pub buffer_size: usize,
    
    /// Storage backend to use (File, LSF, MMap)
    pub storage_type: StorageType,
    
    /// Base path for storage
    pub storage_path: PathBuf,
    
    /// Whether to create storage path if missing
    pub create_path_if_missing: bool,
    
    /// Maximum open files for storage
    pub max_open_files: usize,
    
    /// Whether to sync writes to disk
    pub sync_writes: bool,
    
    /// Fanout/size ratio between levels (T)
    /// This is the ratio of sizes between adjacent levels in the LSM tree
    pub fanout: usize,
    
    /// Compaction policy to use (Tiered, Leveled, etc.)
    pub compaction_policy: CompactionPolicyType,
    
    /// Threshold for triggering compaction (runs per level for tiered,
    /// size ratio for leveled)
    pub compaction_threshold: usize,
}

impl Default for LSMConfig {
    fn default() -> Self {
        Self {
            buffer_size: 128,
            storage_type: StorageType::File,
            storage_path: PathBuf::from("./data"),
            create_path_if_missing: true,
            max_open_files: 1000,
            sync_writes: true,
            fanout: 4, // Default fanout of 4 (each level is 4x larger than the previous)
            compaction_policy: CompactionPolicyType::Tiered,
            compaction_threshold: 4, // Default to compact after 4 runs in a level
        }
    }
}

pub struct LSMTree {
    buffer: Arc<RwLock<Memtable>>,
    levels: Vec<Level>,
    storage: Arc<dyn RunStorage>,
    config: LSMConfig,
    compaction_policy: Box<dyn CompactionPolicy>,
}

impl LSMTree {
    /// Create a new LSM tree with default configuration
    pub fn new(buffer_size: usize) -> Self {
        Self::with_config(LSMConfig {
            buffer_size,
            ..Default::default()
        })
    }
    
    /// Create a new LSM tree with custom configuration
    pub fn with_config(config: LSMConfig) -> Self {
        // Create storage options from config
        let storage_options = StorageOptions {
            base_path: config.storage_path.clone(),
            create_if_missing: config.create_path_if_missing,
            max_open_files: config.max_open_files,
            sync_writes: config.sync_writes,
        };
        
        // Create storage backend
        let storage = StorageFactory::create_from_type(config.storage_type.clone(), storage_options)
            .expect("Failed to create storage backend");
        
        // Create compaction policy
        let compaction_policy = CompactionFactory::create_from_type(
            config.compaction_policy.clone(), 
            config.compaction_threshold
        ).expect("Failed to create compaction policy");
        
        let mut tree = Self {
            buffer: Arc::new(RwLock::new(Memtable::new(config.buffer_size))),
            levels: Vec::new(),
            storage,
            config,
            compaction_policy,
        };
        
        // Recover state from disk if available
        tree.recover_from_disk();
        
        tree
    }
    
    /// Recover state from disk (internal implementation)
    fn recover_from_disk(&mut self) {
        // Initialize empty levels array
        self.levels = Vec::new();
        
        // Attempt to find all levels with runs
        let max_level = 10; // Arbitrary limit for search
        
        for level in 0..max_level {
            match self.storage.list_runs(level) {
                Ok(run_ids) => {
                    if !run_ids.is_empty() {
                        // Create this level since it has runs
                        while self.levels.len() <= level {
                            self.levels.push(Level::new());
                        }
                        
                        // Load all runs for this level
                        for run_id in run_ids {
                            match self.storage.load_run(run_id) {
                                Ok(mut run) => {
                                    // Set run ID for future reference
                                    run.id = Some(run_id);
                                    // Add run to level
                                    self.levels[level].add_run(run);
                                    println!("Recovered run {} in level {}", run_id, level);
                                },
                                Err(e) => {
                                    // Log error but continue recovery
                                    eprintln!("Failed to load run {}: {:?}", run_id, e);
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    // Just log the error and continue
                    eprintln!("Error listing runs for level {}: {:?}", level, e);
                }
            }
        }
        
        println!("Recovered {} levels from disk", self.levels.len());
    }
    
    /// Public method to explicitly recover state from disk
    /// This can be used after a crash or to rebuild the database state
    pub fn recover(&mut self) -> Result<()> {
        // Clear current state
        {
            let buffer = self.buffer.write().unwrap();
            buffer.clear();
        }
        self.levels.clear();
        
        // Recover state from disk
        self.recover_from_disk();
        
        // Return success
        Ok(())
    }
    
    /// Get the current configuration
    pub fn get_config(&self) -> &LSMConfig {
        &self.config
    }

    pub fn put(&mut self, key: Key, value: Value) -> Result<()> {
        let flush_required = {
            let buffer = self.buffer.write().unwrap();
            let result = buffer.put(key, value);
            result.is_ok() && buffer.is_full()
        };

        if flush_required {
            self.flush_buffer_to_level0()?;
            // Check if compaction is needed
            self.compact_if_needed()?;
        }
        Ok(())
    }

    pub fn get(&self, key: Key) -> Option<Value> {
        // Check buffer first
        if let Some(value) = self.buffer.read().unwrap().get(&key) {
            if value == TOMBSTONE {
                return None; // Ignore tombstone values
            }
            return Some(value);
        }

        // Check levels
        for level in &self.levels {
            if let Some(value) = level.get(key) {
                if value == TOMBSTONE {
                    return None; // Ignore tombstone values
                }
                return Some(value);
            }
        }

        None
    }

    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        let mut results = self.buffer.read().unwrap().range(start, end);

        // Add results from levels
        for level in &self.levels {
            results.extend(level.range(start, end));
        }

        // Sort by key and remove duplicates, keeping only the most recent value
        results.sort_by_key(|&(key, _)| key);
        results.dedup_by_key(|&mut (key, _)| key);

        // Filter out tombstones
        results.retain(|&(_, value)| value != TOMBSTONE);

        results
    }

    pub fn delete(&mut self, key: Key) -> Result<()> {
        self.put(key, TOMBSTONE)
    }

    /// Flush the current buffer to level 0
    /// This method is public to allow for testing and manual flushing
    pub fn flush_buffer_to_level0(&mut self) -> Result<()> {
        let data = {
            let buffer = self.buffer.write().unwrap();
            let data = buffer.take_all(); // Get copy of all data
            buffer.clear();
            data
        };

        // Create a new run with a level-optimized Bloom filter
        // Level 0 is the first level after the memtable
        let fanout = self.config.fanout as f64;
        let mut run = Run::new_for_level(data, 0, fanout);
        
        // Store the run using configured storage
        let run_id = run.store(&*self.storage, 0)?;
        
        // Set the run ID for future reference
        run.id = Some(run_id);
        
        // Add the run to level 0
        if self.levels.is_empty() {
            self.levels.push(Level::new());
        }
        self.levels[0].add_run(run);
        
        Ok(())
    }
    
    /// Check all levels and perform compaction if needed according to policy
    pub fn compact_if_needed(&mut self) -> Result<()> {
        // Check each level for compaction needs
        for level_num in 0..self.levels.len() {
            if self.compaction_policy.should_compact(&self.levels[level_num], level_num) {
                self.compact_level(level_num)?;
            }
        }
        
        Ok(())
    }
    
    /// Perform a single compaction from one level to the next
    pub fn compact_level(&mut self, level_num: usize) -> Result<()> {
        // Ensure we have a level to compact into
        let target_level_num = level_num + 1;
        while self.levels.len() <= target_level_num {
            self.levels.push(Level::new());
        }
        
        // Clone source level to avoid borrow issues
        let source_level = self.levels[level_num].clone();
        
        // Perform compaction
        match self.compaction_policy.compact(
            &source_level,
            &mut self.levels[target_level_num],
            &*self.storage,
            level_num,
            target_level_num,
        ) {
            Ok(_) => {
                // Replace the old source level with a new empty level
                self.levels[level_num] = Level::new();
                
                // Check if next level now needs compaction
                if target_level_num < self.levels.len() &&
                   self.compaction_policy.should_compact(&self.levels[target_level_num], target_level_num) {
                    // Recursively compact next level
                    self.compact_level(target_level_num)?;
                }
                
                Ok(())
            },
            Err(e) => Err(e),
        }
    }
    
    /// Force compaction of all levels
    pub fn force_compact_all(&mut self) -> Result<()> {
        let level_count = self.levels.len();
        for level_num in 0..level_count {
            if level_num < self.levels.len() && !self.levels[level_num].get_runs().is_empty() {
                self.compact_level(level_num)?;
            }
        }
        
        Ok(())
    }
    
    /// Get storage statistics for the LSM tree
    pub fn get_storage_stats(&self) -> Result<StorageStats> {
        self.storage.get_stats().map_err(|e| {
            match e {
                run::Error::Io(io_err) => crate::types::Error::Io(io_err),
                _ => crate::types::Error::Other(format!("Storage error: {:?}", e))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_tree() -> (LSMTree, tempfile::TempDir) {
        let temp_dir = tempdir().unwrap();
        let config = LSMConfig {
            buffer_size: 128,
            storage_type: StorageType::File,
            storage_path: temp_dir.path().to_path_buf(),
            create_path_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
            fanout: 4, // Default fanout value
            compaction_policy: CompactionPolicyType::Tiered,
            compaction_threshold: 4, // Default threshold value
        };
        (LSMTree::with_config(config), temp_dir)
    }

    fn create_test_tree_with_data() -> (LSMTree, tempfile::TempDir) {
        let (mut lsm_tree, temp_dir) = create_test_tree();
        
        // Add some data
        lsm_tree.put(1, 100).unwrap();
        lsm_tree.put(2, 200).unwrap();
        lsm_tree.put(3, 300).unwrap();
        
        // Force buffer flush to disk
        lsm_tree.flush_buffer_to_level0().unwrap();
        
        (lsm_tree, temp_dir)
    }

    #[test]
    fn test_put_and_get() {
        let (mut lsm_tree, _temp_dir) = create_test_tree();
        lsm_tree.put(1, 100).unwrap();
        lsm_tree.put(2, 200).unwrap();

        assert_eq!(lsm_tree.get(1), Some(100));
        assert_eq!(lsm_tree.get(2), Some(200));
        assert_eq!(lsm_tree.get(3), None);
    }

    #[test]
    fn test_range_query() {
        let (mut lsm_tree, _temp_dir) = create_test_tree();
        lsm_tree.put(1, 100).unwrap();
        lsm_tree.put(2, 200).unwrap();
        lsm_tree.put(3, 300).unwrap();

        let range = lsm_tree.range(1, 4);
        assert_eq!(range, vec![(1, 100), (2, 200), (3, 300)]);
    }

    #[test]
    fn test_delete() {
        let (mut lsm_tree, _temp_dir) = create_test_tree();
        lsm_tree.put(1, 100).unwrap();
        lsm_tree.delete(1).unwrap();

        assert_eq!(lsm_tree.get(1), None);
    }
    
    #[test]
    fn test_configuration() {
        // Create a fresh, empty directory for testing configuration
        let temp_dir = tempdir().unwrap();
        
        // Create custom config first to avoid conflicts with recovery
        // and use a subdirectory to avoid any existing files
        let config_dir = temp_dir.path().join("config_test");
        std::fs::create_dir_all(&config_dir).unwrap();
        
        let config = LSMConfig {
            buffer_size: 512,
            storage_type: StorageType::File,
            storage_path: config_dir.clone(),
            create_path_if_missing: true, // Need this to create the path
            max_open_files: 50,
            sync_writes: true,
            fanout: 4, // Default fanout value
            compaction_policy: CompactionPolicyType::Tiered,
            compaction_threshold: 4, // Default threshold value
        };
        
        // Create LSM tree with custom config
        let lsm_tree = LSMTree::with_config(config.clone());
        
        // Test configuration is preserved
        assert_eq!(lsm_tree.get_config().buffer_size, 512);
        assert_eq!(lsm_tree.get_config().max_open_files, 50);
        assert_eq!(lsm_tree.get_config().storage_path, config_dir);
        assert_eq!(lsm_tree.get_config().storage_type, StorageType::File);
        assert!(lsm_tree.get_config().create_path_if_missing);
        assert!(lsm_tree.get_config().sync_writes);
        
        // Test default configuration in separate directory
        let default_dir = temp_dir.path().join("default_test");
        std::fs::create_dir_all(&default_dir).unwrap();
        
        // Override default path to use our test directory
        let default_tree = LSMTree::with_config(LSMConfig {
            buffer_size: 256,
            storage_path: default_dir.clone(),
            ..Default::default()
        });
        
        assert_eq!(default_tree.get_config().buffer_size, 256);
        assert_eq!(default_tree.get_config().storage_type, StorageType::File);
        assert!(default_tree.get_config().create_path_if_missing);
    }
    
    #[test]
    fn test_recovery_from_disk() {
        // Create a tree with data and flush to disk
        let (lsm_tree, temp_dir) = create_test_tree_with_data();
        
        // Verify data is present
        assert_eq!(lsm_tree.get(1), Some(100));
        assert_eq!(lsm_tree.get(2), Some(200));
        assert_eq!(lsm_tree.get(3), Some(300));
        
        // Path to storage - keep this for creating a new tree
        let storage_path = temp_dir.path().to_path_buf();
        
        // Drop the tree (simulating a crash or restart)
        drop(lsm_tree);
        
        // Create a new tree with the same storage path
        let config = LSMConfig {
            buffer_size: 128,
            storage_type: StorageType::File,
            storage_path: storage_path.clone(),
            create_path_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
            fanout: 4, // Default fanout value
            compaction_policy: CompactionPolicyType::Tiered,
            compaction_threshold: 4, // Default threshold value
        };
        
        // Create new tree - recovery should happen automatically
        let recovered_tree = LSMTree::with_config(config);
        
        // Verify data is still present after recovery
        assert_eq!(recovered_tree.get(1), Some(100));
        assert_eq!(recovered_tree.get(2), Some(200));
        assert_eq!(recovered_tree.get(3), Some(300));
    }
    
    #[test]
    fn test_explicit_recovery() {
        // Create a tree with data and flush to disk
        let (mut original_tree, temp_dir) = create_test_tree_with_data();
        
        // Add more data but don't flush
        original_tree.put(4, 400).unwrap();
        original_tree.put(5, 500).unwrap();
        
        // Path to storage - keep this for creating a new tree
        let storage_path = temp_dir.path().to_path_buf();
        
        // Drop the tree (simulating a crash)
        drop(original_tree);
        
        // Create a new tree with the same storage path
        let config = LSMConfig {
            buffer_size: 128,
            storage_type: StorageType::File,
            storage_path: storage_path,
            create_path_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
            fanout: 4, // Default fanout value
            compaction_policy: CompactionPolicyType::Tiered,
            compaction_threshold: 4, // Default threshold value
        };
        
        let mut recovered_tree = LSMTree::with_config(config);
        
        // Verify data from disk is recovered
        assert_eq!(recovered_tree.get(1), Some(100));
        assert_eq!(recovered_tree.get(2), Some(200));
        assert_eq!(recovered_tree.get(3), Some(300));
        
        // But data that wasn't flushed is lost
        assert_eq!(recovered_tree.get(4), None);
        assert_eq!(recovered_tree.get(5), None);
        
        // Add some data
        recovered_tree.put(6, 600).unwrap();
        
        // Explicit recovery - should clear memtable but keep disk data
        recovered_tree.recover().unwrap();
        
        // Data from disk is still there
        assert_eq!(recovered_tree.get(1), Some(100));
        assert_eq!(recovered_tree.get(2), Some(200));
        assert_eq!(recovered_tree.get(3), Some(300));
        
        // But unflushed data is lost
        assert_eq!(recovered_tree.get(6), None);
    }
}
