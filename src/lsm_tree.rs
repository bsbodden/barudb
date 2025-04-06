use crate::level::Level;
use crate::memtable::Memtable;
use crate::run::{Run, RunStorage, StorageFactory, StorageOptions};
use crate::types::{Key, Result, Value, TOMBSTONE};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Configuration for LSM tree
#[derive(Debug, Clone)]
pub struct LSMConfig {
    /// Size of the memory buffer in pages
    pub buffer_size: usize,
    
    /// Storage backend to use (file, lsf, mmap)
    pub storage_type: String,
    
    /// Base path for storage
    pub storage_path: PathBuf,
    
    /// Whether to create storage path if missing
    pub create_path_if_missing: bool,
    
    /// Maximum open files for storage
    pub max_open_files: usize,
    
    /// Whether to sync writes to disk
    pub sync_writes: bool,
}

impl Default for LSMConfig {
    fn default() -> Self {
        Self {
            buffer_size: 128,
            storage_type: "file".to_string(),
            storage_path: PathBuf::from("./data"),
            create_path_if_missing: true,
            max_open_files: 1000,
            sync_writes: true,
        }
    }
}

pub struct LSMTree {
    buffer: Arc<RwLock<Memtable>>,
    levels: Vec<Level>,
    storage: Arc<dyn RunStorage>,
    config: LSMConfig,
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
        let storage = StorageFactory::create(&config.storage_type, storage_options)
            .expect("Failed to create storage backend");
        
        let mut tree = Self {
            buffer: Arc::new(RwLock::new(Memtable::new(config.buffer_size))),
            levels: Vec::new(),
            storage,
            config,
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

        // Create a new run
        let mut run = Run::new(data);
        
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_tree() -> (LSMTree, tempfile::TempDir) {
        let temp_dir = tempdir().unwrap();
        let config = LSMConfig {
            buffer_size: 128,
            storage_type: "file".to_string(),
            storage_path: temp_dir.path().to_path_buf(),
            create_path_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
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
        // Test default configuration
        let lsm_tree = LSMTree::new(256);
        assert_eq!(lsm_tree.get_config().buffer_size, 256);
        assert_eq!(lsm_tree.get_config().storage_type, "file");
        assert!(lsm_tree.get_config().create_path_if_missing);
        
        // Test custom configuration
        let temp_dir = tempdir().unwrap();
        let config = LSMConfig {
            buffer_size: 512,
            storage_type: "file".to_string(),
            storage_path: temp_dir.path().to_path_buf(),
            create_path_if_missing: false,
            max_open_files: 50,
            sync_writes: true,
        };
        let lsm_tree = LSMTree::with_config(config.clone());
        assert_eq!(lsm_tree.get_config().buffer_size, 512);
        assert_eq!(lsm_tree.get_config().max_open_files, 50);
        assert_eq!(lsm_tree.get_config().storage_path, temp_dir.path().to_path_buf());
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
            storage_type: "file".to_string(),
            storage_path: storage_path.clone(),
            create_path_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
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
            storage_type: "file".to_string(),
            storage_path: storage_path,
            create_path_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
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
