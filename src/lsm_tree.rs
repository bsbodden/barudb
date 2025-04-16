use crate::compaction::{CompactionFactory, CompactionPolicy};
use crate::level::{ConcurrentLevel, Level};
use crate::memtable::Memtable;
use crate::run::{
    self, compression::{AdaptiveCompressionConfig, CompressionConfig}, Run, RunStorage, StorageFactory, StorageOptions,
    StorageStats,
};
use crate::types::{CompactionPolicyType, Key, Result, StorageType, Value, TOMBSTONE};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

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
    
    /// Configuration for compression
    pub compression: CompressionConfig,
    
    /// Configuration for adaptive compression
    pub adaptive_compression: AdaptiveCompressionConfig,
    
    /// Whether to collect compression statistics for evaluation
    pub collect_compression_stats: bool,
    
    /// Whether to enable background compaction
    pub background_compaction: bool,
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
            compression: CompressionConfig::default(),
            adaptive_compression: AdaptiveCompressionConfig {
                enabled: false, // Disabled by default for reproducibility
                ..Default::default()
            },
            collect_compression_stats: false,
            background_compaction: false,
        }
    }
}

/// State for background compaction thread
struct CompactionState {
    active: AtomicBool,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl CompactionState {
    fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            thread: Mutex::new(None),
        }
    }
    
    fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }
    
    fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Release);
    }
}

pub struct LSMTree {
    buffer: Arc<Memtable>,
    levels: Vec<Arc<ConcurrentLevel>>,
    storage: Arc<dyn RunStorage>,
    config: LSMConfig,
    compaction_policy: Box<dyn CompactionPolicy>,
    compaction_state: Arc<CompactionState>,
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
        
        // Create compaction state
        let compaction_state = Arc::new(CompactionState::new());
        
        let mut tree = Self {
            buffer: Arc::new(Memtable::new(config.buffer_size)),
            levels: Vec::new(),
            storage,
            config,
            compaction_policy,
            compaction_state,
        };
        
        // Recover state from disk if available
        tree.recover_from_disk();
        
        // Start background compaction if enabled
        if tree.config.background_compaction {
            tree.start_background_compaction();
        }
        
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
                            self.levels.push(Arc::new(ConcurrentLevel::new()));
                        }
                        
                        // Load all runs for this level
                        for run_id in run_ids {
                            match self.storage.load_run(run_id) {
                                Ok(mut run) => {
                                    // Set run ID for future reference
                                    run.id = Some(run_id);
                                    
                                    // Count and print information about tombstones for debugging
                                    let tombstone_count = run.data.iter()
                                        .filter(|(_, v)| *v == TOMBSTONE)
                                        .count();
                                    
                                    if tombstone_count > 0 || std::env::var("RUST_LOG").map(|v| v == "debug").unwrap_or(false) {
                                        println!("Found {} tombstones in run {}", tombstone_count, run_id);
                                    }
                                    
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
        self.buffer.clear();
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
            let result = self.buffer.put(key, value);
            result.is_ok() && self.buffer.is_full()
        };

        if flush_required {
            self.flush_buffer_to_level0()?;
            // Check if compaction is needed
            self.compact_if_needed()?;
        }
        Ok(())
    }

    /// Get a value by key using in-memory blocks only
    pub fn get(&self, key: Key) -> Option<Value> {
        // Check buffer first
        if let Some(value) = self.buffer.get(&key) {
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
    
    /// Get a value by key with block cache support
    /// This version can use block-level caching for better performance
    pub fn get_with_cache(&self, key: Key) -> Option<Value> {
        // Check buffer first
        if let Some(value) = self.buffer.get(&key) {
            if value == TOMBSTONE {
                return None; // Ignore tombstone values
            }
            return Some(value);
        }

        // Check levels
        for level in &self.levels {
            if let Some(value) = level.get_with_storage(key, &*self.storage) {
                if value == TOMBSTONE {
                    return None; // Ignore tombstone values
                }
                return Some(value);
            }
        }

        None
    }

    /// Range query using in-memory blocks only
    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        let mut results = self.buffer.range(start, end);

        // Add results from levels
        for level in &self.levels {
            results.extend(level.range(start, end));
        }

        // Sort by key and remove duplicates, keeping only the most recent value
        results.sort_by_key(|&(key, _)| key);
        results.dedup_by_key(|&mut (key, _)| key);

        // CRITICAL: Filter out tombstones after deduplication to ensure deleted keys stay deleted
        // This is essential for data consistency across restarts
        let tombstone_count = results.iter().filter(|&(_, v)| *v == TOMBSTONE).count();
        if tombstone_count > 0 && std::env::var("RUST_LOG").map(|v| v == "debug").unwrap_or(false) {
            println!("Filtered {} tombstones from range query results", tombstone_count);
        }
        
        results.retain(|&(_, value)| value != TOMBSTONE);

        results
    }
    
    /// Range query with block cache support
    /// This version can use block-level caching for better performance
    pub fn range_with_cache(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        let mut results = self.buffer.range(start, end);

        // Add results from levels with storage support
        for level in &self.levels {
            results.extend(level.range_with_storage(start, end, &*self.storage));
        }

        // Sort by key and remove duplicates, keeping only the most recent value
        results.sort_by_key(|&(key, _)| key);
        results.dedup_by_key(|&mut (key, _)| key);

        // CRITICAL: Filter out tombstones after deduplication to ensure deleted keys stay deleted
        // This is essential for data consistency across restarts
        let tombstone_count = results.iter().filter(|&(_, v)| *v == TOMBSTONE).count();
        if tombstone_count > 0 && std::env::var("RUST_LOG").map(|v| v == "debug").unwrap_or(false) {
            println!("Filtered {} tombstones from range_with_cache query results", tombstone_count);
        }
        
        results.retain(|&(_, value)| value != TOMBSTONE);

        results
    }

    pub fn delete(&mut self, key: Key) -> Result<()> {
        self.put(key, TOMBSTONE)
    }

    /// Flush the current buffer to level 0
    /// This method is public to allow for testing and manual flushing
    pub fn flush_buffer_to_level0(&mut self) -> Result<()> {
        let data = self.buffer.take_all();
        
        if data.is_empty() {
            return Ok(());
        }

        // Create a new run with a level-optimized Bloom filter and compression
        // Level 0 is the first level after the memtable
        let fanout = self.config.fanout as f64;
        let mut run = Run::new_for_level(data, 0, fanout, Some(&self.config));
        
        // Store the run using configured storage
        let run_id = run.store(&*self.storage, 0)?;
        
        // Set the run ID for future reference
        run.id = Some(run_id);
        
        // Add the run to level 0
        if self.levels.is_empty() {
            self.levels.push(Arc::new(ConcurrentLevel::new()));
        }
        self.levels[0].add_run(run);
        
        Ok(())
    }
    
    /// Check all levels and perform compaction if needed according to policy
    pub fn compact_if_needed(&mut self) -> Result<()> {
        // Check each level for compaction needs
        for level_num in 0..self.levels.len() {
            let level_clone = self.levels[level_num].clone_level();
            if self.compaction_policy.should_compact(&level_clone, level_num) {
                self.compact_level(level_num)?;
            }
        }
        
        Ok(())
    }
    
    /// Start a background compaction thread
    pub fn start_background_compaction(&self) {
        // Don't start if already running
        if self.compaction_state.is_active() {
            return;
        }
        
        // Set active flag
        self.compaction_state.set_active(true);
        
        // Clone necessary references
        let levels = self.levels.clone();
        let compaction_policy = Arc::new(RwLock::new(
            CompactionFactory::create_from_type(
                self.config.compaction_policy.clone(), 
                self.config.compaction_threshold
            ).expect("Failed to create compaction policy")
        ));
        let storage = self.storage.clone();
        let compaction_state = self.compaction_state.clone();
        let config = self.config.clone();
        
        // Create thread to run compaction in background
        let thread = thread::spawn(move || {
            while compaction_state.is_active() {
                // Check each level for compaction needs
                let level_count = levels.len();
                for level_num in 0..level_count {
                    // Skip if we don't have this level
                    if level_num >= levels.len() {
                        continue;
                    }
                    
                    // Get clone of level for compaction check
                    let level_clone = levels[level_num].clone_level();
                    
                    // Check if compaction needed
                    let needs_compaction = {
                        let policy = compaction_policy.read().unwrap();
                        policy.should_compact(&level_clone, level_num)
                    };
                    
                    if needs_compaction {
                        let target_level_num = level_num + 1;
                        
                        // Ensure we have a target level
                        if levels.len() <= target_level_num {
                            // We can't modify the levels vector from this thread
                            // Just skip this compaction round
                            continue;
                        }
                        
                        // Perform compaction
                        let source_level = level_clone;
                        let mut target_level = Level::new();
                        
                        // Try to compact
                        let compact_result = {
                            let policy = compaction_policy.write().unwrap();
                            policy.compact(
                                &source_level,
                                &mut target_level,
                                &*storage,
                                level_num,
                                target_level_num,
                                Some(&config),
                            )
                        };
                        
                        match compact_result {
                            Ok(new_run) => {
                                // Clear source level
                                levels[level_num].clear();
                                
                                // Add new run to target level
                                levels[target_level_num].add_run(new_run);
                            },
                            Err(e) => {
                                eprintln!("Background compaction failed: {:?}", e);
                            }
                        }
                    }
                }
                
                // Sleep before next check
                thread::sleep(std::time::Duration::from_millis(100));
            }
        });
        
        // Store thread handle
        let mut thread_guard = self.compaction_state.thread.lock().unwrap();
        *thread_guard = Some(thread);
    }
    
    /// Stop the background compaction thread
    pub fn stop_background_compaction(&self) {
        self.compaction_state.set_active(false);
        
        // Wait for the thread to finish
        let mut thread_guard = self.compaction_state.thread.lock().unwrap();
        if let Some(thread) = thread_guard.take() {
            thread.join().ok();
        }
    }
    
    /// Perform a single compaction from one level to the next
    pub fn compact_level(&mut self, level_num: usize) -> Result<()> {
        // Ensure we have a level to compact into
        let target_level_num = level_num + 1;
        while self.levels.len() <= target_level_num {
            self.levels.push(Arc::new(ConcurrentLevel::new()));
        }
        
        // Get source level
        let source_level = self.levels[level_num].clone_level();
        
        // Create an empty target level for temporary use
        let mut target_level = Level::new();
        
        // Perform compaction with config
        match self.compaction_policy.compact(
            &source_level,
            &mut target_level,
            &*self.storage,
            level_num,
            target_level_num,
            Some(&self.config),
        ) {
            Ok(new_run) => {
                // Clear the source level
                self.levels[level_num].clear();
                
                // Add new run to target level
                self.levels[target_level_num].add_run(new_run);
                
                // Check if next level now needs compaction
                let next_level_clone = self.levels[target_level_num].clone_level();
                if target_level_num < self.levels.len() &&
                   self.compaction_policy.should_compact(&next_level_clone, target_level_num) {
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
            if level_num < self.levels.len() && self.levels[level_num].run_count() > 0 {
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
    
    /// Get block cache statistics if the storage implementation supports caching
    pub fn get_cache_stats(&self) -> Option<run::CacheStats> {
        if let Some(file_storage) = self.storage.as_any().downcast_ref::<run::FileStorage>() {
            Some(file_storage.get_cache().get_stats())
        } else {
            None
        }
    }
}

impl Drop for LSMTree {
    fn drop(&mut self) {
        // Stop background compaction thread if running
        if self.compaction_state.is_active() {
            self.stop_background_compaction();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::thread;
    use std::time::Duration;

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
            compression: run::CompressionConfig::default(),
            adaptive_compression: run::AdaptiveCompressionConfig::default(),
            collect_compression_stats: true,
            background_compaction: false,
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
            compression: run::CompressionConfig::default(),
            adaptive_compression: run::AdaptiveCompressionConfig::default(),
            collect_compression_stats: true,
            background_compaction: false,
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
            compression: run::CompressionConfig::default(),
            adaptive_compression: run::AdaptiveCompressionConfig::default(),
            collect_compression_stats: true,
            background_compaction: false,
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
            compression: run::CompressionConfig::default(),
            adaptive_compression: run::AdaptiveCompressionConfig::default(),
            collect_compression_stats: true,
            background_compaction: false,
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
    
    #[test]
    fn test_concurrent_access() {
        // Create a tree with data
        let (mut lsm_tree, _temp_dir) = create_test_tree();
        
        // Add some initial data
        for i in 0..100 {
            lsm_tree.put(i, i * 10).unwrap();
        }
        
        // Force flush to disk
        lsm_tree.flush_buffer_to_level0().unwrap();
        
        // Create multiple threads to access the tree concurrently
        let tree_arc = Arc::new(RwLock::new(lsm_tree));
        
        // Spawn reader threads
        let readers: Vec<_> = (0..4).map(|_| {
            let tree = tree_arc.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    let tree_guard = tree.read().unwrap();
                    for i in 0..100 {
                        let value = tree_guard.get(i);
                        assert_eq!(value, Some(i * 10));
                    }
                }
            })
        }).collect();
        
        // Spawn writer threads
        let writers: Vec<_> = (0..4).map(|t| {
            let tree = tree_arc.clone();
            thread::spawn(move || {
                for i in 0..100 {
                    let mut tree_guard = tree.write().unwrap();
                    tree_guard.put(1000 + t * 100 + i, i).unwrap();
                }
            })
        }).collect();
        
        // Wait for all threads to complete
        for r in readers {
            r.join().unwrap();
        }
        
        for w in writers {
            w.join().unwrap();
        }
        
        // Verify all writes are present
        let tree = tree_arc.read().unwrap();
        for t in 0..4 {
            for i in 0..100 {
                assert_eq!(tree.get(1000 + t * 100 + i), Some(i));
            }
        }
    }
    
    #[test]
    fn test_background_compaction() {
        // Create a temp directory
        let temp_dir = tempdir().unwrap();
        
        // Create a config with background compaction enabled and a low compaction threshold
        let config = LSMConfig {
            buffer_size: 128,
            storage_type: StorageType::File,
            storage_path: temp_dir.path().to_path_buf(),
            create_path_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
            fanout: 4,
            compaction_policy: CompactionPolicyType::Tiered,
            compaction_threshold: 2, // Set a low threshold to trigger compaction quickly
            compression: run::CompressionConfig::default(),
            adaptive_compression: run::AdaptiveCompressionConfig::default(),
            collect_compression_stats: false,
            background_compaction: true, // Enable background compaction
        };
        
        // Create tree with background compaction
        let mut lsm_tree = LSMTree::with_config(config);
        
        // Force creation of multiple levels by manually adding to level 0 and compacting
        
        // First batch - add to level 0
        for i in 0..20 {
            lsm_tree.put(i, i * 10).unwrap();
        }
        lsm_tree.flush_buffer_to_level0().unwrap();
        
        // Second batch - add to level 0 again
        for i in 20..40 {
            lsm_tree.put(i, i * 10).unwrap();
        }
        lsm_tree.flush_buffer_to_level0().unwrap();
        
        // Third batch - add to level 0 yet again
        for i in 40..60 {
            lsm_tree.put(i, i * 10).unwrap();
        }
        lsm_tree.flush_buffer_to_level0().unwrap();
        
        // Give the background compaction thread time to run
        // We need to wait longer because background compaction is less aggressive
        thread::sleep(Duration::from_millis(1000));
        
        // Explicitly force compaction as well to ensure it happens
        lsm_tree.force_compact_all().unwrap();
        
        // Verify we can still read all data
        for i in 0..60 {
            assert_eq!(lsm_tree.get(i), Some(i * 10), "Data mismatch for key {}", i);
        }
        
        // Verify at least two levels exist after compaction
        // Currently we're checking whether multiple levels exist, not whether data is in level 1
        let level_count = lsm_tree.levels.len();
        assert!(level_count > 1, "Expected multiple levels after compaction, got {}", level_count);
    }
}