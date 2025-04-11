use crate::run::{Block, BlockCache, BlockCacheConfig, BlockKey, FencePointers, Result, Run};
use crate::types::{Key, StorageType};
use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;

/// Trait to enable downcasting trait objects
pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

/// Unique identifier for a stored run
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunId {
    pub level: usize,
    pub sequence: u64,
}

impl RunId {
    pub fn new(level: usize, sequence: u64) -> Self {
        Self { level, sequence }
    }

    /// Convert the RunId to a string for use in filenames
    pub fn to_string(&self) -> String {
        format!("L{:02}_R{:010}", self.level, self.sequence)
    }

    /// Parse a RunId from a string
    pub fn from_string(s: &str) -> Option<Self> {
        if s.len() < 15 || !s.starts_with('L') {
            return None;
        }

        let level_str = &s[1..3];
        let seq_str = &s[5..];

        let level = level_str.parse::<usize>().ok()?;
        let sequence = seq_str.parse::<u64>().ok()?;

        Some(Self { level, sequence })
    }
}

/// Statistics about the storage usage
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_size_bytes: u64,
    pub file_count: usize,
    pub runs_per_level: Vec<usize>,
    pub blocks_per_level: Vec<usize>,
    pub entries_per_level: Vec<usize>,
}

/// Metadata about a stored run
#[derive(Debug, Clone)]
pub struct RunMetadata {
    pub id: RunId,
    pub min_key: Key,
    pub max_key: Key,
    pub block_count: usize,
    pub entry_count: usize,
    pub total_size_bytes: u64,
    pub creation_timestamp: u64,
}

/// Options for run storage initialization
#[derive(Debug, Clone)]
pub struct StorageOptions {
    pub base_path: PathBuf,
    pub create_if_missing: bool,
    pub max_open_files: usize,
    pub sync_writes: bool,
}

impl Default for StorageOptions {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("./data"),
            create_if_missing: true,
            max_open_files: 1000,
            sync_writes: true,
        }
    }
}

/// Generic trait for run storage implementations
pub trait RunStorage: Send + Sync + AsAny {
    /// Store a run and return its unique identifier
    fn store_run(&self, level: usize, run: &Run) -> Result<RunId>;

    /// Load a run from storage
    fn load_run(&self, run_id: RunId) -> Result<Run>;

    /// Delete a run from storage
    fn delete_run(&self, run_id: RunId) -> Result<()>;

    /// Get metadata for a run
    fn get_run_metadata(&self, run_id: RunId) -> Result<RunMetadata>;

    /// List all runs in a level
    fn list_runs(&self, level: usize) -> Result<Vec<RunId>>;

    /// Get storage statistics
    fn get_stats(&self) -> Result<StorageStats>;

    /// Check if a run exists
    fn run_exists(&self, run_id: RunId) -> Result<bool>;
    
    /// Load a specific block from a run
    fn load_block(&self, run_id: RunId, block_idx: usize) -> Result<Block> {
        // Default implementation: load the entire run and extract the block
        let run = self.load_run(run_id)?;
        
        if block_idx >= run.blocks.len() {
            return Err(super::Error::Block(format!(
                "Block index {} out of bounds for run {:?} with {} blocks",
                block_idx, run_id, run.blocks.len()
            )));
        }
        
        Ok(run.blocks[block_idx].clone())
    }
    
    /// Get the number of blocks in a run
    fn get_block_count(&self, run_id: RunId) -> Result<usize> {
        // Default implementation: get from metadata
        let metadata = self.get_run_metadata(run_id)?;
        Ok(metadata.block_count)
    }
    
    /// Load run fence pointers only (without loading all blocks)
    fn load_fence_pointers(&self, run_id: RunId) -> Result<FencePointers> {
        // Default implementation: load the entire run and extract fence pointers
        let run = self.load_run(run_id)?;
        Ok(run.fence_pointers.clone())
    }
}

/// Factory for creating RunStorage implementations
pub struct StorageFactory;

impl StorageFactory {
    /// Create a new storage instance with the given type
    pub fn create_from_type(storage_type: StorageType, options: StorageOptions) -> Result<Arc<dyn RunStorage>> {
        match storage_type {
            StorageType::File => Ok(Arc::new(FileStorage::new(options)?)),
            StorageType::LSF => Ok(Arc::new(super::LSFStorage::new(options)?)),
            StorageType::MMap => Err(super::Error::Storage(
                "MMap storage not yet implemented".to_string()
            )),
        }
    }
    
    /// Create a new storage instance from a string type (for backward compatibility)
    pub fn create(storage_type: &str, options: StorageOptions) -> Result<Arc<dyn RunStorage>> {
        if let Some(storage_type) = StorageType::from_str(storage_type) {
            Self::create_from_type(storage_type, options)
        } else {
            Err(super::Error::Storage(format!(
                "Unknown storage type: {}",
                storage_type
            )))
        }
    }
}

/// File-based storage implementation with block caching
#[derive(Debug)]
pub struct FileStorage {
    base_path: PathBuf,
    options: StorageOptions,
    block_cache: Arc<BlockCache>,
}

impl AsAny for FileStorage {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl FileStorage {
    pub fn new(options: StorageOptions) -> Result<Self> {
        // Create a minimal block cache (disabled for tests)
        let cache_config = BlockCacheConfig {
            max_capacity: 10, // Small cache for testing
            ..Default::default()
        };
        let block_cache = Arc::new(BlockCache::new(cache_config));
        
        let storage = Self {
            base_path: options.base_path.clone(),
            options,
            block_cache,
        };

        // Ensure the base directory exists
        storage.init_directories()?;

        Ok(storage)
    }
    
    /// Create a new storage with custom cache configuration
    pub fn with_cache_config(options: StorageOptions, cache_config: BlockCacheConfig) -> Result<Self> {
        let block_cache = Arc::new(BlockCache::new(cache_config));
        
        let storage = Self {
            base_path: options.base_path.clone(),
            options,
            block_cache,
        };

        // Ensure the base directory exists
        storage.init_directories()?;

        Ok(storage)
    }
    
    /// Get a reference to the block cache
    pub fn get_cache(&self) -> &Arc<BlockCache> {
        &self.block_cache
    }

    fn init_directories(&self) -> Result<()> {
        // Create base directory if it doesn't exist
        if !self.base_path.exists() {
            if self.options.create_if_missing {
                std::fs::create_dir_all(&self.base_path)?;
            } else {
                return Err(super::Error::Storage(format!(
                    "Base directory does not exist: {:?}",
                    self.base_path
                )));
            }
        }

        // Create runs directory
        let runs_dir = self.base_path.join("runs");
        if !runs_dir.exists() {
            std::fs::create_dir_all(&runs_dir)?;
        }

        Ok(())
    }

    fn get_level_dir(&self, level: usize) -> PathBuf {
        let level_dir = self.base_path.join("runs").join(format!("level_{}", level));
        if !level_dir.exists() {
            std::fs::create_dir_all(&level_dir).expect("Failed to create level directory");
        }
        level_dir
    }

    fn get_run_data_path(&self, run_id: RunId) -> PathBuf {
        self.get_level_dir(run_id.level)
            .join(format!("{}.bin", run_id.to_string()))
    }

    fn get_run_meta_path(&self, run_id: RunId) -> PathBuf {
        self.get_level_dir(run_id.level)
            .join(format!("{}.meta", run_id.to_string()))
    }

    fn get_next_sequence(&self, level: usize) -> Result<u64> {
        let level_dir = self.get_level_dir(level);

        // Read all entries and find the highest sequence number
        let mut max_seq = 0;

        if level_dir.exists() {
            for entry in std::fs::read_dir(&level_dir)? {
                let entry = entry?;
                let filename = entry.file_name().to_string_lossy().to_string();

                // Only consider .bin files
                if !filename.ends_with(".bin") {
                    continue;
                }

                // Extract the base name without extension
                let base_name = filename.trim_end_matches(".bin");
                if let Some(run_id) = RunId::from_string(base_name) {
                    if run_id.level == level && run_id.sequence > max_seq {
                        max_seq = run_id.sequence;
                    }
                }
            }
        }

        // Return the next sequence number
        Ok(max_seq + 1)
    }
}

impl RunStorage for FileStorage {
    fn store_run(&self, level: usize, run: &Run) -> Result<RunId> {
        // Get the next sequence number for this level
        let sequence = self.get_next_sequence(level)?;
        let run_id = RunId::new(level, sequence);

        // Create paths for data and metadata files
        let data_path = self.get_run_data_path(run_id);
        let meta_path = self.get_run_meta_path(run_id);

        // Clone the run so we can mutate it for serialization
        let mut run_clone = Run {
            data: run.data.clone(),
            block_config: run.block_config.clone(),
            blocks: run.blocks.clone(),
            filter: run.filter.box_clone(),
            compression: run.compression.clone_box(),
            fence_pointers: FencePointers::new(), // Will be rebuilt during serialization
            id: Some(run_id),
            level: run.level, // Preserve level information
        };
        
        // Ensure deterministic serialization by resealing blocks
        for block in &mut run_clone.blocks {
            // Force recomputation of block checksums to ensure determinism
            block.is_sealed = false;
            block.seal()?;
        };

        // Serialize the run
        let run_data = run_clone.serialize()?;

        // Write run data to file
        std::fs::write(&data_path, &run_data)?;

        // Create metadata from the run
        let metadata = RunMetadata {
            id: run_id,
            min_key: run.min_key().unwrap_or(Key::MIN),
            max_key: run.max_key().unwrap_or(Key::MAX),
            block_count: run.blocks.len(),
            entry_count: run.entry_count(),
            total_size_bytes: run_data.len() as u64,
            creation_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Serialize metadata to JSON
        let meta_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| super::Error::Serialization(e.to_string()))?;

        // Write metadata file
        std::fs::write(&meta_path, meta_json)?;

        // Sync if required
        if self.options.sync_writes {
            if let Ok(f) = std::fs::File::open(&data_path) {
                f.sync_all()?;
            }

            if let Ok(f) = std::fs::File::open(&meta_path) {
                f.sync_all()?;
            }
        }

        Ok(run_id)
    }

    fn load_run(&self, run_id: RunId) -> Result<Run> {
        // Check if run exists
        if !self.run_exists(run_id)? {
            return Err(super::Error::Storage(format!(
                "Run not found: {:?}",
                run_id
            )));
        }

        // Read data file
        let data_path = self.get_run_data_path(run_id);
        let run_data = std::fs::read(&data_path)?;

        // Debug output for loaded data - only when RUST_LOG=debug
        if std::env::var("RUST_LOG").map(|v| v == "debug").unwrap_or(false) {
            println!("FileStorage load - Run: {}, Data size: {}", run_id, run_data.len());
        }
        
        // Deserialize the run
        let mut run = Run::deserialize(&run_data)?;
        
        // Verify contents of the deserialized run - only when RUST_LOG=debug
        if std::env::var("RUST_LOG").map(|v| v == "debug").unwrap_or(false) {
            println!("FileStorage load - Blocks: {}, Data items: {}", 
                    run.blocks.len(), run.data.len());
            if !run.data.is_empty() {
                println!("First data item: ({}, {})", run.data[0].0, run.data[0].1);
            }
        }

        // Set the run ID and level
        run.id = Some(run_id);
        run.level = Some(run_id.level);

        Ok(run)
    }

    fn delete_run(&self, run_id: RunId) -> Result<()> {
        let data_path = self.get_run_data_path(run_id);
        let meta_path = self.get_run_meta_path(run_id);

        // Delete files if they exist
        if data_path.exists() {
            std::fs::remove_file(&data_path)?;
        }

        if meta_path.exists() {
            std::fs::remove_file(&meta_path)?;
        }

        Ok(())
    }

    fn get_run_metadata(&self, run_id: RunId) -> Result<RunMetadata> {
        let meta_path = self.get_run_meta_path(run_id);

        if !meta_path.exists() {
            return Err(super::Error::Storage(format!(
                "Metadata not found for run: {:?}",
                run_id
            )));
        }

        let meta_json = std::fs::read_to_string(&meta_path)?;
        let metadata = serde_json::from_str(&meta_json)
            .map_err(|e| super::Error::Serialization(e.to_string()))?;

        Ok(metadata)
    }

    fn list_runs(&self, level: usize) -> Result<Vec<RunId>> {
        let level_dir = self.get_level_dir(level);
        let mut run_ids = Vec::new();

        if !level_dir.exists() {
            return Ok(run_ids);
        }

        for entry in std::fs::read_dir(&level_dir)? {
            let entry = entry?;
            let filename = entry.file_name().to_string_lossy().to_string();

            // Only consider .bin files
            if !filename.ends_with(".bin") {
                continue;
            }

            // Extract the base name without extension
            let base_name = filename.trim_end_matches(".bin");
            if let Some(run_id) = RunId::from_string(base_name) {
                if run_id.level == level {
                    run_ids.push(run_id);
                }
            }
        }

        // Sort by sequence
        run_ids.sort_by_key(|id| id.sequence);

        Ok(run_ids)
    }

    fn get_stats(&self) -> Result<StorageStats> {
        let mut stats = StorageStats {
            total_size_bytes: 0,
            file_count: 0,
            runs_per_level: Vec::new(),
            blocks_per_level: Vec::new(),
            entries_per_level: Vec::new(),
        };

        // Find the highest level
        let runs_dir = self.base_path.join("runs");
        if !runs_dir.exists() {
            return Ok(stats);
        }

        // Find all level directories
        let mut max_level = 0;
        for entry in std::fs::read_dir(&runs_dir)? {
            let entry = entry?;
            let dirname = entry.file_name().to_string_lossy().to_string();

            if !dirname.starts_with("level_") {
                continue;
            }

            if let Some(level_str) = dirname.strip_prefix("level_") {
                if let Ok(level) = level_str.parse::<usize>() {
                    max_level = std::cmp::max(max_level, level);
                }
            }
        }

        // Resize vectors to account for all levels
        stats.runs_per_level.resize(max_level + 1, 0);
        stats.blocks_per_level.resize(max_level + 1, 0);
        stats.entries_per_level.resize(max_level + 1, 0);

        // Walk through each level
        for level in 0..=max_level {
            let run_ids = self.list_runs(level)?;
            stats.runs_per_level[level] = run_ids.len();

            for run_id in run_ids {
                let data_path = self.get_run_data_path(run_id);
                let meta_path = self.get_run_meta_path(run_id);

                // Count files
                stats.file_count += 2; // .bin and .meta files

                // Add file sizes
                if data_path.exists() {
                    stats.total_size_bytes += std::fs::metadata(&data_path)?.len();
                }

                if meta_path.exists() {
                    stats.total_size_bytes += std::fs::metadata(&meta_path)?.len();
                }

                // Get metadata for block and entry counts
                if let Ok(metadata) = self.get_run_metadata(run_id) {
                    stats.blocks_per_level[level] += metadata.block_count;
                    stats.entries_per_level[level] += metadata.entry_count;
                }
            }
        }

        Ok(stats)
    }

    fn run_exists(&self, run_id: RunId) -> Result<bool> {
        let data_path = self.get_run_data_path(run_id);
        let meta_path = self.get_run_meta_path(run_id);

        Ok(data_path.exists() && meta_path.exists())
    }
    
    // Override the default implementation for efficient block loading with caching
    fn load_block(&self, run_id: RunId, block_idx: usize) -> Result<Block> {
        // Create a block cache key
        let cache_key = BlockKey {
            run_id,
            block_idx,
        };
        
        // Try to get from cache first
        if let Some(cached_block) = self.block_cache.get(&cache_key) {
            // Return a clone of the cached block
            return Ok((*cached_block).clone());
        }
        
        // Check if the run exists
        if !self.run_exists(run_id)? {
            return Err(super::Error::Storage(format!(
                "Run not found: {:?}",
                run_id
            )));
        }
        
        // Get metadata to check block count
        let metadata = self.get_run_metadata(run_id)?;
        if block_idx >= metadata.block_count {
            return Err(super::Error::Block(format!(
                "Block index {} out of bounds for run {:?} with {} blocks",
                block_idx, run_id, metadata.block_count
            )));
        }
        
        // Load the run and extract the block
        // TODO: Optimize to load only the specific block
        let run = self.load_run(run_id)?;
        
        if block_idx >= run.blocks.len() {
            return Err(super::Error::Block(format!(
                "Block index {} out of bounds for loaded run {:?} with {} blocks",
                block_idx, run_id, run.blocks.len()
            )));
        }
        
        let block = run.blocks[block_idx].clone();
        
        // Cache the block for future use
        self.block_cache.insert(cache_key, block.clone())?;
        
        Ok(block)
    }
    
    // Override the default implementation for efficient fence pointer loading
    fn load_fence_pointers(&self, run_id: RunId) -> Result<FencePointers> {
        // Check if the run exists
        if !self.run_exists(run_id)? {
            return Err(super::Error::Storage(format!(
                "Run not found: {:?}",
                run_id
            )));
        }
        
        // TODO: Optimize to load only the fence pointers portion of the file
        // For now, we load the entire run and extract fence pointers
        let run = self.load_run(run_id)?;
        Ok(run.fence_pointers.clone())
    }
}

// Need to manually implement serialization for RunMetadata
// until we add serde to the project
mod json_impl {
    use super::*;
    use std::fmt;

    impl serde::Serialize for RunMetadata {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            use serde::ser::SerializeStruct;

            let mut s = serializer.serialize_struct("RunMetadata", 7)?;
            s.serialize_field("id_level", &self.id.level)?;
            s.serialize_field("id_sequence", &self.id.sequence)?;
            s.serialize_field("min_key", &self.min_key)?;
            s.serialize_field("max_key", &self.max_key)?;
            s.serialize_field("block_count", &self.block_count)?;
            s.serialize_field("entry_count", &self.entry_count)?;
            s.serialize_field("total_size_bytes", &self.total_size_bytes)?;
            s.serialize_field("creation_timestamp", &self.creation_timestamp)?;
            s.end()
        }
    }

    impl<'de> serde::Deserialize<'de> for RunMetadata {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            #[derive(serde::Deserialize)]
            struct Helper {
                id_level: usize,
                id_sequence: u64,
                min_key: Key,
                max_key: Key,
                block_count: usize,
                entry_count: usize,
                total_size_bytes: u64,
                creation_timestamp: u64,
            }

            let helper = Helper::deserialize(deserializer)?;

            Ok(RunMetadata {
                id: RunId::new(helper.id_level, helper.id_sequence),
                min_key: helper.min_key,
                max_key: helper.max_key,
                block_count: helper.block_count,
                entry_count: helper.entry_count,
                total_size_bytes: helper.total_size_bytes,
                creation_timestamp: helper.creation_timestamp,
            })
        }
    }

    // Custom Debug implementation for pretty printing
    impl fmt::Display for RunId {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "L{:02}_R{:010}", self.level, self.sequence)
        }
    }

    impl fmt::Display for RunMetadata {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(f, "Run: {}", self.id)?;
            writeln!(f, "  Key Range: {} to {}", self.min_key, self.max_key)?;
            writeln!(f, "  Blocks: {}", self.block_count)?;
            writeln!(f, "  Entries: {}", self.entry_count)?;
            writeln!(f, "  Size: {} bytes", self.total_size_bytes)?;
            writeln!(
                f,
                "  Created: {}",
                chrono::DateTime::from_timestamp(self.creation_timestamp as i64, 0)
                    .map(|dt| dt.to_rfc3339())
                    .unwrap_or_else(|| self.creation_timestamp.to_string())
            )?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_run_id_serialization() {
        let run_id = RunId::new(5, 123456789);
        let s = run_id.to_string();
        assert_eq!(s, "L05_R0123456789");

        let parsed = RunId::from_string(&s).unwrap();
        assert_eq!(parsed.level, 5);
        assert_eq!(parsed.sequence, 123456789);
    }

    #[test]
    fn test_run_id_from_invalid_string() {
        assert!(RunId::from_string("invalid").is_none());
        assert!(RunId::from_string("LXX_R0123456789").is_none());
        assert!(RunId::from_string("L05_RINVALID").is_none());
    }

    #[test]
    fn test_storage_creation() {
        let temp_dir = tempdir().unwrap();
        let options = StorageOptions {
            base_path: temp_dir.path().to_path_buf(),
            create_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
        };

        let _storage = FileStorage::new(options).unwrap();

        // Check that directories are created
        assert!(temp_dir.path().join("runs").exists());
    }

    #[test]
    fn test_run_storage_basic_operations() {
        let temp_dir = tempdir().unwrap();
        let options = StorageOptions {
            base_path: temp_dir.path().to_path_buf(),
            create_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
        };

        let storage = FileStorage::new(options).unwrap();

        // Create a simple run
        let run = Run::new(vec![(1, 100), (2, 200), (3, 300)]);

        // Store the run
        let run_id = storage.store_run(0, &run).unwrap();
        assert_eq!(run_id.level, 0);
        assert_eq!(run_id.sequence, 1);

        // Check that the run exists
        assert!(storage.run_exists(run_id).unwrap());

        // Check that we can get metadata
        let metadata = storage.get_run_metadata(run_id).unwrap();
        assert_eq!(metadata.id, run_id);
        assert_eq!(metadata.entry_count, 3);

        // Get stats
        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.runs_per_level[0], 1);
        assert!(stats.total_size_bytes > 0);

        // List runs
        let runs = storage.list_runs(0).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0], run_id);

        // Delete the run
        storage.delete_run(run_id).unwrap();

        // Verify it's gone
        assert!(!storage.run_exists(run_id).unwrap());
        let runs_after_delete = storage.list_runs(0).unwrap();
        assert_eq!(runs_after_delete.len(), 0);
    }
}
