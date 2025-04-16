use crate::run::{Run, RunStorage};
use crate::types::{Key, Value};
use std::sync::RwLock;

#[derive(Clone)]
pub struct Level {
    runs: Vec<Run>,
    // This field is intentionally kept for future thread safety enhancements
    #[allow(dead_code)]
    lock: std::sync::Arc<()>,
}

impl Level {
    pub fn new() -> Self {
        Level { 
            runs: Vec::new(),
            lock: std::sync::Arc::new(()),
        }
    }

    // Add a new run to this level
    pub fn add_run(&mut self, run: Run) {
        self.runs.push(run);
    }

    // Get the number of runs in this level
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }
    
    // Get a run by index
    pub fn get_run(&self, index: usize) -> Run {
        self.runs[index].clone()
    }
    
    // Get reference to all runs in this level
    pub fn get_runs(&self) -> &[Run] {
        &self.runs
    }
    
    // Remove a run by index
    pub fn remove_run(&mut self, index: usize) -> Run {
        self.runs.remove(index)
    }

    // Retrieve a value for a key by searching all runs (in-memory only)
    pub fn get(&self, key: Key) -> Option<Value> {
        for run in &self.runs {
            if let Some(value) = run.get(key) {
                return Some(value);
            }
        }
        None
    }
    
    // Retrieve a value for a key by searching all runs with storage support
    pub fn get_with_storage(&self, key: Key, storage: &dyn RunStorage) -> Option<Value> {
        for run in &self.runs {
            if let Some(value) = run.get_with_storage(key, storage) {
                return Some(value);
            }
        }
        None
    }

    // Retrieve all key-value pairs in the specified range (in-memory only)
    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        let mut results = Vec::new();
        for run in &self.runs {
            results.extend(run.range(start, end));
        }
        results
    }
    
    // Retrieve all key-value pairs in the specified range with storage support
    pub fn range_with_storage(&self, start: Key, end: Key, storage: &dyn RunStorage) -> Vec<(Key, Value)> {
        let mut results = Vec::new();
        for run in &self.runs {
            results.extend(run.range_with_storage(start, end, storage));
        }
        results
    }
}

/// Thread-safe level structure with fine-grained locking
pub struct ConcurrentLevel {
    inner: RwLock<Level>,
}

impl ConcurrentLevel {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Level::new()),
        }
    }

    // Add a new run to this level (requires write lock)
    pub fn add_run(&self, run: Run) {
        let mut level = self.inner.write().unwrap();
        level.add_run(run);
    }

    // Get the number of runs in this level (requires read lock)
    pub fn run_count(&self) -> usize {
        let level = self.inner.read().unwrap();
        level.run_count()
    }
    
    // Get a run by index (requires read lock)
    pub fn get_run(&self, index: usize) -> Run {
        let level = self.inner.read().unwrap();
        level.get_run(index)
    }
    
    // Get a clone of all runs in this level (requires read lock)
    pub fn get_runs(&self) -> Vec<Run> {
        let level = self.inner.read().unwrap();
        level.get_runs().to_vec()
    }
    
    // Remove a run by index (requires write lock)
    pub fn remove_run(&self, index: usize) -> Run {
        let mut level = self.inner.write().unwrap();
        level.remove_run(index)
    }

    // Retrieve a value for a key by searching all runs (requires read lock)
    pub fn get(&self, key: Key) -> Option<Value> {
        let level = self.inner.read().unwrap();
        level.get(key)
    }
    
    // Retrieve a value for a key by searching all runs with storage support (requires read lock)
    pub fn get_with_storage(&self, key: Key, storage: &dyn RunStorage) -> Option<Value> {
        let level = self.inner.read().unwrap();
        level.get_with_storage(key, storage)
    }

    // Retrieve all key-value pairs in the specified range (requires read lock)
    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        let level = self.inner.read().unwrap();
        level.range(start, end)
    }
    
    // Retrieve all key-value pairs in the specified range with storage support (requires read lock)
    pub fn range_with_storage(&self, start: Key, end: Key, storage: &dyn RunStorage) -> Vec<(Key, Value)> {
        let level = self.inner.read().unwrap();
        level.range_with_storage(start, end, storage)
    }
    
    // Replace the entire level with a new one (for compaction) (requires write lock)
    pub fn replace(&self, new_level: Level) -> Level {
        let mut level = self.inner.write().unwrap();
        std::mem::replace(&mut *level, new_level)
    }
    
    // Get a clone of the level for operations that need the whole level (requires read lock)
    pub fn clone_level(&self) -> Level {
        let level = self.inner.read().unwrap();
        level.clone()
    }
    
    // Clear all runs from the level (requires write lock)
    pub fn clear(&self) {
        let mut level = self.inner.write().unwrap();
        *level = Level::new();
    }
}

impl Default for ConcurrentLevel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::Run;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_level_operations() {
        let mut level = Level::new();
        let data1 = vec![(1, 100), (2, 200)];
        let data2 = vec![(3, 300), (4, 400)];

        // Add runs to the level
        level.add_run(Run::new(data1));
        level.add_run(Run::new(data2));

        // Test key lookups
        assert_eq!(level.get(2), Some(200));
        assert_eq!(level.get(4), Some(400));
        assert_eq!(level.get(5), None);

        // Test range queries
        let range = level.range(2, 4);
        assert_eq!(range, vec![(2, 200), (3, 300)]);
    }
    
    #[test]
    fn test_concurrent_level_operations() {
        let level = ConcurrentLevel::new();
        let data1 = vec![(1, 100), (2, 200)];
        let data2 = vec![(3, 300), (4, 400)];

        // Add runs to the level
        level.add_run(Run::new(data1));
        level.add_run(Run::new(data2));

        // Test key lookups
        assert_eq!(level.get(2), Some(200));
        assert_eq!(level.get(4), Some(400));
        assert_eq!(level.get(5), None);

        // Test range queries
        let range = level.range(2, 4);
        assert_eq!(range, vec![(2, 200), (3, 300)]);
    }
    
    #[test]
    fn test_concurrent_level_access() {
        let level = Arc::new(ConcurrentLevel::new());
        
        // Add initial data
        level.add_run(Run::new(vec![(1, 100), (2, 200)]));
        
        // Clone for threads
        let level1 = level.clone();
        let level2 = level.clone();
        
        // Thread 1: Read operations
        let t1 = thread::spawn(move || {
            for _ in 0..100 {
                assert_eq!(level1.get(1), Some(100));
                assert_eq!(level1.get(2), Some(200));
                let range = level1.range(1, 3);
                assert_eq!(range.len(), 2);
            }
        });
        
        // Thread 2: Write operations
        let t2 = thread::spawn(move || {
            level2.add_run(Run::new(vec![(3, 300), (4, 400)]));
        });
        
        t1.join().unwrap();
        t2.join().unwrap();
        
        // Verify final state
        assert_eq!(level.run_count(), 2);
        assert_eq!(level.get(3), Some(300));
        assert_eq!(level.get(4), Some(400));
    }
    
    #[test]
    fn test_level_replace() {
        let level = ConcurrentLevel::new();
        
        // Add initial data
        level.add_run(Run::new(vec![(1, 100), (2, 200)]));
        
        // Create a new level
        let mut new_level = Level::new();
        new_level.add_run(Run::new(vec![(3, 300), (4, 400)]));
        
        // Replace the level
        let old_level = level.replace(new_level);
        
        // Verify old level
        assert_eq!(old_level.get(1), Some(100));
        assert_eq!(old_level.get(2), Some(200));
        
        // Verify new level
        assert_eq!(level.get(1), None);
        assert_eq!(level.get(2), None);
        assert_eq!(level.get(3), Some(300));
        assert_eq!(level.get(4), Some(400));
    }
}