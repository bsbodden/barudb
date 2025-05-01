use crate::run::{Block, lock_free_block_cache::BlockKey};
use crate::run::lock_free_cache_policies::LockFreeCachePolicy;
use crossbeam_skiplist::SkipMap;
use std::sync::Arc;
use std::time::SystemTime;
use std::any::Any;

/// Cache entry for LRU policy
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached block
    block: Arc<Block>,
    /// When this entry was last accessed
    last_accessed: SystemTime,
    /// Number of times this block has been accessed
    access_count: u64,
}

/// Lock-free LRU cache policy implementation
#[derive(Debug)]
pub struct LockFreeLRUPolicy {
    /// Maximum entries in the cache
    capacity: usize,
    /// Block cache entries - uses lock-free SkipMap from crossbeam
    entries: SkipMap<BlockKey, CacheEntry>,
}

impl LockFreeLRUPolicy {
    /// Create a new lock-free LRU policy with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: SkipMap::new(),
        }
    }
    
    /// Find the least recently used entry (for eviction)
    fn find_lru_candidate(&self) -> Option<BlockKey> {
        let mut oldest_time = SystemTime::now();
        let mut oldest_key = None;
        
        for entry in self.entries.iter() {
            let entry_time = entry.value().last_accessed;
            
            // Use a safe comparison based on elapsed time
            if let (Ok(entry_elapsed), Ok(oldest_elapsed)) = 
                (SystemTime::now().duration_since(entry_time), 
                 SystemTime::now().duration_since(oldest_time)) {
                
                if entry_elapsed > oldest_elapsed {
                    oldest_time = entry_time;
                    oldest_key = Some(*entry.key());
                }
            }
        }
        
        oldest_key
    }
}

impl LockFreeCachePolicy for LockFreeLRUPolicy {
    fn access(&self, key: &BlockKey) -> bool {
        let entry = self.entries.get(key);
        
        if let Some(entry_ref) = entry {
            // Update access metadata by removing and reinserting the entry
            // This effectively moves it to the "back" of the LRU
            let mut entry_value = entry_ref.value().clone();
            entry_value.last_accessed = SystemTime::now();
            entry_value.access_count += 1;
            
            // Remove the old entry and insert the updated one
            self.entries.remove(key);
            self.entries.insert(*key, entry_value);
            
            true
        } else {
            false
        }
    }
    
    fn add(&self, key: BlockKey, block: Arc<Block>) -> Option<BlockKey> {
        let now = SystemTime::now();
        
        // Check if we need to evict due to capacity constraints
        let should_evict = self.entries.len() >= self.capacity && 
                           self.entries.get(&key).is_none();
        
        let mut evicted_key = None;
        
        if should_evict {
            // Find the oldest entry (using last_accessed)
            if let Some(oldest_key) = self.find_lru_candidate() {
                self.entries.remove(&oldest_key);
                evicted_key = Some(oldest_key);
            }
        }
        
        // Insert or update the entry
        self.entries.insert(key, CacheEntry {
            block,
            last_accessed: now,
            access_count: 1,
        });
        
        evicted_key
    }
    
    fn remove(&self, key: &BlockKey) -> Option<Arc<Block>> {
        if let Some(entry) = self.entries.remove(key) {
            Some(entry.value().block.clone())
        } else {
            None
        }
    }
    
    fn get(&self, key: &BlockKey) -> Option<Arc<Block>> {
        let entry = self.entries.get(key);
        
        if let Some(entry_ref) = entry {
            // Get the block to return
            let block = entry_ref.value().block.clone();
            
            // Update access metadata by removing and reinserting the entry
            let mut entry_value = entry_ref.value().clone();
            entry_value.last_accessed = SystemTime::now();
            entry_value.access_count += 1;
            
            // Remove the old entry and insert the updated one
            self.entries.remove(key);
            self.entries.insert(*key, entry_value);
            
            Some(block)
        } else {
            None
        }
    }
    
    fn contains(&self, key: &BlockKey) -> bool {
        self.entries.contains_key(key)
    }
    
    fn clear(&self) {
        // Clear by removing each entry individually
        // (SkipMap doesn't have a clear method)
        let keys: Vec<BlockKey> = self.entries.iter()
            .map(|entry| *entry.key())
            .collect();
            
        for key in keys {
            self.entries.remove(&key);
        }
    }
    
    fn len(&self) -> usize {
        self.entries.len()
    }
    
    fn capacity(&self) -> usize {
        self.capacity
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn box_clone(&self) -> Box<dyn LockFreeCachePolicy> {
        Box::new(LockFreeLRUPolicy::new(self.capacity))
    }
}