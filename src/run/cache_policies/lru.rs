use crate::run::{Block, BlockKey};
use crate::run::cache_policies::CachePolicy;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::fmt::Debug;

/// LRU (Least Recently Used) cache policy implementation
#[derive(Debug)]
pub struct LRUPolicy {
    /// Maximum number of entries
    capacity: usize,
    /// Entries storage
    entries: RwLock<HashMap<BlockKey, Arc<Block>>>,
    /// LRU queue (least recently used at front)
    queue: Mutex<VecDeque<BlockKey>>,
}

impl LRUPolicy {
    /// Create a new LRU policy with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            queue: Mutex::new(VecDeque::with_capacity(capacity)),
        }
    }
    
    /// Move a key to the back of the queue (most recently used)
    fn touch_key(&self, key: &BlockKey) {
        let mut queue = self.queue.lock().unwrap();
        
        // Remove the key from its current position
        if let Some(pos) = queue.iter().position(|k| k == key) {
            queue.remove(pos);
        }
        
        // Add the key to the back (most recently used)
        queue.push_back(*key);
    }
}

impl CachePolicy for LRUPolicy {
    fn access(&self, key: &BlockKey) -> bool {
        let contains = {
            let entries = self.entries.read().unwrap();
            entries.contains_key(key)
        };
        
        if contains {
            self.touch_key(key);
        }
        
        contains
    }
    
    fn add(&self, key: BlockKey, block: Arc<Block>) -> Option<BlockKey> {
        let mut evicted = None;
        
        // Check if we need to evict
        {
            let entries = self.entries.read().unwrap();
            let current_size = entries.len();
            
            // If we're at capacity and this is a new key, we need to evict
            if current_size >= self.capacity && !entries.contains_key(&key) {
                // Get the least recently used key
                let mut queue = self.queue.lock().unwrap();
                evicted = queue.pop_front();
            }
        }
        
        // Perform eviction if needed
        if let Some(evicted_key) = evicted {
            let mut entries = self.entries.write().unwrap();
            entries.remove(&evicted_key);
        }
        
        // Add the new entry
        {
            let mut entries = self.entries.write().unwrap();
            entries.insert(key, block);
        }
        
        // Update the queue
        self.touch_key(&key);
        
        evicted
    }
    
    fn remove(&self, key: &BlockKey) -> Option<Arc<Block>> {
        let removed = {
            let mut entries = self.entries.write().unwrap();
            entries.remove(key)
        };
        
        if removed.is_some() {
            // Remove from queue
            let mut queue = self.queue.lock().unwrap();
            if let Some(pos) = queue.iter().position(|k| k == key) {
                queue.remove(pos);
            }
        }
        
        removed
    }
    
    fn get(&self, key: &BlockKey) -> Option<Arc<Block>> {
        let result = {
            let entries = self.entries.read().unwrap();
            entries.get(key).cloned()
        };
        
        if result.is_some() {
            self.touch_key(key);
        }
        
        result
    }
    
    fn contains(&self, key: &BlockKey) -> bool {
        let entries = self.entries.read().unwrap();
        entries.contains_key(key)
    }
    
    fn clear(&self) {
        {
            let mut entries = self.entries.write().unwrap();
            entries.clear();
        }
        
        {
            let mut queue = self.queue.lock().unwrap();
            queue.clear();
        }
    }
    
    fn len(&self) -> usize {
        let entries = self.entries.read().unwrap();
        entries.len()
    }
    
    fn capacity(&self) -> usize {
        self.capacity
    }
    
    fn box_clone(&self) -> Box<dyn CachePolicy> {
        Box::new(LRUPolicy::new(self.capacity))
    }
}