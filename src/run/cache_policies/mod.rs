use crate::run::{Block, BlockKey};  // Remove unused Result import
use std::sync::Arc;
use std::fmt::Debug;

mod lru;
mod tiny_lfu;
#[cfg(test)]
pub mod tests;  // Make tests module public

pub use lru::LRUPolicy;
pub use tiny_lfu::TinyLFUPolicy;

/// Trait defining a block cache eviction policy
pub trait CachePolicy: Send + Sync + Debug {
    /// Access a key in the cache
    /// Returns true if the key was found
    fn access(&self, key: &BlockKey) -> bool;
    
    /// Add a new entry to the cache
    /// Returns Some(evicted_key) if an entry was evicted, None otherwise
    fn add(&self, key: BlockKey, block: Arc<Block>) -> Option<BlockKey>;
    
    /// Remove a specific key from the cache
    fn remove(&self, key: &BlockKey) -> Option<Arc<Block>>;
    
    /// Get an entry from the cache
    fn get(&self, key: &BlockKey) -> Option<Arc<Block>>;
    
    /// Check if an entry exists in the cache
    fn contains(&self, key: &BlockKey) -> bool;
    
    /// Clear all entries
    fn clear(&self);
    
    /// Get the number of entries in the cache
    fn len(&self) -> usize;
    
    /// Check if the cache is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the maximum capacity of the cache
    fn capacity(&self) -> usize;
    
    /// Clone this policy, creating a new policy with the same configuration
    fn box_clone(&self) -> Box<dyn CachePolicy>;
}

/// Factory for creating different cache policies
pub struct CachePolicyFactory;

/// Available cache policy types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicyType {
    /// Simple LRU eviction policy
    LRU,
    /// TinyLFU eviction policy
    TinyLFU,
    /// TinyLFU with TTL-based expiration
    TinyLFUWithTTL,
    /// Priority-based eviction policy
    PriorityLFU,
}

impl CachePolicyFactory {
    /// Create a new cache policy of the specified type
    pub fn create(policy_type: CachePolicyType, capacity: usize) -> Box<dyn CachePolicy> {
        match policy_type {
            CachePolicyType::LRU => Box::new(LRUPolicy::new(capacity)),
            CachePolicyType::TinyLFU => Box::new(TinyLFUPolicy::new(capacity)),
            CachePolicyType::TinyLFUWithTTL => Box::new(TinyLFUPolicy::new(capacity)), // Fallback to regular TinyLFU
            CachePolicyType::PriorityLFU => Box::new(TinyLFUPolicy::new(capacity)),    // Fallback to regular TinyLFU
        }
    }
}