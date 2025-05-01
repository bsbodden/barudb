use crate::run::{Block, lock_free_block_cache::BlockKey};
use std::sync::Arc;
use std::fmt::Debug;
use std::time::Duration;
use std::any::Any;

mod tiny_lfu;
mod lru;
mod tiny_lfu_ttl;
mod priority_lfu;

pub use tiny_lfu::LockFreeTinyLFUPolicy;
pub use lru::LockFreeLRUPolicy;
pub use tiny_lfu_ttl::LockFreeTinyLFUTTLPolicy;
pub use priority_lfu::LockFreePriorityLFUPolicy;

/// Priority level for cache entries in priority-based policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CachePriority {
    /// Critical priority - evict only as a last resort
    Critical,
    /// High priority - prefer to keep in cache
    High,
    /// Normal priority - standard cache entry
    Normal,
    /// Low priority - candidate for early eviction
    Low,
}

/// Trait defining a lock-free block cache eviction policy
pub trait LockFreeCachePolicy: Send + Sync + Debug {
    /// Access a key in the cache (mark it as recently used)
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
    
    /// Cast to Any for dynamic typing
    fn as_any(&self) -> &dyn Any;
    
    /// Clone this policy, creating a new policy with the same configuration
    fn box_clone(&self) -> Box<dyn LockFreeCachePolicy>;
    
    /// Scan and identify expired entries based on TTL
    /// Returns a list of keys that have expired
    fn scan_expired(&self, _ttl: Duration) -> Vec<BlockKey> {
        // Default implementation returns empty list - no TTL support
        Vec::new()
    }
    
    /// Remove expired entries based on TTL
    /// Returns the number of entries removed
    fn remove_expired(&self, _ttl: Duration) -> usize {
        // Default implementation does nothing - no TTL support
        0
    }
    
    /// Set priority for a specific cache entry
    /// Returns true if the key was found and priority was set
    fn set_priority(&self, _key: &BlockKey, _priority: CachePriority) -> bool {
        // Default implementation does nothing - no priority support
        false
    }
    
    /// Get the priority of a cache entry
    /// Returns the priority if found, None otherwise
    fn get_priority(&self, _key: &BlockKey) -> Option<CachePriority> {
        // Default implementation returns None - no priority support
        None
    }
}

/// Factory for creating different lock-free cache policies
pub struct LockFreeCachePolicyFactory;

impl LockFreeCachePolicyFactory {
    /// Create a new lock-free cache policy of the specified type
    pub fn create(
        policy_type: crate::run::cache_policies::CachePolicyType, 
        capacity: usize
    ) -> Box<dyn LockFreeCachePolicy> {
        match policy_type {
            crate::run::cache_policies::CachePolicyType::LRU => 
                Box::new(LockFreeLRUPolicy::new(capacity)),
            crate::run::cache_policies::CachePolicyType::TinyLFU => 
                Box::new(LockFreeTinyLFUPolicy::new(capacity)),
            crate::run::cache_policies::CachePolicyType::TinyLFUWithTTL => 
                Box::new(LockFreeTinyLFUTTLPolicy::new(capacity, Duration::from_secs(600))),
            crate::run::cache_policies::CachePolicyType::PriorityLFU => 
                Box::new(LockFreePriorityLFUPolicy::new(capacity)),
        }
    }
    
    /// Create a new lock-free cache policy with TTL configuration
    pub fn create_with_ttl(
        policy_type: crate::run::cache_policies::CachePolicyType,
        capacity: usize,
        ttl: Duration
    ) -> Box<dyn LockFreeCachePolicy> {
        match policy_type {
            crate::run::cache_policies::CachePolicyType::TinyLFUWithTTL => 
                Box::new(LockFreeTinyLFUTTLPolicy::new(capacity, ttl)),
            crate::run::cache_policies::CachePolicyType::PriorityLFU => 
                Box::new(LockFreePriorityLFUPolicy::new_with_ttl(capacity, ttl)),
            // For policy types that don't support TTL directly, use the non-TTL version
            _ => Self::create(policy_type, capacity),
        }
    }
}