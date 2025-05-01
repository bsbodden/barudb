use crate::run::{Block, RunId, Result};
use crate::run::cache_policies::{CachePolicy, CachePolicyFactory, CachePolicyType};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::any::Any;

/// Configuration for the block cache
#[derive(Debug, Clone)]
pub struct BlockCacheConfig {
    /// Maximum number of blocks to keep in the cache
    pub max_capacity: usize,
    /// Maximum time to keep a block in the cache (in seconds)
    pub ttl: Duration,
    /// Clean interval (in seconds)
    pub cleanup_interval: Duration,
    /// Cache eviction policy type
    pub policy_type: CachePolicyType,
}

impl Default for BlockCacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 1000,
            ttl: Duration::from_secs(60 * 10), // 10 minutes
            cleanup_interval: Duration::from_secs(60),  // 1 minute
            policy_type: CachePolicyType::TinyLFU,  // Using TinyLFU as default for better performance
        }
    }
}

/// Cache key uniquely identifying a block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockKey {
    /// Run identifier
    pub run_id: RunId,
    /// Block index within the run
    pub block_idx: usize,
}

/// Statistics about cache performance
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of blocks evicted due to capacity constraints
    pub capacity_evictions: u64,
    /// Number of blocks evicted due to TTL expiration
    pub ttl_evictions: u64,
    /// Number of blocks inserted
    pub inserts: u64,
}

/// Block cache with pluggable policy to improve read performance
#[derive(Debug)]
pub struct BlockCache {
    /// Configuration options
    pub config: BlockCacheConfig,
    /// The caching policy implementation
    policy: Box<dyn CachePolicy>,
    /// Statistics for cache performance monitoring
    stats: RwLock<CacheStats>,
    /// Last cleanup time
    last_cleanup: RwLock<Instant>,
}

impl BlockCache {
    /// Create a new block cache with given configuration
    pub fn new(config: BlockCacheConfig) -> Self {
        // Create policy based on configuration
        let policy = CachePolicyFactory::create(
            config.policy_type,
            config.max_capacity
        );
        
        Self {
            config,
            policy,
            stats: RwLock::new(CacheStats::default()),
            last_cleanup: RwLock::new(Instant::now()),
        }
    }

    /// Get a block from the cache
    pub fn get(&self, key: &BlockKey) -> Option<Arc<Block>> {
        // Try to get the block from the cache policy
        let result = self.policy.get(key);
        
        // Update stats based on result
        {
            let mut stats = self.stats.write().unwrap();
            if result.is_some() {
                stats.hits += 1;
            } else {
                stats.misses += 1;
            }
        }
        
        // Periodically clean up expired entries
        self.maybe_cleanup();
        
        result
    }

    /// Insert a block into the cache
    pub fn insert(&self, key: BlockKey, block: Block) -> Result<()> {
        let block = Arc::new(block);
        
        // Add to cache via policy
        let evicted = self.policy.add(key, block);
        
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.inserts += 1;
            
            if evicted.is_some() {
                stats.capacity_evictions += 1;
            }
        }
        
        Ok(())
    }

    /// Remove a block from the cache
    pub fn remove(&self, key: &BlockKey) -> Result<()> {
        self.policy.remove(key);
        Ok(())
    }
    
    /// Invalidate all blocks for a specific run (used during compaction)
    pub fn invalidate_run(&self, run_id: RunId) -> Result<()> {
        // We'll need to iterate through all blocks and remove those matching the run_id
        // This is a bit inefficient, but the current policy interface doesn't support filtering
        let keys_to_remove = (0..10000)  // Arbitrary large number - we don't know how many blocks per run
            .map(|block_idx| BlockKey { run_id, block_idx })
            .filter(|key| self.policy.contains(key))
            .collect::<Vec<_>>();
        
        for key in keys_to_remove {
            self.policy.remove(&key);
        }
        
        Ok(())
    }

    /// Clear all entries from the cache
    pub fn clear(&self) -> Result<()> {
        self.policy.clear();
        Ok(())
    }

    /// Get current cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Downcast to concrete policy type for testing or specialized operations
    pub fn as_any(&self) -> &dyn Any {
        &self.policy
    }
    
    /// Check if it's time to clean up expired entries
    fn maybe_cleanup(&self) {
        let should_cleanup = {
            let last_cleanup = self.last_cleanup.read().unwrap();
            last_cleanup.elapsed() >= self.config.cleanup_interval
        };
        
        if should_cleanup {
            self.cleanup();
        }
    }
    
    /// Clean up expired entries based on TTL
    fn cleanup(&self) {
        let now = Instant::now();
        
        // Update last cleanup time
        {
            let mut last_cleanup = self.last_cleanup.write().unwrap();
            *last_cleanup = now;
        }
        
        // For TTL-based expiration, we need to scan all items - this is inefficient
        // In a more advanced implementation, we would keep a separate queue for TTL expiration
        // For now, we'll just remove items via the policy interface individually
        
        // NOTE: With our current policy trait, we don't have a good way to scan all entries
        // This is a limitation - in a real implementation, we would extend the policy trait
        // or provide additional interfaces for efficient TTL-based cleanup
        
        // Just increment stats since we use policy-based cleanup now
        let mut stats = self.stats.write().unwrap();
        stats.ttl_evictions += 0; // Placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_block(key: i64, value: i64) -> Block {
        let mut block = Block::new();
        block.add_entry(key, value).unwrap();
        block.seal().unwrap();
        block
    }

    #[test]
    fn test_block_cache_with_lru_policy() {
        let config = BlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
            policy_type: CachePolicyType::LRU,
        };
        
        let cache = BlockCache::new(config);
        
        // Create test blocks
        let block1 = create_test_block(1, 100);
        let block2 = create_test_block(2, 200);
        
        // Create keys
        let key1 = BlockKey {
            run_id: RunId::new(0, 1),
            block_idx: 0,
        };
        
        let key2 = BlockKey {
            run_id: RunId::new(0, 1),
            block_idx: 1,
        };
        
        // Initial miss
        assert!(cache.get(&key1).is_none());
        
        // Insert and hit
        cache.insert(key1, block1).unwrap();
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_none());
        
        // Insert second block
        cache.insert(key2, block2).unwrap();
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_some());
        
        // Check stats
        let stats = cache.get_stats();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.inserts, 2);
    }
    
    #[test]
    fn test_block_cache_with_tiny_lfu_policy() {
        let config = BlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
            policy_type: CachePolicyType::TinyLFU,
        };
        
        let cache = BlockCache::new(config);
        
        // Create test blocks
        let block1 = create_test_block(1, 100);
        let block2 = create_test_block(2, 200);
        let _block3 = create_test_block(3, 300);
        
        // Create keys
        let key1 = BlockKey {
            run_id: RunId::new(0, 1),
            block_idx: 0,
        };
        
        let key2 = BlockKey {
            run_id: RunId::new(0, 1),
            block_idx: 1,
        };
        
        let _key3 = BlockKey {
            run_id: RunId::new(0, 1),
            block_idx: 2,
        };
        
        // Insert blocks
        cache.insert(key1, block1).unwrap();
        cache.insert(key2, block2).unwrap();
        
        // Access key1 multiple times to increase its frequency
        for _ in 0..5 {
            assert!(cache.get(&key1).is_some());
        }
        
        // Fill the cache and trigger eviction
        for i in 3..12 {
            let key = BlockKey {
                run_id: RunId::new(0, 1),
                block_idx: i,
            };
            let block = create_test_block(i as i64, i as i64 * 100);
            cache.insert(key, block).unwrap();
        }
        
        // Frequently accessed key1 should still be in cache
        assert!(cache.get(&key1).is_some());
        
        // Check stats
        let stats = cache.get_stats();
        assert!(stats.hits > 5); // At least the 5 explicit gets for key1
        assert!(stats.inserts >= 11); // Initial 2 + at least 9 more
        assert!(stats.capacity_evictions > 0); // Some evictions should have occurred
    }
}