use crate::run::{Block, RunId, Result, LockFreeCachePolicy, LockFreeCachePolicyFactory};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::Duration;
use std::any::Any;

/// Configuration for the block cache
#[derive(Debug, Clone)]
pub struct LockFreeBlockCacheConfig {
    /// Maximum number of blocks to keep in the cache
    pub max_capacity: usize,
    /// Maximum time to keep a block in the cache (in seconds)
    pub ttl: Duration,
    /// Clean interval (in seconds)
    pub cleanup_interval: Duration,
    /// Cache eviction policy type
    pub policy_type: crate::run::cache_policies::CachePolicyType,
}

impl Default for LockFreeBlockCacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 1000,
            ttl: Duration::from_secs(60 * 10), // 10 minutes
            cleanup_interval: Duration::from_secs(60),  // 1 minute
            policy_type: crate::run::cache_policies::CachePolicyType::TinyLFU, // Using TinyLFU as default for better performance
        }
    }
}

/// Cache key uniquely identifying a block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockKey {
    /// Run identifier
    pub run_id: RunId,
    /// Block index within the run
    pub block_idx: usize,
}

/// Statistics about cache performance using atomic counters
#[derive(Debug)]
pub struct LockFreeCacheStats {
    /// Number of cache hits
    hits: AtomicU64,
    /// Number of cache misses
    misses: AtomicU64,
    /// Number of blocks evicted due to capacity constraints
    capacity_evictions: AtomicU64,
    /// Number of blocks evicted due to TTL expiration
    ttl_evictions: AtomicU64,
    /// Number of blocks inserted
    inserts: AtomicU64,
}

impl Default for LockFreeCacheStats {
    fn default() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            capacity_evictions: AtomicU64::new(0),
            ttl_evictions: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
        }
    }
}

/// Clone implementation that captures the current values
impl Clone for LockFreeCacheStats {
    fn clone(&self) -> Self {
        Self {
            hits: AtomicU64::new(self.hits.load(Ordering::Relaxed)),
            misses: AtomicU64::new(self.misses.load(Ordering::Relaxed)), 
            capacity_evictions: AtomicU64::new(self.capacity_evictions.load(Ordering::Relaxed)),
            ttl_evictions: AtomicU64::new(self.ttl_evictions.load(Ordering::Relaxed)),
            inserts: AtomicU64::new(self.inserts.load(Ordering::Relaxed)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LockFreeCacheStatsSnapshot {
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

/// Simple wrapper for tracking time atomically
#[derive(Debug)]
struct AtomicInstant {
    /// Monotonic timestamp in milliseconds
    timestamp_ms: AtomicU64,
}

impl AtomicInstant {
    /// Create a new AtomicInstant with the current time
    fn now() -> Self {
        // Store the current system time in milliseconds since UNIX epoch
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        Self {
            timestamp_ms: AtomicU64::new(now),
        }
    }

    /// Update the instant to the current time
    fn update(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
            
        self.timestamp_ms.store(now, Ordering::SeqCst);
    }

    /// Get the elapsed time since this instant
    fn elapsed(&self) -> Duration {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
            
        let old = self.timestamp_ms.load(Ordering::SeqCst);
        let elapsed_ms = now.saturating_sub(old);
        
        Duration::from_millis(elapsed_ms)
    }
}

/// Lock-free block cache with pluggable policy for improved performance
#[derive(Debug)]
pub struct LockFreeBlockCache {
    /// Configuration options
    config: LockFreeBlockCacheConfig,
    /// The caching policy implementation
    policy: Box<dyn LockFreeCachePolicy>,
    /// Last cleanup time is tracked as an instant timestamp
    last_cleanup: Arc<AtomicInstant>,
    /// Statistics for cache performance monitoring - using atomic counters
    stats: LockFreeCacheStats,
}

impl LockFreeBlockCache {
    /// Create a new block cache with given configuration
    pub fn new(config: LockFreeBlockCacheConfig) -> Self {
        // Create policy based on configuration
        let policy = if config.policy_type == crate::run::cache_policies::CachePolicyType::TinyLFUWithTTL ||
                       config.policy_type == crate::run::cache_policies::CachePolicyType::PriorityLFU {
            // Use TTL-aware policy creation
            LockFreeCachePolicyFactory::create_with_ttl(
                config.policy_type,
                config.max_capacity,
                config.ttl
            )
        } else {
            // Regular policy creation
            LockFreeCachePolicyFactory::create(
                config.policy_type,
                config.max_capacity
            )
        };
        
        Self {
            config,
            policy,
            last_cleanup: Arc::new(AtomicInstant::now()),
            stats: LockFreeCacheStats::default(),
        }
    }

    /// Get a block from the cache
    pub fn get(&self, key: &BlockKey) -> Option<Arc<Block>> {
        // Try to get the block from the cache via policy
        let result = self.policy.get(key);
        
        // Update stats based on result
        if result.is_some() {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
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
        
        // Update statistics
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        
        if evicted.is_some() {
            self.stats.capacity_evictions.fetch_add(1, Ordering::Relaxed);
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
        // We'll need to iterate through all possible blocks and remove those matching run_id
        // This is inefficient but functional - would be better to have policy support for this
        let keys_to_remove = (0..10000)  // Arbitrary large number - we don't know block count
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
    pub fn get_stats(&self) -> LockFreeCacheStatsSnapshot {
        LockFreeCacheStatsSnapshot {
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            capacity_evictions: self.stats.capacity_evictions.load(Ordering::Relaxed),
            ttl_evictions: self.stats.ttl_evictions.load(Ordering::Relaxed),
            inserts: self.stats.inserts.load(Ordering::Relaxed),
        }
    }
    
    /// Get a reference to the policy for testing/inspection
    pub fn as_any(&self) -> &dyn Any {
        self.policy.as_any()
    }
    
    /// Set priority for a specific key (only works with priority-supporting policies)
    pub fn set_priority(&self, key: &BlockKey, priority: crate::run::lock_free_cache_policies::CachePriority) -> bool {
        self.policy.set_priority(key, priority)
    }
    
    /// Get priority for a specific key (only works with priority-supporting policies)
    pub fn get_priority(&self, key: &BlockKey) -> Option<crate::run::lock_free_cache_policies::CachePriority> {
        self.policy.get_priority(key)
    }
    
    /// Check if it's time to clean up expired entries
    fn maybe_cleanup(&self) {
        let should_cleanup = self.last_cleanup.elapsed() >= self.config.cleanup_interval;
        
        if should_cleanup {
            self.cleanup();
        }
    }
    
    /// Clean up expired entries based on TTL
    fn cleanup(&self) {
        // Update last cleanup time
        self.last_cleanup.update();
        
        // Use the policy's remove_expired method to clean up items
        let expired_count = self.policy.remove_expired(self.config.ttl);
        
        // Update TTL eviction statistics
        if expired_count > 0 {
            self.stats.ttl_evictions.fetch_add(expired_count as u64, Ordering::Relaxed);
        }
    }
    
    /// Manually trigger a cleanup (exposed for benchmarking)
    pub fn force_cleanup(&self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::cache_policies::CachePolicyType;
    use crate::run::lock_free_cache_policies::CachePriority;
    use std::thread::sleep;
    use std::thread;

    fn create_test_block(key: i64, value: i64) -> Block {
        let mut block = Block::new();
        block.add_entry(key, value).unwrap();
        block.seal().unwrap();
        block
    }

    #[test]
    fn test_block_cache_with_lru_policy() {
        let config = LockFreeBlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
            policy_type: CachePolicyType::LRU,
        };
        
        let cache = LockFreeBlockCache::new(config);
        
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
        let config = LockFreeBlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
            policy_type: CachePolicyType::TinyLFU,
        };
        
        let cache = LockFreeBlockCache::new(config);
        
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
        
        // Frequently accessed key1 should still be in cache with TinyLFU
        assert!(cache.get(&key1).is_some());
        
        // Check stats
        let stats = cache.get_stats();
        assert!(stats.hits >= 5); // At least the 5 explicit gets
        assert!(stats.inserts >= 11); // Initial 2 + at least 9 more
        assert!(stats.capacity_evictions > 0); // Some evictions should have occurred
    }
    
    #[test]
    fn test_block_cache_with_tiny_lfu_ttl_policy() {
        let config = LockFreeBlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_millis(500), // Very short TTL for testing
            cleanup_interval: Duration::from_millis(100),
            policy_type: CachePolicyType::TinyLFUWithTTL,
        };
        
        let cache = LockFreeBlockCache::new(config);
        
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
        
        // Insert blocks
        cache.insert(key1, block1).unwrap();
        cache.insert(key2, block2).unwrap();
        
        // Verify blocks are in the cache
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_some());
        
        // Wait for TTL to expire
        sleep(Duration::from_millis(600));
        
        // Force a cleanup
        cache.cleanup();
        
        // Keys should no longer be in the cache due to TTL expiration
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_none());
        
        // Check TTL eviction stats
        let stats = cache.get_stats();
        assert!(stats.ttl_evictions >= 2); // At least our 2 blocks were evicted
    }
    
    #[test]
    fn test_block_cache_with_priority_lfu_policy() {
        let config = LockFreeBlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
            policy_type: CachePolicyType::PriorityLFU,
        };
        
        let cache = LockFreeBlockCache::new(config);
        
        // Create test blocks
        let block1 = create_test_block(1, 100);
        let block2 = create_test_block(2, 200);
        let block3 = create_test_block(3, 300);
        
        // Create keys
        let key1 = BlockKey {
            run_id: RunId::new(0, 1),
            block_idx: 0,
        };
        
        let key2 = BlockKey {
            run_id: RunId::new(0, 1),
            block_idx: 1,
        };
        
        let key3 = BlockKey {
            run_id: RunId::new(0, 1),
            block_idx: 2,
        };
        
        // Insert first block as critical priority
        cache.insert(key1, block1).unwrap();
        cache.set_priority(&key1, CachePriority::Critical);
        
        // Insert second block as normal priority
        cache.insert(key2, block2).unwrap();
        cache.set_priority(&key2, CachePriority::Normal);
        
        // Insert third block as low priority
        cache.insert(key3, block3).unwrap();
        cache.set_priority(&key3, CachePriority::Low);
        
        // Verify priorities
        assert_eq!(cache.get_priority(&key1), Some(CachePriority::Critical));
        assert_eq!(cache.get_priority(&key2), Some(CachePriority::Normal));
        assert_eq!(cache.get_priority(&key3), Some(CachePriority::Low));
        
        // Fill the cache to trigger eviction
        for i in 4..15 {
            let key = BlockKey {
                run_id: RunId::new(0, 1),
                block_idx: i,
            };
            let block = create_test_block(i as i64, i as i64 * 100);
            cache.insert(key, block).unwrap();
        }
        
        // Critical priority key1 should still be in cache
        assert!(cache.get(&key1).is_some());
        
        // Low priority key3 likely evicted
        assert!(!cache.get(&key3).is_some());
    }
    
    #[test]
    fn test_eviction_behavior() {
        let config = LockFreeBlockCacheConfig {
            max_capacity: 2,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
            policy_type: CachePolicyType::LRU, // Test with LRU for predictable results
        };
        
        let cache = LockFreeBlockCache::new(config);
        
        // Create keys and blocks
        let key1 = BlockKey { run_id: RunId::new(0, 1), block_idx: 0 };
        let key2 = BlockKey { run_id: RunId::new(0, 1), block_idx: 1 };
        let key3 = BlockKey { run_id: RunId::new(0, 1), block_idx: 2 };
        
        let block1 = create_test_block(1, 100);
        let block2 = create_test_block(2, 200);
        let block3 = create_test_block(3, 300);
        
        // Insert blocks (capacity = 2)
        cache.insert(key1, block1).unwrap();
        cache.insert(key2, block2).unwrap();
        
        // Access key1 to make key2 the LRU candidate
        assert!(cache.get(&key1).is_some());
        
        // Insert block3, should evict one of the existing blocks
        cache.insert(key3, block3).unwrap();
        
        // Check what's in cache - we should have exactly 2 blocks
        let mut found_blocks = 0;
        if cache.get(&key1).is_some() { found_blocks += 1; }
        if cache.get(&key2).is_some() { found_blocks += 1; }
        if cache.get(&key3).is_some() { found_blocks += 1; }
        
        assert_eq!(found_blocks, 2);
        
        // Check stats
        let stats = cache.get_stats();
        assert_eq!(stats.capacity_evictions, 1);
    }
    
    #[test]
    fn test_concurrent_access() {
        let config = LockFreeBlockCacheConfig {
            max_capacity: 100,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
            policy_type: CachePolicyType::TinyLFU, // Use TinyLFU for concurrent test
        };
        
        let cache = Arc::new(LockFreeBlockCache::new(config));
        
        // Create 10 threads, each inserting 10 unique blocks
        let mut handles = vec![];
        
        for i in 0..10 {
            let thread_cache = cache.clone();
            
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let key = BlockKey {
                        run_id: RunId::new(0, 1),
                        block_idx: i * 10 + j,
                    };
                    
                    let block = create_test_block(i as i64 * 10 + j as i64, i as i64 * 100 + j as i64);
                    thread_cache.insert(key, block).unwrap();
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all blocks are accessible
        for i in 0..10 {
            for j in 0..10 {
                let key = BlockKey {
                    run_id: RunId::new(0, 1),
                    block_idx: i * 10 + j,
                };
                
                assert!(cache.get(&key).is_some());
            }
        }
        
        // Check stats
        let stats = cache.get_stats();
        assert_eq!(stats.inserts, 100);
        assert_eq!(stats.hits, 100);
    }
    
    #[test]
    fn test_default_policy_is_tinylfu() {
        // Create cache with default config
        let config = LockFreeBlockCacheConfig::default();
        
        // The default policy should be TinyLFU
        assert_eq!(config.policy_type, CachePolicyType::TinyLFU);
        
        let cache = LockFreeBlockCache::new(config);
        
        // Create a simple workload
        let key1 = BlockKey { run_id: RunId::new(0, 1), block_idx: 0 };
        let block1 = create_test_block(1, 100);
        
        cache.insert(key1, block1).unwrap();
        assert!(cache.get(&key1).is_some());
    }
}