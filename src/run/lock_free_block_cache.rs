use crate::run::{Block, RunId, Result};
use crossbeam_skiplist::SkipMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::Duration;

/// Cache entry containing the block and metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached block
    block: Arc<Block>,
    /// When this entry was last accessed - use SystemTime for consistency with AtomicInstant
    last_accessed: std::time::SystemTime,
    /// Number of times this block has been accessed
    access_count: u64,
}

/// Configuration for the block cache
#[derive(Debug, Clone)]
pub struct LockFreeBlockCacheConfig {
    /// Maximum number of blocks to keep in the cache
    pub max_capacity: usize,
    /// Maximum time to keep a block in the cache (in seconds)
    pub ttl: Duration,
    /// Clean interval (in seconds)
    pub cleanup_interval: Duration,
}

impl Default for LockFreeBlockCacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 1000,
            ttl: Duration::from_secs(60 * 10), // 10 minutes
            cleanup_interval: Duration::from_secs(60),  // 1 minute
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

/// Simple wrapper for Instant that allows atomic updates
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

/// Lock-free LRU cache for blocks to improve read performance
#[derive(Debug)]
pub struct LockFreeBlockCache {
    /// Configuration options
    config: LockFreeBlockCacheConfig,
    /// Block cache entries - uses lock-free SkipMap from crossbeam
    entries: SkipMap<BlockKey, CacheEntry>,
    /// Last cleanup time is tracked as an instant timestamp
    last_cleanup: Arc<AtomicInstant>,
    /// Statistics for cache performance monitoring - using atomic counters
    stats: LockFreeCacheStats,
}

impl LockFreeBlockCache {
    /// Create a new block cache with given configuration
    pub fn new(config: LockFreeBlockCacheConfig) -> Self {
        Self {
            config,
            entries: SkipMap::new(),
            last_cleanup: Arc::new(AtomicInstant::now()),
            stats: LockFreeCacheStats::default(),
        }
    }

    /// Get a block from the cache
    pub fn get(&self, key: &BlockKey) -> Option<Arc<Block>> {
        // Try to get the block from the cache
        let entry = self.entries.get(key);
        
        let result = if let Some(entry_ref) = entry {
            // Get and return the block
            let block = entry_ref.value().block.clone();
            
            // Update access metadata by removing and reinserting the entry
            // This effectively moves it to the "back" of the LRU
            let mut entry_value = entry_ref.value().clone();
            entry_value.last_accessed = std::time::SystemTime::now();
            entry_value.access_count += 1;
            
            // Remove the old entry and insert the updated one
            self.entries.remove(key);
            self.entries.insert(*key, entry_value);
            
            // Increment hit counter
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            
            Some(block)
        } else {
            // Increment miss counter
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        };
        
        // Periodically clean up expired entries
        self.maybe_cleanup();
        
        result
    }

    /// Insert a block into the cache
    pub fn insert(&self, key: BlockKey, block: Block) -> Result<()> {
        let now = std::time::SystemTime::now();
        let block = Arc::new(block);
        
        // Check if we need to evict due to capacity constraints
        if self.entries.len() >= self.config.max_capacity && self.entries.get(&key).is_none() {
            // Find the oldest entry (using last_accessed)
            // This is an O(n) operation but is only performed when the cache is full
            if let Some(oldest_key) = self.find_lru_candidate() {
                self.entries.remove(&oldest_key);
                
                // Increment capacity evictions counter
                self.stats.capacity_evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        // Insert or update the entry
        self.entries.insert(key, CacheEntry {
            block: block.clone(),
            last_accessed: now,
            access_count: 1,
        });
        
        // Increment insert counter
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }

    /// Find the least recently used entry - this is a fallback when the cache is full
    fn find_lru_candidate(&self) -> Option<BlockKey> {
        let mut oldest_time = std::time::SystemTime::now();
        let mut oldest_key = None;
        
        for entry in self.entries.iter() {
            // Use a safe comparison based on relative timestamps
            if entry.value().last_accessed.elapsed().unwrap_or_default() > 
               oldest_time.elapsed().unwrap_or_default() {
                oldest_time = entry.value().last_accessed;
                oldest_key = Some(*entry.key());
            }
        }
        
        oldest_key
    }

    /// Remove a block from the cache
    pub fn remove(&self, key: &BlockKey) -> Result<()> {
        self.entries.remove(key);
        Ok(())
    }
    
    /// Invalidate all blocks for a specific run (used during compaction)
    pub fn invalidate_run(&self, run_id: RunId) -> Result<()> {
        // Collect keys to remove
        let keys_to_remove: Vec<BlockKey> = self.entries.iter()
            .filter(|entry| entry.key().run_id == run_id)
            .map(|entry| *entry.key())
            .collect();
        
        // Remove all matching keys
        for key in keys_to_remove {
            self.entries.remove(&key);
        }
        
        Ok(())
    }

    /// Clear all entries from the cache
    pub fn clear(&self) -> Result<()> {
        // Clear by removing each entry individually
        // (SkipMap doesn't have a clear method)
        let keys: Vec<BlockKey> = self.entries.iter()
            .map(|entry| *entry.key())
            .collect();
            
        for key in keys {
            self.entries.remove(&key);
        }
        
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
        
        let now = std::time::SystemTime::now();
        let ttl = self.config.ttl;
        
        // Collect expired keys
        let expired_keys: Vec<BlockKey> = self.entries.iter()
            .filter(|entry| {
                // Safely handle SystemTime comparison by using elapsed
                if let Ok(entry_age) = now.duration_since(entry.value().last_accessed) {
                    entry_age >= ttl
                } else {
                    false // Handle potential SystemTime errors
                }
            })
            .map(|entry| *entry.key())
            .collect();
        
        if !expired_keys.is_empty() {
            // Remove expired entries
            for key in &expired_keys {
                self.entries.remove(key);
            }
            
            // Update ttl evictions counter
            self.stats.ttl_evictions.fetch_add(expired_keys.len() as u64, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::thread;

    fn create_test_block(key: i64, value: i64) -> Block {
        let mut block = Block::new();
        block.add_entry(key, value).unwrap();
        block.seal().unwrap();
        block
    }

    #[test]
    fn test_block_cache_get_insert() {
        let config = LockFreeBlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
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
    fn test_block_cache_eviction() {
        let config = LockFreeBlockCacheConfig {
            max_capacity: 2,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
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
    fn test_block_cache_ttl_expiration() {
        let config = LockFreeBlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_millis(50), // Very short TTL for testing
            cleanup_interval: Duration::from_millis(10), // Very frequent cleanup
        };
        
        let cache = LockFreeBlockCache::new(config);
        
        // Create key and block
        let key = BlockKey { run_id: RunId::new(0, 1), block_idx: 0 };
        let block = create_test_block(1, 100);
        
        // Insert block
        cache.insert(key, block).unwrap();
        assert!(cache.get(&key).is_some());
        
        // Wait for TTL to expire - wait longer than needed to ensure expiration
        sleep(Duration::from_millis(150));
        
        // Explicitly call cleanup directly to ensure it runs
        cache.cleanup();
        
        // Wait a bit more to ensure cleanup has completed
        sleep(Duration::from_millis(10));
        
        // Original entry should be gone
        assert!(cache.get(&key).is_none());
        
        // Check stats
        let stats = cache.get_stats();
        assert!(stats.ttl_evictions > 0);
    }
    
    #[test]
    fn test_concurrent_access() {
        let config = LockFreeBlockCacheConfig {
            max_capacity: 100,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
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
}