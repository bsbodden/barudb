use crate::run::{Block, RunId, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Cache entry containing the block and metadata
#[derive(Debug)]
struct CacheEntry {
    /// The cached block
    block: Arc<Block>,
    /// When this entry was last accessed
    last_accessed: Instant,
    /// Number of times this block has been accessed
    access_count: u64,
}

/// Configuration for the block cache
#[derive(Debug, Clone)]
pub struct BlockCacheConfig {
    /// Maximum number of blocks to keep in the cache
    pub max_capacity: usize,
    /// Maximum time to keep a block in the cache (in seconds)
    pub ttl: Duration,
    /// Clean interval (in seconds)
    pub cleanup_interval: Duration,
}

impl Default for BlockCacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 1000,
            ttl: Duration::from_secs(60 * 10), // 10 minutes
            cleanup_interval: Duration::from_secs(60),  // 1 minute
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

/// LRU cache for blocks to improve read performance
#[derive(Debug)]
pub struct BlockCache {
    /// Configuration options
    config: BlockCacheConfig,
    /// Block cache entries
    entries: RwLock<HashMap<BlockKey, CacheEntry>>,
    /// LRU queue for eviction (least recently used at front)
    lru_queue: Mutex<VecDeque<BlockKey>>,
    /// Statistics for cache performance monitoring
    stats: RwLock<CacheStats>,
    /// Last cleanup time
    last_cleanup: RwLock<Instant>,
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

impl BlockCache {
    /// Create a new block cache with given configuration
    pub fn new(config: BlockCacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            lru_queue: Mutex::new(VecDeque::new()),
            stats: RwLock::new(CacheStats::default()),
            last_cleanup: RwLock::new(Instant::now()),
        }
    }

    /// Get a block from the cache
    pub fn get(&self, key: &BlockKey) -> Option<Arc<Block>> {
        let now = Instant::now();
        
        // Try to get the block from the cache
        let result = {
            let mut entries = self.entries.write().unwrap();
            
            if let Some(entry) = entries.get_mut(key) {
                // Update access metadata
                entry.last_accessed = now;
                entry.access_count += 1;
                
                // Update cache stats
                let mut stats = self.stats.write().unwrap();
                stats.hits += 1;
                
                // Update LRU queue (move to back)
                let mut lru = self.lru_queue.lock().unwrap();
                if let Some(pos) = lru.iter().position(|k| k == key) {
                    lru.remove(pos);
                }
                lru.push_back(*key);
                
                Some(entry.block.clone())
            } else {
                // Update cache stats for miss
                let mut stats = self.stats.write().unwrap();
                stats.misses += 1;
                None
            }
        };
        
        // Periodically clean up expired entries
        self.maybe_cleanup();
        
        result
    }

    /// Insert a block into the cache
    pub fn insert(&self, key: BlockKey, block: Block) -> Result<()> {
        let now = Instant::now();
        let block = Arc::new(block);
        
        // Add to cache and update LRU queue
        {
            let mut entries = self.entries.write().unwrap();
            let mut lru = self.lru_queue.lock().unwrap();
            
            // Update stats
            let mut stats = self.stats.write().unwrap();
            stats.inserts += 1;
            
            // Check if we need to evict due to capacity constraints
            if entries.len() >= self.config.max_capacity && !entries.contains_key(&key) {
                // Evict the least recently used entry
                if let Some(lru_key) = lru.pop_front() {
                    entries.remove(&lru_key);
                    stats.capacity_evictions += 1;
                }
            }
            
            // Add or update cache entry
            entries.insert(key, CacheEntry {
                block: block.clone(),
                last_accessed: now,
                access_count: 1,
            });
            
            // Update LRU queue (remove if exists, then add to back)
            if let Some(pos) = lru.iter().position(|k| k == &key) {
                lru.remove(pos);
            }
            lru.push_back(key);
        }
        
        Ok(())
    }

    /// Remove a block from the cache
    pub fn remove(&self, key: &BlockKey) -> Result<()> {
        let mut entries = self.entries.write().unwrap();
        let mut lru = self.lru_queue.lock().unwrap();
        
        // Remove from entries
        entries.remove(key);
        
        // Remove from LRU queue
        if let Some(pos) = lru.iter().position(|k| k == key) {
            lru.remove(pos);
        }
        
        Ok(())
    }

    /// Clear all entries from the cache
    pub fn clear(&self) -> Result<()> {
        let mut entries = self.entries.write().unwrap();
        let mut lru = self.lru_queue.lock().unwrap();
        
        entries.clear();
        lru.clear();
        
        Ok(())
    }

    /// Get current cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
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
        
        // Remove expired entries
        let mut entries = self.entries.write().unwrap();
        let mut lru = self.lru_queue.lock().unwrap();
        let mut stats = self.stats.write().unwrap();
        
        let expired_keys: Vec<BlockKey> = entries
            .iter()
            .filter(|(_, entry)| entry.last_accessed.elapsed() > self.config.ttl)
            .map(|(key, _)| *key)
            .collect();
            
        for key in expired_keys {
            entries.remove(&key);
            if let Some(pos) = lru.iter().position(|k| k == &key) {
                lru.remove(pos);
            }
            stats.ttl_evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    fn create_test_block(key: i64, value: i64) -> Block {
        let mut block = Block::new();
        block.add_entry(key, value).unwrap();
        block.seal().unwrap();
        block
    }

    #[test]
    fn test_block_cache_get_insert() {
        let config = BlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
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
    fn test_block_cache_eviction() {
        let config = BlockCacheConfig {
            max_capacity: 2,
            ttl: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(1),
        };
        
        let cache = BlockCache::new(config);
        
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
        
        // Insert block3, should evict block2
        cache.insert(key3, block3).unwrap();
        
        // Check what's in cache
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_none()); // Evicted
        assert!(cache.get(&key3).is_some());
        
        // Check stats
        let stats = cache.get_stats();
        assert_eq!(stats.capacity_evictions, 1);
    }
    
    #[test]
    fn test_block_cache_ttl_expiration() {
        let config = BlockCacheConfig {
            max_capacity: 10,
            ttl: Duration::from_millis(100), // Short TTL for testing
            cleanup_interval: Duration::from_millis(50),
        };
        
        let cache = BlockCache::new(config);
        
        // Create key and block
        let key = BlockKey { run_id: RunId::new(0, 1), block_idx: 0 };
        let block = create_test_block(1, 100);
        
        // Insert block
        cache.insert(key, block).unwrap();
        assert!(cache.get(&key).is_some());
        
        // Wait for TTL to expire
        sleep(Duration::from_millis(200));
        
        // Force cleanup by accessing another key
        let key2 = BlockKey { run_id: RunId::new(0, 2), block_idx: 0 };
        cache.get(&key2);
        
        // Original entry should be gone
        assert!(cache.get(&key).is_none());
        
        // Check stats
        let stats = cache.get_stats();
        assert_eq!(stats.ttl_evictions, 1);
    }
}