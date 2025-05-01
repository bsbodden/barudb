use crate::run::{Block, lock_free_block_cache::BlockKey};
use crate::run::lock_free_cache_policies::LockFreeCachePolicy;
use crossbeam_skiplist::SkipMap;
use std::sync::{Arc, atomic::{AtomicU8, AtomicUsize, Ordering}};
use std::sync::Mutex;
use std::time::SystemTime;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::any::Any;

// Configuration constants
const WINDOW_RATIO: f64 = 0.2;      // Window is 20% of capacity
const PROTECTED_RATIO: f64 = 0.2;   // Protected segment is 20% of main segment
const RESET_AFTER_ENTRIES: usize = 10_000; // Reset frequency counters after this many entries

/// Thread-safe Count-Min Sketch for frequency tracking
#[derive(Debug)]
struct LockFreeCountMinSketch {
    /// Width of each row in the sketch
    width: usize,
    /// Number of hash functions / rows in the sketch
    depth: usize,
    /// The counters matrix - using atomic counters for thread safety
    counters: Vec<Vec<AtomicU8>>,
    /// Number of items added to the sketch - atomic for thread safety
    items_added: AtomicUsize,
}

impl LockFreeCountMinSketch {
    /// Create a new Count-Min sketch
    fn new(capacity: usize) -> Self {
        // Calculate width and depth based on capacity
        let effective_capacity = capacity.max(10);
        
        // Width is the most important parameter - aim for about 4x capacity
        let width = (effective_capacity.min(10000) / 4).max(16);
        let depth = 4; // Using 4 hash functions is common practice
        
        // Initialize all counters to 0
        let mut counters = Vec::with_capacity(depth);
        for _ in 0..depth {
            let mut row = Vec::with_capacity(width);
            for _ in 0..width {
                row.push(AtomicU8::new(0));
            }
            counters.push(row);
        }
        
        Self {
            width,
            depth,
            counters,
            items_added: AtomicUsize::new(0),
        }
    }
    
    /// Hash a key for a specific row
    fn hash(&self, key: &BlockKey, row: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        // Mix the row into the hash
        let hash = hasher.finish() ^ ((row as u64 + 1) * 0x9e3779b9);
        (hash % self.width as u64) as usize
    }
    
    /// Add an item to the sketch
    fn add(&self, key: &BlockKey) {
        for row in 0..self.depth {
            let col = self.hash(key, row);
            // Fetch-add to safely increment the counter
            let current = self.counters[row][col].load(Ordering::Relaxed);
            // Use saturating_add to avoid overflow
            if current < u8::MAX {
                self.counters[row][col].store(current.saturating_add(1), Ordering::Relaxed);
            }
        }
        
        // Check if we need to reset
        let items = self.items_added.fetch_add(1, Ordering::Relaxed);
        if items >= RESET_AFTER_ENTRIES {
            self.try_reset();
        }
    }
    
    /// Try to reset the counters by halving all values
    fn try_reset(&self) {
        // Reset items_added counter if it's above the threshold
        let current = self.items_added.load(Ordering::Relaxed);
        if current >= RESET_AFTER_ENTRIES {
            // Only one thread should perform the reset
            if self.items_added.compare_exchange(
                current, 0, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                // This thread won the race to reset
                for row in &self.counters {
                    for counter in row {
                        let current = counter.load(Ordering::Relaxed);
                        counter.store(current / 2, Ordering::Relaxed);
                    }
                }
            }
        }
    }
    
    /// Estimate the frequency of an item
    fn estimate(&self, key: &BlockKey) -> u8 {
        let mut min_count = u8::MAX;
        for row in 0..self.depth {
            let col = self.hash(key, row);
            let count = self.counters[row][col].load(Ordering::Relaxed);
            min_count = min_count.min(count);
        }
        min_count
    }
}

/// Cache entry for the TinyLFU policy
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached block
    block: Arc<Block>,
    /// When this entry was last accessed (currently only used for tracking, not for eviction decisions)
    _last_accessed: SystemTime,
}

/// Enum to track which segment a key belongs to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Segment {
    Window,
    Protected,
    Probation,
}

/// Lock-free W-TinyLFU implementation
#[derive(Debug)]
pub struct LockFreeTinyLFUPolicy {
    // Core configuration
    capacity: usize,
    window_size: usize,
    protected_size: usize,
    probation_size: usize,
    
    // Window cache (most recently added items)
    window: SkipMap<BlockKey, CacheEntry>,
    window_queue: Mutex<Vec<BlockKey>>, // For tracking order
    
    // Protected cache (items with high frequency)
    protected: SkipMap<BlockKey, CacheEntry>,
    protected_queue: Mutex<Vec<BlockKey>>, // For tracking order
    
    // Probation cache (items not yet frequent enough for protected)
    probation: SkipMap<BlockKey, CacheEntry>,
    probation_queue: Mutex<Vec<BlockKey>>, // For tracking order
    
    // Frequency sketch
    sketch: LockFreeCountMinSketch,
}

impl LockFreeTinyLFUPolicy {
    /// Create a new TinyLFU policy with the specified capacity
    pub fn new(capacity: usize) -> Self {
        // Ensure minimum capacity
        let effective_capacity = capacity.max(5);
        
        // Calculate segment sizes
        let window_size = (effective_capacity as f64 * WINDOW_RATIO).ceil() as usize;
        let main_size = effective_capacity - window_size;
        let protected_size = (main_size as f64 * PROTECTED_RATIO).ceil() as usize;
        let probation_size = main_size - protected_size;
        
        Self {
            capacity: effective_capacity,
            window_size,
            protected_size,
            probation_size,
            
            window: SkipMap::new(),
            window_queue: Mutex::new(Vec::with_capacity(window_size)),
            
            protected: SkipMap::new(),
            protected_queue: Mutex::new(Vec::with_capacity(protected_size)),
            
            probation: SkipMap::new(),
            probation_queue: Mutex::new(Vec::with_capacity(probation_size)),
            
            sketch: LockFreeCountMinSketch::new(effective_capacity),
        }
    }
    
    /// Find which segment contains a key
    fn get_item_segment(&self, key: &BlockKey) -> Option<Segment> {
        if self.window.contains_key(key) {
            Some(Segment::Window)
        } else if self.protected.contains_key(key) {
            Some(Segment::Protected)
        } else if self.probation.contains_key(key) {
            Some(Segment::Probation)
        } else {
            None
        }
    }
    
    /// Move a key to the back of its queue (mark as recently used)
    fn touch_key_in_queue(&self, key: &BlockKey, queue: &mut Vec<BlockKey>) {
        // Remove the key from its current position
        if let Some(pos) = queue.iter().position(|k| k == key) {
            queue.remove(pos);
        }
        
        // Add to the back (most recently used position)
        queue.push(*key);
    }
    
    /// Handle cache hit (item accessed)
    fn handle_hit(&self, key: &BlockKey, segment: Segment) {
        // Always update frequency
        self.sketch.add(key);
        
        match segment {
            Segment::Window => {
                // Move to back of window queue
                let mut queue = self.window_queue.lock().unwrap();
                self.touch_key_in_queue(key, &mut queue);
            },
            Segment::Protected => {
                // Move to back of protected queue
                let mut queue = self.protected_queue.lock().unwrap();
                self.touch_key_in_queue(key, &mut queue);
            },
            Segment::Probation => {
                // Try to promote from probation to protected if protected has space
                let protected_len = self.protected.len();
                
                if protected_len < self.protected_size {
                    // Protected has space, simple promotion
                    if let Some(entry) = self.probation.remove(key) {
                        // Remove from probation queue
                        let mut queue = self.probation_queue.lock().unwrap();
                        if let Some(pos) = queue.iter().position(|k| k == key) {
                            queue.remove(pos);
                        }
                        
                        // Add to protected
                        let block = entry.value().block.clone();
                        self.protected.insert(*key, CacheEntry {
                            block,
                            _last_accessed: SystemTime::now(),
                        });
                        
                        // Add to protected queue
                        let mut protected_queue = self.protected_queue.lock().unwrap();
                        protected_queue.push(*key);
                    }
                } else {
                    // Protected is full, need to demote LRU from protected to make room
                    
                    // First get entry from probation
                    if let Some(entry) = self.probation.remove(key) {
                        let block = entry.value().block.clone();
                        let mut probation_queue = self.probation_queue.lock().unwrap();
                        
                        // Remove from probation queue
                        if let Some(pos) = probation_queue.iter().position(|k| k == key) {
                            probation_queue.remove(pos);
                        }
                        
                        // Find protected victim
                        let protected_victim = {
                            let mut protected_queue = self.protected_queue.lock().unwrap();
                            if protected_queue.is_empty() {
                                None
                            } else {
                                // Get least recently used from protected
                                Some(protected_queue.remove(0))
                            }
                        };
                        
                        if let Some(victim_key) = protected_victim {
                            // Move victim from protected to probation
                            if let Some(victim_entry) = self.protected.remove(&victim_key) {
                                let victim_block = victim_entry.value().block.clone();
                                
                                // Add victim to probation
                                self.probation.insert(victim_key, CacheEntry {
                                    block: victim_block,
                                    _last_accessed: SystemTime::now(),
                                });
                                
                                // Add victim to probation queue
                                probation_queue.push(victim_key);
                                
                                // Add accessed item to protected
                                self.protected.insert(*key, CacheEntry {
                                    block,
                                    _last_accessed: SystemTime::now(),
                                });
                                
                                // Add accessed item to protected queue
                                let mut protected_queue = self.protected_queue.lock().unwrap();
                                protected_queue.push(*key);
                            }
                        } else {
                            // Protected was unexpectedly empty, just re-add to probation
                            self.probation.insert(*key, CacheEntry {
                                block,
                                _last_accessed: SystemTime::now(),
                            });
                            probation_queue.push(*key);
                        }
                    }
                }
            }
        }
    }
}

impl LockFreeCachePolicy for LockFreeTinyLFUPolicy {
    fn access(&self, key: &BlockKey) -> bool {
        if let Some(segment) = self.get_item_segment(key) {
            self.handle_hit(key, segment);
            true
        } else {
            false
        }
    }
    
    fn add(&self, key: BlockKey, block: Arc<Block>) -> Option<BlockKey> {
        // Update the frequency sketch
        self.sketch.add(&key);
        
        // Check if the key already exists in the cache
        if let Some(segment) = self.get_item_segment(&key) {
            self.handle_hit(&key, segment);
            return None;
        }
        
        // First, try to add to the window cache
        let window_full = self.window.len() >= self.window_size;
        
        if !window_full {
            // Simple case: window has space
            self.window.insert(key, CacheEntry {
                block,
                _last_accessed: SystemTime::now(),
            });
            
            let mut queue = self.window_queue.lock().unwrap();
            queue.push(key);
            
            return None;
        }
        
        // Window is full, need to evict the oldest item from window
        let window_victim = {
            let mut queue = self.window_queue.lock().unwrap();
            if queue.is_empty() {
                None
            } else {
                Some(queue.remove(0))
            }
        };
        
        // Add new item to window regardless of victim
        self.window.insert(key, CacheEntry {
            block: block.clone(),
            _last_accessed: SystemTime::now(),
        });
        
        // Add new item to window queue
        {
            let mut queue = self.window_queue.lock().unwrap();
            queue.push(key);
        }
        
        // Handle the evicted item from window
        if let Some(victim_key) = window_victim {
            if let Some(victim_entry) = self.window.remove(&victim_key) {
                let victim_block = victim_entry.value().block.clone();
                
                // Try to find space in main cache (probation + protected)
                let main_cache_full = 
                    (self.probation.len() + self.protected.len()) >= 
                    (self.probation_size + self.protected_size);
                
                if !main_cache_full {
                    // Main cache has space, add to probation
                    self.probation.insert(victim_key, CacheEntry {
                        block: victim_block,
                        _last_accessed: SystemTime::now(),
                    });
                    
                    // Add to probation queue
                    let mut queue = self.probation_queue.lock().unwrap();
                    queue.push(victim_key);
                    
                    return None;
                }
                
                // Main cache is full, need frequency-based admission
                // Compare frequence of window victim with LRU probation item
                
                let probation_victim = {
                    let mut queue = self.probation_queue.lock().unwrap();
                    if queue.is_empty() {
                        None
                    } else {
                        Some(queue.remove(0))
                    }
                };
                
                if let Some(prob_victim_key) = probation_victim {
                    // Compare frequencies to decide admission
                    let victim_freq = self.sketch.estimate(&victim_key);
                    let prob_victim_freq = self.sketch.estimate(&prob_victim_key);
                    
                    if victim_freq > prob_victim_freq {
                        // Window victim has higher frequency, replace probation victim
                        if let Some(_prob_victim_entry) = self.probation.remove(&prob_victim_key) {
                            // Add window victim to probation
                            self.probation.insert(victim_key, CacheEntry {
                                block: victim_block,
                                _last_accessed: SystemTime::now(),
                            });
                            
                            // Add to probation queue
                            let mut queue = self.probation_queue.lock().unwrap();
                            queue.push(victim_key);
                            
                            // Return the evicted probation victim
                            return Some(prob_victim_key);
                        }
                    } else {
                        // Probation victim has higher frequency, return window victim
                        // But first put probation victim back
                        {
                            let mut queue = self.probation_queue.lock().unwrap();
                            queue.push(prob_victim_key);
                        }
                        
                        return Some(victim_key);
                    }
                } else {
                    // No items in probation, try to replace from protected
                    // (unlikely but possible edge case)
                    let protected_victim = {
                        let mut queue = self.protected_queue.lock().unwrap();
                        if queue.is_empty() {
                            None
                        } else {
                            Some(queue.remove(0))
                        }
                    };
                    
                    if let Some(prot_victim_key) = protected_victim {
                        // Compare frequencies
                        let victim_freq = self.sketch.estimate(&victim_key);
                        let prot_victim_freq = self.sketch.estimate(&prot_victim_key);
                        
                        if victim_freq > prot_victim_freq {
                            // Window victim has higher frequency
                            if let Some(prot_victim_entry) = self.protected.remove(&prot_victim_key) {
                                let prot_victim_block = prot_victim_entry.value().block.clone();
                                
                                // Add window victim to protected
                                self.protected.insert(victim_key, CacheEntry {
                                    block: victim_block,
                                    _last_accessed: SystemTime::now(),
                                });
                                
                                // Add window victim to protected queue
                                let mut protected_queue = self.protected_queue.lock().unwrap();
                                protected_queue.push(victim_key);
                                
                                // Move protected victim to probation
                                self.probation.insert(prot_victim_key, CacheEntry {
                                    block: prot_victim_block,
                                    _last_accessed: SystemTime::now(),
                                });
                                
                                // Add protected victim to probation queue
                                let mut probation_queue = self.probation_queue.lock().unwrap();
                                probation_queue.push(prot_victim_key);
                                
                                return None;
                            }
                        } else {
                            // Protected victim has higher frequency
                            // Put it back and evict window victim
                            {
                                let mut queue = self.protected_queue.lock().unwrap();
                                queue.push(prot_victim_key);
                            }
                            
                            return Some(victim_key);
                        }
                    }
                }
                
                // If we reached here, we're evicting the window victim
                return Some(victim_key);
            }
        }
        
        None
    }
    
    fn remove(&self, key: &BlockKey) -> Option<Arc<Block>> {
        // Check each segment and remove if found
        if let Some(entry) = self.window.remove(key) {
            // Remove from window queue
            let mut queue = self.window_queue.lock().unwrap();
            if let Some(pos) = queue.iter().position(|k| k == key) {
                queue.remove(pos);
            }
            return Some(entry.value().block.clone());
        }
        
        if let Some(entry) = self.protected.remove(key) {
            // Remove from protected queue
            let mut queue = self.protected_queue.lock().unwrap();
            if let Some(pos) = queue.iter().position(|k| k == key) {
                queue.remove(pos);
            }
            return Some(entry.value().block.clone());
        }
        
        if let Some(entry) = self.probation.remove(key) {
            // Remove from probation queue
            let mut queue = self.probation_queue.lock().unwrap();
            if let Some(pos) = queue.iter().position(|k| k == key) {
                queue.remove(pos);
            }
            return Some(entry.value().block.clone());
        }
        
        None
    }
    
    fn get(&self, key: &BlockKey) -> Option<Arc<Block>> {
        // Check each segment
        if let Some(entry) = self.window.get(key) {
            let block = entry.value().block.clone();
            self.handle_hit(key, Segment::Window);
            return Some(block);
        }
        
        if let Some(entry) = self.protected.get(key) {
            let block = entry.value().block.clone();
            self.handle_hit(key, Segment::Protected);
            return Some(block);
        }
        
        if let Some(entry) = self.probation.get(key) {
            let block = entry.value().block.clone();
            self.handle_hit(key, Segment::Probation);
            return Some(block);
        }
        
        None
    }
    
    fn contains(&self, key: &BlockKey) -> bool {
        self.window.contains_key(key) || 
        self.protected.contains_key(key) || 
        self.probation.contains_key(key)
    }
    
    fn clear(&self) {
        // Clear each segment (by removing every item)
        
        // Clear window
        let window_keys: Vec<BlockKey> = self.window.iter()
            .map(|entry| *entry.key())
            .collect();
            
        for key in window_keys {
            self.window.remove(&key);
        }
        
        let mut window_queue = self.window_queue.lock().unwrap();
        window_queue.clear();
        
        // Clear protected
        let protected_keys: Vec<BlockKey> = self.protected.iter()
            .map(|entry| *entry.key())
            .collect();
            
        for key in protected_keys {
            self.protected.remove(&key);
        }
        
        let mut protected_queue = self.protected_queue.lock().unwrap();
        protected_queue.clear();
        
        // Clear probation
        let probation_keys: Vec<BlockKey> = self.probation.iter()
            .map(|entry| *entry.key())
            .collect();
            
        for key in probation_keys {
            self.probation.remove(&key);
        }
        
        let mut probation_queue = self.probation_queue.lock().unwrap();
        probation_queue.clear();
    }
    
    fn len(&self) -> usize {
        self.window.len() + self.protected.len() + self.probation.len()
    }
    
    fn capacity(&self) -> usize {
        self.capacity
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn box_clone(&self) -> Box<dyn LockFreeCachePolicy> {
        Box::new(LockFreeTinyLFUPolicy::new(self.capacity))
    }
}