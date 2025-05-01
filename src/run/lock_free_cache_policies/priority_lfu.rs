use crate::run::{Block, lock_free_block_cache::BlockKey};
use crate::run::lock_free_cache_policies::{LockFreeCachePolicy, CachePriority};
use crossbeam_skiplist::SkipMap;
use std::sync::{Arc, atomic::{AtomicU8, AtomicUsize, Ordering}};
use std::sync::Mutex;
use std::time::{SystemTime, Duration};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::any::Any;

// Configuration constants
const WINDOW_RATIO: f64 = 0.2;      // Window is 20% of capacity
const PROTECTED_RATIO: f64 = 0.2;   // Protected segment is 20% of main segment
const RESET_AFTER_ENTRIES: usize = 10_000; // Reset frequency counters after this many entries

// Priority multipliers - higher values make items more likely to be preserved
const CRITICAL_MULTIPLIER: u8 = 5;
const HIGH_MULTIPLIER: u8 = 3;
const NORMAL_MULTIPLIER: u8 = 1;
const LOW_MULTIPLIER: u8 = 0;  // Low priority items kept only if they're very hot

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
    
    /// Estimate the frequency with a priority multiplier
    fn estimate_with_priority(&self, key: &BlockKey, priority: CachePriority) -> u16 {
        let frequency = self.estimate(key) as u16;
        let multiplier = match priority {
            CachePriority::Critical => CRITICAL_MULTIPLIER as u16,
            CachePriority::High => HIGH_MULTIPLIER as u16,
            CachePriority::Normal => NORMAL_MULTIPLIER as u16,
            CachePriority::Low => LOW_MULTIPLIER as u16,
        };
        
        frequency * multiplier
    }
}

/// Cache entry for the Priority LFU policy
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached block
    block: Arc<Block>,
    /// When this entry was last accessed
    last_accessed: SystemTime,
    /// When this entry was created
    created_at: SystemTime,
    /// Priority of this entry
    priority: CachePriority,
}

/// Enum to track which segment a key belongs to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Segment {
    Window,
    Protected,
    Probation,
}

/// Lock-free W-TinyLFU implementation with priority support
#[derive(Debug)]
pub struct LockFreePriorityLFUPolicy {
    // Core configuration
    capacity: usize,
    window_size: usize,
    protected_size: usize,
    probation_size: usize,
    ttl: Option<Duration>,
    
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
    
    // Priority index - for efficient priority lookups
    priorities: SkipMap<BlockKey, CachePriority>,
    
    // TTL index - tracks entries by insertion time for efficient TTL expiration (if TTL is enabled)
    ttl_index: SkipMap<SystemTime, BlockKey>,
}

impl LockFreePriorityLFUPolicy {
    /// Create a new Priority LFU policy
    pub fn new(capacity: usize) -> Self {
        Self::new_with_ttl_option(capacity, None)
    }
    
    /// Create a new Priority LFU policy with TTL
    pub fn new_with_ttl(capacity: usize, ttl: Duration) -> Self {
        Self::new_with_ttl_option(capacity, Some(ttl))
    }
    
    /// Internal helper for creation with optional TTL
    fn new_with_ttl_option(capacity: usize, ttl: Option<Duration>) -> Self {
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
            ttl,
            
            window: SkipMap::new(),
            window_queue: Mutex::new(Vec::with_capacity(window_size)),
            
            protected: SkipMap::new(),
            protected_queue: Mutex::new(Vec::with_capacity(protected_size)),
            
            probation: SkipMap::new(),
            probation_queue: Mutex::new(Vec::with_capacity(probation_size)),
            
            sketch: LockFreeCountMinSketch::new(effective_capacity),
            
            priorities: SkipMap::new(),
            
            ttl_index: SkipMap::new(),
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
    fn handle_hit(&self, key: &BlockKey, segment: Segment) -> bool {
        // Always update frequency
        self.sketch.add(key);
        
        // Check for ttl expiration if it's enabled
        if let Some(ttl) = self.ttl {
            let created_at = match segment {
                Segment::Window => self.window.get(key).map(|e| e.value().created_at),
                Segment::Protected => self.protected.get(key).map(|e| e.value().created_at),
                Segment::Probation => self.probation.get(key).map(|e| e.value().created_at),
            };
            
            if let Some(created_at) = created_at {
                let now = SystemTime::now();
                if now.duration_since(created_at).unwrap_or_default() > ttl {
                    // Item has expired - remove and return false
                    self.remove(key);
                    return false;
                }
            }
        }
        
        match segment {
            Segment::Window => {
                // Update last_accessed time
                if let Some(entry) = self.window.get(key) {
                    // Create updated entry with new last_accessed time
                    let mut updated_entry = entry.value().clone();
                    updated_entry.last_accessed = SystemTime::now();
                    
                    // Replace entry with updated version
                    self.window.insert(*key, updated_entry);
                    
                    // Move to back of window queue
                    let mut queue = self.window_queue.lock().unwrap();
                    self.touch_key_in_queue(key, &mut queue);
                    
                    return true;
                }
                false
            },
            Segment::Protected => {
                // Update last_accessed time
                if let Some(entry) = self.protected.get(key) {
                    // Create updated entry with new last_accessed time
                    let mut updated_entry = entry.value().clone();
                    updated_entry.last_accessed = SystemTime::now();
                    
                    // Replace entry with updated version
                    self.protected.insert(*key, updated_entry);
                    
                    // Move to back of protected queue
                    let mut queue = self.protected_queue.lock().unwrap();
                    self.touch_key_in_queue(key, &mut queue);
                    
                    return true;
                }
                false
            },
            Segment::Probation => {
                // Get current entry and priority
                if let Some(entry) = self.probation.get(key) {
                    let priority = entry.value().priority;
                    let now = SystemTime::now();
                    
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
                            
                            // Create updated entry with new last_accessed time
                            let mut updated_entry = entry.value().clone();
                            updated_entry.last_accessed = now;
                            
                            // Add to protected
                            self.protected.insert(*key, updated_entry);
                            
                            // Add to protected queue
                            let mut protected_queue = self.protected_queue.lock().unwrap();
                            protected_queue.push(*key);
                            
                            return true;
                        }
                    } else {
                        // Protected is full, need to demote LRU from protected to make room
                        
                        // First update entry in probation
                        if let Some(entry) = self.probation.remove(key) {
                            // Create updated entry with new last_accessed time
                            let mut updated_entry = entry.value().clone();
                            updated_entry.last_accessed = now;
                            
                            let mut probation_queue = self.probation_queue.lock().unwrap();
                            
                            // Remove from probation queue
                            if let Some(pos) = probation_queue.iter().position(|k| k == key) {
                                probation_queue.remove(pos);
                            }
                            
                            // Find protected victim
                            let protected_victim = {
                                let mut protected_queue = self.protected_queue.lock().unwrap();
                                
                                // For critical items, don't demote them if the candidate is not critical
                                if priority != CachePriority::Critical {
                                    // Check if the first item is critical
                                    if !protected_queue.is_empty() {
                                        let first_key = protected_queue[0];
                                        if let Some(first_entry) = self.protected.get(&first_key) {
                                            if first_entry.value().priority == CachePriority::Critical {
                                                // Try to find a non-critical item to evict
                                                if let Some(pos) = protected_queue.iter().position(|k| {
                                                    if let Some(entry) = self.protected.get(k) {
                                                        entry.value().priority != CachePriority::Critical
                                                    } else {
                                                        false
                                                    }
                                                }) {
                                                    Some(protected_queue.remove(pos))
                                                } else {
                                                    // All items are critical, use LRU
                                                    if protected_queue.is_empty() { None }
                                                    else { Some(protected_queue.remove(0)) }
                                                }
                                            } else {
                                                // First item is not critical, use it
                                                if protected_queue.is_empty() { None }
                                                else { Some(protected_queue.remove(0)) }
                                            }
                                        } else {
                                            // No entry found, use LRU
                                            if protected_queue.is_empty() { None }
                                            else { Some(protected_queue.remove(0)) }
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    // Current item is critical, use standard LRU for protected
                                    if protected_queue.is_empty() { None }
                                    else { Some(protected_queue.remove(0)) }
                                }
                            };
                            
                            if let Some(victim_key) = protected_victim {
                                // Move victim from protected to probation
                                if let Some(victim_entry) = self.protected.remove(&victim_key) {
                                    // Add victim to probation
                                    self.probation.insert(victim_key, victim_entry.value().clone());
                                    
                                    // Add victim to probation queue
                                    probation_queue.push(victim_key);
                                    
                                    // Add accessed item to protected
                                    self.protected.insert(*key, updated_entry);
                                    
                                    // Add accessed item to protected queue
                                    let mut protected_queue = self.protected_queue.lock().unwrap();
                                    protected_queue.push(*key);
                                    
                                    return true;
                                }
                            } else {
                                // Protected was unexpectedly empty, just re-add to probation
                                self.probation.insert(*key, updated_entry);
                                probation_queue.push(*key);
                                
                                return true;
                            }
                        }
                    }
                }
                false
            }
        }
    }
    
    /// Check if an entry has expired (only if TTL is enabled)
    fn is_expired(&self, created_at: SystemTime) -> bool {
        if let Some(ttl) = self.ttl {
            SystemTime::now()
                .duration_since(created_at)
                .unwrap_or_default() > ttl
        } else {
            false
        }
    }
    
    /// Add an entry to the TTL index (only if TTL is enabled)
    fn add_to_ttl_index(&self, key: BlockKey, created_at: SystemTime) {
        if self.ttl.is_some() {
            self.ttl_index.insert(created_at, key);
        }
    }
    
    /// Remove an entry from the TTL index (only if TTL is enabled)
    fn remove_from_ttl_index(&self, created_at: SystemTime) {
        if self.ttl.is_some() {
            self.ttl_index.remove(&created_at);
        }
    }
    
    /// Get effective priority for a key (defaulting to Normal if not set)
    fn get_effective_priority(&self, key: &BlockKey) -> CachePriority {
        self.priorities.get(key)
            .map(|e| *e.value())
            .unwrap_or(CachePriority::Normal)
    }
}

impl LockFreeCachePolicy for LockFreePriorityLFUPolicy {
    fn access(&self, key: &BlockKey) -> bool {
        if let Some(segment) = self.get_item_segment(key) {
            self.handle_hit(key, segment)
        } else {
            false
        }
    }
    
    fn add(&self, key: BlockKey, block: Arc<Block>) -> Option<BlockKey> {
        // Update the frequency sketch
        self.sketch.add(&key);
        
        // Check if the key already exists in the cache
        if let Some(segment) = self.get_item_segment(&key) {
            if self.handle_hit(&key, segment) {
                return None;
            }
            // If handle_hit returns false, it means the item was expired
            // We'll continue and add the new item
        }
        
        // Get the priority for this key (default to Normal)
        let priority = self.get_effective_priority(&key);
        
        // Add entry with current timestamp and priority
        let now = SystemTime::now();
        let entry = CacheEntry {
            block,
            last_accessed: now,
            created_at: now,
            priority,
        };
        
        // First, try to add to the window cache
        let window_full = self.window.len() >= self.window_size;
        
        if !window_full {
            // Simple case: window has space
            self.window.insert(key, entry);
            
            let mut queue = self.window_queue.lock().unwrap();
            queue.push(key);
            
            // Add to TTL index if TTL is enabled
            self.add_to_ttl_index(key, now);
            
            // Update priority index
            self.priorities.insert(key, priority);
            
            return None;
        }
        
        // Window is full, need to evict the oldest item from window
        let window_victim = {
            let mut queue = self.window_queue.lock().unwrap();
            if queue.is_empty() {
                None
            } else {
                // For critical items, try to find a non-critical victim
                if priority == CachePriority::Critical {
                    // Try to find a non-critical item to evict
                    if let Some(pos) = queue.iter().position(|k| {
                        self.get_effective_priority(k) != CachePriority::Critical
                    }) {
                        Some(queue.remove(pos))
                    } else {
                        // All items are critical, use LRU
                        Some(queue.remove(0))
                    }
                } else {
                    // Use normal LRU for non-critical items
                    Some(queue.remove(0))
                }
            }
        };
        
        // Add new item to window regardless of victim
        self.window.insert(key, entry);
        
        // Add new item to window queue
        {
            let mut queue = self.window_queue.lock().unwrap();
            queue.push(key);
        }
        
        // Add to TTL index if TTL is enabled
        self.add_to_ttl_index(key, now);
        
        // Update priority index
        self.priorities.insert(key, priority);
        
        // Handle the evicted item from window
        if let Some(victim_key) = window_victim {
            if let Some(victim_entry) = self.window.remove(&victim_key) {
                let victim_created_at = victim_entry.value().created_at;
                let victim_block = victim_entry.value().block.clone();
                let victim_priority = victim_entry.value().priority;
                
                // Remove from TTL index if TTL is enabled
                self.remove_from_ttl_index(victim_created_at);
                
                // Check if victim has expired
                if self.is_expired(victim_created_at) {
                    // Remove from priority index
                    self.priorities.remove(&victim_key);
                    return Some(victim_key);
                }
                
                // Try to find space in main cache (probation + protected)
                let main_cache_full = 
                    (self.probation.len() + self.protected.len()) >= 
                    (self.probation_size + self.protected_size);
                
                if !main_cache_full {
                    // Main cache has space, add to probation
                    self.probation.insert(victim_key, CacheEntry {
                        block: victim_block,
                        last_accessed: SystemTime::now(),
                        created_at: victim_created_at,
                        priority: victim_priority,
                    });
                    
                    // Add to probation queue
                    let mut queue = self.probation_queue.lock().unwrap();
                    queue.push(victim_key);
                    
                    // Add back to TTL index if TTL is enabled
                    self.add_to_ttl_index(victim_key, victim_created_at);
                    
                    return None;
                }
                
                // Main cache is full, need frequency-based admission
                // Compare frequency of window victim with LRU probation item,
                // considering priority as a multiplier
                
                // Find probation victim - look for lowest priority first
                let probation_victim = {
                    let mut queue = self.probation_queue.lock().unwrap();
                    
                    // Handle critical items specially - never evict them for non-critical items
                    if victim_priority != CachePriority::Critical {
                        // Try to find a non-critical item to evict
                        if let Some(pos) = queue.iter().position(|k| {
                            self.get_effective_priority(k) != CachePriority::Critical
                        }) {
                            Some(queue.remove(pos))
                        } else if queue.is_empty() {
                            None
                        } else {
                            // All items are critical and victim is not, evict victim
                            self.priorities.remove(&victim_key);
                            return Some(victim_key);
                        }
                    } else if queue.is_empty() {
                        None
                    } else {
                        // Use standard LRU for victim selection
                        Some(queue.remove(0))
                    }
                };
                
                if let Some(prob_victim_key) = probation_victim {
                    // Get priority of probation victim
                    let prob_victim_priority = self.get_effective_priority(&prob_victim_key);
                    
                    // Compare frequencies with priority multipliers to decide admission
                    let victim_score = self.sketch.estimate_with_priority(&victim_key, victim_priority);
                    let prob_victim_score = self.sketch.estimate_with_priority(&prob_victim_key, prob_victim_priority);
                    
                    if victim_score > prob_victim_score {
                        // Window victim has higher priority-adjusted frequency, replace probation victim
                        if let Some(prob_victim_entry) = self.probation.remove(&prob_victim_key) {
                            // Remove probation victim from TTL index if TTL is enabled
                            self.remove_from_ttl_index(prob_victim_entry.value().created_at);
                            
                            // Remove from priority index
                            self.priorities.remove(&prob_victim_key);
                            
                            // Add window victim to probation
                            self.probation.insert(victim_key, CacheEntry {
                                block: victim_block,
                                last_accessed: SystemTime::now(),
                                created_at: victim_created_at,
                                priority: victim_priority,
                            });
                            
                            // Add to probation queue
                            let mut queue = self.probation_queue.lock().unwrap();
                            queue.push(victim_key);
                            
                            // Add back to TTL index if TTL is enabled
                            self.add_to_ttl_index(victim_key, victim_created_at);
                            
                            // Return the evicted probation victim
                            return Some(prob_victim_key);
                        }
                    } else {
                        // Probation victim has higher priority-adjusted frequency, return window victim
                        // But first put probation victim back
                        {
                            let mut queue = self.probation_queue.lock().unwrap();
                            queue.push(prob_victim_key);
                        }
                        
                        // Remove from priority index
                        self.priorities.remove(&victim_key);
                        
                        return Some(victim_key);
                    }
                } else {
                    // No items in probation, try to replace from protected
                    // (unlikely but possible edge case)
                    let protected_victim = {
                        let mut queue = self.protected_queue.lock().unwrap();
                        
                        // Prioritize keeping critical items
                        if victim_priority != CachePriority::Critical {
                            // Try to find a non-critical item to evict
                            if let Some(pos) = queue.iter().position(|k| {
                                self.get_effective_priority(k) != CachePriority::Critical
                            }) {
                                Some(queue.remove(pos))
                            } else if queue.is_empty() {
                                None
                            } else {
                                // All items are critical and victim is not, evict victim
                                self.priorities.remove(&victim_key);
                                return Some(victim_key);
                            }
                        } else if queue.is_empty() {
                            None
                        } else {
                            // Use standard LRU for victim selection when all priorities match
                            Some(queue.remove(0))
                        }
                    };
                    
                    if let Some(prot_victim_key) = protected_victim {
                        // Get priority of protected victim
                        let prot_victim_priority = self.get_effective_priority(&prot_victim_key);
                        
                        // Compare frequencies with priority multiplier
                        let victim_score = self.sketch.estimate_with_priority(&victim_key, victim_priority);
                        let prot_victim_score = self.sketch.estimate_with_priority(&prot_victim_key, prot_victim_priority);
                        
                        if victim_score > prot_victim_score {
                            // Window victim has higher priority-adjusted frequency
                            if let Some(prot_victim_entry) = self.protected.remove(&prot_victim_key) {
                                let prot_victim_block = prot_victim_entry.value().block.clone();
                                let prot_victim_created_at = prot_victim_entry.value().created_at;
                                let prot_victim_priority = prot_victim_entry.value().priority;
                                
                                // Remove protected victim from TTL index if TTL is enabled
                                self.remove_from_ttl_index(prot_victim_created_at);
                                
                                // Add window victim to protected
                                self.protected.insert(victim_key, CacheEntry {
                                    block: victim_block,
                                    last_accessed: SystemTime::now(),
                                    created_at: victim_created_at,
                                    priority: victim_priority,
                                });
                                
                                // Add window victim to protected queue
                                let mut protected_queue = self.protected_queue.lock().unwrap();
                                protected_queue.push(victim_key);
                                
                                // Add back to TTL index if TTL is enabled
                                self.add_to_ttl_index(victim_key, victim_created_at);
                                
                                // Move protected victim to probation
                                self.probation.insert(prot_victim_key, CacheEntry {
                                    block: prot_victim_block,
                                    last_accessed: SystemTime::now(),
                                    created_at: prot_victim_created_at,
                                    priority: prot_victim_priority,
                                });
                                
                                // Add protected victim to probation queue
                                let mut probation_queue = self.probation_queue.lock().unwrap();
                                probation_queue.push(prot_victim_key);
                                
                                // Add back to TTL index if TTL is enabled
                                self.add_to_ttl_index(prot_victim_key, prot_victim_created_at);
                                
                                return None;
                            }
                        } else {
                            // Protected victim has higher priority-adjusted frequency
                            // Put it back and evict window victim
                            {
                                let mut queue = self.protected_queue.lock().unwrap();
                                queue.push(prot_victim_key);
                            }
                            
                            // Remove from priority index
                            self.priorities.remove(&victim_key);
                            
                            return Some(victim_key);
                        }
                    }
                }
                
                // If we reached here, we're evicting the window victim
                // Remove from priority index
                self.priorities.remove(&victim_key);
                
                return Some(victim_key);
            }
        }
        
        None
    }
    
    fn remove(&self, key: &BlockKey) -> Option<Arc<Block>> {
        // Remove from priority index
        self.priorities.remove(key);
        
        // Check each segment and remove if found
        if let Some(entry) = self.window.remove(key) {
            // Remove from window queue
            let mut queue = self.window_queue.lock().unwrap();
            if let Some(pos) = queue.iter().position(|k| k == key) {
                queue.remove(pos);
            }
            
            // Remove from TTL index if TTL is enabled
            self.remove_from_ttl_index(entry.value().created_at);
            
            return Some(entry.value().block.clone());
        }
        
        if let Some(entry) = self.protected.remove(key) {
            // Remove from protected queue
            let mut queue = self.protected_queue.lock().unwrap();
            if let Some(pos) = queue.iter().position(|k| k == key) {
                queue.remove(pos);
            }
            
            // Remove from TTL index if TTL is enabled
            self.remove_from_ttl_index(entry.value().created_at);
            
            return Some(entry.value().block.clone());
        }
        
        if let Some(entry) = self.probation.remove(key) {
            // Remove from probation queue
            let mut queue = self.probation_queue.lock().unwrap();
            if let Some(pos) = queue.iter().position(|k| k == key) {
                queue.remove(pos);
            }
            
            // Remove from TTL index if TTL is enabled
            self.remove_from_ttl_index(entry.value().created_at);
            
            return Some(entry.value().block.clone());
        }
        
        None
    }
    
    fn get(&self, key: &BlockKey) -> Option<Arc<Block>> {
        // Check each segment
        if let Some(entry) = self.window.get(key) {
            // Check if expired first (if TTL is enabled)
            if self.is_expired(entry.value().created_at) {
                // Item has expired - remove it
                self.remove(key);
                return None;
            }
            
            let block = entry.value().block.clone();
            self.handle_hit(key, Segment::Window);
            return Some(block);
        }
        
        if let Some(entry) = self.protected.get(key) {
            // Check if expired first (if TTL is enabled)
            if self.is_expired(entry.value().created_at) {
                // Item has expired - remove it
                self.remove(key);
                return None;
            }
            
            let block = entry.value().block.clone();
            self.handle_hit(key, Segment::Protected);
            return Some(block);
        }
        
        if let Some(entry) = self.probation.get(key) {
            // Check if expired first (if TTL is enabled)
            if self.is_expired(entry.value().created_at) {
                // Item has expired - remove it
                self.remove(key);
                return None;
            }
            
            let block = entry.value().block.clone();
            self.handle_hit(key, Segment::Probation);
            return Some(block);
        }
        
        None
    }
    
    fn contains(&self, key: &BlockKey) -> bool {
        // Check presence and not expired
        if let Some(segment) = self.get_item_segment(key) {
            // Get entry to check expiry (if TTL is enabled)
            if self.ttl.is_some() {
                let created_at = match segment {
                    Segment::Window => self.window.get(key).map(|e| e.value().created_at),
                    Segment::Protected => self.protected.get(key).map(|e| e.value().created_at),
                    Segment::Probation => self.probation.get(key).map(|e| e.value().created_at),
                };
                
                // If we got created_at, check if expired
                if let Some(created_at) = created_at {
                    return !self.is_expired(created_at);
                }
                false
            } else {
                true
            }
        } else {
            false
        }
    }
    
    fn clear(&self) {
        // Clear the priority index first
        let priority_keys: Vec<BlockKey> = self.priorities.iter()
            .map(|entry| *entry.key())
            .collect();
            
        for key in priority_keys {
            self.priorities.remove(&key);
        }
        
        // Clear the TTL index (if TTL is enabled)
        if self.ttl.is_some() {
            let ttl_keys: Vec<SystemTime> = self.ttl_index.iter()
                .map(|entry| *entry.key())
                .collect();
                
            for key in ttl_keys {
                self.ttl_index.remove(&key);
            }
        }
        
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
        Box::new(if let Some(ttl) = self.ttl {
            LockFreePriorityLFUPolicy::new_with_ttl(self.capacity, ttl)
        } else {
            LockFreePriorityLFUPolicy::new(self.capacity)
        })
    }
    
    fn scan_expired(&self, ttl: Duration) -> Vec<BlockKey> {
        // If TTL isn't enabled in the policy, use the provided TTL
        let effective_ttl = self.ttl.unwrap_or(ttl);
        let now = SystemTime::now();
        let mut expired_keys = Vec::new();
        
        if self.ttl.is_some() {
            // Use the TTL index to efficiently find expired items
            for entry in self.ttl_index.iter() {
                let created_at = *entry.key();
                let key = *entry.value();
                
                if now.duration_since(created_at).unwrap_or_default() > effective_ttl {
                    expired_keys.push(key);
                } else {
                    // Since entries are stored in time order, we can stop once we reach non-expired entries
                    break;
                }
            }
        } else {
            // Scan all segments for expired items - less efficient but works without TTL index
            // Window
            for entry in self.window.iter() {
                let key = *entry.key();
                let created_at = entry.value().created_at;
                
                if now.duration_since(created_at).unwrap_or_default() > effective_ttl {
                    expired_keys.push(key);
                }
            }
            
            // Protected
            for entry in self.protected.iter() {
                let key = *entry.key();
                let created_at = entry.value().created_at;
                
                if now.duration_since(created_at).unwrap_or_default() > effective_ttl {
                    expired_keys.push(key);
                }
            }
            
            // Probation
            for entry in self.probation.iter() {
                let key = *entry.key();
                let created_at = entry.value().created_at;
                
                if now.duration_since(created_at).unwrap_or_default() > effective_ttl {
                    expired_keys.push(key);
                }
            }
        }
        
        expired_keys
    }
    
    fn remove_expired(&self, ttl: Duration) -> usize {
        // Get all expired keys
        let expired_keys = self.scan_expired(ttl);
        let count = expired_keys.len();
        
        // Remove each expired key
        for key in expired_keys {
            self.remove(&key);
        }
        
        count
    }
    
    fn set_priority(&self, key: &BlockKey, priority: CachePriority) -> bool {
        // Update the priority in the index
        self.priorities.insert(*key, priority);
        
        // Find the key and update the entry's priority
        let found_and_updated = if let Some(entry) = self.window.get(key) {
            let mut updated_entry = entry.value().clone();
            updated_entry.priority = priority;
            self.window.insert(*key, updated_entry);
            true
        } else if let Some(entry) = self.protected.get(key) {
            let mut updated_entry = entry.value().clone();
            updated_entry.priority = priority;
            self.protected.insert(*key, updated_entry);
            true
        } else if let Some(entry) = self.probation.get(key) {
            let mut updated_entry = entry.value().clone();
            updated_entry.priority = priority;
            self.probation.insert(*key, updated_entry);
            true
        } else {
            // Not found in any segment, just store the priority for when/if the key is added
            false
        };
        
        found_and_updated
    }
    
    fn get_priority(&self, key: &BlockKey) -> Option<CachePriority> {
        // First check the priority index
        if let Some(entry) = self.priorities.get(key) {
            return Some(*entry.value());
        }
        
        // If not in the index, try to find the entry directly
        if let Some(entry) = self.window.get(key) {
            Some(entry.value().priority)
        } else if let Some(entry) = self.protected.get(key) {
            Some(entry.value().priority)
        } else if let Some(entry) = self.probation.get(key) {
            Some(entry.value().priority)
        } else {
            None
        }
    }
}