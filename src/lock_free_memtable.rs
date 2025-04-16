use crate::types::{Error, Key, Result, Value};
use crossbeam_skiplist::SkipMap;
use std::mem;
use std::ops::Bound;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Lock-free memtable implementation using Crossbeam's SkipMap
pub struct LockFreeMemtable {
    /// Data store using lock-free skip list
    data: SkipMap<Key, Value>,
    /// Current number of entries in the memtable
    current_size: AtomicUsize,
    /// Maximum number of entries allowed
    max_size: usize,
    /// Size of a single entry in bytes
    entry_size: usize,
    /// Key range tracking (using atomics)
    min_key: Arc<AtomicKey>,
    max_key: Arc<AtomicKey>,
}

/// Atomic wrapper for Key
#[derive(Debug)]
struct AtomicKey {
    /// The key value
    value: AtomicUsize,
    /// Whether the key has been set (avoids using Option)
    is_set: AtomicUsize,
}

impl AtomicKey {
    /// Create a new unset atomic key
    fn new() -> Self {
        Self {
            value: AtomicUsize::new(0),
            is_set: AtomicUsize::new(0),
        }
    }

    /// Set the key value if it's smaller than the current value
    fn set_min(&self, key: Key) {
        if self.is_set.load(Ordering::Relaxed) == 0 {
            // First time setting the key
            self.value.store(key as usize, Ordering::Relaxed);
            self.is_set.store(1, Ordering::Relaxed);
        } else {
            // Update minimum if smaller
            let mut current = self.value.load(Ordering::Relaxed) as Key;
            while key < current {
                match self.value.compare_exchange(
                    current as usize,
                    key as usize,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(new_val) => current = new_val as Key,
                }
            }
        }
    }

    /// Set the key value if it's larger than the current value
    fn set_max(&self, key: Key) {
        if self.is_set.load(Ordering::Relaxed) == 0 {
            // First time setting the key
            self.value.store(key as usize, Ordering::Relaxed);
            self.is_set.store(1, Ordering::Relaxed);
        } else {
            // Update maximum if larger
            let mut current = self.value.load(Ordering::Relaxed) as Key;
            while key > current {
                match self.value.compare_exchange(
                    current as usize,
                    key as usize,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(new_val) => current = new_val as Key,
                }
            }
        }
    }

    /// Get the current key value if set
    fn get(&self) -> Option<Key> {
        if self.is_set.load(Ordering::Relaxed) == 1 {
            Some(self.value.load(Ordering::Relaxed) as Key)
        } else {
            None
        }
    }

    /// Reset the key to unset state
    fn reset(&self) {
        self.is_set.store(0, Ordering::Relaxed);
    }
}

impl LockFreeMemtable {
    pub fn new(num_pages: usize) -> Self {
        let page_size = page_size::get();
        let entry_size = mem::size_of::<(Key, Value)>();
        let max_pairs = (num_pages * page_size) / entry_size;

        Self {
            data: SkipMap::new(),
            current_size: AtomicUsize::new(0),
            max_size: max_pairs,
            entry_size,
            min_key: Arc::new(AtomicKey::new()),
            max_key: Arc::new(AtomicKey::new()),
        }
    }

    pub fn put(&self, key: Key, value: Value) -> Result<Option<Value>> {
        // Check if this is an update (key already exists)
        let is_update = self.data.contains_key(&key);

        // Check if buffer is full
        if !is_update && self.current_size.load(Ordering::Acquire) >= self.max_size {
            return Err(Error::BufferFull);
        }

        // Update key range
        if !is_update {
            self.min_key.set_min(key);
            self.max_key.set_max(key);
        }

        // Insert the value
        let previous = self.data.get(&key).map(|e| *e.value());
        self.data.insert(key, value);

        // Update size only if this is a new key
        if !is_update {
            self.current_size.fetch_add(1, Ordering::Release);
        }

        Ok(previous)
    }

    pub fn get(&self, key: &Key) -> Option<Value> {
        // Quick range check first
        if let Some(min_key) = self.min_key.get() {
            if key < &min_key {
                return None;
            }
        }
        if let Some(max_key) = self.max_key.get() {
            if key > &max_key {
                return None;
            }
        }

        // Get the value from the skip list
        self.data.get(key).map(|entry| *entry.value())
    }

    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        if start >= end {
            return Vec::new();
        }

        // Get all key-value pairs in the range
        self.data
            .range((Bound::Included(start), Bound::Excluded(end)))
            .map(|entry| (*entry.key(), *entry.value()))
            .collect()
    }

    pub fn clear(&self) {
        // Clear the skip list by removing each key individually
        let keys: Vec<Key> = self.data.iter().map(|entry| *entry.key()).collect();
        for key in keys {
            self.data.remove(&key);
        }

        // Reset key range
        self.min_key.reset();
        self.max_key.reset();

        // Reset size counter
        self.current_size.store(0, Ordering::Release);
    }

    pub fn len(&self) -> usize {
        self.current_size.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_full(&self) -> bool {
        self.len() >= self.max_size
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }

    pub fn memory_usage(&self) -> MemoryStats {
        let page_size = page_size::get();
        MemoryStats {
            total_pages: self.max_size * self.entry_size / page_size,
            used_bytes: self.len() * self.entry_size,
            total_bytes: self.max_size * self.entry_size,
            fragmentation: 0.0,
        }
    }

    pub fn key_range(&self) -> Option<(Key, Key)> {
        match (self.min_key.get(), self.max_key.get()) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        }
    }

    pub fn take_all(&self) -> Vec<(Key, Value)> {
        self.iter()
    }

    pub fn iter(&self) -> Vec<(Key, Value)> {
        // Collect all entries and sort by key
        let mut result: Vec<(Key, Value)> = self.data
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();

        result.sort_by_key(|&(k, _)| k);
        result
    }
}

#[derive(Debug)]
pub struct MemoryStats {
    pub total_pages: usize,
    pub used_bytes: usize,
    pub total_bytes: usize,
    pub fragmentation: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_memtable_operations() {
        let table = LockFreeMemtable::new(1);

        assert!(table.put(1, 100).unwrap().is_none());
        assert_eq!(table.get(&1), Some(100));

        assert_eq!(table.put(1, 200).unwrap(), Some(100));
        assert_eq!(table.get(&1), Some(200));

        assert!(table.put(2, 300).unwrap().is_none());
        assert!(table.put(3, 400).unwrap().is_none());

        let range = table.range(1, 3);
        assert_eq!(range.len(), 2);
        assert_eq!(range[0], (1, 200));
        assert_eq!(range[1], (2, 300));

        assert_eq!(table.key_range(), Some((1, 3)));
        assert_eq!(table.len(), 3);

        table.clear();
        assert!(table.is_empty());
        assert_eq!(table.key_range(), None);
    }

    #[test]
    fn test_size_limits() {
        let table = LockFreeMemtable::new(1);
        let max_size = table.max_size();

        for i in 0..max_size {
            assert!(table.put(i as Key, i as Value).is_ok());
        }

        assert!(table.is_full());
        assert!(matches!(
            table.put(max_size as Key, 0),
            Err(Error::BufferFull)
        ));

        // Updates should still work
        assert!(table.put(0, 100).is_ok());
    }

    #[test]
    fn test_concurrent_updates() {
        let table = Arc::new(LockFreeMemtable::new(1));
        let table1 = table.clone();
        let table2 = table.clone();

        // Test concurrent updates to same key
        let t1 = thread::spawn(move || {
            for i in 0..100 {
                table1.put(1, i).unwrap();
            }
        });

        let t2 = thread::spawn(move || {
            for i in 100..200 {
                table2.put(1, i).unwrap();
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Value should be from one of the threads
        let final_value = table.get(&1).unwrap();
        assert!(final_value >= 0 && final_value < 200);
    }

    #[test]
    fn test_concurrent_reads_and_writes() {
        let table = Arc::new(LockFreeMemtable::new(1));
        let table1 = table.clone();
        let table2 = table.clone();

        // Insert initial data
        table.put(1, 100).unwrap();

        let t1 = thread::spawn(move || {
            // Reader thread
            for _ in 0..100 {
                assert_eq!(table1.get(&1), Some(100));
            }
        });

        let t2 = thread::spawn(move || {
            // Writer thread
            table2.put(2, 200).unwrap();
        });

        t1.join().unwrap();
        t2.join().unwrap();

        assert_eq!(table.get(&1), Some(100));
        assert_eq!(table.get(&2), Some(200));
    }

    #[test]
    fn test_concurrent_inserts() {
        let table = Arc::new(LockFreeMemtable::new(10)); // Larger size for test
        let thread_count = 10;
        let inserts_per_thread = 10;
        
        let mut handles = vec![];
        
        for i in 0..thread_count {
            let t = table.clone();
            let handle = thread::spawn(move || {
                let base = i * inserts_per_thread;
                for j in 0..inserts_per_thread {
                    let k = (base + j) as Key;
                    t.put(k, k * 10).unwrap();
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all entries were inserted
        assert_eq!(table.len(), thread_count * inserts_per_thread);
        
        // Check a few random entries
        assert_eq!(table.get(&5), Some(50));
        assert_eq!(table.get(&27), Some(270));
        assert_eq!(table.get(&99), Some(990));
    }
}