use crate::types::{Error, Key, Result, Value};
use std::collections::BTreeMap;
use std::mem;
use std::ops::Bound;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

/// Number of shards for the memtable
/// This value is chosen as a power of 2 for efficient modulo operations
const SHARD_COUNT: usize = 16;

#[derive(Debug, Default)]
struct KeyRange {
    min_key: Option<Key>,
    max_key: Option<Key>,
}

/// A single shard of the memtable
struct MemtableShard {
    data: RwLock<BTreeMap<Key, Value>>,
    key_range: RwLock<KeyRange>,
}

impl MemtableShard {
    fn new() -> Self {
        Self {
            data: RwLock::new(BTreeMap::new()),
            key_range: RwLock::new(KeyRange::default()),
        }
    }

    fn put(&self, key: Key, value: Value) -> Option<Value> {
        // Update key range if this is a new key
        let is_new_key = {
            let data = self.data.read().unwrap();
            !data.contains_key(&key)
        };

        if is_new_key {
            let mut key_range = self.key_range.write().unwrap();
            key_range.min_key = Some(key_range.min_key.map_or(key, |min| std::cmp::min(min, key)));
            key_range.max_key = Some(key_range.max_key.map_or(key, |max| std::cmp::max(max, key)));
        }

        // Insert the new value
        let mut data = self.data.write().unwrap();
        data.insert(key, value)
    }

    fn get(&self, key: &Key) -> Option<Value> {
        // Quick range check
        {
            let key_range = self.key_range.read().unwrap();
            if let Some(min_key) = key_range.min_key {
                if key < &min_key {
                    return None;
                }
            }
            if let Some(max_key) = key_range.max_key {
                if key > &max_key {
                    return None;
                }
            }
        }

        let data = self.data.read().unwrap();
        data.get(key).copied()
    }

    fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        if start >= end {
            return Vec::new();
        }

        let data = self.data.read().unwrap();
        data.range((Bound::Included(start), Bound::Excluded(end)))
            .map(|(&k, &v)| (k, v))
            .collect()
    }

    fn clear(&self) {
        {
            let mut data = self.data.write().unwrap();
            data.clear();
        }
        *self.key_range.write().unwrap() = KeyRange::default();
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn key_range(&self) -> Option<(Key, Key)> {
        let key_range = self.key_range.read().unwrap();
        match (key_range.min_key, key_range.max_key) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        }
    }

    fn take_all(&self) -> Vec<(Key, Value)> {
        let data = self.data.read().unwrap();
        data.iter().map(|(&k, &v)| (k, v)).collect()
    }
}

pub struct Memtable {
    shards: Vec<MemtableShard>,
    current_size: AtomicUsize,
    max_size: usize,
    entry_size: usize,
}

impl Memtable {
    pub fn new(num_pages: usize) -> Self {
        let page_size = page_size::get();
        let entry_size = mem::size_of::<(Key, Value)>();
        let max_pairs = (num_pages * page_size) / entry_size;

        // Create shards
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(MemtableShard::new());
        }

        Self {
            shards,
            current_size: AtomicUsize::new(0),
            max_size: max_pairs,
            entry_size,
        }
    }

    /// Determine which shard a key belongs to
    #[inline]
    fn shard_for_key(&self, key: &Key) -> &MemtableShard {
        // Simple hash function: use the key modulo shard count
        // Since SHARD_COUNT is a power of 2, we can use a bitwise AND for faster modulo
        let shard_index = (key.wrapping_abs() as usize) & (SHARD_COUNT - 1);
        &self.shards[shard_index]
    }

    pub fn put(&self, key: Key, value: Value) -> Result<Option<Value>> {
        // Check if this is an update
        let shard = self.shard_for_key(&key);
        let is_update = {
            let data = shard.data.read().unwrap();
            data.contains_key(&key)
        };

        if !is_update && self.current_size.load(Ordering::Acquire) >= self.max_size {
            return Err(Error::BufferFull);
        }

        // Insert the value into the appropriate shard
        let previous = shard.put(key, value);

        // Update size only if this is a new key
        if previous.is_none() {
            self.current_size.fetch_add(1, Ordering::Release);
        }

        Ok(previous)
    }

    pub fn get(&self, key: &Key) -> Option<Value> {
        // Query the appropriate shard
        self.shard_for_key(key).get(key)
    }

    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        if start >= end {
            return Vec::new();
        }

        // Collect results from all shards (since a range can span multiple shards)
        let mut results = Vec::new();
        for shard in &self.shards {
            results.extend(shard.range(start, end));
        }

        // Sort results by key since they came from different shards
        results.sort_by_key(|&(k, _)| k);
        results
    }

    pub fn clear(&self) {
        for shard in &self.shards {
            shard.clear();
        }
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
        // Compute the overall key range by merging ranges from all shards
        let mut min_key = None;
        let mut max_key = None;

        for shard in &self.shards {
            if let Some((shard_min, shard_max)) = shard.key_range() {
                min_key = Some(min_key.map_or(shard_min, |min| std::cmp::min(min, shard_min)));
                max_key = Some(max_key.map_or(shard_max, |max| std::cmp::max(max, shard_max)));
            }
        }

        match (min_key, max_key) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        }
    }

    pub fn take_all(&self) -> Vec<(Key, Value)> {
        // Collect data from all shards
        let mut result = Vec::new();
        for shard in &self.shards {
            result.extend(shard.take_all());
        }

        // Return sorted by key
        result.sort_by_key(|&(k, _)| k);
        result
    }

    pub fn iter(&self) -> Vec<(Key, Value)> {
        self.take_all()
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
        let table = Memtable::new(1);

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
        let table = Memtable::new(1);
        let max_size = table.max_size();

        for i in 0..max_size {
            assert!(table.put(i as Key, i as Value).is_ok());
        }

        assert!(table.is_full());
        assert!(matches!(
            table.put(max_size as Key, 0),
            Err(Error::BufferFull)
        ));

        assert!(table.put(0, 100).is_ok());
    }

    #[test]
    fn test_range_queries() {
        let table = Memtable::new(1);

        for i in 0..10 {
            assert!(table.put(i, i * 10).is_ok());
        }

        assert_eq!(table.range(-1, 1).len(), 1);
        
        // Sort check: range queries should return sorted results
        let range_results = table.range(0, 5);
        assert_eq!(range_results.len(), 5);
        for i in 0..4 {
            assert!(range_results[i].0 < range_results[i+1].0);
        }
        
        assert!(table.range(100, 200).is_empty());

        let range = table.range(8, 15);
        assert_eq!(range.len(), 2);
        assert_eq!(range[0], (8, 80));
        assert_eq!(range[1], (9, 90));
    }

    // Test for edge cases
    #[test]
    fn test_edge_cases() {
        let table = Memtable::new(1);

        // Insert edge values
        assert!(table.put(i64::MIN, 100).is_ok());
        assert!(table.put(i64::MAX, 200).is_ok());

        // Retrieve edge values
        assert_eq!(table.get(&i64::MIN), Some(100));
        assert_eq!(table.get(&i64::MAX), Some(200));

        // Test ranges near boundaries
        assert_eq!(table.range(i64::MIN, i64::MIN + 1), vec![(i64::MIN, 100)]);
        assert_eq!(table.range(i64::MAX - 1, i64::MAX), vec![]);

        // Test invalid ranges
        assert!(table.range(0, 0).is_empty());
        assert!(table.range(5, 4).is_empty());

        // Update edge values
        assert_eq!(table.put(i64::MIN, 150).unwrap(), Some(100));
        assert_eq!(table.get(&i64::MIN), Some(150));
    }

    // New test for min/max key tracking
    #[test]
    fn test_min_max_tracking() {
        let table = Memtable::new(1);

        // Empty table should have no range
        assert_eq!(table.key_range(), None);

        // Test single element
        table.put(5, 500).unwrap();
        assert_eq!(table.key_range(), Some((5, 5)));

        // Test adding smaller key
        table.put(3, 300).unwrap();
        assert_eq!(table.key_range(), Some((3, 5)));

        // Test adding larger key
        table.put(7, 700).unwrap();
        assert_eq!(table.key_range(), Some((3, 7)));

        // Test that updates don't affect range
        table.put(5, 550).unwrap();
        assert_eq!(table.key_range(), Some((3, 7)));

        // Test clearing
        table.clear();
        assert_eq!(table.key_range(), None);
    }

    // New test for memory statistics
    #[test]
    fn test_memory_stats() {
        let table = Memtable::new(1);

        // Check initial state
        let initial_stats = table.memory_usage();
        assert_eq!(initial_stats.used_bytes, 0);
        assert!(initial_stats.total_bytes > 0);

        // Add some entries
        table.put(1, 100).unwrap();
        table.put(2, 200).unwrap();

        let stats = table.memory_usage();
        assert_eq!(stats.used_bytes, 2 * mem::size_of::<(Key, Value)>());
        assert_eq!(stats.total_bytes, initial_stats.total_bytes);

        // Check stats after clear
        table.clear();
        let final_stats = table.memory_usage();
        assert_eq!(final_stats.used_bytes, 0);
        assert_eq!(final_stats.total_bytes, initial_stats.total_bytes);
    }

    // New test for iterator behavior
    #[test]
    fn test_iterator_behavior() {
        let table = Memtable::new(1);

        // Empty table iteration
        assert_eq!(table.iter().len(), 0);

        // Add test data
        let test_data = vec![(1, 100), (2, 200), (3, 300)];
        for (k, v) in &test_data {
            table.put(*k, *v).unwrap();
        }

        // Test iterator order and completeness
        let collected: Vec<_> = table.iter();
        assert_eq!(collected.len(), test_data.len());

        for (i, (k, v)) in collected.iter().enumerate() {
            assert_eq!(*k, test_data[i].0);
            assert_eq!(*v, test_data[i].1);
        }

        // Test that iterator reflects sorted order
        let mut last_key = i64::MIN;
        for (k, _) in table.iter() {
            assert!(k > last_key);
            last_key = k;
        }
    }

    #[test]
    fn test_thread_safe_key_range() {
        let table = Arc::new(Memtable::new(1));
        let table1 = table.clone(); // Clone for first thread
        let table2 = table.clone(); // Clone for second thread

        let t1 = thread::spawn(move || {
            table1.put(1, 100).unwrap();
        });

        let t2 = thread::spawn(move || {
            table2.put(2, 200).unwrap();
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let range = table.key_range(); // Use original Arc<Memtable>
        assert!(range.is_some());
        let (min, max) = range.unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 2);
    }

    #[test]
    fn test_concurrent_updates() {
        let table = Arc::new(Memtable::new(1));
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
        let table = Arc::new(Memtable::new(1));
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
    fn test_concurrent_range_queries() {
        let table = Arc::new(Memtable::new(1));
        let table1 = table.clone();
        let table2 = table.clone();

        // Insert test data
        for i in 0..5 {
            table.put(i, i * 10).unwrap();
        }

        let t1 = thread::spawn(move || {
            // Range query thread
            let range = table1.range(1, 4);
            assert_eq!(range.len(), 3); // Correct: includes 1,2,3
        });

        let t2 = thread::spawn(move || {
            // Modification thread
            table2.put(6, 60).unwrap();
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let final_range = table.range(1, 7);
        assert_eq!(final_range.len(), 5);
    }

    #[test]
    fn test_size_management_concurrent() {
        let table = Arc::new(Memtable::new(1)); // Small size to test overflow
        let table1 = table.clone();
        let table2 = table.clone();

        let t1 = thread::spawn(move || {
            // Fill the buffer
            let mut count = 0;
            while table1.put(count, count).is_ok() {
                count += 1;
            }
        });

        let t2 = thread::spawn(move || {
            // Try to insert while buffer is being filled
            let result = table2.put(999, 999);
            // Should either succeed or get BufferFull error
            match result {
                Ok(_) => (),
                Err(Error::BufferFull) => (),
                _ => panic!("Unexpected error"),
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    }
    
    #[test]
    fn test_shard_distribution() {
        let table = Memtable::new(10); // Use larger size to avoid BufferFull
        
        // Add keys to test distribution, up to buffer capacity
        let mut count = 0;
        for i in 0..1000 {
            if table.put(i, i).is_err() {
                // Stop if buffer becomes full
                break;
            }
            count += 1;
        }
        
        // Skip test if we couldn't add enough items for a meaningful test
        if count < SHARD_COUNT * 2 {
            println!("Skipping test_shard_distribution - not enough data points ({} added)", count);
            return;
        }
        
        // Check that data is distributed across shards
        let mut non_empty_shards = 0;
        for shard in &table.shards {
            if !shard.is_empty() {
                non_empty_shards += 1;
            }
        }
        
        // At least half of the shards should have data
        assert!(non_empty_shards >= SHARD_COUNT / 2);
    }
    
    #[test]
    fn test_concurrent_shard_access() {
        // Use a much larger memtable for this test
        let table = Arc::new(Memtable::new(1000));
        
        // Use fewer threads and keys per thread to avoid buffer full errors
        let num_threads = 10;
        let keys_per_thread = 10;
        
        let handles: Vec<_> = (0..num_threads).map(|i| {
            let table_clone = table.clone();
            thread::spawn(move || {
                // Each thread inserts a different set of keys
                // This ensures operations hit different shards
                let mut inserted = 0;
                for j in 0..keys_per_thread {
                    let key = i * 1000 + j;
                    match table_clone.put(key, key * 10) {
                        Ok(_) => inserted += 1,
                        Err(Error::BufferFull) => break, // Stop if buffer full
                        Err(e) => panic!("Unexpected error: {:?}", e),
                    }
                }
                
                // Then read back the values that were inserted
                for j in 0..inserted {
                    let key = i * 1000 + j;
                    assert_eq!(table_clone.get(&key), Some(key * 10));
                }
                
                // Return how many we inserted
                inserted
            })
        }).collect();
        
        // Wait for all threads to complete and count total insertions
        let mut total_inserted = 0;
        for handle in handles {
            total_inserted += handle.join().unwrap();
        }
        
        // Verify total size matches what threads reported
        assert_eq!(table.len(), total_inserted as usize);
    }
}
