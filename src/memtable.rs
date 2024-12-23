use std::mem;
use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::types::{Key, Value, Result, Error};

pub struct Memtable {
    data: BTreeMap<Key, Value>,
    current_size: AtomicUsize,
    max_size: usize,
    min_key: Option<Key>,
    max_key: Option<Key>,
    entry_size: usize,
}

impl Memtable {
    pub fn new(num_pages: usize) -> Self {
        let page_size = page_size::get();
        let entry_size = mem::size_of::<(Key, Value)>();
        let max_pairs = (num_pages * page_size) / entry_size;

        Self {
            data: BTreeMap::new(),
            current_size: AtomicUsize::new(0),
            max_size: max_pairs,
            min_key: None,
            max_key: None,
            entry_size,
        }
    }

    pub fn put(&mut self, key: Key, value: Value) -> Result<Option<Value>> {
        //println!("Before put: min_key={:?}, max_key={:?}, current_size={}", self.min_key, self.max_key, self.len());

        let is_update = self.data.contains_key(&key);

        if !is_update && self.current_size.load(Ordering::Relaxed) >= self.max_size {
            return Err(Error::BufferFull);
        }

        if !is_update {
            self.min_key = Some(self.min_key.map_or(key, |min| std::cmp::min(min, key)));
            self.max_key = Some(self.max_key.map_or(key, |max| std::cmp::max(max, key)));
        }

        let previous = self.data.insert(key, value);
        if previous.is_none() {
            self.current_size.fetch_add(1, Ordering::Relaxed);
        }

        //println!("After put: min_key={:?}, max_key={:?}, current_size={}", self.min_key, self.max_key, self.len());
        Ok(previous)
    }

    pub fn get(&self, key: &Key) -> Option<Value> {
        // Quick range check
        if let Some(min_key) = self.min_key {
            if key < &min_key {
                return None;
            }
        }
        if let Some(max_key) = self.max_key {
            if key > &max_key {
                return None;
            }
        }

        self.data.get(key).copied()
    }

    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        if start >= end {
            return Vec::new();
        }

        self.data
            .range((Bound::Included(start), Bound::Excluded(end)))
            .map(|(&k, &v)| (k, v))
            .collect()
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.current_size.store(0, Ordering::Relaxed);
        self.min_key = None;
        self.max_key = None;
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

    pub fn as_map(&self) -> &BTreeMap<Key, Value> {
        &self.data
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Key, &Value)> {
        self.data.iter()
    }

    pub fn key_range(&self) -> Option<(Key, Key)> {
        match (self.min_key, self.max_key) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        }
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

    #[test]
    fn test_memtable_operations() {
        let mut table = Memtable::new(1);

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
        let mut table = Memtable::new(1);
        let max_size = table.max_size();

        for i in 0..max_size {
            assert!(table.put(i as Key, i as Value).is_ok());
        }

        assert!(table.is_full());
        assert!(matches!(table.put(max_size as Key, 0), Err(Error::BufferFull)));

        assert!(table.put(0, 100).is_ok());
    }

    #[test]
    fn test_range_queries() {
        let mut table = Memtable::new(1);

        for i in 0..10 {
            assert!(table.put(i, i * 10).is_ok());
        }

        assert_eq!(table.range(-1, 1).len(), 1);
        assert_eq!(table.range(0, 5).len(), 5);
        assert!(table.range(100, 200).is_empty());

        let range = table.range(8, 15);
        assert_eq!(range.len(), 2);
        assert_eq!(range[0], (8, 80));
        assert_eq!(range[1], (9, 90));
    }

    // Test for edge cases
    #[test]
    fn test_edge_cases() {
        let mut table = Memtable::new(1);

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
        let mut table = Memtable::new(1);

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
        let mut table = Memtable::new(1);

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
        let mut table = Memtable::new(1);

        // Empty table iteration
        assert_eq!(table.iter().count(), 0);

        // Add test data
        let test_data = vec![(1, 100), (2, 200), (3, 300)];
        for (k, v) in &test_data {
            table.put(*k, *v).unwrap();
        }

        // Test iterator order and completeness
        let collected: Vec<_> = table.iter().collect();
        assert_eq!(collected.len(), test_data.len());

        for (i, (k, v)) in collected.iter().enumerate() {
            assert_eq!(**k, test_data[i].0);
            assert_eq!(**v, test_data[i].1);
        }

        // Test that iterator reflects sorted order
        let mut last_key = i64::MIN;
        for (k, _) in table.iter() {
            assert!(*k > last_key);
            last_key = *k;
        }
    }
}