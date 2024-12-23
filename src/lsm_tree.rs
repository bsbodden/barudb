use std::sync::{Arc, RwLock};
use crate::memtable::Memtable;
use crate::types::{Key, Value, Result, TOMBSTONE};
use crate::run::Run;
use crate::level::Level;

pub struct LSMTree {
    buffer: Arc<RwLock<Memtable>>,
    levels: Vec<Level>,
}

impl LSMTree {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(Memtable::new(buffer_size))),
            levels: Vec::new(),
        }
    }

    pub fn put(&mut self, key: Key, value: Value) -> Result<()> {
        let flush_required = {
            let mut buffer = self.buffer.write().unwrap();
            let result = buffer.put(key, value);
            result.is_ok() && buffer.is_full()
        };

        if flush_required {
            self.flush_buffer_to_level0()?;
        }
        Ok(())
    }

    pub fn get(&self, key: Key) -> Option<Value> {
        // Check buffer first
        if let Some(value) = self.buffer.read().unwrap().get(&key) {
            if value == TOMBSTONE {
                return None; // Ignore tombstone values
            }
            return Some(value);
        }

        // Check levels
        for level in &self.levels {
            if let Some(value) = level.get(key) {
                if value == TOMBSTONE {
                    return None; // Ignore tombstone values
                }
                return Some(value);
            }
        }

        None
    }

    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        let mut results = self.buffer.read().unwrap().range(start, end);

        // Add results from levels
        for level in &self.levels {
            results.extend(level.range(start, end));
        }

        // Sort by key and remove duplicates, keeping only the most recent value
        results.sort_by_key(|&(key, _)| key);
        results.dedup_by_key(|&mut (key, _)| key);

        // Filter out tombstones
        results.retain(|&(_, value)| value != TOMBSTONE);

        results
    }

    pub fn delete(&mut self, key: Key) -> Result<()> {
        self.put(key, TOMBSTONE)
    }

    fn flush_buffer_to_level0(&mut self) -> Result<()> {
        let data = {
            let mut buffer = self.buffer.write().unwrap();
            let data = buffer.as_map().iter().map(|(&k, &v)| (k, v)).collect(); // Convert BTreeMap to Vec
            buffer.clear();
            data
        };

        let run = Run::new(data);
        if self.levels.is_empty() {
            // Create Level 0 if it doesn't exist
            self.levels.push(Level::new()); // Remove the argument here
        }
        self.levels[0].add_run(run);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_and_get() {
        let mut lsm_tree = LSMTree::new(128);
        lsm_tree.put(1, 100).unwrap();
        lsm_tree.put(2, 200).unwrap();

        assert_eq!(lsm_tree.get(1), Some(100));
        assert_eq!(lsm_tree.get(2), Some(200));
        assert_eq!(lsm_tree.get(3), None);
    }

    #[test]
    fn test_range_query() {
        let mut lsm_tree = LSMTree::new(128);
        lsm_tree.put(1, 100).unwrap();
        lsm_tree.put(2, 200).unwrap();
        lsm_tree.put(3, 300).unwrap();

        let range = lsm_tree.range(1, 4);
        assert_eq!(range, vec![(1, 100), (2, 200), (3, 300)]);
    }

    #[test]
    fn test_delete() {
        let mut lsm_tree = LSMTree::new(128);
        lsm_tree.put(1, 100).unwrap();
        lsm_tree.delete(1).unwrap();

        assert_eq!(lsm_tree.get(1), None);
    }
}
