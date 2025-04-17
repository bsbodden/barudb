use super::{Error, Result};
use crate::types::Key;
use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(dead_code)]
pub trait FilterStrategy: Send + Sync {
    fn new(expected_entries: usize) -> Self
    where
        Self: Sized;
    fn add(&mut self, key: &Key) -> Result<()>;
    fn may_contain(&self, key: &Key) -> bool;
    fn false_positive_rate(&self) -> f64;
    fn serialize(&self) -> Result<Vec<u8>>;
    fn deserialize(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;
    
    /// Clone this filter as a boxed trait object
    fn box_clone(&self) -> Box<dyn FilterStrategy>;
    
    /// Check if multiple keys may be in the filter (batch operation)
    /// 
    /// Default implementation processes keys individually. Implementations should
    /// override this with an optimized batch implementation when available.
    /// 
    /// # Arguments
    /// * `keys` - Slice of keys to check
    /// * `results` - Output slice to store results (must be same length as keys)
    fn may_contain_batch(&self, keys: &[Key], results: &mut [bool]) {
        assert_eq!(keys.len(), results.len(), "Keys and results slices must be the same length");
        for (i, key) in keys.iter().enumerate() {
            results[i] = self.may_contain(key);
        }
    }
    
    /// Add multiple keys to the filter (batch operation)
    /// 
    /// Default implementation processes keys individually. Implementations should
    /// override this with an optimized batch implementation when available.
    /// 
    /// # Arguments
    /// * `keys` - Slice of keys to add to the filter
    /// * `concurrent` - If true, use concurrent add operations with contention reduction
    fn add_batch(&mut self, keys: &[Key]) -> Result<()> {
        for key in keys {
            self.add(key)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct NoopFilter {
    entry_count: AtomicUsize,
}

impl FilterStrategy for NoopFilter {
    fn new(expected_entries: usize) -> Self {
        Self {
            entry_count: AtomicUsize::new(expected_entries)
        }
    }

    fn add(&mut self, _key: &Key) -> Result<()> {
        self.entry_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn may_contain(&self, _key: &Key) -> bool {
        true
    }
    
    /// Override the default implementation for efficiency since this
    /// filter always returns true for all keys
    fn may_contain_batch(&self, _keys: &[Key], results: &mut [bool]) {
        // NoopFilter always returns true for all keys
        results.iter_mut().for_each(|r| *r = true);
    }
    
    /// Override the default implementation for efficiency
    fn add_batch(&mut self, keys: &[Key]) -> Result<()> {
        // Simply increment the counter by the batch size
        self.entry_count.fetch_add(keys.len(), Ordering::SeqCst);
        Ok(())
    }

    fn false_positive_rate(&self) -> f64 {
        1.0
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        // Get the value and convert it to bytes
        let count = self.entry_count.load(Ordering::SeqCst);
        Ok(count.to_le_bytes().to_vec())
    }

    fn deserialize(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != std::mem::size_of::<usize>() {
            return Err(Error::Serialization(
                "Invalid number of bytes for NoopFilter".to_string()
            ));
        }

        let mut size_bytes = [0u8; std::mem::size_of::<usize>()];
        size_bytes.copy_from_slice(bytes);
        let entry_count = usize::from_le_bytes(size_bytes);

        Ok(Self {
            entry_count: AtomicUsize::new(entry_count)
        })
    }
    
    fn box_clone(&self) -> Box<dyn FilterStrategy> {
        Box::new(Self {
            entry_count: AtomicUsize::new(self.entry_count.load(Ordering::Relaxed))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_filter_basic() {
        let mut filter = NoopFilter::new(100);

        // Test adding keys
        assert!(filter.add(&42).is_ok());
        assert!(filter.add(&100).is_ok());

        // Test membership - always returns true
        assert!(filter.may_contain(&42));
        assert!(filter.may_contain(&100));
        assert!(filter.may_contain(&999)); // Even for unseen keys

        // Test false positive rate
        assert_eq!(filter.false_positive_rate(), 1.0);
    }

    #[test]
    fn test_noop_filter_serialization() {
        let mut filter = NoopFilter::new(100);
        filter.add(&1).unwrap();
        filter.add(&2).unwrap();

        // Test serialization
        let serialized = filter.serialize().unwrap();
        let deserialized = NoopFilter::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.entry_count.load(Ordering::SeqCst), 102); // 100 initial + 2 added

        // Test invalid deserialization
        let result = NoopFilter::deserialize(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_noop_filter_counting() {
        let mut filter = NoopFilter::new(0);

        // Add some entries and verify count
        for i in 0..10 {
            filter.add(&i).unwrap();
        }

        let serialized = filter.serialize().unwrap();
        let deserialized = NoopFilter::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.entry_count.load(Ordering::SeqCst), 10);
    }
}