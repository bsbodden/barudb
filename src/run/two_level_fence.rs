use crate::run::Result;
use crate::types::Key;
use std::cmp::{max, min};

/// A sparse index that guides lookups to the dense index
#[derive(Debug, Clone)]
pub struct SparseIndex {
    pub guide_keys: Vec<Key>,
    pub dense_indices: Vec<usize>,
}

/// A dense index containing the actual fence pointers
#[derive(Debug, Clone)]
pub struct DenseIndex {
    pub min_key: Key,
    pub max_key: Key,
    pub pointers: Vec<(Key, Key, usize)>, // (min_key, max_key, block_index)
}

/// Two-level fence pointers structure with a sparse upper level
/// and a dense lower level for better memory performance
/// Similar to RocksDB's two-level index structure
#[derive(Debug, Clone)]
pub struct TwoLevelFencePointers {
    pub sparse: SparseIndex,
    pub dense: DenseIndex,
    pub sparse_ratio: usize, // Controls how sparse the top level is
}

impl TwoLevelFencePointers {
    /// Create a new empty two-level fence pointers collection
    pub fn new() -> Self {
        Self {
            sparse: SparseIndex {
                guide_keys: Vec::new(),
                dense_indices: Vec::new(),
            },
            dense: DenseIndex {
                min_key: Key::MAX,
                max_key: Key::MIN,
                pointers: Vec::new(),
            },
            sparse_ratio: 10, // Default - one sparse entry per 10 dense entries
        }
    }

    /// Create a new two-level fence pointers collection with a specific sparse ratio
    pub fn with_ratio(sparse_ratio: usize) -> Self {
        let mut fp = Self::new();
        fp.sparse_ratio = max(1, sparse_ratio); // Ensure at least 1
        fp
    }

    /// Add a new fence pointer for a block
    pub fn add(&mut self, min_key: Key, max_key: Key, block_index: usize) {
        // Update dense index
        self.dense.pointers.push((min_key, max_key, block_index));
        self.dense.min_key = min(self.dense.min_key, min_key);
        self.dense.max_key = max(self.dense.max_key, max_key);
        
        // Rebuild sparse index if needed based on ratio
        self.rebuild_sparse_if_needed();
    }

    /// Get the total number of fence pointers
    pub fn len(&self) -> usize {
        self.dense.pointers.len()
    }

    /// Check if the fence pointers collection is empty
    pub fn is_empty(&self) -> bool {
        self.dense.pointers.is_empty()
    }

    /// Rebuild the sparse index based on the current dense index
    fn rebuild_sparse_if_needed(&mut self) {
        // Always rebuild the sparse index for simplicity in tests
        // In production, we might want to rebuild only on certain thresholds
        self.rebuild_sparse_index();
    }

    /// Rebuilds the sparse index from scratch
    fn rebuild_sparse_index(&mut self) {
        // Clear existing sparse indices
        self.sparse.guide_keys.clear();
        self.sparse.dense_indices.clear();
        
        // If no dense pointers, nothing to do
        if self.dense.pointers.is_empty() {
            return;
        }
        
        // Sort dense pointers by min_key for consistent sparse indexing
        self.dense.pointers.sort_by_key(|&(min_key, _, _)| min_key);
        
        // Always include the first element
        self.sparse.guide_keys.push(self.dense.pointers[0].0);
        self.sparse.dense_indices.push(0);
        
        // Calculate the number of sparse entries we want (at least 1 per sparse_ratio)
        // If sparse_ratio is too large relative to collection size, ensure at least 2 entries
        let num_entries = max(2, self.dense.pointers.len() / self.sparse_ratio);
        
        if num_entries > 1 && self.dense.pointers.len() > 1 {
            // Calculate the step size to achieve desired number of entries
            let step = max(1, self.dense.pointers.len() / (num_entries - 1)); // -1 because we already added first entry
            
            // Add entries at regular intervals, skipping the first one (already added)
            for i in (step..self.dense.pointers.len()).step_by(step) {
                self.sparse.guide_keys.push(self.dense.pointers[i].0);
                self.sparse.dense_indices.push(i);
            }
            
            // Always add the last entry if not already added
            let last_idx = self.dense.pointers.len() - 1;
            if self.sparse.dense_indices.is_empty() || *self.sparse.dense_indices.last().unwrap() != last_idx {
                self.sparse.guide_keys.push(self.dense.pointers[last_idx].0);
                self.sparse.dense_indices.push(last_idx);
            }
        }
        
        // Debug output disabled for benchmarks
        // println!(
        //     "Rebuilt sparse index: ratio={}, dense={}, sparse={}", 
        //     self.sparse_ratio, 
        //     self.dense.pointers.len(), 
        //     self.sparse.guide_keys.len()
        // );
    }

    /// Find a block that may contain the given key
    pub fn find_block_for_key(&self, key: Key) -> Option<usize> {
        // Fast path for empty case
        if self.dense.pointers.is_empty() {
            return None;
        }
        
        // Check bounds first to avoid unnecessary searches
        if key < self.dense.min_key || key > self.dense.max_key {
            return None;
        }
        
        // Use sparse index to narrow down the search range
        let (start_idx, end_idx) = self.find_dense_range(key);
        
        // Search within the identified range in the dense index
        for i in start_idx..end_idx {
            let (min_key, max_key, block_index) = self.dense.pointers[i];
            if key >= min_key && key <= max_key {
                return Some(block_index);
            }
        }
        
        None
    }
    
    /// Find all blocks that may contain keys in the given range
    pub fn find_blocks_in_range(&self, start: Key, end: Key) -> Vec<usize> {
        if start >= end || self.dense.pointers.is_empty() {
            return Vec::new();
        }
        
        // Check if the range overlaps with our fence pointers
        if end <= self.dense.min_key || start > self.dense.max_key {
            return Vec::new();
        }
        
        // Use sparse index to narrow down the search range
        let (start_idx, _) = self.find_dense_range(start);
        let (_, end_idx) = self.find_dense_range(end);
        
        // Collect all potentially overlapping blocks
        let mut result = Vec::new();
        for i in start_idx..end_idx {
            let (min_key, max_key, block_index) = self.dense.pointers[i];
            if min_key < end && max_key >= start {
                result.push(block_index);
            }
        }
        
        result
    }
    
    /// Use the sparse index to find a range in the dense index to search
    fn find_dense_range(&self, key: Key) -> (usize, usize) {
        // Default to full range
        let mut start_idx = 0;
        let mut end_idx = self.dense.pointers.len();
        
        // Use sparse index if available
        if !self.sparse.guide_keys.is_empty() {
            match self.sparse.guide_keys.binary_search(&key) {
                Ok(idx) => {
                    // Direct hit in sparse index
                    start_idx = self.sparse.dense_indices[idx];
                    end_idx = if idx + 1 < self.sparse.dense_indices.len() {
                        self.sparse.dense_indices[idx + 1]
                    } else {
                        self.dense.pointers.len()
                    };
                },
                Err(idx) => {
                    if idx == 0 {
                        // Before first guide key
                        end_idx = self.sparse.dense_indices[0];
                    } else if idx >= self.sparse.guide_keys.len() {
                        // After last guide key
                        start_idx = self.sparse.dense_indices[idx - 1];
                    } else {
                        // Between guide keys
                        start_idx = self.sparse.dense_indices[idx - 1];
                        end_idx = self.sparse.dense_indices[idx];
                    }
                }
            }
        }
        
        (start_idx, end_idx)
    }

    /// Serialize two-level fence pointers to bytes
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        
        // First rebuild sparse index to ensure it's up to date
        let mut clone = self.clone();
        clone.rebuild_sparse_index();
        
        // Write sparse_ratio
        result.extend_from_slice(&(clone.sparse_ratio as u32).to_le_bytes());
        
        // Write sparse index
        let sparse_count = clone.sparse.guide_keys.len() as u32;
        result.extend_from_slice(&sparse_count.to_le_bytes());
        
        for i in 0..clone.sparse.guide_keys.len() {
            // Write guide key
            result.extend_from_slice(&clone.sparse.guide_keys[i].to_le_bytes());
            // Write dense index
            result.extend_from_slice(&(clone.sparse.dense_indices[i] as u32).to_le_bytes());
        }
        
        // Write dense index
        let dense_count = clone.dense.pointers.len() as u32;
        result.extend_from_slice(&dense_count.to_le_bytes());
        result.extend_from_slice(&clone.dense.min_key.to_le_bytes());
        result.extend_from_slice(&clone.dense.max_key.to_le_bytes());
        
        for (min_key, max_key, block_index) in &clone.dense.pointers {
            result.extend_from_slice(&min_key.to_le_bytes());
            result.extend_from_slice(&max_key.to_le_bytes());
            result.extend_from_slice(&(*block_index as u32).to_le_bytes());
        }
        
        Ok(result)
    }
    
    /// Deserialize two-level fence pointers from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Ok(Self::new()); // Return empty fence pointers for compatibility
        }
        
        let mut offset = 0;
        
        // Read sparse_ratio
        let sparse_ratio = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        
        // Read sparse index
        let sparse_count = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        
        let mut guide_keys = Vec::with_capacity(sparse_count);
        let mut dense_indices = Vec::with_capacity(sparse_count);
        
        for _ in 0..sparse_count {
            if offset + 12 > bytes.len() {
                break; // Not enough bytes left
            }
            
            // Read guide key
            let guide_key = Key::from_le_bytes(bytes[offset..offset+8].try_into().unwrap());
            offset += 8;
            
            // Read dense index
            let dense_idx = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            
            guide_keys.push(guide_key);
            dense_indices.push(dense_idx);
        }
        
        // Read dense index
        if offset + 4 > bytes.len() {
            // Not enough bytes for dense count
            return Ok(Self::with_ratio(sparse_ratio));
        }
        
        let dense_count = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        
        if offset + 16 > bytes.len() {
            // Not enough bytes for min/max keys
            return Ok(Self::with_ratio(sparse_ratio));
        }
        
        let min_key = Key::from_le_bytes(bytes[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        let max_key = Key::from_le_bytes(bytes[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        let mut pointers = Vec::with_capacity(dense_count);
        
        for _ in 0..dense_count {
            if offset + 20 > bytes.len() {
                break; // Not enough bytes left
            }
            
            let pointer_min_key = Key::from_le_bytes(bytes[offset..offset+8].try_into().unwrap());
            offset += 8;
            
            let pointer_max_key = Key::from_le_bytes(bytes[offset..offset+8].try_into().unwrap());
            offset += 8;
            
            let block_index = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            
            pointers.push((pointer_min_key, pointer_max_key, block_index));
        }
        
        Ok(Self {
            sparse: SparseIndex {
                guide_keys,
                dense_indices,
            },
            dense: DenseIndex {
                min_key,
                max_key,
                pointers,
            },
            sparse_ratio,
        })
    }
    
    /// Clear all fence pointers
    pub fn clear(&mut self) {
        self.sparse.guide_keys.clear();
        self.sparse.dense_indices.clear();
        self.dense.pointers.clear();
        self.dense.min_key = Key::MAX;
        self.dense.max_key = Key::MIN;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_two_level_fence_pointers_basic() {
        let mut fences = TwoLevelFencePointers::new();
        
        // Add some fence pointers
        fences.add(10, 20, 0);
        fences.add(25, 35, 1);
        fences.add(40, 50, 2);
        
        // Test finding blocks for specific keys
        assert_eq!(fences.find_block_for_key(15), Some(0));
        assert_eq!(fences.find_block_for_key(30), Some(1));
        assert_eq!(fences.find_block_for_key(45), Some(2));
        assert_eq!(fences.find_block_for_key(22), None);
        
        // Test finding blocks in range
        assert_eq!(fences.find_blocks_in_range(15, 35), vec![0, 1]);
        assert_eq!(fences.find_blocks_in_range(5, 15), vec![0]);
        assert_eq!(fences.find_blocks_in_range(22, 27), vec![1]);
        assert_eq!(fences.find_blocks_in_range(37, 42), vec![2]);
        assert_eq!(fences.find_blocks_in_range(5, 60), vec![0, 1, 2]);
    }
    
    #[test]
    fn test_two_level_fence_pointers_serialization() {
        let mut fences = TwoLevelFencePointers::new();
        
        // Add some fence pointers
        fences.add(10, 20, 0);
        fences.add(25, 35, 1);
        
        // Serialize and deserialize
        let serialized = fences.serialize().unwrap();
        let deserialized = TwoLevelFencePointers::deserialize(&serialized).unwrap();
        
        // Check that we get the same data back
        assert_eq!(fences.len(), deserialized.len());
        assert_eq!(fences.sparse_ratio, deserialized.sparse_ratio);
        
        // Ensure the lookups work the same
        assert_eq!(fences.find_block_for_key(15), deserialized.find_block_for_key(15));
        assert_eq!(fences.find_block_for_key(30), deserialized.find_block_for_key(30));
        assert_eq!(fences.find_block_for_key(22), deserialized.find_block_for_key(22));
    }
    
    #[test]
    fn test_two_level_fence_pointers_empty() {
        let fences = TwoLevelFencePointers::new();
        
        // Empty fence pointers should return empty results
        assert_eq!(fences.find_block_for_key(10), None);
        assert!(fences.find_blocks_in_range(10, 20).is_empty());
        
        // Test serialization of empty fence pointers
        let serialized = fences.serialize().unwrap();
        let deserialized = TwoLevelFencePointers::deserialize(&serialized).unwrap();
        assert!(deserialized.is_empty());
    }
    
    #[test]
    fn test_sparse_index_rebuild() {
        // Create a two-level fence pointers structure with a low sparse ratio
        let mut fences = TwoLevelFencePointers::with_ratio(3); // 1 sparse entry per 3 dense entries
        
        // Add fence pointers to trigger sparse index rebuild
        for i in 0..10 {
            let min_key = i as Key * 10;
            let max_key = i as Key * 10 + 9;
            fences.add(min_key, max_key, i as usize);
        }
        
        // Force a rebuild to ensure the sparse index is up to date
        fences.rebuild_sparse_index();
        
        // With a ratio of 3, we should have about 3-4 sparse index entries
        assert!(fences.sparse.guide_keys.len() >= 3);
        assert_eq!(fences.sparse.guide_keys.len(), fences.sparse.dense_indices.len());
        
        // Check lookups with the sparse index
        for i in 0..10 {
            let key = i as Key * 10 + 5; // Middle of each range
            assert_eq!(fences.find_block_for_key(key), Some(i as usize));
        }
    }
    
    #[test]
    fn test_two_level_large_collection() {
        // Create a large collection to test with
        let mut fences = TwoLevelFencePointers::with_ratio(20); // 1 sparse entry per 20 dense entries
        
        // Add many fence pointers
        for i in 0..1000 {
            let min_key = i as Key * 10;
            let max_key = i as Key * 10 + 9;
            fences.add(min_key, max_key, i as usize);
        }
        
        // Force a rebuild to ensure the sparse index is up to date
        fences.rebuild_sparse_index();
        
        // Verify that sparse index has been built correctly
        assert!(!fences.sparse.guide_keys.is_empty());
        assert_eq!(fences.sparse.guide_keys.len(), fences.sparse.dense_indices.len());
        
        // With ratio of 20, we expect sparse entries based on calculation
        // We should have at least 1 entry per sparse ratio, plus first and last entries
        let min_expected = 1000 / 20; // At least one per sparse ratio
        assert!(fences.sparse.guide_keys.len() >= min_expected);
        // And not too many more than that
        assert!(fences.sparse.guide_keys.len() <= min_expected * 2);
        
        // Test lookups across the range
        for i in 0..1000 {
            let key = i as Key * 10 + 5; // Middle of each range
            assert_eq!(fences.find_block_for_key(key), Some(i as usize));
        }
        
        // Test range queries
        let blocks = fences.find_blocks_in_range(500, 530);
        // Should include at least block indices 50, 51, 52
        assert!(blocks.len() >= 3);
        assert!(blocks.contains(&50));
        assert!(blocks.contains(&51));
        assert!(blocks.contains(&52));
    }
    
    #[test]
    fn test_sparse_ratio_impact() {
        // Compare different sparse ratios to verify behavior
        let test_find_block = |ratio: usize, collection_size: usize| {
            let mut fences = TwoLevelFencePointers::with_ratio(ratio);
            
            // Add fence pointers
            for i in 0..collection_size {
                let min_key = i as Key * 10;
                let max_key = i as Key * 10 + 9;
                fences.add(min_key, max_key, i as usize);
            }
            
            // Force a rebuild to ensure the sparse index is up to date
            fences.rebuild_sparse_index();
            
            // Expected minimum number of sparse entries based on ratio
            let min_expected = collection_size / ratio;
            
            // Check actual number of sparse entries
            assert!(fences.sparse.guide_keys.len() >= min_expected);
            // Should not exceed min * 2 (this allows for first/last entry and rounding)
            assert!(fences.sparse.guide_keys.len() <= min_expected * 2 + 2);
            
            // Verify lookups still work
            for i in 0..collection_size {
                let key = i as Key * 10 + 5; // Middle of each range
                assert_eq!(fences.find_block_for_key(key), Some(i as usize));
            }
        };
        
        // Test with different combinations
        test_find_block(5, 100); // 1 sparse per 5 dense, 100 entries
        test_find_block(10, 100); // 1 sparse per 10 dense, 100 entries
        test_find_block(20, 100); // 1 sparse per 20 dense, 100 entries
        test_find_block(100, 1000); // 1 sparse per 100 dense, 1000 entries
    }
}