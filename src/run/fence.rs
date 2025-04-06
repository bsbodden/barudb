use crate::run::Result;
use crate::types::Key;

/// A fence pointer represents a key range and its location in a run
#[derive(Debug, Clone)]
#[repr(align(16))] // Align to 16 bytes for better cache line utilization
pub struct FencePointer {
    pub min_key: Key,
    pub max_key: Key,
    pub block_index: usize,
}

/// A collection of fence pointers for efficient range queries
#[derive(Debug, Clone)]
#[repr(align(64))] // Align to typical cache line size (64 bytes)
pub struct FencePointers {
    pub pointers: Vec<FencePointer>,
}

impl FencePointers {
    /// Create a new empty fence pointers collection
    pub fn new() -> Self {
        Self {
            pointers: Vec::new(),
        }
    }

    /// Add a new fence pointer for a block
    pub fn add(&mut self, min_key: Key, max_key: Key, block_index: usize) {
        self.pointers.push(FencePointer {
            min_key,
            max_key,
            block_index,
        });
    }

    /// Clear all fence pointers
    pub fn clear(&mut self) {
        self.pointers.clear();
    }

    /// Get the total number of fence pointers
    pub fn len(&self) -> usize {
        self.pointers.len()
    }

    /// Check if the fence pointers collection is empty
    pub fn is_empty(&self) -> bool {
        self.pointers.is_empty()
    }

    /// Find all blocks that may contain keys in the given range
    pub fn find_blocks_in_range(&self, start: Key, end: Key) -> Vec<usize> {
        if start >= end {
            return Vec::new();
        }

        self.pointers
            .iter()
            .filter(|fence| fence.min_key < end && fence.max_key >= start)
            .map(|fence| fence.block_index)
            .collect()
    }

    /// Find a block that may contain the given key with optimized binary search
    pub fn find_block_for_key(&self, key: Key) -> Option<usize> {
        // Fast path for empty case
        if self.pointers.is_empty() {
            return None;
        }
        
        // Check bounds first to avoid unnecessary searches
        if key < self.pointers.first().unwrap().min_key || 
           key > self.pointers.last().unwrap().max_key {
            return None;
        }
        
        // Use exponential search for better cache locality on large collections
        let mut low = 0;
        let mut high = self.pointers.len() - 1;
        
        // Improve cache locality with exponential search for larger collections
        if self.pointers.len() > 8 {  // Only use for larger collections
            let mut step = 1;
            let mut pos = 0;
            
            // Find a range containing the key
            while pos < self.pointers.len() && key > self.pointers[pos].max_key {
                pos += step;
                step *= 2;
                
                if pos >= self.pointers.len() {
                    pos = self.pointers.len() - 1;
                    break;
                }
            }
            
            // Set binary search bounds from exponential search
            if pos > 0 {
                low = pos / 2;
            }
            high = pos;
        }
        
        // Binary search within narrowed bounds
        while low <= high {
            let mid = low + (high - low) / 2;
            let fence = &self.pointers[mid];
            
            if key < fence.min_key {
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            } else if key > fence.max_key {
                low = mid + 1;
            } else {
                return Some(fence.block_index);
            }
        }
        
        None
    }

    /// Serialize fence pointers to bytes
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        
        // Write number of fence pointers
        let count = self.pointers.len() as u32;
        result.extend_from_slice(&count.to_le_bytes());
        
        // Write each fence pointer
        for fence in &self.pointers {
            result.extend_from_slice(&fence.min_key.to_le_bytes());
            result.extend_from_slice(&fence.max_key.to_le_bytes());
            result.extend_from_slice(&(fence.block_index as u32).to_le_bytes());
        }
        
        Ok(result)
    }
    
    /// Deserialize fence pointers from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Ok(Self::new()); // Return empty fence pointers for compatibility
        }
        
        let mut offset = 0;
        
        // Read count
        let count = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
        offset += 4;
        
        // Read fence pointers
        let mut pointers = Vec::with_capacity(count as usize);
        for _ in 0..count {
            if offset + 20 > bytes.len() {
                break; // Not enough bytes left
            }
            
            let min_key = Key::from_le_bytes(bytes[offset..offset+8].try_into().unwrap());
            offset += 8;
            
            let max_key = Key::from_le_bytes(bytes[offset..offset+8].try_into().unwrap());
            offset += 8;
            
            let block_index = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            
            pointers.push(FencePointer {
                min_key,
                max_key,
                block_index,
            });
        }
        
        Ok(Self { pointers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fence_pointers_basic() {
        let mut fences = FencePointers::new();
        
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
    fn test_fence_pointers_serialization() {
        let mut fences = FencePointers::new();
        
        // Add some fence pointers
        fences.add(10, 20, 0);
        fences.add(25, 35, 1);
        
        // Serialize and deserialize
        let serialized = fences.serialize().unwrap();
        let deserialized = FencePointers::deserialize(&serialized).unwrap();
        
        // Check that we get the same data back
        assert_eq!(fences.len(), deserialized.len());
        assert_eq!(fences.pointers[0].min_key, deserialized.pointers[0].min_key);
        assert_eq!(fences.pointers[0].max_key, deserialized.pointers[0].max_key);
        assert_eq!(fences.pointers[0].block_index, deserialized.pointers[0].block_index);
        assert_eq!(fences.pointers[1].min_key, deserialized.pointers[1].min_key);
        assert_eq!(fences.pointers[1].max_key, deserialized.pointers[1].max_key);
        assert_eq!(fences.pointers[1].block_index, deserialized.pointers[1].block_index);
    }
    
    #[test]
    fn test_fence_pointers_empty() {
        let fences = FencePointers::new();
        
        // Empty fence pointers should return empty results
        assert_eq!(fences.find_block_for_key(10), None);
        assert!(fences.find_blocks_in_range(10, 20).is_empty());
        
        // Test serialization of empty fence pointers
        let serialized = fences.serialize().unwrap();
        let deserialized = FencePointers::deserialize(&serialized).unwrap();
        assert!(deserialized.is_empty());
    }
    
    #[test]
    fn test_optimized_binary_search() {
        // Create a large collection of fence pointers for testing the optimized search
        let mut fences = FencePointers::new();
        
        // Add fence pointers with non-overlapping ranges
        for i in 0..1000 {
            let min_key = i as Key * 10;
            let max_key = i as Key * 10 + 9;
            fences.add(min_key, max_key, i as usize);
        }
        
        // Test boundary conditions
        assert_eq!(fences.find_block_for_key(0), Some(0)); // First key
        assert_eq!(fences.find_block_for_key(9999), Some(999)); // Last key
        assert_eq!(fences.find_block_for_key(-1), None); // Before first key
        assert_eq!(fences.find_block_for_key(10000), None); // After last key
        
        // Test random keys within ranges
        assert_eq!(fences.find_block_for_key(123), Some(12));
        assert_eq!(fences.find_block_for_key(4567), Some(456));
        assert_eq!(fences.find_block_for_key(9876), Some(987));
        
        // Test keys at boundaries
        assert_eq!(fences.find_block_for_key(100), Some(10)); // Start of a range
        assert_eq!(fences.find_block_for_key(109), Some(10)); // End of a range
        assert_eq!(fences.find_block_for_key(110), Some(11)); // Start of next range
    }
    
    #[test]
    fn test_exponential_search_threshold() {
        // Test with smaller collection (should use standard binary search)
        let mut small_fences = FencePointers::new();
        for i in 0..8 {
            small_fences.add(i as Key * 10, i as Key * 10 + 9, i as usize);
        }
        
        // Test with larger collection (should use exponential search optimization)
        let mut large_fences = FencePointers::new();
        for i in 0..100 {
            large_fences.add(i as Key * 10, i as Key * 10 + 9, i as usize);
        }
        
        // Both should find the correct blocks
        assert_eq!(small_fences.find_block_for_key(45), Some(4));
        assert_eq!(large_fences.find_block_for_key(45), Some(4));
        
        // Test with keys that should utilize the exponential search pattern
        assert_eq!(large_fences.find_block_for_key(950), Some(95));
    }
}