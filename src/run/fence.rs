use crate::run::Result;
use crate::types::Key;
use std::cmp::Ordering;

/// A fence pointer represents a key range and its location in a run
#[derive(Debug, Clone)]
pub struct FencePointer {
    pub min_key: Key,
    pub max_key: Key,
    pub block_index: usize,
}

/// A collection of fence pointers for efficient range queries
#[derive(Debug, Clone)]
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

    /// Find a block that may contain the given key
    pub fn find_block_for_key(&self, key: Key) -> Option<usize> {
        self.pointers
            .binary_search_by(|fence| {
                if key < fence.min_key {
                    Ordering::Greater // Continue search to the left
                } else if key > fence.max_key {
                    Ordering::Less // Continue search to the right
                } else {
                    Ordering::Equal // Found a block that may contain this key
                }
            })
            .ok()
            .map(|idx| self.pointers[idx].block_index)
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
}