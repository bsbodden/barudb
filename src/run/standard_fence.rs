use crate::run::Result;
use crate::types::Key;
use std::cmp::Ordering;
use std::any::Any;

/// Standard fence pointer implementation without optimizations
/// (to compare with optimized implementation)
#[derive(Debug, Clone)]
pub struct StandardFencePointer {
    pub min_key: Key,
    pub max_key: Key,
    pub block_index: usize,
}

/// Standard version without cache alignment or advanced search optimizations
#[derive(Debug, Clone)]
pub struct StandardFencePointers {
    pub pointers: Vec<StandardFencePointer>,
}

// Implement the interface for StandardFencePointers
impl crate::run::FencePointersInterface for StandardFencePointers {
    fn find_block_for_key(&self, key: Key) -> Option<usize> {
        self.find_block_for_key(key)
    }
    
    fn find_blocks_in_range(&self, start: Key, end: Key) -> Vec<usize> {
        self.find_blocks_in_range(start, end)
    }
    
    fn len(&self) -> usize {
        self.len()
    }
    
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
    
    fn clear(&mut self) {
        self.clear()
    }
    
    fn add(&mut self, min_key: Key, max_key: Key, block_index: usize) {
        self.add(min_key, max_key, block_index)
    }
    
    fn optimize(&mut self) {
        // Standard fence pointers don't need optimization
    }
    
    fn memory_usage(&self) -> usize {
        self.memory_usage()
    }
    
    fn serialize(&self) -> crate::run::Result<Vec<u8>> {
        self.serialize()
    }
}

impl StandardFencePointers {
    /// Create a new empty fence pointers collection
    pub fn new() -> Self {
        Self {
            pointers: Vec::new(),
        }
    }

    /// Add a new fence pointer for a block
    pub fn add(&mut self, min_key: Key, max_key: Key, block_index: usize) {
        self.pointers.push(StandardFencePointer {
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
    /// Basic binary search implementation without optimizations
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
            
            pointers.push(StandardFencePointer {
                min_key,
                max_key,
                block_index,
            });
        }
        
        Ok(Self { pointers })
    }
    
    /// Calculate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + 
        self.pointers.capacity() * std::mem::size_of::<StandardFencePointer>()
    }
    
    /// For type conversion in trait implementations
    pub fn as_any(&self) -> &dyn Any {
        self
    }
}

// Legacy FencePointersInterface trait implementation removed