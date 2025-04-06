use crate::run::Result;
use crate::types::Key;

// For hardware prefetching on x86_64
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

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
    /// 
    /// Platform-specific implementation with prefetching for x86_64
    #[cfg(target_arch = "x86_64")]
    pub fn find_blocks_in_range(&self, start: Key, end: Key) -> Vec<usize> {
        if start >= end {
            return Vec::new();
        }

        // Early return if the range is completely outside our pointers' range
        if self.pointers.is_empty() ||
           end <= self.pointers.first().unwrap().min_key || 
           start > self.pointers.last().unwrap().max_key {
            return Vec::new();
        }

        // For large collections, prefetch chunks of pointers during scanning
        let mut result = Vec::new();
        let prefetch_distance = 4; // Adjust based on expected cache line size

        for i in 0..self.pointers.len() {
            // Prefetch ahead
            if i + prefetch_distance < self.pointers.len() {
                unsafe {
                    _mm_prefetch(
                        &self.pointers[i + prefetch_distance] as *const _ as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }

            let fence = &self.pointers[i];
            if fence.min_key < end && fence.max_key >= start {
                result.push(fence.block_index);
            }
        }

        result
    }
    
    /// Default implementation without prefetching for non-x86_64 platforms
    #[cfg(not(target_arch = "x86_64"))]
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
    /// 
    /// Uses hardware prefetching on x86_64 platforms to reduce memory latency
    #[cfg(target_arch = "x86_64")]
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
                // Prefetch ahead in the exponential search pattern
                if pos + step * 2 < self.pointers.len() {
                    unsafe {
                        _mm_prefetch(
                            &self.pointers[pos + step * 2] as *const _ as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }
                
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
            
            // Prefetch next potential search locations based on search direction
            let prefetch_distance = (high - low) / 4;
            if prefetch_distance > 0 {
                if key < self.pointers[mid].min_key && mid > prefetch_distance {
                    // If we're likely going left, prefetch to the left
                    unsafe {
                        _mm_prefetch(
                            &self.pointers[mid - prefetch_distance] as *const _ as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                } else if key > self.pointers[mid].max_key && mid + prefetch_distance < self.pointers.len() {
                    // If we're likely going right, prefetch to the right
                    unsafe {
                        _mm_prefetch(
                            &self.pointers[mid + prefetch_distance] as *const _ as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }
            }
            
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
    
    /// Non-x86_64 implementation without prefetching
    #[cfg(not(target_arch = "x86_64"))]
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
    
    /// Deserialize fence pointers from bytes with prefetching on x86_64
    #[cfg(target_arch = "x86_64")]
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Ok(Self::new()); // Return empty fence pointers for compatibility
        }
        
        let mut offset = 0;
        
        // Read count
        let count = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
        offset += 4;
        
        // Calculate the stride for a single pointer record
        let record_size = 20; // 8 bytes min_key + 8 bytes max_key + 4 bytes block_index
        
        // Read fence pointers with prefetching
        let mut pointers = Vec::with_capacity(count as usize);
        let prefetch_distance = 4; // Prefetch 4 records ahead
        
        for i in 0..count as usize {
            if offset + record_size > bytes.len() {
                break; // Not enough bytes left
            }
            
            // Prefetch ahead to reduce memory latency
            if i + prefetch_distance < count as usize && 
               offset + (prefetch_distance + 1) * record_size <= bytes.len() {
                let prefetch_offset = offset + prefetch_distance * record_size;
                unsafe {
                    _mm_prefetch(
                        &bytes[prefetch_offset] as *const _ as *const i8,
                        _MM_HINT_T0,
                    );
                }
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
    
    /// Deserialize fence pointers from bytes without prefetching for non-x86_64 platforms
    #[cfg(not(target_arch = "x86_64"))]
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
    #[cfg(target_arch = "x86_64")]
    fn test_prefetching_implementation() {
        // Create a large collection to ensure the prefetching code is exercised
        let mut fences = FencePointers::new();
        
        // Add many fence pointers to trigger the prefetching path
        for i in 0..1000 {
            let min_key = i as Key * 10;
            let max_key = i as Key * 10 + 9;
            fences.add(min_key, max_key, i as usize);
        }
        
        // Test point lookups that should exercise the prefetching
        for i in 0..1000 {
            let key = i as Key * 10 + 5; // Middle of each range
            assert_eq!(fences.find_block_for_key(key), Some(i as usize));
        }
        
        // Test range lookups that should exercise the prefetching
        let range1 = fences.find_blocks_in_range(15, 35);
        assert!(range1.len() >= 2); // Should at least include blocks 1 and 2
        
        let range2 = fences.find_blocks_in_range(995, 1015);
        assert!(range2.len() >= 1); // Should at least include block 99
        
        // Test serialization/deserialization with prefetching
        let serialized = fences.serialize().unwrap();
        let deserialized = FencePointers::deserialize(&serialized).unwrap();
        
        // Verify the deserialized pointers still work correctly
        assert_eq!(deserialized.len(), 1000);
        assert_eq!(deserialized.find_block_for_key(505), Some(50));
        
        let range = deserialized.find_blocks_in_range(500, 530);
        assert!(range.len() >= 3); // Should at least include blocks 50, 51, 52
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