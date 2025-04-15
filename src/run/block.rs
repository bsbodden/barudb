use super::{CompressionStrategy, Result};
use crate::types::{Key, Value};
use std::cmp::{max, min};
use std::mem;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BlockConfig {
    pub target_size: usize,
    pub min_fill_ratio: f32,
    pub max_fill_ratio: f32,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            target_size: page_size::get(),
            min_fill_ratio: 0.5,
            max_fill_ratio: 0.9,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BlockHeader {
    pub entry_count: u32,
    pub min_key: Key,
    pub max_key: Key,
    pub compressed_size: u32,
    pub uncompressed_size: u32,
    pub checksum: u64,
}

#[allow(dead_code)]
impl BlockHeader {
    pub fn new() -> Self {
        Self {
            entry_count: 0,
            min_key: Key::MAX,
            max_key: Key::MIN,
            compressed_size: 0,
            uncompressed_size: 0,
            checksum: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub header: BlockHeader,
    pub entries: Vec<(Key, Value)>,
    pub is_sealed: bool,
}

impl Block {
    pub fn new() -> Self {
        Self {
            header: BlockHeader::new(),
            entries: Vec::new(),
            is_sealed: false,
        }
    }

    pub fn estimated_size(&self) -> usize {
        mem::size_of::<BlockHeader>() +
            self.entries.len() * mem::size_of::<(Key, Value)>()
    }

    pub fn add_entry(&mut self, key: Key, value: Value) -> Result<bool> {
        if self.is_sealed {
            return Ok(false);
        }

        self.entries.push((key, value));
        self.header.entry_count = self.entries.len() as u32;
        self.header.min_key = min(self.header.min_key, key);
        self.header.max_key = max(self.header.max_key, key);

        Ok(true)
    }

    pub fn seal(&mut self) -> Result<()> {
        if !self.is_sealed {
            self.entries.sort_by_key(|(k, _)| *k);
            self.header.uncompressed_size = self.estimated_size() as u32;
            self.is_sealed = true;
        }
        Ok(())
    }

    pub fn get(&self, key: &Key) -> Option<Value> {
        if !self.is_sealed {
            return None;
        }

        // Use binary search since entries are sorted
        self.entries
            .binary_search_by_key(key, |(k, _)| *k)
            .ok()
            .map(|idx| self.entries[idx].1)
    }

    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        if !self.is_sealed || start >= end {
            return Vec::new();
        }

        self.entries
            .iter()
            .filter(|(k, _)| *k >= start && *k < end)
            .cloned()
            .collect()
    }

    #[allow(dead_code)]
    pub fn serialize(&mut self, compression: &dyn CompressionStrategy) -> Result<Vec<u8>> {
        if !self.is_sealed {
            self.seal()?;
        }

        // Create a buffer for the full serialized data
        let mut data = Vec::new();
        
        // Write header fields - in exact same order as they'll be read in deserialize
        data.extend_from_slice(&self.header.entry_count.to_le_bytes());
        data.extend_from_slice(&self.header.min_key.to_le_bytes());
        data.extend_from_slice(&self.header.max_key.to_le_bytes());
        data.extend_from_slice(&self.header.compressed_size.to_le_bytes());
        data.extend_from_slice(&self.header.uncompressed_size.to_le_bytes());
        
        // Write entries
        for (key, value) in &self.entries {
            data.extend_from_slice(&key.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
        }
        
        // Calculate checksum (excluding the checksum field itself)
        let checksum = xxhash_rust::xxh3::xxh3_64(&data);
        self.header.checksum = checksum;
        
        // Add checksum to data
        data.extend_from_slice(&checksum.to_le_bytes());
        
        // Ensure the data is a multiple of 16 bytes for compression
        let padding_needed = (16 - (data.len() % 16)) % 16;
        if padding_needed > 0 {
            // Add padding bytes, which won't affect the checksum 
            // since the checksum was calculated before adding the padding
            for _ in 0..padding_needed {
                data.push(0);
            }
        }
        
        // Update header sizes with padded size
        self.header.uncompressed_size = data.len() as u32;
        
        // Compress the block with properly aligned data
        let compressed = compression.compress(&data)?;
        self.header.compressed_size = compressed.len() as u32;
        
        Ok(compressed)
    }

    #[allow(dead_code)]
    pub fn deserialize(bytes: &[u8], compression: &dyn CompressionStrategy) -> Result<Self> {
        // Decompress the data
        let decompressed = compression.decompress(bytes)?;
        
        if decompressed.len() < std::mem::size_of::<BlockHeader>() {
            return Err(super::Error::Serialization("Block data too small".into()));
        }
        
        // First, parse header size to determine the expected data size without padding
        let header_size = 4 + 8 + 8 + 4 + 4; // entry_count + min_key + max_key + compressed_size + uncompressed_size
        
        // Read entry count from the header to calculate the expected size
        let entry_count = u32::from_le_bytes(decompressed[0..4].try_into().unwrap()) as usize;
        
        // Calculate the expected size: header + entries + checksum
        // Each entry is 16 bytes (8 for key, 8 for value)
        let expected_size = header_size + (entry_count * 16) + 8; // +8 for checksum
        
        // The checksum should be at this position (before any padding)
        let checksum_offset = expected_size - 8;
        
        // Calculate checksum of the decompressed data up to the checksum position
        let data_for_checksum = &decompressed[..checksum_offset];
        let computed_checksum = xxhash_rust::xxh3::xxh3_64(data_for_checksum);
        
        // Extract stored checksum
        let stored_checksum_bytes = &decompressed[checksum_offset..checksum_offset + 8];
        let stored_checksum = u64::from_le_bytes(stored_checksum_bytes.try_into().unwrap());
        
        // Verify checksum
        if computed_checksum != stored_checksum {
            // Properly handle the error in production code
            return Err(super::Error::Serialization(format!(
                "Block checksum mismatch: computed={}, stored={}",
                computed_checksum, stored_checksum
            )));
        }
        
        // Parse header (first several bytes)
        let mut offset = 0;
        
        // Read entry count
        let entry_count = u32::from_le_bytes(decompressed[offset..offset+4].try_into().unwrap());
        offset += 4;
        
        // Read min key
        let min_key = i64::from_le_bytes(decompressed[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        // Read max key
        let max_key = i64::from_le_bytes(decompressed[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        // Read compressed size
        let compressed_size = u32::from_le_bytes(decompressed[offset..offset+4].try_into().unwrap());
        offset += 4;
        
        // Read uncompressed size
        let uncompressed_size = u32::from_le_bytes(decompressed[offset..offset+4].try_into().unwrap());
        offset += 4;
        
        // Reconstruct header
        let header = BlockHeader {
            entry_count,
            min_key,
            max_key,
            compressed_size,
            uncompressed_size,
            checksum: stored_checksum,
        };
        
        // Parse entries
        let mut entries = Vec::with_capacity(entry_count as usize);
        
        for _ in 0..entry_count {
            // Key
            let key = i64::from_le_bytes(decompressed[offset..offset+8].try_into().unwrap());
            offset += 8;
            
            // Value
            let value = i64::from_le_bytes(decompressed[offset..offset+8].try_into().unwrap());
            offset += 8;
            
            entries.push((key, value));
        }
        
        // Create and return the block
        Ok(Block {
            header,
            entries,
            is_sealed: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_operations() {
        let mut block = Block::new();

        // Test adding entries
        assert!(block.add_entry(2, 200).unwrap());
        assert!(block.add_entry(1, 100).unwrap());
        assert!(block.add_entry(3, 300).unwrap());

        let initial_size = block.estimated_size();
        assert!(initial_size > 0);

        // Test sealing
        assert!(block.seal().is_ok());
        assert!(block.is_sealed);
        assert_eq!(block.header.uncompressed_size as usize, block.estimated_size());

        // Verify entries are sorted after sealing
        assert_eq!(block.entries[0], (1, 100));
        assert_eq!(block.entries[1], (2, 200));
        assert_eq!(block.entries[2], (3, 300));

        // Test get after sealing
        assert_eq!(block.get(&1), Some(100));
        assert_eq!(block.get(&2), Some(200));
        assert_eq!(block.get(&4), None);

        // Test range after sealing
        let range = block.range(1, 3);
        assert_eq!(range, vec![(1, 100), (2, 200)]);

        // Test adding after sealing
        assert!(!block.add_entry(4, 400).unwrap());
    }

    #[test]
    fn test_block_header() {
        let mut block = Block::new();

        block.add_entry(5, 500).unwrap();
        block.add_entry(3, 300).unwrap();
        block.add_entry(7, 700).unwrap();

        assert_eq!(block.header.entry_count, 3);
        assert_eq!(block.header.min_key, 3);
        assert_eq!(block.header.max_key, 7);
    }
}