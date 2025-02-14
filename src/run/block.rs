use super::{CompressionStrategy, Result};
use crate::types::{Key, Value};
use std::cmp::{max, min};
use std::mem;

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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
        // First serialize entries to bytes
        let mut data = Vec::new();
        for (key, value) in &self.entries {
            data.extend_from_slice(&key.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
        }

        // Compress the data
        let compressed = compression.compress(&data)?;
        self.header.compressed_size = compressed.len() as u32;
        self.header.uncompressed_size = data.len() as u32;

        Ok(compressed)
    }

    #[allow(dead_code)]
    pub fn deserialize(bytes: &[u8], compression: &dyn CompressionStrategy) -> Result<Self> {
        let _decompressed = compression.decompress(bytes)?;
        // TODO: Implement deserialization
        todo!()
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