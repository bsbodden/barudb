mod block;
mod compression;
mod filter;

use crate::types::{Key, Value};
use std::io;

use crate::bloom::Bloom;
pub use block::{Block, BlockConfig};
pub use compression::{CompressionStrategy, NoopCompression};
pub use filter::{FilterStrategy, NoopFilter};

#[derive(Debug)]
#[allow(dead_code)]
pub enum Error {
    Io(io::Error),
    Serialization(String),
    Block(String),
    Filter(String),
    Compression(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[allow(dead_code)]
pub struct Run {
    data: Vec<(Key, Value)>,
    // Add new fields
    block_config: BlockConfig,
    blocks: Vec<Block>,
    filter: Box<dyn FilterStrategy>,
    compression: Box<dyn CompressionStrategy>,
}

impl Run {
    pub fn new(data: Vec<(Key, Value)>) -> Self {
        let block_config = BlockConfig::default();
        let mut blocks = Vec::new();

        // Initialize filter with data size
        // Using 10 bits per entry and 6 probes as shown in the test cases
        let total_bits = (data.len() * 10) as u32;
        let num_probes = 6;
        let mut filter: Box<dyn FilterStrategy> = Box::new(Bloom::new(total_bits, num_probes));

        // Create initial block and populate filter
        if !data.is_empty() {
            let mut block = Block::new();
            for (k, v) in data.iter() {
                block.add_entry(*k, *v).unwrap();
                filter.add(k).unwrap();
            }
            block.seal().unwrap();
            blocks.push(block);
        }

        Run {
            data,
            block_config,
            blocks,
            filter,
            compression: Box::new(NoopCompression),
        }
    }

    pub fn get(&self, key: Key) -> Option<Value> {
        // First check filter
        if !self.filter.may_contain(&key) {
            return None;
        }

        // Check blocks
        for block in &self.blocks {
            if let Some(value) = block.get(&key) {
                return Some(value);
            }
        }

        None
    }

    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        let mut results = Vec::new();

        for block in &self.blocks {
            if block.header.min_key <= end && block.header.max_key >= start {
                results.extend(block.range(start, end));
            }
        }

        results
    }

    #[allow(dead_code)]
    pub fn persist(&mut self) -> Result<()> {
        // Ensure each block is serialized
        for block in &mut self.blocks {
            block.serialize(&*self.compression)?;
        }

        // Also serialize filter (will need this later)
        let _filter_data = self.filter.serialize()?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn restore(bytes: &[u8]) -> Result<Self> {
        // TODO: Implement restoration
        let _filter = NoopFilter::deserialize(bytes)?;
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xxhash_rust::xxh3::xxh3_128;

    #[test]
    fn test_run_operations() {
        let data = vec![(1, 100), (2, 200), (3, 300)];
        let run = Run::new(data);

        // Test basic operations
        assert_eq!(run.get(2), Some(200));
        assert_eq!(run.get(4), None);

        // Test range query
        let range = run.range(1, 3);
        assert_eq!(range, vec![(1, 100), (2, 200)]);

        // Verify blocks were created
        assert!(!run.blocks.is_empty());

        // Verify filter works
        assert!(run.filter.may_contain(&1));
    }

    #[test]
    fn test_compression() {
        let mut run = Run::new(vec![(1, 100), (2, 200)]);

        // Test persistence with NoopCompression
        run.persist().unwrap();

        // With NoopCompression, compressed size should equal uncompressed size
        // and both should be greater than 0
        assert!(run.blocks[0].header.uncompressed_size > 0);
        assert_eq!(
            run.blocks[0].header.compressed_size,
            run.blocks[0].header.uncompressed_size
        );
    }

    #[test]
    fn test_filter_operations() {
        let data = vec![(1i64, 100), (2i64, 200)];
        let run = Run::new(data);

        // Test that filter is properly filtering
        assert!(run.filter.may_contain(&1i64));
        assert!(run.filter.may_contain(&2i64));
        assert!(!run.filter.may_contain(&3i64)); // Not in set

        // Test filter serialization
        let filter_data = run.filter.serialize().unwrap();

        // Deserialize as Bloom filter
        let restored_filter = Bloom::deserialize(&filter_data).unwrap();

        // Convert keys to hashes the same way the FilterStrategy impl does
        for key in [1i64, 2i64] {
            let bytes = key.to_le_bytes();
            let hash = xxh3_128(&bytes) as u32;

            // Original filter uses FilterStrategy trait
            let original_result = run.filter.may_contain(&key);
            // Restored filter uses direct Bloom implementation
            let restored_result = restored_filter.may_contain(hash);

            assert_eq!(
                original_result, restored_result,
                "Behavior mismatch for key {}",
                key
            );
        }

        // Verify false positive rate is reasonable
        let fp_rate = restored_filter.false_positive_rate();
        println!("Restored filter false positive rate: {}", fp_rate);
        assert!(
            fp_rate < 0.1,
            "False positive rate {} is too high (> 0.1)",
            fp_rate
        ); // Less than 10%
    }

    #[test]
    fn test_bloom_filter() {
        let data = vec![(1, 100), (2, 200)];
        let run = Run::new(data);

        // Test filter behavior
        assert!(run.filter.may_contain(&1));
        assert!(run.filter.may_contain(&2));
        assert!(!run.filter.may_contain(&3));
    }
}
