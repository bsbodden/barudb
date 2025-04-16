use crate::run::compression::CompressionStrategy;
use crate::types::{Error, Result};
use lz4::block::{compress, decompress, CompressionMode};
use std::any::Any;

/// LZ4 compression strategy implementation
/// 
/// Uses the LZ4 compression algorithm which is known for its high speed and
/// reasonable compression ratio - often preferred in database systems.
#[derive(Debug, Clone)]
pub struct Lz4Compression {
    /// Compression level (1-12, higher is more compression but slower)
    pub compression_level: u32,
    /// Block size for compression operations
    pub block_size: usize,
}

impl Default for Lz4Compression {
    fn default() -> Self {
        Self {
            compression_level: 4, // Default to medium compression level
            block_size: 1024,     // Default block size
        }
    }
}

impl CompressionStrategy for Lz4Compression {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Store original data length at the beginning for decompression
        let mut result = (data.len() as u32).to_le_bytes().to_vec();
        
        // Compress with LZ4
        let compressed = compress(
            data,
            Some(CompressionMode::FAST(self.compression_level as i32)),
            false, // Don't prepend size - we handle that ourselves
        ).map_err(|e| Error::CompressionFailed(format!("LZ4 compression error: {}", e)))?;
        
        // Append compressed data
        result.extend_from_slice(&compressed);
        
        Ok(result)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        // Read original data length
        if data.len() < 4 {
            return Err(Error::CompressionFailed("Invalid LZ4 compressed data: too short".to_string()));
        }
        
        let original_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        
        // Decompress with LZ4
        let decompressed = decompress(&data[4..], Some(original_len as i32))
            .map_err(|e| Error::CompressionFailed(format!("LZ4 decompression error: {}", e)))?;
        
        Ok(decompressed)
    }

    fn estimate_compressed_size(&self, data: &[u8]) -> usize {
        if data.is_empty() {
            return 0;
        }
        
        // LZ4 typically achieves ~50% compression for general data
        // We add a small overhead for the header and worst-case scenario
        4 + (data.len() * 3 / 4)
    }
    
    fn clone_box(&self) -> Box<dyn CompressionStrategy> {
        Box::new(self.clone())
    }
    
    fn name(&self) -> &'static str {
        "lz4"
    }
    
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_lz4_compression_empty_data() {
        let compression = Lz4Compression::default();
        let empty: &[u8] = &[];
        
        let compressed = compression.compress(empty).unwrap();
        assert!(compressed.is_empty());
        
        let decompressed = compression.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_lz4_compression_small_data() {
        let compression = Lz4Compression::default();
        let data = b"test data for compression";
        
        let compressed = compression.compress(data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(&decompressed, data);
    }

    #[test]
    fn test_lz4_compression_large_data() {
        let compression = Lz4Compression::default();
        
        // Create data with high redundancy for good compression
        let mut data = Vec::with_capacity(10000);
        for _ in 0..100 {
            data.extend_from_slice(b"The quick brown fox jumps over the lazy dog. ");
        }
        
        let compressed = compression.compress(&data).unwrap();
        println!("Original size: {}, Compressed size: {}", data.len(), compressed.len());
        
        // Should achieve good compression with repetitive data
        assert!(compressed.len() < data.len());
        
        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lz4_compression_binary_data() {
        let compression = Lz4Compression::default();
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_lz4_different_compression_levels() {
        let data: Vec<u8> = (0..10000).map(|_| rand::random::<u8>()).collect();
        
        let mut sizes = Vec::new();
        for level in [1, 4, 9] {
            let compression = Lz4Compression {
                compression_level: level,
                block_size: 1024,
            };
            
            let compressed = compression.compress(&data).unwrap();
            sizes.push((level, compressed.len()));
            
            let decompressed = compression.decompress(&compressed).unwrap();
            assert_eq!(decompressed, data);
        }
        
        // Higher compression levels should generally result in smaller sizes for random data,
        // though the difference may be small and is not guaranteed for all input patterns
        println!("LZ4 compression sizes by level: {:?}", sizes);
    }
    
    #[test]
    fn test_lz4_corrupted_data() {
        let compression = Lz4Compression::default();
        let data = b"test data for compression that should be long enough to cause issues when corrupted";
        
        // Create compressed data then corrupt it severely
        let mut compressed = compression.compress(data).unwrap();
        if compressed.len() > 10 {
            // Corrupt the LZ4 frame header which should cause a parsing error
            compressed[4] = 0xFF;
            compressed[5] = 0xFF;
            
            // Create complete garbage in the middle of the stream
            let start_idx = compressed.len() / 3;
            for i in 0..5 {
                if start_idx + i < compressed.len() {
                    compressed[start_idx + i] = 0xFF;
                }
            }
            
            // Modify the original length value (first 4 bytes) to be incorrect
            if compressed.len() >= 4 {
                let impossible_size = u32::MAX - 100;
                let bytes = impossible_size.to_le_bytes();
                compressed[0] = bytes[0];
                compressed[1] = bytes[1]; 
                compressed[2] = bytes[2];
                compressed[3] = bytes[3];
            }
            
            // Should fail to decompress corrupted data
            let result = compression.decompress(&compressed);
            assert!(result.is_err(), "Decompression should fail with corrupted data");
        }
    }
    
    #[test]
    fn test_lz4_incomplete_data() {
        let compression = Lz4Compression::default();
        let data = b"test data for compression";
        
        // Create compressed data then truncate it
        let mut compressed = compression.compress(data).unwrap();
        if compressed.len() > 5 {
            compressed.truncate(compressed.len() / 2);
            
            // Should fail to decompress truncated data
            let result = compression.decompress(&compressed);
            assert!(result.is_err(), "Decompression should fail with truncated data");
        }
    }
    
    #[test]
    fn test_lz4_performance() {
        let compression = Lz4Compression::default();
        
        // Create test data with varying properties
        let sequential_data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let random_data: Vec<u8> = (0..10000).map(|_| rand::random::<u8>()).collect();
        let mut repetitive_data = Vec::with_capacity(10000);
        for _ in 0..250 {
            repetitive_data.extend_from_slice(b"abcd");
        }
        
        // Benchmark compression of sequential data
        let start = Instant::now();
        let compressed_seq = compression.compress(&sequential_data).unwrap();
        let compress_time_seq = start.elapsed();
        
        // Benchmark decompression of sequential data
        let start = Instant::now();
        let _ = compression.decompress(&compressed_seq).unwrap();
        let decompress_time_seq = start.elapsed();
        
        // Benchmark compression of random data
        let start = Instant::now();
        let compressed_rand = compression.compress(&random_data).unwrap();
        let compress_time_rand = start.elapsed();
        
        // Benchmark decompression of random data
        let start = Instant::now();
        let _ = compression.decompress(&compressed_rand).unwrap();
        let decompress_time_rand = start.elapsed();
        
        // Benchmark compression of repetitive data
        let start = Instant::now();
        let compressed_rep = compression.compress(&repetitive_data).unwrap();
        let compress_time_rep = start.elapsed();
        
        // Benchmark decompression of repetitive data
        let start = Instant::now();
        let _ = compression.decompress(&compressed_rep).unwrap();
        let decompress_time_rep = start.elapsed();
        
        println!("\nLZ4 Performance:");
        println!("Sequential data - Compression: {:?}, Decompression: {:?}, Ratio: {:.2}x", 
                 compress_time_seq, decompress_time_seq, 
                 sequential_data.len() as f64 / compressed_seq.len() as f64);
        println!("Random data - Compression: {:?}, Decompression: {:?}, Ratio: {:.2}x", 
                 compress_time_rand, decompress_time_rand,
                 random_data.len() as f64 / compressed_rand.len() as f64);
        println!("Repetitive data - Compression: {:?}, Decompression: {:?}, Ratio: {:.2}x", 
                 compress_time_rep, decompress_time_rep,
                 repetitive_data.len() as f64 / compressed_rep.len() as f64);
    }
}