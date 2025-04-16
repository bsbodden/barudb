use crate::run::compression::CompressionStrategy;
use crate::types::{Error, Result};
use snap::raw::{Encoder, Decoder};
use std::any::Any;

/// Snappy compression strategy implementation
/// 
/// Uses the Snappy compression algorithm developed by Google, which
/// focuses on very high speed at the cost of compression ratio.
#[derive(Debug, Clone)]
pub struct SnappyCompression {
    /// Block size for compression operations
    pub block_size: usize,
}

impl Default for SnappyCompression {
    fn default() -> Self {
        Self {
            block_size: 1024, // Default block size
        }
    }
}

impl CompressionStrategy for SnappyCompression {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        // Create a new Snappy encoder
        let mut encoder = Encoder::new();
        
        // Compress the data
        encoder.compress_vec(data)
            .map_err(|e| Error::CompressionFailed(format!("Snappy compression error: {}", e)))
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        // Create a new Snappy decoder
        let mut decoder = Decoder::new();
        
        // Decompress the data
        decoder.decompress_vec(data)
            .map_err(|e| Error::CompressionFailed(format!("Snappy decompression error: {}", e)))
    }

    fn estimate_compressed_size(&self, data: &[u8]) -> usize {
        if data.is_empty() {
            return 0;
        }
        
        // Snappy typically compresses to 70-90% of original size for typical data
        // This is a rough estimate, actual results will vary by data type
        (data.len() * 80) / 100
    }
    
    fn clone_box(&self) -> Box<dyn CompressionStrategy> {
        Box::new(self.clone())
    }
    
    fn name(&self) -> &'static str {
        "snappy"
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
    fn test_snappy_compression_empty_data() {
        let compression = SnappyCompression::default();
        let empty: &[u8] = &[];
        
        let compressed = compression.compress(empty).unwrap();
        assert!(compressed.is_empty());
        
        let decompressed = compression.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_snappy_compression_small_data() {
        let compression = SnappyCompression::default();
        let data = b"test data for compression";
        
        let compressed = compression.compress(data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(&decompressed, data);
    }

    #[test]
    fn test_snappy_compression_large_data() {
        let compression = SnappyCompression::default();
        
        // Create data with high redundancy for good compression
        let mut data = Vec::with_capacity(10000);
        for _ in 0..100 {
            data.extend_from_slice(b"The quick brown fox jumps over the lazy dog. ");
        }
        
        let compressed = compression.compress(&data).unwrap();
        println!("Original size: {}, Compressed size: {}", data.len(), compressed.len());
        
        // Should achieve some compression with repetitive data
        assert!(compressed.len() < data.len());
        
        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_snappy_compression_binary_data() {
        let compression = SnappyCompression::default();
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_snappy_compression_random_data() {
        let compression = SnappyCompression::default();
        let data: Vec<u8> = (0..10000).map(|_| rand::random::<u8>()).collect();
        
        let compressed = compression.compress(&data).unwrap();
        println!("Random data - Original size: {}, Compressed size: {}", 
            data.len(), compressed.len());
        
        // Random data usually doesn't compress well
        // But the compression should be lossless
        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_snappy_corrupted_data() {
        let compression = SnappyCompression::default();
        let data = b"test data for compression that should be long enough to cause issues when corrupted";
        
        // Create compressed data then corrupt it severely
        let mut compressed = compression.compress(data).unwrap();
        if compressed.len() > 10 {
            // Corrupt the snappy header 
            compressed[0] = 0xFF;
            compressed[1] = 0xFF;
            
            // Create complete garbage in the middle of the stream
            let start_idx = compressed.len() / 3;
            for i in 0..4 {
                if start_idx + i < compressed.len() {
                    compressed[start_idx + i] = 0xFF;
                }
            }
            
            // Snappy has a length header at the beginning - corrupt it to indicate impossible length
            if compressed.len() >= 3 {
                compressed[2] = 0xFF;  // Make length field invalid
            }
            
            // Should fail to decompress corrupted data
            let result = compression.decompress(&compressed);
            assert!(result.is_err(), "Decompression should fail with corrupted data");
        }
    }
    
    #[test]
    fn test_snappy_incomplete_data() {
        let compression = SnappyCompression::default();
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
    fn test_snappy_performance() {
        let compression = SnappyCompression::default();
        
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
        
        println!("\nSnappy Performance:");
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
    
    #[test]
    fn test_snappy_vs_no_compression() {
        // This test compares Snappy against no compression for various data types
        let snappy = SnappyCompression::default();
        let noop = crate::run::compression::NoopCompression::default();
        
        // Create different types of test data
        let sequential_data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
        let random_data: Vec<u8> = (0..5000).map(|_| rand::random::<u8>()).collect();
        let mut repetitive_data = Vec::with_capacity(5000);
        for _ in 0..125 {
            repetitive_data.extend_from_slice(b"abcd");
        }
        
        // Test sequential data
        let start = Instant::now();
        let _ = snappy.compress(&sequential_data).unwrap();
        let snappy_time = start.elapsed();
        
        let start = Instant::now();
        let _ = noop.compress(&sequential_data).unwrap();
        let noop_time = start.elapsed();
        
        println!("\nCompression Time Comparison (Sequential data):");
        println!("Snappy: {:?}, None: {:?}, Speedup: {:.2}x", 
            snappy_time, noop_time, noop_time.as_nanos() as f64 / snappy_time.as_nanos() as f64);
        
        // Test random data
        let start = Instant::now();
        let _ = snappy.compress(&random_data).unwrap();
        let snappy_time = start.elapsed();
        
        let start = Instant::now();
        let _ = noop.compress(&random_data).unwrap();
        let noop_time = start.elapsed();
        
        println!("Compression Time Comparison (Random data):");
        println!("Snappy: {:?}, None: {:?}, Speedup: {:.2}x", 
            snappy_time, noop_time, noop_time.as_nanos() as f64 / snappy_time.as_nanos() as f64);
        
        // Test repetitive data
        let start = Instant::now();
        let _ = snappy.compress(&repetitive_data).unwrap();
        let snappy_time = start.elapsed();
        
        let start = Instant::now();
        let _ = noop.compress(&repetitive_data).unwrap();
        let noop_time = start.elapsed();
        
        println!("Compression Time Comparison (Repetitive data):");
        println!("Snappy: {:?}, None: {:?}, Speedup: {:.2}x", 
            snappy_time, noop_time, noop_time.as_nanos() as f64 / snappy_time.as_nanos() as f64);
    }
}