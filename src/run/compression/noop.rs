use crate::run::compression::CompressionStrategy;
use crate::types::Result;
use std::any::Any;

/// A compression strategy that doesn't actually compress data
/// Used as a baseline for comparison and when compression would be counterproductive
#[derive(Debug, Default, Clone)]
pub struct NoopCompression;

impl CompressionStrategy for NoopCompression {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(data.to_vec())
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(data.to_vec())
    }

    fn estimate_compressed_size(&self, data: &[u8]) -> usize {
        data.len()
    }
    
    fn clone_box(&self) -> Box<dyn CompressionStrategy> {
        Box::new(Self)
    }
    
    fn name(&self) -> &'static str {
        "noop"
    }
    
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_compression() {
        let compression = NoopCompression::default();

        // Test empty data
        let empty: &[u8] = &[];
        assert_eq!(compression.compress(empty).unwrap(), empty);
        assert_eq!(compression.decompress(empty).unwrap(), empty);
        assert_eq!(compression.estimate_compressed_size(empty), 0);

        // Test actual data
        let data = b"test data for compression";
        let compressed = compression.compress(data).unwrap();
        assert_eq!(&compressed, data);

        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(&decompressed, data);

        assert_eq!(compression.estimate_compressed_size(data), data.len());
    }

    #[test]
    fn test_compression_large_data() {
        let compression = NoopCompression::default();
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        let compressed = compression.compress(&data).unwrap();
        assert_eq!(compressed, data);

        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);

        assert_eq!(compression.estimate_compressed_size(&data), 1000);
    }
}