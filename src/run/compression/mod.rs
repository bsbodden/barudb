use crate::types::Result;
use std::fmt::Debug;

mod bit_pack;
mod delta;
mod dictionary;
mod noop;
mod lz4;
mod snappy;

pub use bit_pack::BitPackCompression;
pub use delta::DeltaCompression;
pub use dictionary::DictionaryCompression;
pub use noop::NoopCompression;
pub use lz4::Lz4Compression;
pub use snappy::SnappyCompression;

use std::any::Any;

/// Trait defining the interface for compression strategies.
/// 
/// This allows different compression algorithms to be used
/// interchangeably within the LSM tree.
pub trait CompressionStrategy: Send + Sync + Debug {
    /// Compress a byte array and return the compressed data
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>>;
    
    /// Decompress a previously compressed byte array
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>>;
    
    /// Estimate the compressed size of data without actually compressing it
    /// This is useful for making decisions about whether compression is worthwhile
    fn estimate_compressed_size(&self, data: &[u8]) -> usize;
    
    /// Clone this compression strategy as a boxed trait object
    fn clone_box(&self) -> Box<dyn CompressionStrategy>;
    
    /// Get the name of this compression strategy
    fn name(&self) -> &'static str;
    
    /// Convert to Any for downcasting support
    fn as_any(&mut self) -> &mut dyn Any;
}

/// Compression algorithm type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CompressionType {
    /// No compression (passthrough)
    None,
    /// Delta encoding for sequential integer data
    Delta,
    /// Bit-packing for integer data with limited range
    BitPack,
    /// Dictionary-based run-length encoding for repeated patterns
    Dictionary,
    /// LZ4 compression (fast with good compression ratio)
    Lz4,
    /// Snappy compression (very fast with moderate compression ratio)
    Snappy,
}

impl CompressionType {
    /// Convert compression type to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionType::None => "none",
            CompressionType::Delta => "delta",
            CompressionType::BitPack => "bit_pack",
            CompressionType::Dictionary => "dictionary",
            CompressionType::Lz4 => "lz4",
            CompressionType::Snappy => "snappy",
        }
    }
    
    /// Create compression type from string
    #[allow(dead_code)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "none" => Some(CompressionType::None),
            "delta" => Some(CompressionType::Delta),
            "bit_pack" => Some(CompressionType::BitPack),
            "dictionary" => Some(CompressionType::Dictionary),
            "lz4" => Some(CompressionType::Lz4),
            "snappy" => Some(CompressionType::Snappy),
            _ => None,
        }
    }
}

impl std::fmt::Display for CompressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for CompressionType {
    fn default() -> Self {
        CompressionType::None
    }
}

/// Factory for creating compression strategies
#[allow(dead_code)]
pub struct CompressionFactory;

impl CompressionFactory {
    /// Create a new compression strategy of the specified type
    #[allow(dead_code)]
    pub fn create(compression_type: CompressionType) -> Box<dyn CompressionStrategy> {
        match compression_type {
            CompressionType::None => Box::new(NoopCompression::default()),
            CompressionType::Delta => Box::new(DeltaCompression::default()),
            CompressionType::BitPack => Box::new(BitPackCompression::default()),
            CompressionType::Dictionary => Box::new(DictionaryCompression::default()),
            CompressionType::Lz4 => Box::new(Lz4Compression::default()),
            CompressionType::Snappy => Box::new(SnappyCompression::default()),
        }
    }
}

/// Configuration for compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Is compression enabled
    pub enabled: bool,
    
    /// Per-level compression settings (None = use default for that level)
    /// Index corresponds to level number (0-based)
    pub level_types: Vec<Option<CompressionType>>,
    
    /// Default compression for memtable
    pub memtable_default: CompressionType,
    
    /// Default compression for L0
    pub l0_default: CompressionType,
    
    /// Default compression for lower levels (L1+)
    pub lower_level_default: CompressionType,
    
    /// Block size for compression (number of KV pairs per block)
    pub block_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level_types: vec![None; 10], // Support up to 10 levels by default
            memtable_default: CompressionType::None, // No compression for in-memory
            // Keep using BitPack as default for now to maintain compatibility with tests
            l0_default: CompressionType::BitPack, // Moderate compression for L0
            lower_level_default: CompressionType::BitPack, // More aggressive for lower levels
            block_size: 1024, // Default block size for compression
        }
    }
}

impl CompressionConfig {
    /// Get the compression type to use for a specific level
    pub fn get_for_level(&self, level: usize) -> CompressionType {
        if !self.enabled {
            return CompressionType::None;
        }
        
        if level < self.level_types.len() {
            // If a specific setting exists for this level, use it
            if let Some(compression_type) = self.level_types[level] {
                return compression_type;
            }
        }
        
        // Otherwise use default based on level category
        match level {
            0 => self.l0_default,
            _ => self.lower_level_default,
        }
    }
    
    /// Create a compression strategy for the given level
    pub fn create_strategy_for_level(&self, level: usize) -> Box<dyn CompressionStrategy> {
        let compression_type = self.get_for_level(level);
        let mut strategy = CompressionFactory::create(compression_type);
        
        // Configure block size if the strategy supports it
        if let Some(bit_pack) = strategy.as_any().downcast_mut::<BitPackCompression>() {
            bit_pack.block_size = self.block_size;
        }
        
        strategy
    }
}

/// Configuration for adaptive compression
#[derive(Debug, Clone)]
pub struct AdaptiveCompressionConfig {
    /// Is adaptive compression enabled?
    pub enabled: bool,
    
    /// Sample size for analyzing data
    pub sample_size: usize,
    
    /// Minimum size for applying compression (smaller blocks remain uncompressed)
    pub min_size_threshold: usize,
    
    /// Whether to use level-specific compression choices
    pub level_aware: bool,
    
    /// Minimum compression ratio to apply compression (1.0 = no improvement)
    pub min_compression_ratio: f64,
    
    /// Block size for compression (number of KV pairs per block)
    pub block_size: usize,
}

impl Default for AdaptiveCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_size: 100,
            min_size_threshold: 128,
            level_aware: true,
            min_compression_ratio: 1.05, // Only compress if we get at least 5% improvement
            block_size: 1024,
        }
    }
}

/// Statistics about compression performance for evaluation
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Name of the compression strategy used
    pub strategy_name: String,
    
    /// Original data size in bytes
    pub original_size: usize,
    
    /// Compressed data size in bytes
    pub compressed_size: usize,
    
    /// Compression ratio (original_size / compressed_size)
    pub compression_ratio: f64,
    
    /// Time to compress in milliseconds
    pub compression_time_ms: f64,
    
    /// Time to decompress in milliseconds
    pub decompression_time_ms: f64,
}

/// Adaptive compression strategy that selects the best algorithm based on data characteristics
#[derive(Debug)]
pub struct AdaptiveCompression {
    /// Configuration options
    config: AdaptiveCompressionConfig,
}

impl AdaptiveCompression {
    /// Create a new adaptive compression with default strategies
    #[allow(dead_code)]
    pub fn new(config: AdaptiveCompressionConfig) -> Self {
        Self { config }
    }
    
    /// Select the best compression strategy for the given data
    #[allow(dead_code)]
    pub fn select_best_strategy(&self, data: &[u8], level: usize) -> Box<dyn CompressionStrategy> {
        // If data is too small, don't compress
        if data.len() < self.config.min_size_threshold {
            return Box::new(NoopCompression::default());
        }
        
        // Take a sample of the data for analysis
        let sample = if data.len() <= self.config.sample_size {
            data
        } else {
            // Simple sampling: take first N bytes
            &data[0..self.config.sample_size]
        };
        
        // Analyze sample to determine data characteristics
        let (is_sequential, range, repetition) = self.analyze_sample(sample);
        
        // Apply level-specific heuristics if enabled
        if self.config.level_aware {
            // Lower levels (higher level number) benefit from more aggressive compression
            if level > 2 {
                // For deep levels, prioritize compression ratio
                if is_sequential {
                    return Box::new(DeltaCompression::default());
                } else if range < 1000 {
                    return Box::new(BitPackCompression::default());
                } else if repetition > 0.5 {
                    return Box::new(DictionaryCompression::default());
                } else {
                    // Use LZ4 for general purpose compression with good ratio
                    return Box::new(Lz4Compression::default());
                }
            } else if level == 2 {
                // For middle levels, balance speed and compression
                if is_sequential {
                    return Box::new(DeltaCompression::default());
                } else {
                    // Snappy offers good speed with reasonable compression
                    return Box::new(SnappyCompression::default());
                }
            } else if level <= 1 {
                // For top levels, prioritize speed
                // Use Snappy for level 1 (fast with some compression)
                if level == 1 {
                    return Box::new(SnappyCompression::default());
                } else {
                    // Use no compression for level 0 (fastest)
                    return Box::new(NoopCompression::default());
                }
            }
        }
        
        // General heuristics based on data characteristics
        if is_sequential {
            Box::new(DeltaCompression::default())
        } else if range < 10000 {
            Box::new(BitPackCompression::default())
        } else if repetition > 0.3 {
            Box::new(DictionaryCompression::default())
        } else if data.len() > 10240 {
            // For larger data blocks, use LZ4 for better compression ratio
            Box::new(Lz4Compression::default())
        } else {
            // For smaller blocks, use Snappy for speed
            Box::new(SnappyCompression::default())
        }
    }
    
    /// Analyze a data sample to determine key characteristics
    /// Returns (is_sequential, value_range, repetition_ratio)
    #[allow(dead_code)]
    fn analyze_sample(&self, sample: &[u8]) -> (bool, u64, f64) {
        // Ensure we're working with integer data (assuming 4-byte ints)
        if sample.len() < 8 || sample.len() % 8 != 0 {
            return (false, u64::MAX, 0.0);
        }
        
        let mut values = Vec::with_capacity(sample.len() / 8);
        let mut keys = Vec::with_capacity(sample.len() / 8);
        
        // Extract keys and values
        let mut i = 0;
        while i < sample.len() {
            if i + 8 <= sample.len() {
                let key_bytes = [sample[i], sample[i+1], sample[i+2], sample[i+3]];
                let val_bytes = [sample[i+4], sample[i+5], sample[i+6], sample[i+7]];
                
                let key = i32::from_le_bytes(key_bytes);
                let val = i32::from_le_bytes(val_bytes);
                
                keys.push(key);
                values.push(val);
            }
            i += 8;
        }
        
        // Calculate sequentiality score
        let mut is_sequential = true;
        if keys.len() > 1 {
            let first_diff = (keys[1] - keys[0]).abs();
            for i in 1..keys.len() - 1 {
                if (keys[i+1] - keys[i]).abs() != first_diff {
                    is_sequential = false;
                    break;
                }
            }
        }
        
        // Calculate value range
        let min_val = keys.iter().min().unwrap_or(&0);
        let max_val = keys.iter().max().unwrap_or(&0);
        let range = (*max_val as i64 - *min_val as i64).abs() as u64;
        
        // Calculate repetition score (how many values repeat)
        let mut unique_values = std::collections::HashSet::new();
        for &key in &keys {
            unique_values.insert(key);
        }
        for &val in &values {
            unique_values.insert(val);
        }
        
        let total_values = keys.len() + values.len();
        let repetition = 1.0 - (unique_values.len() as f64 / total_values as f64);
        
        (is_sequential, range, repetition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compression_type_conversion() {
        let types = vec![
            CompressionType::None,
            CompressionType::Delta,
            CompressionType::BitPack,
            CompressionType::Dictionary,
        ];
        
        for compression_type in types {
            let s = compression_type.as_str();
            let converted = CompressionType::from_str(s).unwrap();
            assert_eq!(compression_type, converted);
        }
    }
    
    #[test]
    fn test_compression_type_default() {
        assert_eq!(CompressionType::default(), CompressionType::None);
    }
}