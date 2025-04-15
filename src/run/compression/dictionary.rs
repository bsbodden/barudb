use crate::run::compression::CompressionStrategy;
use crate::types::{Result, Error, Key, Value};
use std::any::Any;
use std::cmp::min;
use std::collections::HashMap;

/// Dictionary-based compression for integer data
/// 
/// Identifies repeated values or patterns and replaces them with shorter codes.
/// This is particularly effective for data with many repeated values.
#[derive(Debug, Clone)]
pub struct DictionaryCompression {
    /// Maximum number of values per block
    pub block_size: usize,
    /// Maximum size of the dictionary (most frequent entries)
    pub max_dict_size: usize,
}

impl Default for DictionaryCompression {
    fn default() -> Self {
        Self {
            block_size: 1024,
            max_dict_size: 0x7FFF, // Max possible size (32767) since we reserve the high bit for literal entries
        }
    }
}

/// Header for a dictionary-compressed block
#[derive(Debug)]
struct DictionaryHeader {
    /// Number of entries in the dictionary
    dict_size: u16,
    /// Number of key-value pairs in this block
    entry_count: u32,
}

impl CompressionStrategy for DictionaryCompression {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Check that data size is a multiple of 16 (Key + Value = 16 bytes with 64-bit integers)
        if data.len() % 16 != 0 {
            return Err(Error::InvalidInput("Data size must be a multiple of 16 bytes".to_string()));
        }
        
        // Count entries
        let entry_count = data.len() / 16;
        if entry_count == 0 {
            return Ok(Vec::new());
        }
        
        // Extract key-value pairs
        let mut entries = Vec::with_capacity(entry_count);
        
        for i in 0..entry_count {
            let offset = i * 16;
            
            let key_bytes = [
                data[offset], 
                data[offset + 1], 
                data[offset + 2], 
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7]
            ];
            
            let val_bytes = [
                data[offset + 8], 
                data[offset + 9], 
                data[offset + 10], 
                data[offset + 11],
                data[offset + 12],
                data[offset + 13],
                data[offset + 14],
                data[offset + 15]
            ];
            
            let key = Key::from_le_bytes(key_bytes);
            let val = Value::from_le_bytes(val_bytes);
            
            entries.push((key, val));
        }
        
        // Process data in blocks
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < entry_count {
            let block_end = min(i + self.block_size, entry_count);
            let block_entries = &entries[i..block_end];
            
            let block_data = self.compress_block(block_entries)?;
            
            // Add block size (4 bytes) + block data
            result.extend_from_slice(&(block_data.len() as u32).to_le_bytes());
            result.extend_from_slice(&block_data);
            
            i = block_end;
        }
        
        Ok(result)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        while offset + 4 <= data.len() {
            // Read block size
            let size_bytes = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
            let block_size = u32::from_le_bytes(size_bytes) as usize;
            offset += 4;
            
            // Ensure we have enough data
            if offset + block_size > data.len() {
                return Err(Error::InvalidData("Incomplete compressed data block".to_string()));
            }
            
            // Decompress block
            let block_data = &data[offset..offset + block_size];
            let decompressed = self.decompress_block(block_data)?;
            result.extend_from_slice(&decompressed);
            
            offset += block_size;
        }
        
        Ok(result)
    }

    fn estimate_compressed_size(&self, data: &[u8]) -> usize {
        if data.len() < 8 {
            return data.len();
        }
        
        // Rough estimate based on number of entries
        let entry_count = data.len() / 16;
        
        // Dictionary size is roughly 16 bytes per dictionary entry (8 for key, 8 for value)
        let dict_size = min(self.max_dict_size, entry_count / 4) * 16;
        
        // Compressed data is roughly index bytes (1-2 bytes per entry) + dictionary
        let compressed_size = entry_count * 2 + dict_size;
        
        // Add overhead for block headers
        let num_blocks = (entry_count + self.block_size - 1) / self.block_size;
        let header_size = num_blocks * 10; // 4 bytes block size + 6 bytes header
        
        compressed_size + header_size
    }
    
    fn clone_box(&self) -> Box<dyn CompressionStrategy> {
        Box::new(self.clone())
    }
    
    fn name(&self) -> &'static str {
        "dictionary"
    }
    
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

impl DictionaryCompression {
    /// Compress a single block of key-value entries
    fn compress_block(&self, entries: &[(Key, Value)]) -> Result<Vec<u8>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        
        // Count frequencies of key-value pairs
        let mut frequencies = HashMap::new();
        for &entry in entries {
            *frequencies.entry(entry).or_insert(0) += 1;
        }
        
        // Sort by frequency
        let mut freq_vec: Vec<_> = frequencies.into_iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency (descending)
        
        // Create dictionary with most frequent entries
        let dict_size = min(self.max_dict_size, freq_vec.len());
        let mut dictionary = Vec::with_capacity(dict_size);
        
        for i in 0..dict_size {
            dictionary.push(freq_vec[i].0);
        }
        
        // Map entries to dictionary indices
        let mut index_map = HashMap::new();
        for (i, &entry) in dictionary.iter().enumerate() {
            index_map.insert(entry, i as u16);
        }
        
        // Create header
        let header = DictionaryHeader {
            dict_size: dictionary.len() as u16,
            entry_count: entries.len() as u32,
        };
        
        // Allocate result vector with space for header, dictionary, and encoded indices
        let header_size = 6; // 2 bytes dict_size + 4 bytes entry_count
        let dict_bytes = dictionary.len() * 8;
        let indices_bytes = entries.len() * 2; // 2 bytes per index (supporting larger dictionaries)
        
        let mut result = Vec::with_capacity(header_size + dict_bytes + indices_bytes);
        
        // Add header
        result.extend_from_slice(&header.dict_size.to_le_bytes());
        result.extend_from_slice(&header.entry_count.to_le_bytes());
        
        // Add dictionary
        for &(key, value) in &dictionary {
            result.extend_from_slice(&key.to_le_bytes());
            result.extend_from_slice(&value.to_le_bytes());
        }
        
        // Add encoded indices
        for &entry in entries {
            if let Some(&index) = index_map.get(&entry) {
                // Dictionary entry - encode as index
                // Ensure index is less than 0x8000 (the high bit used for the literal marker)
                assert!(index < 0x8000, "Dictionary index must be less than 0x8000");
                result.extend_from_slice(&index.to_le_bytes());
            } else {
                // Literal entry - encode with high bit set in the first 2 bytes
                let literal_marker: u16 = 0x8000; // High bit set to indicate literal
                result.extend_from_slice(&literal_marker.to_le_bytes());
                result.extend_from_slice(&entry.0.to_le_bytes());
                result.extend_from_slice(&entry.1.to_le_bytes());
            }
        }
        
        Ok(result)
    }
    
    /// Decompress a single block
    fn decompress_block(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 6 {
            return Err(Error::InvalidData("Invalid dictionary block: too small".to_string()));
        }
        
        // Read header
        let dict_size = u16::from_le_bytes([data[0], data[1]]);
        let entry_count = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);
        
        let header_size = 6;
        let dict_bytes = dict_size as usize * 16;
        
        if dict_size == 0 || entry_count == 0 {
            return Err(Error::InvalidData("Invalid dictionary header values".to_string()));
        }
        
        if header_size + dict_bytes > data.len() {
            return Err(Error::InvalidData("Dictionary data truncated".to_string()));
        }
        
        // Read dictionary
        let mut dictionary = Vec::with_capacity(dict_size as usize);
        for i in 0..dict_size as usize {
            let offset = header_size + i * 16;
            
            let key = Key::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]
            ]);
            
            let value = Value::from_le_bytes([
                data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11],
                data[offset + 12], data[offset + 13], data[offset + 14], data[offset + 15]
            ]);
            
            dictionary.push((key, value));
        }
        
        // Prepare result buffer
        let result_size = entry_count as usize * 16; // Key (8) + Value (8) = 16 bytes per entry
        let mut result = Vec::with_capacity(result_size);
        
        // Read and decode indices
        let mut pos = header_size + dict_bytes;
        
        for _ in 0..entry_count {
            if pos + 2 > data.len() {
                return Err(Error::InvalidData("Unexpected end of dictionary data".to_string()));
            }
            
            let index = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;
            
            if index & 0x8000 != 0 {
                // Literal entry (high bit set)
                if pos + 16 > data.len() {
                    return Err(Error::InvalidData("Unexpected end of dictionary data for literal entry".to_string()));
                }
                
                // Read literal key and value with 64-bit integers
                let key_bytes = [
                    data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
                    data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]
                ];
                pos += 8;
                
                let value_bytes = [
                    data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
                    data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]
                ];
                pos += 8;
                
                // No need to convert to Key/Value and back, just add the bytes directly
                result.extend_from_slice(&key_bytes);
                result.extend_from_slice(&value_bytes);
            } else {
                // Dictionary reference
                let dict_index = index as usize;
                if dict_index >= dictionary.len() {
                    return Err(Error::InvalidData(format!(
                        "Invalid dictionary index: {} (dictionary size: {})", 
                        dict_index, dictionary.len()
                    )));
                }
                
                let (key, value) = dictionary[dict_index];
                
                // Add to result
                result.extend_from_slice(&key.to_le_bytes());
                result.extend_from_slice(&value.to_le_bytes());
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Helper function to create test data with keys and values
    fn create_test_data(entries: &[(Key, Value)]) -> Vec<u8> {
        let mut data = Vec::with_capacity(entries.len() * 16);
        for &(key, value) in entries {
            data.extend_from_slice(&key.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
        }
        data
    }
    
    #[test]
    fn test_dictionary_empty_data() {
        let compression = DictionaryCompression::default();
        let empty: &[u8] = &[];
        
        let compressed = compression.compress(empty).unwrap();
        assert_eq!(compressed.len(), 0);
        
        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 0);
    }
    
    #[test]
    fn test_dictionary_single_entry() {
        let compression = DictionaryCompression::default();
        let entries = [(42, 100)];
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_dictionary_repeated_entries() {
        let compression = DictionaryCompression::default();
        
        // Lots of repetition - ideal for dictionary compression
        let mut entries = Vec::new();
        
        for _ in 0..20 {
            entries.push((1, 100));
            entries.push((2, 200));
            entries.push((3, 300));
            entries.push((4, 400));
            entries.push((5, 500));
        }
        
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_dictionary_unique_entries() {
        let compression = DictionaryCompression::default();
        
        // All unique entries - dictionary won't help much
        let entries: Vec<(Key, Value)> = (0..100).map(|i| (i, i)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_dictionary_mixed_data() {
        let compression = DictionaryCompression::default();
        
        // Mix of repeated and unique entries
        let mut entries = Vec::new();
        
        // Add some repeated entries
        for _ in 0..30 {
            entries.push((1, 100));
            entries.push((2, 200));
        }
        
        // Add some unique entries
        for i in 0..40 {
            entries.push((100 + i, 1000 + i));
        }
        
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_dictionary_small_dict_size() {
        // Create dictionary with only 10 entries
        let compression = DictionaryCompression {
            block_size: 1024,
            max_dict_size: 10,
        };
        
        // Create data with many different values
        let entries: Vec<(Key, Value)> = (0..100).map(|i| (i, i * 10)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_dictionary_multiple_blocks() {
        let mut compression = DictionaryCompression::default();
        compression.block_size = 20; // Small block size to test multiple blocks
        
        // Create data with repetition in each block
        let mut entries = Vec::new();
        
        for block in 0..5 {
            let base = block * 100;
            
            // Add repeated entries for this block
            for _ in 0..10 {
                entries.push((base + 1, base + 100));
                entries.push((base + 2, base + 200));
            }
            
            // Add some unique entries
            for i in 0..5 {
                entries.push((base + 10 + i, base + 1000 + i));
            }
        }
        
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_dictionary_extreme_values() {
        let compression = DictionaryCompression::default();
        
        // Test with extreme 64-bit integers, including ones with high bits set
        let entries = vec![
            (i64::MIN, i64::MAX),
            (i64::MAX, i64::MIN),
            (0, i64::MAX),
            (i64::MIN, 0),
            (-1, -1),
            (1 << 63, 1 << 62),
            (-(1 << 60), -(1 << 59)),
            // Add repeated entries to test dictionary functionality
            (42, 42),
            (42, 42),
            (42, 42),
            (i64::MAX - 1, i64::MIN + 1),
            (i64::MAX - 1, i64::MIN + 1),
            (i64::MAX - 1, i64::MIN + 1),
        ];
        
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
}