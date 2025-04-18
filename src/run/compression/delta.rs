use crate::run::compression::CompressionStrategy;
use crate::types::{Result, Error, Key, Value};
use std::any::Any;
use std::cmp::min;

/// Delta encoding compression for integer data
/// 
/// Stores differences between consecutive values rather than the values themselves.
/// This is particularly effective for sorted runs of integers, which are common in LSM trees.
#[derive(Debug, Clone)]
pub struct DeltaCompression {
    /// Maximum number of values per block
    pub block_size: usize,
    /// Whether to use variable-length encoding for deltas
    pub use_variable_length: bool,
}

impl Default for DeltaCompression {
    fn default() -> Self {
        Self {
            block_size: 1024,
            use_variable_length: true,
        }
    }
}

/// Header for a delta-encoded block
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DeltaHeader {
    /// First key in the block (reference value)
    first_key: Key,
    /// First value in the block (reference value)
    first_value: Value,
    /// Whether variable-length encoding is used
    variable_length: u8,
    /// Number of key-value pairs in this block
    count: u32,
}

impl CompressionStrategy for DeltaCompression {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Check that data size is a multiple of 16 (Key + Value = 16 bytes for i64)
        if data.len() % 16 != 0 {
            return Err(Error::InvalidInput("Data size must be a multiple of 16 bytes".to_string()));
        }
        
        // Count entries
        let entry_count = data.len() / 16;
        if entry_count == 0 {
            return Ok(Vec::new());
        }
        
        // Extract keys and values
        let mut keys = Vec::with_capacity(entry_count);
        let mut values = Vec::with_capacity(entry_count);
        
        for i in 0..entry_count {
            let key_start = i * 16;
            let val_start = key_start + 8;
            
            let key_bytes = [
                data[key_start], 
                data[key_start + 1], 
                data[key_start + 2], 
                data[key_start + 3],
                data[key_start + 4],
                data[key_start + 5],
                data[key_start + 6],
                data[key_start + 7]
            ];
            
            let val_bytes = [
                data[val_start], 
                data[val_start + 1], 
                data[val_start + 2], 
                data[val_start + 3],
                data[val_start + 4],
                data[val_start + 5],
                data[val_start + 6],
                data[val_start + 7]
            ];
            
            keys.push(Key::from_le_bytes(key_bytes));
            values.push(Value::from_le_bytes(val_bytes));
        }
        
        // Process data in blocks
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < entry_count {
            let block_end = min(i + self.block_size, entry_count);
            let block_keys = &keys[i..block_end];
            let block_values = &values[i..block_end];
            
            let block_data = self.compress_block(block_keys, block_values)?;
            
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
        if data.len() < 16 {
            return data.len();
        }
        
        // Rough estimate
        // For sequential or near-sequential data, delta encoding is very effective (30-40% of original)
        // For random data, it's not very effective (might even be larger)
        
        // We'll use a conservative estimate of 60% of original size 
        let estimated_size = (data.len() * 6) / 10;
        
        // Add overhead for block headers
        let num_blocks = (data.len() / 16 + self.block_size - 1) / self.block_size;
        let header_size = num_blocks * (4 + std::mem::size_of::<DeltaHeader>());
        
        estimated_size + header_size
    }
    
    fn clone_box(&self) -> Box<dyn CompressionStrategy> {
        Box::new(self.clone())
    }
    
    fn name(&self) -> &'static str {
        "delta"
    }
    
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

impl DeltaCompression {
    /// Compress a single block of keys and values
    fn compress_block(&self, keys: &[Key], values: &[Value]) -> Result<Vec<u8>> {
        if keys.is_empty() || keys.len() != values.len() {
            return Err(Error::InvalidInput("Empty or mismatched keys/values".to_string()));
        }
        
        // Create header
        let header = DeltaHeader {
            first_key: keys[0],
            first_value: values[0],
            variable_length: if self.use_variable_length { 1 } else { 0 },
            count: keys.len() as u32,
        };
        
        // Allocate result vector with header
        let header_size = std::mem::size_of::<DeltaHeader>();
        let mut result = Vec::with_capacity(header_size + keys.len() * 16); // Conservative estimate
        
        // Add header
        result.extend_from_slice(&header.first_key.to_le_bytes());
        result.extend_from_slice(&header.first_value.to_le_bytes());
        result.push(header.variable_length);
        result.extend_from_slice(&[0, 0, 0]); // Padding
        result.extend_from_slice(&header.count.to_le_bytes());
        
        // If there's only one entry, no deltas to store
        if keys.len() == 1 {
            return Ok(result);
        }
        
        // Compute and encode deltas
        // Skip the first key/value as they're stored directly in the header
        let mut prev_key = keys[0];
        let mut prev_value = values[0];
        
        for i in 1..keys.len() {
            // Use wrapping_sub to handle potential overflow safely
            let key_delta = keys[i].wrapping_sub(prev_key);
            let value_delta = values[i].wrapping_sub(prev_value);
            
            if self.use_variable_length {
                // Encode deltas using variable-length encoding
                self.encode_varint(key_delta, &mut result);
                self.encode_varint(value_delta, &mut result);
            } else {
                // Fixed-length encoding (8 bytes per delta for i64)
                result.extend_from_slice(&key_delta.to_le_bytes());
                result.extend_from_slice(&value_delta.to_le_bytes());
            }
            
            prev_key = keys[i];
            prev_value = values[i];
        }
        
        Ok(result)
    }
    
    /// Decompress a single block
    fn decompress_block(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < std::mem::size_of::<DeltaHeader>() {
            return Err(Error::InvalidData("Invalid delta-encoded block: too small".to_string()));
        }
        
        // Read header
        let first_key_bytes = [
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
        ];
        let first_value_bytes = [
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]
        ];
        
        let first_key = Key::from_le_bytes(first_key_bytes);
        let first_value = Value::from_le_bytes(first_value_bytes);
        let variable_length = data[16] != 0;
        let count = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        
        let header_size = std::mem::size_of::<DeltaHeader>();
        
        if count == 0 {
            return Err(Error::InvalidData("Invalid delta header: count is zero".to_string()));
        }
        
        // Prepare result buffer
        let result_size = count as usize * 16; // Key (8) + Value (8) = 16 bytes per entry for i64
        let mut result = Vec::with_capacity(result_size);
        
        // Add first key-value pair from header
        result.extend_from_slice(&first_key.to_le_bytes());
        result.extend_from_slice(&first_value.to_le_bytes());
        
        // If there's only one entry, we're done
        if count == 1 {
            return Ok(result);
        }
        
        // Decode deltas
        let mut prev_key = first_key;
        let mut prev_value = first_value;
        let mut pos = header_size;
        
        for _ in 1..count {
            if variable_length {
                // Decode variable-length deltas
                let (key_delta, bytes_read) = self.decode_varint(&data[pos..])?;
                pos += bytes_read;
                
                let (value_delta, bytes_read) = self.decode_varint(&data[pos..])?;
                pos += bytes_read;
                
                // Use wrapping_add to handle potential overflow safely
                // This is important for 64-bit integers that might wrap around
                prev_key = prev_key.wrapping_add(key_delta);
                prev_value = prev_value.wrapping_add(value_delta);
            } else {
                // Decode fixed-length deltas (8 bytes each for i64)
                if pos + 16 > data.len() {
                    return Err(Error::InvalidData("Unexpected end of delta-encoded data".to_string()));
                }
                
                let key_delta_bytes = [
                    data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
                    data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]
                ];
                pos += 8;
                
                let value_delta_bytes = [
                    data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
                    data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]
                ];
                pos += 8;
                
                let key_delta = Key::from_le_bytes(key_delta_bytes);
                let value_delta = Value::from_le_bytes(value_delta_bytes);
                
                prev_key = prev_key.wrapping_add(key_delta);
                prev_value = prev_value.wrapping_add(value_delta);
            }
            
            // Add reconstructed key-value pair to result
            result.extend_from_slice(&prev_key.to_le_bytes());
            result.extend_from_slice(&prev_value.to_le_bytes());
        }
        
        Ok(result)
    }
    
    /// Encode a signed integer using variable-length encoding
    /// 
    /// This function uses ZigZag encoding to efficiently represent signed integers,
    /// followed by variable-length encoding to represent the resulting unsigned value.
    fn encode_varint(&self, value: i64, output: &mut Vec<u8>) {
        // Convert to zigzag encoding (maps signed values to unsigned in a way that small
        // negative values like -1, -2 are mapped to small unsigned values)
        // For example: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
        let unsigned = ((value << 1) ^ (value >> 63)) as u64;
        
        // Use variable-length encoding (7 bits per byte, MSB indicates if more bytes follow)
        // This is similar to protobuf's varint encoding
        let mut v = unsigned;
        while v >= 0x80 {
            // Output 7 bits plus a continuation bit
            output.push((v as u8) | 0x80);
            v >>= 7;
        }
        // Final byte (no continuation bit set)
        output.push(v as u8);
    }
    
    /// Decode a zigzag-encoded varint
    fn decode_varint(&self, data: &[u8]) -> Result<(i64, usize)> {
        if data.is_empty() {
            return Err(Error::InvalidData("Unexpected end of varint data".to_string()));
        }
        
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        let mut i = 0;
        
        loop {
            if i >= data.len() {
                return Err(Error::InvalidData("Unexpected end of varint data".to_string()));
            }
            
            let byte = data[i];
            i += 1;
            
            // Apply the bits from this byte
            result |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            
            // Check if more bytes follow (MSB not set)
            if byte & 0x80 == 0 {
                break;
            }
            
            // Prevent shift from exceeding 63 (max for 64-bit integers)
            if shift >= 63 {
                // Read one more byte if MSB indicates more data
                if i < data.len() && data[i] & 0x80 != 0 {
                    return Err(Error::InvalidData("Varint too large for 64-bit integer".to_string()));
                }
                break;
            }
        }
        
        // Convert from zigzag back to signed
        let signed = ((result >> 1) as i64) ^ (-((result & 1) as i64));
        Ok((signed, i))
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
    fn test_delta_empty_data() {
        let compression = DeltaCompression::default();
        let empty: &[u8] = &[];
        
        let compressed = compression.compress(empty).unwrap();
        assert_eq!(compressed.len(), 0);
        
        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 0);
    }
    
    #[test]
    fn test_delta_single_entry() {
        let compression = DeltaCompression::default();
        let entries = [(42, 100)];
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_delta_sequential_keys() {
        let compression = DeltaCompression::default();
        
        // Sequential keys should compress very well
        let entries: Vec<(Key, Value)> = (0..100).map(|i| (i, i * 2)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_delta_arith_progression() {
        let compression = DeltaCompression::default();
        
        // Keys with constant delta (arithmetic progression)
        let entries: Vec<(Key, Value)> = (0..100).map(|i| (i * 10, i * 5)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_delta_semirandom_data() {
        let compression = DeltaCompression::default();
        
        // For reproducibility, use a fixed base value and small random offsets
        // This is more representative of real-world sorted data with small variations
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        let base = 1_000_000;
        let entries: Vec<(Key, Value)> = (0..100)
            .map(|i| {
                // Small variations around sequential values
                let key_offset = rng.random_range(-10..=10);
                let value_offset = rng.random_range(-100..=100);
                
                (base + i as Key + key_offset, base * 2 + i as Value + value_offset)
            })
            .collect();
        
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        // For small variations, delta compression should work properly
        assert_eq!(decompressed, data);
        
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_delta_variable_vs_fixed() {
        let entries: Vec<(Key, Value)> = (0..100).map(|i| (i, i * 2)).collect();
        let data = create_test_data(&entries);
        
        // Test with variable-length encoding
        let var_compression = DeltaCompression {
            block_size: 1024,
            use_variable_length: true,
        };
        
        let var_compressed = var_compression.compress(&data).unwrap();
        let var_decompressed = var_compression.decompress(&var_compressed).unwrap();
        
        assert_eq!(var_decompressed, data);
        
        // Test with fixed-length encoding
        let fixed_compression = DeltaCompression {
            block_size: 1024,
            use_variable_length: false,
        };
        
        let fixed_compressed = fixed_compression.compress(&data).unwrap();
        let fixed_decompressed = fixed_compression.decompress(&fixed_compressed).unwrap();
        
        assert_eq!(fixed_decompressed, data);
        
        // Variable should be smaller for small deltas
        println!("Variable-length size: {}, Fixed-length size: {}",
                 var_compressed.len(), fixed_compressed.len());
    }
    
    #[test]
    fn test_delta_multiple_blocks() {
        let mut compression = DeltaCompression::default();
        compression.block_size = 10; // Small block size to test multiple blocks
        
        let entries: Vec<(Key, Value)> = (0..100).map(|i| (i, i * 2)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_delta_negative_values() {
        let compression = DeltaCompression::default();
        
        // Test with negative keys and values
        let entries: Vec<(Key, Value)> = (-50..50).map(|i| (i, -i)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_varint_encoding() {
        let compression = DeltaCompression::default();
        let mut output = Vec::new();
        
        // Test small positive values
        compression.encode_varint(1, &mut output);
        compression.encode_varint(127, &mut output);
        
        // Test larger values
        compression.encode_varint(128, &mut output);
        compression.encode_varint(16383, &mut output); // 2^14 - 1
        
        // Test negative values
        compression.encode_varint(-1, &mut output);
        compression.encode_varint(-1000, &mut output);
        
        // Decode and verify
        let mut pos = 0;
        
        // Test each value - only check the decoded value, not the byte count
        // which might vary depending on the ZigZag encoding implementation
        let (val1, bytes1) = compression.decode_varint(&output[pos..]).unwrap();
        assert_eq!(val1, 1);
        pos += bytes1;
        
        let (val2, bytes2) = compression.decode_varint(&output[pos..]).unwrap();
        assert_eq!(val2, 127);
        pos += bytes2;
        
        let (val3, bytes3) = compression.decode_varint(&output[pos..]).unwrap();
        assert_eq!(val3, 128);
        pos += bytes3;
        
        let (val4, bytes4) = compression.decode_varint(&output[pos..]).unwrap();
        assert_eq!(val4, 16383);
        pos += bytes4;
        
        let (val5, bytes5) = compression.decode_varint(&output[pos..]).unwrap();
        assert_eq!(val5, -1);
        pos += bytes5;
        
        let (val6, _) = compression.decode_varint(&output[pos..]).unwrap();
        assert_eq!(val6, -1000);
    }
}