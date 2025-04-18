use crate::run::compression::CompressionStrategy;
use crate::types::{Result, Error, Key, Value};
use std::any::Any;
use std::cmp::min;

/// Bit-Packing compression for integer data
/// 
/// Stores integers as offsets from a minimum value using the minimum number of bits required.
/// This is particularly effective for data with small ranges.
#[derive(Debug, Clone)]
pub struct BitPackCompression {
    /// Maximum number of values per block
    pub block_size: usize,
}

impl Default for BitPackCompression {
    fn default() -> Self {
        Self {
            block_size: 1024,
        }
    }
}

/// Header for a bit-packed block
#[derive(Debug, Clone, Copy)]
struct BitPackHeader {
    /// Minimum key value in the block (reference value)
    min_key: Key,
    /// Minimum value in the block (reference value)
    min_value: Value,
    /// Number of bits needed to store each key offset
    key_bits: u8,
    /// Number of bits needed to store each value offset
    value_bits: u8,
    /// Number of key-value pairs in this block
    count: u32,
}

impl CompressionStrategy for BitPackCompression {
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
        
        // Estimate based on bit-packing effectiveness
        // Assuming sequential data compresses to 25-35% of original size,
        // while completely random data compresses to 100% (or slightly more)
        
        // This is a rough estimate and will vary greatly depending on data
        // For extremely compressible data (like all same value), performance will be even better
        let entry_count = data.len() / 16;
        let header_size = std::mem::size_of::<BitPackHeader>();
        
        // Assume 8 bits for keys and 8 bits for values (conservative estimate) 
        // plus overhead for headers
        let num_blocks = (entry_count + self.block_size - 1) / self.block_size;
        let estimated_data_bits = entry_count * (8 + 8); // 8 bits each for keys and values
        let estimated_data_bytes = (estimated_data_bits + 7) / 8;
        let block_headers_size = num_blocks * 4; // 4 bytes per block for size
        let block_internals_size = num_blocks * header_size;
        
        block_headers_size + block_internals_size + estimated_data_bytes
    }
    
    fn clone_box(&self) -> Box<dyn CompressionStrategy> {
        Box::new(self.clone())
    }
    
    fn name(&self) -> &'static str {
        "bit_pack"
    }
    
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

impl BitPackCompression {
    /// Compress a single block of keys and values
    fn compress_block(&self, keys: &[Key], values: &[Value]) -> Result<Vec<u8>> {
        if keys.is_empty() || keys.len() != values.len() {
            return Err(Error::InvalidInput("Empty or mismatched keys/values".to_string()));
        }
        
        // Find min/max values to determine bits needed
        let min_key = *keys.iter().min().unwrap_or(&0);
        let max_key = *keys.iter().max().unwrap_or(&0);
        let min_value = *values.iter().min().unwrap_or(&0);
        let max_value = *values.iter().max().unwrap_or(&0);
        
        // Special case: all keys are the same and all values are the same
        // This is a perfect case for bit packing and will result in significant compression
        if min_key == max_key && min_value == max_value {
            let header = BitPackHeader {
                min_key,
                min_value,
                key_bits: 1,   // Need 1 bit per key (all zeros)
                value_bits: 1, // Need 1 bit per value (all zeros)
                count: keys.len() as u32,
            };
            
            let header_size = std::mem::size_of::<BitPackHeader>();
            let mut result = Vec::with_capacity(header_size + 8); // Very small data size
            
            // Add header
            result.extend_from_slice(&header.min_key.to_le_bytes());
            result.extend_from_slice(&header.min_value.to_le_bytes());
            result.push(header.key_bits);
            result.push(header.value_bits);
            result.extend_from_slice(&[0, 0]); // Padding to keep alignment
            result.extend_from_slice(&header.count.to_le_bytes());
            
            // Add bit-packed data (all zeros, so just add minimum bytes needed)
            // For N keys and N values at 1 bit each, we need (2N + 7) / 8 bytes
            let bytes_needed = (2 * keys.len() + 7) / 8;
            let zeros = vec![0u8; bytes_needed];
            result.extend_from_slice(&zeros);
            
            return Ok(result);
        }
        
        // Check if keys are sequential (i.e., k[i] = min_key + i)
        // This pattern is common in databases and should compress well
        let is_sequential = if keys.len() > 1 {
            let mut seq_pattern = true;
            
            for i in 0..keys.len() {
                if keys[i] != min_key + (i as i64) {
                    seq_pattern = false;
                    break;
                }
            }
            seq_pattern
        } else {
            false
        };
        
        // Calculate bits needed for keys and values
        let (key_range_valid, key_bits) = self.calculate_bits_needed(min_key, max_key);
        let (value_range_valid, value_bits) = self.calculate_bits_needed(min_value, max_value);
        
        // Skip real compression if we can't compress effectively
        if (!key_range_valid || key_bits >= 64) && (!value_range_valid || value_bits >= 64) {
            // Use uncompressed storage if we can't compress effectively
            let mut result = Vec::with_capacity(std::mem::size_of::<BitPackHeader>() + keys.len() * 16);
            
            // Create header with max bit sizes to signal uncompressed storage
            let header = BitPackHeader {
                min_key,
                min_value,
                key_bits: 64,
                value_bits: 64,
                count: keys.len() as u32,
            };
            
            // Add header
            result.extend_from_slice(&header.min_key.to_le_bytes());
            result.extend_from_slice(&header.min_value.to_le_bytes());
            result.push(header.key_bits);
            result.push(header.value_bits);
            result.extend_from_slice(&[0, 0]); // Padding to keep alignment
            result.extend_from_slice(&header.count.to_le_bytes());
            
            // Add data uncompressed
            for i in 0..keys.len() {
                result.extend_from_slice(&keys[i].to_le_bytes());
                result.extend_from_slice(&values[i].to_le_bytes());
            }
            
            return Ok(result);
        }
        
        // If we get here, we're going to use bit packing
        let header_size = std::mem::size_of::<BitPackHeader>();
        let mut result = Vec::with_capacity(header_size + keys.len() * ((key_bits + value_bits) as usize + 7) / 8);
        
        // Create header with computed bit sizes
        let header = BitPackHeader {
            min_key,
            min_value,
            key_bits,
            value_bits,
            count: keys.len() as u32,
        };
        
        // Add header
        result.extend_from_slice(&header.min_key.to_le_bytes());
        result.extend_from_slice(&header.min_value.to_le_bytes());
        result.push(header.key_bits);
        result.push(header.value_bits);
        result.extend_from_slice(&[0, 0]); // Padding to keep alignment
        result.extend_from_slice(&header.count.to_le_bytes());
        
        // Special case optimization for sequential keys
        // For perfectly sequential keys, we can greatly simplify storage
        if is_sequential && keys.len() > 1 {
            // For sequential keys, we only need to store the values since keys can be derived
            // We'll set key_bits to a special value (0) to indicate this pattern
            let modified_header = BitPackHeader {
                min_key,
                min_value,
                key_bits: 0, // Special marker for sequential keys
                value_bits,
                count: keys.len() as u32,
            };
            
            // Re-write the header with our special key_bits value
            result.clear();
            result.extend_from_slice(&modified_header.min_key.to_le_bytes());
            result.extend_from_slice(&modified_header.min_value.to_le_bytes());
            result.push(modified_header.key_bits);
            result.push(modified_header.value_bits);
            result.extend_from_slice(&[0, 0]); // Padding to keep alignment
            result.extend_from_slice(&modified_header.count.to_le_bytes());
            
            // Only store value offsets since keys are sequential from min_key
            let mut writer = BitWriter::new();
            
            // For values, compute the actual offsets from min_value
            for i in 0..values.len() {
                let value_offset = self.calculate_offset(values[i], min_value, value_range_valid);
                writer.write_bits(value_offset, value_bits);
            }
            
            let packed_data = writer.finish();
            result.extend_from_slice(&packed_data);
            
            return Ok(result);
        }
        
        // Normal bit packing for non-sequential data
        let mut writer = BitWriter::new();
        
        // Compress each key-value pair using the calculated bit sizes
        for i in 0..keys.len() {
            // Calculate key offset and write it
            let key_offset = self.calculate_offset(keys[i], min_key, key_range_valid);
            writer.write_bits(key_offset, key_bits);
            
            // Calculate value offset and write it
            let value_offset = self.calculate_offset(values[i], min_value, value_range_valid);
            writer.write_bits(value_offset, value_bits);
        }
        
        let packed_data = writer.finish();
        result.extend_from_slice(&packed_data);
        
        Ok(result)
    }
    
    /// Calculate bits needed for a range and handle edge cases
    fn calculate_bits_needed(&self, min_val: i64, max_val: i64) -> (bool, u8) {
        // Handle case where all values are the same
        if min_val == max_val {
            return (true, 1); // Need at least 1 bit
        }
        
        // Handle case where range fits within a small number of bits
        let range = max_val.saturating_sub(min_val); // Avoid overflow with saturating_sub
        
        // If range is too large or overflows, use full width
        if range < 0 || range > i64::MAX / 2 {
            return (false, 64);
        }
        
        // Convert to u64 for bit counting (safe now that we've checked for overflow)
        let range_u64 = range as u64;
        
        // Calculate bits needed based on range
        // Need to get the position of the highest bit set
        let bits_needed = if range_u64 == 0 {
            1 // Need at least 1 bit even if all values are the same
        } else {
            // Find position of highest bit (log2 ceiling)
            64 - range_u64.leading_zeros() as u8
        };
        
        // Return the number of bits needed
        (true, bits_needed)
    }
    
    /// Calculate offset for bit packing (safely handle potential overflows)
    fn calculate_offset(&self, value: i64, min_value: i64, valid_range: bool) -> u64 {
        if !valid_range {
            // If the range is invalid, just use the value directly
            return value as u64;
        }
        
        // Calculate offset
        let offset = value.wrapping_sub(min_value);
        
        // The offset should always be non-negative, but if by some chance
        // there's an overflow, we'll handle it by treating as unsigned
        offset as u64
    }
    
    /// Decompress a single block
    fn decompress_block(&self, data: &[u8]) -> Result<Vec<u8>> {
        // First check that we have at least a header
        let header_size = std::mem::size_of::<BitPackHeader>();
        if data.len() < header_size {
            return Err(Error::InvalidData("Invalid bit-packed block: too small".to_string()));
        }
        
        // Read header fields
        let min_key = Key::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
        ]);
        let min_value = Value::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]
        ]);
        let key_bits = data[16];
        let value_bits = data[17];
        // Skip padding at 18-19
        let count = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        
        // Check for valid header values
        if count == 0 {
            return Err(Error::InvalidData("Invalid bit-packed header: count is zero".to_string()));
        }
        
        // Special case for uncompressed storage
        if key_bits >= 64 || value_bits >= 64 {
            // Data is stored uncompressed
            if data.len() < header_size + (count as usize * 16) {
                return Err(Error::InvalidData("Invalid bit-packed block: too small for uncompressed data".to_string()));
            }
            
            // Just copy the uncompressed data
            let mut result = Vec::with_capacity(count as usize * 16);
            
            for i in 0..count as usize {
                let offset = header_size + (i * 16);
                if offset + 16 <= data.len() {
                    // Copy key-value pair directly
                    result.extend_from_slice(&data[offset..offset+16]);
                }
            }
            
            return Ok(result);
        }
        
        // Special case for all same values
        if key_bits == 1 && value_bits == 1 && count > 0 {
            // All keys and values are the same
            let mut result = Vec::with_capacity(count as usize * 16);
            
            // Just replicate the same key-value pair count times
            for _ in 0..count {
                result.extend_from_slice(&min_key.to_le_bytes());
                result.extend_from_slice(&min_value.to_le_bytes());
            }
            
            return Ok(result);
        }
        
        // Prepare result buffer
        let result_size = count as usize * 16; // Key (8) + Value (8) = 16 bytes per entry
        let mut result = Vec::with_capacity(result_size);
        
        // Initialize bit reader for the compressed data portion
        let mut bit_reader = BitReader::new(&data[header_size..]);
        
        // Check for the special case of sequential keys (key_bits = 0)
        if key_bits == 0 {
            // This is our special marker for sequential keys
            for i in 0..count {
                // Keys are sequential from min_key
                let key = min_key + (i as i64);
                
                // For values, read the actual bit-packed offset
                let value_offset = bit_reader.read_bits(value_bits)?;
                let value = min_value.wrapping_add(value_offset as Value);
                
                // Add key-value pair to result
                result.extend_from_slice(&key.to_le_bytes());
                result.extend_from_slice(&value.to_le_bytes());
            }
            
            return Ok(result);
        }
        
        // Standard decompression for bit-packed data
        for _ in 0..count {
            // Read key offset and calculate actual key
            let key_offset = bit_reader.read_bits(key_bits)?;
            let key = min_key.wrapping_add(key_offset as Key);
            
            // Read value offset and calculate actual value
            let value_offset = bit_reader.read_bits(value_bits)?;
            let value = min_value.wrapping_add(value_offset as Value);
            
            // Add key-value pair to result
            result.extend_from_slice(&key.to_le_bytes());
            result.extend_from_slice(&value.to_le_bytes());
        }
        
        Ok(result)
    }
}

/// Helper struct for writing bits to a byte array
struct BitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bits_used: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_byte: 0,
            bits_used: 0,
        }
    }
    
    fn write_bits(&mut self, value: u64, num_bits: u8) {
        // Special case for 0 bits
        if num_bits == 0 {
            return;
        }
        
        // Cap at 64 bits (the size of u64)
        let bits_to_write = num_bits.min(64);
        
        // Create a mask to isolate only the bits we want
        let mask = if bits_to_write == 64 {
            u64::MAX
        } else {
            (1u64 << bits_to_write) - 1
        };
        let masked_value = value & mask;
        
        // Process the value in chunks that fit within a byte
        let mut remaining_bits = bits_to_write;
        let mut remaining_value = masked_value;
        
        while remaining_bits > 0 {
            // How many bits can we fit in the current byte?
            let bits_available = 8 - self.bits_used;
            let bits_to_add = remaining_bits.min(bits_available);
            
            // Extract the relevant bits from the value
            let bits_mask = if bits_to_add == 64 {
                u64::MAX
            } else {
                (1u64 << bits_to_add) - 1
            };
            
            // Add bits to the current byte
            let bits = (remaining_value & bits_mask) as u8;
            self.current_byte |= bits << self.bits_used;
            
            // Update our tracking variables
            self.bits_used += bits_to_add;
            remaining_bits -= bits_to_add;
            remaining_value >>= bits_to_add;
            
            // If we've filled a byte, add it to the buffer
            if self.bits_used == 8 {
                self.buffer.push(self.current_byte);
                self.current_byte = 0;
                self.bits_used = 0;
            }
        }
    }
    
    fn finish(mut self) -> Vec<u8> {
        // If we have any bits in the current byte, flush it
        if self.bits_used > 0 {
            self.buffer.push(self.current_byte);
        }
        self.buffer
    }
}

/// Helper struct for reading bits from a byte array
struct BitReader<'a> {
    data: &'a [u8],
    current_byte_index: usize,
    bits_consumed: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            current_byte_index: 0,
            bits_consumed: 0,
        }
    }
    
    fn read_bits(&mut self, num_bits: u8) -> Result<u64> {
        // Make sure we don't try to read more than 64 bits
        let bits_to_read = num_bits.min(64);
        
        // Handle reading 0 bits
        if bits_to_read == 0 {
            return Ok(0);
        }
        
        let mut result: u64 = 0;
        let mut bits_read: u8 = 0;
        
        // Read bits until we've read enough or run out of data
        while bits_read < bits_to_read {
            // If we've run out of data, return an error
            if self.current_byte_index >= self.data.len() {
                return Err(Error::InvalidData("Unexpected end of bit-packed data".to_string()));
            }
            
            // Get current byte
            let current_byte = self.data[self.current_byte_index];
            
            // How many more bits do we need to read?
            let bits_needed = bits_to_read - bits_read;
            
            // How many bits are available in the current byte?
            let bits_available = 8 - self.bits_consumed;
            
            // How many bits will we read from this byte?
            let bits_from_byte = bits_needed.min(bits_available);
            
            // Special case for 0 bits - just skip
            if bits_from_byte == 0 {
                break;
            }
            
            // Read the bits
            let mask = if bits_from_byte == 8 {
                0xFF
            } else {
                ((1u16 << bits_from_byte) - 1) as u8
            };
            
            let bits = (current_byte >> self.bits_consumed) & mask;
            
            // Add the bits to the result
            result |= (bits as u64) << bits_read;
            
            // Update state
            bits_read += bits_from_byte;
            self.bits_consumed += bits_from_byte;
            
            // If we've consumed all bits in the current byte, move to the next one
            if self.bits_consumed == 8 {
                self.current_byte_index += 1;
                self.bits_consumed = 0;
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    
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
    fn test_bit_pack_empty_data() {
        let compression = BitPackCompression::default();
        let empty: &[u8] = &[];
        
        let compressed = compression.compress(empty).unwrap();
        assert_eq!(compressed.len(), 0);
        
        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 0);
    }
    
    #[test]
    fn test_bit_pack_single_entry() {
        let compression = BitPackCompression::default();
        let entries = [(42, 100)];
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_bit_pack_sequential_keys() {
        let compression = BitPackCompression::default();
        
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
    fn test_bit_pack_small_range() {
        let compression = BitPackCompression::default();
        
        // Small range of values (only need a few bits)
        let entries: Vec<(Key, Value)> = (0..100).map(|i| (1000 + (i % 10), 2000 + (i % 5))).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_bit_pack_large_range() {
        let compression = BitPackCompression::default();
        
        // Large range of values
        let entries: Vec<(Key, Value)> = (0..100).map(|i| (i * 1000000, i * 500000)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_bit_pack_random_data() {
        let compression = BitPackCompression::default();
        
        // Random values won't compress as well
        let mut rng = rand::rng();
        let entries: Vec<(Key, Value)> = (0..100)
            .map(|_| (rng.random::<Key>(), rng.random::<Value>()))
            .collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_bit_pack_all_same_value() {
        let compression = BitPackCompression::default();
        
        // All same values should compress extremely well
        let entries: Vec<(Key, Value)> = (0..100).map(|_| (42, 42)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        println!("Original size: {}, Compressed size: {}, Ratio: {:.2}",
                 data.len(), compressed.len(), data.len() as f64 / compressed.len() as f64);
    }
    
    #[test]
    fn test_bit_pack_multiple_blocks() {
        let mut compression = BitPackCompression::default();
        compression.block_size = 10; // Small block size to test multiple blocks
        
        let entries: Vec<(Key, Value)> = (0..100).map(|i| (i, i * 2)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_bit_pack_negative_values() {
        let compression = BitPackCompression::default();
        
        // Test with negative values
        let entries: Vec<(Key, Value)> = (-50..50).map(|i| (i, -i)).collect();
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_bit_writer_reader() {
        let mut writer = BitWriter::new();
        
        // Write a sequence of bits
        writer.write_bits(3, 2);  // 11
        writer.write_bits(7, 3);  // 111
        writer.write_bits(5, 4);  // 0101
        writer.write_bits(0, 1);  // 0
        writer.write_bits(1, 1);  // 1
        
        let buffer = writer.finish();
        
        // Read the bits back using BitReader
        let mut reader = BitReader::new(&buffer);
        
        assert_eq!(reader.read_bits(2).unwrap(), 3);
        assert_eq!(reader.read_bits(3).unwrap(), 7);
        assert_eq!(reader.read_bits(4).unwrap(), 5);
        assert_eq!(reader.read_bits(1).unwrap(), 0);
        assert_eq!(reader.read_bits(1).unwrap(), 1);
    }
    
    #[test]
    fn test_bit_pack_extreme_values() {
        let compression = BitPackCompression::default();
        
        // Test with extreme values (MIN/MAX)
        let entries = [
            (i64::MIN, i64::MAX),
            (i64::MAX, i64::MIN),
            (0, 0),
            (i64::MIN, 0),
            (0, i64::MAX)
        ];
        let data = create_test_data(&entries);
        
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
    }
}