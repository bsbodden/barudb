use std::sync::atomic::{AtomicU64, Ordering};

/// Direct port of RocksDB's FastLocalBloomImpl to Rust
pub struct RocksDBLocalBloom {
    len: u32,               // Length in 64-bit words
    num_probes: u32,        // Number of probes per key
    data: Box<[AtomicU64]>, // The bit array backing the filter
}

impl RocksDBLocalBloom {
    pub fn new(total_bits: u32, num_probes: u32) -> Self {
        assert!(num_probes <= 30, "Too many probes");

        // RocksDB always aligns to 512-bit (64-byte) cache lines
        const CACHE_LINE_BITS: u32 = 512;
        const BITS_PER_WORD: u32 = 64;

        // Round up total_bits to cache line size
        let block_bits = CACHE_LINE_BITS;
        let blocks = (total_bits + block_bits - 1) / block_bits;
        let len = blocks * (block_bits / BITS_PER_WORD);

        let mut data = Vec::with_capacity(len as usize);
        data.extend((0..len).map(|_| AtomicU64::new(0)));

        Self {
            len,
            num_probes,
            data: data.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn add_hash(&self, h1: u32, h2: u32) {
        let cache_bytes = self.get_cache_line_offset(h1);
        self.add_hash_prepared(h2, cache_bytes);
    }

    #[inline]
    pub fn add_hash_concurrently(&self, h1: u32, h2: u32) {
        // Same implementation as add_hash - RocksDB uses the same path for both
        self.add_hash(h1, h2);
    }

    #[inline]
    fn add_hash_prepared(&self, mut h2: u32, byte_offset: usize) {
        for _ in 0..self.num_probes {
            // Extract bit position from top 9 bits (as per RocksDB)
            let bitpos = h2 >> (32 - 9);
            let word_idx = byte_offset + (bitpos as usize >> 6);
            let bit_mask = 1u64 << (bitpos & 63);

            self.data[word_idx].fetch_or(bit_mask, Ordering::Relaxed);

            // RocksDB golden ratio constant for mixing
            h2 = h2.wrapping_mul(0x9e3779b9);
        }
    }

    #[inline]
    pub fn may_contain(&self, h1: u32, h2: u32) -> bool {
        let cache_bytes = self.get_cache_line_offset(h1);
        self.may_contain_prepared(h2, cache_bytes)
    }

    #[inline]
    fn may_contain_prepared(&self, mut h2: u32, byte_offset: usize) -> bool {
        for _ in 0..self.num_probes {
            // Extract bit position from top 9 bits (as per RocksDB)
            let bitpos = h2 >> (32 - 9);
            let word_idx = byte_offset + (bitpos as usize >> 6);
            let bit_mask = 1u64 << (bitpos & 63);

            if (self.data[word_idx].load(Ordering::Relaxed) & bit_mask) == 0 {
                return false;
            }

            // RocksDB golden ratio constant for mixing
            h2 = h2.wrapping_mul(0x9e3779b9);
        }
        true
    }

    #[inline]
    fn get_cache_line_offset(&self, h1: u32) -> usize {
        // This matches RocksDB's FastRange32 implementation exactly
        let num_lines = self.len >> 3; // Divide by 8 words per cache line
        let line = (((h1 as u64).wrapping_mul(num_lines as u64)) >> 32) as usize;
        line << 3 // Multiply by 8 words per cache line
    }

    #[cfg(target_arch = "x86_64")]
    pub fn prefetch(&self, h1: u32) {
        let cache_bytes = self.get_cache_line_offset(h1);
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch(
                self.data.as_ptr().add(cache_bytes) as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn prefetch(&self, _h1: u32) {}

    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<AtomicU64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;
    use xxhash_rust::xxh3::xxh3_128;

    #[test]
    fn test_empty_filter() {
        let bloom = RocksDBLocalBloom::new(100, 2);
        assert!(!bloom.may_contain(0, 0));
        assert!(!bloom.may_contain(1, 1));
        assert!(!bloom.may_contain(100, 100));
    }

    #[test]
    fn test_basic_operations() {
        let bloom = RocksDBLocalBloom::new(1024, 6);

        bloom.add_hash(1, 1);
        bloom.add_hash(2, 2);
        bloom.add_hash(100, 100);

        assert!(bloom.may_contain(1, 1));
        assert!(bloom.may_contain(2, 2));
        assert!(bloom.may_contain(100, 100));

        assert!(!bloom.may_contain(3, 3));
        assert!(!bloom.may_contain(99, 99));
    }

    #[test]
    fn test_concurrent_add() {
        let bloom = Arc::new(RocksDBLocalBloom::new(1024, 6));
        let num_threads = 4;
        let keys_per_thread = 1000;

        let mut threads = vec![];
        for t in 0..num_threads {
            let bloom = Arc::clone(&bloom);
            threads.push(thread::spawn(move || {
                for i in (t..keys_per_thread).step_by(num_threads) {
                    bloom.add_hash_concurrently(i as u32, i as u32);
                }
            }));
        }

        for thread in threads {
            thread.join().unwrap();
        }

        for i in 0..keys_per_thread {
            assert!(bloom.may_contain(i as u32, i as u32));
        }
    }

    #[test]
    fn test_false_positive_rate() {
        // Use same params as RocksDB's test
        let bloom = RocksDBLocalBloom::new(8192, 6);

        // Add sequential keys
        for i in 0..400 {
            let h1 = xxh3_128(&(i as u64).to_le_bytes()) as u32;
            let h2 = (xxh3_128(&(i as u64).to_le_bytes()) >> 32) as u32;
            bloom.add_hash(h1, h2);
        }

        // Test on different range
        let mut false_positives = 0;
        let trials = 10_000;

        for i in 1_000_000..(1_000_000 + trials) {
            let h1 = xxh3_128(&(i as u64).to_le_bytes()) as u32;
            let h2 = (xxh3_128(&(i as u64).to_le_bytes()) >> 32) as u32;
            if bloom.may_contain(h1, h2) {
                false_positives += 1;
            }
        }

        let fp_rate = false_positives as f64 / trials as f64;
        assert!(fp_rate < 0.02, "False positive rate too high: {}", fp_rate);
    }

    #[test]
    fn test_memory_efficiency() {
        for bits_per_key in [10, 20, 40] {
            let num_entries = 100_000;
            let total_bits = num_entries * bits_per_key;

            let bloom = RocksDBLocalBloom::new(total_bits as u32, 6);

            // Add entries
            for i in 0..num_entries {
                let key_bytes = (i as i32).to_le_bytes();
                let hash = xxh3_128(&key_bytes);
                bloom.add_hash(hash as u32, (hash >> 32) as u32);
            }

            // Test false positive rate
            let mut false_positives = 0;
            let test_entries = 10_000;

            for i in 0..test_entries {
                let key_bytes = (i as i32 + 1_000_000_000).to_le_bytes();
                let hash = xxh3_128(&key_bytes);
                if bloom.may_contain(hash as u32, (hash >> 32) as u32) {
                    false_positives += 1;
                }
            }

            let fp_rate = false_positives as f64 / test_entries as f64;
            let memory_bytes = bloom.memory_usage();

            println!("Bits per key: {}", bits_per_key);
            println!("Memory usage: {} KB", memory_bytes / 1024);
            println!("False positive rate: {:.4}%", fp_rate * 100.0);

            assert!(fp_rate <= 0.02, "FP rate too high: {}", fp_rate);
        }
    }

    #[test]
    fn test_different_sizes() {
        for bits in [64, 128, 256, 512, 1024] {
            let bloom = RocksDBLocalBloom::new(bits, 6);

            bloom.add_hash(1, 1);
            bloom.add_hash(2, 2);
            assert!(bloom.may_contain(1, 1));
            assert!(bloom.may_contain(2, 2));
            assert!(!bloom.may_contain(3, 3));
        }
    }

    #[test]
    fn test_varying_lengths() {
        for length in 1..=25 {
            let bits = length * 10;
            let bloom = RocksDBLocalBloom::new(bits as u32, 6);

            // Add elements
            for i in 0..length {
                let hash = xxh3_128(&(i as u64).to_le_bytes());
                bloom.add_hash(hash as u32, (hash >> 32) as u32);
            }

            // Verify all added elements
            for i in 0..length {
                let hash = xxh3_128(&(i as u64).to_le_bytes());
                assert!(
                    bloom.may_contain(hash as u32, (hash >> 32) as u32),
                    "Length {}, value {} not found",
                    length,
                    i
                );
            }

            // Check false positive rate
            let mut false_positives = 0;
            let test_size = 30000;
            for i in 0..test_size {
                let hash = xxh3_128(&((i + 1_000_000_000) as u64).to_le_bytes());
                if bloom.may_contain(hash as u32, (hash >> 32) as u32) {
                    false_positives += 1;
                }
            }

            let rate = false_positives as f64 / test_size as f64;
            println!("Length: {}, FP rate: {:.2}%", length, rate * 100.0);
            assert!(
                rate < 0.2,
                "False positive rate too high: {}% at length {}",
                rate * 100.0,
                length
            );
        }
    }

    #[test]
    fn test_hash_distribution() {
        let mut bit_counts = vec![0; 64]; // Track 64 bits per word

        // Insert test keys and count which bits get set
        for i in 0..1000 {
            let mut h = 0x9e3779b97f4a7c13u64.wrapping_mul(i as u64);

            for _ in 0..3 {
                // 9-bit address within 512 bit cache line
                let bitpos = (h as u32) >> (32 - 9);
                let bit_in_word = bitpos & 63;
                bit_counts[bit_in_word as usize] += 1;
                h = h.wrapping_mul(0x9e3779b9);
            }
        }

        let min_count = bit_counts.iter().copied().min().unwrap();
        let max_count = bit_counts.iter().copied().max().unwrap();
        let avg_count: f64 = bit_counts.iter().sum::<i32>() as f64 / bit_counts.len() as f64;

        println!(
            "Bit distribution - min: {}, max: {}, avg: {:.2}",
            min_count, max_count, avg_count
        );

        // Expect reasonably uniform distribution
        assert!((max_count as f64) < (avg_count * 2.0));
        assert!(min_count > 0);
    }

    #[test]
    #[ignore = "Long-running performance test; run explicitly with 'cargo test test_performance -- --ignored'"]
    fn test_performance() {
        for m in 1..=2 {
            // Reduced iterations for faster tests
            let num_keys = m * 8 * 1024 * 1024;
            println!("Testing {} million keys", m * 8);

            let bloom = RocksDBLocalBloom::new((num_keys * 10) as u32, 6);

            // Measure add performance
            let start = Instant::now();
            for i in 1..=num_keys {
                let hash = xxh3_128(&(i as u64).to_le_bytes());
                bloom.add_hash(hash as u32, (hash >> 32) as u32);
            }
            let elapsed = start.elapsed();
            println!(
                "Add latency: {:.2} ns/key",
                elapsed.as_nanos() as f64 / num_keys as f64
            );

            // Measure query performance
            let mut count = 0;
            let start = Instant::now();
            for i in 1..=num_keys {
                let hash = xxh3_128(&(i as u64).to_le_bytes());
                if bloom.may_contain(hash as u32, (hash >> 32) as u32) {
                    count += 1;
                }
            }
            let elapsed = start.elapsed();
            println!(
                "Query latency: {:.2} ns/key",
                elapsed.as_nanos() as f64 / count as f64
            );
            assert_eq!(count, num_keys);
        }
    }
}
