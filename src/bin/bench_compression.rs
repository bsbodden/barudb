use lsm_tree::run::{BitPackCompression, CompressionStrategy, NoopCompression};
use lsm_tree::types::{Key, Value};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn create_sequential_data(count: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(count * 16);
    for i in 0..count {
        let key = i as Key;
        let value = (i * 2) as Value;
        data.extend_from_slice(&key.to_le_bytes());
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

fn create_small_range_data(count: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(count * 16);
    let mut rng = StdRng::seed_from_u64(42); // Deterministic seed
    
    for _ in 0..count {
        let key = 1000 + (rng.gen::<Key>() % 100);
        let value = 2000 + (rng.gen::<Value>() % 50);
        data.extend_from_slice(&key.to_le_bytes());
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

fn create_random_data(count: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(count * 16);
    let mut rng = StdRng::seed_from_u64(42); // Deterministic seed
    
    for _ in 0..count {
        let key = rng.gen::<Key>();
        let value = rng.gen::<Value>();
        data.extend_from_slice(&key.to_le_bytes());
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

fn create_same_value_data(count: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(count * 16);
    
    for _ in 0..count {
        let key = 42 as Key;
        let value = 42 as Value;
        data.extend_from_slice(&key.to_le_bytes());
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

fn measure_compression<T: CompressionStrategy>(
    compressor: &T, 
    data: &[u8],
    data_type: &str
) -> (f64, f64, f64) {
    // Measure compression time
    let start = Instant::now();
    let compressed = compressor.compress(data).unwrap();
    let compress_time = start.elapsed().as_secs_f64() * 1000.0; // ms
    
    // Measure decompression time
    let start = Instant::now();
    let decompressed = compressor.decompress(&compressed).unwrap();
    let decompress_time = start.elapsed().as_secs_f64() * 1000.0; // ms
    
    // Calculate compression ratio
    let ratio = data.len() as f64 / compressed.len() as f64;
    
    // Verify correctness
    assert_eq!(data, decompressed.as_slice(), "Decompressed data doesn't match original for {}", data_type);
    
    (compress_time, decompress_time, ratio)
}

fn run_benchmark(data_size: usize) {
    println!("\n==== Compression Benchmark Results ====");
    println!("Dataset sizes: {} key-value pairs (16 bytes each)", data_size);
    println!("Original data size: {} bytes", data_size * 16);
    
    let bit_pack = BitPackCompression::default();
    
    // Sample different types of data to test different compression patterns
    let sequential_data = create_sequential_data(data_size);
    let small_range_data = create_small_range_data(data_size);
    let random_data = create_random_data(data_size);
    let same_value_data = create_same_value_data(data_size);
    
    println!("\nBit Pack Compression Results:");
    
    let (c_time, d_time, ratio) = measure_compression(&bit_pack, &sequential_data, "sequential");
    println!("  Sequential data:  {:.2}x ratio, {:.2} ms compress, {:.2} ms decompress", 
             ratio, c_time, d_time);
    
    let (c_time, d_time, ratio) = measure_compression(&bit_pack, &small_range_data, "small range");
    println!("  Small range data: {:.2}x ratio, {:.2} ms compress, {:.2} ms decompress", 
             ratio, c_time, d_time);
    
    let (c_time, d_time, ratio) = measure_compression(&bit_pack, &random_data, "random");
    println!("  Random data:      {:.2}x ratio, {:.2} ms compress, {:.2} ms decompress", 
             ratio, c_time, d_time);
    
    let (c_time, d_time, ratio) = measure_compression(&bit_pack, &same_value_data, "same value");
    println!("  Same value data:  {:.2}x ratio, {:.2} ms compress, {:.2} ms decompress", 
             ratio, c_time, d_time);
    
    // Compare with NoopCompression (baseline)
    let noop = NoopCompression::default();
    println!("\nNoopCompression Results (Baseline):");
    
    let (c_time, d_time, ratio) = measure_compression(&noop, &sequential_data, "noop sequential");
    println!("  Sequential data:  {:.2}x ratio, {:.2} ms compress, {:.2} ms decompress", 
             ratio, c_time, d_time);
}

fn main() {
    // Run benchmarks with different data sizes
    println!("Running small dataset benchmark (1,000 entries)");
    run_benchmark(1_000);
    
    println!("\nRunning medium dataset benchmark (10,000 entries)");
    run_benchmark(10_000);
    
    println!("\nRunning large dataset benchmark (100,000 entries)");
    run_benchmark(100_000);
}