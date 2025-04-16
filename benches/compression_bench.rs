use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lsm_tree::run::compression::{
    BitPackCompression, CompressionFactory, CompressionStrategy, CompressionType,
    NoopCompression, CompressionStats, DeltaCompression, DictionaryCompression,
    Lz4Compression, SnappyCompression
};
use lsm_tree::types::{Key, Value};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

/// Generate different data patterns for testing compression strategies
fn generate_test_data() -> Vec<(String, Vec<(Key, Value)>)> {
    let mut result = Vec::new();
    
    // Sequential keys (common in databases with auto-increment IDs)
    let mut sequential_data = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        sequential_data.push((i, i * 2));
    }
    result.push(("sequential".to_string(), sequential_data));
    
    // Small range data (e.g., counters, enum values)
    let mut rng = StdRng::seed_from_u64(42);
    let mut small_range_data = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let key = 1000 + (rng.gen::<Key>() % 100);
        let value = 2000 + (rng.gen::<Value>() % 50);
        small_range_data.push((key, value));
    }
    result.push(("small_range".to_string(), small_range_data));
    
    // Same value data (e.g., default values in columns)
    let mut same_value_data = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        same_value_data.push((42, 42));
    }
    result.push(("same_value".to_string(), same_value_data));
    
    // Monotonic increasing with small step (e.g., timestamps)
    let mut monotonic_data = Vec::with_capacity(10_000);
    let base = 1_600_000_000; // Unix timestamp base
    for i in 0..10_000 {
        let timestamp = base + i * 60; // One minute intervals
        monotonic_data.push((timestamp, i));
    }
    result.push(("timestamps".to_string(), monotonic_data));
    
    // Random data (worst case for compression)
    let mut random_data = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let key = rng.gen::<Key>();
        let value = rng.gen::<Value>();
        random_data.push((key, value));
    }
    result.push(("random".to_string(), random_data));
    
    result
}

fn data_to_bytes(data: &[(Key, Value)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 16);
    for &(k, v) in data {
        bytes.extend_from_slice(&k.to_le_bytes());
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Measure compression performance
fn measure_compression(
    strategy: &dyn CompressionStrategy,
    data: &[(Key, Value)],
) -> CompressionStats {
    let bytes = data_to_bytes(data);
    
    // Measure compression time
    let start = Instant::now();
    let compressed = strategy.compress(&bytes).unwrap();
    let compression_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    // Measure decompression time
    let start = Instant::now();
    let decompressed = strategy.decompress(&compressed).unwrap();
    let decompression_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    // Verify correctness
    assert_eq!(bytes, decompressed, "Decompression failed to recover original data");
    
    CompressionStats {
        strategy_name: strategy.name().to_string(),
        original_size: bytes.len(),
        compressed_size: compressed.len(),
        compression_ratio: bytes.len() as f64 / compressed.len() as f64,
        compression_time_ms,
        decompression_time_ms,
    }
}

/// Benchmark various compression strategies
fn benchmark_compression(c: &mut Criterion) {
    let test_data = generate_test_data();
    
    // Define compression strategies to test
    let strategies = [
        ("noop", Box::new(NoopCompression::default()) as Box<dyn CompressionStrategy>),
        ("bit_pack", Box::new(BitPackCompression::default()) as Box<dyn CompressionStrategy>),
        ("delta", Box::new(DeltaCompression::default()) as Box<dyn CompressionStrategy>),
        ("dictionary", Box::new(DictionaryCompression::default()) as Box<dyn CompressionStrategy>),
        ("lz4", Box::new(Lz4Compression::default()) as Box<dyn CompressionStrategy>),
        ("snappy", Box::new(SnappyCompression::default()) as Box<dyn CompressionStrategy>),
    ];
    
    let mut group = c.benchmark_group("compression");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    
    // Benchmark each strategy on each data pattern
    for (data_name, data) in &test_data {
        for (strategy_name, strategy) in &strategies {
            let id = format!("{}_{}", strategy_name, data_name);
            
            // Benchmark compression
            group.bench_function(format!("compress_{}_{}", strategy_name, data_name), |b| {
                let bytes = data_to_bytes(data);
                b.iter(|| black_box(strategy).compress(black_box(&bytes)))
            });
            
            // Benchmark decompression
            let bytes = data_to_bytes(data);
            let compressed = strategy.compress(&bytes).unwrap();
            group.bench_function(format!("decompress_{}_{}", strategy_name, data_name), |b| {
                b.iter(|| black_box(strategy).decompress(black_box(&compressed)))
            });
        }
    }
    
    group.finish();
    
    // Print a summary of compression ratios and times
    println!("\n== Compression Strategy Comparison ==\n");
    println!("{:<15} {:<15} {:<10} {:<10} {:<15} {:<15}", 
             "Data Pattern", "Strategy", "Ratio", "Size (KB)", "Comp Time (ms)", "Decomp Time (ms)");
    println!("{}", "-".repeat(80));
    
    for (data_name, data) in &test_data {
        let orig_size = data_to_bytes(data).len() / 1024;
        
        for (_, strategy) in &strategies {
            let stats = measure_compression(&**strategy, data);
            println!("{:<15} {:<15} {:<10.2} {:<10.2} {:<15.3} {:<15.3}", 
                     data_name, stats.strategy_name, stats.compression_ratio, 
                     stats.compressed_size as f64 / 1024.0, 
                     stats.compression_time_ms, stats.decompression_time_ms);
        }
        println!("{}", "-".repeat(80));
    }
}

criterion_group!(benches, benchmark_compression);
criterion_main!(benches);