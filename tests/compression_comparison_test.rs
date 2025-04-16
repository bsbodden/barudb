use lsm_tree::run::compression::{
    CompressionStrategy, NoopCompression, DeltaCompression, BitPackCompression, 
    DictionaryCompression, Lz4Compression, SnappyCompression
};
use std::time::Instant;

/// Test that compares all compression algorithms for different data patterns
#[test]
#[ignore = "Comprehensive compression comparison test; run explicitly with 'cargo test --test compression_comparison_test -- --ignored'"]
fn compare_all_compression_algorithms() {
    compare_compression_on_data_pattern("Sequential Data", generate_sequential_data(100000));
    compare_compression_on_data_pattern("Random Data", generate_random_data(100000));
    compare_compression_on_data_pattern("Repetitive Data", generate_repetitive_data(100000));
    compare_compression_on_data_pattern("Mixed Data", generate_mixed_data(100000));
}

fn compare_compression_on_data_pattern(name: &str, data: Vec<u8>) {
    println!("\n=== {} ({} bytes) ===", name, data.len());
    println!("{:<12} | {:<10} | {:<10} | {:<15} | {:<15} | {:<10}", 
             "Algorithm", "Comp Size", "Ratio", "Comp Time", "Decomp Time", "Throughput");
    println!("{}", "-".repeat(80));
    
    // Test all compression algorithms
    test_strategy("None", &NoopCompression::default(), &data);
    test_strategy("BitPack", &BitPackCompression::default(), &data);
    test_strategy("Delta", &DeltaCompression::default(), &data);
    test_strategy("Dictionary", &DictionaryCompression::default(), &data);
    test_strategy("LZ4", &Lz4Compression::default(), &data);
    test_strategy("Snappy", &SnappyCompression::default(), &data);
    
    // Test LZ4 with different compression levels
    let lz4_low = Lz4Compression { compression_level: 1, block_size: 1024 };
    let lz4_medium = Lz4Compression { compression_level: 6, block_size: 1024 };
    let lz4_high = Lz4Compression { compression_level: 12, block_size: 1024 };
    
    test_strategy("LZ4 (Low)", &lz4_low, &data);
    test_strategy("LZ4 (Med)", &lz4_medium, &data);
    test_strategy("LZ4 (High)", &lz4_high, &data);
}

fn test_strategy(name: &str, strategy: &dyn CompressionStrategy, data: &[u8]) {
    // Compression
    let start = Instant::now();
    let compressed = strategy.compress(data).unwrap();
    let compress_time = start.elapsed();
    
    // Decompression
    let start = Instant::now();
    let decompressed = strategy.decompress(&compressed).unwrap();
    let decompress_time = start.elapsed();
    
    // Verify correctness
    assert_eq!(data, decompressed.as_slice(), 
               "Compression with {} failed verification", name);
    
    // Calculate metrics
    let ratio = if compressed.len() > 0 {
        data.len() as f64 / compressed.len() as f64
    } else {
        0.0
    };
    
    // Calculate throughput (MB/s) for compression
    let throughput = if compress_time.as_secs_f64() > 0.0 {
        (data.len() as f64 / (1024.0 * 1024.0)) / compress_time.as_secs_f64()
    } else {
        f64::INFINITY
    };
    
    println!("{:<12} | {:<10} | {:<10.2}x | {:<15?} | {:<15?} | {:<10.2} MB/s", 
             name, compressed.len(), ratio, compress_time, decompress_time, throughput);
}

/// Test a specific compression algorithm with different data patterns
#[test]
fn test_lz4_vs_snappy_performance() {
    println!("\n=== LZ4 vs Snappy Performance Comparison ===");
    
    let data_sizes = [1_000, 10_000, 100_000, 1_000_000];
    
    for &size in &data_sizes {
        println!("\nData Size: {} bytes", size);
        println!("Sequential Data:");
        compare_two_algorithms(
            &Lz4Compression::default(), 
            &SnappyCompression::default(),
            &generate_sequential_data(size)
        );
        
        println!("Random Data:");
        compare_two_algorithms(
            &Lz4Compression::default(), 
            &SnappyCompression::default(), 
            &generate_random_data(size)
        );
        
        println!("Repetitive Data:");
        compare_two_algorithms(
            &Lz4Compression::default(), 
            &SnappyCompression::default(), 
            &generate_repetitive_data(size)
        );
    }
}

fn compare_two_algorithms(algo1: &dyn CompressionStrategy, algo2: &dyn CompressionStrategy, data: &[u8]) {
    let name1 = algo1.name();
    let name2 = algo2.name();
    
    // Algorithm 1
    let start = Instant::now();
    let compressed1 = algo1.compress(data).unwrap();
    let compress_time1 = start.elapsed();
    
    let start = Instant::now();
    let _ = algo1.decompress(&compressed1).unwrap();
    let decompress_time1 = start.elapsed();
    
    // Algorithm 2
    let start = Instant::now();
    let compressed2 = algo2.compress(data).unwrap();
    let compress_time2 = start.elapsed();
    
    let start = Instant::now();
    let _ = algo2.decompress(&compressed2).unwrap();
    let decompress_time2 = start.elapsed();
    
    // Calculate metrics
    let ratio1 = data.len() as f64 / compressed1.len() as f64;
    let ratio2 = data.len() as f64 / compressed2.len() as f64;
    
    let throughput1 = (data.len() as f64 / (1024.0 * 1024.0)) / compress_time1.as_secs_f64();
    let throughput2 = (data.len() as f64 / (1024.0 * 1024.0)) / compress_time2.as_secs_f64();
    
    // Print comparison
    println!("  {:<6} - Size: {:<8} bytes, Ratio: {:.2}x, Compress: {:?}, Decompress: {:?}, Throughput: {:.2} MB/s", 
             name1, compressed1.len(), ratio1, compress_time1, decompress_time1, throughput1);
    println!("  {:<6} - Size: {:<8} bytes, Ratio: {:.2}x, Compress: {:?}, Decompress: {:?}, Throughput: {:.2} MB/s", 
             name2, compressed2.len(), ratio2, compress_time2, decompress_time2, throughput2);
    
    let size_diff = (compressed1.len() as f64 / compressed2.len() as f64 - 1.0) * 100.0;
    let speed_diff = (throughput2 / throughput1 - 1.0) * 100.0;
    
    println!("  Comparison: {} is {:.1}% {} in size, {:.1}% {} in speed", 
        if compressed1.len() <= compressed2.len() { name1 } else { name2 },
        size_diff.abs(),
        if compressed1.len() <= compressed2.len() { "better" } else { "worse" },
        speed_diff.abs(),
        if throughput1 >= throughput2 { "slower" } else { "faster" }
    );
}

/// Generate sequential data for compression testing
fn generate_sequential_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let full_pairs = size / 8;
    
    for i in 0..full_pairs {
        // Key (sequential)
        data.extend_from_slice(&(i as i32).to_le_bytes());
        // Value (sequential)
        data.extend_from_slice(&((i * 100) as i32).to_le_bytes());
    }
    
    // Fill remaining bytes if needed
    let remaining = size - data.len();
    if remaining > 0 {
        data.extend_from_slice(&vec![0; remaining]);
    }
    
    data
}

/// Generate random data for compression testing
fn generate_random_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let full_pairs = size / 8;
    
    for _ in 0..full_pairs {
        // Random key and value
        data.extend_from_slice(&rand::random::<i32>().to_le_bytes());
        data.extend_from_slice(&rand::random::<i32>().to_le_bytes());
    }
    
    // Fill remaining bytes if needed
    let remaining = size - data.len();
    if remaining > 0 {
        for _ in 0..remaining {
            data.push(rand::random::<u8>());
        }
    }
    
    data
}

/// Generate repetitive data for compression testing
fn generate_repetitive_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let patterns = [
        "This is a test string that will repeat many times.",
        "Another pattern that will be repeated throughout the data.",
        "LSM Trees use efficient compression to store data on disk.",
        "Compression helps reduce storage costs and improve read/write performance.",
    ];
    
    let mut pattern_idx = 0;
    while data.len() < size {
        let pattern = patterns[pattern_idx % patterns.len()];
        let pattern_bytes = pattern.as_bytes();
        let remaining = size - data.len();
        
        if remaining >= pattern_bytes.len() {
            data.extend_from_slice(pattern_bytes);
        } else {
            data.extend_from_slice(&pattern_bytes[0..remaining]);
            break;
        }
        
        pattern_idx += 1;
    }
    
    data
}

/// Generate mixed data patterns for compression testing
fn generate_mixed_data(size: usize) -> Vec<u8> {
    let section_size = size / 3;
    
    let mut data = Vec::with_capacity(size);
    data.extend_from_slice(&generate_sequential_data(section_size));
    data.extend_from_slice(&generate_random_data(section_size));
    data.extend_from_slice(&generate_repetitive_data(size - data.len()));
    
    data
}