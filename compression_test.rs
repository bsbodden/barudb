use lsm_tree::run::compression::{
    CompressionStrategy, NoopCompression, BitPackCompression, DeltaCompression,
    DictionaryCompression, Lz4Compression, SnappyCompression
};
use std::time::Instant;

fn main() {
    println!("Testing compression algorithms performance");
    
    // Generate test data
    let sequential_data = generate_sequential_data(10000);
    let random_data = generate_random_data(10000);
    let repetitive_data = generate_repetitive_data(10000);
    
    // Create compression strategies
    let strategies: Vec<(&str, Box<dyn CompressionStrategy>)> = vec![
        ("noop", Box::new(NoopCompression::default())),
        ("bit_pack", Box::new(BitPackCompression::default())),
        ("delta", Box::new(DeltaCompression::default())),
        ("dictionary", Box::new(DictionaryCompression::default())),
        ("lz4", Box::new(Lz4Compression::default())),
        ("snappy", Box::new(SnappyCompression::default())),
    ];
    
    // Test with sequential data
    println!("\n=== Sequential Data ===");
    test_compression_strategies(&strategies, &sequential_data);
    
    // Test with random data
    println!("\n=== Random Data ===");
    test_compression_strategies(&strategies, &random_data);
    
    // Test with repetitive data
    println!("\n=== Repetitive Data ===");
    test_compression_strategies(&strategies, &repetitive_data);
}

fn test_compression_strategies(
    strategies: &[(&str, Box<dyn CompressionStrategy>)], 
    data: &[u8]
) {
    println!("Original size: {} bytes", data.len());
    
    println!("{:<12} | {:<14} | {:<14} | {:<14} | {:<14}", 
             "Strategy", "Comp Size", "Ratio", "Comp Time", "Decomp Time");
    println!("{}", "-".repeat(75));
    
    for (name, strategy) in strategies {
        // Measure compression
        let start = Instant::now();
        let compressed = strategy.compress(data).unwrap();
        let compression_time = start.elapsed();
        
        // Measure decompression
        let start = Instant::now();
        let decompressed = strategy.decompress(&compressed).unwrap();
        let decompression_time = start.elapsed();
        
        // Verify correctness
        assert_eq!(data, decompressed.as_slice(), 
                   "Decompression failed for {}", name);
        
        // Calculate ratio
        let ratio = if compressed.len() > 0 {
            data.len() as f64 / compressed.len() as f64
        } else {
            0.0
        };
        
        println!("{:<12} | {:<14} | {:<14.2} | {:<14?} | {:<14?}", 
                 name, compressed.len(), ratio, compression_time, decompression_time);
    }
}

fn generate_sequential_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size * 8);
    for i in 0..size {
        // Key (sequential)
        data.extend_from_slice(&(i as i32).to_le_bytes());
        // Value (sequential)
        data.extend_from_slice(&(i * 100 as i32).to_le_bytes());
    }
    data
}

fn generate_random_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size * 8);
    for _ in 0..size {
        // Random key and value
        data.extend_from_slice(&(rand::random::<i32>()).to_le_bytes());
        data.extend_from_slice(&(rand::random::<i32>()).to_le_bytes());
    }
    data
}

fn generate_repetitive_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size * 8);
    let patterns = [
        "This is a test string that will repeat many times.",
        "Another pattern that will be repeated throughout the data.",
        "LSM Trees use efficient compression to store data on disk.",
        "Compression helps reduce storage costs and improve read/write performance.",
    ];
    
    let mut pattern_idx = 0;
    for i in 0..size {
        // Key (sequential)
        data.extend_from_slice(&(i as i32).to_le_bytes());
        
        // Value (one of the repetitive patterns, converted to an integer)
        let pattern = patterns[pattern_idx % patterns.len()];
        let hash = u32::from_be_bytes([
            pattern.as_bytes()[0],
            pattern.as_bytes()[1 % pattern.len()],
            pattern.as_bytes()[2 % pattern.len()],
            pattern.as_bytes()[3 % pattern.len()],
        ]);
        data.extend_from_slice(&(hash as i32).to_le_bytes());
        
        pattern_idx += 1;
    }
    data
}