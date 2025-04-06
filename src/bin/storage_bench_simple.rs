use lsm_tree::run::{
    FileStorage, LSFStorage, Run, RunStorage, StorageOptions
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;

/// Generate test data of specified size
fn generate_test_data(size: usize) -> Vec<(i64, i64)> {
    (0..size).map(|i| (i as i64, (i * 10) as i64)).collect()
}

/// Setup the FileStorage with proper directories
fn setup_file_storage(data: &[(i64, i64)]) -> (Arc<dyn RunStorage>, Run, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    
    // Create directory structure
    let runs_dir = temp_dir.path().join("runs/level_0");
    std::fs::create_dir_all(&runs_dir).unwrap();
    
    // Create options
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage and run
    let storage = Arc::new(FileStorage::new(options).unwrap());
    let run = Run::new(data.to_vec());
    
    (storage, run, temp_dir)
}

/// Setup the LSFStorage with proper directories
fn setup_lsf_storage(data: &[(i64, i64)]) -> (Arc<dyn RunStorage>, Run, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    
    // Create all necessary directories
    let runs_dir = temp_dir.path().join("runs/level_0");
    let logs_dir = temp_dir.path().join("logs");
    let index_dir = temp_dir.path().join("index");
    
    std::fs::create_dir_all(&runs_dir).unwrap();
    std::fs::create_dir_all(&logs_dir).unwrap();
    std::fs::create_dir_all(&index_dir).unwrap();
    
    // Create options
    let options = StorageOptions {
        base_path: temp_dir.path().to_path_buf(),
        create_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
    };
    
    // Create storage and run
    let storage = Arc::new(LSFStorage::new(options).unwrap());
    let run = Run::new(data.to_vec());
    
    (storage, run, temp_dir)
}

/// Benchmark a function multiple times and report the average
fn benchmark<F, R>(name: &str, iterations: usize, setup_fn: F) -> Duration 
where
    F: Fn() -> R
{
    let mut total_duration = Duration::new(0, 0);
    
    for i in 0..iterations {
        let start = Instant::now();
        let _ = setup_fn();
        let duration = start.elapsed();
        total_duration += duration;
        
        if i == 0 {
            println!("  {}: first run took {:?}", name, duration);
        }
    }
    
    let avg_duration = total_duration / iterations as u32;
    println!("  {}: avg over {} runs: {:?}", name, iterations, avg_duration);
    
    avg_duration
}

fn main() {
    println!("\n==== Storage Implementation Benchmark ====");
    
    // Test different data sizes
    for &size in &[10, 100, 1000] {
        println!("\nBenchmarking storage operations with {} records:", size);
        let data = generate_test_data(size);
        let iterations = 10;
        
        // Benchmark FileStorage store
        let file_store_time = benchmark("FileStorage store", iterations, || {
            let (storage, run, _temp_dir) = setup_file_storage(&data);
            storage.store_run(0, &run).unwrap()
        });
        
        // Benchmark LSFStorage store
        let lsf_store_time = benchmark("LSFStorage store", iterations, || {
            let (storage, run, _temp_dir) = setup_lsf_storage(&data);
            storage.store_run(0, &run).unwrap()
        });
        
        // Benchmark FileStorage roundtrip (store+load)
        let file_roundtrip_time = benchmark("FileStorage roundtrip", iterations, || {
            let (storage, run, _temp_dir) = setup_file_storage(&data);
            let run_id = storage.store_run(0, &run).unwrap();
            storage.load_run(run_id).unwrap()
        });
        
        // Benchmark LSFStorage roundtrip (store+load)
        let lsf_roundtrip_time = benchmark("LSFStorage roundtrip", iterations, || {
            let (storage, run, _temp_dir) = setup_lsf_storage(&data);
            let run_id = storage.store_run(0, &run).unwrap();
            storage.load_run(run_id).unwrap()
        });
        
        // Compare performance
        println!("\nComparison for {} records:", size);
        
        // Store operations
        let store_ratio = lsf_store_time.as_nanos() as f64 / file_store_time.as_nanos() as f64;
        println!("  Store: LSFStorage is {:.2}x {} than FileStorage", 
                 store_ratio.abs(),
                 if store_ratio > 1.0 { "slower" } else { "faster" });
                 
        // Roundtrip operations
        let roundtrip_ratio = lsf_roundtrip_time.as_nanos() as f64 / file_roundtrip_time.as_nanos() as f64;
        println!("  Roundtrip: LSFStorage is {:.2}x {} than FileStorage", 
                 roundtrip_ratio.abs(),
                 if roundtrip_ratio > 1.0 { "slower" } else { "faster" });
    }
    
    println!("\nBenchmark complete.");
}