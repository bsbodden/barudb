use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lsm_tree::run::{
    FileStorage, LSFStorage, Run, RunStorage, StorageOptions
};
use std::sync::Arc;
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
    
    // Create all necessary directories - the LSF storage implementation requires these
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

/// Store and load benchmark for different storage implementations
fn bench_store_operations(c: &mut Criterion) {
    // Print a header for better readability
    println!("\n==== Storage Implementation Benchmark ====");
    println!("Tests storage operations for different implementations");
    
    let mut group = c.benchmark_group("storage_operations");
    // Remove summary_only as it might not be available in all criterion versions
    
    // Test with small and medium data sizes
    for size in [10, 100, 1000].iter() {
        let data = generate_test_data(*size);
        
        // Benchmark FileStorage store operation
        group.bench_with_input(
            BenchmarkId::new("file_store", size), 
            &data,
            |b, data| {
                b.iter_with_setup(
                    || setup_file_storage(data),
                    |(storage, run, _temp_dir)| {
                        // What we're measuring: storing a run
                        black_box(storage.store_run(0, &run).unwrap())
                    }
                )
            }
        );
        
        // Benchmark LSFStorage store operation
        group.bench_with_input(
            BenchmarkId::new("lsf_store", size), 
            &data,
            |b, data| {
                b.iter_with_setup(
                    || setup_lsf_storage(data),
                    |(storage, run, _temp_dir)| {
                        // What we're measuring: storing a run
                        black_box(storage.store_run(0, &run).unwrap())
                    }
                )
            }
        );
        
        // Benchmark FileStorage store+load (roundtrip)
        group.bench_with_input(
            BenchmarkId::new("file_roundtrip", size), 
            &data,
            |b, data| {
                b.iter_with_setup(
                    || {
                        let (storage, run, temp_dir) = setup_file_storage(data);
                        let run_id = storage.store_run(0, &run).unwrap();
                        (storage, run_id, temp_dir)
                    },
                    |(storage, run_id, _temp_dir)| {
                        // What we're measuring: loading a run after storing it
                        black_box(storage.load_run(run_id).unwrap())
                    }
                )
            }
        );
        
        // Benchmark LSFStorage store+load (roundtrip)
        group.bench_with_input(
            BenchmarkId::new("lsf_roundtrip", size), 
            &data,
            |b, data| {
                b.iter_with_setup(
                    || {
                        let (storage, run, temp_dir) = setup_lsf_storage(data);
                        let run_id = storage.store_run(0, &run).unwrap();
                        (storage, run_id, temp_dir)
                    },
                    |(storage, run_id, _temp_dir)| {
                        // What we're measuring: loading a run after storing it
                        black_box(storage.load_run(run_id).unwrap())
                    }
                )
            }
        );
    }
    
    group.finish();
}

// Configure the benchmark group
criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)  // Use smaller sample size for faster benchmarks
        .measurement_time(std::time::Duration::from_secs(2))  // Shorter measurement time
        .noise_threshold(0.05)  // Less sensitive to noise (5% threshold)
        .without_plots();  // Skip plot generation to reduce output size
    targets = bench_store_operations
}

criterion_main!(benches);