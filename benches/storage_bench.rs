use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lsm_tree::run::{
    Block, FileStorage, LSFStorage, Run, RunStorage, StorageOptions
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

/// Benchmark for I/O batching operations
fn bench_io_batching(c: &mut Criterion) {
    // Print a header for better readability
    println!("\n==== I/O Batching Benchmark ====");
    println!("Compares regular loading vs batched loading");
    
    let mut group = c.benchmark_group("io_batching");
    
    // Create test data with multiple blocks
    let create_multi_block_run = |block_count: usize| {
        let mut run = Run::new(vec![]);
        
        // Create multiple blocks with different data
        for i in 0..block_count {
            let mut block = Block::new();
            for j in 0..10 {
                let key = (i * 100 + j) as i64;
                block.add_entry(key, key * 10).unwrap();
            }
            block.seal().unwrap();
            run.blocks.push(block);
        }
        
        // Build fence pointers
        // Since we can't call the private rebuild_fence_pointers method directly,
        // we clear and add each pointer manually
        run.fence_pointers.clear();
        for (idx, block) in run.blocks.iter().enumerate() {
            run.fence_pointers.add(block.header.min_key, block.header.max_key, idx);
        }
        run
    };
    
    // Test with runs having different numbers of blocks
    for &block_count in &[5, 10, 20] {
        let temp_dir = tempdir().unwrap();
        let options = StorageOptions {
            base_path: temp_dir.path().to_path_buf(),
            create_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
        };
        
        let storage = Arc::new(FileStorage::new(options).unwrap());
        let run = create_multi_block_run(block_count);
        
        // Store the run
        let run_id = storage.store_run(0, &run).unwrap();
        
        // Generate indices for sequential blocks to load
        let block_indices: Vec<usize> = (0..block_count).collect();
        
        // Benchmark individual loading (one block at a time)
        group.bench_with_input(
            BenchmarkId::new("individual_loading", block_count),
            &block_indices,
            |b, indices| {
                b.iter(|| {
                    let mut blocks = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        blocks.push(storage.load_block(run_id, idx).unwrap());
                    }
                    black_box(blocks)
                })
            }
        );
        
        // Benchmark batch loading (all blocks at once)
        group.bench_with_input(
            BenchmarkId::new("batch_loading", block_count),
            &block_indices,
            |b, indices| {
                b.iter(|| {
                    black_box(storage.load_blocks_batch(run_id, indices).unwrap())
                })
            }
        );
        
        // Benchmark range query performance with varying range sizes
        for &range_size in &[10, 50, 100] {
            // Benchmark traditional range query (one block at a time)
            let run_clone = run.clone();
            let storage_clone = storage.clone();
            group.bench_with_input(
                BenchmarkId::new(format!("range_query_traditional_{}", range_size), block_count),
                &range_size,
                move |b, &range_size| {
                    // Range that spans multiple blocks
                    let start_key = 0;
                    let end_key = range_size;
                    
                    // Clone run for testing
                    let mut test_run = run_clone.clone();
                    test_run.id = Some(run_id);
                    
                    // Override range_with_storage to use individual block loading
                    b.iter(|| {
                        let mut results = Vec::new();
                        
                        // Get candidate blocks
                        let candidate_blocks = test_run.fence_pointers.find_blocks_in_range(start_key, end_key);
                        
                        // Load each block individually
                        for &block_idx in &candidate_blocks {
                            if block_idx < test_run.blocks.len() {
                                results.extend(test_run.blocks[block_idx].range(start_key, end_key));
                            } else {
                                // Load block from storage individually
                                if let Ok(block) = storage_clone.load_block(run_id, block_idx) {
                                    results.extend(block.range(start_key, end_key));
                                }
                            }
                        }
                        
                        black_box(results)
                    })
                }
            );
            
            // Benchmark optimized range query (using batch loading)
            let run_clone = run.clone();
            let storage_clone = storage.clone();
            group.bench_with_input(
                BenchmarkId::new(format!("range_query_optimized_{}", range_size), block_count),
                &range_size,
                move |b, &range_size| {
                    // Range that spans multiple blocks
                    let start_key = 0;
                    let end_key = range_size;
                    
                    // Clone run for testing
                    let mut test_run = run_clone.clone();
                    test_run.id = Some(run_id);
                    
                    b.iter(|| {
                        let mut results = Vec::new();
                        
                        // Get candidate blocks
                        let candidate_blocks = test_run.fence_pointers.find_blocks_in_range(start_key, end_key);
                        
                        // Split into in-memory blocks and blocks to load
                        let mut in_memory_blocks = Vec::new();
                        let mut blocks_to_load = Vec::new();
                        
                        for &block_idx in &candidate_blocks {
                            if block_idx < test_run.blocks.len() {
                                in_memory_blocks.push(block_idx);
                            } else {
                                blocks_to_load.push(block_idx);
                            }
                        }
                        
                        // Process in-memory blocks
                        for &block_idx in &in_memory_blocks {
                            results.extend(test_run.blocks[block_idx].range(start_key, end_key));
                        }
                        
                        // Load blocks from storage in batch if needed
                        if !blocks_to_load.is_empty() {
                            if let Ok(blocks) = storage_clone.load_blocks_batch(run_id, &blocks_to_load) {
                                for block in blocks {
                                    results.extend(block.range(start_key, end_key));
                                }
                            }
                        }
                        
                        black_box(results)
                    })
                }
            );
        }
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
    targets = bench_store_operations, bench_io_batching
}

criterion_main!(benches);