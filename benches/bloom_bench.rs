use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use lsm_tree::bloom::{Bloom, RocksDBLocalBloom, SpeedDbDynamicBloom, create_bloom_for_level};
use fastbloom::BloomFilter;
use rand::{rngs::StdRng, Rng, SeedableRng};
use xxhash_rust::xxh3::xxh3_128;

fn random_numbers(num: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    std::iter::repeat_with(|| rng.random()).take(num).collect()
}

fn bench_bloom_filters(c: &mut Criterion) {
    // Only use the smallest size for faster evaluation
    let sizes = [1_000];
    let mut group = c.benchmark_group("bloom_filters");

    // Add configuration for better small-set resolution
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(200); // Increased samples for better statistical significance

    for size in sizes {
        let bits_per_key = 10;
        let total_bits = size * bits_per_key;

        // Sample data
        let items: Vec<String> = random_numbers(size, 42)
            .into_iter()
            .map(|n| n.to_string())
            .collect();
        let lookup_items: Vec<String> = random_numbers(size, 43)
            .into_iter()
            .map(|n| n.to_string())
            .collect();

        // Create filters
        let speed_db_bloom = SpeedDbDynamicBloom::new(total_bits as u32, 6);
        let bloom = Bloom::new(total_bits as u32, 6);
        let rocks_bloom = RocksDBLocalBloom::new(total_bits as u32, 6);
        let fast_bloom = BloomFilter::with_num_bits(total_bits)
            .expected_items(size);

        // Benchmark individual insertions
        group.bench_function(BenchmarkId::new("speeddb_insert", size), |b| {
            b.iter(|| {
                for item in &items {
                    speed_db_bloom.add_hash(item.parse::<u32>().unwrap());
                }
            })
        });

        group.bench_function(BenchmarkId::new("bloom_insert", size), |b| {
            b.iter(|| {
                for item in &items {
                    bloom.add_hash(item.parse::<u32>().unwrap());
                }
            })
        });

        group.bench_function(BenchmarkId::new("rocksdb_insert", size), |b| {
            b.iter(|| {
                for item in &items {
                    let hash = xxh3_128(&item.parse::<u32>().unwrap().to_le_bytes());
                    rocks_bloom.add_hash(hash as u32, (hash >> 32) as u32);
                }
            })
        });

        group.bench_function(BenchmarkId::new("fastbloom_insert", size), |b| {
            b.iter(|| {
                let mut filter = fast_bloom.clone();
                for item in &items {
                    filter.insert(item);
                }
            })
        });

        // Benchmark batch insertions
        let items_u32: Vec<u32> = items.iter().map(|s| s.parse::<u32>().unwrap()).collect();
        
        group.bench_function(BenchmarkId::new("bloom_insert_batch", size), |b| {
            b.iter(|| {
                bloom.add_hash_batch(&items_u32, false);
            })
        });
        
        group.bench_function(BenchmarkId::new("bloom_insert_batch_concurrent", size), |b| {
            b.iter(|| {
                bloom.add_hash_batch(&items_u32, true);
            })
        });

        // Prepare populated filters for lookups
        let mut populated_fast_bloom = fast_bloom.clone();
        for item in &items {
            populated_fast_bloom.insert(item);
            speed_db_bloom.add_hash(item.parse::<u32>().unwrap());
            bloom.add_hash(item.parse::<u32>().unwrap());
            let hash = xxh3_128(&item.parse::<u32>().unwrap().to_le_bytes());
            rocks_bloom.add_hash(hash as u32, (hash >> 32) as u32);
        }

        // Benchmark individual lookups
        group.bench_function(BenchmarkId::new("speeddb_lookup", size), |b| {
            b.iter(|| {
                for item in &lookup_items {
                    let _ = speed_db_bloom.may_contain(item.parse::<u32>().unwrap());
                }
            })
        });

        group.bench_function(BenchmarkId::new("bloom_lookup", size), |b| {
            b.iter(|| {
                for item in &lookup_items {
                    let _ = bloom.may_contain(item.parse::<u32>().unwrap());
                }
            })
        });

        group.bench_function(BenchmarkId::new("rocksdb_lookup", size), |b| {
            b.iter(|| {
                for item in &lookup_items {
                    let hash = xxh3_128(&item.parse::<u32>().unwrap().to_le_bytes());
                    let _ = rocks_bloom.may_contain(hash as u32, (hash >> 32) as u32);
                }
            })
        });

        group.bench_function(BenchmarkId::new("fastbloom_lookup", size), |b| {
            b.iter(|| {
                for item in &lookup_items {
                    let _ = populated_fast_bloom.contains(item);
                }
            })
        });
        
        // Benchmark batch lookups
        let lookup_items_u32: Vec<u32> = lookup_items.iter().map(|s| s.parse::<u32>().unwrap()).collect();
        
        group.bench_function(BenchmarkId::new("bloom_lookup_batch", size), |b| {
            b.iter(|| {
                let mut results = vec![false; lookup_items_u32.len()];
                bloom.may_contain_batch(&lookup_items_u32, &mut results);
                results
            })
        });

        // Benchmark false positives
        let fp_items: Vec<String> = random_numbers(10_000, 44)
            .into_iter()
            .map(|n| n.to_string())
            .collect();

        group.bench_function(BenchmarkId::new("speeddb_fp", size), |b| {
            b.iter(|| {
                let mut fps = 0;
                for item in &fp_items {
                    if speed_db_bloom.may_contain(item.parse::<u32>().unwrap()) {
                        fps += 1;
                    }
                }
                fps
            })
        });

        group.bench_function(BenchmarkId::new("bloom_fp", size), |b| {
            b.iter(|| {
                let mut fps = 0;
                for item in &fp_items {
                    if bloom.may_contain(item.parse::<u32>().unwrap()) {
                        fps += 1;
                    }
                }
                fps
            })
        });

        group.bench_function(BenchmarkId::new("rocksdb_fp", size), |b| {
            b.iter(|| {
                let mut fps = 0;
                for item in &fp_items {
                    let hash = xxh3_128(&item.parse::<u32>().unwrap().to_le_bytes());
                    if rocks_bloom.may_contain(hash as u32, (hash >> 32) as u32) {
                        fps += 1;
                    }
                }
                fps
            })
        });

        group.bench_function(BenchmarkId::new("fastbloom_fp", size), |b| {
            b.iter(|| {
                let mut fps = 0;
                for item in &fp_items {
                    if populated_fast_bloom.contains(item) {
                        fps += 1;
                    }
                }
                fps
            })
        });
        
        // Benchmark batch false positives
        let fp_items_u32: Vec<u32> = fp_items.iter().map(|s| s.parse::<u32>().unwrap()).collect();
        
        group.bench_function(BenchmarkId::new("bloom_fp_batch", size), |b| {
            b.iter(|| {
                let mut results = vec![false; fp_items_u32.len()];
                bloom.may_contain_batch(&fp_items_u32, &mut results);
                results.iter().filter(|&&r| r).count()
            })
        });
    }
    group.finish();
}

fn bench_monkey_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("monkey_optimization");
    
    // Configuration
    let size = 10000; // Fixed size for all tests
    let fanout = 4.0; // Common fanout ratio
    let levels = 5; // Number of levels to test
    
    // Sample data
    let items: Vec<u32> = random_numbers(size, 42);
    let lookup_items: Vec<u32> = random_numbers(size, 43);
    let fp_items: Vec<u32> = random_numbers(10_000, 44);
    
    // Create Bloom filters for different levels
    let mut level_filters = Vec::with_capacity(levels);
    
    for level in 0..levels {
        let filter = create_bloom_for_level(size, level, fanout);
        level_filters.push(filter);
    }
    
    // Populate all filters with the same data
    for level in 0..levels {
        for &item in &items {
            level_filters[level].add_hash(item);
        }
    }
    
    // Benchmark lookup performance
    for level in 0..levels {
        group.bench_function(BenchmarkId::new("monkey_lookup", level), |b| {
            b.iter(|| {
                for &item in &lookup_items {
                    let _ = level_filters[level].may_contain(item);
                }
            })
        });
    }
    
    // Benchmark false positive rates
    for level in 0..levels {
        group.bench_function(BenchmarkId::new("monkey_fp", level), |b| {
            b.iter(|| {
                let mut fps = 0;
                for &item in &fp_items {
                    if level_filters[level].may_contain(item) {
                        fps += 1;
                    }
                }
                fps
            })
        });
    }
    
    // Compare memory usage (not really a benchmark, but outputs useful metrics)
    for level in 0..levels {
        let bits_per_entry = (level_filters[level].memory_usage() * 8) as f64 / size as f64;
        group.bench_function(BenchmarkId::new("monkey_memory", level), |b| {
            b.iter(|| {
                level_filters[level].memory_usage()
            })
        });
        println!("Level {}: {:.2} bits per entry", level, bits_per_entry);
    }
    
    group.finish();
}

criterion_group!(benches, bench_bloom_filters, bench_monkey_optimization);
criterion_main!(benches);