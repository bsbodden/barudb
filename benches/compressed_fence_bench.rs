use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lsm_tree::run::{
    CompressedFencePointers, AdaptivePrefixFencePointers, TwoLevelFencePointers, StandardFencePointers
};
use lsm_tree::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn generate_random_keys(count: usize, seed: u64) -> Vec<Key> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.gen::<Key>()).collect()
}

fn generate_sorted_keys(count: usize, seed: u64) -> Vec<Key> {
    let mut keys = generate_random_keys(count, seed);
    keys.sort();
    keys
}

fn generate_grouped_keys(count: usize, groups: usize, seed: u64) -> Vec<Key> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(count);
    
    for _ in 0..count {
        // Select a group
        let group = rng.gen_range(0..groups) as Key;
        // Generate a key with the group in the high bits
        let key = (group << 32) | (rng.gen::<u32>() as Key);
        keys.push(key);
    }
    
    keys.sort();
    keys
}

/// Benchmark lookup performance of different fence pointer implementations 
fn bench_lookup_performance(c: &mut Criterion) {
    let sizes = [100usize, 1_000usize, 10_000usize];
    let mut group = c.benchmark_group("fence_pointer_lookup");
    
    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Generate different key distributions
        let sequential_keys = (0..size).map(|i| i as Key).collect::<Vec<_>>();
        let random_keys = generate_sorted_keys(size, 42);
        let grouped_keys = generate_grouped_keys(size, 8, 43);
        
        // Generate lookup keys - mix of hits and misses
        let lookup_count = 1000;
        let lookup_keys = generate_random_keys(lookup_count, 44);
        
        // Create different fence pointer implementations
        struct Implementation {
            name: &'static str,
            build_fn: fn(keys: &[Key]) -> Box<dyn FencePointerLookup>,
        }
        
        trait FencePointerLookup {
            fn find_block_for_key(&self, key: Key) -> Option<usize>;
        }
        
        impl FencePointerLookup for StandardFencePointers {
            fn find_block_for_key(&self, key: Key) -> Option<usize> {
                self.find_block_for_key(key)
            }
        }
        
        impl FencePointerLookup for TwoLevelFencePointers {
            fn find_block_for_key(&self, key: Key) -> Option<usize> {
                self.find_block_for_key(key)
            }
        }
        
        impl FencePointerLookup for CompressedFencePointers {
            fn find_block_for_key(&self, key: Key) -> Option<usize> {
                self.find_block_for_key(key)
            }
        }
        
        impl FencePointerLookup for AdaptivePrefixFencePointers {
            fn find_block_for_key(&self, key: Key) -> Option<usize> {
                self.find_block_for_key(key)
            }
        }
        
        // Setup the implementations to test
        let implementations = [
            Implementation { 
                name: "standard", 
                build_fn: |keys: &[Key]| {
                    let mut fp = StandardFencePointers::new();
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn FencePointerLookup>
                } 
            },
            Implementation { 
                name: "two_level", 
                build_fn: |keys: &[Key]| {
                    let mut fp = TwoLevelFencePointers::with_ratio(20);
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn FencePointerLookup>
                } 
            },
            Implementation { 
                name: "compressed", 
                build_fn: |keys: &[Key]| {
                    let mut fp = CompressedFencePointers::with_group_size(16);
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn FencePointerLookup>
                } 
            },
            Implementation { 
                name: "adaptive", 
                build_fn: |keys: &[Key]| {
                    let mut fp = AdaptivePrefixFencePointers::new();
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn FencePointerLookup>
                } 
            },
        ];
        
        // Test different key distributions
        for (dist_name, keys) in [
            ("sequential", sequential_keys), 
            ("random", random_keys),
            ("grouped", grouped_keys)
        ] {
            // Create all implementations with these keys
            for impl_info in &implementations {
                let fp = (impl_info.build_fn)(&keys);
                
                // Benchmark lookup performance
                group.bench_function(BenchmarkId::new(
                    format!("{}_{}_{}", impl_info.name, dist_name, size), 
                    "lookup"
                ), |b| {
                    b.iter(|| {
                        for key in &lookup_keys {
                            let _ = fp.find_block_for_key(*key);
                        }
                    })
                });
            }
        }
    }
    
    group.finish();
}

/// Benchmark range query performance of different fence pointer implementations
fn bench_range_query_performance(c: &mut Criterion) {
    let sizes = [100usize, 1_000usize, 10_000usize];
    let mut group = c.benchmark_group("fence_pointer_range");
    
    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Generate different key distributions
        let sequential_keys = (0..size).map(|i| i as Key).collect::<Vec<_>>();
        let random_keys = generate_sorted_keys(size, 42);
        let grouped_keys = generate_grouped_keys(size, 8, 43);
        
        // Generate range queries with varying sizes
        let ranges_count = 100;
        let mut rng = StdRng::seed_from_u64(44);
        
        struct RangeImplementation {
            name: &'static str,
            build_fn: fn(keys: &[Key]) -> Box<dyn FencePointerRange>,
        }
        
        trait FencePointerRange {
            fn find_blocks_in_range(&self, start: Key, end: Key) -> Vec<usize>;
        }
        
        impl FencePointerRange for StandardFencePointers {
            fn find_blocks_in_range(&self, start: Key, end: Key) -> Vec<usize> {
                self.find_blocks_in_range(start, end)
            }
        }
        
        impl FencePointerRange for TwoLevelFencePointers {
            fn find_blocks_in_range(&self, start: Key, end: Key) -> Vec<usize> {
                self.find_blocks_in_range(start, end)
            }
        }
        
        impl FencePointerRange for CompressedFencePointers {
            fn find_blocks_in_range(&self, start: Key, end: Key) -> Vec<usize> {
                self.find_blocks_in_range(start, end)
            }
        }
        
        impl FencePointerRange for AdaptivePrefixFencePointers {
            fn find_blocks_in_range(&self, start: Key, end: Key) -> Vec<usize> {
                self.find_blocks_in_range(start, end)
            }
        }
        
        // Setup the implementations to test
        let implementations = [
            RangeImplementation { 
                name: "standard", 
                build_fn: |keys: &[Key]| {
                    let mut fp = StandardFencePointers::new();
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn FencePointerRange>
                } 
            },
            RangeImplementation { 
                name: "two_level", 
                build_fn: |keys: &[Key]| {
                    let mut fp = TwoLevelFencePointers::with_ratio(20);
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn FencePointerRange>
                } 
            },
            RangeImplementation { 
                name: "compressed", 
                build_fn: |keys: &[Key]| {
                    let mut fp = CompressedFencePointers::with_group_size(16);
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn FencePointerRange>
                } 
            },
            RangeImplementation { 
                name: "adaptive", 
                build_fn: |keys: &[Key]| {
                    let mut fp = AdaptivePrefixFencePointers::new();
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn FencePointerRange>
                } 
            },
        ];
        
        // Test different key distributions
        for (dist_name, keys) in [
            ("sequential", sequential_keys), 
            ("random", random_keys),
            ("grouped", grouped_keys)
        ] {
            // Generate ranges appropriate for this key set
            let mut ranges = Vec::with_capacity(ranges_count);
            for _ in 0..ranges_count {
                let start_idx = rng.gen_range(0..keys.len().saturating_sub(1));
                let end_idx = rng.gen_range(start_idx + 1..keys.len());
                ranges.push((keys[start_idx], keys[end_idx]));
            }
            
            // Create all implementations with these keys
            for impl_info in &implementations {
                let fp = (impl_info.build_fn)(&keys);
                
                // Benchmark range query performance
                group.bench_function(BenchmarkId::new(
                    format!("{}_{}_{}", impl_info.name, dist_name, size), 
                    "range"
                ), |b| {
                    b.iter(|| {
                        for (start, end) in &ranges {
                            let _ = fp.find_blocks_in_range(*start, *end);
                        }
                    })
                });
            }
        }
    }
    
    group.finish();
}

/// Benchmark memory efficiency of different fence pointer implementations
fn bench_memory_efficiency(c: &mut Criterion) {
    let sizes = [100usize, 1_000usize, 10_000usize, 100_000usize];
    let mut group = c.benchmark_group("fence_pointer_memory");
    
    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Generate different key distributions
        let sequential_keys = (0..size).map(|i| i as Key).collect::<Vec<_>>();
        let random_keys = generate_sorted_keys(size, 42);
        let grouped_keys = generate_grouped_keys(size, 8, 43);
        
        // Define a trait for memory measurement
        trait MemoryMeasurable {
            fn memory_bytes(&self) -> usize;
        }
        
        impl MemoryMeasurable for StandardFencePointers {
            fn memory_bytes(&self) -> usize {
                // Just count the raw size of stored data
                std::mem::size_of::<Self>() + 
                self.pointers.capacity() * std::mem::size_of::<(Key, Key, usize)>()
            }
        }
        
        impl MemoryMeasurable for TwoLevelFencePointers {
            fn memory_bytes(&self) -> usize {
                // Approximate memory usage
                std::mem::size_of::<Self>() + 
                self.sparse.guide_keys.capacity() * std::mem::size_of::<Key>() +
                self.sparse.dense_indices.capacity() * std::mem::size_of::<usize>() +
                self.dense.pointers.capacity() * std::mem::size_of::<(Key, Key, usize)>()
            }
        }
        
        impl MemoryMeasurable for CompressedFencePointers {
            fn memory_bytes(&self) -> usize {
                self.memory_usage()
            }
        }
        
        impl MemoryMeasurable for AdaptivePrefixFencePointers {
            fn memory_bytes(&self) -> usize {
                self.memory_usage()
            }
        }
        
        struct MemoryImplementation {
            name: &'static str,
            build_fn: fn(keys: &[Key]) -> Box<dyn MemoryMeasurable>,
        }
        
        // Setup the implementations to test
        let implementations = [
            MemoryImplementation { 
                name: "standard", 
                build_fn: |keys: &[Key]| {
                    let mut fp = StandardFencePointers::new();
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn MemoryMeasurable>
                } 
            },
            MemoryImplementation { 
                name: "two_level", 
                build_fn: |keys: &[Key]| {
                    let mut fp = TwoLevelFencePointers::with_ratio(20);
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn MemoryMeasurable>
                } 
            },
            MemoryImplementation { 
                name: "compressed", 
                build_fn: |keys: &[Key]| {
                    let mut fp = CompressedFencePointers::with_group_size(16);
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    Box::new(fp) as Box<dyn MemoryMeasurable>
                } 
            },
            MemoryImplementation { 
                name: "adaptive", 
                build_fn: |keys: &[Key]| {
                    let mut fp = AdaptivePrefixFencePointers::new();
                    for (i, chunk) in keys.chunks(2).enumerate() {
                        if chunk.len() < 2 { continue; }
                        fp.add(chunk[0], chunk[1], i);
                    }
                    fp.optimize();  // Force optimization
                    Box::new(fp) as Box<dyn MemoryMeasurable>
                } 
            },
        ];
        
        // Test different key distributions
        for (dist_name, keys) in [
            ("sequential", sequential_keys), 
            ("random", random_keys),
            ("grouped", grouped_keys)
        ] {
            // Create all implementations with these keys
            for impl_info in &implementations {
                let fp = (impl_info.build_fn)(&keys);
                let memory_bytes = fp.memory_bytes();
                
                // Print memory usage for comparison
                println!(
                    "Memory usage {}_{}_{}:{} bytes ({:.2} bytes per entry)", 
                    impl_info.name, dist_name, size, 
                    memory_bytes,
                    memory_bytes as f64 / (size as f64 / 2.0) // Division by 2 because we chunk by 2
                );
                
                // Benchmark is just a way to record these metrics
                group.bench_function(BenchmarkId::new(
                    format!("{}_{}_{}", impl_info.name, dist_name, size), 
                    "memory"
                ), |b| {
                    b.iter(|| memory_bytes)
                });
            }
        }
    }
    
    group.finish();
}

/// Benchmark serialization/deserialization performance
fn bench_serialization(c: &mut Criterion) {
    let sizes = [1_000usize, 10_000usize];
    let mut group = c.benchmark_group("fence_pointer_serialization");
    
    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Generate grouped keys - good for testing compression efficacy
        let keys = generate_grouped_keys(size, 8, 43);
        
        // Define implementations and build them
        let mut standard = StandardFencePointers::new();
        let mut two_level = TwoLevelFencePointers::with_ratio(20);
        let mut compressed = CompressedFencePointers::with_group_size(16);
        let mut adaptive = AdaptivePrefixFencePointers::new();
        
        // Add all keys to each implementation
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 { continue; }
            standard.add(chunk[0], chunk[1], i);
            two_level.add(chunk[0], chunk[1], i);
            compressed.add(chunk[0], chunk[1], i);
            adaptive.add(chunk[0], chunk[1], i);
        }
        
        // Benchmark serialization
        let standard_serialized = standard.serialize().unwrap();
        let two_level_serialized = two_level.serialize().unwrap();
        let compressed_serialized = compressed.serialize().unwrap();
        let adaptive_serialized = adaptive.serialize().unwrap();
        
        // Print serialized sizes
        println!(
            "Serialized size standard_{}: {} bytes", 
            size, standard_serialized.len()
        );
        println!(
            "Serialized size two_level_{}: {} bytes", 
            size, two_level_serialized.len()
        );
        println!(
            "Serialized size compressed_{}: {} bytes", 
            size, compressed_serialized.len()
        );
        println!(
            "Serialized size adaptive_{}: {} bytes", 
            size, adaptive_serialized.len()
        );
        
        // Benchmark serialization performance
        group.bench_function(BenchmarkId::new("standard", size), |b| {
            b.iter(|| standard.serialize().unwrap())
        });
        
        group.bench_function(BenchmarkId::new("two_level", size), |b| {
            b.iter(|| two_level.serialize().unwrap())
        });
        
        group.bench_function(BenchmarkId::new("compressed", size), |b| {
            b.iter(|| compressed.serialize().unwrap())
        });
        
        group.bench_function(BenchmarkId::new("adaptive", size), |b| {
            b.iter(|| adaptive.serialize().unwrap())
        });
        
        // Benchmark deserialization performance
        group.bench_function(BenchmarkId::new("standard_deserialize", size), |b| {
            b.iter(|| StandardFencePointers::deserialize(&standard_serialized).unwrap())
        });
        
        group.bench_function(BenchmarkId::new("two_level_deserialize", size), |b| {
            b.iter(|| TwoLevelFencePointers::deserialize(&two_level_serialized).unwrap())
        });
        
        group.bench_function(BenchmarkId::new("compressed_deserialize", size), |b| {
            b.iter(|| CompressedFencePointers::deserialize(&compressed_serialized).unwrap())
        });
        
        group.bench_function(BenchmarkId::new("adaptive_deserialize", size), |b| {
            b.iter(|| AdaptivePrefixFencePointers::deserialize(&adaptive_serialized).unwrap())
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_lookup_performance,
    bench_range_query_performance,
    bench_memory_efficiency,
    bench_serialization
);
criterion_main!(benches);