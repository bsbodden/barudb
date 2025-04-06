use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lsm_tree::run::{FencePointers, StandardFencePointers};
use lsm_tree::types::Key;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(target_arch = "x86_64")]
const PLATFORM_NAME: &str = "x86_64_prefetch";
#[cfg(not(target_arch = "x86_64"))]
const PLATFORM_NAME: &str = "generic";

fn generate_random_keys(count: usize, seed: u64) -> Vec<Key> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.gen::<Key>()).collect()
}

fn generate_sorted_keys(count: usize, seed: u64) -> Vec<Key> {
    let mut keys = generate_random_keys(count, seed);
    keys.sort();
    keys
}

fn bench_fence_pointer_lookup(c: &mut Criterion) {
    let fence_sizes = [10, 100, 1_000, 10_000];
    let mut group = c.benchmark_group("fence_pointer_lookup");

    for size in &fence_sizes {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Create optimized fence pointers
        let mut optimized_fence_pointers = FencePointers::new();
        // Create standard fence pointers for comparison
        let mut standard_fence_pointers = StandardFencePointers::new();
        
        let keys = generate_sorted_keys(*size, 42);
        
        // Create fence pointers with sequential blocks
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 {
                continue;
            }
            optimized_fence_pointers.add(chunk[0], chunk[1], i);
            standard_fence_pointers.add(chunk[0], chunk[1], i);
        }
        
        // Generate lookup keys - mix of hits and misses
        let lookup_count = 1000;
        let lookup_keys = generate_random_keys(lookup_count, 43);
        
        // Benchmark optimized lookup performance (with platform-specific name)
        group.bench_function(BenchmarkId::new(
            format!("optimized_find_block_{}", PLATFORM_NAME), 
            size
        ), |b| {
            b.iter(|| {
                for key in &lookup_keys {
                    let _ = optimized_fence_pointers.find_block_for_key(*key);
                }
            })
        });
        
        // Benchmark standard lookup performance
        group.bench_function(BenchmarkId::new("standard_find_block", size), |b| {
            b.iter(|| {
                for key in &lookup_keys {
                    let _ = standard_fence_pointers.find_block_for_key(*key);
                }
            })
        });
    }
    
    group.finish();
}

fn bench_fence_pointer_range(c: &mut Criterion) {
    let fence_sizes = [10, 100, 1_000, 10_000];
    let mut group = c.benchmark_group("fence_pointer_range");

    for size in &fence_sizes {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Create optimized fence pointers
        let mut optimized_fence_pointers = FencePointers::new();
        // Create standard fence pointers for comparison
        let mut standard_fence_pointers = StandardFencePointers::new();
        
        let keys = generate_sorted_keys(*size, 42);
        
        // Create fence pointers with sequential blocks
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 {
                continue;
            }
            optimized_fence_pointers.add(chunk[0], chunk[1], i);
            standard_fence_pointers.add(chunk[0], chunk[1], i);
        }
        
        // Generate range queries with varying selectivity
        let ranges_count = 100;
        let mut ranges = Vec::with_capacity(ranges_count);
        let mut rng = StdRng::seed_from_u64(44);
        
        for _ in 0..ranges_count {
            let start_idx = rng.gen_range(0..keys.len() - 1);
            let end_idx = rng.gen_range(start_idx + 1..keys.len());
            ranges.push((keys[start_idx], keys[end_idx]));
        }
        
        // Benchmark optimized range search performance with platform info
        group.bench_function(BenchmarkId::new(
            format!("optimized_find_blocks_in_range_{}", PLATFORM_NAME), 
            size
        ), |b| {
            b.iter(|| {
                for (start, end) in &ranges {
                    let _ = optimized_fence_pointers.find_blocks_in_range(*start, *end);
                }
            })
        });
        
        // Benchmark standard range search performance
        group.bench_function(BenchmarkId::new("standard_find_blocks_in_range", size), |b| {
            b.iter(|| {
                for (start, end) in &ranges {
                    let _ = standard_fence_pointers.find_blocks_in_range(*start, *end);
                }
            })
        });
    }
    
    group.finish();
}

fn bench_fence_pointer_serialization(c: &mut Criterion) {
    let fence_sizes = [10, 100, 1_000, 10_000];
    let mut group = c.benchmark_group("fence_pointer_serialization");

    for size in &fence_sizes {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Create optimized fence pointers
        let mut optimized_fence_pointers = FencePointers::new();
        // Create standard fence pointers for comparison
        let mut standard_fence_pointers = StandardFencePointers::new();
        
        let keys = generate_sorted_keys(*size, 42);
        
        // Create fence pointers with sequential blocks
        for (i, chunk) in keys.chunks(2).enumerate() {
            if chunk.len() < 2 {
                continue;
            }
            optimized_fence_pointers.add(chunk[0], chunk[1], i);
            standard_fence_pointers.add(chunk[0], chunk[1], i);
        }
        
        // Benchmark optimized serialization performance
        group.bench_function(BenchmarkId::new("optimized_serialize", size), |b| {
            b.iter(|| {
                let _ = optimized_fence_pointers.serialize().unwrap();
            })
        });
        
        // Benchmark standard serialization performance
        group.bench_function(BenchmarkId::new("standard_serialize", size), |b| {
            b.iter(|| {
                let _ = standard_fence_pointers.serialize().unwrap();
            })
        });
        
        // Prepare serialized data for deserialization benchmark
        let optimized_serialized = optimized_fence_pointers.serialize().unwrap();
        let standard_serialized = standard_fence_pointers.serialize().unwrap();
        
        // Benchmark optimized deserialization performance
        group.bench_function(BenchmarkId::new(
            format!("optimized_deserialize_{}", PLATFORM_NAME), 
            size
        ), |b| {
            b.iter(|| {
                let _ = FencePointers::deserialize(&optimized_serialized).unwrap();
            })
        });
        
        // Benchmark standard deserialization performance
        group.bench_function(BenchmarkId::new("standard_deserialize", size), |b| {
            b.iter(|| {
                let _ = StandardFencePointers::deserialize(&standard_serialized).unwrap();
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_fence_pointer_lookup,
    bench_fence_pointer_range,
    bench_fence_pointer_serialization
);
criterion_main!(benches);