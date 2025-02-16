use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lsm_tree::bloom::DynamicBloom;
use fastbloom::BloomFilter;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn random_numbers(num: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    std::iter::repeat_with(|| rng.gen()).take(num).collect()
}

fn bench_bloom_filters(c: &mut Criterion) {
    let sizes = [1000, 10_000, 100_000];
    let mut group = c.benchmark_group("bloom_filters");

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
        let dynamic_bloom = DynamicBloom::new(total_bits as u32, 6);
        let fast_bloom = BloomFilter::with_num_bits(total_bits)
            .expected_items(size);

        // Benchmark insertions
        group.bench_function(BenchmarkId::new("dynamic_insert", size), |b| {
            b.iter(|| {
                for item in &items {
                    dynamic_bloom.add_hash(item.parse::<u32>().unwrap());
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

        // Prepare populated filters for lookups
        let mut populated_fast_bloom = fast_bloom.clone();
        for item in &items {
            populated_fast_bloom.insert(item);
        }
        for item in &items {
            dynamic_bloom.add_hash(item.parse::<u32>().unwrap());
        }

        // Benchmark lookups
        group.bench_function(BenchmarkId::new("dynamic_lookup", size), |b| {
            b.iter(|| {
                for item in &lookup_items {
                    let _ = dynamic_bloom.may_contain(item.parse::<u32>().unwrap());
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

        // Benchmark false positives
        let fp_items: Vec<String> = random_numbers(10_000, 44)
            .into_iter()
            .map(|n| n.to_string())
            .collect();

        group.bench_function(BenchmarkId::new("dynamic_fp", size), |b| {
            b.iter(|| {
                let mut fps = 0;
                for item in &fp_items {
                    if dynamic_bloom.may_contain(item.parse::<u32>().unwrap()) {
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
    }
    group.finish();
}

criterion_group!(benches, bench_bloom_filters);
criterion_main!(benches);