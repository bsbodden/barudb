use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use lsm_tree::lsm_tree::{LSMTree, LSMConfig, DynamicBloomFilterConfig};
use lsm_tree::types::{Key, Value};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tempfile::tempdir;

fn random_keys(num: usize, seed: u64) -> Vec<Key> {
    let mut rng = StdRng::seed_from_u64(seed);
    std::iter::repeat_with(|| rng.random()).take(num).collect()
}

fn bench_dynamic_sizing(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_bloom_sizing");
    
    // Configure for more consistent sampling
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(50);
    
    // Test with different target false positive rates
    let test_scenarios = [
        ("strict_fp_rate", vec![0.001, 0.005, 0.01]),  // Strict FP rates
        ("medium_fp_rate", vec![0.01, 0.03, 0.05]),    // Medium FP rates
        ("relaxed_fp_rate", vec![0.05, 0.10, 0.15]),   // Relaxed FP rates
    ];
    
    // Create test data for different scenarios
    let initial_data_size = 10_000;
    let initial_data: Vec<(Key, Value)> = (0..initial_data_size)
        .map(|i| (i as Key, (i * 10) as Value))
        .collect();
    
    let second_batch: Vec<(Key, Value)> = (initial_data_size..initial_data_size * 2)
        .map(|i| (i as Key, (i * 10) as Value))
        .collect();
    
    // Generate random existing keys for lookups (80% from first batch, 20% from second batch)
    let mut rng = StdRng::seed_from_u64(42);
    let existing_keys: Vec<Key> = (0..5000)
        .map(|_| if rng.random_bool(0.8) {
                rng.random_range(0..initial_data_size) as Key
             } else {
                rng.random_range(initial_data_size..initial_data_size * 2) as Key
             })
        .collect();
    
    // Generate random non-existent keys for lookups - increased to 20,000 for better false positive detection
    let nonexistent_keys: Vec<Key> = (0..20000)
        .map(|_| rng.random_range(initial_data_size * 2 + 1000..initial_data_size * 3) as Key)
        .collect();
    
    for (scenario_name, target_fp_rates) in test_scenarios {
        // Create a new temporary directory for each scenario
        let temp_dir = tempdir().unwrap();
        
        // Create LSM tree with dynamic bloom filter enabled
        let mut config = LSMConfig::default();
        config.storage_path = temp_dir.path().to_path_buf();
        config.dynamic_bloom_filter = DynamicBloomFilterConfig {
            enabled: true,
            target_fp_rates: target_fp_rates.clone(),
            min_bits_per_entry: 4.0,
            max_bits_per_entry: 20.0,
            min_sample_size: 100, // Lower sample size for benchmarking
        };
        
        let mut dynamic_lsm = LSMTree::with_config(config.clone());
        
        // Also create a control LSM tree with standard Monkey optimization (not dynamic)
        config.dynamic_bloom_filter.enabled = false;
        let mut standard_lsm = LSMTree::with_config(config);
        
        // Add initial data to both trees
        for (key, value) in &initial_data {
            dynamic_lsm.put(*key, *value).unwrap();
            standard_lsm.put(*key, *value).unwrap();
        }
        
        // Flush data to level 0
        dynamic_lsm.flush_buffer_to_level0().unwrap();
        standard_lsm.flush_buffer_to_level0().unwrap();
        
        // Add second batch of data
        for (key, value) in &second_batch {
            dynamic_lsm.put(*key, *value).unwrap();
            standard_lsm.put(*key, *value).unwrap();
        }
        
        // Flush again
        dynamic_lsm.flush_buffer_to_level0().unwrap();
        standard_lsm.flush_buffer_to_level0().unwrap();
        
        // Force compaction to create multiple levels
        dynamic_lsm.force_compact_all().unwrap();
        standard_lsm.force_compact_all().unwrap();
        
        // First benchmark "training phase" - this is where the dynamic filter will learn
        // We don't measure this as part of the benchmark since this is just setup
        // This is simulating the system running under normal operations
        // Increased to 10 iterations to give dynamic filters more time to adapt
        println!("\n{} - Training Phase (Dynamic filters learning):", scenario_name);
        for i in 0..10 {
            // Make 30% of lookups for existing keys, 70% for non-existing keys
            // This better simulates real-world scenarios with higher negative lookup rates
            
            // Existing key lookups (subset)
            let start_idx = (i * 500) % (existing_keys.len() - 500);
            for &key in &existing_keys[start_idx..start_idx + 500] {
                dynamic_lsm.get(key);
                standard_lsm.get(key);
            }
            
            // Non-existing key lookups (subset)
            let start_idx = (i * 1000) % (nonexistent_keys.len() - 1000);
            for &key in &nonexistent_keys[start_idx..start_idx + 1000] {
                dynamic_lsm.get(key);
                standard_lsm.get(key);
            }
            
            // Check filter stats every few iterations to see adaptation in progress
            if i % 3 == 2 {
                if let Some(stats) = dynamic_lsm.get_filter_stats_summary() {
                    println!("   Iteration {} - Dynamic Filter FP Rate: {:.4}%, Bits per entry: {:.2}",
                            i + 1, 
                            stats.overall_fp_rate * 100.0,
                            stats.level_stats.first().map_or(0.0, |s| s.avg_bits_per_entry));
                }
                
                if let Some(stats) = standard_lsm.get_filter_stats_summary() {
                    println!("   Iteration {} - Standard Filter FP Rate: {:.4}%, Bits per entry: {:.2}",
                            i + 1, 
                            stats.overall_fp_rate * 100.0,
                            stats.level_stats.first().map_or(0.0, |s| s.avg_bits_per_entry));
                }
            }
        }
        
        // Now benchmark actual performance after the dynamic filter has adapted
        
        // Benchmark gets with existing keys - Dynamic
        group.bench_function(BenchmarkId::new(format!("{}_dynamic", scenario_name), "existing_keys"), |b| {
            b.iter(|| {
                for &key in &existing_keys[0..500] { // Increased to 500 for more comprehensive benchmarking
                    criterion::black_box(dynamic_lsm.get(key));
                }
            });
        });
        
        // Benchmark gets with existing keys - Standard
        group.bench_function(BenchmarkId::new(format!("{}_standard", scenario_name), "existing_keys"), |b| {
            b.iter(|| {
                for &key in &existing_keys[0..500] { // Increased to 500 for more comprehensive benchmarking
                    criterion::black_box(standard_lsm.get(key));
                }
            });
        });
        
        // Benchmark gets with non-existent keys - Dynamic (this is where bloom filter helps the most)
        group.bench_function(BenchmarkId::new(format!("{}_dynamic", scenario_name), "nonexistent_keys"), |b| {
            b.iter(|| {
                for &key in &nonexistent_keys[0..500] { // Increased to 500 for more comprehensive benchmarking
                    criterion::black_box(dynamic_lsm.get(key));
                }
            });
        });
        
        // Benchmark gets with non-existent keys - Standard
        group.bench_function(BenchmarkId::new(format!("{}_standard", scenario_name), "nonexistent_keys"), |b| {
            b.iter(|| {
                for &key in &nonexistent_keys[0..500] { // Increased to 500 for more comprehensive benchmarking
                    criterion::black_box(standard_lsm.get(key));
                }
            });
        });
        
        // Get and report filter statistics for both trees
        let dynamic_stats = dynamic_lsm.get_filter_stats_summary();
        let standard_stats = standard_lsm.get_filter_stats_summary();
        
        println!("\n{} - FINAL COMPARISON:", scenario_name);
        
        // Extract key metrics for comparison
        let dynamic_fp_rate = dynamic_stats.as_ref().map_or(0.0, |s| s.overall_fp_rate * 100.0);
        let standard_fp_rate = standard_stats.as_ref().map_or(0.0, |s| s.overall_fp_rate * 100.0);
        
        let dynamic_bits = dynamic_stats.as_ref().map_or(0.0, |s| 
            s.level_stats.iter().map(|l| l.avg_bits_per_entry * l.runs as f64).sum::<f64>()
        );
        let standard_bits = standard_stats.as_ref().map_or(0.0, |s| 
            s.level_stats.iter().map(|l| l.avg_bits_per_entry * l.runs as f64).sum::<f64>()
        );
        
        println!("┌─────────────────────────┬────────────┬───────────────┐");
        println!("│ Metric                  │ Dynamic    │ Standard      │");
        println!("├─────────────────────────┼────────────┼───────────────┤");
        println!("│ False Positive Rate     │ {:.4}%     │ {:.4}%        │", 
                 dynamic_fp_rate, standard_fp_rate);
        
        // Calculate average bits per entry for each level
        if let (Some(dstats), Some(sstats)) = (dynamic_stats.as_ref(), standard_stats.as_ref()) {
            for level in 0..dstats.level_stats.len().max(sstats.level_stats.len()) {
                let d_bits = dstats.level_stats.get(level).map_or(0.0, |s| s.avg_bits_per_entry);
                let s_bits = sstats.level_stats.get(level).map_or(0.0, |s| s.avg_bits_per_entry);
                
                println!("│ Level {} Bits per entry   │ {:.2}       │ {:.2}          │", 
                         level, d_bits, s_bits);
            }
        }
        
        // Calculate memory savings
        let memory_saving_pct = if standard_bits > 0.0 {
            (1.0 - (dynamic_bits / standard_bits)) * 100.0
        } else {
            0.0
        };
        
        println!("│ Total memory bits       │ {:.2}   │ {:.2}      │", 
                 dynamic_bits, standard_bits);
        println!("│ Memory savings          │ {:.2}%     │               │", 
                 memory_saving_pct);
        println!("└─────────────────────────┴────────────┴───────────────┘");
        
        // Detailed statistics
        if let Some(stats) = dynamic_stats {
            println!("\n{} - Dynamic Bloom Filter Stats (Detailed):", scenario_name);
            println!("   Total checks: {}", stats.total_checks);
            println!("   Observed false positive rate: {:.4}%", stats.overall_fp_rate * 100.0);
            
            for level_stat in &stats.level_stats {
                println!("   Level {} - Bits per entry: {:.2}, Observed FP rate: {:.4}%, Theoretical FP rate: {:.4}%",
                         level_stat.level, 
                         level_stat.avg_bits_per_entry,
                         level_stat.observed_fp_rate * 100.0,
                         level_stat.theoretical_fp_rate * 100.0);
            }
        }
        
        if let Some(stats) = standard_stats {
            println!("\n{} - Standard Bloom Filter Stats (Detailed):", scenario_name);
            println!("   Total checks: {}", stats.total_checks);
            println!("   Observed false positive rate: {:.4}%", stats.overall_fp_rate * 100.0);
            
            for level_stat in &stats.level_stats {
                println!("   Level {} - Bits per entry: {:.2}, Observed FP rate: {:.4}%, Theoretical FP rate: {:.4}%",
                         level_stat.level, 
                         level_stat.avg_bits_per_entry,
                         level_stat.observed_fp_rate * 100.0,
                         level_stat.theoretical_fp_rate * 100.0);
            }
        }
    }
    
    group.finish();
}

criterion_group!(benches, bench_dynamic_sizing);
criterion_main!(benches);