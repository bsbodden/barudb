use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use lsm_tree::lsm_tree::{LSMTree, LSMConfig, DynamicBloomFilterConfig};
use lsm_tree::types::{CompactionPolicyType, StorageType, Key, Value, Result as LsmResult};
use lsm_tree::run::{CompressionConfig, AdaptiveCompressionConfig, CompressionType};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use tempfile::TempDir;
use csv::Writer;
use serde::{Serialize, Deserialize};
use std::sync::Arc;

/// Database interface for benchmarking
trait BenchmarkableDatabase {
    fn name(&self) -> &str;
    fn put(&mut self, key: Key, value: Value) -> Result<(), String>;
    fn get(&self, key: Key) -> Result<Option<Value>, String>;
    fn delete(&mut self, key: Key) -> Result<(), String>;
    fn range(&self, start: Key, end: Key) -> Result<Vec<(Key, Value)>, String>;
    fn flush(&mut self) -> Result<(), String>;
    fn close(self) -> Result<(), String>;
}

/// Our LSM Tree implementation wrapper
struct LsmTreeBenchmark {
    name: String,
    lsm: LSMTree,
    temp_dir: TempDir,
}

impl LsmTreeBenchmark {
    fn new() -> Result<Self, String> {
        let temp_dir = TempDir::new().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        
        // Create an optimized configuration based on our analysis
        let config = LSMConfig {
            buffer_size: 64, // 64 MB
            storage_type: StorageType::File,
            storage_path: temp_dir.path().to_path_buf(),
            create_path_if_missing: true,
            max_open_files: 1000,
            sync_writes: false,
            fanout: 10,
            compaction_policy: CompactionPolicyType::LazyLeveled, // Optimal for mixed workloads
            compaction_threshold: 4,
            compression: CompressionConfig {
                enabled: true,
                l0_default: CompressionType::BitPack, // Best balance of compression ratio/speed
                lower_level_default: CompressionType::BitPack,
                block_size: 4096, // Larger block size for better compression
                ..Default::default()
            },
            adaptive_compression: AdaptiveCompressionConfig {
                enabled: true, // Enable adaptive compression
                sample_size: 1000, // Larger sample size for better algorithm selection
                min_compression_ratio: 1.2, // Only compress if we get at least 20% improvement
                min_size_threshold: 512, // Minimum size to attempt compression
                ..Default::default()
            },
            collect_compression_stats: false,
            background_compaction: true,
            use_lock_free_memtable: true, // Better for write-heavy workloads
            use_lock_free_block_cache: true, // 5x faster than traditional block cache
            dynamic_bloom_filter: DynamicBloomFilterConfig {
                enabled: true,
                target_fp_rates: vec![0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
                min_bits_per_entry: 2.0,
                max_bits_per_entry: 10.0,
                min_sample_size: 1000,
            },
        };
        
        Ok(Self {
            name: "Optimized LSM Tree".to_string(),
            lsm: LSMTree::with_config(config),
            temp_dir,
        })
    }
}

impl BenchmarkableDatabase for LsmTreeBenchmark {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn put(&mut self, key: Key, value: Value) -> Result<(), String> {
        self.lsm.put(key, value).map_err(|e| format!("LSM put error: {}", e))
    }
    
    fn get(&self, key: Key) -> Result<Option<Value>, String> {
        Ok(self.lsm.get(key))
    }
    
    fn delete(&mut self, key: Key) -> Result<(), String> {
        self.lsm.delete(key).map_err(|e| format!("LSM delete error: {}", e))
    }
    
    fn range(&self, start: Key, end: Key) -> Result<Vec<(Key, Value)>, String> {
        Ok(self.lsm.range(start, end))
    }
    
    fn flush(&mut self) -> Result<(), String> {
        self.lsm.flush_buffer_to_level0().map_err(|e| format!("LSM flush error: {}", e))
    }
    
    fn close(self) -> Result<(), String> {
        // LSM Tree will automatically close when dropped
        // The temp_dir will also be cleaned up when dropped
        Ok(())
    }
}

/// Standard LSM Tree implementation with default configuration
struct DefaultLsmTreeBenchmark {
    name: String,
    lsm: LSMTree,
    temp_dir: TempDir,
}

impl DefaultLsmTreeBenchmark {
    fn new() -> Result<Self, String> {
        let temp_dir = TempDir::new().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        
        // Create a configuration matching what's used in the existing benchmarks
        let config = LSMConfig {
            buffer_size: 64, // 64 MB
            storage_type: StorageType::File,
            storage_path: temp_dir.path().to_path_buf(),
            create_path_if_missing: true,
            max_open_files: 1000,
            sync_writes: false,
            fanout: 10,
            compaction_policy: CompactionPolicyType::Tiered, // Default used in benchmarks
            compaction_threshold: 4,
            compression: CompressionConfig::default(), // No compression
            adaptive_compression: AdaptiveCompressionConfig::default(), // No adaptive compression
            collect_compression_stats: false,
            background_compaction: true,
            use_lock_free_memtable: true,
            use_lock_free_block_cache: true,
            dynamic_bloom_filter: DynamicBloomFilterConfig {
                enabled: true,
                target_fp_rates: vec![0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
                min_bits_per_entry: 2.0,
                max_bits_per_entry: 10.0,
                min_sample_size: 1000,
            },
        };
        
        Ok(Self {
            name: "Default LSM Tree".to_string(),
            lsm: LSMTree::with_config(config),
            temp_dir,
        })
    }
}

impl BenchmarkableDatabase for DefaultLsmTreeBenchmark {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn put(&mut self, key: Key, value: Value) -> Result<(), String> {
        self.lsm.put(key, value).map_err(|e| format!("LSM put error: {}", e))
    }
    
    fn get(&self, key: Key) -> Result<Option<Value>, String> {
        Ok(self.lsm.get(key))
    }
    
    fn delete(&mut self, key: Key) -> Result<(), String> {
        self.lsm.delete(key).map_err(|e| format!("LSM delete error: {}", e))
    }
    
    fn range(&self, start: Key, end: Key) -> Result<Vec<(Key, Value)>, String> {
        Ok(self.lsm.range(start, end))
    }
    
    fn flush(&mut self) -> Result<(), String> {
        self.lsm.flush_buffer_to_level0().map_err(|e| format!("LSM flush error: {}", e))
    }
    
    fn close(self) -> Result<(), String> {
        // LSM Tree will automatically close when dropped
        // The temp_dir will also be cleaned up when dropped
        Ok(())
    }
}

/// Operation types
enum Operation {
    Put(Key, Value),
    Get(Key),
    Delete(Key),
    Range(Key, Key),
}

/// Workload generator output parser
struct WorkloadParser {
    file_path: PathBuf,
}

impl WorkloadParser {
    fn new(file_path: PathBuf) -> Self {
        Self {
            file_path,
        }
    }
    
    fn parse(&self) -> Result<Vec<Operation>, String> {
        let file = File::open(&self.file_path)
            .map_err(|e| format!("Failed to open workload file: {}", e))?;
        
        let reader = BufReader::new(file);
        let mut operations = Vec::new();
        
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            
            if parts.is_empty() {
                continue;
            }
            
            match parts[0] {
                "p" => {
                    if parts.len() >= 3 {
                        let key = parts[1].parse::<Key>()
                            .map_err(|e| format!("Invalid key: {}", e))?;
                        let value = parts[2].parse::<Value>()
                            .map_err(|e| format!("Invalid value: {}", e))?;
                        operations.push(Operation::Put(key, value));
                    } else {
                        return Err("Invalid PUT operation format".to_string());
                    }
                },
                "g" => {
                    if parts.len() >= 2 {
                        let key = parts[1].parse::<Key>()
                            .map_err(|e| format!("Invalid key: {}", e))?;
                        operations.push(Operation::Get(key));
                    } else {
                        return Err("Invalid GET operation format".to_string());
                    }
                },
                "d" => {
                    if parts.len() >= 2 {
                        let key = parts[1].parse::<Key>()
                            .map_err(|e| format!("Invalid key: {}", e))?;
                        operations.push(Operation::Delete(key));
                    } else {
                        return Err("Invalid DELETE operation format".to_string());
                    }
                },
                "r" => {
                    if parts.len() >= 3 {
                        let start = parts[1].parse::<Key>()
                            .map_err(|e| format!("Invalid start key: {}", e))?;
                        let end = parts[2].parse::<Key>()
                            .map_err(|e| format!("Invalid end key: {}", e))?;
                        operations.push(Operation::Range(start, end));
                    } else {
                        return Err("Invalid RANGE operation format".to_string());
                    }
                },
                _ => {
                    // Ignore unknown operations
                }
            }
        }
        
        Ok(operations)
    }
}

/// Benchmark result data structure
#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    db_name: String,
    operation: String,
    count: usize,
    total_time_micros: u128,
    avg_time_micros: f64,
    throughput_ops_per_sec: f64,
}

/// Generate a workload file using the generator
fn generate_workload(puts: usize, gets: usize, ranges: usize, deletes: usize) -> Result<PathBuf, String> {
    let output_path = PathBuf::from("generator/benchmark_workload.txt");
    
    let status = std::process::Command::new("./generator/generator")
        .args(&[
            "--puts", &puts.to_string(),
            "--gets", &gets.to_string(),
            "--ranges", &ranges.to_string(),
            "--deletes", &deletes.to_string(),
            "--gets-misses-ratio", "0.1",
            "--gets-skewness", "0.2",
            "--max-range-size", "100",
        ])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn generator: {}", e))?
        .wait_with_output()
        .map_err(|e| format!("Failed to execute generator: {}", e))?;
    
    if !status.status.success() {
        return Err("Generator failed to execute".to_string());
    }
    
    // Write the output to a file
    fs::write(&output_path, &status.stdout)
        .map_err(|e| format!("Failed to write workload file: {}", e))?;
    
    Ok(output_path)
}

/// Run a workload against a database
fn run_workload<T: BenchmarkableDatabase>(
    db: &mut T,
    operations: &[Operation],
) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    // Track operation counts and times
    let mut put_count = 0;
    let mut get_count = 0;
    let mut delete_count = 0;
    let mut range_count = 0;
    
    let mut put_time = Duration::default();
    let mut get_time = Duration::default();
    let mut delete_time = Duration::default();
    let mut range_time = Duration::default();
    
    // Execute operations
    for op in operations {
        match op {
            Operation::Put(key, value) => {
                let start = Instant::now();
                let _ = db.put(*key, *value);
                put_time += start.elapsed();
                put_count += 1;
            },
            Operation::Get(key) => {
                let start = Instant::now();
                let _ = db.get(*key);
                get_time += start.elapsed();
                get_count += 1;
            },
            Operation::Delete(key) => {
                let start = Instant::now();
                let _ = db.delete(*key);
                delete_time += start.elapsed();
                delete_count += 1;
            },
            Operation::Range(start_key, end_key) => {
                let start = Instant::now();
                let _ = db.range(*start_key, *end_key);
                range_time += start.elapsed();
                range_count += 1;
            },
        }
    }
    
    // Flush after all operations
    let _ = db.flush();
    
    // Calculate metrics and create results
    if put_count > 0 {
        let total_micros = put_time.as_micros();
        let avg_micros = total_micros as f64 / put_count as f64;
        let throughput = 1_000_000.0 / avg_micros;
        
        results.push(BenchmarkResult {
            db_name: db.name().to_string(),
            operation: "put".to_string(),
            count: put_count,
            total_time_micros: total_micros,
            avg_time_micros: avg_micros,
            throughput_ops_per_sec: throughput,
        });
    }
    
    if get_count > 0 {
        let total_micros = get_time.as_micros();
        let avg_micros = total_micros as f64 / get_count as f64;
        let throughput = 1_000_000.0 / avg_micros;
        
        results.push(BenchmarkResult {
            db_name: db.name().to_string(),
            operation: "get".to_string(),
            count: get_count,
            total_time_micros: total_micros,
            avg_time_micros: avg_micros,
            throughput_ops_per_sec: throughput,
        });
    }
    
    if delete_count > 0 {
        let total_micros = delete_time.as_micros();
        let avg_micros = total_micros as f64 / delete_count as f64;
        let throughput = 1_000_000.0 / avg_micros;
        
        results.push(BenchmarkResult {
            db_name: db.name().to_string(),
            operation: "delete".to_string(),
            count: delete_count,
            total_time_micros: total_micros,
            avg_time_micros: avg_micros,
            throughput_ops_per_sec: throughput,
        });
    }
    
    if range_count > 0 {
        let total_micros = range_time.as_micros();
        let avg_micros = total_micros as f64 / range_count as f64;
        let throughput = 1_000_000.0 / avg_micros;
        
        results.push(BenchmarkResult {
            db_name: db.name().to_string(),
            operation: "range".to_string(),
            count: range_count,
            total_time_micros: total_micros,
            avg_time_micros: avg_micros,
            throughput_ops_per_sec: throughput,
        });
    }
    
    results
}

/// Save benchmark results to a CSV file
fn save_results_to_csv(results: &[BenchmarkResult], filename: &str) -> Result<(), String> {
    let mut writer = Writer::from_path(filename)
        .map_err(|e| format!("Failed to create CSV writer: {}", e))?;
    
    // Write header
    writer.write_record(&["db_name", "operation", "count", "total_time_micros", "avg_time_micros", "throughput_ops_per_sec"])
        .map_err(|e| format!("Failed to write CSV header: {}", e))?;
    
    // Write data
    for result in results {
        writer.write_record(&[
            &result.db_name,
            &result.operation,
            &result.count.to_string(),
            &result.total_time_micros.to_string(),
            &result.avg_time_micros.to_string(),
            &result.throughput_ops_per_sec.to_string(),
        ])
        .map_err(|e| format!("Failed to write CSV record: {}", e))?;
    }
    
    writer.flush().map_err(|e| format!("Failed to flush CSV: {}", e))?;
    Ok(())
}

/// Main benchmark function for comparing optimized vs default configurations
fn config_comparison_benchmark(c: &mut Criterion) {
    // Define workload sizes for complete testing
    let workload_sizes = [
        (1_000, 100, 10, 10),    // Small workload
        (5_000, 500, 50, 50),    // Medium workload
        (100_000, 10_000, 1_000, 1_000), // Large workload
    ];
    
    let mut group = c.benchmark_group("configuration_comparison");
    
    // Configure for more consistent sampling
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10); // Minimum required by Criterion
    
    let mut all_results = Vec::new();
    
    for (puts, gets, ranges, deletes) in workload_sizes {
        let workload_name = format!("workload_{}k", puts / 1000);
        
        // Generate workload
        println!("Generating workload with {} puts, {} gets, {} ranges, {} deletes", puts, gets, ranges, deletes);
        let workload_path = match generate_workload(puts, gets, ranges, deletes) {
            Ok(path) => path,
            Err(e) => {
                eprintln!("Failed to generate workload: {}", e);
                continue;
            }
        };
        
        // Parse workload
        let parser = WorkloadParser::new(workload_path.clone());
        let operations = match parser.parse() {
            Ok(ops) => ops,
            Err(e) => {
                eprintln!("Failed to parse workload: {}", e);
                continue;
            }
        };
        
        // Benchmark Default LSM Tree Configuration
        group.bench_function(
            BenchmarkId::new(format!("default_lsm_tree_{}", workload_name), ""),
            |b| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::default();
                    
                    for _ in 0..iters {
                        match DefaultLsmTreeBenchmark::new() {
                            Ok(mut db) => {
                                let start = Instant::now();
                                let results = run_workload(&mut db, &operations);
                                total_duration += start.elapsed();
                                all_results.extend(results);
                                let _ = db.close();
                            },
                            Err(e) => {
                                eprintln!("Failed to create default LSM tree: {}", e);
                            }
                        }
                    }
                    
                    total_duration
                })
            },
        );
        
        // Benchmark Optimized LSM Tree Configuration
        group.bench_function(
            BenchmarkId::new(format!("optimized_lsm_tree_{}", workload_name), ""),
            |b| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::default();
                    
                    for _ in 0..iters {
                        match LsmTreeBenchmark::new() {
                            Ok(mut db) => {
                                let start = Instant::now();
                                let results = run_workload(&mut db, &operations);
                                total_duration += start.elapsed();
                                all_results.extend(results);
                                let _ = db.close();
                            },
                            Err(e) => {
                                eprintln!("Failed to create optimized LSM tree: {}", e);
                            }
                        }
                    }
                    
                    total_duration
                })
            },
        );
    }
    
    // Save all collected results to CSV for further analysis
    match save_results_to_csv(&all_results, "sota/optimal_config_comparison_results.csv") {
        Ok(_) => println!("Benchmark results saved to sota/optimal_config_comparison_results.csv"),
        Err(e) => eprintln!("Failed to save benchmark results: {}", e),
    }
    
    group.finish();
}

criterion_group!(benches, config_comparison_benchmark);
criterion_main!(benches);