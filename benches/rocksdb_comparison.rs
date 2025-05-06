use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use lsm_tree::lsm_tree::{LSMTree, LSMConfig, DynamicBloomFilterConfig};
use lsm_tree::types::{CompactionPolicyType, StorageType, Key, Value};
use lsm_tree::run::{CompressionConfig, AdaptiveCompressionConfig, CompressionType};

// Conditional compilation for RocksDB
// The feature is commented out in Cargo.toml due to system dependencies
#[cfg(feature = "use_rocksdb")]
use rocksdb::{DB, Options, ReadOptions, WriteOptions};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use tempfile::TempDir;
use csv::Writer;
use serde::{Serialize, Deserialize};

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
        
        // Use optimized configuration based on analysis and benchmarking
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
            name: "LSM Tree".to_string(),
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

#[cfg(feature = "use_rocksdb")]
/// RocksDB implementation wrapper
struct RocksDbBenchmark {
    name: String,
    db: DB,
    temp_dir: TempDir,
}

#[cfg(feature = "use_rocksdb")]
impl RocksDbBenchmark {
    fn new() -> Result<Self, String> {
        let temp_dir = TempDir::new().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        
        // Configure RocksDB options to match our LSM tree settings as closely as possible
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_max_open_files(1000);
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64 MB, same as our buffer_size
        opts.set_level_compaction_dynamic_level_bytes(true);
        opts.enable_statistics();
        opts.set_compression_type(rocksdb::DBCompressionType::Snappy);
        
        // Bloom filter settings
        opts.set_bloom_locality(10);
        let bits_per_key = 10.0;
        opts.set_memtable_prefix_bloom_ratio(0.1);
        let filter = rocksdb::BlockBasedOptions::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_bloom_filter(bits_per_key, false);
        block_opts.set_cache_index_and_filter_blocks(true);
        opts.set_block_based_table_factory(&block_opts);
        
        // Set background threads similar to our background_compaction
        opts.increase_parallelism(4);
        
        // Open RocksDB instance
        let db = DB::open(&opts, temp_dir.path()).map_err(|e| format!("Failed to open RocksDB: {}", e))?;
        
        Ok(Self {
            name: std::env::var("DB_NAME").unwrap_or("RocksDB".to_string()),
            db,
            temp_dir,
        })
    }
}

#[cfg(feature = "use_rocksdb")]
impl BenchmarkableDatabase for RocksDbBenchmark {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn put(&mut self, key: Key, value: Value) -> Result<(), String> {
        let key_bytes = key.to_le_bytes();
        let value_bytes = value.to_le_bytes();
        
        let mut write_opts = WriteOptions::default();
        write_opts.disable_wal(true); // Disable write-ahead logging for performance
        
        self.db.put_opt(&key_bytes, &value_bytes, &write_opts)
            .map_err(|e| format!("RocksDB put error: {}", e))
    }
    
    fn get(&self, key: Key) -> Result<Option<Value>, String> {
        let key_bytes = key.to_le_bytes();
        
        let read_opts = ReadOptions::default();
        match self.db.get_opt(&key_bytes, &read_opts) {
            Ok(Some(value_bytes)) => {
                if value_bytes.len() == 8 {
                    let mut value_array = [0u8; 8];
                    value_array.copy_from_slice(&value_bytes);
                    let value = i64::from_le_bytes(value_array);
                    Ok(Some(value))
                } else {
                    Err(format!("Invalid value format: expected 8 bytes, got {}", value_bytes.len()))
                }
            },
            Ok(None) => Ok(None),
            Err(e) => Err(format!("RocksDB get error: {}", e)),
        }
    }
    
    fn delete(&mut self, key: Key) -> Result<(), String> {
        let key_bytes = key.to_le_bytes();
        
        let write_opts = WriteOptions::default();
        self.db.delete_opt(&key_bytes, &write_opts)
            .map_err(|e| format!("RocksDB delete error: {}", e))
    }
    
    fn range(&self, start: Key, end: Key) -> Result<Vec<(Key, Value)>, String> {
        let start_bytes = start.to_le_bytes();
        let end_bytes = end.to_le_bytes();
        
        let mut read_opts = ReadOptions::default();
        read_opts.set_iterate_lower_bound(&start_bytes);
        read_opts.set_iterate_upper_bound(&end_bytes);
        
        let mut results = Vec::new();
        let iterator = self.db.iterator_opt(rocksdb::IteratorMode::Start, read_opts);
        
        for item in iterator {
            match item {
                Ok((key_bytes, value_bytes)) => {
                    if key_bytes.len() == 8 && value_bytes.len() == 8 {
                        let mut key_array = [0u8; 8];
                        let mut value_array = [0u8; 8];
                        key_array.copy_from_slice(&key_bytes);
                        value_array.copy_from_slice(&value_bytes);
                        
                        let key = i64::from_le_bytes(key_array);
                        let value = i64::from_le_bytes(value_array);
                        
                        if key >= start && key < end {
                            results.push((key, value));
                        }
                    }
                },
                Err(e) => return Err(format!("RocksDB range error: {}", e)),
            }
        }
        
        Ok(results)
    }
    
    fn flush(&mut self) -> Result<(), String> {
        self.db.flush()
            .map_err(|e| format!("RocksDB flush error: {}", e))
    }
    
    fn close(self) -> Result<(), String> {
        // RocksDB will be closed when the DB instance is dropped
        // The temp_dir will also be cleaned up when dropped
        Ok(())
    }
}

#[cfg(not(feature = "use_rocksdb"))]
/// Mock RocksDB implementation for when RocksDB is not available
struct RocksDbBenchmark {
    name: String,
    data: std::collections::HashMap<Key, Value>,
    temp_dir: TempDir,
}

#[cfg(not(feature = "use_rocksdb"))]
impl RocksDbBenchmark {
    fn new() -> Result<Self, String> {
        let temp_dir = TempDir::new().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        
        Ok(Self {
            name: std::env::var("DB_NAME").unwrap_or("RocksDB (Mock)".to_string()),
            data: std::collections::HashMap::new(),
            temp_dir,
        })
    }
}

#[cfg(not(feature = "use_rocksdb"))]
impl BenchmarkableDatabase for RocksDbBenchmark {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn put(&mut self, key: Key, value: Value) -> Result<(), String> {
        self.data.insert(key, value);
        Ok(())
    }
    
    fn get(&self, key: Key) -> Result<Option<Value>, String> {
        Ok(self.data.get(&key).copied())
    }
    
    fn delete(&mut self, key: Key) -> Result<(), String> {
        self.data.remove(&key);
        Ok(())
    }
    
    fn range(&self, start: Key, end: Key) -> Result<Vec<(Key, Value)>, String> {
        let mut results = Vec::new();
        for (&k, &v) in &self.data {
            if k >= start && k < end {
                results.push((k, v));
            }
        }
        results.sort_by_key(|&(k, _)| k);
        Ok(results)
    }
    
    fn flush(&mut self) -> Result<(), String> {
        // Simulate a flush by writing to a file
        let path = self.temp_dir.path().join("data.bin");
        let data = serde_json::to_string(&self.data)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        fs::write(&path, data).map_err(|e| format!("Failed to write: {}", e))?;
        Ok(())
    }
    
    fn close(self) -> Result<(), String> {
        Ok(())
    }
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

/// Operation types
enum Operation {
    Put(Key, Value),
    Get(Key),
    Delete(Key),
    Range(Key, Key),
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

/// Main benchmark function for comparing LSM tree against RocksDB
fn rocksdb_comparison_benchmark(c: &mut Criterion) {
    // Define workload sizes for complete testing
    let workload_sizes = [
        (1_000, 100, 10, 10),    // Small workload
        (5_000, 500, 50, 50),    // Medium workload
        (100_000, 10_000, 1_000, 1_000), // Large workload
    ];
    
    let mut group = c.benchmark_group("database_comparison");
    
    // Configure for more consistent sampling
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10); // Minimum required by Criterion
    
    let mut all_results = Vec::new();
    
    // Print conditional compilation status
    #[cfg(feature = "use_rocksdb")]
    println!("Benchmarking with actual RocksDB implementation");
    
    #[cfg(not(feature = "use_rocksdb"))]
    println!("Benchmarking with mock RocksDB implementation (RocksDB not available)");
    
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
        
        // Benchmark LSM Tree
        group.bench_function(
            BenchmarkId::new(format!("lsm_tree_{}", workload_name), ""),
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
                                eprintln!("Failed to create LSM tree: {}", e);
                            }
                        }
                    }
                    
                    total_duration
                })
            },
        );
        
        // Benchmark RocksDB
        group.bench_function(
            BenchmarkId::new(format!("rocksdb_{}", workload_name), ""),
            |b| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::default();
                    
                    for _ in 0..iters {
                        match RocksDbBenchmark::new() {
                            Ok(mut db) => {
                                let start = Instant::now();
                                let results = run_workload(&mut db, &operations);
                                total_duration += start.elapsed();
                                all_results.extend(results);
                                let _ = db.close();
                            },
                            Err(e) => {
                                eprintln!("Failed to create RocksDB: {}", e);
                            }
                        }
                    }
                    
                    total_duration
                })
            },
        );
    }
    
    // Save all collected results to CSV for further analysis
    match save_results_to_csv(&all_results, "sota/rocksdb_comparison_results.csv") {
        Ok(_) => println!("Benchmark results saved to sota/rocksdb_comparison_results.csv"),
        Err(e) => eprintln!("Failed to save benchmark results: {}", e),
    }
    
    group.finish();
}

criterion_group!(benches, rocksdb_comparison_benchmark);
criterion_main!(benches);