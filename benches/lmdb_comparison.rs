use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use lsm_tree::types::{Key, Value};
use lsm_tree::lsm_tree::{LSMTree, LSMConfig, DynamicBloomFilterConfig};
use lsm_tree::run::{CompressionConfig, AdaptiveCompressionConfig};
use lsm_tree::types::{CompactionPolicyType, StorageType};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::io::{self, BufRead};
use std::env;
use tempfile::TempDir;
use csv::Writer;
use serde::{Serialize, Deserialize};

// Import the real LMDB library
use lmdb::{Environment, Database, WriteFlags, EnvironmentFlags, Transaction, Cursor};

// Define a common trait for databases we want to benchmark
trait BenchmarkableDatabase {
    fn name(&self) -> &str;
    fn put(&mut self, key: Key, value: Value) -> Result<(), String>;
    fn get(&self, key: Key) -> Result<Option<Value>, String>;
    fn delete(&mut self, key: Key) -> Result<(), String>;
    fn range(&self, start: Key, end: Key) -> Result<Vec<(Key, Value)>, String>;
    fn flush(&mut self) -> Result<(), String>;
    fn close(self) -> Result<(), String>;
}

// Implementation for our LSM tree
struct LsmTreeBenchmark {
    lsm: LSMTree,
    name: String,
    temp_dir: TempDir,
}

impl LsmTreeBenchmark {
    fn new() -> Result<Self, String> {
        let temp_dir = TempDir::new().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        
        let config = LSMConfig {
            buffer_size: 64, // 64 MB
            storage_type: StorageType::File,
            storage_path: temp_dir.path().to_path_buf(),
            create_path_if_missing: true,
            max_open_files: 1000,
            sync_writes: false,
            fanout: 10,
            compaction_policy: CompactionPolicyType::Tiered,
            compaction_threshold: 4,
            compression: CompressionConfig::default(),
            adaptive_compression: AdaptiveCompressionConfig::default(),
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
        
        // Use environment variable if set, otherwise default to "LSM Tree"
        let name = env::var("DB_NAME").unwrap_or_else(|_| "LSM Tree".to_string());
        
        Ok(Self { 
            lsm: LSMTree::with_config(config),
            name,
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
        Ok(())
    }
}

// Real LMDB implementation
struct LmdbDb {
    env: Environment,
    db: Database,
    name: String,
    temp_dir: TempDir,
}

impl LmdbDb {
    fn new() -> Result<Self, String> {
        let temp_dir = TempDir::new().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        
        // Use environment variable if set, otherwise default to "LMDB"
        let name = env::var("DB_NAME").unwrap_or_else(|_| "LMDB".to_string());
        
        println!("Creating real LMDB database at {:?}", temp_dir.path());
        
        // Create and configure the LMDB environment
        let env = Environment::new()
            .set_flags(EnvironmentFlags::NO_SYNC) // Optimize for benchmark speed
            .set_map_size(1024 * 1024 * 1024) // 1GB map size
            .open(temp_dir.path())
            .map_err(|e| format!("Failed to create LMDB environment: {}", e))?;
        
        // Open default database
        let db = env.open_db(None)
            .map_err(|e| format!("Failed to open LMDB database: {}", e))?;
        
        Ok(Self {
            env,
            db,
            name,
            temp_dir,
        })
    }
    
    // Helper function to convert Key to byte array
    fn key_to_bytes(key: Key) -> [u8; 8] {
        key.to_be_bytes()
    }
    
    // Helper function to convert Value to byte array
    fn value_to_bytes(value: Value) -> [u8; 8] {
        value.to_be_bytes()
    }
    
    // Helper function to convert bytes to Value
    fn bytes_to_value(bytes: &[u8]) -> Result<Value, String> {
        if bytes.len() != 8 {
            return Err(format!("Invalid value byte length: {}", bytes.len()));
        }
        
        let mut value_bytes = [0u8; 8];
        value_bytes.copy_from_slice(bytes);
        Ok(Value::from_be_bytes(value_bytes))
    }
}

impl BenchmarkableDatabase for LmdbDb {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn put(&mut self, key: Key, value: Value) -> Result<(), String> {
        let key_bytes = Self::key_to_bytes(key);
        let value_bytes = Self::value_to_bytes(value);
        
        let mut txn = self.env.begin_rw_txn()
            .map_err(|e| format!("Failed to begin LMDB transaction: {}", e))?;
        
        txn.put(self.db, &key_bytes, &value_bytes, WriteFlags::empty())
            .map_err(|e| format!("LMDB put error: {}", e))?;
        
        txn.commit().map_err(|e| format!("Failed to commit LMDB transaction: {}", e))?;
        
        Ok(())
    }
    
    fn get(&self, key: Key) -> Result<Option<Value>, String> {
        let key_bytes = Self::key_to_bytes(key);
        
        let txn = self.env.begin_ro_txn()
            .map_err(|e| format!("Failed to begin LMDB transaction: {}", e))?;
        
        let result = match txn.get(self.db, &key_bytes) {
            Ok(bytes) => {
                let value = Self::bytes_to_value(bytes)?;
                Some(value)
            },
            Err(lmdb::Error::NotFound) => None,
            Err(e) => return Err(format!("LMDB get error: {}", e)),
        };
        
        drop(txn);
        
        Ok(result)
    }
    
    fn delete(&mut self, key: Key) -> Result<(), String> {
        let key_bytes = Self::key_to_bytes(key);
        
        let mut txn = self.env.begin_rw_txn()
            .map_err(|e| format!("Failed to begin LMDB transaction: {}", e))?;
        
        // LMDB del returns NotFound if the key doesn't exist, but we'll allow that
        match txn.del(self.db, &key_bytes, None) {
            Ok(_) => (),
            Err(lmdb::Error::NotFound) => (),
            Err(e) => return Err(format!("LMDB delete error: {}", e)),
        }
        
        txn.commit().map_err(|e| format!("Failed to commit LMDB transaction: {}", e))?;
        
        Ok(())
    }
    
    fn range(&self, start: Key, end: Key) -> Result<Vec<(Key, Value)>, String> {
        let mut results = Vec::new();
        
        let txn = self.env.begin_ro_txn()
            .map_err(|e| format!("Failed to begin LMDB transaction: {}", e))?;
        
        let mut cursor = txn.open_ro_cursor(self.db)
            .map_err(|e| format!("Failed to open LMDB cursor: {}", e))?;
        
        let start_bytes = Self::key_to_bytes(start);
        
        // Start iterating through the database
        // Items returned by iter_from() are already key-value pairs
        for (key_bytes, value_bytes) in cursor.iter_from(&start_bytes) {
            // Convert key bytes back to Key
            if key_bytes.len() != 8 {
                return Err(format!("Invalid key byte length: {}", key_bytes.len()));
            }
            
            let mut key_array = [0u8; 8];
            key_array.copy_from_slice(key_bytes);
            let key = Key::from_be_bytes(key_array);
            
            // Stop if we've reached the end key
            if key >= end {
                break;
            }
            
            // Convert value bytes back to Value
            let value = Self::bytes_to_value(value_bytes)?;
            
            // Add to results
            results.push((key, value));
            
            // The for loop will move to the next entry automatically
        }
        
        // Clean up
        drop(cursor);
        drop(txn);
        
        Ok(results)
    }
    
    fn flush(&mut self) -> Result<(), String> {
        // Sync the LMDB environment to disk
        self.env.sync(true).map_err(|e| format!("Failed to sync LMDB: {}", e))?;
        Ok(())
    }
    
    fn close(self) -> Result<(), String> {
        // LMDB will be closed when Environment is dropped
        Ok(())
    }
}


/// Operation types for the workload
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
        let file = std::fs::File::open(&self.file_path)
            .map_err(|e| format!("Failed to open workload file: {}", e))?;
        
        let reader = io::BufReader::new(file);
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
    
    // Execute operations with better error handling
    for (i, op) in operations.iter().enumerate() {
        if i % 1000 == 0 && i > 0 {
            println!("Processing operation {}/{} for {}", i, operations.len(), db.name());
        }
        
        match op {
            Operation::Put(key, value) => {
                let start = Instant::now();
                if let Err(e) = db.put(*key, *value) {
                    eprintln!("Error in put operation for {}: {}", db.name(), e);
                    continue;
                }
                put_time += start.elapsed();
                put_count += 1;
            },
            Operation::Get(key) => {
                let start = Instant::now();
                if let Err(e) = db.get(*key) {
                    eprintln!("Error in get operation for {}: {}", db.name(), e);
                    continue;
                }
                get_time += start.elapsed();
                get_count += 1;
            },
            Operation::Delete(key) => {
                let start = Instant::now();
                if let Err(e) = db.delete(*key) {
                    eprintln!("Error in delete operation for {}: {}", db.name(), e);
                    continue;
                }
                delete_time += start.elapsed();
                delete_count += 1;
            },
            Operation::Range(start_key, end_key) => {
                let start = Instant::now();
                if let Err(e) = db.range(*start_key, *end_key) {
                    eprintln!("Error in range operation for {}: {}", db.name(), e);
                    continue;
                }
                range_time += start.elapsed();
                range_count += 1;
            },
        }
    }
    
    println!("Flushing database {}...", db.name());
    // Flush after all operations
    if let Err(e) = db.flush() {
        eprintln!("Error flushing {}: {}", db.name(), e);
    }
    
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

/// Main benchmark function for comparing LSM tree against LMDB
fn lmdb_comparison_benchmark(c: &mut Criterion) {
    // Make sure LMDB dependencies are installed
    println!("Ensuring LMDB dependencies are installed...");
    let install_script_path = std::path::PathBuf::from("sota/install_lmdb_deps.sh");
    if !install_script_path.exists() {
        eprintln!("Warning: LMDB installation script not found at {:?}", install_script_path);
    } else {
        // Run the installation script
        match std::process::Command::new("bash")
            .arg(&install_script_path)
            .status() 
        {
            Ok(status) => {
                if !status.success() {
                    eprintln!("Warning: LMDB installation script failed with exit code: {:?}", status.code());
                } else {
                    println!("LMDB dependencies installed successfully.");
                }
            },
            Err(e) => {
                eprintln!("Warning: Failed to run LMDB installation script: {}", e);
            }
        }
    }
    // Define a moderate workload size for testing
    // We'll use medium size for reasonable completion time
    let workload_sizes = [
        (1_000, 100, 10, 10),      // Small workload
        (5_000, 500, 50, 50),      // Medium workload
        // (100_000, 10_000, 1_000, 1_000), // Large workload (commented out for faster testing)
    ];
    
    let mut group = c.benchmark_group("database_comparison");
    
    // Configure for faster benchmarking with reasonable accuracy
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10); // Minimum required by Criterion
    group.measurement_time(Duration::from_secs(5)); // Shorter measurement time
    
    let mut all_results = Vec::new();
    
    println!("Benchmarking with real LMDB implementation (using lmdb v0.8.0)");
    
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
        
        // Benchmark LMDB
        group.bench_function(
            BenchmarkId::new(format!("lmdb_{}", workload_name), ""),
            |b| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::default();
                    
                    for i in 0..iters {
                        println!("Running LMDB benchmark iteration {}/{}", i+1, iters);
                        match LmdbDb::new() {
                            Ok(mut db) => {
                                let start = Instant::now();
                                let results = run_workload(&mut db, &operations);
                                total_duration += start.elapsed();
                                all_results.extend(results);
                                
                                // Log success
                                println!("Completed LMDB benchmark iteration {}/{}", i+1, iters);
                                
                                if let Err(e) = db.close() {
                                    eprintln!("Warning: Failed to close LMDB: {}", e);
                                }
                            },
                            Err(e) => {
                                eprintln!("Failed to create LMDB: {}", e);
                            }
                        }
                    }
                    
                    total_duration
                })
            },
        );
    }
    
    // Save all collected results to CSV for further analysis
    match save_results_to_csv(&all_results, "sota/benchmark_results/lmdb_comparison_results.csv") {
        Ok(_) => println!("Benchmark results saved to sota/benchmark_results/lmdb_comparison_results.csv"),
        Err(e) => eprintln!("Failed to save benchmark results: {}", e),
    }
    
    group.finish();
}

criterion_group!(benches, lmdb_comparison_benchmark);
criterion_main!(benches);