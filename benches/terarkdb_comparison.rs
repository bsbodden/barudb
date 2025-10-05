#![feature(test)]
extern crate test;
extern crate csv; // For CSV writing
extern crate fastrand; // For random number generation

// Using the crate name as defined in Cargo.toml
use barudb::{Key, Value, command::{Command, Response}};
use barudb::lsm_tree::{LsmConfig, LsmTree};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::io::{self, BufRead};
use std::env;
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;
use std::str;

// Import TerarkDB if the feature is enabled
#[cfg(feature = "use_terarkdb")]
extern crate terarkdb_rust as terarkdb;
#[cfg(feature = "use_terarkdb")]
use std::os::raw::{c_char, c_uchar, c_void};

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
struct SpeedDb {
    db: LsmTree,
    name: String,
}

impl SpeedDb {
    fn new(path: &str) -> Result<Self, String> {
        let config = LsmConfig {
            // Configure with desired options
            bloom_filter_bits: 10,
            fanout: 10,
            max_levels: 8,
            ..Default::default()
        };
        
        let db = LsmTree::open(path, config).map_err(|e| format!("Failed to open SpeedDB: {}", e))?;
        
        // Use environment variable if set, otherwise default to "LSM Tree"
        let name = env::var("DB_NAME").unwrap_or_else(|_| "LSM Tree".to_string());
        
        Ok(Self { db, name })
    }
}

impl BenchmarkableDatabase for SpeedDb {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn put(&mut self, key: Key, value: Value) -> Result<(), String> {
        let command = Command::Put { key, value };
        if let Response::Put(result) = self.db.execute(command) {
            result.map_err(|e| format!("Put error: {}", e))
        } else {
            Err("Unexpected response type".to_string())
        }
    }
    
    fn get(&self, key: Key) -> Result<Option<Value>, String> {
        let command = Command::Get { key };
        if let Response::Get(result) = self.db.execute(command) {
            result.map_err(|e| format!("Get error: {}", e))
        } else {
            Err("Unexpected response type".to_string())
        }
    }
    
    fn delete(&mut self, key: Key) -> Result<(), String> {
        let command = Command::Delete { key };
        if let Response::Delete(result) = self.db.execute(command) {
            result.map_err(|e| format!("Delete error: {}", e))
        } else {
            Err("Unexpected response type".to_string())
        }
    }
    
    fn range(&self, start: Key, end: Key) -> Result<Vec<(Key, Value)>, String> {
        let command = Command::Range { start, end };
        if let Response::Range(result) = self.db.execute(command) {
            result.map_err(|e| format!("Range error: {}", e))
        } else {
            Err("Unexpected response type".to_string())
        }
    }
    
    fn flush(&mut self) -> Result<(), String> {
        let command = Command::Flush;
        if let Response::Flush(result) = self.db.execute(command) {
            result.map_err(|e| format!("Flush error: {}", e))
        } else {
            Err("Unexpected response type".to_string())
        }
    }
    
    fn close(self) -> Result<(), String> {
        // LsmTree doesn't need explicit closing
        Ok(())
    }
}

// TerarkDB implementation conditionally included only if the feature is enabled
#[cfg(feature = "use_terarkdb")]
struct TerarkDb {
    db: *mut terarkdb::rocksdb_t,
    options: *mut terarkdb::rocksdb_options_t,
    read_options: *mut terarkdb::rocksdb_readoptions_t,
    write_options: *mut terarkdb::rocksdb_writeoptions_t,
    name: String,
    path: CString,
}

#[cfg(feature = "use_terarkdb")]
impl TerarkDb {
    fn new(path: &str) -> Result<Self, String> {
        // Create the directory if it doesn't exist
        fs::create_dir_all(path).map_err(|e| format!("Failed to create directory: {}", e))?;
        
        let path_cstring = CString::new(path).map_err(|e| format!("Failed to create path CString: {}", e))?;
        
        unsafe {
            // Create options
            let options = terarkdb::rocksdb_options_create();
            terarkdb::rocksdb_options_set_create_if_missing(options, 1);
            
            // Create read/write options
            let read_options = terarkdb::rocksdb_readoptions_create();
            let write_options = terarkdb::rocksdb_writeoptions_create();
            
            // Open database
            let mut error: *mut c_char = ptr::null_mut();
            let db = terarkdb::rocksdb_open(options, path_cstring.as_ptr(), &mut error);
            
            if !error.is_null() {
                let error_message = CStr::from_ptr(error).to_string_lossy().into_owned();
                terarkdb::rocksdb_free(error as *mut c_void);
                return Err(format!("Failed to open TerarkDB: {}", error_message));
            }
            
            // Use environment variable if set, otherwise default to "TerarkDB"
            let name = env::var("DB_NAME").unwrap_or_else(|_| "TerarkDB".to_string());
            
            Ok(Self {
                db,
                options,
                read_options,
                write_options,
                name,
                path: path_cstring,
            })
        }
    }
    
    // Helper to convert Key to byte array
    fn key_to_bytes(&self, key: Key) -> Vec<u8> {
        key.to_be_bytes().to_vec()
    }
    
    // Helper to convert Value to byte array
    fn value_to_bytes(&self, value: Value) -> Vec<u8> {
        value.to_be_bytes().to_vec()
    }
    
    // Helper to convert bytes back to Key
    fn bytes_to_key(&self, bytes: &[u8]) -> Result<Key, String> {
        if bytes.len() != 8 {
            return Err(format!("Invalid key length: {}", bytes.len()));
        }
        
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(u64::from_be_bytes(array))
    }
    
    // Helper to convert bytes back to Value
    fn bytes_to_value(&self, bytes: &[u8]) -> Result<Value, String> {
        if bytes.len() != 8 {
            return Err(format!("Invalid value length: {}", bytes.len()));
        }
        
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(u64::from_be_bytes(array))
    }
}

#[cfg(feature = "use_terarkdb")]
impl BenchmarkableDatabase for TerarkDb {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn put(&mut self, key: Key, value: Value) -> Result<(), String> {
        let key_bytes = self.key_to_bytes(key);
        let value_bytes = self.value_to_bytes(value);
        
        unsafe {
            let mut error: *mut c_char = ptr::null_mut();
            
            terarkdb::rocksdb_put(
                self.db,
                self.write_options,
                key_bytes.as_ptr() as *const c_char,
                key_bytes.len(),
                value_bytes.as_ptr() as *const c_char,
                value_bytes.len(),
                &mut error,
            );
            
            if !error.is_null() {
                let error_message = CStr::from_ptr(error).to_string_lossy().into_owned();
                terarkdb::rocksdb_free(error as *mut c_void);
                return Err(format!("TerarkDB put error: {}", error_message));
            }
        }
        
        Ok(())
    }
    
    fn get(&self, key: Key) -> Result<Option<Value>, String> {
        let key_bytes = self.key_to_bytes(key);
        
        unsafe {
            let mut value_len: usize = 0;
            let mut error: *mut c_char = ptr::null_mut();
            
            let value_ptr = terarkdb::rocksdb_get(
                self.db,
                self.read_options,
                key_bytes.as_ptr() as *const c_char,
                key_bytes.len(),
                &mut value_len,
                &mut error,
            );
            
            if !error.is_null() {
                let error_message = CStr::from_ptr(error).to_string_lossy().into_owned();
                terarkdb::rocksdb_free(error as *mut c_void);
                return Err(format!("TerarkDB get error: {}", error_message));
            }
            
            if value_ptr.is_null() {
                return Ok(None);
            }
            
            let value_slice = slice::from_raw_parts(value_ptr as *const u8, value_len);
            let result = self.bytes_to_value(value_slice);
            
            terarkdb::rocksdb_free(value_ptr as *mut c_void);
            
            result.map(Some)
        }
    }
    
    fn delete(&mut self, key: Key) -> Result<(), String> {
        let key_bytes = self.key_to_bytes(key);
        
        unsafe {
            let mut error: *mut c_char = ptr::null_mut();
            
            terarkdb::rocksdb_delete(
                self.db,
                self.write_options,
                key_bytes.as_ptr() as *const c_char,
                key_bytes.len(),
                &mut error,
            );
            
            if !error.is_null() {
                let error_message = CStr::from_ptr(error).to_string_lossy().into_owned();
                terarkdb::rocksdb_free(error as *mut c_void);
                return Err(format!("TerarkDB delete error: {}", error_message));
            }
        }
        
        Ok(())
    }
    
    fn range(&self, start: Key, end: Key) -> Result<Vec<(Key, Value)>, String> {
        let start_bytes = self.key_to_bytes(start);
        let mut results = Vec::new();
        
        unsafe {
            let iterator = terarkdb::rocksdb_create_iterator(self.db, self.read_options);
            
            // Seek to the start key
            terarkdb::rocksdb_iter_seek(
                iterator,
                start_bytes.as_ptr() as *const c_char,
                start_bytes.len(),
            );
            
            // Iterate through keys in the range
            while terarkdb::rocksdb_iter_valid(iterator) != 0 {
                let mut key_len: usize = 0;
                let mut value_len: usize = 0;
                
                let key_ptr = terarkdb::rocksdb_iter_key(iterator, &mut key_len);
                if key_ptr.is_null() {
                    break;
                }
                
                let key_slice = slice::from_raw_parts(key_ptr as *const u8, key_len);
                let key = match self.bytes_to_key(key_slice) {
                    Ok(k) => k,
                    Err(e) => {
                        terarkdb::rocksdb_iter_destroy(iterator);
                        return Err(e);
                    }
                };
                
                // Stop if beyond end key
                if key >= end {
                    break;
                }
                
                let value_ptr = terarkdb::rocksdb_iter_value(iterator, &mut value_len);
                if value_ptr.is_null() {
                    terarkdb::rocksdb_iter_destroy(iterator);
                    return Err("TerarkDB iterator value is null".to_string());
                }
                
                let value_slice = slice::from_raw_parts(value_ptr as *const u8, value_len);
                let value = match self.bytes_to_value(value_slice) {
                    Ok(v) => v,
                    Err(e) => {
                        terarkdb::rocksdb_iter_destroy(iterator);
                        return Err(e);
                    }
                };
                
                results.push((key, value));
                
                // Move to the next key
                terarkdb::rocksdb_iter_next(iterator);
            }
            
            terarkdb::rocksdb_iter_destroy(iterator);
        }
        
        Ok(results)
    }
    
    fn flush(&mut self) -> Result<(), String> {
        // TerarkDB/RocksDB doesn't have a direct flush API in the C interface
        // In production, you might use a WAL (Write Ahead Log) or other techniques
        // For this benchmark, we'll just return success
        Ok(())
    }
    
    fn close(self) -> Result<(), String> {
        unsafe {
            terarkdb::rocksdb_close(self.db);
            terarkdb::rocksdb_options_destroy(self.options);
            terarkdb::rocksdb_readoptions_destroy(self.read_options);
            terarkdb::rocksdb_writeoptions_destroy(self.write_options);
        }
        
        Ok(())
    }
}

#[cfg(feature = "use_terarkdb")]
impl Drop for TerarkDb {
    fn drop(&mut self) {
        // Safety: Only close the DB if it hasn't been closed yet
        if !self.db.is_null() {
            unsafe {
                terarkdb::rocksdb_close(self.db);
                self.db = ptr::null_mut();
            }
        }
        
        // Cleanup options if they haven't been destroyed yet
        if !self.options.is_null() {
            unsafe {
                terarkdb::rocksdb_options_destroy(self.options);
                self.options = ptr::null_mut();
            }
        }
        
        if !self.read_options.is_null() {
            unsafe {
                terarkdb::rocksdb_readoptions_destroy(self.read_options);
                self.read_options = ptr::null_mut();
            }
        }
        
        if !self.write_options.is_null() {
            unsafe {
                terarkdb::rocksdb_writeoptions_destroy(self.write_options);
                self.write_options = ptr::null_mut();
            }
        }
    }
}

// Shared benchmark code
fn run_benchmark<D: BenchmarkableDatabase>(db_generator: impl Fn() -> Result<D, String>) -> Result<(), String> {
    // Parse any command line arguments for workload size, etc.
    let args: Vec<String> = std::env::args().collect();
    let workload_size = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(100000)
    } else {
        100000
    };
    
    println!("Running benchmark with workload size: {}", workload_size);
    println!("Database: {}", db_generator()?.name());
    
    // Create a temporary directory for the database
    let temp_dir = format!("/tmp/terarkdb_benchmark_{}", std::process::id());
    let _ = fs::remove_dir_all(&temp_dir); // Clean up any previous run
    fs::create_dir_all(&temp_dir).map_err(|e| format!("Failed to create temp directory: {}", e))?;
    
    // Generate results directory
    let results_dir = PathBuf::from("./sota/benchmark_results");
    fs::create_dir_all(&results_dir).map_err(|e| format!("Failed to create results directory: {}", e))?;
    
    // Function to run a single operation benchmark
    let run_operation = |db: &mut D, operation: &str, data: &[(Key, Value)]| -> Result<f64, String> {
        let start_time = Instant::now();
        let mut operations_completed = 0;
        
        match operation {
            "put" => {
                for &(key, value) in data {
                    db.put(key, value)?;
                    operations_completed += 1;
                }
                // Flush after put operations
                db.flush()?;
            },
            "get" => {
                for &(key, _) in data {
                    db.get(key)?;
                    operations_completed += 1;
                }
            },
            "range" => {
                // Find min/max key to determine range queries
                if !data.is_empty() {
                    let min_key = data.iter().map(|(k, _)| *k).min().unwrap_or(0);
                    let max_key = data.iter().map(|(k, _)| *k).max().unwrap_or(0);
                    
                    // Run 100 range queries of different sizes
                    let mut range_sizes = vec![10, 100, 1000];
                    
                    for &range_size in &range_sizes {
                        if range_size > (max_key - min_key) as usize {
                            continue;
                        }
                        
                        for _ in 0..100 {
                            let range_start = min_key + fastrand::u64(..(max_key - min_key - range_size as u64));
                            let range_end = range_start + range_size as u64;
                            
                            db.range(range_start, range_end)?;
                            operations_completed += 1;
                        }
                    }
                }
            },
            _ => return Err(format!("Unknown operation: {}", operation)),
        }
        
        let elapsed = start_time.elapsed();
        let throughput = operations_completed as f64 / elapsed.as_secs_f64();
        
        Ok(throughput)
    };
    
    // Benchmark functions
    let benchmark_operation = |operation: &str, data: &[(Key, Value)]| -> Result<(), String> {
        let mut db = db_generator()?;
        let db_name = db.name().to_string();
        
        println!("Running {} benchmark...", operation);
        let throughput = run_operation(&mut db, operation, data)?;
        
        // Save result to CSV
        let results_file = results_dir.join("benchmark_results.csv");
        let file_exists = results_file.exists();
        
        let mut results = csv::WriterBuilder::new()
            .has_headers(!file_exists)
            .append(file_exists)
            .from_path(&results_file)
            .map_err(|e| format!("Failed to open results file: {}", e))?;
        
        // Write header if the file is new
        if !file_exists {
            results.write_record(&["db_name", "operation", "workload_size", "throughput_ops_per_sec"])
                .map_err(|e| format!("Failed to write header: {}", e))?;
        }
        
        // Write the result row
        results.write_record(&[
            &db_name,
            operation,
            &workload_size.to_string(),
            &throughput.to_string(),
        ]).map_err(|e| format!("Failed to write result: {}", e))?;
        
        results.flush().map_err(|e| format!("Failed to flush results: {}", e))?;
        
        println!("{} throughput: {:.2} ops/sec", operation, throughput);
        
        // Close the database
        db.close()?;
        
        Ok(())
    };
    
    // Generate or parse workload data
    println!("Generating workload data...");
    let workload_file = PathBuf::from("./generator/workload.txt");
    
    let data: Vec<(Key, Value)> = if workload_file.exists() {
        println!("Using existing workload file");
        
        // Parse the workload file
        let file = std::fs::File::open(workload_file)
            .map_err(|e| format!("Failed to open workload file: {}", e))?;
        let reader = io::BufReader::new(file);
        
        let mut data = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            
            if parts.len() >= 2 {
                if let (Ok(key), Ok(value)) = (parts[0].parse::<Key>(), parts[1].parse::<Value>()) {
                    data.push((key, value));
                    if data.len() >= workload_size {
                        break;
                    }
                }
            }
        }
        
        data
    } else {
        println!("Generating random workload");
        
        // Generate random key-value pairs
        let mut data = Vec::with_capacity(workload_size);
        for _ in 0..workload_size {
            let key = fastrand::u64(..);
            let value = fastrand::u64(..);
            data.push((key, value));
        }
        
        data
    };
    
    println!("Generated {} key-value pairs", data.len());
    
    // Run the benchmarks
    benchmark_operation("put", &data)?;
    benchmark_operation("get", &data)?;
    benchmark_operation("range", &data)?;
    
    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);
    println!("Benchmark completed successfully!");
    
    Ok(())
}

#[cfg(feature = "use_terarkdb")]
fn main() -> Result<(), String> {
    println!("Running TerarkDB benchmark");
    run_benchmark(|| {
        let path = format!("/tmp/terarkdb_benchmark_{}", std::process::id());
        TerarkDb::new(&path)
    })
}

#[cfg(not(feature = "use_terarkdb"))]
fn main() -> Result<(), String> {
    println!("TerarkDB feature not enabled. Please rebuild with --features use_terarkdb");
    Ok(())
}