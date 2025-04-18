#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/terark_namespace.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// Use TerarkDB namespace
using namespace terarkdb;

// Benchmark parameters
struct BenchmarkConfig {
    std::string db_path;
    std::string db_name;
    size_t workload_size;
    std::string results_path;
};

// Helper to format numbers with commas
std::string format_with_commas(double value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value;
    return ss.str();
}

// Benchmark implementation
class Benchmark {
private:
    BenchmarkConfig config;
    DB* db;
    Options options;
    std::vector<std::pair<uint64_t, uint64_t>> data;

    // Convert uint64_t to string for storage
    std::string uint64_to_string(uint64_t value) {
        char buffer[sizeof(uint64_t)];
        memcpy(buffer, &value, sizeof(uint64_t));
        return std::string(buffer, sizeof(uint64_t));
    }

    // Convert string back to uint64_t
    uint64_t string_to_uint64(const std::string& str) {
        if (str.size() != sizeof(uint64_t)) {
            throw std::runtime_error("Invalid string size for uint64_t conversion");
        }
        uint64_t value;
        memcpy(&value, str.data(), sizeof(uint64_t));
        return value;
    }

    // Generate random data for the benchmark
    void generate_data() {
        std::cout << "Generating " << config.workload_size << " key-value pairs..." << std::endl;
        
        // Try to read from workload.txt if it exists
        std::string workload_path = "./generator/workload.txt";
        std::ifstream workload_file(workload_path);
        
        if (workload_file.is_open()) {
            std::cout << "Using existing workload file: " << workload_path << std::endl;
            
            std::string line;
            while (std::getline(workload_file, line) && data.size() < config.workload_size) {
                std::istringstream iss(line);
                uint64_t key, value;
                if (iss >> key >> value) {
                    data.emplace_back(key, value);
                }
            }
            
            std::cout << "Read " << data.size() << " key-value pairs from workload file" << std::endl;
        }
        
        // If we need more data or no file was found, generate random data
        if (data.size() < config.workload_size) {
            std::random_device rd;
            std::mt19937_64 gen(rd());
            std::uniform_int_distribution<uint64_t> dist;
            
            size_t additional = config.workload_size - data.size();
            std::cout << "Generating " << additional << " additional random key-value pairs" << std::endl;
            
            for (size_t i = 0; i < additional; ++i) {
                uint64_t key = dist(gen);
                uint64_t value = dist(gen);
                data.emplace_back(key, value);
            }
        }
    }

    // Write results to CSV file
    void write_results(const std::string& operation, double throughput) {
        std::cout << operation << " throughput: " << format_with_commas(throughput) << " ops/sec" << std::endl;
        
        fs::create_directories(fs::path(config.results_path).parent_path());
        
        bool file_exists = fs::exists(config.results_path);
        std::ofstream results_file(config.results_path, std::ios::app);
        
        if (!file_exists) {
            results_file << "db_name,operation,workload_size,throughput_ops_per_sec\n";
        }
        
        results_file << config.db_name << ","
                     << operation << ","
                     << config.workload_size << ","
                     << throughput << "\n";
                     
        results_file.close();
    }

public:
    Benchmark(const BenchmarkConfig& cfg) : config(cfg), db(nullptr) {
        // Configure TerarkDB/RocksDB options
        options.create_if_missing = true;
        options.IncreaseParallelism();
        options.OptimizeLevelStyleCompaction();
        
        // TerarkDB-specific optimizations
        // These options are TerarkDB-specific and will be ignored by normal RocksDB
        options.allow_mmap_reads = true;
        options.allow_mmap_writes = true;
        options.compression = kLZ4Compression;
        options.bottommost_compression = kZSTD;
        
        // Create directory if needed
        fs::create_directories(config.db_path);
    }

    ~Benchmark() {
        close();
    }

    bool open() {
        Status status = DB::Open(options, config.db_path, &db);
        if (!status.ok()) {
            std::cerr << "Failed to open database: " << status.ToString() << std::endl;
            return false;
        }
        return true;
    }

    void close() {
        if (db != nullptr) {
            delete db;
            db = nullptr;
        }
    }

    bool run() {
        if (!open()) {
            return false;
        }

        generate_data();
        
        // Benchmark PUT operations
        benchmark_put();
        
        // Benchmark GET operations
        benchmark_get();
        
        // Benchmark RANGE operations
        benchmark_range();
        
        close();
        return true;
    }

    void benchmark_put() {
        std::cout << "Running PUT benchmark..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Write all key-value pairs
        WriteOptions write_options;
        size_t operations_completed = 0;
        
        for (const auto& [key, value] : data) {
            std::string key_str = uint64_to_string(key);
            std::string value_str = uint64_to_string(value);
            
            Status status = db->Put(write_options, key_str, value_str);
            if (!status.ok()) {
                std::cerr << "PUT error: " << status.ToString() << std::endl;
                continue;
            }
            
            operations_completed++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        double throughput = operations_completed / elapsed.count();
        write_results("put", throughput);
    }

    void benchmark_get() {
        std::cout << "Running GET benchmark..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Read all key-value pairs
        ReadOptions read_options;
        size_t operations_completed = 0;
        
        for (const auto& [key, _] : data) {
            std::string key_str = uint64_to_string(key);
            std::string value_str;
            
            Status status = db->Get(read_options, key_str, &value_str);
            if (status.ok() || status.IsNotFound()) {
                operations_completed++;
            } else {
                std::cerr << "GET error: " << status.ToString() << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        double throughput = operations_completed / elapsed.count();
        write_results("get", throughput);
    }

    void benchmark_range() {
        std::cout << "Running RANGE benchmark..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Find min/max keys
        uint64_t min_key = std::numeric_limits<uint64_t>::max();
        uint64_t max_key = 0;
        
        for (const auto& [key, _] : data) {
            min_key = std::min(min_key, key);
            max_key = std::max(max_key, key);
        }
        
        // Range query sizes
        std::vector<size_t> range_sizes = {10, 100, 1000};
        
        // Random generator for range start points
        std::random_device rd;
        std::mt19937_64 gen(rd());
        
        size_t operations_completed = 0;
        
        for (size_t range_size : range_sizes) {
            if (max_key <= min_key + range_size) {
                continue;  // Skip range sizes that are too large
            }
            
            std::uniform_int_distribution<uint64_t> dist(min_key, max_key - range_size);
            
            for (int i = 0; i < 100; ++i) {  // 100 range queries per size
                uint64_t range_start = dist(gen);
                uint64_t range_end = range_start + range_size;
                
                std::string start_key = uint64_to_string(range_start);
                std::string end_key = uint64_to_string(range_end);
                
                Iterator* it = db->NewIterator(ReadOptions());
                
                int count = 0;
                for (it->Seek(start_key); it->Valid() && it->key().ToString() < end_key; it->Next()) {
                    count++;  // Just count the entries
                }
                
                delete it;
                operations_completed++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        double throughput = operations_completed / elapsed.count();
        write_results("range", throughput);
    }
};

int main(int argc, char* argv[]) {
    // Parse workload size from command line
    size_t workload_size = 100000;  // Default
    if (argc > 1) {
        try {
            workload_size = std::stoul(argv[1]);
        } catch (...) {
            std::cerr << "Invalid workload size. Using default: " << workload_size << std::endl;
        }
    }
    
    // Get database name from environment variable or default
    const char* env_db_name = std::getenv("DB_NAME");
    std::string db_name = (env_db_name != nullptr) ? env_db_name : "TerarkDB";
    
    // Configure benchmark
    BenchmarkConfig config;
    config.db_path = "/tmp/terarkdb_benchmark_" + std::to_string(getpid());
    config.db_name = db_name;
    config.workload_size = workload_size;
    config.results_path = "./sota/benchmark_results/terarkdb_comparison_results.csv";
    
    std::cout << "Running benchmark with:" << std::endl;
    std::cout << "  Database: " << config.db_name << std::endl;
    std::cout << "  Workload size: " << config.workload_size << std::endl;
    std::cout << "  DB path: " << config.db_path << std::endl;
    
    // Run the benchmark
    Benchmark benchmark(config);
    bool success = benchmark.run();
    
    // Clean up
    fs::remove_all(config.db_path);
    
    return success ? 0 : 1;
}