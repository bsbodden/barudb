#!/bin/bash
set -e

# Change to the project root directory
cd "$(dirname "$0")/.."
PROJ_ROOT=$(pwd)

# Skip installation since we've already installed WiredTiger
echo "Skipping WiredTiger installation (already installed)"

# Set the database name for the benchmark results
export DB_NAME="WiredTiger"

# Create the results directory if it doesn't exist
mkdir -p ./sota/benchmark_results

# Clean any previous build artifacts
cargo clean

# Run the LSM tree benchmark first
echo "Running LSM Tree benchmark for comparison..."
cargo build --release --bench wiredtiger_comparison
./target/release/deps/wiredtiger_comparison-* 100000 || true  # Continue even if this fails

# Create a simple C program to benchmark WiredTiger directly
echo "Creating WiredTiger C benchmark..."
cat > ./sota/wiredtiger_bench.c << 'EOF'
#include <wiredtiger.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

// Define operations
#define OP_PUT 1
#define OP_GET 2
#define OP_RANGE 3

typedef struct {
    uint64_t key;
    uint64_t value;
} KeyValuePair;

double run_benchmark(const char* db_path, int operation, KeyValuePair* data, size_t data_size) {
    WT_CONNECTION *conn = NULL;
    WT_SESSION *session = NULL;
    WT_CURSOR *cursor = NULL;
    int ret;
    clock_t start, end;
    double time_spent;
    size_t operations_completed = 0;

    // Create directory if it doesn't exist
    mkdir(db_path, 0755);
    
    // Open a connection to WiredTiger
    ret = wiredtiger_open(db_path, NULL, "create,cache_size=100MB", &conn);
    if (ret != 0) {
        fprintf(stderr, "Error opening WiredTiger connection: %s\n", wiredtiger_strerror(ret));
        return -1.0;
    }
    
    // Open a session
    ret = conn->open_session(conn, NULL, NULL, &session);
    if (ret != 0) {
        fprintf(stderr, "Error opening WiredTiger session: %s\n", wiredtiger_strerror(ret));
        conn->close(conn, NULL);
        return -1.0;
    }
    
    // Create a table
    ret = session->create(session, "table:benchmark", 
                        "key_format=Q,value_format=Q");
    if (ret != 0) {
        fprintf(stderr, "Error creating table: %s\n", wiredtiger_strerror(ret));
        session->close(session, NULL);
        conn->close(conn, NULL);
        return -1.0;
    }
    
    // Start timing
    start = clock();
    
    if (operation == OP_PUT) {
        // Open a cursor for writing
        ret = session->open_cursor(session, "table:benchmark", NULL, NULL, &cursor);
        if (ret != 0) {
            fprintf(stderr, "Error opening cursor: %s\n", wiredtiger_strerror(ret));
            session->close(session, NULL);
            conn->close(conn, NULL);
            return -1.0;
        }
        
        // Insert records
        for (size_t i = 0; i < data_size; i++) {
            cursor->set_key(cursor, data[i].key);
            cursor->set_value(cursor, data[i].value);
            ret = cursor->insert(cursor);
            if (ret != 0) {
                fprintf(stderr, "Error inserting record: %s\n", wiredtiger_strerror(ret));
                cursor->close(cursor);
                session->close(session, NULL);
                conn->close(conn, NULL);
                return -1.0;
            }
            operations_completed++;
        }
        
        // Close cursor
        cursor->close(cursor);
        
        // Checkpoint
        ret = session->checkpoint(session, NULL);
        if (ret != 0) {
            fprintf(stderr, "Error creating checkpoint: %s\n", wiredtiger_strerror(ret));
            session->close(session, NULL);
            conn->close(conn, NULL);
            return -1.0;
        }
    } 
    else if (operation == OP_GET) {
        // Open a cursor for reading
        ret = session->open_cursor(session, "table:benchmark", NULL, NULL, &cursor);
        if (ret != 0) {
            fprintf(stderr, "Error opening cursor: %s\n", wiredtiger_strerror(ret));
            session->close(session, NULL);
            conn->close(conn, NULL);
            return -1.0;
        }
        
        // Get records
        for (size_t i = 0; i < data_size; i++) {
            cursor->set_key(cursor, data[i].key);
            ret = cursor->search(cursor);
            if (ret != 0 && ret != WT_NOTFOUND) {
                fprintf(stderr, "Error searching record: %s\n", wiredtiger_strerror(ret));
                cursor->close(cursor);
                session->close(session, NULL);
                conn->close(conn, NULL);
                return -1.0;
            }
            operations_completed++;
        }
        
        // Close cursor
        cursor->close(cursor);
    } 
    else if (operation == OP_RANGE) {
        // Find min/max key
        uint64_t min_key = UINT64_MAX;
        uint64_t max_key = 0;
        
        for (size_t i = 0; i < data_size; i++) {
            if (data[i].key < min_key) min_key = data[i].key;
            if (data[i].key > max_key) max_key = data[i].key;
        }
        
        // Range sizes to test
        size_t range_sizes[] = {10, 100, 1000};
        
        for (size_t r = 0; r < 3; r++) {
            size_t range_size = range_sizes[r];
            
            // Skip if range size exceeds data range
            if (range_size > (max_key - min_key)) continue;
            
            for (int i = 0; i < 10; i++) {  // Just do 10 ranges
                // Open a cursor for range scan
                ret = session->open_cursor(session, "table:benchmark", NULL, NULL, &cursor);
                if (ret != 0) {
                    fprintf(stderr, "Error opening cursor: %s\n", wiredtiger_strerror(ret));
                    session->close(session, NULL);
                    conn->close(conn, NULL);
                    return -1.0;
                }
                
                // Generate random range start
                uint64_t range_start = min_key + (rand() % (max_key - min_key - range_size));
                uint64_t range_end = range_start + range_size;
                
                // Position cursor at start
                cursor->set_key(cursor, range_start);
                ret = cursor->search(cursor);
                if (ret != 0 && ret != WT_NOTFOUND) {
                    fprintf(stderr, "Error positioning cursor: %s\n", wiredtiger_strerror(ret));
                    cursor->close(cursor);
                    session->close(session, NULL);
                    conn->close(conn, NULL);
                    return -1.0;
                }
                
                // If not found, start from beginning
                if (ret == WT_NOTFOUND) {
                    ret = cursor->reset(cursor);
                    if (ret != 0) {
                        fprintf(stderr, "Error resetting cursor: %s\n", wiredtiger_strerror(ret));
                        cursor->close(cursor);
                        session->close(session, NULL);
                        conn->close(conn, NULL);
                        return -1.0;
                    }
                }
                
                // Iterate through range
                while ((ret = cursor->next(cursor)) == 0) {
                    uint64_t key;
                    cursor->get_key(cursor, &key);
                    
                    if (key >= range_end) break;
                    
                    uint64_t value;
                    cursor->get_value(cursor, &value);
                }
                
                if (ret != 0 && ret != WT_NOTFOUND) {
                    fprintf(stderr, "Error iterating through range: %s\n", wiredtiger_strerror(ret));
                    cursor->close(cursor);
                    session->close(session, NULL);
                    conn->close(conn, NULL);
                    return -1.0;
                }
                
                // Close cursor
                cursor->close(cursor);
                operations_completed++;
            }
        }
    }
    
    // End timing
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Calculate throughput
    double throughput = operations_completed / time_spent;
    
    // Close session and connection
    session->close(session, NULL);
    conn->close(conn, NULL);
    
    return throughput;
}

int main(int argc, char *argv[]) {
    size_t workload_size = (argc > 1) ? atoi(argv[1]) : 10000;
    const char* db_path = "/tmp/wiredtiger_benchmark";
    FILE* output_file;
    
    printf("Running WiredTiger benchmark with workload size: %zu\n", workload_size);
    
    // Remove existing database
    char rm_cmd[256];
    snprintf(rm_cmd, sizeof(rm_cmd), "rm -rf %s", db_path);
    system(rm_cmd);
    
    // Generate random data
    srand(time(NULL));
    KeyValuePair* data = (KeyValuePair*)malloc(workload_size * sizeof(KeyValuePair));
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for data\n");
        return 1;
    }
    
    printf("Generating %zu key-value pairs...\n", workload_size);
    for (size_t i = 0; i < workload_size; i++) {
        data[i].key = (uint64_t)rand() << 32 | rand();
        data[i].value = (uint64_t)rand() << 32 | rand();
    }
    
    // Create results directory
    mkdir("./sota", 0755);
    mkdir("./sota/benchmark_results", 0755);
    
    // Open output file
    output_file = fopen("./sota/benchmark_results/wiredtiger_comparison_results.csv", "w");
    if (!output_file) {
        fprintf(stderr, "Failed to open output file\n");
        free(data);
        return 1;
    }
    
    // Write header
    fprintf(output_file, "db_name,operation,workload_size,throughput_ops_per_sec\n");
    
    // Run PUT benchmark
    printf("Running PUT benchmark...\n");
    double put_throughput = run_benchmark(db_path, OP_PUT, data, workload_size);
    if (put_throughput < 0) {
        fprintf(stderr, "PUT benchmark failed\n");
        fclose(output_file);
        free(data);
        return 1;
    }
    printf("PUT throughput: %.2f ops/sec\n", put_throughput);
    fprintf(output_file, "WiredTiger,put,%zu,%.2f\n", workload_size, put_throughput);
    
    // Run GET benchmark
    printf("Running GET benchmark...\n");
    double get_throughput = run_benchmark(db_path, OP_GET, data, workload_size);
    if (get_throughput < 0) {
        fprintf(stderr, "GET benchmark failed\n");
        fclose(output_file);
        free(data);
        return 1;
    }
    printf("GET throughput: %.2f ops/sec\n", get_throughput);
    fprintf(output_file, "WiredTiger,get,%zu,%.2f\n", workload_size, get_throughput);
    
    // Run RANGE benchmark
    printf("Running RANGE benchmark...\n");
    double range_throughput = run_benchmark(db_path, OP_RANGE, data, workload_size);
    if (range_throughput < 0) {
        fprintf(stderr, "RANGE benchmark failed\n");
        fclose(output_file);
        free(data);
        return 1;
    }
    printf("RANGE throughput: %.2f ops/sec\n", range_throughput);
    fprintf(output_file, "WiredTiger,range,%zu,%.2f\n", workload_size, range_throughput);
    
    // Close output file
    fclose(output_file);
    
    // Clean up
    free(data);
    
    printf("Benchmark completed successfully!\n");
    return 0;
}
EOF

# Compile the C benchmark
echo "Compiling WiredTiger C benchmark..."
gcc -o ./sota/wiredtiger_bench ./sota/wiredtiger_bench.c -lwiredtiger -std=c99

# Run the compiled benchmark
echo "Running WiredTiger C benchmark..."
./sota/wiredtiger_bench 100000

# Run the analysis
cd sota
echo "Running benchmark analysis..."
./run_analysis.sh

echo "WiredTiger benchmarking completed successfully!"