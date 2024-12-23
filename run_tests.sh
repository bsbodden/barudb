#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Step 1: Generate workload
echo -e "${GREEN}Generating workload...${NC}"
cd generator
gcc $(gsl-config --cflags) generator.c $(gsl-config --libs) -o generator
./generator --puts 500 --gets 300 --ranges 200 --deletes 20 --gets-misses-ratio 0.3 --gets-skewness 0.2 --max-range-size 50 --seed 42 > workload.txt
cd ..

# Step 2: Start the server
echo -e "${GREEN}Starting server...${NC}"
cargo build --release
./target/release/server &
SERVER_PID=$!

# Step 3: Run integration tests
echo -e "${GREEN}Running integration tests...${NC}"
sleep 2 # Allow server to initialize
cargo test -- --nocapture || {
    echo -e "${RED}Tests failed. Stopping server...${NC}"
    kill $SERVER_PID
    exit 1
}

# Step 4: Evaluate results
echo -e "${GREEN}Evaluating server output...${NC}"
cd generator
poetry install --no-root
poetry run python evaluate.py workload.txt
cd ..

echo -e "${GREEN}All steps completed successfully!${NC}"
