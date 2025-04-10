#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build the server
echo -e "${GREEN}Building server...${NC}"
cargo build --release

# Check if server is already running
if nc -z localhost 8080 2>/dev/null; then
    echo -e "${YELLOW}Server already running on port 8080, skipping server start${NC}"
    SERVER_STARTED=false
else
    # Start the server
    echo -e "${GREEN}Starting server...${NC}"
    ./target/release/server &
    SERVER_PID=$!
    SERVER_STARTED=true
    
    # Give server time to start
    echo -e "${GREEN}Waiting for server to initialize...${NC}"
    sleep 3
fi

# Run the server command tests
echo -e "${GREEN}Running server command tests...${NC}"
cargo test --test server_commands_test -- --nocapture

# If we started the server, stop it
if [ "$SERVER_STARTED" = true ]; then
    echo -e "${GREEN}Stopping server...${NC}"
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true  # Wait for server to exit
fi

echo -e "${GREEN}All server command tests completed!${NC}"