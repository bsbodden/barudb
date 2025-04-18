#!/bin/bash
set -e

# Install required system dependencies
if ! dpkg -l | grep -q "libaio-dev"; then
    echo "Installing required system dependencies..."
    sudo apt-get update
    sudo apt-get install -y git build-essential cmake libsnappy-dev zlib1g-dev libbz2-dev \
        libzstd-dev liblz4-dev libgflags-dev pkg-config liburing-dev curl wget libaio-dev
else
    echo "Required system dependencies are already installed."
fi

# Set up TerarkDB build directory
TERARKDB_DIR="$HOME/Code/hes/terarkdb"
echo "Setting up TerarkDB in $TERARKDB_DIR..."

# Remove previous build directory if it exists
if [ -d "$TERARKDB_DIR" ]; then
    echo "Removing previous TerarkDB directory..."
    rm -rf "$TERARKDB_DIR"
fi

# Clone TerarkDB repository
echo "Cloning TerarkDB repository..."
git clone https://github.com/bytedance/terarkdb.git "$TERARKDB_DIR"

# Check if clone was successful
if [ ! -d "$TERARKDB_DIR" ]; then
    echo "ERROR: Failed to clone TerarkDB repository."
    exit 1
fi

# Enter the TerarkDB directory
cd "$TERARKDB_DIR"

# Initialize submodules
echo "Initializing git submodules..."
git submodule update --init --recursive

# Build TerarkDB
echo "Building TerarkDB..."
WITH_TESTS=OFF WITH_ZNS=OFF ./build.sh

# Check if the build was successful
if [ ! -f "$TERARKDB_DIR/build/libterarkdb.a" ]; then
    echo "ERROR: Failed to build TerarkDB. Library file not found."
    exit 1
fi

echo "TerarkDB built successfully. Library file: $TERARKDB_DIR/build/libterarkdb.a"
echo "You can now run the benchmark script: ./sota/benchmark_terarkdb.sh"