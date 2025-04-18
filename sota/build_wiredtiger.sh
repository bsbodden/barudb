#!/bin/bash
set -e

echo "Installing WiredTiger dependencies and building WiredTiger..."

# Install dependencies
apt-get update
apt-get install -y libsnappy-dev zlib1g-dev libbz2-dev python3-dev cmake git swig autoconf automake libtool

# Set up build directory
BUILD_DIR="$HOME/wiredtiger_build"
# Clean up previous failed build if it exists
if [ -d "$BUILD_DIR" ]; then
    echo "Removing previous build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone the repository
echo "Cloning WiredTiger repository..."
git clone https://github.com/wiredtiger/wiredtiger.git
cd wiredtiger

# Build WiredTiger
echo "Building WiredTiger..."
mkdir -p build
cd build
cmake -DHAVE_DIAGNOSTIC=0 -DHAVE_ATTACH=0 -DHAVE_BUILTIN_EXTENSION_SNAPPY=1 -DHAVE_PYTHON=0 ..
make -j$(nproc)

# Install WiredTiger
echo "Installing WiredTiger..."
make install

# Update shared library cache
echo "Updating shared library cache..."
ldconfig

echo "WiredTiger has been successfully built and installed!"
echo "Now you can run the benchmark with: ./sota/benchmark_wiredtiger.sh"