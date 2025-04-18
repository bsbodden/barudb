#!/bin/bash
set -e

echo "==== Installing LevelDB Dependencies ===="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu/Pop!_OS
        echo "Detected Debian/Ubuntu/Pop!_OS"
        sudo apt-get update
        sudo apt-get install -y libleveldb-dev leveldb-util libsnappy-dev build-essential
    elif [ -f /etc/fedora-release ]; then
        # Fedora
        echo "Detected Fedora"
        sudo dnf install -y leveldb-devel snappy-devel gcc-c++ make
    elif [ -f /etc/arch-release ]; then
        # Arch Linux
        echo "Detected Arch Linux"
        sudo pacman -Sy --noconfirm leveldb snappy gcc make
    else
        echo "Unknown Linux distribution. Please install LevelDB dependencies manually."
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS"
    brew install leveldb snappy
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "==== LevelDB dependencies installed successfully ===="

# Add leveldb dependency to Cargo.toml
echo "==== Adding LevelDB dependency to Cargo.toml ===="

if ! grep -q "leveldb" ../Cargo.toml; then
    # Add leveldb dependency
    sed -i.bak '/\[dependencies\]/a leveldb = { version = "0.8.6", optional = true }' ../Cargo.toml
    # Add feature
    if ! grep -q "\[features\]" ../Cargo.toml; then
        echo -e "\n[features]\nuse_rocksdb = []\nuse_leveldb = []" >> ../Cargo.toml
    elif ! grep -q "use_leveldb" ../Cargo.toml; then
        sed -i.bak '/\[features\]/a use_leveldb = []' ../Cargo.toml
    fi
    echo "Added LevelDB dependency and feature flag to Cargo.toml"
else
    echo "LevelDB dependency already exists in Cargo.toml"
fi

echo "==== Setup complete! ===="
echo "You can now run the benchmark with: cargo bench --bench leveldb_comparison --features use_leveldb"