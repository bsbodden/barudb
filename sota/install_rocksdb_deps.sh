#!/bin/bash

# Script to install RocksDB dependencies for both Linux and macOS
# Usage: ./install_rocksdb_deps.sh

set -e

# Function to detect the OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ -f /etc/os-release ]]; then
        source /etc/os-release
        if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* || "$ID" == "pop" || "$ID_LIKE" == *"debian"* ]]; then
            echo "debian"
        elif [[ "$ID" == "fedora" || "$ID_LIKE" == *"fedora"* ]]; then
            echo "fedora"
        elif [[ "$ID" == "arch" || "$ID_LIKE" == *"arch"* ]]; then
            echo "arch"
        else
            echo "unknown_linux"
        fi
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo "Detected OS: $OS"

case $OS in
    "macos")
        echo "Installing RocksDB dependencies for macOS..."
        echo "Using Homebrew to install required packages..."
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Please install Homebrew first:"
            echo "/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        
        # Install RocksDB dependencies
        brew install llvm gflags snappy zlib bzip2 lz4 zstd
        
        # Set up environment variables
        echo ""
        echo "Please add the following lines to your ~/.zshrc or ~/.bash_profile:"
        echo 'export LIBRARY_PATH="$LIBRARY_PATH:$(brew --prefix)/lib"'
        echo 'export CPATH="$CPATH:$(brew --prefix)/include"'
        echo 'export LIBCLANG_PATH="$(brew --prefix llvm)/lib"'
        echo ""
        echo "Then run: source ~/.zshrc (or ~/.bash_profile)"
        ;;
        
    "debian")
        echo "Installing RocksDB dependencies for Debian/Ubuntu/Pop!_OS..."
        sudo apt-get update
        
        # Install build dependencies and libraries
        sudo apt-get install -y \
            clang \
            libclang-dev \
            llvm-dev \
            libgflags-dev \
            libsnappy-dev \
            zlib1g-dev \
            libbz2-dev \
            liblz4-dev \
            libzstd-dev \
            build-essential \
            pkg-config
        ;;
        
    "fedora")
        echo "Installing RocksDB dependencies for Fedora..."
        sudo dnf install -y \
            clang \
            clang-devel \
            llvm-devel \
            gflags-devel \
            snappy-devel \
            zlib-devel \
            bzip2-devel \
            lz4-devel \
            libzstd-devel \
            make \
            gcc-c++
        ;;
        
    "arch")
        echo "Installing RocksDB dependencies for Arch Linux..."
        sudo pacman -Sy \
            clang \
            llvm \
            gflags \
            snappy \
            zlib \
            bzip2 \
            lz4 \
            zstd \
            base-devel
        ;;
        
    *)
        echo "Unsupported or unknown operating system."
        echo "You'll need to manually install the following dependencies:"
        echo "- clang/llvm (for libclang)"
        echo "- gflags"
        echo "- snappy"
        echo "- zlib"
        echo "- bzip2"
        echo "- lz4"
        echo "- zstd"
        echo "- C/C++ build tools"
        exit 1
        ;;
esac

echo ""
echo "Dependencies installed successfully."

# Update the Cargo.toml file
echo "Updating your Cargo.toml to enable RocksDB..."
sed -i.bak 's/# rocksdb = "0.21.0"/rocksdb = "0.21.0"/' /home/bsb/Code/hes/cs265-lsm-tree/Cargo.toml

echo ""
echo "You can now run the benchmark with:"
echo "cargo bench --bench rocksdb_comparison --features use_rocksdb"