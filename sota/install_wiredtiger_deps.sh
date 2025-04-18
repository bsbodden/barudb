#!/bin/bash
set -e

# WiredTiger installation script
echo "Installing WiredTiger dependencies..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    DISTRO=$(cat /etc/os-release | grep -oP '(?<=^ID=).+' | tr -d '"')
    echo "Detected Linux distribution: $DISTRO"
    
    if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" || "$DISTRO" == "pop" ]]; then
        echo "Installing WiredTiger dependencies on Debian/Ubuntu/Pop!_OS..."
        sudo apt-get update
        sudo apt-get install -y libsnappy-dev zlib1g-dev libbz2-dev python3-dev cmake git swig
    elif [[ "$DISTRO" == "fedora" ]]; then
        echo "Installing WiredTiger dependencies on Fedora..."
        sudo dnf install -y snappy-devel zlib-devel bzip2-devel python3-devel cmake git
    elif [[ "$DISTRO" == "arch" ]]; then
        echo "Installing WiredTiger dependencies on Arch Linux..."
        sudo pacman -S --noconfirm snappy zlib bzip2 python cmake git
    else
        echo "Unsupported Linux distribution. Please install WiredTiger dependencies manually."
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Installing WiredTiger dependencies on macOS..."
    brew install snappy zlib bzip2 python cmake git
else
    echo "Unsupported operating system. Please install WiredTiger dependencies manually."
    exit 1
fi

# Add libwiredtiger as a dependency to Cargo.toml if not already present
if ! grep -q "wiredtiger-sys" Cargo.toml; then
    echo "Adding wiredtiger-sys to Cargo.toml..."
    # Use sed to add the dependency before the [dev-dependencies] section
    sed -i -e '/\[dependencies\]/a wiredtiger-sys = { version = "0.10.0-rc0", optional = true }' Cargo.toml
    
    # Add the feature flag if not already present
    if ! grep -q "use_wiredtiger" Cargo.toml; then
        echo "Adding use_wiredtiger feature flag..."
        if grep -q "\[features\]" Cargo.toml; then
            # If [features] section exists, add to it
            sed -i -e '/\[features\]/a use_wiredtiger = ["wiredtiger-sys"]' Cargo.toml
        else
            # Otherwise, create the section
            echo -e "\n[features]\nuse_wiredtiger = [\"wiredtiger-sys\"]" >> Cargo.toml
        fi
    fi
fi

# Clone and build WiredTiger
BUILD_DIR="$HOME/wiredtiger_build"

# Clean up previous failed build if it exists
if [ -d "$BUILD_DIR" ]; then
    echo "Removing previous build directory..."
    rm -rf "$BUILD_DIR"
fi

# Build WiredTiger
echo "Cloning and building WiredTiger..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone the repository
git clone https://github.com/wiredtiger/wiredtiger.git
cd wiredtiger

# Build WiredTiger
mkdir -p build
cd build
cmake -DHAVE_DIAGNOSTIC=0 -DHAVE_ATTACH=0 -DHAVE_BUILTIN_EXTENSION_SNAPPY=1 -DHAVE_PYTHON=0 ..
make -j$(nproc)

# Install WiredTiger
sudo make install

# Update ldconfig if on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo ldconfig
fi

echo "WiredTiger has been built and installed successfully."

echo "WiredTiger dependencies installed successfully!"