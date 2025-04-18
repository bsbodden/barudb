#!/bin/bash
set -e

# LMDB installation script
echo "Checking LMDB dependencies..."

# Function to check if LMDB is installed
check_lmdb_installed() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Check if liblmdb-dev is installed on Linux
        if ldconfig -p | grep -q liblmdb; then
            return 0  # LMDB is installed
        else
            return 1  # LMDB is not installed
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Check if LMDB is installed on macOS
        if brew list | grep -q lmdb; then
            return 0  # LMDB is installed
        else
            return 1  # LMDB is not installed
        fi
    fi
    return 1  # Default to not installed
}

# Check if LMDB is already installed
if check_lmdb_installed; then
    echo "LMDB is already installed, skipping installation"
else
    echo "Installing LMDB dependencies..."

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        DISTRO=$(cat /etc/os-release | grep -oP '(?<=^ID=).+' | tr -d '"')
        echo "Detected Linux distribution: $DISTRO"
        
        if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" || "$DISTRO" == "pop" ]]; then
            echo "Installing LMDB on Debian/Ubuntu/Pop!_OS..."
            echo "NOTE: This requires sudo access. Please enter your password when prompted."
            sudo apt-get update
            sudo apt-get install -y liblmdb-dev
        elif [[ "$DISTRO" == "fedora" ]]; then
            echo "Installing LMDB on Fedora..."
            echo "NOTE: This requires sudo access. Please enter your password when prompted."
            sudo dnf install -y lmdb-devel
        elif [[ "$DISTRO" == "arch" ]]; then
            echo "Installing LMDB on Arch Linux..."
            echo "NOTE: This requires sudo access. Please enter your password when prompted."
            sudo pacman -S --noconfirm lmdb
        else
            echo "Unsupported Linux distribution. Please install LMDB manually."
            echo "On Debian/Ubuntu: sudo apt-get install -y liblmdb-dev"
            echo "On Fedora: sudo dnf install -y lmdb-devel"
            echo "On Arch: sudo pacman -S --noconfirm lmdb"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Installing LMDB on macOS..."
        brew install lmdb
    else
        echo "Unsupported operating system. Please install LMDB manually."
        exit 1
    fi
fi

# Add lmdb-rkv as a dependency to Cargo.toml if not already present
if ! grep -q "lmdb-rkv" Cargo.toml; then
    echo "Adding lmdb-rkv to Cargo.toml..."
    # Use sed to add the dependency before the [dev-dependencies] section
    sed -i -e '/\[dependencies\]/a lmdb-rkv = { version = "0.14.0", optional = true }' Cargo.toml
    
    # Add the feature flag if not already present
    if ! grep -q "use_lmdb" Cargo.toml; then
        echo "Adding use_lmdb feature flag..."
        if grep -q "\[features\]" Cargo.toml; then
            # If [features] section exists, add to it
            sed -i -e '/\[features\]/a use_lmdb = ["lmdb-rkv"]' Cargo.toml
        else
            # Otherwise, create the section
            echo -e "\n[features]\nuse_lmdb = [\"lmdb-rkv\"]" >> Cargo.toml
        fi
    fi
fi

echo "LMDB dependencies installed successfully!"