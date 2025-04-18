#!/bin/bash
set -e

# Set up Python environment with Poetry
echo "=== Setting up Python environment with Poetry ==="

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies without the root project
echo "Installing Python dependencies..."
cd "$(dirname "$0")"  # Make sure we're in the sota directory
poetry install --no-root

# Run the analysis
echo "=== Running benchmark analysis ==="
poetry run python analyze_benchmarks.py

echo "=== Analysis complete! ==="
echo "Performance comparison visualizations are available in the visualizations directory."

# List generated files
if [ -d "visualizations" ]; then
    echo "Generated files:"
    ls -la visualizations/
fi