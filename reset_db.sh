#!/bin/bash

# Reset BarúDB Database
# This script removes all database files to start with a fresh state

echo "Removing existing database files..."
rm -rf ./data

echo "Creating empty directory structure..."
mkdir -p ./data/runs
for i in {0..9}; do
  mkdir -p "./data/runs/level_$i"
done

echo "Database reset complete. Ready for demo."
