# LSM-tree Storage Documentation

This directory contains documentation for the LSM-tree storage implementations.

## Available Storage Backends

The LSM-tree supports multiple storage backends:

1. **FileStorage**: A simple file-based approach that stores each run in a separate file
2. **LSFStorage**: A log-structured file approach that combines multiple runs into log segments

## Running Tests

To run the storage tests:

```bash
# Run file storage tests
cargo test --test run_storage_test

# Run LSF storage tests
cargo test --test lsf_storage_test

# Run storage comparison tests
cargo test --test storage_comparison_test -- --nocapture
```

## Storage Performance Comparison

See [storage_comparison.md](storage_comparison.md) for detailed performance metrics comparing the different storage implementations.

## Next Steps

Future storage implementations planned:

1. **Memory-Mapped Storage**: Faster access to disk files without explicit I/O
2. **Tiered Storage**: Combined in-memory and disk storage with automatic promotion/demotion