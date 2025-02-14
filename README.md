# LSM Tree Database (Rust Implementation)

This repository contains a Rust implementation of an LSM tree key-value store, following the specifications from
Harvard's CS265 Systems project. The implementation provides a client-server architecture for managing key-value data
using a Log-Structured Merge Tree.

## Prerequisites

### Rust Installation

You'll need Rust installed on your system. If you haven't installed Rust yet,
see https://www.rust-lang.org/tools/install for installation options

The project requires Rust 1.81.0 or later.

## Building the Project

Clone the repository and build using cargo:

```bash
git clone [repository-url]
cd lsm-tree
cargo build --release
```

## Running the Server

To launch the server:

```bash
cargo run --release --bin server [OPTIONS]
```

### Server Options

| Option              | Default   | Description                                                         |
|---------------------|-----------|---------------------------------------------------------------------|
| `-e <error_rate>`   | 0.01      | Bloom filter error rate                                             |
| `-n <num_pages>`    | 1024      | Size of the buffer by number of disk pages                          |
| `-f <fanout>`       | 2         | LSM tree fanout                                                     |
| `-l <level_policy>` | "leveled" | Compaction policy (options: tiered, leveled, lazy_leveled, partial) |
| `-p <port>`         | 8080      | Port number                                                         |
| `-h`                | N/A       | Print help message                                                  |

## Running the Client

To launch the client:

```bash
cargo run --release --bin client [OPTIONS]
```

### Client Options

| Option      | Description                 |
|-------------|-----------------------------|
| `-p <port>` | Port number (default: 8080) |
| `-q`        | Quiet mode                  |

## Supported Commands

The database supports the following commands:

### Client Commands

| Command           | Description          | Example        |
|-------------------|----------------------|----------------|
| `p <key> <value>` | Put a key-value pair | `p 10 42`      |
| `g <key>`         | Get value for key    | `g 10`         |
| `r <start> <end>` | Range query          | `r 10 20`      |
| `d <key>`         | Delete key           | `d 10`         |
| `l <filename>`    | Load from file       | `l "data.bin"` |
| `s`               | Print stats          | `s`            |
| `q`               | Quit                 | `q`            |

### Server Commands

While the server is running, you can enter these commands in the server terminal:

| Command | Description                |
|---------|----------------------------|
| `bloom` | Print Bloom Filter summary |
| `quit`  | Quit server                |
| `help`  | Print help message         |

## Project Structure

```
lsm-tree/
├── Cargo.toml
├── src/
│   ├── lib.rs        # Shared library code
│   ├── server.rs     # Server implementation
│   └── client.rs     # Client implementation
```

## Development Status

This is an initial implementation focusing on the basic client-server architecture and command processing. Future
updates will include:

- Full LSM tree implementation
- Bloom filter optimization
- Compaction strategies
- Performance optimizations
- Multi-threading support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.