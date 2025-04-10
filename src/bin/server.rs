use lsm_tree::command::Command;
use lsm_tree::lsm_tree::LSMTree;
use lsm_tree::types::{CompactionPolicyType, StorageType};
use std::io;
use std::io::BufRead;
use std::io::{BufReader, Write};
use std::net::TcpListener;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

fn send_response(stream: &mut TcpStream, response: &str) -> io::Result<()> {
    // Send response followed by END_OF_MESSAGE marker
    stream.write_all(response.as_bytes())?;
    stream.write_all(lsm_tree::END_OF_MESSAGE.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn handle_client(
    mut stream: TcpStream,
    termination_flag: Arc<AtomicBool>,
    lsm_tree: Arc<RwLock<LSMTree>>,
) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    println!("Started handling client");

    while !termination_flag.load(Ordering::SeqCst) {
        let mut buffer = String::new();

        match reader.read_line(&mut buffer) {
            Ok(0) => {
                println!("Client disconnected");
                break;
            }
            Ok(_) => {
                println!("Received command: {}", buffer.trim());
                let response = match Command::parse(buffer.trim()) {
                    Some(Command::Put(key, value)) => {
                        println!("Processing Put({}, {})", key, value);
                        let mut tree = lsm_tree.write().unwrap();
                        match tree.put(key, value) {
                            Ok(_) => "OK".to_string(),
                            Err(e) => format!("Error: {:?}", e),
                        }
                    }
                    Some(Command::Get(key)) => {
                        println!("Processing Get({})", key);
                        let tree = lsm_tree.read().unwrap();
                        match tree.get(key) {
                            Some(value) => value.to_string(),
                            None => "".to_string(),
                        }
                    }
                    Some(Command::Range(start, end)) => {
                        println!("Processing Range({}, {})", start, end);
                        let tree = lsm_tree.read().unwrap();
                        tree.range(start, end)
                            .into_iter()
                            .map(|(k, v)| format!("{}:{}", k, v))
                            .collect::<Vec<_>>()
                            .join(" ")
                    }
                    Some(Command::Delete(key)) => {
                        println!("Processing Delete({})", key);
                        let mut tree = lsm_tree.write().unwrap();
                        match tree.delete(key) {
                            Ok(_) => "OK".to_string(),
                            Err(e) => format!("Error: {:?}", e),
                        }
                    }
                    Some(Command::Quit) => {
                        println!("Client requested quit, shutting down server...");
                        termination_flag.store(true, Ordering::SeqCst);
                        "Shutting down the server".to_string()
                    }
                    Some(Command::Load(filename)) => {
                        println!("Processing Load command from file: {}", filename);
                        
                        // Read the specified file
                        match std::fs::read_to_string(&filename) {
                            Ok(contents) => {
                                let mut tree = lsm_tree.write().unwrap();
                                let mut lines_processed = 0;
                                let mut lines_failed = 0;
                                
                                // Process each line in the file
                                for (line_num, line) in contents.lines().enumerate() {
                                    if let Some(cmd) = Command::parse(line.trim()) {
                                        match cmd {
                                            // Process put commands
                                            Command::Put(key, value) => {
                                                match tree.put(key, value) {
                                                    Ok(_) => lines_processed += 1,
                                                    Err(e) => {
                                                        eprintln!("Error on line {}: {:?}", line_num + 1, e);
                                                        lines_failed += 1;
                                                    }
                                                }
                                            },
                                            // Process delete commands
                                            Command::Delete(key) => {
                                                match tree.delete(key) {
                                                    Ok(_) => lines_processed += 1,
                                                    Err(e) => {
                                                        eprintln!("Error on line {}: {:?}", line_num + 1, e);
                                                        lines_failed += 1;
                                                    }
                                                }
                                            },
                                            // Ignore other commands in the file
                                            _ => {
                                                eprintln!("Unsupported command on line {}: {}", line_num + 1, line);
                                                lines_failed += 1;
                                            }
                                        }
                                    } else {
                                        eprintln!("Invalid command on line {}: {}", line_num + 1, line);
                                        lines_failed += 1;
                                    }
                                }
                                
                                // Flush to disk after processing the file
                                match tree.flush_buffer_to_level0() {
                                    Ok(_) => {},
                                    Err(e) => {
                                        eprintln!("Error flushing buffer after load: {:?}", e);
                                        lines_failed += 1;
                                    }
                                }
                                
                                format!("Loaded {} commands from file ({} failed)", lines_processed, lines_failed)
                            },
                            Err(e) => {
                                eprintln!("Error reading file '{}': {:?}", filename, e);
                                format!("Error: Could not read file '{}'", filename)
                            }
                        }
                    }
                    Some(Command::PrintStats) => {
                        println!("Processing PrintStats command");
                        let tree = lsm_tree.read().unwrap();
                        let config = tree.get_config();
                        
                        // Get storage statistics from the LSM tree
                        match tree.get_storage_stats() {
                            Ok(stats) => {
                                // Format statistics as a string
                                let mut result = String::new();
                                result.push_str("# LSM Tree Statistics\n");
                                result.push_str(&format!("Storage type: {}\n", config.storage_type.as_str()));
                                result.push_str(&format!("Compaction policy: {}\n", config.compaction_policy.as_str()));
                                result.push_str(&format!("Compaction threshold: {}\n", config.compaction_threshold));
                                result.push_str(&format!("Fanout: {}\n", config.fanout));
                                result.push_str(&format!("Buffer size: {} pages\n", config.buffer_size));
                                result.push_str("\n# Storage Statistics\n");
                                result.push_str(&format!("Total size: {} bytes\n", stats.total_size_bytes));
                                result.push_str(&format!("File count: {}\n", stats.file_count));
                                
                                // Print level statistics
                                result.push_str("\n# Level Statistics\n");
                                for (level, &run_count) in stats.runs_per_level.iter().enumerate() {
                                    if run_count > 0 {
                                        let blocks = stats.blocks_per_level.get(level).unwrap_or(&0);
                                        let entries = stats.entries_per_level.get(level).unwrap_or(&0);
                                        result.push_str(&format!("Level {}: {} runs, {} blocks, {} entries\n", 
                                                              level, run_count, blocks, entries));
                                    }
                                }
                                
                                result
                            },
                            Err(_) => "Error: Unable to retrieve storage statistics".to_string()
                        }
                    }
                    Some(Command::FlushMemtable) => {
                        println!("Processing FlushMemtable command");
                        let mut tree = lsm_tree.write().unwrap();
                        match tree.flush_buffer_to_level0() {
                            Ok(_) => "OK - Memtable flushed to disk".to_string(),
                            Err(e) => format!("Error flushing memtable: {:?}", e),
                        }
                    }
                    None => {
                        eprintln!("Invalid command received");
                        "Invalid command".to_string()
                    }
                };

                if let Err(e) = send_response(&mut stream, &response) {
                    eprintln!("Failed to send response: {}", e);
                    break;
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            Err(e) => {
                eprintln!("Error reading from client: {}", e);
                break;
            }
        }
    }

    println!("Stopped handling client");
}

fn print_help() {
    println!("LSM Tree Server");
    println!("Usage: server [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -p <port>           Port number to listen on (default: 8080)");
    println!("  -n <num_pages>      Size of buffer in pages (default: 128)");
    println!("  -f <fanout>         Fanout/size ratio between levels (default: 4)");
    println!("  -l <level_policy>   Compaction policy (tiered, leveled, lazy_leveled) (default: tiered)");
    println!("  -t <threshold>      Compaction threshold (default: 4)");
    println!("  -h                  Print this help message");
    println!();
    println!("Server Commands (enter in server console):");
    println!("  help                 Print this help message");
    println!("  quit                 Shut down the server");
    println!();
    println!("Client Commands:");
    println!("  p <key> <value>     Put a key-value pair");
    println!("  g <key>             Get value for key");
    println!("  r <start> <end>     Range query");
    println!("  d <key>             Delete key");
    println!("  l <filename>        Load commands from file");
    println!("  s                   Print statistics");
    println!("  f                   Flush memtable to disk");
    println!("  q                   Quit");
}

fn handle_server_command(
    input: &str, 
    termination_flag: &Arc<AtomicBool>
) -> Option<String> {
    match input.trim() {
        "help" => {
            print_help();
            Some("Help information printed to console".to_string())
        },
        "quit" => {
            println!("Server shutting down...");
            termination_flag.store(true, Ordering::SeqCst);
            Some("Server is shutting down".to_string())
        },
        _ => None,
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut port = match std::env::var("SERVER_PORT") {
        Ok(val) => val.parse().unwrap_or(8080),
        Err(_) => 8080
    };
    let mut buffer_size = 128;
    let mut fanout = 4;
    let mut compaction_policy = CompactionPolicyType::Tiered;
    let mut compaction_threshold = 4;

    // Parse command line arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-p" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or(port);
                    i += 2;
                } else {
                    i += 1;
                }
            },
            "-n" => {
                if i + 1 < args.len() {
                    buffer_size = args[i + 1].parse().unwrap_or(128);
                    i += 2;
                } else {
                    i += 1;
                }
            },
            "-f" => {
                if i + 1 < args.len() {
                    fanout = args[i + 1].parse().unwrap_or(4);
                    i += 2;
                } else {
                    i += 1;
                }
            },
            "-l" => {
                if i + 1 < args.len() {
                    compaction_policy = match args[i + 1].as_str() {
                        "tiered" => CompactionPolicyType::Tiered,
                        "leveled" => CompactionPolicyType::Leveled,
                        "lazy_leveled" => CompactionPolicyType::LazyLeveled,
                        _ => {
                            eprintln!("Unknown compaction policy '{}', defaulting to tiered", args[i + 1]);
                            CompactionPolicyType::Tiered
                        }
                    };
                    i += 2;
                } else {
                    i += 1;
                }
            },
            "-t" => {
                if i + 1 < args.len() {
                    compaction_threshold = args[i + 1].parse().unwrap_or(4);
                    i += 2;
                } else {
                    i += 1;
                }
            },
            "-h" => {
                print_help();
                return Ok(());
            },
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Check for DATA_DIR environment variable (for integration tests)
    let storage_path = match std::env::var("DATA_DIR") {
        Ok(path) => std::path::PathBuf::from(path),
        Err(_) => std::path::PathBuf::from("./data")
    };

    // Create LSM tree with the configured options
    let compaction_policy_clone = compaction_policy.clone();
    let config = lsm_tree::lsm_tree::LSMConfig {
        buffer_size,
        storage_type: StorageType::File,
        storage_path,
        create_path_if_missing: true,
        max_open_files: 1000,
        sync_writes: true,
        fanout,
        compaction_policy,
        compaction_threshold,
    };
    
    println!("Starting server with configuration:");
    println!("  Buffer size: {} pages", buffer_size);
    println!("  Fanout: {}", fanout);
    println!("  Compaction policy: {}", compaction_policy_clone.as_str());
    println!("  Compaction threshold: {}", compaction_threshold);
    
    let lsm_tree = Arc::new(RwLock::new(LSMTree::with_config(config)));

    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr)?;
    listener.set_nonblocking(true)?;
    println!("Server listening on {}", addr);

    let termination_flag = Arc::new(AtomicBool::new(false));
    
    // Set up a stdin reader for server commands
    let server_flag = Arc::clone(&termination_flag);
    
    std::thread::spawn(move || {
        let stdin = io::stdin();
        let mut buffer = String::new();
        
        while !server_flag.load(Ordering::SeqCst) {
            buffer.clear();
            
            print!("> ");
            std::io::stdout().flush().unwrap();
            
            if stdin.read_line(&mut buffer).is_ok() {
                if let Some(response) = handle_server_command(&buffer, &server_flag) {
                    println!("{}", response);
                } else if !buffer.trim().is_empty() {
                    println!("Unknown server command: {}", buffer.trim());
                    println!("Type 'help' for available commands");
                }
            }
            
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    while !termination_flag.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                println!("New client connected");
                let termination_flag = Arc::clone(&termination_flag);
                let lsm_tree = Arc::clone(&lsm_tree);
                std::thread::spawn(move || {
                    handle_client(stream, termination_flag, lsm_tree);
                });
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            Err(e) => eprintln!("Error accepting connection: {}", e),
        }
    }

    println!("Server shut down.");
    Ok(())
}
