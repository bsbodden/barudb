use std::io;
use std::io::{BufReader, Write};
use std::net::TcpStream;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use lsm_tree::command::Command;
use lsm_tree::lsm_tree::LSMTree;
use std::net::TcpListener;
use std::io::BufRead;

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
                    Some(Command::Load(_)) => {
                        eprintln!("Load command is not implemented");
                        "Error: Load command not implemented".to_string()
                    }
                    Some(Command::PrintStats) => {
                        eprintln!("PrintStats command is not implemented");
                        "Error: PrintStats command not implemented".to_string()
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

fn main() -> io::Result<()> {
    // Get port from environment or use default
    let port = std::env::var("SERVER_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr)?;
    listener.set_nonblocking(true)?;
    println!("Server listening on {}", addr);

    let termination_flag = Arc::new(AtomicBool::new(false));
    let lsm_tree = Arc::new(RwLock::new(LSMTree::new(128))); // Initialize LSMTree with buffer size 128

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