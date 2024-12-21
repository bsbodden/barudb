use lsm_tree::command::Command;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn send_response(stream: &mut TcpStream, response: &str) -> io::Result<()> {
    // Send length first
    let response_len = response.len() as u32;
    stream.write_all(&response_len.to_be_bytes())?;

    // Then send the response
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn handle_client(mut stream: TcpStream, termination_flag: Arc<AtomicBool>) {
    stream
        .set_nonblocking(false)
        .expect("Failed to set blocking");
    let mut reader = BufReader::new(stream.try_clone().unwrap());

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
                        println!("Received Put({}, {})", key, value);
                        "OK"
                    }
                    Some(Command::Get(key)) => {
                        println!("Received Get({})", key);
                        "42" // Dummy value
                    }
                    Some(Command::Range(start, end)) => {
                        println!("Received Range({}, {})", start, end);
                        "10:42 11:43 12:44"
                    }
                    Some(Command::Delete(key)) => {
                        println!("Received Delete({})", key);
                        "OK"
                    }
                    Some(Command::Load(filename)) => {
                        println!("Received Load({})", filename);
                        "OK"
                    }
                    Some(Command::PrintStats) => {
                        println!("Received PrintStats");
                        "Logical Pairs: 3\nLVL1: 3\n10:42:L1 11:43:L1 12:44:L1"
                    }
                    Some(Command::Quit) => {
                        println!("Client requested quit, shutting down server...");
                        termination_flag.store(true, Ordering::SeqCst);
                        "Shutting down the server"
                    }
                    None => "Invalid command",
                };

                if let Err(e) = send_response(&mut stream, response) {
                    eprintln!("Failed to send response: {}", e);
                    break;
                }

                if termination_flag.load(Ordering::SeqCst) {
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error reading from client: {}", e);
                break;
            }
        }
    }
}

fn main() -> io::Result<()> {
    let addr = "127.0.0.1:8080";
    let listener = TcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    println!("Server listening on {}", addr);

    let termination_flag = Arc::new(AtomicBool::new(false));

    while !termination_flag.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                println!("New client connected");
                let termination_flag = Arc::clone(&termination_flag);
                std::thread::spawn(move || {
                    handle_client(stream, termination_flag);
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
