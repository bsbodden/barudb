// in test_helpers.rs
use std::process::{Command, Child};
use tokio::net::TcpStream;
use std::net::TcpListener;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::time::{sleep, Duration};
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Instant;

use crate::END_OF_MESSAGE;

// Base port for testing
static PORT_COUNTER: AtomicU16 = AtomicU16::new(8080);

async fn wait_for_port_available(port: u16) -> bool {
    let addr = format!("127.0.0.1:{}", port);
    for _ in 0..10 {
        match TcpListener::bind(&addr) {
            Ok(listener) => {
                drop(listener);
                return true;
            },
            Err(_) => {
                sleep(Duration::from_millis(100)).await;
            }
        }
    }
    false
}

fn find_available_port() -> u16 {
    loop {
        let port = PORT_COUNTER.fetch_add(1, Ordering::SeqCst);
        if port >= 9000 {
            PORT_COUNTER.store(8080, Ordering::SeqCst);
            continue;
        }

        match TcpListener::bind(format!("127.0.0.1:{}", port)) {
            Ok(listener) => {
                let addr = listener.local_addr().unwrap();
                drop(listener);
                return addr.port();
            }
            Err(_) => continue,
        }
    }
}

pub async fn start_server() -> (Child, u16) {
    // Kill any existing server processes
    Command::new("pkill")
        .arg("-f")
        .arg("target/release/server")
        .output()
        .ok();

    sleep(Duration::from_millis(500)).await;

    let port = find_available_port();
    println!("Attempting to start server on port {}", port);

    // Wait for port to be truly available
    if !wait_for_port_available(port).await {
        panic!("Port {} is still in use after cleanup", port);
    }

    let mut server = Command::new("cargo")
        .arg("run")
        .arg("--release")
        .arg("--bin")
        .arg("server")
        .env("SERVER_PORT", port.to_string())
        .spawn()
        .expect("Failed to start server");

    // Wait for server to start accepting connections
    let start = Instant::now();
    let timeout = Duration::from_secs(5);

    while start.elapsed() < timeout {
        match TcpStream::connect(format!("127.0.0.1:{}", port)).await {
            Ok(_) => {
                println!("Server successfully started on port {}", port);
                sleep(Duration::from_millis(100)).await;
                return (server, port);
            }
            Err(_) => {
                if !server.try_wait().map(|s| s.is_none()).unwrap_or(false) {
                    println!("Server process exited prematurely");
                    let _ = server.kill();
                    panic!("Server failed to start: process exited");
                }
                sleep(Duration::from_millis(100)).await;
            }
        }
    }

    // If we get here, server failed to start
    let _ = server.kill();
    panic!("Server failed to start listening on port {} after {:?}", port, timeout);
}

pub async fn send_command(port: u16, command: &str) -> String {
    let mut attempts = 3;
    while attempts > 0 {
        match TcpStream::connect(format!("127.0.0.1:{}", port)).await {
            Ok(mut stream) => {
                stream.write_all(command.as_bytes()).await
                    .expect("Failed to send command");

                let mut response = String::new();
                let mut buffer = vec![0u8; 1024];

                loop {
                    match stream.read(&mut buffer).await {
                        Ok(n) if n == 0 => break,
                        Ok(n) => {
                            response.push_str(&String::from_utf8_lossy(&buffer[..n]));
                            if response.ends_with(END_OF_MESSAGE) {
                                response.truncate(response.len() - END_OF_MESSAGE.len());
                                return response;
                            }
                        }
                        Err(e) => panic!("Failed to read from socket: {}", e),
                    }
                }
                return response;
            }
            Err(_) if attempts > 1 => {
                attempts -= 1;
                sleep(Duration::from_millis(100)).await;
                continue;
            }
            Err(e) => panic!("Failed to connect to server on port {}: {}", port, e),
        }
    }
    panic!("Failed to connect to server after multiple attempts");
}

pub async fn shutdown_server(mut server: Child, port: u16) {
    // Try graceful shutdown first
    match send_command(port, "q\n").await {
        response if response.trim() == "Shutting down the server" => {
            for _ in 0..10 {
                if server.try_wait().map(|s| s.is_some()).unwrap_or(false) {
                    return;
                }
                sleep(Duration::from_millis(100)).await;
            }
        },
        _ => ()  // Failed to get proper shutdown response
    }

    // Force kill if graceful shutdown failed
    let _ = server.kill();
    let _ = Command::new("pkill")
        .arg("-f")
        .arg("target/release/server")
        .output();

    sleep(Duration::from_millis(500)).await;

    // Verify port is freed
    let _ = wait_for_port_available(port).await;
}