// tests/integration_test.rs
use tokio::net::TcpStream;
use std::net::TcpListener;
use std::process::{Child, Command};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::time::Duration;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::io::Write;

static NEXT_PORT: Lazy<Mutex<u16>> = Lazy::new(|| Mutex::new(8080));

struct TestServer {
    port: u16,
    process: Child,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        // Try graceful shutdown first
        println!("Shutting down test server on port {}", self.port);
        if let Ok(mut stream) = std::net::TcpStream::connect(format!("127.0.0.1:{}", self.port)) {
            let _ = stream.write_all(b"q\n");
            let _ = stream.flush();
        }
        let _ = self.process.kill();
    }
}

// Ensure server binary is built before running tests
fn build_server() {
    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--bin")
        .arg("server")
        .status()
        .expect("Failed to build server");

    assert!(status.success(), "Server build failed");
}

async fn start_server() -> TestServer {
    // Clear any existing servers first
    let _ = Command::new("pkill")
        .arg("-f")
        .arg("target/release/server")
        .output();

    tokio::time::sleep(Duration::from_millis(500)).await;

    let mut port = NEXT_PORT.lock().unwrap().clone();

    // Find an available port
    while TcpListener::bind(format!("127.0.0.1:{}", port)).is_err() {
        port += 1;
    }
    *NEXT_PORT.lock().unwrap() = port + 1;

    println!("Starting test server on port {}", port);

    let mut process = Command::new("./target/release/server")  // Use direct path to binary
        .env("SERVER_PORT", port.to_string())
        .spawn()
        .expect("Failed to start server process");

    // Wait for server to be ready
    let mut attempts = 50;  // 5 seconds total
    while attempts > 0 {
        if let Ok(_) = TcpStream::connect(format!("127.0.0.1:{}", port)).await {
            tokio::time::sleep(Duration::from_millis(200)).await;
            println!("Successfully connected to test server on port {}", port);
            return TestServer { port, process };
        }
        attempts -= 1;
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check if process has died
        match process.try_wait() {
            Ok(Some(status)) => {
                panic!("Server process exited prematurely with status: {}", status);
            }
            Err(e) => {
                panic!("Error checking server process: {}", e);
            }
            _ => ()  // Process still running
        }
    }

    // If we get here, server failed to start
    let _ = process.kill();
    panic!("Server failed to start after 50 attempts");
}

async fn send_command(port: u16, cmd: &str) -> String {
    println!("Sending command to server: {}", cmd.trim());
    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port))
        .await
        .expect("Failed to connect to server");

    stream.write_all(cmd.as_bytes()).await.unwrap();

    let mut response = String::new();
    let mut buf = [0; 1024];

    while let Ok(n) = stream.read(&mut buf).await {
        if n == 0 { break; }
        response.push_str(&String::from_utf8_lossy(&buf[..n]));
        if response.ends_with("\r\n\r\n") {
            response.truncate(response.len() - 4);
            println!("Received response: {}", response);
            break;
        }
    }
    response
}

#[tokio::test]
async fn test_put_and_get() {
    build_server();  // Ensure server is built before test
    let server = start_server().await;

    let response = send_command(server.port, "p 10 42\n").await;
    assert_eq!(response, "OK", "Put failed");

    let response = send_command(server.port, "g 10\n").await;
    assert_eq!(response, "42", "Get returned wrong value");

    let response = send_command(server.port, "g 11\n").await;
    assert_eq!(response, "", "Get of non-existent value should return empty string");
}

#[tokio::test]
async fn test_range_query() {
    build_server();  // Ensure server is built before test
    let server = start_server().await;

    let response = send_command(server.port, "p 10 42\n").await;
    assert_eq!(response, "OK", "First put failed");

    let response = send_command(server.port, "p 20 84\n").await;
    assert_eq!(response, "OK", "Second put failed");

    let response = send_command(server.port, "p 30 126\n").await;
    assert_eq!(response, "OK", "Third put failed");

    let response = send_command(server.port, "r 10 30\n").await;
    assert_eq!(response, "10:42 20:84", "Range query returned wrong result");

    let response = send_command(server.port, "r 40 50\n").await;
    assert_eq!(response, "", "Empty range should return empty string");
}