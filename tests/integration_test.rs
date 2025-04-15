use once_cell::sync::Lazy;
use std::io::Write;
use std::net::TcpListener;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
// tests/integration_test.rs
use tokio::net::TcpStream;

static NEXT_PORT: Lazy<Mutex<u16>> = Lazy::new(|| Mutex::new(8080));

struct TestServer {
    port: u16,
    process: Child,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        // Try graceful shutdown first
        println!("Shutting down test server on port {}", self.port);
        
        // First try the graceful approach with the 'q' command
        if let Ok(mut stream) = std::net::TcpStream::connect(format!("127.0.0.1:{}", self.port)) {
            let _ = stream.write_all(b"q\n");
            let _ = stream.flush();
            
            // Give some time for the server to shut down gracefully
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        
        // Then check if the process is still running
        match self.process.try_wait() {
            Ok(Some(_)) => {
                println!("Server on port {} shut down gracefully", self.port);
            },
            _ => {
                // If still running, forcefully kill it
                println!("Forcefully killing server on port {}", self.port);
                let _ = self.process.kill();
                let _ = self.process.wait(); // Clean up zombie process
            }
        }
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

    tokio::time::sleep(Duration::from_millis(1000)).await;

    let mut port = NEXT_PORT.lock().unwrap().clone();

    // Find an available port
    while TcpListener::bind(format!("127.0.0.1:{}", port)).is_err() {
        port += 1;
    }
    *NEXT_PORT.lock().unwrap() = port + 1;

    println!("Starting test server on port {}", port);

    // Use relative path from the current working directory
    let server_path = std::env::current_dir()
        .expect("Failed to get current directory")
        .join("target")
        .join("release")
        .join("server");
        
    println!("Server binary path: {:?}", server_path);

    // Build command, preserving any existing DATA_DIR env var
    let mut cmd = Command::new(&server_path);
    cmd.env("SERVER_PORT", port.to_string());
    
    // Pass through DATA_DIR if it's set
    if let Ok(data_dir) = std::env::var("DATA_DIR") {
        println!("Using DATA_DIR: {}", data_dir);
        cmd.env("DATA_DIR", data_dir);
    }
    
    let mut process = cmd.spawn()
        .expect("Failed to start server process");

    // Wait for server to be ready
    let mut attempts = 100;  // 10 seconds total
    while attempts > 0 {
        if let Ok(_) = TcpStream::connect(format!("127.0.0.1:{}", port)).await {
            tokio::time::sleep(Duration::from_millis(500)).await;
            println!("Successfully connected to test server on port {}", port);
            return TestServer { port, process };
        }
        attempts -= 1;
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check if process has died
        match process.try_wait() {
            Ok(Some(status)) => {
                println!("Server process exited prematurely with status: {}", status);
                // Instead of panicking, try to start again with a different port
                break;
            }
            Err(e) => {
                println!("Error checking server process: {}", e);
                break;
            }
            _ => ()  // Process still running
        }
    }

    // If we get here, try once more with a different port
    port = port + 1;
    while TcpListener::bind(format!("127.0.0.1:{}", port)).is_err() {
        port += 1;
    }
    *NEXT_PORT.lock().unwrap() = port + 1;
    
    println!("Retrying with port {}", port);
    
    // Build command, preserving any existing DATA_DIR env var
    let mut cmd = Command::new(&server_path);
    cmd.env("SERVER_PORT", port.to_string());
    
    // Pass through DATA_DIR if it's set
    if let Ok(data_dir) = std::env::var("DATA_DIR") {
        println!("Using DATA_DIR: {}", data_dir);
        cmd.env("DATA_DIR", data_dir);
    }
    
    let mut process = cmd.spawn()
        .expect("Failed to start server process");
        
    // Wait for server to be ready
    let mut attempts = 100;  // 10 seconds total
    while attempts > 0 {
        if let Ok(_) = TcpStream::connect(format!("127.0.0.1:{}", port)).await {
            tokio::time::sleep(Duration::from_millis(500)).await;
            println!("Successfully connected to test server on port {}", port);
            return TestServer { port, process };
        }
        attempts -= 1;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    // If we get here, server failed to start
    let _ = process.kill();
    panic!("Server failed to start after multiple attempts");
}

async fn send_command(port: u16, cmd: &str) -> String {
    println!("Sending command to server: {}", cmd.trim());
    
    // Add retry logic for connection with more attempts and longer timeout
    let mut retry_count = 20; // Increased from 10
    let mut stream = None;
    
    while retry_count > 0 {
        match TcpStream::connect(format!("127.0.0.1:{}", port)).await {
            Ok(s) => {
                stream = Some(s);
                break;
            },
            Err(e) => {
                if retry_count == 1 {
                    panic!("Failed to connect to server: {}", e);
                }
                println!("Connection attempt failed, retrying in 500ms... ({} attempts left)", retry_count - 1);
                tokio::time::sleep(Duration::from_millis(500)).await;
                retry_count -= 1;
            }
        }
    }
    
    let mut stream = stream.unwrap();
    stream.write_all(cmd.as_bytes()).await.unwrap();

    // Add a small delay to ensure command is processed
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut response = String::new();
    let mut buf = [0; 1024];
    
    // Set a timeout for reading the response
    let read_timeout = tokio::time::timeout(
        Duration::from_secs(5),
        async {
            while let Ok(n) = stream.read(&mut buf).await {
                if n == 0 { break; }
                response.push_str(&String::from_utf8_lossy(&buf[..n]));
                if response.ends_with("\r\n\r\n") {
                    response.truncate(response.len() - 4);
                    println!("Received response: {}", response);
                    break;
                }
            }
        }
    ).await;
    
    if read_timeout.is_err() {
        println!("Warning: Response timeout, received partial response: {}", response);
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

#[tokio::test]
async fn test_recovery_from_restart_original() {
    build_server();  // Ensure server is built before test
    
    // Create a unique subdirectory for this test to isolate the data
    // Use nanoseconds for better uniqueness
    let test_dir = format!("test_recovery_{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos());
    
    // Use the more robust start_server utility instead of manual process creation
    println!("Setting DATA_DIR environment variable to {}", &test_dir);
    std::env::set_var("DATA_DIR", &test_dir);
    
    // Create first server
    let server1 = start_server().await;
    println!("First server started successfully on port {}", server1.port);
    
    // Insert data and force a flush to disk
    let response = send_command(server1.port, "p 100 1000\n").await;
    assert_eq!(response, "OK", "Put failed");
    
    let response = send_command(server1.port, "p 200 2000\n").await;
    assert_eq!(response, "OK", "Put failed");
    
    let response = send_command(server1.port, "p 300 3000\n").await;
    assert_eq!(response, "OK", "Put failed");
    
    // Force a flush to disk using the flush command
    let response = send_command(server1.port, "f\n").await;
    assert!(response.contains("OK"), "Flush failed: {}", response);
    
    // Verify data can still be accessed
    let response = send_command(server1.port, "g 100\n").await;
    assert_eq!(response, "1000", "Get failed before restart");
    
    // Explicitly drop the server to ensure clean shutdown
    println!("Shutting down first server...");
    drop(server1);
    
    // Wait for the server to fully shut down and release resources
    println!("Waiting for server to fully shut down...");
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Start a second server with the same data directory
    // Environment variable DATA_DIR is already set
    println!("Starting second server...");
    let server2 = start_server().await;
    println!("Second server started successfully on port {}", server2.port);
    
    // Check if data is recovered after restart
    let response = send_command(server2.port, "g 100\n").await;
    assert_eq!(response, "1000", "Data not recovered after restart");
    
    let response = send_command(server2.port, "g 200\n").await;
    assert_eq!(response, "2000", "Data not recovered after restart");
    
    let response = send_command(server2.port, "g 300\n").await;
    assert_eq!(response, "3000", "Data not recovered after restart");
    
    // Add new data to verify the server is fully functional
    let response = send_command(server2.port, "p 400 4000\n").await;
    assert_eq!(response, "OK", "Put failed after restart");
    
    let response = send_command(server2.port, "g 400\n").await;
    assert_eq!(response, "4000", "Get failed after restart");
    
    // Verify range query works after restart
    let response = send_command(server2.port, "r 100 300\n").await;
    assert_eq!(response, "100:1000 200:2000", "Range query failed after restart");
}

#[tokio::test]
async fn test_recovery_simplified() {
    build_server();  // Ensure server is built before test
    
    // Create a unique subdirectory for this test to isolate the data
    let test_dir = format!("test_recovery_simple_{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos());
    
    // Set environment variable for data directory
    println!("Setting DATA_DIR environment variable to {}", &test_dir);
    std::env::set_var("DATA_DIR", &test_dir);
    
    // Create a server
    let server = start_server().await;
    println!("First server started successfully on port {}", server.port);
    
    // Add data
    let response = send_command(server.port, "p 100 1000\n").await;
    assert_eq!(response, "OK", "Put failed");
    
    let response = send_command(server.port, "p 200 2000\n").await;
    assert_eq!(response, "OK", "Put failed");
    
    // Flush to disk
    let response = send_command(server.port, "f\n").await;
    assert!(response.contains("OK"), "Flush failed");
    
    // Explicitly drop the server to ensure clean shutdown
    println!("Shutting down first server...");
    drop(server);
    
    // Wait longer for the server to fully shut down
    println!("Waiting for server to fully shut down...");
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Start new server - DATA_DIR environment variable should still be set
    println!("Starting second server...");
    let new_server = start_server().await;
    println!("Second server started successfully on port {}", new_server.port);
    
    // Verify recovery
    let response = send_command(new_server.port, "g 100\n").await;
    assert!(response.contains("1000"), "Data not recovered after restart");
    
    // Also check the second value to be sure
    let response = send_command(new_server.port, "g 200\n").await;
    assert!(response.contains("2000"), "Data not fully recovered after restart");
}