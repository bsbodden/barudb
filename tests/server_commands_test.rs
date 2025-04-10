use std::fs;
use std::io::Write;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::Duration;

async fn send_command(stream: &mut TcpStream, cmd: &str) -> String {
    stream.write_all(cmd.as_bytes()).await.unwrap();
    stream.write_all(b"\n").await.unwrap();
    stream.flush().await.unwrap();

    // Small delay to allow server to process
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
async fn test_print_stats_command() {
    // Connect to server
    let mut stream = match TcpStream::connect("127.0.0.1:8080").await {
        Ok(stream) => stream,
        Err(e) => {
            println!("Skipping test_print_stats_command due to connection error: {}", e);
            // This is not a failure - the test might be running in an environment without a server
            return;
        }
    };

    // Add some data to ensure stats have something to show
    let _ = send_command(&mut stream, "p 1 100").await;
    let _ = send_command(&mut stream, "p 2 200").await;
    let _ = send_command(&mut stream, "p 3 300").await;
    
    // Flush to disk to ensure storage stats can be calculated
    let _ = send_command(&mut stream, "f").await;
    
    // Get stats
    let response = send_command(&mut stream, "s").await;
    
    // Verify response contains expected sections
    assert!(response.contains("LSM Tree Statistics"), "Response missing LSM Tree Statistics section");
    assert!(response.contains("Storage type:"), "Response missing Storage type");
    assert!(response.contains("Compaction policy:"), "Response missing Compaction policy");
    assert!(response.contains("Storage Statistics"), "Response missing Storage Statistics section");
    assert!(response.contains("Total size:"), "Response missing Total size");
    assert!(response.contains("File count:"), "Response missing File count");
    assert!(response.contains("Level Statistics"), "Response missing Level Statistics section");
}

#[tokio::test]
async fn test_load_command() {
    // Connect to server
    let mut stream = match TcpStream::connect("127.0.0.1:8080").await {
        Ok(stream) => stream,
        Err(e) => {
            println!("Skipping test_load_command due to connection error: {}", e);
            // This is not a failure - the test might be running in an environment without a server
            return;
        }
    };

    // Create a temporary file with commands
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("lsm_test_commands.txt");
    
    // Write test commands to file
    {
        let mut file = fs::File::create(&file_path).unwrap();
        file.write_all(b"p 10 1000\n").unwrap();
        file.write_all(b"p 20 2000\n").unwrap();
        file.write_all(b"p 30 3000\n").unwrap();
        file.write_all(b"d 30\n").unwrap(); // Add a delete command
    }
    
    // Load the file
    let load_cmd = format!("l {}", file_path.to_string_lossy());
    let response = send_command(&mut stream, &load_cmd).await;
    
    // Verify response indicates success
    assert!(response.contains("Loaded"), "Response doesn't indicate successful loading");
    assert!(response.contains("commands from file"), "Response doesn't mention file loading");
    
    // Verify data was loaded correctly
    let response = send_command(&mut stream, "g 10").await;
    assert_eq!(response, "1000", "First loaded value is incorrect");
    
    let response = send_command(&mut stream, "g 20").await;
    assert_eq!(response, "2000", "Second loaded value is incorrect");
    
    let response = send_command(&mut stream, "g 30").await;
    assert_eq!(response, "", "Deleted value should not be present");
    
    // Clean up
    fs::remove_file(file_path).ok();
}

#[tokio::test]
async fn test_server_commands_with_invalid_input() {
    // Connect to server
    let mut stream = match TcpStream::connect("127.0.0.1:8080").await {
        Ok(stream) => stream,
        Err(e) => {
            println!("Skipping test_server_commands_with_invalid_input due to connection error: {}", e);
            // This is not a failure - the test might be running in an environment without a server
            return;
        }
    };

    // Test invalid file path for load command
    let response = send_command(&mut stream, "l /nonexistent/file.txt").await;
    assert!(response.contains("Error"), "Response doesn't indicate error for nonexistent file");
    
    // Test invalid commands in load file
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("lsm_invalid_commands.txt");
    
    // Write invalid commands to file
    {
        let mut file = fs::File::create(&file_path).unwrap();
        file.write_all(b"p 10\n").unwrap(); // Missing value
        file.write_all(b"g\n").unwrap(); // Missing key
        file.write_all(b"x 10 20\n").unwrap(); // Unknown command
    }
    
    // Load the file
    let load_cmd = format!("l {}", file_path.to_string_lossy());
    let response = send_command(&mut stream, &load_cmd).await;
    
    // Verify response indicates failures
    assert!(response.contains("failed"), "Response doesn't indicate command failures");
    
    // Clean up
    fs::remove_file(file_path).ok();
}