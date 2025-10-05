use barudb::test_helpers::{send_command, start_server};

#[tokio::test]
#[ignore = "Long-running workload test; run explicitly with 'cargo test test_workload_execution -- --ignored'"]
async fn test_workload_execution() {
    let (mut server, port) = start_server().await;

    let workload = tokio::fs::read_to_string("./generator/workload.txt").await.unwrap();
    for line in workload.lines() {
        let response = send_command(port, &format!("{}\n", line)).await; // Pass `port`
        println!("Command: {}, Response: {}", line, response);
    }

    let _ = send_command(port, "q\n").await; // Pass `port`
    tokio::time::sleep(std::time::Duration::from_secs(1)).await; // Allow shutdown
    server.kill().expect("Failed to kill server"); // Properly kill server
}

