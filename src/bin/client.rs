use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;

fn send_command(stream: &mut TcpStream, command: &str) -> io::Result<()> {
    // Send the command
    stream.write_all(command.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

fn receive_response(stream: &mut TcpStream) -> io::Result<String> {
    let mut response = String::new();
    let mut buffer = [0u8; 1024];

    loop {
        match stream.read(&mut buffer) {
            Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Connection closed")),
            Ok(n) => {
                response.push_str(&String::from_utf8_lossy(&buffer[..n]));
                if response.ends_with(lsm_tree::END_OF_MESSAGE) {
                    // Remove the END_OF_MESSAGE marker and return
                    response.truncate(response.len() - lsm_tree::END_OF_MESSAGE.len());
                    return Ok(response);
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

fn main() -> io::Result<()> {
    let mut stream = TcpStream::connect("127.0.0.1:8080")?;
    println!("Connected to server at 127.0.0.1:8080");

    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut buffer = String::new();

    loop {
        print!("db_client > ");
        io::stdout().flush()?;

        buffer.clear();
        reader.read_line(&mut buffer)?;

        let command = buffer.trim();
        if command.is_empty() {
            continue;
        }

        send_command(&mut stream, command)?;

        match receive_response(&mut stream) {
            Ok(response) => {
                println!("{}", response);
                if response.contains("Shutting down the server") {
                    println!("Server is shutting down. Exiting client...");
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error receiving response: {}", e);
                break;
            }
        }

        if command == "q" {
            break;
        }
    }

    Ok(())
}