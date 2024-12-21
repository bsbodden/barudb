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
    // Read length prefix (4 bytes for u32)
    let mut len_bytes = [0u8; 4];
    stream.read_exact(&mut len_bytes)?;
    let length = u32::from_be_bytes(len_bytes) as usize;

    // Read the actual response
    let mut buffer = vec![0u8; length];
    stream.read_exact(&mut buffer)?;

    String::from_utf8(buffer).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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
