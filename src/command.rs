#[derive(Debug)]
pub enum Command {
    Put(i64, i64),   // key, value
    Get(i64),        // key
    Range(i64, i64), // start, end
    Delete(i64),     // key
    Load(String),    // filename
    PrintStats,
    Quit,
}

impl Command {
    pub fn parse(input: &str) -> Option<Command> {
        let mut parts = input.split_whitespace();
        let cmd = parts.next()?;

        match cmd {
            "p" => {
                let key = parts.next()?.parse().ok()?;
                let value = parts.next()?.parse().ok()?;
                Some(Command::Put(key, value))
            }
            "g" => {
                let key = parts.next()?.parse().ok()?;
                Some(Command::Get(key))
            }
            "r" => {
                let start = parts.next()?.parse().ok()?;
                let end = parts.next()?.parse().ok()?;
                Some(Command::Range(start, end))
            }
            "d" => {
                let key = parts.next()?.parse().ok()?;
                Some(Command::Delete(key))
            }
            "l" => {
                let filename = parts.next()?.to_string();
                Some(Command::Load(filename))
            }
            "s" => Some(Command::PrintStats),
            "q" => Some(Command::Quit),
            _ => None,
        }
    }
}
