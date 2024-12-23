use crate::types::{Key, Value};

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Put(Key, Value),
    Get(Key),
    Range(Key, Key),
    Delete(Key),
    Load(String),
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
                if parts.next().is_some() {
                    eprintln!("Extra parts in Put command: {}", input);
                    return None;
                }
                Some(Command::Put(key, value))
            }
            "g" => {
                let key = parts.next()?.parse().ok()?;
                if parts.next().is_some() {
                    eprintln!("Extra parts in Get command: {}", input);
                    return None;
                }
                Some(Command::Get(key))
            }
            "r" => {
                let start = parts.next()?.parse().ok()?;
                let end = parts.next()?.parse().ok()?;
                if parts.next().is_some() {
                    eprintln!("Extra parts in Range command: {}", input);
                    return None;
                }
                Some(Command::Range(start, end))
            }
            "d" => {
                let key = parts.next()?.parse().ok()?;
                if parts.next().is_some() {
                    eprintln!("Extra parts in Delete command: {}", input);
                    return None;
                }
                Some(Command::Delete(key))
            }
            "l" => {
                let filename = parts.next()?;
                if parts.next().is_some() {
                    eprintln!("Extra parts in Load command: {}", input);
                    return None;
                }
                Some(Command::Load(filename.trim_matches('"').to_string()))
            }
            "s" => {
                if parts.next().is_some() {
                    eprintln!("Extra parts in PrintStats command: {}", input);
                    return None;
                }
                Some(Command::PrintStats)
            }
            "q" => {
                if parts.next().is_some() {
                    eprintln!("Extra parts in Quit command: {}", input);
                    return None;
                }
                Some(Command::Quit)
            }
            _ => {
                eprintln!("Unknown command: {}", cmd);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_command() {
        assert!(matches!(Command::parse("p 10 42"), Some(Command::Put(10, 42))));
        assert!(matches!(Command::parse("p -10 -42"), Some(Command::Put(-10, -42))));
        assert_eq!(Command::parse("p"), None);
        assert_eq!(Command::parse("p 10"), None);
        assert_eq!(Command::parse("p 10 42 extra"), None);
    }

    #[test]
    fn test_get_command() {
        assert!(matches!(Command::parse("g 10"), Some(Command::Get(10))));
        assert!(matches!(Command::parse("g -10"), Some(Command::Get(-10))));
        assert_eq!(Command::parse("g"), None);
        assert_eq!(Command::parse("g 10 extra"), None);
    }

    #[test]
    fn test_range_command() {
        assert!(matches!(Command::parse("r 10 20"), Some(Command::Range(10, 20))));
        assert!(matches!(Command::parse("r -10 20"), Some(Command::Range(-10, 20))));
        assert_eq!(Command::parse("r"), None);
        assert_eq!(Command::parse("r 10"), None);
        assert_eq!(Command::parse("r 10 20 extra"), None);
    }

    #[test]
    fn test_delete_command() {
        assert!(matches!(Command::parse("d 10"), Some(Command::Delete(10))));
        assert!(matches!(Command::parse("d -10"), Some(Command::Delete(-10))));
        assert_eq!(Command::parse("d"), None);
        assert_eq!(Command::parse("d 10 extra"), None);
    }

    #[test]
    fn test_load_command() {
        assert!(matches!(
            Command::parse("l \"/path/to/file.bin\""),
            Some(Command::Load(s)) if s == "/path/to/file.bin"
        ));
        assert!(matches!(
            Command::parse("l file.bin"),
            Some(Command::Load(s)) if s == "file.bin"
        ));
        assert_eq!(Command::parse("l"), None);
    }

    #[test]
    fn test_print_stats_command() {
        assert!(matches!(Command::parse("s"), Some(Command::PrintStats)));
        assert_eq!(Command::parse("s extra"), None);
    }

    #[test]
    fn test_invalid_commands() {
        assert_eq!(Command::parse(""), None);
        assert_eq!(Command::parse("x"), None);
        assert_eq!(Command::parse("p x y"), None);
        assert_eq!(Command::parse("g x"), None);
        assert_eq!(Command::parse("r x y"), None);
    }

    #[test]
    fn test_quit_command() {
        assert!(matches!(Command::parse("q"), Some(Command::Quit)));
        assert_eq!(Command::parse("q extra"), None);
    }
}