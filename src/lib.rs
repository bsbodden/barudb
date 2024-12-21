pub mod command;

pub const DEFAULT_PORT: u16 = 8080;
pub const BUFFER_SIZE: usize = 1024;
pub const END_OF_MESSAGE: &str = "\r\n\r\n";
pub const SERVER_SHUTDOWN: &str = "SERVER_SHUTDOWN";
pub const OK: &str = "OK";
