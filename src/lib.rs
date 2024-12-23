pub mod command;
pub mod memtable;
pub mod types;
pub mod lsm_tree;
pub mod test_helpers;
mod run;
mod level;

// Constants
pub const DEFAULT_PORT: u16 = 8080;
pub const BUFFER_SIZE: usize = 1024;
pub const END_OF_MESSAGE: &str = "\r\n\r\n";
pub const SERVER_SHUTDOWN: &str = "SERVER_SHUTDOWN";
pub const OK: &str = "OK";
