#[cfg(feature = "use_lmdb")]
use lmdb::{Environment, Cursor, DbFlags, WriteFlags};
#[cfg(feature = "use_lmdb")]
use tempfile::TempDir;

#[test]
#[cfg(feature = "use_lmdb")]
fn test_lmdb_works() {
    // Create a temporary directory for the database
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    
    // Create the LMDB environment
    let env = Environment::new()
        .set_map_size(1024 * 1024 * 10) // 10MB
        .set_max_dbs(1)
        .open(temp_dir.path())
        .expect("Failed to create LMDB environment");
    
    // Create a database
    let db = env.open_db(None).expect("Failed to open default database");
    
    // Put a key-value pair
    let key = "test_key";
    let value = "test_value";
    
    let txn = env.begin_rw_txn().expect("Failed to begin transaction");
    txn.put(db, &key, &value, WriteFlags::empty())
        .expect("Failed to write to database");
    txn.commit().expect("Failed to commit transaction");
    
    // Read back the value
    let txn = env.begin_ro_txn().expect("Failed to begin read transaction");
    let result = txn.get(db, &key).expect("Failed to get value");
    
    assert_eq!(result, value.as_bytes());
    
    println!("LMDB test successful!");
}