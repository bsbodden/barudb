use crate::run::{Block, BlockKey, RunId};
use crate::run::cache_policies::{LRUPolicy, TinyLFUPolicy, CachePolicy, CachePolicyType, CachePolicyFactory};
use std::sync::Arc;
use std::thread;

// Helper function to create a test block
fn create_test_block(key: i64) -> Block {
    let mut block = Block::new();
    block.add_entry(key, key * 10).unwrap();
    block.seal().unwrap();
    block
}

// Helper function to create a block key
fn create_block_key(idx: usize) -> BlockKey {
    BlockKey {
        run_id: RunId::new(0, 1),
        block_idx: idx,
    }
}

#[test]
fn test_lru_policy_basic() {
    // Create an LRU policy with capacity 3
    let policy = LRUPolicy::new(3);
    
    // Create blocks and keys
    let key1 = create_block_key(1);
    let key2 = create_block_key(2);
    let key3 = create_block_key(3);
    let key4 = create_block_key(4);
    
    let block1 = Arc::new(create_test_block(1));
    let block2 = Arc::new(create_test_block(2));
    let block3 = Arc::new(create_test_block(3));
    let block4 = Arc::new(create_test_block(4));
    
    // Add blocks to policy
    assert_eq!(policy.add(key1, block1.clone()), None);
    assert_eq!(policy.add(key2, block2.clone()), None);
    assert_eq!(policy.add(key3, block3.clone()), None);
    
    // Check cache contents
    assert!(policy.contains(&key1));
    assert!(policy.contains(&key2));
    assert!(policy.contains(&key3));
    assert!(!policy.contains(&key4));
    
    // Access key1 to move it to the back of the queue
    assert!(policy.access(&key1));
    
    // Add key4, should evict key2 (oldest after key1 was accessed)
    assert_eq!(policy.add(key4, block4.clone()), Some(key2));
    
    // Check cache contents after eviction
    assert!(policy.contains(&key1));
    assert!(!policy.contains(&key2));
    assert!(policy.contains(&key3));
    assert!(policy.contains(&key4));
    
    // Get values
    assert_eq!(policy.get(&key1).unwrap().get(&1), Some(10));
    assert!(policy.get(&key2).is_none());
    assert_eq!(policy.get(&key3).unwrap().get(&3), Some(30));
    assert_eq!(policy.get(&key4).unwrap().get(&4), Some(40));
    
    // Remove a key
    assert!(policy.remove(&key3).is_some());
    assert!(!policy.contains(&key3));
    assert_eq!(policy.len(), 2);
    
    // Clear cache
    policy.clear();
    assert_eq!(policy.len(), 0);
    assert!(!policy.contains(&key1));
    assert!(!policy.contains(&key4));
}

#[test]
fn test_lru_policy_concurrent() {
    // We need to specify the policy type to make the compiler happy
    let policy = Arc::new(LRUPolicy::new(100));
    let mut handles = vec![];
    
    // Spawn 10 threads, each adding 10 blocks
    for t in 0..10 {
        let policy_clone = Arc::clone(&policy);
        
        let handle = thread::spawn(move || {
            for i in 0..10 {
                let key = create_block_key(t * 10 + i);
                let block = Arc::new(create_test_block((t * 10 + i) as i64));
                policy_clone.add(key, block);
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Check that all 100 blocks were added
    assert_eq!(policy.len(), 100);
    
    // Check that all keys are present
    for i in 0..100 {
        let key = create_block_key(i);
        assert!(policy.contains(&key));
    }
}

#[test]
fn test_cache_policy_factory() {
    // Test LRU creation
    let lru = CachePolicyFactory::create(CachePolicyType::LRU, 10);
    assert_eq!(lru.capacity(), 10);
    
    // Test TinyLFU creation
    let tiny_lfu = CachePolicyFactory::create(CachePolicyType::TinyLFU, 20);
    assert_eq!(tiny_lfu.capacity(), 20);
    
    // Test functionality
    let key = create_block_key(1);
    let block = Arc::new(create_test_block(1));
    
    assert!(lru.add(key, block.clone()).is_none());
    assert!(lru.contains(&key));
    assert_eq!(lru.get(&key).unwrap().get(&1), Some(10));
}

#[test]
fn test_tiny_lfu_basic() {
    // Create a TinyLFU policy with small capacity
    let policy = TinyLFUPolicy::new(3);
    
    // Create 3 keys and blocks
    let key1 = create_block_key(1);
    let key2 = create_block_key(2);
    let key3 = create_block_key(3);
    
    let block1 = Arc::new(create_test_block(1));
    let block2 = Arc::new(create_test_block(2));
    let block3 = Arc::new(create_test_block(3));
    
    // Add blocks to cache
    policy.add(key1, block1);
    policy.add(key2, block2);
    policy.add(key3, block3);
    
    // Verify all blocks are in cache
    assert!(policy.contains(&key1));
    assert!(policy.contains(&key2));
    assert!(policy.contains(&key3));
    
    // Verify size
    assert_eq!(policy.len(), 3);
    
    // Clear cache
    policy.clear();
    
    // Verify cache is empty
    assert_eq!(policy.len(), 0);
}
