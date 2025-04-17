use lsm_tree::lsm_tree::{LSMTree, LSMConfig, DynamicBloomFilterConfig};
use lsm_tree::types::{CompactionPolicyType, StorageType};
use tempfile::tempdir;

#[test]
fn test_lsm_tree_with_compaction_policy() {
    // Create a temporary directory for storage
    let temp_dir = tempdir().unwrap();
    
    // Create LSM tree with tiered compaction policy
    let config = LSMConfig {
        buffer_size: 4, // Small buffer for testing
        storage_type: StorageType::File,
        storage_path: temp_dir.path().to_path_buf(),
        create_path_if_missing: true,
        max_open_files: 100,
        sync_writes: false,
        fanout: 4,
        compaction_policy: CompactionPolicyType::Tiered,
        compaction_threshold: 2, // Compact after 2 runs
        compression: lsm_tree::run::CompressionConfig::default(),
        adaptive_compression: lsm_tree::run::AdaptiveCompressionConfig::default(),
        collect_compression_stats: true,
        background_compaction: false,
        use_lock_free_memtable: false,  // Use standard sharded memtable by default
        use_lock_free_block_cache: true, // Use lock-free block cache by default
        dynamic_bloom_filter: DynamicBloomFilterConfig::default(),
    };
    
    let mut tree = LSMTree::with_config(config);
    
    // Insert data to trigger multiple flushes and compaction
    for i in 1..20 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Verify all data is still accessible
    for i in 1..20 {
        assert_eq!(tree.get(i), Some(i * 100), "Lost data at key {}", i);
    }
    
    // Test range query
    let range = tree.range(5, 10);
    assert_eq!(range.len(), 5);
    assert_eq!(range[0], (5, 500));
    assert_eq!(range[4], (9, 900));
    
    // Test deletion
    tree.delete(10).unwrap();
    assert_eq!(tree.get(10), None);
    
    // Add more data to trigger additional compaction
    for i in 20..30 {
        tree.put(i, i * 100).unwrap();
    }
    
    // Forced compaction
    tree.force_compact_all().unwrap();
    
    // Verify all data, including the undeleted data
    for i in 1..10 {
        assert_eq!(tree.get(i), Some(i * 100));
    }
    assert_eq!(tree.get(10), None); // Deleted
    for i in 11..30 {
        assert_eq!(tree.get(i), Some(i * 100));
    }
}