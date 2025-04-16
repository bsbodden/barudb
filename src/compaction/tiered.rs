use crate::compaction::CompactionPolicy;
use crate::level::Level;
use crate::run::{Run, RunStorage};
use crate::types::{Result, Error};

/// Tiered compaction policy merges all runs in a level when the level has
/// reached a run threshold. This is the simplest compaction strategy.
pub struct TieredCompactionPolicy {
    /// Number of runs that trigger compaction
    run_threshold: usize,
}

impl TieredCompactionPolicy {
    /// Create a new tiered compaction policy with the specified run threshold
    pub fn new(run_threshold: usize) -> Self {
        Self { run_threshold }
    }
}

impl CompactionPolicy for TieredCompactionPolicy {
    fn should_compact(&self, level: &Level, _level_num: usize) -> bool {
        // Compact when number of runs reaches or exceeds threshold
        level.run_count() >= self.run_threshold
    }
    
    fn select_runs_to_compact(&self, level: &Level) -> Vec<usize> {
        // In tiered compaction, we select all runs for compaction
        (0..level.run_count()).collect()
    }
    
    fn compact(
        &self,
        source_level: &Level,
        target_level: &mut Level,
        storage: &dyn RunStorage,
        _source_level_num: usize,
        target_level_num: usize,
        config: Option<&crate::lsm_tree::LSMConfig>,
    ) -> Result<Run> {
        let run_indices = self.select_runs_to_compact(source_level);
        
        if run_indices.is_empty() {
            return Err(Error::CompactionError);
        }
        
        // Collect all runs to compact
        let mut runs = Vec::new();
        for idx in run_indices {
            runs.push(source_level.get_run(idx));
        }
        
        if runs.is_empty() {
            return Err(Error::CompactionError);
        }
        
        // Merge all key-value pairs from selected runs
        let mut all_data = Vec::new();
        for run in &runs {
            if let Some(_) = run.min_key() {
                if let Some(_) = run.max_key() {
                    // CRITICAL FIX: Instead of using run.range() which might filter tombstones,
                    // directly access the run's data which includes ALL key-value pairs including tombstones
                    all_data.extend(run.data.clone());
                    
                    // Debug output for tombstones
                    if std::env::var("RUST_LOG").map(|v| v == "debug").unwrap_or(false) {
                        let tombstone_count = run.data.iter()
                            .filter(|(_, v)| *v == crate::types::TOMBSTONE)
                            .count();
                        if tombstone_count > 0 {
                            println!("Found {} tombstones in run during compaction", tombstone_count);
                        }
                    }
                }
            }
        }
        
        // CRITICAL: The standard dedup_by_key doesn't guarantee which element is kept
        // when removing duplicates. For LSM trees, we need to keep the element from the
        // most recent run, which should be the first one in our list.
        
        // First, sort by key to group duplicates
        all_data.sort_by_key(|&(key, _)| key);
        
        // Now custom deduplication that keeps tombstones
        let mut result = Vec::with_capacity(all_data.len());
        let mut current_key = None;
        
        // We need to process the data in REVERSE order to keep the most recent values
        // (since the most recent runs are added first to all_data)
        // NOTE: We need to iterate in reverse so we keep newer entries (including tombstones)
        // when there are duplicates.
        let mut all_data_reversed = all_data; // Take ownership
        all_data_reversed.reverse();  // Process in reverse order!
                
        for (key, value) in all_data_reversed {
            match current_key {
                Some(k) if k == key => {
                    // Skip this duplicate key (we already processed a more recent value)
                    if std::env::var("RUST_LOG").map(|v| v == "debug").unwrap_or(false) {
                        println!("Skipping duplicate key {} with value {}", key, value);
                    }
                    continue;
                }
                _ => {
                    // First time seeing this key, keep it
                    current_key = Some(key);
                    result.push((key, value));
                    
                    // Debug output for tombstones
                    if value == crate::types::TOMBSTONE && 
                       std::env::var("RUST_LOG").map(|v| v == "debug").unwrap_or(false) {
                        println!("Keeping tombstone for key: {}", key);
                    }
                }
            }
        }
        
        // Restore the original order (sort by key) for the result
        result.sort_by_key(|&(key, _)| key);
        
        // Replace all_data with our deduplicated result
        all_data = result;
        
        // Debug output for tombstones after deduplication
        if std::env::var("RUST_LOG").map(|v| v == "debug").unwrap_or(false) {
            let tombstone_count_after = all_data.iter()
                .filter(|(_, v)| *v == crate::types::TOMBSTONE)
                .count();
            
            println!("After deduplication: {} tombstones remain in the merged data", tombstone_count_after);
            
            // Print details of remaining tombstones
            if tombstone_count_after > 0 {
                for (key, value) in &all_data {
                    if *value == crate::types::TOMBSTONE {
                        println!("Merged data contains tombstone for key: {}", key);
                    }
                }
            }
        }
        
        // Create a new run with the merged data
        let fanout = config.map(|c| c.fanout as f64).unwrap_or(4.0);
        let mut merged_run = Run::new_for_level(all_data, target_level_num, fanout, config);
        
        // Store the run in the target level
        let run_id = merged_run.store(storage, target_level_num)?;
        merged_run.id = Some(run_id);
        
        // Add the run to the target level
        target_level.add_run(merged_run.clone());
        
        // Delete the old runs from storage
        for run in runs {
            if let Some(id) = run.id {
                // Ignore errors during cleanup
                let _ = storage.delete_run(id);
            }
        }
        
        Ok(merged_run)
    }
    
    fn box_clone(&self) -> Box<dyn CompactionPolicy> {
        Box::new(Self {
            run_threshold: self.run_threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tiered_compaction_policy_should_compact() {
        let policy = TieredCompactionPolicy::new(3);
        
        // Create a mock level with 2 runs
        let mut level = Level::new();
        level.add_run(Run::new(vec![(1, 100)]));
        level.add_run(Run::new(vec![(2, 200)]));
        
        // Should not compact yet
        assert!(!policy.should_compact(&level, 0));
        
        // Add one more run to reach threshold
        level.add_run(Run::new(vec![(3, 300)]));
        
        // Should now compact
        assert!(policy.should_compact(&level, 0));
    }
    
    #[test]
    fn test_tiered_compaction_policy_select_runs() {
        let policy = TieredCompactionPolicy::new(3);
        
        // Create a mock level with 3 runs
        let mut level = Level::new();
        level.add_run(Run::new(vec![(1, 100)]));
        level.add_run(Run::new(vec![(2, 200)]));
        level.add_run(Run::new(vec![(3, 300)]));
        
        // Should select all runs
        let selected = policy.select_runs_to_compact(&level);
        assert_eq!(selected, vec![0, 1, 2]);
    }
}