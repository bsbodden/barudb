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
            if let Some(min_key) = run.min_key() {
                if let Some(max_key) = run.max_key() {
                    // Add all data from this run
                    all_data.extend(run.range(min_key, max_key + 1));
                }
            }
        }
        
        // Sort data by key and remove duplicates, keeping most recent value
        all_data.sort_by_key(|&(key, _)| key);
        all_data.dedup_by_key(|&mut (key, _)| key);
        
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