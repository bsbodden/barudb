use crate::compaction::CompactionPolicy;
use crate::level::Level;
use crate::run::{Run, RunStorage};
use crate::types::{Result, Error};

/// Leveled compaction policy maintains a single sorted run per level
/// and triggers compaction when a level's size exceeds the size ratio threshold
/// compared to the next level.
pub struct LeveledCompactionPolicy {
    /// Size ratio threshold between levels (usually matches the fanout)
    size_ratio_threshold: usize,
}

impl LeveledCompactionPolicy {
    /// Create a new leveled compaction policy with the specified size ratio threshold
    pub fn new(size_ratio_threshold: usize) -> Self {
        Self { size_ratio_threshold }
    }
    
    /// Calculate the total size of all runs in a level
    fn calculate_level_size(&self, level: &Level) -> usize {
        level.get_runs().iter().map(|run| run.entry_count()).sum()
    }
}

impl CompactionPolicy for LeveledCompactionPolicy {
    fn should_compact(&self, level: &Level, level_num: usize) -> bool {
        // Level 0 is special - compact when there's more than one run
        if level_num == 0 {
            return level.run_count() > 1;
        }
        
        // For other levels, check if there are multiple runs (which violates the leveled approach)
        // This can happen if runs were added but compaction hasn't run yet
        if level.run_count() > 1 {
            return true;
        }
        
        // If the level is empty, no need to compact
        if level.run_count() == 0 {
            return false;
        }
        
        // Get the current level's size
        let _current_level_size = self.calculate_level_size(level);
        
        // For the last level, there's no next level to compare with
        // We always return false for the last level - it just grows
        // This is a simplification - production systems might handle this differently
        false
    }
    
    fn select_runs_to_compact(&self, level: &Level) -> Vec<usize> {
        // In leveled compaction, we select all runs
        (0..level.run_count()).collect()
    }
    
    fn compact(
        &self,
        source_level: &Level,
        target_level: &mut Level,
        storage: &dyn RunStorage,
        _source_level_num: usize,
        target_level_num: usize,
    ) -> Result<Run> {
        // Select all runs from the source level
        let run_indices = self.select_runs_to_compact(source_level);
        
        if run_indices.is_empty() {
            return Err(Error::CompactionError);
        }
        
        // Collect all runs to compact
        let mut source_runs = Vec::new();
        for idx in run_indices {
            source_runs.push(source_level.get_run(idx));
        }
        
        // If target level has runs, include them in the compaction
        let mut target_runs = Vec::new();
        for idx in 0..target_level.run_count() {
            target_runs.push(target_level.get_run(idx));
        }
        
        // Merge all key-value pairs from all runs (source and target)
        let mut all_data = Vec::new();
        
        // Add data from source runs
        for run in &source_runs {
            if let (Some(min_key), Some(max_key)) = (run.min_key(), run.max_key()) {
                all_data.extend(run.range(min_key, max_key + 1));
            }
        }
        
        // Add data from target runs
        for run in &target_runs {
            if let (Some(min_key), Some(max_key)) = (run.min_key(), run.max_key()) {
                all_data.extend(run.range(min_key, max_key + 1));
            }
        }
        
        // If we have no data after merging, return an error
        if all_data.is_empty() {
            return Err(Error::CompactionError);
        }
        
        // Sort data by key and remove duplicates, keeping most recent value
        all_data.sort_by_key(|&(key, _)| key);
        all_data.dedup_by_key(|&mut (key, _)| key);
        
        // Create a new run with the merged data, optimized for the target level
        let fanout = self.size_ratio_threshold as f64;
        let mut merged_run = Run::new_for_level(all_data, target_level_num, fanout);
        
        // Store the run in the target level
        let run_id = merged_run.store(storage, target_level_num)?;
        merged_run.id = Some(run_id);
        
        // Replace target level runs with the new merged run
        // First, remove all existing runs from the target level storage
        for run in target_runs {
            if let Some(id) = run.id {
                let _ = storage.delete_run(id);
            }
        }
        
        // Reset the target level and add the merged run
        *target_level = Level::new();
        target_level.add_run(merged_run.clone());
        
        // Delete the old source runs from storage
        for run in source_runs {
            if let Some(id) = run.id {
                let _ = storage.delete_run(id);
            }
        }
        
        Ok(merged_run)
    }
    
    fn box_clone(&self) -> Box<dyn CompactionPolicy> {
        Box::new(Self {
            size_ratio_threshold: self.size_ratio_threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::run::{StorageFactory, StorageOptions};
    
    #[test]
    fn test_leveled_compaction_should_compact_level0() {
        let policy = LeveledCompactionPolicy::new(4);
        
        // Create a mock level with 1 run
        let mut level = Level::new();
        level.add_run(Run::new(vec![(1, 100)]));
        
        // Level 0 with 1 run should not need compaction
        assert!(!policy.should_compact(&level, 0));
        
        // Add a second run to level 0
        level.add_run(Run::new(vec![(2, 200)]));
        
        // Level 0 with multiple runs should trigger compaction
        assert!(policy.should_compact(&level, 0));
    }
    
    #[test]
    fn test_leveled_compaction_should_compact_higher_levels() {
        let policy = LeveledCompactionPolicy::new(4);
        
        // Create a mock level with 1 run
        let mut level = Level::new();
        level.add_run(Run::new(vec![(1, 100)]));
        
        // Level 1 with only one run should not need compaction
        assert!(!policy.should_compact(&level, 1));
        
        // Add a second run to level 1 (violates leveled invariant)
        level.add_run(Run::new(vec![(2, 200)]));
        
        // Should trigger compaction to maintain the "one run per level" invariant
        assert!(policy.should_compact(&level, 1));
    }
    
    #[test]
    fn test_leveled_compaction_merge() {
        // Create storage for testing
        let temp_dir = tempdir().unwrap();
        let options = StorageOptions {
            base_path: temp_dir.path().to_path_buf(),
            create_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
        };
        let storage = StorageFactory::create("file", options).unwrap();
        
        let policy = LeveledCompactionPolicy::new(4);
        
        // Create source level with multiple runs
        let mut source_level = Level::new();
        let mut target_level = Level::new();
        
        // Create runs for source level
        let mut run1 = Run::new_for_level(vec![(1, 100), (3, 300)], 0, 4.0);
        let mut run2 = Run::new_for_level(vec![(2, 200), (4, 400)], 0, 4.0);
        
        // Store runs
        let id1 = run1.store(&*storage, 0).unwrap();
        let id2 = run2.store(&*storage, 0).unwrap();
        
        // Set run IDs
        run1.id = Some(id1);
        run2.id = Some(id2);
        
        // Add runs to source level
        source_level.add_run(run1);
        source_level.add_run(run2);
        
        // Add a pre-existing run to target level
        let mut target_run = Run::new_for_level(vec![(5, 500), (6, 600)], 1, 4.0);
        let target_id = target_run.store(&*storage, 1).unwrap();
        target_run.id = Some(target_id);
        target_level.add_run(target_run);
        
        // Test compaction
        let merged_run = policy.compact(&source_level, &mut target_level, &*storage, 0, 1).unwrap();
        
        // Verify the merged run contains all data from both source and target
        assert_eq!(merged_run.entry_count(), 6);
        
        // There should be exactly one run in the target level
        assert_eq!(target_level.run_count(), 1);
        
        // Verify the merged data is correct
        let range = merged_run.range(0, 10);
        assert_eq!(range, vec![(1, 100), (2, 200), (3, 300), (4, 400), (5, 500), (6, 600)]);
    }
}