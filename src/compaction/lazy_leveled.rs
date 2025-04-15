use crate::compaction::CompactionPolicy;
use crate::level::Level;
use crate::run::{Run, RunStorage};
use crate::types::{Result, Error};

/// LazyLeveled compaction policy combines aspects of both tiered and leveled compaction.
/// It accumulates runs at level 0 like tiered compaction, but maintains a single
/// run per level for all other levels like leveled compaction.
pub struct LazyLeveledCompactionPolicy {
    /// Threshold for number of runs in level 0 before compaction
    run_threshold: usize,
}

impl LazyLeveledCompactionPolicy {
    /// Create a new lazy leveled compaction policy with the specified threshold
    pub fn new(run_threshold: usize) -> Self {
        Self { run_threshold }
    }
}

impl CompactionPolicy for LazyLeveledCompactionPolicy {
    fn should_compact(&self, level: &Level, level_num: usize) -> bool {
        if level_num == 0 {
            // Use tiered approach for level 0 - compact when run count exceeds threshold
            return level.run_count() >= self.run_threshold;
        } else {
            // Use leveled approach for other levels - compact if more than one run
            return level.run_count() > 1;
        }
    }
    
    fn select_runs_to_compact(&self, level: &Level) -> Vec<usize> {
        // For all levels, select all runs for compaction
        (0..level.run_count()).collect()
    }
    
    fn compact(
        &self,
        source_level: &Level,
        target_level: &mut Level,
        storage: &dyn RunStorage,
        source_level_num: usize,
        target_level_num: usize,
        config: Option<&crate::lsm_tree::LSMConfig>,
    ) -> Result<Run> {
        // Select all runs from source level
        let run_indices = self.select_runs_to_compact(source_level);
        
        if run_indices.is_empty() {
            return Err(Error::CompactionError);
        }
        
        // Collect all runs to compact
        let mut source_runs = Vec::new();
        for idx in run_indices {
            source_runs.push(source_level.get_run(idx));
        }
        
        // Determine target runs based on the level
        let mut target_runs = Vec::new();
        
        // For all levels, we want to maintain a single run per level
        // So include any existing runs in the target level
        if target_level_num > 0 || source_level_num > 0 {
            for idx in 0..target_level.run_count() {
                target_runs.push(target_level.get_run(idx));
            }
        }
        
        // Merge all data from source and target runs
        let mut all_data = Vec::new();
        
        // Add data from source runs
        for run in &source_runs {
            if let (Some(min_key), Some(max_key)) = (run.min_key(), run.max_key()) {
                all_data.extend(run.range(min_key, max_key + 1));
            }
        }
        
        // Add data from target runs (if any)
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
        let fanout = config.map(|c| c.fanout as f64).unwrap_or(4.0);
        let mut merged_run = Run::new_for_level(all_data, target_level_num, fanout, config);
        
        // Store the run in the target level
        let run_id = merged_run.store(storage, target_level_num)?;
        merged_run.id = Some(run_id);
        
        // If target level is not level 0, ensure we maintain the "one run per level" invariant
        if target_level_num > 0 {
            // First, remove all existing runs from the target level storage
            for run in target_runs {
                if let Some(id) = run.id {
                    let _ = storage.delete_run(id);
                }
            }
            
            // Reset the target level and add the merged run
            *target_level = Level::new();
            target_level.add_run(merged_run.clone());
        } else {
            // For level 0, just add the merged run, maintaining tiered behavior
            target_level.add_run(merged_run.clone());
        }
        
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
            run_threshold: self.run_threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::run::{StorageFactory, StorageOptions};
    use crate::types::StorageType;
    
    #[test]
    fn test_lazy_leveled_should_compact_level0() {
        let policy = LazyLeveledCompactionPolicy::new(3);
        
        // Create a mock level with 2 runs
        let mut level = Level::new();
        level.add_run(Run::new(vec![(1, 100)]));
        level.add_run(Run::new(vec![(2, 200)]));
        
        // Level 0 with 2 runs should not trigger compaction yet
        assert!(!policy.should_compact(&level, 0));
        
        // Add one more run to reach threshold
        level.add_run(Run::new(vec![(3, 300)]));
        
        // Level 0 with 3 runs should trigger compaction
        assert!(policy.should_compact(&level, 0));
    }
    
    #[test]
    fn test_lazy_leveled_should_compact_higher_levels() {
        let policy = LazyLeveledCompactionPolicy::new(3);
        
        // Create a mock level with 1 run
        let mut level = Level::new();
        level.add_run(Run::new(vec![(1, 100)]));
        
        // Level 1 with only one run should not need compaction
        assert!(!policy.should_compact(&level, 1));
        
        // Add a second run to level 1
        level.add_run(Run::new(vec![(2, 200)]));
        
        // Level 1 with 2 runs should trigger compaction (leveled invariant)
        assert!(policy.should_compact(&level, 1));
    }
    
    #[test]
    fn test_lazy_leveled_compaction_level0() {
        // Create storage for testing
        let temp_dir = tempdir().unwrap();
        let options = StorageOptions {
            base_path: temp_dir.path().to_path_buf(),
            create_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
        };
        let storage = StorageFactory::create_from_type(StorageType::File, options).unwrap();
        
        let policy = LazyLeveledCompactionPolicy::new(3);
        
        // Create source level with multiple runs (level 0)
        let mut source_level = Level::new();
        let mut target_level = Level::new();
        
        // Create runs for source level
        let mut run1 = Run::new_for_level(vec![(1, 100), (3, 300)], 0, 4.0, None);
        let mut run2 = Run::new_for_level(vec![(2, 200), (4, 400)], 0, 4.0, None);
        let mut run3 = Run::new_for_level(vec![(5, 500), (6, 600)], 0, 4.0, None);
        
        // Store runs
        let id1 = run1.store(&*storage, 0).unwrap();
        let id2 = run2.store(&*storage, 0).unwrap();
        let id3 = run3.store(&*storage, 0).unwrap();
        
        // Set run IDs
        run1.id = Some(id1);
        run2.id = Some(id2);
        run3.id = Some(id3);
        
        // Add runs to source level
        source_level.add_run(run1);
        source_level.add_run(run2);
        source_level.add_run(run3);
        
        // Add an existing run to target level (level 1)
        let mut target_run = Run::new_for_level(vec![(7, 700), (8, 800)], 1, 4.0, None);
        let target_id = target_run.store(&*storage, 1).unwrap();
        target_run.id = Some(target_id);
        target_level.add_run(target_run);
        
        // Test compaction from level 0 to level 1
        let merged_run = policy.compact(&source_level, &mut target_level, &*storage, 0, 1, None).unwrap();
        
        // Verify merged run contains all data
        assert_eq!(merged_run.entry_count(), 8);
        
        // For level 1, there should be exactly one run (leveled behavior)
        assert_eq!(target_level.run_count(), 1);
        
        // Verify merged data is correct
        let range = merged_run.range(0, 10);
        assert_eq!(range, vec![(1, 100), (2, 200), (3, 300), (4, 400), (5, 500), (6, 600), (7, 700), (8, 800)]);
    }
    
    #[test]
    fn test_lazy_leveled_compaction_between_higher_levels() {
        // Create storage for testing
        let temp_dir = tempdir().unwrap();
        let options = StorageOptions {
            base_path: temp_dir.path().to_path_buf(),
            create_if_missing: true,
            max_open_files: 100,
            sync_writes: false,
        };
        let storage = StorageFactory::create_from_type(StorageType::File, options).unwrap();
        
        let policy = LazyLeveledCompactionPolicy::new(3);
        
        // Create source level with multiple runs (level 1)
        let mut source_level = Level::new();
        let mut target_level = Level::new();
        
        // Create runs for source level
        let mut run1 = Run::new_for_level(vec![(1, 100), (3, 300)], 1, 4.0, None);
        let mut run2 = Run::new_for_level(vec![(2, 200), (4, 400)], 1, 4.0, None);
        
        // Store runs
        let id1 = run1.store(&*storage, 1).unwrap();
        let id2 = run2.store(&*storage, 1).unwrap();
        
        // Set run IDs
        run1.id = Some(id1);
        run2.id = Some(id2);
        
        // Add runs to source level (level 1)
        source_level.add_run(run1);
        source_level.add_run(run2);
        
        // Add an existing run to target level (level 2)
        let mut target_run = Run::new_for_level(vec![(5, 500), (6, 600)], 2, 4.0, None);
        let target_id = target_run.store(&*storage, 2).unwrap();
        target_run.id = Some(target_id);
        target_level.add_run(target_run);
        
        // Test compaction from level 1 to level 2
        let merged_run = policy.compact(&source_level, &mut target_level, &*storage, 1, 2, None).unwrap();
        
        // Verify merged run contains all data
        assert_eq!(merged_run.entry_count(), 6);
        
        // For level 2, there should be exactly one run (leveled behavior)
        assert_eq!(target_level.run_count(), 1);
        
        // Verify merged data is correct
        let range = merged_run.range(0, 10);
        assert_eq!(range, vec![(1, 100), (2, 200), (3, 300), (4, 400), (5, 500), (6, 600)]);
    }
}