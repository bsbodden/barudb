use crate::compaction::CompactionPolicy;
use crate::level::Level;
use crate::run::{self, Run, RunStorage};
use crate::types::{Result, Error, Key};

/// Partial Tiered compaction policy merges a subset of runs in a level when the level has
/// reached a run threshold. This reduces I/O costs for large levels compared to standard
/// tiered compaction.
pub struct PartialTieredCompactionPolicy {
    /// Number of runs that trigger compaction
    run_threshold: usize,
    /// Maximum number of runs to compact at once (partial compaction)
    max_runs_per_compaction: usize,
    /// Strategy for selecting runs to compact
    selection_strategy: SelectionStrategy,
}

/// Strategy for selecting which runs to compact when using partial compaction
#[derive(Clone, Copy)]
pub enum SelectionStrategy {
    /// Select the oldest runs first (those with smallest sequence numbers)
    Oldest,
    /// Select runs with the most overlapping key ranges
    MostOverlap,
    /// Select runs with the smallest total size
    SmallestSize,
}

impl PartialTieredCompactionPolicy {
    /// Create a new partial tiered compaction policy with the specified parameters
    pub fn new(run_threshold: usize, max_runs_per_compaction: usize, strategy: SelectionStrategy) -> Self {
        Self {
            run_threshold,
            max_runs_per_compaction,
            selection_strategy: strategy,
        }
    }
    
    /// Create a new partial tiered compaction policy with default parameters
    pub fn with_defaults(run_threshold: usize) -> Self {
        Self {
            run_threshold,
            max_runs_per_compaction: 3, // Default to compacting 3 runs at a time
            selection_strategy: SelectionStrategy::Oldest, // Default to oldest first
        }
    }
    
    /// Find runs with overlapping key ranges
    fn find_overlapping_runs(&self, level: &Level) -> Vec<usize> {
        let runs = level.get_runs();
        if runs.len() <= 1 {
            return vec![];
        }
        
        // Create a structure to store run information for overlap calculation
        let mut run_info: Vec<(usize, Option<Key>, Option<Key>)> = Vec::new();
        for (idx, run) in runs.iter().enumerate() {
            run_info.push((idx, run.min_key(), run.max_key()));
        }
        
        // Sort run_info by min_key to find adjacent runs
        run_info.sort_by(|a, b| {
            match (a.1, b.1) {
                (Some(a_min), Some(b_min)) => a_min.cmp(&b_min),
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, None) => std::cmp::Ordering::Equal,
            }
        });
        
        // Find the group of 'max_runs_per_compaction' consecutive runs with most overlap
        if run_info.len() <= self.max_runs_per_compaction {
            // If we have fewer runs than the max, return them all
            return run_info.into_iter().map(|(idx, _, _)| idx).collect();
        }
        
        // Calculate overlap for each possible group
        let mut best_group = Vec::new();
        let mut max_overlap = 0;
        
        for i in 0..=(run_info.len() - self.max_runs_per_compaction) {
            let group = &run_info[i..(i + self.max_runs_per_compaction)];
            
            // Find the overall min and max keys in this group
            let min_key = group.iter()
                .filter_map(|(_, min_key, _)| *min_key)
                .min();
            
            let max_key = group.iter()
                .filter_map(|(_, _, max_key)| *max_key)
                .max();
                
            if let (Some(min), Some(max)) = (min_key, max_key) {
                // Estimate overlap by the range size
                let overlap = max - min;
                
                if best_group.is_empty() || overlap > max_overlap {
                    max_overlap = overlap;
                    best_group = group.iter().map(|(idx, _, _)| *idx).collect();
                }
            }
        }
        
        if best_group.is_empty() {
            // Fallback to oldest runs if we couldn't calculate overlap
            run_info.into_iter()
                .map(|(idx, _, _)| idx)
                .take(self.max_runs_per_compaction)
                .collect()
        } else {
            best_group
        }
    }
}

impl CompactionPolicy for PartialTieredCompactionPolicy {
    fn should_compact(&self, level: &Level, _level_num: usize) -> bool {
        // Compact when number of runs reaches or exceeds threshold
        level.run_count() >= self.run_threshold
    }
    
    fn select_runs_to_compact(&self, level: &Level) -> Vec<usize> {
        if level.run_count() < 2 {
            return vec![];
        }
        
        let run_count = level.run_count();
        let runs_to_compact = std::cmp::min(self.max_runs_per_compaction, run_count);
        
        match self.selection_strategy {
            SelectionStrategy::Oldest => {
                // Simply select the first N runs (oldest)
                (0..runs_to_compact).collect()
            },
            SelectionStrategy::MostOverlap => {
                // Find runs with the most overlapping key ranges
                self.find_overlapping_runs(level)
            },
            SelectionStrategy::SmallestSize => {
                // Select runs with the smallest total size
                // For simplicity, we'll use entry count as a proxy for size
                let runs = level.get_runs();
                
                // Create a vector of (index, entry_count) pairs
                let mut runs_with_size: Vec<(usize, usize)> = runs.iter()
                    .enumerate()
                    .map(|(idx, run)| (idx, run.entry_count()))
                    .collect();
                
                // Sort by entry count (ascending)
                runs_with_size.sort_by_key(|&(_, size)| size);
                
                // Take the N smallest runs
                runs_with_size.into_iter()
                    .take(runs_to_compact)
                    .map(|(idx, _)| idx)
                    .collect()
            }
        }
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
        // Check if there's anything to compact
        if source_level.run_count() < 2 {
            // Create an empty run to satisfy the return type
            let mut empty_run = Run::new_for_level(vec![], target_level_num, 4.0, config);
            let run_id = empty_run.store(storage, target_level_num)?;
            let mut result_run = empty_run.clone();
            result_run.id = Some(run_id);
            return Ok(result_run);
        }
        
        // Select runs to compact
        let mut run_indices = self.select_runs_to_compact(source_level);
        
        // Make sure we have runs to compact
        if run_indices.is_empty() {
            // If no runs were selected, default to first 2 runs
            let count = std::cmp::min(2, source_level.run_count());
            run_indices = (0..count).collect();
            
            // If still empty, return an empty run
            if run_indices.is_empty() {
                let mut empty_run = Run::new_for_level(vec![], target_level_num, 4.0, config);
                let run_id = empty_run.store(storage, target_level_num)?;
                let mut result_run = empty_run.clone();
                result_run.id = Some(run_id);
                return Ok(result_run);
            }
        }
        
        // Collect selected runs to compact
        let mut runs = Vec::new();
        for idx in &run_indices {
            runs.push(source_level.get_run(*idx));
        }
        
        if runs.is_empty() {
            return Err(Error::CompactionError);
        }
        
        // Merge all key-value pairs from selected runs
        let mut all_data = Vec::new();
        for run in &runs {
            if let Some(min_key) = run.min_key() {
                if let Some(max_key) = run.max_key() {
                    // Use storage-aware range method if run has ID
                    if run.id.is_some() {
                        // Clear any block cache entries for this run before reading
                        // to ensure we get the latest data
                        if let Some(run_id) = run.id {
                            // Use any downcast available to invalidate cache
                            if let Some(file_storage) = storage.as_any().downcast_ref::<run::FileStorage>() {
                                let _ = file_storage.get_cache().invalidate_run(run_id);
                            }
                        }
                        all_data.extend(run.range_with_storage(min_key, max_key + 1, storage));
                    } else {
                        // Fall back to in-memory range method for runs without ID
                        all_data.extend(run.range(min_key, max_key + 1));
                    }
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
        
        // Only delete the specific runs we've compacted - other runs in the source level remain
        for run in runs {
            if let Some(id) = run.id {
                // Ignore errors during cleanup
                let _ = storage.delete_run(id);
                
                // Also invalidate any cache entries for this run since it's being deleted
                if let Some(file_storage) = storage.as_any().downcast_ref::<run::FileStorage>() {
                    let _ = file_storage.get_cache().invalidate_run(id);
                }
            }
        }
        
        Ok(merged_run)
    }
    
    fn box_clone(&self) -> Box<dyn CompactionPolicy> {
        Box::new(Self {
            run_threshold: self.run_threshold,
            max_runs_per_compaction: self.max_runs_per_compaction,
            selection_strategy: self.selection_strategy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_partial_tiered_policy_should_compact() {
        let policy = PartialTieredCompactionPolicy::with_defaults(3);
        
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
    fn test_partial_tiered_policy_select_runs_oldest() {
        let policy = PartialTieredCompactionPolicy::new(
            3, 2, SelectionStrategy::Oldest
        );
        
        // Create a mock level with 4 runs
        let mut level = Level::new();
        level.add_run(Run::new(vec![(1, 100)]));
        level.add_run(Run::new(vec![(2, 200)]));
        level.add_run(Run::new(vec![(3, 300)]));
        level.add_run(Run::new(vec![(4, 400)]));
        
        // Should select first 2 runs (oldest)
        let selected = policy.select_runs_to_compact(&level);
        assert_eq!(selected, vec![0, 1]);
    }
    
    #[test]
    fn test_partial_tiered_policy_select_runs_smallest() {
        let policy = PartialTieredCompactionPolicy::new(
            3, 2, SelectionStrategy::SmallestSize
        );
        
        // Create a mock level with runs of different sizes
        let mut level = Level::new();
        // Run with 3 entries
        level.add_run(Run::new(vec![(1, 100), (2, 200), (3, 300)]));
        // Run with 1 entry
        level.add_run(Run::new(vec![(4, 400)]));
        // Run with 2 entries
        level.add_run(Run::new(vec![(5, 500), (6, 600)]));
        // Run with 4 entries
        level.add_run(Run::new(vec![(7, 700), (8, 800), (9, 900), (10, 1000)]));
        
        // Should select the 2 smallest runs (indices 1 and 2)
        let selected = policy.select_runs_to_compact(&level);
        assert!(selected.contains(&1)); // Run with 1 entry
        assert!(selected.contains(&2)); // Run with 2 entries
    }
}