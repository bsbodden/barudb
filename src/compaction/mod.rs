mod tiered;
mod leveled;
mod lazy_leveled;
mod partial_tiered;

use crate::level::Level;
use crate::run::{Run, RunStorage};
use crate::types::{Result, CompactionPolicyType};

pub use tiered::TieredCompactionPolicy;
pub use leveled::LeveledCompactionPolicy;
pub use lazy_leveled::LazyLeveledCompactionPolicy;
pub use partial_tiered::{PartialTieredCompactionPolicy, SelectionStrategy};

/// Trait for compaction policy implementations
pub trait CompactionPolicy: Send + Sync {
    /// Checks if a level should be compacted based on policy-specific criteria
    fn should_compact(&self, level: &Level, level_num: usize) -> bool;
    
    /// Selects which runs should be compacted from a level
    fn select_runs_to_compact(&self, level: &Level) -> Vec<usize>;
    
    /// Performs compaction, merging selected runs from source level into target level
    /// 
    /// # Arguments
    /// * `source_level` - The level containing runs to compact
    /// * `target_level` - The level where the compacted run will be placed
    /// * `storage` - Storage implementation for reading/writing runs
    /// * `source_level_num` - Level number of the source level
    /// * `target_level_num` - Level number of the target level
    /// 
    /// # Returns
    /// * `Result<Run>` - The newly created run in the target level
    fn compact(
        &self,
        source_level: &Level,
        target_level: &mut Level,
        storage: &dyn RunStorage,
        source_level_num: usize,
        target_level_num: usize,
    ) -> Result<Run>;
    
    /// Creates a clone of this policy
    fn box_clone(&self) -> Box<dyn CompactionPolicy>;
}

impl Clone for Box<dyn CompactionPolicy> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// Factory for creating compaction policies
pub struct CompactionFactory;

impl CompactionFactory {
    /// Create a new compaction policy by policy type
    pub fn create_from_type(policy_type: CompactionPolicyType, threshold: usize) -> Result<Box<dyn CompactionPolicy>> {
        match policy_type {
            CompactionPolicyType::Tiered => Ok(Box::new(TieredCompactionPolicy::new(threshold))),
            CompactionPolicyType::Leveled => Ok(Box::new(LeveledCompactionPolicy::new(threshold))),
            CompactionPolicyType::LazyLeveled => Ok(Box::new(LazyLeveledCompactionPolicy::new(threshold))),
            CompactionPolicyType::PartialTiered => Ok(Box::new(PartialTieredCompactionPolicy::with_defaults(threshold))),
        }
    }
    
    /// Create a new compaction policy by name (legacy method)
    pub fn create(name: &str, threshold: usize) -> Result<Box<dyn CompactionPolicy>> {
        match CompactionPolicyType::from_str(name) {
            Some(policy_type) => Self::create_from_type(policy_type, threshold),
            None => Err(crate::types::Error::CompactionError),
        }
    }
    
    /// Create a new partial tiered compaction policy with custom parameters
    pub fn create_partial_tiered(
        threshold: usize,
        max_runs_per_compaction: usize,
        selection_strategy: SelectionStrategy
    ) -> Box<dyn CompactionPolicy> {
        Box::new(PartialTieredCompactionPolicy::new(
            threshold,
            max_runs_per_compaction,
            selection_strategy
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compaction_factory() {
        // Create a test level with 3 runs
        let mut test_level = Level::new();
        test_level.add_run(Run::new(vec![(1, 100)]));
        test_level.add_run(Run::new(vec![(2, 200)]));
        test_level.add_run(Run::new(vec![(3, 300)]));
        
        // Test creating tiered policy using enum
        let tiered_policy = CompactionFactory::create_from_type(CompactionPolicyType::Tiered, 3).unwrap();
        assert!(tiered_policy.should_compact(&test_level, 0));
        
        // Test creating leveled policy using enum
        let leveled_policy = CompactionFactory::create_from_type(CompactionPolicyType::Leveled, 4).unwrap();
        assert!(leveled_policy.should_compact(&test_level, 0));
        
        // Test creating lazy leveled policy using enum
        let lazy_leveled_policy = CompactionFactory::create_from_type(CompactionPolicyType::LazyLeveled, 3).unwrap();
        assert!(lazy_leveled_policy.should_compact(&test_level, 0));
        
        // Test level 1 behavior (level > 0)
        let mut level1 = Level::new();
        level1.add_run(Run::new(vec![(1, 100)]));
        level1.add_run(Run::new(vec![(2, 200)]));
        
        // All policies should compact level 1 when it has multiple runs
        assert!(leveled_policy.should_compact(&level1, 1));
        assert!(lazy_leveled_policy.should_compact(&level1, 1));
        
        // But tiered would only compact if it reaches threshold
        assert!(!tiered_policy.should_compact(&level1, 1));
        
        // Test legacy string-based creation
        let _legacy_tiered_policy = CompactionFactory::create("tiered", 3).unwrap();
        let _legacy_leveled_policy = CompactionFactory::create("leveled", 4).unwrap();
        let _legacy_lazy_leveled_policy = CompactionFactory::create("lazy_leveled", 3).unwrap();
        
        // Test invalid policy name
        let result = CompactionFactory::create("invalid", 3);
        assert!(result.is_err());
    }
}