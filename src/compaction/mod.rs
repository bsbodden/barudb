mod tiered;

use crate::level::Level;
use crate::run::{Run, RunStorage};
use crate::types::{Result, CompactionPolicyType};

pub use tiered::TieredCompactionPolicy;

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
    pub fn create_from_type(policy_type: CompactionPolicyType, run_threshold: usize) -> Result<Box<dyn CompactionPolicy>> {
        match policy_type {
            CompactionPolicyType::Tiered => Ok(Box::new(TieredCompactionPolicy::new(run_threshold))),
        }
    }
    
    /// Create a new compaction policy by name (legacy method)
    pub fn create(name: &str, run_threshold: usize) -> Result<Box<dyn CompactionPolicy>> {
        match CompactionPolicyType::from_str(name) {
            Some(policy_type) => Self::create_from_type(policy_type, run_threshold),
            None => Err(crate::types::Error::CompactionError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compaction_factory() {
        // Test creating tiered policy using enum
        let policy = CompactionFactory::create_from_type(CompactionPolicyType::Tiered, 3).unwrap();
        assert!(policy.should_compact(&{
            let mut level = Level::new();
            level.add_run(Run::new(vec![(1, 100)]));
            level.add_run(Run::new(vec![(2, 200)]));
            level.add_run(Run::new(vec![(3, 300)]));
            level
        }, 0));
        
        // Test legacy string-based creation
        let _legacy_policy = CompactionFactory::create("tiered", 3).unwrap();
        
        // Test invalid policy name
        let result = CompactionFactory::create("invalid", 3);
        assert!(result.is_err());
    }
}