use crate::run::Run;
use crate::types::{Key, Value};

#[derive(Clone)]
pub struct Level {
    runs: Vec<Run>,
}

impl Level {
    pub fn new() -> Self {
        Level { runs: Vec::new() }
    }

    // Add a new run to this level
    pub fn add_run(&mut self, run: Run) {
        self.runs.push(run);
    }

    // Get the number of runs in this level
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }
    
    // Get a run by index
    pub fn get_run(&self, index: usize) -> Run {
        self.runs[index].clone()
    }
    
    // Get reference to all runs in this level
    pub fn get_runs(&self) -> &[Run] {
        &self.runs
    }
    
    // Remove a run by index
    pub fn remove_run(&mut self, index: usize) -> Run {
        self.runs.remove(index)
    }

    // Retrieve a value for a key by searching all runs
    pub fn get(&self, key: Key) -> Option<Value> {
        for run in &self.runs {
            if let Some(value) = run.get(key) {
                return Some(value);
            }
        }
        None
    }

    // Retrieve all key-value pairs in the specified range
    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        let mut results = Vec::new();
        for run in &self.runs {
            results.extend(run.range(start, end));
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::Run;

    #[test]
    fn test_level_operations() {
        let mut level = Level::new();
        let data1 = vec![(1, 100), (2, 200)];
        let data2 = vec![(3, 300), (4, 400)];

        // Add runs to the level
        level.add_run(Run::new(data1));
        level.add_run(Run::new(data2));

        // Test key lookups
        assert_eq!(level.get(2), Some(200));
        assert_eq!(level.get(4), Some(400));
        assert_eq!(level.get(5), None);

        // Test range queries
        let range = level.range(2, 4);
        assert_eq!(range, vec![(2, 200), (3, 300)]);
    }
}
