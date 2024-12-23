use crate::run::Run;
use crate::types::{Key, Value};

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
