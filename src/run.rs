use crate::types::{Key, Value};

pub struct Run {
    data: Vec<(Key, Value)>,
}

impl Run {
    pub fn new(data: Vec<(Key, Value)>) -> Self {
        Run { data }
    }

    pub fn get(&self, key: Key) -> Option<Value> {
        self.data.iter().find(|&&(k, _)| k == key).map(|&(_, v)| v)
    }

    pub fn range(&self, start: Key, end: Key) -> Vec<(Key, Value)> {
        self.data
            .iter()
            .filter(|&&(k, _)| k >= start && k < end)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_operations() {
        let data = vec![(1, 100), (2, 200), (3, 300)];
        let run = Run::new(data);

        assert_eq!(run.get(2), Some(200));
        assert_eq!(run.get(4), None);

        let range = run.range(1, 3);
        assert_eq!(range, vec![(1, 100), (2, 200)]);
    }
}

