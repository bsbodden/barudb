# Compaction Policies in LSM Tree

## Overview

Compaction is a critical process in LSM trees that merges multiple runs to maintain performance and minimize space usage. This document describes the implementation of three different compaction policies in our LSM tree: Tiered, Leveled, and Lazy Leveled.

## Compaction Policy Interface

All compaction policies implement a common interface defined by the `CompactionPolicy` trait:

```rust
pub trait CompactionPolicy: Send + Sync {
    /// Determines if a level should be compacted
    fn should_compact(&self, level: &Level, level_num: usize) -> bool;
    
    /// Selects runs to compact from a level
    fn select_runs_to_compact(&self, level: &Level, level_num: usize) -> Vec<usize>;
    
    /// Performs the compaction operation
    fn compact(
        &self,
        source_level: &Level,
        target_level: &mut Level,
        storage: &dyn Storage,
        source_level_num: usize,
        target_level_num: usize,
    ) -> Result<()>;
}
```

## Implemented Compaction Policies

### 1. Tiered Compaction Policy

The `TieredCompactionPolicy` allows multiple runs per level and compacts when a threshold is reached:

```rust
pub struct TieredCompactionPolicy {
    /// Number of runs that trigger compaction
    run_threshold: usize,
}

impl CompactionPolicy for TieredCompactionPolicy {
    fn should_compact(&self, level: &Level, level_num: usize) -> bool {
        // Compact when run count reaches or exceeds threshold
        level.run_count() >= self.run_threshold
    }
    
    // Implementation details for selecting runs and performing compaction...
}
```

**Key Characteristics:**
- **Trigger**: Compaction occurs when the number of runs in a level reaches or exceeds a configurable threshold
- **Selection**: All runs in the level are selected for compaction
- **Merge Process**: All key-value pairs from the selected runs are merged, sorted, and duplicates are removed
- **Result**: A single new run is created and stored in the target level

**Performance Characteristics:**
- Lower write amplification (less frequent compaction)
- Higher space amplification (allows multiple overlapping runs)
- Higher read cost (must check multiple runs per level)

### 2. Leveled Compaction Policy

The `LeveledCompactionPolicy` maintains a single run per level:

```rust
pub struct LeveledCompactionPolicy {
    /// Size ratio threshold between levels (usually matches the fanout)
    size_ratio_threshold: usize,
}

impl CompactionPolicy for LeveledCompactionPolicy {
    fn should_compact(&self, level: &Level, level_num: usize) -> bool {
        // For level 0, compact when there's more than one run
        if level_num == 0 {
            return level.run_count() > 1;
        }
        
        // For other levels, check if there are multiple runs (violates the leveled invariant)
        level.run_count() > 1
    }
    
    // Additional implementation details...
}
```

**Key Characteristics:**
- **Trigger**: Compaction occurs when there is more than one run in any level
- **Selection**: All runs in the level are selected for compaction
- **Merge Process**: All key-value pairs from the selected runs are merged with any existing runs in the target level
- **Result**: A single new run is created and replaces all runs in the target level

**Performance Characteristics:**
- Higher write amplification (more frequent compaction)
- Lower space amplification (single run per level)
- Lower read cost (only one run to check per level)

### 3. Lazy Leveled Compaction Policy

The `LazyLeveledCompactionPolicy` is a hybrid approach that combines aspects of tiering and leveling:

```rust
pub struct LazyLeveledCompactionPolicy {
    /// Threshold for number of runs in level 0 before compaction
    run_threshold: usize,
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
    
    // Additional implementation details...
}
```

**Key Characteristics:**
- **Trigger for Level 0**: Like tiered compaction, triggers when run count reaches threshold
- **Trigger for Other Levels**: Like leveled compaction, triggers when there is more than one run
- **Selection**: All runs in the level are selected for compaction
- **Merge Process**: Merges all runs and maintains the single run invariant for levels > 0
- **Result**: For L0->L1 compactions, creates a sorted run in L1; for other levels, ensures a single run per level

**Performance Characteristics:**
- Balances write and read costs
- Better for mixed workloads
- Adaptive to workload characteristics

## Integration with the LSM Tree

The compaction policy is integrated with the LSM tree through the following mechanisms:

### Creation During Initialization

When creating an LSM tree, the appropriate compaction policy is instantiated based on configuration:

```rust
let compaction_policy = CompactionFactory::create(
    &config.compaction_policy, 
    config.compaction_threshold
).expect("Failed to create compaction policy");
```

### Checking for Compaction

After buffer flushes and other operations, the tree checks if compaction is needed:

```rust
pub fn compact_if_needed(&mut self) -> Result<()> {
    // Check each level for compaction needs
    for level_num in 0..self.levels.len() {
        if self.compaction_policy.should_compact(&self.levels[level_num], level_num) {
            self.compact_level(level_num)?;
        }
    }
    
    Ok(())
}
```

### Performing Compaction

The tree delegates the compaction logic to the policy:

```rust
pub fn compact_level(&mut self, level_num: usize) -> Result<()> {
    // Ensure we have a level to compact into
    let target_level_num = level_num + 1;
    while self.levels.len() <= target_level_num {
        self.levels.push(Level::new());
    }
    
    // Clone source level to avoid borrow issues
    let source_level = self.levels[level_num].clone();
    
    // Perform compaction
    match self.compaction_policy.compact(
        &source_level,
        &mut self.levels[target_level_num],
        &*self.storage,
        level_num,
        target_level_num,
    ) {
        Ok(_) => {
            // Replace the old source level with a new empty level
            self.levels[level_num] = Level::new();
            
            // Check if next level now needs compaction
            if target_level_num < self.levels.len() &&
               self.compaction_policy.should_compact(&self.levels[target_level_num], target_level_num) {
                // Recursively compact next level
                self.compact_level(target_level_num)?;
            }
            
            Ok(())
        },
        Err(e) => Err(e),
    }
}
```

## Compaction Process

The compaction process involves several stages:

1. **Selection**: Identify runs to be compacted
2. **Merging**: Combine key-value pairs from all runs, with newer values taking precedence
3. **Deduplication**: Remove duplicate keys, keeping only the most recent value
4. **Tombstone Processing**: Remove keys with tombstone values if they no longer need to be preserved
5. **Run Creation**: Create new run(s) with the resulting key-value pairs
6. **Placement**: Place new run(s) in the target level

## Performance Comparison

Different compaction policies exhibit different performance characteristics under various workloads:

### Write-Heavy Workloads
- **Tiered**: Best performance due to lower write amplification
- **Lazy Leveled**: Good performance, especially when L0 accumulates many runs
- **Leveled**: More frequent compactions lead to higher write amplification

### Read-Heavy Workloads
- **Leveled**: Best performance due to single run per level
- **Lazy Leveled**: Good performance after L0
- **Tiered**: Worst performance due to multiple runs per level

### Mixed Workloads
- **Lazy Leveled**: Best overall balance
- **Tiered**: Good for write-dominated mixed workloads
- **Leveled**: Good for read-dominated mixed workloads

## Future Enhancements

While the current implementation satisfies the project requirements, several potential enhancements could further improve compaction performance:

### 1. Partial Compaction

Implement partial compaction for large levels to reduce I/O costs:

```rust
pub struct PartialCompactionPolicy {
    run_threshold: usize,
    // Maximum size to compact in a single operation
    max_compaction_size: usize,
}

impl CompactionPolicy for PartialCompactionPolicy {
    fn select_runs_to_compact(&self, level: &Level, level_num: usize) -> Vec<usize> {
        // Select runs up to max_compaction_size
        let mut selected = Vec::new();
        let mut total_size = 0;
        
        for (i, run) in level.runs().iter().enumerate() {
            if total_size + run.size() <= self.max_compaction_size {
                selected.push(i);
                total_size += run.size();
            } else {
                break;
            }
        }
        
        selected
    }
    
    // Additional implementation...
}
```

### 2. Background Compaction

Implement compaction in a separate thread to avoid blocking writes:

```rust
pub struct BackgroundCompactionManager {
    compaction_queue: Arc<Mutex<VecDeque<CompactionTask>>>,
    worker_thread: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl BackgroundCompactionManager {
    pub fn new(lsm_tree: Arc<Mutex<LSMTree>>) -> Self {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let running = Arc::new(AtomicBool::new(true));
        
        // Create worker thread
        let worker_queue = Arc::clone(&queue);
        let worker_running = Arc::clone(&running);
        let worker = thread::spawn(move || {
            while worker_running.load(Ordering::Relaxed) {
                // Process compaction tasks
                if let Some(task) = worker_queue.lock().unwrap().pop_front() {
                    let mut tree = lsm_tree.lock().unwrap();
                    let _ = tree.compact_level(task.level_num);
                } else {
                    thread::sleep(Duration::from_millis(100));
                }
            }
        });
        
        Self {
            compaction_queue: queue,
            worker_thread: Some(worker),
            running,
        }
    }
    
    pub fn schedule_compaction(&self, level_num: usize) {
        self.compaction_queue.lock().unwrap().push_back(CompactionTask { level_num });
    }
    
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.worker_thread.take() {
            let _ = handle.join();
        }
    }
}
```

### 3. Temperature-Based Compaction

Implement different compaction strategies for hot vs. cold data:

```rust
pub struct TemperatureAwareCompactionPolicy {
    hot_threshold: f64,
    hot_policy: Box<dyn CompactionPolicy>,
    cold_policy: Box<dyn CompactionPolicy>,
    access_stats: Arc<Mutex<HashMap<Key, AccessStats>>>,
}

impl CompactionPolicy for TemperatureAwareCompactionPolicy {
    fn should_compact(&self, level: &Level, level_num: usize) -> bool {
        // Determine temperature of the level
        let avg_temp = self.calculate_level_temperature(level);
        
        if avg_temp > self.hot_threshold {
            // Use hot policy for frequently accessed data
            self.hot_policy.should_compact(level, level_num)
        } else {
            // Use cold policy for infrequently accessed data
            self.cold_policy.should_compact(level, level_num)
        }
    }
    
    // Additional implementation...
}
```

### 4. Range-Based Compaction

Implement compaction that targets specific key ranges more aggressively:

```rust
pub struct RangeAwareCompactionPolicy {
    base_policy: Box<dyn CompactionPolicy>,
    hot_ranges: Arc<RwLock<Vec<(Key, Key)>>>,
    hot_compaction_frequency: usize,
}

impl CompactionPolicy for RangeAwareCompactionPolicy {
    fn should_compact(&self, level: &Level, level_num: usize) -> bool {
        // Check if any runs overlap with hot ranges
        let hot_ranges = self.hot_ranges.read().unwrap();
        for run in level.runs() {
            for (start, end) in hot_ranges.iter() {
                if run.overlaps_range(*start, *end) {
                    // Compact hot ranges more frequently
                    return level.run_count() >= (self.hot_compaction_frequency);
                }
            }
        }
        
        // Use base policy for cold ranges
        self.base_policy.should_compact(level, level_num)
    }
    
    // Additional implementation...
}
```

## Conclusion

The LSM tree implementation now includes all three required compaction policies: Tiered, Leveled, and Lazy Leveled. Each policy offers different performance characteristics suitable for different workloads. The implementation follows a pluggable architecture that allows for easy switching between policies and future extensions. The next steps will focus on optimizing compaction further with techniques like partial compaction, background processing, and workload-aware approaches.