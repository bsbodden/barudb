#!/usr/bin/env python3
"""
Script to create state-of-the-art (SOTA) comparison charts for the LSM-Tree implementation.
This script generates visualizations comparing our LSM Tree to other key-value stores.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style for better visualizations
sns.set_theme(style="whitegrid")

# Create output directory for plots
PLOTS_DIR = "../images"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Color palette for consistent colors across charts
COLOR_PALETTE = {
    "Our LSM Tree": "#1f77b4",  # blue
    "RocksDB": "#ff7f0e",       # orange
    "LevelDB": "#2ca02c",       # green
    "SpeedB": "#d62728",        # red
    "LMDB": "#9467bd",          # purple
    "WiredTiger": "#8c564b",    # brown
    "TerarkDB": "#e377c2",      # pink
}

def create_write_performance_comparison():
    """Create a chart comparing write performance across different systems."""
    
    # Data from final-report-part11.md
    systems = ["Our LSM Tree", "RocksDB", "LevelDB", "SpeedB", "LMDB", "WiredTiger", "TerarkDB"]
    
    # Write performance data
    sequential = [1854327, 2053642, 731526, 2207318, 521784, 1321453, 1135627]
    random = [1723645, 1982317, 678921, 2118753, 483927, 1276352, 1072845]
    sequential_batch = [5127358, 5317845, 1827519, 5521679, 872514, 3127845, 2874512]
    random_batch = [4872519, 5054721, 1758471, 5231845, 825971, 2998754, 2765389]
    
    # Create DataFrame
    df = pd.DataFrame({
        'System': systems,
        'Sequential Write': sequential,
        'Random Write': random,
        'Sequential Batch Write': sequential_batch,
        'Random Batch Write': random_batch
    })
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Melt the dataframe for easier plotting with seaborn
    df_melted = pd.melt(df, id_vars=['System'], 
                        value_vars=['Sequential Write', 'Random Write', 
                                   'Sequential Batch Write', 'Random Batch Write'],
                        var_name='Write Type', value_name='Operations/sec')
    
    # Create the grouped bar chart
    chart = sns.barplot(x='System', y='Operations/sec', hue='Write Type', data=df_melted,
                palette=sns.color_palette("muted", 4))
    
    # Customize the plot
    plt.title('Write Performance Comparison Across Systems', fontsize=16)
    plt.xlabel('System', fontsize=14)
    plt.ylabel('Operations per Second', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Write Type', fontsize=12)
    
    # Use scientific notation for y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Add value labels on bars (but only on the first bar of each group)
    for i, system in enumerate(systems):
        # Get the sequential write bar height
        height = sequential[i]
        # Format with commas for better readability
        chart.text(i, height + max(sequential) * 0.02, f'{height/1000:.0f}K', 
                  ha='center', va='bottom', rotation=0, fontsize=9, color='black')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, 'write_performance_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Created write performance comparison chart at {plot_path}")

def create_read_performance_comparison():
    """Create a chart comparing read performance across different systems."""
    
    # Data from final-report-part11.md
    systems = ["Our LSM Tree", "RocksDB", "LevelDB", "SpeedB", "LMDB", "WiredTiger", "TerarkDB"]
    
    # Read performance data
    point_query = [142873, 168427, 62381, 185276, 325427, 198745, 147152]
    range_small = [38527, 42183, 18472, 47318, 102183, 56183, 31472]
    range_large = [4782, 5124, 2189, 5832, 12587, 6724, 3821]
    scan_all = [37.5, 41.2, 17.8, 44.7, 97.3, 53.8, 29.4]
    
    # Create DataFrame for point queries and range queries
    df = pd.DataFrame({
        'System': systems,
        'Point Query': point_query,
        'Range Query (Small)': range_small,
        'Range Query (Large)': range_large
    })
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Melt the dataframe for easier plotting with seaborn
    df_melted = pd.melt(df, id_vars=['System'], 
                      value_vars=['Point Query', 'Range Query (Small)', 'Range Query (Large)'],
                      var_name='Query Type', value_name='Operations/sec')
    
    # Create the grouped bar chart
    chart = sns.barplot(x='System', y='Operations/sec', hue='Query Type', data=df_melted,
                palette=sns.color_palette("muted", 3))
    
    # Customize the plot
    plt.title('Read Performance Comparison Across Systems', fontsize=16)
    plt.xlabel('System', fontsize=14)
    plt.ylabel('Operations per Second', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Query Type', fontsize=12)
    
    # Use scientific notation for y-axis for larger values
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Add value labels on the point query bars
    for i, system in enumerate(systems):
        height = point_query[i]
        chart.text(i, height + max(point_query) * 0.02, f'{height/1000:.0f}K', 
                  ha='center', va='bottom', rotation=0, fontsize=9, color='black')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, 'read_performance_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Created read performance comparison chart at {plot_path}")

def create_space_efficiency_comparison():
    """Create a chart comparing space efficiency across different systems."""
    
    # Data from final-report-part11.md
    systems = ["Our LSM Tree", "RocksDB", "LevelDB", "SpeedB", "LMDB", "WiredTiger", "TerarkDB"]
    
    # Space efficiency data
    space_usage = [3.8, 3.2, 4.2, 3.0, 10.5, 4.8, 2.4]  # GB for 10GB dataset
    compression_ratio = [2.63, 3.12, 2.38, 3.33, 0.95, 2.08, 4.17]
    space_amplification = [1.23, 1.14, 1.31, 1.11, 1.42, 1.35, 1.08]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'System': systems,
        'Space Usage (GB)': space_usage,
        'Compression Ratio': compression_ratio,
        'Space Amplification': space_amplification
    })
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot 1: Space Usage (lower is better)
    space_usage_bars = ax1.bar(systems, space_usage, color=[COLOR_PALETTE.get(s, "#333333") for s in systems])
    ax1.set_title('Space Usage (Lower is Better)', fontsize=14)
    ax1.set_xlabel('System', fontsize=12)
    ax1.set_ylabel('Space Usage (GB for 10GB dataset)', fontsize=12)
    ax1.set_ylim(0, max(space_usage) * 1.2)
    # Add a horizontal line at 10GB (original dataset size)
    ax1.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Original Dataset Size (10GB)')
    ax1.legend()
    # Label the bars with values
    for bar in space_usage_bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}GB', ha='center', va='bottom', fontsize=9)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Compression Ratio (higher is better)
    compression_ratio_bars = ax2.bar(systems, compression_ratio, color=[COLOR_PALETTE.get(s, "#333333") for s in systems])
    ax2.set_title('Compression Ratio (Higher is Better)', fontsize=14)
    ax2.set_xlabel('System', fontsize=12)
    ax2.set_ylabel('Compression Ratio (x)', fontsize=12)
    ax2.set_ylim(0, max(compression_ratio) * 1.2)
    # Add a horizontal line at 1.0 (no compression)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No Compression (1.0x)')
    ax2.legend()
    # Label the bars with values
    for bar in compression_ratio_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, 'space_efficiency_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Created space efficiency comparison chart at {plot_path}")

def create_resource_usage_comparison():
    """Create a chart comparing CPU and memory usage across different systems."""
    
    # Data from final-report-part11.md
    systems = ["Our LSM Tree", "RocksDB", "LevelDB", "SpeedB", "LMDB", "WiredTiger", "TerarkDB"]
    
    # Resource usage data
    cpu_usage = [23.5, 28.2, 14.8, 31.5, 12.3, 21.7, 35.2]  # Percentage
    memory_usage = [845, 1250, 320, 1380, 9850, 1050, 2450]  # MB
    write_amplification = [3.7, 4.2, 5.8, 3.9, 1.0, 2.3, 3.1]
    read_amplification = [1.3, 1.2, 1.9, 1.1, 1.0, 1.1, 1.4]
    
    # Create a figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: CPU Usage (lower is better)
    cpu_bars = ax1.bar(systems, cpu_usage, color=[COLOR_PALETTE.get(s, "#333333") for s in systems])
    ax1.set_title('CPU Usage (Lower is Better)', fontsize=14)
    ax1.set_xlabel('System', fontsize=12)
    ax1.set_ylabel('CPU Usage (%)', fontsize=12)
    ax1.set_ylim(0, max(cpu_usage) * 1.2)
    # Label the bars with values
    for bar in cpu_bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Memory Usage (lower is better)
    memory_bars = ax2.bar(systems, memory_usage, color=[COLOR_PALETTE.get(s, "#333333") for s in systems])
    ax2.set_title('Memory Usage (Lower is Better)', fontsize=14)
    ax2.set_xlabel('System', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.set_ylim(0, max(memory_usage) * 1.2)
    # Label the bars with values
    for bar in memory_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}MB', ha='center', va='bottom', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Write Amplification (lower is better)
    write_bars = ax3.bar(systems, write_amplification, color=[COLOR_PALETTE.get(s, "#333333") for s in systems])
    ax3.set_title('Write Amplification (Lower is Better)', fontsize=14)
    ax3.set_xlabel('System', fontsize=12)
    ax3.set_ylabel('Write Amplification Factor', fontsize=12)
    ax3.set_ylim(0, max(write_amplification) * 1.2)
    # Label the bars with values
    for bar in write_bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}x', ha='center', va='bottom', fontsize=9)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Read Amplification (lower is better)
    read_bars = ax4.bar(systems, read_amplification, color=[COLOR_PALETTE.get(s, "#333333") for s in systems])
    ax4.set_title('Read Amplification (Lower is Better)', fontsize=14)
    ax4.set_xlabel('System', fontsize=12)
    ax4.set_ylabel('Read Amplification Factor', fontsize=12)
    ax4.set_ylim(0, max(read_amplification) * 1.2)
    # Label the bars with values
    for bar in read_bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}x', ha='center', va='bottom', fontsize=9)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, 'resource_usage_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Created resource usage comparison chart at {plot_path}")

if __name__ == "__main__":
    create_write_performance_comparison()
    create_read_performance_comparison()
    create_space_efficiency_comparison()
    create_resource_usage_comparison()
    print("All SOTA comparison charts created successfully")