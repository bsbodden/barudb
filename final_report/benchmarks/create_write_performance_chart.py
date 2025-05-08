#!/usr/bin/env python3
"""
Script to create write performance visualization from performance target data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visualizations
sns.set_theme(style="whitegrid")

# Create output directory for plots
PLOTS_DIR = "../images"
os.makedirs(PLOTS_DIR, exist_ok=True)

def create_write_performance_chart():
    """Create a chart visualizing write performance scaling from the performance target data."""
    
    # Data from performance_targets_data.md
    thread_counts = [1, 2, 4, 8, 16]
    ops_per_sec = [1254873, 2341562, 4218726, 7654291, 12874365]
    scaling_factors = [1.00, 1.87, 3.36, 6.10, 10.26]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Thread Count': thread_counts,
        'Operations/sec': ops_per_sec,
        'Scaling Factor': scaling_factors
    })
    
    # Plot write performance scaling
    plt.figure(figsize=(10, 6))
    
    # Primary y-axis for operations/sec
    ax1 = plt.gca()
    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Operations per Second')
    ax1.plot(df['Thread Count'], df['Operations/sec'], 'o-', color='blue', linewidth=2, label='Throughput')
    
    # Add target line for 1M ops/sec
    ax1.axhline(y=1000000, color='red', linestyle='--', label='Target (1M ops/sec)')
    
    # Secondary y-axis for scaling factor
    ax2 = ax1.twinx()
    ax2.set_ylabel('Scaling Factor')
    ax2.plot(df['Thread Count'], df['Scaling Factor'], 'o--', color='green', linewidth=2, label='Scaling Factor')
    
    # Add ideal scaling reference line
    ax2.plot(df['Thread Count'], df['Thread Count'], 'k--', alpha=0.3, label='Ideal Scaling')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Write Performance Scaling with Thread Count')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, 'write_performance.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Created write performance chart at {plot_path}")
    
    # Create a read performance chart too
    create_read_performance_chart()

def create_read_performance_chart():
    """Create a chart visualizing read performance scaling from the performance target data."""
    
    # Data from performance_targets_data.md
    thread_counts = [1, 2, 4, 8, 16]
    ops_per_sec = [73642, 145283, 285762, 541936, 892571]
    scaling_factors = [1.00, 1.97, 3.88, 7.36, 12.12]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Thread Count': thread_counts,
        'Operations/sec': ops_per_sec,
        'Scaling Factor': scaling_factors
    })
    
    # Plot read performance scaling
    plt.figure(figsize=(10, 6))
    
    # Primary y-axis for operations/sec
    ax1 = plt.gca()
    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Operations per Second')
    ax1.plot(df['Thread Count'], df['Operations/sec'], 'o-', color='blue', linewidth=2, label='Throughput')
    
    # Add target line for 50K ops/sec
    ax1.axhline(y=50000, color='red', linestyle='--', label='Target (50K ops/sec)')
    
    # Secondary y-axis for scaling factor
    ax2 = ax1.twinx()
    ax2.set_ylabel('Scaling Factor')
    ax2.plot(df['Thread Count'], df['Scaling Factor'], 'o--', color='green', linewidth=2, label='Scaling Factor')
    
    # Add ideal scaling reference line
    ax2.plot(df['Thread Count'], df['Thread Count'], 'k--', alpha=0.3, label='Ideal Scaling')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Read Performance Scaling with Thread Count')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, 'read_performance.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Created read performance chart at {plot_path}")

if __name__ == "__main__":
    create_write_performance_chart()