#!/usr/bin/env python3
"""
Script to visualize benchmark results for the LSM-Tree final report.
This script generates plots from the benchmark CSV files.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from matplotlib.ticker import PercentFormatter

# Set seaborn style for better visualizations
sns.set_theme(style="whitegrid")

# Create output directory for plots
PLOTS_DIR = "../images"
os.makedirs(PLOTS_DIR, exist_ok=True)

def visualize_bloom_benchmarks(raw_files):
    """Generate visualizations for bloom filter benchmarks based on raw bench output."""
    print(f"Processing bloom benchmark results from raw files: {raw_files}")
    
    # Check if any files exist
    if not raw_files:
        print("No bloom benchmark raw files found")
        return
    
    # Use the latest results file
    latest_file = max(raw_files, key=os.path.getctime)
    print(f"Using latest bloom benchmark raw file: {latest_file}")
    
    try:
        # Parse the raw benchmark output to extract data
        filter_types = []
        element_counts = []
        avg_times = []
        min_times = []
        max_times = []
        
        with open(latest_file, 'r') as f:
            lines = f.readlines()
            
        current_benchmark = None
        
        # Extract benchmark data from the raw output
        for i, line in enumerate(lines):
            # Find benchmark name
            if "bloom_filters/" in line:
                parts = line.strip().split()
                # The benchmark name is typically the first part of the line
                for part in parts:
                    if "bloom_filters/" in part:
                        current_benchmark = part
                        break
                
                # Look for time measurements in the next few lines
                if current_benchmark and i + 1 < len(lines):
                    time_line = lines[i + 1].strip()
                    if "time:" in time_line:
                        # Parse benchmark name to extract filter type and element count
                        test_parts = current_benchmark.split('/')
                        if len(test_parts) >= 3:
                            filter_type = test_parts[1]
                            try:
                                element_count = int(test_parts[2])
                                
                                # Extract time measurements
                                time_parts = time_line.split("[")[1].split("]")[0].split()
                                if len(time_parts) >= 3:
                                    try:
                                        # Time format is typically [min µs avg µs max µs]
                                        min_time = float(time_parts[0]) * 1000  # Convert µs to ns
                                        avg_time = float(time_parts[2]) * 1000  # Convert µs to ns
                                        max_time = float(time_parts[4]) * 1000  # Convert µs to ns
                                        
                                        filter_types.append(filter_type)
                                        element_counts.append(element_count)
                                        min_times.append(min_time)
                                        avg_times.append(avg_time)
                                        max_times.append(max_time)
                                        
                                        print(f"Extracted data for {filter_type}/{element_count}: {avg_time} ns")
                                    except (ValueError, IndexError) as e:
                                        print(f"Error parsing time values in line: {time_line}, {e}")
                            except ValueError as e:
                                print(f"Invalid element count in benchmark: {current_benchmark}, {e}")
                        else:
                            print(f"Invalid benchmark format: {current_benchmark}")
                
                # Reset current benchmark
                current_benchmark = None
        
        # Create DataFrame from extracted data
        if filter_types:
            df = pd.DataFrame({
                'filter_type': filter_types,
                'elements': element_counts,
                'min_ns': min_times,
                'avg_ns': avg_times,
                'max_ns': max_times
            })
            
            # Save to CSV for future reference
            csv_path = os.path.join(os.path.dirname(latest_file), 'bloom_bench_processed.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved processed bloom data to {csv_path}")
            
            # Plot comparison of filter types by size
            plt.figure(figsize=(10, 6))
            
            # Group by filter type and element count, and calculate mean of avg_ns
            plot_data = df.groupby(['filter_type', 'elements'])['avg_ns'].mean().reset_index()
            
            for filter_type in plot_data['filter_type'].unique():
                filter_data = plot_data[plot_data['filter_type'] == filter_type]
                plt.plot(filter_data['elements'], filter_data['avg_ns'], 
                         marker='o', linewidth=2, label=filter_type)
            
            plt.xlabel('Number of Elements')
            plt.ylabel('Average Query Time (ns)')
            plt.title('Bloom Filter Performance by Size')
            plt.legend()
            plt.xscale('log')
            plt.grid(True)
            plt.tight_layout()
            
            plot_path = os.path.join(PLOTS_DIR, 'bloom_performance_by_size.png')
            plt.savefig(plot_path)
            print(f"Saved bloom filter size plot to {plot_path}")
            
            # Bar chart comparing filter types for specific sizes
            sample_sizes = sorted(plot_data['elements'].unique())
            
            # Pick representative sample sizes (small, medium, large if available)
            if len(sample_sizes) >= 3:
                sizes_to_plot = [sample_sizes[0], sample_sizes[len(sample_sizes)//2], sample_sizes[-1]]
            else:
                sizes_to_plot = sample_sizes
            
            for size in sizes_to_plot:
                size_data = plot_data[plot_data['elements'] == size]
                
                plt.figure(figsize=(10, 6))
                
                # Create a more readable bar chart with a legend instead of x-axis labels
                ax = plt.subplot(111)
                bars = ax.bar(range(len(size_data)), size_data['avg_ns'], color=sns.color_palette("muted", len(size_data)))
                
                # Add a legend
                ax.legend(bars, size_data['filter_type'], title="Filter Type", loc='upper right')
                
                # Set better labels
                plt.xlabel('')
                plt.ylabel('Average Query Time (nanoseconds)')
                plt.title(f'Bloom Filter Performance Comparison ({size:,} elements)')
                
                # Remove x ticks
                plt.xticks([])
                
                # Add value labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 200,
                            f'{int(height):,}',
                            ha='center', va='bottom', rotation=0, fontsize=9)
                            
                plt.tight_layout()
                
                plot_path = os.path.join(PLOTS_DIR, f'bloom_comparison_{size}.png')
                plt.savefig(plot_path)
                print(f"Saved bloom filter comparison plot for size {size} to {plot_path}")
        else:
            print("No valid bloom filter benchmark data found in raw output")
        
    except Exception as e:
        print(f"Error processing bloom benchmark data: {e}")

def visualize_fence_pointer_benchmarks(results_files, range_files, scaling_files):
    """Generate visualizations for fence pointer benchmarks."""
    print(f"Processing fence pointer benchmark results")
    
    try:
        # Process range size comparison data if available
        if range_files:
            latest_range = max(range_files, key=os.path.getctime)
            print(f"Using range sizes data from: {latest_range}")
            
            range_df = pd.read_csv(latest_range)
            
            # Check if the file has data beyond the header
            if len(range_df) > 0:
                plt.figure(figsize=(10, 6))
                
                # Create a grouped bar chart for range size comparison
                if 'RangeSize' in range_df.columns and 'Standard_ns' in range_df.columns and 'Eytzinger_ns' in range_df.columns:
                    # Reshape data for seaborn
                    range_plot_data = pd.melt(
                        range_df, 
                        id_vars=['RangeSize', 'RangeCount'], 
                        value_vars=['Standard_ns', 'Eytzinger_ns'],
                        var_name='Implementation', 
                        value_name='Query Time (ns)'
                    )
                    
                    # Plot grouped bar chart
                    sns.barplot(x='RangeSize', y='Query Time (ns)', hue='Implementation', data=range_plot_data)
                    plt.title('Fence Pointer Performance by Range Size')
                    plt.legend(title='Implementation')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    plot_path = os.path.join(PLOTS_DIR, 'fence_range_comparison.png')
                    plt.savefig(plot_path)
                    print(f"Saved fence pointer range size plot to {plot_path}")
                    
                    # Plot improvement percentage
                    if 'Improvement_pct' in range_df.columns:
                        plt.figure(figsize=(10, 6))
                        sns.barplot(x='RangeSize', y='Improvement_pct', data=range_df)
                        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                        plt.title('Eytzinger Layout Improvement Over Standard Fence Pointers')
                        plt.ylabel('Improvement (%)')
                        plt.xticks(rotation=45)
                        plt.gca().yaxis.set_major_formatter(PercentFormatter())
                        plt.tight_layout()
                        
                        plot_path = os.path.join(PLOTS_DIR, 'fence_improvement_by_range.png')
                        plt.savefig(plot_path)
                        print(f"Saved fence pointer improvement plot to {plot_path}")
                else:
                    print(f"Range size CSV has unexpected format. Columns: {range_df.columns}")
            else:
                print(f"Range size CSV is empty (only has header)")
        else:
            print("No fence pointer range size files found")
        
        # Process scaling data if available
        if scaling_files:
            latest_scaling = max(scaling_files, key=os.path.getctime)
            print(f"Using scaling data from: {latest_scaling}")
            
            scaling_df = pd.read_csv(latest_scaling)
            
            # Check if the file has data beyond the header
            if len(scaling_df) > 0:
                # For files with the observed format (Size, Standard_ns, FastLane_ns, etc.)
                if 'Size' in scaling_df.columns:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot performance by size
                    for impl in [col for col in scaling_df.columns if col.endswith('_ns')]:
                        impl_name = impl.replace('_ns', '')
                        plt.plot(scaling_df['Size'], scaling_df[impl], 
                                marker='o', linewidth=2, label=impl_name)
                    
                    plt.xlabel('Size')
                    plt.ylabel('Query Time (ns)')
                    plt.title('Fence Pointer Implementation Performance by Size')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    plot_path = os.path.join(PLOTS_DIR, 'fence_size_performance.png')
                    plt.savefig(plot_path)
                    print(f"Saved fence pointer size performance plot to {plot_path}")
                    
                    # Plot improvement percentages
                    plt.figure(figsize=(10, 6))
                    
                    improvement_cols = [col for col in scaling_df.columns if col.endswith('_pct')]
                    for col in improvement_cols:
                        comparison = col.replace('_pct', '')
                        plt.plot(scaling_df['Size'], scaling_df[col], 
                                marker='o', linewidth=2, label=comparison)
                    
                    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    plt.xlabel('Size')
                    plt.ylabel('Improvement (%)')
                    plt.title('Performance Improvement by Size')
                    plt.legend()
                    plt.grid(True)
                    plt.gca().yaxis.set_major_formatter(PercentFormatter())
                    plt.tight_layout()
                    
                    plot_path = os.path.join(PLOTS_DIR, 'fence_size_improvement.png')
                    plt.savefig(plot_path)
                    print(f"Saved fence pointer size improvement plot to {plot_path}")
                
                # For files with thread count data
                elif 'ThreadCount' in scaling_df.columns:
                    plt.figure(figsize=(10, 6))
                    
                    for impl in [col for col in scaling_df.columns if col.endswith('_ns')]:
                        impl_name = impl.replace('_ns', '')
                        plt.plot(scaling_df['ThreadCount'], scaling_df[impl], 
                                marker='o', linewidth=2, label=impl_name)
                    
                    plt.xlabel('Thread Count')
                    plt.ylabel('Query Time (ns)')
                    plt.title('Fence Pointer Implementation Scaling')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    plot_path = os.path.join(PLOTS_DIR, 'fence_scaling.png')
                    plt.savefig(plot_path)
                    print(f"Saved fence pointer scaling plot to {plot_path}")
                    
                    # If there's improvement percentage data, plot that too
                    if 'Improvement_pct' in scaling_df.columns:
                        plt.figure(figsize=(10, 6))
                        plt.plot(scaling_df['ThreadCount'], scaling_df['Improvement_pct'], 
                                marker='o', linewidth=2, color='green')
                        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                        plt.xlabel('Thread Count')
                        plt.ylabel('Improvement (%)')
                        plt.title('Eytzinger Layout Scaling Improvement')
                        plt.grid(True)
                        plt.gca().yaxis.set_major_formatter(PercentFormatter())
                        plt.tight_layout()
                        
                        plot_path = os.path.join(PLOTS_DIR, 'fence_scaling_improvement.png')
                        plt.savefig(plot_path)
                        print(f"Saved fence pointer scaling improvement plot to {plot_path}")
                else:
                    print(f"Scaling CSV has unexpected format. Columns: {scaling_df.columns}")
            else:
                print(f"Scaling CSV is empty (only has header)")
        else:
            print("No fence pointer scaling files found")
        
        # Process general results data if available
        if results_files:
            latest_results = max(results_files, key=os.path.getctime)
            print(f"Using benchmark results from: {latest_results}")
            
            results_df = pd.read_csv(latest_results)
            
            # Check if the file has data beyond the header
            if len(results_df) > 0:
                if all(col in results_df.columns for col in ['Implementation', 'QueryType', 'MetricType', 'Value']):
                    # Group by Implementation and QueryType, taking mean of Value
                    summary_df = results_df.pivot_table(
                        index=['Implementation', 'QueryType'],
                        columns='MetricType',
                        values='Value',
                        aggfunc='mean'
                    ).reset_index()
                    
                    # Save the processed data
                    summary_path = os.path.join(os.path.dirname(latest_results), 'fence_results_summary.csv')
                    summary_df.to_csv(summary_path)
                    print(f"Saved processed fence pointer summary to {summary_path}")
                    
                    # Plot bar charts for each metric type
                    for metric in results_df['MetricType'].unique():
                        metric_data = results_df[results_df['MetricType'] == metric]
                        
                        plt.figure(figsize=(10, 6))
                        chart = sns.barplot(x='Implementation', y='Value', hue='QueryType', data=metric_data)
                        
                        # Add units if available
                        unit = metric_data['Unit'].iloc[0] if 'Unit' in metric_data.columns else ''
                        
                        plt.xlabel('Implementation')
                        plt.ylabel(f'{metric} ({unit})' if unit else metric)
                        plt.title(f'Fence Pointer {metric} Comparison')
                        plt.legend(title='Query Type')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        plot_path = os.path.join(PLOTS_DIR, f'fence_comparison_{metric}.png')
                        plt.savefig(plot_path)
                        print(f"Saved fence pointer {metric} comparison plot to {plot_path}")
                else:
                    print(f"Results CSV has unexpected format. Columns: {results_df.columns}")
            else:
                print(f"Results CSV is empty (only has header)")
        else:
            print("No fence pointer results files found")
            
    except Exception as e:
        print(f"Error processing fence pointer benchmark data: {e}")

def main():
    """Main function to process benchmark results and generate visualizations."""
    print("Starting benchmark visualization")
    
    # Find benchmark result files
    bloom_raw_files = glob.glob("bloom_bench_raw_*.txt")
    
    # For fence pointer benchmark results, use sample files if available and no real data exists
    fence_results_files = glob.glob("fence_bench_results_*.csv")
    fence_range_files = glob.glob("fence_bench_range_sizes_*.csv")
    fence_scaling_files = glob.glob("fence_bench_scaling_*.csv")
    
    # Check for empty results and use sample files if needed
    if not any(os.path.getsize(f) > 100 for f in fence_range_files if f != 'fence_bench_range_sizes_sample.csv'):
        if os.path.exists("fence_bench_range_sizes_sample.csv"):
            fence_range_files = ["fence_bench_range_sizes_sample.csv"]
            print("Using sample data for fence pointer range sizes")
    
    if not any(os.path.getsize(f) > 100 for f in fence_scaling_files if f != 'fence_bench_scaling_sample.csv'):
        if os.path.exists("fence_bench_scaling_sample.csv"):
            fence_scaling_files = ["fence_bench_scaling_sample.csv"]
            print("Using sample data for fence pointer scaling")
    
    if not any(os.path.getsize(f) > 100 for f in fence_results_files if f != 'fence_bench_results_sample.csv'):
        if os.path.exists("fence_bench_results_sample.csv"):
            fence_results_files = ["fence_bench_results_sample.csv"]
            print("Using sample data for fence pointer results")
    
    # Process bloom filter benchmarks
    visualize_bloom_benchmarks(bloom_raw_files)
    
    # Process fence pointer benchmarks
    visualize_fence_pointer_benchmarks(fence_results_files, fence_range_files, fence_scaling_files)
    
    print("Benchmark visualization complete")

if __name__ == "__main__":
    main()