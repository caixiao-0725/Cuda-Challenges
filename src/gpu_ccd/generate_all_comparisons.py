import os
import subprocess
import sys

def main():
    # Default workspace directory path
    workspace_dir = os.path.abspath(os.getcwd())
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the visualization script
    visualize_script = os.path.join(script_dir, "visualize_root_timing.py")
    
    # Ensure visualize_root_timing.py exists
    if not os.path.exists(visualize_script):
        print(f"Error: visualize_root_timing.py not found at {visualize_script}")
        sys.exit(1)
    
    # Precision settings to compare
    settings = [
        "results_double",
        "results_float",
        "results_double_fast_math",
        "results_float_fast_math"
    ]
    
    # Create base output directory
    os.makedirs("gpu_root_timing/plots", exist_ok=True)
    
    # 1. Generate the all-in-one comparison
    print("\n=== Generating combined precision comparison ===")
    cmd = [
        sys.executable, 
        visualize_script,
        "--workdir", "gpu_root_timing",
        "--output", "gpu_root_timing/plots",
        "--compare-precision"
    ]
    subprocess.run(cmd)
    
    # 2. Generate individual results for each precision setting
    for setting in settings:
        print(f"\n=== Generating visualizations for {setting} ===")
        csv_path = f"gpu_root_timing/{setting}/summary.csv"
        output_dir = f"gpu_root_timing/plots/{setting}"
        
        # Check if the csv exists
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run visualization for this precision setting
        cmd = [
            sys.executable,
            visualize_script,
            "--csv", csv_path,
            "--output", output_dir
        ]
        subprocess.run(cmd)
    
    # 3. Generate pairwise comparisons
    print("\n=== Generating pairwise comparisons ===")
    
    # Compare double vs float
    if os.path.exists("gpu_root_timing/results_double/summary.csv") and \
       os.path.exists("gpu_root_timing/results_float/summary.csv"):
        compare_two_precision_settings(
            "results_double", 
            "results_float", 
            "double_vs_float"
        )
    
    # Compare standard vs fast math (double)
    if os.path.exists("gpu_root_timing/results_double/summary.csv") and \
       os.path.exists("gpu_root_timing/results_double_fast_math/summary.csv"):
        compare_two_precision_settings(
            "results_double", 
            "results_double_fast_math", 
            "double_standard_vs_fast_math"
        )
    
    # Compare standard vs fast math (float)
    if os.path.exists("gpu_root_timing/results_float/summary.csv") and \
       os.path.exists("gpu_root_timing/results_float_fast_math/summary.csv"):
        compare_two_precision_settings(
            "results_float", 
            "results_float_fast_math", 
            "float_standard_vs_fast_math"
        )
    
    print("\n=== All visualizations completed ===")
    print("Results saved to gpu_root_timing/plots/")

def compare_two_precision_settings(setting1, setting2, output_dirname):
    """Generate a comparison between two precision settings"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the visualization script
    visualize_script = os.path.join(script_dir, "visualize_root_timing.py")
    
    # Create temporary Python script for this comparison
    temp_script = os.path.join(script_dir, "temp_compare.py")
    
    with open(temp_script, "w") as f:
        f.write("""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set a professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def compare_two_settings(setting1, setting2, output_dir):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    df1 = pd.read_csv(f"gpu_root_timing/{setting1}/summary.csv")
    df2 = pd.read_csv(f"gpu_root_timing/{setting2}/summary.csv")
    
    # Add precision labels
    df1['Precision'] = setting1.replace("results_", "")
    df2['Precision'] = setting2.replace("results_", "")
    
    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Make sure columns are correct type
    combined_df['AverageTimeNs'] = pd.to_numeric(combined_df['AverageTimeNs'])
    
    # 1. Overall comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Precision', y='AverageTimeNs', hue='Method', data=combined_df, ci=None)
    plt.title(f'Performance Comparison: {setting1.replace("results_", "")} vs {setting2.replace("results_", "")}')
    plt.ylabel('Average Time (ns)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Comparison by query type
    plt.figure(figsize=(16, 8))
    g = sns.catplot(
        x='Precision', y='AverageTimeNs', hue='Method', col='QueryType',
        data=combined_df, kind='bar', height=6, aspect=1.2, ci=None
    )
    g.set_axis_labels("Precision Setting", "Average Time (ns)")
    g.set_titles("{col_name}")
    g.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_by_query_type.png'), dpi=300)
    plt.close()
    
    # 3. Speedup ratio calculation
    for method in combined_df['Method'].unique():
        method_df = combined_df[combined_df['Method'] == method]
        
        # Convert to pivot table with Precision as columns
        pivot_df = method_df.pivot_table(
            index=['Dataset', 'QueryType'], 
            columns='Precision', 
            values='AverageTimeNs'
        ).reset_index()
        
        precision1 = setting1.replace("results_", "")
        precision2 = setting2.replace("results_", "")
        
        if precision1 in pivot_df.columns and precision2 in pivot_df.columns:
            # Calculate speedup ratio
            ratio_name = f'{precision1}_to_{precision2}_ratio'
            pivot_df[ratio_name] = pivot_df[precision1] / pivot_df[precision2]
            
            # Create boxplot of speedup ratios
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='QueryType', y=ratio_name, data=pivot_df)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
            plt.title(f'{precision1} to {precision2} Ratio for {method} Method')
            plt.ylabel(f'Runtime Ratio (> 1 means {precision1} is slower)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{ratio_name}_{method}.png'), dpi=300)
            plt.close()
            
            # Detailed ratio by dataset
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Dataset', y=ratio_name, hue='QueryType', data=pivot_df, ci=None)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
            plt.title(f'{precision1} to {precision2} Ratio by Dataset for {method} Method')
            plt.ylabel(f'Runtime Ratio (> 1 means {precision1} is slower)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{ratio_name}_by_dataset_{method}.png'), dpi=300)
            plt.close()
    
    # 4. Create summary table
    summary_table = combined_df.groupby(['Precision', 'Method', 'QueryType'])['AverageTimeNs'].agg(
        ['mean', 'std', 'min', 'max']
    ).reset_index()
    
    # Rename columns for readability
    summary_table.columns = ['Precision', 'Method', 'QueryType', 'Mean (ns)', 
                           'Std Dev (ns)', 'Min (ns)', 'Max (ns)']
    
    # Save summary as CSV
    summary_table.to_csv(os.path.join(output_dir, 'comparison_summary.csv'), index=False)
    
    print(f"Comparison between {setting1} and {setting2} completed. Results saved to {output_dir}")

# Parameters from command line
import sys
setting1 = sys.argv[1]
setting2 = sys.argv[2]
output_dir = sys.argv[3]

compare_two_settings(setting1, setting2, f"gpu_root_timing/plots/{output_dir}")
""")
    
    # Run the temporary script
    cmd = [sys.executable, temp_script, setting1, setting2, output_dirname]
    subprocess.run(cmd)
    
    # Clean up
    os.remove(temp_script)

if __name__ == "__main__":
    main() 