import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import glob

# Set a professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def visualize_root_timing(csv_path, output_dir):
    """
    Visualize the timing comparison between direct and poly methods.
    
    Args:
        csv_path: Path to the summary.csv file
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Make sure columns are correct type
    df['AverageTimeNs'] = pd.to_numeric(df['AverageTimeNs'])
    
    # 1. Generate overall comparison plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Method', y='AverageTimeNs', data=df, ci=None)
    plt.title('Overall Performance Comparison: Direct vs. Poly')
    plt.ylabel('Average Time (ns)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Generate comparison by query type
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Method', y='AverageTimeNs', hue='QueryType', data=df, ci=None)
    plt.title('Performance by Query Type')
    plt.ylabel('Average Time (ns)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_by_query_type.png'), dpi=300)
    plt.close()
    
    # 3. Generate comparison by dataset - ALL datasets
    # Count number of unique datasets
    num_datasets = df['Dataset'].nunique()
    
    # Adjust figure size based on number of datasets
    fig_width = max(16, num_datasets * 1.2)  # Ensure minimum width of 16 inches
    
    plt.figure(figsize=(fig_width, 10))
    sns.barplot(x='Dataset', y='AverageTimeNs', hue='Method', data=df, ci=None)
    plt.title('Performance by Dataset (All)')
    plt.ylabel('Average Time (ns)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_by_dataset.png'), dpi=300)
    plt.close()
    
    # 4. Calculate and visualize speedup ratio (direct/poly)
    # Pivot the data to get direct and poly times side by side
    pivot_df = df.pivot_table(
        index=['Dataset', 'QueryType'], 
        columns='Method', 
        values='AverageTimeNs'
    ).reset_index()
    
    # Calculate speedup ratio (how many times faster is direct compared to poly)
    pivot_df['SpeedupRatio'] = pivot_df['direct'] / pivot_df['poly']
    
    # Create boxplot of speedup ratios
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='QueryType', y='SpeedupRatio', data=pivot_df)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    plt.title('Speedup Ratio: direct / poly')
    plt.ylabel('Speedup Ratio (> 1 means direct is slower)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_ratio_by_query_type.png'), dpi=300)
    plt.close()
    
    # 5. Detailed speedup ratio by dataset
    plt.figure(figsize=(fig_width, 10))
    sns.barplot(x='Dataset', y='SpeedupRatio', hue='QueryType', data=pivot_df, ci=None)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    plt.title('Speedup Ratio by Dataset: direct / poly')
    plt.ylabel('Speedup Ratio (> 1 means direct is slower)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_ratio_by_dataset.png'), dpi=300)
    plt.close()
    
    # 6. Create a table with average values
    summary_table = df.groupby(['Method', 'QueryType'])['AverageTimeNs'].agg(
        ['mean', 'std', 'min', 'max']
    ).reset_index()
    summary_table.columns = ['Method', 'QueryType', 'Mean (ns)', 'Std Dev (ns)', 'Min (ns)', 'Max (ns)']
    
    # Convert to latex table and save
    with open(os.path.join(output_dir, 'summary_table.txt'), 'w') as f:
        f.write(summary_table.to_latex(index=False, float_format="%.2f"))
    
    # Also save as CSV
    summary_table.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
    
    # 7. Save detailed results by dataset
    dataset_table = df.pivot_table(
        index='Dataset',
        columns=['Method', 'QueryType'],
        values='AverageTimeNs'
    ).reset_index()
    
    dataset_table.to_csv(os.path.join(output_dir, 'detailed_by_dataset.csv'))
    
    print(f"Visualization completed. Results saved to {output_dir}")
    
    # Return the dataframe for further analysis if needed
    return df

def compare_precision_settings(workdir, output_dir):
    """
    Compare results across different precision and optimization settings.
    
    Args:
        workdir: Base workspace directory containing the results folders
        output_dir: Directory to save the visualization outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the precision settings to compare
    settings = [
        "results_double",
        "results_float",
        "results_double_fast_math",
        "results_float_fast_math"
    ]
    
    # Dictionary to store dataframes for each setting
    dfs = {}
    
    # Load all the summary.csv files
    for setting in settings:
        csv_path = os.path.join(workdir, setting, "summary.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Precision'] = setting.replace("results_", "")
            dfs[setting] = df
        else:
            print(f"Warning: {csv_path} not found")
    
    # If no data was loaded, exit
    if not dfs:
        print("Error: No summary.csv files found.")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs.values(), ignore_index=True)
    
    # Make sure columns are correct type
    combined_df['AverageTimeNs'] = pd.to_numeric(combined_df['AverageTimeNs'])
    
    # 1. Overall comparison by precision setting and method
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Precision', y='AverageTimeNs', hue='Method', data=combined_df, ci=None)
    plt.title('Performance by Precision and Math Mode')
    plt.ylabel('Average Time (ns)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Comparison by precision setting, method, and query type
    plt.figure(figsize=(18, 10))
    g = sns.catplot(
        x='Precision', y='AverageTimeNs', hue='Method', col='QueryType',
        data=combined_df, kind='bar', height=6, aspect=1.2, ci=None
    )
    g.set_axis_labels("Precision Setting", "Average Time (ns)")
    g.set_titles("{col_name}")
    g.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_comparison_by_query_type.png'), dpi=300)
    plt.close()
    
    # 3. Calculate speedup between float and double
    # Separate data by method
    for method in combined_df['Method'].unique():
        method_df = combined_df[combined_df['Method'] == method]
        
        # Convert to pivot table with Precision as columns
        pivot_df = method_df.pivot_table(
            index=['Dataset', 'QueryType'], 
            columns='Precision', 
            values='AverageTimeNs'
        ).reset_index()
        
        # Only proceed if we have both double and float data
        if 'double' in pivot_df.columns and 'float' in pivot_df.columns:
            # Calculate speedup ratio (double/float)
            pivot_df['DoubleTofloatRatio'] = pivot_df['double'] / pivot_df['float']
            
            # Create boxplot of speedup ratios
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='QueryType', y='DoubleTofloatRatio', data=pivot_df)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
            plt.title(f'Double to Float Ratio for {method} Method')
            plt.ylabel('Runtime Ratio (> 1 means double is slower)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'double_float_ratio_{method}.png'), dpi=300)
            plt.close()
        
        # Check for fast math versions
        if 'double_fast_math' in pivot_df.columns and 'double' in pivot_df.columns:
            # Calculate speedup ratio (standard/fast_math)
            pivot_df['FastMathSpeedup'] = pivot_df['double'] / pivot_df['double_fast_math']
            
            # Create boxplot of speedup ratios
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='QueryType', y='FastMathSpeedup', data=pivot_df)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
            plt.title(f'Standard vs Fast Math Speedup for {method} Method (Double)')
            plt.ylabel('Speedup Ratio (> 1 means fast math is faster)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'fast_math_speedup_double_{method}.png'), dpi=300)
            plt.close()
    
    # 4. Overall comparison across all combinations
    plt.figure(figsize=(16, 12))
    
    # Create a custom categorical plot
    g = sns.catplot(
        data=combined_df, kind="bar",
        x="Dataset", y="AverageTimeNs", hue="Precision", col="Method", row="QueryType",
        height=4, aspect=1.5, palette="muted", ci=None
    )
    
    # Customize the plot
    g.set_xticklabels(rotation=45, ha="right")
    g.set_titles("{row_name} - {col_name}")
    g.set_axis_labels("Dataset", "Average Time (ns)")
    g.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300)
    plt.close()
    
    # 5. Create a summary table with all combinations
    summary_table = combined_df.groupby(['Precision', 'Method', 'QueryType'])['AverageTimeNs'].agg(
        ['mean', 'std', 'min', 'max', 'count']
    ).reset_index()
    
    # Rename the columns for better readability
    summary_table.columns = ['Precision', 'Method', 'QueryType', 'Mean (ns)', 
                           'Std Dev (ns)', 'Min (ns)', 'Max (ns)', 'Count']
    
    # Save the summary as CSV
    summary_table.to_csv(os.path.join(output_dir, 'precision_summary.csv'), index=False)
    
    # Also save as a LaTeX table
    with open(os.path.join(output_dir, 'precision_summary.txt'), 'w') as f:
        f.write(summary_table.to_latex(index=False, float_format="%.2f"))
    
    print(f"Precision comparison completed. Results saved to {output_dir}")
    
    return combined_df


# usage:
# python src/test/visualize_root_timing.py --workdir /path/to/gpu_root_timing --compare-precision
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize root timing results")
    parser.add_argument("--csv", default="gpu_root_timing/results_double/summary.csv", 
                        help="Path to the summary CSV file")
    parser.add_argument("--output", default="gpu_root_timing/plots", 
                        help="Directory to save visualization outputs")
    parser.add_argument("--workdir", default=None,
                        help="Base workspace directory for precision comparison")
    parser.add_argument("--compare-precision", action="store_true",
                        help="Compare results across different precision settings")
    
    args = parser.parse_args()
    
    if args.compare_precision and args.workdir:
        compare_precision_settings(args.workdir, os.path.join(args.output, "precision_comparison"))
    else:
        visualize_root_timing(args.csv, args.output) 