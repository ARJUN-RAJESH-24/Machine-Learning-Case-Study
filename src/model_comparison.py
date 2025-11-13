"""
Performance Comparison and Reporting Module
Aggregates results across datasets and models with visualizations
"""
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150


def collect_all_metrics(results_dir: Path = Path("results/performance_reports")) -> pd.DataFrame:
    """
    Collect all metrics from JSON files into a single DataFrame
    
    Args:
        results_dir: Directory containing metric files
    
    Returns:
        DataFrame with all metrics
    """
    all_data = []
    
    # Find all JSON metric files
    for json_file in results_dir.glob("*__*__metrics.json"):
        try:
            # Parse filename: dataset__model__metrics.json
            parts = json_file.stem.split("__")
            if len(parts) < 3:
                continue
            
            dataset = parts[0]
            model = parts[1]
            
            # Load metrics
            with open(json_file, 'r') as f:
                metrics = json.load(f)
            
            # Extract main metrics
            row = {
                'dataset': dataset,
                'model': model,
                'accuracy': metrics.get('accuracy', np.nan),
                'precision_macro': metrics.get('precision_macro', np.nan),
                'precision_weighted': metrics.get('precision_weighted', np.nan),
                'recall_macro': metrics.get('recall_macro', np.nan),
                'recall_weighted': metrics.get('recall_weighted', np.nan),
                'f1_macro': metrics.get('f1_macro', np.nan),
                'f1_weighted': metrics.get('f1_weighted', np.nan),
                'matthews_corrcoef': metrics.get('matthews_corrcoef', np.nan),
                'cohen_kappa': metrics.get('cohen_kappa', np.nan),
                'cv_f1_mean': metrics.get('cv_f1_macro_mean', np.nan),
                'cv_f1_std': metrics.get('cv_f1_macro_std', np.nan)
            }
            
            # Add ROC AUC if available
            if 'roc_auc' in metrics:
                row['roc_auc'] = metrics['roc_auc']
            elif 'roc_auc_ovr' in metrics:
                row['roc_auc'] = metrics['roc_auc_ovr']
            
            all_data.append(row)
            
        except Exception as e:
            print(f"⚠️ Error reading {json_file}: {e}")
    
    if not all_data:
        print("⚠️ No metric files found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    print(f"✅ Collected metrics from {len(df)} experiments")
    return df


def plot_model_comparison(df: pd.DataFrame, output_path: Path, 
                         metric: str = 'f1_macro'):
    """
    Plot comparison of models across datasets
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        metric: Metric to compare
    """
    if df.empty:
        return
    
    # Pivot for grouped bar chart
    pivot = df.pivot(index='model', columns='dataset', values=metric)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance Comparison: {metric.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved comparison → {output_path}")


def plot_dataset_comparison(df: pd.DataFrame, output_path: Path, 
                           metric: str = 'f1_macro'):
    """
    Plot comparison of datasets across models
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        metric: Metric to compare
    """
    if df.empty:
        return
    
    # Pivot for grouped bar chart
    pivot = df.pivot(index='dataset', columns='model', values=metric)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'Dataset Performance Comparison: {metric.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved comparison → {output_path}")


def plot_heatmap(df: pd.DataFrame, output_path: Path, 
                metric: str = 'f1_macro'):
    """
    Plot heatmap of model performance across datasets
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        metric: Metric to visualize
    """
    if df.empty:
        return
    
    # Pivot for heatmap
    pivot = df.pivot(index='model', columns='dataset', values=metric)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', 
                ax=ax, cbar_kws={'label': metric.replace('_', ' ').title()},
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Heatmap: {metric.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved heatmap → {output_path}")


def plot_metric_comparison(df: pd.DataFrame, output_path: Path, 
                          metrics: List[str] = None):
    """
    Compare multiple metrics for each model-dataset combination
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        metrics: List of metrics to compare
    """
    if df.empty:
        return
    
    if metrics is None:
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    # Filter available metrics
    metrics = [m for m in metrics if m in df.columns]
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        pivot = df.pivot(index='model', columns='dataset', values=metric)
        pivot.plot(kind='bar', ax=ax)
        
        ax.set_title(metric.replace('_', ' ').title(), 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.legend(title='Dataset', fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved multi-metric comparison → {output_path}")


def generate_ranking_table(df: pd.DataFrame, output_path: Path):
    """
    Generate ranking table showing best models per dataset
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
    """
    if df.empty:
        return
    
    rankings = []
    
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset].copy()
        
        # Rank by F1 macro
        subset = subset.sort_values('f1_macro', ascending=False)
        
        rankings.append({
            'dataset': dataset,
            'best_model': subset.iloc[0]['model'],
            'best_f1': subset.iloc[0]['f1_macro'],
            'second_model': subset.iloc[1]['model'] if len(subset) > 1 else None,
            'second_f1': subset.iloc[1]['f1_macro'] if len(subset) > 1 else None,
            'worst_model': subset.iloc[-1]['model'],
            'worst_f1': subset.iloc[-1]['f1_macro']
        })
    
    ranking_df = pd.DataFrame(rankings)
    ranking_df.to_csv(output_path, index=False)
    print(f"📊 Saved ranking table → {output_path}")
    
    # Print rankings
    print("\n" + "="*60)
    print("🏆 MODEL RANKINGS BY DATASET")
    print("="*60)
    for _, row in ranking_df.iterrows():
        print(f"\n{row['dataset'].upper()}:")
        print(f"  🥇 Best:  {row['best_model'].upper()} (F1: {row['best_f1']:.4f})")
        if row['second_model']:
            print(f"  🥈 2nd:   {row['second_model'].upper()} (F1: {row['second_f1']:.4f})")
        print(f"  📉 Worst: {row['worst_model'].upper()} (F1: {row['worst_f1']:.4f})")
    print("="*60 + "\n")


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """
    Generate comprehensive summary report
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
    """
    if df.empty:
        return
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE PERFORMANCE SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Experiments: {len(df)}\n")
        f.write(f"Datasets: {', '.join(df['dataset'].unique())}\n")
        f.write(f"Models: {', '.join(df['model'].unique())}\n\n")
        
        # Best overall performance
        best_idx = df['f1_macro'].idxmax()
        best = df.loc[best_idx]
        f.write("BEST OVERALL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Model: {best['model'].upper()}\n")
        f.write(f"Dataset: {best['dataset']}\n")
        f.write(f"F1-Score (macro): {best['f1_macro']:.4f}\n")
        f.write(f"Accuracy: {best['accuracy']:.4f}\n\n")
        
        # Average performance by model
        f.write("AVERAGE PERFORMANCE BY MODEL\n")
        f.write("-"*70 + "\n")
        model_avg = df.groupby('model')[['accuracy', 'f1_macro']].mean()
        f.write(model_avg.to_string())
        f.write("\n\n")
        
        # Average performance by dataset
        f.write("AVERAGE PERFORMANCE BY DATASET\n")
        f.write("-"*70 + "\n")
        dataset_avg = df.groupby('dataset')[['accuracy', 'f1_macro']].mean()
        f.write(dataset_avg.to_string())
        f.write("\n\n")
        
        # Detailed results table
        f.write("DETAILED RESULTS\n")
        f.write("-"*70 + "\n")
        summary_cols = ['dataset', 'model', 'accuracy', 'precision_macro', 
                       'recall_macro', 'f1_macro']
        summary_cols = [c for c in summary_cols if c in df.columns]
        f.write(df[summary_cols].to_string(index=False))
        f.write("\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"📄 Saved summary report → {output_path}")


def main():
    """Main function to generate all comparisons and reports"""
    print("\n" + "="*60)
    print("📊 GENERATING PERFORMANCE COMPARISONS")
    print("="*60 + "\n")
    
    # Collect metrics
    df = collect_all_metrics()
    
    if df.empty:
        print("❌ No data to process!")
        return
    
    # Output directory
    output_dir = Path("results/performance_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated data
    df.to_csv(output_dir / "all_metrics.csv", index=False)
    print(f"💾 Saved aggregated metrics → {output_dir / 'all_metrics.csv'}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    plot_model_comparison(df, output_dir / "model_comparison_f1.png", 'f1_macro')
    plot_dataset_comparison(df, output_dir / "dataset_comparison_f1.png", 'f1_macro')
    plot_heatmap(df, output_dir / "performance_heatmap.png", 'f1_macro')
    plot_metric_comparison(df, output_dir / "multi_metric_comparison.png")
    
    # Generate reports
    print("\n📄 Generating reports...")
    
    generate_ranking_table(df, output_dir / "model_rankings.csv")
    generate_summary_report(df, output_dir / "summary_report.txt")
    
    print("\n✅ All comparisons and reports generated!")
    print(f"📁 Check {output_dir} for outputs\n")


if __name__ == "__main__":
    main()
