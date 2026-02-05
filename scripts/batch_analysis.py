#!/usr/bin/env python3
"""
Factoscope Batch Analysis & Visualization
Similar functionality to lm_plots.py but for testing Factoscope model on multiple questions
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import pandas as pd

# Import inference engine
from factoscope_inference import FactoscopeInference

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


class FactoscopeAnalyzer:
    """Analyzer for batch testing and visualization of Factoscope results"""

    def __init__(self, results: List[Dict]):
        """
        Initialize analyzer with results

        Args:
            results: List of prediction results from Factoscope
        """
        self.results = results
        self.df = pd.DataFrame(results)

    def plot_confidence_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of confidence scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(self.df['confidence'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Confidence Scores', fontsize=14)
        ax1.axvline(self.df['confidence'].mean(), color='red', linestyle='--',
                   label=f'Mean: {self.df["confidence"].mean():.3f}')
        ax1.axvline(self.df['confidence'].median(), color='green', linestyle='--',
                   label=f'Median: {self.df["confidence"].median():.3f}')
        ax1.legend()

        # Box plot
        ax2.boxplot([self.df['confidence']], labels=['Confidence'])
        ax2.set_ylabel('Confidence Score', fontsize=12)
        ax2.set_title('Confidence Score Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confidence distribution to: {save_path}")

        plt.show()

    def plot_prediction_accuracy(self, save_path: Optional[str] = None):
        """Plot prediction accuracy breakdown"""
        if 'prediction' not in self.df.columns:
            print("No prediction data available")
            return

        predictions = self.df['prediction'].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        colors = ['#2ecc71' if x == 'correct' else '#e74c3c' for x in predictions.index]
        predictions.plot(kind='bar', ax=ax1, color=colors, edgecolor='black')
        ax1.set_xlabel('Prediction', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Prediction Distribution', fontsize=14)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

        # Add counts on bars
        for i, v in enumerate(predictions):
            ax1.text(i, v + 0.5, str(v), ha='center', fontsize=11, fontweight='bold')

        # Pie chart
        ax2.pie(predictions, labels=predictions.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 12})
        ax2.set_title('Prediction Breakdown', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved prediction accuracy to: {save_path}")

        plt.show()

    def plot_confidence_vs_prediction(self, save_path: Optional[str] = None):
        """Plot confidence scores grouped by prediction"""
        if 'prediction' not in self.df.columns:
            print("No prediction data available")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create violin plot
        correct_conf = self.df[self.df['prediction'] == 'correct']['confidence']
        false_conf = self.df[self.df['prediction'] == 'false']['confidence']

        # Filter out empty arrays
        data_to_plot = []
        positions = []
        labels = []
        colors = []

        if len(correct_conf) > 0:
            data_to_plot.append(correct_conf)
            positions.append(len(positions))
            labels.append('Correct')
            colors.append('#2ecc71')

        if len(false_conf) > 0:
            data_to_plot.append(false_conf)
            positions.append(len(positions))
            labels.append('False')
            colors.append('#e74c3c')

        if len(data_to_plot) == 0:
            print("No data to plot")
            plt.close(fig)
            return

        parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)

        # Color the violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Confidence Distribution by Prediction Type', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confidence vs prediction to: {save_path}")

        plt.show()

    def plot_top_prob_vs_confidence(self, save_path: Optional[str] = None):
        """Scatter plot of top token probability vs confidence"""
        fig, ax = plt.subplots(figsize=(10, 6))

        if 'prediction' in self.df.columns:
            correct_mask = self.df['prediction'] == 'correct'
            ax.scatter(self.df[correct_mask]['top_prob'],
                      self.df[correct_mask]['confidence'],
                      c='#2ecc71', label='Correct', alpha=0.6, s=100, edgecolors='black')
            ax.scatter(self.df[~correct_mask]['top_prob'],
                      self.df[~correct_mask]['confidence'],
                      c='#e74c3c', label='False', alpha=0.6, s=100, edgecolors='black')
            ax.legend(fontsize=11)
        else:
            ax.scatter(self.df['top_prob'], self.df['confidence'],
                      alpha=0.6, s=100, edgecolors='black')

        ax.set_xlabel('Top Token Probability', fontsize=12)
        ax.set_ylabel('Factoscope Confidence', fontsize=12)
        ax.set_title('Top Token Probability vs Factoscope Confidence', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved top_prob vs confidence to: {save_path}")

        plt.show()

    def plot_distance_heatmap(self, save_path: Optional[str] = None):
        """Heatmap showing nearest correct vs false distances"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create scatter plot
        if 'prediction' in self.df.columns:
            correct_mask = self.df['prediction'] == 'correct'
            scatter1 = ax.scatter(self.df[correct_mask]['nearest_correct_distance'],
                                 self.df[correct_mask]['nearest_false_distance'],
                                 c=self.df[correct_mask]['confidence'],
                                 cmap='Greens', s=100, edgecolors='black',
                                 alpha=0.7, label='Predicted Correct')
            scatter2 = ax.scatter(self.df[~correct_mask]['nearest_correct_distance'],
                                 self.df[~correct_mask]['nearest_false_distance'],
                                 c=self.df[~correct_mask]['confidence'],
                                 cmap='Reds', s=100, edgecolors='black',
                                 alpha=0.7, label='Predicted False')
        else:
            scatter1 = ax.scatter(self.df['nearest_correct_distance'],
                                 self.df['nearest_false_distance'],
                                 c=self.df['confidence'],
                                 cmap='viridis', s=100, edgecolors='black', alpha=0.7)

        # Add diagonal line
        max_val = max(self.df['nearest_correct_distance'].max(),
                     self.df['nearest_false_distance'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal Distance')

        ax.set_xlabel('Distance to Nearest Correct', fontsize=12)
        ax.set_ylabel('Distance to Nearest False', fontsize=12)
        ax.set_title('Nearest Neighbor Distances', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter1, ax=ax)
        cbar.set_label('Confidence', fontsize=11)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved distance heatmap to: {save_path}")

        plt.show()

    def plot_rank_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of token ranks"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Log scale for better visualization
        log_ranks = np.log10(self.df['rank'] + 1)

        if 'prediction' in self.df.columns:
            correct_ranks = log_ranks[self.df['prediction'] == 'correct']
            false_ranks = log_ranks[self.df['prediction'] == 'false']

            ax.hist([correct_ranks, false_ranks], bins=20, label=['Correct', 'False'],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            ax.legend(fontsize=11)
        else:
            ax.hist(log_ranks, bins=20, alpha=0.7, edgecolor='black', color='steelblue')

        ax.set_xlabel('Token Rank (log10)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of First Token Ranks', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved rank distribution to: {save_path}")

        plt.show()

    def plot_neighbor_breakdown(self, save_path: Optional[str] = None):
        """Stacked bar chart of k-nearest neighbor breakdown"""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Extract neighbor counts
        correct_neighbors = [r['top_k_neighbors']['correct'] for r in self.results]
        false_neighbors = [r['top_k_neighbors']['false'] for r in self.results]

        x = np.arange(len(self.results))
        width = 0.8

        # Create stacked bars
        p1 = ax.bar(x, correct_neighbors, width, label='Correct Neighbors',
                   color='#2ecc71', edgecolor='black')
        p2 = ax.bar(x, false_neighbors, width, bottom=correct_neighbors,
                   label='False Neighbors', color='#e74c3c', edgecolor='black')

        ax.set_xlabel('Question Index', fontsize=12)
        ax.set_ylabel('Number of Neighbors', fontsize=12)
        ax.set_title('K-Nearest Neighbors Breakdown (k=10)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Limit x-axis if too many questions
        if len(self.results) > 50:
            ax.set_xlim(-1, 50)
            ax.set_xlabel('Question Index (showing first 50)', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved neighbor breakdown to: {save_path}")

        plt.show()

    def plot_all(self, output_dir: str = './plots'):
        """Generate all plots and save to directory"""
        os.makedirs(output_dir, exist_ok=True)

        print("\nGenerating all plots...")

        self.plot_confidence_distribution(os.path.join(output_dir, 'confidence_distribution.png'))
        self.plot_prediction_accuracy(os.path.join(output_dir, 'prediction_accuracy.png'))
        self.plot_confidence_vs_prediction(os.path.join(output_dir, 'confidence_vs_prediction.png'))
        self.plot_top_prob_vs_confidence(os.path.join(output_dir, 'top_prob_vs_confidence.png'))
        self.plot_distance_heatmap(os.path.join(output_dir, 'distance_heatmap.png'))
        self.plot_rank_distribution(os.path.join(output_dir, 'rank_distribution.png'))
        self.plot_neighbor_breakdown(os.path.join(output_dir, 'neighbor_breakdown.png'))

        print(f"\n✓ All plots saved to: {output_dir}")

    def print_summary_statistics(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)

        print(f"\nTotal Questions Tested: {len(self.results)}")

        print(f"\nConfidence Statistics:")
        print(f"  Mean:   {self.df['confidence'].mean():.4f}")
        print(f"  Median: {self.df['confidence'].median():.4f}")
        print(f"  Std:    {self.df['confidence'].std():.4f}")
        print(f"  Min:    {self.df['confidence'].min():.4f}")
        print(f"  Max:    {self.df['confidence'].max():.4f}")

        if 'prediction' in self.df.columns:
            print(f"\nPrediction Breakdown:")
            pred_counts = self.df['prediction'].value_counts()
            for pred, count in pred_counts.items():
                print(f"  {pred.capitalize()}: {count} ({count/len(self.df)*100:.1f}%)")

            print(f"\nConfidence by Prediction:")
            for pred in ['correct', 'false']:
                if pred in self.df['prediction'].values:
                    subset = self.df[self.df['prediction'] == pred]['confidence']
                    print(f"  {pred.capitalize()}: Mean={subset.mean():.4f}, Median={subset.median():.4f}")

        print(f"\nTop Token Probability:")
        print(f"  Mean:   {self.df['top_prob'].mean():.4f}")
        print(f"  Median: {self.df['top_prob'].median():.4f}")

        print(f"\nToken Rank:")
        print(f"  Mean:   {self.df['rank'].mean():.2f}")
        print(f"  Median: {self.df['rank'].median():.2f}")
        print(f"  Rank 0 (top token): {(self.df['rank'] == 0).sum()} ({(self.df['rank'] == 0).sum()/len(self.df)*100:.1f}%)")

        print("\n" + "="*70 + "\n")


def load_dataset(dataset_path: str, limit: Optional[int] = None) -> List[str]:
    """
    Load questions from a dataset file

    Args:
        dataset_path: Path to dataset JSON file
        limit: Maximum number of questions to load

    Returns:
        List of question prompts
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    prompts = [item['prompt'] for item in data]

    if limit:
        prompts = prompts[:limit]

    return prompts


def main():
    parser = argparse.ArgumentParser(description='Factoscope Batch Analysis & Visualization')

    # Model paths
    parser.add_argument('--model_path', type=str, default='./models/Meta-Llama-3-8B',
                       help='Path to LLM')
    parser.add_argument('--factoscope_model', type=str, default='./best_factoscope_model.pt',
                       help='Path to trained factoscope model')
    parser.add_argument('--processed_data', type=str, default='./processed_data.h5',
                       help='Path to processed training data')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')

    # Data sources
    parser.add_argument('--dataset', type=str,
                       help='Path to dataset JSON file from dataset_train folder')
    parser.add_argument('--dataset_dir', type=str,
                       default='./llm_factoscope-main/dataset_train',
                       help='Directory containing datasets')
    parser.add_argument('--dataset_name', type=str,
                       help='Name of dataset to use (e.g., city_country_dataset.json)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of questions to test')
    parser.add_argument('--results_file', type=str,
                       help='Load existing results JSON instead of running inference')

    # Output
    parser.add_argument('--output', type=str, default='batch_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--plot_dir', type=str, default='./plots',
                       help='Directory to save plots')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')

    args = parser.parse_args()

    # Load or generate results
    if args.results_file:
        print(f"Loading results from: {args.results_file}")
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        print(f"✓ Loaded {len(results)} results")
    else:
        # Determine dataset path
        if args.dataset:
            dataset_path = args.dataset
        elif args.dataset_name:
            dataset_path = os.path.join(args.dataset_dir, args.dataset_name)
        else:
            # List available datasets
            dataset_dir = Path(args.dataset_dir)
            if dataset_dir.exists():
                datasets = list(dataset_dir.glob('*.json'))
                print("\nAvailable datasets:")
                for i, ds in enumerate(datasets, 1):
                    print(f"  {i}. {ds.name}")

                choice = input("\nSelect dataset number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return

                try:
                    idx = int(choice) - 1
                    dataset_path = str(datasets[idx])
                except (ValueError, IndexError):
                    print("Invalid selection")
                    return
            else:
                print(f"Error: Dataset directory not found: {args.dataset_dir}")
                return

        # Load questions
        print(f"\nLoading questions from: {dataset_path}")
        prompts = load_dataset(dataset_path, limit=args.limit)
        print(f"✓ Loaded {len(prompts)} questions")

        # Initialize inference engine
        print("\nInitializing Factoscope inference engine...")
        engine = FactoscopeInference(
            model_path=args.model_path,
            factoscope_model_path=args.factoscope_model,
            processed_data_path=args.processed_data,
            device=args.device
        )

        # Run batch inference
        print(f"\nRunning inference on {len(prompts)} questions...")
        results = engine.batch_predict(prompts)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")

    # Analyze and visualize
    analyzer = FactoscopeAnalyzer(results)
    analyzer.print_summary_statistics()

    if not args.no_plots:
        analyzer.plot_all(output_dir=args.plot_dir)

    print("\n✓ Analysis complete!\n")


if __name__ == '__main__':
    main()
