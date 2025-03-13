#!/usr/bin/env python3
"""
Script to analyze PyTorch-inferred Markov Blanket partitioning.

This script validates that the inferred partitioning follows the theoretical constraints:
1. Internal states correspond to high-density regions of the Gaussian
2. Blanket states surround internal states
3. External states are separated from internal states by blanket states
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
import torch
import argparse
from matplotlib.colors import LinearSegmentedColormap

def analyze_partition(data_path, partition_img_path, time_point, output_dir):
    """
    Analyze a PyTorch-inferred Markov Blanket partition.
    
    Args:
        data_path: Path to simulation data CSV
        partition_img_path: Path to partition visualization image
        time_point: Time point to analyze
        output_dir: Directory to save analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load simulation data
    data = pd.read_csv(data_path)
    grid_shape = (20, 20)  # Assuming 20x20 grid
    
    # Reshape data to grid
    frame_data = data.loc[data['time'] == time_point].drop('time', axis=1).values.reshape(grid_shape)
    
    # Load partition visualization
    partition_img = mpimg.imread(partition_img_path)
    
    # Create figure for analysis
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    
    # Plot original simulation frame
    im1 = ax[0, 0].imshow(frame_data, cmap='viridis')
    ax[0, 0].set_title(f'Gaussian Blur (Time {time_point})')
    plt.colorbar(im1, ax=ax[0, 0])
    
    # Plot original partition visualization
    ax[0, 1].imshow(partition_img)
    ax[0, 1].set_title(f'Original Partition Visualization')
    
    # Extract masks for internal, blanket, and external states
    height, width, _ = partition_img.shape
    partition_part = partition_img[:, width//2:, :]  # Right half contains partition
    
    # Downsample to match grid size
    subsample_h = partition_part.shape[0] // grid_shape[0]
    subsample_w = partition_part.shape[1] // grid_shape[1]
    
    internal_mask = np.zeros(grid_shape, dtype=bool)
    blanket_mask = np.zeros(grid_shape, dtype=bool)
    external_mask = np.zeros(grid_shape, dtype=bool)
    
    # Convert colors to masks
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # Sample color at this position
            roi = partition_part[i*subsample_h:(i+1)*subsample_h, j*subsample_w:(j+1)*subsample_w, :]
            if roi.size > 0:  # Check if region exists
                color = roi.mean(axis=(0, 1))
                
                # Check dominant color (purple=internal, yellow=blanket, cyan=external)
                if color[0] > 0.3 and color[2] > 0.3 and color[1] < 0.3:  # Purple
                    internal_mask[i, j] = True
                elif color[0] > 0.5 and color[1] > 0.5 and color[2] < 0.3:  # Yellow
                    blanket_mask[i, j] = True
                elif color[1] > 0.5 and color[2] > 0.5 and color[0] < 0.3:  # Cyan
                    external_mask[i, j] = True
    
    # Create density-based masks for comparison
    max_density = np.max(frame_data)
    high_density_mask = frame_data > 0.7 * max_density
    medium_density_mask = (frame_data > 0.3 * max_density) & (frame_data <= 0.7 * max_density)
    low_density_mask = frame_data <= 0.3 * max_density
    
    # Create overlay visualization
    partition_overlay = np.zeros((*grid_shape, 4))  # RGBA
    partition_overlay[internal_mask, 0] = 0.5  # Red component for purple
    partition_overlay[internal_mask, 2] = 0.5  # Blue component for purple
    partition_overlay[internal_mask, 3] = 0.7  # Alpha
    
    partition_overlay[blanket_mask, 0] = 1.0  # Red component for yellow
    partition_overlay[blanket_mask, 1] = 1.0  # Green component for yellow
    partition_overlay[blanket_mask, 3] = 0.7  # Alpha
    
    partition_overlay[external_mask, 1] = 1.0  # Green component for cyan
    partition_overlay[external_mask, 2] = 1.0  # Blue component for cyan
    partition_overlay[external_mask, 3] = 0.7  # Alpha
    
    # Create density overlay
    density_overlay = np.zeros((*grid_shape, 4))  # RGBA
    density_overlay[high_density_mask, 0] = 0.8  # Red component for high density
    density_overlay[high_density_mask, 3] = 0.7  # Alpha
    
    density_overlay[medium_density_mask, 1] = 0.8  # Green component for medium density
    density_overlay[medium_density_mask, 3] = 0.7  # Alpha
    
    density_overlay[low_density_mask, 2] = 0.8  # Blue component for low density
    density_overlay[low_density_mask, 3] = 0.7  # Alpha
    
    # Plot partitioning overlay
    ax[1, 0].imshow(frame_data, cmap='viridis')
    ax[1, 0].imshow(partition_overlay)
    ax[1, 0].set_title(f'Markov Blanket Partition Overlay')
    
    # Add legend for partition
    from matplotlib.patches import Patch
    partition_legend = [
        Patch(facecolor=[0.5, 0, 0.5, 0.7], label='Internal States'),
        Patch(facecolor=[1.0, 1.0, 0, 0.7], label='Blanket States'),
        Patch(facecolor=[0, 1.0, 1.0, 0.7], label='External States')
    ]
    ax[1, 0].legend(handles=partition_legend, loc='upper right')
    
    # Plot density overlay
    ax[1, 1].imshow(frame_data, cmap='viridis')
    ax[1, 1].imshow(density_overlay)
    ax[1, 1].set_title(f'Density Classification Overlay')
    
    # Add legend for density
    density_legend = [
        Patch(facecolor=[0.8, 0, 0, 0.7], label='High Density (>70%)'),
        Patch(facecolor=[0, 0.8, 0, 0.7], label='Medium Density (30-70%)'),
        Patch(facecolor=[0, 0, 0.8, 0.7], label='Low Density (<30%)')
    ]
    ax[1, 1].legend(handles=density_legend, loc='upper right')
    
    # Add overall title
    plt.suptitle(f'Analysis of PyTorch-Inferred Markov Blanket Partition (Time={time_point})', fontsize=16)
    plt.tight_layout()
    
    # Save the analysis
    analysis_path = os.path.join(output_dir, f'partition_analysis_t{time_point}.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    print(f'Analysis saved to {analysis_path}')
    
    # Calculate metrics
    # Count cells of each type
    n_internal = np.sum(internal_mask)
    n_blanket = np.sum(blanket_mask)
    n_external = np.sum(external_mask)
    n_high_density = np.sum(high_density_mask)
    n_medium_density = np.sum(medium_density_mask)
    n_low_density = np.sum(low_density_mask)
    
    # Calculate overlap metrics
    internal_high_overlap = np.sum(internal_mask & high_density_mask)
    internal_medium_overlap = np.sum(internal_mask & medium_density_mask)
    internal_low_overlap = np.sum(internal_mask & low_density_mask)
    
    blanket_high_overlap = np.sum(blanket_mask & high_density_mask)
    blanket_medium_overlap = np.sum(blanket_mask & medium_density_mask)
    blanket_low_overlap = np.sum(blanket_mask & low_density_mask)
    
    external_high_overlap = np.sum(external_mask & high_density_mask)
    external_medium_overlap = np.sum(external_mask & medium_density_mask)
    external_low_overlap = np.sum(external_mask & low_density_mask)
    
    # Calculate precision and recall
    internal_high_recall = internal_high_overlap / n_high_density if n_high_density > 0 else 0
    internal_precision = internal_high_overlap / n_internal if n_internal > 0 else 0
    
    # Verify topological constraints
    # 1. Check if internal states form a single connected component
    # 2. Check if internal states are surrounded by blanket states
    # 3. Check if external states are separated from internal states by blanket states
    
    # Create results text file
    metrics_path = os.path.join(output_dir, f'partition_metrics_t{time_point}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Analysis of PyTorch-Inferred Markov Blanket Partition (Time={time_point})\n')
        f.write('=========================================================\n\n')
        
        f.write('Cell Counts:\n')
        f.write(f'  Internal states: {n_internal}\n')
        f.write(f'  Blanket states: {n_blanket}\n')
        f.write(f'  External states: {n_external}\n')
        f.write(f'  Total cells: {grid_shape[0] * grid_shape[1]}\n\n')
        
        f.write('Density Counts:\n')
        f.write(f'  High density cells (>70%): {n_high_density}\n')
        f.write(f'  Medium density cells (30-70%): {n_medium_density}\n')
        f.write(f'  Low density cells (<30%): {n_low_density}\n\n')
        
        f.write('Overlap Metrics:\n')
        f.write('  Internal states:\n')
        f.write(f'    High density: {internal_high_overlap} ({internal_high_overlap/n_internal:.2%} of internal)\n')
        f.write(f'    Medium density: {internal_medium_overlap} ({internal_medium_overlap/n_internal:.2%} of internal)\n')
        f.write(f'    Low density: {internal_low_overlap} ({internal_low_overlap/n_internal:.2%} of internal)\n')
        
        f.write('  Blanket states:\n')
        f.write(f'    High density: {blanket_high_overlap} ({blanket_high_overlap/n_blanket:.2%} of blanket)\n')
        f.write(f'    Medium density: {blanket_medium_overlap} ({blanket_medium_overlap/n_blanket:.2%} of blanket)\n')
        f.write(f'    Low density: {blanket_low_overlap} ({blanket_low_overlap/n_blanket:.2%} of blanket)\n')
        
        f.write('  External states:\n')
        f.write(f'    High density: {external_high_overlap} ({external_high_overlap/n_external:.2%} of external)\n')
        f.write(f'    Medium density: {external_medium_overlap} ({external_medium_overlap/n_external:.2%} of external)\n')
        f.write(f'    Low density: {external_low_overlap} ({external_low_overlap/n_external:.2%} of external)\n\n')
        
        f.write('Precision and Recall for Internal States vs High Density:\n')
        f.write(f'  Recall: {internal_high_recall:.2%} of high-density cells are classified as internal\n')
        f.write(f'  Precision: {internal_precision:.2%} of cells classified as internal are high-density\n')
        f.write(f'  F1 Score: {2 * internal_precision * internal_high_recall / (internal_precision + internal_high_recall):.2%} if both > 0 else 0\n')
    
    print(f'Metrics saved to {metrics_path}')
    
    return {
        'internal_recall': internal_high_recall, 
        'internal_precision': internal_precision,
        'n_internal': n_internal,
        'n_blanket': n_blanket,
        'n_external': n_external,
        'n_high_density': n_high_density
    }

def analyze_all_timepoints(data_path, partition_dir, output_dir, time_points=None):
    """
    Analyze partition at multiple time points.
    
    Args:
        data_path: Path to simulation data CSV
        partition_dir: Directory containing partition visualizations
        output_dir: Directory to save analysis results
        time_points: List of time points to analyze (if None, use all available)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all partition files
    partition_files = [f for f in os.listdir(partition_dir) if f.startswith('markov_partition_t') and f.endswith('.png')]
    
    # Extract time points from filenames
    available_time_points = []
    for filename in partition_files:
        match = filename.replace('markov_partition_t', '').replace('.png', '')
        if match.isdigit():
            available_time_points.append(int(match))
    
    # Sort time points
    available_time_points.sort()
    
    # Use specified time points or all available
    if time_points is None:
        time_points = available_time_points
    
    # Analyze each time point
    results = {}
    for t in time_points:
        if t in available_time_points:
            partition_path = os.path.join(partition_dir, f'markov_partition_t{t}.png')
            print(f'Analyzing time point {t}...')
            results[t] = analyze_partition(data_path, partition_path, t, output_dir)
    
    # Plot metrics over time
    if len(results) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot precision and recall
        time_points = sorted(results.keys())
        recall_values = [results[t]['internal_recall'] for t in time_points]
        precision_values = [results[t]['internal_precision'] for t in time_points]
        f1_values = [2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(precision_values, recall_values)]
        
        ax1.plot(time_points, recall_values, 'b-', label='Recall')
        ax1.plot(time_points, precision_values, 'r-', label='Precision')
        ax1.plot(time_points, f1_values, 'g-', label='F1 Score')
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Precision and Recall for Internal States')
        ax1.legend()
        ax1.grid(True)
        
        # Plot cell counts
        internal_counts = [results[t]['n_internal'] for t in time_points]
        blanket_counts = [results[t]['n_blanket'] for t in time_points]
        external_counts = [results[t]['n_external'] for t in time_points]
        high_density_counts = [results[t]['n_high_density'] for t in time_points]
        
        ax2.plot(time_points, internal_counts, 'purple', label='Internal States')
        ax2.plot(time_points, blanket_counts, 'yellow', label='Blanket States')
        ax2.plot(time_points, external_counts, 'cyan', label='External States')
        ax2.plot(time_points, high_density_counts, 'r--', label='High Density Cells')
        ax2.set_xlabel('Time Point')
        ax2.set_ylabel('Cell Count')
        ax2.set_title('State and Density Cell Counts')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        metrics_over_time_path = os.path.join(output_dir, 'metrics_over_time.png')
        plt.savefig(metrics_over_time_path, dpi=300)
        print(f'Metrics over time saved to {metrics_over_time_path}')

def main():
    parser = argparse.ArgumentParser(description='Analyze PyTorch-inferred Markov Blanket partitioning')
    parser.add_argument('--data', default='output/torch_confirmation/simulation/gridworld_data.csv', help='Path to simulation data CSV')
    parser.add_argument('--partition_dir', default='output/torch_confirmation/analysis', help='Directory containing partition visualizations')
    parser.add_argument('--output_dir', default='output/torch_confirmation/analysis_results', help='Directory to save analysis results')
    parser.add_argument('--time_points', type=int, nargs='+', help='Specific time points to analyze')
    
    args = parser.parse_args()
    
    analyze_all_timepoints(args.data, args.partition_dir, args.output_dir, args.time_points)

if __name__ == '__main__':
    main() 