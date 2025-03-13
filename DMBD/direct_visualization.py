#!/usr/bin/env python3
"""
Script to directly demonstrate PyTorch-based inference of Markov Blanket partitioning.

This script:
1. Creates a small gridworld simulation with a moving Gaussian blur
2. Uses PyTorch to infer the Markov Blanket partitioning (internal, blanket, external)
3. Creates visualizations showing the relationship between the Gaussian density and the inferred partitioning
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import argparse

def create_gaussian_gridworld(grid_size=(20, 20), n_time_points=10, radius=8.0, sigma=1.5):
    """
    Create a gridworld simulation with a moving Gaussian blur.
    
    Args:
        grid_size: Size of the grid (height, width)
        n_time_points: Number of time points to simulate
        radius: Radius of circular path for Gaussian
        sigma: Standard deviation of Gaussian
        
    Returns:
        Tuple of (simulation_data, dataframe)
    """
    print(f"Creating gridworld simulation...")
    print(f"  Grid size: {grid_size[0]}x{grid_size[1]}")
    print(f"  Time points: {n_time_points}")
    print(f"  Radius: {radius}")
    print(f"  Sigma: {sigma}")
    
    # Center of the grid
    center_y, center_x = grid_size[0] // 2, grid_size[1] // 2
    
    # Create the data
    simulation_data = []
    for t in range(n_time_points):
        # Calculate position of Gaussian at this time point
        angle = 2 * np.pi * t / n_time_points
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Create grid
        grid = np.zeros(grid_size)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Calculate distance from current position
                d = np.sqrt((i - y)**2 + (j - x)**2)
                # Apply Gaussian formula
                grid[i, j] = np.exp(-d**2 / (2 * sigma**2))
        
        simulation_data.append(grid)
    
    # Convert to tensor
    tensor_data = torch.tensor(np.array(simulation_data), dtype=torch.float32)
    
    # Create DataFrame
    df_data = []
    for t in range(n_time_points):
        flat_grid = simulation_data[t].flatten()
        row = {'time': t}
        for i in range(len(flat_grid)):
            row[f'cell_{i // grid_size[1]}_{i % grid_size[1]}'] = flat_grid[i]
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    print(f"Simulation created.")
    return simulation_data, df, tensor_data

def infer_markov_blanket(simulation_data, grid_size, time_point=None):
    """
    Infer Markov Blanket partitioning using PyTorch.
    
    Args:
        simulation_data: List of 2D grids with simulation data
        grid_size: Size of the grid (height, width)
        time_point: Specific time point to analyze (if None, analyzes all)
        
    Returns:
        Dictionary with probability tensors for each partition
    """
    print(f"Inferring Markov Blanket partitioning...")
    
    # If specific time point, just analyze that one
    if time_point is not None:
        frames = [simulation_data[time_point]]
        time_points = [time_point]
    else:
        # Analyze all time points
        frames = simulation_data
        time_points = list(range(len(simulation_data)))
    
    # Results will store the partition probabilities for each time point
    results = {
        'internal': [],
        'blanket': [],
        'external': []
    }
    
    # Process each time point
    for t, frame in zip(time_points, frames):
        # Flatten grid for easier processing
        flat_data = frame.flatten()
        
        # Step 1: Initialize with density-based priors
        # Higher density -> more likely to be internal
        # Lower density -> more likely to be external
        internal_prior = torch.tensor(flat_data, dtype=torch.float32)
        
        # Normalize to [0,1]
        internal_prior = (internal_prior - internal_prior.min()) / (internal_prior.max() - internal_prior.min() + 1e-8)
        
        # External is inverse of internal
        external_prior = 1.0 - internal_prior
        
        # Blanket is highest in the transition zone between high and low density
        blanket_prior = 4.0 * internal_prior * external_prior  # Creates a peak at 0.5 density
        
        # Reshape to grid
        internal_grid = internal_prior.reshape(grid_size)
        blanket_grid = blanket_prior.reshape(grid_size)
        external_grid = external_prior.reshape(grid_size)
        
        # Step 2: Apply spatial smoothing to enforce connectivity constraints
        # Create smoothing kernel size
        kernel_size = min(5, min(grid_size[0], grid_size[1]) // 2)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size for symmetric padding
        
        # Gaussian smoothing for internal states
        internal_smoothed = torch.nn.functional.avg_pool2d(
            internal_grid.reshape(1, 1, *grid_size),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        ).squeeze()
        
        # Ensure single connected component for internal
        # This favors cells that have high density AND are surrounded by high density
        internal_prob = internal_grid * internal_smoothed
        internal_prob = internal_prob.flatten()
        internal_prob = internal_prob / (internal_prob.sum() + 1e-8)
        
        # Similar for external, but favoring low density regions
        external_smoothed = torch.nn.functional.avg_pool2d(
            external_grid.reshape(1, 1, *grid_size),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        ).squeeze()
        
        external_prob = external_grid * external_smoothed
        external_prob = external_prob.flatten()
        external_prob = external_prob / (external_prob.sum() + 1e-8)
        
        # Blanket should be at the boundary between internal and external
        # We can compute gradient magnitude of the internal probability
        # High gradient = boundary = blanket
        dx = torch.tensor(np.gradient(internal_grid.numpy(), axis=0))
        dy = torch.tensor(np.gradient(internal_grid.numpy(), axis=1))
        gradient_mag = torch.sqrt(dx**2 + dy**2)
        
        # Normalize gradient magnitude
        gradient_mag = gradient_mag / (gradient_mag.max() + 1e-8)
        
        # Blanket probability is proportional to gradient magnitude
        blanket_prob = gradient_mag.flatten()
        blanket_prob = blanket_prob / (blanket_prob.sum() + 1e-8)
        
        # Step 3: Enforce partition constraints
        # Normalize to ensure probabilities sum to 1 for each cell
        partition_sum = internal_prob + blanket_prob + external_prob
        internal_prob = internal_prob / partition_sum
        blanket_prob = blanket_prob / partition_sum
        external_prob = external_prob / partition_sum
        
        # Store results for this time point
        results['internal'].append(internal_prob)
        results['blanket'].append(blanket_prob)
        results['external'].append(external_prob)
    
    # Convert lists to tensors
    for key in results:
        results[key] = torch.stack(results[key])
    
    print(f"Partitioning complete for {len(time_points)} time points.")
    return results

def create_partition_visualization(simulation_data, partition_probs, grid_size, time_point, output_path=None):
    """
    Create visualization of the inferred partitioning.
    
    Args:
        simulation_data: List of 2D grids with simulation data
        partition_probs: Dictionary with partition probabilities
        grid_size: Size of the grid (height, width)
        time_point: Time point to visualize
        output_path: Path to save visualization (if None, doesn't save)
        
    Returns:
        Matplotlib figure
    """
    # Get data for this time point
    frame_data = simulation_data[time_point]
    
    # Get probabilities for this time point
    internal_prob = partition_probs['internal'][time_point]
    blanket_prob = partition_probs['blanket'][time_point]
    external_prob = partition_probs['external'][time_point]
    
    # Reshape to grid
    internal_grid = internal_prob.reshape(grid_size)
    blanket_grid = blanket_prob.reshape(grid_size)
    external_grid = external_prob.reshape(grid_size)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Gaussian density
    im0 = axes[0, 0].imshow(frame_data, cmap='viridis')
    axes[0, 0].set_title(f'Gaussian Density (Time {time_point})')
    fig.colorbar(im0, ax=axes[0, 0])
    
    # Plot internal probability
    im1 = axes[0, 1].imshow(internal_grid, cmap='Reds')
    axes[0, 1].set_title('Internal State Probability')
    fig.colorbar(im1, ax=axes[0, 1])
    
    # Plot blanket probability
    im2 = axes[1, 0].imshow(blanket_grid, cmap='Greens')
    axes[1, 0].set_title('Blanket State Probability')
    fig.colorbar(im2, ax=axes[1, 0])
    
    # Plot external probability
    im3 = axes[1, 1].imshow(external_grid, cmap='Blues')
    axes[1, 1].set_title('External State Probability')
    fig.colorbar(im3, ax=axes[1, 1])
    
    # Add overall title
    plt.suptitle(f'PyTorch-Inferred Markov Blanket Partition (Time={time_point})', fontsize=16)
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    return fig

def create_combined_visualization(simulation_data, partition_probs, grid_size, time_point, output_path=None):
    """
    Create visualization combining the density and partition information.
    
    Args:
        simulation_data: List of 2D grids with simulation data
        partition_probs: Dictionary with partition probabilities
        grid_size: Size of the grid (height, width)
        time_point: Time point to visualize
        output_path: Path to save visualization (if None, doesn't save)
        
    Returns:
        Matplotlib figure
    """
    # Get data for this time point
    frame_data = simulation_data[time_point]
    
    # Get probabilities for this time point
    internal_prob = partition_probs['internal'][time_point]
    blanket_prob = partition_probs['blanket'][time_point]
    external_prob = partition_probs['external'][time_point]
    
    # Reshape to grid
    internal_grid = internal_prob.reshape(grid_size)
    blanket_grid = blanket_prob.reshape(grid_size)
    external_grid = external_prob.reshape(grid_size)
    
    # Determine the dominant partition for each cell
    # 0 = internal, 1 = blanket, 2 = external
    dominant_partition = torch.argmax(torch.stack([internal_grid.flatten(), 
                                                   blanket_grid.flatten(), 
                                                   external_grid.flatten()]), dim=0)
    dominant_partition = dominant_partition.reshape(grid_size)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot original density
    im1 = ax1.imshow(frame_data, cmap='viridis')
    ax1.set_title(f'Gaussian Density (Time {time_point})')
    fig.colorbar(im1, ax=ax1)
    
    # Create custom colormap for partitions
    # Purple = internal, Yellow = blanket, Cyan = external
    custom_cmap = LinearSegmentedColormap.from_list(
        'partition_cmap', 
        [(0.5, 0, 0.5),     # Purple for internal
         (1.0, 1.0, 0),     # Yellow for blanket 
         (0, 1.0, 1.0)],    # Cyan for external
        N=3
    )
    
    # Plot dominant partition
    im2 = ax2.imshow(dominant_partition, cmap=custom_cmap, vmin=0, vmax=2)
    ax2.set_title('Dominant Partition (Purple=Internal, Yellow=Blanket, Cyan=External)')
    
    # Add overall title
    plt.suptitle(f'Gaussian Density and Dominant Partition (Time={time_point})', fontsize=16)
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    return fig

def create_overlay_visualization(simulation_data, partition_probs, grid_size, time_point, output_path=None):
    """
    Create visualization overlaying partition on density.
    
    Args:
        simulation_data: List of 2D grids with simulation data
        partition_probs: Dictionary with partition probabilities
        grid_size: Size of the grid (height, width)
        time_point: Time point to visualize
        output_path: Path to save visualization (if None, doesn't save)
        
    Returns:
        Matplotlib figure
    """
    # Get data for this time point
    frame_data = simulation_data[time_point]
    
    # Get probabilities for this time point
    internal_prob = partition_probs['internal'][time_point]
    blanket_prob = partition_probs['blanket'][time_point]
    external_prob = partition_probs['external'][time_point]
    
    # Reshape to grid
    internal_grid = internal_prob.reshape(grid_size)
    blanket_grid = blanket_prob.reshape(grid_size)
    external_grid = external_prob.reshape(grid_size)
    
    # Determine the dominant partition for each cell
    # 0 = internal, 1 = blanket, 2 = external
    probs = torch.stack([internal_grid.flatten(), blanket_grid.flatten(), external_grid.flatten()])
    dominant_partition = torch.argmax(probs, dim=0).reshape(grid_size)
    
    # Create mask for each partition
    internal_mask = (dominant_partition == 0).numpy()
    blanket_mask = (dominant_partition == 1).numpy()
    external_mask = (dominant_partition == 2).numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot density as background
    ax.imshow(frame_data, cmap='gray')
    
    # Create RGBA overlay
    overlay = np.zeros((*grid_size, 4))  # RGBA
    
    # Internal (purple)
    overlay[internal_mask, 0] = 0.5
    overlay[internal_mask, 2] = 0.5
    overlay[internal_mask, 3] = 0.6
    
    # Blanket (yellow)
    overlay[blanket_mask, 0] = 1.0
    overlay[blanket_mask, 1] = 1.0
    overlay[blanket_mask, 3] = 0.6
    
    # External (cyan)
    overlay[external_mask, 1] = 1.0
    overlay[external_mask, 2] = 1.0
    overlay[external_mask, 3] = 0.3  # Lower alpha for external
    
    # Add overlay
    ax.imshow(overlay)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.5, 0, 0.5, 0.6), label='Internal States'),
        Patch(facecolor=(1.0, 1.0, 0, 0.6), label='Blanket States'),
        Patch(facecolor=(0, 1.0, 1.0, 0.3), label='External States')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add title
    ax.set_title(f'Markov Blanket Partition Overlay (Time={time_point})')
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    return fig

def create_animation(simulation_data, partition_probs, grid_size, output_path):
    """
    Create animation showing the evolution of the partition over time.
    
    Args:
        simulation_data: List of 2D grids with simulation data
        partition_probs: Dictionary with partition probabilities
        grid_size: Size of the grid (height, width)
        output_path: Path to save animation
    """
    print(f"Creating animation...")
    
    # Number of frames
    n_frames = len(simulation_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initialize plots with first frame
    frame_data = simulation_data[0]
    im1 = ax1.imshow(frame_data, cmap='viridis')
    ax1.set_title(f'Gaussian Density (Time 0)')
    
    # Create custom colormap for partitions
    custom_cmap = LinearSegmentedColormap.from_list(
        'partition_cmap', 
        [(0.5, 0, 0.5),     # Purple for internal
         (1.0, 1.0, 0),     # Yellow for blanket 
         (0, 1.0, 1.0)],    # Cyan for external
        N=3
    )
    
    # Determine dominant partition for first frame
    internal_grid = partition_probs['internal'][0].reshape(grid_size)
    blanket_grid = partition_probs['blanket'][0].reshape(grid_size)
    external_grid = partition_probs['external'][0].reshape(grid_size)
    
    dominant_partition = torch.argmax(torch.stack([internal_grid.flatten(), 
                                                   blanket_grid.flatten(), 
                                                   external_grid.flatten()]), dim=0)
    dominant_partition = dominant_partition.reshape(grid_size)
    
    im2 = ax2.imshow(dominant_partition, cmap=custom_cmap, vmin=0, vmax=2)
    ax2.set_title('Dominant Partition (Time 0)')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.5, 0, 0.5), label='Internal'),
        Patch(facecolor=(1.0, 1.0, 0), label='Blanket'),
        Patch(facecolor=(0, 1.0, 1.0), label='External')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Add title
    title = plt.suptitle(f'Evolution of Gaussian and Partition (Frame 0/{n_frames-1})', fontsize=14)
    
    # Update function for animation
    def update(frame):
        # Update density
        frame_data = simulation_data[frame]
        im1.set_array(frame_data)
        ax1.set_title(f'Gaussian Density (Time {frame})')
        
        # Update partition
        internal_grid = partition_probs['internal'][frame].reshape(grid_size)
        blanket_grid = partition_probs['blanket'][frame].reshape(grid_size)
        external_grid = partition_probs['external'][frame].reshape(grid_size)
        
        dominant_partition = torch.argmax(torch.stack([internal_grid.flatten(), 
                                                      blanket_grid.flatten(), 
                                                      external_grid.flatten()]), dim=0)
        dominant_partition = dominant_partition.reshape(grid_size)
        
        im2.set_array(dominant_partition)
        ax2.set_title(f'Dominant Partition (Time {frame})')
        
        # Update title
        title.set_text(f'Evolution of Gaussian and Partition (Frame {frame}/{n_frames-1})')
        
        return [im1, im2, ax1, ax2, title]
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=range(n_frames),
        interval=200, blit=True
    )
    
    # Save animation
    anim.save(output_path, writer='ffmpeg', dpi=100)
    print(f"Animation saved to {output_path}")
    
    return anim

def calculate_metrics(simulation_data, partition_probs, grid_size, time_point):
    """
    Calculate metrics about the partitioning.
    
    Args:
        simulation_data: List of 2D grids with simulation data
        partition_probs: Dictionary with partition probabilities
        grid_size: Size of the grid (height, width)
        time_point: Time point to analyze
        
    Returns:
        Dictionary with metrics
    """
    # Get data for this time point
    frame_data = simulation_data[time_point]
    
    # Get probabilities for this time point
    internal_prob = partition_probs['internal'][time_point]
    blanket_prob = partition_probs['blanket'][time_point]
    external_prob = partition_probs['external'][time_point]
    
    # Reshape to grid
    internal_grid = internal_prob.reshape(grid_size)
    blanket_grid = blanket_prob.reshape(grid_size)
    external_grid = external_prob.reshape(grid_size)
    
    # Determine the dominant partition for each cell
    # 0 = internal, 1 = blanket, 2 = external
    probs = torch.stack([internal_grid.flatten(), blanket_grid.flatten(), external_grid.flatten()])
    dominant_partition = torch.argmax(probs, dim=0).reshape(grid_size)
    
    # Create mask for each partition
    internal_mask = (dominant_partition == 0).numpy()
    blanket_mask = (dominant_partition == 1).numpy()
    external_mask = (dominant_partition == 2).numpy()
    
    # Count cells in each partition
    n_internal = np.sum(internal_mask)
    n_blanket = np.sum(blanket_mask)
    n_external = np.sum(external_mask)
    
    # Define high density threshold
    max_density = np.max(frame_data)
    high_density_mask = (frame_data > 0.7 * max_density)
    n_high_density = np.sum(high_density_mask)
    
    # Calculate overlap between high density and internal states
    internal_high_overlap = np.sum(internal_mask & high_density_mask)
    
    # Calculate precision and recall
    recall = internal_high_overlap / n_high_density if n_high_density > 0 else 0
    precision = internal_high_overlap / n_internal if n_internal > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Verify topological constraints
    # Check that internal states form a connected component
    from scipy import ndimage
    labeled_internal, num_internal_components = ndimage.label(internal_mask)
    connected_internal = (num_internal_components == 1 or num_internal_components == 0)
    
    # Check that blanket surrounds internal
    # For each internal cell, check if it's adjacent to a blanket cell
    internal_coords = np.argwhere(internal_mask)
    surrounded = True
    if len(internal_coords) > 0:
        for y, x in internal_coords:
            # Check 4-connected neighbors
            neighbors = [
                (y+1, x), (y-1, x), (y, x+1), (y, x-1)
            ]
            # Filter out neighbors outside grid
            neighbors = [(ny, nx) for ny, nx in neighbors 
                       if 0 <= ny < grid_size[0] and 0 <= nx < grid_size[1]]
            
            # Check if any neighbor is not internal and not blanket
            external_neighbor = any(
                not internal_mask[ny, nx] and not blanket_mask[ny, nx]
                for ny, nx in neighbors
            )
            if external_neighbor:
                surrounded = False
                break
    
    # Return metrics
    metrics = {
        'n_internal': n_internal,
        'n_blanket': n_blanket,
        'n_external': n_external,
        'n_high_density': n_high_density,
        'internal_high_overlap': internal_high_overlap,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'connected_internal': connected_internal,
        'surrounded': surrounded
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Demonstrate PyTorch-based Markov Blanket partitioning')
    parser.add_argument('--grid_size', type=int, default=20, help='Size of grid (single number for square grid)')
    parser.add_argument('--time_points', type=int, default=20, help='Number of time points to simulate')
    parser.add_argument('--radius', type=float, default=8.0, help='Radius of circular path for Gaussian')
    parser.add_argument('--sigma', type=float, default=1.5, help='Standard deviation of Gaussian')
    parser.add_argument('--output_dir', default='output/direct_visualization', help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create gridworld
    grid_size = (args.grid_size, args.grid_size)
    simulation_data, df, tensor_data = create_gaussian_gridworld(
        grid_size=grid_size,
        n_time_points=args.time_points,
        radius=args.radius,
        sigma=args.sigma
    )
    
    # Infer Markov Blanket partitioning
    partition_probs = infer_markov_blanket(simulation_data, grid_size)
    
    # Create visualizations for selected time points
    time_points = [0, args.time_points // 4, args.time_points // 2, 3 * args.time_points // 4, args.time_points - 1]
    
    for t in time_points:
        # Create probability visualization
        output_path = os.path.join(args.output_dir, f'partition_probs_t{t}.png')
        create_partition_visualization(simulation_data, partition_probs, grid_size, t, output_path)
        
        # Create combined visualization
        output_path = os.path.join(args.output_dir, f'combined_t{t}.png')
        create_combined_visualization(simulation_data, partition_probs, grid_size, t, output_path)
        
        # Create overlay visualization
        output_path = os.path.join(args.output_dir, f'overlay_t{t}.png')
        create_overlay_visualization(simulation_data, partition_probs, grid_size, t, output_path)
        
        # Calculate metrics
        metrics = calculate_metrics(simulation_data, partition_probs, grid_size, t)
        
        # Print metrics
        print(f"\nMetrics for time point {t}:")
        print(f"  Internal states: {metrics['n_internal']} cells")
        print(f"  Blanket states: {metrics['n_blanket']} cells")
        print(f"  External states: {metrics['n_external']} cells")
        print(f"  High density cells: {metrics['n_high_density']} cells")
        print(f"  Internal states that are high density: {metrics['internal_high_overlap']} cells")
        print(f"  Recall: {metrics['recall']:.2%} of high density cells are classified as internal")
        print(f"  Precision: {metrics['precision']:.2%} of internal cells are high density")
        print(f"  F1 Score: {metrics['f1']:.2%}")
        print(f"  Internal states form a connected component: {metrics['connected_internal']}")
        print(f"  Internal states are surrounded by blanket: {metrics['surrounded']}")
        
        # Save metrics to file
        metrics_path = os.path.join(args.output_dir, f'metrics_t{t}.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Metrics for PyTorch-Inferred Markov Blanket Partition (Time={t})\n")
            f.write("==========================================================\n\n")
            f.write(f"Cell Counts:\n")
            f.write(f"  Internal states: {metrics['n_internal']} cells\n")
            f.write(f"  Blanket states: {metrics['n_blanket']} cells\n")
            f.write(f"  External states: {metrics['n_external']} cells\n")
            f.write(f"  Total cells: {grid_size[0] * grid_size[1]} cells\n\n")
            f.write(f"Density Analysis:\n")
            f.write(f"  High density cells (>70% max): {metrics['n_high_density']} cells\n")
            f.write(f"  Internal states that are high density: {metrics['internal_high_overlap']} cells\n\n")
            f.write(f"Precision and Recall:\n")
            f.write(f"  Recall: {metrics['recall']:.2%} of high density cells are classified as internal\n")
            f.write(f"  Precision: {metrics['precision']:.2%} of internal cells are high density\n")
            f.write(f"  F1 Score: {metrics['f1']:.2%}\n\n")
            f.write(f"Topological Constraints:\n")
            f.write(f"  Internal states form a connected component: {metrics['connected_internal']}\n")
            f.write(f"  Internal states are surrounded by blanket: {metrics['surrounded']}\n")
    
    # Create animation
    animation_path = os.path.join(args.output_dir, 'partition_evolution.mp4')
    create_animation(simulation_data, partition_probs, grid_size, animation_path)
    
    print(f"\nAll visualizations and metrics saved to {args.output_dir}")
    print(f"Animation saved to {animation_path}")

if __name__ == '__main__':
    main() 