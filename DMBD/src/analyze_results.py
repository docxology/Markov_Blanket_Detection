#!/usr/bin/env python3
"""
Script to analyze and compare the results of the gridworld simulation and the inferred Markov Blanket partitioning.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
import re
import argparse

def extract_frame_number(filename):
    """Extract frame number from filename."""
    match = re.search(r't(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def compare_simulation_and_partition(simulation_dir, analysis_dir, output_dir, frames=None):
    """
    Compare original simulation frames with inferred partition frames.
    
    Args:
        simulation_dir: Directory containing simulation frames/data
        analysis_dir: Directory containing inferred partition frames
        output_dir: Directory to save comparison images
        frames: List of frame indices to compare (if None, use all available frames)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get simulation frames
    simulation_frame = os.path.join(simulation_dir, 'gridworld_frame.png')
    
    # Get partition frames
    partition_files = [f for f in os.listdir(analysis_dir) if f.startswith('markov_partition_t') and f.endswith('.png')]
    partition_files.sort(key=extract_frame_number)
    
    # If frames not specified, use all frames
    if frames is None:
        frames = [extract_frame_number(f) for f in partition_files]
    
    # Load simulation animation data to extract specific frames
    simulation_animation = os.path.join(simulation_dir, 'gridworld_animation.mp4')
    
    # Check if simulation animation exists
    if os.path.exists(simulation_animation):
        # Load the simulation animation
        cap = cv2.VideoCapture(simulation_animation)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process each frame
        for frame_idx in frames:
            # Find corresponding partition file
            partition_file = None
            for f in partition_files:
                if extract_frame_number(f) == frame_idx:
                    partition_file = f
                    break
            
            if partition_file is None:
                print(f"No partition file found for frame {frame_idx}")
                continue
            
            # Get the correct frame from the simulation animation
            relative_frame = int((frame_idx / max(frames)) * (total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, relative_frame)
            ret, sim_frame = cap.read()
            
            if not ret:
                print(f"Failed to read frame {relative_frame} from simulation animation")
                continue
            
            # Convert BGR to RGB
            sim_frame = cv2.cvtColor(sim_frame, cv2.COLOR_BGR2RGB)
            
            # Load partition frame
            partition_path = os.path.join(analysis_dir, partition_file)
            partition_frame = mpimg.imread(partition_path)
            
            # Create comparison plot
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot simulation frame
            axs[0].imshow(sim_frame)
            axs[0].set_title(f"Simulation Frame (t={frame_idx})")
            axs[0].axis('off')
            
            # Plot partition frame
            axs[1].imshow(partition_frame)
            axs[1].set_title(f"Inferred Markov Blanket Partition (t={frame_idx})")
            axs[1].axis('off')
            
            # Set overall title
            fig.suptitle(f"Comparison of Simulation and Inferred Partition (Frame {frame_idx})", fontsize=16)
            
            # Save comparison
            output_path = os.path.join(output_dir, f"comparison_t{frame_idx}.png")
            plt.savefig(output_path)
            plt.close(fig)
            
            print(f"Created comparison for frame {frame_idx}: {output_path}")
        
        # Release the video capture
        cap.release()
    else:
        print(f"Simulation animation not found: {simulation_animation}")

def analyze_coverage(simulation_dir, analysis_dir, output_dir):
    """
    Analyze how well the internal states cover the high-density regions of the Gaussian blur.
    
    Args:
        simulation_dir: Directory containing simulation data
        analysis_dir: Directory containing inferred partition frames
        output_dir: Directory to save analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get partition frames
    partition_files = [f for f in os.listdir(analysis_dir) if f.startswith('markov_partition_t') and f.endswith('.png')]
    partition_files.sort(key=extract_frame_number)
    
    # Load simulation animation
    simulation_animation = os.path.join(simulation_dir, 'gridworld_animation.mp4')
    
    if not os.path.exists(simulation_animation):
        print(f"Simulation animation not found: {simulation_animation}")
        return
    
    # Load the simulation animation
    cap = cv2.VideoCapture(simulation_animation)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare results
    results = {
        'frame': [],
        'internal_coverage': [],
        'blanket_coverage': [],
        'external_coverage': [],
        'precision': [],
        'recall': []
    }
    
    # Process each partition frame
    for partition_file in partition_files:
        frame_idx = extract_frame_number(partition_file)
        
        # Get the corresponding simulation frame
        relative_frame = int((frame_idx / max([extract_frame_number(f) for f in partition_files])) * (total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, relative_frame)
        ret, sim_frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame {relative_frame} from simulation animation")
            continue
        
        # Convert BGR to RGB and to grayscale for intensity analysis
        sim_frame_rgb = cv2.cvtColor(sim_frame, cv2.COLOR_BGR2RGB)
        sim_frame_gray = cv2.cvtColor(sim_frame, cv2.COLOR_BGR2GRAY)
        
        # Load partition frame
        partition_path = os.path.join(analysis_dir, partition_file)
        partition_frame = mpimg.imread(partition_path)
        
        # Resize simulation frame to match partition frame dimensions
        sim_height, sim_width = sim_frame_gray.shape
        part_height, part_width, _ = partition_frame.shape
        
        # Print the dimensions for debugging
        print(f"Original simulation frame dimensions: {sim_width}x{sim_height}")
        print(f"Partition frame dimensions: {part_width}x{part_height}")
        
        # Resize simulation frame to match partition frame dimensions
        resized_sim_frame = cv2.resize(sim_frame_gray, (part_width, part_height))
        
        # Print the dimensions after resizing
        print(f"Resized simulation frame dimensions: {part_width}x{part_height}")
        
        # Convert partition frame to analysis format
        # In our visualizations, internal states are red, blanket states are green, and external states are blue
        # Extract these channels to identify the states
        red_channel = partition_frame[:,:,0]
        green_channel = partition_frame[:,:,1]
        blue_channel = partition_frame[:,:,2]
        
        # Create masks for each state type
        # This is a simplification; in practice, we'd need to verify the exact colors used
        internal_mask = (red_channel > 0.7) & (green_channel < 0.3) & (blue_channel < 0.3)
        blanket_mask = (red_channel < 0.3) & (green_channel > 0.7) & (blue_channel < 0.3)
        external_mask = (red_channel < 0.3) & (green_channel < 0.3) & (blue_channel > 0.7)
        
        # Create high-density mask from simulation frame
        # Normalize and threshold to identify high-density areas
        norm_sim = resized_sim_frame / 255.0
        high_density_mask = norm_sim > 0.7
        
        # Calculate metrics
        internal_high_density = np.sum(internal_mask & high_density_mask)
        blanket_high_density = np.sum(blanket_mask & high_density_mask)
        external_high_density = np.sum(external_mask & high_density_mask)
        
        total_high_density = np.sum(high_density_mask)
        
        internal_coverage = internal_high_density / total_high_density if total_high_density > 0 else 0
        blanket_coverage = blanket_high_density / total_high_density if total_high_density > 0 else 0
        external_coverage = external_high_density / total_high_density if total_high_density > 0 else 0
        
        # Precision and recall for internal states
        precision = internal_high_density / np.sum(internal_mask) if np.sum(internal_mask) > 0 else 0
        recall = internal_high_density / total_high_density if total_high_density > 0 else 0
        
        # Store results
        results['frame'].append(frame_idx)
        results['internal_coverage'].append(internal_coverage)
        results['blanket_coverage'].append(blanket_coverage)
        results['external_coverage'].append(external_coverage)
        results['precision'].append(precision)
        results['recall'].append(recall)
        
        print(f"Frame {frame_idx}: Internal Coverage = {internal_coverage:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}")
        
        # Create a visualization of the coverage
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Original simulation frame
        plt.subplot(1, 3, 1)
        plt.imshow(resized_sim_frame, cmap='gray')
        plt.title(f"Simulation Frame (t={frame_idx})")
        plt.axis('off')
        
        # Plot 2: Partition
        plt.subplot(1, 3, 2)
        plt.imshow(partition_frame)
        plt.title(f"Markov Blanket Partition")
        plt.axis('off')
        
        # Plot 3: Overlay of high-density and internal states
        plt.subplot(1, 3, 3)
        overlay = np.zeros((part_height, part_width, 3))
        # High density areas in blue
        overlay[high_density_mask, 2] = 1.0
        # Internal states in red
        overlay[internal_mask, 0] = 1.0
        # Overlap in purple
        plt.imshow(overlay)
        plt.title(f"Overlay (Blue=High Density, Red=Internal, Purple=Overlap)")
        plt.axis('off')
        
        # Save the coverage visualization
        coverage_vis_path = os.path.join(output_dir, f"coverage_vis_t{frame_idx}.png")
        plt.savefig(coverage_vis_path)
        plt.close()
    
    # Release the video capture
    cap.release()
    
    # Plot the coverage metrics
    plt.figure(figsize=(12, 8))
    plt.plot(results['frame'], results['internal_coverage'], 'r-', label='Internal Coverage')
    plt.plot(results['frame'], results['blanket_coverage'], 'g-', label='Blanket Coverage')
    plt.plot(results['frame'], results['external_coverage'], 'b-', label='External Coverage')
    plt.xlabel('Frame')
    plt.ylabel('Coverage')
    plt.title('Coverage of High-Density Regions by State Type')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    coverage_plot_path = os.path.join(output_dir, 'coverage_metrics.png')
    plt.savefig(coverage_plot_path)
    
    # Plot precision and recall
    plt.figure(figsize=(12, 8))
    plt.plot(results['frame'], results['precision'], 'g-', label='Precision')
    plt.plot(results['frame'], results['recall'], 'b-', label='Recall')
    plt.plot(results['frame'], [2*p*r/(p+r) if p+r > 0 else 0 for p, r in zip(results['precision'], results['recall'])], 'r-', label='F1 Score')
    plt.xlabel('Frame')
    plt.ylabel('Metric Value')
    plt.title('Precision, Recall, and F1 Score for Internal States')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    pr_plot_path = os.path.join(output_dir, 'precision_recall.png')
    plt.savefig(pr_plot_path)
    
    print(f"Coverage metrics saved to {coverage_plot_path}")
    print(f"Precision and recall metrics saved to {pr_plot_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare gridworld simulation and inferred Markov Blanket partitioning.')
    parser.add_argument('--simulation_dir', default='output/final_analysis/simulation', help='Directory containing simulation data')
    parser.add_argument('--analysis_dir', default='output/final_analysis/analysis', help='Directory containing inferred partition data')
    parser.add_argument('--output_dir', default='output/final_analysis/comparison', help='Directory to save comparison results')
    parser.add_argument('--frames', type=int, nargs='+', help='Specific frames to analyze (default: all frames)')
    parser.add_argument('--analyze_coverage', action='store_true', help='Analyze coverage metrics')
    
    args = parser.parse_args()
    
    # Create side-by-side comparisons
    compare_simulation_and_partition(args.simulation_dir, args.analysis_dir, args.output_dir, args.frames)
    
    # Analyze coverage if requested
    if args.analyze_coverage:
        analyze_coverage(args.simulation_dir, args.analysis_dir, args.output_dir)

if __name__ == '__main__':
    main() 