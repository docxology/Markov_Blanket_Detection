#!/usr/bin/env python3
"""
Gridworld Dynamic Markov Blanket Analysis Runner
===============================================

This script runs a complete analysis of Dynamic Markov Blankets in a gridworld
simulation with a moving Gaussian blur. It generates visualizations and animations
of the evolving blanket partitions over time.

Usage:
    python run_gridworld_analysis.py [--grid-size 30 30] [--time-points 100] 
        [--radius 10.0] [--sigma 2.0] [--threshold 0.1] [--output-dir output/gridworld_analysis]
        [--use-torch]
"""

import os
import sys
import argparse
import time
import torch
from pathlib import Path

from src.gridworld_simulation import GaussianBlurGridworld, generate_gridworld_data
from src.gridworld_dmbd import GridworldMarkovAnalyzer, run_gridworld_dmbd_analysis


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run gridworld Dynamic Markov Blanket analysis'
    )
    
    parser.add_argument(
        '--grid-size', 
        nargs=2, 
        type=int, 
        default=[30, 30],
        help='Size of the grid as height width (default: 30 30)'
    )
    
    parser.add_argument(
        '--time-points', 
        type=int, 
        default=100,
        help='Number of time points to simulate (default: 100)'
    )
    
    parser.add_argument(
        '--radius', 
        type=float, 
        default=10.0,
        help='Radius of the circular path for the Gaussian blur (default: 10.0)'
    )
    
    parser.add_argument(
        '--sigma', 
        type=float, 
        default=2.0,
        help='Standard deviation of the Gaussian blur (default: 2.0)'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.1,
        help='Threshold for identifying Markov blanket components (default: 0.1)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='output/gridworld_analysis',
        help='Directory to save outputs (default: output/gridworld_analysis)'
    )
    
    parser.add_argument(
        '--use-torch', 
        action='store_true',
        help='Use PyTorch for computations (default: True)'
    )
    
    parser.add_argument(
        '--skip-animation', 
        action='store_true',
        help='Skip generating animations (default: False)'
    )
    
    parser.add_argument(
        '--quick-mode', 
        action='store_true',
        help='Run a quicker analysis with fewer time points and a smaller grid (default: False)'
    )
    
    return parser.parse_args()


def main():
    """Run the gridworld DMBD analysis."""
    args = parse_args()
    
    # Print DMBD banner
    print("\n" + "="*80)
    print("    DYNAMIC MARKOV BLANKET DETECTION - GRIDWORLD ANALYSIS")
    print("="*80)
    
    # Apply quick mode if requested
    if args.quick_mode:
        print("\nRunning in quick mode with reduced parameters...")
        args.grid_size = [15, 15]
        args.time_points = 30
        args.output_dir = os.path.join(args.output_dir, 'quick_mode')
        
    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Grid size: {args.grid_size[0]}x{args.grid_size[1]}")
    print(f"  Time points: {args.time_points}")
    print(f"  Radius: {args.radius}")
    print(f"  Sigma: {args.sigma}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Using PyTorch: {args.use_torch}")
    print(f"  Skip animations: {args.skip_animation}")
    
    # Check if PyTorch is available
    if args.use_torch:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"\nPyTorch will use GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("\nPyTorch will use CPU (no GPU available)")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Run the analysis
        result = run_gridworld_dmbd_analysis(
            grid_size=tuple(args.grid_size),
            n_time_points=args.time_points,
            radius=args.radius,
            sigma=args.sigma,
            threshold=args.threshold,
            output_dir=args.output_dir,
            use_torch=args.use_torch
        )
        
        # Print successful completion
        elapsed_time = time.time() - start_time
        print(f"\nAnalysis completed successfully in {elapsed_time:.2f} seconds!")
        print("\nGenerated outputs:")
        for category, paths in result.items():
            print(f"  {category}:")
            if isinstance(paths, dict):
                for name, path in paths.items():
                    print(f"    {name}: {path}")
            else:
                print(f"    {paths}")
        
        # Open the summary file
        summary_path = result.get('summary')
        if summary_path and os.path.exists(summary_path):
            print(f"\nSummary information available in: {summary_path}")
        
        print("\nTo visualize the results, open the generated HTML report or PNG/MP4 files.")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 