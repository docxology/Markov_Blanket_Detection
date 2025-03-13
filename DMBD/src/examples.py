"""
DMBD Examples - Example usage of the DMBD framework
==================================================

This module provides examples of how to use the DMBD framework for
analyzing different types of data.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import os
from pathlib import Path

from .dmbd_analyzer import DMBDAnalyzer


def generate_synthetic_data(
    n_samples: int = 1000, 
    n_vars: int = 10, 
    causal_density: float = 0.3,
    temporal: bool = False, 
    n_time_points: int = 10
) -> pd.DataFrame:
    """
    Generate synthetic data for testing DMBD.
    
    Args:
        n_samples: Number of samples to generate
        n_vars: Number of variables
        causal_density: Density of causal connections
        temporal: Whether to include temporal dependencies
        n_time_points: Number of time points for temporal data
        
    Returns:
        DataFrame with synthetic data
    """
    # Generate random causal graph
    np.random.seed(42)
    
    # Create adjacency matrix for causal graph
    adj_matrix = np.random.rand(n_vars, n_vars) < causal_density
    # Make it a DAG (upper triangular)
    adj_matrix = np.triu(adj_matrix, k=1)
    
    if temporal:
        # Total samples across all time points
        total_samples = n_samples * n_time_points
        
        # Initialize data array
        data = np.zeros((total_samples, n_vars + 1))
        
        # Add time column
        for t in range(n_time_points):
            start_idx = t * n_samples
            end_idx = (t + 1) * n_samples
            data[start_idx:end_idx, 0] = t
        
        # Generate data for each time point
        for t in range(n_time_points):
            start_idx = t * n_samples
            end_idx = (t + 1) * n_samples
            
            # Base features (independent)
            for i in range(n_vars):
                if np.sum(adj_matrix[:, i]) == 0:  # No parents
                    data[start_idx:end_idx, i + 1] = np.random.randn(n_samples)
            
            # Dependent features
            for i in range(n_vars):
                if np.sum(adj_matrix[:, i]) > 0:  # Has parents
                    parents = np.where(adj_matrix[:, i])[0]
                    
                    # Combine parent values with noise
                    parent_values = data[start_idx:end_idx, parents + 1]
                    weights = np.random.randn(len(parents))
                    
                    # Compute weighted sum of parents + noise
                    data[start_idx:end_idx, i + 1] = parent_values @ weights + 0.5 * np.random.randn(n_samples)
        
        # Create column names
        columns = ['time'] + [f'X{i}' for i in range(n_vars)]
        
    else:
        # Initialize data array
        data = np.zeros((n_samples, n_vars))
        
        # Base features (independent)
        for i in range(n_vars):
            if np.sum(adj_matrix[:, i]) == 0:  # No parents
                data[:, i] = np.random.randn(n_samples)
        
        # Dependent features
        for i in range(n_vars):
            if np.sum(adj_matrix[:, i]) > 0:  # Has parents
                parents = np.where(adj_matrix[:, i])[0]
                
                # Combine parent values with noise
                parent_values = data[:, parents]
                weights = np.random.randn(len(parents))
                
                # Compute weighted sum of parents + noise
                data[:, i] = parent_values @ weights + 0.5 * np.random.randn(n_samples)
        
        # Create column names
        columns = [f'X{i}' for i in range(n_vars)]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    return df


def basic_markov_blanket_example() -> None:
    """Example of basic Markov blanket detection."""
    print("Running basic Markov blanket example...")
    
    # Generate synthetic data
    data = generate_synthetic_data(n_samples=500, n_vars=8, causal_density=0.3)
    
    # Initialize analyzer
    analyzer = DMBDAnalyzer(data)
    
    # Analyze target variable
    target_idx = 3
    results = analyzer.analyze_target(target_idx, dynamic=False)
    
    # Print results
    print(f"\nMarkov Blanket for X{target_idx}:")
    print(f"  Parents: {results['markov_blanket']['parents']}")
    print(f"  Children: {results['markov_blanket']['children']}")
    print(f"  Spouses: {results['markov_blanket']['spouses']}")
    print(f"  Total Blanket Size: {results['markov_blanket']['blanket_size']}")
    
    print("\nCognitive Structures:")
    print(f"  Sensory: {results['cognitive']['structures'].get('sensory', [])}")
    print(f"  Action: {results['cognitive']['structures'].get('action', [])}")
    print(f"  Internal State: {results['cognitive']['structures'].get('internal_state', [])}")
    
    print("\nCognitive Metrics:")
    for metric, value in results['cognitive']['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize
    fig1 = analyzer.visualize_markov_blanket(target_idx)
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.show()
    
    fig2 = analyzer.visualize_cognitive_structures(target_idx)
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.show()
    
    fig3 = analyzer.visualize_information_flow(target_idx)
    plt.figure(fig3.number)
    plt.tight_layout()
    plt.show()
    
    print("Basic Markov blanket example completed!")


def dynamic_markov_blanket_example() -> None:
    """Example of dynamic Markov blanket detection with temporal data."""
    print("Running dynamic Markov blanket example...")
    
    # Generate synthetic temporal data
    data = generate_synthetic_data(
        n_samples=200, n_vars=6, causal_density=0.3, 
        temporal=True, n_time_points=5
    )
    
    # Initialize analyzer with time column
    analyzer = DMBDAnalyzer(data, time_column='time', lag=2)
    
    # Analyze target variable
    target_idx = 2  # Index in the data (not counting time column)
    results = analyzer.analyze_target(target_idx, dynamic=True)
    
    # Print dynamic results
    print(f"\nDynamic Markov Blanket for X{target_idx}:")
    
    dynamic_components = results['dynamic_markov_blanket']['dynamic_components']
    for time_key, components in dynamic_components.items():
        print(f"  Time: {time_key}")
        print(f"    Parents: {components['parents']}")
        print(f"    Children: {components['children']}")
        print(f"    Spouses: {components['spouses']}")
    
    # Print temporal cognitive structures
    print("\nTemporal Cognitive Structures:")
    
    temporal_structures = results['temporal_cognitive']['temporal_structures']
    for time_key, structures in temporal_structures.items():
        print(f"  Time: {time_key}")
        print(f"    Sensory: {structures.get('sensory', [])}")
        print(f"    Action: {structures.get('action', [])}")
        print(f"    Internal State: {structures.get('internal_state', [])}")
    
    # Visualize
    fig1 = analyzer.visualize_temporal_dynamics(target_idx)
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.show()
    
    # Basic blanket visualization at current time
    fig2 = analyzer.visualize_markov_blanket(target_idx)
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.show()
    
    print("Dynamic Markov blanket example completed!")


def run_all_examples() -> None:
    """Run all example functions."""
    # Create output directory for visualizations if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Run examples
    basic_markov_blanket_example()
    dynamic_markov_blanket_example()


if __name__ == "__main__":
    run_all_examples() 