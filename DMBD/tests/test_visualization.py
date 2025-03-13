"""
Tests for the visualization module.
"""

import unittest
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from framework.markov_blanket import MarkovBlanket, DynamicMarkovBlanket
from framework.visualization import MarkovBlanketVisualizer
from framework.cognitive_identification import CognitiveIdentification


class TestMarkovBlanketVisualizer(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 100
        n_vars = 5
        
        # Create a simple causal graph
        data = np.zeros((n_samples, n_vars))
        
        # Generate base variables
        data[:, 0] = np.random.randn(n_samples)  # X0
        data[:, 4] = np.random.randn(n_samples)  # X4 (independent)
        
        # Generate dependent variables
        data[:, 1] = 0.7 * data[:, 0] + 0.3 * np.random.randn(n_samples)  # X1
        data[:, 2] = 0.6 * data[:, 0] + 0.4 * np.random.randn(n_samples)  # X2
        data[:, 3] = 0.5 * data[:, 1] + 0.5 * data[:, 2] + 0.3 * np.random.randn(n_samples)  # X3
        
        # Create DataFrame
        self.data = pd.DataFrame(data, columns=[f'X{i}' for i in range(n_vars)])
        
        # Initialize components
        self.mb = MarkovBlanket(self.data)
        self.dmb = DynamicMarkovBlanket(self.data, lag=1)
        self.ci = CognitiveIdentification(self.data, self.mb)
        
        # Initialize visualizer
        self.visualizer = MarkovBlanketVisualizer(figsize=(8, 6))
        
        # Create node labels
        self.node_labels = {i: f"Var{i}" for i in range(n_vars)}
    
    def test_plot_markov_blanket(self):
        """Test plot_markov_blanket method."""
        # Create plot
        fig = self.visualizer.plot_markov_blanket(
            self.mb, 
            target_idx=0, 
            node_labels=self.node_labels,
            show_strengths=True
        )
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Close figure to free memory
        plt.close(fig)
    
    def test_plot_cognitive_structures(self):
        """Test plot_cognitive_structures method."""
        # Identify cognitive structures
        target_idx = 0
        structures = self.ci.identify_cognitive_structures(target_idx)
        
        # Create plot
        fig = self.visualizer.plot_cognitive_structures(
            self.mb, 
            structures, 
            node_labels=self.node_labels
        )
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Close figure to free memory
        plt.close(fig)
    
    def test_plot_temporal_dynamics(self):
        """Test plot_temporal_dynamics method."""
        # Create plot
        fig = self.visualizer.plot_temporal_dynamics(
            self.dmb, 
            target_idx=0
        )
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Close figure to free memory
        plt.close(fig)
    
    def test_plot_information_flow(self):
        """Test plot_information_flow method."""
        # Create plot
        fig = self.visualizer.plot_information_flow(
            self.mb, 
            target_idx=0,
            node_labels=self.node_labels
        )
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Close figure to free memory
        plt.close(fig)
    
    def test_plot_blanket_components(self):
        """Test plot_blanket_components method."""
        # Create plot
        fig = self.visualizer.plot_blanket_components(
            self.mb, 
            target_idx=0
        )
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Close figure to free memory
        plt.close(fig)
    
    def test_custom_axes(self):
        """Test using custom axes for plotting."""
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot on each subplot
        self.visualizer.plot_markov_blanket(
            self.mb, target_idx=0, ax=axes[0, 0]
        )
        
        self.visualizer.plot_information_flow(
            self.mb, target_idx=0, ax=axes[0, 1]
        )
        
        self.visualizer.plot_blanket_components(
            self.mb, target_idx=0, ax=axes[1, 0]
        )
        
        # Get cognitive structures
        target_idx = 0
        structures = self.ci.identify_cognitive_structures(target_idx)
        
        self.visualizer.plot_cognitive_structures(
            self.mb, structures, ax=axes[1, 1]
        )
        
        # Check that the figure is correctly configured
        self.assertEqual(len(fig.axes), 4)
        
        # Close figure to free memory
        plt.close(fig)


if __name__ == '__main__':
    unittest.main() 