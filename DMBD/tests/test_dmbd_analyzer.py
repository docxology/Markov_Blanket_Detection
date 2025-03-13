"""
Tests for the DMBDAnalyzer class.
"""

import unittest
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import tempfile
from pathlib import Path

from src.dmbd_analyzer import DMBDAnalyzer


class TestDMBDAnalyzer(unittest.TestCase):
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
        
        # Initialize analyzer
        self.analyzer = DMBDAnalyzer(self.data)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def test_init_with_dataframe(self):
        """Test initialization with a pandas DataFrame."""
        # Check that components are initialized
        self.assertIsNotNone(self.analyzer.markov_blanket)
        self.assertIsNotNone(self.analyzer.dynamic_markov_blanket)
        self.assertIsNotNone(self.analyzer.data_partitioner)
        self.assertIsNotNone(self.analyzer.cognitive_identifier)
        self.assertIsNotNone(self.analyzer.visualizer)
    
    def test_init_with_tensor(self):
        """Test initialization with a PyTorch tensor."""
        # Create tensor from data
        tensor_data = torch.tensor(self.data.values, dtype=torch.float32)
        
        # Initialize with tensor
        analyzer = DMBDAnalyzer(tensor_data)
        
        # Check that components are initialized
        self.assertIsNotNone(analyzer.markov_blanket)
        self.assertIsNotNone(analyzer.dynamic_markov_blanket)
        self.assertIsNotNone(analyzer.data_partitioner)
        self.assertIsNotNone(analyzer.cognitive_identifier)
        self.assertIsNotNone(analyzer.visualizer)
    
    def test_init_with_csv(self):
        """Test initialization with a CSV file."""
        # Save data to CSV
        csv_path = Path(self.temp_dir.name) / "test_data.csv"
        self.data.to_csv(csv_path, index=False)
        
        # Initialize with CSV path
        analyzer = DMBDAnalyzer(str(csv_path))
        
        # Check that components are initialized
        self.assertIsNotNone(analyzer.markov_blanket)
        self.assertIsNotNone(analyzer.dynamic_markov_blanket)
        self.assertIsNotNone(analyzer.data_partitioner)
        self.assertIsNotNone(analyzer.cognitive_identifier)
        self.assertIsNotNone(analyzer.visualizer)
    
    def test_detect_markov_blanket(self):
        """Test detect_markov_blanket method."""
        # Detect Markov blanket
        target_idx = 0
        result = self.analyzer.detect_markov_blanket(target_idx)
        
        # Check structure of result
        self.assertIn('parents', result)
        self.assertIn('children', result)
        self.assertIn('spouses', result)
        self.assertIn('classifications', result)
        self.assertIn('strengths', result)
        self.assertIn('blanket_size', result)
        
        # Check specific results for X0
        self.assertEqual(len(result['parents']), 0)  # X0 has no parents
        self.assertEqual(set(result['children']), set([1, 2]))  # X0 has X1, X2 as children
    
    def test_detect_dynamic_markov_blanket(self):
        """Test detect_dynamic_markov_blanket method."""
        # Detect dynamic Markov blanket
        target_idx = 0
        result = self.analyzer.detect_dynamic_markov_blanket(target_idx)
        
        # Check structure of result
        self.assertIn('dynamic_components', result)
        self.assertIn('dynamic_classifications', result)
        
        # Check specific components
        dynamic_components = result['dynamic_components']
        self.assertIn('current', dynamic_components)
        self.assertIn('lag_1', dynamic_components)
    
    def test_identify_cognitive_structures(self):
        """Test identify_cognitive_structures method."""
        # Identify cognitive structures
        target_idx = 0
        result = self.analyzer.identify_cognitive_structures(target_idx)
        
        # Check structure of result
        self.assertIn('structures', result)
        self.assertIn('metrics', result)
        
        # Check specific structures
        structures = result['structures']
        self.assertIn('sensory', structures)
        self.assertIn('action', structures)
        self.assertIn('internal_state', structures)
        
        # Check metrics
        metrics = result['metrics']
        self.assertIn('cognitive_capacity', metrics)
        self.assertIn('integration', metrics)
        self.assertIn('complexity', metrics)
    
    def test_identify_temporal_cognitive_structures(self):
        """Test identify_temporal_cognitive_structures method."""
        # Identify temporal cognitive structures
        target_idx = 0
        result = self.analyzer.identify_temporal_cognitive_structures(target_idx)
        
        # Check structure of result
        self.assertIn('temporal_structures', result)
        self.assertIn('temporal_metrics', result)
        
        # Check specific structures
        temporal_structures = result['temporal_structures']
        self.assertIn('current', temporal_structures)
    
    def test_partition_data_by_markov_blanket(self):
        """Test partition_data_by_markov_blanket method."""
        # Partition data
        target_idx = 0
        partitions = self.analyzer.partition_data_by_markov_blanket(target_idx)
        
        # Check keys
        self.assertIn('internal', partitions)
        self.assertIn('blanket', partitions)
        self.assertIn('external', partitions)
        
        # Check shapes
        self.assertEqual(partitions['internal'].shape[1], 1)  # X0 only
        self.assertEqual(partitions['blanket'].shape[1], 2)  # X1, X2
        self.assertEqual(partitions['external'].shape[1], 2)  # X3, X4
    
    def test_analyze_target(self):
        """Test analyze_target method."""
        # Analyze target
        target_idx = 0
        result = self.analyzer.analyze_target(target_idx)
        
        # Check structure of result
        self.assertIn('markov_blanket', result)
        self.assertIn('cognitive', result)
        self.assertIn('partitions', result)
        self.assertIn('dynamic_markov_blanket', result)
        self.assertIn('temporal_cognitive', result)
        
        # Test without dynamic analysis
        result_static = self.analyzer.analyze_target(target_idx, dynamic=False)
        self.assertNotIn('dynamic_markov_blanket', result_static)
        self.assertNotIn('temporal_cognitive', result_static)
    
    def test_visualization_methods(self):
        """Test visualization methods."""
        target_idx = 0
        
        # Test visualize_markov_blanket
        fig1 = self.analyzer.visualize_markov_blanket(target_idx)
        self.assertIsInstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test visualize_cognitive_structures
        fig2 = self.analyzer.visualize_cognitive_structures(target_idx)
        self.assertIsInstance(fig2, plt.Figure)
        plt.close(fig2)
        
        # Test visualize_temporal_dynamics
        fig3 = self.analyzer.visualize_temporal_dynamics(target_idx)
        self.assertIsInstance(fig3, plt.Figure)
        plt.close(fig3)
        
        # Test visualize_information_flow
        fig4 = self.analyzer.visualize_information_flow(target_idx)
        self.assertIsInstance(fig4, plt.Figure)
        plt.close(fig4)
        
        # Test saving to file
        save_path = Path(self.temp_dir.name) / "test_plot.png"
        fig5 = self.analyzer.visualize_markov_blanket(
            target_idx, save_path=str(save_path)
        )
        plt.close(fig5)
        
        # Check that file was created
        self.assertTrue(save_path.exists())
    
    def test_save_load_results(self):
        """Test save_results and load_results methods."""
        # Analyze target
        target_idx = 0
        results = self.analyzer.analyze_target(target_idx)
        
        # Save results
        save_path = Path(self.temp_dir.name) / "test_results.pkl"
        self.analyzer.save_results(results, str(save_path))
        
        # Check that file was created
        self.assertTrue(save_path.exists())
        
        # Load results
        loaded_results = self.analyzer.load_results(str(save_path))
        
        # Check that loaded results match original results
        self.assertEqual(
            results['markov_blanket']['blanket_size'],
            loaded_results['markov_blanket']['blanket_size']
        )
        
        # Test structure of loaded results
        self.assertIn('markov_blanket', loaded_results)
        self.assertIn('cognitive', loaded_results)
        self.assertIn('partitions', loaded_results)


if __name__ == '__main__':
    unittest.main() 