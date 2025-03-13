"""
Tests for the DataPartitioning class.
"""

import unittest
import pandas as pd
import numpy as np
import torch

from framework.data_partitioning import DataPartitioning


class TestDataPartitioning:
    """Tests for the DataPartitioning class."""
    
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
        
        # Initialize DataPartitioning
        self.dp = DataPartitioning(self.data)
        
        # Define classification dictionary
        self.classification = {
            'internal': [0],
            'blanket': [1, 2],
            'external': [3, 4]
        }
    
    def test_partition_data(self):
        """Test partition_data method."""
        # Test with DataFrame input
        partitions = self.dp.partition_data(self.classification)
        
        # Check keys
        self.assertIn('internal', partitions)
        self.assertIn('blanket', partitions)
        self.assertIn('external', partitions)
        
        # Check shapes
        self.assertEqual(partitions['internal'].shape, (100, 1))
        self.assertEqual(partitions['blanket'].shape, (100, 2))
        self.assertEqual(partitions['external'].shape, (100, 2))
        
        # Check types
        self.assertIsInstance(partitions['internal'], pd.DataFrame)
        self.assertIsInstance(partitions['blanket'], pd.DataFrame)
        self.assertIsInstance(partitions['external'], pd.DataFrame)
        
        # Test with return_tensors=True
        tensor_partitions = self.dp.partition_data(self.classification, return_tensors=True)
        
        # Check types
        self.assertIsInstance(tensor_partitions['internal'], torch.Tensor)
        self.assertIsInstance(tensor_partitions['blanket'], torch.Tensor)
        self.assertIsInstance(tensor_partitions['external'], torch.Tensor)
    
    def test_create_train_test_split(self):
        """Test create_train_test_split method."""
        # Test with DataFrame input
        train_data, test_data = self.dp.create_train_test_split(test_size=0.3, random_state=42)
        
        # Check shapes
        self.assertEqual(train_data.shape[0], 70)  # 70% of data
        self.assertEqual(test_data.shape[0], 30)   # 30% of data
        
        # Check types
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)
        
        # Test with return_tensors=True
        train_tensor, test_tensor = self.dp.create_train_test_split(
            test_size=0.3, random_state=42, return_tensors=True
        )
        
        # Check types
        self.assertIsInstance(train_tensor, torch.Tensor)
        self.assertIsInstance(test_tensor, torch.Tensor)
    
    def test_create_temporal_partition(self):
        """Test create_temporal_partition method."""
        # Add time column to data
        data_with_time = self.data.copy()
        data_with_time['time'] = np.repeat(np.arange(10), 10)
        
        # Initialize with time data
        dp_time = DataPartitioning(data_with_time)
        
        # Create temporal partition
        temporal = dp_time.create_temporal_partition('time', n_history=2)
        
        # Check keys
        self.assertIn('history', temporal)
        self.assertIn('future', temporal)
        
        # Check types
        self.assertIsInstance(temporal['history'], pd.DataFrame)
        self.assertIsInstance(temporal['future'], pd.DataFrame)
        
        # Test with return_tensors=True
        temporal_tensors = dp_time.create_temporal_partition(
            'time', n_history=2, return_tensors=True
        )
        
        # Check types
        self.assertIsInstance(temporal_tensors['history'], torch.Tensor)
        self.assertIsInstance(temporal_tensors['future'], torch.Tensor)
    
    def test_partition_by_markov_blanket(self):
        """Test partition_by_markov_blanket method."""
        # Test with classification dictionary
        partitions = self.dp.partition_by_markov_blanket(self.classification)
        
        # Check keys
        self.assertIn('internal', partitions)
        self.assertIn('blanket', partitions)
        self.assertIn('external', partitions)
        
        # Check shapes
        self.assertEqual(partitions['internal'].shape, (100, 1))
        self.assertEqual(partitions['blanket'].shape, (100, 2))
        self.assertEqual(partitions['external'].shape, (100, 2))
        
        # Check types
        self.assertIsInstance(partitions['internal'], torch.Tensor)
        self.assertIsInstance(partitions['blanket'], torch.Tensor)
        self.assertIsInstance(partitions['external'], torch.Tensor)
    
    def test_create_batches(self):
        """Test create_batches method."""
        # Test with DataFrame input
        batches = self.dp.create_batches(batch_size=20, shuffle=True, random_state=42)
        
        # Check number of batches
        self.assertEqual(len(batches), 5)  # 100 samples / 20 batch size = 5 batches
        
        # Check shapes
        self.assertEqual(batches[0].shape, (20, 5))
        
        # Check types
        self.assertIsInstance(batches[0], pd.DataFrame)
        
        # Test with return_tensors=True
        tensor_batches = self.dp.create_batches(
            batch_size=20, shuffle=True, random_state=42, return_tensors=True
        )
        
        # Check types
        self.assertIsInstance(tensor_batches[0], torch.Tensor)
    
    def test_compute_statistics(self):
        """Test compute_statistics method."""
        # Compute statistics
        stats = self.dp.compute_statistics()
        
        # Check keys
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('median', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        
        # Check shapes
        self.assertEqual(stats['mean'].shape, (5,))
        self.assertEqual(stats['std'].shape, (5,))
        
        # Check values
        self.assertAlmostEqual(stats['mean'][0], 0.0, places=1)  # X0 has mean close to 0
    
    def test_constructor_with_tensor(self):
        """Test constructor with tensor input."""
        # Create tensor from data
        tensor_data = torch.tensor(self.data.values, dtype=torch.float32)
        
        # Initialize with tensor
        dp_tensor = DataPartitioning(tensor_data)
        
        # Test partition_data
        partitions = dp_tensor.partition_data(self.classification)
        
        # Check types
        self.assertIsInstance(partitions['internal'], pd.DataFrame)
        
        # Test with return_tensors=True
        tensor_partitions = dp_tensor.partition_data(self.classification, return_tensors=True)
        
        # Check types
        self.assertIsInstance(tensor_partitions['internal'], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
