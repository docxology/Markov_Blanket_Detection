"""
Tests for the MarkovBlanket and DynamicMarkovBlanket classes.
"""

import unittest
import pandas as pd
import numpy as np
import torch
from framework.markov_blanket import MarkovBlanket, DynamicMarkovBlanket


class TestMarkovBlanket(unittest.TestCase):
    """Tests for the MarkovBlanket class."""
    
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
        data[:, 1] = 0.7 * data[:, 0] + 0.3 * np.random.randn(n_samples)  # X1 depends on X0
        data[:, 2] = 0.6 * data[:, 0] + 0.4 * np.random.randn(n_samples)  # X2 depends on X0
        data[:, 3] = 0.5 * data[:, 1] + 0.5 * data[:, 2] + 0.3 * np.random.randn(n_samples)  # X3 depends on X1 and X2
        
        # Create DataFrame
        self.data = pd.DataFrame(data, columns=[f'X{i}' for i in range(n_vars)])
        
        # Initialize MarkovBlanket
        self.mb = MarkovBlanket(self.data)
        
        # Define expected relationships for each variable
        # X0: parents=[], children=[1,2], spouses=[]
        # X1: parents=[0], children=[3], spouses=[2]
        # X2: parents=[0], children=[3], spouses=[1]
        # X3: parents=[1,2], children=[], spouses=[]
        # X4: parents=[], children=[], spouses=[]
        self.expected_parents = {
            0: [],
            1: [0],
            2: [0],
            3: [1, 2],
            4: []
        }
        
        self.expected_children = {
            0: [1, 2],
            1: [3],
            2: [3],
            3: [],
            4: []
        }
        
        self.expected_spouses = {
            0: [],
            1: [2],
            2: [1],
            3: [],
            4: []
        }
    
    def test_detect_blanket(self):
        """Test detect_blanket method."""
        # Test for each variable
        for i in range(5):
            parents, children, spouses = self.mb.detect_blanket(i)
            
            # Check parents
            self.assertEqual(set(parents), set(self.expected_parents[i]),
                            f"Parents incorrect for X{i}")
            
            # Check children
            self.assertEqual(set(children), set(self.expected_children[i]),
                            f"Children incorrect for X{i}")
            
            # Check spouses
            self.assertEqual(set(spouses), set(self.expected_spouses[i]),
                            f"Spouses incorrect for X{i}")
    
    def test_classify_nodes(self):
        """Test classify_nodes method."""
        # Test for X0
        classifications = self.mb.classify_nodes(0)
        
        # Check internal
        self.assertEqual(classifications['internal'], [0])
        
        # Check blanket (children for X0)
        self.assertEqual(set(classifications['blanket']), set([1, 2]))
        
        # Check external (X3 and X4 for X0)
        self.assertEqual(set(classifications['external']), set([3, 4]))
        
        # Test for X3
        classifications = self.mb.classify_nodes(3)
        
        # Check internal
        self.assertEqual(classifications['internal'], [3])
        
        # Check blanket (parents for X3)
        self.assertEqual(set(classifications['blanket']), set([1, 2]))
        
        # Check external (X0 and X4 for X3)
        self.assertEqual(set(classifications['external']), set([0, 4]))
    
    def test_compute_mutual_information(self):
        """Test compute_mutual_information method."""
        # Compute MI between X0 and X1 (should be high)
        mi_0_1 = self.mb.compute_mutual_information(0, 1)
        
        # Compute MI between X0 and X4 (should be low)
        mi_0_4 = self.mb.compute_mutual_information(0, 4)
        
        # Check that MI is higher for dependent variables
        self.assertGreater(mi_0_1, mi_0_4)
        
        # Check that MI is symmetric
        mi_1_0 = self.mb.compute_mutual_information(1, 0)
        self.assertAlmostEqual(mi_0_1, mi_1_0, places=5)
    
    def test_get_blanket_strength(self):
        """Test get_blanket_strength method."""
        # Get blanket strength for X0
        strengths = self.mb.get_blanket_strength(0)
        
        # Check keys (should be X1 and X2)
        self.assertEqual(set(strengths.keys()), set([1, 2]))
        
        # Check values (should be positive)
        for node, strength in strengths.items():
            self.assertGreater(strength, 0)
    
    def test_constructor_with_tensor(self):
        """Test constructor with tensor input."""
        # Create tensor from data
        tensor_data = torch.tensor(self.data.values, dtype=torch.float32)
        
        # Initialize with tensor
        mb_tensor = MarkovBlanket(tensor_data)
        
        # Test detect_blanket
        parents, children, spouses = mb_tensor.detect_blanket(0)
        
        # Check results
        self.assertEqual(set(parents), set(self.expected_parents[0]))
        self.assertEqual(set(children), set(self.expected_children[0]))
        self.assertEqual(set(spouses), set(self.expected_spouses[0]))


class TestDynamicMarkovBlanket(unittest.TestCase):
    """Tests for the DynamicMarkovBlanket class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create synthetic test data with time component
        np.random.seed(42)
        n_time_points = 5
        n_samples_per_time = 20
        n_vars = 5
        
        # Create a simple causal graph with temporal dependencies
        data = []
        time_values = []
        
        for t in range(n_time_points):
            # Generate data for this time point
            time_data = np.zeros((n_samples_per_time, n_vars))
            
            # Generate base variables
            time_data[:, 0] = np.random.randn(n_samples_per_time)  # X0
            time_data[:, 4] = np.random.randn(n_samples_per_time)  # X4 (independent)
            
            # Generate dependent variables
            time_data[:, 1] = 0.7 * time_data[:, 0] + 0.3 * np.random.randn(n_samples_per_time)  # X1 depends on X0
            time_data[:, 2] = 0.6 * time_data[:, 0] + 0.4 * np.random.randn(n_samples_per_time)  # X2 depends on X0
            time_data[:, 3] = 0.5 * time_data[:, 1] + 0.5 * time_data[:, 2] + 0.3 * np.random.randn(n_samples_per_time)  # X3 depends on X1 and X2
            
            # Add to data
            data.append(time_data)
            time_values.extend([t] * n_samples_per_time)
        
        # Combine all time points
        data = np.vstack(data)
        
        # Create DataFrame with time column
        self.data = pd.DataFrame(data, columns=[f'X{i}' for i in range(n_vars)])
        self.data['time'] = time_values
        
        # Initialize DynamicMarkovBlanket
        self.dmb = DynamicMarkovBlanket(self.data, time_column='time', lag=1)
    
    def test_detect_dynamic_blanket(self):
        """Test detect_dynamic_blanket method."""
        # Detect dynamic blanket for X1
        dynamic_blanket = self.dmb.detect_dynamic_blanket(1)
        
        # Check structure
        self.assertIn('dynamic_components', dynamic_blanket)
        self.assertIn('dynamic_classifications', dynamic_blanket)
        
        # Check components
        components = dynamic_blanket['dynamic_components']
        self.assertIn('current', components)
        self.assertIn('lag_1', components)
        
        # Check current time components
        current = components['current']
        self.assertIn('parents', current)
        self.assertIn('children', current)
        self.assertIn('spouses', current)
        
        # Check that X0 is a parent of X1 in current time
        self.assertIn(0, current['parents'])
        
        # Check that X3 is a child of X1 in current time
        self.assertIn(3, current['children'])
    
    def test_classify_dynamic_nodes(self):
        """Test classify_dynamic_nodes method."""
        # Classify dynamic nodes for X1
        classifications = self.dmb.classify_dynamic_nodes(1)
        
        # Check structure
        self.assertIn('current', classifications)
        self.assertIn('lag_1', classifications)
        
        # Check current time classifications
        current = classifications['current']
        self.assertIn('internal', current)
        self.assertIn('blanket', current)
        self.assertIn('external', current)
        
        # Check that X1 is internal
        self.assertEqual(current['internal'], [1])
        
        # Check that X0 and X3 are in the blanket
        self.assertIn(0, current['blanket'])
        self.assertIn(3, current['blanket'])
        
        # Check that X4 is external
        self.assertIn(4, current['external'])
    
    def test_constructor_with_tensor(self):
        """Test constructor with tensor input."""
        # Create tensor from data
        tensor_data = torch.tensor(self.data.values, dtype=torch.float32)
        
        # Initialize with tensor
        dmb_tensor = DynamicMarkovBlanket(tensor_data, time_column=5, lag=1)
        
        # Test detect_dynamic_blanket
        dynamic_blanket = dmb_tensor.detect_dynamic_blanket(1)
        
        # Check structure
        self.assertIn('dynamic_components', dynamic_blanket)
        self.assertIn('dynamic_classifications', dynamic_blanket)


if __name__ == '__main__':
    unittest.main()
