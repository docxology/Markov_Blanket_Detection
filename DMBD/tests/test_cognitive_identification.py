"""
Tests for the CognitiveIdentification class.
"""

import unittest
import pandas as pd
import numpy as np
import torch
from framework.cognitive_identification import CognitiveIdentification
from framework.markov_blanket import MarkovBlanket, DynamicMarkovBlanket


class TestCognitiveIdentification(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Create synthetic test data with known structure
        np.random.seed(42)
        n_samples = 200
        n_vars = 6
        
        # Create a causal graph with a cognitive structure:
        # X0, X1: Sensory nodes (input)
        # X2, X3: Internal state nodes (processing)
        # X4, X5: Action nodes (output)
        # Structure: [X0,X1] -> [X2,X3] -> [X4,X5]
        
        # Initialize data
        data = np.zeros((n_samples, n_vars))
        
        # Generate sensory nodes (independent)
        data[:, 0] = np.random.randn(n_samples)  # X0
        data[:, 1] = np.random.randn(n_samples)  # X1
        
        # Generate internal state nodes (depend on sensory)
        data[:, 2] = 0.7 * data[:, 0] + 0.3 * data[:, 1] + 0.2 * np.random.randn(n_samples)  # X2
        data[:, 3] = 0.4 * data[:, 0] + 0.6 * data[:, 1] + 0.2 * np.random.randn(n_samples)  # X3
        
        # Generate action nodes (depend on internal state)
        data[:, 4] = 0.8 * data[:, 2] + 0.2 * data[:, 3] + 0.1 * np.random.randn(n_samples)  # X4
        data[:, 5] = 0.3 * data[:, 2] + 0.7 * data[:, 3] + 0.1 * np.random.randn(n_samples)  # X5
        
        # Create DataFrame
        self.data = pd.DataFrame(data, columns=[f'X{i}' for i in range(n_vars)])
        
        # Initialize MarkovBlanket
        self.mb = MarkovBlanket(self.data)
        
        # Initialize CognitiveIdentification
        self.ci = CognitiveIdentification(self.data, self.mb)
    
    def test_identify_cognitive_structures(self):
        """Test identify_cognitive_structures method."""
        # Test for target X2 (internal state node)
        target_idx = 2
        structures = self.ci.identify_cognitive_structures(target_idx, threshold=0.1)
        
        # Check structure of result
        self.assertIn('sensory', structures)
        self.assertIn('action', structures)
        self.assertIn('internal_state', structures)
        
        # Check specific nodes
        # X0 and X1 should be detected as sensory for X2
        for node in [0, 1]:
            self.assertIn(node, structures['sensory'])
        
        # X4 and X5 should be detected as action for X2
        for node in [4, 5]:
            self.assertIn(node, structures['action'])
    
    def test_compute_information_flow(self):
        """Test _compute_information_flow method."""
        # Get information flow from X0 to [X2, X3]
        source_idx = 0
        target_indices = [2, 3]
        
        # This is a protected method, but we're testing it directly for thoroughness
        flows = self.ci._compute_information_flow(source_idx, target_indices)
        
        # Check keys
        for target in target_indices:
            self.assertIn(target, flows)
        
        # Flow values should be positive
        for flow in flows.values():
            self.assertGreaterEqual(flow, 0)
    
    def test_compute_average_influence(self):
        """Test _compute_average_influence method."""
        # Compute influence of X2 on [X4, X5]
        node_idx = 2
        other_nodes = [4, 5]
        
        # This is a protected method, but we're testing it directly for thoroughness
        influence = self.ci._compute_average_influence(node_idx, other_nodes)
        
        # Influence should be positive
        self.assertGreaterEqual(influence, 0)
        
        # Empty other_nodes should return 0
        influence_empty = self.ci._compute_average_influence(node_idx, [])
        self.assertEqual(influence_empty, 0)
    
    def test_compute_cognitive_metrics(self):
        """Test compute_cognitive_metrics method."""
        # Identify cognitive structures first
        target_idx = 2
        structures = self.ci.identify_cognitive_structures(target_idx, threshold=0.1)
        
        # Compute metrics
        metrics = self.ci.compute_cognitive_metrics(structures)
        
        # Check keys
        self.assertIn('cognitive_capacity', metrics)
        self.assertIn('integration', metrics)
        self.assertIn('complexity', metrics)
        
        # Metrics should be non-negative
        for metric in metrics.values():
            self.assertGreaterEqual(metric, 0)
    
    def test_identify_temporal_cognitive_structures(self):
        """Test identify_temporal_cognitive_structures method."""
        # Create a DynamicMarkovBlanket first
        dmb = DynamicMarkovBlanket(self.data, lag=1)
        
        # Identify temporal structures
        target_idx = 2
        temporal_structures = self.ci.identify_temporal_cognitive_structures(
            target_idx, dynamic_blanket=dmb, threshold=0.1
        )
        
        # Check structure of result
        self.assertIsInstance(temporal_structures, dict)
        
        # Should have current time point
        self.assertIn('current', temporal_structures)
        
        # Current time point should have cognitive structures
        current = temporal_structures['current']
        self.assertIn('sensory', current)
        self.assertIn('action', current)
        self.assertIn('internal_state', current)
    
    def test_compute_component_interaction(self):
        """Test _compute_component_interaction method."""
        # Compute interaction between sensory [X0, X1] and internal [X2, X3]
        comp1 = [0, 1]
        comp2 = [2, 3]
        
        # This is a protected method, but we're testing it directly for thoroughness
        interaction = self.ci._compute_component_interaction(comp1, comp2)
        
        # Interaction should be positive
        self.assertGreaterEqual(interaction, 0)
        
        # Empty components should return 0
        interaction_empty = self.ci._compute_component_interaction([], comp2)
        self.assertEqual(interaction_empty, 0)
        
        interaction_empty = self.ci._compute_component_interaction(comp1, [])
        self.assertEqual(interaction_empty, 0)
    
    def test_constructor_with_tensor(self):
        """Test construction with a PyTorch tensor."""
        # Create tensor from data
        tensor_data = torch.tensor(self.data.values, dtype=torch.float32)
        
        # Create MarkovBlanket with tensor
        mb_tensor = MarkovBlanket(tensor_data)
        
        # Initialize with tensor
        ci_tensor = CognitiveIdentification(tensor_data, mb_tensor)
        
        # Test basic functionality
        structures = ci_tensor.identify_cognitive_structures(2, threshold=0.1)
        
        # Check structure of result
        self.assertIn('sensory', structures)
        self.assertIn('action', structures)
        self.assertIn('internal_state', structures)


if __name__ == '__main__':
    unittest.main()
