"""
Cognitive Identification in Markov Blankets
===========================================

This module provides tools for identifying cognitive structures 
within Markov blankets using information theoretic measures.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from .markov_blanket import MarkovBlanket, DynamicMarkovBlanket


class CognitiveIdentification:
    """
    Identifies cognitive structures within Markov blankets using 
    information-theoretic and causal measures.
    """
    
    def __init__(self, data: Union[pd.DataFrame, torch.Tensor], markov_blanket: Optional[MarkovBlanket] = None):
        """
        Initialize the cognitive identification system.
        
        Args:
            data: Input data as pandas DataFrame or PyTorch tensor
            markov_blanket: Optional pre-computed MarkovBlanket instance
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
            self.tensor_data = torch.tensor(data.values, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            self.tensor_data = data
            self.data = pd.DataFrame(data.numpy())
        else:
            raise TypeError("Data must be either a pandas DataFrame or a PyTorch tensor")
        
        # Create or store Markov blanket
        self.markov_blanket = markov_blanket or MarkovBlanket(data)
    
    def identify_cognitive_structures(self, target_idx: int, threshold: float = 0.0) -> Dict[str, List[int]]:
        """
        Identify cognitive structures within a Markov blanket.
        
        Args:
            target_idx: Index of the target variable
            threshold: Threshold for information flow to consider a connection significant
            
        Returns:
            structures: Dictionary with cognitive structures
        """
        # Get Markov blanket components
        classifications = self.markov_blanket.classify_nodes(target_idx)
        blanket_nodes = classifications['blanket']
        
        # Compute information flow from blanket nodes to target
        info_flow = {}
        if blanket_nodes:
            info_flow = self._compute_information_flow(blanket_nodes, target_idx)
        
        # Identify sensory and action nodes based on information flow
        sensory_nodes = []
        action_nodes = []
        
        for node in blanket_nodes:
            # Check if node is a parent or child
            is_parent = node in classifications.get('parents', [])
            is_child = node in classifications.get('children', [])
            
            # Sensory nodes are primarily parents (information flows to target)
            if is_parent and not is_child:
                sensory_nodes.append(node)
            # Action nodes are primarily children (information flows from target)
            elif is_child and not is_parent:
                action_nodes.append(node)
            # Nodes that are both parents and children are considered internal state
            # as they have bidirectional information flow
        
        # Internal state nodes are those in the blanket that are neither sensory nor action
        internal_state_nodes = [node for node in blanket_nodes 
                              if node not in sensory_nodes and node not in action_nodes]
        
        return {
            'sensory': sensory_nodes,
            'action': action_nodes,
            'internal_state': internal_state_nodes
        }
    
    def _compute_information_flow(self, source_nodes: Union[int, List[int]], 
                                 target_nodes: Union[int, List[int]]) -> Dict[int, float]:
        """
        Compute information flow from source nodes to target nodes.
        
        Args:
            source_nodes: Source node index or list of indices
            target_nodes: Target node index or list of indices
            
        Returns:
            info_flow: Dictionary mapping source nodes to information flow values
        """
        # Convert single indices to lists
        if isinstance(source_nodes, int):
            source_nodes = [source_nodes]
        if isinstance(target_nodes, int):
            target_nodes = [target_nodes]
        
        info_flow = {}
        for source in source_nodes:
            flow_values = []
            for target in target_nodes:
                # Use mutual information as a measure of information flow
                mi = self.markov_blanket.compute_mutual_information(source, target)
                flow_values.append(mi)
            
            # Average flow across all targets
            info_flow[source] = np.mean(flow_values) if flow_values else 0.0
        
        return info_flow
    
    def _compute_average_influence(self, source_nodes: Union[int, List[int]], 
                                  target_nodes: List[int]) -> float:
        """
        Compute average influence from source nodes to target nodes.
        
        Args:
            source_nodes: Source node index or list of indices
            target_nodes: List of target node indices
            
        Returns:
            avg_influence: Average influence value
        """
        # Convert single index to list
        if isinstance(source_nodes, int):
            source_nodes = [source_nodes]
        
        if not source_nodes or not target_nodes:
            return 0.0
        
        total_influence = 0.0
        count = 0
        
        for source in source_nodes:
            for target in target_nodes:
                # Use mutual information as a measure of influence
                mi = self.markov_blanket.compute_mutual_information(source, target)
                total_influence += mi
                count += 1
        
        return total_influence / count if count > 0 else 0.0
    
    def compute_cognitive_metrics(self, structures: Dict[str, List[int]]) -> Dict[str, float]:
        """
        Compute metrics related to cognitive structures.
        
        Args:
            structures: Dictionary with cognitive structures
            
        Returns:
            metrics: Dictionary with cognitive metrics
        """
        target_idx = -1  # Will be set from the Markov blanket
        sensory_nodes = structures.get('sensory', [])
        action_nodes = structures.get('action', [])
        internal_state_nodes = structures.get('internal_state', [])
        
        # Compute information flow between components
        sensory_to_internal = self._compute_average_influence(sensory_nodes, internal_state_nodes)
        internal_to_action = self._compute_average_influence(internal_state_nodes, action_nodes)
        internal_recurrence = self._compute_average_influence(internal_state_nodes, internal_state_nodes)
        
        # Compute cognitive capacity (information processing capability)
        cognitive_capacity = len(sensory_nodes) * sensory_to_internal + len(action_nodes) * internal_to_action
        
        # Compute integration (internal connectivity)
        integration = internal_recurrence * len(internal_state_nodes)
        
        # Compute complexity (balance between integration and segregation)
        complexity = cognitive_capacity * integration
        
        # Compute autonomy (ratio of internal to external influence)
        external_influence = sensory_to_internal * len(sensory_nodes)
        internal_influence = internal_recurrence * len(internal_state_nodes)
        autonomy = internal_influence / (external_influence + 1e-10)
        
        return {
            'cognitive_capacity': cognitive_capacity,
            'integration': integration,
            'complexity': complexity,
            'autonomy': autonomy,
            'sensory_to_internal': sensory_to_internal,
            'internal_to_action': internal_to_action,
            'internal_recurrence': internal_recurrence
        }
    
    def identify_temporal_cognitive_structures(self, target_idx: int, 
                                             dynamic_blanket: Optional[DynamicMarkovBlanket] = None,
                                             threshold: float = 0.0) -> Dict[str, Dict[str, List[int]]]:
        """
        Identify cognitive structures in a dynamic Markov blanket.
        
        Args:
            target_idx: Index of the target variable
            dynamic_blanket: Optional pre-initialized DynamicMarkovBlanket instance
            threshold: Threshold for information flow to consider a connection significant
            
        Returns:
            temporal_structures: Dictionary with temporal cognitive structures
        """
        # Initialize or use provided DynamicMarkovBlanket
        if dynamic_blanket is None:
            dynamic_blanket = DynamicMarkovBlanket(self.data)
        
        # Get dynamic Markov blanket
        dynamic_result = dynamic_blanket.detect_dynamic_blanket(target_idx)
        dynamic_components = dynamic_result['dynamic_components']
        
        # Process each time point
        temporal_structures = {}
        
        for time_key, components in dynamic_components.items():
            # Extract components for this time point
            parents = components.get('parents', [])
            children = components.get('children', [])
            spouses = components.get('spouses', [])
            
            # Combine to get the full blanket
            blanket_nodes = list(set(parents + children + spouses))
            
            # Identify sensory and action nodes
            sensory_nodes = []
            action_nodes = []
            
            for node in blanket_nodes:
                # Check if node is a parent or child
                is_parent = node in parents
                is_child = node in children
                
                # Sensory nodes are primarily parents
                if is_parent and not is_child:
                    sensory_nodes.append(node)
                # Action nodes are primarily children
                elif is_child and not is_parent:
                    action_nodes.append(node)
            
            # Internal state nodes are those in the blanket that are neither sensory nor action
            internal_state_nodes = [node for node in blanket_nodes 
                                  if node not in sensory_nodes and node not in action_nodes]
            
            # Store structures for this time point
            temporal_structures[time_key] = {
                'sensory': sensory_nodes,
                'action': action_nodes,
                'internal_state': internal_state_nodes
            }
        
        return temporal_structures
    
    def _compute_component_interaction(self, comp1: List[int], comp2: List[int]) -> float:
        """
        Compute interaction between two components.
        
        Args:
            comp1: First component (list of node indices)
            comp2: Second component (list of node indices)
            
        Returns:
            interaction: Interaction value
        """
        # Check if either component is empty
        if not comp1 or not comp2:
            return 0.0
        
        # Compute average influence
        return self._compute_average_influence(comp1, comp2)
