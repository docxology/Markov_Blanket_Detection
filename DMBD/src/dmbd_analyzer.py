"""
DMBD Analyzer - Main interface for Dynamic Markov Blanket Detection
==================================================================

This module provides a high-level interface for analyzing data using
the DMBD framework, with methods for detection, analysis and visualization.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import pickle
from pathlib import Path

from framework.markov_blanket import MarkovBlanket, DynamicMarkovBlanket
from framework.data_partitioning import DataPartitioning
from framework.cognitive_identification import CognitiveIdentification
from framework.visualization import MarkovBlanketVisualizer


class DMBDAnalyzer:
    """
    Main class for performing Dynamic Markov Blanket Detection and analysis.
    
    This class provides a high-level interface that combines the functionality
    of the various components in the DMBD framework.
    """
    
    def __init__(
        self, 
        data: Union[pd.DataFrame, torch.Tensor, str, Path] = None,
        time_column: Optional[str] = None,
        lag: int = 1,
        alpha: float = 0.05,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Initialize the DMBD Analyzer.
        
        Args:
            data: Input data as pandas DataFrame, PyTorch tensor, or path to CSV file
            time_column: Name of the column containing time information
            lag: Number of time lags to consider for temporal dependencies
            alpha: Significance level for conditional independence tests
            figsize: Default figure size for plots
        """
        self.data = None
        self.markov_blanket = None
        self.dynamic_markov_blanket = None
        self.data_partitioner = None
        self.cognitive_identifier = None
        self.visualizer = MarkovBlanketVisualizer(figsize=figsize)
        
        # Parameters
        self.time_column = time_column
        self.lag = lag
        self.alpha = alpha
        
        # Load data if provided
        if data is not None:
            self.load_data(data)
    
    def load_data(self, data: Union[pd.DataFrame, torch.Tensor, str, Path]) -> None:
        """
        Load data from various sources.
        
        Args:
            data: Input data as pandas DataFrame, PyTorch tensor, or path to CSV file
        """
        # Handle different data types
        if isinstance(data, (str, Path)):
            # Load from file
            file_path = Path(data)
            if file_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.pkl':
                with open(file_path, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        else:
            # Direct assignment
            self.data = data
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize all DMBD components with the current data."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Initialize MarkovBlanket
        self.markov_blanket = MarkovBlanket(self.data, alpha=self.alpha)
        
        # Initialize DynamicMarkovBlanket if time column is provided
        if self.time_column is not None:
            self.dynamic_markov_blanket = DynamicMarkovBlanket(
                self.data, time_column=self.time_column, lag=self.lag, alpha=self.alpha
            )
        else:
            # Create without time sorting
            self.dynamic_markov_blanket = DynamicMarkovBlanket(
                self.data, lag=self.lag, alpha=self.alpha
            )
        
        # Initialize DataPartitioning
        self.data_partitioner = DataPartitioning(self.data)
        
        # Initialize CognitiveIdentification
        self.cognitive_identifier = CognitiveIdentification(
            self.data, markov_blanket=self.markov_blanket
        )
    
    def detect_markov_blanket(self, target_idx: int) -> Dict[str, Any]:
        """
        Detect the Markov blanket of a target variable.
        
        Args:
            target_idx: Index of the target variable
            
        Returns:
            Dictionary with detected Markov blanket components and metrics
        """
        # Get static Markov blanket components
        parents, children, spouses = self.markov_blanket.detect_blanket(target_idx)
        
        # Classify nodes
        classifications = self.markov_blanket.classify_nodes(target_idx)
        
        # Compute blanket strengths
        strengths = self.markov_blanket.get_blanket_strength(target_idx)
        
        # Return all results
        return {
            'parents': parents,
            'children': children,
            'spouses': spouses,
            'classifications': classifications,
            'strengths': strengths,
            'blanket_size': len(set(parents + children + spouses))
        }
    
    def detect_dynamic_markov_blanket(
        self, target_idx: int, current_only: bool = False
    ) -> Dict[str, Any]:
        """
        Detect the dynamic Markov blanket of a target variable.
        
        Args:
            target_idx: Index of the target variable
            current_only: If True, only detect blanket in the current time point
            
        Returns:
            Dictionary with dynamic Markov blanket components and metrics
        """
        # Get dynamic components
        dynamic_components = self.dynamic_markov_blanket.detect_dynamic_blanket(
            target_idx, current_only=current_only
        )
        
        # Classify nodes
        dynamic_classifications = self.dynamic_markov_blanket.classify_dynamic_nodes(target_idx)
        
        # Return all results
        return {
            'dynamic_components': dynamic_components,
            'dynamic_classifications': dynamic_classifications
        }
    
    def identify_cognitive_structures(
        self, target_idx: int, threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Identify cognitive structures within the Markov blanket.
        
        Args:
            target_idx: Index of the target variable
            threshold: Threshold for identifying cognitive structures
            
        Returns:
            Dictionary with cognitive structure classifications and metrics
        """
        # Identify cognitive structures
        structures = self.cognitive_identifier.identify_cognitive_structures(
            target_idx, threshold=threshold
        )
        
        # Compute cognitive metrics
        metrics = self.cognitive_identifier.compute_cognitive_metrics(structures)
        
        # Return all results
        return {
            'structures': structures,
            'metrics': metrics
        }
    
    def identify_temporal_cognitive_structures(
        self, target_idx: int, threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Identify cognitive structures in dynamic Markov blankets.
        
        Args:
            target_idx: Index of the target variable
            threshold: Threshold for identifying cognitive structures
            
        Returns:
            Dictionary with temporal cognitive structure classifications and metrics
        """
        # Identify temporal cognitive structures
        temporal_structures = self.cognitive_identifier.identify_temporal_cognitive_structures(
            target_idx, 
            dynamic_blanket=self.dynamic_markov_blanket,
            threshold=threshold
        )
        
        # Compute metrics for each time point
        temporal_metrics = {}
        for time_key, structures in temporal_structures.items():
            temporal_metrics[time_key] = self.cognitive_identifier.compute_cognitive_metrics(structures)
        
        # Return all results
        return {
            'temporal_structures': temporal_structures,
            'temporal_metrics': temporal_metrics
        }
    
    def partition_data_by_markov_blanket(
        self, target_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Partition data based on Markov blanket components.
        
        Args:
            target_idx: Index of the target variable
            
        Returns:
            Dictionary with partitioned data tensors
        """
        # Get classifications
        classifications = self.markov_blanket.classify_nodes(target_idx)
        
        # Partition data
        partitions = self.data_partitioner.partition_data(classifications)
        
        return partitions
    
    def analyze_target(
        self, 
        target_idx: int, 
        threshold: float = 0.1,
        dynamic: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of a target variable.
        
        Args:
            target_idx: Index of the target variable
            threshold: Threshold for identifying cognitive structures
            dynamic: Whether to include dynamic analysis
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        # Basic Markov blanket detection
        mb_results = self.detect_markov_blanket(target_idx)
        
        # Cognitive structure identification
        cognitive_results = self.identify_cognitive_structures(target_idx, threshold=threshold)
        
        # Partition data
        partitions = self.partition_data_by_markov_blanket(target_idx)
        
        # Combine results
        results = {
            'markov_blanket': mb_results,
            'cognitive': cognitive_results,
            'partitions': {k: v.shape for k, v in partitions.items()}
        }
        
        # Add dynamic results if requested
        if dynamic:
            dynamic_results = self.detect_dynamic_markov_blanket(target_idx)
            temporal_cognitive_results = self.identify_temporal_cognitive_structures(
                target_idx, threshold=threshold
            )
            
            results['dynamic_markov_blanket'] = dynamic_results
            results['temporal_cognitive'] = temporal_cognitive_results
        
        return results
    
    def visualize_markov_blanket(
        self, 
        target_idx: int,
        node_labels: Optional[Dict[int, str]] = None,
        show_strengths: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the Markov blanket as a network graph.
        
        Args:
            target_idx: Index of the target variable
            node_labels: Optional mapping from node indices to labels
            show_strengths: Whether to show edge strengths
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure
        """
        fig = self.visualizer.plot_markov_blanket(
            self.markov_blanket, target_idx, node_labels=node_labels, show_strengths=show_strengths
        )
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def visualize_cognitive_structures(
        self,
        target_idx: int,
        threshold: float = 0.1,
        node_labels: Optional[Dict[int, str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize cognitive structures within a Markov blanket.
        
        Args:
            target_idx: Index of the target variable
            threshold: Threshold for identifying cognitive structures
            node_labels: Optional mapping from node indices to labels
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure
        """
        # Identify cognitive structures
        structures = self.cognitive_identifier.identify_cognitive_structures(
            target_idx, threshold=threshold
        )
        
        fig = self.visualizer.plot_cognitive_structures(
            self.markov_blanket, structures, node_labels=node_labels
        )
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def visualize_temporal_dynamics(
        self,
        target_idx: int,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize temporal dynamics of Markov blanket metrics.
        
        Args:
            target_idx: Index of the target variable
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure
        """
        fig = self.visualizer.plot_temporal_dynamics(
            self.dynamic_markov_blanket, target_idx
        )
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def visualize_information_flow(
        self,
        target_idx: int,
        node_labels: Optional[Dict[int, str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the information flow between variables in a Markov blanket.
        
        Args:
            target_idx: Index of the target variable
            node_labels: Optional mapping from node indices to labels
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure
        """
        fig = self.visualizer.plot_information_flow(
            self.markov_blanket, target_idx, node_labels=node_labels
        )
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """
        Save analysis results to a file.
        
        Args:
            results: Analysis results to save
            path: Path to save the results
        """
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    
    def load_results(self, path: str) -> Dict[str, Any]:
        """
        Load analysis results from a file.
        
        Args:
            path: Path to load the results from
            
        Returns:
            Analysis results
        """
        with open(path, 'rb') as f:
            results = pickle.load(f)
        return results 