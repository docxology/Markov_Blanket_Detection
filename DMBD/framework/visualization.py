"""
Visualization Utilities for Markov Blanket Analysis
==================================================

This module provides tools for visualizing Markov blankets,
cognitive structures, and related metrics.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Union, Optional
import seaborn as sns
from .markov_blanket import MarkovBlanket, DynamicMarkovBlanket


class MarkovBlanketVisualizer:
    """Visualization tools for Markov blankets and cognitive structures."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'internal': '#ff6666',  # Red
            'blanket': '#66b3ff',   # Blue
            'external': '#c2c2c2',  # Gray
            'sensory': '#99ff99',   # Light green
            'action': '#ffcc99',    # Light orange
            'internal_state': '#cc99ff'  # Purple
        }
    
    def plot_markov_blanket(
        self, 
        mb: MarkovBlanket, 
        target_idx: int,
        node_labels: Optional[Dict[int, str]] = None,
        show_strengths: bool = True,
        ax: Optional[plt.Axes] = None,
        title: str = "Markov Blanket Structure"
    ) -> plt.Figure:
        """
        Plot the Markov blanket as a network graph.
        
        Args:
            mb: MarkovBlanket instance
            target_idx: Index of the target variable
            node_labels: Optional mapping from node indices to labels
            show_strengths: Whether to show edge strengths
            ax: Optional matplotlib axes to plot on
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        # Get Markov blanket components
        parents, children, spouses = mb.detect_blanket(target_idx)
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with classifications
        classifications = mb.classify_nodes(target_idx)
        
        # Create node labels if not provided
        if node_labels is None:
            node_labels = {i: f"X{i}" for i in classifications['internal'] + classifications['blanket'] + classifications['external'][:5]}  # Limit external nodes for clarity
        
        # Add nodes to graph
        for cat, nodes in classifications.items():
            for node in nodes:
                if node in node_labels:
                    G.add_node(node, label=node_labels[node], category=cat)
        
        # Add edges
        for parent in parents:
            if parent in G.nodes and target_idx in G.nodes:
                G.add_edge(parent, target_idx)
        
        for child in children:
            if child in G.nodes and target_idx in G.nodes:
                G.add_edge(target_idx, child)
        
        for spouse in spouses:
            for child in children:
                if spouse in G.nodes and child in G.nodes:
                    G.add_edge(spouse, child)
        
        # Get edge strengths if needed
        if show_strengths:
            edge_strengths = {}
            for u, v in G.edges():
                edge_strengths[(u, v)] = mb.compute_mutual_information(u, v)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Define node colors and sizes
        node_colors = [self.colors[G.nodes[n]['category']] for n in G.nodes()]
        node_sizes = [800 if n == target_idx else 400 for n in G.nodes()]
        
        # Define node positions with target at center
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
        
        # Draw edges with varying thickness if showing strengths
        if show_strengths:
            # Normalize edge strengths for line widths
            min_width, max_width = 1, 5
            if edge_strengths:
                min_strength = min(edge_strengths.values())
                max_strength = max(edge_strengths.values())
                range_strength = max_strength - min_strength
                
                for (u, v), strength in edge_strengths.items():
                    if range_strength > 0:
                        norm_strength = (strength - min_strength) / range_strength
                        width = min_width + norm_strength * (max_width - min_width)
                    else:
                        width = min_width
                    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], width=width)
            else:
                nx.draw_networkx_edges(G, pos, ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, labels={n: G.nodes[n]['label'] for n in G.nodes()})
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['internal'], markersize=10, label='Internal'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['blanket'], markersize=10, label='Blanket'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['external'], markersize=10, label='External')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(title)
        ax.axis('off')
        
        return fig
    
    def plot_cognitive_structures(
        self,
        mb: MarkovBlanket,
        cognitive_structures: Dict[str, List[int]],
        node_labels: Optional[Dict[int, str]] = None,
        ax: Optional[plt.Axes] = None,
        title: str = "Cognitive Structures"
    ) -> plt.Figure:
        """
        Plot cognitive structures within a Markov blanket.
        
        Args:
            mb: MarkovBlanket instance
            cognitive_structures: Dictionary with cognitive structure classifications
            node_labels: Optional mapping from node indices to labels
            ax: Optional matplotlib axes to plot on
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        # Create graph
        G = nx.DiGraph()
        
        # Extract structures
        sensory = cognitive_structures.get('sensory', [])
        action = cognitive_structures.get('action', [])
        internal_state = cognitive_structures.get('internal_state', [])
        
        # Create node labels if not provided
        all_nodes = sensory + action + internal_state
        if node_labels is None:
            node_labels = {i: f"X{i}" for i in all_nodes}
        
        # Add nodes to graph
        for node in sensory:
            G.add_node(node, label=node_labels.get(node, f"X{node}"), category='sensory')
        
        for node in action:
            G.add_node(node, label=node_labels.get(node, f"X{node}"), category='action')
        
        for node in internal_state:
            G.add_node(node, label=node_labels.get(node, f"X{node}"), category='internal_state')
        
        # Add edges
        for s in sensory:
            for i in internal_state:
                G.add_edge(s, i)
        
        for i in internal_state:
            for a in action:
                G.add_edge(i, a)
        
        # Create direct edges from sensory to action for visualization
        for s in sensory:
            for a in action:
                G.add_edge(s, a, style='dashed')
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Define node colors
        node_colors = []
        for n in G.nodes():
            if n in sensory:
                node_colors.append(self.colors['sensory'])
            elif n in action:
                node_colors.append(self.colors['action'])
            else:
                node_colors.append(self.colors['internal_state'])
        
        # Define node positions with layers (sensory -> internal -> action)
        pos = {}
        
        # Position sensory nodes at left
        for i, node in enumerate(sensory):
            pos[node] = (0, i - len(sensory)/2 + 0.5)
        
        # Position internal nodes in middle
        for i, node in enumerate(internal_state):
            pos[node] = (1, i - len(internal_state)/2 + 0.5)
        
        # Position action nodes at right
        for i, node in enumerate(action):
            pos[node] = (2, i - len(action)/2 + 0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500)
        
        # Draw solid edges (main flow)
        solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style', 'solid') == 'solid']
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=solid_edges, width=2)
        
        # Draw dashed edges (direct sensory-action)
        dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'dashed']
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dashed_edges, width=1, style='dashed', alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, labels={n: G.nodes[n]['label'] for n in G.nodes()})
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['sensory'], markersize=10, label='Sensory'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['internal_state'], markersize=10, label='Internal State'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['action'], markersize=10, label='Action')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(title)
        ax.axis('off')
        
        # Add annotations
        ax.text(-0.1, -0.9, "Sensory Layer", ha='center')
        ax.text(1, -0.9, "Internal Layer", ha='center')
        ax.text(2.1, -0.9, "Action Layer", ha='center')
        
        return fig
    
    def plot_temporal_dynamics(
        self,
        dynamic_mb: DynamicMarkovBlanket,
        target_idx: int,
        metric_func=None,
        ax: Optional[plt.Axes] = None,
        title: str = "Temporal Dynamics of Markov Blanket"
    ) -> plt.Figure:
        """
        Plot temporal dynamics of Markov blanket metrics.
        
        Args:
            dynamic_mb: DynamicMarkovBlanket instance
            target_idx: Index of the target variable
            metric_func: Function to compute metrics from blanket (defaults to blanket size)
            ax: Optional matplotlib axes to plot on
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        # Get dynamic blanket classifications
        dynamic_classes = dynamic_mb.classify_dynamic_nodes(target_idx)
        
        # Default metric function: blanket size
        if metric_func is None:
            def metric_func(mb_dict):
                return {k: len(v) for k, v in mb_dict.items()}
        
        # Compute metrics for each time point
        metrics = {}
        for time_key, classes in dynamic_classes.items():
            metrics[time_key] = metric_func(classes)
        
        # Extract time points and sort
        time_points = sorted(metrics.keys())
        
        # Extract metric types
        metric_types = list(next(iter(metrics.values())).keys())
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Plot each metric
        x = np.arange(len(time_points))
        for i, metric_name in enumerate(metric_types):
            y = [metrics[t][metric_name] for t in time_points]
            ax.plot(x, y, marker='o', label=metric_name)
        
        # Set x-axis labels and ticks
        ax.set_xticks(x)
        ax.set_xticklabels(time_points)
        
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        
        return fig
    
    def plot_information_flow(
        self,
        mb: MarkovBlanket,
        target_idx: int,
        node_labels: Optional[Dict[int, str]] = None,
        ax: Optional[plt.Axes] = None,
        title: str = "Information Flow in Markov Blanket"
    ) -> plt.Figure:
        """
        Plot the information flow between variables in a Markov blanket.
        
        Args:
            mb: MarkovBlanket instance
            target_idx: Index of the target variable
            node_labels: Optional mapping from node indices to labels
            ax: Optional matplotlib axes to plot on
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        # Get Markov blanket components
        parents, children, spouses = mb.detect_blanket(target_idx)
        
        # Combine to get all relevant nodes
        all_nodes = [target_idx] + parents + children + spouses
        
        # Create matrix of mutual information values
        n = len(all_nodes)
        mi_matrix = np.zeros((n, n))
        
        for i, node1 in enumerate(all_nodes):
            for j, node2 in enumerate(all_nodes):
                if i != j:
                    mi_matrix[i, j] = mb.compute_mutual_information(node1, node2)
        
        # Create node labels if not provided
        if node_labels is None:
            node_labels = {i: f"X{i}" for i in all_nodes}
        
        # Get labels for the matrix
        matrix_labels = [node_labels.get(node, f"X{node}") for node in all_nodes]
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Create heatmap
        sns.heatmap(mi_matrix, annot=True, cmap='viridis', xticklabels=matrix_labels, 
                   yticklabels=matrix_labels, ax=ax)
        
        ax.set_title(title)
        
        return fig
    
    def plot_blanket_components(
        self,
        mb: MarkovBlanket,
        target_idx: int,
        ax: Optional[plt.Axes] = None,
        title: str = "Markov Blanket Components"
    ) -> plt.Figure:
        """
        Plot the sizes of Markov blanket components as a bar chart.
        
        Args:
            mb: MarkovBlanket instance
            target_idx: Index of the target variable
            ax: Optional matplotlib axes to plot on
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        # Get Markov blanket components
        parents, children, spouses = mb.detect_blanket(target_idx)
        
        # Get component sizes
        sizes = {
            'Parents': len(parents),
            'Children': len(children),
            'Spouses': len(spouses),
            'Total Blanket': len(set(parents + children + spouses))
        }
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Plot as bar chart
        bars = ax.bar(sizes.keys(), sizes.values(), color=['#ff9999', '#99ff99', '#9999ff', '#ffcc99'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height}', ha='center', va='bottom')
        
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        
        return fig 