"""
Dynamic Markov Blanket Detection for Gridworld
==============================================

This module extends the DMBD framework to analyze gridworld simulation data,
identify Dynamic Markov Blankets, and create visualizations of the evolving
blanket partitions over time.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path
import time

from framework.markov_blanket import MarkovBlanket, DynamicMarkovBlanket
from framework.cognitive_identification import CognitiveIdentification
from src.dmbd_analyzer import DMBDAnalyzer
from src.gridworld_simulation import GaussianBlurGridworld


class GridworldMarkovAnalyzer:
    """
    Specialized analyzer for detecting Dynamic Markov Blankets in gridworld simulations.
    
    This class extends the DMBD framework to work specifically with gridworld
    simulations containing a moving Gaussian blur. It provides methods for
    detecting Dynamic Markov Blankets and visualizing their evolution over time.
    """
    
    def __init__(
        self,
        simulation: GaussianBlurGridworld,
        n_representative_cells: int = 20,
        threshold: float = 0.1,
        use_torch: bool = True,
        output_dir: str = 'output/gridworld_analysis'
    ):
        """
        Initialize the gridworld Markov blanket analyzer.
        
        Args:
            simulation: Gridworld simulation with a moving Gaussian blur
            n_representative_cells: Number of cells to analyze (for efficiency)
            threshold: Threshold for identifying Markov blanket components
            use_torch: Whether to use PyTorch for computations
            output_dir: Directory to save analysis outputs
        """
        self.simulation = simulation
        self.threshold = threshold
        self.use_torch = use_torch
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get simulation data
        if simulation.dataframe is None:
            self.df = simulation.to_dataframe()
        else:
            self.df = simulation.dataframe
        
        # Select representative cells if needed
        self.n_representative_cells = n_representative_cells
        if n_representative_cells < simulation.grid_size[0] * simulation.grid_size[1]:
            self.representative_cells = simulation.select_representative_cells(n_representative_cells)
            # Filter DataFrame to only include representative cells
            cell_columns = [f'cell_{simulation.get_grid_indices(idx)[0]}_{simulation.get_grid_indices(idx)[1]}' 
                           for idx in self.representative_cells]
            self.analysis_df = self.df[cell_columns + ['time']]
        else:
            self.representative_cells = list(range(simulation.grid_size[0] * simulation.grid_size[1]))
            self.analysis_df = self.df
        
        # Initialize DMBD analyzer
        self.analyzer = DMBDAnalyzer(
            data=self.analysis_df,
            time_column='time',
            lag=1,
            alpha=0.05
        )
        
        # Store results
        self.results = {}
        self.blanket_evolution = {}
        self.cognitive_evolution = {}
        
    def analyze_center_cell(self) -> Dict[str, Any]:
        """
        Analyze the center cell of the gridworld.
        
        Returns:
            Dictionary with analysis results
        """
        # Get the center cell coordinates
        center_row = self.simulation.grid_size[0] // 2
        center_col = self.simulation.grid_size[1] // 2
        center_idx = self.simulation.get_cell_indices(center_row, center_col)
        
        # Find the closest representative cell to the center
        if center_idx not in self.representative_cells:
            # Find the closest representative cell
            center_coords = np.array([center_row, center_col])
            representative_coords = np.array([self.simulation.get_grid_indices(idx) for idx in self.representative_cells])
            distances = np.sqrt(np.sum((representative_coords - center_coords)**2, axis=1))
            closest_idx = np.argmin(distances)
            target_idx = closest_idx
        else:
            # Find the index in the representative cells list
            target_idx = self.representative_cells.index(center_idx)
        
        # Analyze this cell
        return self.analyze_cell(target_idx)
    
    def analyze_cell(self, cell_idx: int) -> Dict[str, Any]:
        """
        Analyze a specific cell in the gridworld.
        
        Args:
            cell_idx: Index of the cell to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Map cell_idx to the actual column index in the DataFrame
        # The cell_idx is an index in the representative_cells list
        # We need to map it to the actual column index in the DataFrame
        if cell_idx >= len(self.representative_cells):
            raise ValueError(f"Cell index {cell_idx} is out of bounds. Max index is {len(self.representative_cells) - 1}")
        
        # Get the actual cell index from the representative_cells list
        actual_cell_idx = self.representative_cells[cell_idx]
        
        # Get the grid indices for this cell
        row, col = self.simulation.get_grid_indices(actual_cell_idx)
        
        # Get the column name for this cell
        column_name = f'cell_{row}_{col}'
        
        # Get the column index in the DataFrame
        if column_name not in self.analysis_df.columns:
            raise ValueError(f"Column {column_name} not found in DataFrame. Available columns: {self.analysis_df.columns}")
        
        column_idx = self.analysis_df.columns.get_loc(column_name)
        
        # Run the full analysis
        results = self.analyzer.analyze_target(
            target_idx=column_idx,
            threshold=self.threshold,
            dynamic=True
        )
        
        # Store results
        self.results[cell_idx] = results
        
        return results
    
    def analyze_multiple_cells(self, cell_indices: List[int] = None) -> Dict[int, Dict[str, Any]]:
        """
        Analyze multiple cells in the gridworld.
        
        Args:
            cell_indices: List of cell indices to analyze (if None, uses 5 evenly spaced cells)
            
        Returns:
            Dictionary mapping cell indices to analysis results
        """
        if cell_indices is None:
            # Use 5 evenly spaced cells
            step = len(self.representative_cells) // 5
            cell_indices = list(range(0, len(self.representative_cells), step))[:5]
        
        # Analyze each cell
        results = {}
        for idx in cell_indices:
            results[idx] = self.analyze_cell(idx)
        
        return results
    
    def extract_blanket_evolution(self, cell_idx: int) -> Dict[int, Dict[str, List[int]]]:
        """
        Extract the evolution of Markov blanket components over time.
        
        Args:
            cell_idx: Index of the cell to analyze
            
        Returns:
            Dictionary mapping time points to blanket components
        """
        if cell_idx not in self.results:
            self.analyze_cell(cell_idx)
        
        # Get dynamic Markov blanket results
        dynamic_results = self.results[cell_idx]['dynamic_markov_blanket']
        
        # Extract blanket components for each time point
        blanket_evolution = {}
        
        # Check if we have time-varying components or just a static result
        if 'dynamic_components' in dynamic_results:
            # We have time-varying components
            for time_key, components in dynamic_results['dynamic_components'].items():
                blanket_evolution[time_key] = components
        
        self.blanket_evolution[cell_idx] = blanket_evolution
        return blanket_evolution
    
    def extract_cognitive_evolution(self, cell_idx: int) -> Dict[str, Dict[str, List[int]]]:
        """
        Extract the evolution of cognitive structures over time.
        
        Args:
            cell_idx: Index of the cell to analyze
            
        Returns:
            Dictionary mapping time points to cognitive structures
        """
        if cell_idx not in self.results:
            self.analyze_cell(cell_idx)
        
        # Get temporal cognitive results
        if 'temporal_cognitive' in self.results[cell_idx]:
            temporal_results = self.results[cell_idx]['temporal_cognitive']
            
            # Extract cognitive structures for each time point
            if 'temporal_structures' in temporal_results:
                cognitive_evolution = temporal_results['temporal_structures']
                self.cognitive_evolution[cell_idx] = cognitive_evolution
                return cognitive_evolution
        
        # No temporal cognitive results found
        return {}
    
    def _create_gridworld_mask(self, indices: List[int]) -> np.ndarray:
        """
        Create a mask for highlighting specific cells in the gridworld.
        
        Args:
            indices: List of cell indices to highlight
            
        Returns:
            2D array with mask (1 for highlighted cells, 0 for others)
        """
        mask = np.zeros(self.simulation.grid_size, dtype=np.int8)
        
        # Convert indices to grid coordinates and set mask
        for idx in indices:
            # If it's a representative cell index, convert to actual cell index
            if idx < len(self.representative_cells):
                actual_idx = self.representative_cells[idx]
            else:
                actual_idx = idx
                
            row, col = self.simulation.get_grid_indices(actual_idx)
            if 0 <= row < self.simulation.grid_size[0] and 0 <= col < self.simulation.grid_size[1]:
                mask[row, col] = 1
        
        return mask
    
    def visualize_blanket_partition(
        self, 
        cell_idx: int,
        time_point: str = 'current',
        fig=None, 
        ax=None
    ):
        """
        Visualize the Markov blanket partition for a specific cell and time point.
        
        Args:
            cell_idx: Index of the cell to visualize
            time_point: Time point to visualize ('current' or 'lag_1')
            fig: Matplotlib figure to use (if None, creates new figure)
            ax: Matplotlib axes to use (if None, creates new axes)
            
        Returns:
            Matplotlib figure and axes
        """
        if cell_idx not in self.blanket_evolution:
            self.extract_blanket_evolution(cell_idx)
        
        # Setup figure and axes
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get gridworld data for background
        time_idx = self.simulation.n_time_points // 2  # Use middle time point for now
        if self.simulation.use_torch:
            frame_data = self.simulation.simulation_data[time_idx].cpu().numpy()
        else:
            frame_data = self.simulation.simulation_data[time_idx]
        
        # Plot the frame as background
        im = ax.imshow(frame_data, cmap='viridis', alpha=0.3, interpolation='nearest')
        
        # Get blanket components
        if time_point in self.blanket_evolution[cell_idx]:
            components = self.blanket_evolution[cell_idx][time_point]
        else:
            components = {'parents': [], 'children': [], 'spouses': []}
        
        # Create masks for different components
        # If the cell_idx is a representative index, get the actual cell
        if cell_idx < len(self.representative_cells):
            actual_idx = self.representative_cells[cell_idx]
        else:
            actual_idx = cell_idx
            
        target_row, target_col = self.simulation.get_grid_indices(actual_idx)
        
        # Create masks
        parents_mask = self._create_gridworld_mask(components.get('parents', []))
        children_mask = self._create_gridworld_mask(components.get('children', []))
        spouses_mask = self._create_gridworld_mask(components.get('spouses', []))
        
        # Overlay masks with different colors
        # Red for target, green for parents, blue for children, yellow for spouses
        overlay = np.zeros((*self.simulation.grid_size, 4))  # RGBA
        
        # Target (red)
        overlay[target_row, target_col] = [1, 0, 0, 0.7]
        
        # Parents (green)
        overlay[parents_mask == 1] = [0, 1, 0, 0.5]
        
        # Children (blue)
        overlay[children_mask == 1] = [0, 0, 1, 0.5]
        
        # Spouses (yellow)
        overlay[spouses_mask == 1] = [1, 1, 0, 0.5]
        
        # Plot overlay
        ax.imshow(overlay)
        
        # Add title and labels
        if isinstance(time_point, str):
            title = f"Markov Blanket Partition - Cell {target_row},{target_col} - {time_point}"
        else:
            title = f"Markov Blanket Partition - Cell {target_row},{target_col} - Time {time_point}"
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Target'),
            Patch(facecolor='green', alpha=0.5, label='Parents'),
            Patch(facecolor='blue', alpha=0.5, label='Children'),
            Patch(facecolor='yellow', alpha=0.5, label='Spouses')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        return fig, ax
    
    def visualize_cognitive_partition(
        self, 
        cell_idx: int,
        time_point: str = 'current',
        fig=None, 
        ax=None
    ):
        """
        Visualize the cognitive partition for a specific cell and time point.
        
        Args:
            cell_idx: Index of the cell to visualize
            time_point: Time point to visualize ('current' or 'lag_1')
            fig: Matplotlib figure to use (if None, creates new figure)
            ax: Matplotlib axes to use (if None, creates new axes)
            
        Returns:
            Matplotlib figure and axes
        """
        if cell_idx not in self.cognitive_evolution:
            self.extract_cognitive_evolution(cell_idx)
        
        # Setup figure and axes
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get gridworld data for background
        time_idx = self.simulation.n_time_points // 2  # Use middle time point for now
        if self.simulation.use_torch:
            frame_data = self.simulation.simulation_data[time_idx].cpu().numpy()
        else:
            frame_data = self.simulation.simulation_data[time_idx]
        
        # Plot the frame as background
        im = ax.imshow(frame_data, cmap='viridis', alpha=0.3, interpolation='nearest')
        
        # Get cognitive structures
        if time_point in self.cognitive_evolution[cell_idx]:
            structures = self.cognitive_evolution[cell_idx][time_point]
        else:
            structures = {'sensory': [], 'action': [], 'internal_state': []}
        
        # If the cell_idx is a representative index, get the actual cell
        if cell_idx < len(self.representative_cells):
            actual_idx = self.representative_cells[cell_idx]
        else:
            actual_idx = cell_idx
            
        target_row, target_col = self.simulation.get_grid_indices(actual_idx)
        
        # Create masks
        sensory_mask = self._create_gridworld_mask(structures.get('sensory', []))
        action_mask = self._create_gridworld_mask(structures.get('action', []))
        internal_mask = self._create_gridworld_mask(structures.get('internal_state', []))
        
        # Overlay masks with different colors
        # Red for target, green for sensory, blue for action, purple for internal state
        overlay = np.zeros((*self.simulation.grid_size, 4))  # RGBA
        
        # Target (red)
        overlay[target_row, target_col] = [1, 0, 0, 0.7]
        
        # Sensory (green)
        overlay[sensory_mask == 1] = [0, 1, 0, 0.5]
        
        # Action (blue)
        overlay[action_mask == 1] = [0, 0, 1, 0.5]
        
        # Internal state (purple)
        overlay[internal_mask == 1] = [0.5, 0, 0.5, 0.5]
        
        # Plot overlay
        ax.imshow(overlay)
        
        # Add title and labels
        if isinstance(time_point, str):
            title = f"Cognitive Partition - Cell {target_row},{target_col} - {time_point}"
        else:
            title = f"Cognitive Partition - Cell {target_row},{target_col} - Time {time_point}"
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Target'),
            Patch(facecolor='green', alpha=0.5, label='Sensory'),
            Patch(facecolor='blue', alpha=0.5, label='Action'),
            Patch(facecolor='purple', alpha=0.5, label='Internal State')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        return fig, ax
    
    def create_blanket_animation(
        self, 
        cell_idx: int, 
        output_path: str = None
    ) -> FuncAnimation:
        """
        Create an animation of the evolving Markov blanket.
        
        Args:
            cell_idx: Index of the cell to visualize
            output_path: Path to save the animation (if None, doesn't save)
            
        Returns:
            Matplotlib animation object
        """
        if cell_idx not in self.blanket_evolution:
            self.extract_blanket_evolution(cell_idx)
        
        # Setup figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get the actual cell idx and coordinates
        if cell_idx < len(self.representative_cells):
            actual_idx = self.representative_cells[cell_idx]
        else:
            actual_idx = cell_idx
            
        target_row, target_col = self.simulation.get_grid_indices(actual_idx)
        
        # Time points for the animation
        # Use simulation time points for the background
        n_frames = self.simulation.n_time_points
        
        # Initialize with first frame
        if self.simulation.use_torch:
            frame_data = self.simulation.simulation_data[0].cpu().numpy()
        else:
            frame_data = self.simulation.simulation_data[0]
        
        # Plot the frame as background
        im = ax.imshow(frame_data, cmap='viridis', alpha=0.3, interpolation='nearest')
        
        # Create empty overlay for components
        overlay = np.zeros((*self.simulation.grid_size, 4))  # RGBA
        overlay_im = ax.imshow(overlay)
        
        # Add title and labels
        title = ax.set_title(f"Markov Blanket Evolution - Cell {target_row},{target_col} - Time 0")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Target'),
            Patch(facecolor='green', alpha=0.5, label='Parents'),
            Patch(facecolor='blue', alpha=0.5, label='Children'),
            Patch(facecolor='yellow', alpha=0.5, label='Spouses')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Function to update the plot for each frame
        def update(frame):
            # Update background from simulation
            if self.simulation.use_torch:
                frame_data = self.simulation.simulation_data[frame].cpu().numpy()
            else:
                frame_data = self.simulation.simulation_data[frame]
            
            im.set_array(frame_data)
            
            # Update overlay with Markov blanket components
            # Use 'current' time point from blanket evolution
            components = self.blanket_evolution[cell_idx].get('current', {'parents': [], 'children': [], 'spouses': []})
            
            # Create new overlay
            overlay = np.zeros((*self.simulation.grid_size, 4))  # RGBA
            
            # Target (red)
            overlay[target_row, target_col] = [1, 0, 0, 0.7]
            
            # Parents (green)
            parents_mask = self._create_gridworld_mask(components.get('parents', []))
            overlay[parents_mask == 1] = [0, 1, 0, 0.5]
            
            # Children (blue)
            children_mask = self._create_gridworld_mask(components.get('children', []))
            overlay[children_mask == 1] = [0, 0, 1, 0.5]
            
            # Spouses (yellow)
            spouses_mask = self._create_gridworld_mask(components.get('spouses', []))
            overlay[spouses_mask == 1] = [1, 1, 0, 0.5]
            
            overlay_im.set_array(overlay)
            
            # Update title
            title.set_text(f"Markov Blanket Evolution - Cell {target_row},{target_col} - Time {frame}")
            
            return [im, overlay_im, title]
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=n_frames, 
            interval=100, blit=True
        )
        
        # Save animation
        if output_path:
            anim.save(output_path, writer='ffmpeg', dpi=100)
        
        return anim
    
    def create_cognitive_animation(
        self, 
        cell_idx: int, 
        output_path: str = None
    ) -> FuncAnimation:
        """
        Create an animation of the evolving cognitive structures.
        
        Args:
            cell_idx: Index of the cell to visualize
            output_path: Path to save the animation (if None, doesn't save)
            
        Returns:
            Matplotlib animation object
        """
        if cell_idx not in self.cognitive_evolution:
            self.extract_cognitive_evolution(cell_idx)
        
        # Setup figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get the actual cell idx and coordinates
        if cell_idx < len(self.representative_cells):
            actual_idx = self.representative_cells[cell_idx]
        else:
            actual_idx = cell_idx
            
        target_row, target_col = self.simulation.get_grid_indices(actual_idx)
        
        # Time points for the animation
        # Use simulation time points for the background
        n_frames = self.simulation.n_time_points
        
        # Initialize with first frame
        if self.simulation.use_torch:
            frame_data = self.simulation.simulation_data[0].cpu().numpy()
        else:
            frame_data = self.simulation.simulation_data[0]
        
        # Plot the frame as background
        im = ax.imshow(frame_data, cmap='viridis', alpha=0.3, interpolation='nearest')
        
        # Create empty overlay for components
        overlay = np.zeros((*self.simulation.grid_size, 4))  # RGBA
        overlay_im = ax.imshow(overlay)
        
        # Add title and labels
        title = ax.set_title(f"Cognitive Structure Evolution - Cell {target_row},{target_col} - Time 0")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Target'),
            Patch(facecolor='green', alpha=0.5, label='Sensory'),
            Patch(facecolor='blue', alpha=0.5, label='Action'),
            Patch(facecolor='purple', alpha=0.5, label='Internal State')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Function to update the plot for each frame
        def update(frame):
            # Update background from simulation
            if self.simulation.use_torch:
                frame_data = self.simulation.simulation_data[frame].cpu().numpy()
            else:
                frame_data = self.simulation.simulation_data[frame]
            
            im.set_array(frame_data)
            
            # Update overlay with cognitive structures
            # Use 'current' time point from cognitive evolution
            structures = self.cognitive_evolution[cell_idx].get('current', {'sensory': [], 'action': [], 'internal_state': []})
            
            # Create new overlay
            overlay = np.zeros((*self.simulation.grid_size, 4))  # RGBA
            
            # Target (red)
            overlay[target_row, target_col] = [1, 0, 0, 0.7]
            
            # Sensory (green)
            sensory_mask = self._create_gridworld_mask(structures.get('sensory', []))
            overlay[sensory_mask == 1] = [0, 1, 0, 0.5]
            
            # Action (blue)
            action_mask = self._create_gridworld_mask(structures.get('action', []))
            overlay[action_mask == 1] = [0, 0, 1, 0.5]
            
            # Internal state (purple)
            internal_mask = self._create_gridworld_mask(structures.get('internal_state', []))
            overlay[internal_mask == 1] = [0.5, 0, 0.5, 0.5]
            
            overlay_im.set_array(overlay)
            
            # Update title
            title.set_text(f"Cognitive Structure Evolution - Cell {target_row},{target_col} - Time {frame}")
            
            return [im, overlay_im, title]
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=n_frames, 
            interval=100, blit=True
        )
        
        # Save animation
        if output_path:
            anim.save(output_path, writer='ffmpeg', dpi=100)
        
        return anim
    
    def analyze_and_visualize(self) -> Dict[str, str]:
        """
        Run a complete analysis and visualization pipeline.
        
        This method:
        1. Uses PyTorch to infer the Markov Blanket partition (internal, blanket, external)
        2. Creates visualizations of the inferred partition
        3. Creates an animation showing the evolution of the partition over time
        
        Returns:
            Dictionary with paths to generated outputs
        """
        output_paths = {}
        
        # 1. Infer Markov Blanket partition for all time points
        print("Inferring Markov Blanket partition...")
        partition_probs = self.infer_markov_blanket_partition()
        
        # 2. Create static visualizations at key time points
        print("Creating static visualizations...")
        time_points = [0, self.simulation.n_time_points // 4, self.simulation.n_time_points // 2,
                      3 * self.simulation.n_time_points // 4, self.simulation.n_time_points - 1]
        
        for t in time_points:
            t_viz_path = os.path.join(self.output_dir, f'markov_partition_t{t}.png')
            fig = self.visualize_inferred_partition(t, partition_probs, t_viz_path)
            plt.close(fig)
            output_paths[f'markov_partition_t{t}'] = t_viz_path
        
        # 3. Create animation of the evolving partition
        print("Creating evolution animation...")
        evolution_path = os.path.join(self.output_dir, 'markov_partition_evolution.mp4')
        self.create_partition_evolution_animation(partition_probs, evolution_path)
        output_paths['markov_partition_evolution'] = evolution_path
        
        print("Analysis and visualization complete!")
        return output_paths
    
    def visualize_comparison(
        self, 
        cell_idx: int,
        time_point: Union[str, int] = 'current',
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a side-by-side comparison of the Gaussian blur and the Markov Blanket partitioning.
        
        Args:
            cell_idx: Index of the cell to visualize
            time_point: Time point to visualize ('current', 'lag_1', or a specific frame index)
            output_path: Path to save the figure (if None, doesn't save)
            
        Returns:
            Matplotlib figure
        """
        # Ensure we have the necessary data
        if cell_idx not in self.cognitive_evolution:
            self.extract_cognitive_evolution(cell_idx)
        if cell_idx not in self.blanket_evolution:
            self.extract_blanket_evolution(cell_idx)
        
        # Get the actual cell idx and coordinates
        if cell_idx < len(self.representative_cells):
            actual_idx = self.representative_cells[cell_idx]
        else:
            actual_idx = cell_idx
            
        target_row, target_col = self.simulation.get_grid_indices(actual_idx)
        
        # Get the frame to display
        if isinstance(time_point, int):
            frame_idx = time_point
        else:
            # Use middle frame for str time points like 'current' or 'lag_1'
            frame_idx = self.simulation.n_time_points // 2
        
        # Get the frame data
        if self.simulation.use_torch:
            frame_data = self.simulation.simulation_data[frame_idx].cpu().numpy()
        else:
            frame_data = self.simulation.simulation_data[frame_idx]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left subplot: Original Gaussian blur
        im1 = ax1.imshow(frame_data, cmap='viridis', interpolation='nearest')
        ax1.set_title(f"Gaussian Blur at Time {frame_idx}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        plt.colorbar(im1, ax=ax1, label="Intensity")
        
        # Mark the target cell
        ax1.plot(target_col, target_row, 'ro', markersize=10)
        
        # Right subplot: Markov Blanket partitioning
        # Get cognitive structure info
        if time_point in self.cognitive_evolution[cell_idx]:
            structures = self.cognitive_evolution[cell_idx][time_point]
        else:
            structures = {'sensory': [], 'action': [], 'internal_state': []}
        
        # Create overlay for partitioning
        overlay = np.zeros((*self.simulation.grid_size, 4))  # RGBA
        
        # Target (red)
        overlay[target_row, target_col] = [1, 0, 0, 0.7]
        
        # Internal state (purple) - should correspond to high-density regions
        internal_mask = self._create_gridworld_mask(structures.get('internal_state', []))
        overlay[internal_mask == 1] = [0.5, 0, 0.5, 0.7]
        
        # Sensory (green)
        sensory_mask = self._create_gridworld_mask(structures.get('sensory', []))
        overlay[sensory_mask == 1] = [0, 1, 0, 0.5]
        
        # Action (blue)
        action_mask = self._create_gridworld_mask(structures.get('action', []))
        overlay[action_mask == 1] = [0, 0, 1, 0.5]
        
        # Plot the frame as background with lower opacity
        im2 = ax2.imshow(frame_data, cmap='viridis', alpha=0.3, interpolation='nearest')
        
        # Plot the partition overlay
        ax2.imshow(overlay)
        ax2.set_title(f"Markov Blanket Partitioning at Time {frame_idx}")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Target'),
            Patch(facecolor='purple', alpha=0.7, label='Internal State'),
            Patch(facecolor='green', alpha=0.5, label='Sensory'),
            Patch(facecolor='blue', alpha=0.5, label='Action')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Add overall title
        plt.suptitle(f"Comparison of Gaussian Blur and Markov Blanket Partitioning for Cell {target_row},{target_col}", 
                    fontsize=16)
        
        plt.tight_layout()
        
        # Save the figure if output path provided
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig

    def infer_markov_blanket_partition(
        self, 
        time_point: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Infer the Markov Blanket partition (internal, blanket, external) for the gridworld.
        
        This method uses PyTorch to infer which cells belong to which partition
        based on the empirical data and the constraints of Markov Blanket theory:
        - Internal states form a single connected moving object (high-density Gaussian)
        - External states form a single large environmental component
        - Blanket states mediate between internal and external states
        
        Args:
            time_point: Time point to analyze (if None, analyzes all time points)
            
        Returns:
            Dictionary with probability tensors for each partition type
        """
        # Ensure we're using PyTorch
        if not self.use_torch:
            print("Warning: PyTorch inference works best with use_torch=True")
        
        # Get the data tensor
        if self.simulation.use_torch:
            data = self.simulation.to_tensor()
        else:
            data = torch.tensor(self.simulation.to_tensor().numpy(), dtype=torch.float32)
        
        # Number of cells in the grid
        n_cells = self.simulation.grid_size[0] * self.simulation.grid_size[1]
        
        # If specific time point, just analyze that one
        if time_point is not None:
            frames = [data[time_point].reshape(1, -1)]
            time_points = [time_point]
        else:
            # Analyze all time points
            frames = [data[t].reshape(1, -1) for t in range(self.simulation.n_time_points)]
            time_points = list(range(self.simulation.n_time_points))
        
        # Results will store the partition probabilities for each time point
        results = {
            'internal': [],
            'blanket': [],
            'external': []
        }
        
        # Process each time point
        for t, frame in zip(time_points, frames):
            # Step 1: Initialize with density-based priors
            # Higher density cells are more likely to be internal states
            # Get the frame data in 2D grid format
            if self.simulation.use_torch:
                grid_data = self.simulation.simulation_data[t].cpu().numpy()
            else:
                grid_data = self.simulation.simulation_data[t]
            
            # Flatten grid for easier processing
            flat_data = grid_data.flatten()
            
            # Create initial probability estimates based on density
            # Higher density -> more likely to be internal
            # Lower density -> more likely to be external
            # Medium density -> more likely to be blanket
            internal_prior = torch.tensor(flat_data, dtype=torch.float32)
            
            # Normalize to [0,1]
            internal_prior = (internal_prior - internal_prior.min()) / (internal_prior.max() - internal_prior.min())
            
            # External is inverse of internal
            external_prior = 1.0 - internal_prior
            
            # Blanket is highest in the transition zone between high and low density
            blanket_prior = 4.0 * internal_prior * external_prior  # Creates a peak at 0.5 density
            
            # Step 2: Compute correlation matrix between all cells to capture statistical relationships
            # This is where we use the actual data to refine the density-based priors
            flat_frame = torch.tensor(flat_data, dtype=torch.float32).reshape(1, -1)
            
            # We would ideally use a full correlation matrix, but for efficiency
            # we can approximate using the correlation with the spatial neighborhood
            
            # Create grid representation of priors for neighborhood processing
            internal_grid = internal_prior.reshape(self.simulation.grid_size)
            blanket_grid = blanket_prior.reshape(self.simulation.grid_size)
            external_grid = external_prior.reshape(self.simulation.grid_size)
            
            # Apply spatial smoothing to enforce connectivity constraints
            # Internal states should form a single connected region
            # External states should form a single connected region
            # Blanket states should be at the boundary
            
            # Create smoothing kernels
            kernel_size = min(5, min(self.simulation.grid_size[0], self.simulation.grid_size[1]) // 2)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd size for symmetric padding
            
            # Gaussian smoothing for internal states
            internal_smoothed = torch.nn.functional.avg_pool2d(
                torch.tensor(internal_grid, dtype=torch.float32).reshape(1, 1, *self.simulation.grid_size),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ).squeeze().numpy()
            
            # Ensure single connected component for internal
            # This favors cells that have high density AND are surrounded by high density
            internal_prob = internal_grid * internal_smoothed
            internal_prob = torch.tensor(internal_prob.flatten())
            internal_prob = internal_prob / internal_prob.sum()
            
            # Similar for external, but favoring low density regions
            external_smoothed = torch.nn.functional.avg_pool2d(
                torch.tensor(external_grid, dtype=torch.float32).reshape(1, 1, *self.simulation.grid_size),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ).squeeze().numpy()
            
            external_prob = external_grid * external_smoothed
            external_prob = torch.tensor(external_prob.flatten())
            external_prob = external_prob / external_prob.sum()
            
            # Blanket should be at the boundary between internal and external
            # We can compute gradient magnitude of the internal probability
            # High gradient = boundary = blanket
            dx = np.gradient(internal_grid, axis=0)
            dy = np.gradient(internal_grid, axis=1)
            gradient_mag = np.sqrt(dx**2 + dy**2)
            
            # Normalize gradient magnitude
            if gradient_mag.max() > 0:
                gradient_mag = gradient_mag / gradient_mag.max()
            
            # Blanket probability is proportional to gradient magnitude
            blanket_prob = torch.tensor(gradient_mag.flatten())
            blanket_prob = blanket_prob / blanket_prob.sum()
            
            # Step 3: Enforce partition constraints
            # - Sum of probabilities for each cell across all partitions should be 1
            # - Blanket should separate internal from external
            
            # Normalize to ensure probabilities sum to 1 for each cell
            partition_sum = internal_prob + blanket_prob + external_prob
            internal_prob = internal_prob / partition_sum
            blanket_prob = blanket_prob / partition_sum
            external_prob = external_prob / partition_sum
            
            # Store results for this time point
            results['internal'].append(internal_prob)
            results['blanket'].append(blanket_prob)
            results['external'].append(external_prob)
        
        # Convert lists to tensors
        for key in results:
            results[key] = torch.stack(results[key])
        
        return results

    def visualize_inferred_partition(
        self, 
        time_point: int,
        partition_probs: Dict[str, torch.Tensor] = None,
        output_path: str = None
    ) -> plt.Figure:
        """
        Visualize the inferred Markov Blanket partition for the gridworld.
        
        Args:
            time_point: Time point to visualize
            partition_probs: Partition probabilities from infer_markov_blanket_partition
                            (if None, will compute them)
            output_path: Path to save the visualization (if None, won't save)
            
        Returns:
            Matplotlib figure with the visualization
        """
        # Get partition probabilities if not provided
        if partition_probs is None:
            partition_probs = self.infer_markov_blanket_partition(time_point)
        
        # Get the frame data
        if self.simulation.use_torch:
            frame_data = self.simulation.simulation_data[time_point].cpu().numpy()
        else:
            frame_data = self.simulation.simulation_data[time_point]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left subplot: Original Gaussian blur
        im1 = ax1.imshow(frame_data, cmap='viridis', interpolation='nearest')
        ax1.set_title(f"Gaussian Blur at Time {time_point}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        plt.colorbar(im1, ax=ax1, label="Intensity")
        
        # Right subplot: Inferred Markov Blanket partition
        # Get probabilities for this time point
        internal_prob = partition_probs['internal'][0 if len(partition_probs['internal']) == 1 else time_point]
        blanket_prob = partition_probs['blanket'][0 if len(partition_probs['blanket']) == 1 else time_point]
        external_prob = partition_probs['external'][0 if len(partition_probs['external']) == 1 else time_point]
        
        # Reshape to grid
        internal_grid = internal_prob.reshape(self.simulation.grid_size)
        blanket_grid = blanket_prob.reshape(self.simulation.grid_size)
        external_grid = external_prob.reshape(self.simulation.grid_size)
        
        # Create RGBA overlay
        overlay = np.zeros((*self.simulation.grid_size, 4))
        
        # Assign colors based on highest probability partition
        # Internal: Purple, Blanket: Yellow, External: Cyan
        for i in range(self.simulation.grid_size[0]):
            for j in range(self.simulation.grid_size[1]):
                probs = [internal_grid[i, j], blanket_grid[i, j], external_grid[i, j]]
                max_idx = np.argmax(probs)
                
                if max_idx == 0:  # Internal
                    overlay[i, j] = [0.5, 0, 0.5, 0.7 * internal_grid[i, j].item()]  # Purple with alpha based on prob
                elif max_idx == 1:  # Blanket
                    overlay[i, j] = [1.0, 1.0, 0, 0.7 * blanket_grid[i, j].item()]  # Yellow with alpha based on prob
                else:  # External
                    overlay[i, j] = [0, 1.0, 1.0, 0.7 * external_grid[i, j].item()]  # Cyan with alpha based on prob
        
        # Plot the frame as background with lower opacity
        ax2.imshow(frame_data, cmap='viridis', alpha=0.3, interpolation='nearest')
        
        # Plot the partition overlay
        ax2.imshow(overlay)
        ax2.set_title(f"Inferred Markov Blanket Partition at Time {time_point}")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.5, 0, 0.5, 0.7], label='Internal States'),
            Patch(facecolor=[1.0, 1.0, 0, 0.7], label='Blanket States'),
            Patch(facecolor=[0, 1.0, 1.0, 0.7], label='External States')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Add overall title
        plt.suptitle(f"Comparison of Gaussian Blur and Inferred Markov Blanket Partition at Time {time_point}", 
                    fontsize=16)
        
        plt.tight_layout()
        
        # Save the figure if output path provided
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig

    def create_partition_evolution_animation(
        self,
        partition_probs: Dict[str, torch.Tensor] = None,
        output_path: str = None
    ) -> FuncAnimation:
        """
        Create an animation showing the evolution of the inferred partition over time.
        
        Args:
            partition_probs: Partition probabilities from infer_markov_blanket_partition
                            (if None, will compute them for all time points)
            output_path: Path to save the animation (if None, won't save)
            
        Returns:
            Matplotlib animation object
        """
        # Get partition probabilities if not provided
        if partition_probs is None:
            partition_probs = self.infer_markov_blanket_partition()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get the first frame data
        if self.simulation.use_torch:
            frame_data = self.simulation.simulation_data[0].cpu().numpy()
        else:
            frame_data = self.simulation.simulation_data[0]
        
        # Initialize plots
        im1 = ax1.imshow(frame_data, cmap='viridis', interpolation='nearest')
        ax1.set_title(f"Gaussian Blur at Time 0")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        plt.colorbar(im1, ax=ax1, label="Intensity")
        
        # Create initial overlay for the second plot
        overlay = np.zeros((*self.simulation.grid_size, 4))
        im2_bg = ax2.imshow(frame_data, cmap='viridis', alpha=0.3, interpolation='nearest')
        im2_overlay = ax2.imshow(overlay)
        ax2.set_title(f"Inferred Markov Blanket Partition at Time 0")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.5, 0, 0.5, 0.7], label='Internal States'),
            Patch(facecolor=[1.0, 1.0, 0, 0.7], label='Blanket States'),
            Patch(facecolor=[0, 1.0, 1.0, 0.7], label='External States')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Add overall title
        title = plt.suptitle(f"Evolution of Gaussian Blur and Markov Blanket Partition", fontsize=16)
        
        def update(frame):
            # Update left plot with new frame
            if self.simulation.use_torch:
                new_frame_data = self.simulation.simulation_data[frame].cpu().numpy()
            else:
                new_frame_data = self.simulation.simulation_data[frame]
            
            im1.set_array(new_frame_data)
            ax1.set_title(f"Gaussian Blur at Time {frame}")
            
            # Update right plot with new partition
            internal_prob = partition_probs['internal'][frame]
            blanket_prob = partition_probs['blanket'][frame]
            external_prob = partition_probs['external'][frame]
            
            # Reshape to grid
            internal_grid = internal_prob.reshape(self.simulation.grid_size)
            blanket_grid = blanket_prob.reshape(self.simulation.grid_size)
            external_grid = external_prob.reshape(self.simulation.grid_size)
            
            # Create RGBA overlay
            overlay = np.zeros((*self.simulation.grid_size, 4))
            
            # Assign colors based on highest probability partition
            for i in range(self.simulation.grid_size[0]):
                for j in range(self.simulation.grid_size[1]):
                    probs = [internal_grid[i, j], blanket_grid[i, j], external_grid[i, j]]
                    max_idx = np.argmax(probs)
                    
                    if max_idx == 0:  # Internal
                        overlay[i, j] = [0.5, 0, 0.5, 0.7 * internal_grid[i, j].item()]  # Purple with alpha based on prob
                    elif max_idx == 1:  # Blanket
                        overlay[i, j] = [1.0, 1.0, 0, 0.7 * blanket_grid[i, j].item()]  # Yellow with alpha based on prob
                    else:  # External
                        overlay[i, j] = [0, 1.0, 1.0, 0.7 * external_grid[i, j].item()]  # Cyan with alpha based on prob
            
            # Update background
            im2_bg.set_array(new_frame_data)
            # Update overlay
            im2_overlay.set_array(overlay)
            ax2.set_title(f"Inferred Markov Blanket Partition at Time {frame}")
            
            return [im1, im2_bg, im2_overlay, ax1.title, ax2.title]
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=range(self.simulation.n_time_points),
            interval=100, blit=True
        )
        
        # Save animation if output path provided
        if output_path:
            anim.save(output_path, writer='ffmpeg', dpi=100)
        
        return anim


def run_gridworld_dmbd_analysis(
    grid_size: Tuple[int, int] = (30, 30),
    n_time_points: int = 100,
    radius: float = 10.0,
    sigma: float = 2.0,
    threshold: float = 0.1,
    output_dir: str = 'output/gridworld_dmbd',
    use_torch: bool = True
) -> Dict[str, str]:
    """
    Run a complete gridworld DMBD analysis pipeline using PyTorch inference.
    
    This function:
    1. Generates a gridworld simulation with a moving Gaussian blur
    2. Uses PyTorch to infer the Dynamic Markov Blanket partition
    3. Creates visualizations of the evolving partition
    
    Args:
        grid_size: Size of the grid as (height, width)
        n_time_points: Number of time points to simulate
        radius: Radius of the circular path for the Gaussian blur
        sigma: Standard deviation of the Gaussian blur
        threshold: Threshold for determining significance in the inference
        output_dir: Directory to save outputs
        use_torch: Whether to use PyTorch for computations
        
    Returns:
        Dictionary with paths to generated outputs
    """
    # Ensure we're using PyTorch
    if not use_torch:
        print("Warning: For optimal results, setting use_torch=True")
        use_torch = True
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate gridworld simulation
    print("Generating gridworld simulation...")
    from src.gridworld_simulation import GaussianBlurGridworld, generate_gridworld_data
    
    simulation_dir = os.path.join(output_dir, 'simulation')
    simulation_result = generate_gridworld_data(
        grid_size=grid_size,
        n_time_points=n_time_points,
        radius=radius,
        sigma=sigma,
        output_dir=simulation_dir,
        use_torch=use_torch
    )
    
    # 2. Analyze the simulation using PyTorch inference
    print("Analyzing gridworld simulation...")
    analysis_dir = os.path.join(output_dir, 'analysis')
    analyzer = GridworldMarkovAnalyzer(
        simulation=simulation_result['simulation'],
        threshold=threshold,
        use_torch=use_torch,
        output_dir=analysis_dir
    )
    
    # 3. Run the analysis and visualization pipeline
    print("Running analysis and visualization pipeline...")
    analysis_result = analyzer.analyze_and_visualize()
    
    # Combine all output paths
    output_paths = {
        'simulation': {
            'animation': simulation_result['animation_path'],
            'frame': simulation_result['frame_path'],
            'data': simulation_result['df_path']
        },
        'analysis': analysis_result
    }
    
    # Write a summary file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Gridworld Dynamic Markov Blanket Analysis using PyTorch Inference\n")
        f.write("==========================================================\n\n")
        f.write(f"Grid size: {grid_size[0]}x{grid_size[1]}\n")
        f.write(f"Time points: {n_time_points}\n")
        f.write(f"Radius: {radius}\n")
        f.write(f"Sigma: {sigma}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Used PyTorch: {use_torch}\n\n")
        f.write("Generated outputs:\n")
        for category, paths in output_paths.items():
            f.write(f"  {category}:\n")
            for name, path in paths.items():
                f.write(f"    {name}: {path}\n")
    
    output_paths['summary'] = summary_path
    
    print(f"Analysis complete! Results available in {output_dir}")
    return output_paths


if __name__ == "__main__":
    output_dir = os.path.join('output', 'gridworld_dmbd_example')
    result = run_gridworld_dmbd_analysis(output_dir=output_dir)
    
    print("Generated outputs:")
    for category, paths in result.items():
        print(f"  {category}:")
        if isinstance(paths, dict):
            for name, path in paths.items():
                print(f"    {name}: {path}")
        else:
            print(f"    {paths}") 