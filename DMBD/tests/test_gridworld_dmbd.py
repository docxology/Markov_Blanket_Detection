"""
Tests for the gridworld Dynamic Markov Blanket detection module.

This module contains tests for the gridworld simulation and DMBD analysis
classes and functions defined in src/gridworld_simulation.py and src/gridworld_dmbd.py.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # Use non-interactive backend for testing

from src.gridworld_simulation import GaussianBlurGridworld, generate_gridworld_data
from src.gridworld_dmbd import GridworldMarkovAnalyzer, run_gridworld_dmbd_analysis
from framework.markov_blanket import MarkovBlanket


class TestGaussianBlurGridworld(unittest.TestCase):
    """Tests for the GaussianBlurGridworld class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.grid_size = (10, 10)
        self.n_time_points = 5
        self.simulation = GaussianBlurGridworld(
            grid_size=self.grid_size,
            n_time_points=self.n_time_points,
            radius=5.0,
            sigma=1.0,
            use_torch=False,
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of GaussianBlurGridworld."""
        self.assertEqual(self.simulation.grid_size, self.grid_size)
        self.assertEqual(self.simulation.n_time_points, self.n_time_points)
        self.assertEqual(self.simulation.radius, 5.0)
        self.assertEqual(self.simulation.sigma, 1.0)
        self.assertEqual(self.simulation.use_torch, False)
        self.assertEqual(self.simulation.center, (5, 5))
    
    def test_run_simulation(self):
        """Test running simulation."""
        data = self.simulation.run_simulation()
        self.assertEqual(data.shape, (self.n_time_points, *self.grid_size))
        self.assertTrue(np.all(data >= 0))
        self.assertTrue(np.all(data <= 1))
    
    def test_compute_position(self):
        """Test computation of Gaussian position."""
        # For time point 0, position should be at radius along x-axis
        x, y = self.simulation._compute_position(0)
        self.assertAlmostEqual(x, 10.0, places=4)  # center_x + radius
        self.assertAlmostEqual(y, 5.0, places=4)  # center_y
        
        # For time point 1/4 of the way around, position should be along y-axis
        x, y = self.simulation._compute_position(self.n_time_points // 4)
        self.assertAlmostEqual(x, 5.0, places=4)  # center_x
        self.assertAlmostEqual(y, 10.0, places=4)  # center_y + radius
    
    def test_generate_gaussian(self):
        """Test generation of Gaussian distribution."""
        gaussian = self.simulation._generate_gaussian(5.0, 5.0)
        self.assertEqual(gaussian.shape, self.grid_size)
        self.assertAlmostEqual(gaussian[5, 5], 1.0, places=4)  # center value should be 1
        self.assertTrue(np.all(gaussian >= 0))
        self.assertTrue(np.all(gaussian <= 1))
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        self.simulation.run_simulation()
        df = self.simulation.to_dataframe()
        self.assertEqual(len(df), self.n_time_points)
        self.assertEqual(len(df.columns), self.grid_size[0] * self.grid_size[1] + 1)  # +1 for time column
        self.assertTrue('time' in df.columns)
        self.assertTrue('cell_0_0' in df.columns)
    
    def test_to_tensor(self):
        """Test conversion to tensor."""
        self.simulation.run_simulation()
        tensor = self.simulation.to_tensor()
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual(tensor.shape, (self.n_time_points, self.grid_size[0] * self.grid_size[1]))
    
    def test_visualize_frame(self):
        """Test visualization of a frame."""
        self.simulation.run_simulation()
        fig = self.simulation.visualize_frame(time_point=0)
        self.assertTrue(isinstance(fig, plt.Figure))
    
    def test_create_animation(self):
        """Test creation of animation."""
        self.simulation.run_simulation()
        anim = self.simulation.create_animation()
        self.assertTrue(anim is not None)
    
    def test_get_cell_indices(self):
        """Test conversion between grid and linear indices."""
        idx = self.simulation.get_cell_indices(1, 2)
        self.assertEqual(idx, 1 * self.grid_size[1] + 2)
        
        row, col = self.simulation.get_grid_indices(idx)
        self.assertEqual(row, 1)
        self.assertEqual(col, 2)
    
    def test_select_representative_cells(self):
        """Test selection of representative cells."""
        n_cells = 5
        cells = self.simulation.select_representative_cells(n_cells)
        self.assertEqual(len(cells), n_cells)
        # Check that all indices are valid
        for idx in cells:
            self.assertTrue(0 <= idx < self.grid_size[0] * self.grid_size[1])


class TestGridworldMarkovAnalyzer(unittest.TestCase):
    """Tests for the GridworldMarkovAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.grid_size = (8, 8)
        self.n_time_points = 5
        self.simulation = GaussianBlurGridworld(
            grid_size=self.grid_size,
            n_time_points=self.n_time_points,
            radius=3.0,
            sigma=1.0,
            use_torch=False,
            output_dir=os.path.join(self.temp_dir, 'simulation')
        )
        self.simulation.run_simulation()
        self.analyzer = GridworldMarkovAnalyzer(
            simulation=self.simulation,
            n_representative_cells=10,
            threshold=0.1,
            use_torch=False,
            output_dir=os.path.join(self.temp_dir, 'analysis')
        )
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of GridworldMarkovAnalyzer."""
        self.assertEqual(self.analyzer.simulation, self.simulation)
        self.assertEqual(self.analyzer.threshold, 0.1)
        self.assertEqual(self.analyzer.use_torch, False)
        self.assertEqual(len(self.analyzer.representative_cells), 10)
    
    def test_analyze_center_cell(self):
        """Test analysis of center cell."""
        results = self.analyzer.analyze_center_cell()
        self.assertTrue(isinstance(results, dict))
        self.assertTrue('markov_blanket' in results)
        self.assertTrue('dynamic_markov_blanket' in results)
    
    def test_analyze_cell(self):
        """Test analysis of a specific cell."""
        cell_idx = 0  # Use the first representative cell
        results = self.analyzer.analyze_cell(cell_idx)
        self.assertTrue(isinstance(results, dict))
        self.assertTrue('markov_blanket' in results)
        self.assertTrue('dynamic_markov_blanket' in results)
    
    def test_extract_blanket_evolution(self):
        """Test extraction of blanket evolution."""
        cell_idx = 0  # Use the first representative cell
        self.analyzer.analyze_cell(cell_idx)
        evolution = self.analyzer.extract_blanket_evolution(cell_idx)
        self.assertTrue(isinstance(evolution, dict))
    
    def test_extract_cognitive_evolution(self):
        """Test extraction of cognitive evolution."""
        cell_idx = 0  # Use the first representative cell
        self.analyzer.analyze_cell(cell_idx)
        evolution = self.analyzer.extract_cognitive_evolution(cell_idx)
        self.assertTrue(isinstance(evolution, dict))
    
    def test_create_gridworld_mask(self):
        """Test creation of gridworld mask."""
        indices = [0, 1, 2]
        mask = self.analyzer._create_gridworld_mask(indices)
        self.assertEqual(mask.shape, self.grid_size)
        self.assertEqual(np.sum(mask), len(indices))
    
    def test_visualize_blanket_partition(self):
        """Test visualization of blanket partition."""
        cell_idx = 0  # Use the first representative cell
        self.analyzer.analyze_cell(cell_idx)
        fig, ax = self.analyzer.visualize_blanket_partition(cell_idx, time_point=0)
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(isinstance(ax, plt.Axes))
    
    def test_visualize_cognitive_partition(self):
        """Test visualization of cognitive partition."""
        cell_idx = 0  # Use the first representative cell
        self.analyzer.analyze_cell(cell_idx)
        fig, ax = self.analyzer.visualize_cognitive_partition(cell_idx, time_point=0)
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(isinstance(ax, plt.Axes))


class TestGridworldIntegration(unittest.TestCase):
    """Integration tests for the gridworld DMBD analysis."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_gridworld_data(self):
        """Test the gridworld data generation function."""
        result = generate_gridworld_data(
            grid_size=(5, 5),
            n_time_points=3,
            radius=2.0,
            sigma=0.5,
            use_torch=False,
            output_dir=self.temp_dir
        )
        self.assertTrue('df_path' in result)
        self.assertTrue('animation_path' in result)
        self.assertTrue('frame_path' in result)
        self.assertTrue('simulation' in result)
        self.assertTrue(os.path.exists(result['df_path']))
        self.assertTrue(os.path.exists(result['animation_path']))
        self.assertTrue(os.path.exists(result['frame_path']))
    
    def test_run_gridworld_dmbd_analysis(self):
        """Test the full DMBD analysis pipeline."""
        result = run_gridworld_dmbd_analysis(
            grid_size=(5, 5),
            n_time_points=3,
            radius=2.0,
            sigma=0.5,
            threshold=0.1,
            output_dir=self.temp_dir,
            use_torch=False
        )
        self.assertTrue('simulation' in result)
        self.assertTrue('analysis' in result)
        self.assertTrue('summary' in result)
        self.assertTrue(os.path.exists(result['summary']))
        # Check simulation outputs
        self.assertTrue('animation' in result['simulation'])
        self.assertTrue('frame' in result['simulation'])
        self.assertTrue('data' in result['simulation'])
        # Check analysis outputs - updated for new partition visualization naming
        self.assertTrue('markov_partition_t0' in result['analysis'])
        self.assertTrue('markov_partition_evolution' in result['analysis'])


if __name__ == '__main__':
    unittest.main() 