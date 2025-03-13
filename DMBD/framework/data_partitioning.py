"""
Data Partitioning for Markov Blanket Detection
==============================================

This module provides utilities for partitioning and preprocessing data
for Markov blanket detection and analysis.
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Union, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataPartitioning:
    """
    Class for partitioning data based on Markov blanket components.
    
    This class provides methods to partition data into internal, blanket, and external
    components, as well as create train/test splits and temporal partitions.
    """
    
    def __init__(self, data: Union[pd.DataFrame, torch.Tensor]):
        """
        Initialize the DataPartitioning class.
        
        Args:
            data: Input data as DataFrame or Tensor
        """
        # Convert tensor to DataFrame if needed
        if isinstance(data, torch.Tensor):
            self.data = pd.DataFrame(data.numpy())
            self.tensor_data = data
        else:
            self.data = data
            self.tensor_data = torch.tensor(data.values, dtype=torch.float32)
    
    def partition_data(self, classifications: Dict[str, List[int]], return_tensors: bool = False) -> Dict[str, Union[pd.DataFrame, torch.Tensor]]:
        """
        Partition data based on node classifications.
        
        Args:
            classifications: Dictionary with 'internal', 'blanket', and 'external' keys
            return_tensors: Whether to return PyTorch tensors instead of DataFrames
            
        Returns:
            partitions: Dictionary with partitioned data
        """
        # Check if classifications has the required keys
        required_keys = ['internal', 'blanket', 'external']
        for key in required_keys:
            if key not in classifications:
                raise ValueError(f"Classifications must contain '{key}' key")
        
        # Create partitions
        partitions = {}
        for key in required_keys:
            indices = classifications[key]
            if return_tensors:
                partitions[key] = self.tensor_data[:, indices]
            else:
                partitions[key] = self.data.iloc[:, indices]
        
        return partitions
    
    def create_train_test_split(self, data: Optional[Union[pd.DataFrame, torch.Tensor]] = None, 
                               test_size: float = 0.2, random_state: Optional[int] = None,
                               return_tensors: bool = False) -> Tuple[Union[pd.DataFrame, torch.Tensor], Union[pd.DataFrame, torch.Tensor]]:
        """
        Create a train/test split of the data.
        
        Args:
            data: Data to split (uses self.data if None)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            return_tensors: Whether to return PyTorch tensors instead of DataFrames
            
        Returns:
            train_data: Training data
            test_data: Testing data
        """
        if data is None:
            data = self.data if not return_tensors else self.tensor_data
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Get number of samples
        n_samples = len(data)
        n_test = int(n_samples * test_size)
        
        # Generate random indices
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # Split data
        if isinstance(data, pd.DataFrame) and not return_tensors:
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
        else:
            if isinstance(data, pd.DataFrame):
                data = torch.tensor(data.values, dtype=torch.float32)
            train_data = data[train_indices]
            test_data = data[test_indices]
        
        return train_data, test_data
    
    def create_temporal_partition(self, time_column: Union[str, int], 
                                 n_history: int = 1,
                                 return_tensors: bool = False) -> Dict[str, Union[pd.DataFrame, torch.Tensor]]:
        """
        Create a temporal partition of the data.
        
        Args:
            time_column: Column name or index for time information
            n_history: Number of time steps to include in history
            return_tensors: Whether to return PyTorch tensors instead of DataFrames
            
        Returns:
            temporal_partition: Dictionary with 'history' and 'future' keys
        """
        # Extract time information
        if isinstance(self.data, pd.DataFrame):
            if isinstance(time_column, str):
                time_values = self.data[time_column].values
                data_no_time = self.data.drop(columns=[time_column])
            else:
                time_values = self.data.iloc[:, time_column].values
                data_no_time = self.data.drop(self.data.columns[time_column], axis=1)
        else:
            if isinstance(time_column, int):
                time_values = self.tensor_data[:, time_column].numpy()
                data_no_time = torch.cat([self.tensor_data[:, :time_column], 
                                         self.tensor_data[:, time_column+1:]], dim=1)
            else:
                raise ValueError("For tensor data, time_column must be an integer index")
        
        # Get unique time points and sort them
        unique_times = np.unique(time_values)
        unique_times.sort()
        
        # Create mapping from time to index
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        
        # Create history and future partitions
        history_data = []
        future_data = []
        
        for t_idx in range(n_history, len(unique_times)):
            # Get current time point
            current_time = unique_times[t_idx]
            current_mask = time_values == current_time
            current_data = data_no_time.iloc[current_mask] if isinstance(data_no_time, pd.DataFrame) else data_no_time[current_mask]
            
            # Add to future data
            future_data.append(current_data)
            
            # Get history time points
            history_for_current = []
            for h_idx in range(1, n_history + 1):
                history_time = unique_times[t_idx - h_idx]
                history_mask = time_values == history_time
                history_data_point = data_no_time.iloc[history_mask] if isinstance(data_no_time, pd.DataFrame) else data_no_time[history_mask]
                history_for_current.append(history_data_point)
            
            # Combine history for current time point
            if isinstance(data_no_time, pd.DataFrame):
                combined_history = pd.concat(history_for_current, axis=1)
            else:
                combined_history = torch.cat(history_for_current, dim=1)
            
            # Add to history data
            history_data.append(combined_history)
        
        # Combine all time points
        if isinstance(data_no_time, pd.DataFrame) and not return_tensors:
            combined_history = pd.concat(history_data)
            combined_future = pd.concat(future_data)
        else:
            if isinstance(data_no_time, pd.DataFrame):
                # Convert to tensors
                history_tensors = [torch.tensor(h.values, dtype=torch.float32) for h in history_data]
                future_tensors = [torch.tensor(f.values, dtype=torch.float32) for f in future_data]
                combined_history = torch.cat(history_tensors)
                combined_future = torch.cat(future_tensors)
            else:
                combined_history = torch.cat(history_data)
                combined_future = torch.cat(future_data)
        
        return {
            'history': combined_history,
            'future': combined_future
        }
    
    def partition_by_markov_blanket(self, classifications: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
        """
        Partition data by Markov blanket components.
        
        Args:
            classifications: Dictionary with node classifications
            
        Returns:
            partitions: Dictionary with partitioned data as tensors
        """
        # Create partitions
        partitions = {}
        for key in ['internal', 'blanket', 'external']:
            if key in classifications:
                indices = classifications[key]
                partitions[key] = self.tensor_data[:, indices]
        
        return partitions
    
    def create_batches(self, data: Optional[Union[pd.DataFrame, torch.Tensor]] = None, 
                      batch_size: int = 32, shuffle: bool = True,
                      return_tensors: bool = False) -> List[Union[pd.DataFrame, torch.Tensor]]:
        """
        Create batches from the data.
        
        Args:
            data: Data to batch (uses self.data if None)
            batch_size: Size of each batch
            shuffle: Whether to shuffle the data
            return_tensors: Whether to return PyTorch tensors instead of DataFrames
            
        Returns:
            batches: List of batches
        """
        if data is None:
            data = self.data if not return_tensors else self.tensor_data
        
        # Get number of samples
        n_samples = len(data)
        
        # Generate indices
        indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
        
        # Create batches
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            if isinstance(data, pd.DataFrame) and not return_tensors:
                batch = data.iloc[batch_indices]
            else:
                if isinstance(data, pd.DataFrame):
                    batch = torch.tensor(data.iloc[batch_indices].values, dtype=torch.float32)
                else:
                    batch = data[batch_indices]
            batches.append(batch)
        
        return batches
    
    def compute_statistics(self, data: Optional[Union[pd.DataFrame, torch.Tensor]] = None) -> Dict[str, np.ndarray]:
        """
        Compute basic statistics of the data.
        
        Args:
            data: Data to analyze (uses self.data if None)
            
        Returns:
            stats: Dictionary with statistics
        """
        if data is None:
            data = self.data
        
        # Convert to numpy for consistent processing
        if isinstance(data, pd.DataFrame):
            data_np = data.values
        else:
            data_np = data.numpy() if isinstance(data, torch.Tensor) else data
        
        # Compute statistics
        mean = np.mean(data_np, axis=0)
        std = np.std(data_np, axis=0)
        median = np.median(data_np, axis=0)
        min_vals = np.min(data_np, axis=0)
        max_vals = np.max(data_np, axis=0)
        
        return {
            'mean': mean,
            'std': std,
            'median': median,
            'min': min_vals,
            'max': max_vals
        }
