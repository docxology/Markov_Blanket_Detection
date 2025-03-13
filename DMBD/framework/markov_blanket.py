"""
Module for Markov Blanket detection in static and dynamic data.
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from typing import Tuple, List, Dict, Union, Optional, Any


class MarkovBlanket:
    """
    Class for detecting Markov Blankets in data.
    
    A Markov Blanket of a node X consists of:
    - Parents: Direct causes of X
    - Children: Direct effects of X
    - Spouses: Other direct causes of X's children
    
    This class provides methods to detect these components using
    conditional independence tests.
    """
    
    def __init__(self, data: Union[pd.DataFrame, torch.Tensor], alpha: float = 0.05):
        """
        Initialize the MarkovBlanket detector.
        
        Args:
            data: Input data as DataFrame or Tensor
            alpha: Significance level for independence tests
        """
        self.alpha = alpha
        
        # Convert tensor to DataFrame if needed
        if isinstance(data, torch.Tensor):
            self.data = pd.DataFrame(data.numpy())
            self.tensor_data = data
        else:
            self.data = data
            self.tensor_data = torch.tensor(data.values, dtype=torch.float32)
        
        # Compute correlation matrix for quick independence tests
        self.corr_matrix = np.corrcoef(self.data.values, rowvar=False)
        self.n_vars = self.data.shape[1]
        
        # Store variable names
        if isinstance(data, pd.DataFrame) and data.columns is not None:
            self.var_names = data.columns
        else:
            self.var_names = [f'X{i}' for i in range(self.n_vars)]
    
    def _conditional_independence_test(self, x_idx: int, y_idx: int, 
                                      z_indices: Optional[List[int]] = None) -> Tuple[float, bool]:
        """
        Perform conditional independence test between variables.
        
        Args:
            x_idx: Index of first variable
            y_idx: Index of second variable
            z_indices: Indices of conditioning variables
            
        Returns:
            p_value: P-value of the test
            is_independent: Boolean indicating independence
        """
        # Ensure indices are within bounds
        if x_idx >= self.n_vars or y_idx >= self.n_vars:
            return 1.0, True  # Out of bounds, assume independence
            
        # Simple correlation test if no conditioning set
        if z_indices is None or len(z_indices) == 0:
            corr_val = self.corr_matrix[x_idx, y_idx]
            # Fisher's z-transformation for correlation test
            z_val = 0.5 * np.log((1 + corr_val) / (1 - corr_val + 1e-10))
            std_err = 1 / np.sqrt(self.data.shape[0] - 3)
            p_val = 2 * (1 - stats.norm.cdf(abs(z_val) / std_err))
            return p_val, p_val > self.alpha
        
        # Partial correlation test with conditioning set
        x_data = self.data.iloc[:, x_idx].values
        y_data = self.data.iloc[:, y_idx].values
        z_data = self.data.iloc[:, z_indices].values
        
        # Residualize x and y on z
        x_resid = self._residualize(x_data, z_data)
        y_resid = self._residualize(y_data, z_data)
        
        # Compute correlation between residuals
        corr_val, p_val = stats.pearsonr(x_resid, y_resid)
        
        return p_val, p_val > self.alpha
    
    def _residualize(self, target: np.ndarray, predictors: np.ndarray) -> np.ndarray:
        """
        Residualize target variable on predictors.
        
        Args:
            target: Target variable
            predictors: Predictor variables
            
        Returns:
            residuals: Residuals after regression
        """
        # Add constant term
        if len(predictors.shape) == 1:
            predictors = predictors.reshape(-1, 1)
        
        predictors_with_const = np.column_stack([np.ones(predictors.shape[0]), predictors])
        
        # Compute regression coefficients
        try:
            beta = np.linalg.lstsq(predictors_with_const, target, rcond=None)[0]
            # Compute predicted values and residuals
            predicted = predictors_with_const @ beta
            residuals = target - predicted
            return residuals
        except np.linalg.LinAlgError:
            # If singular matrix, return original target
            return target
    
    def detect_blanket(self, target_idx: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Detect the Markov Blanket of a target variable.
        
        Args:
            target_idx: Index of the target variable
            
        Returns:
            parents: List of parent indices
            children: List of children indices
            spouses: List of spouse indices
        """
        # Initialize empty lists for blanket components
        parents = []
        children = []
        spouses = []
        
        # Step 1: Find potential parents and children (direct neighbors)
        neighbors = []
        for i in range(self.n_vars):
            if i == target_idx:
                continue
                
            # Test marginal independence
            p_val, is_independent = self._conditional_independence_test(target_idx, i)
            if not is_independent:
                neighbors.append(i)
        
        # Step 2: Separate parents and children
        for i in neighbors:
            # Try to find a subset of neighbors that d-separates i from target
            is_child = True
            for j in neighbors:
                if i == j:
                    continue
                    
                # Test if j blocks the path between target and i
                _, is_independent = self._conditional_independence_test(target_idx, i, [j])
                if is_independent:
                    is_child = False
                    break
            
            if is_child:
                children.append(i)
            else:
                parents.append(i)
        
        # Step 3: Find spouses (parents of children)
        for child in children:
            for i in range(self.n_vars):
                if i == target_idx or i in neighbors:
                    continue
                    
                # Test if i is connected to child
                p_val, is_independent = self._conditional_independence_test(child, i)
                if not is_independent:
                    # Test if i is a spouse (connected to child but not to target except through child)
                    _, is_spouse_independent = self._conditional_independence_test(target_idx, i, [child])
                    if not is_spouse_independent and i not in spouses:
                        spouses.append(i)
        
        return parents, children, spouses
    
    def classify_nodes(self, target_idx: int) -> Dict[str, List[int]]:
        """
        Classify nodes into internal, blanket, and external sets.
        
        Args:
            target_idx: Index of the target variable
            
        Returns:
            classifications: Dictionary with node classifications
        """
        parents, children, spouses = self.detect_blanket(target_idx)
        
        # Combine all blanket components
        blanket_nodes = list(set(parents + children + spouses))
        
        # All other nodes are external
        external_nodes = [i for i in range(self.n_vars) 
                         if i != target_idx and i not in blanket_nodes]
        
        return {
            'internal': [target_idx],
            'blanket': blanket_nodes,
            'external': external_nodes,
            'parents': parents,
            'children': children,
            'spouses': spouses
        }
    
    def compute_mutual_information(self, x_idx: int, y_idx: int) -> float:
        """
        Compute mutual information between two variables.
        
        Args:
            x_idx: Index of first variable
            y_idx: Index of second variable
            
        Returns:
            mi: Mutual information value
        """
        # Extract data
        x = self.data.iloc[:, x_idx].values
        y = self.data.iloc[:, y_idx].values
        
        # Compute correlation
        corr = np.corrcoef(x, y)[0, 1]
        
        # Mutual information for Gaussian variables
        mi = -0.5 * np.log(1 - corr**2 + 1e-10)
        
        return mi
    
    def get_blanket_strength(self, target_idx: int) -> Dict[int, float]:
        """
        Compute the strength of connections in the Markov Blanket.
        
        Args:
            target_idx: Index of the target variable
            
        Returns:
            strengths: Dictionary mapping node indices to connection strengths
        """
        parents, children, spouses = self.detect_blanket(target_idx)
        blanket_nodes = list(set(parents + children + spouses))
        
        strengths = {}
        for node in blanket_nodes:
            strengths[node] = self.compute_mutual_information(target_idx, node)
            
        return strengths
    
    def get_var_name(self, idx: int) -> str:
        """
        Get the name of a variable by index.
        
        Args:
            idx: Index of the variable
            
        Returns:
            name: Name of the variable
        """
        return self.var_names[idx]


class DynamicMarkovBlanket(MarkovBlanket):
    """
    Class for detecting Dynamic Markov Blankets in temporal data.
    
    Extends the MarkovBlanket class to handle time-varying data and
    detect temporal dependencies.
    """
    
    def __init__(self, data: Union[pd.DataFrame, torch.Tensor], 
                time_column: Optional[Union[str, int]] = None,
                lag: int = 1, alpha: float = 0.05):
        """
        Initialize the DynamicMarkovBlanket detector.
        
        Args:
            data: Input data as DataFrame or Tensor
            time_column: Column name or index for time information
            lag: Number of time lags to consider
            alpha: Significance level for independence tests
        """
        self.lag = lag
        self.time_column = time_column
        
        # Extract time information if provided
        if time_column is not None:
            if isinstance(data, pd.DataFrame):
                if isinstance(time_column, str):
                    self.time_values = data[time_column].values
                    data_no_time = data.drop(columns=[time_column])
                else:
                    self.time_values = data.iloc[:, time_column].values
                    data_no_time = data.drop(data.columns[time_column], axis=1)
            else:
                if isinstance(time_column, int):
                    self.time_values = data[:, time_column].numpy()
                    data_no_time = torch.cat([data[:, :time_column], data[:, time_column+1:]], dim=1)
                else:
                    raise ValueError("For tensor data, time_column must be an integer index")
        else:
            # Assume sequential time if no time column
            if isinstance(data, pd.DataFrame):
                self.time_values = np.arange(len(data))
                data_no_time = data
            else:
                self.time_values = np.arange(data.shape[0])
                data_no_time = data
        
        # Initialize parent class with data excluding time column
        super().__init__(data_no_time, alpha)
        
        # Create lagged data representation
        self._create_lagged_representation()
    
    def _create_lagged_representation(self):
        """Create a lagged representation of the data for temporal analysis."""
        # Get unique time points and sort them
        unique_times = np.unique(self.time_values)
        unique_times.sort()
        
        # Create mapping from time to index
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        
        # Initialize lagged data
        n_times = len(unique_times)
        n_samples_per_time = sum(self.time_values == unique_times[0])
        
        # For simplicity, assume equal number of samples per time point
        lagged_data = np.zeros((n_samples_per_time * (n_times - self.lag), 
                               self.n_vars * (self.lag + 1)))
        
        # Fill lagged data
        row_idx = 0
        for t_idx in range(self.lag, n_times):
            current_time = unique_times[t_idx]
            current_mask = self.time_values == current_time
            current_data = self.data.iloc[current_mask].values
            
            for lag_idx in range(self.lag + 1):
                lag_time = unique_times[t_idx - lag_idx]
                lag_mask = self.time_values == lag_time
                lag_data = self.data.iloc[lag_mask].values
                
                # Add lagged data to representation
                col_start = lag_idx * self.n_vars
                col_end = (lag_idx + 1) * self.n_vars
                lagged_data[row_idx:row_idx + len(current_data), col_start:col_end] = lag_data
            
            row_idx += len(current_data)
        
        # Create DataFrame for lagged data
        lagged_cols = []
        for lag_idx in range(self.lag + 1):
            lag_prefix = f"lag_{lag_idx}_" if lag_idx > 0 else ""
            lagged_cols.extend([f"{lag_prefix}{col}" for col in self.var_names])
        
        self.lagged_data = pd.DataFrame(lagged_data, columns=lagged_cols)
        
        # Update correlation matrix for lagged data
        self.lagged_corr_matrix = np.corrcoef(self.lagged_data.values, rowvar=False)
    
    def detect_dynamic_blanket(self, target_idx: int, current_only: bool = False) -> Dict[str, Dict[str, List[int]]]:
        """
        Detect the Dynamic Markov Blanket of a target variable.
        
        Args:
            target_idx: Index of the target variable
            current_only: If True, only detect blanket in the current time point
            
        Returns:
            dynamic_blanket: Dictionary with temporal components of the blanket
        """
        # Get current time blanket
        c_parents, c_children, c_spouses = super().detect_blanket(target_idx)
        
        # Initialize with current time components
        dynamic_blanket = {
            'current': {
                'parents': c_parents,
                'children': c_children,
                'spouses': c_spouses
            }
        }
        
        # If current_only is True, skip lagged components
        if current_only or self.lag <= 0:
            dynamic_blanket['lag_1'] = {
                'parents': [],
                'children': [],
                'spouses': []
            }
        else:
            # Get lagged target index
            lagged_target_idx = target_idx + self.n_vars
            
            # For lag 1, detect blanket of the lagged target
            l_parents, l_children, l_spouses = [], [], []
            
            # For each variable, check if it has a temporal effect on target
            for i in range(self.n_vars):
                lagged_var_idx = i + self.n_vars  # Index in the lagged representation
                
                # Test if current target is independent of lagged variable
                p_val, is_independent = self._conditional_independence_test(target_idx, lagged_var_idx)
                
                if not is_independent:
                    # If not independent, it's a temporal parent
                    l_parents.append(i)
            
            # For temporal children, check if lagged target affects current variables
            for i in range(self.n_vars):
                if i == target_idx:
                    continue
                    
                # Test if current variable is independent of lagged target
                p_val, is_independent = self._conditional_independence_test(i, lagged_target_idx)
                
                if not is_independent:
                    # If not independent, it's a temporal child
                    l_children.append(i)
            
            # For temporal spouses, find other variables that affect temporal children
            for child in l_children:
                for i in range(self.n_vars):
                    if i == target_idx or i in l_parents:
                        continue
                        
                    # Test if lagged variable affects the child
                    lagged_var_idx = i + self.n_vars
                    p_val, is_independent = self._conditional_independence_test(child, lagged_var_idx)
                    
                    if not is_independent and i not in l_spouses:
                        l_spouses.append(i)
            
            # Add lagged components to dynamic blanket
            dynamic_blanket['lag_1'] = {
                'parents': l_parents,
                'children': l_children,
                'spouses': l_spouses
            }
        
        # Create dynamic classifications
        dynamic_classifications = self._create_dynamic_classifications(target_idx, dynamic_blanket)
        
        return {
            'dynamic_components': dynamic_blanket,
            'dynamic_classifications': dynamic_classifications
        }
    
    def _create_dynamic_classifications(self, target_idx: int, dynamic_blanket: Dict[str, Dict[str, List[int]]]) -> Dict[str, Dict[str, List[int]]]:
        """
        Create classifications for dynamic Markov blanket.
        
        Args:
            target_idx: Index of the target variable
            dynamic_blanket: Dictionary with dynamic blanket components
            
        Returns:
            dynamic_classifications: Dictionary with temporal node classifications
        """
        blanket_components = dynamic_blanket
        
        # Current time classification
        current_blanket = list(set(
            blanket_components['current']['parents'] + 
            blanket_components['current']['children'] + 
            blanket_components['current']['spouses']
        ))
        
        current_external = [i for i in range(self.n_vars) 
                           if i != target_idx and i not in current_blanket]
        
        # Lagged classification
        if self.lag > 0:
            lagged_blanket = list(set(
                blanket_components['lag_1']['parents'] + 
                blanket_components['lag_1']['children'] + 
                blanket_components['lag_1']['spouses']
            ))
            
            lagged_external = [i for i in range(self.n_vars) 
                              if i != target_idx and i not in lagged_blanket]
        else:
            lagged_blanket = []
            lagged_external = list(range(self.n_vars))
            if target_idx in lagged_external:
                lagged_external.remove(target_idx)
        
        return {
            'current': {
                'internal': [target_idx],
                'blanket': current_blanket,
                'external': current_external
            },
            'lag_1': {
                'internal': [target_idx],
                'blanket': lagged_blanket,
                'external': lagged_external
            }
        }
    
    def classify_dynamic_nodes(self, target_idx: int) -> Dict[str, Dict[str, List[int]]]:
        """
        Classify nodes in the dynamic Markov blanket.
        
        Args:
            target_idx: Index of the target variable
            
        Returns:
            dynamic_classifications: Dictionary with temporal node classifications
        """
        dynamic_result = self.detect_dynamic_blanket(target_idx)
        return dynamic_result['dynamic_classifications']
