"""Comprehensive validation and error handling for Single-Cell Graph Hub."""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
import json
import hashlib
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """Comprehensive data validation for datasets and models."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: Whether to raise exceptions on validation failures
        """
        self.strict_mode = strict_mode
        self.validation_history = []
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules and thresholds."""
        return {
            'min_nodes': 10,
            'max_nodes': 1000000,
            'min_edges': 0,
            'max_edges': 10000000,
            'min_features': 1,
            'max_features': 100000,
            'max_memory_usage_gb': 16,
            'required_attributes': ['x', 'edge_index'],
            'optional_attributes': ['y', 'train_mask', 'val_mask', 'test_mask'],
            'data_types': {
                'x': torch.float32,
                'edge_index': torch.long,
                'y': torch.long
            }
        }
    
    def validate_dataset(self, data: Data, dataset_name: str = "unknown") -> Dict[str, Any]:
        """Comprehensive dataset validation.
        
        Args:
            data: PyTorch Geometric Data object
            dataset_name: Name of the dataset for logging
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'recommendations': []
        }
        
        try:
            # Basic structure validation
            self._validate_data_structure(data, validation_results)
            
            # Feature validation
            self._validate_features(data, validation_results)
            
            # Graph structure validation
            self._validate_graph_structure(data, validation_results)
            
            # Label validation
            self._validate_labels(data, validation_results)
            
            # Split validation
            self._validate_splits(data, validation_results)
            
            # Statistical validation
            self._validate_statistics(data, validation_results)
            
            # Memory and performance checks
            self._validate_memory_usage(data, validation_results)
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation failed with exception: {str(e)}")
            logger.error(f"Dataset validation failed for {dataset_name}: {e}")
            
            if self.strict_mode:
                raise ValidationError(f"Dataset validation failed: {e}")
        
        # Store validation history
        self.validation_history.append(validation_results)
        
        return validation_results
    
    def _validate_data_structure(self, data: Data, results: Dict[str, Any]):
        """Validate basic data structure."""
        # Check required attributes
        required_attrs = ['x', 'edge_index']
        for attr in required_attrs:
            if not hasattr(data, attr) or getattr(data, attr) is None:
                results['errors'].append(f"Missing required attribute: {attr}")
                results['valid'] = False
        
        # Check tensor types
        if hasattr(data, 'x') and data.x is not None:
            if not isinstance(data.x, torch.Tensor):
                results['errors'].append(f"Feature matrix x must be a torch.Tensor, got {type(data.x)}")
                results['valid'] = False
            elif data.x.dtype not in [torch.float32, torch.float64]:
                results['warnings'].append(f"Feature matrix has dtype {data.x.dtype}, recommend float32")
        
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            if not isinstance(data.edge_index, torch.Tensor):
                results['errors'].append(f"Edge index must be a torch.Tensor, got {type(data.edge_index)}")
                results['valid'] = False
            elif data.edge_index.dtype != torch.long:
                results['errors'].append(f"Edge index must have dtype torch.long, got {data.edge_index.dtype}")
                results['valid'] = False
    
    def _validate_features(self, data: Data, results: Dict[str, Any]):
        """Validate feature matrix."""
        if not hasattr(data, 'x') or data.x is None:
            return
        
        x = data.x
        
        # Check dimensions
        if x.dim() != 2:
            results['errors'].append(f"Feature matrix must be 2D, got {x.dim()}D")
            results['valid'] = False
            return
        
        n_nodes, n_features = x.shape
        
        # Check for NaN or inf values
        if torch.isnan(x).any():
            results['errors'].append("Feature matrix contains NaN values")
            results['valid'] = False
        
        if torch.isinf(x).any():
            results['errors'].append("Feature matrix contains infinite values")
            results['valid'] = False
        
        # Statistical checks
        if n_features == 0:
            results['errors'].append("Feature matrix has 0 features")
            results['valid'] = False
        
        if n_nodes == 0:
            results['errors'].append("Feature matrix has 0 nodes")
            results['valid'] = False
        
        # Check for constant features
        feature_std = torch.std(x, dim=0)
        constant_features = torch.sum(feature_std == 0).item()
        if constant_features > 0:
            results['warnings'].append(f"{constant_features} features have zero variance")
        
        # Check feature scale
        feature_mean = torch.mean(x, dim=0)
        feature_max = torch.max(torch.abs(x), dim=0)[0]
        
        if torch.max(feature_max).item() > 1000:
            results['warnings'].append("Features have very large values, consider scaling")
        
        if torch.min(feature_mean).item() < -100 or torch.max(feature_mean).item() > 100:
            results['warnings'].append("Features have extreme mean values, consider centering")
        
        # Store statistics
        results['statistics']['n_nodes'] = n_nodes
        results['statistics']['n_features'] = n_features
        results['statistics']['feature_mean'] = float(torch.mean(x))
        results['statistics']['feature_std'] = float(torch.std(x))
        results['statistics']['feature_min'] = float(torch.min(x))
        results['statistics']['feature_max'] = float(torch.max(x))
    
    def _validate_graph_structure(self, data: Data, results: Dict[str, Any]):
        """Validate graph structure."""
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            return
        
        edge_index = data.edge_index
        
        # Check edge index shape
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            results['errors'].append(f"Edge index must have shape [2, num_edges], got {edge_index.shape}")
            results['valid'] = False
            return
        
        if edge_index.shape[1] == 0:
            results['warnings'].append("Graph has no edges")
            return
        
        # Check node indices are valid
        max_node_idx = torch.max(edge_index).item()
        min_node_idx = torch.min(edge_index).item()
        
        if min_node_idx < 0:
            results['errors'].append(f"Edge index contains negative node indices: {min_node_idx}")
            results['valid'] = False
        
        n_nodes = data.x.shape[0] if hasattr(data, 'x') and data.x is not None else max_node_idx + 1
        
        if max_node_idx >= n_nodes:
            results['errors'].append(f"Edge index references node {max_node_idx} but only {n_nodes} nodes exist")
            results['valid'] = False
        
        # Check for self-loops
        self_loops = torch.sum(edge_index[0] == edge_index[1]).item()
        if self_loops > 0:
            results['warnings'].append(f"Graph contains {self_loops} self-loops")
        
        # Check graph connectivity
        from torch_geometric.utils import to_networkx
        import networkx as nx
        
        try:
            G = to_networkx(data, to_undirected=True)
            n_components = nx.number_connected_components(G)
            if n_components > 1:
                results['warnings'].append(f"Graph has {n_components} connected components")
            
            # Graph statistics
            n_edges = edge_index.shape[1]
            avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
            density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
            
            results['statistics']['n_edges'] = n_edges
            results['statistics']['avg_degree'] = avg_degree
            results['statistics']['graph_density'] = density
            results['statistics']['n_components'] = n_components
            
        except Exception as e:
            results['warnings'].append(f"Failed to compute graph statistics: {e}")
    
    def _validate_labels(self, data: Data, results: Dict[str, Any]):
        """Validate labels if present."""
        if not hasattr(data, 'y') or data.y is None:
            results['warnings'].append("No labels found in dataset")
            return
        
        y = data.y
        
        # Check label tensor
        if not isinstance(y, torch.Tensor):
            results['errors'].append(f"Labels must be a torch.Tensor, got {type(y)}")
            results['valid'] = False
            return
        
        # Check label dimensions
        if y.dim() > 2:
            results['errors'].append(f"Labels must be 1D or 2D, got {y.dim()}D")
            results['valid'] = False
            return
        
        n_nodes = data.x.shape[0] if hasattr(data, 'x') and data.x is not None else 0
        
        if y.shape[0] != n_nodes:
            results['errors'].append(f"Number of labels ({y.shape[0]}) doesn't match number of nodes ({n_nodes})")
            results['valid'] = False
        
        # Check for missing labels
        if torch.isnan(y).any():
            results['warnings'].append("Labels contain NaN values")
        
        # Classification labels
        if y.dtype in [torch.long, torch.int32, torch.int64]:
            unique_labels = torch.unique(y[~torch.isnan(y.float())])
            n_classes = len(unique_labels)
            
            if torch.min(unique_labels).item() < 0:
                results['errors'].append(f"Classification labels contain negative values: {torch.min(unique_labels).item()}")
                results['valid'] = False
            
            # Check class balance
            class_counts = torch.bincount(y[~torch.isnan(y.float())].long())
            min_class_count = torch.min(class_counts).item()
            max_class_count = torch.max(class_counts).item()
            
            if min_class_count < 2:
                results['warnings'].append(f"Some classes have very few samples (min: {min_class_count})")
            
            if max_class_count / min_class_count > 10:
                results['warnings'].append(f"Severe class imbalance detected (ratio: {max_class_count/min_class_count:.1f})")
            
            results['statistics']['n_classes'] = n_classes
            results['statistics']['class_balance_ratio'] = max_class_count / min_class_count if min_class_count > 0 else float('inf')
    
    def _validate_splits(self, data: Data, results: Dict[str, Any]):
        """Validate train/val/test splits if present."""
        split_masks = ['train_mask', 'val_mask', 'test_mask']
        present_masks = [mask for mask in split_masks if hasattr(data, mask) and getattr(data, mask) is not None]
        
        if not present_masks:
            results['warnings'].append("No train/val/test splits found")
            return
        
        n_nodes = data.x.shape[0] if hasattr(data, 'x') and data.x is not None else 0
        
        for mask_name in present_masks:
            mask = getattr(data, mask_name)
            
            if not isinstance(mask, torch.Tensor):
                results['errors'].append(f"{mask_name} must be a torch.Tensor")
                results['valid'] = False
                continue
            
            if mask.dtype != torch.bool:
                results['warnings'].append(f"{mask_name} should have dtype torch.bool, got {mask.dtype}")
            
            if mask.shape[0] != n_nodes:
                results['errors'].append(f"{mask_name} size ({mask.shape[0]}) doesn't match number of nodes ({n_nodes})")
                results['valid'] = False
            
            # Check split size
            split_size = torch.sum(mask).item()
            split_ratio = split_size / n_nodes if n_nodes > 0 else 0
            
            results['statistics'][f'{mask_name}_size'] = split_size
            results['statistics'][f'{mask_name}_ratio'] = split_ratio
            
            if split_ratio < 0.01:
                results['warnings'].append(f"{mask_name} is very small ({split_ratio:.2%})")
            elif split_ratio > 0.8:
                results['warnings'].append(f"{mask_name} is very large ({split_ratio:.2%})")
        
        # Check for overlaps
        if len(present_masks) > 1:
            for i, mask1 in enumerate(present_masks):
                for mask2 in present_masks[i+1:]:
                    mask1_tensor = getattr(data, mask1)
                    mask2_tensor = getattr(data, mask2)
                    
                    overlap = torch.sum(mask1_tensor & mask2_tensor).item()
                    if overlap > 0:
                        results['warnings'].append(f"{mask1} and {mask2} have {overlap} overlapping nodes")
    
    def _validate_statistics(self, data: Data, results: Dict[str, Any]):
        """Validate statistical properties."""
        if not hasattr(data, 'x') or data.x is None:
            return
        
        x = data.x
        
        # Check for data leakage indicators
        # Perfect separability
        if hasattr(data, 'y') and data.y is not None and data.y.dtype in [torch.long, torch.int32, torch.int64]:
            unique_labels = torch.unique(data.y)
            if len(unique_labels) > 1:
                # Simple check: compute class means and see if they're suspiciously different
                class_means = []
                for label in unique_labels:
                    mask = data.y == label
                    if torch.sum(mask) > 0:
                        class_mean = torch.mean(x[mask], dim=0)
                        class_means.append(class_mean)
                
                if len(class_means) > 1:
                    # Compute pairwise distances between class means
                    min_dist = float('inf')
                    for i in range(len(class_means)):
                        for j in range(i+1, len(class_means)):
                            dist = torch.norm(class_means[i] - class_means[j]).item()
                            min_dist = min(min_dist, dist)
                    
                    # If classes are perfectly separated, this might indicate leakage
                    feature_scale = torch.std(x).item()
                    if min_dist > 10 * feature_scale:
                        results['warnings'].append("Classes appear perfectly separated - check for data leakage")
        
        # Check for duplicate samples
        if x.shape[0] > 1:
            # Sample a subset for efficiency
            n_check = min(1000, x.shape[0])
            indices = torch.randperm(x.shape[0])[:n_check]
            x_sample = x[indices]
            
            # Compute pairwise distances
            distances = torch.cdist(x_sample, x_sample)
            # Set diagonal to large value
            distances.fill_diagonal_(float('inf'))
            
            min_dist = torch.min(distances).item()
            if min_dist < 1e-6:
                results['warnings'].append("Very similar or duplicate samples detected")
    
    def _validate_memory_usage(self, data: Data, results: Dict[str, Any]):
        """Validate memory usage and efficiency."""
        import sys
        
        total_memory = 0
        
        if hasattr(data, 'x') and data.x is not None:
            x_memory = data.x.element_size() * data.x.nelement()
            total_memory += x_memory
            results['statistics']['feature_memory_mb'] = x_memory / (1024 * 1024)
        
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_memory = data.edge_index.element_size() * data.edge_index.nelement()
            total_memory += edge_memory
            results['statistics']['edge_memory_mb'] = edge_memory / (1024 * 1024)
        
        results['statistics']['total_memory_mb'] = total_memory / (1024 * 1024)
        
        # Memory warnings
        if total_memory > 1024 * 1024 * 1024:  # 1GB
            results['warnings'].append(f"Dataset uses {total_memory / (1024**3):.1f}GB of memory")
        
        # Check sparsity for potential optimization
        if hasattr(data, 'x') and data.x is not None:
            zero_ratio = torch.sum(data.x == 0).item() / data.x.nelement()
            if zero_ratio > 0.5:
                results['recommendations'].append(f"Feature matrix is {zero_ratio:.1%} sparse - consider sparse representation")
    
    def validate_model_compatibility(self, model: torch.nn.Module, data: Data) -> Dict[str, Any]:
        """Validate model compatibility with dataset.
        
        Args:
            model: PyTorch model
            data: Dataset to validate against
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }
        
        try:
            # Check input dimensions
            if hasattr(data, 'x') and data.x is not None:
                input_dim = data.x.shape[1]
                
                # Try to infer model input dimension
                model_input_dim = None
                for name, param in model.named_parameters():
                    if 'weight' in name and param.dim() == 2:
                        model_input_dim = param.shape[1]
                        break
                
                if model_input_dim and model_input_dim != input_dim:
                    results['errors'].append(
                        f"Model expects {model_input_dim} input features, but dataset has {input_dim}"
                    )
                    results['valid'] = False
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                try:
                    if hasattr(model, 'forward'):
                        # Try minimal forward pass
                        output = model(data.x, data.edge_index)
                        results['model_info']['output_shape'] = list(output.shape)
                        results['model_info']['forward_pass'] = 'successful'
                    
                except Exception as e:
                    results['errors'].append(f"Forward pass failed: {str(e)}")
                    results['valid'] = False
                    results['model_info']['forward_pass'] = 'failed'
            
            # Check device compatibility
            model_device = next(model.parameters()).device
            data_device = data.x.device if hasattr(data, 'x') else torch.device('cpu')
            
            if model_device != data_device:
                results['warnings'].append(
                    f"Model on {model_device}, data on {data_device} - may need to move tensors"
                )
            
            results['model_info']['model_device'] = str(model_device)
            results['model_info']['data_device'] = str(data_device)
            results['model_info']['num_parameters'] = sum(p.numel() for p in model.parameters())
            
        except Exception as e:
            results['errors'].append(f"Model validation failed: {str(e)}")
            results['valid'] = False
        
        return results
    
    def generate_validation_report(self, validation_results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Generate comprehensive validation report.
        
        Args:
            validation_results: Results from validation
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SINGLE-CELL GRAPH HUB VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Dataset: {validation_results.get('dataset_name', 'Unknown')}")
        report_lines.append(f"Timestamp: {validation_results.get('timestamp', 'Unknown')}")
        report_lines.append(f"Status: {'PASS' if validation_results.get('valid', False) else 'FAIL'}")
        report_lines.append("")
        
        # Statistics
        if 'statistics' in validation_results and validation_results['statistics']:
            report_lines.append("Dataset Statistics:")
            report_lines.append("-" * 20)
            for key, value in validation_results['statistics'].items():
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Errors
        if validation_results.get('errors'):
            report_lines.append("ERRORS:")
            report_lines.append("-" * 10)
            for error in validation_results['errors']:
                report_lines.append(f"  ❌ {error}")
            report_lines.append("")
        
        # Warnings
        if validation_results.get('warnings'):
            report_lines.append("WARNINGS:")
            report_lines.append("-" * 10)
            for warning in validation_results['warnings']:
                report_lines.append(f"  ⚠️  {warning}")
            report_lines.append("")
        
        # Recommendations
        if validation_results.get('recommendations'):
            report_lines.append("RECOMMENDATIONS:")
            report_lines.append("-" * 15)
            for rec in validation_results['recommendations']:
                report_lines.append(f"  💡 {rec}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {save_path}")
        
        return report_text


class ConfigValidator:
    """Validator for configuration files and parameters."""
    
    @staticmethod
    def validate_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate preprocessing configuration.
        
        Args:
            config: Preprocessing configuration
            
        Returns:
            Validation results
        """
        results = {'valid': True, 'errors': [], 'warnings': []}
        
        # Check required fields
        if 'steps' in config:
            valid_steps = [
                'filter_cells', 'filter_genes', 'calculate_qc_metrics',
                'normalize_total', 'log1p', 'highly_variable_genes',
                'scale', 'pca', 'neighbors'
            ]
            
            for step in config['steps']:
                if step not in valid_steps:
                    results['warnings'].append(f"Unknown preprocessing step: {step}")
        
        # Validate parameters
        if 'parameters' in config:
            params = config['parameters']
            
            # Filter cells parameters
            if 'filter_cells' in params:
                cell_params = params['filter_cells']
                if 'min_genes' in cell_params and cell_params['min_genes'] < 0:
                    results['errors'].append("min_genes must be non-negative")
                    results['valid'] = False
        
        return results
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Validation results
        """
        results = {'valid': True, 'errors': [], 'warnings': []}
        
        # Check required fields
        required_fields = ['model_name']
        for field in required_fields:
            if field not in config:
                results['errors'].append(f"Missing required field: {field}")
                results['valid'] = False
        
        # Validate model-specific parameters
        if 'model_name' in config:
            model_name = config['model_name']
            
            # Check dimensions
            for dim_param in ['input_dim', 'hidden_dim', 'output_dim']:
                if dim_param in config:
                    if not isinstance(config[dim_param], int) or config[dim_param] <= 0:
                        results['errors'].append(f"{dim_param} must be a positive integer")
                        results['valid'] = False
            
            # Check dropout
            if 'dropout' in config:
                dropout = config['dropout']
                if not isinstance(dropout, (int, float)) or dropout < 0 or dropout > 1:
                    results['errors'].append("dropout must be a float between 0 and 1")
                    results['valid'] = False
        
        return results
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training configuration.
        
        Args:
            config: Training configuration
            
        Returns:
            Validation results
        """
        results = {'valid': True, 'errors': [], 'warnings': []}
        
        # Check learning rate
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                results['errors'].append("learning_rate must be a positive number")
                results['valid'] = False
            elif lr > 1.0:
                results['warnings'].append("learning_rate is very high, may cause instability")
        
        # Check epochs
        if 'epochs' in config:
            epochs = config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                results['errors'].append("epochs must be a positive integer")
                results['valid'] = False
        
        # Check batch size
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                results['errors'].append("batch_size must be a positive integer")
                results['valid'] = False
        
        return results


# Convenience functions
def validate_dataset_config(dataset_config: Dict[str, Any]) -> bool:
    """Quick validation of dataset configuration.
    
    Args:
        dataset_config: Dataset configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['name', 'n_cells', 'n_genes', 'modality']
    
    for field in required_fields:
        if field not in dataset_config:
            logger.error(f"Missing required field in dataset config: {field}")
            return False
    
    # Type checks
    if not isinstance(dataset_config['n_cells'], int) or dataset_config['n_cells'] <= 0:
        logger.error("n_cells must be a positive integer")
        return False
    
    if not isinstance(dataset_config['n_genes'], int) or dataset_config['n_genes'] <= 0:
        logger.error("n_genes must be a positive integer")
        return False
    
    return True


def check_dependencies() -> Dict[str, bool]:
    """Check availability of optional dependencies.
    
    Returns:
        Dictionary mapping package names to availability
    """
    dependencies = {
        'torch': False,
        'torch_geometric': False,
        'scanpy': False,
        'pandas': False,
        'numpy': False,
        'sklearn': False,
        'networkx': False,
        'plotly': False,
        'umap': False,
        'fa2': False
    }
    
    for package in dependencies:
        try:
            if package == 'torch_geometric':
                import torch_geometric
            elif package == 'sklearn':
                import sklearn
            elif package == 'umap':
                import umap
            else:
                __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    return dependencies


def validate_environment() -> Dict[str, Any]:
    """Validate the environment setup.
    
    Returns:
        Environment validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'dependencies': {},
        'system_info': {}
    }
    
    # Check dependencies
    deps = check_dependencies()
    results['dependencies'] = deps
    
    # Check critical dependencies
    critical_deps = ['torch', 'torch_geometric', 'numpy', 'pandas']
    for dep in critical_deps:
        if not deps.get(dep, False):
            results['errors'].append(f"Critical dependency missing: {dep}")
            results['valid'] = False
    
    # System info
    import sys
    import platform
    
    results['system_info'] = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture()[0]
    }
    
    # Check CUDA availability
    if deps.get('torch', False):
        import torch
        results['system_info']['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results['system_info']['cuda_device_count'] = torch.cuda.device_count()
            results['system_info']['cuda_device_name'] = torch.cuda.get_device_name(0)
    
    return results
