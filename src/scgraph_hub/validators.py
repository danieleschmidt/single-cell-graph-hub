"""Comprehensive validation utilities for Single-Cell Graph Hub."""

import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import torch
from torch_geometric.data import Data

from .exceptions import ValidationError, ConfigurationError
from .logging_config import get_logger

logger = get_logger(__name__)


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool = True, errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
    
    def __bool__(self):
        """Return True if validation passed."""
        return self.is_valid
    
    def __str__(self):
        """String representation of validation result."""
        if self.is_valid:
            return "Validation passed"
        else:
            return f"Validation failed with {len(self.errors)} errors"


class DatasetValidator:
    """Validator for dataset configurations and data."""
    
    REQUIRED_FIELDS = [
        "name", "n_cells", "n_genes", "modality", "organism"
    ]
    
    VALID_MODALITIES = [
        "scRNA-seq", "snRNA-seq", "scATAC-seq", "scChIP-seq", 
        "spatial_transcriptomics", "CyTOF", "CITE-seq", 
        "multiome", "scNMT-seq", "SHARE-seq"
    ]
    
    VALID_ORGANISMS = [
        "human", "mouse", "rat", "zebrafish", "drosophila", 
        "c_elegans", "arabidopsis", "yeast"
    ]
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> ValidationResult:
        """Validate dataset configuration."""
        result = ValidationResult()
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in config:
                result.add_error(f"Missing required field: {field}")
        
        if not result.is_valid:
            return result
        
        # Validate field values
        cls._validate_name(config.get("name"), result)
        cls._validate_numeric_fields(config, result)
        cls._validate_modality(config.get("modality"), result)
        cls._validate_organism(config.get("organism"), result)
        cls._validate_optional_fields(config, result)
        
        return result
    
    @classmethod
    def _validate_name(cls, name: str, result: ValidationResult):
        """Validate dataset name."""
        if not isinstance(name, str):
            result.add_error("Dataset name must be a string")
            return
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            result.add_error("Dataset name can only contain letters, numbers, underscores, and hyphens")
        
        if len(name) < 3:
            result.add_error("Dataset name must be at least 3 characters long")
        
        if len(name) > 50:
            result.add_error("Dataset name must be at most 50 characters long")
    
    @classmethod
    def _validate_numeric_fields(cls, config: Dict[str, Any], result: ValidationResult):
        """Validate numeric fields."""
        numeric_fields = ["n_cells", "n_genes"]
        
        for field in numeric_fields:
            value = config.get(field)
            if not isinstance(value, (int, float)):
                result.add_error(f"{field} must be a number")
                continue
            
            if value <= 0:
                result.add_error(f"{field} must be positive")
            
            if field == "n_cells" and value > 10_000_000:
                result.add_warning(f"{field} is very large ({value}), ensure this is correct")
            
            if field == "n_genes" and value > 100_000:
                result.add_warning(f"{field} is very large ({value}), ensure this is correct")
    
    @classmethod
    def _validate_modality(cls, modality: str, result: ValidationResult):
        """Validate modality field."""
        if not isinstance(modality, str):
            result.add_error("Modality must be a string")
            return
        
        if modality not in cls.VALID_MODALITIES:
            result.add_error(f"Invalid modality: {modality}. Valid options: {', '.join(cls.VALID_MODALITIES)}")
    
    @classmethod
    def _validate_organism(cls, organism: str, result: ValidationResult):
        """Validate organism field."""
        if not isinstance(organism, str):
            result.add_error("Organism must be a string")
            return
        
        if organism not in cls.VALID_ORGANISMS:
            result.add_warning(f"Uncommon organism: {organism}. Common options: {', '.join(cls.VALID_ORGANISMS[:5])}")
    
    @classmethod
    def _validate_optional_fields(cls, config: Dict[str, Any], result: ValidationResult):
        """Validate optional fields."""
        # Validate tissue
        if "tissue" in config:
            tissue = config["tissue"]
            if not isinstance(tissue, str):
                result.add_error("Tissue must be a string")
        
        # Validate tasks
        if "tasks" in config:
            tasks = config["tasks"]
            if not isinstance(tasks, list):
                result.add_error("Tasks must be a list")
            else:
                valid_tasks = [
                    "cell_type_prediction", "trajectory_inference", 
                    "gene_imputation", "batch_correction", 
                    "cell_cell_interaction", "spatial_domain_identification"
                ]
                for task in tasks:
                    if task not in valid_tasks:
                        result.add_warning(f"Uncommon task: {task}")
        
        # Validate size_mb
        if "size_mb" in config:
            size_mb = config["size_mb"]
            if not isinstance(size_mb, (int, float)):
                result.add_error("Size in MB must be a number")
            elif size_mb <= 0:
                result.add_error("Size in MB must be positive")
    
    @classmethod
    def validate_data(cls, data: Data) -> ValidationResult:
        """Validate PyTorch Geometric Data object."""
        result = ValidationResult()
        
        # Check if data object exists
        if data is None:
            result.add_error("Data object is None")
            return result
        
        # Validate node features
        if not hasattr(data, 'x') or data.x is None:
            result.add_error("Data object missing node features (x)")
        else:
            cls._validate_tensor(data.x, "node features", result)
        
        # Validate edge index
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            result.add_error("Data object missing edge index")
        else:
            cls._validate_edge_index(data.edge_index, data.x.shape[0] if hasattr(data, 'x') and data.x is not None else None, result)
        
        # Validate labels if present
        if hasattr(data, 'y') and data.y is not None:
            cls._validate_labels(data.y, data.x.shape[0] if hasattr(data, 'x') and data.x is not None else None, result)
        
        # Validate masks if present
        for mask_name in ['train_mask', 'val_mask', 'test_mask']:
            if hasattr(data, mask_name):
                mask = getattr(data, mask_name)
                if mask is not None:
                    cls._validate_mask(mask, mask_name, data.x.shape[0] if hasattr(data, 'x') and data.x is not None else None, result)
        
        return result
    
    @classmethod
    def _validate_tensor(cls, tensor: torch.Tensor, name: str, result: ValidationResult):
        """Validate a tensor."""
        if not isinstance(tensor, torch.Tensor):
            result.add_error(f"{name} must be a torch.Tensor")
            return
        
        if tensor.numel() == 0:
            result.add_error(f"{name} tensor is empty")
        
        if torch.isnan(tensor).any():
            result.add_error(f"{name} tensor contains NaN values")
        
        if torch.isinf(tensor).any():
            result.add_error(f"{name} tensor contains infinite values")
    
    @classmethod
    def _validate_edge_index(cls, edge_index: torch.Tensor, num_nodes: Optional[int], result: ValidationResult):
        """Validate edge index tensor."""
        if not isinstance(edge_index, torch.Tensor):
            result.add_error("Edge index must be a torch.Tensor")
            return
        
        if edge_index.dim() != 2:
            result.add_error(f"Edge index must be 2-dimensional, got {edge_index.dim()}")
            return
        
        if edge_index.shape[0] != 2:
            result.add_error(f"Edge index must have shape [2, num_edges], got {edge_index.shape}")
            return
        
        if edge_index.dtype != torch.long:
            result.add_warning("Edge index should have dtype torch.long")
        
        if num_nodes is not None:
            max_node_idx = edge_index.max().item()
            if max_node_idx >= num_nodes:
                result.add_error(f"Edge index contains invalid node indices (max: {max_node_idx}, num_nodes: {num_nodes})")
        
        if edge_index.min().item() < 0:
            result.add_error("Edge index contains negative node indices")
    
    @classmethod
    def _validate_labels(cls, labels: torch.Tensor, num_nodes: Optional[int], result: ValidationResult):
        """Validate label tensor."""
        if not isinstance(labels, torch.Tensor):
            result.add_error("Labels must be a torch.Tensor")
            return
        
        if labels.dim() != 1:
            result.add_error(f"Labels must be 1-dimensional, got {labels.dim()}")
            return
        
        if num_nodes is not None and labels.shape[0] != num_nodes:
            result.add_error(f"Number of labels ({labels.shape[0]}) must match number of nodes ({num_nodes})")
        
        if labels.min().item() < 0:
            result.add_error("Labels contain negative values")
    
    @classmethod
    def _validate_mask(cls, mask: torch.Tensor, mask_name: str, num_nodes: Optional[int], result: ValidationResult):
        """Validate mask tensor."""
        if not isinstance(mask, torch.Tensor):
            result.add_error(f"{mask_name} must be a torch.Tensor")
            return
        
        if mask.dim() != 1:
            result.add_error(f"{mask_name} must be 1-dimensional, got {mask.dim()}")
            return
        
        if mask.dtype != torch.bool:
            result.add_warning(f"{mask_name} should have dtype torch.bool")
        
        if num_nodes is not None and mask.shape[0] != num_nodes:
            result.add_error(f"{mask_name} length ({mask.shape[0]}) must match number of nodes ({num_nodes})")


class ModelValidator:
    """Validator for model configurations and parameters."""
    
    @classmethod
    def validate_config(cls, model_name: str, config: Dict[str, Any]) -> ValidationResult:
        """Validate model configuration."""
        result = ValidationResult()
        
        # Check required parameters
        required_params = ["input_dim", "output_dim"]
        for param in required_params:
            if param not in config:
                result.add_error(f"Missing required parameter: {param}")
        
        if not result.is_valid:
            return result
        
        # Validate parameter values
        cls._validate_dimensions(config, result)
        cls._validate_architecture_params(config, result)
        cls._validate_training_params(config, result)
        
        return result
    
    @classmethod
    def _validate_dimensions(cls, config: Dict[str, Any], result: ValidationResult):
        """Validate dimension parameters."""
        for dim_name in ["input_dim", "output_dim", "hidden_dim"]:
            if dim_name in config:
                dim_value = config[dim_name]
                if not isinstance(dim_value, int):
                    result.add_error(f"{dim_name} must be an integer")
                elif dim_value <= 0:
                    result.add_error(f"{dim_name} must be positive")
                elif dim_value > 100_000:
                    result.add_warning(f"{dim_name} is very large ({dim_value})")
    
    @classmethod
    def _validate_architecture_params(cls, config: Dict[str, Any], result: ValidationResult):
        """Validate architecture parameters."""
        # Validate num_layers
        if "num_layers" in config:
            num_layers = config["num_layers"]
            if not isinstance(num_layers, int):
                result.add_error("num_layers must be an integer")
            elif num_layers <= 0:
                result.add_error("num_layers must be positive")
            elif num_layers > 20:
                result.add_warning(f"num_layers is very large ({num_layers}), may cause training issues")
        
        # Validate dropout
        if "dropout" in config:
            dropout = config["dropout"]
            if not isinstance(dropout, (int, float)):
                result.add_error("dropout must be a number")
            elif dropout < 0 or dropout >= 1:
                result.add_error("dropout must be in range [0, 1)")
    
    @classmethod
    def _validate_training_params(cls, config: Dict[str, Any], result: ValidationResult):
        """Validate training parameters."""
        # Validate learning rate
        if "lr" in config:
            lr = config["lr"]
            if not isinstance(lr, (int, float)):
                result.add_error("Learning rate must be a number")
            elif lr <= 0:
                result.add_error("Learning rate must be positive")
            elif lr > 1:
                result.add_warning(f"Learning rate is very large ({lr})")
        
        # Validate batch size
        if "batch_size" in config:
            batch_size = config["batch_size"]
            if not isinstance(batch_size, int):
                result.add_error("batch_size must be an integer")
            elif batch_size <= 0:
                result.add_error("batch_size must be positive")
            elif batch_size > 10000:
                result.add_warning(f"batch_size is very large ({batch_size})")


class FileValidator:
    """Validator for file paths and file system operations."""
    
    @classmethod
    def validate_path(cls, path: Union[str, Path], must_exist: bool = False, must_be_file: bool = False, must_be_dir: bool = False) -> ValidationResult:
        """Validate file or directory path."""
        result = ValidationResult()
        
        if not isinstance(path, (str, Path)):
            result.add_error("Path must be a string or Path object")
            return result
        
        path = Path(path)
        
        # Check if path exists
        if must_exist and not path.exists():
            result.add_error(f"Path does not exist: {path}")
            return result
        
        if path.exists():
            # Check if it's a file when required
            if must_be_file and not path.is_file():
                result.add_error(f"Path is not a file: {path}")
            
            # Check if it's a directory when required
            if must_be_dir and not path.is_dir():
                result.add_error(f"Path is not a directory: {path}")
        
        # Check if parent directory exists and is writable
        if not path.exists():
            parent = path.parent
            if not parent.exists():
                result.add_error(f"Parent directory does not exist: {parent}")
            elif not os.access(parent, os.W_OK):
                result.add_error(f"Parent directory is not writable: {parent}")
        
        return result
    
    @classmethod
    def validate_storage_space(cls, path: Union[str, Path], required_space_mb: float) -> ValidationResult:
        """Validate available storage space."""
        result = ValidationResult()
        
        path = Path(path)
        
        try:
            # Get available space
            statvfs = os.statvfs(path.parent if path.is_file() else path)
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
            available_mb = available_bytes / (1024 * 1024)
            
            if available_mb < required_space_mb:
                result.add_error(f"Insufficient storage space. Required: {required_space_mb:.1f} MB, Available: {available_mb:.1f} MB")
            elif available_mb < required_space_mb * 1.2:  # Less than 20% buffer
                result.add_warning(f"Low storage space. Required: {required_space_mb:.1f} MB, Available: {available_mb:.1f} MB")
                
        except Exception as e:
            result.add_warning(f"Could not check storage space: {e}")
        
        return result


def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """Legacy function for backward compatibility."""
    result = DatasetValidator.validate_config(config)
    if not result.is_valid:
        logger.error(f"Dataset validation failed: {'; '.join(result.errors)}")
        return False
    
    if result.warnings:
        logger.warning(f"Dataset validation warnings: {'; '.join(result.warnings)}")
    
    return True


def validate_model_config(model_name: str, config: Dict[str, Any]) -> ValidationResult:
    """Validate model configuration with detailed results."""
    return ModelValidator.validate_config(model_name, config)


def validate_file_path(path: Union[str, Path], **kwargs) -> ValidationResult:
    """Validate file path with detailed results."""
    return FileValidator.validate_path(path, **kwargs)