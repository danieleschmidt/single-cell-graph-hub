"""Comprehensive utility functions and helpers for Single-Cell Graph Hub."""

import logging
import os
import sys
import time
import functools
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
import json
import pickle
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional imports for full functionality
try:
    import torch
    import numpy as np
    import pandas as pd
    from torch_geometric.data import Data
    _TORCH_AVAILABLE = True
    _NUMPY_AVAILABLE = True
    _PANDAS_AVAILABLE = True
except ImportError as e:
    _TORCH_AVAILABLE = False
    _NUMPY_AVAILABLE = False
    _PANDAS_AVAILABLE = False
    warnings.warn(f"Scientific computing libraries not available: {e}")


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('scgraph_hub')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import torch_geometric
    except ImportError:
        missing.append("torch-geometric")
    
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    
    if missing:
        warnings.warn(
            f"Missing dependencies: {', '.join(missing)}. "
            "Install with: pip install single-cell-graph-hub[full]"
        )
        return False
    
    return True


def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """Validate dataset configuration parameters."""
    required_fields = ["name", "n_cells", "n_genes", "modality", "organism"]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate numeric fields
    if config["n_cells"] <= 0:
        raise ValueError("n_cells must be positive")
    
    if config["n_genes"] <= 0:
        raise ValueError("n_genes must be positive")
    
    # Validate categorical fields
    valid_modalities = [
        "scRNA-seq", "snRNA-seq", "scATAC-seq", "scChIP-seq",
        "spatial_transcriptomics", "CITE-seq", "multimodal"
    ]
    if config["modality"] not in valid_modalities:
        warnings.warn(f"Unknown modality: {config['modality']}")
    
    valid_organisms = ["human", "mouse", "zebrafish", "drosophila", "c_elegans"]
    if config["organism"] not in valid_organisms:
        warnings.warn(f"Unknown organism: {config['organism']}")
    
    return True


def estimate_memory_usage(n_cells: int, n_genes: int, n_edges: int) -> Dict[str, float]:
    """Estimate memory usage for a dataset.
    
    Args:
        n_cells: Number of cells
        n_genes: Number of genes
        n_edges: Number of edges
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Estimate sizes (assuming float32)
    node_features_gb = (n_cells * n_genes * 4) / (1024**3)
    edge_index_gb = (2 * n_edges * 8) / (1024**3)  # int64
    labels_gb = (n_cells * 8) / (1024**3)  # int64
    
    total_gb = node_features_gb + edge_index_gb + labels_gb
    
    return {
        "node_features": node_features_gb,
        "edge_index": edge_index_gb,
        "labels": labels_gb,
        "total": total_gb,
        "peak_training": total_gb * 3  # Rough estimate with gradients
    }


def suggest_model_architecture(n_cells: int, n_genes: int) -> Dict[str, Any]:
    """Suggest appropriate model architecture based on dataset size."""
    suggestions = {}
    
    # Model type based on size
    if n_cells < 10000:
        suggestions["model_type"] = "CellGraphGNN"
        suggestions["reason"] = "Small dataset, standard GNN is sufficient"
    elif n_cells < 100000:
        suggestions["model_type"] = "CellGraphSAGE"
        suggestions["reason"] = "Medium dataset, GraphSAGE for scalability"
    else:
        suggestions["model_type"] = "CellGraphSAGE"
        suggestions["reason"] = "Large dataset, GraphSAGE required for memory efficiency"
    
    # Hidden dimension based on gene count
    if n_genes < 1000:
        suggestions["hidden_dim"] = 64
    elif n_genes < 5000:
        suggestions["hidden_dim"] = 128
    else:
        suggestions["hidden_dim"] = 256
    
    # Number of layers based on complexity
    if n_cells < 50000:
        suggestions["num_layers"] = 3
    else:
        suggestions["num_layers"] = 2  # Fewer layers for large datasets
    
    # Dropout based on size
    if n_cells < 10000:
        suggestions["dropout"] = 0.1  # Less regularization for small data
    else:
        suggestions["dropout"] = 0.3  # More regularization for large data
    
    return suggestions


def format_dataset_info(info: Dict[str, Any]) -> str:
    """Format dataset information for display."""
    lines = [
        f"Dataset: {info.get('name', 'Unknown')}",
        f"Description: {info.get('description', 'No description')}",
        "",
        "Data Characteristics:",
        f"  Cells: {info.get('n_cells', 0):,}",
        f"  Genes: {info.get('n_genes', 0):,}",
        f"  Cell types: {info.get('n_cell_types', 'Unknown')}",
        f"  Modality: {info.get('modality', 'Unknown')}",
        f"  Organism: {info.get('organism', 'Unknown')}",
        f"  Tissue: {info.get('tissue', 'Unknown')}",
    ]
    
    if info.get('has_spatial'):
        lines.append("  ✓ Spatial coordinates available")
    
    if info.get('size_mb'):
        lines.extend([
            "",
            f"Storage: {info['size_mb']:.1f} MB"
        ])
    
    if info.get('citation'):
        lines.extend([
            "",
            f"Citation: {info['citation']}"
        ])
    
    return "\n".join(lines)


def validate_graph_data(x, edge_index, y=None) -> bool:
    """Validate graph data tensors."""
    try:
        import torch
    except ImportError:
        warnings.warn("PyTorch not available, skipping validation")
        return True
    
    # Check node features
    if not isinstance(x, torch.Tensor):
        raise TypeError("Node features (x) must be a torch.Tensor")
    
    if x.dim() != 2:
        raise ValueError("Node features must be 2D [num_nodes, num_features]")
    
    n_nodes = x.shape[0]
    
    # Check edge indices
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("Edge indices must be a torch.Tensor")
    
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("Edge indices must be 2D [2, num_edges]")
    
    # Check edge indices are valid
    if edge_index.max() >= n_nodes:
        raise ValueError("Edge indices contain invalid node references")
    
    if edge_index.min() < 0:
        raise ValueError("Edge indices must be non-negative")
    
    # Check labels if provided
    if y is not None:
        if not isinstance(y, torch.Tensor):
            raise TypeError("Labels (y) must be a torch.Tensor")
        
        if y.shape[0] != n_nodes:
            raise ValueError("Number of labels must match number of nodes")
    
    return True


def compute_graph_statistics(edge_index, n_nodes: int) -> Dict[str, float]:
    """Compute basic graph statistics."""
    try:
        import torch
    except ImportError:
        warnings.warn("PyTorch not available for graph statistics")
        return {}
    
    if not _NUMPY_AVAILABLE:
        warnings.warn("NumPy not available for graph statistics")
        return {}
    
    n_edges = edge_index.shape[1]
    
    # Degree statistics
    degrees = torch.zeros(n_nodes, dtype=torch.long)
    degrees.scatter_add_(0, edge_index[1], torch.ones_like(edge_index[1]))
    
    stats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
        "avg_degree": degrees.float().mean().item(),
        "max_degree": degrees.max().item(),
        "min_degree": degrees.min().item(),
    }
    
    return stats


class ProgressCallback:
    """Simple progress callback for training."""
    
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self.step = 0
    
    def __call__(self, loss: float, metrics: Optional[Dict[str, float]] = None):
        """Log progress."""
        self.step += 1
        
        if self.step % self.log_every == 0:
            msg = f"Step {self.step}: Loss {loss:.4f}"
            
            if metrics:
                metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                msg += f", {metric_str}"
            
            print(msg)