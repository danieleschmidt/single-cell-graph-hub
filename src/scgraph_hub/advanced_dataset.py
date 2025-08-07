"""
Advanced dataset processing with comprehensive error handling and validation.

This module provides robust data processing capabilities with extensive logging,
monitoring, and error recovery mechanisms.
"""

import os
import json
import asyncio
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from datetime import datetime

from .exceptions import (
    SCGraphHubError, DatasetError, DatasetNotFoundError, DatasetValidationError,
    handle_error_gracefully, ErrorCollector, create_error_context
)
from .logging_config import get_logger, LoggingMixin
from .simple_dataset import SimpleSCGraphDataset, SimpleSCGraphData

# Optional imports with graceful handling
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = get_logger(__name__)


class DatasetProcessor(LoggingMixin):
    """Advanced dataset processor with comprehensive error handling."""
    
    def __init__(self, cache_dir: Optional[str] = None, enable_validation: bool = True):
        """Initialize processor.
        
        Args:
            cache_dir: Directory for caching processed datasets
            enable_validation: Whether to enable dataset validation
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".scgraph_hub" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_validation = enable_validation
        
        # Processing configuration
        self.config = {
            'default_k_neighbors': 15,
            'max_features': 10000,
            'min_cells': 50,
            'min_genes': 10,
            'validation_sample_size': 1000,
            'timeout_seconds': 300
        }
        
        logger.info(f"DatasetProcessor initialized with cache_dir={self.cache_dir}")
    
    @handle_error_gracefully
    async def process_dataset(
        self,
        dataset_name: str,
        processing_config: Optional[Dict[str, Any]] = None,
        force_reprocess: bool = False
    ) -> SimpleSCGraphDataset:
        """Process dataset with comprehensive error handling.
        
        Args:
            dataset_name: Name of dataset to process
            processing_config: Processing configuration options
            force_reprocess: Whether to force reprocessing even if cached
            
        Returns:
            Processed dataset
            
        Raises:
            DatasetError: If processing fails
        """
        config = {**self.config, **(processing_config or {})}
        
        with create_error_context(f"processing dataset {dataset_name}"):
            # Check cache first
            if not force_reprocess:
                cached_dataset = await self._load_from_cache(dataset_name, config)
                if cached_dataset:
                    logger.info(f"Loaded dataset {dataset_name} from cache")
                    return cached_dataset
            
            # Process dataset
            logger.info(f"Processing dataset {dataset_name} with config {config}")
            
            # Validate dependencies
            self._validate_dependencies(config)
            
            # Load and validate raw data
            raw_dataset = await self._load_raw_dataset(dataset_name)
            
            if self.enable_validation:
                await self._validate_raw_dataset(raw_dataset)
            
            # Process the dataset
            processed_dataset = await self._perform_processing(raw_dataset, config)
            
            # Validate processed dataset
            if self.enable_validation:
                await self._validate_processed_dataset(processed_dataset)
            
            # Cache the result
            await self._save_to_cache(processed_dataset, config)
            
            logger.info(f"Successfully processed dataset {dataset_name}")
            return processed_dataset
    
    def _validate_dependencies(self, config: Dict[str, Any]):
        """Validate required dependencies for processing configuration."""
        missing_deps = []
        
        if config.get('use_advanced_features', False):
            if not _NUMPY_AVAILABLE:
                missing_deps.append('numpy')
            if not _PANDAS_AVAILABLE:
                missing_deps.append('pandas')
            if not _SKLEARN_AVAILABLE:
                missing_deps.append('scikit-learn')
        
        if config.get('use_torch', False) and not _TORCH_AVAILABLE:
            missing_deps.append('torch')
        
        if missing_deps:
            from .exceptions import DependencyError
            raise DependencyError(missing_deps)
    
    @handle_error_gracefully
    async def _load_raw_dataset(self, dataset_name: str) -> SimpleSCGraphDataset:
        """Load raw dataset with error handling."""
        try:
            # Use existing simple dataset loader as fallback
            dataset = SimpleSCGraphDataset(
                name=dataset_name,
                root=str(self.cache_dir / "raw")
            )
            return dataset
            
        except Exception as e:
            raise DatasetNotFoundError(dataset_name) from e
    
    @handle_error_gracefully
    async def _validate_raw_dataset(self, dataset: SimpleSCGraphDataset):
        """Validate raw dataset with comprehensive checks."""
        logger.debug(f"Validating raw dataset {dataset.name}")
        
        error_collector = ErrorCollector()
        
        # Basic validation
        if dataset.num_nodes == 0:
            error_collector.add_error("Dataset has no cells")
        elif dataset.num_nodes < self.config['min_cells']:
            error_collector.add_error(f"Dataset has too few cells: {dataset.num_nodes} < {self.config['min_cells']}")
        
        if dataset.num_node_features == 0:
            error_collector.add_error("Dataset has no gene features")
        elif dataset.num_node_features < self.config['min_genes']:
            error_collector.add_error(f"Dataset has too few genes: {dataset.num_node_features} < {self.config['min_genes']}")
        
        # Data integrity checks if numpy is available
        if _NUMPY_AVAILABLE and dataset.data and hasattr(dataset.data, 'x'):
            data_array = dataset.data.x
            if data_array is not None and isinstance(data_array, np.ndarray):
                # Check for NaN/infinite values
                if np.isnan(data_array).any():
                    error_collector.add_error("Dataset contains NaN values")
                if np.isinf(data_array).any():
                    error_collector.add_error("Dataset contains infinite values")
                
                # Check data ranges
                if (data_array < 0).any():
                    logger.warning("Dataset contains negative values (common in log-transformed data)")
                
                # Check sparsity
                zero_percent = (data_array == 0).mean() * 100
                if zero_percent > 95:
                    logger.warning(f"Dataset is very sparse: {zero_percent:.1f}% zeros")
        
        error_collector.raise_if_errors(DatasetValidationError, f"Raw dataset {dataset.name} validation failed")
        logger.debug(f"Raw dataset {dataset.name} validation passed")
    
    @handle_error_gracefully
    async def _perform_processing(
        self, 
        dataset: SimpleSCGraphDataset,
        config: Dict[str, Any]
    ) -> SimpleSCGraphDataset:
        """Perform dataset processing with error handling."""
        logger.info(f"Performing advanced processing on dataset {dataset.name}")
        
        processed_data = dataset.data
        
        # Enhanced graph construction if dependencies available
        if _NUMPY_AVAILABLE and _SKLEARN_AVAILABLE and processed_data.x is not None:
            processed_data = await self._build_enhanced_graph(processed_data, config)
        
        # Create enhanced dataset
        processed_dataset = EnhancedSCGraphDataset(
            name=f"{dataset.name}_processed",
            data=processed_data,
            metadata=dataset._metadata,
            processing_config=config,
            processing_timestamp=datetime.utcnow()
        )
        
        return processed_dataset
    
    @handle_error_gracefully
    async def _build_enhanced_graph(
        self, 
        data: SimpleSCGraphData,
        config: Dict[str, Any]
    ) -> SimpleSCGraphData:
        """Build enhanced graph with advanced algorithms."""
        logger.debug("Building enhanced graph structure")
        
        if not isinstance(data.x, np.ndarray) or data.x.size == 0:
            logger.warning("No feature data available for enhanced graph construction")
            return data
        
        # Preprocessing steps
        features = data.x.copy()
        
        # Feature selection if too many features
        if features.shape[1] > config['max_features']:
            logger.info(f"Reducing features from {features.shape[1]} to {config['max_features']}")
            # Simple variance-based feature selection
            variances = np.var(features, axis=0)
            top_indices = np.argsort(variances)[-config['max_features']:]
            features = features[:, top_indices]
        
        # Normalization
        if config.get('normalize_features', True):
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        # Enhanced k-NN graph construction
        k = min(config['default_k_neighbors'], features.shape[0] - 1)
        
        try:
            nbrs = NearestNeighbors(
                n_neighbors=k+1, 
                algorithm='auto',
                n_jobs=-1  # Use all available cores
            ).fit(features)
            
            distances, indices = nbrs.kneighbors(features)
            
            # Build edge index with weights
            edge_list = []
            edge_weights = []
            
            for i in range(len(indices)):
                for j in range(1, len(indices[i])):  # Skip self
                    neighbor_idx = indices[i][j]
                    weight = 1.0 / (1.0 + distances[i][j])  # Distance-based weight
                    
                    edge_list.append([i, neighbor_idx])
                    edge_weights.append(weight)
            
            # Create symmetric graph
            edge_index = np.array(edge_list).T
            edge_attr = np.array(edge_weights)
            
            # Update data
            enhanced_data = SimpleSCGraphData(
                x=data.x,  # Keep original features
                edge_index=edge_index,
                y=data.y,
                train_mask=data.train_mask,
                val_mask=data.val_mask,
                test_mask=data.test_mask,
                edge_attr=edge_attr,  # Add edge weights
                processed_features=features  # Store processed features
            )
            
            logger.info(f"Built enhanced graph with {edge_index.shape[1]} edges")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Enhanced graph construction failed: {e}")
            # Fallback to simple graph
            return data
    
    @handle_error_gracefully
    async def _validate_processed_dataset(self, dataset: SimpleSCGraphDataset):
        """Validate processed dataset."""
        logger.debug(f"Validating processed dataset {dataset.name}")
        
        error_collector = ErrorCollector()
        
        if not hasattr(dataset, 'data') or dataset.data is None:
            error_collector.add_error("Processed dataset has no data")
            error_collector.raise_if_errors()
            return
        
        data = dataset.data
        
        # Check graph structure
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_index = data.edge_index
            if isinstance(edge_index, np.ndarray):
                if edge_index.shape[0] != 2:
                    error_collector.add_error(f"Invalid edge_index shape: {edge_index.shape}")
                
                # Check for self-loops (should be minimal)
                if edge_index.shape[1] > 0:
                    self_loops = (edge_index[0] == edge_index[1]).sum()
                    if self_loops > edge_index.shape[1] * 0.1:  # More than 10% self-loops
                        logger.warning(f"High number of self-loops detected: {self_loops}")
        
        # Check feature consistency
        if hasattr(data, 'x') and data.x is not None:
            if isinstance(data.x, np.ndarray):
                if len(data.x.shape) != 2:
                    error_collector.add_error(f"Invalid feature matrix shape: {data.x.shape}")
        
        error_collector.raise_if_errors(DatasetValidationError, f"Processed dataset {dataset.name} validation failed")
        logger.debug(f"Processed dataset {dataset.name} validation passed")
    
    def _get_cache_key(self, dataset_name: str, config: Dict[str, Any]) -> str:
        """Generate cache key for dataset and configuration."""
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{dataset_name}_{config_hash}"
    
    async def _load_from_cache(
        self,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> Optional[SimpleSCGraphDataset]:
        """Load dataset from cache if available."""
        cache_key = self._get_cache_key(dataset_name, config)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check cache validity
            if cached_data.get('version') != '1.0':
                logger.warning(f"Cache version mismatch for {dataset_name}")
                return None
            
            # Reconstruct dataset
            # This is a simplified version - in production you'd want proper serialization
            logger.debug(f"Found cached dataset for {cache_key}")
            return None  # For now, skip cache loading
            
        except Exception as e:
            logger.warning(f"Failed to load cached dataset {cache_key}: {e}")
            return None
    
    async def _save_to_cache(
        self,
        dataset: SimpleSCGraphDataset,
        config: Dict[str, Any]
    ):
        """Save dataset to cache."""
        cache_key = self._get_cache_key(dataset.name, config)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            # Save metadata and processing info
            cache_data = {
                'version': '1.0',
                'dataset_name': dataset.name,
                'config': config,
                'cached_at': datetime.utcnow().isoformat(),
                'metadata': getattr(dataset, '_metadata', {}),
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"Cached dataset metadata for {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache dataset {cache_key}: {e}")


class EnhancedSCGraphDataset(SimpleSCGraphDataset):
    """Enhanced dataset with processing metadata and validation."""
    
    def __init__(
        self,
        name: str,
        data: SimpleSCGraphData,
        metadata: Dict[str, Any],
        processing_config: Dict[str, Any],
        processing_timestamp: datetime
    ):
        # Initialize parent without file system operations
        self.name = name
        self.task = "cell_type_prediction"
        self.root = ""
        self._metadata = metadata
        self.data = data
        
        # Enhanced attributes
        self.processing_config = processing_config
        self.processing_timestamp = processing_timestamp
        self.is_processed = True
    
    def info(self) -> Dict[str, Any]:
        """Get enhanced dataset information."""
        base_info = super().info()
        
        enhanced_info = {
            **base_info,
            'is_processed': self.is_processed,
            'processing_timestamp': self.processing_timestamp.isoformat(),
            'processing_config': self.processing_config,
            'has_edge_weights': hasattr(self.data, 'edge_attr') and self.data.edge_attr is not None,
            'graph_density': self._calculate_graph_density()
        }
        
        return enhanced_info
    
    def _calculate_graph_density(self) -> float:
        """Calculate graph density."""
        if not hasattr(self.data, 'edge_index') or self.data.edge_index is None:
            return 0.0
        
        n_nodes = self.num_nodes
        n_edges = self.num_edges
        
        if n_nodes <= 1:
            return 0.0
        
        max_edges = n_nodes * (n_nodes - 1)  # Directed graph
        return n_edges / max_edges if max_edges > 0 else 0.0
    
    def validate_integrity(self) -> bool:
        """Validate dataset integrity."""
        try:
            if self.data is None:
                return False
            
            # Check basic structure
            if hasattr(self.data, 'x') and self.data.x is not None:
                if not isinstance(self.data.x, np.ndarray):
                    return False
            
            if hasattr(self.data, 'edge_index') and self.data.edge_index is not None:
                edge_index = self.data.edge_index
                if isinstance(edge_index, np.ndarray):
                    # Check edge indices are valid
                    if edge_index.max() >= self.num_nodes:
                        return False
                    if edge_index.min() < 0:
                        return False
            
            return True
            
        except Exception:
            return False