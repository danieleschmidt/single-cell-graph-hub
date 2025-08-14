"""Simplified dataset class for basic functionality without heavy dependencies."""

import os
import json
import warnings
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union


class SimpleSCGraphData:
    """Simplified graph data structure without PyTorch dependencies."""
    
    def __init__(
        self,
        x=None,
        edge_index=None, 
        y=None,
        train_mask=None,
        val_mask=None,
        test_mask=None,
        **kwargs
    ):
        self.x = x  # Node features
        self.edge_index = edge_index  # Edge connectivity
        self.y = y  # Labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        
        # Store additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        if self.x is not None:
            try:
                import numpy as np
                if isinstance(self.x, np.ndarray):
                    return self.x.shape[0]
            except ImportError:
                pass
        return 0
    
    @property
    def num_edges(self) -> int:
        """Number of edges."""
        if self.edge_index is not None:
            try:
                import numpy as np
                if isinstance(self.edge_index, np.ndarray):
                    return self.edge_index.shape[1]
            except ImportError:
                pass
        return 0
    
    @property
    def num_node_features(self) -> int:
        """Number of node features."""
        if self.x is not None:
            try:
                import numpy as np
                if isinstance(self.x, np.ndarray):
                    return self.x.shape[1]
            except ImportError:
                pass
        return 0


class SimpleSCGraphDataset:
    """Simplified single-cell graph dataset without PyTorch Geometric dependencies."""
    
    def __init__(
        self,
        name: str,
        root: str = "./data",
        task: str = "cell_type_prediction",
        download: bool = False,
        **kwargs
    ):
        self.name = name
        self.root = root
        self.task = task
        self._download = download
        
        # Dataset metadata
        self._metadata = self._load_metadata()
        
        # Create directories
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, self.name), exist_ok=True)
        
        # Process or load data
        self.data = self._get_or_create_data()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata from catalog."""
        try:
            from .catalog import get_default_catalog
            catalog = get_default_catalog()
            
            try:
                metadata = catalog.get_info(self.name)
                return metadata
            except KeyError:
                warnings.warn(f"Dataset '{self.name}' not found in catalog, using defaults")
        except ImportError:
            pass
        
        # Default metadata
        return {
            "name": self.name,
            "n_cells": 1000,
            "n_genes": 500,
            "n_classes": 5,
            "modality": "scRNA-seq",
            "organism": "human",
            "tissue": "unknown",
            "description": f"Mock dataset: {self.name}"
        }
    
    def _get_or_create_data(self) -> SimpleSCGraphData:
        """Get existing data or create mock data."""
        processed_file = os.path.join(self.root, self.name, "processed_data.json")
        
        if os.path.exists(processed_file):
            try:
                with open(processed_file, 'r') as f:
                    data_dict = json.load(f)
                return self._dict_to_data(data_dict)
            except Exception as e:
                warnings.warn(f"Failed to load processed data: {e}")
        
        # Create mock data
        data = self._create_mock_data()
        
        # Save for future use
        try:
            self._save_data(data, processed_file)
        except Exception as e:
            warnings.warn(f"Failed to save processed data: {e}")
        
        return data
    
    def _create_mock_data(self) -> SimpleSCGraphData:
        """Create realistic mock single-cell graph data."""
        try:
            import numpy as np
            n_cells = self._metadata["n_cells"]
            n_genes = self._metadata["n_genes"]
            n_classes = self._metadata["n_classes"]
            
            # Set random seed for reproducibility
            np.random.seed(42)
            
            # Generate realistic gene expression matrix
            # Use log-normal distribution with dropout (common in scRNA-seq)
            x = np.random.lognormal(mean=0, sigma=1, size=(n_cells, n_genes))
            
            # Add dropout (zeros) - typical in single-cell data
            dropout_prob = 0.7
            dropout_mask = np.random.binomial(1, 1-dropout_prob, size=(n_cells, n_genes))
            x = x * dropout_mask
            
            # Create k-NN graph
            k = min(15, n_cells - 1)
            edge_index = self._create_knn_graph(x, k)
            
            # Create class labels
            y = np.random.randint(0, n_classes, size=n_cells)
            
            # Create train/val/test splits
            train_mask, val_mask, test_mask = self._create_splits(n_cells)
            
            return SimpleSCGraphData(
                x=x,
                edge_index=edge_index,
                y=y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            
        except ImportError:
            warnings.warn("NumPy not available - creating minimal data structure")
            return SimpleSCGraphData()
    
    def _create_knn_graph(self, features, k: int):
        """Create k-NN graph from features."""
        try:
            from sklearn.neighbors import NearestNeighbors
            import numpy as np
            
            # Fit k-NN
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features)
            distances, indices = nbrs.kneighbors(features)
            
            # Create edge list (excluding self-connections)
            edge_list = []
            for i in range(len(indices)):
                for j in range(1, len(indices[i])):  # Skip self (index 0)
                    neighbor_idx = indices[i][j]
                    # Add both directions for undirected graph
                    edge_list.append([i, neighbor_idx])
                    edge_list.append([neighbor_idx, i])
            
            # Remove duplicates and convert to array
            edge_set = set(map(tuple, edge_list))
            edge_index = np.array(list(edge_set)).T
            return edge_index
            
        except ImportError:
            # Fallback without sklearn - create simple ring graph
            try:
                import numpy as np
                n_nodes = features.shape[0]
                edges = []
                for i in range(n_nodes):
                    # Connect to next few nodes in a ring
                    for j in range(1, min(k+1, n_nodes)):
                        next_node = (i + j) % n_nodes
                        edges.append([i, next_node])
                        edges.append([next_node, i])
                
                edge_index = np.array(edges).T
                return edge_index
                
            except ImportError:
                return None
    
    def _create_splits(self, n_cells: int) -> Tuple:
        """Create train/validation/test splits."""
        try:
            import numpy as np
            indices = np.random.permutation(n_cells)
            
            n_train = int(0.6 * n_cells)
            n_val = int(0.2 * n_cells)
            
            train_mask = np.zeros(n_cells, dtype=bool)
            val_mask = np.zeros(n_cells, dtype=bool)
            test_mask = np.zeros(n_cells, dtype=bool)
            
            train_mask[indices[:n_train]] = True
            val_mask[indices[n_train:n_train + n_val]] = True
            test_mask[indices[n_train + n_val:]] = True
            
            return train_mask, val_mask, test_mask
            
        except ImportError:
            return None, None, None
    
    def _dict_to_data(self, data_dict: dict) -> SimpleSCGraphData:
        """Convert dictionary to data object."""
        try:
            import numpy as np
            
            # Convert lists back to numpy arrays
            converted = {}
            for key, value in data_dict.items():
                if isinstance(value, list):
                    converted[key] = np.array(value)
                else:
                    converted[key] = value
            
            return SimpleSCGraphData(**converted)
            
        except ImportError:
            return SimpleSCGraphData(**data_dict)
    
    def _save_data(self, data: SimpleSCGraphData, filepath: str):
        """Save data to JSON file."""
        try:
            import numpy as np
            
            # Convert numpy arrays to lists for JSON serialization
            data_dict = {}
            for key in ['x', 'edge_index', 'y', 'train_mask', 'val_mask', 'test_mask']:
                value = getattr(data, key, None)
                if value is not None and isinstance(value, np.ndarray):
                    data_dict[key] = value.tolist()
                elif value is not None:
                    data_dict[key] = value
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data_dict, f, indent=2)
                
        except ImportError:
            pass  # Skip saving if numpy not available
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes (cells) in the dataset."""
        return self._metadata.get("n_cells", 0)
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the dataset."""
        return self.data.num_edges if self.data else 0
    
    @property
    def num_node_features(self) -> int:
        """Number of node features (genes)."""
        return self._metadata.get("n_genes", 0)
    
    @property
    def num_classes(self) -> int:
        """Number of classes for classification tasks."""
        return self._metadata.get("n_classes", 0)
    
    def info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.name,
            "task": self.task,
            "num_cells": self.num_nodes,
            "num_genes": self.num_node_features,
            "num_edges": self.num_edges,
            "num_classes": self.num_classes,
            **self._metadata
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Alias for info() method for compatibility."""
        return self.info()