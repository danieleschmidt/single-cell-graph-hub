"""Core dataset classes for single-cell graph data."""

import os
from typing import Optional, Tuple, Dict, Any, List
import warnings

import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


class SCGraphDataset(Dataset):
    """Single-cell graph dataset for PyTorch Geometric.
    
    This class provides a unified interface for loading single-cell omics
    datasets as graph-structured data, where cells are nodes and their
    relationships form edges.
    
    Args:
        name: Name of the dataset to load
        root: Root directory for data storage
        task: Task type for this dataset (e.g., 'cell_type_prediction')
        download: Whether to download the dataset if not found
        transform: Optional transform to apply to each data sample
        pre_transform: Optional pre-transform to apply to the entire dataset
        
    Example:
        >>> dataset = SCGraphDataset(
        ...     name="pbmc_10k",
        ...     root="./data",
        ...     task="cell_type_prediction",
        ...     download=True
        ... )
        >>> print(f"Number of cells: {dataset.num_nodes}")
    """
    
    def __init__(
        self,
        name: str,
        root: str,
        task: str = "cell_type_prediction",
        download: bool = False,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
    ):
        self.name = name
        self.task = task
        self._download = download
        
        # Dataset metadata
        self._metadata = self._load_metadata()
        
        super().__init__(root, transform, pre_transform)
        
        # Load the processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        # For now, return mock metadata for the demo dataset
        if self.name == "pbmc_10k":
            return {
                "n_cells": 10000,
                "n_genes": 2000,
                "n_classes": 8,
                "modality": "scRNA-seq",
                "organism": "human",
                "tissue": "blood",
                "description": "10k PBMCs from a healthy donor"
            }
        else:
            # Return default metadata for unknown datasets
            warnings.warn(f"Unknown dataset '{self.name}', using default metadata")
            return {
                "n_cells": 1000,
                "n_genes": 500,
                "n_classes": 5,
                "modality": "scRNA-seq",
                "organism": "unknown",
                "tissue": "unknown",
                "description": f"Mock dataset: {self.name}"
            }
    
    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return [f"{self.name}.h5ad"]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"{self.name}_{self.task}.pt"]
    
    def download(self):
        """Download the raw data files."""
        if not self._download:
            return
            
        # For demonstration, create a mock dataset
        self._create_mock_dataset()
    
    def _create_mock_dataset(self):
        """Create a mock dataset for demonstration purposes."""
        raw_dir = self.raw_dir
        os.makedirs(raw_dir, exist_ok=True)
        
        # Create a simple mock .h5ad file (just a marker file)
        mock_file = os.path.join(raw_dir, f"{self.name}.h5ad")
        with open(mock_file, 'w') as f:
            f.write(f"# Mock dataset file for {self.name}\n")
    
    def process(self):
        """Process the raw data and save as PyTorch Geometric data."""
        # Generate mock graph data
        data = self._generate_mock_graph()
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        # Save processed data
        torch.save(self.collate([data]), self.processed_paths[0])
    
    def _generate_mock_graph(self) -> Data:
        """Generate a mock graph for demonstration."""
        n_cells = self._metadata["n_cells"]
        n_genes = self._metadata["n_genes"]
        n_classes = self._metadata["n_classes"]
        
        # Generate random gene expression data
        x = torch.randn(n_cells, n_genes)
        
        # Generate random labels for classification
        y = torch.randint(0, n_classes, (n_cells,))
        
        # Create a random k-NN graph (k=10)
        k = min(10, n_cells - 1)
        edge_index = self._create_knn_graph(x, k)
        
        # Create train/val/test masks
        train_mask, val_mask, test_mask = self._create_splits(n_cells)
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
    
    def _create_knn_graph(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """Create a k-nearest neighbor graph from features."""
        from sklearn.neighbors import NearestNeighbors
        
        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features.numpy())
        distances, indices = nbrs.kneighbors(features.numpy())
        
        # Create edge list (excluding self-connections)
        edge_list = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip self (index 0)
                edge_list.append([i, indices[i][j]])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _create_splits(self, n_cells: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/validation/test splits."""
        indices = torch.randperm(n_cells)
        
        n_train = int(0.6 * n_cells)
        n_val = int(0.2 * n_cells)
        
        train_mask = torch.zeros(n_cells, dtype=torch.bool)
        val_mask = torch.zeros(n_cells, dtype=torch.bool)
        test_mask = torch.zeros(n_cells, dtype=torch.bool)
        
        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train:n_train + n_val]] = True
        test_mask[indices[n_train + n_val:]] = True
        
        return train_mask, val_mask, test_mask
    
    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return 1  # Single large graph
    
    def get(self, idx: int) -> Data:
        """Get a single data sample."""
        return self.data
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes (cells) in the dataset."""
        return self._metadata["n_cells"]
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the dataset."""
        if hasattr(self.data, 'edge_index'):
            return self.data.edge_index.shape[1]
        return 0
    
    @property
    def num_node_features(self) -> int:
        """Number of node features (genes)."""
        return self._metadata["n_genes"]
    
    @property
    def num_classes(self) -> int:
        """Number of classes for classification tasks."""
        return self._metadata["n_classes"]
    
    def get_loaders(
        self, 
        batch_size: int = 32, 
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get PyTorch Geometric data loaders for train/val/test splits.
        
        Note: For node-level tasks, we typically work with a single large graph,
        so the loaders will contain the same graph with different masks.
        
        Args:
            batch_size: Batch size (not used for single graph datasets)
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # For single graph datasets, we create loaders that just return the graph
        train_loader = DataLoader([self.data], batch_size=1, shuffle=False, num_workers=num_workers)
        val_loader = DataLoader([self.data], batch_size=1, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader([self.data], batch_size=1, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    
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