"""Core dataset classes for single-cell graph data."""

import os
import json
import h5py
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
import warnings

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
import scanpy as sc


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
        """Load dataset metadata from catalog or file."""
        from .catalog import DatasetCatalog
        
        catalog = DatasetCatalog()
        
        # Try to get metadata from catalog first
        try:
            metadata = catalog.get_info(self.name)
            if metadata:
                return metadata
        except Exception:
            pass
        
        # Try to load from metadata file
        metadata_path = os.path.join(self.root, self.name, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        # Try to infer from raw data
        raw_path = os.path.join(self.raw_dir, f"{self.name}.h5ad")
        if os.path.exists(raw_path):
            try:
                adata = sc.read_h5ad(raw_path)
                return self._extract_metadata_from_adata(adata)
            except Exception as e:
                warnings.warn(f"Could not load metadata from {raw_path}: {e}")
        
        # Return default metadata as fallback
        warnings.warn(f"No metadata found for '{self.name}', using defaults")
        return {
            "n_cells": 1000,
            "n_genes": 500,
            "n_classes": 5,
            "modality": "scRNA-seq",
            "organism": "unknown",
            "tissue": "unknown",
            "description": f"Dataset: {self.name}"
        }
    
    def _extract_metadata_from_adata(self, adata) -> Dict[str, Any]:
        """Extract metadata from AnnData object."""
        n_cells, n_genes = adata.shape
        
        # Try to determine number of classes
        n_classes = 5  # default
        if self.task == "cell_type_prediction":
            if 'cell_type' in adata.obs.columns:
                n_classes = adata.obs['cell_type'].nunique()
            elif 'celltype' in adata.obs.columns:
                n_classes = adata.obs['celltype'].nunique()
            elif 'cluster' in adata.obs.columns:
                n_classes = adata.obs['cluster'].nunique()
        
        # Extract organism and tissue if available
        organism = adata.uns.get('organism', 'unknown')
        tissue = adata.uns.get('tissue', 'unknown')
        
        return {
            "n_cells": n_cells,
            "n_genes": n_genes, 
            "n_classes": n_classes,
            "modality": "scRNA-seq",
            "organism": organism,
            "tissue": tissue,
            "description": f"Single-cell dataset: {self.name}"
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
        
        # Check if dataset exists in catalog
        from .catalog import DatasetCatalog
        catalog = DatasetCatalog()
        
        try:
            # Try to download from catalog
            success = catalog.download_dataset(self.name, self.raw_dir)
            if success:
                return
        except Exception as e:
            warnings.warn(f"Could not download {self.name} from catalog: {e}")
        
        # For demonstration/testing, create a realistic mock dataset
        self._create_realistic_mock_dataset()
    
    def _create_realistic_mock_dataset(self):
        """Create a realistic mock dataset with proper single-cell structure."""
        raw_dir = self.raw_dir
        os.makedirs(raw_dir, exist_ok=True)
        
        # Generate realistic single-cell data
        np.random.seed(42)  # for reproducibility
        n_cells = self._metadata["n_cells"]
        n_genes = self._metadata["n_genes"]
        n_classes = self._metadata["n_classes"]
        
        # Create realistic gene expression matrix
        # Model: log-normal distribution with dropout (common in scRNA-seq)
        base_expression = np.random.lognormal(mean=0, sigma=1, size=(n_cells, n_genes))
        
        # Add dropout (zeros) - typical in single-cell data
        dropout_prob = 0.7  # 70% dropout rate
        dropout_mask = np.random.binomial(1, 1-dropout_prob, size=(n_cells, n_genes))
        expression_matrix = base_expression * dropout_mask
        
        # Create cell type labels with biological structure
        cell_types = [f"CellType_{i}" for i in range(n_classes)]
        # Create clusters of cells (realistic cell type distribution)
        cells_per_type = n_cells // n_classes
        y_labels = []
        for i, ct in enumerate(cell_types):
            count = cells_per_type + (1 if i < n_cells % n_classes else 0)
            y_labels.extend([ct] * count)
        
        # Create gene names
        gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
        
        # Create cell barcodes
        cell_barcodes = [f"Cell_{i:06d}" for i in range(n_cells)]
        
        # Create AnnData object
        adata = sc.AnnData(
            X=expression_matrix,
            obs=pd.DataFrame({
                'cell_type': y_labels,
                'n_genes': np.sum(expression_matrix > 0, axis=1),
                'total_counts': np.sum(expression_matrix, axis=1)
            }, index=cell_barcodes),
            var=pd.DataFrame({
                'gene_name': gene_names,
                'highly_variable': np.random.choice([True, False], n_genes, p=[0.2, 0.8])
            }, index=gene_names)
        )
        
        # Add metadata
        adata.uns['organism'] = self._metadata.get('organism', 'human')
        adata.uns['tissue'] = self._metadata.get('tissue', 'unknown')
        adata.uns['dataset_name'] = self.name
        
        # Save as H5AD file
        mock_file = os.path.join(raw_dir, f"{self.name}.h5ad")
        adata.write_h5ad(mock_file)
        
        # Save metadata
        metadata_file = os.path.join(raw_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)
    
    def process(self):
        """Process the raw data and save as PyTorch Geometric data."""
        # Load and process real data
        raw_path = os.path.join(self.raw_dir, f"{self.name}.h5ad")
        
        if os.path.exists(raw_path):
            data = self._process_h5ad_file(raw_path)
        else:
            # Fallback to mock data if no real data available
            warnings.warn(f"Raw data not found at {raw_path}, generating mock data")
            data = self._generate_mock_graph()
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        # Save processed data
        torch.save(self.collate([data]), self.processed_paths[0])
    
    def _process_h5ad_file(self, file_path: str) -> Data:
        """Process H5AD file into PyTorch Geometric Data object."""
        # Load AnnData
        adata = sc.read_h5ad(file_path)
        
        # Extract features (gene expression)
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        # Convert to float32 tensor
        x = torch.FloatTensor(X)
        
        # Extract labels based on task
        y = self._extract_labels(adata)
        
        # Build graph structure
        edge_index = self._build_cell_graph(x)
        
        # Create train/val/test splits
        train_mask, val_mask, test_mask = self._create_splits(len(adata))
        
        # Create additional node attributes if available
        node_attrs = self._extract_node_attributes(adata)
        
        data = Data(
            x=x,
            edge_index=edge_index, 
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            **node_attrs
        )
        
        return data
    
    def _extract_labels(self, adata) -> torch.Tensor:
        """Extract labels based on the task type."""
        if self.task == "cell_type_prediction":
            # Try different column names for cell types
            for col in ['cell_type', 'celltype', 'cluster', 'louvain', 'leiden']:
                if col in adata.obs.columns:
                    labels = adata.obs[col]
                    # Convert to categorical codes
                    if labels.dtype == 'object' or labels.dtype.name == 'category':
                        label_map = {label: i for i, label in enumerate(labels.unique())}
                        numeric_labels = [label_map[label] for label in labels]
                        return torch.LongTensor(numeric_labels)
                    else:
                        return torch.LongTensor(labels.values)
            
            # If no labels found, create random ones
            warnings.warn(f"No cell type labels found, creating random labels")
            n_classes = self._metadata["n_classes"]
            return torch.randint(0, n_classes, (len(adata),))
        
        elif self.task == "trajectory_inference":
            # Look for pseudotime or trajectory information
            for col in ['dpt_pseudotime', 'pseudotime', 'time', 'age']:
                if col in adata.obs.columns:
                    return torch.FloatTensor(adata.obs[col].values)
            
            # Create mock trajectory
            return torch.FloatTensor(np.random.uniform(0, 1, len(adata)))
        
        else:
            # Default: random classification labels
            n_classes = self._metadata["n_classes"]
            return torch.randint(0, n_classes, (len(adata),))
    
    def _extract_node_attributes(self, adata) -> Dict[str, torch.Tensor]:
        """Extract additional node attributes from AnnData."""
        attrs = {}
        
        # Cell metadata as node attributes
        if 'n_genes' in adata.obs.columns:
            attrs['n_genes'] = torch.FloatTensor(adata.obs['n_genes'].values)
        
        if 'total_counts' in adata.obs.columns:
            attrs['total_counts'] = torch.FloatTensor(adata.obs['total_counts'].values)
        
        # Spatial coordinates if available
        if 'spatial' in adata.obsm.keys():
            spatial_coords = adata.obsm['spatial']
            attrs['spatial_coords'] = torch.FloatTensor(spatial_coords)
        
        # PCA coordinates if available
        if 'X_pca' in adata.obsm.keys():
            pca_coords = adata.obsm['X_pca']
            attrs['pca_coords'] = torch.FloatTensor(pca_coords)
        
        return attrs
    
    def _build_cell_graph(self, features: torch.Tensor, method: str = 'knn') -> torch.Tensor:
        """Build cell-cell graph from gene expression features."""
        if method == 'knn':
            return self._create_knn_graph(features, k=15)
        elif method == 'radius':
            return self._create_radius_graph(features, radius=1.0)
        elif method == 'correlation':
            return self._create_correlation_graph(features, threshold=0.7)
        else:
            raise ValueError(f"Unknown graph construction method: {method}")
    
    def _create_radius_graph(self, features: torch.Tensor, radius: float) -> torch.Tensor:
        """Create a radius-based graph from features."""
        from sklearn.neighbors import NearestNeighbors
        
        # Use radius neighbors
        nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(features.numpy())
        distances, indices = nbrs.radius_neighbors(features.numpy())
        
        # Create edge list
        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:  # No self-loops
                    edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _create_correlation_graph(self, features: torch.Tensor, threshold: float) -> torch.Tensor:
        """Create a correlation-based graph from features."""
        # Compute pairwise correlations
        corr_matrix = torch.corrcoef(features)
        
        # Create edges where correlation > threshold
        edge_indices = torch.where(corr_matrix > threshold)
        
        # Remove self-loops
        mask = edge_indices[0] != edge_indices[1]
        edge_index = torch.stack([edge_indices[0][mask], edge_indices[1][mask]])
        
        return edge_index
    
    def _generate_mock_graph(self) -> Data:
        """Generate a realistic mock graph for demonstration."""
        n_cells = self._metadata["n_cells"]
        n_genes = self._metadata["n_genes"]
        n_classes = self._metadata["n_classes"]
        
        # Generate realistic gene expression data with biological structure
        np.random.seed(42)
        
        # Create cluster-based expression patterns
        cluster_centers = np.random.randn(n_classes, n_genes) * 2
        
        # Assign cells to clusters
        cells_per_cluster = n_cells // n_classes
        x_list = []
        y_list = []
        
        for cluster_id in range(n_classes):
            start_idx = cluster_id * cells_per_cluster
            end_idx = start_idx + cells_per_cluster
            if cluster_id == n_classes - 1:  # Handle remainder
                end_idx = n_cells
            
            n_cluster_cells = end_idx - start_idx
            
            # Generate cells around cluster center
            cluster_data = np.random.multivariate_normal(
                cluster_centers[cluster_id],
                np.eye(n_genes) * 0.5,
                size=n_cluster_cells
            )
            
            # Add dropout (set some values to 0 - common in scRNA-seq)
            dropout_mask = np.random.binomial(1, 0.3, size=cluster_data.shape)
            cluster_data = cluster_data * dropout_mask
            
            # Ensure non-negative values (like gene expression)
            cluster_data = np.maximum(cluster_data, 0)
            
            x_list.append(cluster_data)
            y_list.extend([cluster_id] * n_cluster_cells)
        
        x = torch.FloatTensor(np.vstack(x_list))
        y = torch.LongTensor(y_list)
        
        # Create a biologically-informed k-NN graph
        k = min(15, n_cells - 1)
        edge_index = self._create_knn_graph(x, k)
        
        # Create stratified train/val/test splits (preserve class distribution)
        train_mask, val_mask, test_mask = self._create_stratified_splits(y, n_cells)
        
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
        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features.numpy())
        distances, indices = nbrs.kneighbors(features.numpy())
        
        # Create edge list (excluding self-connections)
        edge_list = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip self (index 0)
                neighbor_idx = indices[i][j]
                # Add both directions for undirected graph
                edge_list.append([i, neighbor_idx])
                edge_list.append([neighbor_idx, i])
        
        # Remove duplicates and convert to tensor
        edge_set = set(map(tuple, edge_list))
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
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
    
    def _create_stratified_splits(self, labels: torch.Tensor, n_cells: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create stratified train/validation/test splits preserving class distribution."""
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(n_cells)
        labels_np = labels.numpy()
        
        # First split: train vs (val + test)
        train_idx, temp_idx, _, temp_labels = train_test_split(
            indices, labels_np, test_size=0.4, stratify=labels_np, random_state=42
        )
        
        # Second split: val vs test
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
        )
        
        # Create masks
        train_mask = torch.zeros(n_cells, dtype=torch.bool)
        val_mask = torch.zeros(n_cells, dtype=torch.bool)
        test_mask = torch.zeros(n_cells, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
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
    
    def get_node_loader(self, mask_name: str = 'train', batch_size: int = 1024) -> DataLoader:
        """Get a node-level data loader for large graphs.
        
        Args:
            mask_name: Which mask to use ('train', 'val', 'test')
            batch_size: Number of nodes per batch
            
        Returns:
            DataLoader for node-level batching
        """
        from torch_geometric.loader import NeighborLoader
        
        # Get the mask
        if mask_name == 'train':
            mask = self.data.train_mask
        elif mask_name == 'val':
            mask = self.data.val_mask
        elif mask_name == 'test':
            mask = self.data.test_mask
        else:
            raise ValueError(f"Unknown mask: {mask_name}")
        
        # Get node indices for this split
        node_indices = torch.where(mask)[0]
        
        # Create neighbor loader for scalable training
        loader = NeighborLoader(
            self.data,
            num_neighbors=[15, 10],  # 2-hop neighborhood
            batch_size=batch_size,
            input_nodes=node_indices,
            shuffle=(mask_name == 'train')
        )
        
        return loader
    
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
    
    def save_processed_data(self, data: Data, filename: Optional[str] = None):
        """Save processed data to disk."""
        if filename is None:
            filename = self.processed_paths[0]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save data
        torch.save(self.collate([data]), filename)
    
    def load_processed_data(self, filename: Optional[str] = None) -> Tuple[Data, Any]:
        """Load processed data from disk."""
        if filename is None:
            filename = self.processed_paths[0]
        
        return torch.load(filename)
    
    @staticmethod
    def from_adata(adata, task: str = "cell_type_prediction", **kwargs) -> 'SCGraphDataset':
        """Create SCGraphDataset directly from AnnData object.
        
        Args:
            adata: AnnData object
            task: Task type
            **kwargs: Additional arguments for SCGraphDataset
            
        Returns:
            SCGraphDataset instance
        """
        # Create temporary directory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Save AnnData to temporary file
        temp_file = os.path.join(temp_dir, "temp_data.h5ad")
        adata.write_h5ad(temp_file)
        
        # Create dataset
        dataset = SCGraphDataset(
            name="temp_data",
            root=temp_dir,
            task=task,
            **kwargs
        )
        
        return dataset