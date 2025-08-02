"""Sample data fixtures for testing."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data


def create_sample_expression_data(
    n_cells: int = 100,
    n_genes: int = 50,
    seed: int = 42
) -> np.ndarray:
    """Create sample gene expression data with realistic properties.
    
    Args:
        n_cells: Number of cells
        n_genes: Number of genes
        seed: Random seed for reproducibility
        
    Returns:
        Gene expression matrix (cells x genes)
    """
    np.random.seed(seed)
    
    # Create realistic gene expression with log-normal distribution
    # Some genes are highly expressed, others lowly expressed
    base_expression = np.random.lognormal(mean=1.0, sigma=1.5, size=(n_cells, n_genes))
    
    # Add some structure to make it more realistic
    # Create cell type specific expression patterns
    n_cell_types = 5
    cells_per_type = n_cells // n_cell_types
    
    for i in range(n_cell_types):
        start_idx = i * cells_per_type
        end_idx = start_idx + cells_per_type
        
        # Create type-specific marker genes
        marker_genes = np.random.choice(n_genes, size=n_genes//10, replace=False)
        base_expression[start_idx:end_idx, marker_genes] *= (2 + i)
    
    return base_expression


def create_sample_metadata(
    n_cells: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """Create sample cell metadata.
    
    Args:
        n_cells: Number of cells
        seed: Random seed
        
    Returns:
        DataFrame with cell metadata
    """
    np.random.seed(seed)
    
    # Cell types with realistic proportions
    cell_types = ["T_cell", "B_cell", "NK_cell", "Monocyte", "Dendritic_cell"]
    cell_type_probs = [0.35, 0.25, 0.15, 0.20, 0.05]
    
    assigned_types = np.random.choice(
        cell_types, 
        size=n_cells, 
        p=cell_type_probs
    )
    
    # Batch information
    batches = np.random.choice(["Batch_A", "Batch_B", "Batch_C"], size=n_cells)
    
    # QC metrics
    total_counts = np.random.lognormal(mean=8.5, sigma=0.5, size=n_cells).astype(int)
    n_genes_detected = np.random.poisson(lam=1500, size=n_cells)
    percent_mito = np.random.beta(a=2, b=10, size=n_cells) * 100
    
    # Donor information
    donor_ids = np.random.choice(["Donor_1", "Donor_2", "Donor_3"], size=n_cells)
    
    # Stimulation condition (for some experimental designs)
    conditions = np.random.choice(["Control", "Stimulated"], size=n_cells, p=[0.6, 0.4])
    
    # Age and sex (for human samples)
    ages = np.random.randint(20, 80, size=n_cells)
    sexes = np.random.choice(["M", "F"], size=n_cells)
    
    return pd.DataFrame({
        "cell_id": [f"Cell_{i:05d}" for i in range(n_cells)],
        "cell_type": assigned_types,
        "batch": batches,
        "total_counts": total_counts,
        "n_genes": n_genes_detected,
        "percent_mito": percent_mito,
        "donor_id": donor_ids,
        "condition": conditions,
        "age": ages,
        "sex": sexes,
        "pass_qc": (percent_mito < 20) & (n_genes_detected > 500)
    })


def create_sample_gene_metadata(
    n_genes: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """Create sample gene metadata.
    
    Args:
        n_genes: Number of genes
        seed: Random seed
        
    Returns:
        DataFrame with gene metadata
    """
    np.random.seed(seed)
    
    # Gene types with realistic proportions
    gene_types = ["protein_coding", "lncRNA", "miRNA", "pseudogene", "misc_RNA"]
    gene_type_probs = [0.75, 0.15, 0.05, 0.03, 0.02]
    
    assigned_types = np.random.choice(
        gene_types, 
        size=n_genes, 
        p=gene_type_probs
    )
    
    # Chromosomes
    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    assigned_chromosomes = np.random.choice(chromosomes, size=n_genes)
    
    # Gene positions
    start_positions = np.random.randint(1000, 200000000, size=n_genes)
    end_positions = start_positions + np.random.randint(1000, 50000, size=n_genes)
    
    # Highly variable genes
    highly_variable = np.random.choice([True, False], size=n_genes, p=[0.3, 0.7])
    
    return pd.DataFrame({
        "gene_id": [f"ENSG{i:011d}" for i in range(n_genes)],
        "gene_symbol": [f"GENE{i}" for i in range(n_genes)],
        "gene_type": assigned_types,
        "chromosome": assigned_chromosomes,
        "start_position": start_positions,
        "end_position": end_positions,
        "strand": np.random.choice(["+", "-"], size=n_genes),
        "highly_variable": highly_variable,
        "mean_expression": np.random.lognormal(1, 1, size=n_genes),
        "variance": np.random.lognormal(1, 1, size=n_genes)
    })


def create_sample_spatial_coordinates(
    n_cells: int = 100,
    tissue_shape: str = "circular",
    seed: int = 42
) -> np.ndarray:
    """Create sample spatial coordinates for spatial transcriptomics.
    
    Args:
        n_cells: Number of cells
        tissue_shape: Shape of tissue ("circular", "rectangular", "irregular")
        seed: Random seed
        
    Returns:
        Spatial coordinates (n_cells x 2)
    """
    np.random.seed(seed)
    
    if tissue_shape == "circular":
        # Create circular tissue layout
        angles = np.random.uniform(0, 2*np.pi, n_cells)
        radii = np.random.uniform(0, 500, n_cells)
        
        x = radii * np.cos(angles) + 500  # Center at (500, 500)
        y = radii * np.sin(angles) + 500
        
    elif tissue_shape == "rectangular":
        # Create rectangular tissue layout
        x = np.random.uniform(0, 1000, n_cells)
        y = np.random.uniform(0, 800, n_cells)
        
    elif tissue_shape == "irregular":
        # Create irregular tissue shape with clusters
        n_clusters = 3
        cluster_centers = np.random.uniform(100, 900, size=(n_clusters, 2))
        
        # Assign cells to clusters
        cluster_assignments = np.random.choice(n_clusters, size=n_cells)
        
        coordinates = []
        for i in range(n_cells):
            cluster_id = cluster_assignments[i]
            center = cluster_centers[cluster_id]
            
            # Add noise around cluster center
            noise = np.random.normal(0, 50, size=2)
            coord = center + noise
            coordinates.append(coord)
        
        coordinates = np.array(coordinates)
        x, y = coordinates[:, 0], coordinates[:, 1]
    
    else:
        raise ValueError(f"Unknown tissue shape: {tissue_shape}")
    
    return np.column_stack([x, y])


def create_sample_graph_data(
    n_cells: int = 100,
    n_genes: int = 50,
    graph_type: str = "knn",
    k: int = 15,
    seed: int = 42
) -> Data:
    """Create sample PyTorch Geometric Data object.
    
    Args:
        n_cells: Number of cells (nodes)
        n_genes: Number of genes (features)
        graph_type: Type of graph ("knn", "radius", "random")
        k: Number of neighbors for knn graph
        seed: Random seed
        
    Returns:
        PyTorch Geometric Data object
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create node features (gene expression)
    expression_data = create_sample_expression_data(n_cells, n_genes, seed)
    x = torch.tensor(expression_data, dtype=torch.float32)
    
    # Create graph structure
    if graph_type == "knn":
        # Create k-NN graph based on expression similarity
        from sklearn.neighbors import kneighbors_graph
        
        # Compute k-NN graph
        knn_graph = kneighbors_graph(
            expression_data, 
            n_neighbors=k, 
            mode='connectivity',
            include_self=False
        )
        
        # Convert to edge_index format
        edge_indices = np.array(knn_graph.nonzero())
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        
    elif graph_type == "radius":
        # Create radius graph (mock implementation)
        edge_list = []
        for i in range(n_cells):
            # Connect to random neighbors within "radius"
            n_neighbors = np.random.randint(3, min(k, n_cells-1))
            neighbors = np.random.choice(
                [j for j in range(n_cells) if j != i], 
                size=n_neighbors, 
                replace=False
            )
            
            for neighbor in neighbors:
                edge_list.append([i, neighbor])
        
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        
    elif graph_type == "random":
        # Create random graph
        n_edges = min(n_cells * k, n_cells * (n_cells - 1) // 2)
        edge_list = []
        
        while len(edge_list) < n_edges:
            i, j = np.random.choice(n_cells, size=2, replace=False)
            if [i, j] not in edge_list and [j, i] not in edge_list:
                edge_list.append([i, j])
        
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Create edge attributes (distances/weights)
    n_edges = edge_index.shape[1]
    edge_attr = torch.randn(n_edges, 1)  # Random edge weights
    
    # Create node labels (cell types)
    metadata = create_sample_metadata(n_cells, seed)
    cell_type_mapping = {
        cell_type: idx 
        for idx, cell_type in enumerate(metadata["cell_type"].unique())
    }
    
    y = torch.tensor([
        cell_type_mapping[cell_type] 
        for cell_type in metadata["cell_type"]
    ], dtype=torch.long)
    
    # Create batch information (single batch for simplicity)
    batch = torch.zeros(n_cells, dtype=torch.long)
    
    # Add positional information if needed
    pos = None
    if "spatial" in graph_type.lower():
        spatial_coords = create_sample_spatial_coordinates(n_cells, seed=seed)
        pos = torch.tensor(spatial_coords, dtype=torch.float32)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        batch=batch,
        pos=pos,
        num_nodes=n_cells
    )


def create_sample_dataset_configs() -> List[Dict]:
    """Create sample dataset configurations for testing.
    
    Returns:
        List of dataset configuration dictionaries
    """
    return [
        {
            "name": "test_pbmc",
            "description": "Test PBMC dataset",
            "organism": "human",
            "tissue": "peripheral blood",
            "n_cells": 1000,
            "n_genes": 2000,
            "modality": "scRNA-seq",
            "has_spatial": False,
            "cell_types": ["T_cell", "B_cell", "NK_cell", "Monocyte"],
            "batches": ["Batch1", "Batch2"],
            "graph_method": "knn",
            "k_neighbors": 15
        },
        {
            "name": "test_brain_spatial",
            "description": "Test brain spatial dataset", 
            "organism": "mouse",
            "tissue": "brain",
            "n_cells": 500,
            "n_genes": 1500,
            "modality": "spatial_transcriptomics",
            "has_spatial": True,
            "cell_types": ["Neuron", "Astrocyte", "Microglia", "Oligodendrocyte"],
            "batches": ["Batch1"],
            "graph_method": "spatial",
            "spatial_radius": 100
        },
        {
            "name": "test_multiome",
            "description": "Test multiome dataset",
            "organism": "human",
            "tissue": "bone marrow",
            "n_cells": 800,
            "n_genes": 2500,
            "modality": "multiome",
            "has_spatial": False,
            "cell_types": ["HSC", "MPP", "GMP", "MEP"],
            "batches": ["Batch1", "Batch2", "Batch3"],
            "graph_method": "multimodal",
            "modalities": ["RNA", "ATAC"]
        }
    ]


def save_sample_data_to_files(
    output_dir: Path,
    n_cells: int = 100,
    n_genes: int = 50,
    seed: int = 42
) -> Dict[str, Path]:
    """Save sample data to files for testing file I/O.
    
    Args:
        output_dir: Directory to save files
        n_cells: Number of cells
        n_genes: Number of genes
        seed: Random seed
        
    Returns:
        Dictionary mapping file types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    expression_data = create_sample_expression_data(n_cells, n_genes, seed)
    cell_metadata = create_sample_metadata(n_cells, seed)
    gene_metadata = create_sample_gene_metadata(n_genes, seed)
    spatial_coords = create_sample_spatial_coordinates(n_cells, seed=seed)
    
    file_paths = {}
    
    # Save as CSV files
    expression_df = pd.DataFrame(
        expression_data,
        index=cell_metadata["cell_id"],
        columns=gene_metadata["gene_symbol"]
    )
    
    expression_path = output_dir / "expression_data.csv"
    cell_metadata_path = output_dir / "cell_metadata.csv"
    gene_metadata_path = output_dir / "gene_metadata.csv"
    spatial_coords_path = output_dir / "spatial_coordinates.csv"
    
    expression_df.to_csv(expression_path)
    cell_metadata.to_csv(cell_metadata_path, index=False)
    gene_metadata.to_csv(gene_metadata_path, index=False)
    
    spatial_df = pd.DataFrame(
        spatial_coords,
        columns=["x", "y"],
        index=cell_metadata["cell_id"]
    )
    spatial_df.to_csv(spatial_coords_path)
    
    file_paths.update({
        "expression": expression_path,
        "cell_metadata": cell_metadata_path,
        "gene_metadata": gene_metadata_path,
        "spatial_coordinates": spatial_coords_path
    })
    
    # Save as numpy arrays
    np.save(output_dir / "expression_array.npy", expression_data)
    np.save(output_dir / "spatial_array.npy", spatial_coords)
    
    file_paths.update({
        "expression_npy": output_dir / "expression_array.npy",
        "spatial_npy": output_dir / "spatial_array.npy"
    })
    
    # Save PyTorch tensors
    graph_data = create_sample_graph_data(n_cells, n_genes, seed=seed)
    torch.save(graph_data, output_dir / "graph_data.pt")
    
    file_paths["graph_data"] = output_dir / "graph_data.pt"
    
    return file_paths