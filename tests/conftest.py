"""Pytest configuration and shared fixtures for single-cell-graph-hub tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

# Configure test environment
os.environ["SCGRAPH_DEV_MODE"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_gene_expression() -> np.ndarray:
    """Create sample gene expression data."""
    np.random.seed(42)
    # 100 cells, 50 genes
    return np.random.lognormal(mean=1.0, sigma=1.5, size=(100, 50))


@pytest.fixture
def sample_cell_metadata() -> pd.DataFrame:
    """Create sample cell metadata."""
    np.random.seed(42)
    n_cells = 100
    
    cell_types = np.random.choice(
        ["T_cell", "B_cell", "NK_cell", "Monocyte", "DC"], 
        size=n_cells, 
        p=[0.3, 0.25, 0.15, 0.2, 0.1]
    )
    
    batches = np.random.choice(["batch_1", "batch_2", "batch_3"], size=n_cells)
    
    return pd.DataFrame({
        "cell_id": [f"cell_{i:03d}" for i in range(n_cells)],
        "cell_type": cell_types,
        "batch": batches,
        "total_counts": np.random.randint(1000, 10000, n_cells),
        "n_genes": np.random.randint(500, 3000, n_cells),
        "percent_mito": np.random.uniform(0, 20, n_cells),
    })


@pytest.fixture
def sample_gene_metadata() -> pd.DataFrame:
    """Create sample gene metadata."""
    n_genes = 50
    
    gene_types = np.random.choice(
        ["protein_coding", "lncRNA", "miRNA", "pseudogene"], 
        size=n_genes,
        p=[0.7, 0.15, 0.1, 0.05]
    )
    
    return pd.DataFrame({
        "gene_id": [f"ENSG{i:08d}" for i in range(n_genes)],
        "gene_name": [f"Gene_{i}" for i in range(n_genes)],
        "gene_type": gene_types,
        "chromosome": np.random.choice([f"chr{i}" for i in range(1, 23)], n_genes),
        "highly_variable": np.random.choice([True, False], n_genes, p=[0.3, 0.7]),
    })


@pytest.fixture
def sample_spatial_coordinates() -> np.ndarray:
    """Create sample spatial coordinates for spatial transcriptomics."""
    np.random.seed(42)
    # 100 cells with x, y coordinates
    return np.random.uniform(0, 1000, size=(100, 2))


@pytest.fixture
def sample_pyg_data() -> Data:
    """Create sample PyTorch Geometric data object."""
    torch.manual_seed(42)
    
    # 100 nodes (cells), 50 features (genes)
    x = torch.randn(100, 50)
    
    # Create a random graph structure
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 1)  # Edge weights
    
    # Node labels (cell types)  
    y = torch.randint(0, 5, (100,))
    
    # Batch information
    batch = torch.zeros(100, dtype=torch.long)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        batch=batch
    )


@pytest.fixture
def mock_h5ad_file(temp_data_dir: Path, sample_gene_expression: np.ndarray, 
                   sample_cell_metadata: pd.DataFrame) -> Path:
    """Create a mock H5AD file for testing."""
    pytest.importorskip("scanpy", reason="scanpy required for H5AD file creation")
    
    import scanpy as sc
    import anndata as ad
    
    # Create AnnData object
    adata = ad.AnnData(X=sample_gene_expression)
    adata.obs = sample_cell_metadata.set_index("cell_id")
    adata.var_names = [f"Gene_{i}" for i in range(sample_gene_expression.shape[1])]
    
    # Add some sample obsm data
    sc.pp.pca(adata, n_comps=10)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    # Save to temporary file
    h5ad_path = temp_data_dir / "sample_data.h5ad"
    adata.write_h5ad(h5ad_path)
    
    return h5ad_path


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def test_config() -> dict:
    """Provide test configuration."""
    return {
        "batch_size": 16,
        "max_epochs": 5,
        "learning_rate": 0.01,
        "patience": 2,
        "seed": 42,
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark tests in unit/ directory
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark tests requiring GPU
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)


def pytest_runtest_setup(item):
    """Skip GPU tests if CUDA is not available."""
    if "gpu" in [mark.name for mark in item.iter_markers()]:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")


# Performance testing utilities
@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "rounds": 3,
        "warmup_rounds": 1,
        "min_time": 0.1,
    }


# Mock external services
@pytest.fixture
def mock_api_client():
    """Mock API client for testing external service integration."""
    from unittest.mock import Mock
    
    mock_client = Mock()
    mock_client.download_dataset.return_value = "mock_dataset_path"
    mock_client.upload_results.return_value = {"status": "success"}
    
    return mock_client


# Database fixtures (if needed)
@pytest.fixture
def mock_database():
    """Mock database for testing."""
    from unittest.mock import Mock
    
    db = Mock()
    db.query.return_value = []
    db.insert.return_value = True
    db.update.return_value = True
    db.delete.return_value = True
    
    return db


# Logging configuration for tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for tests."""
    import logging
    
    # Set logging level based on environment
    if os.getenv("PYTEST_VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)


# Clean up test artifacts
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield
    
    # Clean up any temporary files created during tests
    test_files = [
        "test_output.h5ad",
        "test_model.pth",
        "test_results.json",
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)