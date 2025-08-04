"""Comprehensive pytest configuration and fixtures for Single-Cell Graph Hub tests."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any, List

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

# Configure test environment
os.environ["SCGRAPH_DEV_MODE"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"

# Import our modules for fixtures
try:
    from scgraph_hub.models import create_model
    from scgraph_hub.validation import DataValidator
    from scgraph_hub.monitoring import get_performance_monitor
    SCGRAPH_AVAILABLE = True
except ImportError:
    SCGRAPH_AVAILABLE = False


@pytest.fixture(scope="session")
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp(prefix="scgraph_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


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


# Additional fixtures for comprehensive testing
@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing."""
    n_nodes = 100
    n_features = 50
    n_edges = 300
    n_classes = 5
    
    # Node features
    x = torch.randn(n_nodes, n_features)
    
    # Random edges
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    
    # Random labels
    y = torch.randint(0, n_classes, (n_nodes,))
    
    # Train/val/test splits
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[:60] = True
    val_mask[60:80] = True
    test_mask[80:] = True
    
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )


@pytest.fixture
def large_graph_data():
    """Create large graph data for scalability testing."""
    n_nodes = 10000
    n_features = 100
    n_edges = 50000
    n_classes = 10
    
    # Node features with some structure
    x = torch.randn(n_nodes, n_features)
    
    # Create more realistic edge structure
    edge_list = []
    for i in range(n_nodes):
        # Connect to nearby nodes (simulate biological proximity)
        for j in range(max(0, i-5), min(n_nodes, i+6)):
            if i != j and np.random.random() < 0.3:
                edge_list.append([i, j])
    
    # Add some random long-range connections
    for _ in range(n_edges - len(edge_list)):
        i, j = np.random.choice(n_nodes, 2, replace=False)
        edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list[:n_edges], dtype=torch.long).t().contiguous()
    
    # Structured labels (clusters)
    y = torch.zeros(n_nodes, dtype=torch.long)
    cluster_size = n_nodes // n_classes
    for i in range(n_classes):
        start_idx = i * cluster_size
        end_idx = start_idx + cluster_size if i < n_classes - 1 else n_nodes
        y[start_idx:end_idx] = i
    
    # Train/val/test splits
    indices = torch.randperm(n_nodes)
    train_size = int(0.6 * n_nodes)
    val_size = int(0.2 * n_nodes)
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )


@pytest.fixture
def sample_model():
    """Create sample GNN model for testing."""
    if not SCGRAPH_AVAILABLE:
        pytest.skip("scgraph_hub not available")
    
    return create_model(
        'CellGraphGNN',
        input_dim=50,
        hidden_dim=64,
        output_dim=5,
        num_layers=2,
        dropout=0.1
    )


@pytest.fixture
def sample_dataset_config():
    """Sample dataset configuration for testing."""
    return {
        'name': 'test_dataset',
        'description': 'Test dataset for unit tests',
        'n_cells': 1000,
        'n_genes': 500,
        'n_classes': 5,
        'modality': 'scRNA-seq',
        'organism': 'human',
        'tissue': 'test_tissue',
        'has_spatial': False,
        'graph_method': 'knn',
        'size_mb': 10
    }


@pytest.fixture
def data_validator():
    """Create data validator instance."""
    if not SCGRAPH_AVAILABLE:
        pytest.skip("scgraph_hub not available")
    
    return DataValidator(strict_mode=False)


@pytest.fixture
def performance_monitor():
    """Get performance monitor instance."""
    if not SCGRAPH_AVAILABLE:
        pytest.skip("scgraph_hub not available")
    
    monitor = get_performance_monitor()
    monitor.reset()  # Clear any existing metrics
    return monitor


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        'models': ['CellGraphGNN', 'CellGraphSAGE'],
        'datasets': ['test_small', 'test_medium'],
        'tasks': ['cell_type_prediction'],
        'metrics': ['accuracy', 'f1_score'],
        'n_runs': 2,  # Small number for fast testing
        'train_config': {
            'epochs': 5,
            'learning_rate': 0.01,
            'batch_size': 32
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set to CPU for consistent testing
    torch.set_default_device('cpu')
    
    yield
    
    # Cleanup after test
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture
def mock_training_data():
    """Mock training data for testing training loops."""
    return {
        'losses': [1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28],
        'accuracies': [0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86],
        'val_losses': [1.6, 1.3, 1.1, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38],
        'val_accuracies': [0.25, 0.4, 0.55, 0.65, 0.7, 0.75, 0.77, 0.78, 0.79, 0.8]
    }


@pytest.fixture
def sample_preprocessing_config():
    """Sample preprocessing configuration."""
    return {
        'steps': [
            'filter_cells',
            'filter_genes',
            'normalize_total',
            'log1p',
            'highly_variable_genes',
            'scale',
            'pca',
            'neighbors'
        ],
        'parameters': {
            'filter_cells': {
                'min_genes': 200,
                'max_mt_pct': 20.0
            },
            'filter_genes': {
                'min_cells': 3
            },
            'normalize_total': {
                'target_sum': 10000
            },
            'highly_variable_genes': {
                'n_top_genes': 2000
            },
            'pca': {
                'n_comps': 50
            },
            'neighbors': {
                'n_neighbors': 15
            }
        }
    }


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[50, 100, 500])
def variable_node_count(request):
    """Fixture with variable node counts for parametrized tests."""
    return request.param


@pytest.fixture(params=['cpu', 'cuda'])
def device_param(request):
    """Fixture for testing on different devices."""
    device = request.param
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(device)


@pytest.fixture(params=['CellGraphGNN', 'CellGraphSAGE', 'CellGraphGAT'])
def model_type_param(request):
    """Fixture for testing different model types."""
    return request.param


@pytest.fixture
def cleanup_files():
    """Fixture to clean up created files after test."""
    created_files = []
    
    def add_file(filepath):
        created_files.append(Path(filepath))
    
    yield add_file
    
    # Cleanup
    for filepath in created_files:
        if filepath.exists():
            if filepath.is_file():
                filepath.unlink()
            elif filepath.is_dir():
                shutil.rmtree(filepath, ignore_errors=True)


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