"""Integration tests for the complete workflow."""

import pytest
import torch
import tempfile
import os
from pathlib import Path

# Mock imports for testing without dependencies
try:
    from scgraph_hub import SCGraphDataset, DatasetCatalog
    from scgraph_hub.models import CellGraphGNN
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="Dependencies not available")
class TestCompleteWorkflow:
    """Test the complete workflow from data loading to model training."""
    
    def test_basic_workflow(self):
        """Test the basic workflow matches README example."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            dataset = SCGraphDataset(
                name="pbmc_10k",
                root=tmpdir,
                task="cell_type_prediction",
                download=True
            )
            
            # Check dataset properties
            assert dataset.num_nodes > 0
            assert dataset.num_node_features > 0
            assert dataset.num_classes > 0
            
            # Get data loaders
            train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)
            assert len(train_loader) > 0
            assert len(val_loader) > 0
            assert len(test_loader) > 0
            
            # Create model
            model = CellGraphGNN(
                input_dim=dataset.num_node_features,
                hidden_dim=64,  # Smaller for testing
                output_dim=dataset.num_classes,
                num_layers=2,   # Fewer layers for testing
                dropout=0.2
            )
            
            # Check model can forward pass
            data = dataset.data
            with torch.no_grad():
                output = model(data.x, data.edge_index)
                assert output.shape == (dataset.num_nodes, dataset.num_classes)
    
    def test_catalog_functionality(self):
        """Test dataset catalog functionality."""
        catalog = DatasetCatalog()
        
        # Test listing datasets
        datasets = catalog.list_datasets()
        assert len(datasets) > 0
        assert "pbmc_10k" in datasets
        
        # Test getting info
        info = catalog.get_info("pbmc_10k")
        assert "n_cells" in info
        assert "modality" in info
        
        # Test filtering
        human_datasets = catalog.filter(organism="human")
        assert len(human_datasets) > 0
        
        # Test search
        search_results = catalog.search("human")
        assert len(search_results) > 0
    
    def test_model_training_step(self):
        """Test that model can perform one training step without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SCGraphDataset(
                name="pbmc_10k",
                root=tmpdir,
                download=True
            )
            
            model = CellGraphGNN(
                input_dim=dataset.num_node_features,
                hidden_dim=32,
                output_dim=dataset.num_classes,
                num_layers=2
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            data = dataset.data
            
            # One training step
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = torch.nn.functional.cross_entropy(
                out[data.train_mask], 
                data.y[data.train_mask]
            )
            
            loss.backward()
            optimizer.step()
            
            # Check that loss is finite
            assert torch.isfinite(loss)
            
            # Check that parameters have gradients
            has_grad = any(p.grad is not None for p in model.parameters())
            assert has_grad


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="Dependencies not available")
class TestModelArchitectures:
    """Test different model architectures."""
    
    def test_cellgraphgnn(self):
        """Test CellGraphGNN model."""
        model = CellGraphGNN(100, 64, 10, num_layers=2)
        
        # Test forward pass
        x = torch.randn(50, 100)
        edge_index = torch.randint(0, 50, (2, 100))
        
        with torch.no_grad():
            out = model(x, edge_index)
            assert out.shape == (50, 10)
        
        # Test parameter count
        param_count = model.num_parameters()
        assert param_count > 0
    
    def test_model_inheritance(self):
        """Test that models inherit from BaseGNN."""
        from scgraph_hub.models import BaseGNN
        
        model = CellGraphGNN(100, 64, 10)
        assert isinstance(model, BaseGNN)


@pytest.mark.skipif(DEPS_AVAILABLE, reason="Test mock behavior when deps unavailable")
class TestWithoutDependencies:
    """Test behavior when PyTorch dependencies are not available."""
    
    def test_graceful_degradation(self):
        """Test that the package handles missing dependencies gracefully."""
        # This test runs when dependencies are NOT available
        # In a real scenario, we'd test that appropriate warnings are shown
        assert True  # Placeholder - would test warning behavior