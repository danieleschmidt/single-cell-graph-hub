"""
Basic functionality tests for Single-Cell Graph Hub.

These tests verify core functionality without requiring heavy dependencies.
"""

import sys
import os
import tempfile
import shutil
import pytest
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import scgraph_hub


class TestBasicImports:
    """Test basic package imports and structure."""
    
    def test_package_version(self):
        """Test package version is available."""
        assert hasattr(scgraph_hub, '__version__')
        assert scgraph_hub.__version__ == "0.1.0"
    
    def test_basic_imports(self):
        """Test that basic imports work."""
        assert hasattr(scgraph_hub, 'DatasetCatalog')
        assert hasattr(scgraph_hub, 'get_default_catalog')
        assert hasattr(scgraph_hub, 'SimpleSCGraphDataset')
        assert hasattr(scgraph_hub, 'SimpleSCGraphData')
        assert hasattr(scgraph_hub, 'simple_quick_start')
    
    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        expected_exports = {
            'DatasetCatalog',
            'get_default_catalog', 
            'SimpleSCGraphDataset',
            'SimpleSCGraphData',
            'check_dependencies',
            'validate_dataset_config',
            'quick_start',
            'simple_quick_start'
        }
        assert set(scgraph_hub.__all__) == expected_exports


class TestDatasetCatalog:
    """Test dataset catalog functionality."""
    
    def test_catalog_creation(self):
        """Test catalog can be created."""
        catalog = scgraph_hub.get_default_catalog()
        assert isinstance(catalog, scgraph_hub.DatasetCatalog)
    
    def test_list_datasets(self):
        """Test listing datasets."""
        catalog = scgraph_hub.get_default_catalog()
        datasets = catalog.list_datasets()
        
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert 'pbmc_10k' in datasets
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        catalog = scgraph_hub.get_default_catalog()
        info = catalog.get_info('pbmc_10k')
        
        assert isinstance(info, dict)
        assert info['name'] == 'pbmc_10k'
        assert 'n_cells' in info
        assert 'n_genes' in info
        assert 'modality' in info
    
    def test_filter_datasets(self):
        """Test dataset filtering."""
        catalog = scgraph_hub.get_default_catalog()
        
        # Filter by organism
        human_datasets = catalog.filter(organism="human")
        assert isinstance(human_datasets, list)
        assert 'pbmc_10k' in human_datasets
        
        # Filter by modality
        rna_datasets = catalog.filter(modality="scRNA-seq")
        assert isinstance(rna_datasets, list)
        assert len(rna_datasets) > 0
        
        # Filter by size
        large_datasets = catalog.filter(min_cells=50000)
        assert isinstance(large_datasets, list)
    
    def test_search_datasets(self):
        """Test dataset search."""
        catalog = scgraph_hub.get_default_catalog()
        
        brain_results = catalog.search("brain")
        assert isinstance(brain_results, list)
        
        pbmc_results = catalog.search("pbmc")
        assert isinstance(pbmc_results, list)
        assert any("pbmc" in name for name in pbmc_results)
    
    def test_get_recommendations(self):
        """Test dataset recommendations."""
        catalog = scgraph_hub.get_default_catalog()
        
        recommendations = catalog.get_recommendations("pbmc_10k")
        assert isinstance(recommendations, list)
        assert "pbmc_10k" not in recommendations  # Should not recommend itself
    
    def test_summary_stats(self):
        """Test catalog summary statistics."""
        catalog = scgraph_hub.get_default_catalog()
        stats = catalog.get_summary_stats()
        
        assert isinstance(stats, dict)
        assert 'total_datasets' in stats
        assert 'total_cells' in stats
        assert stats['total_datasets'] > 0
    
    def test_tasks_summary(self):
        """Test tasks summary."""
        catalog = scgraph_hub.get_default_catalog()
        tasks = catalog.get_tasks_summary()
        
        assert isinstance(tasks, dict)
        assert 'cell_type_prediction' in tasks
        assert isinstance(tasks['cell_type_prediction'], list)


class TestSimpleDataset:
    """Test simple dataset functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_dataset_creation(self):
        """Test simple dataset can be created."""
        dataset = scgraph_hub.SimpleSCGraphDataset(
            name="test_dataset",
            root=self.temp_dir
        )
        
        assert dataset.name == "test_dataset"
        assert dataset.task == "cell_type_prediction"  # default
        assert dataset.root == self.temp_dir
    
    def test_dataset_info(self):
        """Test dataset info method."""
        dataset = scgraph_hub.SimpleSCGraphDataset(
            name="pbmc_10k",
            root=self.temp_dir
        )
        
        info = dataset.info()
        assert isinstance(info, dict)
        assert info['name'] == 'pbmc_10k'
        assert 'num_cells' in info
        assert 'num_genes' in info
        assert 'task' in info
    
    def test_dataset_properties(self):
        """Test dataset properties."""
        dataset = scgraph_hub.SimpleSCGraphDataset(
            name="pbmc_10k",
            root=self.temp_dir
        )
        
        # These should work even without data
        assert isinstance(dataset.num_nodes, int)
        assert isinstance(dataset.num_edges, int)
        assert isinstance(dataset.num_node_features, int)
        assert isinstance(dataset.num_classes, int)
    
    def test_simple_quick_start(self):
        """Test simple quick start function."""
        dataset = scgraph_hub.simple_quick_start(
            dataset_name="pbmc_10k",
            root=self.temp_dir
        )
        
        assert isinstance(dataset, scgraph_hub.SimpleSCGraphDataset)
        assert dataset.name == "pbmc_10k"


class TestSimpleSCGraphData:
    """Test simple graph data structure."""
    
    def test_data_creation(self):
        """Test data structure can be created."""
        data = scgraph_hub.SimpleSCGraphData()
        assert data is not None
    
    def test_data_properties(self):
        """Test data structure properties."""
        data = scgraph_hub.SimpleSCGraphData()
        
        # Should have properties even with None values
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'y')
        assert hasattr(data, 'train_mask')
        assert hasattr(data, 'val_mask')
        assert hasattr(data, 'test_mask')
    
    def test_data_num_properties(self):
        """Test numerical properties."""
        data = scgraph_hub.SimpleSCGraphData()
        
        # Should return 0 for empty data
        assert data.num_nodes == 0
        assert data.num_edges == 0  
        assert data.num_node_features == 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_unknown_dataset(self):
        """Test handling of unknown dataset."""
        catalog = scgraph_hub.get_default_catalog()
        
        with pytest.raises(KeyError):
            catalog.get_info("nonexistent_dataset")
    
    def test_empty_search(self):
        """Test empty search results."""
        catalog = scgraph_hub.get_default_catalog()
        results = catalog.search("nonexistent_search_term")
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_invalid_filter(self):
        """Test invalid filter parameters."""
        catalog = scgraph_hub.get_default_catalog()
        
        # Should return empty list for impossible conditions
        results = catalog.filter(min_cells=1000000, max_cells=10)
        assert isinstance(results, list)
        assert len(results) == 0


@pytest.mark.integration
class TestIntegrationWorkflow:
    """Integration tests for complete workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete workflow from catalog to dataset."""
        # 1. Get catalog
        catalog = scgraph_hub.get_default_catalog()
        
        # 2. Find datasets
        human_datasets = catalog.filter(organism="human", min_cells=5000)
        assert len(human_datasets) > 0
        
        # 3. Get dataset info
        dataset_name = human_datasets[0]
        info = catalog.get_info(dataset_name)
        assert info['organism'] == 'human'
        
        # 4. Load dataset
        dataset = scgraph_hub.simple_quick_start(
            dataset_name=dataset_name,
            root=self.temp_dir
        )
        
        # 5. Verify dataset
        dataset_info = dataset.info()
        assert dataset_info['name'] == dataset_name
        assert dataset_info['num_cells'] >= 5000  # Should match filter
    
    def test_dataset_persistence(self):
        """Test that dataset data persists between loads."""
        dataset_name = "test_persistence"
        
        # Create first instance
        dataset1 = scgraph_hub.SimpleSCGraphDataset(
            name=dataset_name,
            root=self.temp_dir
        )
        info1 = dataset1.info()
        
        # Create second instance (should load from saved data)
        dataset2 = scgraph_hub.SimpleSCGraphDataset(
            name=dataset_name,
            root=self.temp_dir
        )
        info2 = dataset2.info()
        
        # Should have same basic info
        assert info1['name'] == info2['name']
        assert info1['task'] == info2['task']


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])