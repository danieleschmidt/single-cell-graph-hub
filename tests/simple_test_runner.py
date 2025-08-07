"""
Simple test runner for basic functionality without external dependencies.
"""

import sys
import os
import tempfile
import shutil
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import scgraph_hub


class SimpleTestRunner:
    """Simple test runner without external dependencies."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def run_test(self, test_name, test_func):
        """Run a single test."""
        self.tests_run += 1
        try:
            print(f"Running {test_name}...", end=" ")
            test_func()
            print("✅ PASS")
            self.tests_passed += 1
        except Exception as e:
            print("❌ FAIL")
            print(f"  Error: {str(e)}")
            self.tests_failed += 1
            self.failures.append((test_name, str(e)))
    
    def run_test_class(self, test_class):
        """Run all tests in a test class."""
        instance = test_class()
        
        # Run setup if available
        if hasattr(instance, 'setup_method'):
            instance.setup_method()
        
        # Find and run test methods
        for attr_name in dir(instance):
            if attr_name.startswith('test_'):
                test_method = getattr(instance, attr_name)
                self.run_test(f"{test_class.__name__}.{attr_name}", test_method)
        
        # Run teardown if available
        if hasattr(instance, 'teardown_method'):
            instance.teardown_method()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print("\nFailures:")
            for test_name, error in self.failures:
                print(f"  {test_name}: {error}")
        
        print("="*60)
        
        return self.tests_failed == 0


# Test Classes
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
        actual_exports = set(scgraph_hub.__all__)
        missing = expected_exports - actual_exports
        extra = actual_exports - expected_exports
        
        if missing:
            raise AssertionError(f"Missing exports: {missing}")
        if extra:
            print(f"  Warning: Extra exports: {extra}")


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
    
    def test_search_datasets(self):
        """Test dataset search."""
        catalog = scgraph_hub.get_default_catalog()
        
        brain_results = catalog.search("brain")
        assert isinstance(brain_results, list)
        
        pbmc_results = catalog.search("pbmc")
        assert isinstance(pbmc_results, list)
        assert any("pbmc" in name for name in pbmc_results)
    
    def test_summary_stats(self):
        """Test catalog summary statistics."""
        catalog = scgraph_hub.get_default_catalog()
        stats = catalog.get_summary_stats()
        
        assert isinstance(stats, dict)
        assert 'total_datasets' in stats
        assert 'total_cells' in stats
        assert stats['total_datasets'] > 0


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
    
    def test_simple_quick_start(self):
        """Test simple quick start function."""
        dataset = scgraph_hub.simple_quick_start(
            dataset_name="pbmc_10k",
            root=self.temp_dir
        )
        
        assert isinstance(dataset, scgraph_hub.SimpleSCGraphDataset)
        assert dataset.name == "pbmc_10k"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_unknown_dataset(self):
        """Test handling of unknown dataset."""
        catalog = scgraph_hub.get_default_catalog()
        
        try:
            catalog.get_info("nonexistent_dataset")
            raise AssertionError("Expected KeyError for nonexistent dataset")
        except KeyError:
            pass  # Expected
    
    def test_empty_search(self):
        """Test empty search results."""
        catalog = scgraph_hub.get_default_catalog()
        results = catalog.search("nonexistent_search_term")
        
        assert isinstance(results, list)
        assert len(results) == 0


def main():
    """Run all tests."""
    print("Single-Cell Graph Hub - Basic Functionality Tests")
    print("="*60)
    
    runner = SimpleTestRunner()
    
    # Run test classes
    test_classes = [
        TestBasicImports,
        TestDatasetCatalog,
        TestSimpleDataset,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 30)
        runner.run_test_class(test_class)
    
    # Print summary
    success = runner.print_summary()
    
    if success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())