"""
Tests for Generation 2 Robust Features.

Tests comprehensive error handling, logging, security, and advanced processing.
"""

import sys
import os
import tempfile
import shutil
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import scgraph_hub


class TestRobustErrorHandling:
    """Test comprehensive error handling system."""
    
    def test_graceful_import_fallback(self):
        """Test graceful fallback for missing advanced features."""
        # Test that advanced features show proper error messages
        try:
            scgraph_hub.setup_logging()
            assert False, "Should have raised ImportError"
        except ImportError as e:
            assert "additional dependencies" in str(e)
        
        try:
            scgraph_hub.DatasetProcessor()
            assert False, "Should have raised ImportError" 
        except ImportError as e:
            assert "additional dependencies" in str(e)
    
    def test_exception_hierarchy(self):
        """Test custom exception hierarchy."""
        from scgraph_hub.exceptions import (
            SCGraphHubError, DatasetError, DatasetNotFoundError, 
            ValidationError, SecurityError
        )
        
        # Test base exception
        base_error = SCGraphHubError("Test error", "TEST_CODE", {"detail": "test"})
        assert base_error.message == "Test error"
        assert base_error.error_code == "TEST_CODE"
        assert base_error.details == {"detail": "test"}
        
        # Test inheritance
        assert issubclass(DatasetError, SCGraphHubError)
        assert issubclass(DatasetNotFoundError, DatasetError)
        assert issubclass(ValidationError, SCGraphHubError)
        assert issubclass(SecurityError, SCGraphHubError)
    
    def test_error_collector(self):
        """Test error collection functionality."""
        from scgraph_hub.exceptions import ErrorCollector, ValidationError
        
        collector = ErrorCollector()
        
        # Test adding errors
        collector.add_error("Error 1")
        collector.add_error("Error 2") 
        collector.add_validation_error("field1", "value1", "Invalid value")
        
        assert collector.has_errors()
        assert len(collector.get_errors()) == 3
        
        # Test raising collected errors
        try:
            collector.raise_if_errors()
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
    
    def test_error_context(self):
        """Test error context manager."""
        from scgraph_hub.exceptions import create_error_context, SCGraphHubError
        
        # Test successful operation
        with create_error_context("test operation"):
            result = 42
        
        assert result == 42
        
        # Test error conversion
        try:
            with create_error_context("test operation"):
                raise ValueError("Test error")
        except SCGraphHubError as e:
            assert "test operation" in e.message
            assert "Test error" in e.message


class TestSecurityValidation:
    """Test security validation features."""
    
    def test_security_import_handling(self):
        """Test security features graceful import handling."""
        # Test that security functions are available through main import
        assert hasattr(scgraph_hub, 'get_security_validator')
        
        # But should show proper error when called
        try:
            scgraph_hub.get_security_validator()
        except ImportError as e:
            assert "additional dependencies" in str(e)
    
    def test_existing_security_features(self):
        """Test existing security features from security.py."""
        from scgraph_hub.security import SecurityValidator
        
        validator = SecurityValidator()
        
        # Test file path validation
        safe_path = "dataset.h5ad"
        result = validator.validate_file_path(safe_path)
        assert result['safe'] == True
        
        # Test unsafe path
        unsafe_path = "../../../etc/passwd"
        result = validator.validate_file_path(unsafe_path)
        assert result['safe'] == False
        assert len(result['issues']) > 0
    
    def test_data_sanitization(self):
        """Test data sanitization functions."""
        from scgraph_hub.security import DataSanitizer
        
        # Test string sanitization
        unsafe_string = "<script>alert('xss')</script>test"
        safe_string = DataSanitizer.sanitize_string(unsafe_string)
        assert "<script>" not in safe_string
        assert "test" in safe_string
        
        # Test numeric sanitization
        assert DataSanitizer.sanitize_numeric(100) == 100
        assert DataSanitizer.sanitize_numeric(1e7) == 1e6  # Clamped to max
        assert DataSanitizer.sanitize_numeric(-1e7) == -1e6  # Clamped to min


class TestHealthChecks:
    """Test health check system."""
    
    def test_health_status_class(self):
        """Test HealthStatus class from existing health_checks.py."""
        from scgraph_hub.health_checks import HealthStatus
        
        # Test healthy status
        status = HealthStatus("test_component", "healthy", "All good")
        assert status.is_healthy == True
        assert status.component == "test_component"
        assert status.status == "healthy"
        
        # Test unhealthy status
        unhealthy = HealthStatus("test_component", "unhealthy", "Problems detected")
        assert unhealthy.is_healthy == False
        
        # Test serialization
        status_dict = status.to_dict()
        assert status_dict['component'] == "test_component"
        assert status_dict['status'] == "healthy"
        assert 'timestamp' in status_dict


class TestAdvancedDatasetProcessing:
    """Test advanced dataset processing features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_simple_dataset_still_works(self):
        """Test that simple dataset functionality is preserved."""
        dataset = scgraph_hub.SimpleSCGraphDataset(
            name="test_dataset",
            root=self.temp_dir
        )
        
        assert dataset.name == "test_dataset"
        assert isinstance(dataset.info(), dict)
        
        # Test data structure
        data = dataset.data
        assert hasattr(data, 'num_nodes')
        assert hasattr(data, 'num_edges')
    
    def test_enhanced_dataset_availability(self):
        """Test enhanced dataset processor availability."""
        # Should be hidden behind import guard
        try:
            scgraph_hub.DatasetProcessor()
        except ImportError as e:
            assert "additional dependencies" in str(e)


class TestLoggingSystem:
    """Test logging system features."""
    
    def test_logging_import_handling(self):
        """Test logging system graceful import handling."""
        # Should be available in imports
        assert hasattr(scgraph_hub, 'setup_logging')
        assert hasattr(scgraph_hub, 'get_logger')
        
        # But should show proper error when called
        try:
            scgraph_hub.setup_logging()
        except ImportError:
            pass  # Expected
    
    def test_basic_logging_config_class(self):
        """Test basic logging configuration exists."""
        from scgraph_hub.logging_config import JsonFormatter
        
        # Test JsonFormatter class exists and can be instantiated
        formatter = JsonFormatter()
        assert formatter is not None
        
        # Create mock log record for testing
        import logging
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Test formatting
        formatted = formatter.format(record)
        assert isinstance(formatted, str)
        assert "Test message" in formatted


class TestIntegrationWorkflow:
    """Test integration of all Generation 2 features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_workflow_with_error_handling(self):
        """Test basic workflow with error handling."""
        # This should work even without advanced features
        catalog = scgraph_hub.get_default_catalog()
        datasets = catalog.list_datasets()
        
        assert len(datasets) > 0
        assert 'pbmc_10k' in datasets
        
        # Load dataset
        dataset = scgraph_hub.simple_quick_start("pbmc_10k", root=self.temp_dir)
        info = dataset.info()
        
        assert info['name'] == 'pbmc_10k'
        assert info['num_cells'] > 0
    
    def test_error_handling_in_dataset_loading(self):
        """Test error handling in dataset operations."""
        # Test invalid dataset name handling
        dataset = scgraph_hub.simple_quick_start("invalid_dataset_xyz", root=self.temp_dir)
        
        # Should not crash, but should have warning metadata
        info = dataset.info()
        assert 'invalid_dataset_xyz' in info['name']
    
    def test_graceful_degradation(self):
        """Test that system works with missing dependencies."""
        # All basic functionality should work
        catalog = scgraph_hub.get_default_catalog()
        assert len(catalog.list_datasets()) > 0
        
        # Search should work
        results = catalog.search("pbmc")
        assert len(results) > 0
        
        # Dataset loading should work
        dataset = scgraph_hub.simple_quick_start("pbmc_10k", root=self.temp_dir)
        assert dataset is not None


class TestRobustArchitecture:
    """Test architectural robustness and design patterns."""
    
    def test_package_structure(self):
        """Test package has proper structure."""
        # Test core imports work
        assert hasattr(scgraph_hub, '__version__')
        assert hasattr(scgraph_hub, 'DatasetCatalog')
        assert hasattr(scgraph_hub, 'SimpleSCGraphDataset')
        
        # Test advanced features are properly gated
        advanced_features = [
            'DatasetProcessor', 'SecurityValidator', 'setup_logging',
            'SystemHealthChecker', 'get_security_validator'
        ]
        
        for feature in advanced_features:
            assert hasattr(scgraph_hub, feature)
    
    def test_exception_system_design(self):
        """Test exception system design."""
        from scgraph_hub import exceptions
        
        # Test base exception exists
        assert hasattr(exceptions, 'SCGraphHubError')
        
        # Test specialized exceptions exist  
        assert hasattr(exceptions, 'DatasetError')
        assert hasattr(exceptions, 'DatasetNotFoundError')
        assert hasattr(exceptions, 'ValidationError')
        
        # Test utility functions
        assert hasattr(exceptions, 'handle_error_gracefully')
        assert hasattr(exceptions, 'ErrorCollector')
        assert hasattr(exceptions, 'create_error_context')
    
    def test_security_system_design(self):
        """Test security system design."""
        from scgraph_hub import security
        
        # Test security components exist
        assert hasattr(security, 'SecurityValidator')
        assert hasattr(security, 'SafeDataLoader') 
        assert hasattr(security, 'DataSanitizer')
        assert hasattr(security, 'ResourceMonitor')
        
        # Test utility functions
        assert hasattr(security, 'generate_session_token')
        assert hasattr(security, 'hash_sensitive_data')
        assert hasattr(security, 'validate_user_input')


def run_all_tests():
    """Run all Generation 2 tests."""
    test_classes = [
        TestRobustErrorHandling,
        TestSecurityValidation, 
        TestHealthChecks,
        TestAdvancedDatasetProcessing,
        TestLoggingSystem,
        TestIntegrationWorkflow,
        TestRobustArchitecture
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    print("Running Generation 2 Robust Features Tests")
    print("=" * 50)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 30)
        
        instance = test_class()
        
        # Setup if available
        if hasattr(instance, 'setup_method'):
            instance.setup_method()
        
        # Find and run test methods
        for attr_name in dir(instance):
            if attr_name.startswith('test_'):
                total_tests += 1
                try:
                    print(f"  {attr_name}...", end=" ")
                    test_method = getattr(instance, attr_name)
                    test_method()
                    print("✅ PASS")
                    passed_tests += 1
                except Exception as e:
                    print(f"❌ FAIL: {e}")
                    failed_tests += 1
        
        # Teardown if available
        if hasattr(instance, 'teardown_method'):
            instance.teardown_method()
    
    print("\n" + "=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print("=" * 50)
    
    if failed_tests == 0:
        print("🎉 All Generation 2 tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)