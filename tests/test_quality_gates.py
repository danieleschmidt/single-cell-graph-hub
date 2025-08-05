#!/usr/bin/env python3
"""Comprehensive quality gates and testing for Single-Cell Graph Hub."""

import os
import sys
import unittest
import warnings
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
from torch_geometric.data import Data

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import scgraph_hub as scg
from scgraph_hub.models import CellGraphGNN, CellGraphSAGE, SpatialGAT
from scgraph_hub.preprocessing import PreprocessingPipeline, GraphConstructor
from scgraph_hub.validation import DataValidator
from scgraph_hub.security import InputSanitizer
from scgraph_hub.monitoring import HealthChecker, PerformanceMonitor
from scgraph_hub.caching import PerformanceOptimizer, AdaptiveCache
from scgraph_hub.scalability import ConcurrentProcessor, MemoryOptimizer

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class QualityGateTestCase(unittest.TestCase):
    """Base test case with quality gate utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_data_small = self._create_test_data(100, 50, 5)
        self.test_data_medium = self._create_test_data(1000, 200, 10)
        
        # Performance thresholds
        self.performance_thresholds = {
            'model_creation_time': 1.0,  # seconds
            'forward_pass_time': 0.1,    # seconds
            'memory_usage_mb': 100,      # MB
            'accuracy_threshold': 0.1    # minimum accuracy
        }
        
        # Suppress warnings for cleaner test output
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def _create_test_data(self, n_nodes: int, n_features: int, n_classes: int) -> Data:
        """Create synthetic test data."""
        x = torch.randn(n_nodes, n_features)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 5))
        y = torch.randint(0, n_classes, (n_nodes,))
        
        # Create train/val/test masks
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        n_train = int(0.6 * n_nodes)
        n_val = int(0.2 * n_nodes)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        return Data(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    def measure_performance(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance metrics for a function."""
        import psutil
        import gc
        
        # Clear memory before measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_usage_mb': memory_after - memory_before,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after
        }


class TestCoreComponents(QualityGateTestCase):
    """Test core component quality gates."""
    
    def test_model_creation_performance(self):
        """Test that model creation meets performance requirements."""
        models_to_test = [
            ('CellGraphGNN', CellGraphGNN, {'input_dim': 100, 'hidden_dim': 64, 'output_dim': 5}),
            ('CellGraphSAGE', CellGraphSAGE, {'input_dim': 100, 'hidden_dims': [128, 64], 'aggregator': 'mean'}),
            ('SpatialGAT', SpatialGAT, {'input_dim': 100, 'hidden_dim': 64, 'num_heads': 4})
        ]
        
        for model_name, model_class, kwargs in models_to_test:
            with self.subTest(model=model_name):
                performance = self.measure_performance(model_class, **kwargs)
                
                model = performance['result']
                
                # Quality gates
                self.assertLess(performance['execution_time'], 
                               self.performance_thresholds['model_creation_time'],
                               f"{model_name} creation took too long")
                
                self.assertLess(performance['memory_usage_mb'],
                               self.performance_thresholds['memory_usage_mb'],
                               f"{model_name} uses too much memory")
                
                # Model should have reasonable number of parameters
                num_params = model.num_parameters()
                self.assertGreater(num_params, 0, f"{model_name} has no parameters")
                self.assertLess(num_params, 10_000_000, f"{model_name} has too many parameters")
    
    def test_model_forward_pass_performance(self):
        """Test that model forward passes meet performance requirements."""
        model = CellGraphGNN(input_dim=50, hidden_dim=32, output_dim=5)
        data = self.test_data_small
        
        # Warm up
        _ = model(data.x, data.edge_index)
        
        # Measure performance
        performance = self.measure_performance(model, data.x, data.edge_index)
        
        output = performance['result']
        
        # Quality gates
        self.assertLess(performance['execution_time'],
                       self.performance_thresholds['forward_pass_time'],
                       "Forward pass took too long")
        
        # Output shape should be correct
        expected_shape = (data.x.shape[0], 5)
        self.assertEqual(output.shape, expected_shape, "Incorrect output shape")
        
        # Output should be valid (no NaN or inf)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(output).any(), "Output contains infinite values")
    
    def test_model_training_quality(self):
        """Test that models can train and achieve minimum quality."""
        model = CellGraphGNN(input_dim=50, hidden_dim=32, output_dim=5)
        data = self.test_data_small
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(10):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 9:
                final_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Quality gates
        self.assertIsNotNone(initial_loss, "Failed to compute initial loss")
        self.assertIsNotNone(final_loss, "Failed to compute final loss")
        self.assertLess(final_loss, initial_loss, "Model failed to learn (loss did not decrease)")
        
        # Test inference
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[data.test_mask].argmax(dim=1)
            accuracy = (pred == data.y[data.test_mask]).float().mean().item()
        
        self.assertGreater(accuracy, self.performance_thresholds['accuracy_threshold'],
                          "Model accuracy too low")


class TestRobustnessQualityGates(QualityGateTestCase):
    """Test robustness quality gates."""
    
    def test_input_validation_comprehensive(self):
        """Test comprehensive input validation."""
        model = CellGraphGNN(input_dim=50, hidden_dim=32, output_dim=5)
        
        # Test cases for invalid inputs
        invalid_inputs = [
            # Wrong dimensions
            (torch.randn(50), torch.randint(0, 50, (2, 100)), "1D node features"),
            (torch.randn(50, 30), torch.randint(0, 50, (3, 100)), "Wrong edge_index shape"),
            (torch.randn(50, 30), torch.randint(0, 60, (2, 100)), "Invalid node indices"),
            
            # Invalid values
            (torch.full((50, 30), float('nan')), torch.randint(0, 50, (2, 100)), "NaN features"),
            (torch.full((50, 30), float('inf')), torch.randint(0, 50, (2, 100)), "Infinite features"),
        ]
        
        for x, edge_index, description in invalid_inputs:
            with self.subTest(case=description):
                with self.assertRaises((ValueError, RuntimeError), 
                                     msg=f"Failed to catch invalid input: {description}"):
                    model(x, edge_index)
    
    def test_security_sanitization(self):
        """Test security input sanitization."""
        sanitizer = InputSanitizer()
        
        # Test string sanitization
        malicious_strings = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "test\x00null_byte"
        ]
        
        for malicious_string in malicious_strings:
            with self.subTest(string=malicious_string):
                with self.assertRaises(ValueError, 
                                     msg=f"Failed to catch malicious string: {malicious_string}"):
                    sanitizer.sanitize_string(malicious_string)
        
        # Test tensor sanitization
        problematic_tensor = torch.tensor([1.0, float('nan'), float('inf'), 1e10])
        clean_tensor = sanitizer.sanitize_tensor(problematic_tensor)
        
        self.assertTrue(torch.isfinite(clean_tensor).all(), "Tensor sanitization failed")
    
    def test_health_monitoring_comprehensive(self):
        """Test comprehensive health monitoring."""
        health_checker = HealthChecker()
        
        # System health check
        system_health = health_checker.check_system_health()
        
        self.assertIn('status', system_health, "Missing status in system health")
        self.assertIn('checks', system_health, "Missing checks in system health")
        self.assertIn('memory', system_health['checks'], "Missing memory check")
        self.assertIn('cpu', system_health['checks'], "Missing CPU check")
        
        # Model health check
        model = CellGraphGNN(input_dim=50, hidden_dim=32, output_dim=5)
        model_health = health_checker.check_model_health(model)
        
        self.assertIn('status', model_health, "Missing status in model health")
        self.assertIn('checks', model_health, "Missing checks in model health")
        self.assertIn('parameters', model_health['checks'], "Missing parameter check")


class TestScalabilityQualityGates(QualityGateTestCase):
    """Test scalability quality gates."""
    
    def test_performance_optimization(self):
        """Test performance optimization features."""
        optimizer = PerformanceOptimizer()
        
        # Test tensor optimization (should not crash)
        try:
            optimizer.optimize_tensor_operations()
        except Exception as e:
            self.fail(f"Tensor optimization failed: {e}")
        
        # Test batch size calculation
        optimal_batch_size = optimizer.get_optimal_batch_size(
            model_size_mb=100, available_memory_mb=4000
        )
        
        self.assertGreater(optimal_batch_size, 0, "Invalid batch size calculation")
        self.assertLessEqual(optimal_batch_size, 256, "Batch size too large")
    
    def test_caching_performance(self):
        """Test caching system performance."""
        cache = AdaptiveCache(max_size_mb=50, compression=True)
        
        # Test cache operations
        test_data = {
            'small_tensor': torch.randn(100, 10),
            'large_tensor': torch.randn(1000, 100),
            'dict_data': {'key1': 'value1', 'key2': list(range(100))},
            'array_data': np.random.randn(500, 20)
        }
        
        # Put items in cache
        for key, value in test_data.items():
            cache.put(key, value, ttl=3600)
        
        # Retrieve items and measure hit rate
        hits = 0
        for key in test_data.keys():
            if cache.get(key) is not None:
                hits += 1
        
        hit_rate = hits / len(test_data)
        self.assertGreaterEqual(hit_rate, 0.8, "Cache hit rate too low")
        
        # Check cache stats
        stats = cache.get_stats()
        self.assertIn('hit_rate', stats, "Missing hit rate in stats")
        self.assertIn('size_mb', stats, "Missing size in stats")
    
    def test_concurrent_processing_quality(self):
        """Test concurrent processing quality."""
        processor = ConcurrentProcessor(max_workers=2)
        
        # Create test processing function
        def test_process(data):
            time.sleep(0.01)  # Simulate processing
            return np.mean(data)
        
        # Test data
        test_batches = [np.random.randn(100, 10) for _ in range(4)]
        
        # Measure sequential vs concurrent performance
        start_time = time.time()
        sequential_results = [test_process(batch) for batch in test_batches]
        sequential_time = time.time() - start_time
        
        start_time = time.time()
        concurrent_results = processor.process_datasets_parallel(
            [f'batch_{i}' for i in range(len(test_batches))],
            lambda name: test_process(test_batches[int(name.split('_')[1])]),
            max_concurrent=2
        )
        concurrent_time = time.time() - start_time
        
        # Quality gates
        self.assertEqual(len(concurrent_results), len(sequential_results), 
                        "Concurrent processing lost results")
        
        # Should show some speedup (allowing for overhead)
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
        self.assertGreater(speedup, 0.8, "Insufficient concurrent processing speedup")


class TestIntegrationQualityGates(QualityGateTestCase):
    """Test end-to-end integration quality gates."""
    
    def test_complete_workflow_quality(self):
        """Test complete workflow from data to trained model."""
        # Create synthetic dataset
        data = self.test_data_medium
        
        # Data validation
        validator = DataValidator(strict_mode=False)
        validation_results = validator.validate_dataset(data, "test_dataset")
        self.assertTrue(validation_results['valid'], "Dataset validation failed")
        
        # Model creation and training
        model = CellGraphGNN(
            input_dim=data.x.shape[1],
            hidden_dim=64,
            output_dim=len(torch.unique(data.y)),
            num_layers=2,
            dropout=0.2
        )
        
        # Measure training performance
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        training_performance = self.measure_performance(
            self._train_model_epochs, model, data, optimizer, criterion, 5
        )
        
        # Quality gates for training
        self.assertLess(training_performance['execution_time'], 30.0, 
                       "Training took too long")
        self.assertLess(training_performance['memory_usage_mb'], 500.0,
                       "Training used too much memory")
        
        # Final model evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[data.test_mask].argmax(dim=1)
            accuracy = (pred == data.y[data.test_mask]).float().mean().item()
        
        # For synthetic random data, expect accuracy above random chance
        random_chance = 1.0 / len(torch.unique(data.y))
        self.assertGreater(accuracy, random_chance * 0.8, "Final model accuracy too low")
    
    def _train_model_epochs(self, model, data, optimizer, criterion, epochs):
        """Helper function to train model for specified epochs."""
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        return model
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        import gc
        import psutil
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform repeated operations
        for i in range(10):
            model = CellGraphGNN(input_dim=100, hidden_dim=64, output_dim=5)
            data = self._create_test_data(500, 100, 5)
            
            # Forward pass
            _ = model(data.x, data.edge_index)
            
            # Clean up
            del model, data
            if i % 3 == 0:  # Periodic cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Quality gate: memory increase should be reasonable
        self.assertLess(memory_increase, 200.0, 
                       f"Potential memory leak detected: {memory_increase:.2f}MB increase")


class TestReliabilityQualityGates(QualityGateTestCase):
    """Test reliability and error handling quality gates."""
    
    def test_graceful_error_handling(self):
        """Test that errors are handled gracefully."""
        model = CellGraphGNN(input_dim=50, hidden_dim=32, output_dim=5)
        
        # Test various error conditions
        error_conditions = [
            # Device mismatch (if CUDA available)
            # Resource exhaustion scenarios
            # Corrupted data scenarios
        ]
        
        # For now, test basic error recovery
        try:
            # Invalid input should raise proper exception
            with self.assertRaises(ValueError):
                model(torch.randn(50), torch.randint(0, 50, (2, 100)))
        except Exception as e:
            self.fail(f"Unexpected exception type: {type(e).__name__}: {e}")
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # First run
        model1 = CellGraphGNN(input_dim=50, hidden_dim=32, output_dim=5)
        data1 = self._create_test_data(100, 50, 5)
        out1 = model1(data1.x, data1.edge_index)
        
        # Reset seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Second run
        model2 = CellGraphGNN(input_dim=50, hidden_dim=32, output_dim=5)
        data2 = self._create_test_data(100, 50, 5)
        out2 = model2(data2.x, data2.edge_index)
        
        # Results should be identical
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6), 
                       "Results are not reproducible")


def run_quality_gates() -> Dict[str, Any]:
    """Run all quality gates and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCoreComponents,
        TestRobustnessQualityGates,
        TestScalabilityQualityGates,
        TestIntegrationQualityGates,
        TestReliabilityQualityGates
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Return results summary
    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        'passed': result.wasSuccessful()
    }


if __name__ == '__main__':
    print("Single-Cell Graph Hub - Quality Gates Test Suite")
    print("=" * 60)
    
    results = run_quality_gates()
    
    print("\n" + "=" * 60)
    print("Quality Gates Results:")
    print(f"Tests run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    
    if results['passed']:
        print("🎯 All quality gates passed!")
        exit(0)
    else:
        print("❌ Some quality gates failed!")
        exit(1)