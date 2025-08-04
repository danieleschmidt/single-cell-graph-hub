"""Comprehensive tests for core Single-Cell Graph Hub functionality."""

import pytest
import torch
import numpy as np
from pathlib import Path

from scgraph_hub.models import create_model, MODEL_REGISTRY
from scgraph_hub.dataset import SCGraphDataset
from scgraph_hub.preprocessing import PreprocessingPipeline
from scgraph_hub.benchmarks import BenchmarkRunner
from scgraph_hub.validation import DataValidator
from scgraph_hub.metrics import compute_comprehensive_metrics
from scgraph_hub.scalability import auto_scale_training
from scgraph_hub.caching import get_global_cache
from scgraph_hub.security import validate_user_input
from scgraph_hub.monitoring import get_system_health


class TestCoreModels:
    """Test core model functionality."""
    
    def test_model_creation(self):
        """Test model creation for all registered models."""
        for model_name in MODEL_REGISTRY.keys():
            model = create_model(
                model_name,
                input_dim=100,
                hidden_dim=64,
                output_dim=10,
                num_layers=2
            )
            assert model is not None
            assert hasattr(model, 'forward')
    
    def test_model_forward_pass(self, sample_graph_data):
        """Test forward pass for all models."""
        for model_name in ['CellGraphGNN', 'CellGraphSAGE', 'CellGraphGAT']:
            model = create_model(
                model_name,
                input_dim=sample_graph_data.x.shape[1],
                hidden_dim=64,
                output_dim=5,
                num_layers=2
            )
            
            model.eval()
            with torch.no_grad():
                output = model(sample_graph_data.x, sample_graph_data.edge_index)
            
            assert output.shape[0] == sample_graph_data.x.shape[0]
            assert output.shape[1] == 5
    
    def test_model_training_step(self, sample_graph_data):
        """Test model training step."""
        model = create_model(
            'CellGraphGNN',
            input_dim=sample_graph_data.x.shape[1],
            hidden_dim=32,
            output_dim=5
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        optimizer.zero_grad()
        
        output = model(sample_graph_data.x, sample_graph_data.edge_index)
        loss = criterion(output[sample_graph_data.train_mask], 
                        sample_graph_data.y[sample_graph_data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
    
    @pytest.mark.parametrized
    def test_model_different_input_sizes(self, variable_node_count):
        """Test models with different input sizes."""
        n_nodes = variable_node_count
        x = torch.randn(n_nodes, 50)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 2))
        
        model = create_model('CellGraphGNN', input_dim=50, hidden_dim=32, output_dim=5)
        
        output = model(x, edge_index)
        assert output.shape == (n_nodes, 5)


class TestDataProcessing:
    """Test data processing functionality."""
    
    def test_dataset_creation(self, temp_dir, sample_dataset_config):
        """Test dataset creation and loading."""
        dataset = SCGraphDataset(
            name=sample_dataset_config['name'],
            root=str(temp_dir),
            download=False
        )
        
        assert dataset.name == sample_dataset_config['name']
        assert hasattr(dataset, 'process')
    
    def test_preprocessing_pipeline(self, mock_anndata, sample_preprocessing_config):
        """Test preprocessing pipeline."""
        pipeline = PreprocessingPipeline(
            steps=sample_preprocessing_config['steps'],
            parameters=sample_preprocessing_config['parameters']
        )
        
        processed_data = pipeline.process(mock_anndata, return_metadata=True)
        adata, metadata = processed_data
        
        assert adata is not None
        assert metadata['steps_applied'] == sample_preprocessing_config['steps']
    
    def test_data_validation(self, sample_graph_data, data_validator):
        """Test data validation."""
        results = data_validator.validate_dataset(sample_graph_data, 'test_dataset')
        
        assert 'valid' in results
        assert 'errors' in results
        assert 'warnings' in results
        assert 'statistics' in results
    
    def test_graph_statistics_computation(self, sample_graph_data):
        """Test graph statistics computation."""
        from scgraph_hub.utils import compute_graph_statistics
        
        stats = compute_graph_statistics(
            sample_graph_data.edge_index,
            sample_graph_data.x.shape[0]
        )
        
        assert 'n_nodes' in stats
        assert 'n_edges' in stats
        assert 'density' in stats
        assert stats['n_nodes'] == sample_graph_data.x.shape[0]


class TestBenchmarking:
    """Test benchmarking functionality."""
    
    @pytest.mark.slow
    def test_benchmark_runner(self, temp_dir, sample_graph_data):
        """Test benchmark runner with simple configuration."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        
        # Simple benchmark configuration
        models = {
            'test_model': {
                'input_dim': sample_graph_data.x.shape[1],
                'hidden_dim': 32,
                'output_dim': 5
            }
        }
        
        # Create mock dataset by saving sample data
        dataset_path = temp_dir / 'test_dataset.pt'
        torch.save(sample_graph_data, dataset_path)
        
        # Mock data manager to return our sample data
        runner.data_manager.load_dataset = lambda name, **kwargs: sample_graph_data
        
        results = runner.run_benchmark(
            models=models,
            datasets=['test_dataset'],
            tasks=['cell_type_prediction'],
            metrics=['accuracy'],
            n_runs=1
        )
        
        assert 'results' in results
        assert 'test_model' in results['results']
    
    def test_metrics_computation(self, sample_graph_data):
        """Test comprehensive metrics computation."""
        # Create mock predictions
        predictions = torch.randn(sample_graph_data.x.shape[0], 5)
        embeddings = torch.randn(sample_graph_data.x.shape[0], 32)
        
        metrics = compute_comprehensive_metrics(
            predictions=predictions,
            embeddings=embeddings,
            ground_truth=sample_graph_data.y,
            task_type='classification'
        )
        
        assert 'accuracy' in metrics
        assert 'f1_macro' in metrics
        assert 'silhouette_score' in metrics


class TestScalability:
    """Test scalability features."""
    
    def test_auto_scaling_configuration(self, sample_graph_data):
        """Test automatic scaling configuration."""
        model = create_model(
            'CellGraphGNN',
            input_dim=sample_graph_data.x.shape[1],
            hidden_dim=64,
            output_dim=5
        )
        
        config = auto_scale_training(
            model=model,
            data=sample_graph_data,
            target_memory_gb=4.0
        )
        
        assert 'config' in config
        assert 'recommendations' in config
        assert config['config'].batch_size > 0
    
    @pytest.mark.large_dataset
    def test_large_graph_processing(self, large_graph_data):
        """Test processing of large graphs."""
        from scgraph_hub.scalability import GraphSampler
        
        sampler = GraphSampler()
        
        # Test neighbor sampling
        node_indices = torch.randint(0, large_graph_data.x.shape[0], (100,))
        subgraph = sampler.sample_subgraph(
            large_graph_data,
            node_indices,
            num_hops=2
        )
        
        assert subgraph.x.shape[0] >= 100  # Should include sampled nodes and neighbors
        assert subgraph.edge_index.shape[0] == 2
    
    def test_memory_optimization(self, sample_graph_data):
        """Test memory optimization features."""
        from scgraph_hub.scalability import MemoryOptimizer
        
        optimizer = MemoryOptimizer(max_memory_gb=4.0)
        optimizer.enable_memory_efficient_mode()
        
        model = create_model(
            'CellGraphGNN',
            input_dim=sample_graph_data.x.shape[1],
            hidden_dim=64,
            output_dim=5
        )
        
        optimal_batch_size = optimizer.get_optimal_batch_size(
            sample_graph_data, model, torch.device('cpu')
        )
        
        assert optimal_batch_size > 0
        assert optimal_batch_size <= 256  # Reasonable upper bound


class TestCaching:
    """Test caching functionality."""
    
    def test_global_cache_operations(self):
        """Test global cache operations."""
        cache = get_global_cache()
        cache.clear()  # Start clean
        
        # Test put and get
        test_data = {'key': 'value', 'number': 42}
        cache.put('test_key', test_data)
        
        retrieved_data = cache.get('test_key')
        assert retrieved_data == test_data
        
        # Test cache miss
        missing_data = cache.get('nonexistent_key')
        assert missing_data is None
    
    def test_dataset_caching(self, sample_graph_data):
        """Test dataset caching functionality."""
        from scgraph_hub.caching import cache_dataset, get_cached_dataset
        
        # Cache dataset
        cache_dataset('test_dataset', sample_graph_data)
        
        # Retrieve cached dataset
        cached_data = get_cached_dataset('test_dataset')
        
        assert cached_data is not None
        assert torch.equal(cached_data.x, sample_graph_data.x)
        assert torch.equal(cached_data.edge_index, sample_graph_data.edge_index)
    
    def test_cache_statistics(self):
        """Test cache statistics reporting."""
        from scgraph_hub.caching import get_cache_stats
        
        stats = get_cache_stats()
        
        assert 'memory' in stats
        assert 'disk' in stats
        assert 'total_entries' in stats


class TestSecurity:
    """Test security features."""
    
    def test_input_validation(self):
        """Test user input validation."""
        # Test valid input
        valid_input = {
            'model_name': 'CellGraphGNN',
            'batch_size': 32,
            'learning_rate': 0.01
        }
        
        result = validate_user_input(valid_input)
        assert result['safe'] == True
        
        # Test suspicious input
        suspicious_input = {
            'command': 'import os; os.system("rm -rf /")',
            'script': 'eval(malicious_code)'
        }
        
        result = validate_user_input(suspicious_input)
        assert len(result['warnings']) > 0
    
    def test_file_path_validation(self):
        """Test file path security validation."""
        from scgraph_hub.security import SecurityValidator
        
        validator = SecurityValidator()
        
        # Test safe path
        safe_result = validator.validate_file_path('data/dataset.h5ad')
        assert safe_result['safe'] == True
        
        # Test path traversal
        unsafe_result = validator.validate_file_path('../../../etc/passwd')
        assert unsafe_result['safe'] == False
    
    def test_configuration_sanitization(self):
        """Test configuration sanitization."""
        from scgraph_hub.security import DataSanitizer
        
        sanitizer = DataSanitizer()
        
        unsafe_config = {
            'model_name': '<script>alert("xss")</script>',
            'batch_size': 'very_large_number_999999999',
            'file_path': '../../../dangerous/path'
        }
        
        sanitized = sanitizer.sanitize_config(unsafe_config)
        
        assert '<script>' not in sanitized['model_name']
        assert isinstance(sanitized['batch_size'], (int, float))


class TestMonitoring:
    """Test monitoring and health checks."""
    
    def test_system_health_check(self):
        """Test system health monitoring."""
        health_status = get_system_health()
        
        assert 'overall_status' in health_status
        assert 'checks' in health_status
        assert 'timestamp' in health_status
        
        assert health_status['overall_status'] in ['healthy', 'degraded', 'unhealthy']
    
    def test_performance_monitoring(self, performance_monitor):
        """Test performance monitoring."""
        # Record some metrics
        performance_monitor.record_metric('test_metric', 1.5)
        performance_monitor.record_metric('test_metric', 2.0)
        performance_monitor.record_metric('test_metric', 1.8)
        
        stats = performance_monitor.get_metric_stats('test_metric')
        
        assert stats['count'] == 3
        assert stats['mean'] == pytest.approx(1.77, abs=0.01)
        assert stats['min'] == 1.5
        assert stats['max'] == 2.0
    
    def test_model_monitoring(self):
        """Test model training monitoring."""
        from scgraph_hub.monitoring import get_model_monitor
        
        monitor = get_model_monitor()
        
        # Simulate training epochs
        for epoch in range(5):
            monitor.on_epoch_start(epoch)
            
            metrics = {
                'loss': 1.0 - epoch * 0.1,
                'accuracy': 0.5 + epoch * 0.1
            }
            
            monitor.on_epoch_end(epoch, metrics)
        
        summary = monitor.get_training_summary()
        
        assert summary['current_epoch'] == 4
        assert summary['total_epochs'] == 5
        assert 'latest_metrics' in summary
        assert 'best_metrics' in summary


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.integration
    def test_complete_training_workflow(self, temp_dir, sample_graph_data):
        """Test complete training workflow integration."""
        # 1. Create and validate model
        model = create_model(
            'CellGraphGNN',
            input_dim=sample_graph_data.x.shape[1],
            hidden_dim=32,
            output_dim=5
        )
        
        validator = DataValidator()
        validation_results = validator.validate_dataset(sample_graph_data)
        assert validation_results['valid']
        
        # 2. Set up monitoring
        from scgraph_hub.monitoring import get_performance_monitor, get_model_monitor
        
        perf_monitor = get_performance_monitor()
        model_monitor = get_model_monitor()
        
        # 3. Train model with monitoring
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        
        for epoch in range(3):  # Short training for test
            model_monitor.on_epoch_start(epoch)
            
            perf_monitor.start_timer('epoch')
            
            optimizer.zero_grad()
            output = model(sample_graph_data.x, sample_graph_data.edge_index)
            loss = criterion(output[sample_graph_data.train_mask], 
                           sample_graph_data.y[sample_graph_data.train_mask])
            loss.backward()
            optimizer.step()
            
            epoch_time = perf_monitor.end_timer('epoch')
            
            # Calculate metrics
            with torch.no_grad():
                val_output = model(sample_graph_data.x, sample_graph_data.edge_index)
                val_loss = criterion(val_output[sample_graph_data.val_mask],
                                   sample_graph_data.y[sample_graph_data.val_mask])
                
                pred = val_output[sample_graph_data.val_mask].argmax(dim=1)
                acc = (pred == sample_graph_data.y[sample_graph_data.val_mask]).float().mean()
            
            metrics = {
                'loss': loss.item(),
                'val_loss': val_loss.item(),
                'val_accuracy': acc.item(),
                'epoch_time': epoch_time
            }
            
            model_monitor.on_epoch_end(epoch, metrics)
        
        # 4. Validate training completed successfully
        training_summary = model_monitor.get_training_summary()
        assert training_summary['total_epochs'] == 3
        
        perf_stats = perf_monitor.get_all_stats()
        assert 'epoch_duration' in perf_stats
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_benchmark(self, temp_dir, sample_graph_data):
        """Test end-to-end benchmarking workflow."""
        # This would test the complete pipeline from data loading to results
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        
        # Mock the data loading
        runner.data_manager.load_dataset = lambda name, **kwargs: sample_graph_data
        
        models = {
            'simple_gnn': {
                'input_dim': sample_graph_data.x.shape[1],
                'hidden_dim': 16,
                'output_dim': 5
            }
        }
        
        results = runner.run_benchmark(
            models=models,
            datasets=['test_dataset'],
            tasks=['cell_type_prediction'],
            metrics=['accuracy'],
            n_runs=1,
            train_config={'epochs': 2, 'learning_rate': 0.01}
        )
        
        # Validate benchmark results
        assert results['config']['n_runs'] == 1
        assert 'simple_gnn' in results['results']
        assert 'test_dataset' in results['results']['simple_gnn']
        
        # Check that training actually happened
        model_results = results['results']['simple_gnn']['test_dataset']['cell_type_prediction']
        assert len(model_results['runs']) == 1
        assert model_results['runs'][0]['status'] == 'completed'


# Stress tests for robustness
class TestStressTesting:
    """Stress tests for robustness validation."""
    
    @pytest.mark.slow
    def test_memory_stress(self, large_graph_data):
        """Test behavior under memory stress."""
        model = create_model(
            'CellGraphGNN',
            input_dim=large_graph_data.x.shape[1],
            hidden_dim=128,
            output_dim=10
        )
        
        # This should not crash even with large data
        with torch.no_grad():
            output = model(large_graph_data.x, large_graph_data.edge_index)
        
        assert output.shape[0] == large_graph_data.x.shape[0]
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        model = create_model('CellGraphGNN', input_dim=50, hidden_dim=32, output_dim=5)
        
        # Test with mismatched dimensions
        with pytest.raises((RuntimeError, ValueError, TypeError)):
            x = torch.randn(100, 30)  # Wrong feature dimension
            edge_index = torch.randint(0, 100, (2, 200))
            model(x, edge_index)
        
        # Test with invalid edge indices
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            x = torch.randn(50, 50)
            edge_index = torch.randint(0, 100, (2, 200))  # Indices exceed node count
            model(x, edge_index)
    
    def test_concurrent_operations(self):
        """Test thread safety of core operations."""
        import threading
        import queue
        
        cache = get_global_cache()
        cache.clear()
        
        results = queue.Queue()
        
        def cache_worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_item_{i}"
                    data = {'worker': worker_id, 'item': i}
                    cache.put(key, data)
                    
                    retrieved = cache.get(key)
                    assert retrieved == data
                
                results.put(('success', worker_id))
            except Exception as e:
                results.put(('error', worker_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            result = results.get()
            if result[0] == 'success':
                success_count += 1
            else:
                pytest.fail(f"Thread {result[1]} failed: {result[2]}")
        
        assert success_count == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
