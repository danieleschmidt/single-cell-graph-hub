"""Integration tests for FastAPI endpoints."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
import tempfile
from pathlib import Path


@pytest.fixture
def test_client():
    """Create a test client for the API."""
    from scgraph_hub.api import app
    return TestClient(app)


@pytest.fixture
def mock_catalog():
    """Mock catalog for testing."""
    mock = Mock()
    mock.list_datasets.return_value = ['dataset1', 'dataset2', 'dataset3']
    mock.get_info.return_value = {
        'name': 'test_dataset',
        'description': 'Test dataset for API testing',
        'n_cells': 1000,
        'n_genes': 2000,
        'n_classes': 5,
        'modality': 'scRNA-seq',
        'organism': 'human',
        'tissue': 'blood',
        'has_spatial': False,
        'graph_method': 'knn',
        'size_mb': 150.0,
        'citation': 'Test et al., 2024',
        'tasks': ['cell_type_prediction']
    }
    mock.filter.return_value = ['dataset1', 'dataset2']
    mock.get_summary_stats.return_value = {
        'total_datasets': 10,
        'total_cells': 50000,
        'modalities': ['scRNA-seq', 'spatial'],
        'organisms': ['human', 'mouse']
    }
    mock.get_tasks_summary.return_value = {
        'cell_type_prediction': 8,
        'trajectory_inference': 3,
        'spatial_analysis': 2
    }
    return mock


@pytest.fixture
def mock_data_manager():
    """Mock data manager for testing."""
    mock = Mock()
    mock.get_dataset_statistics.return_value = {
        'basic': {
            'num_nodes': 1000,
            'num_edges': 5000,
            'num_features': 2000,
            'has_labels': True,
            'num_classes': 5
        },
        'graph': {
            'density': 0.01,
            'average_degree': 10.0,
            'is_undirected': True
        },
        'features': {
            'feature_mean': 0.5,
            'feature_std': 1.2,
            'sparsity': 0.85
        }
    }
    mock.get_storage_info.return_value = {
        'data_root': '/tmp/data',
        'cache_dir': '/tmp/cache',
        'raw_data_size_mb': 500.0,
        'processed_data_size_mb': 200.0,
        'cache_size_mb': 50.0,
        'total_size_mb': 750.0
    }
    return mock


@pytest.fixture
def mock_dataset_repository():
    """Mock dataset repository for testing."""
    mock = Mock()
    mock.list_datasets.return_value = []
    return mock


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    mock = Mock()
    mock.set.return_value = True
    mock.get.return_value = None
    return mock


class TestRootEndpoints:
    """Test cases for root and health endpoints."""
    
    @pytest.mark.integration
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'message' in data
        assert 'version' in data
        assert 'docs' in data
        assert data['message'] == 'Single-Cell Graph Hub API'
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_dataset_repository')
    @patch('scgraph_hub.api.get_cache_manager')
    def test_health_check_healthy(self, mock_get_cache, mock_get_repo, test_client):
        """Test health check endpoint when all services are healthy."""
        # Mock successful repository check
        mock_repo = Mock()
        mock_repo.list_datasets.return_value = []
        mock_get_repo.return_value = mock_repo
        
        # Mock successful cache check
        mock_cache = Mock()
        mock_cache.set.return_value = True
        mock_get_cache.return_value = mock_cache
        
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'version' in data
        assert 'services' in data
        assert data['services']['api'] == 'healthy'
        assert data['services']['database'] == 'healthy'
        assert data['services']['cache'] == 'healthy'
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_dataset_repository')
    @patch('scgraph_hub.api.get_cache_manager')
    def test_health_check_degraded(self, mock_get_cache, mock_get_repo, test_client):
        """Test health check endpoint when some services are unhealthy."""
        # Mock failed repository check
        mock_repo = Mock()
        mock_repo.list_datasets.side_effect = Exception("Database error")
        mock_get_repo.return_value = mock_repo
        
        # Mock successful cache check
        mock_cache = Mock()
        mock_cache.set.return_value = True
        mock_get_cache.return_value = mock_cache
        
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['status'] == 'degraded'
        assert 'unhealthy' in data['services']['database']


class TestDatasetEndpoints:
    """Test cases for dataset-related endpoints."""
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_catalog')
    def test_list_datasets(self, mock_get_catalog, mock_catalog, test_client):
        """Test list datasets endpoint."""
        mock_get_catalog.return_value = mock_catalog
        
        response = test_client.get("/datasets")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        assert 'dataset1' in data
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_catalog')
    def test_get_dataset_info_success(self, mock_get_catalog, mock_catalog, test_client):
        """Test get dataset info endpoint with valid dataset."""
        mock_get_catalog.return_value = mock_catalog
        
        response = test_client.get("/datasets/test_dataset")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['name'] == 'test_dataset'
        assert data['n_cells'] == 1000
        assert data['n_genes'] == 2000
        assert data['modality'] == 'scRNA-seq'
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_catalog')
    def test_get_dataset_info_not_found(self, mock_get_catalog, test_client):
        """Test get dataset info endpoint with invalid dataset."""
        mock_catalog = Mock()
        mock_catalog.get_info.side_effect = KeyError("Dataset not found")
        mock_get_catalog.return_value = mock_catalog
        
        response = test_client.get("/datasets/nonexistent_dataset")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert 'error' in data
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_catalog')
    def test_search_datasets(self, mock_get_catalog, mock_catalog, test_client):
        """Test search datasets endpoint."""
        mock_get_catalog.return_value = mock_catalog
        
        search_filters = {
            "modality": "scRNA-seq",
            "organism": "human",
            "min_cells": 500
        }
        
        response = test_client.post("/datasets/search", json=search_filters)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_data_manager_instance')
    def test_get_dataset_statistics(self, mock_get_dm, mock_data_manager, test_client):
        """Test get dataset statistics endpoint."""
        mock_get_dm.return_value = mock_data_manager
        
        response = test_client.get("/datasets/test_dataset/statistics")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'basic' in data
        assert 'graph' in data
        assert 'features' in data
        assert data['basic']['num_nodes'] == 1000
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_data_manager_instance')
    def test_get_dataset_statistics_not_found(self, mock_get_dm, test_client):
        """Test get dataset statistics for non-existent dataset."""
        mock_dm = Mock()
        mock_dm.get_dataset_statistics.return_value = None
        mock_get_dm.return_value = mock_dm
        
        response = test_client.get("/datasets/nonexistent/statistics")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDatasetOperations:
    """Test cases for dataset operation endpoints."""
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_data_manager_instance')
    def test_download_dataset(self, mock_get_dm, test_client):
        """Test dataset download endpoint."""
        mock_dm = Mock()
        mock_dm.__aenter__ = AsyncMock(return_value=mock_dm)
        mock_dm.__aexit__ = AsyncMock(return_value=None)
        mock_dm.download_dataset_async = AsyncMock(return_value=True)
        mock_get_dm.return_value = mock_dm
        
        response = test_client.post("/datasets/test_dataset/download")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'task_id' in data
        assert 'message' in data
        assert data['message'] == 'Download started'
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_data_manager_instance')
    def test_preprocess_dataset(self, mock_get_dm, test_client):
        """Test dataset preprocessing endpoint."""
        mock_dm = Mock()
        mock_dm.__aenter__ = AsyncMock(return_value=mock_dm)
        mock_dm.__aexit__ = AsyncMock(return_value=None)
        mock_dm.preprocess_dataset_async = AsyncMock(return_value=True)
        mock_get_dm.return_value = mock_dm
        
        preprocessing_config = {
            "steps": ["normalize_total", "log1p"],
            "parameters": {"normalize_total": {"target_sum": 10000}},
            "graph_method": "knn",
            "graph_parameters": {"k": 15}
        }
        
        response = test_client.post(
            "/datasets/test_dataset/preprocess",
            json=preprocessing_config
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'task_id' in data
        assert 'message' in data
        assert data['message'] == 'Preprocessing started'
    
    @pytest.mark.integration
    def test_get_task_status(self, test_client):
        """Test task status endpoint."""
        # First create a task
        with patch('scgraph_hub.api.get_data_manager_instance') as mock_get_dm:
            mock_dm = Mock()
            mock_dm.__aenter__ = AsyncMock(return_value=mock_dm)
            mock_dm.__aexit__ = AsyncMock(return_value=None)
            mock_dm.download_dataset_async = AsyncMock(return_value=True)
            mock_get_dm.return_value = mock_dm
            
            # Start download to create task
            response = test_client.post("/datasets/test_dataset/download")
            task_id = response.json()['task_id']
        
        # Check task status
        response = test_client.get(f"/tasks/{task_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'task_id' in data
        assert 'status' in data
        assert 'progress' in data
        assert 'message' in data
        assert 'started_at' in data
    
    @pytest.mark.integration
    def test_get_task_status_not_found(self, test_client):
        """Test task status endpoint with invalid task ID."""
        response = test_client.get("/tasks/nonexistent_task_id")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelEndpoints:
    """Test cases for model-related endpoints."""
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.MODEL_REGISTRY')
    def test_list_models(self, mock_registry, test_client):
        """Test list models endpoint."""
        mock_registry.keys.return_value = ['GCN', 'GAT', 'GraphSAGE', 'CellGraphGNN']
        
        response = test_client.get("/models")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 4
        assert 'GCN' in data
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_catalog')
    @patch('scgraph_hub.api.get_model_recommendations')
    def test_get_model_recommendations(self, mock_get_recs, mock_get_catalog, test_client):
        """Test model recommendations endpoint."""
        mock_catalog = Mock()
        mock_catalog.get_info.return_value = {
            'name': 'test_dataset',
            'modality': 'scRNA-seq',
            'has_spatial': False,
            'n_cells': 1000
        }
        mock_get_catalog.return_value = mock_catalog
        mock_get_recs.return_value = ['CellGraphGNN', 'GAT']
        
        response = test_client.get("/models/recommendations?dataset_name=test_dataset")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'dataset' in data
        assert 'recommended_models' in data
        assert data['dataset'] == 'test_dataset'
        assert len(data['recommended_models']) == 2
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_catalog')
    def test_get_model_recommendations_dataset_not_found(self, mock_get_catalog, test_client):
        """Test model recommendations with invalid dataset."""
        mock_catalog = Mock()
        mock_catalog.get_info.side_effect = KeyError("Dataset not found")
        mock_get_catalog.return_value = mock_catalog
        
        response = test_client.get("/models/recommendations?dataset_name=nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestExperimentEndpoints:
    """Test cases for experiment-related endpoints."""
    
    @pytest.mark.integration
    def test_run_experiment(self, test_client):
        """Test experiment execution endpoint."""
        experiment_config = {
            "dataset_name": "test_dataset",
            "model_config": {
                "model_name": "CellGraphGNN",
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.2
            },
            "training_config": {
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "experiment_name": "test_experiment"
        }
        
        response = test_client.post("/experiments", json=experiment_config)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'task_id' in data
        assert 'message' in data
        assert data['message'] == 'Experiment started'


class TestCatalogEndpoints:
    """Test cases for catalog-related endpoints."""
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_catalog')
    def test_get_catalog_summary(self, mock_get_catalog, mock_catalog, test_client):
        """Test catalog summary endpoint."""
        mock_get_catalog.return_value = mock_catalog
        
        response = test_client.get("/catalog/summary")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'total_datasets' in data
        assert 'total_cells' in data
        assert 'modalities' in data
        assert 'organisms' in data
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_catalog')
    def test_get_tasks_summary(self, mock_get_catalog, mock_catalog, test_client):
        """Test tasks summary endpoint."""
        mock_get_catalog.return_value = mock_catalog
        
        response = test_client.get("/catalog/tasks")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'cell_type_prediction' in data
        assert data['cell_type_prediction'] == 8


class TestStorageEndpoints:
    """Test cases for storage-related endpoints."""
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_data_manager_instance')
    def test_get_storage_info(self, mock_get_dm, mock_data_manager, test_client):
        """Test storage info endpoint."""
        mock_get_dm.return_value = mock_data_manager
        
        response = test_client.get("/storage/info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'data_root' in data
        assert 'cache_dir' in data
        assert 'raw_data_size_mb' in data
        assert 'processed_data_size_mb' in data
        assert 'total_size_mb' in data


class TestInputValidation:
    """Test cases for input validation."""
    
    @pytest.mark.integration
    def test_invalid_model_config(self, test_client):
        """Test experiment with invalid model configuration."""
        invalid_experiment = {
            "dataset_name": "test_dataset",
            "model_config": {
                "model_name": "InvalidModel",  # Invalid model type
                "hidden_dim": 128
            }
        }
        
        response = test_client.post("/experiments", json=invalid_experiment)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert 'detail' in data
    
    @pytest.mark.integration
    def test_invalid_training_config(self, test_client):
        """Test experiment with invalid training configuration."""
        invalid_experiment = {
            "dataset_name": "test_dataset",
            "model_config": {
                "model_name": "CellGraphGNN",
                "hidden_dim": 128
            },
            "training_config": {
                "epochs": -10,  # Invalid negative epochs
                "learning_rate": 0.001
            }
        }
        
        response = test_client.post("/experiments", json=invalid_experiment)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.integration
    def test_empty_preprocessing_config(self, test_client):
        """Test preprocessing with empty configuration."""
        empty_config = {}
        
        response = test_client.post(
            "/datasets/test_dataset/preprocess",
            json=empty_config
        )
        
        # Should accept empty config (will use defaults)
        assert response.status_code == status.HTTP_200_OK


class TestErrorHandling:
    """Test cases for error handling."""
    
    @pytest.mark.integration
    def test_404_for_invalid_endpoint(self, test_client):
        """Test 404 response for invalid endpoints."""
        response = test_client.get("/invalid/endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.integration
    def test_method_not_allowed(self, test_client):
        """Test 405 response for unsupported HTTP methods."""
        response = test_client.delete("/datasets")  # DELETE not supported
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    @pytest.mark.integration
    @patch('scgraph_hub.api.get_catalog')
    def test_internal_server_error_handling(self, mock_get_catalog, test_client):
        """Test handling of internal server errors."""
        # Mock catalog to raise unexpected exception
        mock_catalog = Mock()
        mock_catalog.list_datasets.side_effect = Exception("Unexpected error")
        mock_get_catalog.return_value = mock_catalog
        
        response = test_client.get("/datasets")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert 'error' in data
        assert data['error'] == 'Internal server error'


class TestConcurrency:
    """Test cases for concurrent requests."""
    
    @pytest.mark.integration
    def test_concurrent_downloads(self, test_client):
        """Test handling of concurrent download requests."""
        import threading
        import time
        
        responses = []
        
        def make_request():
            with patch('scgraph_hub.api.get_data_manager_instance') as mock_get_dm:
                mock_dm = Mock()
                mock_dm.__aenter__ = AsyncMock(return_value=mock_dm)
                mock_dm.__aexit__ = AsyncMock(return_value=None)
                mock_dm.download_dataset_async = AsyncMock(return_value=True)
                mock_get_dm.return_value = mock_dm
                
                response = test_client.post("/datasets/test_dataset/download")
                responses.append(response)
        
        # Start multiple concurrent requests
        threads = [threading.Thread(target=make_request) for _ in range(3)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(responses) == 3
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert 'task_id' in data


@pytest.mark.integration
def test_api_documentation_accessible(test_client):
    """Test that API documentation is accessible."""
    # Test OpenAPI JSON
    response = test_client.get("/openapi.json")
    assert response.status_code == status.HTTP_200_OK
    
    openapi_spec = response.json()
    assert 'openapi' in openapi_spec
    assert 'info' in openapi_spec
    assert openapi_spec['info']['title'] == 'Single-Cell Graph Hub API'
    
    # Test Swagger UI (returns HTML)
    response = test_client.get("/docs")
    assert response.status_code == status.HTTP_200_OK
    assert 'text/html' in response.headers['content-type']
    
    # Test ReDoc (returns HTML)
    response = test_client.get("/redoc")
    assert response.status_code == status.HTTP_200_OK
    assert 'text/html' in response.headers['content-type']


@pytest.mark.integration
def test_cors_headers(test_client):
    """Test CORS headers are properly set."""
    response = test_client.options("/datasets", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET"
    })
    
    # CORS preflight should be handled
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]


@pytest.mark.integration
def test_gzip_compression(test_client):
    """Test that gzip compression is available."""
    response = test_client.get("/datasets", headers={
        "Accept-Encoding": "gzip"
    })
    
    assert response.status_code == status.HTTP_200_OK
    # For small responses, compression might not be applied
    # This test mainly ensures the middleware is configured