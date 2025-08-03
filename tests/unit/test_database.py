"""Unit tests for database functionality."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""
    
    @pytest.mark.unit
    def test_database_manager_initialization(self):
        """Test DatabaseManager initialization."""
        from scgraph_hub.database import DatabaseManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_url = f"sqlite:///{temp_dir}/test.db"
            
            db_manager = DatabaseManager(
                database_url=db_url,
                echo=False,
                pool_size=5
            )
            
            assert db_manager.database_url == db_url
            assert db_manager.engine is not None
            assert db_manager.SessionLocal is not None
    
    @pytest.mark.unit
    def test_database_manager_default_url(self):
        """Test DatabaseManager with default URL."""
        from scgraph_hub.database import DatabaseManager
        
        # Test without DATABASE_URL environment variable
        with patch.dict(os.environ, {}, clear=True):
            if 'DATABASE_URL' in os.environ:
                del os.environ['DATABASE_URL']
            
            db_manager = DatabaseManager()
            assert 'sqlite' in db_manager.database_url
    
    @pytest.mark.unit
    def test_create_tables(self):
        """Test database table creation."""
        from scgraph_hub.database import DatabaseManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_url = f"sqlite:///{temp_dir}/test.db"
            db_manager = DatabaseManager(database_url=db_url)
            
            # Tables should be created during initialization
            # Verify by creating a session and checking if we can query
            session = db_manager.get_session()
            try:
                # This should not raise an exception if tables exist
                from scgraph_hub.database import DatasetMetadata
                result = session.query(DatasetMetadata).first()
                assert result is None  # Should be empty, but table exists
            finally:
                session.close()
    
    @pytest.mark.unit
    def test_get_session(self):
        """Test session creation."""
        from scgraph_hub.database import DatabaseManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_url = f"sqlite:///{temp_dir}/test.db"
            db_manager = DatabaseManager(database_url=db_url)
            
            session = db_manager.get_session()
            assert session is not None
            
            # Test that session is properly configured
            assert hasattr(session, 'query')
            assert hasattr(session, 'add')
            assert hasattr(session, 'commit')
            
            session.close()
    
    @pytest.mark.unit
    def test_close_connections(self):
        """Test closing database connections."""
        from scgraph_hub.database import DatabaseManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_url = f"sqlite:///{temp_dir}/test.db"
            db_manager = DatabaseManager(database_url=db_url)
            
            # Should not raise exception
            db_manager.close()


class TestDatasetMetadata:
    """Test cases for DatasetMetadata model."""
    
    @pytest.mark.unit
    def test_dataset_metadata_creation(self):
        """Test DatasetMetadata model creation."""
        from scgraph_hub.database import DatasetMetadata
        
        dataset = DatasetMetadata(
            name='test_dataset',
            description='Test dataset for unit tests',
            n_cells=1000,
            n_genes=2000,
            n_classes=5,
            modality='scRNA-seq',
            organism='human',
            tissue='blood'
        )
        
        assert dataset.name == 'test_dataset'
        assert dataset.n_cells == 1000
        assert dataset.n_genes == 2000
        assert dataset.organism == 'human'
    
    @pytest.mark.unit
    def test_dataset_metadata_to_dict(self):
        """Test DatasetMetadata to_dict method."""
        from scgraph_hub.database import DatasetMetadata
        
        dataset = DatasetMetadata(
            name='test_dataset',
            description='Test dataset',
            n_cells=1000,
            n_genes=2000,
            modality='scRNA-seq',
            created_at=datetime.utcnow()
        )
        
        data_dict = dataset.to_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict['name'] == 'test_dataset'
        assert data_dict['n_cells'] == 1000
        assert data_dict['n_genes'] == 2000
        assert 'created_at' in data_dict
    
    @pytest.mark.unit
    def test_dataset_metadata_json_fields(self):
        """Test JSON field handling in DatasetMetadata."""
        from scgraph_hub.database import DatasetMetadata
        
        preprocessing_steps = ['normalize_total', 'log1p', 'scale']
        quality_metrics = {'sparsity': 0.85, 'mean_genes_per_cell': 1500}
        supported_tasks = ['cell_type_prediction', 'trajectory_inference']
        
        dataset = DatasetMetadata(
            name='test_dataset',
            preprocessing_steps=preprocessing_steps,
            quality_metrics=quality_metrics,
            supported_tasks=supported_tasks
        )
        
        assert dataset.preprocessing_steps == preprocessing_steps
        assert dataset.quality_metrics == quality_metrics
        assert dataset.supported_tasks == supported_tasks


class TestProcessingLog:
    """Test cases for ProcessingLog model."""
    
    @pytest.mark.unit
    def test_processing_log_creation(self):
        """Test ProcessingLog model creation."""
        from scgraph_hub.database import ProcessingLog
        
        log_entry = ProcessingLog(
            dataset_id=1,
            operation='download',
            status='completed',
            duration_seconds=120.5,
            parameters={'verify_checksum': True},
            results={'file_size': 1024000}
        )
        
        assert log_entry.dataset_id == 1
        assert log_entry.operation == 'download'
        assert log_entry.status == 'completed'
        assert log_entry.duration_seconds == 120.5
        assert log_entry.parameters['verify_checksum'] is True


class TestCacheManager:
    """Test cases for CacheManager class."""
    
    @pytest.mark.unit
    def test_cache_manager_initialization_with_redis(self):
        """Test CacheManager initialization with Redis."""
        from scgraph_hub.database import CacheManager
        
        # Mock Redis to test initialization
        with patch('scgraph_hub.database.redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client
            
            cache_manager = CacheManager(
                redis_url='redis://localhost:6379/1',
                default_ttl=1800
            )
            
            assert cache_manager.redis_url == 'redis://localhost:6379/1'
            assert cache_manager.default_ttl == 1800
            assert cache_manager.redis_client is not None
    
    @pytest.mark.unit
    def test_cache_manager_fallback_to_memory(self):
        """Test CacheManager fallback to memory cache."""
        from scgraph_hub.database import CacheManager
        
        # Mock Redis connection failure
        with patch('scgraph_hub.database.redis.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Redis connection failed")
            
            cache_manager = CacheManager()
            
            assert cache_manager.redis_client is None
            assert hasattr(cache_manager, '_memory_cache')
            assert isinstance(cache_manager._memory_cache, dict)
    
    @pytest.mark.unit
    def test_cache_set_get_with_redis(self):
        """Test cache set and get operations with Redis."""
        from scgraph_hub.database import CacheManager
        
        with patch('scgraph_hub.database.redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_client.setex.return_value = True
            mock_client.get.return_value = json.dumps({'test': 'data'})
            mock_redis.return_value = mock_client
            
            cache_manager = CacheManager()
            
            # Test set operation
            result = cache_manager.set('test_key', {'test': 'data'}, ttl=60)
            assert result is True
            
            # Test get operation
            data = cache_manager.get('test_key')
            assert data == {'test': 'data'}
    
    @pytest.mark.unit
    def test_cache_set_get_with_memory(self):
        """Test cache set and get operations with memory fallback."""
        from scgraph_hub.database import CacheManager
        
        # Force memory fallback
        with patch('scgraph_hub.database.redis.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Redis unavailable")
            
            cache_manager = CacheManager()
            
            # Test set operation
            test_data = {'test': 'data', 'number': 42}
            result = cache_manager.set('test_key', test_data, ttl=60)
            assert result is True
            
            # Test get operation
            retrieved_data = cache_manager.get('test_key')
            assert retrieved_data == test_data
    
    @pytest.mark.unit
    def test_cache_expiration_with_memory(self):
        """Test cache expiration with memory fallback."""
        from scgraph_hub.database import CacheManager
        
        with patch('scgraph_hub.database.redis.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Redis unavailable")
            
            cache_manager = CacheManager()
            
            # Set data with short TTL
            cache_manager.set('test_key', 'test_data', ttl=1)
            
            # Should exist initially
            assert cache_manager.get('test_key') == 'test_data'
            
            # Mock time passing
            with patch('scgraph_hub.database.datetime') as mock_datetime:
                future_time = datetime.utcnow() + timedelta(seconds=2)
                mock_datetime.utcnow.return_value = future_time
                
                # Should be expired
                result = cache_manager.get('test_key')
                assert result is None
    
    @pytest.mark.unit
    def test_cache_delete(self):
        """Test cache delete operation."""
        from scgraph_hub.database import CacheManager
        
        with patch('scgraph_hub.database.redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_client.delete.return_value = 1
            mock_redis.return_value = mock_client
            
            cache_manager = CacheManager()
            
            result = cache_manager.delete('test_key')
            assert result is True
            assert mock_client.delete.called
    
    @pytest.mark.unit
    def test_cache_clear(self):
        """Test cache clear operation."""
        from scgraph_hub.database import CacheManager
        
        with patch('scgraph_hub.database.redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_client.flushdb.return_value = True
            mock_redis.return_value = mock_client
            
            cache_manager = CacheManager()
            
            result = cache_manager.clear()
            assert result is True
            assert mock_client.flushdb.called
    
    @pytest.mark.unit
    def test_cache_stats_redis(self):
        """Test cache statistics with Redis."""
        from scgraph_hub.database import CacheManager
        
        with patch('scgraph_hub.database.redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_client.info.return_value = {
                'connected_clients': 5,
                'used_memory': 1024000,
                'keyspace_hits': 100,
                'keyspace_misses': 10
            }
            mock_redis.return_value = mock_client
            
            cache_manager = CacheManager()
            
            stats = cache_manager.get_stats()
            assert stats['type'] == 'redis'
            assert stats['connected_clients'] == 5
            assert stats['used_memory'] == 1024000
    
    @pytest.mark.unit
    def test_cache_stats_memory(self):
        """Test cache statistics with memory fallback."""
        from scgraph_hub.database import CacheManager
        
        with patch('scgraph_hub.database.redis.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Redis unavailable")
            
            cache_manager = CacheManager()
            cache_manager.set('key1', 'data1')
            cache_manager.set('key2', 'data2')
            
            stats = cache_manager.get_stats()
            assert stats['type'] == 'memory'
            assert stats['cache_size'] == 2


class TestDatasetRepository:
    """Test cases for DatasetRepository class."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository for testing."""
        mock_db_manager = Mock()
        mock_cache_manager = Mock()
        
        from scgraph_hub.database import DatasetRepository
        return DatasetRepository(mock_db_manager, mock_cache_manager)
    
    @pytest.mark.unit
    def test_repository_initialization(self, mock_repository):
        """Test repository initialization."""
        assert mock_repository.db is not None
        assert mock_repository.cache is not None
    
    @pytest.mark.unit
    def test_get_dataset_from_cache(self, mock_repository):
        """Test getting dataset from cache."""
        # Mock cache hit
        cached_data = {'name': 'test_dataset', 'n_cells': 1000}
        mock_repository.cache.get.return_value = cached_data
        
        result = mock_repository.get_dataset('test_dataset')
        
        assert result == cached_data
        assert mock_repository.cache.get.called
        # Database should not be queried
        assert not mock_repository.db.get_session.called
    
    @pytest.mark.unit
    def test_get_dataset_from_database(self, mock_repository):
        """Test getting dataset from database when not in cache."""
        from scgraph_hub.database import DatasetMetadata
        
        # Mock cache miss
        mock_repository.cache.get.return_value = None
        
        # Mock database session and query
        mock_session = Mock()
        mock_dataset = Mock(spec=DatasetMetadata)
        mock_dataset.to_dict.return_value = {'name': 'test_dataset', 'n_cells': 1000}
        mock_dataset.last_accessed = datetime.utcnow()
        mock_dataset.access_count = 0
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_dataset
        mock_repository.db.get_session.return_value.__enter__.return_value = mock_session
        mock_repository.db.get_session.return_value.__exit__.return_value = None
        
        result = mock_repository.get_dataset('test_dataset')
        
        assert result == {'name': 'test_dataset', 'n_cells': 1000}
        assert mock_repository.cache.set.called
        assert mock_session.commit.called
    
    @pytest.mark.unit
    def test_get_dataset_not_found(self, mock_repository):
        """Test getting non-existent dataset."""
        # Mock cache miss
        mock_repository.cache.get.return_value = None
        
        # Mock database session with no result
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_repository.db.get_session.return_value.__enter__.return_value = mock_session
        mock_repository.db.get_session.return_value.__exit__.return_value = None
        
        result = mock_repository.get_dataset('nonexistent_dataset')
        
        assert result is None
    
    @pytest.mark.unit
    def test_list_datasets(self, mock_repository):
        """Test listing datasets."""
        from scgraph_hub.database import DatasetMetadata
        
        # Mock database session
        mock_session = Mock()
        mock_datasets = [Mock(spec=DatasetMetadata) for _ in range(3)]
        for i, dataset in enumerate(mock_datasets):
            dataset.to_dict.return_value = {'name': f'dataset_{i}', 'n_cells': 1000 + i}
        
        mock_session.query.return_value.offset.return_value.limit.return_value.all.return_value = mock_datasets
        mock_repository.db.get_session.return_value.__enter__.return_value = mock_session
        mock_repository.db.get_session.return_value.__exit__.return_value = None
        
        result = mock_repository.list_datasets(limit=10, offset=0)
        
        assert len(result) == 3
        assert all('name' in dataset for dataset in result)
    
    @pytest.mark.unit
    def test_list_datasets_with_filters(self, mock_repository):
        """Test listing datasets with filters."""
        mock_session = Mock()
        mock_query = Mock()
        
        # Chain the query methods
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        mock_repository.db.get_session.return_value.__enter__.return_value = mock_session
        mock_repository.db.get_session.return_value.__exit__.return_value = None
        
        filters = {
            'modality': 'scRNA-seq',
            'organism': 'human',
            'min_cells': 1000,
            'has_spatial': False
        }
        
        result = mock_repository.list_datasets(filters=filters)
        
        # Verify that filters were applied
        assert mock_query.filter.call_count >= len(filters)
    
    @pytest.mark.unit
    def test_create_dataset(self, mock_repository):
        """Test creating dataset metadata."""
        mock_session = Mock()
        mock_repository.db.get_session.return_value.__enter__.return_value = mock_session
        mock_repository.db.get_session.return_value.__exit__.return_value = None
        
        metadata = {
            'name': 'new_dataset',
            'description': 'New test dataset',
            'n_cells': 5000,
            'n_genes': 3000
        }
        
        result = mock_repository.create_dataset(metadata)
        
        assert result is True
        assert mock_session.add.called
        assert mock_session.commit.called
        assert mock_repository.cache.delete.called
    
    @pytest.mark.unit
    def test_update_dataset(self, mock_repository):
        """Test updating dataset metadata."""
        from scgraph_hub.database import DatasetMetadata
        
        mock_session = Mock()
        mock_dataset = Mock(spec=DatasetMetadata)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_dataset
        mock_repository.db.get_session.return_value.__enter__.return_value = mock_session
        mock_repository.db.get_session.return_value.__exit__.return_value = None
        
        updates = {'n_cells': 6000, 'description': 'Updated description'}
        
        result = mock_repository.update_dataset('test_dataset', updates)
        
        assert result is True
        assert mock_session.commit.called
        assert mock_repository.cache.delete.called
    
    @pytest.mark.unit
    def test_log_processing_operation(self, mock_repository):
        """Test logging processing operations."""
        from scgraph_hub.database import DatasetMetadata
        
        mock_session = Mock()
        mock_dataset = Mock(spec=DatasetMetadata)
        mock_dataset.id = 1
        mock_session.query.return_value.filter.return_value.first.return_value = mock_dataset
        mock_repository.db.get_session.return_value.__enter__.return_value = mock_session
        mock_repository.db.get_session.return_value.__exit__.return_value = None
        
        result = mock_repository.log_processing_operation(
            dataset_name='test_dataset',
            operation='preprocess',
            status='completed',
            duration_seconds=300.5,
            parameters={'steps': ['normalize_total']},
            results={'final_cells': 950}
        )
        
        assert result is True
        assert mock_session.add.called
        assert mock_session.commit.called


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    @pytest.mark.unit
    def test_compute_file_checksum(self):
        """Test file checksum computation."""
        from scgraph_hub.database import compute_file_checksum
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test content for checksum')
            temp_file = f.name
        
        try:
            checksum = compute_file_checksum(temp_file)
            
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA256 produces 64-character hex string
            
            # Same file should produce same checksum
            checksum2 = compute_file_checksum(temp_file)
            assert checksum == checksum2
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.unit
    def test_get_system_info(self):
        """Test system information collection."""
        from scgraph_hub.database import get_system_info
        
        info = get_system_info()
        
        assert isinstance(info, dict)
        assert 'hostname' in info
        assert 'python_version' in info
        assert 'platform' in info
        
        # Check that values are strings
        assert isinstance(info['hostname'], str)
        assert isinstance(info['python_version'], str)


class TestGlobalInstances:
    """Test cases for global instance functions."""
    
    @pytest.mark.unit
    def test_get_database_manager_singleton(self):
        """Test global database manager instance."""
        from scgraph_hub.database import get_database_manager
        
        # Get instance twice
        dm1 = get_database_manager()
        dm2 = get_database_manager()
        
        # Should be the same instance
        assert dm1 is dm2
    
    @pytest.mark.unit
    def test_get_cache_manager_singleton(self):
        """Test global cache manager instance."""
        from scgraph_hub.database import get_cache_manager
        
        # Get instance twice
        cm1 = get_cache_manager()
        cm2 = get_cache_manager()
        
        # Should be the same instance
        assert cm1 is cm2
    
    @pytest.mark.unit
    def test_get_dataset_repository_singleton(self):
        """Test global dataset repository instance."""
        from scgraph_hub.database import get_dataset_repository
        
        # Get instance twice
        dr1 = get_dataset_repository()
        dr2 = get_dataset_repository()
        
        # Should be the same instance
        assert dr1 is dr2


@pytest.mark.unit
def test_database_imports():
    """Test that database module imports correctly."""
    try:
        from scgraph_hub.database import (
            DatabaseManager, CacheManager, DatasetRepository,
            DatasetMetadata, ProcessingLog, ModelMetadata, ExperimentRun,
            get_database_manager, get_cache_manager, get_dataset_repository,
            compute_file_checksum, get_system_info
        )
        
        # Verify all imports are available
        assert DatabaseManager is not None
        assert CacheManager is not None
        assert DatasetRepository is not None
        assert DatasetMetadata is not None
        assert ProcessingLog is not None
        assert ModelMetadata is not None
        assert ExperimentRun is not None
        assert get_database_manager is not None
        assert get_cache_manager is not None
        assert get_dataset_repository is not None
        assert compute_file_checksum is not None
        assert get_system_info is not None
    except ImportError as e:
        pytest.fail(f"Failed to import database components: {e}")


@pytest.mark.unit
def test_database_models_structure():
    """Test database model structure and relationships."""
    from scgraph_hub.database import DatasetMetadata, ProcessingLog
    
    # Test that models have expected attributes
    dataset_attrs = ['name', 'description', 'n_cells', 'n_genes', 'modality', 'organism']
    for attr in dataset_attrs:
        assert hasattr(DatasetMetadata, attr)
    
    log_attrs = ['dataset_id', 'operation', 'status', 'started_at']
    for attr in log_attrs:
        assert hasattr(ProcessingLog, attr)