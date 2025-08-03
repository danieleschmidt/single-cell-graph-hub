"""Unit tests for data manager functionality."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import torch
import numpy as np
from torch_geometric.data import Data


class TestDataManager:
    """Test cases for DataManager class."""
    
    @pytest.mark.unit
    def test_data_manager_initialization(self):
        """Test DataManager initialization."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(
                data_root=temp_dir,
                cache_dir=f"{temp_dir}/cache",
                max_workers=2,
                enable_async=True
            )
            
            assert dm.data_root == Path(temp_dir)
            assert dm.cache_dir == Path(f"{temp_dir}/cache")
            assert dm.max_workers == 2
            assert dm.enable_async is True
            
            # Check directories were created
            assert dm.data_root.exists()
            assert dm.cache_dir.exists()
    
    @pytest.mark.unit
    def test_get_dataset_path(self):
        """Test dataset path generation."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Test different file types
            raw_path = dm.get_dataset_path('test_dataset', 'raw')
            processed_path = dm.get_dataset_path('test_dataset', 'processed')
            cache_path = dm.get_dataset_path('test_dataset', 'cache')
            
            assert raw_path.name == 'test_dataset.h5ad'
            assert processed_path.name == 'test_dataset.pt'
            assert cache_path.name == 'test_dataset.pt'
            
            assert 'raw' in str(raw_path)
            assert 'processed' in str(processed_path)
            assert 'cache' in str(cache_path)
    
    @pytest.mark.unit
    def test_invalid_file_type(self):
        """Test error handling for invalid file types."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            with pytest.raises(ValueError):
                dm.get_dataset_path('test_dataset', 'invalid_type')
    
    @pytest.mark.unit
    def test_is_dataset_available(self):
        """Test dataset availability checking."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Initially, dataset should not be available
            assert not dm.is_dataset_available('test_dataset')
            
            # Create a test file
            processed_path = dm.get_dataset_path('test_dataset', 'processed')
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            processed_path.touch()
            
            # Now dataset should be available
            assert dm.is_dataset_available('test_dataset')
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test DataManager async context manager."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir, enable_async=True)
            
            async with dm:
                assert dm._session is not None
            
            # Session should be closed after exiting context
            assert dm._session is None or dm._session.closed
    
    @pytest.mark.unit
    @patch('scgraph_hub.data_manager.aiohttp.ClientSession')
    @pytest.mark.asyncio
    async def test_download_dataset_async_success(self, mock_session_class):
        """Test successful async dataset download."""
        from scgraph_hub.data_manager import DataManager
        
        # Mock aiohttp session and response
        mock_response = AsyncMock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.content.iter_chunked.return_value = [b'test_data'] * 10
        mock_response.raise_for_status.return_value = None
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir, enable_async=True)
            
            # Mock catalog info
            with patch.object(dm.catalog, 'get_info') as mock_get_info:
                mock_get_info.return_value = {
                    'url': 'http://example.com/dataset.h5ad',
                    'checksum': 'test_checksum'
                }
                
                # Mock file checksum verification
                with patch('scgraph_hub.data_manager.compute_file_checksum') as mock_checksum:
                    mock_checksum.return_value = 'test_checksum'
                    
                    async with dm:
                        result = await dm.download_dataset_async(
                            'test_dataset',
                            verify_checksum=True
                        )
                    
                    assert result is True
                    assert mock_get_info.called
    
    @pytest.mark.unit
    @patch('scgraph_hub.data_manager.aiohttp.ClientSession')
    @pytest.mark.asyncio
    async def test_download_dataset_async_failure(self, mock_session_class):
        """Test failed async dataset download."""
        from scgraph_hub.data_manager import DataManager
        
        # Mock session that raises an exception
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Download failed")
        mock_session_class.return_value = mock_session
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir, enable_async=True)
            
            # Mock catalog info
            with patch.object(dm.catalog, 'get_info') as mock_get_info:
                mock_get_info.return_value = {
                    'url': 'http://example.com/dataset.h5ad'
                }
                
                async with dm:
                    result = await dm.download_dataset_async('test_dataset')
                
                assert result is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_preprocess_dataset_async(self):
        """Test async dataset preprocessing."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Create mock raw file
            raw_path = dm.get_dataset_path('test_dataset', 'raw')
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.touch()
            
            # Mock preprocessing function
            with patch('scgraph_hub.data_manager.preprocess_dataset') as mock_preprocess:
                mock_preprocess.return_value = {'status': 'success'}
                
                result = await dm.preprocess_dataset_async(
                    'test_dataset',
                    preprocessing_config={'steps': ['normalize_total']}
                )
                
                assert result is True
                assert mock_preprocess.called
    
    @pytest.mark.unit
    def test_load_dataset(self):
        """Test dataset loading."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Create mock processed file
            processed_path = dm.get_dataset_path('test_dataset', 'processed')
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create test PyG data
            test_data = Data(
                x=torch.randn(10, 5),
                edge_index=torch.randint(0, 10, (2, 20))
            )
            torch.save(test_data, processed_path)
            
            # Load dataset
            loaded_data = dm.load_dataset('test_dataset', use_cache=False)
            
            assert loaded_data is not None
            assert isinstance(loaded_data, Data)
            assert loaded_data.x.shape == (10, 5)
    
    @pytest.mark.unit
    def test_load_dataset_not_found(self):
        """Test loading non-existent dataset."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            loaded_data = dm.load_dataset('nonexistent_dataset')
            assert loaded_data is None
    
    @pytest.mark.unit
    def test_create_dataloader(self):
        """Test DataLoader creation."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Mock load_dataset to return test data
            test_data = Data(
                x=torch.randn(10, 5),
                edge_index=torch.randint(0, 10, (2, 20))
            )
            
            with patch.object(dm, 'load_dataset') as mock_load:
                mock_load.return_value = test_data
                
                dataloader = dm.create_dataloader(
                    'test_dataset',
                    batch_size=2,
                    shuffle=True
                )
                
                assert dataloader is not None
                assert mock_load.called
    
    @pytest.mark.unit
    def test_create_dataloader_multiple_datasets(self):
        """Test DataLoader creation with multiple datasets."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Mock load_dataset to return test data
            test_data = Data(
                x=torch.randn(10, 5),
                edge_index=torch.randint(0, 10, (2, 20))
            )
            
            with patch.object(dm, 'load_dataset') as mock_load:
                mock_load.return_value = test_data
                
                dataloader = dm.create_dataloader(
                    ['dataset1', 'dataset2'],
                    batch_size=2
                )
                
                assert dataloader is not None
                assert mock_load.call_count == 2
    
    @pytest.mark.unit
    def test_get_dataset_statistics(self):
        """Test dataset statistics computation."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Create test data with known properties
            test_data = Data(
                x=torch.randn(100, 50),
                edge_index=torch.randint(0, 100, (2, 200)),
                y=torch.randint(0, 5, (100,))
            )
            
            with patch.object(dm, 'load_dataset') as mock_load:
                mock_load.return_value = test_data
                
                stats = dm.get_dataset_statistics('test_dataset')
                
                assert stats is not None
                assert 'basic' in stats
                assert 'graph' in stats
                assert 'features' in stats
                
                # Check basic statistics
                assert stats['basic']['num_nodes'] == 100
                assert stats['basic']['num_edges'] == 200
                assert stats['basic']['has_labels'] is True
                assert stats['basic']['num_classes'] == 5
    
    @pytest.mark.unit
    def test_cleanup_cache(self):
        """Test cache cleanup functionality."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Create some cache files
            cache_file1 = dm.cache_dir / 'old_file.pt'
            cache_file2 = dm.cache_dir / 'new_file.pt'
            
            cache_file1.touch()
            cache_file2.touch()
            
            # Mock file modification times
            import os
            import time
            
            # Make one file "old"
            old_time = time.time() - 25 * 3600  # 25 hours ago
            os.utime(cache_file1, (old_time, old_time))
            
            # Clean cache with 24 hour threshold
            dm.cleanup_cache(max_age_hours=24)
            
            # Old file should be removed, new file should remain
            assert not cache_file1.exists()
            assert cache_file2.exists()
    
    @pytest.mark.unit
    def test_get_storage_info(self):
        """Test storage information retrieval."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Create some test files
            (dm.data_root / 'raw').mkdir(exist_ok=True)
            (dm.data_root / 'processed').mkdir(exist_ok=True)
            
            test_file = dm.data_root / 'raw' / 'test.h5ad'
            test_file.write_text('test data')
            
            storage_info = dm.get_storage_info()
            
            assert 'data_root' in storage_info
            assert 'cache_dir' in storage_info
            assert 'raw_data_size_mb' in storage_info
            assert 'processed_data_size_mb' in storage_info
            assert 'cache_size_mb' in storage_info
            assert 'total_size_mb' in storage_info
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_download_datasets(self):
        """Test batch dataset downloading."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir, enable_async=True)
            
            # Mock download_dataset_async
            with patch.object(dm, 'download_dataset_async') as mock_download:
                mock_download.return_value = True
                
                async with dm:
                    results = await dm.batch_download_datasets(
                        ['dataset1', 'dataset2', 'dataset3'],
                        max_concurrent=2
                    )
                
                assert len(results) == 3
                assert all(success for success in results.values())
                assert mock_download.call_count == 3
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_preprocess_datasets(self):
        """Test batch dataset preprocessing."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Mock preprocess_dataset_async
            with patch.object(dm, 'preprocess_dataset_async') as mock_preprocess:
                mock_preprocess.return_value = True
                
                results = await dm.batch_preprocess_datasets(
                    ['dataset1', 'dataset2'],
                    preprocessing_configs={
                        'dataset1': {'steps': ['normalize_total']},
                        'dataset2': {'steps': ['log1p']}
                    },
                    max_concurrent=1
                )
                
                assert len(results) == 2
                assert all(success for success in results.values())
                assert mock_preprocess.call_count == 2


class TestDataManagerUtilities:
    """Test cases for data manager utility functions."""
    
    @pytest.mark.unit
    def test_get_data_manager_singleton(self):
        """Test global data manager instance."""
        from scgraph_hub.data_manager import get_data_manager
        
        # Get instance twice
        dm1 = get_data_manager()
        dm2 = get_data_manager()
        
        # Should be the same instance
        assert dm1 is dm2
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience functions."""
        from scgraph_hub.data_manager import download_dataset, preprocess_dataset_async, load_dataset
        
        # Mock the data manager functions
        with patch('scgraph_hub.data_manager.get_data_manager') as mock_get_dm:
            mock_dm = Mock()
            mock_dm.__aenter__ = AsyncMock(return_value=mock_dm)
            mock_dm.__aexit__ = AsyncMock(return_value=None)
            mock_dm.download_dataset_async = AsyncMock(return_value=True)
            mock_dm.preprocess_dataset_async = AsyncMock(return_value=True)
            mock_dm.load_dataset = Mock(return_value=Mock())
            
            mock_get_dm.return_value = mock_dm
            
            # Test convenience functions
            download_result = await download_dataset('test_dataset')
            preprocess_result = await preprocess_dataset_async('test_dataset')
            load_result = load_dataset('test_dataset')
            
            assert download_result is True
            assert preprocess_result is True
            assert load_result is not None


class TestDataManagerErrorHandling:
    """Test cases for error handling in DataManager."""
    
    @pytest.mark.unit
    def test_load_dataset_corrupted_file(self):
        """Test loading corrupted dataset file."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Create corrupted file
            processed_path = dm.get_dataset_path('corrupted_dataset', 'processed')
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            processed_path.write_text('corrupted data')
            
            # Should handle corruption gracefully
            loaded_data = dm.load_dataset('corrupted_dataset')
            assert loaded_data is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_download_without_session(self):
        """Test download fallback when session is not available."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir, enable_async=False)
            
            # Mock sync download
            with patch.object(dm, '_download_dataset_sync') as mock_sync_download:
                mock_sync_download.return_value = True
                
                result = await dm.download_dataset_async('test_dataset')
                
                assert result is True
                assert mock_sync_download.called
    
    @pytest.mark.unit
    def test_statistics_computation_error(self):
        """Test error handling in statistics computation."""
        from scgraph_hub.data_manager import DataManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DataManager(data_root=temp_dir)
            
            # Mock load_dataset to raise exception
            with patch.object(dm, 'load_dataset') as mock_load:
                mock_load.side_effect = Exception("Load failed")
                
                stats = dm.get_dataset_statistics('problematic_dataset')
                assert stats is None


@pytest.mark.unit
def test_data_manager_imports():
    """Test that data manager module imports correctly."""
    try:
        from scgraph_hub.data_manager import (
            DataManager, get_data_manager, download_dataset,
            preprocess_dataset_async, load_dataset, create_dataloader
        )
        
        assert DataManager is not None
        assert get_data_manager is not None
        assert download_dataset is not None
        assert preprocess_dataset_async is not None
        assert load_dataset is not None
        assert create_dataloader is not None
    except ImportError as e:
        pytest.fail(f"Failed to import data manager components: {e}")


@pytest.mark.unit
def test_data_manager_configuration():
    """Test data manager configuration options."""
    from scgraph_hub.data_manager import DataManager
    
    # Test default configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        dm = DataManager(data_root=temp_dir)
        
        assert dm.max_workers >= 1
        assert dm.enable_async in [True, False]
        assert dm.data_root.exists()
        assert dm.cache_dir.exists()


@pytest.mark.unit 
def test_thread_pool_management():
    """Test thread pool creation and shutdown."""
    from scgraph_hub.data_manager import DataManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dm = DataManager(data_root=temp_dir, max_workers=2)
        
        assert dm.thread_pool is not None
        assert dm.process_pool is not None
        assert dm.thread_pool._max_workers == 2
        assert dm.process_pool._max_workers == 2