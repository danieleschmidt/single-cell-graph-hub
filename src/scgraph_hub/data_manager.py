"""Advanced data management system for Single-Cell Graph Hub."""

import os
import logging
import asyncio
import aiofiles
import aiohttp
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncIterator
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import shutil
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from .database import get_dataset_repository, get_cache_manager, compute_file_checksum
from .catalog import DatasetCatalog
from .preprocessing import preprocess_dataset

logger = logging.getLogger(__name__)


class DataManager:
    """Central data management system for datasets, caching, and processing."""
    
    def __init__(self, 
                 data_root: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 max_workers: int = 4,
                 enable_async: bool = True):
        """Initialize data manager.
        
        Args:
            data_root: Root directory for data storage
            cache_dir: Directory for caching processed data
            max_workers: Maximum number of worker processes
            enable_async: Whether to enable async operations
        """
        self.data_root = Path(data_root or os.getenv('DATA_ROOT_DIR', './data'))
        self.cache_dir = Path(cache_dir or os.getenv('CACHE_DIR', './cache'))
        self.max_workers = max_workers
        self.enable_async = enable_async
        
        # Create directories
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.catalog = DatasetCatalog()
        self.repository = get_dataset_repository()
        self.cache = get_cache_manager()
        
        # Thread/process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        
        # Download session for async operations
        self._session = None
        
        logger.info(f"DataManager initialized with data_root={self.data_root}, cache_dir={self.cache_dir}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.enable_async:
            self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
        
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
    
    def get_dataset_path(self, dataset_name: str, file_type: str = 'processed') -> Path:
        """Get path for a dataset file.
        
        Args:
            dataset_name: Name of the dataset
            file_type: Type of file ('raw', 'processed', 'cache')
            
        Returns:
            Path to the dataset file
        """
        if file_type == 'raw':
            return self.data_root / 'raw' / f"{dataset_name}.h5ad"
        elif file_type == 'processed':
            return self.data_root / 'processed' / f"{dataset_name}.pt"
        elif file_type == 'cache':
            return self.cache_dir / f"{dataset_name}.pt"
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def is_dataset_available(self, dataset_name: str, file_type: str = 'processed') -> bool:
        """Check if dataset is available locally.
        
        Args:
            dataset_name: Name of the dataset
            file_type: Type of file to check
            
        Returns:
            True if dataset is available
        """
        return self.get_dataset_path(dataset_name, file_type).exists()
    
    async def download_dataset_async(self, 
                                   dataset_name: str, 
                                   force_redownload: bool = False,
                                   verify_checksum: bool = True,
                                   progress_callback: Optional[callable] = None) -> bool:
        """Download dataset asynchronously.
        
        Args:
            dataset_name: Name of the dataset to download
            force_redownload: Whether to force redownload if file exists
            verify_checksum: Whether to verify file checksum
            progress_callback: Callback for download progress
            
        Returns:
            True if download was successful
        """
        if not self.enable_async or not self._session:
            # Fallback to sync download
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self._download_dataset_sync,
                dataset_name, force_redownload, verify_checksum
            )
        
        try:
            # Get dataset info
            dataset_info = self.catalog.get_info(dataset_name)
            download_url = dataset_info.get('url')
            
            if not download_url:
                logger.error(f"No download URL found for dataset {dataset_name}")
                return False
            
            # Check if file already exists
            raw_path = self.get_dataset_path(dataset_name, 'raw')
            if raw_path.exists() and not force_redownload:
                logger.info(f"Dataset {dataset_name} already exists")
                return True
            
            # Create directory
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            logger.info(f"Downloading {dataset_name} from {download_url}")
            
            async with self._session.get(download_url) as response:
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                async with aiofiles.open(raw_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = downloaded / total_size
                            progress_callback(dataset_name, progress)
            
            # Verify checksum
            if verify_checksum and 'checksum' in dataset_info:
                actual_checksum = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, compute_file_checksum, str(raw_path)
                )
                
                expected_checksum = dataset_info['checksum']
                if actual_checksum != expected_checksum:
                    logger.error(f"Checksum verification failed for {dataset_name}")
                    raw_path.unlink()  # Remove corrupted file
                    return False
            
            # Update database
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.repository.log_processing_operation,
                dataset_name, 'download', 'completed'
            )
            
            logger.info(f"Successfully downloaded {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            
            # Log failure
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.repository.log_processing_operation,
                dataset_name, 'download', 'failed', error_message=str(e)
            )
            
            return False
    
    def _download_dataset_sync(self, 
                              dataset_name: str, 
                              force_redownload: bool = False,
                              verify_checksum: bool = True) -> bool:
        """Synchronous dataset download."""
        return self.catalog.download_dataset(
            dataset_name, 
            str(self.get_dataset_path(dataset_name, 'raw').parent),
            verify_checksum=verify_checksum
        )
    
    async def preprocess_dataset_async(self,
                                     dataset_name: str,
                                     preprocessing_config: Optional[Dict[str, Any]] = None,
                                     force_reprocess: bool = False) -> bool:
        """Preprocess dataset asynchronously.
        
        Args:
            dataset_name: Name of the dataset
            preprocessing_config: Preprocessing configuration
            force_reprocess: Whether to force reprocessing
            
        Returns:
            True if preprocessing was successful
        """
        processed_path = self.get_dataset_path(dataset_name, 'processed')
        
        # Check if already processed
        if processed_path.exists() and not force_reprocess:
            logger.info(f"Dataset {dataset_name} already processed")
            return True
        
        # Ensure raw data is available
        raw_path = self.get_dataset_path(dataset_name, 'raw')
        if not raw_path.exists():
            logger.info(f"Raw data not found, downloading {dataset_name}")
            success = await self.download_dataset_async(dataset_name)
            if not success:
                return False
        
        # Run preprocessing in process pool
        try:
            config = preprocessing_config or {}
            
            metadata = await asyncio.get_event_loop().run_in_executor(
                self.process_pool,
                preprocess_dataset,
                dataset_name,
                str(raw_path),
                str(processed_path),
                config.get('steps'),
                config.get('parameters'),
                config.get('graph_method', 'knn'),
                config.get('graph_parameters'),
                config.get('save_intermediate', False)
            )
            
            logger.info(f"Successfully preprocessed {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to preprocess {dataset_name}: {e}")
            return False
    
    def load_dataset(self, 
                    dataset_name: str, 
                    device: str = 'cpu',
                    use_cache: bool = True) -> Optional[Data]:
        """Load processed dataset.
        
        Args:
            dataset_name: Name of the dataset
            device: Device to load data on
            use_cache: Whether to use cache
            
        Returns:
            PyTorch Geometric Data object
        """
        # Check cache first
        if use_cache:
            cache_key = f"dataset_data:{dataset_name}:{device}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.debug(f"Loading {dataset_name} from cache")
                return cached_data
        
        # Load from file
        processed_path = self.get_dataset_path(dataset_name, 'processed')
        
        if not processed_path.exists():
            logger.error(f"Processed dataset {dataset_name} not found")
            return None
        
        try:
            data = torch.load(processed_path, map_location=device)
            
            # Cache the data
            if use_cache:
                cache_key = f"dataset_data:{dataset_name}:{device}"
                self.cache.set(cache_key, data, ttl=3600)  # 1 hour
            
            # Update access tracking
            self.repository.update_dataset(
                dataset_name, 
                {'last_accessed': datetime.utcnow()}
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return None
    
    def create_dataloader(self,
                         dataset_names: Union[str, List[str]],
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 0,
                         split: Optional[str] = None,
                         device: str = 'cpu') -> Optional[DataLoader]:
        """Create PyTorch Geometric DataLoader.
        
        Args:
            dataset_names: Name(s) of datasets to load
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            split: Data split to use ('train', 'val', 'test')
            device: Device to load data on
            
        Returns:
            DataLoader instance
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        # Load datasets
        datasets = []
        for name in dataset_names:
            data = self.load_dataset(name, device=device)
            if data is not None:
                # Handle splits if specified
                if split and hasattr(data, f'{split}_mask'):
                    mask = getattr(data, f'{split}_mask')
                    # Create subset based on mask
                    # This is simplified - would need proper implementation
                    datasets.append(data)
                else:
                    datasets.append(data)
            else:
                logger.warning(f"Failed to load dataset {name}")
        
        if not datasets:
            logger.error("No datasets loaded successfully")
            return None
        
        try:
            return DataLoader(
                datasets,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            return None
    
    def get_dataset_statistics(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive statistics for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing dataset statistics
        """
        # Check cache first
        cache_key = f"dataset_stats:{dataset_name}"
        cached_stats = self.cache.get(cache_key)
        if cached_stats:
            return cached_stats
        
        # Load dataset
        data = self.load_dataset(dataset_name, use_cache=False)
        if data is None:
            return None
        
        try:
            stats = {
                'basic': {
                    'num_nodes': data.num_nodes,
                    'num_edges': data.num_edges,
                    'num_features': data.num_features if hasattr(data, 'num_features') else data.x.shape[1],
                    'has_labels': hasattr(data, 'y') and data.y is not None,
                    'num_classes': len(torch.unique(data.y)) if hasattr(data, 'y') and data.y is not None else 0
                },
                'graph': {
                    'density': data.num_edges / (data.num_nodes * (data.num_nodes - 1)) if data.num_nodes > 1 else 0,
                    'average_degree': data.num_edges / data.num_nodes if data.num_nodes > 0 else 0,
                    'is_undirected': self._is_undirected(data.edge_index)
                },
                'features': {
                    'feature_mean': float(torch.mean(data.x)) if hasattr(data, 'x') else None,
                    'feature_std': float(torch.std(data.x)) if hasattr(data, 'x') else None,
                    'sparsity': float(torch.mean((data.x == 0).float())) if hasattr(data, 'x') else None
                }
            }
            
            # Add split information if available
            if hasattr(data, 'train_mask'):
                stats['splits'] = {
                    'train_nodes': int(torch.sum(data.train_mask)),
                    'val_nodes': int(torch.sum(data.val_mask)) if hasattr(data, 'val_mask') else 0,
                    'test_nodes': int(torch.sum(data.test_mask)) if hasattr(data, 'test_mask') else 0
                }
            
            # Cache statistics
            self.cache.set(cache_key, stats, ttl=7200)  # 2 hours
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute statistics for {dataset_name}: {e}")
            return None
    
    def _is_undirected(self, edge_index: torch.Tensor) -> bool:
        """Check if graph is undirected."""
        from torch_geometric.utils import is_undirected
        return is_undirected(edge_index)
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached files.
        
        Args:
            max_age_hours: Maximum age of cache files in hours
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for cache_file in self.cache_dir.glob('*.pt'):
            if cache_file.stat().st_mtime < cutoff_time.timestamp():
                try:
                    cache_file.unlink()
                    logger.debug(f"Removed old cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage usage information.
        
        Returns:
            Dictionary with storage information
        """
        def get_dir_size(path: Path) -> int:
            """Get total size of directory in bytes."""
            total = 0
            for file in path.rglob('*'):
                if file.is_file():
                    total += file.stat().st_size
            return total
        
        try:
            raw_size = get_dir_size(self.data_root / 'raw') if (self.data_root / 'raw').exists() else 0
            processed_size = get_dir_size(self.data_root / 'processed') if (self.data_root / 'processed').exists() else 0
            cache_size = get_dir_size(self.cache_dir) if self.cache_dir.exists() else 0
            
            return {
                'data_root': str(self.data_root),
                'cache_dir': str(self.cache_dir),
                'raw_data_size_mb': raw_size / (1024 * 1024),
                'processed_data_size_mb': processed_size / (1024 * 1024),
                'cache_size_mb': cache_size / (1024 * 1024),
                'total_size_mb': (raw_size + processed_size + cache_size) / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {}
    
    async def batch_download_datasets(self,
                                    dataset_names: List[str],
                                    max_concurrent: int = 3,
                                    progress_callback: Optional[callable] = None) -> Dict[str, bool]:
        """Download multiple datasets concurrently.
        
        Args:
            dataset_names: List of dataset names to download
            max_concurrent: Maximum concurrent downloads
            progress_callback: Callback for progress updates
            
        Returns:
            Dictionary mapping dataset names to success status
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(name: str) -> Tuple[str, bool]:
            async with semaphore:
                success = await self.download_dataset_async(
                    name, progress_callback=progress_callback
                )
                return name, success
        
        # Create tasks for all downloads
        tasks = [download_with_semaphore(name) for name in dataset_names]
        
        # Wait for all downloads to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        download_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Download task failed: {result}")
                continue
            
            name, success = result
            download_results[name] = success
        
        return download_results
    
    async def batch_preprocess_datasets(self,
                                      dataset_names: List[str],
                                      preprocessing_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                                      max_concurrent: int = 2) -> Dict[str, bool]:
        """Preprocess multiple datasets concurrently.
        
        Args:
            dataset_names: List of dataset names to preprocess
            preprocessing_configs: Preprocessing configurations per dataset
            max_concurrent: Maximum concurrent preprocessing jobs
            
        Returns:
            Dictionary mapping dataset names to success status
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        configs = preprocessing_configs or {}
        
        async def preprocess_with_semaphore(name: str) -> Tuple[str, bool]:
            async with semaphore:
                config = configs.get(name, {})
                success = await self.preprocess_dataset_async(name, config)
                return name, success
        
        # Create tasks for all preprocessing
        tasks = [preprocess_with_semaphore(name) for name in dataset_names]
        
        # Wait for all preprocessing to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        preprocess_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Preprocessing task failed: {result}")
                continue
            
            name, success = result
            preprocess_results[name] = success
        
        return preprocess_results


# Global data manager instance
_data_manager = None


def get_data_manager(**kwargs) -> DataManager:
    """Get the global data manager instance."""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager(**kwargs)
    return _data_manager


# Convenience functions
async def download_dataset(dataset_name: str, **kwargs) -> bool:
    """Convenience function to download a dataset."""
    async with get_data_manager() as dm:
        return await dm.download_dataset_async(dataset_name, **kwargs)


async def preprocess_dataset_async(dataset_name: str, **kwargs) -> bool:
    """Convenience function to preprocess a dataset."""
    async with get_data_manager() as dm:
        return await dm.preprocess_dataset_async(dataset_name, **kwargs)


def load_dataset(dataset_name: str, **kwargs) -> Optional[Data]:
    """Convenience function to load a dataset."""
    dm = get_data_manager()
    return dm.load_dataset(dataset_name, **kwargs)


def create_dataloader(dataset_names: Union[str, List[str]], **kwargs) -> Optional[DataLoader]:
    """Convenience function to create a DataLoader."""
    dm = get_data_manager()
    return dm.create_dataloader(dataset_names, **kwargs)