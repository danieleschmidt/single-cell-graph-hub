"""
Performance optimization and caching system for Single-Cell Graph Hub.

This module implements high-performance data processing, intelligent caching,
and resource optimization for scalable single-cell graph analysis.
"""

import asyncio
import time
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import hashlib
import pickle
import json
from datetime import datetime, timedelta

from .exceptions import SCGraphHubError, handle_error_gracefully, create_error_context
from .logging_config import get_logger, LoggingMixin
from .health_checks import performance_timer, get_performance_monitor

# Optional imports with graceful handling
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    import joblib.memory
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logger = get_logger(__name__)


class PerformanceCache:
    """High-performance caching system with multiple backends."""
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 max_memory_cache_mb: float = 1024,
                 use_redis: bool = False,
                 redis_url: Optional[str] = None):
        """Initialize performance cache.
        
        Args:
            cache_dir: Directory for persistent cache
            max_memory_cache_mb: Maximum memory cache size in MB
            use_redis: Whether to use Redis for distributed caching
            redis_url: Redis connection URL
        """
        self.cache_dir = cache_dir or Path.home() / ".scgraph_hub" / "performance_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_cache_mb = max_memory_cache_mb
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0, 'misses': 0, 'evictions': 0,
            'memory_usage_mb': 0.0
        }
        
        # Redis setup
        self.redis_client = None
        if use_redis and _REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url or 'redis://localhost:6379')
                self.redis_client.ping()
                logger.info("Redis cache backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.redis_client = None
        
        # Joblib memory cache for expensive computations
        self.joblib_memory = None
        if _JOBLIB_AVAILABLE:
            self.joblib_memory = joblib.memory.Memory(
                location=str(self.cache_dir / "joblib"),
                verbose=0
            )
            logger.info("Joblib memory cache initialized")
    
    def _get_cache_key(self, key: str, params: Dict[str, Any] = None) -> str:
        """Generate consistent cache key."""
        if params:
            param_str = json.dumps(params, sort_keys=True, default=str)
            combined = f"{key}:{param_str}"
        else:
            combined = key
        
        return hashlib.md5(combined.encode()).hexdigest()
    
    @performance_timer("cache_get")
    def get(self, key: str, params: Dict[str, Any] = None) -> Any:
        """Get item from cache with fallback hierarchy."""
        cache_key = self._get_cache_key(key, params)
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            self.cache_stats['hits'] += 1
            logger.debug(f"Memory cache hit for {key}")
            return self.memory_cache[cache_key]['data']
        
        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data = pickle.loads(cached_data)
                    # Store in memory cache for faster subsequent access
                    self._store_in_memory(cache_key, data)
                    self.cache_stats['hits'] += 1
                    logger.debug(f"Redis cache hit for {key}")
                    return data
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # Try disk cache
        disk_file = self.cache_dir / f"{cache_key}.pkl"
        if disk_file.exists():
            try:
                with open(disk_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Store in higher-level caches
                self._store_in_memory(cache_key, data)
                if self.redis_client:
                    try:
                        self.redis_client.setex(
                            cache_key, 
                            timedelta(hours=24),
                            pickle.dumps(data)
                        )
                    except Exception as e:
                        logger.warning(f"Redis cache set failed: {e}")
                
                self.cache_stats['hits'] += 1
                logger.debug(f"Disk cache hit for {key}")
                return data
                
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
                disk_file.unlink(missing_ok=True)
        
        self.cache_stats['misses'] += 1
        return None
    
    @performance_timer("cache_set")
    def set(self, key: str, data: Any, params: Dict[str, Any] = None, 
            ttl_hours: float = 24) -> bool:
        """Set item in cache with multi-level storage."""
        cache_key = self._get_cache_key(key, params)
        
        try:
            # Store in memory cache
            self._store_in_memory(cache_key, data)
            
            # Store in Redis
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        cache_key,
                        timedelta(hours=ttl_hours),
                        pickle.dumps(data)
                    )
                except Exception as e:
                    logger.warning(f"Redis cache set failed: {e}")
            
            # Store on disk for persistence
            disk_file = self.cache_dir / f"{cache_key}.pkl"
            with open(disk_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Cached data for key {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            return False
    
    def _store_in_memory(self, cache_key: str, data: Any):
        """Store data in memory cache with size management."""
        try:
            data_size = len(pickle.dumps(data)) / (1024 * 1024)  # MB
            
            # Check if we need to evict items
            while (self.cache_stats['memory_usage_mb'] + data_size > 
                   self.max_memory_cache_mb and self.memory_cache):
                # Remove oldest item (simple FIFO)
                oldest_key = next(iter(self.memory_cache))
                oldest_size = self.memory_cache[oldest_key]['size_mb']
                del self.memory_cache[oldest_key]
                self.cache_stats['memory_usage_mb'] -= oldest_size
                self.cache_stats['evictions'] += 1
            
            # Store new item
            self.memory_cache[cache_key] = {
                'data': data,
                'timestamp': time.time(),
                'size_mb': data_size
            }
            self.cache_stats['memory_usage_mb'] += data_size
            
        except Exception as e:
            logger.warning(f"Memory cache store failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        hit_rate = (self.cache_stats['hits'] / 
                   max(1, self.cache_stats['hits'] + self.cache_stats['misses']))
        
        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'memory_cache_entries': len(self.memory_cache),
            'redis_available': self.redis_client is not None,
            'joblib_available': self.joblib_memory is not None
        }
    
    def clear(self, pattern: Optional[str] = None):
        """Clear cache entries matching pattern."""
        if pattern is None:
            # Clear all
            self.memory_cache.clear()
            self.cache_stats['memory_usage_mb'] = 0.0
            
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis cache clear failed: {e}")
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)
            
            logger.info("All caches cleared")
        else:
            # Pattern-based clearing (basic implementation)
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.memory_cache[key]


class ConcurrentProcessor:
    """High-performance concurrent processing system."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of worker processes/threads
        """
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count() + 4)
        self.process_pool = None
        self.thread_pool = None
        self.resource_monitor = ResourceOptimizer()
        
        logger.info(f"ConcurrentProcessor initialized with {self.max_workers} max workers")
    
    def __enter__(self):
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
    
    @performance_timer("concurrent_map")
    async def map_async(self, func: Callable, items: List[Any], 
                       use_processes: bool = True) -> List[Any]:
        """Execute function over items concurrently.
        
        Args:
            func: Function to execute
            items: Items to process
            use_processes: Whether to use processes vs threads
            
        Returns:
            Results in original order
        """
        if not items:
            return []
        
        executor = self.process_pool if use_processes else self.thread_pool
        
        # Submit all tasks
        future_to_index = {}
        for i, item in enumerate(items):
            future = executor.submit(func, item)
            future_to_index[future] = i
        
        # Collect results in order
        results = [None] * len(items)
        completed_count = 0
        
        for future in as_completed(future_to_index.keys()):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
                completed_count += 1
                
                # Log progress for long operations
                if len(items) > 10 and completed_count % max(1, len(items) // 10) == 0:
                    progress = completed_count / len(items) * 100
                    logger.info(f"Processing progress: {progress:.1f}% ({completed_count}/{len(items)})")
                    
            except Exception as e:
                logger.error(f"Task {index} failed: {e}")
                results[index] = None
        
        return results
    
    @performance_timer("batch_process")
    def batch_process(self, func: Callable, items: List[Any], 
                     batch_size: int = None) -> List[Any]:
        """Process items in optimally-sized batches.
        
        Args:
            func: Function to execute on each batch
            items: Items to process
            batch_size: Size of each batch (auto-determined if None)
            
        Returns:
            Flattened results
        """
        if not items:
            return []
        
        # Auto-determine optimal batch size
        if batch_size is None:
            batch_size = max(1, len(items) // (self.max_workers * 2))
            batch_size = min(batch_size, 1000)  # Cap at 1000
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        logger.info(f"Processing {len(items)} items in {len(batches)} batches of size {batch_size}")
        
        # Process batches concurrently
        if _JOBLIB_AVAILABLE:
            results = Parallel(n_jobs=self.max_workers)(
                delayed(func)(batch) for batch in batches
            )
        else:
            # Fallback to sequential processing
            results = [func(batch) for batch in batches]
        
        # Flatten results
        flattened = []
        for batch_result in results:
            if isinstance(batch_result, list):
                flattened.extend(batch_result)
            else:
                flattened.append(batch_result)
        
        return flattened


class ResourceOptimizer:
    """Dynamic resource optimization and auto-scaling."""
    
    def __init__(self):
        self.monitoring = True
        self.optimization_history = []
        self.current_limits = {
            'memory_gb': 8.0,
            'cpu_count': multiprocessing.cpu_count(),
            'io_threads': 4
        }
    
    @performance_timer("resource_optimization")
    def optimize_for_workload(self, workload_type: str, 
                            data_size_mb: float) -> Dict[str, Any]:
        """Optimize resource allocation for specific workload.
        
        Args:
            workload_type: Type of workload ('compute', 'io', 'memory')
            data_size_mb: Size of data being processed
            
        Returns:
            Optimized resource configuration
        """
        config = self.current_limits.copy()
        
        if workload_type == 'compute':
            # CPU-intensive: maximize CPU usage
            config['cpu_count'] = multiprocessing.cpu_count()
            config['memory_gb'] = min(16.0, max(2.0, data_size_mb / 1024 * 1.5))
            config['io_threads'] = 2
            
        elif workload_type == 'io':
            # I/O intensive: maximize I/O throughput
            config['io_threads'] = min(32, multiprocessing.cpu_count() * 2)
            config['cpu_count'] = multiprocessing.cpu_count() // 2
            config['memory_gb'] = max(1.0, data_size_mb / 1024)
            
        elif workload_type == 'memory':
            # Memory intensive: optimize for large data
            config['memory_gb'] = max(8.0, data_size_mb / 1024 * 2)
            config['cpu_count'] = min(multiprocessing.cpu_count(), 8)
            config['io_threads'] = 4
        
        # Check system constraints
        if _PSUTIL_AVAILABLE:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            config['memory_gb'] = min(config['memory_gb'], available_memory_gb * 0.8)
        
        logger.info(f"Optimized for {workload_type} workload: {config}")
        return config
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource metrics."""
        metrics = {}
        
        if _PSUTIL_AVAILABLE:
            metrics.update({
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_io_read_mb_s': psutil.disk_io_counters().read_bytes / (1024**2) if psutil.disk_io_counters() else 0,
                'disk_io_write_mb_s': psutil.disk_io_counters().write_bytes / (1024**2) if psutil.disk_io_counters() else 0,
                'network_sent_mb_s': psutil.net_io_counters().bytes_sent / (1024**2),
                'network_recv_mb_s': psutil.net_io_counters().bytes_recv / (1024**2),
            })
        
        return metrics
    
    def should_scale_up(self, current_metrics: Dict[str, float]) -> bool:
        """Determine if resources should be scaled up."""
        if not current_metrics:
            return False
        
        # Scale up if CPU or memory usage is high
        cpu_high = current_metrics.get('cpu_percent', 0) > 80
        memory_high = current_metrics.get('memory_percent', 0) > 85
        
        return cpu_high or memory_high
    
    def should_scale_down(self, current_metrics: Dict[str, float]) -> bool:
        """Determine if resources should be scaled down."""
        if not current_metrics:
            return False
        
        # Scale down if both CPU and memory usage are low
        cpu_low = current_metrics.get('cpu_percent', 100) < 20
        memory_low = current_metrics.get('memory_percent', 100) < 30
        
        return cpu_low and memory_low


class PerformanceOptimizer(LoggingMixin):
    """Main performance optimization coordinator."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize performance optimizer.
        
        Args:
            cache_dir: Directory for performance cache
        """
        self.cache = PerformanceCache(cache_dir)
        self.processor = None  # Lazy initialization
        self.resource_optimizer = ResourceOptimizer()
        self.optimization_enabled = True
        
        logger.info("PerformanceOptimizer initialized")
    
    def cached(self, ttl_hours: float = 24, key_func: Optional[Callable] = None):
        """Decorator for caching function results.
        
        Args:
            ttl_hours: Time to live in hours
            key_func: Function to generate cache key from args
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return func(*args, **kwargs)
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__module__}.{func.__name__}"
                
                params = {'args': str(args), 'kwargs': str(sorted(kwargs.items()))}
                
                # Try cache first
                cached_result = self.cache.get(cache_key, params)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > 1.0:  # Only cache expensive operations
                    self.cache.set(cache_key, result, params, ttl_hours)
                    logger.debug(f"Cached result for {func.__name__} (took {duration:.2f}s)")
                
                return result
            
            return wrapper
        return decorator
    
    @handle_error_gracefully
    async def optimize_dataset_processing(self, dataset_name: str, 
                                        processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize dataset processing with performance enhancements.
        
        Args:
            dataset_name: Name of dataset
            processing_config: Processing configuration
            
        Returns:
            Optimized processing results
        """
        with create_error_context(f"optimizing dataset processing for {dataset_name}"):
            # Determine workload characteristics
            data_size_mb = processing_config.get('estimated_size_mb', 1000)
            workload_type = processing_config.get('workload_type', 'compute')
            
            # Optimize resource allocation
            optimal_config = self.resource_optimizer.optimize_for_workload(
                workload_type, data_size_mb
            )
            
            # Initialize concurrent processor with optimal settings
            max_workers = int(optimal_config['cpu_count'])
            
            with ConcurrentProcessor(max_workers=max_workers) as processor:
                self.processor = processor
                
                # Execute optimized processing
                results = {
                    'dataset_name': dataset_name,
                    'optimization_applied': optimal_config,
                    'processing_time': 0,
                    'cache_stats': self.cache.get_stats()
                }
                
                start_time = time.time()
                
                # Actual processing would go here
                # For now, simulate with the existing processing pipeline
                logger.info(f"Processing {dataset_name} with {max_workers} workers")
                await asyncio.sleep(0.1)  # Simulate processing
                
                results['processing_time'] = time.time() - start_time
                
                logger.info(f"Optimized processing completed in {results['processing_time']:.2f}s")
                return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'cache_stats': self.cache.get_stats(),
            'system_metrics': self.resource_optimizer.get_system_metrics(),
            'resource_limits': self.resource_optimizer.current_limits,
            'optimization_enabled': self.optimization_enabled,
            'performance_monitor': get_performance_monitor().get_summary()
        }
    
    def enable_optimization(self):
        """Enable performance optimizations."""
        self.optimization_enabled = True
        logger.info("Performance optimizations enabled")
    
    def disable_optimization(self):
        """Disable performance optimizations."""
        self.optimization_enabled = False
        logger.info("Performance optimizations disabled")


# Global optimizer instance
_performance_optimizer = None


def get_performance_optimizer(cache_dir: Optional[Path] = None) -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(cache_dir)
    return _performance_optimizer


def high_performance(ttl_hours: float = 24):
    """Decorator to apply performance optimizations to functions."""
    def decorator(func):
        optimizer = get_performance_optimizer()
        return optimizer.cached(ttl_hours=ttl_hours)(func)
    return decorator


# Auto-scaling context manager
class auto_scale:
    """Context manager for automatic resource scaling."""
    
    def __init__(self, workload_type: str = 'compute', 
                 data_size_mb: float = 1000):
        self.workload_type = workload_type
        self.data_size_mb = data_size_mb
        self.optimizer = get_performance_optimizer()
        self.original_limits = None
    
    def __enter__(self):
        # Apply optimizations
        optimal_config = self.optimizer.resource_optimizer.optimize_for_workload(
            self.workload_type, self.data_size_mb
        )
        
        self.original_limits = self.optimizer.resource_optimizer.current_limits.copy()
        self.optimizer.resource_optimizer.current_limits.update(optimal_config)
        
        logger.info(f"Auto-scaling enabled for {self.workload_type} workload")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original limits
        if self.original_limits:
            self.optimizer.resource_optimizer.current_limits = self.original_limits
        
        logger.info("Auto-scaling disabled")