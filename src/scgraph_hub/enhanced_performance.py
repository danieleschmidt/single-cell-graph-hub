"""Generation 3: Enhanced Performance Optimization and Scaling Framework."""

import asyncio
import concurrent.futures
import functools
import hashlib
import json
import os
import pickle
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import queue


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def touch(self):
        """Update last access time and count."""
        self.access_count += 1
        self.last_access = time.time()


class AdvancedCache:
    """High-performance cache with multiple eviction strategies."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = deque()
        self._lock = threading.RLock()
        
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode())
    
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        while len(self._cache) >= self.max_size:
            if not self._access_order:
                break
            
            oldest_key = self._access_order.popleft()
            if oldest_key in self._cache:
                self._remove_entry(oldest_key)
    
    def _evict_memory(self):
        """Evict entries to free memory."""
        current_memory = sum(entry.size_bytes for entry in self._cache.values())
        
        while current_memory > self.max_memory_bytes and self._cache:
            # Find least frequently used entry
            lfu_key = min(self._cache.keys(), 
                         key=lambda k: self._cache[k].access_count)
            current_memory -= self._cache[lfu_key].size_bytes
            self._remove_entry(lfu_key)
    
    def _remove_entry(self, key: str):
        """Remove cache entry."""
        if key in self._cache:
            del self._cache[key]
            self._stats['evictions'] += 1
        
        # Remove from access order
        try:
            self._access_order.remove(key)
        except ValueError:
            pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            # Clean expired entries periodically
            if len(self._cache) % 100 == 0:
                self._evict_expired()
            
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if time.time() - entry.timestamp > self.ttl_seconds:
                self._remove_entry(key)
                self._stats['misses'] += 1
                return None
            
            # Update access info
            entry.touch()
            
            # Update LRU order
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
            self._access_order.append(key)
            
            self._stats['hits'] += 1
            return entry.value
    
    def put(self, key: str, value: Any) -> bool:
        """Put value in cache."""
        with self._lock:
            size_bytes = self._calculate_size(value)
            
            # Check if single item is too large
            if size_bytes > self.max_memory_bytes:
                return False
            
            # Remove existing entry
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict if necessary
            self._evict_lru()
            self._evict_memory()
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._access_order.append(key)
            
            return True
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'memory_usage': 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            memory_usage = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'memory_usage_bytes': memory_usage,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024)
            }


def cached(cache_key_func: Optional[Callable] = None, ttl: int = 3600):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = AdvancedCache(ttl_seconds=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(*args, **kwargs)
            else:
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache.put(key, result)
            
            return result
        
        wrapper._cache = cache
        return wrapper
    
    return decorator


class ConcurrentProcessor:
    """High-performance concurrent processing system."""
    
    def __init__(self, max_workers: int = None, worker_type: str = "thread"):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.worker_type = worker_type
        
        if worker_type == "thread":
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        elif worker_type == "process":
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            raise ValueError("worker_type must be 'thread' or 'process'")
        
        self._active_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._lock = threading.Lock()
    
    def submit_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task for concurrent execution."""
        with self._lock:
            self._active_tasks += 1
        
        def task_wrapper():
            try:
                result = func(*args, **kwargs)
                with self._lock:
                    self._completed_tasks += 1
                return result
            except Exception as e:
                with self._lock:
                    self._failed_tasks += 1
                raise e
            finally:
                with self._lock:
                    self._active_tasks -= 1
        
        return self.executor.submit(task_wrapper)
    
    def map_concurrent(self, func: Callable, items: List[Any], 
                      chunk_size: Optional[int] = None) -> List[Any]:
        """Map function over items concurrently."""
        if chunk_size is None:
            chunk_size = max(1, len(items) // self.max_workers)
        
        # Split items into chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        def process_chunk(chunk):
            return [func(item) for item in chunk]
        
        # Process chunks concurrently
        futures = [self.submit_task(process_chunk, chunk) for chunk in chunks]
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            return {
                'max_workers': self.max_workers,
                'worker_type': self.worker_type,
                'active_tasks': self._active_tasks,
                'completed_tasks': self._completed_tasks,
                'failed_tasks': self._failed_tasks,
                'total_tasks': self._completed_tasks + self._failed_tasks
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)


class ResourcePool:
    """Generic resource pool for expensive objects."""
    
    def __init__(self, factory: Callable, max_size: int = 10, 
                 timeout: float = 30.0, validator: Optional[Callable] = None):
        self.factory = factory
        self.max_size = max_size
        self.timeout = timeout
        self.validator = validator or (lambda x: True)
        
        self._pool = queue.Queue(maxsize=max_size)
        self._created_count = 0
        self._acquired_count = 0
        self._lock = threading.Lock()
    
    def acquire(self) -> Any:
        """Acquire a resource from the pool."""
        try:
            # Try to get existing resource
            resource = self._pool.get_nowait()
            
            # Validate resource
            if self.validator(resource):
                with self._lock:
                    self._acquired_count += 1
                return resource
            else:
                # Resource invalid, create new one
                resource = self._create_resource()
        
        except queue.Empty:
            # No resources available, create new one
            resource = self._create_resource()
        
        with self._lock:
            self._acquired_count += 1
        return resource
    
    def release(self, resource: Any) -> bool:
        """Release a resource back to the pool."""
        try:
            if self.validator(resource):
                self._pool.put_nowait(resource)
                return True
            else:
                return False
        except queue.Full:
            return False
    
    def _create_resource(self) -> Any:
        """Create a new resource."""
        with self._lock:
            if self._created_count >= self.max_size:
                raise RuntimeError("Maximum pool size exceeded")
            
            resource = self.factory()
            self._created_count += 1
            return resource
    
    @contextmanager
    def get_resource(self):
        """Context manager for resource acquisition."""
        resource = self.acquire()
        try:
            yield resource
        finally:
            self.release(resource)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': self._pool.qsize(),
                'max_size': self.max_size,
                'created_count': self._created_count,
                'acquired_count': self._acquired_count,
                'utilization': (self._created_count / self.max_size * 100) if self.max_size > 0 else 0
            }


class PerformanceOptimizer:
    """Comprehensive performance optimization manager."""
    
    def __init__(self):
        self.cache = AdvancedCache()
        self.processor = ConcurrentProcessor()
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._optimization_suggestions: List[Dict[str, Any]] = []
    
    def optimize_function(self, func: Callable, 
                         enable_cache: bool = True,
                         enable_concurrent: bool = False,
                         cache_ttl: int = 3600) -> Callable:
        """Optimize a function with caching and optional concurrency."""
        optimized_func = func
        
        if enable_cache:
            optimized_func = cached(ttl=cache_ttl)(optimized_func)
        
        if enable_concurrent:
            original_func = optimized_func
            
            def concurrent_wrapper(*args, **kwargs):
                return self.processor.submit_task(original_func, *args, **kwargs)
            
            optimized_func = concurrent_wrapper
        
        return optimized_func
    
    def create_resource_pool(self, name: str, factory: Callable, 
                           max_size: int = 10, validator: Optional[Callable] = None) -> ResourcePool:
        """Create a named resource pool."""
        pool = ResourcePool(factory, max_size, validator=validator)
        self.resource_pools[name] = pool
        return pool
    
    def get_resource_pool(self, name: str) -> Optional[ResourcePool]:
        """Get a named resource pool."""
        return self.resource_pools.get(name)
    
    def record_performance(self, operation: str, duration: float, 
                          metadata: Optional[Dict[str, Any]] = None):
        """Record performance metrics for analysis."""
        entry = {
            'timestamp': time.time(),
            'duration': duration,
            'metadata': metadata or {}
        }
        
        self._performance_history[operation].append(entry)
        
        # Analyze for optimization opportunities
        self._analyze_performance(operation)
    
    def _analyze_performance(self, operation: str):
        """Analyze performance and generate optimization suggestions."""
        history = list(self._performance_history[operation])
        
        if len(history) < 10:
            return
        
        recent_times = [entry['duration'] for entry in history[-10:]]
        avg_recent = sum(recent_times) / len(recent_times)
        
        all_times = [entry['duration'] for entry in history]
        avg_all = sum(all_times) / len(all_times)
        
        # Check if performance is degrading
        if avg_recent > avg_all * 1.5:
            self._optimization_suggestions.append({
                'operation': operation,
                'type': 'performance_degradation',
                'description': f'Performance degraded by {((avg_recent/avg_all - 1) * 100):.1f}%',
                'suggestion': 'Consider caching or optimization',
                'timestamp': time.time()
            })
        
        # Check for optimization opportunities
        if avg_all > 1.0:  # Operations taking more than 1 second
            self._optimization_suggestions.append({
                'operation': operation,
                'type': 'slow_operation',
                'description': f'Operation takes {avg_all:.2f}s on average',
                'suggestion': 'Consider caching, concurrency, or algorithm optimization',
                'timestamp': time.time()
            })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'cache_stats': self.cache.get_stats(),
            'processor_stats': self.processor.get_stats(),
            'resource_pools': {
                name: pool.get_stats() 
                for name, pool in self.resource_pools.items()
            },
            'operation_summary': {},
            'optimization_suggestions': self._optimization_suggestions[-10:]  # Last 10
        }
        
        # Summarize operation performance
        for operation, history in self._performance_history.items():
            if history:
                durations = [entry['duration'] for entry in history]
                report['operation_summary'][operation] = {
                    'count': len(durations),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'recent_avg': sum(durations[-10:]) / min(10, len(durations))
                }
        
        return report
    
    def auto_optimize(self) -> Dict[str, Any]:
        """Automatically apply optimizations based on analysis."""
        applied_optimizations = []
        
        # Auto-tune cache size based on hit rate
        cache_stats = self.cache.get_stats()
        if cache_stats['hit_rate'] < 50 and cache_stats['size'] < cache_stats['max_size']:
            # Increase cache size
            self.cache.max_size = min(self.cache.max_size * 2, 10000)
            applied_optimizations.append('increased_cache_size')
        
        # Auto-tune worker count based on task queue
        processor_stats = self.processor.get_stats()
        if processor_stats['active_tasks'] > processor_stats['max_workers'] * 0.8:
            # Consider increasing workers (but don't actually do it automatically)
            applied_optimizations.append('suggested_more_workers')
        
        return {
            'applied_optimizations': applied_optimizations,
            'suggestions': self._optimization_suggestions[-5:]
        }


def performance_optimized(enable_cache: bool = True, enable_concurrent: bool = False):
    """Decorator for performance optimization."""
    def decorator(func: Callable) -> Callable:
        optimizer = get_performance_optimizer()
        return optimizer.optimize_function(
            func, 
            enable_cache=enable_cache, 
            enable_concurrent=enable_concurrent
        )
    
    return decorator


# Global performance optimizer
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for performance monitoring."""
    start_time = time.time()
    optimizer = get_performance_optimizer()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        optimizer.record_performance(operation_name, duration)