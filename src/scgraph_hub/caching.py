"""Intelligent caching system for Single-Cell Graph Hub."""

import logging
import os
import pickle
import hashlib
import time
import json
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps
import threading

import torch
import numpy as np

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, size_bytes: int = 0, ttl: Optional[int] = None):
        """Initialize cache entry.
        
        Args:
            key: Cache key
            value: Cached value
            size_bytes: Size in bytes
            ttl: Time to live in seconds
        """
        self.key = key
        self.value = value
        self.size_bytes = size_bytes
        self.created_at = time.time()
        self.accessed_at = time.time()
        self.access_count = 0
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Check if entry is expired.
        
        Returns:
            True if expired
        """
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class LRUCache:
    """Least Recently Used cache with size and TTL management."""
    
    def __init__(self, max_size_mb: float = 1024, max_entries: int = 10000):
        """Initialize LRU cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of entries
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        
        self._cache = OrderedDict()
        self._total_size = 0
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Check if single entry exceeds cache size
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Entry size ({size_bytes} bytes) exceeds cache size")
                return
            
            # Evict entries to make space
            self._evict_to_fit(size_bytes)
            
            # Add new entry
            entry = CacheEntry(key, value, size_bytes, ttl)
            self._cache[key] = entry
            self._total_size += size_bytes
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes.
        
        Args:
            value: Value to estimate
            
        Returns:
            Estimated size in bytes
        """
        try:
            if isinstance(value, torch.Tensor):
                return value.element_size() * value.nelement()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (str, bytes)):
                return len(value.encode('utf-8')) if isinstance(value, str) else len(value)
            elif isinstance(value, (list, tuple, dict)):
                # Rough estimate for containers
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            else:
                # Default estimate
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimate
            return 1024  # 1KB default
    
    def _evict_to_fit(self, required_size: int):
        """Evict entries to fit required size.
        
        Args:
            required_size: Required size in bytes
        """
        while (self._total_size + required_size > self.max_size_bytes or 
               len(self._cache) >= self.max_entries):
            
            if not self._cache:
                break
            
            # Remove least recently used entry
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._evictions += 1
    
    def _remove_entry(self, key: str):
        """Remove entry from cache.
        
        Args:
            key: Key to remove
        """
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_size -= entry.size_bytes
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_size = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'entries': len(self._cache),
                'size_mb': self._total_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'utilization': self._total_size / self.max_size_bytes if self.max_size_bytes > 0 else 0
            }


class PersistentCache:
    """Persistent cache that survives between sessions."""
    
    def __init__(self, cache_dir: Union[str, Path], max_size_mb: float = 5120):
        """Initialize persistent cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.index_file = self.cache_dir / 'cache_index.json'
        
        # Load existing index
        self.index = self._load_index()
        
        # Clean up expired entries
        self._cleanup_expired()
        
        logger.info(f"Persistent cache initialized at {self.cache_dir}")
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk.
        
        Returns:
            Cache index dictionary
        """
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")
        
        return {}
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        # Create safe filename from key
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key not in self.index:
            return None
        
        entry_info = self.index[key]
        
        # Check if expired
        if entry_info.get('ttl') and entry_info['created_at'] + entry_info['ttl'] < time.time():
            self._remove_entry(key)
            return None
        
        # Load from disk
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            # Remove from index if file doesn't exist
            self._remove_entry(key)
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            
            # Update access info
            entry_info['accessed_at'] = time.time()
            entry_info['access_count'] = entry_info.get('access_count', 0) + 1
            self._save_index()
            
            return value
        
        except Exception as e:
            logger.error(f"Failed to load cached value for key {key}: {e}")
            self._remove_entry(key)
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """Put value in persistent cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        # Remove existing entry if present
        if key in self.index:
            self._remove_entry(key)
        
        # Save to disk
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Get file size
            file_size = cache_path.stat().st_size
            
            # Update index
            self.index[key] = {
                'created_at': time.time(),
                'accessed_at': time.time(),
                'access_count': 0,
                'size_bytes': file_size,
                'ttl': ttl
            }
            
            self._save_index()
            
            # Clean up if cache is too large
            self._cleanup_size()
        
        except Exception as e:
            logger.error(f"Failed to cache value for key {key}: {e}")
            if cache_path.exists():
                cache_path.unlink()
    
    def _remove_entry(self, key: str):
        """Remove entry from cache.
        
        Args:
            key: Key to remove
        """
        if key in self.index:
            del self.index[key]
            
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry_info in self.index.items():
            if entry_info.get('ttl') and entry_info['created_at'] + entry_info['ttl'] < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            self._save_index()
            logger.info(f"Removed {len(expired_keys)} expired cache entries")
    
    def _cleanup_size(self):
        """Remove entries if cache exceeds size limit."""
        total_size = sum(entry['size_bytes'] for entry in self.index.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by access time (least recently used first)
        sorted_entries = sorted(
            self.index.items(),
            key=lambda x: x[1]['accessed_at']
        )
        
        removed_count = 0
        for key, entry_info in sorted_entries:
            if total_size <= self.max_size_bytes:
                break
            
            total_size -= entry_info['size_bytes']
            self._remove_entry(key)
            removed_count += 1
        
        if removed_count > 0:
            self._save_index()
            logger.info(f"Removed {removed_count} cache entries to fit size limit")
    
    def clear(self):
        """Clear all cache entries."""
        for key in list(self.index.keys()):
            self._remove_entry(key)
        
        self._save_index()
        logger.info("Persistent cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = sum(entry['size_bytes'] for entry in self.index.values())
        total_accesses = sum(entry.get('access_count', 0) for entry in self.index.values())
        
        return {
            'entries': len(self.index),
            'size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': total_size / self.max_size_bytes if self.max_size_bytes > 0 else 0,
            'total_accesses': total_accesses,
            'cache_dir': str(self.cache_dir)
        }


class SmartCache:
    """Smart cache that combines memory and persistent caching."""
    
    def __init__(self, 
                 cache_dir: Union[str, Path],
                 memory_size_mb: float = 512,
                 disk_size_mb: float = 5120):
        """Initialize smart cache.
        
        Args:
            cache_dir: Directory for persistent cache
            memory_size_mb: Memory cache size in MB
            disk_size_mb: Disk cache size in MB
        """
        self.memory_cache = LRUCache(max_size_mb=memory_size_mb)
        self.disk_cache = PersistentCache(cache_dir, max_size_mb=disk_size_mb)
        
        # Access patterns for intelligent caching
        self.access_patterns = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from smart cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Track access pattern
        self._update_access_pattern(key)
        
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache if frequently accessed
            if self._should_promote_to_memory(key):
                self.memory_cache.put(key, value)
            return value
        
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, force_disk: bool = False):
        """Put value in smart cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            force_disk: Force storage to disk cache
        """
        if force_disk or self._should_store_on_disk(key, value):
            self.disk_cache.put(key, value, ttl)
        else:
            self.memory_cache.put(key, value, ttl)
            # Also store on disk if value is expensive to compute
            if self._is_expensive_value(value):
                self.disk_cache.put(key, value, ttl)
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for key.
        
        Args:
            key: Cache key
        """
        current_time = time.time()
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'count': 0,
                'last_access': current_time,
                'frequency': 0.0
            }
        
        pattern = self.access_patterns[key]
        pattern['count'] += 1
        
        # Calculate access frequency (accesses per hour)
        time_diff = current_time - pattern['last_access']
        if time_diff > 0:
            pattern['frequency'] = pattern['count'] / (time_diff / 3600)
        
        pattern['last_access'] = current_time
    
    def _should_promote_to_memory(self, key: str) -> bool:
        """Check if key should be promoted to memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if should promote
        """
        if key not in self.access_patterns:
            return False
        
        pattern = self.access_patterns[key]
        return pattern['frequency'] > 5.0  # More than 5 accesses per hour
    
    def _should_store_on_disk(self, key: str, value: Any) -> bool:
        """Check if value should be stored on disk.
        
        Args:
            key: Cache key
            value: Value to store
            
        Returns:
            True if should store on disk
        """
        # Large values go to disk
        size_mb = self.memory_cache._estimate_size(value) / (1024 * 1024)
        if size_mb > 100:  # Values larger than 100MB
            return True
        
        # Infrequently accessed values go to disk
        if key in self.access_patterns:
            pattern = self.access_patterns[key]
            if pattern['frequency'] < 1.0:  # Less than 1 access per hour
                return True
        
        return False
    
    def _is_expensive_value(self, value: Any) -> bool:
        """Check if value is expensive to compute.
        
        Args:
            value: Value to check
            
        Returns:
            True if expensive
        """
        # Heuristics for expensive values
        if isinstance(value, torch.Tensor) and value.numel() > 1000000:  # Large tensors
            return True
        
        if isinstance(value, dict) and 'embeddings' in value:  # Computed embeddings
            return True
        
        return False
    
    def clear(self):
        """Clear both memory and disk cache."""
        self.memory_cache.clear()
        self.disk_cache.clear()
        self.access_patterns.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        memory_stats = self.memory_cache.stats()
        disk_stats = self.disk_cache.stats()
        
        return {
            'memory': memory_stats,
            'disk': disk_stats,
            'access_patterns': len(self.access_patterns),
            'total_entries': memory_stats['entries'] + disk_stats['entries'],
            'total_size_mb': memory_stats['size_mb'] + disk_stats['size_mb']
        }


# Decorator for automatic caching
def cached(cache_key_func: Optional[Callable] = None, ttl: Optional[int] = None, force_disk: bool = False):
    """Decorator for automatic function result caching.
    
    Args:
        cache_key_func: Function to generate cache key from arguments
        ttl: Time to live in seconds
        force_disk: Force storage to disk cache
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Get or create cache instance
        if not hasattr(func, '_cache'):
            cache_dir = Path.home() / '.scgraph_hub' / 'cache'
            func._cache = SmartCache(cache_dir)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.sha256('_'.join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = func._cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Compute result and cache it
            logger.debug(f"Cache miss for {func.__name__}, computing...")
            result = func(*args, **kwargs)
            func._cache.put(cache_key, result, ttl=ttl, force_disk=force_disk)
            
            return result
        
        # Add cache management methods
        wrapper.cache_stats = lambda: func._cache.stats()
        wrapper.cache_clear = lambda: func._cache.clear()
        
        return wrapper
    return decorator


# Global cache instance
_global_cache = None


def get_global_cache() -> SmartCache:
    """Get global cache instance.
    
    Returns:
        Global SmartCache instance
    """
    global _global_cache
    if _global_cache is None:
        cache_dir = Path.home() / '.scgraph_hub' / 'cache'
        _global_cache = SmartCache(cache_dir)
    return _global_cache


# Convenience functions
def cache_dataset(dataset_name: str, data: Any, ttl: Optional[int] = None):
    """Cache dataset data.
    
    Args:
        dataset_name: Name of the dataset
        data: Dataset data to cache
        ttl: Time to live in seconds
    """
    cache = get_global_cache()
    cache.put(f"dataset:{dataset_name}", data, ttl=ttl, force_disk=True)


def get_cached_dataset(dataset_name: str) -> Optional[Any]:
    """Get cached dataset data.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Cached dataset data or None
    """
    cache = get_global_cache()
    return cache.get(f"dataset:{dataset_name}")


def cache_model_embeddings(model_name: str, dataset_name: str, embeddings: torch.Tensor, ttl: Optional[int] = None):
    """Cache model embeddings.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        embeddings: Computed embeddings
        ttl: Time to live in seconds
    """
    cache = get_global_cache()
    cache_key = f"embeddings:{model_name}:{dataset_name}"
    cache.put(cache_key, embeddings, ttl=ttl, force_disk=True)


def get_cached_embeddings(model_name: str, dataset_name: str) -> Optional[torch.Tensor]:
    """Get cached model embeddings.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        
    Returns:
        Cached embeddings or None
    """
    cache = get_global_cache()
    cache_key = f"embeddings:{model_name}:{dataset_name}"
    return cache.get(cache_key)


def clear_all_caches():
    """Clear all caches."""
    cache = get_global_cache()
    cache.clear()
    logger.info("All caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics.
    
    Returns:
        Cache statistics
    """
    cache = get_global_cache()
    return cache.stats()
