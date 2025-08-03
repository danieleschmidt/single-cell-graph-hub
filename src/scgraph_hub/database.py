"""Database layer for Single-Cell Graph Hub metadata and caching."""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import asyncio

import sqlite3
import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Text, 
    DateTime, Boolean, Float, JSON, ForeignKey, Index, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import StaticPool
import redis
from pydantic import BaseModel
import aioredis


logger = logging.getLogger(__name__)

Base = declarative_base()


class DatasetMetadata(Base):
    """SQLAlchemy model for dataset metadata."""
    
    __tablename__ = 'dataset_metadata'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    
    # Basic properties
    n_cells = Column(Integer)
    n_genes = Column(Integer)
    n_classes = Column(Integer)
    modality = Column(String(100))
    organism = Column(String(100))
    tissue = Column(String(100))
    has_spatial = Column(Boolean, default=False)
    
    # File properties
    file_path = Column(String(500))
    file_size_mb = Column(Float)
    checksum = Column(String(64))  # SHA256
    download_url = Column(String(500))
    
    # Processing metadata
    graph_method = Column(String(100))
    preprocessing_steps = Column(JSON)
    quality_metrics = Column(JSON)
    
    # Temporal tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed = Column(DateTime)
    access_count = Column(Integer, default=0)
    
    # Citation and provenance
    citation = Column(Text)
    doi = Column(String(255))
    version = Column(String(50))
    license = Column(String(100))
    
    # Tasks and capabilities
    supported_tasks = Column(JSON)  # List of task types
    benchmark_results = Column(JSON)  # Performance on standard benchmarks
    
    # Relationships
    processing_logs = relationship("ProcessingLog", back_populates="dataset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
            'n_classes': self.n_classes,
            'modality': self.modality,
            'organism': self.organism,
            'tissue': self.tissue,
            'has_spatial': self.has_spatial,
            'file_path': self.file_path,
            'file_size_mb': self.file_size_mb,
            'checksum': self.checksum,
            'download_url': self.download_url,
            'graph_method': self.graph_method,
            'preprocessing_steps': self.preprocessing_steps,
            'quality_metrics': self.quality_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'access_count': self.access_count,
            'citation': self.citation,
            'doi': self.doi,
            'version': self.version,
            'license': self.license,
            'supported_tasks': self.supported_tasks,
            'benchmark_results': self.benchmark_results
        }


class ProcessingLog(Base):
    """SQLAlchemy model for dataset processing logs."""
    
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('dataset_metadata.id'), nullable=False)
    
    operation = Column(String(100), nullable=False)  # 'download', 'preprocess', 'graph_build'
    status = Column(String(50), nullable=False)      # 'started', 'completed', 'failed'
    
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    
    parameters = Column(JSON)  # Operation parameters
    results = Column(JSON)     # Operation results/metrics
    error_message = Column(Text)
    
    # System information
    hostname = Column(String(255))
    python_version = Column(String(50))
    torch_version = Column(String(50))
    cuda_version = Column(String(50))
    
    # Relationships
    dataset = relationship("DatasetMetadata", back_populates="processing_logs")


class ModelMetadata(Base):
    """SQLAlchemy model for trained model metadata."""
    
    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    model_type = Column(String(100), nullable=False)  # 'cellgnn', 'cellsage', etc.
    
    # Architecture details
    architecture_config = Column(JSON)
    num_parameters = Column(Integer)
    model_size_mb = Column(Float)
    
    # Training details
    dataset_name = Column(String(255))
    task_type = Column(String(100))
    training_config = Column(JSON)
    
    # Performance metrics
    validation_metrics = Column(JSON)
    test_metrics = Column(JSON)
    biological_metrics = Column(JSON)
    
    # File information
    model_path = Column(String(500))
    checksum = Column(String(64))
    
    # Temporal tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    training_time_hours = Column(Float)
    
    # Metadata
    description = Column(Text)
    tags = Column(JSON)
    is_public = Column(Boolean, default=False)


class ExperimentRun(Base):
    """SQLAlchemy model for experiment tracking."""
    
    __tablename__ = 'experiment_runs'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(255), nullable=False, index=True)
    run_name = Column(String(255))
    
    # Configuration
    dataset_name = Column(String(255))
    model_config = Column(JSON)
    training_config = Column(JSON)
    
    # Status tracking
    status = Column(String(50))  # 'running', 'completed', 'failed', 'cancelled'
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Results
    final_metrics = Column(JSON)
    best_epoch = Column(Integer)
    artifacts = Column(JSON)  # List of saved artifacts
    
    # System info
    hostname = Column(String(255))
    gpu_info = Column(JSON)
    
    # Notes
    notes = Column(Text)
    tags = Column(JSON)


# Database connection and session management
class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: Optional[str] = None, 
                 echo: bool = False, pool_size: int = 10):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL
            echo: Whether to echo SQL queries
            pool_size: Connection pool size
        """
        if database_url is None:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///./scgraph_hub.db')
        
        self.database_url = database_url
        
        # Create engine with appropriate settings
        if database_url.startswith('sqlite'):
            # SQLite-specific settings
            self.engine = create_engine(
                database_url,
                echo=echo,
                poolclass=StaticPool,
                connect_args={'check_same_thread': False}
            )
        else:
            # PostgreSQL/other database settings
            self.engine = create_engine(
                database_url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=20
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        
        # Create tables
        self.create_tables()
        
        # Set up event listeners
        self._setup_event_listeners()
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")
    
    def _setup_event_listeners(self):
        """Setup database event listeners for automatic updates."""
        @event.listens_for(DatasetMetadata, 'before_update')
        def receive_before_update(mapper, connection, target):
            target.updated_at = datetime.utcnow()
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def close(self):
        """Close database connections."""
        self.engine.dispose()


# Redis cache manager
class CacheManager:
    """Manages Redis-based caching for improved performance."""
    
    def __init__(self, redis_url: Optional[str] = None, 
                 default_ttl: int = 3600):
        """Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
        """
        if redis_url is None:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory fallback")
            self.redis_client = None
            self._memory_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Memory fallback
                if key in self._memory_cache:
                    value, expiry = self._memory_cache[key]
                    if datetime.utcnow() < expiry:
                        return value
                    else:
                        del self._memory_cache[key]
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            
            if self.redis_client:
                return self.redis_client.setex(
                    key, ttl, json.dumps(value, default=str)
                )
            else:
                # Memory fallback
                expiry = datetime.utcnow() + timedelta(seconds=ttl)
                self._memory_cache[key] = (value, expiry)
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                return bool(self._memory_cache.pop(key, None))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache."""
        try:
            if self.redis_client:
                return self.redis_client.flushdb()
            else:
                self._memory_cache.clear()
                return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    'type': 'redis',
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            else:
                return {
                    'type': 'memory',
                    'cache_size': len(self._memory_cache),
                    'memory_usage': 'unknown'
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {'type': 'error', 'error': str(e)}


# Data access layer
class DatasetRepository:
    """Repository for dataset metadata operations."""
    
    def __init__(self, db_manager: DatabaseManager, cache_manager: CacheManager):
        self.db = db_manager
        self.cache = cache_manager
    
    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dataset metadata by name."""
        # Try cache first
        cache_key = f"dataset:{name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Query database
        with self.db.get_session() as session:
            dataset = session.query(DatasetMetadata).filter(
                DatasetMetadata.name == name
            ).first()
            
            if dataset:
                # Update access tracking
                dataset.last_accessed = datetime.utcnow()
                dataset.access_count += 1
                session.commit()
                
                result = dataset.to_dict()
                
                # Cache result
                self.cache.set(cache_key, result, ttl=1800)  # 30 minutes
                
                return result
        
        return None
    
    def list_datasets(self, limit: int = 100, offset: int = 0,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List datasets with optional filtering."""
        with self.db.get_session() as session:
            query = session.query(DatasetMetadata)
            
            # Apply filters
            if filters:
                if 'modality' in filters:
                    query = query.filter(DatasetMetadata.modality == filters['modality'])
                if 'organism' in filters:
                    query = query.filter(DatasetMetadata.organism == filters['organism'])
                if 'min_cells' in filters:
                    query = query.filter(DatasetMetadata.n_cells >= filters['min_cells'])
                if 'max_cells' in filters:
                    query = query.filter(DatasetMetadata.n_cells <= filters['max_cells'])
                if 'has_spatial' in filters:
                    query = query.filter(DatasetMetadata.has_spatial == filters['has_spatial'])
            
            # Apply pagination
            datasets = query.offset(offset).limit(limit).all()
            
            return [dataset.to_dict() for dataset in datasets]
    
    def create_dataset(self, metadata: Dict[str, Any]) -> bool:
        """Create new dataset metadata."""
        try:
            with self.db.get_session() as session:
                dataset = DatasetMetadata(**metadata)
                session.add(dataset)
                session.commit()
                
                # Clear relevant caches
                self.cache.delete(f"dataset:{metadata['name']}")
                
                logger.info(f"Created dataset metadata: {metadata['name']}")
                return True
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return False
    
    def update_dataset(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update dataset metadata."""
        try:
            with self.db.get_session() as session:
                dataset = session.query(DatasetMetadata).filter(
                    DatasetMetadata.name == name
                ).first()
                
                if dataset:
                    for key, value in updates.items():
                        if hasattr(dataset, key):
                            setattr(dataset, key, value)
                    
                    session.commit()
                    
                    # Clear cache
                    self.cache.delete(f"dataset:{name}")
                    
                    return True
        except Exception as e:
            logger.error(f"Failed to update dataset {name}: {e}")
            return False
    
    def delete_dataset(self, name: str) -> bool:
        """Delete dataset metadata."""
        try:
            with self.db.get_session() as session:
                dataset = session.query(DatasetMetadata).filter(
                    DatasetMetadata.name == name
                ).first()
                
                if dataset:
                    session.delete(dataset)
                    session.commit()
                    
                    # Clear cache
                    self.cache.delete(f"dataset:{name}")
                    
                    return True
        except Exception as e:
            logger.error(f"Failed to delete dataset {name}: {e}")
            return False
    
    def log_processing_operation(self, dataset_name: str, operation: str,
                               status: str, **kwargs) -> bool:
        """Log a processing operation for a dataset."""
        try:
            with self.db.get_session() as session:
                # Get dataset ID
                dataset = session.query(DatasetMetadata).filter(
                    DatasetMetadata.name == dataset_name
                ).first()
                
                if not dataset:
                    logger.error(f"Dataset {dataset_name} not found for logging")
                    return False
                
                # Create log entry
                log_entry = ProcessingLog(
                    dataset_id=dataset.id,
                    operation=operation,
                    status=status,
                    parameters=kwargs.get('parameters'),
                    results=kwargs.get('results'),
                    error_message=kwargs.get('error_message'),
                    hostname=kwargs.get('hostname'),
                    python_version=kwargs.get('python_version'),
                    torch_version=kwargs.get('torch_version'),
                    cuda_version=kwargs.get('cuda_version')
                )
                
                if status == 'completed':
                    log_entry.completed_at = datetime.utcnow()
                    if 'duration_seconds' in kwargs:
                        log_entry.duration_seconds = kwargs['duration_seconds']
                
                session.add(log_entry)
                session.commit()
                
                return True
        except Exception as e:
            logger.error(f"Failed to log processing operation: {e}")
            return False
    
    def get_processing_history(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get processing history for a dataset."""
        with self.db.get_session() as session:
            dataset = session.query(DatasetMetadata).filter(
                DatasetMetadata.name == dataset_name
            ).first()
            
            if not dataset:
                return []
            
            logs = session.query(ProcessingLog).filter(
                ProcessingLog.dataset_id == dataset.id
            ).order_by(ProcessingLog.started_at.desc()).all()
            
            return [{
                'id': log.id,
                'operation': log.operation,
                'status': log.status,
                'started_at': log.started_at.isoformat() if log.started_at else None,
                'completed_at': log.completed_at.isoformat() if log.completed_at else None,
                'duration_seconds': log.duration_seconds,
                'parameters': log.parameters,
                'results': log.results,
                'error_message': log.error_message,
                'hostname': log.hostname
            } for log in logs]


# Global instances
_db_manager = None
_cache_manager = None
_dataset_repository = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_dataset_repository() -> DatasetRepository:
    """Get the global dataset repository instance."""
    global _dataset_repository
    if _dataset_repository is None:
        _dataset_repository = DatasetRepository(
            get_database_manager(),
            get_cache_manager()
        )
    return _dataset_repository


# Utility functions
def compute_file_checksum(file_path: str) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_system_info() -> Dict[str, str]:
    """Get system information for logging."""
    import platform
    import sys
    
    info = {
        'hostname': platform.node(),
        'python_version': sys.version,
        'platform': platform.platform()
    }
    
    try:
        import torch
        info['torch_version'] = torch.__version__
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
    except ImportError:
        pass
    
    return info