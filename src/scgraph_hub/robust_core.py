"""
Robust Core Module for Single-Cell Graph Hub
Generation 2: MAKE IT ROBUST

This module provides robust implementations with comprehensive error handling,
validation, logging, and fallback mechanisms for production-ready deployment.
"""

import os
import logging
import traceback
import warnings
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import json
import time
from dataclasses import dataclass
from contextlib import contextmanager


# Configure robust logging
def setup_robust_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup comprehensive logging with file and console output."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Configure specific loggers
    logging.getLogger('scgraph_hub').setLevel(getattr(logging, log_level.upper()))
    return logging.getLogger('scgraph_hub.robust')


logger = setup_robust_logging()


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


class RobustValidator:
    """Comprehensive validation system for all components."""
    
    def __init__(self):
        self.logger = logging.getLogger('scgraph_hub.validator')
    
    def validate_dataset_structure(self, dataset_info: Dict[str, Any]) -> ValidationResult:
        """Validate dataset structure and metadata."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Check required fields
            required_fields = ['name', 'n_cells', 'n_genes', 'modality', 'organism']
            for field in required_fields:
                if field not in dataset_info:
                    errors.append(f"Missing required field: {field}")
            
            # Validate data types and ranges
            if 'n_cells' in dataset_info:
                n_cells = dataset_info['n_cells']
                if not isinstance(n_cells, int) or n_cells <= 0:
                    errors.append(f"Invalid n_cells: {n_cells}")
                elif n_cells < 100:
                    warnings.append(f"Very small dataset: {n_cells} cells")
                elif n_cells > 1000000:
                    warnings.append(f"Very large dataset: {n_cells} cells")
            
            if 'n_genes' in dataset_info:
                n_genes = dataset_info['n_genes']
                if not isinstance(n_genes, int) or n_genes <= 0:
                    errors.append(f"Invalid n_genes: {n_genes}")
                elif n_genes < 100:
                    warnings.append(f"Few genes: {n_genes}")
            
            # Validate organism
            if 'organism' in dataset_info:
                valid_organisms = ['human', 'mouse', 'rat', 'zebrafish', 'drosophila', 'c_elegans']
                if dataset_info['organism'] not in valid_organisms:
                    warnings.append(f"Unknown organism: {dataset_info['organism']}")
            
            # Validate modality
            if 'modality' in dataset_info:
                valid_modalities = ['scRNA-seq', 'snRNA-seq', 'scATAC-seq', 'spatial_transcriptomics', 'multiome']
                if dataset_info['modality'] not in valid_modalities:
                    warnings.append(f"Unknown modality: {dataset_info['modality']}")
            
            metadata['validation_time'] = time.time()
            metadata['fields_checked'] = len(dataset_info)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            self.logger.error(f"Dataset validation failed: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def validate_graph_structure(self, edge_index: Any, num_nodes: int) -> ValidationResult:
        """Validate graph structure."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            if edge_index is None:
                errors.append("Edge index is None")
                return ValidationResult(False, errors, warnings, metadata)
            
            # Check if we have numpy available
            try:
                import numpy as np
                
                if isinstance(edge_index, (list, tuple)):
                    edge_index = np.array(edge_index)
                
                if not isinstance(edge_index, np.ndarray):
                    errors.append(f"Invalid edge_index type: {type(edge_index)}")
                    return ValidationResult(False, errors, warnings, metadata)
                
                # Validate shape
                if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
                    errors.append(f"Invalid edge_index shape: {edge_index.shape}")
                
                # Check node indices
                if edge_index.size > 0:
                    max_node = np.max(edge_index)
                    min_node = np.min(edge_index)
                    
                    if min_node < 0:
                        errors.append(f"Negative node index: {min_node}")
                    
                    if max_node >= num_nodes:
                        errors.append(f"Node index {max_node} >= num_nodes {num_nodes}")
                    
                    # Check for self-loops
                    self_loops = np.sum(edge_index[0] == edge_index[1])
                    if self_loops > 0:
                        warnings.append(f"Found {self_loops} self-loops")
                    
                    # Check connectivity
                    num_edges = edge_index.shape[1]
                    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
                    
                    if avg_degree < 1:
                        warnings.append(f"Very sparse graph: avg degree {avg_degree:.2f}")
                    elif avg_degree > 50:
                        warnings.append(f"Very dense graph: avg degree {avg_degree:.2f}")
                    
                    metadata['num_edges'] = num_edges
                    metadata['avg_degree'] = avg_degree
                    metadata['has_self_loops'] = self_loops > 0
                
            except ImportError:
                warnings.append("NumPy not available - limited validation")
        
        except Exception as e:
            errors.append(f"Graph validation error: {str(e)}")
            self.logger.error(f"Graph validation failed: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )


class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.logger = logging.getLogger('scgraph_hub.error_handler')
        self.error_counts = {}
        self.recovery_strategies = {}
    
    @contextmanager
    def handle_errors(self, operation_name: str, fallback_result=None):
        """Context manager for robust error handling."""
        try:
            self.logger.debug(f"Starting operation: {operation_name}")
            yield
            self.logger.debug(f"Completed operation: {operation_name}")
            
        except ImportError as e:
            self.logger.warning(f"Import error in {operation_name}: {e}")
            self._record_error(operation_name, "ImportError")
            if fallback_result is not None:
                return fallback_result
            raise
            
        except FileNotFoundError as e:
            self.logger.warning(f"File not found in {operation_name}: {e}")
            self._record_error(operation_name, "FileNotFoundError")
            if fallback_result is not None:
                return fallback_result
            raise
            
        except ValueError as e:
            self.logger.error(f"Value error in {operation_name}: {e}")
            self._record_error(operation_name, "ValueError")
            if fallback_result is not None:
                return fallback_result
            raise
            
        except Exception as e:
            self.logger.error(f"Unexpected error in {operation_name}: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            self._record_error(operation_name, "UnexpectedError")
            if fallback_result is not None:
                return fallback_result
            raise
    
    def _record_error(self, operation: str, error_type: str):
        """Record error statistics."""
        key = f"{operation}:{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'error_types': len(set(k.split(':')[1] for k in self.error_counts.keys()))
        }


class RobustDataManager:
    """Robust data management with validation and error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger('scgraph_hub.data_manager')
        self.validator = RobustValidator()
        self.error_handler = RobustErrorHandler()
    
    def safe_load_data(self, file_path: str, file_type: str = "auto") -> Tuple[Any, ValidationResult]:
        """Safely load data with validation."""
        with self.error_handler.handle_errors(f"load_data_{file_type}"):
            if file_type == "auto":
                file_type = self._detect_file_type(file_path)
            
            if file_type == "json":
                return self._load_json_safe(file_path)
            elif file_type == "csv":
                return self._load_csv_safe(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension."""
        ext = Path(file_path).suffix.lower()
        type_map = {
            '.json': 'json',
            '.csv': 'csv',
            '.tsv': 'csv',
            '.txt': 'csv'
        }
        return type_map.get(ext, 'unknown')
    
    def _load_json_safe(self, file_path: str) -> Tuple[Any, ValidationResult]:
        """Safely load JSON data."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            validation = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                metadata={'file_size': os.path.getsize(file_path)}
            )
            
            return data, validation
            
        except json.JSONDecodeError as e:
            validation = ValidationResult(
                is_valid=False,
                errors=[f"JSON decode error: {e}"],
                warnings=[],
                metadata={}
            )
            return None, validation
    
    def _load_csv_safe(self, file_path: str) -> Tuple[Any, ValidationResult]:
        """Safely load CSV data."""
        try:
            import pandas as pd
            data = pd.read_csv(file_path)
            
            validation = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                metadata={
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'memory_usage': data.memory_usage(deep=True).sum()
                }
            )
            
            return data, validation
            
        except ImportError:
            # Fallback without pandas
            validation = ValidationResult(
                is_valid=False,
                errors=["pandas not available for CSV loading"],
                warnings=["Using basic file reading"],
                metadata={}
            )
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            return lines, validation


class RobustCacheManager:
    """Robust caching system with validation and cleanup."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('scgraph_hub.cache')
        self.validator = RobustValidator()
    
    def get(self, key: str, validation_func=None) -> Tuple[Any, bool]:
        """Get cached data with validation."""
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None, False
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Validate cached data if function provided
            if validation_func and not validation_func(cached_data):
                self.logger.warning(f"Cached data validation failed for key: {key}")
                cache_file.unlink()  # Remove invalid cache
                return None, False
            
            self.logger.debug(f"Cache hit for key: {key}")
            return cached_data, True
            
        except Exception as e:
            self.logger.error(f"Cache read error for key {key}: {e}")
            return None, False
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Set cached data with TTL."""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            
            cache_entry = {
                'data': data,
                'timestamp': time.time(),
                'ttl': ttl
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, indent=2)
            
            self.logger.debug(f"Cached data for key: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache write error for key {key}: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        cleaned_count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_entry = json.load(f)
                
                if 'ttl' in cache_entry and cache_entry['ttl']:
                    if current_time - cache_entry['timestamp'] > cache_entry['ttl']:
                        cache_file.unlink()
                        cleaned_count += 1
                        self.logger.debug(f"Cleaned expired cache: {cache_file.name}")
                        
            except Exception as e:
                self.logger.error(f"Error cleaning cache file {cache_file}: {e}")
        
        return cleaned_count


class RobustHealthCheck:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.logger = logging.getLogger('scgraph_hub.health')
        self.checks = {}
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check function."""
        self.checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        failed_checks = 0
        
        for check_name, check_func in self.checks.items():
            try:
                start_time = time.time()
                check_result = check_func()
                duration = time.time() - start_time
                
                if isinstance(check_result, bool):
                    check_result = {'status': 'healthy' if check_result else 'unhealthy'}
                
                check_result['duration_ms'] = int(duration * 1000)
                results['checks'][check_name] = check_result
                
                if check_result.get('status') != 'healthy':
                    failed_checks += 1
                    
            except Exception as e:
                results['checks'][check_name] = {
                    'status': 'error',
                    'error': str(e),
                    'duration_ms': 0
                }
                failed_checks += 1
                self.logger.error(f"Health check {check_name} failed: {e}")
        
        if failed_checks > 0:
            results['overall_status'] = 'degraded' if failed_checks < len(self.checks) else 'unhealthy'
        
        return results


# Default health checks
def check_dependencies() -> Dict[str, Any]:
    """Check if required dependencies are available."""
    deps = {
        'numpy': False,
        'pandas': False,
        'scikit-learn': False,
        'torch': False
    }
    
    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
            deps[dep] = True
        except ImportError:
            pass
    
    essential_deps = ['numpy']
    missing_essential = [dep for dep in essential_deps if not deps[dep]]
    
    return {
        'status': 'healthy' if len(missing_essential) == 0 else 'unhealthy',
        'dependencies': deps,
        'missing_essential': missing_essential
    }


def check_disk_space() -> Dict[str, Any]:
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        usage_percent = (used / total) * 100
        
        status = 'healthy'
        if free_gb < 1:
            status = 'unhealthy'
        elif free_gb < 5:
            status = 'warning'
        
        return {
            'status': status,
            'free_gb': round(free_gb, 2),
            'total_gb': round(total_gb, 2),
            'usage_percent': round(usage_percent, 2)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


# Initialize default health checker
default_health_checker = RobustHealthCheck()
default_health_checker.register_check('dependencies', check_dependencies)
default_health_checker.register_check('disk_space', check_disk_space)


# Export main classes
__all__ = [
    'RobustValidator',
    'RobustErrorHandler', 
    'RobustDataManager',
    'RobustCacheManager',
    'RobustHealthCheck',
    'ValidationResult',
    'setup_robust_logging',
    'default_health_checker'
]