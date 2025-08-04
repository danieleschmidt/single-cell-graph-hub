"""Security and safety utilities for Single-Cell Graph Hub."""

import logging
import hashlib
import secrets
import tempfile
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation for data and code inputs."""
    
    def __init__(self):
        self.allowed_extensions = {
            '.h5ad', '.h5', '.csv', '.tsv', '.txt', '.json', '.pkl', '.pt', '.pth'
        }
        self.max_file_size_mb = 10240  # 10GB
        self.suspicious_patterns = [
            'import os', 'import sys', 'subprocess', 'eval(', 'exec(',
            '__import__', 'open(', 'file(', 'input(', 'raw_input('
        ]
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate file path for security risks.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Validation results
        """
        results = {
            'safe': True,
            'issues': [],
            'warnings': []
        }
        
        path = Path(file_path)
        
        # Check for path traversal
        if '..' in str(path) or str(path).startswith('/'):
            results['safe'] = False
            results['issues'].append('Path traversal detected')
        
        # Check file extension
        if path.suffix.lower() not in self.allowed_extensions:
            results['warnings'].append(f'Unusual file extension: {path.suffix}')
        
        # Check file size if exists
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_file_size_mb:
                results['safe'] = False
                results['issues'].append(f'File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB')
        
        return results
    
    def validate_code_content(self, content: str) -> Dict[str, Any]:
        """Validate code content for potentially dangerous patterns.
        
        Args:
            content: Code content to validate
            
        Returns:
            Validation results
        """
        results = {
            'safe': True,
            'issues': [],
            'warnings': []
        }
        
        content_lower = content.lower()
        
        for pattern in self.suspicious_patterns:
            if pattern in content_lower:
                results['warnings'].append(f'Suspicious pattern found: {pattern}')
        
        # Check for obfuscated code
        if 'base64' in content_lower or 'decode(' in content_lower:
            results['warnings'].append('Potential code obfuscation detected')
        
        # Check for network operations
        network_patterns = ['urllib', 'requests', 'socket', 'http']
        for pattern in network_patterns:
            if pattern in content_lower:
                results['warnings'].append(f'Network operation detected: {pattern}')
        
        return results
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration for security issues.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validation results
        """
        results = {
            'safe': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for dangerous configuration values
        dangerous_keys = ['password', 'secret', 'token', 'key', 'api_key']
        
        def check_dict(d, path=''):
            for key, value in d.items():
                current_path = f'{path}.{key}' if path else key
                
                # Check for sensitive keys
                if any(danger in key.lower() for danger in dangerous_keys):
                    results['warnings'].append(f'Sensitive key found: {current_path}')
                
                # Check for file paths
                if isinstance(value, str) and ('/' in value or '\\' in value):
                    path_validation = self.validate_file_path(value)
                    if not path_validation['safe']:
                        results['issues'].extend([f'{current_path}: {issue}' for issue in path_validation['issues']])
                        results['safe'] = False
                
                # Recursively check nested dictionaries
                if isinstance(value, dict):
                    check_dict(value, current_path)
        
        check_dict(config)
        
        return results


class SafeDataLoader:
    """Safe data loading with validation and sandboxing."""
    
    def __init__(self):
        self.validator = SecurityValidator()
        self.temp_dir = None
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='scgraph_safe_')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def load_file_safely(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load file with security validation.
        
        Args:
            file_path: Path to file
            **kwargs: Additional arguments for loading
            
        Returns:
            Loaded data
            
        Raises:
            SecurityError: If file fails security validation
        """
        path = Path(file_path)
        
        # Validate file path
        validation = self.validator.validate_file_path(path)
        if not validation['safe']:
            raise SecurityError(f"File failed security validation: {validation['issues']}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"File security warning: {warning}")
        
        # Load file based on extension
        try:
            if path.suffix.lower() == '.h5ad':
                import scanpy as sc
                return sc.read_h5ad(path)
            elif path.suffix.lower() in {'.csv', '.tsv'}:
                import pandas as pd
                return pd.read_csv(path, **kwargs)
            elif path.suffix.lower() == '.json':
                import json
                with open(path, 'r') as f:
                    return json.load(f)
            elif path.suffix.lower() in {'.pkl', '.pickle'}:
                # Restricted pickle loading
                return self._safe_pickle_load(path)
            elif path.suffix.lower() in {'.pt', '.pth'}:
                import torch
                return torch.load(path, map_location='cpu')
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        
        except Exception as e:
            logger.error(f"Failed to load file {path}: {e}")
            raise
    
    def _safe_pickle_load(self, file_path: Path) -> Any:
        """Safely load pickle file with restrictions.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Loaded object
        """
        import pickle
        
        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Only allow safe modules
                safe_modules = {
                    'numpy', 'pandas', 'torch', 'scipy',
                    'sklearn', 'matplotlib', 'seaborn'
                }
                
                if module.split('.')[0] not in safe_modules:
                    raise pickle.UnpicklingError(f"Unsafe module: {module}")
                
                return super().find_class(module, name)
        
        with open(file_path, 'rb') as f:
            return SafeUnpickler(f).load()


class DataSanitizer:
    """Sanitize data inputs to prevent injection attacks."""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input.
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
            logger.warning(f"String truncated to {max_length} characters")
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '\x00']
        for char in dangerous_chars:
            value = value.replace(char, '')
        
        return value
    
    @staticmethod
    def sanitize_numeric(value: Union[int, float], min_val: float = -1e6, max_val: float = 1e6) -> Union[int, float]:
        """Sanitize numeric input.
        
        Args:
            value: Numeric value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Sanitized numeric value
        """
        try:
            if isinstance(value, str):
                value = float(value)
            
            if value < min_val:
                logger.warning(f"Value {value} clamped to minimum {min_val}")
                return min_val
            elif value > max_val:
                logger.warning(f"Value {value} clamped to maximum {max_val}")
                return max_val
            
            return value
        
        except (ValueError, TypeError):
            logger.error(f"Invalid numeric value: {value}")
            return 0
    
    @staticmethod
    def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration dictionary.
        
        Args:
            config: Configuration to sanitize
            
        Returns:
            Sanitized configuration
        """
        sanitized = {}
        
        for key, value in config.items():
            # Sanitize key
            clean_key = DataSanitizer.sanitize_string(key, 100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[clean_key] = DataSanitizer.sanitize_string(value)
            elif isinstance(value, (int, float)):
                sanitized[clean_key] = DataSanitizer.sanitize_numeric(value)
            elif isinstance(value, dict):
                sanitized[clean_key] = DataSanitizer.sanitize_config(value)
            elif isinstance(value, list):
                sanitized[clean_key] = [DataSanitizer.sanitize_string(str(item), 100) for item in value[:100]]  # Limit list size
            else:
                sanitized[clean_key] = value
        
        return sanitized


class ResourceMonitor:
    """Monitor resource usage to prevent DoS attacks."""
    
    def __init__(self, max_memory_mb: float = 8192, max_time_seconds: float = 3600):
        """Initialize resource monitor.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_time_seconds: Maximum execution time in seconds
        """
        self.max_memory_mb = max_memory_mb
        self.max_time_seconds = max_time_seconds
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_time_seconds:
                logger.warning(f"Operation took {elapsed:.1f}s, exceeded limit of {self.max_time_seconds}s")
    
    def check_memory_usage(self) -> bool:
        """Check current memory usage.
        
        Returns:
            True if within limits
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if memory_mb > self.max_memory_mb:
                logger.error(f"Memory usage ({memory_mb:.1f}MB) exceeded limit ({self.max_memory_mb}MB)")
                return False
            
            return True
        
        except ImportError:
            logger.warning("psutil not available, cannot check memory usage")
            return True
        except Exception as e:
            logger.error(f"Failed to check memory usage: {e}")
            return True
    
    def check_time_limit(self) -> bool:
        """Check if time limit exceeded.
        
        Returns:
            True if within limits
        """
        if self.start_time is None:
            return True
        
        elapsed = time.time() - self.start_time
        if elapsed > self.max_time_seconds:
            logger.error(f"Time limit ({self.max_time_seconds}s) exceeded: {elapsed:.1f}s")
            return False
        
        return True


class SecurityError(Exception):
    """Custom exception for security violations."""
    pass


# Utility functions
def generate_session_token() -> str:
    """Generate secure session token.
    
    Returns:
        Secure random token
    """
    return secrets.token_urlsafe(32)


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
    """Hash sensitive data for secure storage.
    
    Args:
        data: Data to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Hashed data
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    combined = f"{salt}{data}"
    return hashlib.sha256(combined.encode()).hexdigest()


def secure_random_string(length: int = 16) -> str:
    """Generate cryptographically secure random string.
    
    Args:
        length: Length of string
        
    Returns:
        Random string
    """
    return secrets.token_urlsafe(length)


def validate_user_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize user input.
    
    Args:
        data: User input data
        
    Returns:
        Validation results
    """
    validator = SecurityValidator()
    sanitizer = DataSanitizer()
    
    # Validate configuration
    validation = validator.validate_config(data)
    
    # Sanitize if validation passes
    if validation['safe']:
        sanitized_data = sanitizer.sanitize_config(data)
        validation['sanitized_data'] = sanitized_data
    
    return validation


# Security context manager
class security_context:
    """Context manager for secure operations."""
    
    def __init__(self, max_memory_mb: float = 8192, max_time_seconds: float = 3600):
        self.resource_monitor = ResourceMonitor(max_memory_mb, max_time_seconds)
        self.safe_loader = SafeDataLoader()
    
    def __enter__(self):
        self.resource_monitor.__enter__()
        self.safe_loader.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.safe_loader.__exit__(exc_type, exc_val, exc_tb)
        self.resource_monitor.__exit__(exc_type, exc_val, exc_tb)
    
    def load_file(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load file safely within security context."""
        return self.safe_loader.load_file_safely(file_path, **kwargs)
    
    def check_resources(self) -> bool:
        """Check if resources are within limits."""
        return (self.resource_monitor.check_memory_usage() and 
                self.resource_monitor.check_time_limit())
