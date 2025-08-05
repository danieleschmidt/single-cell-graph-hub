"""Custom exceptions for Single-Cell Graph Hub."""

from typing import Optional, Dict, Any


class SCGraphHubError(Exception):
    """Base exception for Single-Cell Graph Hub."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DatasetError(SCGraphHubError):
    """Exception raised for dataset-related errors."""
    pass


class DatasetNotFoundError(DatasetError):
    """Exception raised when a requested dataset is not found."""
    
    def __init__(self, dataset_name: str):
        super().__init__(
            f"Dataset '{dataset_name}' not found",
            error_code="DATASET_NOT_FOUND",
            details={"dataset_name": dataset_name}
        )


class DatasetDownloadError(DatasetError):
    """Exception raised when dataset download fails."""
    
    def __init__(self, dataset_name: str, reason: str):
        super().__init__(
            f"Failed to download dataset '{dataset_name}': {reason}",
            error_code="DATASET_DOWNLOAD_FAILED",
            details={"dataset_name": dataset_name, "reason": reason}
        )


class DatasetValidationError(DatasetError):
    """Exception raised when dataset validation fails."""
    
    def __init__(self, dataset_name: str, validation_errors: Dict[str, str]):
        error_summary = ", ".join(f"{field}: {error}" for field, error in validation_errors.items())
        super().__init__(
            f"Dataset '{dataset_name}' validation failed: {error_summary}",
            error_code="DATASET_VALIDATION_FAILED",
            details={"dataset_name": dataset_name, "validation_errors": validation_errors}
        )


class ModelError(SCGraphHubError):
    """Exception raised for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Exception raised when a requested model is not found."""
    
    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' not found",
            error_code="MODEL_NOT_FOUND",
            details={"model_name": model_name}
        )


class ModelConfigurationError(ModelError):
    """Exception raised when model configuration is invalid."""
    
    def __init__(self, model_name: str, config_errors: Dict[str, str]):
        error_summary = ", ".join(f"{param}: {error}" for param, error in config_errors.items())
        super().__init__(
            f"Model '{model_name}' configuration error: {error_summary}",
            error_code="MODEL_CONFIG_INVALID",
            details={"model_name": model_name, "config_errors": config_errors}
        )


class PreprocessingError(SCGraphHubError):
    """Exception raised for preprocessing-related errors."""
    pass


class GraphConstructionError(PreprocessingError):
    """Exception raised when graph construction fails."""
    
    def __init__(self, dataset_name: str, method: str, reason: str):
        super().__init__(
            f"Graph construction failed for dataset '{dataset_name}' using method '{method}': {reason}",
            error_code="GRAPH_CONSTRUCTION_FAILED",
            details={"dataset_name": dataset_name, "method": method, "reason": reason}
        )


class CacheError(SCGraphHubError):
    """Exception raised for cache-related errors."""
    pass


class CacheCorruptedError(CacheError):
    """Exception raised when cache data is corrupted."""
    
    def __init__(self, cache_key: str):
        super().__init__(
            f"Cache data corrupted for key '{cache_key}'",
            error_code="CACHE_CORRUPTED",
            details={"cache_key": cache_key}
        )


class APIError(SCGraphHubError):
    """Exception raised for API-related errors."""
    pass


class RateLimitExceededError(APIError):
    """Exception raised when API rate limit is exceeded."""
    
    def __init__(self, retry_after: int):
        super().__init__(
            f"API rate limit exceeded. Retry after {retry_after} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after}
        )


class AuthenticationError(APIError):
    """Exception raised for authentication failures."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message,
            error_code="AUTHENTICATION_FAILED"
        )


class DependencyError(SCGraphHubError):
    """Exception raised when required dependencies are missing."""
    
    def __init__(self, missing_dependencies: list):
        deps_str = ", ".join(missing_dependencies)
        super().__init__(
            f"Missing required dependencies: {deps_str}",
            error_code="MISSING_DEPENDENCIES",
            details={"missing_dependencies": missing_dependencies}
        )


class ConfigurationError(SCGraphHubError):
    """Exception raised for configuration errors."""
    
    def __init__(self, config_field: str, reason: str):
        super().__init__(
            f"Configuration error in '{config_field}': {reason}",
            error_code="CONFIGURATION_ERROR",
            details={"config_field": config_field, "reason": reason}
        )


class ValidationError(SCGraphHubError):
    """Exception raised for general validation errors."""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Validation error for field '{field}' with value '{value}': {reason}",
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": value, "reason": reason}
        )


class StorageError(SCGraphHubError):
    """Exception raised for storage-related errors."""
    pass


class InsufficientStorageError(StorageError):
    """Exception raised when there's insufficient storage space."""
    
    def __init__(self, required_space: int, available_space: int):
        super().__init__(
            f"Insufficient storage space. Required: {required_space} MB, Available: {available_space} MB",
            error_code="INSUFFICIENT_STORAGE",
            details={"required_space": required_space, "available_space": available_space}
        )


class FileCorruptedError(StorageError):
    """Exception raised when a file is corrupted."""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            f"File corrupted: {file_path}. Reason: {reason}",
            error_code="FILE_CORRUPTED",
            details={"file_path": file_path, "reason": reason}
        )