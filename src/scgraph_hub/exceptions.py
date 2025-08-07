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


class SecurityError(SCGraphHubError):
    """Exception raised for security violations."""
    
    def __init__(self, security_issue: str, details: Optional[str] = None):
        message = f"Security violation: {security_issue}"
        if details:
            message += f" - {details}"
        
        super().__init__(
            message=message,
            error_code="SECURITY_VIOLATION",
            details={"security_issue": security_issue, "violation_details": details}
        )


# Enhanced error handling utilities
def handle_error_gracefully(func):
    """Decorator for graceful error handling with logging."""
    import functools
    import logging
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SCGraphHubError as e:
            # Log structured error
            logging.error(
                f"SCGraph error in {func.__name__}: {e.message}",
                extra={'error_details': e.details, 'error_code': e.error_code}
            )
            raise
        except Exception as e:
            # Convert unknown errors to SCGraphHubError
            error = SCGraphHubError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={'original_error': str(e), 'function': func.__name__}
            )
            logging.error(
                f"Unexpected error in {func.__name__}: {str(e)}",
                extra={'error_details': error.details}
            )
            raise error
    
    return wrapper


def validate_required_dependencies(dependencies: list, operation: str = None):
    """Validate that required dependencies are available."""
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        if operation:
            raise DependencyError(missing)
        else:
            raise DependencyError(missing)


class ErrorCollector:
    """Collect multiple errors before raising."""
    
    def __init__(self):
        self.errors = []
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
    
    def add_validation_error(self, field: str, value: Any, reason: str):
        """Add a validation error."""
        self.errors.append(f"Field '{field}' (value: {value}): {reason}")
    
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0
    
    def raise_if_errors(self, error_type: type = ValidationError, message: str = "Validation failed"):
        """Raise exception if errors were collected."""
        if self.has_errors():
            if error_type == ValidationError:
                raise ValidationError("multiple_fields", self.errors, message + ": " + "; ".join(self.errors))
            else:
                raise error_type(message + ": " + "; ".join(self.errors))
    
    def get_errors(self) -> list:
        """Get all collected errors."""
        return self.errors.copy()


def create_error_context(operation: str):
    """Create error context for better error messages."""
    class ErrorContext:
        def __init__(self, operation: str):
            self.operation = operation
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type and issubclass(exc_type, Exception) and not issubclass(exc_type, SCGraphHubError):
                # Convert generic exceptions to SCGraphHubError with context
                raise SCGraphHubError(
                    message=f"Error during {self.operation}: {str(exc_val)}",
                    error_code="OPERATION_FAILED",
                    details={'operation': self.operation, 'original_error': str(exc_val)}
                ) from exc_val
            return False
    
    return ErrorContext(operation)