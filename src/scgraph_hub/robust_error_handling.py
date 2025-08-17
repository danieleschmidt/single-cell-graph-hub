"""Generation 2: Robust Error Handling and Validation Framework."""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    IO = "io"
    COMPUTATION = "computation"
    NETWORK = "network"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: float = field(default_factory=time.time)
    function_name: str = ""
    module_name: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'function_name': self.function_name,
            'module_name': self.module_name,
            'severity': self.severity.value,
            'category': self.category.value,
            'details': self.details,
            'stack_trace': self.stack_trace
        }


class RobustErrorHandler:
    """Robust error handling with context tracking and recovery."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.max_history = 1000
        self.error_counts: Dict[str, int] = {}
        
    def record_error(self, error: Exception, context: ErrorContext) -> None:
        """Record an error with context."""
        context.stack_trace = traceback.format_exc()
        
        # Add to history
        self.error_history.append(context)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Update error counts
        error_key = f"{context.category.value}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log based on severity
        log_msg = f"Error in {context.function_name}: {str(error)}"
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_msg, extra={'context': context.to_dict()})
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_msg, extra={'context': context.to_dict()})
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_msg, extra={'context': context.to_dict()})
        else:
            self.logger.info(log_msg, extra={'context': context.to_dict()})
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'error_counts': self.error_counts.copy(),
            'top_errors': sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'severity_distribution': {
                severity.value: len([e for e in recent_errors if e.severity == severity])
                for severity in ErrorSeverity
            }
        }
    
    def clear_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()


# Global error handler
_global_error_handler = RobustErrorHandler()


def get_error_handler() -> RobustErrorHandler:
    """Get the global error handler."""
    return _global_error_handler


def robust_wrapper(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    fallback_value: Any = None,
    retry_count: int = 0,
    retry_delay: float = 1.0
):
    """Decorator for robust error handling with retry logic."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        function_name=func.__name__,
                        module_name=func.__module__,
                        severity=severity,
                        category=category,
                        details={
                            'attempt': attempt + 1,
                            'max_attempts': retry_count + 1,
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys())
                        }
                    )
                    
                    error_handler.record_error(e, context)
                    
                    # If this is the last attempt, return fallback or re-raise
                    if attempt == retry_count:
                        if fallback_value is not None:
                            return fallback_value
                        raise
                    
                    # Wait before retry
                    if retry_delay > 0:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            
            return fallback_value
        
        return wrapper
    return decorator


class ValidationError(Exception):
    """Custom validation error."""
    pass


class DataValidator:
    """Comprehensive data validation framework."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules: Dict[str, List[Callable]] = {}
        
    def add_rule(self, data_type: str, rule: Callable[[Any], bool], error_msg: str = ""):
        """Add a validation rule."""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = []
        
        def rule_wrapper(data):
            try:
                return rule(data)
            except Exception as e:
                self.logger.warning(f"Validation rule failed: {e}")
                return False
        
        rule_wrapper.error_msg = error_msg or f"Validation failed for {data_type}"
        self.validation_rules[data_type].append(rule_wrapper)
    
    @robust_wrapper(
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.VALIDATION,
        fallback_value=(False, ["Validation process failed"])
    )
    def validate(self, data: Any, data_type: str) -> tuple[bool, List[str]]:
        """Validate data against registered rules."""
        if data_type not in self.validation_rules:
            self.logger.warning(f"No validation rules for data type: {data_type}")
            return True, []
        
        errors = []
        for rule in self.validation_rules[data_type]:
            try:
                if not rule(data):
                    errors.append(getattr(rule, 'error_msg', 'Validation failed'))
            except Exception as e:
                errors.append(f"Rule execution failed: {str(e)}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_dataset(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate dataset structure."""
        errors = []
        
        # Check required fields
        required_fields = ['x', 'y', 'edge_index']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if not errors:
            # Validate data consistency
            try:
                x = data['x']
                y = data['y']
                edge_index = data.get('edge_index', [])
                
                # Check if x and y have same number of samples
                if len(x) != len(y):
                    errors.append(f"Feature-label size mismatch: {len(x)} vs {len(y)}")
                
                # Check edge index validity
                if edge_index and len(edge_index) == 2:
                    max_node = max(max(edge_index[0], default=-1), max(edge_index[1], default=-1))
                    if max_node >= len(x):
                        errors.append(f"Edge index out of bounds: {max_node} >= {len(x)}")
                
            except Exception as e:
                errors.append(f"Data validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_model_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate model configuration."""
        errors = []
        
        required_fields = ['name', 'input_dim', 'output_dim']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required config field: {field}")
        
        # Validate ranges
        if 'input_dim' in config and config['input_dim'] <= 0:
            errors.append("input_dim must be positive")
        
        if 'output_dim' in config and config['output_dim'] <= 0:
            errors.append("output_dim must be positive")
        
        if 'hidden_dim' in config and config['hidden_dim'] <= 0:
            errors.append("hidden_dim must be positive")
        
        if 'dropout' in config and not (0 <= config['dropout'] <= 1):
            errors.append("dropout must be between 0 and 1")
        
        return len(errors) == 0, errors


# Global validator
_global_validator = DataValidator()


def get_validator() -> DataValidator:
    """Get the global validator."""
    return _global_validator


class InputSanitizer:
    """Input sanitization for security and robustness."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    @robust_wrapper(
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.VALIDATION,
        fallback_value=""
    )
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            value = str(value)
        
        # Remove null bytes and control characters
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
            self.logger.warning(f"String truncated to {max_length} characters")
        
        return value
    
    @robust_wrapper(
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.VALIDATION,
        fallback_value={}
    )
    def sanitize_dict(self, data: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
        """Sanitize dictionary input."""
        if not isinstance(data, dict):
            raise ValidationError("Input must be a dictionary")
        
        def _sanitize_recursive(obj, depth=0):
            if depth > max_depth:
                return str(obj)[:100]  # Convert to string and truncate
            
            if isinstance(obj, dict):
                return {
                    self.sanitize_string(str(k), 100): _sanitize_recursive(v, depth + 1)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [_sanitize_recursive(item, depth + 1) for item in obj[:1000]]  # Limit list size
            elif isinstance(obj, str):
                return self.sanitize_string(obj)
            elif isinstance(obj, (int, float, bool)):
                return obj
            else:
                return str(obj)[:100]
        
        return _sanitize_recursive(data)
    
    @robust_wrapper(
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.VALIDATION,
        fallback_value=[]
    )
    def sanitize_list(self, data: List[Any], max_length: int = 10000) -> List[Any]:
        """Sanitize list input."""
        if not isinstance(data, list):
            raise ValidationError("Input must be a list")
        
        # Limit list size
        if len(data) > max_length:
            data = data[:max_length]
            self.logger.warning(f"List truncated to {max_length} items")
        
        return data


# Global sanitizer
_global_sanitizer = InputSanitizer()


def get_sanitizer() -> InputSanitizer:
    """Get the global sanitizer."""
    return _global_sanitizer


def validate_and_sanitize(data_type: str):
    """Decorator that validates and sanitizes input data."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = get_validator()
            sanitizer = get_sanitizer()
            
            # Sanitize arguments if they are dicts or lists
            sanitized_args = []
            for arg in args:
                if isinstance(arg, dict):
                    sanitized_args.append(sanitizer.sanitize_dict(arg))
                elif isinstance(arg, list):
                    sanitized_args.append(sanitizer.sanitize_list(arg))
                elif isinstance(arg, str):
                    sanitized_args.append(sanitizer.sanitize_string(arg))
                else:
                    sanitized_args.append(arg)
            
            sanitized_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    sanitized_kwargs[k] = sanitizer.sanitize_dict(v)
                elif isinstance(v, list):
                    sanitized_kwargs[k] = sanitizer.sanitize_list(v)
                elif isinstance(v, str):
                    sanitized_kwargs[k] = sanitizer.sanitize_string(v)
                else:
                    sanitized_kwargs[k] = v
            
            return func(*sanitized_args, **sanitized_kwargs)
        
        return wrapper
    return decorator


class HealthMonitor:
    """System health monitoring for Generation 2."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {}
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: Any, category: str = "general") -> None:
        """Record a metric."""
        timestamp = time.time()
        
        if category not in self.metrics:
            self.metrics[category] = {}
        
        if name not in self.metrics[category]:
            self.metrics[category][name] = []
        
        self.metrics[category][name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Keep only last 1000 entries per metric
        if len(self.metrics[category][name]) > 1000:
            self.metrics[category][name] = self.metrics[category][name][-1000:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        error_handler = get_error_handler()
        error_stats = error_handler.get_error_stats()
        
        uptime = time.time() - self.start_time
        
        # Calculate health score based on recent errors
        recent_errors = error_stats['recent_errors']
        health_score = max(0, 100 - (recent_errors * 5))  # Decrease by 5 per recent error
        
        status = "healthy" if health_score > 80 else "warning" if health_score > 50 else "critical"
        
        return {
            'status': status,
            'health_score': health_score,
            'uptime_seconds': uptime,
            'error_stats': error_stats,
            'metrics_summary': {
                category: len(metrics) for category, metrics in self.metrics.items()
            }
        }


# Global health monitor
_global_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor."""
    return _global_health_monitor