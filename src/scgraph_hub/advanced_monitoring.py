"""Generation 2: Advanced Monitoring and Observability System."""

import json
import logging
import os
import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    success_rate: float = 100.0
    error_count: int = 0
    call_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'call_count': self.call_count
        }


class MetricsCollector:
    """Advanced metrics collection system."""
    
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.aggregated: Dict[str, PerformanceMetrics] = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def record_execution_time(self, function_name: str, execution_time: float) -> None:
        """Record function execution time."""
        with self.lock:
            self.metrics[f"{function_name}_execution_time"].append({
                'timestamp': time.time(),
                'value': execution_time
            })
    
    def record_error(self, function_name: str, error_type: str) -> None:
        """Record an error occurrence."""
        with self.lock:
            self.metrics[f"{function_name}_errors"].append({
                'timestamp': time.time(),
                'error_type': error_type
            })
    
    def record_success(self, function_name: str) -> None:
        """Record a successful execution."""
        with self.lock:
            self.metrics[f"{function_name}_success"].append({
                'timestamp': time.time()
            })
    
    def get_function_metrics(self, function_name: str, window_seconds: int = 3600) -> PerformanceMetrics:
        """Get aggregated metrics for a function."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            # Get execution times
            exec_times = [
                m['value'] for m in self.metrics[f"{function_name}_execution_time"]
                if m['timestamp'] >= cutoff_time
            ]
            
            # Get error count
            error_count = len([
                m for m in self.metrics[f"{function_name}_errors"]
                if m['timestamp'] >= cutoff_time
            ])
            
            # Get success count
            success_count = len([
                m for m in self.metrics[f"{function_name}_success"]
                if m['timestamp'] >= cutoff_time
            ])
            
            total_calls = error_count + success_count
            
            return PerformanceMetrics(
                execution_time=sum(exec_times) / len(exec_times) if exec_times else 0.0,
                success_rate=(success_count / total_calls * 100) if total_calls > 0 else 100.0,
                error_count=error_count,
                call_count=total_calls
            )
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide metrics overview."""
        with self.lock:
            function_names = set()
            for key in self.metrics.keys():
                if '_' in key:
                    function_names.add(key.rsplit('_', 1)[0])
            
            overview = {
                'uptime': time.time() - self.start_time,
                'total_functions_monitored': len(function_names),
                'functions': {}
            }
            
            for func_name in function_names:
                overview['functions'][func_name] = self.get_function_metrics(func_name).to_dict()
            
            return overview


class AdvancedLogger:
    """Advanced logging with structured output and filtering."""
    
    def __init__(self, name: str, log_dir: str = "./logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        self.structured_logs: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def _setup_handlers(self):
        """Setup log handlers."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / f"{self.name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def log_structured(self, level: str, message: str, **kwargs) -> None:
        """Log with structured data."""
        structured_entry = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            'logger': self.name,
            **kwargs
        }
        
        with self.lock:
            self.structured_logs.append(structured_entry)
            # Keep only last 1000 entries
            if len(self.structured_logs) > 1000:
                self.structured_logs = self.structured_logs[-1000:]
        
        # Also log normally
        getattr(self.logger, level.lower())(message, extra=kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info level."""
        self.log_structured('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level."""
        self.log_structured('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error level."""
        self.log_structured('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug level."""
        self.log_structured('DEBUG', message, **kwargs)
    
    def export_logs(self, filepath: str, since_timestamp: Optional[float] = None) -> bool:
        """Export structured logs to file."""
        try:
            with self.lock:
                logs_to_export = self.structured_logs
                if since_timestamp:
                    logs_to_export = [
                        log for log in logs_to_export 
                        if log['timestamp'] >= since_timestamp
                    ]
            
            with open(filepath, 'w') as f:
                json.dump(logs_to_export, f, indent=2, default=str)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            return False


def monitored_function(logger: Optional[AdvancedLogger] = None, 
                      metrics_collector: Optional[MetricsCollector] = None):
    """Decorator for comprehensive function monitoring."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"
            
            if logger:
                logger.debug(f"Starting {function_name}", 
                           args_count=len(args), kwargs_keys=list(kwargs.keys()))
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if metrics_collector:
                    metrics_collector.record_execution_time(function_name, execution_time)
                    metrics_collector.record_success(function_name)
                
                if logger:
                    logger.debug(f"Completed {function_name}", 
                               execution_time=execution_time, success=True)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_type = type(e).__name__
                
                if metrics_collector:
                    metrics_collector.record_execution_time(function_name, execution_time)
                    metrics_collector.record_error(function_name, error_type)
                
                if logger:
                    logger.error(f"Failed {function_name}", 
                               execution_time=execution_time, 
                               error_type=error_type, 
                               error_message=str(e))
                
                raise
        
        return wrapper
    return decorator


class SystemMonitor:
    """Comprehensive system monitoring."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.metrics_collector = MetricsCollector()
        self.logger = AdvancedLogger("system_monitor", log_dir)
        self.alerts: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            'error_rate': 10.0,  # Percentage
            'avg_response_time': 5.0,  # Seconds
            'memory_usage': 80.0,  # Percentage
        }
        
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        overview = self.metrics_collector.get_system_overview()
        
        for func_name, metrics in overview.get('functions', {}).items():
            # Check error rate
            success_rate = metrics.get('success_rate', 100.0)
            error_rate = 100.0 - success_rate
            
            if error_rate > self.alert_thresholds['error_rate']:
                alerts.append({
                    'type': 'high_error_rate',
                    'function': func_name,
                    'value': error_rate,
                    'threshold': self.alert_thresholds['error_rate'],
                    'timestamp': time.time()
                })
            
            # Check response time
            avg_time = metrics.get('execution_time', 0.0)
            if avg_time > self.alert_thresholds['avg_response_time']:
                alerts.append({
                    'type': 'slow_response',
                    'function': func_name,
                    'value': avg_time,
                    'threshold': self.alert_thresholds['avg_response_time'],
                    'timestamp': time.time()
                })
        
        # Store alerts
        self.alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            self.logger.warning("System alert triggered", alert=alert)
        
        return alerts
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        overview = self.metrics_collector.get_system_overview()
        recent_alerts = [
            alert for alert in self.alerts 
            if time.time() - alert['timestamp'] < 3600  # Last hour
        ]
        
        return {
            'system_overview': overview,
            'recent_alerts': recent_alerts,
            'alert_summary': {
                'total_alerts': len(self.alerts),
                'recent_alerts': len(recent_alerts),
                'alert_types': list(set(alert['type'] for alert in recent_alerts))
            },
            'health_status': self._calculate_health_status(overview, recent_alerts)
        }
    
    def _calculate_health_status(self, overview: Dict[str, Any], 
                                recent_alerts: List[Dict[str, Any]]) -> str:
        """Calculate overall health status."""
        if len(recent_alerts) == 0:
            return "healthy"
        elif len(recent_alerts) < 5:
            return "warning"
        else:
            return "critical"
    
    def export_report(self, filepath: str, time_window_hours: int = 24) -> bool:
        """Export monitoring report."""
        try:
            since_timestamp = time.time() - (time_window_hours * 3600)
            dashboard_data = self.get_dashboard_data()
            
            report = {
                'report_timestamp': time.time(),
                'time_window_hours': time_window_hours,
                'dashboard_data': dashboard_data
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Monitoring report exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return False


# Global instances
_global_metrics_collector = MetricsCollector()
_global_system_monitor = SystemMonitor()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_metrics_collector


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor."""
    return _global_system_monitor


@contextmanager
def performance_context(operation_name: str):
    """Context manager for performance measurement."""
    start_time = time.time()
    collector = get_metrics_collector()
    logger = AdvancedLogger("performance")
    
    logger.debug(f"Starting operation: {operation_name}")
    
    try:
        yield
        execution_time = time.time() - start_time
        collector.record_execution_time(operation_name, execution_time)
        collector.record_success(operation_name)
        logger.debug(f"Completed operation: {operation_name}", execution_time=execution_time)
        
    except Exception as e:
        execution_time = time.time() - start_time
        collector.record_execution_time(operation_name, execution_time)
        collector.record_error(operation_name, type(e).__name__)
        logger.error(f"Failed operation: {operation_name}", 
                    execution_time=execution_time, error=str(e))
        raise