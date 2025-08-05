"""Health checks and system monitoring for Single-Cell Graph Hub."""

import time
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch

from .logging_config import get_logger
from .exceptions import SCGraphHubError

logger = get_logger(__name__)


class HealthStatus:
    """Represents the health status of a component."""
    
    def __init__(self, component: str, status: str = "healthy", message: str = "", details: Optional[Dict[str, Any]] = None):
        self.component = component
        self.status = status  # "healthy", "degraded", "unhealthy"
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
    
    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == "healthy"


class SystemHealthChecker:
    """Comprehensive system health checker."""
    
    def __init__(self):
        self.last_check = None
        self.check_interval = 60  # seconds
        self.memory_threshold = 80  # percent
        self.disk_threshold = 90  # percent
        self.cpu_threshold = 95  # percent
    
    async def check_all(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        logger.info("Running comprehensive system health check")
        
        checks = {
            "system_resources": await self._check_system_resources(),
            "dependencies": await self._check_dependencies(),
            "storage": await self._check_storage(),
            "pytorch": await self._check_pytorch(),
            "catalog": await self._check_catalog(),
        }
        
        self.last_check = datetime.utcnow()
        
        # Log unhealthy components
        unhealthy = [name for name, status in checks.items() if not status.is_healthy]
        if unhealthy:
            logger.warning(f"Unhealthy components detected: {', '.join(unhealthy)}")
        else:
            logger.info("All system components are healthy")
        
        return checks
    
    async def _check_system_resources(self) -> HealthStatus:
        """Check system resource usage."""
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Load average (Unix only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                pass  # Windows doesn't have load average
            
            details = {
                "memory_percent": memory_percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_percent": cpu_percent,
                "load_average": load_avg
            }
            
            # Determine status
            if memory_percent > self.memory_threshold or cpu_percent > self.cpu_threshold:
                return HealthStatus(
                    "system_resources", 
                    "unhealthy",
                    f"High resource usage: Memory {memory_percent:.1f}%, CPU {cpu_percent:.1f}%",
                    details
                )
            elif memory_percent > self.memory_threshold * 0.8 or cpu_percent > self.cpu_threshold * 0.8:
                return HealthStatus(
                    "system_resources",
                    "degraded", 
                    f"Elevated resource usage: Memory {memory_percent:.1f}%, CPU {cpu_percent:.1f}%",
                    details
                )
            else:
                return HealthStatus(
                    "system_resources",
                    "healthy",
                    f"Resource usage normal: Memory {memory_percent:.1f}%, CPU {cpu_percent:.1f}%",
                    details
                )
                
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return HealthStatus(
                "system_resources",
                "unhealthy",
                f"Failed to check system resources: {e}"
            )
    
    async def _check_dependencies(self) -> HealthStatus:
        """Check if required dependencies are available."""
        try:
            from . import check_dependencies
            
            deps_available = check_dependencies()
            
            if deps_available:
                return HealthStatus(
                    "dependencies",
                    "healthy",
                    "All required dependencies are available"
                )
            else:
                return HealthStatus(
                    "dependencies",
                    "degraded",
                    "Some dependencies are missing or outdated"
                )
                
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return HealthStatus(
                "dependencies",
                "unhealthy",
                f"Failed to check dependencies: {e}"
            )
    
    async def _check_storage(self) -> HealthStatus:
        """Check storage availability."""
        try:
            # Check home directory storage
            home_path = Path.home()
            disk_usage = psutil.disk_usage(str(home_path))
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            details = {
                "disk_percent_used": disk_percent,
                "free_space_gb": disk_usage.free / (1024**3),
                "total_space_gb": disk_usage.total / (1024**3)
            }
            
            if disk_percent > self.disk_threshold:
                return HealthStatus(
                    "storage",
                    "unhealthy",
                    f"Disk usage critical: {disk_percent:.1f}%",
                    details
                )
            elif disk_percent > self.disk_threshold * 0.8:
                return HealthStatus(
                    "storage",
                    "degraded",
                    f"Disk usage high: {disk_percent:.1f}%",
                    details
                )
            else:
                return HealthStatus(
                    "storage",
                    "healthy",
                    f"Disk usage normal: {disk_percent:.1f}%",
                    details
                )
                
        except Exception as e:
            logger.error(f"Storage check failed: {e}")
            return HealthStatus(
                "storage",
                "unhealthy",
                f"Failed to check storage: {e}"
            )
    
    async def _check_pytorch(self) -> HealthStatus:
        """Check PyTorch functionality."""
        try:
            # Test basic tensor operations
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.matmul(x, y)
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            details = {
                "cuda_available": cuda_available,
                "cuda_device_count": device_count,
                "torch_version": torch.__version__
            }
            
            return HealthStatus(
                "pytorch",
                "healthy",
                f"PyTorch functional, CUDA: {'available' if cuda_available else 'not available'}",
                details
            )
            
        except Exception as e:
            logger.error(f"PyTorch check failed: {e}")
            return HealthStatus(
                "pytorch",
                "unhealthy",
                f"PyTorch functionality check failed: {e}"
            )
    
    async def _check_catalog(self) -> HealthStatus:
        """Check dataset catalog functionality."""
        try:
            from .catalog import get_default_catalog
            
            catalog = get_default_catalog()
            datasets = catalog.list_datasets()
            
            if len(datasets) > 0:
                return HealthStatus(
                    "catalog",
                    "healthy",
                    f"Catalog functional with {len(datasets)} datasets",
                    {"dataset_count": len(datasets)}
                )
            else:
                return HealthStatus(
                    "catalog",
                    "degraded",
                    "Catalog is empty"
                )
                
        except Exception as e:
            logger.error(f"Catalog check failed: {e}")
            return HealthStatus(
                "catalog",
                "unhealthy",
                f"Catalog check failed: {e}"
            )


class PerformanceMonitor:
    """Monitor performance metrics over time."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.metrics_history = []
        self.start_time = time.time()
    
    def record_operation(self, operation: str, duration: float, success: bool = True, details: Optional[Dict[str, Any]] = None):
        """Record an operation's performance metrics."""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "duration": duration,
            "success": success,
            "details": details or {}
        }
        
        self.metrics_history.append(metric)
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        logger.debug(f"Operation {operation}: {duration:.3f}s, Success: {success}")
    
    def get_summary(self, operation: Optional[str] = None, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get performance summary statistics."""
        metrics = self.metrics_history
        
        # Filter by operation if specified
        if operation:
            metrics = [m for m in metrics if m["operation"] == operation]
        
        # Limit to last N if specified
        if last_n:
            metrics = metrics[-last_n:]
        
        if not metrics:
            return {"error": "No metrics found"}
        
        durations = [m["duration"] for m in metrics]
        success_count = sum(1 for m in metrics if m["success"])
        
        return {
            "operation": operation or "all",
            "total_operations": len(metrics),
            "success_rate": success_count / len(metrics),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "last_operation": metrics[-1]["timestamp"] if metrics else None
        }
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time


# Global instances
_health_checker = None
_performance_monitor = None


def get_health_checker() -> SystemHealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = SystemHealthChecker()
    return _health_checker


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def performance_timer(operation_name: str):
    """Decorator to time function execution and record performance metrics."""
    def decorator(func):
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            monitor = get_performance_monitor()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration, success=False, details={"error": str(e)})
                raise
        
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            monitor = get_performance_monitor()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration, success=False, details={"error": str(e)})
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


async def run_health_check() -> Dict[str, Any]:
    """Run a complete health check and return results."""
    checker = get_health_checker()
    health_results = await checker.check_all()
    
    # Convert to serializable format
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy" if all(status.is_healthy for status in health_results.values()) else "unhealthy",
        "components": {name: status.to_dict() for name, status in health_results.items()},
        "uptime_seconds": get_performance_monitor().get_uptime()
    }
    
    return results