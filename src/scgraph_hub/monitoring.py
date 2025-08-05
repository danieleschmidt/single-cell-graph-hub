"""Comprehensive monitoring and health checks for Single-Cell Graph Hub."""

import logging
import time
import threading
import queue
import psutil
import os
import sys
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings

import torch
import numpy as np

logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health checking for system resources and models."""
    
    def __init__(self):
        """Initialize health checker."""
        self.health_status = {}
        self.alerts = []
        self.thresholds = {
            'memory_usage_percent': 90.0,
            'disk_usage_percent': 95.0,
            'gpu_memory_percent': 95.0,
            'model_inference_time_ms': 10000.0,
            'dataset_load_time_ms': 30000.0
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'checks': {}
        }
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            health_report['checks']['memory'] = {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent,
                'status': 'healthy' if memory.percent < self.thresholds['memory_usage_percent'] else 'warning'
            }
            
            # Disk check
            disk = psutil.disk_usage('/')
            health_report['checks']['disk'] = {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'percent_used': (disk.used / disk.total) * 100,
                'status': 'healthy' if (disk.used / disk.total) * 100 < self.thresholds['disk_usage_percent'] else 'warning'
            }
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            health_report['checks']['cpu'] = {
                'percent_used': cpu_percent,
                'cores': psutil.cpu_count(),
                'status': 'healthy'
            }
            
            # GPU check (if available)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_percent = (gpu_allocated / gpu_memory) * 100
                
                health_report['checks']['gpu'] = {
                    'total_memory_gb': gpu_memory / (1024**3),
                    'allocated_memory_gb': gpu_allocated / (1024**3),
                    'percent_used': gpu_percent,
                    'status': 'healthy' if gpu_percent < self.thresholds['gpu_memory_percent'] else 'warning'
                }
            
            # Python environment check
            health_report['checks']['python'] = {
                'version': sys.version,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'status': 'healthy'
            }
            
            # Determine overall status
            check_statuses = [check.get('status', 'unknown') for check in health_report['checks'].values()]
            if 'error' in check_statuses:
                health_report['status'] = 'error'
            elif 'warning' in check_statuses:
                health_report['status'] = 'warning'
            
        except Exception as e:
            health_report['status'] = 'error'
            health_report['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_report
    
    def check_model_health(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Check model health and performance."""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'checks': {}
        }
        
        try:
            # Parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            health_report['checks']['parameters'] = {
                'total': total_params,
                'trainable': trainable_params,
                'memory_estimate_mb': (total_params * 4) / (1024**2),  # Assuming float32
                'status': 'healthy'
            }
            
            # Gradient check
            has_nan_gradients = False
            gradient_norms = []
            
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norms.append(grad_norm)
                    if torch.isnan(param.grad).any():
                        has_nan_gradients = True
            
            health_report['checks']['gradients'] = {
                'has_nan': has_nan_gradients,
                'max_norm': max(gradient_norms) if gradient_norms else 0,
                'mean_norm': np.mean(gradient_norms) if gradient_norms else 0,
                'status': 'error' if has_nan_gradients else 'healthy'
            }
            
            # Parameter statistics
            param_stats = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    health_report['checks']['parameters']['status'] = 'error'
                
                param_stats.append({
                    'name': name,
                    'shape': list(param.shape),
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'has_nan': torch.isnan(param).any().item(),
                    'has_inf': torch.isinf(param).any().item()
                })
            
            health_report['checks']['parameter_stats'] = param_stats
            
            # Determine overall status
            if has_nan_gradients or health_report['checks']['parameters']['status'] == 'error':
                health_report['status'] = 'error'
                
        except Exception as e:
            health_report['status'] = 'error'
            health_report['error'] = str(e)
            logger.error(f"Model health check failed: {e}")
        
        return health_report


class PerformanceMonitor:
    """Monitor performance metrics during model training and inference."""
    
    def __init__(self, window_size: int = 100):
        """Initialize performance monitor.
        
        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.start_times = {}
        self.total_metrics = defaultdict(list)
        self.callbacks = []
    
    def start_timer(self, name: str):
        """Start timing an operation.
        
        Args:
            name: Name of the operation
        """
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and record duration.
        
        Args:
            name: Name of the operation
            
        Returns:
            Duration in seconds
        """
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.time() - self.start_times[name]
        self.record_metric(f"{name}_duration", duration)
        del self.start_times[name]
        return duration
    
    def record_metric(self, name: str, value: float):
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name].append(value)
        self.total_metrics[name].append(value)
        
        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback(name, value)
            except Exception as e:
                logger.error(f"Callback failed: {e}")
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with statistics
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {}
        
        values = list(self.metrics[name])
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'recent': values[-1] if values else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics.
        
        Returns:
            Dictionary mapping metric names to statistics
        """
        return {name: self.get_metric_stats(name) for name in self.metrics.keys()}
    
    def add_callback(self, callback: Callable[[str, float], None]):
        """Add callback for metric updates.
        
        Args:
            callback: Function to call when metrics are updated
        """
        self.callbacks.append(callback)
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()
        self.total_metrics.clear()
    
    def export_metrics(self, filepath: str):
        """Export metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'window_size': self.window_size,
            'statistics': self.get_all_stats(),
            'total_metrics': {k: list(v) for k, v in self.total_metrics.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self, check_interval: float = 10.0):
        """Initialize resource monitor.
        
        Args:
            check_interval: Interval between resource checks (seconds)
        """
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.resource_queue = queue.Queue()
        self.alerts = []
        
        # Resource thresholds
        self.thresholds = {
            'memory_percent': 90.0,
            'gpu_memory_percent': 95.0,
            'disk_usage_percent': 90.0
        }
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring:
            logger.warning("Resource monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)."""
        while self.monitoring:
            try:
                resources = self._collect_resources()
                self.resource_queue.put(resources)
                self._check_thresholds(resources)
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _collect_resources(self) -> Dict[str, Any]:
        """Collect current resource usage.
        
        Returns:
            Dictionary with resource information
        """
        resources = {
            'timestamp': datetime.now().isoformat()
        }
        
        # CPU and memory
        try:
            import psutil
            
            resources['cpu_percent'] = psutil.cpu_percent(interval=1)
            
            memory = psutil.virtual_memory()
            resources['memory_total_gb'] = memory.total / (1024**3)
            resources['memory_used_gb'] = memory.used / (1024**3)
            resources['memory_percent'] = memory.percent
            
            disk = psutil.disk_usage('/')
            resources['disk_total_gb'] = disk.total / (1024**3)
            resources['disk_used_gb'] = disk.used / (1024**3)
            resources['disk_percent'] = (disk.used / disk.total) * 100
            
        except ImportError:
            logger.warning("psutil not available, limited resource monitoring")
        except Exception as e:
            logger.error(f"Failed to collect system resources: {e}")
        
        # GPU resources
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    resources[f'gpu_{i}_memory_allocated_gb'] = allocated
                    resources[f'gpu_{i}_memory_reserved_gb'] = reserved
                    
                    # Estimate total GPU memory (approximate)
                    if reserved > 0:
                        resources[f'gpu_{i}_memory_percent'] = (allocated / reserved) * 100
            
            except Exception as e:
                logger.error(f"Failed to collect GPU resources: {e}")
        
        return resources
    
    def _check_thresholds(self, resources: Dict[str, Any]):
        """Check if resources exceed thresholds.
        
        Args:
            resources: Resource dictionary
        """
        for metric, threshold in self.thresholds.items():
            if metric in resources and resources[metric] > threshold:
                alert = {
                    'timestamp': resources['timestamp'],
                    'metric': metric,
                    'value': resources[metric],
                    'threshold': threshold,
                    'message': f"{metric} ({resources[metric]:.1f}%) exceeded threshold ({threshold}%)"
                }
                
                self.alerts.append(alert)
                logger.warning(alert['message'])
                
                # Keep only recent alerts
                if len(self.alerts) > 100:
                    self.alerts = self.alerts[-100:]
    
    def get_current_resources(self) -> Optional[Dict[str, Any]]:
        """Get most recent resource measurements.
        
        Returns:
            Resource dictionary or None if no data available
        """
        try:
            return self.resource_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get recent alerts.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = []
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time > cutoff_time:
                recent_alerts.append(alert)
        
        return recent_alerts
    
    def set_threshold(self, metric: str, value: float):
        """Set threshold for a metric.
        
        Args:
            metric: Metric name
            value: Threshold value
        """
        self.thresholds[metric] = value
        logger.info(f"Set threshold for {metric}: {value}")



class ModelMonitor:
    """Monitor model training and performance."""
    
    def __init__(self):
        self.training_history = []
        self.current_epoch = 0
        self.best_metrics = {}
        self.early_stopping_patience = None
        self.early_stopping_counter = 0
        self.callbacks = []
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_start'):
                callback.on_epoch_start(epoch)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics for this epoch
        """
        # Record training history
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.training_history.append(epoch_data)
        
        # Update best metrics
        for metric_name, value in metrics.items():
            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = {'value': value, 'epoch': epoch}
            else:
                # Assume higher is better for most metrics
                if value > self.best_metrics[metric_name]['value']:
                    self.best_metrics[metric_name] = {'value': value, 'epoch': epoch}
        
        # Check early stopping
        if self.early_stopping_patience:
            self._check_early_stopping(metrics)
        
        # Trigger callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, metrics)
    
    def _check_early_stopping(self, metrics: Dict[str, float]):
        """Check if early stopping criteria are met.
        
        Args:
            metrics: Current epoch metrics
        """
        # Simple early stopping based on validation loss
        if 'val_loss' in metrics:
            current_loss = metrics['val_loss']
            
            if len(self.training_history) < 2:
                return
            
            previous_loss = self.training_history[-2].get('val_loss', float('inf'))
            
            if current_loss >= previous_loss:
                self.early_stopping_counter += 1
            else:
                self.early_stopping_counter = 0
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early.
        
        Returns:
            True if training should stop
        """
        return (self.early_stopping_patience and 
                self.early_stopping_counter >= self.early_stopping_patience)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress.
        
        Returns:
            Training summary
        """
        if not self.training_history:
            return {'status': 'no_training_data'}
        
        latest = self.training_history[-1]
        
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': len(self.training_history),
            'latest_metrics': {k: v for k, v in latest.items() if k not in ['epoch', 'timestamp']},
            'best_metrics': self.best_metrics,
            'early_stopping_counter': self.early_stopping_counter,
            'should_stop_early': self.should_stop_early()
        }
    
    def export_training_history(self, filepath: str):
        """Export training history to file.
        
        Args:
            filepath: Path to save training history
        """
        export_data = {
            'training_history': self.training_history,
            'best_metrics': self.best_metrics,
            'summary': self.get_training_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Training history exported to {filepath}")
    
    def add_callback(self, callback):
        """Add training callback.
        
        Args:
            callback: Callback object with on_epoch_start/on_epoch_end methods
        """
        self.callbacks.append(callback)


# Global monitoring instances
_performance_monitor = None
_resource_monitor = None
_health_checker = None
_model_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def get_model_monitor() -> ModelMonitor:
    """Get global model monitor instance."""
    global _model_monitor
    if _model_monitor is None:
        _model_monitor = ModelMonitor()
    return _model_monitor


# Convenience functions
def start_monitoring():
    """Start all monitoring services."""
    resource_monitor = get_resource_monitor()
    resource_monitor.start_monitoring()
    logger.info("All monitoring services started")


def stop_monitoring():
    """Stop all monitoring services."""
    resource_monitor = get_resource_monitor()
    resource_monitor.stop_monitoring()
    logger.info("All monitoring services stopped")


def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status."""
    health_checker = get_health_checker()
    return health_checker.get_overall_health()


def export_all_metrics(output_dir: str):
    """Export all monitoring data to directory.
    
    Args:
        output_dir: Directory to save monitoring data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export performance metrics
    perf_monitor = get_performance_monitor()
    perf_monitor.export_metrics(str(output_path / 'performance_metrics.json'))
    
    # Export health status
    health_checker = get_health_checker()
    health_data = health_checker.get_overall_health()
    with open(output_path / 'health_status.json', 'w') as f:
        json.dump(health_data, f, indent=2)
    
    # Export model training history if available
    model_monitor = get_model_monitor()
    if model_monitor.training_history:
        model_monitor.export_training_history(str(output_path / 'training_history.json'))
    
    logger.info(f"All monitoring data exported to {output_dir}")
