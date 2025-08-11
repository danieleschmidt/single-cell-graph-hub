"""Advanced scalability features for autonomous SDLC execution."""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import numpy as np
from datetime import datetime, timedelta
import queue
import weakref

from .logging_config import get_logger
from .performance import PerformanceOptimizer, PerformanceCache


class ScalingStrategy(Enum):
    """Scaling strategies for different workloads."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class WorkloadType(Enum):
    """Types of workloads for optimization."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    MIXED = "mixed"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    gpu_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingDecision:
    """Decision for scaling operations."""
    strategy: ScalingStrategy
    target_resources: Dict[str, int]
    expected_improvement: float
    cost_estimate: float
    confidence: float


class AdaptiveLoadBalancer:
    """Adaptive load balancer with intelligent distribution."""
    
    def __init__(self, initial_workers: int = None):
        self.logger = get_logger(__name__)
        self.workers = []
        self.worker_metrics = {}
        self.request_queue = asyncio.Queue()
        self.response_futures = {}
        self.load_history = []
        
        # Auto-detect optimal worker count
        if initial_workers is None:
            initial_workers = min(psutil.cpu_count(), 8)
        
        self.min_workers = max(1, initial_workers // 2)
        self.max_workers = initial_workers * 2
        self.current_workers = initial_workers
        
        # Performance tracking
        self.response_times = []
        self.throughput_history = []
        self.adaptive_threshold = 0.1  # 10% improvement threshold
    
    async def start(self):
        """Start the load balancer."""
        self.logger.info(f"Starting adaptive load balancer with {self.current_workers} workers")
        
        # Initialize workers
        await self._scale_workers(self.current_workers)
        
        # Start monitoring and adaptation
        asyncio.create_task(self._monitor_performance())
        asyncio.create_task(self._adaptive_scaling())
    
    async def submit_task(self, task_func: Callable, *args, **kwargs) -> Any:
        """Submit task to the load balancer."""
        request_id = id(task_func)  # Simple ID generation
        future = asyncio.Future()
        
        await self.request_queue.put({
            'id': request_id,
            'func': task_func,
            'args': args,
            'kwargs': kwargs,
            'future': future,
            'submit_time': time.time()
        })
        
        return await future
    
    async def _scale_workers(self, target_count: int):
        """Scale workers to target count."""
        current_count = len(self.workers)
        
        if target_count > current_count:
            # Scale up
            for _ in range(target_count - current_count):
                worker = await self._create_worker()
                self.workers.append(worker)
        elif target_count < current_count:
            # Scale down
            workers_to_remove = current_count - target_count
            for _ in range(workers_to_remove):
                if self.workers:
                    worker = self.workers.pop()
                    await self._shutdown_worker(worker)
        
        self.current_workers = len(self.workers)
        self.logger.info(f"Scaled to {self.current_workers} workers")
    
    async def _create_worker(self) -> Dict[str, Any]:
        """Create a new worker."""
        worker_id = len(self.workers)
        
        # Create worker process/thread pool
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"worker-{worker_id}")
        
        worker = {
            'id': worker_id,
            'executor': executor,
            'load': 0,
            'active_tasks': 0,
            'total_tasks': 0,
            'avg_response_time': 0.0,
            'created_at': time.time()
        }
        
        self.worker_metrics[worker_id] = []
        
        # Start worker task processor
        asyncio.create_task(self._worker_processor(worker))
        
        return worker
    
    async def _shutdown_worker(self, worker: Dict[str, Any]):
        """Shutdown a worker gracefully."""
        worker['executor'].shutdown(wait=True)
        self.logger.debug(f"Shutdown worker {worker['id']}")
    
    async def _worker_processor(self, worker: Dict[str, Any]):
        """Process tasks for a specific worker."""
        worker_id = worker['id']
        
        while worker in self.workers:  # Worker is still active
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Execute task
                start_time = time.time()
                worker['active_tasks'] += 1
                
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        worker['executor'],
                        task['func'],
                        *task['args']
                    )
                    
                    # Record success
                    execution_time = time.time() - start_time
                    worker['total_tasks'] += 1
                    worker['avg_response_time'] = (
                        (worker['avg_response_time'] * (worker['total_tasks'] - 1) + execution_time) /
                        worker['total_tasks']
                    )
                    
                    # Update metrics
                    self.worker_metrics[worker_id].append({
                        'timestamp': time.time(),
                        'execution_time': execution_time,
                        'success': True
                    })
                    
                    # Return result
                    task['future'].set_result(result)
                    
                except Exception as e:
                    # Record failure
                    self.worker_metrics[worker_id].append({
                        'timestamp': time.time(),
                        'execution_time': time.time() - start_time,
                        'success': False,
                        'error': str(e)
                    })
                    
                    task['future'].set_exception(e)
                
                finally:
                    worker['active_tasks'] -= 1
                    self.request_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    def _select_best_worker(self) -> Optional[Dict[str, Any]]:
        """Select the best worker for task assignment."""
        if not self.workers:
            return None
        
        # Calculate load scores for each worker
        scored_workers = []
        for worker in self.workers:
            # Consider active tasks, average response time, and recent performance
            load_score = (
                worker['active_tasks'] * 0.4 +
                worker['avg_response_time'] * 0.3 +
                len([m for m in self.worker_metrics.get(worker['id'], []) 
                     if time.time() - m['timestamp'] < 60]) * 0.3
            )
            scored_workers.append((load_score, worker))
        
        # Return worker with lowest load score
        return min(scored_workers, key=lambda x: x[0])[1]
    
    async def _monitor_performance(self):
        """Monitor overall performance metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Calculate current metrics
                current_time = time.time()
                
                # Queue length
                queue_size = self.request_queue.qsize()
                
                # Average response time across all workers
                total_response_time = sum(w['avg_response_time'] for w in self.workers)
                avg_response_time = total_response_time / len(self.workers) if self.workers else 0
                
                # Throughput (tasks completed in last minute)
                recent_completions = 0
                for worker_id, metrics in self.worker_metrics.items():
                    recent_completions += len([
                        m for m in metrics 
                        if current_time - m['timestamp'] < 60
                    ])
                
                throughput = recent_completions / 60.0  # tasks per second
                
                # Store metrics
                self.throughput_history.append(throughput)
                self.response_times.append(avg_response_time)
                
                # Keep only recent history
                if len(self.throughput_history) > 100:
                    self.throughput_history.pop(0)
                if len(self.response_times) > 100:
                    self.response_times.pop(0)
                
                self.logger.debug(f"Performance: Queue={queue_size}, "
                                f"AvgResponse={avg_response_time:.3f}s, "
                                f"Throughput={throughput:.2f} tasks/s")
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
    
    async def _adaptive_scaling(self):
        """Adaptive scaling based on performance metrics."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Analyze recent performance
                if len(self.throughput_history) < 3:
                    continue
                
                recent_throughput = np.mean(self.throughput_history[-3:])
                recent_response_time = np.mean(self.response_times[-3:])
                queue_size = self.request_queue.qsize()
                
                # Scaling decision logic
                should_scale_up = (
                    queue_size > self.current_workers * 2 or  # Queue building up
                    recent_response_time > 5.0 or            # Response time too high
                    recent_throughput < self.current_workers * 0.5  # Low throughput per worker
                )
                
                should_scale_down = (
                    queue_size == 0 and                      # No queued tasks
                    recent_response_time < 1.0 and           # Fast response times
                    recent_throughput < self.current_workers * 0.2 and  # Very low throughput
                    self.current_workers > self.min_workers
                )
                
                if should_scale_up and self.current_workers < self.max_workers:
                    new_count = min(self.current_workers + 1, self.max_workers)
                    self.logger.info(f"Scaling up from {self.current_workers} to {new_count} workers")
                    await self._scale_workers(new_count)
                
                elif should_scale_down and self.current_workers > self.min_workers:
                    new_count = max(self.current_workers - 1, self.min_workers)
                    self.logger.info(f"Scaling down from {self.current_workers} to {new_count} workers")
                    await self._scale_workers(new_count)
                
            except Exception as e:
                self.logger.error(f"Adaptive scaling error: {e}")


class IntelligentResourceManager:
    """Intelligent resource manager with predictive scaling."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.resource_history: List[ResourceMetrics] = []
        self.scaling_history: List[ScalingDecision] = []
        self.load_balancer = AdaptiveLoadBalancer()
        self.performance_optimizer = PerformanceOptimizer()
        
        # ML-based prediction (simplified)
        self.workload_patterns = {}
        self.prediction_accuracy = 0.8
        
    async def start_management(self):
        """Start intelligent resource management."""
        self.logger.info("Starting intelligent resource management")
        
        # Start load balancer
        await self.load_balancer.start()
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_resources())
        asyncio.create_task(self._predictive_scaling())
        asyncio.create_task(self._optimize_resource_allocation())
    
    async def _monitor_resources(self):
        """Monitor system resource utilization."""
        while True:
            try:
                # Collect current metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                metrics = ResourceMetrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    disk_io=disk_io.read_bytes + disk_io.write_bytes if disk_io else 0,
                    network_io=network_io.bytes_sent + network_io.bytes_recv if network_io else 0
                )
                
                self.resource_history.append(metrics)
                
                # Keep only recent history (last 24 hours at 1-minute intervals)
                if len(self.resource_history) > 1440:
                    self.resource_history.pop(0)
                
                # Analyze for anomalies or trends
                await self._analyze_resource_trends()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
    
    async def _analyze_resource_trends(self):
        """Analyze resource trends for predictive scaling."""
        if len(self.resource_history) < 10:
            return
        
        recent_metrics = self.resource_history[-10:]
        
        # Calculate trends
        cpu_trend = np.polyfit(range(len(recent_metrics)), 
                              [m.cpu_usage for m in recent_metrics], 1)[0]
        memory_trend = np.polyfit(range(len(recent_metrics)), 
                                 [m.memory_usage for m in recent_metrics], 1)[0]
        
        # Detect concerning trends
        if cpu_trend > 2.0:  # CPU usage increasing by 2% per minute
            self.logger.warning("Detected rising CPU trend")
            await self._handle_resource_pressure("cpu", cpu_trend)
        
        if memory_trend > 1.0:  # Memory usage increasing by 1% per minute
            self.logger.warning("Detected rising memory trend")
            await self._handle_resource_pressure("memory", memory_trend)
    
    async def _handle_resource_pressure(self, resource_type: str, trend: float):
        """Handle detected resource pressure."""
        if resource_type == "cpu":
            # Scale horizontally for CPU pressure
            await self._trigger_horizontal_scaling()
        elif resource_type == "memory":
            # Optimize memory usage and consider vertical scaling
            await self._optimize_memory_usage()
            if trend > 5.0:  # Severe memory pressure
                await self._trigger_vertical_scaling()
    
    async def _trigger_horizontal_scaling(self):
        """Trigger horizontal scaling."""
        self.logger.info("Triggering horizontal scaling")
        
        # Increase worker count in load balancer
        target_workers = min(self.load_balancer.current_workers * 2, 
                           self.load_balancer.max_workers)
        await self.load_balancer._scale_workers(target_workers)
    
    async def _trigger_vertical_scaling(self):
        """Trigger vertical scaling (resource optimization)."""
        self.logger.info("Triggering vertical scaling optimization")
        
        # Optimize existing processes
        await self.performance_optimizer.optimize_memory_usage()
        
        # Adjust batch sizes and concurrency
        await self._adjust_processing_parameters()
    
    async def _optimize_memory_usage(self):
        """Optimize current memory usage."""
        self.logger.info("Optimizing memory usage")
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        self.logger.debug(f"Garbage collection freed {collected} objects")
        
        # Clear performance caches if memory pressure is high
        if self.resource_history and self.resource_history[-1].memory_usage > 80:
            await self.performance_optimizer.clear_caches()
    
    async def _adjust_processing_parameters(self):
        """Adjust processing parameters based on resource constraints."""
        current_memory = self.resource_history[-1].memory_usage if self.resource_history else 50
        
        if current_memory > 80:
            # High memory usage - reduce batch sizes
            self.logger.info("Reducing batch sizes due to memory pressure")
            # This would adjust batch sizes in active processing
            
        elif current_memory < 40:
            # Low memory usage - can increase batch sizes
            self.logger.info("Increasing batch sizes due to available memory")
    
    async def _predictive_scaling(self):
        """Predictive scaling based on historical patterns."""
        while True:
            try:
                await asyncio.sleep(300)  # Predict every 5 minutes
                
                if len(self.resource_history) < 60:  # Need at least 1 hour of data
                    continue
                
                # Predict resource needs for next hour
                predictions = await self._predict_resource_needs()
                
                # Make scaling decisions based on predictions
                scaling_decision = await self._decide_scaling_action(predictions)
                
                if scaling_decision:
                    await self._execute_scaling_decision(scaling_decision)
                
            except Exception as e:
                self.logger.error(f"Predictive scaling error: {e}")
    
    async def _predict_resource_needs(self) -> Dict[str, float]:
        """Predict future resource needs based on historical data."""
        # Simple time-series prediction (in practice, would use more sophisticated ML)
        recent_cpu = [m.cpu_usage for m in self.resource_history[-60:]]  # Last hour
        recent_memory = [m.memory_usage for m in self.resource_history[-60:]]
        
        # Linear trend prediction
        cpu_coeffs = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)
        memory_coeffs = np.polyfit(range(len(recent_memory)), recent_memory, 1)
        
        # Predict 1 hour ahead (60 minutes)
        predicted_cpu = cpu_coeffs[0] * 60 + cpu_coeffs[1] + recent_cpu[-1]
        predicted_memory = memory_coeffs[0] * 60 + memory_coeffs[1] + recent_memory[-1]
        
        # Add seasonal patterns (simplified)
        current_hour = datetime.now().hour
        cpu_seasonal = self._get_seasonal_adjustment("cpu", current_hour)
        memory_seasonal = self._get_seasonal_adjustment("memory", current_hour)
        
        return {
            "cpu": max(0, min(100, predicted_cpu + cpu_seasonal)),
            "memory": max(0, min(100, predicted_memory + memory_seasonal)),
            "confidence": self.prediction_accuracy
        }
    
    def _get_seasonal_adjustment(self, resource_type: str, hour: int) -> float:
        """Get seasonal adjustment factor for predictions."""
        # Business hours typically have higher load
        if 9 <= hour <= 17:
            return 5.0 if resource_type == "cpu" else 3.0
        elif 18 <= hour <= 22:
            return 2.0 if resource_type == "cpu" else 1.0
        else:
            return -2.0 if resource_type == "cpu" else -1.0
    
    async def _decide_scaling_action(self, predictions: Dict[str, float]) -> Optional[ScalingDecision]:
        """Decide on scaling action based on predictions."""
        predicted_cpu = predictions["cpu"]
        predicted_memory = predictions["memory"]
        confidence = predictions["confidence"]
        
        # Only act on high-confidence predictions
        if confidence < 0.7:
            return None
        
        # Determine if scaling is needed
        if predicted_cpu > 80 or predicted_memory > 85:
            # Need to scale up
            if predicted_cpu > predicted_memory:
                strategy = ScalingStrategy.HORIZONTAL  # CPU-bound
            else:
                strategy = ScalingStrategy.VERTICAL    # Memory-bound
            
            return ScalingDecision(
                strategy=strategy,
                target_resources={
                    "cpu_limit": min(predicted_cpu * 1.2, 100),
                    "memory_limit": min(predicted_memory * 1.2, 100),
                    "workers": self.load_balancer.current_workers + 1
                },
                expected_improvement=0.3,  # 30% improvement expected
                cost_estimate=1.5,         # 50% cost increase
                confidence=confidence
            )
        
        elif predicted_cpu < 30 and predicted_memory < 40:
            # Can scale down
            return ScalingDecision(
                strategy=ScalingStrategy.HORIZONTAL,
                target_resources={
                    "workers": max(self.load_balancer.min_workers,
                                 self.load_balancer.current_workers - 1)
                },
                expected_improvement=0.0,  # No performance change expected
                cost_estimate=-0.3,        # 30% cost reduction
                confidence=confidence
            )
        
        return None
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        self.logger.info(f"Executing scaling decision: {decision.strategy.value}")
        
        if decision.strategy == ScalingStrategy.HORIZONTAL:
            target_workers = decision.target_resources.get("workers", 
                                                         self.load_balancer.current_workers)
            await self.load_balancer._scale_workers(target_workers)
        
        elif decision.strategy == ScalingStrategy.VERTICAL:
            await self._apply_vertical_optimizations(decision)
        
        # Record decision
        self.scaling_history.append(decision)
        
        # Keep history manageable
        if len(self.scaling_history) > 100:
            self.scaling_history.pop(0)
    
    async def _apply_vertical_optimizations(self, decision: ScalingDecision):
        """Apply vertical scaling optimizations."""
        self.logger.info("Applying vertical optimizations")
        
        # Optimize existing processes
        await self.performance_optimizer.optimize_performance()
        
        # Adjust resource limits if needed
        cpu_limit = decision.target_resources.get("cpu_limit")
        memory_limit = decision.target_resources.get("memory_limit")
        
        if memory_limit:
            await self._adjust_memory_limits(memory_limit)
    
    async def _adjust_memory_limits(self, memory_limit: float):
        """Adjust memory limits for processes."""
        self.logger.info(f"Adjusting memory limits to {memory_limit}%")
        # This would typically involve adjusting process-specific memory limits
    
    async def _optimize_resource_allocation(self):
        """Continuously optimize resource allocation."""
        while True:
            try:
                await asyncio.sleep(600)  # Optimize every 10 minutes
                
                # Analyze current allocation efficiency
                efficiency = await self._calculate_allocation_efficiency()
                
                if efficiency < 0.7:  # Less than 70% efficient
                    await self._rebalance_resources()
                
            except Exception as e:
                self.logger.error(f"Resource allocation optimization error: {e}")
    
    async def _calculate_allocation_efficiency(self) -> float:
        """Calculate current resource allocation efficiency."""
        if not self.resource_history:
            return 1.0
        
        recent_metrics = self.resource_history[-10:] if len(self.resource_history) >= 10 else self.resource_history
        
        # Calculate utilization balance
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        
        # Efficiency is higher when resources are well-balanced
        balance_score = 1.0 - abs(avg_cpu - avg_memory) / 100.0
        
        # Efficiency is higher when utilization is in optimal range (60-80%)
        optimal_cpu = 1.0 - abs(avg_cpu - 70) / 30.0 if avg_cpu <= 100 else 0.5
        optimal_memory = 1.0 - abs(avg_memory - 70) / 30.0 if avg_memory <= 100 else 0.5
        
        return max(0.0, min(1.0, (balance_score + optimal_cpu + optimal_memory) / 3.0))
    
    async def _rebalance_resources(self):
        """Rebalance resource allocation for better efficiency."""
        self.logger.info("Rebalancing resources for better efficiency")
        
        if not self.resource_history:
            return
        
        recent_metrics = self.resource_history[-10:]
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        
        if avg_cpu > avg_memory + 20:
            # CPU bottleneck - increase parallelism
            await self._increase_parallelism()
        elif avg_memory > avg_cpu + 20:
            # Memory bottleneck - optimize memory usage
            await self._optimize_memory_intensive_operations()
        else:
            # General optimization
            await self.performance_optimizer.optimize_performance()
    
    async def _increase_parallelism(self):
        """Increase parallelism to utilize available CPU."""
        self.logger.info("Increasing parallelism for CPU utilization")
        
        # Increase worker count if possible
        if self.load_balancer.current_workers < self.load_balancer.max_workers:
            await self.load_balancer._scale_workers(self.load_balancer.current_workers + 1)
    
    async def _optimize_memory_intensive_operations(self):
        """Optimize memory-intensive operations."""
        self.logger.info("Optimizing memory-intensive operations")
        
        # Reduce batch sizes
        await self._adjust_processing_parameters()
        
        # Clear caches more aggressively
        await self.performance_optimizer.clear_caches()


# Global instances
_load_balancer = None
_resource_manager = None


async def get_load_balancer() -> AdaptiveLoadBalancer:
    """Get global adaptive load balancer."""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = AdaptiveLoadBalancer()
        await _load_balancer.start()
    return _load_balancer


async def get_resource_manager() -> IntelligentResourceManager:
    """Get global intelligent resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = IntelligentResourceManager()
        await _resource_manager.start_management()
    return _resource_manager


# High-level functions for easy usage

async def auto_scale_task(task_func: Callable, *args, **kwargs) -> Any:
    """Execute task with automatic scaling."""
    load_balancer = await get_load_balancer()
    return await load_balancer.submit_task(task_func, *args, **kwargs)


async def optimize_resources():
    """Trigger immediate resource optimization."""
    resource_manager = await get_resource_manager()
    await resource_manager._optimize_resource_allocation()


# Decorators for scalability

def scalable_task(min_workers: int = 1, max_workers: int = None):
    """Decorator for scalable task execution."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            load_balancer = await get_load_balancer()
            if max_workers:
                load_balancer.max_workers = max_workers
            if min_workers:
                load_balancer.min_workers = min_workers
            
            return await load_balancer.submit_task(func, *args, **kwargs)
        
        return wrapper
    return decorator


def resource_optimized(monitor_resources: bool = True):
    """Decorator for resource-optimized execution."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            if monitor_resources:
                resource_manager = await get_resource_manager()
                # Resource monitoring is automatic
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator