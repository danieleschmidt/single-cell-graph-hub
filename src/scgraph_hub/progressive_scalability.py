"""Progressive Scalability Framework - TERRAGON SDLC v6.0 Enhancement.

High-performance distributed processing system that scales dynamically based on
workload patterns and system metrics with intelligent load balancing.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import multiprocessing
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib

try:
    from .progressive_quality_gates import ProgressiveLevel
    from .progressive_resilience import ResilienceStrategy, ProgressiveResilienceOrchestrator
except ImportError:
    from enum import Enum
    
    class ProgressiveLevel(Enum):
        BASIC = "basic"
        INTERMEDIATE = "intermediate"
        ADVANCED = "advanced"
        EXPERT = "expert"
        AUTONOMOUS = "autonomous"
    
    class ResilienceStrategy(Enum):
        CIRCUIT_BREAKER = "circuit_breaker"


class ScalingStrategy(Enum):
    """Scaling strategy types."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    AUTO_ELASTIC = "auto_elastic"
    PREDICTIVE = "predictive"
    WORKLOAD_AWARE = "workload_aware"


class LoadBalancingMethod(Enum):
    """Load balancing methods."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"


class WorkloadType(Enum):
    """Workload classification types."""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    BALANCED = "balanced"


@dataclass
class WorkerMetrics:
    """Metrics for a worker instance."""
    worker_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_response_time: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)
    capacity_score: float = 1.0
    health_score: float = 1.0
    
    def calculate_load_score(self) -> float:
        """Calculate overall load score (0-1, lower is better)."""
        cpu_score = self.cpu_usage / 100.0
        memory_score = self.memory_usage / 100.0
        task_score = min(self.active_tasks / 10.0, 1.0)  # Normalize to 10 max tasks
        response_score = min(self.avg_response_time / 1000.0, 1.0)  # Normalize to 1s
        
        return (cpu_score + memory_score + task_score + response_score) / 4.0
    
    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        return (
            self.health_score > 0.5 and
            self.cpu_usage < 95.0 and
            self.memory_usage < 95.0 and
            (datetime.utcnow() - self.last_update).total_seconds() < 30
        )


@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: datetime
    event_type: str
    strategy: ScalingStrategy
    reason: str
    worker_count_before: int
    worker_count_after: int
    metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'strategy': self.strategy.value,
            'reason': self.reason,
            'worker_count_before': self.worker_count_before,
            'worker_count_after': self.worker_count_after,
            'metrics': self.metrics,
            'success': self.success
        }


@dataclass
class Task:
    """Distributed task representation."""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    workload_type: WorkloadType = WorkloadType.BALANCED
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_worker: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    
    def calculate_hash(self) -> str:
        """Calculate consistent hash for task."""
        content = f"{self.function.__name__}_{str(self.args)}_{str(self.kwargs)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if not self.timeout:
            return False
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.timeout


class ProgressiveLoadBalancer:
    """Intelligent load balancer with adaptive algorithms."""
    
    def __init__(self, method: LoadBalancingMethod = LoadBalancingMethod.ADAPTIVE_WEIGHTED):
        self.method = method
        self.workers: Dict[str, WorkerMetrics] = {}
        self.task_history = deque(maxlen=1000)
        self.performance_cache = {}
        self.lock = threading.Lock()
    
    def register_worker(self, worker_id: str, capacity: float = 1.0):
        """Register a new worker."""
        with self.lock:
            self.workers[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                capacity_score=capacity
            )
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
    
    def update_worker_metrics(self, worker_id: str, metrics: Dict[str, float]):
        """Update worker metrics."""
        with self.lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.cpu_usage = metrics.get('cpu_usage', worker.cpu_usage)
                worker.memory_usage = metrics.get('memory_usage', worker.memory_usage)
                worker.active_tasks = metrics.get('active_tasks', worker.active_tasks)
                worker.avg_response_time = metrics.get('avg_response_time', worker.avg_response_time)
                worker.health_score = metrics.get('health_score', worker.health_score)
                worker.last_update = datetime.utcnow()
    
    def select_worker(self, task: Task) -> Optional[str]:
        """Select best worker for task based on load balancing method."""
        with self.lock:
            healthy_workers = {
                wid: worker for wid, worker in self.workers.items()
                if worker.is_healthy()
            }
            
            if not healthy_workers:
                return None
            
            if self.method == LoadBalancingMethod.ROUND_ROBIN:
                return self._round_robin_selection(healthy_workers)
            elif self.method == LoadBalancingMethod.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_workers)
            elif self.method == LoadBalancingMethod.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection(healthy_workers)
            elif self.method == LoadBalancingMethod.CONSISTENT_HASH:
                return self._consistent_hash_selection(healthy_workers, task)
            else:  # ADAPTIVE_WEIGHTED
                return self._adaptive_weighted_selection(healthy_workers, task)
    
    def _round_robin_selection(self, workers: Dict[str, WorkerMetrics]) -> str:
        """Simple round-robin selection."""
        worker_ids = list(workers.keys())
        if hasattr(self, '_rr_index'):
            self._rr_index = (self._rr_index + 1) % len(worker_ids)
        else:
            self._rr_index = 0
        return worker_ids[self._rr_index]
    
    def _least_connections_selection(self, workers: Dict[str, WorkerMetrics]) -> str:
        """Select worker with least active connections."""
        return min(workers.keys(), key=lambda wid: workers[wid].active_tasks)
    
    def _least_response_time_selection(self, workers: Dict[str, WorkerMetrics]) -> str:
        """Select worker with best response time."""
        return min(workers.keys(), key=lambda wid: workers[wid].avg_response_time)
    
    def _consistent_hash_selection(self, workers: Dict[str, WorkerMetrics], task: Task) -> str:
        """Select worker using consistent hashing."""
        task_hash = int(task.calculate_hash(), 16)
        worker_ids = sorted(workers.keys())
        
        # Simple consistent hash implementation
        selected_index = task_hash % len(worker_ids)
        return worker_ids[selected_index]
    
    def _adaptive_weighted_selection(self, workers: Dict[str, WorkerMetrics], task: Task) -> str:
        """Adaptive weighted selection based on performance and workload."""
        scores = {}
        
        for worker_id, worker in workers.items():
            # Base score from load
            load_score = worker.calculate_load_score()
            
            # Capacity adjustment
            capacity_bonus = worker.capacity_score
            
            # Workload type affinity
            workload_bonus = self._calculate_workload_affinity(worker, task.workload_type)
            
            # Performance history bonus
            perf_bonus = self._get_performance_bonus(worker_id, task.function.__name__)
            
            # Combined score (lower is better for selection)
            combined_score = load_score - (capacity_bonus + workload_bonus + perf_bonus) * 0.1
            scores[worker_id] = max(0.01, combined_score)  # Prevent zero scores
        
        # Weighted random selection (inverse probability - lower score = higher chance)
        total_weight = sum(1.0 / score for score in scores.values())
        
        import random
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for worker_id, score in scores.items():
            cumulative += 1.0 / score
            if r <= cumulative:
                return worker_id
        
        # Fallback to best score
        return min(scores.keys(), key=lambda wid: scores[wid])
    
    def _calculate_workload_affinity(self, worker: WorkerMetrics, workload_type: WorkloadType) -> float:
        """Calculate worker affinity for workload type."""
        # Simple heuristic based on current resource usage
        if workload_type == WorkloadType.CPU_INTENSIVE:
            return 1.0 - (worker.cpu_usage / 100.0)
        elif workload_type == WorkloadType.MEMORY_INTENSIVE:
            return 1.0 - (worker.memory_usage / 100.0)
        elif workload_type == WorkloadType.IO_INTENSIVE:
            return 1.0 - min(worker.active_tasks / 5.0, 1.0)
        else:  # BALANCED or others
            return worker.health_score
    
    def _get_performance_bonus(self, worker_id: str, function_name: str) -> float:
        """Get performance bonus for worker-function combination."""
        key = f"{worker_id}:{function_name}"
        return self.performance_cache.get(key, 0.0)
    
    def record_task_completion(self, worker_id: str, task: Task, 
                             execution_time: float, success: bool):
        """Record task completion for learning."""
        with self.lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                if success:
                    worker.completed_tasks += 1
                    # Update average response time (exponential moving average)
                    alpha = 0.1
                    worker.avg_response_time = (
                        (1 - alpha) * worker.avg_response_time + 
                        alpha * execution_time
                    )
                else:
                    worker.failed_tasks += 1
            
            # Update performance cache
            cache_key = f"{worker_id}:{task.function.__name__}"
            if cache_key in self.performance_cache:
                # Exponential moving average for performance
                alpha = 0.1
                current_bonus = self.performance_cache[cache_key]
                performance_score = 1.0 / max(execution_time, 0.001) if success else -0.1
                self.performance_cache[cache_key] = (
                    (1 - alpha) * current_bonus + alpha * performance_score
                )
            else:
                self.performance_cache[cache_key] = 1.0 / max(execution_time, 0.001) if success else -0.1


class ProgressiveAutoScaler:
    """Auto-scaler that adapts based on workload patterns."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 10,
                 target_utilization: float = 0.7):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scaling_history = deque(maxlen=100)
        self.metrics_history = deque(maxlen=1000)
        self.last_scaling_time = None
        self.cooldown_period = 60.0  # seconds
        
    def should_scale_up(self, current_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Determine if should scale up."""
        avg_utilization = current_metrics.get('avg_utilization', 0.0)
        queue_length = current_metrics.get('queue_length', 0)
        avg_response_time = current_metrics.get('avg_response_time', 0.0)
        
        reasons = []
        
        # Utilization-based scaling
        if avg_utilization > self.target_utilization + 0.1:
            reasons.append(f"High utilization: {avg_utilization:.1%}")
        
        # Queue-based scaling
        if queue_length > 10:
            reasons.append(f"Long queue: {queue_length} tasks")
        
        # Response time-based scaling
        if avg_response_time > 1000.0:  # 1 second
            reasons.append(f"High latency: {avg_response_time:.0f}ms")
        
        # Predictive scaling based on trends
        if len(self.metrics_history) >= 10:
            recent_trend = self._calculate_utilization_trend()
            if recent_trend > 0.1:  # Increasing trend
                reasons.append(f"Rising utilization trend: {recent_trend:.1%}")
        
        should_scale = len(reasons) > 0 and self._can_scale()
        return should_scale, "; ".join(reasons)
    
    def should_scale_down(self, current_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Determine if should scale down."""
        avg_utilization = current_metrics.get('avg_utilization', 0.0)
        queue_length = current_metrics.get('queue_length', 0)
        worker_count = current_metrics.get('worker_count', 1)
        
        reasons = []
        
        # Only scale down if well below target and no queue
        if (avg_utilization < self.target_utilization - 0.2 and 
            queue_length == 0 and 
            worker_count > self.min_workers):
            reasons.append(f"Low utilization: {avg_utilization:.1%}")
        
        # Predictive scaling down based on trends
        if len(self.metrics_history) >= 10:
            recent_trend = self._calculate_utilization_trend()
            if recent_trend < -0.1 and avg_utilization < 0.3:
                reasons.append(f"Declining utilization trend: {recent_trend:.1%}")
        
        should_scale = len(reasons) > 0 and self._can_scale()
        return should_scale, "; ".join(reasons)
    
    def _can_scale(self) -> bool:
        """Check if scaling is allowed (cooldown period)."""
        if not self.last_scaling_time:
            return True
        
        elapsed = (datetime.utcnow() - self.last_scaling_time).total_seconds()
        return elapsed >= self.cooldown_period
    
    def _calculate_utilization_trend(self) -> float:
        """Calculate recent utilization trend."""
        if len(self.metrics_history) < 10:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-10:]
        utilizations = [m.get('avg_utilization', 0.0) for m in recent_metrics]
        
        # Simple linear trend
        n = len(utilizations)
        sum_x = sum(range(n))
        sum_y = sum(utilizations)
        sum_xy = sum(i * y for i, y in enumerate(utilizations))
        sum_x2 = sum(i * i for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def record_scaling_event(self, event: ScalingEvent):
        """Record a scaling event."""
        self.scaling_history.append(event)
        self.last_scaling_time = event.timestamp
    
    def record_metrics(self, metrics: Dict[str, float]):
        """Record metrics for trend analysis."""
        self.metrics_history.append({
            'timestamp': datetime.utcnow(),
            **metrics
        })


class ProgressiveDistributedProcessor:
    """High-performance distributed task processor."""
    
    def __init__(self, level: ProgressiveLevel = ProgressiveLevel.BASIC):
        self.level = level
        self.load_balancer = ProgressiveLoadBalancer()
        self.auto_scaler = ProgressiveAutoScaler()
        self.task_queue = asyncio.Queue()
        self.result_store: Dict[str, Any] = {}
        self.worker_pools: Dict[str, Union[ThreadPoolExecutor, ProcessPoolExecutor]] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0,
            'current_utilization': 0.0
        }
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._processor_task = None
        
    async def start(self):
        """Start the distributed processor."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize worker pools based on level
        await self._initialize_workers()
        
        # Start task processor
        self._processor_task = asyncio.create_task(self._process_tasks())
        
        # Start metrics collection
        asyncio.create_task(self._collect_metrics())
        
        self.logger.info(f"Progressive distributed processor started at {self.level.value} level")
    
    async def stop(self):
        """Stop the distributed processor."""
        if not self._running:
            return
        
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown worker pools
        for pool in self.worker_pools.values():
            pool.shutdown(wait=True)
        
        self.worker_pools.clear()
        self.logger.info("Progressive distributed processor stopped")
    
    async def _initialize_workers(self):
        """Initialize worker pools based on current level."""
        level_config = {
            ProgressiveLevel.BASIC: {'thread_workers': 2, 'process_workers': 0},
            ProgressiveLevel.INTERMEDIATE: {'thread_workers': 4, 'process_workers': 1},
            ProgressiveLevel.ADVANCED: {'thread_workers': 8, 'process_workers': 2},
            ProgressiveLevel.EXPERT: {'thread_workers': 16, 'process_workers': 4},
            ProgressiveLevel.AUTONOMOUS: {'thread_workers': 32, 'process_workers': 8}
        }
        
        config = level_config.get(self.level, level_config[ProgressiveLevel.BASIC])
        
        # Create thread pools
        for i in range(config['thread_workers']):
            worker_id = f"thread_worker_{i}"
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=worker_id)
            self.worker_pools[worker_id] = pool
            self.load_balancer.register_worker(worker_id, capacity=1.0)
        
        # Create process pools
        for i in range(config['process_workers']):
            worker_id = f"process_worker_{i}"
            pool = ProcessPoolExecutor(max_workers=1)
            self.worker_pools[worker_id] = pool
            self.load_balancer.register_worker(worker_id, capacity=1.5)  # Higher capacity
        
        self.logger.info(
            f"Initialized {len(self.worker_pools)} workers "
            f"({config['thread_workers']} thread, {config['process_workers']} process)"
        )
    
    async def submit_task(self, function: Callable, *args, **kwargs) -> str:
        """Submit a task for distributed processing."""
        task = Task(
            task_id=f"task_{int(time.time() * 1000000)}_{len(self.active_tasks)}",
            function=function,
            args=args,
            kwargs=kwargs,
            workload_type=self._classify_workload(function),
            timeout=kwargs.pop('task_timeout', None),
            max_retries=kwargs.pop('max_retries', 3)
        )
        
        await self.task_queue.put(task)
        self.active_tasks[task.task_id] = task
        self.performance_metrics['total_tasks'] += 1
        
        return task.task_id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result."""
        start_time = time.time()
        
        while task_id in self.active_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            await asyncio.sleep(0.1)
        
        if task_id in self.result_store:
            result = self.result_store.pop(task_id)
            if isinstance(result, Exception):
                raise result
            return result
        
        raise KeyError(f"Task {task_id} not found")
    
    def _classify_workload(self, function: Callable) -> WorkloadType:
        """Classify workload type based on function."""
        # Simple heuristic based on function name
        func_name = function.__name__.lower()
        
        if any(keyword in func_name for keyword in ['compute', 'calculate', 'process']):
            return WorkloadType.CPU_INTENSIVE
        elif any(keyword in func_name for keyword in ['read', 'write', 'file', 'io']):
            return WorkloadType.IO_INTENSIVE
        elif any(keyword in func_name for keyword in ['request', 'fetch', 'download']):
            return WorkloadType.NETWORK_INTENSIVE
        elif any(keyword in func_name for keyword in ['memory', 'cache', 'store']):
            return WorkloadType.MEMORY_INTENSIVE
        else:
            return WorkloadType.BALANCED
    
    async def _process_tasks(self):
        """Main task processing loop."""
        while self._running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Check if task expired
                if task.is_expired():
                    self._handle_task_completion(task, None, Exception("Task expired"), 0.0)
                    continue
                
                # Select worker
                worker_id = self.load_balancer.select_worker(task)
                if not worker_id:
                    # No healthy workers available, requeue task
                    await asyncio.sleep(1.0)
                    await self.task_queue.put(task)
                    continue
                
                # Execute task on selected worker
                asyncio.create_task(self._execute_task(task, worker_id))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in task processing loop: {e}")
    
    async def _execute_task(self, task: Task, worker_id: str):
        """Execute task on specific worker."""
        task.assigned_worker = worker_id
        task.started_at = datetime.utcnow()
        
        start_time = time.time()
        
        try:
            # Get worker pool
            pool = self.worker_pools[worker_id]
            
            # Execute task
            loop = asyncio.get_event_loop()
            if isinstance(pool, ThreadPoolExecutor):
                result = await loop.run_in_executor(
                    pool, self._execute_function, task.function, task.args, task.kwargs
                )
            else:  # ProcessPoolExecutor
                result = await loop.run_in_executor(
                    pool, task.function, *task.args
                )
            
            execution_time = (time.time() - start_time) * 1000  # ms
            task.completed_at = datetime.utcnow()
            task.result = result
            
            self._handle_task_completion(task, result, None, execution_time)
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000  # ms
            task.error = e
            task.completed_at = datetime.utcnow()
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.assigned_worker = None
                task.started_at = None
                await self.task_queue.put(task)
                return
            
            self._handle_task_completion(task, None, e, execution_time)
    
    def _execute_function(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function in thread pool."""
        return func(*args, **kwargs)
    
    def _handle_task_completion(self, task: Task, result: Any, 
                              error: Optional[Exception], execution_time: float):
        """Handle task completion."""
        # Update metrics
        if error:
            self.performance_metrics['failed_tasks'] += 1
        else:
            self.performance_metrics['completed_tasks'] += 1
        
        # Update average execution time (exponential moving average)
        alpha = 0.1
        current_avg = self.performance_metrics['avg_execution_time']
        self.performance_metrics['avg_execution_time'] = (
            (1 - alpha) * current_avg + alpha * execution_time
        )
        
        # Record in load balancer
        if task.assigned_worker:
            self.load_balancer.record_task_completion(
                task.assigned_worker, task, execution_time, error is None
            )
        
        # Store result
        self.result_store[task.task_id] = error if error else result
        
        # Remove from active tasks
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
    
    async def _collect_metrics(self):
        """Collect and analyze system metrics."""
        while self._running:
            try:
                # Calculate current metrics
                total_workers = len(self.worker_pools)
                active_task_count = len(self.active_tasks)
                queue_size = self.task_queue.qsize()
                
                # Calculate utilization
                utilization = min(active_task_count / max(total_workers, 1), 1.0)
                self.performance_metrics['current_utilization'] = utilization
                
                # Update worker metrics (mock data for now)
                for worker_id in self.worker_pools:
                    self.load_balancer.update_worker_metrics(worker_id, {
                        'cpu_usage': min(utilization * 100 + 10, 95),
                        'memory_usage': min(utilization * 80 + 15, 90),
                        'active_tasks': active_task_count // total_workers,
                        'health_score': 1.0 if utilization < 0.9 else 0.5
                    })
                
                # Record metrics for auto-scaler
                current_metrics = {
                    'avg_utilization': utilization,
                    'queue_length': queue_size,
                    'worker_count': total_workers,
                    'avg_response_time': self.performance_metrics['avg_execution_time']
                }
                
                self.auto_scaler.record_metrics(current_metrics)
                
                # Check scaling decisions
                await self._check_scaling(current_metrics)
                
                await asyncio.sleep(5.0)  # Collect metrics every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_scaling(self, metrics: Dict[str, float]):
        """Check if scaling is needed."""
        # Only scale if we're at intermediate level or higher
        if self.level == ProgressiveLevel.BASIC:
            return
        
        # Check scale up
        should_scale_up, reason = self.auto_scaler.should_scale_up(metrics)
        if should_scale_up and len(self.worker_pools) < self.auto_scaler.max_workers:
            await self._scale_up(reason)
        
        # Check scale down  
        should_scale_down, reason = self.auto_scaler.should_scale_down(metrics)
        if should_scale_down and len(self.worker_pools) > self.auto_scaler.min_workers:
            await self._scale_down(reason)
    
    async def _scale_up(self, reason: str):
        """Scale up by adding workers."""
        current_count = len(self.worker_pools)
        new_worker_id = f"dynamic_worker_{int(time.time())}"
        
        # Create new thread worker
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=new_worker_id)
        self.worker_pools[new_worker_id] = pool
        self.load_balancer.register_worker(new_worker_id, capacity=1.0)
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=datetime.utcnow(),
            event_type="scale_up",
            strategy=ScalingStrategy.AUTO_ELASTIC,
            reason=reason,
            worker_count_before=current_count,
            worker_count_after=len(self.worker_pools)
        )
        
        self.auto_scaler.record_scaling_event(event)
        self.logger.info(f"Scaled up: {reason} (workers: {current_count} -> {len(self.worker_pools)})")
    
    async def _scale_down(self, reason: str):
        """Scale down by removing workers."""
        current_count = len(self.worker_pools)
        
        # Find a dynamic worker to remove
        dynamic_workers = [
            wid for wid in self.worker_pools.keys() 
            if wid.startswith('dynamic_worker_')
        ]
        
        if dynamic_workers:
            worker_id = dynamic_workers[0]
            
            # Shutdown and remove worker
            pool = self.worker_pools[worker_id]
            pool.shutdown(wait=False)
            del self.worker_pools[worker_id]
            self.load_balancer.unregister_worker(worker_id)
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=datetime.utcnow(),
                event_type="scale_down",
                strategy=ScalingStrategy.AUTO_ELASTIC,
                reason=reason,
                worker_count_before=current_count,
                worker_count_after=len(self.worker_pools)
            )
            
            self.auto_scaler.record_scaling_event(event)
            self.logger.info(f"Scaled down: {reason} (workers: {current_count} -> {len(self.worker_pools)})")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processor status."""
        healthy_workers = sum(
            1 for worker_id in self.worker_pools.keys()
            if worker_id in self.load_balancer.workers and
            self.load_balancer.workers[worker_id].is_healthy()
        )
        
        return {
            'level': self.level.value,
            'running': self._running,
            'total_workers': len(self.worker_pools),
            'healthy_workers': healthy_workers,
            'active_tasks': len(self.active_tasks),
            'queue_size': self.task_queue.qsize() if self._running else 0,
            'performance_metrics': self.performance_metrics.copy(),
            'load_balancing_method': self.load_balancer.method.value,
            'scaling_history_count': len(self.auto_scaler.scaling_history)
        }


# Convenience functions and decorators
_global_processor = None

def get_progressive_processor() -> ProgressiveDistributedProcessor:
    """Get global progressive processor instance."""
    global _global_processor
    if _global_processor is None:
        _global_processor = ProgressiveDistributedProcessor()
    return _global_processor


async def distributed_execute(func: Callable, *args, **kwargs) -> Any:
    """Execute function in distributed processor."""
    processor = get_progressive_processor()
    
    if not processor._running:
        await processor.start()
    
    task_id = await processor.submit_task(func, *args, **kwargs)
    return await processor.get_result(task_id)


def distributed_task(timeout: Optional[float] = None, max_retries: int = 3):
    """Decorator for distributed task execution."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            kwargs['task_timeout'] = timeout
            kwargs['max_retries'] = max_retries
            return await distributed_execute(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            kwargs['task_timeout'] = timeout
            kwargs['max_retries'] = max_retries
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in event loop, create a future
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            distributed_execute(func, *args, **kwargs)
                        )
                        return future.result()
                else:
                    return loop.run_until_complete(distributed_execute(func, *args, **kwargs))
            except RuntimeError:
                return asyncio.run(distributed_execute(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator