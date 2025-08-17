"""Generation 3: Advanced Scalability and Auto-scaling Framework."""

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import queue


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"


class ScalingEvent(Enum):
    """Auto-scaling events."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REBALANCE = "rebalance"


@dataclass
class WorkerNode:
    """Represents a worker node in the system."""
    node_id: str
    capacity: int = 100
    current_load: int = 0
    status: str = "active"
    last_heartbeat: float = field(default_factory=time.time)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def utilization(self) -> float:
        """Get current utilization percentage."""
        return (self.current_load / self.capacity * 100) if self.capacity > 0 else 0
    
    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        return max(0, self.capacity - self.current_load)
    
    def add_load(self, amount: int = 1) -> bool:
        """Add load to the node."""
        if self.current_load + amount <= self.capacity:
            self.current_load += amount
            return True
        return False
    
    def remove_load(self, amount: int = 1) -> None:
        """Remove load from the node."""
        self.current_load = max(0, self.current_load - amount)
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time()
    
    def is_healthy(self, timeout_seconds: int = 30) -> bool:
        """Check if node is healthy based on heartbeat."""
        return time.time() - self.last_heartbeat < timeout_seconds


@dataclass
class Task:
    """Represents a task in the system."""
    task_id: str
    payload: Any
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    status: str = "pending"
    retry_count: int = 0
    max_retries: int = 3
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries


class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.nodes: Dict[str, WorkerNode] = {}
        self.node_weights: Dict[str, float] = {}
        self.round_robin_index = 0
        self.lock = threading.Lock()
        
        # Statistics
        self.request_count = 0
        self.distribution_stats = defaultdict(int)
    
    def add_node(self, node: WorkerNode, weight: float = 1.0) -> None:
        """Add a worker node to the load balancer."""
        with self.lock:
            self.nodes[node.node_id] = node
            self.node_weights[node.node_id] = weight
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a worker node."""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.node_weights.pop(node_id, None)
                return True
            return False
    
    def select_node(self, task: Optional[Task] = None) -> Optional[WorkerNode]:
        """Select the best node for a task based on the strategy."""
        with self.lock:
            healthy_nodes = [
                node for node in self.nodes.values() 
                if node.is_healthy() and node.status == "active"
            ]
            
            if not healthy_nodes:
                return None
            
            self.request_count += 1
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                selected = self._round_robin_select(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                selected = self._least_connections_select(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                selected = self._weighted_round_robin_select(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                selected = self._resource_based_select(healthy_nodes, task)
            else:
                selected = healthy_nodes[0]
            
            if selected:
                self.distribution_stats[selected.node_id] += 1
            
            return selected
    
    def _round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Round-robin selection."""
        selected = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return selected
    
    def _least_connections_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least connections."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _weighted_round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin selection."""
        # Simple weighted selection based on capacity
        weighted_nodes = []
        for node in nodes:
            weight = self.node_weights.get(node.node_id, 1.0)
            # Repeat node based on weight
            repeats = max(1, int(weight * 10))
            weighted_nodes.extend([node] * repeats)
        
        if weighted_nodes:
            selected = weighted_nodes[self.round_robin_index % len(weighted_nodes)]
            self.round_robin_index += 1
            return selected
        
        return nodes[0]
    
    def _resource_based_select(self, nodes: List[WorkerNode], task: Optional[Task]) -> WorkerNode:
        """Select based on available resources and task requirements."""
        # Consider both available capacity and performance metrics
        def score_node(node: WorkerNode) -> float:
            utilization_score = (100 - node.utilization) / 100
            
            # Factor in performance metrics if available
            perf_score = 1.0
            if 'avg_response_time' in node.performance_metrics:
                # Lower response time is better
                response_time = node.performance_metrics['avg_response_time']
                perf_score = max(0.1, 1.0 / (1.0 + response_time))
            
            return utilization_score * perf_score
        
        return max(nodes, key=score_node)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            total_capacity = sum(node.capacity for node in self.nodes.values())
            total_load = sum(node.current_load for node in self.nodes.values())
            
            return {
                'strategy': self.strategy.value,
                'total_nodes': len(self.nodes),
                'healthy_nodes': len([n for n in self.nodes.values() if n.is_healthy()]),
                'total_capacity': total_capacity,
                'total_load': total_load,
                'overall_utilization': (total_load / total_capacity * 100) if total_capacity > 0 else 0,
                'request_count': self.request_count,
                'distribution_stats': dict(self.distribution_stats),
                'node_details': {
                    node_id: {
                        'utilization': node.utilization,
                        'capacity': node.capacity,
                        'current_load': node.current_load,
                        'status': node.status
                    }
                    for node_id, node in self.nodes.items()
                }
            }


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, min_nodes: int = 1, max_nodes: int = 10):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        
        # Scaling thresholds
        self.scale_up_threshold = 80.0  # CPU/utilization percentage
        self.scale_down_threshold = 30.0
        self.scale_up_cooldown = 60  # seconds
        self.scale_down_cooldown = 180  # seconds
        
        # State tracking
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.scaling_history: deque = deque(maxlen=100)
        
        self.lock = threading.Lock()
    
    def should_scale_up(self, current_utilization: float, 
                       current_nodes: int, pending_tasks: int = 0) -> bool:
        """Determine if we should scale up."""
        with self.lock:
            current_time = time.time()
            
            # Check cooldown
            if current_time - self.last_scale_up < self.scale_up_cooldown:
                return False
            
            # Check if we're at max capacity
            if current_nodes >= self.max_nodes:
                return False
            
            # Check utilization threshold
            if current_utilization > self.scale_up_threshold:
                return True
            
            # Check pending tasks
            if pending_tasks > current_nodes * 2:  # More than 2 tasks per node
                return True
            
            return False
    
    def should_scale_down(self, current_utilization: float, 
                         current_nodes: int) -> bool:
        """Determine if we should scale down."""
        with self.lock:
            current_time = time.time()
            
            # Check cooldown
            if current_time - self.last_scale_down < self.scale_down_cooldown:
                return False
            
            # Check if we're at min capacity
            if current_nodes <= self.min_nodes:
                return False
            
            # Check utilization threshold
            if current_utilization < self.scale_down_threshold:
                return True
            
            return False
    
    def record_scaling_event(self, event_type: ScalingEvent, 
                           details: Dict[str, Any]) -> None:
        """Record a scaling event."""
        with self.lock:
            current_time = time.time()
            
            event_record = {
                'timestamp': current_time,
                'event_type': event_type.value,
                'details': details
            }
            
            self.scaling_history.append(event_record)
            
            if event_type == ScalingEvent.SCALE_UP:
                self.last_scale_up = current_time
            elif event_type == ScalingEvent.SCALE_DOWN:
                self.last_scale_down = current_time
    
    def get_scaling_recommendation(self, system_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get scaling recommendation based on system metrics."""
        current_utilization = system_metrics.get('overall_utilization', 0)
        current_nodes = system_metrics.get('healthy_nodes', 0)
        pending_tasks = system_metrics.get('pending_tasks', 0)
        
        if self.should_scale_up(current_utilization, current_nodes, pending_tasks):
            return {
                'action': 'scale_up',
                'current_nodes': current_nodes,
                'recommended_nodes': min(current_nodes + 1, self.max_nodes),
                'reason': f'High utilization: {current_utilization:.1f}%',
                'confidence': min(100, (current_utilization - self.scale_up_threshold) * 2)
            }
        
        elif self.should_scale_down(current_utilization, current_nodes):
            return {
                'action': 'scale_down',
                'current_nodes': current_nodes,
                'recommended_nodes': max(current_nodes - 1, self.min_nodes),
                'reason': f'Low utilization: {current_utilization:.1f}%',
                'confidence': min(100, (self.scale_down_threshold - current_utilization) * 2)
            }
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        with self.lock:
            recent_events = [e for e in self.scaling_history 
                           if time.time() - e['timestamp'] < 3600]  # Last hour
            
            return {
                'min_nodes': self.min_nodes,
                'max_nodes': self.max_nodes,
                'scale_up_threshold': self.scale_up_threshold,
                'scale_down_threshold': self.scale_down_threshold,
                'last_scale_up': self.last_scale_up,
                'last_scale_down': self.last_scale_down,
                'total_scaling_events': len(self.scaling_history),
                'recent_events': len(recent_events),
                'event_types': {
                    event_type.value: len([e for e in recent_events 
                                         if e['event_type'] == event_type.value])
                    for event_type in ScalingEvent
                }
            }


class DistributedTaskManager:
    """Distributed task management with load balancing and auto-scaling."""
    
    def __init__(self, load_balancer: Optional[LoadBalancer] = None,
                 auto_scaler: Optional[AutoScaler] = None):
        self.load_balancer = load_balancer or LoadBalancer()
        self.auto_scaler = auto_scaler or AutoScaler()
        
        # Task queues
        self.pending_tasks: queue.PriorityQueue = queue.PriorityQueue()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.failed_tasks: deque = deque(maxlen=1000)
        
        # Task processing
        self.task_processors: Dict[str, Callable] = {}
        self.default_processor: Optional[Callable] = None
        
        # State management
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_processing_time': 0.0
        }
    
    def register_processor(self, task_type: str, processor: Callable) -> None:
        """Register a task processor for a specific task type."""
        self.task_processors[task_type] = processor
    
    def set_default_processor(self, processor: Callable) -> None:
        """Set the default task processor."""
        self.default_processor = processor
    
    def submit_task(self, task: Task) -> bool:
        """Submit a task for processing."""
        with self.lock:
            # Priority queue uses (priority, item) tuples
            # Lower number = higher priority
            priority = -task.priority  # Negative for higher priority first
            self.pending_tasks.put((priority, task))
            self.stats['tasks_submitted'] += 1
            return True
    
    def start_processing(self, num_workers: int = 4) -> None:
        """Start task processing with specified number of workers."""
        if self.is_running:
            return
        
        self.is_running = True
        
        for i in range(num_workers):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker_thread.start()
            self.worker_threads.append(worker_thread)
    
    def stop_processing(self) -> None:
        """Stop task processing."""
        self.is_running = False
        
        # Wait for workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        self.worker_threads.clear()
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        while self.is_running:
            try:
                # Get next task (blocking with timeout)
                priority, task = self.pending_tasks.get(timeout=1)
                
                # Find a suitable node
                node = self.load_balancer.select_node(task)
                if not node or not node.add_load(1):
                    # No available node, put task back
                    self.pending_tasks.put((priority, task))
                    time.sleep(0.1)
                    continue
                
                # Assign task to node
                task.assigned_node = node.node_id
                task.status = "running"
                
                with self.lock:
                    self.running_tasks[task.task_id] = task
                
                # Process task
                self._process_task(task, node)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker error: {e}")
    
    def _process_task(self, task: Task, node: WorkerNode) -> None:
        """Process a single task."""
        start_time = time.time()
        
        try:
            # Get processor
            processor = self.task_processors.get(
                getattr(task.payload, 'type', 'default'),
                self.default_processor
            )
            
            if not processor:
                raise ValueError("No processor available for task")
            
            # Execute task
            result = processor(task.payload)
            
            # Mark as completed
            task.status = "completed"
            processing_time = time.time() - start_time
            
            with self.lock:
                self.running_tasks.pop(task.task_id, None)
                self.completed_tasks.append(task)
                self.stats['tasks_completed'] += 1
                
                # Update average processing time
                total_time = (self.stats['average_processing_time'] * 
                             (self.stats['tasks_completed'] - 1) + processing_time)
                self.stats['average_processing_time'] = total_time / self.stats['tasks_completed']
            
            # Update node performance
            node.performance_metrics['avg_response_time'] = processing_time
            
        except Exception as e:
            # Handle task failure
            task.status = "failed"
            task.retry_count += 1
            
            with self.lock:
                self.running_tasks.pop(task.task_id, None)
                
                if task.can_retry():
                    # Retry task
                    self.pending_tasks.put((-task.priority, task))
                else:
                    # Task failed permanently
                    self.failed_tasks.append(task)
                    self.stats['tasks_failed'] += 1
            
            logging.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Release node load
            node.remove_load(1)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        lb_stats = self.load_balancer.get_stats()
        scaler_stats = self.auto_scaler.get_stats()
        
        with self.lock:
            pending_count = self.pending_tasks.qsize()
            running_count = len(self.running_tasks)
            
            return {
                'load_balancer': lb_stats,
                'auto_scaler': scaler_stats,
                'task_stats': self.stats.copy(),
                'queue_stats': {
                    'pending_tasks': pending_count,
                    'running_tasks': running_count,
                    'completed_tasks': len(self.completed_tasks),
                    'failed_tasks': len(self.failed_tasks)
                },
                'overall_utilization': lb_stats.get('overall_utilization', 0),
                'healthy_nodes': lb_stats.get('healthy_nodes', 0)
            }
    
    def auto_scale_check(self) -> Optional[Dict[str, Any]]:
        """Check if auto-scaling is needed and return recommendation."""
        metrics = self.get_system_metrics()
        return self.auto_scaler.get_scaling_recommendation(metrics)


# Global distributed task manager
_global_task_manager: Optional[DistributedTaskManager] = None


def get_distributed_task_manager() -> DistributedTaskManager:
    """Get the global distributed task manager."""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = DistributedTaskManager()
    return _global_task_manager


def distributed_task(task_type: str = "default", priority: int = 0):
    """Decorator for distributed task execution."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            task_manager = get_distributed_task_manager()
            
            # Create task
            task = Task(
                task_id=f"{func.__name__}_{int(time.time() * 1000)}",
                payload={'type': task_type, 'func': func, 'args': args, 'kwargs': kwargs},
                priority=priority
            )
            
            # Submit task
            task_manager.submit_task(task)
            
            return task
        
        return wrapper
    return decorator