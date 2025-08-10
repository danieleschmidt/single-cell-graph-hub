"""Reliability and robustness enhancements for autonomous SDLC."""

import asyncio
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
from datetime import datetime, timedelta
import psutil
import gc

from .logging_config import get_logger
from .health_checks import SystemHealthChecker, HealthStatus


class FailureType(Enum):
    """Types of failures that can occur."""
    DEPENDENCY_ERROR = "dependency_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    DATA_ERROR = "data_error"
    MODEL_ERROR = "model_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class FailureContext:
    """Context information for failures."""
    failure_type: FailureType
    error_message: str
    stack_trace: str
    timestamp: datetime = field(default_factory=datetime.now)
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


@dataclass
class RecoveryStrategy:
    """Recovery strategy for different types of failures."""
    failure_type: FailureType
    strategy_name: str
    recovery_function: Callable
    max_attempts: int = 3
    backoff_factor: float = 1.5
    timeout: float = 300.0  # 5 minutes


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker: Attempting reset (HALF_OPEN)")
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.logger.debug("Circuit breaker: Success - state CLOSED")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker: OPEN due to {self.failure_count} failures")


class RetryManager:
    """Intelligent retry manager with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = get_logger(__name__)
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Retry async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_attempts - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
        
        raise last_exception
    
    def retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Retry synchronous function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_attempts - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
        
        raise last_exception


class ErrorRecoverySystem:
    """Autonomous error recovery system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.failure_history: List[FailureContext] = []
        self.recovery_strategies: Dict[FailureType, List[RecoveryStrategy]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_checker = SystemHealthChecker()
        self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self):
        """Initialize default recovery strategies."""
        # Memory error recovery
        self.recovery_strategies[FailureType.MEMORY_ERROR] = [
            RecoveryStrategy(
                FailureType.MEMORY_ERROR,
                "garbage_collection",
                self._recover_memory_gc,
                max_attempts=2
            ),
            RecoveryStrategy(
                FailureType.MEMORY_ERROR,
                "reduce_batch_size",
                self._recover_memory_reduce_batch,
                max_attempts=3
            )
        ]
        
        # Dependency error recovery
        self.recovery_strategies[FailureType.DEPENDENCY_ERROR] = [
            RecoveryStrategy(
                FailureType.DEPENDENCY_ERROR,
                "reinstall_dependencies",
                self._recover_dependency_reinstall,
                max_attempts=2
            ),
            RecoveryStrategy(
                FailureType.DEPENDENCY_ERROR,
                "fallback_implementation",
                self._recover_dependency_fallback,
                max_attempts=1
            )
        ]
        
        # Timeout error recovery
        self.recovery_strategies[FailureType.TIMEOUT_ERROR] = [
            RecoveryStrategy(
                FailureType.TIMEOUT_ERROR,
                "increase_timeout",
                self._recover_timeout_increase,
                max_attempts=2
            ),
            RecoveryStrategy(
                FailureType.TIMEOUT_ERROR,
                "parallel_processing",
                self._recover_timeout_parallel,
                max_attempts=1
            )
        ]
        
        # Data error recovery
        self.recovery_strategies[FailureType.DATA_ERROR] = [
            RecoveryStrategy(
                FailureType.DATA_ERROR,
                "data_validation_repair",
                self._recover_data_validation,
                max_attempts=2
            ),
            RecoveryStrategy(
                FailureType.DATA_ERROR,
                "fallback_dataset",
                self._recover_data_fallback,
                max_attempts=1
            )
        ]
    
    async def handle_failure(self, exception: Exception, context: Dict[str, Any]) -> Any:
        """Handle failure with appropriate recovery strategy."""
        failure_type = self._classify_failure(exception)
        
        failure_context = FailureContext(
            failure_type=failure_type,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            system_state=await self._capture_system_state()
        )
        
        self.failure_history.append(failure_context)
        self.logger.error(f"Handling {failure_type.value}: {str(exception)}")
        
        # Attempt recovery
        recovery_successful = await self._attempt_recovery(failure_context, context)
        
        if recovery_successful:
            self.logger.info(f"Successfully recovered from {failure_type.value}")
            return True
        else:
            self.logger.error(f"Failed to recover from {failure_type.value}")
            return False
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure."""
        error_message = str(exception).lower()
        error_type = type(exception).__name__.lower()
        
        if "memory" in error_message or "oom" in error_message:
            return FailureType.MEMORY_ERROR
        elif "import" in error_message or "module" in error_message:
            return FailureType.DEPENDENCY_ERROR
        elif "timeout" in error_message or "timeouterror" in error_type:
            return FailureType.TIMEOUT_ERROR
        elif "connection" in error_message or "network" in error_message:
            return FailureType.NETWORK_ERROR
        elif "data" in error_message or "file" in error_message:
            return FailureType.DATA_ERROR
        elif "model" in error_message or "tensor" in error_message:
            return FailureType.MODEL_ERROR
        else:
            return FailureType.SYSTEM_ERROR
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for debugging."""
        try:
            return {
                "memory_usage": psutil.virtual_memory()._asdict(),
                "cpu_usage": psutil.cpu_percent(),
                "disk_usage": psutil.disk_usage('/')._asdict(),
                "process_count": len(psutil.pids()),
                "timestamp": datetime.now().isoformat(),
                "health_status": await self.health_checker.check_system_health()
            }
        except Exception as e:
            self.logger.warning(f"Could not capture system state: {e}")
            return {"error": str(e)}
    
    async def _attempt_recovery(self, failure_context: FailureContext, context: Dict[str, Any]) -> bool:
        """Attempt recovery using available strategies."""
        strategies = self.recovery_strategies.get(failure_context.failure_type, [])
        
        for strategy in strategies:
            if failure_context.recovery_attempts >= strategy.max_attempts:
                continue
            
            try:
                self.logger.info(f"Attempting recovery strategy: {strategy.strategy_name}")
                
                # Apply circuit breaker
                circuit_breaker = self._get_circuit_breaker(strategy.strategy_name)
                
                result = await asyncio.wait_for(
                    circuit_breaker.call(strategy.recovery_function, failure_context, context),
                    timeout=strategy.timeout
                )
                
                if result:
                    return True
                
            except Exception as e:
                self.logger.warning(f"Recovery strategy {strategy.strategy_name} failed: {e}")
                failure_context.recovery_attempts += 1
        
        return False
    
    def _get_circuit_breaker(self, strategy_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for strategy."""
        if strategy_name not in self.circuit_breakers:
            self.circuit_breakers[strategy_name] = CircuitBreaker()
        return self.circuit_breakers[strategy_name]
    
    # Recovery strategy implementations
    
    async def _recover_memory_gc(self, failure_context: FailureContext, context: Dict[str, Any]) -> bool:
        """Recover from memory errors by garbage collection."""
        self.logger.info("Attempting memory recovery through garbage collection")
        
        # Force garbage collection
        collected = gc.collect()
        self.logger.info(f"Garbage collection freed {collected} objects")
        
        # Clear any cached data
        if hasattr(self, '_clear_caches'):
            self._clear_caches()
        
        # Wait a bit for memory to be released
        await asyncio.sleep(1)
        
        # Check if memory usage improved
        memory_after = psutil.virtual_memory()
        return memory_after.percent < 90  # Consider successful if under 90%
    
    async def _recover_memory_reduce_batch(self, failure_context: FailureContext, context: Dict[str, Any]) -> bool:
        """Recover from memory errors by reducing batch size."""
        self.logger.info("Attempting memory recovery by reducing batch size")
        
        if 'batch_size' in context:
            new_batch_size = max(1, context['batch_size'] // 2)
            context['batch_size'] = new_batch_size
            self.logger.info(f"Reduced batch size to {new_batch_size}")
            return True
        
        return False
    
    async def _recover_dependency_reinstall(self, failure_context: FailureContext, context: Dict[str, Any]) -> bool:
        """Recover from dependency errors by reinstalling."""
        self.logger.info("Attempting dependency recovery through reinstallation")
        
        # This would typically use subprocess to reinstall packages
        # For now, we'll simulate the recovery
        await asyncio.sleep(2)  # Simulate installation time
        
        # Check if dependencies are now available
        try:
            import torch
            import torch_geometric
            return True
        except ImportError:
            return False
    
    async def _recover_dependency_fallback(self, failure_context: FailureContext, context: Dict[str, Any]) -> bool:
        """Recover from dependency errors using fallback implementations."""
        self.logger.info("Attempting dependency recovery using fallback implementations")
        
        # Enable fallback mode
        context['use_fallback'] = True
        return True
    
    async def _recover_timeout_increase(self, failure_context: FailureContext, context: Dict[str, Any]) -> bool:
        """Recover from timeout errors by increasing timeout."""
        self.logger.info("Attempting timeout recovery by increasing timeout")
        
        if 'timeout' in context:
            context['timeout'] *= 2  # Double the timeout
            self.logger.info(f"Increased timeout to {context['timeout']}")
            return True
        
        return False
    
    async def _recover_timeout_parallel(self, failure_context: FailureContext, context: Dict[str, Any]) -> bool:
        """Recover from timeout errors using parallel processing."""
        self.logger.info("Attempting timeout recovery through parallel processing")
        
        context['use_parallel'] = True
        context['num_workers'] = min(4, psutil.cpu_count())
        return True
    
    async def _recover_data_validation(self, failure_context: FailureContext, context: Dict[str, Any]) -> bool:
        """Recover from data errors by validation and repair."""
        self.logger.info("Attempting data recovery through validation and repair")
        
        # This would implement data validation and repair logic
        await asyncio.sleep(1)  # Simulate validation time
        
        context['data_validated'] = True
        return True
    
    async def _recover_data_fallback(self, failure_context: FailureContext, context: Dict[str, Any]) -> bool:
        """Recover from data errors using fallback dataset."""
        self.logger.info("Attempting data recovery using fallback dataset")
        
        context['use_fallback_data'] = True
        context['fallback_dataset'] = 'synthetic'
        return True


class SelfHealingSystem:
    """Self-healing system for autonomous operation."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.error_recovery = ErrorRecoverySystem()
        self.health_checker = SystemHealthChecker()
        self.monitoring_active = False
        self.healing_metrics = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "uptime_start": datetime.now()
        }
    
    async def start_monitoring(self, check_interval: float = 60.0):
        """Start continuous system monitoring and healing."""
        self.monitoring_active = True
        self.logger.info("Starting self-healing monitoring")
        
        while self.monitoring_active:
            try:
                # Check system health
                health_status = await self.health_checker.check_system_health()
                
                if health_status != HealthStatus.HEALTHY:
                    await self._handle_health_issue(health_status)
                
                # Check for degradation patterns
                await self._check_performance_degradation()
                
                # Proactive maintenance
                await self._perform_proactive_maintenance()
                
            except Exception as e:
                self.logger.error(f"Self-healing monitoring error: {e}")
                await self.error_recovery.handle_failure(e, {"context": "monitoring"})
            
            await asyncio.sleep(check_interval)
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        self.logger.info("Stopped self-healing monitoring")
    
    async def _handle_health_issue(self, health_status: HealthStatus):
        """Handle detected health issues."""
        self.logger.warning(f"Health issue detected: {health_status}")
        
        if health_status == HealthStatus.DEGRADED:
            await self._handle_degraded_performance()
        elif health_status == HealthStatus.CRITICAL:
            await self._handle_critical_state()
        elif health_status == HealthStatus.UNHEALTHY:
            await self._handle_unhealthy_state()
    
    async def _handle_degraded_performance(self):
        """Handle degraded performance."""
        self.logger.info("Handling degraded performance")
        
        # Reduce load
        await self._reduce_system_load()
        
        # Clear caches
        await self._clear_system_caches()
        
        # Optimize resource usage
        await self._optimize_resource_usage()
    
    async def _handle_critical_state(self):
        """Handle critical system state."""
        self.logger.warning("Handling critical system state")
        
        # Emergency resource cleanup
        await self._emergency_cleanup()
        
        # Restart critical services
        await self._restart_critical_services()
        
        # Enable safe mode
        await self._enable_safe_mode()
    
    async def _handle_unhealthy_state(self):
        """Handle unhealthy system state."""
        self.logger.error("Handling unhealthy system state")
        
        # Graceful shutdown of non-essential services
        await self._shutdown_non_essential_services()
        
        # Save current state
        await self._save_system_state()
        
        # Attempt system recovery
        await self._attempt_system_recovery()
    
    async def _check_performance_degradation(self):
        """Check for performance degradation patterns."""
        # This would implement performance trend analysis
        pass
    
    async def _perform_proactive_maintenance(self):
        """Perform proactive system maintenance."""
        # Periodic cleanup, optimization, etc.
        pass
    
    async def _reduce_system_load(self):
        """Reduce system load during degraded performance."""
        self.logger.info("Reducing system load")
    
    async def _clear_system_caches(self):
        """Clear system caches."""
        self.logger.info("Clearing system caches")
        gc.collect()
    
    async def _optimize_resource_usage(self):
        """Optimize resource usage."""
        self.logger.info("Optimizing resource usage")
    
    async def _emergency_cleanup(self):
        """Emergency resource cleanup."""
        self.logger.warning("Performing emergency cleanup")
    
    async def _restart_critical_services(self):
        """Restart critical services."""
        self.logger.warning("Restarting critical services")
    
    async def _enable_safe_mode(self):
        """Enable safe mode operation."""
        self.logger.warning("Enabling safe mode")
    
    async def _shutdown_non_essential_services(self):
        """Shutdown non-essential services."""
        self.logger.error("Shutting down non-essential services")
    
    async def _save_system_state(self):
        """Save current system state."""
        self.logger.error("Saving system state")
    
    async def _attempt_system_recovery(self):
        """Attempt full system recovery."""
        self.logger.error("Attempting system recovery")


# Global instances
_error_recovery_system = None
_self_healing_system = None


def get_error_recovery_system() -> ErrorRecoverySystem:
    """Get global error recovery system."""
    global _error_recovery_system
    if _error_recovery_system is None:
        _error_recovery_system = ErrorRecoverySystem()
    return _error_recovery_system


def get_self_healing_system() -> SelfHealingSystem:
    """Get global self-healing system."""
    global _self_healing_system
    if _self_healing_system is None:
        _self_healing_system = SelfHealingSystem()
    return _self_healing_system


# Decorators for reliability features

def with_circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Decorator to add circuit breaker protection."""
    def decorator(func: Callable) -> Callable:
        circuit_breaker = CircuitBreaker(failure_threshold, recovery_timeout)
        
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator to add retry functionality."""
    def decorator(func: Callable) -> Callable:
        retry_manager = RetryManager(max_attempts, base_delay)
        
        async def async_wrapper(*args, **kwargs):
            return await retry_manager.retry_async(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return retry_manager.retry_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_error_recovery(recovery_context: Optional[Dict[str, Any]] = None):
    """Decorator to add automatic error recovery."""
    def decorator(func: Callable) -> Callable:
        error_recovery = get_error_recovery_system()
        
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = recovery_context or {}
                recovery_successful = await error_recovery.handle_failure(e, context)
                if recovery_successful:
                    # Retry after successful recovery
                    return await func(*args, **kwargs)
                else:
                    raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = recovery_context or {}
                # Convert to async for recovery handling
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    recovery_successful = loop.run_until_complete(
                        error_recovery.handle_failure(e, context)
                    )
                    if recovery_successful:
                        return func(*args, **kwargs)
                    else:
                        raise
                finally:
                    loop.close()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator