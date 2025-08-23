"""Progressive Resilience Framework - TERRAGON SDLC v6.0 Enhancement.

Advanced self-healing and adaptive resilience system that evolves with system
maturity and learns from failure patterns to prevent future incidents.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque

try:
    from .progressive_quality_gates import ProgressiveLevel
    from .robust_error_handling import ErrorSeverity, ErrorCategory, RobustErrorHandler
    from .advanced_monitoring import PerformanceMetrics, SystemMonitor
except ImportError:
    from enum import Enum
    
    class ProgressiveLevel(Enum):
        BASIC = "basic"
        INTERMEDIATE = "intermediate"
        ADVANCED = "advanced"
        EXPERT = "expert"
        AUTONOMOUS = "autonomous"
    
    class ErrorSeverity(Enum):
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        INFO = "info"
    
    class ErrorCategory(Enum):
        SYSTEM = "system"
        NETWORK = "network"
        DATA = "data"
        SECURITY = "security"
        PERFORMANCE = "performance"
        VALIDATION = "validation"
        IO = "io"
        COMPUTATION = "computation"
        DEPENDENCY = "dependency"
        CONFIGURATION = "configuration"
        UNKNOWN = "unknown"


class ResilienceStrategy(Enum):
    """Resilience strategy types."""
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_BACKOFF = "retry_backoff"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    RATE_LIMITING = "rate_limiting"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SELF_HEALING = "self_healing"
    PREDICTIVE_RECOVERY = "predictive_recovery"


class FailurePattern(Enum):
    """Common failure patterns."""
    CASCADING_FAILURE = "cascading_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    THUNDERING_HERD = "thundering_herd"
    SPLIT_BRAIN = "split_brain"
    BYZANTINE_FAILURE = "byzantine_failure"
    TEMPORAL_COUPLING = "temporal_coupling"


@dataclass
class ResilienceEvent:
    """Resilience event tracking."""
    timestamp: datetime
    event_type: str
    severity: ErrorSeverity
    category: ErrorCategory
    description: str
    strategy_applied: Optional[ResilienceStrategy] = None
    recovery_time: Optional[float] = None
    success: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'severity': self.severity.value,
            'category': self.category.value,
            'description': self.description,
            'strategy_applied': self.strategy_applied.value if self.strategy_applied else None,
            'recovery_time': self.recovery_time,
            'success': self.success,
            'context': self.context
        }


@dataclass
class ProgressiveResilienceConfig:
    """Progressive resilience configuration."""
    level: ProgressiveLevel
    enabled_strategies: List[ResilienceStrategy] = field(default_factory=list)
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    circuit_breaker_timeout: float = 60.0
    max_retry_attempts: int = 3
    backoff_multiplier: float = 2.0
    rate_limit_rps: float = 100.0
    bulkhead_concurrency: int = 10
    learning_rate: float = 0.1
    prediction_horizon: int = 300  # 5 minutes
    
    def get_strategy_config(self, strategy: ResilienceStrategy) -> Dict[str, Any]:
        """Get configuration for specific strategy."""
        configs = {
            ResilienceStrategy.CIRCUIT_BREAKER: {
                'failure_threshold': self.failure_threshold,
                'timeout': self.circuit_breaker_timeout
            },
            ResilienceStrategy.RETRY_BACKOFF: {
                'max_attempts': self.max_retry_attempts,
                'backoff_multiplier': self.backoff_multiplier
            },
            ResilienceStrategy.RATE_LIMITING: {
                'requests_per_second': self.rate_limit_rps
            },
            ResilienceStrategy.BULKHEAD: {
                'max_concurrency': self.bulkhead_concurrency
            },
            ResilienceStrategy.TIMEOUT: {
                'timeout': self.recovery_timeout
            }
        }
        return configs.get(strategy, {})


class ProgressiveCircuitBreaker:
    """Circuit breaker that adapts based on failure patterns."""
    
    def __init__(self, name: str, config: ProgressiveResilienceConfig):
        self.name = name
        self.config = config
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.failure_history = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                else:
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure(e)
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit breaker."""
        if not self.last_failure_time:
            return True
        
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.circuit_breaker_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= 3:  # Progressive threshold
                self.state = "CLOSED"
                self.failure_count = 0
                self.last_failure_time = None
        elif self.state == "CLOSED":
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, exception: Exception):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.failure_history.append({
            'timestamp': self.last_failure_time,
            'error': str(exception)
        })
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = "OPEN"


class ProgressiveRetryHandler:
    """Retry handler with adaptive backoff strategies."""
    
    def __init__(self, config: ProgressiveResilienceConfig):
        self.config = config
        self.attempt_history = defaultdict(list)
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Async retry with progressive backoff."""
        last_exception = None
        
        for attempt in range(self.config.max_retry_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retry_attempts:
                    delay = self._calculate_backoff(attempt)
                    await asyncio.sleep(delay)
                    
                    # Record attempt for learning
                    self.attempt_history[func.__name__].append({
                        'attempt': attempt,
                        'error': str(e),
                        'delay': delay,
                        'timestamp': datetime.utcnow()
                    })
        
        raise last_exception
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate progressive backoff delay."""
        base_delay = 1.0
        return base_delay * (self.config.backoff_multiplier ** attempt)


class ProgressiveRateLimiter:
    """Rate limiter that adapts based on system load."""
    
    def __init__(self, config: ProgressiveResilienceConfig):
        self.config = config
        self.requests = deque()
        self.lock = threading.Lock()
        self.current_limit = config.rate_limit_rps
        self.load_history = deque(maxlen=100)
    
    def acquire(self) -> bool:
        """Acquire permission to proceed."""
        with self.lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and now - self.requests[0] > 1.0:
                self.requests.popleft()
            
            # Check current rate
            current_rate = len(self.requests)
            self.load_history.append(current_rate / self.current_limit)
            
            # Adapt limit based on load history
            if len(self.load_history) >= 10:
                avg_load = sum(self.load_history[-10:]) / 10
                if avg_load < 0.5:  # Underutilized
                    self.current_limit = min(
                        self.current_limit * 1.1,
                        self.config.rate_limit_rps * 2
                    )
                elif avg_load > 0.9:  # Overloaded
                    self.current_limit = max(
                        self.current_limit * 0.9,
                        self.config.rate_limit_rps * 0.5
                    )
            
            if current_rate < self.current_limit:
                self.requests.append(now)
                return True
            
            return False


class FailurePredictor:
    """Predicts potential failures based on patterns."""
    
    def __init__(self, config: ProgressiveResilienceConfig):
        self.config = config
        self.failure_patterns = defaultdict(list)
        self.system_metrics = deque(maxlen=1000)
    
    def record_metrics(self, metrics: Dict[str, float]):
        """Record system metrics for analysis."""
        timestamp = time.time()
        self.system_metrics.append({
            'timestamp': timestamp,
            'metrics': metrics.copy()
        })
    
    def predict_failure_probability(self, 
                                  horizon_seconds: int = 300) -> Dict[str, float]:
        """Predict failure probability for different categories."""
        if len(self.system_metrics) < 10:
            return {}
        
        # Simple trend-based prediction
        recent_metrics = list(self.system_metrics)[-10:]
        predictions = {}
        
        for category in ErrorCategory:
            # Analyze trends for each category
            trend_score = self._analyze_trend(recent_metrics, category.value)
            predictions[category.value] = min(1.0, max(0.0, trend_score))
        
        return predictions
    
    def _analyze_trend(self, metrics_history: List[Dict], 
                      category: str) -> float:
        """Analyze trend for specific category."""
        # Simple heuristic-based analysis
        if category == "performance":
            cpu_values = [m['metrics'].get('cpu_percent', 0) for m in metrics_history]
            if len(cpu_values) > 1:
                trend = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
                return trend / 100.0  # Normalize
        
        elif category == "network":
            error_rates = [m['metrics'].get('network_errors', 0) for m in metrics_history]
            if error_rates:
                avg_error_rate = sum(error_rates) / len(error_rates)
                return min(avg_error_rate / 10.0, 1.0)
        
        return 0.0


class ProgressiveResilienceOrchestrator:
    """Main orchestrator for progressive resilience system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path("progressive_resilience_config.json")
        self.level = ProgressiveLevel.BASIC
        self.config = ProgressiveResilienceConfig(level=self.level)
        self.circuit_breakers: Dict[str, ProgressiveCircuitBreaker] = {}
        self.retry_handler = ProgressiveRetryHandler(self.config)
        self.rate_limiter = ProgressiveRateLimiter(self.config)
        self.failure_predictor = FailurePredictor(self.config)
        self.events_history: List[ResilienceEvent] = []
        self.active_strategies: Dict[str, bool] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                self.level = ProgressiveLevel(config_data.get('level', 'basic'))
                self.config.level = self.level
                
                # Load enabled strategies
                strategies = config_data.get('enabled_strategies', [])
                self.config.enabled_strategies = [
                    ResilienceStrategy(s) for s in strategies
                ]
                
                # Load other config values
                for key, value in config_data.get('config', {}).items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}, using defaults")
        
        # Initialize default strategies based on level
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default strategies based on current level."""
        level_strategies = {
            ProgressiveLevel.BASIC: [
                ResilienceStrategy.RETRY_BACKOFF,
                ResilienceStrategy.TIMEOUT
            ],
            ProgressiveLevel.INTERMEDIATE: [
                ResilienceStrategy.RETRY_BACKOFF,
                ResilienceStrategy.TIMEOUT,
                ResilienceStrategy.CIRCUIT_BREAKER,
                ResilienceStrategy.RATE_LIMITING
            ],
            ProgressiveLevel.ADVANCED: [
                ResilienceStrategy.RETRY_BACKOFF,
                ResilienceStrategy.TIMEOUT,
                ResilienceStrategy.CIRCUIT_BREAKER,
                ResilienceStrategy.RATE_LIMITING,
                ResilienceStrategy.BULKHEAD,
                ResilienceStrategy.GRACEFUL_DEGRADATION
            ],
            ProgressiveLevel.EXPERT: [
                ResilienceStrategy.RETRY_BACKOFF,
                ResilienceStrategy.TIMEOUT,
                ResilienceStrategy.CIRCUIT_BREAKER,
                ResilienceStrategy.RATE_LIMITING,
                ResilienceStrategy.BULKHEAD,
                ResilienceStrategy.GRACEFUL_DEGRADATION,
                ResilienceStrategy.SELF_HEALING
            ],
            ProgressiveLevel.AUTONOMOUS: [
                ResilienceStrategy.RETRY_BACKOFF,
                ResilienceStrategy.TIMEOUT,
                ResilienceStrategy.CIRCUIT_BREAKER,
                ResilienceStrategy.RATE_LIMITING,
                ResilienceStrategy.BULKHEAD,
                ResilienceStrategy.GRACEFUL_DEGRADATION,
                ResilienceStrategy.SELF_HEALING,
                ResilienceStrategy.PREDICTIVE_RECOVERY
            ]
        }
        
        if not self.config.enabled_strategies:
            self.config.enabled_strategies = level_strategies.get(
                self.level, 
                level_strategies[ProgressiveLevel.BASIC]
            )
    
    def get_circuit_breaker(self, name: str) -> ProgressiveCircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = ProgressiveCircuitBreaker(name, self.config)
        return self.circuit_breakers[name]
    
    async def execute_with_resilience(self, 
                                    name: str,
                                    func: Callable,
                                    *args, **kwargs) -> Any:
        """Execute function with progressive resilience protection."""
        start_time = time.time()
        strategies_applied = []
        
        try:
            # Apply rate limiting
            if ResilienceStrategy.RATE_LIMITING in self.config.enabled_strategies:
                if not self.rate_limiter.acquire():
                    raise Exception("Rate limit exceeded")
                strategies_applied.append(ResilienceStrategy.RATE_LIMITING)
            
            # Apply circuit breaker
            if ResilienceStrategy.CIRCUIT_BREAKER in self.config.enabled_strategies:
                circuit_breaker = self.get_circuit_breaker(name)
                
                if ResilienceStrategy.RETRY_BACKOFF in self.config.enabled_strategies:
                    # Combine circuit breaker with retry
                    result = await self.retry_handler.retry_async(
                        circuit_breaker.call, func, *args, **kwargs
                    )
                else:
                    result = circuit_breaker.call(func, *args, **kwargs)
                
                strategies_applied.extend([
                    ResilienceStrategy.CIRCUIT_BREAKER,
                    ResilienceStrategy.RETRY_BACKOFF
                ])
            
            elif ResilienceStrategy.RETRY_BACKOFF in self.config.enabled_strategies:
                # Retry only
                result = await self.retry_handler.retry_async(func, *args, **kwargs)
                strategies_applied.append(ResilienceStrategy.RETRY_BACKOFF)
            
            else:
                # Direct execution
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            
            # Record success event
            recovery_time = time.time() - start_time
            self._record_event(
                event_type="execution_success",
                severity=ErrorSeverity.INFO,
                category=ErrorCategory.SYSTEM,
                description=f"Successfully executed {name}",
                strategies_applied=strategies_applied,
                recovery_time=recovery_time,
                success=True
            )
            
            return result
            
        except Exception as e:
            # Record failure event
            recovery_time = time.time() - start_time
            self._record_event(
                event_type="execution_failure",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SYSTEM,
                description=f"Failed to execute {name}: {str(e)}",
                strategies_applied=strategies_applied,
                recovery_time=recovery_time,
                success=False
            )
            
            # Try self-healing if enabled
            if ResilienceStrategy.SELF_HEALING in self.config.enabled_strategies:
                await self._attempt_self_healing(name, e)
            
            raise
    
    async def _attempt_self_healing(self, name: str, exception: Exception):
        """Attempt self-healing recovery."""
        self.logger.info(f"Attempting self-healing for {name}")
        
        # Simple self-healing strategies
        healing_strategies = [
            self._clear_caches,
            self._restart_connections,
            self._reduce_load
        ]
        
        for strategy in healing_strategies:
            try:
                await strategy(name, exception)
                self.logger.info(f"Self-healing strategy {strategy.__name__} completed")
                break
            except Exception as healing_error:
                self.logger.warning(
                    f"Self-healing strategy {strategy.__name__} failed: {healing_error}"
                )
    
    async def _clear_caches(self, name: str, exception: Exception):
        """Clear caches for recovery."""
        # Mock implementation
        await asyncio.sleep(0.1)
    
    async def _restart_connections(self, name: str, exception: Exception):
        """Restart connections for recovery."""
        # Mock implementation
        await asyncio.sleep(0.1)
    
    async def _reduce_load(self, name: str, exception: Exception):
        """Reduce system load for recovery."""
        # Temporarily reduce rate limits
        self.rate_limiter.current_limit *= 0.8
        await asyncio.sleep(0.1)
    
    def _record_event(self, 
                     event_type: str,
                     severity: ErrorSeverity,
                     category: ErrorCategory,
                     description: str,
                     strategies_applied: List[ResilienceStrategy] = None,
                     recovery_time: Optional[float] = None,
                     success: bool = False,
                     context: Dict[str, Any] = None):
        """Record resilience event."""
        event = ResilienceEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            category=category,
            description=description,
            strategy_applied=strategies_applied[0] if strategies_applied else None,
            recovery_time=recovery_time,
            success=success,
            context=context or {}
        )
        
        self.events_history.append(event)
        
        # Keep last 1000 events
        if len(self.events_history) > 1000:
            self.events_history = self.events_history[-1000:]
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get current resilience system status."""
        if not self.events_history:
            return {
                'level': self.level.value,
                'enabled_strategies': [s.value for s in self.config.enabled_strategies],
                'total_events': 0,
                'recent_success_rate': 0.0,
                'recommendations': ['No events recorded yet']
            }
        
        recent_events = [
            e for e in self.events_history[-100:]
            if e.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        success_rate = (
            len([e for e in recent_events if e.success]) / 
            len(recent_events) * 100 if recent_events else 0
        )
        
        # Get failure predictions
        predictions = self.failure_predictor.predict_failure_probability()
        
        status = {
            'level': self.level.value,
            'enabled_strategies': [s.value for s in self.config.enabled_strategies],
            'total_events': len(self.events_history),
            'recent_events': len(recent_events),
            'recent_success_rate': success_rate,
            'circuit_breakers': {
                name: {
                    'state': cb.state,
                    'failure_count': cb.failure_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            'rate_limiter': {
                'current_limit': self.rate_limiter.current_limit,
                'configured_limit': self.config.rate_limit_rps
            },
            'failure_predictions': predictions,
            'recommendations': self._generate_recommendations(recent_events, predictions)
        }
        
        return status
    
    def _generate_recommendations(self, 
                                recent_events: List[ResilienceEvent],
                                predictions: Dict[str, float]) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []
        
        if not recent_events:
            return ["No recent events to analyze"]
        
        # Analyze failure patterns
        failure_events = [e for e in recent_events if not e.success]
        failure_rate = len(failure_events) / len(recent_events)
        
        if failure_rate > 0.1:  # >10% failure rate
            recommendations.append(
                f"High failure rate ({failure_rate*100:.1f}%) - consider enabling additional resilience strategies"
            )
        
        # Check circuit breaker states
        open_breakers = [
            name for name, cb in self.circuit_breakers.items()
            if cb.state == "OPEN"
        ]
        
        if open_breakers:
            recommendations.append(
                f"Circuit breakers open: {', '.join(open_breakers)} - investigate underlying issues"
            )
        
        # Check predictions
        high_risk_categories = [
            cat for cat, prob in predictions.items()
            if prob > 0.7
        ]
        
        if high_risk_categories:
            recommendations.append(
                f"High failure risk predicted for: {', '.join(high_risk_categories)}"
            )
        
        return recommendations or ["System operating normally"]
    
    def save_config(self):
        """Save current configuration."""
        config_data = {
            'level': self.level.value,
            'enabled_strategies': [s.value for s in self.config.enabled_strategies],
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'circuit_breaker_timeout': self.config.circuit_breaker_timeout,
                'max_retry_attempts': self.config.max_retry_attempts,
                'backoff_multiplier': self.config.backoff_multiplier,
                'rate_limit_rps': self.config.rate_limit_rps,
                'learning_rate': self.config.learning_rate
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)


# Convenience functions
_global_orchestrator = None

def get_progressive_resilience() -> ProgressiveResilienceOrchestrator:
    """Get global progressive resilience orchestrator."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = ProgressiveResilienceOrchestrator()
    return _global_orchestrator


async def resilient_execution(name: str, func: Callable, *args, **kwargs) -> Any:
    """Execute function with progressive resilience protection."""
    orchestrator = get_progressive_resilience()
    return await orchestrator.execute_with_resilience(name, func, *args, **kwargs)


def resilient(name: str):
    """Decorator for resilient function execution."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            return await resilient_execution(name, func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create async context
            async def async_func():
                return func(*args, **kwargs)
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in event loop, just call function directly
                    return func(*args, **kwargs)
                else:
                    return loop.run_until_complete(
                        resilient_execution(name, async_func)
                    )
            except RuntimeError:
                # No event loop, create new one
                return asyncio.run(resilient_execution(name, async_func))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator