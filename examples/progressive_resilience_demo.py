#!/usr/bin/env python3
"""Progressive Resilience Demo - TERRAGON SDLC v6.0 Enhancement.

This demo showcases the progressive resilience system with adaptive self-healing,
circuit breakers, and failure prediction capabilities.
"""

import asyncio
import random
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scgraph_hub.progressive_resilience import (
    ProgressiveResilienceOrchestrator,
    ProgressiveLevel,
    ResilienceStrategy,
    resilient_execution,
    resilient
)


class MockService:
    """Mock service for demonstrating resilience patterns."""
    
    def __init__(self, failure_rate: float = 0.3):
        self.failure_rate = failure_rate
        self.call_count = 0
        self.consecutive_failures = 0
    
    async def unstable_operation(self, data: str = "test") -> str:
        """Operation that fails randomly."""
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Simulate failures
        if random.random() < self.failure_rate:
            self.consecutive_failures += 1
            failure_types = [
                "Network timeout",
                "Database connection lost", 
                "Memory allocation failed",
                "Service unavailable",
                "Rate limit exceeded"
            ]
            raise Exception(f"{random.choice(failure_types)} (attempt {self.call_count})")
        
        self.consecutive_failures = 0
        return f"Success: processed '{data}' (attempt {self.call_count})"
    
    def sync_operation(self, data: str = "sync_test") -> str:
        """Synchronous operation for testing."""
        self.call_count += 1
        
        if random.random() < self.failure_rate:
            raise Exception(f"Sync operation failed (attempt {self.call_count})")
        
        return f"Sync success: {data} (attempt {self.call_count})"


async def demonstrate_basic_resilience():
    """Demonstrate basic resilience features."""
    print("🛡️ Basic Resilience Demonstration")
    print("-" * 40)
    
    orchestrator = ProgressiveResilienceOrchestrator()
    service = MockService(failure_rate=0.4)
    
    print(f"Initial resilience level: {orchestrator.level.value}")
    print(f"Enabled strategies: {[s.value for s in orchestrator.config.enabled_strategies]}")
    
    # Test resilient execution
    print(f"\n🔄 Testing resilient execution (high failure rate)...")
    
    for i in range(5):
        try:
            print(f"\n  Test {i+1}:")
            result = await resilient_execution(
                "unstable_service",
                service.unstable_operation,
                f"data_{i}"
            )
            print(f"    ✅ {result}")
            
        except Exception as e:
            print(f"    ❌ Final failure: {str(e)}")
        
        # Show circuit breaker status
        cb = orchestrator.get_circuit_breaker("unstable_service")
        print(f"    Circuit breaker: {cb.state} (failures: {cb.failure_count})")
    
    return orchestrator


async def demonstrate_adaptive_strategies():
    """Demonstrate adaptive strategy evolution."""
    print(f"\n🧠 Adaptive Strategy Evolution")
    print("-" * 40)
    
    # Start with basic level
    orchestrator = ProgressiveResilienceOrchestrator()
    orchestrator.level = ProgressiveLevel.BASIC
    orchestrator._initialize_default_strategies()
    
    service = MockService(failure_rate=0.2)  # Lower failure rate for progression
    
    # Simulate progression through levels
    levels = list(ProgressiveLevel)
    
    for level in levels:
        orchestrator.level = level
        orchestrator._initialize_default_strategies()
        
        print(f"\n📊 Testing {level.value.upper()} level:")
        print(f"  Enabled strategies: {[s.value for s in orchestrator.config.enabled_strategies]}")
        
        # Run several operations
        successes = 0
        for i in range(3):
            try:
                result = await resilient_execution(
                    f"service_level_{level.value}",
                    service.unstable_operation,
                    f"{level.value}_data_{i}"
                )
                successes += 1
                print(f"    ✅ Success {i+1}: Operation completed")
                
            except Exception as e:
                print(f"    ❌ Failure {i+1}: {str(e)[:50]}...")
        
        print(f"  Success rate: {successes}/3 ({successes/3*100:.1f}%)")
        
        # Show resilience status
        status = orchestrator.get_resilience_status()
        if status['recommendations']:
            print(f"  💡 Recommendations: {status['recommendations'][0][:60]}...")


@resilient("decorated_service")
async def decorated_async_function(data: str) -> str:
    """Async function with resilience decorator."""
    await asyncio.sleep(0.1)
    if random.random() < 0.3:
        raise Exception(f"Decorated async function failed for {data}")
    return f"Decorated async success: {data}"


@resilient("decorated_sync_service") 
def decorated_sync_function(data: str) -> str:
    """Sync function with resilience decorator."""
    time.sleep(0.1)
    if random.random() < 0.3:
        raise Exception(f"Decorated sync function failed for {data}")
    return f"Decorated sync success: {data}"


async def demonstrate_decorators():
    """Demonstrate resilience decorators."""
    print(f"\n🎨 Resilience Decorators")
    print("-" * 40)
    
    print("Testing async decorated function:")
    for i in range(3):
        try:
            result = await decorated_async_function(f"async_data_{i}")
            print(f"  ✅ {result}")
        except Exception as e:
            print(f"  ❌ {str(e)}")
    
    print(f"\nTesting sync decorated function:")
    for i in range(3):
        try:
            result = decorated_sync_function(f"sync_data_{i}")
            print(f"  ✅ {result}")
        except Exception as e:
            print(f"  ❌ {str(e)}")


async def demonstrate_failure_prediction():
    """Demonstrate failure prediction capabilities."""
    print(f"\n🔮 Failure Prediction")
    print("-" * 40)
    
    orchestrator = ProgressiveResilienceOrchestrator()
    orchestrator.level = ProgressiveLevel.AUTONOMOUS
    orchestrator._initialize_default_strategies()
    
    # Feed metrics to the predictor
    print("Feeding system metrics for analysis...")
    for i in range(20):
        # Simulate degrading metrics
        metrics = {
            'cpu_percent': 20 + i * 2,  # Increasing CPU
            'memory_percent': 30 + i * 1.5,  # Increasing memory
            'network_errors': i * 0.1,  # Increasing errors
            'response_time': 100 + i * 10  # Increasing latency
        }
        
        orchestrator.failure_predictor.record_metrics(metrics)
        await asyncio.sleep(0.01)  # Simulate time passage
    
    # Get predictions
    predictions = orchestrator.failure_predictor.predict_failure_probability()
    
    print("Failure probability predictions:")
    for category, probability in predictions.items():
        risk_level = "🔴 HIGH" if probability > 0.7 else "🟡 MEDIUM" if probability > 0.3 else "🟢 LOW"
        print(f"  {category}: {probability:.2f} {risk_level}")
    
    # Get comprehensive status
    status = orchestrator.get_resilience_status()
    print(f"\nOverall resilience status:")
    print(f"  Recent success rate: {status['recent_success_rate']:.1f}%")
    print(f"  Total events tracked: {status['total_events']}")
    
    if status['recommendations']:
        print(f"  Recommendations:")
        for rec in status['recommendations'][:3]:
            print(f"    • {rec}")


async def demonstrate_circuit_breaker_patterns():
    """Demonstrate circuit breaker behavior patterns."""
    print(f"\n⚡ Circuit Breaker Patterns")
    print("-" * 40)
    
    orchestrator = ProgressiveResilienceOrchestrator()
    orchestrator.level = ProgressiveLevel.INTERMEDIATE
    orchestrator._initialize_default_strategies()
    
    # Configure for quick demonstration
    orchestrator.config.failure_threshold = 3
    orchestrator.config.circuit_breaker_timeout = 2.0
    
    service = MockService(failure_rate=0.8)  # High failure rate
    
    print("Testing circuit breaker with high failure rate...")
    
    for i in range(10):
        try:
            print(f"\n  Attempt {i+1}:")
            result = await resilient_execution(
                "circuit_test",
                service.unstable_operation,
                f"test_{i}"
            )
            print(f"    ✅ Success: {result[:50]}...")
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)[:50]}...")
        
        # Show circuit breaker state
        cb = orchestrator.get_circuit_breaker("circuit_test")
        print(f"    Circuit: {cb.state} (failures: {cb.failure_count})")
        
        # If circuit is open, wait a bit to show recovery
        if cb.state == "OPEN" and i < 9:
            print(f"    Waiting for circuit breaker timeout...")
            await asyncio.sleep(2.1)


async def run_comprehensive_demo():
    """Run comprehensive progressive resilience demo."""
    print("🌟 TERRAGON SDLC v6.0 - Progressive Resilience System")
    print("Advanced Self-Healing and Adaptive Resilience")
    print("=" * 60)
    
    try:
        # Basic resilience
        orchestrator = await demonstrate_basic_resilience()
        
        # Adaptive strategies
        await demonstrate_adaptive_strategies()
        
        # Decorators
        await demonstrate_decorators()
        
        # Circuit breaker patterns
        await demonstrate_circuit_breaker_patterns()
        
        # Failure prediction
        await demonstrate_failure_prediction()
        
        # Final status report
        print(f"\n📊 Final System Status")
        print("=" * 30)
        
        status = orchestrator.get_resilience_status()
        print(f"Resilience Level: {status['level'].upper()}")
        print(f"Active Strategies: {len(status['enabled_strategies'])}")
        print(f"Total Events: {status['total_events']}")
        print(f"Recent Success Rate: {status['recent_success_rate']:.1f}%")
        
        if status['circuit_breakers']:
            print(f"\nCircuit Breakers:")
            for name, cb_status in status['circuit_breakers'].items():
                print(f"  • {name}: {cb_status['state']} (failures: {cb_status['failure_count']})")
        
        if status['failure_predictions']:
            print(f"\nFailure Predictions:")
            for category, prob in status['failure_predictions'].items():
                print(f"  • {category}: {prob:.2f}")
        
        print(f"\n🎉 Progressive Resilience Demo Complete!")
        print(f"Successfully demonstrated:")
        print(f"  ✅ Adaptive circuit breakers with progressive thresholds")
        print(f"  ✅ Intelligent retry mechanisms with backoff strategies")
        print(f"  ✅ Dynamic rate limiting based on system load")
        print(f"  ✅ Self-healing capabilities with multiple recovery strategies")
        print(f"  ✅ Failure prediction using pattern recognition")
        print(f"  ✅ Level-based strategy evolution")
        print(f"  ✅ Comprehensive resilience monitoring and reporting")
        
        # Save configuration
        orchestrator.save_config()
        print(f"\n💾 Configuration saved for future runs")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo execution."""
    try:
        asyncio.run(run_comprehensive_demo())
    except Exception as e:
        print(f"Failed to run demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()