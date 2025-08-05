"""Demonstration of health checks and monitoring capabilities."""

import asyncio
import json
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scgraph_hub.health_checks import run_health_check, get_performance_monitor, performance_timer
from scgraph_hub.logging_config import setup_logging
from scgraph_hub.validators import DatasetValidator, ValidationResult


@performance_timer("demo_operation")
def demo_operation():
    """Demo operation to test performance monitoring."""
    import time
    time.sleep(0.1)  # Simulate work
    return "Operation completed"


@performance_timer("demo_validation")
def demo_validation():
    """Demo validation to test validation system."""
    config = {
        "name": "test_dataset",
        "n_cells": 5000,
        "n_genes": 2000,
        "modality": "scRNA-seq",
        "organism": "human"
    }
    
    result = DatasetValidator.validate_config(config)
    return result


async def main():
    """Run health check and monitoring demonstration."""
    print("Single-Cell Graph Hub - Health Check & Monitoring Demo")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging(level="INFO", enable_console=True)
    logger.info("Health check demo started")
    
    # Run some demo operations to generate performance metrics
    print("\n1. Running demo operations to generate metrics...")
    for i in range(5):
        result = demo_operation()
        print(f"   Operation {i+1}: {result}")
    
    # Test validation system
    print("\n2. Testing validation system...")
    validation_result = demo_validation()
    print(f"   Validation result: {'PASSED' if validation_result.is_valid else 'FAILED'}")
    if validation_result.warnings:
        print(f"   Warnings: {'; '.join(validation_result.warnings)}")
    
    # Get performance summary
    print("\n3. Performance metrics:")
    monitor = get_performance_monitor()
    summary = monitor.get_summary()
    print(f"   Total operations: {summary['total_operations']}")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Average duration: {summary['avg_duration']:.3f}s")
    print(f"   Uptime: {monitor.get_uptime():.1f}s")
    
    # Run comprehensive health check
    print("\n4. Running comprehensive health check...")
    health_results = await run_health_check()
    
    print(f"   Overall status: {health_results['overall_status'].upper()}")
    print(f"   Check timestamp: {health_results['timestamp']}")
    print(f"   System uptime: {health_results['uptime_seconds']:.1f}s")
    
    print("\n   Component health:")
    for component, status in health_results['components'].items():
        status_icon = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'degraded' else "❌"
        print(f"   {status_icon} {component}: {status['status']} - {status['message']}")
        
        # Show details for system resources
        if component == 'system_resources' and status.get('details'):
            details = status['details']
            print(f"      Memory: {details.get('memory_percent', 0):.1f}%")
            print(f"      CPU: {details.get('cpu_percent', 0):.1f}%")
            print(f"      Available RAM: {details.get('memory_available_gb', 0):.1f} GB")
    
    # Test error handling with invalid data
    print("\n5. Testing error handling with invalid data...")
    try:
        invalid_config = {
            "name": "bad-name!@#",  # Invalid characters
            "n_cells": -100,        # Negative value
            "n_genes": "not_a_number",  # Wrong type
            "modality": "invalid_modality",
            "organism": "unknown_species"
        }
        
        validation_result = DatasetValidator.validate_config(invalid_config)
        print(f"   Validation result: {'PASSED' if validation_result.is_valid else 'FAILED'}")
        if validation_result.errors:
            print("   Errors found:")
            for error in validation_result.errors:
                print(f"     - {error}")
        if validation_result.warnings:
            print("   Warnings:")
            for warning in validation_result.warnings:
                print(f"     - {warning}")
                
    except Exception as e:
        print(f"   Exception caught: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Health check and monitoring demo completed successfully!")
    print("\nKey features demonstrated:")
    print("- Comprehensive system health monitoring")
    print("- Performance metrics collection and analysis")
    print("- Robust data validation with detailed error reporting")
    print("- Structured logging with context")
    print("- Graceful error handling and recovery")


if __name__ == "__main__":
    asyncio.run(main())