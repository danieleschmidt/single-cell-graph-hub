#!/usr/bin/env python3
"""
Generation 2 Robust Demo - MAKE IT ROBUST
Comprehensive error handling, validation, monitoring, and security.
"""

import sys
import os
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Demonstrate Generation 2 robust functionality."""
    print("🛡️ TERRAGON SDLC v4.0 - Generation 2 Demo: MAKE IT ROBUST")
    print("=" * 70)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Import Generation 2 components
        from scgraph_hub import (
            # Generation 1 (basic)
            get_enhanced_loader, create_model, list_available_models,
            # Generation 2 (robust)
            get_error_handler, robust_wrapper, get_validator, get_sanitizer,
            get_health_monitor, get_metrics_collector, get_system_monitor,
            get_security_manager, secure_operation, AdvancedLogger,
            performance_context, monitored_function
        )
        
        print("✅ Successfully imported Generation 2 components")
        
        # 1. Robust Error Handling Demo
        print("\n🛠️ Robust Error Handling Demo")
        print("-" * 40)
        
        error_handler = get_error_handler()
        
        @robust_wrapper(retry_count=2, retry_delay=0.1, fallback_value="fallback_result")
        def potentially_failing_function(should_fail=False):
            if should_fail:
                raise ValueError("Intentional failure for testing")
            return "success_result"
        
        # Test successful operation
        result = potentially_failing_function(should_fail=False)
        print(f"✅ Successful operation result: {result}")
        
        # Test operation with fallback
        result = potentially_failing_function(should_fail=True)
        print(f"✅ Fallback operation result: {result}")
        
        # Check error statistics
        error_stats = error_handler.get_error_stats()
        print(f"Error statistics: {error_stats['total_errors']} total errors")
        
        print("\n🎉 Generation 2 Robust Demo Completed Successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"❌ Import failed: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
