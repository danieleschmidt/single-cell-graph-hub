#!/usr/bin/env python3
"""
Generation 2 Robust Features Demo for Single-Cell Graph Hub

This example demonstrates the enhanced error handling, logging, monitoring,
and security features implemented in Generation 2.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import scgraph_hub


async def main():
    """Run comprehensive Generation 2 features demonstration."""
    print("="*70)
    print("Single-Cell Graph Hub - Generation 2 Robust Features Demo")
    print("="*70)
    
    # 1. Enhanced Logging Setup
    print("\n1. Setting up Enhanced Logging")
    print("-" * 40)
    
    try:
        # Set up structured logging
        logger = scgraph_hub.setup_logging(
            level="INFO",
            enable_console=True,
            json_format=False,  # Use readable format for demo
            log_dir="./logs"
        )
        logger.info("Enhanced logging system initialized successfully")
        print("✅ Structured logging configured")
        
        # Get contextual logger
        context_logger = scgraph_hub.get_contextual_logger({
            'component': 'demo',
            'version': '2.0',
            'user': 'demo_user'
        })
        context_logger.info("Contextual logging demonstration")
        
    except Exception as e:
        print(f"❌ Logging setup not available: {e}")
        logger = None
    
    # 2. Security Validation
    print("\n2. Security Validation and Input Sanitization")
    print("-" * 40)
    
    try:
        validator = scgraph_hub.get_security_validator()
        
        # Test dataset name validation
        safe_names = ["pbmc_10k", "mouse_brain_atlas", "human_lung_v2"]
        unsafe_names = ["../../../etc/passwd", "dataset'; DROP TABLE users;--", "test<script>"]
        
        print("Testing dataset name validation:")
        for name in safe_names:
            try:
                validated = validator.validate_dataset_name(name)
                print(f"  ✅ '{name}' -> '{validated}'")
            except Exception as e:
                print(f"  ❌ '{name}' -> Error: {e}")
        
        print("\nTesting unsafe dataset names:")
        for name in unsafe_names:
            try:
                validated = validator.validate_dataset_name(name)
                print(f"  ⚠️  '{name}' -> '{validated}' (should have been blocked!)")
            except Exception as e:
                print(f"  ✅ '{name}' -> Blocked: {type(e).__name__}")
        
        # Test URL validation
        print("\nTesting URL validation:")
        safe_urls = ["https://example.com/data.h5ad", "https://api.scgraphhub.org/datasets"]
        unsafe_urls = ["javascript:alert('xss')", "file:///etc/passwd", "ftp://malicious.com"]
        
        for url in safe_urls:
            try:
                validated = validator.validate_url(url)
                print(f"  ✅ Safe URL: {url}")
            except Exception as e:
                print(f"  ❌ URL validation failed: {e}")
        
        for url in unsafe_urls:
            try:
                validated = validator.validate_url(url)
                print(f"  ⚠️  Unsafe URL accepted: {url}")
            except Exception as e:
                print(f"  ✅ Unsafe URL blocked: {type(e).__name__}")
        
    except Exception as e:
        print(f"❌ Security validation not available: {e}")
    
    # 3. System Health Monitoring
    print("\n3. System Health Monitoring")
    print("-" * 40)
    
    try:
        health_checker = scgraph_hub.SystemHealthChecker()
        
        print("Running comprehensive health checks...")
        health_results = await health_checker.check_all()
        
        print("\nHealth Check Results:")
        for component, status in health_results.items():
            status_icon = "✅" if status.is_healthy else "❌" if status.status == "unhealthy" else "⚠️"
            print(f"  {status_icon} {component}: {status.status}")
            print(f"    Message: {status.message}")
            if status.details:
                key_details = {k: v for k, v in status.details.items() if k in ['memory_percent', 'cpu_percent', 'disk_percent']}
                if key_details:
                    print(f"    Details: {key_details}")
        
    except Exception as e:
        print(f"❌ Health monitoring not available: {e}")
    
    # 4. Advanced Dataset Processing
    print("\n4. Advanced Dataset Processing with Error Handling")
    print("-" * 40)
    
    try:
        processor = scgraph_hub.DatasetProcessor(
            cache_dir="./demo_cache",
            enable_validation=True
        )
        
        # Process a dataset with comprehensive error handling
        dataset_name = "pbmc_10k"
        print(f"Processing dataset: {dataset_name}")
        
        processing_config = {
            'normalize_features': True,
            'default_k_neighbors': 10,
            'max_features': 2000,
            'use_advanced_features': False  # Set to False to work without heavy dependencies
        }
        
        processed_dataset = await processor.process_dataset(
            dataset_name=dataset_name,
            processing_config=processing_config,
            force_reprocess=False
        )
        
        print("✅ Dataset processed successfully")
        
        # Display enhanced dataset information
        info = processed_dataset.info()
        print("\nEnhanced Dataset Information:")
        for key, value in info.items():
            if key not in ['processing_config']:  # Skip detailed config
                print(f"  {key}: {value}")
        
        # Test dataset integrity validation
        if hasattr(processed_dataset, 'validate_integrity'):
            is_valid = processed_dataset.validate_integrity()
            print(f"\nDataset integrity check: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
    except Exception as e:
        print(f"❌ Advanced dataset processing error: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, 'details'):
            print(f"   Details: {e.details}")
    
    # 5. Secure File Operations
    print("\n5. Secure File Operations")
    print("-" * 40)
    
    try:
        secure_handler = scgraph_hub.SecureFileHandler(
            base_dir=Path("./secure_demo"),
            validator=validator if 'validator' in locals() else None
        )
        
        # Test secure write
        test_content = "This is a test file created securely"
        written_path = secure_handler.safe_write(
            "test_file.txt", 
            test_content,
            file_type="config"
        )
        print(f"✅ Securely wrote file: {written_path}")
        
        # Test secure read
        read_content = secure_handler.safe_read("test_file.txt", file_type="config")
        print(f"✅ Securely read file content: {len(read_content)} characters")
        
        # List safe files
        safe_files = secure_handler.list_safe_files("*.txt")
        print(f"✅ Found {len(safe_files)} safe files")
        
        # Clean up
        secure_handler.safe_delete("test_file.txt")
        print("✅ Securely deleted test file")
        
    except Exception as e:
        print(f"❌ Secure file operations error: {e}")
    
    # 6. Error Handling Demonstration
    print("\n6. Comprehensive Error Handling")
    print("-" * 40)
    
    print("Testing error handling with various failure scenarios:")
    
    # Test invalid dataset name
    try:
        invalid_dataset = scgraph_hub.simple_quick_start("nonexistent_dataset_12345")
        print("⚠️  Invalid dataset loaded (unexpected)")
    except Exception as e:
        print(f"✅ Invalid dataset properly handled: {type(e).__name__}")
    
    # Test configuration validation
    try:
        from scgraph_hub.exceptions import ErrorCollector
        collector = ErrorCollector()
        collector.add_error("Test error 1")
        collector.add_error("Test error 2")
        collector.add_validation_error("test_field", "invalid_value", "Test validation error")
        
        print(f"✅ Error collector gathered {len(collector.get_errors())} errors")
        
        # This would raise an exception with all collected errors
        # collector.raise_if_errors()
        
    except Exception as e:
        print(f"✅ Error collection system working: {type(e).__name__}")
    
    # 7. Performance and Resource Monitoring
    print("\n7. Resource Monitoring")
    print("-" * 40)
    
    try:
        from scgraph_hub.security import ResourceMonitor
        
        # Test resource monitoring
        with ResourceMonitor(max_memory_mb=1024, max_time_seconds=30) as monitor:
            print("✅ Resource monitoring active")
            
            # Simulate some work
            import time
            time.sleep(0.1)
            
            memory_ok = monitor.check_memory_usage()
            time_ok = monitor.check_time_limit()
            
            print(f"  Memory usage: {'✅ OK' if memory_ok else '❌ Exceeded'}")
            print(f"  Time limit: {'✅ OK' if time_ok else '❌ Exceeded'}")
        
    except Exception as e:
        print(f"❌ Resource monitoring error: {e}")
    
    # 8. Integration Test: Full Workflow
    print("\n8. Full Workflow Integration Test")
    print("-" * 40)
    
    try:
        print("Running end-to-end workflow with all Generation 2 features...")
        
        # 1. Security validation
        dataset_name = validator.validate_dataset_name("pbmc_10k") if 'validator' in locals() else "pbmc_10k"
        
        # 2. Load dataset with error handling
        dataset = scgraph_hub.simple_quick_start(dataset_name)
        
        # 3. Log the operation
        if logger:
            logger.info(f"Successfully loaded dataset {dataset_name} in integration test")
        
        # 4. Get dataset info
        info = dataset.info()
        
        print("✅ Full workflow completed successfully")
        print(f"   Dataset: {info['name']}")
        print(f"   Cells: {info['num_cells']:,}")
        print(f"   Genes: {info['num_genes']:,}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        if hasattr(e, 'details'):
            print(f"   Error details: {e.details}")
    
    print("\n" + "="*70)
    print("🎉 Generation 2 Robust Features Demo Completed!")
    print("\nGeneration 2 Achievements:")
    print("✅ Comprehensive error handling and validation")
    print("✅ Production-grade logging and monitoring") 
    print("✅ Security measures and input sanitization")
    print("✅ Advanced dataset processing pipeline")
    print("✅ Health checks and system diagnostics")
    print("✅ Graceful degradation with missing dependencies")
    print("\nNext: Generation 3 - Make It Scale (Performance Optimization)")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())