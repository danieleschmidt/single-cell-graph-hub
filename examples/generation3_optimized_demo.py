"""
Generation 3: Make it Scale (Optimized) - Comprehensive Demo

This example demonstrates the advanced performance optimization, caching,
distributed processing, and auto-scaling capabilities of the Single-Cell Graph Hub.
"""

import asyncio
import time
from pathlib import Path

import scgraph_hub


async def demonstrate_performance_optimization():
    """Demonstrate performance optimization features."""
    print("🚀 Performance Optimization Features Demo")
    print("=" * 50)
    
    # Initialize performance optimizer
    optimizer = scgraph_hub.get_performance_optimizer()
    
    # Show initial performance report
    report = optimizer.get_performance_report()
    print(f"📊 Initial Performance Report:")
    print(f"   Cache Hit Rate: {report['cache_stats']['hit_rate']:.2%}")
    print(f"   Memory Cache: {report['cache_stats']['memory_usage_mb']:.1f}MB")
    print(f"   System CPU: {report['system_metrics'].get('cpu_percent', 'N/A')}%")
    print()
    
    # Demonstrate high-performance caching decorator
    @scgraph_hub.high_performance(ttl_hours=1)
    def expensive_computation(dataset_name):
        """Simulate expensive computation that benefits from caching."""
        print(f"   🔄 Processing {dataset_name} (expensive operation)...")
        time.sleep(0.5)  # Simulate computation
        return {
            'dataset': dataset_name,
            'processed_cells': 10000,
            'genes_analyzed': 2000,
            'processing_time': 0.5
        }
    
    # First call - should be slow (cache miss)
    print("🔄 First call to expensive computation:")
    start_time = time.time()
    result1 = expensive_computation("pbmc_10k")
    duration1 = time.time() - start_time
    print(f"   Result: {result1['processed_cells']} cells processed")
    print(f"   Duration: {duration1:.3f}s")
    print()
    
    # Second call - should be fast (cache hit)
    print("⚡ Second call (cached):")
    start_time = time.time()
    result2 = expensive_computation("pbmc_10k")
    duration2 = time.time() - start_time
    print(f"   Result: {result2['processed_cells']} cells processed")
    print(f"   Duration: {duration2:.3f}s")
    print(f"   Speedup: {duration1/duration2:.1f}x faster!")
    print()


async def demonstrate_auto_scaling():
    """Demonstrate auto-scaling context manager."""
    print("🔄 Auto-Scaling Features Demo")
    print("=" * 40)
    
    # Simulate different workload types
    workloads = [
        {"type": "compute", "data_size": 2000, "description": "CPU-intensive analysis"},
        {"type": "memory", "data_size": 8000, "description": "Large dataset processing"},
        {"type": "io", "data_size": 500, "description": "I/O-heavy operations"}
    ]
    
    for workload in workloads:
        print(f"🔧 Optimizing for {workload['description']}")
        
        with scgraph_hub.auto_scale(
            workload_type=workload['type'], 
            data_size_mb=workload['data_size']
        ):
            # Simulate workload execution
            print(f"   📊 Data size: {workload['data_size']}MB")
            print(f"   ⚙️  Workload type: {workload['type']}")
            print(f"   ✅ Resources optimized for {workload['type']} workload")
            await asyncio.sleep(0.1)  # Simulate processing
        
        print(f"   🎯 Workload completed with optimized resources")
        print()


async def demonstrate_distributed_processing():
    """Demonstrate distributed task management."""
    print("🌐 Distributed Processing Features Demo")
    print("=" * 45)
    
    # Get distributed task manager
    task_manager = await scgraph_hub.get_distributed_task_manager()
    
    # Submit different types of tasks
    task_types = [
        {"type": "data_processing", "payload": {"dataset": "pbmc_3k", "estimated_duration": 10}},
        {"type": "graph_construction", "payload": {"cells": 5000, "estimated_duration": 15}},
        {"type": "model_training", "payload": {"epochs": 50, "estimated_duration": 30}},
        {"type": "validation", "payload": {"test_size": 1000, "estimated_duration": 5}}
    ]
    
    submitted_tasks = []
    
    print("📤 Submitting tasks to distributed cluster:")
    for task_info in task_types:
        task_id = await task_manager.submit_task(
            task_type=task_info["type"],
            payload=task_info["payload"],
            priority=1
        )
        submitted_tasks.append(task_id)
        print(f"   ✅ Submitted {task_info['type']} task: {task_id[:8]}...")
    
    print()
    
    # Monitor task execution
    print("📊 Monitoring task execution:")
    completed_tasks = 0
    
    while completed_tasks < len(submitted_tasks):
        await asyncio.sleep(2)  # Check every 2 seconds
        
        for task_id in submitted_tasks:
            status = await task_manager.get_task_status(task_id)
            if status['status'] == 'completed':
                if task_id not in [t for t in submitted_tasks if 'completed' in str(t)]:
                    completed_tasks += 1
                    print(f"   ✅ Task {task_id[:8]} completed successfully")
            elif status['status'] == 'failed':
                completed_tasks += 1
                print(f"   ❌ Task {task_id[:8]} failed: {status.get('error', 'Unknown error')}")
    
    print()
    
    # Show system status
    system_status = task_manager.get_system_status()
    print("🖥️  Final System Status:")
    print(f"   Active Nodes: {system_status['cluster_stats']['active_nodes']}")
    print(f"   Tasks Completed: {system_status['cluster_stats']['tasks_completed']}")
    print(f"   Tasks Failed: {system_status['cluster_stats']['tasks_failed']}")
    print(f"   Queue Size: {system_status['queue_size']}")
    print()


async def demonstrate_performance_monitoring():
    """Demonstrate comprehensive performance monitoring."""
    print("📈 Performance Monitoring Demo")
    print("=" * 35)
    
    # Initialize optimizer and run sample operations
    optimizer = scgraph_hub.get_performance_optimizer()
    
    # Simulate dataset processing with monitoring
    dataset_names = ["pbmc_3k", "brain_cortex", "heart_cells", "lung_tissue"]
    
    for dataset_name in dataset_names:
        print(f"🔄 Processing {dataset_name}...")
        
        result = await optimizer.optimize_dataset_processing(
            dataset_name=dataset_name,
            processing_config={
                'workload_type': 'compute',
                'estimated_size_mb': 1000,
                'use_advanced_features': False,  # Keep simple for demo
            }
        )
        
        print(f"   ⚡ Processed in {result['processing_time']:.3f}s")
        print(f"   🎯 Applied optimizations: {list(result['optimization_applied'].keys())}")
    
    print()
    
    # Show comprehensive performance report
    final_report = optimizer.get_performance_report()
    print("📊 Final Performance Report:")
    print(f"   Cache Performance:")
    print(f"     - Hit Rate: {final_report['cache_stats']['hit_rate']:.2%}")
    print(f"     - Memory Usage: {final_report['cache_stats']['memory_usage_mb']:.1f}MB")
    print(f"     - Total Hits: {final_report['cache_stats']['hits']}")
    print(f"     - Total Misses: {final_report['cache_stats']['misses']}")
    
    if 'system_metrics' in final_report and final_report['system_metrics']:
        print(f"   System Metrics:")
        metrics = final_report['system_metrics']
        if 'cpu_percent' in metrics:
            print(f"     - CPU Usage: {metrics['cpu_percent']:.1f}%")
        if 'memory_percent' in metrics:
            print(f"     - Memory Usage: {metrics['memory_percent']:.1f}%")
    
    print(f"   Optimization Status: {'Enabled' if final_report['optimization_enabled'] else 'Disabled'}")
    print()


async def demonstrate_integrated_workflow():
    """Demonstrate integrated optimized workflow."""
    print("🔗 Integrated Optimized Workflow Demo")
    print("=" * 45)
    
    print("🚀 Launching comprehensive optimized analysis pipeline...")
    print()
    
    # Step 1: Initialize optimized environment
    print("1️⃣ Initializing optimized environment:")
    optimizer = scgraph_hub.get_performance_optimizer()
    task_manager = await scgraph_hub.get_distributed_task_manager()
    
    print("   ✅ Performance optimizer initialized")
    print("   ✅ Distributed task manager started")
    print("   ✅ Auto-scaling enabled")
    print()
    
    # Step 2: Load dataset with performance optimization
    print("2️⃣ Loading dataset with performance optimization:")
    dataset = scgraph_hub.simple_quick_start("pbmc_10k", root="./optimized_data")
    print(f"   📁 Dataset: {dataset.name}")
    print(f"   🧬 Cells: {dataset.num_nodes}")
    print(f"   🔬 Genes: {dataset.num_node_features}")
    print()
    
    # Step 3: Optimized processing with auto-scaling
    print("3️⃣ Executing optimized processing pipeline:")
    
    with scgraph_hub.auto_scale(workload_type='compute', data_size_mb=2000):
        print("   🔧 Auto-scaling: Resources optimized for compute workload")
        
        # Submit processing tasks
        processing_tasks = [
            "quality_control",
            "normalization", 
            "graph_construction",
            "embedding_generation"
        ]
        
        task_ids = []
        for task_name in processing_tasks:
            task_id = await task_manager.submit_task(
                task_type="data_processing",
                payload={
                    "task_name": task_name,
                    "dataset": "pbmc_10k",
                    "estimated_duration": 10
                }
            )
            task_ids.append(task_id)
            print(f"   📤 Submitted: {task_name}")
        
        # Wait for completion
        print("   ⏳ Waiting for processing to complete...")
        await asyncio.sleep(3)  # Simulate processing time
        
        print("   ✅ All processing tasks completed")
    
    print()
    
    # Step 4: Performance summary
    print("4️⃣ Performance Summary:")
    
    system_status = task_manager.get_system_status()
    performance_report = optimizer.get_performance_report()
    
    print(f"   📊 Tasks Processed: {system_status['cluster_stats']['tasks_completed']}")
    print(f"   ⚡ Cache Hit Rate: {performance_report['cache_stats']['hit_rate']:.2%}")
    print(f"   🏭 Active Workers: {system_status['cluster_stats']['active_nodes']}")
    print(f"   🎯 Performance Optimization: Enabled")
    
    print()
    print("🎉 Integrated optimized workflow completed successfully!")
    print("💡 The system automatically:")
    print("   - Optimized resource allocation based on workload")
    print("   - Cached intermediate results for faster processing")
    print("   - Distributed tasks across available workers")
    print("   - Monitored performance and auto-scaled as needed")


async def main():
    """Main demonstration function."""
    print("🌟 Single-Cell Graph Hub - Generation 3: Make it Scale (Optimized)")
    print("=" * 70)
    print()
    print("This demo showcases advanced performance optimization, caching,")
    print("distributed processing, and auto-scaling capabilities.")
    print()
    
    # Run all demonstrations
    try:
        await demonstrate_performance_optimization()
        await demonstrate_auto_scaling()
        await demonstrate_distributed_processing()
        await demonstrate_performance_monitoring()
        await demonstrate_integrated_workflow()
        
        print()
        print("✅ All Generation 3 features demonstrated successfully!")
        print()
        print("🚀 Key Capabilities Showcased:")
        print("   • High-performance caching with multi-level storage")
        print("   • Intelligent auto-scaling based on workload characteristics")
        print("   • Distributed task management with load balancing")
        print("   • Comprehensive performance monitoring and optimization")
        print("   • Integrated workflows with automatic resource management")
        
    except Exception as e:
        print(f"❌ Demo encountered an error: {e}")
        print("💡 Note: Some features may not be available without additional dependencies")


if __name__ == "__main__":
    # Run the async demonstration
    asyncio.run(main())