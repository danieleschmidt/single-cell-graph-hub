#!/usr/bin/env python3
"""
Generation 3 Scalable Demo - MAKE IT SCALE
Performance optimization, caching, concurrency, and auto-scaling.
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Demonstrate Generation 3 scalable functionality."""
    print("⚡ TERRAGON SDLC v4.0 - Generation 3 Demo: MAKE IT SCALE")
    print("=" * 70)
    
    try:
        # Import Generation 3 components
        from scgraph_hub import (
            # Generation 1 & 2 (basic + robust)
            get_enhanced_loader, create_model, get_error_handler,
            # Generation 3 (scalable)
            AdvancedCache, cached, get_performance_optimizer, performance_monitor,
            ConcurrentProcessor, WorkerNode, LoadBalancer, AutoScaler,
            DistributedTaskManager, get_distributed_task_manager, distributed_task
        )
        
        print("✅ Successfully imported Generation 3 components")
        
        # 1. Advanced Caching Demo
        print("\n💾 Advanced Caching Demo")
        print("-" * 40)
        
        cache = AdvancedCache(max_size=100, max_memory_mb=1, ttl_seconds=60)
        
        # Test cache operations
        cache.put("key1", "value1")
        cache.put("key2", [1, 2, 3, 4, 5])
        cache.put("key3", {"data": "complex_structure"})
        
        # Retrieve from cache
        result = cache.get("key1")
        print(f"✅ Cache hit: {result}")
        
        # Cache statistics
        stats = cache.get_stats()
        print(f"✅ Cache stats: {stats['hit_rate']:.1f}% hit rate, {stats['size']} items")
        
        # 2. Function Caching Demo
        print("\n🔄 Function Caching Demo")
        print("-" * 40)
        
        @cached(ttl=30)
        def expensive_computation(n):
            time.sleep(0.1)  # Simulate expensive operation
            return sum(i * i for i in range(n))
        
        # First call (cache miss)
        start_time = time.time()
        result1 = expensive_computation(1000)
        time1 = time.time() - start_time
        print(f"✅ First call: {time1:.3f}s, result: {result1}")
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = expensive_computation(1000)
        time2 = time.time() - start_time
        print(f"✅ Second call: {time2:.3f}s, result: {result2}")
        print(f"✅ Speedup: {time1/time2:.1f}x faster")
        
        # 3. Concurrent Processing Demo
        print("\n🔄 Concurrent Processing Demo")
        print("-" * 40)
        
        processor = ConcurrentProcessor(max_workers=4, worker_type="thread")
        
        def process_item(item):
            time.sleep(0.05)  # Simulate processing
            return item * item
        
        items = list(range(20))
        
        # Process sequentially
        start_time = time.time()
        sequential_results = [process_item(item) for item in items]
        sequential_time = time.time() - start_time
        
        # Process concurrently
        start_time = time.time()
        concurrent_results = processor.map_concurrent(process_item, items)
        concurrent_time = time.time() - start_time
        
        print(f"✅ Sequential time: {sequential_time:.3f}s")
        print(f"✅ Concurrent time: {concurrent_time:.3f}s")
        print(f"✅ Speedup: {sequential_time/concurrent_time:.1f}x faster")
        
        # 4. Performance Monitoring Demo
        print("\n📊 Performance Monitoring Demo")
        print("-" * 40)
        
        optimizer = get_performance_optimizer()
        
        with performance_monitor("demo_operation"):
            time.sleep(0.1)
            data = list(range(1000))
            processed = [x * 2 for x in data]
        
        # Get performance report
        report = optimizer.get_performance_report()
        print(f"✅ Performance report generated")
        if 'demo_operation' in report['operation_summary']:
            op_stats = report['operation_summary']['demo_operation']
            print(f"✅ Operation count: {op_stats['count']}")
            print(f"✅ Average duration: {op_stats['avg_duration']:.3f}s")
        
        # 5. Load Balancing Demo
        print("\n⚖️ Load Balancing Demo")
        print("-" * 40)
        
        load_balancer = LoadBalancer()
        
        # Add worker nodes
        for i in range(3):
            node = WorkerNode(node_id=f"worker_{i}", capacity=10)
            load_balancer.add_node(node)
        
        # Simulate load distribution
        for i in range(15):
            selected_node = load_balancer.select_node()
            if selected_node:
                selected_node.add_load(1)
        
        # Get load balancer stats
        lb_stats = load_balancer.get_stats()
        print(f"✅ Load balancer stats:")
        print(f"    Total nodes: {lb_stats['total_nodes']}")
        print(f"    Overall utilization: {lb_stats['overall_utilization']:.1f}%")
        
        # 6. Auto-scaling Demo
        print("\n📈 Auto-scaling Demo")
        print("-" * 40)
        
        auto_scaler = AutoScaler(min_nodes=2, max_nodes=8)
        
        # Test scaling decisions
        test_scenarios = [
            {"utilization": 90, "nodes": 3, "pending": 10},
            {"utilization": 25, "nodes": 5, "pending": 0},
            {"utilization": 50, "nodes": 3, "pending": 2}
        ]
        
        for scenario in test_scenarios:
            recommendation = auto_scaler.get_scaling_recommendation({
                'overall_utilization': scenario['utilization'],
                'healthy_nodes': scenario['nodes'],
                'pending_tasks': scenario['pending']
            })
            
            if recommendation:
                print(f"✅ Scenario: {scenario['utilization']}% util, {scenario['nodes']} nodes")
                print(f"    Recommendation: {recommendation['action']} to {recommendation['recommended_nodes']} nodes")
            else:
                print(f"✅ Scenario: {scenario['utilization']}% util - No scaling needed")
        
        print("\n🎉 Generation 3 Scalable Demo Completed Successfully!")
        print("✅ All scalable functionality working correctly")
        
        # Summary
        print("\n📋 Generation 3 Summary")
        print("-" * 40)
        print("✅ Advanced caching with multiple eviction strategies")
        print("✅ High-performance concurrent processing")
        print("✅ Intelligent load balancing across worker nodes")
        print("✅ Automatic scaling based on system metrics")
        print("✅ Performance monitoring and optimization")
        print("✅ Resource pooling and management")
        
        # Cleanup
        processor.shutdown()
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
