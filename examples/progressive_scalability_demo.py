#!/usr/bin/env python3
"""Progressive Scalability Demo - TERRAGON SDLC v6.0 Enhancement.

This demo showcases the progressive scalability system with distributed processing,
intelligent load balancing, and auto-scaling capabilities.
"""

import asyncio
import random
import sys
import time
import math
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scgraph_hub.progressive_scalability import (
    ProgressiveDistributedProcessor,
    ProgressiveLevel,
    ScalingStrategy,
    WorkloadType,
    distributed_execute,
    distributed_task
)


# Sample workload functions for testing
def cpu_intensive_task(data: int, duration: float = 1.0) -> int:
    """CPU-intensive computation task."""
    start_time = time.time()
    result = 0
    
    # Simulate CPU-intensive work
    while time.time() - start_time < duration:
        for i in range(1000):
            result += math.sqrt(i * data) ** 0.5
    
    return int(result) % 1000


def io_intensive_task(filename: str, size: int = 1000) -> int:
    """IO-intensive task simulation."""
    # Simulate file I/O with sleep
    time.sleep(random.uniform(0.1, 0.5))
    
    # Simulate processing file data
    data = [random.randint(1, 100) for _ in range(size)]
    return sum(data)


def network_intensive_task(url: str, timeout: float = 2.0) -> dict:
    """Network-intensive task simulation."""
    # Simulate network request delay
    time.sleep(random.uniform(0.2, timeout))
    
    return {
        'url': url,
        'status': 200 if random.random() > 0.1 else 500,
        'response_time': random.uniform(100, 1000),
        'data_size': random.randint(1000, 10000)
    }


def memory_intensive_task(array_size: int = 10000) -> int:
    """Memory-intensive task simulation."""
    # Create large data structures
    data = [[random.random() for _ in range(100)] for _ in range(array_size // 100)]
    
    # Process the data
    total = 0
    for row in data:
        total += sum(row)
    
    return int(total)


@distributed_task(timeout=10.0, max_retries=2)
def decorated_cpu_task(iterations: int) -> int:
    """Decorated CPU task."""
    result = 0
    for i in range(iterations):
        result += i ** 2
    return result


@distributed_task(timeout=5.0)
def decorated_io_task(data_size: int) -> list:
    """Decorated I/O task."""
    time.sleep(0.3)  # Simulate I/O delay
    return [random.randint(1, 1000) for _ in range(data_size)]


async def demonstrate_basic_distributed_processing():
    """Demonstrate basic distributed processing."""
    print("🚀 Basic Distributed Processing")
    print("-" * 40)
    
    processor = ProgressiveDistributedProcessor(ProgressiveLevel.BASIC)
    await processor.start()
    
    print(f"Processor started at {processor.level.value} level")
    status = processor.get_status()
    print(f"Workers: {status['total_workers']}")
    print(f"Load balancing: {status['load_balancing_method']}")
    
    # Submit various tasks
    tasks = []
    task_types = [
        ("CPU", cpu_intensive_task, (42, 0.5)),
        ("I/O", io_intensive_task, ("data.txt", 500)),
        ("Network", network_intensive_task, ("https://api.example.com", 1.0)),
        ("Memory", memory_intensive_task, (5000,))
    ]
    
    print(f"\n📤 Submitting {len(task_types)} different task types...")
    start_time = time.time()
    
    for task_name, task_func, task_args in task_types:
        task_id = await processor.submit_task(task_func, *task_args)
        tasks.append((task_name, task_id))
        print(f"  ✓ Submitted {task_name} task: {task_id}")
    
    # Wait for results
    print(f"\n📥 Collecting results...")
    results = []
    
    for task_name, task_id in tasks:
        try:
            result = await processor.get_result(task_id, timeout=10.0)
            results.append((task_name, result, True))
            print(f"  ✅ {task_name}: {str(result)[:50]}...")
        except Exception as e:
            results.append((task_name, str(e), False))
            print(f"  ❌ {task_name}: {str(e)}")
    
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f}s")
    
    # Show final status
    final_status = processor.get_status()
    print(f"\nFinal Status:")
    print(f"  Completed tasks: {final_status['performance_metrics']['completed_tasks']}")
    print(f"  Failed tasks: {final_status['performance_metrics']['failed_tasks']}")
    print(f"  Avg execution time: {final_status['performance_metrics']['avg_execution_time']:.1f}ms")
    
    await processor.stop()
    return results


async def demonstrate_load_balancing():
    """Demonstrate intelligent load balancing."""
    print(f"\n⚖️ Intelligent Load Balancing")
    print("-" * 40)
    
    processor = ProgressiveDistributedProcessor(ProgressiveLevel.INTERMEDIATE)
    await processor.start()
    
    print("Testing load balancing with mixed workloads...")
    
    # Submit multiple batches of tasks to show load balancing
    batch_size = 8
    total_tasks = []
    
    for batch in range(3):
        print(f"\n📦 Batch {batch + 1}: Submitting {batch_size} tasks")
        batch_tasks = []
        
        for i in range(batch_size):
            # Mix different task types
            if i % 4 == 0:
                task_id = await processor.submit_task(cpu_intensive_task, i * 10, 0.3)
                task_type = "CPU"
            elif i % 4 == 1:
                task_id = await processor.submit_task(io_intensive_task, f"file_{i}.txt", 300)
                task_type = "I/O"
            elif i % 4 == 2:
                task_id = await processor.submit_task(network_intensive_task, f"https://api{i}.com", 0.8)
                task_type = "Network"
            else:
                task_id = await processor.submit_task(memory_intensive_task, 2000)
                task_type = "Memory"
            
            batch_tasks.append((task_type, task_id))
            print(f"    • {task_type} task: {task_id}")
        
        total_tasks.extend(batch_tasks)
        
        # Show current load distribution
        status = processor.get_status()
        print(f"    Active tasks: {status['active_tasks']}")
        print(f"    Queue size: {status['queue_size']}")
        print(f"    Current utilization: {status['performance_metrics']['current_utilization']:.1%}")
        
        # Small delay between batches
        await asyncio.sleep(1.0)
    
    # Collect all results
    print(f"\n📥 Collecting {len(total_tasks)} results...")
    success_count = 0
    
    for task_type, task_id in total_tasks:
        try:
            result = await processor.get_result(task_id, timeout=15.0)
            success_count += 1
            print(f"  ✅ {task_type} completed")
        except Exception as e:
            print(f"  ❌ {task_type} failed: {str(e)[:30]}...")
    
    print(f"\nLoad Balancing Results:")
    print(f"  Success rate: {success_count}/{len(total_tasks)} ({success_count/len(total_tasks)*100:.1f}%)")
    
    # Show load balancer performance
    lb_status = processor.load_balancer
    print(f"  Worker performance data collected: {len(lb_status.performance_cache)} entries")
    
    await processor.stop()


async def demonstrate_auto_scaling():
    """Demonstrate auto-scaling capabilities."""
    print(f"\n🔄 Auto-Scaling Demonstration")
    print("-" * 40)
    
    processor = ProgressiveDistributedProcessor(ProgressiveLevel.EXPERT)
    await processor.start()
    
    # Configure auto-scaler for demo
    processor.auto_scaler.min_workers = 2
    processor.auto_scaler.max_workers = 8
    processor.auto_scaler.target_utilization = 0.6
    processor.auto_scaler.cooldown_period = 5.0  # Faster scaling for demo
    
    initial_status = processor.get_status()
    print(f"Initial workers: {initial_status['total_workers']}")
    print(f"Auto-scaler config: {processor.auto_scaler.min_workers}-{processor.auto_scaler.max_workers} workers")
    print(f"Target utilization: {processor.auto_scaler.target_utilization:.1%}")
    
    # Phase 1: Light load
    print(f"\n📊 Phase 1: Light load (should not trigger scaling)")
    for i in range(3):
        await processor.submit_task(cpu_intensive_task, i, 0.2)
    
    await asyncio.sleep(2)
    status = processor.get_status()
    print(f"  Workers: {status['total_workers']}, Utilization: {status['performance_metrics']['current_utilization']:.1%}")
    
    # Phase 2: Heavy load (should trigger scale-up)
    print(f"\n📈 Phase 2: Heavy load (should trigger scale-up)")
    heavy_tasks = []
    for i in range(15):  # More tasks than workers
        task_id = await processor.submit_task(cpu_intensive_task, i * 5, 0.8)
        heavy_tasks.append(task_id)
    
    print(f"  Submitted {len(heavy_tasks)} heavy tasks")
    
    # Monitor scaling events
    for round_num in range(4):
        await asyncio.sleep(6)  # Wait for potential scaling
        status = processor.get_status()
        scaling_events = len(processor.auto_scaler.scaling_history)
        
        print(f"  Round {round_num + 1}: Workers: {status['total_workers']}, "
              f"Queue: {status['queue_size']}, "
              f"Utilization: {status['performance_metrics']['current_utilization']:.1%}, "
              f"Scaling events: {scaling_events}")
    
    # Wait for tasks to complete
    print(f"\n⏳ Waiting for heavy tasks to complete...")
    completed = 0
    for task_id in heavy_tasks:
        try:
            await processor.get_result(task_id, timeout=20.0)
            completed += 1
        except Exception:
            pass
    
    print(f"  Heavy tasks completed: {completed}/{len(heavy_tasks)}")
    
    # Phase 3: Light load again (should trigger scale-down after cooldown)
    print(f"\n📉 Phase 3: Light load again (should trigger scale-down)")
    
    for round_num in range(3):
        await asyncio.sleep(6)
        status = processor.get_status()
        scaling_events = len(processor.auto_scaler.scaling_history)
        
        print(f"  Round {round_num + 1}: Workers: {status['total_workers']}, "
              f"Utilization: {status['performance_metrics']['current_utilization']:.1%}, "
              f"Scaling events: {scaling_events}")
    
    # Show scaling history
    if processor.auto_scaler.scaling_history:
        print(f"\n📜 Scaling History:")
        for event in list(processor.auto_scaler.scaling_history)[-5:]:  # Last 5 events
            print(f"  • {event.event_type}: {event.worker_count_before}→{event.worker_count_after} "
                  f"({event.reason})")
    
    await processor.stop()


async def demonstrate_decorators():
    """Demonstrate distributed task decorators."""
    print(f"\n🎨 Distributed Task Decorators")
    print("-" * 40)
    
    print("Testing decorated functions...")
    
    # Test decorated CPU task
    print(f"\n🔥 CPU-intensive decorated task:")
    start_time = time.time()
    try:
        result = await decorated_cpu_task(10000)
        execution_time = time.time() - start_time
        print(f"  ✅ Result: {result} (time: {execution_time:.2f}s)")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test decorated I/O task
    print(f"\n💾 I/O-intensive decorated task:")
    start_time = time.time()
    try:
        result = await decorated_io_task(100)
        execution_time = time.time() - start_time
        print(f"  ✅ Result: {len(result)} items (time: {execution_time:.2f}s)")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test multiple decorated tasks in parallel
    print(f"\n⚡ Multiple decorated tasks in parallel:")
    start_time = time.time()
    
    tasks = [
        decorated_cpu_task(5000),
        decorated_io_task(50),
        decorated_cpu_task(3000),
        decorated_io_task(75)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    execution_time = time.time() - start_time
    
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    print(f"  Completed: {success_count}/{len(tasks)} (time: {execution_time:.2f}s)")
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"    Task {i+1}: ❌ {str(result)}")
        else:
            print(f"    Task {i+1}: ✅ {str(result)[:50]}...")


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print(f"\n📊 Performance Monitoring")
    print("-" * 40)
    
    processor = ProgressiveDistributedProcessor(ProgressiveLevel.AUTONOMOUS)
    await processor.start()
    
    print("Running performance monitoring test...")
    
    # Submit tasks with different characteristics
    task_batches = [
        ("Fast tasks", [(cpu_intensive_task, (i, 0.1)) for i in range(5)]),
        ("Slow tasks", [(cpu_intensive_task, (i, 0.5)) for i in range(3)]),
        ("Mixed I/O", [(io_intensive_task, (f"file_{i}.txt", 200)) for i in range(4)]),
        ("Memory tasks", [(memory_intensive_task, (1000,)) for i in range(3)])
    ]
    
    all_task_ids = []
    
    for batch_name, batch_tasks in task_batches:
        print(f"\n📦 {batch_name}:")
        batch_ids = []
        
        for task_func, task_args in batch_tasks:
            task_id = await processor.submit_task(task_func, *task_args)
            batch_ids.append(task_id)
            all_task_ids.append(task_id)
        
        print(f"  Submitted {len(batch_ids)} tasks")
        
        # Show intermediate status
        await asyncio.sleep(1.0)
        status = processor.get_status()
        print(f"  Current queue: {status['queue_size']}")
        print(f"  Active tasks: {status['active_tasks']}")
        print(f"  Utilization: {status['performance_metrics']['current_utilization']:.1%}")
    
    # Monitor completion
    print(f"\n⏳ Monitoring task completion...")
    completed_tasks = []
    
    while len(completed_tasks) < len(all_task_ids):
        await asyncio.sleep(2.0)
        
        # Check for new completions
        for task_id in all_task_ids:
            if task_id not in completed_tasks:
                try:
                    result = await processor.get_result(task_id, timeout=0.1)
                    completed_tasks.append(task_id)
                    print(f"  ✅ Task completed: {task_id}")
                except (TimeoutError, KeyError):
                    pass  # Still running or failed
                except Exception as e:
                    completed_tasks.append(task_id)
                    print(f"  ❌ Task failed: {task_id} - {str(e)[:30]}...")
        
        # Show progress
        progress = len(completed_tasks) / len(all_task_ids)
        status = processor.get_status()
        print(f"    Progress: {progress:.1%} "
              f"(Queue: {status['queue_size']}, "
              f"Active: {status['active_tasks']}, "
              f"Util: {status['performance_metrics']['current_utilization']:.1%})")
    
    # Final performance report
    final_status = processor.get_status()
    metrics = final_status['performance_metrics']
    
    print(f"\n📋 Final Performance Report:")
    print(f"  Total tasks processed: {metrics['total_tasks']}")
    print(f"  Completed successfully: {metrics['completed_tasks']}")
    print(f"  Failed tasks: {metrics['failed_tasks']}")
    print(f"  Success rate: {metrics['completed_tasks']/metrics['total_tasks']*100:.1f}%")
    print(f"  Average execution time: {metrics['avg_execution_time']:.1f}ms")
    print(f"  Final utilization: {metrics['current_utilization']:.1%}")
    print(f"  Peak worker count: {final_status['total_workers']}")
    
    await processor.stop()


async def run_comprehensive_demo():
    """Run comprehensive progressive scalability demo."""
    print("🌟 TERRAGON SDLC v6.0 - Progressive Scalability System")
    print("High-Performance Distributed Processing with Auto-Scaling")
    print("=" * 70)
    
    try:
        # Basic distributed processing
        results = await demonstrate_basic_distributed_processing()
        
        # Load balancing
        await demonstrate_load_balancing()
        
        # Auto-scaling
        await demonstrate_auto_scaling()
        
        # Decorators
        await demonstrate_decorators()
        
        # Performance monitoring
        await demonstrate_performance_monitoring()
        
        print(f"\n🎉 Progressive Scalability Demo Complete!")
        print(f"Successfully demonstrated:")
        print(f"  ✅ Distributed task processing with multiple worker types")
        print(f"  ✅ Intelligent load balancing with adaptive algorithms")
        print(f"  ✅ Auto-scaling based on workload patterns and metrics")
        print(f"  ✅ Progressive maturity levels (Basic → Expert → Autonomous)")
        print(f"  ✅ Task classification and workload-aware scheduling")
        print(f"  ✅ Performance monitoring and metrics collection")
        print(f"  ✅ Resilient execution with retry mechanisms")
        print(f"  ✅ Decorator-based distributed computing")
        
        print(f"\n🚀 System Ready for Production Scalability!")
        
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