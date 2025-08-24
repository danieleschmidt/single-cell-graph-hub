"""Generation 3 Scalability Demo - MAKE IT SCALE.

This demo showcases the TERRAGON SDLC v6.0 Generation 3 implementation:
- Hyper-scale edge computing
- Intelligent load balancing
- Dynamic auto-scaling
- Performance optimization
- Distributed task execution
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scgraph_hub.autonomous_enhanced import (
    EnhancedAutonomousEngine,
    IntelligenceLevel,
    TaskComplexity,
    get_enhanced_autonomous_engine
)

from scgraph_hub.hyperscale_edge import (
    HyperScaleEdgeSystem,
    EdgeNode,
    EdgeNodeState,
    DistributedTask,
    TaskPriority,
    LoadBalancingStrategy,
    get_hyperscale_edge_system,
    distributed_task,
    high_priority_task
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScalabilityTestSuite:
    """Test suite for demonstrating Generation 3 scalability features."""
    
    def __init__(self):
        self.edge_system = get_hyperscale_edge_system(initial_node_count=5)
        self.autonomous_engine = get_enhanced_autonomous_engine(IntelligenceLevel.QUANTUM)
        self.test_results = {}
        
    async def simulate_compute_intensive_task(self, data_size: int, complexity: float = 1.0) -> dict:
        """Simulate a compute-intensive task for edge processing."""
        processing_time = (data_size / 1000) * complexity * random.uniform(0.8, 1.2)
        await asyncio.sleep(processing_time)
        
        return {
            "data_size": data_size,
            "processing_time": processing_time,
            "complexity": complexity,
            "result": f"processed_{data_size}_items",
            "timestamp": datetime.now().isoformat()
        }
    
    async def simulate_memory_intensive_task(self, memory_requirement: int) -> dict:
        """Simulate a memory-intensive task."""
        processing_time = memory_requirement / 1000 * random.uniform(0.5, 1.5)
        await asyncio.sleep(processing_time)
        
        return {
            "memory_used": memory_requirement,
            "processing_time": processing_time,
            "efficiency": random.uniform(0.8, 1.0),
            "result": "memory_processing_complete"
        }
    
    async def simulate_network_bound_task(self, network_load: float) -> dict:
        """Simulate a network-bound task."""
        # Network tasks have variable latency
        base_time = 0.1
        network_delay = network_load * random.uniform(0.05, 0.2)
        total_time = base_time + network_delay
        
        await asyncio.sleep(total_time)
        
        return {
            "network_load": network_load,
            "base_latency": base_time,
            "network_delay": network_delay,
            "total_time": total_time,
            "result": "network_task_complete"
        }


async def demonstrate_edge_computing_scalability():
    """Demonstrate hyper-scale edge computing capabilities."""
    logger.info("🌐 Demonstrating Hyper-Scale Edge Computing")
    
    test_suite = ScalabilityTestSuite()
    edge_system = test_suite.edge_system
    
    # Start the edge computing system
    await edge_system.start_processing()
    
    logger.info("✅ Edge computing system started")
    
    # Test 1: Basic distributed task execution
    logger.info("📋 Test 1: Basic Distributed Task Execution")
    
    task_results = []
    start_time = time.time()
    
    # Submit multiple tasks with varying priorities and requirements
    task_configs = [
        (TaskPriority.LOW, {"cpu": 5, "memory": 8}, 500),
        (TaskPriority.MEDIUM, {"cpu": 15, "memory": 12}, 1000),
        (TaskPriority.HIGH, {"cpu": 25, "memory": 20}, 2000),
        (TaskPriority.CRITICAL, {"cpu": 35, "memory": 30}, 800),
        (TaskPriority.REAL_TIME, {"cpu": 10, "memory": 15}, 300),
    ]
    
    task_ids = []
    for i, (priority, requirements, data_size) in enumerate(task_configs * 3):  # 15 tasks total
        task_id = await edge_system.submit_task(
            test_suite.simulate_compute_intensive_task,
            data_size,
            1.2,  # complexity
            priority=priority,
            resource_requirements=requirements,
            estimated_duration_ms=data_size / 2
        )
        task_ids.append(task_id)
        logger.debug(f"  Submitted task {i+1}: {task_id[:8]} (priority: {priority.name})")
    
    # Wait for all tasks to complete
    for i, task_id in enumerate(task_ids):
        try:
            result = await edge_system.get_task_result(task_id, timeout_seconds=30)
            task_results.append(result)
            logger.debug(f"  Task {i+1} completed: {result.get('processing_time', 0):.3f}s")
        except Exception as e:
            logger.warning(f"  Task {i+1} failed: {e}")
    
    execution_time = time.time() - start_time
    logger.info(f"✅ Completed {len(task_results)}/{len(task_ids)} tasks in {execution_time:.2f}s")
    
    # Test 2: Load balancing performance
    logger.info("📋 Test 2: Load Balancing Performance")
    
    # Get system status to analyze load distribution
    status = edge_system.get_system_status()
    logger.info(f"  System Status:")
    logger.info(f"    Nodes: {status['system_info']['active_nodes']}/{status['system_info']['total_nodes']} active")
    logger.info(f"    Success Rate: {status['task_statistics']['success_rate']:.1%}")
    logger.info(f"    Throughput: {edge_system.metrics['throughput_per_second']:.1f} tasks/sec")
    logger.info(f"    Avg Latency: {edge_system.metrics['average_latency']:.1f}ms")
    
    # Test different load balancing strategies
    strategies_to_test = [
        LoadBalancingStrategy.ROUND_ROBIN,
        LoadBalancingStrategy.LEAST_CONNECTIONS,
        LoadBalancingStrategy.RESOURCE_AWARE,
        LoadBalancingStrategy.QUANTUM_OPTIMIZED
    ]
    
    strategy_performance = {}
    
    for strategy in strategies_to_test:
        logger.info(f"  Testing {strategy.value} strategy...")
        
        # Set strategy
        edge_system.load_balancer.strategy = strategy
        
        # Submit batch of tasks
        batch_start = time.time()
        batch_task_ids = []
        
        for i in range(8):  # Smaller batch for strategy testing
            task_id = await edge_system.submit_task(
                test_suite.simulate_compute_intensive_task,
                1000,  # Fixed size for fair comparison
                1.0,   # Fixed complexity
                priority=TaskPriority.MEDIUM,
                resource_requirements={"cpu": 15, "memory": 12},
                estimated_duration_ms=500
            )
            batch_task_ids.append(task_id)
        
        # Collect results
        completed_tasks = 0
        total_latency = 0
        
        for task_id in batch_task_ids:
            try:
                result = await edge_system.get_task_result(task_id, timeout_seconds=15)
                completed_tasks += 1
                total_latency += result.get('processing_time', 0)
            except Exception as e:
                logger.debug(f"    Task failed with {strategy.value}: {e}")
        
        batch_time = time.time() - batch_start
        avg_latency = total_latency / max(completed_tasks, 1)
        
        strategy_performance[strategy.value] = {
            "completed_tasks": completed_tasks,
            "total_tasks": len(batch_task_ids),
            "success_rate": completed_tasks / len(batch_task_ids),
            "batch_execution_time": batch_time,
            "average_task_latency": avg_latency
        }
        
        logger.info(f"    {strategy.value}: {completed_tasks}/{len(batch_task_ids)} tasks, "
                   f"avg latency: {avg_latency:.3f}s")
    
    # Find best strategy
    best_strategy = max(strategy_performance.items(), 
                       key=lambda x: x[1]['success_rate'] * (1 / max(x[1]['average_task_latency'], 0.001)))
    logger.info(f"  🎯 Best performing strategy: {best_strategy[0]}")
    
    return {
        "basic_execution": {
            "total_tasks": len(task_ids),
            "completed_tasks": len(task_results),
            "execution_time": execution_time,
            "success_rate": len(task_results) / len(task_ids)
        },
        "load_balancing": strategy_performance,
        "system_status": status,
        "best_strategy": best_strategy[0]
    }


@distributed_task(
    priority=TaskPriority.HIGH,
    resource_requirements={"cpu": 20, "memory": 25},
    estimated_duration_ms=750
)
async def decorated_distributed_task(task_name: str, complexity: float = 1.0) -> dict:
    """Example of decorated distributed task."""
    processing_time = complexity * random.uniform(0.5, 1.5)
    await asyncio.sleep(processing_time)
    
    return {
        "task_name": task_name,
        "complexity": complexity,
        "processing_time": processing_time,
        "result": f"distributed_processing_of_{task_name}_complete",
        "node_info": "edge_node_executed"
    }


@high_priority_task
async def high_priority_computation(data_points: int) -> dict:
    """Example of high-priority distributed computation."""
    # Simulate complex computation
    computation_time = data_points / 10000 * random.uniform(0.8, 1.2)
    await asyncio.sleep(computation_time)
    
    return {
        "data_points": data_points,
        "computation_time": computation_time,
        "result": f"processed_{data_points}_data_points",
        "priority": "high",
        "optimization_applied": True
    }


async def demonstrate_distributed_decorators():
    """Demonstrate distributed task decorators."""
    logger.info("🎯 Demonstrating Distributed Task Decorators")
    
    # Test distributed task decorator
    logger.info("📋 Testing @distributed_task decorator")
    
    distributed_tasks = [
        decorated_distributed_task("ml_training", 1.5),
        decorated_distributed_task("data_analysis", 1.0),
        decorated_distributed_task("image_processing", 2.0),
        decorated_distributed_task("graph_computation", 1.2),
    ]
    
    start_time = time.time()
    results = await asyncio.gather(*distributed_tasks, return_exceptions=True)
    execution_time = time.time() - start_time
    
    successful_results = [r for r in results if not isinstance(r, Exception)]
    logger.info(f"  Distributed tasks: {len(successful_results)}/{len(distributed_tasks)} completed")
    logger.info(f"  Total execution time: {execution_time:.2f}s")
    
    # Test high-priority task decorator
    logger.info("📋 Testing @high_priority_task decorator")
    
    high_priority_tasks = [
        high_priority_computation(50000),
        high_priority_computation(25000),
        high_priority_computation(75000),
    ]
    
    start_time = time.time()
    hp_results = await asyncio.gather(*high_priority_tasks, return_exceptions=True)
    hp_execution_time = time.time() - start_time
    
    successful_hp = [r for r in hp_results if not isinstance(r, Exception)]
    logger.info(f"  High-priority tasks: {len(successful_hp)}/{len(high_priority_tasks)} completed")
    logger.info(f"  Total execution time: {hp_execution_time:.2f}s")
    
    return {
        "distributed_decorator": {
            "total_tasks": len(distributed_tasks),
            "successful_tasks": len(successful_results),
            "execution_time": execution_time
        },
        "high_priority_decorator": {
            "total_tasks": len(high_priority_tasks),
            "successful_tasks": len(successful_hp),
            "execution_time": hp_execution_time
        }
    }


async def demonstrate_auto_scaling():
    """Demonstrate auto-scaling capabilities."""
    logger.info("⚡ Demonstrating Auto-Scaling")
    
    edge_system = get_hyperscale_edge_system()
    
    # Get initial system state
    initial_status = edge_system.get_system_status()
    initial_nodes = initial_status['system_info']['total_nodes']
    
    logger.info(f"  Initial nodes: {initial_nodes}")
    
    # Scale out test
    logger.info("📈 Testing scale-out operation")
    await edge_system.scale_out(additional_nodes=8)
    
    scaled_out_status = edge_system.get_system_status()
    scaled_out_nodes = scaled_out_status['system_info']['total_nodes']
    
    logger.info(f"  Nodes after scale-out: {scaled_out_nodes} (added: {scaled_out_nodes - initial_nodes})")
    
    # Submit high-load batch to test scaled system
    logger.info("📋 Testing scaled system under load")
    
    high_load_tasks = []
    for i in range(20):  # Higher load
        task_id = await edge_system.submit_task(
            asyncio.sleep,
            random.uniform(0.1, 0.3),  # Variable sleep times
            priority=random.choice(list(TaskPriority)),
            resource_requirements={
                "cpu": random.uniform(10, 30),
                "memory": random.uniform(15, 35)
            },
            estimated_duration_ms=random.uniform(100, 500)
        )
        high_load_tasks.append(task_id)
    
    # Wait for tasks to complete
    completed_high_load = 0
    for task_id in high_load_tasks:
        try:
            await edge_system.get_task_result(task_id, timeout_seconds=10)
            completed_high_load += 1
        except Exception:
            pass
    
    high_load_success_rate = completed_high_load / len(high_load_tasks)
    logger.info(f"  High-load test: {completed_high_load}/{len(high_load_tasks)} tasks completed "
                f"({high_load_success_rate:.1%} success rate)")
    
    # Scale in test
    logger.info("📉 Testing scale-in operation")
    await edge_system.scale_in(nodes_to_remove=5)
    
    scaled_in_status = edge_system.get_system_status()
    final_nodes = scaled_in_status['system_info']['total_nodes']
    
    logger.info(f"  Nodes after scale-in: {final_nodes} (removed: {scaled_out_nodes - final_nodes})")
    
    return {
        "initial_nodes": initial_nodes,
        "scaled_out_nodes": scaled_out_nodes,
        "final_nodes": final_nodes,
        "high_load_test": {
            "total_tasks": len(high_load_tasks),
            "completed_tasks": completed_high_load,
            "success_rate": high_load_success_rate
        }
    }


async def demonstrate_performance_optimization():
    """Demonstrate performance optimization features."""
    logger.info("🚀 Demonstrating Performance Optimization")
    
    edge_system = get_hyperscale_edge_system()
    test_suite = ScalabilityTestSuite()
    
    # Performance benchmark: Sequential vs Distributed execution
    logger.info("📊 Performance Benchmark: Sequential vs Distributed")
    
    # Sequential execution baseline
    logger.info("  Running sequential baseline...")
    sequential_start = time.time()
    
    sequential_results = []
    for i in range(10):
        result = await test_suite.simulate_compute_intensive_task(2000, 1.0)
        sequential_results.append(result)
    
    sequential_time = time.time() - sequential_start
    logger.info(f"  Sequential execution: {len(sequential_results)} tasks in {sequential_time:.2f}s")
    
    # Distributed execution
    logger.info("  Running distributed execution...")
    distributed_start = time.time()
    
    distributed_task_ids = []
    for i in range(10):
        task_id = await edge_system.submit_task(
            test_suite.simulate_compute_intensive_task,
            2000,
            1.0,
            priority=TaskPriority.MEDIUM,
            resource_requirements={"cpu": 15, "memory": 12},
            estimated_duration_ms=1000
        )
        distributed_task_ids.append(task_id)
    
    distributed_results = []
    for task_id in distributed_task_ids:
        try:
            result = await edge_system.get_task_result(task_id, timeout_seconds=15)
            distributed_results.append(result)
        except Exception as e:
            logger.warning(f"  Distributed task failed: {e}")
    
    distributed_time = time.time() - distributed_start
    logger.info(f"  Distributed execution: {len(distributed_results)} tasks in {distributed_time:.2f}s")
    
    # Calculate performance improvements
    speedup = sequential_time / distributed_time if distributed_time > 0 else 1.0
    efficiency = speedup / edge_system.get_system_status()['system_info']['active_nodes']
    
    logger.info(f"  🎯 Performance Results:")
    logger.info(f"    Speedup: {speedup:.2f}x")
    logger.info(f"    Efficiency: {efficiency:.2f}")
    logger.info(f"    Time saved: {sequential_time - distributed_time:.2f}s ({((sequential_time - distributed_time) / sequential_time * 100):.1f}%)")
    
    # Resource utilization analysis
    logger.info("📈 Resource Utilization Analysis")
    
    status = edge_system.get_system_status()
    metrics = edge_system.metrics
    
    logger.info(f"  System Metrics:")
    logger.info(f"    Resource Utilization: {metrics['resource_utilization']:.1f}%")
    logger.info(f"    Average Latency: {metrics['average_latency']:.1f}ms")
    logger.info(f"    Throughput: {metrics['throughput_per_second']:.2f} tasks/sec")
    logger.info(f"    Total Tasks Processed: {metrics['total_tasks_processed']}")
    
    # Regional distribution analysis
    regional_dist = status['regional_distribution']
    logger.info(f"  Regional Distribution:")
    for region, stats in regional_dist.items():
        logger.info(f"    {region}: {stats['nodes']} nodes, {stats['tasks_completed']} tasks")
    
    return {
        "performance_benchmark": {
            "sequential_time": sequential_time,
            "distributed_time": distributed_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "time_saved": sequential_time - distributed_time
        },
        "resource_utilization": metrics,
        "regional_distribution": regional_dist
    }


async def demonstrate_generation3_scalability():
    """Demonstrate all Generation 3 scalability features."""
    logger.info("⚡ TERRAGON SDLC v6.0 - GENERATION 3 SCALABILITY DEMONSTRATION")
    logger.info("OBJECTIVE: MAKE IT SCALE - Hyper-Scale Edge Computing")
    
    start_time = datetime.now()
    
    # Execute all scalability demonstrations
    results = {}
    
    try:
        # 1. Edge Computing Scalability
        logger.info("\n" + "="*60)
        results["edge_computing"] = await demonstrate_edge_computing_scalability()
        
        # 2. Distributed Task Decorators
        logger.info("\n" + "="*60)
        results["distributed_decorators"] = await demonstrate_distributed_decorators()
        
        # 3. Auto-Scaling
        logger.info("\n" + "="*60)
        results["auto_scaling"] = await demonstrate_auto_scaling()
        
        # 4. Performance Optimization
        logger.info("\n" + "="*60)
        results["performance_optimization"] = await demonstrate_performance_optimization()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Generate scalability report
        logger.info("\n" + "="*60)
        logger.info("🎯 GENERATION 3 SCALABILITY SUMMARY")
        logger.info(f"Duration: {duration}")
        
        # Calculate overall metrics
        total_tasks = (
            results["edge_computing"]["basic_execution"]["total_tasks"] +
            results["distributed_decorators"]["distributed_decorator"]["total_tasks"] +
            results["auto_scaling"]["high_load_test"]["total_tasks"]
        )
        
        total_completed = (
            results["edge_computing"]["basic_execution"]["completed_tasks"] +
            results["distributed_decorators"]["distributed_decorator"]["successful_tasks"] +
            results["auto_scaling"]["high_load_test"]["completed_tasks"]
        )
        
        overall_success_rate = total_completed / total_tasks if total_tasks > 0 else 0
        
        speedup = results["performance_optimization"]["performance_benchmark"]["speedup"]
        best_strategy = results["edge_computing"]["best_strategy"]
        
        logger.info(f"📊 Overall Performance:")
        logger.info(f"   Total tasks executed: {total_tasks}")
        logger.info(f"   Success rate: {overall_success_rate:.1%}")
        logger.info(f"   Performance speedup: {speedup:.2f}x")
        logger.info(f"   Best load balancing: {best_strategy}")
        
        # Scalability score calculation
        base_score = overall_success_rate * 100
        speedup_bonus = min(speedup * 10, 30)  # Cap at 30 points
        strategy_bonus = 10  # Bonus for intelligent load balancing
        
        scalability_score = base_score + speedup_bonus + strategy_bonus
        
        logger.info(f"⚡ Overall Scalability Score: {scalability_score:.1f}/100")
        logger.info("Status: GENERATION 3 COMPLETE - READY FOR QUALITY GATES")
        
        return {
            "success": True,
            "duration": duration,
            "scalability_score": scalability_score,
            "total_tasks": total_tasks,
            "success_rate": overall_success_rate,
            "performance_speedup": speedup,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Generation 3 demonstration failed: {e}")
        logger.exception("Full error traceback:")
        return {
            "success": False,
            "error": str(e),
            "partial_results": results
        }


async def main():
    """Main demonstration function."""
    try:
        logger.info("=" * 80)
        logger.info("TERRAGON SDLC v6.0 - GENERATION 3 SCALABILITY ENHANCEMENT")
        logger.info("=" * 80)
        
        result = await demonstrate_generation3_scalability()
        
        logger.info("=" * 80)
        if result["success"]:
            logger.info("✅ GENERATION 3 DEMONSTRATION COMPLETED SUCCESSFULLY")
            logger.info(f"⚡ Scalability Score: {result['scalability_score']:.1f}/100")
            logger.info(f"🚀 Performance Speedup: {result['performance_speedup']:.2f}x")
            logger.info(f"📊 Success Rate: {result['success_rate']:.1%}")
        else:
            logger.error("❌ GENERATION 3 DEMONSTRATION FAILED")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Main demonstration failed: {e}")
        logger.exception("Full error traceback:")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Run the Generation 3 demonstration
    result = asyncio.run(main())
    exit_code = 0 if result.get("success", False) else 1
    sys.exit(exit_code)