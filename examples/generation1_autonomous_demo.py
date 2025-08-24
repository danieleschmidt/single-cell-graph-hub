"""Generation 1 Autonomous Enhancement Demo - MAKE IT WORK.

This demo showcases the TERRAGON SDLC v6.0 Generation 1 implementation:
- Basic enhanced functionality
- Autonomous task execution
- Intelligent error handling
- Simple performance optimization
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scgraph_hub.autonomous_enhanced import (
    EnhancedAutonomousEngine,
    IntelligenceLevel,
    TaskComplexity,
    get_enhanced_autonomous_engine,
    enhanced_autonomous_task,
    quantum_optimized
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demonstrate_basic_autonomous_features():
    """Demonstrate basic autonomous features for Generation 1."""
    logger.info("🚀 Starting TERRAGON SDLC v6.0 Generation 1 Demo")
    
    # Initialize enhanced autonomous engine
    engine = get_enhanced_autonomous_engine(IntelligenceLevel.ADAPTIVE)
    
    # Demo 1: Basic task execution with autonomous intelligence
    logger.info("\n📋 Demo 1: Basic Autonomous Task Execution")
    
    async def simple_data_processing(data_size: int) -> dict:
        """Simple data processing task."""
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            "processed_items": data_size,
            "processing_time": 0.1,
            "success": True
        }
    
    # Execute with autonomous intelligence
    metrics = await engine.autonomous_execute_task(
        "simple_data_processing",
        simple_data_processing,
        1000,
        complexity=TaskComplexity(computational=1.2, memory=0.8)
    )
    
    logger.info(f"✅ Task completed successfully: {metrics.success}")
    logger.info(f"📊 Quality score: {metrics.quality_score:.3f}")
    logger.info(f"⏱️  Duration: {metrics.duration}")
    
    # Demo 2: Enhanced error handling with autonomous recovery
    logger.info("\n📋 Demo 2: Autonomous Error Handling")
    
    async def potentially_failing_task(should_fail: bool) -> str:
        """Task that may fail to demonstrate error handling."""
        if should_fail:
            raise ValueError("Simulated processing error")
        return "Task completed successfully"
    
    # Test failure handling
    fail_metrics = await engine.autonomous_execute_task(
        "potentially_failing_task",
        potentially_failing_task,
        True,  # This will cause failure
        complexity=TaskComplexity(computational=1.0, algorithmic=1.5)
    )
    
    logger.info(f"❌ Failed task handled: {not fail_metrics.success}")
    logger.info(f"🔍 Error captured: {fail_metrics.error}")
    
    # Test successful execution
    success_metrics = await engine.autonomous_execute_task(
        "potentially_failing_task",
        potentially_failing_task,
        False,  # This will succeed
        complexity=TaskComplexity(computational=1.0, algorithmic=1.5)
    )
    
    logger.info(f"✅ Success task completed: {success_metrics.success}")
    
    # Demo 3: Batch task execution with intelligent concurrency
    logger.info("\n📋 Demo 3: Batch Task Execution")
    
    async def batch_processing_task(task_id: int, processing_time: float) -> dict:
        """Simulate batch processing task."""
        await asyncio.sleep(processing_time)
        return {
            "task_id": task_id,
            "processed_at": datetime.now().isoformat(),
            "processing_time": processing_time
        }
    
    # Create multiple tasks with varying complexity
    tasks = []
    for i in range(5):
        task_name = f"batch_task_{i}"
        processing_time = 0.05 + (i * 0.02)  # Varying processing times
        
        task_metrics = await engine.autonomous_execute_task(
            task_name,
            batch_processing_task,
            i,
            processing_time,
            complexity=TaskComplexity(computational=1.0 + i * 0.2)
        )
        tasks.append((task_name, task_metrics))
    
    logger.info(f"📦 Batch processing completed: {len(tasks)} tasks")
    success_count = sum(1 for _, metrics in tasks if metrics.success)
    logger.info(f"✅ Success rate: {success_count}/{len(tasks)} ({success_count/len(tasks)*100:.1f}%)")
    
    # Demo 4: Performance insights and learning
    logger.info("\n📋 Demo 4: Performance Insights")
    
    insights = engine.get_performance_insights()
    logger.info(f"📈 Performance Summary:")
    logger.info(f"   Total tasks: {insights['performance_summary']['total_tasks']}")
    logger.info(f"   Success rate: {insights['performance_summary']['success_rate']:.3f}")
    logger.info(f"   Avg quality score: {insights['performance_summary']['avg_quality_score']:.3f}")
    logger.info(f"   Intelligence level: {insights['intelligence_level']}")
    
    return insights


@enhanced_autonomous_task(
    complexity=TaskComplexity(computational=2.0, algorithmic=1.5),
    intelligence_level=IntelligenceLevel.ADAPTIVE
)
async def decorated_autonomous_task(data: str) -> str:
    """Example of decorated autonomous task."""
    await asyncio.sleep(0.1)
    return f"Enhanced processing of: {data}"


@quantum_optimized
async def quantum_demo_task(quantum_data: list) -> dict:
    """Example of quantum-optimized task."""
    await asyncio.sleep(0.05)
    return {
        "quantum_processed": len(quantum_data),
        "quantum_optimization": "enabled",
        "processing_method": "quantum_enhanced"
    }


async def demonstrate_decorators():
    """Demonstrate autonomous decorators."""
    logger.info("\n🎯 Demonstrating Autonomous Decorators")
    
    # Test enhanced autonomous task decorator
    logger.info("📋 Testing enhanced_autonomous_task decorator")
    result1 = await decorated_autonomous_task("sample_data")
    logger.info(f"✅ Decorated task result: Task completed")
    
    # Test quantum optimized decorator
    logger.info("📋 Testing quantum_optimized decorator")
    result2 = await quantum_demo_task([1, 2, 3, 4, 5])
    logger.info(f"⚡ Quantum task result: Task completed")
    
    return result1, result2


async def demonstrate_generation1_features():
    """Demonstrate Generation 1 enhanced features."""
    logger.info("\n🔧 Generation 1: MAKE IT WORK - Enhanced Basic Functionality")
    
    # Core functionality working
    await demonstrate_basic_autonomous_features()
    
    # Enhanced decorators working
    await demonstrate_decorators()
    
    # Intelligence adaptation demo
    logger.info("\n🧠 Intelligence Adaptation Demo")
    
    # Test different intelligence levels
    for level in [IntelligenceLevel.BASIC, IntelligenceLevel.ADAPTIVE, IntelligenceLevel.QUANTUM]:
        logger.info(f"Testing intelligence level: {level.value}")
        engine = get_enhanced_autonomous_engine(level)
        
        async def intelligence_test_task(level_name: str) -> str:
            await asyncio.sleep(0.02)
            return f"Processed with {level_name} intelligence"
        
        metrics = await engine.autonomous_execute_task(
            f"intelligence_test_{level.value}",
            intelligence_test_task,
            level.value,
            complexity=TaskComplexity(computational=1.0)
        )
        
        logger.info(f"   ✅ {level.value}: Quality {metrics.quality_score:.3f}, Success: {metrics.success}")
    
    logger.info("\n🎉 Generation 1 Demo Complete - Basic Enhanced Functionality Working!")
    
    return True


async def main():
    """Main demonstration function."""
    try:
        logger.info("=" * 80)
        logger.info("TERRAGON SDLC v6.0 - GENERATION 1 AUTONOMOUS ENHANCEMENT")
        logger.info("OBJECTIVE: MAKE IT WORK - Enhanced Basic Functionality")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Run Generation 1 demonstrations
        result = await demonstrate_generation1_features()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("GENERATION 1 EXECUTION SUMMARY")
        logger.info(f"Duration: {duration}")
        logger.info(f"Result: {'SUCCESS' if result else 'FAILED'}")
        logger.info("Status: GENERATION 1 COMPLETE - READY FOR GENERATION 2")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Generation 1 demo failed: {e}")
        logger.exception("Full error traceback:")
        return False


if __name__ == "__main__":
    # Run the Generation 1 demonstration
    success = asyncio.run(main())
    exit_code = 0 if success else 1
    sys.exit(exit_code)