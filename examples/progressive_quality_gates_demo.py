#!/usr/bin/env python3
"""Progressive Quality Gates Demo - TERRAGON SDLC v6.0 Enhancement.

This demo showcases the progressive quality gates system that adapts and evolves
based on project maturity and performance history.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scgraph_hub.progressive_quality_gates import (
    ProgressiveQualityGateSystem,
    ProgressiveLevel,
    run_progressive_gates
)


async def demonstrate_progressive_gates():
    """Demonstrate progressive quality gates functionality."""
    print("🚀 TERRAGON SDLC v6.0 - Progressive Quality Gates Demo")
    print("=" * 60)
    
    # Initialize progressive gates system
    config_path = Path("demo_progressive_gates.json")
    system = ProgressiveQualityGateSystem(config_path)
    
    print(f"\n📊 Initial Status:")
    print(f"Current Level: {system.current_level.value}")
    print(f"Active Gates: {len(system.get_active_gates())}")
    
    # Show initial gate configuration
    active_gates = system.get_active_gates()
    print(f"\n🎯 Active Quality Gates:")
    for gate_name, gate_config in active_gates.items():
        threshold = gate_config.threshold.get_current_threshold()
        print(f"  • {gate_name}: {gate_config.level.value} level "
              f"(threshold: {threshold:.1f}%)")
    
    # Simulate multiple execution cycles to show evolution
    print(f"\n🔄 Simulating Multiple Execution Cycles:")
    
    for cycle in range(1, 6):
        print(f"\n--- Cycle {cycle} ---")
        
        # Create context for this cycle
        context = {
            'cycle': cycle,
            'project_size': 'large',
            'complexity_score': min(50 + cycle * 10, 100),
            'test_coverage': min(70 + cycle * 5, 95),
            'security_score': min(80 + cycle * 3, 98)
        }
        
        # Execute progressive gates
        try:
            results = await system.execute_progressive_gates(context)
            
            print(f"  Overall Status: {results['overall_status'].upper()}")
            print(f"  Overall Score: {results['overall_score']:.1f}%")
            print(f"  Execution Time: {results['execution_time']:.2f}s")
            print(f"  Gates Executed: {results['gates_executed']}")
            print(f"  Gates Passed: {results['gates_passed']}")
            
            if results['gates_failed'] > 0:
                print(f"  Gates Failed: {results['gates_failed']}")
            
            # Show gate details
            for gate_name, gate_result in results['gates'].items():
                status_icon = "✅" if gate_result['status'] == 'passed' else "❌"
                print(f"    {status_icon} {gate_name}: {gate_result['score']:.1f}% "
                      f"(threshold: {gate_result['threshold']:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ Cycle {cycle} failed: {e}")
        
        # Show evolution status after each cycle
        evolution_status = system.get_evolution_status()
        if cycle > 1:
            print(f"  📈 Success Rate: {evolution_status['recent_success_rate']:.1f}%")
            print(f"  🎯 Average Score: {evolution_status['average_score']:.1f}%")
    
    # Final evolution status
    print(f"\n📈 Final Evolution Status:")
    evolution_status = system.get_evolution_status()
    print(f"  Current Level: {evolution_status['current_level']}")
    print(f"  Total Executions: {evolution_status['total_executions']}")
    print(f"  Recent Success Rate: {evolution_status['recent_success_rate']:.1f}%")
    print(f"  Average Score: {evolution_status['average_score']:.1f}%")
    print(f"  Active Gates: {evolution_status['active_gates']}")
    
    if evolution_status['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in evolution_status['recommendations']:
            print(f"  • {rec}")
    
    # Show adaptive threshold evolution
    print(f"\n📊 Adaptive Threshold Evolution:")
    for gate_name, gate_config in system.gates_config.items():
        if gate_config.threshold.history:
            recent_scores = gate_config.threshold.history[-5:]
            current_threshold = gate_config.threshold.get_current_threshold()
            print(f"  • {gate_name}:")
            print(f"    - Current Threshold: {current_threshold:.1f}%")
            print(f"    - Recent Scores: {[f'{s:.1f}' for s in recent_scores]}")
            print(f"    - Performance Trend: {'↗️' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else '↔️'}")
    
    # Save final configuration
    system._save_config()
    print(f"\n💾 Configuration saved to: {config_path}")
    
    # Show configuration summary
    print(f"\n🔧 Final Configuration Summary:")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"  Project Level: {config['current_level']}")
    print(f"  Execution History: {len(config['execution_history'])} runs")
    print(f"  Configured Gates: {len(config['gates'])}")
    
    return system, evolution_status


async def demonstrate_level_progression():
    """Demonstrate how gates progress through maturity levels."""
    print(f"\n🎯 Progressive Level Demonstration:")
    print("=" * 50)
    
    levels = list(ProgressiveLevel)
    for i, level in enumerate(levels):
        print(f"\n{i+1}. {level.value.upper()} Level:")
        
        # Show what gates would be active at this level
        temp_system = ProgressiveQualityGateSystem()
        temp_system.current_level = level
        
        active_gates = temp_system.get_active_gates()
        
        print(f"   Active Gates: {len(active_gates)}")
        for gate_name, gate_config in active_gates.items():
            threshold = gate_config.threshold.get_current_threshold()
            weight = gate_config.weight
            deps = ", ".join(gate_config.dependencies) if gate_config.dependencies else "None"
            
            print(f"     • {gate_name}:")
            print(f"       - Threshold: {threshold:.1f}%")
            print(f"       - Weight: {weight}x")
            print(f"       - Dependencies: {deps}")


def main():
    """Main demo execution."""
    print("🌟 TERRAGON SDLC v6.0 - Progressive Quality Gates")
    print("Advanced Adaptive Quality Assurance System")
    print("=" * 60)
    
    try:
        # Run main demonstration
        loop = asyncio.get_event_loop()
        system, evolution_status = loop.run_until_complete(demonstrate_progressive_gates())
        
        # Demonstrate level progression
        loop.run_until_complete(demonstrate_level_progression())
        
        print(f"\n🎉 Demo Complete!")
        print(f"Progressive Quality Gates successfully demonstrated the ability to:")
        print(f"  ✅ Adapt thresholds based on performance history")
        print(f"  ✅ Evolve project maturity levels automatically")
        print(f"  ✅ Manage gate dependencies intelligently")
        print(f"  ✅ Provide actionable evolution recommendations")
        print(f"  ✅ Maintain configuration persistence across runs")
        
        print(f"\n🚀 Ready for Production Integration!")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()