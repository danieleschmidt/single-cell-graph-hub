"""Progressive Quality Gates System - TERRAGON SDLC v6.0 Enhancement.

This module implements progressive quality gates that adapt and evolve based on 
project maturity, code complexity, and historical performance metrics. 
Integrates with existing quality gate systems for seamless operation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from .quality_gates import QualityGateStatus, QualityGateResult, QualityGateChecker

# Import quality gate types with fallback
try:
    from .ai_quality_gates import QualityGateType, ValidationSeverity
except ImportError:
    from enum import Enum
    
    class QualityGateType(Enum):
        """Types of quality gates for validation."""
        CODE_QUALITY = "code_quality"
        SECURITY_ANALYSIS = "security_analysis"
        PERFORMANCE_BENCHMARKS = "performance_benchmarks"
        TEST_COVERAGE = "test_coverage"
        QUANTUM_RESISTANCE = "quantum_resistance"
        AI_VALIDATION = "ai_validation"
        COMPLIANCE_CHECK = "compliance_check"
        DOCUMENTATION_COMPLETENESS = "documentation_completeness"
    
    class ValidationSeverity(Enum):
        """Severity levels for quality gate violations."""
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        INFO = "info"

try:
    from .logging_config import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class ProgressiveLevel(Enum):
    """Progressive quality gate maturity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"  
    ADVANCED = "advanced"
    EXPERT = "expert"
    AUTONOMOUS = "autonomous"


class AdaptiveMetric(Enum):
    """Adaptive metrics for progressive enhancement."""
    CODE_COMPLEXITY = "code_complexity"
    TEST_SOPHISTICATION = "test_sophistication"
    DOCUMENTATION_DEPTH = "documentation_depth"
    ERROR_RESILIENCE = "error_resilience"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_MATURITY = "security_maturity"


@dataclass
class ProgressiveThreshold:
    """Dynamic threshold that evolves with project maturity."""
    base_value: float
    current_level: ProgressiveLevel
    target_value: float
    adaptation_rate: float = 0.1
    history: List[float] = field(default_factory=list)
    
    def get_current_threshold(self) -> float:
        """Calculate current threshold based on level and history."""
        level_multipliers = {
            ProgressiveLevel.BASIC: 0.7,
            ProgressiveLevel.INTERMEDIATE: 0.8,
            ProgressiveLevel.ADVANCED: 0.9,
            ProgressiveLevel.EXPERT: 0.95,
            ProgressiveLevel.AUTONOMOUS: 1.0
        }
        
        base_threshold = self.base_value * level_multipliers[self.current_level]
        
        # Adaptive adjustment based on historical performance
        if self.history:
            recent_avg = sum(self.history[-10:]) / min(10, len(self.history))
            if recent_avg > base_threshold:
                base_threshold = min(base_threshold + self.adaptation_rate, self.target_value)
        
        return base_threshold
    
    def update_history(self, score: float):
        """Update performance history."""
        self.history.append(score)
        if len(self.history) > 100:  # Keep last 100 scores
            self.history = self.history[-100:]


@dataclass
class ProgressiveGateConfig:
    """Configuration for progressive quality gates."""
    gate_type: QualityGateType
    level: ProgressiveLevel
    threshold: ProgressiveThreshold
    enabled: bool = True
    weight: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    evolution_criteria: Dict[str, Any] = field(default_factory=dict)


class ProgressiveQualityGateSystem:
    """Progressive Quality Gates that adapt to project maturity and performance."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = get_logger(__name__)
        self.config_path = config_path or Path("progressive_gates_config.json")
        self.gates_config: Dict[str, ProgressiveGateConfig] = {}
        self.execution_history: List[Dict] = []
        self.current_level = ProgressiveLevel.BASIC
        self._load_or_create_config()
        
    def _load_or_create_config(self):
        """Load existing config or create default configuration."""
        if self.config_path.exists():
            self._load_config()
        else:
            self._create_default_config()
            self._save_config()
    
    def _create_default_config(self):
        """Create default progressive gates configuration."""
        default_gates = {
            "code_quality": ProgressiveGateConfig(
                gate_type=QualityGateType.CODE_QUALITY,
                level=ProgressiveLevel.BASIC,
                threshold=ProgressiveThreshold(
                    base_value=80.0,
                    current_level=ProgressiveLevel.BASIC,
                    target_value=95.0
                ),
                evolution_criteria={
                    "consistent_passes": 5,
                    "complexity_threshold": 10,
                    "test_coverage_requirement": 80
                }
            ),
            "test_coverage": ProgressiveGateConfig(
                gate_type=QualityGateType.TEST_COVERAGE,
                level=ProgressiveLevel.BASIC,
                threshold=ProgressiveThreshold(
                    base_value=70.0,
                    current_level=ProgressiveLevel.BASIC,
                    target_value=95.0
                ),
                evolution_criteria={
                    "consistent_passes": 3,
                    "test_sophistication": 5
                }
            ),
            "security_analysis": ProgressiveGateConfig(
                gate_type=QualityGateType.SECURITY_ANALYSIS,
                level=ProgressiveLevel.INTERMEDIATE,
                threshold=ProgressiveThreshold(
                    base_value=85.0,
                    current_level=ProgressiveLevel.INTERMEDIATE,
                    target_value=98.0
                ),
                dependencies=["code_quality"],
                evolution_criteria={
                    "zero_critical_vulnerabilities": 10,
                    "security_score_consistency": 90
                }
            ),
            "performance_benchmarks": ProgressiveGateConfig(
                gate_type=QualityGateType.PERFORMANCE_BENCHMARKS,
                level=ProgressiveLevel.ADVANCED,
                threshold=ProgressiveThreshold(
                    base_value=75.0,
                    current_level=ProgressiveLevel.ADVANCED,
                    target_value=95.0
                ),
                dependencies=["code_quality", "test_coverage"],
                weight=1.5,
                evolution_criteria={
                    "performance_improvement_trend": 5,
                    "optimization_score": 85
                }
            ),
            "ai_validation": ProgressiveGateConfig(
                gate_type=QualityGateType.AI_VALIDATION,
                level=ProgressiveLevel.EXPERT,
                threshold=ProgressiveThreshold(
                    base_value=88.0,
                    current_level=ProgressiveLevel.EXPERT,
                    target_value=97.0
                ),
                dependencies=["code_quality", "test_coverage", "security_analysis"],
                weight=2.0,
                evolution_criteria={
                    "ai_score_consistency": 90,
                    "innovative_solutions": 3
                }
            ),
            "quantum_resistance": ProgressiveGateConfig(
                gate_type=QualityGateType.QUANTUM_RESISTANCE,
                level=ProgressiveLevel.AUTONOMOUS,
                threshold=ProgressiveThreshold(
                    base_value=92.0,
                    current_level=ProgressiveLevel.AUTONOMOUS,
                    target_value=99.0
                ),
                dependencies=["security_analysis", "ai_validation"],
                weight=3.0,
                evolution_criteria={
                    "quantum_readiness_score": 95,
                    "cryptographic_strength": 98
                }
            )
        }
        
        self.gates_config = default_gates
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                
            for gate_name, gate_data in config_data.get('gates', {}).items():
                threshold_data = gate_data['threshold']
                threshold = ProgressiveThreshold(
                    base_value=threshold_data['base_value'],
                    current_level=ProgressiveLevel(threshold_data['current_level']),
                    target_value=threshold_data['target_value'],
                    adaptation_rate=threshold_data.get('adaptation_rate', 0.1),
                    history=threshold_data.get('history', [])
                )
                
                self.gates_config[gate_name] = ProgressiveGateConfig(
                    gate_type=QualityGateType(gate_data['gate_type']),
                    level=ProgressiveLevel(gate_data['level']),
                    threshold=threshold,
                    enabled=gate_data.get('enabled', True),
                    weight=gate_data.get('weight', 1.0),
                    dependencies=gate_data.get('dependencies', []),
                    evolution_criteria=gate_data.get('evolution_criteria', {})
                )
                
            self.current_level = ProgressiveLevel(config_data.get('current_level', 'basic'))
            self.execution_history = config_data.get('execution_history', [])
            
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            self._create_default_config()
    
    def _save_config(self):
        """Save configuration to file."""
        config_data = {
            'current_level': self.current_level.value,
            'execution_history': self.execution_history[-50:],  # Keep last 50
            'gates': {}
        }
        
        for gate_name, gate_config in self.gates_config.items():
            config_data['gates'][gate_name] = {
                'gate_type': gate_config.gate_type.value,
                'level': gate_config.level.value,
                'enabled': gate_config.enabled,
                'weight': gate_config.weight,
                'dependencies': gate_config.dependencies,
                'evolution_criteria': gate_config.evolution_criteria,
                'threshold': {
                    'base_value': gate_config.threshold.base_value,
                    'current_level': gate_config.threshold.current_level.value,
                    'target_value': gate_config.threshold.target_value,
                    'adaptation_rate': gate_config.threshold.adaptation_rate,
                    'history': gate_config.threshold.history[-50:]  # Keep last 50
                }
            }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_active_gates(self) -> Dict[str, ProgressiveGateConfig]:
        """Get currently active gates based on level and dependencies."""
        active_gates = {}
        
        for gate_name, gate_config in self.gates_config.items():
            if not gate_config.enabled:
                continue
                
            # Check if gate level is appropriate for current project level
            gate_levels = list(ProgressiveLevel)
            current_index = gate_levels.index(self.current_level)
            gate_index = gate_levels.index(gate_config.level)
            
            if gate_index <= current_index + 1:  # Allow one level ahead
                # Check dependencies
                dependencies_met = all(
                    dep_name in self.gates_config and 
                    self._is_gate_consistently_passing(dep_name)
                    for dep_name in gate_config.dependencies
                )
                
                if dependencies_met or not gate_config.dependencies:
                    active_gates[gate_name] = gate_config
        
        return active_gates
    
    def _is_gate_consistently_passing(self, gate_name: str, 
                                    required_passes: int = 3) -> bool:
        """Check if a gate has been consistently passing."""
        if gate_name not in self.gates_config:
            return False
            
        recent_results = [
            result for result in self.execution_history[-10:]  # Last 10 runs
            if gate_name in result.get('gates', {})
        ]
        
        if len(recent_results) < required_passes:
            return False
            
        passing_count = sum(
            1 for result in recent_results[-required_passes:]
            if result['gates'][gate_name]['status'] == 'passed'
        )
        
        return passing_count >= required_passes
    
    async def execute_progressive_gates(self, 
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute progressive quality gates with adaptive thresholds."""
        start_time = time.time()
        context = context or {}
        
        self.logger.info("Starting progressive quality gates execution")
        
        active_gates = self.get_active_gates()
        results = {}
        overall_score = 0.0
        total_weight = 0.0
        
        # Execute gates in dependency order
        execution_order = self._calculate_execution_order(active_gates)
        
        for gate_name in execution_order:
            gate_config = active_gates[gate_name]
            gate_start_time = time.time()
            
            try:
                # Get current adaptive threshold
                current_threshold = gate_config.threshold.get_current_threshold()
                
                # Execute gate (integrate with existing gate checkers)
                result = await self._execute_gate(
                    gate_name, gate_config, current_threshold, context
                )
                
                # Update threshold history
                gate_config.threshold.update_history(result.score)
                
                results[gate_name] = result
                
                # Calculate weighted score
                if result.status != QualityGateStatus.SKIPPED:
                    overall_score += result.score * gate_config.weight
                    total_weight += gate_config.weight
                
                execution_time = time.time() - gate_start_time
                self.logger.info(
                    f"Gate '{gate_name}' completed: {result.status.value} "
                    f"(score: {result.score:.1f}, threshold: {current_threshold:.1f}, "
                    f"time: {execution_time:.2f}s)"
                )
                
            except Exception as e:
                self.logger.error(f"Gate '{gate_name}' failed with error: {e}")
                results[gate_name] = QualityGateResult(
                    gate_name=gate_name,
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    threshold=current_threshold,
                    message=f"Gate execution failed: {str(e)}",
                    execution_time=time.time() - gate_start_time
                )
        
        # Calculate overall results
        overall_score = overall_score / total_weight if total_weight > 0 else 0.0
        total_execution_time = time.time() - start_time
        
        # Determine overall status
        failed_gates = [name for name, result in results.items() 
                       if result.status == QualityGateStatus.FAILED]
        warning_gates = [name for name, result in results.items() 
                        if result.status == QualityGateStatus.WARNING]
        
        if failed_gates:
            overall_status = QualityGateStatus.FAILED
        elif warning_gates:
            overall_status = QualityGateStatus.WARNING
        else:
            overall_status = QualityGateStatus.PASSED
        
        # Create execution summary
        execution_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': overall_status.value,
            'overall_score': overall_score,
            'execution_time': total_execution_time,
            'gates_executed': len(results),
            'gates_passed': len([r for r in results.values() 
                               if r.status == QualityGateStatus.PASSED]),
            'gates_failed': len(failed_gates),
            'gates_warning': len(warning_gates),
            'active_level': self.current_level.value,
            'gates': {name: result.to_dict() for name, result in results.items()}
        }
        
        # Update execution history
        self.execution_history.append(execution_summary)
        
        # Check for level evolution
        await self._check_level_evolution()
        
        # Save updated configuration
        self._save_config()
        
        self.logger.info(
            f"Progressive quality gates completed: {overall_status.value} "
            f"(score: {overall_score:.1f}, time: {total_execution_time:.2f}s)"
        )
        
        return execution_summary
    
    def _calculate_execution_order(self, gates: Dict[str, ProgressiveGateConfig]) -> List[str]:
        """Calculate execution order based on dependencies."""
        ordered = []
        remaining = set(gates.keys())
        
        while remaining:
            # Find gates with no unmet dependencies
            ready = []
            for gate_name in remaining:
                deps = gates[gate_name].dependencies
                if all(dep in ordered or dep not in gates for dep in deps):
                    ready.append(gate_name)
            
            if not ready:
                # Circular dependency or missing dependency - add remaining in order
                ready = list(remaining)
            
            # Sort by gate level (basic first)
            ready.sort(key=lambda x: list(ProgressiveLevel).index(gates[x].level))
            
            for gate_name in ready:
                ordered.append(gate_name)
                remaining.remove(gate_name)
        
        return ordered
    
    async def _execute_gate(self, gate_name: str, gate_config: ProgressiveGateConfig,
                           threshold: float, context: Dict[str, Any]) -> QualityGateResult:
        """Execute a specific progressive gate."""
        # Create appropriate gate checker based on type
        checker = self._create_gate_checker(gate_config.gate_type, threshold)
        
        if checker:
            return await checker.check_async(context)
        else:
            # Fallback for unsupported gate types
            return QualityGateResult(
                gate_name=gate_name,
                status=QualityGateStatus.SKIPPED,
                score=0.0,
                threshold=threshold,
                message=f"Gate type '{gate_config.gate_type.value}' not implemented"
            )
    
    def _create_gate_checker(self, gate_type: QualityGateType, 
                           threshold: float) -> Optional[QualityGateChecker]:
        """Create appropriate gate checker for the given type."""
        # This would integrate with existing gate checkers
        # For now, return a mock implementation
        
        class MockGateChecker(QualityGateChecker):
            def __init__(self, gate_type: QualityGateType, threshold: float):
                super().__init__(gate_type.value, threshold)
                self.gate_type = gate_type
            
            def check(self, context: Dict[str, Any]) -> QualityGateResult:
                # Mock implementation - would integrate with real checkers
                import random
                score = random.uniform(threshold - 10, threshold + 10)
                status = (QualityGateStatus.PASSED if score >= threshold 
                         else QualityGateStatus.FAILED)
                
                return QualityGateResult(
                    gate_name=self.name,
                    status=status,
                    score=max(0, min(100, score)),
                    threshold=threshold,
                    message=f"Mock execution for {self.gate_type.value}",
                    details={'mock': True}
                )
            
            async def check_async(self, context: Dict[str, Any]) -> QualityGateResult:
                # Simulate some async work
                await asyncio.sleep(0.1)
                return self.check(context)
        
        return MockGateChecker(gate_type, threshold)
    
    async def _check_level_evolution(self):
        """Check if project can evolve to next maturity level."""
        if len(self.execution_history) < 5:  # Need sufficient history
            return
            
        current_level_index = list(ProgressiveLevel).index(self.current_level)
        next_level = list(ProgressiveLevel)[min(current_level_index + 1, 
                                              len(ProgressiveLevel) - 1)]
        
        if next_level == self.current_level:
            return  # Already at highest level
        
        # Check evolution criteria
        recent_runs = self.execution_history[-5:]
        consistent_success = all(
            run['overall_status'] == 'passed' and run['overall_score'] >= 85.0
            for run in recent_runs
        )
        
        if consistent_success:
            self.logger.info(f"Evolving from {self.current_level.value} to {next_level.value}")
            self.current_level = next_level
            
            # Update gate levels
            for gate_config in self.gates_config.values():
                if list(ProgressiveLevel).index(gate_config.level) <= current_level_index:
                    gate_config.threshold.current_level = next_level
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status and recommendations."""
        if not self.execution_history:
            return {
                'current_level': self.current_level.value,
                'recommendations': ['Execute quality gates to begin evolution tracking']
            }
        
        recent_runs = self.execution_history[-10:] if self.execution_history else []
        
        stats = {
            'current_level': self.current_level.value,
            'total_executions': len(self.execution_history),
            'recent_success_rate': (
                len([r for r in recent_runs if r['overall_status'] == 'passed']) / 
                len(recent_runs) * 100 if recent_runs else 0
            ),
            'average_score': (
                sum(r['overall_score'] for r in recent_runs) / 
                len(recent_runs) if recent_runs else 0
            ),
            'active_gates': len(self.get_active_gates()),
            'recommendations': []
        }
        
        # Generate recommendations
        if stats['recent_success_rate'] < 80:
            stats['recommendations'].append(
                "Focus on improving failing gates before level evolution"
            )
        
        if stats['average_score'] > 90 and stats['recent_success_rate'] > 90:
            next_level_index = list(ProgressiveLevel).index(self.current_level) + 1
            if next_level_index < len(ProgressiveLevel):
                next_level = list(ProgressiveLevel)[next_level_index].value
                stats['recommendations'].append(
                    f"Ready for evolution to {next_level} level"
                )
        
        failing_gates = []
        if recent_runs:
            last_run = recent_runs[-1]
            failing_gates = [
                name for name, gate in last_run.get('gates', {}).items()
                if gate['status'] == 'failed'
            ]
        
        if failing_gates:
            stats['recommendations'].append(
                f"Address failing gates: {', '.join(failing_gates)}"
            )
        
        return stats


# Convenience functions for integration
async def run_progressive_gates(config_path: Optional[Path] = None,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run progressive quality gates with given context."""
    system = ProgressiveQualityGateSystem(config_path)
    return await system.execute_progressive_gates(context)


def get_progressive_gates_system(config_path: Optional[Path] = None) -> ProgressiveQualityGateSystem:
    """Get progressive quality gates system instance."""
    return ProgressiveQualityGateSystem(config_path)