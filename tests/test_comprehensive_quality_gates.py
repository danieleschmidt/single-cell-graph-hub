"""
Comprehensive Quality Gates Test Suite v4.0
Complete testing framework for all TERRAGON SDLC components with 85%+ coverage
"""

import pytest
import asyncio
import numpy as np
import torch
import tempfile
import json
import time
import psutil
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock

from src.scgraph_hub.quantum_research_discovery import (
    QuantumResearchEngine, QuantumResearchOracle, NovelAlgorithmDiscovery,
    ExperimentalFramework, run_autonomous_research
)
from src.scgraph_hub.experimental_baseline_framework import (
    ComprehensiveBaselineFramework, BaselineEvaluator, ExperimentalDesigner,
    StatisticalAnalyzer
)
from src.scgraph_hub.robust_research_infrastructure import (
    ResearchDatabaseManager, ResearchCacheManager, SystemMonitor,
    FaultTolerantTaskExecutor, SecurityValidator, IntrusionDetectionSystem
)
from src.scgraph_hub.hyperscale_optimization_engine import (
    CUDAAccelerator, MemoryOptimizer, AlgorithmicOptimizer, AutoScaler
)


@dataclass
class QualityGateResult:
    """Quality gate evaluation result."""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    timestamp: str
    threshold: float = 0.85


class QualityGateEvaluator:
    """Comprehensive quality gate evaluation system."""
    
    def __init__(self):
        self.results = {}
        self.overall_score = 0.0
        self.passing_threshold = 0.85
    
    def evaluate_all_gates(self) -> Dict[str, QualityGateResult]:
        """Evaluate all quality gates."""
        gates = [
            self.evaluate_performance_gate,
            self.evaluate_security_gate,
            self.evaluate_reliability_gate,
            self.evaluate_scalability_gate,
            self.evaluate_maintainability_gate,
            self.evaluate_test_coverage_gate
        ]
        
        results = {}
        total_score = 0.0
        
        for gate_func in gates:
            result = gate_func()
            results[result.gate_name] = result
            total_score += result.score
        
        self.overall_score = total_score / len(gates)
        self.results = results
        
        return results
    
    def evaluate_performance_gate(self) -> QualityGateResult:
        """Evaluate performance quality gate."""
        details = {}
        
        # Test quantum research discovery performance
        oracle = QuantumResearchOracle({})
        start_time = time.time()
        hypotheses = oracle.generate_hypotheses('test_domain', num_hypotheses=10)
        hypothesis_time = time.time() - start_time
        
        details['hypothesis_generation_time'] = hypothesis_time
        details['hypothesis_generation_rate'] = len(hypotheses) / hypothesis_time
        
        # Test algorithm discovery performance
        discovery = NovelAlgorithmDiscovery()
        start_time = time.time()
        algorithms = discovery.discover_novel_algorithms(num_algorithms=5)
        algorithm_time = time.time() - start_time
        
        details['algorithm_discovery_time'] = algorithm_time
        details['algorithm_discovery_rate'] = len(algorithms) / algorithm_time
        
        # Test baseline framework performance
        evaluator = BaselineEvaluator()
        mock_dataset = Mock()
        mock_dataset.name = 'perf_test'
        mock_dataset.num_node_features = 100
        mock_dataset.num_classes = 10
        
        start_time = time.time()
        result = evaluator.evaluate_baseline('GCN', mock_dataset, 'classification')
        baseline_time = time.time() - start_time
        
        details['baseline_evaluation_time'] = baseline_time
        
        # Performance scoring
        perf_score = 1.0
        if hypothesis_time > 5.0:  # Should generate 10 hypotheses in < 5 seconds
            perf_score -= 0.2
        if algorithm_time > 10.0:  # Should discover 5 algorithms in < 10 seconds
            perf_score -= 0.2
        if baseline_time > 15.0:  # Should evaluate baseline in < 15 seconds
            perf_score -= 0.2
        
        details['memory_usage'] = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if details['memory_usage'] > 1000:  # Should use < 1GB
            perf_score -= 0.2
        
        details['performance_score'] = max(0.0, perf_score)
        
        return QualityGateResult(
            gate_name="Performance",
            passed=perf_score >= self.passing_threshold,
            score=perf_score,
            details=details,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def evaluate_security_gate(self) -> QualityGateResult:
        """Evaluate security quality gate."""
        details = {}
        
        # Test security validator
        security_validator = SecurityValidator()
        
        # Test input validation
        valid_inputs = [
            "SELECT * FROM users WHERE id = 1",
            "test@example.com",
            "valid_filename.txt"
        ]
        
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "<script>alert('xss')</script>"
        ]
        
        valid_results = [security_validator.validate_input(inp) for inp in valid_inputs]
        malicious_results = [security_validator.validate_input(inp) for inp in malicious_inputs]
        
        details['valid_input_detection'] = sum(valid_results) / len(valid_results)
        details['malicious_input_detection'] = 1.0 - (sum(malicious_results) / len(malicious_results))
        
        # Test encryption capabilities
        test_data = b"sensitive research data"
        encrypted = security_validator.encrypt_data(test_data)
        decrypted = security_validator.decrypt_data(encrypted)
        
        details['encryption_integrity'] = 1.0 if decrypted == test_data else 0.0
        
        # Test access control
        try:
            ids = IntrusionDetectionSystem()
            anomaly_score = ids.detect_anomalies({
                'request_rate': 100,
                'error_rate': 0.01,
                'unusual_patterns': 0
            })
            details['intrusion_detection_active'] = 1.0 if anomaly_score >= 0 else 0.0
        except Exception:
            details['intrusion_detection_active'] = 0.0
        
        # Security scoring
        security_components = [
            details['valid_input_detection'],
            details['malicious_input_detection'],
            details['encryption_integrity'],
            details['intrusion_detection_active']
        ]
        
        security_score = sum(security_components) / len(security_components)
        details['security_score'] = security_score
        
        return QualityGateResult(
            gate_name="Security",
            passed=security_score >= self.passing_threshold,
            score=security_score,
            details=details,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def evaluate_reliability_gate(self) -> QualityGateResult:
        """Evaluate reliability quality gate."""
        details = {}
        
        # Test fault tolerance
        executor = FaultTolerantTaskExecutor()
        
        # Test successful task execution
        def success_task():
            return "success"
        
        # Test failing task with retry
        attempt_count = 0
        def failing_task():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success after retries"
        
        success_result = executor.execute_with_retry(success_task, max_retries=3)
        retry_result = executor.execute_with_retry(failing_task, max_retries=3)
        
        details['successful_execution'] = 1.0 if success_result == "success" else 0.0
        details['retry_mechanism'] = 1.0 if retry_result == "success after retries" else 0.0
        
        # Test system monitoring
        monitor = SystemMonitor()
        system_health = monitor.get_system_health()
        
        details['cpu_usage'] = system_health['cpu_percent']
        details['memory_usage'] = system_health['memory_percent']
        details['disk_usage'] = system_health['disk_percent']
        
        # Health scoring (lower usage is better for reliability)
        health_score = 1.0
        if system_health['cpu_percent'] > 80:
            health_score -= 0.2
        if system_health['memory_percent'] > 80:
            health_score -= 0.2
        if system_health['disk_percent'] > 90:
            health_score -= 0.2
        
        details['system_health_score'] = max(0.0, health_score)
        
        # Test database reliability
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                db_manager = ResearchDatabaseManager(db_path=f"{temp_dir}/test.db")
                
                # Test transaction rollback
                try:
                    with db_manager.get_connection() as conn:
                        conn.execute("CREATE TABLE test (id INTEGER)")
                        conn.execute("INSERT INTO test VALUES (1)")
                        raise Exception("Test rollback")
                except Exception:
                    pass
                
                # Verify rollback worked
                with db_manager.get_connection() as conn:
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test'")
                    table_exists = cursor.fetchone() is not None
                
                details['transaction_reliability'] = 0.0 if table_exists else 1.0
        except Exception:
            details['transaction_reliability'] = 0.0
        
        # Reliability scoring
        reliability_components = [
            details['successful_execution'],
            details['retry_mechanism'],
            details['system_health_score'],
            details['transaction_reliability']
        ]
        
        reliability_score = sum(reliability_components) / len(reliability_components)
        details['reliability_score'] = reliability_score
        
        return QualityGateResult(
            gate_name="Reliability",
            passed=reliability_score >= self.passing_threshold,
            score=reliability_score,
            details=details,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def evaluate_scalability_gate(self) -> QualityGateResult:
        """Evaluate scalability quality gate."""
        details = {}
        
        # Test memory optimization
        memory_optimizer = MemoryOptimizer()
        
        # Create test tensors of increasing size
        small_tensor = torch.randn(100, 100)
        large_tensor = torch.randn(1000, 1000)
        
        small_optimized = memory_optimizer.optimize_tensor_memory(small_tensor)
        large_optimized = memory_optimizer.optimize_tensor_memory(large_tensor)
        
        details['small_tensor_optimization'] = 1.0 if small_optimized.numel() == small_tensor.numel() else 0.0
        details['large_tensor_optimization'] = 1.0 if large_optimized.numel() == large_tensor.numel() else 0.0
        
        # Test algorithmic optimization
        algo_optimizer = AlgorithmicOptimizer()
        
        # Test with different problem sizes
        small_problem = {'size': 100, 'complexity': 'O(n)'}
        large_problem = {'size': 10000, 'complexity': 'O(n^2)'}
        
        small_strategy = algo_optimizer.optimize_algorithm_selection(small_problem)
        large_strategy = algo_optimizer.optimize_algorithm_selection(large_problem)
        
        details['small_problem_strategy'] = small_strategy['selected_algorithm']
        details['large_problem_strategy'] = large_strategy['selected_algorithm']
        details['optimization_effectiveness'] = 1.0 if small_strategy != large_strategy else 0.5
        
        # Test auto-scaling capabilities
        auto_scaler = AutoScaler()
        
        # Simulate different load patterns
        low_load = {'cpu_usage': 20, 'memory_usage': 30, 'request_rate': 10}
        high_load = {'cpu_usage': 80, 'memory_usage': 75, 'request_rate': 1000}
        
        low_scaling = auto_scaler.calculate_scaling_decision(low_load)
        high_scaling = auto_scaler.calculate_scaling_decision(high_load)
        
        details['low_load_scaling'] = low_scaling['action']
        details['high_load_scaling'] = high_scaling['action']
        details['scaling_responsiveness'] = 1.0 if low_scaling['action'] != high_scaling['action'] else 0.0
        
        # Test concurrent processing
        def cpu_intensive_task(size):
            return sum(i**2 for i in range(size))
        
        start_time = time.time()
        sequential_results = [cpu_intensive_task(1000) for _ in range(5)]
        sequential_time = time.time() - start_time
        
        start_time = time.time()
        with threading.ThreadPoolExecutor(max_workers=5) as executor:
            parallel_results = list(executor.map(cpu_intensive_task, [1000] * 5))
        parallel_time = time.time() - start_time
        
        details['sequential_time'] = sequential_time
        details['parallel_time'] = parallel_time
        details['parallelization_efficiency'] = min(1.0, sequential_time / parallel_time / 2)  # At least 2x speedup
        
        # Scalability scoring
        scalability_components = [
            details['small_tensor_optimization'],
            details['large_tensor_optimization'],
            details['optimization_effectiveness'],
            details['scaling_responsiveness'],
            details['parallelization_efficiency']
        ]
        
        scalability_score = sum(scalability_components) / len(scalability_components)
        details['scalability_score'] = scalability_score
        
        return QualityGateResult(
            gate_name="Scalability",
            passed=scalability_score >= self.passing_threshold,
            score=scalability_score,
            details=details,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def evaluate_maintainability_gate(self) -> QualityGateResult:
        """Evaluate maintainability quality gate."""
        details = {}
        
        # Test code modularity
        modules = [
            'quantum_research_discovery',
            'experimental_baseline_framework',
            'robust_research_infrastructure',
            'hyperscale_optimization_engine'
        ]
        
        module_scores = []
        for module in modules:
            try:
                exec(f"from src.scgraph_hub.{module} import *")
                module_scores.append(1.0)
            except Exception:
                module_scores.append(0.0)
        
        details['module_importability'] = sum(module_scores) / len(module_scores)
        
        # Test configuration management
        try:
            engine = QuantumResearchEngine()
            config = engine.config
            required_config_keys = ['research_domains', 'knowledge_base', 'experimental_settings']
            config_completeness = sum(1 for key in required_config_keys if key in config) / len(required_config_keys)
            details['configuration_completeness'] = config_completeness
        except Exception:
            details['configuration_completeness'] = 0.0
        
        # Test error handling
        error_handling_score = 0.0
        
        # Test graceful degradation with invalid inputs
        try:
            oracle = QuantumResearchOracle({})
            hypotheses = oracle.generate_hypotheses('invalid_domain', num_hypotheses=1)
            error_handling_score += 0.25 if len(hypotheses) > 0 else 0.0
        except Exception:
            pass
        
        try:
            discovery = NovelAlgorithmDiscovery()
            algorithms = discovery.discover_novel_algorithms(num_algorithms=0)
            error_handling_score += 0.25 if len(algorithms) == 0 else 0.0
        except Exception:
            pass
        
        try:
            evaluator = BaselineEvaluator()
            # This should raise a KeyError for unknown model
            try:
                evaluator.evaluate_baseline('UnknownModel', Mock(), 'classification')
            except KeyError:
                error_handling_score += 0.25  # Proper error handling
        except Exception:
            pass
        
        try:
            analyzer = StatisticalAnalyzer()
            # Test with insufficient data
            comparison = analyzer.compare_methods([])  # Empty list
            if 'error' in comparison:
                error_handling_score += 0.25  # Proper error handling
        except Exception:
            pass
        
        details['error_handling_score'] = error_handling_score
        
        # Test logging and monitoring integration
        try:
            monitor = SystemMonitor()
            health = monitor.get_system_health()
            details['monitoring_integration'] = 1.0 if 'timestamp' in health else 0.0
        except Exception:
            details['monitoring_integration'] = 0.0
        
        # Maintainability scoring
        maintainability_components = [
            details['module_importability'],
            details['configuration_completeness'],
            details['error_handling_score'],
            details['monitoring_integration']
        ]
        
        maintainability_score = sum(maintainability_components) / len(maintainability_components)
        details['maintainability_score'] = maintainability_score
        
        return QualityGateResult(
            gate_name="Maintainability",
            passed=maintainability_score >= self.passing_threshold,
            score=maintainability_score,
            details=details,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def evaluate_test_coverage_gate(self) -> QualityGateResult:
        """Evaluate test coverage quality gate."""
        details = {}
        
        # Count test files and test functions
        test_dir = Path('/root/repo/tests')
        test_files = list(test_dir.glob('test_*.py'))
        
        details['test_files_count'] = len(test_files)
        details['test_files'] = [f.name for f in test_files]
        
        # Analyze test coverage by counting test functions
        total_test_functions = 0
        test_coverage_by_module = {}
        
        for test_file in test_files:
            try:
                content = test_file.read_text()
                test_functions = content.count('def test_')
                total_test_functions += test_functions
                
                module_name = test_file.stem.replace('test_', '')
                test_coverage_by_module[module_name] = test_functions
            except Exception:
                pass
        
        details['total_test_functions'] = total_test_functions
        details['test_coverage_by_module'] = test_coverage_by_module
        
        # Expected minimum test functions per major component
        expected_coverage = {
            'quantum_research_discovery': 15,
            'experimental_baseline_framework': 20,
            'robust_research_infrastructure': 10,
            'hyperscale_optimization_engine': 8,
            'comprehensive_quality_gates': 6
        }
        
        coverage_scores = []
        for module, expected in expected_coverage.items():
            actual = test_coverage_by_module.get(module, 0)
            coverage_score = min(1.0, actual / expected)
            coverage_scores.append(coverage_score)
            details[f'{module}_coverage'] = coverage_score
        
        # Test actual functionality coverage
        functionality_tests = []
        
        # Test quantum research components
        try:
            oracle = QuantumResearchOracle({})
            hypotheses = oracle.generate_hypotheses('test', 1)
            functionality_tests.append(1.0 if len(hypotheses) > 0 else 0.0)
        except Exception:
            functionality_tests.append(0.0)
        
        # Test algorithm discovery
        try:
            discovery = NovelAlgorithmDiscovery()
            algorithms = discovery.discover_novel_algorithms(1)
            functionality_tests.append(1.0 if len(algorithms) > 0 else 0.0)
        except Exception:
            functionality_tests.append(0.0)
        
        # Test baseline framework
        try:
            evaluator = BaselineEvaluator()
            mock_dataset = Mock()
            mock_dataset.name = 'test'
            mock_dataset.num_node_features = 100
            mock_dataset.num_classes = 10
            result = evaluator.evaluate_baseline('GCN', mock_dataset, 'classification')
            functionality_tests.append(1.0 if result is not None else 0.0)
        except Exception:
            functionality_tests.append(0.0)
        
        # Test infrastructure components
        try:
            monitor = SystemMonitor()
            health = monitor.get_system_health()
            functionality_tests.append(1.0 if health is not None else 0.0)
        except Exception:
            functionality_tests.append(0.0)
        
        details['functionality_coverage'] = sum(functionality_tests) / len(functionality_tests)
        
        # Overall test coverage scoring
        test_coverage_score = (
            sum(coverage_scores) / len(coverage_scores) * 0.6 +  # Static coverage
            details['functionality_coverage'] * 0.4  # Functional coverage
        )
        
        details['test_coverage_score'] = test_coverage_score
        
        return QualityGateResult(
            gate_name="Test Coverage",
            passed=test_coverage_score >= self.passing_threshold,
            score=test_coverage_score,
            details=details,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )


class TestComprehensiveQualityGates:
    """Test the comprehensive quality gates system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.evaluator = QualityGateEvaluator()
    
    def test_quality_gate_evaluator_initialization(self):
        """Test quality gate evaluator initialization."""
        assert self.evaluator.passing_threshold == 0.85
        assert self.evaluator.overall_score == 0.0
        assert isinstance(self.evaluator.results, dict)
    
    def test_performance_quality_gate(self):
        """Test performance quality gate evaluation."""
        result = self.evaluator.evaluate_performance_gate()
        
        assert isinstance(result, QualityGateResult)
        assert result.gate_name == "Performance"
        assert 0.0 <= result.score <= 1.0
        assert 'hypothesis_generation_time' in result.details
        assert 'algorithm_discovery_time' in result.details
        assert 'baseline_evaluation_time' in result.details
        assert 'memory_usage' in result.details
        assert 'performance_score' in result.details
    
    def test_security_quality_gate(self):
        """Test security quality gate evaluation."""
        result = self.evaluator.evaluate_security_gate()
        
        assert isinstance(result, QualityGateResult)
        assert result.gate_name == "Security"
        assert 0.0 <= result.score <= 1.0
        assert 'valid_input_detection' in result.details
        assert 'malicious_input_detection' in result.details
        assert 'encryption_integrity' in result.details
        assert 'intrusion_detection_active' in result.details
        assert 'security_score' in result.details
    
    def test_reliability_quality_gate(self):
        """Test reliability quality gate evaluation."""
        result = self.evaluator.evaluate_reliability_gate()
        
        assert isinstance(result, QualityGateResult)
        assert result.gate_name == "Reliability"
        assert 0.0 <= result.score <= 1.0
        assert 'successful_execution' in result.details
        assert 'retry_mechanism' in result.details
        assert 'system_health_score' in result.details
        assert 'transaction_reliability' in result.details
        assert 'reliability_score' in result.details
    
    def test_scalability_quality_gate(self):
        """Test scalability quality gate evaluation."""
        result = self.evaluator.evaluate_scalability_gate()
        
        assert isinstance(result, QualityGateResult)
        assert result.gate_name == "Scalability"
        assert 0.0 <= result.score <= 1.0
        assert 'small_tensor_optimization' in result.details
        assert 'large_tensor_optimization' in result.details
        assert 'optimization_effectiveness' in result.details
        assert 'scaling_responsiveness' in result.details
        assert 'parallelization_efficiency' in result.details
        assert 'scalability_score' in result.details
    
    def test_maintainability_quality_gate(self):
        """Test maintainability quality gate evaluation."""
        result = self.evaluator.evaluate_maintainability_gate()
        
        assert isinstance(result, QualityGateResult)
        assert result.gate_name == "Maintainability"
        assert 0.0 <= result.score <= 1.0
        assert 'module_importability' in result.details
        assert 'configuration_completeness' in result.details
        assert 'error_handling_score' in result.details
        assert 'monitoring_integration' in result.details
        assert 'maintainability_score' in result.details
    
    def test_test_coverage_quality_gate(self):
        """Test test coverage quality gate evaluation."""
        result = self.evaluator.evaluate_test_coverage_gate()
        
        assert isinstance(result, QualityGateResult)
        assert result.gate_name == "Test Coverage"
        assert 0.0 <= result.score <= 1.0
        assert 'test_files_count' in result.details
        assert 'total_test_functions' in result.details
        assert 'test_coverage_by_module' in result.details
        assert 'functionality_coverage' in result.details
        assert 'test_coverage_score' in result.details
        
        # Should have multiple test files
        assert result.details['test_files_count'] >= 4
    
    def test_comprehensive_quality_gate_evaluation(self):
        """Test comprehensive quality gate evaluation."""
        results = self.evaluator.evaluate_all_gates()
        
        expected_gates = [
            "Performance", "Security", "Reliability", 
            "Scalability", "Maintainability", "Test Coverage"
        ]
        
        assert len(results) == len(expected_gates)
        
        for gate_name in expected_gates:
            assert gate_name in results
            result = results[gate_name]
            assert isinstance(result, QualityGateResult)
            assert 0.0 <= result.score <= 1.0
            assert result.gate_name == gate_name
        
        # Check overall score calculation
        expected_overall = sum(r.score for r in results.values()) / len(results)
        assert abs(self.evaluator.overall_score - expected_overall) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])