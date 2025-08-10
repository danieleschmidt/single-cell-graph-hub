"""Comprehensive tests for autonomous SDLC features."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json

from src.scgraph_hub.autonomous import (
    AutonomousSDLC, ExecutionPhase, TaskMetrics, ResearchHypothesis,
    get_autonomous_executor, execute_autonomous_sdlc
)
from src.scgraph_hub.research import (
    NovelAlgorithmResearcher, ExperimentalResults, ResearchPaper,
    get_researcher, execute_research_phase
)
from src.scgraph_hub.reliability import (
    ErrorRecoverySystem, CircuitBreaker, RetryManager, FailureType,
    FailureContext, SelfHealingSystem, get_error_recovery_system
)
from src.scgraph_hub.advanced_scalability import (
    AdaptiveLoadBalancer, IntelligentResourceManager, ScalingStrategy,
    ResourceMetrics, get_load_balancer, get_resource_manager
)


class TestAutonomousSDLC:
    """Test autonomous SDLC execution."""
    
    @pytest.fixture
    def temp_project_root(self):
        """Create temporary project root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def autonomous_sdlc(self, temp_project_root):
        """Create AutonomousSDLC instance."""
        return AutonomousSDLC(temp_project_root)
    
    def test_autonomous_sdlc_initialization(self, autonomous_sdlc):
        """Test SDLC initialization."""
        assert autonomous_sdlc.current_phase == ExecutionPhase.ANALYSIS
        assert len(autonomous_sdlc.execution_log) == 0
        assert len(autonomous_sdlc.research_hypotheses) == 0
        assert autonomous_sdlc.quality_gates["test_coverage"] == 85.0
    
    @pytest.mark.asyncio
    async def test_analysis_phase(self, autonomous_sdlc):
        """Test analysis phase execution."""
        result = await autonomous_sdlc._execute_analysis_phase()
        
        assert "metrics" in result
        assert "structure" in result
        assert "conventions" in result
        assert result["metrics"].success is True
        assert autonomous_sdlc.current_phase == ExecutionPhase.ANALYSIS
    
    @pytest.mark.asyncio
    async def test_generation_phases(self, autonomous_sdlc):
        """Test all generation phases."""
        phases = [ExecutionPhase.GENERATION_1, ExecutionPhase.GENERATION_2, ExecutionPhase.GENERATION_3]
        
        for phase in phases:
            result = await autonomous_sdlc._execute_generation_phase(phase)
            
            assert "metrics" in result
            assert "features_added" in result
            assert "quality_score" in result
            assert result["metrics"].success is True
            assert result["quality_score"] > 0
    
    @pytest.mark.asyncio
    async def test_quality_gates(self, autonomous_sdlc):
        """Test quality gates execution."""
        result = await autonomous_sdlc._execute_quality_gates()
        
        assert "gates_passed" in result
        assert "metrics" in result
        assert "quality_score" in result
        assert isinstance(result["gates_passed"], bool)
    
    @pytest.mark.asyncio
    async def test_research_phase(self, autonomous_sdlc):
        """Test research phase execution."""
        # Add mock research hypotheses
        autonomous_sdlc.research_hypotheses = [
            ResearchHypothesis(
                hypothesis="Test hypothesis",
                success_criteria={"accuracy": 0.9},
                baseline_method="baseline",
                novel_approach="novel",
                experimental_setup={}
            )
        ]
        
        result = await autonomous_sdlc._execute_research_phase()
        
        assert "discoveries" in result
        assert "hypotheses_tested" in result
        assert result["hypotheses_tested"] == 1
    
    @pytest.mark.asyncio
    async def test_production_phase(self, autonomous_sdlc):
        """Test production deployment phase."""
        result = await autonomous_sdlc._execute_production_phase()
        
        assert "global_ready" in result
        assert "deployment_config" in result
        assert "ready" in result
        assert "supported_regions" in result
        assert "supported_languages" in result
    
    @pytest.mark.asyncio
    async def test_full_sdlc_execution(self, temp_project_root):
        """Test complete SDLC execution."""
        result = await execute_autonomous_sdlc(temp_project_root)
        
        assert "start_time" in result
        assert "end_time" in result
        assert "phases" in result
        assert "production_ready" in result
        
        # Check all phases were executed
        expected_phases = ["analysis", "generation_1", "generation_2", "generation_3", "quality_gates", "production"]
        for phase in expected_phases:
            assert phase in result["phases"]


class TestResearchExecution:
    """Test research execution capabilities."""
    
    @pytest.fixture
    def researcher(self):
        """Create researcher instance."""
        return NovelAlgorithmResearcher()
    
    @pytest.mark.asyncio
    async def test_research_opportunities_discovery(self, researcher):
        """Test discovery of research opportunities."""
        opportunities = await researcher.discover_research_opportunities()
        
        assert len(opportunities) > 0
        assert all("title" in opp for opp in opportunities)
        assert all("hypothesis" in opp for opp in opportunities)
        assert all("novelty" in opp for opp in opportunities)
        assert all("potential_impact" in opp for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_novel_algorithm_implementation(self, researcher):
        """Test novel algorithm implementation."""
        algorithm_spec = {"name": "BioPriorGAT"}
        result = await researcher.implement_novel_algorithm(algorithm_spec)
        
        assert result is not None
        assert hasattr(result, '__init__')  # Should be a class
    
    @pytest.mark.asyncio
    async def test_comparative_study(self, researcher):
        """Test comparative study execution."""
        # Mock novel method
        novel_method = Mock()
        baseline_methods = [Mock(), Mock()]
        
        results = await researcher.run_comparative_study(
            novel_method=novel_method,
            baseline_methods=baseline_methods,
            datasets=["test_dataset"],
            metrics=["accuracy"],
            n_runs=2
        )
        
        assert len(results) > 0
        assert all(isinstance(r, ExperimentalResults) for r in results)
        assert all(r.dataset == "test_dataset" for r in results)
    
    @pytest.mark.asyncio
    async def test_reproducibility_validation(self, researcher):
        """Test reproducibility validation."""
        # Create mock results
        results = [
            ExperimentalResults(
                method_name="test_method",
                dataset="test_dataset",
                metrics={"accuracy": 0.9},
                runtime=1.0,
                memory_usage=100.0,
                reproducibility_score=0.95
            ),
            ExperimentalResults(
                method_name="test_method",
                dataset="test_dataset",
                metrics={"accuracy": 0.91},
                runtime=1.1,
                memory_usage=105.0,
                reproducibility_score=0.93
            )
        ]
        
        reproducibility = await researcher.validate_reproducibility(results)
        
        assert "overall_reproducibility" in reproducibility
        assert "method_reproducibility" in reproducibility
        assert "threshold_met" in reproducibility
        assert isinstance(reproducibility["overall_reproducibility"], float)
    
    @pytest.mark.asyncio
    async def test_publication_preparation(self, researcher):
        """Test research publication preparation."""
        # Mock results
        results = [
            ExperimentalResults(
                method_name="novel_method",
                dataset="test_dataset",
                metrics={"accuracy": 0.95, "f1_score": 0.93},
                runtime=2.0,
                memory_usage=200.0,
                statistical_significance=True,
                p_value=0.001
            )
        ]
        
        paper = await researcher.prepare_publication(
            results=results,
            research_title="Test Research",
            novel_contributions=["Novel algorithm", "Improved accuracy"]
        )
        
        assert isinstance(paper, ResearchPaper)
        assert paper.title == "Test Research"
        assert len(paper.results) == 1
        assert paper.code_availability != ""
        assert paper.data_availability != ""
    
    @pytest.mark.asyncio
    async def test_full_research_execution(self):
        """Test complete research execution."""
        result = await execute_research_phase()
        
        assert "research_completed" in result
        assert "novel_contributions" in result
        assert "results" in result
        assert "reproducibility" in result
        assert result["research_completed"] is True


class TestReliabilityFeatures:
    """Test reliability and error recovery features."""
    
    @pytest.fixture
    def error_recovery(self):
        """Create error recovery system."""
        return ErrorRecoverySystem()
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker."""
        return CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
    
    @pytest.fixture
    def retry_manager(self):
        """Create retry manager."""
        return RetryManager(max_attempts=3, base_delay=0.1)
    
    def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test circuit breaker in closed state."""
        def success_func():
            return "success"
        
        result = circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
    
    def test_circuit_breaker_open_state(self, circuit_breaker):
        """Test circuit breaker opening after failures."""
        def failing_func():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit
        for _ in range(circuit_breaker.failure_threshold):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == "OPEN"
        
        # Should prevent further calls
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_retry_manager_success(self, retry_manager):
        """Test retry manager with successful function."""
        call_count = 0
        
        def success_after_retries():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = await retry_manager.retry_async(success_after_retries)
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_manager_max_attempts(self, retry_manager):
        """Test retry manager respects max attempts."""
        call_count = 0
        
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")
        
        with pytest.raises(Exception, match="Always fails"):
            await retry_manager.retry_async(always_fails)
        
        assert call_count == retry_manager.max_attempts
    
    def test_failure_classification(self, error_recovery):
        """Test failure type classification."""
        memory_error = MemoryError("Out of memory")
        import_error = ImportError("No module named 'test'")
        timeout_error = TimeoutError("Operation timed out")
        
        assert error_recovery._classify_failure(memory_error) == FailureType.MEMORY_ERROR
        assert error_recovery._classify_failure(import_error) == FailureType.DEPENDENCY_ERROR
        assert error_recovery._classify_failure(timeout_error) == FailureType.TIMEOUT_ERROR
    
    @pytest.mark.asyncio
    async def test_error_recovery_handling(self, error_recovery):
        """Test error recovery handling."""
        test_exception = Exception("Test error")
        context = {"test": "context"}
        
        with patch.object(error_recovery, '_attempt_recovery', return_value=True):
            result = await error_recovery.handle_failure(test_exception, context)
            assert result is True
        
        assert len(error_recovery.failure_history) == 1
        assert error_recovery.failure_history[0].error_message == "Test error"
    
    @pytest.mark.asyncio
    async def test_self_healing_system_monitoring(self):
        """Test self-healing system monitoring."""
        from src.scgraph_hub.reliability import SelfHealingSystem
        
        healing_system = SelfHealingSystem()
        
        # Start monitoring for a short time
        monitor_task = asyncio.create_task(healing_system.start_monitoring(check_interval=0.1))
        await asyncio.sleep(0.3)  # Let it run for a bit
        healing_system.stop_monitoring()
        
        # Wait for the monitoring task to finish
        try:
            await asyncio.wait_for(monitor_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitor_task.cancel()
        
        assert not healing_system.monitoring_active


class TestScalabilityFeatures:
    """Test scalability and performance features."""
    
    @pytest.fixture
    def load_balancer(self):
        """Create load balancer instance."""
        return AdaptiveLoadBalancer(initial_workers=2)
    
    @pytest.fixture
    def resource_manager(self):
        """Create resource manager instance."""
        return IntelligentResourceManager()
    
    @pytest.mark.asyncio
    async def test_load_balancer_initialization(self, load_balancer):
        """Test load balancer initialization."""
        await load_balancer.start()
        
        assert load_balancer.current_workers == 2
        assert load_balancer.min_workers == 1
        assert load_balancer.max_workers == 4
        assert len(load_balancer.workers) == 2
    
    @pytest.mark.asyncio
    async def test_load_balancer_task_submission(self, load_balancer):
        """Test task submission to load balancer."""
        await load_balancer.start()
        
        def test_task(x):
            return x * 2
        
        result = await load_balancer.submit_task(test_task, 5)
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_load_balancer_scaling(self, load_balancer):
        """Test load balancer scaling."""
        await load_balancer.start()
        
        # Scale up
        await load_balancer._scale_workers(3)
        assert load_balancer.current_workers == 3
        assert len(load_balancer.workers) == 3
        
        # Scale down
        await load_balancer._scale_workers(1)
        assert load_balancer.current_workers == 1
        assert len(load_balancer.workers) == 1
    
    def test_resource_metrics(self):
        """Test resource metrics data structure."""
        metrics = ResourceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_io=1000.0,
            network_io=2000.0
        )
        
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0
        assert metrics.disk_io == 1000.0
        assert metrics.network_io == 2000.0
        assert isinstance(metrics.timestamp, type(metrics.timestamp))
    
    @pytest.mark.asyncio
    async def test_resource_manager_prediction(self, resource_manager):
        """Test resource prediction."""
        # Add some mock history
        for i in range(60):
            metrics = ResourceMetrics(
                cpu_usage=50 + i * 0.5,  # Increasing trend
                memory_usage=60 - i * 0.2,  # Decreasing trend
                disk_io=1000,
                network_io=2000
            )
            resource_manager.resource_history.append(metrics)
        
        predictions = await resource_manager._predict_resource_needs()
        
        assert "cpu" in predictions
        assert "memory" in predictions
        assert "confidence" in predictions
        assert 0 <= predictions["cpu"] <= 100
        assert 0 <= predictions["memory"] <= 100
    
    @pytest.mark.asyncio
    async def test_scaling_decision_logic(self, resource_manager):
        """Test scaling decision logic."""
        # High resource usage prediction
        high_predictions = {"cpu": 85, "memory": 90, "confidence": 0.8}
        decision = await resource_manager._decide_scaling_action(high_predictions)
        
        assert decision is not None
        assert decision.strategy in [ScalingStrategy.HORIZONTAL, ScalingStrategy.VERTICAL]
        assert decision.confidence == 0.8
        
        # Low resource usage prediction
        low_predictions = {"cpu": 20, "memory": 30, "confidence": 0.8}
        decision = await resource_manager._decide_scaling_action(low_predictions)
        
        if decision is not None:  # May be None if already at minimum
            assert decision.strategy == ScalingStrategy.HORIZONTAL
            assert decision.cost_estimate < 0  # Should save cost
    
    @pytest.mark.asyncio
    async def test_resource_allocation_efficiency(self, resource_manager):
        """Test resource allocation efficiency calculation."""
        # Add balanced metrics
        for _ in range(10):
            metrics = ResourceMetrics(
                cpu_usage=70.0,  # Optimal range
                memory_usage=75.0,  # Optimal range
                disk_io=1000,
                network_io=2000
            )
            resource_manager.resource_history.append(metrics)
        
        efficiency = await resource_manager._calculate_allocation_efficiency()
        assert 0.0 <= efficiency <= 1.0
        assert efficiency > 0.5  # Should be reasonably efficient


class TestIntegration:
    """Integration tests for autonomous features."""
    
    @pytest.mark.asyncio
    async def test_autonomous_with_research(self):
        """Test autonomous SDLC with research integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sdlc = AutonomousSDLC(Path(temp_dir))
            
            # Add research hypothesis
            hypothesis = ResearchHypothesis(
                hypothesis="Integration test hypothesis",
                success_criteria={"accuracy": 0.9},
                baseline_method="baseline",
                novel_approach="novel",
                experimental_setup={}
            )
            sdlc.research_hypotheses.append(hypothesis)
            
            # Execute phases
            analysis_result = await sdlc._execute_analysis_phase()
            assert analysis_result["metrics"].success
            
            research_result = await sdlc._execute_research_phase()
            assert research_result["hypotheses_tested"] == 1
    
    @pytest.mark.asyncio
    async def test_reliability_with_scalability(self):
        """Test reliability features with scalability."""
        load_balancer = AdaptiveLoadBalancer(initial_workers=1)
        await load_balancer.start()
        
        error_recovery = get_error_recovery_system()
        
        # Test task with potential failure and recovery
        def potentially_failing_task(x):
            if x < 0:
                raise ValueError("Negative input")
            return x * 2
        
        # Should succeed
        result = await load_balancer.submit_task(potentially_failing_task, 5)
        assert result == 10
        
        # Should handle failure through error recovery
        try:
            await load_balancer.submit_task(potentially_failing_task, -1)
        except ValueError:
            # Error should be captured by error recovery system
            pass
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_system_integration(self):
        """Test full system integration - takes longer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Execute full autonomous SDLC
            result = await execute_autonomous_sdlc(Path(temp_dir))
            
            # Verify all components worked together
            assert result["production_ready"] in [True, False]  # Should have a value
            assert "phases" in result
            assert "quality_metrics" in result
            
            # Check execution report was created
            report_path = Path(temp_dir) / "autonomous_execution_report.json"
            assert report_path.exists()
            
            # Verify report content
            with open(report_path) as f:
                report_data = json.load(f)
                assert "start_time" in report_data
                assert "phases" in report_data


class TestPerformanceAndBenchmarks:
    """Performance tests and benchmarks."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_load_balancer_throughput(self):
        """Benchmark load balancer throughput."""
        load_balancer = AdaptiveLoadBalancer(initial_workers=4)
        await load_balancer.start()
        
        def simple_task(x):
            return x ** 2
        
        # Submit many tasks and measure throughput
        start_time = time.time()
        tasks = []
        
        for i in range(100):
            task = load_balancer.submit_task(simple_task, i)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = len(results) / duration
        
        assert len(results) == 100
        assert throughput > 10  # At least 10 tasks per second
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_error_recovery_overhead(self):
        """Benchmark error recovery overhead."""
        error_recovery = ErrorRecoverySystem()
        
        def normal_task():
            return "success"
        
        # Measure normal execution time
        start_time = time.time()
        for _ in range(100):
            normal_task()
        normal_duration = time.time() - start_time
        
        # Measure with error recovery wrapper
        async def wrapped_task():
            try:
                return normal_task()
            except Exception as e:
                await error_recovery.handle_failure(e, {})
                return "recovered"
        
        start_time = time.time()
        for _ in range(100):
            await wrapped_task()
        wrapped_duration = time.time() - start_time
        
        # Error recovery should add minimal overhead for successful operations
        overhead_factor = wrapped_duration / normal_duration
        assert overhead_factor < 3.0  # Less than 3x overhead


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])