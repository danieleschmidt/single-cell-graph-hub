"""
Test suite for Quantum Research Discovery Engine v4.0
Comprehensive testing of autonomous research capabilities
"""

import pytest
import asyncio
import numpy as np
import torch
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import asdict

from src.scgraph_hub.quantum_research_discovery import (
    QuantumResearchEngine,
    QuantumResearchOracle,
    NovelAlgorithmDiscovery,
    ExperimentalFramework,
    ResearchHypothesis,
    NovelAlgorithm,
    run_autonomous_research
)


class TestQuantumResearchOracle:
    """Test the quantum research oracle."""
    
    def setup_method(self):
        """Setup test environment."""
        self.knowledge_base = {
            'graph_neural_networks': {
                'attention_mechanisms': 0.3,
                'message_passing': 0.4,
                'biological_priors': 0.3
            }
        }
        self.oracle = QuantumResearchOracle(self.knowledge_base)
    
    def test_initialization(self):
        """Test oracle initialization."""
        assert self.oracle.knowledge_base == self.knowledge_base
        assert 'superposition' in self.oracle.quantum_circuit
        assert 'entanglement' in self.oracle.quantum_circuit
        assert 'measurement' in self.oracle.quantum_circuit
    
    def test_quantum_circuit_initialization(self):
        """Test quantum circuit setup."""
        circuit = self.oracle.quantum_circuit
        
        assert circuit['superposition'].shape == (64, 64)
        assert circuit['entanglement'].shape == (64, 64)
        assert circuit['measurement'].shape == (64,)
        
        # Test unitarity of entanglement matrix (approximately)
        U = circuit['entanglement']
        should_be_identity = U @ U.conj().T
        identity_error = np.abs(should_be_identity - np.eye(64)).max()
        assert identity_error < 1e-10
    
    def test_hypothesis_generation(self):
        """Test hypothesis generation."""
        hypotheses = self.oracle.generate_hypotheses('graph_neural_networks', num_hypotheses=5)
        
        assert len(hypotheses) == 5
        assert all(isinstance(h, ResearchHypothesis) for h in hypotheses)
        assert all(h.domain == 'graph_neural_networks' for h in hypotheses)
        assert all(0 <= h.novelty_score <= 1 for h in hypotheses)
        assert all(0 <= h.feasibility_score <= 1 for h in hypotheses)
        assert all(0 <= h.impact_potential <= 1 for h in hypotheses)
    
    def test_hypothesis_quality(self):
        """Test hypothesis quality metrics."""
        hypotheses = self.oracle.generate_hypotheses('graph_neural_networks', num_hypotheses=10)
        
        # Test that hypotheses are sorted by composite score
        scores = [h.composite_score for h in hypotheses]
        assert scores == sorted(scores, reverse=True)
        
        # Test confidence intervals
        for h in hypotheses:
            ci_low, ci_high = h.confidence_interval
            assert ci_low <= ci_high
            assert 0 <= ci_low <= 1
            assert 0 <= ci_high <= 1
    
    def test_quantum_state_properties(self):
        """Test quantum state properties."""
        hypotheses = self.oracle.generate_hypotheses('graph_neural_networks', num_hypotheses=3)
        
        for h in hypotheses:
            # Test quantum coherence calculation
            assert 0 <= h.quantum_coherence <= 1
            
            # Test quantum state normalization
            state_values = list(h.quantum_state.values())
            total_probability = sum(state_values)
            assert total_probability > 0  # States should have some probability
    
    def test_novelty_scoring(self):
        """Test novelty scoring mechanism."""
        # First hypothesis should be highly novel
        h1 = self.oracle.generate_hypotheses('graph_neural_networks', num_hypotheses=1)[0]
        assert h1.novelty_score >= 0.8
        
        # Subsequent hypotheses should have lower novelty
        h2 = self.oracle.generate_hypotheses('graph_neural_networks', num_hypotheses=1)[0]
        assert h2.novelty_score < h1.novelty_score
    
    def test_experimental_design_generation(self):
        """Test experimental design generation."""
        hypotheses = self.oracle.generate_hypotheses('graph_neural_networks', num_hypotheses=1)
        h = hypotheses[0]
        
        exp_design = h.experimental_design
        assert 'baseline_methods' in exp_design
        assert 'datasets' in exp_design
        assert 'metrics' in exp_design
        assert 'sample_size' in exp_design
        assert 'statistical_test' in exp_design
        
        assert len(exp_design['baseline_methods']) > 0
        assert len(exp_design['datasets']) > 0
        assert len(exp_design['metrics']) > 0
        assert exp_design['sample_size'] > 0


class TestNovelAlgorithmDiscovery:
    """Test novel algorithm discovery system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.discovery = NovelAlgorithmDiscovery()
    
    def test_initialization(self):
        """Test algorithm discovery initialization."""
        assert len(self.discovery.algorithm_space) > 0
        assert 'message_passing' in self.discovery.algorithm_space
        assert 'aggregation' in self.discovery.algorithm_space
        assert isinstance(self.discovery.discovered_algorithms, list)
    
    def test_algorithm_space_completeness(self):
        """Test algorithm space contains necessary components."""
        space = self.discovery.algorithm_space
        
        required_components = [
            'message_passing', 'aggregation', 'update_functions', 
            'readout', 'biological_constraints'
        ]
        
        for component in required_components:
            assert component in space
            assert len(space[component]) > 0
    
    def test_novel_algorithm_generation(self):
        """Test novel algorithm generation."""
        algorithms = self.discovery.discover_novel_algorithms(num_algorithms=3)
        
        assert len(algorithms) == 3
        assert all(isinstance(a, NovelAlgorithm) for a in algorithms)
        assert all(a.name.startswith('Bio') for a in algorithms)
        assert all(a.publication_readiness >= 0 for a in algorithms)
    
    def test_algorithm_mathematical_formulation(self):
        """Test mathematical formulation generation."""
        algorithms = self.discovery.discover_novel_algorithms(num_algorithms=1)
        algorithm = algorithms[0]
        
        formulation = algorithm.mathematical_formulation
        assert 'Algorithm:' in formulation
        assert 'Message Passing:' in formulation
        assert 'h_i^{(l+1)}' in formulation  # Standard GNN notation
        assert 'σ' in formulation  # Activation function
    
    def test_algorithm_implementation_generation(self):
        """Test algorithm implementation generation."""
        algorithms = self.discovery.discover_novel_algorithms(num_algorithms=1)
        algorithm = algorithms[0]
        
        implementation = algorithm.implementation
        assert 'import torch' in implementation
        assert 'torch.nn' in implementation
        assert 'MessagePassing' in implementation
        assert 'def forward(' in implementation
        assert algorithm.name in implementation
    
    def test_performance_evaluation(self):
        """Test algorithm performance evaluation."""
        algorithms = self.discovery.discover_novel_algorithms(num_algorithms=2)
        
        for algorithm in algorithms:
            perf = algorithm.empirical_performance
            assert 'accuracy' in perf
            assert 'f1_score' in perf
            assert 'biological_conservation' in perf
            assert 'training_time' in perf
            assert 'memory_usage' in perf
            
            assert 0 <= perf['accuracy'] <= 1
            assert 0 <= perf['f1_score'] <= 1
            assert 0 <= perf['biological_conservation'] <= 1
            assert perf['training_time'] > 0
            assert perf['memory_usage'] > 0
    
    def test_publication_readiness_assessment(self):
        """Test publication readiness assessment."""
        algorithms = self.discovery.discover_novel_algorithms(num_algorithms=5)
        
        # Test that algorithms are sorted by publication readiness
        readiness_scores = [a.publication_readiness for a in algorithms]
        assert readiness_scores == sorted(readiness_scores, reverse=True)
        
        # Test readiness score bounds
        for algorithm in algorithms:
            assert 0 <= algorithm.publication_readiness <= 1


class TestExperimentalFramework:
    """Test experimental framework."""
    
    def setup_method(self):
        """Setup test environment."""
        self.framework = ExperimentalFramework()
    
    def test_controlled_experiment_design(self):
        """Test controlled experiment design."""
        # Create a test hypothesis
        hypothesis = ResearchHypothesis(
            id='test_hyp_001',
            description='Test hypothesis for neural attention',
            domain='graph_neural_networks',
            novelty_score=0.8,
            feasibility_score=0.9,
            impact_potential=0.7,
            quantum_state={'concept_0': 0.5, 'concept_1': 0.3, 'concept_2': 0.2},
            dependencies=[],
            experimental_design={
                'datasets': ['pbmc_10k', 'mouse_brain'],
                'baseline_methods': ['GCN', 'GAT']
            },
            success_criteria={'accuracy_improvement': 0.05},
            timestamp=datetime.now().isoformat(),
            confidence_interval=(0.1, 0.9)
        )
        
        protocol = self.framework.design_controlled_experiment(hypothesis)
        
        assert protocol['experiment_id'] == f"exp_{hypothesis.id}"
        assert protocol['hypothesis'] == hypothesis.description
        assert 'experimental_design' in protocol
        assert 'data_collection' in protocol
        assert 'statistical_analysis' in protocol
        assert 'reproducibility_measures' in protocol
    
    def test_sample_size_calculation(self):
        """Test sample size calculation."""
        hypothesis = ResearchHypothesis(
            id='test_hyp_002',
            description='Test hypothesis',
            domain='test',
            novelty_score=0.5,
            feasibility_score=0.5,
            impact_potential=0.5,
            quantum_state={},
            dependencies=[],
            experimental_design={},
            success_criteria={'accuracy_improvement': 0.1},
            timestamp=datetime.now().isoformat(),
            confidence_interval=(0.1, 0.9)
        )
        
        sample_size = self.framework._calculate_sample_size(hypothesis)
        
        assert isinstance(sample_size, int)
        assert sample_size > 0
        assert sample_size >= 100  # Minimum reasonable sample size
    
    def test_validation_study_execution(self):
        """Test validation study execution."""
        # Create a test algorithm
        algorithm = NovelAlgorithm(
            name='TestGNN',
            description='Test graph neural network',
            mathematical_formulation='Test formulation',
            implementation='Test implementation',
            theoretical_complexity='O(n)',
            empirical_performance={'accuracy': 0.85},
            biological_motivation='Test motivation',
            comparative_analysis={},
            validation_results={},
            publication_readiness=0.8
        )
        
        datasets = ['test_dataset_1', 'test_dataset_2']
        results = self.framework.run_validation_study(algorithm, datasets)
        
        assert results['algorithm'] == algorithm.name
        assert 'validation_timestamp' in results
        assert 'datasets_tested' in results
        assert 'baseline_comparisons' in results
        assert 'statistical_analysis' in results
        assert 'reproducibility_results' in results
        assert 'biological_validation' in results
        
        assert len(results['datasets_tested']) == len(datasets)
    
    def test_statistical_analysis(self):
        """Test statistical analysis functionality."""
        # Create mock baseline comparisons
        baseline_comparisons = {
            'dataset1': {
                'algorithm_performance': 0.90,
                'baseline_performances': {'GCN': 0.85, 'GAT': 0.87},
                'improvement_over_best_baseline': 0.03,
                'statistical_significance': True,
                'effect_size': 0.6
            },
            'dataset2': {
                'algorithm_performance': 0.88,
                'baseline_performances': {'GCN': 0.82, 'GAT': 0.84},
                'improvement_over_best_baseline': 0.04,
                'statistical_significance': True,
                'effect_size': 0.8
            }
        }
        
        analysis = self.framework._perform_statistical_analysis(baseline_comparisons)
        
        assert 'mean_improvement' in analysis
        assert 'std_improvement' in analysis
        assert 'significant_datasets' in analysis
        assert 'total_datasets' in analysis
        assert 'overall_significance' in analysis
        assert 'effect_size_category' in analysis
        
        assert analysis['total_datasets'] == 2
        assert analysis['significant_datasets'] <= analysis['total_datasets']
    
    def test_reproducibility_testing(self):
        """Test reproducibility testing."""
        algorithm = NovelAlgorithm(
            name='TestGNN',
            description='Test',
            mathematical_formulation='Test',
            implementation='Test',
            theoretical_complexity='O(n)',
            empirical_performance={},
            biological_motivation='Test',
            comparative_analysis={},
            validation_results={},
            publication_readiness=0.5
        )
        
        repro_results = self.framework._test_reproducibility(algorithm)
        
        assert 'individual_runs' in repro_results
        assert 'mean_performance' in repro_results
        assert 'std_performance' in repro_results
        assert 'coefficient_of_variation' in repro_results
        assert 'reproducibility_score' in repro_results
        assert 'confidence_interval' in repro_results
        
        assert len(repro_results['individual_runs']) == 5  # Five random seeds
        assert 0 <= repro_results['reproducibility_score'] <= 1


class TestQuantumResearchEngine:
    """Test the main quantum research engine."""
    
    def setup_method(self):
        """Setup test environment."""
        self.engine = QuantumResearchEngine()
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.oracle is not None
        assert self.engine.algorithm_discovery is not None
        assert self.engine.experimental_framework is not None
        assert isinstance(self.engine.research_archive, dict)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = self.engine.config
        
        assert 'research_domains' in config
        assert 'knowledge_base' in config
        assert 'experimental_settings' in config
        
        assert len(config['research_domains']) > 0
        assert 'num_hypotheses_per_domain' in config['experimental_settings']
    
    @pytest.mark.asyncio
    async def test_research_cycle_execution(self):
        """Test complete research cycle execution."""
        domain = 'graph_neural_networks'
        
        results = await self.engine.execute_research_cycle(domain)
        
        assert results['domain'] == domain
        assert 'cycle_timestamp' in results
        assert 'hypothesis_generation' in results
        assert 'algorithm_discovery' in results
        assert 'experimental_validation' in results
        assert 'publication_preparation' in results
        
        # Check hypothesis generation results
        hyp_results = results['hypothesis_generation']
        assert hyp_results['num_generated'] > 0
        assert 'top_hypotheses' in hyp_results
        assert 'average_novelty' in hyp_results
        assert 'average_feasibility' in hyp_results
        
        # Check algorithm discovery results
        algo_results = results['algorithm_discovery']
        assert algo_results['num_discovered'] > 0
        assert 'algorithms' in algo_results
        assert 'average_publication_readiness' in algo_results
        
        # Check experimental validation
        val_results = results['experimental_validation']
        assert val_results['num_validated'] >= 0
        assert 'validation_results' in val_results
        assert 'statistically_significant' in val_results
    
    def test_contributions_summarization(self):
        """Test research contributions summarization."""
        # Create mock data
        hypotheses = [Mock(novelty_score=0.9), Mock(novelty_score=0.7)]
        algorithms = [Mock(description='novel algorithm'), Mock(description='standard algorithm')]
        validations = [
            {
                'statistical_analysis': {'mean_improvement': 0.05},
                'biological_validation': {'pathway_enrichment_score': 0.8},
                'reproducibility_results': {'reproducibility_score': 0.9}
            }
        ]
        
        contributions = self.engine._summarize_contributions(hypotheses, algorithms, validations)
        
        assert 'novel_hypotheses' in contributions
        assert 'validated_improvements' in contributions
        assert 'biological_relevance' in contributions
        assert 'reproducibility_scores' in contributions
        assert 'theoretical_contributions' in contributions
        
        assert contributions['novel_hypotheses'] == 1  # One hypothesis with novelty > 0.8
        assert contributions['validated_improvements'] == 1  # One validation with improvement > 0.02
    
    def test_next_steps_generation(self):
        """Test next steps generation."""
        cycle_results = {
            'experimental_validation': {'statistically_significant': 2},
            'publication_preparation': {'publication_ready_algorithms': 1},
            'hypothesis_generation': {'average_novelty': 0.85}
        }
        
        next_steps = self.engine._generate_next_steps(cycle_results)
        
        assert isinstance(next_steps, list)
        assert len(next_steps) > 0
        assert all(isinstance(step, str) for step in next_steps)
        
        # Should include steps for significant validations
        steps_text = ' '.join(next_steps)
        assert 'validation' in steps_text.lower() or 'manuscript' in steps_text.lower()
    
    def test_research_report_generation(self):
        """Test research report generation."""
        cycle_results = {
            'domain': 'graph_neural_networks',
            'cycle_timestamp': datetime.now().isoformat(),
            'hypothesis_generation': {
                'num_generated': 5,
                'average_novelty': 0.8,
                'average_feasibility': 0.7
            },
            'algorithm_discovery': {
                'num_discovered': 3,
                'average_publication_readiness': 0.85
            },
            'experimental_validation': {
                'num_validated': 2,
                'statistically_significant': 1
            },
            'publication_preparation': {
                'publication_ready_algorithms': 1,
                'research_contributions': {},
                'next_steps': ['Test next step']
            }
        }
        
        report = self.engine.generate_research_report(cycle_results)
        
        assert isinstance(report, str)
        assert 'Quantum Research Discovery Report' in report
        assert 'Executive Summary' in report
        assert 'Key Findings' in report
        assert 'Next Steps' in report
        assert cycle_results['domain'] in report
        
        # Check that numbers are correctly included
        assert '5' in report  # num_generated
        assert '3' in report  # num_discovered
        assert '2' in report  # num_validated


class TestIntegrationAndPerformance:
    """Integration and performance tests."""
    
    @pytest.mark.asyncio
    async def test_autonomous_research_integration(self):
        """Test full autonomous research integration."""
        domains = ['graph_neural_networks']
        
        results = await run_autonomous_research(domains)
        
        assert 'individual_results' in results
        assert 'comparative_analysis' in results
        assert 'research_archive' in results
        
        individual_results = results['individual_results']
        assert len(individual_results) == len(domains)
        
        for domain in domains:
            assert domain in individual_results
            result = individual_results[domain]
            assert 'hypothesis_generation' in result
            assert 'algorithm_discovery' in result
            assert 'experimental_validation' in result
    
    @pytest.mark.asyncio
    async def test_comparative_analysis(self):
        """Test comparative analysis across domains."""
        domains = ['graph_neural_networks', 'single_cell_analysis']
        
        results = await run_autonomous_research(domains)
        
        comp_analysis = results['comparative_analysis']
        assert 'cross_domain_insights' in comp_analysis
        assert 'performance_comparison' in comp_analysis
        assert 'interdisciplinary_opportunities' in comp_analysis
        assert 'unified_framework_potential' in comp_analysis
        
        perf_comparison = comp_analysis['performance_comparison']
        assert len(perf_comparison) == len(domains)
        
        for domain in domains:
            assert domain in perf_comparison
            metrics = perf_comparison[domain]
            assert 'hypothesis_quality' in metrics
            assert 'validation_success_rate' in metrics
            assert 'publication_readiness' in metrics
    
    def test_memory_efficiency(self):
        """Test memory efficiency of research components."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Create multiple research engines
        engines = [QuantumResearchEngine() for _ in range(5)]
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 100MB per engine)
        memory_per_engine = peak / len(engines)
        assert memory_per_engine < 100 * 1024 * 1024  # 100MB
    
    def test_algorithm_discovery_performance(self):
        """Test algorithm discovery performance."""
        discovery = NovelAlgorithmDiscovery()
        
        import time
        start_time = time.time()
        
        # Discover multiple algorithms
        algorithms = discovery.discover_novel_algorithms(num_algorithms=10)
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (10 seconds)
        assert execution_time < 10.0
        assert len(algorithms) == 10
    
    def test_hypothesis_generation_scalability(self):
        """Test hypothesis generation scalability."""
        oracle = QuantumResearchOracle({})
        
        # Test with different scales
        for num_hypotheses in [1, 10, 50, 100]:
            import time
            start_time = time.time()
            
            hypotheses = oracle.generate_hypotheses('test_domain', num_hypotheses)
            
            execution_time = time.time() - start_time
            
            assert len(hypotheses) == num_hypotheses
            # Time should scale roughly linearly
            assert execution_time < num_hypotheses * 0.1  # 0.1 seconds per hypothesis


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness."""
    
    def test_invalid_domain_handling(self):
        """Test handling of invalid research domains."""
        oracle = QuantumResearchOracle({})
        
        # Should handle unknown domains gracefully
        hypotheses = oracle.generate_hypotheses('unknown_domain', num_hypotheses=1)
        
        assert len(hypotheses) == 1
        assert hypotheses[0].domain == 'unknown_domain'
    
    def test_empty_knowledge_base(self):
        """Test handling of empty knowledge base."""
        oracle = QuantumResearchOracle({})
        
        hypotheses = oracle.generate_hypotheses('test_domain', num_hypotheses=3)
        
        assert len(hypotheses) == 3
        # Should still generate valid hypotheses
        assert all(h.novelty_score >= 0 for h in hypotheses)
    
    def test_algorithm_discovery_edge_cases(self):
        """Test algorithm discovery edge cases."""
        discovery = NovelAlgorithmDiscovery()
        
        # Test with zero algorithms requested
        algorithms = discovery.discover_novel_algorithms(num_algorithms=0)
        assert len(algorithms) == 0
        
        # Test with large number of algorithms
        algorithms = discovery.discover_novel_algorithms(num_algorithms=100)
        assert len(algorithms) == 100
    
    def test_experimental_framework_robustness(self):
        """Test experimental framework robustness."""
        framework = ExperimentalFramework()
        
        # Test with empty algorithm
        algorithm = NovelAlgorithm(
            name='',
            description='',
            mathematical_formulation='',
            implementation='',
            theoretical_complexity='',
            empirical_performance={},
            biological_motivation='',
            comparative_analysis={},
            validation_results={},
            publication_readiness=0.0
        )
        
        # Should handle gracefully
        results = framework.run_validation_study(algorithm, ['test_dataset'])
        assert 'algorithm' in results
        assert results['algorithm'] == ''
    
    def test_quantum_state_edge_cases(self):
        """Test quantum state edge cases."""
        oracle = QuantumResearchOracle({})
        
        # Test with zero probability states
        quantum_state = {'concept_0': 0.0, 'concept_1': 0.0, 'concept_2': 0.0}
        
        # Should handle zero probabilities
        coherence = oracle._quantum_similarity(quantum_state, quantum_state)
        assert not np.isnan(coherence)


@pytest.fixture
def temp_config_file():
    """Create temporary configuration file for testing."""
    config = {
        'research_domains': ['test_domain'],
        'knowledge_base': {'test_domain': {'concept_1': 0.5}},
        'experimental_settings': {
            'num_hypotheses_per_domain': 2,
            'num_algorithms_to_discover': 2,
            'validation_datasets': ['test_dataset'],
            'statistical_significance_threshold': 0.05,
            'effect_size_threshold': 0.02
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        yield f.name
    
    Path(f.name).unlink()


class TestConfigurationManagement:
    """Test configuration management."""
    
    def test_default_config_loading(self):
        """Test loading default configuration."""
        engine = QuantumResearchEngine()
        config = engine.config
        
        assert 'research_domains' in config
        assert 'knowledge_base' in config
        assert 'experimental_settings' in config
        
        # Test default values
        assert len(config['research_domains']) > 0
        assert config['experimental_settings']['num_hypotheses_per_domain'] > 0
    
    def test_custom_config_loading(self, temp_config_file):
        """Test loading custom configuration."""
        engine = QuantumResearchEngine(config_path=temp_config_file)
        config = engine.config
        
        assert config['research_domains'] == ['test_domain']
        assert config['experimental_settings']['num_hypotheses_per_domain'] == 2


# Benchmarking and performance tests
class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.mark.benchmark
    def test_hypothesis_generation_benchmark(self, benchmark):
        """Benchmark hypothesis generation."""
        oracle = QuantumResearchOracle({})
        
        def generate_hypotheses():
            return oracle.generate_hypotheses('test_domain', num_hypotheses=10)
        
        result = benchmark(generate_hypotheses)
        assert len(result) == 10
    
    @pytest.mark.benchmark
    def test_algorithm_discovery_benchmark(self, benchmark):
        """Benchmark algorithm discovery."""
        discovery = NovelAlgorithmDiscovery()
        
        def discover_algorithms():
            return discovery.discover_novel_algorithms(num_algorithms=5)
        
        result = benchmark(discover_algorithms)
        assert len(result) == 5
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_research_cycle_benchmark(self, benchmark):
        """Benchmark full research cycle."""
        engine = QuantumResearchEngine()
        
        async def research_cycle():
            return await engine.execute_research_cycle('test_domain')
        
        result = await benchmark(research_cycle)
        assert 'domain' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])