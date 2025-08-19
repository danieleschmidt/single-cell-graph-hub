"""
Test suite for Experimental Baseline Framework v4.0
Comprehensive testing of baseline establishment and comparison capabilities
"""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import asdict

from src.scgraph_hub.experimental_baseline_framework import (
    ComprehensiveBaselineFramework,
    BaselineEvaluator,
    ExperimentalDesigner,
    StatisticalAnalyzer,
    BaselineResult,
    ExperimentalProtocol,
    BaselineGCN,
    BaselineGAT,
    BaselineGraphSAGE,
    BiologicallyInformedBaseline
)


class TestBaselineModels:
    """Test baseline model implementations."""
    
    def test_baseline_gcn_initialization(self):
        """Test BaselineGCN initialization."""
        model = BaselineGCN(
            input_dim=100,
            hidden_dim=64,
            output_dim=10,
            num_layers=3,
            dropout=0.2
        )
        
        assert len(model.convs) == 3
        assert model.num_layers == 3
        assert model.dropout == 0.2
    
    def test_baseline_gcn_forward_pass(self):
        """Test BaselineGCN forward pass."""
        model = BaselineGCN(input_dim=100, hidden_dim=64, output_dim=10)
        
        # Create test data
        x = torch.randn(50, 100)
        edge_index = torch.randint(0, 50, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (50, 10)
        assert not torch.isnan(output).any()
    
    def test_baseline_gat_initialization(self):
        """Test BaselineGAT initialization."""
        model = BaselineGAT(
            input_dim=100,
            hidden_dim=64,
            output_dim=10,
            num_layers=3,
            heads=4,
            dropout=0.2
        )
        
        assert len(model.convs) == 3
        assert model.heads == 4
        assert model.dropout == 0.2
    
    def test_baseline_gat_forward_pass(self):
        """Test BaselineGAT forward pass."""
        model = BaselineGAT(input_dim=100, hidden_dim=64, output_dim=10, heads=4)
        
        x = torch.randn(50, 100)
        edge_index = torch.randint(0, 50, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (50, 10)
        assert not torch.isnan(output).any()
    
    def test_baseline_graphsage_initialization(self):
        """Test BaselineGraphSAGE initialization."""
        model = BaselineGraphSAGE(
            input_dim=100,
            hidden_dim=64,
            output_dim=10,
            num_layers=3,
            dropout=0.2,
            aggregator='mean'
        )
        
        assert len(model.convs) == 3
        assert model.dropout == 0.2
    
    def test_baseline_graphsage_forward_pass(self):
        """Test BaselineGraphSAGE forward pass."""
        model = BaselineGraphSAGE(input_dim=100, hidden_dim=64, output_dim=10)
        
        x = torch.randn(50, 100)
        edge_index = torch.randint(0, 50, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (50, 10)
        assert not torch.isnan(output).any()
    
    def test_biologically_informed_baseline(self):
        """Test BiologicallyInformedBaseline."""
        model = BiologicallyInformedBaseline(
            input_dim=100,
            hidden_dim=64,
            output_dim=10,
            num_pathways=50,
            pathway_dim=32
        )
        
        x = torch.randn(50, 100)
        edge_index = torch.randint(0, 50, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (50, 10)
        assert not torch.isnan(output).any()
        
        # Test pathway components
        assert hasattr(model, 'pathway_encoder')
        assert hasattr(model, 'pathway_attention')
        assert hasattr(model, 'pathway_matrix')
        assert model.pathway_matrix.shape == (50, 100)


class TestBaselineEvaluator:
    """Test baseline evaluation system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.evaluator = BaselineEvaluator(device='cpu')
        
        # Create mock dataset
        self.mock_dataset = Mock()
        self.mock_dataset.name = 'test_dataset'
        self.mock_dataset.num_node_features = 100
        self.mock_dataset.num_classes = 10
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        assert 'GCN' in self.evaluator.baseline_models
        assert 'GAT' in self.evaluator.baseline_models
        assert 'GraphSAGE' in self.evaluator.baseline_models
        assert 'BiologicalGNN' in self.evaluator.baseline_models
        
        assert isinstance(self.evaluator.results_cache, dict)
        assert isinstance(self.evaluator.evaluation_history, list)
    
    def test_default_hyperparameters(self):
        """Test default hyperparameter generation."""
        gcn_params = self.evaluator._get_default_hyperparameters('GCN')
        
        assert 'hidden_dim' in gcn_params
        assert 'num_layers' in gcn_params
        assert 'dropout' in gcn_params
        assert 'learning_rate' in gcn_params
        assert 'epochs' in gcn_params
        
        assert gcn_params['hidden_dim'] > 0
        assert gcn_params['num_layers'] > 0
        assert 0 <= gcn_params['dropout'] <= 1
        assert gcn_params['learning_rate'] > 0
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        hyperparams = {'hidden_dim': 128, 'dropout': 0.2}
        
        key1 = self.evaluator._generate_cache_key('GCN', 'dataset1', hyperparams, 5)
        key2 = self.evaluator._generate_cache_key('GCN', 'dataset1', hyperparams, 5)
        key3 = self.evaluator._generate_cache_key('GAT', 'dataset1', hyperparams, 5)
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different models should generate different keys
        assert key1 != key3
        
        # Keys should be valid hex strings
        assert all(c in '0123456789abcdef' for c in key1)
    
    def test_baseline_evaluation_structure(self):
        """Test baseline evaluation result structure."""
        result = self.evaluator.evaluate_baseline(
            model_name='GCN',
            dataset=self.mock_dataset,
            task_type='classification',
            cv_folds=3,
            random_seeds=[42, 123]
        )
        
        assert isinstance(result, BaselineResult)
        assert result.model_name == 'GCN'
        assert result.dataset_name == 'test_dataset'
        assert result.task_type == 'classification'
        
        # Check metrics structure
        assert isinstance(result.metrics, dict)
        assert 'accuracy' in result.metrics
        assert 'f1_macro' in result.metrics
        
        # Check timing information
        assert result.training_time >= 0
        assert result.inference_time >= 0
        assert result.memory_usage >= 0
        
        # Check reproducibility information
        assert isinstance(result.cross_validation_scores, list)
        assert len(result.cross_validation_scores) == 5  # Default CV folds
        assert isinstance(result.confidence_intervals, dict)
        assert 0 <= result.reproducibility_score <= 1
    
    def test_baseline_evaluation_caching(self):
        """Test baseline evaluation caching."""
        # First evaluation
        result1 = self.evaluator.evaluate_baseline(
            model_name='GCN',
            dataset=self.mock_dataset,
            task_type='classification',
            hyperparameters={'hidden_dim': 64}
        )
        
        # Second evaluation with same parameters (should use cache)
        result2 = self.evaluator.evaluate_baseline(
            model_name='GCN',
            dataset=self.mock_dataset,
            task_type='classification',
            hyperparameters={'hidden_dim': 64}
        )
        
        # Results should be identical (from cache)
        assert result1.task_id == result2.task_id if hasattr(result1, 'task_id') else True
        assert result1.model_name == result2.model_name
        assert result1.metrics == result2.metrics
    
    def test_cross_validation_scores(self):
        """Test cross-validation score generation."""
        cv_scores = self.evaluator._cross_validate(
            model_name='GCN',
            dataset=self.mock_dataset,
            task_type='classification',
            hyperparameters={},
            seed=42
        )
        
        assert isinstance(cv_scores, list)
        assert len(cv_scores) == 5  # Default 5-fold CV
        assert all(isinstance(score, float) for score in cv_scores)
        assert all(0 <= score <= 1 for score in cv_scores)
    
    def test_metric_aggregation(self):
        """Test metric aggregation across seeds."""
        seed_results = [
            {'accuracy': 0.85, 'f1_score': 0.82},
            {'accuracy': 0.87, 'f1_score': 0.84},
            {'accuracy': 0.86, 'f1_score': 0.83}
        ]
        
        aggregated = self.evaluator._aggregate_seed_results(seed_results)
        
        assert 'accuracy' in aggregated
        assert 'f1_score' in aggregated
        assert 'accuracy_std' in aggregated
        assert 'f1_score_std' in aggregated
        
        # Check accuracy aggregation
        expected_acc = np.mean([0.85, 0.87, 0.86])
        assert abs(aggregated['accuracy'] - expected_acc) < 1e-6
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        seed_results = [
            {'accuracy': 0.85},
            {'accuracy': 0.87},
            {'accuracy': 0.86},
            {'accuracy': 0.84},
            {'accuracy': 0.88}
        ]
        
        confidence_intervals = self.evaluator._calculate_confidence_intervals(seed_results)
        
        assert 'accuracy' in confidence_intervals
        ci_low, ci_high = confidence_intervals['accuracy']
        
        assert ci_low <= ci_high
        assert ci_low >= 0
        assert ci_high <= 1
        
        # CI should contain the mean
        mean_acc = np.mean([r['accuracy'] for r in seed_results])
        assert ci_low <= mean_acc <= ci_high
    
    def test_reproducibility_score_calculation(self):
        """Test reproducibility score calculation."""
        # Low variance results (high reproducibility)
        high_repro_results = [
            {'accuracy': 0.85, 'f1_score': 0.82},
            {'accuracy': 0.851, 'f1_score': 0.821},
            {'accuracy': 0.849, 'f1_score': 0.819}
        ]
        
        high_repro_score = self.evaluator._calculate_reproducibility_score(high_repro_results)
        
        # High variance results (low reproducibility)
        low_repro_results = [
            {'accuracy': 0.7, 'f1_score': 0.6},
            {'accuracy': 0.9, 'f1_score': 0.85},
            {'accuracy': 0.8, 'f1_score': 0.75}
        ]
        
        low_repro_score = self.evaluator._calculate_reproducibility_score(low_repro_results)
        
        assert 0 <= high_repro_score <= 1
        assert 0 <= low_repro_score <= 1
        assert high_repro_score > low_repro_score


class TestExperimentalDesigner:
    """Test experimental design functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.designer = ExperimentalDesigner()
    
    def test_comprehensive_experiment_design(self):
        """Test comprehensive experiment design."""
        protocol = self.designer.design_comprehensive_experiment(
            research_question="Test cell type classification",
            datasets=['dataset1', 'dataset2'],
            novel_methods=['NovelGNN']
        )
        
        assert isinstance(protocol, ExperimentalProtocol)
        assert protocol.name.startswith("Baseline Establishment:")
        assert "Test cell type classification" in protocol.description
        
        # Check datasets
        assert set(protocol.datasets) == {'dataset1', 'dataset2'}
        
        # Check methods include baselines + novel methods
        expected_methods = {'GCN', 'GAT', 'GraphSAGE', 'BiologicalGNN', 'NovelGNN'}
        assert set(protocol.baseline_models) == expected_methods
        
        # Check evaluation metrics
        assert 'accuracy' in protocol.evaluation_metrics
        assert 'f1_macro' in protocol.evaluation_metrics
        assert 'statistical_significance' in protocol.evaluation_metrics
        
        # Check statistical tests
        assert 'paired_t_test' in protocol.statistical_tests
        assert 'wilcoxon_signed_rank' in protocol.statistical_tests
        
        # Check quality controls
        assert 'cross_validation' in protocol.quality_controls
        assert 'multiple_random_seeds' in protocol.quality_controls
    
    def test_sample_size_calculation(self):
        """Test sample size calculation."""
        sample_calc = self.designer._calculate_sample_size(
            effect_size=0.1,
            power=0.8,
            alpha=0.05
        )
        
        assert 'effect_size' in sample_calc
        assert 'power' in sample_calc
        assert 'alpha' in sample_calc
        assert 'n_per_group' in sample_calc
        assert 'total_n' in sample_calc
        assert 'assumptions' in sample_calc
        
        assert sample_calc['effect_size'] == 0.1
        assert sample_calc['power'] == 0.8
        assert sample_calc['alpha'] == 0.05
        assert sample_calc['n_per_group'] > 0
        assert sample_calc['total_n'] == 2 * sample_calc['n_per_group']
    
    def test_protocol_id_uniqueness(self):
        """Test that protocol IDs are unique."""
        protocol1 = self.designer.design_comprehensive_experiment(
            "Question 1", ['dataset1']
        )
        protocol2 = self.designer.design_comprehensive_experiment(
            "Question 2", ['dataset2']
        )
        
        assert protocol1.protocol_id != protocol2.protocol_id
        assert protocol1.protocol_id.startswith('exp_')
        assert protocol2.protocol_id.startswith('exp_')


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.analyzer = StatisticalAnalyzer()
    
    def create_mock_results(self) -> list:
        """Create mock baseline results for testing."""
        results = []
        
        for i, model in enumerate(['GCN', 'GAT', 'GraphSAGE']):
            for j, dataset in enumerate(['dataset1', 'dataset2']):
                result = BaselineResult(
                    model_name=model,
                    dataset_name=dataset,
                    task_type='classification',
                    metrics={
                        'accuracy': 0.8 + i * 0.05 + np.random.normal(0, 0.02),
                        'f1_macro': 0.75 + i * 0.05 + np.random.normal(0, 0.02),
                        'f1_weighted': 0.78 + i * 0.05 + np.random.normal(0, 0.02)
                    },
                    training_time=100 + i * 20,
                    inference_time=5 + i,
                    memory_usage=1000 + i * 200,
                    hyperparameters={'hidden_dim': 128, 'dropout': 0.2},
                    cross_validation_scores=[
                        0.8 + i * 0.05 + np.random.normal(0, 0.01) for _ in range(5)
                    ],
                    confidence_intervals={
                        'accuracy': (0.78 + i * 0.05, 0.82 + i * 0.05)
                    },
                    statistical_significance={},
                    reproducibility_score=0.9 - i * 0.05,
                    timestamp='2024-01-01T00:00:00'
                )
                results.append(result)
        
        return results
    
    def test_method_comparison_structure(self):
        """Test method comparison result structure."""
        results = self.create_mock_results()
        comparison = self.analyzer.compare_methods(results)
        
        assert 'individual_comparisons' in comparison
        assert 'meta_analysis' in comparison
        assert 'recommendations' in comparison
        assert 'analysis_timestamp' in comparison
        
        # Check individual comparisons
        individual = comparison['individual_comparisons']
        expected_keys = ['dataset1_classification', 'dataset2_classification']
        
        for key in expected_keys:
            assert key in individual
            dataset_comparison = individual[key]
            
            assert 'pairwise_comparisons' in dataset_comparison
            assert 'ranking' in dataset_comparison
            assert 'effect_sizes' in dataset_comparison
            assert 'friedman_test' in dataset_comparison
            assert 'best_method' in dataset_comparison
    
    def test_pairwise_statistical_testing(self):
        """Test pairwise statistical testing."""
        # Create two mock results
        result1 = BaselineResult(
            model_name='GCN',
            dataset_name='test',
            task_type='classification',
            metrics={'accuracy': 0.85},
            training_time=100,
            inference_time=5,
            memory_usage=1000,
            hyperparameters={},
            cross_validation_scores=[0.84, 0.85, 0.86, 0.85, 0.84],
            confidence_intervals={},
            statistical_significance={},
            reproducibility_score=0.9,
            timestamp='2024-01-01T00:00:00'
        )
        
        result2 = BaselineResult(
            model_name='GAT',
            dataset_name='test',
            task_type='classification',
            metrics={'accuracy': 0.90},
            training_time=120,
            inference_time=6,
            memory_usage=1200,
            hyperparameters={},
            cross_validation_scores=[0.89, 0.90, 0.91, 0.90, 0.89],
            confidence_intervals={},
            statistical_significance={},
            reproducibility_score=0.85,
            timestamp='2024-01-01T00:00:00'
        )
        
        pairwise = self.analyzer._pairwise_statistical_test(result1, result2)
        
        assert pairwise['method1'] == 'GCN'
        assert pairwise['method2'] == 'GAT'
        assert 'mean_difference' in pairwise
        assert 'confidence_interval' in pairwise
        assert 'paired_t_test' in pairwise
        assert 'wilcoxon_test' in pairwise
        assert 'effect_size' in pairwise
        
        # Check t-test structure
        t_test = pairwise['paired_t_test']
        assert 'statistic' in t_test
        assert 'p_value' in t_test
        assert 'significant' in t_test
        assert isinstance(t_test['significant'], bool)
        
        # Check effect size
        effect_size = pairwise['effect_size']
        assert 'cohens_d' in effect_size
        assert 'magnitude' in effect_size
        assert effect_size['magnitude'] in ['negligible', 'small', 'medium', 'large']
    
    def test_cohens_d_interpretation(self):
        """Test Cohen's d interpretation."""
        assert self.analyzer._interpret_cohens_d(0.1) == 'negligible'
        assert self.analyzer._interpret_cohens_d(0.3) == 'small'
        assert self.analyzer._interpret_cohens_d(0.6) == 'medium'
        assert self.analyzer._interpret_cohens_d(1.0) == 'large'
        assert self.analyzer._interpret_cohens_d(-0.6) == 'medium'  # Absolute value
    
    def test_method_ranking(self):
        """Test method ranking functionality."""
        results = self.create_mock_results()
        
        # Get results for one dataset
        dataset_results = [r for r in results if r.dataset_name == 'dataset1']
        ranking = self.analyzer._rank_methods(dataset_results)
        
        assert len(ranking) == 3  # Three methods
        
        # Check ranking structure
        for rank_info in ranking:
            assert 'rank' in rank_info
            assert 'method' in rank_info
            assert 'composite_score' in rank_info
            assert 'main_metrics' in rank_info
            assert 'reproducibility' in rank_info
        
        # Check that ranking is sorted (best first)
        scores = [info['composite_score'] for info in ranking]
        assert scores == sorted(scores, reverse=True)
        
        # Check rank numbers
        ranks = [info['rank'] for info in ranking]
        assert ranks == [1, 2, 3]
    
    def test_friedman_test(self):
        """Test Friedman test implementation."""
        results = self.create_mock_results()
        
        # Get results for one dataset
        dataset_results = [r for r in results if r.dataset_name == 'dataset1']
        friedman_result = self.analyzer._friedman_test(dataset_results)
        
        assert 'statistic' in friedman_result
        assert 'p_value' in friedman_result
        assert 'significant' in friedman_result
        assert 'methods' in friedman_result
        assert 'interpretation' in friedman_result
        
        assert len(friedman_result['methods']) == 3
        assert isinstance(friedman_result['significant'], bool)
        assert friedman_result['statistic'] >= 0
        assert 0 <= friedman_result['p_value'] <= 1
    
    def test_meta_analysis(self):
        """Test meta-analysis across datasets."""
        comparison_results = {
            'dataset1_classification': {
                'ranking': [
                    {'method': 'GAT', 'composite_score': 0.9, 'consistency': 0.8},
                    {'method': 'GCN', 'composite_score': 0.85, 'consistency': 0.9},
                    {'method': 'GraphSAGE', 'composite_score': 0.8, 'consistency': 0.7}
                ]
            },
            'dataset2_classification': {
                'ranking': [
                    {'method': 'GCN', 'composite_score': 0.88, 'consistency': 0.85},
                    {'method': 'GAT', 'composite_score': 0.87, 'consistency': 0.82},
                    {'method': 'GraphSAGE', 'composite_score': 0.82, 'consistency': 0.75}
                ]
            }
        }
        
        meta_analysis = self.analyzer._overall_meta_analysis(comparison_results)
        
        assert 'overall_ranking' in meta_analysis
        assert 'most_consistent' in meta_analysis
        assert 'highest_performance' in meta_analysis
        assert 'performance_summary' in meta_analysis
        
        # Check overall ranking
        overall_ranking = meta_analysis['overall_ranking']
        assert len(overall_ranking) == 3  # Three methods
        
        for method_info in overall_ranking:
            assert 'method' in method_info
            assert 'mean_score' in method_info
            assert 'std_score' in method_info
            assert 'consistency' in method_info
            assert 'datasets_evaluated' in method_info
    
    def test_recommendation_generation(self):
        """Test recommendation generation."""
        comparison_results = {
            'dataset1_classification': {
                'ranking': [{'method': 'GAT', 'composite_score': 0.9}],
                'pairwise_comparisons': {
                    'GAT_vs_GCN': {'paired_t_test': {'significant': True}}
                }
            }
        }
        
        recommendations = self.analyzer._generate_recommendations(comparison_results)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestComprehensiveBaselineFramework:
    """Test the comprehensive baseline framework."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = ComprehensiveBaselineFramework(results_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        assert self.framework.evaluator is not None
        assert self.framework.designer is not None
        assert self.framework.analyzer is not None
        assert isinstance(self.framework.framework_results, dict)
        assert Path(self.framework.results_dir).exists()
    
    def test_baseline_establishment(self):
        """Test baseline establishment process."""
        research_question = "Test cell type classification"
        datasets = ['test_dataset_1', 'test_dataset_2']
        
        results = self.framework.establish_baselines(
            research_question=research_question,
            datasets=datasets,
            novel_methods=['TestGNN']
        )
        
        assert 'research_question' in results
        assert 'protocol' in results
        assert 'baseline_results' in results
        assert 'statistical_analysis' in results
        assert 'report' in results
        assert 'timestamp' in results
        
        assert results['research_question'] == research_question
        
        # Check protocol
        protocol = results['protocol']
        assert protocol['datasets'] == datasets
        assert 'TestGNN' in protocol['baseline_models']
        
        # Check baseline results
        baseline_results = results['baseline_results']
        assert len(baseline_results) > 0
        
        for result in baseline_results:
            assert 'model_name' in result
            assert 'dataset_name' in result
            assert 'metrics' in result
    
    def test_result_saving(self):
        """Test result saving functionality."""
        research_question = "Test saving"
        datasets = ['test_dataset']
        
        results = self.framework.establish_baselines(
            research_question=research_question,
            datasets=datasets
        )
        
        protocol_id = results['protocol']['protocol_id']
        
        # Check that files were created
        results_file = Path(self.framework.results_dir) / f"baselines_{protocol_id}.json"
        report_file = Path(self.framework.results_dir) / f"baseline_report_{protocol_id}.md"
        
        assert results_file.exists()
        assert report_file.exists()
        
        # Check file contents
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
            assert saved_results['research_question'] == research_question
        
        with open(report_file, 'r') as f:
            report_content = f.read()
            assert 'Baseline Establishment Report' in report_content
            assert research_question in report_content
    
    def test_report_generation(self):
        """Test baseline report generation."""
        # Create mock data
        protocol = ExperimentalProtocol(
            protocol_id='test_protocol',
            name='Test Protocol',
            description='Test Description',
            datasets=['dataset1', 'dataset2'],
            baseline_models=['GCN', 'GAT'],
            evaluation_metrics=['accuracy', 'f1_score'],
            statistical_tests=['t_test'],
            cross_validation_folds=5,
            random_seeds=[42, 123],
            significance_level=0.05,
            effect_size_threshold=0.02,
            sample_size_calculation={},
            quality_controls=['cross_validation']
        )
        
        results = [
            BaselineResult(
                model_name='GCN',
                dataset_name='dataset1',
                task_type='classification',
                metrics={'accuracy': 0.85, 'f1_macro': 0.82},
                training_time=100,
                inference_time=5,
                memory_usage=1000,
                hyperparameters={},
                cross_validation_scores=[0.84, 0.85, 0.86],
                confidence_intervals={},
                statistical_significance={},
                reproducibility_score=0.9,
                timestamp='2024-01-01T00:00:00'
            )
        ]
        
        analysis = {
            'individual_comparisons': {},
            'meta_analysis': {'overall_ranking': []},
            'recommendations': ['Test recommendation']
        }
        
        report = self.framework._generate_baseline_report(
            protocol, results, analysis, "Test Research Question"
        )
        
        assert isinstance(report, str)
        assert 'Baseline Establishment Report' in report
        assert 'Test Research Question' in report
        assert 'Executive Summary' in report
        assert 'Experimental Protocol' in report
        assert 'Results Summary' in report
        assert 'Technical Details' in report
        assert 'Test recommendation' in report


class TestIntegrationAndPerformance:
    """Integration and performance tests."""
    
    def test_end_to_end_baseline_establishment(self):
        """Test end-to-end baseline establishment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ComprehensiveBaselineFramework(results_dir=temp_dir)
            
            results = framework.establish_baselines(
                research_question="Integration test",
                datasets=['integration_dataset'],
                novel_methods=[]
            )
            
            # Verify complete results structure
            assert all(key in results for key in [
                'research_question', 'protocol', 'baseline_results',
                'statistical_analysis', 'report', 'timestamp'
            ])
            
            # Verify statistical analysis was performed
            analysis = results['statistical_analysis']
            assert 'individual_comparisons' in analysis
            assert 'meta_analysis' in analysis
            assert 'recommendations' in analysis
    
    def test_baseline_evaluation_performance(self):
        """Test baseline evaluation performance."""
        evaluator = BaselineEvaluator()
        
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.name = 'performance_test'
        mock_dataset.num_node_features = 1000
        mock_dataset.num_classes = 50
        
        import time
        start_time = time.time()
        
        # Evaluate multiple baselines
        for model_name in ['GCN', 'GAT', 'GraphSAGE']:
            result = evaluator.evaluate_baseline(
                model_name=model_name,
                dataset=mock_dataset,
                task_type='classification',
                cv_folds=3,
                random_seeds=[42, 123]
            )
            assert isinstance(result, BaselineResult)
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 30.0  # 30 seconds for 3 models
    
    def test_memory_efficiency(self):
        """Test memory efficiency of baseline framework."""
        import tracemalloc
        
        tracemalloc.start()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ComprehensiveBaselineFramework(results_dir=temp_dir)
            
            # Run baseline establishment
            framework.establish_baselines(
                research_question="Memory test",
                datasets=['memory_test_dataset']
            )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 500MB)
        assert peak < 500 * 1024 * 1024  # 500MB


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness."""
    
    def test_invalid_model_name(self):
        """Test handling of invalid model names."""
        evaluator = BaselineEvaluator()
        mock_dataset = Mock()
        mock_dataset.name = 'test'
        mock_dataset.num_node_features = 100
        mock_dataset.num_classes = 10
        
        # Should handle unknown model gracefully
        with pytest.raises(KeyError):
            evaluator.evaluate_baseline(
                model_name='UnknownModel',
                dataset=mock_dataset
            )
    
    def test_empty_dataset_list(self):
        """Test handling of empty dataset list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ComprehensiveBaselineFramework(results_dir=temp_dir)
            
            results = framework.establish_baselines(
                research_question="Empty dataset test",
                datasets=[]
            )
            
            # Should handle gracefully
            assert results['baseline_results'] == []
    
    def test_statistical_analysis_with_insufficient_data(self):
        """Test statistical analysis with insufficient data."""
        analyzer = StatisticalAnalyzer()
        
        # Single result (insufficient for comparison)
        single_result = [BaselineResult(
            model_name='GCN',
            dataset_name='test',
            task_type='classification',
            metrics={'accuracy': 0.85},
            training_time=100,
            inference_time=5,
            memory_usage=1000,
            hyperparameters={},
            cross_validation_scores=[0.85],
            confidence_intervals={},
            statistical_significance={},
            reproducibility_score=0.9,
            timestamp='2024-01-01T00:00:00'
        )]
        
        comparison = analyzer.compare_methods(single_result)
        
        # Should return error for insufficient data
        assert 'error' in comparison
        assert 'Need at least 2 methods' in comparison['error']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])