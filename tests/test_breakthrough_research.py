"""
Comprehensive tests for breakthrough research functionality
Tests novel algorithms, validation, and publication readiness
"""

import pytest
import numpy as np
import torch
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Import breakthrough research modules
from scgraph_hub.breakthrough_research import (
    BreakthroughResearchEngine, 
    BreakthroughResult,
    BiologicallyInformedAttention,
    TemporalDynamicsGNN,
    MultiModalIntegrationGNN,
    get_breakthrough_research_engine,
    execute_breakthrough_research
)

from scgraph_hub.academic_validation import (
    AcademicValidator,
    StatisticalValidation,
    PeerReviewChecklist,
    get_academic_validator,
    validate_breakthrough_research
)

from scgraph_hub.publication_engine import (
    PublicationEngine,
    PublicationPackage,
    JournalRequirements,
    AdvancedFigureGenerator,
    get_publication_engine,
    generate_publication_package
)


class TestBreakthroughResearchEngine:
    """Test breakthrough research engine functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = BreakthroughResearchEngine(output_dir=self.temp_dir)
        
    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine.output_dir.exists()
        assert len(self.engine.novel_algorithms) == 3
        assert len(self.engine.baseline_algorithms) == 3
        assert 'BiologicallyInformedGNN' in self.engine.novel_algorithms
        assert 'TemporalDynamicsGNN' in self.engine.novel_algorithms
        assert 'MultiModalIntegrationGNN' in self.engine.novel_algorithms
        
    def test_novel_algorithm_creation(self):
        """Test creation of novel algorithms."""
        # Test BiologicallyInformedGNN creation
        bio_gnn = self.engine._create_bio_informed_gnn(100, 10)
        assert bio_gnn is not None
        
        # Test TemporalDynamicsGNN creation
        temporal_gnn = self.engine._create_temporal_gnn(100, 10)
        assert temporal_gnn is not None
        
        # Test MultiModalIntegrationGNN creation
        multimodal_gnn = self.engine._create_multimodal_gnn(
            100, 10, modality_dims={'rna': 50, 'protein': 50}
        )
        assert multimodal_gnn is not None
        
    def test_baseline_algorithm_creation(self):
        """Test creation of baseline algorithms."""
        gcn = self.engine._create_standard_gcn(100, 10)
        assert gcn is not None
        
        gat = self.engine._create_standard_gat(100, 10)
        assert gat is not None
        
        graphsage = self.engine._create_graphsage(100, 10)
        assert graphsage is not None
        
    def test_research_data_generation(self):
        """Test synthetic research data generation."""
        data = self.engine._generate_research_data('pbmc_10k', 'cell_type_prediction')
        
        assert 'x' in data
        assert 'edge_index' in data
        assert 'y' in data
        assert 'n_classes' in data
        
        assert data['x'].shape[0] == 1000  # n_cells for pbmc_10k
        assert data['x'].shape[1] == 2000  # n_genes for pbmc_10k
        assert data['edge_index'].shape[0] == 2  # [source, target]
        assert data['y'].shape[0] == 1000  # labels for all cells
        
    def test_method_evaluation_simulation(self):
        """Test method evaluation simulation."""
        # Create mock model
        mock_model = Mock()
        mock_model.__class__.__name__ = 'BiologicallyInformedGNN'
        
        # Generate test data
        data = self.engine._generate_research_data('pbmc_10k', 'cell_type_prediction')
        
        # Run evaluation
        metrics = self.engine._simulate_training_evaluation(
            mock_model, data, 'cell_type_prediction'
        )
        
        assert 'accuracy' in metrics
        assert 'f1_macro' in metrics
        assert 'training_time' in metrics
        assert 'inference_time' in metrics
        
        # Check metric ranges
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['f1_macro'] <= 1.0
        assert metrics['training_time'] > 0
        assert metrics['inference_time'] > 0
        
    def test_biological_validation(self):
        """Test biological validation methods."""
        mock_model = Mock()
        data = self.engine._generate_research_data('pbmc_10k', 'cell_type_prediction')
        
        validation_results = self.engine._conduct_biological_validation(mock_model, data)
        
        expected_keys = [
            'pathway_enrichment_score',
            'biological_coherence',
            'functional_annotation_score',
            'cross_species_conservation'
        ]
        
        for key in expected_keys:
            assert key in validation_results
            assert 0.0 <= validation_results[key] <= 1.0
            
    def test_computational_complexity_analysis(self):
        """Test computational complexity analysis."""
        mock_model = Mock()
        # Add parameters to mock model
        mock_param = Mock()
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]
        
        data = self.engine._generate_research_data('pbmc_10k', 'cell_type_prediction')
        
        complexity = self.engine._analyze_computational_complexity(mock_model, data)
        
        expected_keys = [
            'parameters',
            'memory_complexity',
            'time_complexity',
            'scalability_score',
            'gpu_memory_gb'
        ]
        
        for key in expected_keys:
            assert key in complexity
            
    def test_novel_insights_extraction(self):
        """Test novel insights extraction."""
        # Test for biological method
        bio_insights = self.engine._extract_novel_insights(
            'BiologicallyInformedGNN', 
            Mock(), 
            {}, 
            {'accuracy': 0.95}
        )
        assert len(bio_insights) > 0
        assert any('biological' in insight.lower() for insight in bio_insights)
        
        # Test for temporal method
        temporal_insights = self.engine._extract_novel_insights(
            'TemporalDynamicsGNN',
            Mock(),
            {},
            {'accuracy': 0.92}
        )
        assert len(temporal_insights) > 0
        assert any('temporal' in insight.lower() for insight in temporal_insights)
        
    def test_publication_readiness_calculation(self):
        """Test publication readiness score calculation."""
        metrics = {
            'accuracy': 0.95,
            'f1_score': 0.93,
            'training_time': 0.3
        }
        
        # Novel method with high performance
        score = self.engine._calculate_publication_readiness(
            metrics, is_novel=True, n_insights=5
        )
        assert 0.8 <= score <= 1.0
        
        # Baseline method with moderate performance
        score_baseline = self.engine._calculate_publication_readiness(
            {'accuracy': 0.82}, is_novel=False, n_insights=0
        )
        assert score > score_baseline
        
    @pytest.mark.asyncio
    async def test_algorithm_evaluation(self):
        """Test individual algorithm evaluation."""
        data = self.engine._generate_research_data('pbmc_10k', 'cell_type_prediction')
        
        result = await self.engine._evaluate_algorithm(
            'BiologicallyInformedGNN',
            self.engine._create_bio_informed_gnn,
            data,
            'pbmc_10k',
            'cell_type_prediction',
            is_novel=True
        )
        
        assert isinstance(result, BreakthroughResult)
        assert result.algorithm_name == 'BiologicallyInformedGNN'
        assert result.dataset == 'pbmc_10k'
        assert result.task_type == 'cell_type_prediction'
        assert result.performance_metrics
        assert result.biological_validation
        assert result.computational_complexity
        assert len(result.novel_insights) > 0
        assert result.publication_readiness > 0.0


class TestBiologicallyInformedAttention:
    """Test biologically-informed attention mechanism."""
    
    def test_initialization(self):
        """Test attention mechanism initialization."""
        attention = BiologicallyInformedAttention(
            input_dim=100, 
            output_dim=64, 
            attention_heads=4
        )
        
        assert attention.input_dim == 100
        assert attention.output_dim == 64
        assert attention.num_heads == 4
        
    def test_forward_pass(self):
        """Test forward pass through attention mechanism."""
        attention = BiologicallyInformedAttention(
            input_dim=100, 
            output_dim=64, 
            attention_heads=4
        )
        
        # Create test input
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 100)
        
        # Forward pass
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, 64)
        assert not torch.isnan(output).any()
        
    def test_biological_prior_computation(self):
        """Test biological prior computation."""
        attention = BiologicallyInformedAttention(
            input_dim=100, 
            output_dim=64, 
            attention_heads=4
        )
        
        mask = torch.ones(2, 10, 10)
        prior = attention._compute_biological_prior(mask)
        
        expected_shape = (2, 4, 10, 10)  # (batch, heads, seq, seq)
        assert prior.shape == expected_shape


class TestTemporalDynamicsGNN:
    """Test temporal dynamics GNN."""
    
    def test_initialization(self):
        """Test temporal GNN initialization."""
        gnn = TemporalDynamicsGNN(
            input_dim=100,
            hidden_dim=64,
            output_dim=32,
            num_time_steps=5
        )
        
        assert gnn.input_dim == 100
        assert gnn.hidden_dim == 64
        assert gnn.output_dim == 32
        assert gnn.num_time_steps == 5
        
    def test_forward_pass(self):
        """Test forward pass through temporal GNN."""
        gnn = TemporalDynamicsGNN(
            input_dim=100,
            hidden_dim=64,
            output_dim=32
        )
        
        # Create test data
        n_nodes = 50
        x = torch.randn(n_nodes, 100)
        edge_index = torch.randint(0, n_nodes, (2, 200))
        temporal_states = torch.randn(n_nodes, 5, 100)
        
        output = gnn(x, edge_index, temporal_states)
        
        assert output.shape == (n_nodes, 32)
        assert not torch.isnan(output).any()


class TestMultiModalIntegrationGNN:
    """Test multi-modal integration GNN."""
    
    def test_initialization(self):
        """Test multi-modal GNN initialization."""
        modality_dims = {'rna': 100, 'protein': 50}
        gnn = MultiModalIntegrationGNN(
            modality_dims=modality_dims,
            hidden_dim=64,
            output_dim=32
        )
        
        assert gnn.modality_dims == modality_dims
        assert gnn.hidden_dim == 64
        assert gnn.output_dim == 32
        
    def test_forward_pass(self):
        """Test forward pass through multi-modal GNN."""
        modality_dims = {'rna': 100, 'protein': 50}
        gnn = MultiModalIntegrationGNN(
            modality_dims=modality_dims,
            hidden_dim=64,
            output_dim=32
        )
        
        # Create test data
        n_nodes = 50
        modality_data = {
            'rna': torch.randn(n_nodes, 100),
            'protein': torch.randn(n_nodes, 50)
        }
        edge_index = torch.randint(0, n_nodes, (2, 200))
        
        output = gnn(modality_data, edge_index)
        
        assert output.shape == (n_nodes, 32)
        assert not torch.isnan(output).any()


class TestAcademicValidator:
    """Test academic validation framework."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = AcademicValidator(output_dir=self.temp_dir)
        
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert self.validator.output_dir.exists()
        assert len(self.validator.statistical_tests) == 3
        assert 'parametric' in self.validator.statistical_tests
        assert 'nonparametric' in self.validator.statistical_tests
        assert 'categorical' in self.validator.statistical_tests
        
    def test_metrics_extraction(self):
        """Test metrics extraction from results."""
        results = {
            'accuracy': 0.95,
            'f1_score': 0.92,
            'nested': {
                'precision': 0.90,
                'recall': 0.88
            }
        }
        
        metrics = self.validator._extract_metrics(results)
        
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'nested_precision' in metrics
        assert 'nested_recall' in metrics
        
    def test_statistical_tests(self):
        """Test individual statistical tests."""
        group1 = np.random.normal(0.9, 0.05, 30)
        group2 = np.random.normal(0.8, 0.05, 30)
        
        # Test t-test
        result = self.validator._independent_ttest(group1, group2)
        assert isinstance(result, StatisticalValidation)
        assert result.test_name == "Independent t-test"
        assert result.p_value < 0.05  # Should be significant
        assert abs(result.effect_size) > 0.5  # Should be large effect
        
        # Test Mann-Whitney U
        result_mw = self.validator._mann_whitney_u(group1, group2)
        assert isinstance(result_mw, StatisticalValidation)
        assert result_mw.test_name == "Mann-Whitney U test"
        
    def test_effect_size_calculations(self):
        """Test effect size calculations."""
        group1 = np.array([0.95, 0.93, 0.91, 0.94])
        group2 = np.array([0.85, 0.82, 0.84, 0.83])
        
        # Cohen's d
        cohens_d = self.validator._cohens_d(group1, group2)
        assert cohens_d > 1.0  # Large effect size
        
        # Glass's delta
        glass_delta = self.validator._glass_delta(group1, group2)
        assert glass_delta > 0.5
        
        # Hedges' g
        hedges_g = self.validator._hedges_g(group1, group2)
        assert abs(hedges_g - cohens_d) < 0.1  # Should be similar
        
    def test_power_analysis(self):
        """Test statistical power analysis."""
        effect_size = 0.8
        n1, n2 = 30, 30
        
        power = self.validator._calculate_power(effect_size, n1, n2)
        assert 0.8 <= power <= 1.0  # Should be well-powered
        
        required_n = self.validator._sample_size_for_power(effect_size)
        assert required_n > 10  # Should be reasonable
        
        min_effect = self.validator._minimum_detectable_effect(n1, n2)
        assert min_effect > 0.0
        
    def test_reproducibility_assessment(self):
        """Test reproducibility assessment."""
        metadata = {
            'data_publicly_available': True,
            'code_publicly_available': True,
            'random_seeds_fixed': True,
            'environment_documented': True
        }
        
        data_score = self.validator._assess_data_availability(metadata)
        code_score = self.validator._assess_code_availability(metadata)
        seed_score = self.validator._assess_random_seeds(metadata)
        env_score = self.validator._assess_computational_environment(metadata)
        
        assert data_score >= 0.5
        assert code_score >= 0.4
        assert seed_score == 1.0  # Perfect score for fixed seeds
        assert env_score >= 0.4
        
    @pytest.mark.asyncio
    async def test_comprehensive_validation(self):
        """Test comprehensive research validation."""
        experimental_results = {
            'method1': {'accuracy': [0.95, 0.93, 0.94], 'f1': [0.92, 0.91, 0.93]},
            'method2': {'accuracy': [0.92, 0.90, 0.91], 'f1': [0.89, 0.88, 0.90]}
        }
        
        baseline_results = {
            'baseline': {'accuracy': [0.85, 0.83, 0.84], 'f1': [0.82, 0.81, 0.83]}
        }
        
        metadata = {
            'data_publicly_available': True,
            'code_publicly_available': True,
            'random_seeds_fixed': True
        }
        
        report = await self.validator.validate_research_results(
            experimental_results, baseline_results, metadata
        )
        
        assert 'statistical_validations' in report
        assert 'effect_size_analysis' in report
        assert 'power_analysis' in report
        assert 'reproducibility_assessment' in report
        assert 'peer_review_readiness' in report
        assert 'recommendations' in report


class TestPeerReviewChecklist:
    """Test peer review readiness assessment."""
    
    def test_checklist_initialization(self):
        """Test checklist initialization."""
        checklist = PeerReviewChecklist(
            methodology_clarity=0.9,
            statistical_rigor=0.85,
            reproducibility_score=0.95,
            novelty_assessment=0.8,
            biological_relevance=0.9,
            writing_quality=0.8,
            figure_quality=0.85,
            ethical_considerations=0.95,
            data_availability=0.9,
            code_availability=0.9
        )
        
        assert 0.85 <= checklist.overall_readiness <= 0.95
        assert checklist.readiness_category in [
            'publication_ready', 'minor_revisions', 'major_revisions'
        ]
        
    def test_readiness_categories(self):
        """Test readiness categorization."""
        # High readiness
        high_checklist = PeerReviewChecklist(
            **{field: 0.95 for field in [
                'methodology_clarity', 'statistical_rigor', 'reproducibility_score',
                'novelty_assessment', 'biological_relevance', 'writing_quality',
                'figure_quality', 'ethical_considerations', 'data_availability',
                'code_availability'
            ]}
        )
        assert high_checklist.readiness_category == 'publication_ready'
        
        # Low readiness
        low_checklist = PeerReviewChecklist(
            **{field: 0.6 for field in [
                'methodology_clarity', 'statistical_rigor', 'reproducibility_score',
                'novelty_assessment', 'biological_relevance', 'writing_quality',
                'figure_quality', 'ethical_considerations', 'data_availability',
                'code_availability'
            ]}
        )
        assert low_checklist.readiness_category == 'needs_substantial_work'


class TestPublicationEngine:
    """Test publication engine functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = PublicationEngine(output_dir=self.temp_dir)
        
    def test_engine_initialization(self):
        """Test publication engine initialization."""
        assert self.engine.output_dir.exists()
        assert len(self.engine.journals) > 0
        assert 'nature' in self.engine.journals
        assert 'nature_methods' in self.engine.journals
        
    def test_journal_requirements(self):
        """Test journal requirements structure."""
        nature_req = self.engine.journals['nature']
        
        assert isinstance(nature_req, JournalRequirements)
        assert nature_req.impact_factor > 40
        assert nature_req.is_high_impact
        assert 'abstract' in nature_req.word_limits
        assert 'main' in nature_req.figure_limits
        
    def test_title_generation(self):
        """Test manuscript title generation."""
        research_results = {
            'results': [
                {
                    'algorithm_name': 'BiologicallyInformedGNN',
                    'performance_metrics': {'accuracy': 0.95}
                }
            ]
        }
        
        title = self.engine._generate_title(research_results)
        
        assert len(title) > 10
        assert isinstance(title, str)
        assert any(word in title.lower() for word in [
            'graph', 'neural', 'network', 'single', 'cell'
        ])
        
    def test_abstract_generation(self):
        """Test abstract generation."""
        research_results = {
            'results': [
                {
                    'algorithm_name': 'BiologicallyInformedGNN',
                    'performance_metrics': {'accuracy': 0.95, 'f1_score': 0.93}
                }
            ]
        }
        
        validation_results = {
            'reproducibility_assessment': {'overall_reproducibility_score': 0.95}
        }
        
        journal_req = self.engine.journals['nature_methods']
        
        abstract = self.engine._generate_abstract(
            research_results, validation_results, journal_req
        )
        
        assert len(abstract) > 100
        assert len(abstract.split()) <= journal_req.word_limits['abstract']
        
    def test_figure_generation_setup(self):
        """Test figure generator setup."""
        fig_gen = AdvancedFigureGenerator(
            self.engine.output_dir / "test_figures"
        )
        
        assert fig_gen.output_dir.exists()
        assert fig_gen.style == "nature"
        assert len(fig_gen.color_palettes) > 0
        
    def test_manuscript_section_creation(self):
        """Test manuscript section creation."""
        research_results = {'results': []}
        validation_results = {'biological_validation': {}}
        figures = []
        tables = []
        journal_req = self.engine.journals['nature_methods']
        
        sections = asyncio.run(
            self.engine._generate_manuscript_sections(
                research_results, validation_results, figures, tables, journal_req
            )
        )
        
        expected_sections = ['abstract', 'introduction', 'results', 'discussion', 'methods']
        for section in expected_sections:
            assert section in sections
            assert sections[section].word_count > 0
            
    def test_cover_letter_generation(self):
        """Test cover letter generation."""
        journal_req = self.engine.journals['nature']
        manuscript_sections = {}
        
        cover_letter = self.engine._generate_cover_letter(journal_req, manuscript_sections)
        
        assert len(cover_letter) > 200
        assert 'Nature' in cover_letter
        assert 'Editor' in cover_letter
        
    def test_submission_checklist(self):
        """Test submission checklist creation."""
        journal_req = self.engine.journals['nature_methods']
        checklist = self.engine._create_submission_checklist(journal_req)
        
        expected_items = [
            'manuscript_formatted', 'abstract_within_limits', 'figures_high_resolution',
            'tables_formatted', 'cover_letter_included', 'code_available'
        ]
        
        for item in expected_items:
            assert item in checklist
            assert isinstance(checklist[item], bool)
            
    def test_publication_readiness_calculation(self):
        """Test publication readiness score calculation."""
        manuscript_sections = {
            'abstract': Mock(word_count=200),
            'introduction': Mock(word_count=800),
            'results': Mock(word_count=2000)
        }
        
        figures = [Mock() for _ in range(3)]
        tables = [Mock() for _ in range(2)]
        journal_req = self.engine.journals['nature_methods']
        
        score = self.engine._calculate_publication_readiness(
            manuscript_sections, figures, tables, journal_req
        )
        
        assert 0.7 <= score <= 1.0


class TestIntegrationWorkflows:
    """Test integrated research workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_research_workflow(self):
        """Test complete research workflow from algorithms to publication."""
        # This would be a comprehensive integration test
        # For now, we'll test the main function
        
        with patch('scgraph_hub.breakthrough_research.BreakthroughResearchEngine') as mock_engine:
            # Mock the research results
            mock_engine.return_value.conduct_breakthrough_research.return_value = [
                BreakthroughResult(
                    algorithm_name='BiologicallyInformedGNN',
                    dataset='pbmc_10k',
                    task_type='cell_type_prediction',
                    performance_metrics={'accuracy': 0.95},
                    baseline_comparison={},
                    statistical_significance={},
                    biological_validation={'pathway_enrichment': 0.9},
                    computational_complexity={'parameters': 1000000},
                    reproducibility_score=0.95,
                    novel_insights=['Biological priors improve performance'],
                    publication_readiness=0.9
                )
            ]
            
            # Execute research
            results = await execute_breakthrough_research()
            
            assert results['research_completed']
            assert results['breakthrough_achieved']
            assert results['publication_ready'] > 0
            
    @pytest.mark.asyncio
    async def test_validation_workflow(self):
        """Test validation workflow."""
        experimental_results = {
            'novel_method': {'accuracy': 0.95, 'f1': 0.93}
        }
        
        baseline_results = {
            'baseline_method': {'accuracy': 0.85, 'f1': 0.83}
        }
        
        metadata = {
            'data_publicly_available': True,
            'code_publicly_available': True
        }
        
        validation_report = await validate_breakthrough_research(
            experimental_results, baseline_results, metadata
        )
        
        assert validation_report is not None
        # Additional assertions would depend on mock implementations
        
    @pytest.mark.asyncio
    async def test_publication_workflow(self):
        """Test publication generation workflow."""
        research_results = {
            'results': [
                {
                    'algorithm_name': 'BiologicallyInformedGNN',
                    'performance_metrics': {'accuracy': 0.95}
                }
            ]
        }
        
        validation_results = {
            'reproducibility_assessment': {'overall_reproducibility_score': 0.95}
        }
        
        # This would normally generate a full publication package
        # For testing, we'll just test the function call
        with patch('scgraph_hub.publication_engine.PublicationEngine') as mock_engine:
            mock_package = PublicationPackage(
                manuscript={}, figures=[], tables=[], supplementary={},
                cover_letter="", publication_readiness_score=0.9
            )
            mock_engine.return_value.generate_full_publication.return_value = mock_package
            
            package = await generate_publication_package(
                research_results, validation_results, 'nature_methods'
            )
            
            assert package.publication_readiness_score == 0.9


# Fixtures and utilities for testing
@pytest.fixture
def sample_research_data():
    """Sample research data for testing."""
    return {
        'x': torch.randn(100, 50),  # 100 cells, 50 genes
        'edge_index': torch.randint(0, 100, (2, 300)),  # 300 edges
        'y': torch.randint(0, 5, (100,)),  # 5 cell types
        'n_classes': 5
    }


@pytest.fixture
def sample_breakthrough_result():
    """Sample breakthrough result for testing."""
    return BreakthroughResult(
        algorithm_name='TestGNN',
        dataset='test_dataset',
        task_type='classification',
        performance_metrics={'accuracy': 0.90, 'f1_score': 0.88},
        baseline_comparison={'baseline_accuracy': 0.80},
        statistical_significance={'p_value': 0.001},
        biological_validation={'pathway_enrichment': 0.85},
        computational_complexity={'parameters': 1000000},
        reproducibility_score=0.95,
        novel_insights=['Test insight 1', 'Test insight 2'],
        publication_readiness=0.88
    )


# Performance and stress tests
@pytest.mark.performance
class TestPerformance:
    """Performance tests for research workflows."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Test with larger synthetic dataset
        engine = BreakthroughResearchEngine()
        
        # Generate large dataset
        large_data = engine._generate_research_data('brain_atlas', 'cell_type_prediction')
        
        assert large_data['x'].shape[0] == 2000  # 2k cells for brain_atlas
        assert large_data['x'].shape[1] == 3000  # 3k genes
        
        # Ensure memory usage is reasonable
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Run evaluation
        result = await engine._evaluate_algorithm(
            'BiologicallyInformedGNN',
            engine._create_bio_informed_gnn,
            large_data,
            'brain_atlas',
            'cell_type_prediction'
        )
        
        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (< 500MB for test)
        assert memory_increase < 500
        assert result is not None
        
    def test_concurrent_algorithm_evaluation(self):
        """Test concurrent evaluation of multiple algorithms."""
        import concurrent.futures
        import time
        
        engine = BreakthroughResearchEngine()
        data = engine._generate_research_data('pbmc_10k', 'cell_type_prediction')
        
        def evaluate_algorithm(alg_name, creator):
            return asyncio.run(engine._evaluate_algorithm(
                alg_name, creator, data, 'pbmc_10k', 'cell_type_prediction'
            ))
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(evaluate_algorithm, name, creator)
                for name, creator in engine.novel_algorithms.items()
            ]
            
            results = [future.result() for future in futures]
        
        end_time = time.time()
        
        assert len(results) == 3
        assert all(isinstance(result, BreakthroughResult) for result in results)
        
        # Concurrent execution should be faster than sequential
        # (This is a rough check and may vary by system)
        assert end_time - start_time < 10  # Should complete within 10 seconds


if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([
        __file__,
        '-v',
        '--cov=scgraph_hub.breakthrough_research',
        '--cov=scgraph_hub.academic_validation', 
        '--cov=scgraph_hub.publication_engine',
        '--cov-report=html',
        '--cov-report=term-missing'
    ])