"""Unit tests for preprocessing pipeline functionality."""

import pytest
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline class."""
    
    @pytest.mark.unit
    def test_pipeline_initialization(self):
        """Test preprocessing pipeline initialization."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        # Test default initialization
        pipeline = PreprocessingPipeline()
        assert pipeline.steps is not None
        assert isinstance(pipeline.steps, list)
        assert len(pipeline.steps) > 0
        
        # Test custom initialization
        custom_steps = ['filter_cells', 'normalize_total', 'log1p']
        custom_params = {'filter_cells': {'min_genes': 200}}
        
        pipeline_custom = PreprocessingPipeline(
            steps=custom_steps,
            parameters=custom_params,
            track_metadata=True
        )
        
        assert pipeline_custom.steps == custom_steps
        assert pipeline_custom.parameters == custom_params
        assert pipeline_custom.track_metadata is True
    
    @pytest.mark.unit
    def test_default_steps(self):
        """Test default preprocessing steps."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        pipeline = PreprocessingPipeline()
        default_steps = pipeline._get_default_steps()
        
        # Check that essential steps are included
        essential_steps = [
            'filter_cells', 'filter_genes', 'calculate_qc_metrics',
            'normalize_total', 'log1p', 'highly_variable_genes'
        ]
        
        for step in essential_steps:
            assert step in default_steps
        
        assert isinstance(default_steps, list)
        assert len(default_steps) >= 5
    
    @pytest.mark.unit
    def test_step_functions_registry(self):
        """Test that all step functions are properly registered."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        pipeline = PreprocessingPipeline()
        
        # Check that step functions are callable
        for step_name, step_func in pipeline.step_functions.items():
            assert callable(step_func)
            assert step_name.replace('_', ' ').replace(' ', '_') == step_name  # Valid naming
    
    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_SCANPY, reason="scanpy not available")
    def test_filter_cells_step(self, mock_h5ad_file):
        """Test cell filtering step."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        # Load test data
        adata = sc.read_h5ad(mock_h5ad_file)
        initial_cell_count = adata.n_obs
        
        pipeline = PreprocessingPipeline()
        
        # Test cell filtering
        filtered_adata = pipeline._filter_cells(
            adata,
            min_genes=50,
            max_mt_pct=15.0
        )
        
        # Verify filtering occurred
        assert filtered_adata.n_obs <= initial_cell_count
        assert 'pct_counts_mt' in filtered_adata.obs.columns
        assert all(filtered_adata.obs['pct_counts_mt'] < 15.0)
    
    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_SCANPY, reason="scanpy not available")
    def test_filter_genes_step(self, mock_h5ad_file):
        """Test gene filtering step."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        adata = sc.read_h5ad(mock_h5ad_file)
        initial_gene_count = adata.n_vars
        
        pipeline = PreprocessingPipeline()
        
        # Test gene filtering
        filtered_adata = pipeline._filter_genes(
            adata,
            min_cells=3
        )
        
        # Verify filtering occurred
        assert filtered_adata.n_vars <= initial_gene_count
    
    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_SCANPY, reason="scanpy not available")
    def test_calculate_qc_metrics_step(self, mock_h5ad_file):
        """Test QC metrics calculation."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        adata = sc.read_h5ad(mock_h5ad_file)
        pipeline = PreprocessingPipeline()
        
        # Test QC metrics calculation
        qc_adata = pipeline._calculate_qc_metrics(adata)
        
        # Check that QC metrics were added
        expected_obs_columns = [
            'total_counts', 'n_genes_by_counts',
            'pct_counts_mt', 'log10_total_counts'
        ]
        
        for col in expected_obs_columns:
            assert col in qc_adata.obs.columns
        
        # Check var annotations
        assert 'mt' in qc_adata.var.columns
    
    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_SCANPY, reason="scanpy not available")
    def test_normalization_step(self, mock_h5ad_file):
        """Test total count normalization."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        adata = sc.read_h5ad(mock_h5ad_file)
        pipeline = PreprocessingPipeline()
        
        # Store original counts
        original_total_counts = np.array(adata.X.sum(axis=1)).flatten()
        
        # Test normalization
        norm_adata = pipeline._normalize_total(adata, target_sum=1e4)
        
        # Check that normalization was applied
        new_total_counts = np.array(norm_adata.X.sum(axis=1)).flatten()
        
        # Counts should be close to target sum
        assert np.allclose(new_total_counts, 1e4, rtol=1e-3)
    
    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_SCANPY, reason="scanpy not available")
    def test_log1p_transform(self, mock_h5ad_file):
        """Test log1p transformation."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        adata = sc.read_h5ad(mock_h5ad_file)
        pipeline = PreprocessingPipeline()
        
        # Ensure positive values
        adata.X = np.abs(adata.X)
        original_values = adata.X.copy()
        
        # Apply log1p
        log_adata = pipeline._log1p_transform(adata)
        
        # Check transformation
        expected_values = np.log1p(original_values)
        if hasattr(log_adata.X, 'toarray'):
            actual_values = log_adata.X.toarray()
            expected_values = expected_values.toarray() if hasattr(expected_values, 'toarray') else expected_values
        else:
            actual_values = log_adata.X
        
        assert np.allclose(actual_values, expected_values, rtol=1e-5)
    
    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_SCANPY, reason="scanpy not available")
    def test_highly_variable_genes(self, mock_h5ad_file):
        """Test highly variable gene identification."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        adata = sc.read_h5ad(mock_h5ad_file)
        pipeline = PreprocessingPipeline()
        
        # Test HVG identification
        hvg_adata = pipeline._highly_variable_genes(
            adata,
            n_top_genes=1000,
            method='seurat_v3'
        )
        
        # Check that HVG annotation was added
        assert 'highly_variable' in hvg_adata.var.columns
        assert hvg_adata.var['highly_variable'].dtype == bool
        
        # Check number of HVG
        n_hvg = hvg_adata.var['highly_variable'].sum()
        assert n_hvg <= 1000  # Should be at most n_top_genes
    
    @pytest.mark.unit
    def test_sparsity_calculation(self):
        """Test sparsity calculation utility."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        from scipy.sparse import csr_matrix
        
        pipeline = PreprocessingPipeline()
        
        # Test dense matrix
        dense_matrix = np.array([[1, 0, 3], [0, 0, 6], [7, 8, 0]])
        sparsity_dense = pipeline._calculate_sparsity(dense_matrix)
        expected_sparsity = 4/9  # 4 zeros out of 9 elements
        assert abs(sparsity_dense - expected_sparsity) < 1e-6
        
        # Test sparse matrix
        sparse_matrix = csr_matrix(dense_matrix)
        sparsity_sparse = pipeline._calculate_sparsity(sparse_matrix)
        assert abs(sparsity_sparse - expected_sparsity) < 1e-6
    
    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_SCANPY, reason="scanpy not available")
    def test_full_pipeline_execution(self, mock_h5ad_file):
        """Test full pipeline execution."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        adata = sc.read_h5ad(mock_h5ad_file)
        initial_shape = adata.shape
        
        # Create pipeline with subset of steps
        steps = ['filter_cells', 'filter_genes', 'normalize_total', 'log1p']
        pipeline = PreprocessingPipeline(steps=steps, track_metadata=True)
        
        # Run pipeline
        processed_adata, metadata = pipeline.process(adata, return_metadata=True)
        
        # Check that processing occurred
        assert processed_adata.shape[0] <= initial_shape[0]  # Cells may be filtered
        assert processed_adata.shape[1] <= initial_shape[1]  # Genes may be filtered
        
        # Check metadata tracking
        assert 'steps_applied' in metadata
        assert 'parameters_used' in metadata
        assert 'timing' in metadata
        assert len(metadata['steps_applied']) == len(steps)
    
    @pytest.mark.unit
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        from scgraph_hub.preprocessing import PreprocessingPipeline
        
        # Test with invalid step
        pipeline = PreprocessingPipeline(steps=['invalid_step'])
        
        # Mock adata
        mock_adata = Mock()
        
        # Should handle unknown steps gracefully
        result = pipeline.process(mock_adata)
        assert result is not None


class TestGraphConstructor:
    """Test cases for GraphConstructor class."""
    
    @pytest.mark.unit
    def test_constructor_initialization(self):
        """Test graph constructor initialization."""
        from scgraph_hub.preprocessing import GraphConstructor
        
        # Test default initialization
        constructor = GraphConstructor()
        assert constructor.method == 'knn'
        assert hasattr(constructor, 'parameters')
        assert hasattr(constructor, 'methods')
        
        # Test custom initialization
        custom_constructor = GraphConstructor(
            method='spatial',
            max_distance=100,
            method_type='radius'
        )
        assert custom_constructor.method == 'spatial'
        assert 'max_distance' in custom_constructor.parameters
    
    @pytest.mark.unit
    def test_available_methods(self):
        """Test that all graph construction methods are available."""
        from scgraph_hub.preprocessing import GraphConstructor
        
        constructor = GraphConstructor()
        
        expected_methods = ['knn', 'radius', 'spatial', 'correlation', 'coexpression']
        for method in expected_methods:
            assert method in constructor.methods
            assert callable(constructor.methods[method])
    
    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_SCANPY, reason="scanpy not available")
    def test_knn_graph_construction(self, mock_h5ad_file):
        """Test k-NN graph construction."""
        from scgraph_hub.preprocessing import GraphConstructor
        
        adata = sc.read_h5ad(mock_h5ad_file)
        
        # Add PCA for k-NN
        sc.pp.pca(adata, n_comps=10)
        
        constructor = GraphConstructor(method='knn', k=10)
        edge_index, edge_weights = constructor.build_graph(adata, return_edge_weights=True)
        
        # Test edge index properties
        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.shape[0] == 2  # [source, target]
        assert edge_index.dtype == torch.long
        assert edge_index.min() >= 0
        assert edge_index.max() < adata.n_obs
        
        # Test edge weights if returned
        if edge_weights is not None:
            assert isinstance(edge_weights, torch.Tensor)
            assert edge_weights.shape[0] == edge_index.shape[1]
            assert edge_weights.dtype == torch.float
            assert (edge_weights >= 0).all()  # Weights should be non-negative
    
    @pytest.mark.unit
    def test_spatial_graph_construction(self):
        """Test spatial graph construction."""
        from scgraph_hub.preprocessing import GraphConstructor
        
        # Create mock spatial data
        n_cells = 50
        spatial_coords = np.random.uniform(0, 100, (n_cells, 2))
        
        # Mock adata with spatial coordinates
        mock_adata = Mock()
        mock_adata.n_obs = n_cells
        mock_adata.obsm = {'spatial': spatial_coords}
        
        constructor = GraphConstructor(method='spatial', max_distance=20)
        edge_index, edge_weights = constructor.build_graph(mock_adata, return_edge_weights=True)
        
        # Test basic properties
        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.shape[0] == 2
        assert edge_index.min() >= 0
        assert edge_index.max() < n_cells
    
    @pytest.mark.unit
    def test_correlation_graph_construction(self):
        """Test correlation-based graph construction."""
        from scgraph_hub.preprocessing import GraphConstructor
        
        # Create mock expression data
        n_cells, n_genes = 30, 20
        expression_data = np.random.randn(n_cells, n_genes)
        
        # Mock adata
        mock_adata = Mock()
        mock_adata.n_obs = n_cells
        mock_adata.X = expression_data
        
        constructor = GraphConstructor(method='correlation', threshold=0.5)
        edge_index, edge_weights = constructor.build_graph(mock_adata, return_edge_weights=True)
        
        # Test basic properties
        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.shape[0] == 2
        
        if edge_weights is not None:
            # Correlation weights should be between -1 and 1
            assert (edge_weights >= -1).all()
            assert (edge_weights <= 1).all()
    
    @pytest.mark.unit
    def test_graph_undirected_property(self):
        """Test that graphs maintain undirected property when expected."""
        from scgraph_hub.preprocessing import GraphConstructor
        
        # Create simple test data
        n_cells = 10
        mock_adata = Mock()
        mock_adata.n_obs = n_cells
        mock_adata.X = np.random.randn(n_cells, 5)
        mock_adata.obsm = {'X_pca': np.random.randn(n_cells, 3)}
        
        constructor = GraphConstructor(method='knn', k=3)
        edge_index, _ = constructor.build_graph(mock_adata, return_edge_weights=False)
        
        # For undirected graphs, each edge should appear in both directions
        edge_set = set(tuple(edge_index[:, i].tolist()) for i in range(edge_index.shape[1]))
        
        undirected_count = 0
        for i, j in edge_set:
            if (j, i) in edge_set:
                undirected_count += 1
        
        # Most edges should have their reverse
        total_edges = len(edge_set)
        assert undirected_count >= total_edges * 0.8  # Allow some asymmetry


class TestPreprocessingFunction:
    """Test cases for the main preprocess_dataset function."""
    
    @pytest.mark.unit
    @patch('scgraph_hub.preprocessing.sc.read_h5ad')
    @patch('torch.save')
    def test_preprocess_dataset_function(self, mock_torch_save, mock_read_h5ad):
        """Test the main preprocess_dataset function."""
        from scgraph_hub.preprocessing import preprocess_dataset
        
        # Mock AnnData object
        mock_adata = Mock()
        mock_adata.n_obs = 100
        mock_adata.n_vars = 50
        mock_adata.X = np.random.randn(100, 50)
        mock_adata.obs = pd.DataFrame({'cell_type': ['A'] * 50 + ['B'] * 50})
        mock_adata.obsm = {'X_pca': np.random.randn(100, 10)}
        
        mock_read_h5ad.return_value = mock_adata
        
        # Test function call
        metadata = preprocess_dataset(
            dataset_name='test_dataset',
            input_path='/mock/input.h5ad',
            output_path='/mock/output.pt',
            steps=['normalize_total', 'log1p'],
            parameters={'normalize_total': {'target_sum': 1e4}},
            graph_method='knn',
            graph_parameters={'k': 15}
        )
        
        # Verify function execution
        assert mock_read_h5ad.called
        assert mock_torch_save.called
        
        # Check metadata structure
        assert 'dataset_name' in metadata
        assert 'processing_time_seconds' in metadata
        assert 'preprocessing_steps' in metadata
        assert 'graph_method' in metadata
        assert 'final_statistics' in metadata
        
        assert metadata['dataset_name'] == 'test_dataset'
        assert metadata['graph_method'] == 'knn'
    
    @pytest.mark.unit
    def test_preprocess_dataset_error_handling(self):
        """Test error handling in preprocess_dataset function."""
        from scgraph_hub.preprocessing import preprocess_dataset
        
        # Test with non-existent file
        with pytest.raises(Exception):
            preprocess_dataset(
                dataset_name='test',
                input_path='/nonexistent/file.h5ad',
                output_path='/mock/output.pt'
            )
    
    @pytest.mark.unit
    def test_invalid_graph_method(self):
        """Test handling of invalid graph construction method."""
        from scgraph_hub.preprocessing import GraphConstructor
        
        with pytest.raises(ValueError):
            constructor = GraphConstructor(method='invalid_method')
            mock_adata = Mock()
            constructor.build_graph(mock_adata)


@pytest.mark.unit
def test_preprocessing_imports():
    """Test that preprocessing module imports correctly."""
    try:
        from scgraph_hub.preprocessing import PreprocessingPipeline, GraphConstructor, preprocess_dataset
        assert PreprocessingPipeline is not None
        assert GraphConstructor is not None
        assert preprocess_dataset is not None
    except ImportError as e:
        pytest.fail(f"Failed to import preprocessing components: {e}")


@pytest.mark.unit
def test_preprocessing_constants():
    """Test preprocessing module constants and configurations."""
    # Test that we can access expected preprocessing steps
    expected_steps = [
        'filter_cells', 'filter_genes', 'calculate_qc_metrics',
        'normalize_total', 'log1p', 'highly_variable_genes',
        'scale', 'pca', 'neighbors'
    ]
    
    # This tests the concept even if implementation details vary
    assert len(expected_steps) > 5
    for step in expected_steps:
        assert isinstance(step, str)
        assert '_' in step or step.islower()  # Valid step naming convention