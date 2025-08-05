"""Advanced preprocessing pipelines for single-cell graph data."""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import scanpy as sc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse import issparse, csr_matrix, csc_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import umap

# Import database functions conditionally
try:
    from .database import get_dataset_repository
    _DATABASE_AVAILABLE = True
except ImportError:
    _DATABASE_AVAILABLE = False
    def get_dataset_repository():
        """Placeholder when database is not available."""
        class MockRepository:
            def log_processing_operation(self, *args, **kwargs):
                pass
        return MockRepository()

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Comprehensive preprocessing pipeline for single-cell data."""
    
    def __init__(self, steps: Optional[List[str]] = None, 
                 parameters: Optional[Dict[str, Any]] = None,
                 track_metadata: bool = True):
        """Initialize preprocessing pipeline.
        
        Args:
            steps: List of preprocessing step names
            parameters: Parameters for each step
            track_metadata: Whether to track processing metadata
        """
        self.steps = steps or self._get_default_steps()
        self.parameters = parameters or {}
        self.track_metadata = track_metadata
        
        # Processing metadata
        self.metadata = {
            'steps_applied': [],
            'parameters_used': {},
            'statistics': {},
            'timing': {}
        }
        
        # Available step functions
        self.step_functions = {
            'filter_cells': self._filter_cells,
            'filter_genes': self._filter_genes,
            'calculate_qc_metrics': self._calculate_qc_metrics,
            'detect_doublets': self._detect_doublets,
            'normalize_total': self._normalize_total,
            'log1p': self._log1p_transform,
            'highly_variable_genes': self._highly_variable_genes,
            'scale': self._scale_data,
            'pca': self._principal_component_analysis,
            'neighbors': self._compute_neighbors,
            'umap': self._compute_umap,
            'clustering': self._perform_clustering,
            'remove_batch_effects': self._remove_batch_effects,
            'impute_missing': self._impute_missing_values,
            'detect_cell_cycle': self._detect_cell_cycle,
            'pseudotime': self._compute_pseudotime
        }
    
    def _get_default_steps(self) -> List[str]:
        """Get default preprocessing steps."""
        return [
            'filter_cells',
            'filter_genes', 
            'calculate_qc_metrics',
            'normalize_total',
            'log1p',
            'highly_variable_genes',
            'scale',
            'pca',
            'neighbors'
        ]
    
    def process(self, adata, return_metadata: bool = False):
        """Run the complete preprocessing pipeline.
        
        Args:
            adata: AnnData object
            return_metadata: Whether to return processing metadata
            
        Returns:
            Processed AnnData object and optionally metadata
        """
        import time
        
        logger.info(f"Starting preprocessing pipeline with {len(self.steps)} steps")
        
        for i, step in enumerate(self.steps):
            logger.info(f"Step {i+1}/{len(self.steps)}: {step}")
            
            start_time = time.time()
            
            if step not in self.step_functions:
                logger.warning(f"Unknown preprocessing step: {step}")
                continue
            
            # Get step parameters
            step_params = self.parameters.get(step, {})
            
            try:
                # Apply step
                adata = self.step_functions[step](adata, **step_params)
                
                # Track metadata
                if self.track_metadata:
                    duration = time.time() - start_time
                    self.metadata['steps_applied'].append(step)
                    self.metadata['parameters_used'][step] = step_params
                    self.metadata['timing'][step] = duration
                    
                    # Store dataset statistics after each step
                    self.metadata['statistics'][f'{step}_result'] = {
                        'n_obs': adata.n_obs,
                        'n_vars': adata.n_vars,
                        'sparsity': self._calculate_sparsity(adata.X) if hasattr(adata, 'X') else None
                    }
                
                logger.info(f"Completed {step} in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in step {step}: {e}")
                if not self.parameters.get('continue_on_error', False):
                    raise
        
        logger.info("Preprocessing pipeline completed")
        
        if return_metadata:
            return adata, self.metadata
        return adata
    
    def _calculate_sparsity(self, X) -> float:
        """Calculate sparsity of data matrix."""
        if issparse(X):
            return 1.0 - X.nnz / (X.shape[0] * X.shape[1])
        else:
            return np.mean(X == 0)
    
    # Step implementations
    def _filter_cells(self, adata, min_genes: int = 200, max_genes: Optional[int] = None,
                     min_counts: Optional[int] = None, max_counts: Optional[int] = None,
                     mt_gene_pattern: str = '^MT-', max_mt_pct: float = 20.0):
        """Filter cells based on quality metrics."""
        n_cells_before = adata.n_obs
        
        # Calculate basic metrics if not present
        if 'n_genes_by_counts' not in adata.obs.columns:
            sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        # Calculate mitochondrial gene percentage
        adata.var['mt'] = adata.var_names.str.match(mt_gene_pattern, case=False)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        # Apply filters
        sc.pp.filter_cells(adata, min_genes=min_genes)
        
        if max_genes:
            sc.pp.filter_cells(adata, max_genes=max_genes)
        
        if min_counts:
            sc.pp.filter_cells(adata, min_counts=min_counts)
        
        if max_counts:
            sc.pp.filter_cells(adata, max_counts=max_counts)
        
        # Filter high mitochondrial percentage cells
        adata = adata[adata.obs.pct_counts_mt < max_mt_pct, :].copy()
        
        logger.info(f"Filtered cells: {n_cells_before} -> {adata.n_obs} ({n_cells_before - adata.n_obs} removed)")
        
        return adata
    
    def _filter_genes(self, adata, min_cells: int = 3, max_cells: Optional[int] = None,
                     min_counts: Optional[int] = None):
        """Filter genes based on expression criteria."""
        n_genes_before = adata.n_vars
        
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        if max_cells:
            sc.pp.filter_genes(adata, max_cells=max_cells)
        
        if min_counts:
            sc.pp.filter_genes(adata, min_counts=min_counts)
        
        logger.info(f"Filtered genes: {n_genes_before} -> {adata.n_vars} ({n_genes_before - adata.n_vars} removed)")
        
        return adata
    
    def _calculate_qc_metrics(self, adata, mt_gene_pattern: str = '^MT-',
                             ribo_gene_pattern: str = '^RP[SL]'):
        """Calculate comprehensive quality control metrics."""
        # Mitochondrial genes
        adata.var['mt'] = adata.var_names.str.match(mt_gene_pattern, case=False)
        
        # Ribosomal genes
        adata.var['ribo'] = adata.var_names.str.match(ribo_gene_pattern, case=False)
        
        # Hemoglobin genes
        adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', case=False)
        
        # Calculate metrics
        sc.pp.calculate_qc_metrics(
            adata, 
            qc_vars=['mt', 'ribo', 'hb'], 
            percent_top=None, 
            log1p=False, 
            inplace=True
        )
        
        # Additional custom metrics
        adata.obs['log10_total_counts'] = np.log10(adata.obs['total_counts'])
        adata.obs['log10_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
        
        return adata
    
    def _detect_doublets(self, adata, method: str = 'scrublet', **kwargs):
        """Detect potential doublets in the data."""
        try:
            if method == 'scrublet':
                import scrublet as scr
                
                scrub = scr.Scrublet(adata.X, **kwargs)
                doublet_scores, predicted_doublets = scrub.scrub_doublets(
                    min_counts=2, min_cells=3, min_gene_variability_pctl=85,
                    n_prin_comps=30
                )
                
                adata.obs['doublet_score'] = doublet_scores
                adata.obs['predicted_doublet'] = predicted_doublets
                
                logger.info(f"Detected {np.sum(predicted_doublets)} potential doublets")
                
            elif method == 'doubletdetection':
                import doubletdetection as dd
                
                clf = dd.BoostClassifier(n_iters=25, use_phenograph=False, **kwargs)
                doublet_scores = clf.fit(adata.X).predict()
                
                adata.obs['doublet_score'] = doublet_scores
                adata.obs['predicted_doublet'] = doublet_scores > np.percentile(doublet_scores, 90)
                
        except ImportError as e:
            logger.warning(f"Doublet detection method {method} not available: {e}")
        
        return adata
    
    def _normalize_total(self, adata, target_sum: float = 1e4, exclude_highly_expressed: bool = True):
        """Normalize total counts per cell."""
        sc.pp.normalize_total(
            adata, 
            target_sum=target_sum, 
            exclude_highly_expressed=exclude_highly_expressed
        )
        
        return adata
    
    def _log1p_transform(self, adata):
        """Apply log1p transformation."""
        sc.pp.log1p(adata)
        return adata
    
    def _highly_variable_genes(self, adata, n_top_genes: int = 2000, 
                              method: str = 'seurat_v3', batch_key: Optional[str] = None):
        """Identify highly variable genes."""
        if batch_key and batch_key in adata.obs.columns:
            # Batch-aware HVG selection
            sc.pp.highly_variable_genes(
                adata, 
                n_top_genes=n_top_genes, 
                batch_key=batch_key,
                subset=False
            )
        else:
            sc.pp.highly_variable_genes(
                adata, 
                n_top_genes=n_top_genes, 
                flavor=method,
                subset=False
            )
        
        logger.info(f"Identified {np.sum(adata.var['highly_variable'])} highly variable genes")
        
        return adata
    
    def _scale_data(self, adata, max_value: Optional[float] = 10, zero_center: bool = True,
                   use_highly_variable: bool = True):
        """Scale data to unit variance."""
        # Store raw data
        adata.raw = adata
        
        # Use only highly variable genes if available and requested
        if use_highly_variable and 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.var.highly_variable].copy()
        
        sc.pp.scale(adata, max_value=max_value, zero_center=zero_center)
        
        return adata
    
    def _principal_component_analysis(self, adata, n_comps: int = 50, 
                                    use_highly_variable: bool = True,
                                    svd_solver: str = 'arpack'):
        """Perform principal component analysis."""
        sc.tl.pca(
            adata, 
            n_comps=n_comps, 
            use_highly_variable=use_highly_variable,
            svd_solver=svd_solver
        )
        
        return adata
    
    def _compute_neighbors(self, adata, n_neighbors: int = 15, n_pcs: Optional[int] = None,
                          method: str = 'umap', metric: str = 'euclidean'):
        """Compute neighborhood graph."""
        sc.pp.neighbors(
            adata, 
            n_neighbors=n_neighbors, 
            n_pcs=n_pcs,
            method=method,
            metric=metric
        )
        
        return adata
    
    def _compute_umap(self, adata, min_dist: float = 0.5, spread: float = 1.0,
                     n_components: int = 2, alpha: float = 1.0):
        """Compute UMAP embedding."""
        sc.tl.umap(
            adata,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            alpha=alpha
        )
        
        return adata
    
    def _perform_clustering(self, adata, method: str = 'leiden', 
                           resolution: float = 0.5, **kwargs):
        """Perform clustering analysis."""
        if method == 'leiden':
            sc.tl.leiden(adata, resolution=resolution, **kwargs)
        elif method == 'louvain':
            sc.tl.louvain(adata, resolution=resolution, **kwargs)
        else:
            logger.warning(f"Unknown clustering method: {method}")
        
        return adata
    
    def _remove_batch_effects(self, adata, batch_key: str, method: str = 'combat'):
        """Remove batch effects."""
        try:
            if method == 'combat':
                sc.pp.combat(adata, key=batch_key)
            elif method == 'harmony':
                import scanpy.external as sce
                sce.pp.harmony_integrate(adata, key=batch_key)
            elif method == 'scanorama':
                import scanorama
                # Implementation for scanorama
                pass
            else:
                logger.warning(f"Unknown batch correction method: {method}")
        
        except ImportError as e:
            logger.warning(f"Batch correction method {method} not available: {e}")
        
        return adata
    
    def _impute_missing_values(self, adata, method: str = 'magic', **kwargs):
        """Impute missing values in the data."""
        try:
            if method == 'magic':
                import magic
                magic_op = magic.MAGIC(**kwargs)
                adata.X = magic_op.fit_transform(adata.X)
            
            elif method == 'scimpute':
                # Placeholder for scImpute implementation
                logger.warning("scImpute not implemented yet")
            
            elif method == 'dca':
                # Placeholder for DCA implementation
                logger.warning("DCA not implemented yet")
        
        except ImportError as e:
            logger.warning(f"Imputation method {method} not available: {e}")
        
        return adata
    
    def _detect_cell_cycle(self, adata, organism: str = 'human'):
        """Detect cell cycle phase."""
        try:
            if organism == 'human':
                s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMPS', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
                g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
            elif organism == 'mouse':
                # Mouse cell cycle genes (converted from human)
                s_genes = [gene.capitalize() for gene in s_genes]
                g2m_genes = [gene.capitalize() for gene in g2m_genes]
            
            # Score cell cycle
            sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
            
        except Exception as e:
            logger.warning(f"Cell cycle detection failed: {e}")
        
        return adata
    
    def _compute_pseudotime(self, adata, method: str = 'dpt', root_cell: Optional[str] = None):
        """Compute pseudotime using diffusion pseudotime."""
        try:
            if method == 'dpt':
                # Compute diffusion pseudotime
                sc.tl.diffmap(adata)
                
                if root_cell:
                    adata.uns['iroot'] = np.flatnonzero(adata.obs_names == root_cell)[0]
                else:
                    # Automatically select root
                    adata.uns['iroot'] = np.argmin(adata.obsm['X_diffmap'][:, 0])
                
                sc.tl.dpt(adata)
            
            elif method == 'palantir':
                # Placeholder for Palantir implementation
                logger.warning("Palantir pseudotime not implemented yet")
            
        except Exception as e:
            logger.warning(f"Pseudotime computation failed: {e}")
        
        return adata


class GraphConstructor:
    """Constructs various types of graphs from processed single-cell data."""
    
    def __init__(self, method: str = 'knn', **parameters):
        """Initialize graph constructor.
        
        Args:
            method: Graph construction method
            **parameters: Method-specific parameters
        """
        self.method = method
        self.parameters = parameters
        
        self.methods = {
            'knn': self._build_knn_graph,
            'radius': self._build_radius_graph,
            'spatial': self._build_spatial_graph,
            'correlation': self._build_correlation_graph,
            'coexpression': self._build_coexpression_graph,
            'regulatory': self._build_regulatory_graph
        }
    
    def build_graph(self, adata, return_edge_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build graph from AnnData object.
        
        Args:
            adata: Processed AnnData object
            return_edge_weights: Whether to return edge weights
            
        Returns:
            Edge index tensor and optionally edge weights
        """
        if self.method not in self.methods:
            raise ValueError(f"Unknown graph construction method: {self.method}")
        
        return self.methods[self.method](adata, return_edge_weights)
    
    def _build_knn_graph(self, adata, return_edge_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build k-nearest neighbor graph."""
        k = self.parameters.get('k', 15)
        metric = self.parameters.get('metric', 'euclidean')
        use_rep = self.parameters.get('use_rep', 'X_pca')
        
        # Get representation
        if use_rep in adata.obsm:
            X = adata.obsm[use_rep]
        else:
            X = adata.X.toarray() if issparse(adata.X) else adata.X
        
        # Compute k-NN
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Build edge list
        edge_list = []
        edge_weights = []
        
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip self
                neighbor_idx = indices[i][j]
                edge_list.extend([(i, neighbor_idx), (neighbor_idx, i)])  # Undirected
                
                if return_edge_weights:
                    weight = 1.0 / (1.0 + distances[i][j])  # Distance to similarity
                    edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        if return_edge_weights:
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
            return edge_index, edge_weights
        
        return edge_index, None
    
    def _build_radius_graph(self, adata, return_edge_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build radius-based graph."""
        radius = self.parameters.get('radius', 1.0)
        metric = self.parameters.get('metric', 'euclidean')
        use_rep = self.parameters.get('use_rep', 'X_pca')
        
        # Get representation
        if use_rep in adata.obsm:
            X = adata.obsm[use_rep]
        else:
            X = adata.X.toarray() if issparse(adata.X) else adata.X
        
        # Compute radius neighbors
        nbrs = NearestNeighbors(radius=radius, metric=metric).fit(X)
        distances, indices = nbrs.radius_neighbors(X)
        
        edge_list = []
        edge_weights = []
        
        for i, neighbors in enumerate(indices):
            for j, neighbor_idx in enumerate(neighbors):
                if i != neighbor_idx:  # Skip self
                    edge_list.append((i, neighbor_idx))
                    
                    if return_edge_weights:
                        weight = 1.0 / (1.0 + distances[i][j])
                        edge_weights.append(weight)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        if return_edge_weights:
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
            return edge_index, edge_weights
        
        return edge_index, None
    
    def _build_spatial_graph(self, adata, return_edge_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build spatial proximity graph."""
        if 'spatial' not in adata.obsm:
            raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")
        
        coords = adata.obsm['spatial']
        max_distance = self.parameters.get('max_distance', 150)
        method = self.parameters.get('method', 'radius')  # 'radius' or 'delaunay'
        
        if method == 'radius':
            # Radius-based spatial graph
            nbrs = NearestNeighbors(radius=max_distance).fit(coords)
            distances, indices = nbrs.radius_neighbors(coords)
            
            edge_list = []
            edge_weights = []
            
            for i, neighbors in enumerate(indices):
                for j, neighbor_idx in enumerate(neighbors):
                    if i != neighbor_idx:
                        edge_list.append((i, neighbor_idx))
                        
                        if return_edge_weights:
                            weight = 1.0 / (1.0 + distances[i][j])
                            edge_weights.append(weight)
        
        elif method == 'delaunay':
            from scipy.spatial import Delaunay
            
            # Delaunay triangulation
            tri = Delaunay(coords)
            
            edge_list = []
            edge_weights = []
            
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        edge_list.extend([(simplex[i], simplex[j]), (simplex[j], simplex[i])])
                        
                        if return_edge_weights:
                            dist = np.linalg.norm(coords[simplex[i]] - coords[simplex[j]])
                            if dist <= max_distance:  # Filter long edges
                                weight = 1.0 / (1.0 + dist)
                                edge_weights.extend([weight, weight])
                            else:
                                edge_list = edge_list[:-2]  # Remove last two edges
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        if return_edge_weights:
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
            return edge_index, edge_weights
        
        return edge_index, None
    
    def _build_correlation_graph(self, adata, return_edge_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build correlation-based graph."""
        threshold = self.parameters.get('threshold', 0.7)
        method = self.parameters.get('method', 'pearson')
        
        # Get expression data
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        
        # Compute correlation matrix
        if method == 'pearson':
            corr_matrix = np.corrcoef(X)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(X, axis=1)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Create edges where correlation > threshold
        edge_indices = np.where(corr_matrix > threshold)
        
        # Remove self-loops
        mask = edge_indices[0] != edge_indices[1]
        edge_list = list(zip(edge_indices[0][mask], edge_indices[1][mask]))
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        if return_edge_weights:
            edge_weights = torch.tensor(corr_matrix[edge_indices[0][mask], edge_indices[1][mask]], dtype=torch.float)
            return edge_index, edge_weights
        
        return edge_index, None
    
    def _build_coexpression_graph(self, adata, return_edge_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build gene coexpression graph."""
        # This would build edges between genes based on coexpression patterns
        # Implementation depends on specific requirements
        logger.warning("Coexpression graph construction not fully implemented")
        return torch.empty((2, 0), dtype=torch.long), None
    
    def _build_regulatory_graph(self, adata, return_edge_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build gene regulatory network graph."""
        # This would incorporate prior knowledge about gene regulatory relationships
        # Implementation would require external databases (e.g., STRING, RegNetwork)
        logger.warning("Regulatory graph construction not fully implemented")
        return torch.empty((2, 0), dtype=torch.long), None


# Main preprocessing function
def preprocess_dataset(dataset_name: str, 
                     input_path: str,
                     output_path: str,
                     steps: Optional[List[str]] = None,
                     parameters: Optional[Dict[str, Any]] = None,
                     graph_method: str = 'knn',
                     graph_parameters: Optional[Dict[str, Any]] = None,
                     save_intermediate: bool = False) -> Dict[str, Any]:
    """Complete preprocessing pipeline for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        input_path: Path to input H5AD file
        output_path: Path for output processed file
        steps: Preprocessing steps to apply
        parameters: Parameters for preprocessing steps
        graph_method: Graph construction method
        graph_parameters: Parameters for graph construction
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Processing metadata and statistics
    """
    import time
    start_time = time.time()
    
    logger.info(f"Starting preprocessing for dataset: {dataset_name}")
    
    try:
        # Load data
        adata = sc.read_h5ad(input_path)
        logger.info(f"Loaded dataset: {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Initialize preprocessing pipeline
        pipeline = PreprocessingPipeline(steps=steps, parameters=parameters or {})
        
        # Run preprocessing
        adata, preprocessing_metadata = pipeline.process(adata, return_metadata=True)
        
        # Construct graph
        graph_constructor = GraphConstructor(method=graph_method, **(graph_parameters or {}))
        edge_index, edge_weights = graph_constructor.build_graph(adata)
        
        # Create PyTorch Geometric Data object
        from torch_geometric.data import Data
        
        # Get features
        if 'X_pca' in adata.obsm:
            x = torch.FloatTensor(adata.obsm['X_pca'])
        else:
            x = torch.FloatTensor(adata.X.toarray() if issparse(adata.X) else adata.X)
        
        # Get labels (if available)
        y = None
        if 'cell_type' in adata.obs.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = torch.LongTensor(le.fit_transform(adata.obs['cell_type']))
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y)
        
        # Save processed data
        torch.save(data, output_path)
        
        # Save AnnData if requested
        if save_intermediate:
            adata_path = output_path.replace('.pt', '_adata.h5ad')
            adata.write(adata_path)
        
        # Collect final metadata
        processing_time = time.time() - start_time
        
        final_metadata = {
            'dataset_name': dataset_name,
            'processing_time_seconds': processing_time,
            'preprocessing_steps': preprocessing_metadata['steps_applied'],
            'preprocessing_parameters': preprocessing_metadata['parameters_used'],
            'graph_method': graph_method,
            'graph_parameters': graph_parameters or {},
            'final_statistics': {
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars,
                'n_edges': edge_index.shape[1],
                'graph_density': edge_index.shape[1] / (adata.n_obs * (adata.n_obs - 1)),
                'feature_dim': x.shape[1] if x is not None else 0,
                'has_labels': y is not None,
                'n_classes': len(torch.unique(y)) if y is not None else 0
            },
            'file_paths': {
                'input': input_path,
                'output': output_path,
                'adata': adata_path if save_intermediate else None
            }
        }
        
        # Log to database if available
        try:
            repo = get_dataset_repository()
            repo.log_processing_operation(
                dataset_name=dataset_name,
                operation='preprocess',
                status='completed',
                duration_seconds=processing_time,
                parameters={
                    'preprocessing_steps': steps,
                    'preprocessing_parameters': parameters,
                    'graph_method': graph_method,
                    'graph_parameters': graph_parameters
                },
                results=final_metadata['final_statistics']
            )
        except Exception as e:
            logger.warning(f"Failed to log to database: {e}")
        
        logger.info(f"Preprocessing completed successfully in {processing_time:.2f}s")
        
        return final_metadata
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        
        # Log failure to database
        try:
            repo = get_dataset_repository()
            repo.log_processing_operation(
                dataset_name=dataset_name,
                operation='preprocess',
                status='failed',
                error_message=str(e)
            )
        except Exception:
            pass
        
        raise