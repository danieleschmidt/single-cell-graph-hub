"""Comprehensive metrics for evaluating single-cell graph models."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BiologicalMetrics:
    """Biological validation metrics for single-cell analysis."""
    
    def __init__(self):
        self.gene_sets = self._load_gene_sets()
    
    def _load_gene_sets(self) -> Dict[str, List[str]]:
        """Load gene sets for biological validation."""
        # Placeholder gene sets - in practice, these would be loaded from databases
        gene_sets = {
            'cell_cycle': ['PCNA', 'MKI67', 'TOP2A', 'CCNB1', 'CCND1', 'CDK1', 'CDK2'],
            'apoptosis': ['BAX', 'BCL2', 'CASP3', 'CASP8', 'TP53', 'FAS', 'FADD'],
            'immune_response': ['CD3E', 'CD4', 'CD8A', 'CD19', 'CD68', 'FCGR3A'],
            'stress_response': ['HSPA1A', 'HSPA1B', 'HSP90AA1', 'DDIT3', 'ATF4'],
            'ribosomal': ['RPS4X', 'RPS6', 'RPL13A', 'RPL7A', 'RPS27A']
        }
        return gene_sets
    
    def cell_type_purity(self,
                        embeddings: np.ndarray,
                        cell_types: np.ndarray,
                        method: str = 'silhouette') -> float:
        """Measure how well cell types are separated in embedding space.
        
        Args:
            embeddings: Cell embeddings [n_cells, n_dims]
            cell_types: Cell type labels [n_cells]
            method: Metric to use ('silhouette', 'calinski_harabasz', 'davies_bouldin')
            
        Returns:
            Purity score
        """
        if len(np.unique(cell_types)) < 2:
            logger.warning("Less than 2 cell types, cannot compute purity")
            return 0.0
        
        try:
            if method == 'silhouette':
                return silhouette_score(embeddings, cell_types)
            elif method == 'calinski_harabasz':
                return calinski_harabasz_score(embeddings, cell_types)
            elif method == 'davies_bouldin':
                return 1.0 / (1.0 + davies_bouldin_score(embeddings, cell_types))  # Invert so higher is better
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except Exception as e:
            logger.error(f"Failed to compute cell type purity: {e}")
            return 0.0
    
    def biological_conservation(self,
                               original_data: Any,
                               embedded_data: np.ndarray,
                               gene_sets: Optional[str] = 'hallmark') -> float:
        """Measure preservation of biological signal in embeddings.
        
        Args:
            original_data: Original expression data or AnnData object
            embedded_data: Low-dimensional embeddings
            gene_sets: Gene sets to use for validation
            
        Returns:
            Conservation score (0-1)
        """
        try:
            # Extract expression matrix
            if hasattr(original_data, 'X'):
                if hasattr(original_data.X, 'toarray'):
                    expr_matrix = original_data.X.toarray()
                else:
                    expr_matrix = original_data.X
            else:
                expr_matrix = original_data
            
            # Compute correlations for biological pathways
            conservation_scores = []
            
            for pathway, genes in self.gene_sets.items():
                if hasattr(original_data, 'var_names'):
                    # Match genes to features
                    gene_indices = [i for i, gene in enumerate(original_data.var_names) if gene in genes]
                    
                    if len(gene_indices) < 2:
                        continue
                    
                    # Compute pathway activity in original space
                    pathway_activity_orig = np.mean(expr_matrix[:, gene_indices], axis=1)
                    
                    # Compute correlation with embedding coordinates
                    max_corr = 0
                    for dim in range(embedded_data.shape[1]):
                        corr, _ = pearsonr(pathway_activity_orig, embedded_data[:, dim])
                        max_corr = max(max_corr, abs(corr))
                    
                    conservation_scores.append(max_corr)
            
            return np.mean(conservation_scores) if conservation_scores else 0.0
        
        except Exception as e:
            logger.error(f"Failed to compute biological conservation: {e}")
            return 0.0
    
    def trajectory_conservation(self,
                               original_trajectory: np.ndarray,
                               embedded_trajectory: np.ndarray,
                               metric: str = 'kendall_tau') -> float:
        """Measure preservation of trajectory structure.
        
        Args:
            original_trajectory: Original pseudotime or trajectory coordinates
            embedded_trajectory: Embedded trajectory coordinates
            metric: Correlation metric ('kendall_tau', 'spearman', 'pearson')
            
        Returns:
            Trajectory conservation score
        """
        try:
            if metric == 'kendall_tau':
                corr, _ = kendalltau(original_trajectory, embedded_trajectory)
            elif metric == 'spearman':
                corr, _ = spearmanr(original_trajectory, embedded_trajectory)
            elif metric == 'pearson':
                corr, _ = pearsonr(original_trajectory, embedded_trajectory)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            return abs(corr)  # Absolute correlation
        
        except Exception as e:
            logger.error(f"Failed to compute trajectory conservation: {e}")
            return 0.0
    
    def batch_mixing_score(self,
                          embeddings: np.ndarray,
                          batch_labels: np.ndarray,
                          cell_type_labels: np.ndarray,
                          k: int = 50) -> float:
        """Measure how well batches are mixed while preserving cell types.
        
        Args:
            embeddings: Cell embeddings
            batch_labels: Batch assignment for each cell
            cell_type_labels: Cell type labels
            k: Number of neighbors to consider
            
        Returns:
            Batch mixing score (0-1, higher is better mixing)
        """
        try:
            # Find k-NN for each cell
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
            _, indices = nbrs.kneighbors(embeddings)
            
            mixing_scores = []
            
            for i in range(len(embeddings)):
                neighbors = indices[i][1:]  # Exclude self
                
                # Check if neighbors are from same cell type
                same_celltype_neighbors = neighbors[cell_type_labels[neighbors] == cell_type_labels[i]]
                
                if len(same_celltype_neighbors) == 0:
                    continue
                
                # Among same cell type neighbors, count different batches
                neighbor_batches = batch_labels[same_celltype_neighbors]
                unique_batches = len(np.unique(neighbor_batches))
                total_batches = len(np.unique(batch_labels))
                
                mixing_scores.append(unique_batches / total_batches)
            
            return np.mean(mixing_scores) if mixing_scores else 0.0
        
        except Exception as e:
            logger.error(f"Failed to compute batch mixing score: {e}")
            return 0.0
    
    def gene_set_enrichment(self,
                           embeddings: np.ndarray,
                           expression_data: np.ndarray,
                           gene_names: List[str],
                           gene_set: str = 'cell_cycle') -> float:
        """Compute gene set enrichment in embedding space.
        
        Args:
            embeddings: Cell embeddings
            expression_data: Gene expression matrix
            gene_names: Names of genes
            gene_set: Gene set to test
            
        Returns:
            Enrichment score
        """
        try:
            if gene_set not in self.gene_sets:
                logger.warning(f"Unknown gene set: {gene_set}")
                return 0.0
            
            target_genes = self.gene_sets[gene_set]
            gene_indices = [i for i, gene in enumerate(gene_names) if gene in target_genes]
            
            if len(gene_indices) < 2:
                logger.warning(f"Insufficient genes found for {gene_set}")
                return 0.0
            
            # Compute gene set activity
            gene_set_activity = np.mean(expression_data[:, gene_indices], axis=1)
            
            # Find correlation with embedding dimensions
            max_correlation = 0
            for dim in range(embeddings.shape[1]):
                corr, _ = pearsonr(gene_set_activity, embeddings[:, dim])
                max_correlation = max(max_correlation, abs(corr))
            
            return max_correlation
        
        except Exception as e:
            logger.error(f"Failed to compute gene set enrichment: {e}")
            return 0.0


class GraphMetrics:
    """Graph-specific evaluation metrics."""
    
    def edge_prediction_auc(self,
                           true_edges: torch.Tensor,
                           predicted_edges: torch.Tensor,
                           negative_sampling_ratio: float = 1.0) -> float:
        """Compute AUC for edge prediction task.
        
        Args:
            true_edges: Ground truth edges [2, n_edges]
            predicted_edges: Predicted edge probabilities or scores
            negative_sampling_ratio: Ratio of negative to positive samples
            
        Returns:
            AUC score
        """
        try:
            from sklearn.metrics import roc_auc_score
            
            # Create adjacency matrix from true edges
            n_nodes = max(torch.max(true_edges[0]), torch.max(true_edges[1])) + 1
            adj_matrix = torch.zeros(n_nodes, n_nodes)
            adj_matrix[true_edges[0], true_edges[1]] = 1
            
            # Sample negative edges
            n_pos = true_edges.shape[1]
            n_neg = int(n_pos * negative_sampling_ratio)
            
            # Generate negative samples
            neg_edges = []
            while len(neg_edges) < n_neg:
                i, j = torch.randint(0, n_nodes, (2,))
                if i != j and adj_matrix[i, j] == 0:
                    neg_edges.append((i.item(), j.item()))
            
            # Create labels
            y_true = [1] * n_pos + [0] * n_neg
            
            # Get predictions for positive and negative edges
            pos_scores = predicted_edges[:n_pos].cpu().numpy()
            neg_scores = predicted_edges[n_pos:n_pos+n_neg].cpu().numpy()
            y_pred = np.concatenate([pos_scores, neg_scores])
            
            return roc_auc_score(y_true, y_pred)
        
        except Exception as e:
            logger.error(f"Failed to compute edge prediction AUC: {e}")
            return 0.0
    
    def modularity_score(self,
                        graph: torch.Tensor,
                        communities: np.ndarray) -> float:
        """Compute modularity score for community detection.
        
        Args:
            graph: Graph adjacency matrix or edge index
            communities: Community assignments
            
        Returns:
            Modularity score
        """
        try:
            import networkx as nx
            
            # Convert to NetworkX graph
            if graph.shape[0] == 2:  # Edge index format
                edge_list = graph.t().cpu().numpy()
                G = nx.Graph()
                G.add_edges_from(edge_list)
            else:  # Adjacency matrix
                G = nx.from_numpy_array(graph.cpu().numpy())
            
            # Create partition dictionary
            partition = {i: communities[i] for i in range(len(communities))}
            
            # Compute modularity
            return nx.algorithms.community.modularity(G, partition.values())
        
        except Exception as e:
            logger.error(f"Failed to compute modularity score: {e}")
            return 0.0
    
    def graph_properties_preservation(self,
                                    original_graph: torch.Tensor,
                                    embedded_graph: torch.Tensor,
                                    properties: List[str] = ['degree_dist', 'clustering_coef', 'path_length']) -> Dict[str, float]:
        """Measure preservation of graph properties.
        
        Args:
            original_graph: Original graph edge index
            embedded_graph: Reconstructed/embedded graph edge index
            properties: Properties to compare
            
        Returns:
            Dictionary of preservation scores
        """
        preservation_scores = {}
        
        try:
            import networkx as nx
            
            # Convert to NetworkX graphs
            G_orig = nx.Graph()
            G_orig.add_edges_from(original_graph.t().cpu().numpy())
            
            G_emb = nx.Graph()
            G_emb.add_edges_from(embedded_graph.t().cpu().numpy())
            
            for prop in properties:
                if prop == 'degree_dist':
                    # Compare degree distributions
                    degrees_orig = [d for n, d in G_orig.degree()]
                    degrees_emb = [d for n, d in G_emb.degree()]
                    
                    # Pad shorter list with zeros
                    max_len = max(len(degrees_orig), len(degrees_emb))
                    degrees_orig.extend([0] * (max_len - len(degrees_orig)))
                    degrees_emb.extend([0] * (max_len - len(degrees_emb)))
                    
                    corr, _ = pearsonr(degrees_orig, degrees_emb)
                    preservation_scores['degree_distribution'] = abs(corr)
                
                elif prop == 'clustering_coef':
                    # Compare clustering coefficients
                    clustering_orig = nx.average_clustering(G_orig)
                    clustering_emb = nx.average_clustering(G_emb)
                    
                    # Compute similarity (1 - normalized difference)
                    max_clustering = max(clustering_orig, clustering_emb, 1e-6)
                    preservation_scores['clustering_coefficient'] = 1 - abs(clustering_orig - clustering_emb) / max_clustering
                
                elif prop == 'path_length':
                    # Compare average shortest path lengths
                    if nx.is_connected(G_orig) and nx.is_connected(G_emb):
                        path_orig = nx.average_shortest_path_length(G_orig)
                        path_emb = nx.average_shortest_path_length(G_emb)
                        
                        max_path = max(path_orig, path_emb, 1e-6)
                        preservation_scores['path_length'] = 1 - abs(path_orig - path_emb) / max_path
                    else:
                        preservation_scores['path_length'] = 0.0
        
        except Exception as e:
            logger.error(f"Failed to compute graph properties preservation: {e}")
        
        return preservation_scores
    
    def graph_reconstruction_error(self,
                                  original_adj: torch.Tensor,
                                  reconstructed_adj: torch.Tensor,
                                  metric: str = 'mse') -> float:
        """Compute graph reconstruction error.
        
        Args:
            original_adj: Original adjacency matrix
            reconstructed_adj: Reconstructed adjacency matrix
            metric: Error metric ('mse', 'mae', 'binary_crossentropy')
            
        Returns:
            Reconstruction error
        """
        try:
            if metric == 'mse':
                return torch.mean((original_adj - reconstructed_adj) ** 2).item()
            elif metric == 'mae':
                return torch.mean(torch.abs(original_adj - reconstructed_adj)).item()
            elif metric == 'binary_crossentropy':
                # Treat as binary classification
                bce = torch.nn.BCEWithLogitsLoss()
                return bce(reconstructed_adj, original_adj.float()).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        except Exception as e:
            logger.error(f"Failed to compute reconstruction error: {e}")
            return float('inf')


class ClusteringMetrics:
    """Clustering evaluation metrics."""
    
    @staticmethod
    def adjusted_rand_index(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """Compute Adjusted Rand Index."""
        try:
            return adjusted_rand_score(true_labels, pred_labels)
        except Exception as e:
            logger.error(f"Failed to compute ARI: {e}")
            return 0.0
    
    @staticmethod
    def normalized_mutual_information(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """Compute Normalized Mutual Information."""
        try:
            return normalized_mutual_info_score(true_labels, pred_labels)
        except Exception as e:
            logger.error(f"Failed to compute NMI: {e}")
            return 0.0
    
    @staticmethod
    def homogeneity_completeness_v_measure(true_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[float, float, float]:
        """Compute homogeneity, completeness, and V-measure."""
        try:
            h = homogeneity_score(true_labels, pred_labels)
            c = completeness_score(true_labels, pred_labels)
            v = v_measure_score(true_labels, pred_labels)
            return h, c, v
        except Exception as e:
            logger.error(f"Failed to compute clustering metrics: {e}")
            return 0.0, 0.0, 0.0


class TrajectoryMetrics:
    """Trajectory inference evaluation metrics."""
    
    @staticmethod
    def kendall_tau_trajectory(true_pseudotime: np.ndarray, pred_pseudotime: np.ndarray) -> float:
        """Compute Kendall's tau for trajectory ordering."""
        try:
            tau, _ = kendalltau(true_pseudotime, pred_pseudotime)
            return abs(tau)
        except Exception as e:
            logger.error(f"Failed to compute Kendall's tau: {e}")
            return 0.0
    
    @staticmethod
    def branch_assignment_accuracy(true_branches: np.ndarray, pred_branches: np.ndarray) -> float:
        """Compute accuracy of branch assignment."""
        try:
            return np.mean(true_branches == pred_branches)
        except Exception as e:
            logger.error(f"Failed to compute branch accuracy: {e}")
            return 0.0
    
    @staticmethod
    def trajectory_conservation_score(original_distances: np.ndarray, embedded_distances: np.ndarray) -> float:
        """Measure preservation of cell-cell distances along trajectory."""
        try:
            corr, _ = pearsonr(original_distances.flatten(), embedded_distances.flatten())
            return abs(corr)
        except Exception as e:
            logger.error(f"Failed to compute trajectory conservation: {e}")
            return 0.0


# Convenience function to compute all relevant metrics
def compute_comprehensive_metrics(predictions: torch.Tensor,
                                 embeddings: torch.Tensor,
                                 ground_truth: torch.Tensor,
                                 task_type: str,
                                 additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for a given task.
    
    Args:
        predictions: Model predictions
        embeddings: Model embeddings
        ground_truth: Ground truth labels/values
        task_type: Type of task ('classification', 'regression', 'clustering')
        additional_data: Additional data for biological metrics
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Convert to numpy for sklearn compatibility
    pred_np = predictions.cpu().numpy()
    emb_np = embeddings.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()
    
    try:
        if task_type == 'classification':
            pred_labels = np.argmax(pred_np, axis=1)
            
            # Standard classification metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            metrics['accuracy'] = accuracy_score(gt_np, pred_labels)
            metrics['f1_macro'] = f1_score(gt_np, pred_labels, average='macro')
            metrics['f1_weighted'] = f1_score(gt_np, pred_labels, average='weighted')
            metrics['precision_macro'] = precision_score(gt_np, pred_labels, average='macro')
            metrics['recall_macro'] = recall_score(gt_np, pred_labels, average='macro')
            
            # Clustering metrics
            clustering_metrics = ClusteringMetrics()
            metrics['ari'] = clustering_metrics.adjusted_rand_index(gt_np, pred_labels)
            metrics['nmi'] = clustering_metrics.normalized_mutual_information(gt_np, pred_labels)
            
            # Biological metrics if cell types are available
            if len(np.unique(gt_np)) > 1:
                bio_metrics = BiologicalMetrics()
                metrics['silhouette_score'] = bio_metrics.cell_type_purity(emb_np, gt_np, 'silhouette')
        
        elif task_type == 'regression':
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics['mse'] = mean_squared_error(gt_np, pred_np.flatten())
            metrics['mae'] = mean_absolute_error(gt_np, pred_np.flatten())
            metrics['r2'] = r2_score(gt_np, pred_np.flatten())
            
            # Correlation metrics
            metrics['pearson_r'], _ = pearsonr(gt_np, pred_np.flatten())
            metrics['spearman_r'], _ = spearmanr(gt_np, pred_np.flatten())
        
        elif task_type == 'trajectory':
            trajectory_metrics = TrajectoryMetrics()
            metrics['kendall_tau'] = trajectory_metrics.kendall_tau_trajectory(gt_np, pred_np.flatten())
            metrics['pearson_r'], _ = pearsonr(gt_np, pred_np.flatten())
        
        # Add biological conservation if original data available
        if additional_data and 'original_data' in additional_data:
            bio_metrics = BiologicalMetrics()
            metrics['biological_conservation'] = bio_metrics.biological_conservation(
                additional_data['original_data'], emb_np
            )
    
    except Exception as e:
        logger.error(f"Failed to compute comprehensive metrics: {e}")
    
    return metrics
