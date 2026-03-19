"""
single-cell-graph-hub — Graph neural network toolkit for single-cell omics analysis.

Core components:
    SingleCellGraph      : Cell-cell k-NN graph from expression data
    GraphBuilder         : Expression matrix → k-NN adjacency
    CellTypeGNN          : 2-layer GCN for cell-type classification
    SyntheticSCDataset   : Synthetic scRNA-seq data generator
    CellClusterEvaluator : ARI + accuracy metrics
"""

from .dataset import SyntheticSCDataset
from .evaluation import CellClusterEvaluator, adjusted_rand_index, classification_accuracy
from .graph import GraphBuilder, SingleCellGraph
from .models import CellTypeGNN, GCNLayer, normalize_adjacency
from .train import train_gcn

__version__ = "1.0.0"
__all__ = [
    "SingleCellGraph",
    "GraphBuilder",
    "CellTypeGNN",
    "GCNLayer",
    "normalize_adjacency",
    "SyntheticSCDataset",
    "CellClusterEvaluator",
    "adjusted_rand_index",
    "classification_accuracy",
    "train_gcn",
]
