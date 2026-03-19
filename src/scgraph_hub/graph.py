"""
graph.py — Core graph construction for single-cell omics data.

Builds k-NN cell-cell similarity graphs from gene expression matrices.
Nodes = cells, edges = transcriptional similarity, node features = expression vectors.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class GraphBuilder:
    """
    Converts a cell × gene expression matrix into a k-NN graph.

    The k-NN graph captures transcriptional similarity between cells:
    two cells are connected if one is among the other's k nearest neighbors
    in expression space (using cosine or Euclidean distance).

    Parameters
    ----------
    k : int
        Number of nearest neighbors per cell (default: 10).
    metric : str
        Distance metric — 'cosine' or 'euclidean' (default: 'cosine').
    normalize_features : bool
        L2-normalize expression vectors before graph construction (default: True).
    """

    def __init__(self, k: int = 10, metric: str = "cosine", normalize_features: bool = True):
        if k < 1:
            raise ValueError("k must be >= 1")
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.k = k
        self.metric = metric
        self.normalize_features = normalize_features

    def build(self, expression_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a k-NN adjacency and return normalized feature matrix.

        Parameters
        ----------
        expression_matrix : np.ndarray
            Shape (N, G) — N cells × G genes.

        Returns
        -------
        adjacency : np.ndarray
            Shape (N, N), symmetric binary adjacency matrix.
        features : np.ndarray
            Shape (N, G), (optionally) normalized expression features.
        """
        N, G = expression_matrix.shape
        features = expression_matrix.astype(np.float32)

        if self.normalize_features:
            features = normalize(features, norm="l2")

        # sklearn cosine uses (1 - cosine_similarity), so 'cosine' metric works directly
        nn = NearestNeighbors(n_neighbors=min(self.k + 1, N), metric=self.metric)
        nn.fit(features)
        distances, indices = nn.kneighbors(features)

        # Build symmetric adjacency (exclude self-loop at index 0)
        adjacency = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in indices[i, 1:]:  # skip self (index 0)
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

        return adjacency, features


class SingleCellGraph:
    """
    A graph representation of single-cell omics data.

    Wraps a cell-cell k-NN graph with its associated expression features,
    optional cell-type labels, and torch tensors ready for GNN training.

    Parameters
    ----------
    expression_matrix : np.ndarray
        Shape (N, G) — raw or normalized gene expression (N cells × G genes).
    labels : np.ndarray or None
        Integer cell-type labels, shape (N,). Optional.
    k : int
        Nearest neighbors for graph construction (default: 10).
    metric : str
        Distance metric for k-NN (default: 'cosine').
    """

    def __init__(
        self,
        expression_matrix: np.ndarray,
        labels: np.ndarray | None = None,
        k: int = 10,
        metric: str = "cosine",
    ):
        self.expression_matrix = expression_matrix.astype(np.float32)
        self.labels = labels
        self.n_cells, self.n_genes = expression_matrix.shape

        builder = GraphBuilder(k=k, metric=metric, normalize_features=True)
        self.adjacency, self.features = builder.build(expression_matrix)

    # ------------------------------------------------------------------
    # Torch helpers
    # ------------------------------------------------------------------

    def to_torch(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        """
        Return graph components as torch tensors.

        Returns a dict with keys:
          - 'x'  : node features (N, G)
          - 'adj': adjacency matrix (N, N)
          - 'y'  : labels (N,) if available, else None
        """
        tensors: dict[str, torch.Tensor | None] = {
            "x": torch.tensor(self.features, dtype=torch.float32, device=device),
            "adj": torch.tensor(self.adjacency, dtype=torch.float32, device=device),
            "y": None,
        }
        if self.labels is not None:
            tensors["y"] = torch.tensor(self.labels, dtype=torch.long, device=device)
        return tensors

    @property
    def n_edges(self) -> int:
        """Number of undirected edges in the graph."""
        return int(self.adjacency.sum()) // 2

    def __repr__(self) -> str:
        label_str = f", {len(np.unique(self.labels))} types" if self.labels is not None else ""
        return (
            f"SingleCellGraph(cells={self.n_cells}, genes={self.n_genes}, "
            f"edges={self.n_edges}{label_str})"
        )
