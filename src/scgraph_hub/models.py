"""
models.py — Graph Convolutional Network for cell-type classification.

Pure PyTorch implementation — no PyG or DGL required.
Uses symmetric normalized graph convolution: Â = D^{-1/2} A D^{-1/2}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Symmetrically normalize an adjacency matrix.

    Computes: Â = D^{-1/2} (A + I) D^{-1/2}

    Parameters
    ----------
    adj : torch.Tensor
        Shape (N, N), raw adjacency matrix (binary or weighted).
    add_self_loops : bool
        Add identity before normalization (standard GCN trick).

    Returns
    -------
    norm_adj : torch.Tensor
        Shape (N, N), normalized adjacency.
    """
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)

    deg = adj.sum(dim=1)  # degree vector
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    return D_inv_sqrt @ adj @ D_inv_sqrt


class GCNLayer(nn.Module):
    """
    Single graph convolutional layer.

    Implements: H' = σ(Â · H · W + b)

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    bias : bool
        Use bias term (default: True).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        # Glorot uniform initialization
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (N, in_features).
        norm_adj : torch.Tensor
            Normalized adjacency, shape (N, N).

        Returns
        -------
        h : torch.Tensor
            Shape (N, out_features).
        """
        support = self.linear(x)          # (N, out_features)
        return norm_adj @ support          # graph aggregation


class CellTypeGNN(nn.Module):
    """
    2-layer Graph Convolutional Network for cell-type classification.

    Architecture:
        Input (G genes) → GCN layer 1 (hidden_dim) → ReLU + Dropout
                        → GCN layer 2 (n_cell_types) → Log-softmax

    The normalized adjacency is precomputed once and reused every forward pass.

    Parameters
    ----------
    n_genes : int
        Number of input gene features.
    n_cell_types : int
        Number of output classes (cell types).
    hidden_dim : int
        Hidden layer dimension (default: 64).
    dropout : float
        Dropout rate between layers (default: 0.5).
    """

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.gcn1 = GCNLayer(n_genes, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, n_cell_types)
        self.dropout = nn.Dropout(p=dropout)
        self._norm_adj: torch.Tensor | None = None

    def precompute_adjacency(self, adj: torch.Tensor) -> None:
        """
        Precompute and cache the normalized adjacency matrix.

        Call this once before training to avoid recomputing every forward pass.

        Parameters
        ----------
        adj : torch.Tensor
            Raw adjacency matrix, shape (N, N).
        """
        self._norm_adj = normalize_adjacency(adj)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (N, n_genes).
        adj : torch.Tensor or None
            Raw adjacency (N, N). If None, uses precomputed normalized adj.

        Returns
        -------
        log_probs : torch.Tensor
            Shape (N, n_cell_types), log-softmax probabilities.
        """
        if adj is not None:
            norm_adj = normalize_adjacency(adj)
        elif self._norm_adj is not None:
            norm_adj = self._norm_adj
        else:
            raise ValueError("Must provide adj or call precompute_adjacency() first.")

        h = self.gcn1(x, norm_adj)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.gcn2(h, norm_adj)
        return F.log_softmax(h, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return integer class predictions (argmax)."""
        with torch.no_grad():
            log_probs = self.forward(x)
        return log_probs.argmax(dim=1)
