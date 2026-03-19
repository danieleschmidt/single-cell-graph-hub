"""
train.py — Training utilities for CellTypeGNN.

Semi-supervised transductive training: the full graph is visible during
training, but loss is computed only on the training mask.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import CellTypeGNN


def train_gcn(
    model: CellTypeGNN,
    x: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    verbose: bool = True,
    log_every: int = 50,
) -> list[float]:
    """
    Train a CellTypeGNN with standard semi-supervised GCN training.

    Uses the full graph for message passing, but only computes NLL loss
    on the `train_mask` nodes — standard transductive GNN approach.

    Parameters
    ----------
    model : CellTypeGNN
        The GNN model to train.
    x : torch.Tensor
        Node features, shape (N, G).
    adj : torch.Tensor
        Adjacency matrix, shape (N, N).
    labels : torch.Tensor
        Cell-type labels (int64), shape (N,).
    train_mask : torch.Tensor
        Boolean mask for training nodes, shape (N,).
    epochs : int
        Number of training epochs.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularization weight.
    verbose : bool
        Print loss/accuracy during training.
    log_every : int
        Print every N epochs.

    Returns
    -------
    loss_history : list[float]
        Training loss per epoch.
    """
    model.precompute_adjacency(adj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        log_probs = model(x)
        loss = F.nll_loss(log_probs[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                preds = log_probs.argmax(dim=1)
                train_acc = (preds[train_mask] == labels[train_mask]).float().mean().item()
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.4f} | Train Acc: {train_acc:.4f}")

    return loss_history
