"""
evaluation.py — Clustering and classification quality metrics.

Provides adjusted rand index (ARI) and accuracy metrics for comparing
predicted cell-type assignments against ground-truth labels.
"""

from __future__ import annotations

import numpy as np


def _contingency_table(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Build a contingency table for two label arrays."""
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    table = np.zeros((len(classes_true), len(classes_pred)), dtype=np.int64)
    for i, ct in enumerate(classes_true):
        for j, cp in enumerate(classes_pred):
            table[i, j] = np.sum((labels_true == ct) & (labels_pred == cp))
    return table


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute the Adjusted Rand Index (ARI) between two clusterings.

    ARI = 1.0 means perfect agreement; ~0 means random; can be negative.

    Parameters
    ----------
    labels_true : np.ndarray
        Ground-truth integer labels, shape (N,).
    labels_pred : np.ndarray
        Predicted integer labels, shape (N,).

    Returns
    -------
    ari : float
        Adjusted Rand Index in [-1, 1].
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if len(labels_true) != len(labels_pred):
        raise ValueError("Label arrays must have the same length.")

    table = _contingency_table(labels_true, labels_pred)
    n = labels_true.shape[0]

    # Sum of C(n_ij, 2)
    sum_comb_c = float(np.sum(table * (table - 1)) / 2)

    # Row and column sums
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    sum_comb_a = float(np.sum(row_sums * (row_sums - 1)) / 2)
    sum_comb_b = float(np.sum(col_sums * (col_sums - 1)) / 2)

    # Total pairs
    total_comb = float(n * (n - 1) / 2)

    expected_index = sum_comb_a * sum_comb_b / total_comb
    max_index = (sum_comb_a + sum_comb_b) / 2

    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0 if sum_comb_c == expected_index else 0.0

    return (sum_comb_c - expected_index) / denominator


def classification_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute cell-type classification accuracy.

    Parameters
    ----------
    labels_true : np.ndarray
        Ground-truth integer labels, shape (N,).
    labels_pred : np.ndarray
        Predicted integer labels, shape (N,).

    Returns
    -------
    accuracy : float
        Fraction of correctly classified cells.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    return float(np.mean(labels_true == labels_pred))


class CellClusterEvaluator:
    """
    Evaluates clustering and classification quality for single-cell data.

    Compares predicted cell-type assignments (from GNN or clustering)
    against ground-truth labels using ARI and accuracy.

    Parameters
    ----------
    labels_true : np.ndarray
        Ground-truth cell-type labels, shape (N,).
    """

    def __init__(self, labels_true: np.ndarray):
        self.labels_true = np.asarray(labels_true)

    def evaluate(self, labels_pred: np.ndarray) -> dict[str, float]:
        """
        Compute all metrics for a set of predicted labels.

        Parameters
        ----------
        labels_pred : np.ndarray
            Predicted labels, shape (N,).

        Returns
        -------
        metrics : dict
            Keys: 'ari', 'accuracy', 'n_cells', 'n_types_true', 'n_types_pred'.
        """
        labels_pred = np.asarray(labels_pred)
        return {
            "ari": adjusted_rand_index(self.labels_true, labels_pred),
            "accuracy": classification_accuracy(self.labels_true, labels_pred),
            "n_cells": len(self.labels_true),
            "n_types_true": int(np.unique(self.labels_true).size),
            "n_types_pred": int(np.unique(labels_pred).size),
        }

    def report(self, labels_pred: np.ndarray, method_name: str = "Method") -> str:
        """
        Return a formatted evaluation report string.

        Parameters
        ----------
        labels_pred : np.ndarray
            Predicted labels.
        method_name : str
            Label for the method being evaluated.

        Returns
        -------
        report : str
            Human-readable metrics string.
        """
        m = self.evaluate(labels_pred)
        return (
            f"{method_name}:\n"
            f"  Accuracy : {m['accuracy']:.4f} ({m['accuracy']*100:.1f}%)\n"
            f"  ARI      : {m['ari']:.4f}\n"
            f"  Cells    : {m['n_cells']}\n"
            f"  Types (true/pred): {m['n_types_true']}/{m['n_types_pred']}"
        )
