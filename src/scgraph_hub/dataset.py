"""
dataset.py — Synthetic single-cell RNA-seq dataset generator.

Produces realistic-enough synthetic data for testing GNN pipelines:
- N cells across K cell types
- G genes per cell
- Each cell type has a distinct mean expression profile
- Within-type variation simulates biological + technical noise
"""

from __future__ import annotations

import numpy as np


class SyntheticSCDataset:
    """
    Generates synthetic scRNA-seq-like data for benchmarking.

    Each cell type has a random mean expression vector drawn from a
    Dirichlet-like prior, with log-normal noise added per cell to mimic
    the overdispersion typical of UMI count data.

    Parameters
    ----------
    n_cells : int
        Total number of cells (evenly split across cell types).
    n_genes : int
        Number of genes (features per cell).
    n_cell_types : int
        Number of distinct cell types.
    noise_scale : float
        Standard deviation of log-normal noise (default: 0.5).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_cells: int = 200,
        n_genes: int = 50,
        n_cell_types: int = 5,
        noise_scale: float = 0.5,
        seed: int | None = 42,
    ):
        if n_cells < n_cell_types:
            raise ValueError("n_cells must be >= n_cell_types")
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.noise_scale = noise_scale
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._expression, self._labels, self._type_means = self._generate()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = self._rng

        # Each cell type: a random mean expression profile (sparse-ish, positive)
        # Use Dirichlet-scaled random vectors to ensure distinct profiles
        type_means = np.zeros((self.n_cell_types, self.n_genes), dtype=np.float32)
        for k in range(self.n_cell_types):
            # Random subset of genes "activated" for this type
            alpha = rng.uniform(0.1, 1.0, self.n_genes)
            # Make ~30% of genes highly expressed per type (marker genes)
            n_markers = max(1, self.n_genes // 3)
            marker_idx = rng.choice(self.n_genes, n_markers, replace=False)
            alpha[marker_idx] *= 5.0 * (k + 1)  # type-specific amplitude
            type_means[k] = alpha

        # Assign cells to types (round-robin for balanced classes)
        cells_per_type = self.n_cells // self.n_cell_types
        remainder = self.n_cells % self.n_cell_types
        labels = np.repeat(
            np.arange(self.n_cell_types),
            [cells_per_type + (1 if i < remainder else 0) for i in range(self.n_cell_types)],
        ).astype(np.int64)

        # Generate expression: mean + log-normal noise
        expression = np.zeros((self.n_cells, self.n_genes), dtype=np.float32)
        for i, k in enumerate(labels):
            log_noise = rng.normal(0, self.noise_scale, self.n_genes)
            expression[i] = type_means[k] * np.exp(log_noise)

        # Shuffle cell order
        perm = rng.permutation(self.n_cells)
        expression = expression[perm]
        labels = labels[perm]

        return expression, labels, type_means

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def expression(self) -> np.ndarray:
        """Gene expression matrix, shape (n_cells, n_genes)."""
        return self._expression

    @property
    def labels(self) -> np.ndarray:
        """Integer cell-type labels, shape (n_cells,)."""
        return self._labels

    @property
    def type_means(self) -> np.ndarray:
        """Mean expression profile per cell type, shape (n_cell_types, n_genes)."""
        return self._type_means

    def train_test_split(
        self, train_frac: float = 0.8, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return boolean train/test masks.

        Parameters
        ----------
        train_frac : float
            Fraction of cells used for training.
        seed : int or None
            Seed for mask generation.

        Returns
        -------
        train_mask, test_mask : np.ndarray of bool, shape (n_cells,)
        """
        rng = np.random.default_rng(seed)
        n_train = int(self.n_cells * train_frac)
        perm = rng.permutation(self.n_cells)
        train_mask = np.zeros(self.n_cells, dtype=bool)
        train_mask[perm[:n_train]] = True
        return train_mask, ~train_mask

    def __repr__(self) -> str:
        return (
            f"SyntheticSCDataset(cells={self.n_cells}, genes={self.n_genes}, "
            f"types={self.n_cell_types}, noise={self.noise_scale})"
        )
