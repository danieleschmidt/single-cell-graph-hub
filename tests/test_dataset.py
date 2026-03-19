"""Tests for dataset.py — SyntheticSCDataset."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scgraph_hub.dataset import SyntheticSCDataset


class TestSyntheticSCDataset:
    def test_output_shapes(self):
        ds = SyntheticSCDataset(n_cells=100, n_genes=20, n_cell_types=4, seed=0)
        assert ds.expression.shape == (100, 20)
        assert ds.labels.shape == (100,)
        assert ds.type_means.shape == (4, 20)

    def test_label_range(self):
        ds = SyntheticSCDataset(n_cells=50, n_genes=10, n_cell_types=3, seed=1)
        assert set(np.unique(ds.labels)).issubset(set(range(3)))

    def test_all_types_present(self):
        ds = SyntheticSCDataset(n_cells=60, n_genes=10, n_cell_types=3, seed=2)
        assert len(np.unique(ds.labels)) == 3

    def test_expression_positive(self):
        ds = SyntheticSCDataset(n_cells=50, n_genes=10, n_cell_types=3, seed=3)
        assert np.all(ds.expression >= 0)

    def test_reproducibility(self):
        ds1 = SyntheticSCDataset(n_cells=50, n_genes=10, n_cell_types=3, seed=99)
        ds2 = SyntheticSCDataset(n_cells=50, n_genes=10, n_cell_types=3, seed=99)
        np.testing.assert_array_equal(ds1.expression, ds2.expression)
        np.testing.assert_array_equal(ds1.labels, ds2.labels)

    def test_different_seeds_differ(self):
        ds1 = SyntheticSCDataset(n_cells=50, n_genes=10, n_cell_types=3, seed=1)
        ds2 = SyntheticSCDataset(n_cells=50, n_genes=10, n_cell_types=3, seed=2)
        assert not np.array_equal(ds1.expression, ds2.expression)

    def test_train_test_split(self):
        ds = SyntheticSCDataset(n_cells=100, n_genes=10, n_cell_types=3, seed=0)
        train, test = ds.train_test_split(train_frac=0.8)
        assert train.sum() == 80
        assert test.sum() == 20
        assert np.all(train ^ test)  # mutually exclusive

    def test_too_few_cells_raises(self):
        with pytest.raises(ValueError):
            SyntheticSCDataset(n_cells=2, n_genes=10, n_cell_types=5)

    def test_repr(self):
        ds = SyntheticSCDataset(n_cells=50, n_genes=10, n_cell_types=3)
        assert "SyntheticSCDataset" in repr(ds)

    def test_default_200_cells(self):
        ds = SyntheticSCDataset()
        assert ds.n_cells == 200
        assert ds.n_genes == 50
        assert ds.n_cell_types == 5
