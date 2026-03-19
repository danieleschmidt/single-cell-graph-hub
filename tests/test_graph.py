"""Tests for graph.py — GraphBuilder and SingleCellGraph."""

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scgraph_hub.graph import GraphBuilder, SingleCellGraph


class TestGraphBuilder:
    def test_output_shapes(self):
        X = np.random.rand(20, 10).astype(np.float32)
        builder = GraphBuilder(k=5)
        adj, feat = builder.build(X)
        assert adj.shape == (20, 20)
        assert feat.shape == (20, 10)

    def test_adjacency_symmetric(self):
        X = np.random.rand(15, 8).astype(np.float32)
        builder = GraphBuilder(k=3)
        adj, _ = builder.build(X)
        np.testing.assert_array_equal(adj, adj.T)

    def test_no_self_loops(self):
        X = np.random.rand(10, 5).astype(np.float32)
        builder = GraphBuilder(k=3)
        adj, _ = builder.build(X)
        assert np.all(np.diag(adj) == 0)

    def test_binary_adjacency(self):
        X = np.random.rand(10, 5).astype(np.float32)
        builder = GraphBuilder(k=3)
        adj, _ = builder.build(X)
        assert set(np.unique(adj)).issubset({0.0, 1.0})

    def test_normalized_features(self):
        X = np.random.rand(10, 5).astype(np.float32)
        builder = GraphBuilder(k=3, normalize_features=True)
        _, feat = builder.build(X)
        norms = np.linalg.norm(feat, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-6)

    def test_euclidean_metric(self):
        X = np.random.rand(12, 6).astype(np.float32)
        builder = GraphBuilder(k=3, metric="euclidean")
        adj, feat = builder.build(X)
        assert adj.shape == (12, 12)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            GraphBuilder(k=0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            GraphBuilder(metric="manhattan")

    def test_small_dataset(self):
        # k > n_cells-1 should clamp gracefully
        X = np.random.rand(3, 4).astype(np.float32)
        builder = GraphBuilder(k=10)
        adj, feat = builder.build(X)
        assert adj.shape == (3, 3)


class TestSingleCellGraph:
    def setup_method(self):
        np.random.seed(0)
        self.X = np.random.rand(30, 15).astype(np.float32)
        self.labels = np.random.randint(0, 3, 30)

    def test_basic_construction(self):
        g = SingleCellGraph(self.X, labels=self.labels, k=5)
        assert g.n_cells == 30
        assert g.n_genes == 15

    def test_repr(self):
        g = SingleCellGraph(self.X, labels=self.labels, k=5)
        r = repr(g)
        assert "SingleCellGraph" in r
        assert "cells=30" in r

    def test_to_torch_keys(self):
        g = SingleCellGraph(self.X, labels=self.labels, k=5)
        tensors = g.to_torch()
        assert "x" in tensors
        assert "adj" in tensors
        assert "y" in tensors

    def test_to_torch_shapes(self):
        g = SingleCellGraph(self.X, labels=self.labels, k=5)
        t = g.to_torch()
        assert t["x"].shape == (30, 15)
        assert t["adj"].shape == (30, 30)
        assert t["y"].shape == (30,)
        assert isinstance(t["x"], torch.Tensor)

    def test_no_labels(self):
        g = SingleCellGraph(self.X, labels=None, k=5)
        t = g.to_torch()
        assert t["y"] is None

    def test_n_edges_positive(self):
        g = SingleCellGraph(self.X, labels=self.labels, k=5)
        assert g.n_edges > 0
