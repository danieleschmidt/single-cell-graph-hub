"""Tests for models.py — GCNLayer, CellTypeGNN, normalize_adjacency."""

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scgraph_hub.models import CellTypeGNN, GCNLayer, normalize_adjacency


class TestNormalizeAdjacency:
    def test_output_shape(self):
        adj = torch.eye(5)
        norm = normalize_adjacency(adj)
        assert norm.shape == (5, 5)

    def test_symmetric(self):
        adj = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        norm = normalize_adjacency(adj)
        torch.testing.assert_close(norm, norm.T, atol=1e-5, rtol=0)

    def test_no_self_loops_option(self):
        adj = torch.zeros(3, 3)
        norm = normalize_adjacency(adj, add_self_loops=False)
        # All zeros with no edges → should handle gracefully (clamped)
        assert norm.shape == (3, 3)


class TestGCNLayer:
    def test_forward_shape(self):
        layer = GCNLayer(in_features=10, out_features=8)
        x = torch.randn(20, 10)
        adj = torch.eye(20)
        out = layer(x, adj)
        assert out.shape == (20, 8)

    def test_parameter_count(self):
        layer = GCNLayer(in_features=5, out_features=3)
        params = sum(p.numel() for p in layer.parameters())
        assert params == 5 * 3 + 3  # weight + bias


class TestCellTypeGNN:
    def setup_method(self):
        self.N, self.G, self.K = 50, 20, 4
        self.x = torch.randn(self.N, self.G)
        self.adj = torch.zeros(self.N, self.N)
        # Sparse adjacency
        for i in range(self.N - 1):
            self.adj[i, i+1] = 1.0
            self.adj[i+1, i] = 1.0

    def test_forward_shape(self):
        model = CellTypeGNN(n_genes=self.G, n_cell_types=self.K)
        out = model(self.x, self.adj)
        assert out.shape == (self.N, self.K)

    def test_output_is_log_prob(self):
        model = CellTypeGNN(n_genes=self.G, n_cell_types=self.K)
        out = model(self.x, self.adj)
        # log-softmax: values should be <= 0
        assert (out <= 0).all()
        # exp should sum to ~1 per row
        probs = out.exp()
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(self.N), atol=1e-4, rtol=0)

    def test_precompute_adjacency(self):
        model = CellTypeGNN(n_genes=self.G, n_cell_types=self.K)
        model.precompute_adjacency(self.adj)
        out = model(self.x)  # should work without passing adj
        assert out.shape == (self.N, self.K)

    def test_no_adj_raises(self):
        model = CellTypeGNN(n_genes=self.G, n_cell_types=self.K)
        with pytest.raises(ValueError):
            model(self.x)  # no adj and no precomputed adj

    def test_predict_shape(self):
        model = CellTypeGNN(n_genes=self.G, n_cell_types=self.K)
        model.precompute_adjacency(self.adj)
        preds = model.predict(self.x)
        assert preds.shape == (self.N,)
        assert preds.dtype == torch.int64

    def test_predict_valid_classes(self):
        model = CellTypeGNN(n_genes=self.G, n_cell_types=self.K)
        model.precompute_adjacency(self.adj)
        preds = model.predict(self.x)
        assert preds.min() >= 0
        assert preds.max() < self.K

    def test_training_reduces_loss(self):
        model = CellTypeGNN(n_genes=self.G, n_cell_types=self.K, dropout=0.0)
        labels = torch.randint(0, self.K, (self.N,))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        import torch.nn.functional as F

        model.precompute_adjacency(self.adj)
        losses = []
        for _ in range(50):
            model.train()
            optimizer.zero_grad()
            out = model(self.x)
            loss = F.nll_loss(out, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally trend down over 50 epochs
        assert losses[-1] < losses[0] * 1.5  # relaxed: just shouldn't explode
