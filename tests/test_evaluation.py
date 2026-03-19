"""Tests for evaluation.py — CellClusterEvaluator, ARI, accuracy."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scgraph_hub.evaluation import (
    CellClusterEvaluator,
    adjusted_rand_index,
    classification_accuracy,
)


class TestAdjustedRandIndex:
    def test_perfect_agreement(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        ari = adjusted_rand_index(labels, labels)
        assert abs(ari - 1.0) < 1e-10

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            adjusted_rand_index(np.array([0, 1, 2]), np.array([0, 1]))

    def test_single_cluster_pred(self):
        # All predicted as same cluster
        true = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 0, 0, 0, 0, 0])
        ari = adjusted_rand_index(true, pred)
        assert ari <= 0.01  # Should be near 0 or negative

    def test_random_labels_near_zero(self):
        rng = np.random.default_rng(42)
        true = rng.integers(0, 5, 200)
        pred = rng.integers(0, 5, 200)
        ari = adjusted_rand_index(true, pred)
        assert abs(ari) < 0.3  # Random should be close to 0

    def test_range(self):
        rng = np.random.default_rng(0)
        true = rng.integers(0, 3, 50)
        pred = rng.integers(0, 3, 50)
        ari = adjusted_rand_index(true, pred)
        assert -1.0 <= ari <= 1.0


class TestClassificationAccuracy:
    def test_perfect(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        acc = classification_accuracy(labels, labels)
        assert acc == 1.0

    def test_zero(self):
        true = np.array([0, 0, 0])
        pred = np.array([1, 1, 1])
        acc = classification_accuracy(true, pred)
        assert acc == 0.0

    def test_half(self):
        true = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 0, 1])
        acc = classification_accuracy(true, pred)
        assert acc == 0.5

    def test_range(self):
        rng = np.random.default_rng(1)
        true = rng.integers(0, 4, 100)
        pred = rng.integers(0, 4, 100)
        acc = classification_accuracy(true, pred)
        assert 0.0 <= acc <= 1.0


class TestCellClusterEvaluator:
    def setup_method(self):
        self.true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.evaluator = CellClusterEvaluator(self.true)

    def test_evaluate_keys(self):
        pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        metrics = self.evaluator.evaluate(pred)
        assert "ari" in metrics
        assert "accuracy" in metrics
        assert "n_cells" in metrics
        assert "n_types_true" in metrics
        assert "n_types_pred" in metrics

    def test_perfect_accuracy(self):
        metrics = self.evaluator.evaluate(self.true)
        assert metrics["accuracy"] == 1.0
        assert abs(metrics["ari"] - 1.0) < 1e-10

    def test_n_cells(self):
        metrics = self.evaluator.evaluate(self.true)
        assert metrics["n_cells"] == 9

    def test_n_types(self):
        metrics = self.evaluator.evaluate(self.true)
        assert metrics["n_types_true"] == 3

    def test_report_string(self):
        report = self.evaluator.report(self.true, method_name="GCN")
        assert "GCN" in report
        assert "Accuracy" in report
        assert "ARI" in report
