"""
demo_cell_type_gnn.py — Single-cell GNN vs k-means benchmark.

Generates 200 synthetic cells (50 genes, 5 cell types), builds a k-NN
graph, trains a 2-layer GCN, and compares against naive k-means clustering
on raw expression features.

Usage:
    python examples/demo_cell_type_gnn.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from scgraph_hub import (
    CellClusterEvaluator,
    CellTypeGNN,
    SingleCellGraph,
    SyntheticSCDataset,
    train_gcn,
)


def main():
    print("=" * 60)
    print("  Single-Cell GNN — Cell Type Classification Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Generate synthetic scRNA-seq data
    # ------------------------------------------------------------------
    print("\n[1] Generating synthetic scRNA-seq data...")
    dataset = SyntheticSCDataset(n_cells=200, n_genes=50, n_cell_types=5, seed=42)
    print(f"    {dataset}")

    # ------------------------------------------------------------------
    # 2. Build k-NN cell-cell similarity graph
    # ------------------------------------------------------------------
    print("\n[2] Building k-NN graph (k=10, cosine similarity)...")
    graph = SingleCellGraph(dataset.expression, labels=dataset.labels, k=10, metric="cosine")
    print(f"    {graph}")

    # ------------------------------------------------------------------
    # 3. Train/test split
    # ------------------------------------------------------------------
    train_mask_np, test_mask_np = dataset.train_test_split(train_frac=0.8, seed=0)
    n_train = train_mask_np.sum()
    n_test = test_mask_np.sum()
    print(f"\n[3] Train/test split: {n_train} / {n_test} cells")

    # ------------------------------------------------------------------
    # 4. Train GCN
    # ------------------------------------------------------------------
    print("\n[4] Training 2-layer GCN (200 epochs)...")
    tensors = graph.to_torch()
    x, adj, y = tensors["x"], tensors["adj"], tensors["y"]
    train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

    model = CellTypeGNN(
        n_genes=dataset.n_genes,
        n_cell_types=dataset.n_cell_types,
        hidden_dim=64,
        dropout=0.5,
    )

    loss_history = train_gcn(
        model, x, adj, y, train_mask,
        epochs=200, lr=0.01, weight_decay=5e-4,
        verbose=True, log_every=50,
    )

    # ------------------------------------------------------------------
    # 5. Evaluate GCN on test set
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        log_probs = model(x)
    gcn_preds = log_probs.argmax(dim=1).numpy()

    evaluator = CellClusterEvaluator(dataset.labels)

    print("\n[5] GCN Test Results:")
    test_preds = gcn_preds[test_mask_np]
    test_true = dataset.labels[test_mask_np]
    test_eval = CellClusterEvaluator(test_true)
    print("   ", test_eval.report(test_preds, "GCN (test set)").replace("\n", "\n    "))

    # ------------------------------------------------------------------
    # 6. Baseline: k-means on raw expression features
    # ------------------------------------------------------------------
    print("\n[6] Baseline: k-means on L2-normalized expression features...")
    features_norm = normalize(dataset.expression, norm="l2")
    kmeans = KMeans(n_clusters=dataset.n_cell_types, random_state=42, n_init=10)
    kmeans_preds = kmeans.fit_predict(features_norm)

    print("   ", evaluator.report(kmeans_preds, "k-means (all cells)").replace("\n", "\n    "))

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    gcn_full = evaluator.evaluate(gcn_preds)
    km_full = evaluator.evaluate(kmeans_preds)

    print("\n" + "=" * 60)
    print("  Summary: GCN vs k-means (all cells)")
    print("=" * 60)
    print(f"  {'Method':<20} {'Accuracy':>10} {'ARI':>10}")
    print(f"  {'-'*42}")
    print(f"  {'GCN (2-layer)':<20} {gcn_full['accuracy']:>10.4f} {gcn_full['ari']:>10.4f}")
    print(f"  {'k-means':<20} {km_full['accuracy']:>10.4f} {km_full['ari']:>10.4f}")
    print("=" * 60)
    print("\nNote: k-means accuracy uses raw assignments (not Hungarian-matched),")
    print("so ARI is the fairer unsupervised comparison metric.")

    # Final training loss
    print(f"\nFinal training loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()
