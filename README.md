# single-cell-graph-hub

Graph neural network toolkit for single-cell omics analysis.

Models cell-cell interaction networks as graphs — nodes are cells, edges are transcriptional similarity, node features are gene expression vectors. Trains GCNs to classify cell types in a semi-supervised, transductive setting.

---

## What it does

Single-cell RNA sequencing produces high-dimensional expression profiles for thousands of cells. A natural way to analyze these is as a graph: connect cells that look transcriptionally similar (k-NN in expression space), then run a GNN to classify cell types using both local expression and neighborhood context.

This toolkit provides:

- **`SingleCellGraph`** — builds a cell-cell k-NN graph from a gene expression matrix
- **`GraphBuilder`** — converts expression matrix → k-NN adjacency with optional L2 normalization
- **`CellTypeGNN`** — 2-layer GCN for cell-type classification (pure PyTorch, no PyG required)
- **`SyntheticSCDataset`** — generates synthetic scRNA-seq-like data for benchmarking
- **`CellClusterEvaluator`** — measures clustering quality with ARI and accuracy

---

## Quick start

```python
from scgraph_hub import (
    SyntheticSCDataset,
    SingleCellGraph,
    CellTypeGNN,
    CellClusterEvaluator,
    train_gcn,
)
import torch

# 1. Generate synthetic data (or load your own expression matrix)
dataset = SyntheticSCDataset(n_cells=200, n_genes=50, n_cell_types=5, seed=42)

# 2. Build k-NN cell-cell similarity graph
graph = SingleCellGraph(dataset.expression, labels=dataset.labels, k=10)
print(graph)  # SingleCellGraph(cells=200, genes=50, edges=1347, 5 types)

# 3. Train a 2-layer GCN
tensors = graph.to_torch()
x, adj, y = tensors["x"], tensors["adj"], tensors["y"]

train_mask, test_mask = dataset.train_test_split(train_frac=0.8)
train_mask = torch.tensor(train_mask)

model = CellTypeGNN(n_genes=50, n_cell_types=5, hidden_dim=64)
train_gcn(model, x, adj, y, train_mask, epochs=200, lr=0.01)

# 4. Evaluate
import numpy as np
model.eval()
with torch.no_grad():
    preds = model(x).argmax(dim=1).numpy()

evaluator = CellClusterEvaluator(dataset.labels)
print(evaluator.report(preds, "GCN"))
```

---

## Architecture

### Graph construction

Cells are connected if they are among each other's k nearest neighbors in L2-normalized expression space (cosine similarity by default). The resulting adjacency is symmetric and binary.

### GCN model

Standard 2-layer GCN (Kipf & Welling, 2017):

```
H¹ = ReLU(Â · X · W¹)
H² = softmax(Â · H¹ · W²)
```

where `Â = D⁻¹/² (A + I) D⁻¹/²` is the symmetrically normalized adjacency with self-loops. Training is transductive — the full graph is visible, loss is computed on labeled nodes only.

### Evaluation

- **Adjusted Rand Index (ARI)**: symmetric measure of clustering quality, 1.0 = perfect, ~0 = random
- **Accuracy**: fraction of correctly classified cells (for supervised classification)

---

## Demo

```bash
python examples/demo_cell_type_gnn.py
```

Trains GCN on 200 synthetic cells (50 genes, 5 types) and compares against k-means baseline. Both achieve ARI ≈ 1.0 on well-separated synthetic data; real data will show the GNN advantage.

---

## Installation

```bash
pip install -e .
```

Requires: `torch`, `numpy`, `scikit-learn`

---

## Tests

```bash
python -m pytest tests/ -v
```

51 tests covering graph construction, dataset generation, GCN forward/backward, and evaluation metrics.

---

## Bring your own data

The toolkit works with any `np.ndarray` of shape `(n_cells, n_genes)`:

```python
import numpy as np
from scgraph_hub import SingleCellGraph, CellTypeGNN, train_gcn

# Load your expression matrix (cells × genes)
X = np.load("my_expression.npy")          # shape (N, G)
labels = np.load("my_cell_type_labels.npy")  # shape (N,), integers

graph = SingleCellGraph(X, labels=labels, k=15)
```

Compatible with AnnData objects (extract `adata.X` as numpy array).

---

## References

- Kipf, T. N., & Welling, M. (2017). [Semi-supervised classification with graph convolutional networks](https://arxiv.org/abs/1609.02907). ICLR.
- Wolf, F. A., et al. (2018). [SCANPY: large-scale single-cell gene expression data analysis](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0). Genome Biology.
