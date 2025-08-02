# Getting Started Guide

Welcome to Single-Cell Graph Hub! This guide will help you get up and running with graph-based single-cell analysis.

## Installation

### Basic Installation

```bash
pip install single-cell-graph-hub
```

### Full Installation with All Features

```bash
pip install single-cell-graph-hub[full]
```

### Development Installation

```bash
git clone https://github.com/yourusername/single-cell-graph-hub
cd single-cell-graph-hub
pip install -e ".[dev]"
```

## Quick Start

### 1. Load Your First Dataset

```python
from scgraph_hub import SCGraphDataset

# Load a pre-processed dataset
dataset = SCGraphDataset(
    name="pbmc_10k",
    root="./data",
    download=True
)

print(f"Number of cells: {dataset.num_nodes}")
print(f"Number of edges: {dataset.num_edges}")
print(f"Features per cell: {dataset.num_node_features}")
```

### 2. Train a Simple Model

```python
from scgraph_hub.models import CellGraphGNN
import torch
import torch.nn.functional as F

# Create model
model = CellGraphGNN(
    input_dim=dataset.num_node_features,
    hidden_dim=128,
    output_dim=dataset.num_classes
)

# Get data loaders
train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
```

### 3. Evaluate Results

```python
from scgraph_hub.metrics import evaluate_model

# Evaluate on test set
results = evaluate_model(
    model, 
    test_loader, 
    metrics=['accuracy', 'f1_macro', 'f1_weighted']
)

print(f"Test Accuracy: {results['accuracy']:.3f}")
print(f"Test F1 (macro): {results['f1_macro']:.3f}")
```

## Core Concepts

### Datasets
Single-Cell Graph Hub standardizes single-cell data as graphs where:
- **Nodes**: Individual cells
- **Edges**: Relationships between cells (similarity, spatial proximity, etc.)
- **Node Features**: Gene expression, chromatin accessibility, etc.
- **Edge Features**: Distance, confidence, interaction strength

### Graph Construction
Multiple methods for building cell-cell graphs:
- **K-NN Graphs**: Based on expression similarity
- **Spatial Graphs**: For spatial transcriptomics data
- **Biological Graphs**: Using prior knowledge of cell interactions

### Models
Built-in support for various GNN architectures:
- **Graph Convolutional Networks (GCN)**
- **Graph Attention Networks (GAT)**
- **GraphSAGE**: For large-scale data
- **Custom Architectures**: Specialized for biological data

## Common Workflows

### Cell Type Prediction

```python
# Load dataset with cell type labels
dataset = SCGraphDataset("tabula_muris", task="cell_type_prediction")

# Train classifier
from scgraph_hub.workflows import CellTypeClassification
classifier = CellTypeClassification()
results = classifier.train(dataset)
```

### Trajectory Inference

```python
# Load developmental dataset
dataset = SCGraphDataset("embryo_dev", task="trajectory_inference")

# Infer trajectories
from scgraph_hub.workflows import TrajectoryInference
trajectory = TrajectoryInference()
pseudotime = trajectory.infer(dataset)
```

### Batch Correction

```python
# Load multi-batch dataset
dataset = SCGraphDataset("pancreas_integrated", task="batch_correction")

# Correct batch effects
from scgraph_hub.workflows import BatchCorrection
corrector = BatchCorrection()
corrected_data = corrector.correct(dataset)
```

## Working with Custom Data

### Converting AnnData to Graph Format

```python
import scanpy as sc
from scgraph_hub.preprocessing import AnnDataToGraph

# Load your data
adata = sc.read_h5ad("my_data.h5ad")

# Convert to graph
converter = AnnDataToGraph()
graph_data = converter.convert(
    adata,
    graph_method="knn",
    k=20,
    use_pca=True
)
```

### Building Custom Graphs

```python
from scgraph_hub.graph_construction import GraphBuilder

builder = GraphBuilder()

# K-NN graph
knn_graph = builder.build_knn_graph(
    features=adata.X,
    k=15,
    metric="euclidean"
)

# Spatial graph
spatial_graph = builder.build_spatial_graph(
    coordinates=adata.obsm["spatial"],
    radius=100
)
```

## Visualization

### Basic Plotting

```python
from scgraph_hub.visualization import plot_graph

# Plot graph with cell type colors
plot_graph(
    graph_data,
    node_colors=cell_types,
    layout="umap",
    save="cell_graph.png"
)
```

### Interactive Exploration

```python
from scgraph_hub.visualization import InteractiveExplorer

explorer = InteractiveExplorer()
explorer.plot(graph_data, color_by="cell_type")
explorer.save("interactive_plot.html")
```

## Best Practices

### Data Preprocessing
1. **Quality Control**: Filter low-quality cells and genes
2. **Normalization**: Use appropriate normalization methods
3. **Feature Selection**: Select highly variable genes
4. **Graph Construction**: Choose appropriate k for k-NN graphs

### Model Training
1. **Validation**: Always use proper train/validation/test splits
2. **Hyperparameters**: Tune learning rate, hidden dimensions, dropout
3. **Early Stopping**: Prevent overfitting
4. **Evaluation**: Use biologically meaningful metrics

### Performance Optimization
1. **Batch Size**: Adjust based on memory constraints
2. **Graph Sampling**: For very large graphs
3. **GPU Usage**: Enable CUDA acceleration
4. **Memory Management**: Use data loaders efficiently

## Troubleshooting

### Common Issues

**Out of Memory Errors**
```python
# Reduce batch size
train_loader = dataset.get_loader(batch_size=16)

# Use graph sampling
from torch_geometric.loader import NeighborSampler
loader = NeighborSampler(dataset, sizes=[10, 10])
```

**Poor Model Performance**
```python
# Check data quality
dataset.validate()

# Visualize graph structure
plot_graph_statistics(dataset)

# Try different architectures
model = SpatialGAT(...)  # For spatial data
model = HierarchicalGNN(...)  # For multi-scale data
```

**Long Training Times**
```python
# Enable GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

## Next Steps

1. **Explore Datasets**: Browse the full catalog of available datasets
2. **Try Different Models**: Experiment with various GNN architectures
3. **Custom Analysis**: Implement your own graph construction methods
4. **Contribute**: Add new datasets or models to the community

## Getting Help

- **Documentation**: [Full documentation](https://scgraphhub.readthedocs.io)
- **GitHub Issues**: Report bugs or request features
- **Community Forum**: Ask questions and share experiences
- **Tutorials**: Step-by-step guides for specific tasks

Happy graph analyzing! 🧬📊