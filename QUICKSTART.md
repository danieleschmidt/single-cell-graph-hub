# Quick Start Guide

Get up and running with Single-Cell Graph Hub in 5 minutes.

## Installation

```bash
# Basic installation
pip install single-cell-graph-hub

# With visualization tools
pip install single-cell-graph-hub[viz]

# Development installation
git clone https://github.com/yourusername/single-cell-graph-hub
cd single-cell-graph-hub
pip install -e ".[dev]"
```

## 30-Second Example

```python
from scgraph_hub import SCGraphDataset, DatasetCatalog
from scgraph_hub.models import CellGraphGNN
import torch.nn.functional as F

# Browse available datasets
catalog = DatasetCatalog()
print("Available datasets:", catalog.list_datasets())

# Load a dataset
dataset = SCGraphDataset("pbmc_10k", root="./data", download=True)
print(f"Loaded {dataset.num_nodes} cells with {dataset.num_node_features} genes")

# Create and train a model
model = CellGraphGNN(
    input_dim=dataset.num_node_features,
    hidden_dim=128,
    output_dim=dataset.num_classes
)

# Quick training loop
data = dataset.data
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

print("Training complete!")
```

## Common Workflows

### 1. Cell Type Prediction

```python
# Load PBMC dataset
dataset = SCGraphDataset(
    name="pbmc_10k",
    task="cell_type_prediction",
    root="./data"
)

# Use GraphSAGE for scalability
from scgraph_hub.models import CellGraphSAGE
model = CellGraphSAGE(
    input_dim=dataset.num_node_features,
    hidden_dims=[512, 256, 128]
)

# Train model...
```

### 2. Spatial Analysis

```python
# Load spatial dataset
dataset = SCGraphDataset(
    name="spatial_heart",
    task="spatial_domain_identification",
    root="./data"
)

# Use spatial-aware model
from scgraph_hub.models import SpatialGAT
model = SpatialGAT(
    input_dim=dataset.num_node_features,
    hidden_dim=256,
    num_heads=8
)
```

### 3. Dataset Discovery

```python
catalog = DatasetCatalog()

# Find human brain datasets
brain_data = catalog.filter(
    organism="human",
    tissue="brain",
    min_cells=10000
)

# Get dataset recommendations
similar = catalog.get_recommendations("pbmc_10k")

# Search by keyword
covid_data = catalog.search("covid")
```

## Key Concepts

### Graph Structure
- **Nodes**: Individual cells with gene expression features
- **Edges**: Cell-cell relationships (similarity, spatial proximity)
- **Tasks**: Cell type prediction, trajectory inference, etc.

### Model Selection
- **CellGraphGNN**: General-purpose, good starting point
- **CellGraphSAGE**: Large datasets (>100k cells)
- **SpatialGAT**: Spatial transcriptomics data
- **HierarchicalGNN**: Multi-scale analysis

### Data Splits
- **Train/Val/Test**: Pre-computed splits for consistent evaluation
- **Node-level**: Single large graph with masked nodes
- **Cross-validation**: Multiple random splits available

## Next Steps

1. **Explore datasets**: Browse the [catalog](docs/datasets.md)
2. **Try tutorials**: Work through [examples/](examples/)
3. **Read API docs**: See [docs/api/](docs/api/)
4. **Join community**: Visit our [forum](https://forum.scgraphhub.org)

## Need Help?

- 📖 **Documentation**: [scgraphhub.readthedocs.io](https://scgraphhub.readthedocs.io)
- 💬 **Forum**: [forum.scgraphhub.org](https://forum.scgraphhub.org)  
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/single-cell-graph-hub/issues)
- 📧 **Email**: support@scgraphhub.org

## Performance Tips

- Start with smaller datasets for prototyping
- Use GPU acceleration: `model = model.cuda()`
- Enable mixed precision for large models
- Monitor memory usage with large graphs