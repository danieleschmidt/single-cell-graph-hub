# API Reference

This document provides comprehensive API documentation for Single-Cell Graph Hub.

## Core Classes

### SCGraphDataset
Main dataset class for loading single-cell data as graphs.

```python
from scgraph_hub import SCGraphDataset

dataset = SCGraphDataset(
    name="pbmc_10k",          # Dataset name
    root="./data",            # Data directory
    task="cell_type_prediction",  # Analysis task
    download=True             # Auto-download if needed
)
```

**Properties:**
- `num_nodes`: Number of cells in the dataset
- `num_edges`: Number of cell-cell connections
- `num_node_features`: Number of gene features
- `num_classes`: Number of cell type classes

**Methods:**
- `get_loaders(batch_size=32)`: Get PyTorch Geometric data loaders
- `info()`: Get dataset metadata dictionary

### DatasetCatalog
Browse and discover available datasets.

```python
from scgraph_hub import DatasetCatalog

catalog = DatasetCatalog()
datasets = catalog.list_datasets()
info = catalog.get_info("pbmc_10k")
```

**Methods:**
- `list_datasets()`: List all available dataset names
- `get_info(name)`: Get detailed metadata for a dataset
- `filter(**criteria)`: Filter datasets by characteristics
- `search(query)`: Search datasets by text
- `get_recommendations(reference)`: Get similar datasets

### GNN Models

#### CellGraphGNN
Basic graph neural network for cell analysis.

```python
from scgraph_hub.models import CellGraphGNN

model = CellGraphGNN(
    input_dim=2000,    # Number of genes
    hidden_dim=128,    # Hidden layer size
    output_dim=8,      # Number of cell types
    num_layers=3,      # GNN depth
    dropout=0.2        # Regularization
)
```

#### CellGraphSAGE
Scalable GraphSAGE for large datasets.

```python
from scgraph_hub.models import CellGraphSAGE

model = CellGraphSAGE(
    input_dim=2000,
    hidden_dims=[512, 256, 128],
    aggregator="mean"
)
```

#### SpatialGAT
Graph attention network for spatial data.

```python
from scgraph_hub.models import SpatialGAT

model = SpatialGAT(
    input_dim=2000,
    hidden_dim=256,
    num_heads=8,
    spatial_dim=2
)
```

## Data Format

### Graph Structure
Each dataset consists of:
- **Nodes**: Individual cells with gene expression features
- **Edges**: Cell-cell relationships (similarity, spatial proximity, etc.)
- **Labels**: Cell types, disease states, or other annotations

### Input Data
- **Node features (x)**: Gene expression matrix [n_cells, n_genes]
- **Edge indices**: Connection list [2, n_edges]
- **Labels (y)**: Target annotations [n_cells]
- **Masks**: Train/validation/test splits

### Example Data Access
```python
dataset = SCGraphDataset("pbmc_10k", root="./data")
data = dataset.data

print(f"Node features: {data.x.shape}")      # [10000, 2000]
print(f"Edge indices: {data.edge_index.shape}")  # [2, num_edges]
print(f"Labels: {data.y.shape}")             # [10000]
print(f"Train mask: {data.train_mask.sum()}")    # 6000 cells
```

## Training Workflow

### Basic Training Loop
```python
import torch
import torch.nn.functional as F

# Load data and model
dataset = SCGraphDataset("pbmc_10k", root="./data")
model = CellGraphGNN(dataset.num_node_features, 128, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
data = dataset.data
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    # Validation
    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[data.val_mask]).float().mean()
            print(f'Epoch {epoch}: Val Acc: {val_acc:.4f}')
```

### Model Evaluation
```python
# Test evaluation
model.eval()
with torch.no_grad():
    test_out = model(data.x, data.edge_index)
    test_pred = test_out[data.test_mask].argmax(dim=1)
    test_acc = (test_pred == data.y[data.test_mask]).float().mean()
    
    print(f'Test Accuracy: {test_acc:.4f}')
```

## Available Datasets

### Dataset Categories

**By Modality:**
- `scRNA-seq`: Single-cell RNA sequencing
- `snRNA-seq`: Single-nucleus RNA sequencing  
- `scATAC-seq`: Single-cell ATAC sequencing
- `spatial_transcriptomics`: Spatial gene expression
- `multimodal`: Combined modalities

**By Organism:**
- `human`: Human datasets
- `mouse`: Mouse datasets
- `zebrafish`: Zebrafish datasets

**By Tissue:**
- `blood`: Immune/blood cells (PBMC, etc.)
- `brain`: Neural tissues
- `heart`: Cardiac tissues
- `lung`: Pulmonary tissues
- `multi-organ`: Cross-tissue atlases

### Example Datasets

| Name | Cells | Genes | Modality | Organism | Tissue |
|------|-------|-------|----------|----------|---------|
| pbmc_10k | 10,000 | 2,000 | scRNA-seq | human | blood |
| tabula_muris | 100,000 | 23,000 | scRNA-seq | mouse | multi-organ |
| brain_atlas | 75,000 | 3,000 | snRNA-seq | human | brain |
| spatial_heart | 15,000 | 1,800 | spatial | human | heart |

### Filtering Examples
```python
catalog = DatasetCatalog()

# Large human datasets
large_human = catalog.filter(
    organism="human",
    min_cells=50000
)

# Spatial datasets
spatial = catalog.filter(
    has_spatial=True
)

# RNA sequencing data
rna_data = catalog.filter(
    modality="scRNA-seq",
    min_genes=1000
)
```

## Graph Construction

### Similarity Graphs
Cell-cell similarity based on gene expression:
- K-nearest neighbors (k-NN)
- Radius-based connections
- Cosine similarity thresholding

### Spatial Graphs
For spatial transcriptomics data:
- Spatial proximity (radius-based)
- Delaunay triangulation
- k-nearest neighbors in space

### Biological Graphs
Domain-specific connections:
- Developmental trajectories
- Cell-cell communication
- Hierarchical relationships

## Error Handling

### Common Issues

**Import Errors:**
```python
try:
    from scgraph_hub import SCGraphDataset
except ImportError:
    print("Install with: pip install single-cell-graph-hub")
```

**Dataset Not Found:**
```python
try:
    dataset = SCGraphDataset("unknown_dataset")
except KeyError as e:
    print(f"Dataset error: {e}")
    catalog = DatasetCatalog()
    print(f"Available: {catalog.list_datasets()}")
```

**GPU/CPU Compatibility:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

## Performance Tips

### Memory Optimization
```python
# For large datasets, use smaller batch sizes
train_loader, _, _ = dataset.get_loaders(batch_size=16)

# Enable gradient checkpointing for memory efficiency
model = torch.compile(model)  # PyTorch 2.0+
```

### Training Acceleration
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[mask], target[mask])

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Model Selection
- **Small datasets (<10k cells)**: CellGraphGNN
- **Large datasets (>100k cells)**: CellGraphSAGE  
- **Spatial data**: SpatialGAT
- **Multi-scale analysis**: HierarchicalGNN