# Single-Cell Graph Hub

Curates >200 single-cell omics datasets pre-converted into HDF5 + PyG graph objects. Provides out-of-the-box loaders, standardized splits, and leaderboard scripts for Graph Neural Network (GNN) baselines in single-cell analysis.

## Overview

Single-Cell Graph Hub standardizes single-cell omics data as graph-structured datasets, where cells are nodes and their relationships (spatial proximity, developmental trajectories, or molecular similarity) form edges. This enables the single-cell community to leverage state-of-the-art graph neural networks for biological discovery.

## Key Features

- **200+ Curated Datasets**: scRNA-seq, scATAC-seq, spatial transcriptomics, multi-omics
- **Graph-Ready Format**: Pre-computed cell graphs with biological edges
- **Standardized Benchmarks**: Consistent evaluation across methods
- **PyTorch Geometric**: Native PyG dataset format
- **Biological Annotations**: Cell types, tissues, disease states included
- **Leaderboard System**: Track GNN performance on biological tasks

## Installation

```bash
# Basic installation
pip install single-cell-graph-hub

# With visualization tools
pip install single-cell-graph-hub[viz]

# With all preprocessing tools
pip install single-cell-graph-hub[full]

# Development installation
git clone https://github.com/danieleschmidt/single-cell-graph-hub
cd single-cell-graph-hub
pip install -e ".[dev]"
```

## Quick Start

### Loading a Dataset

```python
from scgraph_hub import SCGraphDataset

# Load a single-cell dataset as graph
dataset = SCGraphDataset(
    name="pbmc_10k",
    root="./data",
    task="cell_type_prediction",
    download=True
)

# Dataset information
print(f"Number of cells: {dataset.num_nodes}")
print(f"Number of edges: {dataset.num_edges}")
print(f"Node features: {dataset.num_node_features}")
print(f"Number of classes: {dataset.num_classes}")

# Get train/val/test splits
train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)
```

### Training a GNN

```python
from scgraph_hub.models import CellGraphGNN
import torch.nn.functional as F

# Initialize GNN model
model = CellGraphGNN(
    input_dim=dataset.num_node_features,
    hidden_dim=128,
    output_dim=dataset.num_classes,
    num_layers=3,
    dropout=0.2
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    correct = 0
    for batch in val_loader:
        pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
        correct += (pred == batch.y).sum().item()
    
    acc = correct / len(val_loader.dataset)
    print(f'Epoch {epoch}: Val Acc: {acc:.4f}')
```

## Available Datasets

### Browse Dataset Catalog

```python
from scgraph_hub import DatasetCatalog

catalog = DatasetCatalog()

# List all available datasets
all_datasets = catalog.list_datasets()

# Filter by characteristics
rna_datasets = catalog.filter(
    modality="scRNA-seq",
    organism="human",
    min_cells=10000,
    has_spatial=False
)

# Get dataset info
info = catalog.get_info("tabula_muris")
print(f"Cells: {info['n_cells']}")
print(f"Genes: {info['n_genes']}")
print(f"Tissues: {info['tissues']}")
print(f"Cell types: {info['n_cell_types']}")
print(f"Graph construction: {info['graph_method']}")
```

### Dataset Categories

```python
# By modality
modalities = {
    'transcriptomics': ['scRNA-seq', 'snRNA-seq', 'spatial_transcriptomics'],
    'epigenomics': ['scATAC-seq', 'scChIP-seq', 'scHi-C'],
    'proteomics': ['CyTOF', 'CITE-seq'],
    'multimodal': ['SHARE-seq', 'scNMT-seq', '10X_multiome']
}

# By organism
organisms = ['human', 'mouse', 'zebrafish', 'drosophila', 'c_elegans']

# By tissue/system
systems = [
    'immune', 'brain', 'developmental', 'cancer',
    'heart', 'lung', 'liver', 'kidney', 'pancreas'
]

# By task type
tasks = [
    'cell_type_prediction',
    'trajectory_inference', 
    'gene_imputation',
    'batch_correction',
    'cell_cell_interaction',
    'spatial_domain_identification'
]
```

## Graph Construction Methods

### Cell-Cell Similarity Graphs

```python
from scgraph_hub.graph_construction import SimilarityGraphBuilder

# Build k-NN graph based on gene expression
builder = SimilarityGraphBuilder(
    method='knn',
    k=15,
    metric='euclidean',
    use_pca=True,
    n_components=50
)

# Convert AnnData to PyG Data
import scanpy as sc
adata = sc.read_h5ad('my_data.h5ad')

graph_data = builder.build_graph(
    adata,
    cell_features='X_pca',
    edge_weight='distance'
)

# Add biological edges
graph_data = builder.add_biological_edges(
    graph_data,
    edge_types=['spatial_proximity', 'developmental_similarity'],
    adata=adata
)
```

### Spatial Graphs

```python
from scgraph_hub.graph_construction import SpatialGraphBuilder

# Build spatial proximity graph
spatial_builder = SpatialGraphBuilder(
    method='radius',
    radius=100,  # micrometers
    min_neighbors=3,
    max_neighbors=30
)

# From spatial coordinates
spatial_graph = spatial_builder.build_from_coordinates(
    coordinates=adata.obsm['spatial'],
    features=adata.X,
    max_distance=150
)

# Delaunay triangulation
delaunay_graph = spatial_builder.build_delaunay(
    coordinates=adata.obsm['spatial'],
    prune_long_edges=True,
    max_edge_length=200
)
```

### Hierarchical Graphs

```python
from scgraph_hub.graph_construction import HierarchicalGraphBuilder

# Build multi-level cell graphs
hier_builder = HierarchicalGraphBuilder()

# Create hierarchy: cells -> cell types -> tissues
hierarchical_graph = hier_builder.build_hierarchy(
    adata,
    levels=['cell', 'cell_type', 'tissue'],
    aggregation='mean'
)

# Add cross-level edges
hierarchical_graph = hier_builder.add_cross_level_edges(
    hierarchical_graph,
    connection_type='belongs_to'
)
```

## Benchmark Tasks

### Cell Type Prediction

```python
from scgraph_hub.benchmarks import CellTypeBenchmark

benchmark = CellTypeBenchmark()

# Run standard evaluation
results = benchmark.evaluate(
    model=your_gnn_model,
    datasets=['pbmc_10k', 'tabula_muris', 'brain_atlas'],
    metrics=['accuracy', 'f1_macro', 'f1_weighted'],
    n_runs=5
)

# Compare with baselines
benchmark.compare_methods({
    'GCN': gcn_model,
    'GAT': gat_model,
    'GraphSAGE': graphsage_model,
    'Random Forest': rf_baseline,
    'SVM': svm_baseline
})

# Generate report
benchmark.generate_report(
    results,
    save_path='benchmark_results.html'
)
```

### Trajectory Inference

```python
from scgraph_hub.benchmarks import TrajectoryBenchmark

traj_benchmark = TrajectoryBenchmark()

# Evaluate trajectory reconstruction
traj_results = traj_benchmark.evaluate(
    model=your_trajectory_model,
    datasets=['embryo_development', 'hematopoiesis', 'reprogramming'],
    metrics=['kendall_tau', 'branch_accuracy', 'ti_error']
)

# Visualize inferred trajectories
traj_benchmark.visualize_trajectories(
    model_predictions=traj_results,
    ground_truth=true_trajectories,
    save_dir='trajectory_plots/'
)
```

### Batch Effect Correction

```python
from scgraph_hub.benchmarks import BatchCorrectionBenchmark

batch_benchmark = BatchCorrectionBenchmark()

# Evaluate batch correction
batch_results = batch_benchmark.evaluate(
    model=your_batch_model,
    datasets=['pancreas_integrated', 'immune_covid', 'atlas_merged'],
    metrics=['silhouette_batch', 'silhouette_celltype', 'ari', 'nmi']
)

# Plot integration quality
batch_benchmark.plot_integration(
    corrected_data=model_output,
    batch_key='batch',
    celltype_key='cell_type',
    method='umap'
)
```

## Pre-trained Models

### Model Zoo

```python
from scgraph_hub.models import ModelZoo

zoo = ModelZoo()

# List available pre-trained models
available_models = zoo.list_models()

# Load pre-trained model
pretrained = zoo.load_model(
    'celltype_gat_v2',
    device='cuda'
)

# Fine-tune on new data
finetuned = zoo.finetune(
    pretrained,
    new_dataset=my_dataset,
    epochs=50,
    lr=0.001
)

# Model card information
model_info = zoo.get_model_info('celltype_gat_v2')
print(f"Trained on: {model_info['datasets']}")
print(f"Performance: {model_info['metrics']}")
print(f"Parameters: {model_info['num_parameters']}")
```

### Transfer Learning

```python
from scgraph_hub.transfer import TransferLearning

transfer = TransferLearning()

# Pre-train on large atlas
pretrained = transfer.pretrain(
    model=CellGraphGNN(),
    dataset='human_cell_atlas',
    objective='masked_gene_prediction',
    epochs=100
)

# Transfer to specific tissue
transferred = transfer.finetune(
    pretrained_model=pretrained,
    target_dataset='lung_disease',
    freeze_layers=2,
    epochs=50
)

# Zero-shot transfer
predictions = transfer.zero_shot_predict(
    model=pretrained,
    new_data=unseen_tissue_data,
    adapt_method='prototypical'
)
```

## Graph Neural Network Models

### Specialized Architectures

```python
from scgraph_hub.models import (
    CellGraphSAGE,
    SpatialGAT,
    HierarchicalGNN,
    CellGraphTransformer
)

# GraphSAGE for large-scale data
sage_model = CellGraphSAGE(
    input_dim=3000,
    hidden_dims=[512, 256, 128],
    aggregator='mean',
    dropout=0.3,
    batch_norm=True
)

# Spatial-aware GAT
spatial_gat = SpatialGAT(
    input_dim=2000,
    hidden_dim=256,
    num_heads=8,
    spatial_dim=2,
    use_edge_attr=True
)

# Hierarchical GNN for multi-scale
hier_gnn = HierarchicalGNN(
    level_dims={'cell': 2000, 'type': 100, 'tissue': 20},
    hidden_dim=256,
    pooling='diffpool'
)

# Graph Transformer
graph_transformer = CellGraphTransformer(
    input_dim=3000,
    model_dim=512,
    num_heads=8,
    num_layers=6,
    positional_encoding='laplacian'
)
```

### Custom Model Development

```python
from scgraph_hub.models import BaseGNN
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class CustomCellGNN(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Custom message passing layer
        self.conv1 = BiologicalMessagePassing(input_dim, hidden_dim)
        self.conv2 = BiologicalMessagePassing(hidden_dim, hidden_dim)
        self.conv3 = BiologicalMessagePassing(hidden_dim, output_dim)
        
        # Biological attention
        self.bio_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # First layer with biological constraints
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Attention over cell states
        x, _ = self.bio_attention(x, x, x)
        
        # Final layers
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        
        return x

# Register custom model
zoo.register_model('custom_cell_gnn', CustomCellGNN)
```

## Data Processing Pipeline

### Preprocessing

```python
from scgraph_hub.preprocessing import DataProcessor

processor = DataProcessor()

# Standard single-cell preprocessing
processed_data = processor.preprocess(
    adata,
    steps=[
        'filter_cells',      # min_genes=200
        'filter_genes',      # min_cells=3
        'normalize_total',   # target_sum=1e4
        'log1p',
        'highly_variable_genes',  # n_top_genes=2000
        'scale'             # zero mean, unit variance
    ]
)

# Graph-specific preprocessing
graph_ready = processor.prepare_for_graph(
    processed_data,
    compute_pca=True,
    n_comps=50,
    compute_neighbors=True,
    n_neighbors=30
)
```

### Data Augmentation

```python
from scgraph_hub.augmentation import GraphAugmentation

augmenter = GraphAugmentation()

# Cell-level augmentation
augmented = augmenter.augment_cells(
    graph_data,
    methods=[
        'expression_noise',     # Add Gaussian noise
        'dropout_genes',        # Random gene dropout
        'mixup_cells',         # Mix cell profiles
        'edge_perturbation'    # Add/remove edges
    ],
    augment_prob=0.5
)

# Graph-level augmentation
graph_aug = augmenter.augment_graph(
    graph_data,
    methods=[
        'subgraph_sampling',
        'edge_dropping',
        'feature_masking',
        'graph_crop'
    ]
)
```

## Evaluation Metrics

### Biological Metrics

```python
from scgraph_hub.metrics import BiologicalMetrics

bio_metrics = BiologicalMetrics()

# Cell type purity
purity = bio_metrics.cell_type_purity(
    embeddings=model_embeddings,
    cell_types=true_cell_types,
    method='silhouette'
)

# Biological conservation
conservation = bio_metrics.biological_conservation(
    original_data=adata,
    embedded_data=embeddings,
    gene_sets='hallmark'  # or custom gene sets
)

# Trajectory preservation
traj_score = bio_metrics.trajectory_conservation(
    original_trajectory=true_trajectory,
    embedded_trajectory=predicted_trajectory,
    metric='kendall_tau'
)

# Batch mixing
batch_mixing = bio_metrics.batch_mixing_score(
    embeddings=embeddings,
    batch_labels=batch_labels,
    cell_type_labels=cell_types
)
```

### Graph-Specific Metrics

```python
from scgraph_hub.metrics import GraphMetrics

graph_metrics = GraphMetrics()

# Graph reconstruction
reconstruction_score = graph_metrics.edge_prediction_auc(
    true_edges=original_graph.edge_index,
    predicted_edges=model.predict_edges(),
    negative_sampling_ratio=1.0
)

# Community detection
modularity = graph_metrics.modularity_score(
    graph=cell_graph,
    communities=predicted_cell_types
)

# Graph properties preservation
properties = graph_metrics.graph_properties_preservation(
    original_graph=original,
    embedded_graph=embedded,
    properties=['degree_dist', 'clustering_coef', 'path_length']
)
```

## Visualization

### Interactive Visualization

```python
from scgraph_hub.visualization import InteractiveVisualizer

viz = InteractiveVisualizer()

# Interactive cell graph explorer
explorer = viz.create_graph_explorer(
    graph_data=dataset[0],
    node_colors=cell_types,
    node_sizes=gene_expression['CD3E'],
    edge_colors=edge_weights
)

explorer.add_tooltip([
    'cell_type',
    'n_genes',
    'total_counts',
    'batch'
])

explorer.save('interactive_cell_graph.html')

# 3D embedding with graph overlay
viz.plot_3d_graph(
    embeddings=umap_3d,
    edges=graph_data.edge_index,
    colors=cell_types,
    size=2,
    edge_alpha=0.1,
    save_path='3d_cell_graph.html'
)
```

### Publication Figures

```python
from scgraph_hub.visualization import PublicationFigures

pub_viz = PublicationFigures(style='nature')

# Multi-panel figure
fig = pub_viz.create_figure(rows=2, cols=3, figsize=(15, 10))

# Panel A: Dataset overview
pub_viz.plot_dataset_summary(
    fig.axes[0],
    dataset_stats,
    plot_type='violin'
)

# Panel B: Graph statistics
pub_viz.plot_graph_statistics(
    fig.axes[1],
    graph_properties,
    metrics=['degree', 'clustering', 'components']
)

# Panel C: Model comparison
pub_viz.plot_model_comparison(
    fig.axes[2],
    benchmark_results,
    models=['GCN', 'GAT', 'GraphSAGE'],
    metric='f1_score'
)

# Save publication-ready figure
pub_viz.save_figure(fig, 'figure_2.pdf', dpi=300)
```

## Integration with Single-Cell Tools

### Scanpy Integration

```python
from scgraph_hub.integrations import ScanpyIntegration

sc_integration = ScanpyIntegration()

# Convert graph embeddings back to AnnData
adata_with_graph = sc_integration.add_graph_embeddings(
    adata,
    embeddings=model.get_embeddings(),
    key='X_graph'
)

# Use graph embeddings for standard analysis
sc.pp.neighbors(adata_with_graph, use_rep='X_graph')
sc.tl.umap(adata_with_graph)
sc.tl.leiden(adata_with_graph)

# Export for Cellxgene
sc_integration.export_for_cellxgene(
    adata_with_graph,
    graph_data=graph,
    save_path='graph_enhanced_data.h5ad'
)
```

### Seurat Integration

```python
from scgraph_hub.integrations import SeuratIntegration

seurat_int = SeuratIntegration()

# Export for Seurat
seurat_int.export_to_seurat(
    graph_data=dataset[0],
    metadata=cell_metadata,
    save_path='graph_data_for_seurat.rds'
)

# R script generation
seurat_int.generate_r_script(
    template='graph_analysis',
    output='analyze_in_seurat.R'
)
```

## Contributing

### Adding New Datasets

```python
from scgraph_hub.contrib import DatasetContributor

contributor = DatasetContributor()

# Validate your dataset
validation = contributor.validate_dataset(
    h5ad_file='my_new_dataset.h5ad',
    metadata={
        'name': 'kidney_disease_atlas',
        'organism': 'human',
        'n_cells': 50000,
        'modality': 'scRNA-seq',
        'disease': 'chronic_kidney_disease'
    }
)

# Convert to graph format
graph_dataset = contributor.create_graph_dataset(
    h5ad_file='my_new_dataset.h5ad',
    graph_construction_method='knn',
    k=20,
    tasks=['cell_type_prediction', 'disease_classification']
)

# Submit to hub
contributor.submit_dataset(
    dataset=graph_dataset,
    description='Comprehensive kidney disease atlas with ...',
    citation='Smith et al., Nature 2025'
)
```

### Adding New Models

```python
from scgraph_hub.contrib import ModelContributor

model_contributor = ModelContributor()

# Validate model
validation = model_contributor.validate_model(
    model=your_model,
    test_datasets=['pbmc_10k', 'tabula_muris'],
    required_metrics=['accuracy', 'f1_score']
)

# Submit model
model_contributor.submit_model(
    model=your_model,
    model_card={
        'name': 'BioCellGNN',
        'description': 'Biologically-informed cell GNN',
        'paper': 'arxiv:2025.xxxxx',
        'performance': validation.results
    }
)
```

## Configuration

### Hub Configuration

```yaml
# config/hub_config.yaml
storage:
  backend: "s3"  # or "gcs", "local"
  bucket: "single-cell-graph-hub"
  cache_dir: "~/.scgraph_hub"
  
datasets:
  formats: ["h5ad", "h5", "zarr"]
  compression: "gzip"
  chunk_size: 1000
  
graphs:
  max_edges_per_node: 100
  edge_attributes: ["weight", "distance", "confidence"]
  node_features: ["expression", "chromatin", "protein"]
  
api:
  endpoint: "https://api.scgraphhub.org"
  version: "v1"
  rate_limit: 1000  # requests per hour
```

## API Access

### Python API

```python
from scgraph_hub.api import SCGraphAPI

api = SCGraphAPI(token="your_api_token")

# Search datasets
results = api.search_datasets(
    organism="mouse",
    min_cells=5000,
    has_disease=True,
    modality="scRNA-seq"
)

# Download specific dataset
api.download_dataset(
    name="mouse_brain_atlas",
    version="latest",
    save_path="./data"
)

# Submit benchmark results
api.submit_results(
    model_name="YourGNN",
    dataset="immune_atlas",
    metrics={'accuracy': 0.95, 'f1': 0.93},
    config=your_config
)
```

### REST API

```bash
# List available datasets
curl https://api.scgraphhub.org/v1/datasets

# Get dataset metadata
curl https://api.scgraphhub.org/v1/datasets/pbmc_10k/metadata

# Download dataset
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://api.scgraphhub.org/v1/datasets/pbmc_10k/download \
     -o pbmc_10k.tar.gz

# Submit results
curl -X POST https://api.scgraphhub.org/v1/results \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model": "GCN", "dataset": "pbmc_10k", "accuracy": 0.92}'
```

## Citation

```bibtex
@article{single_cell_graph_hub,
  title={Single-Cell Graph Hub: A Unified Resource for Graph Neural Networks in Single-Cell Omics},
  author={Daniel Schmidt},
  journal={Nature Methods},
  year={2025},
  doi={10.1038/s41592-025-xxxxx}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Single-cell genomics community for datasets
- PyTorch Geometric team for graph framework
- All data contributors and curators

## Resources

- [Documentation](https://scgraphhub.readthedocs.io)
- [Tutorials](https://github.com/danieleschmidt/single-cell-graph-hub/tutorials)
- [Leaderboard](https://scgraphhub.org/leaderboard)
- [Discussion Forum](https://forum.scgraphhub.org)
