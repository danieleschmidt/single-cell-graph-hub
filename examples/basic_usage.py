"""Basic usage example for Single-Cell Graph Hub.

This example demonstrates the core functionality matching the README documentation.
"""

import torch
import torch.nn.functional as F
from scgraph_hub import SCGraphDataset, DatasetCatalog
from scgraph_hub.models import CellGraphGNN


def main():
    """Run the basic usage example."""
    print("Single-Cell Graph Hub - Basic Usage Example")
    print("=" * 50)
    
    # 1. Browse available datasets
    print("\n1. Browsing Dataset Catalog")
    catalog = DatasetCatalog()
    
    # List all available datasets
    all_datasets = catalog.list_datasets()
    print(f"Available datasets: {all_datasets}")
    
    # Filter by characteristics
    rna_datasets = catalog.filter(
        modality="scRNA-seq",
        organism="human",
        min_cells=10000,
        has_spatial=False
    )
    print(f"Human scRNA-seq datasets (>10k cells): {rna_datasets}")
    
    # Get dataset info
    if "pbmc_10k" in all_datasets:
        info = catalog.get_info("pbmc_10k")
        print(f"\nDataset info for pbmc_10k:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # 2. Load a dataset
    print("\n2. Loading Dataset")
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
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 3. Initialize and train a GNN model
    print("\n3. Training GNN Model")
    
    # Initialize GNN model
    model = CellGraphGNN(
        input_dim=dataset.num_node_features,
        hidden_dim=128,
        output_dim=dataset.num_classes,
        num_layers=3,
        dropout=0.2
    )
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Get the graph data
    data = dataset.data
    
    # Simple training loop (just a few epochs for demonstration)
    print("\nTraining...")
    model.train()
    
    for epoch in range(5):  # Just 5 epochs for demo
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Compute loss on training nodes only
        train_mask = data.train_mask
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation accuracy
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[data.val_mask]).float().mean()
        model.train()
        
        print(f'Epoch {epoch + 1}: Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # 4. Final evaluation
    print("\n4. Final Evaluation")
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_pred = test_out[data.test_mask].argmax(dim=1)
        test_acc = (test_pred == data.y[data.test_mask]).float().mean()
        
        print(f'Test Accuracy: {test_acc:.4f}')
    
    # 5. Get embeddings
    print("\n5. Extracting Embeddings")
    with torch.no_grad():
        # Get intermediate representations (before final layer)
        embeddings = model.input_proj(data.x)
        for i, conv in enumerate(model.convs[:-1]):  # All but last layer
            embeddings = conv(embeddings, data.edge_index)
            embeddings = F.relu(embeddings)
        
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding statistics:")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Min: {embeddings.min():.4f}")
        print(f"  Max: {embeddings.max():.4f}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    
    # 6. Dataset summary
    print("\n6. Dataset Summary")
    summary = catalog.get_summary_stats()
    print("Catalog Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 7. Search and recommendations
    print("\n7. Search and Recommendations")
    search_results = catalog.search("human")
    print(f"Datasets matching 'human': {search_results}")
    
    if "pbmc_10k" in all_datasets:
        recommendations = catalog.get_recommendations("pbmc_10k", max_results=3)
        print(f"Datasets similar to pbmc_10k: {recommendations}")


if __name__ == "__main__":
    main()