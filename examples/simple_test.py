"""Simple test for basic Single-Cell Graph Hub functionality."""

import torch
import numpy as np
from torch_geometric.data import Data

def test_basic_imports():
    """Test that basic imports work."""
    print("Testing basic imports...")
    
    try:
        from scgraph_hub import DatasetCatalog
        print("✓ DatasetCatalog import successful")
        
        from scgraph_hub.models import BaseGNN, CellGraphGNN
        print("✓ Model imports successful")
        
        from scgraph_hub.preprocessing import PreprocessingPipeline, GraphConstructor
        print("✓ Preprocessing imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_catalog():
    """Test dataset catalog functionality."""
    print("\nTesting dataset catalog...")
    
    try:
        from scgraph_hub import DatasetCatalog
        
        catalog = DatasetCatalog()
        datasets = catalog.list_datasets()
        print(f"✓ Found {len(datasets)} datasets in catalog")
        
        if "pbmc_10k" in datasets:
            info = catalog.get_info("pbmc_10k")
            print(f"✓ Dataset info: {info['n_cells']} cells, {info['n_genes']} genes")
        
        return True
    except Exception as e:
        print(f"✗ Catalog error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from scgraph_hub.models import CellGraphGNN
        
        # Create a simple model
        model = CellGraphGNN(
            input_dim=100,
            hidden_dim=64,
            output_dim=5,
            num_layers=2
        )
        
        print(f"✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass with mock data
        x = torch.randn(50, 100)
        edge_index = torch.randint(0, 50, (2, 200))
        
        output = model(x, edge_index)
        print(f"✓ Model forward pass: input {x.shape} -> output {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_graph_construction():
    """Test graph construction."""
    print("\nTesting graph construction...")
    
    try:
        from scgraph_hub.preprocessing import GraphConstructor
        import scanpy as sc
        import pandas as pd
        
        # Create mock AnnData
        n_cells, n_genes = 100, 50
        X = np.random.randn(n_cells, n_genes)
        
        adata = sc.AnnData(
            X=X,
            obs=pd.DataFrame({'cell_type': np.random.choice(['A', 'B', 'C'], n_cells)})
        )
        
        # Add PCA representation
        sc.tl.pca(adata, n_comps=10)
        
        # Build graph
        graph_constructor = GraphConstructor(method='knn', k=10)
        edge_index, edge_weights = graph_constructor.build_graph(adata)
        
        print(f"✓ Built k-NN graph: {edge_index.shape[1]} edges")
        print(f"✓ Edge weights shape: {edge_weights.shape if edge_weights is not None else 'None'}")
        
        return True
    except Exception as e:
        print(f"✗ Graph construction error: {e}")
        return False

def test_mock_dataset():
    """Test mock dataset creation."""
    print("\nTesting mock dataset creation...")
    
    try:
        # Create synthetic graph data
        n_cells = 1000
        n_features = 100
        n_classes = 5
        
        # Generate node features (gene expression)
        x = torch.randn(n_cells, n_features)
        
        # Generate k-NN graph structure
        from sklearn.neighbors import NearestNeighbors
        k = 15
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(x.numpy())
        distances, indices = nbrs.kneighbors(x.numpy())
        
        edge_list = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip self
                neighbor_idx = indices[i][j]
                edge_list.extend([(i, neighbor_idx), (neighbor_idx, i)])  # Undirected
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Generate labels
        y = torch.randint(0, n_classes, (n_cells,))
        
        # Create train/val/test masks
        n_train = n_cells // 2
        n_val = n_cells // 4
        
        train_mask = torch.zeros(n_cells, dtype=torch.bool)
        val_mask = torch.zeros(n_cells, dtype=torch.bool)
        test_mask = torch.zeros(n_cells, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train+n_val] = True
        test_mask[n_train+n_val:] = True
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        print(f"✓ Created mock dataset:")
        print(f"  - Nodes: {data.num_nodes}")
        print(f"  - Edges: {data.num_edges}")
        print(f"  - Features: {data.num_node_features}")
        print(f"  - Classes: {n_classes}")
        print(f"  - Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
        
        return data
    except Exception as e:
        print(f"✗ Mock dataset error: {e}")
        return None

def test_training_loop(data):
    """Test a simple training loop."""
    print("\nTesting training loop...")
    
    try:
        from scgraph_hub.models import CellGraphGNN
        import torch.nn.functional as F
        
        # Create model
        model = CellGraphGNN(
            input_dim=data.num_node_features,
            hidden_dim=64,
            output_dim=5,  # 5 classes
            num_layers=2,
            dropout=0.2
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_pred = val_out[data.val_mask].argmax(dim=1)
                val_acc = (val_pred == data.y[data.val_mask]).float().mean()
            model.train()
            
            print(f"  Epoch {epoch+1}: Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
        
        print("✓ Training loop completed successfully")
        return True
    except Exception as e:
        print(f"✗ Training error: {e}")
        return False

def main():
    """Run all tests."""
    print("Single-Cell Graph Hub - Simple Test")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    tests = [
        test_basic_imports,
        test_catalog,
        test_model_creation,
        test_graph_construction,
    ]
    
    for test in tests:
        total_tests += 1
        if test():
            tests_passed += 1
    
    # Test dataset and training
    total_tests += 2
    data = test_mock_dataset()
    if data is not None:
        tests_passed += 1
        if test_training_loop(data):
            tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Basic functionality is working.")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()