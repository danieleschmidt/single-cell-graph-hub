#!/usr/bin/env python3
"""
Basic usage example for Single-Cell Graph Hub - Simplified Version

This example demonstrates basic functionality without requiring heavy dependencies
like PyTorch or scikit-learn. Perfect for getting started and understanding
the data structures.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import scgraph_hub


def main():
    """Run basic example with simplified dataset."""
    print("="*60)
    print("Single-Cell Graph Hub - Basic Example (Simplified)")
    print("="*60)
    
    # 1. Show available datasets
    print("\n1. Exploring Dataset Catalog")
    print("-" * 30)
    catalog = scgraph_hub.get_default_catalog()
    
    all_datasets = catalog.list_datasets()
    print(f"Total available datasets: {len(all_datasets)}")
    print(f"Dataset names: {all_datasets[:5]}...")  # Show first 5
    
    # 2. Get dataset information
    print("\n2. Dataset Information")
    print("-" * 30)
    dataset_name = "pbmc_10k"
    info = catalog.get_info(dataset_name)
    print(f"Dataset: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Number of cells: {info['n_cells']:,}")
    print(f"Number of genes: {info['n_genes']:,}")
    print(f"Organism: {info['organism']}")
    print(f"Tissue: {info['tissue']}")
    print(f"Modality: {info['modality']}")
    
    # 3. Filter datasets
    print("\n3. Dataset Filtering")
    print("-" * 30)
    human_datasets = catalog.filter(organism="human", min_cells=5000)
    mouse_datasets = catalog.filter(organism="mouse")
    spatial_datasets = catalog.filter(has_spatial=True)
    
    print(f"Human datasets (>5k cells): {len(human_datasets)} - {human_datasets}")
    print(f"Mouse datasets: {len(mouse_datasets)} - {mouse_datasets[:3]}...")
    print(f"Spatial datasets: {len(spatial_datasets)} - {spatial_datasets}")
    
    # 4. Load a simple dataset
    print("\n4. Loading Dataset (Simplified)")
    print("-" * 30)
    dataset = scgraph_hub.simple_quick_start(
        dataset_name=dataset_name,
        root="./demo_data"
    )
    
    # Show dataset properties
    dataset_info = dataset.info()
    print(f"Loaded dataset: {dataset_info['name']}")
    print(f"Task: {dataset_info['task']}")
    print(f"Number of cells: {dataset_info['num_cells']:,}")
    print(f"Number of genes: {dataset_info['num_genes']:,}")
    print(f"Number of classes: {dataset_info['num_classes']}")
    
    # 5. Examine the data structure
    print("\n5. Data Structure")
    print("-" * 30)
    data = dataset.data
    if data:
        print(f"Data object type: {type(data).__name__}")
        print(f"Number of nodes (cells): {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Number of node features: {data.num_node_features}")
        
        # Check what attributes are available
        attrs = [attr for attr in dir(data) if not attr.startswith('_')]
        print(f"Available attributes: {attrs}")
    else:
        print("No data structure available (requires NumPy)")
    
    # 6. Search functionality
    print("\n6. Dataset Search")
    print("-" * 30)
    brain_datasets = catalog.search("brain")
    covid_datasets = catalog.search("covid")
    pbmc_datasets = catalog.search("pbmc")
    
    print(f"Brain-related datasets: {brain_datasets}")
    print(f"COVID-related datasets: {covid_datasets}")
    print(f"PBMC datasets: {pbmc_datasets}")
    
    # 7. Get recommendations
    print("\n7. Dataset Recommendations")
    print("-" * 30)
    recommendations = catalog.get_recommendations("pbmc_10k", max_results=3)
    print(f"Similar to pbmc_10k: {recommendations}")
    
    # 8. Tasks summary
    print("\n8. Available Tasks")
    print("-" * 30)
    tasks = catalog.get_tasks_summary()
    for task, datasets in tasks.items():
        print(f"  {task}: {len(datasets)} datasets")
    
    # 9. Summary statistics
    print("\n9. Catalog Statistics")
    print("-" * 30)
    stats = catalog.get_summary_stats()
    print(f"Total datasets: {stats.get('total_datasets', 0)}")
    print(f"Total cells: {stats.get('total_cells', 0):,}")
    print(f"Average cells per dataset: {stats.get('avg_cells_per_dataset', 0):,}")
    print(f"Modalities: {list(stats.get('modalities', {}).keys())}")
    print(f"Organisms: {list(stats.get('organisms', {}).keys())}")
    
    print("\n" + "="*60)
    print("✅ Basic functionality demonstration completed!")
    print("\nNext steps:")
    print("- Install PyTorch and dependencies for full functionality")
    print("- Try the advanced examples with machine learning models")
    print("- Explore graph neural network architectures")
    print("="*60)


if __name__ == "__main__":
    main()