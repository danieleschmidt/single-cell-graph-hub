#!/usr/bin/env python3
"""Command-line interface for Single-Cell Graph Hub."""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import scgraph_hub as scgh


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_info(args):
    """Show system information."""
    print(f"Single-Cell Graph Hub v{scgh.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # Check dependencies
    try:
        dependencies = scgh.check_dependencies()
        print("\nDependency Status:")
        for name, status in dependencies.items():
            status_icon = "✓" if status else "✗"
            print(f"  {status_icon} {name}")
    except Exception as e:
        print(f"Could not check dependencies: {e}")


def cmd_list_datasets(args):
    """List available datasets."""
    try:
        catalog = scgh.get_default_catalog()
        datasets = catalog.list_datasets()
        
        if not datasets:
            print("No datasets found in catalog.")
            return
        
        print(f"Available datasets ({len(datasets)}):")
        print("-" * 60)
        
        for dataset_name in datasets:
            info = catalog.get_info(dataset_name)
            if info:
                cells = info.get('n_cells', 'Unknown')
                genes = info.get('n_genes', 'Unknown') 
                modality = info.get('modality', 'Unknown')
                print(f"  {dataset_name:<20} | {cells:>8} cells | {genes:>8} genes | {modality}")
            else:
                print(f"  {dataset_name:<20} | No info available")
                
    except Exception as e:
        print(f"Error listing datasets: {e}")
        sys.exit(1)


def cmd_list_models(args):
    """List available models."""
    try:
        if hasattr(scgh, 'MODEL_REGISTRY'):
            models = scgh.MODEL_REGISTRY
            print(f"Available models ({len(models)}):")
            print("-" * 40)
            
            for model_name, model_class in models.items():
                doc = model_class.__doc__ or "No description available"
                first_line = doc.split('\n')[0].strip()
                print(f"  {model_name:<15} | {first_line}")
        else:
            print("Model registry not available. Install full dependencies.")
            
    except Exception as e:
        print(f"Error listing models: {e}")
        sys.exit(1)


def cmd_train(args):
    """Train a model."""
    try:
        print(f"Training {args.model} on {args.dataset}...")
        
        # Load dataset  
        if hasattr(scgh, 'load_dataset'):
            data = scgh.load_dataset(args.dataset)
            if data is None:
                print(f"Could not load dataset: {args.dataset}")
                sys.exit(1)
        else:
            print("Dataset loading not available. Install full dependencies.")
            sys.exit(1)
        
        # Create model
        if hasattr(scgh, 'create_model'):
            model_kwargs = {
                'input_dim': data.x.shape[1],
                'output_dim': len(torch.unique(data.y)) if hasattr(data, 'y') else 10,
                'hidden_dim': args.hidden_dim,
            }
            
            model = scgh.create_model(args.model, **model_kwargs)
            print(f"Created {args.model} with {model.num_parameters()} parameters")
        else:
            print("Model creation not available. Install full dependencies.")
            sys.exit(1)
        
        # Training configuration
        device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
        model = model.to(device)
        data = data.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"Training on {device} for {args.epochs} epochs...")
        
        # Training loop
        model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Validation
                model.eval()
                with torch.no_grad():
                    val_out = model(data.x, data.edge_index)
                    val_pred = val_out[data.val_mask].argmax(dim=1)
                    val_acc = (val_pred == data.y[data.val_mask]).float().mean()
                
                print(f"Epoch {epoch+1:3d}: Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
                model.train()
        
        # Save model if requested
        if args.output:
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': args.model,
                'model_kwargs': model_kwargs,
                'training_args': vars(args)
            }, args.output)
            print(f"Model saved to {args.output}")
        
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_test(args):
    """Run system tests."""
    test_type = args.test_type
    
    if test_type == 'basic':
        print("Running basic functionality tests...")
        exit_code = run_basic_tests()
    elif test_type == 'quality':
        print("Running quality gates...")
        exit_code = run_quality_gates()
    elif test_type == 'all':
        print("Running all tests...")
        exit_code = run_all_tests()
    else:
        print(f"Unknown test type: {test_type}")
        sys.exit(1)
    
    sys.exit(exit_code)


def run_basic_tests() -> int:
    """Run basic functionality tests."""
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'examples/simple_test.py'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result.returncode
    except Exception as e:
        print(f"Error running basic tests: {e}")
        return 1


def run_quality_gates() -> int:
    """Run quality gates."""
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'tests/test_quality_gates.py'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result.returncode
    except Exception as e:
        print(f"Error running quality gates: {e}")
        return 1


def run_all_tests() -> int:
    """Run all test suites."""
    print("Running basic tests...")
    basic_result = run_basic_tests()
    
    print("\nRunning quality gates...")
    quality_result = run_quality_gates()
    
    if basic_result == 0 and quality_result == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


def cmd_benchmark(args):
    """Run performance benchmarks."""
    try:
        from scgraph_hub.monitoring import PerformanceMonitor
        from scgraph_hub.caching import PerformanceOptimizer
        
        print("Running performance benchmarks...")
        
        # Optimize system
        optimizer = PerformanceOptimizer()
        optimizer.optimize_tensor_operations()
        
        monitor = PerformanceMonitor()
        
        # Create test data
        n_cells = args.dataset_size
        n_features = 1000
        x = torch.randn(n_cells, n_features)
        edge_index = torch.randint(0, n_cells, (2, n_cells * 10))
        
        # Test model creation
        monitor.start_timer("model_creation")
        model = scgh.CellGraphGNN(input_dim=n_features, hidden_dim=128, output_dim=10)
        model_time = monitor.end_timer("model_creation")
        
        # Test forward pass
        monitor.start_timer("forward_pass")
        output = model(x, edge_index)
        forward_time = monitor.end_timer("forward_pass")
        
        # Results
        print(f"\nBenchmark Results (dataset size: {n_cells} cells):")
        print(f"Model creation: {model_time:.4f}s")
        print(f"Forward pass: {forward_time:.4f}s")
        print(f"Model parameters: {model.num_parameters():,}")
        print(f"Output shape: {output.shape}")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"GPU memory: {memory_mb:.2f}MB")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Single-Cell Graph Hub CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scgraph-hub info                          # Show system info
  scgraph-hub list datasets               # List available datasets  
  scgraph-hub list models                 # List available models
  scgraph-hub train cellgnn pbmc3k        # Train CellGraphGNN on PBMC3k
  scgraph-hub test basic                  # Run basic tests
  scgraph-hub benchmark --size 10000     # Run performance benchmark
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--version', action='version', 
                       version=f'Single-Cell Graph Hub {scgh.__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.set_defaults(func=cmd_info)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available resources')
    list_subparsers = list_parser.add_subparsers(dest='list_type')
    
    datasets_parser = list_subparsers.add_parser('datasets', help='List datasets')
    datasets_parser.set_defaults(func=cmd_list_datasets)
    
    models_parser = list_subparsers.add_parser('models', help='List models')
    models_parser.set_defaults(func=cmd_list_models)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('model', help='Model name')
    train_parser.add_argument('dataset', help='Dataset name')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    train_parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    train_parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    train_parser.add_argument('--output', '-o', help='Output path for trained model')
    train_parser.set_defaults(func=cmd_train)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('test_type', choices=['basic', 'quality', 'all'], 
                            help='Type of tests to run')
    test_parser.set_defaults(func=cmd_test)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--size', type=int, default=5000, 
                                 help='Dataset size for benchmark')
    benchmark_parser.set_defaults(func=cmd_benchmark)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()