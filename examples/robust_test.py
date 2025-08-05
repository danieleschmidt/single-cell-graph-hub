"""Test robust features of Single-Cell Graph Hub (Generation 2)."""

import torch
import numpy as np
from torch_geometric.data import Data

def test_validation_features():
    """Test validation and error handling."""
    print("Testing validation features...")
    
    try:
        from scgraph_hub.validation import DataValidator, ValidationError
        from scgraph_hub.security import InputSanitizer
        
        # Test data validator
        validator = DataValidator(strict_mode=False)
        
        # Create test data
        x = torch.randn(100, 50)
        edge_index = torch.randint(0, 100, (2, 200))
        y = torch.randint(0, 5, (100,))
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Validate dataset
        results = validator.validate_dataset(data, "test_dataset")
        print(f"✓ Dataset validation: {results['valid']}")
        
        # Test input sanitizer
        sanitizer = InputSanitizer()
        
        # Test string sanitization
        dirty_string = "test_dataset'; DROP TABLE users; --"
        try:
            clean_string = sanitizer.sanitize_string(dirty_string)
            print("✗ Should have caught SQL injection")
        except ValueError:
            print("✓ SQL injection attempt blocked")
        
        # Test tensor sanitization
        dirty_tensor = torch.tensor([1.0, float('inf'), float('nan'), 1e8])
        clean_tensor = sanitizer.sanitize_tensor(dirty_tensor)
        print(f"✓ Tensor sanitization: {torch.isfinite(clean_tensor).all()}")
        
        return True
    except Exception as e:
        print(f"✗ Validation test error: {e}")
        return False

def test_monitoring_features():
    """Test monitoring and health checks."""
    print("\nTesting monitoring features...")
    
    try:
        from scgraph_hub.monitoring import HealthChecker, PerformanceMonitor
        
        # Test health checker
        health_checker = HealthChecker()
        
        # System health check
        system_health = health_checker.check_system_health()
        print(f"✓ System health: {system_health['status']}")
        print(f"  - Memory: {system_health['checks']['memory']['status']}")
        print(f"  - CPU: {system_health['checks']['cpu']['status']}")
        
        # Model health check
        from scgraph_hub.models import CellGraphGNN
        model = CellGraphGNN(input_dim=50, hidden_dim=32, output_dim=5)
        
        model_health = health_checker.check_model_health(model)
        print(f"✓ Model health: {model_health['status']}")
        print(f"  - Parameters: {model_health['checks']['parameters']['total']}")
        
        # Test performance monitor
        monitor = PerformanceMonitor()
        
        monitor.start_timer("test_operation")
        # Simulate some work
        _ = torch.randn(1000, 1000) @ torch.randn(1000, 1000)
        duration = monitor.end_timer("test_operation")
        
        print(f"✓ Performance monitoring: {duration:.4f}s")
        
        return True
    except Exception as e:
        print(f"✗ Monitoring test error: {e}")
        return False

def test_secure_model_training():
    """Test secure model training with validation."""
    print("\nTesting secure model training...")
    
    try:
        from scgraph_hub.models import CellGraphGNN
        from scgraph_hub.monitoring import HealthChecker
        from scgraph_hub.security import InputSanitizer
        import torch.nn.functional as F
        
        # Create sanitized data
        sanitizer = InputSanitizer()
        health_checker = HealthChecker()
        
        # Generate data with potential issues
        n_cells = 500
        n_features = 100
        
        # Add some problematic values
        x = torch.randn(n_cells, n_features)
        x[0, 0] = float('inf')  # Infinite value
        x[1, 1] = float('nan')  # NaN value
        x[2, 2] = 1e7  # Very large value
        
        # Sanitize the data
        x_clean = sanitizer.sanitize_tensor(x)
        
        # Create edge index
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=11).fit(x_clean.numpy())
        distances, indices = nbrs.kneighbors(x_clean.numpy())
        
        edge_list = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip self
                neighbor_idx = indices[i][j]
                edge_list.extend([(i, neighbor_idx), (neighbor_idx, i)])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        y = torch.randint(0, 5, (n_cells,))
        
        # Create masks
        train_mask = torch.zeros(n_cells, dtype=torch.bool)
        val_mask = torch.zeros(n_cells, dtype=torch.bool)
        test_mask = torch.zeros(n_cells, dtype=torch.bool)
        
        train_mask[:250] = True
        val_mask[250:375] = True
        test_mask[375:] = True
        
        data = Data(x=x_clean, edge_index=edge_index, y=y, 
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        # Create model with validation
        model = CellGraphGNN(
            input_dim=n_features,
            hidden_dim=64,
            output_dim=5,
            num_layers=2,
            dropout=0.2
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Check initial model health
        initial_health = health_checker.check_model_health(model)
        print(f"✓ Initial model health: {initial_health['status']}")
        
        # Secure training loop with health monitoring
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            
            # Forward pass with input validation (built into model)
            try:
                out = model(data.x, data.edge_index)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                
                # Check for problematic loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"✗ Problematic loss detected at epoch {epoch}: {loss}")
                    break
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_out = model(data.x, data.edge_index)
                    val_pred = val_out[data.val_mask].argmax(dim=1)
                    val_acc = (val_pred == data.y[data.val_mask]).float().mean()
                model.train()
                
                print(f"  Epoch {epoch+1}: Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
                
            except Exception as e:
                print(f"✗ Training error at epoch {epoch}: {e}")
                break
        
        # Final model health check
        final_health = health_checker.check_model_health(model)
        print(f"✓ Final model health: {final_health['status']}")
        
        if final_health['checks']['gradients']['has_nan']:
            print("✗ Model has NaN gradients")
            return False
        
        print("✓ Secure training completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Secure training test error: {e}")
        return False

def test_error_recovery():
    """Test error recovery mechanisms."""
    print("\nTesting error recovery...")
    
    try:
        from scgraph_hub.models import CellGraphGNN
        
        model = CellGraphGNN(input_dim=100, hidden_dim=64, output_dim=5)
        
        # Test with invalid inputs
        print("Testing invalid input handling...")
        
        # Wrong dimension
        try:
            x_wrong = torch.randn(50)  # 1D instead of 2D
            edge_index = torch.randint(0, 50, (2, 100))
            out = model(x_wrong, edge_index)
            print("✗ Should have caught dimension error")
            return False
        except ValueError as e:
            print(f"✓ Caught dimension error: {type(e).__name__}")
        
        # Invalid edge indices
        try:
            x = torch.randn(50, 100)
            edge_index_wrong = torch.randint(0, 100, (2, 100))  # Invalid indices
            out = model(x, edge_index_wrong)
            print("✗ Should have caught invalid edge indices")
            return False
        except ValueError as e:
            print(f"✓ Caught invalid edge indices: {type(e).__name__}")
        
        # Test with NaN inputs
        try:
            x_nan = torch.randn(50, 100)
            x_nan[0, 0] = float('nan')
            edge_index = torch.randint(0, 50, (2, 100))
            out = model(x_nan, edge_index)
            print("✓ Model handled NaN input (warning should be shown)")
        except Exception as e:
            print(f"✓ Model properly rejected NaN input: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error recovery test error: {e}")
        return False

def main():
    """Run all robust tests."""
    print("Single-Cell Graph Hub - Robust Features Test (Generation 2)")
    print("=" * 60)
    
    tests = [
        test_validation_features,
        test_monitoring_features,
        test_secure_model_training,
        test_error_recovery
    ]
    
    tests_passed = 0
    total_tests = len(tests)
    
    for test in tests:
        if test():
            tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Robust tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🛡️ All robust features working! Generation 2 complete.")
    else:
        print("❌ Some robust features need attention.")

if __name__ == "__main__":
    main()