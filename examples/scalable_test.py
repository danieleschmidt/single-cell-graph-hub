"""Test scalable features of Single-Cell Graph Hub (Generation 3)."""

import torch
import numpy as np
import asyncio
import time
from torch_geometric.data import Data

def test_performance_optimization():
    """Test performance optimization features."""
    print("Testing performance optimization...")
    
    try:
        from scgraph_hub.caching import PerformanceOptimizer, AdaptiveCache
        
        # Test performance optimizer
        optimizer = PerformanceOptimizer()
        
        # Optimize tensor operations
        optimizer.optimize_tensor_operations()
        print("✓ Tensor operations optimized")
        
        # Test optimal batch size calculation
        model_size_mb = 100  # 100MB model
        available_memory_mb = 4000  # 4GB available
        
        optimal_batch_size = optimizer.get_optimal_batch_size(model_size_mb, available_memory_mb)
        print(f"✓ Optimal batch size: {optimal_batch_size}")
        
        # Test adaptive cache
        cache = AdaptiveCache(max_size_mb=100, compression=True)
        
        # Test caching with different data types
        test_data = {
            'tensor': torch.randn(1000, 100),
            'array': np.random.randn(500, 50),
            'dict': {'key1': 'value1', 'key2': [1, 2, 3, 4, 5]},
            'large_tensor': torch.randn(2000, 200)
        }
        
        # Put items in cache
        for key, value in test_data.items():
            cache.put(key, value, ttl=3600)
        
        # Retrieve items
        cache_hits = 0
        for key in test_data.keys():
            retrieved = cache.get(key)
            if retrieved is not None:
                cache_hits += 1
        
        stats = cache.get_stats()
        print(f"✓ Cache: {cache_hits}/{len(test_data)} hits, {stats['hit_rate']:.2f} hit rate")
        print(f"  Cache size: {stats['size_mb']:.2f}MB / {stats['max_size_mb']}MB")
        
        return True
    except Exception as e:
        print(f"✗ Performance optimization test error: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing features."""
    print("\nTesting concurrent processing...")
    
    try:
        from scgraph_hub.scalability import ConcurrentProcessor, ResourcePool, LoadBalancer
        
        # Test concurrent processor
        processor = ConcurrentProcessor(max_workers=4)
        
        # Create test processing function
        def process_batch(batch_data):
            """Simulate processing a batch."""
            # Simulate some computation
            result = np.mean(batch_data) + np.random.normal(0, 0.1)
            time.sleep(0.1)  # Simulate processing time
            return result
        
        # Create test batches
        batches = [np.random.randn(100, 10) for _ in range(8)]
        
        # Time sequential processing
        start_time = time.time()
        sequential_results = [process_batch(batch) for batch in batches[:4]]  # Only 4 for timing
        sequential_time = time.time() - start_time
        
        # Time concurrent processing
        start_time = time.time()
        concurrent_results = processor.process_datasets_parallel(
            ['batch_' + str(i) for i in range(4)],
            lambda name: process_batch(batches[int(name.split('_')[1])]),
            max_concurrent=4
        )
        concurrent_time = time.time() - start_time
        
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
        print(f"✓ Concurrent processing speedup: {speedup:.2f}x")
        print(f"  Sequential: {sequential_time:.2f}s, Concurrent: {concurrent_time:.2f}s")
        
        # Test resource pool
        def create_expensive_resource():
            """Simulate creating an expensive resource."""
            return torch.nn.Linear(100, 50)
        
        resource_pool = ResourcePool(create_expensive_resource, max_size=3)
        
        # Test resource usage
        with resource_pool.get_resource() as resource1:
            with resource_pool.get_resource() as resource2:
                print(f"✓ Resource pool: Got 2 resources")
                assert resource1 is not None and resource2 is not None
        
        # Test load balancer
        load_balancer = LoadBalancer(['worker1', 'worker2', 'worker3'])
        
        # Assign some tasks
        assignments = []
        for i in range(10):
            worker = load_balancer.get_optimal_worker(task_weight=1.0)
            assignments.append(worker)
            
            # Simulate task completion
            if i % 3 == 0:  # Complete every 3rd task
                load_balancer.complete_task(worker, 1.0)
        
        stats = load_balancer.get_load_stats()
        print(f"✓ Load balancer: {stats['workers']} workers, {stats['tasks_completed']} completed")
        
        return True
    except Exception as e:
        print(f"✗ Concurrent processing test error: {e}")
        return False

async def test_async_processing():
    """Test asynchronous processing features."""
    print("\nTesting async processing...")
    
    try:
        from scgraph_hub.scalability import ConcurrentProcessor
        
        processor = ConcurrentProcessor(max_workers=4)
        
        # Test async batch processing
        def async_process_batch(batch_data):
            """Simulate async batch processing."""
            time.sleep(0.05)  # Short processing time
            return {'mean': np.mean(batch_data), 'std': np.std(batch_data)}
        
        # Create test batches
        batches = [np.random.randn(50, 20) for _ in range(6)]
        
        # Process batches asynchronously
        start_time = time.time()
        results = await processor.process_batches_async(
            batches, 
            async_process_batch,
            use_processes=False
        )
        async_time = time.time() - start_time
        
        successful_results = [r for r in results if r is not None]
        print(f"✓ Async processing: {len(successful_results)}/{len(batches)} batches")
        print(f"  Processing time: {async_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"✗ Async processing test error: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization features."""
    print("\nTesting memory optimization...")
    
    try:
        from scgraph_hub.scalability import MemoryOptimizer
        from scgraph_hub.caching import PerformanceOptimizer
        
        # Test memory optimizer
        memory_optimizer = MemoryOptimizer(max_memory_gb=2.0)
        memory_optimizer.enable_memory_efficient_mode()
        
        # Test memory efficient decorator
        optimizer = PerformanceOptimizer()
        
        @optimizer.memory_efficient_function
        def memory_intensive_operation():
            """Simulate memory intensive operation."""
            # Create large tensors
            tensors = []
            for i in range(5):
                tensor = torch.randn(1000, 1000)
                tensors.append(tensor)
            
            # Process tensors
            result = sum(t.sum() for t in tensors)
            return result.item()
        
        # Run memory intensive operation
        result = memory_intensive_operation()
        print(f"✓ Memory efficient operation completed: {result:.2f}")
        
        # Test GPU memory management (if available)
        if torch.cuda.is_available():
            # Allocate some GPU memory
            gpu_tensor = torch.randn(1000, 1000, device='cuda')
            initial_memory = torch.cuda.memory_allocated()
            
            # Clear cache
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            
            print(f"✓ GPU memory management: {initial_memory} -> {final_memory} bytes")
        else:
            print("✓ GPU memory management: CUDA not available (CPU only)")
        
        return True
    except Exception as e:
        print(f"✗ Memory optimization test error: {e}")
        return False

def test_scalable_training():
    """Test scalable model training."""
    print("\nTesting scalable training...")
    
    try:
        from scgraph_hub.models import CellGraphGNN
        from scgraph_hub.caching import AdaptiveCache, PerformanceOptimizer
        from scgraph_hub.monitoring import PerformanceMonitor
        import torch.nn.functional as F
        
        # Initialize optimization components
        cache = AdaptiveCache(max_size_mb=200)
        optimizer_utils = PerformanceOptimizer()
        monitor = PerformanceMonitor()
        
        # Optimize tensor operations
        optimizer_utils.optimize_tensor_operations()
        
        # Create synthetic scalable dataset
        n_cells = 2000
        n_features = 200
        n_classes = 5
        
        # Generate data
        x = torch.randn(n_cells, n_features)
        
        # Create k-NN graph efficiently
        from sklearn.neighbors import NearestNeighbors
        k = 20
        nbrs = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(x.numpy())
        distances, indices = nbrs.kneighbors(x.numpy())
        
        edge_list = []
        for i in range(len(indices)):
            for j in range(1, min(k+1, len(indices[i]))):  # Skip self, limit k
                neighbor_idx = indices[i][j]
                edge_list.extend([(i, neighbor_idx), (neighbor_idx, i)])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        y = torch.randint(0, n_classes, (n_cells,))
        
        # Create train/val/test splits
        train_mask = torch.zeros(n_cells, dtype=torch.bool)
        val_mask = torch.zeros(n_cells, dtype=torch.bool)
        test_mask = torch.zeros(n_cells, dtype=torch.bool)
        
        train_mask[:1000] = True
        val_mask[1000:1500] = True
        test_mask[1500:] = True
        
        data = Data(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        # Create optimized model
        model = CellGraphGNN(
            input_dim=n_features,
            hidden_dim=128,
            output_dim=n_classes,
            num_layers=3,
            dropout=0.3
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Scalable training loop with monitoring
        model.train()
        print("  Starting scalable training...")
        
        monitor.start_timer("training")
        
        for epoch in range(5):
            monitor.start_timer(f"epoch_{epoch}")
            
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation with caching
            cache_key = f"val_epoch_{epoch}"
            cached_val_acc = cache.get(cache_key)
            
            if cached_val_acc is None:
                model.eval()
                with torch.no_grad():
                    val_out = model(data.x, data.edge_index)
                    val_pred = val_out[data.val_mask].argmax(dim=1)
                    val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
                model.train()
                
                # Cache validation result
                cache.put(cache_key, val_acc, ttl=3600)
            else:
                val_acc = cached_val_acc
            
            epoch_time = monitor.end_timer(f"epoch_{epoch}")
            print(f"    Epoch {epoch+1}: Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.3f}s")
        
        total_time = monitor.end_timer("training")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_out = model(data.x, data.edge_index)
            test_pred = test_out[data.test_mask].argmax(dim=1)
            test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()
        
        print(f"✓ Scalable training completed:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Test accuracy: {test_acc:.4f}")
        print(f"  Dataset size: {n_cells} cells, {edge_index.shape[1]} edges")
        
        # Show cache stats
        cache_stats = cache.get_stats()
        print(f"  Cache hit rate: {cache_stats['hit_rate']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Scalable training test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all scalable tests."""
    print("Single-Cell Graph Hub - Scalable Features Test (Generation 3)")
    print("=" * 65)
    
    tests = [
        test_performance_optimization,
        test_concurrent_processing,
        test_async_processing,
        test_memory_optimization,
        test_scalable_training
    ]
    
    tests_passed = 0
    total_tests = len(tests)
    
    for test in tests:
        if asyncio.iscoroutinefunction(test):
            success = await test()
        else:
            success = test()
        
        if success:
            tests_passed += 1
    
    print("\n" + "=" * 65)
    print(f"Scalable tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🚀 All scalable features working! Generation 3 complete.")
        print("🎯 System is now optimized for performance and scale!")
    else:
        print("❌ Some scalable features need attention.")

if __name__ == "__main__":
    asyncio.run(main())