"""Scalability and performance optimization features for Single-Cell Graph Hub."""

import logging
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Iterator
from pathlib import Path
import time
import gc
import queue
import threading
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.utils import to_undirected, remove_self_loops
import numpy as np

logger = logging.getLogger(__name__)


class ResourcePool:
    """Thread-safe resource pool for managing expensive resources."""
    
    def __init__(self, resource_factory: Callable, max_size: int = 10):
        """Initialize resource pool.
        
        Args:
            resource_factory: Function to create new resources
            max_size: Maximum pool size
        """
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.pool = queue.Queue(maxsize=max_size)
        self.created_count = 0
        self.lock = threading.Lock()
    
    @contextmanager
    def get_resource(self):
        """Get a resource from the pool (context manager)."""
        resource = None
        try:
            # Try to get existing resource
            try:
                resource = self.pool.get_nowait()
            except queue.Empty:
                # Create new resource if pool is empty and under limit
                with self.lock:
                    if self.created_count < self.max_size:
                        resource = self.resource_factory()
                        self.created_count += 1
                    else:
                        # Wait for available resource
                        resource = self.pool.get()
            
            yield resource
            
        finally:
            if resource is not None:
                # Return resource to pool
                try:
                    self.pool.put_nowait(resource)
                except queue.Full:
                    # Pool is full, discard resource
                    pass


class ConcurrentProcessor:
    """Concurrent processing utilities for batch operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
    
    async def process_batches_async(self, 
                                  batches: List[Any], 
                                  process_func: Callable,
                                  use_processes: bool = False) -> List[Any]:
        """Process batches concurrently.
        
        Args:
            batches: List of batch data
            process_func: Function to process each batch
            use_processes: Whether to use process pool instead of thread pool
            
        Returns:
            List of processed results
        """
        executor = self.process_pool if use_processes else self.thread_pool
        
        # Submit all tasks
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_func, batch)
            for batch in batches
        ]
        
        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} processing failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def process_datasets_parallel(self, 
                                datasets: List[str], 
                                process_func: Callable,
                                max_concurrent: int = 4) -> Dict[str, Any]:
        """Process multiple datasets in parallel.
        
        Args:
            datasets: List of dataset names
            process_func: Function to process each dataset
            max_concurrent: Maximum concurrent processing jobs
            
        Returns:
            Dictionary mapping dataset names to results
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks
            future_to_dataset = {
                executor.submit(process_func, dataset): dataset
                for dataset in datasets
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    results[dataset] = result
                    logger.info(f"Completed processing: {dataset}")
                except Exception as e:
                    logger.error(f"Failed to process {dataset}: {e}")
                    results[dataset] = None
        
        return results
    
    def __del__(self):
        """Clean up executors."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=False)


class LoadBalancer:
    """Load balancing for distributed processing."""
    
    def __init__(self, workers: List[str]):
        """Initialize load balancer.
        
        Args:
            workers: List of worker identifiers
        """
        self.workers = workers
        self.current_loads = {worker: 0 for worker in workers}
        self.task_history = []
        self.lock = threading.Lock()
    
    def get_optimal_worker(self, task_weight: float = 1.0) -> str:
        """Get optimal worker for a task.
        
        Args:
            task_weight: Relative weight of the task
            
        Returns:
            Worker identifier
        """
        with self.lock:
            # Find worker with minimum load
            min_load = min(self.current_loads.values())
            optimal_workers = [
                worker for worker, load in self.current_loads.items()
                if load == min_load
            ]
            
            # If multiple workers have same load, use round-robin
            selected_worker = optimal_workers[len(self.task_history) % len(optimal_workers)]
            
            # Update load
            self.current_loads[selected_worker] += task_weight
            self.task_history.append((selected_worker, task_weight, time.time()))
            
            return selected_worker
    
    def complete_task(self, worker: str, task_weight: float = 1.0):
        """Mark task as completed and update load.
        
        Args:
            worker: Worker that completed the task
            task_weight: Weight of the completed task
        """
        with self.lock:
            self.current_loads[worker] = max(0, self.current_loads[worker] - task_weight)
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            total_load = sum(self.current_loads.values())
            avg_load = total_load / len(self.workers) if self.workers else 0
            
            return {
                'workers': len(self.workers),
                'total_load': total_load,
                'average_load': avg_load,
                'current_loads': self.current_loads.copy(),
                'tasks_completed': len(self.task_history)
            }


@dataclass
class ScalabilityConfig:
    """Configuration for scalability features."""
    max_workers: int = mp.cpu_count()
    batch_size: int = 32
    enable_amp: bool = True  # Automatic Mixed Precision
    enable_checkpointing: bool = True
    memory_efficient: bool = True
    use_distributed: bool = False
    gradient_accumulation_steps: int = 1
    max_memory_gb: float = 8.0


class MemoryOptimizer:
    """Memory optimization utilities for large-scale processing."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        """Initialize memory optimizer.
        
        Args:
            max_memory_gb: Maximum memory usage target in GB
        """
        self.max_memory_gb = max_memory_gb
        self.memory_efficient_mode = False
    
    def enable_memory_efficient_mode(self):
        """Enable memory efficient processing mode."""
        self.memory_efficient_mode = True
        
        # Configure PyTorch for memory efficiency
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False  # Save memory
            torch.backends.cudnn.deterministic = True
        
        logger.info("Memory efficient mode enabled")
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        # Enable gradient checkpointing for transformer-like models
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Convert to half precision if supported
        if torch.cuda.is_available() and self.memory_efficient_mode:
            model = model.half()
            logger.info("Model converted to half precision")
        
        return model
    
    def get_optimal_batch_size(self, data_sample: Data, model: nn.Module, device: torch.device) -> int:
        """Determine optimal batch size based on available memory.
        
        Args:
            data_sample: Sample data object
            model: Model to use for estimation
            device: Target device
            
        Returns:
            Optimal batch size
        """
        # Start with a small batch and increase until memory limit
        model = model.to(device)
        model.eval()
        
        optimal_batch_size = 1
        test_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
        for batch_size in test_batch_sizes:
            try:
                # Create test batch
                batch_data = [data_sample] * batch_size
                batch = Batch.from_data_list(batch_data).to(device)
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(batch.x, batch.edge_index)
                
                # Check memory usage
                if torch.cuda.is_available():
                    memory_used_gb = torch.cuda.memory_allocated(device) / (1024**3)
                    if memory_used_gb > self.max_memory_gb * 0.8:  # 80% threshold
                        break
                
                optimal_batch_size = batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                raise
            finally:
                # Clean up
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        logger.info(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size
    
    def create_memory_efficient_loader(self, 
                                      dataset: List[Data], 
                                      batch_size: Optional[int] = None,
                                      **kwargs) -> DataLoader:
        """Create memory-efficient data loader.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size (auto-determined if None)
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Optimized DataLoader
        """
        if batch_size is None:
            # Use smaller batch size for memory efficiency
            batch_size = min(32, len(dataset))
        
        # Memory-efficient loader settings
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': kwargs.get('shuffle', True),
            'num_workers': min(4, mp.cpu_count()),  # Limit workers to save memory
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': True,
            **kwargs
        }
        
        return DataLoader(dataset, **loader_kwargs)
    
    def cleanup_memory(self):
        """Clean up memory caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug("Memory cleanup completed")


class ParallelProcessor:
    """Parallel processing utilities for large-scale operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = None
        self.process_pool = None
    
    def __enter__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
    
    def process_datasets_parallel(self, 
                                 datasets: List[str],
                                 process_func: Callable[[str], Any],
                                 use_processes: bool = True) -> List[Any]:
        """Process multiple datasets in parallel.
        
        Args:
            datasets: List of dataset names/paths
            process_func: Function to apply to each dataset
            use_processes: Whether to use processes (True) or threads (False)
            
        Returns:
            List of processing results
        """
        executor = self.process_pool if use_processes else self.thread_pool
        
        if executor is None:
            raise RuntimeError("Parallel processor not initialized. Use with context manager.")
        
        futures = [executor.submit(process_func, dataset) for dataset in datasets]
        results = []
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Dataset processing failed: {e}")
                results.append(None)
        
        return results
    
    def batch_process_data(self,
                          data_items: List[Any],
                          process_func: Callable[[List[Any]], Any],
                          batch_size: int = 100) -> List[Any]:
        """Process data in batches for memory efficiency.
        
        Args:
            data_items: List of data items to process
            process_func: Function to apply to each batch
            batch_size: Size of each batch
            
        Returns:
            List of batch processing results
        """
        results = []
        total_batches = (len(data_items) + batch_size - 1) // batch_size
        
        for i in range(0, len(data_items), batch_size):
            batch = data_items[i:i + batch_size]
            batch_idx = i // batch_size + 1
            
            logger.debug(f"Processing batch {batch_idx}/{total_batches}")
            
            try:
                result = process_func(batch)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch {batch_idx} processing failed: {e}")
                results.append(None)
        
        return results
    
    async def async_process_datasets(self,
                                   datasets: List[str],
                                   async_process_func: Callable[[str], Any]) -> List[Any]:
        """Process datasets asynchronously.
        
        Args:
            datasets: List of dataset names/paths
            async_process_func: Async function to apply to each dataset
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(dataset):
            async with semaphore:
                return await async_process_func(dataset)
        
        tasks = [process_with_semaphore(dataset) for dataset in datasets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Async processing failed for dataset {datasets[i]}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results


class ModelScaler:
    """Scale models for large datasets and distributed training."""
    
    def __init__(self, config: ScalabilityConfig):
        """Initialize model scaler.
        
        Args:
            config: Scalability configuration
        """
        self.config = config
        self.scaler = None
        
        if config.enable_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
    
    def wrap_model_for_scaling(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Wrap model with scaling optimizations.
        
        Args:
            model: Model to wrap
            device: Target device
            
        Returns:
            Wrapped model
        """
        model = model.to(device)
        
        # Enable compilation if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Wrap for distributed training if enabled
        if self.config.use_distributed and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Model wrapped for distributed training on {torch.cuda.device_count()} GPUs")
        
        return model
    
    def create_scalable_trainer(self, 
                               model: nn.Module,
                               optimizer: torch.optim.Optimizer,
                               device: torch.device) -> 'ScalableTrainer':
        """Create trainer optimized for scalability.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            device: Training device
            
        Returns:
            Scalable trainer instance
        """
        return ScalableTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            config=self.config,
            scaler=self.scaler
        )


class ScalableTrainer:
    """Scalable trainer with memory and performance optimizations."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 config: ScalabilityConfig,
                 scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """Initialize scalable trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            device: Training device
            config: Scalability configuration
            scaler: AMP scaler for mixed precision training
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scaler = scaler
        
        self.step_count = 0
        self.memory_optimizer = MemoryOptimizer(config.max_memory_gb)
        
        if config.memory_efficient:
            self.memory_optimizer.enable_memory_efficient_mode()
    
    def train_step(self, batch: Batch, criterion: nn.Module) -> Dict[str, float]:
        """Optimized training step with memory and performance optimizations.
        
        Args:
            batch: Training batch
            criterion: Loss function
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        batch = batch.to(self.device)
        
        # Forward pass with mixed precision if enabled
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch.x, batch.edge_index)
                loss = criterion(outputs, batch.y)
        else:
            outputs = self.model(batch.x, batch.edge_index)
            loss = criterion(outputs, batch.y)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights after accumulating gradients
        self.step_count += 1
        if self.step_count % self.config.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Periodic memory cleanup
            if self.step_count % 100 == 0:
                self.memory_optimizer.cleanup_memory()
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate_step(self, batch: Batch, criterion: nn.Module) -> Dict[str, float]:
        """Optimized evaluation step.
        
        Args:
            batch: Evaluation batch
            criterion: Loss function
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch.x, batch.edge_index)
                    loss = criterion(outputs, batch.y)
            else:
                outputs = self.model(batch.x, batch.edge_index)
                loss = criterion(outputs, batch.y)
        
        return {
            'val_loss': loss.item(),
            'predictions': outputs
        }


class BatchProcessor:
    """Efficient batch processing for large-scale inference."""
    
    def __init__(self, model: nn.Module, device: torch.device, batch_size: int = 32):
        """Initialize batch processor.
        
        Args:
            model: Trained model
            device: Inference device
            batch_size: Processing batch size
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.eval()
    
    def process_large_dataset(self, 
                             dataset: List[Data],
                             output_transform: Optional[Callable] = None) -> List[Any]:
        """Process large dataset in batches.
        
        Args:
            dataset: List of data objects
            output_transform: Optional function to transform outputs
            
        Returns:
            List of processed results
        """
        results = []
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in range(0, len(dataset), self.batch_size):
                batch_data = dataset[i:i + self.batch_size]
                batch = Batch.from_data_list(batch_data).to(self.device)
                
                # Process batch
                outputs = self.model(batch.x, batch.edge_index)
                
                # Apply output transform if provided
                if output_transform:
                    outputs = output_transform(outputs)
                
                # Split batch results back to individual items
                batch_results = self._split_batch_outputs(outputs, batch.batch)
                results.extend(batch_results)
                
                # Progress logging
                batch_idx = i // self.batch_size + 1
                if batch_idx % 10 == 0:
                    logger.info(f"Processed batch {batch_idx}/{total_batches}")
        
        return results
    
    def _split_batch_outputs(self, outputs: torch.Tensor, batch_idx: torch.Tensor) -> List[torch.Tensor]:
        """Split batched outputs back to individual results.
        
        Args:
            outputs: Batched output tensor
            batch_idx: Batch indices for splitting
            
        Returns:
            List of individual output tensors
        """
        results = []
        unique_batch_ids = torch.unique(batch_idx)
        
        for batch_id in unique_batch_ids:
            mask = batch_idx == batch_id
            result = outputs[mask]
            results.append(result.cpu())
        
        return results


class GraphSampler:
    """Efficient graph sampling for large-scale graphs."""
    
    def __init__(self, sampling_strategy: str = 'neighbor'):
        """Initialize graph sampler.
        
        Args:
            sampling_strategy: Sampling strategy ('neighbor', 'random', 'cluster')
        """
        self.sampling_strategy = sampling_strategy
    
    def create_neighbor_sampler(self,
                               data: Data,
                               batch_size: int = 1024,
                               num_neighbors: List[int] = [15, 10],
                               **kwargs) -> NeighborLoader:
        """Create neighbor sampling loader for large graphs.
        
        Args:
            data: Full graph data
            batch_size: Number of nodes per batch
            num_neighbors: Number of neighbors to sample per layer
            **kwargs: Additional NeighborLoader arguments
            
        Returns:
            NeighborLoader for scalable training
        """
        # Default arguments for scalability
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': min(4, mp.cpu_count()),
            'persistent_workers': True,
            **kwargs
        }
        
        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            **loader_kwargs
        )
    
    def sample_subgraph(self,
                       data: Data,
                       node_indices: torch.Tensor,
                       num_hops: int = 2) -> Data:
        """Sample subgraph around specified nodes.
        
        Args:
            data: Full graph data
            node_indices: Indices of nodes to sample around
            num_hops: Number of hops for subgraph
            
        Returns:
            Sampled subgraph
        """
        from torch_geometric.utils import k_hop_subgraph
        
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_indices,
            num_hops,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )
        
        # Create subgraph data
        subgraph_data = Data(
            x=data.x[subset],
            edge_index=edge_index,
            y=data.y[subset] if hasattr(data, 'y') and data.y is not None else None
        )
        
        return subgraph_data


# Factory functions for easy scalability setup
def create_scalable_training_setup(model: nn.Module,
                                  data: Data,
                                  device: torch.device,
                                  config: Optional[ScalabilityConfig] = None) -> Tuple[nn.Module, DataLoader, ScalableTrainer]:
    """Create complete scalable training setup.
    
    Args:
        model: Model to train
        data: Training data
        device: Training device
        config: Scalability configuration
        
    Returns:
        Tuple of (optimized_model, data_loader, trainer)
    """
    if config is None:
        config = ScalabilityConfig()
    
    # Memory optimization
    memory_optimizer = MemoryOptimizer(config.max_memory_gb)
    if config.memory_efficient:
        memory_optimizer.enable_memory_efficient_mode()
    
    # Optimize model
    model = memory_optimizer.optimize_model_memory(model)
    
    # Model scaling
    model_scaler = ModelScaler(config)
    model = model_scaler.wrap_model_for_scaling(model, device)
    
    # Create data loader
    if data.num_nodes > 10000:  # Use neighbor sampling for large graphs
        sampler = GraphSampler()
        data_loader = sampler.create_neighbor_sampler(data, batch_size=config.batch_size)
    else:
        # Regular data loader for smaller graphs
        data_loader = DataLoader([data], batch_size=1, shuffle=False)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Create trainer
    trainer = model_scaler.create_scalable_trainer(model, optimizer, device)
    
    return model, data_loader, trainer


def optimize_for_inference(model: nn.Module, 
                          sample_data: Data,
                          device: torch.device) -> nn.Module:
    """Optimize model for fast inference.
    
    Args:
        model: Trained model
        sample_data: Sample data for optimization
        device: Inference device
        
    Returns:
        Optimized model
    """
    model = model.to(device)
    model.eval()
    
    # Convert to half precision for faster inference
    if torch.cuda.is_available():
        model = model.half()
        sample_data = sample_data.to(device).half()
    
    # JIT compilation for faster execution
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (sample_data.x, sample_data.edge_index))
        logger.info("Model traced with TorchScript")
        return traced_model
    
    except Exception as e:
        logger.warning(f"Model tracing failed: {e}, returning original model")
        return model


# Convenience function for auto-scaling
def auto_scale_training(model: nn.Module,
                       data: Data,
                       target_memory_gb: float = 8.0,
                       target_time_per_epoch: float = 300.0) -> Dict[str, Any]:
    """Automatically determine optimal scaling configuration.
    
    Args:
        model: Model to train
        data: Training data
        target_memory_gb: Target memory usage
        target_time_per_epoch: Target time per epoch in seconds
        
    Returns:
        Optimal configuration dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Memory-based optimization
    memory_optimizer = MemoryOptimizer(target_memory_gb)
    optimal_batch_size = memory_optimizer.get_optimal_batch_size(data, model, device)
    
    # Determine if neighbor sampling is needed
    use_neighbor_sampling = data.num_nodes > 10000
    
    # Determine gradient accumulation based on batch size
    gradient_accumulation_steps = max(1, 32 // optimal_batch_size)
    
    config = ScalabilityConfig(
        batch_size=optimal_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_memory_gb=target_memory_gb,
        memory_efficient=True,
        enable_amp=torch.cuda.is_available(),
        use_distributed=torch.cuda.device_count() > 1
    )
    
    return {
        'config': config,
        'use_neighbor_sampling': use_neighbor_sampling,
        'estimated_memory_gb': target_memory_gb * 0.8,  # Conservative estimate
        'recommendations': {
            'batch_size': optimal_batch_size,
            'use_mixed_precision': torch.cuda.is_available(),
            'use_gradient_accumulation': gradient_accumulation_steps > 1
        }
    }
