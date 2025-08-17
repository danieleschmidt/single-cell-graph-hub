"""Enhanced Simple Dataset Loader for Generation 1."""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field


@dataclass
class DatasetInfo:
    """Basic dataset information container."""
    name: str
    size: int = 0
    features: int = 0
    nodes: int = 0
    edges: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'size': self.size,
            'features': self.features,
            'nodes': self.nodes,
            'edges': self.edges,
            'metadata': self.metadata
        }


class EnhancedSimpleLoader:
    """Enhanced simple dataset loader with error handling and monitoring."""
    
    def __init__(self, root_dir: str = "./data", logger: Optional[logging.Logger] = None):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        self._dataset_cache: Dict[str, DatasetInfo] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load dataset cache from disk."""
        cache_file = self.root_dir / "dataset_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                for name, info_dict in cache_data.items():
                    self._dataset_cache[name] = DatasetInfo(**info_dict)
                self.logger.info(f"Loaded cache with {len(self._dataset_cache)} datasets")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self) -> None:
        """Save dataset cache to disk."""
        cache_file = self.root_dir / "dataset_cache.json"
        try:
            cache_data = {name: info.to_dict() for name, info in self._dataset_cache.items()}
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            self.logger.debug("Saved dataset cache")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets."""
        datasets = list(self._dataset_cache.keys())
        
        # Also scan directory for any files
        try:
            for item in self.root_dir.iterdir():
                if item.is_file() and item.suffix in ['.h5', '.h5ad', '.json']:
                    name = item.stem
                    if name not in datasets:
                        datasets.append(name)
        except Exception as e:
            self.logger.warning(f"Error scanning directory: {e}")
        
        return sorted(datasets)
    
    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        """Get information about a dataset."""
        if name in self._dataset_cache:
            return self._dataset_cache[name]
        
        # Try to infer from file
        dataset_path = self.root_dir / f"{name}.json"
        if dataset_path.exists():
            try:
                with open(dataset_path, 'r') as f:
                    metadata = json.load(f)
                info = DatasetInfo(
                    name=name,
                    size=metadata.get('size', 0),
                    features=metadata.get('features', 0),
                    nodes=metadata.get('nodes', 0),
                    edges=metadata.get('edges', 0),
                    metadata=metadata
                )
                self._dataset_cache[name] = info
                self._save_cache()
                return info
            except Exception as e:
                self.logger.warning(f"Failed to load dataset info for {name}: {e}")
        
        return None
    
    def load_dataset(self, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Load a dataset with error handling."""
        self.logger.info(f"Loading dataset: {name}")
        
        try:
            # First try simple JSON format
            json_path = self.root_dir / f"{name}.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Update cache
                info = DatasetInfo(
                    name=name,
                    size=len(data.get('data', [])),
                    features=data.get('num_features', 0),
                    nodes=data.get('num_nodes', 0),
                    edges=data.get('num_edges', 0),
                    metadata=data.get('metadata', {})
                )
                self._dataset_cache[name] = info
                self._save_cache()
                
                self.logger.info(f"Successfully loaded dataset {name}")
                return data
            
            # Try to create a dummy dataset for testing
            if name.startswith('test_') or name == 'dummy' or name.startswith('demo_'):
                return self._create_dummy_dataset(name, **kwargs)
            
            self.logger.warning(f"Dataset {name} not found")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {name}: {e}")
            return None
    
    def _create_dummy_dataset(self, name: str, num_nodes: int = 100, num_features: int = 50) -> Dict[str, Any]:
        """Create a dummy dataset for testing."""
        self.logger.info(f"Creating dummy dataset {name} with {num_nodes} nodes, {num_features} features")
        try:
            import numpy as np
            
            # Generate dummy data
            features = np.random.randn(num_nodes, num_features).astype(np.float32)
            edges = []
            
            # Create simple k-nearest neighbor graph
            k = min(10, num_nodes - 1)
            for i in range(num_nodes):
                neighbors = np.random.choice(
                    [j for j in range(num_nodes) if j != i], 
                    size=min(k, num_nodes - 1), 
                    replace=False
                )
                for neighbor in neighbors:
                    edges.append([i, neighbor])
            
            edge_index = np.array(edges).T if edges else np.empty((2, 0))
            labels = np.random.randint(0, 5, size=num_nodes)
            
            data = {
                'x': features.tolist(),
                'edge_index': edge_index.tolist(),
                'y': labels.tolist(),
                'num_nodes': num_nodes,
                'num_features': num_features,
                'num_edges': len(edges),
                'metadata': {
                    'name': name,
                    'type': 'dummy',
                    'created': 'auto-generated'
                }
            }
            
            # Save to cache
            info = DatasetInfo(
                name=name,
                size=num_nodes,
                features=num_features,
                nodes=num_nodes,
                edges=len(edges),
                metadata=data['metadata']
            )
            self._dataset_cache[name] = info
            self._save_cache()
            
            self.logger.info(f"Created dummy dataset {name} with {num_nodes} nodes")
            return data
            
        except ImportError:
            self.logger.info("NumPy not available, creating simple dummy dataset")
            # Create simple dummy data without NumPy
            import random
            features = [[random.random() for _ in range(num_features)] for _ in range(num_nodes)]
            edges = []
            k = min(5, num_nodes - 1)
            for i in range(num_nodes):
                for j in range(min(k, num_nodes - i - 1)):
                    neighbor = (i + j + 1) % num_nodes
                    edges.append([i, neighbor])
            
            edge_index = [[], []]
            for edge in edges:
                edge_index[0].append(edge[0])
                edge_index[1].append(edge[1])
            
            labels = [random.randint(0, 4) for _ in range(num_nodes)]
            
            data = {
                'x': features,
                'edge_index': edge_index,
                'y': labels,
                'num_nodes': num_nodes,
                'num_features': num_features,
                'num_edges': len(edges),
                'metadata': {'name': name, 'type': 'simple_dummy', 'created': 'auto-generated'}
            }
            
            # Save to cache
            info = DatasetInfo(
                name=name,
                size=num_nodes,
                features=num_features,
                nodes=num_nodes,
                edges=len(edges),
                metadata=data['metadata']
            )
            self._dataset_cache[name] = info
            self._save_cache()
            
            self.logger.info(f"Created simple dummy dataset {name} with {num_nodes} nodes")
            return data
        except Exception as e:
            self.logger.error(f"Failed to create dummy dataset: {e}")
            return None
    
    def save_dataset(self, name: str, data: Dict[str, Any]) -> bool:
        """Save a dataset to disk."""
        try:
            output_path = self.root_dir / f"{name}.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update cache
            info = DatasetInfo(
                name=name,
                size=data.get('num_nodes', 0),
                features=data.get('num_features', 0),
                nodes=data.get('num_nodes', 0),
                edges=data.get('num_edges', 0),
                metadata=data.get('metadata', {})
            )
            self._dataset_cache[name] = info
            self._save_cache()
            
            self.logger.info(f"Saved dataset {name} to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset {name}: {e}")
            return False
    
    def validate_dataset(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate dataset format."""
        errors = []
        
        # Check required fields
        required_fields = ['x', 'edge_index', 'y']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if not errors:
            # Check data consistency
            try:
                num_nodes = len(data['x'])
                num_labels = len(data['y'])
                
                if num_nodes != num_labels:
                    errors.append(f"Node count mismatch: {num_nodes} features vs {num_labels} labels")
                
                if 'edge_index' in data and data['edge_index']:
                    edge_index = data['edge_index']
                    if len(edge_index) == 2:
                        max_node_idx = max(max(edge_index[0]), max(edge_index[1]))
                        if max_node_idx >= num_nodes:
                            errors.append(f"Edge index out of bounds: max {max_node_idx}, nodes {num_nodes}")
                    else:
                        errors.append("Edge index should have shape [2, num_edges]")
                        
            except Exception as e:
                errors.append(f"Data validation error: {e}")
        
        return len(errors) == 0, errors
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the loader."""
        result = {
            'status': 'healthy',
            'root_dir': str(self.root_dir),
            'root_exists': self.root_dir.exists(),
            'cache_size': len(self._dataset_cache),
            'available_datasets': len(self.list_available_datasets()),
            'errors': []
        }
        
        try:
            # Test dummy dataset creation
            dummy_data = self._create_dummy_dataset('health_check_test', num_nodes=5, num_features=3)
            if dummy_data:
                is_valid, validation_errors = self.validate_dataset(dummy_data)
                if not is_valid:
                    result['errors'].extend(validation_errors)
                    result['status'] = 'warning'
            else:
                result['errors'].append("Failed to create dummy dataset")
                result['status'] = 'error'
                
        except Exception as e:
            result['errors'].append(f"Health check error: {e}")
            result['status'] = 'error'
        
        return result


# Global loader instance
_global_loader: Optional[EnhancedSimpleLoader] = None


def get_enhanced_loader(root_dir: str = "./data") -> EnhancedSimpleLoader:
    """Get the global enhanced loader instance."""
    global _global_loader
    if _global_loader is None or str(_global_loader.root_dir) != root_dir:
        _global_loader = EnhancedSimpleLoader(root_dir)
    return _global_loader


def quick_load(dataset_name: str, root_dir: str = "./data", **kwargs) -> Optional[Dict[str, Any]]:
    """Quick dataset loading function."""
    loader = get_enhanced_loader(root_dir)
    return loader.load_dataset(dataset_name, **kwargs)


def list_datasets(root_dir: str = "./data") -> List[str]:
    """List available datasets."""
    loader = get_enhanced_loader(root_dir)
    return loader.list_available_datasets()