"""Basic model interfaces for Generation 1 - Lightweight without heavy dependencies."""

import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Basic model configuration."""
    name: str
    input_dim: int
    output_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'activation': self.activation,
            'metadata': self.metadata
        }


class BaseModelInterface(ABC):
    """Basic model interface for Generation 1."""
    
    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._trained = False
        self._metrics = {}
    
    @abstractmethod
    def forward(self, x: Any, edge_index: Any = None, **kwargs) -> Any:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def train_step(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        pass
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model on data."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'config': self.config.to_dict(),
            'trained': self._trained,
            'metrics': self._metrics,
            'parameters': self.count_parameters()
        }
    
    def count_parameters(self) -> int:
        """Count model parameters."""
        # Basic implementation - subclasses should override
        return self.config.hidden_dim * self.config.num_layers
    
    def save_config(self, path: str) -> bool:
        """Save model configuration."""
        try:
            with open(path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            self.logger.info(f"Saved config to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False
    
    @classmethod
    def load_config(cls, path: str) -> Optional[ModelConfig]:
        """Load model configuration."""
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            return ModelConfig(**config_dict)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load config: {e}")
            return None


class DummyGNNModel(BaseModelInterface):
    """Dummy GNN model for testing without PyTorch dependencies."""
    
    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self._weights = {}
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize dummy weights."""
        try:
            import numpy as np
            # Create dummy weight matrices
            self._weights['input'] = np.random.randn(self.config.input_dim, self.config.hidden_dim) * 0.1
            for i in range(self.config.num_layers - 1):
                self._weights[f'hidden_{i}'] = np.random.randn(self.config.hidden_dim, self.config.hidden_dim) * 0.1
            self._weights['output'] = np.random.randn(self.config.hidden_dim, self.config.output_dim) * 0.1
            self.logger.info("Initialized dummy weights with NumPy")
        except ImportError:
            # Fallback without NumPy
            self._weights = {
                'input': [[0.1] * self.config.hidden_dim for _ in range(self.config.input_dim)],
                'output': [[0.1] * self.config.output_dim for _ in range(self.config.hidden_dim)]
            }
            self.logger.info("Initialized dummy weights without NumPy")
    
    def forward(self, x: Any, edge_index: Any = None, **kwargs) -> Any:
        """Dummy forward pass."""
        try:
            import numpy as np
            if isinstance(x, list):
                x = np.array(x)
            
            # Simple linear transformation
            h = np.dot(x, self._weights['input'])
            
            # Apply activation (dummy)
            if self.config.activation == "relu":
                h = np.maximum(0, h)
            
            # Output layer
            out = np.dot(h, self._weights['output'])
            
            return out
        except ImportError:
            # Simple fallback
            num_nodes = len(x) if isinstance(x, list) else 1
            return [[0.5] * self.config.output_dim for _ in range(num_nodes)]
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            return None
    
    def train_step(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Dummy training step."""
        try:
            x = data.get('x', [])
            y = data.get('y', [])
            
            if not x or not y:
                return {'loss': float('inf'), 'accuracy': 0.0}
            
            # Forward pass
            predictions = self.forward(x, data.get('edge_index'))
            
            if predictions is None:
                return {'loss': float('inf'), 'accuracy': 0.0}
            
            # Dummy loss calculation
            import random
            loss = random.uniform(0.1, 1.0)
            accuracy = random.uniform(0.5, 0.9) if self._trained else random.uniform(0.1, 0.6)
            
            # Simple "training" effect
            self._trained = True
            
            metrics = {'loss': loss, 'accuracy': accuracy}
            self._metrics.update(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
            return {'loss': float('inf'), 'accuracy': 0.0}
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Dummy evaluation."""
        try:
            x = data.get('x', [])
            y = data.get('y', [])
            
            if not x or not y:
                return {'accuracy': 0.0, 'f1_score': 0.0}
            
            # Dummy predictions
            predictions = self.forward(x, data.get('edge_index'))
            
            if predictions is None:
                return {'accuracy': 0.0, 'f1_score': 0.0}
            
            # Dummy metrics
            import random
            accuracy = random.uniform(0.7, 0.95) if self._trained else random.uniform(0.3, 0.6)
            f1_score = accuracy * random.uniform(0.9, 1.0)
            
            return {'accuracy': accuracy, 'f1_score': f1_score}
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {'accuracy': 0.0, 'f1_score': 0.0}
    
    def count_parameters(self) -> int:
        """Count parameters."""
        total = 0
        try:
            import numpy as np
            for weight_matrix in self._weights.values():
                if isinstance(weight_matrix, np.ndarray):
                    total += weight_matrix.size
                elif isinstance(weight_matrix, list):
                    total += len(weight_matrix) * len(weight_matrix[0]) if weight_matrix else 0
        except ImportError:
            # Rough estimate without NumPy
            total = (self.config.input_dim * self.config.hidden_dim + 
                    self.config.hidden_dim * self.config.output_dim)
        
        return total


class ModelRegistry:
    """Simple model registry for Generation 1."""
    
    def __init__(self):
        self._models = {}
        self.logger = logging.getLogger(__name__)
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default models."""
        self._models['dummy_gnn'] = DummyGNNModel
        self.logger.info("Registered default models")
    
    def register(self, name: str, model_class: type):
        """Register a model class."""
        self._models[name] = model_class
        self.logger.info(f"Registered model: {name}")
    
    def create_model(self, name: str, config: ModelConfig) -> Optional[BaseModelInterface]:
        """Create a model instance."""
        if name not in self._models:
            self.logger.error(f"Unknown model: {name}")
            return None
        
        try:
            model_class = self._models[name]
            return model_class(config, self.logger)
        except Exception as e:
            self.logger.error(f"Failed to create model {name}: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """List available models."""
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        if name not in self._models:
            return None
        
        model_class = self._models[name]
        return {
            'name': name,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'doc': model_class.__doc__
        }


# Global registry
_global_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _global_registry


def create_model(name: str, input_dim: int, output_dim: int, **kwargs) -> Optional[BaseModelInterface]:
    """Quick model creation function."""
    config = ModelConfig(
        name=name,
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs
    )
    return _global_registry.create_model(name, config)


def list_available_models() -> List[str]:
    """List all available models."""
    return _global_registry.list_models()


# Training utilities
class SimpleTrainer:
    """Simple trainer for basic models."""
    
    def __init__(self, model: BaseModelInterface, logger: Optional[logging.Logger] = None):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.history = []
    
    def train(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]] = None, 
              epochs: int = 10) -> Dict[str, List[float]]:
        """Train the model."""
        self.logger.info(f"Starting training for {epochs} epochs")
        
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training step
            train_metrics = self.model.train_step(train_data)
            train_losses.append(train_metrics.get('loss', 0.0))
            train_accuracies.append(train_metrics.get('accuracy', 0.0))
            
            # Validation step
            if val_data:
                val_metrics = self.model.evaluate(val_data)
                val_accuracies.append(val_metrics.get('accuracy', 0.0))
            
            if epoch % max(1, epochs // 10) == 0:
                self.logger.info(f"Epoch {epoch}: Loss={train_metrics.get('loss', 0.0):.4f}, "
                               f"Train Acc={train_metrics.get('accuracy', 0.0):.4f}")
        
        history = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_accuracy': val_accuracies
        }
        
        self.history.append(history)
        self.logger.info("Training completed")
        
        return history
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the model."""
        self.logger.info("Evaluating model")
        return self.model.evaluate(test_data)


def quick_train(model_name: str, train_data: Dict[str, Any], input_dim: int, output_dim: int,
                epochs: int = 10, **kwargs) -> Tuple[BaseModelInterface, Dict[str, Any]]:
    """Quick training function."""
    model = create_model(model_name, input_dim, output_dim, **kwargs)
    if model is None:
        raise ValueError(f"Could not create model: {model_name}")
    
    trainer = SimpleTrainer(model)
    history = trainer.train(train_data, epochs=epochs)
    
    return model, history