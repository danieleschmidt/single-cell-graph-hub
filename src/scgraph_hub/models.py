"""Comprehensive GNN models for single-cell graph analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool,
    MessagePassing, Sequential, Linear, ReLU, Dropout, BatchNorm1d,
    GINConv, TransformerConv, DiffGroupNorm
)
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Dict, List, Tuple, Union
import math


# Base class for extensibility
class BaseGNN(nn.Module):
    """Base class for all single-cell GNN models.
    
    Provides common functionality and interface for custom model development.
    Includes biological constraints and standardized interfaces.
    """
    
    def __init__(self):
        super().__init__()
        self._embedding_dim = None
        self._task_type = None
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get node embeddings (before final classification layer).
        
        Should be implemented by subclasses to return intermediate representations.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            **kwargs: Additional arguments
            
        Returns:
            Node embeddings
        """
        raise NotImplementedError("Subclasses must implement get_embeddings()")
    
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Make predictions on new data."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index, **kwargs)
    
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get prediction probabilities."""
        logits = self.predict(x, edge_index, **kwargs)
        return F.softmax(logits, dim=-1)
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        """Get attention weights if model supports it."""
        return None  # Override in attention-based models
    
    def num_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information and statistics."""
        return {
            'model_name': self.__class__.__name__,
            'num_parameters': self.num_parameters(),
            'embedding_dim': getattr(self, '_embedding_dim', None),
            'task_type': getattr(self, '_task_type', None)
        }
    
    def apply_biological_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Apply biological constraints to outputs.
        
        For example, ensuring non-negative gene expression predictions.
        """
        # Default: no constraints
        return x
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """Compute additional regularization losses.
        
        Can include biological priors, sparsity constraints, etc.
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)


class CellGraphGNN(BaseGNN):
    """Basic Graph Neural Network for single-cell analysis.
    
    This model provides a simple but effective architecture for cell-level
    tasks like cell type prediction. It uses Graph Convolutional Networks
    with residual connections and dropout for regularization.
    
    Args:
        input_dim: Number of input features (genes)
        hidden_dim: Hidden layer dimension
        output_dim: Number of output classes
        num_layers: Number of GNN layers
        dropout: Dropout probability
        
    Example:
        >>> model = CellGraphGNN(
        ...     input_dim=2000,
        ...     hidden_dim=128,
        ...     output_dim=8,
        ...     num_layers=3,
        ...     dropout=0.2
        ... )
        >>> out = model(batch.x, batch.edge_index, batch.batch)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self._embedding_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes]
            
        Returns:
            Node-level predictions [num_nodes, output_dim]
        """
        # Get embeddings
        embeddings = self.get_embeddings(x, edge_index)
        
        # Output projection
        x = self.output_proj(embeddings)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get embeddings before final classification layer."""
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GNN layers with residual connections
        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if x.shape == residual.shape:
                x = x + residual
        
        return x


class CellGraphSAGE(BaseGNN):
    """GraphSAGE model for large-scale single-cell data.
    
    GraphSAGE is particularly suitable for large datasets as it uses
    sampling during training to scale to millions of cells.
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        aggregator: Aggregation method ('mean', 'max', 'lstm')
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256, 128],
        aggregator: str = "mean",
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        self.dropout = dropout
        self.use_batch_norm = batch_norm
        self._embedding_dim = hidden_dims[-1]
        
        # Build layer dimensions
        dims = [input_dim] + hidden_dims
        
        # SAGE layers
        self.convs = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.convs.append(SAGEConv(dims[i], dims[i+1], aggr=aggregator))
        
        # Batch normalization
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(dims[i+1]))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.get_embeddings(x, edge_index)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get embeddings from SAGE layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class SpatialGAT(BaseGNN):
    """Graph Attention Network for spatial single-cell data.
    
    This model incorporates spatial information through edge attributes
    and uses attention mechanisms to weight neighboring cells.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension
        num_heads: Number of attention heads
        spatial_dim: Dimension of spatial coordinates
        use_edge_attr: Whether to use edge attributes
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        spatial_dim: int = 2,
        use_edge_attr: bool = True,
    ):
        super().__init__()
        
        self.use_edge_attr = use_edge_attr
        self._embedding_dim = hidden_dim
        
        # Spatial embedding
        if use_edge_attr:
            self.spatial_embed = nn.Linear(spatial_dim, hidden_dim // 4)
        
        # GAT layers
        self.conv1 = GATConv(
            input_dim, 
            hidden_dim, 
            heads=num_heads, 
            edge_dim=hidden_dim // 4 if use_edge_attr else None,
            dropout=0.1
        )
        self.conv2 = GATConv(
            hidden_dim * num_heads, 
            hidden_dim, 
            heads=1, 
            edge_dim=hidden_dim // 4 if use_edge_attr else None,
            dropout=0.1
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        return self.get_embeddings(x, edge_index, edge_attr=edge_attr)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, 
                      edge_attr: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get embeddings from GAT layers."""
        # Process edge attributes (spatial coordinates)
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.spatial_embed(edge_attr)
            edge_attr = F.relu(edge_attr)
        
        # First GAT layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index, edge_attr)
        
        return x


class HierarchicalGNN(BaseGNN):
    """Hierarchical GNN for multi-scale single-cell analysis.
    
    This model processes information at multiple scales:
    cell -> cell type -> tissue, using different pooling operations.
    
    Args:
        level_dims: Dictionary mapping levels to feature dimensions
        hidden_dim: Hidden layer dimension
        pooling: Pooling method ('mean', 'max', 'diffpool')
    """
    
    def __init__(
        self,
        level_dims: dict = {'cell': 2000, 'type': 100, 'tissue': 20},
        hidden_dim: int = 256,
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.level_dims = level_dims
        self.pooling = pooling
        self._embedding_dim = hidden_dim
        
        # Cell-level processing
        self.cell_conv = GCNConv(level_dims['cell'], hidden_dim)
        
        # Type-level processing
        self.type_conv = GCNConv(hidden_dim, hidden_dim)
        
        # Tissue-level processing
        self.tissue_conv = GCNConv(hidden_dim, level_dims['tissue'])
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        cell_type_batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through hierarchy."""
        return self.get_embeddings(x, edge_index, batch=batch, cell_type_batch=cell_type_batch)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, 
                      batch: Optional[torch.Tensor] = None,
                      cell_type_batch: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get hierarchical embeddings."""
        # Cell level
        x = self.cell_conv(x, edge_index)
        x = F.relu(x)
        
        # Pool to cell type level if needed
        if cell_type_batch is not None:
            if self.pooling == 'mean':
                x = global_mean_pool(x, cell_type_batch)
        
        return x


class CellGraphTransformer(BaseGNN):
    """Graph Transformer for single-cell data.
    
    Uses transformer-style attention mechanisms adapted for graphs
    with positional encodings based on graph structure.
    
    Args:
        input_dim: Number of input features
        model_dim: Model dimension
        num_heads: Number of attention heads  
        num_layers: Number of transformer layers
        positional_encoding: Type of positional encoding
    """
    
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        positional_encoding: str = 'laplacian',
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.positional_encoding = positional_encoding
        self._embedding_dim = model_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Positional encoding
        if positional_encoding == 'laplacian':
            self.pos_encoding = nn.Linear(model_dim, model_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.get_embeddings(x, edge_index)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get transformer embeddings."""
        # Project inputs
        x = self.input_proj(x)
        
        # Add positional encoding (simplified)
        if self.positional_encoding == 'laplacian':
            # In practice, would compute Laplacian eigenvectors
            pos_enc = torch.randn_like(x) * 0.1  # Placeholder
            x = x + self.pos_encoding(pos_enc)
        
        # Reshape for transformer (add sequence dimension)
        x = x.unsqueeze(0)  # [1, num_nodes, model_dim]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Remove sequence dimension
        x = x.squeeze(0)  # [num_nodes, model_dim]
        
        return x


class BiologicalMessagePassing(MessagePassing):
    """Message passing layer with biological constraints.
    
    Incorporates biological knowledge like gene regulatory networks,
    protein-protein interactions, or pathway information.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 biological_prior: Optional[torch.Tensor] = None,
                 aggr: str = 'add'):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Learnable transformation
        self.lin = nn.Linear(in_channels, out_channels)
        
        # Biological prior as edge weights
        self.biological_prior = biological_prior
        
        # Attention mechanism for biological relevance
        self.bio_attention = nn.Linear(in_channels * 2, 1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Start propagating messages
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_index_i: torch.Tensor) -> torch.Tensor:
        # Compute biological attention
        bio_score = self.bio_attention(torch.cat([x_i, x_j], dim=-1))
        bio_weight = torch.sigmoid(bio_score)
        
        # Apply biological prior if available
        if self.biological_prior is not None:
            # Assume biological_prior is an adjacency matrix
            prior_weight = self.biological_prior[edge_index_i]
            bio_weight = bio_weight * prior_weight.unsqueeze(-1)
        
        # Transform and weight messages
        message = self.lin(x_j) * bio_weight
        
        return message


class CellGraphGIN(BaseGNN):
    """Graph Isomorphism Network adapted for single-cell data.
    
    GIN is particularly powerful for distinguishing different cell states
    and capturing subtle biological differences.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout: float = 0.2, 
                 eps: float = 0.0, train_eps: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self._embedding_dim = hidden_dim
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                nn_seq = Sequential(
                    Linear(input_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim)
                )
            else:
                nn_seq = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(), 
                    Linear(hidden_dim, hidden_dim)
                )
            
            self.convs.append(GINConv(nn_seq, eps=eps, train_eps=train_eps))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Output projection
        self.classifier = Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get embeddings
        embeddings = self.get_embeddings(x, edge_index)
        
        # Classification
        return self.classifier(embeddings)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


# Model Registry for easy access
MODEL_REGISTRY = {
    'cellgnn': CellGraphGNN,
    'cellsage': CellGraphSAGE,
    'spatialgat': SpatialGAT,
    'hierarchical': HierarchicalGNN,
    'transformer': CellGraphTransformer,
    'gin': CellGraphGIN
}


def create_model(model_name: str, **kwargs) -> BaseGNN:
    """Factory function to create models by name.
    
    Args:
        model_name: Name of the model
        **kwargs: Model-specific arguments
        
    Returns:
        Instantiated model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def get_model_recommendations(dataset_info: Dict[str, any]) -> List[str]:
    """Get recommended models based on dataset characteristics.
    
    Args:
        dataset_info: Dataset metadata
        
    Returns:
        List of recommended model names
    """
    recommendations = []
    
    n_cells = dataset_info.get('n_cells', 0)
    n_genes = dataset_info.get('n_genes', 0)
    modality = dataset_info.get('modality', 'scRNA-seq')
    
    # Large datasets
    if n_cells > 50000:
        recommendations.extend(['cellsage'])
    
    # Spatial data
    if 'spatial' in modality.lower():
        recommendations.append('spatialgat')
    
    # High-dimensional data
    if n_genes > 5000:
        recommendations.extend(['gin', 'transformer'])
    
    # Default recommendations
    if not recommendations:
        recommendations.extend(['cellgnn', 'cellsage', 'gin'])
    
    return recommendations


# Training utilities
class ModelTrainer:
    """Utility class for training single-cell GNN models."""
    
    def __init__(self, model: BaseGNN, optimizer: torch.optim.Optimizer, 
                 device: str = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        self.model.to(device)
    
    def train_epoch(self, data_loader, criterion) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.batch)
            
            # Compute loss (only on training nodes)
            if hasattr(batch, 'train_mask'):
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            else:
                loss = criterion(out, batch.y)
            
            # Add regularization
            reg_loss = self.model.compute_regularization_loss()
            total_loss_val = loss + reg_loss
            
            # Backward pass
            total_loss_val.backward()
            self.optimizer.step()
            
            total_loss += total_loss_val.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def evaluate(self, data_loader, criterion) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Compute loss
                if hasattr(batch, 'val_mask'):
                    loss = criterion(out[batch.val_mask], batch.y[batch.val_mask])
                    pred = out[batch.val_mask].argmax(dim=1)
                    correct += (pred == batch.y[batch.val_mask]).sum().item()
                    total += batch.val_mask.sum().item()
                else:
                    loss = criterion(out, batch.y)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
                
                total_loss += loss.item()
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        
        return avg_loss, accuracy