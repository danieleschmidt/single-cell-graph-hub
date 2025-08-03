"""Basic GNN models for single-cell graph analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from typing import Optional


class CellGraphGNN(nn.Module):
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
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class CellGraphSAGE(nn.Module):
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
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class SpatialGAT(nn.Module):
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


class HierarchicalGNN(nn.Module):
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
        # Cell level
        x = self.cell_conv(x, edge_index)
        x = F.relu(x)
        
        # Pool to cell type level
        if cell_type_batch is not None:
            if self.pooling == 'mean':
                x = global_mean_pool(x, cell_type_batch)
            # Additional pooling methods could be added here
        
        # Type level (would need type-level edge_index)
        # x = self.type_conv(x, type_edge_index)
        # x = F.relu(x)
        
        # For now, just return cell-level features
        return x


class CellGraphTransformer(nn.Module):
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


# Base class for extensibility
class BaseGNN(nn.Module):
    """Base class for all single-cell GNN models.
    
    Provides common functionality and interface for custom model development.
    """
    
    def __init__(self):
        super().__init__()
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings (before final classification layer).
        
        Should be implemented by subclasses to return intermediate representations.
        """
        raise NotImplementedError("Subclasses must implement get_embeddings()")
    
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Make predictions on new data."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index)
    
    def num_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)