"""Advanced visualization tools for single-cell graph data."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualizations will be disabled.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Graph visualizations will be limited.")

logger = logging.getLogger(__name__)


class InteractiveVisualizer:
    """Interactive visualization tools using Plotly."""
    
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualizations. Install with: pip install plotly")
    
    def create_graph_explorer(self,
                             graph_data: Any,
                             node_colors: Optional[np.ndarray] = None,
                             node_sizes: Optional[np.ndarray] = None,
                             edge_colors: Optional[np.ndarray] = None,
                             layout: str = 'spring',
                             title: str = "Interactive Cell Graph") -> go.Figure:
        """Create interactive graph visualization.
        
        Args:
            graph_data: PyTorch Geometric Data object
            node_colors: Node color values
            node_sizes: Node size values
            edge_colors: Edge color values
            layout: Graph layout algorithm
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for graph visualization")
        
        # Convert to NetworkX graph
        edge_list = graph_data.edge_index.t().cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_list)
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=node_sizes if node_sizes is not None else 10,
                color=node_colors if node_colors is not None else 'lightblue',
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02
                )
            )
        )
        
        # Add hover text
        node_adjacencies = []
        node_text = []
        for node in G.nodes():
            adjacencies = list(G.neighbors(node))
            node_adjacencies.append(len(adjacencies))
            node_text.append(f'Cell {node}<br>Connections: {len(adjacencies)}')
        
        node_trace.text = node_text
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Interactive cell graph. Hover over nodes for details.",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def plot_3d_graph(self,
                     embeddings: np.ndarray,
                     edges: Optional[np.ndarray] = None,
                     colors: Optional[np.ndarray] = None,
                     size: int = 2,
                     edge_alpha: float = 0.1,
                     title: str = "3D Cell Graph") -> go.Figure:
        """Create 3D scatter plot with graph overlay.
        
        Args:
            embeddings: 3D coordinates [n_points, 3]
            edges: Edge connectivity [2, n_edges]
            colors: Point colors
            size: Point size
            edge_alpha: Edge transparency
            title: Plot title
            
        Returns:
            Plotly 3D figure
        """
        # Create 3D scatter plot
        scatter = go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=colors if colors is not None else 'lightblue',
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'Cell {i}' for i in range(len(embeddings))],
            hovertemplate='Cell %{text}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}'
        )
        
        data = [scatter]
        
        # Add edges if provided
        if edges is not None:
            edge_x, edge_y, edge_z = [], [], []
            for i in range(edges.shape[1]):
                source, target = edges[0, i], edges[1, i]
                edge_x.extend([embeddings[source, 0], embeddings[target, 0], None])
                edge_y.extend([embeddings[source, 1], embeddings[target, 1], None])
                edge_z.extend([embeddings[source, 2], embeddings[target, 2], None])
            
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color=f'rgba(125,125,125,{edge_alpha})', width=1),
                hoverinfo='none',
                showlegend=False
            )
            data.append(edge_trace)
        
        fig = go.Figure(
            data=data,
            layout=go.Layout(
                title=title,
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
        )
        
        return fig
    
    def add_tooltip(self, fig: go.Figure, tooltip_data: List[str]):
        """Add custom tooltip information to figure."""
        # This would modify the hover template of existing traces
        for trace in fig.data:
            if hasattr(trace, 'hovertemplate'):
                # Extend existing hover template
                additional_info = '<br>'.join([f'{key}: %{{customdata[{i}]}}' for i, key in enumerate(tooltip_data)])
                trace.hovertemplate += f'<br>{additional_info}'
    
    def save(self, fig: go.Figure, filename: str, format: str = 'html'):
        """Save interactive figure.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: Output format ('html', 'png', 'pdf')
        """
        if format == 'html':
            pyo.plot(fig, filename=filename, auto_open=False)
        else:
            fig.write_image(filename)
        
        logger.info(f"Interactive visualization saved to {filename}")


class PublicationFigures:
    """Publication-quality static visualizations."""
    
    def __init__(self, style: str = 'nature'):
        """Initialize with publication style.
        
        Args:
            style: Publication style ('nature', 'science', 'cell')
        """
        self.style = style
        self._set_style()
    
    def _set_style(self):
        """Set matplotlib style for publication."""
        if self.style == 'nature':
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'font.size': 8,
                'axes.titlesize': 10,
                'axes.labelsize': 8,
                'xtick.labelsize': 7,
                'ytick.labelsize': 7,
                'legend.fontsize': 7,
                'figure.titlesize': 12,
                'font.family': 'Arial'
            })
        elif self.style == 'science':
            plt.style.use('seaborn-v0_8-white')
            plt.rcParams.update({
                'font.size': 9,
                'font.family': 'Arial'
            })
    
    def create_figure(self, rows: int, cols: int, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create multi-panel figure.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            figsize: Figure size in inches
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows * cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.axes = axes
        return fig
    
    def plot_dataset_summary(self,
                            ax: plt.Axes,
                            dataset_stats: Dict[str, Any],
                            plot_type: str = 'violin') -> plt.Axes:
        """Plot dataset summary statistics.
        
        Args:
            ax: Matplotlib axes
            dataset_stats: Dataset statistics
            plot_type: Type of plot ('violin', 'box', 'bar')
            
        Returns:
            Modified axes
        """
        # Example implementation for cell counts across datasets
        if 'cell_counts' in dataset_stats:
            cell_counts = dataset_stats['cell_counts']
            dataset_names = dataset_stats.get('dataset_names', list(range(len(cell_counts))))
            
            if plot_type == 'violin':
                parts = ax.violinplot(cell_counts, positions=range(len(cell_counts)))
                ax.set_xticks(range(len(dataset_names)))
                ax.set_xticklabels(dataset_names, rotation=45, ha='right')
            elif plot_type == 'bar':
                ax.bar(range(len(cell_counts)), cell_counts)
                ax.set_xticks(range(len(dataset_names)))
                ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        
        ax.set_ylabel('Number of Cells')
        ax.set_title('Dataset Overview')
        
        return ax
    
    def plot_graph_statistics(self,
                             ax: plt.Axes,
                             graph_properties: Dict[str, List[float]],
                             metrics: List[str] = ['degree', 'clustering', 'components']) -> plt.Axes:
        """Plot graph statistics.
        
        Args:
            ax: Matplotlib axes
            graph_properties: Graph property values
            metrics: Metrics to plot
            
        Returns:
            Modified axes
        """
        x_pos = np.arange(len(metrics))
        
        for i, metric in enumerate(metrics):
            if metric in graph_properties:
                values = graph_properties[metric]
                ax.bar(i, np.mean(values), yerr=np.std(values), capsize=5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Metric Value')
        ax.set_title('Graph Properties')
        
        return ax
    
    def plot_model_comparison(self,
                             ax: plt.Axes,
                             benchmark_results: Dict[str, Dict[str, float]],
                             models: List[str],
                             metric: str = 'f1_score') -> plt.Axes:
        """Plot model comparison.
        
        Args:
            ax: Matplotlib axes
            benchmark_results: Results from benchmarking
            models: Model names to compare
            metric: Metric to compare
            
        Returns:
            Modified axes
        """
        scores = []
        errors = []
        
        for model in models:
            if model in benchmark_results and metric in benchmark_results[model]:
                result = benchmark_results[model][metric]
                if isinstance(result, dict):
                    scores.append(result.get('mean', 0))
                    errors.append(result.get('std', 0))
                else:
                    scores.append(result)
                    errors.append(0)
            else:
                scores.append(0)
                errors.append(0)
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, scores, yerr=errors, capsize=5)
        
        # Color best performing model
        best_idx = np.argmax(scores)
        bars[best_idx].set_color('red')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title('Model Comparison')
        
        return ax
    
    def plot_embedding_2d(self,
                         ax: plt.Axes,
                         embeddings: np.ndarray,
                         colors: Optional[np.ndarray] = None,
                         color_map: str = 'tab10',
                         point_size: int = 1,
                         alpha: float = 0.7,
                         title: str = "2D Embedding") -> plt.Axes:
        """Plot 2D embeddings.
        
        Args:
            ax: Matplotlib axes
            embeddings: 2D coordinates [n_points, 2]
            colors: Point colors/labels
            color_map: Colormap name
            point_size: Point size
            alpha: Point transparency
            title: Plot title
            
        Returns:
            Modified axes
        """
        scatter = ax.scatter(
            embeddings[:, 0], 
            embeddings[:, 1],
            c=colors,
            cmap=color_map,
            s=point_size,
            alpha=alpha
        )
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        
        # Add colorbar if colors are provided
        if colors is not None:
            plt.colorbar(scatter, ax=ax)
        
        return ax
    
    def plot_trajectory(self,
                       ax: plt.Axes,
                       embeddings: np.ndarray,
                       pseudotime: np.ndarray,
                       trajectory_edges: Optional[np.ndarray] = None,
                       color_map: str = 'viridis',
                       title: str = "Cell Trajectory") -> plt.Axes:
        """Plot cell trajectory in 2D.
        
        Args:
            ax: Matplotlib axes
            embeddings: 2D coordinates
            pseudotime: Pseudotime values for coloring
            trajectory_edges: Trajectory backbone edges
            color_map: Colormap for pseudotime
            title: Plot title
            
        Returns:
            Modified axes
        """
        # Plot cells colored by pseudotime
        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=pseudotime,
            cmap=color_map,
            s=10,
            alpha=0.7
        )
        
        # Plot trajectory backbone if provided
        if trajectory_edges is not None:
            for edge in trajectory_edges:
                start, end = edge
                ax.plot(
                    [embeddings[start, 0], embeddings[end, 0]],
                    [embeddings[start, 1], embeddings[end, 1]],
                    'k-', alpha=0.5, linewidth=1
                )
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pseudotime')
        
        return ax
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
        """Save publication-quality figure.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution in DPI
            bbox_inches: Bounding box adjustment
        """
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Publication figure saved to {filename}")


class DashboardCreator:
    """Create interactive dashboards for data exploration."""
    
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for dashboard creation")
    
    def create_dataset_dashboard(self,
                                dataset_info: Dict[str, Any],
                                embeddings: np.ndarray,
                                metadata: pd.DataFrame) -> go.Figure:
        """Create comprehensive dataset exploration dashboard.
        
        Args:
            dataset_info: Dataset metadata
            embeddings: Cell embeddings for visualization
            metadata: Cell metadata
            
        Returns:
            Plotly dashboard figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cell Embedding', 'QC Metrics', 'Cell Type Distribution', 'Gene Expression'),
            specs=[[{'type': 'scatter'}, {'type': 'violin'}],
                   [{'type': 'bar'}, {'type': 'heatmap'}]]
        )
        
        # Cell embedding plot
        fig.add_trace(
            go.Scatter(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                mode='markers',
                marker=dict(color=metadata.get('cell_type', 'blue'), size=3),
                name='Cells'
            ),
            row=1, col=1
        )
        
        # QC metrics violin plot
        if 'total_counts' in metadata.columns:
            fig.add_trace(
                go.Violin(
                    y=metadata['total_counts'],
                    name='Total Counts'
                ),
                row=1, col=2
            )
        
        # Cell type distribution
        if 'cell_type' in metadata.columns:
            cell_type_counts = metadata['cell_type'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=cell_type_counts.index,
                    y=cell_type_counts.values,
                    name='Cell Types'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Dataset Dashboard: {dataset_info.get('name', 'Unknown')}",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_benchmark_dashboard(self, benchmark_results: Dict[str, Any]) -> go.Figure:
        """Create benchmark results dashboard.
        
        Args:
            benchmark_results: Results from benchmark runs
            
        Returns:
            Plotly dashboard figure
        """
        # Extract data for visualization
        models = list(benchmark_results['results'].keys())
        datasets = list(benchmark_results['results'][models[0]].keys()) if models else []
        
        # Create heatmap of model performance
        performance_matrix = []
        for model in models:
            row = []
            for dataset in datasets:
                # Get accuracy or first available metric
                task_results = benchmark_results['results'][model][dataset]
                if task_results:
                    first_task = list(task_results.keys())[0]
                    metrics = task_results[first_task].get('aggregated', {}).get('metrics', {})
                    score = metrics.get('accuracy', {}).get('mean', 0) if 'accuracy' in metrics else 0
                    row.append(score)
                else:
                    row.append(0)
            performance_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix,
            x=datasets,
            y=models,
            colorscale='Viridis',
            hovertemplate='Model: %{y}<br>Dataset: %{x}<br>Score: %{z:.3f}'
        ))
        
        fig.update_layout(
            title="Model Performance Heatmap",
            xaxis_title="Datasets",
            yaxis_title="Models"
        )
        
        return fig


# Convenience functions
def quick_scatter_plot(embeddings: np.ndarray,
                      colors: Optional[np.ndarray] = None,
                      title: str = "Cell Embedding",
                      save_path: Optional[str] = None) -> plt.Figure:
    """Create quick scatter plot of embeddings."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, s=10, alpha=0.7)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    
    if colors is not None:
        plt.colorbar(scatter)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(training_losses: List[float],
                        validation_losses: Optional[List[float]] = None,
                        title: str = "Training Curves",
                        save_path: Optional[str] = None) -> plt.Figure:
    """Plot training and validation curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(training_losses) + 1)
    ax.plot(epochs, training_losses, 'b-', label='Training Loss')
    
    if validation_losses:
        ax.plot(epochs, validation_losses, 'r-', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
