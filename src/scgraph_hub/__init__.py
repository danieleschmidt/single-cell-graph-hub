"""Single-Cell Graph Hub: Graph Neural Networks for Single-Cell Omics."""

__version__ = "0.1.0"

# Handle optional dependencies gracefully
try:
    from .dataset import SCGraphDataset
    from .models import BaseGNN, CellGraphGNN, CellGraphSAGE, SpatialGAT, create_model, MODEL_REGISTRY
    from .preprocessing import PreprocessingPipeline, GraphConstructor
    from .data_manager import DataManager, get_data_manager, load_dataset, create_dataloader
    from .database import get_dataset_repository, get_cache_manager
    _CORE_AVAILABLE = True
except ImportError as e:
    _CORE_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# Catalog doesn't require heavy dependencies
from .catalog import DatasetCatalog, get_default_catalog

# Simple dataset for basic functionality
from .simple_dataset import SimpleSCGraphDataset, SimpleSCGraphData

# Utility functions
from .utils import check_dependencies, validate_dataset_config

# Export available classes - basic functionality
__all__ = [
    "DatasetCatalog", 
    "get_default_catalog",
    "SimpleSCGraphDataset",
    "SimpleSCGraphData",
    "check_dependencies", 
    "validate_dataset_config"
]

# Export core functionality if available
if _CORE_AVAILABLE:
    __all__.extend([
        "SCGraphDataset",
        "BaseGNN", "CellGraphGNN", "CellGraphSAGE", "SpatialGAT", 
        "create_model", "MODEL_REGISTRY",
        "PreprocessingPipeline", "GraphConstructor",
        "DataManager", "get_data_manager", "load_dataset", "create_dataloader",
        "get_dataset_repository", "get_cache_manager"
    ])
else:
    # Create placeholder functions for missing dependencies
    def _missing_dependency_error(*args, **kwargs):
        """Placeholder for functions when dependencies are missing."""
        raise ImportError(
            f"Core functionality requires additional dependencies. "
            f"Install with: pip install single-cell-graph-hub[full]\n"
            f"Original error: {_IMPORT_ERROR}"
        )
    
    # Assign placeholders
    SCGraphDataset = _missing_dependency_error
    BaseGNN = _missing_dependency_error
    CellGraphGNN = _missing_dependency_error
    CellGraphSAGE = _missing_dependency_error
    SpatialGAT = _missing_dependency_error
    create_model = _missing_dependency_error
    MODEL_REGISTRY = {}
    PreprocessingPipeline = _missing_dependency_error
    GraphConstructor = _missing_dependency_error
    DataManager = _missing_dependency_error
    get_data_manager = _missing_dependency_error
    load_dataset = _missing_dependency_error
    create_dataloader = _missing_dependency_error
    get_dataset_repository = _missing_dependency_error
    get_cache_manager = _missing_dependency_error


# Convenience imports for common workflows
def quick_start(dataset_name: str, model_name: str = 'cellgnn', **kwargs):
    """Quick start function for common workflows.
    
    Args:
        dataset_name: Name of the dataset to load
        model_name: Name of the model to create
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (dataset, model, dataloader)
    """
    if not _CORE_AVAILABLE:
        raise ImportError("Quick start requires full installation")
    
    # Load dataset
    data = load_dataset(dataset_name, **kwargs.get('dataset_kwargs', {}))
    if data is None:
        raise ValueError(f"Could not load dataset: {dataset_name}")
    
    # Create model
    model_kwargs = kwargs.get('model_kwargs', {})
    if 'input_dim' not in model_kwargs:
        model_kwargs['input_dim'] = data.x.shape[1]
    if 'output_dim' not in model_kwargs and hasattr(data, 'y'):
        model_kwargs['output_dim'] = len(torch.unique(data.y))
    
    model = create_model(model_name, **model_kwargs)
    
    # Create dataloader
    dataloader = create_dataloader(dataset_name, **kwargs.get('dataloader_kwargs', {}))
    
    return data, model, dataloader


# Simple quick start for basic functionality
def simple_quick_start(dataset_name: str = "pbmc_10k", root: str = "./data", **kwargs):
    """Simple quick start function for basic functionality without heavy dependencies.
    
    Args:
        dataset_name: Name of the dataset to load
        root: Root directory for data storage
        **kwargs: Additional arguments
        
    Returns:
        SimpleSCGraphDataset instance
    """
    return SimpleSCGraphDataset(name=dataset_name, root=root, **kwargs)


# Add to exports
__all__.extend(["quick_start", "simple_quick_start"])