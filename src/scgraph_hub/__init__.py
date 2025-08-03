"""Single-Cell Graph Hub: Graph Neural Networks for Single-Cell Omics."""

__version__ = "0.1.0"

# Handle optional dependencies gracefully
try:
    from .dataset import SCGraphDataset
    _DATASET_AVAILABLE = True
except ImportError as e:
    _DATASET_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# Catalog doesn't require heavy dependencies
from .catalog import DatasetCatalog

# Utility functions
from .utils import check_dependencies, validate_dataset_config

# Export available classes
__all__ = ["DatasetCatalog", "check_dependencies", "validate_dataset_config"]

if _DATASET_AVAILABLE:
    __all__.extend(["SCGraphDataset"])
else:
    def SCGraphDataset(*args, **kwargs):
        """Placeholder for SCGraphDataset when dependencies are missing."""
        raise ImportError(
            f"SCGraphDataset requires additional dependencies. "
            f"Install with: pip install single-cell-graph-hub[full]\n"
            f"Original error: {_IMPORT_ERROR}"
        )