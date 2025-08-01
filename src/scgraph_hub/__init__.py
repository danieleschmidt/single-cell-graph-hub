"""Single-Cell Graph Hub: Graph Neural Networks for Single-Cell Omics."""

__version__ = "0.1.0"

from .dataset import SCGraphDataset
from .catalog import DatasetCatalog

__all__ = ["SCGraphDataset", "DatasetCatalog"]