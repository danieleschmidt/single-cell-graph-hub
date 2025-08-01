"""Basic tests for Single-Cell Graph Hub."""

import pytest
from scgraph_hub import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_import():
    """Test basic imports work."""
    from scgraph_hub import SCGraphDataset, DatasetCatalog
    
    assert SCGraphDataset is not None
    assert DatasetCatalog is not None