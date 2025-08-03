"""Dataset catalog for discovering and managing single-cell graph datasets."""

from typing import Dict, List, Optional, Any, Union
import json
import warnings
import os
import requests
from pathlib import Path
import hashlib
from urllib.parse import urljoin
import tempfile
import shutil


class DatasetCatalog:
    """Catalog for discovering and filtering available single-cell graph datasets.
    
    The catalog provides a unified interface for browsing available datasets,
    filtering by characteristics, and retrieving dataset metadata.
    
    Example:
        >>> catalog = DatasetCatalog()
        >>> all_datasets = catalog.list_datasets()
        >>> rna_datasets = catalog.filter(modality="scRNA-seq", organism="human")
        >>> info = catalog.get_info("pbmc_10k")
    """
    
    def __init__(self):
        """Initialize the dataset catalog."""
        self._datasets = self._load_catalog()
    
    def _load_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load the dataset catalog from metadata."""
        # Try to load from external sources first
        catalog = self._load_external_catalog()
        if catalog:
            return catalog
        
        # Fallback to built-in catalog with diverse datasets
        return {
            "pbmc_10k": {
                "name": "pbmc_10k",
                "description": "10k PBMCs from a healthy donor",
                "n_cells": 10000,
                "n_genes": 2000,
                "modality": "scRNA-seq",
                "organism": "human",
                "tissue": "blood",
                "n_cell_types": 8,
                "has_spatial": False,
                "graph_method": "knn",
                "tasks": ["cell_type_prediction", "gene_imputation"],
                "size_mb": 150,
                "citation": "Zheng et al., Nat Commun 2017",
                "checksum": "abc123def456...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/pbmc_10k.h5ad"
            },
            "tabula_muris": {
                "name": "tabula_muris",
                "description": "Mouse cell atlas with 20 organs",
                "n_cells": 100000,
                "n_genes": 23000,
                "modality": "scRNA-seq",
                "organism": "mouse",
                "tissue": "multi-organ",
                "tissues": ["brain", "heart", "lung", "liver", "kidney"],
                "n_cell_types": 120,
                "has_spatial": False,
                "graph_method": "knn",
                "tasks": ["cell_type_prediction", "trajectory_inference"],
                "size_mb": 2500,
                "citation": "Tabula Muris Consortium, Nature 2018",
                "checksum": "def456ghi789...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/tabula_muris.h5ad"
            },
            "brain_atlas": {
                "name": "brain_atlas",
                "description": "Human brain single-cell atlas",
                "n_cells": 75000,
                "n_genes": 3000,
                "modality": "snRNA-seq",
                "organism": "human",
                "tissue": "brain",
                "n_cell_types": 45,
                "has_spatial": False,
                "graph_method": "spatial_knn",
                "tasks": ["cell_type_prediction", "trajectory_inference"],
                "size_mb": 1800,
                "citation": "Lake et al., Nat Biotechnol 2018",
                "checksum": "ghi789jkl012...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/brain_atlas.h5ad"
            },
            "embryo_development": {
                "name": "embryo_development",
                "description": "Mouse embryonic development time course",
                "n_cells": 25000,
                "n_genes": 2500,
                "modality": "scRNA-seq",
                "organism": "mouse",
                "tissue": "embryo",
                "n_cell_types": 30,
                "has_spatial": False,
                "graph_method": "temporal_knn",
                "tasks": ["trajectory_inference", "cell_type_prediction"],
                "size_mb": 800,
                "citation": "Pijuan-Sala et al., Nat Cell Biol 2019",
                "checksum": "jkl012mno345...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/embryo_development.h5ad"
            },
            "spatial_heart": {
                "name": "spatial_heart",
                "description": "Spatial transcriptomics of human heart",
                "n_cells": 15000,
                "n_genes": 1800,
                "modality": "spatial_transcriptomics",
                "organism": "human", 
                "tissue": "heart",
                "n_cell_types": 12,
                "has_spatial": True,
                "graph_method": "spatial_radius",
                "tasks": ["cell_type_prediction", "spatial_domain_identification"],
                "size_mb": 600,
                "citation": "Tucker et al., Nature 2020",
                "checksum": "mno345pqr678...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/spatial_heart.h5ad"
            },
            "pancreas_integrated": {
                "name": "pancreas_integrated",
                "description": "Integrated pancreas datasets across technologies",
                "n_cells": 80000,
                "n_genes": 2200,
                "modality": "scRNA-seq",
                "organism": "human",
                "tissue": "pancreas",
                "n_cell_types": 15,
                "has_spatial": False,
                "graph_method": "batch_corrected_knn",
                "tasks": ["batch_correction", "cell_type_prediction"],
                "size_mb": 1200,
                "citation": "Luecken et al., Mol Syst Biol 2020",
                "checksum": "pqr678stu901...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/pancreas_integrated.h5ad"
            },
            "immune_covid": {
                "name": "immune_covid",
                "description": "Immune response to COVID-19",
                "n_cells": 45000,
                "n_genes": 2800,
                "modality": "scRNA-seq",
                "organism": "human",
                "tissue": "blood",
                "n_cell_types": 25,
                "has_spatial": False,
                "graph_method": "disease_aware_knn",
                "tasks": ["cell_type_prediction", "disease_classification"],
                "size_mb": 950,
                "citation": "Wilk et al., Nat Med 2020",
                "checksum": "stu901vwx234...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/immune_covid.h5ad"
            },
            "lung_atlas_10x": {
                "name": "lung_atlas_10x",
                "description": "Human lung cell atlas from 10x Genomics",
                "n_cells": 65000,
                "n_genes": 3500,
                "modality": "scRNA-seq",
                "organism": "human",
                "tissue": "lung",
                "n_cell_types": 32,
                "has_spatial": False,
                "graph_method": "adaptive_knn",
                "tasks": ["cell_type_prediction", "gene_imputation"],
                "size_mb": 1400,
                "citation": "Travaglini et al., Nature 2020",
                "checksum": "vwx234yza567...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/lung_atlas_10x.h5ad"
            },
            "retina_development": {
                "name": "retina_development",
                "description": "Mouse retinal development single-cell atlas",
                "n_cells": 35000,
                "n_genes": 2100,
                "modality": "scRNA-seq",
                "organism": "mouse",
                "tissue": "retina",
                "n_cell_types": 18,
                "has_spatial": False,
                "graph_method": "developmental_knn",
                "tasks": ["trajectory_inference", "cell_type_prediction"],
                "size_mb": 750,
                "citation": "Clark et al., Neuron 2019",
                "checksum": "yza567bcd890...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/retina_development.h5ad"
            },
            "liver_zonation": {
                "name": "liver_zonation",
                "description": "Spatial liver zonation with scRNA-seq",
                "n_cells": 20000,
                "n_genes": 1900,
                "modality": "scRNA-seq",
                "organism": "mouse",
                "tissue": "liver",
                "n_cell_types": 8,
                "has_spatial": True,
                "graph_method": "zonation_aware_graph",
                "tasks": ["spatial_domain_identification", "cell_type_prediction"],
                "size_mb": 400,
                "citation": "Halpern et al., Nature 2017",
                "checksum": "bcd890efg123...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/liver_zonation.h5ad"
            },
            "multiome_pbmc": {
                "name": "multiome_pbmc",
                "description": "Multiome (RNA + ATAC) PBMC dataset",
                "n_cells": 12000,
                "n_genes": 2500,
                "modality": "multiome",
                "organism": "human",
                "tissue": "blood",
                "n_cell_types": 10,
                "has_spatial": False,
                "graph_method": "multimodal_knn",
                "tasks": ["cell_type_prediction", "multimodal_integration"],
                "size_mb": 850,
                "citation": "10x Genomics, 2021",
                "checksum": "efg123hij456...",
                "url": "https://scgraphhub.s3.amazonaws.com/datasets/multiome_pbmc.h5ad"
            }
        }
    
    def _load_external_catalog(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Try to load catalog from external sources."""
        try:
            # Try to load from environment variable
            catalog_url = os.environ.get('SCGRAPH_CATALOG_URL')
            if catalog_url:
                response = requests.get(catalog_url, timeout=10)
                response.raise_for_status()
                return response.json()
            
            # Try to load from local file
            local_catalog_paths = [
                os.path.expanduser('~/.scgraph_hub/catalog.json'),
                '/etc/scgraph_hub/catalog.json',
                './catalog.json'
            ]
            
            for path in local_catalog_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        return json.load(f)
        
        except Exception as e:
            warnings.warn(f"Failed to load external catalog: {e}")
        
        return None
    
    def list_datasets(self) -> List[str]:
        """List all available dataset names.
        
        Returns:
            List of dataset names
        """
        return list(self._datasets.keys())
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a specific dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dictionary containing dataset metadata
            
        Raises:
            KeyError: If dataset is not found
        """
        if name not in self._datasets:
            available = ", ".join(self.list_datasets())
            raise KeyError(f"Dataset '{name}' not found. Available datasets: {available}")
        
        return self._datasets[name].copy()
    
    def filter(
        self,
        modality: Optional[str] = None,
        organism: Optional[str] = None,
        tissue: Optional[str] = None,
        min_cells: Optional[int] = None,
        max_cells: Optional[int] = None,
        min_genes: Optional[int] = None,
        max_genes: Optional[int] = None,
        has_spatial: Optional[bool] = None,
        tasks: Optional[List[str]] = None,
    ) -> List[str]:
        """Filter datasets by characteristics.
        
        Args:
            modality: Filter by data modality (e.g., 'scRNA-seq', 'scATAC-seq')
            organism: Filter by organism (e.g., 'human', 'mouse')
            tissue: Filter by tissue type
            min_cells: Minimum number of cells
            max_cells: Maximum number of cells
            min_genes: Minimum number of genes
            max_genes: Maximum number of genes
            has_spatial: Whether dataset has spatial coordinates
            tasks: List of tasks the dataset supports
            
        Returns:
            List of dataset names matching the criteria
        """
        filtered = []
        
        for name, info in self._datasets.items():
            # Check each filter criterion
            if modality is not None and info.get("modality") != modality:
                continue
            if organism is not None and info.get("organism") != organism:
                continue
            if tissue is not None and info.get("tissue") != tissue:
                continue
            if min_cells is not None and info.get("n_cells", 0) < min_cells:
                continue
            if max_cells is not None and info.get("n_cells", float('inf')) > max_cells:
                continue
            if min_genes is not None and info.get("n_genes", 0) < min_genes:
                continue
            if max_genes is not None and info.get("n_genes", float('inf')) > max_genes:
                continue
            if has_spatial is not None and info.get("has_spatial", False) != has_spatial:
                continue
            if tasks is not None:
                dataset_tasks = info.get("tasks", [])
                if not all(task in dataset_tasks for task in tasks):
                    continue
            
            filtered.append(name)
        
        return filtered
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all datasets.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self._datasets:
            return {}
        
        total_datasets = len(self._datasets)
        total_cells = sum(info.get("n_cells", 0) for info in self._datasets.values())
        
        modalities = {}
        organisms = {}
        tissues = {}
        
        for info in self._datasets.values():
            # Count modalities
            mod = info.get("modality", "unknown")
            modalities[mod] = modalities.get(mod, 0) + 1
            
            # Count organisms
            org = info.get("organism", "unknown")
            organisms[org] = organisms.get(org, 0) + 1
            
            # Count tissues
            tissue = info.get("tissue", "unknown")
            tissues[tissue] = tissues.get(tissue, 0) + 1
        
        return {
            "total_datasets": total_datasets,
            "total_cells": total_cells,
            "modalities": modalities,
            "organisms": organisms,
            "tissues": tissues,
            "avg_cells_per_dataset": total_cells // total_datasets if total_datasets > 0 else 0
        }
    
    def search(self, query: str) -> List[str]:
        """Search datasets by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of dataset names matching the query
        """
        query = query.lower()
        matching = []
        
        for name, info in self._datasets.items():
            # Search in name and description
            if (query in name.lower() or 
                query in info.get("description", "").lower() or
                query in info.get("tissue", "").lower() or
                query in info.get("organism", "").lower()):
                matching.append(name)
        
        return matching
    
    def get_recommendations(self, reference_dataset: str, max_results: int = 5) -> List[str]:
        """Get dataset recommendations based on similarity to a reference dataset.
        
        Args:
            reference_dataset: Name of the reference dataset
            max_results: Maximum number of recommendations
            
        Returns:
            List of recommended dataset names
        """
        if reference_dataset not in self._datasets:
            warnings.warn(f"Reference dataset '{reference_dataset}' not found")
            return []
        
        ref_info = self._datasets[reference_dataset]
        recommendations = []
        
        for name, info in self._datasets.items():
            if name == reference_dataset:
                continue
            
            # Simple similarity score based on shared characteristics
            score = 0
            if info.get("organism") == ref_info.get("organism"):
                score += 3
            if info.get("modality") == ref_info.get("modality"):
                score += 2
            if info.get("tissue") == ref_info.get("tissue"):
                score += 2
            if info.get("has_spatial") == ref_info.get("has_spatial"):
                score += 1
            
            # Size similarity (prefer similar sized datasets)
            ref_cells = ref_info.get("n_cells", 0)
            dataset_cells = info.get("n_cells", 0)
            if ref_cells > 0 and dataset_cells > 0:
                size_ratio = min(ref_cells, dataset_cells) / max(ref_cells, dataset_cells)
                score += size_ratio
            
            recommendations.append((name, score))
        
        # Sort by score and return top results
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [name for name, score in recommendations[:max_results]]
    
    def download_dataset(self, name: str, target_dir: str, 
                        base_url: Optional[str] = None, 
                        verify_checksum: bool = True) -> bool:
        """Download a dataset from the catalog.
        
        Args:
            name: Name of the dataset to download
            target_dir: Directory to download the dataset to
            base_url: Base URL for dataset downloads
            verify_checksum: Whether to verify file checksums
            
        Returns:
            True if download was successful, False otherwise
        """
        if name not in self._datasets:
            warnings.warn(f"Dataset '{name}' not found in catalog")
            return False
        
        dataset_info = self._datasets[name]
        
        # Use default base URL if none provided
        if base_url is None:
            base_url = "https://scgraphhub.s3.amazonaws.com/datasets/"
        
        try:
            # Ensure target directory exists
            os.makedirs(target_dir, exist_ok=True)
            
            # Download main dataset file
            dataset_filename = f"{name}.h5ad"
            dataset_url = urljoin(base_url, dataset_filename)
            target_path = os.path.join(target_dir, dataset_filename)
            
            if self._download_file(dataset_url, target_path):
                # Download metadata file if available
                metadata_filename = f"{name}_metadata.json"
                metadata_url = urljoin(base_url, metadata_filename)
                metadata_path = os.path.join(target_dir, "metadata.json")
                
                # Metadata download is optional
                self._download_file(metadata_url, metadata_path, optional=True)
                
                # Verify checksum if requested
                if verify_checksum and 'checksum' in dataset_info:
                    if not self._verify_checksum(target_path, dataset_info['checksum']):
                        warnings.warn(f"Checksum verification failed for {name}")
                        return False
                
                return True
            
        except Exception as e:
            warnings.warn(f"Failed to download dataset '{name}': {e}")
            return False
        
        return False
    
    def _download_file(self, url: str, target_path: str, optional: bool = False) -> bool:
        """Download a file from URL to target path."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            
            if response.status_code == 404 and optional:
                return True  # Optional file not found is OK
            
            response.raise_for_status()
            
            # Download with progress (simplified)
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            if not optional:
                warnings.warn(f"Failed to download {url}: {e}")
            return False
    
    def _verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """Verify file checksum."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            actual_checksum = hasher.hexdigest()
            return actual_checksum == expected_checksum
            
        except Exception:
            return False
    
    def get_dataset_size(self, name: str) -> int:
        """Get dataset size in MB.
        
        Args:
            name: Dataset name
            
        Returns:
            Size in MB, or 0 if unknown
        """
        if name in self._datasets:
            return self._datasets[name].get('size_mb', 0)
        return 0
    
    def validate_dataset(self, name: str, dataset_path: str) -> Dict[str, Any]:
        """Validate a downloaded dataset.
        
        Args:
            name: Dataset name
            dataset_path: Path to the dataset file
            
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata_match': True
        }
        
        if not os.path.exists(dataset_path):
            validation_results['valid'] = False
            validation_results['errors'].append(f"Dataset file not found: {dataset_path}")
            return validation_results
        
        try:
            # Try to load the dataset
            import scanpy as sc
            adata = sc.read_h5ad(dataset_path)
            
            if name in self._datasets:
                expected_info = self._datasets[name]
                
                # Check cell count
                if 'n_cells' in expected_info:
                    expected_cells = expected_info['n_cells']
                    actual_cells = adata.shape[0]
                    if abs(actual_cells - expected_cells) > 0.1 * expected_cells:
                        validation_results['warnings'].append(
                            f"Cell count mismatch: expected {expected_cells}, got {actual_cells}"
                        )
                        validation_results['metadata_match'] = False
                
                # Check gene count
                if 'n_genes' in expected_info:
                    expected_genes = expected_info['n_genes']
                    actual_genes = adata.shape[1]
                    if abs(actual_genes - expected_genes) > 0.1 * expected_genes:
                        validation_results['warnings'].append(
                            f"Gene count mismatch: expected {expected_genes}, got {actual_genes}"
                        )
                        validation_results['metadata_match'] = False
        
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Failed to load dataset: {e}")
        
        return validation_results
    
    def add_dataset(self, name: str, metadata: Dict[str, Any], 
                   update_if_exists: bool = False) -> bool:
        """Add a new dataset to the catalog.
        
        Args:
            name: Dataset name
            metadata: Dataset metadata
            update_if_exists: Whether to update if dataset already exists
            
        Returns:
            True if successful
        """
        if name in self._datasets and not update_if_exists:
            warnings.warn(f"Dataset '{name}' already exists. Use update_if_exists=True to overwrite.")
            return False
        
        # Validate required fields
        required_fields = ['description', 'n_cells', 'n_genes', 'modality', 'organism']
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            warnings.warn(f"Missing required fields: {missing_fields}")
            return False
        
        # Add the dataset
        self._datasets[name] = metadata.copy()
        self._datasets[name]['name'] = name
        
        return True
    
    def remove_dataset(self, name: str) -> bool:
        """Remove a dataset from the catalog.
        
        Args:
            name: Dataset name
            
        Returns:
            True if successful
        """
        if name not in self._datasets:
            warnings.warn(f"Dataset '{name}' not found")
            return False
        
        del self._datasets[name]
        return True
    
    def export_catalog(self, filepath: str) -> bool:
        """Export catalog to JSON file.
        
        Args:
            filepath: Path to save the catalog
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self._datasets, f, indent=2)
            return True
        except Exception as e:
            warnings.warn(f"Failed to export catalog: {e}")
            return False
    
    def import_catalog(self, filepath: str, merge: bool = True) -> bool:
        """Import catalog from JSON file.
        
        Args:
            filepath: Path to the catalog file
            merge: Whether to merge with existing catalog
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                imported_datasets = json.load(f)
            
            if merge:
                self._datasets.update(imported_datasets)
            else:
                self._datasets = imported_datasets
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to import catalog: {e}")
            return False
    
    def get_download_url(self, name: str, base_url: Optional[str] = None) -> Optional[str]:
        """Get download URL for a dataset.
        
        Args:
            name: Dataset name
            base_url: Base URL for downloads
            
        Returns:
            Download URL or None if not found
        """
        if name not in self._datasets:
            return None
        
        if base_url is None:
            base_url = "https://scgraphhub.s3.amazonaws.com/datasets/"
        
        return urljoin(base_url, f"{name}.h5ad")
    
    def get_tasks_summary(self) -> Dict[str, List[str]]:
        """Get summary of available tasks and associated datasets.
        
        Returns:
            Dictionary mapping tasks to dataset lists
        """
        tasks_map = {}
        
        for name, info in self._datasets.items():
            dataset_tasks = info.get('tasks', [])
            for task in dataset_tasks:
                if task not in tasks_map:
                    tasks_map[task] = []
                tasks_map[task].append(name)
        
        return tasks_map
    
    def update_dataset_metadata(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for an existing dataset.
        
        Args:
            name: Dataset name
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful
        """
        if name not in self._datasets:
            warnings.warn(f"Dataset '{name}' not found")
            return False
        
        self._datasets[name].update(updates)
        return True


# Global catalog instance
_default_catalog = None

def get_default_catalog() -> DatasetCatalog:
    """Get the default global catalog instance."""
    global _default_catalog
    if _default_catalog is None:
        _default_catalog = DatasetCatalog()
    return _default_catalog