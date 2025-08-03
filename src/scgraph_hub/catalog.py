"""Dataset catalog for discovering and managing single-cell graph datasets."""

from typing import Dict, List, Optional, Any
import json
import warnings


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
        # For demonstration, create a mock catalog with diverse datasets
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
                "citation": "Zheng et al., Nat Commun 2017"
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
                "citation": "Tabula Muris Consortium, Nature 2018"
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
                "citation": "Lake et al., Nat Biotechnol 2018"
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
                "citation": "Pijuan-Sala et al., Nat Cell Biol 2019"
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
                "citation": "Tucker et al., Nature 2020"
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
                "citation": "Luecken et al., Mol Syst Biol 2020"
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
                "citation": "Wilk et al., Nat Med 2020"
            }
        }
    
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