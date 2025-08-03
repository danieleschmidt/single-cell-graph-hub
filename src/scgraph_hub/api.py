"""FastAPI-based REST API for Single-Cell Graph Hub."""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

from .catalog import DatasetCatalog, get_default_catalog
from .data_manager import get_data_manager, DataManager
from .database import get_dataset_repository
from .models import MODEL_REGISTRY, get_model_recommendations

logger = logging.getLogger(__name__)

# Pydantic models for API
class DatasetInfo(BaseModel):
    """Dataset information response model."""
    name: str
    description: str
    n_cells: int
    n_genes: int
    n_classes: int
    modality: str
    organism: str
    tissue: str
    has_spatial: bool
    graph_method: str
    size_mb: float
    citation: str
    tasks: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "name": "pbmc_10k",
                "description": "10k PBMCs from a healthy donor",
                "n_cells": 10000,
                "n_genes": 2000,
                "n_classes": 8,
                "modality": "scRNA-seq",
                "organism": "human",
                "tissue": "blood",
                "has_spatial": False,
                "graph_method": "knn",
                "size_mb": 150.0,
                "citation": "Zheng et al., Nat Commun 2017",
                "tasks": ["cell_type_prediction", "gene_imputation"]
            }
        }


class DatasetFilter(BaseModel):
    """Dataset filtering parameters."""
    modality: Optional[str] = None
    organism: Optional[str] = None
    tissue: Optional[str] = None
    min_cells: Optional[int] = None
    max_cells: Optional[int] = None
    min_genes: Optional[int] = None
    max_genes: Optional[int] = None
    has_spatial: Optional[bool] = None
    tasks: Optional[List[str]] = None


class DatasetStatistics(BaseModel):
    """Dataset statistics response model."""
    basic: Dict[str, Any]
    graph: Dict[str, Any]
    features: Dict[str, Any]
    splits: Optional[Dict[str, Any]] = None


class ModelConfig(BaseModel):
    """Model configuration for training."""
    model_name: str = Field(..., description="Name of the model type")
    input_dim: Optional[int] = None
    hidden_dim: int = 128
    output_dim: Optional[int] = None
    num_layers: int = 3
    dropout: float = 0.2
    
    @validator('model_name')
    def validate_model_name(cls, v):
        if v not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {v}")
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    weight_decay: float = 1e-4
    early_stopping: bool = True
    patience: int = 10
    
    @validator('epochs')
    def validate_epochs(cls, v):
        if v <= 0:
            raise ValueError("Epochs must be positive")
        return v
    
    @validator('learning_rate')
    def validate_lr(cls, v):
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        return v


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""
    steps: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    graph_method: str = "knn"
    graph_parameters: Optional[Dict[str, Any]] = None
    save_intermediate: bool = False


class ExperimentRequest(BaseModel):
    """Experiment execution request."""
    dataset_name: str
    model_config: ModelConfig
    training_config: TrainingConfig = TrainingConfig()
    preprocessing_config: Optional[PreprocessingConfig] = None
    experiment_name: Optional[str] = None


class TaskProgress(BaseModel):
    """Task progress information."""
    task_id: str
    status: str
    progress: float
    message: str
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]


# Create FastAPI app
app = FastAPI(
    title="Single-Cell Graph Hub API",
    description="REST API for graph-based single-cell omics analysis",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer(auto_error=False)

# Global state
task_store = {}  # In production, use Redis or database


# Dependencies
async def get_catalog() -> DatasetCatalog:
    """Get dataset catalog instance."""
    return get_default_catalog()


async def get_data_manager_instance() -> DataManager:
    """Get data manager instance."""
    return get_data_manager()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token (placeholder implementation)."""
    if not credentials:
        return None  # Allow anonymous access for now
    
    # In production, implement proper JWT verification
    return {"user_id": "anonymous"}


# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Single-Cell Graph Hub API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    # Check service health
    services = {"api": "healthy"}
    
    # Check database
    try:
        repo = get_dataset_repository()
        repo.list_datasets(limit=1)
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = f"unhealthy: {str(e)}"
    
    # Check cache
    try:
        from .database import get_cache_manager
        cache = get_cache_manager()
        cache.set("health_check", "test", ttl=60)
        services["cache"] = "healthy"
    except Exception as e:
        services["cache"] = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if all(
        status == "healthy" for status in services.values()
    ) else "degraded"
    
    return HealthCheck(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="0.1.0",
        services=services
    )


@app.get("/datasets", response_model=List[str])
async def list_datasets(catalog: DatasetCatalog = Depends(get_catalog)):
    """List all available datasets."""
    return catalog.list_datasets()


@app.get("/datasets/{dataset_name}", response_model=DatasetInfo)
async def get_dataset_info(dataset_name: str, catalog: DatasetCatalog = Depends(get_catalog)):
    """Get detailed information about a specific dataset."""
    try:
        info = catalog.get_info(dataset_name)
        return DatasetInfo(**info)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found"
        )


@app.post("/datasets/search", response_model=List[str])
async def search_datasets(
    filters: DatasetFilter,
    catalog: DatasetCatalog = Depends(get_catalog)
):
    """Search datasets with filters."""
    filter_dict = {k: v for k, v in filters.dict().items() if v is not None}
    return catalog.filter(**filter_dict)


@app.get("/datasets/{dataset_name}/statistics", response_model=DatasetStatistics)
async def get_dataset_statistics(
    dataset_name: str,
    dm: DataManager = Depends(get_data_manager_instance)
):
    """Get comprehensive statistics for a dataset."""
    stats = dm.get_dataset_statistics(dataset_name)
    if stats is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found or not processed"
        )
    return DatasetStatistics(**stats)


@app.post("/datasets/{dataset_name}/download")
async def download_dataset(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    force_redownload: bool = False,
    verify_checksum: bool = True,
    dm: DataManager = Depends(get_data_manager_instance)
):
    """Download a dataset."""
    # Create task ID
    task_id = f"download_{dataset_name}_{datetime.utcnow().timestamp()}"
    
    # Initialize task status
    task_store[task_id] = TaskProgress(
        task_id=task_id,
        status="queued",
        progress=0.0,
        message="Download queued",
        started_at=datetime.utcnow()
    )
    
    # Background download
    async def download_task():
        task_store[task_id].status = "running"
        task_store[task_id].message = "Downloading dataset..."
        
        try:
            async with dm:
                success = await dm.download_dataset_async(
                    dataset_name,
                    force_redownload=force_redownload,
                    verify_checksum=verify_checksum,
                    progress_callback=lambda name, progress: update_progress(task_id, progress * 100)
                )
            
            if success:
                task_store[task_id].status = "completed"
                task_store[task_id].progress = 100.0
                task_store[task_id].message = "Download completed"
            else:
                task_store[task_id].status = "failed"
                task_store[task_id].message = "Download failed"
        
        except Exception as e:
            task_store[task_id].status = "failed"
            task_store[task_id].message = f"Download error: {str(e)}"
    
    background_tasks.add_task(download_task)
    
    return {"task_id": task_id, "message": "Download started"}


@app.post("/datasets/{dataset_name}/preprocess")
async def preprocess_dataset(
    dataset_name: str,
    config: PreprocessingConfig,
    background_tasks: BackgroundTasks,
    force_reprocess: bool = False,
    dm: DataManager = Depends(get_data_manager_instance)
):
    """Preprocess a dataset."""
    # Create task ID
    task_id = f"preprocess_{dataset_name}_{datetime.utcnow().timestamp()}"
    
    # Initialize task status
    task_store[task_id] = TaskProgress(
        task_id=task_id,
        status="queued",
        progress=0.0,
        message="Preprocessing queued",
        started_at=datetime.utcnow()
    )
    
    # Background preprocessing
    async def preprocess_task():
        task_store[task_id].status = "running"
        task_store[task_id].message = "Preprocessing dataset..."
        task_store[task_id].progress = 10.0
        
        try:
            async with dm:
                success = await dm.preprocess_dataset_async(
                    dataset_name,
                    preprocessing_config=config.dict(),
                    force_reprocess=force_reprocess
                )
            
            if success:
                task_store[task_id].status = "completed"
                task_store[task_id].progress = 100.0
                task_store[task_id].message = "Preprocessing completed"
            else:
                task_store[task_id].status = "failed"
                task_store[task_id].message = "Preprocessing failed"
        
        except Exception as e:
            task_store[task_id].status = "failed"
            task_store[task_id].message = f"Preprocessing error: {str(e)}"
    
    background_tasks.add_task(preprocess_task)
    
    return {"task_id": task_id, "message": "Preprocessing started"}


@app.get("/tasks/{task_id}", response_model=TaskProgress)
async def get_task_status(task_id: str):
    """Get status of a background task."""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_store[task_id]


@app.get("/models", response_model=List[str])
async def list_models():
    """List available model types."""
    return list(MODEL_REGISTRY.keys())


@app.get("/models/recommendations")
async def get_model_recommendations_endpoint(
    dataset_name: str,
    catalog: DatasetCatalog = Depends(get_catalog)
):
    """Get model recommendations for a dataset."""
    try:
        dataset_info = catalog.get_info(dataset_name)
        recommendations = get_model_recommendations(dataset_info)
        return {"dataset": dataset_name, "recommended_models": recommendations}
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found"
        )


@app.post("/experiments")
async def run_experiment(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks,
    user = Depends(verify_token)
):
    """Run a training experiment."""
    # Create task ID
    task_id = f"experiment_{request.dataset_name}_{datetime.utcnow().timestamp()}"
    
    # Initialize task status
    task_store[task_id] = TaskProgress(
        task_id=task_id,
        status="queued",
        progress=0.0,
        message="Experiment queued",
        started_at=datetime.utcnow()
    )
    
    # Background experiment
    async def experiment_task():
        task_store[task_id].status = "running"
        task_store[task_id].message = "Running experiment..."
        
        try:
            # This would integrate with actual training pipeline
            # For now, simulate experiment
            import asyncio
            for i in range(10):
                await asyncio.sleep(1)
                task_store[task_id].progress = (i + 1) * 10.0
                task_store[task_id].message = f"Training epoch {i+1}/10"
            
            task_store[task_id].status = "completed"
            task_store[task_id].progress = 100.0
            task_store[task_id].message = "Experiment completed"
            task_store[task_id].results = {
                "accuracy": 0.95,
                "f1_score": 0.93,
                "training_time": 600
            }
        
        except Exception as e:
            task_store[task_id].status = "failed"
            task_store[task_id].message = f"Experiment error: {str(e)}"
    
    background_tasks.add_task(experiment_task)
    
    return {"task_id": task_id, "message": "Experiment started"}


@app.get("/catalog/summary")
async def get_catalog_summary(catalog: DatasetCatalog = Depends(get_catalog)):
    """Get catalog summary statistics."""
    return catalog.get_summary_stats()


@app.get("/catalog/tasks")
async def get_tasks_summary(catalog: DatasetCatalog = Depends(get_catalog)):
    """Get summary of available tasks."""
    return catalog.get_tasks_summary()


@app.get("/storage/info")
async def get_storage_info(dm: DataManager = Depends(get_data_manager_instance)):
    """Get storage usage information."""
    return dm.get_storage_info()


# Utility functions
def update_progress(task_id: str, progress: float):
    """Update task progress."""
    if task_id in task_store:
        task_store[task_id].progress = progress


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


# Application factory
def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create FastAPI application with configuration."""
    if config:
        # Apply configuration
        pass
    
    return app


# Development server
def run_dev_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run development server."""
    uvicorn.run(
        "scgraph_hub.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_dev_server()