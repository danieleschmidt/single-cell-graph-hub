"""Single-Cell Graph Hub: Graph Neural Networks for Single-Cell Omics."""

__version__ = "0.1.0"

# Generation 1 Enhancement: Essential imports for error handling
import warnings
import os
import sys
import logging
from typing import Optional, Dict, Any

# Generation 1: Configure basic logging
def _setup_basic_logging():
    """Setup basic logging configuration for Generation 1."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Initialize basic logger
_logger = _setup_basic_logging()

# Generation 1: Enhanced error handling wrapper
def _safe_import_with_fallback(module_path: str, fallback_message: str = None):
    """Safely import module with detailed error logging."""
    try:
        module = __import__(module_path, fromlist=[''])
        _logger.info(f"Successfully imported {module_path}")
        return module, None
    except ImportError as e:
        error_msg = fallback_message or f"Failed to import {module_path}: {str(e)}"
        _logger.warning(error_msg)
        return None, str(e)

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

# Generation 1: Enhanced simple loader
from .enhanced_simple_loader import (
    EnhancedSimpleLoader, DatasetInfo, get_enhanced_loader, 
    quick_load, list_datasets
)

# Generation 1: Basic models (always available)
try:
    from .basic_models import (
        BaseModelInterface, DummyGNNModel, ModelConfig, ModelRegistry,
        get_model_registry, create_model, list_available_models,
        SimpleTrainer, quick_train
    )
    _BASIC_MODELS_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"Basic models not available: {e}")
    _BASIC_MODELS_AVAILABLE = False

# Generation 2: Robust error handling and validation
try:
    from .robust_error_handling import (
        ErrorSeverity, ErrorCategory, ErrorContext, RobustErrorHandler,
        get_error_handler, robust_wrapper, DataValidator, get_validator,
        InputSanitizer, get_sanitizer, validate_and_sanitize, HealthMonitor,
        get_health_monitor
    )
    _ROBUST_ERROR_HANDLING_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"Robust error handling not available: {e}")
    _ROBUST_ERROR_HANDLING_AVAILABLE = False

# Generation 2: Advanced monitoring
try:
    from .advanced_monitoring import (
        PerformanceMetrics, MetricsCollector, AdvancedLogger, monitored_function,
        SystemMonitor, get_metrics_collector, get_system_monitor, performance_context
    )
    _ADVANCED_MONITORING_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"Advanced monitoring not available: {e}")
    _ADVANCED_MONITORING_AVAILABLE = False

# Generation 2: Security framework
try:
    from .security_framework import (
        SecurityLevel, ThreatType, SecurityEvent, PathValidator, InputValidator,
        AccessControl, DataEncryption, SecurityAuditor, SecureOperationManager,
        get_security_manager, secure_operation
    )
    _SECURITY_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"Security framework not available: {e}")
    _SECURITY_FRAMEWORK_AVAILABLE = False

# Generation 3: Enhanced performance optimization
try:
    from .enhanced_performance import (
        CacheEntry, AdvancedCache, cached, ConcurrentProcessor, ResourcePool,
        PerformanceOptimizer, get_performance_optimizer, performance_optimized,
        performance_monitor
    )
    _ENHANCED_PERFORMANCE_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"Enhanced performance not available: {e}")
    _ENHANCED_PERFORMANCE_AVAILABLE = False

# Generation 3: Scalability framework
try:
    from .scalability_framework import (
        LoadBalancingStrategy, ScalingEvent, WorkerNode, Task, LoadBalancer,
        AutoScaler, DistributedTaskManager, get_distributed_task_manager,
        distributed_task
    )
    _SCALABILITY_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"Scalability framework not available: {e}")
    _SCALABILITY_FRAMEWORK_AVAILABLE = False

# TERRAGON SDLC v6.0 Progressive Enhancements
try:
    from .progressive_quality_gates import (
        ProgressiveQualityGateSystem, ProgressiveLevel, run_progressive_gates,
        get_progressive_gates_system
    )
    from .progressive_resilience import (
        ProgressiveResilienceOrchestrator, get_progressive_resilience,
        resilient_execution, resilient
    )
    from .progressive_scalability import (
        ProgressiveDistributedProcessor, get_progressive_processor,
        distributed_execute, distributed_task
    )
    _PROGRESSIVE_ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"Progressive enhancements not available: {e}")
    _PROGRESSIVE_ENHANCEMENTS_AVAILABLE = False

# Advanced functionality with error handling
try:
    from .advanced_dataset import DatasetProcessor, EnhancedSCGraphDataset
    from .security import SecurityValidator, SecureFileHandler, get_security_validator
    from .logging_config import setup_logging, get_logger, get_contextual_logger
    from .health_checks import SystemHealthChecker, HealthStatus
    _ADVANCED_AVAILABLE = True
except ImportError:
    _ADVANCED_AVAILABLE = False

# Legacy scalability features (keeping for backward compatibility)
try:
    from .performance import (
        PerformanceOptimizer as LegacyPerformanceOptimizer, 
        PerformanceCache, ResourceOptimizer,
        high_performance, auto_scale
    )
    from .scalability import (
        DistributedTaskManager as LegacyDistributedTaskManager, 
        LoadBalancer as LegacyLoadBalancer, 
        AutoScaler as LegacyAutoScaler, 
        WorkerNode as LegacyWorkerNode, 
        Task as LegacyTask,
        get_distributed_task_manager as get_legacy_distributed_task_manager, 
        distributed_task as legacy_distributed_task
    )
    _LEGACY_SCALABILITY_AVAILABLE = True
except ImportError:
    _LEGACY_SCALABILITY_AVAILABLE = False

# Utility functions
from .utils import check_dependencies, validate_dataset_config

# Export available classes - basic functionality
__all__ = [
    "DatasetCatalog", 
    "get_default_catalog",
    "SimpleSCGraphDataset",
    "SimpleSCGraphData",
    # Generation 1: Enhanced loader
    "EnhancedSimpleLoader",
    "DatasetInfo", 
    "get_enhanced_loader",
    "quick_load",
    "list_datasets",
    # Generation 1: Basic models
    "BaseModelInterface",
    "DummyGNNModel", 
    "ModelConfig",
    "ModelRegistry",
    "get_model_registry",
    "create_model",
    "list_available_models",
    "SimpleTrainer",
    "quick_train",
    "check_dependencies", 
    "validate_dataset_config"
]

# Add Generation 2: Robust error handling if available
if _ROBUST_ERROR_HANDLING_AVAILABLE:
    __all__.extend([
        "ErrorSeverity", "ErrorCategory", "ErrorContext", "RobustErrorHandler",
        "get_error_handler", "robust_wrapper", "DataValidator", "get_validator",
        "InputSanitizer", "get_sanitizer", "validate_and_sanitize", "HealthMonitor",
        "get_health_monitor"
    ])

# Add Generation 2: Advanced monitoring if available  
if _ADVANCED_MONITORING_AVAILABLE:
    __all__.extend([
        "PerformanceMetrics", "MetricsCollector", "AdvancedLogger", "monitored_function",
        "SystemMonitor", "get_metrics_collector", "get_system_monitor", "performance_context"
    ])

# Add Generation 2: Security framework if available
if _SECURITY_FRAMEWORK_AVAILABLE:
    __all__.extend([
        "SecurityLevel", "ThreatType", "SecurityEvent", "PathValidator", "InputValidator",
        "AccessControl", "DataEncryption", "SecurityAuditor", "SecureOperationManager", 
        "get_security_manager", "secure_operation"
    ])

# Add Generation 3: Enhanced performance if available
if _ENHANCED_PERFORMANCE_AVAILABLE:
    __all__.extend([
        "CacheEntry", "AdvancedCache", "cached", "ConcurrentProcessor", "ResourcePool",
        "PerformanceOptimizer", "get_performance_optimizer", "performance_optimized",
        "performance_monitor"
    ])

# Add Generation 3: Scalability framework if available
if _SCALABILITY_FRAMEWORK_AVAILABLE:
    __all__.extend([
        "LoadBalancingStrategy", "ScalingEvent", "WorkerNode", "Task", "LoadBalancer",
        "AutoScaler", "DistributedTaskManager", "get_distributed_task_manager",
        "distributed_task"
    ])

# Add TERRAGON SDLC v6.0 Progressive Enhancements if available
if _PROGRESSIVE_ENHANCEMENTS_AVAILABLE:
    __all__.extend([
        # Progressive Quality Gates
        "ProgressiveQualityGateSystem", "ProgressiveLevel", "run_progressive_gates",
        "get_progressive_gates_system",
        # Progressive Resilience
        "ProgressiveResilienceOrchestrator", "get_progressive_resilience",
        "resilient_execution", "resilient",
        # Progressive Scalability
        "ProgressiveDistributedProcessor", "get_progressive_processor",
        "distributed_execute", "distributed_task"
    ])

# Add advanced functionality if available
if _ADVANCED_AVAILABLE:
    __all__.extend([
        "DatasetProcessor",
        "EnhancedSCGraphDataset", 
        "SecurityValidator",
        "SecureFileHandler",
        "get_security_validator",
        "setup_logging",
        "get_logger",
        "get_contextual_logger",
        "SystemHealthChecker",
        "HealthStatus"
    ])
else:
    # Provide placeholder functions for advanced features
    def _advanced_feature_unavailable(*args, **kwargs):
        raise ImportError("Advanced features require additional dependencies. Install with: pip install single-cell-graph-hub[full]")
    
    DatasetProcessor = _advanced_feature_unavailable
    EnhancedSCGraphDataset = _advanced_feature_unavailable
    SecurityValidator = _advanced_feature_unavailable
    SecureFileHandler = _advanced_feature_unavailable
    get_security_validator = _advanced_feature_unavailable
    setup_logging = _advanced_feature_unavailable
    get_logger = _advanced_feature_unavailable
    get_contextual_logger = _advanced_feature_unavailable
    SystemHealthChecker = _advanced_feature_unavailable
    HealthStatus = _advanced_feature_unavailable

# Add scalability features if available
# Add legacy scalability features if available
if _LEGACY_SCALABILITY_AVAILABLE:
    __all__.extend([
        "LegacyPerformanceOptimizer",
        "PerformanceCache", 
        "ResourceOptimizer",
        "high_performance",
        "auto_scale",
        "LegacyDistributedTaskManager",
        "LegacyLoadBalancer",
        "LegacyAutoScaler", 
        "LegacyWorkerNode",
        "LegacyTask",
        "get_legacy_distributed_task_manager",
        "legacy_distributed_task"
    ])

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


# Progressive SDLC Quick Start
def progressive_quick_start(level: str = "basic", **kwargs):
    """Quick start function for TERRAGON SDLC v6.0 progressive enhancements.
    
    Args:
        level: Progressive level (basic, intermediate, advanced, expert, autonomous)
        **kwargs: Additional configuration
        
    Returns:
        Tuple of (quality_gates, resilience, scalability) systems
    """
    if not _PROGRESSIVE_ENHANCEMENTS_AVAILABLE:
        raise ImportError("Progressive enhancements require installation")
    
    quality_gates = get_progressive_gates_system()
    resilience = get_progressive_resilience()
    scalability = get_progressive_processor()
    
    # Set level if provided
    if level != "basic":
        from .progressive_quality_gates import ProgressiveLevel
        prog_level = ProgressiveLevel(level)
        quality_gates.current_level = prog_level
        resilience.level = prog_level
        scalability.level = prog_level
    
    return quality_gates, resilience, scalability


# Add to exports
__all__.extend(["quick_start", "simple_quick_start", "progressive_quick_start"])