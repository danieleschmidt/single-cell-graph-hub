"""Comprehensive benchmarking system for single-cell graph models."""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
import json
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader

from .models import BaseGNN, MODEL_REGISTRY, create_model
from .data_manager import get_data_manager
from .metrics import BiologicalMetrics, GraphMetrics
from .database import get_dataset_repository

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Comprehensive benchmarking system for GNN models on single-cell data."""
    
    def __init__(self, 
                 results_dir: Optional[str] = None,
                 device: str = 'cpu',
                 save_predictions: bool = True,
                 track_resources: bool = True):
        """Initialize benchmark runner.
        
        Args:
            results_dir: Directory to save benchmark results
            device: Device for model training/evaluation
            save_predictions: Whether to save model predictions
            track_resources: Whether to track resource usage
        """
        self.results_dir = Path(results_dir or './benchmark_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.save_predictions = save_predictions
        self.track_resources = track_resources
        
        self.data_manager = get_data_manager()
        self.repository = get_dataset_repository()
        
        # Initialize metrics calculators
        self.bio_metrics = BiologicalMetrics()
        self.graph_metrics = GraphMetrics()
        
        # Benchmark history
        self.benchmark_history = []
    
    def run_benchmark(self,
                     models: Dict[str, Union[BaseGNN, Dict[str, Any]]],
                     datasets: List[str],
                     tasks: List[str],
                     metrics: List[str],
                     n_runs: int = 5,
                     train_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark across models, datasets, and tasks.
        
        Args:
            models: Dictionary mapping model names to model instances or configs
            datasets: List of dataset names to benchmark on
            tasks: List of tasks to evaluate
            metrics: List of metrics to compute
            n_runs: Number of independent runs per configuration
            train_config: Training configuration
            
        Returns:
            Complete benchmark results
        """
        benchmark_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting benchmark {benchmark_id}")
        
        results = {
            'benchmark_id': benchmark_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'models': list(models.keys()),
                'datasets': datasets,
                'tasks': tasks,
                'metrics': metrics,
                'n_runs': n_runs,
                'train_config': train_config or {}
            },
            'results': {},
            'summary': {},
            'resource_usage': {}
        }
        
        total_experiments = len(models) * len(datasets) * len(tasks) * n_runs
        experiment_count = 0
        
        for model_name, model_config in models.items():
            results['results'][model_name] = {}
            
            for dataset_name in datasets:
                results['results'][model_name][dataset_name] = {}
                
                # Load dataset
                dataset = self.data_manager.load_dataset(dataset_name, device=self.device)
                if dataset is None:
                    logger.error(f"Failed to load dataset {dataset_name}")
                    continue
                
                for task in tasks:
                    results['results'][model_name][dataset_name][task] = {
                        'runs': [],
                        'aggregated': {}
                    }
                    
                    # Run multiple independent experiments
                    for run_idx in range(n_runs):
                        experiment_count += 1
                        logger.info(
                            f"Running experiment {experiment_count}/{total_experiments}: "
                            f"{model_name} on {dataset_name} for {task} (run {run_idx + 1}/{n_runs})"
                        )
                        
                        try:
                            run_results = self._run_single_experiment(
                                model_name=model_name,
                                model_config=model_config,
                                dataset=dataset,
                                dataset_name=dataset_name,
                                task=task,
                                metrics=metrics,
                                train_config=train_config or {},
                                run_idx=run_idx
                            )
                            
                            results['results'][model_name][dataset_name][task]['runs'].append(run_results)
                            
                        except Exception as e:
                            logger.error(f"Experiment failed: {e}")
                            results['results'][model_name][dataset_name][task]['runs'].append({
                                'status': 'failed',
                                'error': str(e)
                            })
                    
                    # Aggregate results across runs
                    self._aggregate_run_results(
                        results['results'][model_name][dataset_name][task]
                    )
        
        # Generate summary statistics
        results['summary'] = self._generate_benchmark_summary(results)
        
        # Save results
        results_file = self.results_dir / f"{benchmark_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark completed. Results saved to {results_file}")
        
        # Store in benchmark history
        self.benchmark_history.append(results)
        
        return results
    
    def _run_single_experiment(self,
                              model_name: str,
                              model_config: Union[BaseGNN, Dict[str, Any]],
                              dataset: Any,
                              dataset_name: str,
                              task: str,
                              metrics: List[str],
                              train_config: Dict[str, Any],
                              run_idx: int) -> Dict[str, Any]:
        """Run a single experiment."""
        start_time = time.time()
        
        # Create model instance
        if isinstance(model_config, BaseGNN):
            model = model_config
        else:
            # Determine model dimensions from dataset
            model_config = model_config.copy()
            if 'input_dim' not in model_config:
                model_config['input_dim'] = dataset.x.shape[1]
            if 'output_dim' not in model_config and hasattr(dataset, 'y') and dataset.y is not None:
                model_config['output_dim'] = len(torch.unique(dataset.y))
            
            model = create_model(model_name, **model_config)
        
        model = model.to(self.device)
        
        # Set up training
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.get('learning_rate', 0.001),
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
        
        # Task-specific loss function
        if task in ['cell_type_prediction', 'disease_classification']:
            criterion = nn.CrossEntropyLoss()
        elif task in ['trajectory_inference', 'gene_imputation']:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()  # Default
        
        # Training loop
        model.train()
        training_losses = []
        
        epochs = train_config.get('epochs', 100)
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(dataset, 'train_mask'):
                out = model(dataset.x, dataset.edge_index)
                loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
            else:
                out = model(dataset.x, dataset.edge_index)
                loss = criterion(out, dataset.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(dataset.x, dataset.edge_index)
            
            # Get embeddings
            embeddings = model.get_embeddings(dataset.x, dataset.edge_index)
        
        # Compute metrics
        computed_metrics = self._compute_metrics(
            predictions=predictions,
            embeddings=embeddings,
            ground_truth=dataset.y if hasattr(dataset, 'y') else None,
            dataset=dataset,
            task=task,
            metrics=metrics
        )
        
        training_time = time.time() - start_time
        
        # Create result object
        result = {
            'run_idx': run_idx,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'task': task,
            'training_time': training_time,
            'training_losses': training_losses,
            'final_loss': training_losses[-1] if training_losses else None,
            'metrics': computed_metrics,
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
            'status': 'completed'
        }
        
        # Save predictions if requested
        if self.save_predictions:
            pred_file = self.results_dir / f"predictions_{model_name}_{dataset_name}_{task}_{run_idx}.pt"
            torch.save({
                'predictions': predictions.cpu(),
                'embeddings': embeddings.cpu(),
                'ground_truth': dataset.y.cpu() if hasattr(dataset, 'y') else None
            }, pred_file)
            result['predictions_file'] = str(pred_file)
        
        return result
    
    def _compute_metrics(self,
                        predictions: torch.Tensor,
                        embeddings: torch.Tensor,
                        ground_truth: Optional[torch.Tensor],
                        dataset: Any,
                        task: str,
                        metrics: List[str]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        computed_metrics = {}
        
        if ground_truth is None:
            logger.warning("No ground truth labels available, skipping supervised metrics")
            return computed_metrics
        
        # Move to CPU for sklearn compatibility
        pred_cpu = predictions.cpu().numpy()
        gt_cpu = ground_truth.cpu().numpy()
        emb_cpu = embeddings.cpu().numpy()
        
        for metric in metrics:
            try:
                if metric == 'accuracy':
                    if task in ['cell_type_prediction', 'disease_classification']:
                        pred_labels = np.argmax(pred_cpu, axis=1)
                        computed_metrics['accuracy'] = accuracy_score(gt_cpu, pred_labels)
                
                elif metric == 'f1_score':
                    if task in ['cell_type_prediction', 'disease_classification']:
                        pred_labels = np.argmax(pred_cpu, axis=1)
                        computed_metrics['f1_macro'] = f1_score(gt_cpu, pred_labels, average='macro')
                        computed_metrics['f1_weighted'] = f1_score(gt_cpu, pred_labels, average='weighted')
                
                elif metric == 'silhouette_score':
                    if len(np.unique(gt_cpu)) > 1:
                        computed_metrics['silhouette_score'] = silhouette_score(emb_cpu, gt_cpu)
                
                elif metric == 'ari':
                    if task in ['cell_type_prediction', 'disease_classification']:
                        pred_labels = np.argmax(pred_cpu, axis=1)
                        computed_metrics['ari'] = adjusted_rand_score(gt_cpu, pred_labels)
                
                elif metric == 'nmi':
                    if task in ['cell_type_prediction', 'disease_classification']:
                        pred_labels = np.argmax(pred_cpu, axis=1)
                        computed_metrics['nmi'] = normalized_mutual_info_score(gt_cpu, pred_labels)
                
                elif metric == 'mse':
                    if task in ['trajectory_inference', 'gene_imputation']:
                        computed_metrics['mse'] = mean_squared_error(gt_cpu, pred_cpu[:, 0])
                
                elif metric == 'pearson_r':
                    if task in ['trajectory_inference', 'gene_imputation']:
                        r, p = pearsonr(gt_cpu, pred_cpu[:, 0])
                        computed_metrics['pearson_r'] = r
                
                elif metric == 'biological_conservation':
                    # Use biological metrics if available
                    try:
                        bio_score = self.bio_metrics.biological_conservation(
                            original_data=dataset,
                            embedded_data=emb_cpu
                        )
                        computed_metrics['biological_conservation'] = bio_score
                    except Exception as e:
                        logger.warning(f"Failed to compute biological conservation: {e}")
                
                else:
                    logger.warning(f"Unknown metric: {metric}")
            
            except Exception as e:
                logger.error(f"Failed to compute metric {metric}: {e}")
        
        return computed_metrics
    
    def _aggregate_run_results(self, task_results: Dict[str, Any]):
        """Aggregate results across multiple runs."""
        runs = task_results['runs']
        successful_runs = [run for run in runs if run.get('status') == 'completed']
        
        if not successful_runs:
            task_results['aggregated'] = {'status': 'all_failed'}
            return
        
        # Aggregate metrics
        all_metrics = {}
        for run in successful_runs:
            for metric_name, value in run.get('metrics', {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Compute statistics
        aggregated_metrics = {}
        for metric_name, values in all_metrics.items():
            aggregated_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'n_runs': len(values)
            }
        
        # Aggregate training info
        training_times = [run['training_time'] for run in successful_runs]
        final_losses = [run['final_loss'] for run in successful_runs if run['final_loss'] is not None]
        
        task_results['aggregated'] = {
            'status': 'completed',
            'n_successful_runs': len(successful_runs),
            'n_failed_runs': len(runs) - len(successful_runs),
            'metrics': aggregated_metrics,
            'training_time': {
                'mean': np.mean(training_times),
                'std': np.std(training_times)
            },
            'final_loss': {
                'mean': np.mean(final_losses) if final_losses else None,
                'std': np.std(final_losses) if final_losses else None
            }
        }
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across the entire benchmark."""
        summary = {
            'best_performers': {},
            'model_rankings': {},
            'dataset_difficulty': {},
            'task_complexity': {}
        }
        
        # Find best performers for each metric
        for metric in ['accuracy', 'f1_macro', 'silhouette_score']:
            best_performers = []
            
            for model_name, model_results in results['results'].items():
                for dataset_name, dataset_results in model_results.items():
                    for task_name, task_results in dataset_results.items():
                        aggregated = task_results.get('aggregated', {})
                        metrics = aggregated.get('metrics', {})
                        
                        if metric in metrics:
                            score = metrics[metric]['mean']
                            best_performers.append({
                                'model': model_name,
                                'dataset': dataset_name,
                                'task': task_name,
                                'score': score
                            })
            
            # Sort by score (descending)
            best_performers.sort(key=lambda x: x['score'], reverse=True)
            summary['best_performers'][metric] = best_performers[:10]  # Top 10
        
        return summary
    
    def compare_models(self,
                      model_results: Dict[str, Any],
                      metric: str = 'accuracy',
                      save_plot: bool = True) -> pd.DataFrame:
        """Compare model performance across datasets and tasks."""
        comparison_data = []
        
        for model_name, model_data in model_results.items():
            for dataset_name, dataset_data in model_data.items():
                for task_name, task_data in dataset_data.items():
                    aggregated = task_data.get('aggregated', {})
                    metrics = aggregated.get('metrics', {})
                    
                    if metric in metrics:
                        comparison_data.append({
                            'Model': model_name,
                            'Dataset': dataset_name,
                            'Task': task_name,
                            'Score': metrics[metric]['mean'],
                            'Std': metrics[metric]['std'],
                            'N_Runs': metrics[metric]['n_runs']
                        })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_file = self.results_dir / f"model_comparison_{metric}.csv"
        df.to_csv(comparison_file, index=False)
        
        return df
    
    def generate_report(self,
                       results: Dict[str, Any],
                       output_path: Optional[str] = None) -> str:
        """Generate a comprehensive benchmark report."""
        if output_path is None:
            output_path = self.results_dir / f"benchmark_report_{results['benchmark_id']}.html"
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Single-Cell Graph Hub Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metric-table {{ border-collapse: collapse; width: 100%; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .best-score {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Single-Cell Graph Hub Benchmark Report</h1>
                <p>Benchmark ID: {results['benchmark_id']}</p>
                <p>Generated: {results['timestamp']}</p>
            </div>
            
            <div class="section">
                <h2>Configuration</h2>
                <ul>
                    <li>Models: {', '.join(results['config']['models'])}</li>
                    <li>Datasets: {', '.join(results['config']['datasets'])}</li>
                    <li>Tasks: {', '.join(results['config']['tasks'])}</li>
                    <li>Metrics: {', '.join(results['config']['metrics'])}</li>
                    <li>Runs per configuration: {results['config']['n_runs']}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <!-- Summary statistics would be inserted here -->
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                <!-- Detailed results tables would be inserted here -->
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Benchmark report saved to {output_path}")
        return str(output_path)
    
    def load_benchmark_results(self, benchmark_id: str) -> Optional[Dict[str, Any]]:
        """Load previously saved benchmark results."""
        results_file = self.results_dir / f"{benchmark_id}_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def list_benchmarks(self) -> List[str]:
        """List available benchmark results."""
        benchmark_files = list(self.results_dir.glob("*_results.json"))
        benchmark_ids = [f.stem.replace('_results', '') for f in benchmark_files]
        return sorted(benchmark_ids)


class CellTypeBenchmark:
    """Specialized benchmark for cell type prediction tasks."""
    
    def __init__(self, benchmark_runner: Optional[BenchmarkRunner] = None):
        self.runner = benchmark_runner or BenchmarkRunner()
    
    def evaluate(self,
                models: Dict[str, Union[BaseGNN, Dict[str, Any]]],
                datasets: List[str],
                metrics: Optional[List[str]] = None,
                n_runs: int = 5) -> Dict[str, Any]:
        """Evaluate models on cell type prediction task."""
        if metrics is None:
            metrics = ['accuracy', 'f1_score', 'silhouette_score', 'ari', 'nmi']
        
        return self.runner.run_benchmark(
            models=models,
            datasets=datasets,
            tasks=['cell_type_prediction'],
            metrics=metrics,
            n_runs=n_runs
        )


class TrajectoryBenchmark:
    """Specialized benchmark for trajectory inference tasks."""
    
    def __init__(self, benchmark_runner: Optional[BenchmarkRunner] = None):
        self.runner = benchmark_runner or BenchmarkRunner()
    
    def evaluate(self,
                models: Dict[str, Union[BaseGNN, Dict[str, Any]]],
                datasets: List[str],
                metrics: Optional[List[str]] = None,
                n_runs: int = 5) -> Dict[str, Any]:
        """Evaluate models on trajectory inference task."""
        if metrics is None:
            metrics = ['mse', 'pearson_r', 'biological_conservation']
        
        return self.runner.run_benchmark(
            models=models,
            datasets=datasets,
            tasks=['trajectory_inference'],
            metrics=metrics,
            n_runs=n_runs
        )


class BatchCorrectionBenchmark:
    """Specialized benchmark for batch correction tasks."""
    
    def __init__(self, benchmark_runner: Optional[BenchmarkRunner] = None):
        self.runner = benchmark_runner or BenchmarkRunner()
    
    def evaluate(self,
                models: Dict[str, Union[BaseGNN, Dict[str, Any]]],
                datasets: List[str],
                metrics: Optional[List[str]] = None,
                n_runs: int = 5) -> Dict[str, Any]:
        """Evaluate models on batch correction task."""
        if metrics is None:
            metrics = ['silhouette_score', 'ari', 'biological_conservation']
        
        return self.runner.run_benchmark(
            models=models,
            datasets=datasets,
            tasks=['batch_correction'],
            metrics=metrics,
            n_runs=n_runs
        )


# Convenience functions
def quick_benchmark(model_name: str,
                   dataset_names: List[str],
                   task: str = 'cell_type_prediction',
                   n_runs: int = 3) -> Dict[str, Any]:
    """Quick benchmark for a single model on multiple datasets."""
    runner = BenchmarkRunner()
    
    models = {model_name: {}}
    metrics = ['accuracy', 'f1_score'] if task == 'cell_type_prediction' else ['mse', 'pearson_r']
    
    return runner.run_benchmark(
        models=models,
        datasets=dataset_names,
        tasks=[task],
        metrics=metrics,
        n_runs=n_runs
    )


def compare_baseline_models(dataset_names: List[str],
                           task: str = 'cell_type_prediction',
                           n_runs: int = 3) -> Dict[str, Any]:
    """Compare all baseline models on given datasets."""
    runner = BenchmarkRunner()
    
    # Use all available models
    models = {name: {} for name in MODEL_REGISTRY.keys()}
    metrics = ['accuracy', 'f1_score', 'silhouette_score']
    
    return runner.run_benchmark(
        models=models,
        datasets=dataset_names,
        tasks=[task],
        metrics=metrics,
        n_runs=n_runs
    )
