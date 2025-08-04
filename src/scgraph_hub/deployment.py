"""Production deployment utilities for Single-Cell Graph Hub."""

import logging
import os
import json
import time
import subprocess
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

import torch
import torch.nn as nn
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class ModelDeployment:
    """Handle model deployment and serving."""
    
    def __init__(self, deployment_dir: Union[str, Path]):
        """Initialize model deployment.
        
        Args:
            deployment_dir: Directory for deployment artifacts
        """
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_registry = {}
        self.deployment_config = {}
        
        logger.info(f"Model deployment initialized at {self.deployment_dir}")
    
    def prepare_model_for_deployment(self, 
                                   model: nn.Module,
                                   model_name: str,
                                   sample_input: Data,
                                   optimization_level: str = 'standard') -> Dict[str, Any]:
        """Prepare model for production deployment.
        
        Args:
            model: Trained model
            model_name: Name for the deployed model
            sample_input: Sample input for optimization
            optimization_level: Optimization level ('basic', 'standard', 'aggressive')
            
        Returns:
            Deployment information
        """
        model_dir = self.deployment_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        deployment_info = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'optimization_level': optimization_level,
            'artifacts': {}
        }
        
        # Optimize model based on level
        optimized_model = self._optimize_model(model, sample_input, optimization_level)
        
        # Save model artifacts
        artifacts = self._save_model_artifacts(optimized_model, model_dir, sample_input)
        deployment_info['artifacts'] = artifacts
        
        # Generate deployment configuration
        config = self._generate_deployment_config(model_name, optimized_model, sample_input)
        config_path = model_dir / 'deployment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        deployment_info['config_path'] = str(config_path)
        
        # Register model
        self.model_registry[model_name] = deployment_info
        
        logger.info(f"Model {model_name} prepared for deployment")
        return deployment_info
    
    def _optimize_model(self, model: nn.Module, sample_input: Data, level: str) -> nn.Module:
        """Optimize model for deployment.
        
        Args:
            model: Model to optimize
            sample_input: Sample input for optimization
            level: Optimization level
            
        Returns:
            Optimized model
        """
        model.eval()
        
        if level == 'basic':
            # Basic optimizations
            return model
        
        elif level == 'standard':
            # Standard optimizations: TorchScript tracing
            try:
                with torch.no_grad():
                    traced_model = torch.jit.trace(model, (sample_input.x, sample_input.edge_index))
                logger.info("Model traced with TorchScript")
                return traced_model
            except Exception as e:
                logger.warning(f"TorchScript tracing failed: {e}, using original model")
                return model
        
        elif level == 'aggressive':
            # Aggressive optimizations: quantization + tracing
            try:
                # Dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                
                # Try to trace quantized model
                with torch.no_grad():
                    traced_model = torch.jit.trace(quantized_model, (sample_input.x, sample_input.edge_index))
                
                logger.info("Model quantized and traced")
                return traced_model
            
            except Exception as e:
                logger.warning(f"Aggressive optimization failed: {e}, falling back to standard")
                return self._optimize_model(model, sample_input, 'standard')
        
        else:
            raise ValueError(f"Unknown optimization level: {level}")
    
    def _save_model_artifacts(self, model: nn.Module, model_dir: Path, sample_input: Data) -> Dict[str, str]:
        """Save model artifacts for deployment.
        
        Args:
            model: Optimized model
            model_dir: Directory to save artifacts
            sample_input: Sample input for testing
            
        Returns:
            Dictionary mapping artifact names to paths
        """
        artifacts = {}
        
        # Save model
        model_path = model_dir / 'model.pt'
        torch.save(model, model_path)
        artifacts['model'] = str(model_path)
        
        # Save model state dict
        state_dict_path = model_dir / 'model_state_dict.pt'
        torch.save(model.state_dict(), state_dict_path)
        artifacts['state_dict'] = str(state_dict_path)
        
        # Save sample input for testing
        sample_input_path = model_dir / 'sample_input.pt'
        torch.save(sample_input, sample_input_path)
        artifacts['sample_input'] = str(sample_input_path)
        
        # Generate ONNX export if possible
        try:
            onnx_path = model_dir / 'model.onnx'
            torch.onnx.export(
                model,
                (sample_input.x, sample_input.edge_index),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['node_features', 'edge_index'],
                output_names=['output']
            )
            artifacts['onnx'] = str(onnx_path)
            logger.info("ONNX model exported")
        
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
        
        return artifacts
    
    def _generate_deployment_config(self, model_name: str, model: nn.Module, sample_input: Data) -> Dict[str, Any]:
        """Generate deployment configuration.
        
        Args:
            model_name: Name of the model
            model: Optimized model
            sample_input: Sample input
            
        Returns:
            Deployment configuration
        """
        # Run inference to get output shape and timing
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            output = model(sample_input.x, sample_input.edge_index)
            inference_time = time.time() - start_time
        
        config = {
            'model_name': model_name,
            'input_shape': {
                'node_features': list(sample_input.x.shape),
                'edge_index': list(sample_input.edge_index.shape)
            },
            'output_shape': list(output.shape),
            'inference_time_ms': inference_time * 1000,
            'model_size_mb': self._estimate_model_size(model),
            'requirements': self._get_model_requirements(),
            'serving': {
                'max_batch_size': 32,
                'timeout_seconds': 30,
                'memory_limit_mb': 2048
            }
        }
        
        return config
    
    def _estimate_model_size(self, model: nn.Module) -> float:
        """Estimate model size in MB.
        
        Args:
            model: Model to estimate
            
        Returns:
            Estimated size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)
    
    def _get_model_requirements(self) -> Dict[str, str]:
        """Get model deployment requirements.
        
        Returns:
            Requirements dictionary
        """
        return {
            'torch': torch.__version__,
            'python': f">={'.'.join(map(str, [3, 8]))}",
            'cpu_cores': '2',
            'memory_gb': '4'
        }
    
    def create_docker_deployment(self, model_name: str, base_image: str = 'python:3.9-slim') -> str:
        """Create Docker deployment for model.
        
        Args:
            model_name: Name of the model to deploy
            base_image: Base Docker image
            
        Returns:
            Path to generated Dockerfile
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_dir = self.deployment_dir / model_name
        dockerfile_path = model_dir / 'Dockerfile'
        
        # Generate Dockerfile
        dockerfile_content = self._generate_dockerfile(model_name, base_image)
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Generate requirements.txt
        requirements_path = model_dir / 'requirements.txt'
        requirements_content = self._generate_requirements()
        
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        # Generate serving script
        serving_script_path = model_dir / 'serve.py'
        serving_script_content = self._generate_serving_script(model_name)
        
        with open(serving_script_path, 'w') as f:
            f.write(serving_script_content)
        
        logger.info(f"Docker deployment created for {model_name}")
        return str(dockerfile_path)
    
    def _generate_dockerfile(self, model_name: str, base_image: str) -> str:
        """Generate Dockerfile content.
        
        Args:
            model_name: Name of the model
            base_image: Base Docker image
            
        Returns:
            Dockerfile content
        """
        return f"""FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY model.pt .
COPY deployment_config.json .
COPY serve.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run serving script
CMD ["python", "serve.py"]
"""
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt content.
        
        Returns:
            Requirements content
        """
        return """torch>=1.9.0
torch-geometric>=2.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
numpy>=1.19.0
pandas>=1.3.0
scikit-learn>=0.24.0
"""
    
    def _generate_serving_script(self, model_name: str) -> str:
        """Generate serving script.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Serving script content
        """
        return f"""#!/usr/bin/env python3
"""Model serving script for {model_name}."""

import json
import logging
import time
from typing import Dict, Any, List

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and configuration
model = torch.load('model.pt', map_location='cpu')
model.eval()

with open('deployment_config.json', 'r') as f:
    config = json.load(f)

app = FastAPI(title=f"{model_name} API", version="1.0.0")

class PredictionRequest(BaseModel):
    node_features: List[List[float]]
    edge_index: List[List[int]]

class PredictionResponse(BaseModel):
    predictions: List[List[float]]
    inference_time_ms: float
    model_name: str

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {{"status": "healthy", "model": "{model_name}", "timestamp": time.time()}}

@app.get("/info")
async def model_info():
    """Get model information."""
    return config

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions."""
    try:
        # Convert input to tensors
        node_features = torch.tensor(request.node_features, dtype=torch.float32)
        edge_index = torch.tensor(request.edge_index, dtype=torch.long)
        
        # Validate input shapes
        expected_node_shape = config['input_shape']['node_features']
        expected_edge_shape = config['input_shape']['edge_index']
        
        if node_features.shape[1] != expected_node_shape[1]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid node features shape. Expected {{expected_node_shape}}, got {{list(node_features.shape)}}"
            )
        
        if edge_index.shape[0] != expected_edge_shape[0]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid edge index shape. Expected {{expected_edge_shape}}, got {{list(edge_index.shape)}}"
            )
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            predictions = model(node_features, edge_index)
        inference_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            inference_time_ms=inference_time,
            model_name="{model_name}"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
"""
    
    def build_docker_image(self, model_name: str, tag: Optional[str] = None) -> str:
        """Build Docker image for model.
        
        Args:
            model_name: Name of the model
            tag: Docker image tag
            
        Returns:
            Built image tag
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if tag is None:
            tag = f"scgraph-hub-{model_name}:latest"
        
        model_dir = self.deployment_dir / model_name
        
        # Build Docker image
        try:
            cmd = ["docker", "build", "-t", tag, str(model_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            logger.info(f"Docker image built successfully: {tag}")
            return tag
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e.stderr}")
            raise RuntimeError(f"Docker build failed: {e.stderr}")
    
    def create_kubernetes_deployment(self, model_name: str, image_tag: str, replicas: int = 3) -> str:
        """Create Kubernetes deployment configuration.
        
        Args:
            model_name: Name of the model
            image_tag: Docker image tag
            replicas: Number of replicas
            
        Returns:
            Path to generated Kubernetes YAML
        """
        model_dir = self.deployment_dir / model_name
        k8s_dir = model_dir / 'kubernetes'
        k8s_dir.mkdir(exist_ok=True)
        
        # Generate deployment YAML
        deployment_yaml = self._generate_k8s_deployment(model_name, image_tag, replicas)
        deployment_path = k8s_dir / 'deployment.yaml'
        
        with open(deployment_path, 'w') as f:
            f.write(deployment_yaml)
        
        # Generate service YAML
        service_yaml = self._generate_k8s_service(model_name)
        service_path = k8s_dir / 'service.yaml'
        
        with open(service_path, 'w') as f:
            f.write(service_yaml)
        
        logger.info(f"Kubernetes deployment created for {model_name}")
        return str(deployment_path)
    
    def _generate_k8s_deployment(self, model_name: str, image_tag: str, replicas: int) -> str:
        """Generate Kubernetes deployment YAML.
        
        Args:
            model_name: Name of the model
            image_tag: Docker image tag
            replicas: Number of replicas
            
        Returns:
            Deployment YAML content
        """
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_name}-deployment
  labels:
    app: {model_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {model_name}
  template:
    metadata:
      labels:
        app: {model_name}
    spec:
      containers:
      - name: {model_name}
        image: {image_tag}
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: MODEL_NAME
          value: "{model_name}"
"""
    
    def _generate_k8s_service(self, model_name: str) -> str:
        """Generate Kubernetes service YAML.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Service YAML content
        """
        return f"""apiVersion: v1
kind: Service
metadata:
  name: {model_name}-service
spec:
  selector:
    app: {model_name}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
"""
    
    def test_deployment(self, model_name: str) -> Dict[str, Any]:
        """Test deployed model.
        
        Args:
            model_name: Name of the model to test
            
        Returns:
            Test results
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        deployment_info = self.model_registry[model_name]
        model_path = deployment_info['artifacts']['model']
        sample_input_path = deployment_info['artifacts']['sample_input']
        
        # Load model and sample input
        model = torch.load(model_path, map_location='cpu')
        sample_input = torch.load(sample_input_path, map_location='cpu')
        
        # Run test inference
        model.eval()
        test_results = {
            'model_name': model_name,
            'test_timestamp': datetime.now().isoformat(),
            'tests': []
        }
        
        # Test basic inference
        try:
            start_time = time.time()
            with torch.no_grad():
                output = model(sample_input.x, sample_input.edge_index)
            inference_time = time.time() - start_time
            
            test_results['tests'].append({
                'test_name': 'basic_inference',
                'status': 'passed',
                'inference_time_ms': inference_time * 1000,
                'output_shape': list(output.shape)
            })
        
        except Exception as e:
            test_results['tests'].append({
                'test_name': 'basic_inference',
                'status': 'failed',
                'error': str(e)
            })
        
        # Test with different batch sizes
        for batch_size in [1, 8, 32]:
            try:
                # Create batch
                batch_data = [sample_input] * batch_size
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(batch_data)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(batch.x, batch.edge_index)
                inference_time = time.time() - start_time
                
                test_results['tests'].append({
                    'test_name': f'batch_inference_size_{batch_size}',
                    'status': 'passed',
                    'inference_time_ms': inference_time * 1000,
                    'throughput_samples_per_second': batch_size / inference_time
                })
            
            except Exception as e:
                test_results['tests'].append({
                    'test_name': f'batch_inference_size_{batch_size}',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save test results
        test_results_path = self.deployment_dir / model_name / 'test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Deployment tests completed for {model_name}")
        return test_results
    
    def get_deployment_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get deployment status.
        
        Args:
            model_name: Name of specific model (all models if None)
            
        Returns:
            Deployment status
        """
        if model_name:
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not found in registry")
            return self.model_registry[model_name]
        else:
            return {
                'deployment_dir': str(self.deployment_dir),
                'total_models': len(self.model_registry),
                'models': list(self.model_registry.keys()),
                'registry': self.model_registry
            }


# Convenience functions
def deploy_model(model: nn.Module,
                model_name: str,
                sample_input: Data,
                deployment_dir: Union[str, Path],
                optimization_level: str = 'standard',
                create_docker: bool = True,
                create_k8s: bool = False) -> Dict[str, Any]:
    """Deploy model with single function call.
    
    Args:
        model: Trained model
        model_name: Name for the deployed model
        sample_input: Sample input for optimization
        deployment_dir: Directory for deployment artifacts
        optimization_level: Optimization level
        create_docker: Whether to create Docker deployment
        create_k8s: Whether to create Kubernetes deployment
        
    Returns:
        Deployment information
    """
    deployment = ModelDeployment(deployment_dir)
    
    # Prepare model
    deployment_info = deployment.prepare_model_for_deployment(
        model, model_name, sample_input, optimization_level
    )
    
    # Create Docker deployment
    if create_docker:
        dockerfile_path = deployment.create_docker_deployment(model_name)
        deployment_info['dockerfile'] = dockerfile_path
        
        # Build Docker image
        try:
            image_tag = deployment.build_docker_image(model_name)
            deployment_info['docker_image'] = image_tag
        except Exception as e:
            logger.warning(f"Docker build failed: {e}")
    
    # Create Kubernetes deployment
    if create_k8s and create_docker and 'docker_image' in deployment_info:
        try:
            k8s_path = deployment.create_kubernetes_deployment(
                model_name, deployment_info['docker_image']
            )
            deployment_info['kubernetes_deployment'] = k8s_path
        except Exception as e:
            logger.warning(f"Kubernetes deployment creation failed: {e}")
    
    # Test deployment
    test_results = deployment.test_deployment(model_name)
    deployment_info['test_results'] = test_results
    
    return deployment_info
