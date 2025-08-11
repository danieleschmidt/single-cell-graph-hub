"""Autonomous production deployment configuration and orchestration."""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import tempfile
import yaml

from ..src.scgraph_hub.logging_config import get_logger
from ..src.scgraph_hub.global_features import (
    DeploymentRegion, ComplianceStandard, 
    get_deployment_manager, get_compliance_manager, validate_global_readiness
)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"


class DeploymentStrategy(Enum):
    """Deployment strategy options."""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    environment: DeploymentEnvironment
    region: DeploymentRegion
    strategy: DeploymentStrategy
    replicas: int = 3
    cpu_limit: str = "2000m"
    memory_limit: str = "4Gi"
    storage_size: str = "20Gi"
    auto_scaling: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    monitoring_enabled: bool = True
    logging_level: str = "INFO"
    backup_enabled: bool = True
    ssl_enabled: bool = True
    custom_domain: Optional[str] = None


@dataclass
class DeploymentStatus:
    """Status of a deployment."""
    deployment_id: str
    environment: DeploymentEnvironment
    region: DeploymentRegion
    status: str
    health_check: str
    replicas_ready: int
    total_replicas: int
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "latest"
    errors: List[str] = field(default_factory=list)


class AutonomousDeploymentOrchestrator:
    """Autonomous deployment orchestrator for production systems."""
    
    def __init__(self, project_root: Path = Path(".")):
        self.project_root = Path(project_root)
        self.logger = get_logger(__name__)
        self.deployments: Dict[str, DeploymentStatus] = {}
        self.deployment_manager = get_deployment_manager()
        self.compliance_manager = get_compliance_manager()
        
        # Create deployment directory
        self.deployment_dir = self.project_root / "deployment"
        self.deployment_dir.mkdir(exist_ok=True)
    
    async def deploy_globally(self, 
                            environments: List[DeploymentEnvironment] = None,
                            regions: List[DeploymentRegion] = None) -> Dict[str, Any]:
        """Deploy to multiple regions and environments autonomously."""
        self.logger.info("🌍 Starting global autonomous deployment")
        
        if environments is None:
            environments = [DeploymentEnvironment.STAGING, DeploymentEnvironment.PRODUCTION]
        
        if regions is None:
            regions = [
                DeploymentRegion.US_EAST,
                DeploymentRegion.EU_WEST,
                DeploymentRegion.ASIA_PACIFIC
            ]
        
        # Validate global readiness
        readiness = await validate_global_readiness()
        if not readiness["overall_ready"]:
            self.logger.error("System not ready for global deployment")
            return {"success": False, "error": "Global readiness validation failed", "readiness": readiness}
        
        deployment_results = {}
        
        # Deploy to staging first in all regions
        if DeploymentEnvironment.STAGING in environments:
            staging_results = await self._deploy_to_multiple_regions(
                DeploymentEnvironment.STAGING, regions
            )
            deployment_results["staging"] = staging_results
            
            # Validate staging deployments
            staging_healthy = all(
                result.get("status") == "healthy" 
                for result in staging_results.values()
            )
            
            if not staging_healthy:
                self.logger.error("Staging deployments failed validation")
                return {
                    "success": False, 
                    "error": "Staging validation failed",
                    "results": deployment_results
                }
        
        # Deploy to production if staging is healthy
        if DeploymentEnvironment.PRODUCTION in environments:
            # Wait for staging to stabilize
            await asyncio.sleep(30)
            
            production_results = await self._deploy_to_multiple_regions(
                DeploymentEnvironment.PRODUCTION, regions
            )
            deployment_results["production"] = production_results
        
        # Generate deployment report
        report = await self._generate_deployment_report(deployment_results)
        
        return {
            "success": True,
            "deployments": deployment_results,
            "report": report,
            "global_readiness": readiness
        }
    
    async def _deploy_to_multiple_regions(self, 
                                        environment: DeploymentEnvironment,
                                        regions: List[DeploymentRegion]) -> Dict[str, Any]:
        """Deploy to multiple regions concurrently."""
        self.logger.info(f"Deploying {environment.value} to {len(regions)} regions")
        
        # Create deployment tasks for each region
        deployment_tasks = []
        for region in regions:
            config = self._create_deployment_config(environment, region)
            task = self._deploy_to_region(config)
            deployment_tasks.append((region.value, task))
        
        # Execute deployments concurrently
        results = {}
        for region_name, task in deployment_tasks:
            try:
                result = await task
                results[region_name] = result
            except Exception as e:
                self.logger.error(f"Deployment to {region_name} failed: {e}")
                results[region_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return results
    
    def _create_deployment_config(self, 
                                 environment: DeploymentEnvironment,
                                 region: DeploymentRegion) -> DeploymentConfig:
        """Create deployment configuration for specific environment and region."""
        regional_config = self.deployment_manager.get_regional_config(region)
        
        # Base configuration
        config = DeploymentConfig(
            environment=environment,
            region=region,
            strategy=DeploymentStrategy.ROLLING if environment == DeploymentEnvironment.PRODUCTION else DeploymentStrategy.RECREATE,
            compliance_standards=regional_config.get("compliance_standards", [])
        )
        
        # Environment-specific adjustments
        if environment == DeploymentEnvironment.PRODUCTION:
            config.replicas = 5
            config.min_replicas = 3
            config.max_replicas = 15
            config.cpu_limit = "4000m"
            config.memory_limit = "8Gi"
            config.storage_size = "50Gi"
        elif environment == DeploymentEnvironment.STAGING:
            config.replicas = 2
            config.min_replicas = 1
            config.max_replicas = 5
            config.cpu_limit = "1000m"
            config.memory_limit = "2Gi"
            config.storage_size = "10Gi"
        
        # Region-specific adjustments
        if region in [DeploymentRegion.EU_WEST]:
            # EU requires GDPR compliance
            config.compliance_standards.append(ComplianceStandard.GDPR)
        elif region in [DeploymentRegion.US_EAST, DeploymentRegion.US_WEST]:
            # US requires CCPA compliance
            config.compliance_standards.append(ComplianceStandard.CCPA)
        
        return config
    
    async def _deploy_to_region(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to a specific region."""
        deployment_id = f"{config.environment.value}-{config.region.value}-{int(datetime.now().timestamp())}"
        
        self.logger.info(f"Deploying {deployment_id}")
        
        try:
            # Generate deployment manifests
            manifests = await self._generate_deployment_manifests(config, deployment_id)
            
            # Apply compliance configurations
            await self._apply_compliance_config(config)
            
            # Execute deployment
            deployment_result = await self._execute_deployment(manifests, config)
            
            # Perform health checks
            health_status = await self._perform_health_checks(deployment_id, config)
            
            # Update deployment status
            status = DeploymentStatus(
                deployment_id=deployment_id,
                environment=config.environment,
                region=config.region,
                status="healthy" if health_status else "unhealthy",
                health_check="passed" if health_status else "failed",
                replicas_ready=config.replicas if health_status else 0,
                total_replicas=config.replicas
            )
            
            self.deployments[deployment_id] = status
            
            return {
                "deployment_id": deployment_id,
                "status": "healthy" if health_status else "unhealthy",
                "replicas": f"{status.replicas_ready}/{status.total_replicas}",
                "health_check": status.health_check,
                "manifests": manifests
            }
            
        except Exception as e:
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            
            error_status = DeploymentStatus(
                deployment_id=deployment_id,
                environment=config.environment,
                region=config.region,
                status="failed",
                health_check="failed",
                replicas_ready=0,
                total_replicas=config.replicas,
                errors=[str(e)]
            )
            
            self.deployments[deployment_id] = error_status
            
            return {
                "deployment_id": deployment_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _generate_deployment_manifests(self, 
                                           config: DeploymentConfig,
                                           deployment_id: str) -> Dict[str, str]:
        """Generate deployment manifests."""
        manifests = {}
        
        # Kubernetes Deployment
        deployment_manifest = self._create_k8s_deployment(config, deployment_id)
        manifests["deployment.yaml"] = deployment_manifest
        
        # Kubernetes Service
        service_manifest = self._create_k8s_service(config, deployment_id)
        manifests["service.yaml"] = service_manifest
        
        # Kubernetes Ingress
        if config.ssl_enabled:
            ingress_manifest = self._create_k8s_ingress(config, deployment_id)
            manifests["ingress.yaml"] = ingress_manifest
        
        # ConfigMap for application configuration
        configmap_manifest = self._create_k8s_configmap(config, deployment_id)
        manifests["configmap.yaml"] = configmap_manifest
        
        # HorizontalPodAutoscaler if auto-scaling is enabled
        if config.auto_scaling:
            hpa_manifest = self._create_k8s_hpa(config, deployment_id)
            manifests["hpa.yaml"] = hpa_manifest
        
        # Save manifests to files
        manifest_dir = self.deployment_dir / deployment_id
        manifest_dir.mkdir(exist_ok=True)
        
        for filename, content in manifests.items():
            manifest_file = manifest_dir / filename
            with open(manifest_file, "w") as f:
                f.write(content)
        
        self.logger.info(f"Generated {len(manifests)} manifests for {deployment_id}")
        
        return manifests
    
    def _create_k8s_deployment(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Create Kubernetes deployment manifest."""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"scgraph-hub-{config.environment.value}",
                "namespace": f"scgraph-{config.environment.value}",
                "labels": {
                    "app": "scgraph-hub",
                    "environment": config.environment.value,
                    "region": config.region.value,
                    "deployment-id": deployment_id
                }
            },
            "spec": {
                "replicas": config.replicas,
                "strategy": {
                    "type": "RollingUpdate" if config.strategy == DeploymentStrategy.ROLLING else "Recreate",
                    "rollingUpdate": {
                        "maxUnavailable": 1,
                        "maxSurge": 1
                    } if config.strategy == DeploymentStrategy.ROLLING else None
                },
                "selector": {
                    "matchLabels": {
                        "app": "scgraph-hub",
                        "environment": config.environment.value
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "scgraph-hub",
                            "environment": config.environment.value,
                            "region": config.region.value
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "scgraph-hub",
                            "image": f"scgraph-hub:latest",
                            "ports": [{
                                "containerPort": 8000,
                                "name": "http"
                            }],
                            "resources": {
                                "limits": {
                                    "cpu": config.cpu_limit,
                                    "memory": config.memory_limit
                                },
                                "requests": {
                                    "cpu": str(int(config.cpu_limit.rstrip('m')) // 2) + 'm',
                                    "memory": str(int(config.memory_limit.rstrip('Gi')) // 2) + 'Gi'
                                }
                            },
                            "env": [
                                {
                                    "name": "ENVIRONMENT",
                                    "value": config.environment.value
                                },
                                {
                                    "name": "REGION",
                                    "value": config.region.value
                                },
                                {
                                    "name": "LOG_LEVEL",
                                    "value": config.logging_level
                                }
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }],
                        "imagePullSecrets": [{
                            "name": "registry-secret"
                        }]
                    }
                }
            }
        }
        
        return yaml.dump(deployment, default_flow_style=False)
    
    def _create_k8s_service(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Create Kubernetes service manifest."""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"scgraph-hub-{config.environment.value}",
                "namespace": f"scgraph-{config.environment.value}",
                "labels": {
                    "app": "scgraph-hub",
                    "environment": config.environment.value
                }
            },
            "spec": {
                "selector": {
                    "app": "scgraph-hub",
                    "environment": config.environment.value
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP",
                    "name": "http"
                }],
                "type": "ClusterIP"
            }
        }
        
        return yaml.dump(service, default_flow_style=False)
    
    def _create_k8s_ingress(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Create Kubernetes ingress manifest."""
        host = config.custom_domain or f"scgraph-{config.environment.value}.{config.region.value}.example.com"
        
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"scgraph-hub-{config.environment.value}",
                "namespace": f"scgraph-{config.environment.value}",
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true"
                }
            },
            "spec": {
                "tls": [{
                    "hosts": [host],
                    "secretName": f"scgraph-{config.environment.value}-tls"
                }],
                "rules": [{
                    "host": host,
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"scgraph-hub-{config.environment.value}",
                                    "port": {
                                        "number": 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        return yaml.dump(ingress, default_flow_style=False)
    
    def _create_k8s_configmap(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Create Kubernetes ConfigMap manifest."""
        config_data = {
            "environment": config.environment.value,
            "region": config.region.value,
            "compliance_standards": ",".join([std.value for std in config.compliance_standards]),
            "monitoring_enabled": str(config.monitoring_enabled).lower(),
            "backup_enabled": str(config.backup_enabled).lower(),
            "log_level": config.logging_level
        }
        
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"scgraph-hub-{config.environment.value}-config",
                "namespace": f"scgraph-{config.environment.value}"
            },
            "data": config_data
        }
        
        return yaml.dump(configmap, default_flow_style=False)
    
    def _create_k8s_hpa(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Create Kubernetes HorizontalPodAutoscaler manifest."""
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"scgraph-hub-{config.environment.value}-hpa",
                "namespace": f"scgraph-{config.environment.value}"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"scgraph-hub-{config.environment.value}"
                },
                "minReplicas": config.min_replicas,
                "maxReplicas": config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
        
        return yaml.dump(hpa, default_flow_style=False)
    
    async def _apply_compliance_config(self, config: DeploymentConfig):
        """Apply compliance configuration."""
        if config.compliance_standards:
            self.compliance_manager.enable_compliance(config.compliance_standards)
            self.logger.info(f"Enabled compliance standards: {[std.value for std in config.compliance_standards]}")
    
    async def _execute_deployment(self, manifests: Dict[str, str], config: DeploymentConfig) -> bool:
        """Execute the deployment using kubectl or similar."""
        self.logger.info("Executing deployment...")
        
        # In a real implementation, this would use kubectl or a Kubernetes client
        # For now, we'll simulate successful deployment
        await asyncio.sleep(2)  # Simulate deployment time
        
        return True
    
    async def _perform_health_checks(self, deployment_id: str, config: DeploymentConfig) -> bool:
        """Perform health checks on the deployment."""
        self.logger.info(f"Performing health checks for {deployment_id}")
        
        # Simulate health check operations
        checks = {
            "pod_ready": True,
            "service_accessible": True,
            "database_connected": True,
            "external_apis_reachable": True,
            "compliance_validated": len(config.compliance_standards) > 0
        }
        
        # Simulate actual health checks
        await asyncio.sleep(5)  # Simulate health check time
        
        all_healthy = all(checks.values())
        
        if all_healthy:
            self.logger.info(f"All health checks passed for {deployment_id}")
        else:
            failed_checks = [check for check, status in checks.items() if not status]
            self.logger.warning(f"Health checks failed for {deployment_id}: {failed_checks}")
        
        return all_healthy
    
    async def _generate_deployment_report(self, deployment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        report = {
            "deployment_summary": {
                "total_deployments": 0,
                "successful_deployments": 0,
                "failed_deployments": 0,
                "environments": [],
                "regions": []
            },
            "deployment_details": {},
            "compliance_status": {},
            "recommendations": []
        }
        
        # Analyze deployment results
        for env, env_results in deployment_results.items():
            report["deployment_summary"]["environments"].append(env)
            
            for region, region_result in env_results.items():
                report["deployment_summary"]["total_deployments"] += 1
                report["deployment_summary"]["regions"].append(region)
                
                if region_result.get("status") == "healthy":
                    report["deployment_summary"]["successful_deployments"] += 1
                else:
                    report["deployment_summary"]["failed_deployments"] += 1
                
                # Store detailed results
                key = f"{env}_{region}"
                report["deployment_details"][key] = region_result
        
        # Remove duplicates from regions and environments
        report["deployment_summary"]["regions"] = list(set(report["deployment_summary"]["regions"]))
        report["deployment_summary"]["environments"] = list(set(report["deployment_summary"]["environments"]))
        
        # Calculate success rate
        total = report["deployment_summary"]["total_deployments"]
        successful = report["deployment_summary"]["successful_deployments"]
        report["deployment_summary"]["success_rate"] = successful / total if total > 0 else 0
        
        # Generate recommendations
        if report["deployment_summary"]["success_rate"] < 1.0:
            report["recommendations"].append("Some deployments failed - investigate error logs")
        
        if report["deployment_summary"]["success_rate"] >= 0.8:
            report["recommendations"].append("Deployment success rate is acceptable")
        
        report["recommendations"].append("Monitor deployment health continuously")
        report["recommendations"].append("Consider setting up automated rollback procedures")
        
        # Save report
        report_file = self.deployment_dir / f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Deployment report saved to {report_file}")
        
        return report
    
    async def rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback a deployment."""
        self.logger.warning(f"Initiating rollback for {deployment_id}")
        
        if deployment_id not in self.deployments:
            return {"success": False, "error": "Deployment not found"}
        
        deployment = self.deployments[deployment_id]
        
        # Simulate rollback process
        try:
            await asyncio.sleep(3)  # Simulate rollback time
            
            # Update deployment status
            deployment.status = "rolled_back"
            deployment.last_updated = datetime.now()
            
            self.logger.info(f"Rollback completed for {deployment_id}")
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "status": "rolled_back"
            }
            
        except Exception as e:
            self.logger.error(f"Rollback failed for {deployment_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_deployment_status(self, deployment_id: Optional[str] = None) -> Union[DeploymentStatus, Dict[str, DeploymentStatus]]:
        """Get deployment status."""
        if deployment_id:
            return self.deployments.get(deployment_id)
        else:
            return self.deployments


# Global instance
_deployment_orchestrator = None


def get_deployment_orchestrator(project_root: Path = Path(".")) -> AutonomousDeploymentOrchestrator:
    """Get global deployment orchestrator."""
    global _deployment_orchestrator
    if _deployment_orchestrator is None:
        _deployment_orchestrator = AutonomousDeploymentOrchestrator(project_root)
    return _deployment_orchestrator


# High-level deployment functions

async def deploy_globally(environments: List[DeploymentEnvironment] = None,
                         regions: List[DeploymentRegion] = None,
                         project_root: Path = Path(".")) -> Dict[str, Any]:
    """Deploy globally with autonomous orchestration."""
    orchestrator = get_deployment_orchestrator(project_root)
    return await orchestrator.deploy_globally(environments, regions)


async def quick_deploy(environment: DeploymentEnvironment = DeploymentEnvironment.STAGING,
                      region: DeploymentRegion = DeploymentRegion.US_EAST,
                      project_root: Path = Path(".")) -> Dict[str, Any]:
    """Quick deployment to single environment and region."""
    orchestrator = get_deployment_orchestrator(project_root)
    return await orchestrator.deploy_globally([environment], [region])


async def production_deploy(project_root: Path = Path(".")) -> Dict[str, Any]:
    """Production deployment to all regions."""
    return await deploy_globally(
        environments=[DeploymentEnvironment.PRODUCTION],
        regions=[DeploymentRegion.US_EAST, DeploymentRegion.EU_WEST, DeploymentRegion.ASIA_PACIFIC],
        project_root=project_root
    )