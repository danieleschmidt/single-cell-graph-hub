# Deployment Guide

This document provides comprehensive deployment instructions for Single-Cell Graph Hub across different environments.

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/single-cell-graph-hub
cd single-cell-graph-hub

# Install dependencies
pip install -e ".[dev]"

# Run development server
python -m scgraph_hub.cli --help
```

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Access services
# - Main application: http://localhost:8000
# - Jupyter Lab: http://localhost:8888
# - Grafana: http://localhost:3000
# - MinIO Console: http://localhost:9001
```

## Deployment Options

### 1. Docker Deployment

#### Single Container

```bash
# Build image
docker build -t single-cell-graph-hub .

# Run container
docker run -d \
  --name scgraph-hub \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  single-cell-graph-hub
```

#### Multi-Container with Docker Compose

```bash
# Production setup
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Development setup
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Scale application
docker-compose up -d --scale scgraph-hub=3
```

### 2. Kubernetes Deployment

#### Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured
- Helm 3.x (optional)

#### Basic Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=scgraph-hub

# Access via port-forward
kubectl port-forward svc/scgraph-hub-service 8000:8000
```

#### Helm Deployment

```bash
# Install with Helm
helm install scgraph-hub ./helm/scgraph-hub

# Upgrade deployment
helm upgrade scgraph-hub ./helm/scgraph-hub

# Uninstall
helm uninstall scgraph-hub
```

### 3. Cloud Deployments

#### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-west-2.amazonaws.com

docker build -t single-cell-graph-hub .
docker tag single-cell-graph-hub:latest <account>.dkr.ecr.us-west-2.amazonaws.com/single-cell-graph-hub:latest
docker push <account>.dkr.ecr.us-west-2.amazonaws.com/single-cell-graph-hub:latest

# Deploy with ECS CLI or AWS Console
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/single-cell-graph-hub

gcloud run deploy single-cell-graph-hub \
  --image gcr.io/PROJECT-ID/single-cell-graph-hub \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name scgraph-hub-rg --location eastus

# Deploy container
az container create \
  --resource-group scgraph-hub-rg \
  --name scgraph-hub \
  --image single-cell-graph-hub:latest \
  --ports 8000 \
  --memory 4 \
  --cpu 2
```

## Configuration

### Environment Variables

```bash
# Core configuration
SCGRAPH_DATA_DIR=/app/data
SCGRAPH_CACHE_DIR=/app/cache
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/scgraph_hub
REDIS_URL=redis://localhost:6379/0

# Storage
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET_NAME=scgraph-hub-data

# Security
SECRET_KEY=your_secret_key
JWT_EXPIRATION_HOURS=24

# Monitoring
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_PORT=9090
```

### Configuration Files

Create `config/production.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

database:
  url: "${DATABASE_URL}"
  pool_size: 20
  max_overflow: 30

cache:
  url: "${REDIS_URL}"
  ttl: 3600

storage:
  backend: "s3"
  bucket: "${S3_BUCKET_NAME}"
  region: "us-west-2"

logging:
  level: "INFO"
  format: "json"
  
monitoring:
  metrics_enabled: true
  tracing_enabled: true
  health_check_interval: 30
```

## Performance Tuning

### Resource Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **Memory**: 4 GB RAM
- **Storage**: 20 GB
- **Network**: 100 Mbps

#### Recommended (Production)
- **CPU**: 8 cores
- **Memory**: 16 GB RAM
- **Storage**: 100 GB SSD
- **Network**: 1 Gbps
- **GPU**: CUDA-compatible (optional)

#### Large Scale Deployment
- **CPU**: 16+ cores
- **Memory**: 32+ GB RAM
- **Storage**: 500+ GB NVMe SSD
- **Network**: 10+ Gbps
- **GPU**: Multiple CUDA GPUs

### Optimization Settings

#### Docker Resource Limits

```yaml
services:
  scgraph-hub:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

#### Database Tuning

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

#### Redis Configuration

```conf
# redis.conf optimizations
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Security

### Network Security

```bash
# Firewall rules (UFW example)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # Application
sudo ufw enable
```

### SSL/TLS Configuration

#### Let's Encrypt with Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name scgraphhub.example.com;
    
    ssl_certificate /etc/letsencrypt/live/scgraphhub.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/scgraphhub.example.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Access Control

```yaml
# docker-compose.override.yml
services:
  scgraph-hub:
    environment:
      - AUTH_ENABLED=true
      - ALLOWED_HOSTS=scgraphhub.example.com
      - CORS_ORIGINS=https://scgraphhub.example.com
```

## Monitoring and Logging

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database health
docker exec scgraph-postgres pg_isready -U scgraph

# Redis health
docker exec scgraph-redis redis-cli ping
```

### Metrics Collection

Prometheus metrics are available at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'scgraph-hub'
    static_configs:
      - targets: ['scgraph-hub:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Log Aggregation

#### ELK Stack Integration

```yaml
services:
  scgraph-hub:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=scgraph-hub"
```

#### Structured Logging

```python
import structlog

logger = structlog.get_logger()
logger.info("Processing dataset", dataset_id="pbmc_10k", n_cells=10000)
```

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup/postgres"
DATE=$(date +%Y%m%d_%H%M%S)

docker exec scgraph-postgres pg_dump -U scgraph scgraph_hub > \
  "${BACKUP_DIR}/scgraph_hub_${DATE}.sql"

# Keep only last 7 days
find "${BACKUP_DIR}" -name "*.sql" -mtime +7 -delete
```

### Data Backup

```bash
# Backup data volumes
docker run --rm \
  -v scgraph_postgres-data:/data \
  -v $(pwd)/backup:/backup \
  ubuntu tar czf /backup/postgres-data-$(date +%Y%m%d).tar.gz /data

# Backup application data
rsync -av --delete /app/data/ /backup/scgraph-data/
```

### Disaster Recovery

```bash
# Restore database
docker exec -i scgraph-postgres psql -U scgraph scgraph_hub < backup.sql

# Restore data volumes
docker run --rm \
  -v scgraph_postgres-data:/data \
  -v $(pwd)/backup:/backup \
  ubuntu tar xzf /backup/postgres-data-20240101.tar.gz -C /
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
services:
  scgraph-hub:
    deploy:
      replicas: 3
      
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - scgraph-hub
```

### Load Balancing

```nginx
# nginx.conf
upstream scgraph_backend {
    server scgraph-hub-1:8000;
    server scgraph-hub-2:8000;
    server scgraph-hub-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://scgraph_backend;
    }
}
```

### Auto-scaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: scgraph-hub-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: scgraph-hub
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker logs scgraph-hub

# Check resource usage
docker stats scgraph-hub

# Verify configuration
docker exec scgraph-hub env | grep SCGRAPH
```

#### Database Connection Issues
```bash
# Test database connectivity
docker exec scgraph-hub pg_isready -h postgres -p 5432 -U scgraph

# Check database logs
docker logs scgraph-postgres
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check application metrics
curl http://localhost:8000/metrics

# Profile application
docker exec scgraph-hub python -m cProfile -o profile.stats app.py
```

### Debug Mode

```bash
# Enable debug mode
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up -d

# Access debug information
curl http://localhost:8000/debug/info
```

## Maintenance

### Updates

```bash
# Update images
docker-compose pull

# Restart services
docker-compose up -d

# Health check after update
./scripts/health-check.sh
```

### Cleanup

```bash
# Clean unused Docker resources
docker system prune -f

# Clean application cache
docker exec scgraph-hub python -m scgraph_hub.cli cache clear

# Rotate logs
docker exec scgraph-hub logrotate -f /etc/logrotate.conf
```

## Support

For deployment issues:

1. Check the [troubleshooting guide](../troubleshooting/README.md)
2. Review application logs
3. Consult the [FAQ](../FAQ.md)
4. Open an issue on [GitHub](https://github.com/danieleschmidt/single-cell-graph-hub/issues)

## Best Practices

1. **Always use specific image tags** in production
2. **Implement proper health checks** for all services
3. **Use secrets management** for sensitive data
4. **Monitor resource usage** and set appropriate limits
5. **Implement automated backups** and test recovery procedures
6. **Use staging environments** to test deployments
7. **Keep security patches up to date**
8. **Document your specific configuration** and customizations