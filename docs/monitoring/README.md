# Monitoring and Observability

This document describes the comprehensive monitoring and observability setup for Single-Cell Graph Hub.

## Overview

Our monitoring stack provides:

- **Application Metrics**: Performance, usage, and business metrics
- **Infrastructure Metrics**: System resources, containers, databases
- **Distributed Tracing**: Request flow across services
- **Structured Logging**: Centralized log collection and analysis
- **Health Checks**: Service availability monitoring
- **Alerting**: Proactive issue detection and notification

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Application   │───▶│  Prometheus  │───▶│    Grafana      │
│   (Metrics)     │    │  (Collection)│    │ (Visualization) │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                      │
┌─────────────────┐    ┌──────────────┐              │
│     Logs        │───▶│   ELK Stack  │              │
│ (Application)   │    │  (Analysis)  │              │
└─────────────────┘    └──────────────┘              │
                                                      │
┌─────────────────┐    ┌──────────────┐              │
│    Traces       │───▶│    Jaeger    │              │
│  (Distributed)  │    │  (Tracing)   │              │
└─────────────────┘    └──────────────┘              │
                                                      │
                       ┌──────────────┐              │
                       │  AlertManager│◀─────────────┘
                       │  (Alerting)  │
                       └──────────────┘
```

## Metrics Collection

### Application Metrics

The application exposes metrics at `/metrics` endpoint in Prometheus format:

#### Business Metrics
```python
# Dataset usage metrics
dataset_downloads_total = Counter('scgraph_dataset_downloads_total', 'Total dataset downloads', ['dataset_name'])
dataset_processing_duration = Histogram('scgraph_dataset_processing_seconds', 'Dataset processing time', ['dataset_name'])

# Model training metrics  
model_training_duration = Histogram('scgraph_model_training_seconds', 'Model training time', ['model_type'])
model_accuracy = Gauge('scgraph_model_accuracy', 'Model accuracy score', ['model_name', 'dataset'])

# API metrics
api_requests_total = Counter('scgraph_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_request_duration = Histogram('scgraph_api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
```

#### System Metrics
```python
# Resource usage
memory_usage_bytes = Gauge('scgraph_memory_usage_bytes', 'Memory usage in bytes')
cpu_usage_percent = Gauge('scgraph_cpu_usage_percent', 'CPU usage percentage')
gpu_memory_usage_bytes = Gauge('scgraph_gpu_memory_usage_bytes', 'GPU memory usage', ['gpu_id'])

# Database metrics
database_connections_active = Gauge('scgraph_db_connections_active', 'Active database connections')
database_query_duration = Histogram('scgraph_db_query_duration_seconds', 'Database query duration', ['query_type'])

# Cache metrics
cache_hits_total = Counter('scgraph_cache_hits_total', 'Cache hits', ['cache_type'])
cache_misses_total = Counter('scgraph_cache_misses_total', 'Cache misses', ['cache_type'])
```

### Infrastructure Metrics

#### Docker Container Metrics
```yaml
# docker-compose.yml addition
services:
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: scgraph-cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    devices:
      - /dev/kmsg
```

#### Node Metrics
```yaml
  node-exporter:
    image: prom/node-exporter:latest
    container_name: scgraph-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
```

## Health Checks

### Application Health Endpoints

```python
# /health - Basic health check
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00Z",
  "version": "0.1.0",
  "uptime": 3600
}

# /health/detailed - Detailed health status
{
  "status": "healthy",
  "checks": {
    "database": {"status": "healthy", "response_time": "5ms"},
    "redis": {"status": "healthy", "response_time": "2ms"},
    "storage": {"status": "healthy", "response_time": "10ms"},
    "gpu": {"status": "available", "memory_free": "8GB"}
  },
  "resources": {
    "memory_usage": "45%",
    "cpu_usage": "23%",
    "disk_usage": "60%"
  }
}

# /health/ready - Readiness probe
{
  "status": "ready",
  "dependencies": {
    "database": "connected",
    "cache": "connected",
    "storage": "accessible"
  }
}
```

### Kubernetes Health Probes

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: scgraph-hub
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 3
```

## Logging

### Structured Logging Configuration

```python
import structlog
import logging.config

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
            "foreign_pre_chain": [
                structlog.contextvars.merge_contextvars,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.StackInfoRenderer(),
            ],
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/app/logs/scgraph-hub.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
```

### Log Aggregation with ELK Stack

```yaml
# docker-compose.monitoring.yml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    container_name: scgraph-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    container_name: scgraph-logstash
    volumes:
      - ./docker/logstash/pipeline:/usr/share/logstash/pipeline
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    container_name: scgraph-kibana
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Distributed Tracing

### Jaeger Setup

```yaml
# docker-compose.tracing.yml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: scgraph-jaeger
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # HTTP collector
      - "14250:14250"  # gRPC collector
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
```

### Application Tracing

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=14268,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Usage in code
@tracer.start_as_current_span("process_dataset")
def process_dataset(dataset_name: str):
    with tracer.start_as_current_span("load_data") as span:
        span.set_attribute("dataset.name", dataset_name)
        # Load dataset logic
        
    with tracer.start_as_current_span("build_graph"):
        # Graph building logic
        pass
```

## Alerting

### AlertManager Configuration

```yaml
# docker/alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@scgraphhub.org'

templates:
  - '/etc/alertmanager/templates/*.tmpl'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'default'
  email_configs:
  - to: 'admin@scgraphhub.org'
    subject: 'SCGraph Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

- name: 'critical-alerts'
  email_configs:
  - to: 'oncall@scgraphhub.org'
    subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts-critical'
    title: 'Critical Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'warning-alerts'
  email_configs:
  - to: 'team@scgraphhub.org'
    subject: 'WARNING: {{ .GroupLabels.alertname }}'
```

### Alert Rules

```yaml
# docker/prometheus/alert-rules.yml
groups:
- name: scgraph-hub-alerts
  rules:
  # High error rate
  - alert: HighErrorRate
    expr: rate(scgraph_api_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} requests per second"

  # High response time
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(scgraph_api_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }} seconds"

  # Database connection issues
  - alert: DatabaseConnectionFailure
    expr: scgraph_db_connections_active == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failure"
      description: "No active database connections"

  # High memory usage
  - alert: HighMemoryUsage
    expr: scgraph_memory_usage_bytes / (1024^3) > 15  # 15GB
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}GB"

  # GPU memory exhaustion
  - alert: GPUMemoryExhaustion
    expr: scgraph_gpu_memory_usage_bytes / scgraph_gpu_memory_total_bytes > 0.9
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "GPU memory almost exhausted"
      description: "GPU {{ $labels.gpu_id }} memory usage is {{ $value }}%"

  # Service down
  - alert: ServiceDown
    expr: up{job="scgraph-hub"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "SCGraph Hub service is not responding"
```

## Grafana Dashboards

### Application Dashboard

```json
{
  "dashboard": {
    "title": "Single-Cell Graph Hub - Application",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(scgraph_api_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(scgraph_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(scgraph_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(scgraph_api_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
```

### Infrastructure Dashboard

```json
{
  "dashboard": {
    "title": "Single-Cell Graph Hub - Infrastructure", 
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total{name=\"scgraph-hub-app\"}[5m]) * 100",
            "legendFormat": "CPU Usage %"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{name=\"scgraph-hub-app\"} / 1024^3",
            "legendFormat": "Memory Usage GB"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "scgraph_db_connections_active",
            "legendFormat": "Active Connections"
          }
        ]
      }
    ]
  }
}
```

## Performance Monitoring

### Custom Metrics Collection

```python
# Performance monitoring middleware
import time
from prometheus_client import Histogram, Counter

REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration', ['method', 'endpoint'])
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint', 'status'])

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        start_time = time.time()
        method = environ['REQUEST_METHOD']
        path = environ['PATH_INFO']
        
        def new_start_response(status, response_headers, exc_info=None):
            duration = time.time() - start_time
            status_code = status.split()[0]
            
            REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
            REQUEST_COUNT.labels(method=method, endpoint=path, status=status_code).inc()
            
            return start_response(status, response_headers, exc_info)
        
        return self.app(environ, new_start_response)
```

### Business Metrics Tracking

```python
# Dataset usage tracking
from prometheus_client import Counter, Histogram, Gauge

# Dataset metrics
DATASET_DOWNLOADS = Counter('dataset_downloads_total', 'Dataset downloads', ['dataset_name'])
DATASET_PROCESSING_TIME = Histogram('dataset_processing_seconds', 'Processing time', ['dataset_name'])

# Model metrics
MODEL_TRAINING_TIME = Histogram('model_training_seconds', 'Training time', ['model_type'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy', ['model_name', 'dataset'])

# Usage in application
def download_dataset(dataset_name: str):
    DATASET_DOWNLOADS.labels(dataset_name=dataset_name).inc()
    
    start_time = time.time()
    # Download logic here
    duration = time.time() - start_time
    
    DATASET_PROCESSING_TIME.labels(dataset_name=dataset_name).observe(duration)

def train_model(model_type: str, dataset_name: str):
    start_time = time.time()
    # Training logic here
    accuracy = train_model_impl()
    duration = time.time() - start_time
    
    MODEL_TRAINING_TIME.labels(model_type=model_type).observe(duration)
    MODEL_ACCURACY.labels(model_name=model_type, dataset=dataset_name).set(accuracy)
```

## Deployment

### Complete Monitoring Stack

```bash
# Deploy monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Access monitoring services
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - AlertManager: http://localhost:9093
# - Jaeger: http://localhost:16686
# - Kibana: http://localhost:5601
```

### Configuration Validation

```bash
# Validate Prometheus config
docker exec scgraph-prometheus promtool check config /etc/prometheus/prometheus.yml

# Validate alert rules
docker exec scgraph-prometheus promtool check rules /etc/prometheus/alert-rules.yml

# Check AlertManager config
docker exec scgraph-alertmanager amtool config check /etc/alertmanager/alertmanager.yml
```

## Best Practices

1. **Metric Naming**: Use consistent naming conventions
2. **Cardinality Control**: Avoid high-cardinality labels
3. **Alert Hygiene**: Keep alerts actionable and noise-free
4. **Dashboard Organization**: Group related metrics logically
5. **Performance Impact**: Monitor the monitoring overhead
6. **Data Retention**: Configure appropriate retention policies
7. **Security**: Secure monitoring endpoints and data
8. **Documentation**: Document all metrics and alerts

## Troubleshooting

### Common Issues

1. **High Cardinality Metrics**: Review metric labels
2. **Missing Metrics**: Check service discovery and firewall
3. **Alert Fatigue**: Tune alert thresholds and grouping
4. **Storage Issues**: Monitor Prometheus disk usage
5. **Performance Impact**: Profile metrics collection overhead

### Debug Commands

```bash
# Check metric endpoints
curl http://localhost:8000/metrics

# Query Prometheus
curl 'http://localhost:9090/api/v1/query?query=up'

# Test alerting
curl -X POST http://localhost:9093/api/v1/alerts

# View logs
docker logs scgraph-prometheus
docker logs scgraph-grafana
```