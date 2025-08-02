# Single-Cell Graph Hub - Documentation Index

This document provides a comprehensive index of all documentation in the repository.

## 📋 Core Documentation

### Project Overview
- [README.md](../README.md) - Main project overview and quick start guide
- [PROJECT_CHARTER.md](../PROJECT_CHARTER.md) - Project charter with objectives and scope
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design decisions

### Setup and Configuration
- [SETUP_REQUIRED.md](SETUP_REQUIRED.md) - Manual setup steps required after automated setup
- [.env.template](../.env.template) - Environment variables template
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contributing guidelines
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) - Code of conduct

## 🛠️ Development Documentation

### Development Environment
- [.devcontainer/devcontainer.json](../.devcontainer/devcontainer.json) - VS Code dev container configuration
- [.pre-commit-config.yaml](../.pre-commit-config.yaml) - Pre-commit hooks configuration
- [.editorconfig](../.editorconfig) - Editor configuration

### Testing
- [tests/README.md](../tests/README.md) - Testing strategy and guidelines
- [tests/conftest.py](../tests/conftest.py) - Pytest configuration and fixtures

## 🏗️ Infrastructure Documentation

### Containerization
- [Dockerfile](../Dockerfile) - Multi-stage Docker build configuration
- [docker-compose.yml](../docker-compose.yml) - Development and production services
- [.dockerignore](../.dockerignore) - Docker build exclusions

### Monitoring & Observability
- [monitoring/README.md](../monitoring/README.md) - Monitoring setup guide
- [monitoring/prometheus.yml](../monitoring/prometheus.yml) - Prometheus configuration
- [monitoring/grafana/](../monitoring/grafana/) - Grafana dashboards and configuration

## 🔄 CI/CD Documentation

### Workflow Templates
- [workflows/examples/ci.yml](workflows/examples/ci.yml) - Continuous Integration pipeline
- [workflows/examples/security-scan.yml](workflows/examples/security-scan.yml) - Security scanning workflow
- [workflows/examples/dependency-update.yml](workflows/examples/dependency-update.yml) - Automated dependency updates

### Automation Scripts
- [scripts/automation/collect-metrics.py](../scripts/automation/collect-metrics.py) - Metrics collection automation
- [scripts/automation/dependency-checker.py](../scripts/automation/dependency-checker.py) - Dependency security checking
- [scripts/automation/repository-maintenance.py](../scripts/automation/repository-maintenance.py) - Repository maintenance tasks

## 📚 Operational Documentation

### Runbooks
- [runbooks/deployment.md](runbooks/deployment.md) - Deployment procedures
- [runbooks/monitoring.md](runbooks/monitoring.md) - Monitoring and alerting guide
- [runbooks/security.md](runbooks/security.md) - Security procedures
- [runbooks/troubleshooting.md](runbooks/troubleshooting.md) - Common issues and solutions

### Architecture Decision Records (ADR)
- [adr/0001-record-architecture-decisions.md](adr/0001-record-architecture-decisions.md) - ADR process definition
- [adr/0002-python-framework-selection.md](adr/0002-python-framework-selection.md) - Python framework choices
- [adr/0003-testing-strategy.md](adr/0003-testing-strategy.md) - Testing approach and tools
- [adr/0004-containerization-approach.md](adr/0004-containerization-approach.md) - Containerization strategy
- [adr/0005-monitoring-solution.md](adr/0005-monitoring-solution.md) - Monitoring and observability choices

## 🔒 Security Documentation

- [SECURITY.md](../SECURITY.md) - Security policy and vulnerability reporting
- [docs/security/](security/) - Security guidelines and best practices

## 📊 Metrics and Reporting

- [.github/project-metrics.json](../.github/project-metrics.json) - Project metrics configuration
- [scripts/verify-setup.py](../scripts/verify-setup.py) - Setup verification script

## 🚀 Quick Navigation

### For Developers
1. Start with [README.md](../README.md) for project overview
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system understanding
3. Follow [SETUP_REQUIRED.md](SETUP_REQUIRED.md) for environment setup
4. Check [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines

### For Operators
1. Review [runbooks/deployment.md](runbooks/deployment.md) for deployment
2. Check [runbooks/monitoring.md](runbooks/monitoring.md) for monitoring setup
3. Reference [runbooks/troubleshooting.md](runbooks/troubleshooting.md) for issues

### For Security Teams
1. Review [SECURITY.md](../SECURITY.md) for security policy
2. Check [workflows/examples/security-scan.yml](workflows/examples/security-scan.yml) for scanning
3. Review [docs/security/](security/) for security guidelines

---

**Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

This index is automatically maintained. For updates, modify the source documentation files.
