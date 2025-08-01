# Contributing to Single-Cell Graph Hub

Thank you for your interest in contributing! This document provides guidelines for contributing to the Single-Cell Graph Hub project.

## Development Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/yourusername/single-cell-graph-hub
   cd single-cell-graph-hub
   pip install -e ".[dev]"
   ```

2. **Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Contributing Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes following our coding standards
4. **Add** tests for new functionality
5. **Run** tests: `pytest`
6. **Run** linting: `black . && isort . && flake8`
7. **Commit** with descriptive messages
8. **Push** to your fork and submit a **Pull Request**

## Code Standards

- **Black** for code formatting
- **isort** for import sorting
- **Type hints** for all public functions
- **Docstrings** in Google style
- **Test coverage** >90% for new code

## Adding Datasets

See our [Dataset Contribution Guide](docs/contributing/datasets.md) for detailed instructions on contributing new single-cell datasets.

## Questions?

Open an issue or start a discussion in our [GitHub Discussions](https://github.com/yourusername/single-cell-graph-hub/discussions).