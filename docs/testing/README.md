# Testing Framework

This document describes the comprehensive testing framework for Single-Cell Graph Hub.

## Test Structure

Our testing framework is organized into multiple layers:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_dataset.py      # Dataset functionality tests
│   ├── test_models.py       # Model architecture tests
│   └── test_preprocessing.py # Data preprocessing tests
├── integration/             # Integration tests for workflows
│   ├── test_data_pipeline.py # End-to-end data pipeline tests
│   └── test_model_training.py # Training pipeline tests
├── e2e/                     # End-to-end system tests
│   └── test_benchmarks.py   # Full benchmark suite tests
└── fixtures/                # Test data and utilities
    ├── sample_data.py       # Sample data generation
    └── mock_datasets.py     # Mock dataset implementations
```

## Test Categories

### Unit Tests
- **Scope**: Individual functions and classes
- **Speed**: Fast (< 1 second per test)
- **Dependencies**: Minimal external dependencies
- **Marker**: `@pytest.mark.unit`

```python
@pytest.mark.unit
def test_dataset_initialization():
    """Test basic dataset initialization."""
    dataset = SCGraphDataset(name="test", root="./data")
    assert dataset.name == "test"
```

### Integration Tests
- **Scope**: Component interactions and workflows
- **Speed**: Medium (1-30 seconds per test)
- **Dependencies**: External libraries, file I/O
- **Marker**: `@pytest.mark.integration`

```python
@pytest.mark.integration
def test_data_pipeline():
    """Test complete data processing pipeline."""
    raw_data = load_h5ad("sample.h5ad")
    processed_data = preprocess_pipeline(raw_data)
    graph_data = build_graph(processed_data)
    assert graph_data.num_nodes > 0
```

### End-to-End Tests
- **Scope**: Complete system workflows
- **Speed**: Slow (> 30 seconds per test)
- **Dependencies**: Full system stack
- **Marker**: `@pytest.mark.e2e`

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_full_benchmark():
    """Test complete benchmark execution."""
    results = run_benchmark(
        datasets=["pbmc_10k"],
        models=["GCN", "GAT"],
        metrics=["accuracy", "f1"]
    )
    assert len(results) > 0
```

## Test Markers

We use pytest markers to categorize and control test execution:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.parametrize`: Parameterized tests

## Running Tests

### Basic Test Execution

```bash
# Run all tests
make test

# Run only fast tests
make test-fast

# Run with coverage
make test-cov

# Run specific test categories
pytest tests/ -m unit                    # Unit tests only
pytest tests/ -m "not slow"             # Skip slow tests  
pytest tests/ -m "integration or e2e"   # Integration and E2E tests
```

### Parallel Test Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Run with specific number of workers
pytest tests/ -n 4
```

### GPU Tests

```bash
# Run GPU tests (skipped if CUDA unavailable)
pytest tests/ -m gpu

# Skip GPU tests explicitly
pytest tests/ -m "not gpu"
```

## Test Configuration

### Environment Variables

Set these environment variables for testing:

```bash
export SCGRAPH_DEV_MODE=true
export LOG_LEVEL=DEBUG
export PYTEST_TIMEOUT=300
```

### Pytest Configuration

Configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=scgraph_hub --cov-report=html --cov-report=term-missing"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests", 
    "gpu: marks tests requiring GPU",
]
```

## Fixtures

### Common Fixtures

Available in `conftest.py`:

- `temp_data_dir`: Temporary directory for test data
- `sample_gene_expression`: Mock gene expression matrix
- `sample_cell_metadata`: Mock cell metadata DataFrame
- `sample_pyg_data`: Sample PyTorch Geometric data
- `gpu_available`: GPU availability flag
- `test_config`: Test configuration dictionary

### Using Fixtures

```python
def test_with_sample_data(sample_gene_expression, sample_cell_metadata):
    """Test using sample data fixtures."""
    assert sample_gene_expression.shape[0] == len(sample_cell_metadata)
    assert sample_gene_expression.min() >= 0
```

### Custom Fixtures

Create custom fixtures for specific test needs:

```python
@pytest.fixture
def custom_dataset():
    """Create custom dataset for specific tests."""
    return create_mock_dataset(
        n_cells=500,
        n_genes=1000,
        cell_types=["A", "B", "C"]
    )
```

## Mock Data Generation

Use utilities in `tests/fixtures/sample_data.py`:

```python
from tests.fixtures.sample_data import (
    create_sample_expression_data,
    create_sample_metadata,
    create_sample_graph_data
)

# Generate sample data
expression = create_sample_expression_data(n_cells=100, n_genes=50)
metadata = create_sample_metadata(n_cells=100)
graph_data = create_sample_graph_data(n_cells=100, n_genes=50)
```

## Performance Testing

### Benchmarking

Use pytest-benchmark for performance tests:

```python
def test_data_loading_performance(benchmark):
    """Benchmark data loading performance."""
    result = benchmark(load_dataset, "large_dataset")
    assert result is not None
```

### Memory Profiling

Use memory_profiler for memory tests:

```python
@pytest.mark.slow
def test_memory_usage():
    """Test memory usage of large dataset processing."""
    from memory_profiler import profile
    
    @profile
    def process_large_dataset():
        # Processing logic here
        pass
    
    process_large_dataset()
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled runs (daily)

### Test Coverage

Maintain high test coverage:
- **Target**: >90% line coverage
- **Minimum**: >80% line coverage
- **Critical paths**: 100% coverage

### Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=scgraph_hub --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Data Management

### Temporary Files

Use `temp_data_dir` fixture for temporary files:

```python
def test_file_operations(temp_data_dir):
    """Test file operations with temporary directory."""
    test_file = temp_data_dir / "test.h5ad"
    save_data(data, test_file)
    loaded_data = load_data(test_file)
    assert loaded_data is not None
```

### Cleanup

Tests automatically clean up:
- Temporary files
- Test artifacts
- Memory allocations
- GPU resources

## Debugging Tests

### Verbose Output

```bash
# Run with verbose output
pytest tests/ -v

# Run with extra verbose output
pytest tests/ -vv

# Show print statements
pytest tests/ -s
```

### Debug Specific Test

```bash
# Run single test with debugging
pytest tests/unit/test_dataset.py::test_dataset_loading -vv -s

# Drop into debugger on failure
pytest tests/ --pdb
```

### Logging in Tests

```python
import logging

def test_with_logging():
    """Test with logging enabled."""
    logger = logging.getLogger(__name__)
    logger.info("Starting test")
    
    # Test logic here
    
    logger.info("Test completed")
```

## Best Practices

### Test Organization

1. **One test per function**: Each test should test one specific behavior
2. **Descriptive names**: Test names should clearly describe what they test
3. **AAA pattern**: Arrange, Act, Assert structure
4. **Independent tests**: Tests should not depend on each other

### Test Data

1. **Use fixtures**: Prefer fixtures over hardcoded test data
2. **Realistic data**: Use data that resembles real-world scenarios
3. **Edge cases**: Test boundary conditions and edge cases
4. **Error conditions**: Test error handling and recovery

### Performance

1. **Fast feedback**: Keep unit tests fast for quick feedback
2. **Parallel execution**: Use parallel test execution for faster CI
3. **Resource cleanup**: Always clean up resources after tests
4. **Mock external dependencies**: Mock slow external services

### Maintainability

1. **DRY principle**: Don't repeat test setup code
2. **Clear assertions**: Use descriptive assertion messages
3. **Test documentation**: Document complex test scenarios
4. **Regular maintenance**: Keep tests updated with code changes

## Troubleshooting

### Common Issues

1. **Import errors**: Check PYTHONPATH and package installation
2. **Fixture not found**: Verify fixture is defined in conftest.py
3. **Tests hanging**: Check for infinite loops or network timeouts
4. **Memory errors**: Use smaller test datasets or cleanup resources

### Getting Help

- Check test logs for detailed error messages
- Use pytest debugging flags (-vv, -s, --pdb)
- Review fixture definitions and dependencies
- Consult pytest documentation for advanced features