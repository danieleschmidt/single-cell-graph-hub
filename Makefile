# Single-Cell Graph Hub Makefile
# Development and build automation

.PHONY: help install install-dev clean test test-fast test-cov lint format typecheck security docs build publish docker

# Default target
help:
	@echo "Single-Cell Graph Hub Development Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      Install package in production mode"
	@echo "  install-dev  Install package in development mode with all dependencies"
	@echo "  clean        Clean build artifacts and cache files"
	@echo ""
	@echo "Development Commands:"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run fast tests only (skip slow/integration tests)"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run all linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  typecheck    Run mypy type checking"
	@echo "  security     Run security checks with bandit and safety"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs         Build documentation"
	@echo "  docs-serve   Serve documentation locally"
	@echo ""
	@echo "Build Commands:"
	@echo "  build        Build package for distribution"
	@echo "  publish      Publish package to PyPI (requires auth)"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker       Build Docker image"
	@echo "  docker-dev   Build development Docker image"

# Installation commands
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Cleaning commands  
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Testing commands
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-cov:
	pytest tests/ --cov=scgraph_hub --cov-report=html --cov-report=term-missing

test-integration:
	pytest tests/ -v -m "integration"

test-unit:
	pytest tests/ -v -m "unit"

# Code quality commands
lint: lint-ruff lint-flake8 lint-mypy

lint-ruff:
	ruff check src/ tests/

lint-flake8:
	flake8 src/ tests/

lint-mypy:
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check src/ tests/

typecheck:
	mypy src/

# Security commands
security:
	bandit -r src/
	safety check

security-check:
	bandit -r src/ -f json
	safety check --json

# Pre-commit commands
pre-commit:
	pre-commit run --all-files

pre-commit-install:
	pre-commit install

# Documentation commands
docs:
	cd docs && make html

docs-serve:
	cd docs && make livehtml

docs-clean:
	cd docs && make clean

# Build commands
build: clean
	python -m build

build-wheel:
	python -m build --wheel

build-sdist:
	python -m build --sdist

# Publishing commands
publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*

# Docker commands
docker:
	docker build -t single-cell-graph-hub .

docker-dev:
	docker build -f Dockerfile.dev -t single-cell-graph-hub:dev .

docker-run:
	docker run -it --rm single-cell-graph-hub

# Development workflow commands
dev-setup: install-dev pre-commit-install
	@echo "Development environment setup complete!"

dev-check: format lint test-fast security
	@echo "Development checks passed!"

ci-check: format-check lint test security
	@echo "CI checks passed!"

# Utility commands
version:
	@python -c "import scgraph_hub; print(scgraph_hub.__version__)"

deps-update:
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in

# Performance profiling
profile:
	python -m cProfile -s cumulative scripts/profile_example.py

benchmark:
	python scripts/benchmark.py

# Database/migration commands (if applicable)
migrate:
	@echo "No migrations configured yet"

# Jupyter commands
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

notebook-clean:
	find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;

# Environment commands
env-info:
	@echo "Python version:"
	@python --version
	@echo "Pip version:"
	@pip --version
	@echo "Installed packages:"
	@pip list

# Git hooks
hooks-install:
	pre-commit install
	pre-commit install --hook-type commit-msg

hooks-run:
	pre-commit run --all-files

# Quick development cycle
quick-test: format test-fast

# Full quality check
quality: format lint typecheck test security docs
	@echo "Full quality check completed!"