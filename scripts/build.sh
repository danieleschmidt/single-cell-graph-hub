#!/bin/bash
# Build script for Single-Cell Graph Hub
# Handles building, testing, and packaging

set -euo pipefail

# Configuration
PROJECT_NAME="single-cell-graph-hub"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"
DOCKER_USERNAME="${DOCKER_USERNAME:-danieleschmidt}"
VERSION=$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Single-Cell Graph Hub Build Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    clean           Clean build artifacts and caches
    test            Run test suite
    lint            Run code linting
    format          Format code
    build           Build Python package
    docker          Build Docker images
    publish         Publish package to PyPI
    release         Full release pipeline (test, build, publish)
    
Options:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    --skip-tests    Skip running tests
    --skip-lint     Skip linting checks
    --dev           Development mode (additional tools)

Environment Variables:
    DOCKER_REGISTRY     Docker registry (default: ghcr.io)
    DOCKER_USERNAME     Docker username (default: danieleschmidt)
    PYPI_TOKEN         PyPI authentication token
    SKIP_GPU_TESTS     Skip GPU-dependent tests (set to 'true')

Examples:
    $0 clean build          # Clean and build package
    $0 test --skip-lint     # Run tests without linting
    $0 docker --dev         # Build development Docker image
    $0 release              # Full release pipeline

EOF
}

# Clean function
clean() {
    log_info "Cleaning build artifacts..."
    
    # Python artifacts
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type f -name "*.pyd" -delete 2>/dev/null || true
    find . -type f -name ".coverage" -delete 2>/dev/null || true
    
    # Build directories
    rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
    rm -rf htmlcov/ .tox/ .coverage.*
    
    # Docker artifacts
    docker system prune -f || log_warn "Docker cleanup failed"
    
    log_info "Clean completed"
}

# Test function
run_tests() {
    log_info "Running test suite..."
    
    if [[ "${SKIP_TESTS:-false}" == "true" ]]; then
        log_warn "Skipping tests (SKIP_TESTS=true)"
        return 0
    fi
    
    # Set test environment
    export SCGRAPH_DEV_MODE=true
    export LOG_LEVEL=DEBUG
    
    # Run different test categories
    if [[ "${SKIP_GPU_TESTS:-false}" == "true" ]]; then
        log_info "Running tests (skipping GPU tests)..."
        pytest tests/ -v -m "not gpu" --cov=scgraph_hub --cov-report=html --cov-report=term
    else
        log_info "Running all tests..."
        pytest tests/ -v --cov=scgraph_hub --cov-report=html --cov-report=term
    fi
    
    log_info "Tests completed successfully"
}

# Lint function
run_lint() {
    log_info "Running code quality checks..."
    
    if [[ "${SKIP_LINT:-false}" == "true" ]]; then
        log_warn "Skipping lint checks (SKIP_LINT=true)"
        return 0
    fi
    
    # Run linting tools
    log_info "Running ruff..."
    ruff check src/ tests/
    
    log_info "Running flake8..."
    flake8 src/ tests/
    
    log_info "Running mypy..."
    mypy src/
    
    log_info "Running bandit security checks..."
    bandit -r src/ -f json
    
    log_info "Lint checks completed successfully"
}

# Format function
format_code() {
    log_info "Formatting code..."
    
    log_info "Running black..."
    black src/ tests/
    
    log_info "Running isort..."
    isort src/ tests/
    
    log_info "Code formatting completed"
}

# Build function
build_package() {
    log_info "Building Python package..."
    
    # Ensure build tools are up to date
    pip install --upgrade build twine
    
    # Build package
    python -m build
    
    # Verify package
    twine check dist/*
    
    log_info "Package build completed successfully"
    log_info "Built packages:"
    ls -la dist/
}

# Docker build function
build_docker() {
    log_info "Building Docker images..."
    
    # Build arguments
    BUILD_ARGS=(
        --build-arg "BUILD_DATE=${BUILD_DATE}"
        --build-arg "VCS_REF=${VCS_REF}"
        --build-arg "VERSION=${VERSION}"
    )
    
    # Base image
    log_info "Building base runtime image..."
    docker build "${BUILD_ARGS[@]}" \
        --target runtime \
        -t "${PROJECT_NAME}:${VERSION}" \
        -t "${PROJECT_NAME}:latest" \
        .
    
    # Development image
    if [[ "${DEV_MODE:-false}" == "true" ]]; then
        log_info "Building development image..."
        docker build "${BUILD_ARGS[@]}" \
            --target development \
            -t "${PROJECT_NAME}:${VERSION}-dev" \
            -t "${PROJECT_NAME}:dev" \
            .
    fi
    
    # Production image
    log_info "Building production image..."
    docker build "${BUILD_ARGS[@]}" \
        --target production \
        -t "${PROJECT_NAME}:${VERSION}-prod" \
        -t "${PROJECT_NAME}:prod" \
        .
    
    # Tag for registry
    if [[ -n "${DOCKER_REGISTRY:-}" && -n "${DOCKER_USERNAME:-}" ]]; then
        REGISTRY_BASE="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${PROJECT_NAME}"
        
        docker tag "${PROJECT_NAME}:${VERSION}" "${REGISTRY_BASE}:${VERSION}"
        docker tag "${PROJECT_NAME}:latest" "${REGISTRY_BASE}:latest"
        docker tag "${PROJECT_NAME}:prod" "${REGISTRY_BASE}:prod"
        
        if [[ "${DEV_MODE:-false}" == "true" ]]; then
            docker tag "${PROJECT_NAME}:dev" "${REGISTRY_BASE}:dev"
        fi
    fi
    
    log_info "Docker build completed successfully"
    log_info "Built images:"
    docker images "${PROJECT_NAME}"
}

# Publish function
publish_package() {
    log_info "Publishing package to PyPI..."
    
    if [[ -z "${PYPI_TOKEN:-}" ]]; then
        log_error "PYPI_TOKEN environment variable is required for publishing"
        exit 1
    fi
    
    # Publish to PyPI
    twine upload dist/* --username __token__ --password "${PYPI_TOKEN}"
    
    log_info "Package published successfully"
}

# Security scan function
security_scan() {
    log_info "Running security scans..."
    
    # Python security
    log_info "Scanning Python dependencies..."
    safety check
    
    # Docker security (if trivy is available)
    if command -v trivy &> /dev/null; then
        log_info "Scanning Docker image for vulnerabilities..."
        trivy image "${PROJECT_NAME}:latest"
    else
        log_warn "Trivy not found, skipping Docker security scan"
    fi
    
    log_info "Security scan completed"
}

# Release function
release() {
    log_info "Starting release pipeline..."
    
    # Pre-release checks
    if [[ -n $(git status --porcelain) ]]; then
        log_error "Working directory is not clean. Please commit all changes."
        exit 1
    fi
    
    if [[ $(git rev-parse --abbrev-ref HEAD) != "main" ]]; then
        log_warn "Not on main branch. Current branch: $(git rev-parse --abbrev-ref HEAD)"
        read -p "Continue with release? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Release aborted"
            exit 0
        fi
    fi
    
    # Run release pipeline
    clean
    format_code
    run_lint
    run_tests
    security_scan
    build_package
    build_docker
    
    # Confirmation for publishing
    echo
    log_info "Release pipeline completed successfully!"
    log_info "Ready to publish version ${VERSION}"
    echo
    read -p "Publish to PyPI and Docker registry? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        publish_package
        
        if [[ -n "${DOCKER_REGISTRY:-}" && -n "${DOCKER_USERNAME:-}" ]]; then
            log_info "Pushing Docker images..."
            docker push "${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${PROJECT_NAME}:${VERSION}"
            docker push "${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${PROJECT_NAME}:latest"
            docker push "${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${PROJECT_NAME}:prod"
        fi
        
        # Create git tag
        git tag "v${VERSION}"
        git push origin "v${VERSION}"
        
        log_info "Release v${VERSION} completed successfully!"
    else
        log_info "Publish skipped. Build artifacts are ready in dist/"
    fi
}

# Parse command line arguments
VERBOSE=false
SKIP_TESTS=false
SKIP_LINT=false
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-lint)
            SKIP_LINT=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        clean)
            clean
            exit 0
            ;;
        test)
            run_tests
            exit 0
            ;;
        lint)
            run_lint
            exit 0
            ;;
        format)
            format_code
            exit 0
            ;;
        build)
            build_package
            exit 0
            ;;
        docker)
            build_docker
            exit 0
            ;;
        publish)
            publish_package
            exit 0
            ;;
        security)
            security_scan
            exit 0
            ;;
        release)
            release
            exit 0
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
done

# If no command provided, show help
show_help