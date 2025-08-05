# Multi-stage Docker build for Single-Cell Graph Hub
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r scgraph && useradd -r -g scgraph scgraph

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Production stage
FROM base as production

# Copy source code
COPY src/ ./src/
COPY setup.py pyproject.toml MANIFEST.in README.md ./

# Install package
RUN pip install .

# Change ownership to non-root user
RUN chown -R scgraph:scgraph /app

USER scgraph

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import scgraph_hub; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import scgraph_hub; print('Single-Cell Graph Hub ready')"]