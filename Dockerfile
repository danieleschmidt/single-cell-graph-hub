# Multi-stage Dockerfile for Single-Cell Graph Hub
# Optimized for security, performance, and size

# =============================================================================
# Build Stage - Compile dependencies and prepare environment
# =============================================================================
FROM python:3.11-slim as builder

# Set build arguments
ARG TORCH_VERSION=2.1.0
ARG CUDA_VERSION=cu118
ARG BUILD_DATE
ARG VCS_REF

# Add metadata labels
LABEL org.opencontainers.image.title="Single-Cell Graph Hub"
LABEL org.opencontainers.image.description="Graph Neural Networks for Single-Cell Omics Analysis"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.source="https://github.com/danieleschmidt/single-cell-graph-hub"
LABEL org.opencontainers.image.documentation="https://scgraphhub.readthedocs.io"
LABEL org.opencontainers.image.licenses="MIT"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r scgraph && useradd -r -g scgraph -m -s /bin/bash scgraph

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
COPY src/ src/

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (if needed)
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+${CUDA_VERSION} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install PyTorch Geometric
RUN pip install --no-cache-dir \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

# Install the package and dependencies
RUN pip install --no-cache-dir -e .

# =============================================================================
# Runtime Stage - Minimal runtime environment
# =============================================================================
FROM python:3.11-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r scgraph && useradd -r -g scgraph -m -s /bin/bash scgraph

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY --from=builder /app/src ./src
COPY --chown=scgraph:scgraph . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/models /app/logs /app/cache && \
    chown -R scgraph:scgraph /app

# Security: Remove unnecessary packages and files
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variables
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SCGRAPH_DATA_DIR="/app/data"
ENV SCGRAPH_CACHE_DIR="/app/cache"
ENV TORCH_HOME="/app/cache/torch"
ENV HF_HOME="/app/cache/huggingface"

# Expose ports
EXPOSE 8000 8888 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import scgraph_hub; print('OK')" || exit 1

# Switch to non-root user
USER scgraph

# Set entrypoint
ENTRYPOINT ["python", "-m", "scgraph_hub.cli"]
CMD ["--help"]

# =============================================================================
# Development Stage - Additional development tools
# =============================================================================
FROM runtime as development

# Switch back to root for package installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    notebook \
    matplotlib \
    seaborn \
    plotly

# Install additional development tools
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Set up Jupyter configuration
RUN mkdir -p /home/scgraph/.jupyter
COPY docker/jupyter_config.py /home/scgraph/.jupyter/jupyter_lab_config.py
RUN chown -R scgraph:scgraph /home/scgraph/.jupyter

# Switch back to non-root user
USER scgraph

# Override entrypoint for development
ENTRYPOINT ["/bin/bash"]

# =============================================================================
# Production Stage - Optimized for production deployment
# =============================================================================
FROM runtime as production

# Additional security hardening for production
USER root

# Remove development tools and minimize attack surface
RUN apt-get update && apt-get remove -y \
    curl \
    wget \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set strict security policies
RUN echo "scgraph ALL=(ALL) NOPASSWD: /usr/bin/python" > /etc/sudoers.d/scgraph

# Switch to non-root user
USER scgraph

# Production-specific environment variables
ENV SCGRAPH_ENV=production
ENV LOG_LEVEL=INFO
ENV WORKERS=4

# Override entrypoint for production
ENTRYPOINT ["python", "-m", "scgraph_hub.server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]