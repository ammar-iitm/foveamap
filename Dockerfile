# Production CPU Container for FoveaMap (Phase 11 Productization - CPU inference)
# For CUDA GPU deployments, use Dockerfile.gpu
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml ./

# Install CPU PyTorch and Python dependencies
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy application source, checkpoints, dashboard, and metadata
COPY foveamap/ foveamap/
COPY checkpoints/ checkpoints/
COPY dashboard/ dashboard/
COPY README.md LICENSE ./

# Install foveamap package in editable mode with entrypoint scripts
RUN pip install -e . --no-deps

# Create non-root user for security hardening
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["foveamap"]
CMD ["info"]
