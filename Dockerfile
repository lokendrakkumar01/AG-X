# AG-X 2026: Advanced Gravity Research Simulation Platform
# Multi-stage Dockerfile with optional GPU support

# ============================================================================
# Stage 1: Base Python Environment
# ============================================================================
FROM python:3.11-slim as base

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 2: Dependencies Builder
# ============================================================================
FROM base as builder

# Copy project files
COPY pyproject.toml README.md ./
COPY agx/ ./agx/

# Install dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ".[all]" \
    && pip install --target=/app/deps -e .

# ============================================================================
# Stage 3: Production Image
# ============================================================================
FROM base as production

# Create non-root user for security
RUN groupadd --gid 1000 agx \
    && useradd --uid 1000 --gid agx --shell /bin/bash --create-home agx

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY --chown=agx:agx . .

# Create directories for data persistence
RUN mkdir -p /app/data /app/experiments /app/reports /app/logs \
    && chown -R agx:agx /app

USER agx

# Expose ports for web dashboard
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Default command: start web dashboard
CMD ["python", "-m", "agx.web_app"]

# ============================================================================
# Stage 4: GPU-enabled Image (Optional)
# ============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy project and install
COPY pyproject.toml README.md ./
COPY agx/ ./agx/

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ".[gpu]"

COPY . .

EXPOSE 8050

CMD ["python", "-m", "agx.web_app"]
