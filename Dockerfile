# ---------- build ----------
FROM python:3.12-slim AS builder

# Install build dependencies first (better caching)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python dependencies
COPY app/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels -r requirements.txt

# ---------- runtime ----------
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgomp1 \
        tini && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /srv -s /bin/bash appuser

# Create virtual environment and install wheels
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install --no-deps /wheels/*

# Create directories with proper permissions and fix venv ownership
RUN mkdir -p /srv/logs /models /srv && \
    chown -R appuser:appuser /srv /models /opt/venv

# Set log file environment variable to use user-writable directory
ENV LOG_FILE="/srv/logs/reranker.jsonl" \
    LOG_TO_FILE="true" \
    HF_TOKEN=""

WORKDIR /srv

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Switch to non-root user
USER appuser

EXPOSE 8000

# Use tini as init system and run with virtual environment Python
ENTRYPOINT ["tini", "--"]
CMD ["/opt/venv/bin/python", "-m", "app.main"]
