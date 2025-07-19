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

# Accept build arguments for user ID and group ID
ARG UID=1000
ARG GID=1000

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
        wget \
        curl \
        tini && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user with specified UID and GID
RUN if ! getent group ${GID} >/dev/null 2>&1; then \
        groupadd -g ${GID} appuser; \
    else \
        GROUP_NAME=$(getent group ${GID} | cut -d: -f1); \
        echo "Group with GID ${GID} already exists: $GROUP_NAME"; \
    fi && \
    if ! id -u ${UID} >/dev/null 2>&1; then \
        if getent group ${GID} >/dev/null 2>&1; then \
            GROUP_NAME=$(getent group ${GID} | cut -d: -f1); \
            useradd -r -u ${UID} -g $GROUP_NAME -d /srv -s /bin/bash appuser; \
        else \
            useradd -r -u ${UID} -g appuser -d /srv -s /bin/bash appuser; \
        fi; \
    else \
        echo "User with UID ${UID} already exists"; \
    fi

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
    LOG_TO_FILE="true"

WORKDIR /srv

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Switch to non-root user
USER appuser

EXPOSE 8000

# Use tini as init system and run with virtual environment Python
ENTRYPOINT ["tini", "--"]
CMD ["/opt/venv/bin/python", "-m", "app.main"]
