# ---------- build ----------
FROM python:3.12-slim AS builder

# Install build dependencies first (better caching)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python dependencies (excluding llama-cpp-python for now)
COPY app/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "llama-cpp-python" requirements.txt > requirements_base.txt && \
    pip wheel --wheel-dir /wheels -r requirements_base.txt

# ---------- runtime ----------
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Accept build arguments for user ID and group ID
ARG UID=1000
ARG GID=1000

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DOCKER_ARCH=all \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies including CUDA development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgomp1 \
        wget \
        curl \
        libcurl4-openssl-dev \
        tini \
        git \
        cmake \
        ninja-build \
        pkg-config \
        build-essential \
        software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Create symbolic link for libcuda.so.1 to resolve linking issues
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && \
    ldconfig

# Create non-root user with specified UID and GID
RUN groupadd -f -g ${GID} appuser && \
    useradd -o -r -u ${UID} -g ${GID} -d /srv -s /bin/bash appuser 2>/dev/null || \
    echo "User/group with UID ${UID}/GID ${GID} already exists, continuing..."

# Create virtual environment and install wheels
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install --no-deps /wheels/*

# Clone and build llama.cpp with PR #14029 patch (requires CUDA development environment)
RUN git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    git fetch origin pull/14029/head:pr-14029 && \
    git checkout pr-14029 && \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" && \
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DLLAMA_CURL=OFF \
        -DGGML_CUDA_FORCE_DMMV=ON \
        -DCUDA_ARCHITECTURES="60;61;70;75;80;86;89;90" \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DCMAKE_LIBRARY_PATH="/usr/local/cuda/lib64/stubs" && \
    cmake --build build --config Release --target llama -j$(nproc)

# Set environment variables for llama-cpp-python to use our custom build
ENV LLAMA_CPP_LIB=/opt/llama.cpp/build/src/libllama.so
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DGGML_CUDA_FORCE_DMMV=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs"
ENV FORCE_CMAKE=1

# Now build and install llama-cpp-python from source with our custom llama.cpp
RUN --mount=type=cache,target=/root/.cache/pip \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" && \
    /opt/venv/bin/pip install llama-cpp-python --no-binary=llama-cpp-python

# Create directories with proper permissions and fix venv ownership
RUN mkdir -p /srv/logs /models /srv && \
    chown -R ${UID}:${GID} /srv /models /opt/venv

# Set log file environment variable to use user-writable directory
ENV LOG_FILE="/srv/logs/reranker.jsonl" \
    LOG_TO_FILE="true"

WORKDIR /srv

# Copy application code
COPY --chown=${UID}:${GID} app/ ./app/

# Switch to non-root user
USER appuser

EXPOSE 8000

# Use tini as init system and run with virtual environment Python
ENTRYPOINT ["tini", "--"]
CMD ["/opt/venv/bin/python", "-m", "app.main"]
