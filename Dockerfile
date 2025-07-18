# ---------- build ----------
FROM python:3.11-slim AS builder
WORKDIR /build
COPY app/requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---------- runtime ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Python & system deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Install wheels
COPY --from=builder /wheels /wheels
RUN pip3 install --no-cache /wheels/*

# Create log dir
RUN mkdir -p /var/log && chmod 777 /var/log

WORKDIR /srv
COPY app/ ./app/

EXPOSE 8000
CMD ["python3", "-m", "app.main"]
