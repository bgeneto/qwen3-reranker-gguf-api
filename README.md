

# Qwen3 GGUF Reranker API

This repository provides a production-grade Docker Compose stack for serving a Qwen3-based reranker model using `llama-cpp-python`.

The API is designed for performance and security, offloading the entire model to the GPU, protecting routes with bearer token authentication, and providing structured logging and Prometheus metrics.

## Features

*   **CUDA Enabled**: Runs `llama-cpp-python` with CUDA support to offload 100% of the model layers to the GPU for maximum performance.
*   **Configuration Driven**: All settings are managed via a `.env` file (model path, GPU layers, logging, etc.).
*   **Structured Logging**: Asynchronous JSON-L logging to a file, which can be enabled or disabled.
*   **Prometheus Metrics**: Exposes a `/metrics` endpoint with counters for requests, failures, and a histogram for latency.
*   **Authentication**: All routes (except `/health` and `/metrics`) are protected by bearer token authentication.
*   **Dockerized**: Comes with a `Dockerfile` and `docker-compose.yml` for easy setup and deployment.

## Directory Layout

```
.
├── docker-compose.yml
├── Dockerfile
├── .env
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── auth.py
│   ├── metrics.py
│   ├── logger.py
│   └── requirements.txt
└── models/
    └── qwen3-4b-reranker-q4_k_m.gguf
```

## Getting Started

### 1. Prerequisites

*   Docker
*   Docker Compose
*   NVIDIA GPU with CUDA drivers installed

### 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/qwen3-reranker-gguf-api.git
    cd qwen3-reranker-gguf-api
    ```

2.  **Download the Model:**
    Place your GGUF model file inside the `models/` directory. The `.env` file is pre-configured for `qwen3-4b-reranker-q4_k_m.gguf`.

3.  **Configure Environment:**
    Rename the `.env.example` file to `.env` and customize the values, especially the `API_TOKEN`.

    ```bash
    # .env
    # model
    MODEL_PATH=/models/qwen3-4b-reranker-q4_k_m.gguf
    N_CTX=8192               # Max context size for the model's memory
    N_GPU_LAYERS=-1          # -1 = offload everything
    N_BATCH=512
    N_THREADS=0            # 0 = auto

    # logging
    LOG_LEVEL=INFO
    LOG_TO_FILE=true
    LOG_FILE=/var/log/reranker.jsonl

    # auth
    API_TOKEN=change-me-please

    # server
    HOST=0.0.0.0
    PORT=8000
    ```

    **A note on `N_CTX`**: This value defines the model's context window size (in tokens), which is its short-term memory. It must be less than or equal to the model's maximum supported context size. A larger `N_CTX` requires more VRAM/RAM.

### 3. Run the Service

Start the application using Docker Compose:

```bash
docker compose up --build -d
```

The API will be available at `http://localhost:8000`.

## API Usage

### Rerank Endpoint

To use the reranker, send a `POST` request to the `/v1/rerank` endpoint with your query and a list of documents.

**Authentication is required.** Include your `API_TOKEN` as a bearer token in the `Authorization` header.

**Example Request:**

```bash
curl -s http://localhost:8000/v1/rerank \
  -H "Authorization: Bearer change-me-please" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What is the capital of China?",
        "documents": [
            {"text": "The capital of France is Paris."},
            {"text": "The capital of China is Beijing."}
        ],
        "top_n": 1
      }' | jq .
```

**Example Response:**

```json
{
  "model": "qwen3-reranker",
  "results": [
    {
      "index": 1,
      "relevance_score": 0.998
    }
  ]
}
```

### Other Endpoints

*   **Health Check**: `GET /health` - Publicly accessible endpoint to check service status.
*   **Metrics**: `GET /metrics` - Publicly accessible endpoint for Prometheus metrics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

