

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

    **Option 1: Manual Download**
    Place your GGUF model file inside the `models/` directory. The filename should match what you set in `MODEL_FILENAME`.

    **Option 2: Automatic Download (Recommended)**
    Configure the download settings in your `.env` file. The application will automatically download the model from the provided URL if it's not found in the `/models` directory:

    ```bash
    MODEL_FILENAME=model-file.gguf
    MODEL_LINK=https://example.com/path/to/model-file.gguf  # Direct download URL
    ```

3.  **Configure Environment:**
    Rename the `example.env` file to `.env` and customize the values:

    ```bash
    # .env
    # model
    MODEL_FILENAME=Qwen3-Reranker-4B-q4_k_m.gguf         # Model filename
    MODEL_LINK=https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF/resolve/main/Qwen3-Reranker-4B-q4_k_m.gguf  # Direct download URL
    N_CTX=8192               # Max context size for the model's memory
    N_GPU_LAYERS=-1          # -1 = offload everything
    N_BATCH=512
    N_THREADS=0            # 0 = auto
    ```

    **Note**: The model will be automatically saved to `/models/{MODEL_FILENAME}`. You no longer need to specify a full `MODEL_PATH` - just set the filename and the application will handle the rest.

    # logging
    LOG_LEVEL=INFO
    LOG_TO_FILE=true
    LOG_FILE=/var/log/reranker.jsonl

    # auth
    API_TOKEN=change-me-please

    # server
    HOST=0.0.0.0
    PORT=8000

    # user permissions (for Docker volume mounts)
    # Set these to match your host user to avoid permission issues
    # On Linux: UID=$(id -u) GID=$(id -g)
    UID=1000
    GID=1000
    ```

    **Important**: Set `UID` and `GID` to match your host user to avoid file permission issues with Docker volume mounts. On Linux, you can get these values by running:
    ```bash
    ./get-uid-gid.sh
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

*   **Health Check**: `GET /health` - Publicly accessible endpoint to check service status and model information.
*   **Metrics**: `GET /metrics` - Publicly accessible endpoint for Prometheus metrics.

## Automatic Model Download from Hugging Face

This application supports automatic model downloading from direct URLs, which eliminates the need to manually download and manage model files.

### How It Works

1. **First Startup**: If the model file doesn't exist in the `/models` directory, the application will automatically attempt to download it from the configured download URL.

2. **Download Tools**: The application uses `wget` or `curl` (whichever is available) to download the model file.

3. **Caching**: Once downloaded, the model is stored locally and won't be downloaded again unless you delete the local file.

### Configuration

Configure these environment variables in your `.env` file:

- `MODEL_FILENAME`: The name for the downloaded model file
- `MODEL_LINK`: The direct download URL for the model

### Example for Different Models

**For a public GGUF model:**
```bash
MODEL_FILENAME=Qwen3-Reranker-4B-q4_k_m.gguf
MODEL_LINK=https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF/resolve/main/Qwen3-Reranker-4B-q4_k_m.gguf
```

**For another model:**
```bash
MODEL_FILENAME=llama-2-7b-chat.q4_k_m.gguf
MODEL_LINK=https://example.com/path/to/llama-2-7b-chat.q4_k_m.gguf
```

### Download Requirements

The Docker container includes both `wget` and `curl` for downloading models. If running locally, ensure you have one of these tools installed.

### Health Check Enhancement

The `/health` endpoint now provides detailed information about the model status:

```bash
curl http://localhost:8000/health | jq .
```

Response includes:
- Whether the model is loaded
- Local model path and existence
- Configured model filename and download link

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

