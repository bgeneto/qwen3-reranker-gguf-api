Below is a **production-grade** Docker Compose stack that:

* Uses **CUDA-enabled llama-cpp-python** and off-loads the **entire** model to the GPU (`N_GPU_LAYERS=-1`).
* Reads **all** configuration from an `.env` file (model path, GPU layers, log level, etc.).
* Adds **async** file logging (JSON-L) that can be turned on/off with `LOG_TO_FILE`.
* Exposes a **metrics** endpoint (`/metrics`) with Prometheus-style counters & gauges.
* Protects every route (except `/health`) with **Bearer token** authentication.

Let's implement this code now. Correct any issues you find and apply improvements if you may.

──────────────────────────────────────────────
Directory layout
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
──────────────────────────────────────────────

1.  .env  (never commit real tokens)
```bash
# model
MODEL_PATH=/models/qwen3-4b-reranker-q4_k_m.gguf
N_CTX=8192
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

2.  Dockerfile  (CUDA base)
```dockerfile
# ---------- build ----------
FROM python:3.11-slim as builder
WORKDIR /build
COPY app/requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---------- runtime ----------
FROM nvidia/cuda:12.3-runtime-ubuntu22.04

# Python & system deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install wheels
COPY --from=builder /wheels /wheels
RUN pip3 install --no-cache /wheels/*

# Create log dir
RUN mkdir -p /var/log && chmod 777 /var/log

WORKDIR /srv
COPY app/ ./app/

EXPOSE 8000
CMD ["python3", "-m", "app.main"]
```

3.  requirements.txt
```
llama-cpp-python[server]==0.2.77
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-dotenv==1.0.1
prometheus-client==0.19.0
```

4.  logger.py  (async file + console)
```python
import os, json, logging, asyncio
from datetime import datetime
from typing import Any, Dict

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", "/var/log/reranker.jsonl")

class AsyncJSONFileHandler(logging.Handler):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename

    def emit(self, record: logging.LogRecord) -> None:
        payload: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
            "module": record.module,
            "lineno": record.lineno,
        }
        asyncio.create_task(self._write(payload))

    async def _write(self, payload: Dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: open(self.filename, "a").write(json.dumps(payload) + "\n"),
        )

def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)
    # console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)
    # file
    if LOG_TO_FILE:
        logger.addHandler(AsyncJSONFileHandler(LOG_FILE))
```

5.  auth.py
```python
import os
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

TOKEN = os.getenv("API_TOKEN", "change-me")
bearer_scheme = HTTPBearer(auto_error=False)

def verify_token(cred: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if not cred or cred.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
```

6.  metrics.py
```python
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# counters
REQUEST_COUNT = Counter("rerank_requests_total", "Total rerank requests")
FAIL_COUNT    = Counter("rerank_failures_total", "Total rerank failures")
LATENCY       = Histogram("rerank_latency_seconds", "Latency per rerank request")

def metrics_endpoint():
    def latest():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    return latest
```

7.  main.py  (FastAPI app)
```python
import os, asyncio, logging
from typing import List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from dotenv import load_dotenv

from .auth import verify_token
from .metrics import REQUEST_COUNT, FAIL_COUNT, LATENCY, metrics_endpoint
from .logger import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# model init
# ------------------------------------------------------------------
MODEL_PATH   = os.getenv("MODEL_PATH")
N_CTX      = int(os.getenv("N_CTX", "8192"))
N_BATCH    = int(os.getenv("N_BATCH", "512"))
N_THREADS  = int(os.getenv("N_THREADS", "0"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_batch=N_BATCH,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    logits_all=True,
    verbose=False,
)

YES_TOKEN = llm.tokenize(b"yes")[-1]
NO_TOKEN  = llm.tokenize(b"no")[-1]

# ------------------------------------------------------------------
# schema
# ------------------------------------------------------------------
class Document(BaseModel):
    text: str

class RerankRequest(BaseModel):
    model: str = "qwen3-reranker"
    query: str
    documents: List[Document]
    top_n: Optional[int] = None

class RerankResult(BaseModel):
    index: int
    relevance_score: float

class RerankResponse(BaseModel):
    model: str
    results: List[RerankResult]

# ------------------------------------------------------------------
# reranker logic
# ------------------------------------------------------------------
SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
)
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}\n<|im_start|>user\n"
    "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}\n"
    "<|im_start|>assistant\n

\n\n"
)

def rerank_one(instruction: str, query: str, doc: str) -> float:
    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM,
        instruction=instruction,
        query=query,
        doc=doc,
    )
    output = llm(prompt, max_tokens=1, temperature=0.0)
    yes_logit = output["logits"][-1][YES_TOKEN]
    no_logit  = output["logits"][-1][NO_TOKEN]
    import math
    lse = math.log(math.exp(yes_logit) + math.exp(no_logit))
    return math.exp(yes_logit - lse)

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="Qwen3-GGUF-Reranker")

# public
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return metrics_endpoint()

# protected
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    req: RerankRequest,
    _=Depends(verify_token),
):
    REQUEST_COUNT.inc()
    instruction = "Evaluate how relevant the following document is to the query for retrieving useful information to answer or provide context for the query."
    with LATENCY.time():
        try:
            scores = [rerank_one(instruction, req.query, d.text) for d in req.documents]
            results = [
                RerankResult(index=i, relevance_score=s)
                for i, s in enumerate(scores)
            ]
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            if req.top_n:
                results = results[:req.top_n]
            return RerankResponse(model=req.model, results=results)
        except Exception as e:
            FAIL_COUNT.inc()
            logger.exception("Rerank failed")
            raise e
```

8.  docker-compose.yml  (CUDA runtime)
```yaml
version: "3.9"
services:
  reranker:
    build: .
    container_name: qwen3-reranker
    restart: unless-stopped
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./models:/models:ro
      - ./logs:/var/log
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

9.  Run
```bash
# 1. place your GGUF in ./models
# 2. edit .env (especially API_TOKEN)
# 3. start
docker compose up --build -d

# 4. test (Bearer token required)
curl -s http://localhost:8000/v1/rerank \
  -H "Authorization: Bearer change-me-please" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What is the capital of China?",
        "documents": [{"text": "The capital of China is Beijing."}]
      }' | jq .
```

All requests are now **authenticated**, **logged**, **measured**, and the entire model runs on the GPU via CUDA.