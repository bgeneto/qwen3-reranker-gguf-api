import os, asyncio, logging, math, uvicorn
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama

from .auth import verify_token
from .metrics import REQUEST_COUNT, FAIL_COUNT, LATENCY, metrics_endpoint
from .logger import setup_logging
from .config import settings
from .model_downloader import ensure_model_available, get_model_info

setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# model init
# ------------------------------------------------------------------
llm: Optional[Llama] = None


def get_llm():
    global llm
    if llm is None:
        try:
            # Ensure model is available (download if necessary)
            logger.info("Ensuring model is available...")
            model_path = ensure_model_available()
            logger.info(f"Model path resolved to: {model_path}")

            # Verify the model file exists and is readable
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            if not os.access(model_path, os.R_OK):
                raise PermissionError(f"Cannot read model file at: {model_path}")

            file_size = os.path.getsize(model_path)
            logger.info(f"Model file size: {file_size} bytes")

            if file_size == 0:
                raise ValueError(f"Model file is empty: {model_path}")

            logger.info("Initializing Llama model...")

            # Add more detailed logging for debugging
            logger.info(f"Model initialization parameters:")
            logger.info(f"  - model_path: {model_path}")
            logger.info(f"  - n_ctx: {settings.N_CTX}")
            logger.info(f"  - n_batch: {settings.N_BATCH}")
            logger.info(f"  - n_threads: {settings.N_THREADS}")
            logger.info(f"  - n_gpu_layers: {settings.N_GPU_LAYERS}")

            # Try to load the model with different configurations for reranking
            load_attempts = [
                # First attempt: proper reranking configuration with RANK pooling
                {
                    "n_gpu_layers": settings.N_GPU_LAYERS,
                    "n_ctx": settings.N_CTX,
                    "n_batch": settings.N_BATCH,
                    "n_threads": settings.N_THREADS,
                    "embedding": True,  # Enable embedding mode for reranking
                    "pooling_type": 4,  # LLAMA_POOLING_TYPE_RANK (with PR #14029 fallback)
                    "logits_all": False,  # Not needed for reranking
                    "verbose": True,
                },
                # Second attempt: try CLS pooling instead of RANK
                {
                    "n_gpu_layers": settings.N_GPU_LAYERS,
                    "n_ctx": settings.N_CTX,
                    "n_batch": settings.N_BATCH,
                    "n_threads": settings.N_THREADS,
                    "embedding": True,
                    "pooling_type": 1,  # LLAMA_POOLING_TYPE_CLS
                    "logits_all": False,
                    "verbose": True,
                },
                # Third attempt: CPU-only reranking with RANK pooling
                {
                    "n_gpu_layers": 0,
                    "n_ctx": min(settings.N_CTX, 4096),
                    "n_batch": min(settings.N_BATCH, 256),
                    "n_threads": settings.N_THREADS,
                    "embedding": True,
                    "pooling_type": 4,  # LLAMA_POOLING_TYPE_RANK
                    "logits_all": False,
                    "verbose": True,
                },
                # Fourth attempt: CPU-only with CLS pooling
                {
                    "n_gpu_layers": 0,
                    "n_ctx": min(settings.N_CTX, 4096),
                    "n_batch": min(settings.N_BATCH, 256),
                    "n_threads": settings.N_THREADS,
                    "embedding": True,
                    "pooling_type": 1,  # LLAMA_POOLING_TYPE_CLS
                    "logits_all": False,
                    "verbose": True,
                },
                # Fifth attempt: fallback to legacy reranking with logits_all
                {
                    "n_gpu_layers": 0,
                    "n_ctx": 2048,
                    "n_batch": 128,
                    "n_threads": 1,
                    "logits_all": True,
                    "verbose": True,
                    "use_mmap": False,
                    "use_mlock": False,
                },
                # Sixth attempt: ultra-conservative settings
                {
                    "n_gpu_layers": 0,
                    "n_ctx": 512,
                    "n_batch": 32,
                    "n_threads": 1,
                    "logits_all": True,
                    "verbose": True,
                    "use_mmap": False,
                    "use_mlock": False,
                    "f16_kv": False,
                },
            ]

            last_error = None
            for i, config in enumerate(load_attempts):
                try:
                    if i > 0:
                        logger.info(
                            f"Attempting model load with fallback configuration {i}: {config}"
                        )

                    llm = Llama(model_path=model_path, **config)
                    logger.info(
                        f"Model loaded successfully with configuration: {config}"
                    )
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Model load attempt {i+1} failed: {e}")
                    if i == len(load_attempts) - 1:
                        # This was the last attempt
                        raise e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(
                f"Model settings: filename={settings.MODEL_FILENAME}, link={settings.MODEL_LINK}"
            )
            logger.error(f"Expected path: {settings.model_path}")

            # Additional diagnostic information
            try:
                from .model_downloader import verify_gguf_file, calculate_file_hash

                if os.path.exists(model_path):
                    logger.error(f"File exists: True")
                    logger.error(f"File size: {os.path.getsize(model_path)} bytes")
                    logger.error(f"File readable: {os.access(model_path, os.R_OK)}")
                    logger.error(f"Is valid GGUF: {verify_gguf_file(model_path)}")
                    file_hash = calculate_file_hash(model_path)
                    logger.error(f"File hash (SHA256): {file_hash[:32]}...")
                else:
                    logger.error(f"File exists: False")
            except Exception as diag_error:
                logger.error(f"Error during diagnostics: {diag_error}")

            raise RuntimeError(f"Failed to load model: {e}")
    return llm


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
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}\n<|im_start|>user\n"
    "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}\n"
    "<|im_start|>assistant\n"
)
INSTRUCTION = "Evaluate how relevant the following document is to the query for retrieving useful information to answer or provide context for the query."


async def rerank_one_modern(instruction: str, query: str, doc: str) -> float:
    """Modern reranking using embedding/pooling approach for newer models."""
    llm_instance = get_llm()
    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM,
        instruction=instruction,
        query=query,
        doc=doc,
    )

    def _run_embedding():
        # Use embedding mode which should work with pooling_type=4 (RANK)
        embedding_result = llm_instance.create_embedding(prompt)
        # For reranking models, the embedding should contain the relevance score
        return embedding_result

    try:
        result = await asyncio.to_thread(_run_embedding)
        # Extract the relevance score from the embedding result
        # The exact format may vary, but typically it's a single value for reranking
        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0]["embedding"]
            # For reranking models, the first component often represents the relevance score
            # We may need to apply sigmoid or softmax to normalize it to [0,1] range
            if len(embedding) > 0:
                score = embedding[0]
                # Apply sigmoid to map to [0,1] range
                return 1.0 / (1.0 + math.exp(-score))
            else:
                return 0.0
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"Modern reranking failed: {e}, falling back to legacy method")
        return await rerank_one_legacy(instruction, query, doc)


async def rerank_one_legacy(instruction: str, query: str, doc: str) -> float:
    """Legacy reranking using logits approach for compatibility."""
    llm_instance = get_llm()
    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM,
        instruction=instruction,
        query=query,
        doc=doc,
    )

    def _run_llm():
        return llm_instance(prompt, max_tokens=1, temperature=0.0)

    try:
        output = await asyncio.to_thread(_run_llm)

        yes_token = llm_instance.tokenize(b"yes")[-1]
        no_token = llm_instance.tokenize(b"no")[-1]

        yes_logit = output["choices"][0]["logits"][-1][yes_token]
        no_logit = output["choices"][0]["logits"][-1][no_token]
        lse = math.log(math.exp(yes_logit) + math.exp(no_logit))
        return math.exp(yes_logit - lse)
    except Exception as e:
        logger.error(f"Legacy reranking failed: {e}")
        raise


async def rerank_one(instruction: str, query: str, doc: str) -> float:
    """Adaptive reranking that chooses method based on RERANKING_MODE setting."""
    if settings.RERANKING_MODE == "modern":
        return await rerank_one_modern(instruction, query, doc)
    elif settings.RERANKING_MODE == "legacy":
        return await rerank_one_legacy(instruction, query, doc)
    else:  # auto mode
        global llm
        if llm is None:
            llm = get_llm()

        # Check if the model was loaded with embedding mode (indicating modern reranking support)
        try:
            # Try to access the model's embedding property or pooling type
            if hasattr(llm, "_model") and hasattr(llm._model, "pooling_type"):
                if llm._model.pooling_type == 4:  # LLAMA_POOLING_TYPE_RANK
                    logger.info("Using modern reranking (pooling-based)")
                    return await rerank_one_modern(instruction, query, doc)
        except Exception as e:
            logger.debug(f"Modern reranking check failed: {e}")

        # Fall back to legacy method
        logger.info("Using legacy reranking (logits-based)")
        return await rerank_one_legacy(instruction, query, doc)


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="Qwen3-GGUF-Reranker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.on_event("startup")
async def startup_event():
    """Load the model at startup"""
    try:
        logger.info("Starting up application...")
        get_llm()  # Load the model at startup
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise


# public
@app.get("/health")
async def health():
    try:
        model_info = get_model_info()
        llm_instance = get_llm()

        return {
            "status": "ok",
            "model": {
                "loaded": llm_instance is not None,
                "path": model_info["local_path"],
                "exists_locally": model_info["exists_locally"],
                "model_filename": model_info["model_filename"],
                "model_link": model_info["model_link"],
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@app.get("/metrics")
async def metrics():
    return metrics_endpoint()()


# protected
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    req: RerankRequest,
    _=Depends(verify_token),
):
    REQUEST_COUNT.inc()
    with LATENCY.time():
        try:
            if not req.documents:
                return RerankResponse(model=req.model, results=[])

            tasks = [rerank_one(INSTRUCTION, req.query, d.text) for d in req.documents]
            scores = await asyncio.gather(*tasks)

            results = [
                RerankResult(index=i, relevance_score=s) for i, s in enumerate(scores)
            ]
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            if req.top_n:
                results = results[: req.top_n]
            return RerankResponse(model=req.model, results=results)
        except Exception as e:
            FAIL_COUNT.inc()
            logger.exception("Rerank failed")
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        import uvloop

        # Set uvloop as the default event loop policy for better performance (Linux/Unix only)
        uvloop.install()
        logger.info("uvloop installed for better async performance")
    except ImportError:
        # uvloop is not available on Windows
        logger.info("uvloop not available, using default event loop")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        # Don't specify loop and http here when using uvloop.install()
    )
