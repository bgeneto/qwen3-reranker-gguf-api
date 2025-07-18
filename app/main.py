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
            model_path = ensure_model_available()

            llm = Llama(
                model_path=model_path,
                n_ctx=settings.N_CTX,
                n_batch=settings.N_BATCH,
                n_threads=settings.N_THREADS,
                n_gpu_layers=settings.N_GPU_LAYERS,
                logits_all=True,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError("Failed to load model")
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


async def rerank_one(instruction: str, query: str, doc: str) -> float:
    llm_instance = get_llm()
    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM,
        instruction=instruction,
        query=query,
        doc=doc,
    )

    def _run_llm():
        return llm_instance(prompt, max_tokens=1, temperature=0.0)

    output = await asyncio.to_thread(_run_llm)

    yes_token = llm_instance.tokenize(b"yes")[-1]
    no_token = llm_instance.tokenize(b"no")[-1]

    yes_logit = output["choices"][0]["logits"][-1][yes_token]
    no_logit = output["choices"][0]["logits"][-1][no_token]
    lse = math.log(math.exp(yes_logit) + math.exp(no_logit))
    return math.exp(yes_logit - lse)


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
    get_llm()  # Load the model at startup


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
                "hf_repo": model_info["hf_repo"],
                "hf_filename": model_info["hf_filename"],
                "has_hf_token": model_info["has_hf_token"],
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
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        loop="uvloop",
        http="httptools",
    )
