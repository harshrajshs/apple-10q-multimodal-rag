from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("apple-rag-api")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_PATH = BASE_DIR / "artifacts" / "index" / "chunks.json"


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

PIPELINE: RAGPipeline | None = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language question about the Apple 10-Q document.",
    )


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE

    logger.info("Starting Apple 10-Q RAG API")
    logger.info("Loading retrieval index from %s", CHUNKS_PATH)

    if not CHUNKS_PATH.exists():
        logger.warning(
            "Chunk index does not exist at %s. "
            "Run ingestion before serving queries.",
            CHUNKS_PATH,
        )
        PIPELINE = None
    else:
        try:
            PIPELINE = RAGPipeline(str(CHUNKS_PATH))
            logger.info("RAG pipeline loaded successfully")
        except Exception:
            PIPELINE = None
            logger.exception("Failed to initialize RAG pipeline")

    yield

    logger.info("Shutting down Apple 10-Q RAG API")
    PIPELINE = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Apple 10-Q Multimodal RAG",
    version="1.0.0",
    description=(
        "Production-oriented multimodal RAG API for grounded question "
        "answering over Apple's 2022 Q3 10-Q, including text, tables, "
        "and figures."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
def health() -> HealthResponse:
    """
    Lightweight liveness endpoint.

    Returns HTTP 200 when the API process is running, regardless of whether
    the retrieval index has been loaded.
    """
    return HealthResponse(
        status="ok",
        index_loaded=PIPELINE is not None,
    )


@app.get(
    "/ready",
    response_model=HealthResponse,
    tags=["system"],
)
def readiness() -> HealthResponse:
    """
    Readiness endpoint used by deployment platforms/load balancers.
    """
    if PIPELINE is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    return HealthResponse(
        status="ready",
        index_loaded=True,
    )


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/query",
    tags=["rag"],
)
def query(req: QueryRequest, request: Request) -> dict:
    """
    Execute a grounded RAG query against the Apple 10-Q.
    """
    if PIPELINE is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    start = time.perf_counter()

    try:
        result = PIPELINE.query(question)

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        logger.info(
            "Query completed | latency_ms=%s | question_length=%s",
            elapsed_ms,
            len(question),
        )

        return result

    except Exception:
        logger.exception(
            "RAG query failed | question_length=%s",
            len(question),
        )

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the query.",
        ) from None