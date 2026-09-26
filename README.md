# Apple 10-Q Multimodal RAG — Production-Style Information Retrieval

A production-oriented Retrieval-Augmented Generation (RAG) system for answering questions over the **2022 Q3 Apple 10-Q PDF**, including:

- narrative text
- financial tables
- figures / embedded images
- page-level citations and retrieval provenance

The design intentionally separates ingestion, retrieval, generation, and API concerns so the system can evolve from a single-document take-home exercise into a multi-document service.

## Architecture

```text
                    ┌─────────────────────────┐
                    │       Source PDF        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Ingestion Layer     │
                    │ PyMuPDF + pdfplumber     │
                    │ text / tables / images   │
                    └──────┬─────────┬─────────┘
                           │         │
             ┌─────────────▼──┐   ┌─▼────────────────┐
             │ Text/Table      │   │ Figure pipeline  │
             │ normalization   │   │ render/extract   │
             │ + chunking      │   │ + vision caption │
             └───────┬─────────┘   └───────┬──────────┘
                     │                     │
                     └──────────┬──────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ Unified Document Chunks   │
                  │ type/page/source metadata │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ Hybrid Retrieval           │
                  │ BM25 + dense embeddings   │
                  │ + cross-encoder reranker  │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ Context assembler          │
                  │ diversity + page grouping │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ Grounded LLM generation    │
                  │ answer + citations        │
                  └─────────────┬─────────────┘
                                ▼
                       FastAPI /query endpoint
```

## Why this design

A plain vector RAG pipeline is insufficient for this assignment because a 10-Q contains highly structured financial information. A number such as revenue or operating income may occur in multiple sections and tables. The system therefore preserves **content modality and page provenance** and retrieves across modalities.

Key design choices:

1. **Structure-aware ingestion** — text, tables and figures become first-class retrievable objects.
2. **Hybrid retrieval** — lexical retrieval catches exact financial terminology and numbers; dense retrieval catches semantic paraphrases.
3. **Reranking** — a cross-encoder improves precision on the final candidate set.
4. **Grounded generation** — the LLM is explicitly instructed to answer only from retrieved evidence and cite pages.
5. **Abstention** — if evidence is insufficient, the system says so rather than inventing an answer.
6. **Observability hooks** — request IDs, latency, retrieval scores and evidence metadata are returned.
7. **Config-driven components** — embedding/reranking/LLM choices are environment-configurable.

## Repository layout

```text
apple-multimodal-rag/
├── app/
│   ├── api/main.py
│   └── core/
│       ├── config.py
│       ├── models.py
│       ├── ingest.py
│       ├── retrieval.py
│       ├── generation.py
│       └── pipeline.py
├── data/
│   └── README.md
├── scripts/
│   └── ingest.py
├── tests/
│   ├── test_chunking.py
│   └── test_api_contract.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Quick start

### 1. Python environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Put the supplied PDF in `data/`

Download the assignment PDF from the provided source:

```text
https://github.com/docugami/KG-RAG-datasets/blob/main/sec-10-q/data/v1/docs/2022%20Q3%20AAPL.pdf
```

Save it as:

```text
data/2022_Q3_AAPL.pdf
```

### 3. Configure the LLM

```bash
cp .env.example .env
# set OPENAI_API_KEY
```

The retrieval stack can run locally; generation and figure captioning use the configured OpenAI model.

### 4. Ingest

```bash
python scripts/ingest.py --pdf data/2022_Q3_AAPL.pdf --out artifacts/index
```

### 5. Run the API

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What was Apple'\''s net sales for the nine months ended June 25, 2022?"}'
```

The response contains the answer plus evidence objects with page numbers, modality and retrieval scores.

## Production considerations

For a production deployment I would replace the local FAISS/BM25 stores with a managed search service such as Azure AI Search/OpenSearch/Elasticsearch, move figure captioning to an asynchronous worker queue, persist document/version hashes, add authentication/rate limits, and add evaluation telemetry.

The code intentionally keeps these boundaries clean so those changes do not require rewriting the API or generation layer.
