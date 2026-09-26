# Apple 10-Q Multimodal RAG Information Retrieval System

A production-oriented **Retrieval-Augmented Generation (RAG)** system for answering questions over the Apple Inc. Q3 2022 Form 10-Q PDF, including **narrative text, financial tables, and figures**.

## 1. Assignment

The assessment requires a RAG-based information retrieval system capable of answering questions about:

- Text content in the PDF
- Tables contained in the PDF
- Figures present in the PDF

The submission is designed as a maintainable Python service rather than a notebook-only prototype. It includes multimodal ingestion, hybrid retrieval, reranking, grounded generation, citations, abstention, automated evaluation, tests, and Docker deployment.

## 2. Key Design Idea

Text, tables, and figures are treated as different information modalities during ingestion and retrieval.

```text
                         Apple 10-Q PDF
                               |
                               v
                    +----------------------+
                    |    Ingestion Layer   |
                    | PyMuPDF / pdfplumber |
                    | Figure rendering     |
                    +----------+-----------+
                               |
                +--------------+--------------+
                |              |              |
                v              v              v
              TEXT           TABLE         FIGURE
                |              |              |
                +--------------+--------------+
                               |
                               v
                    Typed multimodal chunks
                               |
                +--------------+--------------+
                |                             |
                v                             v
          BM25 lexical search          FAISS dense search
                |                             |
                +--------------+--------------+
                               |
                               v
                    Hybrid candidate pool
                               |
                               v
                     Cross-encoder reranker
                               |
                               v
                  Table-aware evidence score
                               |
                               v
                    Evidence diversification
                               |
                               v
                  Grounded LLM generation
                         /          \
                        v            v
                     Answer      Citations
                               |
                               v
                            FastAPI
```

## 3. Repository Structure

```text
apple_multimodal_rag/
├── app/
│   ├── api/
│   │   └── main.py
│   └── core/
│       ├── config.py
│       ├── models.py
│       ├── ingest.py
│       ├── retrieval.py
│       ├── generation.py
│       └── pipeline.py
├── scripts/
│   ├── ingest.py
│   └── evaluate.py
├── tests/
│   ├── test_chunking.py
│   └── test_api_contract.py
├── artifacts/
│   ├── Apple_10Q_Multimodal_RAG_Design_Report.pdf
│   └── evaluation_results.json
├── data/
│   └── README.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The supplied source PDF, generated retrieval index, `.env`, and API secrets are intentionally excluded from Git.

## 4. Ingestion Pipeline

### 4.1 Text extraction

PyMuPDF is used for page-level text extraction. Text is retained with page metadata so generated answers can cite the source page.

Each chunk contains structured metadata:

```text
chunk_id
document_id
page
modality
text
source
bbox (when available)
image_path (when applicable)
```

### 4.2 Table extraction

Financial tables are treated as first-class retrieval objects rather than relying only on flattened page text.

`pdfplumber` is used to detect/extract tables. The extracted cells are cleaned and reconstructed into a semantic representation containing:

- Table title/context
- Explicit column definitions
- Row labels
- Period-to-value mappings
- Units and relevant footnote context

For example, the income statement is represented conceptually as:

```text
TABLE: Condensed Consolidated Statements Of Operations

COLUMNS:
  Three Months Ended June 25, 2022
  Three Months Ended June 26, 2021
  Nine Months Ended June 25, 2022
  Nine Months Ended June 26, 2021

TABLE ROW: Total net sales
  Three Months Ended June 25, 2022: 82,959
  Three Months Ended June 26, 2021: 81,434
  Nine Months Ended June 25, 2022: 304,182
  Nine Months Ended June 26, 2021: 282,457
```

This transformation is important because PDF table extraction can otherwise separate headers and numeric cells, causing a RAG system to lose the relationship between a metric, reporting period, and value.

### 4.3 Figure handling

Figures are retained as a separate modality. Relevant pages are rendered and the image path is preserved in the chunk metadata. The architecture also supports optional vision-derived descriptions so visual information can participate in retrieval and grounded generation.

This prevents the system from assuming that all useful information in a PDF exists in its text layer.

### 4.4 Chunking

Chunks are modality-aware rather than using one universal chunking strategy.

- Text: semantic/page-aware chunks
- Tables: table-level structured chunks
- Figures: figure/page-level visual evidence

The goal is to preserve enough context for retrieval without destroying the structure needed for financial questions.

## 5. Retrieval Architecture

### 5.1 Modality-aware retrieval

Retrieval is performed independently across text, table, and figure modalities. This prevents a large amount of narrative text from completely dominating table or figure evidence.

### 5.2 BM25

BM25 provides lexical retrieval and is particularly useful for:

- Exact financial terminology
- Company-specific names
- Dates
- Metric names
- Numeric/entity-heavy queries

### 5.3 Dense retrieval

Sentence-transformer embeddings provide semantic retrieval so that paraphrased questions can retrieve relevant evidence even when exact wording differs.

FAISS is used for efficient vector similarity search.

### 5.4 Hybrid retrieval

BM25 and dense candidates are combined before reranking. This combines lexical precision with semantic recall.

### 5.5 Cross-encoder reranking

The candidate set is reranked using a cross-encoder:

```text
Question + Candidate Chunk
          |
          v
    Cross-encoder
          |
          v
  relevance score
```

This is more precise than relying solely on independent query/document embeddings.

### 5.6 Table-aware scoring

During evaluation, a failure mode was identified: a generic cross-encoder can rank a semantically related narrative paragraph above the exact financial table.

For table chunks, the final score therefore combines the normalized reranker score with a structured relevance signal based on:

- Metric matching
- Reporting-period matching
- Date matching
- Direct value matching
- Lexical relevance

The implemented weighting gives the cross-encoder 85% of the base score and the structured table signal 15%.

This is intentionally a targeted correction rather than replacing the general reranker with a hand-built ranking system.

### 5.7 Evidence diversity

The final evidence selection avoids returning only near-duplicate evidence. It preserves strong table evidence when necessary so that numeric questions retain access to the actual structured source.

## 6. Generation and Grounding

The generation layer receives only retrieved evidence and is instructed to answer from that evidence.

Key rules:

1. Prefer direct table evidence for numerical questions.
2. Do not invent unsupported facts.
3. Use source-page citations.
4. Preserve figure modality when answering figure questions.
5. Abstain when the document does not contain enough evidence.

### Citation validation

Generated citations are parsed and validated against the retrieved evidence. A citation is considered valid only when its page/modality corresponds to retrieved evidence.

Supported citation forms include examples such as:

```text
[p.4]
[p.4, table]
[p.1, figure]
[p.4, table; p.10; p.18, table]
```

This creates a second grounding layer after retrieval: the answer may only cite evidence that was actually retrieved.

## 7. Abstention

The system includes an abstention path for unsupported questions.

For example, a question asking for revenue from a product launched in 2035 is outside the supplied 2022 document. The system should state that the evidence is insufficient instead of fabricating an answer.

Abstention behavior is included in the evaluation suite.

## 8. API

The service exposes:

### `GET /health`

Liveness endpoint.

### `GET /ready`

Readiness endpoint. Confirms that the RAG pipeline has loaded successfully.

### `POST /query`

Example:

```json
{
  "question": "What was Apple's total net sales for the nine months ended June 25, 2022?"
}
```

The response includes:

```json
{
  "answer": "...",
  "citations": [
    {
      "page": 4,
      "modality": "text"
    }
  ],
  "latency_ms": 5000,
  "evidence": []
}
```

The evidence field exposes the retrieved source metadata for transparency and debugging.

## 9. Evaluation

The final evaluation contains 10 questions covering:

- Text retrieval
- Financial tables
- Geographic revenue data
- A numerical calculation
- Figure understanding
- Unsupported-question abstention

The final measured results are:

| Metric | Result |
|---|---:|
| Questions | 10 |
| Recall@1 | 60.00% |
| Recall@5 | 100.00% |
| Recall@8 | 100.00% |
| MRR | 0.7750 |
| Answer accuracy | 100.00% |
| Citation accuracy | 100.00% |
| Abstention accuracy | 100.00% |
| Mean latency | 5724.55 ms |
| Median latency | 5042.39 ms |
| Max latency | 8178.92 ms |

### Interpreting the results

Recall@1 is 60%, while Recall@5 and Recall@8 are both 100%. This indicates that relevant evidence is consistently recovered in the candidate set, although the highest-ranked item is not always the exact target table.

The answer, citation, and abstention checks all pass for the final 10-question benchmark.

### Percentage calculation benchmark

One question asks for the increase in total net sales between the two nine-month periods.

The mathematical calculation is approximately:

```text
(304,182 - 282,457) / 282,457 × 100 ≈ 7.69%
```

The source document reports the change rounded to 8%. The evaluation therefore accepts `7.69`, `7.7`, or `8%` so that the benchmark recognizes both the calculated value and the document-reported rounded value.

## 10. Tests

The project includes automated tests for core chunking/preprocessing behavior and the API contract.

Run:

```powershell
pytest -q
```

The validated test run passed all 4 tests.

## 11. Local Setup

### Prerequisites

- Python 3.11+
- Git
- OpenAI API key for generation
- Supplied Apple 2022 Q3 10-Q PDF

### Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure environment

Copy:

```text
.env.example
```

to:

```text
.env
```

and set:

```text
OPENAI_API_KEY=your_key
```

Do not commit `.env`.

### Add the source document

Place the supplied PDF at:

```text
data/2022_Q3_AAPL.pdf
```

The PDF is intentionally not included in Git.

### Run ingestion

From the repository root:

```powershell
$env:PYTHONPATH="."
python -m scripts.ingest
```

The final ingestion run produced:

```text
60 chunks
28 text chunks
31 table chunks
1 figure chunk
```

The generated index is written under:

```text
artifacts/index/
```

This directory is intentionally ignored by Git.

## 12. Run the API locally

```powershell
$env:PYTHONPATH="."
uvicorn app.api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## 13. Docker

Build and start:

```powershell
docker compose up --build -d
```

Check:

```powershell
docker compose ps
```

The API exposes port `8000`.

Readiness:

```text
http://localhost:8000/ready
```

Swagger:

```text
http://localhost:8000/docs
```

The container includes the application, artifacts, and runtime dependencies while keeping secrets supplied through `.env`.

## 14. Reproducibility

The repository intentionally excludes:

- API secrets
- `.env`
- The supplied source PDF
- Generated FAISS/index artifacts
- Evaluation output that is regenerated by the evaluator when appropriate

The source PDF and index can therefore be reproduced using the documented ingestion command.

## 15. Security

Secrets are loaded from environment variables and `.env` is ignored by Git.

`.env.example` contains configuration names but no credentials.

The repository has been checked to ensure:

```text
.env                 -> not tracked
data/2022_Q3_AAPL.pdf -> not tracked
artifacts/index/     -> not tracked
```

## 16. Challenges and Solutions

| Challenge | Solution |
|---|---|
| PDF text is layout-oriented | Page-aware extraction and metadata |
| Table structure can be lost during extraction | Explicit column/row reconstruction |
| Repeated financial dates can become ambiguous | Explicit period-to-value mappings |
| Generic reranking can prefer narrative text | Table-aware structured scoring |
| Exact wording can be missed | BM25 + dense retrieval |
| Semantic retrieval can miss exact financial terminology | Hybrid retrieval |
| Figures contain information outside the text layer | Separate figure modality and rendering |
| LLM can hallucinate unsupported information | Evidence-only generation |
| Generated citations may not match retrieved evidence | Citation validation |
| Unsupported questions can trigger guesses | Abstention logic |
| Large source/index files are unsuitable for Git | `.gitignore` and reproducible ingestion |
| Local and deployment environments differ | Docker/Compose |

## 17. Assumptions and Limitations

### Assumptions

- The supplied PDF is the authoritative source for the assignment.
- Questions are expected to be answerable from that document.
- The evaluation set uses lightweight deterministic answer checks rather than another LLM judge.
- The source document is available locally before ingestion.

### Limitations

- The current evaluation set contains 10 questions and is therefore a small benchmark.
- Mean latency is approximately 5.7 seconds on the tested environment.
- Figure understanding depends on the available visual representation/description path.
- Table extraction can be difficult for highly irregular layouts, merged cells, or complex footnotes.
- The current index is local rather than a persistent production vector database.
- Authentication, rate limiting, distributed tracing, and horizontal scaling are outside the assignment scope.

## 18. Production Improvements

Potential next steps include:

1. OCR fallback for scanned/image-only PDFs.
2. Cell-level table coordinates and stronger merged-cell/footnote handling.
3. Dedicated vision-language retrieval for complex figures.
4. Persistent vector and lexical indexes.
5. Query classification to adapt retrieval depth by modality.
6. Embedding/reranking caching and batching.
7. Larger regression and human-evaluation datasets.
8. CI/CD evaluation gates.
9. Authentication, rate limiting, tracing, and observability.
10. Latency optimization through model warm-up and cached inference.

## 19. Design Report

A separate detailed methodology/design report is included at:

```text
artifacts/Apple_10Q_Multimodal_RAG_Design_Report.pdf
```

It provides the detailed design rationale, architecture, assumptions, challenges, production hardening roadmap, and evaluation methodology.

## 20. Submission

GitHub repository:

https://github.com/harshrajshs/apple-10q-multimodal-rag

The repository contains the complete implementation, Docker configuration, tests, evaluation artifacts, and design documentation required for the assessment.
