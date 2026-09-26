from __future__ import annotations

import time

from .generation import answer
from .retrieval import HybridRetriever, load_retriever


class RAGPipeline:
    def __init__(self, chunks_path: str):
        self.retriever: HybridRetriever = load_retriever(chunks_path)

    def query(self, question: str) -> dict:
        start = time.perf_counter()

        hits = self.retriever.search(question)

        generation_result = answer(
            question=question,
            hits=hits,
        )

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        return {
            "answer": generation_result.answer,
            "citations": [
                citation.to_dict()
                for citation in generation_result.citations
            ],
            "latency_ms": elapsed_ms,
            "evidence": [
                hit.to_dict()
                for hit in hits
            ],
        }