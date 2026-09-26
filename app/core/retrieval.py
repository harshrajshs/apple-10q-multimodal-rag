from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from .models import Chunk, RetrievalHit
from .config import settings


class HybridRetriever:
    """
    Modality-aware hybrid retriever.

    Retrieval is performed independently for:
        - text
        - table
        - figure

    Each modality uses:
        - dense retrieval with FAISS
        - lexical retrieval with BM25

    Candidates from all modalities are then fused and reranked
    using a cross-encoder.

    For tables, a structured relevance signal is additionally
    used because generic text cross-encoders do not always rank
    structured financial tables correctly.
    """

    MODALITIES = ("text", "table", "figure")

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

        # ---------------------------------------------------------
        # 1. Group chunks by modality
        # ---------------------------------------------------------
        self.modality_indices: dict[str, list[int]] = defaultdict(list)

        for idx, chunk in enumerate(self.chunks):
            modality = chunk.modality or "text"
            self.modality_indices[modality].append(idx)

        # ---------------------------------------------------------
        # 2. Load models once
        # ---------------------------------------------------------
        self.embedder = SentenceTransformer(
            settings.embedding_model
        )

        self.reranker = CrossEncoder(
            settings.reranker_model
        )

        # ---------------------------------------------------------
        # 3. Build one FAISS + BM25 index per modality
        # ---------------------------------------------------------
        self.faiss_indices: dict[str, faiss.Index] = {}
        self.bm25_indices: dict[str, BM25Okapi] = {}

        for modality, global_ids in self.modality_indices.items():

            texts = [
                self.chunks[idx].text
                for idx in global_ids
            ]

            if not texts:
                continue

            # ---------- Dense index ----------
            embeddings = self.embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

            index.add(embeddings)

            self.faiss_indices[modality] = index

            # ---------- BM25 index ----------
            tokenized = [
                self._tokens(text)
                for text in texts
            ]

            self.bm25_indices[modality] = BM25Okapi(
                tokenized
            )

    # =============================================================
    # TOKENIZATION
    # =============================================================

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """
        Tokenization used by BM25.

        Keeps numbers, currency symbols and financial
        terminology reasonably intact.
        """
        return re.findall(
            r"[A-Za-z0-9$%.,()-]+",
            text.lower(),
        )

    # =============================================================
    # RECIPROCAL RANK FUSION
    # =============================================================

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        """
        Reciprocal Rank Fusion score.

        RRF avoids directly combining incompatible
        FAISS cosine scores and BM25 scores.
        """
        return 1.0 / (k + rank)

    # =============================================================
    # MODALITY RETRIEVAL
    # =============================================================

    def _retrieve_modality(
        self,
        question: str,
        modality: str,
        dense_k: int,
        bm25_k: int,
    ) -> dict[int, float]:

        if modality not in self.modality_indices:
            return {}

        global_ids = self.modality_indices[modality]

        if not global_ids:
            return {}

        faiss_index = self.faiss_indices[modality]
        bm25 = self.bm25_indices[modality]

        # ---------------------------------------------------------
        # Dense retrieval
        # ---------------------------------------------------------

        qvec = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        dense_k = min(
            dense_k,
            len(global_ids),
        )

        dense_scores, dense_local_ids = faiss_index.search(
            qvec,
            dense_k,
        )

        candidates: dict[int, float] = {}

        for rank, local_idx in enumerate(
            dense_local_ids[0],
            start=1,
        ):
            if local_idx < 0:
                continue

            global_idx = global_ids[int(local_idx)]

            candidates[global_idx] = (
                candidates.get(global_idx, 0.0)
                + self._rrf(rank)
            )

        # ---------------------------------------------------------
        # BM25 retrieval
        # ---------------------------------------------------------

        bm_scores = bm25.get_scores(
            self._tokens(question)
        )

        bm25_k = min(
            bm25_k,
            len(global_ids),
        )

        bm_local_ids = np.argsort(
            bm_scores
        )[::-1][:bm25_k]

        for rank, local_idx in enumerate(
            bm_local_ids,
            start=1,
        ):
            global_idx = global_ids[int(local_idx)]

            candidates[global_idx] = (
                candidates.get(global_idx, 0.0)
                + self._rrf(rank)
            )

        return candidates

    # =============================================================
    # STRUCTURED TABLE RELEVANCE
    # =============================================================

    @staticmethod
    def _normalize_tokens(text: str) -> set[str]:
        """
        Normalize text into comparable lexical tokens.

        This intentionally remains simple and deterministic.
        """
        return set(
            re.findall(
                r"[a-z0-9]+",
                text.lower(),
            )
        )

    @staticmethod
    def _extract_date_phrases(text: str) -> set[str]:
        """
        Extract date phrases such as:

            June 25, 2022
            June 26, 2021

        from table/query text.
        """
        pattern = (
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{1,2},\s+\d{4}"
        )

        return {
            match.lower()
            for match in re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        }

    @staticmethod
    def _extract_period_phrases(text: str) -> set[str]:
        """
        Extract financial reporting periods.

        Examples:
            nine months ended
            three months ended
        """
        pattern = (
            r"(?:three|six|nine|twelve)\s+months?\s+ended"
        )

        return {
            re.sub(
                r"\s+",
                " ",
                match.lower(),
            )
            for match in re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        }

    @classmethod
    def _table_structured_score(
        cls,
        question: str,
        table_text: str,
    ) -> float:
        """
        Calculate structured relevance for financial tables.

        The score focuses on information that has explicit structure
        in SEC-style financial tables:

            - financial row/metric
            - reporting period
            - reporting date
            - general lexical overlap

        Returns a value in [0, 1].
        """

        question_lower = question.lower()
        table_lower = table_text.lower()

        # ---------------------------------------------------------
        # 1. Extract dates
        # ---------------------------------------------------------

        query_dates = cls._extract_date_phrases(
            question
        )

        table_dates = cls._extract_date_phrases(
            table_text
        )

        if query_dates:
            date_score = len(
                query_dates.intersection(table_dates)
            ) / len(query_dates)
        else:
            date_score = 0.0

        # ---------------------------------------------------------
        # 2. Extract reporting periods
        # ---------------------------------------------------------

        query_periods = cls._extract_period_phrases(
            question
        )

        table_periods = cls._extract_period_phrases(
            table_text
        )

        if query_periods:
            period_score = len(
                query_periods.intersection(
                    table_periods
                )
            ) / len(query_periods)
        else:
            period_score = 0.0

        # ---------------------------------------------------------
        # 3. Extract table row labels
        # ---------------------------------------------------------

        row_labels = re.findall(
            r"TABLE ROW:\s*(.+)",
            table_text,
            flags=re.IGNORECASE,
        )

        row_labels = [
            re.sub(
                r"\s+",
                " ",
                label.strip().lower(),
            )
            for label in row_labels
        ]

        # ---------------------------------------------------------
        # 4. Detect important financial metric phrases
        # ---------------------------------------------------------

        financial_terms = [
            "net sales",
            "total net sales",
            "gross margin",
            "operating income",
            "net income",
            "cost of sales",
            "operating expenses",
            "research and development",
            "selling, general and administrative",
            "earnings per share",
            "diluted earnings per share",
            "basic earnings per share",
            "cash and cash equivalents",
            "total assets",
            "total liabilities",
            "shareholders' equity",
            "cash flow",
        ]

        query_metrics = [
            term
            for term in financial_terms
            if term in question_lower
        ]

        metric_score = 0.0

        if query_metrics:
            matched_metrics = 0

            for metric in query_metrics:

                # Exact metric phrase appears in a row label.
                if any(
                    metric in row
                    for row in row_labels
                ):
                    matched_metrics += 1

            metric_score = (
                matched_metrics
                / len(query_metrics)
            )

        # ---------------------------------------------------------
        # 5. General lexical overlap
        # ---------------------------------------------------------

        q_tokens = cls._normalize_tokens(
            question
        )

        t_tokens = cls._normalize_tokens(
            table_text
        )

        if q_tokens:
            lexical_score = len(
                q_tokens.intersection(t_tokens)
            ) / len(q_tokens)
        else:
            lexical_score = 0.0

        # ---------------------------------------------------------
        # 6. Direct phrase match
        #
        # If the exact metric phrase and period/date all appear
        # in the same table, this is particularly strong evidence.
        # ---------------------------------------------------------

        direct_match = 0.0

        if (
            metric_score > 0
            and period_score > 0
            and date_score > 0
        ):
            direct_match = 1.0

        # ---------------------------------------------------------
        # 7. Final structured score
        # ---------------------------------------------------------

        score = (
            0.35 * metric_score
            + 0.25 * period_score
            + 0.25 * date_score
            + 0.10 * direct_match
            + 0.05 * lexical_score
        )

        return float(
            max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )
        )

    @staticmethod
    def _table_evidence_quality(table_text: str) -> float:
        """
        Estimate how clearly a table represents key/value relationships.

        Explicit mappings such as:

            Nine Months Ended June 25, 2022: 304,182

        are preferred over flattened representations such as:

            VALUES: 82,959 | 81,434 | 2 | % | ...

        Returns a value in [0, 1].
        """

        explicit_mappings = re.findall(
            r"^\s*(?:Three|Six|Nine|Twelve)\s+Months\s+Ended.+?:\s*.+$",
            table_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        value_rows = re.findall(
            r"^\s*VALUES:\s*.+$",
            table_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        row_labels = re.findall(
            r"^\s*TABLE ROW:\s*.+$",
            table_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # Explicit period -> value mappings are strong evidence.
        mapping_score = min(
            len(explicit_mappings) / 4.0,
            1.0,
        )

        # Having meaningful row labels improves interpretability.
        row_score = min(
            len(row_labels) / 5.0,
            1.0,
        )

        # Flattened VALUES rows reduce interpretability.
        if value_rows:
            ambiguity_penalty = min(
                len(value_rows) / 5.0,
                1.0,
            )
        else:
            ambiguity_penalty = 0.0

        score = (
            0.60 * mapping_score
            + 0.40 * row_score
            - 0.20 * ambiguity_penalty
        )

        return float(
            max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )
        )

    def _select_diverse_evidence(
    self,
    ranked: list[tuple[int, float]],
    question: str,
    ) -> list[tuple[int, float]]:
        """
        Select final evidence while preserving complementary
        modalities and highly relevant structured tables.

        The ranking score remains the primary signal. A small
        amount of diversity is introduced only when a candidate
        is strongly relevant.
        """

        selected: list[tuple[int, float]] = []

        # ---------------------------------------------------------
        # First pass: normal ranking
        # ---------------------------------------------------------

        for idx, score in ranked:

            if len(selected) >= settings.top_k_final:
                break

            selected.append((idx, score))

        # ---------------------------------------------------------
        # Identify high-confidence table candidates that were
        # excluded by the normal top-k cutoff.
        # ---------------------------------------------------------

        selected_ids = {
            idx
            for idx, _ in selected
        }

        excluded_tables = []

        for idx, score in ranked:

            if idx in selected_ids:
                continue

            chunk = self.chunks[idx]

            if chunk.modality != "table":
                continue

            structured_score = (
                self._table_structured_score(
                    question,
                    chunk.text,
                )
            )

            evidence_quality = (
                self._table_evidence_quality(
                    chunk.text
                )
            )

            if (
                structured_score >= 0.80
                and evidence_quality >= 0.35
            ):
                excluded_tables.append(
                    (
                        idx,
                        score,
                        structured_score,
                        evidence_quality,
                    )
                )

        # ---------------------------------------------------------
        # Replace weakest non-table result with strongest
        # excluded high-confidence table.
        # ---------------------------------------------------------

        if excluded_tables and selected:

            excluded_tables.sort(
                key=lambda x: (
                    x[2] * 0.70
                    + x[3] * 0.30
                ),
                reverse=True,
            )

            table_idx = excluded_tables[0][0]

            replaceable = [
                (
                    position,
                    idx,
                    score,
                )
                for position, (idx, score)
                in enumerate(selected)
                if self.chunks[idx].modality != "table"
            ]

            if replaceable:

                weakest_position, _, _ = min(
                    replaceable,
                    key=lambda x: x[2],
                )

                table_score = next(
                    score
                    for idx, score in ranked
                    if idx == table_idx
                )

                selected[weakest_position] = (
                    table_idx,
                    table_score,
                )

        # ---------------------------------------------------------
        # Re-sort after evidence substitution.
        # ---------------------------------------------------------

        selected.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return selected

    # =============================================================
    # FINAL SEARCH
    # =============================================================

    def search(
        self,
        question: str,
    ) -> list[RetrievalHit]:

        # =========================================================
        # STEP 1: Retrieve independently from every modality
        # =========================================================

        candidate_scores: dict[int, float] = {}

        modality_k = {
            "text": 8,
            "table": 8,
            "figure": 4,
        }

        for modality in self.MODALITIES:

            candidates = self._retrieve_modality(
                question=question,
                modality=modality,
                dense_k=modality_k.get(
                    modality,
                    5,
                ),
                bm25_k=modality_k.get(
                    modality,
                    5,
                ),
            )

            for idx, score in candidates.items():

                candidate_scores[idx] = max(
                    candidate_scores.get(idx, 0.0),
                    score,
                )

        # =========================================================
        # STEP 2: Cross-encoder reranking
        # =========================================================

        candidate_ids = list(
            candidate_scores.keys()
        )

        if not candidate_ids:
            return []

        pairs = [
            (
                question,
                self.chunks[idx].text,
            )
            for idx in candidate_ids
        ]

        rerank_scores = self.reranker.predict(
            pairs
        )

        # =========================================================
        # STEP 3: Normalize cross-encoder scores
        # =========================================================

        rerank_scores = np.asarray(
            rerank_scores,
            dtype=np.float32,
        )

        if (
            len(rerank_scores) > 1
            and float(rerank_scores.max())
            != float(rerank_scores.min())
        ):
            rerank_min = float(
                rerank_scores.min()
            )

            rerank_max = float(
                rerank_scores.max()
            )

            normalized_rerank = (
                rerank_scores - rerank_min
            ) / (
                rerank_max - rerank_min
            )

        else:
            normalized_rerank = np.ones(
                len(rerank_scores),
                dtype=np.float32,
            )

        # =========================================================
        # STEP 4: Fuse cross-encoder + structured table signal
        # =========================================================

        final_scores = []

        for idx, rerank_score in zip(
            candidate_ids,
            normalized_rerank,
        ):

            chunk = self.chunks[idx]

            # Default: cross-encoder drives ranking.
            final_score = float(
                0.85 * rerank_score
            )

            # Tables receive an additional structured
            # relevance signal.
            if chunk.modality == "table":

                structured_score = (
                    self._table_structured_score(
                        question,
                        chunk.text,
                    )
                )

                final_score += (
                    0.15 * structured_score
                )

            final_scores.append(
                (
                    idx,
                    final_score,
                )
            )

        # =========================================================
        # STEP 5: Final ranking
        # =========================================================

        ranked = sorted(
            final_scores,
            key=lambda x: x[1],
            reverse=True,
        )

        # =========================================================
        # STEP 6: Select final evidence
        # =========================================================

        selected = self._select_diverse_evidence(
            ranked=ranked,
            question=question,
        )

        # =========================================================
        # STEP 7: Build final retrieval results
        # =========================================================

        hits: list[RetrievalHit] = []

        for idx, score in selected:

            hits.append(
                RetrievalHit(
                    chunk=self.chunks[idx],
                    score=float(score),
                    source_type=(
                        "modality-aware-hybrid"
                        "+reranker"
                        "+structured-table"
                        "+evidence-diversity"
                    ),
                )
            )

        return hits


def load_retriever(
    chunks_path: str,
) -> HybridRetriever:

    raw = json.loads(
        Path(chunks_path).read_text(
            encoding="utf-8"
        )
    )

    chunks = [
        Chunk(**item)
        for item in raw
    ]

    return HybridRetriever(chunks)