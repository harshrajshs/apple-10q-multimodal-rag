from __future__ import annotations

import base64
import re
from pathlib import Path

from openai import OpenAI

from .config import settings
from .models import Citation, GenerationResult, RetrievalHit


SYSTEM = """You are a document-grounded financial information assistant.

You answer questions ONLY using the supplied evidence from the source document.

Rules:

1. Use only information explicitly supported by the supplied evidence.
2. Never invent figures, dates, table values, labels, trends, or explanations.
3. If the evidence is insufficient, say:
   "The document evidence is insufficient to answer this question."
4. For numerical questions, prefer exact values from table evidence when available.
5. Preserve the units stated in the evidence, such as millions, thousands,
   percentages, or per-share amounts.
6. Do not confuse three-month and nine-month periods.
7. Do not confuse different reporting dates.
8. When evidence from multiple chunks agrees, synthesize it into one concise answer.
9. When evidence conflicts, explicitly state that the supplied evidence conflicts
   rather than choosing a value without support.
10. Every material factual claim must have at least one citation.
11. Citations MUST use one of these formats:
    [p.X]
    [p.X, table]
    [p.X, figure]
12. Multiple citations may be grouped inside one pair of brackets using semicolons.
    Example:
    [p.4, table; p.21, table; p.21]
13. Only cite pages and modalities that actually appear in the supplied evidence.
14. For calculations, show the calculation briefly and cite the source values.
15. For figure evidence, describe only what is visibly supported by the supplied
    figure evidence. Do not infer information that is not visible.
16. Keep answers concise and directly answer the question.
"""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for answer generation."
        )

    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)

    return _client


def _format_evidence(hits: list[RetrievalHit]) -> str:
    parts: list[str] = []

    for i, hit in enumerate(hits, start=1):
        c = hit.chunk

        parts.append(
            f"EVIDENCE {i}\n"
            f"page={c.page}\n"
            f"modality={c.modality}\n"
            f"source={c.source}\n"
            f"retrieval_score={hit.score:.5f}\n"
            f"content:\n{c.text}\n"
        )

    return "\n---\n".join(parts)


def _truncate_evidence(
    hits: list[RetrievalHit],
    max_chars: int,
) -> str:
    """
    Preserve complete evidence chunks whenever possible.

    Never cut a table or figure description in the middle merely because the
    concatenated context reached max_chars.
    """

    parts: list[str] = []
    total_chars = 0

    for i, hit in enumerate(hits, start=1):
        c = hit.chunk

        part = (
            f"EVIDENCE {i}\n"
            f"page={c.page}\n"
            f"modality={c.modality}\n"
            f"source={c.source}\n"
            f"retrieval_score={hit.score:.5f}\n"
            f"content:\n{c.text}\n"
        )

        if total_chars + len(part) > max_chars:
            break

        parts.append(part)
        total_chars += len(part)

    return "\n---\n".join(parts)


# Matches:
#   [p.4]
#   [p.4, table]
#   [p.4, figure]
#
# The regex is deliberately NOT tied to the surrounding brackets because
# citations can also appear grouped:
#
#   [p.4, table; p.21, table; p.21]
#
# We extract each individual citation from the group.
_CITATION_PATTERN = re.compile(
    r"p\.(\d+)(?:,\s*(table|figure))?",
    re.IGNORECASE,
)


def _validate_citations(
    answer_text: str,
    hits: list[RetrievalHit],
) -> list[Citation]:
    """
    Extract citations from the generated answer and keep only citations
    supported by retrieved evidence.

    Supports:
        [p.4]
        [p.4, table]
        [p.4, table; p.21, table; p.21]
        [p.1; p.3; p.17]
    """

    available = {
        (hit.chunk.page, hit.chunk.modality)
        for hit in hits
    }

    citations: list[Citation] = []
    seen: set[tuple[int, str]] = set()

    for match in _CITATION_PATTERN.finditer(answer_text):
        page = int(match.group(1))
        modality_text = match.group(2)

        if modality_text:
            modality = modality_text.lower()

        else:
            # Page-only citation:
            #
            # [p.4]
            #
            # If multiple modalities exist on that page, prefer text.
            candidates = [
                modality
                for candidate_page, modality in available
                if candidate_page == page
            ]

            if not candidates:
                continue

            modality = (
                "text"
                if "text" in candidates
                else candidates[0]
            )

        if (page, modality) not in available:
            continue

        key = (page, modality)

        if key in seen:
            continue

        seen.add(key)

        citations.append(
            Citation(
                page=page,
                modality=modality,
            )
        )

    return citations


def answer(
    question: str,
    hits: list[RetrievalHit],
) -> GenerationResult:
    client = _get_client()

    if not hits:
        return GenerationResult(
            answer=(
                "The document evidence is insufficient to answer this question."
            ),
            citations=[],
        )

    evidence = _truncate_evidence(
        hits,
        settings.max_context_chars,
    )

    if not evidence:
        return GenerationResult(
            answer=(
                "The document evidence is insufficient to answer this question."
            ),
            citations=[],
        )

    response = client.responses.create(
        model=settings.openai_model,
        instructions=SYSTEM,
        input=(
            f"Question:\n{question}\n\n"
            f"Evidence:\n{evidence}"
        ),
    )

    answer_text = response.output_text.strip()

    citations = _validate_citations(
        answer_text,
        hits,
    )

    return GenerationResult(
        answer=answer_text,
        citations=citations,
    )


def caption_figure(image_path: str) -> str:
    """
    Generate a grounded description of a document figure.

    The model is instructed to describe only information visibly present
    in the supplied image.
    """

    client = _get_client()

    data = base64.b64encode(
        Path(image_path).read_bytes()
    ).decode("utf-8")

    response = client.responses.create(
        model=settings.openai_vision_model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Describe only information visibly present in this "
                            "financial-document figure, table, or chart.\n\n"
                            "Extract visible titles, labels, legends, axis values, "
                            "data labels, dates, units, and visible trends.\n\n"
                            "Do not infer values that are not visible.\n"
                            "Do not guess unreadable information.\n"
                            "If something is ambiguous or unreadable, say so explicitly."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{data}",
                    },
                ],
            }
        ],
    )

    return response.output_text.strip()