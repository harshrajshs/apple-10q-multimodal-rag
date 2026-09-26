from dataclasses import dataclass, asdict
from typing import Literal


Modality = Literal["text", "table", "figure"]


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    page: int
    modality: Modality
    text: str
    source: str
    bbox: list[float] | None = None
    image_path: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    source_type: str

    def to_dict(self):
        return {
            "chunk": self.chunk.to_dict(),
            "score": round(float(self.score), 5),
            "retrieval": self.source_type,
        }


@dataclass
class Citation:
    page: int
    modality: Modality

    def to_dict(self):
        return {
            "page": self.page,
            "modality": self.modality,
        }


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]

    def to_dict(self):
        return {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
        }