from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from app.core.pipeline import RAGPipeline


CHUNKS_PATH = "artifacts/index/chunks.json"
OUTPUT_PATH = "artifacts/evaluation_results.json"


# ---------------------------------------------------------------------------
# Gold evaluation set
#
# acceptable_evidence contains (page, modality) pairs that are valid evidence
# for the question. Matching any one acceptable source is sufficient.
#
# expected_answer_contains is used for lightweight answer correctness checks.
# This intentionally avoids using another LLM as a judge.
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "id": "q01",
        "category": "text",
        "question": "What is the purpose of Apple's Form 10-Q?",
        "acceptable_evidence": [
            (1, "text"),
            (3, "text"),
            (17, "text"),
            (25, "text"),
            (28, "text"),
        ],
        "expected_answer_contains": [
            "form 10-q",
        ],
        "should_abstain": False,
    },
    {
        "id": "q02",
        "category": "table",
        "question": (
            "What was Apple's total net sales for the nine months "
            "ended June 25, 2022?"
        ),
        "acceptable_evidence": [
            (4, "table"),
            (4, "text"),
            (10, "text"),
            (18, "table"),
        ],
        "expected_answer_contains": [
            "304,182",
        ],
        "should_abstain": False,
    },
    {
        "id": "q03",
        "category": "table",
        "question": (
            "What were Apple's total operating expenses for the nine months "
            "ended June 25, 2022?"
        ),
        "acceptable_evidence": [
            (4, "table"),
            (4, "text"),
            (10, "text"),
            (18, "table"),
        ],
        "expected_answer_contains": [
            "38,144",
        ],
        "should_abstain": False,
    },
    {
        "id": "q04",
        "category": "table",
        "question": (
            "What was Apple's net income for the nine months "
            "ended June 25, 2022?"
        ),
        "acceptable_evidence": [
            (4, "table"),
            (4, "text"),
            (10, "text"),
            (18, "table"),
        ],
        "expected_answer_contains": [
            "79,082",
        ],
        "should_abstain": False,
    },
    {
        "id": "q05",
        "category": "table",
        "question": (
            "What was Apple's gross margin for the three months "
            "ended June 25, 2022?"
        ),
        "acceptable_evidence": [
            (4, "table"),
            (4, "text"),
            (10, "text"),
            (18, "table"),
        ],
        "expected_answer_contains": [
            "35,885",
        ],
        "should_abstain": False,
    },
    {
        "id": "q06",
        "category": "table",
        "question": (
            "What were Apple's research and development expenses "
            "for the nine months ended June 25, 2022?"
        ),
        "acceptable_evidence": [
            (4, "table"),
            (4, "text"),
            (10, "text"),
            (18, "table"),
        ],
        "expected_answer_contains": [
            "19,490",
        ],
        "should_abstain": False,
    },
    {
        "id": "q07",
        "category": "table",
        "question": (
            "What were Apple's total net sales by geographic region "
            "for the nine months ended June 25, 2022?"
        ),
        "acceptable_evidence": [
            (10, "table"),
            (10, "text"),
            (16, "text"),
            (18, "table"),
        ],
        "expected_answer_contains": [],
        "should_abstain": False,
    },
    {
        "id": "q08",
        "category": "calculation",
        "question": (
            "By what percentage did Apple's total net sales increase "
            "from the nine months ended June 26, 2021 to the "
            "nine months ended June 25, 2022?"
        ),
        "acceptable_evidence": [
            (4, "table"),
            (4, "text"),
            (10, "text"),
            (18, "table"),
        ],
        "expected_answer_contains_any": [
		["7.7", "7.69", "8%"],
	],
        "should_abstain": False,
    },
    {
        "id": "q09",
        "category": "figure",
        "question": "What information is shown in the figure in the document?",
        "acceptable_evidence": [
            (1, "figure"),
        ],
        "expected_answer_contains": [],
        "should_abstain": False,
    },
    {
        "id": "q10",
        "category": "abstention",
        "question": (
            "What was Apple's revenue from a product launched in 2035?"
        ),
        "acceptable_evidence": [],
        "expected_answer_contains": [
            "insufficient",
        ],
        "should_abstain": True,
    },
]


def evidence_key(page: int, modality: str) -> tuple[int, str]:
    return (int(page), modality)


def retrieved_evidence(result: dict) -> set[tuple[int, str]]:
    evidence = set()

    for item in result.get("evidence", []):
        chunk = item["chunk"]

        evidence.add(
            evidence_key(
                chunk["page"],
                chunk["modality"],
            )
        )

    return evidence


def citation_set(result: dict) -> set[tuple[int, str]]:
    return {
        evidence_key(
            item["page"],
            item["modality"],
        )
        for item in result.get("citations", [])
    }


def recall_at_k(
    result: dict,
    acceptable: list[tuple[int, str]],
    k: int,
) -> float:
    """
    Returns 1.0 when at least one acceptable evidence source is present
    in the top-k retrieved results, otherwise 0.0.

    This treats equivalent valid evidence sources as alternatives rather
    than requiring the retriever to return one exact page/modality pair.
    """
    if not acceptable:
        return 1.0

    acceptable_set = set(acceptable)

    for item in result.get("evidence", [])[:k]:
        chunk = item["chunk"]
        evidence = evidence_key(
            chunk["page"],
            chunk["modality"],
        )

        if evidence in acceptable_set:
            return 1.0

    return 0.0


def reciprocal_rank(
    result: dict,
    acceptable: list[tuple[int, str]],
) -> float:
    """
    Returns the reciprocal rank of the first acceptable evidence source.
    """
    if not acceptable:
        return 1.0

    acceptable_set = set(acceptable)

    for rank, item in enumerate(
        result.get("evidence", []),
        start=1,
    ):
        chunk = item["chunk"]

        evidence = evidence_key(
            chunk["page"],
            chunk["modality"],
        )

        if evidence in acceptable_set:
            return 1.0 / rank

    return 0.0


def answer_contains_expected(
    answer: str,
    expected: list[str],
    expected_any: list[list[str]] | None = None,
) -> bool:
    normalized = answer.lower()

    # Every required value must appear in the answer.
    if expected and not all(
        value.lower() in normalized
        for value in expected
    ):
        return False

    # For each alternative group, at least one value must appear.
    if expected_any and not all(
        any(value.lower() in normalized for value in alternatives)
        for alternatives in expected_any
    ):
        return False

    return True

def citation_accuracy(
    result: dict,
) -> float:
    """
    Percentage of generated citations that correspond to retrieved evidence.
    """
    citations = citation_set(result)

    if not citations:
        return 0.0

    retrieved = retrieved_evidence(result)

    valid = sum(
        1
        for citation in citations
        if citation in retrieved
    )

    return valid / len(citations)


def abstention_correct(
    result: dict,
    should_abstain: bool,
) -> bool:
    answer = result.get("answer", "").lower()

    abstention_markers = [
        "insufficient",
        "not enough evidence",
        "cannot answer",
        "can't answer",
        "not supported by the document",
    ]

    actually_abstained = any(
        marker in answer
        for marker in abstention_markers
    )

    return actually_abstained == should_abstain


def evaluate_question(
    pipeline: RAGPipeline,
    item: dict,
) -> dict:
    start = time.perf_counter()

    result = pipeline.query(
        item["question"]
    )

    measured_latency_ms = round(
        (time.perf_counter() - start) * 1000,
        2,
    )

    acceptable = item["acceptable_evidence"]

    r1 = recall_at_k(
        result,
        acceptable,
        1,
    )

    r5 = recall_at_k(
        result,
        acceptable,
        5,
    )

    r8 = recall_at_k(
        result,
        acceptable,
        8,
    )

    mrr = reciprocal_rank(
        result,
        acceptable,
    )

    answer_correct = answer_contains_expected(
        result.get("answer", ""),
        item.get("expected_answer_contains", []),
        item.get("expected_answer_contains_any"),
    )

    citation_validity = citation_accuracy(
        result
    )

    abstention = abstention_correct(
        result,
        item["should_abstain"],
    )

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "retrieved_evidence": [
            {
                "page": x["chunk"]["page"],
                "modality": x["chunk"]["modality"],
                "score": x["score"],
                "source": x["chunk"]["source"],
            }
            for x in result.get("evidence", [])
        ],
        "metrics": {
            "recall_at_1": r1,
            "recall_at_5": r5,
            "recall_at_8": r8,
            "mrr": mrr,
            "answer_correct": answer_correct,
            "citation_accuracy": citation_validity,
            "abstention_correct": abstention,
        },
        "latency_ms": measured_latency_ms,
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return round(
        statistics.mean(values),
        4,
    )


def summarize(results: list[dict]) -> dict:
    metrics = [
        result["metrics"]
        for result in results
    ]

    latencies = [
        result["latency_ms"]
        for result in results
    ]

    answer_scores = [
        float(m["answer_correct"])
        for m in metrics
    ]

    citation_scores = [
        m["citation_accuracy"]
        for m in metrics
    ]

    abstention_scores = [
        float(m["abstention_correct"])
        for m in metrics
    ]

    return {
        "num_questions": len(results),
        "recall_at_1": mean(
            [m["recall_at_1"] for m in metrics]
        ),
        "recall_at_5": mean(
            [m["recall_at_5"] for m in metrics]
        ),
        "recall_at_8": mean(
            [m["recall_at_8"] for m in metrics]
        ),
        "mrr": mean(
            [m["mrr"] for m in metrics]
        ),
        "answer_accuracy": mean(
            answer_scores
        ),
        "citation_accuracy": mean(
            citation_scores
        ),
        "abstention_accuracy": mean(
            abstention_scores
        ),
        "mean_latency_ms": round(
            statistics.mean(latencies),
            2,
        ),
        "median_latency_ms": round(
            statistics.median(latencies),
            2,
        ),
        "max_latency_ms": round(
            max(latencies),
            2,
        ),
    }


def main() -> None:
    print("=" * 72)
    print("Apple 10-Q Multimodal RAG Evaluation")
    print("=" * 72)

    print("\nLoading RAG pipeline...")

    pipeline = RAGPipeline(
        CHUNKS_PATH
    )

    results: list[dict] = []

    for index, item in enumerate(
        QUESTIONS,
        start=1,
    ):
        print(
            f"\n[{index}/{len(QUESTIONS)}] "
            f"{item['question']}"
        )

        result = evaluate_question(
            pipeline,
            item,
        )

        results.append(result)

        metrics = result["metrics"]

        print(
            f"  Recall@1 : {metrics['recall_at_1']:.2f}"
        )
        print(
            f"  Recall@5 : {metrics['recall_at_5']:.2f}"
        )
        print(
            f"  Recall@8 : {metrics['recall_at_8']:.2f}"
        )
        print(
            f"  MRR      : {metrics['mrr']:.2f}"
        )
        print(
            f"  Answer   : "
            f"{'PASS' if metrics['answer_correct'] else 'FAIL'}"
        )
        print(
            f"  Citation : "
            f"{metrics['citation_accuracy']:.2f}"
        )
        print(
            f"  Abstain  : "
            f"{'PASS' if metrics['abstention_correct'] else 'FAIL'}"
        )
        print(
            f"  Latency  : "
            f"{result['latency_ms']:.2f} ms"
        )

    summary = summarize(results)

    output = {
        "summary": summary,
        "results": results,
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Questions          : {summary['num_questions']}"
    )
    print(
        f"Recall@1           : {summary['recall_at_1']:.2%}"
    )
    print(
        f"Recall@5           : {summary['recall_at_5']:.2%}"
    )
    print(
        f"Recall@8           : {summary['recall_at_8']:.2%}"
    )
    print(
        f"MRR                : {summary['mrr']:.4f}"
    )
    print(
        f"Answer accuracy    : {summary['answer_accuracy']:.2%}"
    )
    print(
        f"Citation accuracy  : {summary['citation_accuracy']:.2%}"
    )
    print(
        f"Abstention accuracy: {summary['abstention_accuracy']:.2%}"
    )
    print(
        f"Mean latency       : {summary['mean_latency_ms']:.2f} ms"
    )
    print(
        f"Median latency     : {summary['median_latency_ms']:.2f} ms"
    )
    print(
        f"Max latency        : {summary['max_latency_ms']:.2f} ms"
    )

    print(
        f"\nDetailed results saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()