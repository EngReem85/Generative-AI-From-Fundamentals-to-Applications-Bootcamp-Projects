from __future__ import annotations

import json
import re
from pathlib import Path


def load_eval_set(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def citation_labels(answer: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\[S(\d+)\]", answer)})


def evaluate_case(case: dict, result: dict) -> dict:
    expected = set(case.get("expected_sources", []))
    retrieved = result.get("retrieved", [])
    retrieved_sources = {item["source"] for item in retrieved}

    retrieval_hit = bool(expected & retrieved_sources) if expected else not (
        "I don't have enough information" not in result["answer"]
    )

    labels = citation_labels(result["answer"])
    cited_sources = set()
    for label in labels:
        if 1 <= label <= len(retrieved):
            cited_sources.add(retrieved[label - 1]["source"])

    citation_precision = (
        len(cited_sources & expected) / len(cited_sources)
        if cited_sources else 0.0
    )
    citation_recall = (
        len(cited_sources & expected) / len(expected)
        if expected else (1.0 if not cited_sources else 0.0)
    )

    abstains = "I don't have enough information" in result["answer"]
    abstention_correct = abstains == bool(case.get("expect_abstain"))

    return {
        "id": case["id"],
        "type": case["type"],
        "retrieval_hit": retrieval_hit,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "abstention_correct": abstention_correct,
        "answer": result["answer"],
        "retrieved_sources": list(retrieved_sources),
    }


def summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    return {
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in results) / len(results),
        "citation_precision": sum(r["citation_precision"] for r in results) / len(results),
        "citation_recall": sum(r["citation_recall"] for r in results) / len(results),
        "abstention_accuracy": sum(r["abstention_correct"] for r in results) / len(results),
    }
