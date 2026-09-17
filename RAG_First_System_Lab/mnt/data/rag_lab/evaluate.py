from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.evaluation import evaluate_case, load_eval_set, summarize
from rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="documents")
    parser.add_argument("--eval", default="data/evaluation.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    args = parser.parse_args()

    rag = RAGPipeline(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print("Building knowledge base...")
    print(rag.build_from_directory(args.docs))

    results = []
    for case in load_eval_set(args.eval):
        result = rag.answer(case["question"], top_k=args.top_k)
        results.append(evaluate_case(case, result))
        print(f"\n[{case['type']}] {case['question']}")
        print(result["answer"])
        print("Retrieved:", [x["source"] for x in result["retrieved"]])

    print("\nSummary")
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
