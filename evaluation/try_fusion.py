"""Measurement only: would blending the hybrid-search rank with the reranker rank help?
Changes nothing in the backend. Uses the same questions as run_eval.py."""
import json
import sys

import run_eval as ev


def fused_order(chunks, hybrid_weight, k=60):
    """chunks arrive sorted by reranker score. Each chunk also carries its hybrid-search rank."""
    scored = []
    for rerank_rank, chunk in enumerate(chunks, start=1):
        hybrid_rank = chunk.get("retrieval_rank") or rerank_rank
        score = hybrid_weight / (k + hybrid_rank) + 1.0 / (k + rerank_rank)
        scored.append((score, chunk))
    scored.sort(key=lambda pair: -pair[0])
    return [chunk for _, chunk in scored]


def summarise(name, ranks):
    count = len(ranks)
    def hit(n):
        return sum(1 for r in ranks if r is not None and r <= n) / count
    mrr = sum(1.0 / r for r in ranks if r is not None) / count
    print(f"{name:<34}{hit(1):>7.0%}{hit(3):>7.0%}{hit(5):>7.0%}{mrr:>8.2f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    eval_set = json.loads(ev.DEFAULT_SET.read_text(encoding="utf-8"))
    backend = ev.Backend("http://127.0.0.1:8001")

    methods = {
        "Reranker only (current)": None,
        "Hybrid only (no reranker)": "hybrid",
        "Blend, equal weight": 1.0,
        "Blend, hybrid counts double": 2.0,
    }
    ranks = {name: [] for name in methods}

    for item in eval_set["in_scope"]:
        reranked = backend.search(item["question"], 20, True)
        for name, how in methods.items():
            if how is None:
                ordered = reranked
            elif how == "hybrid":
                ordered = sorted(reranked, key=lambda c: c.get("retrieval_rank") or 999)
            else:
                ordered = fused_order(reranked, how)
            ranks[name].append(ev.first_relevant_rank(ordered[:5], item))

    print(f"\n{'Method':<34}{'Hit@1':>7}{'Hit@3':>7}{'Hit@5':>7}{'MRR':>8}")
    for name in methods:
        summarise(name, ranks[name])
    print(f"({len(eval_set['in_scope'])} questions)")


if __name__ == "__main__":
    main()