"""
RAG-X evaluation runner.

Measures how well retrieval works on YOUR documents, using a small list of
questions with known answers (evaluation/eval_set.json). It talks to the
running backend over HTTP, so it tests the real pipeline.

Run from the project folder, with the backend running:

    uv run python evaluation/run_eval.py              # retrieval metrics
    uv run python evaluation/run_eval.py --validate   # check the question set
    uv run python evaluation/run_eval.py --answers    # also test full answers
                                                      # (use with real Gemini)

Results are saved to evaluation/results/latest.json and shown on the
Evaluation page of the app.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_SET = HERE / "eval_set.json"
RESULTS_DIR = HERE / "results"

# Keep equal to MIN_CONTEXT_SCORE in backend/app/services/context_evaluator.py
CURRENT_THRESHOLD = 0.05

KS = (1, 3, 5)
REFUSAL_TEXT = "i could not find relevant information"


# ------------------------------------------------------------ pure maths

def normalize(text):
    """Fold ligatures (the single fi character becomes f+i) first, then the original rules."""
    import unicodedata
    return _normalize_basic(unicodedata.normalize("NFKC", text))


def _normalize_basic(text: str) -> str:
    """Lower-case and collapse all whitespace, so line breaks do not matter."""
    return " ".join((text or "").lower().split())


def is_relevant(chunk: dict, item: dict) -> bool:
    file_part = (item.get("file") or "").lower()

    if file_part and file_part not in (chunk.get("filename") or "").lower():
        return False

    text = normalize(chunk.get("text"))

    return any(normalize(keyword) in text for keyword in item["keywords"])


def first_relevant_rank(results: list[dict], item: dict) -> int | None:
    for position, chunk in enumerate(results, start=1):
        if is_relevant(chunk, item):
            return position

    return None


def retrieval_metrics(ranks: list[int | None]) -> dict:
    total = len(ranks)

    if total == 0:
        return {"questions": 0}

    metrics = {"questions": total}

    for k in KS:
        metrics[f"hit@{k}"] = sum(
            1 for rank in ranks if rank is not None and rank <= k
        ) / total

    metrics["mrr"] = sum(
        1 / rank for rank in ranks if rank is not None
    ) / total

    return metrics


def threshold_analysis(
    in_scores: list[float],
    out_scores: list[float],
    current: float = CURRENT_THRESHOLD,
) -> dict | None:
    """
    in_scores: best rerank score for questions the documents CAN answer.
    out_scores: best rerank score for questions they can NOT answer.
    A threshold T means: answer if score >= T, otherwise say "not found".
    """
    if not in_scores or not out_scores:
        return None

    def evaluate(threshold: float) -> dict:
        answered = sum(1 for s in in_scores if s >= threshold) / len(in_scores)
        refused = sum(1 for s in out_scores if s < threshold) / len(out_scores)

        return {
            "threshold": threshold,
            "answers_in_scope": answered,
            "refuses_out_of_scope": refused,
            "balanced": (answered + refused) / 2,
        }

    max_out = max(out_scores)
    min_in = min(in_scores)

    if max_out < min_in:
        # A clear gap exists. These scores behave like probabilities, so
        # pick the middle of the gap on a multiplicative scale (geometric
        # mean). The small floor stops a score of 0 from dragging it to 0.
        suggested = (max(max_out, 1e-4) * min_in) ** 0.5
    else:
        # The two groups overlap: take the threshold that gets the most
        # questions right overall.
        values = sorted(set(in_scores) | set(out_scores))
        candidates = [0.0] + [
            (low + high) / 2 for low, high in zip(values, values[1:])
        ] + [values[-1] + 1e-9]

        scored = [evaluate(t) for t in candidates]
        best_value = max(row["balanced"] for row in scored)
        best = [
            row["threshold"] for row in scored if row["balanced"] == best_value
        ]
        suggested = sorted(best)[len(best) // 2]

    return {
        "current": evaluate(current),
        "suggested": evaluate(suggested),
        "highest_out_of_scope_score": max_out,
        "lowest_in_scope_score": min_in,
        "cleanly_separable": max_out < min_in,
    }


# ------------------------------------------------------------- HTTP part

class Backend:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def call(self, method: str, path: str, timeout: int = 120, **kwargs):
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                timeout=timeout,
                **kwargs,
            )
        except requests.exceptions.ConnectionError:
            sys.exit(
                f"Cannot reach the backend at {self.base_url}. "
                "Start it first (uvicorn ... --port 8001)."
            )

        if not response.ok:
            raise RuntimeError(
                f"{method} {path} failed ({response.status_code}): "
                f"{response.text[:200]}"
            )

        return response.json()

    def search(self, query, top_k, rerank, document_ids=None):
        mode = "single" if document_ids and len(document_ids) == 1 else "all"

        return self.call(
            "POST",
            "/search",
            json={
                "query": query,
                "mode": mode,
                "document_ids": document_ids if mode == "single" else None,
                "top_k": top_k,
                "rerank": rerank,
            },
        )["results"]

    def chat(self, query):
        conversation = self.call(
            "POST", "/chat/conversations", json={"title": "evaluation run"}
        )
        conversation_id = conversation["conversation_id"]

        try:
            return self.call(
                "POST",
                f"/chat/conversations/{conversation_id}/chat",
                timeout=180,
                json={"query": query, "mode": "all"},
            )
        finally:
            # Evaluation conversations must not clutter the chat history.
            try:
                self.call("DELETE", f"/chat/conversations/{conversation_id}")
            except Exception:
                pass


def find_document(documents: list[dict], file_part: str) -> dict | None:
    for document in documents:
        if (
            document["status"] == "indexed"
            and file_part.lower() in document["original_filename"].lower()
        ):
            return document

    return None


# ----------------------------------------------------------------- modes

def validate(backend: Backend, eval_set: dict) -> int:
    """Checks that each expected keyword really exists in its document."""
    documents = backend.call("GET", "/documents")
    problems = 0

    print("Checking the question set against your indexed documents ...\n")

    for item in eval_set["in_scope"]:
        document = find_document(documents, item["file"])

        if document is None:
            print(f"MISSING  {item['id']}: no indexed document matches "
                  f"'{item['file']}'")
            problems += 1
            continue

        chunks = backend.search(
            item["question"], 20, False, [document["document_id"]]
        )
        found = [
            keyword
            for keyword in item["keywords"]
            if any(normalize(keyword) in normalize(c["text"]) for c in chunks)
        ]

        if found:
            print(f"OK       {item['id']}")
        else:
            print(f"WARNING  {item['id']}: none of {item['keywords']} was "
                  f"found in the best passages of {document['original_filename']}"
                  " - edit the keywords or the question")
            problems += 1

    print(f"\n{problems} problem(s).")
    return problems


def run(backend: Backend, eval_set: dict, with_answers: bool) -> dict:
    documents = backend.call("GET", "/documents")

    rows = []
    skipped = []
    ranks_reranked = []
    ranks_baseline = []
    in_scores = []

    print(f"Running {len(eval_set['in_scope'])} in-scope questions ...")

    for item in eval_set["in_scope"]:
        if find_document(documents, item["file"]) is None:
            skipped.append(item["id"])
            print(f"  skipped {item['id']} (document not indexed)")
            continue

        reranked = backend.search(item["question"], 10, True)
        baseline = backend.search(item["question"], 10, False)

        rank_reranked = first_relevant_rank(reranked, item)
        rank_baseline = first_relevant_rank(baseline, item)
        top_score = max((c["rerank_score"] for c in reranked[:5]), default=0.0)

        ranks_reranked.append(rank_reranked)
        ranks_baseline.append(rank_baseline)
        in_scores.append(top_score)

        row = {
            "id": item["id"],
            "question": item["question"],
            "expected_file": item["file"],
            "rank_with_reranking": rank_reranked,
            "rank_without_reranking": rank_baseline,
            "top_score": top_score,
            "top_file": reranked[0]["filename"] if reranked else None,
        }

        if with_answers:
            answer = backend.chat(item["question"])
            cited_files = [c["filename"] or "" for c in answer["citations"]]
            row["answer"] = answer["answer"]
            row["grounding_score"] = answer["grounding_score"]
            row["cites_expected_file"] = any(
                item["file"].lower() in name.lower() for name in cited_files
            )
            row["answer_has_keyword"] = any(
                normalize(k) in normalize(answer["answer"])
                for k in item["keywords"]
            )

        rows.append(row)

    out_rows = []
    out_scores = []

    print(f"Running {len(eval_set['out_of_scope'])} out-of-scope questions ...")

    for item in eval_set["out_of_scope"]:
        results = backend.search(item["question"], 5, True)
        score = max((c["rerank_score"] for c in results[:5]), default=0.0)
        out_scores.append(score)

        row = {
            "id": item["id"],
            "question": item["question"],
            "top_score": score,
            "top_file": results[0]["filename"] if results else None,
        }

        if with_answers:
            answer = backend.chat(item["question"])
            row["answer"] = answer["answer"]
            row["refused"] = normalize(answer["answer"]).startswith(REFUSAL_TEXT)

        out_rows.append(row)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend_url": backend.base_url,
        "with_answers": with_answers,
        "skipped_questions": skipped,
        "retrieval_with_reranking": retrieval_metrics(ranks_reranked),
        "retrieval_without_reranking": retrieval_metrics(ranks_baseline),
        "threshold": threshold_analysis(in_scores, out_scores),
        "in_scope": rows,
        "out_of_scope": out_rows,
    }

    if with_answers:
        answered = [r for r in rows if "grounding_score" in r]

        report["answers"] = {
            "average_grounding_score": (
                sum(r["grounding_score"] for r in answered) / len(answered)
                if answered else None
            ),
            "cites_expected_file": (
                sum(1 for r in answered if r["cites_expected_file"])
                / len(answered) if answered else None
            ),
            "answer_has_keyword": (
                sum(1 for r in answered if r["answer_has_keyword"])
                / len(answered) if answered else None
            ),
            "refuses_out_of_scope": (
                sum(1 for r in out_rows if r["refused"]) / len(out_rows)
                if out_rows else None
            ),
        }

    return report


# ----------------------------------------------------------------- output

def percent(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def print_report(report: dict) -> None:
    print("\n=== Retrieval (did the right passage come back?) ===")
    print(f"{'':28}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}")

    for label, key in (
        ("With reranking", "retrieval_with_reranking"),
        ("Without reranking", "retrieval_without_reranking"),
    ):
        m = report[key]

        if m.get("questions"):
            print(
                f"{label:28}{percent(m['hit@1']):>8}{percent(m['hit@3']):>8}"
                f"{percent(m['hit@5']):>8}{m['mrr']:>8.2f}"
            )

    print(f"({report['retrieval_with_reranking'].get('questions', 0)} questions)")

    if report["skipped_questions"]:
        print(f"Skipped (document not indexed): {report['skipped_questions']}")

    print("\n=== Questions where the right passage was NOT in the top 5 ===")
    misses = [
        r for r in report["in_scope"]
        if r["rank_with_reranking"] is None or r["rank_with_reranking"] > 5
    ]

    if not misses:
        print("None.")

    for r in misses:
        print(f"- {r['id']}: {r['question']}  (top hit: {r['top_file']}, "
              f"score {r['top_score']:.3f})")

    analysis = report["threshold"]
    print("\n=== 'No answer' threshold ===")

    if analysis is None:
        print("Not enough questions to analyse.")
    else:
        print(f"Lowest score of an answerable question : "
              f"{analysis['lowest_in_scope_score']:.4f}")
        print(f"Highest score of an unanswerable one   : "
              f"{analysis['highest_out_of_scope_score']:.4f}")
        print("Cleanly separable:", "yes" if analysis["cleanly_separable"] else "NO")

        for label, key in (("Current", "current"), ("Suggested", "suggested")):
            row = analysis[key]
            print(f"{label:10} threshold {row['threshold']:.4f}: answers "
                  f"{percent(row['answers_in_scope'])} of answerable, refuses "
                  f"{percent(row['refuses_out_of_scope'])} of unanswerable")

    if report.get("answers"):
        a = report["answers"]
        print("\n=== Full answers (needs real Gemini) ===")
        print(f"Average grounding score        : {a['average_grounding_score']}")
        print(f"Cites the expected file        : {percent(a['cites_expected_file'])}")
        print(f"Answer contains a key phrase   : {percent(a['answer_has_keyword'])}")
        print(f"Refuses unanswerable questions : {percent(a['refuses_out_of_scope'])}")


def save_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    text = json.dumps(report, indent=2)

    (RESULTS_DIR / f"run-{stamp}.json").write_text(text, encoding="utf-8")
    latest = RESULTS_DIR / "latest.json"
    latest.write_text(text, encoding="utf-8")

    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-X evaluation")
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--set", default=str(DEFAULT_SET), dest="eval_set")
    parser.add_argument("--validate", action="store_true",
                        help="only check that the questions match your documents")
    parser.add_argument("--answers", action="store_true",
                        help="also run full chat answers (use with real Gemini)")
    args = parser.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    backend = Backend(args.url)

    if args.validate:
        sys.exit(1 if validate(backend, eval_set) else 0)

    report = run(backend, eval_set, args.answers)
    print_report(report)

    print(f"\nSaved to {save_report(report)}")


if __name__ == "__main__":
    main()