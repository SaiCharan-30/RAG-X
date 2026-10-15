import json
import sys

import run_eval as ev


def short(text, length=170):
    return ev.normalize(text)[:length]


def show(label, results, item):
    print(f"  {label}")
    for position, chunk in enumerate(results[:5], start=1):
        mark = "MATCH" if ev.is_relevant(chunk, item) else "     "
        score = chunk.get("rerank_score")
        score_text = f"{score:.3f}" if score is not None else "  -  "
        print(f"   #{position} {mark} score {score_text}  {short(chunk['text'])}")
    deeper = ev.first_relevant_rank(results, item)
    print(f"   first matching passage in the top 20: {'#' + str(deeper) if deeper else 'none'}")


def main():
    # Windows consoles/files can't print characters like the "fi" ligature by default.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    wanted = set(sys.argv[1:])
    eval_set = json.loads(ev.DEFAULT_SET.read_text(encoding="utf-8"))
    backend = ev.Backend("http://127.0.0.1:8001")
    for item in eval_set["in_scope"]:
        reranked = backend.search(item["question"], 20, True)
        baseline = backend.search(item["question"], 20, False)
        rank_reranked = ev.first_relevant_rank(reranked[:5], item)
        if wanted and item["id"] not in wanted:
            continue
        if not wanted and rank_reranked is not None:
            continue
        print(f"\n=== {item['id']}: {item['question']}")
        print(f"  keywords: {item['keywords']}")
        show("WITH reranking:", reranked, item)
        show("WITHOUT reranking (hybrid search only):", baseline, item)


if __name__ == "__main__":
    main()