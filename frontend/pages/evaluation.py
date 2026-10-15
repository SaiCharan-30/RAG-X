import json
from pathlib import Path

import pandas as pd
import streamlit as st


# frontend/pages/evaluation.py -> project folder -> evaluation/results
RESULTS_FILE = (
    Path(__file__).resolve().parents[2] / "evaluation" / "results" / "latest.json"
)


def percent(value) -> str:
    return "—" if value is None else f"{value:.0%}"


st.title("📈 Evaluation")

st.write(
    "How well does RAG-X find the right passages? These numbers come from a "
    "list of questions with known answers (evaluation/eval_set.json)."
)

st.divider()


if not RESULTS_FILE.exists():
    st.info(
        "No evaluation results yet. With the backend running, open a terminal "
        "in the project folder and run:"
    )
    st.code("uv run python evaluation/run_eval.py", language="powershell")
    st.stop()


report = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))

st.caption(
    f"Last run: {report['created_at'][:19].replace('T', ' ')} UTC · "
    f"{report['retrieval_with_reranking'].get('questions', 0)} questions"
)

if report["skipped_questions"]:
    st.warning(
        f"{len(report['skipped_questions'])} question(s) were skipped because "
        "their document is not indexed: "
        + ", ".join(report["skipped_questions"])
    )


# ------------------------------------------------------------- retrieval

st.subheader("Retrieval quality")

with_rerank = report["retrieval_with_reranking"]
without_rerank = report["retrieval_without_reranking"]

if not with_rerank.get("questions"):
    st.info("No questions could be run.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Hit@1", percent(with_rerank["hit@1"]))
col2.metric("Hit@3", percent(with_rerank["hit@3"]))
col3.metric("Hit@5", percent(with_rerank["hit@5"]))
col4.metric("MRR", f"{with_rerank['mrr']:.2f}")

st.caption(
    "Hit@k: how often the right passage is among the top k results. "
    "MRR: 1.0 means the right passage is always first."
)

comparison = pd.DataFrame(
    {
        "With reranking": [with_rerank[f"hit@{k}"] for k in (1, 3, 5)],
        "Without reranking": [without_rerank[f"hit@{k}"] for k in (1, 3, 5)],
    },
    index=["Hit@1", "Hit@3", "Hit@5"],
)

st.bar_chart(comparison, stack=False)

# NEW: explain why the two bars are equal.
st.caption(
    "The two bars are equal on purpose: results keep the order hybrid search "
    "found them (it scored better than the reranker's order on these "
    "documents). The reranker adds a relevance score, which decides whether "
    "RAG-X answers or says it found nothing."
)


# ------------------------------------------------------------- threshold

st.subheader("“No answer” rule")

analysis = report["threshold"]

if analysis is None:
    st.info("Not enough questions to analyse the threshold.")
else:
    col1, col2 = st.columns(2)

    col1.metric(
        "Lowest score of an answerable question",
        f"{analysis['lowest_in_scope_score']:.4f}",
    )
    col2.metric(
        "Highest score of an unanswerable question",
        f"{analysis['highest_out_of_scope_score']:.4f}",
    )

    if analysis["cleanly_separable"]:
        st.success(
            "The two groups are cleanly separated, so a threshold between "
            "them answers every answerable question and refuses every "
            "unanswerable one."
        )
    else:
        st.warning(
            "The two groups overlap: some unanswerable questions score "
            "higher than some answerable ones."
        )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "": label,
                    "Threshold": round(row["threshold"], 4),
                    "Answers answerable questions": percent(row["answers_in_scope"]),
                    "Refuses unanswerable questions": percent(row["refuses_out_of_scope"]),
                }
                for label, row in (
                    ("Current", analysis["current"]),
                    ("Suggested", analysis["suggested"]),
                )
            ]
        ),
        hide_index=True,
        width="stretch",
    )


# --------------------------------------------------------------- answers

if report.get("answers"):
    st.subheader("Full answers")

    answers = report["answers"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg. grounding score", percent(answers["average_grounding_score"]))
    col2.metric("Cites the right file", percent(answers["cites_expected_file"]))
    col3.metric("Answer has key phrase", percent(answers["answer_has_keyword"]))
    col4.metric("Refuses unanswerable", percent(answers["refuses_out_of_scope"]))

    # NEW: these numbers depend on which language model produced the answers.
    st.caption(
        "These numbers depend on the language model used for the run. With the "
        "mock LLM (MOCK_LLM=true) every answer is the same placeholder, so "
        "“Answer has key phrase” is 0% and the grounding score means little. "
        "“Cites the right file” and “Refuses unanswerable” still reflect "
        "retrieval. Re-run with real Gemini: "
        "uv run python evaluation/run_eval.py --answers"
    )


# ------------------------------------------------------------- per question

st.subheader("Question by question")

st.dataframe(
    pd.DataFrame(
        [
            {
                "Question": row["question"],
                "Expected file": row["expected_file"],
                "Rank (reranked)": str(row["rank_with_reranking"] or "not found"),
                "Rank (not reranked)": str(row["rank_without_reranking"] or "not found"),
                "Top score": round(row["top_score"], 3),
            }
            for row in report["in_scope"]
        ]
    ),
    hide_index=True,
    width="stretch",
)

with st.expander("Unanswerable questions"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Question": row["question"],
                    "Top score": round(row["top_score"], 4),
                    "Top file": row["top_file"],
                }
                for row in report["out_of_scope"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )