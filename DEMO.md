# RAG-X demo script (about 8 minutes)

## Before you start

1. `docker compose --profile app up -d`, wait for `Application startup complete`
   (`docker compose --profile app logs api --tail 3`).
2. Make sure no old local `uvicorn`/`streamlit` is running (see README, Troubleshooting).
3. For real answers set `MOCK_LLM=false` with a working `GEMINI_API_KEY` in `.env`,
   then `docker compose --profile app up -d --force-recreate api`.
   With the mock LLM, answers are a fixed placeholder; only search and citations are real.
4. Delete leftover test conversations (trash icon in the Chat sidebar).
5. Open http://127.0.0.1:8501.

## 1. Home and Dashboard (30 s)
Show the indexed documents, question and conversation counts.
*Say:* PDFs, Word, text and HTML files are parsed, chunked and indexed in Qdrant;
metadata and chat history live in PostgreSQL.

## 2. Documents (1 min)
Upload a small `.txt` file; show it moving to **Indexed**. Delete it again.
*Say:* unsupported or corrupt files are rejected or marked failed with a clear message.

## 3. Search (1.5 min)
Query: `What optimization algorithm did the authors use to train the neural network, and why?`
Show the relevance percentages, file and page. Untick **Rerank results** to show the
difference. Switch to **Selected documents** and show the results narrow.
*Say:* dense + sparse hybrid retrieval; the reranker scores each passage for relevance.

## 4. Chat with citations (2 min)
Select the neural-network paper and ask: `How is the trial solution of the differential equation constructed?`
Open **Sources** and show file, page and passage.
Then ask the follow-up: `What are its main advantages?`
*Say:* the follow-up is rewritten with the conversation so the search knows what "its" means.

## 5. It refuses instead of guessing (1 min)
Select **All documents** and ask: `How do I bake sourdough bread?`
Expected: "I could not find relevant information in the uploaded documents." with no sources.
*Say:* the reranker score is below the threshold, so Gemini is never asked, which also saves quota.

## 6. Evaluation (1.5 min)
Open **Evaluation**: Hit@1 72%, Hit@5 97%, MRR 0.81 on 29 questions.
*Say:* the reranker alone scored lower on these documents, so the system keeps
hybrid order and uses the reranker only for the no-answer decision. Point out
the threshold table: 0.05 refuses 100% of out-of-scope questions at the cost of
refusing some answerable ones. Full-answer metrics need the real LLM.

## 7. Tests (30 s)
In a terminal: `uv run pytest -q` (14 passed) — and mention the 29 API tests that run
against a separate test database.

## Likely questions
- *Why 2 Gemini calls per question?* One to answer, one to check the answer is grounded.
- *Why is a not-found question slower?* It retries once with 40 candidates before refusing.
- *What would you improve next?* GPU reranking, PDF ligature cleanup, per-user documents.