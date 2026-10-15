# RAG-X — Advanced Multi-Document AI Knowledge Assistant

Ask questions about your own documents and get answers with citations.
RAG-X searches your uploaded files, picks the best passages, asks Gemini to answer
**only from those passages**, and shows exactly which file and page each fact came from.
If the documents do not contain the answer, it says so instead of guessing.

**Stack:** FastAPI · Streamlit · LangGraph · Qdrant · PostgreSQL · Gemini ·
`bge-m3` (dense embeddings) · BM42 (sparse embeddings) · `bge-reranker-base`

## What it does

- Upload **PDF, DOCX, TXT and HTML** files; they are parsed, cleaned, split into
  chunks (about 1000 characters, 150 overlap) and indexed.
- **Hybrid search:** dense + sparse retrieval fused by Qdrant (RRF).
- **Chat with memory:** follow-up questions such as "What are its main advantages?"
  are rewritten using the conversation so far.
- **Document scoping:** ask across all documents, one document, or a selection.
- **Citations:** answers carry `[1]`, `[2]` markers linked to file, page and passage.
- **No-answer protection:** a reranker score decides whether the documents really
  contain the answer; otherwise RAG-X refuses.
- **Grounding check:** a second Gemini call scores how well the answer is supported.
- **Pages:** Chat, Documents, Search, Dashboard, Evaluation, Document Intelligence, Settings.

## How a question is answered

```
question
  -> (rewrite with chat history, only if there is history)
  -> hybrid search: top 20 passages
  -> reranker scores them (0 to 1); passages keep hybrid order
  -> enough evidence?  best score >= 0.05
        yes -> Gemini answers with [n] citations -> grounding check -> answer
        no  -> retry once with 40 candidates -> still no -> "I could not find
               relevant information in the uploaded documents."
```

## Quick start with Docker

Requirements: Docker Desktop (about 8 GB of RAM available to it).

```powershell
Copy-Item .env.example .env        # then edit .env (see Configuration)
docker compose --profile app up --build -d
```

- UI: http://127.0.0.1:8501 — API docs: http://127.0.0.1:8001/docs
- The **first start downloads the AI models** (a few GB) into a Docker volume.
  Watch with `docker compose --profile app logs -f api` until you see
  `Application startup complete`.
- Stop with `docker compose --profile app down`.
  **Never add `-v`**: that deletes your documents and index.
- `docker compose up -d` (without `--profile app`) starts only PostgreSQL and Qdrant,
  which is what you want for local development.

## Run locally (without Docker for the app)

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker (for the databases).

```powershell
uv sync
Copy-Item .env.example .env
docker compose up -d               # PostgreSQL on 5433, Qdrant on 6335
uv run uvicorn backend.app.main:app --port 8001
```

In a second terminal:

```powershell
uv run streamlit run frontend/app.py
```

The UI talks to `http://127.0.0.1:8001`. To use another address set
`RAGX_BACKEND_URL` before starting Streamlit.

Tip: once the models are downloaded, `$env:HF_HUB_OFFLINE = "1"` skips slow
Hugging Face network checks at startup.

## Configuration (`.env`)

| Setting | Meaning |
|---|---|
| `DATABASE_URL` | PostgreSQL connection. Default uses `pg8000` on port 5433. |
| `QDRANT_URL`, `QDRANT_COLLECTION` | Qdrant address (port 6335) and collection name. |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Your key from https://aistudio.google.com/ and the model name. |
| `MOCK_LLM` | `true` = fixed fake answers, no key needed (good for wiring tests). `false` = real Gemini. |
| `DEBUG` | `true` prints every SQL statement (very noisy). Keep `false`. |

Ports 5433/6335 are used so this project does not clash with another local
PostgreSQL/Qdrant. If you change them, change `docker-compose.yml` and `.env` together.
Inside Docker, the app containers reach the databases by service name automatically.

**Real answers need `MOCK_LLM=false` and a valid key.** With the mock LLM, search,
citations and the no-answer rule work, but every answer is the same placeholder sentence.
Each answerable question costs 2 Gemini calls (answer + grounding check), 3 with chat
history, and 0 when the documents have no answer. Free-tier quotas can run out quickly.

## Tests

```powershell
uv run pytest -q                      # 14 fast unit tests, no Docker or models needed
uv run pytest -m integration -q       # 29 API tests, about 1 minute
```

Integration tests need PostgreSQL and Qdrant running. They use a separate
`rag_x_test` database and `rag_x_test_documents` collection (created automatically and
guarded so they can never touch your real documents).

## Evaluation

```powershell
uv run python evaluation/run_eval.py             # retrieval quality + no-answer threshold
uv run python evaluation/run_eval.py --answers   # also generate full answers (uses the LLM)
uv run python evaluation/performance.py          # latency and Gemini calls per question
```

`evaluation/eval_set.json` holds 29 answerable and 12 unanswerable questions about the
indexed sample documents. Results are shown on the **Evaluation** page.

| Retrieval (29 questions) | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Hybrid order (used) | 72% | 90% | 97% | 0.81 |
| Reranker order (rejected) | 55% | 83% | 83% | 0.68 |

Performance (mock LLM, CPU only, 6 documents; add about 1–3 s per Gemini call):

| Step | Typical time |
|---|---|
| Search with reranking | 2.2 s (0.3 s without) |
| Chat, answerable question | 2.5 s |
| Chat, follow-up question | 1.9 s |
| Chat, no answer in documents | 6.2 s |
| Loading the AI models at startup | 20–35 s |

## Design decisions worth knowing

- **Hybrid order beats reranker order on our documents** (table above), so passages are
  returned in hybrid order. The reranker still scores every returned passage and that
  score drives the "is there an answer?" decision. Only the passages actually returned
  are reranked, which cut search time from about 6.7 s to 2.2 s with identical results.
- **The no-answer threshold is 0.05.** Answerable and unanswerable questions overlap in
  score (lowest answerable 0.0043, highest unanswerable 0.0156), so no threshold is
  perfect. 0.05 never answers an out-of-scope question (100% refused) but also refuses
  some answerable ones (83% answered). A lower threshold of about 0.016 answers 86%.
  Change `MIN_CONTEXT_SCORE` in `backend/app/services/context_evaluator.py`.

## Project layout

```
backend/app/api/routes/    FastAPI routes: documents, chat, search, health
backend/app/services/      parsing, chunking, embeddings, search, reranking, Gemini, citations
backend/app/rag/graph/     LangGraph workflow (rewrite, retrieve, rerank, evaluate, generate)
frontend/                  Streamlit app (app.py, pages/, utils/)
evaluation/                eval set, evaluation and performance scripts, latest results
tests/                     pytest suite (unit + integration)
scripts/manual_checks/     old manual check scripts (they touch real data; run with
                           uv run python -m scripts.manual_checks.check_rag_graph)
docker/, docker-compose.yml  container setup
```

## Troubleshooting

- **A page shows errors or stale data after restarting containers:** a leftover local
  `uvicorn`/`streamlit` may still own port 8001/8501 and answer instead of Docker.
  Check with `Get-NetTCPConnection -LocalPort 8001 -State Listen` and stop the old
  `python` process.
- **Windows Smart App Control blocks `psycopg`:** use the `postgresql+pg8000://` URL
  (already the default).
- **Gemini returns 429:** the free quota is used up. Wait, use another key, or set
  `MOCK_LLM=true`.
- **First start is slow:** the models are loading or downloading (see Quick start).

## Known limitations

- Single-user app: no login, no per-user documents, no upload size limit.
- Reranking runs on the CPU, so it is the slowest step.
- Ligatures and hyphenation from PDFs (for example "ﬁnite") are not cleaned yet.
- Supported file types: `.pdf`, `.docx`, `.txt`, `.html`.