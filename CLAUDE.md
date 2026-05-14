# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGweb is a Retrieval-Augmented Generation web app. Users upload PDFs, the app indexes them into a vector store (LanceDB), and then answers questions by retrieving relevant chunks and sending them as context to an LLM.

## Tech Stack

- **UI**: Gradio (`app.py`)
- **Frontend templates**: Jinja2 (`frontend/template.j2` for the LLM prompt, `frontend/template_html.j2` for the Gradio HTML pane)
- **Vector store**: LanceDB (local, stored in `./lancedb`)
- **Embedding model**: `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: DeepSeek V4 Flash, called via the OpenAI-compatible client

## Running the App

```bash
# Copy and fill in the env template
cp .template.env .env
# Set DEEPSEEK_API_KEY in .env

# Install dependencies
pip install -r requirements.txt

# Run (single entry point)
python app.py
```

The Gradio UI launches at `http://localhost:7860`. Upload a PDF, then ask questions in the chat.

## Evaluation

```bash
# Run baseline metrics on SQuAD (100 questions by default)
python -m backend.evaluation --samples 100 --top-k 5
```

Results are printed to stdout and saved to `baseline_metrics.json`. The evaluation:
- Loads SQuAD v1.1 validation set
- Indexes all unique contexts into LanceDB
- Runs retrieval + LLM QA on each question
- Reports Exact Match (EM) and F1 scores

Requires `datasets` package (in requirements.txt) and a valid `DEEPSEEK_API_KEY`.

## Code Quality

```bash
make format   # format all code with ruff
make lint     # check for linting issues
make check    # lint + run tests
```

Formatting and linting are enforced by [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`.

## Architecture & Data Flow

```
PDF → pdf_parser() → chunk_text() → SentenceTransformer.encode() → LanceDB table
                                                                          │
Query ──→ SentenceTransformer.encode() → LanceDB.search() → CrossEncoder.rerank() → top-k chunks
                                                                                        │
                                        Jinja2 template.render(documents, query, history) → LLM streaming response → Gradio chatbot
```

### Backend modules

1. **`backend/propressing.py`** — Document ingestion
   - `pdf_parser(pdf_path)` → list of page dicts (text, page number, char/word counts) using PyMuPDF (fitz)
   - `chunk_text(text, num_sents=4)` → sentence-level chunking via spaCy `sentencizer`; fixed-size sliding window of 4 sentences
   - `embedding(pdf_path)` → orchestrates parse → chunk → embed → write to LanceDB; returns the vector store name (`vs_<sha256hash>`)

2. **`backend/search.py`** — Retrieval + reranking
   - `search(vs_name, query, k=25)` → vector similarity search in LanceDB
   - `retrieve(vs_name, query, k=25, rerank=True, top_k=5)` → optionally reranks results with CrossEncoder (sigmoid activation), returns top-k text chunks

3. **`backend/query.py`** — LLM call
   - `query_deepseek(prompt)` → streams completions from DeepSeek V4 Flash via the OpenAI SDK (base URL `https://api.deepseek.com`)

4. **`app.py`** — Gradio UI + orchestration
   - File upload triggers `embedding()` → stores vector store name in `gr.State`
   - Chat messages flow: user input → `retrieve()` → Jinja2 `template.render()` → `query_deepseek()` → streamed back to `gr.Chatbot`

### Frontend templates

- **`template.j2`**: Builds the raw text prompt sent to the LLM with sections for Instructions, History, Context (retrieved docs), and Query.
- **`template_html.j2`**: Renders the same prompt as HTML shown in a `gr.HTML` pane, with collapsible `<details>` elements for each retrieved document.

## Configuration (`.env`)

Key env vars: `EMB_MODEL`, `RERANK_MODEL`, `DEEPSEEK_API_KEY`, `TEMPERATURE`, `MAX_NEW_TOKENS`, `TOP_P`, `FREQ_PENALTY`, `TOP_K`.

## Notes

- The app uses global module-level variables (`db` in `search.py`, `DEEPSEEK_CLIENT` in `query.py`) — reimporting modules won't re-read env vars or reconnect LanceDB without a process restart.
- LanceDB data persists in `./.lancedb/` (gitignored). The vector store name is a SHA-256 hash of the embedding model + sub-vector count, so re-uploading the same PDF overwrites the previous store for that model config.
- Tests are in the `tests/` directory, run with `pytest tests/`.
