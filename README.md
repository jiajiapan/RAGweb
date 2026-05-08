# RAGweb

Retrieval-Augmented Generation web app. Upload PDFs, ask questions, get answers grounded in your documents.

## Features

- **PDF ingestion** — parses PDFs with PyMuPDF, chunks text at sentence level via spaCy
- **Vector search** — embeds chunks with `sentence-transformers/all-mpnet-base-v2` and indexes them in LanceDB
- **Reranking** — re-ranks retrieved chunks with a cross-encoder for higher relevance
- **LLM-powered answers** — streams responses from DeepSeek with retrieved context via Jinja2 prompt templates
- **Gradio UI** — file upload + chat interface with inline retrieval source display

## Setup

```bash
# Clone and install
git clone <repo-url>
cd RAGweb
pip install -r requirements.txt

# Configure
cp .template.env .env
# Edit .env and set your DEEPSEEK_API_KEY
```

## Usage

```bash
python app.py
```

Open `http://localhost:7860`, upload a PDF, then ask questions in the chat.

## Configuration

Key environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (required) |
| `EMB_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Embedding model (768-dim) |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `TOP_K` | `5` | Number of chunks sent to LLM |
| `TEMPERATURE` | `1.0` | LLM sampling temperature |
| `MAX_NEW_TOKENS` | `512` | Max tokens in LLM response |
| `TOP_P` | `0.4` | Nucleus sampling threshold |

## Architecture

```
PDF → pdf_parser() → chunk_text() → SentenceTransformer.encode() → LanceDB
                                                                        │
Query → SentenceTransformer.encode() → LanceDB.search() → CrossEncoder.rerank()
                                                                        │
                              Jinja2 template.render(docs, query, history) → LLM streaming → Chatbot
```

### Modules

- **`backend/propressing.py`** — document parsing, chunking, embedding, and LanceDB ingestion
- **`backend/search.py`** — vector similarity search + cross-encoder reranking
- **`backend/query.py`** — LLM streaming via OpenAI-compatible client (DeepSeek)
- **`app.py`** — Gradio UI orchestration

### Templates

- **`template.j2`** — LLM prompt with instructions, history, context, and query
- **`template_html.j2`** — HTML rendering of the prompt with expandable document sections

## Tech Stack

| Component | Library |
|---|---|
| UI | Gradio |
| PDF parsing | PyMuPDF |
| Text chunking | spaCy |
| Vector store | LanceDB |
| Embeddings | Sentence Transformers |
| Reranking | Cross-Encoder |
| LLM | DeepSeek V4 Flash (OpenAI SDK) |
| Prompting | Jinja2 |
