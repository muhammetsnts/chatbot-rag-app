# CHATBOT – LLM & RAG-powered Music Gear Assistant

This project is a multi-turn, retrieval-augmented chatbot for a music instruments / gear catalog. It uses LangChain with an OpenRouter-hosted LLM, employs hybrid search (vector + BM25) with Reciprocal Rank Fusion (RRF), and can optionally rerank results. A FastAPI backend serves both the API and a modern web UI (landing + chat).

## Features
- Multi-turn chat with per-session history
- Question rewriting (turns follow-ups into standalone queries)
- **Hybrid search**: Combines vector search (semantic similarity) and BM25 (keyword matching) using RRF
- Chroma vector store (can be loaded from a bundled zip)
- Optional cross-encoder reranker (applied after hybrid search)
- HTML-formatted answers
- FastAPI service + static frontend (landing + chat box)

## Architecture Quick Tour
- `multi_turn_pipeline/rag_pipeline.py`: Full RAG flow; `ask_question` is the main entrypoint.
- `multi_turn_pipeline/history_db.py`: SQLite persistence for chat history.
- `multi_turn_pipeline/settings.py`: Paths and model settings (Chroma dir, .env location, etc.).
- `app.py`: FastAPI server exposing `/` (frontend) and `/api/ask` (RAG query).
- `templates/index.html` + `static/*`: Landing page and chat UI.

## Setup
1) Install dependencies  
```bash
pip install -r requirements.txt
```

2) Environment variable  

Create a new `OPENROUTER_API_KEY` from (OpenRouter web page)[https://openrouter.ai/docs/api/api-reference/api-keys/create-keys] and set at the path defined in `multi_turn_pipeline/settings.py` (`OPENROUTER_API_KEY_PATH`, typically `.env`):
```
OPENROUTER_API_KEY=...your_key...
```

3) Vector database  
- If `chroma_db.zip` is present at repo root, unzip and use as `chroma_db/`.  
- If not, the code will try to unzip automatically; if no zip exists, it will build a fresh Chroma DB (re-embeds docs).

## Run
```bash
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000` for the landing + chat UI.

## API
- `POST /api/ask`
  - Body: `{"question": "...", "session_id": "optional", "k": 10, "use_reranker": false}`
  - Returns: `{"answer_html": "<p>...</p>", "session_id": "..." }`

## Sessions & History
- SQLite file: `chat_history.db` (ignored by git)
- Provide `session_id` for multi-turn context; if omitted, `default_session` is used.

## Retrieval Strategy

The system uses a **hybrid search** approach that combines:
1. **Vector search** (semantic similarity) - Finds documents similar in meaning
2. **BM25 search** (keyword matching) - Finds documents with exact keyword matches
3. **Reciprocal Rank Fusion (RRF)** - Combines results from both methods using rank-based scoring

This hybrid approach captures both semantic understanding and exact keyword matches, improving retrieval quality.

## Reranker
- Optional cross-encoder (HF Transformers) applied **after** hybrid search
- Enable via `ask_question(..., use_reranker=True)`
- Use it for maximum quality; disable for speed (hybrid search alone is already quite effective)

## Frontend
- Modern single page: `templates/index.html`
- Static assets: `static/styles.css`, `static/app.js`
- Includes landing, chat window, reranker toggle, and local `session_id` to preserve multi-turn context.

## Dev Notes
- `chat_history.db` and `.env` files are `.gitignore`d and should not be pushed.
- A working Python env with `requirements.txt` is sufficient.

## Quick Troubleshooting
- “API key missing”: check `.env` location and key value.
- “Searching guitars returns accessories”: hybrid search should help, but you can also enable reranker for better precision.
- Slow responses: try `use_reranker=False` (default=False). Hybrid search is fast; reranker adds latency but improves quality.

## License
Unless stated otherwise, apply your standard project license.