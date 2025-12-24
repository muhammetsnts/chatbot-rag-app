Here’s a **clean rewrite with light, tasteful emojis** (professional but friendly):

---

# 🎸 CHATBOT – LLM & RAG-Powered Music Gear Assistant 🤖

![chatbot_demo](https://github.com/user-attachments/assets/7fcbbd7d-ed82-40f6-b268-165f6ed64588)


This project is a **multi-turn, retrieval-augmented chatbot** for a **music instruments & gear catalog** 🎶.
It uses **LangChain** with an **OpenRouter-hosted LLM**, applies **hybrid search (vector + BM25)** with **Reciprocal Rank Fusion (RRF)**, and optionally supports **reranking** for higher precision.

A **FastAPI backend** serves both the API and a modern **web UI** (landing page + chat).

---

## ✨ Features

* 💬 Multi-turn chat with per-session history
* 🔁 Question rewriting (follow-ups → standalone queries)
* 🔍 **Hybrid search**: semantic vector search + BM25 keyword matching via RRF
* 🧠 Chroma vector store (can be loaded from a bundled zip)
* 🧪 Optional cross-encoder reranker (post-retrieval)
* 🧾 HTML-formatted answers
* ⚡ FastAPI backend + static frontend (landing + chat)

---

## 🏗️ Architecture – Quick Tour

* `multi_turn_pipeline/rag_pipeline.py` → Full RAG pipeline (`ask_question` is the main entrypoint)
* `multi_turn_pipeline/history_db.py` → SQLite persistence for chat history
* `multi_turn_pipeline/settings.py` → Paths & model settings (Chroma dir, `.env`, etc.)
* `app.py` → FastAPI server (`/` frontend, `/api/ask` API)
* `templates/index.html` + `static/*` → Landing page & chat UI

---

## ⚙️ Setup

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Environment variable 🔐

Create an `OPENROUTER_API_KEY` from
👉 [https://openrouter.ai/docs/api/api-reference/api-keys/create-keys](https://openrouter.ai/docs/api/api-reference/api-keys/create-keys)

Set it at the path defined in `multi_turn_pipeline/settings.py`
(`OPENROUTER_API_KEY_PATH`, typically `.env`):

```
OPENROUTER_API_KEY=...your_key...
```

### 3️⃣ Vector database 📦

* If `chroma_db.zip` exists at repo root → unzip as `chroma_db/`
* If not, the app will:

  * try to unzip automatically
  * otherwise build a fresh Chroma DB (re-embeds documents)

---

## ▶️ Run

```bash
uvicorn app:app --reload
```

Open 👉 `http://127.0.0.1:8000` to access the landing page + chat UI.

---

## 🔌 API

**POST** `/api/ask`

**Request body**

```json
{
  "question": "...",
  "session_id": "optional",
  "k": 10,
  "use_reranker": false
}
```

**Response**

```json
{
  "answer_html": "<p>...</p>",
  "session_id": "..."
}
```

---

## 🧠 Sessions & History

* SQLite file: `chat_history.db` (git-ignored)
* Provide `session_id` to preserve multi-turn context
* If omitted, `default_session` is used

---

## 🔎 Retrieval Strategy (Hybrid Search)

The system combines three techniques for high-quality retrieval:

1. 🧠 **Vector search** – semantic similarity
2. 🔑 **BM25 search** – exact keyword matching
3. 🔗 **Reciprocal Rank Fusion (RRF)** – merges results via rank-based scoring

✅ This captures both **meaning** and **keywords**, improving relevance significantly.

---

## 🎯 Reranker (Optional)

* Cross-encoder reranker (HF Transformers)
* Applied **after hybrid search**
* Enable with:

```python
ask_question(..., use_reranker=True)
```

📌 Use reranker for **maximum quality**, disable for **lower latency**
(Hybrid search alone is already strong.)

---

## 🖥️ Frontend

* Modern single-page UI → `templates/index.html`
* Static assets → `static/styles.css`, `static/app.js`
* Includes:

  * Landing page
  * Chat window
  * Reranker toggle
  * Local `session_id` for multi-turn memory

---

## 🛠️ Dev Notes

* `.env` and `chat_history.db` are **git-ignored** 🚫
* A standard Python environment with `requirements.txt` is sufficient

---

## 🧯 Quick Troubleshooting

* ❌ **“API key missing”** → Check `.env` path and key value
* 🎸 **“Guitar search returns accessories”** → Enable reranker for better precision
* 🐢 **Slow responses** → Set `use_reranker=False` (default)

---

## 📄 License

Unless stated otherwise, apply your standard project license.
