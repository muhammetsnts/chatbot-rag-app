from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from multi_turn_pipeline.rag_pipeline import ask_question


app = FastAPI(title="Music Product RAG Chatbot")

# Static files (CSS, JS) and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    k: int = 10
    use_reranker: bool = False


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    """
    Landing page with chat UI.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ask")
async def api_ask(payload: AskRequest):
    """
    RAG question endpoint.

    Accepts:
    - question: user question (required)
    - session_id: client-side session identifier (optional, but recommended)
    - k: number of documents to retrieve (default: 10)
    - use_reranker: whether to enable the cross-encoder reranker
    """
    session_id = payload.session_id or "web_default_session"

    html_answer = ask_question(
        question=payload.question,
        k=payload.k,
        use_reranker=payload.use_reranker,
        session_id=session_id,
    )

    return {
        "answer_html": html_answer,
        "session_id": session_id,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


