from langchain_core.documents.base import Document
import os
import re
import markdown
import torch
import shutil
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .history_db import get_user_session, save_chat_history
from .settings import (CHROMA_DIR, 
                       CHROMA_ARCHIVE,
                       OPENROUTER_API_KEY_PATH, 
                       CLOUD_LLM_MODEL_NAME, 
                       EMBEDDING_MODEL_NAME,
                       RERANKER_MODEL_NAME
                       )


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Simple cross-encoder reranker wrapper using HF Transformers.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, docs: List[str], batch_size: int = 16) -> List[float]:
        """
        Return a list of relevance scores (one per doc) for the given query.
        Handles batching.
        Higher score = more relevant.
        """
        if not docs:
            return []

        scores = []
        # process in batches to avoid OOM
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            queries = [query] * len(batch_docs)

            inputs = self.tokenizer(
                queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model(**inputs).logits  # shape: (batch, n_labels) or (batch, 1)
            # reduce logits -> score per example
            # If multiple labels, you may want to take e.g. logits[:, 1] or logits.max(dim=1)
            if out.ndim == 2 and out.size(1) == 1:
                batch_scores = out.squeeze(-1).cpu().tolist()
            elif out.ndim == 2 and out.size(1) > 1:
                # assume higher logit corresponds to relevance: take max or positive class
                # adjust this line depending on reranker head (binary/class)
                batch_scores = out.max(dim=1).values.cpu().tolist()
            else:
                batch_scores = out.cpu().tolist()
                # ensure list of floats
                batch_scores = [float(x) for x in batch_scores]

            # if single float returned, normalize to list
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            scores.extend(batch_scores)

        return scores

_reranker_instance: Optional[CrossEncoderReranker] = None

def get_bge_reranker() -> CrossEncoderReranker:
    """
    Return a singleton instance of the BGE cross-encoder reranker.
    Avoids re-loading model weights for each request.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker(RERANKER_MODEL_NAME)
    return _reranker_instance


# ---------------------------------------------------------------------------
# Embeddings (bi-encoder)
# ---------------------------------------------------------------------------

def get_bge_embeddings() -> HuggingFaceEmbeddings:
    """
    Create a HuggingFace embeddings object for the configured embedding model.
    - model_name: EMBEDDING_MODEL_NAME (e.g. "BAAI/bge-base-en-v1.5")
    - device: "cuda" if available, else "cpu"
    - normalize_embeddings=True (recommended for cosine similarity)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings

# ---------------------------------------------------------------------------
# Retrieval + reranking
# ---------------------------------------------------------------------------
def rrf_merge(
    ranked_lists: List[List[Document]],
    k: int = 10,
    rrf_k: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF). Reorders the retrieved chunks by combining the vector search and BM25.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [doc_map[key] for key, _ in ranked]


def retrieve_documents_with_hybrid_search(
    query: str,
    vs: Chroma,
    k: int = 10,
) -> List[Document]:
    """
    Retrieve top-k documents for `query` from `vs` using hybrid search (vector search + BM25).
    """
    
    # 1. Vector search
    retriever = vs.as_retriever(search_type="similarity",search_kwargs={"k": k})
    vector_docs = retriever.invoke(query)  # docs from vector search

    # 2. BM25 search
    data = vs.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(data["documents"], data["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k
    bm25_docs = bm25_retriever.invoke(query)  # docs from BM25 search

    # 3. RRF merge
    hybrid_docs = rrf_merge(ranked_lists = [vector_docs, bm25_docs], k=k, rrf_k=60)
    return hybrid_docs

def retrieve_documents_without_reranker(
    query: str,
    vs: Chroma,
    k: int = 10,
) -> List[Document]:
    """
    Retrieve top-k documents for `query` from `vs` without reranker.
    """
    return retrieve_documents_with_hybrid_search(query, vs, k)
    

def retrieve_documents_with_reranker(
    query: str,
    vs: Chroma,
    k: int = 10,
) -> List[Document]:
    """
    Retrieve top-k documents for `query` from `vs` using reranker.
    """

    print(f"Retrieving {k} documents with reranker for query: {query}")

    candidate_docs = retrieve_documents_with_hybrid_search(query, vs, k)  # list[Document]

    if not candidate_docs:
        return []

    reranker = get_bge_reranker()
    texts = [d.page_content for d in candidate_docs]
    scores = reranker.score(query, texts)  # list[float]

    # Pair and sort by score descending
    pairs = list(zip(candidate_docs, scores))
    pairs.sort(key=lambda t: t[1], reverse=True)

    reranked_docs = [doc for doc, _ in pairs[:k]]
    return reranked_docs

def retrieve_documents(
    query: str,
    vs: Chroma,
    k: int = 10,
    use_reranker: bool = False,
) -> List[Document]:
    """
    Retrieve top-k documents for `query` from `vs`.
    """

    if not use_reranker:
        return retrieve_documents_without_reranker(query, vs, k)
    else:
        return retrieve_documents_with_reranker(query, vs, k)


# ---------------------------------------------------------------------------
# Load or unzip an existing Chroma vectorstore (runtime path)
# ---------------------------------------------------------------------------

# flag for vector database loading. If True, skip loading.
_VECTORSTORE: Optional[Chroma] = None

def build_or_load_vectorstore() -> Chroma:
    """
    Load an existing Chroma vectorstore if possible.

    Priority:
      1. If the Chroma directory already exists and is non-empty,
         load it from disk (using the default collection name,
         e.g. 'langchain' as in the shared notebook).
      2. Else, if a zipped archive (chroma_db.zip) exists, unzip it
         into CHROMA_DIR and then load the vectorstore.
      3. If neither directory nor archive exist, build the DB from scratch.

    This avoids re-embedding everything every time someone clones the repo
    and ensures we use the same collection that was created in the
    data teammate's notebook.
    """

    global _VECTORSTORE

    # return cached instance if already loaded
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    # 1) Existing directory → just load it
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"Loading existing Chroma DB from {CHROMA_DIR} ...")
        embeddings = get_bge_embeddings()
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("Vectorstore loaded successfully.")
        
        _VECTORSTORE = vectorstore
        return _VECTORSTORE

    # 2) No directory (or empty), but archive exists → unzip & load
    else:
        print(f"No Chroma directory found, but archive exists at {CHROMA_ARCHIVE}.")
        print("Unpacking Chroma DB archive into CHROMA_DIR ...")

        # Remove any existing directory (empty / wrong)
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)

        shutil.unpack_archive(str(CHROMA_ARCHIVE), extract_dir=str(CHROMA_DIR), format="zip")

        print("Archive unpacked. Loading vectorstore ...")
        embeddings = get_bge_embeddings()
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("Vectorstore loaded successfully.")
        
        _VECTORSTORE = vectorstore
        return _VECTORSTORE

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# Singleton prompt template instance + initialization flag to avoid reloading the prompt template repeatedly
_PROMPT_TEMPLATE: Optional[PromptTemplate] = None

def get_prompt_template() -> PromptTemplate:
    """
    Return the prompt template for RAG QA. If it is not None, return the cached instance. Otherwise, create a new instance.
    """
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is not None:
        return _PROMPT_TEMPLATE
    else:
        prompt_template = """You are a product knowledge assistant specialized in musical instruments and music-related equipment.
Your job is to answer user questions using ONLY the information contained in the retrieved context documents.

The dataset you are working with contains:
- product title, description, features, categories
- product metadata (store, color, rating, rating count, price, etc.)

### Your Rules:
1. Use ONLY the retrieved context to answer questions.  
2. Do NOT invent product details, specifications, or metadata that are not present in the context.  
3. If the context does not contain the required information, say:  
   "The provided product information does not include this detail."
4. If the question is about comparing products, create a clear comparison using only the available data.
5. Summaries must be concise and factual.
6. TRY to keep your answer short and concise.Preserve any numerical values exactly as they appear in the context.
7. If the user asks about availability or stock, respond with:  
   "This dataset does not include real-time availability information."
8. When the question is unclear, ask for clarification.
9. If metadata is available in the context (e.g., store, color, rating), include it in your answer.
10. NEVER output raw JSON, table, or database structures—respond in clean natural language.

### Response Format:
- Always provide a short, direct answer first.
- If relevant, include a structured breakdown:
  - **Key Features**
  - **Specifications / Metadata**
  - **Summary**

### Context:
{context}

### User Question:
{question}

### Final Answer:
(Your answer here)
"""

        _PROMPT_TEMPLATE = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        print("Prompt was created successfully.")
        return _PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# LLM instance
# ---------------------------------------------------------------------------

# Singleton LLM instance + initialization flag to avoid reloading API keys/models repeatedly
_LLM_INSTANCE: Optional[ChatOpenAI] = None
_LLM_INITIALIZED: bool = False

def get_llm() -> ChatOpenAI:
    """
    Return a ChatOpenAI LLM instance using OpenRouter.

    The OPENROUTER_API_KEY is read from the .env file at
    OPENROUTER_API_KEY_PATH.
    """

    global _LLM_INSTANCE, _LLM_INITIALIZED

    # Return cached instance if already created
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE

    # Only load .env once (first initialization). Subsequent calls reuse environment values.
    if not _LLM_INITIALIZED:
        load_dotenv(dotenv_path=OPENROUTER_API_KEY_PATH, override=True)
        _LLM_INITIALIZED = True

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    if OPENROUTER_API_KEY:
        print("✅ API key is present (loaded or already set).")
    else:
        print("❌ API key couldn't be found. Please check the .env file.")
        print(f"🔍 Searched location: {OPENROUTER_API_KEY_PATH}")

    # Create and cache LLM instance
    _LLM_INSTANCE = ChatOpenAI(
        model_name=CLOUD_LLM_MODEL_NAME[0],
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        temperature=0.7,
        max_tokens=2000,
    )

    print("LLM instance created and cached.")

    return _LLM_INSTANCE


# ---------------------------------------------------------------------------
# Format retrieved documents 
# ---------------------------------------------------------------------------
def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        formatted.append(
            f"### Document {i+1}\n"
            f"Content:\n{doc.page_content}\n\n"
            f"Metadata:\n{doc.metadata}\n"
        )
    return "\n\n".join(formatted)


# ---------------------------------------------------------------------------
# Rewrite question with chat history
# ---------------------------------------------------------------------------

def rewrite_question_with_history(llm, question: str, history: Optional[List[Dict]]):
    """Rewrite a follow-up question into a standalone question using chat history."""

    if len(history) == 0:
        return question
    # Keep last N messages to limit tokens
    last_msgs = history[-4:]
    history_text = ""
    for m in last_msgs:
        history_text += f"{m['role'].upper()}: {m['content']}\n"
    rewrite_prompt = f"""You are an assistant that rewrites follow-up questions into fully self-contained questions.

### Instructions:
1. Do NOT answer; only rewrite.
2. Use ONLY the conversation history to resolve references like “this,” “that,” “these,” or “the product.”
3. If a reference is ambiguous, make the best guess based on history.
4. Do NOT add new facts not present in the history.
5. PRESERVE product names, prices, specs, and other metadata EXACTLY as they appear.
6. If the question is already standalone, return it as is.
7. Output ONLY the rewritten question.

### Conversation History:
{history_text}

### User's Latest Question:
{question}

### Rewritten Standalone Question:
(Your answer here)
"""
    resp = llm.invoke(rewrite_prompt)
    return getattr(resp, "content", question).strip()



# ---------------------------------------------------------------------------
# RAG pipeline function
# ---------------------------------------------------------------------------
def ask_question(
    question: str,
    k: int = 8,
    use_reranker: bool = False,
    session_id: str = "default_session",
) -> str:
    """
    Run the full RAG pipeline:
    1) Retrieve top-k relevant documents from Chroma DB.
    2) Format them and create the prompt.
    3) Call the LLM to generate the answer.

    Parameters
    ----------
    question : str
        The user question to answer.
    k : int, optional
        Number of documents to retrieve, by default 10
    use_reranker : bool, optional
        Whether to use the cross-encoder reranker, by default True

    Returns
    -------
    answer : str
        The generated answer from the LLM.
    """

    original_question = question # save for later

    # 1) Get LLM instance
    llm = get_llm()

    # 2) Load chat history for user/session
    # Normalize session_id to avoid accidental None/empty values causing DB NOT NULL errors
    if not session_id:
        session_id = "default_session"
    else:
        session_id = str(session_id).strip() or "default_session"

    history = get_user_session(session_id=session_id) or []

    if len(history) > 0:
        # 1) rephrase if needed
        question = rewrite_question_with_history(llm, question, history)

    # 2) Load or build vectorstore (cached singleton)
    vs: Chroma = build_or_load_vectorstore()

    # 3) Retrieve documents
    docs: List[Document] = retrieve_documents(
        query=question,
        vs=vs,
        k=k,
        use_reranker=use_reranker,
    )

    # 4) Format retrieved documents
    context = format_docs(docs)

    # 5) Create prompt
    prompt = get_prompt_template()

    # 6) Generate answer
    response = llm.invoke(prompt.format(context=context, question=question))
    answer = getattr(response, "content", "")

    # 7) Convert answer to HTML
    html_answer = convert_answer_to_html(answer)

    # 8) Save updated chat history
    new_message_entry = [{"role": "user", "content": original_question}, {"role": "assistant", "content": answer}]
    history.extend(new_message_entry)
    print(f"Saving chat history for session_id={session_id!r}")
    save_chat_history(session_id=session_id, messages=history, html_answer=html_answer)

    return html_answer

# ---------------------------------------------------------------------------
# Convert answer to HTML
# ---------------------------------------------------------------------------

def convert_answer_to_html(answer: str) -> str:
    """
    Convert the LLM answer to HTML format for better display.

    Parameters
    ----------
    answer : str
        The raw answer from the LLM.

    Returns
    -------
    html_answer : str
        The answer converted to HTML format.
    """

    modified_answer = re.sub(
        r"(\*\*[^*\n]+?\*\*)\n(-|\*)",
        r"\1\n\n\2",
        answer,
    )

    html_answer = markdown.markdown(modified_answer)
    return html_answer