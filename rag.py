from __future__ import annotations
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE   = "vectorstore"
LLM_MODEL     = "phi3"
TOP_K         = 6      # fetch more candidates …
MMR_K         = 3      # … then keep only the most diverse ones
FETCH_K       = 20     # MMR candidate pool
MAX_CTX_CHARS = 3000   # cap total context fed to LLM
MAX_HISTORY   = 6      # last N turns kept in memory (3 user + 3 assistant)
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MediChat, a compassionate and knowledgeable AI healthcare assistant.

Your role:
- Provide clear, accurate, evidence-based medical information.
- Explain symptoms, possible causes, and general treatment options in plain language.
- Always recommend consulting a qualified healthcare professional for diagnosis or treatment.
- Never fabricate drug names, dosages, or test results.
- If a question is outside medical scope, say so politely.

Tone: warm, calm, professional — like a knowledgeable friend who happens to be a doctor.

Format rules:
- Use short paragraphs. Bullet points only when listing multiple items.
- Keep answers under 200 words unless the topic requires more depth.
- End with a brief note to see a doctor when the symptom sounds serious."""


def _load_resources() -> FAISS:
    """Load embedding model and FAISS index once at module import."""
    print("Loading embedding model …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Loading vector database …")
    db = FAISS.load_local(
        VECTORSTORE,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("MediChat ready ✓")
    return db


# Module-level singletons — loaded once per process
_db: FAISS = _load_resources()

# Retrieval helpers

def _retrieve_context(question: str) -> str:
    """
    MMR (Maximal Marginal Relevance) retrieval:
    - Fetches FETCH_K candidates by similarity.
    - Re-ranks to keep MMR_K diverse, non-redundant passages.
    This avoids pumping near-duplicate paragraphs into the prompt.
    """
    docs = _db.max_marginal_relevance_search(
        question,
        k=MMR_K,
        fetch_k=FETCH_K,
    )

    # Deduplicate by uid metadata if present
    seen: set[str] = set()
    unique_docs = []
    for doc in docs:
        uid = doc.metadata.get("uid", doc.page_content[:40])
        if uid not in seen:
            seen.add(uid)
            unique_docs.append(doc)

    # Trim total context to MAX_CTX_CHARS
    chunks: list[str] = []
    total = 0
    for doc in unique_docs:
        text = doc.page_content.strip()
        if total + len(text) > MAX_CTX_CHARS:
            text = text[: MAX_CTX_CHARS - total]
        chunks.append(text)
        total += len(text)
        if total >= MAX_CTX_CHARS:
            break

    return "\n\n---\n\n".join(chunks)


def _is_medical_question(question: str) -> bool:
    if len(question.strip()) < 5:
        return False
    non_medical = re.compile(
        r"\b(joke|weather|stock|sport|recipe|cook|game|movie|music)\b", re.I
    )
    return not non_medical.search(question)


# Public API

def ask_medical_question(
    question: str,
    history: list[dict] | None = None,
) -> str:

    question = question.strip()

    if not question:
        return "Please enter a question so I can help you."

    if not _is_medical_question(question):
        return (
            "I'm designed to answer medical and health-related questions. "
            "Could you rephrase or ask something health-related?"
        )

    try:
        context = _retrieve_context(question)

        # Build the user-facing prompt with retrieved context
        user_prompt = (
            f"Relevant medical reference:\n{context}\n\n"
            f"Patient question: {question}\n\n"
            "Please answer helpfully based on the reference above and your medical knowledge."
        )

        # Compose message list: system + trimmed history + current turn
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            trimmed = history[-(MAX_HISTORY):]
            messages.extend(trimmed)

        messages.append({"role": "user", "content": user_prompt})

        response = ollama.chat(model=LLM_MODEL, messages=messages)
        answer: str = response["message"]["content"].strip()

        # Strip any accidental "Helpful Answer:" prefix echoed from old prompt
        answer = re.sub(r"^helpful answer:\s*", "", answer, flags=re.I)

        return answer

    except ollama.ResponseError as e:
        return (
            f"The language model returned an error: {e.error}. "
            "Please make sure Ollama is running and the phi3 model is pulled."
        )
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
