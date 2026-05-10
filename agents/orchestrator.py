from __future__ import annotations
import re
import os
import json
import pickle
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import TypedDict, Annotated, Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator

logger = logging.getLogger(__name__)

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE   = "vectorstore"
GROQ_MODEL    = "llama-3.1-8b-instant"          # free tier, fast
MMR_K         = 3
FETCH_K       = 20
MAX_CTX_CHARS = 3000
TRIAGE_MODEL  = "agents/triage_model.pkl"        # saved from notebook
PUBMED_MAX    = 3                                 # articles to fetch

# Shared State
class AgentState(TypedDict):
    """Shared state passed between every node in the graph."""
    question:         str
    history:          list[dict]
    # Triage outputs
    medical_category: str
    intent:           str          # "symptom_check" | "drug_info" | "emergency" | "general"
    triage_note:      str
    # Agent outputs
    rag_context:      str
    literature:       str
    safety_flag:      bool
    # Final
    answer:           str
    citations:        list[str]


# loaded once
def _load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def _load_faiss(embeddings: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.load_local(
        VECTORSTORE, embeddings, allow_dangerous_deserialization=True
    )

def _load_triage_model():
    """Load the serialised sklearn triage classifier if available."""
    path = Path(TRIAGE_MODEL)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

_embeddings  = _load_embeddings()
_db          = _load_faiss(_embeddings)
_triage_clf  = _load_triage_model()
_llm         = ChatGroq(model=GROQ_MODEL, temperature=0.3, api_key=os.environ.get("GROQ_API_KEY", ""),
)

# Safety patterns
_EMERGENCY_RE = re.compile(
    r"\b(chest pain|heart attack|stroke|can't breathe|suicid|overdose|"
    r"unconscious|seizure|severe bleeding|911|emergency)\b",
    re.IGNORECASE,
)

# AGENT 1 — TriageAgent
# Classifies the question: medical category + intent type
def triage_agent(state: AgentState) -> AgentState:
    """
    Uses the trained LinearSVC classifier (from compare_ML_models.ipynb)
    to predict the medical category, then uses rule-based heuristics +
    a fast LLM call to determine intent.
    """
    question = state["question"]
    logger.info("[TriageAgent] Classifying: %s", question[:60])

    # 1. Safety check — always first
    if _EMERGENCY_RE.search(question):
        return {
            **state,
            "medical_category": "emergency",
            "intent":           "emergency",
            "triage_note":      "Emergency keywords detected.",
            "safety_flag":      True,
        }

    category = "general medicine"
    if _triage_clf is not None:
        try:
            category = _triage_clf.predict([question])[0]
        except Exception:
            pass

    intent_prompt = f"""Classify this medical question into ONE intent:
- symptom_check   (patient describing symptoms)
- drug_info       (asking about medication, dosage, interaction)
- emergency       (urgent, life-threatening)
- general         (general health knowledge, sleep, nutrition, exercise, prevention, etc.)

Question: "{question}"

Reply with ONLY the intent label, nothing else."""

    try:
        resp = _llm.invoke([HumanMessage(content=intent_prompt)])
        intent = resp.content.strip().lower().split()[0]
        valid_intents = {"symptom_check", "drug_info", "emergency", "general"}
        if intent not in valid_intents:
            intent = "general"
    except Exception:
        intent = "general"

    logger.info("[TriageAgent] category=%s intent=%s", category, intent)

    return {
        **state,
        "medical_category": category,
        "intent":           intent,
        "triage_note":      f"Category: {category} | Intent: {intent}",
        "safety_flag":      intent == "emergency",
    }

# AGENT 2 — SafetyAgent
# Handles emergency / crisis questions immediately
def safety_agent(state: AgentState) -> AgentState:
    logger.info("[SafetyAgent] Emergency detected — returning crisis response.")
    answer = (
        "This sounds like a medical emergency. Please call 911 (or your "
        "local emergency number) immediately or go to the nearest emergency room.\n\n"
        "Do not wait for an online response in an emergency situation. "
        "If you are in the US, you can also call Poison Control at 1-800-222-1222 "
        "or the Crisis Lifeline at 988."
    )
    return {**state, "answer": answer, "citations": []}

# AGENT 3 — RAGAgent
# Retrieves relevant passages from your FAISS medical corpus
def rag_agent(state: AgentState) -> AgentState:
    question = state["question"]
    logger.info("[RAGAgent] Retrieving for: %s", question[:60])

    try:
        docs = _db.max_marginal_relevance_search(
            question, k=MMR_K, fetch_k=FETCH_K
        )

        # Deduplicate by uid
        seen: set[str] = set()
        unique_docs: list[Document] = []
        for doc in docs:
            uid = doc.metadata.get("uid", doc.page_content[:40])
            if uid not in seen:
                seen.add(uid)
                unique_docs.append(doc)

        # Trim to budget
        chunks, total = [], 0
        for doc in unique_docs:
            text = doc.page_content.strip()
            if total + len(text) > MAX_CTX_CHARS:
                text = text[: MAX_CTX_CHARS - total]
            chunks.append(text)
            total += len(text)
            if total >= MAX_CTX_CHARS:
                break

        context = "\n\n---\n\n".join(chunks)
    except Exception as e:
        logger.warning("[RAGAgent] FAISS retrieval failed: %s", e)
        context = ""

    return {**state, "rag_context": context}

# AGENT 4 — LiteratureAgent
# Fetches live PubMed abstracts via E-utilities

def _pubmed_search(query: str, max_results: int = PUBMED_MAX) -> list[str]:
    """Returns a list of abstract snippets from PubMed."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    q = urllib.parse.quote(f"{query}[Title/Abstract] AND medline[sb]")

    # Step 1 — search for IDs
    search_url = f"{base}esearch.fcgi?db=pubmed&term={q}&retmax={max_results}&retmode=json"
    try:
        with urllib.request.urlopen(search_url, timeout=5) as r:
            ids = json.loads(r.read())["esearchresult"]["idlist"]
    except Exception:
        return []

    if not ids:
        return []

    # Step 2 — fetch abstracts
    fetch_url = f"{base}efetch.fcgi?db=pubmed&id={','.join(ids)}&rettype=abstract&retmode=xml"
    try:
        with urllib.request.urlopen(fetch_url, timeout=8) as r:
            root = ET.fromstring(r.read())
    except Exception:
        return []

    snippets = []
    for article in root.findall(".//PubmedArticle"):
        title_el    = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        pmid_el     = article.find(".//PMID")

        title    = title_el.text    if title_el    is not None else "Untitled"
        abstract = abstract_el.text if abstract_el is not None else ""
        pmid     = pmid_el.text     if pmid_el     is not None else ""

        if abstract:
            snippet = f"[PubMed PMID:{pmid}] {title}\n{abstract[:500]}"
            snippets.append(snippet)

    return snippets


def literature_agent(state: AgentState) -> AgentState:
    question = state["question"]
    category = state.get("medical_category", "")
    intent   = state.get("intent", "general")

    if intent == "emergency":
        return {**state, "literature": "", "citations": []}

    logger.info("[LiteratureAgent] Querying PubMed for: %s", question[:60])

    query   = f"{category} {question}"[:200]
    snippets = _pubmed_search(query)

    literature = "\n\n".join(snippets)
    citations  = [
        s.split("]")[0].replace("[", "").strip() for s in snippets
    ]

    return {**state, "literature": literature, "citations": citations}

# AGENT 5 — SynthesisAgent
# Merges RAG context + PubMed literature → final answer via Groq LLM

_SYNTHESIS_SYSTEM = """You are MediChat, a compassionate AI clinical decision support assistant.

Your role:
- Provide clear, accurate, evidence-based medical information.
- Synthesise information from both a medical Q&A corpus AND peer-reviewed literature.
- Always recommend consulting a qualified healthcare professional for diagnosis or treatment.
- Never fabricate drug names, dosages, or test results.
- Cite PubMed sources when available.
- If context is insufficient, say so clearly.

Format: Short paragraphs. Bullet points only for lists. Under 250 words unless depth is needed.
If a symptom sounds serious, recommend seeing a doctor or calling 911."""


def synthesis_agent(state: AgentState) -> AgentState:
    question   = state["question"]
    rag_ctx    = state.get("rag_context", "")
    literature = state.get("literature", "")
    history    = state.get("history", [])
    category   = state.get("medical_category", "general")
    citations  = state.get("citations", [])

    logger.info("[SynthesisAgent] Generating final answer.")

    # Build context block
    context_parts = []
    if rag_ctx:
        context_parts.append(f"=== Medical Q&A Corpus ===\n{rag_ctx}")
    if literature:
        context_parts.append(f"=== PubMed Evidence ===\n{literature}")

    context_block = "\n\n".join(context_parts) if context_parts else "No retrieved context available."

    user_prompt = (
        f"Medical category: {category}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"Patient question: {question}\n\n"
        "Please synthesise a helpful, evidence-based answer using the context above."
    )

    messages = [SystemMessage(content=_SYNTHESIS_SYSTEM)]
    for turn in history[-(6):]:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=user_prompt))

    try:
        resp   = _llm.invoke(messages)
        answer = resp.content.strip()
        answer = re.sub(r"^helpful answer:\s*", "", answer, flags=re.I)
    except Exception as e:
        answer = f"I encountered an error generating a response: {e}"

   # Convert all PMID formats to clickable [source] links
    # Convert inline PMID references to clickable [source] links
    answer = re.sub(
        r'\[?PubMed PMID:?\s*(\d+)\]?',
        lambda m: f'([source](https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}))',
        answer
    )
    answer = re.sub(
        r'\(?PMID:?\s*(\d+)\)?',
        lambda m: f'([source](https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}))',
        answer
    )

    # Remove duplicate Sources section at bottom — inline links are enough
    return {**state, "answer": answer, "citations": citations}

def _route_after_triage(state: AgentState) -> str:
    """Router: emergency → safety_agent, else → rag_agent."""
    if state.get("safety_flag"):
        return "safety_agent"
    return "rag_agent"

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("triage_agent",     triage_agent)
    g.add_node("safety_agent",     safety_agent)
    g.add_node("rag_agent",        rag_agent)
    g.add_node("literature_agent", literature_agent)
    g.add_node("synthesis_agent",  synthesis_agent)
    g.set_entry_point("triage_agent")
    g.add_conditional_edges(
        "triage_agent",
        _route_after_triage,
        {
            "safety_agent": "safety_agent",
            "rag_agent":    "rag_agent",
        },
    )

    g.add_edge("safety_agent",     END)
    g.add_edge("rag_agent",        "literature_agent")
    g.add_edge("literature_agent", "synthesis_agent")
    g.add_edge("synthesis_agent",  END)

    return g.compile()

_graph = build_graph()


# Public API
def ask_medical_question(
    question: str,
    history:  list[dict] | None = None,
) -> str:
    """
    Drop-in replacement for the original rag.py ask_medical_question().
    Routes through the full 5-agent pipeline.
    """
    question = question.strip()
    if not question:
        return "Please enter a question so I can help you."

    initial_state: AgentState = {
        "question":         question,
        "history":          history or [],
        "medical_category": "",
        "intent":           "",
        "triage_note":      "",
        "rag_context":      "",
        "literature":       "",
        "safety_flag":      False,
        "answer":           "",
        "citations":        [],
    }

    final_state = _graph.invoke(initial_state)
    return final_state.get("answer", "I was unable to generate a response.")