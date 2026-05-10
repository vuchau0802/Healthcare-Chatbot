import pandas as pd
import os
import hashlib
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CSV_PATH      = "data/cleaned_medical_25mb.csv"
VECTORSTORE   = "vectorstore"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
MAX_ROWS      = 5000   # increase from 500 to better recall coverage
BATCH_SIZE    = 100
CHUNK_SIZE    = 800    # characters per chunk
CHUNK_OVERLAP = 150    # overlap keeps context across splits

def load_csv(path: str, max_rows: int) -> pd.DataFrame:
    print(f"Loading CSV from {path} …")
    df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    df = df.dropna(subset=["Patient", "Doctor"])          # drop empty exchanges
    df = df.head(max_rows)
    print(f"  Loaded {len(df)} usable rows")
    return df

def build_documents(df: pd.DataFrame) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []

    for idx, row in df.iterrows():
        description = str(row.get("Description", "")).strip()
        patient_q   = str(row.get("Patient", "")).strip()
        doctor_a    = str(row.get("Doctor", "")).strip()

        if not patient_q or not doctor_a:
            continue

        # Unique stable id for deduplication
        uid = hashlib.md5(f"{patient_q}{doctor_a}".encode()).hexdigest()[:10]

        # Full passage — used for context retrieval
        full_text = (
            f"CONDITION/TOPIC: {description}\n\n"
            f"PATIENT QUESTION: {patient_q}\n\n"
            f"DOCTOR ANSWER: {doctor_a}"
        )

        metadata = {
            "source_row": int(idx),
            "uid": uid,
            "description": description[:200],
            "has_description": bool(description),
        }

        # Split only if the text is long enough to warrant it
        if len(full_text) > CHUNK_SIZE:
            chunks = splitter.create_documents([full_text], metadatas=[metadata])
            docs.extend(chunks)
        else:
            docs.append(Document(page_content=full_text, metadata=metadata))

    print(f"  Built {len(docs)} document chunks from {len(df)} rows")
    return docs


def build_vectorstore(docs: list[Document], embed_model: str, out_dir: str) -> None:
    print(f"Loading embedding model: {embed_model} …")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},   # cosine similarity
    )

    print(f"Embedding {len(docs)} chunks in batches of {BATCH_SIZE} …")
    db = None

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total     = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total} …")

        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            temp = FAISS.from_documents(batch, embeddings)
            db.merge_from(temp)

    os.makedirs(out_dir, exist_ok=True)
    db.save_local(out_dir)
    print(f"Vectorstore saved to '{out_dir}/' ✓")


if __name__ == "__main__":
    df   = load_csv(CSV_PATH, MAX_ROWS)
    docs = build_documents(df)
    build_vectorstore(docs, EMBED_MODEL, VECTORSTORE)
    print("Done — medical vector database ready!")
