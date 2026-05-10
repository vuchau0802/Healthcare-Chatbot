from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading vector database...")

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Medical chatbot ready!")

def ask_medical_question(question):

    try:

        # Retrieve similar medical examples
        docs = db.similarity_search(question, k=2)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are an AI healthcare assistant.

Use the medical examples below to answer naturally.

IMPORTANT:
- Keep answers short and simple.
- Do not copy the dataset exactly.
- Sound like a helpful doctor.
- Recommend seeing a doctor for severe symptoms.

Medical Examples:
{context}

User Question:
{question}

Helpful Answer:
"""

        response = ollama.chat(
            model="phi3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"Error: {str(e)}"