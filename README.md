# Healthcare Chatbot — Multi-Agent AI Health Assistant

A production-grade, multi-agent RAG system for medical Q&A built with **LangGraph**, **Groq (LLaMA-3.1-8B)**, **FAISS**, and **live PubMed evidence retrieval**. Vectorstore persisted on **AWS S3**.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green)](https://flask.palletsprojects.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-orange)](https://langchain-ai.github.io/langgraph)
[![AWS S3](https://img.shields.io/badge/AWS-S3-yellow)](https://aws.amazon.com/s3)

## Step-by-Step Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/vuchau0802/Healthcare-Chatbot.git
cd Healthcare-Chatbot
```

### Step 2 — Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set up environment variables

Create a `.env` file in the project root:

```bash
New-Item .env -ItemType File
notepad .env
```

Paste and fill in your values:

```env
GROQ_API_KEY=gsk_your_groq_key
FLASK_SECRET_KEY=your_random_hex_string
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=medichat-vectorstore
AWS_REGION=us-region
```

### Step 5 — Build the FAISS vectorstore

```bash
python ingest.py
```

### Step 6 — Train the triage classifier

```bash
python agents/train_triage.py
```

### Step 7 — Upload vectorstore to AWS S3

```bash
python cloud/s3_store.py upload
```

### Step 8 — Run the app

```bash
python app.py
```

## Demo

> Ask any health question — the AI retrieves from 500K+ medical records + live PubMed abstracts and returns a cited answer in under 5 seconds.
![Chat Interface](demo/1.png)

![Chat Interface](demo/2.png)
