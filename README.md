# Healthcare Chatbot — Multi-Agent AI Health Assistant

A production-grade, multi-agent RAG system for medical Q&A built with **LangGraph**, **Groq (LLaMA-3.1-8B)**, **FAISS**, and **live PubMed evidence retrieval**. Evaluated with RAGAS over 50 curated medical Q&A pairs. Vectorstore persisted on **AWS S3**.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green)](https://flask.palletsprojects.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-orange)](https://langchain-ai.github.io/langgraph)
[![AWS S3](https://img.shields.io/badge/AWS-S3-yellow)](https://aws.amazon.com/s3)

## Demo

> Ask any health question — the AI retrieves from 500K+ medical records + live PubMed abstracts and returns a cited answer in under 5 seconds.

![Chat Interface](demo/2.png)
