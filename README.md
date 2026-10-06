# University Knowledge RAG Assistant

## Project Overview

The University Knowledge RAG Assistant is a Python-based Retrieval-Augmented Generation (RAG) project developed to explore the fundamental components of a retrieval-based AI system.

The system is designed to answer questions using a controlled collection of university-related documents rather than relying only on the knowledge contained within a language model.

The project is being developed as Technical Portfolio Project 1 – RAG Fundamentals.

## Project Aim

The main aim of this project is to understand how a basic RAG pipeline works from document ingestion through to retrieval and generated responses.

The project focuses on understanding the individual components rather than relying on a high-level framework to hide the underlying processes.

## Core RAG Pipeline

The planned pipeline is:

Documents
↓
Document Loading
↓
Text Cleaning
↓
Text Chunking
↓
Embedding Generation
↓
Vector Database
↓
Semantic Retrieval
↓
Retrieved Context
↓
LLM Generation
↓
Grounded Answer
↓
Source Attribution

## Current Development Status

The project currently has a working local Streamlit interface and a controlled document collection available through the application.

The current interface identifies documents available in the RAG corpus and provides an interface for asking questions about the university documents.

Further development will focus on validating the complete retrieval and generation pipeline, evaluation, automated testing and final documentation.

## Technologies

- Python 3.12
- VS Code
- Sentence Transformers
- ChromaDB
- Streamlit
- PyPDF
- python-dotenv
- Pytest
- Git and GitHub

## Project Structure

```text
university-knowledge-rag/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── generator.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── evaluation/
├── tests/
├── screenshots/
├── docs/
│
├── .gitignore
├── .env.example
├── requirements.txt
└── README.md