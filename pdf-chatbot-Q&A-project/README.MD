# PDF RAG Chatbot

A simple RAG (Retrieval-Augmented Generation) application to chat with PDF documents using a local LLM.

## 🛠 Tech Stack
- **Backend:** FastAPI, LangChain, HuggingFace (Qwen2.5-0.5B)
- **Frontend:** Streamlit
- **Vector Search:** In-memory DocArray

## 🚀 Key Features
- **Strict RAG:** Answers are grounded only in the provided PDF.
- **Source Citation:** Identifies page numbers for answers.
- **Local:** Runs entirely on your machine (no API costs).

## ⚡ Quick Start

1. **Install Requirements:**
```bash
   pip install fastapi uvicorn streamlit langchain langchain-community transformers torch sentence-transformers docarray
   
