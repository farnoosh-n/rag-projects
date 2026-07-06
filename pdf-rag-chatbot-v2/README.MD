# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to ask natural language questions about a PDF document using a local Large Language Model (LLM).

The project demonstrates the complete RAG workflow, including document loading, text chunking, embedding generation, semantic retrieval, and answer generation through a FastAPI backend and a Streamlit web interface.

---

## Features

- Load and process PDF documents
- Split documents into overlapping text chunks
- Generate embeddings using Sentence Transformers
- Store document embeddings in a FAISS vector database
- Retrieve relevant document chunks using semantic similarity search
- Generate answers with a local LLM using Ollama (Llama 3.2)
- REST API built with FastAPI
- Interactive web interface built with Streamlit
- Dockerized multi-container application

---

## Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM Framework | LangChain |
| Local LLM | Ollama (Llama 3.2) |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| PDF Loader | PyPDF |
| Containerization | Docker & Docker Compose |

---

## Project Architecture

```
                User
                  │
                  ▼
        Streamlit Frontend
                  │
        HTTP POST /ask
                  │
                  ▼
          FastAPI Backend
                  │
        Load Relevant Chunks
                  │
             FAISS Index
                  │
        Sentence Embeddings
                  │
                  ▼
       Ollama (Llama 3.2)
                  │
                  ▼
          Generated Answer
```

---

## Project Structure

```
pdf-rag-chatbot/
│
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── faiss_index/
│
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── data/
│   └── Elsevior_paper.pdf
│
├── docker-compose.yml
└── README.md
```

---

## How It Works

1. A PDF document is loaded using **PyPDFLoader**.
2. The document is split into overlapping chunks using **RecursiveCharacterTextSplitter**.
3. Each chunk is converted into a dense vector using the **Sentence Transformers** embedding model.
4. The vectors are stored in a **FAISS** vector database.
5. When a user submits a question, the system retrieves the most relevant chunks using semantic similarity search.
6. The retrieved context is combined with the user's question.
7. The prompt is sent to **Ollama (Llama 3.2)** to generate the final answer.
8. The answer, retrieved pages, and relevant context are returned to the user.

---

## Running the Project

### Clone the repository

```bash
git clone https://github.com/your-username/pdf-rag-chatbot.git

cd pdf-rag-chatbot
```

### Build and start the containers

```bash
docker compose up --build
```

### Download the language model

```bash
docker exec -it rag-ollama ollama pull llama3.2
```

---

## Access the Application

### Streamlit Frontend

```
http://localhost:8600
```

### FastAPI Documentation

```
http://localhost:9000/docs
```

### Ollama API

```
http://localhost:11434
```

---

## API

### POST `/ask`

Request

```json
{
    "question": "What is the paper about?"
}
```

Response

```json
{
    "question": "...",
    "answer": "...",
    "sources": [2, 3],
    "chunks": [
        "...",
        "..."
    ]
}
```

---

## RAG Pipeline

- PDF Loading
- Text Chunking
- Embedding Generation
- FAISS Vector Search
- Context Retrieval
- Prompt Construction
- Answer Generation using Ollama

---

## Future Improvements

- Support multiple PDF documents
- Add conversation memory
- Improve retrieval using reranking
- Persist vector database between runs
- Deploy the application to a cloud platform

---

## Skills Demonstrated

- Python
- FastAPI
- Streamlit
- LangChain
- Retrieval-Augmented Generation (RAG)
- FAISS
- Sentence Transformers
- Ollama
- Docker
- REST API Development with FastAPI
- Semantic Search using Vector Embeddings