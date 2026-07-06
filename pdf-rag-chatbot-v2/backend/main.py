from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
import torch

app = FastAPI()

# ---------- Load PDF and build vector DB on startup ----------
pdf_file_path = "data/Elsevior_paper.pdf"
loader = PyPDFLoader(pdf_file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------vector store-------------------
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
vectorstore.save_local("faiss_index")

## load vector store
loaded_vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

# ---------- Request body ----------
class QuestionIn(BaseModel):
    question: str


# ------------------llm--------------

llm = OllamaLLM(
    model="llama3.2",
    base_url="http://ollama:11434"
)

# ---------- API endpoint ----------


@app.post("/ask")
def ask_question(data: QuestionIn):
    query = data.question

    related_docs = loaded_vectorstore.similarity_search(query, k=3)

    if related_docs:
        context = "\n\n".join([doc.page_content for doc in related_docs])
        sources = [doc.metadata.get("page", "unknown") for doc in related_docs]

        ### Conversational RAg Chain

        simple_prompt = PromptTemplate.from_template("""
        Answer the question based only on the following context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)

        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        from typing import List

        # Format documents for the prompt
        def format_docs(docs: List[Document]) -> str:
            """Format documents for insertion into prompt"""
            formatted = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown")
                formatted.append(
                    f"Document {i+1} (Source: {source}):\n{doc.page_content}"
                )
            return "\n\n".join(formatted)

        simple_rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | simple_prompt
            | llm
            | StrOutputParser()
        )
        result = simple_rag_chain.invoke(query)

        answer = result

    else:
        answer = "No answer found."
        sources = []

    return {
        "question": query,
        "answer": answer,
        "sources": sources,
        "chunks": [doc.page_content for doc in related_docs],
    }
