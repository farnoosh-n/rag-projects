from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM

from langchain_core.documents import Document
from langchain_core.prompts import  PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from typing import List

app = FastAPI()

# ---------- Load PDF------------

pdf_file_path = "data/Elsevior_paper.pdf"
loader = PyPDFLoader(pdf_file_path)
documents = loader.load()

# ---------- Split Documents ----------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


chunks = text_splitter.split_documents(documents)

# ---------- Embeddings ----------

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------vector store-------------------
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
vectorstore.save_local("faiss_index")

## load vector store

loaded_vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
# ---------- Retriever ----------

retriever = loaded_vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)
# ---------- Request body ----------

class QuestionIn(BaseModel):
    question: str

# ------------------llm--------------

llm = OllamaLLM(
    model="llama3.2",
    base_url="http://ollama:11434"
)

#----------Prompt----------

simple_prompt = PromptTemplate.from_template(
"""
Answer the question based only on the following context.

Context:
{context}

Question:
{question}

Answer:
"""
)

#---------- Format Document ----------

def format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(
            f"Document {i+1} (Source: {source}):\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

# ---------- API ------------


@app.post("/ask")
def ask_question(data: QuestionIn):
    query = data.question

    related_docs = retriever.invoke(query)

    if related_docs :

        sources=list({ doc.metadata.get("page", "unknown") for doc in related_docs})

        simple_rag_chain = (
            {"context":  lambda x: format_docs(related_docs), "question": RunnablePassthrough()}
            | simple_prompt
            | llm
            | StrOutputParser()
        )
        result = simple_rag_chain.invoke(query)

        answer = result

    else:
        answer = "No answer found."
        sources = []
        related_docs = []

    return {
        "question": query,
        "answer": answer,
        "sources": sources,
        "chunks": [doc.page_content for doc in related_docs]
    }
