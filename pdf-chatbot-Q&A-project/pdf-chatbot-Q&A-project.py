from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI()

# ---------- Load PDF and build vector DB on startup ----------
pdf_file_path = "Elsevior_paper.pdf"

loader = PyPDFLoader(pdf_file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
# --------- Load a lightweight instruction-tuned LLM (Qwen2.5-0.5B) 
# for answer generation in the RAG pipeline----------------------------

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False
)


# ---------- Request body ----------
class QuestionIn(BaseModel):
    question: str

# ---------- API endpoint ----------
@app.post("/ask")
def ask_question(data: QuestionIn):
    query = data.question

    related_docs = vector_db.similarity_search(query, k=3)

    if related_docs:
        context = "\n\n".join([doc.page_content for doc in related_docs])
        sources = [doc.metadata.get("page", "unknown") for doc in related_docs]

        prompt = f"""
You are a question answering system.

Answer ONLY using the information provided in the context.
Do NOT add explanations or external knowledge.

If the answer is not in the context, say:
"The answer is not available in the document."

Context:
{context}

Question:
{query}

Answer:
"""


        result = llm(prompt, return_full_text=False)[0]["generated_text"]
        answer = result.strip()

    else:
        answer = "No answer found."
        sources = []

    return {
        "question": query,
        "answer": answer,
        "page": sources,
        "chunks": [doc.page_content for doc in related_docs],
    }
